from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from survng.app.auth_routes import AuthRouteDependencies, create_auth_router
from survng.app.config import AppConfig, WebAuthConfig, WebUserConfig
from survng.app.security import (
    authenticate_password,
    authenticate_session,
    decode_session,
    encode_session,
    hash_password,
    is_public_api_path,
    required_api_scope,
    scopes_for_web_role,
    verify_password,
)


def make_user(username: str = "alex", role: str = "admin", password: str = "correct-horse") -> WebUserConfig:
    return WebUserConfig(
        id=username,
        username=username,
        display_name=username.title(),
        role=role,
        password_hash=hash_password(password),
    )


class PasswordAndSessionTest(unittest.TestCase):
    def test_password_round_trip_and_rejects_wrong_secret(self) -> None:
        digest = hash_password("correct-horse")
        self.assertTrue(digest.startswith("scrypt$"))
        self.assertTrue(verify_password("correct-horse", digest))
        self.assertFalse(verify_password("wrong-horse", digest))

    def test_session_token_expires_and_rejects_tampering(self) -> None:
        token = encode_session("alex", "a" * 64, now=1_700_000_000)
        self.assertEqual(decode_session(token, "a" * 64, now=1_700_000_100), "alex")
        self.assertIsNone(decode_session(token, "b" * 64, now=1_700_000_100))
        self.assertIsNone(decode_session(token[:-1] + ("0" if token[-1] != "0" else "1"), "a" * 64, now=1_700_000_100))
        self.assertIsNone(decode_session(token, "a" * 64, now=1_700_000_000 + 15 * 24 * 60 * 60))

    def test_viewer_scope_is_read_only(self) -> None:
        self.assertEqual(scopes_for_web_role("viewer"), frozenset({"read"}))
        self.assertIn("admin", scopes_for_web_role("admin"))

    def test_auth_and_tls_writes_require_admin_scope(self) -> None:
        self.assertEqual(required_api_scope("GET", "/api/auth/users"), "admin")
        self.assertEqual(required_api_scope("GET", "/api/tls"), "admin")
        self.assertTrue(is_public_api_path("GET", "/api/auth/session"))
        self.assertTrue(is_public_api_path("POST", "/api/auth/login"))


class WebAuthConfigTest(unittest.TestCase):
    def test_sign_in_requires_an_administrator(self) -> None:
        user = make_user("pat", role="viewer")
        with self.assertRaises(ValidationError):
            WebAuthConfig(enabled=True, session_key="a" * 64, users=[user])

    def test_usernames_are_unique(self) -> None:
        with self.assertRaises(ValidationError):
            WebAuthConfig(users=[make_user("alex"), make_user("Alex")])


class AuthRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.admin = make_user("alex", role="admin")
        self.viewer = make_user("pat", role="viewer", password="viewer-pass")
        self.config = AppConfig(
            storage_dir="/tmp/survng-auth-test",
            web_auth=WebAuthConfig(
                enabled=True,
                session_key="a" * 64,
                users=[self.admin, self.viewer],
            ),
        )
        self.apply = Mock(side_effect=self._apply)
        self.app = FastAPI()
        self.app.include_router(create_auth_router(AuthRouteDependencies(
            get_config=lambda: self.config,
            apply_config=self.apply,
            lock=threading.RLock(),
        )))
        self.client = TestClient(self.app)

    def _apply(self, next_config: AppConfig, assign_ids: bool = False) -> tuple[AppConfig, dict[str, object]]:
        self.config = next_config
        return next_config, {"apply_mode": "hot", "subsystems_restarted": []}

    def test_login_sets_http_only_cookie(self) -> None:
        response = self.client.post("/api/auth/login", json={"username": "alex", "password": "correct-horse"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["role"], "admin")
        self.assertIn("survng_session", response.cookies)
        cookie = response.headers.get("set-cookie") or ""
        self.assertIn("HttpOnly", cookie)
        self.assertIn("samesite=lax", cookie.lower())

    def test_login_rejects_bad_password(self) -> None:
        response = self.client.post("/api/auth/login", json={"username": "alex", "password": "nope-nope"})
        self.assertEqual(response.status_code, 401)

    def test_bootstrap_creates_first_admin_when_empty(self) -> None:
        self.config.web_auth = WebAuthConfig()
        response = self.client.post("/api/auth/bootstrap", json={
            "username": "rootadmin",
            "password": "bootstrap-secret",
            "display_name": "Root",
        })
        self.assertEqual(response.status_code, 201)
        self.assertTrue(self.config.web_auth.enabled)
        self.assertEqual(self.config.web_auth.users[0].role, "admin")
        self.assertEqual(len(self.config.web_auth.session_key), 64)

    def test_cannot_delete_last_admin(self) -> None:
        self.config.web_auth.enabled = False
        self.config.web_auth.users = [self.admin]
        response = self.client.delete("/api/auth/users/alex")
        self.assertEqual(response.status_code, 409)


class SessionPrincipalTest(unittest.TestCase):
    def test_cookie_authenticates_configured_user(self) -> None:
        user = make_user()
        auth = WebAuthConfig(enabled=True, session_key="a" * 64, users=[user])
        token = encode_session(user.id, auth.session_key)
        principal = authenticate_session(f"survng_session={token}", auth)
        self.assertIsNotNone(principal)
        self.assertEqual(principal.username, "alex")
        self.assertTrue(principal.permits("admin"))

    def test_password_lookup_is_case_insensitive(self) -> None:
        user = make_user("Alex")
        auth = WebAuthConfig(users=[user])
        matched = authenticate_password("alex", "correct-horse", auth)
        self.assertEqual(matched.id, "Alex")

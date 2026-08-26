from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from survng.app.auth_routes import (
    AuthRouteDependencies,
    create_auth_router,
    load_or_create_bootstrap_token,
)
from survng.app.config import AppConfig, WebAuthConfig, WebUserConfig
from survng.app.config_routes import restore_config_secrets
from survng.app.security import (
    SESSION_COOKIE_NAME,
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

    def test_verify_password_rejects_expensive_parameters(self) -> None:
        huge = "scrypt$1048576$8$1$" + ("ab" * 16) + "$" + ("cd" * 32)
        self.assertFalse(verify_password("correct-horse", huge))

    def test_session_token_expires_and_rejects_tampering(self) -> None:
        token = encode_session("alex", "a" * 64, now=1_700_000_000)
        self.assertEqual(decode_session(token, "a" * 64, now=1_700_000_100), ("alex", 0))
        self.assertIsNone(decode_session(token, "b" * 64, now=1_700_000_100))
        self.assertIsNone(decode_session(token[:-1] + ("0" if token[-1] != "0" else "1"), "a" * 64, now=1_700_000_100))
        self.assertIsNone(decode_session(token, "a" * 64, now=1_700_000_000 + 15 * 24 * 60 * 60))

    def test_session_token_uses_configured_lifetime(self) -> None:
        token = encode_session("alex", "a" * 64, now=1_700_000_000, ttl_seconds=60)
        self.assertEqual(decode_session(token, "a" * 64, now=1_700_000_030), ("alex", 0))
        self.assertIsNone(decode_session(token, "a" * 64, now=1_700_000_061))

    def test_password_change_invalidates_older_session_tokens(self) -> None:
        token = encode_session("alex", "a" * 64, now=1_700_000_000, session_epoch=0)
        self.assertEqual(decode_session(token, "a" * 64, now=1_700_000_100), ("alex", 0))
        later = encode_session("alex", "a" * 64, now=1_700_000_000, session_epoch=2)
        self.assertEqual(decode_session(later, "a" * 64, now=1_700_000_100), ("alex", 2))

    def test_viewer_scope_is_read_only(self) -> None:
        self.assertEqual(scopes_for_web_role("viewer"), frozenset({"read"}))
        self.assertIn("admin", scopes_for_web_role("admin"))

    def test_auth_and_tls_writes_require_admin_scope(self) -> None:
        self.assertEqual(required_api_scope("GET", "/api/auth/users"), "admin")
        self.assertEqual(required_api_scope("GET", "/api/tls"), "admin")
        self.assertTrue(is_public_api_path("GET", "/api/auth/session"))
        self.assertTrue(is_public_api_path("POST", "/api/auth/login"))


class WebAuthConfigTest(unittest.TestCase):
    def test_sign_in_requires_an_administrator_when_users_exist(self) -> None:
        user = make_user("pat", role="viewer")
        with self.assertRaises(ValidationError):
            WebAuthConfig(enabled=True, session_key="a" * 64, users=[user])

    def test_sign_in_can_be_enabled_before_the_first_user(self) -> None:
        auth = WebAuthConfig(enabled=True, users=[])
        self.assertTrue(auth.enabled)
        self.assertEqual(auth.users, [])

    def test_config_save_does_not_reset_session_epoch(self) -> None:
        user = make_user()
        user.session_epoch = 4
        current = AppConfig(
            web_auth=WebAuthConfig(enabled=True, session_key="a" * 64, users=[user]),
        )
        payload = current.model_dump(mode="json")
        payload["web_auth"]["users"][0]["session_epoch"] = 0
        payload["web_auth"]["users"][0]["password_hash"] = "__SURVNG_SECRET_SET__"
        restored = restore_config_secrets(AppConfig.model_validate(payload), current)
        self.assertEqual(restored.web_auth.users[0].session_epoch, 4)

    def test_enabling_sign_in_through_config_save_mints_a_session_key(self) -> None:
        current = AppConfig()
        payload = current.model_dump(mode="json")
        payload["web_auth"]["enabled"] = True
        restored = restore_config_secrets(AppConfig.model_validate(payload), current)
        self.assertTrue(restored.web_auth.enabled)
        self.assertEqual(len(restored.web_auth.session_key), 64)

    def test_usernames_are_unique(self) -> None:
        with self.assertRaises(ValidationError):
            WebAuthConfig(users=[make_user("alex"), make_user("Alex")])

    def test_session_days_are_bounded(self) -> None:
        self.assertEqual(WebAuthConfig().session_days, 14)
        with self.assertRaises(ValidationError):
            WebAuthConfig(session_days=0)
        with self.assertRaises(ValidationError):
            WebAuthConfig(session_days=366)


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

        @self.app.middleware("http")
        async def attach_session_principal(request, call_next):
            auth = self.config.web_auth
            principal = authenticate_session(request.headers.get("cookie") or "", auth)
            if principal is None and not auth.enabled and auth.session_key:
                principal = authenticate_session(
                    request.headers.get("cookie") or "",
                    auth.model_copy(update={"enabled": True}),
                )
            if principal is not None:
                request.scope["survng_principal"] = principal
            return await call_next(request)

        self.app.include_router(create_auth_router(AuthRouteDependencies(
            get_config=lambda: self.config,
            apply_config=self.apply,
            lock=threading.RLock(),
        )))
        self.client = TestClient(self.app)

    def _apply(self, next_config: AppConfig, assign_ids: bool = False) -> tuple[AppConfig, dict[str, object]]:
        self.config = next_config
        return next_config, {"apply_mode": "hot", "subsystems_restarted": []}

    def _sign_in(self, user: WebUserConfig | None = None) -> None:
        actor = user or self.admin
        token = encode_session(actor.id, self.config.web_auth.session_key or "a" * 64)
        self.client.cookies.set(SESSION_COOKIE_NAME, token)

    def test_login_sets_http_only_cookie(self) -> None:
        response = self.client.post("/api/auth/login", json={"username": "alex", "password": "correct-horse"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["role"], "admin")
        self.assertIn("survng_session", response.cookies)
        cookie = response.headers.get("set-cookie") or ""
        self.assertIn("HttpOnly", cookie)
        self.assertIn("samesite=lax", cookie.lower())
        self.assertIn("max-age=1209600", cookie.lower())

    def test_login_cookie_uses_configured_session_days(self) -> None:
        self.config.web_auth.session_days = 2
        response = self.client.post("/api/auth/login", json={"username": "alex", "password": "correct-horse"})
        self.assertEqual(response.status_code, 200)
        cookie = (response.headers.get("set-cookie") or "").lower()
        self.assertIn("max-age=172800", cookie)

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

    def test_session_asks_for_bootstrap_only_when_sign_in_is_on_without_users(self) -> None:
        self.config.web_auth = WebAuthConfig()
        response = self.client.get("/api/auth/session")
        self.assertEqual(response.json(), {
            "enabled": False,
            "bootstrap_required": False,
            "bootstrap_token_required": False,
            "user": None,
        })
        self.config.web_auth.enabled = True
        response = self.client.get("/api/auth/session")
        self.assertEqual(response.json()["bootstrap_required"], True)
        self.assertTrue(response.json()["enabled"])
        self.assertFalse(response.json()["bootstrap_token_required"])
        with patch("survng.app.auth_routes.ip_is_local", return_value=False):
            remote = self.client.get("/api/auth/session")
        self.assertTrue(remote.json()["bootstrap_token_required"])

    def test_settings_can_enable_sign_in_before_the_first_user(self) -> None:
        self.config.web_auth = WebAuthConfig()
        response = self.client.put("/api/auth/settings", json={"enabled": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.config.web_auth.enabled)
        self.assertEqual(self.config.web_auth.users, [])
        self.assertEqual(len(self.config.web_auth.session_key), 64)

    def test_can_delete_last_admin_after_sign_in_is_disabled(self) -> None:
        self.config.web_auth.enabled = False
        self.config.web_auth.users = [self.admin]
        response = self.client.delete("/api/auth/users/alex")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.config.web_auth.users, [])
        self.assertFalse(self.config.web_auth.enabled)

    def test_can_demote_last_admin_after_sign_in_is_disabled(self) -> None:
        self.config.web_auth.enabled = False
        self.config.web_auth.users = [self.admin]
        response = self.client.patch("/api/auth/users/alex", json={"role": "viewer"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.config.web_auth.users[0].role, "viewer")
        self.assertFalse(self.config.web_auth.enabled)

    def test_cannot_demote_last_admin_while_sign_in_is_required(self) -> None:
        self._sign_in()
        self.config.web_auth.users = [self.admin]
        response = self.client.patch("/api/auth/users/alex", json={"role": "viewer"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.config.web_auth.users[0].role, "admin")

    def test_deleting_last_admin_turns_sign_in_off(self) -> None:
        self._sign_in()
        self.config.web_auth.users = [self.admin]
        response = self.client.delete("/api/auth/users/alex")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.config.web_auth.enabled)
        self.assertEqual(self.config.web_auth.users, [])
        cookie = response.headers.get("set-cookie") or ""
        self.assertIn(SESSION_COOKIE_NAME, cookie)

    def test_cannot_delete_own_account_while_another_admin_exists(self) -> None:
        other = make_user("root", role="admin", password="other-admin")
        self.config.web_auth.users = [self.admin, other, self.viewer]
        self._sign_in(self.admin)
        response = self.client.delete("/api/auth/users/alex")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(self.config.web_auth.users), 3)

    def test_password_can_be_changed_while_sign_in_is_on(self) -> None:
        self._sign_in()
        response = self.client.put("/api/auth/users/pat/password", json={"password": "new-viewer-pass"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(authenticate_password("pat", "new-viewer-pass", self.config.web_auth))
        self.assertIsNone(authenticate_password("pat", "viewer-pass", self.config.web_auth))
        self.assertEqual(self.config.web_auth.users[1].session_epoch, 1)

    def test_password_change_rejects_older_session_cookie(self) -> None:
        self._sign_in(self.viewer)
        self.assertIsNotNone(authenticate_session(
            f"{SESSION_COOKIE_NAME}={encode_session('pat', self.config.web_auth.session_key, session_epoch=0)}",
            self.config.web_auth,
        ))
        response = self.client.put("/api/auth/users/pat/password", json={"password": "rotated-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(authenticate_session(
            f"{SESSION_COOKIE_NAME}={encode_session('pat', self.config.web_auth.session_key, session_epoch=0)}",
            self.config.web_auth,
        ))

    def test_password_can_be_changed_while_sign_in_is_off(self) -> None:
        self.config.web_auth.enabled = False
        response = self.client.put("/api/auth/users/alex/password", json={"password": "brand-new-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(authenticate_password("alex", "brand-new-secret", self.config.web_auth))

    def test_public_bootstrap_requires_the_server_setup_token(self) -> None:
        self.config.web_auth = WebAuthConfig()
        with patch("survng.app.auth_routes.ip_is_local", return_value=False):
            denied = self.client.post("/api/auth/bootstrap", json={
                "username": "rootadmin",
                "password": "bootstrap-secret",
            })
            self.assertEqual(denied.status_code, 403)
            token = load_or_create_bootstrap_token(self.config)
            allowed = self.client.post("/api/auth/bootstrap", json={
                "username": "rootadmin",
                "password": "bootstrap-secret",
                "bootstrap_token": token,
            })
        self.assertEqual(allowed.status_code, 201)
        self.assertEqual(self.config.web_auth.users[0].username, "rootadmin")


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

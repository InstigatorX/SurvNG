from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from shutil import which

from survng.app.config import AppConfig, TlsConfig
from survng.app.tls import (
    generate_self_signed_certificate,
    parse_pem_bundle,
    tls_files_present,
    tls_status,
    uvicorn_tls_kwargs,
    write_tls_material,
)


class TlsMaterialTest(unittest.TestCase):
    def test_parse_pem_bundle_requires_markers(self) -> None:
        with self.assertRaises(ValueError):
            parse_pem_bundle("not a cert", kind="CERTIFICATE")
        pem = "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"
        self.assertIn("BEGIN CERTIFICATE", parse_pem_bundle(pem, kind="CERTIFICATE"))

    def test_uvicorn_kwargs_stay_plain_http_until_tls_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as storage:
            config = AppConfig(storage_dir=storage)
            kwargs = uvicorn_tls_kwargs(config, 8088)
            self.assertEqual(kwargs, {"port": 8088})

    @unittest.skipUnless(which("openssl"), "openssl is required to generate certificates")
    def test_self_signed_certificate_is_stored_and_inspectable(self) -> None:
        with tempfile.TemporaryDirectory() as storage:
            config = AppConfig(storage_dir=storage, tls=TlsConfig(hostname="survng.test"))
            status = generate_self_signed_certificate(config, "survng.test")
            self.assertTrue(tls_files_present(config))
            self.assertTrue(status["certificate_present"])
            self.assertIn("survng.test", " ".join(status.get("subject_alt_names") or []) + status.get("subject", ""))
            config.tls.enabled = True
            kwargs = uvicorn_tls_kwargs(config, 8088)
            self.assertEqual(kwargs["port"], 8088)
            self.assertTrue(Path(kwargs["ssl_certfile"]).is_file())
            inspect = tls_status(config)
            self.assertTrue(inspect["certificate_present"])
            self.assertIn("survng.test", inspect.get("subject", "") + " ".join(inspect.get("subject_alt_names") or []))

    @unittest.skipUnless(which("openssl"), "openssl is required to generate certificates")
    def test_uploaded_key_pair_replaces_existing_material(self) -> None:
        with tempfile.TemporaryDirectory() as storage:
            first = AppConfig(storage_dir=storage)
            generate_self_signed_certificate(first, "one.example")
            cert = (Path(storage) / "tls" / "fullchain.pem").read_text(encoding="utf-8")
            key = (Path(storage) / "tls" / "privkey.pem").read_text(encoding="utf-8")
            second = AppConfig(storage_dir=storage)
            write_tls_material(second, cert, key)
            self.assertTrue(tls_files_present(second))

    @unittest.skipUnless(which("openssl"), "openssl is required to generate certificates")
    def test_certificate_route_stores_pasted_pem(self) -> None:
        import threading
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from survng.app.tls_routes import TlsRouteDependencies, create_tls_router

        with tempfile.TemporaryDirectory() as storage:
            config = AppConfig(storage_dir=storage)
            generate_self_signed_certificate(config, "paste.example")
            cert = (Path(storage) / "tls" / "fullchain.pem").read_text(encoding="utf-8")
            key = (Path(storage) / "tls" / "privkey.pem").read_text(encoding="utf-8")
            (Path(storage) / "tls" / "fullchain.pem").unlink()
            (Path(storage) / "tls" / "privkey.pem").unlink()

            def apply(next_config, assign_ids=False):
                return next_config, {}

            app = FastAPI()
            app.include_router(create_tls_router(TlsRouteDependencies(
                get_config=lambda: config,
                apply_config=apply,
                request_server_restart=lambda: {},
                lock=threading.RLock(),
            )))
            client = TestClient(app)
            response = client.post("/api/tls/certificate", json={
                "certificate_pem": cert,
                "private_key_pem": key,
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["certificate_present"])
            self.assertTrue(tls_files_present(config))

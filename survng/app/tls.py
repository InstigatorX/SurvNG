"""HTTPS certificate files for the SurvNG process."""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import ssl
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import AppConfig

CERT_FILENAME = "fullchain.pem"
KEY_FILENAME = "privkey.pem"
DEFAULT_HOSTNAME = "survng.local"
SELF_SIGNED_DAYS = 825


def tls_directory(config: AppConfig) -> Path:
    return Path(config.storage_dir) / "tls"


def tls_certificate_path(config: AppConfig) -> Path:
    return tls_directory(config) / CERT_FILENAME


def tls_private_key_path(config: AppConfig) -> Path:
    return tls_directory(config) / KEY_FILENAME


def tls_files_present(config: AppConfig) -> bool:
    return tls_certificate_path(config).is_file() and tls_private_key_path(config).is_file()


def validate_tls_material(cert_path: Path, key_path: Path) -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert_path), str(key_path))


def _openssl_bin() -> str:
    path = shutil.which("openssl")
    if not path:
        raise RuntimeError("openssl is required to inspect or generate TLS certificates")
    return path


def _openssl_text(cert_path: Path) -> str:
    result = subprocess.run(
        [_openssl_bin(), "x509", "-in", str(cert_path), "-noout", "-text", "-fingerprint", "-sha256"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def _field(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}\s*=\s*(.+)", text)
    return match.group(1).strip() if match else ""


def inspect_certificate(cert_path: Path) -> dict[str, Any]:
    text = _openssl_text(cert_path)
    san = re.search(r"X509v3 Subject Alternative Name:\s*\n\s*(.+)", text)
    names: list[str] = []
    if san:
        names = [part.strip() for part in san.group(1).split(",") if part.strip()]
    not_before = _field(text, "Not Before")
    not_after = _field(text, "Not After")
    fingerprint = _field(text, "sha256 Fingerprint")
    return {
        "subject": _field(text, "Subject"),
        "issuer": _field(text, "Issuer"),
        "not_before": not_before,
        "not_after": not_after,
        "fingerprint_sha256": fingerprint.replace(":", "").lower() if fingerprint else "",
        "subject_alt_names": names,
        "self_signed": _field(text, "Subject") == _field(text, "Issuer") and bool(_field(text, "Subject")),
    }


def tls_status(config: AppConfig) -> dict[str, Any]:
    cert_path = tls_certificate_path(config)
    key_path = tls_private_key_path(config)
    present = cert_path.is_file() and key_path.is_file()
    details: dict[str, Any] = {}
    error = ""
    if present:
        try:
            validate_tls_material(cert_path, key_path)
            details = inspect_certificate(cert_path)
        except Exception as exc:
            error = str(exc)
    return {
        "enabled": config.tls.enabled,
        "port": config.tls.port,
        "hostname": config.tls.hostname,
        "certificate_present": present,
        "certificate_path": str(cert_path) if present else "",
        "error": error,
        **details,
    }


def _san_arguments(hostname: str) -> list[str]:
    host = hostname.strip() or DEFAULT_HOSTNAME
    try:
        address = ipaddress.ip_address(host)
        return [f"subjectAltName=IP:{address}"]
    except ValueError:
        return [f"subjectAltName=DNS:{host}"]


def write_tls_material(config: AppConfig, certificate_pem: str, private_key_pem: str) -> None:
    directory = tls_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    cert_path = tls_certificate_path(config)
    key_path = tls_private_key_path(config)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as cert_handle:
        temp_cert = Path(cert_handle.name)
        cert_handle.write(certificate_pem)
        if not certificate_pem.endswith("\n"):
            cert_handle.write("\n")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as key_handle:
        temp_key = Path(key_handle.name)
        key_handle.write(private_key_pem)
        if not private_key_pem.endswith("\n"):
            key_handle.write("\n")
    try:
        os.chmod(temp_cert, stat.S_IRUSR | stat.S_IWUSR)
        os.chmod(temp_key, stat.S_IRUSR | stat.S_IWUSR)
        validate_tls_material(temp_cert, temp_key)
        os.replace(temp_cert, cert_path)
        os.replace(temp_key, key_path)
        os.chmod(cert_path, stat.S_IRUSR | stat.S_IWUSR)
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        temp_cert.unlink(missing_ok=True)
        temp_key.unlink(missing_ok=True)
        raise


def generate_self_signed_certificate(config: AppConfig, hostname: str = "") -> dict[str, Any]:
    host = (hostname or config.tls.hostname or DEFAULT_HOSTNAME).strip() or DEFAULT_HOSTNAME
    directory = tls_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="survng-tls-") as scratch:
        scratch_path = Path(scratch)
        cert_path = scratch_path / CERT_FILENAME
        key_path = scratch_path / KEY_FILENAME
        command = [
            _openssl_bin(),
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            str(SELF_SIGNED_DAYS),
            "-nodes",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-subj",
            f"/CN={host}",
            "-addext",
            _san_arguments(host)[0],
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
        write_tls_material(
            config,
            cert_path.read_text(encoding="utf-8"),
            key_path.read_text(encoding="utf-8"),
        )
    return tls_status(config)


def ensure_tls_material(config: AppConfig) -> None:
    if not config.tls.enabled:
        return
    if tls_files_present(config):
        validate_tls_material(tls_certificate_path(config), tls_private_key_path(config))
        return
    generate_self_signed_certificate(config)


def uvicorn_tls_kwargs(config: AppConfig, listen_port: int) -> dict[str, Any]:
    if not config.tls.enabled:
        return {"port": listen_port}
    ensure_tls_material(config)
    port = config.tls.port or listen_port
    return {
        "port": port,
        "ssl_certfile": str(tls_certificate_path(config)),
        "ssl_keyfile": str(tls_private_key_path(config)),
    }


def parse_pem_bundle(raw: str, *, kind: str) -> str:
    text = raw.replace("\r\n", "\n").strip()
    begin = f"-----BEGIN {kind}-----"
    end = f"-----END {kind}-----"
    if begin not in text or end not in text:
        raise ValueError(f"uploaded file is not a PEM {kind.lower()}")
    return text + "\n"


def parse_certificate_pem(raw: str) -> str:
    return parse_pem_bundle(raw, kind="CERTIFICATE")


def parse_private_key_pem(raw: str) -> str:
    text = raw.replace("\r\n", "\n")
    for kind in ("PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY"):
        if f"-----BEGIN {kind}-----" in text:
            return parse_pem_bundle(text, kind=kind)
    raise ValueError("uploaded key is not a PEM private key")

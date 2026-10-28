"""Tests for the plain + TLS dual listener in backend.utils.listeners."""

import asyncio
import datetime
import socket
import ssl
from contextlib import asynccontextmanager

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from fastapi import FastAPI

from backend.utils.listeners import TLSListener, serve_with_tls, tls_listener_from_env


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _self_signed(tmp_path):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    certfile, keyfile = tmp_path / "cert.pem", tmp_path / "key.pem"
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return str(certfile), str(keyfile)


def test_tls_listener_unset(monkeypatch):
    monkeypatch.delenv("VOICEBOX_TLS_CERT", raising=False)
    monkeypatch.delenv("VOICEBOX_TLS_KEY", raising=False)
    assert tls_listener_from_env(17493) is None


def test_tls_listener_defaults(monkeypatch):
    monkeypatch.setenv("VOICEBOX_TLS_CERT", "/c.pem")
    monkeypatch.setenv("VOICEBOX_TLS_KEY", "/k.pem")
    monkeypatch.delenv("VOICEBOX_TLS_HOST", raising=False)
    monkeypatch.delenv("VOICEBOX_TLS_PORT", raising=False)
    assert tls_listener_from_env(17493) == TLSListener("0.0.0.0", 17493, "/c.pem", "/k.pem")


def test_tls_listener_requires_both(monkeypatch):
    monkeypatch.setenv("VOICEBOX_TLS_CERT", "/c.pem")
    monkeypatch.delenv("VOICEBOX_TLS_KEY", raising=False)
    with pytest.raises(ValueError):
        tls_listener_from_env(17493)


def test_serves_plain_and_tls_and_runs_lifespan_once(tmp_path):
    certfile, keyfile = _self_signed(tmp_path)
    startups = []

    @asynccontextmanager
    async def lifespan(app):
        startups.append(1)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    plain_port, tls_port = _free_port(), _free_port()
    tls = TLSListener("127.0.0.1", tls_port, certfile, keyfile)

    async def scenario():
        server = asyncio.create_task(serve_with_tls(app, "127.0.0.1", plain_port, tls, log_level="warning"))
        ctx = ssl.create_default_context(cafile=certfile)
        async with httpx.AsyncClient(verify=ctx) as client:
            for _ in range(100):
                try:
                    secure = await client.get(f"https://localhost:{tls_port}/ping")
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.05)
            else:
                pytest.fail("TLS listener never came up")
            plain = await client.get(f"http://127.0.0.1:{plain_port}/ping")
        assert secure.json() == {"ok": True}
        assert plain.json() == {"ok": True}

        server.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server

    asyncio.run(scenario())
    assert startups == [1]

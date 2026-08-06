"""Integration test: mint a token from a mutual-TLS token endpoint.

Runs a real HTTPS server on loopback that REQUIRES a client certificate, with a
throwaway CA minted at test time. This exercises the actual TLS handshake and
the real `_tls_kwargs` path -- a mocked transport cannot show whether a client
certificate is presented, which is exactly the failure mode this configuration
exists to prevent.

No external services and no network access: loopback only.
"""

import datetime
import json
import os
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Tuple

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy_auth.async_oauth2 import (  # noqa: E402
    AsyncOAuth2ClientCredential,
    OAuth2Config,
    OAuth2TokenError,
)

pytest.importorskip("cryptography")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

TOKEN_VALUE = "token-minted-over-mtls"


def _key_pem(key) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _mint_ca() -> Tuple[Any, x509.Certificate]:  # type: ignore[valid-type]
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "h2o-test-ca")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _mint_leaf(ca_key, ca_cert, common_name: str, *, dns_name: Optional[str] = None):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1))
    )
    if dns_name:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(dns_name)]), critical=False
        )
    return key, builder.sign(ca_key, hashes.SHA256())


class _TokenHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0) or 0)
        self.rfile.read(length)
        body = json.dumps({"access_token": TOKEN_VALUE, "expires_in": 3600}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A rejected handshake is an EXPECTED outcome in one of these tests;
        # don't spew a traceback for it.
        pass


@pytest.fixture(scope="module")
def mtls_endpoint(tmp_path_factory):
    """An HTTPS token endpoint that requires a client certificate."""
    tmp = tmp_path_factory.mktemp("mtls")
    ca_key, ca_cert = _mint_ca()
    server_key, server_cert = _mint_leaf(ca_key, ca_cert, "localhost", dns_name="localhost")
    client_key, client_cert = _mint_leaf(ca_key, ca_cert, "h2ogpte-client")

    paths = {
        "ca": tmp / "ca.pem",
        "server_cert": tmp / "server.crt",
        "server_key": tmp / "server.key",
        "client_cert": tmp / "client.crt",
        "client_key": tmp / "client.key",
        "client_combined": tmp / "client-combined.pem",
    }
    paths["ca"].write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    paths["server_cert"].write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    paths["server_key"].write_bytes(_key_pem(server_key))
    paths["client_cert"].write_bytes(client_cert.public_bytes(serialization.Encoding.PEM))
    paths["client_key"].write_bytes(_key_pem(client_key))
    paths["client_combined"].write_bytes(
        client_cert.public_bytes(serialization.Encoding.PEM) + _key_pem(client_key)
    )

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(paths["server_cert"]), str(paths["server_key"]))
    context.load_verify_locations(str(paths["ca"]))
    context.verify_mode = ssl.CERT_REQUIRED  # <- the point of this test

    server = _QuietServer(("127.0.0.1", 0), _TokenHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {"url": f"https://localhost:{port}/token", **{k: str(v) for k, v in paths.items()}}

    server.shutdown()
    server.server_close()


def _config(endpoint, **overrides):
    base = {
        "token_url": endpoint["url"],
        "client_id": "h2ogpte-client",
        "client_secret": "shhh",
        "ca_bundle": endpoint["ca"],
    }
    base.update(overrides)
    return OAuth2Config.from_dict(base)


@pytest.mark.asyncio
async def test_mints_token_presenting_client_certificate(mtls_endpoint):
    credential = AsyncOAuth2ClientCredential(
        _config(
            mtls_endpoint,
            mtls_cert=mtls_endpoint["client_cert"],
            mtls_key=mtls_endpoint["client_key"],
        )
    )
    token = await credential.get_token()
    assert token.token == TOKEN_VALUE


@pytest.mark.asyncio
async def test_combined_pem_client_certificate_is_accepted(mtls_endpoint):
    """mtls_cert alone is valid when the PEM also carries the private key."""
    credential = AsyncOAuth2ClientCredential(
        _config(mtls_endpoint, mtls_cert=mtls_endpoint["client_combined"])
    )
    token = await credential.get_token()
    assert token.token == TOKEN_VALUE


@pytest.mark.asyncio
async def test_without_client_certificate_the_handshake_is_rejected(mtls_endpoint):
    """Proves the server really enforces mTLS, so the positive test means something."""
    credential = AsyncOAuth2ClientCredential(_config(mtls_endpoint))
    with pytest.raises(OAuth2TokenError) as exc:
        await credential.get_token()
    assert "token request to" in str(exc.value)


@pytest.mark.asyncio
async def test_untrusted_ca_is_rejected(mtls_endpoint, tmp_path):
    """A CA bundle that doesn't include the server's issuer must fail closed."""
    other_ca_key, other_ca_cert = _mint_ca()
    other_ca = tmp_path / "other-ca.pem"
    other_ca.write_bytes(other_ca_cert.public_bytes(serialization.Encoding.PEM))

    credential = AsyncOAuth2ClientCredential(
        _config(
            mtls_endpoint,
            ca_bundle=str(other_ca),
            mtls_cert=mtls_endpoint["client_cert"],
            mtls_key=mtls_endpoint["client_key"],
        )
    )
    with pytest.raises(OAuth2TokenError):
        await credential.get_token()


@pytest.mark.asyncio
async def test_insecure_skip_tls_verify_bypasses_ca_check(mtls_endpoint, tmp_path):
    """The bring-up escape hatch works -- and still presents the client cert."""
    credential = AsyncOAuth2ClientCredential(
        _config(
            mtls_endpoint,
            ca_bundle=None,
            insecure_skip_tls_verify=True,
            mtls_cert=mtls_endpoint["client_cert"],
            mtls_key=mtls_endpoint["client_key"],
        )
    )
    token = await credential.get_token()
    assert token.token == TOKEN_VALUE

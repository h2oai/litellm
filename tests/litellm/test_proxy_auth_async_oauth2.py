"""Unit tests for litellm.proxy_auth.async_oauth2.

Covers configuration validation, secret indirection, the three auth styles,
token caching/refresh semantics, single-flight, and TTL derivation.
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy_auth.async_oauth2 import (  # noqa: E402
    CLIENT_ASSERTION_TYPE_JWT_BEARER,
    FALLBACK_TOKEN_TTL_SEC,
    AsyncOAuth2ClientCredential,
    CredentialRegistry,
    OAuth2Config,
    OAuth2ConfigError,
    OAuth2TokenError,
    resolve_secret_ref_to_file,
    resolve_secret_ref,
)

BASE_CONFIG = {
    "token_url": "https://idp.example.com/token",
    "client_id": "test-client",
    "client_secret": "shhh",
}


def _jwt_with_exp(exp: int) -> str:
    import base64

    def seg(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'none'})}.{seg({'exp': exp})}.sig"


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Captures token requests and replies with a canned response."""

    def __init__(self, payload: Dict[str, Any], status_code: int = 200, body: Optional[str] = None):
        self.payload = payload
        self.status_code = status_code
        self.body = body
        self.requests: List[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        if self.body is not None:
            return httpx.Response(self.status_code, text=self.body)
        return httpx.Response(self.status_code, json=self.payload)


def _patch_transport(transport: httpx.AsyncBaseTransport):
    """Force AsyncOAuth2ClientCredential's httpx client to use `transport`."""
    real_init = httpx.AsyncClient.__init__

    def fake_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        # verify/cert are meaningless with a stub transport and httpx would still
        # try to load the files, so drop them here.
        kwargs.pop("verify", None)
        kwargs.pop("cert", None)
        real_init(self, *args, **kwargs)

    return patch.object(httpx.AsyncClient, "__init__", fake_init)


# --------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------


def test_config_requires_token_url_and_client_id():
    with pytest.raises(OAuth2ConfigError, match="token_url is required"):
        OAuth2Config.from_dict({"client_id": "x", "client_secret": "y"})
    with pytest.raises(OAuth2ConfigError, match="client_id is required"):
        OAuth2Config.from_dict({"token_url": "https://x/token", "client_secret": "y"})


def test_config_rejects_unsupported_grant_and_auth_style():
    with pytest.raises(OAuth2ConfigError, match="grant_type"):
        OAuth2Config.from_dict({**BASE_CONFIG, "grant_type": "password"})
    with pytest.raises(OAuth2ConfigError, match="auth_style"):
        OAuth2Config.from_dict({**BASE_CONFIG, "auth_style": "magic"})


def test_config_requires_credential_matching_auth_style():
    with pytest.raises(OAuth2ConfigError, match="client_private_key is required"):
        OAuth2Config.from_dict(
            {"token_url": "https://x/token", "client_id": "c", "auth_style": "private_key_jwt"}
        )
    with pytest.raises(OAuth2ConfigError, match="client_secret is required"):
        OAuth2Config.from_dict({"token_url": "https://x/token", "client_id": "c"})


def test_config_rejects_mtls_key_without_cert():
    with pytest.raises(OAuth2ConfigError, match="mtls_key is set without"):
        OAuth2Config.from_dict({**BASE_CONFIG, "mtls_key": "/tmp/k.pem"})
    # cert alone is legal: a combined PEM.
    cfg = OAuth2Config.from_dict({**BASE_CONFIG, "mtls_cert": "/tmp/combined.pem"})
    assert cfg.mtls_cert == "/tmp/combined.pem"
    assert cfg.mtls_key is None


def test_config_defaults():
    cfg = OAuth2Config.from_dict(BASE_CONFIG)
    assert cfg.auth_style == "client_secret_post"
    assert cfg.header_name == "Authorization"
    assert cfg.header_scheme == "Bearer"
    assert cfg.refresh_buffer_sec == 30
    assert cfg.insecure_skip_tls_verify is False


def test_fingerprint_tracks_config_and_not_resolved_secret():
    a = OAuth2Config.from_dict({**BASE_CONFIG, "client_secret": "os.environ/TOKEN_A"})
    b = OAuth2Config.from_dict({**BASE_CONFIG, "client_secret": "os.environ/TOKEN_A"})
    c = OAuth2Config.from_dict({**BASE_CONFIG, "scope": "extra.scope"})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


# --------------------------------------------------------------------------
# Secret indirection
# --------------------------------------------------------------------------


def test_resolve_secret_ref_env():
    with patch.dict(os.environ, {"MY_SECRET": "resolved-value"}):
        assert resolve_secret_ref("os.environ/MY_SECRET", field_name="client_secret") == "resolved-value"


def test_resolve_secret_ref_env_missing_names_the_variable_not_the_value():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(OAuth2ConfigError) as exc:
            resolve_secret_ref("os.environ/ABSENT_VAR", field_name="client_secret")
    assert "ABSENT_VAR" in str(exc.value)


def test_resolve_secret_ref_file(tmp_path):
    secret_file = tmp_path / "secret.pem"
    secret_file.write_text("file-secret")
    assert resolve_secret_ref(f"file://{secret_file}", field_name="client_private_key") == "file-secret"


def test_resolve_secret_ref_file_must_be_absolute():
    with pytest.raises(OAuth2ConfigError, match="must be absolute"):
        resolve_secret_ref("file://relative/path", field_name="client_private_key")


def test_resolve_secret_ref_literal_and_empty():
    assert resolve_secret_ref("literal", field_name="client_secret") == "literal"
    with pytest.raises(OAuth2ConfigError, match="required but empty"):
        resolve_secret_ref("   ", field_name="client_secret")


# --------------------------------------------------------------------------
# Token requests -- one per auth style
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_secret_post_sends_credentials_in_body():
    transport = _RecordingTransport({"access_token": "tok", "expires_in": 3600})
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict({**BASE_CONFIG, "scope": "a b"}))
    with _patch_transport(transport):
        token = await cred.get_token()
    assert token.token == "tok"
    body = transport.requests[0].content.decode()
    assert "client_secret=shhh" in body
    assert "grant_type=client_credentials" in body
    assert "scope=a+b" in body
    assert "authorization" not in {k.lower() for k in transport.requests[0].headers.keys()}


@pytest.mark.asyncio
async def test_client_secret_basic_uses_http_basic_auth():
    transport = _RecordingTransport({"access_token": "tok", "expires_in": 60})
    cred = AsyncOAuth2ClientCredential(
        OAuth2Config.from_dict({**BASE_CONFIG, "auth_style": "client_secret_basic"})
    )
    with _patch_transport(transport):
        await cred.get_token()
    request = transport.requests[0]
    assert request.headers["authorization"].startswith("Basic ")
    assert "client_secret" not in request.content.decode()


@pytest.mark.asyncio
async def test_private_key_jwt_sends_signed_es256_assertion(tmp_path):
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    key_file = tmp_path / "client.key"
    key_file.write_text(pem)

    transport = _RecordingTransport({"access_token": "tok", "expires_in": 300})
    cred = AsyncOAuth2ClientCredential(
        OAuth2Config.from_dict(
            {
                "token_url": "https://idp.example.com/token",
                "client_id": "assert-client",
                "auth_style": "private_key_jwt",
                "assertion_alg": "ES256",
                "client_private_key": f"file://{key_file}",
            }
        )
    )
    with _patch_transport(transport):
        await cred.get_token()

    body = httpx.QueryParams(transport.requests[0].content.decode())
    assert body["client_assertion_type"] == CLIENT_ASSERTION_TYPE_JWT_BEARER
    claims = jwt.decode(
        body["client_assertion"],
        key.public_key(),
        algorithms=["ES256"],
        audience="https://idp.example.com/token",
    )
    assert claims["iss"] == "assert-client"
    assert claims["sub"] == "assert-client"
    assert claims["jti"]  # replay protection
    assert claims["exp"] > claims["iat"]


@pytest.mark.asyncio
async def test_private_key_jwt_audience_override():
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()

    transport = _RecordingTransport({"access_token": "tok", "expires_in": 300})
    cred = AsyncOAuth2ClientCredential(
        OAuth2Config.from_dict(
            {
                "token_url": "https://idp.example.com/token",
                "client_id": "c",
                "auth_style": "private_key_jwt",
                "client_private_key": pem,
                "audience": "https://issuer.example.com",
            }
        )
    )
    with _patch_transport(transport):
        await cred.get_token()
    body = httpx.QueryParams(transport.requests[0].content.decode())
    claims = jwt.decode(
        body["client_assertion"],
        key.public_key(),
        algorithms=["ES256"],
        audience="https://issuer.example.com",
    )
    assert claims["aud"] == "https://issuer.example.com"


@pytest.mark.asyncio
async def test_private_key_jwt_bad_key_names_the_field_not_the_key():
    cred = AsyncOAuth2ClientCredential(
        OAuth2Config.from_dict(
            {
                "token_url": "https://idp.example.com/token",
                "client_id": "c",
                "auth_style": "private_key_jwt",
                "client_private_key": "not-a-pem",
            }
        )
    )
    with pytest.raises(OAuth2ConfigError) as exc:
        await cred.get_token()
    assert "client_private_key" in str(exc.value)
    assert "not-a-pem" not in str(exc.value)


# --------------------------------------------------------------------------
# Caching, refresh, single flight
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_is_cached_across_calls():
    transport = _RecordingTransport({"access_token": "tok", "expires_in": 3600})
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    with _patch_transport(transport):
        for _ in range(5):
            await cred.get_token()
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_refresh_buffer_forces_early_refresh():
    # expires_in below the refresh buffer => never considered fresh.
    transport = _RecordingTransport({"access_token": "tok", "expires_in": 10})
    cred = AsyncOAuth2ClientCredential(
        OAuth2Config.from_dict({**BASE_CONFIG, "refresh_buffer_sec": 30})
    )
    with _patch_transport(transport):
        await cred.get_token()
        await cred.get_token()
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_single_flight_collapses_concurrent_refreshes():
    class SlowTransport(_RecordingTransport):
        async def handle_async_request(self, request):
            await asyncio.sleep(0.05)
            return await super().handle_async_request(request)

    transport = SlowTransport({"access_token": "tok", "expires_in": 3600})
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    with _patch_transport(transport):
        tokens = await asyncio.gather(*[cred.get_token() for _ in range(20)])
    assert len({t.token for t in tokens}) == 1
    assert len(transport.requests) == 1, "20 concurrent callers must mint exactly one token"


@pytest.mark.asyncio
async def test_invalidate_forces_remint():
    transport = _RecordingTransport({"access_token": "tok", "expires_in": 3600})
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    with _patch_transport(transport):
        await cred.get_token()
        cred.invalidate()
        await cred.get_token()
    assert len(transport.requests) == 2


# --------------------------------------------------------------------------
# TTL derivation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttl_falls_back_to_jwt_exp_when_expires_in_absent():
    exp = int(time.time()) + 1234
    transport = _RecordingTransport({"access_token": _jwt_with_exp(exp)})
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    with _patch_transport(transport):
        token = await cred.get_token()
    assert token.expires_on == exp


@pytest.mark.asyncio
async def test_ttl_falls_back_to_constant_for_opaque_token():
    transport = _RecordingTransport({"access_token": "opaque-token"})
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    before = time.time()
    with _patch_transport(transport):
        token = await cred.get_token()
    assert before + FALLBACK_TOKEN_TTL_SEC - 5 <= token.expires_on <= time.time() + FALLBACK_TOKEN_TTL_SEC


# --------------------------------------------------------------------------
# Error surfaces
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_200_includes_status_and_body_excerpt():
    transport = _RecordingTransport({}, status_code=401, body="invalid_client")
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    with _patch_transport(transport):
        with pytest.raises(OAuth2TokenError) as exc:
            await cred.get_token()
    assert "401" in str(exc.value)
    assert "invalid_client" in str(exc.value)


@pytest.mark.asyncio
async def test_missing_access_token_is_reported():
    transport = _RecordingTransport({"token_type": "Bearer"})
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    with _patch_transport(transport):
        with pytest.raises(OAuth2TokenError, match="no 'access_token'"):
            await cred.get_token()


@pytest.mark.asyncio
async def test_non_json_body_is_reported():
    transport = _RecordingTransport({}, status_code=200, body="<html>gateway</html>")
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    with _patch_transport(transport):
        with pytest.raises(OAuth2TokenError, match="non-JSON"):
            await cred.get_token()


@pytest.mark.asyncio
async def test_transport_error_is_wrapped():
    class BoomTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("no route to host")

    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    with _patch_transport(BoomTransport()):
        with pytest.raises(OAuth2TokenError, match="token request to"):
            await cred.get_token()


# --------------------------------------------------------------------------
# TLS material -- bad paths must be config errors, not TLS mysteries
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_ca_bundle_is_a_config_error():
    cred = AsyncOAuth2ClientCredential(
        OAuth2Config.from_dict({**BASE_CONFIG, "ca_bundle": "/nonexistent/ca.pem"})
    )
    with pytest.raises(OAuth2ConfigError, match="ca_bundle"):
        await cred.get_token()


@pytest.mark.asyncio
async def test_missing_client_cert_is_a_config_error():
    cred = AsyncOAuth2ClientCredential(
        OAuth2Config.from_dict(
            {**BASE_CONFIG, "mtls_cert": "/nonexistent/client.crt", "mtls_key": "/nonexistent/client.key"}
        )
    )
    with pytest.raises(OAuth2ConfigError, match="mtls_cert"):
        await cred.get_token()


def test_no_tls_config_leaves_httpx_defaults_untouched():
    """A credential with no TLS fields must not pass verify= at all, so httpx's
    default context (and litellm's SSL_CERT_FILE handling) applies unchanged."""
    cred = AsyncOAuth2ClientCredential(OAuth2Config.from_dict(BASE_CONFIG))
    assert cred._build_ssl_context() is None


def test_tls_file_reference_resolves_to_path(tmp_path):
    cert_file = tmp_path / "client.crt"
    cert_file.write_text("-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----\n")
    with resolve_secret_ref_to_file(f"file://{cert_file}", field_name="mtls_cert") as path:
        assert path != f"file://{cert_file}"
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == cert_file.read_text()


def test_tls_env_reference_can_point_at_mounted_file(tmp_path):
    cert_file = tmp_path / "client.crt"
    cert_file.write_text("cert")
    with patch.dict(os.environ, {"CLIENT_CERT_PATH": str(cert_file)}):
        with resolve_secret_ref_to_file("os.environ/CLIENT_CERT_PATH", field_name="mtls_cert") as path:
            assert path == str(cert_file)


def test_tls_env_reference_can_contain_pem_material():
    pem = "-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----\n"
    with patch.dict(os.environ, {"CLIENT_CERT_PEM": pem}):
        with resolve_secret_ref_to_file("os.environ/CLIENT_CERT_PEM", field_name="mtls_cert") as path:
            assert os.path.exists(path)
            with open(path) as f:
                assert f.read() == pem
        assert not os.path.exists(path)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_shares_credentials_by_fingerprint():
    registry = CredentialRegistry()
    same_a = registry.get(OAuth2Config.from_dict(BASE_CONFIG))
    same_b = registry.get(OAuth2Config.from_dict(BASE_CONFIG))
    other = registry.get(OAuth2Config.from_dict({**BASE_CONFIG, "client_id": "other"}))
    assert same_a is same_b, "one IdP client => one token => one refresh"
    assert other is not same_a
    assert len(registry) == 2


def test_auth_header_uses_configured_name_and_scheme():
    from litellm.proxy_auth.credentials import AccessToken

    cred = AsyncOAuth2ClientCredential(
        OAuth2Config.from_dict({**BASE_CONFIG, "header_name": "X-Gateway-Token", "header_scheme": ""})
    )
    name, value = cred.auth_header(AccessToken(token="abc", expires_on=0))
    assert (name, value) == ("X-Gateway-Token", "abc")

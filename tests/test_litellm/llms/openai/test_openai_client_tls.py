"""Per-deployment TLS transport config on the openai provider path.

Three things are verified:

1. Extraction of ssl_verify / client_cert / client_key from litellm_params.
2. That TLS config participates in the client cache key. This is the sharp edge:
   `_get_openai_client` builds its cache key from `locals()` filtered by a field
   list, so a parameter that changes the client but is missing from that list
   would let two deployments with DIFFERENT client certificates share one cached
   client -- presenting one model's mTLS identity for another model's requests.
3. End-to-end: a completion against a mutual-TLS OpenAI-compatible endpoint
   succeeds when the deployment declares a client certificate, and fails without
   it. A mocked transport cannot show whether a certificate was presented, which
   is the entire point of the feature.
"""

import datetime
import json
import os
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm  # noqa: E402
from litellm.llms.openai.common_utils import BaseOpenAILLM  # noqa: E402

pytest.importorskip("cryptography")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402


# --------------------------------------------------------------------------
# 1. Extraction from litellm_params
# --------------------------------------------------------------------------


def test_no_litellm_params_yields_no_tls_config():
    assert BaseOpenAILLM.tls_client_kwargs(None) == {"ssl_verify": None, "client_cert": None}
    assert BaseOpenAILLM.tls_client_kwargs({}) == {"ssl_verify": None, "client_cert": None}


def test_model_without_tls_config_is_unaffected():
    params = {"model": "openai/gpt-4o", "api_key": "sk-x", "api_base": "https://api.openai.com/v1"}
    assert BaseOpenAILLM.tls_client_kwargs(params) == {"ssl_verify": None, "client_cert": None}


def test_cert_and_key_pair():
    params = {"client_cert": "/etc/tls/client.crt", "client_key": "/etc/tls/client.key"}
    assert BaseOpenAILLM.tls_client_kwargs(params)["client_cert"] == (
        "/etc/tls/client.crt",
        "/etc/tls/client.key",
    )


def test_combined_pem_without_key():
    params = {"client_cert": "/etc/tls/combined.pem"}
    assert BaseOpenAILLM.tls_client_kwargs(params)["client_cert"] == "/etc/tls/combined.pem"


def test_key_without_cert_is_ignored_not_crashed():
    """httpx cannot use a key alone; surfacing it as None keeps the request path
    intact (the credential layer rejects the same mistake with a clear error)."""
    assert BaseOpenAILLM.tls_client_kwargs({"client_key": "/etc/tls/only.key"})["client_cert"] is None


def test_ssl_verify_passthrough():
    assert BaseOpenAILLM.tls_client_kwargs({"ssl_verify": "/etc/tls/ca.pem"})["ssl_verify"] == "/etc/tls/ca.pem"
    assert BaseOpenAILLM.tls_client_kwargs({"ssl_verify": False})["ssl_verify"] is False


# --------------------------------------------------------------------------
# 2. Cache key must separate clients by TLS identity
# --------------------------------------------------------------------------


def _cache_key(**overrides):
    params = {
        "is_async": True,
        "api_key": "sk-same",
        "api_base": "https://gw.example.com/v1",
        "api_version": None,
        "timeout": 60.0,
        "max_retries": 2,
        "organization": None,
        "ssl_verify": None,
        "client_cert": None,
    }
    params.update(overrides)
    return BaseOpenAILLM.get_openai_client_cache_key(
        client_initialization_params=params, client_type="openai"
    )


def test_different_client_certs_do_not_share_a_cached_client():
    key_a = _cache_key(client_cert=("/etc/tls/a.crt", "/etc/tls/a.key"))
    key_b = _cache_key(client_cert=("/etc/tls/b.crt", "/etc/tls/b.key"))
    assert key_a != key_b, "two mTLS identities must never share one cached client"


def test_cert_bearing_and_plain_deployments_do_not_share_a_cached_client():
    assert _cache_key(client_cert="/etc/tls/a.pem") != _cache_key()


def test_different_ca_bundles_do_not_share_a_cached_client():
    assert _cache_key(ssl_verify="/etc/tls/ca-a.pem") != _cache_key(ssl_verify="/etc/tls/ca-b.pem")


def test_identical_tls_config_reuses_one_client():
    assert _cache_key(client_cert="/etc/tls/a.pem") == _cache_key(client_cert="/etc/tls/a.pem")


# --------------------------------------------------------------------------
# 3. End-to-end against a mutual-TLS OpenAI-compatible endpoint
# --------------------------------------------------------------------------


def _key_pem(key) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _mint_ca():
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


def _mint_leaf(ca_key, ca_cert, common_name: str, dns_name: Optional[str] = None):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
    )
    if dns_name:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(dns_name)]), critical=False
        )
    return key, builder.sign(ca_key, hashes.SHA256())


class _ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0) or 0)
        self.rfile.read(length)
        body = json.dumps(
            {
                "id": "chatcmpl-mtls",
                "object": "chat.completion",
                "created": 0,
                "model": "gw-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "mtls-ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
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
        # A rejected handshake is expected in one of these tests.
        pass


@pytest.fixture(scope="module")
def mtls_llm_endpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("openai-mtls")
    ca_key, ca_cert = _mint_ca()
    server_key, server_cert = _mint_leaf(ca_key, ca_cert, "localhost", dns_name="localhost")
    client_key, client_cert = _mint_leaf(ca_key, ca_cert, "h2ogpte-client")

    paths = {
        "ca": tmp / "ca.pem",
        "server_cert": tmp / "server.crt",
        "server_key": tmp / "server.key",
        "client_cert": tmp / "client.crt",
        "client_key": tmp / "client.key",
    }
    paths["ca"].write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    paths["server_cert"].write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    paths["server_key"].write_bytes(_key_pem(server_key))
    paths["client_cert"].write_bytes(client_cert.public_bytes(serialization.Encoding.PEM))
    paths["client_key"].write_bytes(_key_pem(client_key))

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(paths["server_cert"]), str(paths["server_key"]))
    context.load_verify_locations(str(paths["ca"]))
    context.verify_mode = ssl.CERT_REQUIRED

    server = _QuietServer(("127.0.0.1", 0), _ChatHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    port = server.socket.getsockname()[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    yield {"api_base": f"https://localhost:{port}/v1", **{k: str(v) for k, v in paths.items()}}

    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """Each case must build its own client -- otherwise a cached one from a
    previous case would mask both the success and the failure."""
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
async def test_completion_succeeds_with_per_deployment_client_cert(mtls_llm_endpoint):
    response = await litellm.acompletion(
        model="openai/gw-model",
        messages=[{"role": "user", "content": "hi"}],
        api_base=mtls_llm_endpoint["api_base"],
        api_key="unused",
        ssl_verify=mtls_llm_endpoint["ca"],
        client_cert=mtls_llm_endpoint["client_cert"],
        client_key=mtls_llm_endpoint["client_key"],
    )
    assert response.choices[0].message.content == "mtls-ok"


@pytest.mark.asyncio
async def test_completion_fails_without_client_cert(mtls_llm_endpoint):
    """Proves the endpoint really requires a client certificate, so the positive
    test above is meaningful rather than passing for some unrelated reason.

    The assertion is deliberately on the FAILURE, not on the message: litellm
    normalises a TLS handshake rejection into a generic connection error
    (InternalServerError / "OpenAIException - Connection error."), so matching on
    'certificate' would be asserting litellm's wrapping rather than our behaviour.
    """
    with pytest.raises(Exception) as exc:
        await litellm.acompletion(
            model="openai/gw-model",
            messages=[{"role": "user", "content": "hi"}],
            api_base=mtls_llm_endpoint["api_base"],
            api_key="unused",
            ssl_verify=mtls_llm_endpoint["ca"],
            num_retries=0,
        )
    assert "connection" in str(exc.value).lower() or "ssl" in str(exc.value).lower()

import datetime
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

import httpx
import pytest
import pytest_asyncio
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

import litellm
from litellm.integrations.h2o.litellm_oauth_auth_hook import OAuthAuthHook
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
from litellm.llms.openai.common_utils import BaseOpenAILLM


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
        "client_key": None,
        **overrides,
    }
    return BaseOpenAILLM.get_openai_client_cache_key(client_initialization_params=params, client_type="openai")


@pytest.mark.parametrize(
    "field, value_a, value_b",
    [
        ("client_cert", "/etc/tls/a.crt", "/etc/tls/b.crt"),
        ("client_key", "/etc/tls/a.key", "/etc/tls/b.key"),
        ("ssl_verify", "/etc/tls/ca-a.pem", "/etc/tls/ca-b.pem"),
    ],
)
def test_tls_settings_partition_the_client_cache(field, value_a, value_b):
    assert _cache_key(**{field: value_a}) != _cache_key(**{field: value_b})
    assert _cache_key(**{field: value_a}) == _cache_key(**{field: value_a})


def test_tls_client_kwargs_reads_litellm_params():
    params = {"client_cert": "/c.crt", "client_key": "/c.key", "ssl_verify": "/ca.pem", "api_key": "x"}
    assert BaseOpenAILLM.tls_client_kwargs(params) == {
        "ssl_verify": "/ca.pem",
        "client_cert": "/c.crt",
        "client_key": "/c.key",
    }
    assert BaseOpenAILLM.tls_client_kwargs(None) == {"ssl_verify": None, "client_cert": None, "client_key": None}


def test_ssl_verify_false_with_client_cert_disables_verification(mtls_llm_endpoint):
    from litellm.llms.custom_httpx.http_handler import get_client_cert_ssl_context

    context = get_client_cert_ssl_context(False, mtls_llm_endpoint["client_cert"], mtls_llm_endpoint["client_key"])
    assert (context.verify_mode, context.check_hostname) == (ssl.CERT_NONE, False)


def test_client_cert_with_a_caller_supplied_sslcontext_is_refused():
    from litellm.llms.custom_httpx.http_handler import get_client_cert_ssl_context

    with pytest.raises(TypeError, match="cannot be combined"):
        get_client_cert_ssl_context(ssl.create_default_context(), "/tmp/does-not-matter.pem")


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
        builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName(dns_name)]), critical=False)
    return key, builder.sign(ca_key, hashes.SHA256())


class _RecordingHandler(BaseHTTPRequestHandler):
    seen: List[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0) or 0)))
        peer = self.connection.getpeercert() or {}
        common_name = dict(pair[0] for pair in peer.get("subject", ()))["commonName"]
        _RecordingHandler.seen.append(
            {
                "path": self.path,
                "peer_cn": common_name,
                "body_keys": sorted(body),
                "authorization": self.headers.get("authorization"),
            }
        )
        if self.path.endswith("/responses"):
            response = {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gw-model",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "mtls-ok", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        elif self.path.endswith("/embeddings"):
            response = {
                "object": "list",
                "model": "gw-embed",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        else:
            response = {
                "id": "chatcmpl-mtls",
                "object": "chat.completion",
                "created": 0,
                "model": "gw-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "mtls-ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        pass


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
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

    server = _QuietServer(("127.0.0.1", 0), _RecordingHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    port = server.socket.getsockname()[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    yield {"api_base": f"https://localhost:{port}/v1", **{k: str(v) for k, v in paths.items()}}

    server.shutdown()
    server.server_close()


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def _fresh_state():
    litellm.in_memory_llm_clients_cache.flush_cache()
    _RecordingHandler.seen.clear()
    yield
    await close_litellm_async_clients()
    litellm.in_memory_llm_clients_cache.flush_cache()


def _tls_kwargs(endpoint: dict) -> dict:
    return {
        "api_base": endpoint["api_base"],
        "api_key": "unused",
        "ssl_verify": endpoint["ca"],
        "client_cert": endpoint["client_cert"],
        "client_key": endpoint["client_key"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_acompletion_presents_the_deployment_client_cert(mtls_llm_endpoint, stream):
    response = await litellm.acompletion(
        model="openai/gw-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=stream,
        **_tls_kwargs(mtls_llm_endpoint),
    )
    if stream:
        async for _ in response:
            pass
    else:
        assert response.choices[0].message.content == "mtls-ok"
    assert [request["peer_cn"] for request in _RecordingHandler.seen] == ["h2ogpte-client"]


@pytest.mark.parametrize("stream", [False, True])
def test_sync_completion_presents_the_deployment_client_cert(mtls_llm_endpoint, stream):
    response = litellm.completion(
        model="openai/gw-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=stream,
        **_tls_kwargs(mtls_llm_endpoint),
    )
    if stream:
        for _ in response:
            pass
    else:
        assert response.choices[0].message.content == "mtls-ok"
    assert [request["peer_cn"] for request in _RecordingHandler.seen] == ["h2ogpte-client"]


@pytest.mark.asyncio
async def test_aresponses_presents_the_deployment_client_cert(mtls_llm_endpoint):
    response = await litellm.aresponses(model="openai/gw-model", input="hi", **_tls_kwargs(mtls_llm_endpoint))
    assert response.output[0].content[0].text == "mtls-ok"
    assert [request["peer_cn"] for request in _RecordingHandler.seen] == ["h2ogpte-client"]


def test_sync_responses_presents_the_deployment_client_cert(mtls_llm_endpoint):
    response = litellm.responses(model="openai/gw-model", input="hi", **_tls_kwargs(mtls_llm_endpoint))
    assert response.output[0].content[0].text == "mtls-ok"
    assert [request["peer_cn"] for request in _RecordingHandler.seen] == ["h2ogpte-client"]


@pytest.mark.asyncio
async def test_aembedding_presents_the_deployment_client_cert(mtls_llm_endpoint):
    response = await litellm.aembedding(model="openai/gw-embed", input="hi", **_tls_kwargs(mtls_llm_endpoint))
    assert response.data[0]["embedding"] == [0.5]
    assert [request["peer_cn"] for request in _RecordingHandler.seen] == ["h2ogpte-client"]


def test_sync_embedding_presents_the_deployment_client_cert(mtls_llm_endpoint):
    response = litellm.embedding(model="openai/gw-embed", input="hi", **_tls_kwargs(mtls_llm_endpoint))
    assert response.data[0]["embedding"] == [0.5]
    assert [request["peer_cn"] for request in _RecordingHandler.seen] == ["h2ogpte-client"]


@pytest.mark.asyncio
async def test_tls_params_never_reach_the_upstream_body(mtls_llm_endpoint):
    await litellm.acompletion(
        model="openai/gw-model",
        messages=[{"role": "user", "content": "hi"}],
        **_tls_kwargs(mtls_llm_endpoint),
    )
    assert _RecordingHandler.seen[0]["body_keys"] == ["messages", "model"]


@pytest.mark.asyncio
async def test_completion_fails_without_client_cert(mtls_llm_endpoint):
    with pytest.raises(litellm.InternalServerError, match="Connection error"):
        await litellm.acompletion(
            model="openai/gw-model",
            messages=[{"role": "user", "content": "hi"}],
            api_base=mtls_llm_endpoint["api_base"],
            api_key="unused",
            ssl_verify=mtls_llm_endpoint["ca"],
            num_retries=0,
        )
    assert _RecordingHandler.seen == []


def _global_client(endpoint: dict, is_async: bool):
    import httpx

    context = ssl.create_default_context(cafile=endpoint["ca"])
    context.load_cert_chain(endpoint["server_cert"], endpoint["server_key"])
    return httpx.AsyncClient(verify=context) if is_async else httpx.Client(verify=context)


async def _call(is_async: bool, **kwargs):
    kwargs = {"model": "openai/gw-model", "messages": [{"role": "user", "content": "hi"}], "max_retries": 0, **kwargs}
    if is_async:
        return await litellm.acompletion(**kwargs)
    return litellm.completion(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False])
async def test_client_cert_with_a_global_http_client_is_refused(mtls_llm_endpoint, is_async, monkeypatch):
    attr = "aclient_session" if is_async else "client_session"
    monkeypatch.setattr(litellm, attr, _global_client(mtls_llm_endpoint, is_async))
    with pytest.raises(Exception, match=f"litellm.{attr} is set, so this deployment's client_cert"):
        await _call(is_async, **_tls_kwargs(mtls_llm_endpoint))
    assert _RecordingHandler.seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False])
async def test_global_http_client_is_still_used_without_client_cert(mtls_llm_endpoint, is_async, monkeypatch):
    attr = "aclient_session" if is_async else "client_session"
    monkeypatch.setattr(litellm, attr, _global_client(mtls_llm_endpoint, is_async))
    response = await _call(is_async, api_base=mtls_llm_endpoint["api_base"], api_key="unused")
    assert response.choices[0].message.content == "mtls-ok"
    assert [request["peer_cn"] for request in _RecordingHandler.seen] == ["localhost"]


_MESSAGES = [{"role": "user", "content": "hi"}]
_OAUTH = {
    "token_url": "https://idp.invalid/token",
    "client_id": "h2ogpte",
    "client_private_key": _key_pem(ec.generate_private_key(ec.SECP256R1())).decode(),
}
_OAUTH_CALLS = {
    "acompletion": lambda kw: litellm.acompletion(model="openai/gw-model", messages=_MESSAGES, **kw),
    "acompletion-stream": lambda kw: litellm.acompletion(model="openai/gw-model", messages=_MESSAGES, **kw),
    "aembedding": lambda kw: litellm.aembedding(model="openai/gw-embed", input=["hi"], **kw),
    "aresponses": lambda kw: litellm.aresponses(model="openai/gw-model", input="hi", **kw),
    "anthropic_messages": lambda kw: litellm.anthropic_messages(
        model="openai/gw-model", messages=_MESSAGES, max_tokens=5, **kw
    ),
}


def _oauth_hook() -> OAuthAuthHook:
    return OAuthAuthHook(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"access_token": "minted-jwt", "expires_in": 3600})
        )
    )


def _oauth_deployment(endpoint: dict) -> dict:
    return {**_tls_kwargs(endpoint), "h2o_oauth": _OAUTH}


def _gateway_saw() -> list:
    return [(request["peer_cn"], request["authorization"]) for request in _RecordingHandler.seen]


@pytest.mark.asyncio
@pytest.mark.parametrize("call", list(_OAUTH_CALLS))
async def test_oauth_token_reaches_the_gateway_together_with_the_client_cert(mtls_llm_endpoint, monkeypatch, call):
    monkeypatch.setattr(litellm, "callbacks", [_oauth_hook()])
    await _OAUTH_CALLS[call]({**_oauth_deployment(mtls_llm_endpoint), **({"stream": True} if "stream" in call else {})})
    assert _gateway_saw() == [("h2ogpte-client", "Bearer minted-jwt")]


def test_sync_responses_sends_the_oauth_token_together_with_the_client_cert(mtls_llm_endpoint, monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [_oauth_hook()])
    litellm.responses(model="openai/gw-model", input="hi", **_oauth_deployment(mtls_llm_endpoint))
    assert _gateway_saw() == [("h2ogpte-client", "Bearer minted-jwt")]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["hosted_vllm/gw-model", "anthropic/claude-sonnet-4-5"])
async def test_oauth_token_is_not_sent_by_a_provider_that_drops_the_client_cert(mtls_llm_endpoint, monkeypatch, model):
    monkeypatch.setattr(litellm, "callbacks", [_oauth_hook()])
    with pytest.raises(litellm.AuthenticationError, match="client_cert is only presented on openai/"):
        await litellm.acompletion(
            model=model, messages=_MESSAGES, max_tokens=5, num_retries=0, **_oauth_deployment(mtls_llm_endpoint)
        )
    assert _RecordingHandler.seen == []

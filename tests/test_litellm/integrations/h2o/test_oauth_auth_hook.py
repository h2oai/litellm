import asyncio
import datetime
import json
import ssl
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional, Tuple

import httpx
import jwt
import pytest
import pytest_asyncio
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.h2o.litellm_oauth_auth_hook import OAuthAuthHook
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
from litellm.types.utils import CallTypes

TOKEN_URL = "https://idp.example.com/oauth2/token"
_SIGNING_KEY = ec.generate_private_key(ec.SECP256R1())
SIGNING_KEY_PEM = _SIGNING_KEY.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
).decode()
SIGNING_PUBLIC_PEM = (
    _SIGNING_KEY.public_key()
    .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    .decode()
)


def _config(**overrides) -> dict:
    return {"token_url": TOKEN_URL, "client_id": "h2ogpte", "client_private_key": SIGNING_KEY_PEM, **overrides}


class _Clock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now


class _IdP:
    def __init__(self, respond: Optional[Callable[[int], httpx.Response]] = None, delay: float = 0.0) -> None:
        self.requests: List[Dict[str, str]] = []
        self._respond = respond or (lambda n: httpx.Response(200, json={"access_token": f"tok-{n}", "expires_in": 100}))
        self._delay = delay

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(dict(urllib.parse.parse_qsl(request.content.decode())))
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._respond(len(self.requests))

    def hook(self, clock: Optional[_Clock] = None) -> OAuthAuthHook:
        return OAuthAuthHook(transport=httpx.MockTransport(self.handler), clock=clock or _Clock())


async def _run(hook: OAuthAuthHook, **kwargs) -> Optional[dict]:
    return await hook.async_pre_call_deployment_hook(kwargs, None)


@pytest.mark.asyncio
async def test_deployment_without_oauth_config_is_untouched():
    idp = _IdP()
    assert await _run(idp.hook(), model="openai/x", api_key="static") is None
    assert idp.requests == []


@pytest.mark.asyncio
async def test_token_becomes_api_key_and_config_is_stripped():
    idp = _IdP()
    result = await _run(idp.hook(), model="openai/x", api_key="unused", messages=[], h2o_oauth=_config())
    assert result == {"model": "openai/x", "api_key": "tok-1", "messages": []}


@pytest.mark.parametrize("call_type", [CallTypes.atranscription, CallTypes.atext_completion, None])
@pytest.mark.asyncio
async def test_no_token_is_sent_on_a_call_type_that_drops_the_deployment_client_cert(call_type):
    idp = _IdP()
    with pytest.raises(litellm.AuthenticationError, match="client_cert is only presented on openai/"):
        await idp.hook().async_pre_call_deployment_hook(
            {"model": "openai/x", "client_cert": "/certs/tls.crt", "h2o_oauth": _config()}, call_type
        )
    assert idp.requests == []


@pytest.mark.parametrize(
    "call_type, client_cert",
    [
        (CallTypes.acompletion, "/certs/tls.crt"),
        (CallTypes.anthropic_messages, "/certs/tls.crt"),
        (CallTypes.atranscription, None),
    ],
)
@pytest.mark.asyncio
async def test_token_is_sent_when_the_call_type_presents_the_client_cert_or_none_is_configured(call_type, client_cert):
    kwargs = {"model": "openai/x", "client_cert": client_cert, "h2o_oauth": _config()}
    result = await _IdP().hook().async_pre_call_deployment_hook(kwargs, call_type)
    assert result["api_key"] == "tok-1"


@pytest.mark.asyncio
async def test_token_request_is_a_valid_private_key_jwt_client_credentials_grant():
    idp = _IdP()
    clock = _Clock()
    await _run(idp.hook(clock), h2o_oauth=_config(scope="llm"))

    form = idp.requests[0]
    assertion = jwt.decode(form.pop("client_assertion"), SIGNING_PUBLIC_PEM, algorithms=["ES256"], audience=TOKEN_URL)
    assert form == {
        "grant_type": "client_credentials",
        "client_id": "h2ogpte",
        "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
        "scope": "llm",
    }
    assert assertion["iss"] == assertion["sub"] == "h2ogpte"
    assert assertion["iat"] == int(clock.now)
    assert assertion["exp"] == int(clock.now) + 60
    assert assertion["jti"]


@pytest.mark.asyncio
async def test_scope_is_omitted_when_not_configured():
    idp = _IdP()
    await _run(idp.hook(), h2o_oauth=_config())
    assert "scope" not in idp.requests[0]


@pytest.mark.asyncio
async def test_each_assertion_has_a_unique_jti():
    idp = _IdP()
    clock = _Clock()
    hook = idp.hook(clock)
    await _run(hook, h2o_oauth=_config())
    clock.now += 100
    await _run(hook, h2o_oauth=_config())
    jtis = {jwt.decode(r["client_assertion"], options={"verify_signature": False})["jti"] for r in idp.requests}
    assert len(jtis) == 2


@pytest.mark.parametrize(
    "expires_in, refresh_after",
    [(100, 70.0), (40, 20.0), (None, 270.0)],
)
@pytest.mark.asyncio
async def test_token_is_cached_until_shortly_before_expiry(expires_in, refresh_after):
    body = {"access_token": "tok", **({"expires_in": expires_in} if expires_in is not None else {})}
    idp = _IdP(lambda n: httpx.Response(200, json={**body, "access_token": f"tok-{n}"}))
    clock = _Clock()
    hook = idp.hook(clock)
    start = clock.now

    assert (await _run(hook, h2o_oauth=_config()))["api_key"] == "tok-1"
    clock.now = start + refresh_after - 0.01
    assert (await _run(hook, h2o_oauth=_config()))["api_key"] == "tok-1"
    clock.now = start + refresh_after
    assert (await _run(hook, h2o_oauth=_config()))["api_key"] == "tok-2"
    assert len(idp.requests) == 2


@pytest.mark.parametrize("expires_in", [0, -5])
@pytest.mark.asyncio
async def test_an_already_expired_token_is_not_cached(expires_in):
    idp = _IdP(lambda n: httpx.Response(200, json={"access_token": f"tok-{n}", "expires_in": expires_in}))
    hook = idp.hook()
    assert (await _run(hook, h2o_oauth=_config()))["api_key"] == "tok-1"
    assert (await _run(hook, h2o_oauth=_config()))["api_key"] == "tok-2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, other",
    [("client_id", "b"), ("token_url", "https://other-idp.example.com/token"), ("scope", "other-scope")],
)
async def test_distinct_configs_get_distinct_tokens(field, other):
    idp = _IdP()
    hook = idp.hook()
    first = await _run(hook, h2o_oauth=_config())
    second = await _run(hook, h2o_oauth=_config(**{field: other}))
    again = await _run(hook, h2o_oauth=_config())
    assert (first["api_key"], second["api_key"], again["api_key"]) == ("tok-1", "tok-2", "tok-1")


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_token_request():
    idp = _IdP(delay=0.05)
    hook = idp.hook()
    results = await asyncio.gather(*(_run(hook, h2o_oauth=_config()) for _ in range(20)))
    assert {r["api_key"] for r in results} == {"tok-1"}
    assert len(idp.requests) == 1


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_failure_and_the_next_request_retries():
    idp = _IdP(
        lambda n: httpx.Response(503, text="down") if n == 1 else httpx.Response(200, json={"access_token": "ok"}),
        delay=0.05,
    )
    hook = idp.hook()
    outcomes = await asyncio.gather(*(_run(hook, h2o_oauth=_config()) for _ in range(10)), return_exceptions=True)
    assert all(isinstance(o, litellm.AuthenticationError) for o in outcomes)
    assert len(idp.requests) == 1
    assert (await _run(hook, h2o_oauth=_config()))["api_key"] == "ok"


@pytest.mark.asyncio
async def test_a_cancelled_caller_does_not_cancel_the_shared_token_request():
    idp = _IdP(delay=0.05)
    hook = idp.hook()
    cancelled = asyncio.ensure_future(_run(hook, h2o_oauth=_config()))
    survivor = asyncio.ensure_future(_run(hook, h2o_oauth=_config()))
    await asyncio.sleep(0.01)
    cancelled.cancel()
    assert (await survivor)["api_key"] == "tok-1"
    assert len(idp.requests) == 1


@pytest.mark.asyncio
async def test_a_refresh_in_flight_on_another_event_loop_is_not_awaited_across_loops():
    idp = _IdP(delay=0.3)
    hook = idp.hook()
    other_loop_result: List[Optional[dict]] = []
    thread = threading.Thread(target=lambda: other_loop_result.append(asyncio.run(_run(hook, h2o_oauth=_config()))))
    thread.start()
    await asyncio.sleep(0.1)
    result = await _run(hook, h2o_oauth=_config())
    thread.join()
    assert result["api_key"] and other_loop_result[0]["api_key"]
    assert len(idp.requests) == 2


@pytest.mark.parametrize(
    "response, expected",
    [
        (httpx.Response(401, json={"error": "invalid_client"}), "token endpoint returned 401"),
        (httpx.Response(200, json={"token_type": "Bearer"}), "access_token"),
        (httpx.Response(200, text="<html>"), "unexpected token endpoint response"),
    ],
)
@pytest.mark.asyncio
async def test_token_failures_fail_closed_with_an_authentication_error(response, expected):
    hook = _IdP(lambda n: response).hook()
    with pytest.raises(litellm.AuthenticationError, match=expected):
        await _run(hook, model="openai/x", h2o_oauth=_config())


@pytest.mark.asyncio
async def test_token_endpoint_error_body_is_not_returned_to_the_caller():
    hook = _IdP(lambda n: httpx.Response(400, json={"error_description": "client h2ogpte is locked"})).hook()
    with pytest.raises(litellm.AuthenticationError) as exc:
        await _run(hook, model="openai/x", h2o_oauth=_config())
    assert "token endpoint returned 400" in str(exc.value)
    assert "locked" not in str(exc.value)


@pytest.mark.asyncio
async def test_invalid_config_error_never_echoes_the_private_key():
    idp = _IdP()
    with pytest.raises(litellm.AuthenticationError) as exc:
        await _run(idp.hook(), h2o_oauth={"client_id": "h2ogpte", "client_private_key": SIGNING_KEY_PEM, "typo": 1})
    assert "token_url" in str(exc.value) and "typo" in str(exc.value)
    assert "PRIVATE KEY" not in str(exc.value)
    assert idp.requests == []


@pytest.mark.asyncio
async def test_unusable_private_key_fails_closed_without_echoing_it():
    idp = _IdP()
    with pytest.raises(litellm.AuthenticationError) as exc:
        await _run(idp.hook(), h2o_oauth=_config(client_private_key="not-a-real-key-material"))
    assert "not-a-real-key-material" not in str(exc.value)
    assert idp.requests == []


@pytest.mark.asyncio
async def test_private_key_from_environment_variable(monkeypatch):
    monkeypatch.setenv("H2O_TEST_SIGNING_KEY", SIGNING_KEY_PEM)
    idp = _IdP()
    result = await _run(idp.hook(), h2o_oauth=_config(client_private_key="os.environ/H2O_TEST_SIGNING_KEY"))
    assert result["api_key"] == "tok-1"
    jwt.decode(idp.requests[0]["client_assertion"], SIGNING_PUBLIC_PEM, algorithms=["ES256"], audience=TOKEN_URL)


@pytest.mark.asyncio
async def test_unset_environment_variable_fails_closed(monkeypatch):
    monkeypatch.delenv("H2O_TEST_MISSING_KEY", raising=False)
    with pytest.raises(litellm.AuthenticationError, match="H2O_TEST_MISSING_KEY"):
        await _run(_IdP().hook(), h2o_oauth=_config(client_private_key="os.environ/H2O_TEST_MISSING_KEY"))


@pytest.mark.asyncio
async def test_private_key_from_file(tmp_path):
    key_file = tmp_path / "signing.key"
    key_file.write_text(SIGNING_KEY_PEM)
    idp = _IdP()
    result = await _run(idp.hook(), h2o_oauth=_config(client_private_key=f"file://{key_file}"))
    assert result["api_key"] == "tok-1"
    jwt.decode(idp.requests[0]["client_assertion"], SIGNING_PUBLIC_PEM, algorithms=["ES256"], audience=TOKEN_URL)


@pytest.mark.asyncio
async def test_rs256_assertion_alg():
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = rsa_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    idp = _IdP()
    await _run(idp.hook(), h2o_oauth=_config(client_private_key=pem, assertion_alg="RS256"))
    assertion = idp.requests[0]["client_assertion"]
    assert jwt.get_unverified_header(assertion)["alg"] == "RS256"
    assert jwt.decode(assertion, rsa_key.public_key(), algorithms=["RS256"], audience=TOKEN_URL)["sub"] == "h2ogpte"


@pytest.mark.parametrize("alg", ["HS256", "none", "es256"])
@pytest.mark.asyncio
async def test_non_asymmetric_or_misspelled_assertion_alg_is_rejected_before_any_request(alg):
    idp = _IdP()
    with pytest.raises(litellm.AuthenticationError, match="invalid h2o_oauth config: assertion_alg"):
        await _run(idp.hook(), h2o_oauth=_config(assertion_alg=alg))
    assert idp.requests == []


def _mint_cert(common_name: str, issuer: Optional[Tuple[rsa.RSAPrivateKey, x509.Certificate]] = None):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer[1].subject if issuer else name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
    )
    if issuer is None:
        builder = builder.add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
    else:
        builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
    return key, builder.sign(issuer[0] if issuer else key, hashes.SHA256())


def _write_pair(directory, stem: str, key, cert) -> Tuple[str, str]:
    cert_path, key_path = directory / f"{stem}.crt", directory / f"{stem}.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    return str(cert_path), str(key_path)


class _LoopbackHandler(BaseHTTPRequestHandler):
    seen: List[dict] = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0) or 0))
        peer = self.connection.getpeercert() if isinstance(self.connection, ssl.SSLSocket) else None
        _LoopbackHandler.seen.append(
            {
                "path": self.path,
                "authorization": self.headers.get("authorization"),
                "peer_cn": dict(p[0] for p in peer["subject"])["commonName"] if peer else None,
            }
        )
        if self.path.endswith("/token"):
            response = {"access_token": "gateway-jwt", "expires_in": 3600}
        else:
            response = {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
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


def _serve(context: Optional[ssl.SSLContext]) -> Tuple[_QuietServer, int]:
    server = _QuietServer(("127.0.0.1", 0), _LoopbackHandler)
    if context is not None:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.socket.getsockname()[1]


@pytest.fixture(scope="module")
def loopback(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("h2o-oauth-mtls")
    ca = _mint_cert("test-ca")
    ca_path = tmp / "ca.crt"
    ca_path.write_bytes(ca[1].public_bytes(serialization.Encoding.PEM))
    server_cert, server_key = _write_pair(tmp, "server", *_mint_cert("localhost", ca))
    client_cert, client_key = _write_pair(tmp, "client", *_mint_cert("h2ogpte-client", ca))

    mtls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    mtls.load_cert_chain(server_cert, server_key)
    mtls.load_verify_locations(str(ca_path))
    mtls.verify_mode = ssl.CERT_REQUIRED
    idp, idp_port = _serve(mtls)
    gateway, gateway_port = _serve(None)
    yield {
        "token_url": f"https://localhost:{idp_port}/token",
        "gateway": f"http://127.0.0.1:{gateway_port}",
        "ca": str(ca_path),
        "client_cert": client_cert,
        "client_key": client_key,
    }
    idp.shutdown()
    gateway.shutdown()


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def _fresh_state():
    _LoopbackHandler.seen.clear()
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    await close_litellm_async_clients()
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
async def test_token_endpoint_mtls_presents_the_configured_client_cert(loopback):
    hook = OAuthAuthHook()
    result = await _run(
        hook,
        h2o_oauth=_config(
            token_url=loopback["token_url"],
            ssl_verify=loopback["ca"],
            client_cert=loopback["client_cert"],
            client_key=loopback["client_key"],
        ),
    )
    assert result["api_key"] == "gateway-jwt"
    assert [r["peer_cn"] for r in _LoopbackHandler.seen] == ["h2ogpte-client"]


@pytest.mark.asyncio
async def test_token_endpoint_requiring_mtls_rejects_a_config_without_client_cert(loopback):
    with pytest.raises(litellm.AuthenticationError, match="could not obtain"):
        await _run(OAuthAuthHook(), h2o_oauth=_config(token_url=loopback["token_url"], ssl_verify=loopback["ca"]))
    assert _LoopbackHandler.seen == []


@pytest.mark.asyncio
async def test_router_sends_the_token_only_to_the_deployment_that_configured_it(loopback, monkeypatch):
    hook = OAuthAuthHook()
    monkeypatch.setattr(litellm, "callbacks", [hook])
    oauth = _config(
        token_url=loopback["token_url"],
        ssl_verify=loopback["ca"],
        client_cert=loopback["client_cert"],
        client_key=loopback["client_key"],
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {
                    "model": "openai/m",
                    "api_base": f"{loopback['gateway']}/oauth/v1",
                    "api_key": "unused",
                    "h2o_oauth": oauth,
                },
            },
            {
                "model_name": "group",
                "litellm_params": {
                    "model": "openai/m",
                    "api_base": f"{loopback['gateway']}/static/v1",
                    "api_key": "static-key",
                },
            },
        ]
    )
    for _ in range(12):
        await router.acompletion(model="group", messages=[{"role": "user", "content": "hi"}])

    gateway_calls = [r for r in _LoopbackHandler.seen if not r["path"].endswith("/token")]
    by_deployment = {
        prefix: {r["authorization"] for r in gateway_calls if r["path"].startswith(prefix)}
        for prefix in ("/oauth/", "/static/")
    }
    assert by_deployment == {"/oauth/": {"Bearer gateway-jwt"}, "/static/": {"Bearer static-key"}}
    assert len(_LoopbackHandler.seen) - len(gateway_calls) == 1


_MESSAGES = [{"role": "user", "content": "hi"}]
_AUDIO = ("clip.wav", b"RIFF0000WAVE", "audio/wav")
_UNAPPLIED_CALLS = {
    "completion": lambda kw: litellm.completion(messages=_MESSAGES, **kw),
    "embedding": lambda kw: litellm.embedding(input=["hi"], **kw),
    "responses": lambda kw: litellm.responses(input="hi", **kw),
    "transcription": lambda kw: litellm.transcription(file=_AUDIO, **kw),
    "acompletion": lambda kw: litellm.acompletion(messages=_MESSAGES, **kw),
    "aembedding": lambda kw: litellm.aembedding(input=["hi"], **kw),
    "aresponses": lambda kw: litellm.aresponses(input="hi", **kw),
    "anthropic_messages": lambda kw: litellm.anthropic_messages(messages=_MESSAGES, max_tokens=5, **kw),
    "atranscription": lambda kw: litellm.atranscription(file=_AUDIO, **kw),
}


def _unapplied_kwargs(loopback) -> dict:
    return {
        "model": "openai/m",
        "api_base": f"{loopback['gateway']}/oauth/v1",
        "api_key": "unused",
        "h2o_oauth": _config(token_url=loopback["token_url"]),
        "num_retries": 0,
    }


def _unapplied_call(loopback, call: str):
    return _UNAPPLIED_CALLS[call](_unapplied_kwargs(loopback))


@pytest.mark.parametrize("call", ["completion", "embedding", "transcription"])
def test_sync_call_with_h2o_oauth_fails_closed_instead_of_sending_the_static_api_key(loopback, monkeypatch, call):
    monkeypatch.setattr(litellm, "callbacks", [OAuthAuthHook()])
    with pytest.raises(litellm.AuthenticationError, match="OAuth hook did not run"):
        _unapplied_call(loopback, call)
    assert _LoopbackHandler.seen == []


@pytest.mark.parametrize("call, flag", [("completion", "acompletion"), ("embedding", "aembedding")])
def test_internal_async_flagged_sync_call_with_h2o_oauth_fails_closed(loopback, monkeypatch, call, flag):
    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj = Logging(
        model="openai/m",
        messages=_MESSAGES,
        stream=False,
        call_type=flag,
        start_time=datetime.datetime.now(),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    with pytest.raises(litellm.AuthenticationError, match="OAuth hook did not run"):
        _UNAPPLIED_CALLS[call]({**_unapplied_kwargs(loopback), flag: True, "litellm_logging_obj": logging_obj})
    assert _LoopbackHandler.seen == []


class _OtherDeploymentHook(CustomLogger):
    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        return None


@pytest.mark.parametrize("callbacks", [[], [_OtherDeploymentHook()]], ids=["no-hook", "other-deployment-hook"])
def test_sync_responses_with_h2o_oauth_but_no_oauth_hook_fails_closed(loopback, monkeypatch, callbacks):
    monkeypatch.setattr(litellm, "callbacks", callbacks)
    with pytest.raises(litellm.AuthenticationError, match="OAuth hook did not run"):
        _unapplied_call(loopback, "responses")
    assert _LoopbackHandler.seen == []


@pytest.mark.parametrize("call", ["acompletion", "aembedding", "aresponses", "anthropic_messages", "atranscription"])
@pytest.mark.asyncio
async def test_async_call_with_h2o_oauth_but_no_registered_hook_fails_closed(loopback, monkeypatch, call):
    monkeypatch.setattr(litellm, "callbacks", [])
    with pytest.raises(litellm.AuthenticationError, match="OAuth hook did not run"):
        await _unapplied_call(loopback, call)
    assert _LoopbackHandler.seen == []

"""
Per-deployment OAuth2 client_credentials (private_key_jwt) token, sent upstream as the deployment's api_key.

    litellm_params:
      model: openai/<model>
      api_base: https://gateway.example.com/v1
      api_key: unused
      client_cert: /certs/gateway/tls.crt
      client_key: /certs/gateway/tls.key
      ssl_verify: /certs/gateway/ca.crt
      h2o_oauth:
        token_url: https://idp.example.com/oauth2/token
        client_id: my-client
        client_private_key: os.environ/GATEWAY_SIGNING_KEY
        assertion_alg: ES256
        scope: optional-scope
        client_cert: /certs/idp/tls.crt
        client_key: /certs/idp/tls.key
        ssl_verify: /certs/idp/ca.crt

`client_private_key` is PEM text, `os.environ/NAME` or `file:///path`. Every TLS field and `scope` are optional
"""

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import httpx
import jwt
from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import get_client_cert_ssl_context, get_ssl_configuration
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import CallTypes

CONFIG_KEY: Final = "h2o_oauth"
REFRESH_BEFORE_EXPIRY_SEC: Final = 30.0
DEFAULT_EXPIRES_IN_SEC: Final = 300.0
ASSERTION_TTL_SEC: Final = 60
CLIENT_ASSERTION_TYPE: Final = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
CLIENT_CERT_PROVIDER: Final = "openai"
CLIENT_CERT_CALL_TYPES: Final = frozenset(
    (
        CallTypes.acompletion,
        CallTypes.aembedding,
        CallTypes.responses,
        CallTypes.aresponses,
        CallTypes.anthropic_messages,
    )
)
TOKEN_FETCH_ERRORS: Final = (httpx.HTTPError, jwt.PyJWTError, OSError, ValueError, TypeError)

AssertionAlg = Literal["ES256", "ES384", "ES512", "RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "EdDSA"]


class OAuthConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    token_url: str
    client_id: str
    client_private_key: str
    assertion_alg: AssertionAlg = "ES256"
    scope: str | None = None
    client_cert: str | None = None
    client_key: str | None = None
    ssl_verify: bool | str | None = None
    timeout: float = 30.0


class _TokenResponse(BaseModel):
    access_token: str
    expires_in: float | None = None


class _AssertionClaims(TypedDict):
    iss: ReadOnly[str]
    sub: ReadOnly[str]
    aud: ReadOnly[str]
    iat: ReadOnly[int]
    exp: ReadOnly[int]
    jti: ReadOnly[str]


@dataclass(frozen=True, slots=True)
class _Token:
    value: str
    refresh_at: float


def _resolve_secret(ref: str) -> str:
    if ref.startswith("os.environ/"):
        value: Final = get_secret_str(ref)
        if not value:
            raise ValueError(f"client_private_key references {ref}, which is unset")
        return value
    if ref.startswith("file://"):
        return Path(ref[len("file://") :]).read_text()
    return ref


def _refresh_at(now: float, expires_in: float | None) -> float:
    lifetime: Final = DEFAULT_EXPIRES_IN_SEC if expires_in is None else expires_in
    return now + max(lifetime - REFRESH_BEFORE_EXPIRY_SEC, lifetime / 2)


def _presents_client_cert(model: str, custom_llm_provider: object, call_type: CallTypes | None) -> bool:
    if call_type not in CLIENT_CERT_CALL_TYPES:
        return False
    try:
        _, provider, _, _ = litellm.get_llm_provider(
            model=model, custom_llm_provider=custom_llm_provider if isinstance(custom_llm_provider, str) else None
        )
    except litellm.BadRequestError:
        return False
    return provider == CLIENT_CERT_PROVIDER


def _describe_validation_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
        for err in error.errors(include_url=False, include_input=False)
    )


def _auth_error(message: str, model: str) -> litellm.AuthenticationError:
    return litellm.AuthenticationError(message=message, llm_provider="h2o_oauth", model=model)


class OAuthAuthHook(CustomLogger):
    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__()
        self._transport = transport
        self._clock = clock
        self._tokens: dict[str, _Token] = {}  # mutable-ok: per-process token cache, one entry per OAuth config
        self._inflight: dict[str, asyncio.Task[_Token]] = {}  # mutable-ok: single-flight registry of token fetches

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, object],  # mutable-ok: CustomLogger hook contract
        call_type: CallTypes | None,
    ) -> dict[str, object] | None:  # mutable-ok: CustomLogger hook contract returns the rewritten kwargs
        raw_config: Final = kwargs.get(CONFIG_KEY)
        if raw_config is None:
            return None
        model: Final = str(kwargs.get("model", ""))
        if kwargs.get("client_cert") and not _presents_client_cert(model, kwargs.get("custom_llm_provider"), call_type):
            raise _auth_error(
                "the deployment's client_cert is only presented on openai/ chat, embeddings, responses and "
                "/v1/messages calls, so the h2o_oauth token is not sent without it",
                model,
            )
        try:
            config: Final = OAuthConfig.model_validate(raw_config)
        except ValidationError as e:
            raise _auth_error(f"invalid h2o_oauth config: {_describe_validation_error(e)}", model) from None
        try:
            token: Final = await self._get_token(config)
        except ValidationError as e:
            raise _auth_error(f"unexpected token endpoint response: {_describe_validation_error(e)}", model) from None
        except TOKEN_FETCH_ERRORS as e:
            raise _auth_error(f"could not obtain an h2o_oauth token: {type(e).__name__}: {e}", model) from None
        return {  # mutable-ok: CustomLogger hook contract returns the rewritten kwargs
            k: v for k, v in (*kwargs.items(), ("api_key", token)) if k != CONFIG_KEY
        }

    async def _get_token(self, config: OAuthConfig) -> str:
        key: Final = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
        cached: Final = self._tokens.get(key)
        if cached is not None and self._clock() < cached.refresh_at:
            return cached.value
        inflight: Final = self._inflight.get(key)
        if inflight is not None and inflight.get_loop() is asyncio.get_running_loop():
            return (await asyncio.shield(inflight)).value
        task: Final = asyncio.ensure_future(self._refresh(key, config))
        self._inflight[key] = task
        return (await asyncio.shield(task)).value

    async def _refresh(self, key: str, config: OAuthConfig) -> _Token:
        try:
            token: Final = await self._fetch(config)
            self._tokens[key] = token
            return token
        finally:
            self._inflight.pop(key, None)

    async def _fetch(self, config: OAuthConfig) -> _Token:
        now: Final = self._clock()
        claims: Final[_AssertionClaims] = {
            "iss": config.client_id,
            "sub": config.client_id,
            "aud": config.token_url,
            "iat": int(now),
            "exp": int(now) + ASSERTION_TTL_SEC,
            "jti": str(uuid.uuid4()),
        }
        assertion: Final = jwt.encode(
            dict(claims),  # mutable-ok: PyJWT's encode takes a dict payload
            _resolve_secret(config.client_private_key),
            algorithm=config.assertion_alg,
        )
        form: Final = (
            ("grant_type", "client_credentials"),
            ("client_id", config.client_id),
            ("client_assertion_type", CLIENT_ASSERTION_TYPE),
            ("client_assertion", assertion),
            *((("scope", config.scope),) if config.scope else ()),
        )
        verify: Final = (
            get_client_cert_ssl_context(config.ssl_verify, config.client_cert, config.client_key)
            if config.client_cert
            else get_ssl_configuration(config.ssl_verify)
        )
        async with httpx.AsyncClient(verify=verify, timeout=config.timeout, transport=self._transport) as client:
            response: Final = await client.post(config.token_url, data=MappingProxyType(dict(form)))
        if response.status_code != 200:
            verbose_logger.debug("h2o_oauth token endpoint returned %s: %s", response.status_code, response.text[:200])
            raise ValueError(f"token endpoint returned {response.status_code}")
        body: Final = _TokenResponse.model_validate_json(response.content)
        return _Token(value=body.access_token, refresh_at=_refresh_at(now, body.expires_in))


oauth_auth_hook: Final = OAuthAuthHook()

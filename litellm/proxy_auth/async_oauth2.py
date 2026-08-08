"""
Async OAuth2 client-credentials credentials for OUTBOUND provider authentication.

WHY THIS EXISTS
---------------
`credentials.py` already provides `GenericOAuth2Credential` + `ProxyAuthHandler`,
but that pair cannot be used to authenticate individual model deployments:

  * it is wired through the module-global `litellm.proxy_auth`, so its
    Authorization header is applied to EVERY completion/embedding regardless of
    which deployment the request was routed to;
  * it fetches tokens with a blocking `httpx.post` on the event loop;
  * it has no single-flight guard, so every concurrent request racing an
    expiry mints its own token;
  * it only supports `client_secret` in the request body -- IdPs that require
    `private_key_jwt` (RFC 7523 client assertions) cannot be used;
  * it cannot present a client certificate to a mutual-TLS token endpoint.

This module closes those gaps. It is deliberately transport-only and stateless
with respect to litellm: nothing here reads `litellm.proxy_auth` or mutates
global state, so a credential is scoped to whatever caller owns it (see
`litellm/integrations/h2o/litellm_oauth_auth_hook.py`, which owns one credential
per model configuration).

SECRET HANDLING
---------------
Credential material is never written into configuration. Every secret-bearing
field accepts an indirection instead:

    os.environ/MY_VAR          -> read from the process environment
    file:///abs/path/to/secret -> read from a file (e.g. a mounted k8s secret)
    <literal>                  -> used as-is (test/dev only)

Resolution happens at token-mint time, so a rotated secret is picked up without
a restart, and the secret never enters request kwargs (and therefore can never
be echoed into an upstream request body).

Secret values are never logged and never included in exception messages.
"""

import asyncio
import base64
import hashlib
import json
import os
import ssl
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import certifi
import httpx

from litellm.litellm_core_utils.credential_ref import (
    CredentialRefError,
    credential_ref_to_file,
    resolve_credential_ref,
)

from litellm._logging import verbose_logger

from .credentials import AccessToken

DEFAULT_REFRESH_BUFFER_SEC = 30
DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_ASSERTION_TTL_SEC = 300
# Used only when the token endpoint returns neither `expires_in` nor a decodable
# `exp` claim. Deliberately short: a needless refresh is cheap, serving a dead
# token is not.
FALLBACK_TOKEN_TTL_SEC = 300

AUTH_STYLE_SECRET_POST = "client_secret_post"
AUTH_STYLE_SECRET_BASIC = "client_secret_basic"
AUTH_STYLE_PRIVATE_KEY_JWT = "private_key_jwt"
SUPPORTED_AUTH_STYLES = (
    AUTH_STYLE_SECRET_POST,
    AUTH_STYLE_SECRET_BASIC,
    AUTH_STYLE_PRIVATE_KEY_JWT,
)

SUPPORTED_ASSERTION_ALGS = (
    "ES256",
    "ES384",
    "ES512",
    "RS256",
    "RS384",
    "RS512",
    "PS256",
    "PS384",
    "PS512",
)

CLIENT_ASSERTION_TYPE_JWT_BEARER = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"


class OAuth2ConfigError(ValueError):
    """Raised when an OAuth2 configuration block is invalid.

    Raised at request time (config is validated lazily, when a model that
    declares it is first used) so a typo surfaces against the model that owns
    it rather than taking down proxy startup for unrelated models.
    """


class OAuth2TokenError(RuntimeError):
    """Raised when a token could not be obtained from the token endpoint."""


def resolve_secret_ref(value: Optional[str], *, field_name: str) -> str:
    """Resolve a secret indirection to its value, as an `h2o_oauth` field.

    Thin wrapper over the provider-neutral resolver: it only adds the
    `h2o_oauth.` prefix to the field name and translates the error type, so
    callers keep getting OAuth2ConfigError with a message that names the config
    key they wrote. Never includes the resolved value in raised messages.
    """
    try:
        return resolve_credential_ref(value, field_name=f"h2o_oauth.{field_name}")
    except CredentialRefError as e:
        raise OAuth2ConfigError(str(e)) from None


@contextmanager
def resolve_secret_ref_to_file(value: Optional[str], *, field_name: str):
    """Resolve a secret reference to a filesystem path usable by the ssl APIs.

    See litellm_core_utils.credential_ref.credential_ref_to_file: a mounted path
    is used as-is, PEM text is materialised into a mode-0600 file (memory-backed
    when available) for as long as OpenSSL needs to read it.
    """
    try:
        with credential_ref_to_file(value, field_name=f"h2o_oauth.{field_name}") as path:
            yield path
    except CredentialRefError as e:
        raise OAuth2ConfigError(str(e)) from None


def _require_str(raw: Dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None or not str(value).strip():
        raise OAuth2ConfigError(f"h2o_oauth.{key} is required")
    return str(value).strip()


def _optional_positive_number(raw: Dict[str, Any], key: str, default: Union[int, float]) -> Union[int, float]:
    if key not in raw or raw[key] is None:
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise OAuth2ConfigError(f"h2o_oauth.{key} must be a non-negative number, got {value!r}")
    return value


@dataclass
class OAuth2Config:
    """Validated OAuth2 client-credentials configuration for one deployment."""

    token_url: str
    client_id: str
    auth_style: str = AUTH_STYLE_SECRET_POST
    client_secret: Optional[str] = None
    client_private_key: Optional[str] = None
    assertion_alg: str = "ES256"
    audience: Optional[str] = None
    scope: Optional[str] = None
    refresh_buffer_sec: Union[int, float] = DEFAULT_REFRESH_BUFFER_SEC
    timeout_sec: Union[int, float] = DEFAULT_TIMEOUT_SEC
    assertion_ttl_sec: Union[int, float] = DEFAULT_ASSERTION_TTL_SEC
    header_name: str = "Authorization"
    header_scheme: str = "Bearer"
    ca_bundle: Optional[str] = None
    mtls_cert: Optional[str] = None
    mtls_key: Optional[str] = None
    insecure_skip_tls_verify: bool = False
    extra_token_params: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "OAuth2Config":
        if not isinstance(raw, dict):
            raise OAuth2ConfigError(f"h2o_oauth must be a mapping, got {type(raw).__name__}")

        # Validate the fundamentals first: when a block is largely empty,
        # "token_url is required" is a more useful first message than a
        # complaint about whichever credential the default auth_style wants.
        token_url = _require_str(raw, "token_url")
        client_id = _require_str(raw, "client_id")

        grant_type = str(raw.get("grant_type", "client_credentials")).strip()
        if grant_type != "client_credentials":
            raise OAuth2ConfigError(
                f"h2o_oauth.grant_type={grant_type!r} is not supported (only 'client_credentials')"
            )

        auth_style = str(raw.get("auth_style", AUTH_STYLE_SECRET_POST)).strip()
        if auth_style not in SUPPORTED_AUTH_STYLES:
            raise OAuth2ConfigError(
                f"h2o_oauth.auth_style={auth_style!r} is not supported "
                f"(expected one of {', '.join(SUPPORTED_AUTH_STYLES)})"
            )

        assertion_alg = str(raw.get("assertion_alg", "ES256")).strip().upper()
        if auth_style == AUTH_STYLE_PRIVATE_KEY_JWT and assertion_alg not in SUPPORTED_ASSERTION_ALGS:
            raise OAuth2ConfigError(
                f"h2o_oauth.assertion_alg={assertion_alg!r} is not supported "
                f"(expected one of {', '.join(SUPPORTED_ASSERTION_ALGS)})"
            )

        if auth_style == AUTH_STYLE_PRIVATE_KEY_JWT:
            if not raw.get("client_private_key"):
                raise OAuth2ConfigError(
                    "h2o_oauth.client_private_key is required when auth_style='private_key_jwt' "
                    "(use 'os.environ/VAR' or 'file:///path' -- never the key itself)"
                )
        elif not raw.get("client_secret"):
            raise OAuth2ConfigError(f"h2o_oauth.client_secret is required when auth_style={auth_style!r}")

        mtls_cert = raw.get("mtls_cert")
        mtls_key = raw.get("mtls_key")
        # httpx accepts either a (cert, key) pair or a single combined PEM path, so
        # mtls_cert alone is valid. mtls_key alone is not -- a key with no
        # certificate can never complete a handshake, and httpx would silently
        # ignore it, so reject it here with an actionable message.
        if mtls_key and not mtls_cert:
            raise OAuth2ConfigError(
                "h2o_oauth.mtls_key is set without h2o_oauth.mtls_cert -- set both, "
                "or set only mtls_cert if it points at a combined PEM containing the key"
            )

        extra = raw.get("extra_token_params") or {}
        if not isinstance(extra, dict):
            raise OAuth2ConfigError("h2o_oauth.extra_token_params must be a mapping")

        # `or` would fold an explicitly empty scheme back to "Bearer", making a
        # raw (unprefixed) token impossible to configure -- distinguish absent
        # from empty.
        raw_header_name = raw.get("header_name")
        header_name = "Authorization" if raw_header_name is None else str(raw_header_name).strip()
        if not header_name:
            raise OAuth2ConfigError("h2o_oauth.header_name cannot be empty")
        raw_header_scheme = raw.get("header_scheme")
        header_scheme = "Bearer" if raw_header_scheme is None else str(raw_header_scheme).strip()

        insecure = bool(raw.get("insecure_skip_tls_verify", False))
        if insecure:
            verbose_logger.warning(
                "h2o_oauth: insecure_skip_tls_verify=true -- TLS verification of the token "
                "endpoint is DISABLED. Use only for bring-up, never in production."
            )

        return cls(
            token_url=token_url,
            client_id=client_id,
            auth_style=auth_style,
            client_secret=raw.get("client_secret"),
            client_private_key=raw.get("client_private_key"),
            assertion_alg=assertion_alg,
            audience=(str(raw["audience"]).strip() if raw.get("audience") else None),
            scope=(str(raw["scope"]).strip() if raw.get("scope") else None),
            refresh_buffer_sec=_optional_positive_number(raw, "refresh_buffer_sec", DEFAULT_REFRESH_BUFFER_SEC),
            timeout_sec=_optional_positive_number(raw, "timeout_sec", DEFAULT_TIMEOUT_SEC),
            assertion_ttl_sec=_optional_positive_number(raw, "assertion_ttl_sec", DEFAULT_ASSERTION_TTL_SEC),
            header_name=header_name,
            header_scheme=header_scheme,
            ca_bundle=(str(raw["ca_bundle"]).strip() if raw.get("ca_bundle") else None),
            mtls_cert=(str(mtls_cert).strip() if mtls_cert else None),
            mtls_key=(str(mtls_key).strip() if mtls_key else None),
            insecure_skip_tls_verify=insecure,
            extra_token_params={str(k): str(v) for k, v in extra.items()},
        )

    def fingerprint(self) -> str:
        """Stable identity for credential reuse.

        Hashes the configuration as written (secret *references*, not resolved
        secrets), so two deployments sharing one IdP client share one token,
        and editing the config yields a new credential with a cold cache.
        """
        payload = json.dumps(
            {
                "token_url": self.token_url,
                "client_id": self.client_id,
                "auth_style": self.auth_style,
                "client_secret_ref": self.client_secret,
                "client_private_key_ref": self.client_private_key,
                "assertion_alg": self.assertion_alg,
                "audience": self.audience,
                "scope": self.scope,
                "ca_bundle": self.ca_bundle,
                "mtls_cert": self.mtls_cert,
                "mtls_key": self.mtls_key,
                "insecure_skip_tls_verify": self.insecure_skip_tls_verify,
                "extra_token_params": self.extra_token_params,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _decode_jwt_exp(token: str) -> Optional[int]:
    """Best-effort read of the `exp` claim WITHOUT verifying the signature.

    Only used to derive a cache TTL when the token endpoint omits `expires_in`.
    The token is not trusted for authorization here -- it is forwarded to the
    upstream gateway, which does the verifying.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


class AsyncOAuth2ClientCredential:
    """Async OAuth2 client-credentials token source with caching + single flight.

    One instance owns the token for one configuration. `get_token()` is safe to
    call from arbitrarily many concurrent requests: exactly one of them performs
    the network round trip while the rest await the same result.
    """

    def __init__(self, config: OAuth2Config):
        self.config = config
        self._token: Optional[AccessToken] = None
        # (error, monotonic-ish timestamp) of the most recent mint failure, so a
        # burst of requests during an IdP outage shares one failure instead of
        # each issuing its own token request. Cleared on the next success.
        self._recent_failure: Optional[Tuple[Exception, float]] = None
        # Created lazily: constructing an asyncio.Lock outside a running loop is
        # legal but binds awkwardly if the owner is built at import time.
        self._lock: Optional[asyncio.Lock] = None

    # Short enough that recovery is prompt, long enough to collapse a burst.
    NEGATIVE_CACHE_SEC = 5.0

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _is_fresh(self, token: Optional[AccessToken]) -> bool:
        if token is None:
            return False
        return token.expires_on > time.time() + self.config.refresh_buffer_sec

    async def get_token(self, *, force_refresh: bool = False) -> AccessToken:
        if not force_refresh and self._is_fresh(self._token):
            return self._token  # type: ignore[return-value]

        async with self._get_lock():
            # Re-check: another coroutine may have refreshed while we waited.
            if not force_refresh and self._is_fresh(self._token):
                return self._token  # type: ignore[return-value]
            # Negative cache. Without it the single-flight collapses only on
            # SUCCESS: on failure self._token stays stale, so every waiter takes
            # the lock in turn and issues its own request. Measured against a
            # transport that sleeps 0.3s then 503s: 20 concurrent requests
            # produced 20 token requests over 6.0s strictly serialised, versus 1
            # request in 0.30s on the success path. With timeout_sec defaulting to
            # 30 that is 600s of queueing per burst -- a self-inflicted DoS on the
            # proxy that also hammers the IdP exactly when it is degraded.
            if not force_refresh and self._recent_failure is not None:
                error, when = self._recent_failure
                if time.time() - when < self.NEGATIVE_CACHE_SEC:
                    raise error
            try:
                self._token = await self._fetch_token()
            except (OAuth2ConfigError, OAuth2TokenError) as e:
                self._recent_failure = (e, time.time())
                raise
            self._recent_failure = None
            return self._token

    def invalidate(self) -> None:
        """Drop the cached token so the next get_token() re-mints.

        Used when the upstream rejects a token we believed was still valid
        (clock skew, server-side revocation).
        """
        self._token = None
        # Also clear the negative cache: an explicit invalidate is a request to
        # try again, and leaving a recent failure cached would refuse for up to
        # NEGATIVE_CACHE_SEC.
        self._recent_failure = None

    def _client_assertion(self) -> str:
        import jwt  # PyJWT -- imported lazily so this module loads without it

        private_key = resolve_secret_ref(self.config.client_private_key, field_name="client_private_key")
        now = int(time.time())
        audience = self.config.audience or self.config.token_url
        claims = {
            "iss": self.config.client_id,
            "sub": self.config.client_id,
            "aud": audience,
            "iat": now,
            "exp": now + int(self.config.assertion_ttl_sec),
            "jti": str(uuid.uuid4()),
        }
        try:
            return jwt.encode(claims, private_key, algorithm=self.config.assertion_alg)
        except Exception as e:
            # Message may echo key parsing detail -- keep only the exception type.
            raise OAuth2ConfigError(
                f"failed to sign client_assertion with alg={self.config.assertion_alg} "
                f"({type(e).__name__}) -- check that client_private_key is a PEM private key "
                f"matching that algorithm"
            ) from None

    def _request_parts(self) -> Tuple[Dict[str, str], Optional[Tuple[str, str]]]:
        """Build the token request form data and optional HTTP basic auth."""
        data: Dict[str, str] = {"grant_type": "client_credentials"}
        if self.config.scope:
            data["scope"] = self.config.scope
        data.update(self.config.extra_token_params)

        basic_auth: Optional[Tuple[str, str]] = None

        if self.config.auth_style == AUTH_STYLE_PRIVATE_KEY_JWT:
            data["client_id"] = self.config.client_id
            data["client_assertion"] = self._client_assertion()
            data["client_assertion_type"] = CLIENT_ASSERTION_TYPE_JWT_BEARER
        elif self.config.auth_style == AUTH_STYLE_SECRET_BASIC:
            secret = resolve_secret_ref(self.config.client_secret, field_name="client_secret")
            basic_auth = (self.config.client_id, secret)
        else:  # client_secret_post
            data["client_id"] = self.config.client_id
            data["client_secret"] = resolve_secret_ref(self.config.client_secret, field_name="client_secret")

        return data, basic_auth

    def _build_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Build the TLS context for the token request, or None to use httpx's default.

        WHY WE BUILD THE CONTEXT INSTEAD OF PASSING httpx's `verify=`/`cert=`:
        httpx's create_ssl_context() RETURNS EARLY in its `isinstance(verify, str)`
        branch, before the `if cert:` block that would call load_cert_chain(). So
        the combination this feature exists for -- a custom CA bundle AND a client
        certificate -- silently drops the client certificate, and the handshake
        fails with a bare `TLSV13_ALERT_CERTIFICATE_REQUIRED` that points nowhere
        near the real cause. (Verified against httpx 0.28.1.) Both kwargs are also
        deprecated in favour of `verify=<ssl_context>`, so doing it ourselves is
        both correct today and forward-compatible.

        The context is built per token request, not cached: minting is rare (once
        per token lifetime), and re-reading means a rotated certificate is picked
        up without a restart.
        """
        if not (self.config.ca_bundle or self.config.mtls_cert or self.config.insecure_skip_tls_verify):
            return None

        if self.config.insecure_skip_tls_verify:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        else:
            cafile = self.config.ca_bundle or certifi.where()
            try:
                with resolve_secret_ref_to_file(cafile, field_name="ca_bundle") as ca_path:
                    context = ssl.create_default_context(cafile=ca_path)
            except OSError as e:
                raise OAuth2ConfigError(
                    f"h2o_oauth.ca_bundle could not be loaded: {e.strerror}"
                ) from None

        if self.config.mtls_cert:
            try:
                with resolve_secret_ref_to_file(self.config.mtls_cert, field_name="mtls_cert") as cert_path:
                    if self.config.mtls_key:
                        with resolve_secret_ref_to_file(self.config.mtls_key, field_name="mtls_key") as key_path:
                            context.load_cert_chain(cert_path, key_path)
                    else:
                        context.load_cert_chain(cert_path)
            except (OSError, ssl.SSLError) as e:
                raise OAuth2ConfigError(
                    f"h2o_oauth.mtls_cert could not be loaded ({type(e).__name__}) -- check the path, "
                    f"the PEM contents, and that "
                    f"mtls_key matches the certificate"
                ) from None

        return context

    async def _fetch_token(self) -> AccessToken:
        data, basic_auth = self._request_parts()
        started = time.time()

        client_kwargs: Dict[str, Any] = {"timeout": httpx.Timeout(self.config.timeout_sec)}
        ssl_context = self._build_ssl_context()
        if ssl_context is not None:
            client_kwargs["verify"] = ssl_context

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(self.config.token_url, data=data, auth=basic_auth)
        except (httpx.HTTPError, ssl.SSLError) as e:
            raise OAuth2TokenError(
                f"token request to {self.config.token_url} failed: {type(e).__name__}: {e}"
            ) from None

        if response.status_code != 200:
            # Response bodies from token endpoints carry error codes, not secrets,
            # but cap the length so a stray HTML error page can't flood logs.
            raise OAuth2TokenError(
                f"token endpoint {self.config.token_url} returned "
                f"{response.status_code}: {response.text[:512]}"
            )

        try:
            payload = response.json()
        except ValueError:
            raise OAuth2TokenError(
                f"token endpoint {self.config.token_url} returned a non-JSON body "
                f"(content-type={response.headers.get('content-type')!r})"
            ) from None

        access_token = payload.get("access_token")
        if not access_token or not isinstance(access_token, str):
            raise OAuth2TokenError(
                f"token endpoint {self.config.token_url} returned no 'access_token' "
                f"(keys: {sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__})"
            )

        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool) and expires_in > 0:
            expires_on = int(time.time() + expires_in)
            ttl_source = "expires_in"
        else:
            exp_claim = _decode_jwt_exp(access_token)
            if exp_claim is not None and exp_claim > time.time():
                expires_on = exp_claim
                ttl_source = "jwt-exp"
            else:
                expires_on = int(time.time() + FALLBACK_TOKEN_TTL_SEC)
                ttl_source = "fallback"

        verbose_logger.info(
            "h2o_oauth: minted token client_id=%s token_url=%s ttl=%ss (%s) in %.0fms",
            self.config.client_id,
            self.config.token_url,
            max(0, int(expires_on - time.time())),
            ttl_source,
            (time.time() - started) * 1000,
        )

        return AccessToken(token=access_token, expires_on=expires_on)

    def auth_header(self, token: AccessToken) -> Tuple[str, str]:
        """Header name/value pair carrying `token`."""
        scheme = self.config.header_scheme
        value = f"{scheme} {token.token}".strip() if scheme else token.token
        return self.config.header_name, value


class CredentialRegistry:
    """Fingerprint-keyed credential cache.

    Deployments sharing one IdP client share one token (and one refresh), while
    a configuration edit produces a new fingerprint and therefore a cold cache.
    """

    def __init__(self) -> None:
        self._credentials: Dict[str, AsyncOAuth2ClientCredential] = {}

    def get(self, config: OAuth2Config) -> AsyncOAuth2ClientCredential:
        key = config.fingerprint()
        credential = self._credentials.get(key)
        if credential is None:
            credential = AsyncOAuth2ClientCredential(config)
            self._credentials[key] = credential
        return credential

    def clear(self) -> None:
        self._credentials.clear()

    def __len__(self) -> int:
        return len(self._credentials)

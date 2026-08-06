"""
Common helpers / utils across al OpenAI endpoints
"""

import hashlib
import inspect
import json
import os
import ssl
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import certifi
import httpx
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

if TYPE_CHECKING:
    from aiohttp import ClientSession

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    _DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    AsyncHTTPHandler,
    get_ssl_configuration,
)
from litellm.litellm_core_utils.credential_ref import credential_ref_to_file


def _resolve_ssl_config(
    ssl_verify: Optional[Union[bool, str, ssl.SSLContext]] = None,
    client_cert: Optional[Union[str, Tuple[str, str]]] = None,
) -> Union[bool, str, ssl.SSLContext]:
    """Resolve the TLS configuration for an OpenAI-provider httpx client.

    With no client certificate this is exactly `get_ssl_configuration(ssl_verify)`,
    so behaviour is unchanged for every deployment that configures none.

    With a client certificate we must build a PRIVATE ssl.SSLContext rather than
    load the chain into the one `get_ssl_configuration` returns: those contexts
    are CACHED and shared across deployments (keyed only by CA file + security
    level + curve), so calling load_cert_chain() on a shared context would
    present that deployment's mTLS identity on every other model resolving to the
    same cache key. That is a cross-model credential leak, not a perf detail.
    """
    if not client_cert:
        return get_ssl_configuration(ssl_verify)

    if ssl_verify is False:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    elif isinstance(ssl_verify, ssl.SSLContext):
        # Caller-supplied context: they own it, including its cert chain.
        context = ssl_verify
    else:
        ca_ref: Optional[str] = None
        if isinstance(ssl_verify, str):
            ca_ref = ssl_verify
        elif os.getenv("SSL_CERT_FILE") and os.path.exists(os.environ["SSL_CERT_FILE"]):
            ca_ref = os.environ["SSL_CERT_FILE"]
        with credential_ref_to_file(ca_ref or certifi.where(), field_name="ssl_verify") as ca_path:
            context = ssl.create_default_context(cafile=ca_path)

    if isinstance(client_cert, (tuple, list)):
        with credential_ref_to_file(client_cert[0], field_name="client_cert") as cert_path:
            with credential_ref_to_file(client_cert[1], field_name="client_key") as key_path:
                context.load_cert_chain(cert_path, key_path)
    else:
        with credential_ref_to_file(client_cert, field_name="client_cert") as cert_path:
            context.load_cert_chain(cert_path)
    return context


def _get_client_init_params(cls: type) -> Tuple[str, ...]:
    """Extract __init__ parameter names (excluding 'self') from a class."""
    return tuple(p for p in inspect.signature(cls.__init__).parameters if p != "self")  # type: ignore[misc]


_OPENAI_INIT_PARAMS: Tuple[str, ...] = _get_client_init_params(OpenAI)
_AZURE_OPENAI_INIT_PARAMS: Tuple[str, ...] = _get_client_init_params(AzureOpenAI)


class OpenAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[dict, httpx.Headers]] = None,
        body: Optional[dict] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers
        if request:
            self.request = request
        else:
            self.request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        if response:
            self.response = response
        else:
            self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=self.message,
            headers=self.headers,
            request=self.request,
            response=self.response,
            body=body,
        )


####### Error Handling Utils for OpenAI API #######################
###################################################################
def drop_params_from_unprocessable_entity_error(
    e: Union[openai.UnprocessableEntityError, httpx.HTTPStatusError],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Helper function to read OpenAI UnprocessableEntityError and drop the params that raised an error from the error message.

    Args:
    e (UnprocessableEntityError): The UnprocessableEntityError exception
    data (Dict[str, Any]): The original data dictionary containing all parameters

    Returns:
    Dict[str, Any]: A new dictionary with invalid parameters removed
    """
    invalid_params: List[str] = []
    if isinstance(e, httpx.HTTPStatusError):
        error_json = e.response.json()
        error_message = error_json.get("error", {})
        error_body = error_message
    else:
        error_body = e.body
    if error_body is not None and isinstance(error_body, dict) and error_body.get("message"):
        message = error_body.get("message", {})
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                message = {"detail": message}
        detail = message.get("detail")

        if isinstance(detail, List) and len(detail) > 0 and isinstance(detail[0], dict):
            for error_dict in detail:
                if (
                    error_dict.get("loc")
                    and isinstance(error_dict.get("loc"), list)
                    and len(error_dict.get("loc")) == 2
                ):
                    invalid_params.append(error_dict["loc"][1])

    new_data = {k: v for k, v in data.items() if k not in invalid_params}

    return new_data


class BaseOpenAILLM:
    """
    Base class for OpenAI LLMs for getting their httpx clients and SSL verification settings
    """

    @staticmethod
    def get_cached_openai_client(
        client_initialization_params: dict, client_type: Literal["openai", "azure"]
    ) -> Optional[Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]]:
        """Retrieves the OpenAI client from the in-memory cache based on the client initialization parameters"""
        _cache_key = BaseOpenAILLM.get_openai_client_cache_key(
            client_initialization_params=client_initialization_params,
            client_type=client_type,
        )
        _cached_client = litellm.in_memory_llm_clients_cache.get_cache(_cache_key)
        return _cached_client

    @staticmethod
    def set_cached_openai_client(
        openai_client: Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI],
        client_type: Literal["openai", "azure"],
        client_initialization_params: dict,
    ):
        """Stores the OpenAI client in the in-memory cache for _DEFAULT_TTL_FOR_HTTPX_CLIENTS SECONDS"""
        _cache_key = BaseOpenAILLM.get_openai_client_cache_key(
            client_initialization_params=client_initialization_params,
            client_type=client_type,
        )
        litellm.in_memory_llm_clients_cache.set_cache(
            key=_cache_key,
            value=openai_client,
            ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
        )

    @staticmethod
    def tls_client_kwargs(litellm_params: Optional[dict]) -> Dict[str, Any]:
        """Per-deployment TLS transport config, ready to splat into _get_openai_client.

        Reads three optional litellm_params:

            ssl_verify  -- bool | path to a CA bundle | ssl.SSLContext
            client_cert -- path to a client certificate (or a combined PEM)
            client_key  -- path to the client private key, when not in client_cert

        Returns both keys unconditionally with None values when unset, so a
        deployment that configures none of them produces the same client as
        before this existed.
        """
        if not litellm_params:
            return {"ssl_verify": None, "client_cert": None}

        client_cert = litellm_params.get("client_cert")
        client_key = litellm_params.get("client_key")
        cert: Optional[Union[str, Tuple[str, str]]]
        if client_cert and client_key:
            cert = (client_cert, client_key)
        elif client_cert:
            # A lone cert is valid: a combined PEM carrying the private key.
            cert = client_cert
        else:
            cert = None

        return {"ssl_verify": litellm_params.get("ssl_verify"), "client_cert": cert}

    @staticmethod
    def get_openai_client_cache_key(client_initialization_params: dict, client_type: Literal["openai", "azure"]) -> str:
        """Creates a cache key for the OpenAI client based on the client initialization parameters"""
        hashed_api_key = None
        if client_initialization_params.get("api_key") is not None:
            hash_object = hashlib.sha256(client_initialization_params.get("api_key", "").encode())
            # Hexadecimal representation of the hash
            hashed_api_key = hash_object.hexdigest()

        # Create a more readable cache key using a list of key-value pairs
        key_parts = [
            f"hashed_api_key={hashed_api_key}",
            f"is_async={client_initialization_params.get('is_async')}",
        ]

        LITELLM_CLIENT_SPECIFIC_PARAMS = (
            "timeout",
            "max_retries",
            "organization",
            "api_base",
            # TLS transport config MUST take part in the cache key: two deployments
            # differing only by client certificate (or CA bundle) must not share a
            # cached client, or one model's mTLS identity would be presented for
            # another's requests. These are not OpenAI SDK __init__ params, so
            # _OPENAI_INIT_PARAMS (derived from that signature) does not cover them.
            "ssl_verify",
            "client_cert",
        )
        openai_client_fields = (
            BaseOpenAILLM.get_openai_client_initialization_param_fields(client_type=client_type)
            + LITELLM_CLIENT_SPECIFIC_PARAMS
        )

        for param in openai_client_fields:
            key_parts.append(f"{param}={client_initialization_params.get(param)}")

        _cache_key = ",".join(key_parts)
        return _cache_key

    @staticmethod
    def get_openai_client_initialization_param_fields(
        client_type: Literal["openai", "azure"],
    ) -> Tuple[str, ...]:
        """Returns a tuple of fields that are used to initialize the OpenAI client"""
        if client_type == "openai":
            return _OPENAI_INIT_PARAMS
        else:
            return _AZURE_OPENAI_INIT_PARAMS

    @staticmethod
    def _get_async_http_client(
        shared_session: Optional["ClientSession"] = None,
        ssl_verify: Optional[Union[bool, str, ssl.SSLContext]] = None,
        client_cert: Optional[Union[str, Tuple[str, str]]] = None,
    ) -> Optional[httpx.AsyncClient]:
        """Build the httpx client for OpenAI-provider routes.

        `ssl_verify` and `client_cert` come from the deployment's litellm_params
        (see openai.py). Both default to None, in which case this behaves exactly
        as before: get_ssl_configuration(None) IS the previous no-argument call,
        and httpx treats cert=None as "no client certificate" -- so a deployment
        that configures neither gets a byte-identical client.

        litellm already accepted a per-model `ssl_verify` litellm_param, but this
        call site dropped it (it called get_ssl_configuration() with no arguments),
        so a private CA on an openai/-prefixed endpoint was silently ignored.
        Threading it through fixes that; `client_cert` adds the mutual-TLS leg,
        which this path never supported at all.

        Deliberately NOT wired here: the global SSL_CERTIFICATE / litellm.ssl_certificate
        fallback that the sibling HTTPHandler.create_client honours. Reading it
        would change behaviour for existing deployments that set it globally --
        they would start presenting a client certificate on openai routes without
        asking. Client certificates here are opt-in per deployment only.
        """
        if litellm.aclient_session is not None:
            return litellm.aclient_session

        if getattr(litellm, "network_mock", False):
            from litellm.llms.custom_httpx.mock_transport import MockOpenAITransport

            return httpx.AsyncClient(transport=MockOpenAITransport())

        # Get unified SSL configuration. NOTE: when an explicit `transport` is
        # passed, httpx IGNORES the client-level verify=/cert= arguments -- the
        # transport performs the handshake -- so the client certificate has to be
        # carried by the ssl_context handed to the transport below, not by cert=.
        ssl_config = _resolve_ssl_config(ssl_verify, client_cert)

        return httpx.AsyncClient(
            verify=ssl_config,
            transport=AsyncHTTPHandler._create_async_transport(
                ssl_context=(ssl_config if isinstance(ssl_config, ssl.SSLContext) else None),
                ssl_verify=ssl_config if isinstance(ssl_config, bool) else None,
                shared_session=shared_session,
            ),
            follow_redirects=True,
        )

    @staticmethod
    def _get_sync_http_client(
        ssl_verify: Optional[Union[bool, str, ssl.SSLContext]] = None,
        client_cert: Optional[Union[str, Tuple[str, str]]] = None,
    ) -> Optional[httpx.Client]:
        """Sync counterpart of _get_async_http_client -- same defaults, same guarantees."""
        if litellm.client_session is not None:
            return litellm.client_session

        if getattr(litellm, "network_mock", False):
            from litellm.llms.custom_httpx.mock_transport import MockOpenAITransport

            return httpx.Client(transport=MockOpenAITransport())

        # Get unified SSL configuration (carries the client certificate when one
        # is configured -- see _resolve_ssl_config).
        ssl_config = _resolve_ssl_config(ssl_verify, client_cert)

        return httpx.Client(
            verify=ssl_config,
            follow_redirects=True,
        )


class OpenAICredentials(NamedTuple):
    api_base: str
    api_key: Optional[str]
    organization: Optional[str]


def get_openai_credentials(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
) -> OpenAICredentials:
    """Resolve OpenAI credentials from params, litellm globals, and env vars."""
    resolved_api_base = (
        api_base
        or litellm.api_base
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )
    resolved_organization = organization or litellm.organization or os.getenv("OPENAI_ORGANIZATION", None) or None
    resolved_api_key = api_key or litellm.api_key or litellm.openai_key or os.getenv("OPENAI_API_KEY")
    return OpenAICredentials(
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        organization=resolved_organization,
    )

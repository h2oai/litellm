"""
Proxy Authentication module for LiteLLM SDK.

This module provides OAuth2/JWT token management for authenticating
with LiteLLM Proxy or any OAuth2-protected endpoint.

Usage:
    from litellm.proxy_auth import AzureADCredential, ProxyAuthHandler

    litellm.proxy_auth = ProxyAuthHandler(
        credential=AzureADCredential(),
        scope="api://my-proxy/.default"
    )
"""

from .credentials import (
    AccessToken,
    TokenCredential,
    AzureADCredential,
    GenericOAuth2Credential,
    ProxyAuthHandler,
)
from .async_oauth2 import (
    AsyncOAuth2ClientCredential,
    CredentialRegistry,
    OAuth2Config,
    OAuth2ConfigError,
    OAuth2TokenError,
    resolve_secret_ref,
)

__all__ = [
    "AccessToken",
    "TokenCredential",
    "AzureADCredential",
    "GenericOAuth2Credential",
    "ProxyAuthHandler",
    # Async, per-deployment client-credentials support (private_key_jwt + mTLS).
    # Unlike ProxyAuthHandler these are NOT wired to the litellm.proxy_auth
    # global -- the caller owns the credential and decides which request it
    # applies to.
    "AsyncOAuth2ClientCredential",
    "CredentialRegistry",
    "OAuth2Config",
    "OAuth2ConfigError",
    "OAuth2TokenError",
    "resolve_secret_ref",
]

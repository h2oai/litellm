"""Resolve credential references, and materialise them for APIs that need paths.

Provider-neutral on purpose. Both the OpenAI-provider TLS path
(`llms/openai/common_utils.py`) and the OAuth2 credential
(`proxy_auth/async_oauth2.py`) need this, and neither should have to import the
other -- in particular the openai provider path must not depend on an auth module.

Two things live here:

  resolve_credential_ref()  -- os.environ/NAME | file:///path | literal
  credential_ref_to_file()  -- the same, but yielding a filesystem PATH, because
                               ssl.SSLContext.load_cert_chain() and
                               ssl.create_default_context(cafile=...) accept only
                               paths, while operators may supply PEM text through
                               an environment variable.

Resolved values are never included in raised messages: the caller names the
FIELD, never its contents.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

ENV_PREFIX = "os.environ/"
FILE_PREFIX = "file://"

# PEM preamble, used to tell "this string IS the credential" from "this string is
# a path to it".
_PEM_MARKER = "-----BEGIN "

# Prefer a memory-backed directory when materialising key material, so a private
# key supplied as PEM text never lands on a container's disk layer (in most images
# /tmp is the overlay filesystem, not a tmpfs). Falls back to the system default
# when none is available: the file is mode 0600 and unlinked immediately either
# way, but memory-backed means a hard kill between write and unlink cannot leave
# key material behind on disk.
_PREFERRED_TMPDIRS = ("/dev/shm",)


class CredentialRefError(ValueError):
    """Raised when a credential reference cannot be resolved."""


def resolve_credential_ref(value: Optional[str], *, field_name: str) -> str:
    """Resolve `os.environ/NAME`, `file:///abs/path`, or a literal value.

    Never includes the resolved value (or any part of it) in raised messages.
    """
    if value is None or not str(value).strip():
        raise CredentialRefError(f"{field_name} is required but empty")

    raw = str(value).strip()

    if raw.startswith(ENV_PREFIX):
        env_name = raw[len(ENV_PREFIX) :].strip()
        if not env_name:
            raise CredentialRefError(f"{field_name}: '{ENV_PREFIX}' given with no variable name")
        resolved = os.environ.get(env_name)
        if resolved is None or not resolved.strip():
            raise CredentialRefError(
                f"{field_name} references environment variable '{env_name}', "
                f"which is unset or empty in this process"
            )
        return resolved

    if raw.startswith(FILE_PREFIX):
        path = raw[len(FILE_PREFIX) :]
        if not path.startswith("/"):
            raise CredentialRefError(
                f"{field_name}: file reference must be absolute, "
                f"e.g. 'file:///etc/credentials/{field_name}'"
            )
        try:
            with open(path, "r") as f:
                contents = f.read()
        except OSError as e:
            raise CredentialRefError(
                f"{field_name} references file '{path}' which could not be read: {e.strerror}"
            ) from None
        if not contents.strip():
            raise CredentialRefError(f"{field_name} references file '{path}', which is empty")
        return contents

    return raw


def _materialisation_dir() -> Optional[str]:
    for candidate in _PREFERRED_TMPDIRS:
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    return None


@contextmanager
def credential_ref_to_file(value: Optional[str], *, field_name: str) -> Iterator[str]:
    """Yield a filesystem path for `value`, materialising PEM text when needed.

    A path that exists is yielded as-is and NOTHING is written -- so mounting the
    credential as a file keeps it off any writable filesystem. PEM text is written
    to a mode-0600 temporary file that exists only long enough for OpenSSL to read
    it into memory.

    Note the asymmetry, which is easy to misread: a `file:///path` reference
    resolves to the file's CONTENTS (that is what resolve_credential_ref does), so
    it takes the materialisation path. Only a bare path avoids it.
    """
    resolved = resolve_credential_ref(value, field_name=field_name)

    if os.path.exists(resolved):
        yield resolved
        return

    # Tolerate leading/trailing whitespace: environment variables and YAML block
    # scalars very commonly carry a leading newline, and without stripping, PEM
    # text would be mistaken for a path and produce a confusing ssl error.
    candidate = resolved.strip()
    if not candidate.startswith(_PEM_MARKER):
        # Neither an existing path nor PEM text -- yield it so the ssl API raises
        # its own path-oriented error, which the caller wraps without echoing the
        # value.
        yield resolved
        return

    tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=_materialisation_dir())
    try:
        tmp.write(candidate + "\n")
        tmp.flush()
        tmp.close()
        yield tmp.name
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass

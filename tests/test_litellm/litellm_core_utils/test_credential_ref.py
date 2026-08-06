"""Tests for litellm_core_utils.credential_ref."""

import os
import stat
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.credential_ref import (  # noqa: E402
    CredentialRefError,
    _materialisation_dir,
    credential_ref_to_file,
    resolve_credential_ref,
)

PEM = "-----BEGIN PRIVATE KEY-----\nMIIBOgIBAAJBAK\n-----END PRIVATE KEY-----"


# --------------------------------------------------------------------------
# resolve_credential_ref
# --------------------------------------------------------------------------


def test_env_reference():
    with patch.dict(os.environ, {"MY_CRED": "value"}):
        assert resolve_credential_ref("os.environ/MY_CRED", field_name="client_cert") == "value"


def test_missing_env_names_the_variable_not_the_value():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(CredentialRefError) as exc:
            resolve_credential_ref("os.environ/ABSENT", field_name="client_cert")
    assert "ABSENT" in str(exc.value)
    assert "client_cert" in str(exc.value)


def test_file_reference_returns_contents(tmp_path):
    f = tmp_path / "cred.pem"
    f.write_text(PEM)
    assert resolve_credential_ref(f"file://{f}", field_name="client_cert") == PEM


def test_file_reference_must_be_absolute():
    with pytest.raises(CredentialRefError, match="must be absolute"):
        resolve_credential_ref("file://relative", field_name="client_cert")


def test_literal_and_empty():
    assert resolve_credential_ref("/etc/tls/ca.pem", field_name="ssl_verify") == "/etc/tls/ca.pem"
    with pytest.raises(CredentialRefError, match="required but empty"):
        resolve_credential_ref("  ", field_name="ssl_verify")


# --------------------------------------------------------------------------
# credential_ref_to_file
# --------------------------------------------------------------------------


def test_existing_path_is_yielded_without_writing_anything(tmp_path):
    f = tmp_path / "ca.pem"
    f.write_text(PEM)
    with credential_ref_to_file(str(f), field_name="ssl_verify") as path:
        assert path == str(f), "a mounted file must be used in place, never copied"


def test_pem_text_is_materialised_then_removed():
    with patch.dict(os.environ, {"CERT_PEM": PEM}):
        with credential_ref_to_file("os.environ/CERT_PEM", field_name="client_cert") as path:
            assert os.path.exists(path)
            assert open(path).read().startswith("-----BEGIN ")
            leaked = path
    assert not os.path.exists(leaked), "materialised key material must not outlive the context"


def test_materialised_file_is_not_readable_by_others():
    with patch.dict(os.environ, {"CERT_PEM": PEM}):
        with credential_ref_to_file("os.environ/CERT_PEM", field_name="client_cert") as path:
            mode = stat.S_IMODE(os.stat(path).st_mode)
    assert not (mode & 0o077), f"expected 0600-style mode, got {oct(mode)}"


def test_pem_with_leading_whitespace_is_still_recognised():
    """Environment variables and YAML block scalars routinely carry a leading
    newline. Without tolerating it, PEM text is mistaken for a path and OpenSSL
    raises a confusing error naming a nonexistent file."""
    with patch.dict(os.environ, {"CERT_PEM": "\n  " + PEM + "\n\n"}):
        with credential_ref_to_file("os.environ/CERT_PEM", field_name="client_cert") as path:
            body = open(path).read()
    assert body.startswith("-----BEGIN ")
    assert body.rstrip().endswith("-----END PRIVATE KEY-----")


def test_materialisation_prefers_a_memory_backed_dir():
    preferred = _materialisation_dir()
    if preferred is None:
        pytest.skip("no memory-backed tmpdir available on this host")
    with patch.dict(os.environ, {"CERT_PEM": PEM}):
        with credential_ref_to_file("os.environ/CERT_PEM", field_name="client_cert") as path:
            assert os.path.dirname(path) == preferred


def test_non_pem_non_path_is_passed_through():
    """Left to the ssl API so it raises its own path-oriented error, rather than
    this layer guessing and echoing the value."""
    with credential_ref_to_file("/nonexistent/ca.pem", field_name="ssl_verify") as path:
        assert path == "/nonexistent/ca.pem"

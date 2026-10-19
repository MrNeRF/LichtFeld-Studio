# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Credential storage contracts independent of the portal/network client."""

import os

import pytest

from lfs_plugins.credential_storage import CredentialStorage, DPAPIBackend


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI backend")
def test_windows_credentials_remain_encrypted_and_roundtrip(tmp_path):
    path = tmp_path / "credentials.json"
    storage = CredentialStorage(path)
    value = b'{"access_token":"test-only-secret"}'
    storage.write(value)
    assert isinstance(storage.backend, DPAPIBackend)
    assert not path.exists()
    assert value not in path.with_suffix(".dpapi").read_bytes()
    assert storage.read() == value
    storage.delete()
    assert storage.read() is None


def test_storage_failure_does_not_fall_back_to_plaintext(tmp_path):
    class FailingBackend:
        def write(self, value):
            raise PermissionError("storage denied")

    path = tmp_path / "credentials.json"
    storage = CredentialStorage(path, backend=FailingBackend())
    with pytest.raises(PermissionError):
        storage.write(b"test-only-secret")
    assert not path.exists()
    assert not path.with_suffix(".migrated").exists()

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Private directory contracts, including real Windows UAC token transitions."""
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import uuid

import pytest

from lfs_plugins.private_directory import mkdir_private


def test_private_directory_parents_and_existing_contents(tmp_path):
    path = tmp_path / "parent" / "private"
    mkdir_private(path, parents=True)
    marker = path / "marker"
    marker.write_text("preserve")
    mkdir_private(path, exist_ok=True)
    assert marker.read_text() == "preserve"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        mkdir_private(path)
    with pytest.raises(FileExistsError):
        mkdir_private(marker, exist_ok=True)


def test_private_directory_requires_parent(tmp_path):
    with pytest.raises(FileNotFoundError):
        mkdir_private(tmp_path / "missing" / "private")


@pytest.mark.skipif(os.name != "nt", reason="Windows private DACL")
def test_windows_private_acl_grants_user_and_preserves_existing_acl(tmp_path):
    import ctypes
    from ctypes import wintypes

    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security.GetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                        ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD)]
    security.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.DWORD)]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p

    def dacl(path):
        size = wintypes.DWORD()
        security.GetFileSecurityW(str(path), 4, None, 0, ctypes.byref(size))
        assert size.value
        descriptor = ctypes.create_string_buffer(size.value)
        if not security.GetFileSecurityW(str(path), 4, descriptor, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        text = ctypes.c_void_p()
        if not security.ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor, 1, 4, ctypes.byref(text), None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.wstring_at(text)
        finally:
            kernel.LocalFree(text)

    path = tmp_path / "private"
    mkdir_private(path)
    before = dacl(path)
    assert before.startswith("D:P")
    assert before.count("(A;OICI;FA;;;") == 3
    assert ";;;SY)" in before and ";;;BA)" in before
    assert ";;;S-1-" in before  # Explicit user SID, rather than OWNER RIGHTS.
    mkdir_private(path, exist_ok=True)
    assert dacl(path) == before
    marker = path / "credentials.lock"
    marker.write_bytes(b"0")
    assert ";;;S-1-" in dacl(marker)


@contextmanager
def _filtered_user(*, require_elevated=True):
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    desktop = ctypes.WinDLL("user32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    desktop.GetShellWindow.argtypes = []
    desktop.GetShellWindow.restype = wintypes.HWND
    desktop.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                         ctypes.POINTER(wintypes.HANDLE)]
    security.OpenThreadToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL,
                                        ctypes.POINTER(wintypes.HANDLE)]
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    security.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                            ctypes.c_void_p, wintypes.DWORD,
                                            ctypes.POINTER(wintypes.DWORD)]
    security.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    security.ImpersonateLoggedOnUser.argtypes = [wintypes.HANDLE]

    def token_dword(handle, kind):
        value, size = wintypes.DWORD(), wintypes.DWORD()
        if not security.GetTokenInformation(handle, kind, ctypes.byref(value),
                                            ctypes.sizeof(value), ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.value

    def token_user(handle):
        size = wintypes.DWORD()
        security.GetTokenInformation(handle, 1, None, 0, ctypes.byref(size))
        if not size.value:
            raise ctypes.WinError(ctypes.get_last_error())
        value = ctypes.create_string_buffer(size.value)
        if not security.GetTokenInformation(handle, 1, value, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    token, normal, thread = wintypes.HANDLE(), wintypes.HANDLE(), wintypes.HANDLE()
    process = None
    impersonating = False
    try:
        if not security.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        if require_elevated and token_dword(token, 18) != 2:  # TokenElevationTypeFull
            if os.environ.get("LFS_TEST_REQUIRE_UAC") == "1":
                pytest.fail("Run this UAC integration test from an elevated user prompt")
            pytest.skip("Requires an elevated UAC user and the same user's normal desktop")

        # TokenLinkedToken can expose only an identification-level token, which
        # cannot be promoted to impersonation (WinError 1346). Use the desktop's
        # real primary token, checking identity, session and limited elevation.
        window = desktop.GetShellWindow()
        if not window:
            raise RuntimeError("UAC test requires the current user's normal desktop shell")
        pid = wintypes.DWORD()
        if not desktop.GetWindowThreadProcessId(window, ctypes.byref(pid)):
            raise ctypes.WinError(ctypes.get_last_error())
        process = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        if not security.OpenProcessToken(process, 0x000A, ctypes.byref(normal)):
            raise ctypes.WinError(ctypes.get_last_error())  # TOKEN_QUERY | TOKEN_DUPLICATE
        caller_user, desktop_user = token_user(token), token_user(normal)
        caller_sid = ctypes.cast(caller_user, ctypes.POINTER(ctypes.c_void_p))[0]
        desktop_sid = ctypes.cast(desktop_user, ctypes.POINTER(ctypes.c_void_p))[0]
        assert security.EqualSid(caller_sid, desktop_sid), "Desktop must belong to the same user"
        assert token_dword(normal, 12) == token_dword(token, 12), "Desktop must be in the same session"
        assert token_dword(normal, 8) == 1, "Desktop must supply a primary token"
        assert token_dword(normal, 18) == 3, "Desktop must have a normal, limited UAC token"
        if not security.ImpersonateLoggedOnUser(normal):
            raise ctypes.WinError(ctypes.get_last_error())
        impersonating = True
        if not security.OpenThreadToken(kernel.GetCurrentThread(), 0x0008, True, ctypes.byref(thread)):
            raise ctypes.WinError(ctypes.get_last_error())
        assert token_dword(thread, 18) == 3, "File operations must use the normal token"
        yield
    finally:
        if impersonating and not security.RevertToSelf():
            # Continuing elevated tests while impersonating would invalidate their results.
            os._exit(1)
        for handle in (thread, normal, token, process):
            if handle:
                kernel.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows UAC and DPAPI integration")
def test_elevated_and_normal_tokens_share_encrypted_credentials_and_lock():
    from lfs_plugins.credential_storage import CredentialStorage
    from lfs_plugins.portal_account import _locked_sidecar

    # Use a workspace parent whose ACL allows the real user to traverse it under
    # either token; pytest's elevated temporary root may itself be owner-only.
    build = Path(__file__).resolve().parents[2] / "build-windows-release"
    root = build / ("verify-uac-" + uuid.uuid4().hex)
    mkdir_private(root, parents=True)
    try:
        legacy = root / "legacy"
        legacy.mkdir(mode=0o700)
        (legacy / "probe").write_bytes(b"dummy")
        path = root / "account" / "credentials.json"
        storage = CredentialStorage(path)
        elevated_value = b'{"access_token":"dummy-elevated"}'
        normal_value = b'{"access_token":"dummy-normal"}'
        storage.write(elevated_value)
        lock = path.with_suffix(path.suffix + ".lock")
        with _locked_sidecar(lock):
            pass
        with _filtered_user():
            # Negative control: reproduce the original permission failure.
            with pytest.raises(PermissionError):
                (legacy / "probe").read_bytes()
            with _locked_sidecar(lock):
                assert storage.read() == elevated_value
                storage.write(normal_value)
            reverse = root / "normal-created"
            mkdir_private(reverse)
            (reverse / "probe").write_bytes(b"dummy-normal")
        with _locked_sidecar(lock):
            assert storage.read() == normal_value
        assert (reverse / "probe").read_bytes() == b"dummy-normal"
        assert not path.exists()
        ciphertext = path.with_suffix(".dpapi").read_bytes()
        assert normal_value not in ciphertext
    finally:
        shutil.rmtree(root)


@pytest.mark.skipif(os.name != "nt", reason="Windows UAC workspace")
def test_uac_workspace_without_build_directory_reaches_skip_and_cleans_up(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    build = checkout / "build-windows-release"
    assert not build.exists()
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "__file__", str(checkout / "tests/python/test_private_directory.py"))

    def unavailable_normal_token():
        pytest.skip("UAC prerequisite unavailable")

    monkeypatch.setattr(module, "_filtered_user", unavailable_normal_token)
    with pytest.raises(pytest.skip.Exception, match="UAC prerequisite unavailable"):
        test_elevated_and_normal_tokens_share_encrypted_credentials_and_lock()
    assert build.is_dir()
    assert list(build.iterdir()) == []

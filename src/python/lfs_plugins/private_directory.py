# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Create private directories accessible across same-user Windows elevation."""
from __future__ import annotations

import os
from pathlib import Path


def _windows_create(path: Path) -> None:
    import ctypes
    from ctypes import wintypes

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("descriptor", ctypes.c_void_p),
                    ("inherit", wintypes.BOOL)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                         ctypes.POINTER(wintypes.HANDLE)]
    security.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                            ctypes.c_void_p, wintypes.DWORD,
                                            ctypes.POINTER(wintypes.DWORD)]
    security.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_void_p)]
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.DWORD)]
    kernel.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(SecurityAttributes)]

    token = wintypes.HANDLE()
    sid_text = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    try:
        if not security.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        size = wintypes.DWORD()
        security.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))  # TokenUser
        if not size.value:
            raise ctypes.WinError(ctypes.get_last_error())
        user = ctypes.create_string_buffer(size.value)
        if not security.GetTokenInformation(token, 1, user, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        sid = ctypes.cast(user, ctypes.POINTER(ctypes.c_void_p))[0]
        if not security.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        # TokenUser remains the same under UAC; TokenOwner can become Administrators.
        # Protect the DACL from inherited broad permissions; files inherit these grants.
        sddl = "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;%s)" % ctypes.wstring_at(sid_text)
        if not security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, 1, ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        if not kernel.CreateDirectoryW(str(path), ctypes.byref(attributes)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if descriptor:
            kernel.LocalFree(descriptor)
        if sid_text:
            kernel.LocalFree(sid_text)
        if token:
            kernel.CloseHandle(token)


def mkdir_private(path, *, parents=False, exist_ok=False) -> None:
    """Secure only newly created directories; never rewrite existing directory ACLs."""
    path = Path(path)
    if os.name != "nt":
        path.mkdir(mode=0o700, parents=parents, exist_ok=exist_ok)
        return
    try:
        _windows_create(path)
    except FileNotFoundError:
        if not parents or path.parent == path:
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        mkdir_private(path, exist_ok=exist_ok)
    except FileExistsError:
        if not exist_ok or not path.is_dir():
            raise

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""User-scoped portal credential storage. Backend failures never downgrade encryption."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Protocol


class CredentialBackend(Protocol):
    def read(self) -> bytes | None: ...
    def write(self, value: bytes) -> None: ...
    def delete(self) -> None: ...


class FileBackend:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        try:
            with self.path.open('rb') as source:
                value = source.read(1024 * 1024 + 1)
            if len(value) > 1024 * 1024:
                raise OSError('Credential storage exceeds its size limit')
            if os.name != 'nt':
                self.path.chmod(0o600)
            return value
        except FileNotFoundError:
            return None

    def write(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as output:
                temporary = Path(output.name)
                output.write(value)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    def delete(self):
        self.path.unlink(missing_ok=True)


def _dpapi(value, *, protect):
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_ubyte))]

    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    operation = crypt.CryptProtectData if protect else crypt.CryptUnprotectData
    operation.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.POINTER(Blob),
                          ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    operation.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    incoming, outgoing = Blob(len(value), buffer), Blob()
    # User scope: deliberately omit CRYPTPROTECT_LOCAL_MACHINE (4).
    if not operation(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)):
        raise OSError('Windows credential protection failed')
    try:
        return ctypes.string_at(outgoing.pbData, outgoing.cbData)
    finally:
        kernel.LocalFree(outgoing.pbData)


class DPAPIBackend(FileBackend):
    def read(self):
        value = super().read()
        return _dpapi(value, protect=False) if value is not None else None

    def write(self, value):
        super().write(_dpapi(value, protect=True))


class KeychainBackend:
    def __init__(self, path, executable):
        self.executable = executable
        self.account = hashlib.sha256(str(Path(path).absolute()).encode()).hexdigest()
        self.service = 'io.lichtfeld.studio.portal'

    def _run(self, *args, input=None):
        try:
            return subprocess.run([self.executable, *args], input=input, capture_output=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            raise OSError('Keychain storage is unavailable') from None

    def read(self):
        result = self._run('find-generic-password', '-s', self.service, '-a', self.account, '-w')
        if result.returncode == 44:  # errSecItemNotFound
            return None
        if result.returncode:
            raise OSError('Keychain storage is unavailable')
        try:
            return base64.b64decode(result.stdout.strip(), validate=True)
        except ValueError:
            raise OSError('Keychain credential format is invalid') from None

    def write(self, value):
        # Interactive commands travel through stdin, never process arguments.
        encoded = base64.b64encode(value).decode('ascii')
        command = f'add-generic-password -U -s {self.service} -a {self.account} -w {encoded}\n'
        if self._run('-i', input=command.encode('ascii')).returncode:
            raise OSError('Keychain did not save the credentials')
        if self.read() != value:
            raise OSError('Keychain did not save the credentials')

    def delete(self):
        result = self._run('delete-generic-password', '-s', self.service, '-a', self.account)
        if result.returncode not in (0, 44):
            raise OSError('Keychain credentials could not be removed')


def default_backend(path):
    system = platform.system()
    if system == 'Windows':
        return DPAPIBackend(Path(path).with_suffix('.dpapi'))
    if system == 'Darwin':
        executable = shutil.which('security')
        if executable or Path(path).with_suffix('.migrated').exists():
            return KeychainBackend(path, executable or '/usr/bin/security')
    return FileBackend(path)


class CredentialStorage:
    def __init__(self, path, backend=None):
        self.legacy = FileBackend(path)
        self.migrated = FileBackend(Path(path).with_suffix(".migrated"))
        self.backend = backend if backend is not None else default_backend(path)
        self.plaintext = type(self.backend) is FileBackend and self.backend.path == self.legacy.path

    def read(self):
        if self.plaintext:
            return self.backend.read()
        # A receipt is written only after an equal readback. Even if the secure
        # item later becomes unreadable, a leftover plaintext copy must go.
        if self.migrated.read() is not None:
            self.legacy.delete()
            return self.backend.read()
        value = self.backend.read()
        legacy = self.legacy.read()
        if legacy is not None:
            # A pre-existing secure item may belong to an older account.
            self.write(legacy)
            return legacy
        return value

    def write(self, value):
        self.backend.write(value)
        if self.backend.read() != value:
            raise OSError('Credential migration could not be verified')
        if not self.plaintext:
            self.migrated.write(b'verified\n')
            self.legacy.delete()

    def delete(self):
        errors = []
        backends = [self.backend, self.legacy, FileBackend(self.legacy.path.with_suffix('.dpapi')), self.migrated]
        if platform.system() == 'Darwin' and not isinstance(self.backend, KeychainBackend):
            # The system binary remains addressable when a restricted PATH hid it.
            backends.append(KeychainBackend(self.legacy.path, shutil.which('security') or '/usr/bin/security'))
        for backend in backends:
            try:
                backend.delete()
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise OSError('Local portal credentials could not be fully removed') from None

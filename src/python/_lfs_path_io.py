# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Logical application paths with extended Windows names at Python I/O boundaries.

No registry changes or monkeypatches. str(Path) stays suitable for UI and JSON;
os.fspath(Path) supplies the native representation to filesystem consumers.
"""
import builtins
import hashlib
import ntpath
import os
import pathlib
import sys
import tempfile as _tempfile


def logical_path(value):
    if os.name != "nt" or not isinstance(value, str):
        return value
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\") and len(value) >= 7 and value[5] == ":":
        return value[4:]
    return value


def _dos_device_name(value):
    stem = value.split(".", 1)[0].rstrip(" .").upper()
    return stem in {"CON", "PRN", "AUX", "NUL"} or (len(stem) == 4 and stem[:3] in {"COM", "LPT"} and stem[3] in "123456789\u00b9\u00b2\u00b3")


def _needs_verbatim(value):
    _, tail = ntpath.splitdrive(value)
    return any(part not in {"", ".", ".."} and (part.endswith((".", " ")) or _dos_device_name(part)) for part in tail.replace("/", "\\").split("\\"))


def native_path(value, *, force_extended=False):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return value  # open also accepts a file descriptor or a file-like object.
    name = os.fspath(value)
    if os.name != "nt" or not name:
        return name
    name = os.fsdecode(name)
    if name.startswith(("\\\\?\\", "\\\\.\\")):
        return name
    bare_device = not ntpath.dirname(name) and not ntpath.splitext(name)[1] and _dos_device_name(name)
    if bare_device and not force_extended:
        return name
    verbatim = force_extended or _needs_verbatim(name)
    drive, tail = ntpath.splitdrive(name)
    if drive and (tail.startswith(("\\", "/")) or drive.startswith("\\\\")):
        absolute = name
    elif drive:
        absolute = ntpath.join(logical_path(ntpath.abspath(drive + ".")), tail)
    else:
        absolute = ntpath.join(logical_path(os.getcwd()), name)
    if not verbatim and not _needs_verbatim(ntpath.dirname(absolute)) and len(absolute.encode("utf-16-le", errors="surrogatepass")) // 2 < 248:
        return name
    absolute = ntpath.normpath(absolute)
    return "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute


if os.name != "nt":
    Path = pathlib.Path
else:
    class Path(pathlib.WindowsPath):
        def __new__(cls, *parts, **kwargs):
            parts = tuple(logical_path(str(p) if isinstance(p, pathlib.PurePath) else os.fspath(p)) for p in parts)
            return super().__new__(cls, *parts, **kwargs)

        def __init__(self, *parts, **kwargs):
            if sys.version_info >= (3, 12):
                parts = tuple(logical_path(str(p) if isinstance(p, pathlib.PurePath) else os.fspath(p)) for p in parts)
                super().__init__(*parts, **kwargs)

        def __fspath__(self):
            return native_path(str(self))

        def resolve(self, strict=False):
            return type(self)(logical_path(str(super().resolve(strict=strict))))

        def rename(self, target):
            os.rename(self, native_path(target))
            return type(self)(target)

        def replace(self, target):
            os.replace(self, native_path(target))
            return type(self)(target)

        def samefile(self, other):
            return super().samefile(type(self)(other))

        def hardlink_to(self, target):
            return super().hardlink_to(type(self)(target))


def open(file, *args, **kwargs):
    return builtins.open(native_path(file), *args, **kwargs)


def walk(top, *args, **kwargs):
    for root, directories, files in os.walk(native_path(top, force_extended=True), *args, **kwargs):
        yield logical_path(root), directories, files


class TemporaryDirectory(_tempfile.TemporaryDirectory):
    @classmethod
    def _rmtree(cls, name, *args, **kwargs):
        return super()._rmtree(native_path(name, force_extended=True), *args, **kwargs)


def _component_units(name):
    return len(name.encode("utf-16-le", errors="surrogatepass")) // 2 if os.name == "nt" else len(os.fsencode(name))


def bounded_sibling_path(source, suffix):
    source = Path(source)
    name = source.name
    if _component_units(name + suffix) > 255:
        name = ".lfs-artifact-" + hashlib.sha256(name.encode("utf-8", errors="surrogatepass")).hexdigest()[:32]
    return source.with_name(name + suffix)


def temporary_prefix(source):
    source = Path(source)
    prefix = source.stem + "."
    # tempfile's random component plus .tmp; keep room for implementation changes.
    if _component_units(prefix) > 220:
        prefix = ".lfs-temp-" + hashlib.sha256(source.name.encode("utf-8", errors="surrogatepass")).hexdigest()[:32] + "."
    return prefix


def executable_kwargs(command):
    """Adapt an explicit executable filename without changing argv or PATH lookup."""
    if os.name == "nt" and command:
        executable = os.fsdecode(os.fspath(command[0]))
        if ntpath.dirname(executable):
            return {"executable": native_path(executable)}
    return {}

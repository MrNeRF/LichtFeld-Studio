# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Immutable local snapshots. No application imports; shared by both clients."""
import ctypes
from contextlib import contextmanager
import hashlib
from functools import lru_cache
import json
import math
import mmap
import os
import stat
from _lfs_path_io import Path, native_path
import threading
import time
import uuid

PROTOCOL = 1
TERMINAL = frozenset({"completed", "stopped", "failed", "superseded"})
FIELDS = ("means", "scale", "rotation", "opacity", "sh")
MAX_BYTES = 2 * 1024 ** 3
MAX_METADATA_CHARS = 262144
MAX_DOCUMENT_CHARS = 32 * 1024 * 1024


@lru_cache(maxsize=1)
def _kernel32():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateFileW": ([ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p,
                         ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p], ctypes.c_void_p),
        "SetFileInformationByHandle": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_ulong], ctypes.c_int),
        "OpenProcess": ([ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong], ctypes.c_void_p),
        "GetExitCodeProcess": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)], ctypes.c_int),
        "GetProcessTimes": ([ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4, ctypes.c_int),
        "OpenFileMappingW": ([ctypes.c_ulong, ctypes.c_int, ctypes.c_wchar_p], ctypes.c_void_p),
        "MapViewOfFile": ([ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
                           ctypes.c_size_t], ctypes.c_void_p),
        "UnmapViewOfFile": ([ctypes.c_void_p], ctypes.c_int),
        "CloseHandle": ([ctypes.c_void_p], ctypes.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes, function.restype = arguments, result
    return kernel


def replace_metadata(source, destination):
    if os.name != "nt":
        return os.replace(source, destination)
    kernel = _kernel32()
    name = native_path(Path(destination).resolve())
    encoded_name = name.encode("utf-16-le")
    class RenameInfo(ctypes.Structure):
        _fields_ = [("flags", ctypes.c_ulong), ("root", ctypes.c_void_p),
                    ("length", ctypes.c_ulong), ("name", ctypes.c_ubyte * (len(encoded_name) + 2))]
    info = RenameInfo()
    info.flags = 3  # REPLACE_IF_EXISTS | POSIX_SEMANTICS (Windows 10+).
    info.length = len(encoded_name)
    ctypes.memmove(ctypes.addressof(info) + RenameInfo.name.offset, encoded_name, info.length)
    handle = kernel.CreateFileW(native_path(Path(source).resolve()), 0x10000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise PermissionError(ctypes.get_last_error(), "Cannot publish metadata", str(source))
    try:
        success = kernel.SetFileInformationByHandle(handle, 22, ctypes.byref(info),
                                                    RenameInfo.name.offset + info.length)
        error = ctypes.get_last_error()
    finally:
        kernel.CloseHandle(handle)
    if not success:
        if error in (1, 50, 87):
            return os.replace(native_path(source), native_path(destination))
        raise PermissionError(error, "Cannot atomically publish metadata", str(destination))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False))
        for attempt in range(8):
            try:
                replace_metadata(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.005 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def bounded_metadata(stream, max_chars=MAX_METADATA_CHARS):
    if type(max_chars) is not int or not 0 < max_chars <= MAX_DOCUMENT_CHARS:
        raise ValueError("Invalid JSON document size limit")
    text = stream.read(max_chars + 1)
    if len(text) > max_chars:
        raise ValueError(f"Training JSON document exceeds its {max_chars:,}-character size limit")
    return text


def read_text_shared(path, *, max_chars=MAX_METADATA_CHARS):
    if os.name != "nt":
        with Path(path).open("r", encoding="utf-8") as stream:
            return bounded_metadata(stream, max_chars)
    import msvcrt
    kernel = _kernel32()
    # Readers must allow rename/delete so atomic publication can replace metadata.
    handle = kernel.CreateFileW(native_path(Path(path).resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        code = ctypes.get_last_error()
        if code in (2, 3):
            raise FileNotFoundError(path)
        raise PermissionError(code, "Cannot read session metadata", str(path))
    descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        return bounded_metadata(stream, max_chars)


def read_json(path, default=None):
    try:
        return json.loads(read_text_shared(path))
    except (OSError, ValueError, UnicodeError, RecursionError):
        return default


def registry_root():
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "LichtFeld" / "training-sessions"


def process_alive(pid):
    if type(pid) is not int or not 0 < pid <= 0xffffffff:
        return False
    if os.name == "nt":
        kernel = _kernel32()
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_identity(pid):
    """OS process creation token; distinguishes PID reuse without another dependency."""
    if type(pid) is not int or not 0 < pid <= 0xffffffff:
        return None
    if os.name == "nt":
        kernel = _kernel32()
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            values = [ctypes.c_ulonglong() for _ in range(4)]
            if kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
                return str(values[0].value)
        finally:
            kernel.CloseHandle(handle)
    else:
        try:
            # The command field can itself contain spaces and parentheses.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return fields[19]
        except (OSError, IndexError):
            return None
    return None


def session_liveness(record):
    if not process_alive(record.get("pid")):
        return "exited"
    token = record.get("process_identity")
    if token is not None:
        current = process_identity(record["pid"])
        if current is None:
            return "unresponsive"
        if current != token:
            return "exited"
    heartbeat = record.get("heartbeat", 0)
    if (type(heartbeat) not in (int, float) or not 0 <= heartbeat <= 2**63 - 1
            or not 0 <= time.time() - heartbeat < 10):
        return "unresponsive"
    return "responsive"


def validate_session(record, expected_id=None):
    if not isinstance(record, dict) or not record:
        raise ValueError("Training session metadata is unavailable")
    if type(record.get("protocol")) is not int or record["protocol"] != PROTOCOL:
        raise ValueError("Incompatible training-session protocol. Use matching LichtFeld versions.")
    if (not isinstance(record.get("id"), str) or not record["id"]
            or type(record.get("pid")) is not int or not 0 < record["pid"] <= 0xffffffff):
        raise ValueError("Invalid training session identity")
    if expected_id is not None and record["id"] != expected_id:
        raise ValueError("Training session identity changed; select the trainer again")
    for key in ("iteration", "iterations", "count"):
        if key in record and (type(record[key]) is not int or not 0 <= record[key] <= 2**63 - 1):
            raise ValueError("Invalid training session counter: " + key)
    if "label" in record and (not isinstance(record["label"], str) or len(record["label"]) > 1024):
        raise ValueError("Invalid training session label")
    # Optional result metadata is validated by its consumer. A damaged artifact
    # must not hide terminal status or keep an observer's lease alive.
    return record


def validate_result(result):
    """Validate result metadata without filesystem access or numerical imports."""
    if not isinstance(result, dict) or result.get("kind") not in ("best", "last"):
        raise ValueError("No saved training model is available")
    for key in ("root", "path"):
        if not isinstance(result.get(key), str) or not 0 < len(result[key]) <= 32768:
            raise ValueError("Invalid saved model path")
    if not Path(result["root"]).is_absolute() or Path(result["path"]).is_absolute():
        raise ValueError("Saved model paths require an absolute root and relative artifact")
    if Path(result["path"]).suffix.lower() != ".ply":
        raise ValueError("Unsupported saved model format")
    for key, minimum, maximum in (("bytes", 1, 2**63 - 1), ("count", 1, 2**31 - 1),
                                  ("iteration", 0, 2**63 - 1), ("sh_degree", 0, 3)):
        if type(result.get(key)) is not int or not minimum <= result[key] <= maximum:
            raise ValueError("Invalid saved model metadata: " + key)
    checksum = result.get("sha256")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Invalid saved model checksum")
    value = result.get("presentation")
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 1:
        raise ValueError("Invalid saved model presentation")
    return result


def artifact_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Artifact path escapes its output directory")
    return path


def digest_stream(stream, should_cancel=lambda: False):
    digest = hashlib.sha256()
    while True:
        if should_cancel():
            raise InterruptedError("Artifact verification cancelled")
        block = stream.read(4 * 1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


@contextmanager
def verified_artifact(path, record, should_cancel=lambda: False):
    """Yield the verified stream at its start; decoding reuses this same handle."""
    path = Path(path)
    if should_cancel():
        raise InterruptedError("Artifact verification cancelled")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Artifact must be a regular file")
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size != record["bytes"]:
            raise ValueError("Artifact size changed")
        if digest_stream(stream, should_cancel) != record["sha256"]:
            raise ValueError("Artifact integrity check failed")
        stream.seek(0)
        yield stream


def resolve_result(record, should_cancel=lambda: False):
    """Verify a finished local trainer's artifact in place on a worker."""
    if not isinstance(record, dict) or record.get("state") not in TERMINAL:
        raise ValueError("The trainer has not finished saving its result")
    result = validate_result(record.get("result"))
    root, relative = Path(result["root"]), Path(result["path"])
    path = artifact_path(root, relative)
    from . import geometry
    presentation = geometry.validate(result.get("presentation"))
    with verified_artifact(path, result, should_cancel):
        pass
    return path, presentation


def prune_session(directory, record, max_age=86400):
    """Remove only protocol-owned temporary files of old, dead sessions."""
    stamp = record.get("heartbeat")
    if (not isinstance(stamp, (int, float)) or time.time() - stamp < max_age
            or process_alive(record.get("pid"))):
        return
    directory = Path(directory)
    try:
        for subdir, pattern in (("clients", "*.json"), ("buffers", "lfs_snapshot_*.bin")):
            parent = directory / subdir
            if parent.is_symlink():
                continue
            for path in parent.glob(pattern):
                if path.is_file() and not path.is_symlink():
                    path.unlink(missing_ok=True)
            if parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    pass
        geometry = directory / "view_geometry.json"
        if geometry.is_file() and not geometry.is_symlink():
            geometry.unlink(missing_ok=True)
        (directory / "session.json").unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass  # Never remove unrelated files.
    except OSError:
        pass


def discover():
    def order(item):
        return item.get("started", 0), item["id"]

    latest = {}
    for path in registry_root().glob("*/session.json"):
        if path.parent.is_symlink() or path.is_symlink():
            continue
        record = read_json(path, {})
        try:
            validate_session(record, path.parent.name)
        except ValueError:
            continue
        prune_session(path.parent, record)
        record["directory"] = str(path.parent)
        record["liveness"] = session_liveness(record)
        record["alive"] = record["liveness"] == "responsive"
        record["active"] = record["alive"] and record.get("state") not in TERMINAL
        if record["liveness"] != "responsive":
            continue
        # A LichtFeld process owns one current training run. A newer terminal
        # registration also supersedes earlier registrations from that process.
        previous = latest.get(record["pid"])
        if previous is None or order(record) > order(previous):
            latest[record["pid"]] = record
    return sorted((item for item in latest.values() if item["active"]),
                  key=order, reverse=True)


class SnapshotPool:
    """Each generation has a new name: readers never race an overwritten buffer."""
    def __init__(self, directory, retain=3):
        self.directory = Path(directory)
        self.retain = retain
        self.mappings = []

    def write(self, arrays, iteration, kind="preview"):
        import numpy as np
        owned = {key: np.ascontiguousarray(arrays[key], dtype="<f4") for key in FIELDS}
        count = len(owned["means"])
        degree = int(round(owned["sh"].shape[1] ** 0.5)) - 1
        shapes = {"means": (count, 3), "scale": (count, 3), "rotation": (count, 4),
                  "opacity": (count, 1), "sh": (count, (degree + 1) ** 2, 3)}
        if not count or not 0 <= degree <= 3:
            raise ValueError("Invalid snapshot size or SH degree")
        for key, array in owned.items():
            if array.shape != shapes[key] or not np.isfinite(array).all():
                raise ValueError(f"Invalid snapshot array: {key}")
        size = sum(array.nbytes for array in owned.values())
        if not 0 < size <= MAX_BYTES:
            raise ValueError("Snapshot exceeds the shared-memory limit")
        token = "lfs_snapshot_" + uuid.uuid4().hex
        path = None
        if os.name == "nt":
            region = mmap.mmap(-1, size, tagname=token, access=mmap.ACCESS_WRITE)
            location = {"transport": "win32_mapping", "name": token}
        else:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = self.directory / (token + ".bin")
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "w+b") as stream:
                    stream.truncate(size)
                    region = mmap.mmap(stream.fileno(), size)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            location = {"transport": "mapped_file", "path": str(path.resolve())}
        layout, offset = {}, 0
        try:
            for key, array in owned.items():
                layout[key] = {"offset": offset, "shape": list(array.shape)}
                region.write(memoryview(array).cast("B"))
                offset += array.nbytes
            record = {"protocol": PROTOCOL, "id": token, "bytes": size,
                      "count": count, "sh_degree": degree, "iteration": iteration,
                      "kind": kind, "layout": layout, "created": time.time(), **location}
            self.mappings.append((region, path))
            while len(self.mappings) > self.retain:
                old, old_path = self.mappings.pop(0)
                old.close()
                if old_path:
                    old_path.unlink(missing_ok=True)
            return record
        except Exception:
            region.close()
            if path:
                path.unlink(missing_ok=True)
            raise

    def close(self):
        for region, path in self.mappings:
            region.close()
            if path:
                path.unlink(missing_ok=True)
        self.mappings.clear()


def read_snapshot(record):
    """Return owned arrays; release OS handles before touching application data."""
    import numpy as np

    if (not isinstance(record, dict) or type(record.get("protocol")) is not int
            or record["protocol"] != PROTOCOL):
        raise ValueError("Unsupported snapshot protocol")
    for key in ("iteration", "count", "sh_degree", "bytes"):
        if type(record.get(key)) is not int or record[key] < 0:
            raise ValueError("Invalid snapshot counter: " + key)
    count, degree, size = record["count"], record["sh_degree"], record["bytes"]
    if count < 1 or not 0 <= degree <= 3 or not 0 < size <= MAX_BYTES:
        raise ValueError("Invalid snapshot dimensions")
    shapes = {"means": (count, 3), "scale": (count, 3), "rotation": (count, 4),
              "opacity": (count, 1), "sh": (count, (degree + 1) ** 2, 3)}
    offset = 0
    for key, shape in shapes.items():
        entry = record["layout"][key]
        length = math.prod(shape) * 4
        if tuple(entry["shape"]) != shape or entry["offset"] != offset or offset + length > size:
            raise ValueError("Invalid snapshot layout")
        offset += length
    if offset != size:
        raise ValueError("Invalid snapshot size")
    close = None
    if record["transport"] == "win32_mapping" and os.name == "nt":
        kernel = _kernel32()
        handle = kernel.OpenFileMappingW(4, False, record["name"])
        if not handle:
            raise FileNotFoundError("Snapshot expired; waiting for the next generation")
        pointer = kernel.MapViewOfFile(handle, 4, 0, 0, size)
        if not pointer:
            kernel.CloseHandle(handle)
            raise OSError("Cannot map the published snapshot")
        buffer = memoryview((ctypes.c_ubyte * size).from_address(pointer)).cast("B")
        def close():
            kernel.UnmapViewOfFile(pointer)
            kernel.CloseHandle(handle)
    elif record["transport"] == "mapped_file" and os.name != "nt":
        with Path(record["path"]).open("rb") as stream:
            region = mmap.mmap(stream.fileno(), size, access=mmap.ACCESS_READ)
        buffer = memoryview(region)
        close = region.close
    else:
        raise ValueError("Snapshot transport is unavailable on this platform")
    try:
        return {key: np.frombuffer(buffer, dtype="<f4", count=math.prod(shape),
                                   offset=record["layout"][key]["offset"]).reshape(shape).copy()
                for key, shape in shapes.items()}
    finally:
        buffer.release()
        close()


class SessionPublisher:
    def __init__(self, metadata, status_provider=None):
        self.id = uuid.uuid4().hex
        self.root = registry_root() / self.id
        self.root.mkdir(mode=0o700, parents=True)
        self.pool = SnapshotPool(self.root / "buffers")
        self.lock = threading.Lock()
        self.io_lock = threading.Lock()
        self.status_provider = status_provider
        self.record = {**metadata, "protocol": PROTOCOL, "id": self.id, "pid": os.getpid(),
                       "started": time.time(), "state": metadata.get("state", "active"),
                       "process_identity": process_identity(os.getpid())}
        self.stop = threading.Event()
        self._clients_at = 0.0
        self._clients_cache = []
        self.heartbeat()
        self.thread = threading.Thread(target=self._beat, daemon=True)
        self.thread.start()

    def heartbeat(self):
        self.clients()
        values = {}
        if self.status_provider and not self.stop.is_set() and self.record.get('state') not in TERMINAL:
            try:
                values = self.status_provider() or {}
            except Exception:
                pass  # A status probe must never break training/publication.
        with self.io_lock:
            with self.lock:
                if not self.stop.is_set() and self.record.get('state') not in TERMINAL:
                    self.record.update(values)
                record = {**self.record, "heartbeat": time.time()}
            atomic_json(self.root / "session.json", record)

    def _beat(self):
        while not self.stop.wait(2):
            try:
                self.heartbeat()
            except OSError:
                pass  # The next heartbeat can recover transient filesystem faults.

    def update(self, *, flush=True, **values):
        with self.lock:
            self.record.update(values)
        if flush:
            self.heartbeat()

    @property
    def has_clients(self):
        """Heartbeat-maintained demand; safe-boundary hooks do no filesystem IO."""
        return bool(self._clients_cache)

    def publish(self, arrays, iteration, kind="preview", *, update_progress=True):
        snapshot = self.pool.write(arrays, iteration, kind)
        values = {"snapshot": snapshot}
        if update_progress:
            values.update(iteration=iteration, count=snapshot["count"])
        self.update(**values)
        return snapshot

    def clients(self):
        if time.monotonic() - self._clients_at < 0.5:
            return self._clients_cache
        self._clients_at = time.monotonic()
        clients = []
        for path in (self.root / "clients").glob("*.json"):
            value = read_json(path, {})
            if not isinstance(value, dict):
                continue
            updated = value.get("updated", 0)
            liveness = session_liveness({**value, "heartbeat": updated})
            if liveness == "responsive":
                clients.append(value)
            elif liveness == "exited":
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self._clients_cache = clients
        return self._clients_cache

    def finish(self, state="completed", **values):
        try:
            self.update(state=state, **values)
        finally:
            self.stop.set()
        # Keep last mappings until process exit, so late readers can still attach.
        # Do not wait for a filesystem operation in the heartbeat thread.


class SessionClient:
    def __init__(self, directory, label):
        self.root = Path(directory)
        self.id = uuid.uuid4().hex
        self.label = label
        self.identity = process_identity(os.getpid())
        self._touch_at = None
        self.touch()

    def touch(self):
        if self._touch_at is not None and time.monotonic() - self._touch_at < 1:
            return
        atomic_json(self.root / "clients" / (self.id + ".json"),
                    {"pid": os.getpid(), "process_identity": self.identity,
                     "updated": time.time(), "label": self.label})
        self._touch_at = time.monotonic()

    def status(self):
        self.touch()
        return read_json(self.root / "session.json", {})

    def close(self):
        (self.root / "clients" / (self.id + ".json")).unlink(missing_ok=True)

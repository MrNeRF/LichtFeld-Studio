# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Install the released LichtFeld extension through Blender's own CLI."""

import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from http.client import HTTPException
from threading import Lock, Thread
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request

from _lfs_path_io import Path, native_path

from .http import urlopen
from .persisted_settings import load_object, write_object

PROJECT = "MrNeRF/lichtfeld-blender-plugin"
RELEASES = f"https://github.com/{PROJECT}/releases"
EXTENSION = "lichtfeld_training"
_LOG = logging.getLogger(__name__)
_OPERATION = Lock()
MAX_DOWNLOAD = 32 * 1024 * 1024
MIN_BLENDER = (5, 3, 0)
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-beta\.(\d+))?$")
_PROBE = f"""\
import bpy, json, pathlib
if bpy.app.version < {MIN_BLENDER!r}:
    print("LFS_BLENDER_INFO " + json.dumps(dict(version=bpy.app.version)))
    raise SystemExit(0)
import tomllib
repos = bpy.context.preferences.extensions.repos
local = next((r for r in repos if r.enabled and r.module == "user_default"
              and r.source == "USER" and not r.use_remote_url), None)
installed = []
for repo in repos:
    if not repo.directory:
        continue
    path = pathlib.Path(repo.directory) / "lichtfeld_training" / "blender_manifest.toml"
    if path.is_file():
        installed.append((repo.module, tomllib.loads(path.read_text(encoding="utf-8"))["version"]))
legacy_dir = pathlib.Path(bpy.utils.user_resource("SCRIPTS")) / "addons"
legacy = (legacy_dir / "lichtfeld_training").is_dir() or (legacy_dir / "lichtfeld_training.py").is_file()
print("LFS_BLENDER_INFO " + json.dumps(dict(version=bpy.app.version, config=bpy.utils.user_resource("CONFIG"),
    destination=local.directory if local else "", installed=installed,
    enabled="bl_ext.user_default.lichtfeld_training" in bpy.context.preferences.addons,
    legacy=legacy)))
"""


@dataclass(frozen=True)
class Release:
    version: str
    url: str
    digest: str
    size: int


class IntegrationError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def version_key(value):
    match = _VERSION.fullmatch(value)
    if not match:
        raise ValueError("Unsupported extension version")
    major, minor, patch, beta = match.groups()
    return int(major), int(minor), int(patch), beta is None, int(beta or 0)


def _settings_path():
    base = Path(os.environ.get("LFS_RESOLVED_DATA_DIR") or Path.home() / ".lichtfeld" / "data")
    return base / "blender-extension.json"


def preferences():
    raw, _ = load_object(_settings_path())
    path = raw.get("path")
    channel = raw.get("channel")
    return path if isinstance(path, str) else "", channel if channel in ("stable", "beta") else ""


def save_preferences(path, channel):
    target = _settings_path()
    current, damaged = load_object(target)
    values = {"path": path, "channel": channel}
    if not damaged and current == values:
        return
    try:
        write_object(target, values, damaged=damaged)
    except OSError:
        _LOG.exception("Could not save Blender integration preferences")


def _read_release_url(url, limit, timeout):
    try:
        headers = {"User-Agent": "LichtFeld-Studio"}
        if url.startswith("https://api.github.com/"):
            headers["Accept"] = "application/vnd.github+json"
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            return response.read(limit), getattr(response, "headers", {}).get("Link", "")
    except HTTPError as exc:
        _LOG.warning("GitHub release request failed with HTTP %s", exc.code)
        headers = exc.headers or {}
        if exc.code == 429 or (exc.code == 403 and (
                headers.get("X-RateLimit-Remaining") == "0" or headers.get("Retry-After"))):
            raise IntegrationError("rate_limited", "GitHub release request was rate limited") from exc
        raise IntegrationError("github_response", "GitHub rejected the release request") from exc
    except (OSError, HTTPException) as exc:
        raise IntegrationError("network", "Could not reach the GitHub release") from exc


def releases():
    url = f"https://api.github.com/repos/{PROJECT}/releases?per_page=100"
    entries = []
    visited = set()
    page_limit = 8 * 1024 * 1024
    try:
        while url:
            if url in visited or urlsplit(url).netloc != "api.github.com" or urlsplit(url).scheme != "https":
                raise ValueError("Invalid release catalog pagination link")
            visited.add(url)
            body, links = _read_release_url(url, page_limit + 1, 12)
            if len(body) > page_limit:
                raise ValueError("Release catalog page exceeds the response limit")
            page = json.loads(body)
            if not isinstance(page, list):
                raise ValueError("The release catalog page must be a list")
            entries.extend(page)
            match = re.search(r'<([^>]+)>;\s*rel="next"', links)
            url = match[1] if match else ""
        return _select_releases(entries)
    except IntegrationError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise IntegrationError("catalog", "GitHub returned an invalid release list") from exc


def _select_releases(entries):
    if not isinstance(entries, list):
        raise ValueError("The release catalog must be a list")
    options = {}
    for entry in entries:
        if entry["draft"]:
            continue
        version = entry["tag_name"].removeprefix("v")
        match = _VERSION.fullmatch(version)
        if not match or bool(match[4]) != entry["prerelease"]:
            continue
        channel = "beta" if entry["prerelease"] else "stable"
        filename = f"{EXTENSION}-{version}.zip"
        expected = f"{RELEASES}/download/{entry['tag_name']}/{filename}"
        asset = next((asset for asset in entry["assets"] if asset["name"] == filename), None)
        if not asset or asset["browser_download_url"] != expected:
            continue
        digest = asset.get("digest") or ""
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest) or not 0 < asset["size"] <= MAX_DOWNLOAD:
            continue
        candidate = Release(version, expected, digest[7:], asset["size"])
        options.setdefault(channel, {}).setdefault(version, candidate)
    return {
        channel: tuple(sorted(entries.values(), key=lambda item: version_key(item.version), reverse=True))
        for channel, entries in options.items()
    }


def _entered_path(value):
    return value.strip().strip('"')


def _executable_path(executable):
    path = Path(executable)
    if not path.is_absolute() or not path.is_file():
        raise IntegrationError("invalid_path", "Choose an absolute path to Blender")
    if path.name.casefold() != ("blender.exe" if os.name == "nt" else "blender"):
        raise IntegrationError("invalid_path", "Choose the Blender executable")
    return path


def _blender_environment():
    library_dir = os.environ.get("LFS_LAUNCHER_LIBRARY_DIR")
    if os.name == "nt" or not library_dir:
        return None
    env = os.environ.copy()
    env.pop("LFS_LAUNCHER_LIBRARY_DIR")
    first, _, remaining = env.get("LD_LIBRARY_PATH", "").partition(":")
    if first == library_dir:
        if remaining:
            env["LD_LIBRARY_PATH"] = remaining
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


def _run(executable, args, *, timeout=60):
    path = _executable_path(executable)
    try:
        result = subprocess.run([native_path(path), *args], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace", timeout=timeout, env=_blender_environment(),
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired as exc:
        _LOG.warning("Blender timed out after %s seconds. Output tail: %r", timeout, (exc.stdout or b"")[-4096:])
        raise IntegrationError("timeout", "Blender did not respond before the timeout") from exc
    except OSError as exc:
        raise IntegrationError("invalid_path", "Could not launch the selected Blender executable") from exc
    if result.returncode:
        _LOG.warning("Blender exited with status %s. Output tail: %s", result.returncode, result.stdout[-4096:])
        if "lock is held by other session" in result.stdout and "Force Unlock Repository" in result.stdout:
            raise IntegrationError("repository_locked", "Blender reported a repository lock")
        raise ValueError(f"Blender exited with code {result.returncode}")
    return result.stdout


def probe(executable):
    output = _run(executable, ["--background", "--disable-autoexec", "--python-exit-code", "1",
                               "--python-expr", _PROBE])
    prefix = "LFS_BLENDER_INFO "
    line = next((line for line in output.splitlines() if line.startswith(prefix)), None)
    if line is None:
        _LOG.warning("Blender returned no profile information. Output tail: %s", output[-4096:])
        raise ValueError("Blender did not report its version and profile")
    info = json.loads(line[len(prefix):])
    if tuple(info["version"]) < MIN_BLENDER:
        raise IntegrationError("blender_version_required", "Blender 5.3 or newer is required")
    if not info["destination"] or not info["config"]:
        raise IntegrationError("repository", "Blender does not have an enabled User Default repository")
    installed = info["installed"]
    if info["legacy"] or installed and (len(installed) != 1 or installed[0][0] != "user_default"):
        raise IntegrationError("existing", "Manage the existing installation inside Blender")
    return info


def install(executable, release, expected, *, replace_existing=False):
    installed = expected["installed"]
    if installed and not replace_existing and version_key(release.version) <= version_key(installed[0][1]):
        raise ValueError("The selected release would replace the same or a newer version")
    data, _ = _read_release_url(release.url, release.size + 1, 20)
    if len(data) != release.size or hashlib.sha256(data).hexdigest() != release.digest:
        raise IntegrationError("package", "The downloaded extension failed its integrity check")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            manifest_entry = archive.getinfo("blender_manifest.toml")
            if manifest_entry.file_size > 64 * 1024:
                raise IntegrationError("package", "The extension manifest exceeds its supported size")
            manifest = tomllib.loads(archive.read(manifest_entry).decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise IntegrationError("package", "The release ZIP has no readable Blender manifest") from exc
    if manifest.get("id") != EXTENSION or manifest.get("version") != release.version:
        raise IntegrationError("package", "The extension package does not match the selected release")
    version = tuple(expected["version"])
    try:
        minimum = tuple(map(int, manifest["blender_version_min"].split(".")))
        maximum = tuple(map(int, manifest["blender_version_max"].split("."))) if "blender_version_max" in manifest else None
    except (AttributeError, KeyError, ValueError) as exc:
        raise IntegrationError("package", "The release ZIP has invalid Blender version metadata") from exc
    if version < minimum or maximum and version >= maximum:
        raise IntegrationError("incompatible", "The release is incompatible with this Blender version")
    if probe(executable) != expected:
        raise IntegrationError("profile_changed", "The Blender profile or installation changed")
    with tempfile.TemporaryDirectory() as directory:
        package = Path(directory) / f"{EXTENSION}-{release.version}.zip"
        package.write_bytes(data)
        output = _run(executable, ["--command", "extension", "install-file", "-r", "user_default", "-e",
                                   native_path(package)], timeout=180)
    try:
        verified = probe(executable)
    except (OSError, ValueError):
        _LOG.warning("Could not verify the installed extension. Installer output tail: %s", output[-4096:])
        raise
    if (any(verified[key] != expected[key] for key in ("config", "destination", "version"))
            or verified["installed"] != [["user_default", release.version]] or not verified["enabled"]):
        _LOG.warning("Blender did not report the expected installation. Installer output tail: %s", output[-4096:])
        raise ValueError("Blender did not report the requested extension after installation")
    return verified


class PanelSection:
    """Main-thread data bindings for a single asynchronous operation at a time."""

    def __init__(self):
        self.path, stored_channel = preferences()
        self.saved_path = self.path
        self.path = self.path or shutil.which("blender") or ""
        self.channel = stored_channel or "stable"
        self.info = None
        self.options = {}
        self.selected_version = ""
        self.busy = False
        self.status = "unchecked"
        self.handle = None
        self.version_picker_open = False
        self.version_query = ""
        self._published_versions = None

    def bind(self, model):
        import lichtfeld as lf

        model.bind("blender_path", lambda: self.path, self._set_path)
        model.bind("blender_channel", lambda: self.channel, self._set_channel)
        model.bind("blender_selected_version", lambda: self.selected_version, self._set_version)
        model.bind_record_list("blender_release_versions")
        model.bind("blender_version_query", lambda: self.version_query, self._set_version_query)
        model.bind_func("blender_version_picker_open", lambda: self.version_picker_open)
        model.bind_func("blender_has_version_matches", lambda: bool(self._published_versions))
        model.bind_event("blender_toggle_version_picker", lambda *_: self._toggle_version_picker())
        model.bind_event("blender_choose_version",
                         lambda _handle, _event, args: self._set_version(str(args[0])) if args else None)
        model.bind_func("blender_has_releases", lambda: bool(self.options.get(self.channel)))
        model.bind_func("blender_can_replace", self._can_replace)
        model.bind_func("blender_replace_action", lambda: self._tr(self._replace_action()))
        model.bind_func("blender_busy", lambda: self.busy)
        model.bind_func("blender_version", lambda: ".".join(map(str, self.info["version"])) if self.info else "")
        model.bind_func("blender_destination", lambda: self.info["destination"] if self.info else "")
        model.bind_func("blender_installed", lambda: self._installed() or "—")

        model.bind_func("blender_action", lambda: self._tr("blender_integration.update" if self._installed() else "blender_integration.install"))
        model.bind_func("blender_installable", self._installable)
        model.bind_func("blender_status", self._status_label)
        model.bind_event("blender_browse", lambda *_: self._browse())
        model.bind_event("blender_open", lambda *_: self._open_blender())
        model.bind_event("blender_check", lambda *_: self._check())
        model.bind_event("blender_install", lambda *_: self._confirm())
        model.bind_event("blender_replace", lambda *_: self._confirm(replace_existing=True))
        model.bind_event("blender_releases", lambda *_: lf.ui.open_url(RELEASES))

    @staticmethod
    def _tr(key):
        import lichtfeld as lf
        return lf.ui.tr(key)

    def _installed(self):
        return self.info["installed"][0][1] if self.info and self.info["installed"] else ""

    def _release(self):
        releases = self.options.get(self.channel, ())
        if self.selected_version:
            return next((release for release in releases if release.version == self.selected_version), None)
        return releases[0] if releases else None

    def _can_replace(self):
        return bool(self.info and self._installed() and self._release() and not self._is_new_install())

    def _replace_action(self):
        if not self._can_replace():
            return "blender_integration.replace"
        try:
            same_version = version_key(self._installed()) == version_key(self._release().version)
        except ValueError:
            return "blender_integration.replace"
        return "blender_integration.reinstall" if same_version else "blender_integration.downgrade"

    def _installable(self):
        return not self.busy and self._is_new_install()

    def _is_new_install(self):
        release = self._release()
        if not self.info or not release:
            return False
        installed = self._installed()
        try:
            return not installed or version_key(release.version) > version_key(installed)
        except ValueError:
            return False

    def _status_label(self):
        if self.status == "ready":
            if self.channel not in self.options:
                return self._tr("blender_integration.unavailable")
            if self._installed():
                try:
                    version_key(self._installed())
                except ValueError:
                    return self._tr("blender_integration.unrecognized")
                if not self.info["enabled"] and not self._installable():
                    return self._tr("blender_integration.disabled")
            if self._installed() and version_key(self.options[self.channel][0].version) <= version_key(self._installed()):
                return self._tr("blender_integration.current")
        return self._tr(f"blender_integration.{self.status}")

    def set_handle(self, handle):
        self.handle = handle
        self._published_versions = None
        self._dirty()

    def _toggle_version_picker(self):
        if not self.busy:
            self.version_picker_open = not self.version_picker_open
            self.version_query = ""
            self._dirty()

    def _set_version_query(self, value):
        if not self.busy and self.version_picker_open and value != self.version_query:
            self.version_query = value
            self._dirty()

    def _version_matches(self):
        if not self.version_picker_open:
            return ()
        versions = self.options.get(self.channel, ())
        query = self.version_query.strip().lower().removeprefix("v")
        if not query:
            return tuple(release.version for release in versions[:8])
        exact = ()
        matches = []
        for release in versions:
            if release.version == query:
                exact = (release.version,)
            elif query in release.version and len(matches) < 8:
                matches.append(release.version)
        return exact + tuple(matches[:8 - len(exact)])

    def _dirty(self):
        if self.handle:
            versions = self._version_matches()
            if versions != self._published_versions:
                self._published_versions = versions
                self.handle.update_record_list("blender_release_versions", [
                    {"version": version} for version in versions
                ])
            for name in ("path", "channel", "busy", "version", "destination", "installed",
                         "selected_version", "has_releases", "can_replace", "replace_action", "action", "installable", "status",
                         "version_picker_open", "version_query", "has_version_matches"):
                self.handle.dirty("blender_" + name)

    def _browse(self):
        import lichtfeld as lf

        if self.busy:
            return
        current = Path(_entered_path(self.path))
        start = str(current.parent) if current.is_absolute() else ""
        chosen = lf.ui.open_file_dialog(start)
        if chosen:
            self._set_path(chosen)

    def _open_blender(self):
        if self.busy:
            return
        try:
            path = _executable_path(_entered_path(self.path))
            process = subprocess.Popen([native_path(path)], close_fds=True,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=os.name != "nt", env=_blender_environment())
            Thread(target=process.wait, daemon=True).start()
        except IntegrationError as exc:
            self.status = exc.status
            self._dirty()
        except OSError:
            _LOG.exception("Could not open Blender")
            self.status = "failed"
            self._dirty()

    def _set_path(self, value):
        if not self.busy and self.path != value:
            self.path = value
            self.info = None
            self.options = {}
            self.selected_version = ""
            self.version_picker_open = False
            self.version_query = ""
            self.status = "unchecked"
            self._dirty()

    def _set_channel(self, value):
        if not self.busy and value in ("stable", "beta") and self.channel != value:
            self.channel = value
            self.selected_version = ""
            self.version_picker_open = False
            self.version_query = ""
            selected = self._release()
            self.selected_version = selected.version if selected else ""
            if self.status == "complete":
                self.status = "ready"
            self._dirty()
            save_preferences(self.saved_path, self.channel)

    def _set_version(self, value):
        if self.busy:
            return
        if value and all(release.version != value for release in self.options.get(self.channel, ())):
            return
        self.selected_version = value
        self.version_picker_open = False
        self.version_query = ""
        if self.status == "complete":
            self.status = "ready"
        self._dirty()

    def _start(self, status, action, complete):
        import lichtfeld as lf

        if not _OPERATION.acquire(blocking=False):
            self.status = "busy"
            self._dirty()
            return
        self.busy = True
        self.status = status
        self._dirty()

        def run():
            try:
                result = action()
            except Exception as exc:
                status_key = exc.status if isinstance(exc, IntegrationError) else "failed"
                _LOG.exception("Blender extension operation failed")

                def finish():
                    self.info = None
                    self.busy = False
                    self.status = status_key
                    self._dirty()
            else:
                def finish():
                    self.busy = False
                    complete(result)
                    self._dirty()

            finally:
                _OPERATION.release()

            lf.ui.schedule_on_ui_thread(finish)

        Thread(target=run, daemon=True).start()

    def _check(self):
        if self.busy:
            return
        executable = _entered_path(self.path)
        chosen = self.channel

        def action():
            try:
                info = probe(executable)
            except IntegrationError as exc:
                return None, None, exc.status
            save_preferences(executable, chosen)
            try:
                return releases(), info, "ready"
            except IntegrationError as exc:
                return None, info, exc.status

        def complete(result):
            options, self.info, self.status = result
            self.path = executable
            if options is not None:
                self.options = options
                available = options.get(self.channel, ())
                self.selected_version = available[0].version if available else ""
            if self.info is not None:
                self.saved_path = executable

        self._start("checking", action, complete)

    def _confirm(self, *, replace_existing=False):
        import lichtfeld as lf

        allowed = self._can_replace if replace_existing else self._installable
        if self.busy or not allowed():
            return
        executable, release, expected = self.path, self._release(), self.info
        if replace_existing:
            title = self._tr(self._replace_action())
        else:
            title = self._tr("blender_integration.update" if self._installed() else "blender_integration.install")

        def confirmed(button):
            if (button != title or self.handle is None or self.busy or not allowed()
                    or self.path != executable or self.info != expected):
                return
            if self._release() != release:
                return

            def complete(info):
                self.info = info
                self.status = "complete"

            self._start("installing", lambda: install(
                executable, release, expected, replace_existing=replace_existing), complete)

        message = self._tr("blender_integration.confirm_replace" if replace_existing else "blender_integration.confirm")
        lf.ui.confirm_dialog(title, message.format(installed=self._installed(),
            version=release.version, destination=expected["destination"]),
            [lf.ui.tr("common.cancel"), title], confirmed)

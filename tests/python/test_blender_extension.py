# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the Blender extension installation boundary."""

import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import posixpath
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/python"))
try:
    from lfs_plugins import blender_extension as integration
finally:
    sys.path.pop(0)


def release_entry(version, *, draft=False):
    name = f"{integration.EXTENSION}-{version}.zip"
    tag = f"v{version}"
    return {
        "draft": draft, "tag_name": tag, "prerelease": "-beta." in version,
        "assets": [{"name": name, "size": 42, "digest": "sha256:" + "a" * 64,
                    "browser_download_url": f"{integration.RELEASES}/download/{tag}/{name}"}],
    }


class BlenderExtensionTests(unittest.TestCase):
    def test_catalog_selects_highest_valid_version_per_channel(self):
        wrong_url = release_entry("2.0.0")
        wrong_url["assets"][0]["browser_download_url"] = "https://example.org/release.zip"
        wrong_digest = release_entry("2.1.0")
        wrong_digest["assets"][0]["digest"] = None
        entries = [
            release_entry("1.0.0-beta.10"), release_entry("1.0.0-beta.9"),
            release_entry("1.0.0"), release_entry("3.0.0", draft=True), wrong_url, wrong_digest,
        ]
        with patch.object(integration, "urlopen", return_value=io.BytesIO(json.dumps(entries).encode())):
            options = integration.releases()
        self.assertEqual({key: value[0].version for key, value in options.items()},
                         {"beta": "1.0.0-beta.10", "stable": "1.0.0"})
        self.assertGreater(integration.version_key(options["stable"][0].version),
                           integration.version_key(options["beta"][0].version))

    def test_catalog_finds_stable_after_many_beta_releases(self):
        endpoint = f"https://api.github.com/repos/{integration.PROJECT}/releases?per_page=100"
        pages = []
        for page in range(1, 11):
            entries = [release_entry(f"1.0.0-beta.{number}") for number in range(
                (page - 1) * 100 + 1, page * 100 + 1)]
            response = io.BytesIO(json.dumps(entries).encode())
            response.headers = {"Link": f'<{endpoint}&page={page + 1}>; rel="next"'}
            pages.append(response)
        pages.append(io.BytesIO(json.dumps([release_entry("1.0.0")]).encode()))
        with patch.object(integration, "urlopen", side_effect=pages) as request:
            options = integration.releases()
        self.assertEqual([call.args[0].full_url for call in request.call_args_list],
                         [endpoint] + [f"{endpoint}&page={page}" for page in range(2, 12)])
        self.assertEqual(len(options["beta"]), 1000)
        self.assertEqual(options["beta"][0].version, "1.0.0-beta.1000")
        self.assertEqual(options["stable"][0].version, "1.0.0")

    def test_version_picker_finds_old_versions_without_rendering_the_full_catalog(self):
        with patch.object(integration, "preferences", return_value=("", "beta")):
            section = integration.PanelSection()
        section.options = integration._select_releases([
            release_entry(f"1.0.0-beta.{number}") for number in range(1, 1001)
        ])
        self.assertEqual(section._release().version, "1.0.0-beta.1000")
        self.assertEqual(section._version_matches(), ())
        section._toggle_version_picker()
        self.assertEqual(len(section._version_matches()), 8)
        section._set_version_query("v1.0.0-beta.1")
        self.assertEqual(section._version_matches()[0], "1.0.0-beta.1")
        section._set_version("1.0.0-beta.1")
        self.assertEqual(section._release().version, "1.0.0-beta.1")
        self.assertFalse(section.version_picker_open)

    def test_check_keeps_stable_selected_until_user_chooses_beta(self):
        info = {"installed": [], "destination": "/extensions"}
        release = integration.Release("1.0.0-beta.1", "unused", "unused", 1)
        with (patch.object(integration, "preferences", return_value=("blender", "")),
              patch.object(integration, "probe", return_value=info),
              patch.object(integration, "releases", return_value={"beta": (release,)}),
              patch.object(integration, "save_preferences") as save):
            section = integration.PanelSection()
            with patch.object(section, "_start", side_effect=lambda _, action, complete: complete(action())):
                section._check()
            self.assertEqual(section.channel, "stable")
            self.assertFalse(section._installable())
            section._set_channel("beta")
            self.assertTrue(section._installable())
            save.assert_called_with("blender", "beta")

    def test_probe_rejects_older_blender_before_accessing_extension_repository(self):
        with patch.object(integration, "_run", return_value='LFS_BLENDER_INFO {"version":[5,2,9]}'):
            with self.assertRaises(integration.IntegrationError) as error:
                integration.probe("blender")
        self.assertEqual(error.exception.status, "blender_version_required")

    def test_probe_rejects_conflicting_installations(self):
        for installed, legacy in [([["remote", "1.0.0"]], False),
                                  ([["user_default", "1.0.0"], ["remote", "1.0.0"]], False),
                                  ([], True)]:
            with self.subTest(installed=installed, legacy=legacy):
                info = {"version": [5, 3, 0], "destination": "/extensions", "config": "/config",
                        "installed": installed, "legacy": legacy}
                with patch.object(integration, "_run", return_value="LFS_BLENDER_INFO " + json.dumps(info)):
                    with self.assertRaises(integration.IntegrationError) as error:
                        integration.probe("blender")
                self.assertEqual(error.exception.status, "existing")

    def test_confirmation_starts_install_only_on_affirmative_label(self):
        def translate(key):
            return key if not key.endswith(".confirm") else "Install {version} in {destination}?"

        with patch.object(integration, "preferences", return_value=("blender", "beta")):
            section = integration.PanelSection()
        section.handle = object()
        section.info = {"installed": [], "destination": "/extensions"}
        section.options = {"beta": (integration.Release("1.0.0-beta.2", "unused", "unused", 1),)}
        callbacks = []
        ui = SimpleNamespace(tr=translate, confirm_dialog=lambda *args: callbacks.append(args[-1]))
        with patch.dict(sys.modules, {"lichtfeld": SimpleNamespace(ui=ui)}), patch.object(section, "_start") as start:
            section._confirm()
            callback, = callbacks
            callback("common.cancel")
            start.assert_not_called()
            callback("blender_integration.install")
            start.assert_called_once()

    def test_replacing_unrecognized_version_requires_explicit_confirmation(self):
        with patch.object(integration, "preferences", return_value=("blender", "beta")):
            section = integration.PanelSection()
        section.handle = object()
        section.info = {"installed": [["user_default", "1.0.0-dev.1"]], "destination": "/extensions"}
        release = integration.Release("1.0.0-beta.1", "unused", "unused", 1)
        section.options = {"beta": (release,)}
        dialog = Mock()
        ui = SimpleNamespace(
            tr=lambda key: "{installed} to {version} at {destination}" if key.endswith(".confirm_replace") else key,
            confirm_dialog=dialog,
        )
        with (patch.dict(sys.modules, {"lichtfeld": SimpleNamespace(ui=ui)}),
              patch.object(section, "_start") as start,
              patch.object(integration, "install") as install):
            section._confirm()
            dialog.assert_not_called()
            section._confirm(replace_existing=True)
            title, message, _, callback = dialog.call_args.args
            self.assertEqual(title, "blender_integration.replace")
            self.assertIn("1.0.0-dev.1 to 1.0.0-beta.1 at /extensions", message)
            callback("common.cancel")
            start.assert_not_called()
            callback(title)
            start.assert_called_once()
            start.call_args.args[1]()
        install.assert_called_once_with("blender", release, section.info, replace_existing=True)

    def test_browse_keeps_the_checked_profile_on_cancel_and_clears_it_on_change(self):
        with patch.object(integration, "preferences", return_value=("", "beta")):
            section = integration.PanelSection()
        section.path = "previous"
        section.info = {"installed": []}
        ui = SimpleNamespace(open_file_dialog=Mock(side_effect=["", "chosen"]))
        with patch.dict(sys.modules, {"lichtfeld": SimpleNamespace(ui=ui)}):
            section._browse()
            self.assertIsNotNone(section.info)
            section._browse()
        self.assertEqual(section.path, "chosen")
        self.assertIsNone(section.info)
        self.assertEqual(section.status, "unchecked")

    def test_open_blender_uses_the_selected_executable_and_inherited_profile(self):
        blender = Path(sys.executable).with_name("blender.exe" if sys.platform == "win32" else "blender")
        with patch.object(integration, "preferences", return_value=(f'"{blender}"', "beta")):
            section = integration.PanelSection()
        with (patch.object(integration.Path, "is_file", return_value=True),
              patch.object(integration.subprocess, "Popen") as launch,
              patch.object(integration, "Thread") as thread):
            section._open_blender()
        launch.assert_called_once_with(
            [integration.native_path(integration.Path(blender))], close_fds=True,
            stdin=integration.subprocess.DEVNULL, stdout=integration.subprocess.DEVNULL,
            stderr=integration.subprocess.DEVNULL, start_new_session=integration.os.name != "nt", env=None)
        thread.assert_called_once_with(target=launch.return_value.wait, daemon=True)
        thread.return_value.start.assert_called_once()

    def test_blender_child_removes_only_studios_first_library_path(self):
        for value, remaining in (
            ("/studio/bin/../lib:/user/lib", "/user/lib"),
            ("/studio/lib", None),
            ("/user/lib:/studio/lib", "/user/lib:/studio/lib"),
        ):
            with (self.subTest(value=value),
                  patch.object(integration.sys, "platform", "linux"),
                  patch.object(integration.os, "readlink", return_value="/studio/bin/LichtFeld-Studio"),
                  patch.object(integration.os, "path", posixpath),
                  patch.dict(integration.os.environ, {
                      "LD_LIBRARY_PATH": value, "BLENDER_USER_CONFIG": "/user/config",
                  })):
                env = integration._blender_environment()
                if value.startswith("/user/"):
                    self.assertIsNone(env)
                else:
                    self.assertEqual(env.get("LD_LIBRARY_PATH"), remaining)
                    self.assertEqual(env["BLENDER_USER_CONFIG"], "/user/config")
                self.assertEqual(integration.os.environ["LD_LIBRARY_PATH"], value)

    def test_release_failure_keeps_verified_profile_and_saves_path(self):
        info = {"installed": [], "destination": "/extensions"}
        for payload, expected in [(b"[", "catalog"), (b"[{}]", "catalog"), (None, "network")]:
            with self.subTest(expected=expected, payload=payload):
                response = io.BytesIO(payload or b"")
                if payload is None:
                    response.read = Mock(side_effect=IncompleteRead(b"partial"))
                with (patch.object(integration, "preferences", return_value=(" blender ", "")),
                      patch.object(integration, "probe", return_value=info),
                      patch.object(integration, "save_preferences") as save,
                      patch.object(integration, "urlopen", return_value=response)):
                    section = integration.PanelSection()
                    with patch.object(section, "_start", side_effect=lambda _, action, complete: complete(action())):
                        section._check()
                self.assertIs(section.info, info)
                self.assertEqual(section.options, {})
                self.assertEqual(section.status, expected)
                save.assert_called_once_with("blender", "stable")

    def test_github_http_failures_distinguish_rate_limits(self):
        for code, headers, status in (
            (429, {}, "rate_limited"),
            (403, {"X-RateLimit-Remaining": "0"}, "rate_limited"),
            (403, {"Retry-After": "60"}, "rate_limited"),
            (403, {}, "github_response"),
            (404, {}, "github_response"),
        ):
            with self.subTest(code=code, headers=headers), patch.object(
                integration, "urlopen", side_effect=HTTPError("https://api.github.com/", code, "", headers, None)
            ):
                with self.assertRaises(integration.IntegrationError) as error:
                    integration.releases()
                self.assertEqual(error.exception.status, status)

    def test_failed_confirmed_install_requires_new_profile_check(self):
        with patch.object(integration, "preferences", return_value=("blender", "stable")):
            section = integration.PanelSection()
        options = integration._select_releases([release_entry("1.0.0")])
        section.options = options
        info = {"installed": [], "destination": "/extensions"}
        section.info = info
        dialog = Mock()
        ui = SimpleNamespace(tr=str, confirm_dialog=dialog, schedule_on_ui_thread=lambda callback: callback())
        section.handle = Mock()
        with (patch.dict(sys.modules, {"lichtfeld": SimpleNamespace(ui=ui)}),
              patch.object(integration, "Thread", side_effect=lambda *, target, daemon: SimpleNamespace(start=target)),
              patch.object(integration, "install", side_effect=integration.IntegrationError("package", "bad package")) as install):
            section._confirm()
            title, _, _, callback = dialog.call_args.args
            callback(title)
            install.assert_called_once_with("blender", options["stable"][0], info, replace_existing=False)
            section._confirm()
            dialog.assert_called_once()
        self.assertEqual(section.status, "package")
        self.assertIsNone(section.info)
        self.assertIs(section.options, options)
        self.assertFalse(section.busy)
        self.assertFalse(section._installable())
        self.assertFalse(integration._OPERATION.locked())

    def test_invalid_probe_clears_previously_verified_installation(self):
        with patch.object(integration, "preferences", return_value=("blender", "stable")):
            section = integration.PanelSection()
        options = integration._select_releases([release_entry("1.0.0")])
        section.options = options
        section.info = {"installed": [["user_default", "0.9.0"]]}
        self.assertTrue(section._installable())
        lf = SimpleNamespace(ui=SimpleNamespace(schedule_on_ui_thread=lambda callback: callback()))
        with (patch.dict(sys.modules, {"lichtfeld": lf}),
              patch.object(integration, "Thread", side_effect=lambda *, target, daemon: SimpleNamespace(start=target)),
              patch.object(integration, "probe", side_effect=ValueError("Invalid Blender response")),
              patch.object(integration, "releases") as fetch):
            section._check()
        fetch.assert_not_called()
        self.assertIs(section.options, options)
        self.assertIsNone(section.info)
        self.assertEqual(section.status, "failed")
        self.assertFalse(section.busy)
        self.assertFalse(section._installable())
        self.assertFalse(section._can_replace())
        self.assertFalse(integration._OPERATION.locked())

    def test_check_rejected_by_busy_worker_keeps_verified_installation(self):
        with patch.object(integration, "preferences", return_value=("blender", "stable")):
            section = integration.PanelSection()
        options = integration._select_releases([release_entry("1.0.0")])
        info = {"installed": [], "destination": "/extensions"}
        section.options = options
        section.info = info
        with (integration._OPERATION, patch.dict(sys.modules, {"lichtfeld": SimpleNamespace()}),
              patch.object(integration, "Thread") as thread):
            section._check()
            self.assertTrue(integration._OPERATION.locked())
        thread.assert_not_called()
        self.assertIs(section.info, info)
        self.assertIs(section.options, options)
        self.assertEqual(section.status, "busy")
        self.assertFalse(section.busy)

    def test_failed_catalog_refresh_preserves_last_complete_catalog_and_error(self):
        info = {"installed": [], "destination": "/extensions"}
        with patch.object(integration, "preferences", return_value=("blender", "")):
            section = integration.PanelSection()
        options = integration._select_releases([release_entry("1.0.0"), release_entry("0.9.0")])
        section.options = options
        section.selected_version = "1.0.0"
        with (patch.object(integration, "probe", return_value=info),
              patch.object(integration, "save_preferences"),
              patch.object(integration, "releases", side_effect=integration.IntegrationError("network", "offline")),
              patch.object(section, "_start", side_effect=lambda _, action, complete: complete(action()))):
            section._check()
        self.assertIs(section.options, options)
        self.assertIs(section.info, info)
        section._set_version("0.9.0")
        self.assertEqual(section._release().version, "0.9.0")
        self.assertEqual(section.status, "network")

    def test_replace_button_distinguishes_downgrade_reinstall_and_unknown(self):
        with patch.object(integration, "preferences", return_value=("", "")):
            section = integration.PanelSection()
        section.options = integration._select_releases([release_entry("1.0.0")])
        for installed, action in (("1.1.0", "downgrade"), ("1.0.0", "reinstall"), ("dev", "replace")):
            with self.subTest(installed=installed):
                section.info = {"installed": [["user_default", installed]]}
                self.assertEqual(section._replace_action(), f"blender_integration.{action}")

    def test_channel_change_recomputes_status_and_disabled_install_can_update(self):
        with patch.object(integration, "preferences", return_value=("blender", "beta")):
            section = integration.PanelSection()
        section.info = {"installed": [["user_default", "1.0.0-beta.2"]], "enabled": False}
        section.options = {
            "beta": (integration.Release("1.0.0-beta.2", "unused", "unused", 1),),
            "stable": (integration.Release("1.0.0", "unused", "unused", 1),),
        }
        section.status = "complete"
        with patch.object(integration, "save_preferences"), patch.object(section, "_tr", side_effect=str):
            section._set_channel("stable")
            self.assertTrue(section._installable())
            self.assertEqual(section._status_label(), "blender_integration.ready")
            section._set_channel("beta")
            self.assertFalse(section._installable())
            self.assertEqual(section._status_label(), "blender_integration.disabled")

    def test_blender_lock_failure_points_to_repository_recovery(self):
        report = 'Error: lock exists (lock is held by other session). Use "Force Unlock Repository".'
        result = SimpleNamespace(returncode=1, stdout=report)
        executable = integration.Path(sys.executable)
        with (patch.object(integration, "_executable_path", return_value=executable),
              patch.object(integration.subprocess, "run", return_value=result),
              self.assertLogs(integration.__name__, level="WARNING") as logs):
            with self.assertRaises(integration.IntegrationError) as error:
                integration._run(str(executable), ["--command", "extension", "install-file"])
        self.assertEqual(error.exception.status, "repository_locked")
        self.assertIn(report, logs.output[0])

    def test_downgrade_never_downloads_or_invokes_blender(self):
        installed = {"installed": [["user_default", "1.0.0"]]}
        older = integration.Release("1.0.0-beta.2", "unused", "unused", 1)
        with (patch.object(integration, "urlopen") as request,
              patch.object(integration, "_run") as run):
            with self.assertRaisesRegex(ValueError, "same or a newer version"):
                integration.install("blender", older, installed)
        request.assert_not_called()
        run.assert_not_called()

    def test_bad_package_or_changed_profile_never_runs_installer(self):
        version = "1.0.0-beta.2"
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("blender_manifest.toml",
                             f'id = "{integration.EXTENSION}"\nversion = "{version}"\n'
                             'blender_version_min = "5.3.0"\nblender_version_max = "5.4.0"')
        data = payload.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        for sha256, blender, reason in [("bad", [5, 3, 0], "package"),
                                         (digest, [5, 4, 0], "incompatible"),
                                         (digest, [5, 3, 0], "profile_changed")]:
            with self.subTest(reason=reason):
                published = integration.Release(version, "https://example.org/package.zip", sha256, len(data))
                expected = {"installed": [], "version": blender, "destination": "/original"}
                with (patch.object(integration, "urlopen", return_value=io.BytesIO(data)),
                      patch.object(integration, "probe", return_value={**expected, "destination": "/changed"}),
                      patch.object(integration, "_run") as run):
                    with self.assertRaises(integration.IntegrationError) as error:
                        integration.install("blender", published, expected)
                self.assertEqual(error.exception.status, reason)
                run.assert_not_called()

        expected = {"installed": [], "version": [5, 3, 0], "destination": "/original", "config": "/config"}
        disabled = {**expected, "installed": [["user_default", version]], "enabled": False}
        published = integration.Release(version, "https://example.org/package.zip", digest, len(data))
        with (patch.object(integration, "urlopen", return_value=io.BytesIO(data)),
              patch.object(integration, "probe", side_effect=[expected, disabled]),
              patch.object(integration, "_run", return_value="installer output") as run):
            with self.assertRaisesRegex(ValueError, "did not report the requested extension"):
                integration.install("blender", published, expected)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()

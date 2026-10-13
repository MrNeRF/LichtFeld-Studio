# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test the TIFF mirror with real vcpkg downloads against a local HTTP server."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[2]
VCPKG_ROOT = Path(os.environ.get("VCPKG_ROOT", ROOT.parent / "vcpkg"))
VCPKG = VCPKG_ROOT / ("vcpkg.exe" if os.name == "nt" else "vcpkg")
CMAKE = shutil.which("cmake")
PRIMARY = b"verified GitLab archive fixture"
MIRROR = b"verified release archive fixture with different bytes"


@unittest.skipUnless(CMAKE and VCPKG.is_file(), "CMake and VCPKG_ROOT are required")
class TiffDownloadFallbackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.downloads = self.root / "downloads"
        self.downloads.mkdir()
        self.responses = {"/primary": PRIMARY, "/mirror": MIRROR}
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append(self.path)
                data = owner.responses[self.path]
                self.send_response(503 if data is None else 200)
                self.end_headers()
                if data is not None:
                    self.wfile.write(data)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def run_download(self, offline=False):
        url = f"http://127.0.0.1:{self.server.server_port}"
        script = self.root / "test.cmake"
        # Adapt the two vcpkg helpers that normally receive their context from
        # ports.cmake. The downloader and its checksum/cache logic are real.
        script.write_text(f'''
set(DOWNLOADS "{self.downloads.as_posix()}")
set(_VCPKG_NO_DOWNLOADS {"ON" if offline else "OFF"})
include("{VCPKG_ROOT.as_posix()}/scripts/cmake/z_vcpkg_function_arguments.cmake")
include("{VCPKG_ROOT.as_posix()}/scripts/cmake/vcpkg_list.cmake")
function(vcpkg_execute_in_download_mode)
    cmake_parse_arguments(PARSE_ARGV 0 run "" "RESULT_VARIABLE" "")
    execute_process(${{run_UNPARSED_ARGUMENTS}} RESULT_VARIABLE result)
    set(${{run_RESULT_VARIABLE}} "${{result}}" PARENT_SCOPE)
endfunction()
include("{VCPKG_ROOT.as_posix()}/scripts/cmake/vcpkg_download_distfile.cmake")
function(vcpkg_download_distfile out_var)
    z_vcpkg_download_distfile(path ${{ARGN}})
    set(${{out_var}} "${{path}}" PARENT_SCOPE)
endfunction()
include("{ROOT.as_posix()}/.github/overlays/tiff/download_with_fallback.cmake")
lfs_download_with_fallback(archive
    PRIMARY_URL "{url}/primary" PRIMARY_FILENAME "primary.tar.gz"
    PRIMARY_SHA512 "{hashlib.sha512(PRIMARY).hexdigest()}"
    MIRROR_URL "{url}/mirror" MIRROR_FILENAME "mirror.tar.gz"
    MIRROR_SHA512 "{hashlib.sha512(MIRROR).hexdigest()}")
file(WRITE "{self.root.as_posix()}/selected.txt" "${{archive}}")
''', encoding="utf-8")
        env = dict(os.environ, VCPKG_COMMAND=str(VCPKG), X_VCPKG_ASSET_SOURCES="clear")
        result = subprocess.run(
            [CMAKE, "-P", str(script)], env=env, text=True, capture_output=True, timeout=60
        )
        selected = self.root / "selected.txt"
        return result, Path(selected.read_text()).name if selected.exists() else None

    def assert_selected(self, name, offline=False):
        result, selected = self.run_download(offline)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(selected, name)

    def test_primary_success_does_not_contact_mirror(self):
        self.assert_selected("primary.tar.gz")
        self.assertEqual(self.requests, ["/primary"])

    def test_http_failure_uses_mirror(self):
        self.responses["/primary"] = None
        self.assert_selected("mirror.tar.gz")
        self.assertIn("/mirror", self.requests)

    def test_wrong_primary_checksum_uses_mirror(self):
        self.responses["/primary"] = b"corrupted"
        self.assert_selected("mirror.tar.gz")

    def test_wrong_mirror_checksum_fails(self):
        self.responses.update({"/primary": None, "/mirror": b"corrupted"})
        result, selected = self.run_download()
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(selected)
        self.assertIn("/mirror", self.requests)
        self.assertIn("Download failed", result.stderr)

    def test_both_servers_unavailable_fails(self):
        self.responses.update({"/primary": None, "/mirror": None})
        result, selected = self.run_download()
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(selected)
        self.assertIn("/mirror", self.requests)
        self.assertIn("Download failed", result.stderr)

    def test_cached_primary_works_offline(self):
        (self.downloads / "primary.tar.gz").write_bytes(PRIMARY)
        self.assert_selected("primary.tar.gz", offline=True)
        self.assertEqual(self.requests, [])

    def test_cached_mirror_works_offline_with_invalid_primary(self):
        (self.downloads / "primary.tar.gz").write_bytes(b"corrupted")
        (self.downloads / "mirror.tar.gz").write_bytes(MIRROR)
        self.assert_selected("mirror.tar.gz", offline=True)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()

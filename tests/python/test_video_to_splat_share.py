# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The free share link: a token-guarded HTTP server for one package folder."""
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import share  # noqa: E402


@pytest.fixture
def package(tmp_path):
    folder = tmp_path / "package"
    folder.mkdir()
    (folder / "viewer.html").write_text("<html><body>viewer</body></html>", encoding="utf-8")
    (folder / "splat.spz").write_bytes(b"\x00\x01\x02")
    (folder / "mesh_textured.glb").write_bytes(b"glTF")
    (tmp_path / "secret.txt").write_text("outside")
    return folder


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def test_share_serves_the_viewer_and_the_files_under_a_token(package):
    server = share.ShareServer(package, host="127.0.0.1").start()
    try:
        base = server.url("127.0.0.1")
        assert base.startswith("http://127.0.0.1:%d/" % server.port) and len(server.token) >= 16
        status, kind, body = _get(base)
        assert status == 200 and kind.startswith("text/html") and b"viewer" in body
        status, kind, body = _get(base + "splat.spz")
        assert status == 200 and body == b"\x00\x01\x02"
        status, kind, body = _get(base + "mesh_textured.glb")
        assert kind == "model/gltf-binary"
        status, kind, body = _get(server.files_url("127.0.0.1"))
        assert b"splat.spz" in body and b"mesh_textured.glb" in body and b"Open the viewer" in body
    finally:
        server.stop()


def test_share_refuses_other_tokens_and_paths_outside_the_package(package):
    server = share.ShareServer(package, host="127.0.0.1").start()
    try:
        root = "http://127.0.0.1:%d/" % server.port
        for path in ("", "wrong-token/", server.token + "/../secret.txt", server.token + "/%2e%2e/secret.txt"):
            with pytest.raises(urllib.error.HTTPError) as err:
                _get(root + path)
            assert err.value.code == 404, path
    finally:
        server.stop()


def test_module_share_replaces_the_previous_server_and_stops(package, tmp_path):
    first = share.share(package, lan=False)
    assert first["sharing"] is True and first["url"].endswith("/") and first["package"] == str(package)
    other = tmp_path / "other"
    other.mkdir()
    (other / "viewer.html").write_text("other")
    second = share.share(other, lan=False)
    assert second["lan"] is False and second["url"].startswith("http://127.0.0.1:")
    assert second["port"] != 0 and share.status()["package"] == str(other)
    assert _get("http://127.0.0.1:%d/%s/" % (second["port"], second["url"].rstrip("/").rsplit("/", 1)[1]))[2] == b"other"
    assert share.stop() == {"sharing": False} and share.status() == {"sharing": False}
    with pytest.raises(FileNotFoundError):
        share.share(tmp_path / "missing", lan=False)

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""A free share link for a package: a small HTTP server inside the app, no account needed.

The cloud service hands out a link to a hosted viewer. Here the link points at this machine:
``http://<lan address>:<port>/<random token>/`` opens the package's viewer.html, and the files of
the package (splat, meshes, point clouds, camera poses) are downloadable next to it. The token keeps
casual visitors out; the server only ever serves files inside the one package folder, and it stops
with the app or on request.
"""
from __future__ import annotations

import html
import secrets
import socket
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

VIEWER = "viewer.html"
CONTENT_TYPES = {".ply": "application/octet-stream", ".spz": "application/octet-stream", ".sog": "application/octet-stream",
                 ".glb": "model/gltf-binary", ".usdz": "model/vnd.usdz+zip", ".json": "application/json",
                 ".png": "image/png", ".html": "text/html; charset=utf-8"}


def lan_address() -> str:
    """The address other machines on the network reach this one by (never actually sends anything)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


class _Handler(SimpleHTTPRequestHandler):
    server: "ShareServer"

    def log_message(self, _format, *_args):  # keep the app console quiet
        pass

    def _resolve(self) -> Optional[Path]:
        parts = [unquote(p) for p in self.path.split("?", 1)[0].split("/") if p]
        if not parts or parts[0] != self.server.token:
            return None
        rest = parts[1:]
        self.listing_requested = rest == ["files"]  # <token>/files lists the downloads
        if self.listing_requested:
            rest = []
        if any(p in ("..", ".") for p in rest):
            return None
        target = self.server.root.joinpath(*rest) if rest else self.server.root
        try:
            target.resolve().relative_to(self.server.root.resolve())
        except ValueError:
            return None
        return target

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        target = self._resolve()
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if target.is_dir():
            viewer = target / VIEWER
            if viewer.is_file() and not self.listing_requested:
                self._send(viewer.read_bytes(), CONTENT_TYPES[".html"])
                return
            self._send(self._listing(target).encode("utf-8"), CONTENT_TYPES[".html"])
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(target.read_bytes(), CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream"))

    def _listing(self, folder: Path) -> str:
        rows = "".join('<li><a href="%s">%s</a> (%d bytes)</li>' % (quote(p.name), html.escape(p.name), p.stat().st_size)
                       for p in sorted(folder.iterdir()) if p.is_file())
        return ("<!doctype html><html><head><meta charset='utf-8'><title>LichtFeld package</title></head><body>"
                "<h1>Package files</h1><p><a href='./'>Open the viewer</a></p><ul>%s</ul></body></html>" % rows)

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class ShareServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: Path, host: str = "0.0.0.0", port: int = 0, token: Optional[str] = None):
        self.root = Path(root)
        self.token = token or secrets.token_urlsafe(12)
        super().__init__((host, port), _Handler)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ShareServer":
        self._thread = threading.Thread(target=self.serve_forever, daemon=True, name="VideoToSplatShare")
        self._thread.start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def url(self, address: Optional[str] = None) -> str:
        return "http://%s:%d/%s/" % (address or lan_address(), self.port, self.token)

    def files_url(self, address: Optional[str] = None) -> str:
        return self.url(address) + "files"


_current: Optional[ShareServer] = None
_lock = threading.Lock()


def share(package: Path, port: int = 0, lan: bool = True) -> dict:
    """Serve ``package`` and return {url, files_url, port, package}; a previous share is stopped first.

    ``lan=True`` listens on every interface so other devices can open the link (Windows asks once to allow
    it through the firewall); ``lan=False`` stays on this machine only."""
    global _current
    package = Path(package)
    if not package.is_dir():
        raise FileNotFoundError(str(package))
    with _lock:
        if _current is not None:
            _current.stop()
        _current = ShareServer(package, host="0.0.0.0" if lan else "127.0.0.1", port=port).start()
        return _status_locked()


def stop() -> dict:
    global _current
    with _lock:
        if _current is not None:
            _current.stop()
            _current = None
    return {"sharing": False}


def status() -> dict:
    with _lock:
        return _status_locked()


def _status_locked() -> dict:
    """Caller holds _lock (threading.Lock is not reentrant)."""
    if _current is None:
        return {"sharing": False}
    address = None if _current.server_address[0] == "0.0.0.0" else "127.0.0.1"
    return {"sharing": True, "url": _current.url(address), "files_url": _current.files_url(address),
            "port": _current.port, "package": str(_current.root), "lan": address is None}

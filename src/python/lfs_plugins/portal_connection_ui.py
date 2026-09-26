# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared presentation helpers for Portal connection controls."""

import os
import struct
import zlib
from pathlib import Path

from ._qrcodegen import QrCode


def connection_state(account, *, busy=False):
    if account is None:
        return "not_connected"
    if account.linking or account.disconnecting or busy:
        return "busy"
    if account.signed_in:
        return "connected"
    if getattr(account, "authorized", False):
        return "switched_off"
    return "not_connected"


def connection_action(state):
    """Return the connection action and its shared label key for a state."""
    return {
        "busy": (None, "portal.status.busy"),
        "connected": (None, "portal.status.connected_as"),
        "switched_off": ("turn_on", "portal.status.turn_on"),
        "not_connected": ("connect", "portal.status.connect"),
    }.get(state, ("connect", "portal.status.connect"))


def qr_image_path(payload, directory):
    """Write a PNG QR image for an approval URL and return its absolute path."""
    qr = QrCode.encode_text(payload, QrCode.Ecc.MEDIUM)
    scale = 5
    quiet = 4
    size = (qr.get_size() + quiet * 2) * scale
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        module_y = y // scale - quiet
        for x in range(size):
            module_x = x // scale - quiet
            dark = (0 <= module_x < qr.get_size() and 0 <= module_y < qr.get_size()
                    and qr.get_module(module_x, module_y))
            value = 0 if dark else 255
            raw.extend((value, value, value))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))
    path = Path(directory) / "portal-approval-qr.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(png)
    os.replace(temporary, path)
    return str(path.resolve())

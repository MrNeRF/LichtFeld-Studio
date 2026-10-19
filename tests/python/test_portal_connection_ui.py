# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

import struct
from pathlib import Path
from types import SimpleNamespace

from lfs_plugins import portal_connection_ui


def test_connection_action_maps_every_account_state():
    states = (
        (SimpleNamespace(signed_in=False, authorized=False, linking=False, disconnecting=False),
         "not_connected", "connect", "portal.status.connect"),
        (SimpleNamespace(signed_in=False, authorized=True, linking=False, disconnecting=False),
         "switched_off", "turn_on", "portal.status.turn_on"),
        (SimpleNamespace(signed_in=False, authorized=False, linking=True, disconnecting=False),
         "busy", None, "portal.status.busy"),
        (SimpleNamespace(signed_in=True, authorized=True, linking=False, disconnecting=False),
         "connected", None, "portal.status.connected_as"),
    )
    for account, expected_state, expected_action, expected_label in states:
        state = portal_connection_ui.connection_state(account)
        assert state == expected_state
        assert portal_connection_ui.connection_action(state) == (expected_action, expected_label)


def test_approval_qr_encodes_the_prefilled_verification_url(monkeypatch, tmp_path):
    url = "https://portal.example/link/?code=KXQ4-7RTM"
    encoded = []
    original = portal_connection_ui.QrCode.encode_text

    def capture(payload, error_correction):
        encoded.append(payload)
        return original(payload, error_correction)

    monkeypatch.setattr(portal_connection_ui.QrCode, "encode_text", capture)
    image = portal_connection_ui.qr_image_path(url, tmp_path)

    assert encoded == [url]
    assert image.endswith("portal-approval-qr.png")
    raw = Path(image).read_bytes()
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack_from(">II", raw, 16)
    assert width == height and width > 100

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Preview geometry must follow RmlUI's deferred chrome layout."""

import struct
import zlib
from types import SimpleNamespace

import pytest

from test_rmlui_image_sources import panel_modules, _DocumentStub, _ElementStub


@pytest.mark.parametrize("field", ["_show_info", "_show_filmstrip"])
def test_chrome_toggle_remeasures_after_layout_then_stops(panel_modules, monkeypatch, tmp_path, field):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    path = tmp_path / 'generated.png'
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data)))
    pixels = (b'\0' + b'\xdc\x1e\x1e' * 640) * 480
    path.write_bytes(b'\x89PNG\r\n\x1a\n'
                     + chunk(b'IHDR', struct.pack('>2I5B', 640, 480, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))
    panel._image_paths = [path]
    panel._get_image_info = lambda _: (640, 480, 3)
    viewport = _ElementStub()
    viewport.client_width = 612
    viewport.client_height = 542
    image = _ElementStub()
    doc = _DocumentStub({'image-viewport': viewport})
    panel._doc = doc
    requests = []
    panel._handle = SimpleNamespace(dirty=lambda _: None)
    panel._request_model_update = lambda: requests.append(1)
    for method in ['_update_localized_labels', '_update_filmstrip', '_update_sidebar',
                   '_update_nav_arrows', '_update_view_controls', '_update_status']:
        monkeypatch.setattr(panel, method, lambda *args: None)
    panel._update_main_image = lambda *args: panel._apply_zoom(image, path)
    panel.on_update(doc)
    panel.on_update(doc)
    requests.clear()
    setattr(panel, field, False)
    panel._mark_dirty()
    panel.on_update(doc)
    # RmlUI processes style and layout after the panel's update hook.
    viewport.client_width = 814
    assert len(requests) >= 2, 'new viewport bounds must get a scheduled measurement'
    panel.on_update(doc)
    assert image.properties['width'] == '640dp'
    assert image.properties['left'] == '87dp'
    count = len(requests)
    assert panel.on_update(doc) is False
    assert len(requests) == count

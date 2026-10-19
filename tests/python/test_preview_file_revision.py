# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reopening changed image sources must refresh metadata and texture identity."""

import os
import struct
import zlib

from test_rmlui_image_sources import panel_modules


def png(path, width, height, color):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    raw = (b'\0' + bytes(color) * width) * height
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def test_reopen_changed_image_refreshes_dimensions_and_both_texture_sizes(panel_modules, monkeypatch, tmp_path):
    module, _ = panel_modules
    path = tmp_path / 'frame.png'
    def info(source):
        return (*struct.unpack('>II', open(source, 'rb').read()[16:24]), 3)
    monkeypatch.setattr(module.lf.ui, 'get_image_info', info, raising=False)
    panel = module.ImagePreviewPanel()
    png(path, 320, 240, (220, 30, 30))
    panel.open([path], [], 0)
    assert panel._get_image_info(path) == (320, 240, 3)
    old_urls = [panel._make_preview_url(path, -1, thumb=size) for size in (0, 256)]
    png(path, 160, 320, (30, 30, 220))
    panel.open([path], [], 0)
    assert panel._get_image_info(path) == (160, 320, 3)
    assert all(panel._make_preview_url(path, -1, thumb=size) != url
               for size, url in zip((0, 256), old_urls))
    new_urls = [panel._make_preview_url(path, -1, thumb=size) for size in (0, 256)]
    panel.open([path], [], 0)
    assert panel._get_image_info(path) == (160, 320, 3)
    assert new_urls == [panel._make_preview_url(path, -1, thumb=size) for size in (0, 256)]


def test_unchanged_image_reuses_info_without_frame_stat(panel_modules, monkeypatch, tmp_path):
    module, _ = panel_modules
    path = tmp_path / 'frame.png'
    png(path, 16, 12, (10, 20, 30))
    calls = []
    monkeypatch.setattr(module.lf.ui, 'get_image_info', lambda _: calls.append(1) or (16, 12, 3), raising=False)
    panel = module.ImagePreviewPanel()
    panel.open([path], [], 0)
    panel._get_image_info(path)
    panel.open([path], [], 0)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'stat', lambda _: (_ for _ in ()).throw(AssertionError('unexpected stat')))
        assert panel._get_image_info(path) == (16, 12, 3)
        panel._make_preview_url(path, -1)
    assert calls == [1]


def test_atomic_replacement_with_same_size_and_mtime_refreshes_cache(panel_modules, monkeypatch, tmp_path):
    module, _ = panel_modules
    path = tmp_path / 'frame.png'
    replacement = tmp_path / 'replacement.png'
    png(path, 1, 1, (10, 20, 30))
    png(replacement, 1, 1, (40, 50, 60))
    original_bytes = path.read_bytes()
    original_stat = path.stat()
    assert replacement.stat().st_size == original_stat.st_size
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    info_calls = []
    monkeypatch.setattr(module.lf.ui, 'get_image_info',
                        lambda source: info_calls.append(source) or (1, 1, 3), raising=False)
    panel = module.ImagePreviewPanel()
    panel.open([path], [], 0)
    assert panel._get_image_info(path) == (1, 1, 3)
    old_urls = [panel._make_preview_url(path, -1, thumb=size) for size in (0, 256)]

    replacement.replace(path)
    assert path.read_bytes() != original_bytes
    assert path.stat().st_size == original_stat.st_size
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    panel.open([path], [], 0)
    assert panel._get_image_info(path) == (1, 1, 3)
    assert info_calls == [str(path), str(path)]
    assert all(panel._make_preview_url(path, -1, thumb=size) != url
               for size, url in zip((0, 256), old_urls))

    new_urls = [panel._make_preview_url(path, -1, thumb=size) for size in (0, 256)]
    panel.open([path], [], 0)
    assert panel._get_image_info(path) == (1, 1, 3)
    assert info_calls == [str(path), str(path)]
    assert new_urls == [panel._make_preview_url(path, -1, thumb=size) for size in (0, 256)]

"""Capability gating and cancellation before photo reconstruction starts."""
import threading

import pytest
from test_file_menu_recent import _load_file_menu


def _photo_items(menu):
    imports = next(item for item in menu.FileMenu().menu_items()
                   if item.get("label") == "tr:menu.file.import")
    return [item for item in imports["items"]
            if item.get("label") == "tr:menu.file.create_splat_from_photo"]


@pytest.mark.parametrize("ready", [False, True])
def test_photo_menu_requires_runtime_capability(monkeypatch, ready):
    menu = _load_file_menu(monkeypatch)
    menu.lf.create_splat_from_photo = lambda _: None
    menu.lf.apple_reframe_available = lambda: ready
    menu._probe_apple_reframe(menu.lf.apple_reframe_available)
    assert bool(_photo_items(menu)) is ready


def test_photo_menu_hidden_without_build_binding(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    menu.lf.apple_reframe_available = lambda: True
    assert not _photo_items(menu)


def test_failed_probe_hides_photo_menu(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    menu.lf.create_splat_from_photo = lambda _: None
    def failed():
        raise RuntimeError("private ABI changed")
    menu.lf.apple_reframe_available = failed
    menu.lf.log.debug = lambda _: None
    menu._probe_apple_reframe(failed)
    assert not _photo_items(menu)


def test_unsupported_operator_never_opens_dialog(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    menu.lf.apple_reframe_available = lambda: False
    menu.lf.ui.open_image_file_dialog = lambda _: pytest.fail("unsupported dialog")
    assert menu.CreateSplatFromPhotoOperator().execute(None) == {"CANCELLED"}


def test_photo_picker_cancel_does_not_create_splats(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    menu.lf.apple_reframe_available = lambda: True
    menu.lf.create_splat_from_photo = lambda _: pytest.fail("cancelled picker reconstructed")
    menu.lf.ui.open_image_file_dialog = lambda _: ""
    assert menu.CreateSplatFromPhotoOperator().execute(None) == {"CANCELLED"}


def test_photo_operator_passes_unicode_path(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    calls = []
    menu.lf.apple_reframe_available = lambda: True
    menu.lf.create_splat_from_photo = calls.append
    path = "/tmp/Foto vacanze/città 東京.heic"
    menu.lf.ui.open_image_file_dialog = lambda _: path
    assert menu.CreateSplatFromPhotoOperator().execute(None) == {"FINISHED"}
    assert calls == [path]


def test_menu_never_waits_for_probe_or_rebuilds_operator_identity(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def blocked_probe():
        entered.set()
        release.wait(5)
        return True
    original = menu._probe_apple_reframe
    def probe(available):
        try:
            original(available)
        finally:
            finished.set()
    monkeypatch.setattr(menu, "_probe_apple_reframe", probe)
    menu.lf.apple_reframe_available = blocked_probe
    menu.lf.create_splat_from_photo = lambda _: None
    try:
        before = menu.FileMenu().menu_items()
        assert entered.wait(1)
        assert not _photo_items(menu)
        release.set()
        assert finished.wait(1)
        after = menu.FileMenu().menu_items()
        assert _photo_items(menu)
        def leaves(items):
            for item in items:
                if item.get("type") == "submenu":
                    yield from leaves(item["items"])
                elif item.get("type") != "separator":
                    yield item
        before_items, after_items = list(leaves(before)), list(leaves(after))
        assert [item["operator_id"] for item in before_items if item["type"] == "operator"] == [
            item["operator_id"] for item in after_items if item["type"] == "operator"
            and item["operator_id"] != menu.CreateSplatFromPhotoOperator._class_id()]
        # Schema callbacks count only action/toggle entries; operators dispatch by id.
        assert [(item.get("label"), item.get("action_id")) for item in before_items
                if item["type"] != "operator"] == [
            (item.get("label"), item.get("action_id")) for item in after_items
            if item["type"] != "operator"]
    finally:
        release.set()


def test_execute_does_not_probe_again(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    menu.lf.apple_reframe_available = lambda: pytest.fail("probe on UI thread")
    calls = []
    menu.lf.create_splat_from_photo = calls.append
    menu.lf.ui.open_image_file_dialog = lambda _: "/tmp/photo.jpg"
    assert menu.CreateSplatFromPhotoOperator().execute(None) == {"FINISHED"}
    assert calls == ["/tmp/photo.jpg"]

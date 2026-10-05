"""Capability gating and cancellation before photo reconstruction starts."""
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
    assert not _photo_items(menu)


def test_unsupported_operator_never_opens_dialog(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    menu.lf.apple_reframe_available = lambda: False
    menu.lf.create_splat_from_photo = lambda _: pytest.fail("unsupported reconstruction")
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

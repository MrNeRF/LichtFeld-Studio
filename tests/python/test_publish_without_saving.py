from importlib import import_module
from types import SimpleNamespace

import pytest

from test_file_menu_recent import _load_file_menu


def test_unsaved_scene_enables_publish_and_offers_three_choices(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    menu.lf.has_scene = lambda: True
    menu.lf.project_poll_write = lambda: {"path": ""}
    entry = next(item for item in menu.FileMenu().menu_items()
                 if item.get("label") == "tr:menu.file.publish_to_gallery")
    assert entry["enabled"]
    entry["callback"]()
    assert len(menu.lf.confirm_dialogs) == 1
    assert menu.lf.confirm_dialogs[0][2] == [
        "tr:menu.file.save_and_publish", "tr:menu.file.publish_without_saving", "tr:common.cancel"]


@pytest.mark.parametrize("choice", ["save", "unlinked", "cancel"])
def test_unsaved_publish_choice(monkeypatch, choice):
    menu = _load_file_menu(monkeypatch)
    menu.lf.has_scene = lambda: True
    menu.lf.project_poll_write = lambda: {"path": ""}
    calls = []
    monkeypatch.setattr(menu, "_open_unlinked_gallery_review", lambda: calls.append("review"), raising=False)
    menu.lf.project_save_as = lambda path: calls.append("save") or True
    menu._publish_current_project_to_gallery()
    buttons = menu.lf.confirm_dialogs[0][2]
    menu.lf.confirm_dialogs[0][3](buttons[{"save": 0, "unlinked": 1, "cancel": 2}[choice]])
    assert calls == {"save": ["save"], "unlinked": ["review"], "cancel": []}[choice]


def test_cancelled_save_as_opens_no_review(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    menu.lf.has_scene = lambda: True
    menu.lf.project_poll_write = lambda: {"path": ""}
    calls = []
    menu.lf.project_save_as = lambda path: calls.append("save") or False
    monkeypatch.setattr(menu, "_open_unlinked_gallery_review", lambda: calls.append("review"), raising=False)
    menu._publish_current_project_to_gallery()
    menu.lf.confirm_dialogs[0][3](menu.lf.confirm_dialogs[0][2][0])
    assert calls == ["save"]
    assert menu.lf.message_dialogs == []


def test_save_as_completion_opens_saved_review(monkeypatch, tmp_path):
    menu = _load_file_menu(monkeypatch)
    path = tmp_path / "saved.licht"
    path.write_bytes(b"saved")
    polls = iter([{"running": True, "path": ""}, {"running": False, "path": str(path)}])
    menu.lf.project_save_as = lambda _: True
    menu.lf.project_poll_write = lambda: next(polls)
    calls = []
    monkeypatch.setattr(menu.threading, "Timer", lambda _delay, callback: SimpleNamespace(
        start=lambda: callback(), daemon=True))
    menu.lf.ui.schedule_on_ui_thread = lambda callback: callback()
    monkeypatch.setattr(menu, "_publish_current_project_to_gallery", lambda: calls.append("review"))
    menu._save_then_publish()
    assert calls == ["review"]


def test_file_menu_can_update_a_live_snapshot_without_card_action(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    assert menu._file_menu_publish_action(None, {"liveSnapshot": True}) == "update"
    assert menu._file_menu_publish_action(None, {"liveSnapshot": False}) is None
    assert menu._file_menu_publish_action({"id": "check"}, {"liveSnapshot": True}) == "check"

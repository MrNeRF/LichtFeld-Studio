# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys


def _install_lf_stub(monkeypatch):
    lf_stub = ModuleType("lichtfeld")
    lf_stub.__path__ = []
    ui_stub = ModuleType("lichtfeld.ui")
    ui_stub.key = SimpleNamespace(
        SPACE=32,
        ESCAPE=256,
        ENTER=257,
        TAB=258,
        BACKSPACE=259,
        INSERT=260,
        DELETE=261,
        RIGHT=262,
        LEFT=263,
        DOWN=264,
        UP=265,
        A=65,
        NUM_0=48,
        F1=290,
        KP_0=320,
        KP_1=321,
        KP_2=322,
        KP_3=323,
        KP_4=324,
        KP_5=325,
        KP_6=326,
        KP_7=327,
        KP_8=328,
        KP_9=329,
        KP_ENTER=335,
    )
    ui_stub.ModalEventType = SimpleNamespace(MouseButton=0, MouseMove=1, Scroll=2, Key=3)
    ui_stub.mouse = SimpleNamespace(LEFT=0, RIGHT=1, MIDDLE=2)
    ui_stub.action = SimpleNamespace(PRESS=1, RELEASE=0)
    lf_stub.ui = ui_stub
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    monkeypatch.setitem(sys.modules, "lichtfeld.ui", ui_stub)


def _import_event_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.event", None)
    _install_lf_stub(monkeypatch)
    return import_module("lfs_plugins.event")


def test_event_wrapper_maps_numpad_digits_like_number_row(monkeypatch):
    event_module = _import_event_module(monkeypatch)
    raw_event = SimpleNamespace(
        type=event_module.Event.TYPE_KEY,
        key=event_module.ui.key.KP_7,
        action=event_module.ui.action.PRESS,
    )

    wrapped = event_module.Event(raw_event)

    assert wrapped.type_str == "7"
    assert wrapped.value_str == "PRESS"


def test_event_wrapper_maps_numpad_enter_like_enter(monkeypatch):
    event_module = _import_event_module(monkeypatch)
    raw_event = SimpleNamespace(
        type=event_module.Event.TYPE_KEY,
        key=event_module.ui.key.KP_ENTER,
        action=event_module.ui.action.RELEASE,
    )

    wrapped = event_module.Event(raw_event)

    assert wrapped.type_str == "ENTER"
    assert wrapped.value_str == "RELEASE"

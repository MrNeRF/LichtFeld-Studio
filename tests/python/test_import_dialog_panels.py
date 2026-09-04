# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the retained checkpoint resume dialog."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def import_dialog_module(monkeypatch, tmp_path):
    source_python = Path(__file__).parents[2] / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    for name in list(sys.modules):
        if name == "lfs_plugins" or name.startswith("lfs_plugins."):
            sys.modules.pop(name, None)

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    checkpoint = tmp_path / "scene.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    state = SimpleNamespace(
        enabled=[],
        loads=[],
        header=SimpleNamespace(iteration=12, num_gaussians=99),
        params=SimpleNamespace(dataset_path=str(dataset), output_path=str(tmp_path / "output")),
    )
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(CONTENT="CONTENT"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        get_current_language=lambda: "en",
        set_panel_enabled=lambda panel_id, enabled: state.enabled.append((panel_id, enabled)),
        tr=lambda key: key,
        open_dataset_folder_dialog=lambda: str(dataset),
        confirm_dialog=lambda *_args: None,
    )
    lf_stub.read_checkpoint_header = lambda _path: state.header
    lf_stub.read_checkpoint_params = lambda _path: state.params
    lf_stub.load_checkpoint_for_training = lambda *args: state.loads.append(args)
    lf_stub.project_is_dirty = lambda: False
    lf_stub.project_has_path = lambda: True
    lf_stub.is_training_active = lambda: False
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return import_module("lfs_plugins.import_panels"), state, checkpoint, dataset


def test_resume_checkpoint_panel_loads(import_dialog_module):
    module, state, checkpoint, dataset = import_dialog_module
    panel = module.ResumeCheckpointPanel()

    assert panel.show(str(checkpoint)) is True
    panel._on_do_load()

    assert state.loads == [
        (str(checkpoint), str(dataset), str(dataset.parent / "output"))
    ]
    assert state.enabled == [("lfs.resume_checkpoint", True), ("lfs.resume_checkpoint", False)]


def test_import_panel_registration_keeps_only_new_project_and_resume():
    source = (Path(__file__).parents[2] / "src" / "python" / "lfs_plugins" / "import_panels.py").read_text()
    assert "NewProjectPanel" in source
    assert "lfs.new_project" in source


def test_new_project_typed_source_path_validates(import_dialog_module, monkeypatch):
    module, _state, _checkpoint, dataset = import_dialog_module
    panel = module.NewProjectPanel()
    panel._dialog_mounted = True
    panel._set_name("untitled")

    panel._handle = SimpleNamespace(
        dirty=lambda _field: None,
        dirty_all=lambda: None,
        request_update=lambda: None,
    )
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    timers = []

    class _Timer:
        def __init__(self, _delay, callback):
            timers.append(callback)

        def start(self):
            pass

    monkeypatch.setattr(module.threading, "Timer", _Timer)
    browse_calls = []
    monkeypatch.setattr(
        module.lf.ui,
        "open_dataset_folder_dialog",
        lambda: browse_calls.append(True),
    )
    module.lf.is_dataset_path = lambda path: str(path) == str(dataset)
    module.lf.detect_dataset_info = lambda _path: SimpleNamespace(
        sparse_path="",
        has_masks=False,
        image_count=0,
        mask_count=0,
    )

    panel._set_source_path(str(dataset))
    assert panel._source_kind == "checking"
    assert len(timers) == 1

    now[0] += 0.31
    timers.pop()()
    panel.on_update(None)

    assert panel._source_kind == "dataset"
    assert panel._can_create() is True
    assert browse_calls == []

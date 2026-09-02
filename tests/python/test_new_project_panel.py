# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused behavior tests for the New Project panel."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


class _Handle:
    def dirty(self, _name):
        pass

    def dirty_all(self):
        pass


@pytest.fixture
def new_project_module(monkeypatch, tmp_path):
    source_python = Path(__file__).parents[2] / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    for name in list(sys.modules):
        if name == "lfs_plugins" or name.startswith("lfs_plugins."):
            sys.modules.pop(name, None)

    location = tmp_path / "projects"
    location.mkdir()
    dataset = tmp_path / "garden"
    dataset.mkdir()
    splat = tmp_path / "model.ply"
    splat.write_bytes(b"ply")
    info = SimpleNamespace(
        base_path=dataset,
        images_path=dataset / "images",
        sparse_path=dataset / "sparse",
        masks_path=dataset / "masks",
        has_masks=False,
        image_count=4,
        mask_count=0,
    )
    state = SimpleNamespace(
        location=location,
        dataset=dataset,
        splat=splat,
        info=info,
        enabled=[],
        prompts=[],
        calls=[],
        dirty=False,
        training=False,
        embed_default=False,
    )

    def confirm(title, message, buttons, callback=None):
        state.prompts.append((title, message, buttons, callback))

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(CONTENT="CONTENT"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        get_current_language=lambda: "en",
        get_project_location=lambda: str(state.location),
        get_default_project_location=lambda: str(location),
        set_panel_enabled=lambda panel_id, enabled: state.enabled.append((panel_id, enabled)),
        tr=lambda key: key,
        confirm_dialog=confirm,
        open_dataset_folder_dialog=lambda: str(dataset),
        open_ply_file_dialog=lambda _start="": str(splat),
        get_embed_dataset_by_default=lambda: state.embed_default,
    )
    lf_stub.is_dataset_path = lambda path: str(path) == str(dataset)
    lf_stub.detect_dataset_info = lambda _path: info
    lf_stub.optimization_params = lambda: None
    lf_stub.project_is_dirty = lambda: state.dirty
    lf_stub.project_has_path = lambda: True
    lf_stub.is_training_active = lambda: state.training
    lf_stub.project_create = lambda *args, **kwargs: state.calls.append(("create", args, kwargs))
    lf_stub.load_file = lambda *args, **kwargs: state.calls.append(("load", args, kwargs))
    lf_stub.project_embed_dataset = lambda: state.calls.append(("embed", (), {}))
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return import_module("lfs_plugins.import_panels"), state


def _panel(module):
    panel = module.NewProjectPanel()
    panel._handle = _Handle()
    return panel


def test_dataset_create_emits_project_before_load_without_output_path(new_project_module):
    module, state = new_project_module
    panel = _panel(module)

    assert panel.show(str(state.dataset)) is True
    panel._on_do_create()

    assert [call[0] for call in state.calls] == ["create", "load"]
    assert state.calls[0][1][0].endswith("garden.licht")
    assert state.calls[1][2]["is_dataset"] is True
    assert "output_path" not in state.calls[1][2]


def test_splat_and_blank_create(new_project_module):
    module, state = new_project_module
    panel = _panel(module)
    panel.show(str(state.splat))
    panel._on_do_create()
    assert len(state.calls) == 2
    assert state.calls[1][2] == {"path": str(state.splat), "is_dataset": False, "discard_changes": True}

    state.calls.clear()
    panel.show("")
    panel._on_do_create()
    assert len(state.calls) == 1
    assert state.calls[0][0] == "create"


def test_dataset_checkbox_defaults_from_preference_and_embeds_after_load(new_project_module):
    module, state = new_project_module
    state.embed_default = True
    panel = _panel(module)

    assert panel.show(str(state.dataset)) is True
    assert panel._source_kind == "dataset"
    assert panel._embed_dataset is True
    panel._on_do_create()
    assert [call[0] for call in state.calls] == ["create", "load", "embed"]

    state.calls.clear()
    panel.show(str(state.splat))
    assert panel._source_kind == "splat"
    assert panel._embed_dataset is True
    panel._on_do_create()
    assert [call[0] for call in state.calls] == ["create", "load"]


def test_create_prompts_only_when_pressed_and_propagates_stop_training(new_project_module):
    module, state = new_project_module
    panel = _panel(module)
    state.dirty = True
    state.training = True
    panel.show("")
    assert state.prompts == []
    panel._on_do_create()
    assert len(state.prompts) == 1
    state.prompts[0][3]("unsaved_work.continue_without_saving")
    assert len(state.prompts) == 2
    state.prompts[1][3]("common.yes")
    assert state.calls[0][2]["stop_training"] is True


@pytest.mark.parametrize("name", ["", "bad/name", "bad?name", "bad.", "CON", "Lpt1.txt"])
def test_name_validation_gates_creation(new_project_module, name):
    module, _state = new_project_module
    panel = _panel(module)
    panel.show("")
    panel._set_name(name)
    assert panel._name_is_valid() is False
    assert panel._can_create() is False


def test_exists_check_and_dedupe(new_project_module):
    module, state = new_project_module
    (state.location / "garden.licht").write_bytes(b"existing")
    panel = _panel(module)
    panel.show(str(state.dataset))
    assert panel._name == "garden-2"
    panel._set_name("garden")
    assert panel._target_exists() is True
    assert panel._can_create() is False


def test_min_track_length_and_keyboard_boundaries(new_project_module):
    module, state = new_project_module
    sparse = state.dataset / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    panel = _panel(module)
    panel.show(str(state.dataset))
    assert panel._show_min_track_length() is True
    panel._set_min_track_length_str("4")
    panel._on_min_track_length_step(args=["1"])
    assert panel._min_track_length == 5
    panel._set_init_path("seed.ply")
    assert panel._show_min_track_length_warning() is True
    panel._on_do_cancel()
    assert state.enabled[-1] == ("lfs.new_project", False)

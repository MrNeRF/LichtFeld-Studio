# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for retained dataset and checkpoint import dialogs."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch, tmp_path):
    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_dir = dataset_dir / "output"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "scene.ckpt"
    dataset_infos = {
        str(dataset_dir): SimpleNamespace(
            base_path=dataset_dir,
            images_path=dataset_dir / "images",
            sparse_path=dataset_dir / "sparse",
            masks_path=dataset_dir / "masks",
            has_masks=True,
            image_count=24,
            mask_count=24,
        )
    }

    state = SimpleNamespace(
        language=["en"],
        panel_enabled_calls=[],
        load_file_calls=[],
        load_checkpoint_calls=[],
        dataset_browse_path=str(tmp_path / "dataset_browse"),
        output_browse_path=str(tmp_path / "output_browse"),
        init_browse_path=str(tmp_path / "seed.ply"),
        dataset_info=dataset_infos[str(dataset_dir)],
        dataset_infos=dataset_infos,
        checkpoint_header=SimpleNamespace(iteration=128, num_gaussians=4096),
        checkpoint_params=SimpleNamespace(
            dataset_path=str(dataset_dir),
            output_path=str(output_dir),
        ),
        checkpoint_path=str(checkpoint_path),
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        get_current_language=lambda: state.language[0],
        set_panel_enabled=lambda panel_id, enabled: state.panel_enabled_calls.append((panel_id, enabled)),
        open_dataset_folder_dialog=lambda: state.output_browse_path,
        open_ply_file_dialog=lambda _start_dir="": state.init_browse_path,
    )
    lf_stub.detect_dataset_info = lambda path: state.dataset_infos[str(path)]
    lf_stub.is_dataset_path = lambda path: str(path) in state.dataset_infos
    lf_stub.optimization_params = lambda: None
    lf_stub.load_file = lambda path, is_dataset=False, output_path="", init_path="", centralize_dataset="off", max_width=None: state.load_file_calls.append(
        {
            "path": path,
            "is_dataset": is_dataset,
            "output_path": output_path,
            "init_path": init_path,
            "centralize_dataset": centralize_dataset,
            "max_width": max_width,
        }
    )
    lf_stub.read_checkpoint_header = lambda _path: state.checkpoint_header
    lf_stub.read_checkpoint_params = lambda _path: state.checkpoint_params
    lf_stub.load_checkpoint_for_training = lambda checkpoint_path, dataset_path, output_path: state.load_checkpoint_calls.append(
        {
            "checkpoint_path": checkpoint_path,
            "dataset_path": dataset_path,
            "output_path": output_path,
        }
    )

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


@pytest.fixture
def import_dialog_module(monkeypatch, tmp_path):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.import_panels", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch, tmp_path)
    module = import_module("lfs_plugins.import_panels")
    return module, state


class _HandleStub:
    def __init__(self):
        self.dirty_fields = []
        self.dirty_all_calls = 0

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_all_calls += 1


class _ElementStub:
    def __init__(self):
        self.listeners = {}

    def add_event_listener(self, event, callback):
        self.listeners[event] = callback


class _EventStub:
    def __init__(self, key_identifier):
        self._key_identifier = key_identifier
        self.propagation_stopped = False

    def get_parameter(self, name, default=""):
        if name == "key_identifier":
            return str(self._key_identifier)
        return default

    def stop_propagation(self):
        self.propagation_stopped = True


class _DocumentStub:
    def __init__(self):
        self.listeners = {}
        self.close_btn = _ElementStub()

    def add_event_listener(self, event, callback):
        self.listeners[event] = callback

    def get_element_by_id(self, element_id):
        if element_id == "close-btn":
            return self.close_btn
        return None

    def query_selector_all(self, _selector):
        return []


def test_dataset_import_panel_show_and_load(import_dialog_module):
    module, state = import_dialog_module
    panel = module.DatasetImportPanel()
    panel._handle = _HandleStub()

    assert panel.show(str(state.dataset_info.base_path)) is True
    assert panel._dataset_path == str(state.dataset_info.base_path)
    assert panel._output_path == str(Path(state.dataset_info.base_path) / "output")
    assert panel._init_path == ""
    assert state.panel_enabled_calls == [("lfs.dataset_import", True)]

    state.output_browse_path = "/tmp/custom_output"
    panel._on_browse_output()
    state.init_browse_path = "/tmp/seed_points.ply"
    panel._on_browse_init()
    panel._on_do_load()

    assert state.load_file_calls == [
        {
            "path": str(state.dataset_info.base_path),
            "is_dataset": True,
            "output_path": "/tmp/custom_output",
            "init_path": "/tmp/seed_points.ply",
            "centralize_dataset": "off",
            "max_width": 3840,
        }
    ]
    assert state.panel_enabled_calls[-1] == ("lfs.dataset_import", False)


def test_dataset_import_panel_preserves_unicode_paths(import_dialog_module):
    module, state = import_dialog_module
    panel = module.DatasetImportPanel()
    panel._handle = _HandleStub()

    base_path = Path("/tmp/日本語_データセット")
    state.dataset_info.base_path = base_path
    state.dataset_info.images_path = base_path / "images"
    state.dataset_info.sparse_path = base_path / "sparse"
    state.dataset_info.masks_path = base_path / "masks"
    state.dataset_infos[str(base_path)] = state.dataset_info
    state.output_browse_path = "/tmp/出力フォルダ"
    state.init_browse_path = "/tmp/初期化ポイント.ply"

    assert panel.show(str(base_path)) is True

    panel._on_browse_output()
    panel._on_browse_init()
    panel._on_do_load()

    assert state.load_file_calls == [
        {
            "path": str(base_path),
            "is_dataset": True,
            "output_path": "/tmp/出力フォルダ",
            "init_path": "/tmp/初期化ポイント.ply",
            "centralize_dataset": "off",
            "max_width": 3840,
        }
    ]
    assert state.panel_enabled_calls[-1] == ("lfs.dataset_import", False)


def test_dataset_import_panel_loads_updated_dataset_path(import_dialog_module, tmp_path):
    module, state = import_dialog_module
    panel = module.DatasetImportPanel()
    panel._handle = _HandleStub()

    second_base = tmp_path / "replacement_dataset"
    second_base.mkdir()
    state.dataset_infos[str(second_base)] = SimpleNamespace(
        base_path=second_base,
        images_path=second_base / "images",
        sparse_path=second_base / "sparse" / "0",
        masks_path=second_base / "masks",
        has_masks=False,
        image_count=48,
        mask_count=0,
    )

    assert panel.show(str(state.dataset_info.base_path)) is True

    panel._set_dataset_path(str(second_base))
    panel._on_do_load()

    assert state.load_file_calls == [
        {
            "path": str(second_base),
            "is_dataset": True,
            "output_path": str(second_base / "output"),
            "init_path": "",
            "centralize_dataset": "off",
            "max_width": 3840,
        }
    ]


def test_dataset_import_panel_clears_init_and_sidecar_on_dataset_change(import_dialog_module, tmp_path):
    module, state = import_dialog_module
    panel = module.DatasetImportPanel()
    panel._handle = _HandleStub()

    second_base = tmp_path / "replacement_dataset"
    second_base.mkdir()
    state.dataset_infos[str(second_base)] = SimpleNamespace(
        base_path=second_base,
        images_path=second_base / "images",
        sparse_path=second_base / "sparse" / "0",
        masks_path=second_base / "masks",
        has_masks=False,
        image_count=48,
        mask_count=0,
    )

    assert panel.show(str(state.dataset_info.base_path)) is True

    panel._set_init_path("/tmp/seed_points.ply")
    panel._set_ppisp_sidecar_path("/tmp/seed.ppisp")
    panel._set_dataset_path(str(second_base))

    assert panel._init_path == ""
    assert panel._ppisp_sidecar_path == ""

    panel._on_do_load()

    assert state.load_file_calls == [
        {
            "path": str(second_base),
            "is_dataset": True,
            "output_path": str(second_base / "output"),
            "init_path": "",
            "centralize_dataset": "off",
            "max_width": 3840,
        }
    ]


def test_resume_checkpoint_panel_validates_dataset_and_loads(import_dialog_module, tmp_path):
    module, state = import_dialog_module
    panel = module.ResumeCheckpointPanel()
    panel._handle = _HandleStub()

    assert panel.show(state.checkpoint_path) is True
    assert panel._dataset_valid is True
    assert state.panel_enabled_calls == [("lfs.resume_checkpoint", True)]

    invalid_path = str(tmp_path / "missing_dataset")
    panel._set_dataset_path(invalid_path)
    panel._on_do_load()

    assert state.load_checkpoint_calls == []
    assert panel._dataset_status_text() == "resume_checkpoint_popup.invalid"

    valid_path = str(tmp_path / "replacement_dataset")
    Path(valid_path).mkdir()
    state.output_browse_path = str(tmp_path / "replacement_output")
    panel._set_dataset_path(valid_path)
    panel._on_browse_output()
    panel._on_do_load()

    assert state.load_checkpoint_calls == [
        {
            "checkpoint_path": state.checkpoint_path,
            "dataset_path": valid_path,
            "output_path": state.output_browse_path,
        }
    ]
    assert state.panel_enabled_calls[-1] == ("lfs.resume_checkpoint", False)


def test_resume_checkpoint_panel_preserves_unicode_paths(import_dialog_module, tmp_path):
    module, state = import_dialog_module
    panel = module.ResumeCheckpointPanel()
    panel._handle = _HandleStub()

    dataset_path = tmp_path / "日本語_再開データセット"
    dataset_path.mkdir()
    state.checkpoint_path = str(tmp_path / "チェックポイント.resume")
    state.checkpoint_params.dataset_path = str(dataset_path)
    state.checkpoint_params.output_path = str(tmp_path / "出力先")
    state.output_browse_path = str(tmp_path / "別の出力先")

    assert panel.show(state.checkpoint_path) is True

    panel._on_browse_output()
    panel._on_do_load()

    assert state.load_checkpoint_calls == [
        {
            "checkpoint_path": state.checkpoint_path,
            "dataset_path": str(dataset_path),
            "output_path": state.output_browse_path,
        }
    ]
    assert state.panel_enabled_calls[-1] == ("lfs.resume_checkpoint", False)


def test_import_dialogs_dirty_all_on_language_change(import_dialog_module):
    module, state = import_dialog_module
    dataset_panel = module.DatasetImportPanel()
    resume_panel = module.ResumeCheckpointPanel()
    dataset_panel._handle = _HandleStub()
    resume_panel._handle = _HandleStub()
    dataset_panel._last_lang = "en"
    resume_panel._last_lang = "en"

    state.language[0] = "de"

    assert dataset_panel.on_update(None) is True
    assert resume_panel.on_update(None) is True
    assert dataset_panel._handle.dirty_all_calls == 1
    assert resume_panel._handle.dirty_all_calls == 1


def test_dataset_import_panel_binds_enter_and_escape(import_dialog_module):
    module, state = import_dialog_module
    panel = module.DatasetImportPanel()
    panel._handle = _HandleStub()
    document = _DocumentStub()

    panel.on_mount(document)
    assert panel.show(str(state.dataset_info.base_path)) is True

    enter_event = _EventStub(module.KI_RETURN)
    document.listeners["keydown"](enter_event)

    assert state.load_file_calls == [
        {
            "path": str(state.dataset_info.base_path),
            "is_dataset": True,
            "output_path": str(Path(state.dataset_info.base_path) / "output"),
            "init_path": "",
            "centralize_dataset": "off",
            "max_width": 3840,
        }
    ]
    assert enter_event.propagation_stopped is True

    escape_event = _EventStub(module.KI_ESCAPE)
    document.listeners["keydown"](escape_event)

    assert state.panel_enabled_calls[-1] == ("lfs.dataset_import", False)
    assert escape_event.propagation_stopped is True


def test_resume_checkpoint_panel_binds_enter_and_escape(import_dialog_module):
    module, state = import_dialog_module
    panel = module.ResumeCheckpointPanel()
    panel._handle = _HandleStub()
    document = _DocumentStub()

    panel.on_mount(document)
    assert panel.show(state.checkpoint_path) is True

    enter_event = _EventStub(module.KI_RETURN)
    document.listeners["keydown"](enter_event)

    assert state.load_checkpoint_calls == [
        {
            "checkpoint_path": state.checkpoint_path,
            "dataset_path": str(state.dataset_info.base_path),
            "output_path": str(Path(state.dataset_info.base_path) / "output"),
        }
    ]
    assert enter_event.propagation_stopped is True

    escape_event = _EventStub(module.KI_ESCAPE)
    document.listeners["keydown"](escape_event)

    assert state.panel_enabled_calls[-1] == ("lfs.resume_checkpoint", False)
    assert escape_event.propagation_stopped is True

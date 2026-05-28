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

    def _load_file(
        path,
        is_dataset=False,
        output_path="",
        init_path="",
        centralize_dataset="off",
        max_width=None,
        **_kwargs,
    ):
        state.load_file_calls.append(
            {
                "path": path,
                "is_dataset": is_dataset,
                "output_path": output_path,
                "init_path": init_path,
                "centralize_dataset": centralize_dataset,
                "max_width": max_width,
            }
        )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        get_current_language=lambda: state.language[0],
        set_panel_enabled=lambda panel_id, enabled: state.panel_enabled_calls.append((panel_id, enabled)),
        set_save_asset_callback=lambda _callback: None,
        open_dataset_folder_dialog=lambda: state.output_browse_path,
        open_ply_file_dialog=lambda _start_dir="": state.init_browse_path,
    )
    lf_stub.detect_dataset_info = lambda path: state.dataset_infos[str(path)]
    lf_stub.is_dataset_path = lambda path: str(path) in state.dataset_infos
    lf_stub.optimization_params = lambda: None
    lf_stub.load_file = _load_file
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
        self.request_update_count = 0

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_all_calls += 1

    def request_update(self):
        self.request_update_count += 1


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
        self.removed_models = []

    def add_event_listener(self, event, callback):
        self.listeners[event] = callback

    def get_element_by_id(self, element_id):
        if element_id == "close-btn":
            return self.close_btn
        return None

    def query_selector_all(self, _selector):
        return []

    def remove_data_model(self, name):
        self.removed_models.append(name)


class _ThreadStub:
    def __init__(self, alive=True):
        self.alive = alive
        self.join_calls = []

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        self.alive = False

    def is_alive(self):
        return self.alive


class _TimerStub(_ThreadStub):
    def __init__(self, alive=True):
        super().__init__(alive=alive)
        self.cancel_calls = 0

    def cancel(self):
        self.cancel_calls += 1


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


def test_watch_directory_discovery_imports_resume_checkpoints(import_dialog_module, tmp_path):
    module, _state = import_dialog_module
    scanner_module = import_module("lfs_plugins.asset_scanner")
    index_module = import_module("lfs_plugins.asset_index")

    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    checkpoint_path = watched_dir / "checkpoint.resume"
    checkpoint_path.write_bytes(b"checkpoint data")

    index = index_module.AssetIndex(tmp_path / "asset_manager" / "library.json")
    index.ensure_default_catalog()
    project = index.create_project("Default")

    metadata_list = module._discover_asset_metadata(
        scanner_module.AssetScanner(),
        str(watched_dir),
    )
    assert [metadata["path"] for metadata in metadata_list] == [str(checkpoint_path)]
    assert metadata_list[0]["type"] == "checkpoint"

    created_assets = module._register_discovered_assets(
        index,
        None,
        metadata_list,
        project_id=project.id,
    )
    assert [asset.name for asset in created_assets] == ["checkpoint.resume"]

    reloaded = index_module.AssetIndex(tmp_path / "asset_manager" / "library.json")
    assert reloaded.load() is True
    assert list(reloaded.assets.values())[0]["type"] == "checkpoint"


def test_watch_directory_import_allows_same_path_in_multiple_projects(
    import_dialog_module,
    tmp_path,
):
    module, _state = import_dialog_module
    index_module = import_module("lfs_plugins.asset_index")

    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    checkpoint_path = watched_dir / "checkpoint.resume"
    checkpoint_path.write_bytes(b"checkpoint data")

    index = index_module.AssetIndex(tmp_path / "asset_manager" / "library.json")
    index.ensure_default_catalog()
    default_project = index.create_project("Default")
    target_project = index.create_project("Target")

    metadata_list = [{"path": str(checkpoint_path), "type": "checkpoint"}]
    default_assets = module._register_discovered_assets(
        index,
        None,
        metadata_list,
        project_id=default_project.id,
    )
    target_assets = module._register_discovered_assets(
        index,
        None,
        metadata_list,
        project_id=target_project.id,
    )

    assert len(default_assets) == 1
    assert len(target_assets) == 1
    assert default_assets[0].id != target_assets[0].id
    assert default_assets[0].absolute_path == target_assets[0].absolute_path

    reloaded = index_module.AssetIndex(tmp_path / "asset_manager" / "library.json")
    assert reloaded.load() is True
    assert len(reloaded.list_assets(project_id=default_project.id)) == 1
    assert len(reloaded.list_assets(project_id=target_project.id)) == 1


def test_watch_dialog_uses_loaded_catalog_state(
    import_dialog_module,
    monkeypatch,
    tmp_path,
):
    module, _state = import_dialog_module
    index_module = import_module("lfs_plugins.asset_index")

    index = index_module.AssetIndex(tmp_path / "asset_manager" / "library.json")
    index.ensure_default_catalog()
    project = index.create_project("Default")
    index.set_watch_dirs(project.id, [str(tmp_path / "watched")])
    monkeypatch.setattr(module, "load_asset_index", lambda: index)

    panel = module.WatchDirsDialogPanel()
    assert panel.show(project.id) is True
    assert panel._project_id == project.id
    assert panel._watch_dirs == [str(tmp_path / "watched")]


def test_watch_dialog_does_not_mutate_active_panel_selection(import_dialog_module, monkeypatch, tmp_path):
    module, _state = import_dialog_module
    index_module = import_module("lfs_plugins.asset_index")

    index = index_module.AssetIndex(tmp_path / "asset_manager" / "library.json")
    index.ensure_default_catalog()
    default_project = index.create_project("Default")
    target_project = index.create_project("Target")

    selection_calls = []

    class _PanelStub:
        def __init__(self):
            self._asset_index = index
            self._selected_project_id = default_project.id

        def _select_project_id(self, project_id):
            selection_calls.append(project_id)
            self._selected_project_id = project_id
            return True

    active_panel = _PanelStub()
    monkeypatch.setattr(module, "get_asset_manager_panel", lambda: active_panel)
    monkeypatch.setattr(module, "load_asset_index", lambda: index)

    panel = module.WatchDirsDialogPanel()
    assert panel.show(target_project.id) is True
    assert selection_calls == []
    assert panel._project_id == target_project.id
    assert active_panel._selected_project_id == default_project.id


def test_asset_scanner_rejects_html_assets(import_dialog_module):
    scanner_module = import_module("lfs_plugins.asset_scanner")
    scanner = scanner_module.AssetScanner()

    assert scanner.detect_type("viewer.html") is None


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


def test_import_dialogs_use_dirty_update_policy(import_dialog_module):
    module, _state = import_dialog_module
    assert module.DatasetImportPanel.update_policy == "dirty"
    assert module.ResumeCheckpointPanel.update_policy == "dirty"
    assert module.URLImportPanel.update_policy == "dirty"
    assert module.WatchDirsDialogPanel.update_policy == "dirty"
    assert "update_interval_ms" not in module.DatasetImportPanel.__dict__
    assert "update_interval_ms" not in module.ResumeCheckpointPanel.__dict__
    assert "update_interval_ms" not in module.URLImportPanel.__dict__
    assert "update_interval_ms" not in module.WatchDirsDialogPanel.__dict__
    assert "update_interval_ms" not in module._ImportDialogPanel.__dict__


def test_import_dialogs_request_update_on_language_generation(import_dialog_module):
    module, _state = import_dialog_module
    dataset_panel = module.DatasetImportPanel()
    resume_panel = module.ResumeCheckpointPanel()
    dataset_panel._handle = _HandleStub()
    resume_panel._handle = _HandleStub()
    module.RuntimeState.language_generation._fallback = 0

    dataset_panel._subscribe_reactive_state()
    resume_panel._subscribe_reactive_state()
    module.RuntimeState.language_generation.value = 1

    assert dataset_panel._handle.request_update_count == 1
    assert resume_panel._handle.request_update_count == 1
    assert dataset_panel._handle.dirty_all_calls == 0
    assert resume_panel._handle.dirty_all_calls == 0

    dataset_panel._unsubscribe_reactive_state()
    resume_panel._unsubscribe_reactive_state()


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


def test_url_import_panel_unmount_cancels_and_joins_worker_threads(import_dialog_module):
    module, _state = import_dialog_module
    panel = module.URLImportPanel()
    document = _DocumentStub()
    worker = _ThreadStub()
    close_timer = _TimerStub()

    panel._url_import_in_progress = True
    panel._url_import_cancelled = False
    panel._url_import_thread = worker
    panel._url_import_close_timer = close_timer

    panel.on_unmount(document)

    assert panel._url_import_cancelled is True
    assert close_timer.cancel_calls == 1
    assert close_timer.join_calls == [module.THREAD_JOIN_TIMEOUT_SEC]
    assert worker.join_calls == [module.THREAD_JOIN_TIMEOUT_SEC]
    assert panel._url_import_thread is None
    assert panel._url_import_close_timer is None
    assert document.removed_models == ["url_import"]


def test_watch_dirs_dialog_unmount_cancels_and_joins_scan_thread(import_dialog_module):
    module, _state = import_dialog_module
    panel = module.WatchDirsDialogPanel()
    document = _DocumentStub()
    scan_thread = _ThreadStub()

    panel._scan_thread = scan_thread

    panel.on_unmount(document)

    assert panel._scan_cancel_event.is_set() is True
    assert scan_thread.join_calls == [module.THREAD_JOIN_TIMEOUT_SEC]
    assert panel._scan_thread is None
    assert document.removed_models == ["watch_dirs_dialog"]

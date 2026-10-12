# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera previews must keep the identity of the displayed source."""

from types import SimpleNamespace
import struct
import zlib

import pytest

from test_rmlui_image_sources import panel_modules


@pytest.fixture
def preview(panel_modules, monkeypatch, tmp_path):
    module, _ = panel_modules
    node = SimpleNamespace(id=4, uuid="original", type="camera", camera_uid=7,
                           image_path=str(tmp_path / "source.png"), name="source",
                           training_enabled=True, parent_id=-1, mask_path="")
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">2I5B", 1, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\0\x20\x40\x80"))
           + chunk(b"IEND", b""))
    (tmp_path / "source.png").write_bytes(png)
    nodes = [node]
    scene = SimpleNamespace(get_nodes=lambda: nodes,
                            get_node_by_id=lambda node_id: next(
                                (n for n in nodes if n.id == node_id), None))
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene, raising=False)
    monkeypatch.setattr(module.lf, "scene", SimpleNamespace(
        NodeType=SimpleNamespace(CAMERA="camera")), raising=False)
    calls = []
    monkeypatch.setattr(module.lf.ui, "go_to_camera_view", lambda uid: calls.append(uid), raising=False)
    monkeypatch.setattr(module.lf.ui, "is_gt_comparison_active", lambda: False, raising=False)
    monkeypatch.setattr(module.lf.ui, "toggle_gt_comparison", lambda: calls.append("gt"), raising=False)
    monkeypatch.setattr(module.lf, "set_camera_training_enabled",
                        lambda *args: calls.append(args), raising=False)
    monkeypatch.setattr(module.lf.ui, "set_panel_enabled", lambda *args: calls.append(args))
    panel = module.ImagePreviewPanel()
    panel.open([node.image_path], [], 0, [7], {7: (node.id, node.uuid)})
    return module, panel, nodes, calls


@pytest.mark.parametrize("action", ["_action_go_to_camera_view", "_action_open_in_gt_compare",
                                    "_action_toggle_training"])
def test_reused_camera_uid_cannot_target_replacement(preview, action):
    _, panel, nodes, calls = preview
    nodes[0] = SimpleNamespace(**{**vars(nodes[0]), "uuid": "replacement"})
    getattr(panel, action)()
    assert calls == []
    assert panel._current_camera_node() is None


def test_scene_switch_clears_dataset_preview(preview):
    module, panel, nodes, calls = preview
    panel._subscribe_reactive_state()
    try:
        nodes.clear()
        module.RuntimeState.scene_generation.value += 1
        assert panel._image_paths == []
        assert panel._camera_uids == []
        assert ("lfs.image_preview", False) in calls
    finally:
        panel._unsubscribe_reactive_state()


def test_scene_notification_preserves_live_camera_and_user_view(preview):
    module, panel, nodes, calls = preview
    panel._zoom = 2.5
    panel._pan_x = 12
    panel._subscribe_reactive_state()
    try:
        nodes[0].name = "renamed"
        module.RuntimeState.scene_generation.value += 1
        assert panel._image_paths
        assert panel._zoom == 2.5 and panel._pan_x == 12
        panel._action_go_to_camera_view()
        assert calls == [7]
        panel._action_toggle_training()
        assert calls[-1] == ("renamed", False)
    finally:
        panel._unsubscribe_reactive_state()


def test_live_camera_path_with_dot_component_stays_bound(preview):
    module, panel, nodes, calls = preview
    nodes[0].image_path = str(panel._image_paths[0].parent) + "/./source.png"
    module.open_camera_preview_by_uid(7)
    panel._subscribe_reactive_state()
    try:
        module.RuntimeState.scene_generation.value += 1
        assert panel._image_paths
        assert ("lfs.image_preview", False) not in calls
        panel._action_go_to_camera_view()
        assert calls == [("lfs.image_preview", True), 7]
    finally:
        panel._unsubscribe_reactive_state()


def test_direct_preview_with_dot_path_keeps_camera_actions(preview):
    module, panel, nodes, calls = preview
    nodes[0].image_path = str(panel._image_paths[0].parent) + "/./source.png"
    panel.open([nodes[0].image_path], [], 0, [7])
    panel._action_go_to_camera_view()
    assert calls == [7]


def test_direct_uid_open_defers_scene_enumeration_until_camera_action(preview, monkeypatch):
    module, panel, nodes, calls = preview
    scene = module.lf.get_scene()
    enumerations = []

    def get_nodes():
        enumerations.append(True)
        return nodes

    monkeypatch.setattr(scene, "get_nodes", get_nodes)
    panel.open([nodes[0].image_path], [], 0, [7])
    assert enumerations == []

    panel._action_go_to_camera_view()
    assert calls == [7]
    assert len(enumerations) == 1
    panel._action_toggle_training()
    assert calls[-1] == ("source", False)
    assert len(enumerations) == 1


def test_direct_uid_waiting_for_action_rejects_scene_replacement(preview):
    module, panel, nodes, calls = preview
    panel.open([nodes[0].image_path], [], 0, [7])
    panel._subscribe_reactive_state()
    try:
        nodes[0] = SimpleNamespace(**{**vars(nodes[0]), "uuid": "replacement"})
        module.RuntimeState.scene_generation.value += 1
        panel._action_go_to_camera_view()
        assert panel._image_paths == []
        assert 7 not in calls
    finally:
        panel._unsubscribe_reactive_state()


def test_camera_caller_reuses_its_scene_nodes_for_identity(preview, monkeypatch):
    module, panel, nodes, _ = preview
    nodes.extend(SimpleNamespace(id=i, type="other") for i in range(100, 5100))
    scene = module.lf.get_scene()
    calls = []
    def get_nodes():
        calls.append(True)
        return nodes
    monkeypatch.setattr(scene, "get_nodes", get_nodes)

    module.open_camera_preview_by_uid(7)

    assert len(calls) == 1
    assert panel._camera_nodes == {7: (4, "original")}
    assert panel._current_camera_node() is nodes[0]


def test_scene_switch_preserves_standalone_preview(preview):
    module, panel, nodes, calls = preview
    panel.open(panel._image_paths, [], 0)
    panel._subscribe_reactive_state()
    try:
        nodes.clear()
        module.RuntimeState.scene_generation.value += 1
        assert panel._image_paths
        assert calls == []
    finally:
        panel._unsubscribe_reactive_state()

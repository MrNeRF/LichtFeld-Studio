# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repeated retained-panel registration must leave the native UI usable."""

import base64
import json
import struct
import time

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor, _ui_editor
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture(autouse=True)
def dismiss_initial_association_prompt(camera_runtime):
    modal = _tool(camera_runtime, "ui_modal_get", {})
    if modal.get("open") and modal.get("title") == "Set as Default Application":
        _tool(camera_runtime, "ui_modal_press", {"label": "Not Now"})


@pytest.mark.parametrize("camera_runtime", [{"preferences": {
    "native_floating_windows": True,
    "floating_window_geometry": {"lfs.logging": [100000, -10000, 9000, 9000]},
}}], indirect=True)
def test_native_document_first_mount_uses_recovered_window_size(camera_runtime):
    endpoint = camera_runtime
    _editor(endpoint, "import lichtfeld as lf\n"
            "from lfs_plugins.logging_panel import LoggingPanel\n"
            "class SizedLoggingPanel(LoggingPanel):\n"
            "    id = 'lfs.logging'\n"
            "    def on_mount(self, doc):\n"
            "        self.first_size = [doc.client_width, doc.client_height]\n"
            "        super().on_mount(doc)\n"
            "lf.register_class(SizedLoggingPanel)\n"
            "lf.ui.set_panel_enabled(SizedLoggingPanel.id, True)")
    for _ in range(50):
        capture = _call(endpoint, "tools/call", {"name": "render_capture_window", "arguments": {
            "panel_id": "lfs.logging",
        }})
        images = [part for part in capture.get("content", []) if part.get("type") == "image"]
        if images:
            break
        time.sleep(.1)
    assert images, capture
    width, height = struct.unpack(">II", base64.b64decode(images[0]["data"])[16:24])
    first = json.loads(_editor(endpoint, "print(__import__('json').dumps("
                              "lf.ui.get_panel_object('lfs.logging').first_size))"))
    assert first == [width, height]


@pytest.mark.parametrize("camera_runtime", [{"preferences": {
    "native_floating_windows": True,
    "floating_window_geometry": {"lfs.image_preview": [0, 100, 640, 480]},
}}], indirect=True)
def test_replacing_enabled_image_preview_retires_its_native_session(camera_runtime, tmp_path):
    endpoint = camera_runtime
    _editor(endpoint, "import lichtfeld as lf\n"
            "review_panel = lf.ui.get_panel_object('lfs.image_preview')._load()\n"
            "lf.ui.set_panel_enabled(review_panel.id, True)")
    _frame(endpoint)
    assert _tool(endpoint, "ui_panel_describe", {"panel_id": "lfs.image_preview"})["panel"]["desktop_window"]
    # Missing input/explicit weights fail before model acquisition or CUDA inference.
    _editor(endpoint, "lf.ui.schedule_on_ui_thread(lambda p=review_panel, api=lf: "
            "setattr(p, 'review_request', api.ui.sam_preview_submit("
            f"{str(tmp_path / 'absent.png')!r}, 'absent', 0, 0, [[.5, .5, 1]], "
            f"weights={str(tmp_path / 'absent.lfw')!r})))")

    def native_state():
        _editor(endpoint, "lf.ui.schedule_on_ui_thread(lambda p=review_panel, api=lf: "
                "setattr(p, 'review_state', api.ui.sam_preview_poll()))")
        _frame(endpoint)
        return json.loads(_editor(endpoint, "print(__import__('json').dumps(review_panel.review_state))"))

    for _ in range(30):
        state = native_state()
        if state.get("phase") == "error":
            break
        time.sleep(.1)
    assert state.get("phase") == "error", state
    assert state["request"] == int(_editor(endpoint, "print(review_panel.review_request)"))
    _ui_editor(endpoint, "from lfs_plugins.image_preview_panel import ImagePreviewPanel\n"
            "review_panel._sam_sequence = {'run': '00000000000000000000000000000001'}\n"
            "class ReplacementPreview(ImagePreviewPanel):\n"
            "    id = 'lfs.image_preview'\n"
            "    options = set()\n"
            "lf.register_class(ReplacementPreview)")
    _frame(endpoint)
    _editor(endpoint, "assert lf.ui.is_panel_enabled(review_panel.id)\n"
            "assert lf.ui.get_panel_object(review_panel.id) is not review_panel\n"
            "assert review_panel._sam_sequence is None, (review_panel._sam_sequence, lf.ui.rml.get_document(review_panel.id))")
    for _ in range(30):
        state = native_state()
        if state.get("phase") == "idle":
            break
        time.sleep(.1)
    assert state.get("phase") == "idle", state


def _frame(endpoint):
    _call(endpoint, "tools/call", {"name": "render_capture_window", "arguments": {}})
    _call(endpoint, "resources/read", {"uri": "lichtfeld://ui/state"})


def test_builtin_panels_can_register_again(camera_runtime, tmp_path):
    endpoint = camera_runtime
    panels = _call(endpoint, "resources/read", {"uri": "lichtfeld://ui/panels"})
    (tmp_path / "startup-panels.json").write_text(json.dumps(panels, indent=2))
    capture = _call(endpoint, "tools/call", {"name": "render_capture_window", "arguments": {}})
    for part in capture.get("content", []):
        if part["type"] == "image":
            (tmp_path / "startup.png").write_bytes(base64.b64decode(part["data"]))
    for _ in range(3):
        _editor(endpoint, "import lfs_plugins; assert lfs_plugins.register_builtin_panels()")
        _frame(endpoint)
        _editor(endpoint, "import lichtfeld as lf; assert lf.ui.get_panel_object('lfs.rendering') is not None")


@pytest.mark.parametrize("retained", [True, False])
@pytest.mark.parametrize("unregister_first", ["none", "class", "module", "all"])
def test_panel_id_can_register_again(camera_runtime, tmp_path, retained, unregister_first):
    endpoint = camera_runtime
    # The built-in FILL scene panel consumes this region; give the probe room.
    _editor(endpoint, "import lichtfeld as lf; lf.ui.set_panel_enabled('lfs.scene', False)")
    template = tmp_path / "panel.rml"
    template.write_text("<rml><head><style>body { height: 48px; }</style></head>"
                        "<body><div id='value'>Panel registration</div></body></rml>")
    extra = f"    template = {str(template)!r}\n" if retained else ""
    _editor(endpoint, "import lichtfeld as lf\n"
            "mounts = unmounts = draws = 0\n"
            "class RepeatedPanel(lf.ui.Panel):\n"
            "    id = 'test.repeated_panel'\n"
            "    label = 'Repeated panel'\n"
            "    space = lf.ui.PanelSpace.SCENE_HEADER\n"
            + extra
            + ("    def on_mount(self, doc):\n"
               "        global mounts\n"
               "        mounts += 1\n"
               "    def on_unmount(self, doc):\n"
               "        global unmounts\n"
               "        unmounts += 1\n" if retained else
               "    def draw(self, ui):\n"
               "        global draws\n"
               "        draws += 1\n"
               "        ui.label('Panel registration')\n")
            + "lf.register_class(RepeatedPanel)\n")
    _frame(endpoint)
    if retained:
        _editor(endpoint, "assert mounts == 1, mounts")
    for index in range(3):
        code = {
            "none": "",
            "class": "lf.unregister_class(RepeatedPanel)\n",
            "module": "lf.ui.unregister_panels_for_module(RepeatedPanel.__module__)\n",
            "all": "lf.ui.unregister_all_panels()\n",
        }[unregister_first]
        _editor(endpoint, code + "lf.register_class(RepeatedPanel)")
        _frame(endpoint)
        _editor(endpoint, "assert lf.ui.get_panel_names(lf.ui.PanelSpace.SCENE_HEADER).count(RepeatedPanel.id) == 1")
        if retained:
            _editor(endpoint, "assert lf.ui.rml.get_document(RepeatedPanel.id) is not None; "
                    f"assert mounts == {index + 2}, mounts; assert unmounts == {index + 1}, unmounts")
        else:
            _editor(endpoint, "assert draws > 0; draws = 0")
    _editor(endpoint, "lf.unregister_class(RepeatedPanel); lf.unregister_class(RepeatedPanel)")
    _frame(endpoint)
    _editor(endpoint, "assert RepeatedPanel.id not in lf.ui.get_panel_names(lf.ui.PanelSpace.SCENE_HEADER)")
    if retained:
        _editor(endpoint, "assert unmounts == 4, unmounts")


def test_panels_without_an_id_keep_distinct_class_ids(camera_runtime):
    # Fails if a panel without its own id registers as "": its tab could not be
    # selected, and a second such panel replaced the first.
    _editor(camera_runtime, "import lichtfeld as lf\n"
            "class FirstTab(lf.ui.Panel):\n"
            "    label = 'First tab'\n"
            "    space = lf.ui.PanelSpace.MAIN_PANEL_TAB\n"
            "class SecondTab(lf.ui.Panel):\n"
            "    label = 'Second tab'\n"
            "    space = lf.ui.PanelSpace.MAIN_PANEL_TAB\n"
            "lf.register_class(FirstTab)\n"
            "lf.register_class(SecondTab)\n"
            "ids = [tab.id for tab in lf.ui.get_main_panel_tabs()]\n"
            "assert FirstTab._class_id() in ids and SecondTab._class_id() in ids, ids\n"
            "lf.unregister_class(FirstTab)\n"
            "lf.unregister_class(SecondTab)\n"
            "ids = [tab.id for tab in lf.ui.get_main_panel_tabs()]\n"
            "assert FirstTab._class_id() not in ids and SecondTab._class_id() not in ids, ids\n")

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repeated retained-panel registration must leave the native UI usable."""

import base64
import json

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor
from test_render_on_demand_idle import _call

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


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
    template = tmp_path / "panel.rml"
    template.write_text("<rml><head><style>body { height: 48px; }</style></head>"
                        "<body><div id='value'>Panel registration</div></body></rml>")
    extra = f"    template = {str(template)!r}\n" if retained else ""
    # Render before the native scene tree, which fills the available editor height.
    _editor(endpoint, "import lichtfeld as lf\n"
            "mounts = unmounts = draws = 0\n"
            "class RepeatedPanel(lf.ui.Panel):\n"
            "    id = 'test.repeated_panel'\n"
            "    label = 'Repeated panel'\n"
            "    space = lf.ui.PanelSpace.SCENE_HEADER\n"
            "    order = -100\n"
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

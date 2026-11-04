# SPDX-License-Identifier: GPL-3.0-or-later
import lichtfeld as lf
import lfs_plugins.training_runtime.panel as panel

assert panel._viewer and panel._viewer.node_id is not None
lf.select_node(panel._viewer.name)
lf.focus_selection()
lf.ui.set_main_panel_active_tab("lfs.training_sessions")
lf.ui.request_redraw()

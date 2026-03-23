# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Internal single source of truth for the public Python Panel base class."""

from __future__ import annotations

from os import PathLike


_PANEL_CLASS_SOURCE = """
class Panel:
    '''Public base class for all Python UI panels.'''

    id: str = ''
    label: str = ''
    space: PanelSpace = DEFAULT_SPACE
    parent: str = ''
    order: int = 100
    options: set[PanelOption] = set()
    poll_dependencies: set[PollDependency] = DEFAULT_POLL_DEPENDENCIES.copy()
    size: tuple[float, float] | None = None
    template: str | PathLike[str] = ''
    style: str = ''
    height_mode: PanelHeightMode = DEFAULT_HEIGHT_MODE
    update_interval_ms: int = 100

    @classmethod
    def _class_id(cls) -> str:
        return f'{cls.__module__}.{cls.__qualname__}'

    @classmethod
    def poll(cls, context) -> bool:
        del context
        return True

    def draw(self, ui):
        del ui

    def on_bind_model(self, ctx):
        del ctx

    def on_mount(self, doc):
        import lichtfeld as lf

        close_btn = doc.get_element_by_id('close-btn')
        if close_btn and self.id:
            close_btn.add_event_listener(
                'click',
                lambda _ev: lf.ui.set_panel_enabled(self.id, False),
            )

    def on_unmount(self, doc):
        del doc

    def on_update(self, doc):
        del doc

    def on_scene_changed(self, doc):
        del doc
"""


def _build_panel_base(
    *,
    module_name: str,
    panel_space_type: object,
    panel_option_type: object,
    poll_dependency_type: object,
    panel_height_mode_type: object,
    default_space: object,
    default_poll_dependencies: set[object],
    default_height_mode: object,
):
    namespace = {
        "__name__": module_name,
        "PathLike": PathLike,
        "PanelSpace": panel_space_type,
        "PanelOption": panel_option_type,
        "PollDependency": poll_dependency_type,
        "PanelHeightMode": panel_height_mode_type,
        "DEFAULT_SPACE": default_space,
        "DEFAULT_POLL_DEPENDENCIES": set(default_poll_dependencies),
        "DEFAULT_HEIGHT_MODE": default_height_mode,
    }
    exec(_PANEL_CLASS_SOURCE, namespace, namespace)
    return namespace["Panel"]


def install_runtime_panel_base(ui_module):
    """Install the canonical Panel base class into ``lichtfeld.ui``."""
    if hasattr(ui_module, "Panel"):
        return ui_module.Panel

    panel_type = _build_panel_base(
        module_name=ui_module.__name__,
        panel_space_type=ui_module.PanelSpace,
        panel_option_type=ui_module.PanelOption,
        poll_dependency_type=ui_module.PollDependency,
        panel_height_mode_type=ui_module.PanelHeightMode,
        default_space=ui_module.PanelSpace.MAIN_PANEL_TAB,
        default_poll_dependencies={
            ui_module.PollDependency.SCENE,
            ui_module.PollDependency.SELECTION,
            ui_module.PollDependency.TRAINING,
        },
        default_height_mode=ui_module.PanelHeightMode.FILL,
    )
    ui_module.Panel = panel_type
    return panel_type


FallbackPanel = _build_panel_base(
    module_name=__name__,
    panel_space_type=object,
    panel_option_type=object,
    poll_dependency_type=object,
    panel_height_mode_type=object,
    default_space="MAIN_PANEL_TAB",
    default_poll_dependencies={"SCENE", "SELECTION", "TRAINING"},
    default_height_mode="fill",
)

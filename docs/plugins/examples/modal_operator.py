"""Interactive modal operator example.

Demonstrates a rectangular selection tool that captures mouse input
and draws a selection rectangle on the viewport.
"""

import lichtfeld as lf
from lfs_plugins.types import Operator, Event
from lfs_plugins.props import FloatProperty


class RectSelectOperator(Operator):
    label = "Rectangle Select"
    description = "Draw a rectangle to select gaussians"
    options = {"UNDO"}

    threshold: float = FloatProperty(default=0.5, min=0.0, max=1.0)

    def __init__(self):
        super().__init__()
        self.start_x = 0.0
        self.start_y = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.dragging = False

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def invoke(self, context, event: Event) -> set:
        self.start_x = event.mouse_x
        self.start_y = event.mouse_y
        self.current_x = event.mouse_x
        self.current_y = event.mouse_y
        self.dragging = False
        lf.log.info("Rectangle select: click and drag to select")
        return {"RUNNING_MODAL"}

    def modal(self, context, event: Event) -> set:
        if event.type == "ESC":
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            self.current_x = event.mouse_x
            self.current_y = event.mouse_y
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS":
                self.start_x = event.mouse_x
                self.start_y = event.mouse_y
                self.dragging = True
                return {"RUNNING_MODAL"}

            if event.value == "RELEASE" and self.dragging:
                x0 = min(self.start_x, self.current_x)
                y0 = min(self.start_y, self.current_y)
                x1 = max(self.start_x, self.current_x)
                y1 = max(self.start_y, self.current_y)

                width = x1 - x0
                height = y1 - y0
                lf.log.info(
                    f"Selected region: ({x0:.0f}, {y0:.0f}) to ({x1:.0f}, {y1:.0f}), "
                    f"size: {width:.0f}x{height:.0f}"
                )
                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self.dragging = False
        lf.log.info("Rectangle select cancelled")


class PointInfoOperator(Operator):
    label = "Point Info"
    description = "Click a point to show its info"

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def invoke(self, context, event: Event) -> set:
        lf.log.info("Click a point to inspect...")
        return {"RUNNING_MODAL"}

    def modal(self, context, event: Event) -> set:
        if event.type == "ESC":
            return {"CANCELLED"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            if event.over_gui:
                return {"PASS_THROUGH"}

            lf.log.info(
                f"Clicked at viewport ({event.mouse_x:.0f}, {event.mouse_y:.0f})"
            )

            if event.shift:
                lf.log.info("Shift-click: adding to selection")
            if event.ctrl:
                lf.log.info("Ctrl-click: removing from selection")

            return {"FINISHED"}

        return {"RUNNING_MODAL"}


class ModalDemoPanel(lf.ui.Panel):
    label = "Modal Tools"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 160

    def draw(self, ui):
        ui.heading("Modal Operators")

        if RectSelectOperator.poll(None):
            if ui.button("Rectangle Select", (-1, 0)):
                lf.ui.ops.invoke(RectSelectOperator._class_id())

            if ui.button("Point Info", (-1, 0)):
                lf.ui.ops.invoke(PointInfoOperator._class_id())
        else:
            ui.text_disabled("Load a scene to use these tools")


_classes = [RectSelectOperator, PointInfoOperator, ModalDemoPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)

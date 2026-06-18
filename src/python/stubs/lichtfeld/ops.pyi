"""Operator system"""

import enum

import lichtfeld


class OperatorResult(enum.Enum):
    FINISHED = 0

    CANCELLED = 1

    RUNNING_MODAL = 2

    PASS_THROUGH = 3

class BuiltinOp(enum.Enum):
    SelectionStroke = 0

    TransformSet = 1

    TransformTranslate = 2

    TransformRotate = 3

    TransformScale = 4

    TransformApplyBatch = 5

    AlignPickPoint = 6

    Undo = 7

    Redo = 8

    Delete = 9

    SelectionClear = 10

    SceneSelectNode = 11

    CropBoxAdd = 12

    CropBoxSet = 13

    CropBoxFit = 14

    CropBoxReset = 15

    EllipsoidAdd = 16

    EllipsoidSet = 17

    EllipsoidFit = 18

    EllipsoidReset = 19

class BuiltinTool(enum.Enum):
    Select = 0

    Translate = 1

    Rotate = 2

    Scale = 3

    Mirror = 4

    Align = 5

class OperatorFlags(enum.Enum):
    NONE = 0

    REGISTER = 1

    UNDO = 2

    UNDO_GROUPED = 4

    INTERNAL = 8

    MODAL = 16

    BLOCKING = 32

    def __or__(self, arg: OperatorFlags, /) -> OperatorFlags: ...

    def __and__(self, arg: OperatorFlags, /) -> OperatorFlags: ...

class OperatorDescriptor:
    def __init__(self) -> None: ...

    @property
    def id(self) -> str:
        """Unique operator identifier"""

    @property
    def label(self) -> str:
        """Display label"""

    @label.setter
    def label(self, arg: str, /) -> None: ...

    @property
    def description(self) -> str:
        """Tooltip description"""

    @description.setter
    def description(self, arg: str, /) -> None: ...

    @property
    def icon(self) -> str:
        """Icon name"""

    @icon.setter
    def icon(self, arg: str, /) -> None: ...

    @property
    def shortcut(self) -> str:
        """Keyboard shortcut string"""

    @shortcut.setter
    def shortcut(self, arg: str, /) -> None: ...

    @property
    def flags(self) -> OperatorFlags:
        """Operator behavior flags"""

    @flags.setter
    def flags(self, arg: OperatorFlags, /) -> None: ...

def invoke(arg0: str, /, **kwargs) -> lichtfeld.OperatorReturnValue:
    """Invoke an operator by id with optional kwargs"""

def poll(id: str) -> bool:
    """Check if an operator can run"""

def get_all() -> list[str]:
    """Get all registered operator IDs"""

def get_descriptor(id: str) -> OperatorDescriptor | None:
    """Get operator descriptor by ID (None if not found)"""

def has_modal() -> bool:
    """Check if a modal operator is running"""

def cancel_modal() -> None:
    """Cancel the active modal operator"""

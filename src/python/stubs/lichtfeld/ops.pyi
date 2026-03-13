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

    BrushStroke = 1

    TransformSet = 2

    TransformTranslate = 3

    TransformRotate = 4

    TransformScale = 5

    TransformApplyBatch = 6

    AlignPickPoint = 7

    Undo = 8

    Redo = 9

    Delete = 10

    SelectionClear = 11

    SceneSelectNode = 12

    CropBoxAdd = 13

    CropBoxSet = 14

    CropBoxFit = 15

    CropBoxReset = 16

    EllipsoidAdd = 17

    EllipsoidSet = 18

    EllipsoidFit = 19

    EllipsoidReset = 20

class BuiltinTool(enum.Enum):
    Select = 0

    Translate = 1

    Rotate = 2

    Scale = 3

    Mirror = 4

    Brush = 5

    Align = 6

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

def undo() -> None:
    """Undo the last operation"""

def redo() -> None:
    """Redo the last undone operation"""

def can_undo() -> bool:
    """Check if undo is available"""

def can_redo() -> bool:
    """Check if redo is available"""

def has_modal() -> bool:
    """Check if a modal operator is running"""

def cancel_modal() -> None:
    """Cancel the active modal operator"""

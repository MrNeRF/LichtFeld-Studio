"""Keymap configuration"""

import enum


class Action(enum.Enum):
    NONE = 0

    CAMERA_ORBIT = 1

    CAMERA_PAN = 2

    CAMERA_ZOOM = 3

    CAMERA_ROLL = 4

    CAMERA_MOVE_FORWARD = 5

    CAMERA_MOVE_BACKWARD = 6

    CAMERA_MOVE_LEFT = 7

    CAMERA_MOVE_RIGHT = 8

    CAMERA_MOVE_UP = 9

    CAMERA_MOVE_DOWN = 10

    CAMERA_RESET_HOME = 11

    CAMERA_FOCUS_SELECTION = 12

    CAMERA_SET_PIVOT = 13

    CAMERA_NEXT_VIEW = 14

    CAMERA_PREV_VIEW = 15

    CAMERA_SPEED_UP = 16

    CAMERA_SPEED_DOWN = 17

    ZOOM_SPEED_UP = 18

    ZOOM_SPEED_DOWN = 19

    TOGGLE_SPLIT_VIEW = 20

    TOGGLE_INDEPENDENT_SPLIT_VIEW = 21

    TOGGLE_GT_COMPARISON = 22

    TOGGLE_DEPTH_MODE = 23

    CYCLE_PLY = 24

    DELETE_SELECTED = 25

    DELETE_NODE = 26

    UNDO = 27

    REDO = 28

    INVERT_SELECTION = 29

    DESELECT_ALL = 30

    SELECT_ALL = 31

    COPY_SELECTION = 32

    PASTE_SELECTION = 33

    DEPTH_ADJUST_FAR = 34

    DEPTH_ADJUST_SIDE = 35

    TOGGLE_SELECTION_DEPTH_FILTER = 36

    TOGGLE_SELECTION_CROP_FILTER = 37

    BRUSH_RESIZE = 38

    CYCLE_BRUSH_MODE = 39

    CONFIRM_POLYGON = 40

    CANCEL_POLYGON = 41

    UNDO_POLYGON_VERTEX = 42

    CYCLE_SELECTION_VIS = 43

    SELECTION_REPLACE = 44

    SELECTION_ADD = 45

    SELECTION_REMOVE = 46

    SELECT_MODE_CENTERS = 47

    SELECT_MODE_RECTANGLE = 48

    SELECT_MODE_POLYGON = 49

    SELECT_MODE_LASSO = 50

    SELECT_MODE_RINGS = 51

    APPLY_CROP_BOX = 52

    NODE_PICK = 53

    NODE_RECT_SELECT = 54

    TOGGLE_UI = 55

    TOGGLE_FULLSCREEN = 56

    SEQUENCER_ADD_KEYFRAME = 57

    SEQUENCER_UPDATE_KEYFRAME = 58

    SEQUENCER_PLAY_PAUSE = 59

    TOOL_SELECT = 60

    TOOL_TRANSLATE = 61

    TOOL_ROTATE = 62

    TOOL_SCALE = 63

    TOOL_MIRROR = 64

    TOOL_BRUSH = 65

    TOOL_ALIGN = 66

    PIE_MENU = 67

    DEPTH_ADJUST_NEAR = 68

class ToolMode(enum.Enum):
    GLOBAL = 0

    SELECTION = 1

    BRUSH = 2

    TRANSLATE = 3

    ROTATE = 4

    SCALE = 5

    ALIGN = 6

    CROP_BOX = 7

class Modifier(enum.Enum):
    NONE = 0

    SHIFT = 1

    CTRL = 2

    ALT = 4

    SUPER = 8

class MouseButton(enum.Enum):
    LEFT = 0

    RIGHT = 1

    MIDDLE = 2

class KeyTrigger:
    def __init__(self, key: int, modifiers: int = Modifier.NONE, on_repeat: bool = False) -> None: ...

    @property
    def key(self) -> int:
        """Key code"""

    @key.setter
    def key(self, arg: int, /) -> None: ...

    @property
    def modifiers(self) -> int:
        """Modifier key bitmask"""

    @modifiers.setter
    def modifiers(self, arg: int, /) -> None: ...

    @property
    def on_repeat(self) -> bool:
        """Whether to trigger on key repeat"""

    @on_repeat.setter
    def on_repeat(self, arg: bool, /) -> None: ...

class MouseButtonTrigger:
    def __init__(self, button: MouseButton, modifiers: int = Modifier.NONE, double_click: bool = False) -> None: ...

    @property
    def button(self) -> MouseButton:
        """Mouse button"""

    @button.setter
    def button(self, arg: MouseButton, /) -> None: ...

    @property
    def modifiers(self) -> int:
        """Modifier key bitmask"""

    @modifiers.setter
    def modifiers(self, arg: int, /) -> None: ...

    @property
    def double_click(self) -> bool:
        """Whether to require double-click"""

    @double_click.setter
    def double_click(self, arg: bool, /) -> None: ...

def get_action_for_key(mode: ToolMode, key: int, modifiers: int = 0) -> Action:
    """Get action bound to a key in given mode"""

def get_key_for_action(action: Action, mode: ToolMode = ToolMode.GLOBAL) -> int:
    """Get key code bound to an action"""

def get_trigger_description(action: Action, mode: ToolMode = ToolMode.GLOBAL) -> str:
    """Get human-readable description of action's trigger"""

def set_binding(mode: ToolMode, action: Action, key: int, modifiers: int = 0) -> None:
    """Bind a key to an action in given mode"""

def clear_binding(mode: ToolMode, action: Action) -> None:
    """Remove binding for an action in given mode"""

def get_action_name(action: Action) -> str:
    """Get display name for an action"""

def get_key_name(key: int) -> str:
    """Get display name for a key code"""

def get_modifier_string(modifiers: int) -> str:
    """Get display string for modifier bitmask"""

def get_available_profiles() -> list[str]:
    """Get list of available keymap profile names"""

def get_current_profile() -> str:
    """Get name of active keymap profile"""

def load_profile(name: str) -> None:
    """Load a keymap profile by name"""

def save_profile(name: str) -> None:
    """Save current bindings as a named profile"""

def export_profile(path: str) -> bool:
    """Export current profile to file"""

def import_profile(path: str) -> bool:
    """Import profile from file"""

def start_capture(mode: ToolMode, action: Action) -> None:
    """Start capturing input for rebinding"""

def cancel_capture() -> None:
    """Cancel active capture"""

def is_capturing() -> bool:
    """Check if capture mode is active"""

def is_waiting_for_double_click() -> bool:
    """Check if waiting for potential double-click"""

def get_captured_trigger() -> object:
    """Get captured trigger (clears it), returns None if nothing captured"""

def get_bindings_for_mode(mode: ToolMode) -> list:
    """Get all bindings for a tool mode"""

def reset_to_default() -> None:
    """Reset to default bindings"""

def get_tool_mode_name(mode: ToolMode) -> str:
    """Get human-readable name for tool mode"""

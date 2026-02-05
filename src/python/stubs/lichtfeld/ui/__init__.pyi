from collections.abc import Callable, Sequence
import enum
from typing import overload

from . import (
    action as action,
    key as key,
    mod as mod,
    mouse as mouse,
    ops as ops,
    signals as signals
)


class AppContext:
    def __init__(self) -> None: ...

    @property
    def has_scene(self) -> bool: ...

    @property
    def scene_generation(self) -> int: ...

    @property
    def has_trainer(self) -> bool: ...

    @property
    def is_training(self) -> bool: ...

    @property
    def is_paused(self) -> bool: ...

    @property
    def iteration(self) -> int: ...

    @property
    def max_iterations(self) -> int: ...

    @property
    def loss(self) -> float: ...

    @property
    def has_selection(self) -> bool: ...

    @property
    def num_gaussians(self) -> int: ...

    @property
    def selection_submode(self) -> int: ...

    @property
    def pivot_mode(self) -> int: ...

    @property
    def transform_space(self) -> int: ...

    @property
    def viewport_bounds(self) -> tuple[float, float, float, float]: ...

    @property
    def viewport_valid(self) -> bool: ...

    @property
    def scene(self) -> object: ...

    @property
    def selected_objects(self) -> object: ...

    @property
    def active_object(self) -> object: ...

    @property
    def selected_gaussians(self) -> object:
        """Gaussian selection mask tensor"""

def context() -> AppContext:
    """Get the current application context"""

class ThemePalette:
    @property
    def background(self) -> tuple[float, float, float, float]: ...

    @property
    def surface(self) -> tuple[float, float, float, float]: ...

    @property
    def surface_bright(self) -> tuple[float, float, float, float]: ...

    @property
    def primary(self) -> tuple[float, float, float, float]: ...

    @property
    def primary_dim(self) -> tuple[float, float, float, float]: ...

    @property
    def secondary(self) -> tuple[float, float, float, float]: ...

    @property
    def text(self) -> tuple[float, float, float, float]: ...

    @property
    def text_dim(self) -> tuple[float, float, float, float]: ...

    @property
    def border(self) -> tuple[float, float, float, float]: ...

    @property
    def success(self) -> tuple[float, float, float, float]: ...

    @property
    def warning(self) -> tuple[float, float, float, float]: ...

    @property
    def error(self) -> tuple[float, float, float, float]: ...

    @property
    def info(self) -> tuple[float, float, float, float]: ...

    @property
    def toolbar_background(self) -> tuple[float, float, float, float]: ...

    @property
    def row_even(self) -> tuple[float, float, float, float]: ...

    @property
    def row_odd(self) -> tuple[float, float, float, float]: ...

    @property
    def overlay_border(self) -> tuple[float, float, float, float]: ...

    @property
    def overlay_icon(self) -> tuple[float, float, float, float]: ...

    @property
    def overlay_text(self) -> tuple[float, float, float, float]: ...

    @property
    def overlay_text_dim(self) -> tuple[float, float, float, float]: ...

class ThemeSizes:
    @property
    def window_rounding(self) -> float: ...

    @property
    def frame_rounding(self) -> float: ...

    @property
    def popup_rounding(self) -> float: ...

    @property
    def scrollbar_rounding(self) -> float: ...

    @property
    def tab_rounding(self) -> float: ...

    @property
    def border_size(self) -> float: ...

    @property
    def window_padding(self) -> tuple[float, float]: ...

    @property
    def frame_padding(self) -> tuple[float, float]: ...

    @property
    def item_spacing(self) -> tuple[float, float]: ...

    @property
    def toolbar_button_size(self) -> float: ...

    @property
    def toolbar_padding(self) -> float: ...

    @property
    def toolbar_spacing(self) -> float: ...

class Theme:
    @property
    def name(self) -> str: ...

    @property
    def palette(self) -> ThemePalette: ...

    @property
    def sizes(self) -> ThemeSizes: ...

def theme() -> Theme:
    """Get the current theme"""

class PanelSpace(enum.Enum):
    SIDE_PANEL = 0

    FLOATING = 1

    VIEWPORT_OVERLAY = 2

    DOCKABLE = 3

    MAIN_PANEL_TAB = 4

    SCENE_HEADER = 5

    STATUS_BAR = 6

def register_panel(cls: object) -> None:
    """Register a panel class for rendering in the UI"""

def unregister_panel(cls: object) -> None:
    """Unregister a panel class"""

def unregister_all_panels() -> None:
    """Unregister all Python panels"""

def get_panel_names(space: str = 'FLOATING') -> list[str]:
    """Get registered panel names for a given space"""

def set_panel_enabled(label: str, enabled: bool) -> None:
    """Enable or disable a panel by label"""

def is_panel_enabled(label: str) -> bool:
    """Check if a panel is enabled"""

def get_main_panel_tabs() -> list:
    """Get all main panel tabs as list of dicts"""

def get_panel(idname: str) -> object:
    """Get panel info by idname (None if not found)"""

def set_panel_label(idname: str, label: str) -> bool:
    """Set the display label for a panel"""

def set_panel_order(idname: str, order: int) -> bool:
    """Set the sort order for a panel"""

def set_panel_space(idname: str, space: str) -> bool:
    """Set the panel space (where it renders)"""

def has_main_panel_tabs() -> bool:
    """Check if any main panel tabs are registered"""

class HookPosition(enum.Enum):
    PREPEND = 0

    APPEND = 1

def add_hook(panel: str, section: str, callback: object, position: str = 'append') -> None:
    """Add a UI hook callback to a panel section"""

def remove_hook(panel: str, section: str, callback: object) -> None:
    """Remove a specific UI hook callback"""

def clear_hooks(panel: str, section: str = '') -> None:
    """Clear all hooks for a panel or panel/section"""

def clear_all_hooks() -> None:
    """Clear all registered UI hooks"""

def get_hook_points() -> list[str]:
    """Get all registered hook point identifiers"""

def invoke_hooks(panel: str, section: str, prepend: bool = False) -> None:
    """
    Invoke all hooks for a panel/section (prepend=True for prepend hooks, False for append)
    """

def hook(panel: str, section: str, position: str = 'append') -> object:
    """Decorator to register a UI hook for a panel section"""

class MenuLocation(enum.Enum):
    FILE = 0

    EDIT = 1

    VIEW = 2

    WINDOW = 3

    HELP = 4

    MENU_BAR = 5

def unregister_all_menus() -> None:
    """Unregister all Python menus"""

class OperatorProperties:
    def __init__(self, operator_id: str) -> None: ...

    def __setattr__(self, arg0: str, arg1: object, /) -> None:
        """Set an operator property value"""

    def __getattr__(self, arg: str, /) -> object:
        """Get an operator property value"""

    @property
    def properties(self) -> dict:
        """All properties as a dictionary"""

    @property
    def operator_id(self) -> str:
        """Operator identifier string"""

def unregister_operator(id: str) -> None:
    """Unregister an operator"""

def unregister_all_operators() -> None:
    """Unregister all Python operators"""

def execute_operator(id: str) -> bool:
    """Execute an operator by id"""

def poll_operator(id: str) -> bool:
    """Check if an operator can run"""

def get_operator_ids() -> list[str]:
    """Get list of registered operator ids"""

def confirm_dialog(title: str, message: str, buttons: Sequence[str] = ['OK', 'Cancel'], callback: object | None = None) -> None:
    """Show a confirmation dialog with custom buttons"""

def input_dialog(title: str, message: str, default_value: str = '', callback: object | None = None) -> None:
    """Show an input dialog"""

def message_dialog(title: str, message: str, style: str = 'info', callback: object | None = None) -> None:
    """Show a message dialog (style: 'info', 'warning', or 'error')"""

def request_redraw() -> None:
    """Request a UI redraw on next frame"""

def consume_redraw_request() -> bool:
    """Consume and return pending redraw request flag"""

class Event:
    def __init__(self) -> None:
        """Create a default Event"""

    @property
    def type(self) -> str:
        """Event type ('MOUSEMOVE', 'LEFTMOUSE', 'KEY_A', etc.)"""

    @type.setter
    def type(self, arg: str, /) -> None: ...

    @property
    def value(self) -> str:
        """Event value ('PRESS', 'RELEASE', 'NOTHING')"""

    @value.setter
    def value(self, arg: str, /) -> None: ...

    @property
    def mouse_x(self) -> float:
        """Mouse X position"""

    @mouse_x.setter
    def mouse_x(self, arg: float, /) -> None: ...

    @property
    def mouse_y(self) -> float:
        """Mouse Y position"""

    @mouse_y.setter
    def mouse_y(self, arg: float, /) -> None: ...

    @property
    def mouse_region_x(self) -> float:
        """Mouse X position relative to region"""

    @mouse_region_x.setter
    def mouse_region_x(self, arg: float, /) -> None: ...

    @property
    def mouse_region_y(self) -> float:
        """Mouse Y position relative to region"""

    @mouse_region_y.setter
    def mouse_region_y(self, arg: float, /) -> None: ...

    @property
    def delta_x(self) -> float:
        """Mouse delta X (for drag operations)"""

    @delta_x.setter
    def delta_x(self, arg: float, /) -> None: ...

    @property
    def delta_y(self) -> float:
        """Mouse delta Y (for drag operations)"""

    @delta_y.setter
    def delta_y(self, arg: float, /) -> None: ...

    @property
    def scroll_x(self) -> float:
        """Scroll X offset"""

    @scroll_x.setter
    def scroll_x(self, arg: float, /) -> None: ...

    @property
    def scroll_y(self) -> float:
        """Scroll Y offset"""

    @scroll_y.setter
    def scroll_y(self, arg: float, /) -> None: ...

    @property
    def shift(self) -> bool:
        """Shift modifier is held"""

    @shift.setter
    def shift(self, arg: bool, /) -> None: ...

    @property
    def ctrl(self) -> bool:
        """Ctrl modifier is held"""

    @ctrl.setter
    def ctrl(self, arg: bool, /) -> None: ...

    @property
    def alt(self) -> bool:
        """Alt modifier is held"""

    @alt.setter
    def alt(self, arg: bool, /) -> None: ...

    @property
    def pressure(self) -> float:
        """Tablet pressure (1.0 for mouse)"""

    @pressure.setter
    def pressure(self, arg: float, /) -> None: ...

    @property
    def over_gui(self) -> bool:
        """Mouse is over GUI element"""

    @over_gui.setter
    def over_gui(self, arg: bool, /) -> None: ...

    @property
    def key_code(self) -> int:
        """Raw key code for KEY events"""

    @key_code.setter
    def key_code(self, arg: int, /) -> None: ...

    def __repr__(self) -> str:
        """Return string representation of the event"""

class SubLayout:
    def __enter__(self) -> SubLayout: ...

    def __exit__(self, *args) -> bool: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def active(self) -> bool: ...

    @active.setter
    def active(self, arg: bool, /) -> None: ...

    @property
    def alert(self) -> bool: ...

    @alert.setter
    def alert(self, arg: bool, /) -> None: ...

    def row(self) -> SubLayout: ...

    def column(self) -> SubLayout: ...

    def split(self, factor: float = 0.5) -> SubLayout: ...

    def box(self) -> SubLayout: ...

    def grid_flow(self, columns: int = 0, even_columns: bool = True, even_rows: bool = True) -> SubLayout: ...

    def prop_enum(self, data: object, prop_id: str, value: str, text: str = '') -> bool: ...

    def label(self, text: str) -> None: ...

    def button(self, label: str, size: tuple[float, float] = (0.0, 0.0)) -> bool: ...

    def button_styled(self, label: str, style: str, size: tuple[float, float] = (0.0, 0.0)) -> bool: ...

    def prop(self, data: object, prop_id: str, text: str | None = None) -> tuple[bool, object]: ...

    def checkbox(self, label: str, value: bool) -> tuple[bool, bool]: ...

    def slider_float(self, label: str, value: float, min: float, max: float) -> tuple[bool, float]: ...

    def slider_int(self, label: str, value: int, min: int, max: int) -> tuple[bool, int]: ...

    def drag_float(self, label: str, value: float, speed: float = 1.0, min: float = 0.0, max: float = 0.0) -> tuple[bool, float]: ...

    def drag_int(self, label: str, value: int, speed: float = 1.0, min: int = 0, max: int = 0) -> tuple[bool, int]: ...

    def input_text(self, label: str, value: str) -> tuple[bool, str]: ...

    def combo(self, label: str, current_idx: int, items: Sequence[str]) -> tuple[bool, int]: ...

    def separator(self) -> None: ...

    def spacing(self) -> None: ...

    def heading(self, text: str) -> None: ...

    def collapsing_header(self, label: str, default_open: bool = False) -> bool: ...

    def tree_node(self, label: str) -> bool: ...

    def tree_pop(self) -> None: ...

    def progress_bar(self, fraction: float, overlay: str = '', width: float = 0.0) -> None: ...

    def text_colored(self, text: str, color: tuple[float, float, float, float]) -> None: ...

    def text_wrapped(self, text: str) -> None: ...

    def __getattr__(self, arg: str, /) -> object: ...

class WindowFlags:
    None: int = ...
    """No flags set"""

    NoScrollbar: int = ...
    """Disable scrollbar"""

    NoScrollWithMouse: int = ...
    """Disable mouse wheel scrolling"""

    MenuBar: int = ...
    """Enable menu bar"""

    NoResize: int = ...
    """Disable window resizing"""

    NoMove: int = ...
    """Disable window moving"""

    NoCollapse: int = ...
    """Disable window collapsing"""

    AlwaysAutoResize: int = ...
    """Auto-resize window to fit content"""

    NoTitleBar: int = ...
    """Hide window title bar"""

    NoNavFocus: int = ...
    """Disable navigation focus"""

    NoInputs: int = ...
    """Disable all input capture"""

    NoBackground: int = ...
    """Disable window background"""

    NoFocusOnAppearing: int = ...
    """Disable focus when window appears"""

    NoBringToFrontOnFocus: int = ...
    """Disable bringing window to front on focus"""

class UILayout:
    def __init__(self) -> None:
        """Create a UILayout for drawing UI elements"""

    WindowFlags: lichtfeld.ui.WindowFlags = ...
    """Window flags constants"""

    def label(self, text: str) -> None:
        """Draw a text label"""

    def label_centered(self, text: str) -> None:
        """Draw a horizontally centered text label"""

    def heading(self, text: str) -> None:
        """Draw a bold heading text"""

    def text_colored(self, text: str, color: tuple[float, float, float, float]) -> None:
        """Draw text with RGBA color tuple"""

    def text_colored_centered(self, text: str, color: tuple[float, float, float, float]) -> None:
        """Draw centered text with RGBA color tuple"""

    def text_selectable(self, text: str, height: float = 0.0) -> None:
        """Draw selectable read-only text area"""

    def text_wrapped(self, text: str) -> None:
        """Draw word-wrapped text"""

    def text_disabled(self, text: str) -> None:
        """Draw greyed-out disabled text"""

    def bullet_text(self, text: str) -> None:
        """Draw text with a bullet point prefix"""

    def button(self, label: str, size: tuple[float, float] = (0.0, 0.0)) -> bool:
        """Draw a button, returns True if clicked"""

    def button_callback(self, label: str, callback: object | None = None, size: tuple[float, float] = (0.0, 0.0)) -> bool:
        """Draw a button that invokes callback on click"""

    def small_button(self, label: str) -> bool:
        """Draw a small inline button, returns True if clicked"""

    def checkbox(self, label: str, value: bool) -> tuple[bool, bool]:
        """Draw a checkbox, returns (changed, value)"""

    def radio_button(self, label: str, current: int, value: int) -> tuple[bool, int]:
        """Draw a radio button, returns (clicked, selected_value)"""

    def slider_float(self, label: str, value: float, min: float, max: float) -> tuple[bool, float]:
        """Draw a float slider, returns (changed, value)"""

    def slider_int(self, label: str, value: int, min: int, max: int) -> tuple[bool, int]:
        """Draw an int slider, returns (changed, value)"""

    def slider_float2(self, label: str, value: tuple[float, float], min: float, max: float) -> tuple[bool, tuple[float, float]]:
        """Draw a 2-component float slider, returns (changed, value)"""

    def slider_float3(self, label: str, value: tuple[float, float, float], min: float, max: float) -> tuple[bool, tuple[float, float, float]]:
        """Draw a 3-component float slider, returns (changed, value)"""

    def drag_float(self, label: str, value: float, speed: float = 1.0, min: float = 0.0, max: float = 0.0) -> tuple[bool, float]:
        """Draw a draggable float input, returns (changed, value)"""

    def drag_int(self, label: str, value: int, speed: float = 1.0, min: int = 0, max: int = 0) -> tuple[bool, int]:
        """Draw a draggable int input, returns (changed, value)"""

    def input_text(self, label: str, value: str) -> tuple[bool, str]:
        """Draw a text input field, returns (changed, value)"""

    def input_text_with_hint(self, label: str, hint: str, value: str) -> tuple[bool, str]:
        """Draw a text input with placeholder hint, returns (changed, value)"""

    def input_float(self, label: str, value: float, step: float = 0.0, step_fast: float = 0.0, format: str = '%.3f') -> tuple[bool, float]:
        """Draw a float input field with step buttons, returns (changed, value)"""

    def input_int(self, label: str, value: int, step: int = 1, step_fast: int = 100) -> tuple[bool, int]:
        """Draw an int input field with step buttons, returns (changed, value)"""

    def input_int_formatted(self, label: str, value: int, step: int = 0, step_fast: int = 0) -> tuple[bool, int]:
        """Draw a formatted int input field, returns (changed, value)"""

    def path_input(self, label: str, value: str, folder_mode: bool = True, dialog_title: str = '') -> tuple[bool, str]:
        """Draw a path input with browse button, returns (changed, path)"""

    def color_edit3(self, label: str, color: tuple[float, float, float]) -> tuple[bool, tuple[float, float, float]]:
        """Draw an RGB color editor, returns (changed, color)"""

    def color_edit4(self, label: str, color: tuple[float, float, float, float]) -> tuple[bool, tuple[float, float, float, float]]:
        """Draw an RGBA color editor, returns (changed, color)"""

    def color_button(self, label: str, color: tuple[float, float, float, float], size: tuple[float, float] = (0.0, 0.0)) -> bool:
        """Draw a color swatch button, returns True if clicked"""

    def combo(self, label: str, current_idx: int, items: Sequence[str]) -> tuple[bool, int]:
        """Draw a combo dropdown, returns (changed, index)"""

    def listbox(self, label: str, current_idx: int, items: Sequence[str], height_items: int = -1) -> tuple[bool, int]:
        """Draw a listbox, returns (changed, index)"""

    def separator(self) -> None:
        """Draw a horizontal separator line"""

    def spacing(self) -> None:
        """Add vertical spacing"""

    def same_line(self, offset: float = 0.0, spacing: float = -1.0) -> None:
        """Place next element on the same line"""

    def new_line(self) -> None:
        """Move cursor to a new line"""

    def indent(self, width: float = 0.0) -> None:
        """Increase indentation level"""

    def unindent(self, width: float = 0.0) -> None:
        """Decrease indentation level"""

    def set_next_item_width(self, width: float) -> None:
        """Set width of the next UI element"""

    def begin_group(self) -> None:
        """Begin a layout group"""

    def end_group(self) -> None:
        """End a layout group"""

    def collapsing_header(self, label: str, default_open: bool = False) -> bool:
        """Draw a collapsible header, returns True if open"""

    def tree_node(self, label: str) -> bool:
        """Draw a tree node, returns True if open"""

    def tree_node_ex(self, label: str, flags: str = '') -> bool:
        """Draw a tree node with flags string, returns True if open"""

    def tree_pop(self) -> None:
        """Pop a tree node level"""

    def begin_table(self, id: str, columns: int) -> bool:
        """Begin a table with given column count, returns True if visible"""

    def table_setup_column(self, label: str, width: float = 0.0) -> None:
        """Set up a table column with optional fixed width"""

    def end_table(self) -> None:
        """End the current table"""

    def table_next_row(self) -> None:
        """Advance to the next table row"""

    def table_next_column(self) -> None:
        """Advance to the next table column"""

    def table_set_column_index(self, column: int) -> bool:
        """Set active column by index, returns True if visible"""

    def table_headers_row(self) -> None:
        """Draw the table header row"""

    def table_set_bg_color(self, target: int, color: tuple[float, float, float, float]) -> None:
        """Set table background color for target region"""

    def button_styled(self, label: str, style: str, size: tuple[float, float] = (0.0, 0.0)) -> bool:
        """Draw a themed button (primary, success, warning, error, secondary)"""

    def push_item_width(self, width: float) -> None:
        """Push item width onto the stack"""

    def pop_item_width(self) -> None:
        """Pop item width from the stack"""

    def plot_lines(self, label: str, values: Sequence[float], scale_min: float = 3.4028234663852886e+38, scale_max: float = 3.4028234663852886e+38, size: tuple[float, float] = (0.0, 0.0)) -> None:
        """Draw a line plot from float values"""

    def selectable(self, label: str, selected: bool = False, height: float = 0.0) -> bool:
        """Draw a selectable item, returns True if clicked"""

    def begin_context_menu(self, id: str = '') -> bool:
        """Begin a styled right-click context menu"""

    def end_context_menu(self) -> None:
        """End context menu"""

    def begin_popup(self, id: str) -> bool:
        """Begin a popup by id, returns True if open"""

    def open_popup(self, id: str) -> None:
        """Open a popup by id"""

    def end_popup(self) -> None:
        """End the current popup"""

    def menu_item(self, label: str, enabled: bool = True) -> bool:
        """Draw a menu item, returns True if clicked"""

    def begin_menu(self, label: str) -> bool:
        """Begin a sub-menu, returns True if open"""

    def end_menu(self) -> None:
        """End the current sub-menu"""

    def input_text_enter(self, label: str, value: str) -> tuple[bool, str]:
        """Draw a text input that confirms on Enter, returns (entered, value)"""

    def set_keyboard_focus_here(self) -> None:
        """Set keyboard focus to the next widget"""

    def is_window_focused(self) -> bool:
        """Check if current window is focused"""

    def is_window_hovered(self) -> bool:
        """Check if current window is hovered"""

    def capture_keyboard_from_app(self, capture: bool = True) -> None:
        """Set keyboard capture flag for the application"""

    def capture_mouse_from_app(self, capture: bool = True) -> None:
        """Set mouse capture flag for the application"""

    def set_scroll_here_y(self, center_y_ratio: float = 0.5) -> None:
        """Scroll to current cursor Y position"""

    def get_cursor_screen_pos(self) -> tuple[float, float]:
        """Get cursor position in screen coordinates as (x, y)"""

    def get_window_pos(self) -> tuple[float, float]:
        """Get window position in screen coordinates as (x, y)"""

    def get_window_width(self) -> float:
        """Get current window width in pixels"""

    def get_text_line_height(self) -> float:
        """Get height of a single text line in pixels"""

    def begin_popup_modal(self, title: str) -> bool:
        """Begin a modal popup, returns True if visible"""

    def end_popup_modal(self) -> None:
        """End the current modal popup"""

    def close_current_popup(self) -> None:
        """Close the currently open popup"""

    def set_next_window_pos_center(self) -> None:
        """Center the next window on the main viewport"""

    def set_next_window_pos_viewport_center(self, always: bool = False) -> None:
        """Center the next window on the 3D viewport"""

    def set_next_window_focus(self) -> None:
        """Set focus to the next window"""

    def push_modal_style(self) -> None:
        """Push modal dialog style onto the style stack"""

    def pop_modal_style(self) -> None:
        """Pop modal dialog style from the style stack"""

    def get_content_region_avail(self) -> tuple[float, float]:
        """Get available content region as (width, height)"""

    def get_cursor_pos(self) -> tuple[float, float]:
        """Get cursor position within the window as (x, y)"""

    def set_cursor_pos_x(self, x: float) -> None:
        """Set horizontal cursor position within the window"""

    def calc_text_size(self, text: str) -> tuple[float, float]:
        """Calculate text dimensions as (width, height)"""

    def begin_disabled(self, disabled: bool = True) -> None:
        """Begin a disabled UI region"""

    def end_disabled(self) -> None:
        """End a disabled UI region"""

    def image(self, texture_id: int, size: tuple[float, float], tint: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
        """Draw an image from a GL texture ID"""

    def image_uv(self, texture_id: int, size: tuple[float, float], uv0: tuple[float, float], uv1: tuple[float, float], tint: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
        """Draw an image with custom UV coordinates"""

    def image_button(self, id: str, texture_id: int, size: tuple[float, float], tint: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> bool:
        """Draw an image button, returns True if clicked"""

    def toolbar_button(self, id: str, texture_id: int, size: tuple[float, float], selected: bool = False, disabled: bool = False, tooltip: str = '') -> bool:
        """Draw a toolbar-style icon button with selection state"""

    def begin_drag_drop_source(self) -> bool:
        """Begin a drag-drop source on the last item, returns True if dragging"""

    def set_drag_drop_payload(self, type: str, data: str) -> None:
        """Set the drag-drop payload type and data string"""

    def end_drag_drop_source(self) -> None:
        """End the drag-drop source"""

    def begin_drag_drop_target(self) -> bool:
        """Begin a drag-drop target on the last item, returns True if active"""

    def accept_drag_drop_payload(self, type: str) -> str | None:
        """Accept a drag-drop payload by type, returns data string or None"""

    def end_drag_drop_target(self) -> None:
        """End the drag-drop target"""

    def progress_bar(self, fraction: float, overlay: str = '', width: float = 0.0) -> None:
        """Draw a progress bar with fraction 0.0-1.0"""

    def set_tooltip(self, text: str) -> None:
        """Show tooltip on hover of the previous item"""

    def is_item_hovered(self) -> bool:
        """Check if the previous item is hovered"""

    def is_item_clicked(self, button: int = 0) -> bool:
        """Check if the previous item was clicked"""

    def is_item_active(self) -> bool:
        """Check if the previous item is active"""

    def is_mouse_double_clicked(self, button: int = 0) -> bool:
        """Check if mouse button was double-clicked this frame"""

    def is_mouse_dragging(self, button: int = 0) -> bool:
        """Check if mouse button is being dragged"""

    def get_mouse_wheel(self) -> float:
        """Get mouse wheel delta for this frame"""

    def get_mouse_delta(self) -> tuple[float, float]:
        """Get mouse movement delta as (dx, dy)"""

    def invisible_button(self, id: str, size: tuple[float, float]) -> bool:
        """Draw an invisible button region, returns True if clicked"""

    def set_cursor_pos(self, pos: tuple[float, float]) -> None:
        """Set cursor position within the window as (x, y)"""

    def begin_child(self, id: str, size: tuple[float, float], border: bool = False) -> bool:
        """Begin a child window region, returns True if visible"""

    def end_child(self) -> None:
        """End the child window region"""

    def begin_menu_bar(self) -> bool:
        """Begin the window menu bar, returns True if visible"""

    def end_menu_bar(self) -> None:
        """End the window menu bar"""

    def menu_item_toggle(self, label: str, shortcut: str, selected: bool) -> bool:
        """Draw a toggleable menu item with shortcut text"""

    def menu_item_shortcut(self, label: str, shortcut: str, enabled: bool = True) -> bool:
        """Draw a menu item with shortcut text"""

    def push_id(self, id: str) -> None:
        """Push a string ID onto the ID stack"""

    def push_id_int(self, id: int) -> None:
        """Push an integer ID onto the ID stack"""

    def pop_id(self) -> None:
        """Pop the last ID from the ID stack"""

    def begin_window(self, title: str, flags: int = 0) -> bool:
        """Begin a window, returns True if not collapsed"""

    def begin_window_closable(self, title: str, flags: int = 0) -> tuple[bool, bool]:
        """Begin a closable window, returns (visible, open)"""

    def end_window(self) -> None:
        """End the current window"""

    def push_window_style(self) -> None:
        """Push themed window rounding and padding styles"""

    def pop_window_style(self) -> None:
        """Pop window styles pushed by push_window_style"""

    def set_next_window_pos(self, pos: tuple[float, float], first_use: bool = False) -> None:
        """Set position of the next window as (x, y)"""

    def set_next_window_size(self, size: tuple[float, float], first_use: bool = False) -> None:
        """Set size of the next window as (width, height)"""

    def set_next_window_pos_centered(self, first_use: bool = False) -> None:
        """Center the next window on the main viewport"""

    def set_next_window_bg_alpha(self, alpha: float) -> None:
        """Set background alpha of the next window"""

    def get_viewport_pos(self) -> tuple[float, float]:
        """Get 3D viewport position as (x, y)"""

    def get_viewport_size(self) -> tuple[float, float]:
        """Get 3D viewport size as (width, height)"""

    def get_dpi_scale(self) -> float:
        """Get current DPI scale factor"""

    def set_mouse_cursor_hand(self) -> None:
        """Set mouse cursor to hand pointer"""

    def push_style_var(self, var: str, value: float) -> None:
        """Push a float style variable by name"""

    def push_style_var_vec2(self, var: str, value: tuple[float, float]) -> None:
        """Push a vec2 style variable by name"""

    def pop_style_var(self, count: int = 1) -> None:
        """Pop style variables from the stack"""

    def push_style_color(self, col: str, color: tuple[float, float, float, float]) -> None:
        """Push a style color override by name"""

    def pop_style_color(self, count: int = 1) -> None:
        """Pop style colors from the stack"""

    def prop(self, data: object, prop_id: str, text: str | None = None) -> tuple[bool, object]:
        """Draw a property widget based on metadata (auto-selects widget type)"""

    def row(self) -> SubLayout:
        """Create a horizontal row sub-layout"""

    def column(self) -> SubLayout:
        """Create a vertical column sub-layout"""

    def split(self, factor: float = 0.5) -> SubLayout:
        """Create a split sub-layout with given factor"""

    def box(self) -> SubLayout:
        """Create a bordered box sub-layout"""

    def grid_flow(self, columns: int = 0, even_columns: bool = True, even_rows: bool = True) -> SubLayout:
        """Create a responsive grid sub-layout"""

    def prop_enum(self, data: object, prop_id: str, value: str, text: str = '') -> bool:
        """Draw an enum toggle button for a property value"""

    def prop_search(self, data: object, prop_id: str, search_data: object, search_prop: str, text: str = '') -> tuple[bool, int]:
        """Searchable dropdown for selecting from a collection"""

    def template_list(self, list_type_id: str, list_id: str, data: object, prop_id: str, active_data: object, active_prop: str, rows: int = 5) -> tuple[int, int]:
        """UIList template for drawing custom lists"""

    def menu(self, menu_id: str, text: str = '', icon: str = '') -> None:
        """Inline menu reference"""

    def popover(self, panel_id: str, text: str = '', icon: str = '') -> None:
        """Panel popover"""

    def draw_circle(self, x: float, y: float, radius: float, color: tuple[float, float, float, float], segments: int = 32, thickness: float = 1.0) -> None:
        """Draw a circle outline at (x, y) with given radius and color"""

    def draw_circle_filled(self, x: float, y: float, radius: float, color: tuple[float, float, float, float], segments: int = 32) -> None:
        """Draw a filled circle at (x, y) with given radius and color"""

    def draw_rect(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a rectangle outline from (x0,y0) to (x1,y1)"""

    def draw_rect_filled(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float], background: bool = False) -> None:
        """Draw a filled rectangle from (x0,y0) to (x1,y1)"""

    def draw_rect_rounded(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float], rounding: float, thickness: float = 1.0, background: bool = False) -> None:
        """Draw a rounded rectangle outline"""

    def draw_rect_rounded_filled(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float], rounding: float, background: bool = False) -> None:
        """Draw a filled rounded rectangle"""

    def draw_triangle_filled(self, x0: float, y0: float, x1: float, y1: float, x2: float, y2: float, color: tuple[float, float, float, float], background: bool = False) -> None:
        """Draw a filled triangle"""

    def draw_line(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a line from (x0,y0) to (x1,y1)"""

    def draw_polyline(self, points: Sequence[tuple[float, float]], color: tuple[float, float, float, float], closed: bool = False, thickness: float = 1.0) -> None:
        """Draw a polyline through the given points"""

    def draw_poly_filled(self, points: Sequence[tuple[float, float]], color: tuple[float, float, float, float]) -> None:
        """Draw a filled convex polygon"""

    def draw_text(self, x: float, y: float, text: str, color: tuple[float, float, float, float], background: bool = False) -> None:
        """Draw text at (x, y) with given color"""

    def draw_window_rect_filled(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float]) -> None:
        """Draw a filled rectangle on the window draw list"""

    def draw_window_rect(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a rectangle outline on the window draw list"""

    def draw_window_rect_rounded(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float], rounding: float, thickness: float = 1.0) -> None:
        """Draw a rounded rectangle outline on the window draw list"""

    def draw_window_rect_rounded_filled(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float], rounding: float) -> None:
        """Draw a filled rounded rectangle on the window draw list"""

    def draw_window_line(self, x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a line on the window draw list"""

    def draw_window_text(self, x: float, y: float, text: str, color: tuple[float, float, float, float]) -> None:
        """Draw text on the window draw list"""

    def draw_window_triangle_filled(self, x0: float, y0: float, x1: float, y1: float, x2: float, y2: float, color: tuple[float, float, float, float]) -> None:
        """Draw a filled triangle on the window draw list"""

    def crf_curve_preview(self, label: str, gamma: float, toe: float, shoulder: float, gamma_r: float = 0.0, gamma_g: float = 0.0, gamma_b: float = 0.0) -> None:
        """Draw CRF tone curve preview widget"""

    def chromaticity_diagram(self, label: str, red_x: float, red_y: float, green_x: float, green_y: float, blue_x: float, blue_y: float, neutral_x: float, neutral_y: float, range: float = 0.5) -> tuple[bool, list[float]]:
        """Draw chromaticity diagram widget for color correction"""

def open_image_dialog(start_dir: str = '') -> str:
    """
    Open a file dialog to select an image file. Returns empty string if cancelled.
    """

def open_folder_dialog(title: str = 'Select Folder', start_dir: str = '') -> str:
    """Open a folder selection dialog. Returns empty string if cancelled."""

def open_ply_file_dialog(start_dir: str = '') -> str:
    """
    Open a file dialog to select a PLY file. Returns empty string if cancelled.
    """

def open_checkpoint_file_dialog() -> str:
    """
    Open a file dialog to select a checkpoint file. Returns empty string if cancelled.
    """

def open_json_file_dialog() -> str:
    """
    Open a file dialog to select a JSON config file. Returns empty string if cancelled.
    """

def save_json_file_dialog(default_name: str = 'config.json') -> str:
    """
    Open a save file dialog for JSON files. Returns empty string if cancelled.
    """

def save_ply_file_dialog(default_name: str = 'export.ply') -> str:
    """
    Open a save file dialog for PLY files. Returns empty string if cancelled.
    """

def save_sog_file_dialog(default_name: str = 'export.sog') -> str:
    """
    Open a save file dialog for SOG files. Returns empty string if cancelled.
    """

def save_spz_file_dialog(default_name: str = 'export.spz') -> str:
    """
    Open a save file dialog for SPZ files. Returns empty string if cancelled.
    """

def save_html_file_dialog(default_name: str = 'viewer.html') -> str:
    """
    Open a save file dialog for HTML viewer files. Returns empty string if cancelled.
    """

def open_dataset_folder_dialog() -> str:
    """
    Open a folder dialog to select a dataset. Returns empty string if cancelled.
    """

def open_video_file_dialog() -> str:
    """
    Open a file dialog to select a video file. Returns empty string if cancelled.
    """

def open_url(url: str) -> None:
    """Open a URL in the default browser."""

def set_tool(tool: str) -> None:
    """
    Switch to a toolbar tool (none, selection, translate, rotate, scale, mirror, brush, align, cropbox)
    """

class Key(enum.Enum):
    ESCAPE = 526

    ENTER = 525

    TAB = 512

    BACKSPACE = 523

    DELETE = 522

    SPACE = 524

    LEFT = 513

    RIGHT = 514

    UP = 515

    DOWN = 516

    HOME = 519

    END = 520

    F = 551

    I = 554

    M = 558

    R = 563

    T = 565

    _1 = 537

    MINUS = 598

    EQUAL = 602

    F2 = 573

def is_key_pressed(key: Key, repeat: bool = True) -> bool:
    """Check if a key was pressed this frame"""

def is_key_down(key: Key) -> bool:
    """Check if a key is currently held down"""

@overload
def tr(key: str) -> str:
    """Get localized string by key"""

@overload
def tr(key: str) -> str:
    """Translate a string key"""

def loc_set(key: str, value: str) -> None:
    """Override a localization string at runtime"""

def loc_clear(key: str) -> None:
    """Clear a localization override"""

def loc_clear_all() -> None:
    """Clear all localization overrides"""

def register_popup_draw_callback(callback: object) -> None:
    """Register a callback for drawing popup content"""

def unregister_popup_draw_callback(callback: object) -> None:
    """Unregister the popup draw callback"""

def on_show_dataset_load_popup(callback: object) -> None:
    """Register callback for ShowDatasetLoadPopup event"""

def on_show_resume_checkpoint_popup(callback: object) -> None:
    """Register callback for ShowResumeCheckpointPopup event"""

def on_request_exit(callback: object) -> None:
    """Register callback for RequestExit event"""

def set_exit_popup_open(open: bool) -> None:
    """Set exit popup open state (for window close callback)"""

def get_active_tool() -> str:
    """Get the currently active tool id from C++ EditorContext"""

def set_active_tool(id: str) -> None:
    """Set the active tool via C++ event"""

def set_active_operator(id: str, gizmo_type: str = '') -> None:
    """Set active operator with optional gizmo type"""

def get_active_operator() -> str:
    """Get the currently active operator id"""

def get_gizmo_type() -> str:
    """Get the gizmo type for the current operator"""

def clear_active_operator() -> None:
    """Clear the active operator"""

def has_active_operator() -> bool:
    """Check if an operator is currently active"""

def set_gizmo_type(type: str) -> None:
    """Set gizmo type without blocking camera"""

def clear_gizmo() -> None:
    """Clear gizmo type"""

def get_active_submode() -> str:
    """Get active selection submode"""

def set_selection_mode(mode: str) -> None:
    """Set selection mode"""

def execute_mirror(axis: str) -> None:
    """Execute mirror on axis (x, y, z)"""

def go_to_camera_view(cam_uid: int) -> None:
    """Go to camera view by UID"""

def apply_cropbox() -> None:
    """Apply the selected cropbox"""

def fit_cropbox_to_scene(use_percentile: bool = False) -> None:
    """Fit cropbox to scene bounds"""

def reset_cropbox() -> None:
    """Reset the selected cropbox"""

def add_cropbox(node_name: str) -> None:
    """Add a cropbox to the specified node"""

def add_ellipsoid(node_name: str) -> None:
    """Add an ellipsoid to the specified node"""

def apply_ellipsoid() -> None:
    """Apply the selected ellipsoid"""

def reset_ellipsoid() -> None:
    """Reset the selected ellipsoid"""

def fit_ellipsoid_to_scene(use_percentile: bool = False) -> None:
    """Fit ellipsoid to scene bounds"""

def duplicate_node(name: str) -> None:
    """Duplicate a node and its children"""

def merge_group(name: str) -> None:
    """Merge group children into a single PLY"""

def load_image_texture(path: str) -> tuple:
    """Load image as GL texture, returns (texture_id, width, height)"""

def load_thumbnail(path: str, max_size: int) -> tuple:
    """
    Load downscaled image as GL texture, returns (texture_id, width, height)
    """

def release_texture(texture_id: int) -> None:
    """Release an OpenGL texture"""

def preload_image_async(path: str) -> None:
    """Start async preload of image data"""

def is_preload_ready(path: str) -> bool:
    """Check if preloaded image is ready"""

def get_preloaded_texture(path: str) -> tuple:
    """Get preloaded image as GL texture, returns (texture_id, width, height)"""

def cancel_preload(path: str) -> None:
    """Cancel a pending preload"""

def clear_preload_cache() -> None:
    """Clear all pending preloads"""

def is_sequencer_visible() -> bool:
    """Check if sequencer panel is visible"""

def section_header(text: str) -> None:
    """Draw a section header with text and separator"""

def set_sequencer_visible(visible: bool) -> None:
    """Set sequencer panel visibility"""

def is_drag_hovering() -> bool:
    """Check if files are being dragged over the window"""

def is_startup_visible() -> bool:
    """Check if startup overlay is visible"""

def is_scene_empty() -> bool:
    """Check if no scene is loaded"""

def get_export_state() -> dict:
    """Get current export progress state"""

def cancel_export() -> None:
    """Cancel an ongoing export operation"""

def get_import_state() -> dict:
    """Get current import progress state"""

def dismiss_import() -> None:
    """Dismiss the import completion overlay"""

def get_video_export_state() -> dict:
    """Get current video export progress state"""

def cancel_video_export() -> None:
    """Cancel an ongoing video export operation"""

class SequencerUIState:
    @property
    def show_camera_path(self) -> bool:
        """Whether camera path is displayed in viewport"""

    @show_camera_path.setter
    def show_camera_path(self, arg: bool, /) -> None: ...

    @property
    def snap_to_grid(self) -> bool:
        """Whether keyframe snapping is enabled"""

    @snap_to_grid.setter
    def snap_to_grid(self, arg: bool, /) -> None: ...

    @property
    def snap_interval(self) -> float:
        """Snap grid interval in frames"""

    @snap_interval.setter
    def snap_interval(self, arg: float, /) -> None: ...

    @property
    def playback_speed(self) -> float:
        """Playback speed multiplier"""

    @playback_speed.setter
    def playback_speed(self, arg: float, /) -> None: ...

    @property
    def follow_playback(self) -> bool:
        """Whether viewport follows playback position"""

    @follow_playback.setter
    def follow_playback(self, arg: bool, /) -> None: ...

    @property
    def pip_preview_scale(self) -> float:
        """Picture-in-picture preview scale factor"""

    @pip_preview_scale.setter
    def pip_preview_scale(self, arg: float, /) -> None: ...

def get_sequencer_state() -> SequencerUIState:
    """Get sequencer UI state for modification"""

def has_keyframes() -> bool:
    """Check if sequencer has any keyframes"""

def save_camera_path(path: str) -> bool:
    """Save camera path to JSON file"""

def load_camera_path(path: str) -> bool:
    """Load camera path from JSON file"""

def clear_keyframes() -> None:
    """Clear all keyframes"""

def set_playback_speed(speed: float) -> None:
    """Set sequencer playback speed"""

def export_video(width: int, height: int, framerate: int, crf: int) -> None:
    """Export video with specified settings"""

def add_keyframe() -> None:
    """Add a keyframe at current camera position"""

def update_keyframe() -> None:
    """Update selected keyframe to current camera position"""

def play_pause() -> None:
    """Toggle sequencer playback"""

def draw_tools_section() -> None:
    """Draw tools section (C++ implementation)"""

def draw_console_button() -> None:
    """Draw system console button (C++ implementation)"""

def get_pivot_mode() -> int:
    """Get pivot mode (0=Origin, 1=Bounds)"""

def set_pivot_mode(mode: int) -> None:
    """Set pivot mode (0=Origin, 1=Bounds)"""

def get_transform_space() -> int:
    """Get transform space (0=Local, 1=World)"""

def set_transform_space(space: int) -> None:
    """Set transform space (0=Local, 1=World)"""

def request_thumbnail(video_id: str) -> None:
    """Request download of a YouTube thumbnail for the given video ID"""

def process_thumbnails() -> None:
    """Process pending thumbnail downloads (call every frame)"""

def is_thumbnail_ready(video_id: str) -> bool:
    """Check if a thumbnail is ready to be displayed"""

def get_thumbnail_texture(video_id: str) -> int:
    """Get the OpenGL texture ID for a downloaded thumbnail (0 if not ready)"""

def load_icon(name: str) -> int:
    """Load icon by name (e.g., 'selection.png'), returns texture ID"""

def load_scene_icon(name: str) -> int:
    """Load scene icon by name (e.g., 'visible', 'splat'), returns texture ID"""

def load_plugin_icon(icon_name: str, plugin_path: str, plugin_name: str) -> int:
    """Load icon from plugin folder with fallback to assets"""

def free_plugin_icons(plugin_name: str) -> None:
    """Free all icons associated with a plugin"""

def get_selected_camera_uid() -> int:
    """
    Get the UID of the currently selected camera, or -1 if no camera is selected
    """

def get_invert_masks() -> bool:
    """Get whether masks are inverted"""

def set_theme(name: str) -> None:
    """Set theme ('dark' or 'light')"""

def get_theme() -> str:
    """Get current theme name ('Dark' or 'Light')"""

def set_language(lang_code: str) -> None:
    """Set language by code (e.g., 'en', 'de')"""

def get_current_language() -> str:
    """Get current language code"""

def get_languages() -> list[tuple[str, str]]:
    """Get available languages as list of (code, name) tuples"""

def show_input_settings() -> None:
    """Show input settings window"""

def show_python_console() -> None:
    """Show Python console"""

def get_time() -> float:
    """Get time in seconds since application start"""

def set_cancel_operator_callback(callback: Callable) -> None:
    """Set callback for operator cancellation (called on ESC)"""

def get_selection_submode() -> int:
    """
    Get current selection sub-mode (0=Brush, 1=Rectangle, 2=Polygon, 3=Lasso, 4=Rings)
    """

def request_keyboard_capture(owner_id: str) -> None:
    """Request exclusive keyboard capture for a named owner"""

def release_keyboard_capture(owner_id: str) -> None:
    """Release keyboard capture for a named owner"""

def has_keyboard_capture_request() -> bool:
    """Check if any keyboard capture is currently active"""

class ModalEventType(enum.Enum):
    MouseButton = 0

    MouseMove = 1

    Scroll = 2

    Key = 3

class ModalEvent:
    @property
    def type(self) -> ModalEventType:
        """Event type (MouseButton, MouseMove, Scroll, Key)"""

    @property
    def x(self) -> float:
        """Mouse X position"""

    @property
    def y(self) -> float:
        """Mouse Y position"""

    @property
    def delta_x(self) -> float:
        """Mouse delta X"""

    @property
    def delta_y(self) -> float:
        """Mouse delta Y"""

    @property
    def button(self) -> int:
        """Mouse button index"""

    @property
    def action(self) -> int:
        """Action code (press, release, repeat)"""

    @property
    def key(self) -> int:
        """Key code for keyboard events"""

    @property
    def mods(self) -> int:
        """Modifier key bitmask (shift, ctrl, alt)"""

    @property
    def scroll_x(self) -> float:
        """Horizontal scroll offset"""

    @property
    def scroll_y(self) -> float:
        """Vertical scroll offset"""

    @property
    def over_gui(self) -> bool:
        """Whether mouse is over a GUI element"""

def set_modal_event_callback(callback: Callable) -> None:
    """Set callback for modal events (input dispatch to active operator)"""

def is_point_cloud_forced() -> bool:
    """Check if point cloud mode is forced (pre-training mode)"""

def get_fps() -> float:
    """Get current FPS"""

def get_gpu_memory() -> tuple[int, int]:
    """Get GPU memory (used, total) in bytes"""

def get_content_type() -> str:
    """Get content type (empty, splat_files, dataset)"""

def get_git_commit() -> str:
    """Get git commit hash"""

def get_split_view_info() -> dict:
    """Get split view info"""

def get_current_camera_id() -> int:
    """Get current camera ID for GT comparison"""

def get_split_view_mode() -> str:
    """Get split view mode (none, gt_comparison, ply_comparison)"""

def get_speed_overlay() -> tuple[float, float, float, float]:
    """
    Get speed overlay state (wasd_speed, wasd_alpha, zoom_speed, zoom_alpha)
    """

def register_property_group(group_id: str, group_name: str, property_group_class: object) -> None:
    """Register a Python PropertyGroup class with the property registry"""

def unregister_property_group(group_id: str) -> None:
    """Unregister a Python PropertyGroup from the property registry"""

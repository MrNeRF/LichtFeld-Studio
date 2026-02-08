from collections.abc import Sequence
import enum

import lichtfeld


class SelectionMode(enum.Enum):
    Replace = 0

    Add = 1

    Remove = 2

def begin_stroke() -> None:
    """Begin a new selection stroke (saves undo state)"""

def get_stroke_selection() -> lichtfeld.Tensor | None:
    """Get the current stroke selection tensor [N] uint8"""

def commit_stroke(mode: SelectionMode) -> bool:
    """Commit stroke to selection with given mode (Replace/Add/Remove)"""

def cancel_stroke() -> None:
    """Cancel current stroke (discard changes)"""

def is_stroke_active() -> bool:
    """Check if a stroke is currently active"""

def brush_select(x: float, y: float, radius: float) -> None:
    """
    Brush select at (x, y) with given radius. Accumulates into stroke selection.
    """

def ring_select(index: int, add: bool = True) -> None:
    """Select/deselect a single gaussian by index (for ring selection mode)."""

def rect_select(x0: float, y0: float, x1: float, y1: float) -> None:
    """Rectangle select from (x0, y0) to (x1, y1). Sets stroke selection."""

def polygon_select(vertices: Sequence[tuple[float, float]]) -> None:
    """
    Polygon select with given vertices [(x, y), ...]. Sets stroke selection.
    """

def lasso_select(points: Sequence[tuple[float, float]]) -> None:
    """Lasso (freehand polygon) select. Sets stroke selection."""

def set_preview(add_mode: bool = True) -> None:
    """Set current stroke as preview selection (green = add, red = remove)"""

def clear_preview() -> None:
    """Clear preview selection overlay"""

def draw_brush_circle(x: float, y: float, radius: float, add_mode: bool = True) -> None:
    """Draw brush circle overlay at (x, y)"""

def clear_brush_state() -> None:
    """Clear brush circle overlay"""

def draw_rect_preview(x0: float, y0: float, x1: float, y1: float, add_mode: bool = True) -> None:
    """Draw rectangle selection preview"""

def clear_rect_preview() -> None:
    """Clear rectangle selection preview"""

def draw_polygon_preview(points: Sequence[tuple[float, float]], closed: bool = False, add_mode: bool = True) -> None:
    """Draw polygon selection preview"""

def clear_polygon_preview() -> None:
    """Clear polygon selection preview"""

def draw_lasso_preview(points: Sequence[tuple[float, float]], add_mode: bool = True) -> None:
    """Draw lasso selection preview"""

def clear_lasso_preview() -> None:
    """Clear lasso selection preview"""

def set_output_screen_positions(enable: bool) -> None:
    """Enable/disable screen positions output during rendering"""

def has_screen_positions() -> bool:
    """Check if screen positions are available"""

def get_screen_positions() -> lichtfeld.Tensor | None:
    """Get screen positions tensor [N, 2]"""

def set_depth_filter(enabled: bool, depth_far: float = 100.0, frustum_half_width: float = 50.0) -> None:
    """Set depth filter for selection (frustum-shaped filter in camera space)"""

def get_depth_filter() -> tuple[bool, float, float]:
    """Get depth filter state: (enabled, depth_far, frustum_half_width)"""

def set_crop_filter(enabled: bool) -> None:
    """Enable/disable crop box filtering for selection"""

def apply_crop_filter() -> None:
    """Apply crop box filter to current stroke selection"""

def get_viewport_bounds() -> tuple[float, float, float, float]:
    """Get viewport bounds (x, y, width, height)"""

def get_render_scale() -> float:
    """Get current render scale factor"""

def screen_to_render(screen_x: float, screen_y: float) -> tuple[float, float]:
    """Convert screen coordinates to render coordinates"""

def get_hovered_gaussian_id() -> int:
    """Get ID of gaussian under cursor (-1 if none)"""

class PickResult:
    @property
    def index(self) -> int:
        """Gaussian index at current cursor position (-1 if unavailable)"""

    @property
    def depth(self) -> float:
        """Camera-space depth"""

    @property
    def world_position(self) -> tuple[float, float, float]:
        """Hit point in world coordinates"""

def pick_at_screen(screen_x: float, screen_y: float) -> PickResult | None:
    """
    Pick at screen coordinates. Returns PickResult with depth and world_position at the given coords. The index field reflects the gaussian under the current cursor, not the queried coordinates.
    """

def get_active_group() -> int:
    """Get the active selection group ID"""

def set_active_group(group_id: int) -> None:
    """Set the active selection group ID"""

def is_group_locked(group_id: int) -> bool:
    """Check if a selection group is locked"""

def trigger_flash() -> None:
    """Trigger selection flash animation feedback"""

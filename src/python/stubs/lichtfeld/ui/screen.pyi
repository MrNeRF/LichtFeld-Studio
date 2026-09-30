"""Editor screen areas and 3D views"""



def areas() -> list[dict]:
    """List the screen's areas as dicts (id, editor, geometry, view flags)"""

def editors() -> list[dict]:
    """List registered editor types"""

def split(area: int, direction: str = 'vertical', factor: float = 0.5) -> int:
    """
    Split an area. direction is 'vertical' (side by side) or 'horizontal' (stacked).
    """

def join(keep: int, remove: int) -> bool:
    """Join two neighbouring areas, keeping `keep`"""

def close(area: int) -> bool:
    """Close an area"""

def swap(a: int, b: int) -> bool:
    """Swap the editors of two areas"""

def set_editor(area: int, editor: str) -> bool:
    """Show an editor in an area"""

def open_editor(editor: str) -> int:
    """Open an editor, splitting if needed. Returns the area id."""

def close_editor(editor: str) -> bool:
    """Close every area showing this editor"""

def toggle_maximized(area: int) -> bool:
    """Maximize an area, or restore if it is already maximized"""

def reset() -> None:
    """Reset the screen to the default layout"""

def active_view() -> int:
    """Id of the active 3D view"""

def set_active_view(view: int) -> bool:
    """Set the active 3D view"""

def view_command(view: int, command: str) -> bool:
    """Run a view command on a 3D view"""

def view_camera(view: int) -> dict:
    """Camera state of a 3D view"""

def set_view_camera(view: int, eye: object, target: object, up: object = (0.0, 1.0, 0.0)) -> bool:
    """Point a 3D view's camera at target from eye"""

def view_settings(view: int) -> dict:
    """ViewSettings of a 3D view as a dict"""

def set_view_settings(arg0: int, /, **kwargs) -> None:
    """Update ViewSettings fields on a 3D view. Unknown fields raise."""

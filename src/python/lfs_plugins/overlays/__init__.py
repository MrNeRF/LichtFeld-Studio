"""Viewport overlay panels for progress indicators and empty states."""

from .empty_state import EmptyStateOverlay
from .drag_drop import DragDropOverlay
from .import_progress import ImportProgressOverlay
from .video_progress import VideoProgressOverlay


def register():
    """Register all overlay panels."""
    import lichtfeld as lf
    lf.register_class(EmptyStateOverlay)
    lf.register_class(DragDropOverlay)
    lf.register_class(ImportProgressOverlay)
    lf.register_class(VideoProgressOverlay)


def unregister():
    """Unregister all overlay panels."""
    pass

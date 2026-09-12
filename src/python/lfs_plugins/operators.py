"""Register callbacks for the native operator system."""

import lichtfeld as lf


def _on_cancel_active_operator():
    """Callback from C++ when ESC is pressed."""
    lf.ui.ops.cancel_modal()


def register():
    """Register operator system callbacks."""
    lf.ui.set_cancel_operator_callback(_on_cancel_active_operator)


def unregister():
    """Unregister operator system."""
    pass

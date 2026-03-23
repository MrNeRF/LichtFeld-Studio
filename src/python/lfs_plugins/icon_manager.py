# Centralized icon management for Python plugins.
# All icon loading defers to C++ which maintains a global cache.


def _runtime():
    import lichtfeld as lf

    return lf


def get_icon(name: str) -> int:
    """Load a generic icon from assets/icon/{name}.png

    Returns the OpenGL texture ID, or 0 if loading failed.
    C++ maintains the cache - this is the preferred API for toolbar icons.
    """
    try:
        lf = _runtime()
        return lf.load_icon(name)
    except Exception:
        return 0


def get_ui_icon(name: str) -> int:
    """Load a UI icon from assets/icon/{name}

    Note: name should include the file extension (e.g., "move.png").
    Returns the OpenGL texture ID, or 0 if loading failed.
    """
    try:
        lf = _runtime()
        return lf.ui.load_icon(name)
    except Exception:
        return 0


def get_scene_icon(name: str) -> int:
    """Load a scene panel icon from assets/icon/scene/{name}.png

    Returns the OpenGL texture ID, or 0 if loading failed.
    """
    try:
        lf = _runtime()
        return lf.ui.load_scene_icon(name)
    except Exception:
        return 0


def get_plugin_icon(name: str, plugin_path: str, plugin_name: str) -> int:
    """Load icon from plugin folder with fallback to assets.

    Looks for {plugin_path}/icons/{name}.png first, then falls back
    to assets/icon/{name}.png.

    Returns the OpenGL texture ID, or 0 if loading failed.
    """
    try:
        lf = _runtime()
        return lf.ui.load_plugin_icon(name, plugin_path, plugin_name)
    except Exception:
        return 0

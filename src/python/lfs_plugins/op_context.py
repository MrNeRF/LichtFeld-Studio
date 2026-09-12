# Scene context used by tool activation checks
# Named op_context.py to avoid conflict with the existing context.py (plugin context)

import lichtfeld as lf


class OperatorContext:
    """Scene queries used to decide whether a tool can activate."""

    def _get_app_context(self):
        """Get the unified application context."""
        return lf.ui.context()

    @property
    def has_scene(self) -> bool:
        return self._get_app_context().has_scene

    @property
    def has_selection(self) -> bool:
        return self._get_app_context().has_selection

    @property
    def num_gaussians(self) -> int:
        return self._get_app_context().num_gaussians

    @property
    def can_transform(self) -> bool:
        return lf.can_transform_selection()

# Global context singleton
_context: OperatorContext = None


def get_context() -> OperatorContext:
    """Get the global operator context."""
    global _context
    if _context is None:
        _context = OperatorContext()
    return _context

"""Unified undo system"""



def undo() -> None:
    """Undo last operation"""

def redo() -> None:
    """Redo last undone operation"""

def can_undo() -> bool:
    """Check if undo is available"""

def can_redo() -> bool:
    """Check if redo is available"""

def undo_name() -> str:
    """Get name of next undo operation"""

def redo_name() -> str:
    """Get name of next redo operation"""

def clear() -> None:
    """Clear undo history"""

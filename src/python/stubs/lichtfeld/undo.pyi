

def push(name: str, undo: object, redo: object, validate: bool = False) -> None:
    """Push an undo step with undo/redo functions"""

def undo() -> None:
    """Undo last operation"""

def redo() -> None:
    """Redo last undone operation"""

def can_undo() -> bool:
    """Check if undo is available"""

def can_redo() -> bool:
    """Check if redo is available"""

def clear() -> None:
    """Clear undo history"""

def get_undo_name() -> str:
    """Get name of next undo operation"""

def get_redo_name() -> str:
    """Get name of next redo operation"""

class Transaction:
    def __init__(self, name: str = 'Grouped Changes') -> None: ...

    def __enter__(self) -> Transaction:
        """Begin transaction context"""

    def __exit__(self, *args) -> bool:
        """Commit transaction on context exit"""

    def add(self, undo: object, redo: object) -> None:
        """Add an undo/redo pair to the transaction"""

def transaction(name: str = 'Grouped Changes') -> Transaction:
    """Create a transaction for grouping undo steps"""

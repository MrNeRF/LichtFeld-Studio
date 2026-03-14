"""Undo/redo system"""

from collections.abc import Callable


def push(name: str, undo: object, redo: object, validate: bool = False) -> None:
    """Push an undo step with undo/redo functions"""

def undo() -> bool:
    """Undo last operation"""

def redo() -> bool:
    """Redo last undone operation"""

def jump(stack: str, count: int) -> dict:
    """Apply multiple undo/redo steps for history navigation"""

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

def undo_names() -> list[str]:
    """Get the undo stack names, newest first"""

def redo_names() -> list[str]:
    """Get the redo stack names, newest first"""

def undo_bytes() -> int:
    """Get estimated bytes retained by undo history"""

def redo_bytes() -> int:
    """Get estimated bytes retained by redo history"""

def transaction_bytes() -> int:
    """Get estimated bytes retained by active grouped history transactions"""

def total_bytes() -> int:
    """Get estimated bytes retained by undo and redo history"""

def has_active_transaction() -> bool:
    """Check if a grouped history transaction is active"""

def transaction_depth() -> int:
    """Get the current grouped history transaction nesting depth"""

def active_transaction_name() -> str:
    """Get the current grouped history transaction label"""

def generation() -> int:
    """Get the shared history change generation"""

def subscribe(callback: Callable) -> int:
    """Subscribe to shared history changes and return a subscription id"""

def unsubscribe(subscription_id: int) -> None:
    """Unsubscribe a shared history observer"""

def stack() -> dict:
    """Get the structured undo/redo stack state"""

class Transaction:
    def __init__(self, name: str = 'Grouped Changes') -> None: ...

    def __enter__(self) -> Transaction:
        """Begin transaction context"""

    def __exit__(self, arg0: object, arg1: object, arg2: object, /) -> bool:
        """Commit transaction on context exit"""

    def add(self, undo: object, redo: object) -> None:
        """Add an undo/redo pair to the transaction"""

def transaction(name: str = 'Grouped Changes') -> Transaction:
    """Create a transaction for grouping undo steps"""

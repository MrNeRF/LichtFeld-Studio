"""Reactive C++ app store bridge"""



def set(field: str, value: object | None) -> None:
    """Set an app store field"""

def get(arg: str, /) -> object:
    """Get an app store field"""

def subscribe(arg0: str, arg1: object, /) -> int:
    """Subscribe to an app store field"""

def unsubscribe(arg: int, /) -> None:
    """Unsubscribe from an app store field"""

def begin_batch() -> None:
    """Begin a batched app store update"""

def end_batch() -> None:
    """End a batched app store update"""

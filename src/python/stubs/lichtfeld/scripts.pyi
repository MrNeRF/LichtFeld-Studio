"""Python script management"""

from collections.abc import Sequence


def get_scripts() -> list:
    """Get list of loaded scripts with their state"""

def set_script_enabled(index: int, enabled: bool) -> None:
    """Enable or disable a script by index"""

def set_script_error(index: int, error: str) -> None:
    """Set error message for a script (empty to clear)"""

def clear_errors() -> None:
    """Clear all script errors"""

def clear() -> None:
    """Clear all scripts"""

def run(paths: Sequence[str]) -> dict:
    """Run scripts by paths, returns {success: bool, error: str}"""

def get_enabled_paths() -> list[str]:
    """Get list of enabled script paths"""

def count() -> int:
    """Get number of loaded scripts"""

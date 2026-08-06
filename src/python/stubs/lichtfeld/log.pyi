"""Logging utilities"""



def info(message: str) -> None:
    """Log an info message"""

def debug(message: str) -> None:
    """Log a debug message"""

def warn(message: str) -> None:
    """Log a warning message"""

def error(message: str) -> None:
    """Log an error message"""

def buffered_text(max_bytes: int = 1048576) -> str:
    """Return the tail of the buffered session log at a line boundary."""

def log_file_path() -> str:
    """Return the durable session log path."""

def previous_session() -> dict | None:
    """Return the previous process session breadcrumb, if available."""

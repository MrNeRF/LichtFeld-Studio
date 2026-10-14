"""Logging utilities"""



def snapshot(previous_generation: int = -1) -> object:
    """Return a consistent bounded log snapshot, or None if unchanged."""

def clear_buffer() -> None:
    """Clear the in-memory log buffer without modifying log files."""

def set_buffer_limit(limit: int) -> None: ...

def capture_level() -> int: ...

def set_capture_level(level: int) -> None: ...

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

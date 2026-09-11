# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin data structures."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any
from enum import Enum
import re


class PluginState(Enum):
    """Plugin lifecycle states."""

    UNLOADED = "unloaded"
    INSTALLING = "installing"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


_PLUGIN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_plugin_name(name: str) -> str:
    """Validate a plugin name before it is used as a filesystem component."""
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid plugin name: project.name must be a non-empty string")
    if name in {".", ".."} or not _PLUGIN_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid plugin name {name!r}: use one safe path component "
            "(ASCII letters, numbers, '.', '_' or '-') of at most 128 characters"
        )
    return name


@dataclass
class PluginInfo:
    """Plugin metadata parsed from pyproject.toml."""

    name: str
    version: str
    path: Path
    description: str = ""
    author: str = ""
    entry_point: str = "__init__"
    dependencies: List[str] = field(default_factory=list)
    auto_start: bool = False
    hot_reload: bool = True
    plugin_api: str = ""
    lichtfeld_version: str = ""
    required_features: List[str] = field(default_factory=list)


@dataclass
class PluginInstance:
    """Runtime state of a loaded plugin."""

    info: PluginInfo
    state: PluginState = PluginState.UNLOADED
    module: Optional[Any] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    venv_path: Optional[Path] = None
    file_mtimes: dict = field(default_factory=dict)
    sys_paths: List[str] = field(default_factory=list)

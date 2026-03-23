# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin data structures."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any
from enum import Enum


class PluginState(Enum):
    """Plugin lifecycle states."""

    UNLOADED = "unloaded"
    INSTALLING = "installing"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


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

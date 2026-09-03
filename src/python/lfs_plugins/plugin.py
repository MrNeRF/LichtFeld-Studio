# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin data structures."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional
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
    localization_tokens: List[int] = field(default_factory=list)


def iter_plugin_watch_files(plugin_path: Path) -> Iterator[Path]:
    """Yield source files whose changes require a plugin reload."""
    for py_file in plugin_path.rglob("*.py"):
        if ".venv" not in py_file.parts:
            yield py_file

    locales_dir = plugin_path / "locales"
    if locales_dir.is_dir():
        yield from (
            path
            for path in locales_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )

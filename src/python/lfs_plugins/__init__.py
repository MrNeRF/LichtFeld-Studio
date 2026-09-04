# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""LichtFeld Plugin System."""

from typing import TYPE_CHECKING

from .types import Menu, Operator, Panel
from .capabilities import Capability, CapabilityRegistry, CapabilitySchema
from .context import CapabilityBroker, PluginContext, SceneContext, ViewContext
from .errors import (
    PluginDependencyError,
    PluginError,
    PluginLoadCancelled,
    PluginLoadError,
    PluginNotFoundError,
    PluginVersionError,
    RegistryError,
    RegistryOfflineError,
    VersionNotFoundError,
)
from .plugin import PluginInfo, PluginInstance, PluginState
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .utils import cleanup_torch_model, get_gpu_memory, log_gpu_memory

if TYPE_CHECKING:
    from .panels import PluginMarketplacePanel as PluginMarketplacePanel


def _load_builtin_panel_api():
    from .panels import PluginMarketplacePanel, register_builtin_panels as _register_builtin_panels

    return PluginMarketplacePanel, _register_builtin_panels


def register_builtin_panels():
    try:
        _, builtin_register = _load_builtin_panel_api()
    except ModuleNotFoundError as exc:
        if exc.name != "lichtfeld":
            raise
        raise RuntimeError("register_builtin_panels() requires the lichtfeld runtime")
    return builtin_register()


_LAZY_EXPORTS = {
    "PluginManager": ("manager", "PluginManager"),
    "PluginMarketplaceCatalog": ("marketplace", "PluginMarketplaceCatalog"),
    "MarketplacePluginEntry": ("marketplace", "MarketplacePluginEntry"),
    "RegistryClient": ("registry", "RegistryClient"),
    "RegistryPluginInfo": ("registry", "RegistryPluginInfo"),
    "RegistryVersionInfo": ("registry", "RegistryVersionInfo"),
    "PluginSettings": ("settings", "PluginSettings"),
    "SettingsManager": ("settings", "SettingsManager"),
    "create_plugin": ("templates", "create_plugin"),
}
_LAZY_MODULES = {"manager", "marketplace", "registry", "settings", "templates", "installer"}


def __getattr__(name):
    if name == "PluginMarketplacePanel":
        panel_cls, _ = _load_builtin_panel_api()
        return panel_cls

    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attribute = _LAZY_EXPORTS[name]
        return getattr(import_module(f"{__name__}.{module_name}"), attribute)

    if name in _LAZY_MODULES:
        from importlib import import_module

        return import_module(f"{__name__}.{name}")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Panel",
    "Operator",
    "Menu",
    "PluginManager",
    "PluginMarketplaceCatalog",
    "MarketplacePluginEntry",
    "PluginInfo",
    "PluginState",
    "PluginInstance",
    "PluginError",
    "PluginLoadError",
    "PluginDependencyError",
    "PluginLoadCancelled",
    "PluginVersionError",
    "RegistryError",
    "RegistryOfflineError",
    "PluginNotFoundError",
    "VersionNotFoundError",
    "RegistryClient",
    "RegistryPluginInfo",
    "RegistryVersionInfo",
    "PluginMarketplacePanel",
    "register_builtin_panels",
    "Capability",
    "CapabilityRegistry",
    "CapabilitySchema",
    "PluginContext",
    "SceneContext",
    "ViewContext",
    "CapabilityBroker",
    "PluginSettings",
    "SettingsManager",
    "ScrubFieldController",
    "ScrubFieldSpec",
    "create_plugin",
    "get_gpu_memory",
    "log_gpu_memory",
    "cleanup_torch_model",
]

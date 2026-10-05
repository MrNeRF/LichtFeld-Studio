# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""LichtFeld Plugin System."""

from typing import TYPE_CHECKING

# Install the process-wide Python -> native logger bridge before any lazy
# plugin module creates a child logger.
try:
    from . import logging_bridge as _logging_bridge
    _logging_bridge.install()
except Exception:
    import logging
    logging.getLogger(__name__).exception("Could not install the native logging bridge")

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
    "Menu": ("types", "Menu"),
    "Operator": ("types", "Operator"),
    "Panel": ("types", "Panel"),
    "Capability": ("capabilities", "Capability"),
    "CapabilityRegistry": ("capabilities", "CapabilityRegistry"),
    "CapabilitySchema": ("capabilities", "CapabilitySchema"),
    "CapabilityBroker": ("context", "CapabilityBroker"),
    "PluginContext": ("context", "PluginContext"),
    "SceneContext": ("context", "SceneContext"),
    "ViewContext": ("context", "ViewContext"),
    "PluginDependencyError": ("errors", "PluginDependencyError"),
    "PluginError": ("errors", "PluginError"),
    "PluginLoadCancelled": ("errors", "PluginLoadCancelled"),
    "PluginLoadError": ("errors", "PluginLoadError"),
    "PluginNotFoundError": ("errors", "PluginNotFoundError"),
    "PluginVersionError": ("errors", "PluginVersionError"),
    "RegistryError": ("errors", "RegistryError"),
    "RegistryOfflineError": ("errors", "RegistryOfflineError"),
    "VersionNotFoundError": ("errors", "VersionNotFoundError"),
    "PluginInfo": ("plugin", "PluginInfo"),
    "PluginInstance": ("plugin", "PluginInstance"),
    "PluginState": ("plugin", "PluginState"),
    "ScrubFieldController": ("scrub_fields", "ScrubFieldController"),
    "ScrubFieldSpec": ("scrub_fields", "ScrubFieldSpec"),
    "cleanup_torch_model": ("utils", "cleanup_torch_model"),
    "get_gpu_memory": ("utils", "get_gpu_memory"),
    "log_gpu_memory": ("utils", "log_gpu_memory"),
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

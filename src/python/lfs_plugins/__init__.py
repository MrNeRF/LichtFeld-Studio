# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""LichtFeld Plugin System."""

from .types import Operator, Panel
from .capabilities import Capability, CapabilityRegistry, CapabilitySchema
from .context import CapabilityBroker, PluginContext, SceneContext, ViewContext
from .errors import (
    PluginDependencyError,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    PluginVersionError,
    RegistryError,
    RegistryOfflineError,
    VersionNotFoundError,
)
from .manager import PluginManager
from .marketplace import (
    MarketplacePluginEntry,
    PluginMarketplaceCatalog,
)
from .panels import PluginMarketplacePanel, register_builtin_panels
from .plugin import PluginInfo, PluginInstance, PluginState
from .registry import RegistryClient, RegistryPluginInfo, RegistryVersionInfo
from .settings import PluginSettings, SettingsManager
from .templates import create_plugin
from .utils import cleanup_torch_model, get_gpu_memory, log_gpu_memory

__all__ = [
    "Panel",
    "Operator",
    "PluginManager",
    "PluginMarketplaceCatalog",
    "MarketplacePluginEntry",
    "PluginInfo",
    "PluginState",
    "PluginInstance",
    "PluginError",
    "PluginLoadError",
    "PluginDependencyError",
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
    "create_plugin",
    "get_gpu_memory",
    "log_gpu_memory",
    "cleanup_torch_model",
]

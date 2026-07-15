# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin system exceptions."""


class PluginError(Exception):
    """Base exception for plugin errors."""


class PluginLoadError(PluginError):
    """Failed to load plugin."""


class PluginDependencyError(PluginError):
    """Failed to install dependencies."""


class PluginLoadCancelled(PluginError):
    """Plugin loading was cancelled by the host."""


class PluginVersionError(PluginError):
    """Plugin version incompatible with current LichtFeld."""


class RegistryError(PluginError):
    """Base exception for registry errors."""


class RegistryOfflineError(RegistryError):
    """Cannot reach registry and no cache available."""


class PluginNotFoundError(RegistryError):
    """Plugin not found in registry."""


class VersionNotFoundError(RegistryError):
    """Requested version not found."""

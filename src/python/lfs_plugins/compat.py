# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin API compatibility contract for the v1 release."""

from __future__ import annotations

from collections.abc import Iterable

try:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover - packaging is expected in release builds
    InvalidSpecifier = InvalidVersion = None
    SpecifierSet = Version = None


LICHTFELD_VERSION = "0.5.0"
PLUGIN_API_VERSION = "1.0"
SUPPORTED_PLUGIN_FEATURES = (
    "capabilities.v1",
    "menus.v1",
    "operators.v1",
    "panels.v1",
    "settings.v1",
    "signals.v1",
)
DEFAULT_PLUGIN_API_SPEC = ">=1,<2"
DEFAULT_LICHTFELD_VERSION_SPEC = ">=0.4.2"
_V1_MANIFEST_HINT = (
    "v1 manifest requires tool.lichtfeld.plugin_api, "
    "tool.lichtfeld.lichtfeld_version, and tool.lichtfeld.required_features"
)


def validate_specifier(field_name: str, value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"{field_name} must be a non-empty version specifier string"
    if SpecifierSet is None:
        return None
    try:
        SpecifierSet(value.strip())
    except InvalidSpecifier:
        return f"{field_name} must be a valid PEP 440 version specifier"
    return None


def validate_required_features(value: object) -> str | None:
    if not isinstance(value, list):
        return "tool.lichtfeld.required_features must be an array of strings"
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return "tool.lichtfeld.required_features must contain only non-empty strings"
    return None


def missing_required_manifest_fields(lf_section: dict) -> list[str]:
    errors = []
    if "plugin_api" not in lf_section:
        errors.append(
            "pyproject.toml: missing tool.lichtfeld.plugin_api "
            f"({_V1_MANIFEST_HINT})"
        )
    if "lichtfeld_version" not in lf_section:
        errors.append(
            "pyproject.toml: missing tool.lichtfeld.lichtfeld_version "
            f"({_V1_MANIFEST_HINT})"
        )
    if "required_features" not in lf_section:
        errors.append(
            "pyproject.toml: missing tool.lichtfeld.required_features "
            f"({_V1_MANIFEST_HINT})"
        )
    return errors


def validate_manifest_compatibility_fields(lf_section: dict) -> list[str]:
    errors = missing_required_manifest_fields(lf_section)
    if errors:
        return errors

    plugin_api_error = validate_specifier("tool.lichtfeld.plugin_api", lf_section.get("plugin_api"))
    if plugin_api_error:
        errors.append(f"pyproject.toml: {plugin_api_error}")

    lichtfeld_error = validate_specifier("tool.lichtfeld.lichtfeld_version", lf_section.get("lichtfeld_version"))
    if lichtfeld_error:
        errors.append(f"pyproject.toml: {lichtfeld_error}")

    features_error = validate_required_features(lf_section.get("required_features"))
    if features_error:
        errors.append(f"pyproject.toml: {features_error}")

    return errors


def compatibility_errors(
    plugin_api: str,
    lichtfeld_version: str,
    required_features: Iterable[str],
    *,
    current_plugin_api: str = PLUGIN_API_VERSION,
    current_lichtfeld_version: str = LICHTFELD_VERSION,
    supported_features: Iterable[str] = SUPPORTED_PLUGIN_FEATURES,
) -> list[str]:
    errors = []

    if SpecifierSet is not None and Version is not None:
        try:
            if Version(current_plugin_api) not in SpecifierSet(plugin_api):
                errors.append(
                    f"requires plugin API {plugin_api}, but host provides {current_plugin_api}"
                )
        except (InvalidSpecifier, InvalidVersion):
            errors.append(f"declares invalid plugin API specifier {plugin_api!r}")

        try:
            if Version(current_lichtfeld_version) not in SpecifierSet(lichtfeld_version):
                errors.append(
                    f"requires LichtFeld {lichtfeld_version}, but host is {current_lichtfeld_version}"
                )
        except (InvalidSpecifier, InvalidVersion):
            errors.append(f"declares invalid LichtFeld version specifier {lichtfeld_version!r}")

    missing = sorted(set(required_features) - set(supported_features))
    if missing:
        errors.append(f"requires unsupported features: {', '.join(missing)}")

    return errors

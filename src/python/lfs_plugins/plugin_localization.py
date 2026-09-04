# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation and loading for plugin-owned localization catalogs."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


MAX_CATALOG_BYTES = 1024 * 1024
MAX_CATALOG_ENTRIES = 4096
MAX_KEY_LENGTH = 256
MAX_NESTING_DEPTH = 32
MAX_OWNER_LENGTH = 128
MAX_VALUE_BYTES = 16 * 1024

# Keep the standalone checker independent of the native UI. A regression test
# requires this set to match the host's locale_index.json exactly.
SUPPORTED_LANGUAGES = frozenset({"de", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "zh"})

_LANGUAGE_RE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*\Z")
_OWNER_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_RELATIVE_KEY_RE = re.compile(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*\Z")


class PluginCatalogError(ValueError):
    """A plugin localization catalog violates the host contract."""


@dataclass(frozen=True)
class PluginCatalogBundle:
    owner_id: str
    catalogs: dict[str, dict[str, str]]


def canonical_plugin_id(project_name: str) -> str:
    """Return the stable namespace component derived from project.name."""
    owner_id = re.sub(r"[-_.]+", "-", project_name).lower()
    if (
        not owner_id
        or len(owner_id) > MAX_OWNER_LENGTH
        or _OWNER_RE.fullmatch(owner_id) is None
    ):
        raise PluginCatalogError(
            "project.name cannot be converted to a valid localization owner id"
        )
    return owner_id


def owner_collision_errors(project_names: Iterable[str]) -> dict[str, str]:
    """Reserve one canonical namespace per distinct discovered project name."""
    owners: dict[str, set[str]] = {}
    for name in project_names:
        try:
            owner = canonical_plugin_id(name)
        except PluginCatalogError:
            continue  # Legacy plugins without catalogs need no localization id.
        owners.setdefault(owner, set()).add(name)
    return {
        name: f"localization owner 'plugins.{owner}' collides between projects "
        f"{', '.join(repr(item) for item in sorted(names))}; use distinct project names"
        for owner, names in owners.items() if len(names) > 1
        for name in names
    }


def _neighbor_plugin_names(plugin_dir: Path) -> list[str]:
    """Read only sibling manifests for the standalone installed-plugin checker."""
    names = []
    for entry in plugin_dir.parent.iterdir():
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        manifest = entry / "pyproject.toml"
        if not manifest.is_file():
            continue
        try:
            with manifest.open("rb") as stream:
                raw = stream.read(MAX_CATALOG_BYTES + 1)
            if len(raw) > MAX_CATALOG_BYTES:
                continue
            data = tomllib.loads(raw.decode("utf-8"))
            name = data.get("project", {}).get("name")
            if isinstance(name, str) and "lichtfeld" in data.get("tool", {}):
                names.append(name)
        except (OSError, UnicodeError, ValueError, AttributeError, TypeError):
            continue  # Malformed neighboring manifests are handled by discovery.
    return names


def _contained_path(path: Path, plugin_root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PluginCatalogError(f"{path.name!r}: cannot resolve catalog path") from exc
    if not resolved.is_relative_to(plugin_root):
        raise PluginCatalogError(f"{path.name!r}: catalog path must stay inside the plugin directory")
    return resolved


def read_plugin_catalogs(plugin_dir: str | Path, project_name: str) -> PluginCatalogBundle:
    """Read and validate optional ``locales/*.json`` plugin catalogs."""
    locales_dir = Path(plugin_dir) / "locales"
    if not locales_dir.exists() and not locales_dir.is_symlink():
        return PluginCatalogBundle(owner_id="", catalogs={})
    plugin_root = Path(plugin_dir).resolve(strict=True)
    locales_dir = _contained_path(locales_dir, plugin_root)
    if not locales_dir.is_dir():
        raise PluginCatalogError("locales must be a directory")

    catalog_paths = sorted(
        (
            path
            for path in locales_dir.iterdir()
            if path.suffix.lower() == ".json"
        ),
        key=lambda path: path.name,
    )
    if not catalog_paths:
        return PluginCatalogBundle(owner_id="", catalogs={})

    owner_id = canonical_plugin_id(project_name)
    catalogs: dict[str, dict[str, str]] = {}
    seen_languages: dict[str, str] = {}
    for path in catalog_paths:
        if path.suffix != ".json":
            raise PluginCatalogError(f"locales/{path.name}: extension must be lowercase .json")

        language = path.stem
        normalized_language = language.lower()
        if language != normalized_language or not _valid_language_code(language):
            raise PluginCatalogError(
                f"locales/{path.name}: filename must be a lowercase language code"
            )
        if normalized_language in seen_languages:
            raise PluginCatalogError(
                f"locales/{path.name}: duplicates language catalog "
                f"locales/{seen_languages[normalized_language]}"
            )
        seen_languages[normalized_language] = path.name
        if language not in SUPPORTED_LANGUAGES:
            raise PluginCatalogError(
                f"locales/{path.name}: unsupported host language; expected one of "
                f"{', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )
        resolved = _contained_path(path, plugin_root)
        if not resolved.is_file():
            raise PluginCatalogError(f"locales/{path.name}: catalog must be a regular file")
        catalogs[language] = _read_catalog(path)

    if "en" not in catalogs:
        raise PluginCatalogError("locales/en.json is required when plugin catalogs are present")

    english = catalogs["en"]
    english_placeholders = {
        key: _placeholder_signature(value, f"locales/en.json:{key}")
        for key, value in english.items()
    }
    for language, translations in catalogs.items():
        if language == "en":
            continue
        for key, value in translations.items():
            if key not in english:
                raise PluginCatalogError(
                    f"locales/{language}.json:{key}: key is not defined in locales/en.json"
                )
            signature = _placeholder_signature(
                value, f"locales/{language}.json:{key}"
            )
            if signature != english_placeholders[key]:
                raise PluginCatalogError(
                    f"locales/{language}.json:{key}: placeholders do not match locales/en.json"
                )

    return PluginCatalogBundle(owner_id=owner_id, catalogs=catalogs)


def validate_plugin_catalogs(plugin_dir: str | Path, project_name: str) -> list[str]:
    """Return actionable catalog errors for the local plugin checker."""
    try:
        bundle = read_plugin_catalogs(plugin_dir, project_name)
        if bundle.catalogs:
            conflicts = owner_collision_errors(
                [project_name, *_neighbor_plugin_names(Path(plugin_dir).resolve())]
            )
            if project_name in conflicts:
                raise PluginCatalogError(conflicts[project_name])
    except (OSError, PluginCatalogError) as exc:
        return [f"plugin localization: {exc}"]
    return []


def _valid_language_code(language: str) -> bool:
    return len(language) <= 35 and _LANGUAGE_RE.fullmatch(language) is not None


def _read_catalog(path: Path) -> dict[str, str]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_CATALOG_BYTES + 1)
    except OSError as exc:
        raise PluginCatalogError(f"locales/{path.name}: cannot be read: {exc}") from exc

    if len(raw) > MAX_CATALOG_BYTES:
        raise PluginCatalogError(f"locales/{path.name}: exceeds the 1 MiB size limit")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PluginCatalogError(f"locales/{path.name}: UTF-8 BOM is not allowed")

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PluginCatalogError(f"locales/{path.name}: invalid UTF-8") from exc

    try:
        document = json.loads(
            text,
            object_pairs_hook=lambda pairs: _object_without_duplicate_keys(path, pairs),
        )
    except json.JSONDecodeError as exc:
        raise PluginCatalogError(
            f"locales/{path.name}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    except RecursionError as exc:
        raise PluginCatalogError(f"locales/{path.name}: JSON nesting is too deep") from exc

    if "\ufffd" in text:
        raise PluginCatalogError(
            f"locales/{path.name}: Unicode replacement characters are not allowed"
        )

    if not isinstance(document, dict):
        raise PluginCatalogError(f"locales/{path.name}: root must be a JSON object")

    entries: dict[str, str] = {}
    _flatten_catalog(path, document, "", entries, 0)
    if not entries:
        raise PluginCatalogError(f"locales/{path.name}: catalog must not be empty")
    if len(entries) > MAX_CATALOG_ENTRIES:
        raise PluginCatalogError(
            f"locales/{path.name}: exceeds the {MAX_CATALOG_ENTRIES}-entry limit"
        )
    return entries


def _object_without_duplicate_keys(path: Path, pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PluginCatalogError(f"locales/{path.name}: duplicate JSON key {key!r}")
        result[key] = value
    return result


def _flatten_catalog(
    path: Path,
    value: dict,
    prefix: str,
    entries: dict[str, str],
    depth: int,
) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise PluginCatalogError(f"locales/{path.name}: JSON nesting is too deep")
    for raw_key, child in value.items():
        key = raw_key if not prefix else f"{prefix}.{raw_key}"
        if (
            len(key) > MAX_KEY_LENGTH
            or key.startswith("plugins.")
            or _RELATIVE_KEY_RE.fullmatch(key) is None
        ):
            raise PluginCatalogError(
                f"locales/{path.name}:{key!r}: invalid relative localization key"
            )

        if isinstance(child, dict):
            _flatten_catalog(path, child, key, entries, depth + 1)
            continue
        if not isinstance(child, str):
            raise PluginCatalogError(
                f"locales/{path.name}:{key}: value must be a string or nested object"
            )
        if not child.strip():
            raise PluginCatalogError(f"locales/{path.name}:{key}: value must not be blank")
        if "\0" in child:
            raise PluginCatalogError(f"locales/{path.name}:{key}: NUL characters are not allowed")
        if "\ufffd" in child:
            raise PluginCatalogError(
                f"locales/{path.name}:{key}: Unicode replacement characters are not allowed"
            )
        try:
            encoded = child.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise PluginCatalogError(
                f"locales/{path.name}:{key}: value contains an invalid Unicode surrogate"
            ) from exc
        if len(encoded) > MAX_VALUE_BYTES:
            raise PluginCatalogError(
                f"locales/{path.name}:{key}: value exceeds the 16 KiB limit"
            )
        if key in entries:
            raise PluginCatalogError(
                f"locales/{path.name}:{key}: flattened localization key is duplicated"
            )
        entries[key] = child


def _placeholder_signature(value: str, location: str) -> Counter:
    fields = Counter()
    numbering = set()

    def parse(text: str, depth: int = 0) -> None:
        for _, field_name, format_spec, conversion in string.Formatter().parse(text):
            if field_name is None:
                continue
            if depth > 1:
                raise ValueError("format placeholders are nested too deeply")
            argument = re.split(r"[.\[]", field_name, maxsplit=1)[0]
            if not argument:
                numbering.add("automatic")
            elif argument.isdecimal():
                numbering.add("manual")
            if len(numbering) > 1:
                raise ValueError("cannot mix automatic and manual argument numbering")
            if conversion not in (None, "s", "r", "a"):
                raise ValueError("unsupported format conversion")
            fields[(field_name, conversion or "", format_spec or "")] += 1
            if format_spec:
                parse(format_spec, depth + 1)

    try:
        parse(value)
        return fields
    except ValueError as exc:
        raise PluginCatalogError(f"{location}: malformed format placeholder: {exc}") from exc

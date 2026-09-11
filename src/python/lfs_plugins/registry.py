# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin registry client."""

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .environment import value as environment_value

from .compat import (
    LICHTFELD_VERSION,
    PLUGIN_API_VERSION,
    SUPPORTED_PLUGIN_FEATURES,
    compatibility_errors,
    validate_manifest_compatibility_fields,
)
from .errors import PluginNotFoundError, RegistryOfflineError, VersionNotFoundError
from .http import urlopen

try:
    from packaging.version import Version
except ImportError:
    Version = None

_log = logging.getLogger(__name__)

DEFAULT_REGISTRY_URLS = (
    "https://lichtfeld.io/plugin-registry",
    "https://lichtfeld.github.io/plugin-registry",
)
CACHE_TTL_HOURS = 1
HTTP_TIMEOUT_SEC = 10


@dataclass(frozen=True)
class RegistryPluginInfo:
    """Plugin metadata from registry."""

    name: str
    namespace: str
    display_name: str
    description: str
    author: str
    latest_version: str
    keywords: Tuple[str, ...] = field(default_factory=tuple)
    downloads: int = 0
    repository: Optional[str] = None

    @property
    def full_id(self) -> str:
        return f"{self.namespace}:{self.name}"


@dataclass(frozen=True)
class RegistryVersionInfo:
    """Version-specific metadata."""

    version: str
    plugin_api: str
    lichtfeld_version: str
    required_features: Tuple[str, ...] = field(default_factory=tuple)
    dependencies: Tuple[str, ...] = field(default_factory=tuple)
    checksum: str = ""
    download_url: str = ""
    git_ref: Optional[str] = None


class RegistryClient:
    """Fetches and caches registry data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        resolved_cache = os.environ.get("LFS_RESOLVED_CACHE_DIR")
        self._cache_dir = cache_dir or (
            Path(resolved_cache) / "registry"
            if resolved_cache
            else Path.home() / ".lichtfeld" / "cache" / "registry"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: Optional[Dict] = None
        self._index_loaded_at: Optional[float] = None
        override = environment_value("LFS_PLUGIN_REGISTRY_URL")
        self._registry_urls: Tuple[str, ...] = ((override,) if override else DEFAULT_REGISTRY_URLS)

    def search(
        self,
        query: str,
        compatible_only: bool = True,
        lichtfeld_version: str = LICHTFELD_VERSION,
        plugin_api: str = PLUGIN_API_VERSION,
        supported_features: Tuple[str, ...] = SUPPORTED_PLUGIN_FEATURES,
    ) -> List[RegistryPluginInfo]:
        """Search plugins by name, description, or keywords."""
        index = self._get_index()
        query_lower = query.lower()
        results = []

        for index_position, entry in enumerate(index.get("plugins", [])):
            try:
                if not isinstance(entry, dict):
                    raise ValueError("entry is not an object")
                name = entry.get("name")
                latest_version = entry.get("latest_version")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("missing name")
                if not isinstance(latest_version, str) or not latest_version.strip():
                    raise ValueError("missing latest_version")
                raw_keywords = entry.get("keywords", [])
                if not isinstance(raw_keywords, (list, tuple)) or not all(
                    isinstance(keyword, str) for keyword in raw_keywords
                ):
                    raise ValueError("keywords must be a list of strings")

                searchable = (
                    f"{name} {entry.get('summary', '')} {' '.join(raw_keywords)}"
                ).lower()
                if query_lower not in searchable:
                    continue

                info = RegistryPluginInfo(
                    name=name,
                    namespace=entry.get("namespace", "community"),
                    display_name=entry.get("display_name", name),
                    description=entry.get("summary", ""),
                    author=entry.get("author", ""),
                    latest_version=latest_version,
                    keywords=tuple(raw_keywords),
                    downloads=entry.get("downloads", 0),
                    repository=entry.get("repository"),
                )
                if compatible_only:
                    try:
                        self.resolve_version(
                            info.full_id,
                            None,
                            lichtfeld_version,
                            plugin_api=plugin_api,
                            supported_features=supported_features,
                        )
                    except PluginNotFoundError:
                        pass
                    except VersionNotFoundError:
                        continue
                results.append(info)
            except Exception as exc:
                _log.warning("Skipping malformed registry entry %d: %s", index_position, exc)
        return results

    def get_plugin(self, plugin_id: str) -> Dict:
        """Get detailed plugin info from registry or cache."""
        namespace, name = self._parse_id(plugin_id)
        cache_path = self._plugin_cache_path(namespace, name)

        if cache_path.exists() and self._cache_is_fresh(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                _log.debug("Ignoring invalid registry cache '%s': %s", cache_path, exc)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for url in self._plugin_detail_urls(namespace, name):
            try:
                data = self._fetch_json(url)
                self._atomic_write_json(cache_path, data)
                return data
            except Exception as exc:
                last_error = exc

        # A stale but valid detail record is still useful when the registry is
        # offline or GitHub/registry infrastructure is rate-limiting requests.
        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    stale = json.load(f)
                _log.warning("Using stale registry detail cache for '%s': %s", plugin_id, last_error)
                return stale
            except Exception:
                pass
        raise PluginNotFoundError(f"Plugin '{plugin_id}' not found: {last_error}") from last_error

    def resolve_version(
        self,
        plugin_id: str,
        requested_version: Optional[str],
        lichtfeld_version: str,
        *,
        plugin_api: str = PLUGIN_API_VERSION,
        supported_features: Tuple[str, ...] = SUPPORTED_PLUGIN_FEATURES,
    ) -> RegistryVersionInfo:
        """Resolve the best matching version for the current host contract."""
        plugin = self.get_plugin(plugin_id)
        versions = plugin.get("versions", {})

        if not versions:
            raise VersionNotFoundError(f"No versions found for {plugin_id}")

        if requested_version:
            if requested_version not in versions:
                raise VersionNotFoundError(f"Version {requested_version} not found for {plugin_id}")
            v = versions[requested_version]
            issues = self._get_compatibility_issues(
                v,
                plugin_api=plugin_api,
                lichtfeld_version=lichtfeld_version,
                supported_features=supported_features,
            )
            if issues:
                raise VersionNotFoundError(
                    f"Version {requested_version} of {plugin_id} is incompatible: {'; '.join(issues)}"
                )
        elif Version is None:
            compatible = [
                (ver, info)
                for ver, info in versions.items()
                if not self._get_compatibility_issues(
                    info,
                    plugin_api=plugin_api,
                    lichtfeld_version=lichtfeld_version,
                    supported_features=supported_features,
                )
            ]
            if not compatible:
                raise VersionNotFoundError(
                    f"No compatible version for plugin API {plugin_api} and LichtFeld {lichtfeld_version}"
                )
            compatible.sort(key=lambda item: item[0], reverse=True)
            selected_version, v = compatible[0]
        else:
            compatible = [
                (Version(ver), info)
                for ver, info in versions.items()
                if not self._get_compatibility_issues(
                    info,
                    plugin_api=plugin_api,
                    lichtfeld_version=lichtfeld_version,
                    supported_features=supported_features,
                )
            ]
            if not compatible:
                raise VersionNotFoundError(
                    f"No compatible version for plugin API {plugin_api} and LichtFeld {lichtfeld_version}"
                )
            compatible.sort(key=lambda x: x[0], reverse=True)
            selected_version, v = compatible[0]

        return RegistryVersionInfo(
            version=v.get("version", requested_version or str(selected_version)),
            plugin_api=v["plugin_api"],
            lichtfeld_version=v["lichtfeld_version"],
            required_features=tuple(v.get("required_features", [])),
            dependencies=tuple(v.get("dependencies", [])),
            checksum=v.get("checksum", ""),
            download_url=v.get("download_url", ""),
            git_ref=v.get("git_ref"),
        )

    def _get_compatibility_issues(
        self,
        version_info: Dict,
        *,
        plugin_api: str,
        lichtfeld_version: str,
        supported_features: Tuple[str, ...],
    ) -> List[str]:
        metadata_errors = validate_manifest_compatibility_fields(version_info)
        if metadata_errors:
            return [error.removeprefix("pyproject.toml: ") for error in metadata_errors]

        return compatibility_errors(
            version_info["plugin_api"],
            version_info["lichtfeld_version"],
            version_info["required_features"],
            current_plugin_api=plugin_api,
            current_lichtfeld_version=lichtfeld_version,
            supported_features=supported_features,
        )

    def verify_checksum(self, path: Path, expected: str) -> bool:
        """Verify SHA-256 checksum of a file."""
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        return actual == expected.removeprefix("sha256:")

    def _get_index(self) -> Dict:
        """Get index from cache or fetch from registry."""
        if self._index is not None and self._index_loaded_at is not None:
            if time.monotonic() - self._index_loaded_at < CACHE_TTL_HOURS * 3600:
                return self._index
            self._index = None
            self._index_loaded_at = None

        cache_path = self._cache_dir / "index.json"
        timestamp_path = self._cache_dir / "last_update"
        cache_ttl = timedelta(hours=CACHE_TTL_HOURS)

        if cache_path.exists() and timestamp_path.exists():
            if datetime.now() - datetime.fromtimestamp(timestamp_path.stat().st_mtime) < cache_ttl:
                try:
                    with open(cache_path, encoding="utf-8") as f:
                        self._index = json.load(f)
                    self._index_loaded_at = time.monotonic()
                    return self._index
                except Exception as exc:
                    _log.warning("Ignoring invalid registry index cache '%s': %s", cache_path, exc)

        try:
            self._index = self._fetch_json_with_fallback(
                [f"{base_url}/index.json" for base_url in self._registry_urls]
            )
            self._atomic_write_json(cache_path, self._index)
            timestamp_path.touch()
            self._index_loaded_at = time.monotonic()
            return self._index
        except Exception:
            if cache_path.exists():
                _log.debug("Registry offline, using cached index")
                with open(cache_path, encoding="utf-8") as f:
                    self._index = json.load(f)
                    self._index_loaded_at = time.monotonic()
                    return self._index
            raise RegistryOfflineError("Cannot reach registry and no cache available")

    def _fetch_json(self, url: str) -> Dict:
        """Fetch JSON from URL."""
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "LichtFeld-PluginManager/1.0"})
        with urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _fetch_json_with_fallback(self, urls: List[str]) -> Dict:
        last_error: Exception | None = None
        for url in urls:
            try:
                return self._fetch_json(url)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    @staticmethod
    def _cache_is_fresh(path: Path) -> bool:
        try:
            return time.time() - path.stat().st_mtime < CACHE_TTL_HOURS * 3600
        except OSError:
            return False

    @staticmethod
    def _atomic_write_json(path: Path, data: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            fd, temporary_path = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    def _plugin_cache_path(self, namespace: str, name: str) -> Path:
        safe_namespace = self._safe_cache_component(namespace)
        safe_name = self._safe_cache_component(name)
        return self._cache_dir / "plugins" / safe_namespace / f"{safe_name}.json"

    def _plugin_detail_urls(self, namespace: str, name: str) -> Tuple[str, ...]:
        from urllib.parse import quote

        namespace_q = quote(namespace, safe="")
        name_q = quote(name, safe="")
        full_id_q = quote(f"{namespace}:{name}", safe="")
        urls = []
        for base_url in self._registry_urls:
            urls.extend([
                f"{base_url}/plugins/{namespace_q}/{name_q}.json",
                f"{base_url}/plugins/{full_id_q}.json",
                f"{base_url}/plugins/{name_q}.json",
            ])
        deduped = []
        for url in urls:
            if url not in deduped:
                deduped.append(url)
        return tuple(deduped)

    def _safe_cache_component(self, value: str) -> str:
        return value.replace("/", "_").replace("\\", "_").replace(":", "_")

    def _parse_id(self, plugin_id: str) -> Tuple[str, str]:
        """Parse 'namespace:name' into (namespace, name). Defaults to 'lichtfeld' namespace."""
        if ":" in plugin_id:
            namespace, name = plugin_id.split(":", 1)
            return namespace, name
        return "lichtfeld", plugin_id

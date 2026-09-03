# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin marketplace catalog backed by the plugin registry."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

from .http import urlopen

_log = logging.getLogger(__name__)

try:
    import lichtfeld as _lf

    class _LfLogHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                if record.levelno >= logging.ERROR:
                    _lf.log.error(msg)
                elif record.levelno >= logging.WARNING:
                    _lf.log.warn(msg)
                else:
                    _lf.log.info(msg)
            except Exception:
                # Logging must never turn a per-entry network failure into a
                # catalog-wide failure (and keeps lightweight test stubs safe).
                pass

    _log.addHandler(_LfLogHandler())
    _log.setLevel(logging.DEBUG)
except Exception:
    pass

GITHUB_TIMEOUT_SEC = 4
GITHUB_MAX_WORKERS = 4
CATALOG_CACHE_TTL_SEC = 300
REFRESH_RETRY_COOLDOWN_SEC = 30
REFRESH_RETRY_MAX_COOLDOWN_SEC = 300
GITHUB_API_URL = "https://api.github.com/repos"

CURATED_PLUGIN_URLS: Tuple[str, ...] = (
    "https://github.com/shadygm/Lichtfeld-Densification-Plugin",
    "https://github.com/shadygm/Lichtfeld-ml-sharp-Plugin",
    "https://github.com/jacobvanbeets/360_record",
    "https://github.com/jacobvanbeets/lichtfeld-depthmap-plugin",
    "https://github.com/jacobvanbeets/splat-vr-viewer",
    "https://github.com/jacobvanbeets/lichtfeld-measurement-plugin",
)


@dataclass(frozen=True)
class MarketplacePluginEntry:
    """Resolved metadata for a marketplace plugin entry."""

    source_url: str
    github_url: str
    owner: str
    repo: str
    name: str
    description: str
    stars: int = 0
    downloads: int = 0
    language: str = ""
    topics: Tuple[str, ...] = ()
    registry_id: str = ""
    error: str = ""


@dataclass(frozen=True)
class _CatalogCache:
    entries: Tuple[MarketplacePluginEntry, ...]
    registry_loaded: bool
    github_enriched: bool
    stored_at: float
    next_retry_at: float = 0.0
    failure_count: int = 0


_catalog_cache_lock = threading.Lock()
_catalog_cache: Optional[_CatalogCache] = None
# Refresh ownership is deliberately per catalog instance.  A global in-flight
# flag used to make a second panel return without ever receiving the first
# panel's result.


class PluginMarketplaceCatalog:
    """Dual-source catalog: registry entries merged with curated URL list."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: List[MarketplacePluginEntry] = _build_curated_fallback()
        self._loading = False
        self._registry_loaded = False
        self._github_enriched = False
        self._on_change: Optional[Callable[[], None]] = None
        self._restore_cached_catalog(time.monotonic())

    def set_on_change(self, callback: Optional[Callable[[], None]]) -> None:
        with self._lock:
            self._on_change = callback

    def _notify_change(self) -> None:
        with self._lock:
            callback = self._on_change
        if callback is None:
            return
        try:
            callback()
        except Exception:
            _log.debug("Plugin marketplace change callback failed", exc_info=True)

    def _restore_cached_catalog(self, now: float) -> None:
        with _catalog_cache_lock:
            cache = _catalog_cache
            if cache is None:
                return
            if cache.registry_loaded:
                usable = now - cache.stored_at < CATALOG_CACHE_TTL_SEC
            else:
                usable = now < cache.next_retry_at
            if not usable:
                return

        with self._lock:
            self._entries = list(cache.entries)
            self._registry_loaded = cache.registry_loaded
            self._github_enriched = cache.github_enriched

    @staticmethod
    def _cache_can_serve(
        cache: Optional[_CatalogCache],
        now: float,
        require_github_enrichment: bool,
    ) -> bool:
        if cache is None:
            return False
        if now < cache.next_retry_at:
            return True
        if not cache.registry_loaded or now - cache.stored_at >= CATALOG_CACHE_TTL_SEC:
            return False
        return not require_github_enrichment or cache.github_enriched

    def refresh_async(self, force: bool = False, require_github_enrichment: bool = False) -> None:
        """Fetch registry entries, optionally enriching curated entries with GitHub metadata."""
        now = time.monotonic()
        with _catalog_cache_lock:
            cache = _catalog_cache
            cache_can_serve = self._cache_can_serve(
                cache, now, require_github_enrichment
            )
        with self._lock:
            if self._loading:
                return
            needs_github_upgrade = (
                require_github_enrichment
                and not self._github_enriched
                and not (cache and cache.github_enriched)
            )
            if not force and cache_can_serve and not needs_github_upgrade:
                return
            if not force and cache is not None and now < cache.next_retry_at:
                return
            self._loading = True
        self._notify_change()

        def worker():
            global _catalog_cache

            registry_entries: List[MarketplacePluginEntry] = []
            registry_ok = False
            registry_error = None
            github_enrichment_succeeded = False
            backoff = None
            try:
                try:
                    from .manager import PluginManager

                    mgr = PluginManager.instance()
                    for info in mgr.search(""):
                        registry_entries.append(_from_registry(info))
                    registry_ok = True
                except Exception as exc:
                    registry_error = exc

                try:
                    curated_entries = (
                        _resolve_curated_from_github()
                        if require_github_enrichment
                        else _build_curated_fallback()
                    )
                    github_enrichment_succeeded = require_github_enrichment
                except Exception as exc:
                    curated_entries = _build_curated_fallback()
                    registry_error = registry_error or exc
                merged = _merge_entries(registry_entries, curated_entries)
                if registry_ok:
                    _log.info(
                        "Plugin marketplace registry loaded: %d registry entries, %d total catalog entries",
                        len(registry_entries),
                        len(merged),
                    )
                with self._lock:
                    self._entries = merged
                    self._registry_loaded = registry_ok
                    self._github_enriched = (
                        self._github_enriched or github_enrichment_succeeded
                    )

                with _catalog_cache_lock:
                    previous_failures = (
                        _catalog_cache.failure_count
                        if _catalog_cache is not None and not registry_ok
                        else 0
                    )
                    failure_count = previous_failures + 1 if not registry_ok else 0
                    if failure_count:
                        backoff = min(
                            REFRESH_RETRY_COOLDOWN_SEC * 2 ** (failure_count - 1),
                            REFRESH_RETRY_MAX_COOLDOWN_SEC,
                        )
                        next_retry_at = time.monotonic() + backoff
                    else:
                        next_retry_at = 0.0
                    _catalog_cache = _CatalogCache(
                        entries=tuple(merged),
                        registry_loaded=registry_ok,
                        github_enriched=self._github_enriched,
                        stored_at=time.monotonic(),
                        next_retry_at=next_retry_at,
                        failure_count=failure_count,
                    )
                if registry_error is not None:
                    if backoff is not None:
                        _log.warning(
                            "Plugin marketplace refresh failed; retrying in %.0f s: %s",
                            backoff,
                            registry_error,
                        )
                    else:
                        _log.warning(
                            "Plugin marketplace GitHub enrichment failed: %s",
                            registry_error,
                        )
            finally:
                with self._lock:
                    self._loading = False
                self._notify_change()

        threading.Thread(target=worker, daemon=True).start()

    def snapshot(self) -> Tuple[List[MarketplacePluginEntry], bool, bool]:
        """Return (entries, is_loading, registry_loaded)."""
        with self._lock:
            return list(self._entries), self._loading, self._registry_loaded


def _entry_key(owner: str, repo: str) -> str:
    if not owner or not repo:
        return ""
    return f"{owner}/{repo}".lower()


def _from_registry(info) -> MarketplacePluginEntry:
    owner, repo = "", ""
    github_url = ""
    if info.repository:
        try:
            from .installer import parse_github_url

            owner, repo, _ = parse_github_url(info.repository)
            github_url = f"https://github.com/{owner}/{repo}"
        except Exception:
            pass
    return MarketplacePluginEntry(
        source_url=info.repository or "",
        github_url=github_url,
        owner=owner,
        repo=repo,
        name=info.display_name or info.name,
        description=info.description,
        downloads=info.downloads,
        topics=info.keywords,
        registry_id=info.full_id,
    )


def _build_curated_fallback() -> List[MarketplacePluginEntry]:
    """Instant fallback entries from curated URLs (no network)."""
    from .installer import parse_github_url

    entries: List[MarketplacePluginEntry] = []
    for url in CURATED_PLUGIN_URLS:
        try:
            owner, repo, _ = parse_github_url(url)
            entries.append(MarketplacePluginEntry(
                source_url=url,
                github_url=f"https://github.com/{owner}/{repo}",
                owner=owner,
                repo=repo,
                name=repo,
                description="",
            ))
        except Exception as exc:
            _log.debug("Skipping invalid curated URL '%s': %s", url, exc)
    return entries


def _resolve_curated_from_github() -> List[MarketplacePluginEntry]:
    """Resolve curated URLs via GitHub API (runs in background thread)."""
    from .installer import parse_github_url

    requests = []
    for url in CURATED_PLUGIN_URLS:
        try:
            owner, repo, _ = parse_github_url(url)
        except Exception:
            continue
        requests.append((url, owner, repo))

    if not requests:
        return []

    # Keep the catalog responsive when GitHub is slow or rate-limiting.  Each
    # task returns a fallback-shaped entry even when its request fails, so one
    # bad repository cannot discard the rest of the catalog.
    with ThreadPoolExecutor(max_workers=min(GITHUB_MAX_WORKERS, len(requests))) as pool:
        futures = [pool.submit(_resolve_github_entry, *request) for request in requests]
        entries = []
        for request, future in zip(requests, futures):
            try:
                entries.append(future.result())
            except Exception as exc:
                url, owner, repo = request
                _log.warning("GitHub metadata lookup failed for %s/%s: %s", owner, repo, exc)
                entries.append(_fallback_github_entry(url, owner, repo))
        return entries


def _fallback_github_entry(source_url: str, owner: str, repo: str) -> MarketplacePluginEntry:
    return MarketplacePluginEntry(
        source_url=source_url,
        github_url=f"https://github.com/{owner}/{repo}",
        owner=owner,
        repo=repo,
        name=repo,
        description="",
    )


def _resolve_github_entry(source_url: str, owner: str, repo: str) -> MarketplacePluginEntry:
    github_url = f"https://github.com/{owner}/{repo}"
    name = repo
    description = ""
    stars = 0
    language = ""
    topics: Tuple[str, ...] = ()

    try:
        data = _fetch_repo_metadata(owner, repo)
        api_name = str(data.get("name", "")).strip()
        api_description = data.get("description")
        api_stars = data.get("stargazers_count", 0)
        api_language = data.get("language")
        api_topics = data.get("topics")
        name = api_name or name
        description = api_description.strip() if isinstance(api_description, str) else ""
        stars = int(api_stars) if isinstance(api_stars, (int, float)) else 0
        github_url = str(data.get("html_url") or github_url)
        language = api_language.strip() if isinstance(api_language, str) else ""
        topics = (
            tuple(t.strip() for t in api_topics if isinstance(t, str) and t.strip())
            if isinstance(api_topics, list)
            else ()
        )
    except Exception as exc:
        _log.debug("GitHub metadata lookup failed for %s/%s: %s", owner, repo, exc)

    return MarketplacePluginEntry(
        source_url=source_url,
        github_url=github_url,
        owner=owner,
        repo=repo,
        name=name,
        description=description,
        stars=stars,
        language=language,
        topics=topics,
    )


def _fetch_repo_metadata(owner: str, repo: str) -> dict:
    import urllib.request

    url = f"{GITHUB_API_URL}/{owner}/{repo}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "LichtFeld-PluginMarketplace/1.0",
        },
    )
    with urlopen(req, timeout=GITHUB_TIMEOUT_SEC) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _unique_key(entry: MarketplacePluginEntry) -> str:
    key = _entry_key(entry.owner, entry.repo)
    if key:
        return key
    return entry.registry_id or entry.source_url or ""


def _merge_entries(
    registry: List[MarketplacePluginEntry],
    curated: List[MarketplacePluginEntry],
) -> List[MarketplacePluginEntry]:
    """Registry entries take priority; curated entries fill gaps."""
    seen: Set[str] = set()
    merged: List[MarketplacePluginEntry] = []

    for entry in registry:
        key = _unique_key(entry)
        if key and key not in seen:
            seen.add(key)
            merged.append(entry)

    for entry in curated:
        key = _unique_key(entry)
        if key and key not in seen:
            seen.add(key)
            merged.append(entry)

    return merged

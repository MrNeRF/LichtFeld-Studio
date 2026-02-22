# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin marketplace catalog backed by the plugin registry."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

_log = logging.getLogger(__name__)

GITHUB_TIMEOUT_SEC = 10
REFRESH_RETRY_COOLDOWN_SEC = 30
GITHUB_API_URL = "https://api.github.com/repos"

CURATED_PLUGIN_URLS: Tuple[str, ...] = (
    "https://github.com/shadygm/Lichtfeld-Densification-Plugin",
    "https://github.com/shadygm/Lichtfeld-ml-sharp-Plugin",
    "https://github.com/jacobvanbeets/360_record",
    "https://github.com/jacobvanbeets/lichtfeld-depthmap-plugin",
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


class PluginMarketplaceCatalog:
    """Dual-source catalog: registry entries merged with curated URL list."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: List[MarketplacePluginEntry] = _build_curated_fallback()
        self._loading = False
        self._loaded_once = False
        self._last_attempt: float = 0.0

    def refresh_async(self, force: bool = False) -> None:
        """Fetch registry entries in a background thread and merge with curated list."""
        with self._lock:
            if self._loading:
                return
            if self._loaded_once and not force:
                return
            now = time.monotonic()
            if not force and self._last_attempt > 0 and (now - self._last_attempt) < REFRESH_RETRY_COOLDOWN_SEC:
                return
            self._loading = True
            self._last_attempt = now

        def worker():
            from .manager import PluginManager

            mgr = PluginManager.instance()
            registry_entries: List[MarketplacePluginEntry] = []
            registry_ok = False
            try:
                for info in mgr.search(""):
                    registry_entries.append(_from_registry(info))
                registry_ok = True
            except Exception as exc:
                _log.debug("Registry search failed: %s", exc)

            curated_resolved = _resolve_curated_from_github()
            merged = _merge_entries(registry_entries, curated_resolved)
            with self._lock:
                self._entries = merged
                self._loading = False
                self._loaded_once = registry_ok

        threading.Thread(target=worker, daemon=True).start()

    def snapshot(self) -> Tuple[List[MarketplacePluginEntry], bool]:
        """Return (entries, is_loading)."""
        with self._lock:
            return list(self._entries), self._loading


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

    entries: List[MarketplacePluginEntry] = []
    for url in CURATED_PLUGIN_URLS:
        try:
            owner, repo, _ = parse_github_url(url)
        except Exception:
            continue
        entries.append(_resolve_github_entry(url, owner, repo))
    return entries


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
    url = f"{GITHUB_API_URL}/{owner}/{repo}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "LichtFeld-PluginMarketplace/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=GITHUB_TIMEOUT_SEC) as resp:
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

# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin marketplace catalog and GitHub metadata resolution."""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .installer import parse_github_url

_log = logging.getLogger(__name__)

GITHUB_TIMEOUT_SEC = 10
GITHUB_API_URL = "https://api.github.com/repos"

# User-editable list of marketplace plugins. Add/remove GitHub links here.
PLUGIN_MARKETPLACE_URLS: List[str] = [
    "https://github.com/shadygm/Lichtfeld-Densification-Plugin",
    "https://github.com/shadygm/Lichtfeld-ml-sharp-Plugin",
    "https://github.com/jacobvanbeets/360_record",
    "https://github.com/jacobvanbeets/lichtfeld-depthmap-plugin"
]


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
    language: str = ""
    topics: Tuple[str, ...] = ()
    error: str = ""


def set_plugin_marketplace_urls(urls: Sequence[str]) -> None:
    """Replace configured marketplace plugin URLs."""
    PLUGIN_MARKETPLACE_URLS.clear()
    PLUGIN_MARKETPLACE_URLS.extend(_clean_urls(urls))


def get_plugin_marketplace_urls() -> List[str]:
    """Get a copy of configured marketplace plugin URLs."""
    return list(PLUGIN_MARKETPLACE_URLS)


class PluginMarketplaceCatalog:
    """Asynchronous resolver for marketplace entries."""

    def __init__(self, urls: Sequence[str] | None = None):
        self._lock = threading.Lock()
        self._urls: List[str] = _clean_urls(urls if urls is not None else PLUGIN_MARKETPLACE_URLS)
        self._entries: List[MarketplacePluginEntry] = self._build_fallback_entries(self._urls)
        self._loading = False
        self._loaded_once = False

    def set_urls(self, urls: Sequence[str]) -> None:
        """Update URLs and reset entries."""
        cleaned = _clean_urls(urls)
        with self._lock:
            self._urls = cleaned
            self._entries = self._build_fallback_entries(cleaned)
            self._loaded_once = False

    def refresh_async(self, force: bool = False) -> None:
        """Resolve metadata from GitHub API in a background thread."""
        with self._lock:
            if self._loading:
                return
            if self._loaded_once and not force:
                return
            urls = list(self._urls)
            self._loading = True

        def worker():
            entries = self._resolve_entries(urls)
            with self._lock:
                self._entries = entries
                self._loading = False
                self._loaded_once = True

        threading.Thread(target=worker, daemon=True).start()

    def snapshot(self) -> Tuple[List[MarketplacePluginEntry], bool]:
        """Return (entries, is_loading)."""
        with self._lock:
            return list(self._entries), self._loading

    def _resolve_entries(self, urls: Sequence[str]) -> List[MarketplacePluginEntry]:
        resolved: List[MarketplacePluginEntry] = []
        for source_url in urls:
            resolved.append(self._resolve_one(source_url))
        return resolved

    def _resolve_one(self, source_url: str) -> MarketplacePluginEntry:
        source_url = source_url.strip()
        try:
            owner, repo, _ = parse_github_url(source_url)
        except Exception as exc:
            _log.warning("Invalid marketplace URL '%s': %s", source_url, exc)
            return MarketplacePluginEntry(
                source_url=source_url,
                github_url=source_url,
                owner="",
                repo=source_url,
                name=source_url,
                description="Invalid GitHub URL",
                stars=0,
                error=str(exc),
            )

        github_url = f"https://github.com/{owner}/{repo}"
        name = repo
        description = ""
        stars = 0

        try:
            data = self._fetch_repo_metadata(owner, repo)
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
            language = ""
            topics = ()

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

    def _fetch_repo_metadata(self, owner: str, repo: str) -> dict:
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

    def _build_fallback_entries(self, urls: Sequence[str]) -> List[MarketplacePluginEntry]:
        return [self._fallback_entry(url) for url in urls]

    def _fallback_entry(self, source_url: str) -> MarketplacePluginEntry:
        source_url = source_url.strip()
        try:
            owner, repo, _ = parse_github_url(source_url)
            return MarketplacePluginEntry(
                source_url=source_url,
                github_url=f"https://github.com/{owner}/{repo}",
                owner=owner,
                repo=repo,
                name=repo,
                description="",
                stars=0,
                language="",
                topics=(),
            )
        except Exception as exc:
            return MarketplacePluginEntry(
                source_url=source_url,
                github_url=source_url,
                owner="",
                repo=source_url,
                name=source_url,
                description="Invalid GitHub URL",
                stars=0,
                error=str(exc),
            )


def _clean_urls(urls: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for url in urls:
        value = str(url).strip()
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
    return cleaned

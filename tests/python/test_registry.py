# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the plugin registry system."""

import json
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def registry_cache_dir():
    """Create temporary registry cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_registry_index():
    """Sample registry index data."""
    return {
        "version": 1,
        "plugins": [
            {
                "name": "colmap",
                "namespace": "lichtfeld",
                "display_name": "COLMAP Integration",
                "summary": "Structure-from-Motion reconstruction",
                "author": "LichtFeld Team",
                "latest_version": "1.2.0",
                "keywords": ["sfm", "reconstruction", "poses"],
                "downloads": 1500,
                "repository": "https://github.com/lichtfeld/lichtfeld-plugin-colmap",
            },
            {
                "name": "sam-segmentation",
                "namespace": "community",
                "display_name": "SAM Segmentation",
                "summary": "Segment Anything Model integration",
                "author": "Community",
                "latest_version": "0.5.0",
                "keywords": ["segmentation", "ai", "masks"],
                "downloads": 500,
            },
        ],
    }


@pytest.fixture
def mock_plugin_detail():
    """Sample plugin detail data."""
    return {
        "name": "colmap",
        "namespace": "lichtfeld",
        "display_name": "COLMAP Integration",
        "description": "Full COLMAP integration for SfM",
        "author": "LichtFeld Team",
        "repository": "https://github.com/lichtfeld/lichtfeld-plugin-colmap",
        "versions": {
            "1.2.0": {
                "version": "1.2.0",
                "plugin_api": ">=1,<2",
                "lichtfeld_version": ">=0.4.2",
                "required_features": [],
                "dependencies": ["pycolmap>=0.6.0"],
                "checksum": "sha256:abc123",
                "download_url": "https://example.com/colmap-1.2.0.tar.gz",
                "git_ref": "v1.2.0",
            },
            "1.1.0": {
                "version": "1.1.0",
                "plugin_api": ">=1,<2",
                "lichtfeld_version": ">=0.9",
                "required_features": [],
                "dependencies": ["pycolmap>=0.5.0"],
                "checksum": "sha256:def456",
                "download_url": "https://example.com/colmap-1.1.0.tar.gz",
                "git_ref": "v1.1.0",
            },
        },
    }


class TestRegistryClient:
    """Tests for RegistryClient."""

    def test_search_by_name(self, registry_cache_dir, mock_registry_index):
        """Should find plugins by name."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        with patch.object(client, "_fetch_json", return_value=mock_registry_index):
            results = client.search("colmap", compatible_only=False)

        assert len(results) == 1
        assert results[0].name == "colmap"
        assert results[0].namespace == "lichtfeld"

    def test_search_by_keyword(self, registry_cache_dir, mock_registry_index):
        """Should find plugins by keyword."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        with patch.object(client, "_fetch_json", return_value=mock_registry_index):
            results = client.search("segmentation", compatible_only=False)

        assert len(results) == 1
        assert results[0].name == "sam-segmentation"

    def test_search_by_description(self, registry_cache_dir, mock_registry_index):
        """Should find plugins by description."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        with patch.object(client, "_fetch_json", return_value=mock_registry_index):
            results = client.search("reconstruction", compatible_only=False)

        assert len(results) == 1
        assert results[0].name == "colmap"

    def test_search_case_insensitive(self, registry_cache_dir, mock_registry_index):
        """Search should be case insensitive."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        with patch.object(client, "_fetch_json", return_value=mock_registry_index):
            results = client.search("COLMAP", compatible_only=False)

        assert len(results) == 1
        assert results[0].name == "colmap"

    def test_parse_plugin_id_with_namespace(self, registry_cache_dir):
        """Should parse namespace:name format."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        namespace, name = client._parse_id("community:sam-segmentation")
        assert namespace == "community"
        assert name == "sam-segmentation"

    def test_parse_plugin_id_without_namespace(self, registry_cache_dir):
        """Should default to lichtfeld namespace."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        namespace, name = client._parse_id("colmap")
        assert namespace == "lichtfeld"
        assert name == "colmap"

    def test_get_plugin_uses_namespaced_cache_keys(self, registry_cache_dir):
        """Plugins with the same name in different namespaces should not collide in cache."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)
        official_detail = {
            "name": "foo",
            "namespace": "official",
            "repository": "https://example.com/official/foo",
            "versions": {},
        }
        community_detail = {
            "name": "foo",
            "namespace": "community",
            "repository": "https://example.com/community/foo",
            "versions": {},
        }

        def _fetch(url):
            if url.endswith("/plugins/official/foo.json"):
                return official_detail
            if url.endswith("/plugins/community/foo.json"):
                return community_detail
            raise AssertionError(url)

        with patch.object(client, "_fetch_json", side_effect=_fetch):
            assert client.get_plugin("official:foo")["repository"] == official_detail["repository"]
            assert client.get_plugin("community:foo")["repository"] == community_detail["repository"]

        assert (registry_cache_dir / "plugins" / "official" / "foo.json").exists()
        assert (registry_cache_dir / "plugins" / "community" / "foo.json").exists()

        with patch.object(client, "_fetch_json", side_effect=AssertionError("cache should be used")):
            assert client.get_plugin("official:foo")["repository"] == official_detail["repository"]
            assert client.get_plugin("community:foo")["repository"] == community_detail["repository"]


class TestVersionResolution:
    """Tests for version resolution."""

    def test_resolve_specific_version(self, registry_cache_dir, mock_plugin_detail):
        """Should resolve specific requested version."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        with patch.object(client, "get_plugin", return_value=mock_plugin_detail):
            version_info = client.resolve_version("colmap", "1.1.0", "1.0.0")

        assert version_info.version == "1.1.0"
        assert version_info.git_ref == "v1.1.0"

    def test_resolve_latest_compatible(self, registry_cache_dir, mock_plugin_detail):
        """Should resolve latest compatible version when none specified."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        with patch.object(client, "get_plugin", return_value=mock_plugin_detail):
            version_info = client.resolve_version("colmap", None, "1.0.0")

        assert version_info.version == "1.2.0"
        assert version_info.plugin_api == ">=1,<2"
        assert version_info.lichtfeld_version == ">=0.4.2"

    def test_resolve_version_rejects_incompatible_plugin_api(self, registry_cache_dir, mock_plugin_detail):
        """Should reject registry versions that target a different plugin API major."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.errors import VersionNotFoundError
        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)
        incompatible_detail = json.loads(json.dumps(mock_plugin_detail))
        incompatible_detail["versions"]["1.2.0"]["plugin_api"] = ">=2,<3"
        incompatible_detail["versions"]["1.1.0"]["plugin_api"] = ">=2,<3"

        with patch.object(client, "get_plugin", return_value=incompatible_detail):
            with pytest.raises(VersionNotFoundError, match="plugin API 1.0"):
                client.resolve_version("colmap", None, "1.0.0")

    def test_search_compatible_only_filters_incompatible_plugins(self, registry_cache_dir, mock_registry_index, mock_plugin_detail):
        """Should exclude registry entries without a compatible version."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)
        incompatible_detail = json.loads(json.dumps(mock_plugin_detail))
        incompatible_detail["versions"]["1.2.0"]["required_features"] = ["future-ui.v2"]
        incompatible_detail["versions"]["1.1.0"]["required_features"] = ["future-ui.v2"]

        def _get_plugin(plugin_id):
            if plugin_id.endswith("colmap"):
                return mock_plugin_detail
            return incompatible_detail

        with patch.object(client, "_fetch_json", return_value=mock_registry_index):
            with patch.object(client, "get_plugin", side_effect=_get_plugin):
                results = client.search("")

        assert [result.name for result in results] == ["colmap"]

    def test_resolve_version_uses_namespaced_plugin_details(self, registry_cache_dir):
        """Version resolution should fetch the matching namespaced plugin detail."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)
        official_detail = {
            "name": "foo",
            "namespace": "official",
            "versions": {
                "2.0.0": {
                    "version": "2.0.0",
                    "plugin_api": ">=1,<2",
                    "lichtfeld_version": ">=0.4.2",
                    "required_features": [],
                    "checksum": "sha256:official",
                    "download_url": "https://example.com/official/foo-2.0.0.tar.gz",
                },
            },
        }
        community_detail = {
            "name": "foo",
            "namespace": "community",
            "versions": {
                "1.0.0": {
                    "version": "1.0.0",
                    "plugin_api": ">=1,<2",
                    "lichtfeld_version": ">=0.4.2",
                    "required_features": [],
                    "checksum": "sha256:community",
                    "download_url": "https://example.com/community/foo-1.0.0.tar.gz",
                },
            },
        }

        def _fetch(url):
            if url.endswith("/plugins/official/foo.json"):
                return official_detail
            if url.endswith("/plugins/community/foo.json"):
                return community_detail
            raise AssertionError(url)

        with patch.object(client, "_fetch_json", side_effect=_fetch):
            official = client.resolve_version("official:foo", None, "1.0.0")
            community = client.resolve_version("community:foo", None, "1.0.0")

        assert official.checksum == "sha256:official"
        assert community.checksum == "sha256:community"

    def test_resolve_version_not_found(self, registry_cache_dir, mock_plugin_detail):
        """Should raise VersionNotFoundError for unknown version."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient
        from lfs_plugins.errors import VersionNotFoundError

        client = RegistryClient(cache_dir=registry_cache_dir)

        with patch.object(client, "get_plugin", return_value=mock_plugin_detail):
            with pytest.raises(VersionNotFoundError):
                client.resolve_version("colmap", "9.9.9", "1.0.0")


class TestCaching:
    """Tests for registry caching."""

    def test_cache_index(self, registry_cache_dir, mock_registry_index):
        """Should cache index to disk."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        with patch.object(client, "_fetch_json", return_value=mock_registry_index):
            client.search("test")

        cache_file = registry_cache_dir / "index.json"
        assert cache_file.exists()

        with open(cache_file) as f:
            cached = json.load(f)
        assert cached["version"] == 1
        assert len(cached["plugins"]) == 2

    def test_use_cached_index(self, registry_cache_dir, mock_registry_index):
        """Should use cached index within TTL."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        cache_file = registry_cache_dir / "index.json"
        with open(cache_file, "w") as f:
            json.dump(mock_registry_index, f)

        timestamp_file = registry_cache_dir / "last_update"
        timestamp_file.touch()

        client = RegistryClient(cache_dir=registry_cache_dir)
        client._fetch_json = MagicMock(side_effect=Exception("Should not be called"))

        results = client.search("colmap")
        assert len(results) == 1


class TestChecksumVerification:
    """Tests for checksum verification."""

    def test_verify_checksum_valid(self, registry_cache_dir):
        """Should return True for valid checksum."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient
        import hashlib

        client = RegistryClient(cache_dir=registry_cache_dir)

        test_file = registry_cache_dir / "test.txt"
        test_file.write_text("test content")
        expected = hashlib.sha256(test_file.read_bytes()).hexdigest()

        assert client.verify_checksum(test_file, f"sha256:{expected}") is True

    def test_verify_checksum_invalid(self, registry_cache_dir):
        """Should return False for invalid checksum."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        client = RegistryClient(cache_dir=registry_cache_dir)

        test_file = registry_cache_dir / "test.txt"
        test_file.write_text("test content")

        assert client.verify_checksum(test_file, "sha256:wronghash") is False


class TestOfflineFallback:
    """Tests for offline behavior."""

    def test_offline_uses_cache(self, registry_cache_dir, mock_registry_index):
        """Should fall back to cache when offline."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient

        cache_file = registry_cache_dir / "index.json"
        with open(cache_file, "w") as f:
            json.dump(mock_registry_index, f)

        client = RegistryClient(cache_dir=registry_cache_dir)
        client._index = None

        with patch.object(client, "_fetch_json", side_effect=Exception("Offline")):
            results = client.search("colmap")

        assert len(results) == 1

    def test_offline_no_cache_raises(self, registry_cache_dir):
        """Should raise RegistryOfflineError when offline without cache."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.registry import RegistryClient
        from lfs_plugins.errors import RegistryOfflineError

        client = RegistryClient(cache_dir=registry_cache_dir)

        with patch.object(client, "_fetch_json", side_effect=Exception("Offline")):
            with pytest.raises(RegistryOfflineError):
                client.search("test")

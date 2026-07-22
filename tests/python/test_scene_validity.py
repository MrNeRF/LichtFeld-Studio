# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for scene validity API availability.

Note: Full validity testing (generation increments, is_valid checks) requires
GUI mode where SceneManager calls set_application_scene(). See the C++ test
test_scene_validity.cpp for comprehensive testing of the underlying mechanism.
"""

import pytest


class TestSceneValidityAPI:
    """Tests for scene validity API existence and basic behavior."""

    def test_get_scene_generation_exists(self, lf):
        """Test get_scene_generation function is exposed."""
        assert hasattr(lf, "get_scene_generation")
        gen = lf.get_scene_generation()
        assert isinstance(gen, int)
        assert gen >= 0

    def test_get_scene_returns_none_without_context(self, lf):
        """Test get_scene returns None in headless mode."""
        scene = lf.get_scene()
        assert scene is None

    def test_scene_class_has_is_valid_method(self, lf):
        """Test Scene class has is_valid method defined."""
        # Can't instantiate Scene without context, but we can check
        # the module structure via introspection
        assert hasattr(lf.scene, "Scene")
        # Check the class has the method
        scene_class = lf.scene.Scene
        assert hasattr(scene_class, "is_valid")
        assert hasattr(scene_class, "generation")

    def test_scene_uuid_api_is_additive(self, lf):
        """UUID lookup/properties coexist with the existing name-based API."""
        scene_class = lf.scene.Scene
        node_class = lf.scene.SceneNode
        assert hasattr(scene_class, "get_node")
        assert hasattr(scene_class, "get_node_by_uuid")
        assert hasattr(scene_class, "duplicate_node")
        assert hasattr(node_class, "id")
        assert hasattr(node_class, "uuid")

    def test_generation_is_stable_without_changes(self, lf):
        """Test generation doesn't change spontaneously."""
        gen1 = lf.get_scene_generation()
        gen2 = lf.get_scene_generation()
        gen3 = lf.get_scene_generation()
        assert gen1 == gen2 == gen3


class TestSceneLoadHeadless:
    """Tests for scene loading in headless mode."""

    def test_load_sog_returns_result(self, lf, test_sog):
        """Test loading SOG file returns a result object."""
        result = lf.io.load(str(test_sog))
        assert result is not None

    def test_no_scene_context_after_headless_load(self, lf, test_sog):
        """Test no scene context available after headless load.

        In headless mode without GUI, SceneManager doesn't run,
        so set_application_scene() is never called.
        """
        result = lf.io.load(str(test_sog))
        scene = lf.get_scene()
        # Expected: None in headless mode
        assert scene is None

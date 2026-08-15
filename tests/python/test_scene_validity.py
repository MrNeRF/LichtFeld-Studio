# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for scene validity API availability.

Note: Full validity testing (generation increments, is_valid checks) requires
GUI mode where SceneManager calls set_application_scene(). See the C++ test
test_scene_validity.cpp for comprehensive testing of the underlying mechanism.
"""

class TestSceneValidityAPI:
    """Tests for scene validity API existence and basic behavior."""

    def test_generation_is_stable_without_changes(self, lf):
        """Test generation doesn't change spontaneously."""
        gen1 = lf.get_scene_generation()
        gen2 = lf.get_scene_generation()
        gen3 = lf.get_scene_generation()
        assert isinstance(gen1, int)
        assert gen1 >= 0
        assert gen1 == gen2 == gen3


class TestSceneLoadHeadless:
    """Tests for scene loading in headless mode."""

    def test_no_scene_context_after_headless_load(self, lf, test_sog):
        """Test no scene context available after headless load.

        In headless mode without GUI, SceneManager doesn't run,
        so set_application_scene() is never called.
        """
        result = lf.io.load(str(test_sog))
        assert result is not None
        scene = lf.get_scene()
        # Expected: None in headless mode
        assert scene is None

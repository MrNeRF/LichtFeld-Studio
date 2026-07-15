"""Tests for the Blender-style operator system.

These tests verify the operator registration, invocation, and lifecycle methods.
All tests run without the visualizer GUI backend, so EditorContext is not available.
"""

import pytest


class TestOperatorAPIWithoutBackend:
    """Tests for operator API without C++ backend.

    Without a running visualizer, EditorContext is not available.
    These tests verify the API exists and doesn't crash.
    """

    def test_get_active_operator_returns_empty_without_backend(self, lf):
        """get_active_operator returns empty string without backend."""
        assert lf.ui.get_active_operator() == ""

    def test_get_gizmo_type_returns_empty_without_backend(self, lf):
        """get_gizmo_type returns empty string without backend."""
        assert lf.ui.get_gizmo_type() == ""

    def test_has_active_operator_returns_false_without_backend(self, lf):
        """has_active_operator returns False without backend."""
        assert not lf.ui.has_active_operator()


class TestOperatorAPISignatures:
    """Tests that verify API signatures exist correctly."""

    def test_active_operator_mutators_are_noops_without_backend(self, lf):
        lf.ui.set_active_operator("test.op")
        for gizmo_type in ("translate", "rotate", "scale", ""):
            lf.ui.set_active_operator("test.op", gizmo_type)
            assert lf.ui.get_active_operator() == ""
            assert lf.ui.get_gizmo_type() == ""
        lf.ui.clear_active_operator()
        assert not lf.ui.has_active_operator()


class TestSceneAPIs:
    """Tests for scene manipulation APIs used by operators."""

    def test_has_scene_without_backend(self, lf):
        """has_scene returns False without backend."""
        # Without SceneManager, should return False
        assert lf.has_scene() is False

    def test_has_selection_without_backend(self, lf):
        """has_selection returns False without backend."""
        assert lf.has_selection() is False

    def test_can_transform_selection_without_backend(self, lf):
        """can_transform_selection returns False without backend."""
        assert lf.can_transform_selection() is False

    def test_get_num_gaussians_without_backend(self, lf):
        """get_num_gaussians returns 0 without backend."""
        assert lf.get_num_gaussians() == 0

    def test_get_selected_node_transform_without_backend(self, lf):
        """get_selected_node_transform returns None without selection."""
        result = lf.get_selected_node_transform()
        assert result is None

    def test_get_selection_center_without_backend(self, lf):
        """get_selection_center returns None without selection."""
        result = lf.get_selection_center()
        assert result is None

    def test_get_selection_world_center_without_backend(self, lf):
        """get_selection_world_center warns and returns None without selection."""
        with pytest.warns(
            DeprecationWarning,
            match=r"lichtfeld\.get_selection_world_center\(\) is deprecated; use "
                  r"lichtfeld\.get_selection_visualizer_world_center\(\) instead",
        ):
            result = lf.get_selection_world_center()
        assert result is None

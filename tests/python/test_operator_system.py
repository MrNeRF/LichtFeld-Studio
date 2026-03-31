"""Tests for the Blender-style operator system.

These tests verify the operator registration, invocation, and lifecycle methods.
All tests run without GUI/OpenGL context, so EditorContext is not available.
"""

import pytest


class TestOperatorAPIWithoutBackend:
    """Tests for operator API without C++ backend.

    Without a running visualizer, EditorContext is not available.
    These tests verify the API exists and doesn't crash.
    """

    def test_set_active_operator_does_not_crash(self, lf):
        """set_active_operator doesn't crash without backend."""
        # Without EditorContext, this is a no-op
        lf.ui.set_active_operator("test.op", "translate")

    def test_clear_active_operator_does_not_crash(self, lf):
        """clear_active_operator doesn't crash without backend."""
        lf.ui.clear_active_operator()

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

    def test_set_active_operator_signature(self, lf):
        """set_active_operator accepts id and optional gizmo_type."""
        # Call with just id
        lf.ui.set_active_operator("test.op")
        # Call with both
        lf.ui.set_active_operator("test.op", "translate")
        # Call with empty gizmo type
        lf.ui.set_active_operator("test.op", "")

    def test_gizmo_type_values(self, lf):
        """All valid gizmo type strings are accepted."""
        # These are the valid gizmo types from the plan
        valid_types = ["translate", "rotate", "scale", ""]
        for gt in valid_types:
            lf.ui.set_active_operator("test.op", gt)


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

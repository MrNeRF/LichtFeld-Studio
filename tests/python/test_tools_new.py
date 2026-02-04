# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the new declarative tool system."""

import sys

import pytest

sys.path.insert(0, "build/python")

from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef, PivotModeDef
from lfs_plugins.tool_defs.builtin import (
    BUILTIN_TOOLS,
    get_tool_by_id,
    get_tools_by_group,
    get_all_groups,
)


class MockContext:
    """Mock context for poll testing."""

    def __init__(self, has_scene=True, num_gaussians=1000):
        self.has_scene = has_scene
        self.num_gaussians = num_gaussians


class TestToolDef:
    """Tests for ToolDef dataclass."""

    def test_basic_tool_definition(self):
        """ToolDef should hold basic properties."""
        tool = ToolDef(
            id="test.tool",
            label="Test Tool",
            icon="test-icon",
            group="test",
        )
        assert tool.id == "test.tool"
        assert tool.label == "Test Tool"
        assert tool.icon == "test-icon"
        assert tool.group == "test"

    def test_default_values(self):
        """ToolDef should have sensible defaults."""
        tool = ToolDef(id="test", label="Test", icon="test")
        assert tool.order == 100
        assert tool.description == ""
        assert tool.shortcut == ""
        assert tool.gizmo == ""
        assert tool.operator == ""
        assert tool.submodes == ()
        assert tool.pivot_modes == ()
        assert tool.poll is None

    def test_tool_with_submodes(self):
        """ToolDef should hold submodes."""
        tool = ToolDef(
            id="test",
            label="Test",
            icon="test",
            submodes=(
                SubmodeDef("mode1", "Mode 1", "icon1"),
                SubmodeDef("mode2", "Mode 2", "icon2"),
            ),
        )
        assert len(tool.submodes) == 2
        assert tool.submodes[0].id == "mode1"
        assert tool.submodes[1].label == "Mode 2"

    def test_tool_with_pivot_modes(self):
        """ToolDef should hold pivot modes."""
        tool = ToolDef(
            id="test",
            label="Test",
            icon="test",
            pivot_modes=(
                PivotModeDef("origin", "Origin", "origin-icon"),
                PivotModeDef("bounds", "Bounds", "bounds-icon"),
            ),
        )
        assert len(tool.pivot_modes) == 2

    def test_tool_with_poll(self):
        """ToolDef should support poll function."""

        def my_poll(ctx):
            return ctx.has_scene

        tool = ToolDef(id="test", label="Test", icon="test", poll=my_poll)
        assert tool.poll is not None
        assert tool.can_activate(MockContext(has_scene=True)) is True
        assert tool.can_activate(MockContext(has_scene=False)) is False

    def test_can_activate_without_poll(self):
        """can_activate should return True if no poll."""
        tool = ToolDef(id="test", label="Test", icon="test")
        assert tool.can_activate(MockContext()) is True

    def test_to_dict(self):
        """to_dict should convert tool to dictionary."""
        tool = ToolDef(
            id="test",
            label="Test",
            icon="test",
            submodes=(SubmodeDef("m1", "Mode 1", "i1"),),
        )
        d = tool.to_dict()
        assert d["id"] == "test"
        assert d["label"] == "Test"
        assert len(d["submodes"]) == 1
        assert d["submodes"][0]["id"] == "m1"

    def test_frozen(self):
        """ToolDef should be immutable."""
        tool = ToolDef(id="test", label="Test", icon="test")
        with pytest.raises(AttributeError):
            tool.id = "changed"


class TestSubmodeDef:
    """Tests for SubmodeDef dataclass."""

    def test_basic_submode(self):
        """SubmodeDef should hold properties."""
        s = SubmodeDef("local", "Local", "local-icon")
        assert s.id == "local"
        assert s.label == "Local"
        assert s.icon == "local-icon"
        assert s.shortcut == ""

    def test_submode_with_shortcut(self):
        """SubmodeDef can have shortcut."""
        s = SubmodeDef("local", "Local", "local-icon", shortcut="L")
        assert s.shortcut == "L"


class TestPivotModeDef:
    """Tests for PivotModeDef dataclass."""

    def test_basic_pivot_mode(self):
        """PivotModeDef should hold properties."""
        p = PivotModeDef("origin", "Origin", "origin-icon")
        assert p.id == "origin"
        assert p.label == "Origin"
        assert p.icon == "origin-icon"


class TestBuiltinTools:
    """Tests for builtin tool definitions."""

    def test_builtin_tools_defined(self):
        """BUILTIN_TOOLS should have tools defined."""
        assert len(BUILTIN_TOOLS) >= 7

    def test_required_tools_exist(self):
        """Required tools should exist."""
        ids = [t.id for t in BUILTIN_TOOLS]
        assert "builtin.select" in ids
        assert "builtin.translate" in ids
        assert "builtin.rotate" in ids
        assert "builtin.scale" in ids

    def test_get_tool_by_id(self):
        """get_tool_by_id should return correct tool."""
        tool = get_tool_by_id("builtin.translate")
        assert tool is not None
        assert tool.label == "Move"

    def test_get_tool_by_id_not_found(self):
        """get_tool_by_id should return None for unknown ID."""
        tool = get_tool_by_id("nonexistent.tool")
        assert tool is None

    def test_get_tools_by_group(self):
        """get_tools_by_group should return group tools."""
        transform_tools = get_tools_by_group("transform")
        assert len(transform_tools) >= 3
        for tool in transform_tools:
            assert tool.group == "transform"

    def test_get_tools_by_group_sorted(self):
        """get_tools_by_group should return sorted by order."""
        tools = get_tools_by_group("transform")
        orders = [t.order for t in tools]
        assert orders == sorted(orders)

    def test_get_all_groups(self):
        """get_all_groups should return unique groups."""
        groups = get_all_groups()
        assert "select" in groups
        assert "transform" in groups
        assert len(groups) == len(set(groups))

    def test_select_tool_has_submodes(self):
        """Selection tool should have submodes."""
        tool = get_tool_by_id("builtin.select")
        assert len(tool.submodes) >= 4

    def test_transform_tools_have_pivot_modes(self):
        """Transform tools should have pivot modes."""
        for tool_id in ["builtin.translate", "builtin.rotate", "builtin.scale"]:
            tool = get_tool_by_id(tool_id)
            assert len(tool.pivot_modes) >= 2

    def test_transform_tools_have_gizmo(self):
        """Transform tools should have gizmo type."""
        assert get_tool_by_id("builtin.translate").gizmo == "translate"
        assert get_tool_by_id("builtin.rotate").gizmo == "rotate"
        assert get_tool_by_id("builtin.scale").gizmo == "scale"

    def test_tools_have_shortcuts(self):
        """Tools should have keyboard shortcuts."""
        tool = get_tool_by_id("builtin.select")
        assert tool.shortcut == "1"
        tool = get_tool_by_id("builtin.translate")
        assert tool.shortcut == "2"


class TestToolPolling:
    """Tests for tool poll functions."""

    def test_select_requires_gaussians(self):
        """Select tool should require gaussians."""
        tool = get_tool_by_id("builtin.select")
        assert tool.can_activate(MockContext(has_scene=True, num_gaussians=100)) is True
        assert tool.can_activate(MockContext(has_scene=True, num_gaussians=0)) is False

    def test_translate_requires_scene(self):
        """Translate tool should require scene."""
        tool = get_tool_by_id("builtin.translate")
        assert tool.can_activate(MockContext(has_scene=True)) is True
        assert tool.can_activate(MockContext(has_scene=False)) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

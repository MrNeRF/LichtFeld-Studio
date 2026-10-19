import math

import pytest

from lfs_plugins.asset_layout import (
    LIST_ACTION_COLUMN_WIDTH,
    breakpoint_for_width,
    breakpoint_metrics,
    grid_columns,
    grid_slot_width,
    list_column_widths,
)


@pytest.mark.parametrize(
    ("width", "name"),
    [(359, "compact"), (360, "narrow"), (679, "narrow"),
     (680, "medium"), (999, "medium"), (1000, "wide")],
)
def test_breakpoints_use_frozen_boundaries(width, name):
    assert breakpoint_for_width(width) == name
    assert breakpoint_metrics(width)["breakpoint"] == name


@pytest.mark.parametrize("width", [260, 500, 679])
def test_stacked_inspector_metrics_preserve_a_readable_bottom_panel(width):
    metrics = breakpoint_metrics(width)
    assert metrics["inspector_placement"] == "bottom"
    assert metrics["inspector_default"] == 1000.0
    assert metrics["inspector_min"] == 180.0
    assert metrics["inspector_max"] == 1000.0


@pytest.mark.parametrize("width", [680, 900, 999])
def test_medium_keeps_the_navigator_compact_beside_the_inspector(width):
    metrics = breakpoint_metrics(width)
    assert metrics["navigator_mode"] == "dropdown"
    assert metrics["inspector_placement"] == "column"


def test_wide_navigator_preserves_core_list_affordances_when_it_appears():
    after = list_column_widths(1000 - 320 - 200)
    assert after["name"] >= 80.0
    assert after["gallery"] == 32.0
    assert sum(after.values()) + 24 + 16 + 32 + 8 + LIST_ACTION_COLUMN_WIDTH <= 480.1


@pytest.mark.parametrize("width", [260, 500, 760, 1100])
def test_grid_subtracts_gap_and_limits_card_stretch(width):
    logical_width = width
    columns = grid_columns(logical_width, 168)
    slot = grid_slot_width(logical_width, 168)
    assert columns >= 1
    assert slot <= 168 * 1.15 + 1e-9
    expected = min(
        (max(0, logical_width - 24) - 12 * (columns - 1)) / columns,
        168 * 1.15,
    )
    assert slot == pytest.approx(math.floor(expected * 10) / 10)
    assert columns * slot + (columns - 1) * 12 <= logical_width - 24


@pytest.mark.parametrize("width", [260, 360, 560, 700, 1100])
def test_list_columns_reserve_the_row_action_column(width):
    widths = list_column_widths(width)
    fixed_chrome = 24 + 16 + 32 + 8 + LIST_ACTION_COLUMN_WIDTH
    assert sum(widths.values()) + fixed_chrome <= width + 0.1


@pytest.mark.parametrize("width", [260, 480, 800, 1280])
def test_gallery_is_always_a_fixed_icon_column(width):
    assert list_column_widths(width)["gallery"] == 32.0

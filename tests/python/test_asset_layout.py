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
    [(419, "compact"), (420, "narrow"), (639, "narrow"),
     (640, "medium"), (899, "medium"), (900, "wide")],
)
def test_breakpoints_use_frozen_boundaries(width, name):
    assert breakpoint_for_width(width) == name
    assert breakpoint_metrics(width)["breakpoint"] == name


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

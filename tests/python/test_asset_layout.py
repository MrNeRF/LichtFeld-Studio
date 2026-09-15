import math

import pytest

from lfs_plugins.asset_layout import (
    breakpoint_for_width,
    breakpoint_metrics,
    card_geometry,
    grid_columns,
    grid_slot_width,
    list_row_height,
)


@pytest.mark.parametrize(
    ("width", "name"),
    [(260, "compact"), (320, "compact"), (420, "narrow"), (639, "narrow"),
     (640, "medium"), (760, "medium"), (899, "medium"), (900, "wide"), (1100, "wide")],
)
def test_breakpoints_use_frozen_boundaries(width, name):
    assert breakpoint_for_width(width) == name
    assert breakpoint_metrics(width)["breakpoint"] == name


@pytest.mark.parametrize("width", [260, 320, 500, 760, 1100])
@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_grid_subtracts_gap_and_limits_card_stretch(width, scale):
    logical_width = width * scale / scale
    columns = grid_columns(logical_width, 168)
    slot = grid_slot_width(logical_width, 168)
    assert columns >= 1
    assert slot <= 168 * 1.15 + 1e-9
    expected = min(
        (max(0, logical_width - 24) - 12 * (columns - 1)) / columns,
        168 * 1.15,
    )
    assert slot == pytest.approx(math.floor(expected * 10) / 10)


def test_card_geometry_is_sixteen_to_ten_with_fixed_body():
    geometry = card_geometry(168)
    assert geometry["thumbnail_height"] == pytest.approx(105)
    assert geometry["height"] == pytest.approx(145)


def test_list_row_height_changes_with_gallery_column():
    assert list_row_height(gallery_column_visible=True) == 40
    assert list_row_height(gallery_column_visible=False) == 48


def test_sub_640_inspector_is_a_real_open_strip():
    assert breakpoint_metrics(500)["inspector_placement"] == "strip"
    assert breakpoint_metrics(500)["inspector_default"] == 32

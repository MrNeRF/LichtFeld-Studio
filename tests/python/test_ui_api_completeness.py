# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral contracts for the completed Python UI APIs."""

import ctypes
import ctypes.util

import pytest


def test_draw_hook_receives_overlay_capable_layout(lf):
    observed = {}

    def draw(layout):
        layout.set_next_window_pos((12.0, 18.0))
        layout.set_next_window_size((240.0, 80.0))
        observed["opened"] = layout.begin_window("Hook test")
        observed["position"] = layout.get_window_pos()
        observed["width"] = layout.get_window_width()
        observed["text_size"] = layout.calc_text_size("Overlay")
        observed["draw_result"] = layout.draw_line(
            0.0, 0.0, 10.0, 10.0, (1.0, 1.0, 1.0, 1.0)
        )
        layout.end_window()

    lf.ui.add_hook("ui_api_completeness", "draw", draw)
    try:
        lf.ui.invoke_hooks("ui_api_completeness", "draw")
    finally:
        lf.ui.remove_hook("ui_api_completeness", "draw", draw)

    if not observed:
        pytest.skip("hook invocation requires an initialized LichtFeld GUI Python bridge")

    assert observed["opened"] is True
    assert observed["position"] == pytest.approx((12.0, 18.0))
    assert observed["width"] == pytest.approx(240.0)
    assert observed["text_size"][0] > 0.0
    assert observed["text_size"][1] > 0.0
    assert observed["draw_result"] is None


def test_is_key_down_matches_live_sdl_state_when_a_mapped_key_is_held(lf):
    library_name = ctypes.util.find_library("SDL3")
    if not library_name:
        pytest.skip("shared SDL3 state is unavailable in this headless test process")

    sdl = ctypes.CDLL(library_name)
    sdl.SDL_GetKeyboardState.argtypes = [ctypes.POINTER(ctypes.c_int)]
    sdl.SDL_GetKeyboardState.restype = ctypes.POINTER(ctypes.c_bool)
    count = ctypes.c_int()
    state = sdl.SDL_GetKeyboardState(ctypes.byref(count))

    mapped_keys = (
        (lf.ui.Key.ESCAPE, 41),
        (lf.ui.Key.ENTER, 40),
        (lf.ui.Key.TAB, 43),
        (lf.ui.Key.SPACE, 44),
        (lf.ui.Key.LEFT, 80),
        (lf.ui.Key.RIGHT, 79),
        (lf.ui.Key.UP, 82),
        (lf.ui.Key.DOWN, 81),
        (lf.ui.Key.F, 9),
        (lf.ui.Key.I, 12),
        (lf.ui.Key.M, 16),
        (lf.ui.Key.R, 21),
        (lf.ui.Key.T, 23),
    )
    held = [(key, scancode) for key, scancode in mapped_keys
            if scancode < count.value and state[scancode]]
    if not held:
        pytest.skip("no mapped key is held during this headless-safe state probe")

    for key, _ in held:
        assert lf.ui.is_key_down(key) is True


def test_key_queries_are_boolean_in_headless_processes(lf):
    assert isinstance(lf.ui.is_key_down(lf.ui.Key.SPACE), bool)
    assert isinstance(lf.ui.is_key_pressed(lf.ui.Key.SPACE), bool)
    assert "repeat: bool = False" in lf.ui.is_key_pressed.__doc__
    assert "key repeat is not emitted" in lf.ui.is_key_pressed.__doc__


def test_path_input_binding_round_trips_all_arguments_headlessly(lf):
    doc = lf.ui.RmlUILayout.path_input.__doc__
    assert "folder_mode: bool = True" in doc
    assert "dialog_title: str = ''" in doc
    assert "native file or folder browser" in doc

    changed, path = lf.ui.UILayout().path_input(
        "Output", "/tmp/example", False, "Choose output"
    )
    assert changed is False
    assert path == "/tmp/example"


def test_split_and_grid_flow_bindings_document_composition_arguments(lf):
    split_doc = lf.ui.RmlUILayout.split.__doc__
    grid_doc = lf.ui.RmlUILayout.grid_flow.__doc__
    assert "factor: float = 0.5" in split_doc
    assert "columns: int = 0" in grid_doc
    assert "even_columns: bool = True" in grid_doc
    assert "even_rows: bool = True" in grid_doc


def test_table_row_binding_documents_stable_and_positional_identity(lf):
    doc = lf.ui.RmlUILayout.table_next_row.__doc__
    assert "push_id()/pop_id()" in doc
    assert "position identity" in doc


def test_legacy_template_stubs_raise_instead_of_claiming_success(lf):
    layout = lf.ui.UILayout()

    with pytest.raises(TypeError, match="RmlUILayout.template_list"):
        layout.template_list("TEST_UL_items", "main", object(), "items", 0)
    with pytest.raises(TypeError, match="RmlUILayout.template_list"):
        layout.template_list(
            "TEST_UL_items", "main", object(), "items", object(), "active"
        )
    with pytest.raises(TypeError, match="RmlUILayout.combo"):
        layout.template_id("Item", ["a", "b"], "a")
    with pytest.raises(TypeError, match="RmlUILayout.tree_node/tree_pop"):
        layout.template_tree("Items", lambda _: None)

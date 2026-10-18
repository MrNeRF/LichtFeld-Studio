# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stable GPU identity, hardware inventory, and unavailable-device presentation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/python"))
from lfs_plugins.gpu_preferences import device_choices, hardware_rows, selected_value


def gpu(uuid, index=0, name="Same GPU", **values):
    return dict(uuid=uuid, index=index, name=name, **values)


def test_duplicate_names_keep_distinct_stable_choices_after_reordering():
    devices = [gpu("a" * 32, 1), gpu("b" * 32, 0)]
    choices = device_choices(devices, "GPU-" + "a" * 32, "tensor", str)
    assert selected_value(devices, "GPU-" + "a" * 32) == "a" * 32
    assert choices[1]["label"].endswith("aaaaaaaa")
    assert choices[2]["label"].endswith("bbbbbbbb")
    assert selected_value(devices, "0") == "b" * 32


def test_missing_saved_gpu_uses_auto_without_adding_a_phantom_device():
    choices = device_choices([gpu("a" * 32)], "missing", "tensor", str)
    assert len(choices) == 2
    assert all(choice["value"] != "missing" for choice in choices)
    assert selected_value([gpu("a" * 32)], "missing") == ""
    assert choices[0]["value"] == ""
    assert choices[0]["disabled"] is False


def test_changed_radeon_identity_only_lists_the_detected_vulkan_name():
    old = "00000000c70000000000000000000000"
    current = "00000000c80000000000000000000000"
    devices = [gpu(current, name="AMD Radeon(TM) 890M Graphics")]
    for role in ("tensor", "rendering"):
        assert device_choices(devices, old, role, str) == [
            dict(value="", label="preferences.tensor_auto", disabled=False),
            dict(value=current, label="AMD Radeon(TM) 890M Graphics", disabled=False),
        ]
    assert devices[0]["uuid"] == current


def test_capabilities_are_specific_to_each_role():
    devices = [gpu("a" * 32, tensor_reason="Missing feature", rendering_reason="")]
    assert device_choices(devices, "", "tensor", str)[1]["disabled"]
    assert not device_choices(devices, "", "rendering", str)[1]["disabled"]


def test_inventory_merges_api_identities_but_not_duplicate_names():
    rows = hardware_rows([gpu("a" * 32), gpu("b" * 32, 1)],
                         [gpu("GPU-" + "a" * 32)], str)
    assert len(rows) == 2
    assert rows[0]["backends"] == "Vulkan · CUDA"
    assert rows[1]["backends"] == "Vulkan"


def test_shared_heap_is_never_presented_as_dedicated_vram():
    def tr(key):
        return {"preferences.gpu_memory_shared": "Shared {size} GiB", "preferences.gpu_memory_dedicated": "GPU {size} GiB"}.get(key, key)
    rows = hardware_rows([gpu("a" * 32, memory_bytes=4 * 1024 ** 3, shared_memory=True)], [], tr)
    assert rows[0]["memory"] == "Shared 4.0 GiB"


def test_same_name_and_uuid_suffix_still_have_distinct_labels():
    devices = [gpu("a" * 24 + "0" * 8, 0), gpu("b" * 24 + "0" * 8, 1)]
    choices = device_choices(devices, "", "tensor", str)
    assert choices[1]["label"] != choices[2]["label"]


def test_inventory_distinguishes_identical_cards_and_matches_both_api_indices():
    rows = hardware_rows([gpu("a" * 32, 1), gpu("b" * 32, 2)], [gpu("a" * 32, 0)], str)
    assert rows[0]["name"] != rows[1]["name"]
    assert "Vulkan #1 / CUDA #0" in rows[0]["name"]
    assert rows[0]["name"].endswith("aaaaaaaa")
    assert "Vulkan #2" in rows[1]["name"]

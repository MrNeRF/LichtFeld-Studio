# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Production tool registry exposes two independent crop workflows."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "python"))
from lfs_plugins.tools import ToolRegistry


class CropRoleRegistryTest(unittest.TestCase):
    def test_roi_has_canonical_identity_and_legacy_alias(self):
        roi = ToolRegistry.get("builtin.roi")
        legacy = ToolRegistry.get("builtin.cropbox")
        crop = ToolRegistry.get("builtin.crop")
        self.assertIsNotNone(roi)
        self.assertIsNotNone(legacy)
        self.assertIsNotNone(crop)
        self.assertEqual(roi.id, "builtin.roi")
        self.assertEqual(legacy.id, "builtin.cropbox")
        self.assertEqual(roi.label, "ROI")
        self.assertEqual(crop.label, "Crop")
        ids = [tool.id for tool in ToolRegistry.get_all()]
        self.assertEqual(ids.count("builtin.roi"), 1)
        self.assertEqual(ids.count("builtin.cropbox"), 1)
    def test_roi_has_distinct_toolbar_icon(self):
        self.assertEqual(ToolRegistry.get("builtin.roi").icon, "roi")
        self.assertEqual(ToolRegistry.get("builtin.crop").icon, "cropbox")
        self.assertEqual(ToolRegistry.get("builtin.cropbox").icon, "cropbox")

    def test_existing_and_second_crop_tools_are_registered(self):
        original = ToolRegistry.get("builtin.cropbox")
        second = ToolRegistry.get("builtin.crop")
        self.assertIsNotNone(original)
        self.assertIsNotNone(second)
        self.assertNotEqual(original.id, second.id)


if __name__ == "__main__":
    unittest.main()

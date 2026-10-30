# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""SDK-free coverage of the typed host loader's versioned export inventory."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
HIP = ROOT / "src/core/tensor/backend/hip"


class HipHostExports(unittest.TestCase):
    def test_every_native_export_is_bound_exactly_once(self):
        declarations = []
        for path in HIP.glob("*_api.h"):
            declarations.extend(re.findall(
                r"LFS_HIP_API\s+(?:int|uint32_t)\s+lfs_hip_(\w+)\s*\(", path.read_text(encoding="utf-8")))
        bindings = re.findall(r"^LFS_HIP_RUNTIME_EXPORT\((\w+)\)$",
                              (HIP / "runtime_exports.def").read_text(encoding="utf-8"), re.MULTILINE)
        self.assertTrue(declarations)
        self.assertEqual(len(bindings), len(set(bindings)))
        self.assertEqual(set(bindings) | {"runtime_abi"}, set(declarations))
        self.assertNotIn("runtime_abi", bindings)


if __name__ == "__main__":
    unittest.main()

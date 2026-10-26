# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU-only contracts for catalog-driven HIP code generation; no SDK required."""
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/core/tensor/backend/hip/prepare_pointwise.py"
SOURCE = ROOT / "src/core/include/core/detail"
spec = importlib.util.spec_from_file_location("hip_pointwise_generator", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class HipPointwiseGeneration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "source"
        self.source.mkdir()
        for name in ("pointwise_ops.def", "tensor_functors.hpp", "pointwise_lowering.hpp"):
            (self.source / name).write_bytes((SOURCE / name).read_bytes())

    def test_shared_catalog_and_reachable_half_dispatch(self):
        names = re.findall(r'LFS_POINTWISE_OP\((\w+),', (SOURCE / "pointwise_ops.def").read_text())
        generated = generator.generate(self.source)
        for dtype in ("f32", "f16"):
            actual = re.findall(r'case LFS_HIP_OP_(\w+):', generated[f"hip_pointwise_{dtype}.inc"])
            self.assertEqual(actual, names if dtype == "f32" else [name for name in names if name in generator.HALF_OPERATIONS])
        self.assertEqual(len(generator.HALF_OPERATIONS), 17)
        self.assertEqual(generated["hip_pointwise_functors.hpp"], '#pragma once\n#include "tensor_functors.hpp"\n')
        self.assertEqual((self.source / "tensor_functors.hpp").read_bytes(), (SOURCE / "tensor_functors.hpp").read_bytes())

    def test_duplicate_catalog_entry_is_rejected(self):
        catalog = self.source / "pointwise_ops.def"
        with catalog.open("a") as stream:
            stream.write('\nLFS_POINTWISE_OP(AddScalar, ignored, "duplicate")\n')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            generator.generate(self.source)

    def test_empty_or_malformed_catalog_is_rejected(self):
        for text, message in (("", "empty"), ("unknown preprocessor directive", "catalog row")):
            with self.subTest(text=text):
                (self.source / "pointwise_ops.def").write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    generator.generate(self.source)

    def test_unknown_scalar_requires_explicit_support(self):
        catalog = self.source / "pointwise_ops.def"
        catalog.write_text(catalog.read_text().replace("AddScalar", "UnknownScalar", 1))
        with self.assertRaisesRegex(ValueError, "Unsupported scalar"):
            generator.generate(self.source)

    def test_unsupported_shared_lowering_is_rejected(self):
        lowering = self.source / "pointwise_lowering.hpp"
        lowering.write_text(lowering.read_text().replace("ops::scalar_right_op<ops::add_op, float>", "unknown_binding"))
        with self.assertRaisesRegex(ValueError, "Unsupported scalar"):
            generator.generate(self.source)

    def test_cli_preserves_unchanged_output_timestamps(self):
        output = Path(self.temporary.name) / "generated"
        command = [sys.executable, str(SCRIPT), "--source", str(self.source), "--output", str(output)]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        timestamps = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
        self.assertEqual(len(timestamps), 8)
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        self.assertEqual({path.name: path.stat().st_mtime_ns for path in output.iterdir()}, timestamps)


if __name__ == "__main__":
    unittest.main()

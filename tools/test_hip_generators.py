# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU-only smoke contracts for adapting the current upstream kernel sources."""
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


class HipGeneratorContracts(unittest.TestCase):
    def test_adaptations_are_repeatable_and_leave_upstream_sources_untouched(self):
        cases = [
            ('src/training/hip/prepare_fast.py', '--root', ROOT,
             ROOT / 'src/training/rasterization/fastgs', 'backward.cu'),
            ('src/training/hip/prepare_gsplat.py', '--source', ROOT / 'src/training/rasterization/gsplat',
             ROOT / 'src/training/rasterization/gsplat', 'RasterizeToPixelsFromWorld3DGSBwd.cu'),
            ('src/core/tensor/backend/hip/prepare_tensor_ops.py', '--source', ROOT / 'src/core/include/core/detail',
             ROOT / 'src/core/include/core/detail', 'hip_tensor_dispatch.inc'),
        ]
        with tempfile.TemporaryDirectory(prefix='lfs-hip-generators-') as temporary:
            for index, (script, flag, source, original, required) in enumerate(cases):
                with self.subTest(script=script):
                    before = hashes(original)
                    output = Path(temporary) / str(index)
                    output.mkdir()
                    command = [sys.executable, str(ROOT / script), flag, str(source), '--output', str(output)]
                    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
                    self.assertTrue((output / required).is_file())
                    first = hashes(output)
                    self.assertTrue(first)
                    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
                    self.assertEqual(hashes(output), first)
                    self.assertEqual(hashes(original), before)
                    if required == 'hip_tensor_dispatch.inc':
                        declared = set(re.findall(r'LFS_POINTWISE_OP\((\w+),',
                                                  (source / 'pointwise_ops.def').read_text()))
                        generated = set(re.findall(r'case LFS_HIP_OP_(\w+):', (output / required).read_text()))
                        self.assertTrue(declared)
                        self.assertEqual(generated, declared)


if __name__ == '__main__':
    unittest.main()

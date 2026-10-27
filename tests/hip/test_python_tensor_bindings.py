# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise production PyTensor bindings without loading the application."""
import argparse
import os
from pathlib import Path
import sys
import unittest
p = argparse.ArgumentParser()
p.add_argument("module_directory", type=Path)
p.add_argument("--runtime-directory", type=Path)
p.add_argument("--hip", action="store_true")
a = p.parse_args()
if sys.platform == "win32" and a.runtime_directory:
    # Match the native contract executables' dependency search paths; Python
    # 3.8+ deliberately ignores PATH for extension DLL dependencies.
    directories = [a.runtime_directory, *(Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p)]
    dlls = [os.add_dll_directory(str(p.resolve())) for p in directories if p.is_dir()]
sys.path.insert(0, str(a.module_directory.resolve()))
import _tensor_binding_contracts as bindings
class TensorBindings(unittest.TestCase):
    def test_cpu_identity_and_dlpack(self):
        t = bindings.Tensor.zeros([3], device="cpu")
        self.assertEqual(t.backend, "cpu")
        self.assertEqual(t.device, "cpu")
        self.assertEqual(t.__dlpack_device__(), (1, 0))
        # Import takes ownership of the capsule and its deleter.
        result = bindings.Tensor.from_dlpack(t)
        self.assertEqual(result.tolist(), [0, 0, 0])
    def test_cpu_export_ignores_consumer_gpu_stream(self):
        tensor = bindings.Tensor.zeros([2], device="cpu")
        capsule = tensor.__dlpack__(stream=object())
        result = bindings.Tensor.from_dlpack(capsule)
        self.assertEqual(result.tolist(), [0, 0])
    @unittest.skipUnless(a.hip, "HIP hardware execution explicitly disabled")
    def test_hip_identity_and_reject_both_dlpack_entries(self):
        for empty in (False, True):
            t = bindings.hip_tensor(empty)
            self.assertEqual(t.backend, "hip")
            self.assertEqual(t.device, "hip")
            self.assertFalse(t.is_cuda)
            with self.assertRaisesRegex(RuntimeError, "HIP DLPack.*cpu"):
                t.__dlpack_device__()
            for stream in (None, -1, 1, 123):
                with self.assertRaisesRegex(RuntimeError, "HIP DLPack.*cpu"):
                    t.__dlpack__(stream=stream)
            self.assertEqual(t.cpu().__dlpack_device__(), (1, 0))
unittest.main(argv=[sys.argv[0]])

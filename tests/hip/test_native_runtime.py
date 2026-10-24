# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for the SDK-only C boundary; no application or Torch import."""
import argparse
import ctypes as c
import os
from pathlib import Path
import unittest

options = None


class Error(c.Structure):
    _fields_ = [("native_code", c.c_int32), ("message", c.c_char * 256)]


class Device(c.Structure):
    _fields_ = [("name", c.c_char * 256), ("architecture", c.c_char * 256),
                ("pci_bus_id", c.c_char * 32), ("total_bytes", c.c_uint64),
                ("warp_size", c.c_int32)]


def bind(module, name, arguments, result=c.c_int):
    function = getattr(module, "lfs_hip_" + name)
    function.argtypes = arguments
    function.restype = result
    return function


class RuntimeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if options is None:
            raise unittest.SkipTest("Native GPU checks require the explicit module CLI")
        cls.dll = c.CDLL(str(options.module.resolve()))
        pointer = c.c_void_p
        error = c.POINTER(Error)
        signatures = {
            "runtime_abi": ([], c.c_uint32),
            "device_count": ([c.POINTER(c.c_int32), error], c.c_int),
            "device_info": ([c.c_int32, c.POINTER(Device), error], c.c_int),
            "context_create": ([c.c_int32, c.POINTER(pointer), error], c.c_int),
            "context_destroy": ([pointer, error], c.c_int),
            "context_synchronize": ([pointer, error], c.c_int),
            "buffer_allocate": ([pointer, c.c_size_t, c.POINTER(pointer), error], c.c_int),
            "buffer_release": ([pointer, pointer, error], c.c_int),
            "upload": ([pointer, pointer, c.c_size_t, pointer, c.c_size_t, error], c.c_int),
            "download": ([pointer, pointer, c.c_size_t, pointer, c.c_size_t, error], c.c_int),
            "affine_f32": ([pointer, pointer, pointer, c.c_size_t, c.c_float, c.c_float, error], c.c_int),
            "fill_f32": ([pointer, pointer, c.c_size_t, c.c_size_t, c.c_float, error], c.c_int),
        }
        cls.calls = {name: bind(cls.dll, name, args, result)
                     for name, (args, result) in signatures.items()}

    def checked(self, name, *args):
        error = Error()
        status = self.calls[name](*args, c.byref(error))
        self.assertEqual(status, 0, f"{name}: {error.native_code}: {error.message!r}")

    def test_abi(self):
        self.assertEqual(self.calls["runtime_abi"](), 1)

    def test_inventory_and_native_math(self):
        count = c.c_int32()
        self.checked("device_count", c.byref(count))
        self.assertGreater(count.value, options.device)
        info = Device()
        self.checked("device_info", options.device, c.byref(info))
        self.assertTrue(info.name)
        self.assertIn(info.warp_size, (32, 64))
        context, source, destination = c.c_void_p(), c.c_void_p(), c.c_void_p()
        self.checked("context_create", options.device, c.byref(context))
        try:
            values = (c.c_float * 513)(*(float(i - 256) / 8 for i in range(513)))
            actual = (c.c_float * 513)()
            size = c.sizeof(values)
            self.checked("buffer_allocate", context, size, c.byref(source))
            self.checked("buffer_allocate", context, size, c.byref(destination))
            self.checked("upload", context, source, 0, values, size)
            self.checked("affine_f32", context, source, destination, len(values), 2, -3)
            self.checked("download", context, destination, 0, actual, size)
            self.assertEqual(list(actual), [2 * x - 3 for x in values])
            error = Error()
            self.assertNotEqual(self.calls["download"](
                context, destination, size, actual, 4, c.byref(error)), 0)
            self.assertTrue(error.message)
            # An invalid call must not poison the queue or alter valid data.
            self.checked("download", context, destination, 0, actual, size)
            self.assertEqual(list(actual), [2 * x - 3 for x in values])
        finally:
            # Context destruction also owns any allocation left by a failed test.
            self.checked("context_destroy", context)

    def test_invalid_device_is_recoverable(self):
        context = c.c_void_p(1)
        error = Error()
        self.assertNotEqual(self.calls["context_create"](-1, c.byref(context), c.byref(error)), 0)
        self.assertIsNone(context.value)
        self.assertTrue(error.message)

    def test_buffer_ownership_and_release(self):
        owner, other, buffer = c.c_void_p(), c.c_void_p(), c.c_void_p()
        self.checked("context_create", options.device, c.byref(owner))
        try:
            self.checked("context_create", options.device, c.byref(other))
            self.checked("buffer_allocate", owner, 4, c.byref(buffer))
            error = Error()
            self.assertNotEqual(self.calls["buffer_release"](other, buffer, c.byref(error)), 0)
            self.assertTrue(error.message)
            self.checked("fill_f32", owner, buffer, 0, 1, 7)
            value = c.c_float()
            self.checked("download", owner, buffer, 0, c.byref(value), 4)
            self.assertEqual(value.value, 7)
            self.checked("buffer_release", owner, buffer)
            # A rejected foreign handle or duplicate release must not be dereferenced.
            self.assertNotEqual(self.calls["buffer_release"](owner, buffer, c.byref(error)), 0)
            self.checked("context_synchronize", owner)
        finally:
            if other.value:
                self.checked("context_destroy", other)
            self.checked("context_destroy", owner)

    def test_ranges_alignment_and_error_reset(self):
        context, buffer = c.c_void_p(), c.c_void_p()
        self.checked("context_create", options.device, c.byref(context))
        try:
            error = Error()
            rejected = c.c_void_p(1)
            self.assertNotEqual(self.calls["buffer_allocate"](
                context, 0, c.byref(rejected), c.byref(error)), 0)
            self.assertIsNone(rejected.value)
            self.checked("buffer_allocate", context, 513 * 4, c.byref(buffer))
            invalid = ((1, 1), (513 * 4, 1), (0, c.c_size_t(-1).value))
            for offset, count in invalid:
                with self.subTest(offset=offset, count=count):
                    self.assertNotEqual(self.calls["fill_f32"](
                        context, buffer, offset, count, 1, c.byref(error)), 0)
                    self.assertTrue(error.message)
            self.assertEqual(self.calls["fill_f32"](
                context, buffer, 0, 513, 2, c.byref(error)), 0)
            self.assertEqual(error.native_code, 0)
            self.assertEqual(error.message, b"")
            self.checked("fill_f32", context, buffer, 513 * 4, 0, 9)
            self.checked("affine_f32", context, buffer, buffer, 513, 2, -3)
            actual = (c.c_float * 513)()
            self.checked("download", context, buffer, 0, actual, c.sizeof(actual))
            self.assertEqual(list(actual), [1] * 513)
            # Empty host transfers permit null storage only for a valid range.
            self.checked("upload", context, buffer, 513 * 4, None, 0)
            self.checked("download", context, buffer, 513 * 4, None, 0)
        finally:
            self.checked("context_destroy", context)

    def test_invalid_arguments_clear_outputs(self):
        error, buffer = Error(), c.c_void_p(1)
        self.assertNotEqual(self.calls["buffer_allocate"](
            None, 4, c.byref(buffer), c.byref(error)), 0)
        self.assertIsNone(buffer.value)
        self.assertNotEqual(self.calls["context_create"](
            options.device, None, c.byref(error)), 0)
        self.assertNotEqual(self.calls["device_count"](None, c.byref(error)), 0)
        self.assertNotEqual(self.calls["context_destroy"](None, None), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--abi-only", action="store_true",
                        help="Check module ABI and null arguments without GPU queries or kernels")
    options = parser.parse_args()
    if not 0 <= options.device <= 2147483647:
        parser.error("--device must be an ordinal in 0..2147483647")
    directory = None
    if options.runtime_dir and os.name == "nt":
        directory = os.add_dll_directory(str(options.runtime_dir.resolve()))
    try:
        tests = ["RuntimeContract.test_abi", "RuntimeContract.test_invalid_arguments_clear_outputs"] if options.abi_only else None
        unittest.main(argv=[__file__], defaultTest=tests)
    finally:
        if directory is not None:
            directory.close()

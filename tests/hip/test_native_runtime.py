# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for the SDK-only C boundary; no application or Torch import."""
import argparse
import ctypes as c
from contextlib import contextmanager
import os
from pathlib import Path
import unittest

EXPECTED_ABI = 2
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
        actual_abi = bind(cls.dll, "runtime_abi", [], c.c_uint32)()
        if actual_abi != EXPECTED_ABI:
            raise RuntimeError(f"Native HIP ABI mismatch: expected {EXPECTED_ABI}, got {actual_abi}")
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
            "readback_create": ([pointer, c.POINTER(pointer), error], c.c_int),
            "readback_enqueue": ([pointer, pointer, pointer, c.c_size_t, c.c_size_t, error], c.c_int),
            "staged_upload": ([pointer, pointer, pointer, c.c_size_t, pointer, c.c_size_t, error], c.c_int),
            "marker_record": ([pointer, pointer, error], c.c_int),
            "readback_reserve": ([pointer, pointer, c.c_size_t, c.POINTER(c.c_uint64), error], c.c_int),
            "readback_append": ([pointer, pointer, pointer, c.c_size_t, c.c_size_t, c.c_size_t, error], c.c_int),
            "readback_wait": ([pointer, pointer, error], c.c_int),
            "readback_poll": ([pointer, pointer, pointer, c.c_size_t, c.POINTER(c.c_int32), error], c.c_int),
            "readback_destroy": ([pointer, pointer, error], c.c_int),
        }
        cls.calls = {name: bind(cls.dll, name, args, result)
                     for name, (args, result) in signatures.items()}

    def checked(self, name, *args):
        error = Error()
        status = self.calls[name](*args, c.byref(error))
        self.assertEqual(status, 0, f"{name}: {error.native_code}: {error.message!r}")

    def test_abi(self):
        self.assertEqual(self.calls["runtime_abi"](), EXPECTED_ABI)
        if not options.faults:
            self.assertFalse(hasattr(self.dll, "lfs_hip_test_fail_once"))
            self.assertFalse(hasattr(self.dll, "lfs_hip_test_live_resources"))

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
        slot = c.c_void_p(1)
        self.assertNotEqual(self.calls["readback_create"](None, c.byref(slot), c.byref(error)), 0)
        self.assertIsNone(slot.value)
        ready = c.c_int32(7)
        self.assertNotEqual(self.calls["readback_poll"](None, None, None, 0, c.byref(ready), c.byref(error)), 0)
        self.assertEqual(ready.value, 0)

    @contextmanager
    def staging_resources(self, size=4096, slots=2):
        context, buffer = c.c_void_p(), c.c_void_p()
        created = []
        self.checked("context_create", options.device, c.byref(context))
        try:
            self.checked("buffer_allocate", context, size, c.byref(buffer))
            for _ in range(slots):
                slot = c.c_void_p()
                self.checked("readback_create", context, c.byref(slot))
                created.append(slot)
            yield context, buffer, created
        finally:
            for slot in reversed(created):
                self.checked("readback_destroy", context, slot)
            self.checked("context_destroy", context)

    def consume(self, context, slot, destination=None, destination_bytes=None):
        self.checked("readback_wait", context, slot)
        ready = c.c_int32()
        if destination_bytes is None:
            destination_bytes = c.sizeof(destination) if destination is not None else 0
        self.checked("readback_poll", context, slot, destination, destination_bytes, c.byref(ready))
        self.assertEqual(ready.value, 1)
        self.checked("readback_poll", context, slot, None, 0, c.byref(ready))
        self.assertEqual(ready.value, 0)

    def test_small_poll_destination_does_not_consume_or_write(self):
        with self.staging_resources(slots=1) as (context, buffer, slots):
            slot = slots[0]
            self.checked("fill_f32", context, buffer, 0, 4, 11)
            self.checked("readback_enqueue", context, slot, buffer, 0, 16)
            actual = (c.c_float * 4)(-1, -2, -3, -4)
            error, ready = Error(), c.c_int32(7)
            self.assertNotEqual(self.calls["readback_poll"](
                context, slot, actual, 4, c.byref(ready), c.byref(error)), 0)
            self.assertEqual(ready.value, 0)
            self.assertEqual(list(actual), [-1, -2, -3, -4])
            self.consume(context, slot, actual)
            self.assertEqual(list(actual), [11] * 4)

    def test_borrowed_host_subranges_can_overlap_host_copies(self):
        with self.staging_resources(slots=1) as (context, buffer, slots):
            slot = slots[0]
            host = c.c_uint64()
            self.checked("readback_reserve", context, slot, 64, c.byref(host))
            payload = (c.c_float * 16).from_address(host.value)
            payload[:] = list(range(16))
            self.checked("staged_upload", context, slot, buffer, 0, c.c_void_p(host.value + 4), 32)
            self.consume(context, slot)
            self.checked("readback_enqueue", context, slot, buffer, 0, 32)
            self.consume(context, slot, c.c_void_p(host.value + 4), 60)
            self.assertEqual(list(payload[1:9]), list(range(1, 9)))
            self.assertEqual(list(payload[9:]), list(range(9, 16)))

    def test_staged_upload_owns_snapshot_and_preserves_order(self):
        with self.staging_resources() as (context, buffer, slots):
            upload, download = slots
            for count in (1, 513, 17):
                values = (c.c_float * count)(*(float(i) for i in range(count)))
                self.checked("staged_upload", context, upload, buffer, 0, values, c.sizeof(values))
                # The caller can immediately reuse its original pageable memory.
                values[:] = [-999] * count
                self.checked("affine_f32", context, buffer, buffer, count, 2, -3)
                self.checked("readback_enqueue", context, download, buffer, 0, c.sizeof(values))
                actual = (c.c_float * count)()
                self.consume(context, download, actual)
                self.consume(context, upload)
                self.assertEqual(list(actual), [2 * float(i) - 3 for i in range(count)])

    def test_staging_ownership_pending_and_teardown(self):
        with self.staging_resources(slots=1) as (context, buffer, slots):
            slot = slots[0]
            other = c.c_void_p()
            self.checked("context_create", options.device, c.byref(other))
            try:
                error = Error()
                self.assertNotEqual(self.calls["readback_destroy"](other, slot, c.byref(error)), 0)
                self.assertNotEqual(self.calls["context_destroy"](context, c.byref(error)), 0)
                self.checked("fill_f32", context, buffer, 0, 8, 7)
                self.checked("readback_enqueue", context, slot, buffer, 0, 32)
                self.assertNotEqual(self.calls["readback_enqueue"](context, slot, buffer, 0, 32, c.byref(error)), 0)
                self.checked("readback_wait", context, slot)
                self.assertNotEqual(self.calls["readback_enqueue"](context, slot, buffer, 0, 32, c.byref(error)), 0)
                host = c.c_uint64(123)
                self.assertNotEqual(self.calls["readback_reserve"](context, slot, 64, c.byref(host), c.byref(error)), 0)
                self.assertEqual(host.value, 0)
                self.assertNotEqual(self.calls["marker_record"](context, slot, c.byref(error)), 0)
                actual = (c.c_float * 8)()
                self.consume(context, slot, actual)
                self.assertEqual(list(actual), [7] * 8)
                # Exit with a pending transfer: slot destruction must drain it.
                self.checked("readback_enqueue", context, slot, buffer, 0, 32)
            finally:
                self.checked("context_destroy", other)

    def test_packed_readback_respects_reservation_after_shrinking(self):
        with self.staging_resources(slots=1) as (context, buffer, slots):
            slot = slots[0]
            self.checked("fill_f32", context, buffer, 0, 16, 3)
            host = c.c_uint64()
            for size in (128, 16, 64):
                self.checked("readback_reserve", context, slot, size, c.byref(host))
                self.assertNotEqual(host.value, 0)
                error = Error()
                self.assertNotEqual(self.calls["readback_append"](context, slot, buffer, 0, 4, size, c.byref(error)), 0)
                self.checked("readback_append", context, slot, buffer, 0, 4, 0)
                self.checked("readback_append", context, slot, buffer, 4, 4, size - 4)
                actual = (c.c_ubyte * (size + 16))(*([0xA5] * (size + 16)))
                self.consume(context, slot, actual)
                self.assertEqual(bytes(actual[size:]), bytes([0xA5] * 16))
                self.assertEqual(c.c_float.from_buffer(actual).value, 3)
                self.assertEqual(c.c_float.from_buffer(actual, size - 4).value, 3)
                self.assertEqual(bytes(actual[4:size - 4]), bytes(size - 8))

    def test_empty_transfers_and_reusable_markers(self):
        with self.staging_resources(slots=1) as (context, buffer, slots):
            marker = slots[0]
            for _ in range(3):
                self.checked("fill_f32", context, buffer, 0, 1, 5)
                self.checked("marker_record", context, marker)
                self.checked("marker_record", context, marker)
                self.consume(context, marker)
            self.checked("staged_upload", context, marker, None, 0, None, 0)
            self.consume(context, marker)
            self.checked("readback_enqueue", context, marker, None, 0, 0)
            self.consume(context, marker)
            error = Error()
            self.assertNotEqual(self.calls["readback_enqueue"](context, marker, None, 1, 0, c.byref(error)), 0)


class FaultRecovery(RuntimeContract):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fail_once = bind(cls.dll, "test_fail_once", [c.c_int], None)
        cls.live_resources = bind(cls.dll, "test_live_resources", [], c.c_int)

    def tearDown(self):
        self.fail_once(0)
        self.assertEqual(self.live_resources(), 0)

    def reject(self, name, *arguments):
        error = Error()
        self.assertNotEqual(self.calls[name](*arguments, c.byref(error)), 0)
        self.assertTrue(error.message)

    def discard_failure(self, context, slot):
        self.reject("readback_wait", context, slot)
        actual = c.c_float(-123)
        ready = c.c_int32(7)
        self.reject("readback_poll", context, slot, c.byref(actual), 4, c.byref(ready))
        self.assertEqual(ready.value, 0)
        self.assertEqual(actual.value, -123)
        self.reject("readback_poll", context, slot, None, 0, c.byref(ready))
        self.assertEqual(ready.value, 0)
        self.checked("readback_poll", context, slot, None, 0, c.byref(ready))
        self.assertEqual(ready.value, 0)

    def test_failed_copies_never_expose_successful_snapshots(self):
        for operation in ("upload", "readback", "append"):
            with self.subTest(operation=operation), self.staging_resources(slots=1) as (context, buffer, slots):
                slot = slots[0]
                old = c.c_float(7)
                self.checked("staged_upload", context, slot, buffer, 0, c.byref(old), 4)
                self.consume(context, slot)
                self.checked("fill_f32", context, buffer, 0, 1, 9)
                if operation == "append":
                    host = c.c_uint64()
                    self.checked("readback_reserve", context, slot, 16, c.byref(host))
                    self.checked("readback_append", context, slot, buffer, 0, 4, 4)
                self.fail_once(6)
                if operation == "upload":
                    self.reject("staged_upload", context, slot, buffer, 0, c.byref(old), 4)
                elif operation == "readback":
                    self.reject("readback_enqueue", context, slot, buffer, 0, 4)
                else:
                    self.reject("readback_append", context, slot, buffer, 0, 4, 0)
                    self.reject("readback_append", context, slot, buffer, 0, 4, 0)
                # A too-small destination must neither write nor consume the failure.
                if operation == "append":
                    ready = c.c_int32(7)
                    self.reject("readback_poll", context, slot, c.byref(old), 4, c.byref(ready))
                    self.assertEqual(ready.value, 0)
                    self.assertEqual(old.value, 7)
                    self.reject("readback_wait", context, slot)
                    self.reject("readback_poll", context, slot, None, 0, c.byref(ready))
                    self.assertEqual(ready.value, 0)
                else:
                    self.discard_failure(context, slot)
                self.checked("readback_enqueue", context, slot, buffer, 0, 4)
                actual = c.c_float()
                self.consume(context, slot, c.byref(actual), c.sizeof(actual))
                self.assertEqual(actual.value, 9)

    def test_event_record_failure_keeps_accepted_copy_valid(self):
        for upload in (False, True):
            with self.subTest(upload=upload), self.staging_resources(slots=1) as (context, buffer, slots):
                slot = slots[0]
                value = c.c_float(13)
                self.checked("fill_f32", context, buffer, 0, 1, 13)
                self.fail_once(3)
                if upload:
                    self.reject("staged_upload", context, slot, buffer, 0, c.byref(value), 4)
                else:
                    self.reject("readback_enqueue", context, slot, buffer, 0, 4)
                self.consume(context, slot, c.byref(value), c.sizeof(value))
                self.assertEqual(value.value, 13)

    def test_growth_and_cleanup_failures_are_retryable(self):
        with self.staging_resources(slots=1) as (context, buffer, slots):
            slot = slots[0]
            host = c.c_uint64()
            self.checked("readback_reserve", context, slot, 64, c.byref(host))
            for fault, size in ((1, 128), (2, 256)):
                self.fail_once(fault)
                self.reject("readback_reserve", context, slot, size, c.byref(host))
                self.assertEqual(host.value, 0)
                self.reject("readback_append", context, slot, buffer, 0, 4, 0)
                self.checked("readback_reserve", context, slot, size, c.byref(host))
            self.checked("readback_enqueue", context, slot, buffer, 0, 4)
            self.fail_once(7)
            self.reject("readback_destroy", context, slot)
            self.checked("readback_wait", context, slot)
            self.fail_once(4)
            self.reject("readback_destroy", context, slot)
            self.fail_once(2)
            self.reject("readback_destroy", context, slot)
            self.checked("readback_destroy", context, slot)
            slots.clear()
            rejected = c.c_void_p(123)
            self.fail_once(5)
            self.reject("readback_create", context, c.byref(rejected))
            self.assertIsNone(rejected.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--abi-only", action="store_true",
                        help="Check module ABI and null arguments without GPU queries or kernels")
    parser.add_argument("--faults", action="store_true",
                        help="Run error recovery checks against the separate instrumented test module")
    options = parser.parse_args()
    if not 0 <= options.device <= 2147483647:
        parser.error("--device must be an ordinal in 0..2147483647")
    if options.abi_only and options.faults:
        parser.error("--abi-only and --faults are mutually exclusive")
    directory = None
    if options.runtime_dir and os.name == "nt":
        directory = os.add_dll_directory(str(options.runtime_dir.resolve()))
    try:
        tests = ["RuntimeContract.test_abi", "RuntimeContract.test_invalid_arguments_clear_outputs"] if options.abi_only else ["RuntimeContract"]
        if options.faults:
            tests = ["FaultRecovery.test_failed_copies_never_expose_successful_snapshots",
                     "FaultRecovery.test_event_record_failure_keeps_accepted_copy_valid",
                     "FaultRecovery.test_growth_and_cleanup_failures_are_retryable"]
        unittest.main(argv=[__file__], defaultTest=tests)
    finally:
        if directory is not None:
            directory.close()

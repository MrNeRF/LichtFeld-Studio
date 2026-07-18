# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 9 typed-error boundary tests.

Covers the C++->Python translation hierarchy (Section 1), the Python->C++
containment extraction round-trip (Section 2.1), the frame-callback disable
policy (Section 2.3 site 1), and the DLPack TypeError narrowing (Section 1.5).

Exercises the built `lichtfeld` module directly via the module-internal,
underscore-prefixed `_testing` hooks (raise_error / probe_python_error /
tick_frame / raise_memory_allocation_error). Assertions target exception
types and attributes, never localized message strings.
"""

import pytest


class TestTranslationHierarchy:
    """Section 1.1 / 1.2: each ErrorCode maps to its typed subclass + attributes."""

    def test_not_found_maps_to_notfounderror(self, lf):
        with pytest.raises(lf.NotFoundError) as excinfo:
            lf._testing.raise_error("NotFound", "IO", "missing file")
        e = excinfo.value
        assert isinstance(e, lf.Error)
        assert isinstance(e, RuntimeError)
        assert isinstance(e, FileNotFoundError)
        assert e.code == "NotFound"
        assert e.domain == "IO"
        assert str(e) == "missing file"
        assert str(e) == e.user_message

    def test_invalid_argument_maps_to_valueerror(self, lf):
        with pytest.raises(lf.InvalidArgumentError) as excinfo:
            lf._testing.raise_error("InvalidArgument", "Core", "bad arg")
        e = excinfo.value
        assert isinstance(e, lf.Error)
        assert isinstance(e, ValueError)
        assert e.code == "InvalidArgument"

    def test_bounds_violation_maps_to_invalid_argument(self, lf):
        with pytest.raises(lf.InvalidArgumentError) as excinfo:
            lf._testing.raise_error("BoundsViolation", "Core", "out of range")
        assert excinfo.value.code == "BoundsViolation"

    def test_cancelled_maps_to_cancellederror(self, lf):
        with pytest.raises(lf.CancelledError) as excinfo:
            lf._testing.raise_error("Cancelled", "IO", "cancelled")
        e = excinfo.value
        assert isinstance(e, lf.Error)
        assert e.code == "Cancelled"

    def test_resource_exhausted_maps_to_resourceerror_with_bytes(self, lf):
        with pytest.raises(lf.ResourceError) as excinfo:
            lf._testing.raise_error(
                "ResourceExhausted", "CUDA", "out of memory", requested_bytes=4096
            )
        e = excinfo.value
        assert isinstance(e, lf.Error)
        assert e.code == "ResourceExhausted"
        assert e.requested_bytes == 4096
        assert e.available_bytes is None

    def test_unmapped_code_rides_base_error(self, lf):
        # ErrorCode::Internal has no subclass row -> base lichtfeld.Error.
        with pytest.raises(lf.Error) as excinfo:
            lf._testing.raise_error("Internal", "Python", "internal failure")
        e = excinfo.value
        assert type(e) is lf.Error
        assert e.code == "Internal"
        assert e.domain == "Python"

    def test_instance_attributes_present(self, lf):
        with pytest.raises(lf.Error) as excinfo:
            lf._testing.raise_error("Internal", "Core", "attrs")
        e = excinfo.value
        assert isinstance(e.operation_id, int)
        assert isinstance(e.retryable, bool)
        assert isinstance(e.details, str)
        assert len(e.details) > 0
        assert len(e.details.encode("utf-8")) <= 32768
        assert isinstance(e.context, tuple)


class TestRealConvertedGroup:
    """Section 1.5 demo / BINDING FIX 2: py_io is the converted reference group."""

    def test_io_load_missing_file_is_typed_notfound(self, lf):
        with pytest.raises(lf.NotFoundError) as excinfo:
            lf.io.load("/nonexistent/path/x.ply")
        e = excinfo.value
        assert isinstance(e, FileNotFoundError)
        assert isinstance(e, RuntimeError)
        assert e.code == "NotFound"
        assert e.domain == "IO"

    def test_io_load_missing_file_still_catchable_as_runtimeerror(self, lf):
        # Compatibility pin (AMB 2): existing `except RuntimeError` keeps working.
        with pytest.raises(RuntimeError):
            lf.io.load("/nonexistent/path/x.ply")


class TestLegacyTypeMapping:
    """Section 1.3 rule 2: core::MemoryAllocationError -> lichtfeld.ResourceError."""

    def test_memory_allocation_error_maps_to_resource_error(self, lf):
        with pytest.raises(lf.ResourceError) as excinfo:
            lf._testing.raise_memory_allocation_error(1 << 20)
        e = excinfo.value
        assert isinstance(e, lf.Error)
        assert e.code == "ResourceExhausted"


class TestContainmentRoundTrip:
    """Section 2.1: error_from_python extraction (code table + traceback)."""

    def test_zero_division_is_internal_with_traceback(self, lf):
        info = lf._testing.probe_python_error(lambda: 1 / 0)
        assert info["code"] == "Internal"
        assert info["domain"] == "Python"
        assert info["py_type"] == "ZeroDivisionError"
        assert "Traceback" in info["detail"]

    def test_keyboard_interrupt_maps_to_cancelled(self, lf):
        def raise_ki():
            raise KeyboardInterrupt()

        info = lf._testing.probe_python_error(raise_ki)
        assert info["code"] == "Cancelled"

    def test_cancelled_error_round_trips(self, lf):
        def raise_cancelled():
            raise lf.CancelledError("stop")

        info = lf._testing.probe_python_error(raise_cancelled)
        assert info["code"] == "Cancelled"

    def test_memory_error_maps_to_resource_exhausted(self, lf):
        def raise_mem():
            raise MemoryError("oom")

        info = lf._testing.probe_python_error(raise_mem)
        assert info["code"] == "ResourceExhausted"

    def test_file_not_found_maps_to_not_found(self, lf):
        def raise_fnf():
            raise FileNotFoundError("nope")

        info = lf._testing.probe_python_error(raise_fnf)
        assert info["code"] == "NotFound"

    def test_asyncio_cancelled_error_maps_to_cancelled(self, lf):
        import asyncio

        def raise_cancelled():
            raise asyncio.CancelledError()

        info = lf._testing.probe_python_error(raise_cancelled)
        assert info["code"] == "Cancelled"

    def test_lookalike_cancelled_error_stays_internal(self, lf):
        class CancelledError(Exception):  # name collision, not a cancellation
            pass

        def raise_fake():
            raise CancelledError("not a real cancel")

        info = lf._testing.probe_python_error(raise_fake)
        assert info["code"] == "Internal"


class TestFrameCallbackDisable:
    """Section 2.3 site 1: a failing on_frame callback is unregistered after one tick."""

    def test_failing_frame_callback_disables_after_first_tick(self, lf):
        calls = []

        def cb(dt):
            calls.append(dt)
            raise RuntimeError("frame boom")

        lf.on_frame(cb)
        try:
            lf._testing.tick_frame(0.016)
            lf._testing.tick_frame(0.016)
        finally:
            lf.stop_animation()
        assert len(calls) == 1


class TestDlpackNarrowing:
    """Section 1.5 / doc :202: only a TypeError triggers the legacy zero-arg retry."""

    def test_typeerror_triggers_legacy_retry(self, lf):
        class TypeErrorProducer:
            def __init__(self):
                self.calls = []

            def __dlpack__(self, stream=None):
                self.calls.append(stream)
                if stream is not None:
                    raise TypeError("__dlpack__ received an unexpected stream argument")
                raise RuntimeError("legacy-zero-arg-reached")

            def __dlpack_device__(self):
                return (1, 0)

        producer = TypeErrorProducer()
        with pytest.raises(RuntimeError, match="legacy-zero-arg-reached"):
            lf.Tensor.from_dlpack(producer)
        assert len(producer.calls) == 2
        assert producer.calls[0] is not None
        assert producer.calls[1] is None

    def test_non_typeerror_propagates_without_retry(self, lf):
        class ValueErrorProducer:
            def __init__(self):
                self.calls = []

            def __dlpack__(self, stream=None):
                self.calls.append(stream)
                raise ValueError("producer-bug")

            def __dlpack_device__(self):
                return (1, 0)

        producer = ValueErrorProducer()
        with pytest.raises(ValueError, match="producer-bug"):
            lf.Tensor.from_dlpack(producer)
        assert len(producer.calls) == 1

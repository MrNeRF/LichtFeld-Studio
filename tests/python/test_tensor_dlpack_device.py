# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Storage-free DLPack exports retain the selected CUDA device ordinal."""

import ctypes

import pytest


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int32),
        ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [("tensor", DLTensor), ("context", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


def capsule_device(tensor):
    capsule = tensor.__dlpack__()
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(capsule, b"dltensor")
    managed = ctypes.cast(pointer, ctypes.POINTER(DLManagedTensor)).contents
    try:
        return managed.tensor.device.device_type, managed.tensor.device.device_id
    finally:
        # This export uses a consumer-owned deleter and no capsule destructor.
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(managed.deleter)(pointer)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [[0], [2, 0, 3], [0, 0]])
def test_empty_cuda_dlpack_reports_the_selected_device(lf, gpu_available, shape):
    if not gpu_available:
        pytest.skip("GPU unavailable")
    resident = lf.Tensor.zeros([1], device="cuda")
    if resident.backend != "cuda":
        pytest.skip("Requires the CUDA tensor backend")
    expected = resident.__dlpack_device__()
    assert expected[0] == 2
    assert expected[1] >= 0
    assert capsule_device(resident) == expected
    empty = lf.Tensor.zeros(shape, device="cuda")
    assert empty.__dlpack_device__() == expected
    assert capsule_device(empty) == expected


def test_empty_cpu_dlpack_keeps_cpu_device_zero(lf):
    empty = lf.Tensor.zeros([0], device="cpu")
    assert empty.__dlpack_device__() == (1, 0)
    assert capsule_device(empty) == (1, 0)

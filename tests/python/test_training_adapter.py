# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Public bounded SH extraction and GIL-release contracts."""

import os
from pathlib import Path
import struct
import subprocess
import sys
import textwrap

import pytest


def test_owned_host_numpy_export(lf, numpy):
    original = numpy.arange(48, dtype=numpy.float32).reshape(8, 6)
    tensor = lf.Tensor.from_numpy(original)
    host = tensor.cpu()
    host.sync()
    exported = host.numpy(copy=True)
    numpy.testing.assert_array_equal(exported, original)
    exported[:] = -10
    numpy.testing.assert_array_equal(tensor.numpy(), original)


@pytest.mark.gpu
def test_sync_releases_gil_for_cuda_host_callback(gpu_available, tmp_path):
    if not gpu_available:
        pytest.skip("CUDA GPU required")
    # A PyDLL enqueue keeps the GIL held until Tensor.sync. With no GIL release
    # there, the queued Python callback and the device synchronization deadlock.
    # A subprocess timeout contains that regression without hanging pytest.
    code = textwrap.dedent('''\
        import ctypes, ctypes.util, os, sys
        from pathlib import Path
        build = Path(os.environ["LFS_TEST_BUILD_DIR"])
        if os.name == "nt":
            dll_directory = os.add_dll_directory(str(build))
        import lichtfeld as lf
        import numpy as np
        tensor = lf.Tensor.from_numpy(np.ones(1, dtype=np.float32)).cuda()
        tensor.sync()
        library = "cudart64_12.dll" if os.name == "nt" else ctypes.util.find_library("cudart")
        runtime = ctypes.PyDLL(library)
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        runtime.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        runtime.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
        runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        stream = ctypes.c_void_p()
        assert runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1) == 0
        calls = []
        callback = callback_type(lambda _: calls.append(True))
        sync = tensor.sync
        previous_interval = sys.getswitchinterval()
        try:
            sys.setswitchinterval(60)
            assert runtime.cudaLaunchHostFunc(stream, callback, None) == 0
            sync()
            assert calls == [True]
        finally:
            sys.setswitchinterval(previous_interval)
            assert runtime.cudaStreamDestroy(stream) == 0
        print("CUDA_CALLBACK_GIL_RELEASE_PASS")
    ''')
    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("LFS_TEST_BUILD_DIR", os.environ.get("LFS_BUILD_DIR", root / "build")))
    environment = {**os.environ, "LFS_TEST_BUILD_DIR": str(build),
                   "PYTHONPATH": os.pathsep.join(sys.path), "LFS_HOME": str(tmp_path / "runtime")}
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True,
                            text=True, timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA_CALLBACK_GIL_RELEASE_PASS" in result.stdout


@pytest.fixture
def sh_model(lf, gpu_available, tmp_path):
    if not gpu_available:
        pytest.skip("CUDA GPU required for PLY loading")
    properties = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    properties += [f"f_rest_{i}" for i in range(45)]
    properties += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = ["ply", "format binary_little_endian 1.0", "element vertex 40"]
    header += [f"property float {name}" for name in properties]
    header.append("end_header")
    payload = bytearray(("\n".join(header) + "\n").encode("ascii"))
    for i in range(40):
        row = [i * .01, 0, 0, 0, 0, 0]
        row += [(i * 45 + j) * .001 for j in range(45)]
        row += [0, -3, -3, -3, 1, 0, 0, 0]
        payload.extend(struct.pack("<" + "f" * len(row), *row))
    path = tmp_path / "sh3.ply"
    path.write_bytes(payload)
    model = lf.io.load(str(path)).splat_data
    assert model is not None
    return model


@pytest.mark.gpu
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_selected_sh_rows_match_canonical_and_are_owned(lf, numpy, sh_model, degree, dtype):
    requested = numpy.asarray([39, 0, 31, 32, 31], dtype=dtype)
    canonical = sh_model.shN_raw.cpu().numpy()
    selected = sh_model.shN_rows(lf.Tensor.from_numpy(requested), degree)
    result = selected.numpy(copy=False)
    expected = canonical[requested, : (degree + 1) ** 2 - 1]
    assert selected.device == "cpu"
    assert selected.dtype == "float32"
    numpy.testing.assert_array_equal(result, expected)
    if result.size:
        result[:] = 100
        numpy.testing.assert_array_equal(sh_model.shN_raw.cpu().numpy(), canonical)


@pytest.mark.gpu
def test_selected_sh_rows_reject_invalid_requests(lf, numpy, sh_model):
    for rows in (numpy.asarray([-1], dtype="int64"), numpy.asarray([40], dtype="int64")):
        with pytest.raises(IndexError):
            sh_model.shN_rows(lf.Tensor.from_numpy(rows), 1)
    for rows in (numpy.asarray([0.0], dtype="float32"), numpy.asarray([[0]], dtype="int64")):
        with pytest.raises(ValueError):
            sh_model.shN_rows(lf.Tensor.from_numpy(rows), 1)
    rows = lf.Tensor.from_numpy(numpy.asarray([], dtype="int64"))
    assert sh_model.shN_rows(rows, 3).shape == (0, 15, 3)
    for degree in (-1, 4):
        with pytest.raises(ValueError):
            sh_model.shN_rows(rows, degree)

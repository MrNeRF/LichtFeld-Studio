# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Public tensor/SH APIs and real headless training hook outcomes."""

import json
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


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("outcome", ["completed", "stopped", "failed"])
def test_headless_training_hook_outcome(tmp_path, gpu_available, outcome):
    if not gpu_available:
        pytest.skip("CUDA GPU required for training")
    from test_mcp_dataset_load_training import _dataset

    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("LFS_TEST_BUILD_DIR", os.environ.get("LFS_BUILD_DIR", root / "build")))
    executable = Path(os.environ.get("LFS_EXECUTABLE", build / ("LichtFeld-Studio.exe" if os.name == "nt" else "LichtFeld-Studio")))
    assert executable.is_file(), f"Build the executable first: {executable}"
    dataset = _dataset(tmp_path / "dataset", 60)
    script = tmp_path / "hooks.py"
    script.write_text(textwrap.dedent('''\
        import json, os
        from pathlib import Path
        import lichtfeld as lf
        steps = []
        if os.environ["LFS_TEST_OUTCOME"] == "completed":
            # First-use adapter setup must launch uv with a valid environment
            # block, including non-ASCII paths. This creates no downloaded packages.
            runtime = Path(lf.packages.init())
            assert (runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")).is_file()
        @lf.on_post_step
        def step(context):
            steps.append(dict(context))
            if os.environ["LFS_TEST_OUTCOME"] == "stopped":
                lf.session().request_stop()
        @lf.on_training_end
        def end(context):
            Path(os.environ["LFS_TEST_HOOK_OUTPUT"]).write_text(
                json.dumps({"end": dict(context), "steps": steps}), encoding="utf-8")
    '''), encoding="utf-8")
    checkpoint = tmp_path / "project.licht"
    if outcome == "failed":
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("Block the terminal project save", encoding="utf-8")
        checkpoint = blocked / "project.licht"
    proof = tmp_path / "hooks.json"
    environment = {**os.environ, "LFS_HOME": str(tmp_path / "runtime-\u00e0-\u6e2c"),
                   "LFS_TEST_OUTCOME": outcome, "LFS_TEST_HOOK_OUTPUT": str(proof)}
    for name in ("LFS_TRAINING_PUBLISHER", "LFS_BLENDER_JOB", "PYTHONHOME", "PYTHONPATH"):
        environment.pop(name, None)
    command = [str(executable), "--headless", "-d", dataset, "-o", str(tmp_path / "output"),
               "--iter", "4", "--max-cap", "1000", "--sh-degree", "1", "--strategy", "mcmc",
               "--min-track-length", "0", "--centralize", "off", "--no-download",
               "--save-project-at-iter", "4", "--save-project-path", str(checkpoint),
               "--python-script", str(script)]
    with (tmp_path / "trainer.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=60,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert proof.is_file(), (tmp_path / "trainer.log").read_text(encoding="utf-8")[-8000:]
    payload = json.loads(proof.read_text(encoding="utf-8"))
    assert payload["end"]["terminal_state"] == outcome
    assert payload["end"]["iterations"] == 4
    assert type(payload["end"]["is_paused"]) is bool
    if outcome == "stopped":
        assert 0 <= payload["end"]["iteration"] < 4
    else:
        assert payload["end"]["iteration"] == 4
    assert payload["steps"]
    for step in payload["steps"]:
        assert step["iterations"] == 4
        assert type(step["is_paused"]) is bool
        assert "terminal_state" not in step
    assert (process.returncode != 0) == (outcome == "failed")

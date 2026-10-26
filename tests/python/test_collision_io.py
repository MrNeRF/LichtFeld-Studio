"""Collision I/O is a computation API, not a panel-opening shim."""
import json

import pytest


@pytest.mark.gpu
def test_generate_save_and_decode_voxels(lf, numpy, gpu_available, tmp_path):
    if not gpu_available:
        pytest.skip("CUDA device unavailable")
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    point = numpy.zeros(1, dtype=[(name, "<f4") for name in fields])
    point["opacity"], point["rot_0"] = 2, 1
    for name in ("scale_0", "scale_1", "scale_2"):
        point[name] = numpy.log(.02)
    source = tmp_path / "small.ply"
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
    header += "".join(f"property float {name}\n" for name in fields) + "end_header\n"
    source.write_bytes(header.encode() + point.tobytes())
    settings = lf.io.VoxelSettings()
    settings.voxel_size = .05
    volume = lf.io.generate_voxels(source, settings)
    assert 0 < volume.occupied_count < 1000
    assert volume.voxel_size == pytest.approx(.05)
    path = tmp_path / "small.voxel.json"
    binary = path.with_suffix(".bin")
    lf.io.save_voxels(volume, path)
    metadata = json.loads(path.read_text())
    words = numpy.frombuffer(binary.read_bytes(), dtype="<u4")
    nodes, leaves = metadata["nodeCount"], metadata["leafDataCount"]
    assert metadata["version"] == "1.1" and metadata["leafSize"] == 4
    assert metadata["voxelResolution"] == pytest.approx(volume.voxel_size)
    assert len(words) == nodes + leaves
    assert leaves == 2 * metadata["numMixedLeaves"]

    # No uniform solid nodes: independently count the raw little-endian leaf bits.
    assert nodes == metadata["numInteriorNodes"] + metadata["numMixedLeaves"]
    assert metadata["numMixedLeaves"] > 0
    assert all(int(word) != 0xff000000 for word in words[:nodes])
    assert sum(int(word).bit_count() for word in words[nodes:]) == volume.occupied_count
    original = (path.read_bytes(), binary.read_bytes())
    with pytest.raises(Exception, match="overwrite"):
        lf.io.save_voxels(volume, path)
    with pytest.raises(Exception):
        lf.io.save_voxels(volume, path, overwrite=True, progress=lambda *_: False)
    assert (path.read_bytes(), binary.read_bytes()) == original
    lf.io.save_voxels(volume, path, overwrite=True)
    assert (path.read_bytes(), binary.read_bytes()) == original

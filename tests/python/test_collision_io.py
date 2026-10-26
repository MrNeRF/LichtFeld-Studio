"""Collision I/O is a computation API, not a panel-opening shim."""
import json

import pytest
from test_splat_data_color_range import _load_splat


@pytest.mark.gpu
def test_generate_save_and_decode_voxels(lf, numpy, gpu_available, tmp_path):
    if not gpu_available:
        pytest.skip("CUDA device unavailable")
    splats = _load_splat(lf, tmp_path)
    splats.scaling_raw.fill_(float(numpy.log(.02)))
    source = tmp_path / "small.ply"
    lf.io.save_ply(splats, source)
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

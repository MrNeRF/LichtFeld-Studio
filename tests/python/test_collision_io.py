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
    if lf.Tensor.ones([1], device="gpu").backend != "cuda":
        # Voxelization has CUDA kernels only; other tensor backends reject it with a translated error.
        with pytest.raises(lf.Error) as caught:
            lf.io.generate_voxels(source, settings)
        assert caught.value.message_key == "collision.cuda_required"
        return
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
    with pytest.raises(lf.Error, match="overwrite") as caught:
        lf.io.save_voxels(volume, path)
    assert caught.value.message_key == "collision.replace_required"
    assert caught.value.message_details == str(path)
    with pytest.raises(lf.Error) as caught:
        lf.io.save_voxels(volume, path, overwrite=True, transform=[1.])
    assert caught.value.message_key == "collision.export_transform"
    with pytest.raises(Exception):
        lf.io.save_voxels(volume, path, overwrite=True, progress=lambda *_: False)
    assert (path.read_bytes(), binary.read_bytes()) == original
    lf.io.save_voxels(volume, path, overwrite=True)
    assert (path.read_bytes(), binary.read_bytes()) == original


@pytest.mark.parametrize("kind,key", [
    ("missing", "collision.source_load_failed"),
    ("dataset", "collision.source_unsupported"),
    pytest.param("points", "collision.source_unsupported", marks=pytest.mark.gpu),
])
def test_source_errors_preserve_translation_metadata(lf, tmp_path, kind, key, request):
    if kind == "points" and not request.getfixturevalue("gpu_available"):
        pytest.skip("CUDA device unavailable")
    path = tmp_path / ("transforms.json" if kind == "dataset" else "source.ply")
    if kind == "dataset":
        path.write_text("{}")
    elif kind == "points":
        path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
    with pytest.raises(lf.Error) as caught:
        lf.io.generate_voxels(path)
    assert caught.value.message_key == key
    assert caught.value.message_details == str(path)
    assert caught.value.details


def test_invalid_collision_uuid_preserves_translation_metadata(lf, tmp_path):
    with pytest.raises(lf.Error) as caught:
        lf.scene.export_collision("invalid-uuid", str(tmp_path / "output.voxel.json"))
    assert caught.value.message_key == "collision.select_export"


def test_headless_collision_authoring_requires_gui(lf):
    with pytest.raises(lf.Error) as caught:
        lf.scene.generate_collision()
    assert caught.value.message_key == "collision.gui_required"

"""Collision I/O is a computation API, not a panel-opening shim."""
import pytest
import lichtfeld as lf


def test_voxel_api_is_public_and_volume_is_immutable():
    settings = lf.io.VoxelSettings()
    assert settings.voxel_size > 0
    assert settings.cleanup
    assert callable(lf.io.generate_voxels)
    assert callable(lf.io.save_voxels)
    with pytest.raises(TypeError):
        lf.io.VoxelVolume()


def test_generation_failure_does_not_publish_files(tmp_path):
    path = tmp_path / "missing.ply"
    with pytest.raises(Exception, match="(?i)(path|exist|found)"):
        lf.io.generate_voxels(path)
    assert list(tmp_path.iterdir()) == []

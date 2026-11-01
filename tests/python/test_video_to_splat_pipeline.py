# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Orchestration tests for the video-to-splat pipeline with every side effect injected."""

import json
import sqlite3
import struct
import sys
import threading
from pathlib import Path
from subprocess import CompletedProcess

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import pipeline, toolchain  # noqa: E402


class Fakes:
    """Injected side effects. Records calls, lets tests script outcomes."""

    def __init__(self, project: Path, frames: int = 50, registered: int = 45, colmap_rc: int = 0):
        self.project = project
        self.frames = frames
        self.registered = registered
        self.colmap_rc = colmap_rc
        self.calls: list[str] = []
        self.commands: list[list[str]] = []
        self.progress: list[tuple] = []
        self.cancel = threading.Event()
        self.confirmed_weak = True
        self.extract_params = None
        self.create_args = None
        self.gpu_features_ok = True
        self.gpu_matches_ok = True

    def resolve_colmap(self, on_progress):
        self.calls.append("toolchain")
        on_progress("found colmap")
        return toolchain.Resolved(Path("colmap.exe"), (4, 2, 1))

    def probe(self, path):
        return {"width": 5760, "height": 2880, "fps": 30.0, "duration": 10.0, "rotation": 0, "spherical": True}

    def extract(self, params, on_progress, should_cancel):
        self.calls.append("extract")
        self.extract_params = params
        images = params.dataset / "images"
        images.mkdir(parents=True, exist_ok=True)
        for i in range(self.frames):
            (images / f"frame_{i:06d}.jpg").write_bytes(b"\xff\xd8")
        # The real extractor drops its report next to the frames; COLMAP must never see it.
        (images / "extraction_metadata.json").write_text("{}")
        on_progress(self.frames, self.frames, 0)
        return {"outcome": "completed", "error": ""}

    def run(self, cmd, on_output, should_cancel):
        self.calls.append(cmd[1])
        self.commands.append(list(cmd))
        database = Path(cmd[cmd.index("--database_path") + 1])
        if cmd[1] == "feature_extractor":
            gpu = cmd[cmd.index("--FeatureExtraction.use_gpu") + 1] == "1"
            rows = 500 if (self.gpu_features_ok or not gpu) else 0
            write_colmap_database(database, descriptor_rows=[rows] * self.frames)
        if cmd[1] == "sequential_matcher":
            gpu = cmd[cmd.index("--FeatureMatching.use_gpu") + 1] == "1"
            rows = 40 if (self.gpu_matches_ok or not gpu) else 0
            write_colmap_database(database, geometry_rows=[rows] * (self.frames - 1))
        if cmd[1] == "global_mapper":
            model = Path(cmd[cmd.index("--output_path") + 1]) / "0"
            model.mkdir(parents=True, exist_ok=True)
            (model / "images.bin").write_bytes(struct.pack("<Q", self.registered))
            (model / "cameras.bin").write_bytes(struct.pack("<Q", 1))
            (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))
        on_output(f"{cmd[1]} running")
        return CompletedProcess(cmd, self.colmap_rc, "", "")

    def create_project(self, project, dataset, images_folder):
        self.calls.append("create_project")
        self.create_args = (project, dataset, images_folder)

    def train(self, mode):
        self.calls.append("train")
        self.train_mode = mode

    def confirm_weak(self, ratio, registered, total):
        self.calls.append("confirm_weak")
        return self.confirmed_weak

    def hooks(self) -> pipeline.Hooks:
        return pipeline.Hooks(
            resolve_colmap=self.resolve_colmap, probe=self.probe, extract=self.extract, run=self.run,
            create_project=self.create_project, train=self.train, confirm_weak=self.confirm_weak,
            free_bytes=lambda path: 100 * 1024 ** 3,
        )


def write_colmap_database(path: Path, descriptor_rows=(), geometry_rows=()) -> None:
    """Create or extend a database with the tables the pipeline's stage checks read."""
    db = sqlite3.connect(path)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS descriptors "
                   "(image_id INTEGER PRIMARY KEY, type INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
        db.execute("CREATE TABLE IF NOT EXISTS two_view_geometries "
                   "(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        if descriptor_rows:
            db.execute("DELETE FROM descriptors")
        for idx, rows in enumerate(descriptor_rows, start=1):
            db.execute("INSERT INTO descriptors VALUES (?, 0, ?, 128, NULL)", (idx, rows))
        if geometry_rows:
            db.execute("DELETE FROM two_view_geometries")
        for idx, rows in enumerate(geometry_rows, start=1):
            db.execute("INSERT INTO two_view_geometries VALUES (?, ?, 2, NULL)", (idx, rows))
        db.commit()
    finally:
        db.close()  # Windows cannot unlink a database that is still open


def gpu_flags(fakes: "Fakes") -> list[str]:
    """use_gpu value of every COLMAP feature/matching command, in call order."""
    flags = []
    for cmd in fakes.commands:
        for option in ("--FeatureExtraction.use_gpu", "--FeatureMatching.use_gpu"):
            if option in cmd:
                flags.append(cmd[cmd.index(option) + 1])
    return flags


def make_request(project: Path, video: Path) -> pipeline.Request:
    return pipeline.Request(video=video, project=project, mode_override="auto", preset="balanced")


def work_dir(project: Path) -> Path:
    return project.parent / project.stem


@pytest.fixture
def clip(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    return video


def test_full_run_order_and_state(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    state = pipeline.run(make_request(project, clip), fakes.hooks(),
                         on_progress=lambda *a: fakes.progress.append(a), should_cancel=fakes.cancel.is_set)
    assert fakes.calls == ["toolchain", "extract", "feature_extractor", "sequential_matcher", "global_mapper",
                           "create_project", "train"]
    saved = json.loads((work_dir(project) / "pipeline.json").read_text())
    assert [s["name"] for s in saved["completed"]] == pipeline.STAGE_NAMES
    assert saved["mode"] == "equirectangular"
    assert saved["registered"] == 45 and saved["frames"] == 50
    assert state.registered == 45 and state.frames == 50
    assert fakes.progress[0][0] == "toolchain"
    assert fakes.progress[-1][0] == "train" and fakes.progress[-1][1] == 1.0
    # extraction used the balanced preset against a 5760 px wide source: 2 fps, scaled to 4096 px
    assert fakes.extract_params.fps == 2.0
    assert fakes.extract_params.scale == pytest.approx(4096 / 5760)
    assert fakes.extract_params.dataset == tmp_path / "proj" / "dataset"
    assert fakes.create_args == (project, tmp_path / "proj" / "dataset", "images")
    assert fakes.train_mode == "equirectangular"


def test_rerun_skips_completed_stages(tmp_path, clip):
    project = tmp_path / "proj.licht"
    first = Fakes(project)
    pipeline.run(make_request(project, clip), first.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    # Truncate state so only toolchain..match are complete, as if map crashed.
    saved = json.loads((work_dir(project) / "pipeline.json").read_text())
    saved["completed"] = saved["completed"][:4]
    (work_dir(project) / "pipeline.json").write_text(json.dumps(saved))
    second = Fakes(project)
    pipeline.run(make_request(project, clip), second.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    # COLMAP must still be located because global_mapper has to run, but frames and the database are kept.
    assert second.calls == ["toolchain", "global_mapper", "create_project", "train"]


def test_changed_preset_invalidates_resume(tmp_path, clip):
    project = tmp_path / "proj.licht"
    first = Fakes(project)
    pipeline.run(make_request(project, clip), first.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    second = Fakes(project)
    request = pipeline.Request(video=clip, project=project, mode_override="auto", preset="high")
    pipeline.run(request, second.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert second.calls[0] == "toolchain" and "extract" in second.calls


def test_corrupt_state_file_starts_over(tmp_path, clip):
    project = tmp_path / "proj.licht"
    work_dir(project).mkdir()
    (work_dir(project) / "pipeline.json").write_text("{not json")
    fakes = Fakes(project)
    pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert fakes.calls[0] == "toolchain" and "extract" in fakes.calls


def test_too_few_frames_stops(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project, frames=7)
    with pytest.raises(pipeline.PipelineError) as err:
        pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "7" in str(err.value)
    assert "feature_extractor" not in fakes.calls
    saved = json.loads((work_dir(project) / "pipeline.json").read_text())
    assert [s["name"] for s in saved["completed"]] == ["toolchain"]


def test_extraction_failure_is_reported(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    fakes.extract = lambda params, on_progress, should_cancel: {"outcome": "failed", "error": "codec unsupported"}
    with pytest.raises(pipeline.PipelineError) as err:
        pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "codec unsupported" in str(err.value)


def test_extraction_cancel_propagates(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    fakes.extract = lambda params, on_progress, should_cancel: {"outcome": "cancelled", "error": ""}
    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)


def test_weak_reconstruction_asks_and_can_stop(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project, frames=50, registered=20)  # 40 percent
    fakes.confirmed_weak = False
    with pytest.raises(pipeline.PipelineStopped):
        pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "confirm_weak" in fakes.calls and "create_project" not in fakes.calls
    saved = json.loads((work_dir(project) / "pipeline.json").read_text())
    assert saved["registered"] == 20


def test_weak_reconstruction_can_continue(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project, frames=50, registered=20)
    fakes.confirmed_weak = True
    pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert fakes.calls[-3:] == ["confirm_weak", "create_project", "train"]


def test_strong_reconstruction_does_not_ask(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project, frames=50, registered=45)
    pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "confirm_weak" not in fakes.calls


def test_largest_model_is_moved_to_sparse_zero(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    original_run = fakes.run

    def run(cmd, on_output, should_cancel):
        result = original_run(cmd, on_output, should_cancel)
        if cmd[1] == "global_mapper":
            sparse = Path(cmd[cmd.index("--output_path") + 1])
            (sparse / "0" / "images.bin").write_bytes(struct.pack("<Q", 3))
            big = sparse / "1"
            big.mkdir()
            (big / "images.bin").write_bytes(struct.pack("<Q", 44))
            (big / "cameras.bin").write_bytes(struct.pack("<Q", 1))
            (big / "points3D.bin").write_bytes(struct.pack("<Q", 0))
        return result

    fakes.run = run
    state = pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None,
                         should_cancel=lambda: False)
    assert state.registered == 44
    sparse = work_dir(project) / "dataset" / "sparse"
    assert Path(state.model_dir) == sparse / "0"
    assert struct.unpack("<Q", (sparse / "0" / "images.bin").read_bytes()[:8])[0] == 44
    assert not (sparse / "1").exists()


def test_no_model_is_an_error(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    original_run = fakes.run

    def run(cmd, on_output, should_cancel):
        if cmd[1] == "global_mapper":
            fakes.calls.append(cmd[1])
            return CompletedProcess(cmd, 0, "", "")
        return original_run(cmd, on_output, should_cancel)

    fakes.run = run
    with pytest.raises(pipeline.PipelineError) as err:
        pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "no reconstruction" in str(err.value).lower()


def test_colmap_failure_raises_with_tail(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project, colmap_rc=1)
    with pytest.raises(pipeline.PipelineError) as err:
        pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "feature_extractor" in str(err.value)
    assert "feature_extractor running" in str(err.value)


def test_cancel_before_map_keeps_state(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    original_run = fakes.run

    def run(cmd, on_output, should_cancel):
        if cmd[1] == "global_mapper":
            fakes.cancel.set()
            raise pipeline.ProcessCancelled("killed")
        return original_run(cmd, on_output, should_cancel)

    fakes.run = run
    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None,
                     should_cancel=fakes.cancel.is_set)
    saved = json.loads((work_dir(project) / "pipeline.json").read_text())
    assert [s["name"] for s in saved["completed"]] == ["toolchain", "extract", "features", "match"]


def test_cancel_between_stages_is_honoured(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    original_extract = fakes.extract

    def extract(params, on_progress, should_cancel):
        result = original_extract(params, on_progress, should_cancel)
        fakes.cancel.set()
        return result

    fakes.extract = extract
    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None,
                     should_cancel=fakes.cancel.is_set)
    assert "feature_extractor" not in fakes.calls
    saved = json.loads((work_dir(project) / "pipeline.json").read_text())
    assert [s["name"] for s in saved["completed"]] == ["toolchain", "extract"]


def test_low_disk_space_warns(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    hooks = fakes.hooks()
    hooks = pipeline.Hooks(**{**hooks.__dict__, "free_bytes": lambda p: 2 * 1024 ** 3})
    messages = []
    pipeline.run(make_request(project, clip), hooks, on_progress=lambda stage, frac, msg, tail: messages.append(msg),
                 should_cancel=lambda: False)
    assert any("GB" in m for m in messages)


def test_progress_carries_colmap_log_tail(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    tails = []
    pipeline.run(make_request(project, clip), fakes.hooks(),
                 on_progress=lambda stage, frac, msg, tail: tails.append((stage, tail)), should_cancel=lambda: False)
    assert any(stage == "features" and "feature_extractor running" in tail for stage, tail in tails)


def test_request_fingerprint_tracks_inputs(tmp_path, clip):
    project = tmp_path / "proj.licht"
    base = make_request(project, clip)
    assert base.fingerprint() == make_request(project, clip).fingerprint()
    assert base.fingerprint() != pipeline.Request(video=clip, project=project, preset="fast").fingerprint()
    assert base.fingerprint() != pipeline.Request(video=clip, project=project, mode_override="off").fingerprint()
    assert base.fingerprint() != pipeline.Request(video=clip, project=project, use_gpu=False).fingerprint()
    assert base.work_dir == tmp_path / "proj"
    assert base.dataset == tmp_path / "proj" / "dataset"


def test_extraction_report_is_moved_out_of_the_images_folder(tmp_path, clip):
    # COLMAP reads every file in --image_path; its CPU feature extractor crashes on a JSON file.
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    dataset = work_dir(project) / "dataset"
    assert not (dataset / "images" / "extraction_metadata.json").exists()
    assert (dataset / "extraction_metadata.json").read_text() == "{}"


def test_gpu_feature_extraction_without_features_falls_back_to_cpu(tmp_path, clip):
    # The official CUDA build has no kernels for older GPUs: SIFT fails on every image but COLMAP exits 0.
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    fakes.gpu_features_ok = False
    messages = []
    pipeline.run(make_request(project, clip), fakes.hooks(),
                 on_progress=lambda stage, frac, msg, tail: messages.append((stage, msg)), should_cancel=lambda: False)
    assert fakes.calls == ["toolchain", "extract", "feature_extractor", "feature_extractor", "sequential_matcher",
                           "global_mapper", "create_project", "train"]
    assert gpu_flags(fakes) == ["1", "0", "0"]
    assert any(stage == "features" and "CPU" in msg for stage, msg in messages)
    saved = json.loads((work_dir(project) / "pipeline.json").read_text())
    assert saved["use_gpu"] is False


def test_gpu_matching_without_pairs_falls_back_to_cpu(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    fakes.gpu_matches_ok = False
    pipeline.run(make_request(project, clip), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert fakes.calls == ["toolchain", "extract", "feature_extractor", "sequential_matcher", "sequential_matcher",
                           "global_mapper", "create_project", "train"]
    assert gpu_flags(fakes) == ["1", "1", "0"]


def test_cpu_fallback_persists_across_resume(tmp_path, clip):
    project = tmp_path / "proj.licht"
    first = Fakes(project)
    first.gpu_features_ok = False
    original_run = first.run

    def run(cmd, on_output, should_cancel):
        if cmd[1] == "global_mapper":
            first.cancel.set()
            raise pipeline.ProcessCancelled("killed")
        return original_run(cmd, on_output, should_cancel)

    first.run = run
    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.run(make_request(project, clip), first.hooks(), on_progress=lambda *a: None,
                     should_cancel=first.cancel.is_set)
    second = Fakes(project)
    pipeline.run(make_request(project, clip), second.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert second.calls == ["toolchain", "global_mapper", "create_project", "train"]
    # A resumed run must not go back to the GPU for a COLMAP stage that has to be redone.
    third = Fakes(project)
    saved = json.loads((work_dir(project) / "pipeline.json").read_text())
    saved["completed"] = saved["completed"][:3]  # toolchain, extract, features
    (work_dir(project) / "pipeline.json").write_text(json.dumps(saved))
    pipeline.run(make_request(project, clip), third.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert gpu_flags(third) == ["0"]


def test_no_features_on_cpu_is_an_error(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    original_run = fakes.run

    def run(cmd, on_output, should_cancel):
        result = original_run(cmd, on_output, should_cancel)
        if cmd[1] == "feature_extractor":
            write_colmap_database(Path(cmd[cmd.index("--database_path") + 1]), descriptor_rows=[0] * fakes.frames)
        return result

    fakes.run = run
    request = pipeline.Request(video=clip, project=project, use_gpu=False)
    with pytest.raises(pipeline.PipelineError) as err:
        pipeline.run(request, fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "no features" in str(err.value).lower()
    assert fakes.calls.count("feature_extractor") == 1
    assert "sequential_matcher" not in fakes.calls


def test_no_verified_pairs_on_cpu_is_an_error(tmp_path, clip):
    project = tmp_path / "proj.licht"
    fakes = Fakes(project)
    original_run = fakes.run

    def run(cmd, on_output, should_cancel):
        result = original_run(cmd, on_output, should_cancel)
        if cmd[1] == "sequential_matcher":
            write_colmap_database(Path(cmd[cmd.index("--database_path") + 1]), geometry_rows=[0] * 5)
        return result

    fakes.run = run
    request = pipeline.Request(video=clip, project=project, use_gpu=False)
    with pytest.raises(pipeline.PipelineError) as err:
        pipeline.run(request, fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "no matches" in str(err.value).lower()
    assert "global_mapper" not in fakes.calls

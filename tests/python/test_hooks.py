# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for lichtfeld training hooks."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(autouse=True)
def clear_training_hooks(lf):
    """Keep process-global decorator registrations isolated per test."""
    yield
    lf._clear_training_hooks()


class TestHookDecorators:
    """Tests for hook decorator existence and registration."""

    def test_on_training_start_exists(self, lf):
        """Test on_training_start decorator exists."""
        assert hasattr(lf, "on_training_start")
        assert callable(lf.on_training_start)

    def test_on_iteration_start_exists(self, lf):
        """Test on_iteration_start decorator exists."""
        assert hasattr(lf, "on_iteration_start")
        assert callable(lf.on_iteration_start)

    def test_on_post_step_exists(self, lf):
        """Test on_post_step decorator exists."""
        assert hasattr(lf, "on_post_step")
        assert callable(lf.on_post_step)

    def test_on_pre_optimizer_step_exists(self, lf):
        """Test on_pre_optimizer_step decorator exists."""
        assert hasattr(lf, "on_pre_optimizer_step")
        assert callable(lf.on_pre_optimizer_step)

    def test_on_training_end_exists(self, lf):
        """Test on_training_end decorator exists."""
        assert hasattr(lf, "on_training_end")
        assert callable(lf.on_training_end)

    def test_decorator_returns_function(self, lf):
        """Test decorators return the decorated function."""

        @lf.on_training_start
        def my_callback(ctx):
            pass

        assert my_callback is not None
        assert callable(my_callback)

    def test_decorator_with_lambda(self, lf):
        """Test decorators work with lambdas."""
        callback = lf.on_iteration_start(lambda ctx: None)
        assert callback is not None


class TestHookEnum:
    """Tests for Hook enum."""

    def test_hook_enum_exists(self, lf):
        """Test Hook enum exists."""
        assert hasattr(lf, "Hook")

    def test_hook_enum_values(self, lf):
        """Test Hook enum has expected values."""
        assert hasattr(lf.Hook, "training_start")
        assert hasattr(lf.Hook, "iteration_start")
        assert hasattr(lf.Hook, "pre_optimizer_step")
        assert hasattr(lf.Hook, "post_step")
        assert hasattr(lf.Hook, "training_end")


class TestSessionClass:
    """Tests for Session class (training control)."""

    def test_session_class_exists(self, lf):
        """Test Session class exists."""
        assert hasattr(lf, "Session")

    def test_session_instantiation(self, lf):
        """Test Session can be instantiated."""
        session = lf.Session()
        assert session is not None

    def test_session_has_control_methods(self, lf):
        """Test Session has training control methods."""
        session = lf.Session()
        assert hasattr(session, "pause")
        assert hasattr(session, "resume")
        assert hasattr(session, "request_stop")
        assert hasattr(session, "optimizer")
        assert hasattr(session, "model")


class TestContextView:
    """Tests for Context view class."""

    def test_context_class_exists(self, lf):
        """Test Context class exists."""
        assert hasattr(lf, "Context")

    def test_context_instantiation(self, lf):
        """Test Context can be instantiated."""
        ctx = lf.Context()
        assert ctx is not None

    def test_context_properties(self, lf):
        """Test Context has expected properties."""
        ctx = lf.Context()
        # These should exist and return defaults when no training
        assert hasattr(ctx, "iteration")
        assert hasattr(ctx, "max_iterations")
        assert hasattr(ctx, "loss")
        assert hasattr(ctx, "num_gaussians")
        assert hasattr(ctx, "is_refining")
        assert hasattr(ctx, "is_training")
        assert hasattr(ctx, "is_paused")
        assert hasattr(ctx, "phase")
        assert hasattr(ctx, "strategy")

    def test_context_values_without_training(self, lf):
        """Test Context returns sensible defaults without training."""
        ctx = lf.Context()
        ctx.refresh()
        assert ctx.iteration >= 0
        assert ctx.loss >= 0.0 or ctx.loss == 0.0
        assert ctx.is_training is False


class TestGaussiansView:
    """Tests for Gaussians view class."""

    def test_gaussians_class_exists(self, lf):
        """Test Gaussians class exists."""
        assert hasattr(lf, "Gaussians")

    def test_gaussians_instantiation(self, lf):
        """Test Gaussians can be instantiated."""
        g = lf.Gaussians()
        assert g is not None

    def test_gaussians_properties(self, lf):
        """Test Gaussians has expected properties."""
        g = lf.Gaussians()
        assert hasattr(g, "count")
        assert hasattr(g, "sh_degree")
        assert hasattr(g, "max_sh_degree")

    def test_gaussians_values_without_training(self, lf):
        """Test Gaussians returns 0 without training."""
        g = lf.Gaussians()
        assert g.count == 0
        assert g.sh_degree == 0


class TestContextFunction:
    """Tests for context() convenience function."""

    def test_context_function_exists(self, lf):
        """Test context() function exists."""
        assert hasattr(lf, "context")
        assert callable(lf.context)

    def test_context_function_returns_context(self, lf):
        """Test context() returns Context object."""
        ctx = lf.context()
        assert isinstance(ctx, lf.Context)


class TestGaussiansFunction:
    """Tests for gaussians() convenience function."""

    def test_gaussians_function_exists(self, lf):
        """Test gaussians() function exists."""
        assert hasattr(lf, "gaussians")
        assert callable(lf.gaussians)

    def test_gaussians_function_returns_gaussians(self, lf):
        """Test gaussians() returns Gaussians object."""
        g = lf.gaussians()
        assert isinstance(g, lf.Gaussians)


class TestSessionFunction:
    """Tests for session() convenience function."""

    def test_session_function_exists(self, lf):
        """Test session() function exists."""
        assert hasattr(lf, "session")
        assert callable(lf.session)

    def test_session_function_returns_pyession(self, lf):
        """Test session() returns session object with expected methods."""
        s = lf.session()
        assert hasattr(s, "pause")
        assert hasattr(s, "resume")
        assert hasattr(s, "request_stop")
        assert hasattr(s, "optimizer")
        assert hasattr(s, "model")


class TestOptimizerView:
    """Tests for Optimizer view class."""

    def test_optimizer_class_exists(self, lf):
        """Test Optimizer class exists."""
        assert hasattr(lf, "Optimizer")

    def test_optimizer_methods(self, lf):
        """Test Optimizer has expected methods."""
        opt = lf.Optimizer()
        assert hasattr(opt, "scale_lr")
        assert hasattr(opt, "set_lr")
        assert hasattr(opt, "get_lr")


class TestModelView:
    """Tests for Model view class."""

    def test_model_class_exists(self, lf):
        """Test Model class exists."""
        assert hasattr(lf, "Model")

    def test_model_methods(self, lf):
        """Test Model has expected methods."""
        model = lf.Model()
        assert hasattr(model, "clamp")
        assert hasattr(model, "scale")
        assert hasattr(model, "set")


class TestFrameCallback:
    """Tests for frame callback for animations."""

    def test_on_frame_exists(self, lf):
        """Test on_frame function exists."""
        assert hasattr(lf, "on_frame")
        assert callable(lf.on_frame)

    def test_stop_animation_exists(self, lf):
        """Test stop_animation function exists."""
        assert hasattr(lf, "stop_animation")
        assert callable(lf.stop_animation)


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
    for name in ("PYTHONHOME", "PYTHONPATH"):
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

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Strategy edits must not replace the settings of an existing training run."""

import pytest

from test_mcp_dataset_load_training import _dataset, _wait, endpoint  # noqa: F401
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("initial,target", [("mcmc", "mrnf"), ("mrnf", "mcmc")])
@pytest.mark.parametrize("route,paused", [("mcp", True), ("mcp", False),
                                         ("python_method", True), ("python_property", True)])
def test_strategy_update_preserves_training(tmp_path, endpoint, initial, target, route, paused):
    def tool(name, **arguments):
        return _tool(endpoint, name, arguments)

    def state():
        return tool("training_get_state")

    assert tool("scene_load_dataset", path=_dataset(tmp_path / "source", 60),
                images_folder="images", min_track_length=0,
                output_path=str(tmp_path / "output"))["success"]
    tool("runtime_job_wait", job_id="import.dataset", until="inactive", timeout_ms=90000)
    # Changing strategy before the first start remains supported.
    assert tool("training_params_set", values={"strategy": initial, "iterations": 1000000,
                                              "max_cap": 2000, "sh_degree": 1})["success"]
    tool("training_start")
    try:
        _wait(lambda: state()["iteration"] > 10)
        if paused:
            tool("runtime_job_control", job_id="training.main", action="pause")
            _wait(lambda: state()["is_paused"])
        # Re-selecting the current strategy and editing another field are valid.
        assert tool("training_params_set", values={"strategy": initial, "means_lr": 0.00012})["success"]
        localized_error = "Strategy change requires a training reset (localized)."
        tool("editor_run", code=(
            "import lichtfeld as lf\n"
            f"lf.ui.loc_set('training.reset_before_strategy_change', {localized_error!r})"),
            show_console=False, wait_for_completion=True, wait_for_output=True, timeout_ms=30000)
        before = tool("training_params_get")
        iteration = state()["iteration"]
        if route == "mcp":
            response = _call(endpoint, "tools/call", {"name": "training_params_set", "arguments": {
                "values": {"strategy": target, "means_lr": 0.00009}}})
            assert response["isError"], response
            assert localized_error in str(response), response
        else:
            expression = (f"p.set_strategy({target!r})" if route == "python_method"
                          else f"p.set('strategy', {target!r})")
            result = tool("editor_run", code=(
                "import lichtfeld as lf\np = lf.optimization_params()\n"
                "try:\n    " + expression + "\n"
                "except (ValueError, RuntimeError) as error:\n    print('STRATEGY_REJECTED', error)\n"
                "else:\n    print('STRATEGY_ACCEPTED')\n"),
                show_console=False, wait_for_completion=True, wait_for_output=True, timeout_ms=30000)
            assert "STRATEGY_REJECTED" in str(result), result
            assert localized_error in str(result), result
            assert "STRATEGY_ACCEPTED" not in str(result), result
        assert tool("training_params_get") == before, "rejected edit changed preset values"
        assert state()["strategy"] == initial
        if paused:
            assert state()["is_paused"]
            assert state()["iteration"] == iteration
            tool("runtime_job_control", job_id="training.main", action="resume")
        _wait(lambda: state()["iteration"] >= iteration + 50)
        assert state()["max_iterations"] == 1000000
        assert state()["strategy"] == initial
        assert tool("ui_modal_get").get("title") != "Training Failed"
    finally:
        if tool("runtime_job_describe", job_id="training.main")["actions"]["cancel"]:
            tool("runtime_job_control", job_id="training.main", action="cancel")
        tool("runtime_job_wait", job_id="training.main", until="inactive", timeout_ms=90000)


def test_paused_iteration_limit_stays_ahead_of_completed_step(tmp_path, endpoint):
    def tool(name, **arguments):
        return _tool(endpoint, name, arguments)

    def state():
        return tool("training_get_state")

    assert tool("scene_load_dataset", path=_dataset(tmp_path / "source", 60),
                images_folder="images", min_track_length=0,
                output_path=str(tmp_path / "output"))["success"]
    tool("runtime_job_wait", job_id="import.dataset", until="inactive", timeout_ms=90000)
    assert tool("training_params_set", values={"strategy": "mcmc", "iterations": 1000000,
                                              "max_cap": 2000, "enable_eval": False})["success"]
    tool("training_start")
    try:
        _wait(lambda: state()["iteration"] > 10)
        tool("runtime_job_control", job_id="training.main", action="pause")
        _wait(lambda: state()["is_paused"])
        completed = state()["iteration"]
        original = tool("training_params_get")

        for invalid in (completed - 1, completed):
            response = _call(endpoint, "tools/call", {"name": "training_params_set",
                                                       "arguments": {"values": {"iterations": invalid}}})
            assert response["isError"], response
            assert tool("training_params_get") == original
            assert state()["is_paused"]
            assert state()["iteration"] == completed

        direct = tool("editor_run", code=(
            "import lichtfeld as lf\n"
            "params = lf.optimization_params()\n"
            f"try:\n    params.iterations = {completed - 1}\n"
            "except (ValueError, RuntimeError) as error:\n"
            "    print('LIMIT_REJECTED', error)\n"
            "else:\n    print('LIMIT_ACCEPTED')\n"),
            show_console=False, wait_for_completion=True, wait_for_output=True,
            timeout_ms=30000)
        assert "LIMIT_REJECTED" in str(direct), direct
        assert "LIMIT_ACCEPTED" not in str(direct), direct
        assert tool("training_params_get") == original

        target = completed + 30
        assert tool("training_params_set", values={"iterations": target})["success"]
        assert next(p["value"] for p in tool("training_params_get")["properties"]
                    if p["id"] == "iterations") == target
        tool("runtime_job_control", job_id="training.main", action="resume")
        _wait(lambda: not state()["is_running"] and not state()["is_paused"])
        assert state()["iteration"] == target
        assert state()["max_iterations"] == target
        assert tool("ui_modal_get")["title"] == "Training Complete"
    finally:
        if tool("runtime_job_describe", job_id="training.main")["actions"]["cancel"]:
            tool("runtime_job_control", job_id="training.main", action="cancel")
        tool("runtime_job_wait", job_id="training.main", until="inactive", timeout_ms=90000)

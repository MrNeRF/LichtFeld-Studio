# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""A strict CI gate must not report success when checks failed to execute."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_property_contracts", ROOT / "tools/check_property_contracts.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def enum_group():
    return {"properties": [{"id": "resolution", "type": "enum", "default": 2048,
                            "default_identifier": "medium",
                            "items": [{"identifier": "small", "value": 512},
                                      {"identifier": "medium", "value": 2048}]}]}


def test_source_only_checks_do_not_import_the_native_runtime_or_plugin(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lichtfeld", None)
    (tmp_path / "settings.py").write_text('''from lfs_plugins.props import EnumProperty
raise RuntimeError("This source must not execute")
mode = EnumProperty(items=[("ok", "OK", "")], default="missing")
''')
    findings = checker.source_findings([tmp_path])
    assert len(findings) == 1 and "is not in items" in findings[0]
    assert sys.modules["lichtfeld"] is None


def test_missing_source_roots_and_empty_checks_fail_closed(tmp_path):
    assert checker.source_findings([])
    assert checker.source_findings([tmp_path / "missing"])


def test_unparseable_source_is_not_a_successful_empty_audit(tmp_path):
    (tmp_path / "broken.py").write_text("value = (\n")
    assert "cannot check property declarations" in checker.source_findings([tmp_path])[0]


def test_source_check_honors_python_encoding_declarations(tmp_path):
    (tmp_path / "legacy.py").write_bytes(
        b'# coding: latin-1\nfrom lfs_plugins.props import EnumProperty\n'
        b'x = EnumProperty(items=[("caf\xe9", "Label", "")], default="caf\xe9")\n')
    assert checker.source_findings([tmp_path]) == []


def test_app_audit_resolves_relative_sdk_imports_without_guessing_plugin_local_modules(tmp_path):
    helper_spec = importlib.util.spec_from_file_location("relative_property_audit", ROOT / "src/python/lfs_plugins/property_contract.py")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    (tmp_path / "settings.py").write_text('from .props import EnumProperty as E\nx = E(items=[], default="bad")\n')
    assert helper.check_property_defaults(tmp_path) == []
    assert len(helper.check_property_defaults(tmp_path, package_prefix="lfs_plugins")) == 1


def test_valid_sparse_values_and_automatic_scalar_sentinels():
    group = enum_group()
    group["properties"].append({"id": "automatic_budget", "type": "int", "default": 0, "min": 1})
    assert checker.group_findings("render_settings", group) == []
    group["properties"][0]["items"].reverse()
    assert checker.group_findings("render_settings", group) == []


@pytest.mark.parametrize("change", [
    lambda p: p.pop("default"), lambda p: p.pop("default_identifier"),
    lambda p: p.update(default=1), lambda p: p.update(default=True),
    lambda p: p.update(default_identifier="small"),
    lambda p: p["items"].append({"identifier": "medium", "value": 4096}),
    lambda p: p["items"].append({"identifier": "other", "value": 2048}),
    lambda p: p.update(items=[]), lambda p: p["items"][0].update(identifier=""),
    lambda p: p["items"][0].update(value="512"),
])
def test_imprecise_built_in_defaults_fail_ci_instead_of_becoming_skips(change):
    group = enum_group()
    change(group["properties"][0])
    assert checker.group_findings("render_settings", group)


def test_missing_groups_enum_declarations_and_duplicate_properties_fail():
    assert checker.group_findings("dataset", {})
    assert checker.group_findings("render_settings", {"properties": [{"id": "x", "type": "float"}]})
    group = enum_group()
    group["properties"].append(dict(group["properties"][0]))
    assert checker.group_findings("render_settings", group)


def test_native_mode_requires_a_build_directory():
    with pytest.raises(SystemExit) as failure:
        checker.main(["--native"])
    assert failure.value.code == 2


def test_native_import_failure_is_an_error_not_a_skip(tmp_path, monkeypatch, capsys):
    (tmp_path / "src/python").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "lichtfeld", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LFS_HOME", "preserve-this-profile")
    assert checker.main(["--native", "--build-dir", str(tmp_path)]) == 1
    assert "could not complete" in capsys.readouterr().err
    import os
    assert os.environ["LFS_HOME"] == "preserve-this-profile"


def test_native_mode_rejects_a_different_installed_build(tmp_path, monkeypatch):
    (tmp_path / "src/python").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "lichtfeld", SimpleNamespace(__file__=str(tmp_path / "other/lichtfeld.pyd")))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert "different build" in checker.native_findings(tmp_path, "Release")[0]


@pytest.mark.parametrize("report, returncode", [
    (None, 0), ("<testsuites/>", 0),
    ('<testsuites><testcase classname="Other" status="run"/></testsuites>', 0),
    ('<testsuites><testcase classname="PropertyDefaults" status="run"><skipped/></testcase></testsuites>', 0),
    ('<testsuites><testcase classname="PropertyDefaults" status="notrun"/></testsuites>', 0),
    ('<testsuites><testcase classname="PropertyDefaults" status="run"><failure/></testcase></testsuites>', 0),
    ('<testsuites><testcase classname="PropertyDefaults" status="run"/></testsuites>', 1),
])
def test_cpu_checks_fail_for_absent_empty_failed_or_skipped_execution(tmp_path, monkeypatch, report, returncode):
    exe = tmp_path / "contracts.exe"
    exe.touch()

    def run(command, **kwargs):
        if report is not None:
            Path(command[2].split("xml:", 1)[1]).write_text(report)
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)
    assert checker.cpu_findings(exe)


def test_cpu_checks_require_execution_and_ignore_ambient_gtest_filters(tmp_path, monkeypatch):
    exe = tmp_path / "contracts.exe"
    exe.touch()
    monkeypatch.setenv("GTEST_FILTER", "Other.*")
    monkeypatch.setenv("GTEST_SHARD_INDEX", "1")

    def run(command, **kwargs):
        assert command[1] == "--gtest_filter=PropertyDefaults.*"
        assert not any(key.startswith("GTEST_") for key in kwargs["env"])
        Path(command[2].split("xml:", 1)[1]).write_text(
            '<testsuites><testcase classname="PropertyDefaults" status="run"/></testsuites>')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)
    assert checker.cpu_findings(exe) == []


def test_missing_executable_and_timeout_are_errors(tmp_path, monkeypatch):
    assert checker.main(["--cpu-tests", str(tmp_path / "missing")]) == 1
    exe = tmp_path / "contracts.exe"
    exe.touch()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("contracts", 60)

    monkeypatch.setattr(checker.subprocess, "run", timeout)
    assert checker.main(["--cpu-tests", str(exe)]) == 1


def test_ci_wiring_runs_light_and_native_contracts_without_gpu_jobs():
    light = (ROOT / ".github/workflows/property-contracts.yml").read_text()
    assert "--source-only" in light and "test_check_property_contracts.py" in light
    assert "cmake --build" not in light and "self-hosted" not in light
    for name in ("ubuntu", "windows"):
        workflow = (ROOT / f".github/workflows/{name}.yml").read_text()
        assert "tools/check_property_contracts.py --cpu-tests" in workflow
    ubuntu = (ROOT / ".github/workflows/ubuntu.yml").read_text()
    assert "tools/check_property_contracts.py --native --build-dir build" in ubuntu
    assert 'export LD_LIBRARY_PATH="$RUNNER_TEMP/cuda-driver-stub' in ubuntu

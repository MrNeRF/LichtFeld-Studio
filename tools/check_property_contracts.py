#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Strict CI checks of app defaults, without starting a GUI or running GPU work."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def source_findings(roots):
    if not roots:
        return ["No source roots requested"]
    # Load only the pure module. Importing lfs_plugins would initialize its
    # runtime package and could load a locally installed native extension.
    path = ROOT / "src/python/lfs_plugins/property_contract.py"
    spec = importlib.util.spec_from_file_location("property_contract_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    findings = []
    for root in roots:
        if not root.is_dir():
            findings.append(f"Missing source directory: {root}")
        else:
            package = "lfs_plugins" if root.resolve() == (ROOT / "src/python/lfs_plugins").resolve() else None
            findings.extend(f"{root}: {issue}" for issue in module.check_property_defaults(root, package_prefix=package))
    return findings


def group_findings(group_id, group):
    properties = group.get("properties", [])
    if not properties:
        return [f"{group_id}: missing or empty built-in property group"]
    findings = []
    if group_id in {"render_settings", "optimization", "optimization.session"} and not any(
            prop.get("type") == "enum" for prop in properties):
        findings.append(f"{group_id}: built-in enum declarations are missing")
    seen = set()
    for prop in properties:
        name = prop.get("id", "")
        prefix = f"{group_id}.{name}"
        if not name or name in seen:
            findings.append(f"{prefix}: empty or duplicate property id")
        seen.add(name)
        if prop.get("type") != "enum":
            continue
        items = prop.get("items", [])
        identifiers = [item.get("identifier") for item in items]
        values = [item.get("value") for item in items]
        if (not items or any(not isinstance(i, str) or not i for i in identifiers)
                or any(type(v) is not int for v in values)):
            findings.append(f"{prefix}: malformed enum items")
            continue
        if len(set(identifiers)) != len(identifiers) or len(set(values)) != len(values):
            findings.append(f"{prefix}: ambiguous enum values or identifiers")
        value = prop.get("default")
        if type(value) is not int or value not in values:
            findings.append(f"{prefix}: missing or invalid declared enum default")
        elif prop.get("default_identifier") != identifiers[values.index(value)]:
            findings.append(f"{prefix}: default identifier does not match enum value")
    return findings


def native_findings(build_dir, configuration):
    module_dir = build_dir / "src/python"
    if not module_dir.is_dir():
        return [f"Missing compiled Python module directory: {module_dir}"]
    sys.path[:0] = [str(ROOT / "src/python"), str(module_dir)]
    dll_handles = []
    if sys.platform == "win32":
        runtime = build_dir / "vcpkg_installed/x64-windows"
        directories = [build_dir, module_dir, runtime / ("debug/bin" if configuration == "Debug" else "bin"), runtime / "bin"]
        if os.environ.get("CUDA_PATH"):
            directories.append(Path(os.environ["CUDA_PATH"]) / "bin")
        for directory in directories:
            if directory.is_dir():
                dll_handles.append(os.add_dll_directory(str(directory)))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # No user projects/settings are needed for metadata introspection.
    with tempfile.TemporaryDirectory(prefix="lfs-property-audit-") as home:
        previous_home = os.environ.get("LFS_HOME")
        os.environ["LFS_HOME"] = home
        try:
            import lichtfeld as lf
            if Path(lf.__file__).resolve().parent != module_dir.resolve():
                return [f"Native module came from a different build: {lf.__file__}"]
            findings = []
            for group_id in ("render_settings", "optimization", "dataset"):
                findings.extend(group_findings(group_id, lf.ui.property_group_info(group_id)))
            # Session defaults are context-resolved. Do not compare their
            # numeric bounds to raw automatic/factory sentinel declarations.
            params = lf.optimization_params()
            enums = [p for p in lf.ui.property_group_info("optimization").get("properties", []) if p.get("type") == "enum"]
            resolved = {"properties": [params.prop_info(p["id"]) for p in enums]}
            findings.extend(group_findings("optimization.session", resolved))
            return findings
        finally:
            if previous_home is None:
                os.environ.pop("LFS_HOME", None)
            else:
                os.environ["LFS_HOME"] = previous_home


def cpu_findings(executable):
    if not executable.is_file():
        return [f"Missing CPU contract executable: {executable}"]
    with tempfile.TemporaryDirectory(prefix="lfs-property-cpu-") as directory:
        report = Path(directory) / "contracts.xml"
        env = {key: value for key, value in os.environ.items() if not key.startswith("GTEST_")}
        result = subprocess.run([str(executable.resolve()), "--gtest_filter=PropertyDefaults.*",
                                 f"--gtest_output=xml:{report}"], env=env, timeout=60,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            return [f"CPU contracts failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"]
        if not report.is_file():
            return ["CPU contracts did not produce the required test report"]
        cases = ET.parse(report).findall(".//testcase")
        if not cases or any(case.get("classname") != "PropertyDefaults" for case in cases):
            return ["CPU property contracts were not executed"]
        if any(case.find("skipped") is not None or case.find("failure") is not None
               or case.get("status") != "run" for case in cases):
            return ["CPU property contracts contain failed or skipped tests"]
        return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--source-only", action="store_true", help="Audit shipped Python declarations without importing plugins (default)")
    modes.add_argument("--native", action="store_true", help="Require the compiled app registry; import failures are errors, never skips")
    modes.add_argument("--cpu-tests", type=Path, help="Require all selected C++ CPU contracts to execute successfully")
    parser.add_argument("--build-dir", type=Path, help="Build containing the required native Python module")
    parser.add_argument("--configuration", choices=("Release", "Debug"), default="Release")
    args = parser.parse_args(argv)
    if args.native and args.build_dir is None:
        parser.error("--native requires --build-dir")
    try:
        if args.native:
            findings = native_findings(args.build_dir.resolve(), args.configuration)
        elif args.cpu_tests:
            findings = cpu_findings(args.cpu_tests)
        else:
            findings = source_findings([ROOT / "src/python/lfs_plugins", ROOT / "docs/plugins/examples"])
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError, TypeError,
            ET.ParseError, subprocess.TimeoutExpired) as error:
        findings = [f"Property contract check could not complete: {error}"]
    for finding in findings:
        print(f"ERROR: {finding}", file=sys.stderr)
    if findings:
        return 1
    print("Property contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Nested lf.run calls restore the outer script context."""

import json
from pathlib import Path


def test_nested_run_restores_outer_file_and_import_path(lf, tmp_path):
    outer_dir = tmp_path / "outer"
    inner_dir = tmp_path / "inner"
    outer_dir.mkdir()
    inner_dir.mkdir()
    (outer_dir / "outer_neighbor.txt").write_text("outer", encoding="utf-8")
    (outer_dir / "shared_probe.py").write_text("VALUE = 'outer'\n", encoding="utf-8")
    (inner_dir / "shared_probe.py").write_text("VALUE = 'inner'\n", encoding="utf-8")

    result_path = tmp_path / "result.json"
    inner_script = inner_dir / "inner.py"
    inner_script.write_text("inner_ran = True\n", encoding="utf-8")
    outer_script = outer_dir / "outer.py"
    outer_script.write_text(
        "import builtins, json, pathlib, sys\n"
        "import lichtfeld as lf\n"
        f"lf.run({str(inner_script)!r})\n"
        "sys.modules.pop('shared_probe', None)\n"
        "import shared_probe\n"
        "payload = {\n"
        "    'file': builtins.__file__,\n"
        "    'neighbor': (pathlib.Path(builtins.__file__).parent / 'outer_neighbor.txt').read_text(),\n"
        "    'module': shared_probe.VALUE,\n"
        "}\n"
        f"pathlib.Path({str(result_path)!r}).write_text(json.dumps(payload), encoding='utf-8')\n",
        encoding="utf-8",
    )

    lf.run(str(outer_script))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result == {
        "file": str(outer_script.resolve()),
        "neighbor": "outer",
        "module": "outer",
    }

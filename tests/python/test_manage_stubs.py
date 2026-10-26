# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for Python stub sync/check workflow."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANAGE_STUBS_PATH = PROJECT_ROOT / "src" / "python" / "scripts" / "manage_stubs.py"


def _load_manage_stubs():
    spec = spec_from_file_location("manage_stubs_under_test", MANAGE_STUBS_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_check_stubs_ignores_volatile_build_info(tmp_path):
    manage_stubs = _load_manage_stubs()
    generated = tmp_path / "generated"
    committed = tmp_path / "committed"

    (generated / "lichtfeld").mkdir(parents=True)
    (committed / "lichtfeld").mkdir(parents=True)

    (generated / "lichtfeld" / "__init__.pyi").write_text("value: int\n")
    (committed / "lichtfeld" / "__init__.pyi").write_text("value: int\n")
    (generated / "lichtfeld" / "build_info.pyi").write_text("commit: str = 'new'\n")
    (committed / "lichtfeld" / "build_info.pyi").write_text("commit: str = 'old'\n")

    assert manage_stubs.check_stubs(generated, committed) == 0


def test_sync_stubs_skips_and_removes_volatile_build_info(tmp_path):
    manage_stubs = _load_manage_stubs()
    generated = tmp_path / "generated"
    committed = tmp_path / "committed"

    (generated / "lichtfeld").mkdir(parents=True)
    (committed / "lichtfeld").mkdir(parents=True)

    (generated / "lichtfeld" / "__init__.pyi").write_text("value: int\n")
    (generated / "lichtfeld" / "build_info.pyi").write_text("commit: str = 'new'\n")
    (committed / "lichtfeld" / "build_info.pyi").write_text("commit: str = 'old'\n")

    assert manage_stubs.sync_stubs(generated, committed) == 0
    assert (committed / "lichtfeld" / "__init__.pyi").read_text() == "value: int\n"
    assert not (committed / "lichtfeld" / "build_info.pyi").exists()


def test_check_stubs_prunes_stale_generated_submodules(tmp_path):
    manage_stubs = _load_manage_stubs()
    generated = tmp_path / "generated"
    committed = tmp_path / "committed"

    (generated / "lichtfeld" / "pipeline").mkdir(parents=True)
    (committed / "lichtfeld" / "pipeline").mkdir(parents=True)

    (generated / "lichtfeld" / "__init__.pyi").write_text(
        "from . import pipeline as pipeline\n",
        encoding="utf-8",
    )
    (generated / "lichtfeld" / "py.typed").write_text("", encoding="utf-8")
    (generated / "lichtfeld" / "pipeline" / "__init__.pyi").write_text(
        "from . import edit as edit\n",
        encoding="utf-8",
    )
    (generated / "lichtfeld" / "pipeline" / "edit.pyi").write_text("value: int\n", encoding="utf-8")
    (generated / "lichtfeld" / "pipeline" / "undo.pyi").write_text("stale: int\n", encoding="utf-8")

    (committed / "lichtfeld" / "__init__.pyi").write_text(
        "from . import pipeline as pipeline\n",
        encoding="utf-8",
    )
    (committed / "lichtfeld" / "py.typed").write_text("", encoding="utf-8")
    (committed / "lichtfeld" / "pipeline" / "__init__.pyi").write_text(
        "from . import edit as edit\n",
        encoding="utf-8",
    )
    (committed / "lichtfeld" / "pipeline" / "edit.pyi").write_text("value: int\n", encoding="utf-8")

    assert manage_stubs.check_stubs(generated, committed) == 0
    assert not (generated / "lichtfeld" / "pipeline" / "undo.pyi").exists()


def test_collision_aliases_reexport_one_canonical_signature():
    import ast
    root = Path(__file__).resolve().parents[2]
    stubs = root / "src/python/stubs/lichtfeld"
    names = {"generate_collision", "export_collision", "cancel_collision", "get_collision_state", "set_walk_settings"}
    scene = ast.parse((stubs / "scene.pyi").read_text())
    top = ast.parse((stubs / "__init__.pyi").read_text())
    for name in names:
        assert sum(isinstance(node, ast.FunctionDef) and node.name == name for node in scene.body) == 1
        assert not any(isinstance(node, ast.FunctionDef) and node.name == name for node in top.body)
    aliases = {alias.name: alias.asname for node in top.body
               if isinstance(node, ast.ImportFrom) and node.module == "scene" and node.level == 1
               for alias in node.names}
    assert all(aliases[name] == name for name in names)

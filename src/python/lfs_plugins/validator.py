# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin validation utilities."""

import ast
import re
import sys
from pathlib import Path

from .compat import (
    LICHTFELD_VERSION,
    PLUGIN_API_VERSION,
    SUPPORTED_PLUGIN_FEATURES,
    compatibility_errors,
    validate_manifest_compatibility_fields,
)

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def validate_plugin(plugin_path: str | Path) -> list[str]:
    """Validate plugin structure and manifest. Returns list of errors (empty if valid)."""
    plugin_dir = Path(plugin_path)
    errors = []

    if not plugin_dir.exists():
        return [f"Plugin not found: {plugin_dir}"]

    manifest = plugin_dir / "pyproject.toml"
    if not manifest.exists():
        errors.append("Missing pyproject.toml")
    else:
        try:
            data = tomllib.loads(manifest.read_text())
            lf = data.get("tool", {}).get("lichtfeld", {})
            if not lf:
                errors.append("pyproject.toml: missing [tool.lichtfeld] section")
            project = data.get("project", {})
            for field in ("name", "version", "description"):
                if field not in project:
                    errors.append(f"pyproject.toml: missing project.{field}")
            if "hot_reload" not in lf:
                errors.append("pyproject.toml: missing tool.lichtfeld.hot_reload")
            errors.extend(validate_manifest_compatibility_fields(lf))
            if not errors:
                errors.extend(
                    f"pyproject.toml: {issue}"
                    for issue in compatibility_errors(
                        lf["plugin_api"],
                        lf["lichtfeld_version"],
                        lf["required_features"],
                        current_plugin_api=PLUGIN_API_VERSION,
                        current_lichtfeld_version=LICHTFELD_VERSION,
                        supported_features=SUPPORTED_PLUGIN_FEATURES,
                    )
                )
        except Exception as e:
            errors.append(f"pyproject.toml: {e}")

    init_py = plugin_dir / "__init__.py"
    if not init_py.exists():
        errors.append("Missing __init__.py")
    else:
        content = init_py.read_text()
        if "def on_load" not in content:
            errors.append("Missing on_load() function")
        if "def on_unload" not in content:
            errors.append("Missing on_unload() function")

    errors.extend(_check_panel_assets(plugin_dir))
    errors.extend(_check_venv(plugin_dir))

    return errors


def _check_venv(plugin_dir: Path) -> list[str]:
    """Check venv health for a plugin directory."""
    errors = []
    venv = plugin_dir / ".venv"

    has_deps = (plugin_dir / "pyproject.toml").exists() and \
        any((plugin_dir / "pyproject.toml").read_text().find(s) >= 0
            for s in ("dependencies",))

    if not venv.exists():
        if has_deps:
            errors.append("venv: missing — run plugin install to create")
        return errors

    if sys.platform == "win32":
        python = venv / "Scripts" / "python.exe"
    else:
        python = venv / "bin" / "python"

    if not python.exists():
        errors.append(f"venv: broken — missing {python.relative_to(plugin_dir)}")
        return errors

    stamp = venv / ".deps_installed"
    if has_deps and not stamp.exists():
        errors.append("venv: dependencies not installed")

    return errors


def _check_panel_assets(plugin_dir: Path) -> list[str]:
    errors = []
    for py_file in plugin_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        try:
            module = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue

        for class_node in (node for node in module.body if isinstance(node, ast.ClassDef)):
            if not _is_panel_class(class_node):
                continue

            template_ref = _extract_class_template(class_node, py_file)
            if not template_ref:
                continue

            template_path = _resolve_template_path(plugin_dir, py_file.parent, template_ref)
            if template_path is None:
                continue

            if not template_path.exists():
                errors.append(f"Missing template file: {_display_path(plugin_dir, template_path)}")
                continue

            errors.extend(_check_rml_links(plugin_dir, template_path))

    return errors


def _is_panel_class(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        if _expr_name(base).endswith("Panel"):
            return True
    return False


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _extract_class_template(class_node: ast.ClassDef, source_file: Path) -> Path | None:
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "template":
                    return _eval_template_expr(stmt.value, source_file)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == "template":
                return _eval_template_expr(stmt.value, source_file)
    return None


def _eval_template_expr(node: ast.AST | None, source_file: Path) -> Path | None:
    if node is None:
        return None
    if isinstance(node, ast.Name) and node.id == "__file__":
        return source_file
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        parent = _eval_template_expr(node.value, source_file)
        return parent.parent if parent is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval_template_expr(node.left, source_file)
        right = _eval_template_expr(node.right, source_file)
        if left is None or right is None or right.is_absolute():
            return None
        return left / right
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1:
            return _eval_template_expr(node.args[0], source_file)
        if _expr_name(node.func).endswith("Path") and len(node.args) == 1:
            return _eval_template_expr(node.args[0], source_file)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"resolve", "absolute"} and not node.args:
                return _eval_template_expr(node.func.value, source_file)
            if node.func.attr == "with_name" and len(node.args) == 1:
                base = _eval_template_expr(node.func.value, source_file)
                arg = node.args[0]
                if base is not None and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    return base.with_name(arg.value)
    return None


def _resolve_template_path(plugin_dir: Path, panel_dir: Path, template_ref: Path) -> Path | None:
    if not template_ref:
        return None

    if not template_ref.is_absolute() and template_ref.parts and template_ref.parts[0] == "rmlui":
        return None

    if template_ref.is_absolute():
        return template_ref

    candidates = [
        panel_dir / template_ref,
        plugin_dir / template_ref,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _check_rml_links(plugin_dir: Path, template_path: Path) -> list[str]:
    errors = []
    text = template_path.read_text()
    for href in re.findall(r'<link[^>]+type="text/rcss"[^>]+href="([^"]+)"', text):
        asset_path = template_path.parent / href
        if not asset_path.exists():
            errors.append(f"Missing stylesheet file: {_display_path(plugin_dir, asset_path)}")
    return errors


def _display_path(plugin_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(plugin_dir).as_posix()
    except ValueError:
        return str(path)

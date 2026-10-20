# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Non-mutating default checks shared by plugin preflight and registration."""

import ast
import math


UNKNOWN = object()
PROPERTY_KINDS = {"EnumProperty", "FloatProperty", "IntProperty", "BoolProperty", "StringProperty"}


def default_metadata_issues(kind, *, default=UNKNOWN, items=UNKNOWN, minimum=UNKNOWN, maximum=UNKNOWN):
    """Check known values only. Dynamic source expressions remain unknown."""
    if kind not in PROPERTY_KINDS:
        return []
    issues = []
    if kind == "EnumProperty":
        identifiers = None
        if items is not UNKNOWN:
            if not isinstance(items, (tuple, list)):
                issues.append("enum items must be a list or tuple")
            else:
                identifiers = []
                for item in items:
                    if (not isinstance(item, (tuple, list)) or len(item) < 2
                            or not isinstance(item[0], str) or not item[0]):
                        issues.append("enum items need a non-empty string identifier and a label")
                        continue
                    if item[0] in identifiers:
                        issues.append(f"duplicate enum identifier {item[0]!r}")
                    identifiers.append(item[0])
        if default is not UNKNOWN and default is not None:
            if not isinstance(default, str):
                issues.append("enum default must be a string identifier")
            elif identifiers is not None and default not in identifiers:
                issues.append(f"enum default {default!r} is not in items")
        return issues
    if default is UNKNOWN:
        return []
    valid = {
        "BoolProperty": lambda: isinstance(default, bool),
        "IntProperty": lambda: isinstance(default, int) and not isinstance(default, bool)
        and -(2**31) <= default < 2**31,
        "FloatProperty": lambda: isinstance(default, (int, float)) and not isinstance(default, bool)
        and abs(default) <= 3.4028234663852886e38 and math.isfinite(default),
        "StringProperty": lambda: isinstance(default, str),
    }[kind]()
    if not valid:
        return [f"default has an incompatible type or value for {kind}"]
    if kind in {"FloatProperty", "IntProperty"}:
        if isinstance(minimum, (int, float)) and default < minimum:
            issues.append("default is below the declared minimum")
        if isinstance(maximum, (int, float)) and default > maximum:
            issues.append("default is above the declared maximum")
    return issues


def descriptor_default_issues(descriptor):
    """Inspect resolved descriptors without changing their ordinary behavior."""
    return default_metadata_issues(
        type(descriptor).__name__, default=getattr(descriptor, "default", UNKNOWN),
        items=getattr(descriptor, "items", UNKNOWN), minimum=getattr(descriptor, "min", UNKNOWN),
        maximum=getattr(descriptor, "max", UNKNOWN))


_PROPERTY_CONSTRUCTORS = {f"{prefix}.{kind}" for prefix in ("lfs_plugins.props", "lichtfeld.props")
                          for kind in PROPERTY_KINDS}


def _expr_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def check_property_defaults(plugin_dir, *, package_prefix=None) -> list[str]:
    """Check literal declarations without importing or executing plugin code."""
    findings = []
    for source in plugin_dir.rglob("*.py"):
        if any(part in {".venv", "venv", "__pycache__", ".git"} for part in source.relative_to(plugin_dir).parts):
            continue
        try:
            # Bytes let Python honor coding declarations and UTF-8 BOMs.
            module = ast.parse(source.read_bytes(), filename=str(source))
        except SyntaxError as error:
            findings.append(f"{source.relative_to(plugin_dir).as_posix()}:{error.lineno or 1}: "
                            f"cannot check property declarations: {error.msg}")
            continue
        aliases = {}
        for node in module.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"lichtfeld", "lichtfeld.props", "lfs_plugins.props"}:
                        aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module
                if node.level:
                    if package_prefix is None:
                        continue
                    package = [package_prefix, *source.relative_to(plugin_dir).parent.parts]
                    if node.level > len(package):
                        continue
                    module_name = ".".join([*package[:len(package) - node.level + 1], *(node.module or "").split(".")]).rstrip(".")
                if module_name in {"lichtfeld", "lichtfeld.props", "lfs_plugins", "lfs_plugins.props"}:
                    for alias in node.names:
                        aliases[alias.asname or alias.name] = f"{module_name}.{alias.name}"
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            name = _expr_name(node.func).split(".")
            resolved = ".".join([aliases.get(name[0], name[0]), *name[1:]])
            kind = resolved.rsplit(".", 1)[-1]
            if resolved not in _PROPERTY_CONSTRUCTORS:
                continue
            values = {}
            # Only EnumProperty accepts `items` as its first positional argument.
            positional = ("items", "default") if kind == "EnumProperty" else ("default",)
            arguments = [(key, value) for key, value in zip(positional, node.args)]
            arguments.extend((kw.arg, kw.value) for kw in node.keywords if kw.arg in {"items", "default", "min", "max"})
            for key, value in arguments:
                try:
                    values[{"min": "minimum", "max": "maximum"}.get(key, key)] = ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    pass  # Runtime registration checks expressions once resolved.
            if (not any(key == "default" for key, _ in arguments)
                    and not any(kw.arg is None for kw in node.keywords)
                    and not any(isinstance(arg, ast.Starred) for arg in node.args)):
                values["default"] = {"EnumProperty": None, "FloatProperty": 0.0, "IntProperty": 0,
                                     "BoolProperty": False, "StringProperty": ""}[kind]
            for issue in default_metadata_issues(kind, **values):
                findings.append(f"{source.relative_to(plugin_dir).as_posix()}:{node.lineno}: {kind}: {issue}; "
                                "automatic default metadata unavailable (plugin loading is unaffected)")
    return findings

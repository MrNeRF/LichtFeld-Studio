# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Non-mutating default checks shared by plugin preflight and registration."""

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

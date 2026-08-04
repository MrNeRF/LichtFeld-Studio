# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Registry-backed retained-mode property rows."""

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import lichtfeld as lf


LEARNING_RATES = [
    "means_lr",
    "shs_lr",
    "opacity_lr",
    "scaling_lr",
    "rotation_lr",
]


@dataclass(frozen=True)
class SectionSpec:
    id: str
    header_locale_key: str
    prop_ids: tuple[str, ...]
    visibility_condition_id: Optional[str] = None


SECTIONS = (
    SectionSpec(
        id="lr",
        header_locale_key="training.opt.learning_rates",
        prop_ids=tuple(LEARNING_RATES),
    ),
)


_INT_INPUT_RE = re.compile(r"^\s*[+-]?\d[\d,]*\s*$")

_FLOAT_INPUT_RE = re.compile(
    r"""
    ^\s*
    [+-]?
    (?:
        (?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?
        |
        \.\d+
    )
    (?:[eE][+-]?\d+)?
    \s*$
    """,
    re.VERBOSE,
)


def format_number(value, *, is_int=False, precision=None):
    """Format a property value using the retained training-panel rules."""
    if is_int:
        return f"{int(value):,}"
    if precision is None:
        precision = 3
    return f"%.{int(precision)}f" % float(value)


def format_legacy_number(value, dtype, fmt):
    """Compatibility adapter for the legacy NUM_PROP_DEFS format strings."""
    if dtype == int:
        return format_number(value, is_int=True)
    return fmt % value


def parse_number(value, dtype):
    """Validate and normalize numeric text without changing legacy comma rules."""
    text = str(value).strip()
    if dtype == int:
        if not _INT_INPUT_RE.fullmatch(text):
            raise ValueError(f"invalid integer input: {value!r}")
        return text.replace(",", "")

    if not _FLOAT_INPUT_RE.fullmatch(text):
        raise ValueError(f"invalid numeric input: {value!r}")
    return text.replace(",", "")


def parse_clamped_number(value, *, is_int, min_value=None, max_value=None):
    """Parse numeric text and clamp it to registry bounds."""
    dtype = int if is_int else float
    parsed = dtype(parse_number(value, dtype))
    if min_value is not None:
        parsed = max(parsed, dtype(min_value))
    if max_value is not None:
        parsed = min(parsed, dtype(max_value))
    return parsed


def _params_value(params, prop_id):
    if params is None:
        raise RuntimeError("optimization parameters are unavailable")
    if hasattr(params, "has_params") and not params.has_params():
        raise RuntimeError("optimization parameters are unavailable")
    if isinstance(params, dict):
        return params[prop_id]
    getter = getattr(params, "get", None)
    if callable(getter):
        return getter(prop_id)
    return getattr(params, prop_id)


def _set_params_value(params, prop_id, value):
    if params is None:
        raise RuntimeError("optimization parameters are unavailable")
    if hasattr(params, "has_params") and not params.has_params():
        raise RuntimeError("optimization parameters are unavailable")
    if isinstance(params, dict):
        params[prop_id] = value
        return
    setter = getattr(params, "set", None)
    if callable(setter):
        setter(prop_id, value)
        return
    setattr(params, prop_id, value)


def _row_kind(meta):
    prop_type = str(meta.get("type", ""))
    if prop_type == "bool":
        return "checkbox"
    if prop_type == "enum":
        return "select"
    if prop_type == "float":
        if meta.get("precision") is None:
            return "slider"
        return "number"
    if prop_type in {"int", "size_t"}:
        return "number"
    return "select"


def build_rows(group_info, prop_ids, params_accessor):
    """Build ordered property-row metadata from a property-group snapshot."""
    properties = {
        str(meta["id"]): meta for meta in group_info.get("properties", [])
    }
    params = params_accessor() if callable(params_accessor) else params_accessor
    rows = []
    for prop_id in prop_ids:
        if prop_id not in properties:
            raise KeyError(f"property group is missing {prop_id!r}")
        meta = properties[prop_id]
        prop_type = str(meta.get("type", ""))
        is_int = prop_type in {"int", "size_t"}
        precision = meta.get("precision")
        row = {
            "id": prop_id,
            "kind": _row_kind(meta),
            "label_key": str(meta.get("locale_key", "")),
            "tooltip_key": str(meta.get("tooltip_key", "")),
            "precision": int(precision) if precision is not None else None,
            "step": meta.get("step", 1.0),
            "min": meta.get("min"),
            "max": meta.get("max"),
            "is_int": is_int,
            "name": str(meta.get("name", prop_id)),
            "items": list(meta.get("items", [])),
        }
        try:
            row["text"] = format_number(
                _params_value(params, prop_id),
                is_int=is_int,
                precision=row["precision"],
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            row["text"] = ""
        rows.append(row)
    return rows


class SectionBinding:
    """Own the draft buffers and record array for one generated section."""

    def __init__(
        self,
        section_id,
        rows,
        params_accessor,
        text_bufs,
        dirty=None,
    ):
        self.section_id = str(section_id)
        self.rows = tuple(rows)
        self.model_key = f"pv_{self.section_id}_rows"
        self._rows_by_id = {row["id"]: row for row in self.rows}
        self._params_accessor = params_accessor
        self._text_bufs = text_bufs
        self._dirty = dirty
        self._handle = None
        self._edit_snapshots = {}
        self.sync_text_bufs(publish=False)

    def contains(self, prop_id):
        return str(prop_id) in self._rows_by_id

    def input_key(self, prop_id):
        return f"pv_{self.section_id}_{prop_id}_str"

    def attach_handle(self, handle):
        self._handle = handle
        self.publish()

    def _params(self):
        return (
            self._params_accessor()
            if callable(self._params_accessor)
            else self._params_accessor
        )

    def canonical_text(self, prop_id):
        row = self._rows_by_id[str(prop_id)]
        try:
            value = _params_value(self._params(), row["id"])
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return ""
        return format_number(
            value,
            is_int=row["is_int"],
            precision=row["precision"],
        )

    def sync_text_bufs(self, publish=True):
        for row in self.rows:
            self._text_bufs[self.input_key(row["id"])] = self.canonical_text(
                row["id"]
            )
        if publish:
            self.publish()

    def update_draft(self, prop_id, value):
        prop_id = str(prop_id)
        if prop_id not in self._rows_by_id:
            return False
        self._text_bufs[self.input_key(prop_id)] = str(value)
        return True

    def capture(self, prop_id):
        return self.canonical_text(prop_id)

    def begin_edit(self, prop_id):
        prop_id = str(prop_id)
        if prop_id not in self._rows_by_id:
            return False
        self._edit_snapshots[prop_id] = self.capture(prop_id)
        return True

    def finish_edit(self, prop_id):
        self._edit_snapshots.pop(str(prop_id), None)

    def cancel_edit(self, prop_id):
        prop_id = str(prop_id)
        if prop_id not in self._rows_by_id:
            return False
        snapshot = self._edit_snapshots.pop(prop_id, None)
        if snapshot is None:
            snapshot = self.capture(prop_id)
        self.restore(prop_id, snapshot)
        return True

    def restore(self, prop_id, snapshot):
        prop_id = str(prop_id)
        if prop_id not in self._rows_by_id:
            return
        self._text_bufs[self.input_key(prop_id)] = str(snapshot or "")
        self.publish()

    def commit(self, prop_id, *, publish=True):
        prop_id = str(prop_id)
        row = self._rows_by_id.get(prop_id)
        if row is None:
            return False

        key = self.input_key(prop_id)
        original = self._text_bufs.get(key)
        updated = False
        if original is not None and str(original).strip():
            try:
                value = parse_clamped_number(
                    original,
                    is_int=row["is_int"],
                    min_value=row["min"],
                    max_value=row["max"],
                )
                _set_params_value(self._params(), prop_id, value)
                updated = True
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

        canonical = self.canonical_text(prop_id)
        if original != canonical:
            self._text_bufs[key] = canonical
            if publish:
                self.publish()
        return updated

    def step(self, prop_id, direction):
        prop_id = str(prop_id)
        row = self._rows_by_id.get(prop_id)
        if row is None:
            return False
        dtype = int if row["is_int"] else float
        try:
            current = _params_value(self._params(), prop_id)
            value = dtype(current + float(row["step"]) * int(direction))
            if row["min"] is not None:
                value = max(value, dtype(row["min"]))
            if row["max"] is not None:
                value = min(value, dtype(row["max"]))
            _set_params_value(self._params(), prop_id, value)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return False

        self._text_bufs[self.input_key(prop_id)] = self.canonical_text(prop_id)
        self.publish()
        return True

    def _records(self):
        records = []
        for row in self.rows:
            label_key = row["label_key"]
            label = lf.ui.tr(label_key) if label_key else row["name"]
            records.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "label": label,
                    "tooltip_key": row["tooltip_key"],
                    "text": str(
                        self._text_bufs.get(self.input_key(row["id"]), "")
                    ),
                }
            )
        return records

    def publish(self):
        if self._handle is None:
            return
        self._handle.update_record_list(self.model_key, self._records())
        if callable(self._dirty):
            self._dirty(self.model_key)


def bind_section(
    model,
    section_id,
    prop_ids: Sequence[str],
    params_accessor,
    text_bufs,
    dirty=None,
):
    """Bind an ordered registry-backed section and return its runtime state."""
    group_info = lf.ui.property_group_info("optimization")
    rows = build_rows(group_info, prop_ids, params_accessor)
    binding = SectionBinding(
        section_id,
        rows,
        params_accessor,
        text_bufs,
        dirty,
    )
    model.bind_record_list(binding.model_key)
    return binding

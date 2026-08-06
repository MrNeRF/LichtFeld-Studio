# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Registry-backed retained-mode property rows for the training panel."""

import re
from dataclasses import dataclass
from typing import Callable, Optional

import lichtfeld as lf


@dataclass(frozen=True)
class SectionRunSpec:
    id: str
    prop_ids: tuple[str, ...]
    visibility_condition_id: Optional[str] = None
    disabled_condition_id: Optional[str] = None


@dataclass(frozen=True)
class SectionSpec:
    id: str
    header_locale_key: str
    runs: tuple[SectionRunSpec, ...] = ()

    @property
    def header_model_key(self):
        return f"pv_header_{self.id}"


NUMBER_PROPS = (
    "iterations",
    "max_cap",
    "steps_scaler",
    "means_lr",
    "shs_lr",
    "opacity_lr",
    "scaling_lr",
    "rotation_lr",
    "refine_every",
    "start_refine",
    "stop_refine",
    "grow_until_iter",
    "reset_every",
    "sh_degree_interval",
    "bilateral_grid_x",
    "bilateral_grid_y",
    "bilateral_grid_w",
    "bilateral_grid_lr",
    "mask_opacity_penalty_weight",
    "mask_opacity_penalty_power",
    "mask_threshold",
    "depth_loss_weight",
    "normal_loss_weight",
    "normal_consistency_weight",
    "normal_flatten_weight",
    "opacity_reg",
    "scale_reg",
    "tv_loss_weight",
    "init_scaling",
    "init_num_pts",
    "init_extent",
    "min_opacity",
    "prune_opacity",
    "sparsify_steps",
    "init_rho",
    "ppisp_controller_lr",
)

BOOL_PROPS = (
    "use_bilateral_grid",
    "invert_masks",
    "use_alpha_as_mask",
    "use_depth_loss",
    "use_normal_loss",
    "enable_sparsity",
    "gut",
    "undistort",
    "mip_filter",
    "ppisp",
    "ppisp_use_controller",
    "ppisp_freeze_from_sidecar",
    "ppisp_freeze_gaussians",
    "random",
    "enable_eval",
)

SELECT_PROPS = ("mask_mode", "bg_mode", "normal_loss_space")
MIGRATED_PROP_IDS = NUMBER_PROPS + BOOL_PROPS + SELECT_PROPS

# These registered properties are intentionally represented by bespoke widgets or
# are runtime-only. Keep the reasons here so registry auto-placement stays auditable.
BESPOKE_OR_HIDDEN = {
    "sh_degree": "finite-value select with bespoke labels and tooltips",
    "lambda_dssim": "scrub slider",
    "init_opacity": "scrub slider",
    "depth_loss_mode": "bespoke fixed-value string select",
    "strategy": "strategy conflict-confirmation select",
    "ppisp_controller_activation_step": "derived -1 schedule semantics",
    "bg_modulation": "legacy mirror controlled by bg_mode",
    "headless": "runtime-only read-only flag",
    "prune_ratio": "scrub slider",
    "steps_scaler": "driven by apply_step_scaling via the iterations lock; raw edits desync step counts",
}

AUTO_ADVANCED_RUN_ID = "advanced_registry"


def _run(
    run_id,
    *prop_ids,
    visibility_condition_id=None,
    disabled_condition_id=None,
):
    return SectionRunSpec(
        id=run_id,
        prop_ids=tuple(prop_ids),
        visibility_condition_id=visibility_condition_id,
        disabled_condition_id=disabled_condition_id,
    )


BASIC_RUNS = (
    _run("basic_struct", "iterations", "max_cap"),
    _run(
        "basic_live_start",
        "use_bilateral_grid",
        "mask_mode",
        "use_depth_loss",
    ),
    _run(
        "basic_depth_weight",
        "depth_loss_weight",
        visibility_condition_id="dep_depth_loss",
    ),
    _run("basic_normal_toggle", "use_normal_loss"),
    _run(
        "basic_normal_weights",
        "normal_loss_weight",
        "normal_consistency_weight",
        "normal_flatten_weight",
        "normal_loss_space",
        visibility_condition_id="dep_normal_loss",
    ),
    _run("mask_invert", "invert_masks", visibility_condition_id="dep_mask_mode"),
    _run(
        "mask_threshold",
        "mask_threshold",
        visibility_condition_id="dep_mask_threshold",
    ),
    _run(
        "mask_alpha",
        "use_alpha_as_mask",
        visibility_condition_id="dep_mask_mode",
    ),
    _run(
        "mask_penalties",
        "mask_opacity_penalty_weight",
        "mask_opacity_penalty_power",
        visibility_condition_id="dep_mask_segment",
    ),
    _run("basic_sparsity_toggle", "enable_sparsity"),
    _run("basic_gut", "gut", disabled_condition_id="gut_disabled"),
    _run("basic_after_gut", "undistort", "mip_filter", "ppisp"),
    _run(
        "ppisp_freeze",
        "ppisp_freeze_from_sidecar",
        visibility_condition_id="dep_ppisp",
    ),
    _run(
        "ppisp_controller",
        "ppisp_use_controller",
        visibility_condition_id="dep_ppisp",
    ),
    _run(
        "ppisp_controller_tail",
        "ppisp_controller_lr",
        "ppisp_freeze_gaussians",
        visibility_condition_id="dep_ppisp_controller",
    ),
    _run("bg_mode", "bg_mode"),
)

DATASET_RUNS = (
    _run("dataset_eval", "enable_eval", visibility_condition_id="has_dataset"),
)

OPTIMIZATION_RUNS = (
    _run(
        "learning_rates",
        "means_lr",
        "shs_lr",
        "opacity_lr",
        "scaling_lr",
        "rotation_lr",
    ),
    _run(
        "refinement_locked",
        "refine_every",
        "start_refine",
        "stop_refine",
        disabled_condition_id="step_scaling_params_locked",
    ),
    _run(
        "refinement_grow",
        "grow_until_iter",
        visibility_condition_id="dep_mrnf",
        disabled_condition_id="step_scaling_params_locked",
    ),
    _run(
        "refinement_locked_tail",
        "reset_every",
        "sh_degree_interval",
        disabled_condition_id="step_scaling_params_locked",
    ),
    _run(
        "refinement_igs",
        "prune_opacity",
        visibility_condition_id="dep_igs",
    ),
)

BILATERAL_RUNS = (
    _run(
        "bilateral",
        "bilateral_grid_x",
        "bilateral_grid_y",
        "bilateral_grid_w",
        "bilateral_grid_lr",
        visibility_condition_id="dep_bilateral",
    ),
)

LOSS_RUNS = (_run("loss_numbers", "opacity_reg", "scale_reg", "tv_loss_weight"),)

INIT_RUNS = (
    _run("init_main", "init_scaling", "random"),
    _run(
        "init_random",
        "init_num_pts",
        "init_extent",
        visibility_condition_id="dep_random",
    ),
)

SPARSITY_RUNS = (
    _run(
        "sparsity",
        "sparsify_steps",
        "init_rho",
        visibility_condition_id="dep_sparsity",
    ),
)

SECTIONS = (
    SectionSpec("basic_params", "training.section.basic_params", BASIC_RUNS),
    SectionSpec("advanced_params", "training.section.advanced_params"),
    SectionSpec("dataset", "training.section.dataset", DATASET_RUNS),
    SectionSpec("optimization", "training.section.optimization", OPTIMIZATION_RUNS),
    SectionSpec("learning_rates", "training.opt.learning_rates"),
    SectionSpec("refinement", "training.section.refinement"),
    SectionSpec("bilateral", "training.section.bilateral_grid", BILATERAL_RUNS),
    SectionSpec("losses", "training.section.losses", LOSS_RUNS),
    SectionSpec("init", "training.section.initialization", INIT_RUNS),
    SectionSpec("sparsity", "training_panel.sparsity", SPARSITY_RUNS),
    SectionSpec("save_steps", "training_panel.save_eval_steps"),
    SectionSpec(
        "advanced_registry",
        "training.section.advanced_registry",
        (_run(AUTO_ADVANCED_RUN_ID),),
    ),
)

RUNS = tuple(run for section in SECTIONS for run in section.runs)

SEARCH_SECTION_RUN_IDS = {
    "basic_params": tuple(run.id for run in BASIC_RUNS),
    "advanced_registry": (AUTO_ADVANCED_RUN_ID,),
    "dataset": tuple(run.id for run in DATASET_RUNS),
    "optimization": tuple(run.id for run in OPTIMIZATION_RUNS),
    "learning_rates": ("learning_rates",),
    "refinement": tuple(
        run.id for run in OPTIMIZATION_RUNS if run.id != "learning_rates"
    ),
    "bilateral": tuple(run.id for run in BILATERAL_RUNS),
    "losses": tuple(run.id for run in LOSS_RUNS),
    "init": tuple(run.id for run in INIT_RUNS),
    "sparsity": tuple(run.id for run in SPARSITY_RUNS),
}
SEARCH_VISIBILITY_MODEL_KEYS = tuple(
    f"pv_section_{section_id}_visible" for section_id in SEARCH_SECTION_RUN_IDS
) + ("pv_section_advanced_params_visible",)

LEARNING_RATES = list(OPTIMIZATION_RUNS[0].prop_ids)

_ENUM_OPTION_TOOLTIP_KEYS = {
    ("mask_mode", 0): "training.tooltip.opt.mask_mode.none",
    ("mask_mode", 1): "training.tooltip.opt.mask_mode.segment",
    ("mask_mode", 2): "training.tooltip.opt.mask_mode.ignore",
    ("mask_mode", 3): "training.tooltip.opt.mask_mode.segment_and_ignore",
    ("mask_mode", 4): "training.tooltip.opt.mask_mode.alpha_consistent",
    ("bg_mode", 0): "training.tooltip.opt.bg_mode.color",
    ("bg_mode", 1): "training.tooltip.opt.bg_mode.modulation",
    ("bg_mode", 2): "training.tooltip.opt.bg_mode.image",
    ("bg_mode", 3): "training.tooltip.opt.bg_mode.random",
}

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


def parse_number(value, dtype):
    """Validate and normalize numeric text without changing comma rules."""
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
        return "number"
    if prop_type in {"int", "size_t"}:
        return "number"
    return None


def canonical_strategy_name(strategy):
    strategy = str(strategy or "")
    if strategy in {"mrnf", "mnrf", "lfs"}:
        return "mrnf"
    return strategy


def auto_advanced_prop_ids(group_info):
    """Return declaration-ordered scalar properties omitted from curated layout."""
    placed = {
        prop_id
        for run in RUNS
        if run.id != AUTO_ADVANCED_RUN_ID
        for prop_id in run.prop_ids
    }
    return tuple(
        str(meta["id"])
        for meta in group_info.get("properties", [])
        if str(meta["id"]) not in placed
        and str(meta["id"]) not in BESPOKE_OR_HIDDEN
        and _row_kind(meta) is not None
    )


def resolve_runs(group_info):
    """Resolve dynamic schema runs against one immutable registry snapshot."""
    advanced_ids = auto_advanced_prop_ids(group_info)
    return tuple(
        SectionRunSpec(
            run.id,
            advanced_ids if run.id == AUTO_ADVANCED_RUN_ID else run.prop_ids,
            run.visibility_condition_id,
            run.disabled_condition_id,
        )
        for run in RUNS
    )


def row_matches_query(prop_id, localized_label, query):
    normalized = str(query or "").strip().casefold()
    if not normalized:
        return True
    return normalized in str(prop_id).casefold() or normalized in str(
        localized_label
    ).casefold()


def _localized(key, fallback):
    if not key:
        return fallback
    value = lf.ui.tr(key)
    return value if value and value != key else fallback


def _active_strategy(params):
    try:
        return canonical_strategy_name(_params_value(params, "strategy"))
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        try:
            return canonical_strategy_name(getattr(params, "strategy"))
        except (AttributeError, TypeError, ValueError):
            return ""


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
        kind = _row_kind(meta)
        if kind is None:
            raise ValueError(f"property {prop_id!r} has no retained-mode row kind")
        precision = meta.get("precision")
        if kind == "number" and prop_type == "float" and precision is None:
            precision = 3
        items = []
        for item in meta.get("items", []):
            value = int(item["value"])
            items.append(
                {
                    **item,
                    "value": value,
                    "tooltip_key": _ENUM_OPTION_TOOLTIP_KEYS.get(
                        (prop_id, value), ""
                    ),
                }
            )
        row = {
            "id": prop_id,
            "kind": kind,
            "label_key": str(meta.get("locale_key", "")),
            "tooltip_key": str(meta.get("tooltip_key", "")),
            "precision": int(precision) if precision is not None else None,
            "step": meta.get("step", 1.0),
            "min": meta.get("min"),
            "max": meta.get("max"),
            "is_int": is_int,
            "name": str(meta.get("name", prop_id)),
            "items": items,
            "strategies": tuple(
                str(strategy) for strategy in meta.get("strategies", ())
            ),
        }
        if row["kind"] == "number":
            try:
                row["text"] = format_number(
                    _params_value(params, prop_id),
                    is_int=is_int,
                    precision=row["precision"],
                )
            except (
                AttributeError,
                KeyError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                row["text"] = ""
        rows.append(row)
    return rows


class SectionBinding:
    """Own draft buffers and record arrays for one generated row run."""

    def __init__(
        self,
        section_id,
        rows,
        params_accessor,
        text_bufs,
        publisher: Callable,
        value_setter: Optional[Callable] = None,
        search_accessor: Optional[Callable] = None,
        visibility_condition_id: Optional[str] = None,
        visibility_predicate: Optional[Callable] = None,
    ):
        self.section_id = str(section_id)
        self.rows = tuple(rows)
        self.model_key = f"pv_{self.section_id}_rows"
        self.options_model_key = f"pv_{self.section_id}_options"
        self._rows_by_id = {row["id"]: row for row in self.rows}
        self._params_accessor = params_accessor
        self._text_bufs = text_bufs
        if not callable(publisher):
            raise TypeError("SectionBinding requires a deferred publisher callback")
        self._publisher = publisher
        self._value_setter = value_setter
        self._search_accessor = search_accessor
        self._visibility_condition_id = visibility_condition_id
        self._visibility_predicate = visibility_predicate
        self._handle = None
        self._edit_snapshots = {}
        self.sync_text_bufs(publish=False)

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

    def _request_publish(self):
        self._publisher(self)

    def _search_query(self):
        if not callable(self._search_accessor):
            return ""
        return str(self._search_accessor() or "").strip()

    def _condition_visible(self):
        if not self._visibility_condition_id:
            return True
        if not callable(self._visibility_predicate):
            return True
        return bool(self._visibility_predicate(self._visibility_condition_id))

    def _write_value(self, prop_id, value):
        try:
            if callable(self._value_setter):
                result = self._value_setter(prop_id, value)
                return result is not False
            _set_params_value(self._params(), prop_id, value)
            return True
        except (
            AttributeError,
            KeyError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return False

    def canonical_text(self, prop_id):
        row = self._rows_by_id[str(prop_id)]
        try:
            value = _params_value(self._params(), row["id"])
        except (
            AttributeError,
            KeyError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return ""
        return format_number(
            value,
            is_int=row["is_int"],
            precision=row["precision"],
        )

    def sync_text_bufs(self, publish=True):
        for row in self.rows:
            if row["kind"] != "number":
                continue
            self._text_bufs[self.input_key(row["id"])] = self.canonical_text(
                row["id"]
            )
        if publish:
            self._request_publish()

    def update_draft(self, prop_id, value):
        prop_id = str(prop_id)
        row = self._rows_by_id.get(prop_id)
        if row is None or row["kind"] != "number":
            return False
        self._text_bufs[self.input_key(prop_id)] = str(value)
        return True

    def capture(self, prop_id):
        return self.canonical_text(prop_id)

    def begin_edit(self, prop_id):
        prop_id = str(prop_id)
        row = self._rows_by_id.get(prop_id)
        if row is None or row["kind"] != "number":
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
        self._request_publish()

    def commit(self, prop_id):
        prop_id = str(prop_id)
        row = self._rows_by_id.get(prop_id)
        if row is None or row["kind"] != "number":
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
                updated = self._write_value(prop_id, value)
            except (TypeError, ValueError):
                pass

        self._text_bufs[key] = self.canonical_text(prop_id)
        self._request_publish()
        return updated

    def step(self, prop_id, direction):
        prop_id = str(prop_id)
        row = self._rows_by_id.get(prop_id)
        if row is None or row["kind"] != "number":
            return False
        dtype = int if row["is_int"] else float
        try:
            current = _params_value(self._params(), prop_id)
            value = dtype(current + float(row["step"]) * int(direction))
            if row["min"] is not None:
                value = max(value, dtype(row["min"]))
            if row["max"] is not None:
                value = min(value, dtype(row["max"]))
        except (
            AttributeError,
            KeyError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return False
        if not self._write_value(prop_id, value):
            return False

        self._text_bufs[self.input_key(prop_id)] = self.canonical_text(prop_id)
        self._request_publish()
        return True

    def set_value(self, prop_id, value):
        prop_id = str(prop_id)
        row = self._rows_by_id.get(prop_id)
        if row is None:
            return False
        if row["kind"] == "checkbox":
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                value = bool(value)
        elif row["kind"] == "select":
            try:
                value = int(value)
            except (TypeError, ValueError):
                return False
            if value not in {int(item["value"]) for item in row["items"]}:
                return False
        else:
            return False
        updated = self._write_value(prop_id, value)
        self._request_publish()
        return updated

    def _records(self):
        records = []
        query = self._search_query()
        if query and not self._condition_visible():
            return records
        params = self._params()
        active_strategy = _active_strategy(params)
        for row in self.rows:
            allowed_strategies = row.get("strategies", ())
            if allowed_strategies and active_strategy not in allowed_strategies:
                continue
            label = _localized(row["label_key"], row["name"])
            if not row_matches_query(row["id"], label, query):
                continue
            record = {
                "id": row["id"],
                "kind": row["kind"],
                "label": label,
                "tooltip_key": row["tooltip_key"],
                "text": "",
                "checked": False,
                "value": "",
            }
            if row["kind"] == "number":
                record["text"] = str(
                    self._text_bufs.get(self.input_key(row["id"]), "")
                )
            else:
                try:
                    value = _params_value(params, row["id"])
                    if row["kind"] == "checkbox":
                        record["checked"] = bool(value)
                    elif row["kind"] == "select":
                        record["value"] = str(int(value))
                except (
                    AttributeError,
                    KeyError,
                    OverflowError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    pass
            records.append(record)
        return records

    def _option_records(self):
        records = []
        for row in self.rows:
            if row["kind"] != "select":
                continue
            for item in row["items"]:
                records.append(
                    {
                        "prop_id": row["id"],
                        "value": str(int(item["value"])),
                        "label": _localized(
                            str(item.get("locale_key", "")),
                            str(item.get("name", item["value"])),
                        ),
                        "tooltip_key": str(item.get("tooltip_key", "")),
                    }
                )
        return records

    def is_visible(self):
        return not self._search_query() or bool(self._records())

    def publish(self):
        """Replace bound arrays; callers must not invoke this in event dispatch."""
        if self._handle is None:
            return
        self._handle.update_record_list(self.model_key, self._records())
        self._handle.update_record_list(
            self.options_model_key, self._option_records()
        )


def bind_headers(model):
    for section in SECTIONS:
        model.bind_func(
            section.header_model_key,
            lambda key=section.header_locale_key: _localized(key, key),
        )


def bind_run(
    model,
    run_spec,
    params_accessor,
    text_bufs,
    publisher,
    value_setter=None,
    group_info=None,
    search_accessor=None,
    visibility_predicate=None,
):
    """Bind one ordered registry-backed row run and return its runtime state."""
    if group_info is None:
        group_info = lf.ui.property_group_info("optimization")
    rows = build_rows(group_info, run_spec.prop_ids, params_accessor)
    binding = SectionBinding(
        run_spec.id,
        rows,
        params_accessor,
        text_bufs,
        publisher,
        value_setter,
        search_accessor,
        run_spec.visibility_condition_id,
        visibility_predicate,
    )
    model.bind_record_list(binding.model_key)
    model.bind_record_list(binding.options_model_key)
    return binding


def section_is_visible(bindings, section_id):
    run_ids = set(SEARCH_SECTION_RUN_IDS.get(str(section_id), ()))
    selected = [binding for binding in bindings if binding.section_id in run_ids]
    return any(binding.is_visible() for binding in selected)


def _bind_search_visibility(model, bindings, search_accessor):
    for section_id in SEARCH_SECTION_RUN_IDS:
        model.bind_func(
            f"pv_section_{section_id}_visible",
            lambda section=section_id: section_is_visible(bindings, section),
        )

    advanced_sections = tuple(
        section_id
        for section_id in SEARCH_SECTION_RUN_IDS
        if section_id != "basic_params"
    )
    model.bind_func(
        "pv_section_advanced_params_visible",
        lambda: (
            not str(search_accessor() or "").strip()
            or any(section_is_visible(bindings, section) for section in advanced_sections)
        ),
    )


def bind_sections(
    model,
    params_accessor,
    text_bufs,
    publisher,
    value_setter=None,
    search_accessor=None,
    visibility_predicate=None,
):
    """Bind the complete training property-view schema."""
    bind_headers(model)
    group_info = lf.ui.property_group_info("optimization")
    bindings = tuple(
        bind_run(
            model,
            run,
            params_accessor,
            text_bufs,
            publisher,
            value_setter,
            group_info,
            search_accessor,
            visibility_predicate,
        )
        for run in resolve_runs(group_info)
    )
    if not callable(search_accessor):
        search_accessor = lambda: ""
    _bind_search_visibility(model, bindings, search_accessor)
    return bindings

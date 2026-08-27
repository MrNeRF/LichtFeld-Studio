# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Floating RmlUi panel for in-app bug reports."""

from __future__ import annotations

import threading
from collections.abc import Callable

import lichtfeld as lf

from . import bug_report
from .account_panel import _localized_error_message
from .portal_account import get_portal_account_service
from .types import Panel
from .ui.store import (
    PanelStateBinding,
    RuntimeState,
    invalidate_panel,
    new_bug_report_state,
)

__lfs_panel_classes__ = ["BugReportPanel"]
__lfs_panel_ids__ = ["lfs.bug_report"]

_SYSTEM_FIELDS = (
    ("app_version", "App version"),
    ("app_commit", "App commit"),
    ("build_type", "Build type"),
    ("os", "OS"),
    ("os_build", "OS build"),
    ("cpu", "CPU"),
    ("ram_mb", "RAM (MB)"),
    ("gpu", "GPU"),
    ("gpu_driver", "GPU driver"),
    ("cuda_runtime", "CUDA runtime"),
    ("vram_mb", "VRAM (MB)"),
    ("vram_used_mb", "VRAM used (MB)"),
)
_TRAINING_FIELDS = (
    ("strategy", "Strategy"),
    ("iteration", "Iteration"),
    ("max_iterations", "Max iterations"),
    ("num_gaussians", "Gaussians"),
    ("sh_degree", "SH degree"),
    ("resize_factor", "Resize factor"),
    ("max_width", "Max width"),
    ("image_count", "Image count"),
    ("dataset_fingerprint", "Dataset fingerprint"),
    ("render_mode", "Render mode"),
)

_open_cycle_generation = 0


def request_bug_report_open() -> None:
    """Notify the panel that the next enable is a new open cycle."""
    global _open_cycle_generation
    _open_cycle_generation += 1


class BugReportPanel(Panel):
    """Collect, review, and submit a portal bug report."""

    id = "lfs.bug_report"
    label = "Report a bug"
    space = lf.ui.PanelSpace.FLOATING
    order = 96
    options = {lf.ui.PanelOption.DEFAULT_CLOSED}
    template = "rmlui/bug_report_panel.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (520, 0)

    def __init__(self):
        super().__init__()
        self._service = get_portal_account_service()
        self._handle = None
        self._state_binding = PanelStateBinding()
        self._diagnostics: dict[str, object] = {}
        self._form: dict[str, object] = {}
        self._log_text = ""
        self._consent_to_logs = False
        self._include_previous_log = False
        self._previous_log_available = False
        self._system_expanded = False
        self._training_expanded = False
        self._log_expanded = False
        self._open_cycle_generation = -1

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("bug_report")
        if model is None:
            return

        model.bind_func("panel_label", lambda: "@tr:bugreport.title")
        for field in (
            "title",
            "category_detail",
            "expected",
            "actual",
            "step_1",
            "step_2",
            "step_3",
            "frequency",
            "stage",
        ):
            model.bind(
                field,
                lambda key=field: str(self._form.get(key, "")),
                lambda value, key=field: self._set_form_field(key, value),
            )
        model.bind_func("category", lambda: str(self._form.get("category", "crash")))
        model.bind_func(
            "show_category_detail",
            lambda: str(self._form.get("category", "")) == "other",
        )
        model.bind_func("category_detail_counter", self._category_detail_counter)
        model.bind("log_text", lambda: self._log_text, self._set_log_text)
        model.bind(
            "consent_to_logs",
            lambda: self._consent_to_logs,
            self._set_consent_to_logs,
        )
        model.bind(
            "include_previous_log",
            lambda: self._include_previous_log,
            self._set_include_previous_log,
        )

        model.bind_func("show_auth_required", self._show_auth_required)
        model.bind_func("auth_description", self._auth_description)
        model.bind_func("show_form", self._show_form)
        model.bind_func("show_success", lambda: bool(self._report_state().get("success")))
        model.bind_func("show_error", lambda: bool(self._report_state().get("error")))
        model.bind_func("error_message", self._error_message)
        model.bind_func("error_detail", self._error_detail)
        model.bind_func("show_error_detail", lambda: bool(self._error_detail()))
        model.bind_func("can_submit", self._can_submit)
        model.bind_func("validation_hint", self._validation_hint)
        model.bind_func("submit_label", self._submit_label)
        model.bind_func("show_training", lambda: bool(self._training_rows()))
        model.bind_func("show_previous_log", lambda: self._previous_log_available)
        model.bind_func("system_expanded", lambda: self._system_expanded)
        model.bind_func("training_expanded", lambda: self._training_expanded)
        model.bind_func("log_expanded", lambda: self._log_expanded)
        model.bind_func("log_counter", self._log_counter)
        model.bind_func(
            "log_over_limit",
            lambda: len(self._log_text.encode("utf-8")) > bug_report.LOG_MAX_BYTES,
        )
        model.bind_func("portal_url", lambda: str(self._report_state().get("url", "")))
        model.bind_func("show_open_portal", self._show_open_portal)
        model.bind_func("success_status", self._success_status)
        model.bind_func(
            "show_completeness_problems",
            lambda: bool(self._report_state().get("completeness_problems")),
        )

        model.bind_event("set_category", self._on_set_category)
        model.bind_event("toggle_system", self._on_toggle_system)
        model.bind_event("toggle_training", self._on_toggle_training)
        model.bind_event("toggle_log", self._on_toggle_log)
        model.bind_event("open_account", self._on_open_account)
        model.bind_event("submit_report", self._on_submit)
        model.bind_event("open_portal", self._on_open_portal)
        model.bind_record_list("system_rows")
        model.bind_record_list("training_rows")
        model.bind_string_list("completeness_problems")

        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        close_btn = doc.get_element_by_id("close-btn") if doc else None
        if close_btn:
            close_btn.add_event_listener("click", lambda _ev: self._mark_closed_cycle())
        self._prepare_form()
        self._open_cycle_generation = _open_cycle_generation
        self._state_binding.close()
        self._state_binding = PanelStateBinding().watch(
            RuntimeState.account_state,
            RuntimeState.bug_report_state,
            refresh=self._schedule_state_update,
        )
        self._service.initialize_async()

    def on_update(self, doc):
        del doc
        if self._open_cycle_generation == _open_cycle_generation:
            return False
        self._open_cycle_generation = _open_cycle_generation
        self._prepare_form()
        return True

    def on_unmount(self, doc):
        self._state_binding.close()
        self._handle = None
        super().on_unmount(doc)

    def _prepare_form(self) -> None:
        self._diagnostics = bug_report.collect_diagnostics()
        self._form = {
            "category": "crash",
            "category_detail": "",
            "title": "",
            "expected": "",
            "actual": "",
            "step_1": "",
            "step_2": "",
            "step_3": "",
            "frequency": "sometimes",
            "stage": bug_report.stage_infer(),
        }
        self._log_text = bug_report.current_session_log()
        self._consent_to_logs = False
        self._previous_log_available = bug_report.previous_session_log_available()
        self._include_previous_log = False
        RuntimeState.bug_report_state.value = new_bug_report_state()
        self._update_lists()
        invalidate_panel(self._handle, "*")

    def _set_form_field(self, field: str, value: object) -> None:
        rendered = str(value)
        if field == "category_detail":
            rendered = rendered[: bug_report.CATEGORY_DETAIL_MAX_LENGTH]
        self._form[field] = rendered
        self._dirty_form()

    def _category_detail_counter(self) -> str:
        return lf.ui.tr("bugreport.category_detail_counter").format(
            current=len(str(self._form.get("category_detail", ""))),
            maximum=bug_report.CATEGORY_DETAIL_MAX_LENGTH,
        )

    def _set_log_text(self, value: object) -> None:
        self._log_text = str(value)
        self._dirty_form()

    def _set_consent_to_logs(self, value: object) -> None:
        self._consent_to_logs = bool(value)
        self._dirty_form()

    def _set_include_previous_log(self, value: object) -> None:
        self._include_previous_log = bool(value) and self._previous_log_available
        self._dirty_form()

    def _dirty_form(self) -> None:
        invalidate_panel(self._handle, "*")

    def _report_state(self) -> dict[str, object]:
        value = RuntimeState.bug_report_state.value
        return value if isinstance(value, dict) else new_bug_report_state()

    def _show_auth_required(self) -> bool:
        snapshot = self._service.snapshot()
        return not snapshot.signed_in or snapshot.membership_required

    def _auth_description(self) -> str:
        key = (
            "bugreport.membership_required"
            if self._service.snapshot().membership_required
            else "bugreport.signed_out"
        )
        return lf.ui.tr(key)

    def _show_form(self) -> bool:
        return not self._show_auth_required() and not bool(self._report_state().get("success"))

    def _can_submit(self) -> bool:
        return (
            self._show_form()
            and not bool(self._report_state().get("submitting"))
            and len(str(self._form.get("title", "")).strip()) >= 12
            and len(str(self._form.get("expected", "")).strip()) >= 20
            and len(str(self._form.get("actual", "")).strip()) >= 20
        )

    def _validation_hint(self) -> str:
        if not self._show_form() or self._report_state().get("submitting") or self._can_submit():
            return ""
        return lf.ui.tr("bugreport.validation_hint")

    def _submit_label(self) -> str:
        key = "bugreport.submitting" if self._report_state().get("submitting") else "bugreport.submit"
        return lf.ui.tr(key)

    def _log_counter(self) -> str:
        return lf.ui.tr("bugreport.log_counter").format(
            current=len(self._log_text.encode("utf-8")),
            maximum=bug_report.LOG_MAX_BYTES,
        )

    def _error_message(self) -> str:
        error = str(self._report_state().get("error", ""))
        message = (
            lf.ui.tr("bugreport.error.generic")
            if error == "generic"
            else _localized_error_message(error)
        )
        retry_after = self._report_state().get("retry_after")
        if error == "rate_limited" and isinstance(retry_after, (int, float)):
            message = f"{message} " + lf.ui.tr("bugreport.retry_after").format(
                seconds=max(0, int(retry_after))
            )
        return message

    def _error_detail(self) -> str:
        detail = self._report_state().get("detail")
        if not isinstance(detail, dict):
            return ""
        lines = []
        for key, value in detail.items():
            if isinstance(value, list):
                rendered = "; ".join(str(item) for item in value)
            else:
                rendered = str(value)
            lines.append(f"{key}: {rendered}")
        return "\n".join(lines)

    def _success_status(self) -> str:
        state = self._report_state()
        if state.get("status") == "submitted":
            return lf.ui.tr("bugreport.success_submitted")
        return lf.ui.tr("bugreport.success_draft")

    def _show_open_portal(self) -> bool:
        url = str(self._report_state().get("url", ""))
        return bug_report.is_safe_portal_url(url, self._service.base_url)

    def _system_rows(self) -> list[dict[str, str]]:
        return self._rows(_SYSTEM_FIELDS)

    def _training_rows(self) -> list[dict[str, str]]:
        return self._rows(_TRAINING_FIELDS)

    def _rows(self, fields) -> list[dict[str, str]]:
        return [
            {"name": name, "value": str(self._diagnostics[key])}
            for key, name in fields
            if key in self._diagnostics and self._diagnostics[key] not in (None, "")
        ]

    def _update_lists(self) -> None:
        if not self._handle:
            return
        self._handle.update_record_list("system_rows", self._system_rows())
        self._handle.update_record_list("training_rows", self._training_rows())
        problems = self._report_state().get("completeness_problems", [])
        self._handle.update_string_list(
            "completeness_problems",
            [str(item) for item in problems] if isinstance(problems, list) else [],
        )

    def _on_set_category(self, _handle, _event, args) -> None:
        if args and str(args[0]) in bug_report.CATEGORIES:
            category = str(args[0])
            self._form["category"] = category
            if category != "other":
                self._form["category_detail"] = ""
            self._dirty_form()

    def _toggle(self, attribute: str) -> None:
        setattr(self, attribute, not bool(getattr(self, attribute)))
        self._dirty_form()

    def _on_toggle_system(self, _handle, _event, _args) -> None:
        self._toggle("_system_expanded")

    def _on_toggle_training(self, _handle, _event, _args) -> None:
        self._toggle("_training_expanded")

    def _on_toggle_log(self, _handle, _event, _args) -> None:
        self._toggle("_log_expanded")

    def _on_open_account(self, _handle, _event, _args) -> None:
        self._mark_closed_cycle()
        lf.ui.set_panel_enabled("lfs.account", True)
        lf.ui.set_panel_enabled(self.id, False)

    def _mark_closed_cycle(self) -> None:
        self._open_cycle_generation = -1

    def _on_submit(self, _handle, _event, _args) -> None:
        if not self._can_submit():
            return
        payload = bug_report.build_payload(
            self._form,
            consent_to_logs=self._consent_to_logs,
            current_log_text=self._log_text,
            include_previous_log=self._include_previous_log,
        )
        if bug_report.submit_async(payload):
            self._dirty_form()

    def _on_open_portal(self, _handle, _event, _args) -> None:
        bug_report.open_portal_url(str(self._report_state().get("url", "")))

    def _schedule_state_update(self) -> None:
        def run_update() -> None:
            self._update_lists()
            invalidate_panel(self._handle, "*")

        self._schedule_on_ui_thread(run_update)

    @staticmethod
    def _schedule_on_ui_thread(callback: Callable[[], None]) -> None:
        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if not callable(scheduler):
            scheduler = getattr(lf.ui, "_run_on_ui_thread", None)
        if callable(scheduler):
            try:
                scheduler(callback)
                return
            except Exception:
                pass
        if threading.current_thread() is threading.main_thread():
            callback()
            return
        request_redraw = getattr(lf.ui, "request_redraw", None)
        if callable(request_redraw):
            try:
                request_redraw()
            except Exception:
                pass

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Portal account panel."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import lichtfeld as lf

from .portal_account import get_portal_account_service
from .types import Panel
from .ui.store import PanelStateBinding, RuntimeState, invalidate_panel

__lfs_panel_classes__ = ["AccountPanel"]
__lfs_panel_ids__ = ["lfs.account"]

_ERROR_TRANSLATION_KEYS = {
    "access_denied": "account.error.access_denied",
    "authorization_pending": "account.error.authorization_pending",
    "expired_token": "account.error.expired_token",
    "invalid_grant": "account.error.invalid_grant",
    "invalid_token": "account.error.invalid_token",
    "membership_required": "account.error.membership_required",
    "portal_origin_mismatch": "account.error.portal_origin_mismatch",
    "sign_in_failed": "account.error.generic",
    "sign_in_unavailable": "account.error.unavailable",
    "slow_down": "account.error.slow_down",
}


def _localized_error_message(error: str) -> str:
    if not error:
        return ""
    key = _ERROR_TRANSLATION_KEYS.get(error, "account.error.generic")
    return lf.ui.tr(key)


def _format_connected_since(value: str) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class AccountPanel(Panel):
    """Floating panel for the LichtFeld portal account."""

    id = "lfs.account"
    label = "Account"
    space = lf.ui.PanelSpace.FLOATING
    order = 95
    template = "rmlui/account_panel.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (440, 0)

    def __init__(self):
        super().__init__()
        self._service = get_portal_account_service()
        self._handle = None
        self._state_binding = PanelStateBinding()
        self._opened_verification_uri = ""

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("account")
        if model is None:
            return

        snapshot = self._service.snapshot
        model.bind_func("panel_label", lambda: "@tr:account.title")
        model.bind_func(
            "show_signed_out",
            lambda: not snapshot().signed_in and not snapshot().linking,
        )
        model.bind_func("show_linking", lambda: snapshot().linking)
        model.bind_func(
            "show_signed_in",
            lambda: snapshot().signed_in and not snapshot().membership_required,
        )
        model.bind_func(
            "show_membership_required",
            lambda: snapshot().membership_required,
        )
        model.bind_func("show_error", lambda: bool(snapshot().error))
        model.bind_func("error_message", lambda: _localized_error_message(snapshot().error))
        model.bind_func("user_code", lambda: snapshot().user_code)
        model.bind_func("countdown", self._countdown_text)
        model.bind_func("display_name", lambda: snapshot().display_name or "—")
        model.bind_func("email", lambda: snapshot().email or "—")
        model.bind_func("tier", lambda: snapshot().tier or "—")
        model.bind_func(
            "connected_since",
            lambda: _format_connected_since(snapshot().connected_since),
        )
        model.bind_func("portal_host", lambda: snapshot().portal_host)
        model.bind_func("show_portal_host", lambda: snapshot().custom_portal)

        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)

        self._state_binding.close()
        self._state_binding = PanelStateBinding().watch(
            RuntimeState.account_state,
            refresh=self._schedule_account_state_update,
        )

        self._add_click_listener(doc, "account-sign-in", lambda _ev: self._service.start_device_flow())
        self._add_click_listener(doc, "account-open-link", self._open_verification_uri)
        self._add_click_listener(doc, "account-copy-code", self._copy_user_code)
        self._add_click_listener(doc, "account-cancel", lambda _ev: self._service.cancel_device_flow())
        self._add_click_listener(doc, "account-open-portal", lambda _ev: lf.ui.open_url(self._service.portal_url))
        self._add_click_listener(doc, "account-open-billing", lambda _ev: lf.ui.open_url(self._service.billing_url))
        self._add_click_listener(doc, "account-sign-out", lambda _ev: self._service.sign_out_async())
        self._add_click_listener(doc, "account-membership-sign-out", lambda _ev: self._service.sign_out_async())

        self._service.initialize_async()

    def on_unmount(self, doc):
        self._state_binding.close()
        self._handle = None
        super().on_unmount(doc)

    def _countdown_text(self) -> str:
        remaining = self._service.snapshot().countdown_seconds
        return f"{remaining // 60}:{remaining % 60:02d}"

    @staticmethod
    def _add_click_listener(doc, element_id: str, callback) -> None:
        element = doc.get_element_by_id(element_id)
        if element:
            element.add_event_listener("click", callback)

    def _open_verification_uri(self, _event) -> None:
        uri = self._service.snapshot().verification_uri_complete
        if uri:
            self._opened_verification_uri = uri
            lf.ui.open_url(uri)

    def _maybe_open_verification_uri(self) -> None:
        uri = self._service.snapshot().verification_uri_complete
        if uri and uri != self._opened_verification_uri:
            self._opened_verification_uri = uri
            lf.ui.open_url(uri)

    def _schedule_account_state_update(self) -> None:
        def run_update() -> None:
            self._maybe_open_verification_uri()
            # bind_func variables only re-evaluate via dirty_all; request_update alone
            # never marks them dirty and the panel keeps rendering the stale state.
            invalidate_panel(self._handle, "*")

        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if not callable(scheduler):
            scheduler = getattr(lf.ui, "_run_on_ui_thread", None)

        if callable(scheduler):
            try:
                scheduler(run_update)
                return
            except Exception:
                pass

        if threading.current_thread() is threading.main_thread():
            run_update()
            return

        request_redraw = getattr(lf.ui, "request_redraw", None)
        if callable(request_redraw):
            try:
                request_redraw()
            except Exception:
                pass

    def _copy_user_code(self, _event) -> None:
        code = self._service.snapshot().user_code
        if code:
            lf.ui.set_clipboard_text(code)

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate durable diagnostic messages at the UI boundary, never on a worker.

Old journals contain English diagnostics. Keep those diagnostic bytes for
recovery, while presenting concise localized reasons and phases in both panels.
"""
import re


def localize_message(message):
    if not message:
        return ""
    import lichtfeld as lf
    text = str(message)
    lower = text.casefold()
    # Already localized messages/keys are not fed back through the diagnostic
    # classifier. These patterns describe only the old English journal format.
    rules = (
        (r'account changed|account or .*changed|previous account', 'error.account_changed'),
        (r'sign out and reconnect|approve gallery', 'error.access'),
        (r'sign in|account details.*loading', 'sidebar.sign_in'),
        (r'access.*unavailable', 'error.access'),
        (r'project.*changed|camera track changed|preview changed|hdr background changed|preparation.*changed', 'error.project_changed'),
        (r'could not be saved|could not.*save|save.*error', 'error.save'),
        (r'not.*linked|linked.*different|linked.*another|linked.*unavailable|select.*linked|already.*linked', 'error.link'),
        (r'no visible splats', 'error.empty'),
        (r'unlock|locked|child.*preserved', 'error.locked'),
        (r'format|portal.*needs.*update|portal.*cannot receive|update the portal', 'error.format'),
        (r'title.*characters|description.*characters', 'error.details'),
        (r'backup.*no longer|recovery copy.*no longer', 'error.backup'),
        (r'couldn.t read.*links|sync record|history.*too large', 'error.storage'),
        (r'connection|disk space|local storage', 'error.connection'),
        (r'destination.*exists|unused.*filename', 'error.destination'),
        (r'^wait\b|^finish\b|^stop training|^pause the transfer|already has|^another .*window|^discard the', 'error.busy'),
        (r'no longer available|refresh.*gallery|refresh to connect', 'sidebar.refresh'),
        (r'could not|failed|invalid|redirected|no longer|cannot|not.*ready', 'error.failed'),
        (r'gallery item changed|selected gallery item changed', 'state.diverged'),
        (r'interrupted', 'state.interrupted'),
        (r'paused|stopped waiting|resume when ready', 'state.paused'),
        (r'cancel|discarded', 'info.canceled'),
        (r'saving|saving its gallery link', 'info.saving'),
        (r'keeping.*recovery|recovery copy before', 'info.backup'),
        (r'sending.*camera', 'info.sending_track'),
        (r'getting.*camera', 'info.getting_track'),
        (r'camera track (?:sent|received|applied)', 'info.track_done'),
        (r'prepar|checking downloaded|opening', 'state.preparing'),
        (r'upload complete|uploaded', 'info.upload_done'),
        (r'uploading', 'info.uploading'),
        (r'download complete|downloaded|project opened|project updated|linked project updated', 'info.pull_done'),
        (r'downloading', 'info.downloading'),
        (r'checking scene|assembling|publishing scene', 'state.processing'),
        (r'removed from gallery', 'state.remote_deleted'),
        (r'gallery details saved|finished transfers cleared|transfer cleared', 'info.done'),
        (r'gallery is up to date', 'info.checked'),
        (r'^save .*project', 'error.save_first'),
    )
    key = next((key for pattern, key in rules if re.search(pattern, lower)), None)
    if key is None:
        # Text from a modern localized command is already user-ready; external
        # technical diagnostics remain useful as the reason beside its badge.
        return text
    full_key = 'asset_manager.gallery.' + key
    translated = lf.ui.tr(full_key)
    return text if translated == full_key else translated


def report_poll_error(owner, exc, context):
    """Localize on the UI thread and log a repeated polling failure only once."""
    import traceback
    import lichtfeld as lf
    signature = (type(exc).__name__, str(exc))
    if getattr(owner, "_last_poll_error", None) != signature:
        owner._last_poll_error = signature
        lf.log.error(context + "\n" + traceback.format_exc())
    return localize_message(str(exc))

# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise live language changes in a disposable, connected GUI."""
import html
import threading
import time
import lichtfeld as lf
from lfs_plugins.training_runtime import panel
from lfs_plugins.training_runtime.presentation import preview_heading


def on_ui(action):
    done, errors = threading.Event(), []
    def invoke():
        try:
            action()
        except Exception as error:
            errors.append(error)
        finally:
            done.set()
    lf.ui.schedule_on_ui_thread(invoke)
    lf.ui.request_redraw()
    assert done.wait(10), "UI language action timed out"
    if errors:
        raise errors[0]


viewer = panel._viewer
assert viewer is not None
original = lf.ui.get_current_language()
checked = []
try:
    for language, _name in lf.ui.get_languages():
        on_ui(lambda: lf.ui.set_language(language))
        deadline = time.monotonic() + 6
        while True:
            def verify():
                current = lf.ui.get_panel_object("lfs.training_sessions")
                assert current.doc is not None
                current.render()
                for identifier, key in (("retry-help", "retry_help"), ("frame", "frame"),
                                        ("copy", "copy"), ("disconnect", "disconnect")):
                    element = current.doc.get_element_by_id("et-" + identifier)
                    assert html.unescape(element.get_inner_rml()).strip() == lf.ui.tr("external_training." + key)
                assert current.doc.get_element_by_id("et-copy").get_attribute("title", "") == lf.ui.tr("external_training.copy_tooltip")
                status = viewer.reader.status
                expected = preview_heading(status, viewer.stop.is_set(), viewer.reader.error, viewer.snapshot_in_scene())
                assert html.unescape(current.doc.get_element_by_id("et-current-state").get_inner_rml()).strip() == expected
                assert viewer.snapshot_in_scene(), "Language change lost the preview"
            try:
                on_ui(verify)
                break
            except AssertionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)  # Editor worker only; the UI keeps rendering.
        checked.append(language)
finally:
    on_ui(lambda: lf.ui.set_language(original))
assert len(checked) == 10, checked
print("LIVE_PREVIEW_LOCALIZATION_PASS", checked)

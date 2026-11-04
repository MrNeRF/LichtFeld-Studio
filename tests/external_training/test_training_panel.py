# SPDX-License-Identifier: GPL-3.0-or-later
"""Panel state contracts: identity, available actions and honest empty states."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types
import unittest
import xml.etree.ElementTree as ET
from concurrent.futures import Future
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/python"))
from lfs_plugins.training_runtime import presentation

LOCALES = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
def locale_strings(language):
    result = {}
    def flatten(value, prefix=""):
        for key, child in value.items():
            name = prefix + key
            if isinstance(child, dict):
                flatten(child, name + ".")
            else:
                result[name] = child
    flatten(json.loads((LOCALES / (language + ".json")).read_text(encoding="utf-8")))
    return result

ENGLISH = locale_strings("en")

fake_lf = types.SimpleNamespace(get_scene=lambda: "current-generation", ui=types.SimpleNamespace(Panel=object,
    PanelSpace=types.SimpleNamespace(MAIN_PANEL_TAB=1), PanelHeightMode=types.SimpleNamespace(CONTENT=1, FILL=2),
    request_redraw=lambda: None, set_panel_label=lambda *_: None, tr=ENGLISH.__getitem__))
path = Path(__file__).resolve().parents[2] / "src/python/lfs_plugins/training_runtime/panel.py"
spec = importlib.util.spec_from_file_location("lfs_plugins.training_runtime.panel_contract", path)
panel = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, lichtfeld=fake_lf):
    spec.loader.exec_module(panel)


def record(identifier, **values):
    return {"id": identifier, "directory": identifier, "pid": 123456, "label": identifier,
            "active": True, "state": "training", **values}


class Element:
    def __init__(self, identifier):
        self.id, self.text, self.markup = identifier, "", ""
        self.attrs, self.classes, self.listeners = {}, set(), {}
    def scroll_into_view(self, align): self.scrolled = align
    def set_text(self, text): self.text = text
    def set_class(self, name, value):
        if value: self.classes.add(name)
        else: self.classes.discard(name)
    def set_attribute(self, name, value): self.attrs[name] = value
    def remove_attribute(self, name): self.attrs.pop(name, None)
    def has_attribute(self, name): return name in self.attrs
    def set_inner_rml(self, markup): self.markup = markup
    def add_event_listener(self, name, callback): self.listeners[name] = callback


class Document:
    def __init__(self):
        path = Path(__file__).resolve().parents[2] / "src/visualizer/gui/rmlui/resources/external_training.rml"
        root = ET.parse(path).getroot()
        self.elements = {item.attrib["id"]: Element(item.attrib["id"]) for item in root.iter() if "id" in item.attrib}
    def get_element_by_id(self, identifier): return self.elements.get(identifier)


class Viewer:
    def __init__(self, info, snapshot=True, stopped=False, busy=False, error=""):
        self.session_info = info
        self.reader = types.SimpleNamespace(status={"state": info["state"], "iteration": 40, "iterations": 100}, error=error)
        self.stop = threading.Event()
        if stopped: self.stop.set()
        self.shown_iteration, self.copy_busy, self.copy_error = 30, busy, ""
        self.snapshot = snapshot
        self.display_arrays = object() if snapshot else None
        self.display_snapshot = {"snapshot": {"count": 3, "sh_degree": 1}, "model_count": 3} if snapshot else {}
        self.display_geometry = {"schema": 1, "model_to_data": "fixture transform"}
        self.display_bounds = ([1, 2, 3], 4)
        self.copies = 0
        self.cameras_visible, self.camera_count, self.camera_error = True, 4, ""
    def toggle_cameras(self): self.cameras_visible = not self.cameras_visible
    def snapshot_in_scene(self): return self.snapshot
    def create_local_copy(self): self.copies += 1
    def close(self, update_status=True): self.stop.set()
    def discard(self): self.close(); self.snapshot = False


class PanelContracts(unittest.TestCase):
    def setUp(self):
        context = mock.patch.dict(sys.modules, lichtfeld=fake_lf)
        context.start()
        self.addCleanup(context.stop)
        self.previous = panel._viewer
        panel._viewer = None
        with mock.patch.object(panel._discovery, "submit", return_value=Future()):
            self.panel = panel.TrainingSessionsPanel()
        self.panel.scan = None
        self.panel.loaded = True
    def tearDown(self): panel._viewer = self.previous

    def test_language_change_reformats_current_preview_without_changing_actions(self):
        panel._viewer = Viewer(record("Current"))
        english = self.panel.view_model()
        for language in (path.stem for path in sorted(LOCALES.glob("*.json"))):
            localized = locale_strings(language)
            with mock.patch.object(fake_lf.ui, "tr", side_effect=localized.__getitem__):
                model = self.panel.view_model()
            self.assertEqual(model["enabled"], english["enabled"])
            self.assertEqual(model["visible"], english["visible"])
            self.assertEqual(model["text"]["current-state"], localized["external_training.heading_connected"].format(
                state=localized["external_training.state_training"]))
            self.assertNotIn("external_training.", str(model["text"]))

    def test_current_identity_is_separate_from_browsed_trainer(self):
        current, other = record("Current model"), record("Other model")
        panel._viewer = Viewer(current)
        self.panel.records = [current, other]
        model = self.panel.view_model()
        self.assertEqual(model["text"]["current-name"], "Current model")
        self.assertEqual([x["id"] for x in self.panel.sessions], ["Other model"])
        self.assertEqual(model["text"]["open"], "Switch preview")
        self.assertEqual(model["text"]["preview-step"], "Preview shown: step 30")

    def test_preview_quality_cost_and_saved_checkpoint_are_explicit(self):
        current = record("Current", state="completed", active=False)
        viewer = Viewer(current)
        viewer.reader.status.update(snapshot={"count": 100000, "sh_degree": 1}, model_count=1000000,
            preview_capture_seconds=.012, preview_publish_seconds=.003, saved_project="/saved/project.licht")
        viewer.display_snapshot = {"snapshot": dict(viewer.reader.status["snapshot"]), "model_count": 1000000}
        panel._viewer = viewer
        self.panel.records = [current]
        model = self.panel.view_model()
        self.assertIn("100,000 of 1,000,000", model["text"]["preview-quality"])
        self.assertIn("SH1", model["text"]["preview-quality"])
        self.assertIn("12.0 ms capture", model["text"]["preview-cost"])
        self.assertIn("/saved/project.licht", model["text"]["saved-project"])
        self.assertTrue(model["visible"]["saved-project"])
        self.assertIn("saved project", model["text"]["current-message"])
        self.assertEqual(presentation.preview_cost({"preview_capture_seconds": float("nan")}), "")

    def test_reconnecting_keeps_the_displayed_snapshot_and_its_step(self):
        current = record("Current")
        previous = Viewer(current, stopped=True)
        previous.cameras_visible = False
        panel._viewer = previous
        replacement = Viewer(current, snapshot=False)
        with mock.patch.object(panel, "Viewer", return_value=replacement):
            self.panel.connect(current)
        self.assertIs(replacement.display_arrays, previous.display_arrays)
        self.assertIs(replacement.display_snapshot, previous.display_snapshot)
        self.assertEqual(replacement.shown_iteration, previous.shown_iteration)
        self.assertIs(replacement.display_geometry, previous.display_geometry)
        self.assertEqual(replacement.display_bounds, previous.display_bounds)
        self.assertFalse(replacement.cameras_visible)
        self.assertEqual(replacement.camera_count, previous.camera_count)

    def test_pending_or_failed_snapshot_does_not_change_displayed_quality(self):
        viewer = Viewer(record('Current'))
        panel._viewer = viewer
        baseline = self.panel.view_model()['text']['preview-quality']
        for stopped, error in ((False, ''), (False, 'failed to install'), (True, '')):
            viewer.reader.status.update(snapshot={'count': 4000, 'sh_degree': 3}, model_count=10000)
            viewer.reader.error = error
            if stopped: viewer.stop.set()
            model = self.panel.view_model()
            self.assertEqual(model['text']['preview-quality'], baseline)
            self.assertIn('SH1', baseline)
            self.assertEqual(model['text']['preview-step'], 'Preview shown: step 30')

    def test_same_trainer_is_not_offered_again(self):
        current = record("Current")
        panel._viewer = Viewer(current)
        self.panel.records = [current]
        model = self.panel.view_model()
        self.assertFalse(self.panel.sessions)
        self.assertFalse(model["visible"]["choices"])
        self.assertEqual(model["text"]["empty"], "No other trainers are running.")

    def test_closed_without_snapshot_does_not_keep_waiting(self):
        current = record("Current")
        panel._viewer = Viewer(current, snapshot=False, stopped=True)
        self.panel.records = [current]
        model = self.panel.view_model()
        self.assertNotIn("Waiting", model["text"]["current-message"])
        self.assertFalse(model["visible"]["copy"])
        self.assertTrue(model["visible"]["reconnect"])

    def test_busy_copy_is_disabled_even_if_action_is_dispatched(self):
        panel._viewer = Viewer(record("Current"), busy=True)
        doc = Document()
        self.panel.on_mount(doc)
        self.assertTrue(doc.elements["et-copy"].has_attribute("disabled"))
        self.panel.dispatch("copy", None)
        self.assertEqual(panel._viewer.copies, 0)
        self.assertNotIn("hidden", doc.elements["et-copy-busy"].classes)

    def test_terminal_session_cannot_open(self):
        self.panel.records = [record("Failed", active=False, state="failed")]
        doc = Document()
        self.panel.on_mount(doc)
        with mock.patch.object(panel, "Viewer") as factory:
            self.panel.dispatch("open", None)
            factory.assert_not_called()
        self.assertTrue(doc.elements["et-open"].has_attribute("disabled"))
        self.assertFalse(self.panel.sessions)

    def test_discovery_does_not_clear_connection_error_or_selection(self):
        self.panel.error = "Connection failed"
        self.panel.selected_id = "Second"
        future = Future()
        future.set_result([record("Second"), record("First")])
        self.panel.scan = future
        self.panel.on_update(None)
        self.assertEqual(self.panel.error, "Connection failed")
        model = self.panel.view_model()
        self.assertEqual(self.panel.selected_id, "Second")
        self.assertEqual(model["text"]["connect-error"], "Could not connect: Connection failed")

    def test_initial_search_does_not_claim_no_trainers(self):
        self.panel.loaded = False
        model = self.panel.view_model()
        self.assertEqual(model["text"]["empty"], "Searching for trainers…")

    def test_completed_without_snapshot_does_not_claim_preparing(self):
        current = record("Done", state="completed", active=False)
        panel._viewer = Viewer(current, snapshot=False)
        self.panel.records = [current]
        self.assertEqual(self.panel.view_model()["text"]["current-message"], "Training ended before a snapshot was received.")

    def test_dom_keeps_action_order_across_snapshot_and_disconnect(self):
        current = record("Current")
        panel._viewer = Viewer(current, snapshot=False)
        self.panel.records = [current]
        doc = Document()
        order = list(doc.elements)
        self.panel.on_mount(doc)
        panel._viewer.snapshot = True
        panel._viewer.display_arrays = object()
        self.panel.render()
        self.assertEqual(order, list(doc.elements))
        self.assertNotIn("hidden", doc.elements["et-copy"].classes)
        self.assertNotIn("hidden", doc.elements["et-disconnect"].classes)
        self.panel.dispatch("disconnect", None)
        self.assertIn("hidden", doc.elements["et-disconnect"].classes)
        self.assertNotIn("hidden", doc.elements["et-reconnect"].classes)
        self.assertEqual(order, list(doc.elements))

    def test_remove_preview_does_not_leave_a_current_binding(self):
        old = Viewer(record("Current"))
        panel._viewer = old
        self.panel.dispatch("discard", None)
        self.assertTrue(old.stop.is_set())
        self.assertIsNone(panel._viewer)

    def test_switch_discards_previous_transient_preview(self):
        old = Viewer(record("Previous"))
        panel._viewer = old
        replacement = Viewer(record("Next"), snapshot=False)
        with mock.patch.object(panel, "Viewer", return_value=replacement):
            self.panel.connect(record("Next"))
        self.assertFalse(old.snapshot)
        self.assertIs(panel._viewer, replacement)
        self.assertEqual(replacement.scene, "current-generation")

    def test_failed_switch_closes_unbound_replacement(self):
        old = Viewer(record("Previous"))
        panel._viewer = old
        replacement = Viewer(record("Next"), snapshot=False)
        with mock.patch.object(panel, "Viewer", return_value=replacement), mock.patch.object(
                old, "discard", side_effect=RuntimeError("Move child nodes out")):
            self.panel.connect(record("Next"))
        self.assertTrue(replacement.stop.is_set())
        self.assertIs(panel._viewer, old)
        self.assertIn("Move child nodes", self.panel.error)

    def test_project_name_is_used_instead_of_generic_label(self):
        self.assertEqual(presentation.title({"label": "Blender training", "project": r"D:\work\Chair.blend"}), "Chair")
        self.assertEqual(presentation.preview_heading({"state": "connecting"}, False, "", False), "Connecting to trainer")

    def test_refresh_stays_enabled_and_pending_scan_does_not_queue_duplicates(self):
        self.panel.records = [record('First')]
        before = self.panel.view_model()
        future = Future()
        self.panel.scan = future
        during = self.panel.view_model()
        self.assertEqual(before, during)
        self.assertTrue(during['enabled']['refresh'])
        with mock.patch.object(panel._discovery, 'submit', return_value=Future()) as submit:
            self.panel.dispatch('refresh', None)
            submit.assert_not_called()
            self.panel.scan = None
            self.panel.dispatch('refresh', None)
            self.panel.dispatch('refresh', None)
            submit.assert_called_once()

    def test_connected_collapsed_browser_uses_reader_and_skips_discovery(self):
        current = record('Current')
        panel._viewer = Viewer(current)
        panel._viewer.reader.status.update(id='Current', state='paused', liveness='responsive')
        self.panel.records = [current]
        self.panel.next_scan = 0
        with mock.patch.object(panel._discovery, 'submit', return_value=Future()) as submit:
            self.panel.on_update(None)
            submit.assert_not_called()
            self.assertEqual(self.panel.current_record()['state'], 'paused')
            self.assertTrue(self.panel.current_record()['active'])
            self.panel.browser_open = True
            self.panel.on_update(None)
            submit.assert_called_once()
        panel._viewer.reader.status.update(state='completed')
        self.assertFalse(self.panel.current_record()['active'])


if __name__ == "__main__": unittest.main()

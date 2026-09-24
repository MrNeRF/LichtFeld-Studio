# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""File-scoped gallery review, separate from the asset selection and transfers."""
from copy import deepcopy
from functools import partial
from pathlib import Path

from .gallery_logging import failure as log_failure
import lichtfeld as lf

from .gallery_messages import localize_message, tr as gallery_tr
from .panels import panel_class
from .types import Panel

tr = partial(gallery_tr, prefix="projects.gallery.")
__lfs_panel_classes__ = ["GalleryFilePanel"]
__lfs_panel_ids__ = ["lfs.gallery_file"]


def estimate_upload_size(publication, upload_format):
    """A preview estimate, never an eligibility or quota proof.

    SOG follows the native exporter's texture budget and compression heuristic.
    SSOG budgets two levels' worth of points. SPZ uses the packed byte budget
    before compression; the Studio format uses the binary PLY stride.
    """
    import math
    count, degree = publication.get("estimatedPoints"), publication.get("shDegree", 3)
    if type(count) is not int or count <= 0 or type(degree) is not int or not 0 <= degree <= 3:
        return None
    if upload_format == "studio":
        return 8192 + count * (14 + 3 * (degree + 1)**2) * 4
    if upload_format == "spz":
        return 8192 + count * (19 + 3 * ((degree + 1)**2 - 1))
    width = math.ceil(math.sqrt(count) / 4) * 4
    height = math.ceil(count / width / 4) * 4
    sog = 8192 + int(width * height * 4 * (7 if degree else 5) * 0.4)
    return sog * 2 if upload_format == "ssog" else sog


def open_gallery_file_panel(**review):
    panel = lf.ui.get_panel_object("lfs.gallery_file")
    if not panel:
        raise ValueError("The Gallery review could not be opened. Please try again.")
    panel.show(**review)


@panel_class("gallery_file")
class GalleryFilePanel(Panel):
    def __init__(self):
        super().__init__()
        self._review = None
        self._fields = {}
        self._state = {}
        self._error = ""
        self._handle = None
        self._unsubscribe = None
        self._submitting = False
        self._description_focused = False

    def poll(self, _context):
        return self._review is not None

    def apply_chrome(self, _payload):
        # Opening a project restores saved panel visibility. The review belongs
        # to the ongoing Gallery operation, so a saved layout must not hide it
        # while it still blocks input to the Project Manager.
        lf.ui.set_panel_enabled(self.id, self._review is not None)

    def show(self, *, controller, asset, scene, action, fields, includes="", quota="",
             warning="", publish_new=False, open_after=False, on_done=None,
             mode="publish", groups=(), on_submit=None, apply_only=False,
             expected_project_path=None):
        identity = controller.service.identity()
        key = (identity, asset["id"], action, publish_new, open_after, expected_project_path, mode, apply_only)
        if self._review and self._review["key"] == key:
            lf.ui.set_panel_enabled(self.id, True)
            self._dirty()
            return
        self._finish(False)
        self._review = dict(controller=controller, asset=deepcopy(asset), scene=deepcopy(scene),
                            action=action, identity=identity, key=key, includes=includes,
                            quota=quota, warning=warning, publish_new=publish_new,
                            open_after=open_after, on_done=on_done, mode=mode,
                            groups=deepcopy(list(groups)), on_submit=on_submit, apply_only=apply_only,
                            expected_project_path=expected_project_path)
        current_path = lf.project_poll_write().get("path") if mode == "publish" and action in ("publish", "update") else None
        open_project = bool(current_path and Path(current_path).resolve() == Path(asset["path"]).resolve())
        self._review["open_project"] = bool(open_project)
        self._fields = dict(fields)
        self._fields.setdefault("use_cover", False)
        if open_project:
            self._fields.setdefault("save_project", bool(lf.project_is_dirty()))
        self._error = ""
        if self._unsubscribe:
            self._unsubscribe()
        self._unsubscribe = controller.subscribe(self._changed)
        self._update_library_input()
        self._dirty()
        lf.ui.set_panel_enabled(self.id, True)

    def _update_library_input(self):
        library = lf.ui.get_panel_object("lfs.asset_manager")
        if not library:
            return
        doc = getattr(library, "_doc", None)
        if self._review and doc:
            for element in doc.query_selector_all(":focus"):
                element.blur()
        handle = getattr(library, "_handle", None)
        if handle:
            handle.dirty("gallery_review_open")

    def _changed(self, state):
        self._state = state
        if self._review and state.get("identity") != self._review["identity"]:
            self._close(False)
        self._dirty()

    def _dirty(self):
        if self._handle:
            self._handle.update_record_list("choices", (self._review or {}).get("groups", []))
            self._handle.dirty_all()
        lf.ui.request_redraw()

    def _set(self, name, value):
        if name in ("use_cover", "save_project"):
            self._fields[name] = value is True or str(value).lower() in ("true", "1")
            self._dirty()
            return
        value = str(value)
        if name == "upload_format" and value not in ("studio", "sog", "ssog", "spz"):
            return
        self._fields[name] = value
        self._error = ""
        self._dirty()

    def _is_pull(self):
        return bool(self._review and self._review["action"] == "pull")

    def _can_submit(self):
        if not self._review or self._submitting or self._state.get("busy"):
            return False
        if not self._state.get("signed_in") or self._state.get("relink_required"):
            return False
        if self._review.get("mode") in ("conflict", "replacement"):
            return True
        if self._is_pull():
            return bool(self._fields.get("pull_folder", "").strip() and self._fields.get("pull_name", "").strip())
        from .gallery_actions import gallery_eligibility
        if gallery_eligibility(self._review["asset"], self._eligibility_facts())["status"] == "blocked":
            return False
        return 0 < len(self._fields.get("title", "").strip()) <= 120 and len(self._fields.get("description", "")) <= 5000

    def _submit_label(self):
        if self._review and self._review.get("mode") == "conflict":
            return tr("conflict.apply_only" if self._review["apply_only"] else "conflict.apply_update")
        if self._is_pull():
            return tr("action.pull_open" if self._review["open_after"] else "action.pull")
        if self._review and self._review["action"] == "update":
            return tr("action.update")
        return tr("action.publish").rstrip("…")

    def _panel_label(self):
        mode = (self._review or {}).get("mode")
        if mode == "replacement":
            return tr("replacement.title")
        if mode == "conflict":
            if self._review["apply_only"]:
                return tr("action.apply")
            from .asset_index import display_name
            return tr("conflict.title", title=display_name(self._review["asset"]))
        return tr("dialog.download" if self._is_pull() else "action.update" if (self._review or {}).get("action") == "update" else "action.publish")

    def _choose(self, args):
        if not self._review or len(args) != 2:
            return
        identifier, choice = str(args[0]), str(args[1])
        for row in self._review.get("groups", []):
            if row["id"] == identifier and choice in (("mine", "gallery", "both") if row["can_both"] else ("mine", "gallery")):
                row["choice"] = choice
        self._dirty()

    def _replacement(self, args):
        if not self._review or not args:
            return
        action = str(args[0])
        if action != "locate" and not self._can_submit():
            return
        try:
            self._review["on_submit"](action)
            self._close(True)
        except Exception as exc:
            log_failure("_replacement", exc, path=getattr(self, "_path", ""))
            self._error = localize_message(str(exc))
            self._dirty()

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("gallery_file")
        if model is None:
            return
        for name in ("title", "description", "upload_format", "pull_folder", "pull_name", "use_cover", "save_project"):
            model.bind(name, lambda n=name: self._fields.get(n, False if n in ("use_cover", "save_project") else ""), lambda v, n=name: self._set(n, v))
        values = {
            "panel_label": self._panel_label,
            "file_name": lambda: (self._review or {}).get("asset", {}).get("name", ""),
            "is_pull": self._is_pull,
            "is_publish": lambda: not self._is_pull() and (self._review or {}).get("mode") == "publish",
            "show_local_apply": lambda: (self._review or {}).get("mode") == "conflict" and not self._review.get("apply_only"),
            "is_conflict": lambda: (self._review or {}).get("mode") == "conflict",
            "is_replacement": lambda: (self._review or {}).get("mode") == "replacement",
            "show_format": lambda: not self._is_pull() and (self._review or {}).get("mode") == "publish",
            "connection_reason": lambda: tr("eligibility.connect") if not self._state.get("signed_in") or self._state.get("relink_required") else "",
            "eligibility_reason": self._eligibility_reason,
            "estimate": lambda: tr("review.estimate", size=self._estimate()),
            "checkpoint_warning": lambda: tr("conflict.checkpoint_loss") if any(row["id"] == "content" for row in (self._review or {}).get("groups", [])) else "",
            "replacement_warning": lambda: tr("replacement.warning", path=(self._review or {}).get("asset", {}).get("path", ""), title=((self._review or {}).get("scene") or {}).get("title", "")),
            "can_submit": self._can_submit,
            "can_cover": lambda: bool((self._review or {}).get("asset", {}).get("has_preview")),
            "show_save_project": lambda: bool((self._review or {}).get("open_project")),
            "show_unsaved_hint": lambda: bool((self._review or {}).get("open_project") and not self._fields.get("save_project") and lf.project_is_dirty()),
            "submit_label": self._submit_label,
            "format_hint": lambda: tr("format." + self._fields.get("upload_format", "sog") + "_hint"),
            "includes": lambda: (self._review or {}).get("includes", ""),
            "quota": lambda: (self._review or {}).get("quota", ""),
            "warning": lambda: (self._review or {}).get("warning", ""),
            "error": lambda: self._error,
            "has_error": lambda: bool(self._error),
            "waiting": lambda: bool(self._state.get("busy")),
        }
        for name, getter in values.items():
            model.bind_func(name, getter)
        model.bind_record_list("choices")
        model.bind_event("choose", lambda _h, _e, args: self._choose(args))
        model.bind_event("replacement", lambda _h, _e, args: self._replacement(args))
        model.bind_func("cancel_label", lambda: tr("replacement.later") if (self._review or {}).get("mode") == "replacement" else tr("action.cancel"))
        for key in ("review.title", "review.description", "review.upload_as",
                    "format.studio", "format.sog", "format.ssog", "format.spz", "info.folder", "info.filename", "action.cancel"):
            model.bind_func("g_" + key.replace(".", "_"), lambda k=key: tr(k))
        model.bind_event("apply_local", lambda _h, _e, _args: self._submit(local_only=True))
        model.bind_event("submit", lambda _h, _e, _args: self._submit())
        model.bind_event("cancel", lambda _h, _e, _args: self._close(False))
        self._handle = model.get_handle()

    def _submit(self, *, local_only=False):
        if not self._can_submit() or self._review.get("mode") == "replacement":
            return
        review = self._review
        if local_only and review.get("mode") != "conflict":
            return
        controller = review["controller"]
        if controller.service.identity() != review["identity"]:
            self._close(False)
            return
        expected_path = review.get("expected_project_path")
        if expected_path is not None:
            try:
                current_path = lf.project_poll_write().get("path")
                same_project = (lf.project_has_path() and current_path
                                and Path(current_path).resolve() == Path(expected_path).resolve())
            except Exception:
                same_project = False
            if not same_project:
                self._error = localize_message(tr("error.project_changed"))
                self._dirty()
                return
        self._submitting = True
        try:
            if self._is_pull():
                name = self._fields["pull_name"].strip()
                if name in (".", "..") or "/" in name or "\\" in name:
                    raise ValueError(tr("error.destination"))
                controller.pull_asset(review["asset"], review["scene"],
                                      str(Path(self._fields["pull_folder"].strip()) / name),
                                      open_after=review["open_after"])
            elif review.get("mode") == "conflict":
                controller._decision_pending = False
                decisions = {row["id"]: row["choice"] for row in review["groups"]}
                if local_only:
                    review["on_submit"](decisions, local_only=True)
                else:
                    review["on_submit"](decisions)
            else:
                details = {
                    "title": self._fields["title"].strip(),
                    "description": self._fields["description"],
                }
                details["useEmbeddedPreview"] = bool(self._fields.get("use_cover", False))
                if review["open_project"]:
                    details["saveProject"] = bool(self._fields["save_project"])
                controller.upload_format = self._fields["upload_format"]
                controller.publish_asset(review["asset"], details, self._fields["upload_format"],
                                         update=review["action"] == "update", publish_as_new=review["publish_new"])
            # Publishing an update can hand off to conflict resolution, which
            # replaces this panel's review synchronously. Only close the review
            # that submitted; closing whatever is current would dismiss the
            # newly opened conflict choices.
            if self._review is review:
                self._close(True)
        except Exception as exc:
            log_failure("_submit", exc, path=getattr(self, "_path", ""))
            self._error = localize_message(str(exc))
        finally:
            self._submitting = False
            self._dirty()

    def _eligibility_reason(self):
        if not self._review or self._review.get("mode") != "publish" or self._is_pull():
            return ""
        from .gallery_actions import gallery_eligibility
        result = gallery_eligibility(self._review["asset"], self._eligibility_facts())
        return result["reason"] or (tr("eligibility.not_checked") if result["status"] == "not_checked" else "")

    def _eligibility_facts(self):
        replaced = ((self._review or {}).get("scene") or {}).get("contentLength", 0) if (self._review or {}).get("action") == "update" else 0
        return dict(self._state, replacedBytes=replaced, upload_format=self._fields.get("upload_format"))

    def _estimate(self):
        from .asset_format import format_size
        asset = (self._review or {}).get("asset", {})
        size = asset.get("publication", {}).get("estimatedBytes")
        link = self._state.get("links", {}).get(asset.get("id"), {})
        if not size and link.get("uploadFormat") == self._fields.get("upload_format"):
            size = ((self._review or {}).get("scene") or {}).get("contentLength")
        if not size:
            size = estimate_upload_size(asset.get("publication", {}), self._fields.get("upload_format", "sog"))
        return format_size(size) if size else tr("review.estimate_pending")

    def _finish(self, submitted):
        review, self._review = self._review, None
        self._update_library_input()
        if review and review["on_done"]:
            review["on_done"](submitted)

    def _close(self, submitted):
        self._finish(submitted)
        lf.ui.set_panel_enabled(self.id, False)

    def on_mount(self, doc):
        self._doc = doc
        self._update_library_input()
        if self._review and not self._unsubscribe:
            self._unsubscribe = self._review["controller"].subscribe(self._changed)
        self._dirty()
        close_btn = doc.get_element_by_id("close-btn")
        if close_btn:
            close_btn.add_event_listener("click", lambda _event: self._close(False))
        from . import rml_widgets
        from .rml_keys import KI_ESCAPE, KI_RETURN

        def keydown(event):
            key = int(event.get_parameter("key_identifier", "0"))
            target = event.target()
            if key == KI_ESCAPE:
                self._close(False)
                event.stop_propagation()
            elif key == KI_RETURN and not self._description_focused:
                action = target.get_attribute("data-event-click", "")
                if action == "apply_local":
                    self._submit(local_only=True)
                elif action == "cancel":
                    self._close(False)
                else:
                    self._submit()
                event.stop_propagation()

        doc.add_event_listener("keydown", keydown)
        description = doc.get_element_by_id("gallery-file-description")
        self._description_focused = False
        if description:
            description.add_event_listener(
                "focus", lambda _event: setattr(self, "_description_focused", True)
            )
            description.add_event_listener(
                "blur", lambda _event: setattr(self, "_description_focused", False)
            )
        title = doc.get_element_by_id("gallery-file-title")
        if title and not self._is_pull():
            rml_widgets.bind_select_all_on_focus(title)
            title.focus()

    def on_unmount(self, doc):
        self._doc = None
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        self._handle = None
        super().on_unmount(doc)

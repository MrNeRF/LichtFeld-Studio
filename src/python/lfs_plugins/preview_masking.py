# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-space prompt drafts. Native inference owns all model/GPU resources."""

from dataclasses import dataclass, field


@dataclass
class MaskDraft:
    revision: str
    points: list = field(default_factory=list)
    box: tuple | None = None
    history: list = field(default_factory=list)

    def remember(self):
        self.history.append((list(self.points), self.box))
        self.history = self.history[-64:]


class FrameMaskEditor:
    def __init__(self):
        self.enabled = False
        self.weights = ""
        self.tool = "positive"
        self.drafts = {}
        self.key = None
        self.request = 0
        self.phase = "idle"
        self.error = ""
        self.overlays = []
        self.scores = []
        self.candidate = 0
        self.refinement = None
        self.refinement_signature = None
        self.submitted_signature = None
        self.selected = None
        self.point_drag = None
        self.show_proposal = True
        self.drag_start = None
        self.drag_end = None

    @property
    def draft(self):
        return self.drafts.get(self.key[0]) if self.key else None

    def invalidate(self, keep_refinement=False):
        if not keep_refinement:
            self.refinement = None
        self.request = 0
        self.phase = "idle"
        self.error = ""
        self.overlays = []
        self.scores = []
        self.candidate = 0
        self.drag_start = None
        self.drag_end = None
        self.point_drag = None

    @property
    def signature(self):
        return (tuple(self.draft.points), self.draft.box) if self.draft else None

    @property
    def can_refine(self):
        return bool(self.refinement and self.signature != self.refinement_signature)

    def bind(self, path, revision):
        key = (str(path), revision) if path and revision else None
        if key == self.key:
            return False
        self.key = key
        self.selected = None
        self.invalidate()
        if key and (key[0] not in self.drafts or self.drafts[key[0]].revision != revision):
            self.drafts[key[0]] = MaskDraft(revision)
        return True

    def point(self, x, y, label):
        if self.draft and len(self.draft.points) < 256:
            self.draft.remember()
            self.draft.points.append((x, y, label))
            self.invalidate(keep_refinement=True)
            self.selected = len(self.draft.points) - 1

    def remove_selected(self):
        if not self.draft or self.selected is None:
            return
        self.draft.remember()
        if self.selected == "box":
            self.draft.box = None
        else:
            self.draft.points.pop(self.selected)
        self.selected = None
        self.invalidate()

    def move_selected(self, x, y):
        if self.draft and isinstance(self.selected, int):
            old = self.draft.points[self.selected]
            if old[:2] != (x, y):
                self.draft.remember()
                self.draft.points[self.selected] = (x, y, old[2])
                self.invalidate()

    def flip_selected(self):
        if self.draft and isinstance(self.selected, int):
            self.draft.remember()
            x, y, label = self.draft.points[self.selected]
            self.draft.points[self.selected] = (x, y, 1 - label)
            self.invalidate()

    def set_box(self, start, end):
        if not self.draft:
            return
        x0, x1 = sorted((start[0], end[0]))
        y0, y1 = sorted((start[1], end[1]))
        if x1 <= x0 or y1 <= y0:
            return
        self.draft.remember()
        self.draft.box = (x0, y0, x1, y1)
        self.invalidate(keep_refinement=True)
        self.selected = "box"

    def undo(self):
        if self.draft and self.draft.history:
            self.draft.points, self.draft.box = self.draft.history.pop()
            self.selected = None
            self.invalidate()

    def clear(self):
        if self.draft:
            self.draft.remember()
            self.draft.points = []
            self.draft.box = None
        self.invalidate()
        self.selected = None

    def accept(self, snapshot):
        if (not self.key or not self.request or snapshot.get("request") != self.request
                or snapshot.get("revision") != self.key[1]):
            return False
        phase = snapshot.get("phase", "idle")
        changed = phase != self.phase
        self.phase = phase
        self.error = snapshot.get("error", "")
        if phase == "ready" and snapshot.get("overlays") != self.overlays:
            self.overlays = list(snapshot["overlays"])
            self.scores = list(snapshot.get("scores", ()))
            self.candidate = min(self.candidate, max(0, len(self.overlays) - 1))
            self.refinement = (self.request, self.candidate)
            self.refinement_signature = self.submitted_signature
            changed = True
        return changed

    @property
    def overlay(self):
        return self.overlays[self.candidate] if self.overlays else None

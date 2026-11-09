# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-space prompt drafts. Native inference owns all model/GPU resources."""

from dataclasses import dataclass, field


@dataclass
class MaskDraft:
    revision: str
    points: list = field(default_factory=list)
    box: tuple | None = None


@dataclass(frozen=True)
class MaskApply:
    request: int
    proposal_request: int
    camera_uuid: str | None
    key: tuple
    kind: str = "apply"


class FrameMaskEditor:
    BUSY_PHASES = frozenset(("queued", "loading", "encoding", "predicting", "rendering", "closing", "tracking", "storing", "saving"))

    def __init__(self, on_edit=None):
        self.on_edit = on_edit
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
        self.box_drag = None
        self.show_proposal = True
        self.drag_start = None
        self.drag_end = None

    @property
    def draft(self):
        return self.drafts.get(self.key[0]) if self.key else None

    @property
    def busy(self):
        return self.phase in self.BUSY_PHASES

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
        self.box_drag = None

    @property
    def dragging(self):
        return self.point_drag is not None or self.drag_start is not None or self.box_drag is not None

    def dragged_box(self, point):
        box, start, edges = self.box_drag
        if edges:
            result = list(box)
            for index in edges:
                result[index] = point[index % 2]
            return tuple(result)
        dx = min(1 - box[2], max(-box[0], point[0] - start[0]))
        dy = min(1 - box[3], max(-box[1], point[1] - start[1]))
        return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)

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
            before = self.signature
            self.draft.points.append((x, y, label))
            self.invalidate(keep_refinement=True)
            self.selected = len(self.draft.points) - 1
            self._edited(before)

    def remove_selected(self):
        if not self.draft or self.selected is None:
            return
        before = self.signature
        if self.selected == "box":
            self.draft.box = None
        else:
            self.draft.points.pop(self.selected)
        self.selected = None
        self.invalidate()
        self._edited(before)

    def move_selected(self, x, y):
        if self.draft and isinstance(self.selected, int):
            old = self.draft.points[self.selected]
            if old[:2] != (x, y):
                before = self.signature
                self.draft.points[self.selected] = (x, y, old[2])
                self.invalidate()
                self._edited(before)

    def flip_selected(self):
        if self.draft and isinstance(self.selected, int):
            before = self.signature
            x, y, label = self.draft.points[self.selected]
            self.draft.points[self.selected] = (x, y, 1 - label)
            self.invalidate()
            self._edited(before)

    def set_box(self, start, end):
        if not self.draft:
            return
        x0, x1 = sorted((start[0], end[0]))
        y0, y1 = sorted((start[1], end[1]))
        if x1 <= x0 or y1 <= y0:
            return
        before = self.signature
        keep_refinement = self.draft.box is None
        if before[1] == (x0, y0, x1, y1):
            return
        self.draft.box = (x0, y0, x1, y1)
        self.invalidate(keep_refinement=keep_refinement)
        self.selected = "box"
        self._edited(before)

    def _edited(self, before):
        if self.on_edit and before != self.signature:
            self.on_edit(self.key, before, self.signature)

    def restore(self, key, state):
        draft = self.drafts.get(key[0])
        if not draft or draft.revision != key[1]:
            return False
        draft.points, draft.box = list(state[0]), state[1]
        if key == self.key:
            self.selected = None
            self.invalidate()
        return True

    def clear(self):
        before = self.signature
        if self.draft:
            self.draft.points = []
            self.draft.box = None
        self.invalidate()
        self.selected = None
        self._edited(before)

    def accept(self, snapshot):
        if (not self.key or not self.request or snapshot.get("request") != self.request
                or snapshot.get("revision") != self.key[1]):
            return False
        phase = snapshot.get("phase", "idle")
        changed = phase != self.phase
        self.phase = phase
        # A repeated completed result must not erase feedback from a local
        # validation/approval action performed after accepting that result.
        if changed or phase != "ready" or not self.overlays:
            self.error = snapshot.get("error", "")
        if phase == "ready" and not self.overlays:
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

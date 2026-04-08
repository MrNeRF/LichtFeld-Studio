# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared histogram feature definitions and mode gating."""

from __future__ import annotations

from dataclasses import dataclass

import lichtfeld as lf

from .ui.state import AppState


def histogram_tr(key: str, fallback: str) -> str:
    try:
        value = lf.ui.tr(key)
    except Exception:
        value = ""
    return fallback if not value or value == key else value


@dataclass(frozen=True)
class HistogramMetric:
    id: str
    label_key: str
    label_fallback: str
    description_key: str
    description_fallback: str

    def label(self) -> str:
        return histogram_tr(self.label_key, self.label_fallback)

    def description(self) -> str:
        return histogram_tr(self.description_key, self.description_fallback)


METRICS = (
    HistogramMetric(
        "opacity",
        "histogram.metric.opacity.label",
        "Opacity",
        "histogram.metric.opacity.description",
        "Activated Gaussian opacity in [0, 1].",
    ),
    HistogramMetric(
        "scale_x",
        "histogram.metric.scale_x.label",
        "Scale X",
        "histogram.metric.scale_x.description",
        "Activated Gaussian scale on the X axis.",
    ),
    HistogramMetric(
        "scale_y",
        "histogram.metric.scale_y.label",
        "Scale Y",
        "histogram.metric.scale_y.description",
        "Activated Gaussian scale on the Y axis.",
    ),
    HistogramMetric(
        "scale_z",
        "histogram.metric.scale_z.label",
        "Scale Z",
        "histogram.metric.scale_z.description",
        "Activated Gaussian scale on the Z axis.",
    ),
    HistogramMetric(
        "scale_max",
        "histogram.metric.scale_max.label",
        "Scale Max",
        "histogram.metric.scale_max.description",
        "Largest activated scale component per Gaussian.",
    ),
    HistogramMetric(
        "distance",
        "histogram.metric.distance.label",
        "Distance",
        "histogram.metric.distance.description",
        "Distance from the current scene center.",
    ),
)

METRIC_BY_ID = {metric.id: metric for metric in METRICS}

_ACTIVE_TRAINING_STATES = {"running", "paused", "stopping"}


def histogram_mode_available(context=None) -> bool:
    """Return whether histogram analysis should be available right now."""
    try:
        ctx = context if context is not None else lf.ui.context()
    except Exception:
        ctx = None

    if ctx is None:
        return False

    if not bool(getattr(ctx, "has_scene", False)):
        return False

    if int(getattr(ctx, "num_gaussians", 0) or 0) <= 0:
        return False

    try:
        if lf.ui.get_content_type() != "splat_files":
            return False
    except Exception:
        return False

    if bool(getattr(ctx, "is_training", False)) or bool(getattr(ctx, "is_paused", False)):
        return False

    if AppState.trainer_state.value in _ACTIVE_TRAINING_STATES:
        return False

    try:
        if lf.ui.is_point_cloud_forced():
            return False
    except Exception:
        pass

    return True

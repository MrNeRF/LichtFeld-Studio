"""Viewport overlay showing live gaussian statistics."""

import lichtfeld as lf
from lfs_plugins.ui import RuntimeState


class StatsOverlay(lf.ui.Panel):
    label = "Analyzer Stats"
    space = lf.ui.PanelSpace.VIEWPORT_OVERLAY
    order = 20

    @classmethod
    def poll(cls, context) -> bool:
        return RuntimeState.has_scene.value

    def draw(self, ui):
        y = 10
        white = (1.0, 1.0, 1.0, 0.9)
        gray = (0.7, 0.7, 0.7, 0.7)

        ui.draw_text(10, y, f"Gaussians: {RuntimeState.num_gaussians.value:,}", white)
        y += 20

        if RuntimeState.is_training.value:
            ui.draw_text(10, y, f"Iter: {RuntimeState.iteration.value}", gray)
            y += 20
            ui.draw_text(10, y, f"Loss: {RuntimeState.loss.value:.6f}", gray)
            y += 20
            ui.draw_text(10, y, f"PSNR: {RuntimeState.psnr.value:.2f} dB", gray)

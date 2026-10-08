"""Viewport overlay showing live gaussian statistics."""

import lichtfeld as lf
from lfs_plugins.ui import RuntimeState


class StatsOverlay(lf.ui.Panel):
    label = "Analyzer Stats"
    space = lf.ui.PanelSpace.VIEWPORT_OVERLAY
    order = 20

    @classmethod
    def poll(cls, context) -> bool:
        return context.has_scene

    def draw(self, ui):
        x, y = ui.get_viewport_pos()
        x += 10
        y += 10
        white = (1.0, 1.0, 1.0, 0.9)
        gray = (0.7, 0.7, 0.7, 0.7)

        ui.draw_text(x, y, f"Gaussians: {RuntimeState.num_gaussians.value:,}", white)
        y += 20

        if RuntimeState.is_training.value:
            ui.draw_text(x, y, f"Iter: {RuntimeState.iteration.value}", gray)
            y += 20
            ui.draw_text(x, y, f"Loss: {RuntimeState.loss.value:.6f}", gray)
            y += 20
            ui.draw_text(x, y, f"PSNR: {RuntimeState.psnr.value:.2f} dB", gray)

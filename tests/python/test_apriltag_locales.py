"""AprilTag history and runtime labels must ship in every supported locale."""
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("language", ["de", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "zh"])
def test_apriltag_labels_are_translated(language):
    root = Path(__file__).resolve().parents[2]
    strings = json.loads((root / "src/visualizer/gui/resources/locales" / f"{language}.json").read_text(encoding="utf-8"))
    assert strings["transform"]["scale_apriltag"]
    for key in ("apriltag_scan", "apriltag_scanning", "apriltag_solving", "apriltag_scene_changed", "apriltag_busy", "apriltag_worker_required"):
        assert strings["runtime"][key]

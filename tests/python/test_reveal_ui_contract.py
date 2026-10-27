"""Checks the shipped sequencer reveal controls and localization bindings."""

import unittest
import json
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]


class RevealUiContractTest(unittest.TestCase):
    def test_effect_names_are_available_in_each_shipped_locale(self):
        effects = ("magic", "spread", "unroll", "twister", "rain", "unveil", "light_edge",
                   "rainbow", "dark_to_light", "distortion", "camera_light")
        for path in (ROOT / "src/visualizer/gui/resources/locales").glob("*.json"):
            with self.subTest(locale=path.stem):
                translations = json.loads(path.read_text(encoding="utf-8"))["sequencer"]
                for effect in effects:
                    self.assertTrue(translations.get("reveal_effect_" + effect), effect)

    def test_reveal_settings_have_localized_softness_control(self):
        root = ElementTree.parse(ROOT / "src/visualizer/gui/rmlui/resources/sequencer.rml").getroot()
        settings = root.find(".//*[@id='reveal-settings']")
        self.assertIsNotNone(settings)
        self.assertIsNotNone(settings.find(".//*[@id='reveal-softness']"))
        labels = [element.text.strip() for element in settings.iter()
                  if element.tag == "span" and element.text and element.text.strip()]
        self.assertTrue(labels and all(label.startswith("@tr:") for label in labels), labels)


if __name__ == "__main__":
    unittest.main()

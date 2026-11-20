"""Keep app chrome on shared, reloadable theme roles."""
from pathlib import Path
from html.parser import HTMLParser
import json
import re
import unittest

from rcss_test_support import rule_body

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "src/visualizer/gui/rmlui/resources"
BORDER = re.compile(
    r"\b(border(?:-(?:top|right|bottom|left))?(?:-color)?|"
    r"border(?:-(?:top-left|top-right|bottom-left|bottom-right))?-radius)"
    r"\s*:\s*((?:@\{[^}]+\}|[^;{}])+);"
)


class BorderPaletteTest(unittest.TestCase):
    def test_video_extractor_uses_theme_rules_for_presentation(self):
        source = (RESOURCES / "video_extractor.rcss").read_text()
        source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
        self.assertNotRegex(
            source, r"\b(?:color|background-color|font-size|font-family|opacity)\s*:")
        document = (RESOURCES / "video_extractor.rml").read_text()
        self.assertRegex(document, r'id="rotation-value" class="text-body"')
        self.assertRegex(document, r'id="time-label" class="text-body"')

    def test_rml_markup_does_not_embed_visual_style_rules(self):
        class StyleAudit(HTMLParser):
            def handle_starttag(self, tag, attrs):
                for name, value in attrs:
                    if name != "style" or not value:
                        continue
                    for declaration in value.split(";"):
                        prop = declaration.partition(":")[0].strip()
                        if (prop == "color" or prop.endswith("-color")
                                or prop.startswith(("font-", "border-"))
                                or prop in ("border", "background", "box-shadow")):
                            self.visual_properties.append(prop)

        for path in RESOURCES.glob("*.rml"):
            parser = StyleAudit()
            parser.visual_properties = []
            parser.feed(path.read_text())
            with self.subTest(document=path.name):
                self.assertEqual(parser.visual_properties, [])

    def test_sequencer_runtime_only_sets_keyframe_state_and_position(self):
        source = (ROOT / "src/visualizer/sequencer/rml_sequencer_panel.cpp").read_text()
        body = source.split("void RmlSequencerPanel::rebuildKeyframes()", 1)[1]
        body = body.split("void RmlSequencerPanel::rebuildPlySequenceClip()", 1)[0]
        for property_name in re.findall(r'SetProperty\("([^"]+)"', body):
            self.assertEqual(property_name, "left")
        self.assertNotIn(".palette", body)

    def test_native_pie_menu_uses_shared_style_roles(self):
        source = (ROOT / "src/visualizer/gui/pie_menu.cpp").read_text()
        self.assertIn("ui_style::menuStyle", source)
        self.assertNotIn(".palette", source)
        self.assertNotRegex(source, r"toOverlay\([^\n]+,\s*0\.\d+f\)")

    def test_border_widths_use_one_normal_and_one_emphasis_stroke(self):
        geometry = {
            ".easing-indicator.ease-in, .easing-indicator.ease-out": {
                "border-left-width", "border-right-width"},
            ".easing-indicator.ease-in": {"border-bottom-width"},
            ".easing-indicator.ease-out": {"border-top-width"},
        }
        triangle_edges = 0
        for path in RESOURCES.glob("*.rcss"):
            source = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
            source = source.replace("@{stroke.normal}dp", "1dp")
            source = source.replace("@{stroke.emphasis}dp", "2dp")
            source = re.sub(r"@\{[^}]+\}", "THEME_TOKEN", source)
            for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", source):
                selector = " ".join(selector.split())
                for prop, value in re.findall(
                        r"\b(border(?:-(?:top|right|bottom|left))?-width)\s*:\s*([^;]+);", body):
                    with self.subTest(sheet=path.name, selector=selector, property=prop):
                        if (path.name == "sequencer.rcss" and value.strip() == "4dp"
                                and prop in geometry.get(selector, set())):
                            triangle_edges += 1
                        else:
                            self.assertTrue(set(value.split()) <= {"0", "1dp", "2dp"})
        # These four edges draw triangles; they are not UI outlines.
        self.assertEqual(triangle_edges, 4)

    def test_font_weights_match_the_loaded_regular_and_semibold_faces(self):
        paths = list(RESOURCES.glob("*.rcss")) + [
            ROOT / "src/visualizer/gui/rmlui/elements/terminal_element.cpp",
            ROOT / "src/python/lfs_plugins/templates.py",
        ]
        for path in paths:
            source = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
            for weight in re.findall(r"font-weight\s*:\s*([^;]+);", source):
                with self.subTest(source=path.name, weight=weight):
                    self.assertIn(weight.strip(), {"400", "600"})

    def test_base_styles_do_not_override_theme_edges(self):
        for path in RESOURCES.glob("*.rcss"):
            if path.name.endswith(".theme.rcss"):
                continue
            source = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
            with self.subTest(sheet=path.name):
                self.assertEqual(BORDER.findall(source), [])

    def test_theme_edges_use_only_shared_roles(self):
        colors = {"transparent", *("@{" + r + "}" for r in
                   ("border", "primary", "success", "warning", "error"))}
        radii = {"0", *("@{radius." + r + "}dp" for r in
                        ("control", "surface", "round"))}
        for path in RESOURCES.glob("*.theme.rcss"):
            source = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
            # These outlines draw glyph ink and image annotations, not chrome.
            drawing_ink = {
                "components.theme.rcss": {
                    ".window-maximize-box, .window-restore-box": "@{text}",
                    "button.window-control .window-maximize-box, button.window-control .window-restore-box": "@{text_dim}",
                },
                "image_preview.theme.rcss": {
                    ".sam-point": "@{white}",
                    ".sam-box-handle": "@{white}",
                },
            }
            blocks = re.finditer(r"([^{}]+)\{((?:@\{[^}]+\}|[^{}])*)\}", source)
            declarations = ((" ".join(m[1].split()), prop, value)
                            for m in blocks for prop, value in BORDER.findall(m[2]))
            for selector, prop, value in declarations:
                ink = drawing_ink.get(path.name, {}).get(selector)
                if prop == "border-color" and ink == value.strip():
                    continue
                with self.subTest(sheet=path.name, property=prop, value=value):
                    if prop.endswith("radius"):
                        for corner in value.split():
                            if (path.name == "components.theme.rcss"
                                    and selector in {"scrollbarvertical slidertrack", "scrollbarvertical sliderbar",
                                                     "scrollbarhorizontal slidertrack", "scrollbarhorizontal sliderbar"}):
                                self.assertEqual(corner, "@{components.scrollbar_rounding}dp")
                            else:
                                self.assertIn(corner, radii)
                    else:
                        if not prop.endswith("color"):
                            value = value.split(maxsplit=1)[1]
                        self.assertIn(value.strip(), colors)

    def test_hybrid_and_immediate_panels_share_theme_rules(self):
        theme = (RESOURCES.parent / "rml_theme.cpp").read_text()
        shared = theme.split("const std::string& getComponentsThemeMedia()", 1)[1]
        shared = shared.split("void invalidateThemeMediaCache()", 1)[0]
        self.assertIn('loadBaseRCSS("rmlui/im_mode_panel.theme.rcss")', shared)
        widgets = (RESOURCES / "im_mode_panel.theme.rcss").read_text()
        self.assertIn("border-radius: @{radius.control}dp;", rule_body(widgets, ".im-box"))

    def test_copy_button_has_a_transparent_idle_border(self):
        css = (RESOURCES / "image_preview.theme.rcss").read_text()
        self.assertIn("border-color: transparent;", rule_body(css, ".btn-copy"))
        self.assertIn("border-color: @{primary};", rule_body(css, ".btn-copy:focus-visible"))

    def test_round_controls_use_the_round_radius(self):
        controls = {
            "sequencer": (".play-btn", ".keyframe.loop-point", "#playhead-handle"),
            "image_preview": (".nav-arrow",),
            "components": ('input[type="range"] sliderbar',),
            "plugin_marketplace": (".mp-empty-mark",),
        }
        for name, selectors in controls.items():
            css = (RESOURCES / f"{name}.theme.rcss").read_text()
            for selector in selectors:
                with self.subTest(sheet=name, selector=selector):
                    self.assertIn("border-radius: @{radius.round}dp;", rule_body(css, selector))

    def test_plugin_examples_use_theme_presentation(self):
        examples = ROOT / "docs/plugins/examples"
        for name in ("03_hybrid_plugin", "scrub_controls_demo"):
            panel = examples / name / "panels"
            with self.subTest(example=name):
                base = (panel / "main_panel.rcss").read_text()
                self.assertNotRegex(base, r"\b(?:color|background-color|border-radius|font-weight)\s*:")
                self.assertIn("@{radius.surface}dp", (panel / "main_panel.theme.rcss").read_text())
                self.assertIn("text-body", (panel / "main_panel.rml").read_text())
        status_bar = (examples / "02_status_bar_mixed.py").read_text()
        self.assertNotRegex(status_bar, r"color\s*:|font-weight\s*:\s*bold")

    def test_bundled_themes_share_two_corner_sizes(self):
        variants = 0
        for path in (ROOT / "src/visualizer/gui/assets/themes").glob("*.json"):
            data = json.loads(path.read_text())
            for variant in data.get("variants", {}).values():
                variants += 1
                sizes = variant["sizes"]
                with self.subTest(theme=path.name, variant=variant.get("id")):
                    self.assertEqual(sizes["frame_rounding"], 4)
                    self.assertEqual(sizes["window_rounding"], 8)
                    self.assertIn("border", variant["palette"])
        self.assertGreater(variants, 0)


if __name__ == "__main__":
    unittest.main()

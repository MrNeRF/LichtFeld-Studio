"""Keep app chrome on shared, reloadable theme roles."""
from pathlib import Path
from html.parser import HTMLParser
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

    def test_border_widths_use_shared_strokes_or_preserved_geometry(self):
        geometry = {
            ".easing-indicator.ease-in, .easing-indicator.ease-out": {
                "border-left-width", "border-right-width"},
            ".easing-indicator.ease-in": {"border-bottom-width"},
            ".easing-indicator.ease-out": {"border-top-width"},
        }
        preserved = {
            ("components.rcss", ".section-content", "border-bottom-width"): "4dp",
            ("components.rcss", ".section-gap", "border-top-width"): "4dp",
            ("components.theme.rcss", ".section-header:focus-visible", "border-top-width"): "4dp",
            ("external_training.rcss", ".et-current", "border-left-width"): "3dp",
            ("image_preview.rcss", ".thumb-item", "border-width"): "1.5dp",
            ("image_preview.rcss", ".sam-point-selected", "border-width"): "3dp",
            ("plugin_marketplace.rcss", ".plugin-list-row", "border-left-width"): "3dp",
            ("preferences.theme.rcss", ".preferences-nav-item.active", "border-left-width"): "3dp",
            ("sequencer.rcss", ".film-thumb.contains-selected .film-thumb-outline", "border-width"): "2px",
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
                        elif value.strip() != preserved.get((path.name, selector, prop)):
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
        radii = {
            "@{radius.control}dp", "@{radius.surface}dp", "@{radius.round}dp",
            "@{num(size.frame_rounding)}dp", "@{num(size.window_rounding)}dp",
            "@{window.rounding}dp", "@{components.scrollbar_rounding}dp",
        }
        drawing_ink = {
            "components.theme.rcss": {
                ".window-maximize-box, .window-restore-box": "@{text}",
                "button.window-control .window-maximize-box, button.window-control .window-restore-box": "@{text_dim}",
                ".section-header": "@{menu.background}",
                ".section-gap, .section-content": "@{menu.background}",
                ".section-header:hover, .section-header.is-expanded": "@{components.header_state_border}",
            },
            "image_preview.theme.rcss": {
                ".sam-point": "@{white}",
                ".sam-box-handle": "@{white}",
            },
            "sequencer.theme.rcss": {
                "#ply-sequence-clip.selected": "rgba(240, 245, 255, 255)",
                ".keyframe.selected": "@{text}",
                ".easing-indicator.secondary.ease-in": "@{secondary}",
                ".easing-indicator.secondary.ease-out": "@{secondary}",
                ".film-thumb.hovered .film-thumb-outline": "@{text}",
                ".film-thumb.contains-hovered-keyframe .film-thumb-outline": "@{alpha(secondary,0.50)}",
                ".film-thumb.contains-selected .film-thumb-outline": "@{alpha(primary,0.65)}",
            },
        }
        for path in RESOURCES.glob("*.theme.rcss"):
            source = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
            blocks = re.finditer(r"([^{}]+)\{((?:@\{[^}]+\}|[^{}])*)\}", source)
            declarations = ((" ".join(m[1].split()), prop, value)
                            for m in blocks for prop, value in BORDER.findall(m[2]))
            for selector, prop, value in declarations:
                ink = drawing_ink.get(path.name, {}).get(selector)
                if prop.endswith("-color") and ink == value.strip():
                    continue
                with self.subTest(sheet=path.name, property=prop, value=value):
                    if prop.endswith("radius"):
                        for corner in value.split():
                            if (path.name == "components.theme.rcss"
                                    and selector in {"scrollbarvertical slidertrack", "scrollbarvertical sliderbar",
                                                     "scrollbarhorizontal slidertrack", "scrollbarhorizontal sliderbar"}):
                                self.assertEqual(corner, "@{components.scrollbar_rounding}dp")
                            else:
                                if corner not in radii:
                                    self.assertRegex(corner, r"^(?:0|[0-9]+(?:\.[0-9]+)?(?:dp|px))$")
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
        self.assertIn("border-radius: 3dp;", rule_body(widgets, ".im-box"))

    def test_copy_button_has_a_transparent_idle_border(self):
        css = (RESOURCES / "image_preview.theme.rcss").read_text()
        self.assertIn("border-color: transparent;", rule_body(css, ".btn-copy"))
        self.assertIn("border-color: @{primary};", rule_body(css, ".btn-copy:focus-visible"))

    def test_controls_keep_their_original_geometry(self):
        controls = {
            "sequencer": {
                ".play-btn": "15dp",
                ".keyframe": "1dp",
                ".keyframe.loop-point": "6dp",
                "#playhead-handle": "7dp",
                ".snap-check": "2dp",
                ".easing-indicator.ease-in-out": "2dp",
                ".film-strip-sprocket": "1px",
                ".film-strip-marker-cap": "1.5px",
                "#panel.film-strip-attached": "@{num(size.window_rounding)}dp @{num(size.window_rounding)}dp 0dp 0dp",
                "#film-strip-panel": "0dp 0dp @{num(size.window_rounding)}dp @{num(size.window_rounding)}dp",
            },
            "image_preview": {".nav-arrow": "16dp"},
            "components": {'input[type="range"] sliderbar': "6dp"},
            "plugin_marketplace": {".mp-empty-mark": "12dp"},
            "preferences": {".preferences-mcp-switch-track": "6dp"},
            "viewport_overlay": {".vram-hud-legend-swatch": "2dp"},
        }
        for name, selectors in controls.items():
            css = (RESOURCES / f"{name}.theme.rcss").read_text()
            for selector, radius in selectors.items():
                with self.subTest(sheet=name, selector=selector):
                    self.assertIn(f"border-radius: {radius};", rule_body(css, selector))

    def test_asset_buttons_only_emit_disabled_when_true(self):
        document = (RESOURCES / "asset_manager.rml").read_text()
        contents_buttons = re.findall(
            r'<button\b[^>]*data-event-click="contents_action\([^"]*\)"[^>]*>', document)
        self.assertEqual(len(contents_buttons), 4)
        for button in contents_buttons:
            self.assertIn("data-attrif-disabled=", button)
            self.assertNotIn("data-attr-disabled=", button)

    def test_gradient_header_states_keep_their_border(self):
        css = (RESOURCES / "components.theme.rcss").read_text()
        for selector in (".section-header:hover", ".section-header.is-expanded"):
            self.assertIn("border-color: @{components.header_state_border};", rule_body(css, selector))
        self.assertIn("border-top-width: 4dp;", rule_body(css, ".section-header:focus-visible"))
        for selector in (".section-gap", ".section-content"):
            self.assertIn("border-color: @{menu.background};", rule_body(css, selector))

    def test_video_extractor_keeps_timeline_and_trim_colors(self):
        css = (RESOURCES / "video_extractor.theme.rcss").read_text()
        self.assertIn("background-color: #11131a;", rule_body(css, "#preview-shell"))
        self.assertIn("background-color: #435966;", rule_body(css, "#timeline"))
        self.assertIn("background-color: rgba(166, 227, 161, 36);",
                      rule_body(css, "input.time-input--start"))
        self.assertIn("background-color: rgba(243, 139, 168, 34);",
                      rule_body(css, "input.time-input--end"))

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



if __name__ == "__main__":
    unittest.main()

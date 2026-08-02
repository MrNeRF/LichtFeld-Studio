# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless contracts for shipped localization resources and UI bindings."""

import json
import importlib.util
import re
import subprocess
import string
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCALES = ROOT / "src" / "visualizer" / "gui" / "resources" / "locales"
RML_DIR = ROOT / "src" / "visualizer" / "gui" / "rmlui" / "resources"


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _flatten(nested, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, value


def _load(locale):
    return json.loads((LOCALES / f"{locale}.json").read_text(encoding="utf-8"))


def _fields(text):
    return tuple(sorted(
        f"{{{field_name}{f'!{conversion}' if conversion else ''}{f':{format_spec}' if format_spec else ''}}}"
        for _, field_name, format_spec, conversion in string.Formatter().parse(text)
        if field_name is not None
    ))


def test_shipped_locales_match_english_keys_and_placeholders():
    english = dict(_flatten(_load("en")))
    for path in sorted(LOCALES.glob("*.json")):
        localized = dict(_flatten(json.loads(path.read_text(encoding="utf-8"))))
        assert localized.keys() == english.keys(), path.name
        for key, english_text in english.items():
            assert _fields(str(localized[key])) == _fields(str(english_text)), f"{path.name}: {key}"


def test_locale_json_uses_one_key_per_line():
    key_pattern = re.compile(r'"(?:\\.|[^"\\])+"\s*:')
    indented_key = re.compile(r'^( +)"(?:\\.|[^"\\])+"\s*:')
    for path in sorted(LOCALES.glob("*.json")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert len(key_pattern.findall(line)) <= 1, f"{path.name}:{line_number} has multiple keys"
            match = indented_key.match(line)
            if match:
                assert len(match.group(1)) % 2 == 0, f"{path.name}:{line_number} has odd indentation"


def test_rml_translation_directives_resolve():
    directive = re.compile(r"@tr:([A-Za-z0-9_.-]+)")
    directives = {
        path: directive.findall(path.read_text(encoding="utf-8"))
        for path in (ROOT / "src" / "visualizer" / "gui" / "rmlui" / "resources").rglob("*.rml")
    }
    for locale_path in sorted(LOCALES.glob("*.json")):
        localized = dict(_flatten(json.loads(locale_path.read_text(encoding="utf-8"))))
        for path, keys in directives.items():
            for key in keys:
                assert key in localized, f"{locale_path.name}: {path}: missing {key}"
                assert str(localized[key]).strip(), f"{locale_path.name}: {path}: empty {key}"


def test_literal_localization_calls_resolve():
    keys = set(dict(_flatten(_load("en"))))
    patterns = [re.compile(r'\bLOC(?:F)?\(\s*"([A-Za-z0-9_.-]+)"'),
                re.compile(r'\blf\.ui\.tr\(\s*"([A-Za-z0-9_.-]+)"'),
                re.compile(r'\b_tr(?:_format)?\(\s*"([A-Za-z0-9_.-]+)"')]
    roots = [ROOT / "src" / "visualizer" / "gui", ROOT / "src" / "visualizer" / "sequencer",
             ROOT / "src" / "python" / "lfs_plugins"]
    for source_root in roots:
        for path in source_root.rglob("*"):
            if path.suffix not in {".cpp", ".hpp", ".h", ".py"}:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                for key in pattern.findall(source):
                    assert key in keys, f"{path}: missing {key}"


def test_hardcoded_ui_audit_has_no_candidates():
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "check_ui_hardcoded.py")],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "No likely hardcoded UI strings found." in result.stdout


def test_hardcoded_ui_audit_detects_common_bypasses():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_ui_hardcoded as audit
    finally:
        sys.path.pop(0)

    allowlist, patterns = set(), []
    assert audit.is_candidate("Cancel", allowlist, patterns)
    assert audit.is_candidate("Export {count}", allowlist, patterns)
    assert not audit.is_candidate("COLMAP", allowlist, patterns)
    assert not audit.is_candidate("{count:,}", allowlist, patterns)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "panel.py"
        source.write_text(
            'self._set_status(f"Export {count}")\nlabel = "Overview"\non_progress("Working")\n'
            'title = _ui_label("ui.overview", "Overview")\n',
            encoding="utf-8",
        )
        rml = root / "panel.rml"
        rml.write_text('<button title="Cancel">Export</button>\n', encoding="utf-8")
        source_findings = audit.scan_source(source, allowlist, patterns)
        source_texts = {finding.text for finding in source_findings}
        rml_texts = {finding.text for finding in audit.scan_rml(rml, allowlist, patterns)}
        assert {"Export {count}", "Overview", "Working"} <= source_texts
        assert sum(finding.text == "Overview" for finding in source_findings) == 1
        assert {"Cancel", "Export"} <= rml_texts


def test_watch_directory_scan_messages_format_in_every_locale():
    values = {"count": 2, "path": "assets/sample", "folders": 3, "assets": 4,
              "processed": 2, "total": 4, "added": 1, "discovered": 4,
              "created": 1, "skipped": 3, "status": "ok", "error": "failure"}
    for path in sorted(LOCALES.glob("*.json")):
        messages = json.loads(path.read_text(encoding="utf-8"))["watch_dirs"]
        for key, text in messages.items():
            if key.startswith("scan_"):
                text.format(**values)


def test_counted_messages_use_supported_plural_forms():
    spec = importlib.util.spec_from_file_location(
        "localization_helpers", ROOT / "src" / "python" / "lfs_plugins" / "localization.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helpers)

    assert helpers.plural_form("en", 1) == "one"
    assert helpers.plural_form("en", 2) == "other"
    assert helpers.plural_form("pl", 1) == "one"
    assert helpers.plural_form("pl", 2) == "few"
    assert helpers.plural_form("pl", 5) == "other"
    assert helpers.plural_form("pl", 22) == "few"
    assert helpers.plural_form("pl", 12) == "other"

    keys = dict(_flatten(_load("en")))
    for key in (
        "asset_manager.status.showing_assets",
        "plugin_marketplace.registry_loaded",
        "plugin_marketplace.registry_unavailable",
    ):
        for form in ("one", "few", "other"):
            assert f"{key}.{form}" in keys


def test_language_generation_is_part_of_cached_localized_ui_state():
    required = {
        "src/python/lfs_plugins/toolbar.py": "language_generation",
        "src/python/lfs_plugins/depth_view_controls.py": "language_generation",
        "src/python/lfs_plugins/viewport_export_controls.py": "language_generation",
        "src/python/lfs_plugins/selection_controls.py": 'changed in {"active_tool", "language_generation"}',
        "src/python/lfs_plugins/transform_controls.py": "language_generation",
        "src/python/lfs_plugins/gt_compare_controls.py": "language_generation",
        "src/python/lfs_plugins/overlays/__init__.py": "language_generation",
    }
    for relative, evidence in required.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert evidence in source, f"{relative}: missing language-change invalidation"


def test_mcp_task_status_uses_stable_outcomes_not_localized_stages():
    source = (ROOT / "src" / "app" / "mcp_runtime_tools.cpp").read_text(encoding="utf-8")
    for localized_stage in ("Complete", "Failed", "Cancelled"):
        assert f'stage == "{localized_stage}"' not in source
    for outcome_getter in (
        "getExportOutcome()",
        "getImportOutcome()",
        "getVideoExportOutcome()",
        "getMesh2SplatOutcome()",
    ):
        assert outcome_getter in source


def test_localized_formatting_does_not_bypass_safe_locale_fallback():
    for path in (ROOT / "src").rglob("*.cpp"):
        source = path.read_text(encoding="utf-8", errors="ignore")
        assert "fmt::runtime(LOC(" not in source, path


def test_startup_language_picker_refreshes_after_forwarded_input():
    source = (ROOT / "src" / "visualizer" / "gui" / "startup_overlay.cpp").read_text(encoding="utf-8")
    assert "language_generation_after_input" in source
    assert "updateLocalizedText();" in source[source.index("language_generation_after_input"):]


def test_runtime_localization_refresh_updates_live_rml_documents():
    source = (ROOT / "src" / "visualizer" / "gui" / "gui_manager.cpp").read_text(encoding="utf-8")
    pending_refresh = source[source.index("if (pending_localization_ui_refresh_"):]
    refresh_block = pending_refresh.split("// Hot-reload themes", 1)[0]
    assert "refreshLocalizedDocuments()" in refresh_block
    assert "reloadRmlResources()" not in refresh_block
    assert "ui_layout_settle_frames_" in refresh_block

    documents = (ROOT / "src" / "visualizer" / "gui" / "rmlui" / "rml_document_utils.cpp").read_text(encoding="utf-8")
    assert "preserveTranslationDirectives" in documents
    assert 'data-lfs-i18n-' in documents
    assert "bool refreshLocalizedContent" in documents
    assert "select->SetSelection(select->GetSelection())" in documents

    translated_text = re.compile(
        r"<[A-Za-z][^>]*>[ \t\r\n]*@tr:[A-Za-z0-9_.-]+[ \t\r\n]*</[A-Za-z][^>]*>"
    )
    translated_attribute = re.compile(
        r"(?:title|placeholder)\s*=\s*([\"'])@tr:[A-Za-z0-9_.-]+\1",
        re.IGNORECASE,
    )
    for path in RML_DIR.rglob("*.rml"):
        rml = path.read_text(encoding="utf-8")
        remaining = translated_text.sub("", rml)
        remaining = translated_attribute.sub("", remaining)
        assert "@tr:" not in remaining, f"{path}: unsupported runtime translation directive shape"


def test_cached_native_panels_request_a_frame_for_language_changes():
    video = (ROOT / "src" / "visualizer" / "gui" / "windows" / "video_extractor_dialog.cpp").read_text(encoding="utf-8")
    video_demand = video[video.index("bool VideoExtractorDialog::needsAnimationFrame() const") :]
    assert "current_language != last_language_" in video_demand.split("return hasDynamicState()", 1)[0]

    sequencer_panel = (ROOT / "src" / "visualizer" / "sequencer" / "rml_sequencer_panel.cpp").read_text(encoding="utf-8")
    assert "bool RmlSequencerPanel::needsLocalizationFrame() const" in sequencer_panel
    sequencer_manager = (ROOT / "src" / "visualizer" / "gui" / "sequencer_ui_manager.cpp").read_text(encoding="utf-8")
    sequencer_demand = sequencer_manager[sequencer_manager.index("bool SequencerUIManager::needsAnimationFrame() const") :]
    assert "panel_->needsLocalizationFrame()" in sequencer_demand.split("controller_.isPlaying()", 1)[0]


def test_rendering_labels_preserve_fullwidth_colons():
    source = (ROOT / "src" / "python" / "lfs_plugins" / "rendering_panel.py").read_text(encoding="utf-8")
    assert 'text.endswith((":", "："))' in source


def test_operator_property_registration_preserves_localization_keys():
    source = (ROOT / "src" / "python" / "lfs" / "py_ui.cpp").read_text(encoding="utf-8")
    registration = source[source.index("void register_class_api"):source.index('m.def(\n            "unregister_class"')]
    assert 'register_python_property_group("operator." + idname, label, cls);' in registration
    assert "localization.get(label)" not in registration


def test_cached_python_panels_request_a_frame_on_language_change():
    source = (ROOT / "src" / "python" / "lfs" / "rml_python_panel_adapter.cpp").read_text(encoding="utf-8")
    needs_frame = source[source.index("bool RmlPythonPanelAdapter::needsAnimationFrame() const"):]
    assert "current_language != last_language_" in needs_frame
    assert "return true;" in needs_frame.split("if (!dirty_driven_updates_)", 1)[0]


if __name__ == "__main__":
    contracts = [
        test_shipped_locales_match_english_keys_and_placeholders,
        test_locale_json_uses_one_key_per_line,
        test_rml_translation_directives_resolve,
        test_literal_localization_calls_resolve,
        test_hardcoded_ui_audit_has_no_candidates,
        test_hardcoded_ui_audit_detects_common_bypasses,
        test_watch_directory_scan_messages_format_in_every_locale,
        test_counted_messages_use_supported_plural_forms,
        test_language_generation_is_part_of_cached_localized_ui_state,
        test_mcp_task_status_uses_stable_outcomes_not_localized_stages,
        test_localized_formatting_does_not_bypass_safe_locale_fallback,
        test_startup_language_picker_refreshes_after_forwarded_input,
        test_runtime_localization_refresh_updates_live_rml_documents,
        test_cached_native_panels_request_a_frame_for_language_changes,
        test_rendering_labels_preserve_fullwidth_colons,
        test_operator_property_registration_preserves_localization_keys,
        test_cached_python_panels_request_a_frame_on_language_change,
    ]
    for contract in contracts:
        contract()
        print(f"PASS {contract.__name__}")

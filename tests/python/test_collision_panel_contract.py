# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep native collision status bindings and collapsible controls usable."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET


GUI = Path(__file__).resolve().parents[2] / "src/visualizer/gui"


def test_native_status_bindings_exist_in_the_document_or_its_template():
    document = ET.parse(GUI / "rmlui/resources/collision.rml").getroot()
    template = ET.parse(GUI / "rmlui/resources/floating_window.rml").getroot()
    ids = [element.attrib["id"] for element in document.iter() if "id" in element.attrib]
    assert len(ids) == len(set(ids))
    available = set(ids) | {element.attrib.get("id") for element in template.iter()}
    references = set(re.findall(r'GetElementById\("([^"]+)"\)', (GUI / "collision_panel.cpp").read_text()))
    assert references <= available


def test_collapsible_headers_match_their_initial_visible_state():
    document = ET.parse(GUI / "rmlui/resources/collision.rml").getroot()
    ids = {element.attrib["id"]: element for element in document.iter() if "id" in element.attrib}
    for header in document.iter():
        if "data-collision-section" not in header.attrib:
            continue
        content = ids[header.attrib["data-collision-section"]]
        expanded = "collapsed" not in content.attrib.get("class", "").split()
        assert header.tag == "button"
        assert header.attrib["aria-expanded"] == str(expanded).lower()
        assert ("is-expanded" in header.attrib.get("class", "").split()) == expanded


def test_progress_and_errors_stay_visible_when_settings_are_collapsed():
    document = ET.parse(GUI / "rmlui/resources/collision.rml").getroot()
    setup = next(element for element in document.iter() if element.attrib.get("id") == "setup")
    assert not {element.attrib.get("id") for element in setup.iter()} & {"job-progress", "collision-error"}

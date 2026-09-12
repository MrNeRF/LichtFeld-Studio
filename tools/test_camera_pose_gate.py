# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of report validation, not tests of the native renderer."""

import unittest
import xml.etree.ElementTree as ET

from check_camera_pose_gate import GRADIENT, RECOVERY, SUITE, TESTS, inspect_gate
from check_camera_pose_gate import CONTROLLER_RECOVERY, CONTROLLER_SUITE, CONTROLLER_TESTS, inspect_controller_gate


def valid_report():
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", name=SUITE)
    for name in sorted(TESTS):
        case = ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
        if name == GRADIENT:
            props = ET.SubElement(case, "properties")
            for scale in (1, 2):
                ET.SubElement(props, "property", name=f"gradient_relative_error_scale_{scale}", value="0.001")
        if name == RECOVERY:
            props = ET.SubElement(case, "properties")
            for trial in range(3):
                for metric, value in (("loss_ratio", "0.02"), ("center_error_ratio", "0.1"),
                                      ("rotation_error_ratio", "0.1"), ("accepted_steps", "7")):
                    ET.SubElement(props, "property", name=f"trial_{trial}_{metric}", value=value)
    return root


def valid_controller_report():
    root = valid_report()
    suite = root.find("testsuite")
    case = ET.SubElement(suite, "testcase", name=CONTROLLER_RECOVERY, status="run", result="completed")
    props = ET.SubElement(case, "properties")
    for trial in range(3):
        for metric, value in (("loss_ratio", "0.02"), ("center_error_ratio", "0.1"),
                              ("rotation_error_ratio", "0.1"), ("accepted_steps", "20"),
                              ("candidate_renders", "30")):
            ET.SubElement(props, "property", name=f"trial_{trial}_{metric}", value=value)
    suite = ET.SubElement(root, "testsuite", name=CONTROLLER_SUITE)
    for name in CONTROLLER_TESTS:
        ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
    return root


class CameraPoseControllerGateReportTests(unittest.TestCase):
    def test_accepts_complete_controller_report(self):
        result = inspect_controller_gate(valid_controller_report())
        self.assertEqual(result["tests"], 17)
        self.assertEqual(len(result["controller_trials"]), 3)

    def test_rejects_checkpoint_a_without_controller(self):
        with self.assertRaises(ValueError):
            inspect_controller_gate(valid_report())

    def test_rejects_missing_skipped_failed_duplicate_and_unknown_controller_tests(self):
        for mutation in ("missing", "skipped", "failure", "duplicate", "unknown"):
            root = valid_controller_report()
            suite = root.find(f"testsuite[@name='{CONTROLLER_SUITE}']")
            case = suite.find("testcase")
            if mutation == "missing":
                suite.remove(case)
            elif mutation == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif mutation == "unknown":
                case.set("name", "Unknown")
            else:
                ET.SubElement(case, mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_controller_gate(root)

    def test_rejects_invalid_controller_metrics(self):
        for metric, value in (("loss_ratio", "0.1"), ("loss_ratio", "nan"),
                              ("center_error_ratio", "0.3"), ("rotation_error_ratio", "0.3"),
                              ("accepted_steps", "81"), ("accepted_steps", "0"),
                              ("candidate_renders", "641"), ("candidate_renders", "1.5"),
                              ("candidate_renders", "19"), ("candidate_renders", None)):
            root = valid_controller_report()
            props = root.find(f".//testcase[@name='{CONTROLLER_RECOVERY}']/properties")
            prop = props.find(f"property[@name='trial_0_{metric}']")
            if value is None:
                props.remove(prop)
            else:
                prop.set("value", value)
            with self.subTest(metric=metric, value=value), self.assertRaises(ValueError):
                inspect_controller_gate(root)


class CameraPoseGateReportTests(unittest.TestCase):
    def test_accepts_complete_success_and_reports_all_trials(self):
        result = inspect_gate(valid_report())
        self.assertEqual(result["tests"], 8)
        self.assertEqual(len(result["trials"]), 3)
        self.assertIn("not certified", result["scope"])

    def test_rejects_old_binary_and_zero_tests(self):
        for root in (ET.Element("testsuites"), ET.Element("testsuite", name=SUITE)):
            with self.subTest(root=root.tag), self.assertRaises(ValueError):
                inspect_gate(root)

    def test_rejects_missing_duplicate_and_unexpected_tests(self):
        for mutation in ("missing", "duplicate", "unexpected"):
            root = valid_report()
            suite = root.find("testsuite")
            if mutation == "missing":
                suite.remove(suite.find("testcase"))
            else:
                name = next(iter(TESTS)) if mutation == "duplicate" else "UnknownTest"
                ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_gate(root)

    def test_rejects_skips_failures_errors_and_unexecuted_tests(self):
        for reason in ("skipped", "failure", "error", "notrun", "suppressed"):
            root = valid_report()
            case = root.find("./testsuite/testcase")
            if reason in ("skipped", "failure", "error"):
                ET.SubElement(case, reason)
            elif reason == "notrun":
                case.set("status", "notrun")
            else:
                case.set("result", "suppressed")
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                inspect_gate(root)

    def test_rejects_absent_duplicate_and_nonfinite_metrics(self):
        for value in (None, "duplicate", "nan", "inf", "-0.1", "0.1"):
            root = valid_report()
            props = root.find(f".//testcase[@name='{RECOVERY}']/properties")
            prop = props.find("property")
            if value is None:
                props.remove(prop)
            elif value == "duplicate":
                ET.SubElement(props, "property", **prop.attrib)
            else:
                prop.set("value", value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                inspect_gate(root)

    def test_rejects_invalid_accepted_counts(self):
        for value in ("0", "25", "1.5"):
            root = valid_report()
            prop = root.find(".//property[@name='trial_0_accepted_steps']")
            prop.set("value", value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                inspect_gate(root)


    def test_requires_both_fine_gradient_scales(self):
        for scale in (1, 2):
            for value in (None, "duplicate", "nan", "inf", "-0.1", "0.07", "bad"):
                root = valid_report()
                props = root.find(f".//testcase[@name='{GRADIENT}']/properties")
                prop = props.find(f"property[@name='gradient_relative_error_scale_{scale}']")
                if value is None:
                    props.remove(prop)
                elif value == "duplicate":
                    ET.SubElement(props, "property", **prop.attrib)
                else:
                    prop.set("value", value)
                with self.subTest(scale=scale, value=value), self.assertRaises(ValueError):
                    inspect_gate(root)


if __name__ == "__main__":
    unittest.main()

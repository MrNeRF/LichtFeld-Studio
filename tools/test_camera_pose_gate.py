# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of report validation, not tests of the native renderer."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
import re

from check_camera_pose_gate import GRADIENT, RECOVERY, SUITE, TESTS, inspect_gate
from check_camera_pose_gate import CONTROLLER_RECOVERY, CONTROLLER_SUITE, CONTROLLER_TESTS, inspect_controller_gate
from check_camera_pose_gate import SESSION_SUITE, SESSION_TESTS, inspect_session_gate
from check_camera_pose_gate import require_production_evaluator
from check_camera_pose_gate import TRAINER_SUITE, TRAINER_TESTS, inspect_trainer_gate
from check_camera_pose_gate import REPROJECTION_SUITE, REPROJECTION_TESTS
from check_camera_pose_gate import VIEW_SUITE, VIEW_TESTS, inspect_view_gate
from check_camera_pose_gate import ACTIVATION_SUITE, ACTIVATION_TESTS, inspect_activation_gate
from check_camera_pose_gate import require_no_report_failures
from check_camera_pose_gate import SPARSE_POINT_SUITE, SPARSE_POINT_TESTS, inspect_shared_points_gate
from check_camera_pose_gate import JOINT_SESSION_SUITE, JOINT_SESSION_TESTS, inspect_joint_session_gate
from check_camera_pose_gate import JOINT_INTEGRATION_SUITE, JOINT_INTEGRATION_TESTS, inspect_joint_integration_gate
from check_camera_pose_gate import DIAGNOSTICS_SUITE, DIAGNOSTICS_TESTS, inspect_diagnostics_gate
from check_camera_pose_gate import SCHUR_SUITE, SCHUR_TESTS, inspect_schur_gate
from check_camera_pose_gate import COMBINED_SUITES, inspect_combined_gate


class ReportFailureTests(unittest.TestCase):
    def test_gate_inventory_matches_native_sources(self):
        tests_dir = Path(__file__).resolve().parents[1] / "tests"
        source = "\n".join((tests_dir / name).read_text(encoding="utf-8") for name in (
            "test_camera_pose_controller.cpp", "test_camera_pose_photometric.cpp", "test_camera_pose_session.cpp"))
        found = {}
        for suite, case in re.findall(r"\bTEST(?:_F)?\(\s*(\w+)\s*,\s*(\w+)\s*\)", source):
            found.setdefault(suite, set()).add(case)
        for suite, expected in ((SUITE, TESTS | {CONTROLLER_RECOVERY}),
                                (CONTROLLER_SUITE, CONTROLLER_TESTS), (SESSION_SUITE, SESSION_TESTS),
                                (TRAINER_SUITE, TRAINER_TESTS), (REPROJECTION_SUITE, REPROJECTION_TESTS),
                                (VIEW_SUITE, VIEW_TESTS), (ACTIVATION_SUITE, ACTIVATION_TESTS),
                                (SPARSE_POINT_SUITE, SPARSE_POINT_TESTS), (JOINT_SESSION_SUITE, JOINT_SESSION_TESTS),
                                (JOINT_INTEGRATION_SUITE, JOINT_INTEGRATION_TESTS),
                                (DIAGNOSTICS_SUITE, DIAGNOSTICS_TESTS), (SCHUR_SUITE, SCHUR_TESTS)):
            with self.subTest(suite=suite):
                self.assertEqual(found.get(suite), expected)
        for suite, expected in COMBINED_SUITES.items():
            with self.subTest(suite=suite):
                self.assertEqual(found.get(suite), expected)

    def test_accepts_passing_extra_suite(self):
        root = valid_report()
        suite = ET.SubElement(root, "testsuite", name="Other", failures="0", errors="0")
        ET.SubElement(suite, "testcase", name="Pass", status="run", result="completed")
        require_no_report_failures(root)

    def test_rejects_failure_outside_pose_subset(self):
        for tag in ("failure", "error"):
            with self.subTest(tag=tag):
                root = valid_report()
                suite = ET.SubElement(root, "testsuite", name="TrainerConstructionTest")
                case = ET.SubElement(suite, "testcase", name="Resume")
                ET.SubElement(case, tag)
                with self.assertRaisesRegex(ValueError, "TrainerConstructionTest.Resume"):
                    require_no_report_failures(root)

    def test_rejects_nonzero_or_invalid_summary_counts(self):
        for tag in ("testsuites", "testsuite"):
            for key in ("failures", "errors"):
                for value in ("1", "-1", "invalid"):
                    with self.subTest(tag=tag, key=key, value=value):
                        root = ET.Element(tag, **{key: value})
                        with self.assertRaises(ValueError):
                            require_no_report_failures(root)


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


class CameraPoseSessionGateReportTests(unittest.TestCase):
    def test_requires_production_evaluator_marker(self):
        root = valid_controller_report()
        with self.assertRaises(ValueError):
            require_production_evaluator(root)
        props = root.find(f".//testcase[@name='{CONTROLLER_RECOVERY}']/properties")
        marker = ET.SubElement(props, "property", name="production_evaluator", value="1")
        require_production_evaluator(root)
        marker.set("value", "0")
        with self.assertRaises(ValueError):
            require_production_evaluator(root)
        marker.set("value", "1")
        ET.SubElement(props, "property", **marker.attrib)
        with self.assertRaises(ValueError):
            require_production_evaluator(root)

    def report(self):
        root = valid_controller_report()
        suite = ET.SubElement(root, "testsuite", name=SESSION_SUITE)
        for name in sorted(SESSION_TESTS):
            ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
        return root

    def test_accepts_complete_session_report(self):
        self.assertEqual(inspect_session_gate(self.report())["tests"], 35)

    def test_rejects_missing_session_and_missing_previous_gate(self):
        with self.assertRaises(ValueError):
            inspect_session_gate(valid_controller_report())
        root = self.report()
        root.remove(root.find(f"testsuite[@name='{CONTROLLER_SUITE}']"))
        with self.assertRaises(ValueError):
            inspect_session_gate(root)

    def test_rejects_invalid_session_evidence(self):
        for mutation in ("missing", "duplicate", "unknown", "failure", "error", "skipped", "notrun"):
            root = self.report()
            suite = root.find(f"testsuite[@name='{SESSION_SUITE}']")
            case = suite.find("testcase")
            if mutation == "missing":
                suite.remove(case)
            elif mutation == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif mutation == "unknown":
                case.set("name", "Unknown")
            elif mutation == "notrun":
                case.set("status", "notrun")
            else:
                ET.SubElement(case, mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_session_gate(root)


class CameraPoseTrainerGateReportTests(unittest.TestCase):
    def report(self):
        root = valid_controller_report()
        for suite_name, tests in ((SESSION_SUITE, SESSION_TESTS), (TRAINER_SUITE, TRAINER_TESTS),
                                  (REPROJECTION_SUITE, REPROJECTION_TESTS)):
            suite = ET.SubElement(root, "testsuite", name=suite_name)
            for name in sorted(tests):
                ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
        props = root.find(f".//testcase[@name='{CONTROLLER_RECOVERY}']/properties")
        ET.SubElement(props, "property", name="production_evaluator", value="1")
        return root

    def test_accepts_complete_trainer_report(self):
        result = inspect_trainer_gate(self.report())
        self.assertEqual(result["tests"], 45)
        self.assertTrue(result["sparse_reprojection_contracts"])

    def test_requires_complete_sparse_reprojection_evidence(self):
        for mutation in ("suite", "missing", "duplicate", "failure", "skipped", "notrun"):
            root = self.report()
            suite = root.find(f"testsuite[@name='{REPROJECTION_SUITE}']")
            case = suite.find("testcase")
            if mutation == "suite":
                root.remove(suite)
            elif mutation == "missing":
                suite.remove(case)
            elif mutation == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif mutation == "notrun":
                case.set("status", "notrun")
            else:
                ET.SubElement(case, mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_trainer_gate(root)

    def test_rejects_incomplete_trainer_evidence(self):
        for mutation in ("suite", "missing", "duplicate", "unknown", "failure", "error", "skipped", "notrun"):
            root = self.report()
            suite = root.find(f"testsuite[@name='{TRAINER_SUITE}']")
            case = suite.find("testcase")
            if mutation == "suite":
                root.remove(suite)
            elif mutation == "missing":
                suite.remove(case)
            elif mutation == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif mutation == "unknown":
                case.set("name", "Unknown")
            elif mutation == "notrun":
                case.set("status", "notrun")
            else:
                ET.SubElement(case, mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_trainer_gate(root)


class CameraPoseViewGateReportTests(unittest.TestCase):
    def report(self):
        root = CameraPoseTrainerGateReportTests().report()
        suite = ET.SubElement(root, "testsuite", name=VIEW_SUITE)
        for name in sorted(VIEW_TESTS):
            ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
        return root

    def test_accepts_complete_view_report(self):
        result = inspect_view_gate(self.report())
        self.assertEqual(result["tests"], 48)
        self.assertTrue(result["view_pose_contracts"])

    def test_rejects_incomplete_view_evidence(self):
        for mutation in ("suite", "missing", "duplicate", "unknown", "failure", "error", "skipped", "notrun"):
            root = self.report()
            suite = root.find(f"testsuite[@name='{VIEW_SUITE}']")
            case = suite.find("testcase")
            if mutation == "suite":
                root.remove(suite)
            elif mutation == "missing":
                suite.remove(case)
            elif mutation == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif mutation == "unknown":
                case.set("name", "Unknown")
            elif mutation == "notrun":
                case.set("status", "notrun")
            else:
                ET.SubElement(case, mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_view_gate(root)


class CameraPoseSharedPointGateReportTests(unittest.TestCase):
    def report(self):
        root = CameraPoseActivationGateReportTests().report()
        suite = ET.SubElement(root, "testsuite", name=SPARSE_POINT_SUITE)
        for name in sorted(SPARSE_POINT_TESTS):
            ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
        return root

    def test_accepts_shared_point_contracts_without_claiming_joint_training(self):
        result = inspect_shared_points_gate(self.report())
        self.assertEqual(result["tests"], 56)
        self.assertTrue(result["shared_point_proposals"])
        self.assertFalse(result["joint_training_integrated"])

    def test_rejects_incomplete_shared_point_report(self):
        for mutation in ("suite", "missing", "duplicate", "unknown", "notrun", "failure", "error", "skipped"):
            root = self.report()
            suite = root.find(f"testsuite[@name='{SPARSE_POINT_SUITE}']")
            case = suite.find("testcase")
            if mutation == "suite":
                root.remove(suite)
            elif mutation == "missing":
                suite.remove(case)
            elif mutation == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif mutation == "unknown":
                case.set("name", "Unknown")
            elif mutation == "notrun":
                case.set("status", "notrun")
            else:
                ET.SubElement(case, mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_shared_points_gate(root)


class CameraPoseJointSessionGateReportTests(unittest.TestCase):
    def report(self):
        root = CameraPoseSharedPointGateReportTests().report()
        suite = ET.SubElement(root, "testsuite", name=JOINT_SESSION_SUITE)
        for name in sorted(JOINT_SESSION_TESTS):
            ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
        return root

    def test_accepts_session_contracts_without_claiming_training_integration(self):
        result = inspect_joint_session_gate(self.report())
        self.assertEqual(result["tests"], 63)
        self.assertTrue(result["joint_session_contracts"])
        self.assertFalse(result["joint_training_integrated"])

    def test_rejects_missing_or_incomplete_session_evidence(self):
        for mutation in ("suite", "missing", "duplicate", "unknown", "notrun", "failure", "error", "skipped"):
            root = self.report()
            suite = root.find(f"testsuite[@name='{JOINT_SESSION_SUITE}']")
            case = suite.find("testcase")
            if mutation == "suite":
                root.remove(suite)
            elif mutation == "missing":
                suite.remove(case)
            elif mutation == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif mutation == "unknown":
                case.set("name", "Unknown")
            elif mutation == "notrun":
                case.set("status", "notrun")
            else:
                ET.SubElement(case, mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_joint_session_gate(root)


class CameraPoseJointIntegrationGateReportTests(unittest.TestCase):
    def report(self):
        root = CameraPoseJointSessionGateReportTests().report()
        suite = ET.SubElement(root, "testsuite", name=JOINT_INTEGRATION_SUITE)
        for name in sorted(JOINT_INTEGRATION_TESTS):
            ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
        return root

    def test_accepts_integration_without_claiming_reconstruction_quality(self):
        result = inspect_joint_integration_gate(self.report())
        self.assertEqual(result["tests"], 65)
        self.assertTrue(result["joint_training_integrated"])
        self.assertFalse(result["reconstruction_quality_validated"])

    def test_rejects_missing_or_unexecuted_integration(self):
        with self.assertRaises(ValueError):
            inspect_joint_integration_gate(CameraPoseJointSessionGateReportTests().report())
        for defect in ("skipped", "failure", "error", "notrun", "duplicate", "unknown"):
            root = self.report()
            suite = root.find(f"testsuite[@name='{JOINT_INTEGRATION_SUITE}']")
            case = suite.find("testcase")
            if defect == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif defect == "notrun":
                case.set("status", "notrun")
            elif defect == "unknown":
                case.set("name", "Unknown")
            else:
                ET.SubElement(case, defect)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                inspect_joint_integration_gate(root)


class CameraPoseDiagnosticsGateTests(unittest.TestCase):
    def test_diagnostics_require_complete_executed_inventory(self):
        root = CameraPoseJointIntegrationGateReportTests().report()
        with self.assertRaises(ValueError):
            inspect_diagnostics_gate(root)
        suite = ET.SubElement(root, "testsuite", name=DIAGNOSTICS_SUITE)
        for name in sorted(DIAGNOSTICS_TESTS):
            ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
        result = inspect_diagnostics_gate(root)
        self.assertEqual(result["tests"], 68)
        self.assertTrue(result["diagnostics_contracts"])
        self.assertFalse(result["reconstruction_quality_validated"])
        ET.SubElement(suite.find("testcase"), "skipped")
        with self.assertRaises(ValueError):
            inspect_diagnostics_gate(root)


class CameraPoseSchurGateTests(unittest.TestCase):
    def report(self):
        root = CameraPoseJointIntegrationGateReportTests().report()
        for name, inventory in ((DIAGNOSTICS_SUITE, DIAGNOSTICS_TESTS), (SCHUR_SUITE, SCHUR_TESTS)):
            suite = ET.SubElement(root, "testsuite", name=name)
            for case in sorted(inventory):
                ET.SubElement(suite, "testcase", name=case, status="run", result="completed")
        return root

    def test_requires_complete_schur_contracts_without_claiming_quality(self):
        result = inspect_schur_gate(self.report())
        self.assertEqual(result["tests"], 73)
        self.assertTrue(result["schur_proposal_contracts"])
        self.assertFalse(result["reconstruction_quality_validated"])

    def test_rejects_incomplete_or_unexecuted_schur_suite(self):
        for defect in ("suite", "duplicate_suite", "missing", "duplicate", "unknown", "notrun", "failure", "error", "skipped"):
            root = self.report()
            suite = root.find(f"testsuite[@name='{SCHUR_SUITE}']")
            case = suite.find("testcase")
            if defect == "suite":
                root.remove(suite)
            elif defect == "duplicate_suite":
                root.append(suite)
            elif defect == "missing":
                suite.remove(case)
            elif defect == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif defect == "unknown":
                case.set("name", "Unknown")
            elif defect == "notrun":
                case.set("status", "notrun")
            else:
                ET.SubElement(case, defect)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                inspect_schur_gate(root)


class CameraPoseCombinedGateTests(unittest.TestCase):
    def report(self):
        root = CameraPoseSchurGateTests().report()
        for name, inventory in COMBINED_SUITES.items():
            suite = ET.SubElement(root, "testsuite", name=name)
            for case in sorted(inventory):
                ET.SubElement(suite, "testcase", name=case, status="run", result="completed")
        return root

    def test_combined_contracts_do_not_certify_quality(self):
        result = inspect_combined_gate(self.report())
        self.assertEqual(result["tests"], 81)
        self.assertTrue(result["combined_objective_contracts"])
        self.assertFalse(result["reconstruction_quality_validated"])

    def test_missing_or_unexecuted_combined_cases_fail(self):
        for name in COMBINED_SUITES:
            for defect in ("suite", "duplicate_suite", "missing", "duplicate", "unknown", "notrun", "failure", "error", "skipped"):
                root = self.report()
                suite = root.find(f"testsuite[@name='{name}']")
                case = suite.find("testcase")
                if defect == "suite":
                    root.remove(suite)
                elif defect == "duplicate_suite":
                    root.append(suite)
                elif defect == "missing":
                    suite.remove(case)
                elif defect == "duplicate":
                    suite.append(case)
                elif defect == "unknown":
                    case.set("name", "Unknown")
                elif defect == "notrun":
                    case.set("status", "notrun")
                else:
                    ET.SubElement(case, defect)
                with self.subTest(suite=name, defect=defect), self.assertRaises(ValueError):
                    inspect_combined_gate(root)


class CameraPoseActivationGateReportTests(unittest.TestCase):
    def report(self):
        root = CameraPoseViewGateReportTests().report()
        suite = ET.SubElement(root, "testsuite", name=ACTIVATION_SUITE)
        for name in sorted(ACTIVATION_TESTS):
            ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
        return root

    def test_accepts_complete_activation_report(self):
        result = inspect_activation_gate(self.report())
        self.assertEqual(result["tests"], 52)
        self.assertTrue(result["activation_contracts"])

    def test_rejects_missing_or_failed_activation(self):
        for mutation in ("suite", "missing", "duplicate", "failure", "error", "skipped", "notrun"):
            root = self.report()
            suite = root.find(f"testsuite[@name='{ACTIVATION_SUITE}']")
            case = suite.find("testcase")
            if mutation == "suite":
                root.remove(suite)
            elif mutation == "missing":
                suite.remove(case)
            elif mutation == "duplicate":
                ET.SubElement(suite, "testcase", **case.attrib)
            elif mutation == "notrun":
                case.set("status", "notrun")
            else:
                ET.SubElement(case, mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_activation_gate(root)


class CameraPoseControllerGateReportTests(unittest.TestCase):
    def test_accepts_complete_controller_report(self):
        result = inspect_controller_gate(valid_controller_report())
        self.assertEqual(result["tests"], 22)
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
        self.assertEqual(result["tests"], 12)
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

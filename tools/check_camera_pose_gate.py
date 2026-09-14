#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate a gtest XML report for the first camera-pose photometric gate.

This does not build or run native code. Missing tests, skips, missing recovery
metrics and non-finite results are failures, not evidence of successful recovery.
The XML must come from the current sources; this checker cannot attest provenance.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SUITE = "CameraPosePhotometricTest"
TESTS = frozenset({
    "AnisotropicSH3MatchesSixAxisFiniteDifferences",
    "MipCameraGradientMatchesFiniteDifferences",
    "OnAxisAnisotropicRollHasNonzeroGradient",
    "SHViewDirectionSurvivesGeometryCancellation",
    "CameraOnlyPreservesModelOptimizerAndSource",
    "SparseGuardRejectsDriftBeforePhotometricRendering",
    "SparseGuardSupportsUndistortionWithoutMutatingMeasurements",
    "UndistortionInverseMappingSupportsModelsCropScaleAndInvalidInputs",
    "TiledGradientMatchesFullImage",
    "OptionalGradientPreservesJointGaussianUpdate",
    "RejectsInvalidContractsBeforeBackward",
    "RecoversPerturbedPoseFromImagesWithFixedGeometry",
})
RECOVERY = "RecoversPerturbedPoseFromImagesWithFixedGeometry"
GRADIENT = "AnisotropicSH3MatchesSixAxisFiniteDifferences"
CONTROLLER_RECOVERY = "BoundedControllerRecoversPoseFromImages"
CONTROLLER_SUITE = "CameraPoseControllerTest"
CONTROLLER_TESTS = frozenset({
    "GeometricProposalStillRequiresPhotometricDescentAndBoundedBudget",
    "RejectsInvalidSourcesAndConfiguration",
    "FrozenAnchorAndEvaluationNeverRenderOrMove",
    "StaleAndInvalidGradientsAreRejectedBeforeRendering",
    "RejectsWorseAndNonfiniteLossWithoutMoving",
    "ExceptionRollsBackControllerAndSnapshotsAreValues",
    "CumulativeBoundsSurviveRepeatedStepsAndReset",
    "ChangingModelInvalidatesCurvatureHistory",
    "CumulativeRotationBoundIsIndependentOfTranslation",
})


def inspect_controller_gate(root: ET.Element) -> dict:
    # Check the additional test before extracting the unchanged checkpoint A
    # report. No failure or unknown test is silently filtered from acceptance.
    root = copy.deepcopy(root)
    photo = [s for s in root.iter("testsuite") if s.get("name") == SUITE]
    control = [s for s in root.iter("testsuite") if s.get("name") == CONTROLLER_SUITE]
    if len(photo) != 1 or len(control) != 1:
        raise ValueError("Controller gate requires one photometric and one controller suite")
    recovery = [c for c in photo[0].findall("testcase") if c.get("name") == CONTROLLER_RECOVERY]
    if len(recovery) != 1:
        raise ValueError("Controller photometric recovery test missing or duplicated")
    cases = control[0].findall("testcase")
    names = [c.get("name") for c in cases]
    if len(names) != len(CONTROLLER_TESTS) or set(names) != CONTROLLER_TESTS:
        raise ValueError("Wrong controller test inventory")
    for case in cases + recovery:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Controller test not successfully executed: {case.get('name')}")
    properties = recovery[0].findall("./properties/property")
    values = {p.get("name"): p.get("value") for p in properties}
    if len(values) != len(properties):
        raise ValueError("Duplicate controller recovery properties")
    trials = []
    for trial in range(3):
        metrics = {}
        for metric, upper in (("loss_ratio", 0.10), ("center_error_ratio", 0.30),
                              ("rotation_error_ratio", 0.30), ("accepted_steps", 81),
                              ("candidate_renders", 641)):
            key = f"trial_{trial}_{metric}"
            try:
                value = float(values[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Missing or invalid controller metric: {key}") from error
            if not math.isfinite(value) or not 0 <= value < upper:
                raise ValueError(f"Controller threshold failed: {key}={value}")
            if metric in ("accepted_steps", "candidate_renders") and (value < 1 or not value.is_integer()):
                raise ValueError(f"Invalid controller count: {key}={value}")
            metrics[metric] = value
        if metrics["candidate_renders"] < metrics["accepted_steps"]:
            raise ValueError("Accepted steps exceed candidate renders")
        trials.append(metrics)
    photo[0].remove(recovery[0])
    result = inspect_gate(root)
    result.update(gate="bounded_controller_photometric_recovery", tests=len(TESTS)+len(CONTROLLER_TESTS)+1,
                  controller_trials=trials)
    return result


def inspect_gate(root: ET.Element) -> dict:
    suites = [node for node in root.iter("testsuite") if node.get("name") == SUITE]
    if len(suites) != 1:
        raise ValueError(f"Expected exactly one {SUITE} suite; found {len(suites)}")
    cases = list(suites[0].findall("testcase"))
    names = [case.get("name") for case in cases]
    if len(names) != len(set(names)) or set(names) != TESTS:
        missing = sorted(TESTS - set(names))
        unexpected = sorted(str(name) for name in set(names) - TESTS)
        raise ValueError(f"Wrong test inventory: missing={missing}, unexpected={unexpected}, duplicates={len(names) != len(set(names))}")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or case.find("failure") is not None or case.find("error") is not None
                or case.find("skipped") is not None):
            raise ValueError(f"Test not successfully executed: {case.get('name')}")
    gradient = next(case for case in cases if case.get("name") == GRADIENT)
    gradient_properties = gradient.findall("./properties/property")
    gradient_values = {node.get("name"): node.get("value") for node in gradient_properties}
    if len(gradient_values) != len(gradient_properties):
        raise ValueError("Duplicate gradient properties")
    fine_errors = []
    for scale in (1, 2):
        key = f"gradient_relative_error_scale_{scale}"
        try:
            error = float(gradient_values[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid two-scale gradient evidence: {key}") from exc
        if not math.isfinite(error) or not 0 <= error < 0.07:
            raise ValueError(f"Gradient threshold failed: {key}={error}")
        fine_errors.append(error)
    recovery = next(case for case in cases if case.get("name") == RECOVERY)
    properties = recovery.findall("./properties/property")
    values = {node.get("name"): node.get("value") for node in properties}
    if len(values) != len(properties):
        raise ValueError("Duplicate recovery properties")
    trials = []
    for trial in range(3):
        metrics = {}
        for metric, upper in (("loss_ratio", 0.10), ("center_error_ratio", 0.30),
                              ("rotation_error_ratio", 0.30), ("accepted_steps", 25)):
            key = f"trial_{trial}_{metric}"
            try:
                number = float(values[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Missing or invalid recovery metric: {key}") from error
            if not math.isfinite(number) or not 0 <= number < upper:
                raise ValueError(f"Recovery threshold failed: {key}={number}")
            if metric == "accepted_steps" and (number < 1 or not number.is_integer()):
                raise ValueError(f"Invalid accepted-step count: {number}")
            metrics[metric] = number
        trials.append(metrics)
    return {
        "gate": "fixed_geometry_photometric_recovery",
        "passed": True,
        "tests": len(cases),
        "fine_gradient_relative_errors": fine_errors,
        "trials": trials,
        "scope": "Synthetic fixed geometry only; joint training, real-data quality and viewport are not certified.",
    }


SESSION_SUITE = "CameraPoseSessionTest"
SESSION_TESTS = {
    "GeometricProposalReachesControllerWithoutChangingImageObjective",
    "GeometricRejectionDoesNotRenderOrMoveAndExceptionsRollBack",
    "DeterministicAnchorsExcludeEvaluationAndSortByUid",
    "RejectsAmbiguousMembershipAndDegenerateGauge",
    "ExplicitAnchorsAndIndependentCameraCadence",
    "WarmupPauseAndFinalFreezeDoNotEvaluate",
    "PublishedSnapshotsRemainImmutableAcrossUpdateAndReset",
    "CancellationRollsBackWholeBurstAndPreservesRetryCadence",
    "EvaluatorExceptionRollsBackPoseHistoryAndCadence",
    "RejectionAndZeroGradientDoNotClaimConvergence",
    "DurableStateRoundTripPreservesPosesPauseAndCadence",
    "CorruptDurableStateNeverPartiallyChangesLiveSession",
    "DurableStateCannotMoveAnchorsOrEvaluationCameras",
}


def inspect_session_gate(root: ET.Element) -> dict:
    suites = [node for node in root.iter("testsuite") if node.get("name") == SESSION_SUITE]
    if len(suites) != 1:
        raise ValueError("Expected exactly one camera pose session suite")
    cases = suites[0].findall("testcase")
    names = [case.get("name") for case in cases]
    if len(names) != len(set(names)) or set(names) != SESSION_TESTS:
        raise ValueError("Wrong camera pose session test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Session test not successfully executed: {case.get('name')}")
    result = inspect_controller_gate(root)
    result.update(gate="multi_camera_session_contracts", tests=result["tests"] + len(cases))
    return result


TRAINER_SUITE = "CameraPoseTrainerIntegrationTest"
TRAINER_TESTS = {
    "PhotometricObjectiveMatchesTrainerLoss",
    "CheckpointRestoresPoseStateWithModel",
    "UnsupportedTrainingCombinationsAreExplicit",
}

REPROJECTION_SUITE = "CameraPoseReprojectionTest"
REPROJECTION_TESTS = {
    "GeometricProposalRecoversCoupledRotationAndTranslation",
    "GeometricProposalRejectsMissingAndUnobservableGeometry",
    "AccurateCalibrationRejectsPhotometricDrift",
    "PermitsCorrectionButKeepsImmutableSourceCeiling",
    "ResizeAndWorldTranslationPreserveDecisions",
    "FixedSupportCannotDisappearOrHideBehindOutliers",
    "MissingOrConcentratedEvidenceDoesNotClaimProtection",
}


def inspect_reprojection_gate(root: ET.Element) -> int:
    suites = [node for node in root.iter("testsuite") if node.get("name") == REPROJECTION_SUITE]
    if len(suites) != 1:
        raise ValueError("Missing or duplicated sparse reprojection suite")
    cases = suites[0].findall("testcase")
    if len(cases) != len(REPROJECTION_TESTS) or {case.get("name") for case in cases} != REPROJECTION_TESTS:
        raise ValueError("Wrong sparse reprojection test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Sparse reprojection test not successfully executed: {case.get('name')}")
    return len(cases)


def inspect_trainer_gate(root: ET.Element) -> dict:
    result = inspect_session_gate(root)
    reprojection_tests = inspect_reprojection_gate(root)
    require_production_evaluator(root)
    suites = [node for node in root.iter("testsuite") if node.get("name") == TRAINER_SUITE]
    if len(suites) != 1:
        raise ValueError("Missing or duplicated Trainer integration suite")
    cases = suites[0].findall("testcase")
    names = [case.get("name") for case in cases]
    if len(names) != len(set(names)) or set(names) != TRAINER_TESTS:
        raise ValueError("Wrong Trainer integration test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Trainer integration test not successfully executed: {case.get('name')}")
    result.update(tests=result["tests"] + len(cases) + reprojection_tests, production_evaluator=True,
                  trainer_checkpoint_contracts=True, sparse_reprojection_contracts=True)
    return result


VIEW_SUITE = "CameraPoseViewTest"
VIEW_TESTS = {"CurrentPoseUsesUIDAndKeepsSourceImmutable", "FrustumAndFocusSharePoseAndSceneAxes",
              "DisplacementLabelClearsAndUsesNetMovement"}


def inspect_view_gate(root: ET.Element) -> dict:
    result = inspect_trainer_gate(root)
    suites = [node for node in root.iter("testsuite") if node.get("name") == VIEW_SUITE]
    if len(suites) != 1:
        raise ValueError("Missing or duplicated camera pose view suite")
    cases = suites[0].findall("testcase")
    if len(cases) != len(VIEW_TESTS) or {case.get("name") for case in cases} != VIEW_TESTS:
        raise ValueError("Wrong camera pose view test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Camera pose view test not successfully executed: {case.get('name')}")
    result.update(tests=result["tests"] + len(cases), view_pose_contracts=True)
    return result


ACTIVATION_SUITE = "CameraPoseActivationTest"
ACTIVATION_TESTS = {"ConfigurationPreservesOptInAndRejectsUnsupportedTraining",
                    "CommandLineCapturesOptInAndRejectsConflict",
                    "ScheduleRoundTripValidationAndCliOverrides",
                    "SharedThreeDgsFeaturesRemainCompatible"}


def inspect_activation_gate(root: ET.Element) -> dict:
    result = inspect_view_gate(root)
    suites = [node for node in root.iter("testsuite") if node.get("name") == ACTIVATION_SUITE]
    if len(suites) != 1:
        raise ValueError("Missing or duplicated camera pose activation suite; run a binary compiled from the current sources")
    cases = suites[0].findall("testcase")
    if len(cases) != len(ACTIVATION_TESTS) or {case.get("name") for case in cases} != ACTIVATION_TESTS:
        raise ValueError("Wrong camera pose activation test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Camera pose activation test not successfully executed: {case.get('name')}")
    result.update(tests=result["tests"] + len(cases), activation_contracts=True)
    return result


SPARSE_POINT_SUITE = "CameraPoseSparsePointTest"
SPARSE_POINT_TESTS = {
    "RecoversPointFromMultipleViewsWithoutMutatingInputs",
    "KeepsTrackIdentityAndExcludesEvaluationMeasurements",
    "RejectsAmbiguousTracksAndDegenerateGeometry",
    "BoundsCumulativePointMovementAndRejectsInvalidInputs",
}


def inspect_shared_points_gate(root: ET.Element) -> dict:
    result = inspect_activation_gate(root)
    suites = [s for s in root.iter("testsuite") if s.get("name") == SPARSE_POINT_SUITE]
    if len(suites) != 1:
        raise ValueError("Missing or duplicated shared-point suite")
    cases = suites[0].findall("testcase")
    if len(cases) != len(SPARSE_POINT_TESTS) or {c.get("name") for c in cases} != SPARSE_POINT_TESTS:
        raise ValueError("Wrong shared-point test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Shared-point test not successfully executed: {case.get('name')}")
    result.update(tests=result["tests"] + len(cases), shared_point_proposals=True,
                  joint_training_integrated=False)
    return result


JOINT_SESSION_SUITE = "CameraPoseJointGeometryTest"
JOINT_SESSION_TESTS = {
    "PointOnlyRefinementPersistsAndResetRestoresSource",
    "CancellationAndExceptionsDoNotCommitPointsOrPoses",
    "AcceptedPoseAndSharedPointsRoundTripTogether",
    "ChangedMeasurementsRejectRestoreWithoutMutation",
    "CorruptPointStateCannotPartiallyRestoreSession",
    "LegacyStateCannotSilentlySwitchGeometryModel",
    "EvaluationAndDisabledMeasurementsNeverEnterGraph",
}


def inspect_joint_session_gate(root: ET.Element) -> dict:
    result = inspect_shared_points_gate(root)
    suites = [s for s in root.iter("testsuite") if s.get("name") == JOINT_SESSION_SUITE]
    if len(suites) != 1:
        raise ValueError("Missing or duplicated joint-session suite; use a binary compiled from current sources")
    cases = suites[0].findall("testcase")
    if len(cases) != len(JOINT_SESSION_TESTS) or {c.get("name") for c in cases} != JOINT_SESSION_TESTS:
        raise ValueError("Wrong joint-session test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Joint-session test not successfully executed: {case.get('name')}")
    result.update(tests=result["tests"] + len(cases), joint_session_contracts=True)
    return result


JOINT_INTEGRATION_SUITE = "CameraPoseJointIntegrationTest"
JOINT_INTEGRATION_TESTS = {
    "ProductionEvaluatorCommitsSharedPointsWithoutChangingGaussians",
    "SharedGeometrySurvivesCheckpointEnvelopeAndModelLoad",
}


def inspect_joint_integration_gate(root: ET.Element) -> dict:
    result = inspect_joint_session_gate(root)
    suites = [s for s in root.iter("testsuite") if s.get("name") == JOINT_INTEGRATION_SUITE]
    if len(suites) != 1:
        raise ValueError("Missing or duplicated joint integration suite")
    cases = suites[0].findall("testcase")
    if len(cases) != len(JOINT_INTEGRATION_TESTS) or {c.get("name") for c in cases} != JOINT_INTEGRATION_TESTS:
        raise ValueError("Wrong joint integration test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Joint integration test not successfully executed: {case.get('name')}")
    result.update(tests=result["tests"] + len(cases), joint_training_integrated=True,
                  joint_adapter_checkpoint_contracts=True, reconstruction_quality_validated=False,
                  scope="Synthetic contracts include joint training integration and checkpoint persistence; real-data quality and live viewport behavior are not certified.")
    return result


DIAGNOSTICS_SUITE = "CameraPoseDiagnosticsTest"
DIAGNOSTICS_TESTS = {
    "SeparatesConstraintAndImageRejectionsWithoutChangingCadence",
    "MeasuresPointWorkAndDoesNotPersistDiagnostics",
    "ExceptionsAndCancellationCountWorkWithoutCommitting",
}


def inspect_diagnostics_gate(root: ET.Element) -> dict:
    result = inspect_joint_integration_gate(root)
    suites = [s for s in root.iter("testsuite") if s.get("name") == DIAGNOSTICS_SUITE]
    if len(suites) != 1:
        raise ValueError("Missing or duplicated diagnostics suite")
    cases = suites[0].findall("testcase")
    if len(cases) != len(DIAGNOSTICS_TESTS) or {c.get("name") for c in cases} != DIAGNOSTICS_TESTS:
        raise ValueError("Wrong diagnostics test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Diagnostics test not successfully executed: {case.get('name')}")
    result.update(tests=result["tests"] + len(cases), diagnostics_contracts=True)
    return result


SCHUR_SUITE = "CameraPoseSchurTest"
SCHUR_TESTS = {
    "RecoversPoseWithMovableStructureWithoutMutatingSources",
    "ProposalIsInvariantToWorldUnits",
    "InvalidOrUnobservableGeometryNeverProducesProposal",
    "NormalizedFactorSolvesCoupledSystem",
    "ReducedStepMatchesFullNumericalNormalEquations",
}


def inspect_schur_gate(root: ET.Element) -> dict:
    result = inspect_diagnostics_gate(root)
    suites = [s for s in root.iter("testsuite") if s.get("name") == SCHUR_SUITE]
    if len(suites) != 1:
        raise ValueError("Missing or duplicated Schur proposal suite")
    cases = suites[0].findall("testcase")
    if len(cases) != len(SCHUR_TESTS) or {c.get("name") for c in cases} != SCHUR_TESTS:
        raise ValueError("Wrong Schur proposal test inventory")
    for case in cases:
        if (case.get("status") != "run" or case.get("result") != "completed"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Schur proposal test not successfully executed: {case.get('name')}")
    result.update(tests=result["tests"] + len(cases), schur_proposal_contracts=True,
                  reconstruction_quality_validated=False)
    return result


COMBINED_SUITES = {
    "CameraPoseCombinedObjectiveTest": {
        "AnalyticGradientMatchesLeftRetractionAndResolutionScaling",
        "PhotometricGainCanOutweighReprojectionIncreaseWithoutNestedPointSolves",
        "ReprojectionGainCanOutweighPhotometricIncrease",
        "CoupledDirectionSolvesDampedSystemAndPreservesWorldUnits",
        "ObservationSumAndOnePixelHuberAreNotDatasetAverages",
        "VersionThreePreservesObjectiveAndLegacyRemainsExplicit",
        "CancelledOrInvalidEvaluationNeverCommitsPartialGeometry",
    },
    "CameraPoseCombinedIntegrationTest": {
        "ProductionObjectiveAndVersionThreeCheckpointRoundTrip",
    },
}


def inspect_combined_gate(root: ET.Element) -> dict:
    result = inspect_schur_gate(root)
    for name, inventory in COMBINED_SUITES.items():
        suites = [s for s in root.iter("testsuite") if s.get("name") == name]
        if len(suites) != 1:
            raise ValueError(f"Missing or duplicated combined objective suite: {name}")
        cases = suites[0].findall("testcase")
        if len(cases) != len(inventory) or {c.get("name") for c in cases} != inventory:
            raise ValueError(f"Wrong combined objective test inventory: {name}")
        for case in cases:
            if (case.get("status") != "run" or case.get("result") != "completed"
                    or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
                raise ValueError(f"Combined objective test not successfully executed: {case.get('name')}")
        result["tests"] += len(cases)
    result.update(combined_objective_contracts=True, reconstruction_quality_validated=False)
    return result


def require_production_evaluator(root: ET.Element) -> None:
    properties = root.findall(f".//testcase[@name='{CONTROLLER_RECOVERY}']/properties/property[@name='production_evaluator']")
    if len(properties) != 1 or properties[0].get("value") != "1":
        raise ValueError("Missing production evaluator evidence; rebuild by the user is required for this gate")


def require_no_report_failures(root: ET.Element) -> None:
    """Never certify a subset while the supplied report contains failures."""
    for suite in root.iter("testsuite"):
        for case in suite.findall("testcase"):
            if any(case.find(tag) is not None for tag in ("failure", "error")):
                raise ValueError(f"Report contains failed test: {suite.get('name')}.{case.get('name')}")
    for node in root.iter():
        if node.tag in ("testsuites", "testsuite"):
            for key in ("failures", "errors"):
                if int(node.get(key, "0")) != 0:
                    raise ValueError(f"Report contains {key}: {node.get(key)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="gtest XML generated from the current camera-pose sources")
    parser.add_argument("--controller", action="store_true", help="Require checkpoint B controller tests and image-driven recovery as well")
    parser.add_argument("--session", action="store_true", help="Require controller and multi-camera session contracts")
    parser.add_argument("--evaluator", action="store_true", help="Require session contracts using the production 3DGS pose evaluator")
    parser.add_argument("--trainer", action="store_true", help="Require pose, loss and checkpoint contracts")
    parser.add_argument("--view", action="store_true", help="Require pose, checkpoint and view contracts")
    parser.add_argument("--activation", action="store_true", help="Require pose, checkpoint, view and activation contracts")
    parser.add_argument("--shared-points", action="store_true", help="Also require shared-point proposal contracts; not joint training validation")
    parser.add_argument("--joint-session", action="store_true", help="Also require joint session transactions and checkpoint contracts; not joint training validation")
    parser.add_argument("--joint-integration", action="store_true", help="Also require production joint evaluator and checkpoint envelope contracts; not reconstruction quality validation")
    parser.add_argument("--diagnostics", action="store_true", help="Also require non-persistent diagnostics contracts")
    parser.add_argument("--schur", action="store_true", help="Also require joint Schur proposal contracts; not reconstruction quality validation")
    parser.add_argument("--combined", action="store_true", help="Also require combined objective and version-three checkpoint contracts; not reconstruction quality validation")
    args = parser.parse_args()
    try:
        inspect = inspect_session_gate if args.session or args.evaluator else inspect_controller_gate if args.controller else inspect_gate
        root = ET.parse(args.report).getroot()
        require_no_report_failures(root)
        result = inspect_combined_gate(root) if args.combined else inspect_schur_gate(root) if args.schur else inspect_diagnostics_gate(root) if args.diagnostics else inspect_joint_integration_gate(root) if args.joint_integration else inspect_joint_session_gate(root) if args.joint_session else inspect_shared_points_gate(root) if args.shared_points else inspect_activation_gate(root) if args.activation else inspect_view_gate(root) if args.view else inspect_trainer_gate(root) if args.trainer else inspect(root)
        if args.evaluator:
            require_production_evaluator(root)
            result.update(production_evaluator=True)
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"CAMERA POSE GATE FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

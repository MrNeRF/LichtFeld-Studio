#!/usr/bin/env python3
"""Run deterministic fixture, CLI, recovery, and spec-oracle self-tests."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
from typing import Callable, Sequence

try:
    from .licht_inspect import (
        FormatError,
        UnsupportedNewer,
        WriteRefused,
        assess_write_compatibility,
        classify_open,
        container_to_dict,
        evaluate_recovery,
        extract_payload,
        main as cli_main,
        open_container,
        require_safe_write,
        verify_container,
    )
    from .make_fixtures import build_fixture_bytes
    from .oracle_corpus import PREVIEW_CORRUPTION_CASE_COUNT, iter_oracle_cases
    from .crc32c import crc32c
    from .foreign_preview import read_foreign_preview
except ImportError:  # Direct script execution.
    from licht_inspect import (
        FormatError,
        UnsupportedNewer,
        WriteRefused,
        assess_write_compatibility,
        classify_open,
        container_to_dict,
        evaluate_recovery,
        extract_payload,
        main as cli_main,
        open_container,
        require_safe_write,
        verify_container,
    )
    from make_fixtures import build_fixture_bytes
    from oracle_corpus import PREVIEW_CORRUPTION_CASE_COUNT, iter_oracle_cases
    from crc32c import crc32c
    from foreign_preview import read_foreign_preview


BYTE_ERROR_RE = re.compile(r": 0x[0-9a-f]{16} [^:]+: expected .+, got .+")


class SelfTest:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, label: str, action: Callable[[], str]) -> None:
        try:
            detail = action()
        except Exception as exc:  # Continue to show every fixture class.
            message = f"FAIL {label}: {type(exc).__name__}: {exc}"
            self.failures.append(message)
            print(message)
        else:
            print(f"PASS {label}: {detail}")


def _fixture_check(
    fixture_dir: Path,
    name: str,
    info: dict[str, object],
) -> str:
    path = fixture_dir / name
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != info["sha256"]:
        raise AssertionError(f"sha256 expected {info['sha256']}, got {digest}")
    if len(data) != info["bytes"]:
        raise AssertionError(f"size expected {info['bytes']}, got {len(data)}")
    outcome, detail = classify_open(path)
    if outcome != info["expected_outcome"]:
        raise AssertionError(
            f"outcome expected {info['expected_outcome']}, got {outcome}: {detail}"
        )
    if outcome in ("hard_fail", "repair_only"):
        if not BYTE_ERROR_RE.search(detail):
            raise AssertionError(f"diagnostic is not byte-precise: {detail}")
        return f"{outcome}; byte-precise rejection"

    if outcome == "unsupported_newer":
        try:
            open_container(path)
        except UnsupportedNewer as error:
            inspection = error.inspection
        else:
            raise AssertionError("unsupported-newer fixture opened semantically")
        if inspection.open_state != "unsupported_newer":
            raise AssertionError(
                f"inspection state expected unsupported_newer, got {inspection.open_state}"
            )
        if inspection.selected_head.commit.generation != 1:
            raise AssertionError("inspection did not retain the authoritative commit")
        if not inspection.index.rows:
            raise AssertionError("inspection did not retain the validated index")
        if not BYTE_ERROR_RE.search(detail):
            raise AssertionError(f"diagnostic is not byte-precise: {detail}")
        return "unsupported_newer; headers/index inspectable; semantic open refused"

    container = open_container(path)
    selected_generation = int(outcome.rsplit("_", 1)[1])
    if container.commit.generation != selected_generation:
        raise AssertionError(
            f"selected generation expected {selected_generation}, got {container.commit.generation}"
        )
    reports: list[str] = []
    if info.get("verify"):
        payload_reports = verify_container(container)
        reports.append(f"verify={len(payload_reports)} live payload(s)")
    if info.get("write_expected") == "unsafe":
        compatibility = container.write_compatibility
        if compatibility.safe:
            raise AssertionError("fixture unexpectedly classified write-safe")
        if len(compatibility.reasons) != 2:
            raise AssertionError(
                f"fixture must exercise both independent writer gates, got {compatibility.reasons}"
            )
        try:
            require_safe_write(container)
        except WriteRefused as error:
            if not BYTE_ERROR_RE.search(str(error)):
                raise AssertionError(f"write refusal is not byte-precise: {error}")
        else:
            raise AssertionError("declared-safe write was not refused")

        version_satisfied = assess_write_compatibility(
            container,
            writer_version=(1, 1),
            writer_capabilities=0xFF,
        )
        if version_satisfied.safe or len(version_satisfied.reasons) != 1:
            raise AssertionError(
                "unknown writer capability did not independently force write-unsafe"
            )
        try:
            require_safe_write(
                container,
                writer_version=(1, 1),
                writer_capabilities=0xFF,
            )
        except WriteRefused as error:
            if error.offset != container.commit.offset + 208:
                raise AssertionError(
                    f"writer-capability refusal offset expected commit+208, got 0x{error.offset:x}"
                )
        else:
            raise AssertionError("unknown writer capability did not refuse write")

        capability_satisfied = assess_write_compatibility(
            container,
            writer_version=(1, 0),
            writer_capabilities=0x1FF,
        )
        if capability_satisfied.safe or len(capability_satisfied.reasons) != 1:
            raise AssertionError(
                "min_safe_writer_version did not independently force write-unsafe"
            )
        try:
            require_safe_write(
                container,
                writer_version=(1, 0),
                writer_capabilities=0x1FF,
            )
        except WriteRefused as error:
            if error.offset != container.commit.offset + 188:
                raise AssertionError(
                    f"writer-version refusal offset expected commit+188, got 0x{error.offset:x}"
                )
        else:
            raise AssertionError("newer minimum safe writer did not refuse write")
        reports.append(
            "write_safe=no; version_gate=refused; capability_gate=refused"
        )
    if info["fixture_class"] == "orphan_tail":
        if container.physical_size <= container.commit.committed_file_end:
            raise AssertionError("fixture has no orphan tail")
        reports.append(
            f"ignored_tail={container.physical_size - container.commit.committed_file_end} B"
        )
    if info["fixture_class"] == "torn_head":
        if not any("recovery warning" in warning for warning in container.warnings):
            raise AssertionError("torn head fallback did not surface a recovery warning")
        reports.append("fallback_warning=yes")
    if info["fixture_class"] == "duplicate_slot_write":
        if not any("duplicate-slot" in warning for warning in container.warnings):
            raise AssertionError("duplicate-slot case was not diagnosed")
        reports.append("duplicate_diagnostic=yes")
    if info["fixture_class"] == "tombstone":
        if not any(row.kind_name == "tombstone" for row in container.index.rows):
            raise AssertionError("selected index lacks tombstone row")
        reports.append("tombstone_encoding=yes")
    if info["fixture_class"] == "preview_locator":
        if container.preview is None:
            raise AssertionError("preview fixture has no selected locator")
        foreign_locator, foreign_bytes = read_foreign_preview(path)
        if (
            foreign_locator.offset != container.preview.offset
            or foreign_locator.bytes != container.preview.bytes
            or not foreign_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise AssertionError(
                "foreign preview path disagrees with the full reader"
            )
        reports.append("foreign_preview=yes")
    return f"{outcome}; " + ", ".join(reports)


def _determinism_check(fixture_dir: Path, manifest: dict[str, object]) -> str:
    regenerated, regenerated_manifest = build_fixture_bytes()
    fixture_meta = manifest["fixtures"]
    regenerated_meta = regenerated_manifest["fixtures"]
    assert isinstance(fixture_meta, dict) and isinstance(regenerated_meta, dict)
    if set(regenerated) != set(fixture_meta):
        raise AssertionError(
            f"file set differs: generated={sorted(regenerated)}, manifest={sorted(fixture_meta)}"
        )
    for name, data in regenerated.items():
        on_disk = (fixture_dir / name).read_bytes()
        if data != on_disk:
            raise AssertionError(f"{name} differs from a fresh deterministic generation")
        info = regenerated_meta[name]
        assert isinstance(info, dict)
        if info["sha256"] != fixture_meta[name]["sha256"]:
            raise AssertionError(f"{name} regenerated manifest hash differs")
    return f"{len(regenerated)} binary fixtures reproduce byte-for-byte"


def _cli_surface_check(fixture_dir: Path) -> str:
    source = fixture_dir / "minimal-valid-single-generation.licht"
    container = open_container(source)
    row = next(row for row in container.index.rows if row.kind_name == "live")
    decoded = io.BytesIO()
    written = extract_payload(container, row, decoded)
    if written != row.uncompressed_bytes or not decoded.getvalue().startswith(b"{"):
        raise AssertionError("API extraction did not return the expected JSON payload")
    dumped = container_to_dict(container)
    if dumped["selected"]["generation"] != 1:  # type: ignore[index]
        raise AssertionError("inspect serialization selected wrong generation")

    unsupported = fixture_dir / "unsupported-newer-single-head.licht"
    write_unsafe = fixture_dir / "write-unsafe.licht"
    preview_source = fixture_dir / "preview-locator.licht"

    with tempfile.TemporaryDirectory(prefix="licht-selftest-cli-") as temp_dir:
        output = Path(temp_dir) / "proj.json"
        preview_output = Path(temp_dir) / "preview.png"
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            inspect_rc = cli_main(["inspect", str(source), "--json"])
            verify_rc = cli_main(["verify", str(source), "--json"])
            extract_rc = cli_main(
                [
                    "extract",
                    str(source),
                    "--fourcc",
                    row.fourcc.decode("ascii"),
                    "--uuid",
                    str(row.instance_uuid),
                    "--output",
                    str(output),
                ]
            )
            preview_rc = cli_main(
                [
                    "preview",
                    str(preview_source),
                    "--output",
                    str(preview_output),
                ]
            )
            unsupported_start = sink.tell()
            unsupported_rc = cli_main(["inspect", str(unsupported), "--json"])
            unsupported_dump = sink.getvalue()[unsupported_start:]
            write_unsafe_start = sink.tell()
            write_unsafe_rc = cli_main(["inspect", str(write_unsafe), "--json"])
            write_unsafe_dump = sink.getvalue()[write_unsafe_start:]
        if (inspect_rc, verify_rc, extract_rc, preview_rc) != (0, 0, 0, 0):
            raise AssertionError(
                "CLI return codes "
                f"{(inspect_rc, verify_rc, extract_rc, preview_rc)}: "
                f"{sink.getvalue()}"
            )
        if output.read_bytes() != decoded.getvalue():
            raise AssertionError("CLI extraction differs from API extraction")
        _, preview_bytes = read_foreign_preview(preview_source)
        if preview_output.read_bytes() != preview_bytes:
            raise AssertionError("CLI foreign preview extraction differs")
        if unsupported_rc != 4:
            raise AssertionError(
                f"unsupported inspect expected rc 4, got {unsupported_rc}: {unsupported_dump}"
            )
        unsupported_json = json.loads(unsupported_dump)
        if unsupported_json.get("open_state") != "unsupported_newer":
            raise AssertionError("unsupported inspect omitted terminal state")
        if not unsupported_json.get("heads") or not unsupported_json.get("selected"):
            raise AssertionError("unsupported inspect did not dump heads/commit/index")
        if "terminal_error" not in unsupported_json:
            raise AssertionError("unsupported inspect omitted terminal diagnostic")
        if write_unsafe_rc != 0:
            raise AssertionError(
                f"write-unsafe inspect expected rc 0, got {write_unsafe_rc}: {write_unsafe_dump}"
            )
        write_unsafe_json = json.loads(write_unsafe_dump)
        write_compatibility = write_unsafe_json.get("write_compatibility")
        if not isinstance(write_compatibility, dict) or write_compatibility.get("safe") is not False:
            raise AssertionError("inspect did not classify the file write-unsafe")
    return (
        "inspect, extract, verify, foreign preview, unsupported-newer dump, and write-unsafe "
        "classification passed"
    )


def _recovery_check(fixture_dir: Path) -> str:
    master = fixture_dir / "autosave-master.licht"
    valid = fixture_dir / "autosave-sidecar-valid.licht.autosave"
    stale = fixture_dir / "autosave-sidecar-stale-base.licht.autosave"
    valid_decision = evaluate_recovery(master, [valid])
    if valid_decision.outcome != "offer":
        raise AssertionError(f"valid sidecar expected offer, got {valid_decision}")
    stale_decision = evaluate_recovery(master, [stale])
    if stale_decision.outcome != "none":
        raise AssertionError(f"stale sidecar expected no offer, got {stale_decision}")
    if [candidate.status for candidate in stale_decision.candidates] != ["stale_delete"]:
        raise AssertionError(
            f"stale sidecar expected stale_delete, got {stale_decision.candidates}"
        )
    return "valid sidecar=offer; stale base=ignore/delete"


def _oracle_check(fixture_dir: Path, random_sample: int) -> str:
    counts: dict[str, list[int]] = {}
    total = 0
    with tempfile.TemporaryDirectory(prefix="licht-selftest-oracle-") as temp_dir:
        case_path = Path(temp_dir) / "mutated.licht"
        for case in iter_oracle_cases(
            fixture_dir,
            random_cases=random_sample,
            include_catalog=True,
        ):
            total += 1
            case_path.write_bytes(case.mutated_file)
            actual, detail = classify_open(case_path)
            family = (
                case.case_id.rsplit("-", 1)[-1]
                if case.case_id.startswith("random-")
                else "catalog"
            )
            counts.setdefault(family, [0, 0])[1] += 1
            if actual != case.expected_outcome:
                raise AssertionError(
                    f"{case.case_id}: oracle={case.expected_outcome}, parser={actual}; "
                    f"mutation={case.mutation_description}; parser_detail={detail}"
                )
            if case.verify_expected == "hard_fail" and actual.startswith("open_gen_"):
                try:
                    verify_container(open_container(case_path))
                except FormatError:
                    pass
                else:
                    raise AssertionError(
                        f"{case.case_id}: lazy payload mutation unexpectedly passed full verify"
                    )
            if case.write_expected == "unsafe" and actual.startswith("open_gen_"):
                container = open_container(case_path)
                if container.write_compatibility.safe:
                    raise AssertionError(
                        f"{case.case_id}: oracle requires write-unsafe classification"
                    )
                try:
                    require_safe_write(container)
                except WriteRefused:
                    pass
                else:
                    raise AssertionError(
                        f"{case.case_id}: declared-safe write was not refused"
                    )
            counts[family][0] += 1

    family_summary = ", ".join(
        f"{family}={passed}/{attempted}"
        for family, (passed, attempted) in sorted(counts.items())
    )
    return f"{total} authoritative cases agreed ({family_summary})"


def _independence_check(tool_dir: Path) -> str:
    parser_text = "\n".join(
        (tool_dir / name).read_text(encoding="utf-8")
        for name in ("licht_inspect.py", "geometry_payloads.py")
    )
    forbidden = ("make_fixtures", "src.io", "project_container", "checkpoint_format")
    hits = [token for token in forbidden if token in parser_text]
    if hits:
        raise AssertionError(f"read-only parser references writer/application tokens: {hits}")
    if crc32c(b"123456789") != 0xE3069283:
        raise AssertionError("CRC32c Castagnoli check vector failed")
    return "parser has no fixture-writer/application dependency; CRC32c vector passed"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    parser.add_argument(
        "--oracle-sample",
        type=int,
        default=200,
        help="number of seeded randomized cases in addition to the full catalogue",
    )
    args = parser.parse_args(argv)
    if args.oracle_sample < 23:
        parser.error("--oracle-sample must be at least 23 so every mutation family runs")

    fixture_dir = args.fixture_dir
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    fixtures = manifest["fixtures"]
    assert isinstance(fixtures, dict)
    test = SelfTest()

    print("LichtFeld .licht P0 self-test")
    print(f"fixtures: {fixture_dir}")
    print(f"oracle randomized sample: {args.oracle_sample} (fixed seeds)")

    test.check(
        "independent-parser",
        lambda: _independence_check(Path(__file__).resolve().parent),
    )
    test.check(
        "fixture-generation-determinism",
        lambda: _determinism_check(fixture_dir, manifest),
    )
    for name, raw_info in sorted(fixtures.items()):
        assert isinstance(raw_info, dict)
        fixture_class = str(raw_info["fixture_class"])
        test.check(
            f"fixture {fixture_class}",
            lambda name=name, info=raw_info: _fixture_check(
                fixture_dir, name, info
            ),
        )
    test.check("sidecar-recovery-predicate", lambda: _recovery_check(fixture_dir))
    test.check(
        "CLI inspect/extract/verify/preview",
        lambda: _cli_surface_check(fixture_dir),
    )
    test.check(
        "spec-oracle-corpus",
        lambda: _oracle_check(fixture_dir, args.oracle_sample),
    )

    if test.failures:
        print(f"SELFTEST FAIL: {len(test.failures)} failure(s)")
        return 1
    print(
        f"SELFTEST PASS: {len(fixtures)} fixture classes + recovery + CLI + "
        f"{args.oracle_sample + len(fixtures) + PREVIEW_CORRUPTION_CASE_COUNT} "
        "oracle cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

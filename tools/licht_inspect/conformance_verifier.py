"""Integration between the independent byte-table verifier and main parser."""

from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

try:
    from .conformance_common import CategoryResult, ConformanceConfig, timed_result
    from .licht_inspect import (
        FormatError,
        UnsupportedNewer,
        classify_open,
        container_to_dict,
        open_container,
    )
    from .spec_byte_verifier import derive_byte_table
except ImportError:  # Direct script execution.
    from conformance_common import CategoryResult, ConformanceConfig, timed_result
    from licht_inspect import (
        FormatError,
        UnsupportedNewer,
        classify_open,
        container_to_dict,
        open_container,
    )
    from spec_byte_verifier import derive_byte_table


def run_independent_verifier(
    fixture_dir: Path,
    config: ConformanceConfig,
) -> CategoryResult:
    started = time.monotonic()
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    fixtures = manifest["fixtures"]
    assert isinstance(fixtures, dict)
    field_checks = 0

    for name, raw_info in sorted(fixtures.items()):
        assert isinstance(raw_info, dict)
        path = fixture_dir / name
        view = derive_byte_table(path)
        if not view.superblock_crc_valid:
            raise AssertionError(f"{name}: independent superblock CRC failed")
        if view.project_uuid != uuid.UUID(manifest["fixed_uuids"]["project"]):
            raise AssertionError(f"{name}: independent project UUID mismatch")
        field_checks += 2

        generations = raw_info["generations"]
        assert isinstance(generations, list)
        generation_by_offset = {
            int(generation["commit_offset"]): generation
            for generation in generations
        }
        for head in view.heads:
            if head.blank or head.commit is None:
                continue
            generation = generation_by_offset.get(head.commit.offset)
            if generation is None:
                continue
            if head.commit.generation != int(generation["generation"]):
                raise AssertionError(f"{name}: independent commit generation mismatch")
            if str(head.commit.commit_uuid) != generation["commit_uuid"]:
                raise AssertionError(f"{name}: independent commit UUID mismatch")
            if head.commit.index_offset != int(generation["index_offset"]):
                raise AssertionError(f"{name}: independent index offset mismatch")
            if not head.commit.crc_valid or not head.commit.index_crc_valid:
                raise AssertionError(f"{name}: independent metadata CRC envelope failed")
            if head.commit.index_generation != head.commit.generation:
                raise AssertionError(f"{name}: independent index generation echo mismatch")
            if head.commit.index_commit_uuid != head.commit.commit_uuid:
                raise AssertionError(f"{name}: independent index commit UUID echo mismatch")
            field_checks += 7

        actual, detail = classify_open(path)
        expected = str(raw_info["expected_outcome"])
        if actual != expected:
            raise AssertionError(
                f"{name}: manifest oracle expected {expected}, parser returned {actual}: {detail}"
            )
        try:
            container = open_container(path)
        except UnsupportedNewer as error:
            container = error.inspection
        except FormatError:
            container = None
        if container is None:
            continue
        dumped = container_to_dict(container)
        if dumped["superblock"]["project_uuid"] != str(view.project_uuid):  # type: ignore[index]
            raise AssertionError(f"{name}: parser/byte-table project UUID disagreement")
        if dumped["superblock"]["file_uuid"] != str(view.file_uuid):  # type: ignore[index]
            raise AssertionError(f"{name}: parser/byte-table file UUID disagreement")
        selected_slot = int(dumped["selected"]["slot_id"])  # type: ignore[index]
        raw_head = view.heads[selected_slot]
        if raw_head.head_sequence != dumped["selected"]["head_sequence"]:  # type: ignore[index]
            raise AssertionError(f"{name}: parser/byte-table head sequence disagreement")
        if raw_head.commit is None:
            raise AssertionError(f"{name}: selected parser head has no raw commit")
        if raw_head.commit.index_offset != dumped["selected"]["index_offset"]:  # type: ignore[index]
            raise AssertionError(f"{name}: parser/byte-table selected index disagreement")
        parser_keys = [row.key for row in container.index.rows]
        raw_keys = [row.key for row in raw_head.commit.rows]
        if parser_keys != raw_keys:
            raise AssertionError(f"{name}: parser/byte-table row-key disagreement")
        field_checks += 5 + len(parser_keys)

    return timed_result(
        "independent-verifier",
        started,
        len(fixtures),
        f"fixtures={len(fixtures)}, fixed-field/row cross-checks={field_checks}",
    )

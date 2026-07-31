"""Per-fourcc structural hostility beneath the semantic C++ chapter tests."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import time

try:
    from .conformance_common import CategoryResult, deterministic_uuid, timed_result
    from .licht_inspect import (
        FormatError,
        classify_open,
        extract_payload,
        open_container,
        verify_container,
        verify_json_chapter,
    )
    from .make_fixtures import ChunkSpec, FixtureWriter
except ImportError:  # Direct script execution.
    from conformance_common import CategoryResult, deterministic_uuid, timed_result
    from licht_inspect import (
        FormatError,
        classify_open,
        extract_payload,
        open_container,
        verify_container,
        verify_json_chapter,
    )
    from make_fixtures import ChunkSpec, FixtureWriter


CHAPTER_FOURCCS = (
    b"PROJ",
    b"REFS",
    b"SCNG",
    b"SELM",
    b"PRMS",
    b"CKPT",
    b"PPIS",
    b"SPLT",
    b"PCLD",
    b"MESH",
    b"GUIL",
    b"EDTR",
    b"VIEW",
    b"SEQR",
    b"METR",
    b"THMB",
)
JSON_FOURCCS = (
    b"PROJ",
    b"REFS",
    b"SCNG",
    b"PRMS",
    b"GUIL",
    b"EDTR",
    b"VIEW",
    b"SEQR",
)


def run_chapter_structural_hostility() -> CategoryResult:
    started = time.monotonic()
    cases = 0
    with tempfile.TemporaryDirectory(prefix="licht-chapter-structural-") as root:
        temporary = Path(root)
        for index, fourcc in enumerate(CHAPTER_FOURCCS):
            namespace = 0xC800 + index
            writer = FixtureWriter(
                project_uuid=deterministic_uuid(namespace, 1),
                file_uuid=deterministic_uuid(namespace, 2),
            )
            layout = writer.append_generation(
                commit_uuid=deterministic_uuid(namespace, 3),
                snapshot_uuid=deterministic_uuid(namespace, 4),
                changes=(
                    ChunkSpec(
                        fourcc,
                        deterministic_uuid(namespace, 5),
                        b'{"structural":"payload-crc-witness"}',
                    ),
                ),
                head_slot=0,
                head_sequence=1,
            )
            row = layout.rows[0]
            hostile = bytearray(writer.bytes())
            hostile[row.payload_offset] ^= 0x20
            path = temporary / f"{fourcc.decode('ascii')}-payload-crc.licht"
            path.write_bytes(hostile)

            outcome, _ = classify_open(path)
            if outcome != "open_gen_1":
                raise AssertionError(
                    f"{fourcc.decode()}: metadata open expected open_gen_1, got {outcome}"
                )
            container = open_container(path)
            try:
                verify_container(container)
            except FormatError as error:
                if fourcc.decode("ascii") not in str(error):
                    raise AssertionError(
                        f"{fourcc.decode()}: payload failure lost fourcc context: {error}"
                    ) from error
            else:
                raise AssertionError(
                    f"{fourcc.decode()}: corrupt selected payload verified successfully"
                )
            cases += 1

        for index, fourcc in enumerate(JSON_FOURCCS):
            namespace = 0xC900 + index
            writer = FixtureWriter(
                project_uuid=deterministic_uuid(namespace, 1),
                file_uuid=deterministic_uuid(namespace, 2),
            )
            writer.append_generation(
                commit_uuid=deterministic_uuid(namespace, 3),
                snapshot_uuid=deterministic_uuid(namespace, 4),
                changes=(
                    ChunkSpec(
                        fourcc,
                        deterministic_uuid(namespace, 5),
                        b'{"duplicate":1,"duplicate":2}',
                    ),
                ),
                head_slot=0,
                head_sequence=1,
            )
            path = temporary / f"{fourcc.decode('ascii')}-duplicate-key.licht"
            path.write_bytes(writer.bytes())
            container = open_container(path)
            row = container.index.rows[0]
            payload = io.BytesIO()
            extract_payload(container, row, payload)
            try:
                verify_json_chapter(str(path), row, payload.getvalue())
            except FormatError as error:
                if "duplicate JSON object key" not in str(error):
                    raise AssertionError(
                        f"{fourcc.decode()}: duplicate-key diagnostic disagrees: {error}"
                    ) from error
            else:
                raise AssertionError(
                    f"{fourcc.decode()}: duplicate JSON key was accepted"
                )
            cases += 1

    return timed_result(
        "chapter-structural-hostility",
        started,
        cases,
        f"fourccs={len(CHAPTER_FOURCCS)}, metadata_open={len(CHAPTER_FOURCCS)}, "
        f"payload_refused={len(CHAPTER_FOURCCS)}, duplicate_json_refused={len(JSON_FOURCCS)}",
    )

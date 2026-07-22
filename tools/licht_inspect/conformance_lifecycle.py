"""Randomized append lifecycle and generation-pinned reader properties."""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path
import random
import tempfile
import time

try:
    from .conformance_common import (
        CategoryResult,
        ConformanceConfig,
        deterministic_uuid,
        timed_result,
    )
    from .licht_inspect import (
        ROW_LIVE,
        ROW_TOMBSTONE,
        evaluate_recovery,
        extract_payload,
        open_container,
        verify_container,
    )
    from .make_fixtures import ChunkSpec, FixtureWriter, ROLE_SIDECAR
except ImportError:  # Direct script execution.
    from conformance_common import (
        CategoryResult,
        ConformanceConfig,
        deterministic_uuid,
        timed_result,
    )
    from licht_inspect import (
        ROW_LIVE,
        ROW_TOMBSTONE,
        evaluate_recovery,
        extract_payload,
        open_container,
        verify_container,
    )
    from make_fixtures import ChunkSpec, FixtureWriter, ROLE_SIDECAR


FOURCCS = (b"A000", b"A001", b"PROJ", b"SCNG", b"VIEW", b"Z999")


@dataclasses.dataclass(frozen=True)
class ModelEntry:
    payload: bytes | None
    source_generation: int


@dataclasses.dataclass(frozen=True)
class PinnedState:
    container: object
    generation: int
    row_signature: tuple[tuple[object, ...], ...]


def _row_signature(container) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row.key,
            row.row_kind,
            row.header_offset,
            row.payload_offset,
            row.stored_bytes,
            row.source_generation,
            row.payload_crc32c,
            row.header_crc32c,
        )
        for row in container.index.rows
    )


def _assert_model(container, model: dict[bytes, ModelEntry]) -> int:
    rows = {row.key: row for row in container.index.rows}
    if set(rows) != set(model):
        raise AssertionError(
            f"generation {container.commit.generation}: index keys differ from model; "
            f"expected={sorted(model)}, got={sorted(rows)}"
        )
    payload_checks = 0
    for key, entry in model.items():
        row = rows[key]
        if row.source_generation != entry.source_generation:
            raise AssertionError(
                f"{row.key_text}: source_generation expected "
                f"{entry.source_generation}, got {row.source_generation}"
            )
        if entry.payload is None:
            if row.row_kind != ROW_TOMBSTONE:
                raise AssertionError(f"{row.key_text}: model tombstone became {row.kind_name}")
            continue
        if row.row_kind != ROW_LIVE:
            raise AssertionError(f"{row.key_text}: model live row became {row.kind_name}")
        output = io.BytesIO()
        extract_payload(container, row, output)
        if output.getvalue() != entry.payload:
            raise AssertionError(
                f"{row.key_text}: payload differs from Python-side state model"
            )
        payload_checks += 1
    return payload_checks


def _sidecar_cycle(
    temp_dir: Path,
    master_path: Path,
    master_writer: FixtureWriter,
    scenario: int,
    generation: int,
) -> int:
    live_base_rows = [row for row in master_writer.current_rows.values() if row.row_kind == 0]
    if not live_base_rows:
        return 0
    base_commit = master_writer.generations[-1].commit_uuid
    project_uuid = master_writer.project_uuid
    sidecar_path = temp_dir / f"scenario-{scenario}.licht.autosave"

    def build(replacement: int, autosave_sequence: int) -> FixtureWriter:
        snapshot = deterministic_uuid(0xC100 + scenario, generation * 100 + replacement)
        sidecar = FixtureWriter(
            project_uuid=project_uuid,
            file_uuid=deterministic_uuid(
                0xC200 + scenario, generation * 100 + replacement
            ),
            role=ROLE_SIDECAR,
            base_explicit_commit_uuid=base_commit,
            autosave_sequence=autosave_sequence,
            sidecar_snapshot_uuid=snapshot,
        )
        sidecar.seed_base_refs(live_base_rows)
        chosen = sorted(live_base_rows, key=lambda row: row.key)[0]
        sidecar.append_generation(
            commit_uuid=deterministic_uuid(
                0xC300 + scenario, generation * 100 + replacement
            ),
            snapshot_uuid=snapshot,
            changes=(
                ChunkSpec(
                    chosen.fourcc,
                    chosen.instance_uuid,
                    f"sidecar-s{scenario}-g{generation}-r{replacement}".encode(),
                ),
            ),
            head_slot=0,
            head_sequence=1,
        )
        return sidecar

    first = build(1, generation * 2 - 1)
    sidecar_path.write_bytes(first.bytes())
    opened_first = open_container(sidecar_path)
    if opened_first.commit.generation != 1:
        raise AssertionError("new sidecar did not open generation 1")
    decision = evaluate_recovery(master_path, [sidecar_path])
    if decision.outcome != "offer":
        raise AssertionError(f"new sidecar expected recovery offer, got {decision}")

    replacement = build(2, generation * 2)
    sidecar_path.write_bytes(replacement.bytes())
    opened_replacement = open_container(sidecar_path)
    if opened_replacement.superblock.autosave_sequence != generation * 2:
        raise AssertionError("replacement sidecar sequence was not published")
    decision = evaluate_recovery(master_path, [sidecar_path])
    if decision.outcome != "offer":
        raise AssertionError(f"replacement sidecar expected offer, got {decision}")
    sidecar_path.unlink()
    if sidecar_path.exists():
        raise AssertionError("discarded sidecar still exists")
    return 3


def run_append_lifecycle(config: ConformanceConfig) -> CategoryResult:
    started = time.monotonic()
    step_cases = 0
    payload_checks = 0
    pinned_checks = 0
    sidecar_actions = 0

    with tempfile.TemporaryDirectory(prefix="licht-conformance-lifecycle-") as temp:
        temp_dir = Path(temp)
        for scenario in range(config.lifecycle_sequences):
            rng = random.Random(
                config.seed ^ (scenario * 0x9E3779B97F4A7C15)
            )
            project_uuid = deterministic_uuid(0xB000 + scenario, 1)
            writer = FixtureWriter(
                project_uuid=project_uuid,
                file_uuid=deterministic_uuid(0xB100 + scenario, 2),
            )
            master_path = temp_dir / f"scenario-{scenario}.licht"
            keys = [
                (fourcc, deterministic_uuid(0xB200 + scenario, index + 1))
                for index, fourcc in enumerate(FOURCCS)
            ]
            model: dict[bytes, ModelEntry] = {}
            pinned: list[PinnedState] = []

            for generation in range(1, config.lifecycle_steps + 1):
                changes: list[ChunkSpec] = []
                delete_keys: list[tuple[bytes, object]] = []
                # Roughly one fifth of commits are metadata-only, explicitly
                # exercising unchanged-row carry-forward.
                edit_count = 0 if generation > 1 and rng.random() < 0.2 else rng.randint(1, 3)
                edit_indices = rng.sample(range(len(keys)), edit_count)
                for key_index in edit_indices:
                    fourcc, instance_uuid = keys[key_index]
                    key = fourcc + instance_uuid.bytes
                    if key in model and model[key].payload is not None and rng.random() < 0.28:
                        delete_keys.append((fourcc, instance_uuid))
                        model[key] = ModelEntry(None, generation)
                    else:
                        payload = (
                            f"scenario={scenario};generation={generation};key={key_index};"
                            f"value={rng.getrandbits(64):016x}"
                        ).encode()
                        changes.append(ChunkSpec(fourcc, instance_uuid, payload))
                        model[key] = ModelEntry(payload, generation)

                writer.append_generation(
                    commit_uuid=deterministic_uuid(
                        0xB300 + scenario, generation
                    ),
                    snapshot_uuid=deterministic_uuid(
                        0xB400 + scenario, generation
                    ),
                    changes=tuple(changes),
                    delete_keys=tuple(delete_keys),
                    head_slot=(generation - 1) & 1,
                    head_sequence=generation,
                )
                master_path.write_bytes(writer.bytes())
                container = open_container(master_path)
                if container.commit.generation != generation:
                    raise AssertionError(
                        f"scenario {scenario}: expected generation {generation}, "
                        f"got {container.commit.generation}"
                    )
                payload_checks += _assert_model(container, model)
                step_cases += 1

                # All previously pinned readers retain their parsed table and
                # can still verify their old payload spans after this append.
                for prior in pinned:
                    if prior.container.commit.generation != prior.generation:
                        raise AssertionError("pinned container generation changed")
                    if _row_signature(prior.container) != prior.row_signature:
                        raise AssertionError("pinned index changed after later publication")
                    verify_container(prior.container)
                    pinned_checks += 1
                pinned.append(
                    PinnedState(container, generation, _row_signature(container))
                )

                if generation % 6 == 0:
                    sidecar_actions += _sidecar_cycle(
                        temp_dir,
                        master_path,
                        writer,
                        scenario,
                        generation,
                    )

    total = step_cases + payload_checks + pinned_checks + sidecar_actions
    return timed_result(
        "append-lifecycle",
        started,
        total,
        f"steps={step_cases}, model_payloads={payload_checks}, "
        f"pinned_reads={pinned_checks}, sidecar_actions={sidecar_actions}",
    )

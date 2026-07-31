#!/usr/bin/env python3
"""P8 dual-parser compatibility job and release-corpus manifest writer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Sequence


PROVENANCE_SHA = "8ca8028e6214b1f424c373b24d479cd90ff2e918"
EXPECTED_ARTIFACTS = (
    "save.licht",
    "save.licht.autosave",
    "save-as.licht",
    "compaction.licht",
    "recovered-commit.licht",
    "headless-train-output.licht",
    "foreign-preview.licht",
)
PRODUCER_METADATA: dict[str, dict[str, object]] = {
    "save.licht": {
        "producer": "ProjectDocument::save",
        "reproducible": True,
        "identity_set": "p8-release-v1",
        "reproduction": {
            "project_uuid": "00013880-0000-4000-8000-000000013880",
            "file_uuid": "000138e6-0000-4000-8000-0000000138e6",
            "commit_uuid": "000138e4-0000-4000-8000-0000000138e4",
            "snapshot_uuid": "000138e5-0000-4000-8000-0000000138e5",
            "creation_time_unix_ns": 1735689600000080000,
            "wallclock_unix_ns": 1735689601000080100,
        },
    },
    "save.licht.autosave": {
        "producer": "ProjectDocument::save_autosave",
        "reproducible": True,
        "identity_set": "p8-release-v1",
        "reproduction": {
            "file_uuid": "00013948-0000-4000-8000-000000013948",
            "base_commit_uuid": "000138e4-0000-4000-8000-0000000138e4",
            "commit_uuid": "54671195-b9fb-47b8-9e0a-006a4e6e2961",
            "snapshot_uuid": "00013949-0000-4000-8000-000000013949",
            "autosave_sequence": 1,
            "wallclock_unix_ns": 1785533976918336597,
        },
    },
    "save-as.licht": {
        "producer": "ProjectDocument::save_as",
        "reproducible": True,
        "identity_set": "p8-release-v1",
        "reproduction": {
            "file_uuid": "000139ae-0000-4000-8000-0000000139ae",
            "compaction_commit_uuid": "144956bd-fa0a-4c74-948d-c4395dbfb187",
            "commit_uuid": "000139ac-0000-4000-8000-0000000139ac",
            "snapshot_uuid": "000139ad-0000-4000-8000-0000000139ad",
            "creation_time_unix_ns": 1785533976923675966,
            "wallclock_unix_ns": 1735689601000080300,
        },
    },
    "compaction.licht": {
        "producer": "ProjectWriter::compact",
        "reproducible": True,
        "identity_set": "p8-release-v1",
        "reproduction": {
            "file_uuid": "00013a10-0000-4000-8000-000000013a10",
            "commit_uuid": "00013a11-0000-4000-8000-000000013a11",
            "snapshot_uuid": "00013a12-0000-4000-8000-000000013a12",
            "creation_time_unix_ns": 1735689600000080400,
            "wallclock_unix_ns": 1735689601000080400,
        },
    },
    "recovered-commit.licht": {
        "producer": "materialize_recovered_project",
        "reproducible": True,
        "identity_set": "p8-release-v1",
        "relocked_from_sha256": (
            "311828f4c4b2c22c3eee1d091d859d62b227fc8a18fc090aeeab36343907b1cf"
        ),
        "relock_reason": (
            "Rebuilt through bound-sidecar recovery materialization instead of a "
            "direct CommitKind::Recovered save"
        ),
        "reproduction": {
            "file_uuid": "00013a74-0000-4000-8000-000000013a74",
            "base_commit_uuid": "000138e4-0000-4000-8000-0000000138e4",
            "commit_uuid": "00013a75-0000-4000-8000-000000013a75",
            "snapshot_uuid": "00013949-0000-4000-8000-000000013949",
            "wallclock_unix_ns": 1735689601000080500,
        },
    },
    "headless-train-output.licht": {
        "producer": "headless training final project publish",
        "reproducible": False,
        "reason": (
            "CUDA training tensor bytes are not reproducible across supported GPU and "
            "driver combinations; this row remains SHA-locked and reader-verified"
        ),
    },
    "foreign-preview.licht": {
        "producer": "ProjectDocument::save with foreign preview",
        "reproducible": True,
        "identity_set": "p8-release-v1",
        "reproduction": {
            "project_uuid": "00013ad8-0000-4000-8000-000000013ad8",
            "file_uuid": "00013ae4-0000-4000-8000-000000013ae4",
            "commit_uuid": "00013ae2-0000-4000-8000-000000013ae2",
            "snapshot_uuid": "00013ae3-0000-4000-8000-000000013ae3",
            "creation_time_unix_ns": 1735689600000080600,
            "wallclock_unix_ns": 1735689601000080610,
        },
    },
}
FROZEN_TREE_HASH_FILE = "FROZEN_TREE_SHA256"
STRUCTURAL_DUMP_EXCLUSIONS = (
    "Container.path (the same bytes may be mounted at different locations)",
    "Container.warnings (live-only semantic JSON diagnostics)",
    "WriteCompatibility.reasons (diagnostic wording, not container structure)",
    "HeadAttempt.error text (diagnostic wording, not selected authority structure)",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_root(rows: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["path"])):
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _frozen_tree_files(tagged: Path) -> list[Path]:
    return [
        path
        for path in sorted(tagged.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
        and path.name != FROZEN_TREE_HASH_FILE
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]


def _frozen_tree_sha256(tagged: Path) -> str:
    digest = hashlib.sha256()
    for path in _frozen_tree_files(tagged):
        relative = path.relative_to(tagged).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_tagged_tree(
    repository: Path,
    tagged: Path,
    expected_tree_sha256: str,
    *,
    compare_git: bool,
) -> None:
    actual = _frozen_tree_sha256(tagged)
    if actual != expected_tree_sha256:
        raise AssertionError(
            f"tagged parser tree hash mismatch: expected={expected_tree_sha256} actual={actual}"
        )
    frozen = (tagged / "FROZEN_SHA").read_text(encoding="utf-8").strip()
    if frozen != PROVENANCE_SHA:
        raise AssertionError(
            f"tagged parser provenance mismatch: expected={PROVENANCE_SHA} actual={frozen}"
        )
    if not compare_git:
        return
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{frozen}^{{commit}}"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if available.returncode != 0:
        return
    for module in (path for path in _frozen_tree_files(tagged) if path.suffix == ".py"):
        relative = module.relative_to(tagged).as_posix()
        source = subprocess.run(
            ["git", "show", f"{frozen}:tools/licht_inspect/{relative}"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if source.returncode != 0:
            raise AssertionError(
                f"tagged module has no provenance source at {frozen}: {relative}"
            )
        if source.stdout != module.read_bytes():
            raise AssertionError(
                f"tagged module differs from {frozen}:tools/licht_inspect/{relative}"
            )


def _register_hashes(repository: Path) -> tuple[str, str]:
    register = (repository / "docs/compatibility.md").read_text(encoding="utf-8")
    prefix = "Published-grammar register line: `licht/1.0 "
    line = next((line for line in register.splitlines() if line.startswith(prefix)), None)
    if line is None:
        raise AssertionError("compatibility register lacks the machine-readable 1.0 line")
    fields: dict[str, str] = {}
    for token in line.removeprefix(prefix).removesuffix("`.").split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    manifest = fields.get("manifest", "")
    tagged_tree = fields.get("tagged_tree", "")
    if len(manifest) != 64 or len(tagged_tree) != 64:
        raise AssertionError(
            "compatibility register must record 64-hex manifest and tagged_tree hashes"
        )
    return manifest, tagged_tree


def _live_modules(repository: Path):
    tools = repository / "tools"
    sys.path.insert(0, str(tools))
    try:
        from licht_inspect.licht_inspect import (  # type: ignore
            classify_open,
            open_container,
            verify_container,
        )
        from licht_inspect.spec_byte_verifier import derive_byte_table  # type: ignore
        from licht_inspect.crc32c import crc32c  # type: ignore

        return classify_open, open_container, verify_container, derive_byte_table, crc32c
    finally:
        sys.path.pop(0)


def _capabilities(container: object) -> dict[str, object]:
    commit = container.commit
    return {
        "min_reader": f"{commit.min_reader_version[0]}.{commit.min_reader_version[1]}",
        "min_safe_writer": (
            f"{commit.min_safe_writer_version[0]}.{commit.min_safe_writer_version[1]}"
        ),
        "required_reader": f"0x{commit.reader_capabilities:032x}",
        "required_writer": f"0x{commit.writer_capabilities:032x}",
    }


def write_manifest(repository: Path, note: str) -> str:
    if os.environ.get("RELEASE_FIXTURE_UPDATE") != "1":
        raise RuntimeError("--write-manifest requires RELEASE_FIXTURE_UPDATE=1")
    compatibility = repository / "docs/compatibility.md"
    if not note.strip() or note not in compatibility.read_text(encoding="utf-8"):
        raise RuntimeError("--note must be non-empty and already present in docs/compatibility.md")
    release = repository / "tools/licht_inspect/fixtures/release_corpus"
    manifest_path = release / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_rows = {
        str(row["path"]): row for row in previous.get("fixtures", [])
    }
    producing_commit = _git_head(repository)
    _, open_container, _, _, _ = _live_modules(repository)
    rows: list[dict[str, object]] = []
    for name in EXPECTED_ARTIFACTS:
        path = release / name
        if not path.is_file():
            raise RuntimeError(f"missing production release artifact: {path}")
        container = open_container(path)
        metadata = dict(PRODUCER_METADATA[name])
        digest = _sha256(path)
        if bool(metadata["reproducible"]):
            prior = previous_rows.get(name)
            writer_sha = (
                str(prior["writer_sha"])
                if prior is not None and prior.get("sha256") == digest
                else producing_commit
            )
        else:
            prior = previous_rows.get(name)
            if prior is None or prior.get("sha256") != digest:
                raise RuntimeError(
                    f"non-reproducible release row changed without an existing provenance lock: {name}"
                )
            writer_sha = str(prior["writer_sha"])
        rows.append(
            {
                "path": name,
                "sha256": digest,
                "writer_sha": writer_sha,
                "capabilities": _capabilities(container),
                **metadata,
            }
        )
    root = _manifest_root(rows)
    manifest = {
        "schema": 2,
        "authority_sha": PROVENANCE_SHA,
        "manifest_root_sha256": root,
        "fixtures": rows,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _selected_spec_head(table: object):
    candidates = [
        head
        for head in table.heads
        if not head.blank
        and head.head_crc_valid
        and head.commit is not None
        and head.commit.crc_valid
    ]
    if not candidates:
        raise AssertionError(f"no structurally valid head in {table.path}")
    return max(candidates, key=lambda head: int(head.head_sequence or 0)), candidates


def _mutate_newer(source: Path, destination: Path, capability: bool, crc32c) -> str:
    # The mutation changes only reserved compatibility declarations and repairs
    # the commit/head CRC envelopes. It never assigns a registry bit.
    data = bytearray(source.read_bytes())
    head_candidates: list[tuple[int, int, int]] = []
    for slot, offset in enumerate((4096, 8192)):
        if data[offset : offset + 8] != b"LFSHEAD\x00":
            continue
        sequence = struct.unpack_from("<Q", data, offset + 16)[0]
        commit_offset = struct.unpack_from("<Q", data, offset + 80)[0]
        head_candidates.append((sequence, slot, commit_offset))
    if not head_candidates:
        raise AssertionError(f"no head to mutate in {source}")
    _, slot, commit_offset = max(head_candidates)
    if capability:
        data[commit_offset + 193] |= 0x01  # synthetic reader bit 8
    else:
        struct.pack_into("<HH", data, commit_offset + 184, 1, 1)
    commit_crc = crc32c(bytes(data[commit_offset : commit_offset + 252]))
    struct.pack_into("<I", data, commit_offset + 252, commit_crc)
    head_offset = (4096, 8192)[slot]
    struct.pack_into("<I", data, head_offset + 104, commit_crc)
    head_crc = crc32c(bytes(data[head_offset : head_offset + 4092]))
    struct.pack_into("<I", data, head_offset + 4092, head_crc)
    destination.write_bytes(data)
    return "reader_bit_8" if capability else "min_reader_1.1"


def _run_tagged(repository: Path, paths: Sequence[Path]) -> list[str]:
    tagged = repository / "tools/licht_inspect/tagged/v1_0"
    frozen = (tagged / "FROZEN_SHA").read_text(encoding="utf-8").strip()
    if frozen != PROVENANCE_SHA:
        raise AssertionError(f"tagged parser SHA mismatch: {frozen}")
    live = (repository / "tools/licht_inspect").resolve()
    payload = json.dumps([str(path.resolve()) for path in paths])
    code = r'''
import json, os, pathlib, sys
import licht_inspect
paths = json.loads(os.environ["LFS_COMPAT_PATHS"])
outcomes = [licht_inspect.classify_open(path)[0] for path in paths]
tagged = pathlib.Path(os.environ["LFS_TAGGED_ROOT"]).resolve()
live = pathlib.Path(os.environ["LFS_LIVE_ROOT"]).resolve()
bad = []
for name, module in sorted(sys.modules.items()):
    filename = getattr(module, "__file__", None)
    if not filename:
        continue
    resolved = pathlib.Path(filename).resolve()
    if resolved.is_relative_to(live) and not resolved.is_relative_to(tagged):
        bad.append(f"{name}={resolved}")
if bad:
    raise SystemExit("live parser import leak: " + "; ".join(bad))
print(json.dumps(outcomes))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tagged)
    env["LFS_COMPAT_PATHS"] = payload
    env["LFS_TAGGED_ROOT"] = str(tagged)
    env["LFS_LIVE_ROOT"] = str(live)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tagged,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.strip() or result.stdout.strip())
    return list(json.loads(result.stdout))


def _run_parser_structural_dump(
    repository: Path, paths: Sequence[Path], *, tagged: bool
) -> dict[str, Any]:
    tagged_root = repository / "tools/licht_inspect/tagged/v1_0"
    live_root = repository / "tools/licht_inspect"
    parser_root = tagged_root if tagged else repository / "tools"
    module_name = "licht_inspect" if tagged else "licht_inspect.licht_inspect"
    code = r'''
import dataclasses, hashlib, importlib, json, os, pathlib, sys, uuid

parser = importlib.import_module(os.environ["LFS_PARSER_MODULE"])
paths = json.loads(os.environ["LFS_COMPAT_PATHS"])
append_base = int(getattr(parser, "APPEND_REGION_OFFSET", 12288))

def scalar(value):
    if dataclasses.is_dataclass(value):
        return fields(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): scalar(item) for key, item in value.items()}
    return value

def fields(value, excluded=()):
    return {
        field.name: scalar(getattr(value, field.name))
        for field in dataclasses.fields(value)
        if field.name not in excluded
    }

def append_relative(value):
    return 0 if value == 0 else value - append_base

def dump(path):
    container = parser.open_container(path)
    superblock = fields(container.superblock)
    commit = fields(container.commit)
    for name in ("offset", "parent_commit_offset", "index_offset", "committed_file_end"):
        commit[name] = append_relative(commit[name])
    head = fields(container.selected_head, ("commit", "index"))
    for name in ("commit_offset", "committed_file_end"):
        head[name] = append_relative(head[name])
    rows = []
    first_row_offset = container.index.rows[0].row_offset if container.index.rows else 0
    for row in container.index.rows:
        item = fields(row, ("block_table",))
        item["fourcc"] = row.fourcc.decode("ascii")
        item["row_offset"] = row.row_offset - first_row_offset
        item["header_offset"] = append_relative(row.header_offset)
        item["payload_offset"] = (
            0 if row.payload_offset == 0 else row.payload_offset - row.header_offset
        )
        if row.block_table is None:
            item["block_table"] = None
        else:
            table = fields(row.block_table)
            table["offset"] = append_relative(row.block_table.offset)
            table["payload_offset"] = (
                row.block_table.payload_offset - row.block_table.offset
            )
            item["block_table"] = table
        rows.append(item)
    index = {
        "generation": container.index.generation,
        "commit_uuid": str(container.index.commit_uuid),
        "flags": container.index.flags,
        "stored_sha256": hashlib.sha256(container.index.stored_bytes).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(container.index.uncompressed_bytes).hexdigest(),
        "rows": rows,
    }
    return {
        "classification": parser.classify_open(path)[0],
        "superblock": superblock,
        "head": head,
        "commit": commit,
        "index": index,
        "verification": scalar(parser.verify_container(container)),
    }

dumps = [dump(path) for path in paths]
bad = []
if os.environ.get("LFS_CHECK_TAGGED_IMPORTS") == "1":
    tagged_root = pathlib.Path(os.environ["LFS_TAGGED_ROOT"]).resolve()
    live_root = pathlib.Path(os.environ["LFS_LIVE_ROOT"]).resolve()
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        resolved = pathlib.Path(filename).resolve()
        if resolved.is_relative_to(live_root) and not resolved.is_relative_to(tagged_root):
            bad.append(f"{name}={resolved}")
print(json.dumps({"dumps": dumps, "import_leaks": bad}, sort_keys=True))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(parser_root)
    env["LFS_PARSER_MODULE"] = module_name
    env["LFS_COMPAT_PATHS"] = json.dumps([str(path.resolve()) for path in paths])
    env["LFS_TAGGED_ROOT"] = str(tagged_root.resolve())
    env["LFS_LIVE_ROOT"] = str(live_root.resolve())
    env["LFS_CHECK_TAGGED_IMPORTS"] = "1" if tagged else "0"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tagged_root if tagged else repository,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.strip() or result.stdout.strip())
    payload = json.loads(result.stdout)
    if payload["import_leaks"]:
        raise AssertionError("live parser import leak: " + "; ".join(payload["import_leaks"]))
    return payload


def _run_cpp_reader_job(repository: Path) -> None:
    binary = repository / "build/tests/lichtfeld_tests"
    if not binary.is_file():
        raise AssertionError(
            "C++ compatibility reader is not built; run the guarded build first"
        )
    filters = ":".join(
        (
            "P8CompatibilityTest.ReleaseCorpusCppReaderOpensEveryLockedArtifact",
            "P8CompatibilityTest.V10CorpusAndSyntheticNewerClassificationRefusePayloadAccess",
            "P8CompatibilityTest.ReleaseCorpusProductionPathsReproduceManifestSha256",
        )
    )
    result = subprocess.run(
        [
            "nice",
            "-n",
            "15",
            str(binary),
            f"--gtest_filter={filters}",
            "--gtest_color=no",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "production C++ reader compatibility job failed:\n"
            + (result.stdout + result.stderr).strip()
        )


def verify_frozen_content(repository: Path, *, run_cpp_proof: bool) -> dict[str, Any]:
    head = _git_head(repository)
    release = repository / "tools/licht_inspect/fixtures/release_corpus"
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["fixtures"]
    if manifest.get("schema") != 2:
        raise AssertionError("release manifest must use content-addressed schema 2")
    if manifest.get("authority_sha") != PROVENANCE_SHA:
        raise AssertionError("release manifest provenance SHA mismatch")
    if [row["path"] for row in rows] != list(EXPECTED_ARTIFACTS):
        raise AssertionError("release corpus path/order differs from the seven locked P8 artifacts")
    manifest_root = _manifest_root(rows)
    if manifest["manifest_root_sha256"] != manifest_root:
        raise AssertionError("release manifest root mismatch")
    register_root, register_tree = _register_hashes(repository)
    if register_root != manifest_root:
        raise AssertionError(
            f"compatibility register manifest root mismatch: register={register_root} actual={manifest_root}"
        )
    tagged = repository / "tools/licht_inspect/tagged/v1_0"
    tree_file = (tagged / FROZEN_TREE_HASH_FILE).read_text(encoding="utf-8").strip()
    if tree_file != register_tree:
        raise AssertionError(
            f"compatibility register tagged tree mismatch: register={register_tree} file={tree_file}"
        )
    _verify_tagged_tree(repository, tagged, tree_file, compare_git=True)

    release_paths = [release / row["path"] for row in rows]
    for row, path in zip(rows, release_paths, strict=True):
        expected_metadata = PRODUCER_METADATA[path.name]
        for field, value in expected_metadata.items():
            if row.get(field) != value:
                raise AssertionError(f"release provenance field mismatch for {path.name}: {field}")
        if _sha256(path) != row["sha256"]:
            raise AssertionError(f"SHA lock mismatch for {path.name}")
        writer_sha = row.get("writer_sha")
        if not isinstance(writer_sha, str) or len(writer_sha) != 40:
            raise AssertionError(f"invalid per-row writer provenance for {path.name}")

    with tempfile.TemporaryDirectory(prefix="lfs-p8-release-mutation-") as temporary:
        mutated_release = Path(temporary) / "release_corpus"
        shutil.copytree(release, mutated_release)
        target = mutated_release / str(rows[0]["path"])
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 0x01
        target.write_bytes(payload)
        if all(
            _sha256(mutated_release / str(row["path"])) == row["sha256"]
            for row in rows
        ):
            raise AssertionError("release corpus mutation self-test was not detected")

    with tempfile.TemporaryDirectory(prefix="lfs-p8-tagged-mutation-") as temporary:
        mutated = Path(temporary) / "v1_0"
        shutil.copytree(tagged, mutated)
        target = next(path for path in _frozen_tree_files(mutated) if path.suffix == ".py")
        payload = bytearray(target.read_bytes())
        payload[0] ^= 0x01
        target.write_bytes(payload)
        try:
            _verify_tagged_tree(repository, mutated, tree_file, compare_git=False)
        except AssertionError:
            pass
        else:
            raise AssertionError("tagged parser mutation self-test was not detected")

    if run_cpp_proof:
        _run_cpp_reader_job(repository)
    return {
        "head": head,
        "provenance": PROVENANCE_SHA,
        "manifest_root": manifest_root,
        "tagged_tree": tree_file,
        "rows": rows,
        "release_paths": release_paths,
        "release_mutation_detected": 1,
        "tagged_tree_mutation_detected": 1,
    }


def check_matrix(repository: Path) -> dict[str, Any]:
    frozen = verify_frozen_content(repository, run_cpp_proof=True)
    rows = frozen["rows"]
    release_paths = frozen["release_paths"]

    classify_open, open_container, verify_container, derive_byte_table, crc32c = (
        _live_modules(repository)
    )

    live_outcomes: list[str] = []
    for row, path in zip(rows, release_paths, strict=True):
        container = open_container(path)
        if container.commit.min_reader_version > (1, 0):
            raise AssertionError(f"{path.name} is not a format-1.0-capability file")
        if container.commit.reader_capabilities & ~0xFF:
            raise AssertionError(f"{path.name} assigns a reader capability above bit 7")
        # A successful verifier returns one evidence line per live payload;
        # malformed bytes raise a byte-precise exception.
        verify_container(container)
        outcome = classify_open(path)[0]
        if not outcome.startswith("open_gen_"):
            raise AssertionError(f"live parser refused release file {path.name}: {outcome}")
        live_outcomes.append(outcome)
        table = derive_byte_table(path)
        selected, _ = _selected_spec_head(table)
        if not table.superblock_crc_valid or selected.commit is None or not selected.commit.crc_valid:
            raise AssertionError(f"spec byte verifier rejected {path.name}")

    live_structural = _run_parser_structural_dump(
        repository, release_paths, tagged=False
    )
    tagged_structural = _run_parser_structural_dump(
        repository, release_paths, tagged=True
    )
    live_dumps = live_structural["dumps"]
    tagged_dumps = tagged_structural["dumps"]
    if tagged_dumps != live_dumps:
        raise AssertionError(
            "tagged/live canonical structural dump disagreement:\n"
            f"tagged={json.dumps(tagged_dumps, sort_keys=True)}\n"
            f"live={json.dumps(live_dumps, sort_keys=True)}"
        )
    tagged_outcomes = [item["classification"] for item in tagged_dumps]
    if tagged_outcomes != live_outcomes:
        raise AssertionError(
            f"tagged/live parser classification disagreement: tagged={tagged_outcomes} live={live_outcomes}"
        )

    synthetic_paths: list[Path] = []
    expected_newer: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lfs-p8-newer-") as temporary:
        root = Path(temporary)
        for index, source in enumerate(release_paths):
            destination = root / f"{index:02}-{source.name}"
            _mutate_newer(source, destination, index % 2 == 1, crc32c)
            synthetic_paths.append(destination)
            table = derive_byte_table(source)
            selected, candidates = _selected_spec_head(table)
            has_supported_older = any(
                head.slot_id != selected.slot_id
                and int(head.head_sequence or 0) < int(selected.head_sequence or 0)
                for head in candidates
            )
            expected_newer.append("hard_fail" if has_supported_older else "unsupported_newer")
        live_newer = [classify_open(path)[0] for path in synthetic_paths]
        tagged_newer = _run_tagged(repository, synthetic_paths)
        if live_newer != expected_newer or tagged_newer != expected_newer:
            raise AssertionError(
                "synthetic-newer oracle disagreement: "
                f"expected={expected_newer} live={live_newer} tagged={tagged_newer}"
            )
        for path in synthetic_paths:
            table = derive_byte_table(path)
            selected, _ = _selected_spec_head(table)
            if selected.commit is None or not selected.commit.crc_valid or not selected.head_crc_valid:
                raise AssertionError(f"synthetic CRC repair failed for {path.name}")

    return {
        "head": frozen["head"],
        "provenance": frozen["provenance"],
        "release": len(release_paths),
        "synthetic_newer": len(expected_newer),
        "cpp_release": len(release_paths),
        "cpp_oracles": 3,
        "structural_matches": len(live_dumps),
        "structural_exclusions": len(STRUCTURAL_DUMP_EXCLUSIONS),
        "release_mutation_detected": frozen["release_mutation_detected"],
        "tagged_tree_mutation_detected": frozen["tagged_tree_mutation_detected"],
        "tagged_import_leaks": 0,
        "unclassified": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    try:
        if args.write_manifest:
            root = write_manifest(repository, args.note)
            print(f"release manifest written: root={root}")
        result = check_matrix(repository)
    except Exception as error:
        print(f"compat-matrix FAIL: {error}", file=sys.stderr)
        return 1
    print("compat-matrix PASS: " + " ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

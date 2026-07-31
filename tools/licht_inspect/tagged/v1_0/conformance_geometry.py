"""Independent PCLD/MESH valid and hostile-payload conformance cases."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import time

try:
    from .conformance_common import (
        CategoryResult,
        ConformanceConfig,
        case_deadline,
        deterministic_uuid,
        timed_result,
    )
    from .geometry_payloads import (
        GeometryPayloadError,
        decode_geometry_payload,
    )
    from .licht_inspect import (
        extract_payload,
        main as cli_main,
        open_container,
    )
    from .make_fixtures import ChunkSpec, FixtureWriter
except ImportError:  # Direct script execution.
    from conformance_common import (
        CategoryResult,
        ConformanceConfig,
        case_deadline,
        deterministic_uuid,
        timed_result,
    )
    from geometry_payloads import GeometryPayloadError, decode_geometry_payload
    from licht_inspect import extract_payload, main as cli_main, open_container
    from make_fixtures import ChunkSpec, FixtureWriter


DESCRIPTOR_BYTES = 48


def _align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _attribute_names(names: tuple[str, ...]) -> bytes:
    output = bytearray(struct.pack("<I", len(names)))
    for name in names:
        encoded = name.encode("utf-8")
        output.extend(struct.pack("<H", len(encoded)))
        output.extend(encoded)
    return bytes(output)


def _material(name: str, texture_index: int = 0) -> bytes:
    encoded_name = name.encode("utf-8")
    output = bytearray(struct.pack("<H", len(encoded_name)))
    output.extend(encoded_name)
    output.extend(
        struct.pack(
            "<10f",
            0.25,
            0.5,
            0.75,
            1.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.6,
            0.8,
        )
    )
    output.extend(struct.pack("<5i", *(texture_index for _ in range(5))))
    output.extend(b"\x01" + b"\0" * 7)
    output.extend(b"\0" * (_align(len(output), 8) - len(output)))
    return bytes(output)


def _texture(width: int, height: int, channels: int, pixels: bytes) -> bytes:
    return struct.pack(
        "<IIIQ", width, height, channels, len(pixels)
    ) + pixels


def _build_geometry(
    fourcc: bytes,
    count_a: int,
    count_b: int | None,
    properties: tuple[tuple[str, int, int, int, bytes], ...],
) -> bytes:
    if fourcc == b"PCLD":
        header = bytearray(
            struct.pack(
                "<4sHHQH6x",
                b"LPCD",
                1,
                1,
                count_a,
                len(properties),
            )
        )
    elif fourcc == b"MESH":
        assert count_b is not None
        header = bytearray(
            struct.pack(
                "<4sHHQQH6x",
                b"LMSH",
                1,
                0,
                count_a,
                count_b,
                len(properties),
            )
        )
    else:
        raise ValueError(fourcc)

    descriptor_start = len(header)
    output = header + bytearray(len(properties) * DESCRIPTOR_BYTES)
    output.extend(b"\0" * (_align(len(output)) - len(output)))
    for index, (name, components, dtype, encoding, plane) in enumerate(properties):
        plane_offset = _align(len(output))
        output.extend(b"\0" * (plane_offset - len(output)))
        output.extend(plane)
        descriptor = descriptor_start + index * DESCRIPTOR_BYTES
        encoded_name = name.encode("utf-8")
        if not (1 <= len(encoded_name) <= 16):
            raise ValueError(name)
        output[descriptor : descriptor + len(encoded_name)] = encoded_name
        struct.pack_into(
            "<HHIQQQ",
            output,
            descriptor + 16,
            components,
            dtype,
            encoding,
            plane_offset,
            len(plane),
            0,
        )
    return bytes(output)


def _valid_pcld() -> bytes:
    return _build_geometry(
        b"PCLD",
        2,
        None,
        (
            ("means", 3, 1, 1, struct.pack("<6f", 0, 1, 2, 3, 4, 5)),
            ("colors", 3, 3, 1, bytes((255, 128, 64, 0, 192, 255))),
            (
                "attribute_names",
                1,
                3,
                2,
                _attribute_names(("x", "y", "z", "red", "green", "blue")),
            ),
            ("vendor_score", 1, 4, 77, b"\x10\x20\x30\x40"),
        ),
    )


def _valid_mesh() -> bytes:
    return _build_geometry(
        b"MESH",
        3,
        3,
        (
            (
                "vertices",
                3,
                1,
                1,
                struct.pack("<9f", -1, -1, 0, 1, -1, 0, 0, 1, 0),
            ),
            ("indices", 1, 5, 1, struct.pack("<3I", 0, 1, 2)),
            ("textures", 1, 3, 4, _texture(1, 1, 3, b"\x01\x02\x03")),
            ("materials", 1, 3, 3, _material("plane")),
            ("submeshes", 3, 5, 2, struct.pack("<3I", 0, 3, 0)),
            ("vendor_ids", 1, 5, 91, struct.pack("<3I", 10, 20, 30)),
        ),
    )


def _descriptor(payload: bytes | bytearray, fourcc: str, index: int) -> int:
    return (24 if fourcc == "PCLD" else 32) + index * DESCRIPTOR_BYTES


def _plane_offset(payload: bytes | bytearray, fourcc: str, index: int) -> int:
    return struct.unpack_from("<Q", payload, _descriptor(payload, fourcc, index) + 24)[0]


def _expect_rejected(
    label: str,
    payload: bytes | bytearray,
    fourcc: str,
    config: ConformanceConfig,
) -> None:
    with case_deadline(config.per_case_timeout_seconds):
        try:
            decode_geometry_payload(payload, fourcc, path=f"{label}.bin")
        except GeometryPayloadError as error:
            if error.offset < 0 or not error.field:
                raise AssertionError(
                    f"{label}: rejection is not byte precise: {error}"
                ) from error
        else:
            raise AssertionError(f"{label}: hostile payload was accepted")


def _hostile_cases(
    pcld: bytes,
    mesh: bytes,
) -> tuple[tuple[str, bytes, str], ...]:
    cases: list[tuple[str, bytes, str]] = []

    mutated = bytearray(pcld)
    mutated[18] = 1
    cases.append(("pcld-header-reserved", bytes(mutated), "PCLD"))

    mutated = bytearray(pcld)
    means_offset = _plane_offset(mutated, "PCLD", 0)
    struct.pack_into(
        "<Q", mutated, _descriptor(mutated, "PCLD", 1) + 24, means_offset
    )
    cases.append(("pcld-overlapping-planes", bytes(mutated), "PCLD"))

    mutated = bytearray(pcld)
    struct.pack_into(
        "<Q",
        mutated,
        _descriptor(mutated, "PCLD", 0) + 32,
        len(mutated) + 1,
    )
    cases.append(("pcld-plane-out-of-bounds", bytes(mutated), "PCLD"))

    mutated = bytearray(pcld)
    struct.pack_into("<Q", mutated, 8, (1 << 64) - 1)
    cases.append(("pcld-count-overflow", bytes(mutated), "PCLD"))

    mutated = bytearray(pcld)
    struct.pack_into(
        "<H", mutated, _descriptor(mutated, "PCLD", 0) + 18, 3
    )
    cases.append(("pcld-means-dtype-mismatch", bytes(mutated), "PCLD"))

    mutated = bytearray(pcld)
    struct.pack_into("<Q", mutated, 8, 3)
    cases.append(("pcld-raw-length-mismatch", bytes(mutated), "PCLD"))

    mutated = bytearray(pcld)
    second = _descriptor(mutated, "PCLD", 1)
    mutated[second : second + 16] = mutated[
        _descriptor(mutated, "PCLD", 0) : _descriptor(mutated, "PCLD", 0) + 16
    ]
    cases.append(("pcld-duplicate-property", bytes(mutated), "PCLD"))

    cases.append(("pcld-truncated-table", pcld[:50], "PCLD"))

    mutated = bytearray(mesh)
    mutated[26] = 1
    cases.append(("mesh-header-reserved", bytes(mutated), "MESH"))

    mutated = bytearray(mesh)
    vertices_offset = _plane_offset(mutated, "MESH", 0)
    struct.pack_into(
        "<Q", mutated, _descriptor(mutated, "MESH", 1) + 24, vertices_offset
    )
    cases.append(("mesh-overlapping-planes", bytes(mutated), "MESH"))

    mutated = bytearray(mesh)
    struct.pack_into(
        "<Q",
        mutated,
        _descriptor(mutated, "MESH", 0) + 32,
        len(mutated) + 1,
    )
    cases.append(("mesh-plane-out-of-bounds", bytes(mutated), "MESH"))

    mutated = bytearray(mesh)
    struct.pack_into("<Q", mutated, 8, (1 << 64) - 1)
    cases.append(("mesh-count-overflow", bytes(mutated), "MESH"))

    mutated = bytearray(mesh)
    indices_offset = _plane_offset(mutated, "MESH", 1)
    struct.pack_into("<I", mutated, indices_offset + 8, 3)
    cases.append(("mesh-index-out-of-range", bytes(mutated), "MESH"))

    mutated = bytearray(mesh)
    struct.pack_into("<Q", mutated, 16, 4)
    cases.append(("mesh-non-triangle-index-count", bytes(mutated), "MESH"))

    mutated = bytearray(mesh)
    texture_offset = _plane_offset(mutated, "MESH", 2)
    struct.pack_into("<Q", mutated, texture_offset + 12, 4)
    cases.append(("mesh-texture-size-mismatch", bytes(mutated), "MESH"))

    mutated = bytearray(mesh)
    struct.pack_into(
        "<Q", mutated, _descriptor(mutated, "MESH", 3) + 32, 3
    )
    cases.append(("mesh-truncated-material", bytes(mutated), "MESH"))

    mutated = bytearray(mesh)
    submesh_offset = _plane_offset(mutated, "MESH", 4)
    struct.pack_into("<I", mutated, submesh_offset + 4, 4)
    cases.append(("mesh-submesh-out-of-range", bytes(mutated), "MESH"))

    mutated = bytearray(mesh)
    struct.pack_into(
        "<I", mutated, _descriptor(mutated, "MESH", 4) + 20, 1
    )
    cases.append(("mesh-special-encoding-mismatch", bytes(mutated), "MESH"))

    return tuple(cases)


def _container_and_cli_case(
    temp_dir: Path,
    pcld: bytes,
    mesh: bytes,
) -> int:
    namespace = 0xE300
    project_uuid = deterministic_uuid(namespace, 1)
    pcld_uuid = deterministic_uuid(namespace, 2)
    mesh_uuid = deterministic_uuid(namespace, 3)
    writer = FixtureWriter(
        project_uuid=project_uuid,
        file_uuid=deterministic_uuid(namespace, 4),
    )
    writer.append_generation(
        commit_uuid=deterministic_uuid(namespace, 5),
        snapshot_uuid=deterministic_uuid(namespace, 6),
        changes=(
            ChunkSpec(b"PCLD", pcld_uuid, pcld, tensor=True),
            ChunkSpec(b"MESH", mesh_uuid, mesh, tensor=True),
        ),
        head_slot=0,
        head_sequence=1,
    )
    path = temp_dir / "geometry.licht"
    path.write_bytes(writer.bytes())
    container = open_container(path)
    decoded = 0
    for fourcc, instance_uuid in (("PCLD", pcld_uuid), ("MESH", mesh_uuid)):
        row = container.find_row(fourcc, instance_uuid)
        output = io.BytesIO()
        extract_payload(container, row, output)
        report = decode_geometry_payload(
            output.getvalue(),
            fourcc,
            path=str(path),
            base_offset=row.payload_offset,
        )
        if report["fourcc"] != fourcc:
            raise AssertionError(f"{fourcc}: decoded wrong payload family")
        decoded += 1

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        rc = cli_main(
            [
                "decode",
                str(path),
                "--fourcc",
                "PCLD",
                "--uuid",
                str(pcld_uuid),
                "--json",
            ]
        )
    if rc != 0:
        raise AssertionError(f"decode CLI returned {rc}: {sink.getvalue()}")
    cli_report = json.loads(sink.getvalue())
    if cli_report.get("point_count") != 2:
        raise AssertionError(f"decode CLI emitted wrong report: {cli_report}")
    return decoded + 1


def run_geometry_payloads(config: ConformanceConfig) -> CategoryResult:
    started = time.monotonic()
    pcld = _valid_pcld()
    mesh = _valid_mesh()

    pcld_report = decode_geometry_payload(pcld, "PCLD", path="valid-pcld.bin")
    if (
        pcld_report["point_count"] != 2
        or pcld_report["property_count"] != 4
        or pcld_report["attribute_names"] != ["x", "y", "z", "red", "green", "blue"]
    ):
        raise AssertionError(f"valid PCLD decoded incorrectly: {pcld_report}")
    pcld_properties = pcld_report["properties"]
    assert isinstance(pcld_properties, list)
    if not any(
        prop["name"] == "vendor_score" and prop["known"] is False
        for prop in pcld_properties
    ):
        raise AssertionError("valid PCLD did not retain its opaque property")

    mesh_report = decode_geometry_payload(mesh, "MESH", path="valid-mesh.bin")
    if (
        mesh_report["vertex_count"] != 3
        or mesh_report["triangle_count"] != 1
        or mesh_report["material_names"] != ["plane"]
        or mesh_report["texture_count"] != 1
        or mesh_report["submesh_count"] != 1
    ):
        raise AssertionError(f"valid MESH decoded incorrectly: {mesh_report}")
    mesh_properties = mesh_report["properties"]
    assert isinstance(mesh_properties, list)
    if not any(
        prop["name"] == "vendor_ids" and prop["known"] is False
        for prop in mesh_properties
    ):
        raise AssertionError("valid MESH did not retain its opaque property")

    hostile = _hostile_cases(pcld, mesh)
    for label, payload, fourcc in hostile:
        _expect_rejected(label, payload, fourcc, config)

    with tempfile.TemporaryDirectory(prefix="licht-conformance-geometry-") as temp:
        integration_cases = _container_and_cli_case(Path(temp), pcld, mesh)

    return timed_result(
        "geometry-payloads",
        started,
        2 + len(hostile) + integration_cases,
        (
            f"valid=2, hostile={len(hostile)}, "
            f"container/API/CLI={integration_cases}"
        ),
    )


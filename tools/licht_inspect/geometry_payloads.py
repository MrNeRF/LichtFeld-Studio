"""Independent PCLD/MESH v1 decoder for the read-only .licht inspector."""

from __future__ import annotations

import dataclasses
import sys
from typing import Final


PCLD_MAGIC: Final = b"LPCD"
MESH_MAGIC: Final = b"LMSH"
PAYLOAD_VERSION: Final = 1
PCLD_HEADER_BYTES: Final = 24
MESH_HEADER_BYTES: Final = 32
DESCRIPTOR_BYTES: Final = 48
PROPERTY_ALIGNMENT: Final = 64
U64_MAX: Final = (1 << 64) - 1
I32_MAX: Final = (1 << 31) - 1

DTYPE_BYTES: Final[dict[int, int]] = {
    1: 4,  # f32
    2: 2,  # f16
    3: 1,  # u8
    4: 2,  # u16
    5: 4,  # u32
    6: 4,  # i32
    7: 8,  # f64
}
DTYPE_NAMES: Final[dict[int, str]] = {
    1: "f32",
    2: "f16",
    3: "u8",
    4: "u16",
    5: "u32",
    6: "i32",
    7: "f64",
}

RAW: Final = 1
PCLD_ATTRIBUTE_NAMES: Final = 2
MESH_SUBMESHES: Final = 2
MESH_MATERIALS: Final = 3
MESH_TEXTURES: Final = 4


class GeometryPayloadError(ValueError):
    """A byte-precise semantic error in a PCLD or MESH payload."""

    def __init__(
        self,
        path: str,
        offset: int,
        field: str,
        expected: object,
        got: object,
    ) -> None:
        self.path = path
        self.offset = offset
        self.field = field
        self.expected = expected
        self.got = got
        super().__init__(
            f"{path}: 0x{offset:016x} {field}: "
            f"expected {expected}, got {got}"
        )


@dataclasses.dataclass(frozen=True)
class _Property:
    name: str
    components: int
    dtype: int
    encoding: int
    byte_offset: int
    byte_length: int
    descriptor_offset: int

    def summary(self, known: bool) -> dict[str, object]:
        return {
            "name": self.name,
            "components": self.components,
            "dtype": DTYPE_NAMES[self.dtype],
            "encoding": self.encoding,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
            "known": known,
        }


def _u16(data: memoryview, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "little")


def _u32(data: memoryview, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "little")


def _i32(data: memoryview, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "little", signed=True)


def _u64(data: memoryview, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 8], "little")


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _fail(
    path: str,
    base_offset: int,
    relative_offset: int,
    field: str,
    expected: object,
    got: object,
) -> GeometryPayloadError:
    return GeometryPayloadError(
        path, base_offset + relative_offset, field, expected, got
    )


def _checked_mul(
    lhs: int,
    rhs: int,
    *,
    path: str,
    base_offset: int,
    relative_offset: int,
    field: str,
) -> int:
    if lhs != 0 and rhs > U64_MAX // lhs:
        raise _fail(
            path,
            base_offset,
            relative_offset,
            field,
            "u64 multiplication without overflow",
            f"{lhs} * {rhs}",
        )
    return lhs * rhs


def _raw_length(
    element_count: int,
    prop: _Property,
    *,
    path: str,
    base_offset: int,
) -> int:
    elements = _checked_mul(
        element_count,
        prop.components,
        path=path,
        base_offset=base_offset,
        relative_offset=prop.descriptor_offset + 32,
        field=f"properties[{prop.name}].byte_length",
    )
    return _checked_mul(
        elements,
        DTYPE_BYTES[prop.dtype],
        path=path,
        base_offset=base_offset,
        relative_offset=prop.descriptor_offset + 32,
        field=f"properties[{prop.name}].byte_length",
    )


def _require_shape(
    prop: _Property,
    *,
    components: int,
    dtype: int,
    encoding: int = RAW,
    path: str,
    base_offset: int,
) -> None:
    actual = (prop.components, prop.dtype, prop.encoding)
    expected = (components, dtype, encoding)
    if actual != expected:
        raise _fail(
            path,
            base_offset,
            prop.descriptor_offset + 16,
            f"properties[{prop.name}].descriptor",
            expected,
            actual,
        )


def _parse_properties(
    data: memoryview,
    *,
    table_offset: int,
    property_count: int,
    minimum_plane_offset: int,
    path: str,
    base_offset: int,
) -> list[_Property]:
    table_end = table_offset + property_count * DESCRIPTOR_BYTES
    if table_end > len(data):
        raise _fail(
            path,
            base_offset,
            table_offset,
            "property_table",
            f"end <= payload bytes {len(data)}",
            table_end,
        )

    result: list[_Property] = []
    names: set[str] = set()
    for index in range(property_count):
        offset = table_offset + index * DESCRIPTOR_BYTES
        raw_name = bytes(data[offset : offset + 16])
        zero = raw_name.find(b"\0")
        name_bytes = raw_name if zero < 0 else raw_name[:zero]
        padding = b"" if zero < 0 else raw_name[zero:]
        if not name_bytes:
            raise _fail(
                path,
                base_offset,
                offset,
                f"properties[{index}].name",
                "nonempty UTF-8",
                raw_name,
            )
        if padding and any(padding):
            raise _fail(
                path,
                base_offset,
                offset + zero,
                f"properties[{index}].name_padding",
                "zero",
                padding,
            )
        try:
            name = name_bytes.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise _fail(
                path,
                base_offset,
                offset + error.start,
                f"properties[{index}].name",
                "UTF-8",
                name_bytes,
            ) from error
        if name in names:
            raise _fail(
                path,
                base_offset,
                offset,
                f"properties[{index}].name",
                "unique property name",
                name,
            )
        names.add(name)

        components = _u16(data, offset + 16)
        dtype = _u16(data, offset + 18)
        encoding = _u32(data, offset + 20)
        byte_offset = _u64(data, offset + 24)
        byte_length = _u64(data, offset + 32)
        if components == 0:
            raise _fail(
                path,
                base_offset,
                offset + 16,
                f"properties[{name}].components",
                "nonzero",
                components,
            )
        if dtype not in DTYPE_BYTES:
            raise _fail(
                path,
                base_offset,
                offset + 18,
                f"properties[{name}].dtype",
                "defined dtype 1..7",
                dtype,
            )
        if encoding == 0:
            raise _fail(
                path,
                base_offset,
                offset + 20,
                f"properties[{name}].encoding",
                "nonzero",
                encoding,
            )
        if byte_offset % PROPERTY_ALIGNMENT != 0:
            raise _fail(
                path,
                base_offset,
                offset + 24,
                f"properties[{name}].byte_offset",
                f"{PROPERTY_ALIGNMENT}-byte aligned",
                byte_offset,
            )
        if byte_offset < minimum_plane_offset:
            raise _fail(
                path,
                base_offset,
                offset + 24,
                f"properties[{name}].byte_offset",
                f">= {minimum_plane_offset}",
                byte_offset,
            )
        byte_end = byte_offset + byte_length
        if byte_end > U64_MAX or byte_end > len(data):
            raise _fail(
                path,
                base_offset,
                offset + 32,
                f"properties[{name}].byte_range",
                f"subset of {len(data)}-byte payload",
                (byte_offset, byte_end),
            )
        reserved = bytes(data[offset + 40 : offset + 48])
        if any(reserved):
            raise _fail(
                path,
                base_offset,
                offset + 40,
                f"properties[{name}].reserved",
                "zero",
                reserved,
            )
        result.append(
            _Property(
                name,
                components,
                dtype,
                encoding,
                byte_offset,
                byte_length,
                offset,
            )
        )

    nonempty = sorted(
        (prop for prop in result if prop.byte_length),
        key=lambda prop: prop.byte_offset,
    )
    for prior, current in zip(nonempty, nonempty[1:]):
        prior_end = prior.byte_offset + prior.byte_length
        if current.byte_offset < prior_end:
            raise _fail(
                path,
                base_offset,
                current.descriptor_offset + 24,
                "property_planes",
                "non-overlapping planes",
                f"{prior.name} overlaps {current.name}",
            )
    return result


def _attribute_names(
    plane: memoryview,
    *,
    path: str,
    base_offset: int,
    prop: _Property,
) -> list[str]:
    if len(plane) < 4:
        raise _fail(
            path,
            base_offset,
            prop.byte_offset,
            "properties[attribute_names]",
            "four-byte string count",
            f"{len(plane)} bytes",
        )
    count = _u32(plane, 0)
    if count > (len(plane) - 4) // 2:
        raise _fail(
            path,
            base_offset,
            prop.byte_offset,
            "properties[attribute_names].count",
            "at most one entry per remaining u16 length",
            count,
        )
    cursor = 4
    result: list[str] = []
    for index in range(count):
        if cursor + 2 > len(plane):
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + cursor,
                f"properties[attribute_names][{index}]",
                "u16 length",
                "truncated",
            )
        length = _u16(plane, cursor)
        cursor += 2
        end = cursor + length
        if end > len(plane):
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + cursor,
                f"properties[attribute_names][{index}]",
                f"{length} UTF-8 bytes",
                f"{len(plane) - cursor} remain",
            )
        raw = bytes(plane[cursor:end])
        try:
            result.append(raw.decode("utf-8", "strict"))
        except UnicodeDecodeError as error:
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + cursor + error.start,
                f"properties[attribute_names][{index}]",
                "UTF-8",
                raw,
            ) from error
        cursor = end
    if cursor != len(plane):
        raise _fail(
            path,
            base_offset,
            prop.byte_offset + cursor,
            "properties[attribute_names]",
            "no trailing bytes",
            len(plane) - cursor,
        )
    return result


def _decode_pcld(
    data: memoryview,
    *,
    path: str,
    base_offset: int,
) -> dict[str, object]:
    if len(data) < PCLD_HEADER_BYTES:
        raise _fail(
            path,
            base_offset,
            0,
            "PCLD.header",
            f"at least {PCLD_HEADER_BYTES} bytes",
            len(data),
        )
    if bytes(data[:4]) != PCLD_MAGIC:
        raise _fail(
            path, base_offset, 0, "PCLD.magic", PCLD_MAGIC, bytes(data[:4])
        )
    version = _u16(data, 4)
    if version != PAYLOAD_VERSION:
        raise _fail(
            path, base_offset, 4, "PCLD.payload_version", PAYLOAD_VERSION, version
        )
    coord_encoding = _u16(data, 6)
    if coord_encoding != 1:
        raise _fail(
            path, base_offset, 6, "PCLD.coord_encoding", 1, coord_encoding
        )
    point_count = _u64(data, 8)
    property_count = _u16(data, 16)
    if any(data[18:24]):
        raise _fail(
            path,
            base_offset,
            18,
            "PCLD.reserved",
            "zero",
            bytes(data[18:24]),
        )
    minimum_plane_offset = _align(
        PCLD_HEADER_BYTES + property_count * DESCRIPTOR_BYTES,
        PROPERTY_ALIGNMENT,
    )
    properties = _parse_properties(
        data,
        table_offset=PCLD_HEADER_BYTES,
        property_count=property_count,
        minimum_plane_offset=minimum_plane_offset,
        path=path,
        base_offset=base_offset,
    )
    by_name = {prop.name: prop for prop in properties}
    if "means" not in by_name:
        raise _fail(
            path,
            base_offset,
            16,
            "PCLD.properties",
            "required means property",
            sorted(by_name),
        )

    known_names = {
        "means",
        "colors",
        "normals",
        "sh0",
        "shN",
        "opacity",
        "scaling",
        "rotation",
        "attribute_names",
    }
    attribute_names: list[str] | None = None
    for prop in properties:
        if prop.name == "means":
            _require_shape(
                prop,
                components=3,
                dtype=1,
                path=path,
                base_offset=base_offset,
            )
        elif prop.name == "colors":
            if (
                prop.components != 3
                or prop.dtype not in (1, 3)
                or prop.encoding != RAW
            ):
                raise _fail(
                    path,
                    base_offset,
                    prop.descriptor_offset + 16,
                    "properties[colors].descriptor",
                    "raw f32x3 or u8x3",
                    (prop.components, prop.dtype, prop.encoding),
                )
        elif prop.name in ("normals", "scaling"):
            _require_shape(
                prop,
                components=3,
                dtype=1,
                path=path,
                base_offset=base_offset,
            )
        elif prop.name == "opacity":
            _require_shape(
                prop,
                components=1,
                dtype=1,
                path=path,
                base_offset=base_offset,
            )
        elif prop.name == "rotation":
            _require_shape(
                prop,
                components=4,
                dtype=1,
                path=path,
                base_offset=base_offset,
            )
        elif prop.name in ("sh0", "shN"):
            if (
                prop.components % 3 != 0
                or prop.dtype != 1
                or prop.encoding != RAW
            ):
                raise _fail(
                    path,
                    base_offset,
                    prop.descriptor_offset + 16,
                    f"properties[{prop.name}].descriptor",
                    "raw f32 components in RGB triples",
                    (prop.components, prop.dtype, prop.encoding),
                )
        elif prop.name == "attribute_names":
            _require_shape(
                prop,
                components=1,
                dtype=3,
                encoding=PCLD_ATTRIBUTE_NAMES,
                path=path,
                base_offset=base_offset,
            )
            plane = data[prop.byte_offset : prop.byte_offset + prop.byte_length]
            attribute_names = _attribute_names(
                plane, path=path, base_offset=base_offset, prop=prop
            )

        if prop.encoding == RAW:
            expected = _raw_length(
                point_count, prop, path=path, base_offset=base_offset
            )
            if prop.byte_length != expected:
                raise _fail(
                    path,
                    base_offset,
                    prop.descriptor_offset + 32,
                    f"properties[{prop.name}].byte_length",
                    expected,
                    prop.byte_length,
                )

    result: dict[str, object] = {
        "fourcc": "PCLD",
        "payload_version": version,
        "coord_encoding": "F32_ABSOLUTE_LOCAL",
        "point_count": point_count,
        "property_count": property_count,
        "properties": [
            prop.summary(prop.name in known_names) for prop in properties
        ],
    }
    if attribute_names is not None:
        result["attribute_names"] = attribute_names
    return result


def _decode_materials(
    plane: memoryview,
    *,
    path: str,
    base_offset: int,
    prop: _Property,
) -> tuple[list[str], list[tuple[int, ...]]]:
    cursor = 0
    names: list[str] = []
    texture_indices: list[tuple[int, ...]] = []
    while cursor < len(plane):
        if cursor + 2 > len(plane):
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + cursor,
                "materials.name_bytes",
                "u16",
                "truncated",
            )
        name_bytes = _u16(plane, cursor)
        record_end = cursor + 70 + name_bytes
        if record_end > len(plane):
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + cursor,
                "materials.record",
                f"end <= {len(plane)}",
                record_end,
            )
        raw_name = bytes(plane[cursor + 2 : cursor + 2 + name_bytes])
        try:
            names.append(raw_name.decode("utf-8", "strict"))
        except UnicodeDecodeError as error:
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + cursor + 2 + error.start,
                "materials.name",
                "UTF-8",
                raw_name,
            ) from error
        texture_offset = cursor + 2 + name_bytes + 40
        texture_indices.append(
            tuple(_i32(plane, texture_offset + index * 4) for index in range(5))
        )
        double_sided_offset = texture_offset + 20
        double_sided = int(plane[double_sided_offset])
        reserved = bytes(
            plane[double_sided_offset + 1 : double_sided_offset + 8]
        )
        if double_sided > 1 or any(reserved):
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + double_sided_offset,
                "materials.boolean_reserved",
                "boolean followed by seven zero bytes",
                bytes(plane[double_sided_offset : double_sided_offset + 8]),
            )
        aligned_end = _align(record_end, 8)
        if aligned_end > len(plane) or any(plane[record_end:aligned_end]):
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + record_end,
                "materials.padding",
                "zero padding to eight-byte alignment",
                bytes(plane[record_end:min(aligned_end, len(plane))]),
            )
        cursor = aligned_end
    return names, texture_indices


def _decode_textures(
    plane: memoryview,
    *,
    path: str,
    base_offset: int,
    prop: _Property,
) -> list[dict[str, int]]:
    cursor = 0
    result: list[dict[str, int]] = []
    while cursor < len(plane):
        if cursor + 20 > len(plane):
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + cursor,
                "textures.record_header",
                "20 bytes",
                f"{len(plane) - cursor} remain",
            )
        width = _u32(plane, cursor)
        height = _u32(plane, cursor + 4)
        channels = _u32(plane, cursor + 8)
        pixel_bytes = _u64(plane, cursor + 12)
        pixels = _checked_mul(
            width,
            height,
            path=path,
            base_offset=base_offset,
            relative_offset=prop.byte_offset + cursor + 12,
            field="textures.pixel_bytes",
        )
        pixels = _checked_mul(
            pixels,
            channels,
            path=path,
            base_offset=base_offset,
            relative_offset=prop.byte_offset + cursor + 12,
            field="textures.pixel_bytes",
        )
        if pixels != pixel_bytes:
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + cursor + 12,
                "textures.pixel_bytes",
                pixels,
                pixel_bytes,
            )
        record_end = cursor + 20 + pixel_bytes
        if record_end > len(plane):
            raise _fail(
                path,
                base_offset,
                prop.byte_offset + cursor,
                "textures.record",
                f"end <= {len(plane)}",
                record_end,
            )
        result.append(
            {
                "width": width,
                "height": height,
                "channels": channels,
                "pixel_bytes": pixel_bytes,
            }
        )
        cursor = record_end
    return result


def _decode_mesh(
    data: memoryview,
    *,
    path: str,
    base_offset: int,
) -> dict[str, object]:
    if len(data) < MESH_HEADER_BYTES:
        raise _fail(
            path,
            base_offset,
            0,
            "MESH.header",
            f"at least {MESH_HEADER_BYTES} bytes",
            len(data),
        )
    if bytes(data[:4]) != MESH_MAGIC:
        raise _fail(
            path, base_offset, 0, "MESH.magic", MESH_MAGIC, bytes(data[:4])
        )
    version = _u16(data, 4)
    if version != PAYLOAD_VERSION:
        raise _fail(
            path, base_offset, 4, "MESH.payload_version", PAYLOAD_VERSION, version
        )
    if _u16(data, 6) != 0 or any(data[26:32]):
        raise _fail(
            path,
            base_offset,
            6,
            "MESH.reserved",
            "zero",
            bytes(data[6:8]) + bytes(data[26:32]),
        )
    vertex_count = _u64(data, 8)
    index_count = _u64(data, 16)
    if index_count % 3:
        raise _fail(
            path,
            base_offset,
            16,
            "MESH.index_count",
            "divisible by three",
            index_count,
        )
    property_count = _u16(data, 24)
    minimum_plane_offset = _align(
        MESH_HEADER_BYTES + property_count * DESCRIPTOR_BYTES,
        PROPERTY_ALIGNMENT,
    )
    properties = _parse_properties(
        data,
        table_offset=MESH_HEADER_BYTES,
        property_count=property_count,
        minimum_plane_offset=minimum_plane_offset,
        path=path,
        base_offset=base_offset,
    )
    by_name = {prop.name: prop for prop in properties}
    if "vertices" not in by_name or "indices" not in by_name:
        raise _fail(
            path,
            base_offset,
            24,
            "MESH.properties",
            "vertices and indices",
            sorted(by_name),
        )

    known_shapes = {
        "vertices": (3, 1),
        "normals": (3, 1),
        "tangents": (4, 1),
        "texcoords": (2, 1),
        "colors": (4, 1),
        "indices": (1, 5),
    }
    known_names = set(known_shapes) | {"submeshes", "materials", "textures"}
    material_names: list[str] = []
    material_texture_indices: list[tuple[int, ...]] = []
    textures: list[dict[str, int]] = []
    for prop in properties:
        if prop.encoding == RAW:
            element_count = index_count if prop.name == "indices" else vertex_count
            expected = _raw_length(
                element_count, prop, path=path, base_offset=base_offset
            )
            if prop.byte_length != expected:
                raise _fail(
                    path,
                    base_offset,
                    prop.descriptor_offset + 32,
                    f"properties[{prop.name}].byte_length",
                    expected,
                    prop.byte_length,
                )
        if prop.name in known_shapes:
            components, dtype = known_shapes[prop.name]
            _require_shape(
                prop,
                components=components,
                dtype=dtype,
                path=path,
                base_offset=base_offset,
            )
        elif prop.name == "submeshes":
            _require_shape(
                prop,
                components=3,
                dtype=5,
                encoding=MESH_SUBMESHES,
                path=path,
                base_offset=base_offset,
            )
            if prop.byte_length % 12:
                raise _fail(
                    path,
                    base_offset,
                    prop.descriptor_offset + 32,
                    "submeshes.byte_length",
                    "multiple of 12",
                    prop.byte_length,
                )
        elif prop.name == "materials":
            _require_shape(
                prop,
                components=1,
                dtype=3,
                encoding=MESH_MATERIALS,
                path=path,
                base_offset=base_offset,
            )
            plane = data[prop.byte_offset : prop.byte_offset + prop.byte_length]
            material_names, material_texture_indices = _decode_materials(
                plane, path=path, base_offset=base_offset, prop=prop
            )
        elif prop.name == "textures":
            _require_shape(
                prop,
                components=1,
                dtype=3,
                encoding=MESH_TEXTURES,
                path=path,
                base_offset=base_offset,
            )
            plane = data[prop.byte_offset : prop.byte_offset + prop.byte_length]
            textures = _decode_textures(
                plane, path=path, base_offset=base_offset, prop=prop
            )

    indices = by_name["indices"]
    index_plane = data[
        indices.byte_offset : indices.byte_offset + indices.byte_length
    ]
    for index in range(index_count):
        stored = _u32(index_plane, index * 4)
        if stored > I32_MAX or stored >= vertex_count:
            raise _fail(
                path,
                base_offset,
                indices.byte_offset + index * 4,
                f"indices[{index}]",
                f"runtime i32 in [0, {vertex_count})",
                stored,
            )

    texture_count = len(textures)
    for material_index, indices_tuple in enumerate(material_texture_indices):
        for field_index, texture_index in enumerate(indices_tuple):
            if texture_index < -1 or texture_index >= texture_count:
                raise _fail(
                    path,
                    base_offset,
                    by_name["materials"].byte_offset,
                    f"materials[{material_index}].texture[{field_index}]",
                    f"[-1, {texture_count})",
                    texture_index,
                )

    if "submeshes" in by_name:
        submeshes = by_name["submeshes"]
        plane = data[
            submeshes.byte_offset : submeshes.byte_offset + submeshes.byte_length
        ]
        for offset in range(0, len(plane), 12):
            start_index = _u32(plane, offset)
            count = _u32(plane, offset + 4)
            material_index = _u32(plane, offset + 8)
            if (
                start_index + count > index_count
                or not material_names
                or material_index >= len(material_names)
            ):
                raise _fail(
                    path,
                    base_offset,
                    submeshes.byte_offset + offset,
                    "submeshes.record",
                    (
                        f"range within {index_count} indices and material "
                        f"within {len(material_names)}"
                    ),
                    (start_index, count, material_index),
                )

    return {
        "fourcc": "MESH",
        "payload_version": version,
        "vertex_count": vertex_count,
        "index_count": index_count,
        "triangle_count": index_count // 3,
        "property_count": property_count,
        "properties": [
            prop.summary(prop.name in known_names) for prop in properties
        ],
        "material_count": len(material_names),
        "material_names": material_names,
        "texture_count": texture_count,
        "textures": textures,
        "submesh_count": (
            by_name["submeshes"].byte_length // 12
            if "submeshes" in by_name
            else 0
        ),
    }


def decode_geometry_payload(
    payload: bytes | bytearray | memoryview,
    fourcc: str | bytes,
    *,
    path: str = "<payload>",
    base_offset: int = 0,
) -> dict[str, object]:
    """Decode and validate one uncompressed PCLD or MESH v1 payload."""

    if isinstance(fourcc, bytes):
        try:
            fourcc_text = fourcc.decode("ascii", "strict")
        except UnicodeDecodeError as error:
            raise ValueError("geometry fourcc must be ASCII PCLD or MESH") from error
    else:
        fourcc_text = fourcc
    data = memoryview(payload).cast("B")
    if len(data) > sys.maxsize:
        raise GeometryPayloadError(
            path,
            base_offset,
            f"{fourcc_text}.payload_size",
            f"<= process size_t {sys.maxsize}",
            len(data),
        )
    if fourcc_text == "PCLD":
        return _decode_pcld(data, path=path, base_offset=base_offset)
    if fourcc_text == "MESH":
        return _decode_mesh(data, path=path, base_offset=base_offset)
    raise ValueError("geometry fourcc must be PCLD or MESH")

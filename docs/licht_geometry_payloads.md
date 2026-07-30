# `PCLD` and `MESH` Payload Layouts, Version 1

Status: **normative** for the chunk payloads named `PCLD` and `MESH` (owner decision 2026-07-30,
matrix gap #12). The container grammar (`docs/licht_format_spec.md`) treats these bytes as an
opaque chunk payload; this document defines what is inside. Field ownership is
`docs/licht_ownership_matrix.md`. Design rationale: self-describing property tables in the RAD shape — extensible by value, not
by record format — explicitly avoiding LAS-style closed record layouts.

Common rules (both payloads):

1. All integers little-endian; all offsets are relative to the start of the *uncompressed*
   payload; u64 sizes throughout.
2. `chunk_version` starts at 1 per fourcc. Adding a property name or an encoding value does
   **not** bump `chunk_version`; only a change to the descriptor-table layout itself or a
   re-meaning of an existing tag does.
3. Unknown properties MUST be carried forward opaquely (name + byte span) by declared-safe
   writers and by compaction — the container's opaque-preservation promise extended inside the
   payload.
4. No reserved/deprecated padding graveyards: a dead field is deleted with a version bump,
   never left as a hole (LAS `deprecated1..5` is the counterexample).
5. Property data is stored as planes (one property after another), each plane starting at a
   64-byte-aligned payload offset so bounded-window and mmap reads stay aligned.
6. Coordinates are raw IEEE-754 — no quantization. The world offset lives in the `PROJ`
   georeference block, never in the payload.

## 1. `PCLD` version 1

```
offset  size  field
0       4     magic 'LPCD'
4       2     u16 payload_version = 1
6       2     u16 coord_encoding      1 = F32_ABSOLUTE_LOCAL (only defined value;
                                      2 = I32_SCALED reserved-unimplemented)
8       8     u64 point_count
16      2     u16 property_count
18      6     zero
24      property_count × 48-byte descriptors
...     64-byte-aligned property planes
```

Property descriptor (48 bytes):

```
0    16   name        UTF-8, zero-padded, zero-terminated unless 16 bytes exact
16   2    u16 components        (e.g. 3 for means, 1 for opacity)
18   2    u16 dtype             1=f32 2=f16 3=u8 4=u16 5=u32 6=i32 7=f64
20   4    u32 encoding          1=raw planes; other values are additive
24   8    u64 byte_offset       64-byte aligned, within payload
32   8    u64 byte_length       must equal point_count × components × dtype_size for encoding 1
40   8    zero
```

Well-known property names (from the matrix row set): `means`, `colors`, `normals`, `sh0`,
`shN`, `opacity`, `scaling`, `rotation`. Additional names come from
`PointCloud::attribute_names` verbatim. Readers MUST ignore unknown names (carrying them per
common rule 3) and MUST NOT require any particular property to be present except `means`.

Validation: descriptor table bounds-checked against payload size before any allocation;
overlapping planes are corruption; `byte_length` arithmetic uses checked multiplication.

## 2. `MESH` version 1

Same envelope as `PCLD` with magic `LMSH` and `element_count` semantics per property:

```
0    4    magic 'LMSH'
4    2    u16 payload_version = 1
6    2    zero
8    8    u64 vertex_count
16   8    u64 index_count
24   2    u16 property_count
26   6    zero
32   property_count × 48-byte descriptors (same layout as PCLD)
...  64-byte-aligned planes
```

Well-known properties: `vertices` (f32×3, vertex_count), `normals`, `tangents`, `texcoords`,
`colors` (vertex_count elements each), `indices` (u32×1, index_count), `submeshes` (encoding
2 = packed `{u32 start_index, u32 index_count, u32 material_index}` records), `materials`
(encoding 3 = CBOR-free tagged binary records defined below), `textures` (encoding 4 =
concatenated `{u32 width, u32 height, u32 channels, u64 pixel_bytes, pixels}` records).

Material record (encoding 3), repeated per material, each 8-byte aligned:

```
u16 name_bytes, UTF-8 name
f32[4] base_color, f32[3] emissive, f32 metallic, f32 roughness, f32 ao
i32 albedo_tex, i32 normal_tex, i32 metallic_roughness_tex, i32 emissive_tex, i32 ao_tex
     (texture indices into `textures`, -1 = none)
u8  double_sided, 7 bytes zero
```

Original texture source paths (`*_tex_path`) are import provenance and live in `PROJ`, not
here. `MeshData::generation_` is derived and never serialized.

## 3. Conformance obligations (P3 exit)

- Round-trip gtests per payload: build → serialize → mutate-unrelated → reserialize → byte
  compare planes; unknown-property carry-forward proven.
- Hostile-descriptor battery cases: overlapping planes, out-of-bounds offsets, absurd counts
  rejected before allocation, dtype/component mismatches.
- The independent Python parser gains a `PCLD`/`MESH` decoder used by the conformance battery.

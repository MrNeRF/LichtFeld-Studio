I'll analyze SPZ v4 and LichtFeld-Studio's related I/O and quantization plans as a read-only code dive, then produce a self-contained report with file:line references.LichtFeld already has SPZ I/O and an vendored SPZ tree — next I'll read the v4 codec and LFS packing paths in detail.# SPZ v4 × LichtFeld-Studio: What to Steal Without Losing Quality

Read-only analysis of Niantic SPZ at `/home/gauss/projects/spz` (focus **version 4**) vs LFS I/O, `SplatData`, and `SPEED_VRAM_OPTIMIZATION_PLAN.md` Phase 2.

**Headline:** LFS should treat SPZ as a **distribution / interchange / export** format (and **bump the vendored library from v3→v4**). Almost none of SPZ’s **lossy attribute codecs** are safe as **training-resident VRAM** storage. Phase 2’s 16-bit block SH + joint Adam codec remains the right VRAM path. Steal **container ideas** (per-attribute streams + parallel ZSTD + TOC), not the 8-bit param encodings, for checkpoints.

---

## 1. Exact SPZ v4 format

### Magic / version detection

| Magic | Meaning |
|--------|---------|
| `0x5053474E` (“NGSP” LE) | v4 multi-stream **ZSTD** |
| `0x1f 0x8b` | Legacy v1–3 **gzip** single stream (read-only) |

Constants: `LATEST_SPZ_HEADER_VERSION = 4`, `MIN_ZSTD_SPZ_HEADER_VERSION = 4`, `MIN_SMALLEST_THREE_QUATERNIONS_VERSION = 3` — `spz/src/cc/load-spz.h:67-77`.

**Note:** Sample files in `spz/samples/*.spz` are **gzip/legacy**, not NGSP v4.

### File layout (v4)

From `spz/README.md:131-138` and `saveSpz` / `loadPackedGaussiansFromNgsp` in `load-spz.cc`:

```
Bytes  0–31:              NgspFileHeader (32 B, plaintext)
Bytes  32..(tbo−1):       Extension ILV records if flags & 0x2
Bytes  tbo..(tbo+N×16−1): TOC — N × [compressedSize u64, uncompressedSize u64]
Bytes  (tbo+N×16)..end:   N independent ZSTD-compressed attribute streams
```

`tbo = tocByteOffset`; with no extensions, `tbo = 32`.

### Header (`NgspFileHeader`, 32 bytes, little-endian)

```c
// load-spz.cc:123-134  (static_assert sizeof == 32)
struct NgspFileHeader {
  uint32_t magic;          // 0x5053474e "NGSP"
  uint32_t version;        // 4
  uint32_t numPoints;
  uint8_t  shDegree;       // 0..4
  uint8_t  fractionalBits; // fixed-point fractional bits for positions
  uint8_t  flags;          // bit0 antialiased, bit1 has extensions
  uint8_t  numStreams;     // typically 6 (5 if SH empty / degree 0)
  uint32_t tocByteOffset;
  uint8_t  reserved[12];   // must be zero
};
```

Flags: `FlagAntialiased = 0x1`, `FlagHasExtensions = 0x2` — `load-spz.cc:102-103`.

### Per-attribute streams (order)

Order matches `kAllSplatAttributes` (`splat-types.h:88-91`) and `saveSpz` src list (`load-spz.cc:936-943`):

| Stream index | Attribute | Uncompressed size (v3+ quat) |
|--------------|-----------|------------------------------|
| 0 | Positions | `N × 9` (3×24-bit) |
| 1 | Alphas | `N × 1` |
| 2 | Colors (SH DC) | `N × 3` |
| 3 | Scales | `N × 3` |
| 4 | Rotations | `N × 4` (smallest-three) |
| 5 | SH rest | `N × dim(shDegree) × 3` |

Empty streams (e.g. SH degree 0) are **skipped** on compress (`if (s.second == 0) continue` — `load-spz.cc:740-741`), so `numStreams` may be 5.

### ZSTD settings

- API: `ZSTD_compress2` with `ZSTD_c_compressionLevel` default **12** — `compressZstd` at `load-spz.cc:222-237`.
- Decompress: `ZSTD_decompress` into pre-sized destination buffers (no intermediate concat) — `decompressNgspStreams` at `load-spz.cc:640-719`.

### Parallelism

- **Native (non-WASM):** each stream compresses/decompresses via `std::async(std::launch::async)` — `load-spz.cc:700-717`, `738-757`.
- **Emscripten:** sequential only (TODO comments).

### Safety bounds

- `kMaxCompressionRatio = 1024`, `kMinBytesPerPoint = 9` — reject absurd `numPoints` vs file size (`load-spz.cc:105-121`, `1007-1013`).
- `numPoints` capped at `INT32_MAX`.

### Legacy (v1–3) for contrast

16-byte gzip header (`LegacyPackedGaussiansHeader`), single gzip stream of concatenated attribute blobs — `load-spz.cc:136-146`, `918-924`. LFS vendors **this** generation (see §4).

---

## 2. Per-attribute encoding, bit-by-bit

All pack paths are in `packGaussians` (`load-spz.cc:299-422`). Unpack mirrors in `unpackGaussians` / `PackedGaussian::unpack`.

### Positions — 24-bit signed fixed-point

| Field | Value |
|-------|--------|
| Storage | 3 bytes/component × 3 = **9 B/splat** |
| Default `fractionalBits` | **12** (hardcoded on pack: `load-spz.cc:327-332`) |
| Step | \(2^{-12} ≈ 2.44×10^{-4}\) world units |
| Integer range | 12 bits → roughly **±2048** world units (sign-extended 24-bit) |
| Encode | `round(pos × 2^{frac})`, little-endian 24-bit (`load-spz.cc:349-366`) |
| Decode | assemble + sign-extend bit 23, × `2^{-frac}` (`load-spz.cc:435-442`) |

**Error bound:** max abs error ≈ **½ LSB ≈ 1.22×10⁻⁴** (for frac=12).

**Round-trip claim vs math:** Tests use `atol=1/2048` (= \(2^{-11}\), ~half-LSB scale) — `test_io.py:31`. Comment says “~0.25 millimeter resolution” (`load-spz.cc:327`) — **that assumes unit = meters**; scene units are not enforced. **Speculation:** visually lossless for typical scene scales; large outdoor scenes near ±2048 can clip.

### Scales — 8-bit log encoding

| Field | Value |
|-------|--------|
| Domain | **log-scale** (same as LFS `scaling_raw`) |
| Encode | `u8 = clamp(round((log_s + 10) × 16), 0, 255)` (`load-spz.cc:368-370`) |
| Decode | `log_s = u8/16 − 10` (`load-spz.cc:448-450`) |
| Log range | roughly **[−10, 5.9375]** |
| Step (log) | **1/16 = 0.0625** |
| Max abs error (log) | **≈ 0.03125** |
| Relative linear scale | \(e^{0.03125}-1 ≈ **3.2%**` per axis |

Tests: `atol=1/32` on log scales — `test_io.py:32`.

**Not lossless.** Fine for export/viewers; **bad for training state**.

### Rotations (v3/v4) — smallest-three quaternion

| Field | Value |
|-------|--------|
| Size | **4 bytes/splat** |
| Scheme | 2-bit index of largest component + 3×(1 sign + 9 mag) = **32 bits** (`packQuaternionSmallestThree`, `load-spz.cc:260-297`) |
| Mag range | \([0, 1/\sqrt{2}]\) mapped to 9-bit |
| Mag step | \(1/\sqrt{2}/511 ≈ 1.38×10^{-3}\) |
| Max component err | ~½ step ≈ **6.9×10⁻⁴** |
| Largest component | reconstructed as \(\sqrt{1-\sum s_i^2}\); sign of largest forced positive |

v2 used “first three” 8-bit xyz only (3 B) — worse.

Tests: unit quaternion + rotated-vector cosine ≥ 1−1e−4 — `test_io.py:40-50`. **Strong for viewing; still lossy.**

**Order note:** SPZ stores **xyzw**; LFS `SplatData` is **wxyz** — converted in `spz.cpp:92-98`, `151-158`.

### Opacity / alpha

| Field | Value |
|-------|--------|
| Encode | `u8 = round(sigmoid(logit) × 255)` (`load-spz.cc:384-387`) |
| Decode | `logit = invSigmoid(u8/255)` (`load-spz.cc:461`) |
| Step (alpha) | 1/255 |
| Max alpha err | ~0.5/255 ≈ 0.002 |

Tests: `atol=0.01` on **logits** — `test_io.py:52` (loose because inv-sigmoid amplifies near 0/1).

### Colors (SH DC / `f_dc`)

| Field | Value |
|-------|--------|
| Encode | `u8 = round(sh0 × colorScale × 255 + 127.5)`, `colorScale = 0.15` (`splat-utils.h:15`, `load-spz.cc:389-392`) |
| Decode | `sh0 = (u8/255 − 0.5) / 0.15` |
| Step in SH-DC space | \(1/(0.15×255) ≈ 0.0261\) |
| Max abs err | ~**0.013** in SH-DC units |

Wider than [0,1] RGB intentionally (higher bands can pull color back). **Visually usually fine; not bit-exact vs PLY fp32.**

### SH rest — 8-bit storage + **band-dependent entropy quant**

| Degree | Coeffs/point (per RGB set) | Bytes/splat (×3 channels) |
|--------|----------------------------|---------------------------|
| 0 | 0 | 0 |
| 1 | 3 | 9 |
| 2 | 8 | 24 |
| 3 | 15 | 45 |
| 4 | 24 | 72 |

Layout (channel inner): `sh1n1_r,g,b, sh10_r,g,b, …` — README + `splat-types.h:368-376`.

**Storage:** each coeff as `uint8`, decode `(x − 128)/128` → nominally **[−1, 1]** (`unquantizeSH`, `splat-utils.h:53-55`).

**Pack-time quantization** (does not change stored width; zeros LSBs for better ZSTD):

```
bucketSize = 1 << (8 − bits)
q = round(x * 128) + 128
q = round_to_bucket(q, bucketSize)   // quantizeSH, load-spz.cc:54-58
```

Defaults: **`sh1Bits = 5`**, **`shRestBits = 4`** (`load-spz.h:64-65`, `144-145`). Configurable 1–8 via `PackOptions`.

Degree-1 (first 3 angular coeffs × RGB = first 9 bytes of SH stream) uses `sh1Bits`; degree ≥2 uses `shRestBits` (`load-spz.cc:412-417`).

**Error bound** (library’s own formula, `test_utils.py:13-29`):

\[
\varepsilon(N) = 2^{-N} + 0.5/128
\]

| Bits | \(\varepsilon\) (max abs on decoded SH) |
|------|------------------------------------------|
| 4 (rest default) | **≈ 0.0664** |
| 5 (deg-1 default) | **≈ 0.0352** |
| 8 (full byte) | **≈ 0.0078** |

Tests assert defaults with these epsilons — `test_io.py:53-56`.

**Niantic claim:** “typically around **10×** smaller than .ply … **minimal visual differences**” (`README.md:6-7`).  
**Math:** raw packed alone is only ~3.5–3.7× smaller than fp32 PLY; **ZSTD** supplies the rest of the 10×. “Visually lossless” is **not** a formal guarantee — it is a perceptual claim for export. Higher bands get fewer effective bits because they usually have lower energy / higher compressibility; **error bounds are still large** relative to fp32/fp16 training noise.

### Coordinate system

Default pack target **RUB** (OpenGL/three.js); LFS packs with `from = RDF` (PLY convention) — `spz.cpp:252-253`. Full converter including cross-family \(R_x(±π/2)\) and Wigner-like SH band transforms — `splat-types.h:237-341`.

---

## 3. Compression ratio math (bytes/splat)

### SPZ packed (pre-ZSTD), v3+ smallest-three

| SH degree | pos | α | rgb | scale | rot | SH | **Total packed** |
|-----------|-----|---|-----|-------|-----|----|------------------|
| 0 | 9 | 1 | 3 | 3 | 4 | 0 | **20** |
| 1 | 9 | 1 | 3 | 3 | 4 | 9 | **29** |
| 2 | 9 | 1 | 3 | 3 | 4 | 24 | **44** |
| 3 | 9 | 1 | 3 | 3 | 4 | 45 | **65** |
| 4 | 9 | 1 | 3 | 3 | 4 | 72 | **92** |

Matches comment “at most 92 bytes for degree 4” — `load-spz.h:93-94`.

### Raw PLY (fp32 Gaussian attributes only)

Standard: `xyz + opacity + scale×3 + rot×4 + f_dc×3 + f_rest×(K×3)`:

| SH | floats | **B/splat** |
|----|--------|-------------|
| 0 | 14 | **56** |
| 1 | 23 | **92** |
| 2 | 38 | **152** |
| 3 | 59 | **236** |

**LFS PLY also writes `nx,ny,nz`** (usually zeros) — `ply.cpp:2676-2679`, `2981` → **+12 B** → SH3 ≈ **248 B/splat** on disk before ASCII/binary header.

### LFS in-memory params (SH3, live N)

From `splat_data.hpp` + Phase 2 ledger (`SPEED_VRAM_OPTIMIZATION_PLAN.md:46-56`):

| Buffer | Layout | B/splat |
|--------|--------|---------|
| Geometry + SH0 | means 12 + scale 12 + rot 16 + opacity 4 + sh0 12 | **56** |
| SH rest | swizzled float4 slots (12×16 for SH3) | **192** |
| **Params total** | | **248** |
| Adam (current) | u8 m/v + 48 B fp32 per-prim scales | **172** |
| Densify aux | | **8** |
| **Persistent train** | | **~428** |

Checkpoint model path writes **canonical** `[N,K,3]` fp32 for shN (no float4 pad) — `splat_data.cpp:1039-1042` → params ≈ **236 B/splat** + per-tensor headers.

### Ratios (SH3)

| Representation | B/splat | vs PLY 236 | vs LFS mem 248 |
|----------------|---------|------------|----------------|
| PLY fp32 attrs | 236–248 | 1× | ~1× |
| SPZ packed raw | **65** | **3.6×** smaller | **3.8×** |
| SPZ + ZSTD (typical) | ~**20–30** *(speculation; scene-dependent)* | ~**8–12×** | — |
| Niantic README claim | ~**0.1× PLY** | **~10×** | — |
| LFS checkpoint model only | ~236 + headers | ~1× | ~same as PLY body |
| LFS checkpoint + Adam | ≫ model (moments serialized via strategy) | large | — |

Packed/ZSTD sizes are **not** measured on LFS scenes in this pass; ratios above for compressed size are **speculation** bounded by Niantic’s 10× claim and the hard 65 B floor before entropy coding.

---

## 4. What LFS should adopt for on-disk

### What LFS does today

| Path | Format | Precision | Notes |
|------|--------|-----------|-------|
| **Export PLY** | Binary float32 vertex attrs + normals | Full fp32 | `ply.cpp` `save_ply` / `get_ply_attribute_names` |
| **Export SPZ** | Vendored **SPZ v3 / gzip** | Lossy as above | `src/io/formats/spz.cpp:230-260` → `external/spz` |
| **Import SPZ** | Same library; unpack to RDF | Lossy | `spz.cpp:202-205`; rejects SH degree > 3 (`spz.cpp:33`) |
| **Checkpoint (.lfkp)** | Header + strategy name + **raw Tensor fp32/u8 payloads** + JSON | Full param precision; Adam as currently quantized in optimizer | `checkpoint_format.hpp:16-47`, `checkpoint.cpp:150-176`, `splat_data.cpp:1026-1053` |
| **RAD** | Chunked hierarchical + per-plane quant (f16/r8/…) + optional ZSTD-class stack | Lossy by design for streaming | `rad.hpp`, `rad_dequant_math.hpp` |
| **SOGS** | Separate path | — | `sogs.cpp` |

**Critical gap:** LFS `external/spz` is **pre-v4**:

- Header `version = 3`, gzip path only — `external/spz/load-spz.cc:156-157`, `636-637`
- Hardcoded `sh1Bits=5`, `shRestBits=4` (no PackOptions bits) — `external/spz/load-spz.cc:374-375`
- **Cannot write or fully interoperate with modern NGSP v4 files** written by current upstream

Upstream v4 is in `/home/gauss/projects/spz`. LFS already depends on **zstd** (`vcpkg.json`).

### Measured-in-code size deltas (SH3, N=1e6)

| Artifact | Formula | ≈ size |
|----------|---------|--------|
| LFS PLY body (w/ normals) | 248 × N | **~248 MB** |
| LFS PLY body (no normals) | 236 × N | **~236 MB** |
| SPZ packed only | 65 × N | **~65 MB** |
| SPZ + ZSTD (if ~10× vs PLY) | ~24 × N | **~24 MB** *(speculation)* |
| Checkpoint model tensors only | ~236 × N + ~6× tensor headers | **~236 MB** |
| Checkpoint + densify + Adam (~172) | ~(236+8+172)×N | **~416 MB** before grids/PPISP |

Checkpoint is **not compressed** today (`operator<<` writes raw bytes — `tensor_serialization.cpp:50-71`). That is the big on-disk win independent of SPZ semantics.

### Exact on-disk recommendations

1. **Yes — export `.spz` as a first-class distribution format**, but **upgrade vendored SPZ to upstream v4** (ZSTD multi-stream). Keep PLY as the lossless interchange with other 3DGS tools.
2. **Do not use SPZ as the training checkpoint format.** Checkpoints need exact-ish param resume + Adam state; SPZ drops both (no optimizer) and quantizes params too hard.
3. **Adopt the *container pattern* for a future checkpoint codec (optional v2):**  
   plaintext header + TOC + **per-tensor / per-attribute ZSTD streams** + parallel codec — without SPZ’s 8-bit param encodings. Store params as **fp32 or Phase-2-native packed** (16-bit SH blocks), Adam as **current or Phase-2 joint codec** blobs.
4. **Expose `sh1Bits` / `shRestBits`** on export UI (defaults 5/4; offer 8/8 for higher fidelity export).
5. RAD already owns **out-of-core streaming** with better structure for LFS LOD; SPZ is for **single-cloud share/view**, not billion-splat streaming.

---

## 5. What transfers to VRAM-resident storage

Compare to Phase 2 (`SPEED_VRAM_OPTIMIZATION_PLAN.md:107-119`):

| Idea | Phase 2 plan | SPZ | Transfer? |
|------|--------------|-----|-----------|
| SH rest storage | **16-bit** linear block quant, **per-256-splat** min/max, decode in registers | **4–5 effective bits**, global [−1,1], 8-bit cells | **Idea only:** per-band bit budgets |
| SH deg-1 vs rest bits | Uniform 16-bit | **5 vs 4** | **Maybe** as *energy-weighted* bit allocation **on top of** 16-bit or as export only |
| Adam | Joint `(u, log_s)` 16/8-bit block | **None** | No |
| Positions | fp32 train | 24-bit fixed | **Render-only / viewer copy** OK; not train |
| Scales | fp32 log | u8 log | **Not for train** (3% linear / huge vs Adam step) |
| Rotation | fp32 | ~10-bit smallest-three | **Viewer** OK; train needs fp32 |
| Opacity | fp32 logit | u8 α | Viewer OK; train no |
| SH DC | fp32 | u8 wide RGB | Viewer OK; train no |

### Honest quality-risk split

**Safe for render-only / viewer GPU copies / export decode → temporary VRAM:**

- Full SPZ decode to fp32 (what LFS does today).
- Optionally keep **positions** more compact on a viewer path if range known (24-bit or f16) — still usually not worth the shader complexity vs Phase 2 SH savings.

**Not quality-safe for training (gradients through quantized params):**

- **u8 log-scale:** step 0.0625 in log space dwarfs typical per-step updates; freezes/staircase scales.
- **SH 4–5 bit global quant:** error ~0.03–0.07 vs Phase 2’s ~1e−4–1e−5-class 16-bit block endpoints; destroys view-dependent detail and pollutes SH Jacobians.
- **u8 opacity in α domain:** logit gradients ill-conditioned near 0/1 after quant.
- **Fixed-point means:** OK for display; densify/grad accumulation expects continuous fp32.
- **Any pack that is not re-encoded with STE / FPBO single-writer discipline** will desync Adam state.

**Smarter than uniform 16-bit?**  
SPZ’s **lower bits on higher bands** matches known SH energy decay. Phase 2 could *optionally* allocate e.g. 16-bit deg-1 / 12-bit deg-2+ **inside** block quant — but spirulae’s default **uniform 16-bit** is already the conservative quality choice, and LFS gate G6 is “no regression.” **Recommendation: keep Phase 2 uniform 16-bit; treat per-band budgets as a later experiment, not a SPZ port.**

**Rasterizer needs:** FastGS wants continuous means, log-scales, unit quats, logits, SH in the swizzled/canonical float domain (or Phase-2 decode-to-register). SPZ encodings are a **disk codec**, not a training tensor format.

---

## 6. Other clever bits worth stealing

| Idea | Where | Value for LFS |
|------|-------|----------------|
| **Per-attribute independent ZSTD + TOC** | v4 layout | High for checkpoint/export speed & partial I/O |
| **Parallel stream codec** | `std::async` per stream | High if not already on large checkpoints |
| **Direct decompress into destination buffers** | `decompressNgspStreams` | Avoid peak RAM spikes |
| **Chunked attribute unpack** | `chunked-unpack.h` (`unpackChunk`) | Good for WASM/streaming viewers; LFS RAD is stronger for LOD |
| **Coordinate-system converter + SH band transforms** | `splat-types.h` | Keep; LFS already converts RDF↔RUB on SPZ I/O |
| **ILV vendor extensions** | `extensions/README.md` | Useful for camera limits / custom metadata on export; low training value |
| **SH entropy quant (zero LSBs)** | `quantizeSH` | Disk-only; improves ZSTD ratio without changing container |
| **Antialiasing flag** | header flags | Propagate if mip-splatting trained |
| **Safe orbit / Adobe coordinate extensions** | `extensions/` | Product/viewer features |

LFS already has a **richer** streaming stack in **RAD** (chunk frames, f16/r8 planes, GPU dequant, meta sidecar). Steal SPZ’s *simple cloud* packaging for interchange; don’t replace RAD with SPZ for huge scenes.

---

## Ranked recommendations (value / effort)

| Rank | Tag | Recommendation | Effort | Quality risk |
|------|-----|----------------|--------|--------------|
| **1** | **[disk]** | **Upgrade `external/spz` → upstream v4** (NGSP + ZSTD level 12, parallel streams). Keep export/import API; default PackOptions; optional `sh1Bits/shRestBits`. | Low–med | **None** if still decode to fp32 for training load; export remains intentionally lossy |
| **2** | **[disk]** | **Checkpoint v2: per-tensor ZSTD streams + TOC** (SPZ container idea, **not** SPZ param codecs). Keep fp32 (or Phase-2 native) params + Adam blobs. | Med | **None** if lossless (or bit-exact zstd); huge size/speed win vs raw LFKP |
| **3** | **[disk]** | Prefer **`.spz` for share/view/HTML export**; keep **`.ply` for lossless** toolchains; document size/quality tradeoff. | Low | Export quality = SPZ loss (known, acceptable for viewers) |
| **4** | **[vram]** | **Execute Phase 2 as planned** (16-bit SH block + joint Adam). Do **not** port SPZ 4–5-bit SH or u8 geom. | High (already planned) | Low with G6; spirulae-proven |
| **5** | **[vram]** | Optional experiment: **per-band bit budgets** (more bits deg-1) *within* block quant — inspired by SPZ, not SPZ bit depths. | Med | Medium; needs G6 A/B |
| **6** | **[disk]** | Expose high-quality SPZ export preset (`sh1Bits=shRestBits=8`) for archival-ish compressed export. | Trivial after v4 | Still not lossless (scales/α/rot/pos still quant) |
| **7** | **[vram]/[disk]** | Use SPZ-decoded clouds only as **init/import**, never as mid-train checkpoint. | Policy | Avoids silent quality death |
| **8** | **[disk]** | Optional: store **render snapshot** beside checkpoint as `.spz` for fast previews (loss OK). | Low | None for training resume |
| **9** | **[vram]** | Smallest-three / log-u8 / 24-bit means as **viewer GPU path only** | High for little gain | Not for train; LFS already fp32 in raster path |

### Explicit non-recommendations

- **Do not** replace Phase 2 SH quant with SPZ 5/4-bit.  
- **Do not** store Adam or densify state in SPZ.  
- **Do not** train with straight-through 8-bit SPZ params without a full redesign (would fight densify + Adam).  
- **Do not** assume sample `.spz` files validate v4 paths (they’re gzip v3).

---

## LFS integration status (facts)

| Item | Status |
|------|--------|
| SPZ load/save in product | Yes — `src/io/formats/spz.{hpp,cpp}`, loaders `spz_loader.*` |
| Library version | **Vendored v3 gzip** (`external/spz`), not repo `/spz` v4 |
| Coord handling | RDF on pack/unpack (`spz.cpp:204, 253`) |
| SH degree | LFS caps SPZ at **degree ≤ 3**; upstream allows **4** |
| zstd already in tree | Yes (`vcpkg.json`) |
| Phase 2 SH/Adam | Planned; **orthogonal** to SPZ disk codec |

---

## Summary

**SPZ v4** is a well-engineered **lossy, stream-parallel, ZSTD-compressed** gaussian cloud format: 24-bit fixed means, u8 log-scales, 10-bit-class smallest-three quats, u8 opacity & DC, and **band-dependent SH entropy quant** (default 5/4 bits) in 8-bit cells. Niantic’s ~10× vs PLY comes from **quant + ZSTD**, not from packing alone (~3.6× raw for SH3).

**Without losing quality for training:** adopt SPZ **only** as export/import/viewer interchange (after **v4 upgrade**), and steal **multi-stream ZSTD packaging** for a future **lossless/near-lossless checkpoint** — while Phase 2 remains the correct **VRAM** strategy. SPZ’s aggressive param quant is **viewer-safe, training-unsafe**.

---

*Speculation markers: compressed B/splat for real LFS scenes; “0.25 mm” physical interpretation of fractionalBits; optional per-band Phase-2 experiment outcomes.*

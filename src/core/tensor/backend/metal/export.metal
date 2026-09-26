
// ===========================================================================
// Export kernels, ported from the export_*.slang modules. They keep those
// parameter layouts and phase numbers (kOp), so export_pipeline.cpp drives
// both backends. Loops stride over the whole grid like the Slang shaders.

constant uint kExportWidth = 256;

// ---------------------------------------------------------------------------
// Morton keys, from export_morton.slang: phase 0 reduces block bounds, 1
// reduces those into the six scene bounds, 2 writes 63-bit keys.

struct ExportMortonParams {
    device const float* positions;
    device float* partials;
    device ulong* keys;
    uint n, blocks, padding;
    float min_x, min_y, min_z, mul_x, mul_y, mul_z;
    uint tail;
};

// Spreads 21 bits to every third bit, as morton_spread does.
static ulong morton_spread(uint x) {
    ulong v = ulong(x) & 0x1fffffUL;
    v = (v | (v << 32)) & 0x001f00000000ffffUL;
    v = (v | (v << 16)) & 0x001f0000ff0000ffUL;
    v = (v | (v << 8)) & 0x100f00f00f00f00fUL;
    v = (v | (v << 4)) & 0x10c30c30c30c30c3UL;
    return (v | (v << 2)) & 0x1249249249249249UL;
}

kernel void export_morton(constant ExportMortonParams& p [[buffer(0)]], uint gid [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]], uint groups [[threadgroups_per_grid]],
                          uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float3 lows[32], highs[32];
    if (kOp == 2) {
        const float axis_max = float((1u << 21) - 1u);
        const float low[3] = {p.min_x, p.min_y, p.min_z}, mul[3] = {p.mul_x, p.mul_y, p.mul_z};
        for (uint i = gid * kExportWidth + tid; i < p.n; i += groups * kExportWidth) {
            ulong code = 0;
            for (uint a = 0; a < 3; ++a) {
                const float scaled = (p.positions[i * 3 + a] - low[a]) * mul[a];
                const uint q = scaled <= 0 ? 0u : scaled >= axis_max ? uint(axis_max) : uint(scaled);
                code |= morton_spread(q) << a;
            }
            p.keys[i] = code;
        }
        return;
    }
    if (kOp == 1 && gid != 0)
        return;
    float3 lo = float3(INFINITY), hi = float3(-INFINITY);
    if (kOp == 0) {
        for (uint i = gid * kExportWidth + tid; i < p.n; i += groups * kExportWidth) {
            const float3 v(p.positions[i * 3], p.positions[i * 3 + 1], p.positions[i * 3 + 2]);
            lo = min(lo, v);
            hi = max(hi, v);
        }
    } else {
        for (uint b = tid; b < p.blocks; b += kExportWidth) {
            lo = min(lo, float3(p.partials[b * 6], p.partials[b * 6 + 1], p.partials[b * 6 + 2]));
            hi = max(hi, float3(p.partials[b * 6 + 3], p.partials[b * 6 + 4], p.partials[b * 6 + 5]));
        }
    }
    lo = simd_min(lo);
    hi = simd_max(hi);
    if (simd_is_first()) {
        lows[simd] = lo;
        highs[simd] = hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0)
        return;
    for (uint s = 1; s < simds; ++s) {
        lo = min(lo, lows[s]);
        hi = max(hi, highs[s]);
    }
    device float* const out = kOp == 0 ? p.partials + gid * 6 : (device float*)p.keys;
    for (uint a = 0; a < 3; ++a) {
        out[a] = lo[a];
        out[3 + a] = hi[a];
    }
}

// ---------------------------------------------------------------------------
// Stable 4-bit LSD radix over ulong keys with int payloads, from
// export_radix.slang. A thread ranks 8 consecutive elements of its block, so
// equal digits keep input order. Phase 0 counts the digits of every block in
// digit-major order, 2 turns the counts into exclusive offsets, 1 scatters.
// Digit counts travel in pairs of 16-bit fields, which cannot carry because a
// block holds 2048 elements.

constant uint kExportRadixPerThread = 8;

struct ExportRadixParams {
    device const ulong* keys_in;
    device ulong* keys_out;
    device const int* index_in;
    device int* index_out;
    device uint* counts;
    device uint* scratch;
    uint n, blocks, shift, padding;
};

static uint radix_field(uint pair, uint digit) {
    return (pair >> ((digit & 1u) * 16u)) & 0xffffu;
}

kernel void export_radix(constant ExportRadixParams& p [[buffer(0)]], uint gid [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]], uint groups [[threadgroups_per_grid]],
                         uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup uint totals[32 * 8];
    if (kOp == 2) {
        const uint count = p.blocks * 16u;
        uint carry = 0;
        for (uint chunk = 0; chunk < count; chunk += kExportWidth) {
            const uint i = chunk + tid;
            const uint value = i < count ? p.counts[i] : 0u;
            const uint prefix = simd_prefix_exclusive_sum(value);
            const uint sum = simd_sum(value);
            if (simd_is_first())
                totals[simd] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint before = 0, all = 0;
            for (uint s = 0; s < simds; ++s) {
                before += s < simd ? totals[s] : 0u;
                all += totals[s];
            }
            if (i < count)
                p.counts[i] = carry + before + prefix;
            carry += all;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }
    for (uint block = gid; block < p.blocks; block += groups) {
        const uint first = (block * kExportWidth + tid) * kExportRadixPerThread;
        uint pairs[8] = {};
        uint digits = 0;
        for (uint e = 0; e < kExportRadixPerThread && first + e < p.n; ++e) {
            const uint digit = uint(p.keys_in[first + e] >> p.shift) & 15u;
            pairs[digit >> 1] += 1u << ((digit & 1u) * 16u);
            digits |= digit << (4u * e);
        }
        // Offsets of this thread's digits within the block, and the block totals.
        uint offsets[8], block_totals[8];
        for (uint j = 0; j < 8; ++j) {
            offsets[j] = simd_prefix_exclusive_sum(pairs[j]);
            const uint sum = simd_sum(pairs[j]);
            if (simd_is_first())
                totals[simd * 8 + j] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 8; ++j) {
            uint before = 0, all = 0;
            for (uint s = 0; s < simds; ++s) {
                before += s < simd ? totals[s * 8 + j] : 0u;
                all += totals[s * 8 + j];
            }
            offsets[j] += before;
            block_totals[j] = all;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kOp == 0) {
            if (tid == 0) {
                for (uint d = 0; d < 16; ++d)
                    p.counts[d * p.blocks + block] = radix_field(block_totals[d >> 1], d);
            }
            continue;
        }
        for (uint e = 0; e < kExportRadixPerThread && first + e < p.n; ++e) {
            const uint digit = (digits >> (4u * e)) & 15u;
            const uint at = p.counts[digit * p.blocks + block] + radix_field(offsets[digit >> 1], digit);
            offsets[digit >> 1] += 1u << ((digit & 1u) * 16u);
            p.keys_out[at] = p.keys_in[first + e];
            p.index_out[at] = p.index_in[first + e];
        }
    }
}

// ---------------------------------------------------------------------------
// Palette k-means over swizzled SH points, from export_kmeans.slang. Phases:
// 0 seeds centroids from point indices and 1 computes their norms; 2 assigns
// points to the nearest centroid (11 for SH3, 16 for dense SH3 rows); 4 moves
// centroids to their means and reseeds empty ones; 7, 9 and 10 are 2, 0 and 4
// over dense [row, dims] points; 13 and 14 key points and centroids by norm
// for the screened assignment; 17 keys points by label, 18 finds the run
// offsets of sorted keys, 19 splits runs into tasks of 256 points and 20 lists
// the four nearest centroids of every centroid. The float-atomic sums make
// the sorted-run phase 21 unnecessary.

struct ExportKmeansParams {
    device const float* sh;
    device float* centroids;
    device const int* indices;
    device int* labels;
    device float* sums;
    device int* counts;
    device float* norms;
    uint n, k, dims, slots, seed, padding;
};

static float kmeans_sh_dim(device const float* sh, uint point, uint dim, uint slots) {
    return sh[((point / 32u) * (slots * 32u) + (dim / 4u) * 32u + point % 32u) * 4u + dim % 4u];
}

// A point of phase 2, 7, 11 or 16.
static float kmeans_point_dim(constant ExportKmeansParams& p, uint point, uint dim) {
    return kOp == 7 || kOp == 16 ? p.sh[point * (kOp == 16 ? 45u : p.dims) + dim]
                                 : kmeans_sh_dim(p.sh, point, dim, p.slots);
}

// The LCG pick of phase 4 and 10 for an empty centroid.
static uint kmeans_reseed(constant ExportKmeansParams& p, uint c) {
    uint rng = p.seed ^ (c * 1664525u + 1013904223u);
    rng = rng * 1664525u + 1013904223u;
    return p.n == 0 ? 0 : rng % p.n;
}

kernel void export_kmeans(constant ExportKmeansParams& p [[buffer(0)]], uint gid [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]], uint groups [[threadgroups_per_grid]]) {
    threadgroup float tile[32 * 48];
    threadgroup float tile_norm[32];
    const uint first = gid * kExportWidth + tid, stride = groups * kExportWidth;
    if (kOp == 13 || kOp == 14) {
        device ulong* keys = (device ulong*)p.sums;
        for (uint i = first; i < (kOp == 13 ? p.n : p.k); i += stride) {
            float norm = 0;
            if (kOp == 13) {
                for (uint d = 0; d < 45u; ++d) {
                    const float v = kmeans_sh_dim(p.sh, i, d, p.slots);
                    norm = fma(v, v, norm);
                }
            } else {
                norm = p.norms[i];
            }
            const uint bits = norm == 0 ? 0u : as_type<uint>(norm);
            keys[i] = kOp == 13 ? bits >> 16u : bits;
            p.counts[i] = int(i);
        }
    } else if (kOp == 17) {
        device ulong* keys = (device ulong*)p.sums;
        for (uint i = first; i < p.n; i += stride) {
            keys[i] = uint(p.indices[i]);
            p.counts[i] = int(i);
        }
    } else if (kOp == 18) {
        device const ulong* keys = (device const ulong*)p.sh;
        for (uint group = first; group < p.k; group += stride) {
            uint lo = 0, hi = p.n;
            while (lo < hi) {
                const uint mid = lo + (hi - lo) / 2u;
                if (keys[mid] < ulong(group))
                    lo = mid + 1u;
                else
                    hi = mid;
            }
            p.labels[group] = int(lo);
        }
    } else if (kOp == 19) {
        if (first == 0) {
            int total = 0;
            p.labels[0] = 0;
            for (uint group = 0; group < 256u; ++group) {
                total += (p.indices[group + 1u] - p.indices[group] + 255) / 256;
                p.labels[group + 1u] = total;
            }
        }
    } else if (kOp == 20) {
        for (uint group = first; group < p.k; group += stride) {
            float values[45];
            for (uint d = 0; d < 45u; ++d)
                values[d] = p.centroids[group * 45u + d];
            float best[4] = {1e30f, 1e30f, 1e30f, 1e30f};
            int ids[4] = {};
            for (uint other = 0; other < p.k; ++other) {
                float dot = 0;
                for (uint d = 0; d < 45u; ++d)
                    dot = fma(values[d], p.centroids[other * 45u + d], dot);
                const float dist = fma(-2.0f, dot, p.norms[other]);
                if (dist > best[3] || (dist == best[3] && int(other) >= ids[3]))
                    continue;
                int at = 3;
                while (at > 0 && (dist < best[at - 1] || (dist == best[at - 1] && int(other) < ids[at - 1]))) {
                    best[at] = best[at - 1];
                    ids[at] = ids[at - 1];
                    --at;
                }
                best[at] = dist;
                ids[at] = int(other);
            }
            for (uint a = 0; a < 4u; ++a)
                p.labels[group * 4u + a] = ids[a];
        }
    } else if (kOp == 0 || kOp == 9) {
        for (uint c = first; c < p.k; c += stride) {
            const uint source = uint(p.indices[c]);
            for (uint d = 0; d < p.dims; ++d)
                p.centroids[c * p.dims + d] = kOp == 9 ? p.sh[source * p.dims + d] : kmeans_sh_dim(p.sh, source, d, p.slots);
        }
    } else if (kOp == 1) {
        for (uint c = first; c < p.k; c += stride) {
            float norm = 0;
            for (uint d = 0; d < p.dims; ++d) {
                const float v = p.centroids[c * p.dims + d];
                norm = fma(v, v, norm);
            }
            p.norms[c] = norm;
        }
    } else if (kOp == 4 || kOp == 10) {
        for (uint c = first; c < p.k; c += stride) {
            const int count = p.counts[c];
            const uint pick = kmeans_reseed(p, c);
            for (uint d = 0; d < p.dims; ++d) {
                p.centroids[c * p.dims + d] = count > 0     ? p.sums[c * p.dims + d] / float(count)
                                              : kOp == 10 ? p.sh[pick * p.dims + d]
                                                          : kmeans_sh_dim(p.sh, pick, d, p.slots);
            }
        }
    } else if (kOp == 2 || kOp == 7 || kOp == 11 || kOp == 16) {
        // Tiles of 32 centroids in threadgroup memory; ties keep the lower index.
        const bool sh3 = kOp == 11 || kOp == 16;
        const uint dims = sh3 ? 45u : p.dims;
        for (uint base = 0; base < p.n; base += stride) {
            const uint point = base + first;
            float values[45];
            if (sh3) {
                for (uint d = 0; d < 45u; ++d)
                    values[d] = point < p.n ? kmeans_point_dim(p, point, d) : 0;
            }
            float best = 1e30f;
            int best_id = int(p.k);
            for (uint t = 0; t < (p.k + 31u) / 32u; ++t) {
                for (uint i = tid; i < 32u * dims; i += kExportWidth) {
                    const uint c = i / dims, d = i - c * dims, global = t * 32u + c;
                    tile[c * 48u + d] = global < p.k ? p.centroids[global * dims + d] : 0;
                }
                if (tid < 32) {
                    const uint global = t * 32u + tid;
                    tile_norm[tid] = global < p.k ? p.norms[global] : 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (point < p.n) {
                    for (uint c = 0; c < 32u && t * 32u + c < p.k; ++c) {
                        float dot = 0;
                        if (sh3) {
                            for (uint d = 0; d < 45u; ++d)
                                dot = fma(values[d], tile[c * 48u + d], dot);
                        } else {
                            for (uint d = 0; d < dims; ++d)
                                dot = fma(kmeans_point_dim(p, point, d), tile[c * 48u + d], dot);
                        }
                        const float dist = fma(-2.0f, dot, tile_norm[c]);
                        const int id = int(t * 32u + c);
                        if (dist < best || (dist == best && id < best_id)) {
                            best = dist;
                            best_id = id;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (point < p.n)
                p.labels[point] = best_id;
        }
    }
}

// Per-label sums and counts with float atomics, from
// export_kmeans_accumulate.slang: phase 3 reads swizzled points, 8 dense rows.
kernel void export_kmeans_accumulate(constant ExportKmeansParams& p [[buffer(0)]],
                                     uint gid [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     uint groups [[threadgroups_per_grid]]) {
    for (uint i = gid * kExportWidth + tid; i < p.n; i += groups * kExportWidth) {
        const int label = p.labels[i];
        if (label < 0 || uint(label) >= p.k)
            continue;
        for (uint d = 0; d < p.dims; ++d) {
            const float value = kOp == 3 ? kmeans_sh_dim(p.sh, i, d, p.slots) : p.sh[i * p.dims + d];
            atomic_fetch_add_explicit((device atomic_float*)&p.sums[uint(label) * p.dims + d], value,
                                      memory_order_relaxed);
        }
        atomic_fetch_add_explicit((device atomic_int*)&p.counts[uint(label)], 1, memory_order_relaxed);
    }
}

// Half centroid rows of 48 in norm order for the screened assignment, from
// export_kmeans_pack.slang.
kernel void export_kmeans_pack(constant ExportKmeansParams& p [[buffer(0)]], uint gid [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]], uint groups [[threadgroups_per_grid]]) {
    device half* packed = (device half*)p.sums;
    for (uint c = gid * kExportWidth + tid; c < p.k; c += groups * kExportWidth) {
        for (uint d = 0; d < 48u; ++d)
            packed[c * 48u + d] = d < p.dims ? half(p.centroids[uint(p.indices[c]) * p.dims + d]) : half(0);
    }
}

// Screened SH3 assignment, from export_kmeans_screen.slang. A SIMD group
// scores 16 points sorted by norm. Half matrix products only reject centroids
// that cannot win; every survivor is scored with the ordered FP32 FMAs of
// phase 11, so the labels are its exact argmin.
struct ExportScreenParams {
    device const float* sh;
    device const float* centroids;
    device const half* half_centroids;
    device const float* norms;
    device int* labels;
    device const int* point_order;
    device const int* centroid_order;
    uint n, k, slots, have_labels;
};

// The first centroid in norm order whose norm is not below value (or, with
// upper, not above it).
static uint kmeans_norm_bound(constant ExportScreenParams& p, float value, bool upper) {
    uint lo = 0, hi = p.k;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2u;
        const float norm = p.norms[uint(p.centroid_order[mid])];
        if (norm < value || (upper && norm == value))
            lo = mid + 1u;
        else
            hi = mid;
    }
    return lo;
}

constant uint kScreenSimdgroups = kExportWidth / 32u;

kernel void export_kmeans_screen(constant ExportScreenParams& p [[buffer(0)]],
                                 uint gid [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
                                 ushort simd [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half packed[kScreenSimdgroups * 16 * 48];
    threadgroup float approx[kScreenSimdgroups * 16 * 16];
    threadgroup half* points = packed + simd * 16u * 48u;
    threadgroup float* estimates = approx + simd * 16u * 16u;
    const uint point_lane = lane >> 1u;
    const uint tiles = (p.n + 15u) / 16u;
    const float last_norm = p.norms[uint(p.centroid_order[p.k - 1u])];
    const bool norms_safe = last_norm >= 0 && last_norm < 4.0e9f;
    for (uint tile = gid * kScreenSimdgroups + simd; tile < tiles; tile += groups * kScreenSimdgroups) {
        const uint point0 = tile * 16u;
        if (lane < 16) {
            for (uint d = 0; d < 48u; ++d) {
                const bool present = point0 + lane < p.n && d < 45u;
                points[lane * 48u + d] =
                    present ? half(kmeans_sh_dim(p.sh, uint(p.point_order[point0 + lane]), d, p.slots)) : half(0);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const bool live = point0 + point_lane < p.n;
        const uint point = live ? uint(p.point_order[point0 + point_lane]) : 0;
        float best = 1e30f;
        int best_id = int(p.k);
        float values[45];
        float point_norm = 0;
        bool safe = true;
        for (uint d = 0; d < 45u; ++d) {
            const float v = live ? kmeans_sh_dim(p.sh, point, d, p.slots) : 0;
            values[d] = v;
            point_norm = fma(v, v, point_norm);
            safe = safe && isfinite(v) && abs(v) <= 65000.0f;
        }
        if (live && p.have_labels != 0) {
            const int seed = p.labels[point];
            if (seed >= 0 && uint(seed) < p.k) {
                float dot = 0;
                for (uint d = 0; d < 45u; ++d)
                    dot = fma(values[d], p.centroids[uint(seed) * 45u + d], dot);
                best = fma(-2.0f, dot, p.norms[uint(seed)]);
                best_id = seed;
            }
        }
        uint first = p.k, last = 0;
        if (live) {
            first = 0;
            last = p.k;
            if (safe && norms_safe && point_norm < 4.0e9f && isfinite(best) && best < 1e30f) {
                // A winner satisfies ny - 2*sqrt(nx*ny) - m*(nx+ny) <= best + eta.
                const float m = 0.0011f;
                const float a = 1.0f - m;
                const float root = sqrt(point_norm);
                const float radius = sqrt(m * (2.0f - m) * point_norm + a * max(point_norm + best + 1e-8f, 0.0f));
                const float pad = 1e-5f * (root + radius) + 1e-10f;
                const float lower = max(0.0f, (root - radius - pad) / a);
                const float upper = (root + radius + pad) / a;
                first = kmeans_norm_bound(p, lower * lower, false);
                last = kmeans_norm_bound(p, upper * upper, true);
            }
        }
        simdgroup_half8x8 a[2][6];
        for (uint r = 0; r < 2u; ++r) {
            for (uint b = 0; b < 6u; ++b)
                simdgroup_load(a[r][b], points + r * 8u * 48u + b * 8u, 48);
        }
        const uint last_tile = (simd_max(last) + 15u) / 16u;
        for (uint t = simd_min(first) / 16u; t < last_tile; ++t) {
            for (uint c = 0; c < 2u; ++c) {
                // The rows hold centroids, so the loads transpose them into dims x centroids.
                simdgroup_half8x8 b[6];
                for (uint i = 0; i < 6u; ++i)
                    simdgroup_load(b[i], p.half_centroids + (t * 16u + c * 8u) * 48u + i * 8u, 48, ulong2(0), true);
                for (uint r = 0; r < 2u; ++r) {
                    simdgroup_float8x8 product = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    for (uint i = 0; i < 6u; ++i)
                        simdgroup_multiply_accumulate(product, a[r][i], b[i], product);
                    simdgroup_store(product, estimates + r * 8u * 16u + c * 8u, 16);
                }
            }
            // Lane c holds the id and norm of the tile's centroid c.
            const uint tile_id = t * 16u + (lane & 15u) < p.k ? uint(p.centroid_order[t * 16u + (lane & 15u)]) : 0u;
            const float tile_norm = p.norms[tile_id];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = lane & 1u; c < 16u; c += 2u) {
                const uint id = simd_shuffle(tile_id, ushort(c));
                const float norm = simd_shuffle(tile_norm, ushort(c));
                const uint sorted = t * 16u + c;
                if (live && sorted >= first && sorted < last) {
                    const float estimate = fma(-2.0f, estimates[point_lane * 16u + c], norm);
                    const float margin = 0.0011f * (point_norm + norm) + 1e-8f;
                    if (!safe || !(norm >= 0.0f && norm < 4.0e9f) || !isfinite(estimate) || estimate <= best + margin) {
                        float dot = 0;
                        for (uint d = 0; d < 45u; ++d)
                            dot = fma(values[d], p.centroids[id * 45u + d], dot);
                        const float dist = fma(-2.0f, dot, norm);
                        if (dist < best || (dist == best && int(id) < best_id)) {
                            best = dist;
                            best_id = int(id);
                        }
                    }
                }
            }
            // The next tile overwrites the estimates.
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float other_best = simd_shuffle_xor(best, 1);
        const int other_id = simd_shuffle_xor(best_id, 1);
        if (other_best < best || (other_best == best && other_id < best_id))
            best_id = other_id;
        if ((lane & 1u) == 0 && live)
            p.labels[point] = best_id;
    }
}

// Hierarchical SH3 assignment, from export_kmeans_group.slang. A task is 256
// points that share a super-cluster; they are scored only against the member
// lists of that super's four nearest supers.
struct ExportGroupParams {
    device const float* sh;
    device const float* centroids;
    device const float* norms;
    device int* labels;
    device const int* sorted;
    device const int* group_offsets;
    device const int* task_offsets;
    device const int* members;
    device const int* member_offsets;
    device const int* nearest;
    uint n, slots;
};

kernel void export_kmeans_group(constant ExportGroupParams& p [[buffer(0)]], uint gid [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]], uint groups [[threadgroups_per_grid]]) {
    threadgroup float tile[32 * 45];
    threadgroup float tile_norm[32];
    threadgroup int tile_id[32];
    const uint tasks = uint(p.task_offsets[256]);
    for (uint task = gid; task < tasks; task += groups) {
        uint lo = 0, hi = 256;
        while (lo + 1 < hi) {
            const uint mid = (lo + hi) >> 1;
            if (uint(p.task_offsets[mid]) <= task)
                lo = mid;
            else
                hi = mid;
        }
        const uint group = lo;
        const uint begin = uint(p.group_offsets[group]), end = uint(p.group_offsets[group + 1]);
        const uint sorted = begin + (task - uint(p.task_offsets[group])) * kExportWidth + tid;
        const bool valid = sorted < end && sorted < p.n;
        const uint point = valid ? uint(p.sorted[sorted]) : 0;
        float values[45];
        for (uint d = 0; d < 45; ++d)
            values[d] = valid ? kmeans_sh_dim(p.sh, point, d, p.slots) : 0;
        float best = 1e30f;
        int best_id = 0x7fffffff;
        uint starts[4], lengths[4], count = 0;
        for (uint a = 0; a < 4u; ++a) {
            const uint super = uint(p.nearest[group * 4u + a]);
            starts[a] = uint(p.member_offsets[super]);
            lengths[a] = uint(p.member_offsets[super + 1u]) - starts[a];
            count += lengths[a];
        }
        for (uint t = 0; t < (count + 31u) / 32u; ++t) {
            if (tid < 32) {
                uint cursor = t * 32u + tid;
                int id = -1;
                for (uint a = 0; a < 4u; ++a) {
                    if (cursor < lengths[a]) {
                        id = p.members[starts[a] + cursor];
                        break;
                    }
                    cursor -= lengths[a];
                }
                tile_id[tid] = id;
                tile_norm[tid] = id >= 0 ? p.norms[uint(id)] : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < 32u * 45u; i += kExportWidth) {
                const uint c = i / 45u, d = i - c * 45u;
                const int id = tile_id[c];
                tile[c * 45u + d] = id >= 0 ? p.centroids[uint(id) * 45u + d] : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (valid) {
                for (uint c = 0; c < 32; ++c) {
                    const int id = tile_id[c];
                    if (id < 0)
                        continue;
                    float dot = 0;
                    for (uint d = 0; d < 45; ++d)
                        dot = fma(values[d], tile[c * 45u + d], dot);
                    const float dist = fma(-2.0f, dot, tile_norm[c]);
                    if (dist < best || (dist == best && id < best_id)) {
                        best = dist;
                        best_id = id;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (valid)
            p.labels[point] = best_id;
    }
}

// ---------------------------------------------------------------------------
// Float-float arithmetic, from df64.slangh. Contraction is off, so every
// compensation step rounds on its own as the rounded_math helpers ensure.

struct Df {
    float hi;
    float lo;
};

static Df df_from(float x) {
    return {x, 0};
}

static Df df_two_sum(float a, float b) {
    const float s = a + b, v = s - a;
    return {s, (a - (s - v)) + (b - v)};
}

static Df df_renorm(float hi, float lo) {
    const float s = hi + lo;
    return {s, lo - (s - hi)};
}

static Df df_add(Df a, Df b) {
    const Df s = df_two_sum(a.hi, b.hi);
    return df_renorm(s.hi, (a.lo + b.lo) + s.lo);
}

static Df df_sub(Df a, Df b) {
    return df_add(a, {-b.hi, -b.lo});
}

static Df df_mul(Df a, Df b) {
    const float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e);
    e = fma(a.lo, b.hi, e);
    e = fma(a.lo, b.lo, e);
    return df_renorm(p, e);
}

static Df df_div(Df a, Df b) {
    const float q0 = a.hi / b.hi;
    Df r = df_sub(a, df_mul(df_from(q0), b));
    const float q1 = q0 + r.hi / b.hi;
    r = df_sub(a, df_mul(df_from(q1), b));
    return df_renorm(q1, r.hi / b.hi);
}

static Df df_sqrt(Df a) {
    const float v = a.hi + a.lo;
    if (!(v > 0))
        return df_from(0);
    // Two Newton steps from a float sqrt recover the float-float square root.
    Df y = df_from(sqrt(v));
    y = df_mul(df_from(0.5f), df_add(y, df_div(a, y)));
    return df_mul(df_from(0.5f), df_add(y, df_div(a, y)));
}

static bool df_gt(Df a, Df b) {
    const bool ainf = isinf(a.hi), binf = isinf(b.hi);
    if (ainf || binf)
        return ainf && a.hi > 0 && !(binf && b.hi > 0);
    const Df d = df_sub(a, b);
    return !isnan(d.hi) && (d.hi > 0 || (d.hi == 0 && d.lo > 0));
}

static float df_ldexp(float x, int n) {
    const uint u = as_type<uint>(x);
    int e = int((u >> 23) & 0xffu);
    if (e == 0 || e == 255 || x == 0)
        return x;
    e += n;
    if (e >= 255)
        return as_type<float>((u & 0x80000000u) | 0x7f800000u);
    if (e <= 0)
        return as_type<float>(u & 0x80000000u);
    return as_type<float>((u & 0x807fffffu) | (uint(e) << 23));
}

static Df df_scale2(Df a, int n) {
    return {df_ldexp(a.hi, n), df_ldexp(a.lo, n)};
}

static Df df_exp(Df x) {
    return df_from(exp(x.hi + x.lo));
}

// Logistic without overflow: exp(-|x|) never exceeds one.
static Df df_sigmoid(float x) {
    const Df e = df_exp(df_from(-abs(x)));
    const Df sum = df_add(df_from(1), e);
    return x >= 0 ? df_div(df_from(1), sum) : df_div(e, sum);
}

static Df df_exp_precise(Df x) {
    if (!isfinite(x.hi) || x.hi < -80 || x.hi > 80)
        return df_exp(x);
    const int n = int(round(x.hi * 1.4426950408889634f));
    const Df r = df_sub(x, df_mul(df_from(float(n)), {0.6931471824645996f, -1.904654323148236e-9f}));
    // Range reduction bounds |r| by ln(2)/2; the degree-12 remainder is below 3e-16.
    constexpr float coefficients[12][2] = {
        {2.5052107943679403e-08f, 4.4176230446483665e-16f}, {2.755731998149713e-07f, -7.5751122090511949e-15f},
        {2.7557318844628753e-06f, 3.7935712242972291e-14f}, {2.4801587642286904e-05f, -3.4069960936668198e-13f},
        {0.00019841270113829523f, -2.7255968749334558e-12f}, {0.0013888889225199819f, -3.3631094437103215e-11f},
        {0.0083333337679505348f, -4.3461720333759502e-10f}, {0.041666667908430099f, -1.2417634698280722e-09f},
        {0.1666666716337204f, -4.9670538793122887e-09f}, {0.5f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
    Df value = {2.0876755879584152e-09f, 1.1082839809204342e-16f};
    for (uint c = 0; c < 12; ++c)
        value = df_add(df_mul(value, r), {coefficients[c][0], coefficients[c][1]});
    return df_scale2(value, n);
}

static Df df_log(Df a) {
    const float v = a.hi + a.lo;
    return df_from(v > 0 ? log(v) : -80.0f);
}

static Df df_pow(Df a, Df b) {
    return df_from(pow(max(a.hi + a.lo, 0.0f), b.hi + b.lo));
}

static float df_round_up(Df a) {
    if (!(a.lo > 0) || !(a.hi < 3.402823e38f))
        return a.hi;
    uint bits = as_type<uint>(a.hi);
    if ((bits & 0x7fffffffu) == 0)
        return as_type<float>(1u << 23);
    bits = a.hi >= 0 ? bits + 1 : bits - 1;
    return as_type<float>(bits);
}

static float df_as(Df a) {
    return a.hi + a.lo;
}

static Df df_abs(Df a) {
    return a.hi < 0 || (a.hi == 0 && a.lo < 0) ? Df{-a.hi, -a.lo} : a;
}

static Df df_max(Df a, Df b) {
    return df_gt(a, b) ? a : b;
}

static Df df_min(Df a, Df b) {
    return df_gt(a, b) ? b : a;
}

static Df df_det(thread const Df* s) {
    const Df t0 = df_sub(df_mul(s[4], s[8]), df_mul(s[5], s[7]));
    const Df t1 = df_sub(df_mul(s[3], s[8]), df_mul(s[5], s[6]));
    const Df t2 = df_sub(df_mul(s[3], s[7]), df_mul(s[4], s[6]));
    return df_add(df_sub(df_mul(s[0], t0), df_mul(s[1], t1)), df_mul(s[2], t2));
}

constant Df kDf4Pi = {12.566370964050293f, -3.4969110629390343e-07f};
constant Df kDfLog2Pi = {1.8378770351409912f, 3.126835323996602e-08f};
constant Df kDfAreaPower = {1.6074999570846558f, 4.2915345943583816e-08f};
constant Df kDfAreaRoot = {0.6220839619636536f, 1.9373826987134635e-08f};
constant Df kDfJitter = {9.9999999392252903e-09f, 6.077470660972466e-17f};

// Knud Thomsen's ellipsoid surface area.
static Df df_area(Df x, Df y, Df z) {
    const Df xy = df_pow(df_mul(x, y), kDfAreaPower);
    const Df xz = df_pow(df_mul(x, z), kDfAreaPower);
    const Df yz = df_pow(df_mul(y, z), kDfAreaPower);
    const Df mean = df_div(df_add(df_add(xy, xz), yz), df_from(3));
    return df_mul(kDf4Pi, df_pow(mean, kDfAreaRoot));
}

// Rotation matrix of a quaternion, normalized with the given floor.
static void df_rotation(device const float* rot, uint i, thread Df* r) {
    const Df q0 = df_from(rot[i * 4]), q1 = df_from(rot[i * 4 + 1]), q2 = df_from(rot[i * 4 + 2]),
             q3 = df_from(rot[i * 4 + 3]);
    const Df inv = df_div(df_from(1), df_max(df_sqrt(df_add(df_add(df_mul(q0, q0), df_mul(q1, q1)),
                                                             df_add(df_mul(q2, q2), df_mul(q3, q3)))),
                                               df_from(1e-12f)));
    const Df w = df_mul(q0, inv), x = df_mul(q1, inv), y = df_mul(q2, inv), z = df_mul(q3, inv);
    const Df two = df_from(2);
    r[0] = df_sub(df_from(1), df_mul(two, df_add(df_mul(y, y), df_mul(z, z))));
    r[1] = df_mul(two, df_sub(df_mul(x, y), df_mul(w, z)));
    r[2] = df_mul(two, df_add(df_mul(x, z), df_mul(w, y)));
    r[3] = df_mul(two, df_add(df_mul(x, y), df_mul(w, z)));
    r[4] = df_sub(df_from(1), df_mul(two, df_add(df_mul(x, x), df_mul(z, z))));
    r[5] = df_mul(two, df_sub(df_mul(y, z), df_mul(w, x)));
    r[6] = df_mul(two, df_sub(df_mul(x, z), df_mul(w, y)));
    r[7] = df_mul(two, df_add(df_mul(y, z), df_mul(w, x)));
    r[8] = df_sub(df_from(1), df_mul(two, df_add(df_mul(x, x), df_mul(y, y))));
}

// ---------------------------------------------------------------------------
// Decimation neighbours, from export_decimate.slang: phase 0 bounds the
// Morton-ordered leaves of 8 points, 1 builds a level of the implicit tree
// from `begin`, 2 keeps the 16 nearest points of every point.

struct ExportDecimateParams {
    device const float* pos;
    device const uint* order;
    device float* boxes;
    device uint* neighbors;
    uint n, leaves, begin, padding;
};

// A squared distance is m * 2^e with m in [1, 2). Each pair is scaled by its own largest
// delta, so no scene-wide scale can underflow nearby distances or overflow distant ones.
struct Wd {
    int e;
    Df m;
};

constant int kWdZero = -1000000;
constant int kWdInf = 1000000;

static Wd wd_inf() {
    return {kWdInf, df_from(1)};
}

static int float_exponent(float x) {
    return int((as_type<uint>(x) >> 23) & 0xffu) - 127;
}

static bool wd_gt(Wd a, Wd b) {
    return a.e != b.e ? a.e > b.e : df_gt(a.m, b.m);
}

static Wd squared_distance(device const float* pos, uint i, uint j) {
    Df delta[3];
    int top = kWdZero;
    for (uint a = 0; a < 3; ++a) {
        delta[a] = df_sub(df_from(pos[i * 3 + a]), df_from(pos[j * 3 + a]));
        if (delta[a].hi != 0)
            top = max(top, float_exponent(delta[a].hi));
    }
    if (top == kWdZero)
        return {kWdZero, df_from(0)};
    Df d2 = df_from(0);
    for (uint a = 0; a < 3; ++a) {
        const Df scaled = df_scale2(delta[a], -top);
        d2 = df_add(d2, df_mul(scaled, scaled));
    }
    if (!isfinite(d2.hi))
        return wd_inf();
    const int t = float_exponent(d2.hi);
    return {2 * top + t, df_scale2(d2, -t)};
}

struct NeighbourHeap {
    Wd distance[16];
    uint ids[16];
    Wd worst;
    float search;
    uint worst_id;
};

static float unscaled_limit(Wd worst) {
    if (worst.e >= 128)
        return INFINITY;
    // Below the normal float range the float pre-filter cannot resolve distances; keep all.
    if (worst.e < -120)
        return as_type<float>(8u << 23);
    // Slightly wider than the double round-up so a short float-float residual
    // cannot prune a neighbour the double heap would keep.
    return df_round_up(df_scale2(df_mul(worst.m, df_from(1.000004f)), worst.e));
}

static void consider(thread NeighbourHeap& heap, device const float* pos, uint i, uint j) {
    if (j == i)
        return;
    float approx = 0;
    for (uint a = 0; a < 3; ++a) {
        const float delta = pos[i * 3 + a] - pos[j * 3 + a];
        approx += delta * delta;
    }
    if (approx > heap.search)
        return;
    const Wd d2 = squared_distance(pos, i, j);
    const bool farther = wd_gt(d2, heap.worst);
    const bool equal = !farther && !wd_gt(heap.worst, d2);
    if (farther || (equal && j >= heap.worst_id))
        return;
    int slot = 0;
    while (slot * 2 + 1 < 16) {
        int child = slot * 2 + 1;
        if (child + 1 < 16 && wd_gt(heap.distance[child + 1], heap.distance[child]))
            ++child;
        else if (child + 1 < 16 && !wd_gt(heap.distance[child], heap.distance[child + 1]) &&
                 heap.ids[child + 1] > heap.ids[child])
            ++child;
        const bool child_farther =
            wd_gt(heap.distance[child], d2) || (!wd_gt(d2, heap.distance[child]) && heap.ids[child] > j);
        if (!child_farther)
            break;
        heap.distance[slot] = heap.distance[child];
        heap.ids[slot] = heap.ids[child];
        slot = child;
    }
    heap.distance[slot] = d2;
    heap.ids[slot] = j;
    heap.worst = heap.ids[0] == 0xffffffffu ? wd_inf() : heap.distance[0];
    heap.worst_id = heap.ids[0];
    heap.search = unscaled_limit(heap.worst);
}

static float box_distance(float3 point, device const float* boxes, uint node) {
    float d = 0;
    for (uint a = 0; a < 3; ++a) {
        const float delta = max(max(boxes[node * 6 + a] - point[a], point[a] - boxes[node * 6 + 3 + a]), 0.0f);
        d += delta * delta;
    }
    return d;
}

kernel void export_decimate(constant ExportDecimateParams& p [[buffer(0)]], uint gid [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]], uint groups [[threadgroups_per_grid]]) {
    const uint first = gid * kExportWidth + tid, stride = groups * kExportWidth;
    if (kOp == 0) {
        for (uint leaf = first; leaf < p.leaves; leaf += stride) {
            float3 lo = float3(INFINITY), hi = float3(-INFINITY);
            for (uint t = leaf * 8u; t < (leaf + 1u) * 8u && t < p.n; ++t) {
                const uint i = p.order[t];
                const float3 v(p.pos[i * 3], p.pos[i * 3 + 1], p.pos[i * 3 + 2]);
                lo = min(lo, v);
                hi = max(hi, v);
            }
            const uint node = p.leaves + leaf;
            for (uint a = 0; a < 3; ++a) {
                p.boxes[node * 6 + a] = lo[a];
                p.boxes[node * 6 + 3 + a] = hi[a];
            }
        }
    } else if (kOp == 1) {
        for (uint t = first; t < p.begin; t += stride) {
            const uint i = p.begin + t;
            for (uint a = 0; a < 3; ++a) {
                p.boxes[i * 6 + a] = min(p.boxes[i * 12 + a], p.boxes[i * 12 + 6 + a]);
                p.boxes[i * 6 + 3 + a] = max(p.boxes[i * 12 + 3 + a], p.boxes[i * 12 + 9 + a]);
            }
        }
    } else {
        for (uint row = first; row < p.n; row += stride) {
            const uint i = p.order[row];
            const float3 point(p.pos[i * 3], p.pos[i * 3 + 1], p.pos[i * 3 + 2]);
            NeighbourHeap heap;
            for (uint a = 0; a < 16; ++a) {
                heap.distance[a] = wd_inf();
                heap.ids[a] = 0xffffffffu;
            }
            heap.worst = wd_inf();
            heap.search = INFINITY;
            heap.worst_id = 0xffffffffu;
            // Seed the heap from the Morton neighbourhood, then descend the tree.
            const uint seed_begin = row > 32u ? row - 32u : 0u, seed_end = min(p.n, seed_begin + 65u);
            for (uint cursor = seed_begin; cursor < seed_end; ++cursor)
                consider(heap, p.pos, i, p.order[cursor]);
            uint node = 1u;
            while (node != 0) {
                if (!(box_distance(point, p.boxes, node) > heap.search)) {
                    if (node < p.leaves) {
                        node *= 2u;
                        continue;
                    }
                    const uint begin = (node - p.leaves) * 8u;
                    for (uint cursor = begin; cursor < begin + 8u && cursor < p.n; ++cursor) {
                        if (cursor < seed_begin || cursor >= seed_end)
                            consider(heap, p.pos, i, p.order[cursor]);
                    }
                }
                // In the implicit tree, an even node has an unvisited right sibling.
                while ((node & 1u) != 0)
                    node >>= 1u;
                if (node != 0)
                    ++node;
            }
            for (uint a = 0; a < 16; ++a)
                p.neighbors[i * 16u + a] = heap.ids[a];
        }
    }
}

// ---------------------------------------------------------------------------
// Decimation edge costs, from export_decimate_cost.slang. Phase 0 caches the
// per-splat Gaussian terms, 1 keeps the four cheapest merges of every splat.
// Values round to float32 at the end, the precision the decimator sorts.

struct ExportCostParams {
    device const float* pos;
    device const float* rot;
    device const float* scale;
    device const float* opacity;
    device const float* dc;
    device const float* sh;
    device const uint* neighbors;
    device float* cache;
    device uint* idx;
    device float* cost;
    uint n, rest, pad0, pad1;
};

struct GCache {
    float r[9];
    float v[3];
    float invv[3];
    float sigma[9];
    float logdet;
    float mass;
    Df sample[3];
    Df self_log;
};

static Df df_floor(Df a, float bound) {
    return df_as(a) > bound ? a : df_from(bound);
}

// Mahalanobis log density of x under the cached Gaussian at m.
static Df df_logpdf(thread const Df* x, float3 m, thread const GCache& c) {
    Df d[3];
    for (uint a = 0; a < 3; ++a)
        d[a] = df_sub(x[a], df_from(m[a]));
    Df q = df_from(0);
    for (uint a = 0; a < 3; ++a) {
        const Df y = df_add(df_add(df_mul(d[0], df_from(c.r[a])), df_mul(d[1], df_from(c.r[3 + a]))),
                            df_mul(d[2], df_from(c.r[6 + a])));
        q = df_add(q, df_mul(df_mul(y, y), df_from(c.invv[a])));
    }
    return df_mul(df_from(-0.5f), df_add(df_add(df_mul(df_from(3), kDfLog2Pi), df_from(c.logdet)), q));
}

static void cache_one(constant ExportCostParams& p, uint i, thread GCache& c) {
    Df variance[3], axes[3];
    Df ld = df_from(0);
    for (uint a = 0; a < 3; ++a) {
        const Df s = df_floor(df_exp_precise(df_from(p.scale[i * 3 + a])), 1e-12f);
        axes[a] = s;
        variance[a] = df_add(df_mul(s, s), kDfJitter);
        c.v[a] = df_as(variance[a]);
        c.invv[a] = df_as(df_div(df_from(1), df_floor(variance[a], 1e-30f)));
        ld = df_add(ld, df_log(df_floor(variance[a], 1e-30f)));
    }
    c.logdet = df_as(ld);
    c.mass = df_as(df_add(df_mul(df_sigmoid(p.opacity[i]), df_area(axes[0], axes[1], axes[2])), df_from(1e-12f)));
    const Df q0 = df_from(p.rot[i * 4]), q1 = df_from(p.rot[i * 4 + 1]), q2 = df_from(p.rot[i * 4 + 2]),
             q3 = df_from(p.rot[i * 4 + 3]);
    const Df inv = df_div(df_from(1), df_floor(df_sqrt(df_add(df_add(df_mul(q0, q0), df_mul(q1, q1)),
                                                               df_add(df_mul(q2, q2), df_mul(q3, q3)))),
                                                 1e-12f));
    const Df w = df_mul(q0, inv), x = df_mul(q1, inv), y = df_mul(q2, inv), z = df_mul(q3, inv);
    const Df two = df_from(2);
    c.r[0] = df_as(df_sub(df_from(1), df_mul(two, df_add(df_mul(y, y), df_mul(z, z)))));
    c.r[1] = df_as(df_mul(two, df_sub(df_mul(x, y), df_mul(w, z))));
    c.r[2] = df_as(df_mul(two, df_add(df_mul(x, z), df_mul(w, y))));
    c.r[3] = df_as(df_mul(two, df_add(df_mul(x, y), df_mul(w, z))));
    c.r[4] = df_as(df_sub(df_from(1), df_mul(two, df_add(df_mul(x, x), df_mul(z, z)))));
    c.r[5] = df_as(df_mul(two, df_sub(df_mul(y, z), df_mul(w, x))));
    c.r[6] = df_as(df_mul(two, df_sub(df_mul(x, z), df_mul(w, y))));
    c.r[7] = df_as(df_mul(two, df_add(df_mul(y, z), df_mul(w, x))));
    c.r[8] = df_as(df_sub(df_from(1), df_mul(two, df_add(df_mul(x, x), df_mul(y, y)))));
    for (uint a = 0; a < 3; ++a) {
        for (uint b = a; b < 3; ++b) {
            Df acc = df_from(0);
            for (uint k = 0; k < 3; ++k)
                acc = df_add(acc, df_mul(df_mul(df_from(c.r[a * 3 + k]), df_from(c.r[b * 3 + k])), variance[k]));
            c.sigma[a * 3 + b] = c.sigma[b * 3 + a] = df_as(acc);
        }
    }
    const Df zed[3] = {{1.6264322996139526f, 8.576314769470628e-09f},
                       {0.003369733924046159f, 1.0921582388467854e-10f},
                       {1.0509958267211914f, 1.7497320925485838e-08f}};
    Df scaled[3];
    for (uint a = 0; a < 3; ++a)
        scaled[a] = df_mul(zed[a], df_sqrt(df_floor(df_from(c.v[a]), 0)));
    for (uint a = 0; a < 3; ++a) {
        c.sample[a] = df_add(df_from(p.pos[i * 3 + a]),
                             df_add(df_add(df_mul(scaled[0], df_from(c.r[a * 3])), df_mul(scaled[1], df_from(c.r[a * 3 + 1]))),
                                    df_mul(scaled[2], df_from(c.r[a * 3 + 2]))));
    }
    c.self_log = df_logpdf(c.sample, float3(p.pos[i * 3], p.pos[i * 3 + 1], p.pos[i * 3 + 2]), c);
}

static Df df_logadd(Df a, Df b) {
    const Df m = df_as(a) > df_as(b) ? a : b;
    return df_add(m, df_log(df_add(df_exp(df_sub(a, m)), df_exp(df_sub(b, m)))));
}

static float edge_cost(constant ExportCostParams& p, thread const GCache& a, thread const GCache& b, uint i, uint j) {
    const Df w = df_add(df_from(a.mass), df_from(b.mass));
    Df pa = df_div(df_from(a.mass), df_as(w) > 0 ? w : df_from(1));
    const float pf = df_as(pa);
    pa = pf < 1e-12f ? df_from(1e-12f) : (pf > 1 - 1e-12f ? df_from(1 - 1e-12f) : pa);
    const Df qa = df_sub(df_from(1), pa);
    const Df lp = df_log(pa), lq = df_log(qa);
    const float3 xi(p.pos[i * 3], p.pos[i * 3 + 1], p.pos[i * 3 + 2]), xj(p.pos[j * 3], p.pos[j * 3 + 1], p.pos[j * 3 + 2]);
    Df d[3], e[3], s[9];
    for (uint c = 0; c < 3; ++c) {
        const Df m = df_add(df_mul(pa, df_from(xi[c])), df_mul(qa, df_from(xj[c])));
        d[c] = df_sub(df_from(xi[c]), m);
        e[c] = df_sub(df_from(xj[c]), m);
    }
    for (uint c = 0; c < 9; ++c) {
        const uint row = c / 3, col = c - row * 3;
        s[c] = df_add(df_add(df_add(df_mul(pa, df_from(a.sigma[c])), df_mul(qa, df_from(b.sigma[c]))),
                             df_mul(pa, df_mul(d[row], d[col]))),
                      df_mul(qa, df_mul(e[row], e[col])));
    }
    const Df s13 = df_mul(df_from(0.5f), df_add(s[1], s[3]));
    const Df s26 = df_mul(df_from(0.5f), df_add(s[2], s[6]));
    const Df s57 = df_mul(df_from(0.5f), df_add(s[5], s[7]));
    s[1] = s[3] = s13;
    s[2] = s[6] = s26;
    s[5] = s[7] = s57;
    s[0] = df_add(s[0], kDfJitter);
    s[4] = df_add(s[4], kDfJitter);
    s[8] = df_add(s[8], kDfJitter);
    const Df cross_a = df_logadd(df_add(lp, a.self_log), df_add(lq, df_logpdf(a.sample, xj, b)));
    const Df cross_b = df_logadd(df_add(lp, df_logpdf(b.sample, xi, a)), df_add(lq, b.self_log));
    Df cost = df_add(df_add(df_mul(pa, cross_a), df_mul(qa, cross_b)),
                     df_mul(df_from(0.5f), df_add(df_add(df_mul(df_from(3), kDfLog2Pi), df_log(df_floor(df_det(s), 1e-30f))),
                                                  df_from(3))));
    for (uint c = 0; c < 3 + p.rest * 3; ++c) {
        const float left = c < 3 ? p.dc[i * 3 + c] : p.sh[i * p.rest * 3 + c - 3];
        const float right = c < 3 ? p.dc[j * 3 + c] : p.sh[j * p.rest * 3 + c - 3];
        const float delta = left - right;
        cost = df_add(cost, df_from(delta * delta));
    }
    return df_as(cost);
}

static void store_cache(device float* o, thread const GCache& c) {
    for (uint a = 0; a < 9; ++a)
        o[a] = c.r[a];
    for (uint a = 0; a < 3; ++a)
        o[9 + a] = c.invv[a];
    for (uint a = 0; a < 9; ++a)
        o[12 + a] = c.sigma[a];
    o[21] = c.logdet;
    o[22] = c.mass;
    for (uint a = 0; a < 3; ++a) {
        o[23 + 2 * a] = c.sample[a].hi;
        o[24 + 2 * a] = c.sample[a].lo;
    }
    o[29] = c.self_log.hi;
    o[30] = c.self_log.lo;
}

static void load_cache(device const float* o, thread GCache& c) {
    for (uint a = 0; a < 9; ++a)
        c.r[a] = o[a];
    for (uint a = 0; a < 3; ++a)
        c.invv[a] = o[9 + a];
    for (uint a = 0; a < 9; ++a)
        c.sigma[a] = o[12 + a];
    c.logdet = o[21];
    c.mass = o[22];
    for (uint a = 0; a < 3; ++a)
        c.sample[a] = {o[23 + 2 * a], o[24 + 2 * a]};
    c.self_log = {o[29], o[30]};
}

kernel void export_decimate_cost(constant ExportCostParams& p [[buffer(0)]], uint gid [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]], uint groups [[threadgroups_per_grid]]) {
    for (uint i = gid * kExportWidth + tid; i < p.n; i += groups * kExportWidth) {
        if (kOp == 0) {
            GCache built;
            cache_one(p, i, built);
            store_cache(p.cache + i * 32, built);
            continue;
        }
        GCache mine;
        load_cache(p.cache + i * 32, mine);
        float best[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
        uint chosen[4] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        for (uint a = 0; a < 16; ++a) {
            const uint j = p.neighbors[i * 16 + a];
            if (j == 0xffffffffu)
                continue;
            GCache other;
            load_cache(p.cache + j * 32, other);
            const float value = edge_cost(p, mine, other, i, j);
            if (!isfinite(value) || value > best[3] || (value == best[3] && j >= chosen[3]))
                continue;
            int at = 3;
            while (at > 0 && (value < best[at - 1] || (value == best[at - 1] && j < chosen[at - 1]))) {
                best[at] = best[at - 1];
                chosen[at] = chosen[at - 1];
                --at;
            }
            best[at] = value;
            chosen[at] = j;
        }
        for (uint a = 0; a < 4; ++a) {
            p.idx[i * 4 + a] = chosen[a];
            p.cost[i * 4 + a] = chosen[a] == 0xffffffffu ? INFINITY : best[a];
        }
    }
}

// ---------------------------------------------------------------------------
// Decimation merge, from export_decimate_merge.slang: moment matching for one
// kept row, groups of at most four members. Float-float sums track the double
// host merge; stored channels are float32. The table holds the addresses of
// the attributes, the group tables and the outputs.

struct ExportMergeParams {
    device const ulong* table;
    uint n, rest, pad0, pad1;
};

struct Splats {
    device const float* pos;
    device const float* rot;
    device const float* scale;
    device const float* opacity;
    device const float* dc;
    device const float* sh;
};

struct MergedSplats {
    device float* pos;
    device float* rot;
    device float* scale;
    device float* opacity;
    device float* dc;
    device float* sh;
};

static Df splat_mass(Splats in, uint i, float eps) {
    Df axes[3];
    for (uint a = 0; a < 3; ++a)
        axes[a] = df_max(df_exp(df_from(in.scale[i * 3 + a])), df_from(1e-12f));
    return df_add(df_mul(df_sigmoid(in.opacity[i]), df_area(axes[0], axes[1], axes[2])), df_from(eps));
}

static void covariance(thread const Df* r, thread const Df* variance, thread Df* s) {
    for (uint a = 0; a < 3; ++a) {
        for (uint b = a; b < 3; ++b) {
            Df acc = df_from(0);
            for (uint k = 0; k < 3; ++k)
                acc = df_add(acc, df_mul(df_mul(r[a * 3 + k], r[b * 3 + k]), variance[k]));
            s[a * 3 + b] = s[b * 3 + a] = acc;
        }
    }
}

// Jacobi eigen-decomposition into descending scales and a proper rotation.
static void decompose(thread Df* a, thread Df* scales, thread Df* quaternion) {
    Df v[9] = {df_from(1), df_from(0), df_from(0), df_from(0), df_from(1), df_from(0), df_from(0), df_from(0), df_from(1)};
    for (int iter = 0; iter < 24; ++iter) {
        int p = 0, q = 1;
        Df largest = df_abs(a[1]);
        if (df_gt(df_abs(a[2]), largest)) {
            q = 2;
            largest = df_abs(a[2]);
        }
        if (df_gt(df_abs(a[5]), largest)) {
            p = 1;
            q = 2;
            largest = df_abs(a[5]);
        }
        if (df_as(largest) < 1e-12f)
            break;
        const int pp = 4 * p, qq = 4 * q, pq = 3 * p + q;
        const Df app = a[pp], aqq = a[qq], apq = a[pq];
        const Df tau = df_div(df_sub(aqq, app), df_mul(df_from(2), apq));
        const float sign = df_as(tau) > 0 ? 1 : (df_as(tau) < 0 ? -1 : 0);
        const Df t = df_div(df_from(sign), df_add(df_abs(tau), df_sqrt(df_add(df_from(1), df_mul(tau, tau)))));
        const Df c = df_div(df_from(1), df_sqrt(df_add(df_from(1), df_mul(t, t))));
        const Df s = df_mul(t, c);
        for (int k = 0; k < 3; ++k) {
            if (k == p || k == q)
                continue;
            const int kp = 3 * k + p, kq = 3 * k + q;
            const Df akp = a[kp], akq = a[kq];
            a[kp] = a[3 * p + k] = df_sub(df_mul(c, akp), df_mul(s, akq));
            a[kq] = a[3 * q + k] = df_add(df_mul(s, akp), df_mul(c, akq));
        }
        a[pp] = df_add(df_sub(df_mul(df_mul(c, c), app), df_mul(df_mul(df_from(2), s), df_mul(c, apq))),
                       df_mul(df_mul(s, s), aqq));
        a[qq] = df_add(df_add(df_mul(df_mul(s, s), app), df_mul(df_mul(df_from(2), s), df_mul(c, apq))),
                       df_mul(df_mul(c, c), aqq));
        a[pq] = a[3 * q + p] = df_from(0);
        for (int k = 0; k < 3; ++k) {
            const int kp = 3 * k + p, kq = 3 * k + q;
            const Df vkp = v[kp], vkq = v[kq];
            v[kp] = df_sub(df_mul(c, vkp), df_mul(s, vkq));
            v[kq] = df_add(df_mul(s, vkp), df_mul(c, vkq));
        }
    }
    int order[3] = {0, 1, 2};
    for (int i = 1; i < 3; ++i) {
        const int key = order[i];
        int j = i;
        while (j > 0 && df_gt(a[4 * key], a[4 * order[j - 1]])) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = key;
    }
    Df r[9];
    for (uint c = 0; c < 3; ++c) {
        scales[c] = df_sqrt(df_max(a[4 * order[c]], df_from(1e-18f)));
        for (uint row = 0; row < 3; ++row)
            r[row * 3 + c] = v[row * 3 + order[c]];
    }
    if (df_gt(df_from(0), df_det(r))) {
        r[2] = df_sub(df_from(0), r[2]);
        r[5] = df_sub(df_from(0), r[5]);
        r[8] = df_sub(df_from(0), r[8]);
    }
    Df w, x, y, z;
    const Df tr = df_add(df_add(r[0], r[4]), r[8]);
    if (df_gt(tr, df_from(0))) {
        const Df s = df_mul(df_sqrt(df_add(tr, df_from(1))), df_from(2));
        w = df_mul(df_from(0.25f), s);
        x = df_div(df_sub(r[7], r[5]), s);
        y = df_div(df_sub(r[2], r[6]), s);
        z = df_div(df_sub(r[3], r[1]), s);
    } else if (df_gt(r[0], r[4]) && df_gt(r[0], r[8])) {
        const Df s = df_mul(df_sqrt(df_add(df_sub(df_sub(df_from(1), r[4]), r[8]), r[0])), df_from(2));
        w = df_div(df_sub(r[7], r[5]), s);
        x = df_mul(df_from(0.25f), s);
        y = df_div(df_add(r[1], r[3]), s);
        z = df_div(df_add(r[2], r[6]), s);
    } else if (df_gt(r[4], r[8])) {
        const Df s = df_mul(df_sqrt(df_add(df_sub(df_sub(df_from(1), r[0]), r[8]), r[4])), df_from(2));
        w = df_div(df_sub(r[2], r[6]), s);
        x = df_div(df_add(r[1], r[3]), s);
        y = df_mul(df_from(0.25f), s);
        z = df_div(df_add(r[5], r[7]), s);
    } else {
        const Df s = df_mul(df_sqrt(df_add(df_sub(df_sub(df_from(1), r[0]), r[4]), r[8])), df_from(2));
        w = df_div(df_sub(r[3], r[1]), s);
        x = df_div(df_add(r[2], r[6]), s);
        y = df_div(df_add(r[5], r[7]), s);
        z = df_mul(df_from(0.25f), s);
    }
    const Df inv = df_div(df_from(1), df_max(df_sqrt(df_add(df_add(df_mul(w, w), df_mul(x, x)),
                                                             df_add(df_mul(y, y), df_mul(z, z)))),
                                               df_from(1e-12f)));
    quaternion[0] = df_mul(w, inv);
    quaternion[1] = df_mul(x, inv);
    quaternion[2] = df_mul(y, inv);
    quaternion[3] = df_mul(z, inv);
}

static void copy_one(Splats in, MergedSplats out, uint rest, uint i, uint row) {
    for (uint a = 0; a < 3; ++a) {
        out.pos[row * 3 + a] = in.pos[i * 3 + a];
        out.scale[row * 3 + a] = in.scale[i * 3 + a];
        out.dc[row * 3 + a] = in.dc[i * 3 + a];
    }
    for (uint a = 0; a < 4; ++a)
        out.rot[row * 4 + a] = in.rot[i * 4 + a];
    out.opacity[row] = in.opacity[i];
    for (uint a = 0; a < rest * 3; ++a)
        out.sh[row * rest * 3 + a] = in.sh[i * rest * 3 + a];
}

static void merge_one(Splats in, MergedSplats out, device const uint* members, uint rest, uint begin, uint count,
                      uint row) {
    Df weights[4];
    Df mass = df_from(0);
    for (uint m = 0; m < count; ++m) {
        weights[m] = splat_mass(in, members[begin + m], 1e-30f);
        mass = df_add(mass, weights[m]);
    }
    Df mean[3] = {df_from(0), df_from(0), df_from(0)};
    for (uint m = 0; m < count; ++m) {
        weights[m] = df_div(weights[m], mass);
        const uint id = members[begin + m];
        for (uint a = 0; a < 3; ++a)
            mean[a] = df_add(mean[a], df_mul(weights[m], df_from(in.pos[id * 3 + a])));
    }
    Df sigma[9];
    for (uint a = 0; a < 9; ++a)
        sigma[a] = df_from(0);
    for (uint m = 0; m < count; ++m) {
        const uint id = members[begin + m];
        Df r[9], variance[3], delta[3], own[9];
        df_rotation(in.rot, id, r);
        for (uint a = 0; a < 3; ++a) {
            const Df axis = df_max(df_exp(df_from(in.scale[id * 3 + a])), df_from(1e-12f));
            variance[a] = df_mul(axis, axis);
            delta[a] = df_sub(df_from(in.pos[id * 3 + a]), mean[a]);
        }
        covariance(r, variance, own);
        for (uint a = 0; a < 9; ++a)
            sigma[a] = df_add(sigma[a], df_mul(weights[m], df_add(df_mul(delta[a / 3], delta[a % 3]), own[a])));
    }
    sigma[0] = df_add(sigma[0], df_from(1e-8f));
    sigma[4] = df_add(sigma[4], df_from(1e-8f));
    sigma[8] = df_add(sigma[8], df_from(1e-8f));
    Df scales[3], quaternion[4];
    decompose(sigma, scales, quaternion);
    for (uint a = 0; a < 3; ++a) {
        out.pos[row * 3 + a] = df_as(mean[a]);
        out.scale[row * 3 + a] = df_as(df_log(scales[a]));
    }
    for (uint a = 0; a < 4; ++a)
        out.rot[row * 4 + a] = df_as(quaternion[a]);
    const Df area = df_max(df_area(scales[0], scales[1], scales[2]), df_from(1e-30f));
    // The reference clamps in double; both bounds need float-float constants.
    const Df alpha = df_max({1.0000000116860974e-07f, -1.1686097468332243e-15f},
                            df_min({0.99999988079071045f, 1.9209290158528347e-08f}, df_div(mass, area)));
    out.opacity[row] = df_as(df_log(df_div(alpha, df_sub(df_from(1), alpha))));
    for (uint c = 0; c < 3 + rest * 3; ++c) {
        Df acc = df_from(0);
        for (uint m = 0; m < count; ++m) {
            const uint id = members[begin + m];
            acc = df_add(acc, df_mul(weights[m], df_from(c < 3 ? in.dc[id * 3 + c] : in.sh[id * rest * 3 + c - 3])));
        }
        if (c < 3)
            out.dc[row * 3 + c] = df_as(acc);
        else
            out.sh[row * rest * 3 + c - 3] = df_as(acc);
    }
}

kernel void export_decimate_merge(constant ExportMergeParams& p [[buffer(0)]], uint gid [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]], uint groups [[threadgroups_per_grid]]) {
    device const ulong* t = p.table;
    const Splats in = {(device const float*)t[0], (device const float*)t[1], (device const float*)t[2],
                       (device const float*)t[3], (device const float*)t[4], (device const float*)t[5]};
    device const int* member = (device const int*)t[6];
    device const uint* minimum = (device const uint*)t[7];
    device const uint* members = (device const uint*)t[8];
    device const uint* offsets = (device const uint*)t[9];
    device const uint* rows = (device const uint*)t[10];
    const MergedSplats out = {(device float*)t[11], (device float*)t[12], (device float*)t[13],
                              (device float*)t[14], (device float*)t[15], (device float*)t[16]};
    for (uint i = gid * kExportWidth + tid; i < p.n; i += groups * kExportWidth) {
        const int g = member[i];
        if (g < 0)
            copy_one(in, out, p.rest, i, rows[i]);
        else if (i == minimum[g])
            merge_one(in, out, members, p.rest, offsets[g], offsets[g + 1] - offsets[g], rows[i]);
    }
}

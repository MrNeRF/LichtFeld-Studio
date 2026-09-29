// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// FastRasterOps forward: ports of fastgs kernels_forward.cuh, visibility and
// kernel_utils.cuh. Tile/depth keys sort with the LSD radix sort below.
// Threadgroup staging and simdgroup-reduced atomics follow metal-gauss
// (https://github.com/nandometzger/metal-gauss, MIT, Nando Metzger).
//
// Per-primitive buffers cover every primitive (no visible compaction); a
// primitive is visible when its touched-tile count is nonzero.

constant constexpr uint kFastTileSize = 16u;
constant constexpr uint kFastTilePixels = 256u;
constant constexpr uint kFastBlendThreads = 128u;
constant constexpr float kFastMinAlpha = 1.0f / 255.0f;
constant constexpr float kFastMinAlphaRcp = 255.0f;
constant constexpr float kFastMaxFragmentAlpha = 0.999f;
constant constexpr float kFastMinDeterminant = 1e-6f;
constant constexpr float kFastTransmittanceThreshold = 1e-4f;
constant constexpr float kFastMaxRawScale = 20.0f;
constant constexpr float kFastMaxBlendColor = 4.0f;
constant constexpr float kFastDilation = 0.3f;
constant constexpr float kFastDilationMip = 0.1f;
constant constexpr uint kFastInvalid = 0xffffffffu;

// Function constants of this family use indices 60..69.
constant uint kFastShBases [[function_constant(60)]];
constant uint kFastRenderDepth [[function_constant(61)]];
constant uint kFastRenderNormal [[function_constant(62)]];
constant uint kFastDensification [[function_constant(63)]];
constant uint kFastNormalChannel [[function_constant(64)]];
constant uint kFastMipFilter [[function_constant(65)]];
constant uint kFastShLayoutRest [[function_constant(66)]];
constant uint kFastShStorage [[function_constant(67)]];
// Backward only: whether the blend receives a depth gradient and edge weights.
constant uint kFastDepthGrad [[function_constant(68)]];
constant uint kFastEdgeWeight [[function_constant(69)]];

// Screen mean plus the conservative pixel AABB [x0, x1) x [y0, y1) of the
// contribution ellipse.
struct FastMeanBox {
    float2 mean;
    ushort4 bbox;
};

struct FastTileInfo {
    ushort4 bounds;
    uint depth_key;
    uint unused;
};

enum FastShStorage : uint {
    kFastShFloat32 = 0u,
    kFastShFloat16 = 1u,
    kFastShQ16 = 2u,
};

// Pad-dropped Q16 cells per primitive of the SH layout.
static uint fast_q16_cells() { return kFastShStorage == kFastShQ16 ? kFastShLayoutRest * 3u : 0u; }

static float3 fast_safe_normalize(const float3 v) {
    const float norm_sq = dot(v, v);
    if (norm_sq < 1e-12f)
        return float3(0.0f, 0.0f, 1.0f);
    return v * rsqrt(norm_sq);
}

// Rotation matrix from an unnormalized (r, x, y, z) quaternion; rows.
struct FastRotation {
    float3 r0, r1, r2;
    float qxx, qyy, qzz, qxy, qxz, qyz, qrx, qry, qrz, inv_q_norm_sq;
};

static FastRotation fast_rotation(const float4 q) {
    FastRotation m;
    const float qr = q.x, qx = q.y, qy = q.z, qz = q.w;
    const float q_norm_sq = qr * qr + qx * qx + qy * qy + qz * qz;
    m.inv_q_norm_sq = 2.0f / fmax(q_norm_sq, 1e-8f);
    const float s = m.inv_q_norm_sq;
    m.qxx = qx * qx * s;
    m.qyy = qy * qy * s;
    m.qzz = qz * qz * s;
    m.qxy = qx * qy * s;
    m.qxz = qx * qz * s;
    m.qyz = qy * qz * s;
    m.qrx = qr * qx * s;
    m.qry = qr * qy * s;
    m.qrz = qr * qz * s;
    m.r0 = float3(1.0f - (m.qyy + m.qzz), m.qxy - m.qrz, m.qry + m.qxz);
    m.r1 = float3(m.qrz + m.qxy, 1.0f - (m.qxx + m.qzz), m.qyz - m.qrx);
    m.r2 = float3(m.qxz - m.qry, m.qrx + m.qyz, 1.0f - (m.qxx + m.qyy));
    return m;
}

// Upper triangle of R diag(variance) R^T: (m11, m12, m13, m22, m23, m33).
struct FastCov3 {
    float m11, m12, m13, m22, m23, m33;
};

static FastCov3 fast_cov3d(const FastRotation m, const float3 variance) {
    const float3 s0 = m.r0 * variance, s1 = m.r1 * variance, s2 = m.r2 * variance;
    return {dot(s0, m.r0), dot(s0, m.r1), dot(s0, m.r2), dot(s1, m.r1), dot(s1, m.r2), dot(s2, m.r2)};
}

static float3 fast_cov_mul(const FastCov3 c, const float3 v) {
    return float3(v.x * c.m11 + v.y * c.m12 + v.z * c.m13, v.x * c.m12 + v.y * c.m22 + v.z * c.m23,
                  v.x * c.m13 + v.y * c.m23 + v.z * c.m33);
}

// SH-rest coefficient c (0..14) of primitive p, decoded from float, IEEE half
// or Q16 storage (see sh_storage.metal).
static float3 fast_sh_rest(device const uchar* shN, device const float2* bounds, const uint storage,
                           const uint p, const uint coeff, const uint layout_rest, const uint q16_cells) {
    if (storage == kFastShQ16) {
        const uint base = coeff * 3u;
        if (base + 2u >= q16_cells)
            return float3(0.0f);
        device const ushort* codes = reinterpret_cast<device const ushort*>(shN);
        const float2 mm = bounds[p / 256u];
        return float3(sh_q16_decode(codes[sh_q16_index(p, base, q16_cells)], mm.x, mm.y),
                      sh_q16_decode(codes[sh_q16_index(p, base + 1u, q16_cells)], mm.x, mm.y),
                      sh_q16_decode(codes[sh_q16_index(p, base + 2u, q16_cells)], mm.x, mm.y));
    }
    if (layout_rest == 0u)
        return float3(0.0f);
    float3 result;
    for (uint c = 0; c < 3u; ++c) {
        const uint linear = coeff * 3u + c;
        const uint index = sh_swizzled_index(p, linear / 4u, layout_rest) * 4u + linear % 4u;
        result[c] = storage == kFastShFloat16 ? float(reinterpret_cast<device const half*>(shN)[index])
                                              : reinterpret_cast<device const float*>(shN)[index];
    }
    return result;
}

// Real SH basis of the rest coefficients for a unit direction.
static void fast_sh_basis(const float3 d, thread float* basis) {
    const float x = d.x, y = d.y, z = d.z;
    const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z;
    basis[0] = -0.48860251190291987f * y;
    basis[1] = 0.48860251190291987f * z;
    basis[2] = -0.48860251190291987f * x;
    basis[3] = 1.0925484305920792f * xy;
    basis[4] = -1.0925484305920792f * yz;
    basis[5] = 0.94617469575755997f * zz - 0.31539156525251999f;
    basis[6] = -1.0925484305920792f * xz;
    basis[7] = 0.54627421529603959f * xx - 0.54627421529603959f * yy;
    basis[8] = 0.59004358992664352f * y * (-3.0f * xx + yy);
    basis[9] = 2.8906114426405538f * xy * z;
    basis[10] = 0.45704579946446572f * y * (1.0f - 5.0f * zz);
    basis[11] = 0.3731763325901154f * z * (5.0f * zz - 3.0f);
    basis[12] = 0.45704579946446572f * x * (1.0f - 5.0f * zz);
    basis[13] = 1.4453057213202769f * z * (xx - yy);
    basis[14] = 0.59004358992664352f * x * (-xx + 3.0f * yy);
}

// Port of convert_sh_to_color: 0.5 + C0 * sh0 + sum of the active rest bands.
static float3 fast_sh_color(device const packed_float3* sh0, device const uchar* shN, device const float2* bounds,
                            const uint storage, const uint p, const float3 mean, const float3 camera,
                            const uint layout_rest, const uint q16_cells) {
    float3 result = 0.5f + 0.28209479177387814f * float3(sh0[p]);
    if (kFastShBases <= 1u)
        return result;
    float basis[15];
    fast_sh_basis(fast_safe_normalize(mean - camera), basis);
    const uint rest = kFastShBases > 9u ? 15u : (kFastShBases > 4u ? 8u : 3u);
    for (uint i = 0; i < rest; ++i)
        result += basis[i] * fast_sh_rest(shN, bounds, storage, p, i, layout_rest, q16_cells);
    return result;
}

static uint fast_depth_key(const float depth, const uint depth_bits) {
    if (depth_bits == 0u)
        return 0u;
    float normalized = (2.0f * depth + 1.0f) / (depth + 1.0f);
    normalized = fmin(fmax(normalized, 1.0f), as_type<float>(0x3fffffffu));
    return (as_type<uint>(normalized) & 0x7fffffu) >> (23u - depth_bits);
}

static int fast_floor_int(const float x) { return int(floor(clamp(x, -1.0e9f, 1.0e9f))); }
static int fast_ceil_int(const float x) { return int(ceil(clamp(x, -1.0e9f, 1.0e9f))); }

// Half-open tile AABB [x0, x1) x [y0, y1) (compute_screen_tile_bounds).
static uint4 fast_screen_tile_bounds(const float2 mean, const float extent_x, const float extent_y,
                                     const uint grid_w, const uint grid_h) {
    const float t = float(kFastTileSize);
    const int x_min = max(0, fast_ceil_int((mean.x - extent_x) / t) - 1);
    const int x_max = max(0, fast_floor_int((mean.x + extent_x) / t) + 1);
    const int y_min = max(0, fast_ceil_int((mean.y - extent_y) / t) - 1);
    const int y_max = max(0, fast_floor_int((mean.y + extent_y) / t) + 1);
    return uint4(min(grid_w, uint(x_min)), min(grid_w, uint(x_max)), min(grid_h, uint(y_min)),
                 min(grid_h, uint(y_max)));
}

static float2 fast_ellipse_range(const float3 conic, const float radius_sq, const float y0, const float y1) {
    const float a = conic.x, b = conic.y, c = conic.z;
    const float det = fmax(a * c - b * b, 1e-20f);
    const float ym = -b / c * sqrt(fmax(c * radius_sq / det, 0.0f));
    const float v0 = fmin(fmax(-ym, y0), y1);
    const float v1 = fmin(fmax(ym, y0), y1);
    const float bv0 = -b * v0;
    const float bv1 = -b * v1;
    const float inv_a = 1.0f / a;
    const float x0 = inv_a * (bv0 - sqrt(fmax(bv0 * bv0 - a * (c * v0 * v0 - radius_sq), 0.0f)));
    const float x1 = inv_a * (bv1 + sqrt(fmax(bv1 * bv1 - a * (c * v1 * v1 - radius_sq), 0.0f)));
    return float2(x0, x1);
}

static uint fast_tile_clamped(const float coord, const uint lo, const uint hi, const int extra) {
    const int tile = fast_floor_int(coord / float(kFastTileSize)) + extra;
    return uint(min(max(tile, int(lo)), int(hi)));
}

// Tiles touched in one scan row (scan_along_x) or column; ellipse_touched_tile_span.
static uint2 fast_tile_span(const float3 conic, const float radius_sq, const float2 shifted, const bool along_x,
                            const uint scan, const uint cross0, const uint cross1) {
    const float t = float(kFastTileSize);
    if (along_x) {
        const float y0 = float(scan * kFastTileSize) - shifted.y;
        const float2 bound = fast_ellipse_range(conic, radius_sq, y0, y0 + t);
        return uint2(fast_tile_clamped(bound.x + shifted.x, cross0, cross1, 0),
                     fast_tile_clamped(bound.y + shifted.x, cross0, cross1, 1));
    }
    const float x0 = float(scan * kFastTileSize) - shifted.x;
    const float2 bound = fast_ellipse_range(float3(conic.z, conic.y, conic.x), radius_sq, x0, x0 + t);
    return uint2(fast_tile_clamped(bound.x + shifted.y, cross0, cross1, 0),
                 fast_tile_clamped(bound.y + shifted.y, cross0, cross1, 1));
}

struct FastTileWalk {
    float3 conic;
    float radius_sq;
    float2 shifted;
    bool along_x;
    uint scan0, scan1, cross0, cross1;
};

static FastTileWalk fast_tile_walk(const float2 mean, const float3 conic, const float opacity, const uint4 bounds) {
    FastTileWalk w;
    w.conic = conic;
    w.radius_sq = 2.0f * log(opacity * kFastMinAlphaRcp);
    w.shifted = mean - 0.5f;
    w.along_x = (bounds.w - bounds.z) <= (bounds.y - bounds.x);
    w.scan0 = w.along_x ? bounds.z : bounds.x;
    w.scan1 = w.along_x ? bounds.w : bounds.y;
    w.cross0 = w.along_x ? bounds.x : bounds.z;
    w.cross1 = w.along_x ? bounds.y : bounds.w;
    return w;
}

static uint2 fast_walk_span(const FastTileWalk w, const uint scan) {
    return fast_tile_span(w.conic, w.radius_sq, w.shifted, w.along_x, scan, w.cross0, w.cross1);
}

struct FastPreprocessParams {
    device const packed_float3* means;
    device const packed_float3* scales;
    device const float4* rotations;
    device const float* opacities;
    device const packed_float3* sh0;
    device const uchar* shN;
    device const float2* sh_bounds;
    device const float4* view;
    device const float* camera;
    device FastMeanBox* mean_box;
    device float4* conic_opacity;
    device float4* color_depth;
    device FastTileInfo* tile_info;
    device uint* n_touched;
    device float4* normals;
    device atomic_uint* max_screen_share;
    uint n, grid_w, grid_h, depth_bits;
    uint mip_filter, unused0, unused1, unused2;
    float fx, fy, cx, cy;
    float clip_left, clip_right, clip_top, clip_bottom;
    float near_plane, far_plane;
};

// Port of preprocess_cu: projection, EWA covariance, culling, exact tile count,
// SH color and the camera-space normal.
kernel void fast_preprocess(constant FastPreprocessParams& p [[buffer(0)]], const uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n)
        return;
    p.n_touched[idx] = 0u;

    const float3 mean3d = float3(p.means[idx]);
    const float4 w1 = p.view[0], w2 = p.view[1], w3 = p.view[2];
    const float depth = w3.x * mean3d.x + w3.y * mean3d.y + w3.z * mean3d.z + w3.w;
    if (depth < p.near_plane || depth > p.far_plane)
        return;

    const float raw_opacity = p.opacities[idx];
    const float opacity = 1.0f / (1.0f + exp(-raw_opacity));
    if (opacity < kFastMinAlpha)
        return;

    const float3 raw_scale = float3(p.scales[idx]);
    const float3 variance = exp(2.0f * fmin(raw_scale, kFastMaxRawScale));
    const float4 q = p.rotations[idx];
    if (dot(q, q) < 1e-8f)
        return;
    const FastRotation rot = fast_rotation(q);
    const FastCov3 cov3d = fast_cov3d(rot, variance);

    const float inv_depth = 1.0f / depth;
    const float x = (w1.x * mean3d.x + w1.y * mean3d.y + w1.z * mean3d.z + w1.w) * inv_depth;
    const float y = (w2.x * mean3d.x + w2.y * mean3d.y + w2.z * mean3d.z + w2.w) * inv_depth;
    const float tx = clamp(x, p.clip_left, p.clip_right);
    const float ty = clamp(y, p.clip_top, p.clip_bottom);
    const float j11 = p.fx * inv_depth;
    const float j13 = -j11 * tx;
    const float j22 = p.fy * inv_depth;
    const float j23 = -j22 * ty;
    const float3 jw1 = j11 * w1.xyz + j13 * w3.xyz;
    const float3 jw2 = j22 * w2.xyz + j23 * w3.xyz;
    const float3 jwc1 = fast_cov_mul(cov3d, jw1);
    const float3 jwc2 = fast_cov_mul(cov3d, jw2);
    float3 cov2d = float3(dot(jwc1, jw1), dot(jwc1, jw2), dot(jwc2, jw2));

    const bool mip = p.mip_filter != 0u;
    const float det_raw = mip ? fmax(cov2d.x * cov2d.z - cov2d.y * cov2d.y, 0.0f) : 0.0f;
    const float kernel_size = mip ? kFastDilationMip : kFastDilation;
    cov2d.x += kernel_size;
    cov2d.z += kernel_size;
    const float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det < kFastMinDeterminant)
        return;
    const float det_rcp = 1.0f / det;
    const float output_opacity = mip ? opacity * sqrt(det_raw * det_rcp) : opacity;
    if (output_opacity < kFastMinAlpha)
        return;

    const float3 conic = float3(cov2d.z * det_rcp, -cov2d.y * det_rcp, cov2d.x * det_rcp);
    const float2 mean2d = float2(x * p.fx + p.cx, y * p.fy + p.cy);

    const float power_threshold = log(output_opacity * kFastMinAlphaRcp);
    const float factor = sqrt(2.0f * power_threshold);
    const float extent_x = fmax(factor * sqrt(cov2d.x) - 0.5f, 0.0f);
    const float extent_y = fmax(factor * sqrt(cov2d.z) - 0.5f, 0.0f);
    const uint4 bounds = fast_screen_tile_bounds(mean2d, extent_x, extent_y, p.grid_w, p.grid_h);
    if ((bounds.y - bounds.x) * (bounds.w - bounds.z) == 0u)
        return;

    // A negative radius is no ellipse; zero is a point that still shades the
    // pixel whose center coincides with the mean.
    const FastTileWalk walk = fast_tile_walk(mean2d, conic, output_opacity, bounds);
    uint touched = 0u;
    if (!(walk.radius_sq < 0.0f)) {
        for (uint scan = walk.scan0; scan < walk.scan1; ++scan) {
            const uint2 span = fast_walk_span(walk, scan);
            touched += span.y >= span.x ? span.y - span.x : 0u;
        }
    }
    if (touched == 0u)
        return;

    const float3 camera = float3(p.camera[0], p.camera[1], p.camera[2]);
    if (p.max_screen_share != nullptr)
        atomic_max_float(p.max_screen_share + idx, gaussian_screen_share(mean3d, camera, raw_scale, raw_opacity));

    p.n_touched[idx] = touched;
    p.tile_info[idx] = FastTileInfo{ushort4(bounds), fast_depth_key(depth, p.depth_bits), 0u};
    const float extent_x_full = factor * sqrt(cov2d.x);
    const float extent_y_full = factor * sqrt(cov2d.z);
    const int px_min = max(0, fast_floor_int(mean2d.x - extent_x_full - 0.5f));
    const int py_min = max(0, fast_floor_int(mean2d.y - extent_y_full - 0.5f));
    const int px_max = max(px_min, fast_ceil_int(mean2d.x + extent_x_full - 0.5f) + 1);
    const int py_max = max(py_min, fast_ceil_int(mean2d.y + extent_y_full - 0.5f) + 1);
    p.mean_box[idx] = FastMeanBox{mean2d, ushort4(uint4(min(px_min, 65535), min(px_max, 65535),
                                                        min(py_min, 65535), min(py_max, 65535)))};
    p.conic_opacity[idx] = float4(conic, output_opacity);
    const float3 color = fast_sh_color(p.sh0, p.shN, p.sh_bounds, kFastShStorage, idx, mean3d, camera,
                                       kFastShLayoutRest, fast_q16_cells());
    p.color_depth[idx] = float4(color, depth);

    if (p.normals != nullptr) {
        // Rotation column of the smallest axis, facing the camera, in camera space.
        const float3 axis = (variance.x <= variance.y && variance.x <= variance.z)
                                ? float3(rot.r0.x, rot.r1.x, rot.r2.x)
                            : (variance.y <= variance.z) ? float3(rot.r0.y, rot.r1.y, rot.r2.y)
                                                         : float3(rot.r0.z, rot.r1.z, rot.r2.z);
        const float3 world = dot(axis, mean3d - camera) > 0.0f ? -axis : axis;
        p.normals[idx] = float4(dot(w1.xyz, world), dot(w2.xyz, world), dot(w3.xyz, world), 0.0f);
    }
}

// Exclusive scan of uint arrays: per-block sums, one threadgroup over the
// block sums, then the block-local scan. Totals are exact in 64 bits.
constant constexpr uint kFastScanThreads = 256u;
constant constexpr uint kFastScanBlock = 2048u;

struct FastScanParams {
    device const uint* input;
    device uint* output;
    device uint* block_sums;
    device uint* total;
    uint n;
    uint n_blocks;
};

kernel void fast_scan_reduce(constant FastScanParams& p [[buffer(0)]],
                             const uint group [[threadgroup_position_in_grid]],
                             const uint lane [[thread_index_in_threadgroup]],
                             const uint simd_lane [[thread_index_in_simdgroup]],
                             const uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partial[kFastScanThreads / 32u];
    const uint base = group * kFastScanBlock;
    uint sum = 0u;
    for (uint j = 0; j < kFastScanBlock / kFastScanThreads; ++j) {
        const uint i = base + j * kFastScanThreads + lane;
        if (i < p.n)
            sum += p.input[i];
    }
    sum = simd_sum(sum);
    if (simd_lane == 0u)
        partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        uint total = 0u;
        for (uint s = 0; s < kFastScanThreads / 32u; ++s)
            total += partial[s];
        p.block_sums[group] = total;
    }
}

// One threadgroup of 1024 threads.
kernel void fast_scan_top(constant FastScanParams& p [[buffer(0)]],
                          const uint lane [[thread_index_in_threadgroup]],
                          const uint simd_lane [[thread_index_in_simdgroup]],
                          const uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partial[32];
    threadgroup ulong carry_shared;
    ulong carry = 0ul;
    for (uint base = 0; base < p.n_blocks; base += 1024u) {
        const uint i = base + lane;
        const uint value = i < p.n_blocks ? p.block_sums[i] : 0u;
        const uint prefix = simd_prefix_exclusive_sum(value);
        if (simd_lane == 31u)
            partial[simd_group] = prefix + value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0u) {
            const uint group_sum = partial[simd_lane];
            const uint group_prefix = simd_prefix_exclusive_sum(group_sum);
            partial[simd_lane] = group_prefix;
            if (simd_lane == 31u)
                carry_shared = carry + ulong(group_prefix) + ulong(group_sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i < p.n_blocks)
            p.block_sums[i] = uint(carry) + partial[simd_group] + prefix;
        carry = carry_shared;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u && p.total != nullptr) {
        p.total[0] = uint(carry & 0xffffffffu);
        p.total[1] = uint(carry >> 32);
    }
}

kernel void fast_scan_down(constant FastScanParams& p [[buffer(0)]],
                           const uint group [[threadgroup_position_in_grid]],
                           const uint lane [[thread_index_in_threadgroup]],
                           const uint simd_lane [[thread_index_in_simdgroup]],
                           const uint simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint per_thread = kFastScanBlock / kFastScanThreads;
    threadgroup uint values[kFastScanBlock];
    threadgroup uint partial[kFastScanThreads / 32u];
    const uint base = group * kFastScanBlock;
    for (uint j = 0; j < per_thread; ++j) {
        const uint i = base + j * kFastScanThreads + lane;
        values[j * kFastScanThreads + lane] = i < p.n ? p.input[i] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint local[per_thread];
    uint sum = 0u;
    for (uint j = 0; j < per_thread; ++j) {
        local[j] = sum;
        sum += values[lane * per_thread + j];
    }
    const uint prefix = simd_prefix_exclusive_sum(sum);
    if (simd_lane == 31u)
        partial[simd_group] = prefix + sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset = p.block_sums[group] + prefix;
    for (uint s = 0; s < simd_group; ++s)
        offset += partial[s];
    for (uint j = 0; j < per_thread; ++j)
        values[lane * per_thread + j] = offset + local[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < per_thread; ++j) {
        const uint i = base + j * kFastScanThreads + lane;
        if (i < p.n)
            p.output[i] = values[j * kFastScanThreads + lane];
    }
}

struct FastInstanceParams {
    device const FastMeanBox* mean_box;
    device const float4* conic_opacity;
    device const FastTileInfo* tile_info;
    device const uint* n_touched;
    device const uint* offsets;
    device uint* keys;
    device uint* values;
    uint n, grid_w, depth_bits, unused;
};

// Port of create_instances_cu: one (tile << depth_bits | depth) key per
// touched tile, walked exactly as the preprocess count. A walk that falls
// short of its count pads with invalid values, which the blends skip.
kernel void fast_create_instances(constant FastInstanceParams& p [[buffer(0)]], const uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n)
        return;
    const uint count = p.n_touched[idx];
    if (count == 0u)
        return;
    const FastTileInfo info = p.tile_info[idx];
    const float4 co = p.conic_opacity[idx];
    const FastTileWalk walk = fast_tile_walk(p.mean_box[idx].mean, co.xyz, co.w, uint4(info.bounds));
    const uint begin = p.offsets[idx];
    const uint end = begin + count;
    uint write_at = begin;
    uint key = ((uint(info.bounds.z) * p.grid_w + uint(info.bounds.x)) << p.depth_bits) | info.depth_key;
    for (uint scan = walk.scan0; scan < walk.scan1 && write_at < end; ++scan) {
        const uint2 span = fast_walk_span(walk, scan);
        for (uint t = span.x; t < span.y && write_at < end; ++t) {
            const uint tile = walk.along_x ? scan * p.grid_w + t : t * p.grid_w + scan;
            key = (tile << p.depth_bits) | info.depth_key;
            p.keys[write_at] = key;
            p.values[write_at] = idx;
            ++write_at;
        }
    }
    for (; write_at < end; ++write_at) {
        p.keys[write_at] = key;
        p.values[write_at] = kFastInvalid;
    }
}

// Stable LSD radix sort of (key, value) pairs, 8 bits per pass. A block of
// 2048 keys counts its digits, the digit-major counts are scanned, then the
// block scatters with ballot-computed stable ranks.
constant constexpr uint kFastSortThreads = 256u;
constant constexpr uint kFastSortBlock = 2048u;

struct FastSortParams {
    device const uint* keys_in;
    device const uint* values_in;
    device uint* keys_out;
    device uint* values_out;
    device uint* histogram;
    uint n, n_blocks, shift, unused;
};

kernel void fast_sort_histogram(constant FastSortParams& p [[buffer(0)]],
                                const uint group [[threadgroup_position_in_grid]],
                                const uint lane [[thread_index_in_threadgroup]],
                                const uint simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint groups = kFastSortThreads / 32u;
    threadgroup atomic_uint counts[groups * 256u];
    for (uint i = lane; i < groups * 256u; i += kFastSortThreads)
        atomic_store_explicit(&counts[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint base = group * kFastSortBlock;
    for (uint j = 0; j < kFastSortBlock / kFastSortThreads; ++j) {
        const uint i = base + j * kFastSortThreads + lane;
        if (i < p.n)
            atomic_fetch_add_explicit(&counts[simd_group * 256u + ((p.keys_in[i] >> p.shift) & 255u)], 1u,
                                      memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = 0u;
    for (uint s = 0; s < groups; ++s)
        total += atomic_load_explicit(&counts[s * 256u + lane], memory_order_relaxed);
    p.histogram[lane * p.n_blocks + group] = total;
}

kernel void fast_sort_scatter(constant FastSortParams& p [[buffer(0)]],
                              const uint group [[threadgroup_position_in_grid]],
                              const uint lane [[thread_index_in_threadgroup]],
                              const uint simd_lane [[thread_index_in_simdgroup]],
                              const uint simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint groups = kFastSortThreads / 32u;
    threadgroup uint digit_base[256];
    threadgroup uint offsets[groups * 256u];
    digit_base[lane] = p.histogram[lane * p.n_blocks + group];
    const uint below = (1u << simd_lane) - 1u;
    const uint base = group * kFastSortBlock;
    for (uint chunk = base; chunk < min(base + kFastSortBlock, p.n); chunk += kFastSortThreads) {
        for (uint s = 0; s < groups; ++s)
            offsets[s * 256u + lane] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint i = chunk + lane;
        const bool valid = i < p.n;
        const uint key = valid ? p.keys_in[i] : 0u;
        const uint digit = (key >> p.shift) & 255u;
        uint peers = uint(static_cast<simd_vote::vote_t>(simd_ballot(valid)));
        for (uint b = 0; b < 8u; ++b) {
            const bool bit = ((digit >> b) & 1u) != 0u;
            const uint vote = uint(static_cast<simd_vote::vote_t>(simd_ballot(bit)));
            peers &= bit ? vote : ~vote;
        }
        const uint rank = popcount(peers & below);
        if (valid && rank == 0u)
            offsets[simd_group * 256u + digit] = popcount(peers);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint running = digit_base[lane];
        for (uint s = 0; s < groups; ++s) {
            const uint count = offsets[s * 256u + lane];
            offsets[s * 256u + lane] = running;
            running += count;
        }
        digit_base[lane] = running;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            const uint destination = offsets[simd_group * 256u + digit] + rank;
            p.keys_out[destination] = key;
            p.values_out[destination] = p.values_in[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct FastRangeParams {
    device const uint* keys;
    device uint* ranges;
    uint n_instances, n_tiles, depth_bits, unused;
};

// Port of extract_instance_ranges_cu. Range ends are written as separate
// words: a uint2 component store may rewrite its neighbour.
kernel void fast_tile_ranges(constant FastRangeParams& p [[buffer(0)]], const uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n_instances)
        return;
    const uint tile = p.keys[idx] >> p.depth_bits;
    if (tile >= p.n_tiles)
        return;
    if (idx == 0u) {
        p.ranges[2u * tile] = 0u;
    } else {
        const uint previous = p.keys[idx - 1u] >> p.depth_bits;
        if (previous >= p.n_tiles)
            return;
        if (tile != previous) {
            p.ranges[2u * previous + 1u] = idx;
            p.ranges[2u * tile] = idx;
        }
    }
    if (idx == p.n_instances - 1u)
        p.ranges[2u * tile + 1u] = p.n_instances;
}

struct FastFillParams {
    device uint* data;
    uint count;
    uint value;
};

kernel void fast_fill(constant FastFillParams& p [[buffer(0)]], const uint idx [[thread_position_in_grid]]) {
    if (idx < p.count)
        p.data[idx] = p.value;
}

// A 16x16 tile is eight 8x4 sub-tiles; SIMD group w owns sub-tiles w and w + 4,
// so every thread shades two pixels.
struct FastTilePixels {
    uint2 pix0, pix1;
    uint rank0, rank1;
    uint2 sub0, sub1;
    bool inside0, inside1;
};

static FastTilePixels fast_tile_pixels(const uint2 tile, const uint lane, const uint warp, const uint width,
                                       const uint height) {
    FastTilePixels t;
    const uint local_x = lane & 7u;
    const uint local_y = lane >> 3;
    const uint st1 = warp + 4u;
    const uint2 origin = tile * kFastTileSize;
    const uint2 o0 = uint2((warp & 1u) * 8u, (warp >> 1) * 4u);
    const uint2 o1 = uint2((st1 & 1u) * 8u, (st1 >> 1) * 4u);
    t.sub0 = origin + o0;
    t.sub1 = origin + o1;
    t.pix0 = t.sub0 + uint2(local_x, local_y);
    t.pix1 = t.sub1 + uint2(local_x, local_y);
    t.rank0 = (o0.y + local_y) * kFastTileSize + o0.x + local_x;
    t.rank1 = (o1.y + local_y) * kFastTileSize + o1.x + local_x;
    t.inside0 = t.pix0.x < width && t.pix0.y < height;
    t.inside1 = t.pix1.x < width && t.pix1.y < height;
    return t;
}

static bool fast_bbox_hits(const ushort4 bb, const uint2 sub) {
    return uint(bb.x) < sub.x + 8u && uint(bb.y) > sub.x && uint(bb.z) < sub.y + 4u && uint(bb.w) > sub.y;
}

// Port of ellipse_box_overlap_test / splat_overlaps_subtile_ellipse on an
// 8x4 sub-tile whose integer pixel origin is `sub`; `power` is
// log(opacity * kFastMinAlphaRcp), hoisted out of the per-sub-tile tests.
static bool fast_overlaps_subtile(const float2 mean, const float3 conic, const float opacity, const float power,
                                  const float2 sub) {
    if (!(opacity >= kFastMinAlpha))
        return false;
    const float x0 = (sub.x + 0.5f) - mean.x;
    const float x1 = (sub.x + 8.0f - 0.5f) - mean.x;
    const float y0 = (sub.y + 0.5f) - mean.y;
    const float y1 = (sub.y + 4.0f - 0.5f) - mean.y;
    const float mc = fmax(fmax(x0, -x1), fmax(y0, -y1));
    if (!(power > 0.0f))
        return power == 0.0f && mc <= 0.0f;
    const float inv_scale = 1.0f / (2.0f * power);
    const float a = conic.x * inv_scale, b0 = conic.y * inv_scale, c = conic.z * inv_scale;
    const float wx = -b0 / c;
    const float wy = -b0 / a;
    const float u0 = fmin(fmax(x0 * wx, y0), y1);
    const float u1 = fmin(fmax(x1 * wx, y0), y1);
    const float v0 = fmin(fmax(y0 * wy, x0), x1);
    const float v1 = fmin(fmax(y1 * wy, x0), x1);
    const float b = 2.0f * b0;
    const float mx = fmin(a * x0 * x0 + b * x0 * u0 + c * u0 * u0, a * x1 * x1 + b * x1 * u1 + c * u1 * u1);
    const float my = fmin(a * v0 * v0 + b * v0 * y0 + c * y0 * y0, a * v1 * v1 + b * v1 * y1 + c * y1 * y1);
    return fmin(mc, fmin(mx, my) - 1.0f) <= 0.0f;
}

static float3 fast_background(device const float* bg_color, device const float* bg_image, const uint pixel,
                              const uint n_pixels) {
    if (bg_image != nullptr)
        return float3(bg_image[pixel], bg_image[pixel + n_pixels], bg_image[pixel + 2u * n_pixels]);
    if (bg_color != nullptr)
        return float3(bg_color[0], bg_color[1], bg_color[2]);
    return float3(0.0f);
}

struct FastBlendParams {
    device const uint2* ranges;
    device const uint* values;
    device const FastMeanBox* mean_box;
    device const float4* conic_opacity;
    device const float4* color_depth;
    device const float4* normals;
    device float* image;
    device float* alpha;
    device float* depth;
    device float* normal;
    device uint* n_contrib;
    device float* final_transmittance;
    device const float* bg_color;
    device const float* bg_image;
    uint width, height, grid_w, unused;
};

struct FastPixelState {
    float3 color;
    float3 normal;
    float depth;
    float transmittance;
    uint possible;
    uint contributions;
    bool done;
};

static void fast_blend_pixel(thread FastPixelState& s, const float2 pixel, const float2 mean, const float4 conic_opacity,
                             const float4 color_log_alpha, const float depth, const float3 normal) {
    const float2 d = mean - pixel;
    const float sigma_over_2 = (conic_opacity.x * d.x * d.x + conic_opacity.z * d.y * d.y) + conic_opacity.y * d.x * d.y;
    // Skip only when even a slightly high exp would stay below 1/255; the
    // survivors take the exact alpha test.
    if (sigma_over_2 < 0.0f || sigma_over_2 > color_log_alpha.w + 1.0e-5f)
        return;
    const float alpha = fmin(conic_opacity.w * exp(-sigma_over_2), kFastMaxFragmentAlpha);
    if (alpha < kFastMinAlpha)
        return;
    const float weight = s.transmittance * alpha;
    s.color += weight * color_log_alpha.xyz;
    s.depth += weight * depth;
    if (kFastRenderNormal != 0u)
        s.normal += weight * normal;
    s.transmittance *= 1.0f - alpha;
    s.contributions = s.possible;
    if (s.transmittance < kFastTransmittanceThreshold)
        s.done = true;
}

static void fast_store_pixel(constant FastBlendParams& p, const FastPixelState s, const uint2 pix) {
    const uint n_pixels = p.width * p.height;
    const uint pixel = p.width * pix.y + pix.x;
    const float3 bg = fast_background(p.bg_color, p.bg_image, pixel, n_pixels);
    p.image[pixel] = s.color.x + s.transmittance * bg.x;
    p.image[pixel + n_pixels] = s.color.y + s.transmittance * bg.y;
    p.image[pixel + 2u * n_pixels] = s.color.z + s.transmittance * bg.z;
    p.alpha[pixel] = 1.0f - s.transmittance;
    if (kFastRenderDepth != 0u)
        p.depth[pixel] = s.depth;
    if (kFastRenderNormal != 0u) {
        p.normal[pixel] = s.normal.x;
        p.normal[pixel + n_pixels] = s.normal.y;
        p.normal[pixel + 2u * n_pixels] = s.normal.z;
    }
    p.n_contrib[pixel] = s.contributions;
}

// Port of blend_cu: 128 threads per tile, front to back, with the exact
// ellipse sub-tile cull.
kernel void fast_blend_forward(constant FastBlendParams& p [[buffer(0)]],
                               const uint2 tile [[threadgroup_position_in_grid]],
                               const uint rank [[thread_index_in_threadgroup]],
                               const uint lane [[thread_index_in_simdgroup]],
                               const uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 s_mean[kFastBlendThreads];
    threadgroup ushort4 s_bbox[kFastBlendThreads];
    threadgroup float4 s_conic[kFastBlendThreads];
    threadgroup float4 s_color[kFastBlendThreads];
    threadgroup float s_depth[kFastBlendThreads];
    threadgroup float3 s_normal[kFastBlendThreads];
    threadgroup uint s_done[kFastBlendThreads / 32u];

    const FastTilePixels t = fast_tile_pixels(tile, lane, warp, p.width, p.height);
    const float2 pixel0 = float2(t.pix0) + 0.5f;
    const float2 pixel1 = float2(t.pix1) + 0.5f;
    const uint tile_idx = tile.y * p.grid_w + tile.x;
    const uint2 range = p.ranges[tile_idx];
    const int total = int(range.y - range.x);

    FastPixelState s0 = {float3(0.0f), float3(0.0f), 0.0f, 1.0f, 0u, 0u, !t.inside0};
    FastPixelState s1 = {float3(0.0f), float3(0.0f), 0.0f, 1.0f, 0u, 0u, !t.inside1};

    for (int remaining = total, batch = 0; remaining > 0; remaining -= int(kFastBlendThreads), batch += int(kFastBlendThreads)) {
        const bool all_done = simd_all(s0.done && s1.done);
        if (lane == 0u)
            s_done[warp] = all_done ? 1u : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s_done[0] + s_done[1] + s_done[2] + s_done[3] == kFastBlendThreads / 32u)
            break;
        const int fetch = int(range.x) + batch + int(rank);
        if (fetch < int(range.y)) {
            const uint prim = p.values[fetch];
            if (prim != kFastInvalid) {
                const FastMeanBox geom = p.mean_box[prim];
                float4 co = p.conic_opacity[prim];
                const float4 cd = p.color_depth[prim];
                // The staged conic folds in the 0.5 of the exponent.
                co.x *= 0.5f;
                co.z *= 0.5f;
                s_mean[rank] = geom.mean;
                s_bbox[rank] = geom.bbox;
                s_conic[rank] = co;
                s_color[rank] = float4(fmin(fmax(cd.xyz, 0.0f), kFastMaxBlendColor), log(co.w * kFastMinAlphaRcp));
                s_depth[rank] = cd.w;
                if (kFastRenderNormal != 0u)
                    s_normal[rank] = p.normals[prim].xyz;
            } else {
                s_bbox[rank] = ushort4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int batch_size = min(int(kFastBlendThreads), remaining);
        for (int j_base = 0; j_base < batch_size; j_base += 32) {
            const int j_test = j_base + int(lane);
            bool hit0 = false, hit1 = false;
            if (j_test < batch_size) {
                // The exact ellipse test of the backward keeps splats whose box
                // but not ellipse reaches a sub-tile out of the serial walk.
                const ushort4 bb = s_bbox[j_test];
                const float2 mean = s_mean[j_test];
                const float4 co = s_conic[j_test];
                const float3 conic = float3(2.0f * co.x, co.y, 2.0f * co.z);
                const float power = s_color[j_test].w;
                hit0 = fast_bbox_hits(bb, t.sub0) && fast_overlaps_subtile(mean, conic, co.w, power, float2(t.sub0));
                hit1 = fast_bbox_hits(bb, t.sub1) && fast_overlaps_subtile(mean, conic, co.w, power, float2(t.sub1));
            }
            const uint mask0 = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit0)));
            const uint mask1 = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit1)));
            // Walk only splats that touch a sub-tile of this SIMD group. A pixel's
            // done flag changes only when it blends, so every skipped splat counts
            // toward `possible` exactly as a one-by-one walk would.
            const uint chunk = uint(min(32, batch_size - j_base));
            uint live = mask0 | mask1;
            uint walked = 0u;
            while (live != 0u) {
                const uint k = ctz(live);
                live &= live - 1u;
                const int j = j_base + int(k);
                const bool walk0 = !s0.done;
                const bool walk1 = !s1.done;
                s0.possible += walk0 ? k + 1u - walked : 0u;
                s1.possible += walk1 ? k + 1u - walked : 0u;
                walked = k + 1u;
                const bool use0 = walk0 && ((mask0 >> k) & 1u) != 0u;
                const bool use1 = walk1 && ((mask1 >> k) & 1u) != 0u;
                if (!use0 && !use1)
                    continue;
                const float2 mean = s_mean[j];
                const float4 co = s_conic[j];
                const float4 c = s_color[j];
                const float depth = s_depth[j];
                const float3 normal = kFastRenderNormal != 0u ? s_normal[j] : float3(0.0f);
                if (use0)
                    fast_blend_pixel(s0, pixel0, mean, co, c, depth, normal);
                if (use1)
                    fast_blend_pixel(s1, pixel1, mean, co, c, depth, normal);
            }
            s0.possible += !s0.done ? chunk - walked : 0u;
            s1.possible += !s1.done ? chunk - walked : 0u;
        }
    }

    if (t.inside0)
        fast_store_pixel(p, s0, t.pix0);
    if (t.inside1)
        fast_store_pixel(p, s1, t.pix1);
    p.final_transmittance[tile_idx * kFastTilePixels + t.rank0] = t.inside0 ? s0.transmittance : 1.0f;
    p.final_transmittance[tile_idx * kFastTilePixels + t.rank1] = t.inside1 ? s1.transmittance : 1.0f;
}

// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// LpipsOps kernels of lpips_kernels.cu: the weight taps and the fused pool reduce. The convolutions run on the
// nn convolution (lpips_metal.cpp). Products of fp16 operands accumulate in fp32.

struct LpipsTapsParams {
    device const half* weight;
    device half* taps;
    uint cout, cin;
};

// taps[tap][out][in] = weight[out][in][tap].
kernel void lpips_weight_taps(constant LpipsTapsParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index >= 9 * p.cout * p.cin)
        return;
    const uint ic = index % p.cin;
    const uint n = (index / p.cin) % p.cout;
    const uint tap = index / (p.cin * p.cout);
    p.taps[index] = p.weight[(n * p.cin + ic) * 9 + tap];
}

struct LpipsPoolParams {
    device const half* x;
    device const half* y;
    device const half* lin;
    device float* score;
    device half* pooled_x; // both or neither
    device half* pooled_y;
    int n, channels, h, w;
    int y0, y1, x0, x1;
    float inverse_count;
    device const float* weights;
    int weights_width, weights_y0, weights_x0;
};

static float lpips_tap(constant LpipsPoolParams& p, device const half* data, const int base, const int row,
                       const int col) {
    return row < p.h && col < p.w ? float(data[base + row * p.w + col]) : 0.0f;
}

// One thread per column of a row pair: the pair's channel norms, the
// lin-weighted normalized squared distance over the interior, and for even
// columns the 2x2 max pool of both feature maps.
kernel void lpips_pool_reduce(constant LpipsPoolParams& p [[buffer(0)]], uint index [[thread_position_in_grid]],
                              uint local [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float partials[32];
    const int pairs = (p.h + 1) / 2;
    const int col = int(index) % p.w;
    const int r = (int(index) / p.w) % pairs;
    const int ni = int(index) / (p.w * pairs);
    const int row0 = 2 * r;
    const int plane = p.h * p.w;
    float score = 0.0f;
    if (ni < p.n) {
        float sx0 = 0.0f, sx1 = 0.0f, sy0 = 0.0f, sy1 = 0.0f;
        for (int c = 0; c < p.channels; ++c) {
            const int base = (ni * p.channels + c) * plane;
            const float a0 = lpips_tap(p, p.x, base, row0, col), a1 = lpips_tap(p, p.x, base, row0 + 1, col);
            const float b0 = lpips_tap(p, p.y, base, row0, col), b1 = lpips_tap(p, p.y, base, row0 + 1, col);
            sx0 += a0 * a0;
            sx1 += a1 * a1;
            sy0 += b0 * b0;
            sy1 += b1 * b1;
        }
        const float ix0 = rsqrt(sx0 + 1.0e-10f), ix1 = rsqrt(sx1 + 1.0e-10f);
        const float iy0 = rsqrt(sy0 + 1.0e-10f), iy1 = rsqrt(sy1 + 1.0e-10f);
        const bool in_x = col >= p.x0 && col < p.x1;
        const bool score0 = in_x && row0 >= p.y0 && row0 < p.y1;
        const bool score1 = in_x && row0 + 1 >= p.y0 && row0 + 1 < p.y1;
        const bool pool = p.pooled_x && col % 2 == 0 && col + 1 < p.w && row0 + 1 < p.h;
        const int out_h = p.h / 2, out_w = p.w / 2;
        for (int c = 0; c < p.channels; ++c) {
            const int base = (ni * p.channels + c) * plane;
            const float a0 = lpips_tap(p, p.x, base, row0, col), a1 = lpips_tap(p, p.x, base, row0 + 1, col);
            const float b0 = lpips_tap(p, p.y, base, row0, col), b1 = lpips_tap(p, p.y, base, row0 + 1, col);
            const float wv = float(p.lin[c]);
            const float d0 = a0 * ix0 - b0 * iy0;
            const float d1 = a1 * ix1 - b1 * iy1;
            if (score0)
                score += wv * d0 * d0 * (p.weights ? p.weights[(row0 + p.weights_y0) * p.weights_width + col + p.weights_x0] : 1.0f);
            if (score1)
                score += wv * d1 * d1 * (p.weights ? p.weights[(row0 + 1 + p.weights_y0) * p.weights_width + col + p.weights_x0] : 1.0f);
            if (pool) {
                const int o = ((ni * p.channels + c) * out_h + r) * out_w + col / 2;
                p.pooled_x[o] = half(fmax(fmax(a0, a1), fmax(lpips_tap(p, p.x, base, row0, col + 1),
                                                             lpips_tap(p, p.x, base, row0 + 1, col + 1))));
                p.pooled_y[o] = half(fmax(fmax(b0, b1), fmax(lpips_tap(p, p.y, base, row0, col + 1),
                                                             lpips_tap(p, p.y, base, row0 + 1, col + 1))));
            }
        }
    }
    const float total = ppisp_group_sum(score, partials, lane, simd, simds);
    if (local == 0 && total != 0.0f)
        ppisp_atomic_add(p.score, total * p.inverse_count);
}

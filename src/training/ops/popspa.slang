// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// POPSpa per-row kernels for devices without FP64. Scores are unevaluated float
// pairs hi + lo; the effective-rank terms are rearranged so that float keeps the
// near-rank-one derivative the CUDA kernels get from double intermediates.
struct Parameters {
    Ptr<float2, Access.ReadWrite> scores;
    Ptr<float, Access.Read> a, b, c;
    Ptr<uint8_t, Access.Read> active, frozen, keep;
    Ptr<float, Access.ReadWrite> out0, out1, out2;
    uint count;
    uint flag;
    float rho, inv_count, rank_weight, thin_weight, epsilon, unused;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

bool row_active(uint i) {
    return (uint64_t)p.active == 0 || p.active[i] != 0;
}
bool row_frozen(uint i) {
    return (uint64_t)p.frozen != 0 && p.frozen[i] != 0;
}
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}
// Accurate log(1 + x) and exp(x) - 1 near zero (Goldberg, Kahan); they need the
// precise floating-point mode this program is compiled with.
float log_one_plus(float x) {
    float u = 1.0f + x;
    return u == 1.0f ? x : log(u) * (x / (u - 1.0f));
}
float exp_minus_one(float x) {
    float u = exp(x);
    if (u == 1.0f)
        return x;
    float m = u - 1.0f;
    return m == -1.0f ? -1.0f : m * (x / log(u));
}

// scores += a (one view), compensated: hi stays the float nearest the sum.
[shader("compute")][numthreads(256, 1, 1)]
void fold(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count)
        return;
    float2 s = p.scores[i];
    float x = p.a[i];
    float sum = s.x + x;
    float bp = sum - s.x;
    float lo = s.y + ((s.x - (sum - bp)) + (x - bp));
    float hi = sum + lo;
    p.scores[i] = float2(hi, lo - (hi - sum));
}

// Binary64 bits of hi + lo, exactly: a float pair spans at most 49 mantissa bits.
// Rows are nonnegative; lo bits more than 32 below hi's last bit are dropped.
uint2 float64_bits(float2 pair) {
    // Bit tests: devices may flush subnormal operands of float compares.
    uint hb = asuint(pair.x), lb = asuint(pair.y);
    if ((hb & 0x7fffffff) == 0)
        return uint2(0, 0);
    int he = int((hb >> 23) & 0xff), le = int((lb >> 23) & 0xff);
    int64_t hm = int64_t(he == 0 ? (hb & 0x7fffff) : ((hb & 0x7fffff) | 0x800000));
    int64_t lm = int64_t(le == 0 ? (lb & 0x7fffff) : ((lb & 0x7fffff) | 0x800000));
    he = he == 0 ? -126 : he - 127;
    le = le == 0 ? -126 : le - 127;
    if ((lb >> 31) != 0)
        lm = -lm;
    // Scale 2^(he - 55): hi keeps 32 guard bits below its last mantissa bit.
    int64_t sum = hm << 32;
    int shift = le - he + 32;
    if ((lb & 0x7fffffff) != 0 && shift > -64)
        sum += shift >= 0 ? lm << shift : lm >> (-shift);
    uint64_t u = uint64_t(sum);
    uint high = uint(u >> 32);
    int msb = high != 0 ? 32 + int(firstbithigh(high)) : int(firstbithigh(uint(u)));
    uint64_t mantissa = msb >= 52 ? u >> uint(msb - 52) : u << uint(52 - msb);
    uint64_t bits = (uint64_t(he - 55 + msb + 1023) << 52) | (mantissa & 0xfffffffffffffull);
    return uint2(uint(bits), uint(bits >> 32));
}

// out2 (as uint2 rows) = binary64 bits of each pair.
[shader("compute")][numthreads(256, 1, 1)]
void export_float64(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count)
        return;
    ((Ptr<uint2, Access.ReadWrite>)p.out2)[i] = float64_bits(p.scores[i]);
}

// out0 = hi, out1 = lo.
[shader("compute")][numthreads(256, 1, 1)]
void split(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count)
        return;
    float2 s = p.scores[i];
    p.out0[i] = s.x;
    p.out1[i] = s.y;
}

// a = raw opacity, b = u; out0 = sigmoid(raw) + u.
[shader("compute")][numthreads(256, 1, 1)]
void shifted(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count)
        return;
    p.out0[i] = sigmoid(p.a[i]) + p.b[i];
}

// a = raw opacity, b = shifted; out0 = z, out1 = u. flag: update the dual.
[shader("compute")][numthreads(256, 1, 1)]
void proxy(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count)
        return;
    float z = p.keep[i] != 0 ? p.b[i] : 0.0f;
    p.out0[i] = z;
    if (p.flag != 0 && row_active(i) && !row_frozen(i))
        p.out1[i] = p.out1[i] + (sigmoid(p.a[i]) - z);
}

// a = raw opacity, b = z, c = u; out0 = opacity gradient, out1 = per-row loss.
[shader("compute")][numthreads(256, 1, 1)]
void penalty(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count)
        return;
    float loss = 0.0f;
    if (row_active(i) && !row_frozen(i)) {
        float a = sigmoid(p.a[i]);
        float d = a - p.b[i] + p.c[i];
        p.out0[i] = p.out0[i] + p.rho * d * a * (1.0f - a);
        loss = 0.5f * p.rho * d * d;
    }
    p.out1[i] = loss;
}

// a = log scales [N,3]; out0 = gradient [N,3], out1 = per-row loss.
// Variance shares p_d = s_d^2 / max(sum s^2, eps) with log shares clamped at log(eps);
// entropy H, needle max(-log(expm1(H) + eps), 0), plus the smallest scale.
[shader("compute")][numthreads(256, 1, 1)]
void effective_rank(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count)
        return;
    if (!row_active(i)) {
        p.out1[i] = 0.0f;
        return;
    }
    float eps = p.epsilon;
    float log_eps = log(eps);
    float raw[3] = { p.a[3 * i], p.a[3 * i + 1], p.a[3 * i + 2] };
    uint minimum = 0;
    uint peak = 0;
    [unroll] for (uint d = 1; d < 3; ++d) {
        if (raw[d] < raw[minimum])
            minimum = d;
        if (raw[d] > raw[peak])
            peak = d;
    }
    // q_d = s_d^2 / s_peak^2, so the variance sum is s_peak^2 * (1 + rest).
    float q[3];
    float rest = 0.0f;
    [unroll] for (uint d = 0; d < 3; ++d) {
        q[d] = exp(2.0f * (raw[d] - raw[peak]));
        if (d != peak)
            rest += q[d];
    }
    float log_mass = log_one_plus(rest);
    bool normalized = 2.0f * raw[peak] + log_mass >= log_eps;
    float share[3], log_share[3], clamped[3];
    [unroll] for (uint d = 0; d < 3; ++d) {
        float exact = normalized ? 2.0f * (raw[d] - raw[peak]) - log_mass : 2.0f * raw[d] - log_eps;
        log_share[d] = exact;
        share[d] = exp(exact);
        clamped[d] = max(exact, log_eps);
    }
    float entropy = 0.0f;
    [unroll] for (uint d = 0; d < 3; ++d)
        entropy -= share[d] * clamped[d];
    float argument = exp_minus_one(entropy) + eps;
    float needle = max(-log(argument), 0.0f);
    float scale_min = exp(raw[minimum]);
    p.out1[i] = (p.rank_weight * needle + p.thin_weight * scale_min) * p.inv_count;
    if (row_frozen(i))
        return;
    float rank_factor = argument <= 1.0f ? -p.rank_weight * exp(entropy) / argument : 0.0f;
    // dH/dlog s_d = 2 p_d (dh_d - sum_j p_j dh_j) with dh = -log p - [p >= eps]. In the
    // normalized case the shares sum to one, so the bracket is sum_j p_j (dh_d - dh_j).
    [unroll] for (uint d = 0; d < 3; ++d) {
        float indicator_d = share[d] >= eps ? 1.0f : 0.0f;
        float bracket;
        if (normalized) {
            bracket = 0.0f;
            [unroll] for (uint j = 0; j < 3; ++j) {
                if (j == d)
                    continue;
                float indicator_j = share[j] >= eps ? 1.0f : 0.0f;
                float log_gap = (log_share[d] >= log_eps && log_share[j] >= log_eps) ? 2.0f * (raw[d] - raw[j]) : clamped[d] - clamped[j];
                bracket += share[j] * (-log_gap - (indicator_d - indicator_j));
            }
        } else {
            bracket = -clamped[d] - indicator_d;
        }
        float thin = d == minimum ? p.thin_weight * scale_min : 0.0f;
        p.out0[3 * i + d] = p.out0[3 * i + d] + (rank_factor * 2.0f * share[d] * bracket + thin) * p.inv_count;
    }
}

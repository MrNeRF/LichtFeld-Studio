[SpecializationConstant] const uint kOperation = 0;

struct Parameters {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
    uint64_t e;
    uint64_t f;
    uint64_t g;
    uint64_t h;
    uint64_t seed;
    uint count;
    uint count2;
    uint mode;
    uint reserved;
    float first;
    float second;
};

uint4 philox_round(uint4 counter, uint2 key) {
    const uint64_t p0 = uint64_t(0xD2511F53u) * counter.x;
    const uint64_t p1 = uint64_t(0xCD9E8D57u) * counter.z;
    return uint4(uint(p1 >> 32) ^ counter.y ^ key.x, uint(p1),
                 uint(p0 >> 32) ^ counter.w ^ key.y, uint(p0));
}

uint4 philox(uint4 counter, uint2 key) {
    for (uint round = 0; round < 10; ++round) {
        counter = philox_round(counter, key);
        key += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return counter;
}

uint4 random_words(uint64_t seed, uint row) {
    return philox(uint4(0u, 0u, row, 0u), uint2(uint(seed), uint(seed >> 32)));
}

float cuda_uniform(uint word) {
    return float(word) * 2.3283064e-10f + (2.3283064e-10f / 2.0f);
}

float cuda_pow(float value, float exponent) {
    return exp2(log2(value) * exponent);
}

float cuda_exp(float value) {
    return exp2(value * 1.4426950408889634f);
}

float4 cuda_normal4(uint4 words) {
    const float u0 = float(words.x) * 2.3283064e-10f + (2.3283064e-10f / 2.0f);
    const float v0 = float(words.y) * (2.3283064e-10f * 6.2831855f) +
                     ((2.3283064e-10f * 6.2831855f) / 2.0f);
    const float u1 = float(words.z) * 2.3283064e-10f + (2.3283064e-10f / 2.0f);
    const float v1 = float(words.w) * (2.3283064e-10f * 6.2831855f) +
                     ((2.3283064e-10f * 6.2831855f) / 2.0f);
    const float radius0 = sqrt(-2.0f * log(u0));
    const float radius1 = sqrt(-2.0f * log(u1));
    float sin0, cos0, sin1, cos1;
    sincos(v0, sin0, cos0);
    sincos(v1, sin1, cos1);
    return float4(radius0 * sin0, radius0 * cos0, radius1 * sin1, radius1 * cos1);
}

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform Parameters p) {
    const uint i = id.x;
    if (kOperation == 0u) {
        if (i >= p.count) return;
        Ptr<float, Access.Read> opacity = (Ptr<float, Access.Read>)p.a;
        Ptr<float, Access.Read> scales = (Ptr<float, Access.Read>)p.b;
        Ptr<int, Access.Read> ratios = (Ptr<int, Access.Read>)p.c;
        Ptr<float, Access.ReadWrite> new_opacity = (Ptr<float, Access.ReadWrite>)p.d;
        Ptr<float, Access.ReadWrite> new_scales = (Ptr<float, Access.ReadWrite>)p.e;
        Ptr<float, Access.Read> coeff = (Ptr<float, Access.Read>)p.f;
        const int n = ratios[i];
        const float o = clamp(opacity[i], 1e-6f, 1.0f - 1e-6f);
        const float o_new = clamp(1.0f - cuda_pow(1.0f - o, 1.0f / float(n)), max(1e-6f, p.first), 1.0f - 1e-6f);
        new_opacity[i] = o_new;
        float denominator = 0.0f;
        for (int row = 1; row <= n; ++row)
            for (int k = 0; k < row; ++k)
                denominator = fma(coeff[(row - 1) * 51 + k], cuda_pow(o_new, float(k + 1)), denominator);
        float safe_denominator = max(abs(denominator), 1e-8f);
        if (denominator < 0.0f) safe_denominator = -safe_denominator;
        const float factor = clamp(o / safe_denominator, -1e6f, 1e6f);
        for (uint axis = 0; axis < 3; ++axis)
            new_scales[i * 3u + axis] = max(abs(factor * scales[i * 3u + axis]), 1e-10f);
        return;
    }
    if (kOperation == 1u) {
        if (i >= p.count) return;
        Ptr<float, Access.Read> opacity = (Ptr<float, Access.Read>)p.a;
        Ptr<float, Access.Read> scales = (Ptr<float, Access.Read>)p.b;
        Ptr<float, Access.Read> quats = (Ptr<float, Access.Read>)p.c;
        Ptr<uint8_t, Access.Read> frozen = (Ptr<uint8_t, Access.Read>)p.d;
        Ptr<float, Access.ReadWrite> means = (Ptr<float, Access.ReadWrite>)p.e;
        if (i < p.count2 && frozen[i] != 0u) return;
        const float w = quats[i * 4u], x = quats[i * 4u + 1u];
        const float y = quats[i * 4u + 2u], z = quats[i * 4u + 3u];
        const float inverse_norm = min(rsqrt(x*x + y*y + z*z + w*w), 1e12f);
        const float nw = w * inverse_norm, nx = x * inverse_norm;
        const float ny = y * inverse_norm, nz = z * inverse_norm;
        const float r00 = 1.0f - 2.0f * (ny*ny + nz*nz);
        const float r01 = 2.0f * (nx*ny - nw*nz);
        const float r02 = 2.0f * (nx*nz + nw*ny);
        const float r10 = 2.0f * (nx*ny + nw*nz);
        const float r11 = 1.0f - 2.0f * (nx*nx + nz*nz);
        const float r12 = 2.0f * (ny*nz - nw*nx);
        const float r20 = 2.0f * (nx*nz - nw*ny);
        const float r21 = 2.0f * (ny*nz + nw*nx);
        const float r22 = 1.0f - 2.0f * (nx*nx + ny*ny);
        const float sx = cuda_exp(2.0f * scales[i * 3u]);
        const float sy = cuda_exp(2.0f * scales[i * 3u + 1u]);
        const float sz = cuda_exp(2.0f * scales[i * 3u + 2u]);
        const float c00 = (r00 * sx) * r00 + (r01 * sy) * r01 + (r02 * sz) * r02;
        const float c01 = (r00 * sx) * r10 + (r01 * sy) * r11 + (r02 * sz) * r12;
        const float c02 = (r00 * sx) * r20 + (r01 * sy) * r21 + (r02 * sz) * r22;
        const float c10 = (r10 * sx) * r00 + (r11 * sy) * r01 + (r12 * sz) * r02;
        const float c11 = (r10 * sx) * r10 + (r11 * sy) * r11 + (r12 * sz) * r12;
        const float c12 = (r10 * sx) * r20 + (r11 * sy) * r21 + (r12 * sz) * r22;
        const float c20 = (r20 * sx) * r00 + (r21 * sy) * r01 + (r22 * sz) * r02;
        const float c21 = (r20 * sx) * r10 + (r21 * sy) * r11 + (r22 * sz) * r12;
        const float c22 = (r20 * sx) * r20 + (r21 * sy) * r21 + (r22 * sz) * r22;
        const float4 noise = cuda_normal4(random_words(p.seed, i));
        const float sigmoid = 1.0f / (1.0f + cuda_exp(-opacity[i]));
        const float opacity_scale = 1.0f / (1.0f + cuda_exp(fma(100.0f, sigmoid, -0.5f)));
        const float factor = p.first * opacity_scale;
        means[i * 3u] += factor * (c00 * noise.x + c01 * noise.y + c02 * noise.z);
        means[i * 3u + 1u] += factor * (c10 * noise.x + c11 * noise.y + c12 * noise.z);
        means[i * 3u + 2u] += factor * (c20 * noise.x + c21 * noise.y + c22 * noise.z);
        return;
    }
    if (kOperation == 2u) {
        if (i >= p.count) return;
        Ptr<int64_t, Access.Read> indices = (Ptr<int64_t, Access.Read>)p.a;
        Ptr<float, Access.Read> new_scales = (Ptr<float, Access.Read>)p.b;
        Ptr<float, Access.Read> new_opacity = (Ptr<float, Access.Read>)p.c;
        Ptr<float, Access.ReadWrite> scales = (Ptr<float, Access.ReadWrite>)p.d;
        Ptr<float, Access.ReadWrite> opacity = (Ptr<float, Access.ReadWrite>)p.e;
        const int64_t target = indices[i];
        if (target < 0 || target >= int64_t(p.count2)) return;
        for (uint axis = 0; axis < 3; ++axis) scales[uint(target) * 3u + axis] = new_scales[i * 3u + axis];
        opacity[target] = new_opacity[i];
        return;
    }
    if (kOperation == 3u) {
        if (i >= p.count) return;
        Ptr<int64_t, Access.Read> source_indices = (Ptr<int64_t, Access.Read>)p.a;
        Ptr<int64_t, Access.Read> destination_indices = (Ptr<int64_t, Access.Read>)p.b;
        Ptr<float, Access.ReadWrite> means = (Ptr<float, Access.ReadWrite>)p.c;
        Ptr<float, Access.ReadWrite> sh0 = (Ptr<float, Access.ReadWrite>)p.d;
        Ptr<float, Access.ReadWrite> scales = (Ptr<float, Access.ReadWrite>)p.e;
        Ptr<float, Access.ReadWrite> quats = (Ptr<float, Access.ReadWrite>)p.f;
        Ptr<float, Access.ReadWrite> opacity = (Ptr<float, Access.ReadWrite>)p.g;
        const int64_t source = source_indices[i], destination = destination_indices[i];
        if (source < 0 || source >= int64_t(p.count2) || destination < 0 || destination >= int64_t(p.count2)) return;
        for (uint axis = 0; axis < 3; ++axis) {
            means[uint(destination) * 3u + axis] = means[uint(source) * 3u + axis];
            sh0[uint(destination) * 3u + axis] = sh0[uint(source) * 3u + axis];
            scales[uint(destination) * 3u + axis] = scales[uint(source) * 3u + axis];
        }
        for (uint axis = 0; axis < 4; ++axis) quats[uint(destination) * 4u + axis] = quats[uint(source) * 4u + axis];
        opacity[destination] = opacity[source];
        return;
    }
    if (kOperation == 4u) {
        if (i >= p.count) return;
        Ptr<float, Access.Read> cdf = (Ptr<float, Access.Read>)p.a;
        Ptr<int64_t, Access.Read> alive = (Ptr<int64_t, Access.Read>)p.b;
        Ptr<float, Access.Read> opacity = (Ptr<float, Access.Read>)p.c;
        Ptr<float, Access.Read> scales = (Ptr<float, Access.Read>)p.d;
        Ptr<int64_t, Access.ReadWrite> indices = (Ptr<int64_t, Access.ReadWrite>)p.e;
        Ptr<float, Access.ReadWrite> output_opacity = (Ptr<float, Access.ReadWrite>)p.f;
        Ptr<float, Access.ReadWrite> output_scales = (Ptr<float, Access.ReadWrite>)p.g;
        const uint domain_count = p.count2;
        const float total = cdf[domain_count - 1u];
        if (total <= 0.0f) {
            indices[i] = 0;
            output_opacity[i] = 0.0f;
            output_scales[i * 3u] = 0.0f;
            output_scales[i * 3u + 1u] = 0.0f;
            output_scales[i * 3u + 2u] = 0.0f;
            return;
        }
        // Keep small positive draws normal even when the device flushes subnormals.
        const float probability_scale = total < 1.0f ? 18446744073709551616.0f : 1.0f; // 2^64
        const float u = cuda_uniform(random_words(p.seed, i).x) * (total * probability_scale);
        uint left = 0u, right = domain_count - 1u, selected = domain_count - 1u;
        while (left <= right) {
            const uint middle = left + (right - left) / 2u;
            if (cdf[middle] * probability_scale >= u) { selected = middle; if (middle == 0u) break; right = middle - 1u; }
            else left = middle + 1u;
        }
        const int64_t global_index = p.mode != 0u ? alive[selected] : int64_t(selected);
        indices[i] = global_index;
        output_opacity[i] = opacity[global_index];
        for (uint axis = 0; axis < 3; ++axis) output_scales[i * 3u + axis] = cuda_exp(scales[uint(global_index) * 3u + axis]);
        return;
    }
    if (kOperation == 5u) {
        if (i >= p.count) return;
        Ptr<float, Access.ReadWrite> error = (Ptr<float, Access.ReadWrite>)p.a;
        Ptr<float, Access.ReadWrite> densification = (Ptr<float, Access.ReadWrite>)p.b;
        error[i] = max(error[i], densification[p.count + i]);
        densification[i] = 0.0f;
        densification[p.count + i] = 0.0f;
        return;
    }
    if (kOperation == 6u) {
        if (i >= p.count) return;
        Ptr<float, Access.Read> weights = (Ptr<float, Access.Read>)p.a;
        Ptr<int64_t, Access.Read> alive = (Ptr<int64_t, Access.Read>)p.b;
        Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)p.c;
        output[i] = weights[alive[i]];
    }
}

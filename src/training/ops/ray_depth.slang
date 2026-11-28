// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct Parameters {
    Ptr<float, Access.ReadWrite> state, means, scales;
    Ptr<float, Access.Read> rotations, gradient, camera;
    Ptr<float, Access.ReadWrite> proposals;
    Ptr<uint8_t, Access.Read> frozen, crop, far;
    uint count, frozenCount, cropCount, farCount;
    float lr, meanLr, farLimit, frozenScale, cropScale, medianExtent;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
static const float unobserved = 1e30;
float3 viewPosition() { return float3(p.camera[0], p.camera[1], p.camera[2]); }
// A Gaussian's row at its 64-bit byte address, not as an index from the buffer's pointer: NVIDIA's Vulkan computes an
// access chain's byte offset in 32 bits without VK_EXT_shader_64bit_indexing, and the 6-float state passes 4 GiB at
// 179.0M Gaussians, the rotations at 268.4M and the 3-float rows at 357.9M. The addresses and the bytes are the ones
// the indices gave, so every backend computes the same values.
Ptr<float, Access.ReadWrite> row(Ptr<float, Access.ReadWrite> base, uint i, uint width) {
    return (Ptr<float, Access.ReadWrite>)(uint64_t(base) + uint64_t(i) * width * 4u);
}
Ptr<float, Access.Read> readRow(Ptr<float, Access.Read> base, uint i, uint width) {
    return (Ptr<float, Access.Read>)(uint64_t(base) + uint64_t(i) * width * 4u);
}

[shader("compute")]
[numthreads(256,1,1)]
void propose(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count) return;
    p.proposals[i] = unobserved;
    const Ptr<float, Access.ReadWrite> state = row(p.state, i, 6u);
    const Ptr<float, Access.ReadWrite> mean = row(p.means, i, 3u), scale = row(p.scales, i, 3u);
    const Ptr<float, Access.Read> gradient = readRow(p.gradient, i, 3u);
    float3 g, x, scales, anchor;
    [unroll] for (uint d = 0; d < 3; ++d) {
        x[d] = mean[d];
        scales[d] = scale[d];
        g[d] = gradient[d];
        anchor[d] = state[d];
    }
    float mask = 1;
    if (i < p.frozenCount && p.frozen[i] != 0) mask *= p.frozenScale;
    if (i < p.cropCount && p.crop[i] != 0) mask *= p.cropScale;
    float bound = (10.0 / 3.0) * mask * p.meanLr;
    if (i < p.farCount && p.far[i] != 0 && p.medianExtent > 0)
        bound *= clamp(exp((scales.x + scales.y + scales.z) / 3) / p.medianExtent, 1., 300.);
    if (!any(g != 0) || !(bound > 0)) return;
    float3 camera = viewPosition();
    float prevV = state[5];
    if (prevV < 0) {
        [unroll] for (uint d = 0; d < 3; ++d) {
            state[d] = camera[d];
            state[3+d] = 0;
        }
        return;
    }
    float3 b = camera - anchor, r = x - camera;
    float d2 = dot(r,r);
    if (!(d2 > 0)) return;
    r *= rsqrt(d2);
    float rb = dot(r,b), gRay = dot(b,g);
    float across = max(dot(b,b)-rb*rb, 0.);
    float spread = state[3];
    state[3] = spread == 0 && across > 0 ? across : spread + (across - spread) / 32.;
    float m = 0.9f * state[4] + (1.f-0.9f)*gRay;
    float v = prevV == 0 ? gRay*gRay : 0.999f*prevV + (1.f-0.999f)*gRay*gRay;
    state[4] = m;
    state[5] = v;
    p.proposals[i] = 0;
    if (!(v > 0)) return;
    float4 q;
    const Ptr<float, Access.Read> rotation = readRow(p.rotations, i, 4u);
    [unroll] for(uint j=0;j<4;++j) q[j] = rotation[j];
    float qq = dot(q,q), s = qq > 0 ? 2/qq : 0;
    float3 axes[3] = {
        float3(1-s*(q.z*q.z+q.w*q.w), s*(q.y*q.z+q.x*q.w), s*(q.y*q.w-q.x*q.z)),
        float3(s*(q.y*q.z-q.x*q.w), 1-s*(q.y*q.y+q.w*q.w), s*(q.z*q.w+q.x*q.y)),
        float3(s*(q.y*q.w+q.x*q.z), s*(q.z*q.w-q.x*q.y), 1-s*(q.y*q.y+q.z*q.z))};
    float variance = 0;
    [unroll] for(uint d=0;d<3;++d) {
        float3 a = cross(axes[d], r);
        variance += exp(2*scales[d])*dot(a,a);
    }
    float resolution = spread / max(variance*0.5f, 1e-30f);
    float dt = -p.lr*mask*clamp(m*rsqrt(v), -1., 1.);
    dt = dt > 0 && resolution < 0.01f ? 0 : dt/(1+resolution);
    float ratio = sqrt(max(spread, across))*dt/bound;
    p.proposals[i] = dt*rsqrt(1+ratio*ratio);
}

[shader("compute")]
[numthreads(256,1,1)]
void commit(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count) return;
    float dt = p.proposals[i];
    if (dt == unobserved) return;
    float3 camera = viewPosition(), anchor, x;
    const Ptr<float, Access.ReadWrite> state = row(p.state, i, 6u);
    const Ptr<float, Access.ReadWrite> mean = row(p.means, i, 3u), scale = row(p.scales, i, 3u);
    [unroll] for(uint d=0;d<3;++d) {
        x[d] = mean[d];
        anchor[d] = state[d];
        state[d] = anchor[d] + (camera[d]-anchor[d])/32.;
    }
    // Preserve small displacements when exp(dt) rounds to one.
    float e = abs(dt) < 1e-3f ? dt*(1+dt*(0.5f+dt/6)) : exp(dt)-1;
    float3 moved = x + e*(x-anchor);
    float halfFar = p.farLimit*0.5f;
    if (dt > 0 && dot(moved-camera,moved-camera) > halfFar*halfFar) return;
    [unroll] for(uint d=0;d<3;++d) {
        mean[d] = moved[d];
        scale[d] += dt;
    }
}

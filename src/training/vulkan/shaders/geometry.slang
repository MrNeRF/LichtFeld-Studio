#ifndef GEOMETRY_FP64
#define GEOMETRY_FP64 1
#endif
// Devices without shaderFloat64 (Apple GPUs through MoltenVK) accumulate the
// statistics in fp32, as the Metal family does.
#if GEOMETRY_FP64
typedef double real;
#else
typedef float real;
#endif

[SpecializationConstant] const uint operation = 0;
struct Parameters {
    uint64_t normal, depth, alpha, target, weight;
    uint64_t gradNormal, gradDepth, gradAlpha, loss, partials;
    uint64_t points, view, pairs, sampleCount;
    float loX, loY, loZ, hiX, hiY, hiZ;
    uint width, height, blocks, hasWeight, samples, stride, capacity;
    int anchorModel;
    uint prior;
    float fx, fy, cx, cy, lossWeight, lambdaGrad, quantStep, floorOverride;
    float anchorScale, anchorShift, anchorFloor, minCount, minWeight, nearPlane;
    uint64_t rays;
    uint wrapHorizontal;
};
groupshared real reduction[256];

float rd(uint64_t a, uint i) { return ((Ptr<float, Access.Read>)a)[i]; }
void wr(uint64_t a, uint i, float v) { ((Ptr<float, Access.ReadWrite>)a)[i] = v; }
bool finitef(float v) { return (asuint(v) & 0x7f800000u) != 0x7f800000u; }
float pixelWeight(Parameters p, uint i) {
    if (p.hasWeight == 0)
        return 1.0f;
    float w = rd(p.weight, i);
    return finitef(w) && w > 0.0f ? w : 0.0f;
}

// This XOR tree mirrors warp_reduce::block_reduce_sum for fp64 partials.
real reduceReal(real value, uint lane) {
    reduction[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 16; off > 0; off >>= 1) {
        real next = reduction[lane] + reduction[lane ^ off];
        // All readers must finish before any lane overwrites its source.
        GroupMemoryBarrierWithGroupSync();
        reduction[lane] = next;
        GroupMemoryBarrierWithGroupSync();
    }
    real warpSum = lane < 8 ? reduction[lane * 32] : 0.0;
    GroupMemoryBarrierWithGroupSync();
    reduction[lane] = warpSum;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 16; off > 0; off >>= 1) {
        real next = lane < 32 ? reduction[lane] + reduction[lane ^ off] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        if (lane < 32)
            reduction[lane] = next;
        GroupMemoryBarrierWithGroupSync();
    }
    real result = reduction[0];
    GroupMemoryBarrierWithGroupSync();
    return result;
}
void writeStats(Parameters p, uint lane, uint group, real vals[3], uint n, uint slotCount) {
    Ptr<real, Access.ReadWrite> out = (Ptr<real, Access.ReadWrite>)(p.partials + uint64_t(slotCount) * 4);
    for (uint k = 0; k < n; ++k) {
        real total = reduceReal(vals[k], lane);
        if (lane == 0)
            out[k * p.blocks + group] = total;
    }
}
void reduceStats(Parameters p, uint lane, inout real vals[3], uint n, uint slotCount) {
    Ptr<real, Access.Read> in = (Ptr<real, Access.Read>)(p.partials + uint64_t(slotCount) * 4);
    for (uint k = 0; k < n; ++k) {
        real s = 0.0;
        for (uint i = lane; i < p.blocks; i += 256)
            s += in[k * p.blocks + i];
        vals[k] = reduceReal(s, lane);
    }
}
void writeFinal(Parameters p, uint slot, float value) { wr(p.partials, slot, value); }
float finalValue(Parameters p, uint slot) { return rd(p.partials, slot); }

bool depthActive(Parameters p, uint i) {
    float t = rd(p.target, i), d = rd(p.depth, i), a = rd(p.alpha, i);
    return t > 0 && a > 1.0e-3f && finitef(t) && finitef(d) && finitef(a);
}
struct DepthSample {
    bool ok;
    float alpha, wa, e, inv, aligned, delta;
};
DepthSample depthSample(Parameters p, uint i, int model, float scale, float shift, float floorF, float pMax, float halfStep) {
    DepthSample s = {false, 0, 0, 0, 0, 0, 0};
    if (!depthActive(p, i))
        return s;
    float w = pixelWeight(p, i);
    if (w == 0)
        return s;
    float a = rd(p.alpha, i), d = rd(p.depth, i), t = rd(p.target, i);
    s.alpha = a;
    s.wa = a * w;
    s.e = max(d, 0.0f) / a;
    s.inv = 1.0f / (s.e + floorF);
    float fit = scale * t + shift;
    if (!(fit > 0))
        return s;
    if (model == 0) {
        s.aligned = min(fit, pMax);
        s.delta = halfStep;
    } else {
        s.aligned = 1.0f / (fit + floorF);
        s.delta = halfStep * s.aligned * s.aligned;
    }
    s.ok = true;
    return s;
}
float depthRho(float x) {
    float x2 = x * x;
    return 0.5f * x2 / (1.0f + x2);
}
float depthPsi(float x) {
    float x2 = x * x, d = 1.0f + x2;
    return x / (d * d);
}
float deadband(float r, float d) {
    float e = abs(r) - d;
    return e > 0 ? (r > 0 ? e : (r < 0 ? -e : 0)) : 0;
}

struct NormalSample {
    bool active;
    float weight, cos, norm;
    float3 tHat, nHat;
};
float3 normalizeGeometry(float3 value, out float norm) {
    float norm_sq = value.x * value.x + value.y * value.y + value.z * value.z;
    norm = float(sqrt(real(norm_sq)));
    float reciprocal = float(1.0 / real(norm));
    return float3(value.x * reciprocal, value.y * reciprocal, value.z * reciprocal);
}
NormalSample normalSample(Parameters p, uint i, uint pixels) {
    NormalSample s = {false, 0, 0, 0, float3(0), float3(0)};
    float a = rd(p.alpha, i);
    float3 t = float3(rd(p.target, i), rd(p.target, pixels + i), rd(p.target, 2 * pixels + i));
    float3 n = float3(rd(p.normal, i), rd(p.normal, pixels + i), rd(p.normal, 2 * pixels + i));
    if (!finitef(a) || !finitef(t.x) || !finitef(t.y) || !finitef(t.z) || !finitef(n.x) || !finitef(n.y) || !finitef(n.z) || a <= 1e-3f)
        return s;
    float w = pixelWeight(p, i);
    if (w == 0)
        return s;
    float tn;
    s.tHat = normalizeGeometry(t, tn);
    if (tn < 0.5f)
        return s;
    float nn;
    s.nHat = normalizeGeometry(n, nn);
    if (nn < 0.1f)
        return s;
    s.cos = s.nHat.x * s.tHat.x + s.nHat.y * s.tHat.y + s.nHat.z * s.tHat.z;
    s.weight = a * w;
    s.norm = nn;
    s.active = true;
    return s;
}
float3 ray(Parameters p, int x, int y) {
    if(p.wrapHorizontal!=0) x=(x+int(p.width))%int(p.width);
    if(p.rays!=0) { uint i=3*(y*p.width+x); return float3(rd(p.rays,i),rd(p.rays,i+1),rd(p.rays,i+2)); }
    float px = float(x) + 0.5f - p.cx, py = float(y) + 0.5f - p.cy;
    return float3(px / p.fx, py / p.fy, 1.0f);
}
bool expectedDepth(Parameters p, uint i, out float e, out float a) {
    e = 0.0f;
    a = rd(p.alpha, i);
    float d = rd(p.depth, i);
    if (!finitef(a) || !finitef(d) || a < 0.5f)
        return false;
    e = max(d, 0.0f) / a;
    return e >= 1e-6f;
}
struct DepthNormal {
    bool active;
    float alpha, sign, norm;
    float3 nd, tx, ty;
    float4 ne, na;
};
DepthNormal depthNormal(Parameters p, int x, int y) {
    DepthNormal s = (DepthNormal)0;
    if ((!p.wrapHorizontal && (x <= 0 || x >= int(p.width) - 1)) || y <= 0 || y >= int(p.height) - 1)
        return s;
    uint w = p.width, i = uint(y) * w + uint(x);
    float ec, ac;
    if (!expectedDepth(p, i, ec, ac))
        return s;
    uint ids[4] = {uint(y)*w+(uint(x)+1)%w, uint(y)*w+(uint(x)+w-1)%w, i+w, i-w};
    float jump = 0.05f * ec;
    for (uint k = 0; k < 4; ++k)
        if (!expectedDepth(p, ids[k], s.ne[k], s.na[k]) || abs(s.ne[k] - ec) > jump)
            return s;
    float3 rxp = ray(p, x + 1, y), rxm = ray(p, x - 1, y);
    float3 ryp = ray(p, x, y + 1), rym = ray(p, x, y - 1);
    s.tx = float3(fma(s.ne.x, rxp.x, -s.ne.y * rxm.x),
                  fma(s.ne.x, rxp.y, -s.ne.y * rxm.y),
                  fma(s.ne.x, rxp.z, -s.ne.y * rxm.z));
    s.ty = float3(fma(s.ne.z, ryp.x, -s.ne.w * rym.x),
                  fma(s.ne.z, ryp.y, -s.ne.w * rym.y),
                  fma(s.ne.z, ryp.z, -s.ne.w * rym.z));
    float3 n = float3(fma(s.tx.y, s.ty.z, -s.tx.z * s.ty.y),
                      fma(s.tx.z, s.ty.x, -s.tx.x * s.ty.z),
                      fma(s.tx.x, s.ty.y, -s.tx.y * s.ty.x));
    float ns = fma(n.z, n.z, fma(n.y, n.y, n.x * n.x));
    if (!finitef(ns) || ns < 1e-24f)
        return s;
    float3 r = ray(p, x, y);
    float face = fma(n.z, r.z, fma(n.y, r.y, n.x * r.x));
    s.sign = face > 0 ? -1.0f : 1.0f;
    s.norm = sqrt(ns);
    float nd_rcp = float(real(s.sign) / real(s.norm));
    s.nd = n * nd_rcp;
    s.alpha = ac;
    s.active = true;
    return s;
}
bool renderedHat(Parameters p, uint i, uint n, out float3 hat, out float norm) {
    float3 v = float3(rd(p.normal, i), rd(p.normal, n + i), rd(p.normal, 2 * n + i));
    norm = 0;
    hat = 0;
    if (!finitef(v.x) || !finitef(v.y) || !finitef(v.z))
        return false;
    float norm_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    norm = sqrt(norm_sq);
    if (norm < 0.1f)
        return false;
    float rcp_norm = float(1.0 / real(norm));
    hat = float3(v.x * rcp_norm, v.y * rcp_norm, v.z * rcp_norm);
    return true;
}
bool priorHat(Parameters p, uint i, uint n, out float3 hat) {
    float3 v = float3(rd(p.normal, i), rd(p.normal, n + i), rd(p.normal, 2 * n + i));
    hat = 0;
    if (!finitef(v.x) || !finitef(v.y) || !finitef(v.z))
        return false;
    float nrm;
    float norm_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    nrm = sqrt(norm_sq);
    if (nrm < 0.5f)
        return false;
    float rcp_norm = float(1.0 / real(nrm));
    hat = float3(v.x * rcp_norm, v.y * rcp_norm, v.z * rcp_norm);
    return true;
}
void atomicFloat(uint64_t addr, uint i, float val) {
    Ptr<uint, Access.ReadWrite> q = (Ptr<uint, Access.ReadWrite>)addr;
    uint old = q[i], seen;
    do {
        InterlockedCompareExchange(q[i], old, asuint(asfloat(old) + val), seen);
        if (seen == old)
            return;
        old = seen;
    } while (true);
}
void scatterDepthNormal(Parameters p, DepthNormal s, float3 target, float cosine, float gw, int x, int y) {
    float scale = -gw * s.sign / s.norm;
    float3 raw = scale * (target - cosine * s.nd);
    float3 gx = cross(s.ty, raw), gy = cross(raw, s.tx);
    float3 rxp = ray(p, x + 1, y), rxm = ray(p, x - 1, y), ryp = ray(p, x, y + 1), rym = ray(p, x, y - 1);
    float ge[4] = {dot(gx, rxp), -dot(gx, rxm), dot(gy, ryp), -dot(gy, rym)};
    uint w = p.width, i = uint(y) * w + uint(x), ids[4] = {uint(y)*w+(uint(x)+1)%w, uint(y)*w+(uint(x)+w-1)%w, i+w, i-w};
    for (uint k = 0; k < 4; ++k) {
        float inv = 1.0f / s.na[k];
        atomicFloat(p.gradDepth, ids[k], ge[k] * inv);
        atomicFloat(p.gradAlpha, ids[k], -ge[k] * s.ne[k] * inv);
    }
}

[numthreads(256, 1, 1)] void main(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex, uint3 tid : SV_DispatchThreadID, uniform Parameters p) {
    uint3 ng = spirv_asm {
result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    uint pixels = p.width * p.height;
    if (operation == 0 || operation == 2 || operation == 4 || operation == 6 || operation == 8 || operation == 10 || operation == 12) {
        real sums[3] = {0.0, 0.0, 0.0};
        if (operation == 0) {
            for (uint i = groupId.x * 256 + lane; i < pixels; i += p.blocks * 256)
                if (depthActive(p, i)) {
                    float w = pixelWeight(p, i);
                    if (w != 0) {
                        float a = rd(p.alpha, i), d = rd(p.depth, i);
                        real aw = real(a) * real(w), e = max(d, 0.0f) / a;
                        sums[0] += aw;
                        sums[1] += aw * real(e);
                        sums[2] += 1.0;
                    }
                }
            writeStats(p, lane, groupId.x, sums, 3, 10);
            return;
        }
        if (operation == 2) {
            float floorF = p.floorOverride > 0 ? p.floorOverride : finalValue(p, 4);
            bool valid = finalValue(p, 0) > 0.5f;
            if (valid)
                for (uint i = groupId.x * 256 + lane; i < pixels; i += p.blocks * 256)
                    if (depthActive(p, i)) {
                        float w = pixelWeight(p, i);
                        if (w != 0) {
                            float a = rd(p.alpha, i), d = rd(p.depth, i), e = max(d, 0.0f) / a, ipf = 1.0f / (e + floorF);
                            real ip = real(ipf), aw = real(a) * real(w);
                            sums[0] += aw * ip;
                            sums[1] += aw * ip * ip;
                        }
                    }
            writeStats(p, lane, groupId.x, sums, 2, 10);
            return;
        }
        if (operation == 4) {
            int slot = 10;
            real vals[3] = {0, 0, 0};
            float valid = finalValue(p, 0), invNorm = finalValue(p, 5), modelF = finalValue(p, 1), scale = finalValue(p, 2), shift = finalValue(p, 3), floorF = finalValue(p, 4), sigma = finalValue(p, 9);
            int model = int(modelF);
            float pMax = 1.0f / floorF, halfStep = 0.5f * max(p.quantStep, 0.0f) * abs(scale), invSigma = valid > 0.5f && sigma > 0 ? 1.0f / (2.0f * sigma) : 0.0f, lambda = p.lambdaGrad;
            for (uint i = groupId.x * 256 + lane; i < pixels; i += p.blocks * 256) {
                uint x = i % p.width, y = i / p.width;
                if (valid <= 0.5f) {
                    wr(p.gradDepth, i, 0);
                    wr(p.gradAlpha, i, 0);
                    continue;
                }
                DepthSample c = depthSample(p, i, model, scale, shift, floorF, pMax, halfStep);
                float gp = 0;
                if (c.ok) {
                    float xr = deadband(c.inv - c.aligned, c.delta) * invSigma;
                    if (xr != 0) {
                        vals[0] += real(c.wa) * real(depthRho(xr));
                        gp += c.wa * depthPsi(xr) * invSigma;
                    }
                    if (x + 1 < p.width) {
                        DepthSample n = depthSample(p, i + 1, model, scale, shift, floorF, pMax, halfStep);
                        if (n.ok) {
                            float q = deadband((n.inv - c.inv) - (n.aligned - c.aligned), c.delta + n.delta) * invSigma;
                            if (q != 0) {
                                float w = min(c.wa, n.wa);
                                vals[1] += real(w) * real(depthRho(q));
                                gp -= lambda * w * depthPsi(q) * invSigma;
                            }
                        }
                    }
                    if (y + 1 < p.height) {
                        DepthSample n = depthSample(p, i + p.width, model, scale, shift, floorF, pMax, halfStep);
                        if (n.ok) {
                            float q = deadband((n.inv - c.inv) - (n.aligned - c.aligned), c.delta + n.delta) * invSigma;
                            if (q != 0) {
                                float w = min(c.wa, n.wa);
                                vals[1] += real(w) * real(depthRho(q));
                                gp -= lambda * w * depthPsi(q) * invSigma;
                            }
                        }
                    }
                    if (x > 0) {
                        DepthSample n = depthSample(p, i - 1, model, scale, shift, floorF, pMax, halfStep);
                        if (n.ok) {
                            float q = deadband((c.inv - n.inv) - (c.aligned - n.aligned), c.delta + n.delta) * invSigma;
                            if (q != 0)
                                gp += lambda * min(n.wa, c.wa) * depthPsi(q) * invSigma;
                        }
                    }
                    if (y > 0) {
                        DepthSample n = depthSample(p, i - p.width, model, scale, shift, floorF, pMax, halfStep);
                        if (n.ok) {
                            float q = deadband((c.inv - n.inv) - (c.aligned - n.aligned), c.delta + n.delta) * invSigma;
                            if (q != 0)
                                gp += lambda * min(n.wa, c.wa) * depthPsi(q) * invSigma;
                        }
                    }
                }
                if (c.ok && gp != 0) {
                    float g = p.lossWeight * invNorm * gp * (-c.inv * c.inv);
                    wr(p.gradDepth, i, g / c.alpha);
                    wr(p.gradAlpha, i, -g * c.e / c.alpha);
                } else {
                    wr(p.gradDepth, i, 0);
                    wr(p.gradAlpha, i, 0);
                }
            }
            writeStats(p, lane, groupId.x, vals, 2, 10);
            return;
        }
        if (operation == 6) {
            for (uint i = groupId.x * 256 + lane; i < pixels; i += p.blocks * 256) {
                NormalSample s = normalSample(p, i, pixels);
                if (s.active) {
                    sums[0] += real(s.weight);
                    sums[1] += 1.0;
                    sums[2] += real(s.weight) * real(s.cos);
                }
            }
            writeStats(p, lane, groupId.x, sums, 3, 6);
            return;
        }
        if (operation == 8) {
            real vals[3] = {0, 0, 0};
            float invNorm = finalValue(p, 4);
            for (uint i = groupId.x * 256 + lane; i < pixels; i += p.blocks * 256) {
                NormalSample s = normalSample(p, i, pixels);
                if (!s.active || invNorm == 0) {
                    wr(p.gradNormal, i, 0);
                    wr(p.gradNormal, pixels + i, 0);
                    wr(p.gradNormal, 2 * pixels + i, 0);
                    continue;
                }
                vals[0] += real(s.weight) * (1.0 - real(s.cos));
                float scale = -invNorm * s.weight / s.norm;
                float3 g = scale * (s.tHat - s.cos * s.nHat);
                wr(p.gradNormal, i, g.x);
                wr(p.gradNormal, pixels + i, g.y);
                wr(p.gradNormal, 2 * pixels + i, g.z);
            }
            writeStats(p, lane, groupId.x, vals, 1, 6);
            return;
        }
        if (operation == 10) {
            for (uint i = groupId.x * 256 + lane; i < pixels; i += p.blocks * 256) {
                uint x = i % p.width, y = i / p.width;
                DepthNormal s = depthNormal(p, int(x), int(y));
                if (!s.active)
                    continue;
                float3 hat;
                float norm = 0;
                bool ok = p.prior != 0 ? priorHat(p, i, pixels, hat) : renderedHat(p, i, pixels, hat, norm);
                if (!ok)
                    continue;
                float cosine = fma(s.nd.z, hat.z, fma(s.nd.y, hat.y, s.nd.x * hat.x));
                float aw = s.alpha * pixelWeight(p, i);
                if (aw != 0) {
                    sums[0] += real(aw);
                    sums[1] += 1.0;
                    sums[2] += real(aw) * real(cosine);
                }
            }
            writeStats(p, lane, groupId.x, sums, 3, 6);
            return;
        }
        if (operation == 12) {
            real vals[3] = {0, 0, 0};
            float invNorm = finalValue(p, 4);
            for (uint i = groupId.x * 256 + lane; i < pixels; i += p.blocks * 256) {
                if (invNorm == 0)
                    continue;
                uint x = i % p.width, y = i / p.width;
                DepthNormal s = depthNormal(p, int(x), int(y));
                if (!s.active)
                    continue;
                float3 hat;
                float norm = 0;
                bool ok = p.prior != 0 ? priorHat(p, i, pixels, hat) : renderedHat(p, i, pixels, hat, norm);
                if (!ok)
                    continue;
                float cosine = fma(s.nd.z, hat.z, fma(s.nd.y, hat.y, s.nd.x * hat.x));
                float aw = s.alpha * pixelWeight(p, i);
                if (aw == 0)
                    continue;
                vals[0] += real(aw) * (1.0 - real(cosine));
                float gw = invNorm * aw;
                if (p.prior == 0) {
                    float3 g = (-gw / norm) * (s.nd - cosine * hat);
                    wr(p.gradNormal, i, rd(p.gradNormal, i) + g.x);
                    wr(p.gradNormal, pixels + i, rd(p.gradNormal, pixels + i) + g.y);
                    wr(p.gradNormal, 2 * pixels + i, rd(p.gradNormal, 2 * pixels + i) + g.z);
                }
                scatterDepthNormal(p, s, hat, cosine, gw, int(x), int(y));
            }
            writeStats(p, lane, groupId.x, vals, 1, 6);
            return;
        }
    }
    if (operation == 1 || operation == 3 || operation == 5 || operation == 7 || operation == 9 || operation == 11 || operation == 13) {
        real vals[3] = {0, 0, 0};
        uint n = (operation == 1) ? 3 : (operation == 3)                  ? 2
                                    : (operation == 5)                    ? 2
                                    : (operation == 7 || operation == 11) ? 3
                                    : (operation == 9 || operation == 13) ? 1
                                                                          : 0;
        uint slots = (operation <= 5) ? 10 : 6;
        reduceStats(p, lane, vals, n, slots);
        if (lane != 0)
            return;
        if (operation == 1) {
            real sa = vals[0], count = vals[2];
            bool valid = sa > 0;
            real me = valid ? vals[1] / sa : 0;
            writeFinal(p, 0, valid ? 1 : 0);
            writeFinal(p, 4, max(1e-8f, 0.05f * float(me)));
            writeFinal(p, 6, float(sa));
            writeFinal(p, 7, float(count));
            writeFinal(p, 8, float(me));
            return;
        }
        if (operation == 3) {
            float floorF = p.floorOverride > 0 ? p.floorOverride : finalValue(p, 4);
            bool valid = finalValue(p, 0) > 0.5f && p.anchorFloor > 0 && finitef(p.anchorScale) && finitef(p.anchorShift);
            real sa = real(finalValue(p, 6)), sigma = 0;
            if (valid) {
                real mean = vals[0] / sa, var = max(vals[1] / sa - mean * mean, 0.0);
                sigma = sqrt(max(var, 1e-20));
                writeFinal(p, 4, p.anchorFloor);
            }
            writeFinal(p, 0, valid ? 1 : 0);
            writeFinal(p, 1, float(p.anchorModel));
            writeFinal(p, 2, p.anchorScale);
            writeFinal(p, 3, p.anchorShift);
            writeFinal(p, 9, float(sigma));
            writeFinal(p, 5, valid ? float(1.0 / sa) : 0);
            return;
        }
        if (operation == 5) {
            bool valid = finalValue(p, 0) > 0.5f;
            float inv = finalValue(p, 5);
            wr(p.loss, 0, valid ? p.lossWeight * inv * float(vals[0] + real(p.lambdaGrad) * vals[1]) : 0);
            return;
        }
        if (operation == 7 || operation == 11) {
            float sa = float(vals[0]), count = float(vals[1]);
            bool valid = count >= p.minCount && sa >= p.minWeight;
            writeFinal(p, 0, valid ? 1 : 0);
            writeFinal(p, 1, sa);
            writeFinal(p, 2, count);
            writeFinal(p, 3, sa > 0 ? float(vals[2] / vals[0]) : 0);
            writeFinal(p, 4, valid ? float(real(p.lossWeight) / max(vals[0], 1.0)) : 0);
            return;
        }
        if (operation == 9 || operation == 13) {
            wr(p.loss, 0, finalValue(p, 4) * float(vals[0]));
            return;
        }
    }
    if (operation == 14) {
        uint s = tid.x;
        if (s >= p.samples)
            return;
        uint i = s * p.stride;
        float x = rd(p.points, 3 * i), y = rd(p.points, 3 * i + 1), zw = rd(p.points, 3 * i + 2);
        if (!finitef(x) || !finitef(y) || !finitef(zw) || x < p.loX || x > p.hiX || y < p.loY || y > p.hiY || zw < p.loZ || zw > p.hiZ)
            return;
        float z = rd(p.view, 8) * x + rd(p.view, 9) * y + rd(p.view, 10) * zw + rd(p.view, 11);
        if (!(z > p.nearPlane) || !finitef(z))
            return;
        float xc = rd(p.view, 0) * x + rd(p.view, 1) * y + rd(p.view, 2) * zw + rd(p.view, 3), yc = rd(p.view, 4) * x + rd(p.view, 5) * y + rd(p.view, 6) * zw + rd(p.view, 7);
        int u = int(floor(p.fx * xc / z + p.cx)), v = int(floor(p.fy * yc / z + p.cy));
        if (u < 0 || u >= int(p.width) || v < 0 || v >= int(p.height))
            return;
        float t = rd(p.target, uint(v) * p.width + uint(u));
        if (!(t > 0) || !finitef(t))
            return;
        Ptr<int, Access.ReadWrite> count = (Ptr<int, Access.ReadWrite>)p.sampleCount;
        int slot;
        InterlockedAdd(count[0], 1, slot);
        if (slot < int(p.capacity)) {
            Ptr<float, Access.ReadWrite> pair = (Ptr<float, Access.ReadWrite>)p.pairs;
            pair[uint(slot) * 2] = t;
            pair[uint(slot) * 2 + 1] = z;
        }
        return;
    }
}

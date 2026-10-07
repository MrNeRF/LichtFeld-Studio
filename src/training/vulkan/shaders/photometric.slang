/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
[SpecializationConstant] const uint operation = 0;
struct Params {
    uint64_t corrected, raw, target, mask, horizontal, partials;
    uint64_t ssim, cs, grad, gradRaw, losses, normalizer;
    uint count, channels, height, width, batch, stage, path, targetByte;
    uint maskByte, validPadding, partialStride, reserved;
    float weight, denominator;
};
struct Push { uint64_t parameters; };
[[vk::push_constant]] Push push;
static const float weights[6] = {0.26601171493530273f, 0.21300552785396576f,
    0.10936068743467331f, 0.036000773310661316f, 0.0075987582094967365f, 0.001028380123898387f};
float readFloat(uint64_t address, uint i) { return ((Ptr<float, Access.Read>)address)[i]; }
void writeFloat(uint64_t address, uint i, float v) { ((Ptr<float, Access.ReadWrite>)address)[i] = v; }
bool decoupled(Params p) { return (operation >> 3u) == 3 || (operation >> 3u) == 5; }
bool masked(Params p) { return (operation >> 3u) >= 4; }
float targetAt(Params p, uint i) {
    return p.targetByte != 0 ? float(((Ptr<uint8_t, Access.Read>)p.target)[i]) * (1.0f / 255.0f) : readFloat(p.target, i);
}
float maskAt(Params p, uint i) {
    if (!masked(p)) return 1.0f;
    return p.maskByte != 0 ? (((Ptr<uint8_t, Access.Read>)p.mask)[i] != 0 ? 1.0f : 0.0f) : readFloat(p.mask, i);
}
float moment(Params p, uint base, int x, int y, uint field) {
    if (x < 0 || y < 0 || x >= int(p.width) || y >= int(p.height)) return 0;
    uint i = base + uint(y) * p.width + uint(x);
    float a = readFloat(p.corrected, i), t = targetAt(p, i);
    float r = decoupled(p) ? readFloat(p.raw, i) : a;
    if (field == 0) return a;
    if (field == 1) return r;
    if (field == 2) return r * r;
    if (field == 3) return t;
    if (field == 4) return t * t;
    return r * t;
}
float momentPair(Params p, uint base, int x, int y, int d, uint field) {
    if (field == 2 || field == 4 || field == 5) {
        uint source = field == 4 ? 3 : 1;
        float left = moment(p,base,x-d,y,source), right = moment(p,base,x+d,y,source);
        if (field != 5) return fma(left,left,right*right);
        return fma(left,moment(p,base,x-d,y,3),right*moment(p,base,x+d,y,3));
    }
    return moment(p,base,x-d,y,field)+moment(p,base,x+d,y,field);
}
float horizontalAt(Params p, uint base, int x, int y, uint field) {
    if (x < 0 || y < 0 || x >= int(p.width) || y >= int(p.height)) return 0;
    return readFloat(p.horizontal, field * p.count + base + uint(y) * p.width + uint(x));
}
float chain(Params p, int x, int y) {
    if (x < 0 || y < 0 || x >= int(p.width) || y >= int(p.height)) return 0;
    if (masked(p))
        return maskAt(p, uint(y) * p.width + uint(x)) / (p.denominator > 0 ? p.denominator : readFloat(p.normalizer, 0) * float(p.batch) * float(p.channels) + 1e-8f);
    bool valid = (operation >> 3u) != 0 && p.validPadding != 0;
    bool cropX = valid && p.width > 10, cropY = valid && p.height > 10;
    if ((cropX && (x < 5 || x >= int(p.width) - 5)) ||
        (cropY && (y < 5 || y >= int(p.height) - 5))) return 0;
    uint width = cropX ? p.width - 10 : p.width;
    uint height = cropY ? p.height - 10 : p.height;
    return 1.0f / float(p.batch * p.channels * width * height);
}
void storePartial(Params p, uint i, uint field, float value) {
    if (p.partials == 0) return;
    ((Ptr<uint16_t, Access.ReadWrite>)p.partials)[field * p.partialStride + i] = uint16_t(f32tof16(value));
}
float partial(Params p, uint base, int x, int y, uint field) {
    if (x < 0 || y < 0 || x >= int(p.width) || y >= int(p.height)) return 0;
    uint i = base + uint(y) * p.width + uint(x);
    float value = f16tof32(((Ptr<uint16_t, Access.Read>)p.partials)[field * p.partialStride + i]);
    float weight = decoupled(p) && field > 0 ? 1.0f : p.weight;
    return -weight * value * chain(p, x, y);
}
// The horizontal derivative filter reuses neighboring partials across lanes.
// Keep the existing workspace shape: allocation placement must not increase VRAM.
groupshared float derivativeTile[4][266];
void horizontalDerivatives(Params p, uint group, uint lane, uint groupCount) {
    uint pixels = p.width * p.height;
    uint fields = decoupled(p) ? 4 : 3;
    for (uint start = group * 256; start < p.count; start += groupCount * 256) {
        for (uint j = lane; j < 266; j += 256) {
            int source = int(start) + int(j) - 5;
            bool valid = source >= 0 && uint(source) < p.count;
            uint index = valid ? uint(source) : 0;
            uint base = (index / pixels) * pixels;
            int x = int(index % p.width), y = int((index % pixels) / p.width);
            [unroll] for (uint field = 0; field < 4; ++field)
                if (field < fields)
                    derivativeTile[field][j] = valid ? partial(p, base, x, y, field) : 0;
        }
        GroupMemoryBarrierWithGroupSync();
        uint i = start + lane;
        if (i < p.count) {
            int x = int(i % p.width);
            [unroll] for (uint field = 0; field < 4; ++field) {
                if (field >= fields) continue;
                float sum = 0;
                [unroll] for (int d = 1; d <= 5; ++d) {
                    float left = x >= d ? derivativeTile[field][lane + 5 - d] : 0;
                    float right = x + d < int(p.width) ? derivativeTile[field][lane + 5 + d] : 0;
                    sum = fma(left + right, weights[d], sum);
                }
                writeFloat(p.horizontal, field * p.count + i,
                           fma(derivativeTile[field][lane + 5], weights[0], sum));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Params p = ((Ptr<Params, Access.Read>)push.parameters)[0];
    if ((operation & 7u) == 2) {
        horizontalDerivatives(p, gid.x, lane, groups.x);
        return;
    }
    uint pixels = p.width * p.height;
    for (uint i = gid.x * 256 + lane; i < p.count; i += groups.x * 256) {
        uint base = (i / pixels) * pixels;
        int x = int(i % p.width), y = int((i % pixels) / p.width);
        if ((operation & 7u) == 0) {
            float sums[6] = {0, 0, 0, 0, 0, 0};
            for (int d = 1; d <= 5; ++d) {
                float al = moment(p, base, x-d, y, 0), ar = moment(p, base, x+d, y, 0);
                float rl = decoupled(p) ? moment(p, base, x-d, y, 1) : al;
                float rr = decoupled(p) ? moment(p, base, x+d, y, 1) : ar;
                float tl = moment(p, base, x-d, y, 3), tr = moment(p, base, x+d, y, 3);
                sums[0] = fma(al + ar, weights[d], sums[0]);
                sums[1] = fma(rl + rr, weights[d], sums[1]);
                sums[2] = fma(fma(rl, rl, rr * rr), weights[d], sums[2]);
                sums[3] = fma(tl + tr, weights[d], sums[3]);
                sums[4] = fma(fma(tl, tl, tr * tr), weights[d], sums[4]);
                sums[5] = fma(fma(rl, tl, rr * tr), weights[d], sums[5]);
            }
            [unroll] for (uint field = 0; field < 6; ++field)
                if (decoupled(p) || field != 1) writeFloat(p.horizontal, field * p.count + i,
                           fma(moment(p, base, x, y, field), weights[0], sums[field]));
        } else if ((operation & 7u) == 1) {
            float m[6];
            [unroll] for (uint field = 0; field < 6; ++field) {
                if (!decoupled(p) && field == 1) { m[field] = m[0]; continue; }
                float sum = 0;
                for (int d = 1; d <= 5; ++d)
                    sum = fma((horizontalAt(p, base, x, y-d, field) + horizontalAt(p, base, x, y+d, field)), weights[d], sum);
                m[field] = fma(horizontalAt(p, base, x, y, field) , weights[0], sum);
            }
            float a = m[0], r = m[1], t = m[3];
            float A = fma(a,a,t*t) + 0.01f*0.01f;
            float B = fma(-r,r,m[2]) + fma(-t,t,m[4]) + 0.03f*0.03f;
            float C = fma(2*a,t,0.01f*0.01f), D = fma(2.0f,fma(-r,t,m[5]),0.03f*0.03f);
            float invAB = 1.0f/(A*B), invAAB = 1.0f/(A*A*B), invABB = 1.0f/(A*B*B);
            float similarity;
            if (decoupled(p)) {
                float lum = C/A, cs = D/B;
                similarity = lum*cs;
                storePartial(p, i, 0, cs * (2*(t*A-a*C)/(A*A)));
                float dv = p.weight * (-(lum*D)/(B*B));
                float dc = p.weight * ((2*lum)/B);
                storePartial(p, i, 1, dv*(-2*r) + dc*(-t));
                storePartial(p, i, 2, dv);
                storePartial(p, i, 3, dc);
            } else {
                similarity = (C*D)*invAB;
                storePartial(p, i, 0, (t*2*D)*invAB - (t*2*C)*invAB - (a*2*C*D)*invAAB + (a*2*C*D)*invABB);
                storePartial(p, i, 1, (-C*D)*invABB);
                storePartial(p, i, 2, (2*C)*invAB);
            }
            writeFloat(p.ssim, i, similarity);
            writeFloat(p.cs, i, D/B);
        } else if ((operation & 7u) == 2) {
            uint fields = decoupled(p) ? 4 : 3;
            for (uint field = 0; field < fields; ++field) {
                float sum = 0;
                for (int d = 1; d <= 5; ++d)
                    sum = fma((partial(p, base, x-d, y, field) + partial(p, base, x+d, y, field)), weights[d], sum);
                sum = fma(partial(p, base, x, y, field), weights[0], sum);
                writeFloat(p.horizontal, field*p.count+i, sum);
            }
        } else if ((operation & 7u) == 3) {
            float m[4];
            uint fields = decoupled(p) ? 4 : 3;
            for (uint field = 0; field < fields; ++field) {
                float sum = 0;
                for (int d = 1; d <= 5; ++d)
                    sum = fma((horizontalAt(p, base, x, y-d, field) + horizontalAt(p, base, x, y+d, field)), weights[d], sum);
                m[field] = fma(horizontalAt(p, base, x, y, field), weights[0], sum);
            }
            float a = readFloat(p.corrected, i), t = targetAt(p,i);
            float l1 = (1-p.weight)*sign(a-t)*chain(p,x,y);
            writeFloat(p.grad, i, l1 + (decoupled(p) ? m[0] : fma(t,m[2],fma(2*a,m[1],m[0]))));
            if (decoupled(p)) writeFloat(p.gradRaw, i, fma(t,m[3],fma(2*readFloat(p.raw,i),m[2],m[1])));
        } else if ((operation & 7u) == 4) {
            float a = readFloat(p.corrected,i), t = targetAt(p,i);
            float loss = abs(a-t);
            if ((operation >> 3u) != 0) loss = (1-p.weight)*loss + p.weight*(1-readFloat(p.ssim,i));
            writeFloat(p.losses, i, loss*maskAt(p,i%pixels));
            if ((operation >> 3u) == 0) writeFloat(p.grad,i,sign(a-t)/float(p.count));
        } else if ((operation & 7u) == 5) {
            // Channel mean, one invocation per output pixel.
            uint maps = p.batch * pixels;
            if (i >= maps) continue;
            uint b = i/pixels, pixel = i%pixels;
            float s = 0, c = 0;
            for (uint channel = 0; channel < p.channels; ++channel) {
                s += readFloat(p.ssim, (b*p.channels+channel)*pixels+pixel);
                c += readFloat(p.cs, (b*p.channels+channel)*pixels+pixel);
            }
            writeFloat(p.grad,i,s/float(p.channels));
            writeFloat(p.gradRaw,i,c/float(p.channels));
        } else if ((operation & 7u) == 6) {
            if (i >= pixels) continue;
            float sum = 0;
            for (uint channel = 0; channel < p.channels; ++channel) sum += readFloat(p.ssim,channel*pixels+i);
            writeFloat(p.grad,i,max(1-sum/float(p.channels),0.0f));
        }
    }
}

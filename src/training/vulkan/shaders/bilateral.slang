#include "ppisp_math.slangh"
static const float C2G_R = 0.299f;
static const float C2G_G = 0.587f;
static const float C2G_B = 0.114f;
[SpecializationConstant] const uint operation = 0;

struct Parameters {
    uint64_t grid, rgb, gradOutput, offset, output, gradGrid, partials;
    uint count;
    int N, C, L, H, W, h, w, normN, partialCount;
    uint chw, exposureChroma, perImage;
    float gradLoss, lr, beta1, beta2, bc1Rcp, bc2SqrtRcp, eps, scale1, scale2, spatial, invSpatial;
};

struct Cell { int x0,y0,z0,x1,y1,z1; float fx,fy,fz,z; };
groupshared float reduction[256];

float loadRgb(Parameters p, uint pixel, uint channel) {
    const uint hw = uint(p.h * p.w);
    return ((Ptr<float, Access.Read>)p.rgb)[p.chw != 0 ? channel * hw + pixel : pixel * 3 + channel];
}
void storeRgb(Parameters p, uint pixel, float3 value) {
    const uint hw = uint(p.h * p.w);
    Ptr<float, Access.ReadWrite> out = (Ptr<float, Access.ReadWrite>)p.output;
    for (uint c = 0; c < 3; ++c) out[p.chw != 0 ? c * hw + pixel : pixel * 3 + c] = value[c];
}
float3 finiteRgb(Parameters p, uint pixel, uint64_t address, float fallback) {
    float3 v;
    for (uint c = 0; c < 3; ++c) {
        const uint hw = uint(p.h * p.w);
        const uint at = p.chw != 0 ? c * hw + pixel : pixel * 3 + c;
        const float x = ((Ptr<float, Access.Read>)address)[at];
        v[c] = isfinite(x) ? x : fallback;
    }
    return v;
}
Cell cellAt(Parameters p, uint pixel, float3 rgb) {
    const int hi = int(pixel / uint(p.w)), wi = int(pixel % uint(p.w));
    const float x = p.w > 1 ? float(wi) / float(p.w - 1) * float(p.W - 1) : 0.0f;
    const float y = p.h > 1 ? float(hi) / float(p.h - 1) * float(p.H - 1) : 0.0f;
    const float g = clamp(C2G_R * rgb.x + C2G_G * rgb.y + C2G_B * rgb.z, 0.0f, 1.0f);
    Cell c;
    c.z = g * float(p.L - 1); c.x0 = int(floor(x)); c.y0 = int(floor(y));
    c.z0 = clamp(int(floor(c.z)), 0, p.L - 1); c.x1 = min(c.x0 + 1, p.W - 1);
    c.y1 = min(c.y0 + 1, p.H - 1); c.z1 = clamp(int(floor(c.z)) + 1, 0, p.L - 1);
    c.fx = x - float(c.x0); c.fy = y - float(c.y0); c.fz = c.z - float(c.z0);
    return c;
}
float sampleGrid(Parameters p, Cell c, uint channel) {
    Ptr<float, Access.Read> grid = (Ptr<float, Access.Read>)p.grid;
    const uint plane = uint(p.L * p.H * p.W), base = channel * plane;
    const uint a = base + uint((c.z0 * p.H + c.y0) * p.W + c.x0);
    const uint b = base + uint((c.z0 * p.H + c.y0) * p.W + c.x1);
    const uint d = base + uint((c.z0 * p.H + c.y1) * p.W + c.x0);
    const uint e = base + uint((c.z0 * p.H + c.y1) * p.W + c.x1);
    const uint f = base + uint((c.z1 * p.H + c.y0) * p.W + c.x0);
    const uint g = base + uint((c.z1 * p.H + c.y0) * p.W + c.x1);
    const uint h = base + uint((c.z1 * p.H + c.y1) * p.W + c.x0);
    const uint i = base + uint((c.z1 * p.H + c.y1) * p.W + c.x1);
    const float v00 = grid[a] * (1-c.fx) + grid[b] * c.fx;
    const float v01 = grid[d] * (1-c.fx) + grid[e] * c.fx;
    const float v10 = grid[f] * (1-c.fx) + grid[g] * c.fx;
    const float v11 = grid[h] * (1-c.fx) + grid[i] * c.fx;
    const float offset = p.offset != 0 ? ((Ptr<float, Access.Read>)p.offset)[channel] : 0.0f;
    return (v00 * (1-c.fy) + v01 * c.fy) * (1-c.fz) + (v10 * (1-c.fy) + v11 * c.fy) * c.fz + offset;
}
void atomicFloatAdd(uint64_t address, uint index, float value) {
    Ptr<uint, Access.ReadWrite> word = (Ptr<uint, Access.ReadWrite>)address;
    uint assumed = word[index], observed;
    do {
        InterlockedCompareExchange(word[index], assumed, asuint(asfloat(assumed) + value), observed);
        if (observed == assumed) return;
        assumed = observed;
    } while (true);
}

[numthreads(256,1,1)]
void main(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex, uint3 tid : SV_DispatchThreadID, uniform Parameters p) {
    uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    if (operation == 0 || operation == 1 || operation == 2) {
        const uint pixels = uint(p.h * p.w);
        for (uint pixel = tid.x; pixel < pixels; pixel += groups.x * 256) {
            const float3 rgb = finiteRgb(p, pixel, p.rgb, 0.5f);
            const Cell c = cellAt(p, pixel, rgb);
            if (operation == 0) {
                float3 out = 0.0f;
                if (p.exposureChroma != 0) {
                    float sampled[9];
                    for (uint ci = 0; ci < 9; ++ci) sampled[ci] = sampleGrid(p, c, ci);
                    PpColor color;
                    color.b = float2(sampled[1], sampled[2]); color.r = float2(sampled[3], sampled[4]);
                    color.g = float2(sampled[5], sampled[6]); color.n = float2(sampled[7], sampled[8]);
                    out = ppApplyColor(ppApplyExposure(rgb, sampled[0]), color);
                } else {
                    for (uint ci = 0; ci < 12; ++ci) {
                        const float coeff = ci % 4 == 3 ? 1.0f : rgb[ci % 4];
                        out[ci / 4] += sampleGrid(p, c, ci) * coeff;
                    }
                }
                for (uint ch = 0; ch < 3; ++ch) if (!isfinite(out[ch])) out[ch] = 0.5f;
                storeRgb(p, pixel, out);
            } else {
                const float3 grad = finiteRgb(p, pixel, p.gradOutput, 0.0f);
                const float wx0 = 1.0f - c.fx, wy0 = 1.0f - c.fy;
                const float w00 = wx0 * wy0, w01 = c.fx * wy0, w10 = wx0 * c.fy, w11 = c.fx * c.fy;
                float3 gradRgb = 0.0f;
                float gradZ = 0.0f;
                Ptr<float, Access.Read> grid = (Ptr<float, Access.Read>)p.grid;
                if (p.exposureChroma != 0) {
                    float sampled[9], dl[9] = {0,0,0,0,0,0,0,0,0}, dc[8] = {0,0,0,0,0,0,0,0};
                    for (uint ci=0; ci<9; ++ci) sampled[ci]=sampleGrid(p,c,ci);
                    PpColor color; color.b=float2(sampled[1],sampled[2]); color.r=float2(sampled[3],sampled[4]);
                    color.g=float2(sampled[5],sampled[6]); color.n=float2(sampled[7],sampled[8]);
                    const float3 exposed=ppApplyExposure(rgb,sampled[0]);
                    const float3 gradExposed=ppApplyColorBwd(exposed,color,grad,dc);
                    gradRgb=ppApplyExposureBwd(rgb,sampled[0],gradExposed,dl[0]);
                    for(uint i=0;i<8;++i) dl[i+1]=dc[i];
                    const int xs[8]={c.x0,c.x1,c.x0,c.x1,c.x0,c.x1,c.x0,c.x1};
                    const int ys[8]={c.y0,c.y0,c.y1,c.y1,c.y0,c.y0,c.y1,c.y1};
                    const int zs[8]={c.z0,c.z0,c.z0,c.z0,c.z1,c.z1,c.z1,c.z1};
                    const float dwdz[8]={-w00,-w01,-w10,-w11,w00,w01,w10,w11};
                    for(uint ci=0;ci<9;++ci) for(uint k=0;k<8;++k) {
                        const uint at=(ci*uint(p.L)+uint(zs[k]))*uint(p.H*p.W)+uint(ys[k]*p.W+xs[k]);
                        gradZ+=dwdz[k]*float(p.L-1)*grid[at]*dl[ci];
                    }
                    if(c.z0==c.z || c.z1==c.z) gradZ=0.0f;
                    storeRgb(p,pixel,gradRgb+float3(C2G_R,C2G_G,C2G_B)*gradZ);
                    if(operation==2) {
                        const float weights[8]={w00*(1-c.fz),w01*(1-c.fz),w10*(1-c.fz),w11*(1-c.fz),w00*c.fz,w01*c.fz,w10*c.fz,w11*c.fz};
                        for(uint ci=0;ci<9;++ci) for(uint k=0;k<8;++k) {
                            const uint at=(ci*uint(p.L)+uint(zs[k]))*uint(p.H*p.W)+uint(ys[k]*p.W+xs[k]);
                            atomicFloatAdd(p.gradGrid,at,weights[k]*dl[ci]);
                        }
                    }
                } else for (uint ci = 0; ci < 12; ++ci) {
                    const uint si = ci % 4, di = ci / 4;
                    const float coeff = si == 3 ? 1.0f : rgb[si];
                    const uint plane = uint(p.L*p.H*p.W), base0 = ci * plane + uint(c.z0*p.H*p.W);
                    const uint base1 = ci * plane + uint(c.z1*p.H*p.W);
                    const uint r0 = uint(c.y0*p.W), r1 = uint(c.y1*p.W);
                    const float off = p.offset != 0 ? ((Ptr<float,Access.Read>)p.offset)[ci] : 0.0f;
                    const float a = w00*grid[base0+r0+uint(c.x0)] + w01*grid[base0+r0+uint(c.x1)] + w10*grid[base0+r1+uint(c.x0)] + w11*grid[base0+r1+uint(c.x1)] + off;
                    const float b = w00*grid[base1+r0+uint(c.x0)] + w01*grid[base1+r0+uint(c.x1)] + w10*grid[base1+r1+uint(c.x0)] + w11*grid[base1+r1+uint(c.x1)] + off;
                    const float v = a + (b-a)*c.fz;
                    if (si < 3) gradRgb[si] += v*grad[di];
                    gradZ += (b-a)*float(p.L-1)*coeff*grad[di];
                }
                if (p.exposureChroma == 0) {
                if (c.z0 == c.z || c.z1 == c.z) gradZ = 0.0f;
                storeRgb(p, pixel, gradRgb + float3(C2G_R,C2G_G,C2G_B)*gradZ);
                if (operation == 2) {
                    const int xs[8] = {c.x0,c.x1,c.x0,c.x1,c.x0,c.x1,c.x0,c.x1};
                    const int ys[8] = {c.y0,c.y0,c.y1,c.y1,c.y0,c.y0,c.y1,c.y1};
                    const int zs[8] = {c.z0,c.z0,c.z0,c.z0,c.z1,c.z1,c.z1,c.z1};
                    const float weights[8] = {w00*(1-c.fz),w01*(1-c.fz),w10*(1-c.fz),w11*(1-c.fz),w00*c.fz,w01*c.fz,w10*c.fz,w11*c.fz};
                    for (uint ci=0; ci<12; ++ci) {
                        const uint si=ci%4, di=ci/4; const float coeff=si==3?1.0f:rgb[si]; const float g=coeff*grad[di];
                        for(uint k=0;k<8;++k) {
                            const uint at=(ci*uint(p.L)+uint(zs[k]))*uint(p.H*p.W)+uint(ys[k]*p.W+xs[k]);
                            atomicFloatAdd(p.gradGrid, at, weights[k]*g);
                        }
                    }
                }
                }
            }
        }
        return;
    }
    if (operation == 3) {
        const uint total = uint(p.N*p.L*p.H*p.W);
        float sum=0.0f;
        for(uint idx=tid.x;idx<total;idx+=groups.x*256) {
            const int wi=int(idx%uint(p.W)), hi=int((idx/uint(p.W))%uint(p.H)), li=int((idx/uint(p.W*p.H))%uint(p.L)), ni=int(idx/uint(p.W*p.H*p.L));
            for(int ci=0;ci<p.C;++ci) {
                const uint at=uint(((ni*p.C+ci)*p.L+li)*p.H*p.W+hi*p.W+wi); const float v=((Ptr<float,Access.Read>)p.grid)[at];
                if(wi>0){const float d=v-((Ptr<float,Access.Read>)p.grid)[at-1];sum+=d*d/float(p.L*p.H*(p.W-1));}
                if(hi>0){const float d=v-((Ptr<float,Access.Read>)p.grid)[at-uint(p.W)];sum+=d*d/float(p.L*(p.H-1)*p.W);}
                if(li>0){const float d=v-((Ptr<float,Access.Read>)p.grid)[at-uint(p.W*p.H)];sum+=d*d/float((p.L-1)*p.H*p.W);}
            }
        }
        reduction[lane]=sum/float(p.C*p.normN); GroupMemoryBarrierWithGroupSync();
        for(uint step=128;step>0;step>>=1){if(lane<step)reduction[lane]+=reduction[lane+step];GroupMemoryBarrierWithGroupSync();}
        if(lane==0)((Ptr<float,Access.ReadWrite>)p.partials)[groupId.x]=reduction[0]; return;
    }
    if (operation == 4) {
        float sum=0.0f; for(uint i=lane;i<uint(p.partialCount);i+=256)sum+=((Ptr<float,Access.Read>)p.partials)[i];
        reduction[lane]=sum;GroupMemoryBarrierWithGroupSync();
        for(uint step=128;step>0;step>>=1){if(lane<step)reduction[lane]+=reduction[lane+step];GroupMemoryBarrierWithGroupSync();}
        if(lane==0)((Ptr<float,Access.ReadWrite>)p.output)[0]=reduction[0];return;
    }
    if (operation == 5) {
        const uint total=uint(p.N*p.L*p.H*p.W);
        for(uint idx=tid.x;idx<total;idx+=groups.x*256){
            const int wi=int(idx%uint(p.W)),hi=int((idx/uint(p.W))%uint(p.H)),li=int((idx/uint(p.W*p.H))%uint(p.L)),ni=int(idx/uint(p.W*p.H*p.L));
            for(int ci=0;ci<p.C;++ci){const uint at=uint(((ni*p.C+ci)*p.L+li)*p.H*p.W+hi*p.W+wi);const float v=((Ptr<float,Access.Read>)p.grid)[at];float g=0;
                if(p.W>1){const float s=2*p.gradLoss/float(p.C*p.normN*p.L*p.H*(p.W-1));if(wi>0)g+=(v-((Ptr<float,Access.Read>)p.grid)[at-1])*s;if(wi<p.W-1)g+=(v-((Ptr<float,Access.Read>)p.grid)[at+1])*s;}
                if(p.H>1){const float s=2*p.gradLoss/float(p.C*p.normN*p.L*(p.H-1)*p.W);if(hi>0)g+=(v-((Ptr<float,Access.Read>)p.grid)[at-uint(p.W)])*s;if(hi<p.H-1)g+=(v-((Ptr<float,Access.Read>)p.grid)[at+uint(p.W)])*s;}
                if(p.L>1){const float s=2*p.gradLoss/float(p.C*p.normN*(p.L-1)*p.H*p.W);if(li>0)g+=(v-((Ptr<float,Access.Read>)p.grid)[at-uint(p.W*p.H)])*s;if(li<p.L-1)g+=(v-((Ptr<float,Access.Read>)p.grid)[at+uint(p.W*p.H)])*s;}
                ((Ptr<float,Access.ReadWrite>)p.output)[at]+=g;
            }
        }return;
    }
    if(operation==6){for(uint i=tid.x;i<p.count;i+=groups.x*256){const uint spatial=uint(p.L*p.H*p.W);const uint ci=(i/spatial)%uint(p.C),ni=i/(uint(p.C)*spatial);const float m=((Ptr<float,Access.Read>)p.partials)[p.perImage!=0?ni*uint(p.C)+ci:ci];((Ptr<float,Access.ReadWrite>)p.output)[i]+=((Ptr<float,Access.Read>)p.offset)[ci]-m;}return;}
    if(operation==7){for(uint i=tid.x;i<p.count;i+=groups.x*256){const float value=((Ptr<float,Access.Read>)p.output)[i];((Ptr<float,Access.ReadWrite>)p.grid)[i]+=(((Ptr<float,Access.Read>)p.partials)[i]-((Ptr<float,Access.Read>)p.rgb)[i])*p.spatial;((Ptr<float,Access.ReadWrite>)p.offset)[i]=((Ptr<float,Access.Read>)p.gradOutput)[i]-((Ptr<float,Access.Read>)p.grid)[i]*p.invSpatial;}return;}
    if(operation==8){for(uint i=tid.x;i<p.count;i+=groups.x*256){const float g=((Ptr<float,Access.Read>)p.gradOutput)[i];const float m=p.beta1*((Ptr<float,Access.Read>)p.rgb)[i]+(1-p.beta1)*g;const float v=p.beta2*((Ptr<float,Access.Read>)p.offset)[i]+(1-p.beta2)*g*g;((Ptr<float,Access.ReadWrite>)p.rgb)[i]=m;((Ptr<float,Access.ReadWrite>)p.offset)[i]=v;((Ptr<float,Access.ReadWrite>)p.grid)[i]-=p.lr*m*p.bc1Rcp/(sqrt(v)*p.bc2SqrtRcp+p.eps);}return;}
    if(operation==9){for(uint i=tid.x;i<p.count;i+=groups.x*256){((Ptr<float,Access.ReadWrite>)p.grid)[i]*=p.scale1;((Ptr<float,Access.ReadWrite>)p.rgb)[i]*=p.scale2;}return;}
}

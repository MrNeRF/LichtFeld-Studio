#include "../../../core/include/core/rounded_math.slangh"

struct Params {
    uint64_t parameter, packed, bounds, gradient, valueBounds;
    uint64_t frozenMask, cropMask, rawScales, farMask, screenShare;
    uint64_t indices, scratch;
    uint primitives, attributes, bits, operation;
    uint slots, activeBases, valueBits, valueCells;
    uint indexCount, layout, entryCount, stage;
    uint frozenCount, cropCount, scaleCount, farCount;
    uint screenCount;
    float lr, bc1, bc2, beta1, beta2, eps;
    float frozenScale, cropScale, medianExtent, rMin;
    float rMax, screenLimit, screenPenalty, stepSize;
};
struct Push { uint64_t parameters; };

groupshared float red[4][256];
groupshared float4 newMomentBounds;
groupshared float2 newValueBounds;
groupshared float4 oldMomentBounds;
groupshared float2 oldValueBounds;

// The kernel replaces a block's bounds in place, and bounds are read through
// read-only pointers the compiler may re-load at every use (SPIRV-Cross on
// MoltenVK does). Snapshot them once in shared memory so no thread decodes
// with bounds another thread already replaced. Every thread must call it.
float4 loadOldBounds(Params p, uint block, uint lane) {
    if (lane == 0) {
        oldMomentBounds = ((Ptr<float4, Access.Read>)p.bounds)[block];
        oldValueBounds = p.valueBounds != 0 ? ((Ptr<float2, Access.Read>)p.valueBounds)[block] : float2(0);
    }
    GroupMemoryBarrierWithGroupSync();
    return oldMomentBounds;
}

float qRound(float x) { float base = floor(abs(x)); return sign(x) * (base + (abs(x) - base >= 0.5f ? 1.0f : 0.0f)); }
float qMin(float a, float b) { return isnan(a) ? b : isnan(b) ? a : min(a, b); }
float qMax(float a, float b) { return isnan(a) ? b : isnan(b) ? a : max(a, b); }
// The caller consumes the result in lane zero; every lane reaches both barriers.
float4 blockMin4(float4 value, uint lane) {
    if (WaveGetLaneCount() == 32) {
        float4 part = WaveActiveMin(value);
        if (WaveGetLaneIndex() == 0) {
            [unroll] for (uint i = 0; i < 4; ++i)
                red[i][lane / 32] = part[i];
        }
        GroupMemoryBarrierWithGroupSync();
        float4 total = lane < 8 ? float4(red[0][lane], red[1][lane], red[2][lane], red[3][lane]) : float4(1e30f);
        total = WaveActiveMin(total);
        GroupMemoryBarrierWithGroupSync();
        return total;
    }
    [unroll] for (uint i = 0; i < 4; ++i)
        red[i][lane] = value[i];
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            [unroll] for (uint i = 0; i < 4; ++i)
                red[i][lane] = qMin(red[i][lane], red[i][lane + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    return float4(red[0][0], red[1][0], red[2][0], red[3][0]);
}
uint qCode(float x, float lo, float hi, uint bits) {
    float qmax = bits == 16 ? 65535.0f : 255.0f;
    float inv = 1.0f / max(hi - lo, 1e-15f);
    return uint(clamp(qRound(roundedMul(roundedMul(qmax, roundedSub(x,lo)), inv)), 0.0f, qmax));
}
float qDecode(uint code, float lo, float hi, uint bits) {
    float qmax = bits == 16 ? 65535.0f : 255.0f;
    return fma(hi - lo, float(code) * (1.0f / qmax), lo);
}
float2 loadUS(Params p, uint64_t cell, float4 mm) {
    uint u, s;
    if (p.bits == 16) {
        Ptr<uint16_t, Access.Read> packed = (Ptr<uint16_t, Access.Read>)p.packed;
        u = packed[cell * 2]; s = packed[cell * 2 + 1];
    } else {
        Ptr<uint8_t, Access.Read> packed = (Ptr<uint8_t, Access.Read>)p.packed;
        u = packed[cell * 2]; s = packed[cell * 2 + 1];
    }
    return float2(qDecode(u, mm.x, mm.y, p.bits), qDecode(s, mm.z, mm.w, p.bits));
}
float expm1Small(float x) {
    return x * (1.0f + x * (0.5f + x * (0.1666666716f + x *
           (0.0416666679f + x * (0.0083333338f + x * 0.0013888889f)))));
}
float log1pSmall(float x) {
    float y = x / (2.0f + x), y2 = y * y;
    return 2.0f * y * (1.0f + y2 * (0.3333333433f + y2 *
           (0.2f + y2 * (0.1428571492f + y2 * (0.1111111119f + y2 * 0.0909090936f)))));
}
float2 decodeMV(float2 us) {
    float x = max(us.y, 0.0f);
    float root = x > 0.118f ? exp2(us.y * 1.4426950408889634f) - 1.0f : expm1Small(us.y);
    float r = 1e-15f * root;
    float v = r * r;
    float m = v == 0.0f ? 0.0f : us.x * (r + 1e-15f);
    return float2(m, v);
}
float2 encodeUS(float m, float v) {
    float r = sqrt(max(v, 0.0f));
    float x = max(r, 0.0f) * (1.0f / 1e-15f);
    return float2(m * (1.0f / (r + 1e-15f)), x > 0.125f ? log2(1.0f + x) * 0.6931471805599453f : log1pSmall(x));
}
void storeUS(Params p, uint64_t cell, float2 us, float4 mm) {
    uint u = qCode(us.x, mm.x, mm.y, p.bits), s = qCode(us.y, mm.z, mm.w, p.bits);
    if (p.bits == 16) {
        Ptr<uint16_t, Access.ReadWrite> packed = (Ptr<uint16_t, Access.ReadWrite>)p.packed;
        packed[cell * 2] = uint16_t(u); packed[cell * 2 + 1] = uint16_t(s);
    } else {
        Ptr<uint8_t, Access.ReadWrite> packed = (Ptr<uint8_t, Access.ReadWrite>)p.packed;
        packed[cell * 2] = uint8_t(u); packed[cell * 2 + 1] = uint8_t(s);
    }
}
uint shSlot(uint row, uint k, uint slots) { return (row / 32 * slots + k) * 32 + row % 32; }
uint shQ16Index(Params p, uint row, uint k, uint c) {
    uint cell = k * 4 + c;
    return ((row / 32 * p.valueCells + cell) * 32 + row % 32);
}
float loadShValue(Params p, uint row, uint slot, uint k, uint c, float2 bounds) {
    if (p.valueBits == 16 && p.valueBounds != 0 && p.valueCells > 0) {
        uint cell = k * 4 + c;
        if (cell >= p.valueCells) return 0.0f;
        uint code = ((Ptr<uint16_t, Access.Read>)p.parameter)[shQ16Index(p, row, k, c)];
        return bounds.x + (bounds.y - bounds.x) * (float(code) * (1.0f / 65535.0f));
    }
    if (p.valueBits == 16)
        return f16tof32(((Ptr<uint16_t, Access.Read>)p.parameter)[slot * 4 + c]);
    return ((Ptr<float, Access.Read>)p.parameter)[slot * 4 + c];
}
void storeShValue(Params p, uint row, uint slot, uint k, uint c, float value, float2 bounds) {
    if (p.valueBits == 16 && p.valueBounds != 0 && p.valueCells > 0) {
        uint cell = k * 4 + c;
        if (cell >= p.valueCells) return;
        uint code = qCode(value, bounds.x, bounds.y, 16);
        ((Ptr<uint16_t, Access.ReadWrite>)p.parameter)[shQ16Index(p, row, k, c)] = uint16_t(code);
    } else if (p.valueBits == 16) {
        ((Ptr<uint16_t, Access.ReadWrite>)p.parameter)[slot * 4 + c] = uint16_t(f32tof16(value));
    } else {
        ((Ptr<float, Access.ReadWrite>)p.parameter)[slot * 4 + c] = value;
    }
}
float rowScale(Params p, uint row, bool sh) {
    float scale = sh ? p.stepSize : p.lr * p.bc1;
    if (p.frozenMask != 0 && row < p.frozenCount && ((Ptr<uint8_t, Access.Read>)p.frozenMask)[row] != 0)
        scale *= p.frozenScale;
    if (p.cropMask != 0 && row < p.cropCount && ((Ptr<uint8_t, Access.Read>)p.cropMask)[row] != 0)
        scale *= p.cropScale;
    if (!sh && (p.entryCount & 1) != 0 && p.rawScales != 0 && p.farMask != 0 &&
        row < p.farCount && row * 3 + 2 < p.scaleCount && ((Ptr<uint8_t, Access.Read>)p.farMask)[row] != 0 && p.medianExtent > 0.0f) {
        Ptr<float, Access.Read> s = (Ptr<float, Access.Read>)p.rawScales;
        float extent = exp((s[row * 3] + s[row * 3 + 1] + s[row * 3 + 2]) / 3.0f);
        scale *= clamp(extent / p.medianExtent, p.rMin, p.rMax);
    }
    return scale;
}
float hinge(Params p, uint row, float oldV) {
    if ((p.entryCount & 2) == 0 || p.screenShare == 0 || !(p.screenLimit > 0.0f && p.screenLimit < 1.0f) || !(p.screenPenalty > 0.0f)) return 0.0f;
    if (row >= p.screenCount) return 0.0f;
    float share = ((Ptr<float, Access.Read>)p.screenShare)[row];
    if (!(share > p.screenLimit)) return 0.0f;
    float h = p.screenPenalty * log2(share / p.screenLimit);
    return h * (sqrt(oldV) * p.bc2 + p.eps);
}

[numthreads(256,1,1)]
void main(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex, uniform Push push) {
    Params p = *((Ptr<Params, Access.Read>)push.parameters);
    uint row = groupId.x * 256 + lane;
    if (p.operation == 0) {
        uint attrs = min(p.attributes, 16u);
        float4 oldmm = loadOldBounds(p, groupId.x, lane);
        float usU[16], usS[16], grad[16], m[16], v[16];
        float umin = 1e30f, umax = -1e30f, smin = 1e30f, smax = -1e30f;
        bool valid = row < p.primitives;
        bool apply = valid;
        float step = rowScale(p, row, false);
        if (valid && p.frozenMask != 0 && row < p.frozenCount && ((Ptr<uint8_t, Access.Read>)p.frozenMask)[row] != 0 && p.frozenScale == 0.0f) apply = false;
        if (valid && p.cropMask != 0 && row < p.cropCount && ((Ptr<uint8_t, Access.Read>)p.cropMask)[row] != 0 && p.cropScale == 0.0f) apply = false;
        if (valid) for (uint a=0; a<attrs; ++a) {
            uint64_t cell = uint64_t(row) * attrs + a;
            float2 mv = decodeMV(loadUS(p, cell, oldmm));
            m[a]=mv.x; v[a]=mv.y;
            float g=((Ptr<float, Access.Read>)p.gradient)[cell];
            if (p.screenShare != 0 && row < p.primitives && (p.entryCount & 2) != 0)
                g += hinge(p,row,mv.y);
            if (apply) {
                m[a] = fma(p.beta1, mv.x, (1.0f-p.beta1)*g);
                v[a] = fma(p.beta2, mv.y, roundedMul(roundedMul(1.0f-p.beta2,g),g));
                ((Ptr<float, Access.ReadWrite>)p.parameter)[cell] -= step * m[a] / (sqrt(v[a])*p.bc2+p.eps);
            }
            float2 us=encodeUS(m[a],v[a]); usU[a]=us.x; usS[a]=us.y;
            umin=qMin(umin,us.x); umax=qMax(umax,us.x); smin=qMin(smin,us.y); smax=qMax(smax,us.y);
        }
        float4 reduced = blockMin4(float4(umin, -umax, smin, -smax), lane);
        if(lane==0) {
            float4 nb=reduced.x>-reduced.y?float4(0):float4(reduced.x,-reduced.y,reduced.z,-reduced.w);
            newMomentBounds=nb; ((Ptr<float4, Access.ReadWrite>)p.bounds)[groupId.x]=nb;
        }
        GroupMemoryBarrierWithGroupSync();
        if(valid) for(uint a=0;a<attrs;++a) storeUS(p,uint64_t(row)*attrs+a,float2(usU[a],usS[a]),newMomentBounds);
        return;
    }
    if (p.operation == 2) {
        uint block=groupId.x, start=block*256;
        float4 oldmm=loadOldBounds(p, block, lane);
        bool touched=false;
        if(row<p.primitives) for(uint i=0;i<p.indexCount;++i) {
            int64_t idx=((Ptr<int64_t,Access.Read>)p.indices)[i];
            if(idx>=int64_t(start)&&idx<int64_t(start+256)) touched=true;
        }
        if(lane==0) newMomentBounds=float4(min(oldmm.x,0.0f),max(oldmm.y,0.0f),min(oldmm.z,0.0f),max(oldmm.w,0.0f));
        GroupMemoryBarrierWithGroupSync();
        if(!touched || row>=p.primitives) return;
        uint n=p.layout==0?p.attributes:p.slots*4;
        bool selected=false;
        for(uint i=0;i<p.indexCount;++i) if(((Ptr<int64_t,Access.Read>)p.indices)[i]==int64_t(row)) selected=true;
        for(uint a=0;a<n;++a) {
            uint64_t cell=p.layout==0?uint64_t(row)*n+a:uint64_t(shSlot(row,a/4,p.slots)*4+a%4);
            if(selected) storeUS(p,cell,float2(0),newMomentBounds);
            else storeUS(p,cell,loadUS(p,cell,oldmm),newMomentBounds);
        }
        if(lane==0)((Ptr<float4,Access.ReadWrite>)p.bounds)[block]=newMomentBounds;
        return;
    }

    uint slots=p.slots;
    uint activeSlots=p.activeBases>9?12u:(p.activeBases>4?6u:3u);
    uint oldIndex=groupId.x;
    float4 oldmm=loadOldBounds(p, oldIndex, lane);
    float2 oldvb=oldValueBounds;
    bool valid=row<p.primitives;
    bool apply=valid;
    if(valid&&p.frozenMask!=0&&row<p.frozenCount&&((Ptr<uint8_t,Access.Read>)p.frozenMask)[row]!=0&&p.frozenScale==0) apply=false;
    if(valid&&p.cropMask!=0&&row<p.cropCount&&((Ptr<uint8_t,Access.Read>)p.cropMask)[row]!=0&&p.cropScale==0) apply=false;
    float step=rowScale(p,row,true);
    float umin=1e30f,umax=-1e30f,smin=1e30f,smax=-1e30f,vmin=1e30f,vmax=-1e30f;
    if(valid) for(uint k=0;k<slots;++k) for(uint c=0;c<4;++c) {
        uint slot=shSlot(row,k,slots); uint64_t cell=uint64_t(slot)*4+c;
        float2 mv=decodeMV(loadUS(p,cell,oldmm));
        float value=loadShValue(p,row,slot,k,c,oldvb);
        float g=k<activeSlots?((Ptr<float,Access.Read>)p.gradient)[cell]:0.0f;
        float m=mv.x,v=mv.y;
        if(apply) {
            m=fma(p.beta1,mv.x,(1.0f-p.beta1)*g);
            v=fma(p.beta2,mv.y,roundedMul(roundedMul(1.0f-p.beta2,g),g));
            if(k<activeSlots) value-=step*m/(sqrt(v)*p.bc2+p.eps);
            if(p.valueBits==16&&p.valueBounds==0) value=f16tof32(f32tof16(value));
        }
        float2 us=encodeUS(m,v); umin=qMin(umin,us.x);umax=qMax(umax,us.x);smin=qMin(smin,us.y);smax=qMax(smax,us.y);
        if(p.valueBounds!=0&&p.valueCells>k*4+c){vmin=qMin(vmin,value);vmax=qMax(vmax,value);}
    }
    float4 reduced = blockMin4(float4(umin, -umax, smin, -smax), lane);
    if(lane==0){newMomentBounds=reduced.x>-reduced.y?float4(0):float4(reduced.x,-reduced.y,reduced.z,-reduced.w);if(p.bits==8)((Ptr<float4,Access.ReadWrite>)p.bounds)[oldIndex]=newMomentBounds;}
    if(p.valueBounds!=0&&p.valueCells>0){
        float4 values = blockMin4(float4(vmin, -vmax, 1e30f, 1e30f), lane);
        if(lane==0){newValueBounds=values.x>-values.y?float2(0):float2(values.x,-values.y);((Ptr<float2,Access.ReadWrite>)p.valueBounds)[oldIndex]=newValueBounds;}
    }
    GroupMemoryBarrierWithGroupSync();
    if(valid) for(uint k=0;k<slots;++k) for(uint c=0;c<4;++c){
        uint slot=shSlot(row,k,slots);uint64_t cell=uint64_t(slot)*4+c;float2 mv=decodeMV(loadUS(p,cell,oldmm));
        float value=loadShValue(p,row,slot,k,c,oldvb);float g=k<activeSlots?((Ptr<float,Access.Read>)p.gradient)[cell]:0.0f;
        if(apply){mv.x=fma(p.beta1,mv.x,(1-p.beta1)*g);mv.y=fma(p.beta2,mv.y,roundedMul(roundedMul(1-p.beta2,g),g));if(k<activeSlots)value-=step*mv.x/(sqrt(mv.y)*p.bc2+p.eps);}
        storeUS(p,cell,encodeUS(mv.x,mv.y),newMomentBounds);
        if(p.valueBounds!=0&&p.valueCells>k*4+c) storeShValue(p,row,slot,k,c,value,newValueBounds);
        else if(p.valueBounds==0) storeShValue(p,row,slot,k,c,value,float2(0));
    }
}

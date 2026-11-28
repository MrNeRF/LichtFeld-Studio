[SpecializationConstant] const uint operation = 0;
struct Parameters {
    uint64_t a,b,c,d,e,f,g,h,i,j,k,l;
    uint count, countA, countB, countC, countD, rows, opacityColumns, reserved;
    float minimumOpacity, limit, median, unused;
};
groupshared uint4 partial[256];
groupshared uint medianHistogram[256];
float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
uint longestAxis(float3 v) {
    float m=max(v.x,max(v.y,v.z)); return m==v.x ? 0u : (m==v.y ? 1u : 2u);
}
// Row i of `width` floats at its 64-bit byte address, its values as small indices from there: NVIDIA's Vulkan computes
// an access chain's byte offset in 32 bits without VK_EXT_shader_64bit_indexing, and a per-Gaussian buffer of 3 floats
// (means, scales, sh0) passes 4 GiB at 357.9M Gaussians, one of 4 (rotations) at 268.4M.
Ptr<float,Access.ReadWrite> floatRow(uint64_t base,uint i,uint width){return (Ptr<float,Access.ReadWrite>)(base+uint64_t(i)*width*4u);}
float logOnePlus(float x) {
    precise float y = 1.0f + x;
    return y == 1.0f ? x : log(y) * (x / (y - 1.0f));
}
[numthreads(256,1,1)]
void main(uint3 groupId:SV_GroupID,uint lane:SV_GroupIndex,uniform Parameters p) {
    uint3 groups=spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    uint index=groupId.x*256u+lane;
    if (operation == 9) {
        Ptr<float, Access.Read> values = (Ptr<float, Access.Read>)p.a;
        Ptr<uint, Access.ReadWrite> state = (Ptr<uint, Access.ReadWrite>)p.b;
        medianHistogram[lane] = 0;
        GroupMemoryBarrierWithGroupSync();
        uint prefix = state[256];
        for (uint i = index; i < p.count; i += groups.x * 256u) {
            float value = values[i];
            uint bits = asuint(value);
            if (value > 0.0f && (((bits ^ prefix) >> p.reserved) >> 8u) == 0u)
                InterlockedAdd(medianHistogram[(bits >> p.reserved) & 255u], 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (medianHistogram[lane] != 0u)
            InterlockedAdd(state[lane], medianHistogram[lane]);
        return;
    }
    if (operation == 10) {
        Ptr<uint, Access.ReadWrite> state = (Ptr<uint, Access.ReadWrite>)p.b;
        const uint count = state[lane];
        medianHistogram[lane] = count;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 1u; offset < 256u; offset <<= 1u) {
            const uint add = lane >= offset ? medianHistogram[lane - offset] : 0u;
            GroupMemoryBarrierWithGroupSync();
            medianHistogram[lane] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        const uint total = medianHistogram[255];
        const uint rank = p.reserved == 24u ? total / 2u : state[257];
        const uint prefix = p.reserved == 24u ? 0u : state[256];
        const uint valid = p.reserved == 24u ? uint(total > 0u) : state[258];
        const uint below = medianHistogram[lane] - count;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u && p.reserved == 24u)
            state[258] = valid;
        if (valid != 0u && rank >= below && rank - below < count) {
            state[256] = prefix | (lane << p.reserved);
            state[257] = rank - below;
        }
        GroupMemoryBarrierWithGroupSync();
        state[lane] = 0u;
        return;
    }
    if (operation == 4) {
        if (index < p.count) {
            Ptr<float, Access.ReadWrite> values = (Ptr<float, Access.ReadWrite>)p.a;
            Ptr<uint, Access.Read> state = (Ptr<uint, Access.Read>)p.b;
            values[index] = state[258] == 0u ? 0.0f : values[index] / max(asfloat(state[256]), 1e-9f);
        }
        return;
    }
    if(operation==0) {
        if(index>=p.count)return;
        Ptr<float,Access.ReadWrite> opacity=(Ptr<float,Access.ReadWrite>)p.e;
        Ptr<float,Access.ReadWrite> childOpacity=(Ptr<float,Access.ReadWrite>)p.j;
        uint src=uint(((Ptr<int64_t,Access.Read>)p.k)[index]);
        Ptr<float,Access.ReadWrite> means=floatRow(p.a,src,3u), rotations=floatRow(p.b,src,4u), scales=floatRow(p.c,src,3u), sh0=floatRow(p.d,src,3u);
        Ptr<float,Access.ReadWrite> childMeans=floatRow(p.f,index,3u), childRotations=floatRow(p.g,index,4u);
        Ptr<float,Access.ReadWrite> childScales=floatRow(p.h,index,3u), childSh0=floatRow(p.i,index,3u);
        float3 pos=float3(means[0],means[1],means[2]);
        float4 q=float4(rotations[0],rotations[1],rotations[2],rotations[3]);
        float3 s=float3(scales[0],scales[1],scales[2]);
        // Trained quaternions drift from unit norm; projection normalizes them.
        float inverseNorm=min(rsqrt(dot(q,q)),1e12f);
        float w=q.x*inverseNorm,x=q.y*inverseNorm,y=q.z*inverseNorm,z=q.w*inverseNorm;
        float3 cols[3]={float3(1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),
                        float3(2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),
                        float3(2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y))};
        uint axis=longestAxis(s); uint other0=(axis+1)%3, other1=(axis+2)%3;
        float3 ns = s;
        float newOpacity;
        if (p.limit > 0) {
            uint first = axis == 0 ? 1 : 0, second = axis == 2 ? 1 : 2;
            uint smallest = s[first] <= s[second] ? first : second;
            ns[axis] += log(p.limit);
            ns[smallest] += log(p.limit);
            float value = opacity[src];
            float tail = logOnePlus(exp(-abs(value)));
            float logS = -0.5f * (max(value, 0.0f) + tail);
            newOpacity = -max(-value, 0.0f) - tail - logOnePlus(exp(logS)) - logS;
        } else {
            ns[axis] += log(0.5f);
            ns[other0] += log(0.85f);
            ns[other1] += log(0.85f);
            float probability = clamp(sigmoid(opacity[src]) * 0.6f, 1e-7f, 1.0f - 1e-7f);
            newOpacity = log(probability / (1.0f - probability));
        }
        float3 offset=cols[axis]*(0.5f*exp(s[axis]));
        for(uint j=0;j<3;++j){means[j]=pos[j]+offset[j]; childMeans[j]=pos[j]-offset[j]; scales[j]=ns[j]; childScales[j]=ns[j]; childSh0[j]=sh0[j];}
        for(uint j=0;j<4;++j)childRotations[j]=q[j];
        opacity[src]=newOpacity; childOpacity[index]=newOpacity; return;
    }
    if(operation==1) {
        if(index>=p.count)return;
        int64_t t=((Ptr<int64_t,Access.Read>)p.a)[index]; if(t<0 || uint64_t(t)>=p.rows)return;
        uint dst=uint(t); Ptr<float,Access.Read> opacity=(Ptr<float,Access.Read>)p.f;
        Ptr<float,Access.ReadWrite> means=floatRow(p.b,index,3u), rotations=floatRow(p.c,index,4u), scales=floatRow(p.d,index,3u), sh=floatRow(p.e,index,3u);
        Ptr<float,Access.ReadWrite> dm=floatRow(p.g,dst,3u), dr=floatRow(p.h,dst,4u), ds=floatRow(p.i,dst,3u), dh=floatRow(p.j,dst,3u), dop=(Ptr<float,Access.ReadWrite>)p.k;
        for(uint j=0;j<3;++j){dm[j]=means[j];ds[j]=scales[j];dh[j]=sh[j];}
        for(uint j=0;j<4;++j)dr[j]=rotations[j];
        dop[dst]=opacity[index]; if(p.l!=0)((Ptr<uint8_t,Access.ReadWrite>)p.l)[dst]=0; return;
    }
    if(operation==2) {
        uint4 v=uint4(0);
        Ptr<uint8_t,Access.Read> ba=(Ptr<uint8_t,Access.Read>)p.a, bb=(Ptr<uint8_t,Access.Read>)p.b;
        Ptr<float,Access.Read> fc=(Ptr<float,Access.Read>)p.c, fd=(Ptr<float,Access.Read>)p.d;
        for(uint n=lane;n<p.countA;n+=256u)v.x+=ba[n]!=0?1u:0u;
        for(uint n=lane;n<p.countB;n+=256u)v.y+=bb[n]!=0?1u:0u;
        for(uint n=lane;n<p.countC;n+=256u)v.z+=fc[n]>0.0f?1u:0u;
        for(uint n=lane;n<p.countD;n+=256u)v.w+=fd[n]>0.0f?1u:0u;
        partial[lane]=v; GroupMemoryBarrierWithGroupSync();
        for(uint offset=128;offset>0;offset>>=1){if(lane<offset)partial[lane]+=partial[lane+offset];GroupMemoryBarrierWithGroupSync();}
        if(lane==0){Ptr<int64_t,Access.ReadWrite> out=(Ptr<int64_t,Access.ReadWrite>)p.e;out[0]=int64_t(partial[0].x);out[1]=int64_t(partial[0].y);out[2]=int64_t(partial[0].z);out[3]=int64_t(partial[0].w);} return;
    }
    if(operation==3) { if(index<p.count){Ptr<float,Access.ReadWrite> v=(Ptr<float,Access.ReadWrite>)p.a; if(isnan(v[index]))v[index]=0.0f;} return; }
    if(operation==5) {
        if(index>=p.count)return; Ptr<uint8_t,Access.Read> frozen=(Ptr<uint8_t,Access.Read>)p.a;
        if(p.countA!=0 && index<p.countA && frozen[index]!=0)return;
        Ptr<float,Access.ReadWrite> scales=floatRow(p.b,index,3u); Ptr<float,Access.Read> shares=(Ptr<float,Access.Read>)p.c; float share=shares[index];
        if(!(share>p.limit))return; float delta=min(log(share/p.limit),log(1.5f)); uint axis=longestAxis(float3(scales[0],scales[1],scales[2]));scales[axis]-=delta;return;
    }
    if(operation==6) {
        if(index>=p.count)return; Ptr<uint8_t,Access.Read> frozen=(Ptr<uint8_t,Access.Read>)p.a; Ptr<float,Access.Read> error=(Ptr<float,Access.Read>)p.b, shares=(Ptr<float,Access.Read>)p.c; Ptr<float,Access.ReadWrite> score=(Ptr<float,Access.ReadWrite>)p.d;
        score[index]=p.countA!=0 && index<p.countA && frozen[index]!=0 ? 0.0f : (!(p.limit>0.0f) || !(shares[index]>p.limit) || !(error[index]>0.0f) ? 0.0f : sqrt(error[index])*shares[index]/p.limit);return;
    }
    if(operation==7 || operation==8) {
        if(index>=p.count)return; Ptr<float,Access.ReadWrite> q=floatRow(p.b,index,4u); float norm=0.0f;for(uint j=0;j<4;++j)norm+=q[j]*q[j];
        ((Ptr<uint8_t,Access.ReadWrite>)p.c)[index]=(norm<1e-8f || (operation==7 && ((Ptr<float,Access.Read>)p.a)[index]<=p.minimumOpacity))?1:0;
    }
}

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
        if (lane == 0u) {
            if (p.reserved == 24u) {
                uint total = 0;
                for (uint bin = 0; bin < 256u; ++bin) total += state[bin];
                state[257] = total / 2u;
                state[258] = total > 0u ? 1u : 0u;
            }
            uint rank = state[257];
            if (state[258] != 0u) {
                for (uint bin = 0; bin < 256u; ++bin) {
                    uint count = state[bin];
                    if (rank < count) {
                        state[256] |= bin << p.reserved;
                        state[257] = rank;
                        break;
                    }
                    rank -= count;
                }
            }
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
        Ptr<float,Access.ReadWrite> means=(Ptr<float,Access.ReadWrite>)p.a;
        Ptr<float,Access.Read> rotations=(Ptr<float,Access.Read>)p.b;
        Ptr<float,Access.ReadWrite> scales=(Ptr<float,Access.ReadWrite>)p.c;
        Ptr<float,Access.Read> sh0=(Ptr<float,Access.Read>)p.d;
        Ptr<float,Access.ReadWrite> opacity=(Ptr<float,Access.ReadWrite>)p.e;
        Ptr<float,Access.ReadWrite> childMeans=(Ptr<float,Access.ReadWrite>)p.f;
        Ptr<float,Access.ReadWrite> childRotations=(Ptr<float,Access.ReadWrite>)p.g;
        Ptr<float,Access.ReadWrite> childScales=(Ptr<float,Access.ReadWrite>)p.h;
        Ptr<float,Access.ReadWrite> childSh0=(Ptr<float,Access.ReadWrite>)p.i;
        Ptr<float,Access.ReadWrite> childOpacity=(Ptr<float,Access.ReadWrite>)p.j;
        uint src=uint(((Ptr<int64_t,Access.Read>)p.k)[index]);
        float3 pos=float3(means[src*3],means[src*3+1],means[src*3+2]);
        float4 q=float4(rotations[src*4],rotations[src*4+1],rotations[src*4+2],rotations[src*4+3]);
        float3 s=float3(scales[src*3],scales[src*3+1],scales[src*3+2]);
        float w=q.x,x=q.y,y=q.z,z=q.w;
        float3 cols[3]={float3(1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),
                        float3(2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),
                        float3(2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y))};
        uint axis=longestAxis(s); uint other0=(axis+1)%3, other1=(axis+2)%3;
        float3 ns=s; ns[axis]+=log(0.5f); ns[other0]+=log(0.85f); ns[other1]+=log(0.85f);
        float probability=clamp(sigmoid(opacity[src])*0.6f,1e-7f,1.0f-1e-7f);
        float newOpacity=log(probability/(1.0f-probability));
        float3 offset=cols[axis]*(0.5f*exp(s[axis]));
        for(uint j=0;j<3;++j){means[src*3+j]=pos[j]+offset[j]; childMeans[index*3+j]=pos[j]-offset[j]; scales[src*3+j]=ns[j]; childScales[index*3+j]=ns[j]; childSh0[index*3+j]=sh0[src*3+j];}
        for(uint j=0;j<4;++j)childRotations[index*4+j]=q[j];
        opacity[src]=newOpacity; childOpacity[index]=newOpacity; return;
    }
    if(operation==1) {
        if(index>=p.count)return;
        int64_t t=((Ptr<int64_t,Access.Read>)p.a)[index]; if(t<0 || uint64_t(t)>=p.rows)return;
        uint dst=uint(t); Ptr<float,Access.Read> means=(Ptr<float,Access.Read>)p.b, rotations=(Ptr<float,Access.Read>)p.c, scales=(Ptr<float,Access.Read>)p.d, sh=(Ptr<float,Access.Read>)p.e, opacity=(Ptr<float,Access.Read>)p.f;
        Ptr<float,Access.ReadWrite> dm=(Ptr<float,Access.ReadWrite>)p.g, dr=(Ptr<float,Access.ReadWrite>)p.h, ds=(Ptr<float,Access.ReadWrite>)p.i, dh=(Ptr<float,Access.ReadWrite>)p.j, dop=(Ptr<float,Access.ReadWrite>)p.k;
        for(uint j=0;j<3;++j){dm[dst*3+j]=means[index*3+j];ds[dst*3+j]=scales[index*3+j];dh[dst*3+j]=sh[index*3+j];}
        for(uint j=0;j<4;++j)dr[dst*4+j]=rotations[index*4+j];
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
        Ptr<float,Access.ReadWrite> scales=(Ptr<float,Access.ReadWrite>)p.b; Ptr<float,Access.Read> shares=(Ptr<float,Access.Read>)p.c; float share=shares[index];
        if(!(share>p.limit))return; float delta=min(log(share/p.limit),log(1.5f)); uint axis=longestAxis(float3(scales[index*3],scales[index*3+1],scales[index*3+2]));scales[index*3+axis]-=delta;return;
    }
    if(operation==6) {
        if(index>=p.count)return; Ptr<uint8_t,Access.Read> frozen=(Ptr<uint8_t,Access.Read>)p.a; Ptr<float,Access.Read> error=(Ptr<float,Access.Read>)p.b, shares=(Ptr<float,Access.Read>)p.c; Ptr<float,Access.ReadWrite> score=(Ptr<float,Access.ReadWrite>)p.d;
        score[index]=p.countA!=0 && index<p.countA && frozen[index]!=0 ? 0.0f : (!(p.limit>0.0f) || !(shares[index]>p.limit) || !(error[index]>0.0f) ? 0.0f : sqrt(error[index])*shares[index]/p.limit);return;
    }
    if(operation==7 || operation==8) {
        if(index>=p.count)return; Ptr<float,Access.Read> q=(Ptr<float,Access.Read>)p.b; float norm=0.0f;for(uint j=0;j<4;++j)norm+=q[index*4+j]*q[index*4+j];
        ((Ptr<uint8_t,Access.ReadWrite>)p.c)[index]=(norm<1e-8f || (operation==7 && ((Ptr<float,Access.Read>)p.a)[index]<=p.minimumOpacity))?1:0;
    }
}

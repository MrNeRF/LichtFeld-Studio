[SpecializationConstant] const uint operation = 0;
struct Parameters {
    uint64_t a,b,c,d,e,f,g,h,i,j,k,l;
    uint64_t seed;
    uint count,countA,countB,countC,countD,width,height,channels;
    float x,y,z,w,u0,u1,scalar0,scalar1;
};
uint4 philox(uint4 c,uint2 key) {
    for(uint r=0;r<10;++r){uint64_t p0=uint64_t(0xD2511F53u)*c.x,p1=uint64_t(0xCD9E8D57u)*c.z;
        c=uint4(uint(p1>>32)^c.y^key.x,uint(p1),uint(p0>>32)^c.w^key.y,uint(p0));key+=uint2(0x9E3779B9u,0xBB67AE85u);}
    return c;
}
uint4 mrnfBlock(uint64_t seed,uint64_t row){return philox(uint4(0u,0u,uint(row),uint(row>>32)),uint2(uint(seed),uint(seed>>32)));}
float uniformCurand(uint v){return fma(float(v),2.3283064e-10f,1.1641532e-10f);}
float4 normalCurand(uint4 v){const float k=2.3283064e-10f*6.2831855f;float u0=uniformCurand(v.x),a=fma(float(v.y),k,k*0.5f),u1=uniformCurand(v.z),b=fma(float(v.w),k,k*0.5f);float s0=sqrt(-2.0f*log(u0)),s1=sqrt(-2.0f*log(u1));return float4(sin(a)*s0,cos(a)*s0,sin(b)*s1,cos(b)*s1);}
bool flagAt(uint64_t address,uint n,uint i){return address!=0 && i<n && ((Ptr<uint8_t,Access.Read>)address)[i]!=0;}
float rowReach(float peak, float logit) {
    float opacity = max(1.0f / (1.0f + exp(-logit)), 1e-6f);
    return peak >= opacity ? 1.0f : saturate(peak / opacity);
}
[numthreads(256,1,1)]
void main(uint3 groupId:SV_GroupID,uint lane:SV_GroupIndex,uniform Parameters p){
    uint3 groups=spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    uint i=groupId.x*256u+lane;
    if(operation==0){if(i>=p.count)return;if(flagAt(p.d,p.countA,i)||!(((Ptr<float,Access.Read>)p.c)[i]>0.0f))return;float logit=((Ptr<float,Access.Read>)p.b)[i];float a=pow(max(1.0f-1.0f/(1.0f+exp(-logit)),0.0f),150.0f)*p.x*p.y;if(a<1e-12f)return;float3 n=normalCurand(mrnfBlock(p.seed,uint64_t(i))).xyz;Ptr<float,Access.ReadWrite> means=(Ptr<float,Access.ReadWrite>)p.a;for(uint j=0;j<3;++j)means[i*3+j]+=clamp(a*n[j],-p.z,p.z);return;}
    if (operation == 1) {
        if (i >= p.count)
            return;
        Ptr<float, Access.ReadWrite> opacity = (Ptr<float, Access.ReadWrite>)p.a, scales = (Ptr<float, Access.ReadWrite>)p.b;
        bool frozen = flagAt(p.c, p.countA, i);
        if (frozen)
            return;
        float od = p.x, sd = p.y;
        float reach = p.e != 0 ? saturate(((Ptr<float,Access.Read>)p.e)[i]) : 1.0f;
        float t = (1.0f - p.w) * reach;
        if ((p.d == 0 || ((Ptr<float, Access.Read>)p.d)[i] > 0.0f) && (od * t != 0.0f || isinf(opacity[i])) && !isnan(opacity[i])) {
            float prob = 1.0f / (1.0f + exp(-opacity[i]));
            prob = clamp(prob - od * t, 1e-12f, asfloat(0x3f7fffffu));
            opacity[i] = log(prob / (1.0f - prob));
        }
        if (!(reach > 0.0f)) return;
        float factor = 1.0f - sd * t;
        for (uint j = 0; j < 3; ++j)
            scales[i * 3 + j] = log(max(exp(scales[i * 3 + j]) * factor, 1e-12f));
        return;
    }
    if(operation==3){if(i<p.count){Ptr<float,Access.ReadWrite> wm=(Ptr<float,Access.ReadWrite>)p.a,d=(Ptr<float,Access.ReadWrite>)p.b;wm[i]=max(wm[i],d[p.count+i]);d[p.count+i]=0.0f;}return;}
    if(operation==11){if(i>=p.count)return;Ptr<float,Access.Read> means=(Ptr<float,Access.Read>)p.a,scale=(Ptr<float,Access.Read>)p.b;Ptr<uint8_t,Access.ReadWrite> mask=(Ptr<uint8_t,Access.ReadWrite>)p.c;float maxAxis=max(abs(means[i*3]-p.x),max(abs(means[i*3+1]-p.y),abs(means[i*3+2]-p.z)));bool finiteCoord=isfinite(means[i*3])&&isfinite(means[i*3+1])&&isfinite(means[i*3+2]);mask[i]=(scale[i]>p.u0 || (finiteCoord&&maxAxis>p.w))?1u:mask[i];return;}
    if(operation==12){if(i>=p.count)return;float v=((Ptr<float,Access.Read>)p.a)[i]*(((Ptr<float,Access.Read>)p.b)[i]>0.0f?1.0f:0.0f);if(p.c!=0)v*=flagAt(p.c,p.countA,i)?1.0f:0.0f;if(p.d!=0)v*=flagAt(p.d,p.countB,i)?1.0f:0.0f;if(p.e!=0)v*=((Ptr<float,Access.Read>)p.e)[i];((Ptr<float,Access.ReadWrite>)p.f)[i]=v;return;}
    if(operation==13){if(i>=p.count)return;uint source=i;if(p.b!=0)source=uint(((Ptr<int64_t,Access.Read>)p.b)[i]);float weight=((Ptr<float,Access.Read>)p.a)[source];float key=-1.0e30f;if(weight>0.0f){uint u=mrnfBlock(p.seed,uint64_t(i)).x;float uniform=clamp(uniformCurand(u),1.0e-10f,1.0f-1.0e-7f);key=log(weight)-log(-log(uniform));}uint bits=asuint(key);uint ordered=(bits&0x80000000u)!=0u?~bits:(bits|0x80000000u);((Ptr<uint,Access.ReadWrite>)p.c)[i]=~ordered;((Ptr<uint,Access.ReadWrite>)p.d)[i]=i;return;}
    if(operation==14){if(i>=p.count)return;uint rank=p.a==0?i:uint(((Ptr<uint,Access.Read>)p.a)[i]);int64_t value=p.b==0?int64_t(rank):((Ptr<int64_t,Access.Read>)p.b)[rank];((Ptr<int64_t,Access.ReadWrite>)p.c)[i]=value;return;}
    if (i >= p.count) return;
    if (operation == 15) {
        let error = (Ptr<float, Access.ReadWrite>)p.a;
        let info = (Ptr<float, Access.ReadWrite>)p.b;
        float coverage = info[i], value = info[p.count + i];
        float compressed = value > 0 ? pow(value, 0.2f) : 0;
        error[i] += compressed;
        if (p.c && coverage > 0) ((Ptr<float, Access.ReadWrite>)p.c)[i] += 1;
        if (compressed > 0) info[3 * p.count + i] += 1;
        info[2 * p.count + i] += coverage;
        info[i] = info[p.count + i] = 0;
    } else if (operation == 16) {
        ((Ptr<float, Access.ReadWrite>)p.a)[i] /= max(((Ptr<float, Access.Read>)p.b)[3 * p.count + i], 1.0f);
    } else if (operation == 17) {
        let error = (Ptr<float, Access.ReadWrite>)p.a;
        float e = error[i];
        error[i] = e > 0 ? pow(min(e / p.x, p.y), p.z) : 0;
    } else if (operation == 18) {
        let step = (Ptr<float, Access.ReadWrite>)p.a;
        let peak = (Ptr<float, Access.ReadWrite>)p.b;
        float value = step[i];
        if (value > 0) ((Ptr<float, Access.ReadWrite>)p.c)[i] += 1;
        peak[i] = max(peak[i] * p.x, value);
        // Cleared after its last use: through MoltenVK the uses after an earlier clear saw 0.
        step[i] = 0;
    } else if (operation == 19) {
        float reach = ((Ptr<float, Access.Read>)p.b)[i] >= p.x
            ? rowReach(((Ptr<float, Access.Read>)p.a)[i], ((Ptr<float, Access.Read>)p.c)[i]) : 1.0f;
        let scales = (Ptr<float, Access.Read>)p.d;
        float largest = max(scales[3*i], max(scales[3*i + 1], scales[3*i + 2]));
        ((Ptr<float, Access.ReadWrite>)p.e)[i] = reach < p.y && largest >= p.z ? reach : 1.0f;
    } else if (operation == 20) {
        float peak = ((Ptr<float, Access.Read>)p.a)[i];
        ((Ptr<float, Access.ReadWrite>)p.c)[i] = peak < p.x ? peak - 1 : rowReach(peak, ((Ptr<float, Access.Read>)p.b)[i]);
    } else if (operation == 21) {
        uint bits = asuint(((Ptr<float, Access.Read>)p.a)[i]);
        uint key = bits == 0x80000000u ? bits : (bits & 0x80000000u ? ~bits : bits | 0x80000000u);
        if ((bits & 0x7fffffffu) > 0x7f800000u) key = 0xfffffffeu;
        ((Ptr<uint, Access.ReadWrite>)p.c)[i] = flagAt(p.b, p.count, i) ? key : 0xffffffffu;
        ((Ptr<uint, Access.ReadWrite>)p.d)[i] = i;
    }
}

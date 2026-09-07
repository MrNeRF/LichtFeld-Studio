#include "ppisp_math.slangh"
[SpecializationConstant] const uint operation = 0;

struct Parameters {
    uint64_t exposure, vignetting, color, crf, rgb, gradOutput;
    uint64_t out, gradExposure, gradVignetting, gradColor, gradCrf, gradRgb;
    uint64_t moment1, moment2, gradient, loss;
    uint count, cameras, frames, height, width, yOffset, fullHeight;
    int cameraIndex, frameIndex;
    float lr, beta1, beta2, bc1Rcp, bc2SqrtRcp, eps, center, channel, nonPositive;
    uint xOffset, fullWidth;
};
groupshared float reduceData[256];

float load(uint64_t a, uint i) { return ((Ptr<float,Access.Read>)a)[i]; }
float3 loadRgb(Parameters p,uint64_t a,uint i) { const uint n=p.height*p.width; return float3(load(a,i),load(a,n+i),load(a,2*n+i)); }
void storeRgb(Parameters p,uint64_t a,uint i,float3 v) { const uint n=p.height*p.width; Ptr<float,Access.ReadWrite> q=(Ptr<float,Access.ReadWrite>)a; q[i]=v.x;q[n+i]=v.y;q[2*n+i]=v.z; }
void atomicAdd(uint64_t a,uint i,float v) {
    Ptr<uint,Access.ReadWrite> q=(Ptr<uint,Access.ReadWrite>)a; uint old=q[i],seen;
    do { InterlockedCompareExchange(q[i],old,asuint(asfloat(old)+v),seen); if(seen==old) return; old=seen; } while(true);
}
float3 applyVig(float3 rgb,uint64_t addr,float2 pixel,float rx,float ry) {
    Ptr<float,Access.Read> v=(Ptr<float,Access.Read>)addr; const float mr=max(rx,ry);
    float2 uv=float2((pixel.x-rx*0.5f)/mr,(pixel.y-ry*0.5f)/mr); float3 out=rgb;
    for(uint ch=0;ch<3;++ch) { uint b=ch*5; float dx=uv.x-ppFinite0(v[b]),dy=uv.y-ppFinite0(v[b+1]);
        float r2=fma(dx,dx,dy*dy),r4=r2*r2,r6=r4*r2;
        float f=fma(ppFinite0(v[b+4]),r6,fma(ppFinite0(v[b+3]),r4,fma(ppFinite0(v[b+2]),r2,1.0f)));
        out[ch]*=clamp(f,0.0f,1.0f); }
    return out;
}
float ppPow(float x,float y) { return x>0.0f?exp2(y*log2(x)):0.0f; }
float3 applyCrf(float3 rgb,uint64_t addr) {
    Ptr<float,Access.Read> cr=(Ptr<float,Access.Read>)addr; rgb=clamp(rgb,0.0f,1.0f); float3 out;
    for(uint ch=0;ch<3;++ch) { uint b=ch*4; float toe=ppBounded(cr[b],0.3f),shoulder=ppBounded(cr[b+1],0.3f),gamma=ppBounded(cr[b+2],0.1f),center=ppClamped(cr[b+3]);
        float lerp=fma(shoulder-toe,center,toe),a=shoulder*center/lerp,bv=1.0f-a,x=rgb[ch];
        float y=x<=center?a*ppPow(x/center,toe):1.0f-bv*ppPow((1.0f-x)/(1.0f-center),shoulder);
        out[ch]=ppPow(max(0.0f,y),gamma); }
    return out;
}
float3 ppVigBwd(float3 rgb,uint64_t addr,float2 pixel,float rx,float ry,float3 grad,inout float g[15]) {
    Ptr<float,Access.Read> v=(Ptr<float,Access.Read>)addr;float2 uv=float2((pixel.x-rx*0.5f)/max(rx,ry),(pixel.y-ry*0.5f)/max(rx,ry));float3 gi;
    for(uint ch=0;ch<3;++ch){uint b=ch*5;float cx=ppFinite0(v[b]),cy=ppFinite0(v[b+1]),a0=ppFinite0(v[b+2]),a1=ppFinite0(v[b+3]),a2=ppFinite0(v[b+4]);
        float dx=uv.x-cx,dy=uv.y-cy,r2=fma(dx,dx,dy*dy),r4=r2*r2,r6=r4*r2;
        float f=fma(a2,r6,fma(a1,r4,fma(a0,r2,1.0f)));gi[ch]=grad[ch]*clamp(f,0.0f,1.0f);float gf=grad[ch]*rgb[ch];
        if(f>=0.0f&&f<=1.0f){g[b+2]+=gf*r2;g[b+3]+=gf*r4;g[b+4]+=gf*r6;float gr=gf*fma(3.0f*a2,r4,fma(2.0f*a1,r2,a0));g[b]-=gr*2.0f*dx;g[b+1]-=gr*2.0f*dy;}}
    return gi;
}
float ppBoundBwd(float raw,float grad){return !ppFinite(raw)||!ppFinite(grad)||raw>=32.0f?0.0f:grad*ppSigmoid(raw);}
float ppClampBwd(float raw,float grad){if(!ppFinite(raw)||!ppFinite(grad))return 0.0f;float s=ppSigmoid(raw);return s<=1e-4f||s>=1.0f-1e-4f?0.0f:grad*s*(1-s);}
float3 ppCrfBwd(float3 rgb,uint64_t addr,float3 go,inout float gc[12]) {
    Ptr<float,Access.Read> cr=(Ptr<float,Access.Read>)addr;rgb=clamp(rgb,0.0f,1.0f);float3 gi;
    for(uint ch=0;ch<3;++ch){uint b=ch*4;float toe=ppBounded(cr[b],0.3f),sh=ppBounded(cr[b+1],0.3f),gm=ppBounded(cr[b+2],0.1f),ce=ppClamped(cr[b+3]);
        float lv=fma(sh-toe,ce,toe),a=sh*ce/lv,bb=1-a,x=rgb[ch],base,y;
        if(x<=ce){base=x/ce;y=a*ppPow(base,toe);}else{base=(1-x)/(1-ce);y=1-bb*ppPow(base,sh);}
        float out=ppPow(max(y,0.0f),gm),gy=y>0?go[ch]*gm*ppPow(y,gm-1):0.0f,gx=0,gt=0,gs=0,ggm=0,gcenter=0,ga=0,gb=0;
        if(x<=ce&&ce>0&&base>0)gx=gy*a*toe*ppPow(base,toe-1)/ce;
        else if(x>ce&&ce<1&&base>0)gx=gy*bb*sh*ppPow(base,sh-1)/(1-ce);
        gi[ch]=gx;ggm=y>0?go[ch]*out*log(max(y,0.0f)+1e-8f):0.0f;
        if(x<=ce&&ce>0&&base>0){float pw=ppPow(base,toe);ga+=gy*pw;gt+=gy*a*pw*log(base+1e-8f);gcenter+=gy*a*toe*ppPow(base,toe-1)*(-x/(ce*ce));}
        else if(x>ce&&ce<1&&base>0){float pw=ppPow(base,sh);gb-=gy*pw;gs-=gy*bb*pw*log(base+1e-8f);gcenter+=gy*(-bb*sh*ppPow(base,sh-1))*((1-x)/((1-ce)*(1-ce)));}
        ga-=gb;if(abs(lv)>1e-8f){float al=sh*ce/lv;gs+=ga*ce/lv;gcenter+=ga*sh/lv;float gl=-ga*al/lv;gs+=gl*ce;gt+=gl*(1-ce);gcenter+=gl*(sh-toe);}
        gc[b]+=ppBoundBwd(cr[b],gt);gc[b+1]+=ppBoundBwd(cr[b+1],gs);gc[b+2]+=ppBoundBwd(cr[b+2],ggm);gc[b+3]+=ppClampBwd(cr[b+3],gcenter);
    } return gi;
}

[numthreads(256,1,1)]
void main(uint3 groupId:SV_GroupID,uint lane:SV_GroupIndex,uint3 tid:SV_DispatchThreadID,uniform Parameters p) {
    uint3 groups=spirv_asm { result:$$uint3=OpLoad builtin(NumWorkgroups:uint3); };
    if(operation==0){for(uint i=tid.x;i<p.height*p.width;i+=groups.x*256){const uint n=p.height*p.width;uint y=i/p.width,x=i%p.width;
        float3 rgb=loadRgb(p,p.rgb,i);float2 xy=float2(float(p.xOffset+x)+0.5f,float(p.yOffset+y)+0.5f);
        if(p.frameIndex!=-1)rgb=ppApplyExposure(rgb,load(p.exposure,uint(p.frameIndex)));
        if(p.cameraIndex!=-1)rgb=applyVig(rgb,p.vignetting+uint64_t(p.cameraIndex)*60,xy,float(p.fullWidth),float(p.fullHeight));
        if(p.frameIndex!=-1){Ptr<float,Access.Read> c=(Ptr<float,Access.Read>)p.color;uint k=uint(p.frameIndex)*8;PpColor col;col.b=float2(c[k],c[k+1]);col.r=float2(c[k+2],c[k+3]);col.g=float2(c[k+4],c[k+5]);col.n=float2(c[k+6],c[k+7]);rgb=ppApplyColor(rgb,col);}
        if(p.cameraIndex!=-1)rgb=applyCrf(rgb,p.crf+uint64_t(p.cameraIndex)*48);storeRgb(p,p.out,i,rgb);}return;}
    if(operation==1){
        // Per-thread sums over the grid-stride pixels, reduced per workgroup before
        // one atomic per parameter (as Metal does): per-pixel CAS loops on the same
        // 36 words serialized the whole image.
        float gex=0;float gv[15]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};float gc[8]={0,0,0,0,0,0,0,0};float gcr[12]={0,0,0,0,0,0,0,0,0,0,0,0};
        for(uint i=tid.x;i<p.height*p.width;i+=groups.x*256){uint n=p.height*p.width,y=i/p.width,x=i%p.width;float3 input=loadRgb(p,p.rgb,i),rgb=input,afterE=input,afterV=input,afterC=input;
            float2 xy=float2(float(x)+0.5f,float(y)+0.5f);
            if(p.frameIndex!=-1){afterE=ppApplyExposure(rgb,load(p.exposure,uint(p.frameIndex)));rgb=afterE;}
            if(p.cameraIndex!=-1){afterV=applyVig(rgb,p.vignetting+uint64_t(p.cameraIndex)*60,xy,float(p.width),float(p.height));rgb=afterV;}else afterV=rgb;
            if(p.frameIndex!=-1){Ptr<float,Access.Read> c=(Ptr<float,Access.Read>)p.color;uint k=uint(p.frameIndex)*8;PpColor col;col.b=float2(c[k],c[k+1]);col.r=float2(c[k+2],c[k+3]);col.g=float2(c[k+4],c[k+5]);col.n=float2(c[k+6],c[k+7]);afterC=ppApplyColor(rgb,col);}else afterC=rgb;
            float3 grad=loadRgb(p,p.gradOutput,i);
            if(p.cameraIndex!=-1)grad=ppCrfBwd(afterC,p.crf+uint64_t(p.cameraIndex)*48,grad,gcr);
            if(p.frameIndex!=-1){Ptr<float,Access.Read> c=(Ptr<float,Access.Read>)p.color;uint k=uint(p.frameIndex)*8;PpColor col;col.b=float2(c[k],c[k+1]);col.r=float2(c[k+2],c[k+3]);col.g=float2(c[k+4],c[k+5]);col.n=float2(c[k+6],c[k+7]);grad=ppApplyColorBwd(afterV,col,grad,gc);}
            if(p.cameraIndex!=-1)grad=ppVigBwd(afterE,p.vignetting+uint64_t(p.cameraIndex)*60,xy,float(p.width),float(p.height),grad,gv);
            if(p.frameIndex!=-1){float ge=0;grad=ppApplyExposureBwd(input,load(p.exposure,uint(p.frameIndex)),grad,ge);gex+=ge;}
            storeRgb(p,p.gradRgb,i,grad);
        }
        for(uint slot=0;slot<36;++slot){
            reduceData[lane]=slot==0?gex:slot<9?gc[slot-1]:slot<24?gv[slot-9]:gcr[slot-24];GroupMemoryBarrierWithGroupSync();
            for(uint step=128;step>0;step>>=1){if(lane<step)reduceData[lane]+=reduceData[lane+step];GroupMemoryBarrierWithGroupSync();}
            if(lane==0){float total=reduceData[0];
                if(slot<9){if(p.frameIndex!=-1)atomicAdd(slot==0?p.gradExposure:p.gradColor,slot==0?uint(p.frameIndex):uint(p.frameIndex)*8+slot-1,total);}
                else if(p.cameraIndex!=-1)atomicAdd(slot<24?p.gradVignetting:p.gradCrf,slot<24?uint(p.cameraIndex)*15+slot-9:uint(p.cameraIndex)*12+slot-24,total);}
            GroupMemoryBarrierWithGroupSync();
        }return;
    }
    if(operation==2){const uint i=tid.x;if(i>=p.count)return;Ptr<float,Access.ReadWrite> prm=(Ptr<float,Access.ReadWrite>)p.out,m1=(Ptr<float,Access.ReadWrite>)p.moment1,m2=(Ptr<float,Access.ReadWrite>)p.moment2;Ptr<float,Access.Read> gr=(Ptr<float,Access.Read>)p.gradient;
        float old=prm[i],g=gr[i],m=m1[i],v=m2[i];bool valid=ppFinite(old);float par=valid?old:0.0f;g=valid&&ppFinite(g)?g:0;m=valid&&ppFinite(m)?m:0;v=valid&&ppFinite(v)&&v>=0?v:0;m=p.beta1*m+(1-p.beta1)*g;v=p.beta2*v+(1-p.beta2)*g*g;float upd=par-p.lr*m*p.bc1Rcp/(sqrt(v)*p.bc2SqrtRcp+p.eps);bool ok=ppFinite(upd)&&ppFinite(m)&&ppFinite(v);prm[i]=ok?upd:par;m1[i]=ok?m:0;m2[i]=ok?v:0;return;}
    if(operation==3){float local=0.0f;const uint cam=tid.x;if(cam<p.cameras){Ptr<float,Access.Read> v=(Ptr<float,Access.Read>)p.vignetting;Ptr<float,Access.ReadWrite> g=(Ptr<float,Access.ReadWrite>)p.gradVignetting;uint base=cam*15;
        float ic=1.0f/float(p.cameras*3),inp=1.0f/float(p.cameras*9),ich=1.0f/float(p.cameras*15);
        if(p.center!=0){float s=p.center*2*ic;for(uint ch=0;ch<3;++ch){float cx=v[base+ch*5],cy=v[base+ch*5+1];local+=p.center*(cx*cx+cy*cy)*ic;if(p.gradVignetting!=0){g[base+ch*5]+=s*cx;g[base+ch*5+1]+=s*cy;}}}
        if(p.nonPositive!=0)for(uint ch=0;ch<3;++ch)for(uint a=0;a<3;++a){uint k=base+ch*5+2+a;float x=v[k];if(x>0){local+=p.nonPositive*x*inp;if(p.gradVignetting!=0)g[k]+=p.nonPositive*inp;}}
        if(p.channel!=0)for(uint k=0;k<5;++k){float a=v[base+k],b=v[base+5+k],c=v[base+10+k],mean=(a+b+c)/3;float vals[3]={a,b,c};for(uint ch=0;ch<3;++ch){float d=vals[ch]-mean;local+=p.channel*d*d*ich;if(p.gradVignetting!=0)g[base+ch*5+k]+=p.channel*2*ich*d;}}
    } if(p.loss!=0&&local!=0)atomicAdd(p.loss,0,local);return;}
    if(operation==4){const uint f=p.frames;Ptr<float,Access.ReadWrite> e=(Ptr<float,Access.ReadWrite>)p.exposure,c=(Ptr<float,Access.ReadWrite>)p.color;float sum=0;
        if(groupId.x==0){for(uint i=lane;i<f;i+=256)sum+=e[i];}
        else {uint ch=groupId.x-1;for(uint i=lane;i<f;i+=256)sum+=c[i*8+ch];}
        reduceData[lane]=sum;GroupMemoryBarrierWithGroupSync();for(uint step=128;step>0;step>>=1){if(lane<step)reduceData[lane]+=reduceData[lane+step];GroupMemoryBarrierWithGroupSync();}
        const float mean=reduceData[0]/float(f);
        if(groupId.x==0){for(uint i=lane;i<f;i+=256)e[i]-=mean;}
        else {uint ch=groupId.x-1;for(uint i=lane;i<f;i+=256)c[i*8+ch]-=mean;}return;}
    if(operation==5){for(uint i=tid.x;i<p.count;i+=groups.x*256){if(i<p.frames)((Ptr<float,Access.ReadWrite>)p.exposure)[i]=0;if(i<p.cameras*15)((Ptr<float,Access.ReadWrite>)p.vignetting)[i]=0;if(i<p.frames*8)((Ptr<float,Access.ReadWrite>)p.color)[i]=0;if(i<p.cameras*12){uint k=i%4;
        // CUDA's fast exp/log inverse values for the identity curve, kept byte-exact.
        ((Ptr<float,Access.ReadWrite>)p.crf)[i]=k==0||k==1?asfloat(0x3c5fc9f4u):k==2?asfloat(0x3ec19ec3u):0.0f;}}return;}
}

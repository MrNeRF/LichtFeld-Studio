// Exact grid k=1 queries and batched pinhole visibility.
static const uint LFS_LOCAL_SIZE_X=256;
static const uint LFS_PUSH_CONSTANT_SIZE=64;
[SpecializationConstant] const uint kMode=0;
struct Parameters { uint64_t queries,targets,heads,next,output; uint nq,nt,mask; float width,maximum; uint padding; };
float3 loadPoint(uint64_t address,uint i) {
    Ptr<float,Access.Read> data=(Ptr<float,Access.Read>)address;
    return float3(data[3*i],data[3*i+1],data[3*i+2]);
}
int3 gridCell(float3 p,float width) { return int3(clamp(floor(p/width),-268435456.0f,268435456.0f)); }
uint gridHash(int3 c,uint mask) { return ((uint(c.x)*73856093u)^(uint(c.y)*19349663u)^(uint(c.z)*83492791u))&mask; }
int nearest(Parameters p,float3 point) {
    if(!all(isfinite(point))) return -1;
    Ptr<int,Access.Read> heads=(Ptr<int,Access.Read>)p.heads,next=(Ptr<int,Access.Read>)p.next;
    int3 centre=gridCell(point,p.width);
    float best=1.0f/0.0f; int index=-1;
    for(int shell=0;shell<8;++shell) {
        for(int z=-shell;z<=shell;++z) for(int y=-shell;y<=shell;++y) for(int x=-shell;x<=shell;++x) {
            if(shell!=0 && abs(x)!=shell && abs(y)!=shell && abs(z)!=shell) continue;
            int3 target=centre+int3(x,y,z);
            for(int j=heads[gridHash(target,p.mask)];j>=0;j=next[j]) {
                float3 q=loadPoint(p.targets,uint(j));
                if(any(gridCell(q,p.width)!=target)) continue;
                float3 delta=point-q; float d=dot(delta,delta);
                if(d<best || (d==best && (index<0 || j<index))) { best=d;index=j; }
            }
        }
        float3 lower=point-float3(centre-shell)*p.width;
        float3 upper=float3(centre+shell+1)*p.width-point;
        float3 bounds=min(lower,upper); float bound=min(bounds.x,min(bounds.y,bounds.z));
        if(index>=0 && bound>0 && best<bound*bound) return index;
    }
    for(uint j=0;j<p.nt;++j) {
        float3 delta=point-loadPoint(p.targets,j); float d=dot(delta,delta);
        if(d<best || (d==best && (index<0 || int(j)<index))) { best=d;index=int(j); }
    }
    return index;
}
int coverage(Parameters p,float3 point) {
    Ptr<float,Access.Read> cameras=(Ptr<float,Access.Read>)p.targets;
    int count=0;
    for(uint c=0;c<p.nt;++c) {
        uint b=16*c;
        float x=dot(float3(cameras[b],cameras[b+1],cameras[b+2]),point)+cameras[b+3];
        float y=dot(float3(cameras[b+4],cameras[b+5],cameras[b+6]),point)+cameras[b+7];
        float z=dot(float3(cameras[b+8],cameras[b+9],cameras[b+10]),point)+cameras[b+11];
        float3 delta=point-float3(cameras[b+12],cameras[b+13],cameras[b+14]);
        count+=int(z>1e-6f && x>=0 && y>=0 && x<=z && y<=z && (p.maximum<=0 || dot(delta,delta)<=p.maximum*p.maximum));
    }
    return count;
}
[numthreads(LFS_LOCAL_SIZE_X,1,1)]
void main(uint3 tid:SV_DispatchThreadID,uniform Parameters p) {
    uint3 groups=spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    for(uint i=tid.x;i<(kMode==0?p.nt:p.nq);i+=groups.x*LFS_LOCAL_SIZE_X) {
        if(kMode==0) {
            float3 q=loadPoint(p.targets,i); if(!all(isfinite(q))) continue;
            uint bucket=gridHash(gridCell(q,p.width),p.mask);
            int previous;
            InterlockedExchange(((Ptr<int,Access.ReadWrite>)p.heads)[bucket],int(i),previous);
            ((Ptr<int,Access.ReadWrite>)p.next)[i]=previous;
        } else ((Ptr<int,Access.ReadWrite>)p.output)[i]=kMode==1?nearest(p,loadPoint(p.queries,i)):coverage(p,loadPoint(p.queries,i));
    }
}

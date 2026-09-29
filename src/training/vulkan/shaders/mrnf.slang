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
[numthreads(256,1,1)]
void main(uint3 groupId:SV_GroupID,uint lane:SV_GroupIndex,uniform Parameters p){
    uint3 groups=spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    uint i=groupId.x*256u+lane;
    if(operation==0){if(i>=p.count)return;if(flagAt(p.d,p.countA,i)||!(((Ptr<float,Access.Read>)p.c)[i]>0.0f))return;float logit=((Ptr<float,Access.Read>)p.b)[i];float a=pow(max(1.0f-1.0f/(1.0f+exp(-logit)),0.0f),150.0f)*p.x*p.y;if(a<1e-12f)return;float3 n=normalCurand(mrnfBlock(p.seed,uint64_t(i))).xyz;Ptr<float,Access.ReadWrite> means=(Ptr<float,Access.ReadWrite>)p.a;for(uint j=0;j<3;++j)means[i*3+j]+=clamp(a*n[j],-p.z,p.z);return;}
    if(operation==1){if(i>=p.count)return;Ptr<float,Access.ReadWrite> opacity=(Ptr<float,Access.ReadWrite>)p.a,scales=(Ptr<float,Access.ReadWrite>)p.b;bool frozen=flagAt(p.c,p.countA,i),far=flagAt(p.d,p.countB,i);if(frozen)return;float od=p.x,sd=p.y;if(far){od*=p.z;sd*=p.z;}float t=1.0f-p.w;if((od!=0.0f || isinf(opacity[i]))&&!isnan(opacity[i])){float prob=1.0f/(1.0f+exp(-opacity[i]));prob=clamp(prob-od*t,1e-12f,asfloat(0x3f7fffffu));opacity[i]=log(prob/(1.0f-prob));}float factor=1.0f-sd*t;for(uint j=0;j<3;++j)scales[i*3+j]=log(max(exp(scales[i*3+j])*factor,1e-12f));return;}
    if(operation==2){if(i>=p.count)return;Ptr<float,Access.ReadWrite> vis=(Ptr<float,Access.ReadWrite>)p.a,wm=(Ptr<float,Access.ReadWrite>)p.b,dens=(Ptr<float,Access.ReadWrite>)p.c;float v=vis[i]+dens[i];float e=dens[p.count+i];vis[i]=v;wm[i]=max(wm[i],e);dens[i]=0.0f;dens[p.count+i]=0.0f;if(p.d!=0){Ptr<float,Access.ReadWrite> ratio=(Ptr<float,Access.ReadWrite>)p.d;float r=0.0f;if(v>=0.05f)r=p.y>0.0f?e/pow(v,p.y):e/v;ratio[i]=max(ratio[i],r);}return;}
    if(operation==3){if(i<p.count){Ptr<float,Access.ReadWrite> wm=(Ptr<float,Access.ReadWrite>)p.a,d=(Ptr<float,Access.ReadWrite>)p.b;wm[i]=max(wm[i],d[p.count+i]);d[p.count+i]=0.0f;}return;}
    if(operation==4){if(i>=p.count)return;Ptr<float,Access.Read> means=(Ptr<float,Access.Read>)p.a,view=(Ptr<float,Access.Read>)p.b;Ptr<float,Access.ReadWrite> out=(Ptr<float,Access.ReadWrite>)p.c,visible=(Ptr<float,Access.ReadWrite>)p.d;float x=means[i*3],y=means[i*3+1],z=means[i*3+2];float cx=view[0]*x+view[1]*y+view[2]*z+view[3],cy=view[4]*x+view[5]*y+view[6]*z+view[7],cz=view[8]*x+view[9]*y+view[10]*z+view[11];bool ok=isfinite(cx)&&isfinite(cy)&&isfinite(cz)&&cz>p.x;if(!ok){out[i*2]=0;out[i*2+1]=0;visible[i]=0;return;}float px=p.y*(cx/cz)+p.u0,py=p.z*(cy/cz)+p.u1;out[i*2]=px;out[i*2+1]=py;visible[i]=(px>=0.0f&&px<float(p.width)&&py>=0.0f&&py<float(p.height))?1.0f:0.0f;return;}
    if(operation==5){if(i>=p.count)return;Ptr<float,Access.Read> xy=(Ptr<float,Access.Read>)p.a,vis=(Ptr<float,Access.Read>)p.b,err=(Ptr<float,Access.Read>)p.c;int width=int(p.width),height=int(p.height);if(!(vis[i]>0.0f)){((Ptr<float,Access.ReadWrite>)p.d)[i]=0;return;}int x=clamp(int(floor(xy[i*2])),0,width-1),y=clamp(int(floor(xy[i*2+1])),0,height-1);((Ptr<float,Access.ReadWrite>)p.d)[i]=err[y*width+x];return;}
    if(operation==6){if(i>=p.count)return;Ptr<float,Access.Read> m=(Ptr<float,Access.Read>)p.a;float dx=m[i*3]-p.x,dy=m[i*3+1]-p.y,dz=m[i*3+2]-p.z;((Ptr<uint8_t,Access.ReadWrite>)p.b)[i]=(dx*dx+dy*dy+dz*dz>p.w)?1u:0u;return;}
    if(operation==7){if(i>=p.count)return;Ptr<float,Access.Read> a=(Ptr<float,Access.Read>)p.a,b=(Ptr<float,Access.Read>)p.b;float sum=0;for(uint c=0;c<p.channels;++c)sum+=abs(a[c*p.count+i]-b[c*p.count+i]);((Ptr<float,Access.ReadWrite>)p.c)[i]=sum/float(p.channels);return;}
    if(operation==8){if(i<p.count)((Ptr<float,Access.ReadWrite>)p.c)[i]=((Ptr<float,Access.Read>)p.a)[i]*(1.0f-((Ptr<float,Access.Read>)p.b)[i]);return;}
    if(operation==9){if(i>=p.count)return;Ptr<int64_t,Access.Read> idx=(Ptr<int64_t,Access.Read>)p.a;Ptr<float,Access.Read> target=(Ptr<float,Access.Read>)p.b,alpha=(Ptr<float,Access.Read>)p.c;Ptr<float,Access.ReadWrite> rgb=(Ptr<float,Access.ReadWrite>)p.e,ao=(Ptr<float,Access.ReadWrite>)p.f,dep=(Ptr<float,Access.ReadWrite>)p.g;int64_t raw=idx[i];uint pixel=(raw>=0&&uint64_t(raw)<p.countA)?uint(raw):0u;for(uint c=0;c<3;++c)rgb[i*3+c]=target[min(c,p.channels-1)*p.countA+pixel];ao[i]=alpha[pixel];dep[i]=p.d==0?0.0f:((Ptr<float,Access.Read>)p.d)[pixel];return;}
    if(operation==10){if(i>=p.count)return;Ptr<float,Access.ReadWrite> weights=(Ptr<float,Access.ReadWrite>)p.a;float vis=((Ptr<float,Access.Read>)p.b)[i];if(vis==0.0f){weights[i]=0.0f;return;}weights[i]*=0.0026f+pow(clamp(1.0f-vis/max(p.x,1.19209290e-07f),0.0f,1.0f),1.72f);return;}
    if(operation==11){if(i>=p.count)return;Ptr<float,Access.Read> means=(Ptr<float,Access.Read>)p.a,scale=(Ptr<float,Access.Read>)p.b;Ptr<uint8_t,Access.ReadWrite> mask=(Ptr<uint8_t,Access.ReadWrite>)p.c;float maxAxis=max(abs(means[i*3]-p.x),max(abs(means[i*3+1]-p.y),abs(means[i*3+2]-p.z)));bool finiteCoord=isfinite(means[i*3])&&isfinite(means[i*3+1])&&isfinite(means[i*3+2]);mask[i]=(scale[i]>p.u0 || (finiteCoord&&maxAxis>p.w))?1u:mask[i];return;}
    if(operation==12){if(i>=p.count)return;float v=((Ptr<float,Access.Read>)p.a)[i]*(((Ptr<float,Access.Read>)p.b)[i]>0.0f?1.0f:0.0f);if(p.c!=0)v*=flagAt(p.c,p.countA,i)?1.0f:0.0f;if(p.d!=0)v*=flagAt(p.d,p.countB,i)?1.0f:0.0f;if(p.e!=0)v*=((Ptr<float,Access.Read>)p.e)[i];((Ptr<float,Access.ReadWrite>)p.f)[i]=v;return;}
    if(operation==13){if(i>=p.count)return;uint source=i;if(p.b!=0)source=uint(((Ptr<int64_t,Access.Read>)p.b)[i]);float weight=((Ptr<float,Access.Read>)p.a)[source];float key=-1.0e30f;if(weight>0.0f){uint u=mrnfBlock(p.seed,uint64_t(i)).x;float uniform=clamp(uniformCurand(u),1.0e-10f,1.0f-1.0e-7f);key=log(weight)-log(-log(uniform));}uint bits=asuint(key);uint ordered=(bits&0x80000000u)!=0u?~bits:(bits|0x80000000u);((Ptr<uint,Access.ReadWrite>)p.c)[i]=~ordered;((Ptr<uint,Access.ReadWrite>)p.d)[i]=i;return;}
    if(operation==14){if(i>=p.count)return;uint rank=p.a==0?i:uint(((Ptr<uint,Access.Read>)p.a)[i]);int64_t value=p.b==0?int64_t(rank):((Ptr<int64_t,Access.Read>)p.b)[rank];((Ptr<int64_t,Access.ReadWrite>)p.c)[i]=value;return;}
    if(operation==15){if(i>=p.count)return;Ptr<float,Access.Read> scales=(Ptr<float,Access.Read>)p.a;float value=exp((scales[i*3]+scales[i*3+1]+scales[i*3+2])*(1.0f/3.0f));((Ptr<float,Access.ReadWrite>)p.b)[i]=isfinite(value)&&value>0.0f?value:0.0f;return;}
}

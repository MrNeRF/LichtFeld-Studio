// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
static const uint LFS_LOCAL_SIZE_X=256;
static const uint LFS_PUSH_CONSTANT_SIZE=128;
static const float pi=3.14159265358979323846f;
[SpecializationConstant] const uint panorama=0;
struct Parameters {
    uint64_t rgb, alpha, environment, output;
    float rotation[9];
    int full_width,full_height,band_width,band_height,y_offset;
    float focal_x,focal_y,center_x,center_y;
    int equirect_view;
    float exposure_factor,env_rotation_radians;
    int env_width,env_height;
    uint padding;
};
float3 normalized(float3 v) {const float len=sqrt(v.x*v.x+v.y*v.y+v.z*v.z);return len<=0?v:v*(1.0f/len);}
float3 blend(float3 a,float3 b,float t) {return a+(b-a)*t;}
float3 fetch(Parameters p,int x,int y) {
    Ptr<float,Access.Read> env=(Ptr<float,Access.Read>)p.environment;
    const uint i=(y*p.env_width+x)*3;
    return float3(env[i],env[i+1],env[i+2]);
}
uint3 pixel(Parameters p,uint idx) {
    const int x=int(idx)%p.band_width,y=int(idx)/p.band_width+p.y_offset;
    const float u=(float(x)+0.5f)/p.full_width, v=1.0f-(float(y)+0.5f)/p.full_height;
    float3 local;
    if(panorama!=0) {
        const float lon=(u-0.5f)*(2.0f*pi),lat=(v-0.5f)*pi,c=cos(lat);
        local=normalized(float3(sin(lon)*c,sin(lat),-cos(lon)*c));
    } else local=normalized(float3((u*p.full_width-p.center_x)/max(p.focal_x,1e-6f),
                                   (v*p.full_height-p.center_y)/max(p.focal_y,1e-6f),-1));
    float3 dir=normalized(float3(p.rotation[0]*local.x+p.rotation[3]*local.y+p.rotation[6]*local.z,
        p.rotation[1]*local.x+p.rotation[4]*local.y+p.rotation[7]*local.z,
        p.rotation[2]*local.x+p.rotation[5]*local.y+p.rotation[8]*local.z));
    const float c=cos(p.env_rotation_radians),s=sin(p.env_rotation_radians);
    dir=normalized(float3(c*dir.x+s*dir.z,dir.y,-s*dir.x+c*dir.z));
    float eu=atan2(dir.x,-dir.z)/(2.0f*pi)+0.5f;
    const float ev=clamp(0.5f-asin(clamp(dir.y,-1.0f,1.0f))/pi,0.0f,1.0f);
    eu-=floor(eu);
    const float ex=eu*(p.env_width-1),ey=ev*(p.env_height-1);
    const int x0=clamp(int(floor(ex)),0,p.env_width-1),y0=clamp(int(floor(ey)),0,p.env_height-1);
    const int x1=(x0+1)%p.env_width,y1=min(y0+1,p.env_height-1);
    float3 hdr=blend(blend(fetch(p,x0,y0),fetch(p,x1,y0),ex-x0),blend(fetch(p,x0,y1),fetch(p,x1,y1),ex-x0),ey-y0)*p.exposure_factor;
    float3 bg=clamp(hdr*(2.51f*hdr+0.03f)/(hdr*(2.43f*hdr+0.59f)+0.14f),0.0f,1.0f);
    bg=clamp(pow(bg,float3(1.0f/2.2f)),0.0f,1.0f);
    const int n=p.band_width*p.band_height;
    Ptr<float,Access.Read> rgb=(Ptr<float,Access.Read>)p.rgb;
    const float a=((Ptr<float,Access.Read>)p.alpha)[idx];
    return uint3(clamp(blend(bg,float3(rgb[idx],rgb[n+idx],rgb[2*n+idx]),a),0.0f,1.0f)*255.0f+0.5f);
}
[numthreads(LFS_LOCAL_SIZE_X,1,1)]
void main(uint3 tid:SV_DispatchThreadID,uniform Parameters p) {
    const uint3 groups=spirv_asm {result:$$uint3=OpLoad builtin(NumWorkgroups:uint3);};
    const uint count=p.band_width*p.band_height;
    Ptr<uint,Access.ReadWrite> dst=(Ptr<uint,Access.ReadWrite>)p.output;
    for(uint base=tid.x*4;base<count;base+=groups.x*LFS_LOCAL_SIZE_X*4) {
        uint bytes[12];
        for(uint i=0;i<4;++i) {
            const uint3 c=base+i<count?pixel(p,base+i):uint3(0);
            bytes[3*i]=c.x;bytes[3*i+1]=c.y;bytes[3*i+2]=c.z;
        }
        for(uint j=0;j<3 && base*3+j*4<count*3;++j)
            dst[base*3/4+j]=bytes[j*4]|(bytes[j*4+1]<<8)|(bytes[j*4+2]<<16)|(bytes[j*4+3]<<24);
    }
}

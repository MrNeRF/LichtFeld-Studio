/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: GPL-3.0-or-later AND Apache-2.0 */

static const uint LFS_LOCAL_SIZE_X=256;
static const uint LFS_PUSH_CONSTANT_SIZE=32;
struct Parameters {uint64_t input,output,parameters;int width,height;};
struct PpispParams {
    float exposure_factor;
    float vignetting[15];
    float color_matrix[9];
    float crf[15];
    int y_offset,full_height;
};
[numthreads(LFS_LOCAL_SIZE_X,1,1)]
void main(uint3 tid:SV_DispatchThreadID,uniform Parameters args) {
    const uint3 groups=spirv_asm {result:$$uint3=OpLoad builtin(NumWorkgroups:uint3);};
    const PpispParams p=((Ptr<PpispParams,Access.Read>)args.parameters)[0];
    Ptr<float,Access.Read> src=(Ptr<float,Access.Read>)args.input;
    Ptr<float,Access.ReadWrite> dst=(Ptr<float,Access.ReadWrite>)args.output;
    const int width=args.width,height=args.height,plane=width*height;
    const int full_height=p.full_height>0?p.full_height:height;
    const float resolution=float(max(width,full_height));
    for(uint i=tid.x;i<uint(plane);i+=groups.x*LFS_LOCAL_SIZE_X) {
        const float x=(float(i%width)+0.5f-width*0.5f)/resolution;
        const float y=(float(i/width+p.y_offset)+0.5f-full_height*0.5f)/resolution;
        float3 rgb;
        for(int c=0;c<3;++c) {
            const int k=c*5;
            const float dx=x-p.vignetting[k],dy=y-p.vignetting[k+1],r2=fma(dx,dx,dy*dy),r4=r2*r2,r6=r4*r2;
            const float falloff=max(0.0f,min(1.0f,fma(p.vignetting[k+4],r6,fma(p.vignetting[k+3],r4,fma(p.vignetting[k+2],r2,1.0f)))));
            rgb[c]=max((src[c*plane+i]*p.exposure_factor)*falloff,0.0f);
        }
        const float intensity=rgb.x+rgb.y+rgb.z;
        float3 rgi;
        for(int c=0;c<3;++c) {
            const int k=c*3;
            rgi[c]=fma(p.color_matrix[k],rgb.x,fma(p.color_matrix[k+1],rgb.y,p.color_matrix[k+2]*intensity));
        }
        const float norm=intensity/(max(rgi.z,0.0f)+1e-5f);
        rgb.x=rgi.x*norm;rgb.y=rgi.y*norm;rgb.z=rgi.z*norm-rgb.x-rgb.y;
        for(int c=0;c<3;++c) {
            const int k=c*5;
            const float value=clamp(rgb[c],0.0f,1.0f),mid=p.crf[k+3],a=p.crf[k+4];
            const float curve=value<=mid?a*pow(value/mid,p.crf[k]):1.0f-(1.0f-a)*pow((1.0f-value)/(1.0f-mid),p.crf[k+1]);
            dst[c*plane+i]=pow(max(0.0f,curve),p.crf[k+2]);
        }
    }
}

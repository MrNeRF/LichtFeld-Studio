// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include <metal_stdlib>
using namespace metal;
#define LFS_COLOR_INLINE inline
#define LFS_COLOR_VEC3 float3
#define LFS_COLOR_UINT uint
#define LFS_COLOR_MIX mix
// Shared desktop tone curves are inserted here at configure time.
@LFS_METAL_DISPLAY_COLOR@
float normalized_depth(float depth,float lo,float hi) {
    lo=max(lo,1e-4f); hi=max(hi,lo+1e-4f); depth=clamp(depth,lo,hi);
    const float linear=clamp((depth-lo)/max(hi-lo,1e-5f),0.0f,1.0f);
    const float logarithmic=clamp(log2(depth/lo)/max(log2(hi/lo),1e-4f),0.0f,1.0f);
    return smoothstep(0.0f,1.0f,mix(linear,logarithmic,smoothstep(1.75f,24.0f,hi/lo)));
}
struct PointParameters {
    float4x4 view_projection,view,crop_to_local;
    float4 crop_min,crop_max,voxel_focal_ortho;
    uint4 counts;
};
struct PointObject {float4x4 transform;float4 camera;uint4 flags;};
struct PointVertex {
    float4 position [[position]];
    float point_size [[point_size]];
    float3 color;
    float depth;
};
vertex PointVertex point_vertex(uint source [[vertex_id]],
    device const packed_float3* positions [[buffer(0)]],device const packed_float3* colors [[buffer(1)]],
    device const PointObject* objects [[buffer(2)]],device const int* indices [[buffer(3)]],
    device const uchar* selection [[buffer(4)]],device const uchar* preview [[buffer(5)]],
    device const float4* palette [[buffer(6)]],device const uchar* deleted [[buffer(7)]],
    constant PointParameters& p [[buffer(8)]]){
    PointVertex out{float4(2,2,2,1),0,float3(0),0};
    const uint flags=p.counts.z;
    if((flags&512u)&&deleted[source])return out;
    float3 world=float3(positions[source]);
    if(p.counts.x){
        const int node=(flags&16u)?clamp(indices[source],0,int(p.counts.x)-1):0;
        if(!objects[node].flags.x)return out;
        const float4 transformed=objects[node].transform*float4(world,1);
        world=abs(transformed.w)>1e-6f?transformed.xyz/transformed.w:transformed.xyz;
    }
    bool desaturate=false;
    if(flags&1u){
        const float3 local=(p.crop_to_local*float4(world,1)).xyz;
        const float3 radius=max(abs(p.crop_min.xyz),1e-8f);
        const bool inside=(flags&1024u)?dot(local/radius,local/radius)<=1.f:
            all(local>=p.crop_min.xyz)&&all(local<=p.crop_max.xyz);
        const bool visible=(flags&2u)?!inside:inside;
        if(!visible){if(!(flags&4u))return out;desaturate=true;}
    }
    const float4 view=p.view*float4(world,1);
    float4 clip=p.view_projection*float4(world,1);
    if(abs(clip.w)<=1e-6f || (!(flags&8u)&&-view.z<1e-4f))return out;
    const float voxel=max(p.voxel_focal_ortho.x,1e-5f);
    const float radius=max(1.f,ceil((flags&8u)?voxel*max(p.voxel_focal_ortho.z,1e-5f)*.5f:
        voxel*max(p.voxel_focal_ortho.y,1.f)/max(-view.z,1e-4f)));
    const float max_size=float(p.counts.w);
    float diameter=clamp(2*radius,1.f,max_size);
    float3 color=clamp(float3(colors[source]),0.f,1.f);
    if(desaturate)color=mix(color,float3(dot(color,float3(.299f,.587f,.114f))),.75f);
    const uint group=(flags&32u)?selection[source]:0;
    const bool in_preview=(flags&64u)&&preview[source];
    const bool highlight=(in_preview&&!group&&(flags&128u))||(in_preview&&group&&!(flags&128u));
    if(highlight){color=mix(color,palette[256].xyz,.9f);diameter=clamp(max(diameter+2,diameter*1.6f),1.f,max_size);}
    else if(group){color=mix(color,palette[group].xyz,.75f);diameter=clamp(max(diameter+2,diameter*1.35f),1.f,max_size);}
    // The shared request has Vulkan Y-down clip coordinates and OpenGL depth.
    clip.y=-clip.y; clip.z=.5f*(clip.z+clip.w);
    return {clip,diameter,color,-view.z};
}
struct PointFragment {float4 color [[color(0)]];float depth [[color(1)]];};
float3 depth_palette(float t);
fragment PointFragment point_fragment(PointVertex in [[stage_in]],float2 coord [[point_coord]],
    constant PointParameters& p [[buffer(0)]]){
    const float2 delta=coord*2-1;
    if(dot(delta,delta)>1)discard_fragment();
    float3 color=in.color;
    if(p.voxel_focal_ortho.w!=0){
        const float hi=p.crop_max.w<=p.crop_min.w+1e-5f?p.crop_min.w+1:p.crop_max.w;
        const float t=1-normalized_depth(in.depth,p.crop_min.w,hi);
        color=(p.counts.z&256u)?float3(t):depth_palette(t);
    }
    return {float4(color,1),in.depth};
}
float3 depth_palette(float t) {
    t=clamp(t,0.0f,1.0f);
    const float3 far0={.050f,.040f,.150f},far1={.060f,.195f,.500f};
    const float3 mid0={0,.500f,.650f},mid1={.360f,.735f,.410f};
    const float3 near0={.965f,.820f,.300f},near1={.985f,.430f,.125f};
    if(t<.20f) return mix(far0,far1,smoothstep(0.0f,.20f,t));
    if(t<.43f) return mix(far1,mid0,smoothstep(.20f,.43f,t));
    if(t<.67f) return mix(mid0,mid1,smoothstep(.43f,.67f,t));
    if(t<.86f) return mix(mid1,near0,smoothstep(.67f,.86f,t));
    return mix(near0,near1,smoothstep(.86f,1.0f,t));
}

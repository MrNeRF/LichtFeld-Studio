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
struct PresentParameters {
    float exposure; uint tone; uint transparent; uint has_previous;
    float depth_min,depth_max; uint depth_view,depth_mode;
    float4 background;
    uint4 capture;
};
struct FrameStatus { ulong required; uint error; uint unused; };
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
kernel void present_viewer(texture2d<float, access::read> color [[texture(0)]],
    texture2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::write> rgba [[texture(2)]],
    texture2d<float, access::write> linear_depth [[texture(3)]],
    texture2d<float, access::read> previous_color [[texture(4)]],
    texture2d<float, access::read> previous_depth [[texture(5)]],
    constant PresentParameters& p [[buffer(0)]],
    device const FrameStatus& status [[buffer(1)]], uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=rgba.get_width() || pixel.y>=rgba.get_height()) return;
    if(status.error && p.has_previous) {
        const uint2 previous=uint2(ulong(pixel.x)*previous_color.get_width()/rgba.get_width(),
                                  ulong(pixel.y)*previous_color.get_height()/rgba.get_height());
        rgba.write(previous_color.read(previous),pixel);
        linear_depth.write(previous_depth.read(previous),pixel);
        return;
    }
    float4 c=color.read(pixel);
    const float4 d=depth.read(pixel);
    // Match expected_depth_finalize.slang: normalized alpha-weighted view Z,
    // with the same empty-coverage sentinel. Display depth remains median.
    // In expected capture, channel Z contains the accumulated valid-depth
    // weight, independently of visible alpha and invalid GUT contributors.
    const float output_depth=p.capture.x?(d.z>1e-4f?d.x/d.z:1e10f):d.w;
    linear_depth.write(float4(output_depth),pixel);
    if(p.depth_view) {
        const bool empty=d.y<.02f || d.w>=1e9f || d.w<=0;
        const float hi=p.depth_max<=p.depth_min+1e-5f?p.depth_min+1:p.depth_max;
        const float near_t=empty?0:1-normalized_depth(d.w,p.depth_min,hi);
        const float3 rgb=p.depth_mode==1?float3(near_t):depth_palette(near_t);
        const float coverage=empty?0:smoothstep(.02f,.72f,d.y);
        rgba.write(p.transparent?float4(rgb,coverage):float4(mix(p.background.rgb,rgb,coverage),1),pixel);
        return;
    }
    if(p.transparent && c.a>1e-6f) c.rgb/=c.a;
    c.rgb=lfsDisplayTone(max(c.rgb,0.0f),p.tone,p.exposure);
    rgba.write(c,pixel);
}

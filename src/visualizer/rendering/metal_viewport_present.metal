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
};
struct FrameStatus { ulong required; uint error; uint unused; };
float normalized_depth(float depth,float lo,float hi) {
    lo=max(lo,1e-4f); hi=max(hi,lo+1e-4f); depth=clamp(depth,lo,hi);
    const float linear=clamp((depth-lo)/max(hi-lo,1e-5f),0.0f,1.0f);
    const float logarithmic=clamp(log2(depth/lo)/max(log2(hi/lo),1e-4f),0.0f,1.0f);
    return smoothstep(0.0f,1.0f,mix(linear,logarithmic,smoothstep(1.75f,24.0f,hi/lo)));
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
    linear_depth.write(float4(d.w),pixel);
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

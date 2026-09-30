// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include <metal_stdlib>
using namespace metal;

constant uint sh_storage [[function_constant(0)]];
constant uint sh_degree [[function_constant(1)]];
constant uint primitive_mode [[function_constant(2)]];

struct Projection {
    float4x4 model_to_world, world_to_camera;
    float4 camera_local, intrinsics, clip_scale;
    uint4 extent;
};
struct InputLayout { uint count, rest, has_deleted, objects, half_attrs; };
struct SceneObject { float4x4 model_to_world; float4 camera_local; uint4 flags; };
struct ProjectedSplat { float4 mean_depth, conic_opacity, color; uint4 bounds; };

// Same cell swizzle and per-256 bounds as core/sh_value_codec.cuh.
// Q16 stores integer codes; it MUST NOT be interpreted as IEEE half.
float sh_component(device const uchar* bytes, device const float2* bounds,
                   uint primitive, uint component, uint rest) {
    const uint block=primitive/32u, lane=primitive%32u;
    if(sh_storage==3u) {
        const ulong index=ulong(block)*(rest*3u*32u)+component*32u+lane;
        const float2 mm=bounds[primitive/256u];
        const ushort q=reinterpret_cast<device const ushort*>(bytes)[index];
        return fma(mm.y-mm.x,float(q)*(1.0f/65535.0f),mm.x);
    }
    const uint slots=(rest*3u+3u)/4u;
    const ulong index=sh_storage==0u ? ulong(primitive)*rest*3u+component
        : (ulong(block)*slots*32u+(component/4u)*32u+lane)*4u+component%4u;
    if(sh_storage==2u) return float(reinterpret_cast<device const half*>(bytes)[index]);
    return reinterpret_cast<device const float*>(bytes)[index];
}

float3 evaluate_sh(float3 dc, float3 direction, device const uchar* rest,
                   device const float2* bounds, uint primitive, uint layout_rest, uint active_degree) {
    float3 color=0.5f+0.28209479177387814f*dc;
    if(sh_degree==0u || active_degree==0u) return max(color,0.0f);
    const float norm2=dot(direction,direction);
    const float3 d=norm2>1e-20f?direction*rsqrt(norm2):float3(0,0,1);
    const float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z;
    const float basis[15]={-0.4886025119029199f*y,0.4886025119029199f*z,-0.4886025119029199f*x,
        1.0925484305920792f*x*y,-1.0925484305920792f*y*z,0.31539156525252005f*(2*zz-xx-yy),
        -1.0925484305920792f*x*z,0.5462742152960396f*(xx-yy),
        0.5900435899266435f*y*(-3*xx+yy),2.890611442640554f*x*y*z,
        0.4570457994644658f*y*(xx+yy-4*zz),0.3731763325901154f*z*(2*zz-3*xx-3*yy),
        0.4570457994644658f*x*(xx+yy-4*zz),1.445305721320277f*z*(xx-yy),
        0.5900435899266435f*x*(-xx+3*yy)};
    const uint count=(active_degree+1u)*(active_degree+1u)-1u;
    for(uint k=0;k<count;++k) {
        const uint c=k*3u;
        color+=basis[k]*float3(sh_component(rest,bounds,primitive,c,layout_rest),
            sh_component(rest,bounds,primitive,c+1u,layout_rest),
            sh_component(rest,bounds,primitive,c+2u,layout_rest));
    }
    return max(color,0.0f);
}

float3 rotate_axis(float4 q,float3 v) {
    return v+2.0f*cross(q.yzw,cross(q.yzw,v)+q.x*v);
}

kernel void project_splats(device const packed_float3* means [[buffer(0)]],
    device const packed_float3* scales [[buffer(1)]], device const float4* rotations [[buffer(2)]],
    device const float* opacity [[buffer(3)]], device const packed_float3* sh0 [[buffer(4)]],
    device const uchar* rest [[buffer(5)]], device const float2* bounds [[buffer(6)]],
    device const uchar* deleted [[buffer(7)]], device ProjectedSplat* output [[buffer(8)]],
    constant Projection& frame [[buffer(9)]], constant InputLayout& layout [[buffer(10)]],
    device const uint* object_indices [[buffer(11)]], device const SceneObject* objects [[buffer(12)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=layout.count) return;
    output[i]=ProjectedSplat{};
    if(layout.has_deleted && deleted[i]) return;
    const float3 p=float3(means[i]);
    float4x4 model_to_world=frame.model_to_world;
    float3 camera_local=frame.camera_local.xyz;
    uint active_degree=sh_degree;
    if(layout.objects) {
        const uint index=object_indices[i];
        if(index>=layout.objects) return;
        const auto object=objects[index];
        if(!object.flags.x) return;
        model_to_world=object.model_to_world;
        camera_local=object.camera_local.xyz;
        active_degree=min(active_degree,object.flags.y);
    }
    const float4x4 matrix=frame.world_to_camera*model_to_world;
    const float3 view=(matrix*float4(p,1)).xyz;
    if(!all(isfinite(view)) || view.z<=frame.clip_scale.x || view.z>=frame.clip_scale.y) return;
    const float logit=layout.half_attrs?float(reinterpret_cast<device const half*>(opacity)[i]):opacity[i];
    float alpha=1.0f/(1.0f+exp(-logit));
    if(!isfinite(alpha) || alpha<1.0f/255.0f) return;
    const bool orthographic=frame.extent.z!=0;
    const float2 center=frame.intrinsics.xy*view.xy/(orthographic?1.0f:view.z)+frame.intrinsics.zw;
    float3 conic;
    float radius;
    if(primitive_mode==1u) {
        // Independent point path: no covariance, quaternion or Gaussian scale work.
        radius=2.0f;
        conic=float3(1,0,1);
    } else {
        float4 q=layout.half_attrs?float4(reinterpret_cast<device const half4*>(rotations)[i]):rotations[i];
        const float norm2=dot(q,q);
        if(!isfinite(norm2) || norm2<1e-20f) return;
        q*=rsqrt(norm2);
        const float3 log_scale=layout.half_attrs?float3(reinterpret_cast<device const packed_half3*>(scales)[i]):float3(scales[i]);
        const float3 s=exp(log_scale)*frame.clip_scale.z;
        if(!all(isfinite(s))) return;
        // Bound the covariance Jacobian near the frustum, as in 3DGS projection.
        const float2 limit=1.3f*float2(frame.extent.xy)*0.5f/frame.intrinsics.xy;
        const float2 ratio=clamp(view.xy/view.z,-limit,limit);
        const float3 jx=orthographic?float3(frame.intrinsics.x,0,0):float3(frame.intrinsics.x/view.z,0,-frame.intrinsics.x*ratio.x/view.z);
        const float3 jy=orthographic?float3(0,frame.intrinsics.y,0):float3(0,frame.intrinsics.y/view.z,-frame.intrinsics.y*ratio.y/view.z);
        const float3x3 linear=float3x3(matrix[0].xyz,matrix[1].xyz,matrix[2].xyz);
        const float3 a=linear*rotate_axis(q,float3(s.x,0,0));
        const float3 b=linear*rotate_axis(q,float3(0,s.y,0));
        const float3 c=linear*rotate_axis(q,float3(0,0,s.z));
        const float3 u=float3(dot(jx,a),dot(jx,b),dot(jx,c));
        const float3 v=float3(dot(jy,a),dot(jy,b),dot(jy,c));
        const float raw_xx=dot(u,u),raw_yy=dot(v,v);
        const float xx=raw_xx+frame.clip_scale.w,xy=dot(u,v),yy=raw_yy+frame.clip_scale.w;
        const float det=xx*yy-xy*xy;
        if(!isfinite(det) || det<=1e-12f) return;
        if(frame.extent.w) alpha*=sqrt(clamp((raw_xx*raw_yy-xy*xy)/det,0.0f,1.0f));
        if(alpha<1.0f/255.0f) return;
        conic=float3(yy,-xy,xx)/det;
        const float eigen=.5f*(xx+yy)+sqrt(max(0.0f,.25f*(xx-yy)*(xx-yy)+xy*xy));
        radius=primitive_mode==2u?3.0f*sqrt(eigen):sqrt(max(0.0f,2.0f*log(alpha*255.0f)*eigen));
    }
    if(!all(isfinite(center)) || !isfinite(radius)) return;
    const float2 extent=float2(frame.extent.xy);
    const float2 lo=clamp(floor(center-radius),0.0f,extent),hi=clamp(ceil(center+radius),0.0f,extent);
    if(any(lo>=hi)) return;
    const float3 color=evaluate_sh(float3(sh0[i]),p-camera_local,rest,bounds,i,layout.rest,active_degree);
    if(!all(isfinite(color))) return;
    output[i]={float4(center,view.z,radius),float4(conic,alpha),float4(color,1),uint4(uint2(lo),uint2(hi))};
}

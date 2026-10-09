// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../vulkan/shaders/gsplat_common.slangh"
struct Parameters {
    Ptr<float,Access.ReadWrite> camera,view,radial,tangent,prism,points,prior,pairs;
    Ptr<uint,Access.ReadWrite> count;
    float fx,fy,cx,cy,near_plane;
    uint width,height,model,samples,stride,capacity;
    float lo_x,lo_y,lo_z,hi_x,hi_y,hi_z,padding;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
[shader("compute")][numthreads(256,1,1)]
void setup(uint3 id:SV_DispatchThreadID) {
    if(id.x) return;
    GsplatSetupParams q;
    q.view=(uint64_t)p.view; q.radial=p.radial; q.tangential=p.tangent; q.prism=p.prism; q.camera=(uint64_t)p.camera;
    q.fx=p.fx; q.fy=p.fy; q.cx=p.cx; q.cy=p.cy; q.width=p.width; q.height=p.height; q.model=p.model;
    q.radial_count=6; q.tangential_count=2; q.prism_count=4;
    gsplat_camera_setup(q);
}
[shader("compute")][numthreads(256,1,1)]
void collect(uint3 id:SV_DispatchThreadID) {
    if(id.x>=p.samples) return;
    uint i=id.x*p.stride;
    float3 world=float3(p.points[3*i],p.points[3*i+1],p.points[3*i+2]);
    if(!all(isfinite(world)) || any(world<float3(p.lo_x,p.lo_y,p.lo_z)) || any(world>float3(p.hi_x,p.hi_y,p.hi_z))) return;
    GsplatCamera cam=gsplat_camera_load(p.camera);
    float3 point=mul(world,cam.R)+cam.t;
    float depth=cam.model==kGsplatPinhole?point.z:length(point);
    if(!(depth>p.near_plane) || !isfinite(depth)) return;
    float2 uv;
    if(!gsplat_camera_to_image(cam,point,uv) || !all(isfinite(uv))) return;
    int2 pixel=int2(floor(uv));
    if(any(pixel<0) || pixel.x>=p.width || pixel.y>=p.height) return;
    float prior=p.prior[pixel.y*p.width+pixel.x];
    if(!(prior>0) || !isfinite(prior)) return;
    uint slot;
    InterlockedAdd(p.count[0],1,slot);
    if(slot<p.capacity) { p.pairs[2*slot]=prior; p.pairs[2*slot+1]=depth; }
}

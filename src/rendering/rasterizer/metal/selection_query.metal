// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    float4x4 world_to_camera;
    float4 intrinsics;
    uint4 image, source, scene, payload, aabb;
    float4 ring;
};
struct Buffers {
    device const packed_float3* means;
    device const float* scales;
    device const float4* rotations;
    device const float* opacity;
    device const uchar* deleted;
    device const float4x4* transforms;
    device const int* indices;
    device const uchar* visibility;
};
float3 rotate(float4 q, float3 v) {
    return v+2.f*cross(q.yzw,cross(q.yzw,v)+q.x*v);
}
float3 camera_project(float3 view, constant Parameters& p) {
    if(p.image.z==2u) {
        const float length_view=length(view);
        if(!isfinite(length_view)||length_view<=1e-7f)return float3(0);
        const float3 direction=view/length_view;
        return float3((float2(atan2(direction.x,direction.z)/(2.f*M_PI_F),asin(clamp(direction.y,-1.f,1.f))/M_PI_F)+.5f)*float2(p.image.xy),length_view);
    }
    return float3(p.intrinsics.xy*view.xy/(p.image.z==1u?1.f:view.z)+p.intrinsics.zw,view.z);
}
bool in_margin(float2 point, constant Parameters& p, float margin) {
    const float2 extent=float2(p.image.xy);
    return all(point>=-margin*extent)&&all(point<(1.f+margin)*extent);
}
float3 cap_covariance(float3 covariance, float opacity) {
    const float power=max(4.f,log(max(opacity,.5f/255.f+1e-8f)*510.f));
    const float maximum=512.5f*512.5f/(2.f*power);
    const float average=.5f*(covariance.x+covariance.z);
    const float delta=sqrt(max(0.f,average*average-(covariance.x*covariance.z-covariance.y*covariance.y)));
    const float e1=average+delta,e2=max(average-delta,0.f);
    const float c1=min(e1,maximum),c2=min(e2,maximum);
    if(c1>=e1&&c2>=e2)return covariance;
    const float2 a=abs(covariance.y)>1e-6f?normalize(float2(covariance.y,e1-covariance.x)):(covariance.x>=covariance.z?float2(1,0):float2(0,1));
    const float2 b=float2(a.y,-a.x);
    return float3(c1*a.x*a.x+c2*b.x*b.x,c1*a.x*a.y+c2*b.x*b.y,c1*a.y*a.y+c2*b.y*b.y);
}
// Queries deliberately use center geometry, including zero-opacity splats for
// brush/rectangle/polygon, as in the shared editor's selection_mask contract.
bool project_query(uint id, constant Parameters& p, Buffers b, bool ring,
                   thread float2& center, thread float3& conic, thread float& opacity, thread float& depth) {
    if(id<p.scene.w&&b.deleted[id])return false;
    int object=p.scene.y?b.indices[id]:0;
    if(p.scene.y&&p.scene.z&&(object<0||uint(object)>=p.scene.z||!b.visibility[object]))return false;
    const float4x4 model=p.scene.x?b.transforms[uint(clamp(object,0,int(p.scene.x)-1))]:float4x4(1);
    const float3 mean=float3(b.means[id]);
    const float4x4 matrix=p.world_to_camera*model;
    const float3 world=(model*float4(mean,1)).xyz;
    const float3 view=(p.world_to_camera*float4(world,1)).xyz;
    const float3 projected=camera_project(view,p);
    if(projected.z<=p.ring.y||!all(isfinite(projected)))return false;
    center=projected.xy-.5f;
    depth=projected.z;
    if(!p.image.w&&!ring)return true;
    float4 q=p.payload.x?float4(reinterpret_cast<device const half4*>(b.rotations)[id]):b.rotations[id];
    const float norm2=dot(q,q);
    if(!isfinite(norm2)||(!ring&&norm2<=0))return false;
    q=norm2>(ring?1e-8f:0.f)?q*rsqrt(norm2):float4(1,0,0,0);
    const float3 raw=p.payload.x?float3(reinterpret_cast<device const packed_half3*>(b.scales)[id]):float3(reinterpret_cast<device const packed_float3*>(b.scales)[id]);
    const float3 scale=exp(ring?min(raw,float3(20.f)):raw);
    if(!all(isfinite(scale)))return false;
    const float3x3 linear=float3x3(matrix[0].xyz,matrix[1].xyz,matrix[2].xyz);
    const float3 a=linear*rotate(q,float3(scale.x,0,0));
    const float3 bb=linear*rotate(q,float3(0,scale.y,0));
    const float3 c=linear*rotate(q,float3(0,0,scale.z));
    float3 covariance=0;
    if(p.image.w) {
        const float lambda=.1f*.1f*3.f-3.f,denominator=3.f+lambda;
        const float sigma_scale=sqrt(denominator),mean_weight=lambda/denominator,sigma_weight=1.f/(2.f*denominator);
        const float covariance_weight=mean_weight+(1.f-.1f*.1f+2.f);
        // Keep the shared editor's source -> object -> camera evaluation order.
        // Combining matrices before constructing sigma points changes narrow
        // ring hits after the large signed UT weights amplify FP32 rounding.
        const float3 axes[3]={rotate(q,float3(scale.x,0,0)),rotate(q,float3(0,scale.y,0)),rotate(q,float3(0,0,scale.z))};
        float2 points[7];
        for(uint n=0;n<7;++n) {
            const float3 source=n?mean+(n<=3u?1.f:-1.f)*sigma_scale*axes[(n-1u)%3u]:mean;
            const float3 sigma_world=(model*float4(source,1)).xyz;
            const auto sample=camera_project((p.world_to_camera*float4(sigma_world,1)).xyz,p);
            if(sample.z<=0||!all(isfinite(sample))||!in_margin(sample.xy,p,.1f))return false;
            points[n]=sample.xy;
            if(p.image.z==2u&&n)points[n].x-=float(p.image.x)*round((points[n].x-points[0].x)/float(p.image.x));
        }
        float2 mean2d=0;
        for(uint n=0;n<7;++n)mean2d+=(n?sigma_weight:mean_weight)*points[n];
        if(ring)for(uint n=0;n<7;++n) {
            const float2 delta=points[n]-mean2d;
            covariance+=(n?sigma_weight:covariance_weight)*float3(delta.x*delta.x,delta.x*delta.y,delta.y*delta.y);
        }
        if(p.image.z==2u)mean2d.x-=float(p.image.x)*floor(mean2d.x/float(p.image.x));
        center=mean2d-.5f;
        if(!ring)return all(isfinite(center));
    } else {
        if(!in_margin(projected.xy,p,.2f))return false;
        const float2 margin=.3f*.5f*float2(p.image.xy)/p.intrinsics.xy;
        const float2 positive=(float2(p.image.xy)-p.intrinsics.zw)/p.intrinsics.xy+margin;
        const float2 negative=p.intrinsics.zw/p.intrinsics.xy+margin;
        const float2 ratio=clamp(view.xy/view.z,-negative,positive);
        const float3 jx=p.image.z==1u?float3(p.intrinsics.x,0,0):float3(p.intrinsics.x/view.z,0,-p.intrinsics.x*ratio.x/view.z);
        const float3 jy=p.image.z==1u?float3(0,p.intrinsics.y,0):float3(0,p.intrinsics.y/view.z,-p.intrinsics.y*ratio.y/view.z);
        const float3 u=float3(dot(jx,a),dot(jx,bb),dot(jx,c)),v=float3(dot(jy,a),dot(jy,bb),dot(jy,c));
        covariance=float3(dot(u,u),dot(u,v),dot(v,v));
    }
    const float before=covariance.x*covariance.z-covariance.y*covariance.y;
    const float dilation=p.payload.y?.1f:.3f;
    covariance.x+=dilation;covariance.z+=dilation;
    const float det=covariance.x*covariance.z-covariance.y*covariance.y;
    if(!isfinite(det)||det<=0)return false;
    const float raw_opacity=p.payload.x?float(reinterpret_cast<device const half*>(b.opacity)[id]):b.opacity[id];
    opacity=1.f/(1.f+exp(-raw_opacity))*(p.payload.y?sqrt(max(0.f,before/det)):1.f);
    if(opacity<=.5f/255.f||!isfinite(opacity))return false;
    if(!p.image.w)covariance=cap_covariance(covariance,opacity);
    const float final_det=covariance.x*covariance.z-covariance.y*covariance.y;
    conic=float3(covariance.z,-covariance.y,covariance.x)/final_det;
    return all(isfinite(conic));
}
bool primitive_hit(float2 point, float4 primitive, uint shape) {
    if(shape==1u)return all(point>=primitive.xy)&&all(point<=primitive.zw);
    const float2 delta=point-primitive.xy;
    return dot(delta,delta)<=primitive.z;
}
bool ring_hit(float2 point,float3 conic,float opacity,float4 primitive,float width) {
    const float boundary=(.5f/255.f)/max(opacity,1e-7f),band=max(width,0.f)*10.f;
    const float outer=boundary*(1.f+band),inner=boundary*max(0.f,1.f-band),padding=max(primitive.z,0.f);
    const float2 samples[5]={primitive.xy,primitive.xy+float2(padding,0),primitive.xy-float2(padding,0),primitive.xy+float2(0,padding),primitive.xy-float2(0,padding)};
    for(uint n=0;n<(padding>0?5u:1u);++n) {
        const float2 d=samples[n]-point;
        const float power=.5f*(conic.x*d.x*d.x+conic.z*d.y*d.y)+conic.y*d.x*d.y;
        if(power<0||!isfinite(power))continue;
        const float value=exp(-power);
        if(value<outer&&value>inner)return true;
    }
    return false;
}
kernel void polygon_coverage(constant Parameters& p [[buffer(0)]],device const packed_float2* vertices [[buffer(1)]],device uchar* output [[buffer(2)]],
                             uint2 pixel [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 cached[2048];
    const uint cached_count=min(p.source.w,2048u);
    for(uint n=lane;n<cached_count;n+=64u)cached[n]=float2(vertices[n]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(pixel.x>=p.aabb.z||pixel.y>=p.aabb.w)return;
    const float2 point=float2(p.aabb.xy+pixel)+.5f;
    bool inside=false;
    uint previous=p.source.w-1u;
    for(uint n=0;n<p.source.w;++n) {
        const float2 a=n<cached_count?cached[n]:float2(vertices[n]),b=previous<cached_count?cached[previous]:float2(vertices[previous]);
        if((a.y>point.y)!=(b.y>point.y)) {
            const float edge=(b.x-a.x)*(point.y-a.y)/(b.y-a.y)+a.x;
            if(point.x<edge)inside=!inside;
        }
        previous=n;
    }
    output[pixel.y*p.aabb.z+pixel.x]=inside?255u:0u;
}
kernel void selection_query(constant Parameters& p [[buffer(0)]],device const packed_float3* means [[buffer(1)]],device const float* scales [[buffer(2)]],
                            device const float4* rotations [[buffer(3)]],device const float* opacity [[buffer(4)]],device const uchar* deleted [[buffer(5)]],
                            device const float4x4* transforms [[buffer(6)]],device const int* indices [[buffer(7)]],device const uchar* visibility [[buffer(8)]],
                            device const float4* primitives [[buffer(9)]],device const uchar* coverage [[buffer(10)]],device uchar* output [[buffer(11)]],
                            device atomic_uint* pick [[buffer(12)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.source.x)return;
    if(p.payload.z==3u) {
        output[id]=atomic_load_explicit(pick,memory_order_relaxed)!=0xffffffffu&&atomic_load_explicit(pick+1,memory_order_relaxed)==id;
        return;
    }
    const Buffers b{means,scales,rotations,opacity,deleted,transforms,indices,visibility};
    float2 point=0;float3 conic=0;float alpha=0,depth=0;
    bool selected=false;
    if(project_query(id,p,b,p.source.y==3u,point,conic,alpha,depth)) {
        if(p.source.y==2u) {
            const int2 at=int2(floor(point))-int2(p.aabb.xy);
            selected=all(at>=0)&&all(uint2(at)<p.aabb.zw)&&coverage[uint(at.y)*p.aabb.z+uint(at.x)]!=0;
        } else for(uint n=0;n<p.source.z;++n)
            if(p.source.y==3u?ring_hit(point,conic,alpha,primitives[n],p.ring.x):primitive_hit(point,primitives[n],p.source.y)) {selected=true;break;}
    }
    if(p.source.y==3u) {
        if(selected&&isfinite(depth)&&depth>0) {
            const uint key=as_type<uint>(depth);
            if(p.payload.z==1u)atomic_fetch_min_explicit(pick,key,memory_order_relaxed);
            else if(atomic_load_explicit(pick,memory_order_relaxed)==key)atomic_fetch_min_explicit(pick+1,id,memory_order_relaxed);
        }
    } else output[id]=selected;
}

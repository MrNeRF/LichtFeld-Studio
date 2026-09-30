// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include <metal_stdlib>
using namespace metal;
@LFS_METAL_OVERLAY@
#define LFS_PORTAL_INLINE inline
#define LFS_PORTAL_VEC3 float3
#define LFS_PORTAL_VEC4 float4
@LFS_METAL_PORTAL@
#define LFS_COLOR_INLINE inline
#define LFS_COLOR_VEC3 float3
#define LFS_COLOR_UINT uint
#define LFS_COLOR_MIX mix
@LFS_METAL_DISPLAY_COLOR@

constant uint sh_storage [[function_constant(0)]];
constant uint sh_degree [[function_constant(1)]];
constant uint primitive_mode [[function_constant(2)]];

struct Projection {
    float4x4 model_to_world, world_to_camera;
    float4 camera_local, intrinsics, clip_scale;
    uint4 extent;
    float4 rasterization, display, panorama;
};
struct InputLayout { uint count, rest, has_deleted, objects, half_attrs, overlay, object_indexed, draw_count, lod, page_splats; };
struct SceneObject { float4x4 model_to_world; float4 camera_local; uint4 flags; };
struct GutSplat { float4 inverse0, inverse1, inverse2, mean_opacity; };
struct ProjectedSplat { float4 mean_depth, conic_opacity, color; uint4 bounds; };

// Full camera coordinates are retained when rendering a subregion.
float2 panorama_project(float3 view, float2 extent) {
    const float distance=length(view);
    const float3 direction=view/max(distance,1e-8f);
    return float2(atan2(direction.x,direction.z)/(2.f*M_PI_F)+.5f,
                  asin(clamp(direction.y,-1.f,1.f))/M_PI_F+.5f)*extent;
}

// Intersect periodic support with the actual pixel viewport before converting
// to tiles. Wrapping the padded tile grid instead loses pixels at odd widths.
uint2 panorama_tile_span(float center, float radius, float period, uint width) {
    const uint columns=(width+15u)/16u;
    if(2.f*radius>=period)return uint2(0,columns*16u);
    center+=period*round((.5f*float(width)-center)/period);
    uint first=columns, end=0, second=columns, second_end=0;
    for(int shift=-1;shift<=1;++shift) {
        const float lo=max(0.f,center+float(shift)*period-radius);
        const float hi=min(float(width),center+float(shift)*period+radius);
        if(lo>=hi)continue;
        const uint x0=uint(floor(lo/16.f)),x1=min(columns,uint(ceil(hi/16.f)));
        if(first==columns){first=x0;end=x1;}
        else {second=x0;second_end=x1;}
    }
    if(first==columns)return uint2(0);
    if(second==columns)return uint2(first,end)*16u;
    if(second<first){const uint a=first,b=end;first=second;end=second_end;second=a;second_end=b;}
    if(end>=second)return uint2(0,columns*16u);
    return uint2(second,columns+end)*16u;
}

// Matches the Studio reference's covariance extent limit before inversion.
float3 clamp_covariance_extent(float3 covariance, float opacity) {
    const float power=max(4.f,log(max(opacity,.5f/255.f+1e-8f)*510.f));
    const float maximum=512.5f*512.5f/(2.f*power);
    const float average=.5f*(covariance.x+covariance.z);
    const float delta=sqrt(max(0.f,average*average-(covariance.x*covariance.z-covariance.y*covariance.y)));
    const float eigen1=average+delta, eigen2=max(average-delta,0.f);
    const float capped1=min(eigen1,maximum), capped2=min(eigen2,maximum);
    if(capped1>=eigen1 && capped2>=eigen2) return covariance;
    const float2 axis=abs(covariance.y)>1e-6f?normalize(float2(covariance.y,eigen1-covariance.x)):
        (covariance.x>=covariance.z?float2(1,0):float2(0,1));
    const float2 other=float2(axis.y,-axis.x);
    return float3(capped1*axis.x*axis.x+capped2*other.x*other.x,
                  capped1*axis.x*axis.y+capped2*other.x*other.y,
                  capped1*axis.y*axis.y+capped2*other.y*other.y);
}

// Portal billboard eigen-axis limits, in source viewport pixel units.
float3 portal_covariance(float3 covariance, float2 extent) {
    const float limit=min(1024.f,min(extent.x,extent.y));
    const float cap=max(1.f,sqrt(8.f*max(covariance.x,covariance.z))/limit);
    covariance/=cap*cap;
    const float mid=.5f*(covariance.x+covariance.z);
    const float radius=length(float2(.5f*(covariance.x-covariance.z),covariance.y));
    const float first=mid+radius,second=max(mid-radius,.025f);
    const float maximum=limit*limit/8.f;
    const float a=min(first,maximum),b=min(second,maximum);
    const float2 axis=abs(covariance.y)>1e-6f?normalize(float2(covariance.y,first-covariance.x)):
        (covariance.x>=covariance.z?float2(1,0):float2(0,1));
    const float2 other=float2(axis.y,-axis.x);
    return float3(a*axis.x*axis.x+b*other.x*other.x,
                  a*axis.x*axis.y+b*other.x*other.y,
                  a*axis.y*axis.y+b*other.y*other.y);
}

// Same cell swizzle and per-256 bounds as core/sh_value_codec.cuh.
// Q16 stores integer codes; it MUST NOT be interpreted as IEEE half.
float sh_component(device const uchar* bytes, device const float2* bounds,
                   uint primitive, uint component, uint rest, uint page_splats) {
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
    if(sh_storage==4u) {
        // RAD uses signed-byte components in the same 32-row float4 cell
        // swizzle, with separate scales for SH degrees one, two and three.
        const float4 maxima=reinterpret_cast<device const float4*>(bounds)[(primitive/page_splats)*4u];
        const uint band=component<9u?0u:component<24u?1u:2u;
        return (float(reinterpret_cast<device const char*>(bytes)[index])/127.f)*maxima[band];
    }
    if(sh_storage==2u) return float(reinterpret_cast<device const half*>(bytes)[index]);
    return reinterpret_cast<device const float*>(bytes)[index];
}

float3 evaluate_sh(float3 dc, float3 direction, device const uchar* rest,
                   device const float2* bounds, uint primitive, uint layout_rest, uint active_degree, uint page_splats) {
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
        color+=basis[k]*float3(sh_component(rest,bounds,primitive,c,layout_rest,page_splats),
            sh_component(rest,bounds,primitive,c+1u,layout_rest,page_splats),
            sh_component(rest,bounds,primitive,c+2u,layout_rest,page_splats));
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
    device const float4* params [[buffer(13)]], device uint* overlay_flags [[buffer(14)]],
    device const uchar* node_mask [[buffer(15)]],
    device GutSplat* gut_output [[buffer(16)]],
    device const uint* lod_indices [[buffer(17)]],
    device const uint* logical_indices [[buffer(18)]],
    device const uint* lod_levels [[buffer(19)]],
    device const float* lod_weights [[buffer(20)]],
    device const uint* lod_count [[buffer(21)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=layout.draw_count) return;
    output[i]=ProjectedSplat{};
    if(layout.overlay)overlay_flags[i]=0;
    // Clear unused slots before reading GPU-private indirection. An overflow
    // cannot publish a truncated cut; the selector retries its budget on GPU.
    if((layout.lod&32u) && (lod_count[0]>layout.draw_count || i>=lod_count[0]))return;
    const uint source=(layout.lod&1u)?lod_indices[i]:i;
    const uint logical=(layout.lod&2u)?logical_indices[i]:source;
    if(source>=layout.count || logical>=layout.count)return;
    if(layout.has_deleted && deleted[source]) return;
    const float3 p=float3(means[source]);
    float4x4 model_to_world=frame.model_to_world;
    float3 camera_local=frame.camera_local.xyz;
    uint active_degree=sh_degree;
    int node=0;
    if(layout.objects) {
        const uint index=layout.object_indexed?object_indices[logical]:0;
        node=int(index);
        if(index>=layout.objects) return;
        const auto object=objects[index];
        if(!object.flags.x) return;
        model_to_world=object.model_to_world;
        camera_local=object.camera_local.xyz;
        active_degree=min(active_degree,object.flags.y);
    }
    const float4x4 matrix=frame.world_to_camera*model_to_world;
    const float3 view=(matrix*float4(p,1)).xyz;
    const bool spark=frame.display.z==1.f;
    const bool portal=frame.rasterization.w==1.f && !spark;
    const bool equirectangular=frame.extent.z==2u;
    const bool orthographic=frame.extent.z==1u;
    const float projection_depth=equirectangular?length(view):view.z;
    if(!all(isfinite(view)) || projection_depth<=frame.clip_scale.x || projection_depth>=frame.clip_scale.y) return;
    uint flags=0;
    if(layout.overlay){
        bool active=true;
        const float3 world=(model_to_world*float4(p,1)).xyz;
        overlay_filter(params,0,false,node,world,active,flags);
        for(uint n=0;n<15 && overlay_enabled(params[26+n*7].x);++n)overlay_filter(params,26+n*7,false,node,world,active,flags);
        overlay_filter(params,7,true,node,world,active,flags);
        for(uint n=0;n<15 && overlay_enabled(params[131+n*5].x);++n)overlay_filter(params,131+n*5,true,node,world,active,flags);
        if(active&&overlay_enabled(params[13].x)){
            float4 k=params[12];
            if(k.x<=0)k=float4(frame.intrinsics.xy,float2(frame.extent.xy)*.5f);
            if(orthographic)k=float4(frame.intrinsics.xy,float2(frame.extent.xy)*.5f);
            const float2 full_extent=equirectangular?frame.panorama.xy:float2(frame.extent.xy);
            const float2 center=equirectangular?panorama_project(view,full_extent):k.xy*view.xy/(orthographic?1.f:view.z)+k.zw;
            const float2 half_extent=.5f*full_extent*params[206].xy;
            const float2 window=.5f*full_extent+float2(params[14].w,params[15].w)*(.5f*full_extent-half_extent);
            const bool inside=all(abs(center-window)<=half_extent)&&projection_depth>=-params[15].z&&projection_depth<=-params[14].z;
            if(!inside){flags|=2u;if(overlay_enabled(params[13].z))flags|=1u;if(overlay_enabled(params[13].y))active=false;}
        }
        const float4 emphasis=params[20];
        if(active&&overlay_enabled(emphasis.y)&&node>=0&&node<int(round(emphasis.z))){
            if(overlay_enabled(emphasis.x)&&!node_mask[node])flags|=3u;
            if(emphasis.w>0&&node_mask[node])flags|=4u;
        }
        overlay_flags[i]=flags;
        if(!active)return;
    }
    const float logit=layout.half_attrs?float(reinterpret_cast<device const half*>(opacity)[source]):opacity[source];
    float alpha=spark?max(logit,0.f):1.0f/(1.0f+exp(-logit));
    if(spark && alpha>1.f)alpha=min(alpha*4.f-3.f,5.f);
    if(!isfinite(alpha) || alpha<0.5f/255.0f) return;
    if(layout.lod&8u)alpha*=clamp(lod_weights[i],0.f,1.f);
    if(!isfinite(alpha))return;
    if(portal)alpha=lfsPortalCompactOpacity(alpha);
    if(portal && alpha<=1.f/255.f)return;
    const float source_alpha=alpha;
    // Vulkan's project_splat subtracts half a pixel before integer sampling.
    float2 center=equirectangular?panorama_project(view,frame.panorama.xy)-frame.panorama.zw-.5f:
        frame.intrinsics.xy*view.xy/(orthographic?1.0f:view.z)+frame.intrinsics.zw-.5f;
    float3 conic;
    float radius;
    float2 panorama_radius=0,portal_axis=0,portal_extent=0;
    if(primitive_mode==1u) {
        // Independent point path: no covariance, quaternion or Gaussian scale work.
        radius=2.0f;
        conic=float3(1,0,1);
    } else {
        float4 q=layout.half_attrs?float4(reinterpret_cast<device const half4*>(rotations)[source]):rotations[source];
        const float norm2=dot(q,q);
        if(!isfinite(norm2)) return;
        q=norm2>1e-8f?q*rsqrt(norm2):float4(1,0,0,0);
        if(portal){q=lfsPortalCompactRotation(q);q*=rsqrt(max(dot(q,q),1e-8f));}
        const float3 log_scale=sh_storage==4u?float3(reinterpret_cast<device const half4*>(scales)[source].xyz):
            layout.half_attrs?float3(reinterpret_cast<device const packed_half3*>(scales)[source]):float3(scales[source]);
        float3 s=exp(min(log_scale,float3(20.f)))*frame.clip_scale.z;
        if(portal)s=lfsPortalCompactScales(s);
        if(!all(isfinite(s))) return;
        const float3x3 linear=float3x3(matrix[0].xyz,matrix[1].xyz,matrix[2].xyz);
        const float3 a=linear*rotate_axis(q,float3(s.x,0,0));
        const float3 b=linear*rotate_axis(q,float3(0,s.y,0));
        const float3 c=linear*rotate_axis(q,float3(0,0,s.z));
        float3 raw_covariance;
        if(primitive_mode!=3u) {
            // Bound the covariance Jacobian near the frustum, as in 3DGS projection.
            const float2 margin=.3f*.5f*float2(frame.extent.xy)/frame.intrinsics.xy;
            const float2 positive=(float2(frame.extent.xy)-frame.intrinsics.zw)/frame.intrinsics.xy+margin;
            const float2 negative=frame.intrinsics.zw/frame.intrinsics.xy+margin;
            const float2 ratio=portal?view.xy/view.z:clamp(view.xy/view.z,-negative,positive);
            const float3 jx=orthographic?float3(frame.intrinsics.x,0,0):float3(frame.intrinsics.x/view.z,0,-frame.intrinsics.x*ratio.x/view.z);
            const float3 jy=orthographic?float3(0,frame.intrinsics.y,0):float3(0,frame.intrinsics.y/view.z,-frame.intrinsics.y*ratio.y/view.z);
            const float3 u=float3(dot(jx,a),dot(jx,b),dot(jx,c));
            const float3 v=float3(dot(jy,a),dot(jy,b),dot(jy,c));
            raw_covariance=float3(dot(u,u),dot(u,v),dot(v,v));
        }
        if(primitive_mode==3u) {
            // Match project_gaussian_to_camera_gut, including FP32 UT weights.
            const float lambda=.1f*.1f*3.f-3.f;
            const float denominator=3.f+lambda;
            const float sigma_scale=sqrt(denominator);
            const float mean_weight=lambda/denominator;
            const float covariance_weight=mean_weight+(1.f-.1f*.1f+2.f);
            const float sigma_weight=1.f/(2.f*denominator);
            const float3 sigma[7]={view,view+sigma_scale*a,view+sigma_scale*b,view+sigma_scale*c,
                                 view-sigma_scale*a,view-sigma_scale*b,view-sigma_scale*c};
            float2 image[7];
            const float raster_scale=frame.rasterization.x>0?frame.rasterization.x:1.f;
            const float2 panorama_extent=max(float2(1),round(frame.panorama.xy/raster_scale))*raster_scale;
            for(uint n=0;n<7;++n) {
                if(!all(isfinite(sigma[n])) || (equirectangular?length(sigma[n])<=1e-8f:sigma[n].z<=frame.clip_scale.x))return;
                image[n]=equirectangular?panorama_project(sigma[n],panorama_extent):
                    frame.intrinsics.xy*sigma[n].xy/(orthographic?1.f:sigma[n].z)+frame.intrinsics.zw;
                if(equirectangular && n) {
                    float dx=image[n].x-image[0].x;
                    dx-=panorama_extent.x*round(dx/panorama_extent.x);
                    image[n].x=image[0].x+dx;
                }
            }
            // Accumulate relative offsets for the spherical projection: the
            // UT weights sum to one but a negative weight near -99 amplifies
            // cancellation of absolute pixel coordinates at large extents.
            float2 mean=0;
            if(equirectangular) {
                float2 offset=0;
                for(uint n=1;n<7;++n)offset+=image[n]-image[0];
                mean=fma(float2(sigma_weight),offset,image[0]);
            } else {
                for(uint n=0;n<7;++n)mean+=(n? sigma_weight:mean_weight)*image[n];
            }
            raw_covariance=0;
            for(uint n=0;n<7;++n) {
                const float2 d=image[n]-mean;
                raw_covariance+=(n?sigma_weight:covariance_weight)*float3(d.x*d.x,d.x*d.y,d.y*d.y);
            }
            if(equirectangular)mean.x-=panorama_extent.x*floor(mean.x/panorama_extent.x);
            center=mean-.5f-(equirectangular?frame.panorama.zw:float2(0));
            // Actual 3D alpha uses source scale, independently of binning scale
            // modifier/mip compensation, matching load_splat_gut_alphablend.
            const float3 source_scale=portal?lfsPortalCompactScales(exp(log_scale)):exp(log_scale);
            const float3 base_scale=max(source_scale,float3(1e-8f));
            const float3 ga=linear*rotate_axis(q,float3(base_scale.x,0,0));
            const float3 gb=linear*rotate_axis(q,float3(0,base_scale.y,0));
            const float3 gc=linear*rotate_axis(q,float3(0,0,base_scale.z));
            const float determinant=dot(ga,cross(gb,gc));
            if(!isfinite(determinant)||abs(determinant)<=1e-30f)return;
            gut_output[i]={float4(cross(gb,gc)/determinant,0),float4(cross(gc,ga)/determinant,0),
                           float4(cross(ga,gb)/determinant,0),float4(view,alpha)};
        }
        // Dilation and the covariance cap use source viewport pixels,
        // then scale to output pixels, matching high-resolution Vulkan exports.
        const float raster_scale=isfinite(frame.rasterization.x)&&frame.rasterization.x>0?frame.rasterization.x:1.f;
        const float variance_scale=raster_scale*raster_scale;
        const float raw_xx=raw_covariance.x/variance_scale,raw_yy=raw_covariance.z/variance_scale;
        float3 covariance=float3(raw_xx+frame.clip_scale.w,raw_covariance.y/variance_scale,raw_yy+frame.clip_scale.w);
        float det=covariance.x*covariance.z-covariance.y*covariance.y;
        if(!isfinite(det) || det<=1e-12f) return;
        if(frame.extent.w || (spark && primitive_mode!=3u)) alpha*=sqrt(max((raw_xx*raw_yy-covariance.y*covariance.y)/det,0.f));
        if(alpha<0.5f/255.0f) return;
        const float2 camera_extent=all(frame.panorama.xy>0)?frame.panorama.xy:float2(frame.extent.xy);
        const float2 source_extent=max(float2(1),round(camera_extent/raster_scale));
        if(portal && primitive_mode!=3u)covariance=portal_covariance(covariance,source_extent);
        else if(!portal && primitive_mode!=3u)covariance=clamp_covariance_extent(covariance,alpha);
        covariance*=variance_scale;
        const float xx=covariance.x,xy=covariance.y,yy=covariance.z;
        det=xx*yy-xy*xy;
        conic=float3(yy,-xy,xx)/det;
        float3 bin_covariance=covariance;
        if(portal && primitive_mode==3u) {
            // Preserve raw conic for rings while binning the clamped billboard.
            const float3 billboard=portal_covariance(covariance,camera_extent);
            const float mid=.5f*(billboard.x+billboard.z);
            const float delta=length(float2(.5f*(billboard.x-billboard.z),billboard.y));
            const float first=mid+delta,second=max(mid-delta,.025f);
            portal_axis=abs(billboard.y)>1e-6f?normalize(float2(billboard.y,first-billboard.x)):
                (billboard.x>=billboard.z?float2(1,0):float2(0,1));
            const float clip=min(1.f,.5f*sqrt(max(0.f,log(source_alpha*255.f))));
            portal_extent=sqrt(8.f*float2(first,second))*clip;
            gut_output[i].inverse0.w=portal_axis.x;
            gut_output[i].inverse1.w=portal_axis.y;
            gut_output[i].inverse2.w=portal_extent.y;
            bin_covariance=2.f*billboard;
        }
        const float eigen=.5f*(bin_covariance.x+bin_covariance.z)+sqrt(max(0.0f,.25f*(bin_covariance.x-bin_covariance.z)*(bin_covariance.x-bin_covariance.z)+bin_covariance.y*bin_covariance.y));
        radius=primitive_mode==2u?3.0f*sqrt(eigen):sqrt(2.0f*max(4.0f,log(alpha*510.0f))*eigen);
        if(equirectangular) {
            const float extent_factor=sqrt(2.f*max(4.f,log(source_alpha*510.f)));
            panorama_radius=min(sqrt(bin_covariance.xz)*extent_factor,frame.panorama.xy*float2(1.f,.49f));
            radius=max(panorama_radius.x,panorama_radius.y);
        }
    }
    if(!all(isfinite(center)) || !isfinite(radius)) return;
    const float2 extent=float2(frame.extent.xy);
    const float2 support=equirectangular?panorama_radius:float2(radius);
    float2 lo=clamp(floor(center-support),0.0f,extent),hi=clamp(ceil(center+support),0.0f,extent);
    if(equirectangular) {
        const uint2 span=panorama_tile_span(center.x,support.x,frame.panorama.x,frame.extent.x);
        lo.x=float(span.x);hi.x=float(span.y);
    }
    if(any(lo>=hi)) return;
    // Match view_direction_to_point + extract_rotation_rows in Vulkan.
    // SH follows the object's rotation; nonuniform scale must not distort it.
    const float3x3 model_linear=float3x3(model_to_world[0].xyz,model_to_world[1].xyz,model_to_world[2].xyz);
    float3 direction=orthographic?float3(frame.world_to_camera[0].z,frame.world_to_camera[1].z,frame.world_to_camera[2].z):
        model_linear*(p-camera_local);
    const float3 axis_lengths=float3(length(model_linear[0]),length(model_linear[1]),length(model_linear[2]));
    if(all(axis_lengths>1e-8f))direction=float3(dot(model_linear[0]/axis_lengths.x,direction),
        dot(model_linear[1]/axis_lengths.y,direction),dot(model_linear[2]/axis_lengths.z,direction));
    float3 color=evaluate_sh(sh_storage==4u?float3(reinterpret_cast<device const half4*>(sh0)[source].xyz):float3(sh0[source]),direction,rest,bounds,source,layout.rest,active_degree,layout.page_splats);
    if(layout.lod&16u) {
        const uint level=(layout.lod&4u)?lod_levels[i]%5u:0u;
        const float3 palette[5]={float3(1,0,0),float3(0,1,0),float3(0,0,1),float3(1,1,0),float3(1,0,1)};
        color*=palette[level];
    }
    if(portal)color=lfsPortalCompactColor(color);
    // Portal tone is still per Gaussian for Spark density, while compact
    // codecs and normalized tails apply only to the standard opacity profile.
    if(frame.rasterization.w==1.f && frame.display.x>0)
        color=lfsDisplaySplat(color,uint(frame.display.x),frame.display.y);
    if(layout.overlay)color=overlay_projection_color(color,center+(equirectangular?frame.panorama.zw:float2(0)),flags,params);
    if(!all(isfinite(color))) return;
    // The viewer reference sorts by radial distance squared, not camera Z.
    // Retain positive projection depth for admission; the ray path evaluates view-Z depth.
    // color.w is reserved for the radial sort metric.
    output[i]={float4(center,projection_depth,portal && primitive_mode==3u?portal_extent.x:radius),float4(conic,alpha),float4(color,dot(view,view)),uint4(uint2(lo),uint2(hi))};
}

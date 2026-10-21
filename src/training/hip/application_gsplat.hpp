/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_gsplat_api.h"
#include "GeometryFeatures.h"
namespace lfs_hip_gsplat_detail {
template<class T=float>T* ptr(uint64_t v){return reinterpret_cast<T*>(static_cast<uintptr_t>(v));}
struct Work {
    Budget budget;int count,width,height,channels,bases,tile_w,tile_h;uint64_t intersections=0;size_t scan_bytes=0;
    Buffer radii,means2d,depths,conics,counts,prefix,colors,directions,offsets,last,grad_colors,grad_params,grad_sh;
    std::unique_ptr<Buffer> scratch;std::unique_ptr<Bins> bins;
    Work(uint64_t bytes,const LfsHipGsplatForward& p,hipStream_t q)
        :budget{bytes},count(p.count),width(p.width),height(p.height),channels(p.channels),bases((p.active_degree+1)*(p.active_degree+1)),tile_w((width+15)/16),tile_h((height+15)/16),
        radii(budget,multiply(count,8)),means2d(budget,multiply(count,8)),depths(budget,multiply(count,4)),conics(budget,multiply(count,12)),counts(budget,multiply(count,4)),prefix(budget,multiply(count,8)),colors(budget,multiply(count,channels*4)),directions(budget,multiply(count,12)),offsets(budget,multiply(size_t(tile_w)*tile_h+1,4)),last(budget,multiply(size_t(width)*height,4)),grad_colors(budget,multiply(count,channels*4)),grad_params(budget,multiply(count,11*4)),grad_sh(budget,multiply(count,bases*12)) {
        hip_check(hipcub::DeviceScan::InclusiveSum(nullptr,scan_bytes,CountIterator(counts.as<int32_t>(),Widen{}),prefix.as<int64_t>(),count,q),"Gsplat scan query");scratch=std::make_unique<Buffer>(budget,scan_bytes);
    }
    void prepare(const LfsHipGsplatForward& p,hipStream_t q) {
        gsplat_lfs::launch_projection_ut_3dgs_fused_kernel(ptr(p.means),ptr(p.quats),ptr(p.scales),ptr(p.opacities),ptr(p.view),nullptr,ptr(p.K),count,1,width,height,.3f,.01f,10000.f,0,static_cast<CameraModelType>(p.camera_model),{},ShutterType::GLOBAL,ptr(p.radial),ptr(p.tangential),ptr(p.thin_prism),radii.as<int32_t>(),means2d.as(),depths.as(),conics.as(),nullptr,q);
        gsplat_lfs::launch_intersect_tile_kernel(means2d.as(),radii.as<int32_t>(),depths.as(),nullptr,nullptr,1,count,0,false,16,tile_w,tile_h,nullptr,counts.as<int32_t>(),nullptr,nullptr,q);
        auto bytes=scan_bytes;hip_check(hipcub::DeviceScan::InclusiveSum(scratch->pointer,bytes,CountIterator(counts.as<int32_t>(),Widen{}),prefix.as<int64_t>(),count,q),"Gsplat count scan");
        int64_t total=0;hip_check(hipMemcpyAsync(&total,prefix.as<int64_t>()+count-1,sizeof(total),hipMemcpyDeviceToHost,q),"Gsplat intersection count");hip_check(hipStreamSynchronize(q),"Gsplat intersection count completion");
        require(total>=0&&total<INT_MAX,"Gsplat intersection count exceeds int32");intersections=total;
        if(!total){offsets.zero(q);return;}
        const size_t needed=size_t(total)+1;size_t sort_bytes=0;
        hip_check(hipcub::DeviceRadixSort::SortPairs(nullptr,sort_bytes,bins?bins->keys.as<int64_t>():nullptr,bins?bins->sorted_keys.as<int64_t>():nullptr,bins?bins->ids.as<int32_t>():nullptr,bins?bins->sorted_ids.as<int32_t>():nullptr,needed,0,64,q),"Gsplat sort query");
        const auto replacement_bytes=sort_bytes>scratch->bytes?std::max(scan_bytes,sort_bytes):0;
        if(replacement_bytes>budget.limit-budget.live)throw Failure(LFS_HIP_TRAIN_RESOURCE_EXHAUSTED,"Gsplat replacement scratch exceeds budget");
        std::unique_ptr<Bins> next;std::unique_ptr<Buffer> next_scratch;
        if(!bins||needed>bins->capacity){const auto available=(budget.limit-budget.live-replacement_bytes)/24;const auto proposed=bins?std::max(needed,bins->capacity+bins->capacity/2):needed;const auto capacity=std::min<size_t>(INT_MAX,std::min(proposed,available));if(capacity<needed)throw Failure(LFS_HIP_TRAIN_RESOURCE_EXHAUSTED,"Gsplat intersection cache exceeds budget");next=std::make_unique<Bins>(budget,capacity);}
        if(replacement_bytes)next_scratch=std::make_unique<Buffer>(budget,replacement_bytes);if(next)bins.swap(next);if(next_scratch)scratch.swap(next_scratch);
        gsplat_lfs::launch_fill_isect_sentinels_kernel(bins->keys.as<int64_t>(),bins->ids.as<int32_t>(),needed,std::numeric_limits<int64_t>::max(),q);
        gsplat_lfs::launch_intersect_tile_kernel(means2d.as(),radii.as<int32_t>(),depths.as(),nullptr,nullptr,1,count,0,false,16,tile_w,tile_h,prefix.as<int64_t>(),nullptr,bins->keys.as<int64_t>(),bins->ids.as<int32_t>(),q,total);
        hip_check(hipcub::DeviceRadixSort::SortPairs(scratch->pointer,sort_bytes,bins->keys.as<int64_t>(),bins->sorted_keys.as<int64_t>(),bins->ids.as<int32_t>(),bins->sorted_ids.as<int32_t>(),needed,0,64,q),"Gsplat tile sort");
        gsplat_lfs::launch_intersect_offset_kernel(bins->sorted_keys.as<int64_t>(),needed,1,tile_w,tile_h,offsets.as<int32_t>(),q);
    }
    template<unsigned C>void forward(const LfsHipGsplatForward& p,hipStream_t q) {
        gsplat_lfs::launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<C>(ptr(p.means),ptr(p.quats),ptr(p.scales),colors.as(),ptr(p.opacities),ptr(p.background),ptr(p.image_background),nullptr,1,count,intersections,width,height,16,ptr(p.view),nullptr,ptr(p.K),static_cast<CameraModelType>(p.camera_model),{},ShutterType::GLOBAL,ptr(p.radial),ptr(p.tangential),ptr(p.thin_prism),offsets.as<int32_t>(),bins?bins->sorted_ids.as<int32_t>():nullptr,ptr(p.image),ptr(p.alpha),last.as<int32_t>(),q);
    }
    template<unsigned C>void backward(const LfsHipGsplatForward& p,const LfsHipGsplatBackward& b,hipStream_t q) {
        gsplat_lfs::launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<C>(ptr(p.means),ptr(p.quats),ptr(p.scales),colors.as(),ptr(p.opacities),ptr(p.background),ptr(p.image_background),nullptr,1,count,intersections,width,height,16,ptr(p.view),nullptr,ptr(p.K),static_cast<CameraModelType>(p.camera_model),{},ShutterType::GLOBAL,ptr(p.radial),ptr(p.tangential),ptr(p.thin_prism),offsets.as<int32_t>(),bins?bins->sorted_ids.as<int32_t>():nullptr,ptr(p.alpha),last.as<int32_t>(),ptr(b.image),ptr(b.alpha),grad_params.as(),grad_params.as(count*3),grad_params.as(count*7),grad_colors.as(),grad_params.as(count*10),ptr(b.densification),ptr(b.error_map),ptr(b.edge_map),ptr(b.edge_scores),q);
    }
};
__global__ void accumulate(float* destination,const float* source,size_t count){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)destination[i]+=source[i];}
__global__ void accumulate_sh(const float* source,float* dc,float* rest,int n,int bases,int layout_bases) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*bases*3)return;const auto c=i%3,k=(i/3)%bases,row=i/(bases*3);
    if(!k)dc[row*3+c]+=source[i];else if(rest){const unsigned slots=((layout_bases-1)*3+3)/4;const unsigned cell=(k-1)*3+c;rest[((row/32)*slots*32+(cell/4)*32+row%32)*4+cell%4]+=source[i];}
}
__global__ void metrics(const float* gm,const int32_t* radii,const float* means,float* density,float* shares,int count,int width,int height,bool norm) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    if(norm&&density)density[i]+=sqrtf(gm[i*3]*gm[i*3]+gm[i*3+1]*gm[i*3+1]+gm[i*3+2]*gm[i*3+2]);
    if(shares&&radii[i*2]>0&&radii[i*2+1]>0){float x=means[i*2],y=means[i*2+1],rx=radii[i*2],ry=radii[i*2+1];float dx=fmaxf(0,fminf(width,x+rx)-fmaxf(0,x-rx)),dy=fmaxf(0,fminf(height,y+ry)-fmaxf(0,y-ry));shares[i]=fmaxf(shares[i],dx/width*dy/height);}
}
}
struct LfsHipApplicationGsplat {
    int device;hipStream_t queue;uint64_t limit;bool live=false,poisoned=false;LfsHipGsplatForward input{};std::unique_ptr<lfs_hip_gsplat_detail::Work> work;
};
namespace lfs_hip_gsplat_detail {
template<class F>int call(LfsHipApplicationGsplat* s,LfsHipError* error,F&& fn){return boundary(nullptr,error,[&]{require(s&&!s->poisoned,"Missing or poisoned HIP Gsplat state");DeviceScope device(s->device);try{fn();}catch(...){auto exception=std::current_exception();s->live=false;if(auto status=hipStreamSynchronize(s->queue);status!=hipSuccess){s->poisoned=true;hip_check(status,"Failed HIP Gsplat completion");}std::rethrow_exception(exception);}});}
}
extern "C" int lfs_hip_application_gsplat_create(int32_t device,uint64_t queue,uint64_t budget,LfsHipApplicationGsplat** output,LfsHipError* error){if(output)*output=nullptr;return boundary(nullptr,error,[&]{require(output&&queue&&queue<=UINTPTR_MAX&&budget>0,"Invalid HIP Gsplat creation descriptor");DeviceScope selected(device);auto s=std::make_unique<LfsHipApplicationGsplat>();s->device=device;s->queue=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(queue));s->limit=budget;*output=s.release();});}
extern "C" int lfs_hip_application_gsplat_forward(LfsHipApplicationGsplat* s,const LfsHipGsplatForward* p,LfsHipError* error) {
    using namespace lfs_hip_gsplat_detail;return call(s,error,[&] {
        require(p&&p->struct_bytes==sizeof(*p)&&p->count>0&&p->count<=INT_MAX/48&&p->width>0&&p->height>0&&uint64_t(p->width)*p->height<=INT_MAX/8,"Invalid HIP Gsplat forward dimensions");
        require((p->channels==1||p->channels==3||p->channels==4||p->channels==8)&&p->camera_model>=0&&p->camera_model<=4&&p->camera_model!=1&&p->active_degree>=0&&p->active_degree<=3&&p->layout_degree>=p->active_degree&&p->layout_degree<=3,"Invalid HIP Gsplat camera or SH policy");
        require(p->means&&p->quats&&p->scales&&p->opacities&&p->view&&p->K&&p->image&&p->alpha&&(p->channels==1||p->sh0)&&(!p->active_degree||p->shN),"Missing HIP Gsplat forward binding");
        s->live=false;auto& w=s->work;
        const auto bases=(p->active_degree+1)*(p->active_degree+1);
        if(!w||w->count!=p->count||w->width!=p->width||w->height!=p->height||w->channels!=p->channels||w->bases!=bases){hip_check(hipStreamSynchronize(s->queue),"Gsplat workspace replacement completion");w.reset();w=std::make_unique<Work>(s->limit,*p,s->queue);}
        s->input=*p;w->prepare(*p,s->queue);w->colors.zero(s->queue);
        if(p->channels!=1){if(p->active_degree)gsplat_lfs::compute_view_dirs(ptr(p->means),ptr(p->view),1,p->count,w->directions.as(),s->queue);gsplat_lfs::launch_spherical_harmonics_swizzled_fwd_kernel(p->active_degree,p->layout_degree,p->channels,p->active_degree?w->directions.as():nullptr,ptr(p->sh0),ptr(p->shN),nullptr,p->count,w->colors.as(),s->queue);}
        if(p->channels!=3)gsplat_lfs::geometry_features_fwd(ptr(p->means),ptr(p->quats),ptr(p->scales),ptr(p->view),w->colors.as(),p->count,1,p->channels,static_cast<CameraModelType>(p->camera_model),s->queue);
        switch(p->channels){case 1:w->forward<1>(*p,s->queue);break;case 3:w->forward<3>(*p,s->queue);break;case 4:w->forward<4>(*p,s->queue);break;case 8:w->forward<8>(*p,s->queue);break;}
        if(p->rays)gsplat_lfs::geometry_camera_rays(ptr(p->rays),p->width,p->height,ptr(p->K),static_cast<CameraModelType>(p->camera_model),ptr(p->radial),ptr(p->tangential),ptr(p->thin_prism),s->queue);
        hip_check(hipGetLastError(),"HIP Gsplat forward");s->live=true;
    });
}
extern "C" int lfs_hip_application_gsplat_backward(LfsHipApplicationGsplat* s,const LfsHipGsplatBackward* b,LfsHipError* error) {
    using namespace lfs_hip_gsplat_detail;return call(s,error,[&] {
        require(b&&b->struct_bytes==sizeof(*b)&&s->live&&b->image&&b->alpha&&b->grad_means&&b->grad_quats&&b->grad_scales&&b->grad_opacity&&std::isfinite(b->flatten_weight),"Invalid HIP Gsplat backward descriptor");
        require(s->input.channels==1||(b->grad_sh0&&(!s->input.active_degree||b->grad_shN)),"Missing HIP Gsplat SH gradient binding");
        auto& w=*s->work;const auto& p=s->input;w.grad_params.zero(s->queue);w.grad_colors.zero(s->queue);w.grad_sh.zero(s->queue);
        require(b->expected_depth<=1,"Invalid HIP Gsplat expected-depth flag");
        if(b->expected_depth){require(p.channels==1||p.channels==4,"Expected depth requires a depth channel");const size_t offset=p.channels==1?0:size_t(p.width)*p.height*3;gsplat_lfs::expected_depth_bwd(ptr(p.image)+offset,ptr(p.alpha),ptr(b->image)+offset,ptr(b->alpha),p.width*p.height,s->queue);}
        switch(p.channels){case 1:w.backward<1>(p,*b,s->queue);break;case 3:w.backward<3>(p,*b,s->queue);break;case 4:w.backward<4>(p,*b,s->queue);break;case 8:w.backward<8>(p,*b,s->queue);break;}
        if(p.channels!=3)gsplat_lfs::geometry_features_bwd(ptr(p.means),ptr(p.quats),ptr(p.scales),ptr(p.view),w.grad_colors.as(),w.grad_params.as(),w.grad_params.as(p.count*3),p.count,1,p.channels,static_cast<CameraModelType>(p.camera_model),s->queue);
        gsplat_lfs::flatten_scale_grad(ptr(p.scales),w.grad_params.as(p.count*7),p.count,b->flatten_weight,s->queue);
        const uint64_t destinations[]={b->grad_means,b->grad_quats,b->grad_scales,b->grad_opacity};const unsigned attrs[]={3,4,3,1};size_t offset=0;
        for(unsigned group=0;group<4;++group){size_t n=size_t(p.count)*attrs[group];hipLaunchKernelGGL(accumulate,dim3(blocks(n)),dim3(256),0,s->queue,ptr(destinations[group]),w.grad_params.as(offset),n);offset+=n;}
        if(p.channels!=1){require(b->grad_sh0,"Missing HIP Gsplat DC gradient");gsplat_lfs::launch_spherical_harmonics_swizzled_bwd_kernel(p.active_degree,p.layout_degree,p.channels,p.active_degree?w.directions.as():nullptr,ptr(p.sh0),ptr(p.shN),nullptr,w.grad_colors.as(),p.count,w.bases,false,w.grad_sh.as(),nullptr,s->queue);hipLaunchKernelGGL(accumulate_sh,dim3(blocks(size_t(p.count)*w.bases*3)),dim3(256),0,s->queue,w.grad_sh.as(),ptr(b->grad_sh0),ptr(b->grad_shN),p.count,w.bases,(p.layout_degree+1)*(p.layout_degree+1));}
        hipLaunchKernelGGL(metrics,dim3(blocks(p.count)),dim3(256),0,s->queue,w.grad_params.as(),w.radii.as<int32_t>(),w.means2d.as(),ptr(b->densification),ptr(b->screen_share),p.count,p.width,p.height,!b->error_map);hip_check(hipGetLastError(),"HIP Gsplat backward");s->live=false;
    });
}
extern "C" int lfs_hip_application_gsplat_release(LfsHipApplicationGsplat* s,LfsHipError* error){return lfs_hip_gsplat_detail::call(s,error,[&]{hip_check(hipStreamSynchronize(s->queue),"HIP Gsplat release completion");s->live=false;s->input={};});}
extern "C" int lfs_hip_application_gsplat_info(LfsHipApplicationGsplat* s,uint64_t* bytes,uint64_t* intersections,LfsHipError* error){return lfs_hip_gsplat_detail::call(s,error,[&]{require(bytes&&intersections,"Missing HIP Gsplat counters");*bytes=s->work?s->work->budget.live:0;*intersections=s->work?s->work->intersections:0;});}
extern "C" int lfs_hip_application_gsplat_destroy(LfsHipApplicationGsplat* s,LfsHipError* error){if(!s)return 0;const auto code=boundary(nullptr,error,[&]{DeviceScope device(s->device);hip_check(hipStreamSynchronize(s->queue),"HIP Gsplat teardown completion");s->work.reset();});if(!code)delete s;return code;}

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""HIP copies of native mask kernels; the Core Tensor host ABI stays outside."""
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def write(name,text):
    path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or path.read_text(encoding='utf-8')!=text:path.write_text(text,encoding='utf-8')
def hipify(text):
    text=text.replace('<cuda_runtime.h>','<hip/hip_runtime.h>').replace('"core/cuda_error.hpp"','"hip_application_diagnostics.hpp"').replace('"kernel_stream.hpp"','"hip_application_stream.hpp"').replace('"training/kernels/warp_reduce.cuh"','"warp_reduce.cuh"')
    return re.sub(r'\bcuda(\w+)',r'hip\1',text)
domain=(a.source/'../include/lfs/training/ops/domain_types.hpp').read_text(encoding='utf-8')
enums=[]
for name in ('MaskPhotoMode','MaskOpacityMode'):
    found=re.findall(r'enum class '+name+r'\s*:\s*int\s*\{[^}]+\};',domain)
    if len(found)!=1:raise RuntimeError('Review mask mode declarations')
    enums+=found
mask_header=(a.source/'../include/lfs/training/ops/masks.hpp').read_text(encoding='utf-8')
constants=re.findall(r'inline constexpr float kMask\w+ = [^;]+;',mask_header)
if len(constants)!=2:raise RuntimeError('Review mask band constants')
write('hip_mask_types.hpp','#pragma once\nnamespace lfs::hip_mask_ops {\n'+'\n'.join(enums)+'\n}\nnamespace lfs::training::kernels {\n'+'\n'.join(constants)+'\n}\n')
for name in ('mask_preprocess.hpp','mask_preprocess.cu'):
    text=hipify((a.source/name).read_text(encoding='utf-8')).replace('"lfs/training/ops/masks.hpp"','"hip_mask_types.hpp"').replace('lfs::gpu_ops::','lfs::hip_mask_ops::')
    write(name,text)
root=a.source.parent.parent
image=(root/'core/include/core/tensor_image.hpp').read_text(encoding='utf-8')
match=re.search(r'struct UndistortParams\s*\{.*?\n    \};',image,re.S)
if not match:raise RuntimeError('Review distortion descriptor')
write('hip_mask_distortion.hpp','#pragma once\n#include "core/camera_types.h"\nnamespace lfs::core {\n'+match[0]+'\n}\n')
write('hip_mask_distortion_model.hpp',(root/'core/cuda/undistort/distortion_model.cuh').read_text(encoding='utf-8'))
text=(a.source/'mesh_mask.cu').read_text(encoding='utf-8')
begin=text.index('        constexpr int THREADS_PER_BLOCK')
end=text.index('        CameraData pack_camera(',begin)
body=text[begin:end]
start=text.index('        __global__ void splat_points_kernel(')
end=text.index('        // One separable pass',start)
point=text[start:end].replace('const MeshMaskCamera camera','const NativePointCamera camera')
prelude='''#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include "application_masks_api.h"
#include "hip_mask_distortion.hpp"
#include "hip_mask_distortion_model.hpp"
namespace lfs::training::kernels {
namespace {
struct NativePointCamera {float world_to_camera[12];float fx,fy,cx,cy;int width,height;};
'''
wrapper='''
} // namespace
void launch_hip_mesh(const LfsHipMeshCall& p,hipStream_t stream) {
    const auto& k=p.camera;CameraData c{};
    for(int r=0;r<3;++r)c.rows[r]=make_float4(k.rows[r*4],k.rows[r*4+1],k.rows[r*4+2],k.rows[r*4+3]);
    c.fx=k.fx;c.fy=k.fy;c.cx=k.cx;c.cy=k.cy;c.width=k.width;c.height=k.height;
    c.guard_min_x=k.guard_min_x;c.guard_max_x=k.guard_max_x;c.guard_min_y=k.guard_min_y;c.guard_max_y=k.guard_max_y;c.distorted=k.distorted!=0;
    auto& d=c.distortion;d.src_fx=k.src_fx;d.src_fy=k.src_fy;d.src_cx=k.src_cx;d.src_cy=k.src_cy;d.dst_fx=k.dst_fx;d.dst_fy=k.dst_fy;
    d.model_type=static_cast<lfs::core::CameraModelType>(k.model);d.num_distortion=k.num_distortion;for(int i=0;i<12;++i)d.distortion[i]=k.distortion[i];
    auto f=[](uint64_t v){return reinterpret_cast<float*>(static_cast<uintptr_t>(v));};auto i=[](uint64_t v){return reinterpret_cast<int32_t*>(static_cast<uintptr_t>(v));};
    auto* mask=reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(p.mask));
    if(p.operation==2) {
        NativePointCamera pc{};for(int j=0;j<12;++j)pc.world_to_camera[j]=k.rows[j];pc.fx=k.fx;pc.fy=k.fy;pc.cx=k.cx;pc.cy=k.cy;pc.width=k.width;pc.height=k.height;
        splat_points_kernel<<<(p.vertex_count+255)/256,256,0,stream>>>(f(p.vertices),p.vertex_count,pc,p.radius,c.distorted,d,mask);
    } else if(p.operation==0) {
        const auto groups=std::min((p.face_count+255)/256,65535);
        prepare_and_rasterize_small_kernel<<<groups,256,0,stream>>>(f(p.vertices),i(p.indices),p.vertex_count,p.face_count,c,p.z_near,f(p.samples),i(p.large_faces),i(p.counters),mask);
    } else {
        rasterize_large_kernel<<<std::min(p.face_count,2048),256,0,stream>>>(f(p.vertices),i(p.indices),p.vertex_count,c,p.z_near,f(p.samples),i(p.large_faces),i(p.counters),mask);
    }
}
} // namespace
'''
write('mesh_coverage.cu',prelude+body+point+wrapper)

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build HIP copies of repository bilateral/ISP kernels, preserving originals."""
import argparse,re
from pathlib import Path

FILES=('bilateral_grid.cuh','bilateral_grid_forward.cu','bilateral_grid_backward.cu',
       'bilateral_grid_tv.cu','memory_ops.cuh','warp_reduce.cuh','ppisp_math.cuh',
       'ppisp_math_bwd.cuh','ppisp.cuh','ppisp.cu','depth_loss.hpp','depth_loss.cu',
       'normal_loss.hpp','normal_loss.cu','normal_consistency_loss.hpp','normal_consistency_loss.cu')
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def write(name,text):
    path=a.output/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or path.read_text(encoding='utf-8')!=text:path.write_text(text,encoding='utf-8')
for name in FILES:
    text=(a.source/name).read_text(encoding='utf-8')
    if name=='depth_loss.cu':
        text=text.replace('"../rasterization/gsplat/Cameras.cuh"','"Cameras.cuh"')
        text=text.replace('#include "core/logger.hpp"','#include "hip_geometry_scratch.hpp"')
        text=re.sub(r'::lfs::core::cuda_record_range\(stream, "[^"]+"\)', '0', text)
        text=text.replace('LOG_ERROR(', 'lfs_hip_geometry_log(')
        begin=text.index('        float2* pairs_dev = nullptr;')
        end=text.index('        const auto anchor_ticket',begin)
        text=text[:begin]+'''        HipGeometryScratch scratch(stream);
        auto* pairs_dev = scratch.allocate<float2>(kMaxAnchorSamples);
        auto* count_dev = scratch.allocate<int>(1);
        LFS_CUDA_CHECK(cudaMemsetAsync(count_dev, 0, sizeof(int), stream));

'''+text[end:]
        text=re.sub(r'^\s*cudaFreeAsync\((?:pairs_dev|count_dev), stream\);', '', text, flags=re.M)
        text=text.replace('cudaMemcpyAsync(&pair_count, count_dev, sizeof(int), cudaMemcpyDeviceToHost, stream);',
                          'LFS_CUDA_CHECK(cudaMemcpyAsync(&pair_count, count_dev, sizeof(int), cudaMemcpyDeviceToHost, stream));')
        text=text.replace('cudaMemcpyAsync(pairs.data(), pairs_dev, sizeof(float2) * pair_count,\n                        cudaMemcpyDeviceToHost, stream);',
                          'LFS_CUDA_CHECK(cudaMemcpyAsync(pairs.data(), pairs_dev, sizeof(float2) * pair_count,\n                        cudaMemcpyDeviceToHost, stream));')
    text=text.replace('<cuda_runtime.h>','<hip/hip_runtime.h>').replace('<cub/cub.cuh>','<hipcub/hipcub.hpp>').replace('cub::','hipcub::')
    text=text.replace('"core/cuda_error.hpp"','"hip_application_diagnostics.hpp"').replace('"kernel_stream.hpp"','"hip_application_stream.hpp"')
    text=re.sub(r'#include "training/kernels/([^"]+)"',r'#include "\1"',text)
    text=re.sub(r'\bcuda(\w+)',r'hip\1',text)
    text=text.replace('__CUDA_ARCH__','__HIP_DEVICE_COMPILE__').replace('__CUDACC__','__HIPCC__')
    text=re.sub(r'\b0xffffffff[uU]?\b','0x00000000ffffffffull',text)
    if name=='ppisp_math.cuh':
        # HIP already defines the vector arithmetic. Keep matrix overloads.
        removed=[0]
        def remove_vector(m):
            if 'float2x2' in m[0] or 'float3x3' in m[0]:return m[0]
            removed[0]+=1;return ''
        text=re.sub(r'__device__ __forceinline__ float[23] operator[+*\-]\([^)]*\)\s*\{[^{}]*\}',remove_vector,text)
        if removed[0]!=8:raise RuntimeError('Review changed ISP vector operator boundary')
    if re.search(r'#include[^\n]*(?:cuda|training/kernels/|core/)',text):raise RuntimeError('Unconverted dependency in '+name)
    write(name,text)
write('lfs/training/ops/geometry_types.hpp',(a.source/'../include/lfs/training/ops/geometry_types.hpp').read_text(encoding='utf-8'))
write('hip_geometry_scratch.hpp','''/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "hip_application_diagnostics.hpp"
#include <cstdio>
#include <cstdarg>
#include <vector>
inline void lfs_hip_geometry_log(const char* format,...) {
    std::fputs("[error] HIP geometry: ",stderr);va_list args;va_start(args,format);std::vfprintf(stderr,format,args);va_end(args);std::fputc('\\n',stderr);
}
// Only the bounded startup anchor collector needs temporary allocations.
// Failed completion cannot release memory still referenced by the GPU.
class HipGeometryScratch {
    hipStream_t stream;std::vector<void*> allocations;
public:
    explicit HipGeometryScratch(hipStream_t s):stream(s){allocations.reserve(2);}
    template<class T>T* allocate(size_t count) {
        void* p=nullptr;lfs_hip_color_check(hipMalloc(&p,sizeof(T)*count),"HIP anchor scratch allocation");
        allocations.push_back(p);return static_cast<T*>(p);
    }
    ~HipGeometryScratch() {
        if(allocations.empty())return;
        auto status=hipStreamSynchronize(stream);
        if(status!=hipSuccess){lfs_hip_geometry_log("retaining anchor scratch after failed completion: %s",hipGetErrorString(status));return;}
        for(auto* p:allocations)if(auto status=hipFree(p);status!=hipSuccess)lfs_hip_geometry_log("anchor scratch release failed: %s",hipGetErrorString(status));
    }
};
''')
write('hip_application_stream.hpp','''/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <hip/hip_runtime.h>
#include <stdexcept>
namespace lfs {
inline hipStream_t resolve_stream(hipStream_t stream) {
    if(!stream)throw std::invalid_argument("Native HIP color kernels require an explicit borrowed queue");
    return stream;
}
inline hipStream_t resolve_stream(void* stream){return resolve_stream(reinterpret_cast<hipStream_t>(stream));}
}
''')
write('hip_application_diagnostics.hpp','''/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>
inline void lfs_hip_color_check(hipError_t status,const char* what) {
    if(status!=hipSuccess)throw std::runtime_error(std::string(what)+": "+hipGetErrorString(status));
}
#define LFS_CUDA_CHECK(call) lfs_hip_color_check((call),#call)
#define LFS_CUDA_CHECK_MSG(call,...) lfs_hip_color_check((call),#call)
#define LFS_CUDA_LAUNCH_CHECK(stream,name) lfs_hip_color_check(hipGetLastError(),name)
#define LFS_CUDA_AWAIT(ticket,call,name) lfs_hip_color_check((call),name)
''')

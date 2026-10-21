/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/structure_hip.hpp"
#include "structure_gpu.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_structure_api.h"
#include <climits>
#include <cstring>
namespace lfs::training {
namespace {
struct Backend {
    static uint64_t address(const core::Tensor& tensor) {
        if(!tensor.is_valid() || !tensor.numel())return 0;
        if(core::gpu_backend_of(tensor)!=core::GpuBackend::HIP || !tensor.is_contiguous())
            throw std::invalid_argument("HIP structure requires contiguous native operands");
        (void)const_cast<core::Tensor&>(tensor).data_ptr();
        auto& ops=static_cast<core::internal::HipBackendOps&>(core::internal::backend_ops(core::GpuBackend::HIP));
        return ops.storage_address(core::internal::storage_ref(tensor),tensor.bytes());
    }
    static void launch(uint32_t operation,const structure_detail::Parameters& p,
                       std::initializer_list<const core::Tensor*> reads,
                       std::initializer_list<const core::Tensor*> writes,size_t items) {
        if(!items || items>INT_MAX || p.width<=0 || p.height<=0 || uint64_t(p.width)*p.height>INT_MAX)
            throw std::invalid_argument("HIP structure dimensions exceed the native index range");
        hip::ApplicationCall owner;
        for(const auto* tensor:reads)if(tensor->is_valid())owner.bind(*tensor,tensor->dtype());
        for(const auto* tensor:writes)if(tensor->is_valid())owner.bind(*tensor,tensor->dtype());
        LfsHipStructureCall call{};
        call.struct_bytes=sizeof(call);call.operation=operation;call.device=owner.queue.device;
        call.queue=owner.queue.native_queue;call.dispatch_count=uint32_t(items);
        static_assert(sizeof(call.parameters)==sizeof(p));
        std::memcpy(&call.parameters,&p,sizeof(p));
        const auto fn=owner.module->symbol<decltype(&lfs_hip_application_structure)>("lfs_hip_application_structure");
        LfsHipError error{};
        if(int code=fn(&call,&error))throw hip::TrainingError(code,error);
    }
};
}
const gpu_ops::StructureOps& hip_structure_ops() {
    return structure_detail::Implementation<Backend>::table();
}
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_module.hpp"
#include "session.hpp"
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include <cstdlib>
#include <vector>
#ifndef LFS_HIP_TRAINING_DEFAULT_MODULE
#define LFS_HIP_TRAINING_DEFAULT_MODULE ""
#endif
namespace lfs::training::hip {
struct ApplicationCall {
    core::internal::HipBackendOps& ops;
    core::internal::HipQueueLease queue;
    std::shared_ptr<ApplicationModule> module;
    std::vector<core::Tensor> retained;
    ApplicationCall() : ops(static_cast<core::internal::HipBackendOps&>(core::internal::backend_ops(core::GpuBackend::HIP))), queue(ops.execution_lease({})) {
        const char* path=std::getenv("LFS_HIP_TRAINING_MODULE");
        module=cached_application_module(path ? path : LFS_HIP_TRAINING_DEFAULT_MODULE);
        ops.retain_application_module(module);
    }
    uint64_t bind(const core::Tensor& tensor, core::DataType dtype) {
        if(!tensor.is_valid() || !tensor.numel()) return 0;
        if(core::gpu_backend_of(tensor)!=core::GpuBackend::HIP || !tensor.is_contiguous() || tensor.dtype()!=dtype)
            throw std::invalid_argument("HIP application operation requires contiguous native tensors of the declared dtype");
        (void)const_cast<core::Tensor&>(tensor).data_ptr();
        retained.push_back(tensor);
        core::pin_operands({&retained.back()});
        return ops.storage_address(core::internal::storage_ref(retained.back()),retained.back().bytes());
    }
};
}

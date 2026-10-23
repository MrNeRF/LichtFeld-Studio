/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "external_access.hpp"
#include "hip_backend_ops.hpp"
#include "../../internal/tensor_impl.hpp"
#include <core/tensor.hpp>
#include <core/tensor_execution.hpp>
#include <core/memory_pressure.hpp>
#include <cstdio>
#include <stdexcept>

namespace lfs::core::internal {
int hip_tensor_scratch_allocate(void*, int device, uint64_t producer_queue, uint64_t bytes,
                                uint64_t* address, void** owner, LfsHipError* error) noexcept {
    if(address)*address=0;
    if(owner)*owner=nullptr;
    if(error)*error={};
    const auto fail=[&](int code,const char* message) {
        if(error){error->native_code=code;std::snprintf(error->message,sizeof(error->message),"%s",message);}
        return 1;
    };
    try {
        if(!address||!owner||!bytes||bytes>SIZE_MAX)
            throw std::invalid_argument("Invalid HIP Tensor scratch allocation");
        const TensorExecutionTarget::Scope scope(TensorExecutionTarget::default_queue(GpuBackend::HIP));
        auto& backend=static_cast<HipBackendOps&>(backend_ops(GpuBackend::HIP));
        const auto lease=backend.execution_lease({});
        if(lease.device!=device)
            throw std::invalid_argument("HIP scratch device differs from the Tensor context");
        auto storage=std::make_unique<Tensor>(Tensor::empty({static_cast<size_t>(bytes)},Device::GPU,DataType::UInt8));
        const auto native=backend.storage_address(storage_ref(*storage),static_cast<size_t>(bytes));
        // The normal pool orders reuse on Core's queue. Standalone reference
        // sessions can have another producer, which must not race a preceding
        // Core user of a recycled allocation. Application training uses the
        // Core queue and needs no additional fence here.
        if(producer_queue!=lease.native_queue)
            TensorExecutionTarget::default_queue(GpuBackend::HIP).wait();
        *address=native;
        *owner=storage.release();
        return 0;
    } catch(const MemoryAllocationError& e){return fail(2,e.what());}
      catch(const HipBackendError& e){return fail(e.native_code,e.what());}
      catch(const std::bad_alloc&){return fail(2,"HIP Tensor scratch host allocation failed");}
      catch(const std::exception& e){ /* LFS-CENSUS-OK(empty-catch): Translate to the native C ABI error/status and keep failed output owners cleared. */ return fail(-1,e.what());}
      catch(...){ /* LFS-CENSUS-OK(empty-catch): Translate to the native C ABI error/status and keep failed output owners cleared. */ return fail(-1,"HIP Tensor scratch allocation failed");}
}
void hip_tensor_scratch_release(void*,void* owner) noexcept {
    delete static_cast<Tensor*>(owner);
}
HipExternalAccess& hip_external_access() {
    // Backend singletons may release allocations during static teardown.
    // Keep the CPU-only coordinator alive; it owns no GPU allocations.
    static auto* const access=new HipExternalAccess;
    return *access;
}
int hip_external_call_begin(void*,LfsHipError* error) noexcept {
    try {hip_external_access().begin_hip();return 0;}
    catch(const std::exception& failure){ /* LFS-CENSUS-OK(empty-catch): Translate to the native C ABI error/status and keep failed output owners cleared. */ if(error){error->native_code=-1;std::snprintf(error->message,sizeof(error->message),"%s",failure.what());}return 1;}
    catch(...){ /* LFS-CENSUS-OK(empty-catch): Translate to the native C ABI error/status and keep failed output owners cleared. */ if(error){error->native_code=-1;std::snprintf(error->message,sizeof(error->message),"HIP external access failed");}return 1;}
}
void hip_external_call_end(void*) noexcept {hip_external_access().end_hip();}
}

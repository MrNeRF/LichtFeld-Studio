/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "external_access.hpp"
#include "hip_backend_ops.hpp"
#include "../../internal/tensor_impl.hpp"
#include <core/tensor.hpp>
#include <core/tensor_execution.hpp>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

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
    } catch(const HipBackendError& e){return fail(e.native_code,e.what());}
      catch(const std::bad_alloc&){return fail(2,"HIP Tensor scratch host allocation failed");}
      catch(const std::exception& e){return fail(-1,e.what());}
      catch(...){return fail(-1,"HIP Tensor scratch allocation failed");}
}
void hip_tensor_scratch_release(void*,void* owner) noexcept {
    delete static_cast<Tensor*>(owner);
}
namespace {
struct ThreadAccess { size_t hip_depth=0; bool consumer=false, synchronizing=false; };
thread_local std::unordered_map<HipExternalAccess*,ThreadAccess> thread_access;
}
void HipExternalAccess::register_storage(const std::shared_ptr<HipExternalStorage>& storage) {
    if(!storage)throw std::invalid_argument("Missing HIP external storage");
    std::lock_guard lock(mutex_);
    std::erase_if(storages_,[](const auto& p){return p.expired();});
    storages_.push_back(storage);
}
void HipExternalAccess::begin_hip() {
    auto& local=thread_access[this];
    if(local.hip_depth){++local.hip_depth;return;}
    if(local.consumer&&!local.synchronizing)
        throw std::logic_error("HIP work cannot be submitted inside a Vulkan consumer recording scope");
    if(local.synchronizing){++local.hip_depth;return;}
    std::unique_lock lock(mutex_);
    ++waiting_hip_;
    try {changed_.wait(lock,[&]{return !transitioning_&&!consumer_&&
        (!waiting_consumers_||(!prefer_consumer_&&active_hip_==0));});}
    catch(...){--waiting_hip_;changed_.notify_all();throw;}
    --waiting_hip_;prefer_consumer_=true;
    if(active_hip_){++active_hip_;local.hip_depth=1;return;}
    transitioning_=true;
    std::vector<std::shared_ptr<HipExternalStorage>> live;
    try {
        for(auto& weak:storages_)if(auto storage=weak.lock())live.push_back(std::move(storage));
        lock.unlock();
        for(auto& storage:live)storage->release_to_hip();
        lock.lock();
        transitioning_=false;active_hip_=1;local.hip_depth=1;
    } catch(...) {
        if(!lock.owns_lock())lock.lock();
        transitioning_=false;changed_.notify_all();throw;
    }
    changed_.notify_all();
}
void HipExternalAccess::end_hip() noexcept {
    auto found=thread_access.find(this);
    assert(found!=thread_access.end()&&found->second.hip_depth);
    if(found==thread_access.end()||!found->second.hip_depth)std::terminate();
    auto& local=found->second;
    if(--local.hip_depth||local.synchronizing)return;
    std::lock_guard lock(mutex_);
    assert(active_hip_);--active_hip_;changed_.notify_all();
}
void HipExternalAccess::begin_consumer(std::span<const std::shared_ptr<HipExternalStorage>> storages,
                                       const std::function<void()>& synchronize_hip) {
    auto& local=thread_access[this];
    if(local.hip_depth||local.consumer)throw std::logic_error("Nested HIP/Vulkan consumer scopes are invalid");
    std::unique_lock lock(mutex_);
    ++waiting_consumers_;
    try {changed_.wait(lock,[&]{return !transitioning_&&!consumer_&&active_hip_==0&&
        (!waiting_hip_||prefer_consumer_);});}
    catch(...){--waiting_consumers_;changed_.notify_all();throw;}
    --waiting_consumers_;prefer_consumer_=false;
    transitioning_=true;local.synchronizing=true;
    try {
        lock.unlock();
        synchronize_hip();
        for(auto& storage:storages) {
            if(!storage)throw std::invalid_argument("Missing Vulkan consumer storage");
            storage->acquire_to_vulkan();
        }
        lock.lock();
        local.synchronizing=false;local.consumer=true;consumer_=true;transitioning_=false;
    } catch(...) {
        if(!lock.owns_lock())lock.lock();
        local.synchronizing=false;transitioning_=false;changed_.notify_all();throw;
    }
    changed_.notify_all();
}
void HipExternalAccess::end_consumer() noexcept {
    auto found=thread_access.find(this);
    assert(found!=thread_access.end()&&found->second.consumer);
    if(found==thread_access.end()||!found->second.consumer)std::terminate();
    std::lock_guard lock(mutex_);
    found->second.consumer=false;consumer_=false;changed_.notify_all();
}
HipExternalAccess& hip_external_access() {
    // Backend singletons may release allocations during static teardown.
    // Keep the CPU-only coordinator alive; it owns no GPU allocations.
    static auto* const access=new HipExternalAccess;
    return *access;
}
int hip_external_call_begin(void*,LfsHipError* error) noexcept {
    try {hip_external_access().begin_hip();return 0;}
    catch(const std::exception& failure){if(error){error->native_code=-1;std::snprintf(error->message,sizeof(error->message),"%s",failure.what());}return 1;}
    catch(...){if(error){error->native_code=-1;std::snprintf(error->message,sizeof(error->message),"HIP external access failed");}return 1;}
}
void hip_external_call_end(void*) noexcept {hip_external_access().end_hip();}
}

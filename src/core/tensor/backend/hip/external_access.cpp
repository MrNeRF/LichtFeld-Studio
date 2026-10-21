/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "external_access.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

namespace lfs::core::internal {
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

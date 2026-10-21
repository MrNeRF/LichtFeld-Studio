/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor/backend/hip/external_access.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <cstdio>
using namespace lfs::core::internal;
using namespace std::chrono_literals;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct Storage final:HipExternalStorage {
    bool vulkan=false,fail_release=false,fail_acquire=false;
    unsigned releases=0,acquires=0;
    void release_to_hip() override {if(fail_release)throw std::runtime_error("injected release failure");if(vulkan){vulkan=false;++releases;}}
    void acquire_to_vulkan() override {if(fail_acquire)throw std::runtime_error("injected acquire failure");if(!vulkan){vulkan=true;++acquires;}}
};
int main() try {
    HipExternalAccess access;auto storage=std::make_shared<Storage>();access.register_storage(storage);
    const std::array<std::shared_ptr<HipExternalStorage>,1> storages{storage};
    access.begin_hip();access.begin_hip();
    std::promise<void> started,entered,finish;
    auto started_future=started.get_future(),entered_future=entered.get_future(),finish_future=finish.get_future();
    auto consumer=std::async(std::launch::async,[&]{started.set_value();access.begin_consumer(storages,[&]{access.begin_hip();access.end_hip();});entered.set_value();finish_future.wait();access.end_consumer();});
    started_future.wait();require(entered_future.wait_for(30ms)==std::future_status::timeout,"Consumer entered during nested HIP submission");
    access.end_hip();require(entered_future.wait_for(30ms)==std::future_status::timeout,"Nested HIP work released its outer submission");
    access.end_hip();require(entered_future.wait_for(2s)==std::future_status::ready,"Consumer did not resume after HIP submission");
    std::atomic<bool> submitted=false;std::promise<void> hip_started;auto hip_started_future=hip_started.get_future();
    auto producer=std::async(std::launch::async,[&]{hip_started.set_value();access.begin_hip();require(submitted.load(),"HIP began before the consumer submitted its commands");access.end_hip();});
    hip_started_future.wait();require(producer.wait_for(30ms)==std::future_status::timeout,"HIP entered while Vulkan commands were being recorded");
    submitted=true;finish.set_value();consumer.get();require(producer.wait_for(2s)==std::future_status::ready,"HIP failed to resume after consumer submission");producer.get();
    require(!storage->vulkan&&storage->releases==1&&storage->acquires==1,"Queue ownership was not transferred back to HIP");
    access.begin_consumer(storages,[]{});
    bool rejected=false;try{access.begin_hip();}catch(const std::logic_error&){rejected=true;}
    require(rejected,"Same-thread HIP work inside a consumer recording scope was accepted");
    access.end_consumer();storage->fail_release=true;
    rejected=false;try{access.begin_hip();}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Injected release failure disappeared");storage->fail_release=false;access.begin_hip();access.end_hip();
    storage->fail_acquire=true;rejected=false;try{access.begin_consumer(storages,[]{});}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Injected acquire failure disappeared");storage->fail_acquire=false;access.begin_hip();access.end_hip();
    std::atomic<unsigned> active=0;std::array<std::future<void>,4> workers;
    for(auto& worker:workers)worker=std::async(std::launch::async,[&]{for(unsigned i=0;i<200;++i){access.begin_hip();++active;std::this_thread::yield();--active;access.end_hip();}});
    for(unsigned i=0;i<200;++i){access.begin_consumer(storages,[&]{require(active==0,"Consumer synchronization overlapped HIP submission");});require(active==0,"Consumer recording overlapped HIP submission");access.end_consumer();}
    for(auto& worker:workers)worker.get();
    access.begin_hip();access.end_hip();
    std::puts("PASS HIP external access: nested calls, recording/submission exclusion, queue ownership, sync reentrancy, error recovery and concurrent producers/consumer");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}

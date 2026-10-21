/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_completion.hpp>
#include <core/tensor_execution.hpp>
#include <cstdio>
#include <stdexcept>
using namespace lfs;
using namespace core;
namespace lfs::training {
unsigned other_tables=0;
const TrainingOps& cuda_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::CUDA}; return t; }
const TrainingOps& metal_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Metal}; return t; }
const TrainingOps& vulkan_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Vulkan}; return t; }
}
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
int main() try {
    const auto configured=configured_gpu_backend(); const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.session && !table.fast && !training::other_tables && !gpu_backend_live(GpuBackend::HIP),"Session discovery initialized another backend");
    const auto& ops=*table.session;
    require(!ops.last_error() && !ops.arena_memory_info(),"Idle HIP session reports nonexistent fault/arena");
    ops.sample_memory(); ops.resize_arena("idle",true); ops.reset_arena(); ops.profile(true); ops.profile(false);
    require(!gpu_backend_live(GpuBackend::HIP),"Idle session hooks initialized HIP context");
    const auto target=TensorExecutionTarget::default_queue(GpuBackend::HIP);
    const auto borrow=ops.borrow_idle_arena(1<<20,1024,target);
    require(!borrow.data && !borrow.owner && !borrow.bytes && !ops.zero_idle_arena(borrow,1024,target),"Session lent nonexistent/shared live raster arena");
    ops.release_idle_arena(borrow,target);
    bool rejected=false;
    try { (void)ops.borrow_idle_arena(1024,1024,TensorExecutionTarget::default_queue(GpuBackend::CUDA)); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"HIP session accepted foreign arena target");
    require(ops.set_arena_timeout(100)==0 && ops.set_arena_timeout(200)==100 && ops.set_arena_timeout(0)==200,"Session timeout restoration failed");
    {
        const GpuBackendScope scope(GpuBackend::HIP);
        auto tensor=Tensor::ones({4096},Device::GPU);
        auto result=tensor.mul(2.f);
        ops.sample_memory(); require(ops.device_baseline_bytes()>=tensor.bytes(),"HIP session did not report native memory");
        require(ops.allocation_bytes(tensor.bytes())>=tensor.bytes(),"HIP allocation accounting undercounts");
        require(!ops.last_error(),"Healthy native queue reported an error");
        target.wait();
        require(result.cpu().to_vector()==std::vector<float>(4096,2),"Session health/memory hooks changed queued work");
        ops.resize_arena("training boundary",true);
        require(tensor.cpu().to_vector()==std::vector<float>(4096,1),"Session trim changed live Tensor");
    }
    ops.resize_arena("shutdown",true);
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"Session changed default or leaked resources");
    std::puts("PASS HIP application session: lazy discovery, native memory/queue health, no live-frame arena borrowing, target rejection, timeout restoration, trim and teardown");
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"HIP SESSION CONTRACT FAILED: %s\n",e.what()); return 1; }

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include <algorithm>
#include <cstdio>
#include <lfs/training/ops/registry.hpp>
#include <stdexcept>
namespace lfs::training {
    unsigned existing_table_requests = 0;
    const TrainingOps& cuda_training_ops_table() {
        ++existing_table_requests;
        static const TrainingOps value{.backend = core::GpuBackend::CUDA};
        return value;
    }
    const TrainingOps& vulkan_training_ops_table() {
        ++existing_table_requests;
        static const TrainingOps value{.backend = core::GpuBackend::Vulkan};
        return value;
    }
    const TrainingOps& metal_training_ops_table() {
        ++existing_table_requests;
        static const TrainingOps value{.backend = core::GpuBackend::Metal};
        return value;
    }
} // namespace lfs::training
int main() try {
    using namespace lfs;
    const auto before = core::configured_gpu_backend();
    const auto& table = training::training_ops(core::GpuBackend::HIP);
    if (table.backend != core::GpuBackend::HIP || training::existing_table_requests)
        throw std::runtime_error("HIP training fell through to another backend table");
    training::FamilySet required;
    for (auto family : {training::Family::Session, training::Family::Gsplat, training::Family::Photometric, training::Family::Adam, training::Family::Refine})
        required.set(size_t(family));
    const auto missing = training::missing_training_families(table, required);
    if (missing.size() != 5 || !training::unavailable_training_family(core::GpuBackend::HIP, training::Family::Gsplat))
        throw std::runtime_error("Incomplete HIP training was admitted");
    const auto reason = training::training_unavailable_message(core::GpuBackend::HIP, missing);
    if (reason.find("HIP training") == std::string::npos || reason.find("Vulkan") != std::string::npos)
        throw std::runtime_error("Incorrect HIP training diagnostic");
    training::FamilySet popspa_required;
    popspa_required.set(size_t(training::Family::Popspa));
    const auto missing_popspa = training::missing_training_families(table, popspa_required);
    if (missing_popspa.size() != 1 || missing_popspa.front() != "Popspa" ||
        !training::unavailable_training_family(core::GpuBackend::HIP, training::Family::Popspa))
        throw std::runtime_error("POPSpa was admitted without its HIP callbacks");
    const auto popspa_reason = training::training_unavailable_message(core::GpuBackend::HIP, missing_popspa);
    if (popspa_reason.find("Popspa") == std::string::npos || training::existing_table_requests)
        throw std::runtime_error("Missing POPSpa did not produce an explicit HIP-only diagnostic");
    if (!core::set_default_gpu_backend(core::GpuBackend::HIP) || core::configured_gpu_backend() != core::GpuBackend::HIP || core::gpu_backend_live(core::GpuBackend::HIP))
        throw std::runtime_error("Explicit HIP configuration failed or initialized a context before training admission");
    if (std::ranges::find(core::kCompiledGpuBackends, core::GpuBackend::HIP) != core::kCompiledGpuBackends.end())
        throw std::runtime_error("Experimental HIP adapter was advertised as an automatic candidate");
    std::puts("PASS actual training registry: missing HIP families rejected, explicit opt-in without context initialization or automatic candidacy, no other backend fallback");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP ADMISSION FAILED: %s\n", error.what());
    return 1;
}

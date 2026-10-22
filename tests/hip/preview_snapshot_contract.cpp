/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../src/training/components/hip_preview_snapshots.hpp"
#include <cstdio>
#include <thread>
#include <vector>

using namespace lfs::core;
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
static auto fields(const SplatData& model) {
    return std::array<const Tensor*, 8>{&model.means_raw(), &model.sh0_raw(),
        &model.rotation_raw(), &model.scaling_raw(), &model.opacity_raw(),
        &model.shN_raw(), &model.shN_value_bounds(), &model.deleted()};
}
static SplatData make_model(size_t count, float value, DataType dtype) {
    SplatData model;
    model.means_raw() = Tensor::full({count, 3}, value, Device::GPU, dtype);
    model.sh0_raw() = Tensor::full({count, 1, 3}, value, Device::GPU, dtype);
    model.rotation_raw() = Tensor::full({count, 4}, value, Device::GPU, dtype);
    model.scaling_raw() = Tensor::full({count, 3}, value, Device::GPU, dtype);
    model.opacity_raw() = Tensor::full({count, 1}, value, Device::GPU, dtype);
    model.shN_raw() = Tensor::empty({0}, Device::GPU);
    model.deleted() = Tensor::zeros({count}, Device::GPU, DataType::Bool);
    return model;
}
static void verify(const SplatData& model, size_t count, float expected, DataType dtype) {
    require(model.size() == count && model.get_active_sh_degree() == 0, "Snapshot metadata changed");
    size_t index = 0;
    for (const auto* field : fields(model)) {
        if (field->is_valid()) {
            require(field->device() == Device::CPU && !gpu_backend_of(*field), "Snapshot retained producer storage");
            if (index < 5) {
                const auto reference = Tensor::full(field->shape(), expected, Device::CPU, dtype);
                require(field->dtype() == dtype && field->bytes() == reference.bytes() &&
                        std::memcmp(field->data_ptr(), reference.data_ptr(), field->bytes()) == 0,
                        "Snapshot mixed fields or producer generations");
            }
        }
        ++index;
    }
}
int main(int argc, char**) try {
    Logger::get().init(LogLevel::Info);
    require(bool(set_default_gpu_backend(GpuBackend::HIP)), "Cannot select HIP");
    const size_t count = argc > 1 ? 1000000 : 257;
    for (const auto dtype : {DataType::Float32, DataType::Float16}) {
        TensorWorkQueue queue(GpuBackend::HIP);
        SplatData model;
        const auto replace = [&](float value) {
            queue.execute([&] {
                model = make_model(count, value, dtype);
            });
        };
        replace(3.f);
        lfs::training::HipPreviewSnapshots previews;
        require(!previews.request(), "Preview exists before the first complete capture");
        previews.advance(model, queue);
        replace(5.f);
        queue.wait();
        previews.advance(model, queue);
        auto first = previews.request();
        require(bool(first), "Completed preview was not published");
        verify(*first, count, 3.f, dtype);

        if (count == 257 && dtype == DataType::Float32) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2100));
            require(previews.request() == first, "A delayed producer invalidated a complete navigation snapshot");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        replace(7.f);
        previews.advance(model, queue);
        replace(11.f); // Source allocations may retire before publication.
        queue.wait();
        previews.advance(model, queue);
        auto second = previews.request();
        require(second && second != first, "Next complete generation was not published");
        require(first->param_layout_generation() && first->param_layout_generation() != second->param_layout_generation(),
                "Immutable preview generations can alias after allocator address reuse");
        verify(*first, count, 3.f, dtype);
        verify(*second, count, 7.f, dtype);

        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        previews.advance(model, queue);
        previews.suspend("contract: optional consumer publication rejected");
        require(!previews.request(), "Suspended publication did not restore the live-model path");
        previews.advance(model, queue);
        previews.reset(); // Drain an optional capture before producer teardown.
        require(!previews.request(), "Reset exposed a previous project snapshot");
        verify(*first, count, 3.f, dtype);
        verify(*second, count, 7.f, dtype);
        queue.wait();
    }
    {
        TensorWorkQueue queue(GpuBackend::HIP);
        SplatData model;
        queue.execute([&] { model = make_model(257, 3.f, DataType::Float32); });
        lfs::training::HipPreviewSnapshots previews;
        (void)previews.request();
        previews.advance(model, queue);
        queue.wait();
        previews.advance(model, queue);
        auto retained = previews.request();
        require(bool(retained), "Budget fixture has no initial snapshot");
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        queue.execute([&] { model = make_model(2000000, 7.f, DataType::Float32); }).wait();
        const void* producer = model.means_raw().data_ptr();
        previews.advance(model, queue);
        require(!previews.request(), "Over-budget preview exposed stale model metadata");
        require(model.size() == 2000000 && model.means_raw().data_ptr() == producer,
                "Optional preview failure changed the producer model");
        verify(*retained, 257, 3.f, DataType::Float32);
        previews.reset();
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)), "HIP snapshot leaked producer storage");
    std::puts("PASS HIP preview snapshots: FP32/FP16, complete generations, source replacement, reset and retained CPU readers");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP PREVIEW FAILED: %s\n", error.what());
    return 1;
}

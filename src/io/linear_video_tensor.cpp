// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/gpu_kernel_module.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor_upload.hpp"
#include "linear_video_program.hpp"
#include "media/video_color.hpp"
#include <cstring>
namespace lfs::io {
    namespace {
        class LinearTensorRenderer final : public media::detail::LinearVideoRenderer {
            std::unique_ptr<core::GpuKernelModule> program_;
            core::GpuBackend backend_;

        public:
            LinearTensorRenderer(std::unique_ptr<core::GpuKernelModule> program, core::GpuBackend backend) : program_(std::move(program)), backend_(backend) {}
            media::SinkResult convert(std::span<const uint8_t> bytes, const media::detail::VideoColorParameters& color,
                                      int width, int height, std::span<uint8_t> output) override {
                using core::DataType;
                using core::Device;
                using core::GpuKernelModule;
                using core::Tensor;
                core::GpuBackendScope scope(backend_);
                auto planes = Tensor::empty({bytes.size()}, Device::GPU, DataType::UInt8);
                auto constants = Tensor::empty({sizeof(color)}, Device::GPU, DataType::UInt8);
                auto result = Tensor::empty({static_cast<size_t>(height), static_cast<size_t>(width), 3}, Device::GPU, DataType::Float32);
                core::TensorUpload plane_upload, color_upload;
                plane_upload.enqueue(planes, std::as_bytes(bytes), nullptr);
                color_upload.enqueue(constants, std::as_bytes(std::span(&color, 1)), nullptr);
                plane_upload.wait();
                color_upload.wait();
                struct Parameters {
                    uint64_t planes = 0, color = 0, output = 0;
                    uint32_t width, height;
                };
                const Parameters parameters{0, 0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
                const std::array bindings{GpuKernelModule::Binding{0, &planes}, GpuKernelModule::Binding{8, &constants},
                                          GpuKernelModule::Binding{16, &result, GpuKernelModule::Access::ReadWrite}};
                auto dispatched = program_->dispatch({.function = "linearVideo", .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings}, .groups = {GpuKernelModule::groups_for(width, 8), GpuKernelModule::groups_for(height, 8), 1}, .group = {8, 8, 1}});
                if (!dispatched)
                    return media::SinkResult::failure(std::move(dispatched).error());
                auto host = result.cpu();
                std::memcpy(output.data(), host.ptr<float>(), output.size());
                return {};
            }
        };
    } // namespace
    std::unique_ptr<media::detail::LinearVideoRenderer> createLinearTensorRenderer() {
        const auto backend = core::default_gpu_backend();
        auto loaded = core::GpuKernelModule::load(linear_video_program_entries(), backend);
        if (!loaded) {
            LOG_INFO("Linear video tensor backend unavailable: {}", loaded.error().detail());
            return nullptr;
        }
        return std::make_unique<LinearTensorRenderer>(std::move(*loaded), backend);
    }
} // namespace lfs::io

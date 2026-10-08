// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/tensor_backend.hpp"
#include "core/tensor_color.hpp"
#include "core/tensor_upload.hpp"
#include "media/video_color.hpp"
#include <cstring>
#include <format>
namespace lfs::io {
    namespace {
        class LinearTensorRenderer final : public media::detail::LinearVideoRenderer {
            core::GpuBackend backend_;

        public:
            explicit LinearTensorRenderer(core::GpuBackend backend) : backend_(backend) {}
            media::SinkResult convert(std::span<const uint8_t> bytes, const media::detail::VideoColorParameters& color,
                                      int width, int height, std::span<uint8_t> output) override {
                core::GpuBackendScope scope(backend_);
                auto planes = core::Tensor::empty({bytes.size()}, core::Device::GPU, core::DataType::UInt8);
                core::TensorUpload upload;
                upload.enqueue(planes, std::as_bytes(bytes), nullptr);
                upload.wait();
                auto result = core::video_to_linear_rgb(planes, color, width, height);
                if (!result)
                    return media::SinkResult::failure(std::move(result).error());
                if (output.size() != result->bytes())
                    return media::SinkResult::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Tensor, .detail = std::format("Linear output storage must match tensor bytes (output={}, tensor={})", output.size(), result->bytes()), .detection = LFS_SOURCE_SITE_CURRENT()}));
                auto host = result->cpu();
                std::memcpy(output.data(), host.ptr<float>(), output.size());
                return {};
            }
        };
    } // namespace
    std::unique_ptr<media::detail::LinearVideoRenderer> createLinearTensorRenderer() {
        const auto backend = core::default_gpu_backend();
        return core::gpu_backend_available(backend) ? std::make_unique<LinearTensorRenderer>(backend) : nullptr;
    }
} // namespace lfs::io

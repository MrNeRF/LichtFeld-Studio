// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/tensor_color.hpp"
#include "core/crash_handler.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_upload.hpp"
#include "linear_video_program.hpp"
#include "rgb_to_yuv_program.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <format>
#include <map>
#include <mutex>
#include <tbb/parallel_for.h>
namespace lfs::core {
    namespace {
        using std::floor;
        using std::isnan;
        using std::max;
        using std::min;
#include "programs/rgb_to_yuv_math.h"

        Result<void> validateRgb(const Tensor& rgb) {
            if (!rgb.is_valid() || rgb.dtype() != DataType::Float32 || rgb.ndim() != 3 || rgb.size(2) != 3 ||
                !rgb.size(0) || !rgb.size(1) || (rgb.size(0) & 1) || (rgb.size(1) & 1) ||
                rgb.size(0) > uint32_t(INT_MAX / 3) / rgb.size(1))
                return Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Tensor, .detail = std::format("RGB to YUV420p requires even positive HWC Float32 RGB within pixel budget (valid={}, shape={}, dtype={})", rgb.is_valid(), rgb.shape().str(), static_cast<int>(rgb.dtype())), .detection = LFS_SOURCE_SITE_CURRENT()}));
            return {};
        }

        struct ColorPrograms {
            std::map<GpuBackend, std::unique_ptr<GpuKernelModule>> modules;
            std::map<GpuBackend, std::unique_ptr<GpuKernelModule>> linear_modules;
        };
        struct ColorRegistry {
            std::mutex mutex;
            std::vector<std::weak_ptr<ColorPrograms>> threads;
        };
        ColorRegistry& colorRegistry() {
            static ColorRegistry registry;
            return registry;
        }
        [[maybe_unused]] const bool shutdown_registered = [] {
            register_gpu_pre_shutdown_hook([]() noexcept {
                auto& registry = colorRegistry();
                std::lock_guard lock(registry.mutex);
                for (const auto& weak : registry.threads)
                    if (auto programs = weak.lock()) {
                        programs->modules.clear();
                        programs->linear_modules.clear();
                    }
                registry.threads.clear();
            });
            return true;
        }();
        ColorPrograms& colorPrograms() {
            thread_local auto programs = [] {
                auto cache = std::make_shared<ColorPrograms>();
                auto& registry = colorRegistry();
                std::lock_guard lock(registry.mutex);
                std::erase_if(registry.threads, [](const auto& weak) { return weak.expired(); });
                registry.threads.push_back(cache);
                return cache;
            }();
            return *programs;
        }
    } // namespace
    Result<Tensor> video_to_linear_rgb(const Tensor& bytes, const color::VideoColorParameters& color,
                                       uint32_t width, uint32_t height) {
        if (!bytes.is_valid() || bytes.dtype() != DataType::UInt8 || bytes.ndim() != 1 || !bytes.is_contiguous() ||
            !width || !height || uint64_t(width) * height > UINT32_MAX / 3 ||
            !color.width || !color.height || color.width > INT_MAX || color.height > INT_MAX ||
            color.rgb > 1 || color.big_endian > 1 || color.transfer > 2)
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Tensor, .detail = std::format("Video conversion requires contiguous UInt8 storage and addressable positive extents (valid={}, shape={}, dtype={}, source={}x{}, output={}x{}, rgb={}, endian={}, transfer={})", bytes.is_valid(), bytes.shape().str(), static_cast<int>(bytes.dtype()), color.width, color.height, width, height, color.rgb, color.big_endian, color.transfer), .detection = LFS_SOURCE_SITE_CURRENT()});
        for (size_t i = 0; i < 3; ++i) {
            const auto& c = color.component[i];
            const uint64_t sample_bytes = (uint64_t(c.depth) + c.shift + 7) / 8;
            const uint64_t row_bytes = c.width ? uint64_t(c.width - 1) * c.step + sample_bytes : 0;
            const uint64_t end = uint64_t(c.offset) + (c.height ? uint64_t(c.height - 1) * c.pitch : 0) + row_bytes;
            if (!c.width || !c.height || c.width > color.width || c.height > color.height ||
                (i == 0 && (c.width != color.width || c.height != color.height)) ||
                (color.rgb && (c.width != color.width || c.height != color.height)) ||
                !c.depth || c.depth > 16 || c.shift >= 32 || uint64_t(c.depth) + c.shift > 32 ||
                c.step < sample_bytes || c.pitch < row_bytes || end > bytes.numel() || end > UINT32_MAX)
                return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Tensor, .detail = std::format("Video component must fit the input storage (component={}, extent={}x{}, offset={}, pitch={}, step={}, depth={}, shift={}, end={}, storage={})", i, c.width, c.height, c.offset, c.pitch, c.step, c.depth, c.shift, end, bytes.numel()), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        for (const float value : {color.chroma_x_scale, color.chroma_x_offset, color.chroma_y_scale, color.chroma_y_offset,
                                  color.scale[0], color.scale[1], color.scale[2], color.bias[0], color.bias[1], color.bias[2],
                                  color.decode[0], color.decode[1], color.decode[2], color.decode[3], color.decode[4], color.decode[5], color.decode[6], color.decode[7], color.decode[8]})
            if (!std::isfinite(value))
                return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Tensor, .detail = std::format("Video conversion parameters must be finite (observed={})", value), .detection = LFS_SOURCE_SITE_CURRENT()});
        auto output = Tensor::empty_like(bytes, {height, width, 3}, DataType::Float32);
        if (const auto backend = gpu_backend_of(bytes)) {
            if (gpu_process_teardown_started())
                return make_error({.code = ErrorCode::FailedPrecondition, .domain = ErrorDomain::Tensor, .detail = "Video conversion cannot dispatch after GPU teardown (teardown_started=true)", .detection = LFS_SOURCE_SITE_CURRENT()});
            auto& program = colorPrograms().linear_modules[*backend];
            if (!program) {
                auto loaded = GpuKernelModule::load(linear_video_program_entries(), *backend);
                if (!loaded)
                    return std::move(loaded).error();
                program = std::move(*loaded);
            }
            auto constants = Tensor::empty_like(bytes, {sizeof(color)}, DataType::UInt8);
            TensorUpload upload;
            upload.enqueue(constants, std::as_bytes(std::span(&color, 1)), nullptr);
            upload.wait();
            struct Parameters {
                uint64_t planes = 0, color = 0, output = 0;
                uint32_t width, height;
            } parameters{.width = width, .height = height};
            const std::array bindings{GpuKernelModule::Binding{0, &bytes}, GpuKernelModule::Binding{8, &constants},
                                      GpuKernelModule::Binding{16, &output, GpuKernelModule::Access::ReadWrite}};
            auto result = program->dispatch({.function = "linearVideo", .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings}, .groups = {GpuKernelModule::groups_for(width, 8), GpuKernelModule::groups_for(height, 8), 1}, .group = {8, 8, 1}});
            if (!result)
                return std::move(result).error();
        } else {
            struct Reader {
                const uint8_t* data;
                uint32_t load(uint32_t address) { return data[address]; }
            };
            color::VideoColorSampler<Reader> sampler{color, {bytes.ptr<uint8_t>()}};
            auto* pixels = output.ptr<float>();
            tbb::parallel_for(uint32_t(0), height, [&](uint32_t y) {
                auto row_sampler = sampler;
                for (uint32_t x = 0; x < width; ++x) {
                    const auto value = row_sampler.resizedRgb(x, y, width, height);
                    for (uint32_t c = 0; c < 3; ++c)
                        pixels[(size_t(y) * width + x) * 3 + c] = value.channel[c];
                }
            });
        }
        return output;
    }
    Result<void> rgb_to_yuv420p_into(const Tensor& source, Yuv420Planes& out) {
        if (auto valid = validateRgb(source); !valid)
            return valid;
        const auto height = static_cast<uint32_t>(source.size(0)), width = static_cast<uint32_t>(source.size(1));
        const auto source_backend = gpu_backend_of(source);
        const std::array planes{&out.y, &out.u, &out.v};
        for (size_t i = 0; i < planes.size(); ++i) {
            const auto& plane = *planes[i];
            const auto plane_backend = gpu_backend_of(plane);
            const auto expected_height = i ? height / 2 : height, expected_width = i ? width / 2 : width;
            if (!plane.is_valid() || plane.dtype() != DataType::UInt8 || !plane.is_contiguous() || plane.ndim() != 2 ||
                plane.size(0) != expected_height || plane.size(1) != expected_width || plane.device() != source.device() || plane_backend != source_backend)
                return Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Tensor, .detail = std::format("YUV output plane must be contiguous UInt8 on the input device with visible extent (plane={}, valid={}, shape={}, expected={}x{}, dtype={}, device={}, expected_device={}, backend={}, expected_backend={})", i, plane.is_valid(), plane.shape().str(), expected_height, expected_width, static_cast<int>(plane.dtype()), static_cast<int>(plane.device()), static_cast<int>(source.device()), plane_backend ? gpu_backend_name(*plane_backend) : "CPU", source_backend ? gpu_backend_name(*source_backend) : "CPU"), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
        for (size_t i = 0; i < planes.size(); ++i) {
            if (internal::shares_storage(*planes[i], source))
                return Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Tensor, .detail = std::format("YUV output storage must not alias RGB input (plane={})", i), .detection = LFS_SOURCE_SITE_CURRENT()}));
            for (size_t j = 0; j < i; ++j)
                if (internal::shares_storage(*planes[i], *planes[j]))
                    return Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Tensor, .detail = std::format("YUV output planes require separate storage (planes={}, {})", j, i), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
        auto rgb = source.contiguous();
        if (const auto backend = gpu_backend_of(rgb)) {
            // Reuse the pipeline; per-thread storage avoids concurrent mutation
            // of program bindings without a global lock in the export hot path.
            if (gpu_process_teardown_started())
                return Result<void>::failure(make_error({.code = ErrorCode::FailedPrecondition, .domain = ErrorDomain::Tensor, .detail = "RGB conversion cannot dispatch after GPU teardown (teardown_started=true)", .detection = LFS_SOURCE_SITE_CURRENT()}));
            auto& program = colorPrograms().modules[*backend];
            if (!program) {
                auto loaded = GpuKernelModule::load(rgb_to_yuv_program_entries(), *backend);
                if (!loaded)
                    return Result<void>::failure(std::move(loaded).error());
                program = std::move(*loaded);
            }
            struct Params {
                uint64_t rgb = 0, y = 0, u = 0, v = 0;
                uint32_t width, height;
            } p{.width = width, .height = height};
            const std::array bindings{GpuKernelModule::Binding{0, &rgb},
                                      GpuKernelModule::Binding{8, &out.y, GpuKernelModule::Access::ReadWrite},
                                      GpuKernelModule::Binding{16, &out.u, GpuKernelModule::Access::ReadWrite},
                                      GpuKernelModule::Binding{24, &out.v, GpuKernelModule::Access::ReadWrite}};
            auto result = program->dispatch({.function = "rgbToYuv420p", .arguments = {std::as_bytes(std::span(&p, 1)), bindings}, .groups = {GpuKernelModule::groups_for(width / 2, 16), GpuKernelModule::groups_for(height / 2, 16), 1}, .group = {16, 16, 1}});
            if (!result)
                return Result<void>::failure(std::move(result).error());
        } else {
            const auto* pixels = rgb.ptr<float>();
            auto* yp = out.y.ptr<uint8_t>();
            auto* up = out.u.ptr<uint8_t>();
            auto* vp = out.v.ptr<uint8_t>();
            for (uint32_t row = 0; row < height; row += 2)
                for (uint32_t col = 0; col < width; col += 2) {
                    int rs = 0, gs = 0, bs = 0;
                    bool invalid_chroma = false;
                    for (uint32_t dy = 0; dy < 2; ++dy)
                        for (uint32_t dx = 0; dx < 2; ++dx) {
                            const auto index = (row + dy) * width + col + dx;
                            const auto rf = pixels[3 * index], gf = pixels[3 * index + 1], bf = pixels[3 * index + 2];
                            const bool invalid = isnan(rf) || isnan(gf) || isnan(bf);
                            invalid_chroma |= invalid;
                            const auto r = colorRgbByte(rf), g = colorRgbByte(gf), b = colorRgbByte(bf);
                            yp[index] = invalid ? 0 : colorLuma(r, g, b);
                            rs += r;
                            gs += g;
                            bs += b;
                        }
                    const auto index = (row / 2) * (width / 2) + col / 2;
                    up[index] = invalid_chroma ? 0 : colorChromaU(rs / 4, gs / 4, bs / 4);
                    vp[index] = invalid_chroma ? 0 : colorChromaV(rs / 4, gs / 4, bs / 4);
                }
        }
        return {};
    }
    Result<Yuv420Planes> rgb_to_yuv420p(const Tensor& rgb) {
        if (auto valid = validateRgb(rgb); !valid)
            return std::move(valid).error();
        Yuv420Planes out{Tensor::empty_like(rgb, {rgb.size(0), rgb.size(1)}, DataType::UInt8),
                         Tensor::empty_like(rgb, {rgb.size(0) / 2, rgb.size(1) / 2}, DataType::UInt8), Tensor::empty_like(rgb, {rgb.size(0) / 2, rgb.size(1) / 2}, DataType::UInt8)};
        auto result = rgb_to_yuv420p_into(rgb, out);
        if (!result)
            return std::move(result).error();
        return out;
    }
} // namespace lfs::core

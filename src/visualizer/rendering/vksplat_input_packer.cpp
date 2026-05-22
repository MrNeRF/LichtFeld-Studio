/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_input_packer.hpp"

#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor.hpp"
#include "rendering/rasterizer/vksplat_fwd/src/config.h"
#include "vksplat_input_packer_cuda.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
#include <string_view>
#include <vector>

namespace lfs::vis::vksplat {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        [[nodiscard]] std::expected<std::vector<float>, std::string> tensorToCpuVector(
            Tensor tensor,
            const std::string_view label) {
            if (!tensor.is_valid() || tensor.numel() == 0) {
                return std::vector<float>{};
            }
            try {
                const Tensor* current = &tensor;
                Tensor float_tensor;
                if (current->dtype() != DataType::Float32) {
                    float_tensor = current->to(DataType::Float32);
                    current = &float_tensor;
                }
                Tensor cpu_tensor;
                if (current->device() != Device::CPU) {
                    cpu_tensor = current->to(Device::CPU);
                    current = &cpu_tensor;
                }
                Tensor contiguous_tensor;
                if (!current->is_contiguous()) {
                    contiguous_tensor = current->contiguous();
                    current = &contiguous_tensor;
                }
                if (current->dtype() != DataType::Float32 || current->device() != Device::CPU) {
                    return std::unexpected(std::format("VkSplat failed to stage {} as CPU float32", label));
                }
                const float* ptr = current->ptr<float>();
                if (!ptr) {
                    return std::unexpected(std::format("VkSplat got a null CPU pointer for {}", label));
                }
                std::vector<float> result(current->numel());
                std::memcpy(result.data(), ptr, result.size() * sizeof(float));
                return result;
            } catch (const std::exception& e) {
                return std::unexpected(std::format("VkSplat failed to stage {}: {}", label, e.what()));
            }
        }

        [[nodiscard]] float readSwizzledRestCoeff(
            const std::vector<float>& shN,
            const std::size_t primitive_idx,
            const std::size_t rest_coeff_idx,
            const std::size_t channel,
            const std::uint32_t layout_rest) {
            const std::size_t packed_offset = rest_coeff_idx * 3 + channel;
            const std::size_t slot = packed_offset / 4;
            const std::size_t component = packed_offset % 4;
            const std::size_t float4_index =
                lfs::core::sh_swizzled_index(
                    static_cast<std::uint32_t>(primitive_idx),
                    static_cast<std::uint32_t>(slot),
                    layout_rest);
            return shN[float4_index * 4 + component];
        }

        template <typename PackedSh>
        [[nodiscard]] std::expected<void, std::string> copySwizzledRestToPaddedSh(
            const lfs::core::SplatData& splat_data,
            const std::size_t n,
            PackedSh& packed_sh,
            const std::string_view label) {
            const std::size_t layout_rest = splat_data.max_sh_coeffs_rest();
            const Tensor& shN_tensor = splat_data.shN();
            if (layout_rest == 0 || !shN_tensor.is_valid() || shN_tensor.numel() == 0) {
                return {};
            }
            if (shN_tensor.ndim() != 1) {
                return std::unexpected("VkSplat expected swizzled SH rest coefficients as a 1D tensor");
            }

            auto shN = tensorToCpuVector(shN_tensor, label);
            if (!shN) {
                return std::unexpected(shN.error());
            }
            const std::size_t expected_floats =
                lfs::core::sh_swizzled_float_count(n, static_cast<std::uint32_t>(layout_rest));
            if (shN->size() < expected_floats) {
                return std::unexpected("VkSplat staged swizzled SH rest tensor is smaller than expected");
            }

            const std::size_t rest = std::min<std::size_t>(15, layout_rest);
            const auto layout_rest_u32 = static_cast<std::uint32_t>(rest);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t k = 0; k < rest; ++k) {
                    for (std::size_t c = 0; c < 3; ++c) {
                        packed_sh[((i * 16) + (k + 1)) * 3 + c] =
                            readSwizzledRestCoeff(*shN, i, k, c, layout_rest_u32);
                    }
                }
            }
            return {};
        }
    } // namespace

    std::expected<void, std::string> packHostInputs(
        const lfs::core::SplatData& splat_data,
        Buffer<float>& xyz_ws,
        Buffer<float>& rotations,
        Buffer<float>& scales_opacs,
        Buffer<float>& sh_coeffs) {
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }

        const Tensor means_tensor = splat_data.get_means();
        const Tensor rotations_tensor = splat_data.get_rotation();
        const Tensor scales_tensor = splat_data.get_scaling();
        const Tensor opacities_tensor = splat_data.get_opacity();
        if (means_tensor.ndim() != 2 || means_tensor.size(0) != n || means_tensor.size(1) != 3 ||
            rotations_tensor.ndim() != 2 || rotations_tensor.size(0) != n || rotations_tensor.size(1) != 4 ||
            scales_tensor.ndim() != 2 || scales_tensor.size(0) != n || scales_tensor.size(1) != 3 ||
            opacities_tensor.ndim() != 1 || opacities_tensor.size(0) != n) {
            return std::unexpected("VkSplat input tensor shapes do not match [N,3]/[N,4]/[N]");
        }

        auto means = tensorToCpuVector(means_tensor, "model.means");
        auto rotations_cpu = tensorToCpuVector(rotations_tensor, "model.rotation");
        auto scales = tensorToCpuVector(scales_tensor, "model.scaling");
        auto opacities = tensorToCpuVector(opacities_tensor, "model.opacity");
        if (!means || !rotations_cpu || !scales || !opacities) {
            return std::unexpected(!means           ? means.error()
                                   : !rotations_cpu ? rotations_cpu.error()
                                   : !scales        ? scales.error()
                                                    : opacities.error());
        }
        if (means->size() != 3 * n || rotations_cpu->size() != 4 * n ||
            scales->size() != 3 * n || opacities->size() != n) {
            return std::unexpected("VkSplat staged input tensor sizes do not match the model splat count");
        }

        xyz_ws.assign(means->begin(), means->end());
        rotations.assign(rotations_cpu->begin(), rotations_cpu->end());
        VulkanGSPipelineBuffers::assignScalesOpacs(scales_opacs, n, scales->data(), opacities->data());

        sh_coeffs.assign(n * 16 * 3, 0.0f);
        const Tensor& sh0_tensor = splat_data.sh0();
        if (!sh0_tensor.is_valid() || sh0_tensor.numel() == 0 ||
            sh0_tensor.size(0) != n || sh0_tensor.size(sh0_tensor.ndim() - 1) != 3) {
            return std::unexpected("VkSplat expected SH DC coefficients shaped [N, 1, 3] or [N, 3]");
        }
        auto sh0 = tensorToCpuVector(sh0_tensor, "model.sh0");
        if (!sh0) {
            return std::unexpected(sh0.error());
        }
        if (sh0->size() < 3 * n) {
            return std::unexpected("VkSplat staged SH DC tensor is smaller than [N, 3]");
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t c = 0; c < 3; ++c) {
                sh_coeffs[(i * 16) * 3 + c] = (*sh0)[i * 3 + c];
            }
        }

        auto shN_copy = copySwizzledRestToPaddedSh(splat_data, n, sh_coeffs, "model.shN");
        if (!shN_copy) {
            return std::unexpected(shN_copy.error());
        }
        VulkanGSPipelineBuffers::reorderSH(sh_coeffs);
        return {};
    }

    namespace {

        constexpr int kShDim = 12;            // 16 SH coefficients * 3 channels / 4 floats per chunk
        constexpr int kShChunkFloats = 4;     // float4 chunks expected by the rasterizer
        constexpr int kShCoeffsPerSplat = 16; // 1 DC + up to 15 shN slots
        constexpr int kShChannels = 3;

        [[nodiscard]] Tensor activatedRotation(const Tensor& rotation_raw) {
            Tensor squared = rotation_raw.square();
            Tensor sum_squared = squared.sum({1}, true);
            Tensor norm = sum_squared.sqrt().clamp_min(1e-12f);
            return rotation_raw.div(norm);
        }

        [[nodiscard]] std::expected<Tensor, std::string> buildPackedShTensor(
            const lfs::core::SplatData& splat_data) {
            const std::size_t n = static_cast<std::size_t>(splat_data.size());

            const Tensor& sh0_raw = splat_data.sh0();
            if (!sh0_raw.is_valid() || sh0_raw.numel() == 0 ||
                sh0_raw.size(0) != n || sh0_raw.size(sh0_raw.ndim() - 1) != 3) {
                return std::unexpected("VkSplat expected SH DC coefficients shaped [N, 1, 3] or [N, 3]");
            }
            Tensor sh0 = sh0_raw;
            if (sh0.dtype() != DataType::Float32) {
                sh0 = sh0.to(DataType::Float32);
            }
            if (sh0.device() != Device::CUDA) {
                sh0 = sh0.to(Device::CUDA);
            }
            if (sh0.ndim() == 2) {
                sh0 = sh0.unsqueeze(1); // [N, 1, 3]
            }
            if (sh0.size(1) != 1) {
                return std::unexpected("VkSplat expected SH DC tensor with a single coefficient slot");
            }
            sh0 = sh0.contiguous();

            Tensor shN = splat_data.shN();
            if (shN.is_valid() && shN.numel() > 0) {
                if (shN.dtype() != DataType::Float32) {
                    shN = shN.to(DataType::Float32);
                }
                if (shN.device() != Device::CUDA) {
                    shN = shN.to(Device::CUDA);
                }
                if (!shN.is_contiguous()) {
                    shN = shN.contiguous();
                }
            }
            const float* shN_ptr = (shN.is_valid() && shN.numel() > 0) ? shN.ptr<float>() : nullptr;

            const std::size_t padded_n = lfs::core::sh_swizzled_padded_n(n);
            const std::size_t n_groups = padded_n / static_cast<std::size_t>(SH_REORDER_SIZE);
            Tensor packed = Tensor::empty({n_groups,
                                           static_cast<std::size_t>(kShDim),
                                           static_cast<std::size_t>(SH_REORDER_SIZE),
                                           static_cast<std::size_t>(kShChunkFloats)},
                                          Device::CUDA,
                                          DataType::Float32);
            lfs::core::sh_swizzled_pack_full_from_split(
                sh0.ptr<float>(),
                shN_ptr,
                packed.ptr<float>(),
                n,
                static_cast<std::uint32_t>(splat_data.max_sh_coeffs_rest()));
            return packed;
        }

        [[nodiscard]] bool validOpacityShape(const Tensor& opacity, const std::size_t n) {
            return opacity.ndim() != 0 &&
                   ((opacity.ndim() == 1 && opacity.size(0) == n) ||
                    (opacity.ndim() == 2 && opacity.size(0) == n && opacity.size(1) == 1));
        }

        [[nodiscard]] bool validSh0Shape(const Tensor& sh0, const std::size_t n) {
            if (!sh0.is_valid() || sh0.numel() == 0) {
                return false;
            }
            const std::size_t dims = sh0.ndim();
            if (dims != 2 && dims != 3) {
                return false;
            }
            return sh0.size(0) == n &&
                   sh0.size(dims - 1) == 3 &&
                   (dims == 2 || sh0.size(1) == 1);
        }

        [[nodiscard]] std::expected<void, std::string> requireCudaFloat32Contiguous(
            const Tensor& tensor,
            const std::string_view label) {
            if (!tensor.is_valid() || tensor.data_ptr() == nullptr) {
                return std::unexpected(std::format("VkSplat {} tensor is invalid", label));
            }
            if (tensor.dtype() != DataType::Float32) {
                return std::unexpected(std::format("VkSplat {} tensor must be Float32", label));
            }
            if (tensor.device() != Device::CUDA) {
                return std::unexpected(std::format("VkSplat {} tensor must be CUDA-resident", label));
            }
            if (!tensor.is_contiguous()) {
                return std::unexpected(std::format("VkSplat {} tensor must be contiguous", label));
            }
            return {};
        }

        [[nodiscard]] std::expected<void, std::string> waitForInputStream(
            const cudaStream_t stream,
            const Tensor& tensor,
            const std::string_view label) {
            try {
                lfs::core::waitForCUDAStream(stream, tensor.stream());
                return {};
            } catch (const std::exception& e) {
                return std::unexpected(std::format(
                    "VkSplat failed to order {} stream before direct input pack: {}",
                    label,
                    e.what()));
            }
        }

        [[nodiscard]] std::string cudaErrorMessage(
            const char* const operation,
            const cudaError_t status) {
            return std::format("{} failed: {} ({})",
                               operation,
                               cudaGetErrorName(status),
                               cudaGetErrorString(status));
        }

    } // namespace

    std::expected<DeviceInputLayout, std::string> deviceInputLayout(
        const lfs::core::SplatData& splat_data) {
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }

        const Tensor& means_raw = splat_data.means_raw();
        const Tensor& rotation_raw = splat_data.rotation_raw();
        const Tensor& scaling_raw = splat_data.scaling_raw();
        const Tensor& opacity_raw = splat_data.opacity_raw();
        if (means_raw.ndim() != 2 || means_raw.size(0) != n || means_raw.size(1) != 3 ||
            rotation_raw.ndim() != 2 || rotation_raw.size(0) != n || rotation_raw.size(1) != 4 ||
            scaling_raw.ndim() != 2 || scaling_raw.size(0) != n || scaling_raw.size(1) != 3) {
            return std::unexpected("VkSplat input tensor shapes do not match [N,3]/[N,4]/[N,3]");
        }
        if (!validOpacityShape(opacity_raw, n)) {
            return std::unexpected("VkSplat opacity tensor must be [N] or [N, 1]");
        }
        if (!validSh0Shape(splat_data.sh0(), n)) {
            return std::unexpected("VkSplat expected SH DC coefficients shaped [N, 1, 3] or [N, 3]");
        }
        const Tensor& shN = splat_data.shN();
        if (splat_data.max_sh_coeffs_rest() > 0 && (!shN.is_valid() || shN.numel() == 0)) {
            return std::unexpected("VkSplat expected swizzled SH rest coefficients for max SH degree");
        }
        if (shN.is_valid() && shN.numel() > 0 && shN.ndim() != 1) {
            return std::unexpected("VkSplat expected swizzled SH rest coefficients as a 1D tensor");
        }

        const auto layout_rest = static_cast<std::uint32_t>(splat_data.max_sh_coeffs_rest());
        const std::size_t sh_padded_floats = lfs::core::sh_swizzled_float_count(n);
        const std::size_t expected_shN_floats = lfs::core::sh_swizzled_float_count(n, layout_rest);
        if (shN.is_valid() && shN.numel() > 0 &&
            static_cast<std::size_t>(shN.numel()) < expected_shN_floats) {
            return std::unexpected("VkSplat swizzled SH rest tensor is smaller than expected");
        }

        return DeviceInputLayout{
            .num_splats = n,
            .xyz_bytes = n * 3 * sizeof(float),
            .rotations_bytes = n * 4 * sizeof(float),
            .scales_opacs_bytes = n * 4 * sizeof(float),
            .sh_coeffs_bytes = sh_padded_floats * sizeof(float),
            .sh_padded_floats = sh_padded_floats,
        };
    }

    std::expected<RawDeviceInputLayout, std::string> rawDeviceInputLayout(
        const lfs::core::SplatData& splat_data) {
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }

        const Tensor& means_raw = splat_data.means_raw();
        const Tensor& rotation_raw = splat_data.rotation_raw();
        const Tensor& scaling_raw = splat_data.scaling_raw();
        const Tensor& opacity_raw = splat_data.opacity_raw();
        const Tensor& sh0_raw = splat_data.sh0_raw();
        const Tensor& shN_raw = splat_data.shN_raw();
        if (means_raw.ndim() != 2 || means_raw.size(0) != n || means_raw.size(1) != 3 ||
            rotation_raw.ndim() != 2 || rotation_raw.size(0) != n || rotation_raw.size(1) != 4 ||
            scaling_raw.ndim() != 2 || scaling_raw.size(0) != n || scaling_raw.size(1) != 3) {
            return std::unexpected("VkSplat raw input tensor shapes do not match [N,3]/[N,4]/[N,3]");
        }
        if (!validOpacityShape(opacity_raw, n)) {
            return std::unexpected("VkSplat opacity tensor must be [N] or [N, 1]");
        }
        if (!validSh0Shape(sh0_raw, n)) {
            return std::unexpected("VkSplat expected SH DC coefficients shaped [N, 1, 3] or [N, 3]");
        }
        const auto layout_rest = static_cast<std::uint32_t>(splat_data.max_sh_coeffs_rest());
        const std::size_t layout_shN_bytes =
            lfs::core::sh_swizzled_float_count(n, layout_rest) * sizeof(float);
        const std::size_t shN_bytes = layout_rest == 0 ? sizeof(float4) : layout_shN_bytes;
        if (shN_raw.is_valid() && shN_raw.numel() > 0) {
            if (shN_raw.ndim() != 1) {
                return std::unexpected("VkSplat expected swizzled SH rest coefficients as a 1D tensor");
            }
            if (static_cast<std::size_t>(shN_raw.numel()) * sizeof(float) < layout_shN_bytes) {
                return std::unexpected("VkSplat swizzled SH rest tensor is smaller than expected");
            }
        } else if (splat_data.max_sh_coeffs_rest() > 0) {
            return std::unexpected("VkSplat expected swizzled SH rest coefficients for max SH degree");
        }

        return RawDeviceInputLayout{
            .num_splats = n,
            .xyz_bytes = n * 3 * sizeof(float),
            .sh0_bytes = n * 3 * sizeof(float),
            .shN_bytes = shN_bytes,
            .rotations_bytes = n * 4 * sizeof(float),
            .scaling_bytes = n * 3 * sizeof(float),
            .opacity_bytes = n * sizeof(float),
        };
    }

    std::expected<void, std::string> copyRawDeviceInputsToBuffer(
        const lfs::core::SplatData& splat_data,
        void* const xyz_dst,
        void* const sh0_dst,
        void* const shN_dst,
        void* const rotations_dst,
        void* const scaling_dst,
        void* const opacity_dst,
        const cudaStream_t stream) {
        auto layout = rawDeviceInputLayout(splat_data);
        if (!layout) {
            return std::unexpected(layout.error());
        }
        if (xyz_dst == nullptr || sh0_dst == nullptr || shN_dst == nullptr ||
            rotations_dst == nullptr || scaling_dst == nullptr || opacity_dst == nullptr) {
            return std::unexpected("VkSplat raw input copy received a null destination region");
        }

        const Tensor& means_raw = splat_data.means_raw();
        const Tensor& rotation_raw = splat_data.rotation_raw();
        const Tensor& scaling_raw = splat_data.scaling_raw();
        const Tensor& opacity_raw = splat_data.opacity_raw();
        const Tensor& sh0_raw = splat_data.sh0_raw();
        const Tensor& shN_raw = splat_data.shN_raw();

        if (auto ok = requireCudaFloat32Contiguous(means_raw, "means"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = requireCudaFloat32Contiguous(rotation_raw, "rotation"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = requireCudaFloat32Contiguous(scaling_raw, "scaling"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = requireCudaFloat32Contiguous(opacity_raw, "opacity"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = requireCudaFloat32Contiguous(sh0_raw, "sh0"); !ok) {
            return std::unexpected(ok.error());
        }
        const bool has_shN = shN_raw.is_valid() && shN_raw.numel() > 0;
        if (has_shN) {
            if (auto ok = requireCudaFloat32Contiguous(shN_raw, "shN"); !ok) {
                return std::unexpected(ok.error());
            }
        }

        if (auto ok = waitForInputStream(stream, means_raw, "means"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = waitForInputStream(stream, rotation_raw, "rotation"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = waitForInputStream(stream, scaling_raw, "scaling"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = waitForInputStream(stream, opacity_raw, "opacity"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = waitForInputStream(stream, sh0_raw, "sh0"); !ok) {
            return std::unexpected(ok.error());
        }
        if (has_shN) {
            if (auto ok = waitForInputStream(stream, shN_raw, "shN"); !ok) {
                return std::unexpected(ok.error());
            }
        }

        const auto copy = [&](void* dst,
                              const Tensor& src,
                              const std::size_t bytes,
                              const char* const label) -> std::expected<void, std::string> {
            const cudaError_t status = cudaMemcpyAsync(
                dst, src.data_ptr(), bytes, cudaMemcpyDeviceToDevice, stream);
            if (status != cudaSuccess) {
                return std::unexpected(cudaErrorMessage(label, status));
            }
            return {};
        };

        if (auto ok = copy(xyz_dst, means_raw, layout->xyz_bytes,
                           "cudaMemcpyAsync(VkSplat raw means -> Vulkan input)");
            !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = copy(sh0_dst, sh0_raw, layout->sh0_bytes,
                           "cudaMemcpyAsync(VkSplat raw sh0 -> Vulkan input)");
            !ok) {
            return std::unexpected(ok.error());
        }
        if (has_shN) {
            if (auto ok = copy(shN_dst, shN_raw, layout->shN_bytes,
                               "cudaMemcpyAsync(VkSplat raw shN -> Vulkan input)");
                !ok) {
                return std::unexpected(ok.error());
            }
        } else {
            const cudaError_t status = cudaMemsetAsync(shN_dst, 0, layout->shN_bytes, stream);
            if (status != cudaSuccess) {
                return std::unexpected(cudaErrorMessage("cudaMemsetAsync(VkSplat raw shN fallback)", status));
            }
        }
        if (auto ok = copy(rotations_dst, rotation_raw, layout->rotations_bytes,
                           "cudaMemcpyAsync(VkSplat raw rotation -> Vulkan input)");
            !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = copy(scaling_dst, scaling_raw, layout->scaling_bytes,
                           "cudaMemcpyAsync(VkSplat raw scaling -> Vulkan input)");
            !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = copy(opacity_dst, opacity_raw, layout->opacity_bytes,
                           "cudaMemcpyAsync(VkSplat raw opacity -> Vulkan input)");
            !ok) {
            return std::unexpected(ok.error());
        }

        return {};
    }

    std::expected<void, std::string> packDeviceInputsToBuffer(
        const lfs::core::SplatData& splat_data,
        void* const xyz_dst,
        void* const rotations_dst,
        void* const scales_opacs_dst,
        void* const sh_coeffs_dst,
        const cudaStream_t stream) {
        auto layout = deviceInputLayout(splat_data);
        if (!layout) {
            return std::unexpected(layout.error());
        }
        if (xyz_dst == nullptr || rotations_dst == nullptr || scales_opacs_dst == nullptr || sh_coeffs_dst == nullptr) {
            return std::unexpected("VkSplat direct input pack received a null destination region");
        }

        const Tensor& means_raw = splat_data.means_raw();
        const Tensor& rotation_raw = splat_data.rotation_raw();
        const Tensor& scaling_raw = splat_data.scaling_raw();
        const Tensor& opacity_raw = splat_data.opacity_raw();
        const Tensor& sh0_raw = splat_data.sh0();
        const Tensor& shN_raw = splat_data.shN();

        if (auto ok = requireCudaFloat32Contiguous(means_raw, "means"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = requireCudaFloat32Contiguous(rotation_raw, "rotation"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = requireCudaFloat32Contiguous(scaling_raw, "scaling"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = requireCudaFloat32Contiguous(opacity_raw, "opacity"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = requireCudaFloat32Contiguous(sh0_raw, "sh0"); !ok) {
            return std::unexpected(ok.error());
        }
        const bool has_shN = shN_raw.is_valid() && shN_raw.numel() > 0;
        if (has_shN) {
            if (auto ok = requireCudaFloat32Contiguous(shN_raw, "shN"); !ok) {
                return std::unexpected(ok.error());
            }
        }

        if (auto ok = waitForInputStream(stream, means_raw, "means"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = waitForInputStream(stream, rotation_raw, "rotation"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = waitForInputStream(stream, scaling_raw, "scaling"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = waitForInputStream(stream, opacity_raw, "opacity"); !ok) {
            return std::unexpected(ok.error());
        }
        if (auto ok = waitForInputStream(stream, sh0_raw, "sh0"); !ok) {
            return std::unexpected(ok.error());
        }
        if (has_shN) {
            if (auto ok = waitForInputStream(stream, shN_raw, "shN"); !ok) {
                return std::unexpected(ok.error());
            }
        }

        cudaError_t status = cudaMemcpyAsync(
            xyz_dst,
            means_raw.data_ptr(),
            layout->xyz_bytes,
            cudaMemcpyDeviceToDevice,
            stream);
        if (status != cudaSuccess) {
            return std::unexpected(cudaErrorMessage("cudaMemcpyAsync(VkSplat means -> Vulkan input)", status));
        }

        status = detail::launchPackActivatedRotations(
            rotation_raw.ptr<float>(),
            static_cast<float*>(rotations_dst),
            layout->num_splats,
            stream);
        if (status != cudaSuccess) {
            return std::unexpected(cudaErrorMessage("VkSplat rotation direct pack", status));
        }

        status = detail::launchPackScalesOpacs(
            scaling_raw.ptr<float>(),
            opacity_raw.ptr<float>(),
            static_cast<float*>(scales_opacs_dst),
            layout->num_splats,
            stream);
        if (status != cudaSuccess) {
            return std::unexpected(cudaErrorMessage("VkSplat scale/opacity direct pack", status));
        }

        lfs::core::sh_swizzled_pack_full_from_split(
            sh0_raw.ptr<float>(),
            has_shN ? shN_raw.ptr<float>() : nullptr,
            static_cast<float*>(sh_coeffs_dst),
            layout->num_splats,
            static_cast<std::uint32_t>(splat_data.max_sh_coeffs_rest()),
            stream);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return std::unexpected(cudaErrorMessage("VkSplat SH direct pack", status));
        }

        return {};
    }

    std::expected<DevicePackedInputs, std::string> packDeviceInputs(
        const lfs::core::SplatData& splat_data) {
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }

        const Tensor& means_raw = splat_data.means_raw();
        const Tensor& rotation_raw = splat_data.rotation_raw();
        const Tensor& scaling_raw = splat_data.scaling_raw();
        const Tensor& opacity_raw = splat_data.opacity_raw();
        if (means_raw.ndim() != 2 || means_raw.size(0) != n || means_raw.size(1) != 3 ||
            rotation_raw.ndim() != 2 || rotation_raw.size(0) != n || rotation_raw.size(1) != 4 ||
            scaling_raw.ndim() != 2 || scaling_raw.size(0) != n || scaling_raw.size(1) != 3) {
            return std::unexpected("VkSplat input tensor shapes do not match [N,3]/[N,4]/[N,3]");
        }
        if (opacity_raw.ndim() == 0 ||
            (opacity_raw.ndim() == 1 && opacity_raw.size(0) != n) ||
            (opacity_raw.ndim() == 2 && (opacity_raw.size(0) != n || opacity_raw.size(1) != 1))) {
            return std::unexpected("VkSplat opacity tensor must be [N] or [N, 1]");
        }

        try {
            const auto to_cuda_f32 = [](const Tensor& t) {
                Tensor out = t;
                if (out.dtype() != DataType::Float32) {
                    out = out.to(DataType::Float32);
                }
                if (out.device() != Device::CUDA) {
                    out = out.to(Device::CUDA);
                }
                return out.contiguous();
            };

            DevicePackedInputs result;
            result.num_splats = n;

            result.xyz_ws = to_cuda_f32(means_raw);

            const Tensor rotation_in = to_cuda_f32(rotation_raw);
            result.rotations = activatedRotation(rotation_in).contiguous();

            const Tensor scaling_in = to_cuda_f32(scaling_raw);
            const Tensor scales_exp = scaling_in.exp().contiguous();
            const Tensor opacity_in = to_cuda_f32(opacity_raw);
            const Tensor opacity_2d = opacity_in.ndim() == 1 ? opacity_in.unsqueeze(1) : opacity_in;
            const Tensor opacity_sig = opacity_2d.sigmoid().contiguous();
            // Cat through a 3D shape so the tensor library's last-dim path
            // (which hardcodes alpha=1.0f for [N,3]+[N,1] floats) is bypassed.
            const Tensor scales_3d = scales_exp.unsqueeze(2);
            const Tensor opacity_3d = opacity_sig.unsqueeze(2);
            result.scales_opacs = Tensor::cat({scales_3d, opacity_3d}, 1)
                                      .reshape(lfs::core::TensorShape{n, std::size_t{4}})
                                      .contiguous();

            auto sh_packed = buildPackedShTensor(splat_data);
            if (!sh_packed) {
                return std::unexpected(sh_packed.error());
            }
            result.sh_coeffs = std::move(*sh_packed);
            result.sh_padded_floats = static_cast<std::size_t>(result.sh_coeffs.numel());

            assert(static_cast<std::size_t>(result.xyz_ws.numel()) == n * 3);
            assert(static_cast<std::size_t>(result.rotations.numel()) == n * 4);
            assert(static_cast<std::size_t>(result.scales_opacs.numel()) == n * 4);
            assert(result.sh_padded_floats == lfs::core::sh_swizzled_float_count(n));

            return result;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat device-side packing failed: {}", e.what()));
        }
    }

    std::expected<std::vector<float>, std::string> buildPaddedShReference(
        const lfs::core::SplatData& splat_data) {
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }
        std::vector<float> sh(n * 16 * 3, 0.0f);

        const Tensor& sh0_tensor = splat_data.sh0();
        if (!sh0_tensor.is_valid() || sh0_tensor.numel() == 0 ||
            sh0_tensor.size(0) != n || sh0_tensor.size(sh0_tensor.ndim() - 1) != 3) {
            return std::unexpected("VkSplat expected SH DC coefficients shaped [N, 1, 3] or [N, 3]");
        }
        auto sh0 = tensorToCpuVector(sh0_tensor, "model.sh0");
        if (!sh0) {
            return std::unexpected(sh0.error());
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t c = 0; c < 3; ++c) {
                sh[(i * 16) * 3 + c] = (*sh0)[i * 3 + c];
            }
        }

        auto shN_copy = copySwizzledRestToPaddedSh(splat_data, n, sh, "model.shN");
        if (!shN_copy) {
            return std::unexpected(shN_copy.error());
        }
        return sh;
    }

} // namespace lfs::vis::vksplat

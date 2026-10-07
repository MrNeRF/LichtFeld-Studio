/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_fwd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::core {

    // A Slang program compiled at build time for the enabled tensor backends.
    // New callers bind tensors, never native resources or addresses. All work
    // participates in the tensor timeline, including raster attachment copies.
    class LFS_CORE_API GpuKernelModule {
    public:
        enum class Stage : uint8_t { Compute,
                                     Vertex,
                                     Fragment };
        struct Entry {
            std::string_view name;
            Stage stage;
            GpuBackend backend;
            std::span<const std::byte> code; // SPIR-V, Slang-generated MSL, or PTX
            uint32_t parameter_bytes;        // build-time Slang reflection
            std::span<const uint32_t> tensor_offsets;
            std::array<uint32_t, 3> thread_group{1, 1, 1};
            // Metal compiles the entry with relaxed math (reassociation, fast
            // functions; Inf/NaN still honored), as native opaque blending does.
            // Other backends keep precise floating point.
            bool relaxed_math = false;
        };
        enum class Access : uint8_t { Read,
                                      ReadWrite };
        struct Binding {
            uint32_t parameter_offset; // offset of the Slang Ptr<T> field
            // nullptr leaves the pointer 0: an optional input the shader tests.
            const Tensor* tensor;
            Access access = Access::Read;
        };
        struct Arguments {
            // One global [[vk::push_constant]] ConstantBuffer, C/scalar layout.
            // Pointer fields are zero in caller data and patched by core.
            std::span<const std::byte> parameters;
            std::span<const Binding> tensors;
        };
        struct Dispatch {
            std::string_view function;
            Arguments arguments;
            std::array<uint32_t, 3> groups{1, 1, 1};
            std::array<uint32_t, 3> group{64, 1, 1};
            // When set, the threadgroup counts are three uint32 that GPU work
            // wrote at element `indirect_offset` of this Int32/UInt32 tensor;
            // `groups` is ignored. Vulkan and Metal only.
            const Tensor* indirect = nullptr;
            size_t indirect_offset = 0;
        };
        enum class Blend : uint8_t { Opaque,
                                     StraightAlpha,
                                     PremultipliedAlpha };
        // Front faces wind counter-clockwise in NDC (+Y up), as in OpenGL.
        enum class Cull : uint8_t { None,
                                    Back,
                                    Front };
        enum class Compare : uint8_t { Always,
                                       Less,
                                       LessEqual };
        struct Scissor {
            uint32_t x = 0;
            uint32_t y = 0;
            uint32_t width = 0;
            uint32_t height = 0;
        };
        // Top-left-origin pixel rectangle that NDC [-1,1] maps to.
        struct Viewport {
            float x = 0;
            float y = 0;
            float width = 0;
            float height = 0;
        };
        struct Draw {
            std::string_view vertex;
            std::string_view fragment;
            Arguments arguments;
            // Color: contiguous [H,W,4], Float32 or UInt8 (linear UNORM).
            // Depth: optional contiguous Float32 [H,W], zero-to-one depth.
            // Pixel (0,0) is top-left; NDC +Y is up. Triangle-list topology.
            Tensor* color = nullptr;
            Tensor* depth = nullptr;
            uint32_t vertex_count = 0;
            uint32_t first_vertex = 0;
            uint32_t instance_count = 1;
            // Optional top-left-origin pixel rectangle. Clears always affect the
            // full attachment; the scissor clips rasterized fragments only.
            std::optional<Scissor> scissor;
            // Defaults to the whole attachment.
            std::optional<Viewport> viewport;
            Blend blend = Blend::Opaque;
            Cull cull = Cull::None;
            Compare depth_compare = Compare::Less;
            bool depth_write = true;
            bool clear_color = false;
            std::array<float, 4> color_clear{0, 0, 0, 0};
            bool clear_depth = false;
            float depth_clear = 1.0f;
        };

        [[nodiscard]] static Result<std::unique_ptr<GpuKernelModule>> load(
            std::span<const Entry> entries, GpuBackend backend = default_gpu_backend());
        [[nodiscard]] Result<void> dispatch(const Dispatch& dispatch);
        [[nodiscard]] Result<void> draw(const Draw& draw);
        // One submission for draws that share the first draw's color/depth
        // attachments: one attachment load, every draw in order, one store.
        // Only the first draw may clear.
        [[nodiscard]] Result<void> draw_batch(std::span<const Draw> draws);
        [[nodiscard]] bool supports_raster() const;
        // False after the program's backend context has shut down. Does not
        // initialize a device; cached callers reload for the next context.
        [[nodiscard]] bool is_live() const noexcept;

        // Legacy MSL-only training kernels. Kept until the separately scoped
        // single-source training migration; no new shaders use this path.
        struct Launch {
            std::string_view function;
            std::span<const std::byte> params;
            // Every tensor the kernel reads or writes.
            std::span<const Tensor* const> uses;
            std::array<uint32_t, 3> groups{1, 1, 1};
            std::array<uint32_t, 3> group{256, 1, 1};
            // Function constants by index; every value is a uint.
            std::span<const std::pair<uint32_t, uint32_t>> constants;
        };

        // fast_math trades IEEE edge cases for speed, as the CUDA kernels built
        // with -use_fast_math do.
        GpuKernelModule(GpuBackend backend, std::string source, bool fast_math);
        ~GpuKernelModule();
        GpuKernelModule(const GpuKernelModule&) = delete;
        GpuKernelModule& operator=(const GpuKernelModule&) = delete;

        [[nodiscard]] GpuBackend backend() const;
        // Device address of the tensor's first element; 0 for an invalid or
        // empty tensor. The tensor must live on this module's backend.
        [[nodiscard]] uint64_t address(const Tensor& tensor) const;
        void launch(const Launch& launch);

        // Threadgroups that cover `count` items at `width` per group.
        [[nodiscard]] static uint32_t groups_for(size_t count, uint32_t width);

    private:
        GpuKernelModule();
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::core

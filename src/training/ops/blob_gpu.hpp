/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/assert.hpp"
#include "lfs/training/ops/blob.hpp"
#include <algorithm>
#include <cmath>
namespace lfs::training::blob_detail {
    struct Parameters {
        uint64_t a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
        int32_t width = 0, height = 0, source_width = 0, source_height = 0;
        uint32_t count = 0, capacity = 0, factor = 0, byte_input = 0;
        uint32_t radius = 0, horizontal = 0, x_max = 0, y_max = 0;
        int32_t view = 0, neighbors = 0;
        float weights[7]{};
    };
    static_assert(sizeof(Parameters) == 136 && offsetof(Parameters, weights) == 104);
    static_assert(sizeof(kernels::blob_seeding::SweepView) == 96);
    static_assert(offsetof(kernels::blob_seeding::SweepView, bitmap_offset) == 72);
    template <class Backend>
    struct Implementation {
        using Tensor = core::Tensor;
        using Workspace = kernels::blob_seeding::DetectionWorkspace;
        using ViewPeaks = kernels::blob_seeding::ViewPeaks;
        static void reserve(Tensor& t, size_t count, core::DataType dtype) {
            if (!t.is_valid() || t.numel() < count)
                t = Tensor::empty({count}, core::Device::GPU, dtype);
        }
        static Tensor downsample(const Tensor& image, int factor, Workspace& workspace) {
            LFS_ASSERT(image.device() == core::Device::GPU && image.ndim() == 3 && image.shape()[0] >= 3 && factor >= 1);
            LFS_ASSERT(image.dtype() == core::DataType::UInt8 || image.dtype() == core::DataType::Float32);
            auto source = image.contiguous();
            const auto h = source.shape()[1] / factor, w = source.shape()[2] / factor;
            reserve(workspace.rgb, 3 * h * w, core::DataType::Float32);
            auto output = workspace.rgb.slice(0, 0, 3 * h * w).reshape(core::TensorShape{3, h, w});
            if (h == 0 || w == 0)
                return output;
            Parameters p;
            p.a = Backend::address(source);
            p.b = Backend::address(output);
            p.width = static_cast<int>(w);
            p.height = static_cast<int>(h);
            p.source_width = static_cast<int>(source.shape()[2]);
            p.source_height = static_cast<int>(source.shape()[1]);
            p.factor = factor;
            p.byte_input = source.dtype() == core::DataType::UInt8;
            Backend::launch(0, p, {&source}, {&output}, h * w);
            return output;
        }
        static ViewPeaks detect(const Tensor& image, int view, Workspace& workspace) {
            LFS_ASSERT(image.device() == core::Device::GPU && image.dtype() == core::DataType::Float32);
            LFS_ASSERT(image.ndim() == 3 && image.shape()[0] >= 3);
            auto source = image.contiguous();
            const auto h = source.shape()[1], w = source.shape()[2], n = h * w, words = (n + 15) / 16, capacity = n / 4 + 1;
            reserve(workspace.rows, n, core::DataType::Float32);
            reserve(workspace.a, 2 * n, core::DataType::Float32);
            reserve(workspace.b, 2 * n, core::DataType::Float32);
            reserve(workspace.c, 2 * n, core::DataType::Float32);
            reserve(workspace.mask, n, core::DataType::UInt8);
            reserve(workspace.peaks, capacity * 7, core::DataType::Float32);
            reserve(workspace.counters, 3, core::DataType::Int32);
            reserve(workspace.bitmap, words, core::DataType::UInt32);
            workspace.counters.zero_();
            auto peaks = workspace.peaks.slice(0, 0, capacity * 7).reshape(core::TensorShape{capacity, 7});
            auto bitmap = workspace.bitmap.slice(0, 0, words);
            Parameters p;
            p.width = static_cast<int>(w);
            p.height = static_cast<int>(h);
            p.count = static_cast<uint32_t>(n);
            p.capacity = static_cast<uint32_t>(capacity);
            p.view = view;
            float sum = 0;
            for (int k = -3; k <= 3; ++k) {
                p.weights[k + 3] = std::exp(-0.5f * k * k / (0.7f * 0.7f));
                sum += p.weights[k + 3];
            }
            for (auto& weight : p.weights)
                weight /= sum;
            p.a = Backend::address(source);
            p.b = Backend::address(workspace.rows);
            Backend::launch(1, p, {&source}, {&workspace.rows}, n);
            p.a = p.b;
            p.b = Backend::address(workspace.c);
            Backend::launch(2, p, {&workspace.rows}, {&workspace.c}, n);
            auto extrema = [&](const Tensor& input, Tensor& output, uint32_t radius, bool horizontal, bool x_max, bool y_max) {
                p.a = Backend::address(input);
                p.b = Backend::address(output);
                p.radius = radius;
                p.horizontal = horizontal;
                p.x_max = x_max;
                p.y_max = y_max;
                Backend::launch(3, p, {&input}, {&output}, n);
            };
            extrema(workspace.c, workspace.a, 3, true, false, true);
            extrema(workspace.a, workspace.b, 3, false, false, true);
            extrema(workspace.b, workspace.a, 3, true, true, false);
            extrema(workspace.a, workspace.b, 3, false, true, false);
            p.a = Backend::address(workspace.c);
            p.b = Backend::address(workspace.b);
            p.c = Backend::address(workspace.a);
            Backend::launch(4, p, {&workspace.c, &workspace.b}, {&workspace.a}, n);
            extrema(workspace.a, workspace.b, 2, true, true, true);
            extrema(workspace.b, workspace.c, 2, false, true, true);
            p.a = Backend::address(workspace.a);
            p.b = Backend::address(workspace.c);
            p.c = Backend::address(source);
            p.d = Backend::address(workspace.mask);
            p.e = Backend::address(peaks);
            p.f = Backend::address(workspace.counters);
            Backend::launch(5, p, {&workspace.a, &workspace.c, &source, &workspace.counters}, {&workspace.mask, &peaks, &workspace.counters}, n);
            p.a = p.d;
            p.b = Backend::address(bitmap);
            p.c = p.f;
            Backend::launch(6, p, {&workspace.mask, &workspace.counters}, {&bitmap, &workspace.counters}, words);
            const auto host = workspace.counters.cpu();
            const int* counts = host.template ptr<int>();
            const size_t emitted = std::min(size_t(counts[0]), capacity);
            ViewPeaks result;
            if (emitted > 0)
                result.peaks = peaks.slice(0, 0, emitted);
            result.bitmap = bitmap;
            result.density = {float(static_cast<uint32_t>(counts[1])) / float(n), float(static_cast<uint32_t>(counts[2])) / float(n)};
            return result;
        }
        static Tensor sweep(const Tensor& peaks, const Tensor& views, const Tensor& neighbors, const Tensor& bitmaps) {
            LFS_ASSERT(peaks.device() == core::Device::GPU && peaks.ndim() == 2 && peaks.shape()[1] == 7);
            LFS_ASSERT(neighbors.device() == core::Device::GPU && neighbors.dtype() == core::DataType::Int32 && neighbors.ndim() == 2);
            LFS_ASSERT(views.device() == core::Device::GPU && views.bytes() == neighbors.shape()[0] * sizeof(kernels::blob_seeding::SweepView));
            LFS_ASSERT(bitmaps.device() == core::Device::GPU && bitmaps.dtype() == core::DataType::UInt32);
            const auto n = peaks.shape()[0];
            auto output = Tensor::empty({n, 2}, core::Device::GPU);
            Parameters p;
            p.a = Backend::address(peaks);
            p.b = Backend::address(views);
            p.c = Backend::address(neighbors);
            p.d = Backend::address(bitmaps);
            p.e = Backend::address(output);
            p.neighbors = static_cast<int>(neighbors.shape()[1]);
            if (n)
                Backend::launch(7, p, {&peaks, &views, &neighbors, &bitmaps}, {&output}, n * 256);
            return output;
        }
        static const gpu_ops::BlobOps& table() {
            static const gpu_ops::BlobOps result{detect, downsample, sweep};
            return result;
        }
    };
} // namespace lfs::training::blob_detail

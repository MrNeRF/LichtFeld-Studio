/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor.hpp"
#include <cstdint>
#include <vector>

// Row views and selection tables shared by the CPU reference and the GPU decimation backends.
namespace lfs::core::decimate {
    constexpr uint32_t invalid = 0xffffffffu;
    constexpr int knn_k = 16;
    constexpr int candidates_k = 4;
    struct View {
        float *pos, *rot, *scale, *opacity, *dc, *sh;
        int rest;
    };
    struct Data {
        Tensor pos, rot, scale, opacity, dc, sh;
        size_t n;
        int rest;
        View view() const {
            auto ptr = [](const Tensor& t) { return const_cast<float*>(t.ptr<float>()); };
            return {ptr(pos), ptr(rot), ptr(scale), ptr(opacity), ptr(dc), rest ? ptr(sh) : nullptr, rest};
        }
    };
    struct Candidates {
        std::vector<uint32_t> idx;
        std::vector<float> cost;
    };
    struct Selection {
        std::vector<int> member_group;
        std::vector<uint32_t> members, offsets, minimum;
        size_t removed = 0;
    };
    inline Data allocate(size_t n, int rest, Device device) {
        return {Tensor::empty({n, 3}, device), Tensor::empty({n, 4}, device), Tensor::empty({n, 3}, device),
                Tensor::empty({n, 1}, device), Tensor::empty({n, 1, 3}, device),
                Tensor::empty({n, size_t(rest), 3}, device), n, rest};
    }
} // namespace lfs::core::decimate

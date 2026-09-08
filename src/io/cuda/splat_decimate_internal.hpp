/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <core/tensor.hpp>
#include <cstdint>
#include <vector>

namespace lfs::io::decimate {
    constexpr uint32_t invalid = 0xffffffffu;
    constexpr int max_knn = 32;
    struct View {
        float *pos, *rot, *scale, *opacity, *dc, *sh;
        int rest;
    };
    struct Data {
        core::Tensor pos, rot, scale, opacity, dc, sh;
        size_t n;
        int rest;
        View view() const {
            auto ptr = [](const core::Tensor& t) { return const_cast<float*>(t.ptr<float>()); };
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
    Data allocate(size_t n, int rest, core::Device device);
    Selection select(const Candidates&, size_t n, int k, size_t needed);
    Candidates cpu_candidates(const Data&, int knn, int k);
    Candidates gpu_candidates(const Data&, int knn, int k);
    Data cpu_merge(const Data&, const Selection&);
    Data gpu_merge(const Data&, const Selection&);
} // namespace lfs::io::decimate

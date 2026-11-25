/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "sharpness_score.hpp"
#include "sharpness_edge.hpp"
#include <functional>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

namespace lfs::io {
    namespace {
        template <SharpnessAlgorithm Algorithm>
        long long countSharpEdges(const uint8_t* rgb, int w, int h) {
            const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(w) * 3;
            const auto count_rows = [&](const tbb::blocked_range<int>& rows, long long edges) {
                for (int y = rows.begin(); y < rows.end(); ++y) {
                    for (int x = 1; x < w - 1; ++x)
                        edges += detail::sharpEdge(rgb + static_cast<std::ptrdiff_t>(y) * stride + static_cast<std::ptrdiff_t>(x) * 3 + 1, stride, Algorithm);
                }
                return edges;
            };
            const tbb::blocked_range<int> rows(1, h - 1, 16);
            if (static_cast<long long>(w) * h < 256 * 256)
                return count_rows(rows, 0);
            return tbb::parallel_reduce(rows, 0LL, count_rows, std::plus<long long>{});
        }
    } // namespace

    [[nodiscard]] double computeSharpnessScore(const uint8_t* rgb, const int w, const int h, const SharpnessAlgorithm algo) {
        if (w < 3 || h < 3)
            return 0.0;
        long long edges;
        switch (algo) {
        case SharpnessAlgorithm::LAPLACIAN:
            edges = countSharpEdges<SharpnessAlgorithm::LAPLACIAN>(rgb, w, h);
            break;
        case SharpnessAlgorithm::COMBINED:
            edges = countSharpEdges<SharpnessAlgorithm::COMBINED>(rgb, w, h);
            break;
        default:
            edges = countSharpEdges<SharpnessAlgorithm::TENENGRAD>(rgb, w, h);
            break;
        }
        return detail::edgeRatio(edges, w, h);
    }
} // namespace lfs::io

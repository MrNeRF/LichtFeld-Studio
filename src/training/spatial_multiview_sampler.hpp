/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace lfs::training {

    /// Sampler that groups cameras by spatial proximity for multi-view training.
    ///
    /// Each epoch, cameras are arranged into spatially coherent groups of K using
    /// greedy nearest-neighbor selection from random seeds. Every camera appears
    /// exactly once per epoch (same coverage as InfiniteRandomSampler). Auto-resets
    /// when exhausted, producing a different grouping each epoch.
    class SpatialMultiViewSampler {
    public:
        /// @param size Number of cameras
        /// @param positions Flat vector of 3D positions (3 floats per camera, length = size * 3)
        /// @param group_size Number of cameras per spatial group (K)
        SpatialMultiViewSampler(size_t size, std::vector<float> positions, int group_size)
            : size_(size),
              positions_(std::move(positions)),
              group_size_(group_size),
              index_(0) {
            assert(positions_.size() == size_ * 3);
            assert(group_size_ > 0);
            build_spatial_ordering();
        }

        void reset(std::optional<size_t> new_size = std::nullopt) {
            if (new_size) {
                size_ = *new_size;
                assert(positions_.size() == size_ * 3);
            }
            index_ = 0;
            build_spatial_ordering();
        }

        std::optional<std::vector<size_t>> next(size_t batch_size) {
            if (index_ >= size_) {
                reset();
            }

            const size_t end = std::min(index_ + batch_size, size_);
            std::vector<size_t> batch(indices_.begin() + index_, indices_.begin() + end);
            index_ = end;
            return batch;
        }

        size_t size() const { return size_; }

        float avg_intra_group_distance() const { return avg_intra_group_dist_; }

    private:
        void build_spatial_ordering() {
            indices_.clear();
            indices_.reserve(size_);

            std::vector<bool> visited(size_, false);
            std::mt19937 rng(std::random_device{}());

            // Collect unvisited indices for seed selection
            std::vector<size_t> unvisited;
            unvisited.resize(size_);
            std::iota(unvisited.begin(), unvisited.end(), 0);

            double total_intra_dist = 0.0;
            int group_count = 0;

            while (indices_.size() < size_) {
                // Remove already-visited from unvisited pool
                std::erase_if(unvisited, [&](size_t i) { return visited[i]; });
                if (unvisited.empty())
                    break;

                // Pick random seed from unvisited
                std::uniform_int_distribution<size_t> dist(0, unvisited.size() - 1);
                const size_t seed_idx = unvisited[dist(rng)];

                std::vector<size_t> group;
                group.reserve(static_cast<size_t>(group_size_));
                group.push_back(seed_idx);
                visited[seed_idx] = true;

                // Greedy nearest-neighbor: add K-1 nearest unvisited cameras
                while (static_cast<int>(group.size()) < group_size_) {
                    // Compute group centroid
                    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                    for (size_t gi : group) {
                        cx += positions_[gi * 3 + 0];
                        cy += positions_[gi * 3 + 1];
                        cz += positions_[gi * 3 + 2];
                    }
                    const float inv_n = 1.0f / static_cast<float>(group.size());
                    cx *= inv_n;
                    cy *= inv_n;
                    cz *= inv_n;

                    // Find nearest unvisited camera to centroid
                    size_t best = SIZE_MAX;
                    float best_dist = std::numeric_limits<float>::max();
                    for (size_t i = 0; i < size_; ++i) {
                        if (visited[i])
                            continue;
                        const float dx = positions_[i * 3 + 0] - cx;
                        const float dy = positions_[i * 3 + 1] - cy;
                        const float dz = positions_[i * 3 + 2] - cz;
                        const float d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 < best_dist) {
                            best_dist = d2;
                            best = i;
                        }
                    }

                    if (best == SIZE_MAX)
                        break;

                    group.push_back(best);
                    visited[best] = true;
                }

                // Accumulate intra-group distance for stats
                if (group.size() > 1) {
                    double group_dist = 0.0;
                    int pairs = 0;
                    for (size_t a = 0; a < group.size(); ++a) {
                        for (size_t b = a + 1; b < group.size(); ++b) {
                            const float dx = positions_[group[a] * 3] - positions_[group[b] * 3];
                            const float dy = positions_[group[a] * 3 + 1] - positions_[group[b] * 3 + 1];
                            const float dz = positions_[group[a] * 3 + 2] - positions_[group[b] * 3 + 2];
                            group_dist += std::sqrt(dx * dx + dy * dy + dz * dz);
                            ++pairs;
                        }
                    }
                    total_intra_dist += group_dist / pairs;
                    ++group_count;
                }

                // Shuffle within the group
                std::shuffle(group.begin(), group.end(), rng);

                indices_.insert(indices_.end(), group.begin(), group.end());
            }

            assert(indices_.size() == size_);
            avg_intra_group_dist_ =
                group_count > 0 ? static_cast<float>(total_intra_dist / group_count) : 0.0f;
        }

        size_t size_;
        std::vector<float> positions_; // [N*3] flat CPU positions
        int group_size_;
        size_t index_;
        std::vector<size_t> indices_;
        float avg_intra_group_dist_ = 0.0f;
    };

} // namespace lfs::training

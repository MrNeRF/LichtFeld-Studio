/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Build-time generator of the scalar STBN volume used by SteadySplats sampling.
// No third-party code or textures are used. Implements the generalized
// void-and-cluster construction in Wolfe et al., Spatiotemporal Blue Noise
// Masks (2022), supplemental section 4.2:
// https://cseweb.ucsd.edu/~ravir/stbn_supplemental.pdf
// The energy is a toroidal XY Gaussian plus a toroidal Z Gaussian, NOT a 3D
// Gaussian, so each XY slice and each pixel's temporal sequence are blue.
// Sigma 1.9, radius 8, integer weights and a fixed seed make it deterministic.
// Writes a C initializer of side*side*frames bytes in Z/Y/X order.
// Run: generate_stbn output.inc [side=128] [frames=64]

#include "core/native_fs.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    class Mask {
    public:
        Mask(std::uint32_t side, std::uint32_t frames)
            : side_(side), frames_(frames), count_(volumeSize(side, frames)), energy_(count_), occupied_(count_), extrema_(2 * count_) {
            // Integer energy makes incremental updates exact and deterministic.
            constexpr double sigma = 1.9;
            constexpr int radius = 8;
            for (int y = -radius; y <= radius; ++y)
                for (int x = -radius; x <= radius; ++x)
                    kernel_.push_back({x, y, 0, weight(x * x + y * y, sigma)});
            for (int z = -radius; z <= radius; ++z)
                kernel_.push_back({0, 0, z, weight(z * z, sigma)});
        }

        std::vector<std::uint8_t> generate() {
            std::mt19937 random(0x5354424eu);
            std::vector<std::uint32_t> order(count_);
            std::iota(order.begin(), order.end(), 0u);
            // Specify the shuffle rather than depending on std::shuffle's library.
            for (std::uint32_t i = count_ - 1; i > 0; --i)
                std::swap(order[i], order[bounded(random, i + 1)]);
            const auto initial_count = count_ / 10u;
            for (std::uint32_t i = 0; i < initial_count; ++i)
                occupied_[order[i]] = true;
            rebuild();
            for (;;) {
                const auto cluster = extrema_[1].maximum;
                set(cluster, false);
                const auto empty = extrema_[1].minimum;
                set(empty, true);
                if (cluster == empty)
                    break;
            }
            const auto initial = occupied_;
            std::vector<std::uint32_t> ranks(count_);
            for (auto rank = initial_count; rank > 0; --rank) {
                const auto cluster = extrema_[1].maximum;
                ranks[cluster] = rank - 1;
                set(cluster, false);
            }
            occupied_ = initial;
            rebuild();
            for (auto rank = initial_count; rank < count_ / 2; ++rank) {
                const auto empty = extrema_[1].minimum;
                ranks[empty] = rank;
                set(empty, true);
            }
            for (auto& bit : occupied_)
                bit = !bit;
            rebuild();
            for (auto rank = count_ / 2; rank < count_; ++rank) {
                const auto cluster = extrema_[1].maximum;
                ranks[cluster] = rank;
                set(cluster, false);
            }
            std::vector<std::uint8_t> result(count_);
            std::vector<std::uint32_t> histogram(256);
            const auto shift = std::bit_width(count_) - 1u - 8u;
            for (std::uint32_t i = 0; i < count_; ++i)
                ++histogram[result[i] = static_cast<std::uint8_t>(ranks[i] >> shift)];
            // Ranks form a permutation, so every byte value is equally common.
            if (std::ranges::any_of(histogram, [&](auto n) { return n != count_ / 256u; }))
                throw std::runtime_error("the ranks are not a permutation");
            return result;
        }

    private:
        static std::uint32_t volumeSize(std::uint32_t side, std::uint32_t frames) {
            if (!std::has_single_bit(side) || !std::has_single_bit(frames) || side < 32 || frames < 32 ||
                side > 512 || frames > 256 || std::uint64_t{side} * side * frames > 16777216u)
                throw std::runtime_error("use power-of-two dimensions of at least 32, at most 16M voxels");
            return side * side * frames;
        }
        static constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
        struct Extrema {
            std::uint32_t minimum = invalid;
            std::uint32_t maximum = invalid;
            bool operator==(const Extrema&) const = default;
        };
        struct Tap {
            int x, y, z;
            std::int32_t value;
        };
        static std::int32_t weight(int squared, double sigma) {
            return static_cast<std::int32_t>(std::lround(std::exp(-squared / (2.0 * sigma * sigma)) * 1048576.0));
        }
        static std::uint32_t bounded(std::mt19937& random, std::uint32_t limit) {
            const auto threshold = -limit % limit;
            std::uint32_t value;
            do {
                value = random();
            } while (value < threshold);
            return value % limit;
        }
        std::uint32_t better(std::uint32_t a, std::uint32_t b, bool maximum) const {
            if (a == invalid)
                return b;
            if (b == invalid)
                return a;
            if (energy_[a] == energy_[b])
                return std::min(a, b);
            return (maximum ? energy_[a] > energy_[b] : energy_[a] < energy_[b]) ? a : b;
        }
        Extrema merge(const Extrema& a, const Extrema& b) const {
            return {better(a.minimum, b.minimum, false), better(a.maximum, b.maximum, true)};
        }
        template <typename F>
        void neighbors(std::uint32_t index, F&& update) {
            const auto x = index % side_;
            const auto y = (index / side_) % side_;
            const auto z = index / (side_ * side_);
            for (const auto& tap : kernel_) {
                const auto other = ((z + tap.z) & (frames_ - 1)) * side_ * side_ +
                                   ((y + tap.y) & (side_ - 1)) * side_ +
                                   ((x + tap.x) & (side_ - 1));
                update(other, tap.value);
            }
        }
        void rebuild() {
            std::fill(energy_.begin(), energy_.end(), 0);
            for (std::uint32_t i = 0; i < count_; ++i)
                if (occupied_[i])
                    neighbors(i, [&](auto j, auto value) { energy_[j] += value; });
            for (std::uint32_t i = 0; i < count_; ++i)
                extrema_[count_ + i] = occupied_[i] ? Extrema{invalid, i} : Extrema{i, invalid};
            for (auto node = count_ - 1; node > 0; --node)
                extrema_[node] = merge(extrema_[2 * node], extrema_[2 * node + 1]);
        }
        void refresh(std::uint32_t index) {
            auto node = count_ + index;
            extrema_[node] = occupied_[index] ? Extrema{invalid, index} : Extrema{index, invalid};
            while ((node /= 2) != 0) {
                const auto updated = merge(extrema_[2 * node], extrema_[2 * node + 1]);
                // An unchanged winner is only unchanged in value if it is
                // not the point whose energy we just updated.
                if (updated == extrema_[node] && updated.minimum != index && updated.maximum != index)
                    break;
                extrema_[node] = updated;
            }
        }
        void set(std::uint32_t index, bool occupied) {
            occupied_[index] = occupied;
            neighbors(index, [&](auto other, auto value) {
                energy_[other] += occupied ? value : -value;
                refresh(other);
            });
        }
        std::uint32_t side_, frames_, count_;
        std::vector<std::int32_t> energy_;
        std::vector<std::uint8_t> occupied_;
        std::vector<Extrema> extrema_;
        std::vector<Tap> kernel_;
    };
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4)
            throw std::runtime_error("usage: generate_stbn output.inc [side=128] [frames=64]");
        const auto side = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 128u;
        const auto frames = argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 64u;
        const auto data = Mask(side, frames).generate();
        std::ofstream output(lfs::core::native_fs::native_path(argv[1]));
        for (std::size_t i = 0; i < data.size(); ++i)
            output << static_cast<unsigned>(data[i]) << ((i + 1) % 32 == 0 ? ",\n" : ",");
        if (!output)
            throw std::runtime_error("could not write output");
    } catch (const std::exception& e) {
        // LFS-CENSUS-OK(empty-catch): a build tool reports on stderr and fails the build step.
        std::cerr << e.what() << '\n';
        return 1;
    }
}

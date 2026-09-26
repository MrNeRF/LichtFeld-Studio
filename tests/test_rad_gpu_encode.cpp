/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The selected GPU backend emits the same RAD bytes as the CPU encoder.

#include "core/tensor_backend.hpp"
#include "io/formats/rad.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace {

    struct ChunkData {
        std::uint32_t count = 0;
        std::vector<float> means, alpha, rgb, scales, rotation, shN;
        std::vector<std::uint16_t> child_count;
        std::vector<std::uint32_t> child_start;
    };

    // Pack-domain chunk fixtures spanning the encoder edge cases. `kind`
    // selects the alpha branch and degenerate planes.
    ChunkData makeChunk(const std::uint32_t count, const int sh_coeffs,
                        const int kind, const unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> pos(-80.0f, 80.0f);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::uniform_real_distribution<float> sh(-0.35f, 0.35f);
        std::uniform_real_distribution<float> scale(1.0e-5f, 0.2f);
        std::uniform_real_distribution<float> quat(-1.0f, 1.0f);

        ChunkData c;
        c.count = count;
        c.means.resize(count * 3);
        c.alpha.resize(count);
        c.rgb.resize(count * 3);
        c.scales.resize(count * 3);
        c.rotation.resize(count * 4);
        if (sh_coeffs > 0) {
            c.shN.resize(static_cast<std::size_t>(count) * sh_coeffs * 3);
        }
        c.child_count.assign(count, 0);
        c.child_start.assign(count, 0);

        for (std::uint32_t i = 0; i < count; ++i) {
            c.means[i * 3 + 0] = pos(rng);
            c.means[i * 3 + 1] = pos(rng);
            c.means[i * 3 + 2] = pos(rng) * 0.2f;
            switch (kind) {
            case 1: // f16 alpha branch: values above 1 (merged interiors)
                c.alpha[i] = 0.5f + 1.5f * unit(rng);
                break;
            case 3: // r8/f16 decision boundary: max exactly 1.0
                c.alpha[i] = 1.0f;
                break;
            default:
                c.alpha[i] = unit(rng);
                break;
            }
            for (int d = 0; d < 3; ++d) {
                c.rgb[i * 3 + d] = kind == 2 ? 0.7f : unit(rng);
                c.scales[i * 3 + d] = scale(rng);
            }
            float q[4] = {1.0f + quat(rng), quat(rng), quat(rng), quat(rng)};
            const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            for (int d = 0; d < 4; ++d) {
                c.rotation[i * 4 + d] = q[d] / n;
            }
            for (int k = 0; k < sh_coeffs * 3; ++k) {
                c.shN[static_cast<std::size_t>(i) * sh_coeffs * 3 + k] = sh(rng) * 0.5f;
            }
        }

        if (count >= 16 && kind == 0) {
            c.means[0] = -0.0f;
            c.means[1] = 1.0e20f;
            c.means[2] = 1.0e-20f;
            c.alpha[1] = 0.0f;
            c.alpha[2] = 1.0f;
            c.alpha[3] = 0.5f;
            c.rgb[3] = 0.0f;
            c.rgb[4] = 1.0f;
            c.rgb[5] = 0.5f;
            if (!c.shN.empty()) {
                c.shN[0] = 0.0f;
                c.shN[1] = 1.0e-41f; // subnormal: catches FTZ divergence
                c.shN[2] = -0.0f;
                c.shN[3] = 0.35f;
            }
        }
        return c;
    }

    std::vector<ChunkData> makeChunkSet(const int sh_coeffs, const std::size_t n_chunks = 5) {
        std::vector<ChunkData> chunks;
        for (std::size_t i = 0; i + 1 < n_chunks; ++i) {
            chunks.push_back(makeChunk(2048, sh_coeffs, static_cast<int>(i % 4),
                                       static_cast<unsigned>(101 * (i + 1))));
        }
        chunks.push_back(makeChunk(1234, sh_coeffs, 0, 505));
        return chunks;
    }

    void writeRadFile(const std::filesystem::path& path,
                      const std::vector<ChunkData>& chunks,
                      const int sh_degree,
                      const int sh_coeffs,
                      const bool gpu) {
        std::uint64_t total = 0;
        for (const auto& c : chunks) {
            total += c.count;
        }
        lfs::io::RadStreamWriter writer(
            path, total, sh_degree, /*lod_tree=*/true,
            /*compression_level=*/6, /*emit_meta_sidecar=*/false,
            lfs::io::kRadNativeChunkSplats,
            gpu ? lfs::io::RadGpuQuantization::Auto
                : lfs::io::RadGpuQuantization::Disabled);
        ASSERT_TRUE(writer.open().has_value());
        std::vector<lfs::io::RadStreamChunkSource> sources(chunks.size());
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            const auto& c = chunks[i];
            sources[i] = {
                .count = c.count,
                .means = c.means.data(),
                .alpha = c.alpha.data(),
                .rgb = c.rgb.data(),
                .scales = c.scales.data(),
                .rotation = c.rotation.data(),
                .shN = sh_coeffs > 0 ? c.shN.data() : nullptr,
                .child_count = c.child_count.data(),
                .child_start = c.child_start.data(),
            };
        }
        ASSERT_TRUE(writer.append_batch(sources).has_value());
        ASSERT_TRUE(writer.finish().has_value());
    }

    std::vector<std::uint8_t> readFile(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        EXPECT_TRUE(in.good());
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    void expectIdenticalFiles(const int sh_degree, const int sh_coeffs,
                              const std::size_t n_chunks = 5) {
        const auto dir = std::filesystem::temp_directory_path() / "rad_gpu_encode_test";
        std::filesystem::create_directories(dir);
        const auto cpu_path = dir / "cpu.rad";
        const auto gpu_path = dir / "gpu.rad";

        const auto chunks = makeChunkSet(sh_coeffs, n_chunks);
        writeRadFile(cpu_path, chunks, sh_degree, sh_coeffs, false);
        writeRadFile(gpu_path, chunks, sh_degree, sh_coeffs, true);

        const auto cpu_bytes = readFile(cpu_path);
        const auto gpu_bytes = readFile(gpu_path);
        ASSERT_GT(cpu_bytes.size(), 0u);
        EXPECT_EQ(cpu_bytes.size(), gpu_bytes.size());
        if (cpu_bytes.size() == gpu_bytes.size()) {
            const auto mismatch = std::mismatch(cpu_bytes.begin(), cpu_bytes.end(), gpu_bytes.begin());
            EXPECT_TRUE(mismatch.first == cpu_bytes.end())
                << "First RAD byte mismatch at " << std::distance(cpu_bytes.begin(), mismatch.first);
        }

        std::filesystem::remove_all(dir);
    }

    class RadTensorEncodeTest : public ::testing::TestWithParam<lfs::core::GpuBackend> {};

    TEST_P(RadTensorEncodeTest, StreamWriterMatchesCpu) {
        const auto backend = GetParam();
        if (!lfs::core::gpu_backend_available(backend)) {
            GTEST_SKIP() << "GPU backend unavailable";
        }
        const lfs::core::GpuBackendScope scope(backend);
        expectIdenticalFiles(3, 15);
        expectIdenticalFiles(1, 3);
        expectIdenticalFiles(0, 0);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, RadTensorEncodeTest,
                             ::testing::Values(lfs::core::GpuBackend::CUDA,
                                               lfs::core::GpuBackend::Vulkan));

} // namespace

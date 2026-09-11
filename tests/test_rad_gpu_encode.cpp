/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// CPU/GPU bit-parity for the RAD encode quantization path. The CUDA
// quantizer must reproduce the CPU PropertyEncoder Auto-profile planes
// (center f32_lebytes, alpha r8/f16, rgb r8_delta, shN s8) byte for byte:
// once against in-test references of the encoder formulas, and end to end
// by asserting RadStreamWriter emits identical files with the GPU path
// enabled and disabled. Fixtures cover both alpha branches, the rgb
// degenerate-range guard, subnormals (no FTZ on the GPU side), and a
// partial trailing chunk.

#include "io/cuda/rad_encode_quant.hpp"
#include "io/formats/rad.hpp"
#include "io/formats/rad_dequant_math.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace {

    namespace radmath = lfs::io::radmath;

    bool cudaAvailable() {
        int n = 0;
        return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    }

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
        EXPECT_TRUE(cpu_bytes == gpu_bytes) << "GPU-encoded RAD file differs from CPU encode";

        std::filesystem::remove_all(dir);
    }

    // ====================================================================
    // CPU references of the PropertyEncoder formulas (rad.cpp), used to
    // assert the quantizer planes directly so a silent CPU fallback inside
    // the writer can't mask a kernel regression.
    // ====================================================================

    std::vector<std::uint8_t> refCenterLeBytes(const std::vector<float>& data,
                                               const std::size_t count) {
        std::vector<std::uint8_t> out(count * 12);
        const std::size_t stride = count * 3;
        for (std::size_t b = 0; b < 4; ++b) {
            for (std::size_t d = 0; d < 3; ++d) {
                for (std::size_t i = 0; i < count; ++i) {
                    std::uint32_t bits;
                    std::memcpy(&bits, &data[i * 3 + d], 4);
                    out[b * stride + d * count + i] =
                        static_cast<std::uint8_t>((bits >> (8 * b)) & 0xFFu);
                }
            }
        }
        return out;
    }

    std::vector<std::uint8_t> refR8(const float* data, const std::size_t dims,
                                    const std::size_t count, const float min_val,
                                    const float max_val) {
        float range = max_val - min_val;
        if (range < 1e-7f) {
            range = 1e-7f;
        }
        std::vector<std::uint8_t> out(count * dims);
        std::size_t idx = 0;
        for (std::size_t d = 0; d < dims; ++d) {
            for (std::size_t i = 0; i < count; ++i) {
                const float normalized = (data[i * dims + d] - min_val) / range;
                out[idx++] = static_cast<std::uint8_t>(
                    std::clamp(std::round(normalized * 255.0f), 0.0f, 255.0f));
            }
        }
        return out;
    }

    std::vector<std::uint8_t> refR8Delta(const float* data, const std::size_t dims,
                                         const std::size_t count, const float min_val,
                                         const float max_val) {
        const auto q = refR8(data, dims, count, min_val, max_val);
        std::vector<std::uint8_t> out(q.size());
        for (std::size_t d = 0; d < dims; ++d) {
            std::uint8_t last = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const std::uint8_t v = q[d * count + i];
                out[d * count + i] = static_cast<std::uint8_t>(v - last);
                last = v;
            }
        }
        return out;
    }

    TEST(RadGpuEncode, QuantizerMatchesCpuFormulas) {
        if (!cudaAvailable()) {
            GTEST_SKIP() << "No CUDA device";
        }
        constexpr int kShCoeffs = 15;
        const auto chunks = makeChunkSet(kShCoeffs);

        std::vector<lfs::io::cuda::RadEncodeQuantChunkIn> in(chunks.size());
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            in[i] = {chunks[i].count, chunks[i].means.data(), chunks[i].alpha.data(),
                     chunks[i].rgb.data(), chunks[i].shN.data()};
        }
        std::vector<lfs::io::cuda::RadEncodeQuantChunkOut> out(chunks.size());
        lfs::io::cuda::RadEncodeGpuQuantizer quantizer;
        ASSERT_TRUE(quantizer.quantize_batch(in, kShCoeffs, /*lod_tree=*/true, out));

        constexpr int kBandStart[3] = {0, 3, 8};
        constexpr int kBandCoeffs[3] = {3, 5, 7};

        for (std::size_t ci = 0; ci < chunks.size(); ++ci) {
            const auto& c = chunks[ci];
            const auto& o = out[ci];
            const std::size_t count = c.count;
            SCOPED_TRACE(ci);

            const auto center = refCenterLeBytes(c.means, count);
            EXPECT_EQ(0, std::memcmp(o.center, center.data(), center.size()));

            float alpha_max = 0.0f;
            for (const float v : c.alpha) {
                alpha_max = std::max(alpha_max, v);
            }
            ASSERT_EQ(o.alpha_f16, alpha_max > 1.0f);
            if (o.alpha_f16) {
                std::vector<std::uint8_t> ref(count * 2);
                for (std::size_t i = 0; i < count; ++i) {
                    const std::uint16_t h = radmath::floatToHalf(c.alpha[i]);
                    ref[i * 2] = static_cast<std::uint8_t>(h & 0xFFu);
                    ref[i * 2 + 1] = static_cast<std::uint8_t>(h >> 8);
                }
                EXPECT_EQ(0, std::memcmp(o.alpha, ref.data(), ref.size()));
            } else {
                EXPECT_EQ(o.alpha_min, 0.0f);
                EXPECT_EQ(o.alpha_max, 2.0f);
                const auto ref = refR8(c.alpha.data(), 1, count, 0.0f, 2.0f);
                EXPECT_EQ(0, std::memcmp(o.alpha, ref.data(), ref.size()));
            }

            float rgb_min = c.rgb[0];
            float rgb_max = c.rgb[0];
            for (const float v : c.rgb) {
                rgb_min = std::min(rgb_min, v);
                rgb_max = std::max(rgb_max, v);
            }
            EXPECT_EQ(0, std::memcmp(&o.rgb_min, &rgb_min, 4));
            EXPECT_EQ(0, std::memcmp(&o.rgb_max, &rgb_max, 4));
            const auto rgb = refR8Delta(c.rgb.data(), 3, count, rgb_min, rgb_max);
            EXPECT_EQ(0, std::memcmp(o.rgb, rgb.data(), rgb.size()));

            for (int b = 0; b < 3; ++b) {
                SCOPED_TRACE(b);
                ASSERT_NE(o.sh[b], nullptr);
                const int dims = kBandCoeffs[b] * 3;
                float max_abs = 0.0f;
                for (std::size_t i = 0; i < count; ++i) {
                    for (int k = 0; k < dims; ++k) {
                        max_abs = std::max(
                            max_abs,
                            std::abs(c.shN[i * kShCoeffs * 3 + kBandStart[b] * 3 + k]));
                    }
                }
                max_abs = std::max(max_abs, 1e-6f);
                EXPECT_EQ(0, std::memcmp(&o.sh_max_abs[b], &max_abs, 4));
                std::vector<std::uint8_t> ref(count * dims);
                for (int d = 0; d < dims; ++d) {
                    for (std::size_t i = 0; i < count; ++i) {
                        const float scaled =
                            c.shN[i * kShCoeffs * 3 + kBandStart[b] * 3 + d] / max_abs * 127.0f;
                        ref[static_cast<std::size_t>(d) * count + i] =
                            static_cast<std::uint8_t>(static_cast<std::int8_t>(
                                std::clamp(std::round(scaled), -127.0f, 127.0f)));
                    }
                }
                EXPECT_EQ(0, std::memcmp(o.sh[b], ref.data(), ref.size()));
            }
        }
    }

    TEST(RadGpuEncode, StreamWriterBitIdenticalSh3) {
        if (!cudaAvailable()) {
            GTEST_SKIP() << "No CUDA device";
        }
        expectIdenticalFiles(3, 15);
    }

    // 80 chunks: a single batch larger than the converter's typical flush,
    // covering whole-batch arena sizing and out-view stability.
    TEST(RadGpuEncode, StreamWriterBitIdenticalSh1LargeBatch) {
        if (!cudaAvailable()) {
            GTEST_SKIP() << "No CUDA device";
        }
        expectIdenticalFiles(1, 3, 80);
    }

    TEST(RadGpuEncode, StreamWriterBitIdenticalSh0) {
        if (!cudaAvailable()) {
            GTEST_SKIP() << "No CUDA device";
        }
        expectIdenticalFiles(0, 0);
    }

} // namespace

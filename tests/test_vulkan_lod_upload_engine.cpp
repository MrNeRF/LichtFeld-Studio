/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/rad_dequant_math.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/lod_upload_engine.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <thread>

namespace {
    using namespace lfs::core;
    using lfs::vis::LodUploadEngine;
    namespace math = lfs::core::radmath;
    constexpr std::size_t kPage = lfs::vis::LodPageCache::kChunkSplats;
    constexpr std::size_t kPages = 32;
    constexpr std::array<std::size_t, 9> kSizes{
        kPage * 12, kPage * 8, kPage * 48, kPage * 8, kPage * 8,
        kPage * 2, 64, kPage * 8, kPage * 12};

    struct Packet {
        RadPagePackedDesc desc;
        std::vector<std::uint8_t> bytes;
        void add(RadPackedKind kind, RadPackedEncoding encoding, std::span<const std::uint8_t> plane,
                 float lo = 0, float hi = 1, float scale = 1) {
            bytes.resize((bytes.size() + 15) & ~std::size_t(15), 0);
            auto& p = desc.props[desc.property_count++];
            p.kind = static_cast<std::uint32_t>(kind);
            p.encoding = static_cast<std::uint32_t>(encoding);
            p.plane_offset = bytes.size();
            p.plane_bytes = plane.size();
            p.min_val = lo;
            p.max_val = hi;
            p.scale = scale;
            bytes.insert(bytes.end(), plane.begin(), plane.end());
            desc.used_bytes = bytes.size();
        }
    };

    Packet meansPacket(float value, std::uint32_t count = 1025) {
        Packet p;
        p.desc.count = count;
        std::vector<float> values(count * 3, value);
        p.add(RadPackedKind::Means, RadPackedEncoding::F32,
              {reinterpret_cast<const std::uint8_t*>(values.data()), values.size() * sizeof(float)});
        return p;
    }

    class LodUpload : public testing::TestWithParam<GpuBackend> {
    protected:
        std::unique_ptr<GpuBackendScope> backend;
        std::array<Tensor, 9> tensors;
        std::unique_ptr<LodUploadEngine> engine;
        LodUploadEngine::DeviceLayout layout;
        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Requested GPU backend unavailable";
            backend = std::make_unique<GpuBackendScope>(GetParam());
            layout.pool.page_splats = kPage;
            layout.pool.sh_slots = 12;
            for (size_t i = 0; i < 9; ++i) {
                tensors[i] = Tensor::empty({kSizes[i] * kPages}, Device::GPU, DataType::UInt8);
                layout.pool.regions[i] = tensors[i];
            }
            engine = std::make_unique<LodUploadEngine>();
            (void)engine->configure(layout);
        }
        void TearDown() override {
            engine.reset();
            layout = {};
            tensors = {};
            backend.reset();
        }
        void submit(const Packet& packet, std::uint32_t page = 0, std::uint64_t generation = 42) {
            auto* slot = engine->acquireStagingSlot();
            ASSERT_NE(slot, nullptr);
            std::memcpy(slot->data, packet.bytes.data(), packet.bytes.size());
            engine->submitPackedPage(slot, packet.desc, page, generation);
        }
        template <typename T>
        std::vector<T> read(std::size_t region) {
            const auto cpu = tensors[region].cpu();
            std::vector<T> result(cpu.bytes() / sizeof(T));
            std::memcpy(result.data(), cpu.data_ptr(), cpu.bytes());
            return result;
        }
        void complete() {
            const auto results = engine->drainAndSync();
            ASSERT_EQ(results.size(), 1);
            ASSERT_TRUE(results[0].error.empty()) << results[0].error;
            EXPECT_EQ(results[0].generation, 42);
            EXPECT_GT(engine->lastPublishedSignalValue(), 0);
            EXPECT_TRUE(engine->idle());
        }
    };

    TEST_P(LodUpload, FloatPlanesAndOddTailsOverwriteReusedPages) {
        for (std::uint32_t enc = 0; enc < 8; ++enc) {
            SCOPED_TRACE(enc);
            Packet p;
            p.desc.count = 1025;
            const std::size_t n = p.desc.count * 3;
            const std::size_t width = enc <= 1 ? 4 : (enc <= 3 || enc == 7 ? 2 : 1);
            std::vector<std::uint8_t> plane(n * width);
            std::vector<float> expected(n);
            for (std::size_t e = 0; e < n; ++e) {
                const float value = float(int(e % 31) - 15) * 0.125f;
                const auto half = math::floatToHalf(value);
                if (enc <= 1) {
                    const auto bits = std::bit_cast<std::uint32_t>(value);
                    for (std::size_t b = 0; b < 4; ++b)
                        plane[enc == 0 ? e * 4 + b : b * n + e] = bits >> (b * 8);
                    expected[e] = value;
                } else if (enc <= 3 || enc == 7) {
                    for (std::size_t b = 0; b < 2; ++b)
                        plane[enc == 3 ? b * n + e : e * 2 + b] = half >> (b * 8);
                    expected[e] = enc == 7 ? std::exp(math::halfToFloat(half)) : math::halfToFloat(half);
                } else {
                    plane[e] = e % 256;
                    if (enc == 4)
                        expected[e] = math::dequantR8(plane[e], -2, 5);
                    if (enc == 5)
                        expected[e] = math::dequantS8(static_cast<std::int8_t>(plane[e]), 3);
                    if (enc == 6)
                        expected[e] = math::dequantLn0R8(plane[e], -2, 3);
                }
            }
            p.add(RadPackedKind::Means, static_cast<RadPackedEncoding>(enc), plane, -2, 3, 1);
            submit(p);
            complete();
            const auto means = read<float>(0);
            for (std::size_t i = 0; i < kPage; ++i)
                for (std::size_t d = 0; d < 3; ++d) {
                    const float want = i < p.desc.count ? expected[d * p.desc.count + i] : 0;
                    if (enc >= 4)
                        ASSERT_NEAR(means[i * 3 + d], want, std::max(1e-6f, std::abs(want) * 3e-6f));
                    else
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(means[i * 3 + d]), std::bit_cast<std::uint32_t>(want))
                            << "splat=" << i << " dim=" << d << " actual=" << means[i * 3 + d] << " expected=" << want;
                }
            const auto rotation = read<std::uint16_t>(3);
            const auto sh = read<std::uint8_t>(2);
            const auto links = read<std::uint32_t>(8);
            for (std::size_t i = 0; i < kPage; ++i) {
                ASSERT_EQ(rotation[i * 4], 0x3c00);
                for (std::size_t d = 1; d < 4; ++d)
                    ASSERT_EQ(rotation[i * 4 + d], 0);
                for (std::size_t d = 0; d < 3; ++d)
                    ASSERT_EQ(links[i * 3 + d], 0xffffffff);
            }
            ASSERT_TRUE(std::all_of(sh.begin(), sh.begin() + kSizes[2], [](auto b) { return b == 0; }));
        }
    }

    TEST_P(LodUpload, QuantizedAttributesFramesAndMetadataMatchTheFileCodec) {
        auto p = meansPacket(4);
        const auto n = p.desc.count;
        p.desc.sh_coeffs_rest = 15;
        p.desc.lod_opacity = 1;
        const std::vector<std::uint8_t> rgb(n * 3, 127), alpha(n, 219);
        p.add(RadPackedKind::Sh0, RadPackedEncoding::R8, rgb);
        p.add(RadPackedKind::Alpha, RadPackedEncoding::R8, alpha);
        std::vector<std::uint16_t> scale(n * 3, 0xc100); // log scale -2.5
        p.add(RadPackedKind::Scales, RadPackedEncoding::LnF16,
              {reinterpret_cast<const std::uint8_t*>(scale.data()), scale.size() * 2});
        std::vector<std::uint8_t> rotation(n * 3);
        for (std::size_t i = 0; i < n; ++i) {
            rotation[i * 3] = i % 256;
            rotation[i * 3 + 1] = (i * 37) % 256;
            rotation[i * 3 + 2] = (i * 17) % 256;
        }
        p.add(RadPackedKind::Rotation, RadPackedEncoding::Oct88R8, rotation);
        for (std::uint32_t band = 0; band < 3; ++band) {
            std::vector<std::uint8_t> sh(n * (band * 6 + 9));
            for (std::size_t i = 0; i < sh.size(); ++i)
                sh[i] = (i * 17 + band * 43) % 255;
            p.add(static_cast<RadPackedKind>(5 + band), RadPackedEncoding::S8, sh, -0.5, 0.5, 0.5);
        }
        p.bytes.resize((p.bytes.size() + 15) & ~std::size_t(15));
        p.desc.meta_node_count = n;
        p.desc.meta_bounds_offset = p.bytes.size();
        std::vector<std::uint32_t> metadata(n * 5);
        for (std::size_t i = 0; i < metadata.size(); ++i)
            metadata[i] = 0x10001000 + i;
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(metadata.data());
        p.bytes.insert(p.bytes.end(), bytes, bytes + metadata.size() * 4);
        p.desc.meta_links_offset = p.desc.meta_bounds_offset + n * 8;
        p.desc.used_bytes = p.bytes.size();
        p.desc.frame = {{-3, 4, 2}, {2, 5, 8}, -4, 9};
        submit(p, 3);
        complete();
        const auto sh0 = read<std::uint16_t>(1), scales = read<std::uint16_t>(4),
                   opacity = read<std::uint16_t>(5), quats = read<std::uint16_t>(3);
        const auto sh = read<std::uint8_t>(2);
        const auto bounds = read<std::uint32_t>(7), links = read<std::uint32_t>(8);
        for (std::size_t i = 0; i < kPage; ++i) {
            const auto dst = kPage * 3 + i;
            for (std::size_t d = 0; d < 3; ++d) {
                ASSERT_EQ(sh0[dst * 4 + d], i < n ? math::floatToHalf(math::sh0Transform(math::dequantR8(127, 0, 1))) : 0);
                ASSERT_EQ(scales[dst * 4 + d], i < n ? 0xc100 : 0);
            }
            ASSERT_EQ(opacity[dst], i < n ? math::floatToHalf(math::dequantR8(219, 0, 1)) : 0);
            float xyzw[4] = {0, 0, 0, 1};
            if (i < n)
                math::dequantQuatOct88R8(rotation[i * 3], rotation[i * 3 + 1], rotation[i * 3 + 2], xyzw);
            for (std::size_t d = 0; d < 4; ++d)
                ASSERT_NEAR(math::halfToFloat(quats[dst * 4 + d]), xyzw[(d + 3) % 4], 0.001f);
            for (std::size_t c = 0; c < 48; ++c) {
                const auto index = ((dst / 32 * 12 + c / 4) * 32 + dst % 32) * 4 + c % 4;
                std::uint8_t want = 0;
                if (i < n && c < 45) {
                    const auto band = c < 9 ? 0 : (c < 24 ? 1 : 2);
                    const auto local = c - (band == 0 ? 0 : (band == 1 ? 9 : 24));
                    want = ((local * n + i) * 17 + band * 43) % 255;
                }
                ASSERT_EQ(sh[index], want);
            }
            for (std::size_t d = 0; d < 2; ++d)
                ASSERT_EQ(bounds[dst * 2 + d], i < n ? metadata[i * 2 + d] : 0);
            for (std::size_t d = 0; d < 3; ++d)
                ASSERT_EQ(links[dst * 3 + d], i < n ? metadata[n * 2 + i * 3 + d] : 0xffffffff);
        }
        const auto frames = read<float>(6);
        const std::array<float, 16> expected{0.5, 0.5, 0.5, 0, -3, 4, 2, -4, 2, 5, 8, 9, 0, 0, 0, 0};
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), frames.begin() + 3 * 16));
    }

    TEST_P(LodUpload, ConcurrentProducersReuseSlotsWithoutPublishingPartialPages) {
        std::atomic_uint failures{0};
        std::vector<std::thread> workers;
        for (std::uint32_t worker = 0; worker < 8; ++worker)
            workers.emplace_back([&, worker] {
                for (std::uint32_t page = worker; page < kPages; page += 8) {
                    const auto p = meansPacket(float(page + 1));
                    auto* slot = engine->acquireStagingSlot();
                    if (!slot) {
                        ++failures;
                        return;
                    }
                    std::memcpy(slot->data, p.bytes.data(), p.bytes.size());
                    auto desc = p.desc;
                    desc.chunk = page + 100;
                    engine->submitPackedPage(slot, desc, page, 9);
                }
            });
        for (auto& worker : workers)
            worker.join();
        EXPECT_EQ(failures.load(), 0);
        auto results = engine->collectPublished();
        auto rest = engine->drainAndSync();
        results.insert(results.end(), rest.begin(), rest.end());
        ASSERT_EQ(results.size(), kPages);
        for (const auto& result : results) {
            ASSERT_TRUE(result.error.empty()) << result.error;
            EXPECT_EQ(result.chunk, result.page + 100);
            EXPECT_EQ(result.generation, 9);
        }
        const auto means = read<float>(0);
        for (std::size_t page = 0; page < kPages; ++page)
            for (std::size_t i = 0; i < kPage; ++i)
                ASSERT_EQ(means[(page * kPage + i) * 3], i < 1025 ? float(page + 1) : 0);
        EXPECT_GT(engine->lastPublishedSignalValue(), 0);
    }

    TEST_P(LodUpload, InvalidPagesReleaseTheirSlotsAndKeepTheEngineUsable) {
        for (int invalid = 0; invalid < 5; ++invalid) {
            auto p = meansPacket(7);
            if (invalid == 0)
                p.desc.count = kPage + 1;
            if (invalid == 1)
                p.desc.property_count = 9;
            if (invalid == 2)
                p.desc.props[0].plane_offset = p.desc.used_bytes;
            if (invalid == 3)
                p.desc.meta_node_count = kPage;
            submit(p, invalid == 4 ? kPages : 0);
            const auto results = engine->drainAndSync();
            ASSERT_EQ(results.size(), 1);
            EXPECT_FALSE(results[0].error.empty());
            EXPECT_EQ(engine->lastPublishedSignalValue(), 0);
        }
        submit(meansPacket(7));
        complete();
        EXPECT_EQ(read<float>(0)[0], 7);
        const auto semaphore = engine->timeline();
        (void)engine->configure({});
        EXPECT_EQ(engine->acquireStagingSlot(), nullptr);
        (void)engine->configure(layout);
        submit(meansPacket(9));
        complete();
        EXPECT_EQ(read<float>(0)[0], 9);
        (void)semaphore;
    }
    TEST_P(LodUpload, OctahedralRotationHalfBitsMatchAcrossBackends) {
        if (GetParam() != GpuBackend::CUDA)
            GTEST_SKIP() << "Cross-backend parity is checked once";
        const auto other = GpuBackend::Vulkan;
        if (!gpu_backend_available(other))
            GTEST_SKIP() << "Cross-backend parity needs both devices";
        constexpr uint32_t pages = 4;
        std::array<Packet, pages> packets;
        for (uint32_t page = 0; page < pages; ++page) {
            auto& packet = packets[page];
            packet.desc.count = kPage;
            std::vector<uint8_t> rotations(kPage * 3);
            for (size_t i = 0; i < kPage; ++i) {
                const size_t global = page * kPage + i;
                rotations[i * 3] = global & 255;
                rotations[i * 3 + 1] = (global >> 5) & 255;
                rotations[i * 3 + 2] = (global * 37) & 255;
            }
            packet.add(RadPackedKind::Rotation, RadPackedEncoding::Oct88R8, rotations);
            submit(packet, page);
            complete();
        }
        const auto actual = read<uint16_t>(3);
        std::vector<uint16_t> expected;
        {
            GpuBackendScope scope(other);
            LodUploadEngine reference;
            LodUploadEngine::DeviceLayout reference_layout;
            reference_layout.pool.page_splats = kPage;
            for (size_t i = 0; i < 9; ++i) {
                if (i != 2)
                    reference_layout.pool.regions[i] = Tensor::empty({kSizes[i] * pages}, Device::GPU, DataType::UInt8);
            }
            (void)reference.configure(reference_layout);
            for (uint32_t page = 0; page < pages; ++page) {
                auto* slot = reference.acquireStagingSlot();
                ASSERT_NE(slot, nullptr);
                std::memcpy(slot->data, packets[page].bytes.data(), packets[page].bytes.size());
                reference.submitPackedPage(slot, packets[page].desc, page, 42);
            }
            const auto results = reference.drainAndSync();
            ASSERT_EQ(results.size(), pages);
            for (const auto& result : results)
                ASSERT_TRUE(result.error.empty()) << result.error;
            const auto cpu = reference_layout.pool.regions[3].cpu();
            const auto* data = static_cast<const uint16_t*>(cpu.data_ptr());
            expected.assign(data, data + pages * kPage * 4);
        }
        size_t mismatches = 0;
        for (size_t i = 0; i < expected.size(); ++i)
            if (actual[i] != expected[i]) {
                if (mismatches < 8)
                    ADD_FAILURE() << "component=" << i << " actual=" << actual[i] << " expected=" << expected[i];
                ++mismatches;
            }
        EXPECT_EQ(mismatches, 0);
    }

    TEST_P(LodUpload, ResidentPagesMatchHalfCodecAndClearSlack) {
        RadPageSources source;
        const uint32_t n = 1025;
        source.means = Tensor::full({n, 3}, 2.5f, Device::GPU);
        source.sh0 = Tensor::full({n, 3}, -0.25f, Device::GPU);
        source.rotation = Tensor::full({n, 4}, 0.5f, Device::GPU);
        source.scaling = Tensor::full({n, 3}, -2.5f, Device::GPU);
        source.opacity = Tensor::full({n}, 0.75f, Device::GPU);
        source.shN = Tensor::full({((n + 31) / 32) * 32 * 3 * 4}, 0.25f, Device::GPU);
        source.sh_rest = 3;
        source.count = n;
        const std::array pages{LodUploadEngine::ResidentPage{1, 0, n}};
        engine->quantizeResident(source, pages).wait();
        const auto means = read<float>(0);
        const auto sh = read<uint8_t>(2);
        const auto rgb = read<uint16_t>(1), rot = read<uint16_t>(3), scale = read<uint16_t>(4), alpha = read<uint16_t>(5);
        for (size_t i = 0; i < kPage; ++i) {
            const auto dst = kPage + i;
            for (size_t c = 0; c < 3; ++c) {
                ASSERT_EQ(means[dst * 3 + c], i < n ? 2.5f : 0);
                ASSERT_EQ(rgb[dst * 4 + c], i < n ? math::floatToHalf(-0.25f) : 0);
                ASSERT_EQ(scale[dst * 4 + c], i < n ? math::floatToHalf(-2.5f) : 0);
            }
            ASSERT_EQ(alpha[dst], i < n ? math::floatToHalf(0.75f) : 0);
            for (size_t c = 0; c < 4; ++c)
                ASSERT_EQ(rot[dst * 4 + c], i < n ? math::floatToHalf(0.5f) : (c == 0 ? 0x3c00 : 0));
        }
        for (size_t i = 0; i < kPage; ++i)
            for (size_t c = 0; c < 12; ++c) {
                const size_t dst = kPage + i;
                const size_t index = ((dst / 32 * 12 + c / 4) * 32 + dst % 32) * 4 + c % 4;
                ASSERT_EQ(sh[index], i < n && c < 9 ? 127 : 0) << "splat=" << i << " component=" << c;
            }
        const auto frames = read<float>(6);
        EXPECT_EQ(frames[16], 0.25f);
        EXPECT_EQ(frames[17], 0);
    }

    INSTANTIATE_TEST_SUITE_P(TensorBackends, LodUpload, testing::ValuesIn(kGpuBackends),
                             [](const testing::TestParamInfo<GpuBackend>& p) { return p.param == GpuBackend::CUDA ? "CUDA" : "Vulkan"; });
} // namespace

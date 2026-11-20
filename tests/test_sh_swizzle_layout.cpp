/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/cuda/emeans.hpp"
#include "io/formats/ply.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

namespace {

    // Look for a real trained PLY we can use; skip cleanly if none is available.
    [[nodiscard]] std::filesystem::path find_real_ply() {
        for (const char* candidate : {
                 "/tmp/swizzle_fold/splat_7000.ply",
                 "output_swizzle_test/splat_7000.ply",
                 "output_swizzle_test/splat_5.ply",
             }) {
            if (std::filesystem::exists(candidate))
                return candidate;
        }
        return {};
    }

    // Round-trip the shN buffer: canonical -> swizzled -> canonical and verify equality.
    // Layout bijects the 45 active floats per primitive onto 12 float4 slots (48 floats
    // with 3 of tail padding); the active range round-trips bitwise.
    TEST(ShSwizzleLayout, CanonicalRoundTrip_RealData) {
        using namespace lfs::core;

        const auto ply_path = find_real_ply();
        if (ply_path.empty()) {
            GTEST_SKIP() << "No trained PLY available; run the smoke test first";
        }

        auto loaded = lfs::io::load_ply(ply_path);
        ASSERT_TRUE(loaded.has_value()) << "Failed to load PLY at " << ply_path;

        SplatData splat = std::move(loaded->value);
        if (splat.get_active_sh_degree() == 0) {
            GTEST_SKIP() << "PLY has SH degree 0, no shN to round-trip";
        }

        const Tensor canonical = splat.shN_canonical();
        ASSERT_TRUE(canonical.is_valid());
        ASSERT_EQ(canonical.ndim(), 3);

        const size_t N = canonical.shape()[0];
        const std::uint32_t K = static_cast<std::uint32_t>(canonical.shape()[1]);
        ASSERT_GT(N, 0u);
        ASSERT_GT(K, 0u);
        ASSERT_EQ(canonical.shape()[2], 3u);

        // Build a swizzled buffer from the canonical view.
        const size_t swizzled_floats = sh_swizzled_float_count(N, K);
        Tensor swizzled = Tensor::zeros({swizzled_floats}, Device::CUDA);
        reorder_sh_to_swizzled(canonical.ptr<float>(), swizzled.ptr<float>(), N, K);

        // Deswizzle back into a fresh canonical buffer.
        Tensor recovered = Tensor::empty({N, K, 3}, Device::CUDA);
        undo_reorder_sh_from_swizzled(swizzled.ptr<float>(), recovered.ptr<float>(), N, K);
        cudaDeviceSynchronize();

        // Bitwise compare.
        auto cpu_canonical = canonical.contiguous().to(Device::CPU);
        auto cpu_recovered = recovered.contiguous().to(Device::CPU);
        const float* a = cpu_canonical.ptr<float>();
        const float* b = cpu_recovered.ptr<float>();
        const size_t total = N * K * 3;

        size_t mismatches = 0;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < total; ++i) {
            const float d = std::fabs(a[i] - b[i]);
            if (d > 0.0f) {
                ++mismatches;
                if (d > max_abs_diff)
                    max_abs_diff = d;
            }
        }
        EXPECT_EQ(mismatches, 0u) << "max_abs_diff=" << max_abs_diff;
    }

    TEST(ShSwizzleLayout, CompactStorageMatchesActiveDegree) {
        using namespace lfs::core;
        constexpr std::size_t N = 70;
        constexpr std::size_t BLOCKS = 3;

        EXPECT_EQ(sh_float4_slots_for_rest(0), 0u);
        EXPECT_EQ(sh_float4_slots_for_rest(3), 3u);
        EXPECT_EQ(sh_float4_slots_for_rest(8), 6u);
        EXPECT_EQ(sh_float4_slots_for_rest(15), 12u);

        EXPECT_EQ(sh_swizzled_float_count(N, 0), 0u);
        EXPECT_EQ(sh_swizzled_float_count(N, 3), BLOCKS * 3u * kShReorderSize * 4u);
        EXPECT_EQ(sh_swizzled_float_count(N, 8), BLOCKS * 6u * kShReorderSize * 4u);
        EXPECT_EQ(sh_swizzled_float_count(N, 15), BLOCKS * 12u * kShReorderSize * 4u);
        EXPECT_EQ(sh_swizzled_float_count(N), sh_swizzled_float_count(N, 15));
    }

    // Verify the float4 shuffle: the swizzled buffer's float4 at sh_swizzled_index(p, k)
    // holds 4 consecutive floats of the linear canonical row for primitive p (with the last
    // slot's tail floats zero-padded). Equivalent to vksplat's tight pack of 45 floats into
    // 12 float4 = 48 floats with 3 of tail padding.
    TEST(ShSwizzleLayout, IndexFormulaMatchesKernel) {
        using namespace lfs::core;
        constexpr std::uint32_t N = 96; // 3 full blocks of 32
        constexpr std::uint32_t K = 15;
        constexpr std::uint32_t FLOATS_PER_PRIM = K * 3u; // 45
        constexpr std::uint32_t SLOTS_PER_PRIM = 12u;     // SH_REST_FLOAT4_PER_PRIMITIVE

        // Build a canonical buffer where each (p, k, c) holds the value p*1000 + k*10 + c.
        std::vector<float> host_canonical(N * K * 3);
        for (std::uint32_t p = 0; p < N; ++p) {
            for (std::uint32_t k = 0; k < K; ++k) {
                for (std::uint32_t c = 0; c < 3; ++c) {
                    host_canonical[p * K * 3 + k * 3 + c] = static_cast<float>(p * 1000 + k * 10 + c);
                }
            }
        }

        Tensor canonical = Tensor::from_vector(host_canonical, {N, K, 3}, Device::CUDA);
        const size_t swizzled_floats = sh_swizzled_float_count(N);
        Tensor swizzled = Tensor::zeros({swizzled_floats}, Device::CUDA);
        reorder_sh_to_swizzled(canonical.ptr<float>(), swizzled.ptr<float>(), N, K);
        cudaDeviceSynchronize();

        auto host = swizzled.to(Device::CPU);
        const float* sw = host.ptr<float>();

        // For each primitive's float4 slot k, the 4 components map to the canonical row at
        // offsets [k*4 .. k*4+3]; out-of-range offsets are zero.
        for (std::uint32_t p : {0u, 1u, 31u, 32u, 64u, 95u}) {
            for (std::uint32_t k = 0; k < SLOTS_PER_PRIM; ++k) {
                const std::uint32_t base_float = sh_swizzled_index(p, k) * 4u;
                for (std::uint32_t i = 0; i < 4; ++i) {
                    const std::uint32_t canonical_off = k * 4u + i;
                    const float expected = canonical_off < FLOATS_PER_PRIM
                                               ? host_canonical[p * FLOATS_PER_PRIM + canonical_off]
                                               : 0.0f;
                    EXPECT_EQ(sw[base_float + i], expected)
                        << "p=" << p << " k=" << k << " i=" << i;
                }
            }
        }
    }

    TEST(ShSwizzleLayout, GatherSelectedRowsToLinearMatchesCanonical) {
        using namespace lfs::core;
        constexpr std::uint32_t N = 70;
        constexpr std::uint32_t K = 15;
        constexpr std::uint32_t FLOATS_PER_PRIM = K * 3u;

        std::vector<float> host_canonical(N * K * 3);
        for (std::uint32_t p = 0; p < N; ++p) {
            for (std::uint32_t k = 0; k < K; ++k) {
                for (std::uint32_t c = 0; c < 3; ++c) {
                    host_canonical[p * K * 3 + k * 3 + c] =
                        static_cast<float>(p * 1000 + k * 10 + c);
                }
            }
        }

        Tensor canonical = Tensor::from_vector(host_canonical, {N, K, 3}, Device::CUDA);
        Tensor swizzled = Tensor::zeros({sh_swizzled_float_count(N)}, Device::CUDA);
        reorder_sh_to_swizzled(canonical.ptr<float>(), swizzled.ptr<float>(), N, K);

        const std::vector<int> selected = {0, 5, 31, 32, 69};
        Tensor indices = Tensor::from_vector(selected, {selected.size()}, Device::CUDA).to(DataType::Int64);
        Tensor gathered = Tensor::empty({selected.size(), K, 3}, Device::CUDA);
        shN_swizzled_gather_to_linear_i64(
            swizzled.ptr<float>(), indices.ptr<std::int64_t>(),
            gathered.ptr<float>(), selected.size(), K);
        cudaDeviceSynchronize();

        auto gathered_cpu = gathered.to(Device::CPU);
        const float* g = gathered_cpu.ptr<float>();
        for (size_t row = 0; row < selected.size(); ++row) {
            const auto p = static_cast<std::uint32_t>(selected[row]);
            for (std::uint32_t off = 0; off < FLOATS_PER_PRIM; ++off) {
                EXPECT_EQ(g[row * FLOATS_PER_PRIM + off],
                          host_canonical[p * FLOATS_PER_PRIM + off])
                    << "row=" << row << " primitive=" << p << " off=" << off;
            }
        }
    }

    // Both lane padding (primitives in the trailing block beyond N) and the tail padding
    // (slot 11's .y/.z/.w per primitive) must be zero after reorder.
    TEST(ShSwizzleLayout, PaddingLanesAreZero) {
        using namespace lfs::core;
        constexpr std::uint32_t N = 70; // last block has 6 padding lanes
        constexpr std::uint32_t K = 15;
        constexpr std::uint32_t SLOTS_PER_PRIM = 12u;

        std::vector<float> host_canonical(N * K * 3, 1.0f); // non-zero source so padding is visibly distinct
        Tensor canonical = Tensor::from_vector(host_canonical, {N, K, 3}, Device::CUDA);
        const size_t swizzled_floats = sh_swizzled_float_count(N);
        Tensor swizzled = Tensor::zeros({swizzled_floats}, Device::CUDA);
        reorder_sh_to_swizzled(canonical.ptr<float>(), swizzled.ptr<float>(), N, K);
        cudaDeviceSynchronize();

        auto host = swizzled.to(Device::CPU);
        const float* sw = host.ptr<float>();

        // (a) Lane padding: primitives in [N, ceil(N/32)*32) must be all-zero across all 12
        // float4 slots.
        const std::uint32_t padded_n = static_cast<std::uint32_t>(sh_swizzled_padded_n(N));
        for (std::uint32_t p = N; p < padded_n; ++p) {
            for (std::uint32_t k = 0; k < SLOTS_PER_PRIM; ++k) {
                const std::uint32_t base = sh_swizzled_index(p, k) * 4u;
                for (std::uint32_t i = 0; i < 4u; ++i) {
                    EXPECT_EQ(sw[base + i], 0.0f) << "lane padding p=" << p << " k=" << k << " i=" << i;
                }
            }
        }

        // (b) Tail padding: slot 11's .y/.z/.w (offsets 1..3) of every valid primitive must be 0.
        for (std::uint32_t p = 0; p < N; ++p) {
            const std::uint32_t base = sh_swizzled_index(p, 11u) * 4u;
            EXPECT_EQ(sw[base + 1u], 0.0f) << "tail .y p=" << p;
            EXPECT_EQ(sw[base + 2u], 0.0f) << "tail .z p=" << p;
            EXPECT_EQ(sw[base + 3u], 0.0f) << "tail .w p=" << p;
        }
    }

    // A model built from canonical SH that is still uploading on a non-blocking
    // stream must swizzle the uploaded values. The upload is held behind a host
    // gate until the constructor has returned, so a swizzle that does not wait
    // for the upload's stream reads the buffer before the copy lands.
    TEST(ShSwizzleLayout, ConstructionWaitsForTheCanonicalUploadStream) {
        using namespace lfs::core;
        constexpr size_t N = 4099;
        constexpr size_t K = 15;
        std::vector<float> coefficients(N * K * 3);
        for (size_t i = 0; i < coefficients.size(); ++i)
            coefficients[i] = 0.25f + static_cast<float>(i % 4093) / 64.0f;
        const auto make_model = [&](const std::vector<float>& values) {
            return std::make_unique<SplatData>(
                3, Tensor::zeros({N, 3}, Device::CUDA), Tensor::zeros({N, 1, 3}, Device::CUDA),
                Tensor::from_vector(values, {N, K, 3}, Device::CUDA),
                Tensor::zeros({N, 3}, Device::CUDA), Tensor::ones({N, 4}, Device::CUDA),
                Tensor::zeros({N, 1}, Device::CUDA), 1.0f);
        };

        cudaStream_t producer{};
        cudaStream_t gate_holder{};
        cudaEvent_t gate{};
        ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&gate_holder, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming), cudaSuccess);
        std::atomic<bool> released{false};
        std::atomic<bool> constructed{false};
        struct OpenGate {
            std::atomic<bool>& flag;
            ~OpenGate() { flag.store(true, std::memory_order_release); }
        } open_on_exit{released};
        std::unique_ptr<SplatData> model;
        {
            CUDAStreamGuard guard(producer);
            // Load the kernels and fill the device and pinned caches first, so
            // the gated construction below has nothing that waits on the device.
            // Other values, so a reused buffer cannot already hold the expected ones.
            make_model(std::vector<float>(coefficients.size(), -1.0f)).reset();
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

            ASSERT_EQ(cudaLaunchHostFunc(
                          gate_holder,
                          [](void* flag) {
                              while (!static_cast<std::atomic<bool>*>(flag)->load(std::memory_order_acquire)) {
                              }
                          },
                          &released),
                      cudaSuccess);
            ASSERT_EQ(cudaEventRecord(gate, gate_holder), cudaSuccess);
            ASSERT_EQ(cudaStreamWaitEvent(producer, gate, 0), cudaSuccess);
            // Opens the gate even if construction blocks on the gated stream.
            std::thread opener([&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                while (!constructed.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
                    std::this_thread::yield();
                released.store(true, std::memory_order_release);
            });
            model = make_model(coefficients);
            const bool gate_held = !released.load(std::memory_order_acquire);
            constructed.store(true, std::memory_order_release);
            opener.join();
            ASSERT_TRUE(gate_held) << "construction waited for the gated stream, so the test cannot observe the order";
        }
        const auto canonical = model->shN_canonical_cpu();
        ASSERT_EQ(canonical.numel(), coefficients.size());
        size_t mismatches = 0;
        for (size_t i = 0; i < coefficients.size(); ++i)
            mismatches += canonical.ptr<float>()[i] != coefficients[i];
        EXPECT_EQ(mismatches, 0u);

        model.reset();
        ASSERT_EQ(cudaStreamSynchronize(producer), cudaSuccess);
        CudaMemoryPool::instance().release_stream(producer);
        EXPECT_EQ(cudaStreamDestroy(producer), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(gate_holder), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(gate), cudaSuccess);
    }

    // Holds `stream` behind a host gate while `work` runs, then opens it. Returns
    // whether the gate was still closed when `work` returned. If `work` itself waits
    // for the held stream, the gate opens after two seconds and the order under test
    // was not observed. Before opening, the gate stays closed for up to a quarter of
    // a second while `settle` still has work, so an access there that does not wait
    // for the held stream completes first.
    template <class Work>
    [[nodiscard]] bool run_while_held(const cudaStream_t stream, Work&& work, const cudaStream_t settle = nullptr) {
        cudaStream_t holder{};
        cudaEvent_t gate{};
        EXPECT_EQ(cudaStreamCreateWithFlags(&holder, cudaStreamNonBlocking), cudaSuccess);
        EXPECT_EQ(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming), cudaSuccess);
        std::atomic<bool> open{false};
        std::atomic<bool> done{false};
        EXPECT_EQ(cudaLaunchHostFunc(
                      holder,
                      [](void* flag) {
                          while (!static_cast<std::atomic<bool>*>(flag)->load(std::memory_order_acquire)) {
                          }
                      },
                      &open),
                  cudaSuccess);
        EXPECT_EQ(cudaEventRecord(gate, holder), cudaSuccess);
        EXPECT_EQ(cudaStreamWaitEvent(stream, gate, 0), cudaSuccess);
        std::thread opener([&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            open.store(true, std::memory_order_release);
        });
        bool held = false;
        {
            struct Join {
                std::atomic<bool>& done;
                std::thread& opener;
                ~Join() {
                    done.store(true, std::memory_order_release);
                    opener.join();
                }
            } join{done, opener};
            work();
            held = !open.load(std::memory_order_acquire);
            const auto settle_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            while (cudaStreamQuery(settle) == cudaErrorNotReady && std::chrono::steady_clock::now() < settle_deadline)
                std::this_thread::yield();
            (void)cudaGetLastError();
        }
        EXPECT_EQ(cudaStreamSynchronize(holder), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(holder), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(gate), cudaSuccess);
        return held;
    }

    struct OwnedStream {
        cudaStream_t stream{};
        OwnedStream() { EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess); }
        ~OwnedStream() {
            EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            lfs::core::CudaMemoryPool::instance().release_stream(stream);
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
        OwnedStream(const OwnedStream&) = delete;
        OwnedStream& operator=(const OwnedStream&) = delete;
    };

    constexpr size_t kOrderRows = 4099;
    constexpr size_t kDegreeOneRest = 3;

    // Every row carries the same SH: canonical value i of a row is base + i / 4.
    [[nodiscard]] float uniform_sh_value(const float base, const size_t i) {
        return base + static_cast<float>(i) * 0.25f;
    }

    [[nodiscard]] std::unique_ptr<lfs::core::SplatData> uniform_sh_model(const float base,
                                                                         const size_t rows = kOrderRows) {
        using namespace lfs::core;
        std::vector<float> values(rows * kDegreeOneRest * 3);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = uniform_sh_value(base, i % (kDegreeOneRest * 3));
        return std::make_unique<SplatData>(
            1, Tensor::zeros({rows, 3}, Device::CUDA), Tensor::zeros({rows, 1, 3}, Device::CUDA),
            Tensor::from_vector(values, {rows, kDegreeOneRest, 3}, Device::CUDA),
            Tensor::zeros({rows, 3}, Device::CUDA), Tensor::ones({rows, 4}, Device::CUDA),
            Tensor::zeros({rows, 1}, Device::CUDA), 1.0f);
    }

    // Values of a canonical [rows, K, 3] tensor whose first three coefficients
    // differ from uniform_sh_model(base). Paths that store SH compactly round the
    // values slightly; values from another base differ by at least 4.
    [[nodiscard]] size_t uniform_sh_mismatches(const lfs::core::Tensor& canonical, const float base) {
        const auto host = canonical.cpu();
        const size_t rows = host.size(0);
        const size_t k = host.size(1);
        size_t mismatches = 0;
        for (size_t row = 0; row < rows; ++row)
            for (size_t i = 0; i < kDegreeOneRest * 3; ++i)
                mismatches += std::abs(host.ptr<float>()[row * k * 3 + i] - uniform_sh_value(base, i)) > 0.01f;
        return mismatches;
    }

    // `work` runs while the SH of `model` is rewritten with `source`'s on a held
    // stream that is not the stream `work` runs on; `work` must read the rewritten SH.
    // Work that waits on the host for the stream it reads on, once that stream waits
    // for the held one, opens the gate on its timer; its read is enqueued before that
    // wait, so the values alone show whether the read was ordered after the write.
    template <class Work>
    void expect_reads_sh_written_on_its_stream(lfs::core::SplatData& model, const lfs::core::SplatData& source,
                                               Work&& work, const bool waits_for_device = false) {
        OwnedStream writer;
        model.shN_raw().set_stream(writer.stream);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        // Work that replaces the SH would free the old buffer, and freeing device
        // storage waits for the whole device; keep it until the stream is released.
        lfs::core::Tensor previous = model.shN_raw();
        const bool held = run_while_held(writer.stream, [&] {
            model.shN_raw().copy_from(source.shN_raw());
            work();
        });
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        previous = {};
        model.shN_raw().set_stream(nullptr);
        if (!waits_for_device)
            ASSERT_TRUE(held) << "the work waited for the held stream, so the order cannot be observed";
    }

    TEST(ShSwizzleLayout, CanonicalReadWaitsForTheStreamThatWroteTheSH) {
        using namespace lfs::core;
        (void)uniform_sh_model(-7.0f)->shN_canonical();
        auto model = uniform_sh_model(-3.0f);
        const auto source = uniform_sh_model(1.0f);
        Tensor canonical;
        expect_reads_sh_written_on_its_stream(*model, *source, [&] { canonical = model->shN_canonical(); });
        EXPECT_EQ(uniform_sh_mismatches(canonical, 1.0f), 0u);
    }

    TEST(ShSwizzleLayout, CanonicalResultIsReadyForReadersOnItsHomeStream) {
        using namespace lfs::core;
        OwnedStream reader;
        const auto model = uniform_sh_model(1.0f);
        Tensor copy;
        {
            CUDAStreamGuard guard(reader.stream);
            // Leaves the reader stream's cached blocks holding other values.
            (void)uniform_sh_model(-3.0f)->shN_canonical().clone();
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_TRUE(run_while_held(nullptr, [&] { copy = model->shN_canonical().clone(); }, reader.stream)) << "the read waited for the held legacy stream, so the order cannot be observed";
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(uniform_sh_mismatches(copy, 1.0f), 0u);
    }

    TEST(ShSwizzleLayout, DegreeChangeReadsTheSHWrittenOnItsStream) {
        uniform_sh_model(-7.0f)->set_max_sh_degree(2);
        auto model = uniform_sh_model(-3.0f);
        const auto source = uniform_sh_model(1.0f);
        expect_reads_sh_written_on_its_stream(*model, *source, [&] { model->set_max_sh_degree(2); });
        EXPECT_EQ(uniform_sh_mismatches(model->shN_canonical(), 1.0f), 0u);
    }

    TEST(ShSwizzleLayout, SwizzledConstructionWaitsForTheStreamThatWroteTheSH) {
        using namespace lfs::core;
        const auto source = uniform_sh_model(1.0f);
        const auto build = [](Tensor shN) {
            return std::make_unique<SplatData>(
                2, Tensor::zeros({kOrderRows, 3}, Device::CUDA), Tensor::zeros({kOrderRows, 1, 3}, Device::CUDA),
                std::move(shN), Tensor::zeros({kOrderRows, 3}, Device::CUDA),
                Tensor::ones({kOrderRows, 4}, Device::CUDA), Tensor::zeros({kOrderRows, 1}, Device::CUDA), 1.0f,
                SplatData::ShNLayout::Swizzled);
        };
        (void)build(uniform_sh_model(-7.0f)->shN_raw().clone());
        OwnedStream writer;
        Tensor staged;
        {
            CUDAStreamGuard guard(writer.stream);
            staged = Tensor::zeros({static_cast<size_t>(source->shN_raw().numel())}, Device::CUDA);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::unique_ptr<SplatData> model;
        ASSERT_TRUE(run_while_held(writer.stream, [&] {
            staged.copy_from(source->shN_raw());
            model = build(std::move(staged));
        })) << "construction waited for the held stream, so the order cannot be observed";
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(uniform_sh_mismatches(model->shN_canonical(), 1.0f), 0u);
    }

    TEST(ShSwizzleLayout, CapacityGrowthCopiesTheSHWrittenOnItsStream) {
        uniform_sh_model(-7.0f)->reserve_capacity(2 * kOrderRows);
        auto model = uniform_sh_model(-3.0f);
        const auto source = uniform_sh_model(1.0f);
        expect_reads_sh_written_on_its_stream(
            *model, *source, [&] { model->reserve_capacity(2 * kOrderRows); }, true);
        EXPECT_EQ(uniform_sh_mismatches(model->shN_canonical(), 1.0f), 0u);
    }

    TEST(ShSwizzleLayout, RandomChoiceGathersTheSHWrittenOnItsStream) {
        using namespace lfs::core;
        random_choose(*uniform_sh_model(-7.0f), static_cast<int>(kOrderRows / 2), 7);
        auto model = uniform_sh_model(-3.0f);
        const auto source = uniform_sh_model(1.0f);
        expect_reads_sh_written_on_its_stream(
            *model, *source, [&] { random_choose(*model, static_cast<int>(kOrderRows / 2), 7); }, true);
        EXPECT_EQ(model->size(), kOrderRows / 2);
        EXPECT_EQ(uniform_sh_mismatches(model->shN_canonical(), 1.0f), 0u);
    }

    TEST(ShSwizzleLayout, SHPaletteReadsTheSHWrittenOnItsStream) {
        using namespace lfs::core;
        constexpr size_t rows = 64;
        const auto source = uniform_sh_model(1.0f, rows);
        const auto stale = uniform_sh_model(-3.0f, rows);
        (void)lfs::io::emeans_sh_swizzled(stale->shN_raw(), static_cast<int>(rows), 3, static_cast<int>(rows));
        OwnedStream writer;
        Tensor swizzled;
        {
            CUDAStreamGuard guard(writer.stream);
            swizzled = stale->shN_raw().clone();
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        // With as many palette entries as rows, each centroid is its row's SH.
        // The search waits for the whole device, so the gate opens on its timer
        // and only the values show whether the rows were read after the write.
        Tensor centroids;
        (void)run_while_held(writer.stream, [&] {
            swizzled.copy_from(source->shN_raw());
            centroids = std::get<0>(lfs::io::emeans_sh_swizzled(swizzled, static_cast<int>(rows), 3,
                                                                static_cast<int>(rows)));
        });
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_TRUE(centroids.is_valid());
        const auto host = centroids.cpu();
        size_t stale_values = 0;
        for (size_t i = 0; i < host.numel(); ++i)
            stale_values += host.ptr<float>()[i] < 1.0f;
        EXPECT_EQ(stale_values, 0u);
    }

    TEST(ShSwizzleLayout, HalfPrecisionGrowthKeepsTheCopiedSHAfterItsZeroFill) {
        using namespace lfs::core;
        const auto half_model = [] {
            auto half = uniform_sh_model(1.0f)->shN_raw().to(DataType::Float16);
            return std::make_unique<SplatData>(
                1, Tensor::zeros({kOrderRows, 3}, Device::CUDA), Tensor::zeros({kOrderRows, 1, 3}, Device::CUDA),
                std::move(half), Tensor::zeros({kOrderRows, 3}, Device::CUDA),
                Tensor::ones({kOrderRows, 4}, Device::CUDA), Tensor::zeros({kOrderRows, 1}, Device::CUDA), 1.0f,
                SplatData::ShNLayout::Swizzled);
        };
        {
            auto warm = half_model();
            warm->means().reserve(2 * kOrderRows);
            warm->set_max_sh_degree(1);
        }
        auto model = half_model();
        ASSERT_TRUE(model->shN_ieee_f16());
        const auto expected = model->shN_canonical_cpu();
        OwnedStream home;
        model->shN_raw().set_stream(home.stream);
        model->means().reserve(2 * kOrderRows);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        // The grown buffer is zeroed on the legacy stream; holding it keeps that fill pending.
        ASSERT_TRUE(run_while_held(nullptr, [&] { model->set_max_sh_degree(1); }, home.stream)) << "the growth waited for the held legacy stream, so the order cannot be observed";
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        model->shN_raw().set_stream(nullptr);
        const auto actual = model->shN_canonical_cpu();
        ASSERT_EQ(actual.numel(), expected.numel());
        EXPECT_EQ(std::memcmp(actual.ptr<float>(), expected.ptr<float>(), expected.bytes()), 0);
    }

} // namespace

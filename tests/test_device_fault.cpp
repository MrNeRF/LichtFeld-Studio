/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Phase 6C-P3 — device-fault reference kernel (index_select) + ValidatedIndexToken
// + ABI / graph-capture / first-fault / device-trap coverage.
// Spec: .codex_tmp/phase-6c-device-fault-spec.md §1.10, §3, §9.

#include "core/device_fault.hpp"
#include "core/error.hpp"
#include "core/tensor.hpp"
#include "device_fault_cuda_utils.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

    [[nodiscard]] bool cuda_device_available() {
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        return device >= 0;
    }

    [[nodiscard]] const lfs::SmallFields::Entry* find_field(
        const lfs::Error& error, const std::string_view key) {
        if (error.frames().empty()) {
            return nullptr;
        }
        for (const auto& entry : error.frames().front().fields.entries()) {
            if (entry.key == key) {
                return &entry;
            }
        }
        return nullptr;
    }

    class DeviceFaultTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::core::reset_cuda_diagnostics_for_testing();
            if (!cuda_device_available()) {
                GTEST_SKIP() << "a live CUDA device is required";
            }
        }

        void TearDown() override {
            (void)cudaGetLastError();
            lfs::core::reset_cuda_diagnostics_for_testing();
        }
    };

    class DeviceFaultDeathTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // threadsafe re-execs the binary so the child initializes CUDA from
            // scratch (fork children cannot init CUDA — see test_tensor_launch_check).
            saved_death_test_style_ = GTEST_FLAG_GET(death_test_style);
            GTEST_FLAG_SET(death_test_style, "threadsafe");
            lfs::core::reset_cuda_diagnostics_for_testing();
        }

        void TearDown() override {
            GTEST_FLAG_SET(death_test_style, saved_death_test_style_);
            (void)cudaGetLastError();
            lfs::core::reset_cuda_diagnostics_for_testing();
        }

    private:
        std::string saved_death_test_style_;
    };

    void enable_device_trap_for_subprocess() {
#if defined(_WIN32)
        (void)_putenv_s("LFS_CUDA_SYNC_DEBUG", "device-trap");
#else
        (void)setenv("LFS_CUDA_SYNC_DEBUG", "device-trap", 1);
#endif
    }

    // ---------------------------------------------------------------------------
    // ABI static_assert compile coverage (host side; nvcc side is in
    // device_fault_cuda_utils.cu + header static_asserts).
    // ---------------------------------------------------------------------------
    TEST(DeviceFaultAbi, HostLayoutMatchesFrozenContract) {
        using Rec = lfs::core::DeviceFaultRecord;
        static_assert(std::is_trivially_copyable_v<Rec>);
        static_assert(std::is_standard_layout_v<Rec>);
        static_assert(sizeof(Rec) == 32);
        static_assert(alignof(Rec) == 8);
        static_assert(offsetof(Rec, code) == 0);
        static_assert(offsetof(Rec, op_id) == 4);
        static_assert(offsetof(Rec, value) == 8);
        static_assert(offsetof(Rec, bound) == 16);
        static_assert(offsetof(Rec, thread_id) == 24);
        EXPECT_EQ(sizeof(Rec), 32u);
        EXPECT_EQ(static_cast<std::uint32_t>(lfs::core::DeviceFaultCode::NoFault), 0u);
        EXPECT_EQ(static_cast<std::uint32_t>(lfs::core::DeviceFaultCode::IndexOutOfBounds), 1u);
    }

    // ---------------------------------------------------------------------------
    // ValidatedIndexToken (§1.10): unforgeable, match/mismatch → checked path.
    // ---------------------------------------------------------------------------
    TEST(ValidatedIndexToken, IssuerProducesMatchingToken) {
        const void* storage = reinterpret_cast<const void*>(0x1000);
        constexpr std::uint64_t version = 7;
        constexpr int device = 0;
        constexpr std::uint64_t producer = 42;

        auto token = lfs::core::issue_validated_index_token(storage, version, device, producer);
        EXPECT_TRUE(token.matches(storage, version, device, producer));
        EXPECT_EQ(token.storage_identity(), storage);
        EXPECT_EQ(token.mutation_version(), version);
        EXPECT_EQ(token.device_ordinal(), device);
        EXPECT_EQ(token.producer_event_or_range(), producer);
    }

    TEST(ValidatedIndexToken, StaleVersionSelectsCheckedPath) {
        const void* storage = reinterpret_cast<const void*>(0x2000);
        auto token = lfs::core::issue_validated_index_token(storage, /*version=*/1, 0, 0);
        // Stale version → matches() false → launcher must take checked path (not assert).
        EXPECT_FALSE(token.matches(storage, /*version=*/2, 0, 0));
    }

    TEST(ValidatedIndexToken, WrongDeviceSelectsCheckedPath) {
        const void* storage = reinterpret_cast<const void*>(0x3000);
        auto token = lfs::core::issue_validated_index_token(storage, 1, /*device=*/0, 0);
        EXPECT_FALSE(token.matches(storage, 1, /*device=*/1, 0));
    }

    TEST(ValidatedIndexToken, WrongStorageSelectsCheckedPath) {
        auto token = lfs::core::issue_validated_index_token(
            reinterpret_cast<const void*>(0x4000), 1, 0, 0);
        EXPECT_FALSE(token.matches(reinterpret_cast<const void*>(0x4001), 1, 0, 0));
    }

    TEST(ValidatedIndexToken, MoveOnlyNotCopyable) {
        static_assert(!std::is_copy_constructible_v<lfs::core::ValidatedIndexToken>);
        static_assert(!std::is_copy_assignable_v<lfs::core::ValidatedIndexToken>);
        static_assert(std::is_move_constructible_v<lfs::core::ValidatedIndexToken>);
        static_assert(std::is_move_assignable_v<lfs::core::ValidatedIndexToken>);

        auto token = lfs::core::issue_validated_index_token(
            reinterpret_cast<const void*>(0x5000), 3, 0, 9);
        auto moved = std::move(token);
        EXPECT_TRUE(moved.matches(reinterpret_cast<const void*>(0x5000), 3, 0, 9));
    }

    // Test-only unchecked stub: accepts a live matching token and skips the
    // device-fault protocol. Mismatch forces the checked path flag.
    [[nodiscard]] bool unchecked_index_path_selected(
        const lfs::core::ValidatedIndexToken& token,
        const void* storage,
        const std::uint64_t version,
        const int device,
        const std::uint64_t producer) {
        return token.matches(storage, version, device, producer);
    }

    TEST(ValidatedIndexToken, MatchingTokenAcceptedByUncheckedStub) {
        const void* storage = reinterpret_cast<const void*>(0x6000);
        auto token = lfs::core::issue_validated_index_token(storage, 1, 0, 1);
        EXPECT_TRUE(unchecked_index_path_selected(token, storage, 1, 0, 1));
    }

    TEST(ValidatedIndexToken, MismatchedTokenRejectedByUncheckedStub) {
        const void* storage = reinterpret_cast<const void*>(0x7000);
        auto token = lfs::core::issue_validated_index_token(storage, 1, 0, 1);
        EXPECT_FALSE(unchecked_index_path_selected(token, storage, 99, 0, 1));
    }

    // ---------------------------------------------------------------------------
    // index_select fault injection + success path
    // ---------------------------------------------------------------------------
    TEST_F(DeviceFaultTest, IndexSelectOobThrowsBoundsViolationWithFields) {
        using namespace lfs::core;
        // Input [0,1,2,3,4]; single OOB index 99.
        auto input = Tensor::from_vector(std::vector<float>{0.f, 1.f, 2.f, 3.f, 4.f},
                                         {5}, Device::CUDA);
        auto indices = Tensor::from_vector(std::vector<int>{99}, {1}, Device::CUDA);

        // Assert mode drains inline (replaces the old synchronous D2H scan):
        // the op itself throws BoundsViolation; the slot is consumed by the op.
        try {
            auto out = input.index_select(0, indices, BoundaryMode::Assert);
            FAIL() << "expected BoundsViolation from Assert-mode index_select";
        } catch (const lfs::Exception& ex) {
            const lfs::Error& error = ex.error();
            EXPECT_EQ(error.code(), lfs::ErrorCode::BoundsViolation);
            EXPECT_EQ(error.domain(), lfs::ErrorDomain::CUDA);
            ASSERT_NE(find_field(error, "value"), nullptr);
            EXPECT_EQ(std::get<std::int64_t>(find_field(error, "value")->value), 99);
            ASSERT_NE(find_field(error, "bound"), nullptr);
            EXPECT_EQ(std::get<std::int64_t>(find_field(error, "bound")->value), 5);
            ASSERT_NE(find_field(error, "op_id"), nullptr);
            ASSERT_NE(find_field(error, "thread_id"), nullptr);
        }
    }

    TEST_F(DeviceFaultTest, IndexSelectValidIndicesNoThrowRegression) {
        using namespace lfs::core;
        auto input = Tensor::from_vector(std::vector<float>{10.f, 20.f, 30.f, 40.f},
                                         {4}, Device::CUDA);
        auto indices = Tensor::from_vector(std::vector<int>{0, 2, 3}, {3}, Device::CUDA);

        Tensor out;
        EXPECT_NO_THROW({
            out = input.index_select(0, indices, BoundaryMode::Assert);
        });

        // Consumer drain: harvest must be clean after a valid launch.
        EXPECT_NO_THROW({
            device_fault_await_and_consume_or_throw(
                input.stream(), "tensor.masking.index_select", LFS_SOURCE_SITE_CURRENT());
        });

        auto cpu = out.cpu();
        ASSERT_EQ(cpu.numel(), 3u);
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 10.f);
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[1], 30.f);
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[2], 40.f);
    }

    // ---------------------------------------------------------------------------
    // First-fault-wins under many concurrent OOB recorders
    // ---------------------------------------------------------------------------
    TEST_F(DeviceFaultTest, FirstFaultWinsUnderManyOobThreads) {
        using namespace lfs::core;

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        DeviceFaultRecord* device_record = nullptr;
        ASSERT_EQ(device_fault_slot_acquire(stream, &device_record), cudaSuccess);
        ASSERT_NE(device_record, nullptr);
        ASSERT_EQ(device_fault_slot_enqueue_reset(stream), cudaSuccess);

        constexpr std::uint32_t op_id = 0xA11u;
        constexpr std::int64_t value = -7;
        constexpr std::int64_t bound = 4;
        // 32 blocks * 128 threads = 4096 concurrent OOB recorders.
        ASSERT_EQ(device_fault_test::launch_many_oob_record(
                      device_record, op_id, value, bound,
                      /*trap_after_record=*/false,
                      /*blocks=*/32, /*threads_per_block=*/128, stream),
                  cudaSuccess);
        ASSERT_EQ(device_fault_slot_enqueue_harvest(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const DeviceFaultRecord record = device_fault_slot_consume(stream);
        EXPECT_EQ(record.code,
                  static_cast<std::uint32_t>(DeviceFaultCode::IndexOutOfBounds));
        EXPECT_EQ(record.op_id, op_id);
        EXPECT_EQ(record.value, value);
        EXPECT_EQ(record.bound, bound);
        // Exactly one winner: value/bound/op_id are stable for the first CAS.
        // thread_id may legitimately be 0 (block 0 / thread 0 winner).

        // Second consume without a new record remains the staging contents until
        // reset; reset + harvest of a clean kernel yields NoFault.
        ASSERT_EQ(device_fault_slot_enqueue_reset(stream), cudaSuccess);
        ASSERT_EQ(device_fault_slot_enqueue_harvest(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const DeviceFaultRecord clean = device_fault_slot_consume(stream);
        EXPECT_EQ(clean.code, static_cast<std::uint32_t>(DeviceFaultCode::NoFault));

        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    // ---------------------------------------------------------------------------
    // Graph capture → Unsupported (cudaStreamBeginCapture; no GPU replay)
    // ---------------------------------------------------------------------------
    TEST_F(DeviceFaultTest, GraphCaptureYieldsUnsupported) {
        using namespace lfs::core;

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        // Allocate tensors BEFORE capture begins (allocations must not join the graph).
        auto input = Tensor::from_vector(std::vector<float>{1.f, 2.f, 3.f},
                                         {3}, Device::CUDA);
        auto indices = Tensor::from_vector(std::vector<int>{0}, {1}, Device::CUDA);
        input.set_stream(stream);
        indices.set_stream(stream);
        // Materialize a result tensor on the same stream for index_select_into.
        auto out = Tensor::zeros({1}, Device::CUDA, DataType::Float32);
        out.set_stream(stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        cudaGraph_t graph = nullptr;
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);

        bool threw = false;
        try {
            // Host entry of checked index_select rejects capture with Unsupported.
            // No GPU graph replay is required (spec §1.9 unit-test contract).
            input.index_select_into(out, 0, indices, BoundaryMode::Assert);
        } catch (const lfs::Exception& ex) {
            threw = true;
            EXPECT_EQ(ex.error().code(), lfs::ErrorCode::Unsupported);
            EXPECT_EQ(ex.error().domain(), lfs::ErrorDomain::CUDA);
        } catch (...) {
            threw = true;
            ADD_FAILURE() << "expected lfs::Exception(Unsupported), got other exception";
        }

        // End capture cleanly whether or not the op threw (capture may be empty).
        const cudaError_t end_status = cudaStreamEndCapture(stream, &graph);
        if (end_status == cudaSuccess && graph != nullptr) {
            (void)cudaGraphDestroy(graph);
        } else {
            (void)cudaGetLastError();
        }
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

        EXPECT_TRUE(threw) << "checked index_select under capture must throw Unsupported";
    }

    // ---------------------------------------------------------------------------
    // Device-trap subprocess (records then aborts on harvest when DeviceTrap on)
    // ---------------------------------------------------------------------------
    [[noreturn]] void device_trap_oob_subprocess() {
        // Diagnostics must be enabled before any mode cache is populated.
        enable_device_trap_for_subprocess();
        lfs::core::reset_cuda_diagnostics_for_testing();

        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) {
            (void)cudaGetLastError();
            std::_Exit(4);
        }

        try {
            using namespace lfs::core;
            auto input = Tensor::from_vector(std::vector<float>{0.f, 1.f, 2.f},
                                             {3}, Device::CUDA);
            auto indices = Tensor::from_vector(std::vector<int>{99}, {1}, Device::CUDA);
            (void)input.index_select(0, indices, BoundaryMode::Assert);
            // Harvest is the preferred host trap site (§1.8): records then abort.
            device_fault_await_and_consume_or_throw(
                input.stream(), "tensor.masking.index_select", LFS_SOURCE_SITE_CURRENT());
            // If trap is off or harvest is clean, fail the death test contract.
            std::_Exit(2);
        } catch (const lfs::Exception&) {
            // BoundsViolation without abort means DeviceTrap did not fire.
            std::_Exit(3);
        } catch (...) {
            std::_Exit(5);
        }
    }

    TEST_F(DeviceFaultDeathTest, DeviceTrapRecordsThenAbortsOnHarvest) {
        // EXPECT_DEATH / EXPECT_EXIT: abort from throw_device_fault_error under
        // DeviceTrap. Exit code is platform-dependent for abort; accept any death.
        EXPECT_DEATH(device_trap_oob_subprocess(), "");
    }

    // ---------------------------------------------------------------------------
    // Positive/negative nvcc TU coverage (Ruling 1)
    // Positive: device_fault_cuda_utils.cu is compiled into this binary and
    // includes the FOUR allowed headers (device_fault, cuda_error, error_codes,
    // source_site) while exercising device_fault_try_record_first.
    // Negative: fixtures under tests/cuda_fixtures/ are NOT build sources; they
    // deliberately #include forbidden host-only surfaces. Assert presence +
    // content so the ratchet cannot be silently deleted.
    // ---------------------------------------------------------------------------
    TEST(DeviceFaultNvccFixtures, PositiveTuIsLinkedViaManyOobLauncher) {
        // If device_fault_cuda_utils.cu failed to compile under nvcc (e.g. because
        // a forbidden header leaked into device_fault.hpp's CUDA-safe path), the
        // link of launch_many_oob_record would fail. Touching the symbol here
        // proves the positive TU is part of this binary.
        EXPECT_NE(&device_fault_test::launch_many_oob_record, nullptr);
    }

    TEST(DeviceFaultNvccFixtures, NegativeFixturesExistAndIncludeForbiddenHeaders) {
#ifndef PROJECT_ROOT_PATH
        GTEST_SKIP() << "PROJECT_ROOT_PATH not defined";
#else
        const std::string root = PROJECT_ROOT_PATH;
        const std::string error_hpp_fixture =
            root + "/tests/cuda_fixtures/device_fault_negative_error_hpp.cu";
        const std::string result_fixture =
            root + "/tests/cuda_fixtures/device_fault_negative_result.cu";

        auto read_all = [](const std::string& path) -> std::string {
            std::ifstream in(path);
            EXPECT_TRUE(in.good()) << "missing negative fixture: " << path;
            std::ostringstream ss;
            ss << in.rdbuf();
            return ss.str();
        };

        const std::string error_hpp_src = read_all(error_hpp_fixture);
        EXPECT_NE(error_hpp_src.find("core/error.hpp"), std::string::npos);
        EXPECT_NE(error_hpp_src.find("MUST fail"), std::string::npos);

        const std::string result_src = read_all(result_fixture);
        EXPECT_NE(result_src.find("std::expected"), std::string::npos);
        EXPECT_NE(result_src.find("core/error.hpp"), std::string::npos);

        // Guard: negative fixtures must never appear in the TEST_FILES / SOURCES
        // of lichtfeld_tests. We cannot re-read CMakeLists from the binary
        // reliably after install, so the presence of the "NOT be added to any
        // production or test target" comment is the local ratchet.
        EXPECT_NE(error_hpp_src.find("NOT be added to any"), std::string::npos);
#endif
    }

} // namespace

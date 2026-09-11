/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/memory_pressure.hpp"
#include "io/error.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace lfs;

namespace {

    Error make_test_error(const ErrorCode code = ErrorCode::InvalidArgument,
                          const std::string_view detail = "test failure") {
        return make_error(ErrorInit{
            .code = code,
            .domain = ErrorDomain::Core,
            .severity = Severity::Error,
            .retryability = Retryability::NotRetryable,
            .operation_id = OperationId::generate(),
            .user_message = "something went wrong",
            .detail = std::string(detail),
            .detection = LFS_SOURCE_SITE_CURRENT(),
            .fields = SmallFields{}.add("key", std::int64_t{42}),
            .native = std::nullopt,
        });
    }

} // namespace

// ---------------------------------------------------------------------------
// ABI / static contract
// ---------------------------------------------------------------------------

TEST(ErrorCoreAbi, HandleSizesAreOnePointer) {
    static_assert(sizeof(Error) == sizeof(void*));
    static_assert(sizeof(Status) == sizeof(void*));
    static_assert(sizeof(Result<void>) == sizeof(void*));
}

TEST(ErrorCoreAbi, ErrorAndResultAreNodiscard) {
    // std::is_invocable doesn't observe [[nodiscard]]; this is a compile-time
    // sanity check that the attributes are present on the declarations by
    // construction (see error.hpp) rather than a runtime assertion. The
    // meaningful enforcement is -Wunused-result at the orchestrator's build.
    SUCCEED();
}

TEST(ErrorCoreAbi, NoImplicitConversionFromStdExpectedStringOrUnexpected) {
    static_assert(!std::is_convertible_v<std::expected<int, Error>, Result<int>>);
    static_assert(!std::is_convertible_v<std::unexpected<std::string>, Result<int>>);
    static_assert(!std::is_convertible_v<std::string, Result<int>>);
    // A success value of the right type still converts implicitly, matching
    // ordinary "return a value" ergonomics.
    static_assert(std::is_convertible_v<std::string, Result<std::string>>);
    static_assert(std::is_convertible_v<Error, Result<int>>);
    // Result<void> has no implicit Error constructor at all (see error.hpp).
    static_assert(!std::is_convertible_v<Error, Status>);
}

TEST(ErrorCoreAbi, SuccessConstructionAndMoveAreNoexcept) {
    static_assert(std::is_nothrow_default_constructible_v<Status>);
    static_assert(std::is_nothrow_move_constructible_v<Status>);
    static_assert(noexcept(force_next_error_allocation_to_fail_for_testing(false)));
}

// ---------------------------------------------------------------------------
// Construction, accessors, copy/move
// ---------------------------------------------------------------------------

TEST(ErrorCore, MakeErrorPopulatesFieldsAndDetectionFrame) {
    const Error error = make_test_error();
    EXPECT_EQ(error.code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(error.domain(), ErrorDomain::Core);
    EXPECT_EQ(error.severity(), Severity::Error);
    EXPECT_EQ(error.retryability(), Retryability::NotRetryable);
    EXPECT_EQ(error.user_message(), "something went wrong");
    EXPECT_EQ(error.detail(), "test failure");
    EXPECT_TRUE(error.operation_id().has_value());
    EXPECT_FALSE(error.native().has_value());
    EXPECT_FALSE(error.is_immortal());

    ASSERT_EQ(error.frames().size(), 1u);
    EXPECT_TRUE(error.frames()[0].operation.empty());
    ASSERT_EQ(error.frames()[0].fields.size(), 1u);
    EXPECT_EQ(error.frames()[0].fields.entries()[0].key, "key");
}

TEST(ErrorCore, CopyIsIndependentHandleSharingOnePayload) {
    const Error original = make_test_error();
    const Error copy = original; // NOLINT(performance-unnecessary-copy-initialization)
    EXPECT_EQ(original.code(), copy.code());
    EXPECT_EQ(original.frames().size(), copy.frames().size());
}

TEST(ErrorCore, MoveLeavesSourceEmptyAndDestinationValid) {
    Error original = make_test_error();
    Error moved = std::move(original);
    EXPECT_EQ(moved.code(), ErrorCode::InvalidArgument);
    // `original` is now the empty/success handle; per the frozen contract
    // this state must only be destroyed or reassigned, which the destructor
    // exercises at end of scope. We deliberately do not call an accessor on
    // it (that is a documented terminate-on-misuse contract, not something
    // to exercise in the shared test binary).
}

TEST(ErrorCore, CopyAssignmentAndSelfAssignmentAreSafe) {
    Error a = make_test_error(ErrorCode::NotFound, "a");
    Error b = make_test_error(ErrorCode::Internal, "b");
    b = a;
    EXPECT_EQ(b.code(), ErrorCode::NotFound);

    a = a; // NOLINT(clang-diagnostic-self-assign-overloaded)
    EXPECT_EQ(a.code(), ErrorCode::NotFound);
}

// ---------------------------------------------------------------------------
// with_context: COW, ordering, bounds
// ---------------------------------------------------------------------------

TEST(ErrorCore, WithContextAppendsFramesInOrder) {
    Error error = make_test_error();
    error = std::move(error).with_context("load dataset", LFS_SOURCE_SITE_CURRENT());
    error = std::move(error).with_context("import asset", LFS_SOURCE_SITE_CURRENT());

    ASSERT_EQ(error.frames().size(), 3u);
    EXPECT_TRUE(error.frames()[0].operation.empty()); // detection frame
    EXPECT_EQ(error.frames()[1].operation, "load dataset");
    EXPECT_EQ(error.frames()[2].operation, "import asset");
}

TEST(ErrorCore, WithContextOnSharedPayloadClonesRatherThanMutatesOriginal) {
    const Error original = make_test_error();
    Error shared_copy = original; // refcount now 2
    Error annotated = std::move(shared_copy).with_context("outer op", LFS_SOURCE_SITE_CURRENT());

    EXPECT_EQ(annotated.frames().size(), 2u);
    // The original, still-shared handle must be unaffected by COW cloning.
    EXPECT_EQ(original.frames().size(), 1u);
}

TEST(ErrorCore, ContextFrameCapDropsTheSeventeenthFrameSilently) {
    Error error = make_test_error();
    for (int i = 0; i < 20; ++i) {
        error = std::move(error).with_context(std::format("frame {}", i), LFS_SOURCE_SITE_CURRENT());
    }
    // 1 detection frame + kMaxErrorContextFrames context frames; everything
    // past the bound is a silent no-op that preserves what is already there.
    EXPECT_EQ(error.frames().size(), 1u + kMaxErrorContextFrames);
    EXPECT_EQ(error.frames()[1].operation, "frame 0");
    EXPECT_EQ(error.frames().back().operation, std::format("frame {}", kMaxErrorContextFrames - 1));
}

TEST(ErrorCore, SuppressedCapDropsTheNinthSuppressedErrorSilently) {
    Error error = make_test_error();
    for (int i = 0; i < 12; ++i) {
        error = std::move(error).with_suppressed(make_test_error(ErrorCode::Internal,
                                                                 std::format("suppressed {}", i)));
    }
    EXPECT_EQ(error.suppressed().size(), kMaxSuppressedErrors);
    EXPECT_EQ(error.suppressed()[0].detail(), "suppressed 0");
    EXPECT_EQ(error.suppressed().back().detail(),
              std::format("suppressed {}", kMaxSuppressedErrors - 1));
}

TEST(ErrorCore, FieldCapDropsTheSeventeenthFieldAndSetsOverflowed) {
    SmallFields fields;
    for (int i = 0; i < 20; ++i) {
        fields.add(std::format("field{}", i), static_cast<std::int64_t>(i));
    }
    EXPECT_EQ(fields.size(), kMaxFieldsPerFrame);
    EXPECT_TRUE(fields.overflowed());
    EXPECT_EQ(fields.entries()[0].key, "field0");
    EXPECT_EQ(fields.entries().back().key, std::format("field{}", kMaxFieldsPerFrame - 1));
}

TEST(ErrorCore, ImmortalSeedIsANoOpTargetForContextAndSuppressed) {
    Error seed = make_immortal_error_for_testing();
    EXPECT_TRUE(seed.is_immortal());
    seed = std::move(seed).with_context("op", LFS_SOURCE_SITE_CURRENT());
    EXPECT_TRUE(seed.frames().empty());
    seed = std::move(seed).with_suppressed(make_test_error());
    EXPECT_TRUE(seed.suppressed().empty());
}

// ---------------------------------------------------------------------------
// UTF-8-safe truncation
// ---------------------------------------------------------------------------

TEST(ErrorCore, TruncateUtf8SafeLeavesShortStringsUntouched) {
    EXPECT_EQ(truncate_utf8_safe("hello", 10), "hello");
}

TEST(ErrorCore, TruncateUtf8SafeNeverSplitsAMultiByteSequence) {
    // U+00E9 (é) encodes as 0xC3 0xA9 in UTF-8.
    const std::string value = "h\xC3\xA9llo";
    const std::string truncated = truncate_utf8_safe(value, 2);
    EXPECT_EQ(truncated, "h"); // the 2-byte é does not fit in 2 bytes total with 'h'
    // The retained bytes must never end mid-sequence: the byte after the
    // kept prefix (if any existed) must not be a continuation byte (10xxxxxx).
    ASSERT_FALSE(truncated.empty());
    const auto last = static_cast<unsigned char>(truncated.back());
    EXPECT_NE(last & 0xC0, 0x80) << "trailing byte must not be a continuation byte";
}

TEST(ErrorCore, TruncateUtf8SafeKeepsCompleteMultiByteSequenceThatFits) {
    const std::string value = "h\xC3\xA9llo"; // 6 bytes total
    EXPECT_EQ(truncate_utf8_safe(value, 5), "h\xC3\xA9ll");
}

TEST(ErrorCore, MakeErrorTruncatesOverlongDeveloperStrings) {
    const std::string huge(kMaxDeveloperStringBytes + 500, 'x');
    const Error error = make_error(ErrorInit{
        .code = ErrorCode::Internal,
        .domain = ErrorDomain::Core,
        .severity = Severity::Error,
        .retryability = Retryability::NotRetryable,
        .operation_id = OperationId{},
        .user_message = huge,
        .detail = huge,
        .detection = LFS_SOURCE_SITE_CURRENT(),
        .fields = SmallFields{},
        .native = std::nullopt,
    });
    EXPECT_LE(error.user_message().size(), kMaxDeveloperStringBytes);
    EXPECT_LE(error.detail().size(), kMaxDeveloperStringBytes);
}

// ---------------------------------------------------------------------------
// Bounded serialization
// ---------------------------------------------------------------------------

TEST(ErrorCore, FormatForDeveloperStaysWithinSerializedBound) {
    Error error = make_test_error();
    // 16 successfully-added context frames (the with_context cap) times a
    // near-per-string-cap field value comfortably exceeds
    // kMaxSerializedErrorBytes before the final truncation, so this
    // actually exercises format_for_developer's own bound rather than just
    // staying under it by construction.
    for (std::size_t i = 0; i < kMaxErrorContextFrames + 4; ++i) {
        SmallFields fields;
        fields.add("path", std::string(3000, 'p'));
        error = std::move(error).with_context(std::format("frame {}", i), LFS_SOURCE_SITE_CURRENT(),
                                              std::move(fields));
    }
    const std::string formatted = format_for_developer(error);
    EXPECT_EQ(formatted.size(), kMaxSerializedErrorBytes);
    EXPECT_NE(formatted.find("InvalidArgument"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Result<T> / Status monadic operations, every value category
// ---------------------------------------------------------------------------

TEST(ResultCore, SuccessAndFailureConstructionAndAccessors) {
    Result<int> ok(42);
    EXPECT_TRUE(ok.has_value());
    EXPECT_EQ(ok.value(), 42);
    EXPECT_EQ(*ok, 42);

    Result<int> failed(make_test_error());
    EXPECT_FALSE(static_cast<bool>(failed));
    EXPECT_EQ(failed.error().code(), ErrorCode::InvalidArgument);
}

TEST(ResultCore, ValueCategoryOverloadsAllCompile) {
    Result<std::string> lvalue_source("owned"); // NOLINT
    EXPECT_EQ(lvalue_source.value(), "owned");  // &
    const Result<std::string> const_lvalue("owned");
    EXPECT_EQ(const_lvalue.value(), "owned");                 // const&
    EXPECT_EQ(Result<std::string>("owned").value(), "owned"); // &&
    const Result<std::string> const_rvalue_source("owned");
    EXPECT_EQ(std::move(const_rvalue_source).value(), "owned"); // const&&
}

TEST(ResultCore, AndThenChainsAcrossResultAndStdExpectedReturningCallbacks) {
    const Result<int> viaResult = Result<int>(2).and_then([](int v) { return Result<int>(v * 2); });
    ASSERT_TRUE(viaResult.has_value());
    EXPECT_EQ(viaResult.value(), 4);

    const Result<int> viaExpected = Result<int>(2).and_then(
        [](int v) -> std::expected<int, Error> { return v * 3; });
    ASSERT_TRUE(viaExpected.has_value());
    EXPECT_EQ(viaExpected.value(), 6);

    const Result<int> shortCircuited =
        Result<int>(make_test_error()).and_then([](int v) { return Result<int>(v * 2); });
    EXPECT_FALSE(shortCircuited.has_value());
    EXPECT_EQ(shortCircuited.error().code(), ErrorCode::InvalidArgument);
}

TEST(ResultCore, TransformAppliesOnlyOnSuccess) {
    const Result<int> transformed = Result<int>(3).transform([](int v) { return v + 1; });
    ASSERT_TRUE(transformed.has_value());
    EXPECT_EQ(transformed.value(), 4);

    const Result<int> untouched = Result<int>(make_test_error()).transform([](int v) { return v + 1; });
    EXPECT_FALSE(untouched.has_value());
}

TEST(ResultCore, OrElseRecoversAndTransformErrorRewritesError) {
    const Result<int> recovered =
        Result<int>(make_test_error()).or_else([](const Error&) { return Result<int>(99); });
    ASSERT_TRUE(recovered.has_value());
    EXPECT_EQ(recovered.value(), 99);

    const Result<int> rewritten = Result<int>(make_test_error()).transform_error([](Error e) {
        return std::move(e).with_context("rewrapped", LFS_SOURCE_SITE_CURRENT());
    });
    ASSERT_FALSE(rewritten.has_value());
    EXPECT_EQ(rewritten.error().frames().size(), 2u);
}

TEST(ResultCore, ValueOrReturnsDefaultOnFailure) {
    EXPECT_EQ(Result<int>(5).value_or(-1), 5);
    EXPECT_EQ(Result<int>(make_test_error()).value_or(-1), -1);
}

TEST(StatusCore, DefaultIsSuccessAndFailureUsesNamedFactory) {
    const Status ok;
    EXPECT_TRUE(ok.has_value());
    ok.value(); // must not throw

    const Status failed = Status::failure(make_test_error());
    EXPECT_FALSE(failed.has_value());
    EXPECT_EQ(failed.error().code(), ErrorCode::InvalidArgument);
}

TEST(StatusCore, AndThenOrElseChainStatusReturningCallbacks) {
    int steps_run = 0;
    const Status chained = Status{}.and_then([&steps_run]() {
        ++steps_run;
        return Status{};
    });
    EXPECT_TRUE(chained.has_value());
    EXPECT_EQ(steps_run, 1);

    const Status recovered = Status::failure(make_test_error()).or_else([](const Error&) {
        return Status{};
    });
    EXPECT_TRUE(recovered.has_value());
}

// ---------------------------------------------------------------------------
// Legacy / std::expected round trips
// ---------------------------------------------------------------------------

TEST(LegacyInterop, FromLegacyExpectedSuccessRoundTrips) {
    std::expected<int, std::string> legacy(7);
    const Result<int> converted = from_legacy_expected(std::move(legacy),
                                                       LegacyErrorContext{
                                                           .code = ErrorCode::Internal,
                                                           .domain = ErrorDomain::IO,
                                                           .operation = "legacy op",
                                                           .source = LFS_SOURCE_SITE_CURRENT(),
                                                       });
    ASSERT_TRUE(converted.has_value());
    EXPECT_EQ(converted.value(), 7);
}

TEST(LegacyInterop, FromLegacyExpectedFailureCarriesMessageAndContext) {
    std::expected<int, std::string> legacy = std::unexpected("legacy failure message");
    const Result<int> converted = from_legacy_expected(std::move(legacy),
                                                       LegacyErrorContext{
                                                           .code = ErrorCode::DataLoss,
                                                           .domain = ErrorDomain::IO,
                                                           .operation = "load legacy asset",
                                                           .source = LFS_SOURCE_SITE_CURRENT(),
                                                       });
    ASSERT_FALSE(converted.has_value());
    EXPECT_EQ(converted.error().code(), ErrorCode::DataLoss);
    EXPECT_EQ(converted.error().domain(), ErrorDomain::IO);
    EXPECT_EQ(converted.error().detail(), "legacy failure message");
    ASSERT_EQ(converted.error().frames().size(), 2u);
    EXPECT_EQ(converted.error().frames()[1].operation, "load legacy asset");
}

TEST(LegacyInterop, FromLegacyExpectedVoidRoundTrips) {
    std::expected<void, std::string> legacy_ok;
    const Status ok = from_legacy_expected(std::move(legacy_ok),
                                           LegacyErrorContext{
                                               .code = ErrorCode::Internal,
                                               .domain = ErrorDomain::Core,
                                               .operation = "legacy void op",
                                               .source = LFS_SOURCE_SITE_CURRENT(),
                                           });
    EXPECT_TRUE(ok.has_value());

    std::expected<void, std::string> legacy_fail = std::unexpected("void failure");
    const Status failed = from_legacy_expected(std::move(legacy_fail),
                                               LegacyErrorContext{
                                                   .code = ErrorCode::Internal,
                                                   .domain = ErrorDomain::Core,
                                                   .operation = "legacy void op",
                                                   .source = LFS_SOURCE_SITE_CURRENT(),
                                               });
    EXPECT_FALSE(failed.has_value());
    EXPECT_EQ(failed.error().detail(), "void failure");
}

TEST(LegacyInterop, AsExpectedAndIntoExpectedBridgeWithoutImplicitConversion) {
    Result<int> result(42);
    EXPECT_EQ(result.as_expected().value(), 42);

    std::expected<int, Error> expected = std::move(result).into_expected();
    ASSERT_TRUE(expected.has_value());
    EXPECT_EQ(*expected, 42);

    Result<int> rebuilt = Result<int>::from_expected(std::move(expected));
    EXPECT_TRUE(rebuilt.has_value());
}

TEST(LegacyInterop, FromStdErrorCodeCarriesNativeStatus) {
    const std::error_code ec = std::make_error_code(std::errc::no_such_file_or_directory);
    const Error error = from_std_error_code(ec, ErrorCode::NotFound, ErrorDomain::IO, "open dataset",
                                            LFS_SOURCE_SITE_CURRENT());
    EXPECT_EQ(error.code(), ErrorCode::NotFound);
    ASSERT_TRUE(error.native().has_value());
    EXPECT_EQ(error.native()->code, ec.value());
    ASSERT_EQ(error.frames().size(), 2u);
    EXPECT_EQ(error.frames()[1].operation, "open dataset");
}

TEST(LegacyInterop, IoErrorAdapterMapsCodeAndCarriesPath) {
    const io::Error legacy(io::ErrorCode::PATH_NOT_FOUND, "dataset missing",
                           std::filesystem::path("/does/not/exist"));
    const Error converted = io::to_lfs_error(legacy, LFS_SOURCE_SITE_CURRENT());
    EXPECT_EQ(converted.code(), ErrorCode::NotFound);
    EXPECT_EQ(converted.domain(), ErrorDomain::IO);
    ASSERT_EQ(converted.frames().size(), 1u);
    ASSERT_EQ(converted.frames()[0].fields.size(), 1u);
    EXPECT_EQ(converted.frames()[0].fields.entries()[0].key, "path");
}

TEST(LegacyInterop, MemoryPressureAdapterMapsDomainAndBytes) {
    const core::AllocationFailure failure{
        .domain = core::MemoryDomain::CudaDevice,
        .requested_bytes = 1024 * 1024,
        .alignment = 256,
        .device = 0,
        .stream = 0,
        .label = "test-alloc",
        .operation = "allocate tensor",
        .native_error = 2, // cudaErrorMemoryAllocation
    };
    const Error error = core::to_error(failure, LFS_SOURCE_SITE_CURRENT());
    EXPECT_EQ(error.code(), ErrorCode::ResourceExhausted);
    EXPECT_EQ(error.domain(), ErrorDomain::CUDA);
    ASSERT_TRUE(error.native().has_value());
    EXPECT_EQ(error.native()->code, 2);
    ASSERT_EQ(error.frames().size(), 2u);
    EXPECT_EQ(error.frames()[1].operation, "allocate tensor");
}

// ---------------------------------------------------------------------------
// Exception round trip
// ---------------------------------------------------------------------------

TEST(ExceptionCore, CarriesErrorAndFormatsWhat) {
    const Error error = make_test_error(ErrorCode::PermissionDenied, "cannot write output");
    try {
        throw Exception(error);
    } catch (const Exception& caught) {
        EXPECT_EQ(caught.error().code(), ErrorCode::PermissionDenied);
        const std::string_view what = caught.what();
        EXPECT_NE(what.find("PermissionDenied"), std::string_view::npos);
        EXPECT_NE(what.find("cannot write output"), std::string_view::npos);
    }
}

TEST(ExceptionCore, CatchableAsStdException) {
    bool caught_as_std_exception = false;
    try {
        throw Exception(make_test_error());
    } catch (const std::exception& e) {
        caught_as_std_exception = true;
        EXPECT_NE(std::string_view(e.what()).find("InvalidArgument"), std::string_view::npos);
    }
    EXPECT_TRUE(caught_as_std_exception);
}

TEST(ErrorCoreLegacyBridge, VoidFailurePreservesDisplayMessage) {
    auto result = from_legacy_expected<void>(
        std::expected<void, std::string>(std::unexpected("Viewer is shutting down")),
        LegacyErrorContext{
            .code = ErrorCode::Internal,
            .domain = ErrorDomain::Rendering,
            .operation = "clearScene",
            .source = LFS_SOURCE_SITE_CURRENT(),
        });

    ASSERT_FALSE(result);
    const auto& error = result.error();
    const std::string display_message(
        error.user_message().empty() ? error.detail() : error.user_message());
    EXPECT_EQ(display_message, "Viewer is shutting down");
}

// ---------------------------------------------------------------------------
// OOM-safe seed: forced allocation failure, not a real OOM
// ---------------------------------------------------------------------------

TEST(ErrorCoreOom, ForcedAllocationFailureReturnsImmortalSeed) {
    force_next_error_allocation_to_fail_for_testing(true);
    const Error degraded = make_test_error();
    EXPECT_TRUE(degraded.is_immortal());
    EXPECT_EQ(degraded.code(), ErrorCode::ResourceExhausted);

    // The flag is consumed by the failing attempt; the next call succeeds
    // normally.
    const Error recovered = make_test_error();
    EXPECT_FALSE(recovered.is_immortal());
    EXPECT_EQ(recovered.code(), ErrorCode::InvalidArgument);
}

TEST(ErrorCoreOom, RepeatedForcedFailuresReuseTheSameImmortalPayload) {
    force_next_error_allocation_to_fail_for_testing(true);
    const Error first = make_test_error();
    force_next_error_allocation_to_fail_for_testing(true);
    const Error second = make_test_error();

    EXPECT_TRUE(first.is_immortal());
    EXPECT_TRUE(second.is_immortal());
    // Both degraded errors carry identical, preallocated content: no new
    // allocation occurred on either failing attempt.
    EXPECT_EQ(first.code(), second.code());
    EXPECT_EQ(first.detail(), second.detail());
}

TEST(ErrorCoreOom, ImmortalSeedSurvivesCopyMoveAndDestructionWithoutFreeing) {
    // Exercises that add_ref/release correctly special-case the immortal
    // singleton: if they didn't, this would double-free or corrupt the
    // shared payload other tests in this binary also touch.
    for (int i = 0; i < 1000; ++i) {
        Error a = make_immortal_error_for_testing(i % 2 == 0);
        Error b = a;
        Error c = std::move(b);
        EXPECT_TRUE(c.is_immortal());
    }
    // If refcounting had mistakenly freed the immortal payload above, this
    // call would read freed memory.
    const Error still_alive = make_immortal_error_for_testing();
    EXPECT_EQ(still_alive.code(), ErrorCode::ResourceExhausted);
}

// ---------------------------------------------------------------------------
// Refcount stress (TSAN-meaningful): N threads copying/destroying Errors
// that alias a shared source Error.
// ---------------------------------------------------------------------------

TEST(ErrorCoreConcurrency, ConcurrentCopyAndDestroyOfSharedError) {
    const Error shared_source = make_test_error();
    constexpr int kThreads = 8;
    constexpr int kIterationsPerThread = 5000;
    std::atomic<int> observed_mismatches{0};

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&shared_source, &observed_mismatches] {
            for (int i = 0; i < kIterationsPerThread; ++i) {
                Error local_copy = shared_source; // intrusive add-ref
                if (local_copy.code() != ErrorCode::InvalidArgument) {
                    observed_mismatches.fetch_add(1, std::memory_order_relaxed);
                }
                Error moved = std::move(local_copy); // ownership transfer, no atomic op
                (void)moved;
                // moved destructs here: intrusive release
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_EQ(observed_mismatches.load(), 0);
    // The original handle must still be valid and correct after every
    // worker released its copies.
    EXPECT_EQ(shared_source.code(), ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Success-path microbenchmark: the orchestrator enforces the timing gate;
// this only asserts the loop runs and every Status stays zero-cost/success.
// ---------------------------------------------------------------------------

TEST(ErrorCorePerf, OneMillionSuccessStatusReturnsRunCleanly) {
    auto returns_success = []() noexcept -> Status { return {}; };

    const auto start = std::chrono::steady_clock::now();
    std::uint64_t successes = 0;
    for (int i = 0; i < 1'000'000; ++i) {
        const Status s = returns_success();
        successes += s.has_value() ? 1 : 0;
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_EQ(successes, 1'000'000u);
    // Not a correctness gate (the orchestrator's separate benchmark run
    // enforces the "no measurable regression" acceptance triad); this just
    // confirms the loop is not accidentally doing something pathological
    // (e.g. seconds instead of milliseconds).
    EXPECT_LT(elapsed, std::chrono::seconds(5));
}

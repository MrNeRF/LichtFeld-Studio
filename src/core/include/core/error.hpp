/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#if defined(__CUDACC__)
#error "core/error.hpp is host-C++23-only and must never be parsed by nvcc. " \
       "CUDA translation units use core/error_codes.hpp (ErrorCode/ErrorDomain) " \
       "and core/source_site.hpp instead; convert to lfs::Error only in an " \
       "adjacent host .cpp."
#endif

#if defined(_MSC_VER)
#if defined(_MSVC_LANG) && _MSVC_LANG < 202302L
#error "core/error.hpp requires /std:c++23."
#endif
#elif __cplusplus < 202302L
#error "core/error.hpp requires a host C++23 compiler."
#endif

#include "core/assert.hpp"
#include "core/error_codes.hpp"
#include "core/export.hpp"
#include "core/source_site.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// lfs::Error / lfs::Result<T> — see .codex_tmp/error-architecture-analysis.md
// Section 5.2 / 7.2 for the frozen design rationale. Summary of the load
// bearing invariants enforced by error.cpp:
//   * sizeof(Error) == sizeof(void*); a default Error is a null payload
//     pointer and represents success only (Result<void>'s empty state).
//   * ErrorPayload is allocated once, published, and never mutated in place
//     while shared. Error's copy constructor is a thread-safe intrusive
//     add-ref; the destructor releases and frees on last release. Success
//     construction/move touches neither the heap nor the refcount.
//   * with_context()/with_suppressed() are rvalue, copy-on-write: an
//     exclusively-owned payload grows in place, a shared one is cloned first.
//     Both are best-effort — if the bounded cap is already full, or the COW
//     clone allocation fails, the call is a no-op and returns the error
//     unchanged rather than throwing.
namespace lfs {

    // Frozen bounds (Section 7.2). Exceeding a bound silently drops the new
    // item and preserves everything already recorded; construction/context
    // addition never fails or throws because of a bound.
    inline constexpr std::size_t kMaxErrorContextFrames = 16;
    inline constexpr std::size_t kMaxSuppressedErrors = 8;
    inline constexpr std::size_t kMaxFieldsPerFrame = 16;
    inline constexpr std::size_t kMaxDeveloperStringBytes = 4096;
    inline constexpr std::size_t kMaxSerializedErrorBytes = 32768;

    // Truncates `value` to at most `max_bytes` bytes without splitting a
    // UTF-8 multi-byte sequence. No-op if already within bound. Exposed
    // because every developer-facing string field (user_message, detail,
    // NativeError::name, frame operation, SmallFields string values) is
    // truncated through this single, independently-testable routine.
    [[nodiscard]] LFS_CORE_API std::string truncate_utf8_safe(std::string value,
                                                              std::size_t max_bytes) noexcept;

    // How urgently a failure needs attention. Not part of the frozen
    // taxonomy (Section 5.2 names the field but does not enumerate values);
    // kept small and unopinionated about presentation, which is Phase 8's
    // job (ErrorBus severity-to-surface mapping).
    enum class Severity : std::uint8_t {
        Info,
        Warning,
        Error,
        Fatal,
    };

    // Whether the operation that produced this error is safe to retry as-is.
    // Retryability values are implementer latitude per Section 7.1 (Phase 1
    // freezes the payload shape, not this enum's members); Phase 5 pins the
    // exact one-retry FastGS forward-attempt policy on top of this.
    enum class Retryability : std::uint8_t {
        NotRetryable,
        Retryable,
        RetryableWithBackoff,
    };

    // Opaque correlation id for one owning action (training run, import,
    // MCP request, export). Generated once at the owning boundary and
    // copied into every log/event/error touched by that action; the empty
    // id (value() == 0) means "no operation context was attached."
    class LFS_CORE_API OperationId {
    public:
        constexpr OperationId() noexcept = default;

        // Process-wide, monotonically increasing; never returns the empty id.
        [[nodiscard]] static OperationId generate() noexcept;

        [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }
        [[nodiscard]] constexpr bool has_value() const noexcept { return value_ != 0; }

        friend constexpr bool operator==(OperationId, OperationId) noexcept = default;

    private:
        constexpr explicit OperationId(const std::uint64_t value) noexcept
            : value_(value) {}

        std::uint64_t value_ = 0;
    };

    // Native status preserved alongside the stable ErrorCode, e.g.
    // {ErrorDomain::CUDA, cudaErrorMemoryAllocation, "cudaErrorMemoryAllocation"}
    // or {ErrorDomain::Vulkan, VK_ERROR_OUT_OF_DEVICE_MEMORY, "VK_ERROR_..."}.
    // `name` is constructed only on the failure branch.
    struct NativeError {
        ErrorDomain domain;
        std::int64_t code;
        std::string name;
    };

    // Bounded, ordered key/value fields attached to one ErrorFrame (path,
    // iteration, stream, bytes, job id, ...). Cold-path builder type: not
    // itself COW or refcounted, just moved into the frame it describes.
    // Capacity is fixed at kMaxFieldsPerFrame; add() beyond that is a silent
    // no-op and sets overflowed() so a formatter can note the truncation.
    class LFS_CORE_API SmallFields {
    public:
        using Value = std::variant<std::monostate, bool, std::int64_t, std::uint64_t, double, std::string>;

        struct Entry {
            std::string key;
            Value value;
        };

        SmallFields() noexcept = default;

        // Unqualified (not &-only): the common pattern is chaining directly
        // off a temporary, e.g. SmallFields{}.add("path", p).add("bytes", n).
        SmallFields& add(std::string_view key, bool value);
        SmallFields& add(std::string_view key, std::int64_t value);
        SmallFields& add(std::string_view key, std::uint64_t value);
        SmallFields& add(std::string_view key, double value);
        SmallFields& add(std::string_view key, std::string value);
        SmallFields& add(std::string_view key, std::string_view value);

        [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
        [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
        // True once one or more add() calls were dropped by the capacity bound.
        [[nodiscard]] bool overflowed() const noexcept { return overflowed_; }
        [[nodiscard]] std::span<const Entry> entries() const noexcept { return entries_; }

    private:
        SmallFields& add_entry(std::string_view key, Value value);

        std::vector<Entry> entries_;
        bool overflowed_ = false;
    };

    // One step of an error's causal chain: the detection site (frame 0,
    // created by make_error from ErrorInit) or one outer caller's context
    // (appended by with_context). Frame 0's `operation` is always empty —
    // ErrorInit has no operation field, only detail/user_message describe
    // the detection site itself.
    struct ErrorFrame {
        std::string operation;
        core::SourceSite source;
        SmallFields fields;
    };

    struct ErrorPayload; // Private, cold-path representation; defined in error.cpp.

    template <class T>
    class Result;

    // Input to make_error(). No `Error(std::string)` constructor exists
    // anywhere in this API; this is the only way to construct a new,
    // non-legacy Error, and it always requires a code, domain, and
    // detection source. Defined before Error itself: it does not reference
    // Error and Error's friend declaration for make_error() needs it complete.
    struct ErrorInit {
        ErrorCode code;
        ErrorDomain domain;
        Severity severity = Severity::Error;
        Retryability retryability = Retryability::NotRetryable;
        OperationId operation_id;
        std::string user_message;
        std::string detail;
        core::SourceSite detection;
        SmallFields fields;
        std::optional<NativeError> native;
    };

    class Error;

    // Declared before Error so the friend declarations inside the class are
    // not the first declarations MSVC sees — attaching dllexport/dllimport
    // only on a later redeclaration is C2375 (redefinition; different linkage).
    [[nodiscard]] LFS_CORE_API Error make_error(ErrorInit init) noexcept;
    [[nodiscard]] LFS_CORE_API Error make_immortal_error_for_testing(bool unknown_seed) noexcept;

    // One-pointer immutable COW error handle. See the file-level comment for
    // the ownership/COW contract. Public surface matches Section 5.2/7.2
    // exactly, plus two additive, non-contradictory members not shown in
    // that pseudocode but required to make the frozen suppressed-error bound
    // constructible/observable: with_suppressed() and suppressed() (see
    // error.cpp for the reasoning) and is_immortal(), used by tests and by
    // future degraded-diagnostics handling to recognize the OOM-safe seed.
    class [[nodiscard("discarded LichtFeld error")]] LFS_CORE_API Error {
    public:
        Error(const Error& other) noexcept;
        Error(Error && other) noexcept;
        Error& operator=(const Error& other) noexcept;
        Error& operator=(Error&& other) noexcept;
        ~Error();

        [[nodiscard]] ErrorCode code() const noexcept;
        [[nodiscard]] ErrorDomain domain() const noexcept;
        [[nodiscard]] Severity severity() const noexcept;
        [[nodiscard]] Retryability retryability() const noexcept;
        [[nodiscard]] std::string_view user_message() const noexcept;
        [[nodiscard]] std::string_view detail() const noexcept;
        [[nodiscard]] std::span<const ErrorFrame> frames() const noexcept;
        [[nodiscard]] const std::optional<NativeError>& native() const noexcept;
        [[nodiscard]] OperationId operation_id() const noexcept;

        // True for the two immortal diagnostic-OOM/unknown-failure seeds
        // returned by make_error() when constructing the real payload failed.
        [[nodiscard]] bool is_immortal() const noexcept;

        // Additive to the frozen Section 5.2 surface: bounded, ordered
        // secondary failures (e.g. a cleanup error alongside the primary
        // cause). Same rvalue/COW/best-effort contract as with_context();
        // beyond kMaxSuppressedErrors the call is a silent no-op.
        [[nodiscard]] std::span<const Error> suppressed() const noexcept;
        Error&& with_suppressed(Error secondary)&&;

        // Appends one outer-caller context frame. Best-effort: a no-op past
        // kMaxErrorContextFrames, on an immortal seed, or if the COW clone
        // allocation fails (the original error is returned unchanged).
        Error&& with_context(std::string operation, core::SourceSite source,
                             SmallFields fields = {})&&;

    private:
        Error() noexcept = default; // empty handle, Result<void> success only
        explicit Error(ErrorPayload * payload) noexcept;

        ErrorPayload* payload_ = nullptr; // exactly one machine word

        template <class>
        friend class Result;
        friend Error make_error(ErrorInit init) noexcept;
        friend Error make_immortal_error_for_testing(bool unknown_seed) noexcept;
    };

    static_assert(sizeof(Error) == sizeof(void*));

    // Allocates and publishes a new Error. Never throws: if allocating the
    // payload fails, returns the immortal diagnostic-OOM seed instead (see
    // error.cpp for where that seed lives and how it stays allocation-free
    // on the failing path).
    [[nodiscard]] LFS_CORE_API Error make_error(ErrorInit init) noexcept;

    // Test-only fault injection for the OOM-during-construction path. While
    // armed, the next (and only the next) make_error() call behaves as if
    // its payload allocation threw, exercising the exact catch branch a real
    // allocator failure would take without requiring a real OOM.
    LFS_CORE_API void force_next_error_allocation_to_fail_for_testing(bool should_fail) noexcept;

    // Returns one of the two immortal seeds directly, for identity/bench
    // tests that want to reference them without forcing an allocation
    // failure. `unknown_seed` selects the "unknown terminal failure" seed
    // instead of the default diagnostic-OOM seed.
    [[nodiscard]] LFS_CORE_API Error make_immortal_error_for_testing(bool unknown_seed = false) noexcept;

    namespace detail {

        // Normalizes a callback's return type for and_then/or_else: it may
        // return either Result<U> or std::expected<U, Error>. Primary
        // template is intentionally undefined; result_like is satisfied only
        // for the two specializations below.
        template <class R>
        struct result_like_traits;

        template <class U>
        struct result_like_traits<std::expected<U, Error>> {
            using value_type = U;

            static std::expected<U, Error> to_std_expected(std::expected<U, Error> value) {
                return value;
            }
        };

        template <class U>
        struct result_like_traits<Result<U>> {
            using value_type = U;

            static std::expected<U, Error> to_std_expected(Result<U> value) {
                return std::move(value).into_expected();
            }
        };

        template <class R>
        concept result_like = requires {
                                  typename result_like_traits<std::remove_cvref_t<R>>::value_type;
                              };

    } // namespace detail

    // [[nodiscard]] wrapper around std::expected<T, Error>. Deliberately
    // does NOT inherit std::expected (Section 5.2): this type owns its
    // storage so Result<void> can specialize to a bare Error with no
    // std::expected discriminator overhead. There is no implicit conversion
    // to/from std::expected<T, Error>, std::string, or
    // std::unexpected<std::string> in either direction — from_expected()/
    // as_expected()/into_expected() are the only bridge, and
    // from_legacy_expected() is the only generic string bridge.
    template <class T>
    class [[nodiscard("handle or propagate lfs::Result")]] Result {
    public:
        using value_type = T;
        using error_type = Error;

        template <class U = T>
            requires(!std::same_as<std::remove_cvref_t<U>, Result> &&
                     !std::same_as<std::remove_cvref_t<U>, Error> &&
                     std::constructible_from<T, U &&>)
        Result(U && value) noexcept(std::is_nothrow_constructible_v<T, U&&>) // NOLINT(*-explicit-*)
            : value_(std::forward<U>(value)) {}

        Result(Error error) noexcept // NOLINT(*-explicit-*)
            : value_(std::unexpect, std::move(error)) {}

        Result(const Result&) = default;
        Result(Result &&) = default;
        Result& operator=(const Result&) = default;
        Result& operator=(Result&&) = default;
        ~Result() = default;

        [[nodiscard]] static Result from_expected(std::expected<T, Error> && value) {
            return Result{std::move(value)};
        }

        [[nodiscard]] std::expected<T, Error>& as_expected()& noexcept { return value_; }
        [[nodiscard]] const std::expected<T, Error>& as_expected() const& noexcept { return value_; }
        [[nodiscard]] std::expected<T, Error>&& as_expected()&& noexcept { return std::move(value_); }
        [[nodiscard]] std::expected<T, Error> into_expected()&& { return std::move(value_); }

        [[nodiscard]] explicit operator bool() const noexcept { return value_.has_value(); }
        [[nodiscard]] bool has_value() const noexcept { return value_.has_value(); }

        [[nodiscard]] T& value()& { return value_.value(); }
        [[nodiscard]] const T& value() const& { return value_.value(); }
        [[nodiscard]] T&& value()&& { return std::move(value_).value(); }
        [[nodiscard]] const T&& value() const&& { return std::move(value_).value(); }

        [[nodiscard]] Error& error()& { return value_.error(); }
        [[nodiscard]] const Error& error() const& { return value_.error(); }
        [[nodiscard]] Error&& error()&& { return std::move(value_).error(); }
        [[nodiscard]] const Error&& error() const&& { return std::move(value_).error(); }

        [[nodiscard]] T& operator*()& noexcept { return *value_; }
        [[nodiscard]] const T& operator*() const& noexcept { return *value_; }
        [[nodiscard]] T&& operator*()&& noexcept { return *std::move(value_); }
        [[nodiscard]] const T&& operator*() const&& noexcept { return *std::move(value_); }
        [[nodiscard]] T* operator->() noexcept { return value_.operator->(); }
        [[nodiscard]] const T* operator->() const noexcept { return value_.operator->(); }

        template <class U>
        [[nodiscard]] T value_or(U && default_value) const& {
            return value_.value_or(std::forward<U>(default_value));
        }
        template <class U>
        [[nodiscard]] T value_or(U && default_value)&& {
            return std::move(value_).value_or(std::forward<U>(default_value));
        }

        template <class F>
        [[nodiscard]] auto and_then(F && f)& {
            return and_then_impl(*this, std::forward<F>(f));
        }
        template <class F>
        [[nodiscard]] auto and_then(F && f) const& {
            return and_then_impl(*this, std::forward<F>(f));
        }
        template <class F>
        [[nodiscard]] auto and_then(F && f)&& {
            return and_then_impl(std::move(*this), std::forward<F>(f));
        }
        template <class F>
        [[nodiscard]] auto and_then(F && f) const&& {
            return and_then_impl(std::move(*this), std::forward<F>(f));
        }

        template <class F>
        [[nodiscard]] auto transform(F && f)& {
            return Result<std::remove_cv_t<std::invoke_result_t<F, T&>>>::from_expected(
                value_.transform(std::forward<F>(f)));
        }
        template <class F>
        [[nodiscard]] auto transform(F && f) const& {
            return Result<std::remove_cv_t<std::invoke_result_t<F, const T&>>>::from_expected(
                value_.transform(std::forward<F>(f)));
        }
        template <class F>
        [[nodiscard]] auto transform(F && f)&& {
            return Result<std::remove_cv_t<std::invoke_result_t<F, T&&>>>::from_expected(
                std::move(value_).transform(std::forward<F>(f)));
        }
        template <class F>
        [[nodiscard]] auto transform(F && f) const&& {
            return Result<std::remove_cv_t<std::invoke_result_t<F, const T&&>>>::from_expected(
                std::move(value_).transform(std::forward<F>(f)));
        }

        template <class F>
        [[nodiscard]] auto or_else(F && f)& {
            return or_else_impl(*this, std::forward<F>(f));
        }
        template <class F>
        [[nodiscard]] auto or_else(F && f) const& {
            return or_else_impl(*this, std::forward<F>(f));
        }
        template <class F>
        [[nodiscard]] auto or_else(F && f)&& {
            return or_else_impl(std::move(*this), std::forward<F>(f));
        }
        template <class F>
        [[nodiscard]] auto or_else(F && f) const&& {
            return or_else_impl(std::move(*this), std::forward<F>(f));
        }

        template <class F>
        [[nodiscard]] auto transform_error(F && f)& {
            return Result::from_expected(value_.transform_error(std::forward<F>(f)));
        }
        template <class F>
        [[nodiscard]] auto transform_error(F && f) const& {
            return Result::from_expected(value_.transform_error(std::forward<F>(f)));
        }
        template <class F>
        [[nodiscard]] auto transform_error(F && f)&& {
            return Result::from_expected(std::move(value_).transform_error(std::forward<F>(f)));
        }
        template <class F>
        [[nodiscard]] auto transform_error(F && f) const&& {
            return Result::from_expected(std::move(value_).transform_error(std::forward<F>(f)));
        }

    private:
        explicit Result(std::expected<T, Error> && value) noexcept(std::is_nothrow_move_constructible_v<T>)
            : value_(std::move(value)) {}

        template <class Self, class F>
        static auto and_then_impl(Self && self, F && f) {
            using Ret = std::invoke_result_t<F, decltype(*std::forward<Self>(self).as_expected())>;
            static_assert(detail::result_like<Ret>,
                          "Result<T>::and_then callback must return lfs::Result<U> or "
                          "std::expected<U, lfs::Error>");
            using U = typename detail::result_like_traits<std::remove_cvref_t<Ret>>::value_type;
            auto raw = std::forward<Self>(self).as_expected().and_then(
                [&f](auto&& value) {
                    return detail::result_like_traits<std::remove_cvref_t<Ret>>::to_std_expected(
                        std::invoke(std::forward<F>(f), std::forward<decltype(value)>(value)));
                });
            return Result<U>::from_expected(std::move(raw));
        }

        template <class Self, class F>
        static auto or_else_impl(Self && self, F && f) {
            using Ret = std::invoke_result_t<F, decltype(std::forward<Self>(self).error())>;
            static_assert(detail::result_like<Ret>,
                          "Result<T>::or_else callback must return lfs::Result<T> or "
                          "std::expected<T, lfs::Error>");
            using U = typename detail::result_like_traits<std::remove_cvref_t<Ret>>::value_type;
            static_assert(std::same_as<U, T>,
                          "Result<T>::or_else callback must not change the success value type");
            auto raw = std::forward<Self>(self).as_expected().or_else(
                [&f](auto&& error) {
                    return detail::result_like_traits<std::remove_cvref_t<Ret>>::to_std_expected(
                        std::invoke(std::forward<F>(f), std::forward<decltype(error)>(error)));
                });
            return Result<U>::from_expected(std::move(raw));
        }

        std::expected<T, Error> value_;
    };

    // Pointer-only success/failure handle. A default-constructed Result<void>
    // is success and constructs/moves without allocation or refcounting —
    // its entire representation is Error's null-payload empty state.
    // Deliberately asymmetric with Result<T>: there is no implicit
    // Result<void>(Error) constructor (only failure()), because unlike
    // Result<T> there is no second, unambiguous constructor argument type to
    // overload against for "this is the success case."
    template <>
    class [[nodiscard("handle or propagate lfs::Status")]] Result<void> {
    public:
        using value_type = void;
        using error_type = Error;

        Result() noexcept = default;

        [[nodiscard]] static Result failure(Error error) noexcept {
            Result result;
            result.error_ = std::move(error);
            return result;
        }

        [[nodiscard]] static Result from_expected(std::expected<void, Error> && value) {
            if (value.has_value()) {
                return Result{};
            }
            return failure(std::move(value.error()));
        }

        [[nodiscard]] explicit operator bool() const noexcept { return !is_failure(); }
        [[nodiscard]] bool has_value() const noexcept { return !is_failure(); }

        void value() const {
            LFS_ASSERT_MSG(has_value(), "lfs::Status::value() called on a failed Status");
        }

        [[nodiscard]] Error& error()& {
            LFS_ASSERT_MSG(!has_value(), "lfs::Status::error() called on a successful Status");
            return error_;
        }
        [[nodiscard]] const Error& error() const& {
            LFS_ASSERT_MSG(!has_value(), "lfs::Status::error() called on a successful Status");
            return error_;
        }
        [[nodiscard]] Error&& error()&& {
            LFS_ASSERT_MSG(!has_value(), "lfs::Status::error() called on a successful Status");
            return std::move(error_);
        }

        [[nodiscard]] std::expected<void, Error> into_expected()&& {
            if (has_value()) {
                return {};
            }
            return std::unexpected(std::move(error_));
        }

        template <class F>
        [[nodiscard]] Result and_then(F && f) const {
            if (!has_value()) {
                return *this;
            }
            return std::invoke(std::forward<F>(f));
        }

        template <class F>
        [[nodiscard]] Result or_else(F && f) const {
            if (has_value()) {
                return *this;
            }
            return std::invoke(std::forward<F>(f), error_);
        }

        template <class F>
        [[nodiscard]] Result transform_error(F && f) const {
            if (has_value()) {
                return *this;
            }
            return failure(std::invoke(std::forward<F>(f), error_));
        }

    private:
        // error_ is default-constructed (empty/null payload) exactly when
        // this Status is a success; there is no separate discriminator.
        // Result<T> is a friend of Error for every T, so this reaches
        // Error's private payload_ directly instead of needing a new public
        // Error accessor just for this one internal check.
        [[nodiscard]] bool is_failure() const noexcept { return error_.payload_ != nullptr; }

        Error error_;
    };

    using Status = Result<void>;

    static_assert(sizeof(Status) == sizeof(void*));

    // Bridge for cold exception-oriented code (constructors, deep legacy
    // call chains, foreign-library catch boundaries). Carries the same
    // Error a Result would have, so catching does not destroy structure.
    class LFS_CORE_API Exception final : public std::exception {
    public:
        explicit Exception(Error error) noexcept
            : error_(std::move(error)) {}

        [[nodiscard]] const Error& error() const noexcept { return error_; }
        [[nodiscard]] const char* what() const noexcept override;

    private:
        Error error_;
        mutable std::string what_cache_;
        mutable bool what_cached_ = false;
    };

    // Bounded (<= kMaxSerializedErrorBytes), human-readable rendering of the
    // full cause chain: code/domain/severity/retryability/operation id,
    // native status if present, user_message/detail, each frame's operation/
    // source/fields, and a suppressed-error count. This is the developer
    // diagnostic used by Exception::what(); it is not the Phase 10 wire JSON
    // schema (Section 5.10), which is a distinct, not-yet-implemented format.
    [[nodiscard]] LFS_CORE_API std::string format_for_developer(const Error& error);

    // Context required to bridge one legacy std::expected<T, std::string>
    // (or std::error_code) call into the typed taxonomy. Every call site is
    // meant to be CI-counted/allowlisted per Section 5.2; this is a named,
    // explicit adapter, not an implicit conversion.
    struct LegacyErrorContext {
        ErrorCode code;
        ErrorDomain domain;
        std::string operation;
        core::SourceSite source;
        OperationId operation_id{};
    };

    // Shared out-of-line implementation for from_legacy_expected<T> below;
    // keeps the Error-construction logic out of the header regardless of T.
    // Declared (not just used) before from_legacy_expected: the call inside
    // that template does not depend on T, so ordinary unqualified lookup
    // resolves it at the template's definition point, not at instantiation.
    [[nodiscard]] LFS_CORE_API Error make_legacy_error(std::string legacy_message,
                                                       LegacyErrorContext context);

    // The only generic string bridge. A successful legacy expected converts
    // to a successful Result<T>; a failed one becomes a typed Error carrying
    // the legacy message as both user_message and detail, with one context
    // frame recording `context.operation` at `context.source`.
    template <class T>
    [[nodiscard]] Result<T> from_legacy_expected(std::expected<T, std::string>&& value,
                                                 LegacyErrorContext context) {
        if (value.has_value()) {
            if constexpr (std::is_void_v<T>) {
                return Result<T>{};
            } else {
                return Result<T>(std::move(*value));
            }
        }
        Error error = make_legacy_error(std::move(value.error()), std::move(context));
        if constexpr (std::is_void_v<T>) {
            return Result<T>::failure(std::move(error));
        } else {
            return Result<T>(std::move(error));
        }
    }

    // Explicit adapter from a std::error_code (POSIX/filesystem/etc) call
    // site. `code`/`domain` classify it into the stable taxonomy; the native
    // std::error_code value/category name is preserved in NativeError.
    [[nodiscard]] LFS_CORE_API Error from_std_error_code(std::error_code ec, ErrorCode code,
                                                         ErrorDomain domain, std::string operation,
                                                         core::SourceSite source);

    // Closed reason set for LFS_FATAL_INVARIANT (allocator/crash
    // implementation files only; see Section 5.12). The macro itself is out
    // of Phase 1's scope: it must not be reachable from a public header, and
    // Phase 1 adds zero call sites. This enum exists now because
    // ErrorPayload/Error never reference it and freezing it early costs
    // nothing.
    enum class FatalReason : std::uint8_t {
        HeapMetadataCorruption,
        OwnershipCorruption,
        DiagnosticRequested,
    };

} // namespace lfs

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"

#include "core/memory_pressure.hpp"

#include <atomic>
#include <format>
#include <new>
#include <utility>
#include <variant>

namespace lfs {

    // Private, cold-path representation of a published Error. Never exposed
    // through error.hpp beyond the forward declaration; only reachable via
    // Error's opaque payload_ pointer.
    struct ErrorPayload {
        std::atomic<std::uint32_t> references{1};
        ErrorCode code;
        ErrorDomain domain;
        Severity severity;
        Retryability retryability;
        OperationId operation_id;
        std::string user_message;
        std::string detail;
        std::optional<NativeError> native;
        std::vector<ErrorFrame> frames;
        std::vector<Error> suppressed;

        ErrorPayload(const ErrorCode code, const ErrorDomain domain, const Severity severity,
                     const Retryability retryability, const OperationId operation_id,
                     std::string user_message, std::string detail,
                     std::optional<NativeError> native)
            : code(code),
              domain(domain),
              severity(severity),
              retryability(retryability),
              operation_id(operation_id),
              user_message(std::move(user_message)),
              detail(std::move(detail)),
              native(std::move(native)) {}

        // Copy-on-write clone: same content, a fresh exclusive-ownership
        // refcount, and its own frames/suppressed vectors (deep-copying the
        // vectors but only add-refing the suppressed Errors they hold).
        ErrorPayload(const ErrorPayload& other)
            : references(1),
              code(other.code),
              domain(other.domain),
              severity(other.severity),
              retryability(other.retryability),
              operation_id(other.operation_id),
              user_message(other.user_message),
              detail(other.detail),
              native(other.native),
              frames(other.frames),
              suppressed(other.suppressed) {}

        ErrorPayload& operator=(const ErrorPayload&) = delete;
        ErrorPayload(ErrorPayload&&) = delete;
        ErrorPayload& operator=(ErrorPayload&&) = delete;
    };

    namespace {

        // Test-only OOM injection: while armed, the next make_error() payload
        // allocation is forced to throw instead of actually attempting `new`,
        // exercising the identical catch branch a real allocator failure
        // would take.
        std::atomic<bool>& forced_allocation_failure_flag() noexcept {
            static std::atomic<bool> flag{false};
            return flag;
        }

        bool consume_forced_allocation_failure() noexcept {
            return forced_allocation_failure_flag().exchange(false, std::memory_order_acq_rel);
        }

        // Immortal, preallocated seeds for "allocation failed while
        // constructing diagnostics" and "unknown terminal failure" (Section
        // 5.2's OOM-safe seed). Magic statics: constructed once, on first
        // use, with no members that require anything beyond a
        // small-string-optimized std::string and empty vectors, so the
        // realistic failure mode this guards against — a make_error() call
        // that cannot allocate — finds these already built from whichever
        // earlier, healthy make_error()/format_for_developer() call touched
        // them first. This mirrors the existing allocation-free
        // AllocationFailure seed's role in memory_pressure.hpp, adapted to
        // the fact that ErrorPayload itself is not a POD.
        ErrorPayload& immortal_payload(const bool unknown_seed) noexcept {
            static ErrorPayload oom_seed(ErrorCode::ResourceExhausted, ErrorDomain::Core,
                                         Severity::Fatal, Retryability::NotRetryable,
                                         OperationId{}, "Out of memory", "diag alloc OOM",
                                         std::nullopt);
            static ErrorPayload unknown_seed_payload(ErrorCode::Internal, ErrorDomain::Core,
                                                     Severity::Fatal, Retryability::NotRetryable,
                                                     OperationId{}, "Unknown error",
                                                     "internal fault", std::nullopt);
            return unknown_seed ? unknown_seed_payload : oom_seed;
        }

        bool is_immortal_payload(const ErrorPayload* payload) noexcept {
            return payload == &immortal_payload(false) || payload == &immortal_payload(true);
        }

        void add_ref(ErrorPayload* payload) noexcept {
            if (!payload || is_immortal_payload(payload)) {
                return;
            }
            payload->references.fetch_add(1, std::memory_order_relaxed);
        }

        // Standard intrusive-refcount release: relaxed/release decrement,
        // acquire fence only on the thread that observes the last
        // reference, matching the codebase's other refcounted handles.
        void release(ErrorPayload* payload) noexcept {
            if (!payload || is_immortal_payload(payload)) {
                return;
            }
            if (payload->references.fetch_sub(1, std::memory_order_release) == 1) {
                std::atomic_thread_fence(std::memory_order_acquire);
                delete payload;
            }
        }

        const char* to_c_string(const Severity severity) noexcept {
            switch (severity) {
            case Severity::Info: return "Info";
            case Severity::Warning: return "Warning";
            case Severity::Error: return "Error";
            case Severity::Fatal: return "Fatal";
            }
            return "Unknown";
        }

        const char* to_c_string(const Retryability retryability) noexcept {
            switch (retryability) {
            case Retryability::NotRetryable: return "NotRetryable";
            case Retryability::Retryable: return "Retryable";
            case Retryability::RetryableWithBackoff: return "RetryableWithBackoff";
            }
            return "Unknown";
        }

        void append_field_value(std::string& out, const SmallFields::Value& value) {
            std::visit(
                [&out]<class V>(const V& held) {
                    if constexpr (std::is_same_v<V, std::monostate>) {
                        out += "null";
                    } else if constexpr (std::is_same_v<V, bool>) {
                        out += held ? "true" : "false";
                    } else if constexpr (std::is_same_v<V, std::string>) {
                        out += held;
                    } else {
                        out += std::format("{}", held);
                    }
                },
                value);
        }

    } // namespace

    std::string truncate_utf8_safe(std::string value, const std::size_t max_bytes) noexcept {
        if (value.size() <= max_bytes) {
            return value;
        }
        std::size_t cut = max_bytes;
        // Back off while the first excluded byte is a UTF-8 continuation
        // byte (10xxxxxx): that means `cut` lands mid-sequence, so shrink
        // the kept prefix until it lands exactly on a sequence boundary.
        while (cut > 0 && (static_cast<unsigned char>(value[cut]) & 0xC0) == 0x80) {
            --cut;
        }
        value.resize(cut);
        return value;
    }

    OperationId OperationId::generate() noexcept {
        static std::atomic<std::uint64_t> counter{0};
        return OperationId{counter.fetch_add(1, std::memory_order_relaxed) + 1};
    }

    SmallFields& SmallFields::add_entry(const std::string_view key, Value value) {
        if (entries_.size() >= kMaxFieldsPerFrame) {
            overflowed_ = true;
            return *this;
        }
        if (auto* str = std::get_if<std::string>(&value)) {
            *str = truncate_utf8_safe(std::move(*str), kMaxDeveloperStringBytes);
        }
        entries_.push_back(Entry{std::string(key), std::move(value)});
        return *this;
    }

    SmallFields& SmallFields::add(const std::string_view key, const bool value) {
        return add_entry(key, Value{value});
    }
    SmallFields& SmallFields::add(const std::string_view key, const std::int64_t value) {
        return add_entry(key, Value{value});
    }
    SmallFields& SmallFields::add(const std::string_view key, const std::uint64_t value) {
        return add_entry(key, Value{value});
    }
    SmallFields& SmallFields::add(const std::string_view key, const double value) {
        return add_entry(key, Value{value});
    }
    SmallFields& SmallFields::add(const std::string_view key, std::string value) {
        return add_entry(key, Value{std::move(value)});
    }
    SmallFields& SmallFields::add(const std::string_view key, const std::string_view value) {
        return add_entry(key, Value{std::string(value)});
    }

    Error::Error(const Error& other) noexcept
        : payload_(other.payload_) {
        add_ref(payload_);
    }

    Error::Error(Error&& other) noexcept
        : payload_(std::exchange(other.payload_, nullptr)) {}

    Error& Error::operator=(const Error& other) noexcept {
        if (this != &other) {
            add_ref(other.payload_);
            release(payload_);
            payload_ = other.payload_;
        }
        return *this;
    }

    Error& Error::operator=(Error&& other) noexcept {
        if (this != &other) {
            release(payload_);
            payload_ = std::exchange(other.payload_, nullptr);
        }
        return *this;
    }

    Error::~Error() {
        release(payload_);
    }

    Error::Error(ErrorPayload* payload) noexcept
        : payload_(payload) {}

    ErrorCode Error::code() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr, "lfs::Error::code() called on an empty (success) handle");
        return payload_->code;
    }

    ErrorDomain Error::domain() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr, "lfs::Error::domain() called on an empty (success) handle");
        return payload_->domain;
    }

    Severity Error::severity() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr, "lfs::Error::severity() called on an empty (success) handle");
        return payload_->severity;
    }

    Retryability Error::retryability() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr,
                       "lfs::Error::retryability() called on an empty (success) handle");
        return payload_->retryability;
    }

    std::string_view Error::user_message() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr,
                       "lfs::Error::user_message() called on an empty (success) handle");
        return payload_->user_message;
    }

    std::string_view Error::detail() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr, "lfs::Error::detail() called on an empty (success) handle");
        return payload_->detail;
    }

    std::span<const ErrorFrame> Error::frames() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr, "lfs::Error::frames() called on an empty (success) handle");
        return payload_->frames;
    }

    const std::optional<NativeError>& Error::native() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr, "lfs::Error::native() called on an empty (success) handle");
        return payload_->native;
    }

    OperationId Error::operation_id() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr,
                       "lfs::Error::operation_id() called on an empty (success) handle");
        return payload_->operation_id;
    }

    bool Error::is_immortal() const noexcept {
        return is_immortal_payload(payload_);
    }

    std::span<const Error> Error::suppressed() const noexcept {
        LFS_ASSERT_MSG(payload_ != nullptr,
                       "lfs::Error::suppressed() called on an empty (success) handle");
        return payload_->suppressed;
    }

    Error&& Error::with_suppressed(Error secondary) && {
        if (!payload_ || is_immortal_payload(payload_)) {
            return std::move(*this);
        }
        if (payload_->suppressed.size() >= kMaxSuppressedErrors) {
            return std::move(*this); // bound reached: drop silently, keep what is already recorded
        }
        if (payload_->references.load(std::memory_order_acquire) == 1) {
            try {
                payload_->suppressed.push_back(std::move(secondary));
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): best-effort append; leave the payload exactly as it was.
            }
            return std::move(*this);
        }
        try {
            auto* clone = new ErrorPayload(*payload_);
            clone->suppressed.push_back(std::move(secondary));
            release(payload_);
            payload_ = clone;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): diagnostic allocation failed; keep the original, still-shared error.
        }
        return std::move(*this);
    }

    Error&& Error::with_context(std::string operation, const core::SourceSite source,
                                SmallFields fields) && {
        if (!payload_ || is_immortal_payload(payload_)) {
            return std::move(*this); // success handle or degraded seed: best-effort no-op
        }
        // kMaxErrorContextFrames bounds the frames with_context() adds, on
        // top of (not counting) frame 0, which make_error() always publishes
        // as the detection site. So the vector may hold up to
        // 1 + kMaxErrorContextFrames entries before a new one is dropped.
        if (payload_->frames.size() >= 1 + kMaxErrorContextFrames) {
            return std::move(*this); // bound reached: drop silently, root cause frame is preserved
        }
        ErrorFrame frame{truncate_utf8_safe(std::move(operation), kMaxDeveloperStringBytes), source,
                         std::move(fields)};
        if (payload_->references.load(std::memory_order_acquire) == 1) {
            try {
                payload_->frames.push_back(std::move(frame));
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): best-effort append; leave the payload exactly as it was.
            }
            return std::move(*this);
        }
        try {
            auto* clone = new ErrorPayload(*payload_);
            clone->frames.push_back(std::move(frame));
            release(payload_);
            payload_ = clone;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): diagnostic allocation failed; keep the original, still-shared error.
        }
        return std::move(*this);
    }

    Error make_error(ErrorInit init) noexcept {
        try {
            if (consume_forced_allocation_failure()) {
                throw std::bad_alloc();
            }
            auto* payload = new ErrorPayload(
                init.code, init.domain, init.severity, init.retryability, init.operation_id,
                truncate_utf8_safe(std::move(init.user_message), kMaxDeveloperStringBytes),
                truncate_utf8_safe(std::move(init.detail), kMaxDeveloperStringBytes), std::nullopt);
            if (init.native) {
                NativeError native = std::move(*init.native);
                native.name = truncate_utf8_safe(std::move(native.name), kMaxDeveloperStringBytes);
                payload->native = std::move(native);
            }
            payload->frames.push_back(ErrorFrame{
                std::string{}, // frame 0 has no operation name: ErrorInit carries none
                init.detection,
                std::move(init.fields),
            });
            return Error{payload};
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): make_error is noexcept; failure falls back to the immortal seed.
            return Error{&immortal_payload(false)};
        }
    }

    void force_next_error_allocation_to_fail_for_testing(const bool should_fail) noexcept {
        forced_allocation_failure_flag().store(should_fail, std::memory_order_release);
    }

    Error make_immortal_error_for_testing(const bool unknown_seed) noexcept {
        return Error{&immortal_payload(unknown_seed)};
    }

    const char* Exception::what() const noexcept {
        if (!what_cached_) {
            try {
                what_cache_ = format_for_developer(error_);
                what_cached_ = true;
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): what() is noexcept; a static literal avoids allocating here.
                return "lfs::Exception (formatting failed)";
            }
        }
        return what_cache_.c_str();
    }

    std::string format_for_developer(const Error& error) {
        std::string out = std::format(
            "lfs::Error[{}/{}] severity={} retryability={}\n", to_string(error.code()),
            to_string(error.domain()), to_c_string(error.severity()), to_c_string(error.retryability()));
        if (error.operation_id().has_value()) {
            out += std::format("  operation_id={}\n", error.operation_id().value());
        }
        if (const auto& native = error.native(); native.has_value()) {
            out += std::format("  native: {}/{} \"{}\"\n", to_string(native->domain), native->code,
                               native->name);
        }
        if (!error.user_message().empty()) {
            out += std::format("  user_message: {}\n", error.user_message());
        }
        if (!error.detail().empty()) {
            out += std::format("  detail: {}\n", error.detail());
        }
        std::size_t frame_index = 0;
        for (const ErrorFrame& frame : error.frames()) {
            out += std::format("  frame[{}] {}:{} {}", frame_index, frame.source.file_name(),
                               frame.source.line(), frame.source.function_name());
            if (!frame.operation.empty()) {
                out += std::format(" — {}", frame.operation);
            }
            out += "\n";
            for (const auto& entry : frame.fields.entries()) {
                out += std::format("    {}=", entry.key);
                append_field_value(out, entry.value);
                out += "\n";
            }
            if (frame.fields.overflowed()) {
                out += "    (additional fields dropped: over the per-frame cap)\n";
            }
            ++frame_index;
        }
        if (!error.suppressed().empty()) {
            out += std::format("  suppressed: {} error(s)\n", error.suppressed().size());
        }
        return truncate_utf8_safe(std::move(out), kMaxSerializedErrorBytes);
    }

    Error make_legacy_error(std::string legacy_message, LegacyErrorContext context) {
        Error error = make_error(ErrorInit{
            .code = context.code,
            .domain = context.domain,
            .severity = Severity::Error,
            .retryability = Retryability::NotRetryable,
            .operation_id = context.operation_id,
            .user_message = legacy_message,
            .detail = std::move(legacy_message),
            .detection = context.source,
            .fields = SmallFields{},
            .native = std::nullopt,
        });
        return std::move(error).with_context(std::move(context.operation), context.source);
    }

    Error from_std_error_code(const std::error_code ec, const ErrorCode code, const ErrorDomain domain,
                              std::string operation, const core::SourceSite source) {
        NativeError native{
            .domain = domain,
            .code = ec.value(),
            .name = std::format("{}: {}", ec.category().name(), ec.message()),
        };
        Error error = make_error(ErrorInit{
            .code = code,
            .domain = domain,
            .severity = Severity::Error,
            .retryability = Retryability::NotRetryable,
            .operation_id = OperationId{},
            .user_message = ec.message(),
            .detail = ec.message(),
            .detection = source,
            .fields = SmallFields{},
            .native = std::move(native),
        });
        return std::move(error).with_context(std::move(operation), source);
    }

} // namespace lfs

namespace lfs::core {

    namespace {

        ErrorDomain map_memory_domain(const MemoryDomain domain) noexcept {
            switch (domain) {
            case MemoryDomain::CudaDevice:
            case MemoryDomain::CudaVmm:
                return ErrorDomain::CUDA;
            case MemoryDomain::VulkanDevice:
                return ErrorDomain::Vulkan;
            case MemoryDomain::PinnedHost:
            case MemoryDomain::PageableHost:
                return ErrorDomain::Core;
            }
            return ErrorDomain::Core;
        }

    } // namespace

    // Phase 1 error-architecture adapter (declared in memory_pressure.hpp,
    // defined here so that header stays free of core/error.hpp and therefore
    // safe for any future CUDA translation unit that includes it).
    Error to_error(const AllocationFailure& failure, const SourceSite site) {
        const ErrorDomain domain = map_memory_domain(failure.domain);

        SmallFields fields;
        fields.add("memory_domain", std::string_view(to_string(failure.domain)));
        fields.add("requested_bytes", static_cast<std::uint64_t>(failure.requested_bytes));
        if (failure.alignment != 0) {
            fields.add("alignment", static_cast<std::uint64_t>(failure.alignment));
        }
        fields.add("device", static_cast<std::int64_t>(failure.device));
        if (failure.stream != 0) {
            fields.add("stream", static_cast<std::uint64_t>(failure.stream));
        }
        if (failure.label) {
            fields.add("label", std::string_view(failure.label));
        }

        std::optional<NativeError> native;
        if (failure.native_error != 0) {
            native = NativeError{
                .domain = domain,
                .code = failure.native_error,
                .name = "native allocation status",
            };
        }

        Error error = make_error(ErrorInit{
            .code = ErrorCode::ResourceExhausted,
            .domain = domain,
            .severity = Severity::Error,
            .retryability = Retryability::Retryable,
            .operation_id = OperationId{},
            .user_message = "Out of memory",
            .detail = std::format("allocation of {} bytes failed", failure.requested_bytes),
            .detection = site,
            .fields = std::move(fields),
            .native = std::move(native),
        });
        return std::move(error).with_context(failure.operation ? failure.operation : "allocate", site);
    }

} // namespace lfs::core

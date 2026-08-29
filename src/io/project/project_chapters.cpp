/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_chapters.hpp"

#include "core/path_utils.hpp"

#include <xxhash.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace lfs::io::project {

    namespace {

        using Json = JsonChapterDom::Json;
        constexpr std::size_t FINGERPRINT_WINDOW_BYTES = 64 * 1024;
        constexpr std::size_t MAX_DIRECTORY_ENTRIES = 1'000'000;
        constexpr std::size_t MAX_DIRECTORY_INVENTORY_BYTES = 256 * 1024 * 1024;
        constexpr std::size_t FULL_HASH_STREAM_BYTES = 1024 * 1024;

        lfs::Error chapter_error(const lfs::ErrorCode code, std::string message,
                                 std::string detail, const std::string_view chapter,
                                 const std::string_view field = {}) {
            lfs::SmallFields fields;
            fields.add("chapter", chapter);
            if (!field.empty()) {
                fields.add("field", field);
            }
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            });
        }

        template <typename T>
        lfs::Result<T> fail(const lfs::ErrorCode code, std::string message,
                            std::string detail, const std::string_view chapter,
                            const std::string_view field = {}) {
            if constexpr (std::same_as<T, void>) {
                return lfs::Result<void>::failure(
                    chapter_error(code, std::move(message), std::move(detail), chapter, field));
            } else {
                return chapter_error(
                    code, std::move(message), std::move(detail), chapter, field);
            }
        }

        template <typename... Results>
        std::optional<lfs::Error> first_error(Results&... results) {
            std::optional<lfs::Error> error;
            const auto inspect = [&error](auto& result) {
                if (!error && !result) {
                    error.emplace(std::move(result).error());
                }
            };
            (inspect(results), ...);
            return error;
        }

        lfs::Result<void> require_object(const Json& value, const std::string_view chapter,
                                         const std::string_view field) {
            if (!value.is_object()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss, "The project chapter has an invalid structure.",
                    std::format("{}.{} is {}, expected object", chapter, field, value.type_name()),
                    chapter, field);
            }
            return {};
        }

        template <typename T>
        lfs::Result<T> required(const Json& object, const std::string_view key,
                                const std::string_view chapter,
                                const std::string_view field_prefix = {}) {
            const std::string field =
                field_prefix.empty() ? std::string(key)
                                     : std::format("{}.{}", field_prefix, key);
            if (!object.is_object()) {
                return fail<T>(
                    lfs::ErrorCode::DataLoss, "The project chapter has an invalid structure.",
                    std::format("Parent of {}.{} is {}, expected object", chapter, field,
                                object.type_name()),
                    chapter, field);
            }
            const auto found = object.find(std::string(key));
            if (found == object.end()) {
                return fail<T>(
                    lfs::ErrorCode::DataLoss, "The project chapter is missing a required field.",
                    std::format("{}.{} is missing", chapter, field), chapter, field);
            }
            try {
                return found->template get<T>();
            } catch (const nlohmann::json::exception& error) {
                return fail<T>(
                    lfs::ErrorCode::DataLoss, "The project chapter contains an invalid field.",
                    std::format("{}.{} cannot be decoded: {}", chapter, field, error.what()),
                    chapter, field);
            }
        }

        template <typename T>
        lfs::Result<std::optional<T>> optional(const Json& object,
                                               const std::string_view key,
                                               const std::string_view chapter,
                                               const std::string_view field_prefix = {}) {
            const std::string field =
                field_prefix.empty() ? std::string(key)
                                     : std::format("{}.{}", field_prefix, key);
            if (!object.is_object()) {
                return fail<std::optional<T>>(
                    lfs::ErrorCode::DataLoss, "The project chapter has an invalid structure.",
                    std::format("Parent of {}.{} is {}, expected object", chapter, field,
                                object.type_name()),
                    chapter, field);
            }
            const auto found = object.find(std::string(key));
            if (found == object.end() || found->is_null()) {
                return std::optional<T>{};
            }
            try {
                return std::optional<T>(found->template get<T>());
            } catch (const nlohmann::json::exception& error) {
                return fail<std::optional<T>>(
                    lfs::ErrorCode::DataLoss, "The project chapter contains an invalid field.",
                    std::format("{}.{} cannot be decoded: {}", chapter, field, error.what()),
                    chapter, field);
            }
        }

        lfs::Result<lfs::core::Uuid> parse_uuid(const Json& value,
                                                const std::string_view chapter,
                                                const std::string_view field) {
            if (!value.is_string()) {
                return fail<lfs::core::Uuid>(
                    lfs::ErrorCode::DataLoss, "The project contains an invalid UUID.",
                    std::format("{}.{} is {}, expected canonical UUID string", chapter, field,
                                value.type_name()),
                    chapter, field);
            }
            auto parsed = lfs::core::Uuid::from_string(value.get_ref<const std::string&>());
            if (!parsed || parsed->is_nil()) {
                return fail<lfs::core::Uuid>(
                    lfs::ErrorCode::DataLoss, "The project contains an invalid UUID.",
                    std::format("{}.{} is not a non-null canonical UUID", chapter, field),
                    chapter, field);
            }
            return *parsed;
        }

        lfs::Result<std::optional<lfs::core::Uuid>>
        parse_optional_uuid(const Json& object, const std::string_view key,
                            const std::string_view chapter,
                            const std::string_view field_prefix = {}) {
            const std::string field =
                field_prefix.empty() ? std::string(key)
                                     : std::format("{}.{}", field_prefix, key);
            if (!object.is_object()) {
                return fail<std::optional<lfs::core::Uuid>>(
                    lfs::ErrorCode::DataLoss, "The project chapter has an invalid structure.",
                    std::format("Parent of {}.{} is {}, expected object", chapter, field,
                                object.type_name()),
                    chapter, field);
            }
            const auto found = object.find(std::string(key));
            if (found == object.end() || found->is_null()) {
                return std::optional<lfs::core::Uuid>{};
            }
            auto parsed = parse_uuid(*found, chapter, field);
            if (!parsed) {
                return std::move(parsed).error();
            }
            return std::optional<lfs::core::Uuid>(*parsed);
        }

        template <typename T, std::size_t N>
        lfs::Result<std::array<T, N>> fixed_array(const Json& value,
                                                  const std::string_view chapter,
                                                  const std::string_view field) {
            if (!value.is_array() || value.size() != N) {
                return fail<std::array<T, N>>(
                    lfs::ErrorCode::DataLoss, "The project contains an invalid fixed-size array.",
                    std::format("{}.{} has {} entries, expected {}", chapter, field,
                                value.is_array() ? value.size() : 0, N),
                    chapter, field);
            }
            std::array<T, N> result{};
            try {
                for (std::size_t i = 0; i < N; ++i) {
                    result[i] = value[i].template get<T>();
                    if constexpr (std::floating_point<T>) {
                        if (!std::isfinite(result[i])) {
                            return fail<std::array<T, N>>(
                                lfs::ErrorCode::DataLoss,
                                "The project contains a non-finite numeric field.",
                                std::format("{}.{}[{}] is not finite", chapter, field, i),
                                chapter, field);
                        }
                    }
                }
            } catch (const nlohmann::json::exception& error) {
                return fail<std::array<T, N>>(
                    lfs::ErrorCode::DataLoss, "The project contains an invalid numeric array.",
                    std::format("{}.{} cannot be decoded: {}", chapter, field, error.what()),
                    chapter, field);
            }
            return result;
        }

        template <typename T, std::size_t N>
        Json json_array(const std::array<T, N>& value) {
            Json result = Json::array();
            for (const T item : value) {
                result.push_back(item);
            }
            return result;
        }

        Json semantic_version_json(const SemanticVersion& value) {
            return Json{
                {"major", value.major},
                {"minor", value.minor},
                {"patch", value.patch},
            };
        }

        lfs::Result<SemanticVersion> parse_semantic_version(
            const Json& value, const std::string_view chapter,
            const std::string_view field) {
            if (auto valid = require_object(value, chapter, field); !valid) {
                return std::move(valid).error();
            }
            auto major = required<std::uint16_t>(value, "major", chapter, field);
            auto minor = required<std::uint16_t>(value, "minor", chapter, field);
            auto patch = required<std::uint16_t>(value, "patch", chapter, field);
            if (!major) {
                return std::move(major).error();
            }
            if (!minor) {
                return std::move(minor).error();
            }
            if (!patch) {
                return std::move(patch).error();
            }
            return SemanticVersion{*major, *minor, *patch};
        }

        std::string_view origin_provenance_name(const WorldOriginProvenance value) {
            switch (value) {
            case WorldOriginProvenance::None:
                return "none";
            case WorldOriginProvenance::CentralizeByPointCloud:
                return "centralize_by_pointcloud";
            case WorldOriginProvenance::CentralizeByCameras:
                return "centralize_by_cameras";
            case WorldOriginProvenance::User:
                return "user";
            case WorldOriginProvenance::Import:
                return "import";
            }
            return {};
        }

        std::optional<WorldOriginProvenance> parse_origin_provenance(
            const std::string_view value) {
            if (value == "none") {
                return WorldOriginProvenance::None;
            }
            if (value == "centralize_by_pointcloud") {
                return WorldOriginProvenance::CentralizeByPointCloud;
            }
            if (value == "centralize_by_cameras") {
                return WorldOriginProvenance::CentralizeByCameras;
            }
            if (value == "user") {
                return WorldOriginProvenance::User;
            }
            if (value == "import") {
                return WorldOriginProvenance::Import;
            }
            return std::nullopt;
        }

        std::string_view locator_base_name(const LocatorBase value) {
            switch (value) {
            case LocatorBase::Project:
                return "project";
            case LocatorBase::Dataset:
                return "dataset";
            case LocatorBase::Absolute:
                return "absolute";
            case LocatorBase::SearchRoot:
                return "search_root";
            }
            return {};
        }

        std::optional<LocatorBase> parse_locator_base(const std::string_view value) {
            if (value == "project") {
                return LocatorBase::Project;
            }
            if (value == "dataset") {
                return LocatorBase::Dataset;
            }
            if (value == "absolute") {
                return LocatorBase::Absolute;
            }
            if (value == "search_root") {
                return LocatorBase::SearchRoot;
            }
            return std::nullopt;
        }

        Json locator_json(const ReferenceLocator& value) {
            Json result{
                {"preferred", value.preferred},
                {"base", locator_base_name(value.base)},
            };
            if (value.absolute_fallback) {
                result["absolute_fallback"] = *value.absolute_fallback;
            }
            return result;
        }

        lfs::Result<ReferenceLocator> parse_locator(const Json& value,
                                                    const std::string_view chapter,
                                                    const std::string_view field) {
            if (auto valid = require_object(value, chapter, field); !valid) {
                return std::move(valid).error();
            }
            auto preferred = required<std::string>(value, "preferred", chapter, field);
            auto base = required<std::string>(value, "base", chapter, field);
            auto fallback =
                optional<std::string>(value, "absolute_fallback", chapter, field);
            if (!preferred) {
                return std::move(preferred).error();
            }
            if (!base) {
                return std::move(base).error();
            }
            if (!fallback) {
                return std::move(fallback).error();
            }
            auto parsed_base = parse_locator_base(*base);
            if (!parsed_base) {
                return fail<ReferenceLocator>(
                    lfs::ErrorCode::DataLoss, "The project contains an invalid locator base.",
                    std::format("{}.{}.base '{}' is unknown", chapter, field, *base),
                    chapter, std::format("{}.base", field));
            }
            if (preferred->empty()) {
                return fail<ReferenceLocator>(
                    lfs::ErrorCode::DataLoss, "The project contains an empty locator.",
                    std::format("{}.{}.preferred must not be empty", chapter, field),
                    chapter, std::format("{}.preferred", field));
            }
            return ReferenceLocator{
                .preferred = std::move(*preferred),
                .base = *parsed_base,
                .absolute_fallback = std::move(*fallback),
            };
        }

        Json fingerprint_json(const ReferenceFingerprint& value) {
            Json result{
                {"kind", value.kind == FingerprintKind::File ? "file" : "directory"},
                {"size", value.size},
                {"mtime_unix_ns", value.mtime_unix_ns},
                {"head_xxh3_128", value.head_xxh3.to_hex()},
                {"tail_xxh3_128", value.tail_xxh3.to_hex()},
            };
            if (value.full_xxh3) {
                result["full_xxh3_128"] = value.full_xxh3->to_hex();
            }
            return result;
        }

        lfs::Result<ReferenceFingerprint> parse_fingerprint(
            const Json& value, const std::string_view chapter,
            const std::string_view field) {
            if (auto valid = require_object(value, chapter, field); !valid) {
                return std::move(valid).error();
            }
            auto kind = required<std::string>(value, "kind", chapter, field);
            auto size = required<std::uint64_t>(value, "size", chapter, field);
            auto mtime = required<std::int64_t>(value, "mtime_unix_ns", chapter, field);
            auto head = required<std::string>(value, "head_xxh3_128", chapter, field);
            auto tail = required<std::string>(value, "tail_xxh3_128", chapter, field);
            auto full = optional<std::string>(value, "full_xxh3_128", chapter, field);
            if (!kind) {
                return std::move(kind).error();
            }
            if (!size) {
                return std::move(size).error();
            }
            if (!mtime) {
                return std::move(mtime).error();
            }
            if (!head) {
                return std::move(head).error();
            }
            if (!tail) {
                return std::move(tail).error();
            }
            if (!full) {
                return std::move(full).error();
            }
            if (*kind != "file" && *kind != "directory") {
                return fail<ReferenceFingerprint>(
                    lfs::ErrorCode::DataLoss,
                    "The project contains an invalid fingerprint kind.",
                    std::format("{}.{}.kind '{}' is unknown", chapter, field, *kind),
                    chapter, std::format("{}.kind", field));
            }
            auto parsed_head = Hash128::from_hex(*head);
            auto parsed_tail = Hash128::from_hex(*tail);
            std::optional<Hash128> parsed_full;
            if (*full) {
                parsed_full = Hash128::from_hex(**full);
            }
            if (!parsed_head || !parsed_tail || (*full && !parsed_full)) {
                return fail<ReferenceFingerprint>(
                    lfs::ErrorCode::DataLoss, "The project contains an invalid fingerprint hash.",
                    std::format("{}.{} contains a non-canonical 128-bit hash", chapter, field),
                    chapter, field);
            }
            return ReferenceFingerprint{
                .kind = *kind == "file" ? FingerprintKind::File
                                        : FingerprintKind::Directory,
                .size = *size,
                .mtime_unix_ns = *mtime,
                .head_xxh3 = *parsed_head,
                .tail_xxh3 = *parsed_tail,
                .full_xxh3 = parsed_full,
            };
        }

        Json merge_known(Json existing, const Json& known) {
            if (!existing.is_object() || !known.is_object()) {
                return known;
            }
            for (auto it = known.begin(); it != known.end(); ++it) {
                const auto current = existing.find(it.key());
                if (current != existing.end() && current->is_object() &&
                    it->is_object()) {
                    *current = merge_known(*current, *it);
                } else {
                    existing[it.key()] = *it;
                }
            }
            return existing;
        }

        void initialize_json_chapter(JsonChapterDom& dom,
                                     const std::string_view array_name = {}) {
            const auto schema = dom.set("schema_version", JSON_CHAPTER_SCHEMA_VERSION);
            (void)schema;
            if (!array_name.empty()) {
                const auto array = dom.set_json(array_name, Json::array());
                (void)array;
            }
        }

        std::int64_t file_time_ns(const std::filesystem::file_time_type value) {
#if defined(_MSC_VER)
            const auto system_time =
                std::chrono::clock_cast<std::chrono::system_clock>(value);
#else
            const auto system_time = std::filesystem::file_time_type::clock::to_sys(value);
#endif
            constexpr std::int64_t NS_PER_SECOND = 1'000'000'000;
            constexpr auto MIN_NS = std::numeric_limits<std::int64_t>::min();
            constexpr auto MAX_NS = std::numeric_limits<std::int64_t>::max();
            const auto elapsed = system_time.time_since_epoch();
            using Elapsed = std::remove_cv_t<decltype(elapsed)>;
            const auto seconds =
                std::chrono::duration_cast<std::chrono::seconds>(elapsed);
            const auto remainder =
                elapsed - std::chrono::duration_cast<Elapsed>(seconds);
            const auto remainder_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(remainder)
                    .count();
            const auto seconds_count = seconds.count();
            if (seconds_count > MAX_NS / NS_PER_SECOND ||
                (seconds_count == MAX_NS / NS_PER_SECOND &&
                 remainder_ns > MAX_NS % NS_PER_SECOND)) {
                return MAX_NS;
            }
            if (seconds_count < MIN_NS / NS_PER_SECOND ||
                (seconds_count == MIN_NS / NS_PER_SECOND &&
                 remainder_ns < MIN_NS % NS_PER_SECOND)) {
                return MIN_NS;
            }
            return seconds_count * NS_PER_SECOND + remainder_ns;
        }

        lfs::Result<std::vector<std::byte>> read_file_window(
            const std::filesystem::path& path, const std::uint64_t offset,
            const std::size_t bytes) {
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                return fail<std::vector<std::byte>>(
                    lfs::ErrorCode::NotFound, "The referenced file could not be opened.",
                    std::format("Cannot open '{}'", lfs::core::path_to_utf8(path)),
                    "REFS", "fingerprint");
            }
            if (offset > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamoff>::max())) {
                return fail<std::vector<std::byte>>(
                    lfs::ErrorCode::BoundsViolation,
                    "The referenced file is too large for this build.",
                    std::format("File window offset {} is not representable", offset),
                    "REFS", "fingerprint");
            }
            stream.seekg(static_cast<std::streamoff>(offset));
            if (!stream) {
                return fail<std::vector<std::byte>>(
                    lfs::ErrorCode::Unavailable, "The referenced file could not be read.",
                    std::format("Cannot seek '{}' to byte {}", lfs::core::path_to_utf8(path),
                                offset),
                    "REFS", "fingerprint");
            }
            std::vector<std::byte> result(bytes);
            if (bytes != 0) {
                stream.read(reinterpret_cast<char*>(result.data()),
                            static_cast<std::streamsize>(bytes));
                if (stream.gcount() != static_cast<std::streamsize>(bytes)) {
                    return fail<std::vector<std::byte>>(
                        lfs::ErrorCode::Unavailable, "The referenced file could not be read.",
                        std::format("Short read from '{}': expected {}, got {}",
                                    lfs::core::path_to_utf8(path), bytes, stream.gcount()),
                        "REFS", "fingerprint");
                }
            }
            return result;
        }

        Hash128 hash128_from_xxh(const XXH128_hash_t hash) {
            Hash128 result;
            for (std::size_t i = 0; i < 8; ++i) {
                result.bytes[i] =
                    static_cast<std::uint8_t>((hash.low64 >> (i * 8)) & 0xff);
                result.bytes[8 + i] =
                    static_cast<std::uint8_t>((hash.high64 >> (i * 8)) & 0xff);
            }
            return result;
        }

        lfs::Result<Hash128>
        hash_file_streaming(const std::filesystem::path& path) {
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                return fail<Hash128>(
                    lfs::ErrorCode::PermissionDenied,
                    "The referenced file could not be opened.",
                    std::format("Cannot open '{}' for full fingerprinting",
                                lfs::core::path_to_utf8(path)),
                    "REFS", "fingerprint.full_xxh3_128");
            }
            using StatePtr =
                std::unique_ptr<XXH3_state_t, decltype(&XXH3_freeState)>;
            StatePtr state(XXH3_createState(), &XXH3_freeState);
            if (!state || XXH3_128bits_reset(state.get()) == XXH_ERROR) {
                return fail<Hash128>(
                    lfs::ErrorCode::ResourceExhausted,
                    "The full reference fingerprint could not be initialized.",
                    "XXH3 state allocation/reset failed", "REFS",
                    "fingerprint.full_xxh3_128");
            }
            std::vector<char> buffer(FULL_HASH_STREAM_BYTES);
            while (stream) {
                stream.read(buffer.data(),
                            static_cast<std::streamsize>(buffer.size()));
                const auto count = stream.gcount();
                if (count > 0 &&
                    XXH3_128bits_update(
                        state.get(), buffer.data(),
                        static_cast<std::size_t>(count)) == XXH_ERROR) {
                    return fail<Hash128>(
                        lfs::ErrorCode::Internal,
                        "The full reference fingerprint could not be computed.",
                        "XXH3 streaming update failed", "REFS",
                        "fingerprint.full_xxh3_128");
                }
            }
            if (!stream.eof()) {
                return fail<Hash128>(
                    lfs::ErrorCode::Unavailable,
                    "The referenced file could not be read completely.",
                    std::format("I/O failed while fingerprinting '{}'",
                                lfs::core::path_to_utf8(path)),
                    "REFS", "fingerprint.full_xxh3_128");
            }
            return hash128_from_xxh(XXH3_128bits_digest(state.get()));
        }

        lfs::Result<std::vector<std::byte>> directory_inventory(
            const std::filesystem::path& root, std::uint64_t& total_size) {
            struct Entry {
                std::string path;
                std::uint64_t size;
                char kind;
            };
            std::vector<Entry> entries;
            std::error_code error;
            std::filesystem::recursive_directory_iterator iterator(
                root, std::filesystem::directory_options::skip_permission_denied, error);
            if (error) {
                return fail<std::vector<std::byte>>(
                    lfs::ErrorCode::Unavailable,
                    "The referenced directory could not be enumerated.",
                    std::format("Cannot enumerate '{}': {}", lfs::core::path_to_utf8(root),
                                error.message()),
                    "REFS", "fingerprint");
            }
            for (const auto end = std::filesystem::recursive_directory_iterator{};
                 iterator != end; iterator.increment(error)) {
                if (error) {
                    return fail<std::vector<std::byte>>(
                        lfs::ErrorCode::Unavailable,
                        "The referenced directory could not be enumerated.",
                        std::format("Cannot enumerate '{}': {}",
                                    lfs::core::path_to_utf8(root), error.message()),
                        "REFS", "fingerprint");
                }
                if (entries.size() == MAX_DIRECTORY_ENTRIES) {
                    return fail<std::vector<std::byte>>(
                        lfs::ErrorCode::ResourceExhausted,
                        "The referenced directory is too large to fingerprint.",
                        std::format("'{}' contains more than {} entries",
                                    lfs::core::path_to_utf8(root), MAX_DIRECTORY_ENTRIES),
                        "REFS", "fingerprint");
                }
                const auto relative = std::filesystem::relative(iterator->path(), root, error);
                if (error) {
                    return fail<std::vector<std::byte>>(
                        lfs::ErrorCode::Unavailable,
                        "A referenced directory entry could not be identified.",
                        std::format("Cannot make '{}' relative to '{}': {}",
                                    lfs::core::path_to_utf8(iterator->path()),
                                    lfs::core::path_to_utf8(root), error.message()),
                        "REFS", "fingerprint");
                }
                const auto status = iterator->symlink_status(error);
                if (error) {
                    return fail<std::vector<std::byte>>(
                        lfs::ErrorCode::Unavailable,
                        "A referenced directory entry could not be inspected.",
                        std::format("Cannot stat '{}': {}",
                                    lfs::core::path_to_utf8(iterator->path()), error.message()),
                        "REFS", "fingerprint");
                }
                Entry entry{
                    .path = lfs::core::path_to_utf8(relative.generic_string()),
                    .size = 0,
                    .kind = std::filesystem::is_directory(status)
                                ? 'd'
                                : (std::filesystem::is_regular_file(status) ? 'f' : 'o'),
                };
                if (entry.kind == 'f') {
                    entry.size = iterator->file_size(error);
                    if (error || entry.size >
                                     std::numeric_limits<std::uint64_t>::max() - total_size) {
                        return fail<std::vector<std::byte>>(
                            lfs::ErrorCode::BoundsViolation,
                            "The referenced directory size cannot be represented.",
                            std::format("Cannot account for '{}'",
                                        lfs::core::path_to_utf8(iterator->path())),
                            "REFS", "fingerprint");
                    }
                    total_size += entry.size;
                }
                entries.push_back(std::move(entry));
            }
            std::ranges::sort(entries, {}, &Entry::path);

            std::vector<std::byte> inventory;
            for (const Entry& entry : entries) {
                const std::string row =
                    std::format("{}\t{}\t{}\n", entry.kind, entry.size, entry.path);
                if (row.size() > MAX_DIRECTORY_INVENTORY_BYTES - inventory.size()) {
                    return fail<std::vector<std::byte>>(
                        lfs::ErrorCode::ResourceExhausted,
                        "The referenced directory inventory is too large.",
                        std::format("'{}' exceeds {} inventory bytes",
                                    lfs::core::path_to_utf8(root),
                                    MAX_DIRECTORY_INVENTORY_BYTES),
                        "REFS", "fingerprint");
                }
                const auto row_bytes =
                    std::as_bytes(std::span(row.data(), row.size()));
                inventory.insert(inventory.end(), row_bytes.begin(), row_bytes.end());
            }
            return inventory;
        }

        lfs::Result<void> ensure_schema(const JsonChapterDom& dom,
                                        const std::string_view chapter) {
            const auto schema = dom.get<std::uint32_t>("schema_version");
            if (!schema) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "The project chapter has no valid schema version.",
                    std::format("{} schema_version is missing or invalid", chapter),
                    chapter, "schema_version");
            }
            if (*schema != JSON_CHAPTER_SCHEMA_VERSION) {
                return fail<void>(
                    lfs::ErrorCode::Unsupported,
                    "This project chapter version is not supported.",
                    std::format("{} schema version {} is not supported (expected {})",
                                chapter, *schema, JSON_CHAPTER_SCHEMA_VERSION),
                    chapter, "schema_version");
            }
            return {};
        }

    } // namespace

    std::string Hash128::to_hex() const {
        std::string result;
        result.reserve(32);
        constexpr std::string_view HEX = "0123456789abcdef";
        for (const std::uint8_t byte : bytes) {
            result.push_back(HEX[byte >> 4]);
            result.push_back(HEX[byte & 0x0f]);
        }
        return result;
    }

    std::optional<Hash128> Hash128::from_hex(const std::string_view text) {
        if (text.size() != 32) {
            return std::nullopt;
        }
        const auto nibble = [](const char value) -> std::optional<std::uint8_t> {
            if (value >= '0' && value <= '9') {
                return static_cast<std::uint8_t>(value - '0');
            }
            if (value >= 'a' && value <= 'f') {
                return static_cast<std::uint8_t>(10 + value - 'a');
            }
            return std::nullopt;
        };
        Hash128 result;
        for (std::size_t i = 0; i < result.bytes.size(); ++i) {
            const auto high = nibble(text[i * 2]);
            const auto low = nibble(text[i * 2 + 1]);
            if (!high || !low) {
                return std::nullopt;
            }
            result.bytes[i] = static_cast<std::uint8_t>((*high << 4) | *low);
        }
        return result;
    }

    Hash128 xxh3_128(const std::span<const std::byte> bytes) {
        return hash128_from_xxh(XXH3_128bits(bytes.data(), bytes.size()));
    }

    struct Hash128Stream::Impl {
        using StatePtr = std::unique_ptr<XXH3_state_t, decltype(&XXH3_freeState)>;
        StatePtr state{nullptr, &XXH3_freeState};
    };

    Hash128Stream::Hash128Stream() : impl_(std::make_unique<Impl>()) {
        impl_->state.reset(XXH3_createState());
        if (!impl_->state || XXH3_128bits_reset(impl_->state.get()) == XXH_ERROR) {
            impl_->state.reset();
        }
    }

    Hash128Stream::Hash128Stream(Hash128Stream&&) noexcept = default;
    Hash128Stream& Hash128Stream::operator=(Hash128Stream&&) noexcept = default;
    Hash128Stream::~Hash128Stream() = default;

    bool Hash128Stream::valid() const noexcept {
        return impl_ && impl_->state;
    }

    bool Hash128Stream::update(const std::span<const std::byte> bytes) noexcept {
        if (!valid()) {
            return false;
        }
        if (bytes.empty()) {
            return true;
        }
        return XXH3_128bits_update(impl_->state.get(), bytes.data(),
                                   bytes.size()) != XXH_ERROR;
    }

    Hash128 Hash128Stream::digest() const noexcept {
        if (!valid()) {
            return {};
        }
        return hash128_from_xxh(XXH3_128bits_digest(impl_->state.get()));
    }

    lfs::Result<ReferenceFingerprint> fingerprint_path(
        const std::filesystem::path& path, const bool include_full_hash) {
        std::error_code error;
        const auto status = std::filesystem::status(path, error);
        if (error || !std::filesystem::exists(status)) {
            return fail<ReferenceFingerprint>(
                lfs::ErrorCode::NotFound, "The referenced path is missing.",
                std::format("'{}' does not exist", lfs::core::path_to_utf8(path)),
                "REFS", "fingerprint");
        }
        const auto mtime = std::filesystem::last_write_time(path, error);
        if (error) {
            return fail<ReferenceFingerprint>(
                lfs::ErrorCode::Unavailable, "The referenced path could not be inspected.",
                std::format("Cannot read mtime for '{}': {}",
                            lfs::core::path_to_utf8(path), error.message()),
                "REFS", "fingerprint");
        }

        ReferenceFingerprint result;
        result.mtime_unix_ns = file_time_ns(mtime);
        if (std::filesystem::is_regular_file(status)) {
            result.kind = FingerprintKind::File;
            result.size = std::filesystem::file_size(path, error);
            if (error) {
                return fail<ReferenceFingerprint>(
                    lfs::ErrorCode::Unavailable,
                    "The referenced file size could not be read.",
                    std::format("Cannot read size for '{}': {}",
                                lfs::core::path_to_utf8(path), error.message()),
                    "REFS", "fingerprint");
            }
            const auto window = static_cast<std::size_t>(
                std::min<std::uint64_t>(result.size, FINGERPRINT_WINDOW_BYTES));
            auto head = read_file_window(path, 0, window);
            if (!head) {
                return std::move(head).error();
            }
            auto tail = read_file_window(path, result.size - window, window);
            if (!tail) {
                return std::move(tail).error();
            }
            result.head_xxh3 = xxh3_128(*head);
            result.tail_xxh3 = xxh3_128(*tail);
            if (include_full_hash) {
                auto full = hash_file_streaming(path);
                if (!full) {
                    return std::move(full).error();
                }
                result.full_xxh3 = *full;
            }
            return result;
        }
        if (!std::filesystem::is_directory(status)) {
            return fail<ReferenceFingerprint>(
                lfs::ErrorCode::InvalidArgument,
                "The referenced path type is not supported.",
                std::format("'{}' is neither a regular file nor a directory",
                            lfs::core::path_to_utf8(path)),
                "REFS", "fingerprint");
        }

        result.kind = FingerprintKind::Directory;
        auto inventory = directory_inventory(path, result.size);
        if (!inventory) {
            return std::move(inventory).error();
        }
        const std::size_t window =
            std::min(inventory->size(), FINGERPRINT_WINDOW_BYTES);
        result.head_xxh3 = xxh3_128(
            std::span<const std::byte>(*inventory).first(window));
        result.tail_xxh3 = xxh3_128(
            std::span<const std::byte>(*inventory).last(window));
        if (include_full_hash) {
            result.full_xxh3 = xxh3_128(*inventory);
        }
        return result;
    }

    lfs::Result<FingerprintCheck> check_fingerprint(
        const std::filesystem::path& path,
        const ReferenceFingerprint& expected) {
        std::error_code error;
        const auto status = std::filesystem::status(path, error);
        if (error || !std::filesystem::exists(status)) {
            return FingerprintCheck{
                .disposition = FingerprintDisposition::Missing,
                .observed = std::nullopt,
                .diagnostic = std::format("'{}' is missing",
                                          lfs::core::path_to_utf8(path)),
            };
        }
        const FingerprintKind observed_kind =
            std::filesystem::is_regular_file(status) ? FingerprintKind::File
                                                     : FingerprintKind::Directory;
        if ((!std::filesystem::is_regular_file(status) &&
             !std::filesystem::is_directory(status)) ||
            observed_kind != expected.kind) {
            return FingerprintCheck{
                .disposition = FingerprintDisposition::TypeMismatch,
                .observed = std::nullopt,
                .diagnostic = "reference type changed",
            };
        }

        const auto mtime = std::filesystem::last_write_time(path, error);
        if (error) {
            return fail<FingerprintCheck>(
                lfs::ErrorCode::Unavailable,
                "The referenced path could not be inspected.",
                std::format("Cannot read mtime for '{}': {}",
                            lfs::core::path_to_utf8(path), error.message()),
                "REFS", "fingerprint");
        }
        const std::int64_t observed_mtime = file_time_ns(mtime);
        if (observed_kind == FingerprintKind::File) {
            const std::uint64_t observed_size =
                std::filesystem::file_size(path, error);
            if (error) {
                return fail<FingerprintCheck>(
                    lfs::ErrorCode::Unavailable,
                    "The referenced file size could not be read.",
                    std::format("Cannot read size for '{}': {}",
                                lfs::core::path_to_utf8(path),
                                error.message()),
                    "REFS", "fingerprint");
            }
            // Size and mtime are cache hints. A copied/relinked file can carry
            // stale metadata while retaining the exact content fingerprint,
            // so only the all-metadata-match case may take the no-I/O path.
            if (observed_size == expected.size &&
                observed_mtime == expected.mtime_unix_ns) {
                ReferenceFingerprint observed = expected;
                observed.mtime_unix_ns = observed_mtime;
                return FingerprintCheck{
                    .disposition = FingerprintDisposition::MatchFastPath,
                    .observed = observed,
                    .diagnostic = "mtime and size match",
                };
            }
        }
        if (observed_kind != FingerprintKind::File &&
            observed_mtime == expected.mtime_unix_ns) {
            ReferenceFingerprint observed = expected;
            observed.mtime_unix_ns = observed_mtime;
            return FingerprintCheck{
                .disposition = FingerprintDisposition::MatchFastPath,
                .observed = observed,
                .diagnostic =
                    observed_kind == FingerprintKind::File
                        ? "mtime and size match"
                        : "directory mtime matches",
            };
        }

        auto observed = fingerprint_path(path, expected.full_xxh3.has_value());
        if (!observed) {
            return std::move(observed).error();
        }
        if (observed->head_xxh3 != expected.head_xxh3 ||
            observed->tail_xxh3 != expected.tail_xxh3 ||
            (expected.full_xxh3 &&
             observed->full_xxh3 != expected.full_xxh3)) {
            return FingerprintCheck{
                .disposition = FingerprintDisposition::ContentMismatch,
                .observed = *observed,
                .diagnostic = "xxh3 fingerprint changed",
            };
        }
        return FingerprintCheck{
            .disposition =
                observed->mtime_unix_ns == expected.mtime_unix_ns &&
                        observed->size == expected.size
                    ? FingerprintDisposition::MatchFastPath
                    : FingerprintDisposition::MatchMtimeRefreshed,
            .observed = *observed,
            .diagnostic =
                observed->mtime_unix_ns == expected.mtime_unix_ns &&
                        observed->size == expected.size
                    ? "mtime, size, and content fingerprint match"
                    : "metadata changed; content fingerprint matches",
        };
    }

    ProjectChapter::ProjectChapter() {
        initialize_json_chapter(dom_);
        const auto decisions = dom_.set_json("embed_decisions", Json::array());
        const auto provenance = dom_.set_json("provenance", Json::array());
        const auto embedded = dom_.set_json("embedded_payloads", Json::array());
        (void)decisions;
        (void)provenance;
        (void)embedded;
    }

    ProjectChapter::ProjectChapter(JsonChapterDom dom)
        : dom_(std::move(dom)) {}

    lfs::Result<ProjectChapter> ProjectChapter::parse(const std::string_view bytes) {
        auto dom = JsonChapterDom::parse(bytes);
        if (!dom) {
            return std::move(dom).error();
        }
        if (auto schema = ensure_schema(*dom, "PROJ"); !schema) {
            return std::move(schema).error();
        }
        return ProjectChapter(std::move(*dom));
    }

    lfs::Result<ProjectChapter> ProjectChapter::from_bytes(
        const std::span<const std::byte> bytes) {
        return parse(std::string_view(
            reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    }

    lfs::Result<ProjectManifest> ProjectChapter::manifest() const {
        const auto value = dom_.get_json("manifest");
        if (!value) {
            return fail<ProjectManifest>(
                lfs::ErrorCode::DataLoss, "The project manifest is missing.",
                "PROJ.manifest is missing", "PROJ", "manifest");
        }
        if (auto valid = require_object(*value, "PROJ", "manifest"); !valid) {
            return std::move(valid).error();
        }
        auto app_name =
            required<std::string>(*value, "application_name", "PROJ", "manifest");
        if (!app_name) {
            return std::move(app_name).error();
        }
        const auto parse_version_member =
            [&](const std::string_view key) -> lfs::Result<SemanticVersion> {
            const auto found = value->find(std::string(key));
            if (found == value->end()) {
                return fail<SemanticVersion>(
                    lfs::ErrorCode::DataLoss,
                    "The project manifest is missing a version.",
                    std::format("PROJ.manifest.{} is missing", key), "PROJ",
                    std::format("manifest.{}", key));
            }
            return parse_semantic_version(
                *found, "PROJ", std::format("manifest.{}", key));
        };
        auto application = parse_version_member("application_version");
        auto schema = parse_version_member("schema_version");
        auto reader = parse_version_member("minimum_reader_version");
        auto writer = parse_version_member("minimum_safe_writer_version");
        auto required_caps = required<std::vector<std::string>>(
            *value, "required_capabilities", "PROJ", "manifest");
        auto optional_caps = required<std::vector<std::string>>(
            *value, "optional_capabilities", "PROJ", "manifest");
        if (!application) {
            return std::move(application).error();
        }
        if (!schema) {
            return std::move(schema).error();
        }
        if (!reader) {
            return std::move(reader).error();
        }
        if (!writer) {
            return std::move(writer).error();
        }
        if (!required_caps) {
            return std::move(required_caps).error();
        }
        if (!optional_caps) {
            return std::move(optional_caps).error();
        }
        const auto has_invalid_capability = [](const std::vector<std::string>& values) {
            return std::ranges::any_of(values, [](const std::string& value) {
                return value.empty();
            });
        };
        if (has_invalid_capability(*required_caps) ||
            has_invalid_capability(*optional_caps)) {
            return fail<ProjectManifest>(
                lfs::ErrorCode::DataLoss,
                "The project manifest contains an empty capability name.",
                "PROJ manifest capability names must be non-empty", "PROJ",
                "manifest.capabilities");
        }
        return ProjectManifest{
            .application_name = std::move(*app_name),
            .application_version = *application,
            .schema_version = *schema,
            .minimum_reader_version = *reader,
            .minimum_safe_writer_version = *writer,
            .required_capabilities = std::move(*required_caps),
            .optional_capabilities = std::move(*optional_caps),
        };
    }

    lfs::Result<void> ProjectChapter::set_manifest(const ProjectManifest& value) {
        if (value.application_name.empty() ||
            std::ranges::any_of(value.required_capabilities,
                                [](const std::string& item) { return item.empty(); }) ||
            std::ranges::any_of(value.optional_capabilities,
                                [](const std::string& item) { return item.empty(); })) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The project manifest contains an empty name.",
                "Application and capability names must be non-empty", "PROJ", "manifest");
        }
        Json known{
            {"application_name", value.application_name},
            {"application_version", semantic_version_json(value.application_version)},
            {"schema_version", semantic_version_json(value.schema_version)},
            {"minimum_reader_version",
             semantic_version_json(value.minimum_reader_version)},
            {"minimum_safe_writer_version",
             semantic_version_json(value.minimum_safe_writer_version)},
            {"required_capabilities", value.required_capabilities},
            {"optional_capabilities", value.optional_capabilities},
        };
        return dom_.set_json(
            "manifest", merge_known(dom_.get_json("manifest").value_or(Json::object()),
                                    known));
    }

    lfs::Result<lfs::core::Uuid> ProjectChapter::project_uuid() const {
        const auto value = dom_.get_json("project_uuid");
        if (!value) {
            return fail<lfs::core::Uuid>(
                lfs::ErrorCode::DataLoss, "The project UUID is missing.",
                "PROJ.project_uuid is missing", "PROJ", "project_uuid");
        }
        return parse_uuid(*value, "PROJ", "project_uuid");
    }

    lfs::Result<void> ProjectChapter::set_project_uuid(
        const lfs::core::Uuid& value) {
        if (value.is_nil()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument, "The project UUID cannot be null.",
                "PROJ.project_uuid must be non-null", "PROJ", "project_uuid");
        }
        return dom_.set("project_uuid", value.to_string());
    }

    lfs::Result<std::uint64_t> ProjectChapter::created_at_unix_ns() const {
        const auto value = dom_.get<std::uint64_t>("created_at_unix_ns");
        if (!value) {
            return fail<std::uint64_t>(
                lfs::ErrorCode::DataLoss,
                "The project creation timestamp is missing.",
                "PROJ.created_at_unix_ns is missing or invalid", "PROJ",
                "created_at_unix_ns");
        }
        return *value;
    }

    lfs::Result<void> ProjectChapter::set_created_at_unix_ns(
        const std::uint64_t value) {
        return dom_.set("created_at_unix_ns", value);
    }

    lfs::Result<std::uint64_t> ProjectChapter::modified_at_unix_ns() const {
        const auto value = dom_.get<std::uint64_t>("modified_at_unix_ns");
        if (!value) {
            return fail<std::uint64_t>(
                lfs::ErrorCode::DataLoss,
                "The project modification timestamp is missing.",
                "PROJ.modified_at_unix_ns is missing or invalid", "PROJ",
                "modified_at_unix_ns");
        }
        return *value;
    }

    lfs::Result<void> ProjectChapter::set_modified_at_unix_ns(
        const std::uint64_t value) {
        return dom_.set("modified_at_unix_ns", value);
    }

    lfs::Result<std::optional<lfs::core::Uuid>>
    ProjectChapter::dataset_reference() const {
        const auto root = dom_.get_json("dataset_reference_uuid");
        if (!root || root->is_null()) {
            return std::optional<lfs::core::Uuid>{};
        }
        auto parsed = parse_uuid(*root, "PROJ", "dataset_reference_uuid");
        if (!parsed) {
            return std::move(parsed).error();
        }
        return std::optional<lfs::core::Uuid>(*parsed);
    }

    lfs::Result<void> ProjectChapter::set_dataset_reference(
        const std::optional<lfs::core::Uuid> value) {
        if (value && value->is_nil()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The dataset reference UUID cannot be null.",
                "PROJ.dataset_reference_uuid must be non-null when present", "PROJ",
                "dataset_reference_uuid");
        }
        return value ? dom_.set("dataset_reference_uuid", value->to_string())
                     : dom_.set_json("dataset_reference_uuid", nullptr);
    }

    lfs::Result<std::vector<lfs::core::Uuid>>
    ProjectChapter::project_lineage() const {
        const auto value = dom_.get_json("project_lineage");
        if (!value) {
            return std::vector<lfs::core::Uuid>{};
        }
        if (!value->is_array()) {
            return fail<std::vector<lfs::core::Uuid>>(
                lfs::ErrorCode::DataLoss, "The project lineage is invalid.",
                "PROJ.project_lineage must be an array", "PROJ", "project_lineage");
        }
        std::vector<lfs::core::Uuid> result;
        result.reserve(value->size());
        for (std::size_t i = 0; i < value->size(); ++i) {
            auto parsed =
                parse_uuid((*value)[i], "PROJ", std::format("project_lineage[{}]", i));
            if (!parsed) {
                return std::move(parsed).error();
            }
            if (std::ranges::find(result, *parsed) != result.end()) {
                return fail<std::vector<lfs::core::Uuid>>(
                    lfs::ErrorCode::DataLoss,
                    "The project lineage contains a duplicate UUID.",
                    std::format("PROJ.project_lineage[{}] is duplicated", i), "PROJ",
                    "project_lineage");
            }
            result.push_back(*parsed);
        }
        return result;
    }

    lfs::Result<void> ProjectChapter::set_project_lineage(
        const std::span<const lfs::core::Uuid> value) {
        Json result = Json::array();
        std::unordered_set<lfs::core::Uuid> seen;
        for (const lfs::core::Uuid& uuid : value) {
            if (uuid.is_nil() || !seen.insert(uuid).second) {
                return fail<void>(
                    lfs::ErrorCode::InvalidArgument,
                    "The project lineage contains an invalid UUID.",
                    "PROJ.project_lineage must contain unique non-null UUIDs", "PROJ",
                    "project_lineage");
            }
            result.push_back(uuid.to_string());
        }
        return dom_.set_json("project_lineage", std::move(result));
    }

    lfs::Result<ProjectGeoreference> ProjectChapter::georeference() const {
        const auto value = dom_.get_json("georeference");
        if (!value) {
            return ProjectGeoreference{};
        }
        if (auto valid = require_object(*value, "PROJ", "georeference"); !valid) {
            return std::move(valid).error();
        }
        auto crs = optional<std::string>(*value, "crs", "PROJ", "georeference");
        const auto origin_member = value->find("world_origin");
        if (origin_member == value->end()) {
            return fail<ProjectGeoreference>(
                lfs::ErrorCode::DataLoss, "The georeference world origin is missing.",
                "PROJ.georeference.world_origin is missing", "PROJ",
                "georeference.world_origin");
        }
        auto origin = fixed_array<double, 3>(
            *origin_member, "PROJ", "georeference.world_origin");
        auto scale =
            required<double>(*value, "world_unit_scale", "PROJ", "georeference");
        auto provenance = required<std::string>(
            *value, "world_origin_provenance", "PROJ", "georeference");
        if (!crs) {
            return std::move(crs).error();
        }
        if (!origin) {
            return std::move(origin).error();
        }
        if (!scale) {
            return std::move(scale).error();
        }
        if (!provenance) {
            return std::move(provenance).error();
        }
        const auto parsed_provenance = parse_origin_provenance(*provenance);
        if (!parsed_provenance || !std::isfinite(*scale) || *scale <= 0.0) {
            return fail<ProjectGeoreference>(
                lfs::ErrorCode::DataLoss, "The georeference block is invalid.",
                std::format("Invalid scale {} or provenance '{}'", *scale, *provenance),
                "PROJ", "georeference");
        }
        return ProjectGeoreference{
            .crs = std::move(*crs),
            .world_origin = *origin,
            .world_unit_scale = *scale,
            .world_origin_provenance = *parsed_provenance,
        };
    }

    lfs::Result<void> ProjectChapter::set_georeference(
        const ProjectGeoreference& value) {
        if (!std::isfinite(value.world_unit_scale) ||
            value.world_unit_scale <= 0.0 ||
            std::ranges::any_of(value.world_origin,
                                [](const double item) { return !std::isfinite(item); })) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The georeference block contains an invalid numeric value.",
                "world_unit_scale must be positive and all values must be finite",
                "PROJ", "georeference");
        }
        Json known{
            {"world_origin", json_array(value.world_origin)},
            {"world_unit_scale", value.world_unit_scale},
            {"world_origin_provenance",
             origin_provenance_name(value.world_origin_provenance)},
        };
        if (value.crs) {
            known["crs"] = *value.crs;
        }
        Json merged = merge_known(
            dom_.get_json("georeference").value_or(Json::object()), known);
        if (!value.crs) {
            merged.erase("crs");
        }
        return dom_.set_json("georeference", std::move(merged));
    }

    lfs::Result<std::vector<EmbedDecision>> ProjectChapter::embed_decisions() const {
        auto items = dom_.array_items("embed_decisions");
        if (!items) {
            return std::move(items).error();
        }
        std::vector<EmbedDecision> result;
        result.reserve(items->size());
        for (const auto& [id, element] : *items) {
            auto uuid = lfs::core::Uuid::from_string(id);
            auto node_text = JsonChapterDom::read<std::string>(element, "node_uuid");
            auto node = node_text ? lfs::core::Uuid::from_string(*node_text) : std::nullopt;
            auto fourcc = JsonChapterDom::read<std::string>(element, "payload_fourcc");
            auto decision = JsonChapterDom::read<std::string>(element, "decision");
            auto reason = JsonChapterDom::read<std::string>(element, "reason");
            auto ref_text = JsonChapterDom::read<std::string>(element, "reference_uuid");
            std::optional<lfs::core::Uuid> reference;
            if (ref_text) {
                reference = lfs::core::Uuid::from_string(*ref_text);
            }
            if (!uuid || !node || !fourcc || fourcc->size() != 4 || !decision ||
                (*decision != "embedded" && *decision != "external") || !reason ||
                (ref_text && !reference)) {
                return fail<std::vector<EmbedDecision>>(
                    lfs::ErrorCode::DataLoss,
                    "The project contains an invalid embed decision.",
                    std::format("PROJ.embed_decisions UUID {} is incomplete or invalid", id),
                    "PROJ", "embed_decisions");
            }
            result.push_back(EmbedDecision{
                .uuid = *uuid,
                .node_uuid = *node,
                .payload_fourcc = std::move(*fourcc),
                .decision = std::move(*decision),
                .reference_uuid = reference,
                .reason = std::move(*reason),
            });
        }
        return result;
    }

    lfs::Result<void> ProjectChapter::upsert_embed_decision(
        const EmbedDecision& value) {
        if (value.uuid.is_nil() || value.node_uuid.is_nil() ||
            value.payload_fourcc.size() != 4 ||
            (value.decision != "embedded" && value.decision != "external") ||
            (value.reference_uuid && value.reference_uuid->is_nil())) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument, "The embed decision is invalid.",
                "UUIDs, fourcc, decision, and reference must be valid", "PROJ",
                "embed_decisions");
        }
        auto element =
            dom_.array_upsert("embed_decisions", value.uuid.to_string());
        if (!element) {
            return lfs::Result<void>::failure(std::move(element).error());
        }
        if (auto result = element->set("node_uuid", value.node_uuid.to_string()); !result) {
            return result;
        }
        if (auto result = element->set("payload_fourcc", value.payload_fourcc); !result) {
            return result;
        }
        if (auto result = element->set("decision", value.decision); !result) {
            return result;
        }
        if (auto result = element->set("reason", value.reason); !result) {
            return result;
        }
        if (value.reference_uuid) {
            return element->set("reference_uuid", value.reference_uuid->to_string());
        }
        auto removed = element->remove("reference_uuid");
        return removed ? lfs::Result<void>{}
                       : lfs::Result<void>::failure(std::move(removed).error());
    }

    lfs::Result<std::vector<ProvenanceRecord>> ProjectChapter::provenance() const {
        auto items = dom_.array_items("provenance");
        if (!items) {
            return std::move(items).error();
        }
        std::vector<ProvenanceRecord> result;
        result.reserve(items->size());
        for (const auto& [id, element] : *items) {
            auto uuid = lfs::core::Uuid::from_string(id);
            auto kind = JsonChapterDom::read<std::string>(element, "kind");
            auto value = JsonChapterDom::read<std::string>(element, "value");
            if (!uuid || !kind || kind->empty() || !value) {
                return fail<std::vector<ProvenanceRecord>>(
                    lfs::ErrorCode::DataLoss,
                    "The project contains an invalid provenance record.",
                    std::format("PROJ.provenance UUID {} is incomplete", id), "PROJ",
                    "provenance");
            }
            result.push_back({*uuid, std::move(*kind), std::move(*value)});
        }
        return result;
    }

    lfs::Result<void> ProjectChapter::upsert_provenance(
        const ProvenanceRecord& value) {
        if (value.uuid.is_nil() || value.kind.empty()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument, "The provenance record is invalid.",
                "UUID and kind must be non-empty", "PROJ", "provenance");
        }
        auto element = dom_.array_upsert("provenance", value.uuid.to_string());
        if (!element) {
            return lfs::Result<void>::failure(std::move(element).error());
        }
        if (auto result = element->set("kind", value.kind); !result) {
            return result;
        }
        return element->set("value", value.value);
    }

    lfs::Result<std::vector<EmbeddedPayloadProvenance>>
    ProjectChapter::embedded_payload_provenance() const {
        auto items = dom_.array_items("embedded_payloads");
        if (!items) {
            return std::move(items).error();
        }
        std::vector<EmbeddedPayloadProvenance> result;
        result.reserve(items->size());
        for (const auto& [id, element] : *items) {
            auto uuid = lfs::core::Uuid::from_string(id);
            const auto node_value = JsonChapterDom::read_json(element, "node_uuid");
            auto node = node_value ? parse_uuid(*node_value, "PROJ",
                                                "embedded_payloads.node_uuid")
                                   : fail<lfs::core::Uuid>(
                                         lfs::ErrorCode::DataLoss,
                                         "Embedded payload provenance is incomplete.",
                                         "node_uuid is missing", "PROJ",
                                         "embedded_payloads.node_uuid");
            auto fourcc = JsonChapterDom::read<std::string>(element, "fourcc");
            const auto locator_value = JsonChapterDom::read_json(element, "import_locator");
            const auto fingerprint_value =
                JsonChapterDom::read_json(element, "import_fingerprint");
            auto hash_text = JsonChapterDom::read<std::string>(element, "content_xxh3_128");
            if (!uuid || !node || !fourcc || fourcc->size() != 4 ||
                !locator_value || !fingerprint_value || !hash_text) {
                return fail<std::vector<EmbeddedPayloadProvenance>>(
                    lfs::ErrorCode::DataLoss,
                    "Embedded payload provenance is incomplete.",
                    std::format("PROJ.embedded_payloads UUID {} is incomplete", id),
                    "PROJ", "embedded_payloads");
            }
            auto locator = parse_locator(
                *locator_value, "PROJ", "embedded_payloads.import_locator");
            auto fingerprint = parse_fingerprint(
                *fingerprint_value, "PROJ",
                "embedded_payloads.import_fingerprint");
            auto hash = Hash128::from_hex(*hash_text);
            if (!locator) {
                return std::move(locator).error();
            }
            if (!fingerprint) {
                return std::move(fingerprint).error();
            }
            if (!hash) {
                return fail<std::vector<EmbeddedPayloadProvenance>>(
                    lfs::ErrorCode::DataLoss,
                    "Embedded payload provenance has an invalid content hash.",
                    std::format("PROJ.embedded_payloads UUID {} has invalid hash", id),
                    "PROJ", "embedded_payloads.content_xxh3_128");
            }
            result.push_back(EmbeddedPayloadProvenance{
                .uuid = *uuid,
                .node_uuid = *node,
                .fourcc = std::move(*fourcc),
                .import_locator = std::move(*locator),
                .import_fingerprint = std::move(*fingerprint),
                .content_xxh3_128 = *hash,
            });
        }
        return result;
    }

    lfs::Result<void> ProjectChapter::upsert_embedded_payload_provenance(
        const EmbeddedPayloadProvenance& value) {
        if (value.uuid.is_nil() || value.node_uuid.is_nil() ||
            value.fourcc.size() != 4 || value.import_locator.preferred.empty()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "Embedded payload provenance is invalid.",
                "UUIDs, fourcc, and locator must be valid", "PROJ",
                "embedded_payloads");
        }
        auto element =
            dom_.array_upsert("embedded_payloads", value.uuid.to_string());
        if (!element) {
            return lfs::Result<void>::failure(std::move(element).error());
        }
        if (auto result = element->set("node_uuid", value.node_uuid.to_string()); !result) {
            return result;
        }
        if (auto result = element->set("fourcc", value.fourcc); !result) {
            return result;
        }
        if (auto result = element->set_json("import_locator",
                                            locator_json(value.import_locator));
            !result) {
            return result;
        }
        if (auto result = element->set_json(
                "import_fingerprint", fingerprint_json(value.import_fingerprint));
            !result) {
            return result;
        }
        return element->set("content_xxh3_128", value.content_xxh3_128.to_hex());
    }

    ReferencesChapter::ReferencesChapter() {
        initialize_json_chapter(dom_, "references");
    }

    ReferencesChapter::ReferencesChapter(JsonChapterDom dom)
        : dom_(std::move(dom)) {}

    lfs::Result<ReferencesChapter> ReferencesChapter::parse(
        const std::string_view bytes) {
        auto dom = JsonChapterDom::parse(bytes);
        if (!dom) {
            return std::move(dom).error();
        }
        if (auto schema = ensure_schema(*dom, "REFS"); !schema) {
            return std::move(schema).error();
        }
        return ReferencesChapter(std::move(*dom));
    }

    lfs::Result<ReferencesChapter> ReferencesChapter::from_bytes(
        const std::span<const std::byte> bytes) {
        return parse(std::string_view(
            reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    }

    lfs::Result<std::vector<ReferenceRecord>> ReferencesChapter::records() const {
        auto items = dom_.array_item_refs("references");
        if (!items) {
            return std::move(items).error();
        }
        std::vector<ReferenceRecord> result;
        result.reserve(items->size());
        std::unordered_set<std::string> keys;
        for (const auto& [id, element] : *items) {
            auto uuid = lfs::core::Uuid::from_string(id);
            auto key = JsonChapterDom::read<std::string>(*element, "key");
            auto kind = JsonChapterDom::read<std::string>(*element, "kind");
            auto unresolved = JsonChapterDom::read<bool>(*element, "unresolved");
            const JsonChapterDom::Json* locator_value =
                JsonChapterDom::read_json_ref(*element, "locator");
            const JsonChapterDom::Json* fingerprint_value =
                JsonChapterDom::read_json_ref(*element, "fingerprint");
            if (!uuid || !key || key->empty() || !kind || kind->empty() ||
                !unresolved || !locator_value || !fingerprint_value ||
                !keys.insert(*key).second) {
                return fail<std::vector<ReferenceRecord>>(
                    lfs::ErrorCode::DataLoss,
                    "The project contains an invalid external reference.",
                    std::format("REFS row UUID {} is incomplete or has a duplicate key", id),
                    "REFS", "references");
            }
            auto locator = parse_locator(*locator_value, "REFS", "references.locator");
            auto fingerprint =
                parse_fingerprint(*fingerprint_value, "REFS", "references.fingerprint");
            if (!locator) {
                return std::move(locator).error();
            }
            if (!fingerprint) {
                return std::move(fingerprint).error();
            }
            result.push_back(ReferenceRecord{
                .uuid = *uuid,
                .key = std::move(*key),
                .kind = std::move(*kind),
                .locator = std::move(*locator),
                .fingerprint = std::move(*fingerprint),
                .unresolved = *unresolved,
            });
        }
        return result;
    }

    lfs::Result<std::optional<ReferenceRecord>> ReferencesChapter::find(
        const lfs::core::Uuid& uuid) const {
        auto all = records();
        if (!all) {
            return std::move(all).error();
        }
        const auto found = std::ranges::find(all.value(), uuid, &ReferenceRecord::uuid);
        if (found == all->end()) {
            return std::optional<ReferenceRecord>{};
        }
        return std::optional<ReferenceRecord>(*found);
    }

    lfs::Result<void> ReferencesChapter::upsert(const ReferenceRecord& record) {
        if (record.uuid.is_nil() || record.key.empty() || record.kind.empty() ||
            record.locator.preferred.empty()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The external reference record is invalid.",
                "UUID, key, kind, and preferred locator must be present", "REFS",
                "references");
        }
        auto existing = records();
        if (!existing) {
            return lfs::Result<void>::failure(std::move(existing).error());
        }
        if (std::ranges::any_of(*existing, [&](const ReferenceRecord& value) {
                return value.key == record.key && value.uuid != record.uuid;
            })) {
            return fail<void>(
                lfs::ErrorCode::AlreadyExists,
                "The external reference key is already in use.",
                std::format("REFS key '{}' belongs to a different UUID", record.key),
                "REFS", "references.key");
        }
        auto element = dom_.array_upsert("references", record.uuid.to_string());
        if (!element) {
            return lfs::Result<void>::failure(std::move(element).error());
        }
        if (auto result = element->set("key", record.key); !result) {
            return result;
        }
        if (auto result = element->set("kind", record.kind); !result) {
            return result;
        }
        if (auto result = element->set_json("locator", locator_json(record.locator));
            !result) {
            return result;
        }
        if (auto result =
                element->set_json("fingerprint", fingerprint_json(record.fingerprint));
            !result) {
            return result;
        }
        return element->set("unresolved", record.unresolved);
    }

    lfs::Result<bool> ReferencesChapter::remove(const lfs::core::Uuid& uuid) {
        if (uuid.is_nil()) {
            return fail<bool>(
                lfs::ErrorCode::InvalidArgument,
                "The external reference UUID cannot be null.",
                "REFS reference UUID must be non-null", "REFS", "references.uuid");
        }
        return dom_.array_remove("references", uuid.to_string());
    }

    lfs::Result<FingerprintCheck> ReferencesChapter::verify_and_refresh(
        const lfs::core::Uuid& uuid,
        const std::filesystem::path& resolved_path) {
        auto record = find(uuid);
        if (!record) {
            return std::move(record).error();
        }
        if (!*record) {
            return fail<FingerprintCheck>(
                lfs::ErrorCode::NotFound, "The external reference record is missing.",
                std::format("REFS UUID {} does not exist", uuid.to_string()), "REFS",
                "references.uuid");
        }
        auto check = check_fingerprint(resolved_path, (*record)->fingerprint);
        if (!check) {
            return std::move(check).error();
        }
        if (check->disposition == FingerprintDisposition::MatchMtimeRefreshed) {
            auto element = dom_.array_find("references", uuid.to_string());
            if (!element || !check->observed) {
                return fail<FingerprintCheck>(
                    lfs::ErrorCode::Internal,
                    "The external reference could not be refreshed.",
                    "Matching REFS row or observed fingerprint disappeared", "REFS",
                    "references");
            }
            if (auto updated = element->set(
                    "fingerprint.mtime_unix_ns",
                    check->observed->mtime_unix_ns);
                !updated) {
                return std::move(updated).error();
            }
            if (auto updated = element->set(
                    "fingerprint.size",
                    check->observed->size);
                !updated) {
                return std::move(updated).error();
            }
            if (auto unresolved = element->set("unresolved", false); !unresolved) {
                return std::move(unresolved).error();
            }
            return *check;
        }
        if (check->disposition == FingerprintDisposition::MatchFastPath) {
            auto element = dom_.array_find("references", uuid.to_string());
            if (element) {
                if (auto unresolved = element->set("unresolved", false); !unresolved) {
                    return std::move(unresolved).error();
                }
            }
            return *check;
        }

        if (auto element = dom_.array_find("references", uuid.to_string())) {
            if (auto unresolved = element->set("unresolved", true); !unresolved) {
                return std::move(unresolved).error();
            }
        }
        return fail<FingerprintCheck>(
            lfs::ErrorCode::FailedPrecondition,
            "An external project reference must be relinked.",
            std::format("REFS UUID {} failed verification: {}", uuid.to_string(),
                        check->diagnostic),
            "REFS", "references.fingerprint");
    }

    lfs::Result<void> ReferencesChapter::relink(
        const lfs::core::Uuid& uuid, const ReferenceLocator& locator,
        const std::filesystem::path& resolved_path,
        const bool accept_content_change) {
        auto record = find(uuid);
        if (!record) {
            return lfs::Result<void>::failure(std::move(record).error());
        }
        if (!*record) {
            return fail<void>(
                lfs::ErrorCode::NotFound, "The external reference record is missing.",
                std::format("REFS UUID {} does not exist", uuid.to_string()), "REFS",
                "references.uuid");
        }
        auto observed = fingerprint_path(
            resolved_path, (*record)->fingerprint.full_xxh3.has_value());
        if (!observed) {
            return lfs::Result<void>::failure(std::move(observed).error());
        }
        const bool same_content =
            observed->kind == (*record)->fingerprint.kind &&
            observed->head_xxh3 == (*record)->fingerprint.head_xxh3 &&
            observed->tail_xxh3 == (*record)->fingerprint.tail_xxh3 &&
            (!(*record)->fingerprint.full_xxh3 ||
             observed->full_xxh3 == (*record)->fingerprint.full_xxh3);
        if (!same_content && !accept_content_change) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The selected relink target has different content.",
                std::format("REFS UUID {} size/hash does not match the stored fingerprint",
                            uuid.to_string()),
                "REFS", "references.fingerprint");
        }
        ReferenceRecord updated = **record;
        updated.locator = locator;
        updated.fingerprint = *observed;
        updated.unresolved = false;
        return upsert(updated);
    }

    namespace {

        std::filesystem::path
        absolute_lexically(const std::filesystem::path& path) {
            std::error_code error;
            auto absolute = std::filesystem::absolute(path, error);
            if (error) {
                return path.lexically_normal();
            }
            return absolute.lexically_normal();
        }

        bool path_is_under(
            const std::filesystem::path& root,
            const std::filesystem::path& candidate) {
            if (root.empty()) {
                return false;
            }
            const auto relative =
                candidate.lexically_relative(root);
            if (relative.empty() || relative == ".") {
                return true;
            }
            const auto text = relative.generic_string();
            return !text.starts_with("..");
        }

        bool fingerprint_content_matches(
            const ReferenceFingerprint& expected,
            const ReferenceFingerprint& observed) {
            return expected.kind == observed.kind &&
                   expected.size == observed.size &&
                   expected.head_xxh3 == observed.head_xxh3 &&
                   expected.tail_xxh3 == observed.tail_xxh3 &&
                   (!expected.full_xxh3 ||
                    expected.full_xxh3 == observed.full_xxh3);
        }

        std::vector<std::filesystem::path>
        locator_candidates(
            const ReferenceLocator& locator,
            const std::filesystem::path& project_root) {
            std::vector<std::filesystem::path> candidates;
            const auto preferred =
                lfs::core::utf8_to_path(locator.preferred);
            switch (locator.base) {
            case LocatorBase::Project:
                if (!project_root.empty()) {
                    candidates.push_back(
                        absolute_lexically(project_root / preferred));
                }
                candidates.push_back(absolute_lexically(preferred));
                break;
            case LocatorBase::Absolute:
            case LocatorBase::SearchRoot:
            case LocatorBase::Dataset:
                candidates.push_back(absolute_lexically(preferred));
                if (!project_root.empty() && preferred.is_relative()) {
                    candidates.push_back(
                        absolute_lexically(project_root / preferred));
                }
                break;
            }
            if (locator.absolute_fallback &&
                !locator.absolute_fallback->empty()) {
                candidates.push_back(absolute_lexically(
                    lfs::core::utf8_to_path(*locator.absolute_fallback)));
            }
            std::vector<std::filesystem::path> unique;
            unique.reserve(candidates.size());
            for (const auto& candidate : candidates) {
                if (candidate.empty()) {
                    continue;
                }
                if (std::ranges::find(unique, candidate) == unique.end()) {
                    unique.push_back(candidate);
                }
            }
            return unique;
        }

    } // namespace

    lfs::Result<lfs::core::Uuid> upsert_path_reference(
        ReferencesChapter& references,
        const std::filesystem::path& project_root,
        const std::filesystem::path& live_path,
        const std::string_view key,
        const std::string_view kind,
        std::optional<lfs::core::Uuid> existing_uuid) {
        if (key.empty() || kind.empty()) {
            return fail<lfs::core::Uuid>(
                lfs::ErrorCode::InvalidArgument,
                "A path reference key or kind is empty.",
                "REFS upsert_path_reference requires non-empty key and kind",
                "REFS", "references");
        }
        if (live_path.empty()) {
            return fail<lfs::core::Uuid>(
                lfs::ErrorCode::InvalidArgument,
                "A path reference path is empty.",
                "REFS upsert_path_reference requires a non-empty live path",
                "REFS", "references");
        }
        auto fingerprint = fingerprint_path(live_path);
        if (!fingerprint) {
            return std::move(fingerprint).error();
        }
        const auto absolute = absolute_lexically(live_path);
        const auto absolute_text = lfs::core::path_to_utf8(absolute);
        ReferenceLocator locator{
            .preferred = absolute_text,
            .base = LocatorBase::Absolute,
            .absolute_fallback = absolute_text,
        };
        if (!project_root.empty()) {
            const auto root = absolute_lexically(project_root);
            if (path_is_under(root, absolute)) {
                const auto relative = absolute.lexically_relative(root);
                if (!relative.empty() && relative != "." &&
                    !relative.generic_string().starts_with("..")) {
                    locator.preferred =
                        lfs::core::path_to_utf8(relative.generic_string());
                    locator.base = LocatorBase::Project;
                }
            }
        }

        auto records = references.records();
        if (!records) {
            return std::move(records).error();
        }
        lfs::core::Uuid uuid;
        if (existing_uuid && !existing_uuid->is_nil()) {
            uuid = *existing_uuid;
        } else if (const auto by_key = std::ranges::find(
                       *records, std::string(key), &ReferenceRecord::key);
                   by_key != records->end()) {
            uuid = by_key->uuid;
        } else {
            uuid = lfs::core::generate_uuid_v4();
        }

        ReferenceRecord record{
            .uuid = uuid,
            .key = std::string(key),
            .kind = std::string(kind),
            .locator = std::move(locator),
            .fingerprint = std::move(*fingerprint),
            .unresolved = false,
        };
        if (auto status = references.upsert(record); !status) {
            return std::move(status).error();
        }
        return uuid;
    }

    std::optional<std::filesystem::path> resolve_path_reference(
        const ReferencesChapter& references,
        const std::filesystem::path& project_root,
        const lfs::core::Uuid& uuid,
        const std::filesystem::path& hint) {
        if (uuid.is_nil()) {
            return hint.empty()
                       ? std::optional<std::filesystem::path>{}
                       : hint;
        }
        auto record = references.find(uuid);
        if (!record || !*record) {
            return hint.empty()
                       ? std::optional<std::filesystem::path>{}
                       : hint;
        }
        for (const auto& candidate :
             locator_candidates((*record)->locator, project_root)) {
            std::error_code error;
            if (!std::filesystem::exists(candidate, error) || error) {
                continue;
            }
            auto check =
                check_fingerprint(candidate, (*record)->fingerprint);
            if (check && check->matches()) {
                return candidate;
            }
            // Content mismatch: try next candidate. Missing fingerprint
            // observations also fall through.
            if (check && check->observed &&
                fingerprint_content_matches(
                    (*record)->fingerprint, *check->observed)) {
                return candidate;
            }
        }
        if (!hint.empty()) {
            return hint;
        }
        // Last resort: preferred path even without a match so callers can
        // surface a missing/mismatched file at the stored locator.
        const auto preferred = lfs::core::utf8_to_path(
            (*record)->locator.preferred);
        if ((*record)->locator.base == LocatorBase::Project &&
            !project_root.empty() && preferred.is_relative()) {
            return absolute_lexically(project_root / preferred);
        }
        if (!preferred.empty()) {
            return absolute_lexically(preferred);
        }
        if ((*record)->locator.absolute_fallback &&
            !(*record)->locator.absolute_fallback->empty()) {
            return absolute_lexically(lfs::core::utf8_to_path(
                *(*record)->locator.absolute_fallback));
        }
        return std::nullopt;
    }

    namespace {

        lfs::Result<PayloadBinding> parse_payload_binding(
            const Json& value, const std::string_view field) {
            if (auto valid = require_object(value, "SCNG", field); !valid) {
                return std::move(valid).error();
            }
            auto fourcc = required<std::string>(value, "fourcc", "SCNG", field);
            const auto instance_member = value.find("instance_uuid");
            auto reference =
                parse_optional_uuid(value, "reference_uuid", "SCNG", field);
            auto source =
                required<std::string>(value, "source_kind", "SCNG", field);
            if (!fourcc) {
                return std::move(fourcc).error();
            }
            if (instance_member == value.end()) {
                return fail<PayloadBinding>(
                    lfs::ErrorCode::DataLoss,
                    "The scene payload binding is incomplete.",
                    std::format("SCNG.{}.instance_uuid is missing", field), "SCNG",
                    std::format("{}.instance_uuid", field));
            }
            auto instance = parse_uuid(
                *instance_member, "SCNG", std::format("{}.instance_uuid", field));
            if (!reference) {
                return std::move(reference).error();
            }
            if (!source) {
                return std::move(source).error();
            }
            if (!instance) {
                return std::move(instance).error();
            }
            if (fourcc->size() != 4 || source->empty()) {
                return fail<PayloadBinding>(
                    lfs::ErrorCode::DataLoss,
                    "The scene payload binding is invalid.",
                    std::format("SCNG.{} requires a four-byte fourcc and source kind", field),
                    "SCNG", field);
            }
            return PayloadBinding{
                .fourcc = std::move(*fourcc),
                .instance_uuid = *instance,
                .reference_uuid = *reference,
                .source_kind = std::move(*source),
            };
        }

        Json payload_binding_json(const PayloadBinding& value) {
            Json result{
                {"fourcc", value.fourcc},
                {"instance_uuid", value.instance_uuid.to_string()},
                {"source_kind", value.source_kind},
            };
            if (value.reference_uuid) {
                result["reference_uuid"] = value.reference_uuid->to_string();
            }
            return result;
        }

        lfs::Result<CropBoxRecord> parse_cropbox(const Json& value,
                                                 const std::string_view field) {
            if (auto valid = require_object(value, "SCNG", field); !valid) {
                return std::move(valid).error();
            }
            const auto min_member = value.find("min");
            const auto max_member = value.find("max");
            const auto color_member = value.find("color");
            if (min_member == value.end() || max_member == value.end() ||
                color_member == value.end()) {
                return fail<CropBoxRecord>(
                    lfs::ErrorCode::DataLoss, "A scene crop box is incomplete.",
                    std::format("SCNG.{} is missing min, max, or color", field), "SCNG",
                    field);
            }
            auto min = fixed_array<float, 3>(*min_member, "SCNG",
                                             std::format("{}.min", field));
            auto max = fixed_array<float, 3>(*max_member, "SCNG",
                                             std::format("{}.max", field));
            auto color = fixed_array<float, 3>(*color_member, "SCNG",
                                               std::format("{}.color", field));
            auto inverse = required<bool>(value, "inverse", "SCNG", field);
            auto enabled = required<bool>(value, "enabled", "SCNG", field);
            auto line_width = required<float>(value, "line_width", "SCNG", field);
            if (!min) {
                return std::move(min).error();
            }
            if (!max) {
                return std::move(max).error();
            }
            if (!color) {
                return std::move(color).error();
            }
            if (!inverse) {
                return std::move(inverse).error();
            }
            if (!enabled) {
                return std::move(enabled).error();
            }
            if (!line_width) {
                return std::move(line_width).error();
            }
            if (!std::isfinite(*line_width) || *line_width <= 0.0f ||
                std::ranges::any_of(
                    std::views::iota(std::size_t{0}, std::size_t{3}),
                    [&](const std::size_t i) { return (*min)[i] > (*max)[i]; })) {
                return fail<CropBoxRecord>(
                    lfs::ErrorCode::DataLoss, "A scene crop box is invalid.",
                    std::format("SCNG.{} has inverted bounds or invalid line width", field),
                    "SCNG", field);
            }
            return CropBoxRecord{*min, *max, *inverse, *enabled, *color, *line_width};
        }

        Json cropbox_json(const CropBoxRecord& value) {
            return Json{
                {"min", json_array(value.min)},
                {"max", json_array(value.max)},
                {"inverse", value.inverse},
                {"enabled", value.enabled},
                {"color", json_array(value.color)},
                {"line_width", value.line_width},
            };
        }

        lfs::Result<EllipsoidRecord> parse_ellipsoid(
            const Json& value, const std::string_view field) {
            if (auto valid = require_object(value, "SCNG", field); !valid) {
                return std::move(valid).error();
            }
            const auto radii_member = value.find("radii");
            const auto color_member = value.find("color");
            if (radii_member == value.end() || color_member == value.end()) {
                return fail<EllipsoidRecord>(
                    lfs::ErrorCode::DataLoss, "A scene ellipsoid is incomplete.",
                    std::format("SCNG.{} is missing radii or color", field), "SCNG",
                    field);
            }
            auto radii = fixed_array<float, 3>(*radii_member, "SCNG",
                                               std::format("{}.radii", field));
            auto color = fixed_array<float, 3>(*color_member, "SCNG",
                                               std::format("{}.color", field));
            auto inverse = required<bool>(value, "inverse", "SCNG", field);
            auto enabled = required<bool>(value, "enabled", "SCNG", field);
            auto line_width = required<float>(value, "line_width", "SCNG", field);
            if (!radii) {
                return std::move(radii).error();
            }
            if (!color) {
                return std::move(color).error();
            }
            if (!inverse) {
                return std::move(inverse).error();
            }
            if (!enabled) {
                return std::move(enabled).error();
            }
            if (!line_width) {
                return std::move(line_width).error();
            }
            if (!std::isfinite(*line_width) || *line_width <= 0.0f ||
                std::ranges::any_of(*radii,
                                    [](const float radius) { return radius <= 0.0f; })) {
                return fail<EllipsoidRecord>(
                    lfs::ErrorCode::DataLoss, "A scene ellipsoid is invalid.",
                    std::format("SCNG.{} has non-positive radii or line width", field),
                    "SCNG", field);
            }
            return EllipsoidRecord{
                *radii, *inverse, *enabled, *color, *line_width};
        }

        Json ellipsoid_json(const EllipsoidRecord& value) {
            return Json{
                {"radii", json_array(value.radii)},
                {"inverse", value.inverse},
                {"enabled", value.enabled},
                {"color", json_array(value.color)},
                {"line_width", value.line_width},
            };
        }

        lfs::Result<std::vector<float>> finite_float_vector(
            const Json& object, const std::string_view key,
            const std::string_view field, const std::size_t maximum_size) {
            auto value = required<std::vector<float>>(object, key, "SCNG", field);
            if (!value) {
                return std::move(value).error();
            }
            if (value->size() > maximum_size ||
                std::ranges::any_of(*value,
                                    [](const float item) { return !std::isfinite(item); })) {
                return fail<std::vector<float>>(
                    lfs::ErrorCode::DataLoss,
                    "A scene camera distortion vector is invalid.",
                    std::format("SCNG.{}.{} exceeds {} finite values", field, key,
                                maximum_size),
                    "SCNG", std::format("{}.{}", field, key));
            }
            return value;
        }

        lfs::Result<CameraRecord> parse_camera(const Json& value,
                                               const std::string_view field) {
            if (auto valid = require_object(value, "SCNG", field); !valid) {
                return std::move(valid).error();
            }
            const auto rotation_member = value.find("rotation");
            const auto translation_member = value.find("translation");
            if (rotation_member == value.end() || translation_member == value.end()) {
                return fail<CameraRecord>(
                    lfs::ErrorCode::DataLoss, "A scene camera pose is incomplete.",
                    std::format("SCNG.{} is missing rotation or translation", field),
                    "SCNG", field);
            }
            auto rotation = fixed_array<float, 9>(
                *rotation_member, "SCNG", std::format("{}.rotation", field));
            auto translation = fixed_array<float, 3>(
                *translation_member, "SCNG", std::format("{}.translation", field));
            auto radial =
                finite_float_vector(value, "radial_distortion", field, 16);
            auto tangential =
                finite_float_vector(value, "tangential_distortion", field, 16);
            auto uid = required<std::int32_t>(value, "uid", "SCNG", field);
            auto camera_id =
                required<std::int32_t>(value, "camera_id", "SCNG", field);
            auto focal_x = required<float>(value, "focal_x", "SCNG", field);
            auto focal_y = required<float>(value, "focal_y", "SCNG", field);
            auto center_x = required<float>(value, "center_x", "SCNG", field);
            auto center_y = required<float>(value, "center_y", "SCNG", field);
            auto model = required<std::int32_t>(
                value, "camera_model_type", "SCNG", field);
            auto camera_width =
                required<std::int32_t>(value, "camera_width", "SCNG", field);
            auto camera_height =
                required<std::int32_t>(value, "camera_height", "SCNG", field);
            auto image_width =
                required<std::int32_t>(value, "image_width", "SCNG", field);
            auto image_height =
                required<std::int32_t>(value, "image_height", "SCNG", field);
            auto image_name =
                required<std::string>(value, "image_name", "SCNG", field);
            auto image_path =
                required<std::string>(value, "image_path", "SCNG", field);
            auto mask_path =
                required<std::string>(value, "mask_path", "SCNG", field);
            auto depth_path =
                required<std::string>(value, "depth_path", "SCNG", field);
            auto normal_path =
                required<std::string>(value, "normal_path", "SCNG", field);
            auto has_alpha = required<bool>(value, "has_alpha", "SCNG", field);
            auto has_image =
                optional<bool>(value, "has_image", "SCNG", field);
            auto split = required<std::string>(value, "split", "SCNG", field);
            if (!rotation) {
                return std::move(rotation).error();
            }
            if (!translation) {
                return std::move(translation).error();
            }
            if (!radial) {
                return std::move(radial).error();
            }
            if (!tangential) {
                return std::move(tangential).error();
            }
            if (auto error =
                    first_error(uid, camera_id, focal_x, focal_y, center_x, center_y,
                                model, camera_width, camera_height, image_width,
                                image_height, image_name, image_path, mask_path,
                                depth_path, normal_path, has_alpha, has_image,
                                split)) {
                return std::move(*error);
            }
            if (*camera_width < 0 || *camera_height < 0 || *image_width < 0 ||
                *image_height < 0 || (*split != "train" && *split != "eval") ||
                !std::isfinite(*focal_x) || !std::isfinite(*focal_y) ||
                !std::isfinite(*center_x) || !std::isfinite(*center_y)) {
                return fail<CameraRecord>(
                    lfs::ErrorCode::DataLoss, "A scene camera record is invalid.",
                    std::format("SCNG.{} contains invalid dimensions, split, or intrinsics",
                                field),
                    "SCNG", field);
            }
            return CameraRecord{
                .uid = *uid,
                .camera_id = *camera_id,
                .rotation = *rotation,
                .translation = *translation,
                .focal_x = *focal_x,
                .focal_y = *focal_y,
                .center_x = *center_x,
                .center_y = *center_y,
                .radial_distortion = std::move(*radial),
                .tangential_distortion = std::move(*tangential),
                .camera_model_type = *model,
                .camera_width = *camera_width,
                .camera_height = *camera_height,
                .image_width = *image_width,
                .image_height = *image_height,
                .image_name = std::move(*image_name),
                .image_path = std::move(*image_path),
                .mask_path = std::move(*mask_path),
                .depth_path = std::move(*depth_path),
                .normal_path = std::move(*normal_path),
                .has_alpha = *has_alpha,
                .has_image = has_image->value_or(true),
                .split = std::move(*split),
            };
        }

        Json camera_json(const CameraRecord& value) {
            return Json{
                {"uid", value.uid},
                {"camera_id", value.camera_id},
                {"rotation", json_array(value.rotation)},
                {"translation", json_array(value.translation)},
                {"focal_x", value.focal_x},
                {"focal_y", value.focal_y},
                {"center_x", value.center_x},
                {"center_y", value.center_y},
                {"radial_distortion", value.radial_distortion},
                {"tangential_distortion", value.tangential_distortion},
                {"camera_model_type", value.camera_model_type},
                {"camera_width", value.camera_width},
                {"camera_height", value.camera_height},
                {"image_width", value.image_width},
                {"image_height", value.image_height},
                {"image_name", value.image_name},
                {"image_path", value.image_path},
                {"mask_path", value.mask_path},
                {"depth_path", value.depth_path},
                {"normal_path", value.normal_path},
                {"has_alpha", value.has_alpha},
                {"has_image", value.has_image},
                {"split", value.split},
            };
        }

        lfs::Result<SceneNodeRecord> parse_scene_node(
            const Json& element, const lfs::core::Uuid& uuid) {
            auto type = JsonChapterDom::read<std::string>(element, "type");
            auto name = JsonChapterDom::read<std::string>(element, "name");
            const Json* parent_value =
                JsonChapterDom::read_json_ref(element, "parent_uuid");
            std::optional<lfs::core::Uuid> parent;
            if (parent_value && !parent_value->is_null()) {
                auto parsed = parse_uuid(*parent_value, "SCNG", "nodes.parent_uuid");
                if (!parsed) {
                    return std::move(parsed).error();
                }
                parent = *parsed;
            }
            auto order = JsonChapterDom::read<std::uint32_t>(element, "child_order");
            const Json* transform_value =
                JsonChapterDom::read_json_ref(element, "local_transform");
            auto visible = JsonChapterDom::read<bool>(element, "visible");
            auto locked = JsonChapterDom::read<bool>(element, "locked");
            auto training = JsonChapterDom::read<bool>(element, "training_enabled");
            auto diverged = JsonChapterDom::read<bool>(element, "payload_diverged");
            if (!type || type->empty() || !name || name->empty() || !order ||
                !transform_value || !visible || !locked || !training || !diverged) {
                return fail<SceneNodeRecord>(
                    lfs::ErrorCode::DataLoss, "A scene node record is incomplete.",
                    std::format("SCNG node UUID {} is missing a required field",
                                uuid.to_string()),
                    "SCNG", "nodes");
            }
            auto transform = fixed_array<float, 16>(
                *transform_value, "SCNG", "nodes.local_transform");
            if (!transform) {
                return std::move(transform).error();
            }
            SceneNodeRecord result{
                .uuid = uuid,
                .type = std::move(*type),
                .name = std::move(*name),
                .parent_uuid = parent,
                .child_order = *order,
                .local_transform = *transform,
                .visible = *visible,
                .locked = *locked,
                .training_enabled = *training,
                .payload_diverged = *diverged,
                .georef_pose = std::nullopt,
                .payload = std::nullopt,
                .cropbox = std::nullopt,
                .ellipsoid = std::nullopt,
                .camera = std::nullopt,
            };

            if (const Json* pose =
                    JsonChapterDom::read_json_ref(element, "georef_pose");
                pose) {
                if (auto valid = require_object(*pose, "SCNG", "nodes.georef_pose");
                    !valid) {
                    return std::move(valid).error();
                }
                const auto rotation = pose->find("rotation");
                const auto translation = pose->find("translation");
                if (rotation == pose->end() || translation == pose->end()) {
                    return fail<SceneNodeRecord>(
                        lfs::ErrorCode::DataLoss,
                        "A node georeference pose is incomplete.",
                        "SCNG.nodes.georef_pose requires rotation and translation",
                        "SCNG", "nodes.georef_pose");
                }
                auto r = fixed_array<double, 4>(
                    *rotation, "SCNG", "nodes.georef_pose.rotation");
                auto t = fixed_array<double, 3>(
                    *translation, "SCNG", "nodes.georef_pose.translation");
                if (!r) {
                    return std::move(r).error();
                }
                if (!t) {
                    return std::move(t).error();
                }
                const double norm = std::sqrt(
                    (*r)[0] * (*r)[0] + (*r)[1] * (*r)[1] +
                    (*r)[2] * (*r)[2] + (*r)[3] * (*r)[3]);
                if (!std::isfinite(norm) || norm < 1e-12) {
                    return fail<SceneNodeRecord>(
                        lfs::ErrorCode::DataLoss,
                        "A node georeference quaternion is invalid.",
                        "SCNG.nodes.georef_pose.rotation has zero norm", "SCNG",
                        "nodes.georef_pose.rotation");
                }
                result.georef_pose = GeorefPose{*r, *t};
            }
            if (const Json* payload =
                    JsonChapterDom::read_json_ref(element, "payload");
                payload) {
                auto parsed = parse_payload_binding(*payload, "nodes.payload");
                if (!parsed) {
                    return std::move(parsed).error();
                }
                result.payload = std::move(*parsed);
            }
            if (const Json* cropbox =
                    JsonChapterDom::read_json_ref(element, "cropbox");
                cropbox) {
                auto parsed = parse_cropbox(*cropbox, "nodes.cropbox");
                if (!parsed) {
                    return std::move(parsed).error();
                }
                result.cropbox = std::move(*parsed);
            }
            if (const Json* ellipsoid =
                    JsonChapterDom::read_json_ref(element, "ellipsoid");
                ellipsoid) {
                auto parsed = parse_ellipsoid(*ellipsoid, "nodes.ellipsoid");
                if (!parsed) {
                    return std::move(parsed).error();
                }
                result.ellipsoid = std::move(*parsed);
            }
            if (const Json* camera =
                    JsonChapterDom::read_json_ref(element, "camera");
                camera) {
                auto parsed = parse_camera(*camera, "nodes.camera");
                if (!parsed) {
                    return std::move(parsed).error();
                }
                result.camera = std::move(*parsed);
            }
            return result;
        }

    } // namespace

    SceneGraphChapter::SceneGraphChapter() {
        initialize_json_chapter(dom_, "nodes");
        const auto training = dom_.set_json("training_model_uuid", nullptr);
        (void)training;
    }

    SceneGraphChapter::SceneGraphChapter(JsonChapterDom dom)
        : dom_(std::move(dom)) {}

    lfs::Result<SceneGraphChapter> SceneGraphChapter::parse(
        const std::string_view bytes) {
        auto dom = JsonChapterDom::parse(bytes);
        if (!dom) {
            return std::move(dom).error();
        }
        if (auto schema = ensure_schema(*dom, "SCNG"); !schema) {
            return std::move(schema).error();
        }
        SceneGraphChapter result(std::move(*dom));
        if (auto valid = result.validate_hierarchy(); !valid) {
            return std::move(valid).error();
        }
        return result;
    }

    lfs::Result<SceneGraphChapter> SceneGraphChapter::from_bytes(
        const std::span<const std::byte> bytes) {
        return parse(std::string_view(
            reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    }

    lfs::Result<std::optional<lfs::core::Uuid>>
    SceneGraphChapter::training_model_uuid() const {
        const JsonChapterDom::Json* value = dom_.get_json_ref("training_model_uuid");
        if (!value || value->is_null()) {
            return std::optional<lfs::core::Uuid>{};
        }
        auto parsed = parse_uuid(*value, "SCNG", "training_model_uuid");
        if (!parsed) {
            return std::move(parsed).error();
        }
        return std::optional<lfs::core::Uuid>(*parsed);
    }

    void SceneGraphChapter::invalidate_parsed_nodes() noexcept {
        cached_nodes_.reset();
        hierarchy_valid_ = false;
    }

    lfs::Result<void> SceneGraphChapter::set_training_model_uuid(
        const std::optional<lfs::core::Uuid> value) {
        hierarchy_valid_ = false;
        if (value && value->is_nil()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The training-model node UUID cannot be null.",
                "SCNG.training_model_uuid must be non-null when present", "SCNG",
                "training_model_uuid");
        }
        return value ? dom_.set("training_model_uuid", value->to_string())
                     : dom_.set_json("training_model_uuid", nullptr);
    }

    lfs::Result<std::vector<SceneNodeRecord>> SceneGraphChapter::nodes() const {
        if (cached_nodes_) {
            return *cached_nodes_;
        }
        auto items = dom_.array_item_refs("nodes");
        if (!items) {
            return std::move(items).error();
        }
        std::vector<SceneNodeRecord> result;
        result.reserve(items->size());
        for (const auto& [id, element] : *items) {
            const auto uuid = lfs::core::Uuid::from_string(id);
            if (!uuid) {
                return fail<std::vector<SceneNodeRecord>>(
                    lfs::ErrorCode::DataLoss, "A scene node UUID is invalid.",
                    std::format("SCNG node UUID '{}' cannot be decoded", id), "SCNG",
                    "nodes.uuid");
            }
            auto parsed = parse_scene_node(*element, *uuid);
            if (!parsed) {
                return std::move(parsed).error();
            }
            result.push_back(std::move(*parsed));
        }
        cached_nodes_ = std::move(result);
        return *cached_nodes_;
    }

    lfs::Result<std::optional<SceneNodeRecord>> SceneGraphChapter::find(
        const lfs::core::Uuid& uuid) const {
        if (uuid.is_nil()) {
            return fail<std::optional<SceneNodeRecord>>(
                lfs::ErrorCode::InvalidArgument, "The scene node UUID cannot be null.",
                "SCNG lookup UUID must be non-null", "SCNG", "nodes.uuid");
        }
        const auto element = dom_.array_get("nodes", uuid.to_string());
        if (!element) {
            return std::optional<SceneNodeRecord>{};
        }
        auto parsed = parse_scene_node(*element, uuid);
        if (!parsed) {
            return std::move(parsed).error();
        }
        return std::optional<SceneNodeRecord>(std::move(*parsed));
    }

    lfs::Result<void> SceneGraphChapter::upsert_node(
        const SceneNodeRecord& value) {
        invalidate_parsed_nodes();
        if (value.uuid.is_nil() || value.type.empty() || value.name.empty() ||
            (value.parent_uuid && value.parent_uuid->is_nil()) ||
            std::ranges::any_of(value.local_transform,
                                [](const float item) { return !std::isfinite(item); }) ||
            value.type == "keyframe" || value.type == "keyframe_group") {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument, "The scene node record is invalid.",
                "UUID/type/name/transform must be valid and generated keyframes are excluded",
                "SCNG", "nodes");
        }
        if (value.payload &&
            (value.payload->fourcc.size() != 4 ||
             value.payload->instance_uuid.is_nil() ||
             value.payload->source_kind.empty() ||
             (value.payload->reference_uuid &&
              value.payload->reference_uuid->is_nil()))) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The scene node payload binding is invalid.",
                "Payload fourcc, UUID, source kind, and reference must be valid", "SCNG",
                "nodes.payload");
        }
        std::optional<Json> pose;
        if (value.georef_pose) {
            const double norm = std::sqrt(
                value.georef_pose->rotation[0] * value.georef_pose->rotation[0] +
                value.georef_pose->rotation[1] * value.georef_pose->rotation[1] +
                value.georef_pose->rotation[2] * value.georef_pose->rotation[2] +
                value.georef_pose->rotation[3] * value.georef_pose->rotation[3]);
            if (!std::isfinite(norm) || norm < 1e-12 ||
                std::ranges::any_of(value.georef_pose->translation, [](const double item) { return !std::isfinite(item); })) {
                return fail<void>(
                    lfs::ErrorCode::InvalidArgument,
                    "The node georeference pose is invalid.",
                    "Quaternion must have non-zero norm and all values must be finite",
                    "SCNG", "nodes.georef_pose");
            }
            pose = Json{
                {"rotation", json_array(value.georef_pose->rotation)},
                {"translation", json_array(value.georef_pose->translation)},
            };
        }

        const std::string uuid = value.uuid.to_string();
        Json element = dom_.array_get("nodes", uuid).value_or(Json{{"uuid", uuid}});
        element["type"] = value.type;
        element["name"] = value.name;
        if (value.parent_uuid) {
            element["parent_uuid"] = value.parent_uuid->to_string();
        } else {
            element["parent_uuid"] = nullptr;
        }
        element["child_order"] = value.child_order;
        element["local_transform"] = json_array(value.local_transform);
        element["visible"] = value.visible;
        element["locked"] = value.locked;
        element["training_enabled"] = value.training_enabled;
        element["payload_diverged"] = value.payload_diverged;
        if (pose) {
            element["georef_pose"] = std::move(*pose);
        } else {
            element.erase("georef_pose");
        }
        if (value.payload) {
            element["payload"] = payload_binding_json(*value.payload);
        } else {
            element.erase("payload");
        }
        if (value.cropbox) {
            element["cropbox"] = cropbox_json(*value.cropbox);
        } else {
            element.erase("cropbox");
        }
        if (value.ellipsoid) {
            element["ellipsoid"] = ellipsoid_json(*value.ellipsoid);
        } else {
            element.erase("ellipsoid");
        }
        if (value.camera) {
            element["camera"] = camera_json(*value.camera);
        } else {
            element.erase("camera");
        }
        return dom_.array_put("nodes", uuid, std::move(element));
    }

    lfs::Result<bool> SceneGraphChapter::remove_node(
        const lfs::core::Uuid& uuid) {
        invalidate_parsed_nodes();
        if (uuid.is_nil()) {
            return fail<bool>(
                lfs::ErrorCode::InvalidArgument, "The scene node UUID cannot be null.",
                "SCNG node UUID must be non-null", "SCNG", "nodes.uuid");
        }
        return dom_.array_remove("nodes", uuid.to_string());
    }

    lfs::Result<void> SceneGraphChapter::validate_hierarchy() const {
        if (hierarchy_valid_) {
            return {};
        }
        auto all = nodes();
        if (!all) {
            return lfs::Result<void>::failure(std::move(all).error());
        }
        if (auto valid = validate_hierarchy(*all); !valid) {
            return valid;
        }
        hierarchy_valid_ = true;
        return {};
    }

    lfs::Result<void> SceneGraphChapter::validate_hierarchy(
        const std::span<const SceneNodeRecord> all) const {
        std::unordered_map<lfs::core::Uuid, std::size_t> positions;
        positions.reserve(all.size());
        std::unordered_map<lfs::core::Uuid, std::set<std::uint32_t>> orders;
        std::set<std::uint32_t> root_orders;
        for (std::size_t i = 0; i < all.size(); ++i) {
            positions.emplace(all[i].uuid, i);
        }
        for (std::size_t i = 0; i < all.size(); ++i) {
            const SceneNodeRecord& node = all[i];
            if (node.parent_uuid) {
                const auto parent = positions.find(*node.parent_uuid);
                if (parent == positions.end()) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A scene node references a missing parent.",
                        std::format("SCNG node {} parent {} does not exist",
                                    node.uuid.to_string(),
                                    node.parent_uuid->to_string()),
                        "SCNG", "nodes.parent_uuid");
                }
                if (parent->second >= i) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "The scene hierarchy is not parent-first.",
                        std::format("SCNG node {} occurs before its parent {}",
                                    node.uuid.to_string(),
                                    node.parent_uuid->to_string()),
                        "SCNG", "nodes");
                }
                if (!orders[*node.parent_uuid].insert(node.child_order).second) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "Two scene children have the same saved order.",
                        std::format("Parent {} has duplicate child_order {}",
                                    node.parent_uuid->to_string(), node.child_order),
                        "SCNG", "nodes.child_order");
                }
            } else if (!root_orders.insert(node.child_order).second) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "Two root scene nodes have the same saved order.",
                    std::format("Root child_order {} is duplicated", node.child_order),
                    "SCNG", "nodes.child_order");
            }
        }
        const auto contiguous = [](const std::set<std::uint32_t>& values) {
            std::uint32_t expected = 0;
            for (const std::uint32_t value : values) {
                if (value != expected++) {
                    return false;
                }
            }
            return true;
        };
        if (!contiguous(root_orders) ||
            std::ranges::any_of(orders, [&](const auto& item) {
                return !contiguous(item.second);
            })) {
            return fail<void>(
                lfs::ErrorCode::DataLoss,
                "The saved scene child order contains gaps.",
                "Every sibling set must use contiguous child_order values starting at zero",
                "SCNG", "nodes.child_order");
        }
        auto training = training_model_uuid();
        if (!training) {
            return lfs::Result<void>::failure(std::move(training).error());
        }
        if (*training && !positions.contains(**training)) {
            return fail<void>(
                lfs::ErrorCode::DataLoss,
                "The training-model scene node is missing.",
                std::format("SCNG.training_model_uuid {} is not in nodes",
                            (**training).to_string()),
                "SCNG", "training_model_uuid");
        }
        return {};
    }

    namespace {

        Json ordered_json_from(const nlohmann::json& value) {
            return Json::parse(value.dump());
        }

        Json parameter_json(
            const lfs::core::param::OptimizationParameters& value,
            const ParameterManagerSnapshot::ReferenceBindings& references) {
            Json result = ordered_json_from(value.to_json());
            result.erase("headless");
            result.erase("config_file");
            result.erase("bg_image_path");
            result.erase("ppisp_sidecar_path");
            if (references.background_image_reference) {
                result["background_image_reference_uuid"] =
                    references.background_image_reference->to_string();
            }
            if (references.ppisp_reference) {
                result["ppisp_reference_uuid"] =
                    references.ppisp_reference->to_string();
            }
            return result;
        }

        struct ParsedParameterPreset {
            lfs::core::param::OptimizationParameters parameters;
            ParameterManagerSnapshot::ReferenceBindings references;
        };

        lfs::Result<ParsedParameterPreset>
        parse_parameter_json(const Json& value, const std::string_view field) {
            if (!value.is_object()) {
                return fail<ParsedParameterPreset>(
                    lfs::ErrorCode::DataLoss,
                    "A pending parameter preset is not an object.",
                    std::format("PRMS.{} is {}, expected object", field,
                                value.type_name()),
                    "PRMS", field);
            }
            auto background_reference = parse_optional_uuid(
                value, "background_image_reference_uuid", "PRMS", field);
            auto ppisp_reference = parse_optional_uuid(
                value, "ppisp_reference_uuid", "PRMS", field);
            if (!background_reference) {
                return std::move(background_reference).error();
            }
            if (!ppisp_reference) {
                return std::move(ppisp_reference).error();
            }
            try {
                Json adapted = value;
                adapted.erase("background_image_reference_uuid");
                adapted.erase("ppisp_reference_uuid");
                adapted.erase("bg_image_path");
                adapted.erase("ppisp_sidecar_path");
                adapted.erase("config_file");
                adapted["headless"] = false;
                auto result =
                    lfs::core::param::OptimizationParameters::from_json(
                        nlohmann::json::parse(adapted.dump()));
                result.headless = false;
                result.config_file.clear();
                result.bg_image_path.clear();
                result.ppisp_sidecar_path.clear();
                if (const std::string invalid = result.validate(); !invalid.empty()) {
                    return fail<ParsedParameterPreset>(
                        lfs::ErrorCode::DataLoss,
                        "A pending parameter preset is invalid.",
                        std::format("PRMS.{}: {}", field, invalid), "PRMS", field);
                }
                return ParsedParameterPreset{
                    .parameters = std::move(result),
                    .references =
                        ParameterManagerSnapshot::ReferenceBindings{
                            .background_image_reference =
                                std::move(*background_reference),
                            .ppisp_reference =
                                std::move(*ppisp_reference),
                        },
                };
            } catch (const nlohmann::json::exception& error) {
                return fail<ParsedParameterPreset>(
                    lfs::ErrorCode::DataLoss,
                    "A pending parameter preset could not be decoded.",
                    std::format("PRMS.{}: {}", field, error.what()), "PRMS", field);
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): the legacy parameter parser throws
                // foreign standard exceptions; normalize them at the chapter boundary.
                return fail<ParsedParameterPreset>(
                    lfs::ErrorCode::DataLoss,
                    "A pending parameter preset could not be decoded.",
                    std::format("PRMS.{}: {}", field, error.what()), "PRMS", field);
            }
        }

        Json dataset_json(const lfs::core::param::DatasetConfig& value) {
            return Json{
                {"images", value.images},
                {"resize_factor", value.resize_factor},
                {"test_every", value.test_every},
                {"timelapse_images", value.timelapse_images},
                {"timelapse_every", value.timelapse_every},
                {"max_width", value.max_width},
                {"min_track_length", value.min_track_length},
                {"invert_masks", value.invert_masks},
                {"mask_threshold", value.mask_threshold},
                {"centralize_dataset", value.centralize_dataset},
                {"loading_params", ordered_json_from(value.loading_params.to_json())},
            };
        }

        lfs::Result<lfs::core::param::DatasetConfig> parse_dataset_json(
            const Json& value) {
            if (!value.is_object()) {
                return fail<lfs::core::param::DatasetConfig>(
                    lfs::ErrorCode::DataLoss,
                    "The pending dataset parameters are invalid.",
                    std::format("PRMS.dataset is {}, expected object", value.type_name()),
                    "PRMS", "dataset");
            }
            auto images = required<std::string>(value, "images", "PRMS", "dataset");
            auto resize =
                required<int>(value, "resize_factor", "PRMS", "dataset");
            auto test_every =
                required<int>(value, "test_every", "PRMS", "dataset");
            auto timelapse_images = required<std::vector<std::string>>(
                value, "timelapse_images", "PRMS", "dataset");
            auto timelapse_every =
                required<int>(value, "timelapse_every", "PRMS", "dataset");
            auto max_width =
                required<int>(value, "max_width", "PRMS", "dataset");
            auto min_track =
                required<int>(value, "min_track_length", "PRMS", "dataset");
            auto invert =
                required<bool>(value, "invert_masks", "PRMS", "dataset");
            auto threshold =
                required<float>(value, "mask_threshold", "PRMS", "dataset");
            auto centralize = required<std::string>(
                value, "centralize_dataset", "PRMS", "dataset");
            const auto loading = value.find("loading_params");
            if (auto error =
                    first_error(images, resize, test_every, timelapse_images,
                                timelapse_every, max_width, min_track, invert,
                                threshold, centralize)) {
                return std::move(*error);
            }
            if (loading == value.end() || !loading->is_object()) {
                return fail<lfs::core::param::DatasetConfig>(
                    lfs::ErrorCode::DataLoss,
                    "The pending loading policy is missing.",
                    "PRMS.dataset.loading_params must be an object", "PRMS",
                    "dataset.loading_params");
            }
            lfs::core::param::DatasetConfig result;
            try {
                result.loading_params =
                    lfs::core::param::LoadingParams::from_json(
                        nlohmann::json::parse(loading->dump()));
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): normalize the legacy JSON parser.
                return fail<lfs::core::param::DatasetConfig>(
                    lfs::ErrorCode::DataLoss,
                    "The pending loading policy could not be decoded.",
                    std::format("PRMS.dataset.loading_params: {}", error.what()),
                    "PRMS", "dataset.loading_params");
            }
            result.images = std::move(*images);
            result.resize_factor = *resize;
            result.test_every = *test_every;
            result.timelapse_images = std::move(*timelapse_images);
            result.timelapse_every = *timelapse_every;
            result.max_width = *max_width;
            result.min_track_length = *min_track;
            result.invert_masks = *invert;
            result.mask_threshold = *threshold;
            result.centralize_dataset = std::move(*centralize);
            result.data_path.clear();
            result.output_path.clear();
            result.output_name.clear();
            if (const std::string invalid = result.validate(); !invalid.empty()) {
                return fail<lfs::core::param::DatasetConfig>(
                    lfs::ErrorCode::DataLoss,
                    "The pending dataset parameters are invalid.",
                    std::format("PRMS.dataset: {}", invalid), "PRMS", "dataset");
            }
            return result;
        }

        bool valid_pending_strategy(const std::string_view value) {
            return value == lfs::core::param::kStrategyMCMC ||
                   lfs::core::param::is_mrnf_strategy(value) ||
                   value == lfs::core::param::kStrategyIGSPlus;
        }

        bool matches_parameter_role(
            const lfs::core::param::OptimizationParameters& value,
            const std::string_view role) {
            if (role == "mcmc") {
                return value.strategy == lfs::core::param::kStrategyMCMC;
            }
            if (role == "mrnf") {
                return lfs::core::param::is_mrnf_strategy(value.strategy);
            }
            return role == "igs+" &&
                   value.strategy == lfs::core::param::kStrategyIGSPlus;
        }

    } // namespace

    ParametersChapter::ParametersChapter() {
        initialize_json_chapter(dom_);
    }

    ParametersChapter::ParametersChapter(JsonChapterDom dom)
        : dom_(std::move(dom)) {}

    lfs::Result<ParametersChapter> ParametersChapter::parse(
        const std::string_view bytes) {
        auto dom = JsonChapterDom::parse(bytes);
        if (!dom) {
            return std::move(dom).error();
        }
        if (auto schema = ensure_schema(*dom, "PRMS"); !schema) {
            return std::move(schema).error();
        }
        ParametersChapter result(std::move(*dom));
        auto validated = result.snapshot();
        if (!validated) {
            return std::move(validated).error();
        }
        return result;
    }

    lfs::Result<ParametersChapter> ParametersChapter::from_bytes(
        const std::span<const std::byte> bytes) {
        return parse(std::string_view(
            reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    }

    lfs::Result<ParameterManagerSnapshot> ParametersChapter::snapshot() const {
        if (cached_snapshot_) {
            return *cached_snapshot_;
        }
        const auto active = dom_.get<std::string>("active_strategy");
        const JsonChapterDom::Json* presets = dom_.get_json_ref("presets");
        const JsonChapterDom::Json* dataset = dom_.get_json_ref("dataset");
        if (!active || !valid_pending_strategy(*active) || !presets ||
            !presets->is_object() || !dataset) {
            return fail<ParameterManagerSnapshot>(
                lfs::ErrorCode::DataLoss,
                "The pending parameter snapshot is incomplete.",
                "PRMS requires active_strategy, presets, and dataset", "PRMS");
        }
        const auto parse_preset =
            [&](const std::string_view strategy,
                const std::string_view role)
            -> lfs::Result<ParsedParameterPreset> {
            const auto strategy_object = presets->find(std::string(strategy));
            if (strategy_object == presets->end() || !strategy_object->is_object()) {
                return fail<ParsedParameterPreset>(
                    lfs::ErrorCode::DataLoss,
                    "A pending strategy preset is missing.",
                    std::format("PRMS.presets.{} is missing", strategy), "PRMS",
                    std::format("presets.{}", strategy));
            }
            const auto role_object = strategy_object->find(std::string(role));
            if (role_object == strategy_object->end()) {
                return fail<ParsedParameterPreset>(
                    lfs::ErrorCode::DataLoss,
                    "A pending strategy role is missing.",
                    std::format("PRMS.presets.{}.{} is missing", strategy, role),
                    "PRMS", std::format("presets.{}.{}", strategy, role));
            }
            return parse_parameter_json(
                *role_object, std::format("presets.{}.{}", strategy, role));
        };
        auto mcmc_session = parse_preset("mcmc", "session");
        auto mrnf_session = parse_preset("mrnf", "session");
        auto igs_session = parse_preset("igs+", "session");
        auto mcmc_current = parse_preset("mcmc", "current");
        auto mrnf_current = parse_preset("mrnf", "current");
        auto igs_current = parse_preset("igs+", "current");
        auto parsed_dataset = parse_dataset_json(*dataset);
        if (!mcmc_session) {
            return std::move(mcmc_session).error();
        }
        if (!mrnf_session) {
            return std::move(mrnf_session).error();
        }
        if (!igs_session) {
            return std::move(igs_session).error();
        }
        if (!mcmc_current) {
            return std::move(mcmc_current).error();
        }
        if (!mrnf_current) {
            return std::move(mrnf_current).error();
        }
        if (!igs_current) {
            return std::move(igs_current).error();
        }
        if (!parsed_dataset) {
            return std::move(parsed_dataset).error();
        }
        const std::array roles{
            std::pair{"mcmc", &mcmc_session->parameters},
            std::pair{"mrnf", &mrnf_session->parameters},
            std::pair{"igs+", &igs_session->parameters},
            std::pair{"mcmc", &mcmc_current->parameters},
            std::pair{"mrnf", &mrnf_current->parameters},
            std::pair{"igs+", &igs_current->parameters},
        };
        if (std::ranges::any_of(
                roles, [](const auto& entry) {
                    return !matches_parameter_role(*entry.second, entry.first);
                })) {
            return fail<ParameterManagerSnapshot>(
                lfs::ErrorCode::DataLoss,
                "A pending parameter preset is stored under the wrong strategy.",
                "Each PRMS preset must carry the strategy named by its role",
                "PRMS", "presets");
        }
        cached_snapshot_ = ParameterManagerSnapshot{
            .active_strategy = *active,
            .mcmc_session = std::move(mcmc_session->parameters),
            .mrnf_session = std::move(mrnf_session->parameters),
            .igs_session = std::move(igs_session->parameters),
            .mcmc_current = std::move(mcmc_current->parameters),
            .mrnf_current = std::move(mrnf_current->parameters),
            .igs_current = std::move(igs_current->parameters),
            .mcmc_session_references =
                std::move(mcmc_session->references),
            .mrnf_session_references =
                std::move(mrnf_session->references),
            .igs_session_references =
                std::move(igs_session->references),
            .mcmc_current_references =
                std::move(mcmc_current->references),
            .mrnf_current_references =
                std::move(mrnf_current->references),
            .igs_current_references =
                std::move(igs_current->references),
            .dataset = std::move(*parsed_dataset),
        };
        return *cached_snapshot_;
    }

    lfs::Result<void> ParametersChapter::set_snapshot(
        const ParameterManagerSnapshot& value) {
        cached_snapshot_.reset();
        if (!valid_pending_strategy(value.active_strategy)) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The pending active strategy is invalid.",
                std::format("'{}' is not mcmc, mrnf, or igs+",
                            value.active_strategy),
                "PRMS", "active_strategy");
        }
        struct ParameterRole {
            std::string_view field;
            std::string_view role;
            const lfs::core::param::OptimizationParameters* parameters;
            const ParameterManagerSnapshot::ReferenceBindings* references;
        };
        const std::array parameter_values{
            ParameterRole{"mcmc.session", "mcmc", &value.mcmc_session,
                          &value.mcmc_session_references},
            ParameterRole{"mrnf.session", "mrnf", &value.mrnf_session,
                          &value.mrnf_session_references},
            ParameterRole{"igs+.session", "igs+", &value.igs_session,
                          &value.igs_session_references},
            ParameterRole{"mcmc.current", "mcmc", &value.mcmc_current,
                          &value.mcmc_current_references},
            ParameterRole{"mrnf.current", "mrnf", &value.mrnf_current,
                          &value.mrnf_current_references},
            ParameterRole{"igs+.current", "igs+", &value.igs_current,
                          &value.igs_current_references},
        };
        for (const auto& [field, role, params, references] : parameter_values) {
            if (!matches_parameter_role(*params, role)) {
                return fail<void>(
                    lfs::ErrorCode::InvalidArgument,
                    "A pending parameter preset has the wrong strategy.",
                    std::format("PRMS.presets.{} carries strategy '{}'",
                                field, params->strategy),
                    "PRMS", std::format("presets.{}", field));
            }
            if (const std::string invalid = params->validate();
                !invalid.empty()) {
                return fail<void>(
                    lfs::ErrorCode::InvalidArgument,
                    "A pending parameter preset is invalid.",
                    std::format("PRMS.presets.{}: {}", field, invalid), "PRMS",
                    std::format("presets.{}", field));
            }
            if ((references->background_image_reference &&
                 references->background_image_reference->is_nil()) ||
                (references->ppisp_reference &&
                 references->ppisp_reference->is_nil())) {
                return fail<void>(
                    lfs::ErrorCode::InvalidArgument,
                    "A pending parameter reference UUID is null.",
                    std::format(
                        "PRMS.presets.{} contains a null logical reference",
                        field),
                    "PRMS", std::format("presets.{}", field));
            }
        }
        if (const std::string invalid = value.dataset.validate(); !invalid.empty()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The pending dataset parameters are invalid.",
                std::format("PRMS.dataset: {}", invalid), "PRMS", "dataset");
        }
        if (auto result = dom_.set("active_strategy", value.active_strategy); !result) {
            return result;
        }
        const auto merge_at =
            [&](const std::string_view path, Json known) -> lfs::Result<void> {
            Json existing = dom_.get_json(path).value_or(Json::object());
            Json merged = merge_known(std::move(existing), known);
            if (known.is_object() &&
                !known.contains("background_image_reference_uuid")) {
                merged.erase("background_image_reference_uuid");
            }
            if (known.is_object() &&
                !known.contains("ppisp_reference_uuid")) {
                merged.erase("ppisp_reference_uuid");
            }
            merged.erase("bg_image_path");
            merged.erase("ppisp_sidecar_path");
            merged.erase("config_file");
            merged.erase("headless");
            return dom_.set_json(path, std::move(merged));
        };
        if (auto result =
                merge_at("presets.mcmc.session",
                         parameter_json(value.mcmc_session,
                                        value.mcmc_session_references));
            !result) {
            return result;
        }
        if (auto result =
                merge_at("presets.mrnf.session",
                         parameter_json(value.mrnf_session,
                                        value.mrnf_session_references));
            !result) {
            return result;
        }
        if (auto result =
                merge_at("presets.igs+.session",
                         parameter_json(value.igs_session,
                                        value.igs_session_references));
            !result) {
            return result;
        }
        if (auto result =
                merge_at("presets.mcmc.current",
                         parameter_json(value.mcmc_current,
                                        value.mcmc_current_references));
            !result) {
            return result;
        }
        if (auto result =
                merge_at("presets.mrnf.current",
                         parameter_json(value.mrnf_current,
                                        value.mrnf_current_references));
            !result) {
            return result;
        }
        if (auto result =
                merge_at("presets.igs+.current",
                         parameter_json(value.igs_current,
                                        value.igs_current_references));
            !result) {
            return result;
        }
        return merge_at("dataset", dataset_json(value.dataset));
    }

    lfs::Result<ReverseReferenceIndex> build_reverse_reference_index(
        const ReferencesChapter& references, const ProjectChapter& project,
        const SceneGraphChapter& scene,
        const std::span<const ReferenceOwnerBinding> additional_bindings) {
        auto records = references.records();
        if (!records) {
            return std::move(records).error();
        }
        auto nodes = scene.nodes();
        if (!nodes) {
            return std::move(nodes).error();
        }
        return build_reverse_reference_index(
            *records, project, *nodes, additional_bindings);
    }

    lfs::Result<ReverseReferenceIndex> build_reverse_reference_index(
        const std::span<const ReferenceRecord> records,
        const ProjectChapter& project,
        const std::span<const SceneNodeRecord> nodes,
        const std::span<const ReferenceOwnerBinding> additional_bindings) {
        ReverseReferenceIndex result;
        for (const ReferenceRecord& record : records) {
            result.try_emplace(record.uuid);
        }
        const auto append = [&](const ReferenceOwnerBinding& binding)
            -> lfs::Result<void> {
            const auto found = result.find(binding.reference_uuid);
            if (found == result.end()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "A project object refers to a missing REFS row.",
                    std::format("{} field '{}' references absent UUID {}",
                                binding.chapter, binding.field,
                                binding.reference_uuid.to_string()),
                    binding.chapter, binding.field);
            }
            auto& values = found->second;
            if (std::ranges::find(values, binding) == values.end()) {
                values.push_back(binding);
            }
            return {};
        };

        auto dataset = project.dataset_reference();
        if (!dataset) {
            return std::move(dataset).error();
        }
        if (*dataset) {
            if (auto status = append(ReferenceOwnerBinding{
                    .reference_uuid = **dataset,
                    .chapter = "PROJ",
                    .owner_uuid = std::nullopt,
                    .field = "dataset_reference_uuid",
                });
                !status) {
                return std::move(status).error();
            }
        }
        for (const SceneNodeRecord& node : nodes) {
            if (node.payload && node.payload->reference_uuid) {
                if (auto status = append(ReferenceOwnerBinding{
                        .reference_uuid = *node.payload->reference_uuid,
                        .chapter = "SCNG",
                        .owner_uuid = node.uuid,
                        .field = "nodes.payload.reference_uuid",
                    });
                    !status) {
                    return std::move(status).error();
                }
            }
        }
        for (const ReferenceOwnerBinding& binding : additional_bindings) {
            if (binding.reference_uuid.is_nil() || binding.chapter.empty() ||
                binding.field.empty() ||
                (binding.owner_uuid && binding.owner_uuid->is_nil())) {
                return fail<ReverseReferenceIndex>(
                    lfs::ErrorCode::InvalidArgument,
                    "A reverse-reference binding is invalid.",
                    "Binding UUID, chapter, and field must be valid", "REFS",
                    "reverse_owner_index");
            }
            if (auto status = append(binding); !status) {
                return std::move(status).error();
            }
        }
        for (auto& [uuid, bindings] : result) {
            (void)uuid;
            std::ranges::sort(bindings, [](const ReferenceOwnerBinding& lhs,
                                           const ReferenceOwnerBinding& rhs) {
                if (lhs.chapter != rhs.chapter) {
                    return lhs.chapter < rhs.chapter;
                }
                if (lhs.owner_uuid != rhs.owner_uuid) {
                    if (!lhs.owner_uuid) {
                        return true;
                    }
                    if (!rhs.owner_uuid) {
                        return false;
                    }
                    return lhs.owner_uuid->to_string() <
                           rhs.owner_uuid->to_string();
                }
                return lhs.field < rhs.field;
            });
        }
        return result;
    }

    lfs::Result<ReverseReferenceIndex> apply_parameter_snapshot_to_index(
        ReverseReferenceIndex result, const ParameterManagerSnapshot& snapshot) {
        struct PresetReferences {
            std::string_view path;
            const ParameterManagerSnapshot::ReferenceBindings* references;
        };
        const std::array presets{
            PresetReferences{"presets.mcmc.session",
                             &snapshot.mcmc_session_references},
            PresetReferences{"presets.mrnf.session",
                             &snapshot.mrnf_session_references},
            PresetReferences{"presets.igs+.session",
                             &snapshot.igs_session_references},
            PresetReferences{"presets.mcmc.current",
                             &snapshot.mcmc_current_references},
            PresetReferences{"presets.mrnf.current",
                             &snapshot.mrnf_current_references},
            PresetReferences{"presets.igs+.current",
                             &snapshot.igs_current_references},
        };
        const auto append =
            [&](const lfs::core::Uuid& reference_uuid,
                const std::string& field,
                const bool may_target_ppis) -> lfs::Result<void> {
            const auto found = result.find(reference_uuid);
            if (found == result.end()) {
                if (may_target_ppis) {
                    return {};
                }
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "A pending parameter refers to a missing REFS row.",
                    std::format("PRMS field '{}' references absent UUID {}",
                                field, reference_uuid.to_string()),
                    "PRMS", field);
            }
            ReferenceOwnerBinding binding{
                .reference_uuid = reference_uuid,
                .chapter = "PRMS",
                .owner_uuid = std::nullopt,
                .field = field,
            };
            if (std::ranges::find(found->second, binding) ==
                found->second.end()) {
                found->second.push_back(std::move(binding));
            }
            return {};
        };
        for (const auto& preset : presets) {
            if (preset.references->background_image_reference) {
                if (auto status = append(
                        *preset.references->background_image_reference,
                        std::format(
                            "{}.background_image_reference_uuid",
                            preset.path),
                        false);
                    !status) {
                    return std::move(status).error();
                }
            }
            if (preset.references->ppisp_reference) {
                if (auto status = append(
                        *preset.references->ppisp_reference,
                        std::format("{}.ppisp_reference_uuid", preset.path),
                        true);
                    !status) {
                    return std::move(status).error();
                }
            }
        }
        for (auto& [uuid, bindings] : result) {
            (void)uuid;
            std::ranges::sort(
                bindings,
                [](const ReferenceOwnerBinding& lhs,
                   const ReferenceOwnerBinding& rhs) {
                    if (lhs.chapter != rhs.chapter) {
                        return lhs.chapter < rhs.chapter;
                    }
                    if (lhs.owner_uuid != rhs.owner_uuid) {
                        if (!lhs.owner_uuid) {
                            return true;
                        }
                        if (!rhs.owner_uuid) {
                            return false;
                        }
                        return lhs.owner_uuid->bytes <
                               rhs.owner_uuid->bytes;
                    }
                    return lhs.field < rhs.field;
                });
        }
        return result;
    }

    lfs::Result<ReverseReferenceIndex> build_reverse_reference_index(
        const ReferencesChapter& references, const ProjectChapter& project,
        const SceneGraphChapter& scene, const ParametersChapter& parameters,
        const std::span<const ReferenceOwnerBinding> additional_bindings) {
        auto result = build_reverse_reference_index(
            references, project, scene, additional_bindings);
        if (!result) {
            return std::move(result).error();
        }
        auto snapshot = parameters.snapshot();
        if (!snapshot) {
            return std::move(snapshot).error();
        }
        return apply_parameter_snapshot_to_index(
            std::move(*result), *snapshot);
    }

    lfs::Result<ReverseReferenceIndex> build_reverse_reference_index(
        const std::span<const ReferenceRecord> records,
        const ProjectChapter& project,
        const std::span<const SceneNodeRecord> nodes,
        const ParameterManagerSnapshot& parameters,
        const std::span<const ReferenceOwnerBinding> additional_bindings) {
        auto result = build_reverse_reference_index(
            records, project, nodes, additional_bindings);
        if (!result) {
            return std::move(result).error();
        }
        return apply_parameter_snapshot_to_index(
            std::move(*result), parameters);
    }

} // namespace lfs::io::project

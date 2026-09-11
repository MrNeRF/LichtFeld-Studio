/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_envelope.hpp"

#include "core/error.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>

namespace lfs::core {

    namespace {

        constexpr std::size_t kMaxDetailEntries = 8;
        constexpr std::size_t kMaxDetailStringBytes = 256;
        constexpr std::size_t kMessageDefenseBytes = 1024;

        const char* generic_message(const ErrorCode code) noexcept {
            switch (code) {
            case ErrorCode::Cancelled: return "Operation cancelled";
            case ErrorCode::InvalidArgument: return "Invalid argument";
            case ErrorCode::BoundsViolation: return "Index out of bounds";
            case ErrorCode::FailedPrecondition: return "Operation not allowed in the current state";
            case ErrorCode::NotFound: return "Not found";
            case ErrorCode::PermissionDenied: return "Permission denied";
            case ErrorCode::AlreadyExists: return "Already exists";
            case ErrorCode::ResourceExhausted: return "Out of resources";
            case ErrorCode::DeadlineExceeded: return "Operation timed out";
            case ErrorCode::Unavailable: return "Service or device unavailable";
            case ErrorCode::DataLoss: return "Data corrupt or truncated";
            case ErrorCode::Unsupported: return "Not supported";
            case ErrorCode::DeviceLost: return "GPU device lost";
            case ErrorCode::Internal: return "Internal error";
            case ErrorCode::ContractViolation: return "Internal contract violation";
            }
            return "Internal error";
        }

        // truncate_utf8_safe only avoids splitting a sequence at the byte cap; it
        // never validates interior bytes, so a legacy or native string (e.g. a
        // non-UTF-8 filesystem path carried in e.what()) can still hold ill-formed
        // UTF-8. nlohmann::dump throws type_error.316 on such bytes, which would
        // defeat this header's "always valid JSON" contract at every downstream
        // strict dump. Replace every ill-formed byte with U+FFFD so the wire string
        // is total regardless of the caller's dump handler.
        std::string sanitize_utf8(const std::string_view value) {
            std::string out;
            out.reserve(value.size());
            const std::size_t n = value.size();
            std::size_t i = 0;
            while (i < n) {
                const unsigned char lead = static_cast<unsigned char>(value[i]);
                if (lead < 0x80) {
                    out.push_back(static_cast<char>(lead));
                    ++i;
                    continue;
                }
                std::size_t len = 0;
                std::uint32_t cp = 0;
                std::uint32_t min_cp = 0;
                if ((lead & 0xE0) == 0xC0) {
                    len = 2;
                    cp = lead & 0x1F;
                    min_cp = 0x80;
                } else if ((lead & 0xF0) == 0xE0) {
                    len = 3;
                    cp = lead & 0x0F;
                    min_cp = 0x800;
                } else if ((lead & 0xF8) == 0xF0) {
                    len = 4;
                    cp = lead & 0x07;
                    min_cp = 0x10000;
                }

                bool valid = len != 0 && i + len <= n;
                for (std::size_t k = 1; valid && k < len; ++k) {
                    const unsigned char cont = static_cast<unsigned char>(value[i + k]);
                    if ((cont & 0xC0) != 0x80) {
                        valid = false;
                    } else {
                        cp = (cp << 6) | (cont & 0x3F);
                    }
                }
                if (valid && (cp < min_cp || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) {
                    valid = false;
                }

                if (valid) {
                    out.append(value.substr(i, len));
                    i += len;
                } else {
                    out.append("\xEF\xBF\xBD");
                    ++i;
                }
            }
            return out;
        }

        constexpr std::array<std::string_view, 7> kAllowlistedFields = {
            "path", "command", "parameter", "format", "iteration",
            "requested_bytes", "available_bytes"};

        [[nodiscard]] bool is_allowlisted(const std::string_view key) noexcept {
            for (const std::string_view allowed : kAllowlistedFields) {
                if (allowed == key) {
                    return true;
                }
            }
            return false;
        }

        // Adds one allowlisted field to `details`, transforming per Section 1.4:
        // `path` collapses to its filename, string values truncate to 256 bytes,
        // everything else crosses verbatim. Returns false when nothing was added.
        [[nodiscard]] bool add_allowlisted_field(nlohmann::json& details, const std::string& key,
                                                 const SmallFields::Value& value) {
            if (key == "path") {
                const auto* path = std::get_if<std::string>(&value);
                if (!path) {
                    return false;
                }
                details[key] = sanitize_utf8(std::filesystem::path(*path).filename().string());
                return true;
            }
            return std::visit(
                [&details, &key]<class V>(const V& held) -> bool {
                    if constexpr (std::is_same_v<V, std::monostate>) {
                        return false;
                    } else if constexpr (std::is_same_v<V, std::string>) {
                        details[key] = truncate_utf8_safe(sanitize_utf8(held), kMaxDetailStringBytes);
                        return true;
                    } else {
                        details[key] = held;
                        return true;
                    }
                },
                value);
        }

        nlohmann::json extract_details(const Error& error) {
            nlohmann::json details = nlohmann::json::object();
            std::size_t count = 0;
            for (const ErrorFrame& frame : error.frames()) {
                for (const auto& entry : frame.fields.entries()) {
                    if (count >= kMaxDetailEntries) {
                        break;
                    }
                    if (!is_allowlisted(entry.key) || details.contains(entry.key)) {
                        continue;
                    }
                    if (add_allowlisted_field(details, entry.key, entry.value)) {
                        ++count;
                    }
                }
                if (count >= kMaxDetailEntries) {
                    break;
                }
            }

            if (const auto& native = error.native(); native.has_value()) {
                details["native_name"] = truncate_utf8_safe(sanitize_utf8(native->name), kMaxDetailStringBytes);
                details["native_code"] = native->code;
            }
            return details;
        }

        std::size_t serialized_size(const nlohmann::json& j) {
            return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace).size();
        }

    } // namespace

    void to_json(nlohmann::json& j, const WireError& e) {
        j = nlohmann::json{
            {"code", e.code},
            {"domain", e.domain},
            {"message", e.message},
            {"retryable", e.retryable},
            {"operation_id", e.operation_id},
        };
    }

    WireError to_wire_error(const Error& error) {
        WireError wire;
        wire.code = to_string(error.code());
        wire.domain = to_string(error.domain());
        const std::string_view user_message = error.user_message();
        wire.message = user_message.empty()
                           ? generic_message(error.code())
                           : truncate_utf8_safe(sanitize_utf8(user_message), kMaxDeveloperStringBytes);
        wire.operation_id = error.operation_id().value();
        wire.retryable = error.retryability() != Retryability::NotRetryable;
        return wire;
    }

    nlohmann::json to_wire_envelope(const Error& error) {
        nlohmann::json envelope = to_wire_error(error);
        if (nlohmann::json details = extract_details(error); !details.empty()) {
            envelope["details"] = std::move(details);
        }

        if (serialized_size(envelope) <= kMaxSerializedErrorBytes) {
            return envelope;
        }
        envelope.erase("details");
        if (serialized_size(envelope) > kMaxSerializedErrorBytes) {
            envelope["message"] = truncate_utf8_safe(
                envelope["message"].get<std::string>(), kMessageDefenseBytes);
        }
        return envelope;
    }

} // namespace lfs::core

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace lfs::core::detail {

    template <typename... Args>
    class CudaSafeFormatString {
    public:
        template <size_t N>
        consteval CudaSafeFormatString(const char (&format)[N])
            : format_(format, N - 1) {
            size_t field_count = 0;
            for (size_t index = 0; index + 1 < N; ++index) {
                if (format[index] == '{') {
                    if (format[index + 1] != '}') {
                        throw "CUDA diagnostic formatting supports only '{}' fields";
                    }
                    ++field_count;
                    ++index;
                } else if (format[index] == '}') {
                    throw "unmatched '}' in CUDA diagnostic format string";
                }
            }
            if (field_count != sizeof...(Args)) {
                throw "CUDA diagnostic format argument count mismatch";
            }
        }

        [[nodiscard]] constexpr std::string_view get() const noexcept {
            return format_;
        }

    private:
        std::string_view format_;
    };

    // CUDA-parsed diagnostics use only default "{}" fields. Keep this
    // formatter scalar-only so nvcc never has to lower std::format or a helper
    // that returns formatting state as an aggregate.
    template <typename T>
        requires std::is_unsigned_v<T>
    void append_cuda_safe_unsigned(std::string& output,
                                   T value,
                                   const T base = 10) {
        constexpr char DIGITS[] = "0123456789abcdef";
        char buffer[3 * sizeof(T) + 1];
        char* const end = buffer + sizeof(buffer);
        char* cursor = end;
        do {
            *--cursor = DIGITS[value % base];
            value /= base;
        } while (value != 0);
        output.append(cursor, end);
    }

    inline void append_cuda_safe_argument(std::string& output,
                                          const std::string_view value) {
        output.append(value);
    }

    inline void append_cuda_safe_argument(std::string& output,
                                          const std::string& value) {
        output.append(value);
    }

    inline void append_cuda_safe_argument(std::string& output,
                                          const char* value) {
        output.append(value != nullptr ? value : "(null)");
    }

    inline void append_cuda_safe_argument(std::string& output,
                                          char* value) {
        append_cuda_safe_argument(output, static_cast<const char*>(value));
    }

    inline void append_cuda_safe_argument(std::string& output,
                                          const bool value) {
        output.append(value ? "true" : "false");
    }

    inline void append_cuda_safe_argument(std::string& output,
                                          std::nullptr_t) {
        output.append("0x0");
    }

    template <typename T>
        requires(std::is_integral_v<std::remove_cvref_t<T>> &&
                 !std::is_same_v<std::remove_cvref_t<T>, bool>)
    void append_cuda_safe_argument(std::string& output, const T value) {
        using Value = std::remove_cvref_t<T>;
        using Unsigned = std::make_unsigned_t<Value>;
        if constexpr (std::is_same_v<Value, char>) {
            output.push_back(value);
        } else if constexpr (std::is_signed_v<Value>) {
            const Unsigned unsigned_value = static_cast<Unsigned>(value);
            if (value < 0) {
                output.push_back('-');
                append_cuda_safe_unsigned(output, Unsigned{0} - unsigned_value);
            } else {
                append_cuda_safe_unsigned(output, unsigned_value);
            }
        } else {
            append_cuda_safe_unsigned(output, static_cast<Unsigned>(value));
        }
    }

    template <typename T>
        requires(std::is_pointer_v<std::remove_cvref_t<T>> &&
                 !std::is_same_v<std::remove_cvref_t<T>, const char*> &&
                 !std::is_same_v<std::remove_cvref_t<T>, char*>)
    void append_cuda_safe_argument(std::string& output, const T value) {
        output.append("0x");
        append_cuda_safe_unsigned(
            output, reinterpret_cast<uintptr_t>(value), uintptr_t{16});
    }

    inline void format_cuda_safe_into(std::string& output,
                                      const std::string_view format) {
        if (format.find("{}") != std::string_view::npos) {
            throw std::invalid_argument("too few CUDA diagnostic arguments");
        }
        output.append(format);
    }

    template <typename Argument, typename... Args>
    void format_cuda_safe_into(std::string& output,
                               const std::string_view format,
                               Argument&& argument,
                               Args&&... remaining) {
        const size_t placeholder = format.find("{}");
        if (placeholder == std::string_view::npos) {
            throw std::invalid_argument("too many CUDA diagnostic arguments");
        }

        output.append(format.substr(0, placeholder));
        append_cuda_safe_argument(output, std::forward<Argument>(argument));
        format_cuda_safe_into(
            output, format.substr(placeholder + 2), std::forward<Args>(remaining)...);
    }

    template <typename... Args>
    [[nodiscard]] std::string format_cuda_safe(
        CudaSafeFormatString<std::type_identity_t<Args>...> format,
        Args&&... args) {
        const std::string_view format_view = format.get();
        std::string output;
        output.reserve(format_view.size() + sizeof...(Args) * 8);
        format_cuda_safe_into(
            output, format_view, std::forward<Args>(args)...);
        return output;
    }

} // namespace lfs::core::detail

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <filesystem>
#include <fstream>
#include <iterator>
#include <system_error>
#include <type_traits>
#include <utility>

namespace lfs::core::fs {
    namespace fs = lfs::core::fs;
    using namespace std::filesystem;

    // This representation belongs only at an OS I/O boundary, never in persisted
    // project references, UI text, URLs, or command-line arguments to other apps.
    [[nodiscard]] inline path native_path(const path& value, const bool force_extended = false) {
#ifdef _WIN32
        if (value.empty())
            return value;
        const auto& name = value.native();
        if (name.starts_with(L"\\\\?\\") || name.starts_with(L"\\\\.\\"))
            return value;
        std::error_code error;
        auto absolute = std::filesystem::absolute(value, error);
        if (error || (!force_extended && absolute.native().size() < 248))
            return value;
        absolute = absolute.lexically_normal();
        absolute.make_preferred();
        const auto& normalized = absolute.native();
        if (normalized.starts_with(L"\\\\"))
            return path(L"\\\\?\\UNC\\" + normalized.substr(2));
        return path(L"\\\\?\\" + normalized);
#else
        (void)force_extended;
        return value;
#endif
    }

    [[nodiscard]] inline path display_path(const path& value) {
#ifdef _WIN32
        const auto& name = value.native();
        if (name.starts_with(L"\\\\?\\UNC\\"))
            return path(L"\\\\" + name.substr(8));
        if (name.starts_with(L"\\\\?\\") && name.size() >= 7 && name[5] == L':')
            return path(name.substr(4));
#endif
        return value;
    }
    template <typename... Args>
    inline auto exists(const path& value, Args&&... args) {
        return std::filesystem::exists(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto is_regular_file(const path& value, Args&&... args) {
        return std::filesystem::is_regular_file(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto is_directory(const path& value, Args&&... args) {
        return std::filesystem::is_directory(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto is_symlink(const path& value, Args&&... args) {
        return std::filesystem::is_symlink(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto is_block_file(const path& value, Args&&... args) {
        return std::filesystem::is_block_file(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto is_character_file(const path& value, Args&&... args) {
        return std::filesystem::is_character_file(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto is_fifo(const path& value, Args&&... args) {
        return std::filesystem::is_fifo(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto is_socket(const path& value, Args&&... args) {
        return std::filesystem::is_socket(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto is_other(const path& value, Args&&... args) {
        return std::filesystem::is_other(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto is_empty(const path& value, Args&&... args) {
        return std::filesystem::is_empty(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto status(const path& value, Args&&... args) {
        return std::filesystem::status(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto symlink_status(const path& value, Args&&... args) {
        return std::filesystem::symlink_status(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto file_size(const path& value, Args&&... args) {
        return std::filesystem::file_size(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto hard_link_count(const path& value, Args&&... args) {
        return std::filesystem::hard_link_count(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto space(const path& value, Args&&... args) {
        return std::filesystem::space(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto remove(const path& value, Args&&... args) {
        return std::filesystem::remove(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto remove_all(const path& value, Args&&... args) {
        return std::filesystem::remove_all(native_path(value, true), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto create_directories(const path& value, Args&&... args) {
        return std::filesystem::create_directories(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto resize_file(const path& value, Args&&... args) {
        return std::filesystem::resize_file(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto permissions(const path& value, Args&&... args) {
        return std::filesystem::permissions(native_path(value), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto last_write_time(const path& value, Args&&... args) {
        return std::filesystem::last_write_time(native_path(value), std::forward<Args>(args)...);
    }
    inline bool exists(file_status value) noexcept { return std::filesystem::exists(value); }
    inline bool is_regular_file(file_status value) noexcept { return std::filesystem::is_regular_file(value); }
    inline bool is_directory(file_status value) noexcept { return std::filesystem::is_directory(value); }
    inline bool is_symlink(file_status value) noexcept { return std::filesystem::is_symlink(value); }
    inline bool is_block_file(file_status value) noexcept { return std::filesystem::is_block_file(value); }
    inline bool is_character_file(file_status value) noexcept { return std::filesystem::is_character_file(value); }
    inline bool is_fifo(file_status value) noexcept { return std::filesystem::is_fifo(value); }
    inline bool is_socket(file_status value) noexcept { return std::filesystem::is_socket(value); }
    inline bool is_other(file_status value) noexcept { return std::filesystem::is_other(value); }
    template <typename... Args>
    inline auto copy_file(const path& source, const path& destination, Args&&... args) {
        return std::filesystem::copy_file(native_path(source), native_path(destination), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto copy(const path& source, const path& destination, Args&&... args) {
        return std::filesystem::copy(native_path(source, true), native_path(destination, true), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto copy_symlink(const path& source, const path& destination, Args&&... args) {
        return std::filesystem::copy_symlink(native_path(source), native_path(destination), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto rename(const path& source, const path& destination, Args&&... args) {
        return std::filesystem::rename(native_path(source), native_path(destination), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto equivalent(const path& source, const path& destination, Args&&... args) {
        return std::filesystem::equivalent(native_path(source), native_path(destination), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto create_hard_link(const path& source, const path& destination, Args&&... args) {
        return std::filesystem::create_hard_link(native_path(source), native_path(destination), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto create_symlink(const path& source, const path& destination, Args&&... args) {
        return std::filesystem::create_symlink(source, native_path(destination), std::forward<Args>(args)...);
    }
    template <typename... Args>
    inline auto create_directory_symlink(const path& source, const path& destination, Args&&... args) {
        return std::filesystem::create_directory_symlink(source, native_path(destination), std::forward<Args>(args)...);
    }

    inline bool create_directory(const path& value) { return std::filesystem::create_directory(native_path(value)); }
    inline bool create_directory(const path& value, std::error_code& error) { return std::filesystem::create_directory(native_path(value), error); }
    inline bool create_directory(const path& value, const path& attributes) { return std::filesystem::create_directory(native_path(value), native_path(attributes)); }
    inline bool create_directory(const path& value, const path& attributes, std::error_code& error) { return std::filesystem::create_directory(native_path(value), native_path(attributes), error); }

    inline path canonical(const path& value, std::error_code& error) {
        return display_path(std::filesystem::canonical(native_path(value), error));
    }
    inline path canonical(const path& value) { return display_path(std::filesystem::canonical(native_path(value))); }
    inline path weakly_canonical(const path& value, std::error_code& error) {
#ifdef _WIN32
        auto existing = native_path(std::filesystem::absolute(value, error));
        if (error)
            return {};
        path missing;
        while (!std::filesystem::exists(existing, error)) {
            if (error)
                return {};
            if (!existing.has_relative_path()) {
                error = std::make_error_code(std::errc::no_such_file_or_directory);
                return {};
            }
            missing = missing.empty() ? existing.filename() : existing.filename() / missing;
            existing = existing.parent_path();
        }
        auto resolved = std::filesystem::canonical(existing, error);
        if (error)
            return {};
        if (!missing.empty())
            resolved /= missing;
        return display_path(resolved.lexically_normal());
#else
        return std::filesystem::weakly_canonical(value, error);
#endif
    }
    inline path weakly_canonical(const path& value) {
        std::error_code error;
        auto result = lfs::core::fs::weakly_canonical(value, error);
        if (error)
            throw filesystem_error("weakly_canonical", value, error);
        return result;
    }
    inline path relative(const path& value, const path& base, std::error_code& error) {
        auto target = lfs::core::fs::weakly_canonical(value, error);
        if (error)
            return {};
        auto origin = lfs::core::fs::weakly_canonical(base, error);
        return error ? path{} : target.lexically_relative(origin);
    }
    inline path relative(const path& value, const path& base = std::filesystem::current_path()) {
        std::error_code error;
        auto result = lfs::core::fs::relative(value, base, error);
        if (error)
            throw filesystem_error("relative", value, base, error);
        return result;
    }
    inline path read_symlink(const path& value) { return display_path(std::filesystem::read_symlink(native_path(value))); }
    inline path read_symlink(const path& value, std::error_code& error) { return display_path(std::filesystem::read_symlink(native_path(value), error)); }
    inline path current_path() { return display_path(std::filesystem::current_path()); }
    inline path current_path(std::error_code& error) { return display_path(std::filesystem::current_path(error)); }
    inline void current_path(const path& value) { std::filesystem::current_path(native_path(value)); }
    inline void current_path(const path& value, std::error_code& error) { std::filesystem::current_path(native_path(value), error); }

    template <typename Stream, std::ios_base::openmode DefaultMode>
    class native_file_stream : public Stream {
    public:
        native_file_stream() = default;
        native_file_stream(native_file_stream&&) = default;
        native_file_stream& operator=(native_file_stream&&) = default;
        explicit native_file_stream(const fs::path& value, std::ios_base::openmode mode = DefaultMode) {
            open(value, mode);
        }
        void open(const fs::path& value, std::ios_base::openmode mode = DefaultMode) {
            Stream::open(native_path(value), mode);
        }
    };
    using ifstream = native_file_stream<std::ifstream, std::ios::in>;
    using ofstream = native_file_stream<std::ofstream, std::ios::out>;
    using fstream = native_file_stream<std::fstream, std::ios::in | std::ios::out>;

    class directory_entry : public std::filesystem::directory_entry {
        fs::path display_;

    public:
        directory_entry() = default;
        explicit directory_entry(const std::filesystem::directory_entry& entry)
            : std::filesystem::directory_entry(entry), display_(display_path(entry.path())) {}
        explicit directory_entry(const fs::path& value)
            : std::filesystem::directory_entry(native_path(value)), display_(display_path(value)) {}
        directory_entry(const fs::path& value, std::error_code& error)
            : std::filesystem::directory_entry(native_path(value), error), display_(display_path(value)) {}
        const fs::path& path() const noexcept { return display_; }
        operator const fs::path&() const noexcept { return display_; }
        void assign(const fs::path& value) {
            std::filesystem::directory_entry::assign(native_path(value));
            display_ = display_path(value);
        }
        void assign(const fs::path& value, std::error_code& error) {
            std::filesystem::directory_entry::assign(native_path(value), error);
            display_ = display_path(value);
        }
        void replace_filename(const fs::path& value) { assign(display_.parent_path() / value); }
        void replace_filename(const fs::path& value, std::error_code& error) { assign(display_.parent_path() / value, error); }
    };

    template <bool Recursive>
    class basic_directory_iterator {
        using Native = std::conditional_t<Recursive, std::filesystem::recursive_directory_iterator, std::filesystem::directory_iterator>;
        Native iterator_;
        mutable directory_entry entry_;

    public:
        using value_type = directory_entry;
        using difference_type = std::ptrdiff_t;
        using pointer = const directory_entry*;
        using reference = const directory_entry&;
        using iterator_category = std::input_iterator_tag;
        basic_directory_iterator() = default;
        template <typename... Args>
        explicit basic_directory_iterator(const fs::path& value, Args&&... args)
            : iterator_(native_path(value, true), std::forward<Args>(args)...) {}
        reference operator*() const {
            entry_ = directory_entry(*iterator_);
            return entry_;
        }
        pointer operator->() const { return &operator*(); }
        basic_directory_iterator& operator++() {
            ++iterator_;
            return *this;
        }
        struct postincrement_proxy {
            directory_entry entry;
            const directory_entry& operator*() const { return entry; }
        };
        postincrement_proxy operator++(int) {
            postincrement_proxy previous{operator*()};
            ++*this;
            return previous;
        }
        basic_directory_iterator& increment(std::error_code& error) {
            iterator_.increment(error);
            return *this;
        }
        friend bool operator==(const basic_directory_iterator& left, const basic_directory_iterator& right) { return left.iterator_ == right.iterator_; }
        directory_options options() const noexcept
            requires Recursive
        { return iterator_.options(); }
        int depth() const noexcept
            requires Recursive
        { return iterator_.depth(); }
        bool recursion_pending() const noexcept
            requires Recursive
        { return iterator_.recursion_pending(); }
        void disable_recursion_pending() noexcept
            requires Recursive
        { iterator_.disable_recursion_pending(); }
        void pop()
            requires Recursive
        { iterator_.pop(); }
        void pop(std::error_code& error)
            requires Recursive
        { iterator_.pop(error); }
    };
    using directory_iterator = basic_directory_iterator<false>;
    using recursive_directory_iterator = basic_directory_iterator<true>;
    template <bool R>
    inline basic_directory_iterator<R> begin(basic_directory_iterator<R> value) { return value; }
    template <bool R>
    inline basic_directory_iterator<R> end(basic_directory_iterator<R>) { return {}; }
} // namespace lfs::core::fs

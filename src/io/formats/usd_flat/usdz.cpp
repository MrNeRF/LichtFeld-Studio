/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "usdz.hpp"
#include "crate.hpp"
#include "usda_read.hpp"
#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

namespace lfs::io::usd_flat {

    lfs::Result<FlatStage> read_usdz(const std::filesystem::path& path) {
        archive* archive = archive_read_new();
        if (!archive) {
            return make_flat_error("Failed to allocate USDZ reader");
        }
        if (archive_read_support_format_zip(archive) != ARCHIVE_OK ||
            archive_read_support_filter_all(archive) != ARCHIVE_OK) {
            const char* detail = archive_error_string(archive);
            const std::string error = detail ? detail : "unknown error";
            archive_read_free(archive);
            return make_flat_error("Failed to configure USDZ archive reader: " + error);
        }
        const auto cleanup = [&]() {
            archive_read_free(archive);
        };
        int open_result = ARCHIVE_FAILED;
#ifdef _WIN32
        open_result = archive_read_open_filename_w(archive, path.wstring().c_str(), 1024 * 1024);
#else
        open_result = archive_read_open_filename(archive, path.c_str(), 1024 * 1024);
#endif
        if (open_result != ARCHIVE_OK) {
            const std::string error = archive_error_string(archive) ? archive_error_string(archive) : "unknown error";
            cleanup();
            return make_flat_error("Failed to open USDZ file: " + error);
        }

        std::vector<std::uint8_t> bytes;
        archive_entry* entry = nullptr;
        while (archive_read_next_header(archive, &entry) == ARCHIVE_OK) {
            const char* name = archive_entry_pathname(entry);
            if (!name) {
                archive_read_data_skip(archive);
                continue;
            }
            std::string candidate(name);
            std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](const unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            const auto has_suffix = [&candidate](std::string_view suffix) {
                return candidate.size() >= suffix.size() &&
                       candidate.compare(candidate.size() - suffix.size(), suffix.size(), suffix) == 0;
            };
            if (!has_suffix(".usda") && !has_suffix(".usdc") && !has_suffix(".usd")) {
                archive_read_data_skip(archive);
                continue;
            }
            const la_int64_t size = archive_entry_size(entry);
            if (size < 0 || static_cast<std::uint64_t>(size) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                cleanup();
                return make_flat_error("Invalid root USDZ entry size");
            }
            bytes.resize(static_cast<std::size_t>(size));
            std::size_t offset = 0;
            while (offset < bytes.size()) {
                const la_ssize_t read = archive_read_data(archive, bytes.data() + offset, bytes.size() - offset);
                if (read <= 0) {
                    cleanup();
                    return make_flat_error("Failed to read root USDZ entry");
                }
                offset += static_cast<std::size_t>(read);
            }
            break;
        }
        archive_read_free(archive);
        if (bytes.empty()) {
            return make_flat_error("USDZ archive has no root USD file");
        }

        static constexpr std::array<std::uint8_t, 8> crate_magic = {'P', 'X', 'R', '-', 'U', 'S', 'D', 'C'};
        if (bytes.size() >= crate_magic.size() && std::equal(crate_magic.begin(), crate_magic.end(), bytes.begin())) {
            return read_usdc_bytes(bytes.data(), bytes.size());
        }
        if (bytes.size() >= 5 && std::string_view(reinterpret_cast<const char*>(bytes.data()), 5) == "#usda") {
            return read_usda_bytes(bytes.data(), bytes.size());
        }
        return make_flat_error("Unsupported USDZ root: expected a crate or #usda header");
    }

} // namespace lfs::io::usd_flat

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Re-export public API
#include "io/exporter.hpp"

namespace lfs::io {

    struct SogEncodeOptions : SogSaveOptions {
        bool presorted = false;
        // Lossless pixels, lower compression effort for streamed units.
        bool fast_webp = false;
    };

    class SogSink {
    public:
        virtual ~SogSink() = default;
        virtual Result<void> open() { return {}; }
        virtual Result<void> add_file(const std::string& name, const void* data, size_t size) = 0;
        virtual Result<void> close() = 0;
    };

    Result<void> encode_sog(const SplatData&, const SogEncodeOptions&, SogSink&);
    Result<void> encode_sog_directory(const SplatData&, const SogEncodeOptions&);
    // CPU-only I/O and WebP decode. Invoke the returned closure on the owning CUDA thread.
    using SogDirectoryReconstruct = std::move_only_function<Result<SplatData>()>;
    // Bounded entry access shared by directory and bundle readers.
    using SogEntryReader = std::function<Result<std::vector<uint8_t>>(const std::string&, size_t)>;
    Result<SogDirectoryReconstruct> prepare_sog_entries(const SogEntryReader&, const std::string& prefix);
    std::unique_ptr<SogSink> make_sog_archive(const std::filesystem::path&);
    Result<SogDirectoryReconstruct> prepare_sog_directory(const std::filesystem::path&);
    Result<SplatData> read_sog_directory(const std::filesystem::path&);

    // Internal: Loading function (not in public API)
    Result<SplatData> load_sog(const std::filesystem::path& filepath);

} // namespace lfs::io

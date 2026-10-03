/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_exr.hpp"
#include "core/path_utils.hpp"

#include <OpenEXR/openexr.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>

namespace lfs::core::image_codecs {
    namespace {
        // OpenEXR delivers diagnostics on the thread that caused the error.
        // Scope the destination to this context so nested readers remain independent.
        thread_local exr_const_context_t diagnostic_context = nullptr;
        thread_local std::string* diagnostic_error = nullptr;
        struct DiagnosticScope {
            exr_const_context_t previous_context = diagnostic_context;
            std::string* previous_error = diagnostic_error;
            DiagnosticScope(exr_const_context_t context, std::string& error) {
                diagnostic_context = context;
                diagnostic_error = &error;
            }
            ~DiagnosticScope() {
                diagnostic_context = previous_context;
                diagnostic_error = previous_error;
            }
        };

        struct ParallelBudget {
            std::mutex mutex;
            unsigned used_workers = 0;
            size_t used_bytes = 0;
            const unsigned capacity = std::clamp(std::thread::hardware_concurrency(), 1u, 8u) - 1;
            static constexpr size_t memory = size_t{64} * 1024 * 1024;
            static constexpr size_t overhead = size_t{1024} * 1024;
        };
        ParallelBudget& parallel_budget() {
            // Loader threads can still be joining during static destruction.
            // Keep this small process-wide reservation state alive until exit.
            static auto* budget = new ParallelBudget;
            return *budget;
        }

        // Non-blocking reservations bound extra workers across simultaneous files.
        // Reserve a conservative scratch estimate; large chunks stay sequential.
        struct WorkerLease {
            unsigned count = 0;
            size_t bytes = 0;
            WorkerLease(unsigned requested, uint64_t chunk_bytes) {
                if (!requested || chunk_bytes > (ParallelBudget::memory - ParallelBudget::overhead) / 4)
                    return;
                const size_t per_worker = size_t(chunk_bytes) * 4 + ParallelBudget::overhead;
                auto& budget = parallel_budget();
                std::lock_guard lock(budget.mutex);
                count = std::min({requested, budget.capacity - budget.used_workers,
                                  unsigned((ParallelBudget::memory / 2) / per_worker),
                                  unsigned((ParallelBudget::memory - budget.used_bytes) / per_worker)});
                bytes = count * per_worker;
                budget.used_workers += count;
                budget.used_bytes += bytes;
            }
            ~WorkerLease() {
                if (!count)
                    return;
                auto& budget = parallel_budget();
                std::lock_guard lock(budget.mutex);
                budget.used_workers -= count;
                budget.used_bytes -= bytes;
            }
            WorkerLease(const WorkerLease&) = delete;
            WorkerLease& operator=(const WorkerLease&) = delete;
        };

        // Context and pipeline own all OpenEXR allocations, including on errors.
        struct Context {
            exr_context_t value = nullptr;
            ~Context() {
                if (value)
                    exr_finish(&value);
            }
        };

        struct Pipeline {
            exr_decode_pipeline_t value = EXR_DECODE_PIPELINE_INITIALIZER;
            exr_const_context_t context;
            ~Pipeline() { exr_decoding_destroy(context, &value); }
        };

        bool check(exr_result_t status, std::string& error) {
            if (status == EXR_ERR_SUCCESS) {
                error.clear();
                return true;
            }
            const auto detail = std::move(error);
            error = std::string("EXR: ") + exr_get_error_code_as_string(status);
            if (!detail.empty())
                error += ": " + detail;
            return false;
        }

        struct Header {
            exr_attr_box2i_t window{};
            exr_storage_t storage{};
            int width = 0;
            int height = 0;
            bool grayscale = false;
            int source_channels = 0;
            std::array<int, 4> channels{-1, -1, -1, -1};
        };

        bool open(const std::filesystem::path& path, Context& context, Header& header,
                  std::string& error) {
            exr_context_initializer_t init = EXR_DEFAULT_CONTEXT_INITIALIZER;
            // Keep the library's useful details without printing or throwing through C.
            init.user_data = &error;
            init.error_handler_fn = [](exr_const_context_t ctxt, exr_result_t, const char* message) noexcept {
                void* user = nullptr;
                if (ctxt == diagnostic_context)
                    user = diagnostic_error;
                else
                    exr_get_user_data(ctxt, &user);
                if (message && user) {
                    try {
                        *static_cast<std::string*>(user) = message;
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): the result code still reports the error when
                        // storing its detail fails.
                        static_cast<std::string*>(user)->clear();
                    }
                }
            };
            const auto name = path_to_utf8(path);
            if (!check(exr_start_read(&context.value, name.c_str(), &init), error) ||
                !check(exr_get_storage(context.value, 0, &header.storage), error) ||
                !check(exr_get_data_window(context.value, 0, &header.window), error))
                return false;
            if (header.storage != EXR_STORAGE_SCANLINE && header.storage != EXR_STORAGE_TILED) {
                error = "EXR: deep images are not supported";
                return false;
            }
            const auto width = int64_t(header.window.max.x) - header.window.min.x + 1;
            const auto height = int64_t(header.window.max.y) - header.window.min.y + 1;
            // OpenEXR's output line stride is int32_t. Validate before any allocation.
            if (width <= 0 || height <= 0 || width > int64_t(std::numeric_limits<int32_t>::max() / (4 * sizeof(float))) ||
                height > std::numeric_limits<int>::max() ||
                uint64_t(width) * uint64_t(height) > std::numeric_limits<size_t>::max() / (4 * sizeof(float))) {
                error = "EXR: invalid or oversized data window";
                return false;
            }
            header.width = static_cast<int>(width);
            header.height = static_cast<int>(height);
            const exr_attr_chlist_t* channels = nullptr;
            if (!check(exr_get_channels(context.value, 0, &channels), error))
                return false;
            // Preserve LoadEXR's contract: a single channel is replicated to RGBA;
            // otherwise select the unlayered RGB channels and optional alpha.
            header.grayscale = channels->num_channels == 1;
            constexpr std::array<const char*, 4> names{"R", "G", "B", "A"};
            for (int i = 0; i < channels->num_channels; ++i) {
                for (int c = 0; c < 4; ++c) {
                    if (std::strcmp(channels->entries[i].name.str, names[c]) == 0)
                        header.channels[c] = i;
                }
            }
            if (header.grayscale)
                header.channels = {0, -1, -1, -1};
            if (!header.grayscale && (header.channels[0] < 0 || header.channels[1] < 0 || header.channels[2] < 0)) {
                error = "EXR: RGB channels not found";
                return false;
            }
            // Auxiliary channels such as Z do not make an RGB camera have alpha.
            header.source_channels = header.grayscale ? 1 : header.channels[3] >= 0 ? 4
                                                                                    : 3;
            for (const int index : header.channels) {
                if (index >= 0 && (channels->entries[index].x_sampling != 1 || channels->entries[index].y_sampling != 1)) {
                    error = "EXR: subsampled RGB/alpha channels are not supported";
                    return false;
                }
            }
            return true;
        }
    } // namespace

    bool probe_exr(const std::filesystem::path& path, Probe& result, std::string& error) {
        error.clear();
        Context context;
        Header header;
        if (!open(path, context, header, error))
            return false;
        // Header only: do not read the chunk table or decompress pixels for camera import.
        result = {header.width, header.height, header.source_channels, SampleType::Float32};
        return true;
    }

    bool decode_exr(const std::filesystem::path& path, Image& result, std::string& error, unsigned max_workers) {
        error.clear();
        Context context;
        Header header;
        if (!open(path, context, header, error))
            return false;
        std::error_code file_error;
        const auto file_size = std::filesystem::file_size(path, file_error);
        if (file_error) {
            error = "EXR: unable to read file size: " + file_error.message();
            return false;
        }
        int32_t chunk_count = 0;
        uint64_t table_offset = 0;
        if (!check(exr_get_chunk_count(context.value, 0, &chunk_count), error) ||
            !check(exr_get_chunk_table_offset(context.value, 0, &table_offset), error))
            return false;
        // A real file must contain its offset table. Check before the library or
        // the output buffer allocates from dimensions in a potentially damaged header.
        // Do not impose a compression-ratio limit: uniform HDR images compress well.
        if (chunk_count <= 0 || table_offset > file_size ||
            uint64_t(chunk_count) > (file_size - table_offset) / sizeof(uint64_t)) {
            error = "EXR: data window requires a chunk table larger than the file";
            return false;
        }
        struct Chunk {
            exr_chunk_info_t info;
            int x, y;
        };
        struct Extent {
            int width, height;
        };
        std::vector<Chunk> chunks;
        uint64_t largest_chunk = 0;
        auto add_chunk = [&](const exr_chunk_info_t& chunk, int x, int y, Extent extent) {
            // Exact extents ensure worker destinations never overlap, including
            // partial edge tiles and the final scanline block.
            if (chunk.width != extent.width || chunk.height != extent.height) {
                error = "EXR: chunk outside data window";
                return false;
            }
            if (chunk.data_offset > file_size || chunk.packed_size > file_size - chunk.data_offset ||
                (chunk.compression == EXR_COMPRESSION_NONE && chunk.packed_size != chunk.unpacked_size)) {
                error = "EXR: chunk size is inconsistent with the file or data window";
                return false;
            }
            largest_chunk = std::max({largest_chunk, chunk.packed_size, chunk.unpacked_size});
            chunks.push_back({chunk, x, y});
            return true;
        };
        if (header.storage == EXR_STORAGE_SCANLINE) {
            int32_t lines = 0;
            if (!check(exr_get_scanlines_per_chunk(context.value, 0, &lines), error))
                return false;
            if (lines <= 0) {
                error = "EXR: invalid scanline chunk size";
                return false;
            }
            for (int64_t y = 0; y < header.height; y += lines) {
                exr_chunk_info_t chunk{};
                if (!check(exr_read_scanline_chunk_info(context.value, 0, static_cast<int>(header.window.min.y + y), &chunk), error) ||
                    !add_chunk(chunk, 0, static_cast<int>(y), {header.width, static_cast<int>(std::min<int64_t>(lines, header.height - y))}))
                    return false;
            }
        } else {
            int32_t tile_width = 0, tile_height = 0;
            if (!check(exr_get_tile_sizes(context.value, 0, 0, 0, &tile_width, &tile_height), error))
                return false;
            if (tile_width <= 0 || tile_height <= 0) {
                error = "EXR: invalid tile size";
                return false;
            }
            // As with TinyEXR, use the full-resolution level of mip/ripmap images.
            for (int64_t y = 0, ty = 0; y < header.height; y += tile_height, ++ty) {
                for (int64_t x = 0, tx = 0; x < header.width; x += tile_width, ++tx) {
                    exr_chunk_info_t chunk{};
                    if (!check(exr_read_tile_chunk_info(context.value, 0, static_cast<int>(tx), static_cast<int>(ty), 0, 0, &chunk), error) ||
                        !add_chunk(chunk, static_cast<int>(x), static_cast<int>(y), {static_cast<int>(std::min<int64_t>(tile_width, header.width - x)), static_cast<int>(std::min<int64_t>(tile_height, header.height - y))}))
                        return false;
                }
            }
        }
        // Validate every destination and payload before allocating or starting workers.
        Image image;
        image.width = header.width;
        image.height = header.height;
        image.channels = 4;
        image.sample_type = SampleType::Float32;
        image.data.resize(size_t(header.width) * header.height * 4 * sizeof(float));
        auto* pixels = reinterpret_cast<float*>(image.data.data());
        if (!header.grayscale && header.channels[3] < 0) {
            for (size_t i = 0; i < image.data.size() / sizeof(float); i += 4)
                pixels[i + 3] = 1.0f;
        }

        const bool automatic = max_workers == 0;
        const bool parallel = !automatic || (image.data.size() >= size_t{16} * 1024 * 1024 &&
                                             chunks.front().info.compression != EXR_COMPRESSION_NONE);
        const unsigned requested = parallel ? static_cast<unsigned>(std::min<size_t>({chunks.size(), 8, automatic ? 4 : max_workers})) : 1;
        WorkerLease lease(requested - 1, largest_chunk);
        struct Worker {
            std::string error;
            std::exception_ptr exception;
        };
        std::array<Worker, 8> workers;
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        auto run = [&](unsigned index) noexcept {
            auto& worker = workers[index];
            DiagnosticScope diagnostics(context.value, worker.error);
            try {
                Pipeline pipeline{EXR_DECODE_PIPELINE_INITIALIZER, context.value};
                bool initialized = false;
                while (!failed.load(std::memory_order_relaxed)) {
                    const auto position = next.fetch_add(1, std::memory_order_relaxed);
                    if (position >= chunks.size())
                        break;
                    const auto& chunk = chunks[position];
                    const auto status = initialized
                                            ? exr_decoding_update(context.value, 0, &chunk.info, &pipeline.value)
                                            : exr_decoding_initialize(context.value, 0, &chunk.info, &pipeline.value);
                    if (!check(status, worker.error)) {
                        failed.store(true, std::memory_order_relaxed);
                        break;
                    }
                    initialized = true;
                    for (int i = 0; i < pipeline.value.channel_count; ++i) {
                        auto& channel = pipeline.value.channels[i];
                        channel.decode_to_ptr = nullptr;
                        for (int c = 0; c < 4; ++c) {
                            if (header.channels[c] == i) {
                                channel.user_data_type = EXR_PIXEL_FLOAT;
                                channel.user_bytes_per_element = sizeof(float);
                                channel.user_pixel_stride = 4 * sizeof(float);
                                channel.user_line_stride = static_cast<int32_t>(size_t(header.width) * 4 * sizeof(float));
                                channel.decode_to_ptr = reinterpret_cast<uint8_t*>(pixels + (size_t(chunk.y) * header.width + chunk.x) * 4 + c);
                            }
                        }
                    }
                    if (!check(exr_decoding_choose_default_routines(context.value, 0, &pipeline.value), worker.error) ||
                        !check(exr_decoding_run(context.value, 0, &pipeline.value), worker.error)) {
                        failed.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): rethrown once every worker has joined.
                worker.exception = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
            }
        };
        std::vector<std::jthread> threads;
        threads.reserve(lease.count);
        for (unsigned i = 1; i <= lease.count; ++i) {
            try {
                threads.emplace_back(run, i);
            } catch (const std::system_error&) {
                // Already started workers and the caller still drain the shared queue.
                break;
            }
        }
        run(0);
        for (auto& thread : threads)
            thread.join();
        for (const auto& worker : workers) {
            if (worker.exception)
                std::rethrow_exception(worker.exception);
            if (!worker.error.empty()) {
                error = worker.error;
                return false;
            }
        }
        if (header.grayscale) {
            for (size_t i = 0; i < image.data.size() / sizeof(float); i += 4)
                pixels[i + 1] = pixels[i + 2] = pixels[i + 3] = pixels[i];
        }
        result = std::move(image);
        return true;
    }
} // namespace lfs::core::image_codecs

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ply.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <format>
#include <fstream>
#include <mutex>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

// TBB includes
#include <tbb/parallel_for.h>

// Platform-specific includes
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// SIMD includes (with fallback)
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace gs::loader {

    namespace ply_constants {
        constexpr int MAX_DC_COMPONENTS = 48;
        constexpr int MAX_REST_COMPONENTS = 135;
        constexpr int COLOR_CHANNELS = 3;
        constexpr int POSITION_DIMS = 3;
        constexpr int SCALE_DIMS = 3;
        constexpr int QUATERNION_DIMS = 4;
        constexpr float DEFAULT_LOG_SCALE = -5.0f;
        constexpr float IDENTITY_QUATERNION_W = 1.0f;
        constexpr float SCENE_SCALE_FACTOR = 0.5f;
        constexpr int SH_DEGREE_3_REST_COEFFS = 15;
        constexpr int SH_DEGREE_OFFSET = 1;

        // Block sizes for parallel processing
        constexpr size_t BLOCK_SIZE_SMALL = 1024;
        constexpr size_t BLOCK_SIZE_LARGE = 2048;
        constexpr size_t PLY_MIN_SIZE = 10;
        constexpr size_t FILE_SIZE_THRESHOLD_MB = 50;

        // SIMD constants
        constexpr int SIMD_WIDTH = 8;
        constexpr int SIMD_WIDTH_MINUS_1 = SIMD_WIDTH - 1;

        using namespace std::string_view_literals;
        constexpr auto VERTEX_ELEMENT = "vertex"sv;
        constexpr auto POS_X = "x"sv;
        constexpr auto POS_Y = "y"sv;
        constexpr auto POS_Z = "z"sv;
        constexpr auto OPACITY = "opacity"sv;
        constexpr auto DC_PREFIX = "f_dc_"sv;
        constexpr auto REST_PREFIX = "f_rest_"sv;
        constexpr auto SCALE_PREFIX = "scale_"sv;
        constexpr auto ROT_PREFIX = "rot_"sv;
    } // namespace ply_constants

    struct FastPropertyLayout {
        size_t vertex_count;
        size_t vertex_stride;

        // Pre-computed offsets for zero-copy access
        size_t pos_x_offset = SIZE_MAX, pos_y_offset = SIZE_MAX, pos_z_offset = SIZE_MAX;
        size_t opacity_offset = SIZE_MAX;
        size_t scale_offsets[3] = {SIZE_MAX, SIZE_MAX, SIZE_MAX};
        size_t rot_offsets[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
        size_t dc_start_offset = SIZE_MAX;
        size_t rest_start_offset = SIZE_MAX;
        int dc_count = 0, rest_count = 0;

        [[nodiscard]] bool has_positions() const { return pos_x_offset != SIZE_MAX; }
        [[nodiscard]] bool has_opacity() const { return opacity_offset != SIZE_MAX; }
        [[nodiscard]] bool has_scaling() const { return scale_offsets[0] != SIZE_MAX; }
        [[nodiscard]] bool has_rotation() const { return rot_offsets[0] != SIZE_MAX; }
    };

    struct MMappedFile {
        void* data = nullptr;
        size_t size = 0;

#ifdef _WIN32
        HANDLE file_handle = INVALID_HANDLE_VALUE;
        HANDLE mapping_handle = INVALID_HANDLE_VALUE;

        ~MMappedFile() {
            if (data)
                UnmapViewOfFile(data);
            if (mapping_handle != INVALID_HANDLE_VALUE)
                CloseHandle(mapping_handle);
            if (file_handle != INVALID_HANDLE_VALUE)
                CloseHandle(file_handle);
        }

        [[nodiscard]] bool map(const std::filesystem::path& filepath) {
            auto wide_path = filepath.wstring();
            file_handle = CreateFileW(wide_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                      nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (file_handle == INVALID_HANDLE_VALUE) {
                LOG_ERROR("Failed to open file for mapping: {}", filepath.string());
                return false;
            }

            LARGE_INTEGER file_size_li;
            if (!GetFileSizeEx(file_handle, &file_size_li)) {
                LOG_ERROR("Failed to get file size: {}", filepath.string());
                return false;
            }
            size = static_cast<size_t>(file_size_li.QuadPart);

            mapping_handle = CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (!mapping_handle) {
                LOG_ERROR("Failed to create file mapping: {}", filepath.string());
                return false;
            }

            data = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
            if (!data) {
                LOG_ERROR("Failed to map view of file: {}", filepath.string());
            }
            return data != nullptr;
        }
#else
        int fd = -1;

        ~MMappedFile() {
            if (data && data != MAP_FAILED)
                munmap(data, size);
            if (fd >= 0)
                close(fd);
        }

        [[nodiscard]] bool map(const std::filesystem::path& filepath) {
            fd = open(filepath.c_str(), O_RDONLY);
            if (fd < 0) {
                LOG_ERROR("Failed to open file for mapping: {}", filepath.string());
                return false;
            }

            struct stat st{};
            if (fstat(fd, &st) < 0) {
                LOG_ERROR("Failed to stat file: {}", filepath.string());
                return false;
            }
            size = st.st_size;

            data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (data == MAP_FAILED) {
                LOG_ERROR("Failed to mmap file: {}", filepath.string());
                return false;
            }

            // Prefetching based on file size
            if (size > ply_constants::FILE_SIZE_THRESHOLD_MB * 1024 * 1024) { // Only for files > 50MB
                if (madvise(data, size, MADV_SEQUENTIAL) == 0) {
                    LOG_DEBUG("Applied sequential access optimization for large file");
                }
            }

            return true;
        }
#endif

        [[nodiscard]] std::span<const char> as_span() const {
            return std::span{static_cast<const char*>(data), size};
        }
    };

    [[nodiscard]] std::expected<std::pair<size_t, FastPropertyLayout>, std::string>
    parse_header(const char* data, size_t file_size) {
        LOG_TIMER_TRACE("PLY header parsing");

        // Check for PLY magic with both Unix and Windows line endings
        if (file_size < ply_constants::PLY_MIN_SIZE) {
            LOG_ERROR("File too small to be valid PLY: {} bytes", file_size);
            throw std::runtime_error("File too small to be valid PLY");
        }

        bool has_crlf = false;
        if (std::strncmp(data, "ply\r\n", 5) == 0) {
            has_crlf = true;
        } else if (std::strncmp(data, "ply\n", 4) != 0) {
            LOG_ERROR("Invalid PLY file - missing PLY header");
            throw std::runtime_error("Invalid PLY file - missing PLY header");
        }

        const char* ptr = data + (has_crlf ? 5 : 4);
        const char* end = data + file_size;

        FastPropertyLayout layout = {};
        bool is_binary = false;
        bool found_vertex = false;
        size_t lines_parsed = 0;
        constexpr size_t MAX_HEADER_LINES = 10000; // Prevent infinite loops

        while (ptr < end && lines_parsed < MAX_HEADER_LINES) {
            const char* line_start = ptr;
            const char* line_end = nullptr;

            // Handle both \n and \r\n line endings efficiently
            for (const char* p = ptr; p < end; ++p) {
                if (*p == '\n') {
                    line_end = p;
                    ptr = p + 1;
                    break;
                } else if (*p == '\r' && p + 1 < end && *(p + 1) == '\n') {
                    line_end = p;
                    ptr = p + 2;
                    break;
                }
            }

            if (!line_end) {
                // No more complete lines
                break;
            }

            size_t line_len = line_end - line_start;
            lines_parsed++;

            // Skip empty lines and comments
            if (line_len == 0 || (line_len > 0 && line_start[0] == '#'))
                continue;

            // Progress reporting for large headers
            if (lines_parsed % 1000 == 0) {
                LOG_TRACE("Parsed {} header lines...", lines_parsed);
            }

            // Ultra-fast line parsing with minimal allocations
            if (line_len >= 27 && std::strncmp(line_start, "format binary_little_endian", 27) == 0) {
                is_binary = true;
            } else if (line_len >= 15 && std::strncmp(line_start, "element vertex ", 15) == 0) {
                layout.vertex_count = std::strtoull(line_start + 15, nullptr, 10);
                layout.vertex_stride = 0;
                found_vertex = true;
            } else if (line_len >= 15 && std::strncmp(line_start, "property float ", 15) == 0 && found_vertex) {
                const char* prop_name = line_start + 15;
                size_t name_len = line_len - 15;

                // Remove trailing whitespace/CR
                while (name_len > 0 && (prop_name[name_len - 1] == ' ' ||
                                        prop_name[name_len - 1] == '\t' ||
                                        prop_name[name_len - 1] == '\r')) {
                    name_len--;
                }

                // property recognition using first character + length
                if (name_len == 1) {
                    switch (*prop_name) {
                    case 'x': layout.pos_x_offset = layout.vertex_stride; break;
                    case 'y': layout.pos_y_offset = layout.vertex_stride; break;
                    case 'z': layout.pos_z_offset = layout.vertex_stride; break;
                    default:
                        // Ignore unknown single-character properties
                        break;
                    }
                } else if (name_len == 7 && std::strncmp(prop_name, "opacity", 7) == 0) {
                    layout.opacity_offset = layout.vertex_stride;
                } else if (name_len >= 5 && std::strncmp(prop_name, "f_dc_", 5) == 0) {
                    int idx = std::atoi(prop_name + 5);
                    if (idx == 0)
                        layout.dc_start_offset = layout.vertex_stride;
                    if (idx >= layout.dc_count)
                        layout.dc_count = idx + 1;
                } else if (name_len >= 7 && std::strncmp(prop_name, "f_rest_", 7) == 0) {
                    int idx = std::atoi(prop_name + 7);
                    if (idx == 0)
                        layout.rest_start_offset = layout.vertex_stride;
                    if (idx >= layout.rest_count)
                        layout.rest_count = idx + 1;
                } else if (name_len == 7 && std::strncmp(prop_name, "scale_", 6) == 0) {
                    int idx = prop_name[6] - '0';
                    if (idx >= 0 && idx < 3)
                        layout.scale_offsets[idx] = layout.vertex_stride;
                } else if (name_len == 5 && std::strncmp(prop_name, "rot_", 4) == 0) {
                    int idx = prop_name[4] - '0';
                    if (idx >= 0 && idx < 4)
                        layout.rot_offsets[idx] = layout.vertex_stride;
                }

                layout.vertex_stride += 4; // All properties are float32
            } else if (line_len >= 10 && std::strncmp(line_start, "end_header", 10) == 0) {
                if (!is_binary || !found_vertex) {
                    LOG_ERROR("Only binary PLY with position supported");
                    throw std::runtime_error("Only binary PLY with position supported");
                }
                LOG_DEBUG("Header parsed - {} lines, stride: {} bytes, dc: {}, rest: {}",
                          lines_parsed, layout.vertex_stride, layout.dc_count, layout.rest_count);
                return std::make_pair(ptr - data, layout);
            }
        }

        if (lines_parsed >= MAX_HEADER_LINES) {
            std::string error_msg = std::format("Header too large - exceeded {} lines", MAX_HEADER_LINES);
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        LOG_ERROR("No end_header found in PLY file");
        throw std::runtime_error("No end_header found in PLY file");
    }

    // SIMD position extraction
    void extract_positions(const char* vertex_data, const FastPropertyLayout& layout, const torch::Tensor& means) {
        const size_t count = layout.vertex_count;
        const size_t stride = layout.vertex_stride;
        float* output = means.data_ptr<float>();

        if (!layout.has_positions())
            return;

        LOG_DEBUG("Position extraction using TBB + SIMD for {} Gaussians", count);

#ifdef HAS_AVX2_SUPPORT
        // Thread-safe AVX2 detection using std::once_flag
        static std::once_flag avx2_flag;
        static bool has_avx2 = false;

        std::call_once(avx2_flag, []() {
#ifdef _WIN32
            int cpuInfo[4];
            __cpuid(cpuInfo, 7);
            has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
            __builtin_cpu_init();
            has_avx2 = __builtin_cpu_supports("avx2");
#else
            has_avx2 = false; // Fallback for other compilers
#endif
        });

        if (has_avx2) {
            LOG_TRACE("Using AVX2 SIMD acceleration");

            // TBB parallel SIMD processing with larger blocks to reduce overhead
            tbb::parallel_for(tbb::blocked_range<size_t>(0, count, ply_constants::BLOCK_SIZE_LARGE),
                              [&](const tbb::blocked_range<size_t>& range) {
                                  size_t start = range.begin();
                                  size_t end = range.end();
                                  size_t range_size = end - start;
                                  size_t simd_end = start + (range_size & ~ply_constants::SIMD_WIDTH_MINUS_1); // 8-element aligned

                                  // C++23: [[assume]] for optimization
                                  [[assume(layout.pos_x_offset < stride)]];
                                  [[assume(layout.pos_y_offset < stride)]];
                                  [[assume(layout.pos_z_offset < stride)]];

                                  // Process 8 vertices at a time with SIMD
                                  for (size_t i = start; i < simd_end; i += ply_constants::SIMD_WIDTH) {
                    // Portable prefetch
#ifdef _MSC_VER
                                      _mm_prefetch((const char*)(vertex_data + (i + 16) * stride), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
                        __builtin_prefetch(vertex_data + (i + 16) * stride, 0, 1);
#endif

                                      // Load 8 x-coordinates
                                      __m256 x_vals = _mm256_set_ps(
                                          *reinterpret_cast<const float*>(vertex_data + (i + ply_constants::SIMD_WIDTH_MINUS_1) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 6) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 5) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 4) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 3) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 2) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 1) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_x_offset));

                                      __m256 y_vals = _mm256_set_ps(
                                          *reinterpret_cast<const float*>(vertex_data + (i + ply_constants::SIMD_WIDTH_MINUS_1) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 6) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 5) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 4) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 3) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 2) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 1) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_y_offset));

                                      __m256 z_vals = _mm256_set_ps(
                                          *reinterpret_cast<const float*>(vertex_data + (i + ply_constants::SIMD_WIDTH_MINUS_1) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 6) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 5) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 4) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 3) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 2) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 1) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_z_offset));

                                      // Store efficiently with proper alignment
                                      alignas(32) float temp_x[8], temp_y[8], temp_z[8];
                                      _mm256_store_ps(temp_x, x_vals);
                                      _mm256_store_ps(temp_y, y_vals);
                                      _mm256_store_ps(temp_z, z_vals);

                                      // Interleave XYZ (reverse order due to _mm256_set_ps)
                                      for (int j = 0; j < ply_constants::SIMD_WIDTH; ++j) {
                                          const size_t idx = i + (ply_constants::SIMD_WIDTH_MINUS_1 - j);
                                          output[idx * 3 + 0] = temp_x[ply_constants::SIMD_WIDTH_MINUS_1 - j];
                                          output[idx * 3 + 1] = temp_y[ply_constants::SIMD_WIDTH_MINUS_1 - j];
                                          output[idx * 3 + 2] = temp_z[ply_constants::SIMD_WIDTH_MINUS_1 - j];
                                      }
                                  }

                                  // Handle remaining elements in this range
                                  for (size_t i = simd_end; i < end; ++i) {
                                      output[i * 3 + 0] = *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_x_offset);
                                      output[i * 3 + 1] = *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_y_offset);
                                      output[i * 3 + 2] = *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_z_offset);
                                  }
                              });
        } else
#endif
        {
            LOG_TRACE("Using optimized scalar processing");

            // TBB parallel scalar processing
            tbb::parallel_for(tbb::blocked_range<size_t>(0, count, ply_constants::BLOCK_SIZE_LARGE),
                              [&](const tbb::blocked_range<size_t>& range) {
                                  for (size_t i = range.begin(); i < range.end(); ++i) {
                                      output[i * 3 + 0] = *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_x_offset);
                                      output[i * 3 + 1] = *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_y_offset);
                                      output[i * 3 + 2] = *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_z_offset);
                                  }
                              });
        }
    }

    // SH coefficient extraction
    void extract_sh_coefficients(const char* vertex_data, const FastPropertyLayout& layout,
                                 size_t start_offset, int coeff_count, int channels,
                                 const torch::Tensor& output) {
        if (coeff_count == 0 || start_offset == SIZE_MAX)
            return;

        const size_t count = layout.vertex_count;
        const size_t stride = layout.vertex_stride;
        const int B = coeff_count / channels;

        auto data_ptr = output.data_ptr<float>();

        LOG_TRACE("Extracting {} SH coefficients for {} vertices", coeff_count, count);

        // Extract coefficients matching the original's stack -> reshape -> transpose pattern
        tbb::parallel_for(tbb::blocked_range<size_t>(0, count, ply_constants::BLOCK_SIZE_SMALL),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  // For each vertex, read f_dc_0, f_dc_1, ..., f_dc_(coeff_count-1)
                                  for (int j = 0; j < coeff_count; ++j) {
                                      float value = *reinterpret_cast<const float*>(vertex_data + i * stride + start_offset + j * 4);

                                      // Reshape logic: j maps to (channel, b)
                                      int channel = j / B; // Which color channel (0, 1, 2)
                                      int b = j % B;       // Which basis function

                                      // Store in [N, B, 3] layout
                                      data_ptr[i * B * channels + b * channels + channel] = value;
                                  }
                              }
                          });
    }

    // Single property extraction
    void extract_property(const char* vertex_data, const FastPropertyLayout& layout,
                          size_t property_offset, float* output) {
        if (property_offset == SIZE_MAX)
            return;

        const size_t count = layout.vertex_count;
        const size_t stride = layout.vertex_stride;

        // Parallel extraction
        tbb::parallel_for(tbb::blocked_range<size_t>(0, count, ply_constants::BLOCK_SIZE_LARGE),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  output[i] = *reinterpret_cast<const float*>(vertex_data + i * stride + property_offset);
                              }
                          });
    }

    // Main function
    [[nodiscard]] std::expected<SplatData, std::string> load_ply(const std::filesystem::path& filepath) {
        try {
            LOG_TIMER("PLY File Loading");
            auto start_time = std::chrono::high_resolution_clock::now();

            if (!std::filesystem::exists(filepath)) {
                std::string error_msg = std::format("PLY file does not exist: {}", filepath.string());
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            }

            // Memory map
            MMappedFile mapped_file;
            if (!mapped_file.map(filepath)) {
                LOG_ERROR("Failed to memory map PLY file: {}", filepath.string());
                throw std::runtime_error("Failed to memory map PLY file");
            }

            const char* data = static_cast<const char*>(mapped_file.data);
            const size_t file_size = mapped_file.size;

            // Ultra-fast header parsing
            auto parse_result = parse_header(data, file_size);
            if (!parse_result) {
                LOG_ERROR("Failed to parse PLY header: {}", parse_result.error());
                throw std::runtime_error(parse_result.error());
            }

            auto [data_offset, layout] = parse_result.value();
            const char* vertex_data = data + data_offset;

            LOG_INFO("Extracting {} Gaussians from PLY", layout.vertex_count);

            auto options = torch::TensorOptions().dtype(torch::kFloat32);

            // Position extraction
            auto means = torch::zeros({static_cast<int64_t>(layout.vertex_count), 3}, options);
            extract_positions(vertex_data, layout, means);

            // SH coefficient extraction
            torch::Tensor sh0, shN;

            if (layout.dc_count > 0 && layout.dc_count % ply_constants::COLOR_CHANNELS == 0) {
                int B0 = layout.dc_count / ply_constants::COLOR_CHANNELS;
                sh0 = torch::zeros({static_cast<int64_t>(layout.vertex_count), B0, ply_constants::COLOR_CHANNELS}, options);
                extract_sh_coefficients(vertex_data, layout, layout.dc_start_offset,
                                        layout.dc_count, ply_constants::COLOR_CHANNELS, sh0);
            } else {
                sh0 = torch::zeros({static_cast<int64_t>(layout.vertex_count), 1, ply_constants::COLOR_CHANNELS}, options);
            }

            if (layout.rest_count > 0 && layout.rest_count % ply_constants::COLOR_CHANNELS == 0) {
                int Bn = layout.rest_count / ply_constants::COLOR_CHANNELS;
                shN = torch::zeros({static_cast<int64_t>(layout.vertex_count), Bn, ply_constants::COLOR_CHANNELS}, options);
                extract_sh_coefficients(vertex_data, layout, layout.rest_start_offset,
                                        layout.rest_count, ply_constants::COLOR_CHANNELS, shN);
            } else {
                shN = torch::zeros({static_cast<int64_t>(layout.vertex_count), ply_constants::SH_DEGREE_3_REST_COEFFS, ply_constants::COLOR_CHANNELS}, options);
            }

            // property extraction
            auto opacity_tensor = torch::zeros({static_cast<int64_t>(layout.vertex_count), 1}, options);
            if (layout.has_opacity()) {
                extract_property(vertex_data, layout, layout.opacity_offset, opacity_tensor.data_ptr<float>());
            }

            auto scaling = torch::zeros({static_cast<int64_t>(layout.vertex_count), 3}, options);
            if (layout.has_scaling()) {
                // Extract scale components individually then stack
                std::vector<float> s0(layout.vertex_count), s1(layout.vertex_count), s2(layout.vertex_count);

                extract_property(vertex_data, layout, layout.scale_offsets[0], s0.data());
                extract_property(vertex_data, layout, layout.scale_offsets[1], s1.data());
                extract_property(vertex_data, layout, layout.scale_offsets[2], s2.data());

                // Stack into final tensor
                auto scaling_ptr = scaling.data_ptr<float>();
                tbb::parallel_for(tbb::blocked_range<size_t>(0, layout.vertex_count, ply_constants::BLOCK_SIZE_SMALL),
                                  [&](const tbb::blocked_range<size_t>& range) {
                                      for (size_t i = range.begin(); i < range.end(); ++i) {
                                          scaling_ptr[i * 3 + 0] = s0[i];
                                          scaling_ptr[i * 3 + 1] = s1[i];
                                          scaling_ptr[i * 3 + 2] = s2[i];
                                      }
                                  });
            } else {
                scaling.fill_(ply_constants::DEFAULT_LOG_SCALE);
            }

            auto rotation = torch::zeros({static_cast<int64_t>(layout.vertex_count), 4}, options);
            if (layout.has_rotation()) {
                // Extract rotation components individually then stack
                std::vector<float> r0(layout.vertex_count), r1(layout.vertex_count), r2(layout.vertex_count), r3(layout.vertex_count);

                extract_property(vertex_data, layout, layout.rot_offsets[0], r0.data());
                extract_property(vertex_data, layout, layout.rot_offsets[1], r1.data());
                extract_property(vertex_data, layout, layout.rot_offsets[2], r2.data());
                extract_property(vertex_data, layout, layout.rot_offsets[3], r3.data());

                // Stack into final tensor
                auto rotation_ptr = rotation.data_ptr<float>();
                tbb::parallel_for(tbb::blocked_range<size_t>(0, layout.vertex_count, ply_constants::BLOCK_SIZE_SMALL),
                                  [&](const tbb::blocked_range<size_t>& range) {
                                      for (size_t i = range.begin(); i < range.end(); ++i) {
                                          rotation_ptr[i * 4 + 0] = r0[i];
                                          rotation_ptr[i * 4 + 1] = r1[i];
                                          rotation_ptr[i * 4 + 2] = r2[i];
                                          rotation_ptr[i * 4 + 3] = r3[i];
                                      }
                                  });
            } else {
                rotation.select(1, 0).fill_(ply_constants::IDENTITY_QUATERNION_W);
            }

            LOG_DEBUG("Transferring tensors to CUDA");
            // Batch CUDA transfer for maximum speed
            means = means.to(torch::kCUDA);
            sh0 = sh0.to(torch::kCUDA);
            shN = shN.to(torch::kCUDA);
            scaling = scaling.to(torch::kCUDA);
            rotation = rotation.to(torch::kCUDA);
            opacity_tensor = opacity_tensor.to(torch::kCUDA);

            int sh_degree = static_cast<int>(std::sqrt(shN.size(1) + ply_constants::SH_DEGREE_OFFSET)) - ply_constants::SH_DEGREE_OFFSET;

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            LOG_INFO("PLY loaded: {} MB, {} Gaussians with SH degree {} in {}ms",
                     file_size / (1024 * 1024), layout.vertex_count, sh_degree, duration.count());

            return SplatData(
                sh_degree,
                means,
                sh0,
                shN,
                scaling,
                rotation,
                opacity_tensor,
                ply_constants::SCENE_SCALE_FACTOR);

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load PLY file: {}", e.what());
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }
    }

} // namespace gs::loader
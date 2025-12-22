/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ply.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "io/error.hpp"
#include "tinyply.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <future>
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

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

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
        size_t dc_offsets[ply_constants::MAX_DC_COMPONENTS];
        size_t rest_offsets[ply_constants::MAX_REST_COMPONENTS];
        int dc_count = 0, rest_count = 0;

        FastPropertyLayout() {
            std::fill(std::begin(dc_offsets), std::end(dc_offsets), SIZE_MAX);
            std::fill(std::begin(rest_offsets), std::end(rest_offsets), SIZE_MAX);
        }

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

            struct stat st {};
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
        bool has_vertex_element = false;
        bool parsing_vertex = false;
        size_t lines_parsed = 0;
        constexpr size_t MAX_HEADER_LINES = 10000;

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

            if (!line_end)
                break;

            size_t line_len = line_end - line_start;
            lines_parsed++;

            // Skip empty lines and comments
            if (line_len == 0 || (line_len > 0 && line_start[0] == '#'))
                continue;

            if (lines_parsed % 1000 == 0) {
                LOG_TRACE("Parsed {} header lines...", lines_parsed);
            }

            // Line parsing
            if (line_len >= 27 && std::strncmp(line_start, "format binary_little_endian", 27) == 0) {
                is_binary = true;
            } else if (line_len >= 8 && std::strncmp(line_start, "element ", 8) == 0) {
                if (line_len >= 15 && std::strncmp(line_start, "element vertex ", 15) == 0) {
                    layout.vertex_count = std::strtoull(line_start + 15, nullptr, 10);
                    layout.vertex_stride = 0;
                    has_vertex_element = true;
                    parsing_vertex = true;
                } else {
                    parsing_vertex = false;
                }
            } else if (line_len >= 15 && std::strncmp(line_start, "property float ", 15) == 0 && parsing_vertex) {
                const char* prop_name = line_start + 15;
                size_t name_len = line_len - 15;

                // Remove trailing whitespace/CR
                while (name_len > 0 && (prop_name[name_len - 1] == ' ' ||
                                        prop_name[name_len - 1] == '\t' ||
                                        prop_name[name_len - 1] == '\r')) {
                    name_len--;
                }

                // Property recognition using first character + length
                if (name_len == 1) {
                    switch (*prop_name) {
                    case 'x': layout.pos_x_offset = layout.vertex_stride; break;
                    case 'y': layout.pos_y_offset = layout.vertex_stride; break;
                    case 'z': layout.pos_z_offset = layout.vertex_stride; break;
                    default: break;
                    }
                } else if (name_len == 7 && std::strncmp(prop_name, "opacity", 7) == 0) {
                    layout.opacity_offset = layout.vertex_stride;
                } else if (name_len >= 5 && std::strncmp(prop_name, "f_dc_", 5) == 0) {
                    int idx = std::atoi(prop_name + 5);
                    if (idx >= 0 && idx < ply_constants::MAX_DC_COMPONENTS) {
                        layout.dc_offsets[idx] = layout.vertex_stride;
                        if (idx >= layout.dc_count)
                            layout.dc_count = idx + 1;
                    }
                } else if (name_len >= 7 && std::strncmp(prop_name, "f_rest_", 7) == 0) {
                    int idx = std::atoi(prop_name + 7);
                    if (idx >= 0 && idx < ply_constants::MAX_REST_COMPONENTS) {
                        layout.rest_offsets[idx] = layout.vertex_stride;
                        if (idx >= layout.rest_count)
                            layout.rest_count = idx + 1;
                    }
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
                if (!is_binary || !has_vertex_element) {
                    LOG_ERROR("Only binary PLY with vertex element supported");
                    throw std::runtime_error("Only binary PLY with vertex element supported");
                }
                LOG_DEBUG("Header parsed: {} vertices, stride {} bytes, dc {}, rest {}",
                          layout.vertex_count, layout.vertex_stride, layout.dc_count, layout.rest_count);
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

    // SIMD position extraction to host memory
    void extract_positions_to_host(const char* vertex_data, const FastPropertyLayout& layout, float* output) {
        const size_t count = layout.vertex_count;
        const size_t stride = layout.vertex_stride;

        if (!layout.has_positions())
            return;

        LOG_DEBUG("Position extraction using TBB + SIMD for {} Gaussians", count);

#ifdef HAS_AVX2_SUPPORT
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
            has_avx2 = false;
#endif
        });

        if (has_avx2) {
            LOG_TRACE("Using AVX2 SIMD acceleration");

            tbb::parallel_for(tbb::blocked_range<size_t>(0, count, ply_constants::BLOCK_SIZE_LARGE),
                              [&](const tbb::blocked_range<size_t>& range) {
                                  size_t start = range.begin();
                                  size_t end = range.end();
                                  size_t range_size = end - start;
                                  size_t simd_end = start + (range_size & ~ply_constants::SIMD_WIDTH_MINUS_1);

                                  // C++23: [[assume]] for optimization
                                  [[assume(layout.pos_x_offset < stride)]];
                                  [[assume(layout.pos_y_offset < stride)]];
                                  [[assume(layout.pos_z_offset < stride)]];

                                  for (size_t i = start; i < simd_end; i += ply_constants::SIMD_WIDTH) {
#ifdef _MSC_VER
                                      _mm_prefetch((const char*)(vertex_data + (i + 16) * stride), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
                        __builtin_prefetch(vertex_data + (i + 16) * stride, 0, 1);
#endif

                                      __m256 x_vals = _mm256_set_ps(
                                          *reinterpret_cast<const float*>(vertex_data + (i + 7) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 6) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 5) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 4) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 3) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 2) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 1) * stride + layout.pos_x_offset),
                                          *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_x_offset));

                                      __m256 y_vals = _mm256_set_ps(
                                          *reinterpret_cast<const float*>(vertex_data + (i + 7) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 6) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 5) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 4) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 3) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 2) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 1) * stride + layout.pos_y_offset),
                                          *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_y_offset));

                                      __m256 z_vals = _mm256_set_ps(
                                          *reinterpret_cast<const float*>(vertex_data + (i + 7) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 6) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 5) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 4) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 3) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 2) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + (i + 1) * stride + layout.pos_z_offset),
                                          *reinterpret_cast<const float*>(vertex_data + i * stride + layout.pos_z_offset));

                                      alignas(32) float temp_x[8], temp_y[8], temp_z[8];
                                      _mm256_store_ps(temp_x, x_vals);
                                      _mm256_store_ps(temp_y, y_vals);
                                      _mm256_store_ps(temp_z, z_vals);

                                      for (int j = 0; j < ply_constants::SIMD_WIDTH; ++j) {
                                          const size_t idx = i + (7 - j);
                                          output[idx * 3 + 0] = temp_x[7 - j];
                                          output[idx * 3 + 1] = temp_y[7 - j];
                                          output[idx * 3 + 2] = temp_z[7 - j];
                                      }
                                  }

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

    // SH coefficient extraction with per-coefficient offsets (handles arbitrary PLY property order)
    void extract_sh_coefficients_to_host(const char* __restrict__ vertex_data,
                                         const FastPropertyLayout& layout,
                                         const size_t* __restrict__ coeff_offsets,
                                         const int coeff_count, const int channels,
                                         float* __restrict__ output) {
        if (coeff_count == 0)
            return;

        const size_t count = layout.vertex_count;
        const size_t stride = layout.vertex_stride;
        const int B = coeff_count / channels;

        tbb::parallel_for(tbb::blocked_range<size_t>(0, count, ply_constants::BLOCK_SIZE_SMALL),
                          [=](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  const size_t base = i * stride;
                                  const size_t out_base = i * B * channels;
                                  for (int j = 0; j < coeff_count; ++j) {
                                      const size_t offset = coeff_offsets[j];
                                      const float value = (offset != SIZE_MAX)
                                                              ? *reinterpret_cast<const float*>(vertex_data + base + offset)
                                                              : 0.0f;
                                      const int channel = j / B;
                                      const int b = j % B;
                                      output[out_base + b * channels + channel] = value;
                                  }
                              }
                          });
    }

    // Single property extraction to host memory
    void extract_property_to_host(const char* vertex_data, const FastPropertyLayout& layout,
                                  size_t property_offset, float* output) {
        if (property_offset == SIZE_MAX)
            return;

        const size_t count = layout.vertex_count;
        const size_t stride = layout.vertex_stride;

        tbb::parallel_for(tbb::blocked_range<size_t>(0, count, ply_constants::BLOCK_SIZE_LARGE),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  output[i] = *reinterpret_cast<const float*>(vertex_data + i * stride + property_offset);
                              }
                          });
    }

    // Main function - returns SplatData
    [[nodiscard]] std::expected<SplatData, std::string>
    load_ply(const std::filesystem::path& filepath) {
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

            const size_t N = layout.vertex_count;

            // Extract positions to host memory
            std::vector<float> host_means(N * 3);
            extract_positions_to_host(vertex_data, layout, host_means.data());

            // Determine SH dimensions
            int sh0_dim1 = 1, sh0_dim2 = ply_constants::COLOR_CHANNELS;
            int shN_dim1 = 0, shN_dim2 = ply_constants::COLOR_CHANNELS;

            // Extract SH coefficients
            std::vector<float> host_sh0;
            std::vector<float> host_shN;

            if (layout.dc_count > 0 && layout.dc_count % ply_constants::COLOR_CHANNELS == 0) {
                int B0 = layout.dc_count / ply_constants::COLOR_CHANNELS;
                sh0_dim1 = B0;
                host_sh0.resize(N * B0 * ply_constants::COLOR_CHANNELS);
                extract_sh_coefficients_to_host(vertex_data, layout, layout.dc_offsets,
                                                layout.dc_count, ply_constants::COLOR_CHANNELS, host_sh0.data());
            } else {
                host_sh0.resize(N * ply_constants::COLOR_CHANNELS, 0.0f);
            }

            if (layout.rest_count > 0 && layout.rest_count % ply_constants::COLOR_CHANNELS == 0) {
                int Bn = layout.rest_count / ply_constants::COLOR_CHANNELS;
                shN_dim1 = Bn;
                host_shN.resize(N * Bn * ply_constants::COLOR_CHANNELS);
                extract_sh_coefficients_to_host(vertex_data, layout, layout.rest_offsets,
                                                layout.rest_count, ply_constants::COLOR_CHANNELS, host_shN.data());
            } else {
                shN_dim1 = ply_constants::SH_DEGREE_3_REST_COEFFS;
                host_shN.resize(N * ply_constants::SH_DEGREE_3_REST_COEFFS * ply_constants::COLOR_CHANNELS, 0.0f);
            }

            // Extract other properties
            std::vector<float> host_opacity(N, 0.0f);
            if (layout.has_opacity()) {
                extract_property_to_host(vertex_data, layout, layout.opacity_offset, host_opacity.data());
            }

            std::vector<float> host_scaling(N * 3);
            if (layout.has_scaling()) {
                std::vector<float> s0(N), s1(N), s2(N);
                extract_property_to_host(vertex_data, layout, layout.scale_offsets[0], s0.data());
                extract_property_to_host(vertex_data, layout, layout.scale_offsets[1], s1.data());
                extract_property_to_host(vertex_data, layout, layout.scale_offsets[2], s2.data());

                tbb::parallel_for(tbb::blocked_range<size_t>(0, N, ply_constants::BLOCK_SIZE_SMALL),
                                  [&](const tbb::blocked_range<size_t>& range) {
                                      for (size_t i = range.begin(); i < range.end(); ++i) {
                                          host_scaling[i * 3 + 0] = s0[i];
                                          host_scaling[i * 3 + 1] = s1[i];
                                          host_scaling[i * 3 + 2] = s2[i];
                                      }
                                  });
            } else {
                std::fill(host_scaling.begin(), host_scaling.end(), ply_constants::DEFAULT_LOG_SCALE);
            }

            std::vector<float> host_rotation(N * 4, 0.0f);
            if (layout.has_rotation()) {
                std::vector<float> r0(N), r1(N), r2(N), r3(N);
                extract_property_to_host(vertex_data, layout, layout.rot_offsets[0], r0.data());
                extract_property_to_host(vertex_data, layout, layout.rot_offsets[1], r1.data());
                extract_property_to_host(vertex_data, layout, layout.rot_offsets[2], r2.data());
                extract_property_to_host(vertex_data, layout, layout.rot_offsets[3], r3.data());

                tbb::parallel_for(tbb::blocked_range<size_t>(0, N, ply_constants::BLOCK_SIZE_SMALL),
                                  [&](const tbb::blocked_range<size_t>& range) {
                                      for (size_t i = range.begin(); i < range.end(); ++i) {
                                          host_rotation[i * 4 + 0] = r0[i];
                                          host_rotation[i * 4 + 1] = r1[i];
                                          host_rotation[i * 4 + 2] = r2[i];
                                          host_rotation[i * 4 + 3] = r3[i];
                                      }
                                  });
            } else {
                // Set identity quaternion
                for (size_t i = 0; i < N; ++i) {
                    host_rotation[i * 4 + 0] = ply_constants::IDENTITY_QUATERNION_W;
                }
            }

            LOG_DEBUG("Creating Tensor objects and uploading to CUDA");

            // Create Tensors directly from vectors (uploads to CUDA)
            Tensor means = Tensor::from_vector(host_means, {N, 3}, Device::CUDA);
            Tensor sh0 = Tensor::from_vector(host_sh0, {N, static_cast<size_t>(sh0_dim1), static_cast<size_t>(sh0_dim2)}, Device::CUDA);
            Tensor shN = Tensor::from_vector(host_shN, {N, static_cast<size_t>(shN_dim1), static_cast<size_t>(shN_dim2)}, Device::CUDA);
            Tensor scaling = Tensor::from_vector(host_scaling, {N, 3}, Device::CUDA);
            Tensor rotation = Tensor::from_vector(host_rotation, {N, 4}, Device::CUDA);
            Tensor opacity = Tensor::from_vector(host_opacity, {N, 1}, Device::CUDA);

            // Calculate SH degree
            int sh_degree = static_cast<int>(std::sqrt(shN_dim1 + ply_constants::SH_DEGREE_OFFSET)) - ply_constants::SH_DEGREE_OFFSET;

            // Create SplatData
            SplatData splat_data(
                sh_degree,
                std::move(means),
                std::move(sh0),
                std::move(shN),
                std::move(scaling),
                std::move(rotation),
                std::move(opacity),
                ply_constants::SCENE_SCALE_FACTOR);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            LOG_INFO("PLY loaded: {} MB, {} Gaussians with SH degree {} in {}ms",
                     file_size / (1024 * 1024), N, sh_degree, duration.count());

            return splat_data;

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load PLY file: {}", e.what());
            LOG_ERROR("{}", error_msg);
            return std::unexpected(error_msg);
        }
    }

    // ============================================================================
    // PLY Save Implementation
    // ============================================================================

    namespace {

        std::mutex g_save_mutex;
        std::vector<std::future<void>> g_save_futures;

        void cleanup_finished_saves() {
            std::lock_guard lock(g_save_mutex);
            g_save_futures.erase(
                std::remove_if(g_save_futures.begin(), g_save_futures.end(),
                               [](const std::future<void>& f) {
                                   return !f.valid() || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                               }),
                g_save_futures.end());
        }

        void write_ply_binary(const PointCloud& pc, const std::filesystem::path& output_path) {
            std::vector<Tensor> tensors;
            tensors.push_back(pc.means.cpu().contiguous());

            if (pc.normals.is_valid()) {
                tensors.push_back(pc.normals.cpu().contiguous());
            }

            auto process_sh = [](const Tensor& sh) -> Tensor {
                if (sh.ndim() == 3) {
                    auto transposed = sh.transpose(1, 2).contiguous();
                    return transposed.flatten(1).cpu().contiguous();
                }
                return sh.cpu().contiguous();
            };

            if (pc.sh0.is_valid())
                tensors.push_back(process_sh(pc.sh0));
            if (pc.shN.is_valid())
                tensors.push_back(process_sh(pc.shN));
            if (pc.opacity.is_valid())
                tensors.push_back(pc.opacity.cpu().contiguous());
            if (pc.scaling.is_valid())
                tensors.push_back(pc.scaling.cpu().contiguous());
            if (pc.rotation.is_valid())
                tensors.push_back(pc.rotation.cpu().contiguous());

            // Write using tinyply
            tinyply::PlyFile ply;
            size_t attr_off = 0;

            for (const auto& tensor : tensors) {
                const size_t cols = tensor.size(1);
                std::vector<std::string> attrs(pc.attribute_names.begin() + attr_off,
                                               pc.attribute_names.begin() + attr_off + cols);

                ply.add_properties_to_element(
                    "vertex", attrs, tinyply::Type::FLOAT32, tensor.size(0),
                    reinterpret_cast<uint8_t*>(const_cast<float*>(tensor.ptr<float>())),
                    tinyply::Type::INVALID, 0);

                attr_off += cols;
            }

            std::filebuf fb;
            fb.open(output_path, std::ios::out | std::ios::binary);
            std::ostream out_stream(&fb);
            ply.write(out_stream, true);
        }

    } // anonymous namespace

    PointCloud to_point_cloud(const SplatData& splat_data) {
        PointCloud pc;

        pc.means = splat_data.means().cpu().contiguous();
        pc.normals = Tensor::zeros_like(pc.means);

        auto process_sh = [](const Tensor& sh) -> Tensor {
            const auto sh_cpu = sh.cpu().contiguous();
            if (sh_cpu.ndim() == 3) {
                const auto transposed = sh_cpu.transpose(1, 2);
                const size_t N = transposed.shape()[0];
                const size_t flat_dim = transposed.shape()[1] * transposed.shape()[2];
                return transposed.reshape({static_cast<int>(N), static_cast<int>(flat_dim)});
            }
            return sh_cpu;
        };

        if (splat_data.sh0().is_valid())
            pc.sh0 = process_sh(splat_data.sh0());
        if (splat_data.shN().is_valid())
            pc.shN = process_sh(splat_data.shN());
        if (splat_data.opacity_raw().is_valid())
            pc.opacity = splat_data.opacity_raw().cpu().contiguous();
        if (splat_data.scaling_raw().is_valid())
            pc.scaling = splat_data.scaling_raw().cpu().contiguous();

        if (splat_data.rotation_raw().is_valid()) {
            pc.rotation = splat_data.get_rotation().cpu().contiguous();
        }

        pc.attribute_names = get_ply_attribute_names(splat_data);
        return pc;
    }

    std::vector<std::string> get_ply_attribute_names(const SplatData& splat_data) {
        std::vector<std::string> attrs{"x", "y", "z", "nx", "ny", "nz"};

        auto add_indexed_attrs = [&attrs](const std::string& prefix, const size_t count) {
            for (size_t i = 0; i < count; ++i) {
                attrs.emplace_back(prefix + std::to_string(i));
            }
        };

        auto get_feature_count = [](const Tensor& t) -> size_t {
            if (t.ndim() == 3)
                return t.shape()[1] * t.shape()[2];
            if (t.ndim() == 2)
                return t.shape()[1];
            return 0;
        };

        if (splat_data.sh0().is_valid())
            add_indexed_attrs("f_dc_", get_feature_count(splat_data.sh0()));
        if (splat_data.shN().is_valid())
            add_indexed_attrs("f_rest_", get_feature_count(splat_data.shN()));

        attrs.emplace_back("opacity");

        if (splat_data.scaling_raw().is_valid())
            add_indexed_attrs("scale_", splat_data.scaling_raw().shape()[1]);
        if (splat_data.rotation_raw().is_valid())
            add_indexed_attrs("rot_", splat_data.rotation_raw().shape()[1]);

        return attrs;
    }

    Result<void> save_ply(const SplatData& splat_data, const PlySaveOptions& options) {
        auto pc = lfs::io::to_point_cloud(splat_data);
        return save_ply(pc, options);
    }

    Result<void> save_ply(const PointCloud& point_cloud, const PlySaveOptions& options) {
        // Calculate estimated file size for disk space check
        // PLY binary: header (~500 bytes) + vertex_count * stride (floats)
        const size_t vertex_count = point_cloud.means.size(0);
        size_t floats_per_vertex = 3; // positions

        if (point_cloud.normals.is_valid())
            floats_per_vertex += 3;
        if (point_cloud.sh0.is_valid()) {
            floats_per_vertex += point_cloud.sh0.ndim() == 3
                                     ? point_cloud.sh0.size(1) * point_cloud.sh0.size(2)
                                     : point_cloud.sh0.size(1);
        }
        if (point_cloud.shN.is_valid()) {
            floats_per_vertex += point_cloud.shN.ndim() == 3
                                     ? point_cloud.shN.size(1) * point_cloud.shN.size(2)
                                     : point_cloud.shN.size(1);
        }
        if (point_cloud.opacity.is_valid())
            floats_per_vertex += 1;
        if (point_cloud.scaling.is_valid())
            floats_per_vertex += 3;
        if (point_cloud.rotation.is_valid())
            floats_per_vertex += 4;

        const size_t estimated_size = 1024 + vertex_count * floats_per_vertex * sizeof(float);

        // Check disk space with 10% margin
        if (auto space_check = check_disk_space(options.output_path, estimated_size, 1.1f); !space_check) {
            return std::unexpected(space_check.error());
        }

        // Verify path is writable
        if (auto writable_check = verify_writable(options.output_path); !writable_check) {
            return std::unexpected(writable_check.error());
        }

        // Create parent directories
        std::error_code ec;
        std::filesystem::create_directories(options.output_path.parent_path(), ec);
        if (ec) {
            return make_error(ErrorCode::PERMISSION_DENIED,
                              std::format("Cannot create directory: {}", ec.message()),
                              options.output_path.parent_path());
        }

        if (options.async) {
            cleanup_finished_saves();
            std::lock_guard lock(g_save_mutex);
            g_save_futures.emplace_back(
                std::async(std::launch::async, [pc = point_cloud, path = options.output_path]() {
                    try {
                        write_ply_binary(pc, path);
                        LOG_INFO("PLY saved: {}", path.string());
                    } catch (const std::exception& e) {
                        // Log error - async saves report via logs
                        LOG_ERROR("Async PLY save failed for '{}': {}", path.string(), e.what());
                    }
                }));
            // Note: Async save errors are logged but not returned
            // The disk space check above prevents most failures
        } else {
            try {
                write_ply_binary(point_cloud, options.output_path);
                LOG_INFO("PLY saved: {}", options.output_path.string());
            } catch (const std::exception& e) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Failed to write PLY: {}", e.what()),
                                  options.output_path);
            }
        }
        return {};
    }

    bool is_gaussian_splat_ply(const std::filesystem::path& filepath) {
        if (!std::filesystem::exists(filepath))
            return false;

        std::ifstream file(filepath, std::ios::binary);
        if (!file)
            return false;

        std::string line;
        bool has_opacity = false, has_scale = false, has_rotation = false;

        while (std::getline(file, line)) {
            if (line.find("end_header") != std::string::npos)
                break;
            if (line.find("property") != std::string::npos) {
                if (line.find("opacity") != std::string::npos)
                    has_opacity = true;
                if (line.find("scale_0") != std::string::npos)
                    has_scale = true;
                if (line.find("rot_0") != std::string::npos)
                    has_rotation = true;
            }
        }
        return has_opacity && has_scale && has_rotation;
    }

    std::expected<lfs::core::PointCloud, std::string> load_ply_point_cloud(const std::filesystem::path& filepath) {
        constexpr uint8_t DEFAULT_COLOR = 255;

        if (!std::filesystem::exists(filepath)) {
            return std::unexpected(std::format("File not found: {}", filepath.string()));
        }

        try {
            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                return std::unexpected(std::format("Cannot open: {}", filepath.string()));
            }

            tinyply::PlyFile ply;
            ply.parse_header(file);

            std::shared_ptr<tinyply::PlyData> vertices;
            try {
                vertices = ply.request_properties_from_element("vertex", {"x", "y", "z"});
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Missing vertices: {}", e.what()));
            }

            std::shared_ptr<tinyply::PlyData> colors;
            bool has_colors = false;
            try {
                colors = ply.request_properties_from_element("vertex", {"red", "green", "blue"});
                has_colors = true;
            } catch (...) {}

            ply.read(file);

            const size_t N = vertices->count;
            LOG_DEBUG("Point cloud: {} points", N);

            using namespace lfs::core;
            Tensor positions = Tensor::zeros({N, 3}, Device::CPU, DataType::Float32);
            float* const pos_ptr = positions.ptr<float>();

            if (vertices->t == tinyply::Type::FLOAT32) {
                std::memcpy(pos_ptr, vertices->buffer.get(), N * 3 * sizeof(float));
            } else if (vertices->t == tinyply::Type::FLOAT64) {
                const auto* src = reinterpret_cast<const double*>(vertices->buffer.get());
                for (size_t i = 0; i < N * 3; ++i)
                    pos_ptr[i] = static_cast<float>(src[i]);
            } else {
                return std::unexpected("Unsupported vertex type");
            }

            Tensor color_tensor;
            if (has_colors && colors && colors->count == N) {
                if (colors->t == tinyply::Type::UINT8) {
                    color_tensor = Tensor::zeros({N, 3}, Device::CPU, DataType::UInt8);
                    std::memcpy(color_tensor.ptr<uint8_t>(), colors->buffer.get(), N * 3);
                } else if (colors->t == tinyply::Type::FLOAT32) {
                    Tensor float_colors = Tensor::zeros({N, 3}, Device::CPU, DataType::Float32);
                    std::memcpy(float_colors.ptr<float>(), colors->buffer.get(), N * 3 * sizeof(float));
                    color_tensor = (float_colors * 255.0f).clamp(0, 255).to(DataType::UInt8);
                } else {
                    color_tensor = Tensor::full({N, 3}, DEFAULT_COLOR, Device::CPU, DataType::UInt8);
                }
            } else {
                color_tensor = Tensor::full({N, 3}, DEFAULT_COLOR, Device::CPU, DataType::UInt8);
            }

            return PointCloud(std::move(positions), std::move(color_tensor));
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Load failed: {}", e.what()));
        }
    }

} // namespace lfs::io
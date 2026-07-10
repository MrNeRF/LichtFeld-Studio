/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ply.hpp"
#include "core/assert.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "io/error.hpp"
#include "io/ply_export_internal.hpp"
#include "tinyply.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <unordered_set>
#include <vector>

// TBB includes
#include <tbb/parallel_for.h>

// CUDA runtime for pinned host memory + async H2D copies
#include <cuda_runtime.h>

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
    using lfs::core::TensorShape;

    namespace ply_constants {
        constexpr int MAX_DC_COMPONENTS = 48;
        constexpr int MAX_REST_COMPONENTS = 135;
        constexpr int COLOR_CHANNELS = 3;
        constexpr int POSITION_DIMS = 3;
        constexpr int SCALE_DIMS = 3;
        constexpr int QUATERNION_DIMS = 4;
        constexpr float IDENTITY_QUATERNION_W = 1.0f;
        constexpr int SH_DEGREE_3_REST_COEFFS = 15;
        constexpr int SH_DEGREE_OFFSET = 1;

        // Block sizes for parallel processing
        constexpr size_t BLOCK_SIZE_SMALL = 1024;
        constexpr size_t BLOCK_SIZE_LARGE = 2048;
        constexpr size_t PLY_MIN_SIZE = 10;
        constexpr size_t FILE_SIZE_THRESHOLD_MB = 50;
        constexpr size_t VALIDATION_CANCEL_INTERVAL = 65536;
        constexpr float MIN_ROTATION_NORM_SQUARED = 1.0e-12f;

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

    namespace {

        bool is_valid_ply_property_name_token(const std::string_view name) {
            if (name.empty()) {
                return false;
            }

            return std::ranges::all_of(name, [](const char ch) {
                const unsigned char uchar = static_cast<unsigned char>(ch);
                return !std::iscntrl(uchar) && !std::isspace(uchar);
            });
        }

        bool is_reserved_ply_vertex_property_name(const std::string_view name) {
            using namespace ply_constants;

            return name == POS_X || name == POS_Y || name == POS_Z || name == "nx" || name == "ny" || name == "nz" || name == "red" || name == "green" || name == "blue" || name == OPACITY || name.starts_with(DC_PREFIX) || name.starts_with(REST_PREFIX) || name.starts_with(SCALE_PREFIX) || name.starts_with(ROT_PREFIX);
        }

        [[nodiscard]] std::string_view trim_ascii_whitespace(std::string_view value) {
            while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
                value.remove_prefix(1);
            }
            while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
                value.remove_suffix(1);
            }
            return value;
        }

        [[nodiscard]] bool parse_size_token(std::string_view token, size_t& value) {
            token = trim_ascii_whitespace(token);
            if (token.empty() || token.front() == '-') {
                return false;
            }

            size_t parsed = 0;
            const auto* const begin = token.data();
            const auto* const end = begin + token.size();
            const auto [ptr, ec] = std::from_chars(begin, end, parsed);
            if (ec != std::errc{} || ptr != end) {
                return false;
            }

            value = parsed;
            return true;
        }

        [[nodiscard]] bool parse_property_index(const std::string_view name,
                                                const std::string_view prefix,
                                                const int max_exclusive,
                                                int& index) {
            if (!name.starts_with(prefix) || name.size() == prefix.size()) {
                return false;
            }

            const std::string_view suffix = name.substr(prefix.size());
            if (suffix.front() == '-') {
                return false;
            }

            int parsed = 0;
            const auto* const begin = suffix.data();
            const auto* const end = begin + suffix.size();
            const auto [ptr, ec] = std::from_chars(begin, end, parsed);
            if (ec != std::errc{} || ptr != end || parsed < 0 || parsed >= max_exclusive) {
                return false;
            }

            index = parsed;
            return true;
        }

        [[nodiscard]] bool checked_mul_size(const size_t a, const size_t b, size_t& result) {
            if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
                return false;
            }
            result = a * b;
            return true;
        }

        [[nodiscard]] bool ply_scalar_property_size(const std::string_view type, size_t& size) {
            if (type == "char" || type == "uchar" ||
                type == "int8" || type == "uint8") {
                size = 1;
                return true;
            }
            if (type == "short" || type == "ushort" ||
                type == "int16" || type == "uint16") {
                size = 2;
                return true;
            }
            if (type == "int" || type == "uint" ||
                type == "int32" || type == "uint32" ||
                type == "float" || type == "float32") {
                size = 4;
                return true;
            }
            if (type == "double" || type == "float64") {
                size = 8;
                return true;
            }
            return false;
        }

        [[nodiscard]] bool is_float32_ply_type(const std::string_view type) {
            return type == "float" || type == "float32";
        }

    } // namespace

    std::vector<std::string> make_ply_extra_attribute_names(const std::string_view base_name,
                                                            const size_t count) {
        std::vector<std::string> names;
        names.reserve(count);
        if (count == 1) {
            names.emplace_back(base_name);
            return names;
        }

        for (size_t i = 0; i < count; ++i) {
            names.emplace_back(std::format("{}_{}", base_name, i));
        }
        return names;
    }

    Result<void> validate_reserved_ply_extra_attribute_names(const std::span<const std::string> names,
                                                             const std::filesystem::path& output_path) {
        for (const auto& name : names) {
            if (is_reserved_ply_vertex_property_name(name)) {
                return make_error(
                    ErrorCode::INTERNAL_ERROR,
                    std::format(
                        "Extra PLY attribute name '{}' is reserved by the exporter; choose a different name",
                        name),
                    output_path);
            }
        }

        return {};
    }

    struct FastPropertyLayout {
        size_t vertex_count = 0;
        size_t vertex_stride = 0;
        size_t vertex_property_count = 0;

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

        [[nodiscard]] bool has_positions() const {
            return pos_x_offset != SIZE_MAX &&
                   pos_y_offset != SIZE_MAX &&
                   pos_z_offset != SIZE_MAX;
        }
        [[nodiscard]] bool has_opacity() const { return opacity_offset != SIZE_MAX; }
        [[nodiscard]] bool has_any_scaling() const {
            return scale_offsets[0] != SIZE_MAX ||
                   scale_offsets[1] != SIZE_MAX ||
                   scale_offsets[2] != SIZE_MAX;
        }
        [[nodiscard]] bool has_scaling() const {
            return scale_offsets[0] != SIZE_MAX &&
                   scale_offsets[1] != SIZE_MAX &&
                   scale_offsets[2] != SIZE_MAX;
        }
        [[nodiscard]] bool has_any_rotation() const {
            return rot_offsets[0] != SIZE_MAX ||
                   rot_offsets[1] != SIZE_MAX ||
                   rot_offsets[2] != SIZE_MAX ||
                   rot_offsets[3] != SIZE_MAX;
        }
        [[nodiscard]] bool has_rotation() const {
            return rot_offsets[0] != SIZE_MAX &&
                   rot_offsets[1] != SIZE_MAX &&
                   rot_offsets[2] != SIZE_MAX &&
                   rot_offsets[3] != SIZE_MAX;
        }
    };

    struct PlyImportValidation {
        std::vector<size_t> valid_rows;
        size_t invalid_count = 0;
        size_t non_finite_value_count = 0;
        size_t zero_rotation_count = 0;

        [[nodiscard]] size_t output_count(const size_t vertex_count) const {
            return valid_rows.empty() ? vertex_count : valid_rows.size();
        }
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
                LOG_ERROR("Failed to open file for mapping: {}", lfs::core::path_to_utf8(filepath));
                return false;
            }

            LARGE_INTEGER file_size_li;
            if (!GetFileSizeEx(file_handle, &file_size_li)) {
                LOG_ERROR("Failed to get file size: {}", lfs::core::path_to_utf8(filepath));
                return false;
            }
            size = static_cast<size_t>(file_size_li.QuadPart);

            mapping_handle = CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (!mapping_handle) {
                LOG_ERROR("Failed to create file mapping: {}", lfs::core::path_to_utf8(filepath));
                return false;
            }

            data = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
            if (!data) {
                LOG_ERROR("Failed to map view of file: {}", lfs::core::path_to_utf8(filepath));
                return false;
            }

            // PrefetchVirtualMemory is Win8+; resolved dynamically so the binary
            // still loads on older Windows.
            if (size > ply_constants::FILE_SIZE_THRESHOLD_MB * 1024 * 1024) {
                struct PrefetchEntry {
                    PVOID VirtualAddress;
                    SIZE_T NumberOfBytes;
                };
                using PrefetchFn = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PrefetchEntry*, ULONG);
                static const PrefetchFn prefetch = []() -> PrefetchFn {
                    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
                    return k32 ? reinterpret_cast<PrefetchFn>(
                                     GetProcAddress(k32, "PrefetchVirtualMemory"))
                               : nullptr;
                }();
                if (prefetch) {
                    PrefetchEntry entry{data, size};
                    prefetch(GetCurrentProcess(), 1, &entry, 0);
                }
            }

            return true;
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
                LOG_ERROR("Failed to open file for mapping: {}", lfs::core::path_to_utf8(filepath));
                return false;
            }

            struct stat st {};
            if (fstat(fd, &st) < 0) {
                LOG_ERROR("Failed to stat file: {}", lfs::core::path_to_utf8(filepath));
                return false;
            }
            size = st.st_size;

            data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (data == MAP_FAILED) {
                LOG_ERROR("Failed to mmap file: {}", lfs::core::path_to_utf8(filepath));
                return false;
            }

            // Kick off readahead so disk I/O overlaps with header parse + allocations.
            if (size > ply_constants::FILE_SIZE_THRESHOLD_MB * 1024 * 1024) {
                madvise(data, size, MADV_SEQUENTIAL);
                madvise(data, size, MADV_WILLNEED);
                posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
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

        LFS_ASSERT_MSG(data != nullptr, "PLY parser received a null buffer");

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
        bool has_format = false;
        bool has_vertex_element = false;
        bool parsing_vertex = false;
        std::unordered_set<std::string> vertex_property_names;
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
            const std::string_view line(line_start, line_len);
            if (line.starts_with("format ")) {
                LFS_ASSERT_MSG(!has_format, "PLY header must contain exactly one format line");
                LFS_ASSERT_MSG(line == "format binary_little_endian 1.0",
                               "Only PLY format binary_little_endian 1.0 is supported");
                is_binary = true;
                has_format = true;
            } else if (line_len >= 8 && std::strncmp(line_start, "element ", 8) == 0) {
                LFS_ASSERT_MSG(has_format, "PLY format line must precede element declarations");
                if (line_len >= 15 && std::strncmp(line_start, "element vertex ", 15) == 0) {
                    LFS_ASSERT_MSG(!has_vertex_element,
                                   "PLY header must not declare the vertex element more than once");
                    const std::string_view count_token(line_start + 15, line_len - 15);
                    if (!parse_size_token(count_token, layout.vertex_count)) {
                        throw std::runtime_error("Invalid PLY vertex count");
                    }
                    layout.vertex_stride = 0;
                    has_vertex_element = true;
                    parsing_vertex = true;
                } else {
                    parsing_vertex = false;
                }
            } else if (line_len >= 9 && std::strncmp(line_start, "property ", 9) == 0 && parsing_vertex) {
                std::string_view property_line(line_start + 9, line_len - 9);
                property_line = trim_ascii_whitespace(property_line);
                if (property_line.starts_with("list ")) {
                    throw std::runtime_error("PLY vertex list properties are not supported");
                }

                const size_t type_end = property_line.find_first_of(" \t\r");
                if (type_end == std::string_view::npos) {
                    throw std::runtime_error("Malformed PLY property line");
                }

                const std::string_view type = property_line.substr(0, type_end);
                std::string_view prop_name = trim_ascii_whitespace(property_line.substr(type_end));
                const size_t name_end = prop_name.find_first_of(" \t\r");
                if (name_end != std::string_view::npos) {
                    LFS_ASSERT_MSG(trim_ascii_whitespace(prop_name.substr(name_end)).empty(),
                                   "PLY property declarations must contain exactly a type and name");
                    prop_name = prop_name.substr(0, name_end);
                }
                if (prop_name.empty()) {
                    throw std::runtime_error("Malformed PLY property line");
                }
                LFS_ASSERT_MSG(is_valid_ply_property_name_token(prop_name),
                               "PLY vertex property names must be non-empty tokens");
                LFS_ASSERT_MSG(vertex_property_names.emplace(prop_name).second,
                               std::format("Duplicate PLY vertex property '{}'", prop_name));
                ++layout.vertex_property_count;

                size_t property_size = 0;
                if (!ply_scalar_property_size(type, property_size)) {
                    throw std::runtime_error(std::format("Unsupported PLY vertex property type '{}'", type));
                }

                const bool gaussian_float_property =
                    prop_name == ply_constants::POS_X ||
                    prop_name == ply_constants::POS_Y ||
                    prop_name == ply_constants::POS_Z ||
                    prop_name == ply_constants::OPACITY ||
                    prop_name.starts_with(ply_constants::DC_PREFIX) ||
                    prop_name.starts_with(ply_constants::REST_PREFIX) ||
                    prop_name.starts_with(ply_constants::SCALE_PREFIX) ||
                    prop_name.starts_with(ply_constants::ROT_PREFIX);
                LFS_ASSERT_MSG(!gaussian_float_property || is_float32_ply_type(type),
                               std::format("PLY Gaussian property '{}' must be float32", prop_name));

                if (is_float32_ply_type(type)) {
                    // Property recognition using first character + length
                    if (prop_name.size() == 1) {
                        switch (prop_name.front()) {
                        case 'x': layout.pos_x_offset = layout.vertex_stride; break;
                        case 'y': layout.pos_y_offset = layout.vertex_stride; break;
                        case 'z': layout.pos_z_offset = layout.vertex_stride; break;
                        default: break;
                        }
                    } else if (prop_name == ply_constants::OPACITY) {
                        layout.opacity_offset = layout.vertex_stride;
                    } else if (prop_name.starts_with(ply_constants::DC_PREFIX)) {
                        int idx = 0;
                        LFS_ASSERT_MSG(parse_property_index(prop_name,
                                                            ply_constants::DC_PREFIX,
                                                            ply_constants::MAX_DC_COMPONENTS,
                                                            idx),
                                       std::format("Invalid PLY DC coefficient property '{}'", prop_name));
                        layout.dc_offsets[idx] = layout.vertex_stride;
                        if (idx >= layout.dc_count)
                            layout.dc_count = idx + 1;
                    } else if (prop_name.starts_with(ply_constants::REST_PREFIX)) {
                        int idx = 0;
                        LFS_ASSERT_MSG(parse_property_index(prop_name,
                                                            ply_constants::REST_PREFIX,
                                                            ply_constants::MAX_REST_COMPONENTS,
                                                            idx),
                                       std::format("Invalid PLY higher-order coefficient property '{}'", prop_name));
                        layout.rest_offsets[idx] = layout.vertex_stride;
                        if (idx >= layout.rest_count)
                            layout.rest_count = idx + 1;
                    } else if (prop_name.starts_with(ply_constants::SCALE_PREFIX)) {
                        int idx = 0;
                        LFS_ASSERT_MSG(parse_property_index(prop_name, ply_constants::SCALE_PREFIX, 3, idx),
                                       std::format("Invalid PLY scale property '{}'", prop_name));
                        layout.scale_offsets[idx] = layout.vertex_stride;
                    } else if (prop_name.starts_with(ply_constants::ROT_PREFIX)) {
                        int idx = 0;
                        LFS_ASSERT_MSG(parse_property_index(prop_name, ply_constants::ROT_PREFIX, 4, idx),
                                       std::format("Invalid PLY rotation property '{}'", prop_name));
                        layout.rot_offsets[idx] = layout.vertex_stride;
                    }
                }

                if (property_size > std::numeric_limits<size_t>::max() - layout.vertex_stride) {
                    throw std::runtime_error("PLY vertex stride is too large");
                }
                layout.vertex_stride += property_size;
            } else if (line == "end_header") {
                if (!has_format || !is_binary || !has_vertex_element) {
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

    [[nodiscard]] float read_unaligned_float32(const char* ptr) {
        float value = 0.0f;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }

    template <typename Fn>
    void parallel_for_ply_rows(const size_t vertex_count,
                               const std::span<const size_t> rows,
                               const size_t block_size,
                               const Fn& fn) {
        if (!rows.empty()) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, rows.size(), block_size),
                              [&](const tbb::blocked_range<size_t>& range) {
                                  for (size_t output_row = range.begin(); output_row < range.end(); ++output_row) {
                                      fn(output_row, rows[output_row]);
                                  }
                              });
            return;
        }

        tbb::parallel_for(tbb::blocked_range<size_t>(0, vertex_count, block_size),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t row = range.begin(); row < range.end(); ++row) {
                                  fn(row, row);
                              }
                          });
    }

    void validate_ply_layout_for_import(const FastPropertyLayout& layout) {
        if (layout.vertex_count == 0) {
            throw std::runtime_error("PLY contains no vertices");
        }

        if (!layout.has_positions()) {
            throw std::runtime_error("PLY vertex properties must include x, y, and z");
        }

        if (layout.has_any_scaling() && !layout.has_scaling()) {
            throw std::runtime_error("PLY scaling properties must include scale_0, scale_1, and scale_2");
        }

        if (layout.has_any_rotation() && !layout.has_rotation()) {
            throw std::runtime_error("PLY rotation properties must include rot_0, rot_1, rot_2, and rot_3");
        }
        for (int i = 0; i < layout.dc_count; ++i) {
            LFS_ASSERT_MSG(layout.dc_offsets[i] != SIZE_MAX,
                           "PLY f_dc coefficient indices must be contiguous from zero");
        }
        for (int i = 0; i < layout.rest_count; ++i) {
            LFS_ASSERT_MSG(layout.rest_offsets[i] != SIZE_MAX,
                           "PLY f_rest coefficient indices must be contiguous from zero");
        }
        LFS_ASSERT_MSG(layout.dc_count == 0 ||
                           layout.dc_count % ply_constants::COLOR_CHANNELS == 0,
                       "PLY f_dc coefficient count must be divisible by three color channels");
        LFS_ASSERT_MSG(layout.rest_count == 0 ||
                           layout.rest_count % ply_constants::COLOR_CHANNELS == 0,
                       "PLY f_rest coefficient count must be divisible by three color channels");
        if (layout.rest_count > 0) {
            const int coefficients_per_channel =
                layout.rest_count / ply_constants::COLOR_CHANNELS;
            const int root = static_cast<int>(std::sqrt(coefficients_per_channel + 1));
            LFS_ASSERT_MSG(root * root == coefficients_per_channel + 1,
                           "PLY f_rest coefficient count does not describe a complete SH degree");
        }

        const auto assert_offset = [&](const size_t offset, const std::string_view property) {
            if (offset != SIZE_MAX) {
                LFS_ASSERT_MSG(offset <= layout.vertex_stride &&
                                   sizeof(float) <= layout.vertex_stride - offset,
                               std::format("PLY property '{}' extends past the vertex stride", property));
            }
        };
        assert_offset(layout.pos_x_offset, "x");
        assert_offset(layout.pos_y_offset, "y");
        assert_offset(layout.pos_z_offset, "z");
        assert_offset(layout.opacity_offset, "opacity");
        for (const size_t offset : layout.scale_offsets)
            assert_offset(offset, "scale");
        for (const size_t offset : layout.rot_offsets)
            assert_offset(offset, "rotation");
        for (const size_t offset : layout.dc_offsets)
            assert_offset(offset, "f_dc");
        for (const size_t offset : layout.rest_offsets)
            assert_offset(offset, "f_rest");
    }

    [[nodiscard]] PlyImportValidation validate_ply_vertex_payload(const char* vertex_data,
                                                                  const FastPropertyLayout& layout,
                                                                  const LoadOptions& options) {
        LOG_TIMER_TRACE("PLY payload validation");

        PlyImportValidation validation;

        const auto read_field = [&](const char* row, const size_t offset, bool& invalid) {
            LFS_DEBUG_ASSERT(offset <= layout.vertex_stride &&
                             sizeof(float) <= layout.vertex_stride - offset);
            const float value = read_unaligned_float32(row + offset);
            if (!std::isfinite(value)) {
                invalid = true;
                ++validation.non_finite_value_count;
            }
            return value;
        };

        for (size_t i = 0; i < layout.vertex_count; ++i) {
            if ((i % ply_constants::VALIDATION_CANCEL_INTERVAL) == 0) {
                throw_if_load_cancel_requested(options, "PLY validation cancelled");
            }

            const char* const row = vertex_data + i * layout.vertex_stride;
            bool invalid = false;

            (void)read_field(row, layout.pos_x_offset, invalid);
            (void)read_field(row, layout.pos_y_offset, invalid);
            (void)read_field(row, layout.pos_z_offset, invalid);

            if (layout.has_opacity()) {
                (void)read_field(row, layout.opacity_offset, invalid);
            }

            if (layout.has_scaling()) {
                (void)read_field(row, layout.scale_offsets[0], invalid);
                (void)read_field(row, layout.scale_offsets[1], invalid);
                (void)read_field(row, layout.scale_offsets[2], invalid);
            }

            if (layout.has_rotation()) {
                const float r0 = read_field(row, layout.rot_offsets[0], invalid);
                const float r1 = read_field(row, layout.rot_offsets[1], invalid);
                const float r2 = read_field(row, layout.rot_offsets[2], invalid);
                const float r3 = read_field(row, layout.rot_offsets[3], invalid);
                if (!invalid) {
                    const float norm_squared = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
                    if (!std::isfinite(norm_squared) ||
                        norm_squared <= ply_constants::MIN_ROTATION_NORM_SQUARED) {
                        invalid = true;
                        ++validation.zero_rotation_count;
                    }
                }
            }

            for (int coefficient = 0; coefficient < layout.dc_count; ++coefficient) {
                (void)read_field(row, layout.dc_offsets[coefficient], invalid);
            }
            for (int coefficient = 0; coefficient < layout.rest_count; ++coefficient) {
                (void)read_field(row, layout.rest_offsets[coefficient], invalid);
            }

            if (invalid) {
                if (validation.invalid_count == 0) {
                    validation.valid_rows.reserve(layout.vertex_count - 1);
                    for (size_t kept = 0; kept < i; ++kept) {
                        validation.valid_rows.push_back(kept);
                    }
                }
                ++validation.invalid_count;
            } else if (validation.invalid_count > 0) {
                validation.valid_rows.push_back(i);
            }
        }

        if (validation.invalid_count == 0) {
            return validation;
        }

        if (validation.invalid_count == layout.vertex_count) {
            throw std::runtime_error(std::format(
                "PLY contains no valid Gaussian splats after validation ({} invalid rows, {} non-finite values, {} zero-length rotations)",
                validation.invalid_count,
                validation.non_finite_value_count,
                validation.zero_rotation_count));
        }

        LOG_WARN("PLY validation will discard {} invalid splats before import ({} non-finite values, {} zero-length rotations)",
                 validation.invalid_count,
                 validation.non_finite_value_count,
                 validation.zero_rotation_count);
        return validation;
    }

    void extract_positions_to_host(const char* vertex_data,
                                   const FastPropertyLayout& layout,
                                   const std::span<const size_t> rows,
                                   float* output) {
        if (!layout.has_positions())
            return;

        const size_t stride = layout.vertex_stride;
        if (!rows.empty()) {
            parallel_for_ply_rows(layout.vertex_count, rows, ply_constants::BLOCK_SIZE_LARGE, [&](const size_t output_row, const size_t source_row) {
                output[output_row * 3 + 0] = read_unaligned_float32(vertex_data + source_row * stride + layout.pos_x_offset);
                output[output_row * 3 + 1] = read_unaligned_float32(vertex_data + source_row * stride + layout.pos_y_offset);
                output[output_row * 3 + 2] = read_unaligned_float32(vertex_data + source_row * stride + layout.pos_z_offset);
            });
            return;
        }

        const size_t count = layout.vertex_count;
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
                                          read_unaligned_float32(vertex_data + (i + 7) * stride + layout.pos_x_offset),
                                          read_unaligned_float32(vertex_data + (i + 6) * stride + layout.pos_x_offset),
                                          read_unaligned_float32(vertex_data + (i + 5) * stride + layout.pos_x_offset),
                                          read_unaligned_float32(vertex_data + (i + 4) * stride + layout.pos_x_offset),
                                          read_unaligned_float32(vertex_data + (i + 3) * stride + layout.pos_x_offset),
                                          read_unaligned_float32(vertex_data + (i + 2) * stride + layout.pos_x_offset),
                                          read_unaligned_float32(vertex_data + (i + 1) * stride + layout.pos_x_offset),
                                          read_unaligned_float32(vertex_data + i * stride + layout.pos_x_offset));

                                      __m256 y_vals = _mm256_set_ps(
                                          read_unaligned_float32(vertex_data + (i + 7) * stride + layout.pos_y_offset),
                                          read_unaligned_float32(vertex_data + (i + 6) * stride + layout.pos_y_offset),
                                          read_unaligned_float32(vertex_data + (i + 5) * stride + layout.pos_y_offset),
                                          read_unaligned_float32(vertex_data + (i + 4) * stride + layout.pos_y_offset),
                                          read_unaligned_float32(vertex_data + (i + 3) * stride + layout.pos_y_offset),
                                          read_unaligned_float32(vertex_data + (i + 2) * stride + layout.pos_y_offset),
                                          read_unaligned_float32(vertex_data + (i + 1) * stride + layout.pos_y_offset),
                                          read_unaligned_float32(vertex_data + i * stride + layout.pos_y_offset));

                                      __m256 z_vals = _mm256_set_ps(
                                          read_unaligned_float32(vertex_data + (i + 7) * stride + layout.pos_z_offset),
                                          read_unaligned_float32(vertex_data + (i + 6) * stride + layout.pos_z_offset),
                                          read_unaligned_float32(vertex_data + (i + 5) * stride + layout.pos_z_offset),
                                          read_unaligned_float32(vertex_data + (i + 4) * stride + layout.pos_z_offset),
                                          read_unaligned_float32(vertex_data + (i + 3) * stride + layout.pos_z_offset),
                                          read_unaligned_float32(vertex_data + (i + 2) * stride + layout.pos_z_offset),
                                          read_unaligned_float32(vertex_data + (i + 1) * stride + layout.pos_z_offset),
                                          read_unaligned_float32(vertex_data + i * stride + layout.pos_z_offset));

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
                                      output[i * 3 + 0] = read_unaligned_float32(vertex_data + i * stride + layout.pos_x_offset);
                                      output[i * 3 + 1] = read_unaligned_float32(vertex_data + i * stride + layout.pos_y_offset);
                                      output[i * 3 + 2] = read_unaligned_float32(vertex_data + i * stride + layout.pos_z_offset);
                                  }
                              });
        } else
#endif
        {
            LOG_TRACE("Using optimized scalar processing");

            tbb::parallel_for(tbb::blocked_range<size_t>(0, count, ply_constants::BLOCK_SIZE_LARGE),
                              [&](const tbb::blocked_range<size_t>& range) {
                                  for (size_t i = range.begin(); i < range.end(); ++i) {
                                      output[i * 3 + 0] = read_unaligned_float32(vertex_data + i * stride + layout.pos_x_offset);
                                      output[i * 3 + 1] = read_unaligned_float32(vertex_data + i * stride + layout.pos_y_offset);
                                      output[i * 3 + 2] = read_unaligned_float32(vertex_data + i * stride + layout.pos_z_offset);
                                  }
                              });
        }
    }

    // SH coefficient extraction with per-coefficient offsets (handles arbitrary PLY property order)
    void extract_sh_coefficients_to_host(const char* __restrict__ vertex_data,
                                         const FastPropertyLayout& layout,
                                         const std::span<const size_t> rows,
                                         const size_t* __restrict__ coeff_offsets,
                                         const int coeff_count, const int channels,
                                         float* __restrict__ output) {
        if (coeff_count == 0)
            return;

        const size_t stride = layout.vertex_stride;
        const int B = coeff_count / channels;

        parallel_for_ply_rows(layout.vertex_count, rows, ply_constants::BLOCK_SIZE_SMALL, [=](const size_t output_row, const size_t source_row) {
            const size_t base = source_row * stride;
            const size_t out_base = output_row * B * channels;
            for (int j = 0; j < coeff_count; ++j) {
                const size_t offset = coeff_offsets[j];
                const float value = (offset != SIZE_MAX)
                                        ? read_unaligned_float32(vertex_data + base + offset)
                                        : 0.0f;
                const int channel = j / B;
                const int b = j % B;
                output[out_base + b * channels + channel] = value;
            }
        });
    }

    void extract_sh_coefficients_to_swizzled_host(const char* __restrict__ vertex_data,
                                                  const FastPropertyLayout& layout,
                                                  const std::span<const size_t> rows,
                                                  const size_t* __restrict__ coeff_offsets,
                                                  const int coeff_count,
                                                  const int channels,
                                                  const std::uint32_t layout_coeffs_rest,
                                                  float* __restrict__ output) {
        if (coeff_count == 0 || layout_coeffs_rest == 0)
            return;

        const size_t stride = layout.vertex_stride;
        const int B = coeff_count / channels;
        const auto max_component_count =
            static_cast<std::uint32_t>(layout_coeffs_rest * static_cast<std::uint32_t>(channels));

        const auto extract_row = [=](const size_t output_row, const size_t source_row) {
            const size_t base = source_row * stride;
            for (int j = 0; j < coeff_count; ++j) {
                const size_t offset = coeff_offsets[j];
                if (offset == SIZE_MAX) {
                    continue;
                }

                const int channel = j / B;
                const int b = j % B;
                const auto canonical_component =
                    static_cast<std::uint32_t>(b * channels + channel);
                if (canonical_component >= max_component_count) {
                    continue;
                }

                const auto slot = canonical_component / 4u;
                const auto component = canonical_component % 4u;
                const size_t dst_offset =
                    static_cast<size_t>(lfs::core::sh_swizzled_index(
                        static_cast<std::uint32_t>(output_row),
                        slot,
                        layout_coeffs_rest)) *
                        4u +
                    component;
                output[dst_offset] =
                    read_unaligned_float32(vertex_data + base + offset);
            }
        };

        parallel_for_ply_rows(layout.vertex_count, rows, ply_constants::BLOCK_SIZE_SMALL, extract_row);
    }

    [[nodiscard]] Tensor tensor_from_host_floats(std::span<const float> data,
                                                 TensorShape shape,
                                                 const LoadOptions& options,
                                                 std::string_view name,
                                                 size_t capacity = 0) {
        if (shape.elements() != data.size()) {
            return {};
        }

        Tensor tensor;
        if (data.empty()) {
            tensor = Tensor::zeros(std::move(shape), Device::CUDA, DataType::Float32);
        } else if (options.splat_tensor_allocator) {
            const size_t row_capacity = capacity != 0
                                            ? capacity
                                            : (shape.rank() > 0 ? shape[0] : 0);
            tensor = options.splat_tensor_allocator(shape, row_capacity, DataType::Float32, name);
        } else {
            tensor = Tensor::empty(shape, Device::CUDA, DataType::Float32);
        }
        tensor.set_name(std::string{name});

        if (!tensor.is_valid() || data.empty()) {
            return tensor;
        }

        if (tensor.device() == Device::CUDA) {
            // Upload on the caller's current CUDA stream rather than the legacy default
            // stream. A default-stream cudaMemcpy inserts a device-wide barrier, so when a
            // background thread (e.g. the PLY-sequence streaming player) uploads here it
            // would serialise against the render thread's GPU work and stall it for the
            // whole copy. cudaMemcpyAsync on the thread's (non-blocking) stream avoids that.
            const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
            cudaError_t status = cudaMemcpyAsync(
                tensor.data_ptr(),
                data.data(),
                data.size_bytes(),
                cudaMemcpyHostToDevice,
                stream);
            if (status == cudaSuccess) {
                status = cudaStreamSynchronize(stream);
            }
            if (status != cudaSuccess) {
                throw std::runtime_error(std::format(
                    "CUDA upload failed for '{}': {} ({})",
                    name,
                    cudaGetErrorName(status),
                    cudaGetErrorString(status)));
            }
        } else {
            std::memcpy(tensor.data_ptr(), data.data(), data.size_bytes());
        }
        return tensor;
    }

    // Single property extraction to host memory
    void extract_property_to_host(const char* vertex_data, const FastPropertyLayout& layout,
                                  const std::span<const size_t> rows,
                                  size_t property_offset, float* output) {
        if (property_offset == SIZE_MAX)
            return;

        const size_t stride = layout.vertex_stride;
        parallel_for_ply_rows(layout.vertex_count, rows, ply_constants::BLOCK_SIZE_LARGE, [&](const size_t output_row, const size_t source_row) {
            output[output_row] = read_unaligned_float32(vertex_data + source_row * stride + property_offset);
        });
    }

    void extract_scaling_fused_to_host(const char* __restrict__ vertex_data,
                                       const FastPropertyLayout& layout,
                                       const std::span<const size_t> rows,
                                       float* __restrict__ output) {
        if (!layout.has_scaling())
            return;

        const size_t stride = layout.vertex_stride;
        const size_t s0 = layout.scale_offsets[0];
        const size_t s1 = layout.scale_offsets[1];
        const size_t s2 = layout.scale_offsets[2];

        parallel_for_ply_rows(layout.vertex_count, rows, ply_constants::BLOCK_SIZE_LARGE, [=](const size_t output_row, const size_t source_row) {
            const char* p = vertex_data + source_row * stride;
            output[output_row * 3 + 0] = read_unaligned_float32(p + s0);
            output[output_row * 3 + 1] = read_unaligned_float32(p + s1);
            output[output_row * 3 + 2] = read_unaligned_float32(p + s2);
        });
    }

    void extract_rotation_fused_to_host(const char* __restrict__ vertex_data,
                                        const FastPropertyLayout& layout,
                                        const std::span<const size_t> rows,
                                        float* __restrict__ output) {
        if (!layout.has_rotation())
            return;

        const size_t stride = layout.vertex_stride;
        const size_t r0 = layout.rot_offsets[0];
        const size_t r1 = layout.rot_offsets[1];
        const size_t r2 = layout.rot_offsets[2];
        const size_t r3 = layout.rot_offsets[3];

        parallel_for_ply_rows(layout.vertex_count, rows, ply_constants::BLOCK_SIZE_LARGE, [=](const size_t output_row, const size_t source_row) {
            const char* p = vertex_data + source_row * stride;
            output[output_row * 4 + 0] = read_unaligned_float32(p + r0);
            output[output_row * 4 + 1] = read_unaligned_float32(p + r1);
            output[output_row * 4 + 2] = read_unaligned_float32(p + r2);
            output[output_row * 4 + 3] = read_unaligned_float32(p + r3);
        });
    }

    // Pageable, not value-initialized. Pinning ~1.5 GB via cudaHostAlloc cost
    // ~700 ms on this path — far more than async H->D overlap could recoup.
    struct HostBuffer {
        float* ptr = nullptr;
        size_t count = 0;

        HostBuffer() = default;
        explicit HostBuffer(size_t element_count) : count(element_count) {
            if (count == 0)
                return;
            if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
                LOG_ERROR("Host buffer allocation size overflow for {} floats", count);
                count = 0;
                return;
            }
            ptr = static_cast<float*>(std::malloc(count * sizeof(float)));
            if (!ptr) {
                LOG_ERROR("malloc failed for {} MB host buffer", (count * sizeof(float)) / (1024 * 1024));
                count = 0;
            }
        }

        ~HostBuffer() {
            if (ptr)
                std::free(ptr);
        }

        HostBuffer(const HostBuffer&) = delete;
        HostBuffer& operator=(const HostBuffer&) = delete;

        HostBuffer(HostBuffer&& other) noexcept { swap(other); }
        HostBuffer& operator=(HostBuffer&& other) noexcept {
            if (this != &other) {
                if (ptr)
                    std::free(ptr);
                ptr = nullptr;
                count = 0;
                swap(other);
            }
            return *this;
        }

        [[nodiscard]] size_t bytes() const { return count * sizeof(float); }

    private:
        void swap(HostBuffer& other) noexcept {
            std::swap(ptr, other.ptr);
            std::swap(count, other.count);
        }
    };

    // Main function - returns SplatData
    [[nodiscard]] std::expected<SplatData, std::string>
    load_ply(const std::filesystem::path& filepath, const LoadOptions& options) {
        try {
            LOG_TIMER("PLY File Loading");
            auto start_time = std::chrono::high_resolution_clock::now();

            if (!std::filesystem::exists(filepath)) {
                std::string error_msg = std::format("PLY file does not exist: {}", lfs::core::path_to_utf8(filepath));
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            }
            throw_if_load_cancel_requested(options, "PLY load cancelled");

            // Memory map
            MMappedFile mapped_file;
            if (!mapped_file.map(filepath)) {
                LOG_ERROR("Failed to memory map PLY file: {}", lfs::core::path_to_utf8(filepath));
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

            if (layout.vertex_stride == 0) {
                std::string error_msg = "PLY header declares no vertex properties";
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            }

            const size_t body_bytes_available = file_size - data_offset;
            size_t body_bytes_required = 0;
            if (!checked_mul_size(layout.vertex_count, layout.vertex_stride, body_bytes_required)) {
                throw std::runtime_error("PLY header declares an impossibly large vertex body");
            }
            if (body_bytes_required > body_bytes_available) {
                const size_t missing = body_bytes_required - body_bytes_available;
                const size_t complete_vertices = body_bytes_available / layout.vertex_stride;
                std::string error_msg = std::format(
                    "PLY file is truncated: header declares {} vertices ({} bytes/vertex = {} bytes), "
                    "but file body has only {} bytes ({} complete vertices). Missing {} bytes (~{} MB). "
                    "Re-export or re-download the source file.",
                    layout.vertex_count, layout.vertex_stride, body_bytes_required,
                    body_bytes_available, complete_vertices, missing, missing / (1024 * 1024));
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            }

            validate_ply_layout_for_import(layout);
            const PlyImportValidation validation = validate_ply_vertex_payload(vertex_data, layout, options);
            throw_if_load_cancel_requested(options, "PLY load cancelled");

            const std::span<const size_t> rows_to_load(validation.valid_rows);
            const size_t N = validation.output_count(layout.vertex_count);
            LOG_INFO("Extracting {} Gaussians from PLY", N);

            // Determine SH dimensions
            int sh0_dim1 = 1, sh0_dim2 = ply_constants::COLOR_CHANNELS;
            int shN_dim1 = 0;
            std::uint32_t layout_rest = 0;
            if (layout.dc_count > 0 && layout.dc_count % ply_constants::COLOR_CHANNELS == 0) {
                sh0_dim1 = layout.dc_count / ply_constants::COLOR_CHANNELS;
            }
            if (layout.rest_count > 0 && layout.rest_count % ply_constants::COLOR_CHANNELS == 0) {
                shN_dim1 = layout.rest_count / ply_constants::COLOR_CHANNELS;
                layout_rest =
                    lfs::core::sh_rest_coefficients_for_degree(
                        static_cast<int>(std::sqrt(shN_dim1 + ply_constants::SH_DEGREE_OFFSET)) -
                        ply_constants::SH_DEGREE_OFFSET);
            }

            auto checked_float_count = [](const size_t a,
                                          const size_t b,
                                          const std::string_view label) {
                size_t result = 0;
                if (!checked_mul_size(a, b, result)) {
                    throw std::runtime_error(std::format(
                        "PLY load size overflow while allocating {}", label));
                }
                return result;
            };

            size_t shN_swizzled_count = 0;
            if (layout_rest > 0) {
                const size_t block_count = lfs::core::sh_swizzled_block_count(N);
                const size_t slot_floats =
                    static_cast<size_t>(lfs::core::sh_float4_slots_for_rest(layout_rest)) *
                    static_cast<size_t>(lfs::core::kShReorderSize) * 4u;
                shN_swizzled_count = checked_float_count(block_count, slot_floats, "SplatData.shN");
            }

            const size_t means_count = checked_float_count(N, 3, "SplatData.means");
            const size_t sh0_count = checked_float_count(
                checked_float_count(N, static_cast<size_t>(sh0_dim1), "SplatData.sh0"),
                static_cast<size_t>(sh0_dim2),
                "SplatData.sh0");
            const size_t opacity_count = N;
            const size_t scaling_count = checked_float_count(N, 3, "SplatData.scaling");
            const size_t rotation_count = checked_float_count(N, 4, "SplatData.rotation");

            HostBuffer host_means(means_count);
            HostBuffer host_sh0(sh0_count);
            HostBuffer host_shN_swizzled(shN_swizzled_count);
            HostBuffer host_opacity(opacity_count);
            HostBuffer host_scaling(scaling_count);
            HostBuffer host_rotation(rotation_count);

            if (!host_means.ptr || !host_sh0.ptr || !host_scaling.ptr ||
                !host_rotation.ptr || !host_opacity.ptr ||
                (shN_swizzled_count > 0 && !host_shN_swizzled.ptr)) {
                throw std::runtime_error("Failed to allocate host staging buffers for PLY load");
            }

            extract_positions_to_host(vertex_data, layout, rows_to_load, host_means.ptr);

            if (layout.dc_count > 0 && layout.dc_count % ply_constants::COLOR_CHANNELS == 0) {
                extract_sh_coefficients_to_host(vertex_data,
                                                layout,
                                                rows_to_load,
                                                layout.dc_offsets,
                                                layout.dc_count,
                                                ply_constants::COLOR_CHANNELS,
                                                host_sh0.ptr);
            } else {
                std::fill(host_sh0.ptr, host_sh0.ptr + host_sh0.count, 0.0f);
            }

            if (shN_swizzled_count > 0) {
                std::fill(host_shN_swizzled.ptr,
                          host_shN_swizzled.ptr + host_shN_swizzled.count,
                          0.0f);
                extract_sh_coefficients_to_swizzled_host(vertex_data,
                                                         layout,
                                                         rows_to_load,
                                                         layout.rest_offsets,
                                                         layout.rest_count,
                                                         ply_constants::COLOR_CHANNELS,
                                                         layout_rest,
                                                         host_shN_swizzled.ptr);
            }

            if (layout.has_opacity()) {
                extract_property_to_host(vertex_data, layout, rows_to_load, layout.opacity_offset, host_opacity.ptr);
            } else {
                std::fill(host_opacity.ptr, host_opacity.ptr + host_opacity.count, 0.0f);
            }

            if (layout.has_scaling()) {
                extract_scaling_fused_to_host(vertex_data, layout, rows_to_load, host_scaling.ptr);
            } else {
                std::fill(host_scaling.ptr,
                          host_scaling.ptr + host_scaling.count,
                          ply_constants::DEFAULT_LOG_SCALE);
            }

            if (layout.has_rotation()) {
                extract_rotation_fused_to_host(vertex_data, layout, rows_to_load, host_rotation.ptr);
            } else {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, N, ply_constants::BLOCK_SIZE_LARGE),
                                  [&](const tbb::blocked_range<size_t>& range) {
                                      for (size_t i = range.begin(); i < range.end(); ++i) {
                                          host_rotation.ptr[i * 4 + 0] = ply_constants::IDENTITY_QUATERNION_W;
                                          host_rotation.ptr[i * 4 + 1] = 0.0f;
                                          host_rotation.ptr[i * 4 + 2] = 0.0f;
                                          host_rotation.ptr[i * 4 + 3] = 0.0f;
                                      }
                                  });
            }

            const auto host_span = [](const HostBuffer& buffer) -> std::span<const float> {
                return {buffer.ptr, buffer.count};
            };

            LOG_DEBUG("Creating Tensor objects and uploading to CUDA");

            Tensor means = tensor_from_host_floats(host_span(host_means), {N, 3}, options, "SplatData.means");
            Tensor sh0 = tensor_from_host_floats(
                host_span(host_sh0),
                {N, static_cast<size_t>(sh0_dim1), static_cast<size_t>(sh0_dim2)},
                options,
                "SplatData.sh0");
            Tensor shN = tensor_from_host_floats(
                host_span(host_shN_swizzled),
                {host_shN_swizzled.count},
                options,
                "SplatData.shN",
                host_shN_swizzled.count);
            Tensor scaling = tensor_from_host_floats(host_span(host_scaling), {N, 3}, options, "SplatData.scaling");
            Tensor rotation = tensor_from_host_floats(host_span(host_rotation), {N, 4}, options, "SplatData.rotation");
            Tensor opacity = tensor_from_host_floats(host_span(host_opacity), {N, 1}, options, "SplatData.opacity");

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
                ply_constants::SCENE_SCALE_FACTOR,
                SplatData::ShNLayout::Swizzled);

            splat_data.set_tensor_allocator(options.splat_tensor_allocator);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            LOG_INFO("PLY loaded: {} MB, {} Gaussians with SH degree {} in {}ms ({} discarded)",
                     file_size / (1024 * 1024), splat_data.size(), sh_degree, duration.count(),
                     validation.invalid_count);

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
        using TensorWithNames = std::pair<Tensor, std::vector<std::string>>;

        [[nodiscard]] bool float_tensor_values_are_finite(const Tensor& values) {
            LFS_ASSERT(values.is_valid());
            LFS_ASSERT(values.dtype() == DataType::Float32);
            const Tensor cpu = values.cpu().contiguous();
            const float* const data = cpu.ptr<float>();
            for (size_t i = 0; i < cpu.numel(); ++i) {
                if (!std::isfinite(data[i])) {
                    return false;
                }
            }
            return true;
        }

        Result<void> validate_float_export_tensor(const Tensor& values,
                                                  const std::string_view label,
                                                  const size_t expected_rows,
                                                  const std::span<const int> allowed_ranks,
                                                  const std::optional<size_t> expected_columns,
                                                  const bool allow_empty_payload,
                                                  const std::filesystem::path& output_path) {
            if (!values.is_valid() || values.dtype() != DataType::Float32) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  std::format("{} must be a valid float32 tensor", label),
                                  output_path);
            }
            if (std::ranges::find(allowed_ranks, values.ndim()) == allowed_ranks.end()) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  std::format("{} has unsupported rank {}", label, values.ndim()),
                                  output_path);
            }
            if (static_cast<size_t>(values.size(0)) != expected_rows) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  std::format("{} row count {} does not match point count {}",
                                              label, values.size(0), expected_rows),
                                  output_path);
            }
            if (expected_columns &&
                (values.ndim() != 2 || static_cast<size_t>(values.size(1)) != *expected_columns)) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  std::format("{} must be shaped [N,{}]", label, *expected_columns),
                                  output_path);
            }
            for (size_t dim = 1; dim < values.ndim(); ++dim) {
                if (values.size(dim) == 0 && !allow_empty_payload) {
                    return make_error(ErrorCode::INTERNAL_ERROR,
                                      std::format("{} dimensions must be non-empty", label),
                                      output_path);
                }
            }
            if (!float_tensor_values_are_finite(values)) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  std::format("{} contains NaN or infinity", label),
                                  output_path);
            }
            return {};
        }

        Result<void> validate_point_cloud_for_ply_write(const PointCloud& pc,
                                                        const std::filesystem::path& output_path) {
            if (!pc.means.is_valid() || pc.means.ndim() != 2 || pc.means.size(1) != 3 ||
                pc.means.size(0) <= 0) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  "PointCloud.means must be a non-empty [N,3] tensor",
                                  output_path);
            }

            const size_t rows = static_cast<size_t>(pc.means.size(0));
            constexpr std::array rank2{2};
            constexpr std::array rank2_or_3{2, 3};
            if (auto result = validate_float_export_tensor(
                    pc.means, "PointCloud.means", rows, rank2, 3, false, output_path);
                !result) {
                return result;
            }

            const auto validate_optional = [&](const Tensor& values,
                                               const std::string_view label,
                                               const std::span<const int> ranks,
                                               const std::optional<size_t> columns,
                                               const bool allow_empty_payload = false) -> Result<void> {
                if (!values.is_valid()) {
                    return {};
                }
                return validate_float_export_tensor(
                    values, label, rows, ranks, columns, allow_empty_payload, output_path);
            };

            if (auto result = validate_optional(pc.normals, "PointCloud.normals", rank2, 3); !result)
                return result;
            if (auto result = validate_optional(pc.sh0, "PointCloud.sh0", rank2_or_3, std::nullopt); !result)
                return result;
            if (auto result = validate_optional(
                    pc.shN, "PointCloud.shN", rank2_or_3, std::nullopt, true);
                !result)
                return result;
            if (auto result = validate_optional(pc.opacity, "PointCloud.opacity", rank2, 1); !result)
                return result;
            if (auto result = validate_optional(pc.scaling, "PointCloud.scaling", rank2, 3); !result)
                return result;
            if (auto result = validate_optional(pc.rotation, "PointCloud.rotation", rank2, 4); !result)
                return result;

            if (pc.rotation.is_valid()) {
                const Tensor rotation = pc.rotation.cpu().contiguous();
                const float* const data = rotation.ptr<float>();
                for (size_t row = 0; row < rows; ++row) {
                    const float norm_squared = data[row * 4 + 0] * data[row * 4 + 0] +
                                               data[row * 4 + 1] * data[row * 4 + 1] +
                                               data[row * 4 + 2] * data[row * 4 + 2] +
                                               data[row * 4 + 3] * data[row * 4 + 3];
                    if (!std::isfinite(norm_squared) ||
                        norm_squared <= ply_constants::MIN_ROTATION_NORM_SQUARED) {
                        return make_error(ErrorCode::INTERNAL_ERROR,
                                          std::format("PointCloud.rotation row {} has a zero-length quaternion", row),
                                          output_path);
                    }
                }
            }

            if (pc.colors.is_valid()) {
                if (pc.colors.ndim() != 2 || pc.colors.size(0) != rows ||
                    pc.colors.size(1) != 3 ||
                    (pc.colors.dtype() != DataType::UInt8 && pc.colors.dtype() != DataType::Float32)) {
                    return make_error(ErrorCode::INTERNAL_ERROR,
                                      "PointCloud.colors must be [N,3] uint8 or float32",
                                      output_path);
                }
                if (pc.colors.dtype() == DataType::Float32 &&
                    !float_tensor_values_are_finite(pc.colors)) {
                    return make_error(ErrorCode::INTERNAL_ERROR,
                                      "PointCloud.colors contains NaN or infinity",
                                      output_path);
                }
                if (pc.colors.dtype() == DataType::Float32) {
                    const Tensor colors = pc.colors.cpu().contiguous();
                    const float* const data = colors.ptr<float>();
                    for (size_t i = 0; i < colors.numel(); ++i) {
                        if (data[i] < 0.0f || data[i] > 1.0f) {
                            return make_error(ErrorCode::INTERNAL_ERROR,
                                              "PointCloud float colors must be in [0,1]",
                                              output_path);
                        }
                    }
                }
            }

            return {};
        }

        void cleanup_finished_saves() {
            std::lock_guard lock(g_save_mutex);
            g_save_futures.erase(
                std::remove_if(g_save_futures.begin(), g_save_futures.end(),
                               [](const std::future<void>& f) {
                                   return !f.valid() || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                               }),
                g_save_futures.end());
        }

        std::vector<std::string> make_indexed_names(const std::string& prefix, const size_t count) {
            std::vector<std::string> names;
            names.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                names.emplace_back(prefix + std::to_string(i));
            }
            return names;
        }

        Result<void> validate_extra_attribute_tensor(const Tensor& values,
                                                     const std::filesystem::path& output_path) {
            if (!values.is_valid() || values.numel() == 0) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  "Extra PLY attribute tensors must not be empty",
                                  output_path);
            }

            if (values.ndim() != 1 && values.ndim() != 2) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  "Extra PLY attribute tensors must be shaped [N] or [N,C]",
                                  output_path);
            }
            if (values.dtype() == DataType::Bool) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  "Extra PLY attribute tensors must have a numeric dtype",
                                  output_path);
            }

            return {};
        }

        Result<void> validate_vertex_property_schema(const std::span<const TensorWithNames> float_blocks,
                                                     const size_t extra_block_offset,
                                                     const bool has_colors,
                                                     const std::filesystem::path& output_path) {
            std::unordered_set<std::string> seen;

            auto register_name = [&](const std::string_view name, const bool is_extra_attribute) -> Result<void> {
                if (!is_valid_ply_property_name_token(name)) {
                    return make_error(
                        ErrorCode::INTERNAL_ERROR,
                        std::format(
                            "PLY property name '{}' is invalid; names must be non-empty tokens without whitespace or control characters",
                            name),
                        output_path);
                }

                if (is_extra_attribute && is_reserved_ply_vertex_property_name(name)) {
                    return make_error(
                        ErrorCode::INTERNAL_ERROR,
                        std::format(
                            "Extra PLY attribute name '{}' is reserved by the exporter; choose a different name",
                            name),
                        output_path);
                }

                if (!seen.emplace(name).second) {
                    return make_error(
                        ErrorCode::INTERNAL_ERROR,
                        std::format("Duplicate PLY property name '{}' in vertex schema", name),
                        output_path);
                }

                return {};
            };

            if (has_colors) {
                for (const auto color_name : {std::string_view{"red"}, std::string_view{"green"}, std::string_view{"blue"}}) {
                    if (auto result = register_name(color_name, false); !result) {
                        return result;
                    }
                }
            }

            for (size_t block_index = 0; block_index < float_blocks.size(); ++block_index) {
                const bool is_extra_attribute = block_index >= extra_block_offset;
                const auto& names = float_blocks[block_index].second;
                for (const auto& name : names) {
                    if (auto result = register_name(name, is_extra_attribute); !result) {
                        return result;
                    }
                }
            }

            return {};
        }

        Result<TensorWithNames> prepare_extra_attribute_block(const PlyAttributeBlock& block,
                                                              const size_t expected_rows,
                                                              const std::filesystem::path& output_path) {
            if (auto result = validate_extra_attribute_tensor(block.values, output_path); !result) {
                return std::unexpected(result.error());
            }

            if (static_cast<size_t>(block.values.size(0)) != expected_rows) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  std::format("Extra PLY attribute row count {} does not match point count {}",
                                              block.values.size(0), expected_rows),
                                  output_path);
            }

            Tensor prepared = block.values;
            size_t cols = 1;
            if (prepared.ndim() == 1) {
                prepared = prepared.unsqueeze(1);
            } else {
                cols = static_cast<size_t>(prepared.size(1));
            }

            if (cols == 0 || block.names.size() != cols) {
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  std::format("Extra PLY attribute names count {} does not match tensor columns {}",
                                              block.names.size(), cols),
                                  output_path);
            }

            prepared = prepared.to(DataType::Float32).cpu().contiguous();
            if (!float_tensor_values_are_finite(prepared)) {
                return make_error(
                    ErrorCode::INTERNAL_ERROR,
                    "Extra PLY attribute tensor contains NaN or infinity",
                    output_path);
            }
            return TensorWithNames{std::move(prepared), block.names};
        }

        Result<std::vector<PlyAttributeBlock>> filter_extra_attributes_for_splat_export(
            const SplatData& splat_data,
            const std::vector<PlyAttributeBlock>& extra_attributes,
            const std::filesystem::path& output_path) {
            if (extra_attributes.empty() || !splat_data.has_deleted_mask()) {
                return extra_attributes;
            }

            const auto keep_mask = splat_data.deleted().logical_not();
            const auto raw_count = static_cast<size_t>(splat_data.size());
            const auto visible_count = static_cast<size_t>(splat_data.visible_count());

            std::vector<PlyAttributeBlock> filtered;
            filtered.reserve(extra_attributes.size());

            for (const auto& block : extra_attributes) {
                if (auto result = validate_extra_attribute_tensor(block.values, output_path); !result) {
                    return std::unexpected(result.error());
                }

                const auto rows = static_cast<size_t>(block.values.size(0));
                if (rows == visible_count) {
                    filtered.push_back(block);
                    continue;
                }

                if (rows != raw_count) {
                    return make_error(ErrorCode::INTERNAL_ERROR,
                                      std::format("Extra PLY attribute row count {} must match either raw point count {} or visible point count {}",
                                                  rows, raw_count, visible_count),
                                      output_path);
                }

                auto mask = keep_mask.device() == block.values.device() ? keep_mask : keep_mask.to(block.values.device());

                PlyAttributeBlock filtered_block;
                filtered_block.values = block.values.index_select(0, mask);
                filtered_block.names = block.names;
                filtered.push_back(std::move(filtered_block));
            }

            return filtered;
        }

        class ProgressReportingStreamBuf final : public std::streambuf {
        public:
            ProgressReportingStreamBuf(std::streambuf& target,
                                       ExportProgressCallback progress_callback,
                                       const size_t total_bytes)
                : target_(&target),
                  progress_callback_(std::move(progress_callback)),
                  total_bytes_(std::max<size_t>(total_bytes, 1)),
                  next_report_bytes_(kReportIntervalBytes) {}

            [[nodiscard]] bool cancelled() const { return cancelled_; }

            bool report_final() {
                bytes_written_ = std::max(bytes_written_, total_bytes_);
                return report(true);
            }

        protected:
            std::streamsize xsputn(const char* s, std::streamsize count) override {
                if (cancelled_ || !target_)
                    return 0;

                const auto written = target_->sputn(s, count);
                if (written > 0) {
                    bytes_written_ += static_cast<size_t>(written);
                    if (!report(false))
                        return 0;
                }
                return written;
            }

            int_type overflow(int_type ch) override {
                if (traits_type::eq_int_type(ch, traits_type::eof()))
                    return traits_type::not_eof(ch);
                if (cancelled_ || !target_)
                    return traits_type::eof();

                const auto result = target_->sputc(traits_type::to_char_type(ch));
                if (!traits_type::eq_int_type(result, traits_type::eof())) {
                    ++bytes_written_;
                    if (!report(false))
                        return traits_type::eof();
                }
                return result;
            }

            int sync() override {
                return target_ ? target_->pubsync() : -1;
            }

        private:
            static constexpr size_t kReportIntervalBytes = 16 * 1024 * 1024;

            bool report(const bool force) {
                if (!progress_callback_)
                    return true;
                if (!force && bytes_written_ < next_report_bytes_)
                    return true;

                next_report_bytes_ = bytes_written_ + kReportIntervalBytes;
                const auto ratio = static_cast<float>(
                    std::min<double>(1.0, static_cast<double>(bytes_written_) / static_cast<double>(total_bytes_)));
                if (!progress_callback_(ratio, "Writing PLY")) {
                    cancelled_ = true;
                    return false;
                }
                return true;
            }

            std::streambuf* target_ = nullptr;
            ExportProgressCallback progress_callback_;
            size_t total_bytes_ = 1;
            size_t bytes_written_ = 0;
            size_t next_report_bytes_ = 0;
            bool cancelled_ = false;
        };

        std::filesystem::path make_temp_output_path(const std::filesystem::path& output_path) {
            static std::atomic<uint64_t> counter{0};
            const auto now = std::chrono::steady_clock::now().time_since_epoch();
            const auto ticks = std::chrono::duration_cast<std::chrono::microseconds>(now).count();

            for (int attempt = 0; attempt < 128; ++attempt) {
                auto candidate = output_path;
                candidate += std::format(".{}.{}.tmp", ticks, counter.fetch_add(1, std::memory_order_relaxed));

                std::error_code ec;
                if (!std::filesystem::exists(candidate, ec))
                    return candidate;
            }

            auto candidate = output_path;
            candidate += std::format(".{}.tmp", counter.fetch_add(1, std::memory_order_relaxed));
            return candidate;
        }

        struct ScopedTempOutputFile {
            std::filesystem::path path;
            bool remove_on_destroy = true;

            ~ScopedTempOutputFile() {
                if (!remove_on_destroy || path.empty())
                    return;

                std::error_code ec;
                std::filesystem::remove(path, ec);
            }

            void dismiss() {
                remove_on_destroy = false;
            }
        };

        Result<void> replace_output_file(const std::filesystem::path& temp_path,
                                         const std::filesystem::path& output_path) {
#ifdef _WIN32
            if (!MoveFileExW(temp_path.wstring().c_str(),
                             output_path.wstring().c_str(),
                             MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Cannot replace output file: Windows error {}", GetLastError()),
                                  output_path);
            }
#else
            std::error_code ec;
            std::filesystem::rename(temp_path, output_path, ec);
            if (ec) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Cannot replace output file: {}", ec.message()),
                                  output_path);
            }
#endif
            return {};
        }

        Result<void> validate_written_ply_file(const std::filesystem::path& path,
                                               const bool binary,
                                               const size_t expected_vertices,
                                               const size_t expected_properties,
                                               const size_t expected_binary_stride) {
            try {
                if (binary) {
                    MMappedFile mapped_file;
                    LFS_ASSERT_MSG(mapped_file.map(path),
                                   "Could not reopen temporary PLY output for verification");
                    const auto parsed = parse_header(
                        static_cast<const char*>(mapped_file.data), mapped_file.size);
                    LFS_ASSERT_MSG(parsed.has_value(), "Could not parse temporary PLY output");
                    const auto& [body_offset, layout] = *parsed;
                    LFS_ASSERT_MSG(layout.vertex_count == expected_vertices,
                                   "PLY writer round-trip changed the vertex count");
                    LFS_ASSERT_MSG(layout.vertex_stride == expected_binary_stride,
                                   "PLY writer round-trip changed the vertex stride");
                    LFS_ASSERT_MSG(layout.vertex_property_count == expected_properties,
                                   "PLY writer round-trip changed the property count");
                    size_t body_size = 0;
                    LFS_ASSERT_MSG(checked_mul_size(expected_vertices, expected_binary_stride, body_size),
                                   "PLY writer verification size overflow");
                    LFS_ASSERT_MSG(body_offset <= mapped_file.size &&
                                       body_size == mapped_file.size - body_offset,
                                   "PLY writer emitted a body size inconsistent with its header");
                    return {};
                }

                std::ifstream stream(path);
                LFS_ASSERT_MSG(stream.is_open(),
                               "Could not reopen temporary ASCII PLY output for verification");
                bool saw_magic = false;
                bool saw_format = false;
                bool saw_vertex = false;
                bool parsing_vertex = false;
                bool saw_end_header = false;
                size_t property_count = 0;
                std::string line;
                while (std::getline(stream, line)) {
                    if (!line.empty() && line.back() == '\r')
                        line.pop_back();
                    if (!saw_magic) {
                        LFS_ASSERT_MSG(line == "ply", "ASCII PLY output has invalid magic");
                        saw_magic = true;
                        continue;
                    }
                    if (line == "format ascii 1.0") {
                        LFS_ASSERT_MSG(!saw_format, "ASCII PLY output has duplicate format lines");
                        saw_format = true;
                    } else if (line.starts_with("element vertex ")) {
                        LFS_ASSERT_MSG(!saw_vertex, "ASCII PLY output has duplicate vertex elements");
                        size_t count = 0;
                        LFS_ASSERT_MSG(parse_size_token(line.substr(15), count) &&
                                           count == expected_vertices,
                                       "ASCII PLY writer round-trip changed the vertex count");
                        saw_vertex = true;
                        parsing_vertex = true;
                    } else if (line.starts_with("element ")) {
                        parsing_vertex = false;
                    } else if (parsing_vertex && line.starts_with("property ")) {
                        ++property_count;
                    } else if (line == "end_header") {
                        saw_end_header = true;
                        break;
                    }
                }
                LFS_ASSERT_MSG(saw_magic && saw_format && saw_vertex && saw_end_header,
                               "ASCII PLY output has an incomplete header");
                LFS_ASSERT_MSG(property_count == expected_properties,
                               "ASCII PLY writer round-trip changed the property count");

                size_t rows = 0;
                while (std::getline(stream, line)) {
                    if (trim_ascii_whitespace(line).empty())
                        continue;
                    size_t tokens = 0;
                    bool inside_token = false;
                    for (const char ch : line) {
                        if (std::isspace(static_cast<unsigned char>(ch))) {
                            inside_token = false;
                        } else if (!inside_token) {
                            inside_token = true;
                            ++tokens;
                        }
                    }
                    LFS_ASSERT_MSG(tokens == expected_properties,
                                   "ASCII PLY writer emitted a row with the wrong property count");
                    ++rows;
                }
                LFS_ASSERT_MSG(rows == expected_vertices,
                               "ASCII PLY writer emitted a row count inconsistent with its header");
                return {};
            } catch (const std::exception& e) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("PLY writer round-trip verification failed: {}", e.what()),
                                  path);
            }
        }

        Result<void> write_ply_binary(const PointCloud& pc, const std::filesystem::path& output_path,
                                      bool binary = true,
                                      std::span<const PlyAttributeBlock> extra_attributes = {},
                                      ExportProgressCallback progress_callback = nullptr) {
            if (auto result = validate_point_cloud_for_ply_write(pc, output_path); !result) {
                return result;
            }

            // Write using tinyply
            tinyply::PlyFile ply;
            const size_t N = pc.means.size(0);

            std::vector<TensorWithNames> float_blocks;
            float_blocks.reserve(8 + extra_attributes.size());
            size_t attr_off = 0;

            auto append_builtin_block = [&](Tensor tensor, std::vector<std::string> fallback_names) {
                const size_t cols = static_cast<size_t>(tensor.size(1));
                std::vector<std::string> attrs = std::move(fallback_names);
                if (pc.attribute_names.size() >= attr_off + cols) {
                    attrs.assign(pc.attribute_names.begin() + static_cast<std::ptrdiff_t>(attr_off),
                                 pc.attribute_names.begin() + static_cast<std::ptrdiff_t>(attr_off + cols));
                }
                float_blocks.emplace_back(std::move(tensor), std::move(attrs));
                attr_off += cols;
            };

            append_builtin_block(pc.means.cpu().contiguous(), {"x", "y", "z"});

            if (pc.normals.is_valid()) {
                append_builtin_block(pc.normals.cpu().contiguous(), {"nx", "ny", "nz"});
            }

            auto process_sh = [](const Tensor& sh) -> Tensor {
                if (sh.ndim() == 3) {
                    auto transposed = sh.transpose(1, 2).contiguous();
                    return transposed.flatten(1).cpu().contiguous();
                }
                return sh.cpu().contiguous();
            };

            if (pc.sh0.is_valid()) {
                auto t = process_sh(pc.sh0);
                const auto cols = static_cast<size_t>(t.size(1));
                append_builtin_block(std::move(t), make_indexed_names("f_dc_", cols));
            }
            if (pc.shN.is_valid()) {
                auto t = process_sh(pc.shN);
                const auto cols = static_cast<size_t>(t.size(1));
                append_builtin_block(std::move(t), make_indexed_names("f_rest_", cols));
            }
            if (pc.opacity.is_valid())
                append_builtin_block(pc.opacity.cpu().contiguous(), {"opacity"});
            if (pc.scaling.is_valid()) {
                auto t = pc.scaling.cpu().contiguous();
                const auto cols = static_cast<size_t>(t.size(1));
                append_builtin_block(std::move(t), make_indexed_names("scale_", cols));
            }
            if (pc.rotation.is_valid()) {
                auto t = pc.rotation.cpu().contiguous();
                const auto cols = static_cast<size_t>(t.size(1));
                append_builtin_block(std::move(t), make_indexed_names("rot_", cols));
            }
            const size_t extra_block_offset = float_blocks.size();
            for (const auto& extra_attribute : extra_attributes) {
                auto prepared = prepare_extra_attribute_block(extra_attribute, N, output_path);
                if (!prepared) {
                    return std::unexpected(prepared.error());
                }
                float_blocks.emplace_back(std::move(*prepared));
            }

            // Optional colors: normalize to uchar red/green/blue before schema validation.
            Tensor colors_u8;
            if (pc.colors.is_valid() && pc.colors.numel() > 0) {
                auto colors_cpu = pc.colors.cpu().contiguous();
                if (colors_cpu.ndim() == 2 && static_cast<size_t>(colors_cpu.size(0)) == N && colors_cpu.size(1) == 3) {
                    if (colors_cpu.dtype() == DataType::UInt8) {
                        colors_u8 = colors_cpu;
                    } else if (colors_cpu.dtype() == DataType::Float32) {
                        // PLY convention: float colors are typically [0,1]
                        colors_u8 = (colors_cpu * 255.0f).clamp(0, 255).to(DataType::UInt8);
                    } else {
                        colors_u8 = colors_cpu.to(DataType::UInt8);
                    }
                }
            }

            if (auto result = validate_vertex_property_schema(float_blocks, extra_block_offset, colors_u8.is_valid(), output_path);
                !result) {
                return std::unexpected(result.error());
            }

            size_t estimated_write_bytes = 4096;

            if (colors_u8.is_valid()) {
                estimated_write_bytes += static_cast<size_t>(colors_u8.numel()) * sizeof(uint8_t);
                ply.add_properties_to_element(
                    "vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, N,
                    const_cast<uint8_t*>(colors_u8.ptr<uint8_t>()),
                    tinyply::Type::INVALID, 0);
            }

            for (auto& [t, attrs] : float_blocks) {
                LFS_ASSERT_MSG(attrs.size() == static_cast<size_t>(t.size(1)),
                               "PLY property names must match their tensor columns");

                estimated_write_bytes += static_cast<size_t>(t.size(0)) *
                                         static_cast<size_t>(t.size(1)) *
                                         sizeof(float);

                ply.add_properties_to_element(
                    "vertex", attrs, tinyply::Type::FLOAT32, t.size(0),
                    reinterpret_cast<uint8_t*>(const_cast<float*>(t.ptr<float>())),
                    tinyply::Type::INVALID, 0);
            }

            const size_t expected_properties =
                (colors_u8.is_valid() ? 3 : 0) +
                std::accumulate(float_blocks.begin(), float_blocks.end(), size_t{0},
                                [](const size_t total, const TensorWithNames& block) {
                                    return total + block.second.size();
                                });
            const size_t expected_binary_stride =
                (colors_u8.is_valid() ? 3 : 0) +
                (expected_properties - (colors_u8.is_valid() ? 3 : 0)) * sizeof(float);

            const auto temp_path = make_temp_output_path(output_path);
            ScopedTempOutputFile temp_file{temp_path};

            std::filebuf fb;
#ifdef _WIN32
            fb.open(temp_path.wstring(), std::ios::out | std::ios::binary);
#else
            fb.open(temp_path, std::ios::out | std::ios::binary);
#endif
            if (!fb.is_open()) {
                return make_error(ErrorCode::WRITE_FAILURE, "Cannot open temporary file", temp_path);
            }

            if (progress_callback) {
                ProgressReportingStreamBuf progress_buf(fb, std::move(progress_callback), estimated_write_bytes);
                std::ostream out_stream(&progress_buf);
                ply.write(out_stream, binary);
                out_stream.flush();
                const bool close_ok = fb.close() != nullptr;

                if (progress_buf.cancelled()) {
                    return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);
                }
                if (!out_stream.good() || !close_ok) {
                    return make_error(ErrorCode::WRITE_FAILURE, "Write failed", output_path);
                }
                if (!progress_buf.report_final()) {
                    return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);
                }
            } else {
                std::ostream out_stream(&fb);
                ply.write(out_stream, binary);
                out_stream.flush();
                const bool close_ok = fb.close() != nullptr;

                if (!out_stream.good() || !close_ok) {
                    return make_error(ErrorCode::WRITE_FAILURE, "Write failed", output_path);
                }
            }

            if (auto result = validate_written_ply_file(
                    temp_path, binary, N, expected_properties, expected_binary_stride);
                !result) {
                return result;
            }

            if (auto result = replace_output_file(temp_path, output_path); !result) {
                return std::unexpected(result.error());
            }

            temp_file.dismiss();
            return {};
        }

        Tensor normalize_rotation_cpu(Tensor rotation_cpu) {
            if (!rotation_cpu.is_valid() || rotation_cpu.dtype() != DataType::Float32 ||
                rotation_cpu.ndim() != 2 || rotation_cpu.size(1) != 4) {
                return rotation_cpu;
            }

            auto* const rotations = rotation_cpu.ptr<float>();
            const size_t rows = rotation_cpu.size(0);
            for (size_t row = 0; row < rows; ++row) {
                float* const quat = rotations + row * 4;
                const float norm = std::sqrt(
                    quat[0] * quat[0] +
                    quat[1] * quat[1] +
                    quat[2] * quat[2] +
                    quat[3] * quat[3]);
                const float inv_norm = 1.0f / std::max(norm, 1e-12f);
                quat[0] *= inv_norm;
                quat[1] *= inv_norm;
                quat[2] *= inv_norm;
                quat[3] *= inv_norm;
            }
            return rotation_cpu;
        }

        bool report_export_progress(const ExportProgressCallback& progress_callback,
                                    const float progress,
                                    const std::string& stage) {
            if (!progress_callback)
                return true;
            return progress_callback(std::clamp(progress, 0.0f, 1.0f), stage);
        }

        ExportProgressCallback scale_export_progress(const ExportProgressCallback& progress_callback,
                                                     const float start,
                                                     const float end) {
            if (!progress_callback)
                return nullptr;
            return [progress_callback, start, end](const float progress, const std::string& stage) {
                const float clamped = std::clamp(progress, 0.0f, 1.0f);
                return progress_callback(start + (end - start) * clamped, stage);
            };
        }

        Result<PointCloud> to_point_cloud_with_progress(const SplatData& splat_data,
                                                        const ExportProgressCallback& progress_callback,
                                                        const std::filesystem::path& output_path) {
            PointCloud pc;

            if (!report_export_progress(progress_callback, 0.0f, "Preparing splats"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);

            // Filter out deleted splats if deletion mask exists
            Tensor means, sh0, shN, opacity, scaling, rotation;

            // Export wants canonical [N, K, 3], but resident shN is swizzled. Unpack on
            // the host so saving cannot allocate a full canonical SH tensor in VRAM.
            const auto shN_canonical_cpu = splat_data.shN_canonical_cpu();

            if (splat_data.has_deleted_mask()) {
                // Create keep mask (inverse of deleted mask)
                const auto keep_mask = splat_data.deleted().logical_not();
                const auto keep_mask_cpu = keep_mask.cpu().contiguous();

                // Filter all tensors by keep mask
                means = splat_data.means().index_select(0, keep_mask);
                if (splat_data.sh0().is_valid())
                    sh0 = splat_data.sh0().index_select(0, keep_mask);
                if (shN_canonical_cpu.is_valid() && shN_canonical_cpu.numel() > 0)
                    shN = shN_canonical_cpu.index_select(0, keep_mask_cpu);
                if (splat_data.opacity_raw().is_valid())
                    opacity = splat_data.opacity_raw().index_select(0, keep_mask);
                if (splat_data.scaling_raw().is_valid())
                    scaling = splat_data.scaling_raw().index_select(0, keep_mask);
                if (splat_data.rotation_raw().is_valid())
                    rotation = splat_data.rotation_raw().index_select(0, keep_mask);
            } else {
                // No deletion mask, use original tensors (canonical shN view)
                means = splat_data.means();
                sh0 = splat_data.sh0();
                shN = shN_canonical_cpu;
                opacity = splat_data.opacity_raw();
                scaling = splat_data.scaling_raw();
                rotation = splat_data.rotation_raw();
            }

            if (!report_export_progress(progress_callback, 0.10f, "Copying positions"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);
            pc.means = means.cpu().contiguous();
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

            if (!report_export_progress(progress_callback, 0.25f, "Copying SH DC"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);
            if (sh0.is_valid())
                pc.sh0 = process_sh(sh0);

            if (!report_export_progress(progress_callback, 0.40f, "Copying SH coefficients"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);
            if (shN.is_valid())
                pc.shN = process_sh(shN);

            if (!report_export_progress(progress_callback, 0.65f, "Copying opacity"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);
            if (opacity.is_valid())
                pc.opacity = opacity.cpu().contiguous();

            if (!report_export_progress(progress_callback, 0.72f, "Copying scales"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);
            if (scaling.is_valid())
                pc.scaling = scaling.cpu().contiguous();

            if (!report_export_progress(progress_callback, 0.82f, "Copying rotations"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);
            if (rotation.is_valid()) {
                pc.rotation = normalize_rotation_cpu(rotation.cpu().contiguous());
            }

            pc.attribute_names = get_ply_attribute_names(splat_data);

            if (!report_export_progress(progress_callback, 1.0f, "PLY data prepared"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", output_path);
            return pc;
        }

    } // anonymous namespace

    PointCloud to_point_cloud(const SplatData& splat_data) {
        auto pc = to_point_cloud_with_progress(splat_data, nullptr, {});
        return pc ? std::move(*pc) : PointCloud{};
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
        // shN is stored swizzled at max SH degree; export all resident coefficients.
        const size_t layout_rest = splat_data.max_sh_coeffs_rest();
        if (layout_rest > 0)
            add_indexed_attrs("f_rest_", layout_rest * 3);

        attrs.emplace_back("opacity");

        if (splat_data.scaling_raw().is_valid())
            add_indexed_attrs("scale_", splat_data.scaling_raw().shape()[1]);
        if (splat_data.rotation_raw().is_valid())
            add_indexed_attrs("rot_", splat_data.rotation_raw().shape()[1]);

        return attrs;
    }

    Result<void> save_ply(const SplatData& splat_data, const PlySaveOptions& options) {
        if (!report_export_progress(options.progress_callback, 0.0f, "Preparing PLY"))
            return make_error(ErrorCode::CANCELLED, "Export cancelled by user", options.output_path);

        auto filtered_extra_attributes = filter_extra_attributes_for_splat_export(
            splat_data, options.extra_attributes, options.output_path);
        if (!filtered_extra_attributes) {
            return std::unexpected(filtered_extra_attributes.error());
        }

        auto pc = to_point_cloud_with_progress(
            splat_data,
            scale_export_progress(options.progress_callback, 0.05f, 0.80f),
            options.output_path);
        if (!pc)
            return std::unexpected(pc.error());

        auto filtered_options = options;
        filtered_options.extra_attributes = std::move(*filtered_extra_attributes);
        filtered_options.progress_callback = scale_export_progress(options.progress_callback, 0.80f, 1.0f);
        return save_ply(*pc, filtered_options);
    }

    Result<void> save_ply(const PointCloud& point_cloud, const PlySaveOptions& options) {
        if (auto result = validate_point_cloud_for_ply_write(point_cloud, options.output_path); !result) {
            return result;
        }

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
        for (const auto& extra_attribute : options.extra_attributes) {
            if (!extra_attribute.values.is_valid() || extra_attribute.values.numel() == 0)
                continue;
            if (extra_attribute.values.ndim() == 1) {
                floats_per_vertex += 1;
            } else if (extra_attribute.values.ndim() == 2) {
                floats_per_vertex += static_cast<size_t>(extra_attribute.values.size(1));
            }
        }

        size_t estimated_size = 1024 + vertex_count * floats_per_vertex * sizeof(float);
        if (point_cloud.colors.is_valid() && point_cloud.colors.numel() > 0) {
            estimated_size += vertex_count * 3; // uchar RGB
        }

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

        if (!report_export_progress(options.progress_callback, 0.1f, "Preparing PLY data"))
            return make_error(ErrorCode::CANCELLED, "Export cancelled by user", options.output_path);

        if (options.async) {
            cleanup_finished_saves();
            const std::lock_guard lock(g_save_mutex);
            g_save_futures.emplace_back(
                std::async(std::launch::async, [pc = point_cloud, opts = options]() {
                    auto write_progress_callback = scale_export_progress(opts.progress_callback, 0.5f, 1.0f);
                    if (const auto result = write_ply_binary(
                            pc, opts.output_path, opts.binary, opts.extra_attributes, write_progress_callback);
                        !result) {
                        LOG_ERROR("PLY save failed: {}", result.error().format());
                    }
                }));
        } else {
            if (!report_export_progress(options.progress_callback, 0.5f, "Writing PLY"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", options.output_path);

            auto write_progress_callback = scale_export_progress(options.progress_callback, 0.5f, 1.0f);
            if (const auto result = write_ply_binary(
                    point_cloud, options.output_path, options.binary, options.extra_attributes, write_progress_callback);
                !result) {
                return std::unexpected(result.error());
            }

            if (!report_export_progress(options.progress_callback, 1.0f, "Done"))
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user", options.output_path);
        }
        return {};
    }

    bool ply_has_faces(const std::filesystem::path& filepath) {
        if (!std::filesystem::exists(filepath))
            return false;

        std::ifstream file;
        if (!lfs::core::open_file_for_read(filepath, std::ios::binary, file))
            return false;

        std::string line;
        while (std::getline(file, line)) {
            if (line.find("end_header") != std::string::npos)
                break;
            if (line.compare(0, 13, "element face ") == 0) {
                const int count = std::atoi(line.c_str() + 13);
                if (count > 0)
                    return true;
            }
        }
        return false;
    }

    bool is_gaussian_splat_ply(const std::filesystem::path& filepath) {
        if (!std::filesystem::exists(filepath))
            return false;

        std::ifstream file;
        if (!lfs::core::open_file_for_read(filepath, std::ios::binary, file))
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

    std::expected<lfs::core::PointCloud, std::string> load_ply_point_cloud(const std::filesystem::path& filepath,
                                                                           const LoadOptions& options) {
        constexpr uint8_t DEFAULT_COLOR = 255;

        if (!std::filesystem::exists(filepath)) {
            return std::unexpected(std::format("File not found: {}", lfs::core::path_to_utf8(filepath)));
        }

        if (is_load_cancel_requested(options)) {
            return std::unexpected("Load cancelled");
        }

        try {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(filepath, std::ios::binary, file)) {
                return std::unexpected(std::format("Cannot open: {}", lfs::core::path_to_utf8(filepath)));
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

            std::shared_ptr<tinyply::PlyData> normals;
            bool has_normals = false;
            try {
                normals = ply.request_properties_from_element("vertex", {"nx", "ny", "nz"});
                has_normals = true;
            } catch (...) {}

            throw_if_load_cancel_requested(options, "PLY read cancelled");
            ply.read(file);
            throw_if_load_cancel_requested(options, "PLY read cancelled");

            const size_t N = vertices->count;
            LFS_ASSERT_MSG(N > 0, "PLY point cloud contains no vertices");
            LOG_DEBUG("Point cloud: {} points", N);

            using namespace lfs::core;
            Tensor positions = Tensor::zeros({N, 3}, Device::CPU, DataType::Float32);
            float* const pos_ptr = positions.ptr<float>();

            if (vertices->t == tinyply::Type::FLOAT32) {
                LFS_ASSERT_MSG(vertices->buffer.size_bytes() == N * 3 * sizeof(float),
                               "PLY vertex buffer size is inconsistent with its count");
                std::memcpy(pos_ptr, vertices->buffer.get(), N * 3 * sizeof(float));
            } else if (vertices->t == tinyply::Type::FLOAT64) {
                LFS_ASSERT_MSG(vertices->buffer.size_bytes() == N * 3 * sizeof(double),
                               "PLY vertex buffer size is inconsistent with its count");
                const auto* src = reinterpret_cast<const double*>(vertices->buffer.get());
                for (size_t i = 0; i < N * 3; ++i) {
                    if ((i % 4096) == 0) {
                        throw_if_load_cancel_requested(options, "PLY vertex conversion cancelled");
                    }
                    pos_ptr[i] = static_cast<float>(src[i]);
                }
            } else {
                return std::unexpected("Unsupported vertex type");
            }
            LFS_ASSERT_MSG(float_tensor_values_are_finite(positions),
                           "PLY point positions must be finite");

            Tensor color_tensor;
            if (has_colors && colors) {
                LFS_ASSERT_MSG(colors->count == N,
                               "PLY color count must match the vertex count");
                if (colors->t == tinyply::Type::UINT8) {
                    LFS_ASSERT_MSG(colors->buffer.size_bytes() == N * 3,
                                   "PLY color buffer size is inconsistent with its count");
                    color_tensor = Tensor::zeros({N, 3}, Device::CPU, DataType::UInt8);
                    std::memcpy(color_tensor.ptr<uint8_t>(), colors->buffer.get(), N * 3);
                } else if (colors->t == tinyply::Type::FLOAT32) {
                    LFS_ASSERT_MSG(colors->buffer.size_bytes() == N * 3 * sizeof(float),
                                   "PLY color buffer size is inconsistent with its count");
                    Tensor float_colors = Tensor::zeros({N, 3}, Device::CPU, DataType::Float32);
                    std::memcpy(float_colors.ptr<float>(), colors->buffer.get(), N * 3 * sizeof(float));
                    LFS_ASSERT_MSG(float_tensor_values_are_finite(float_colors),
                                   "PLY float colors must be finite");
                    color_tensor = (float_colors * 255.0f).clamp(0, 255).to(DataType::UInt8);
                } else {
                    color_tensor = Tensor::full({N, 3}, DEFAULT_COLOR, Device::CPU, DataType::UInt8);
                }
            } else {
                color_tensor = Tensor::full({N, 3}, DEFAULT_COLOR, Device::CPU, DataType::UInt8);
            }

            Tensor normal_tensor;
            if (has_normals && normals) {
                LFS_ASSERT_MSG(normals->count == N,
                               "PLY normal count must match the vertex count");
                normal_tensor = Tensor::zeros({N, 3}, Device::CPU, DataType::Float32);
                float* const normal_ptr = normal_tensor.ptr<float>();
                if (normals->t == tinyply::Type::FLOAT32) {
                    LFS_ASSERT_MSG(normals->buffer.size_bytes() == N * 3 * sizeof(float),
                                   "PLY normal buffer size is inconsistent with its count");
                    std::memcpy(normal_ptr, normals->buffer.get(), N * 3 * sizeof(float));
                } else if (normals->t == tinyply::Type::FLOAT64) {
                    LFS_ASSERT_MSG(normals->buffer.size_bytes() == N * 3 * sizeof(double),
                                   "PLY normal buffer size is inconsistent with its count");
                    const auto* src = reinterpret_cast<const double*>(normals->buffer.get());
                    for (size_t i = 0; i < N * 3; ++i) {
                        if ((i % 4096) == 0) {
                            throw_if_load_cancel_requested(options, "PLY normal conversion cancelled");
                        }
                        normal_ptr[i] = static_cast<float>(src[i]);
                    }
                } else {
                    normal_tensor = Tensor();
                }
                LFS_ASSERT_MSG(!normal_tensor.is_valid() || float_tensor_values_are_finite(normal_tensor),
                               "PLY normals must be finite");
            }

            throw_if_load_cancel_requested(options, "PLY point cloud load cancelled");
            PointCloud point_cloud(std::move(positions), std::move(color_tensor));
            point_cloud.normals = std::move(normal_tensor);
            return point_cloud;
        } catch (const LoadCancelledError& e) {
            return std::unexpected(std::string(e.what()));
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Load failed: {}", e.what()));
        }
    }

} // namespace lfs::io

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// PLY import past the float4 slot wrap: the importer writes each row's SH rest into float4-swizzled host storage at
// sh_swizzled_index(row, slot), (row / 32) * 384 + slot * 32 + row % 32 at SH degree 3, which passes 2^32 in the block
// of row 357,913,920. The file has 357,913,984 rows (LFS_TEST_LARGE_PLY_ROWS for a smaller check). The first 256 rows
// and the last 128 (around the wrap at the default size) hold SH values; every other row is a zero splat with the
// identity rotation. The importer reads a PLY as Gaussian splats only when it has opacity, scales and rotations, and
// drops rows with a zero rotation, so every row is written. The imported SH of the head and far rows must equal what
// the file holds; an index computed in 32 bits writes the far rows' SH over the first rows' and leaves the far rows
// zero.
//
// Runs with LFS_TEST_HUGE_SH=1 and, on a shared machine, alone in an exclusive hold of the GPU lock: about 89 GB of
// host staging and as much device memory (estimated), and an 84.5 GB file in TMPDIR while it runs.

#include "large_model_test_support.hpp"

#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "io/loader.hpp"
#include "lfs/training/sh_value_codec.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::ShFormat;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;

    constexpr size_t kDefaultRows = lm::kSlotWrapRow + 2 * lm::kReorder;
    constexpr size_t kFarRows = 4 * lm::kReorder; // the last rows: 64 below the wrap and 64 past it by default
    constexpr size_t kHeadRows = 256;

    size_t rows_from_environment() {
        if (const char* value = std::getenv("LFS_TEST_LARGE_PLY_ROWS"))
            return static_cast<size_t>(std::strtoull(value, nullptr, 10));
        return kDefaultRows;
    }

    // x y z, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3: 59 floats a row.
    constexpr size_t kWidth = 59;
    constexpr size_t kRestColumn = 6;
    constexpr size_t kRotColumn = 55;

    // f_rest_j of row r (j = channel * 15 + coefficient): integers / 1024, exact in fp32.
    float rest_value(const size_t r, const size_t j) {
        return static_cast<float>(static_cast<int>((r * 45 + j) % 2047) - 1023) / 1024.0f;
    }

    std::vector<float> row_values(const size_t r) {
        std::vector<float> v(kWidth, 0.0f);
        v[0] = static_cast<float>(r % 1024);
        v[1] = static_cast<float>(r / 1024 % 1024);
        v[2] = 0.5f;
        for (size_t j = 0; j < 45; ++j)
            v[kRestColumn + j] = rest_value(r, j);
        v[52] = -2.0f;
        v[53] = -2.0f;
        v[54] = -2.0f;
        v[kRotColumn] = 1.0f;
        return v;
    }

    TEST(LargeModelPly, ImportPastTheSlotWrap) {
#ifdef _WIN32
        GTEST_SKIP() << "the PLY writer uses POSIX file calls";
#else
        const size_t rows = rows_from_environment();
        ASSERT_GE(rows, kHeadRows + kFarRows);
        const size_t far_begin = rows - kFarRows;
        if (const auto reason = lm::skip_reason(rows * 248, rows * 248))
            GTEST_SKIP() << *reason;
        lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(false);
        struct Restore {
            ~Restore() { lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt); }
        } restore;
        static std::atomic<int> counter{0};
        const fs::path folder =
            fs::temp_directory_path() /
            std::format("lfs_large_model_ply_{}_{}", std::chrono::steady_clock::now().time_since_epoch().count(), counter++);
        fs::create_directories(folder);
        struct Remove {
            fs::path folder;
            ~Remove() {
                std::error_code ignored;
                fs::remove_all(folder, ignored);
            }
        } remove{folder};

        const fs::path path = folder / "large.ply";
        std::string header = std::format("ply\nformat binary_little_endian 1.0\nelement vertex {}\n", rows);
        for (const char* name : {"x", "y", "z"})
            header += std::format("property float {}\n", name);
        for (int i = 0; i < 3; ++i)
            header += std::format("property float f_dc_{}\n", i);
        for (int i = 0; i < 45; ++i)
            header += std::format("property float f_rest_{}\n", i);
        header += "property float opacity\n";
        for (int i = 0; i < 3; ++i)
            header += std::format("property float scale_{}\n", i);
        for (int i = 0; i < 4; ++i)
            header += std::format("property float rot_{}\n", i);
        header += "end_header\n";
        {
            // Every row in pieces of 1M rows: a zero splat with the identity rotation, or the row's values for the
            // head and far rows.
            const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
            ASSERT_GE(fd, 0) << "can't create " << path;
            ASSERT_EQ(::write(fd, header.data(), header.size()), static_cast<ssize_t>(header.size()));
            constexpr size_t kPiece = size_t{1} << 20;
            std::vector<float> piece(kPiece * kWidth);
            for (size_t first = 0; first < rows; first += kPiece) {
                const size_t count = std::min(kPiece, rows - first);
                for (size_t i = 0; i < count; ++i) {
                    float* const row = piece.data() + i * kWidth;
                    const size_t r = first + i;
                    if (r < kHeadRows || r >= far_begin) {
                        const auto v = row_values(r);
                        std::copy(v.begin(), v.end(), row);
                    } else {
                        std::fill(row, row + kWidth, 0.0f);
                        row[kRotColumn] = 1.0f;
                    }
                }
                const size_t bytes = count * kWidth * sizeof(float);
                const auto* at = reinterpret_cast<const char*>(piece.data());
                for (size_t done = 0; done < bytes;) {
                    const ssize_t written = ::write(fd, at + done, bytes - done);
                    ASSERT_GT(written, 0) << "writing " << path << " stopped at row " << first;
                    done += static_cast<size_t>(written);
                }
            }
            ::close(fd);
        }

        auto loaded = lfs::io::Loader::create()->load(path);
        ASSERT_TRUE(loaded) << loaded.error().format();
        const auto* const data = std::get_if<std::shared_ptr<SplatData>>(&loaded->data);
        ASSERT_NE(data, nullptr);
        ASSERT_TRUE(*data);
        const SplatData& model = **data;
        ASSERT_EQ(static_cast<size_t>(model.size()), rows) << "rows were dropped or added on import";
        ASSERT_EQ(model.get_max_sh_degree(), 3);

        // The written rows' SH: from float4 storage directly, from Q16 storage within one quantization step.
        const bool q16 = model.shN_value_quantized();
        const ShFormat format = q16 ? ShFormat::Q16 : model.shN().dtype() == DataType::Float16 ? ShFormat::Float16
                                                                                               : ShFormat::Float32;
        for (const auto& [first, count] : {std::pair{size_t{0}, kHeadRows}, std::pair{far_begin, kFarRows}}) {
            const lm::Store store{.format = format, .rows = rows, .values = model.shN(),
                                  .bounds = q16 ? model.shN_value_bounds() : Tensor{}};
            // Decode the rows to canonical on the host through the CPU codec (64-bit offsets).
            const size_t aligned_first = first / lm::kQuantBlock * lm::kQuantBlock;
            const size_t aligned_end = (first + count + lm::kQuantBlock - 1) / lm::kQuantBlock * lm::kQuantBlock;
            const size_t n = std::min(aligned_end, rows) - aligned_first;
            const size_t padded = (n + lm::kReorder - 1) / lm::kReorder * lm::kReorder;
            lm::Store host{.format = format, .rows = n,
                           .values = lm::values_of_rows(store.values, format, aligned_first, padded).cpu().contiguous()};
            if (q16)
                host.bounds = store.bounds.slice(0, aligned_first / lm::kQuantBlock * 2,
                                                 (aligned_first + n + lm::kQuantBlock - 1) / lm::kQuantBlock * 2)
                                  .cpu()
                                  .contiguous();
            lm::Store canonical = lm::cpu_store(ShFormat::Canonical, n);
            lm::convert(host, canonical, {.count = n});
            const auto* const values = canonical.values.ptr<float>();
            size_t wrong = 0;
            std::string first_wrong;
            for (size_t r = first; r < first + count; ++r)
                for (uint32_t cell = 0; cell < lm::kCells; ++cell) {
                    const float want = rest_value(r, (cell % 3) * 15 + cell / 3);
                    const float got = values[(r - aligned_first) * lm::kCells + cell];
                    const float tolerance =
                        q16 ? (host.bounds.ptr<float>()[(r - aligned_first) / lm::kQuantBlock * 2 + 1] -
                               host.bounds.ptr<float>()[(r - aligned_first) / lm::kQuantBlock * 2]) /
                                  65535.0f
                            : 0.0f;
                    if (std::fabs(got - want) > tolerance && wrong++ == 0)
                        first_wrong = std::format("row {}, cell {}: {} instead of {}", r, cell, got, want);
                }
            EXPECT_EQ(wrong, 0u) << lm::format_name(format) << " SH of rows " << first << "-" << first + count - 1 << ": "
                                 << wrong << " values wrong; first " << first_wrong;
        }
#endif
    }

} // namespace

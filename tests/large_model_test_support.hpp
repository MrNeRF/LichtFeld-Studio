/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Shared by the tests past the 32-bit wraps of large models (tests/test_large_model_*.cpp).
//
// At SH degree 3 a row holds 45 canonical values, 45 Q16 cells or 12 float4 slots (48 floats), in 32-row swizzle
// blocks, with Q16 bounds per 256 rows. An element index computed in 32 bits wraps:
// - in float4 storage (fp32, f16) in the block of row 89,478,464, from its slot 8 on;
// - in Q16 storage in the block of row 95,443,712, from its cell 8 on;
// - in canonical rows (row * 45 + cell) at row 95,443,717;
// - as a float4 slot index (row / 32 * 384 + slot * 32 + row % 32) in the block of row 357,913,920.
//
// The tests store values only in a few blocks of rows: H (the first rows, where a wrapped write lands), Z (rows that
// must keep their contents), F (around the float4 wrap) and Q (from one block before the Q16 wrap to the end). Each
// block starts and ends on a multiple of 256 rows, so the blocks alone, packed one after another into a small
// "compact" store, hold exactly the same swizzle blocks and Q16 bounds. The CPU codec (64-bit offsets) on the compact
// store is the reference for the GPU result on the big one.
//
// Two tiers, by the device memory a test needs (its own count of the tensors it allocates):
// - LFS_TEST_LARGE_SH=1: at most 28 GB, which fits a 32 GB card;
// - LFS_TEST_HUGE_SH=1: more than that.

#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor.hpp"
#include "core/tensor_sh.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lfs::test::large_model {

    using core::DataType;
    using core::Device;
    using core::ShFormat;
    using core::Tensor;

    // ---- Tiers -------------------------------------------------------------------------------------------------

    inline constexpr double kTierOneDeviceGB = 28.0;
    inline constexpr double kGB = 1e9;

    inline bool env_is_one(const char* const name) {
        const char* const value = std::getenv(name);
        return value != nullptr && std::string(value) == "1";
    }

    // Empty when a test that needs `device_bytes` of device memory (and `host_bytes` of host memory) may run,
    // otherwise the reason it skips, with what it needs.
    inline std::optional<std::string> skip_reason(const size_t device_bytes, const size_t host_bytes = 0) {
        const double device_gb = static_cast<double>(device_bytes) / kGB;
        const double host_gb = static_cast<double>(host_bytes) / kGB;
        const bool tier_one = device_gb <= kTierOneDeviceGB;
        const char* const variable = tier_one ? "LFS_TEST_LARGE_SH" : "LFS_TEST_HUGE_SH";
        if (env_is_one(variable))
            return std::nullopt;
        return std::format("needs {:.1f} GB of device memory{}: set {}=1 to run (tier {}, {} 28 GB of device memory)",
                           device_gb, host_bytes ? std::format(" and {:.1f} GB of host memory", host_gb) : "",
                           variable, tier_one ? 1 : 2, tier_one ? "at most" : "more than");
    }

    // ---- SH degree 3 and its wrap rows -------------------------------------------------------------------------

    inline constexpr uint32_t kRest = 15;       // SH degree 3
    inline constexpr uint32_t kCells = 45;      // canonical values and Q16 cells a row
    inline constexpr uint32_t kSlots = 12;      // float4 slots a row
    inline constexpr uint32_t kFloatWidth = 48; // floats a row in float4 storage
    inline constexpr size_t kReorder = 32;      // rows a swizzle block
    inline constexpr size_t kQuantBlock = 256;  // rows a Q16 bounds block
    inline constexpr size_t kTwoTo32 = size_t{1} << 32;

    inline constexpr size_t kFloat4WrapRow = 89'478'464;
    inline constexpr size_t kQ16WrapRow = 95'443'712;
    inline constexpr size_t kCanonicalWrapRow = 95'443'717;
    inline constexpr size_t kSlotWrapRow = 357'913'920;

    // The block of each wrap row is the first whose last element passes 2^32.
    static_assert(kFloat4WrapRow % kReorder == 0 && kQ16WrapRow % kReorder == 0 && kSlotWrapRow % kReorder == 0);
    static_assert(kFloat4WrapRow * kFloatWidth < kTwoTo32 && (kFloat4WrapRow + kReorder) * kFloatWidth > kTwoTo32);
    static_assert(kQ16WrapRow * kCells < kTwoTo32 && (kQ16WrapRow + kReorder) * kCells > kTwoTo32);
    static_assert((kCanonicalWrapRow - 1) * kCells + kCells - 1 < kTwoTo32 &&
                  kCanonicalWrapRow * kCells + kCells - 1 >= kTwoTo32);
    static_assert(kSlotWrapRow * kSlots < kTwoTo32 && (kSlotWrapRow + kReorder) * kSlots > kTwoTo32);

    // Rows of every codec and storage test: past the float4, Q16 and canonical wraps, and past them as a row count
    // times a row's width (the codec's destination enumeration).
    inline constexpr size_t kRows = kQ16WrapRow + 2 * kQuantBlock;
    static_assert(kRows % kQuantBlock == 0 && kRows * kCells > kTwoTo32 && kRows * kFloatWidth > kTwoTo32);

    struct Block {
        size_t begin = 0, end = 0; // rows of the big store
        size_t compact = 0;        // first row in the compact store
        const char* name = "";
        bool holds_values = true; // false: zero in the sources, and must keep its contents
        [[nodiscard]] constexpr size_t rows() const { return end - begin; }
        [[nodiscard]] constexpr bool contains(const size_t row) const { return row >= begin && row < end; }
    };

    inline constexpr std::array<Block, 4> kBlocks{{
        {0, 256, 0, "H (rows 0-255)", true},
        {256, 1024, 256, "Z (rows 256-1023)", false},
        {89'478'400, 89'478'912, 1024, "F (rows 89,478,400-89,478,911)", true},
        {kQ16WrapRow - kQuantBlock, kRows, 1536, "Q (rows 95,443,456-95,444,223)", true},
    }};
    inline constexpr size_t kCompactRows = 2304;
    static_assert(kBlocks[2].contains(kFloat4WrapRow) && kBlocks[3].contains(kQ16WrapRow) &&
                  kBlocks[3].contains(kCanonicalWrapRow));

    inline size_t compact_row(const size_t row) {
        for (const auto& block : kBlocks)
            if (block.contains(row))
                return block.compact + (row - block.begin);
        ADD_FAILURE() << "row " << row << " is in no block";
        return 0;
    }

    // Integers in [1, 2047]: exact in fp32 and in f16. A full Q16 block of them spans exactly [1, 2047].
    inline float value(const size_t row, const uint32_t cell) {
        return static_cast<float>(1 + (row * kCells + cell) % 2047);
    }
    // What the writing cases put into rows that already hold value().
    inline float second_value(const size_t row, const uint32_t cell) {
        return static_cast<float>(1 + (row * kCells + cell + 1009) % 2047);
    }

    // ---- Stores ------------------------------------------------------------------------------------------------

    inline const char* format_name(const ShFormat format) {
        switch (format) {
        case ShFormat::Canonical: return "Canonical";
        case ShFormat::Float32: return "F32";
        case ShFormat::Float16: return "F16";
        case ShFormat::Q16: return "Q16";
        }
        return "?";
    }

    inline DataType dtype_of(const ShFormat format) {
        return format == ShFormat::Float16 || format == ShFormat::Q16 ? DataType::Float16 : DataType::Float32;
    }

    // Elements of one row inside a 32-row aligned range.
    inline size_t row_elements(const ShFormat format) {
        return format == ShFormat::Float32 || format == ShFormat::Float16 ? kFloatWidth : kCells;
    }

    inline size_t value_elements(const ShFormat format, const size_t rows) {
        if (format == ShFormat::Canonical)
            return rows * kCells;
        if (format == ShFormat::Q16)
            return core::sh_value_quant::sh_value_u16_count(rows, kRest);
        return core::sh_swizzled_float_count(rows, kRest);
    }

    inline size_t bound_elements(const ShFormat format, const size_t rows) {
        return format == ShFormat::Q16 ? core::sh_value_quant::n_bounds_for_prims(rows) * 2 : 0;
    }

    inline size_t store_bytes(const ShFormat format, const size_t rows) {
        return value_elements(format, rows) * (dtype_of(format) == DataType::Float16 ? 2 : 4) +
               bound_elements(format, rows) * 4;
    }

    struct Store {
        ShFormat format = ShFormat::Float32;
        size_t rows = 0;
        Tensor values;
        Tensor bounds; // Q16 only
        [[nodiscard]] const Tensor* bounds_or_null() const { return format == ShFormat::Q16 ? &bounds : nullptr; }
        [[nodiscard]] Tensor* bounds_or_null() { return format == ShFormat::Q16 ? &bounds : nullptr; }
    };

    // A zeroed store. On the GPU this waits for the clear by reading back the last element (a device may report a
    // failed clear only when its commands complete) and returns nullopt when the store can't be allocated or
    // cleared, so that the caller skips.
    inline std::optional<Store> zero_store(const ShFormat format, const size_t rows, const Device device) {
        try {
            Store store{.format = format, .rows = rows};
            const size_t elements = value_elements(format, rows);
            store.values = Tensor::zeros({elements}, device, dtype_of(format));
            if (format == ShFormat::Q16)
                store.bounds = Tensor::zeros({bound_elements(format, rows)}, device, DataType::Float32);
            if (device == Device::GPU)
                (void)store.values.slice(0, elements - 1, elements).cpu();
            return store;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    inline Store cpu_store(const ShFormat format, const size_t rows) {
        auto store = zero_store(format, rows, Device::CPU);
        if (!store)
            throw std::runtime_error("can't allocate a host store");
        return std::move(*store);
    }

    // Runs the codec. `indices` may be null.
    inline void convert(const Store& source, Store& destination, core::ShCodec codec, const Tensor* indices = nullptr) {
        codec.source_format = source.format;
        codec.destination_format = destination.format;
        codec.source_rest = kRest;
        codec.destination_rest = kRest;
        if (codec.source_rows == 0)
            codec.source_rows = source.rows;
        if (codec.destination_rows == 0)
            codec.destination_rows = destination.rows;
        core::sh_codec(source.values, destination.values, codec, indices, source.bounds_or_null(),
                       destination.bounds_or_null());
    }

    // A CPU canonical store of `rows` rows whose row r holds `fill(first_row(r), cell)`, or zeros where
    // `first_row` returns nullopt.
    template <class RowOf, class Fill>
    Store canonical_rows(const size_t rows, RowOf&& row_of, Fill&& fill) {
        Store store = cpu_store(ShFormat::Canonical, rows);
        auto* const data = store.values.ptr<float>();
        for (size_t r = 0; r < rows; ++r) {
            const std::optional<size_t> row = row_of(r);
            if (!row)
                continue;
            for (uint32_t c = 0; c < kCells; ++c)
                data[r * kCells + c] = fill(*row, c);
        }
        return store;
    }

    // The compact store of the blocks in `format` (CPU): fill() (value() by default) in H, F and Q, zero in Z and
    // in rows from `limit` on (a model of `limit` rows).
    inline Store compact_blocks(const ShFormat format, const size_t limit = kRows,
                                float (*const fill)(size_t, uint32_t) = value) {
        Store canonical = canonical_rows(
            kCompactRows,
            [limit](const size_t compact) -> std::optional<size_t> {
                for (const auto& block : kBlocks)
                    if (compact >= block.compact && compact < block.compact + block.rows()) {
                        const size_t row = block.begin + (compact - block.compact);
                        return block.holds_values && row < limit ? std::optional(row) : std::nullopt;
                    }
                return std::nullopt;
            },
            fill);
        if (format == ShFormat::Canonical)
            return canonical;
        Store converted = cpu_store(format, kCompactRows);
        convert(canonical, converted, {.count = kCompactRows});
        return converted;
    }

    // A canonical row of 45 values.
    using CanonicalRow = std::array<float, kCells>;

    inline CanonicalRow row_of_values(const size_t row, float (*const fill)(size_t, uint32_t)) {
        CanonicalRow values{};
        for (uint32_t c = 0; c < kCells; ++c)
            values[c] = fill(row, c);
        return values;
    }

    // The values and the bounds of rows [first_row, first_row + rows) of a store, first_row a multiple of 32 (of
    // 256 for the bounds).
    inline Tensor values_of_rows(const Tensor& values, const ShFormat format, const size_t first_row,
                                 const size_t rows) {
        const size_t width = row_elements(format);
        return values.slice(0, first_row * width, (first_row + rows) * width);
    }

    inline Tensor bounds_of_rows(const Tensor& bounds, const size_t first_row, const size_t rows) {
        return bounds.slice(0, first_row / kQuantBlock * 2, (first_row + rows) / kQuantBlock * 2);
    }

    // Copies the compact store's blocks into the big store at their rows.
    inline void place_blocks(Store& big, const Store& compact) {
        const Tensor values = compact.values.gpu();
        const Tensor bounds = compact.format == ShFormat::Q16 ? compact.bounds.gpu() : Tensor{};
        for (const auto& block : kBlocks) {
            values_of_rows(big.values, big.format, block.begin, block.rows())
                .copy_(values_of_rows(values, compact.format, block.compact, block.rows()));
            if (big.format == ShFormat::Q16)
                bounds_of_rows(big.bounds, block.begin, block.rows())
                    .copy_(bounds_of_rows(bounds, block.compact, block.rows()));
        }
    }

    // What a Q16 store holds after rows were overwritten through a decode, overwrite and re-encode of the 256-row
    // blocks they touch (scatter, zero, append, a mutation batch): `writes` maps compact rows to their new values,
    // applied in order. Untouched blocks keep their codes and bounds.
    inline Store reencoded(const Store& compact_q16, const std::vector<std::pair<size_t, CanonicalRow>>& writes) {
        Store canonical = cpu_store(ShFormat::Canonical, compact_q16.rows);
        convert(compact_q16, canonical, {.count = compact_q16.rows});
        std::vector<bool> touched(core::sh_value_quant::n_bounds_for_prims(compact_q16.rows), false);
        auto* const data = canonical.values.ptr<float>();
        for (const auto& [row, values] : writes) {
            std::memcpy(data + row * kCells, values.data(), sizeof(float) * kCells);
            touched[row / kQuantBlock] = true;
        }
        Store result{.format = ShFormat::Q16, .rows = compact_q16.rows, .values = compact_q16.values.clone(),
                     .bounds = compact_q16.bounds.clone()};
        for (size_t block = 0; block < touched.size(); ++block) {
            if (!touched[block])
                continue;
            const size_t first = block * kQuantBlock;
            Store rows{.format = ShFormat::Canonical, .rows = kQuantBlock,
                       .values = values_of_rows(canonical.values, ShFormat::Canonical, first, kQuantBlock).contiguous()};
            Store encoded = cpu_store(ShFormat::Q16, kQuantBlock);
            convert(rows, encoded, {.count = kQuantBlock});
            values_of_rows(result.values, ShFormat::Q16, first, kQuantBlock).copy_(encoded.values);
            bounds_of_rows(result.bounds, first, kQuantBlock).copy_(encoded.bounds);
        }
        return result;
    }

    template <class T>
    std::vector<T> host_copy(const Tensor& tensor) {
        const Tensor cpu = tensor.cpu().contiguous();
        std::vector<T> values(cpu.bytes() / sizeof(T));
        if (!values.empty())
            std::memcpy(values.data(), cpu.data_ptr(), values.size() * sizeof(T));
        return values;
    }

    // Row and cell of element `index` of a store in `format` whose element 0 is the start of row `first_row`
    // (a multiple of 32).
    inline std::string describe_element(const ShFormat format, const size_t index, const size_t first_row) {
        size_t row = 0, cell = 0;
        if (format == ShFormat::Canonical) {
            row = index / kCells;
            cell = index % kCells;
        } else if (format == ShFormat::Q16) {
            row = index / (kCells * kReorder) * kReorder + index % kReorder;
            cell = index % (kCells * kReorder) / kReorder;
        } else {
            row = index / (kFloatWidth * kReorder) * kReorder + index / 4 % kReorder;
            cell = index % (kFloatWidth * kReorder) / (4 * kReorder) * 4 + index % 4;
        }
        return std::format("row {}, cell {}", first_row + row, cell);
    }

    // Values must match bit for bit, except Q16 codes against the CPU codec, which may differ by one: the CPU codec
    // multiplies by the reciprocal of the range where the GPU divides, so a value within an ulp of a half step rounds
    // apart (code_tolerance 1; 0 where one kernel is compared with itself). Q16 bounds are a block's minimum and
    // maximum and must match exactly.
    inline void expect_values_match(const Tensor& actual, const Tensor& expected, const ShFormat format,
                                    const size_t first_row, const std::string& what, const int code_tolerance = 1) {
        ASSERT_EQ(actual.numel(), expected.numel()) << what;
        size_t wrong = 0;
        std::string first;
        const auto note = [&](const size_t i, const std::string& got, const std::string& want) {
            if (wrong++ == 0)
                first = std::format("{}: {} instead of {}", describe_element(format, i, first_row), got, want);
        };
        if (dtype_of(format) == DataType::Float32) {
            const auto a = host_copy<uint32_t>(actual), e = host_copy<uint32_t>(expected);
            for (size_t i = 0; i < a.size(); ++i)
                if (a[i] != e[i]) {
                    float af, ef;
                    std::memcpy(&af, &a[i], 4);
                    std::memcpy(&ef, &e[i], 4);
                    note(i, std::format("{}", af), std::format("{}", ef));
                }
        } else {
            const auto a = host_copy<uint16_t>(actual), e = host_copy<uint16_t>(expected);
            const int tolerance = format == ShFormat::Q16 ? code_tolerance : 0;
            for (size_t i = 0; i < a.size(); ++i)
                if (std::abs(int(a[i]) - int(e[i])) > tolerance)
                    note(i, std::format("0x{:04x}", a[i]), std::format("0x{:04x}", e[i]));
        }
        EXPECT_EQ(wrong, 0u) << what << ": " << wrong << " of " << actual.numel() << " values wrong; first at "
                             << first;
    }

    inline void expect_bounds_match(const Tensor& actual, const Tensor& expected, const size_t first_row,
                                    const std::string& what) {
        ASSERT_EQ(actual.numel(), expected.numel()) << what;
        const auto a = host_copy<float>(actual), e = host_copy<float>(expected);
        size_t wrong = 0;
        std::string first;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::memcmp(&a[i], &e[i], sizeof(float)) != 0 && wrong++ == 0)
                first = std::format("block of rows {}-{}: {} {} instead of {}", first_row + i / 2 * kQuantBlock,
                                    first_row + i / 2 * kQuantBlock + kQuantBlock - 1, i % 2 ? "hi" : "lo", a[i], e[i]);
        EXPECT_EQ(wrong, 0u) << what << ": " << wrong << " of " << a.size() << " bounds wrong; first: " << first;
    }

    // Rows [actual_row, +rows) of one store against rows [expected_row, +rows) of another: starts and counts on
    // multiples of 32 rows, and of 256 for Q16 (whose bounds are compared too).
    inline void expect_rows_match(const Store& actual, const size_t actual_row, const Store& expected,
                                  const size_t expected_row, const size_t rows, const std::string& what,
                                  const int code_tolerance = 1) {
        expect_values_match(values_of_rows(actual.values, actual.format, actual_row, rows),
                            values_of_rows(expected.values, expected.format, expected_row, rows), actual.format,
                            actual_row, what, code_tolerance);
        if (actual.format == ShFormat::Q16)
            expect_bounds_match(bounds_of_rows(actual.bounds, actual_row, rows),
                                bounds_of_rows(expected.bounds, expected_row, rows), actual_row, what);
    }

    // Every block of the big store against the same block of the compact reference.
    inline void expect_blocks_match(const Store& big, const Store& compact, const std::string& what,
                                    const int code_tolerance = 1) {
        for (const auto& block : kBlocks) {
            const std::string where = what + ", block " + block.name;
            expect_values_match(values_of_rows(big.values, big.format, block.begin, block.rows()),
                                values_of_rows(compact.values, compact.format, block.compact, block.rows()),
                                big.format, block.begin, where, code_tolerance);
            if (big.format == ShFormat::Q16)
                expect_bounds_match(bounds_of_rows(big.bounds, block.begin, block.rows()),
                                    bounds_of_rows(compact.bounds, block.compact, block.rows()), block.begin, where);
        }
    }

    // A whole small store against its reference.
    inline void expect_store_matches(const Store& actual, const Store& expected, const std::string& what) {
        expect_values_match(actual.values, expected.values, actual.format, 0, what);
        if (actual.format == ShFormat::Q16)
            expect_bounds_match(actual.bounds, expected.bounds, 0, what);
    }

    template <class T>
    Tensor index_tensor(const std::vector<size_t>& rows, const Device device) {
        Tensor indices = Tensor::empty({rows.size()}, Device::CPU,
                                       sizeof(T) == 8 ? DataType::Int64 : DataType::Int32);
        auto* const data = static_cast<T*>(indices.data_ptr());
        for (size_t i = 0; i < rows.size(); ++i)
            data[i] = static_cast<T>(rows[i]);
        return device == Device::GPU ? indices.gpu() : indices;
    }

} // namespace lfs::test::large_model

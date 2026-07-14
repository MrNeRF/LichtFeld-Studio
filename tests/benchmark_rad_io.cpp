/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/environment.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "io/formats/rad.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>

namespace {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    uint64_t fnv1a(const void* data, size_t bytes, uint64_t h) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < bytes; ++i) {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    uint64_t hash_tensor(const Tensor& t, uint64_t h) {
        if (!t.is_valid() || t.numel() == 0) {
            return h;
        }
        const auto cpu = t.contiguous().to(Device::CPU);
        return fnv1a(cpu.data_ptr(), cpu.bytes(), h);
    }

    uint64_t hash_splats(const SplatData& d) {
        uint64_t h = 1469598103934665603ull;
        h = hash_tensor(d.means_raw(), h);
        h = hash_tensor(d.sh0_raw(), h);
        h = hash_tensor(d.shN_raw(), h);
        h = hash_tensor(d.scaling_raw(), h);
        h = hash_tensor(d.rotation_raw(), h);
        h = hash_tensor(d.opacity_raw(), h);
        if (d.lod_tree) {
            h = fnv1a(d.lod_tree->child_count.data(), d.lod_tree->child_count.size() * sizeof(uint16_t), h);
            h = fnv1a(d.lod_tree->child_start.data(), d.lod_tree->child_start.size() * sizeof(uint32_t), h);
        }
        return h;
    }

    double ms_since(std::chrono::high_resolution_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - start)
            .count();
    }

} // namespace

TEST(RadIoBenchmark, LoadSaveRoundtrip) {
    const auto env_path = lfs::core::environment::value("LFS_RAD_BENCH_FILE");
    const std::filesystem::path input =
        env_path ? std::filesystem::path(*env_path)
                 : std::filesystem::path(PROJECT_ROOT_PATH) / "splat_30000.rad";
    ASSERT_TRUE(std::filesystem::exists(input)) << "Benchmark input missing: " << input;

    const auto input_size = std::filesystem::file_size(input);
    printf("input: %s (%.1f MB)\n", input.string().c_str(),
           static_cast<double>(input_size) / (1024.0 * 1024.0));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto first = lfs::io::load_rad(input);
    const double load_cold_ms = ms_since(t0);
    ASSERT_TRUE(first.has_value()) << first.error();

    t0 = std::chrono::high_resolution_clock::now();
    auto warm = lfs::io::load_rad(input);
    const double load_warm_ms = ms_since(t0);
    ASSERT_TRUE(warm.has_value()) << warm.error();

    const uint64_t load_hash = hash_splats(*first);
    ASSERT_EQ(load_hash, hash_splats(*warm));

    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "rad_io_benchmark.rad";

    lfs::io::RadSaveOptions options;
    options.output_path = output;

    t0 = std::chrono::high_resolution_clock::now();
    auto save1 = lfs::io::save_rad(*first, options);
    const double save_cold_ms = ms_since(t0);
    ASSERT_TRUE(save1.has_value());

    t0 = std::chrono::high_resolution_clock::now();
    auto save2 = lfs::io::save_rad(*first, options);
    const double save_warm_ms = ms_since(t0);
    ASSERT_TRUE(save2.has_value());

    const auto output_size = std::filesystem::file_size(output);

    t0 = std::chrono::high_resolution_clock::now();
    auto reloaded = lfs::io::load_rad(output);
    const double reload_ms = ms_since(t0);
    ASSERT_TRUE(reloaded.has_value()) << reloaded.error();
    const uint64_t reload_hash = hash_splats(*reloaded);

    printf("splats: %lld, sh_degree: %d\n",
           static_cast<long long>(first->size()), first->get_max_sh_degree());
    printf("load:   cold %.0f ms, warm %.0f ms\n", load_cold_ms, load_warm_ms);
    printf("save:   cold %.0f ms, warm %.0f ms\n", save_cold_ms, save_warm_ms);
    printf("reload: %.0f ms\n", reload_ms);
    printf("output: %.1f MB (input %.1f MB)\n",
           static_cast<double>(output_size) / (1024.0 * 1024.0),
           static_cast<double>(input_size) / (1024.0 * 1024.0));
    printf("load_hash:   %016llx\n", static_cast<unsigned long long>(load_hash));
    printf("reload_hash: %016llx\n", static_cast<unsigned long long>(reload_hash));

    std::filesystem::remove(output);
}

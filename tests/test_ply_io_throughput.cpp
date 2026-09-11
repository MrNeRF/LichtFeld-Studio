/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::io;

namespace {

    SplatData make_synthetic_splat(const size_t n, const int sh_degree = 3) {
        constexpr int SH_REST[] = {0, 3, 8, 15};
        const size_t sh_rest = sh_degree > 0 ? static_cast<size_t>(SH_REST[sh_degree]) : 0;

        auto means = Tensor::empty({n, 3}, Device::CPU, DataType::Float32);
        auto sh0 = Tensor::empty({n, 1, 3}, Device::CPU, DataType::Float32);
        auto scaling = Tensor::empty({n, 3}, Device::CPU, DataType::Float32);
        auto rotation = Tensor::empty({n, 4}, Device::CPU, DataType::Float32);
        auto opacity = Tensor::empty({n, 1}, Device::CPU, DataType::Float32);
        Tensor shN;
        if (sh_rest > 0) {
            shN = Tensor::empty({n, sh_rest, 3}, Device::CPU, DataType::Float32);
        }

        auto* m = means.ptr<float>();
        auto* s0 = sh0.ptr<float>();
        auto* sc = scaling.ptr<float>();
        auto* rot = rotation.ptr<float>();
        auto* op = opacity.ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            m[i * 3 + 0] = static_cast<float>(i % 1024) * 0.01f;
            m[i * 3 + 1] = static_cast<float>((i / 1024) % 1024) * 0.01f;
            m[i * 3 + 2] = static_cast<float>(i / (1024 * 1024)) * 0.01f;
            s0[i * 3 + 0] = 0.5f;
            s0[i * 3 + 1] = 0.4f;
            s0[i * 3 + 2] = 0.3f;
            sc[i * 3 + 0] = -4.0f;
            sc[i * 3 + 1] = -4.0f;
            sc[i * 3 + 2] = -4.0f;
            rot[i * 4 + 0] = 1.0f;
            rot[i * 4 + 1] = 0.0f;
            rot[i * 4 + 2] = 0.0f;
            rot[i * 4 + 3] = 0.0f;
            op[i] = -1.0f + 0.001f * static_cast<float>(i % 1000);
        }
        if (sh_rest > 0) {
            auto* sn = shN.ptr<float>();
            for (size_t i = 0; i < n * sh_rest * 3; ++i) {
                sn[i] = 0.01f * static_cast<float>(static_cast<int>(i % 7) - 3);
            }
        }

        return SplatData(sh_degree, std::move(means), std::move(sh0), std::move(shN),
                         std::move(scaling), std::move(rotation), std::move(opacity), 1.0f);
    }

    double max_abs_diff(const Tensor& a, const Tensor& b, const size_t max_elems) {
        auto ac = a.cpu().contiguous();
        auto bc = b.cpu().contiguous();
        const auto* ap = ac.ptr<float>();
        const auto* bp = bc.ptr<float>();
        const size_t n = std::min(static_cast<size_t>(ac.numel()), max_elems);
        double max_abs = 0.0;
        for (size_t i = 0; i < n; ++i) {
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(ap[i] - bp[i])));
        }
        return max_abs;
    }

} // namespace

TEST(PlyIoRoundtrip, OneMillionSplatSaveLoadRoundtrip) {
    constexpr size_t kN = 100'000;
    constexpr int kShDegree = 3;

    const fs::path out_dir = fs::temp_directory_path() / "lfs_ply_io_throughput";
    fs::create_directories(out_dir);
    const fs::path ply_path = out_dir / "synth_1m_sh3.ply";
    std::error_code ec;
    fs::remove(ply_path, ec);

    auto splat = make_synthetic_splat(kN, kShDegree);
    ASSERT_EQ(splat.size(), kN);

    PlySaveOptions opts;
    opts.output_path = ply_path;
    opts.binary = true;

    auto save_result = save_ply(splat, opts);
    ASSERT_TRUE(save_result.has_value()) << save_result.error().format();
    ASSERT_TRUE(fs::exists(ply_path));

    const auto file_bytes = fs::file_size(ply_path);
    auto loader = Loader::create();
    auto load_result = loader->load(ply_path);
    ASSERT_TRUE(load_result.has_value()) << load_result.error().format();
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<SplatData>>(load_result->data));
    auto& loaded = *std::get<std::shared_ptr<SplatData>>(load_result->data);
    EXPECT_EQ(loaded.size(), kN);

    // Spot-check means + opacity (format-preserving write). First 4k + last 64 rows.
    const double means_err_head =
        max_abs_diff(splat.get_means(), loaded.get_means(), 4096 * 3);
    EXPECT_LT(means_err_head, 1e-5) << "means drift after PLY roundtrip (head)";

    const double opacity_err =
        max_abs_diff(splat.get_opacity(), loaded.get_opacity(), 4096);
    EXPECT_LT(opacity_err, 1e-5) << "opacity drift after PLY roundtrip";

    // Tail means: compare last 64 vertices via host views.
    {
        auto om = splat.get_means().cpu().contiguous();
        auto lm = loaded.get_means().cpu().contiguous();
        const auto* op = om.ptr<float>();
        const auto* lp = lm.ptr<float>();
        double tail_err = 0.0;
        for (size_t i = (kN - 64) * 3; i < kN * 3; ++i) {
            tail_err = std::max(tail_err, static_cast<double>(std::fabs(op[i] - lp[i])));
        }
        EXPECT_LT(tail_err, 1e-5) << "means drift after PLY roundtrip (tail)";
    }

    EXPECT_GT(file_bytes, size_t{1});

    fs::remove(ply_path, ec);
    fs::remove_all(out_dir, ec);
}

// cancel during write must not UAF the stdio buffer (declaration order).
TEST(PlyExportCancel, CancelMidWriteDoesNotCrash) {
    constexpr size_t kN = 50'000; // large enough to enter buffered write path
    constexpr int kShDegree = 1;

    const fs::path out_dir = fs::temp_directory_path() / "lfs_ply_cancel";
    fs::create_directories(out_dir);
    const fs::path ply_path = out_dir / "cancel.ply";
    std::error_code ec;
    fs::remove(ply_path, ec);

    auto splat = make_synthetic_splat(kN, kShDegree);
    PlySaveOptions opts;
    opts.output_path = ply_path;
    opts.binary = true;
    int callbacks = 0;
    opts.progress_callback = [&](float /*ratio*/, const std::string& /*stage*/) {
        ++callbacks;
        return false; // cancel immediately
    };

    auto result = save_ply(splat, opts);
    ASSERT_FALSE(result.has_value()) << "cancel must yield an error result";
    EXPECT_EQ(result.error().code, ErrorCode::CANCELLED)
        << result.error().format();
    EXPECT_GE(callbacks, 1);

    // Process still healthy after cancel (no heap corruption from buffer UAF).
    {
        PlySaveOptions ok_opts;
        ok_opts.output_path = out_dir / "ok.ply";
        ok_opts.binary = true;
        auto ok = save_ply(make_synthetic_splat(1024, 0), ok_opts);
        EXPECT_TRUE(ok.has_value()) << ok.error().format();
        fs::remove(ok_opts.output_path, ec);
    }

    fs::remove(ply_path, ec);
    fs::remove_all(out_dir, ec);
}

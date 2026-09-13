/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Completed-work selection benchmark with full-scene CPU references.
// Build: cmake --build build --target bench_selection_tensor
// Run: build/tests/bench_selection_tensor scene.ply --backend cuda|vulkan|all
// Uses three warmups and eleven measured samples. Not registered with ctest.

#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/selection_ops.hpp"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::GpuBackend;
using lfs::core::GpuBackendScope;
using lfs::core::Tensor;
using lfs::core::TensorShape;
using lfs::rendering::ScreenWindowCameraModel;

namespace {

    constexpr int kWarmup = 3;
    constexpr int kSamples = 11;
    constexpr int kWidth = 1920;
    constexpr int kHeight = 1080;
    constexpr float kFx = 1200.0f;
    constexpr float kFy = 1200.0f;
    constexpr float kCx = 960.0f;
    constexpr float kCy = 540.0f;
    constexpr float kBrushRadius = 80.0f;
    constexpr float kInvalidThreshold = lfs::rendering::kInvalidScreenPositionThreshold;
    constexpr float kInvalidFill = kInvalidThreshold * 100000.0f;
    // Sub-pixel floor (matches SelectionTensorProjection pinhole), relative
    // scale for large off-screen coords, hard cap of one pixel.
    constexpr float kPixelAbsTol = 1.0e-3f;
    constexpr std::array<float, 9> kIdentity{
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    constexpr std::array<float, 3> kOrigin{0.0f, 0.0f, 0.0f};
    const std::vector<float> kQuad{
        200.0f,
        100.0f,
        1700.0f,
        100.0f,
        1700.0f,
        980.0f,
        200.0f,
        980.0f,
    };

    [[noreturn]] void fail(const std::string& message) {
        std::fprintf(stderr, "bench_selection_tensor: %s\n", message.c_str());
        std::fflush(stderr);
        std::exit(1);
    }

    void wait_complete(const GpuBackend backend) {
        lfs::core::internal::backend_ops(backend).synchronize_device();
    }

    void materialize(const Tensor& tensor) {
        if (tensor.is_valid()) {
            (void)tensor.data_ptr();
        }
    }

    [[nodiscard]] const char* backend_name(const GpuBackend backend) {
        return backend == GpuBackend::CUDA ? "cuda" : "vulkan";
    }

    [[nodiscard]] double median_ms(std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());
        const size_t n = samples.size();
        if (n == 0) {
            return 0.0;
        }
        if (n % 2 == 1) {
            return samples[n / 2];
        }
        return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    }

    [[nodiscard]] std::vector<float> load_ply_xyz(const char* const path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            fail(std::string("failed to open ") + path);
        }
        tinyply::PlyFile ply;
        ply.parse_header(file);
        std::shared_ptr<tinyply::PlyData> verts;
        try {
            verts = ply.request_properties_from_element("vertex", {"x", "y", "z"});
        } catch (const std::exception& e) {
            fail(std::string("PLY has no vertex x,y,z: ") + e.what());
        }
        ply.read(file);
        if (!verts || verts->count == 0) {
            fail("PLY has no vertices");
        }
        if (verts->t != tinyply::Type::FLOAT32) {
            fail("vertex x,y,z must be float32");
        }
        std::vector<float> xyz(static_cast<size_t>(verts->count) * 3);
        std::memcpy(xyz.data(), verts->buffer.get(), xyz.size() * sizeof(float));
        return xyz;
    }

    [[nodiscard]] bool is_invalid(const float x, const float y) {
        return !(x >= kInvalidThreshold) || !(y >= kInvalidThreshold);
    }

    [[nodiscard]] bool coords_near(const float got, const float expected) {
        const float abs_err = std::abs(got - expected);
        const float mag = std::max(std::abs(got), std::abs(expected));
        // Off-screen projections can exceed millions of pixels. Compare in
        // float32 ULPs there; subpixel coordinates keep an absolute tolerance.
        const float ulp = std::nextafter(mag, std::numeric_limits<float>::infinity()) - mag;
        const float tol = std::max(kPixelAbsTol, 8.0f * ulp);
        return abs_err <= tol;
    }

    void cpu_project_pinhole(const std::vector<float>& means, std::vector<float>& out_xy) {
        const size_t n = means.size() / 3;
        out_xy.assign(n * 2, kInvalidFill);
        for (size_t i = 0; i < n; ++i) {
            const float view_x = means[i * 3];
            const float view_y = means[i * 3 + 1];
            const float view_z = means[i * 3 + 2];
            if (!std::isfinite(view_x) || !std::isfinite(view_y) || !std::isfinite(view_z)) {
                continue;
            }
            if (view_z >= -1.0e-6f) {
                continue;
            }
            const float depth = -view_z;
            out_xy[i * 2] = kCx + view_x * kFx / depth;
            out_xy[i * 2 + 1] = kCy - view_y * kFy / depth;
        }
    }

    [[nodiscard]] bool cpu_disk(const float x, const float y, const float mx, const float my,
                                const float radius) {
        if (is_invalid(x, y) || !std::isfinite(x) || !std::isfinite(y)) {
            return false;
        }
        const float dx = x - mx;
        const float dy = y - my;
        return dx * dx + dy * dy <= radius * radius;
    }

    // Even-odd, same as polygonSelectKernel / polygon_select_tensor_program.
    // Horizontal edges (yi == yj) never cross; no N×vertex matrix.
    [[nodiscard]] bool cpu_even_odd(const float px, const float py, const std::vector<float>& poly) {
        if (is_invalid(px, py) || !std::isfinite(px) || !std::isfinite(py)) {
            return false;
        }
        const int n = static_cast<int>(poly.size() / 2);
        bool inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            const float yi = poly[static_cast<size_t>(i) * 2 + 1];
            const float yj = poly[static_cast<size_t>(j) * 2 + 1];
            if (yi == yj) {
                continue;
            }
            if ((yi > py) != (yj > py)) {
                const float xi = poly[static_cast<size_t>(i) * 2];
                const float xj = poly[static_cast<size_t>(j) * 2];
                if (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                    inside = !inside;
                }
            }
        }
        return inside;
    }

    [[nodiscard]] size_t count_valid(const std::vector<float>& xy) {
        size_t n = 0;
        for (size_t i = 0; i + 1 < xy.size(); i += 2) {
            if (!is_invalid(xy[i], xy[i + 1]) && std::isfinite(xy[i]) && std::isfinite(xy[i + 1])) {
                ++n;
            }
        }
        return n;
    }

    [[nodiscard]] size_t cpu_disk_count(const std::vector<float>& xy, const float mx, const float my,
                                        const float radius) {
        size_t n = 0;
        for (size_t i = 0; i + 1 < xy.size(); i += 2) {
            n += cpu_disk(xy[i], xy[i + 1], mx, my, radius) ? 1 : 0;
        }
        return n;
    }

    [[nodiscard]] size_t cpu_even_odd_count(const std::vector<float>& xy, const std::vector<float>& poly) {
        size_t n = 0;
        for (size_t i = 0; i + 1 < xy.size(); i += 2) {
            n += cpu_even_odd(xy[i], xy[i + 1], poly) ? 1 : 0;
        }
        return n;
    }

    void assert_projected(const std::vector<float>& got, const std::vector<float>& expected,
                          const char* const tag) {
        if (got.size() != expected.size()) {
            fail(std::string(tag) + " projected size mismatch: got " + std::to_string(got.size()) +
                 " expected " + std::to_string(expected.size()));
        }
        const size_t n = got.size() / 2;
        size_t nan_count = 0;
        size_t mask_mismatch = 0;
        size_t coord_mismatch = 0;
        size_t first = n;
        float max_abs_err = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float gx = got[i * 2];
            const float gy = got[i * 2 + 1];
            const float ex = expected[i * 2];
            const float ey = expected[i * 2 + 1];
            if (!std::isfinite(gx) || !std::isfinite(gy)) {
                ++nan_count;
                if (first == n) {
                    first = i;
                }
                continue;
            }
            const bool g_inv = is_invalid(gx, gy);
            const bool e_inv = is_invalid(ex, ey) || !std::isfinite(ex) || !std::isfinite(ey);
            if (g_inv != e_inv) {
                ++mask_mismatch;
                if (first == n) {
                    first = i;
                }
                continue;
            }
            if (g_inv) {
                continue;
            }
            const float dx = std::abs(gx - ex);
            const float dy = std::abs(gy - ey);
            max_abs_err = std::max(max_abs_err, std::max(dx, dy));
            if (!coords_near(gx, ex) || !coords_near(gy, ey)) {
                ++coord_mismatch;
                if (first == n) {
                    first = i;
                }
            }
        }
        std::printf("check=%s n=%zu valid=%zu nan=%zu invalid_mask_mismatch=%zu "
                    "coord_mismatch=%zu max_abs_err=%.6g first=%zu\n",
                    tag, n, count_valid(got), nan_count, mask_mismatch, coord_mismatch, max_abs_err,
                    first);
        std::fflush(stdout);
        if (nan_count != 0 || mask_mismatch != 0 || coord_mismatch != 0) {
            const size_t i = first < n ? first : 0;
            fail(std::string(tag) + " projection mismatch at " + std::to_string(i) + " got=(" +
                 std::to_string(got[i * 2]) + "," + std::to_string(got[i * 2 + 1]) + ") expected=(" +
                 std::to_string(expected[i * 2]) + "," + std::to_string(expected[i * 2 + 1]) +
                 ") nan=" + std::to_string(nan_count) +
                 " mask=" + std::to_string(mask_mismatch) +
                 " coord=" + std::to_string(coord_mismatch));
        }
    }

    template <typename Pred>
    void assert_selection_counts(const std::vector<bool>& gpu_mask, const std::vector<float>& gpu_xy,
                                 const std::vector<float>& cpu_xy, const size_t cpu_on_gpu_xy,
                                 const size_t cpu_on_cpu_xy, const char* const tag, Pred&& predicate) {
        if (gpu_mask.size() * 2 != gpu_xy.size() || gpu_xy.size() != cpu_xy.size()) {
            fail(std::string(tag) + " selection size mismatch");
        }
        size_t gpu_count = 0;
        size_t first = gpu_mask.size();
        for (size_t i = 0; i < gpu_mask.size(); ++i) {
            const bool expected = predicate(gpu_xy[i * 2], gpu_xy[i * 2 + 1]);
            const bool got = gpu_mask[i];
            gpu_count += got ? 1 : 0;
            if (got != expected && first == gpu_mask.size()) {
                first = i;
            }
        }
        std::printf("check=%s gpu_count=%zu cpu_on_gpu_xy=%zu cpu_on_cpu_xy=%zu first_mask=%zu\n",
                    tag, gpu_count, cpu_on_gpu_xy, cpu_on_cpu_xy, first);
        std::fflush(stdout);
        if (first != gpu_mask.size()) {
            fail(std::string(tag) + " GPU mask != CPU predicate on GPU xy at " +
                 std::to_string(first));
        }
        if (gpu_count != cpu_on_gpu_xy) {
            fail(std::string(tag) + " GPU count " + std::to_string(gpu_count) +
                 " != CPU on GPU xy " + std::to_string(cpu_on_gpu_xy));
        }
        if (gpu_count != cpu_on_cpu_xy) {
            fail(std::string(tag) + " GPU count " + std::to_string(gpu_count) +
                 " != CPU on CPU xy " + std::to_string(cpu_on_cpu_xy) +
                 " (end-to-end; not a GPU==CPU-fallback timing comparison)");
        }
    }

    template <typename Fn>
    [[nodiscard]] double time_ms(const GpuBackend backend, Fn&& fn) {
        wait_complete(backend);
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        wait_complete(backend);
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    template <typename Fn>
    [[nodiscard]] double time_median_gpu(const GpuBackend backend, Fn&& fn) {
        for (int i = 0; i < kWarmup; ++i) {
            fn();
            wait_complete(backend);
        }
        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(kSamples));
        for (int i = 0; i < kSamples; ++i) {
            samples.push_back(time_ms(backend, fn));
        }
        return median_ms(std::move(samples));
    }

    template <typename Fn>
    [[nodiscard]] double time_median_cpu(Fn&& fn) {
        for (int i = 0; i < kWarmup; ++i) {
            fn();
        }
        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(kSamples));
        for (int i = 0; i < kSamples; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            const auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return median_ms(std::move(samples));
    }

    void print_gpu_metric(const GpuBackend backend, const char* const name, const double median_ms,
                          const char* const extra) {
        std::printf("backend=%s %s_median_ms=%.3f warmup=%d samples=%d%s\n", backend_name(backend),
                    name, median_ms, kWarmup, kSamples, extra);
        std::fflush(stdout);
    }

    Tensor project_dispatch(const Tensor& means) {
        Tensor out = lfs::rendering::project_screen_positions_tensor(
            means, kWidth, kHeight, kIdentity, kOrigin, kFx, kFy, kCx, kCy,
            ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {});
        materialize(out);
        return out;
    }

    Tensor project_program(const Tensor& means) {
        Tensor out = lfs::rendering::project_screen_positions_tensor_program(
            means, kWidth, kHeight, kIdentity, kOrigin, kFx, kFy, kCx, kCy,
            ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {});
        materialize(out);
        return out;
    }

    void bench_backend(const GpuBackend backend, const std::vector<float>& xyz,
                       const std::vector<float>& cpu_xy, const size_t cpu_brush_count,
                       const size_t cpu_polygon_count) {
        if (!lfs::core::gpu_backend_available(backend)) {
            std::printf("backend=%s skipped (unavailable)\n", backend_name(backend));
            std::fflush(stdout);
            return;
        }
        GpuBackendScope scope(backend);
        const size_t n = xyz.size() / 3;

        const auto t_up0 = std::chrono::steady_clock::now();
        const Tensor means = Tensor::from_vector(xyz, TensorShape{n, 3}, Device::CPU).to(Device::GPU);
        wait_complete(backend);
        const auto t_up1 = std::chrono::steady_clock::now();
        const double upload_ms = std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();
        std::printf("backend=%s n=%zu upload_means_ms=%.3f note=host_to_device_not_gpu_kernel\n",
                    backend_name(backend), n, upload_ms);
        std::fflush(stdout);

        Tensor screen = project_dispatch(means);
        wait_complete(backend);
        std::vector<float> gpu_xy = screen.to_vector();
        assert_projected(gpu_xy, cpu_xy, (std::string(backend_name(backend)) + ".project").c_str());

        const double project_gpu_ms = time_median_gpu(backend, [&] { screen = project_dispatch(means); });
        print_gpu_metric(backend, "project_gpu", project_gpu_ms,
                         " includes=device_sync_materialize note=no_download");

        Tensor fallback_screen;
        std::vector<float> host_means;
        std::vector<float> host_screen;
        const double host_fallback_ms = time_median_gpu(backend, [&] {
            host_means = means.to_vector();
            cpu_project_pinhole(host_means, host_screen);
            fallback_screen = Tensor::from_vector(host_screen, TensorShape{n, 2}, Device::CPU)
                                  .to(Device::GPU);
        });
        assert_projected(fallback_screen.to_vector(), cpu_xy,
                         (std::string(backend_name(backend)) + ".host_fallback").c_str());
        print_gpu_metric(backend, "cpu_projection_roundtrip", host_fallback_ms,
                         " includes=means_download_cpu_projection_screen_upload");

        std::vector<float> downloaded;
        const double project_fallback_ms = time_median_gpu(backend, [&] {
            screen = project_dispatch(means);
            downloaded = screen.to_vector();
        });
        print_gpu_metric(backend, "project_with_download", project_fallback_ms,
                         " includes=download note=not_equivalent_to_project_gpu");

        if (backend == GpuBackend::CUDA) {
            Tensor program_screen = project_program(means);
            wait_complete(backend);
            assert_projected(program_screen.to_vector(), cpu_xy,
                             (std::string(backend_name(backend)) + ".project_program").c_str());
            const double program_gpu_ms =
                time_median_gpu(backend, [&] { program_screen = project_program(means); });
            print_gpu_metric(backend, "project_program_gpu", program_gpu_ms,
                             " includes=device_sync_materialize note=no_download");
            const double program_fallback_ms = time_median_gpu(backend, [&] {
                program_screen = project_program(means);
                downloaded = program_screen.to_vector();
            });
            print_gpu_metric(backend, "project_program_with_download", program_fallback_ms,
                             " includes=download note=not_equivalent_to_project_program_gpu");
        }

        screen = project_dispatch(means);
        wait_complete(backend);
        gpu_xy = screen.to_vector();
        assert_projected(gpu_xy, cpu_xy,
                         (std::string(backend_name(backend)) + ".project_for_select").c_str());

        Tensor selection = Tensor::zeros({n}, Device::GPU, DataType::Bool);
        auto run_brush = [&] {
            selection.zero_();
            lfs::rendering::brush_select_tensor(screen, kCx, kCy, kBrushRadius, selection);
            materialize(selection);
        };
        run_brush();
        wait_complete(backend);
        const auto brush_mask = selection.to_vector_bool();
        const size_t cpu_brush_on_gpu_xy = cpu_disk_count(gpu_xy, kCx, kCy, kBrushRadius);
        assert_selection_counts(
            brush_mask, gpu_xy, cpu_xy, cpu_brush_on_gpu_xy, cpu_brush_count,
            (std::string(backend_name(backend)) + ".brush").c_str(),
            [](const float x, const float y) { return cpu_disk(x, y, kCx, kCy, kBrushRadius); });

        const double brush_gpu_ms = time_median_gpu(backend, run_brush);
        print_gpu_metric(backend, "brush_gpu", brush_gpu_ms,
                         " includes=device_sync_materialize note=no_download");
        std::vector<bool> downloaded_mask;
        const double brush_fallback_ms = time_median_gpu(backend, [&] {
            run_brush();
            downloaded_mask = selection.to_vector_bool();
        });
        print_gpu_metric(backend, "brush_with_download", brush_fallback_ms,
                         " includes=download note=not_equivalent_to_brush_gpu");

        const Tensor polygon = Tensor::from_vector(kQuad, TensorShape{4, 2}, Device::CPU).to(Device::GPU);
        wait_complete(backend);
        auto run_polygon = [&] {
            selection.zero_();
            lfs::rendering::polygon_select_tensor(screen, polygon, selection);
            materialize(selection);
        };
        run_polygon();
        wait_complete(backend);
        const auto polygon_mask = selection.to_vector_bool();
        const size_t cpu_poly_on_gpu_xy = cpu_even_odd_count(gpu_xy, kQuad);
        assert_selection_counts(
            polygon_mask, gpu_xy, cpu_xy, cpu_poly_on_gpu_xy, cpu_polygon_count,
            (std::string(backend_name(backend)) + ".polygon").c_str(),
            [](const float x, const float y) { return cpu_even_odd(x, y, kQuad); });

        const double polygon_gpu_ms = time_median_gpu(backend, run_polygon);
        print_gpu_metric(backend, "polygon_gpu", polygon_gpu_ms,
                         " includes=device_sync_materialize note=no_NxV_matrix");
        const double polygon_fallback_ms = time_median_gpu(backend, [&] {
            run_polygon();
            downloaded_mask = selection.to_vector_bool();
        });
        print_gpu_metric(backend, "polygon_with_download", polygon_fallback_ms,
                         " includes=download note=not_equivalent_to_polygon_gpu");
        (void)downloaded;
        (void)downloaded_mask;
    }

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: bench_selection_tensor PLY [--backend cuda|vulkan|all]\n");
        return 2;
    }
    const char* ply_path = argv[1];
    std::string backend_flag = "all";
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--backend" && i + 1 < argc) {
            backend_flag = argv[++i];
        } else {
            fail(std::string("unknown argument: ") + argv[i]);
        }
    }

    const auto xyz = load_ply_xyz(ply_path);
    const size_t n = xyz.size() / 3;
    std::printf("ply=%s n=%zu downsample=0 warmup=%d samples=%d\n", ply_path, n, kWarmup, kSamples);
    std::fflush(stdout);

    std::vector<float> cpu_xy;
    const double cpu_project_ms = time_median_cpu([&] { cpu_project_pinhole(xyz, cpu_xy); });
    std::printf("cpu_project_median_ms=%.3f valid=%zu warmup=%d samples=%d\n", cpu_project_ms,
                count_valid(cpu_xy), kWarmup, kSamples);
    std::fflush(stdout);

    size_t cpu_brush_count = 0;
    const double cpu_brush_ms =
        time_median_cpu([&] { cpu_brush_count = cpu_disk_count(cpu_xy, kCx, kCy, kBrushRadius); });
    std::printf("cpu_brush_median_ms=%.3f count=%zu rule=disk warmup=%d samples=%d\n", cpu_brush_ms,
                cpu_brush_count, kWarmup, kSamples);
    std::fflush(stdout);

    size_t cpu_polygon_count = 0;
    const double cpu_polygon_ms =
        time_median_cpu([&] { cpu_polygon_count = cpu_even_odd_count(cpu_xy, kQuad); });
    std::printf("cpu_polygon_median_ms=%.3f count=%zu rule=evenodd warmup=%d samples=%d "
                "note=no_NxV_matrix\n",
                cpu_polygon_ms, cpu_polygon_count, kWarmup, kSamples);
    std::fflush(stdout);

    const bool want_cuda = backend_flag == "all" || backend_flag == "cuda";
    const bool want_vk = backend_flag == "all" || backend_flag == "vulkan";
    if (!want_cuda && !want_vk) {
        fail("backend must be cuda, vulkan, or all");
    }
    if (want_cuda) {
        bench_backend(GpuBackend::CUDA, xyz, cpu_xy, cpu_brush_count, cpu_polygon_count);
    }
    if (want_vk) {
        bench_backend(GpuBackend::Vulkan, xyz, cpu_xy, cpu_brush_count, cpu_polygon_count);
    }
    return 0;
}

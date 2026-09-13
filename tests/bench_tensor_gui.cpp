/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Standalone GUI-workload tensor bench. Built with tests, NOT registered with
// ctest. Times CUDA or Vulkan through tensor_backend public selection plus
// internal GpuBackendOps::synchronize_device() — never CUDA-only events.
//
// Invocation:
//   bench_tensor_gui --backend cuda|vulkan [--repetitions N] [--case NAME]
//
// Default N=5, ten warmups per case. Validate host-equivalent output outside
// the timed region (fatal on mismatch), then report median wall-time that
// includes device completion. Lazy expressions are materialized inside the
// timed region via data_ptr().

#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor_backend.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::GpuBackend;
using lfs::core::GpuBackendScope;
using lfs::core::Tensor;
using lfs::core::TensorShape;

namespace {

    constexpr int kDefaultRepetitions = 5;
    constexpr int kWarmup = 10;
    constexpr int kSmallDispatchInner = 256;
    constexpr size_t kMaskItems = 1'000'000;
    constexpr size_t kDispatchElems = 64;
    constexpr float kGain = 1.15f;
    constexpr float kBias = -0.02f;
    constexpr float kContrast = 0.98f;
    constexpr float kFloatAbsTol = 1.0e-6f;

    [[noreturn]] void fail(const std::string& message) {
        std::fprintf(stderr, "bench_tensor_gui: %s\n", message.c_str());
        std::fflush(stderr);
        std::exit(1);
    }

    [[noreturn]] void usage_error(const std::string& message) {
        std::fprintf(stderr,
                     "bench_tensor_gui: %s\n"
                     "Usage: bench_tensor_gui --backend cuda|vulkan "
                     "[--repetitions N] [--case NAME]\n",
                     message.c_str());
        std::fflush(stderr);
        std::exit(2);
    }

    void wait_complete(const GpuBackend backend) {
        lfs::core::internal::backend_ops(backend).synchronize_device();
    }

    // Deferred expressions must not escape the timed region unevaluated.
    void materialize(const Tensor& tensor) {
        (void)tensor.data_ptr();
    }

    [[nodiscard]] const char* backend_csv_id(const GpuBackend backend) {
        switch (backend) {
        case GpuBackend::CUDA:
            return "cuda";
        case GpuBackend::Vulkan:
            return "vulkan";
        }
        return "unknown";
    }

    [[nodiscard]] std::optional<GpuBackend> parse_backend(const std::string_view text) {
        if (text == "cuda" || text == "CUDA" || text == "Cuda") {
            return GpuBackend::CUDA;
        }
        if (text == "vulkan" || text == "Vulkan" || text == "VULKAN") {
            return GpuBackend::Vulkan;
        }
        return std::nullopt;
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

    template <typename Fn>
    [[nodiscard]] double time_ms(const GpuBackend backend, Fn&& fn) {
        wait_complete(backend);
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        wait_complete(backend);
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    void print_csv_row(const GpuBackend backend, const char* name, const double median,
                       const int repetitions) {
        std::printf("%s,%s,%.6f,%d\n", backend_csv_id(backend), name, median, repetitions);
        std::fflush(stdout);
    }

    void require_backend(const Tensor& tensor, const GpuBackend expected, const char* what) {
        const auto tagged = lfs::core::gpu_backend_of(tensor);
        if (!tagged.has_value() || *tagged != expected) {
            fail(std::format("{} is not a {} GPU tensor (tag={})",
                             what,
                             backend_csv_id(expected),
                             tagged ? lfs::core::gpu_backend_name(*tagged) : "none"));
        }
    }

    // Deterministic values spanning below 0, (0,1), and above 1 for clamp paths.
    [[nodiscard]] float pattern(const size_t i) {
        return static_cast<float>(static_cast<int>(i % 401) - 50) / 300.0f;
    }

    [[nodiscard]] std::vector<float> patterned(const size_t n) {
        std::vector<float> values(n);
        for (size_t i = 0; i < n; ++i) {
            values[i] = pattern(i);
        }
        return values;
    }

    Tensor upload_gpu(const std::vector<float>& host, const TensorShape& shape,
                      const GpuBackend backend) {
        Tensor gpu = Tensor::from_vector(host, shape, Device::GPU);
        require_backend(gpu, backend, "upload");
        return gpu;
    }

    // Matches export_post_process / image_io: clamp[0,1] * 255 + 0.5, then uint8.
    [[nodiscard]] uint8_t image_path_u8(const float value) {
        return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    void expect_u8_eq(const std::vector<uint8_t>& got, const std::vector<uint8_t>& expected,
                      const char* name) {
        if (got.size() != expected.size()) {
            fail(std::format("{} size mismatch: got {} expected {}", name, got.size(),
                             expected.size()));
        }
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != expected[i]) {
                fail(std::format("{} mismatch at {}: got {} expected {}", name, i,
                                 static_cast<unsigned>(got[i]),
                                 static_cast<unsigned>(expected[i])));
            }
        }
    }

    void expect_f32_eq(const std::vector<float>& got, const std::vector<float>& expected,
                       const char* name) {
        if (got.size() != expected.size()) {
            fail(std::format("{} size mismatch: got {} expected {}", name, got.size(),
                             expected.size()));
        }
        for (size_t i = 0; i < got.size(); ++i) {
            if (!(std::isfinite(got[i]) && std::isfinite(expected[i])) ||
                std::abs(got[i] - expected[i]) > kFloatAbsTol) {
                fail(std::format("{} mismatch at {}: got {} expected {}", name, i, got[i],
                                 expected[i]));
            }
        }
    }

    template <typename Work, typename Validate>
    void run_timed(const char* name, const GpuBackend backend, const int repetitions, Work work,
                   Validate validate) {
        for (int w = 0; w < kWarmup; ++w) {
            auto warmup_out = work();
            wait_complete(backend);
            validate(warmup_out);
        }

        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(repetitions));
        for (int i = 0; i < repetitions; ++i) {
            samples.push_back(time_ms(backend, [&] { (void)work(); }));
        }
        print_csv_row(backend, name, median_ms(std::move(samples)), repetitions);
    }

    Tensor chw_rgb_to_hwc_u8(const Tensor& chw) {
        Tensor bytes = chw.clamp(0.0f, 1.0f).mul(255.0f).add(0.5f).to(DataType::UInt8);
        Tensor hwc = bytes.permute({1, 2, 0}).contiguous();
        materialize(hwc);
        return hwc;
    }

    std::vector<uint8_t> cpu_chw_rgb_to_hwc_u8(const std::vector<float>& chw, const size_t height,
                                               const size_t width) {
        const size_t pixels = height * width;
        std::vector<uint8_t> hwc(pixels * 3);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t p = y * width + x;
                for (size_t c = 0; c < 3; ++c) {
                    hwc[p * 3 + c] = image_path_u8(chw[c * pixels + p]);
                }
            }
        }
        return hwc;
    }

    Tensor fused_pointwise(const Tensor& input) {
        Tensor out = input.mul(kGain).add(kBias).mul(kContrast).clamp(0.0f, 1.0f);
        materialize(out);
        return out;
    }

    std::vector<float> cpu_fused_pointwise(const std::vector<float>& input) {
        std::vector<float> out(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            // GPU: mul/add/mul fuse, then eager clamp — same algebra, no mid clamp.
            out[i] = std::clamp((input[i] * kGain + kBias) * kContrast, 0.0f, 1.0f);
        }
        return out;
    }

    struct CopyResult {
        Tensor hwc;
        Tensor chw;
    };

    CopyResult noncontig_chw_hwc_copy(const Tensor& chw) {
        Tensor hwc = chw.permute({1, 2, 0}).contiguous();
        Tensor chw_back = hwc.permute({2, 0, 1}).contiguous();
        materialize(hwc);
        materialize(chw_back);
        return CopyResult{std::move(hwc), std::move(chw_back)};
    }

    std::vector<float> cpu_chw_to_hwc(const std::vector<float>& chw, const size_t height,
                                      const size_t width) {
        const size_t pixels = height * width;
        std::vector<float> hwc(pixels * 3);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t p = y * width + x;
                for (size_t c = 0; c < 3; ++c) {
                    hwc[p * 3 + c] = chw[c * pixels + p];
                }
            }
        }
        return hwc;
    }

    struct MaskResult {
        size_t count = 0;
        Tensor mask;
    };

    MaskResult mask_logical_reduce(const Tensor& values) {
        Tensor pred_hi = values.gt(0.0f);
        Tensor pred_lo = values.lt(0.5f);
        materialize(pred_hi);
        materialize(pred_lo);
        Tensor both = pred_hi.logical_and(pred_lo);
        materialize(both);
        MaskResult result;
        result.count = both.count_nonzero();
        result.mask = std::move(both);
        return result;
    }

    size_t cpu_mask_count(const std::vector<float>& values) {
        size_t count = 0;
        for (const float v : values) {
            if (v > 0.0f && v < 0.5f) {
                ++count;
            }
        }
        return count;
    }

    Tensor small_dispatch_add(const Tensor& a, const Tensor& b) {
        Tensor c;
        for (int k = 0; k < kSmallDispatchInner; ++k) {
            c = a.add(b);
            materialize(c);
        }
        materialize(c);
        return c;
    }

    std::vector<float> cpu_add(const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> out(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            out[i] = a[i] + b[i];
        }
        return out;
    }

    void bench_chw_rgb_to_hwc_u8(const GpuBackend backend, const int repetitions,
                                 const size_t height, const size_t width, const char* name) {
        const TensorShape chw_shape({size_t{3}, height, width});
        const auto host = patterned(chw_shape.elements());
        const auto expected = cpu_chw_rgb_to_hwc_u8(host, height, width);
        const Tensor gpu = upload_gpu(host, chw_shape, backend);

        run_timed(
            name, backend, repetitions, [&] { return chw_rgb_to_hwc_u8(gpu); },
            [&](const Tensor& out) {
                require_backend(out, backend, name);
                if (out.dtype() != DataType::UInt8 || out.ndim() != 3 ||
                    out.shape()[0] != height || out.shape()[1] != width || out.shape()[2] != 3) {
                    fail(std::format("{} produced unexpected shape/dtype", name));
                }
                expect_u8_eq(out.to_vector_uint8(), expected, name);
            });
    }

    void bench_fused_pointwise(const GpuBackend backend, const int repetitions,
                               const size_t height, const size_t width, const char* name) {
        const TensorShape shape({size_t{3}, height, width});
        const auto host = patterned(shape.elements());
        const auto expected = cpu_fused_pointwise(host);
        const Tensor gpu = upload_gpu(host, shape, backend);

        run_timed(
            name, backend, repetitions, [&] { return fused_pointwise(gpu); },
            [&](const Tensor& out) {
                require_backend(out, backend, name);
                expect_f32_eq(out.to_vector(), expected, name);
            });
    }

    void bench_noncontig_copy(const GpuBackend backend, const int repetitions,
                              const size_t height, const size_t width, const char* name) {
        const TensorShape chw_shape({size_t{3}, height, width});
        const TensorShape hwc_shape({height, width, size_t{3}});
        const auto host = patterned(chw_shape.elements());
        const auto expected_hwc = cpu_chw_to_hwc(host, height, width);
        const Tensor gpu = upload_gpu(host, chw_shape, backend);

        run_timed(
            name, backend, repetitions, [&] { return noncontig_chw_hwc_copy(gpu); },
            [&](const CopyResult& out) {
                require_backend(out.hwc, backend, name);
                require_backend(out.chw, backend, name);
                if (out.hwc.shape() != hwc_shape) {
                    fail(std::format("{} HWC shape mismatch", name));
                }
                if (out.chw.shape() != chw_shape) {
                    fail(std::format("{} CHW shape mismatch", name));
                }
                expect_f32_eq(out.hwc.to_vector(), expected_hwc, name);
                expect_f32_eq(out.chw.to_vector(), host, name);
            });
    }

    void bench_mask_logical_reduce(const GpuBackend backend, const int repetitions,
                                   const size_t items = kMaskItems,
                                   const char* name = "mask_logical_reduce_1m") {
        const auto host = patterned(items);
        const size_t expected = cpu_mask_count(host);
        const Tensor gpu = upload_gpu(host, TensorShape({items}), backend);

        run_timed(
            name, backend, repetitions, [&] { return mask_logical_reduce(gpu); },
            [&](const MaskResult& out) {
                require_backend(out.mask, backend, name);
                if (out.count != expected) {
                    fail(std::format("{} count mismatch: got {} expected {}", name, out.count,
                                     expected));
                }
                const auto bits = out.mask.to_vector_bool();
                if (bits.size() != host.size()) {
                    fail(std::format("{} mask size mismatch: got {} expected {}", name, bits.size(),
                                     host.size()));
                }
                for (size_t i = 0; i < host.size(); ++i) {
                    const bool want = host[i] > 0.0f && host[i] < 0.5f;
                    if (static_cast<bool>(bits[i]) != want) {
                        fail(std::format("{} mask mismatch at {}: got {} expected {}", name, i,
                                         static_cast<int>(bits[i]), static_cast<int>(want)));
                    }
                }
            });
    }

    void bench_dispatch_overhead_small(const GpuBackend backend, const int repetitions) {
        const char* name = "dispatch_overhead_small";
        const auto host_a = patterned(kDispatchElems);
        auto host_b = patterned(kDispatchElems);
        for (size_t i = 0; i < host_b.size(); ++i) {
            host_b[i] = pattern(i + 17);
        }
        const auto expected = cpu_add(host_a, host_b);
        const Tensor a = upload_gpu(host_a, TensorShape({kDispatchElems}), backend);
        const Tensor b = upload_gpu(host_b, TensorShape({kDispatchElems}), backend);

        run_timed(
            name, backend, repetitions, [&] { return small_dispatch_add(a, b); },
            [&](const Tensor& out) {
                require_backend(out, backend, name);
                expect_f32_eq(out.to_vector(), expected, name);
            });
    }

    void bench_byte_conversion(const GpuBackend backend, const int repetitions,
                               const size_t elements, const char* name) {
        std::vector<float> host(elements);
        std::vector<uint8_t> expected(elements);
        for (size_t i = 0; i < elements; ++i) {
            host[i] = static_cast<float>(i % 256);
            expected[i] = static_cast<uint8_t>(i % 256);
        }
        const Tensor input = upload_gpu(host, TensorShape({elements}), backend);
        run_timed(name, backend, repetitions, [&] {
            Tensor result = input.to(DataType::UInt8);
            materialize(result);
            return result; }, [&](const Tensor& result) {
            require_backend(result, backend, name);
            expect_u8_eq(result.to_vector_uint8(), expected, name); });
    }

    struct CaseSpec {
        const char* name;
        void (*run)(GpuBackend, int);
    };

    void run_rgb_720p(const GpuBackend backend, const int reps) {
        bench_chw_rgb_to_hwc_u8(backend, reps, 720, 1280, "chw_rgb_to_hwc_u8_720p");
    }
    void run_rgb_1080p(const GpuBackend backend, const int reps) {
        bench_chw_rgb_to_hwc_u8(backend, reps, 1080, 1920, "chw_rgb_to_hwc_u8_1080p");
    }
    void run_fused_720p(const GpuBackend backend, const int reps) {
        bench_fused_pointwise(backend, reps, 720, 1280, "fused_pointwise_720p");
    }
    void run_fused_1080p(const GpuBackend backend, const int reps) {
        bench_fused_pointwise(backend, reps, 1080, 1920, "fused_pointwise_1080p");
    }
    void run_copy_720p(const GpuBackend backend, const int reps) {
        bench_noncontig_copy(backend, reps, 720, 1280, "noncontig_chw_hwc_copy_720p");
    }
    void run_copy_1080p(const GpuBackend backend, const int reps) {
        bench_noncontig_copy(backend, reps, 1080, 1920, "noncontig_chw_hwc_copy_1080p");
    }
    void run_mask(const GpuBackend backend, const int reps) {
        bench_mask_logical_reduce(backend, reps);
    }
    void run_mask_16m(const GpuBackend backend, const int reps) {
        bench_mask_logical_reduce(backend, reps, 16'000'000, "mask_logical_reduce_16m");
    }
    void run_convert_1080p(const GpuBackend backend, const int reps) {
        bench_byte_conversion(backend, reps, 1920 * 1080 * 3, "float_to_uint8_1080p");
    }
    void run_dispatch(const GpuBackend backend, const int reps) {
        bench_dispatch_overhead_small(backend, reps);
    }

    constexpr CaseSpec kCases[] = {
        {"chw_rgb_to_hwc_u8_720p", run_rgb_720p},
        {"chw_rgb_to_hwc_u8_1080p", run_rgb_1080p},
        {"mask_logical_reduce_1m", run_mask},
        {"mask_logical_reduce_16m", run_mask_16m},
        {"float_to_uint8_1080p", run_convert_1080p},
        {"fused_pointwise_720p", run_fused_720p},
        {"fused_pointwise_1080p", run_fused_1080p},
        {"noncontig_chw_hwc_copy_720p", run_copy_720p},
        {"noncontig_chw_hwc_copy_1080p", run_copy_1080p},
        {"dispatch_overhead_small", run_dispatch},
    };

    void print_help() {
        std::printf(
            "Usage: bench_tensor_gui --backend cuda|vulkan [--repetitions N] [--case NAME]\n"
            "\n"
            "Standalone GUI-workload tensor benchmark (not ctest).\n"
            "  --backend cuda|vulkan   Required. Public tensor_backend selection.\n"
            "  --repetitions N         Timed samples per case (default %d). Median reported.\n"
            "  --case NAME             Run one case (default: all).\n"
            "\n"
            "Cases:\n",
            kDefaultRepetitions);
        for (const CaseSpec& spec : kCases) {
            std::printf("  %s\n", spec.name);
        }
        std::printf(
            "\nStdout CSV:\n"
            "  tensor_backend,<cuda|vulkan>\n"
            "  gpu_backend_name,<CUDA|Vulkan>\n"
            "  backend,case,median_ms,repetitions\n"
            "  ...\n");
        std::fflush(stdout);
    }

    [[nodiscard]] std::string_view take_value(const int argc, char** argv, int& i,
                                              const std::string_view arg,
                                              const std::string_view flag) {
        const std::string prefix = std::string(flag) + "=";
        if (arg.starts_with(prefix)) {
            return arg.substr(prefix.size());
        }
        if (arg != flag) {
            return {};
        }
        if (i + 1 >= argc) {
            usage_error(std::format("{} requires a value", flag));
        }
        ++i;
        return argv[i];
    }

} // namespace

int main(int argc, char** argv) {
    try {
        std::optional<GpuBackend> backend;
        int repetitions = kDefaultRepetitions;
        std::string case_filter;

        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_help();
                return 0;
            }
            if (arg == "--backend" || arg.starts_with("--backend=")) {
                const auto value = take_value(argc, argv, i, arg, "--backend");
                backend = parse_backend(value);
                if (!backend) {
                    usage_error(std::format("invalid --backend '{}'", value));
                }
                continue;
            }
            if (arg == "--repetitions" || arg.starts_with("--repetitions=")) {
                const auto value = take_value(argc, argv, i, arg, "--repetitions");
                int parsed = 0;
                const auto result =
                    std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
                    parsed < 1) {
                    usage_error(std::format("invalid --repetitions '{}'", value));
                }
                repetitions = parsed;
                continue;
            }
            if (arg == "--case" || arg.starts_with("--case=")) {
                case_filter = std::string(take_value(argc, argv, i, arg, "--case"));
                if (case_filter.empty()) {
                    usage_error("--case requires a name");
                }
                continue;
            }
            usage_error(std::format("unknown argument '{}'", arg));
        }

        if (!backend) {
            usage_error("missing required --backend cuda|vulkan");
        }

        if (!lfs::core::gpu_backend_available(*backend)) {
            fail(std::format("GPU backend '{}' is unavailable",
                             lfs::core::gpu_backend_name(*backend)));
        }

        const lfs::Status status = lfs::core::set_default_gpu_backend(*backend);
        if (!status.has_value()) {
            fail(std::string(status.error().user_message()));
        }
        GpuBackendScope scope(*backend);

        std::printf("tensor_backend,%s\n", backend_csv_id(*backend));
        std::printf("gpu_backend_name,%s\n", lfs::core::gpu_backend_name(*backend));
        std::printf("backend,case,median_ms,repetitions\n");
        std::fflush(stdout);

        bool ran = false;
        for (const CaseSpec& spec : kCases) {
            if (!case_filter.empty() && case_filter != spec.name) {
                continue;
            }
            spec.run(*backend, repetitions);
            ran = true;
        }
        if (!ran) {
            usage_error(std::format("unknown --case '{}'", case_filter));
        }
        return 0;
    } catch (const std::exception& ex) {
        fail(ex.what());
    }
}

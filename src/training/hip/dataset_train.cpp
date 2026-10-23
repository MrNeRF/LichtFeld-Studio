/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Explicit experimental fixed-topology runner. Not registered as an application
// backend; no automatic fallback, refinement, checkpoint resume or GUI access.
#include "camera_frame.hpp"
#include "prepared_image.hpp"
#include "tensor_model.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <core/image_io.hpp>
#include <core/logger.hpp>
#include <core/splat_data.hpp>
#include <core/tensor_backend.hpp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <io/exporter.hpp>
#include <io/loader.hpp>
#include <io/project_document.hpp>
#include <iomanip>
#include <numeric>
#include <random>
#include <tbb/task_arena.h>

namespace {
    using namespace lfs::core;
    using namespace lfs::training::hip;
    volatile std::sig_atomic_t interrupted = 0;
    void interrupt(int) { interrupted = 1; }
    unsigned integer(const char* value) {
        unsigned result{};
        const std::string_view input(value);
        auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), result);
        if (error != std::errc{} || end != input.data() + input.size() || !result)
            throw std::invalid_argument("Expected a positive integer");
        return result;
    }
    double elapsed(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    // PPM is an exact, dependency-free inspection artifact of float CHW RGB.
    void image(const std::filesystem::path& path, const LfsHipTrainFrame& frame, const std::vector<float>& values) {
        std::ofstream stream(path, std::ios::binary);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream << "P6\n"
               << frame.width << ' ' << frame.height << "\n255\n";
        const size_t pixels = size_t(frame.width) * frame.height;
        for (size_t p = 0; p < pixels; ++p)
            for (size_t c = 0; c < 3; ++c) {
                const auto byte = static_cast<unsigned char>(std::lround(std::clamp(values[c * pixels + p], 0.f, 1.f) * 255));
                stream.write(reinterpret_cast<const char*>(&byte), 1);
            }
    }
} // namespace
int main(int argc, char** argv) try {
    if (argc != 8) {
        std::fprintf(stderr, "Usage: lfs_hip_dataset_train MODULE DATASET NEW_OUTPUT STEPS MAX_WIDTH VIEWS SEED_PLY_OR_PROJECT_OR_points\n"
                             "Experimental: fixed topology, SH0, black background, L1+SSIM; Media Ingest decode, native HIP image preparation and training.\n"
                             "points uses sparse SfM points with explicit isotropic scale 0.01 world units and opacity 0.1.\n");
        return 2;
    }
    const auto module = std::filesystem::absolute(argv[1]);
    const auto dataset = std::filesystem::absolute(argv[2]);
    const auto output = std::filesystem::absolute(argv[3]);
    const auto steps = integer(argv[4]), max_width = integer(argv[5]), max_views = integer(argv[6]);
    if (max_width > 16384 || max_views > 100000 || steps > 10000000)
        throw std::invalid_argument("Experimental runner limit exceeded");
    if (!std::filesystem::is_regular_file(module) || !std::filesystem::is_directory(dataset))
        throw std::invalid_argument("Training module or dataset does not exist");
    if (!std::filesystem::create_directory(output))
        throw std::invalid_argument("Output directory already exists; choose a new directory");
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Logger::get().init(LogLevel::Debug, (output / "runtime.log").string(), "", true, output.string());
    std::signal(SIGINT, interrupt);
    GpuBackendScope scope(GpuBackend::HIP);
    const char* cpu_prep = std::getenv("LFS_HIP_CPU_IMAGE_PREP");
    const bool cpu_image_prep = cpu_prep && std::string_view(cpu_prep) == "1";
    const char* validate_prep = std::getenv("LFS_HIP_VALIDATE_IMAGE_PREP");
    const bool validate_image_prep = !cpu_image_prep && validate_prep && std::string_view(validate_prep) == "1";
    if (validate_image_prep)
        std::puts("[info] Diagnostic image validation enabled: CPU reference cost is included in preparation time.");
    std::printf("[info] Image preparation: %s; CPU decoded originals and persistent CPU targets; GPU source, when used, is transient and serialized.\n",
                cpu_image_prep ? "explicit CPU reference" : "native HIP area/Lanczos warp");
    std::puts("[info] Experimental fixed-topology SH0 training: HIP ordinal 0; Media Ingest CPU codec, combined resize/undistortion, bounded prepared targets in RAM; no training backend fallback, no refinement.");
    std::puts("[info] Experimental subset cache: no PipelinedImageLoader, JPEG blob cache or asynchronous prefetch; only prepared targets are uploaded to HIP.");
    // Initialize the native runtime on the controlling thread before the I/O
    // loader dispatches camera construction to worker threads.
    if (!gpu_backend_available(GpuBackend::HIP))
        throw std::runtime_error("HIP backend is unavailable; no fallback");
    const auto initialized = Tensor::zeros({1}, Device::GPU).cpu();
    std::puts("[info] Native HIP runtime initialized on the controlling thread.");
    auto loader = lfs::io::Loader::create();
    lfs::io::LoadOptions options;
    options.resize_factor = 1;
    options.max_width = max_width;
    options.splat_tensor_allocator = [](TensorShape shape, size_t, DataType dtype, std::string_view) {
        return Tensor::empty(shape, Device::CPU, dtype);
    };
    std::puts("[info] Loading dataset through application I/O...");
    tbb::task_arena loader_arena(1);
    auto loaded = loader_arena.execute([&] { return loader->load(dataset, options); });
    std::puts("[info] Dataset loader returned.");
    if (!loaded)
        throw std::runtime_error(loaded.error().format());
    auto* scene = std::get_if<lfs::io::LoadedScene>(&loaded->data);
    if (!scene || scene->cameras.empty())
        throw std::runtime_error("Dataset has no cameras");
    std::array<std::vector<float>, 5> parameters;
    if (std::string_view(argv[7]) == "points") {
        if (!scene->point_cloud || !scene->point_cloud->size())
            throw std::runtime_error("Dataset has no sparse points; provide a Gaussian SH0 PLY seed");
        if (scene->point_cloud->size() > 1000000)
            throw std::invalid_argument("Sparse points exceed the experimental one-million limit; no automatic truncation");
        parameters[0] = scene->point_cloud->means.cpu().contiguous().to_vector();
        const size_t n = parameters[0].size() / 3;
        auto colors = scene->point_cloud->colors.cpu().to(DataType::Float32).contiguous().to_vector();
        if (colors.size() != n * 3)
            throw std::runtime_error("Sparse point color count differs from positions");
        const float divisor = scene->point_cloud->colors.dtype() == DataType::UInt8 ? 255.f : 1.f;
        parameters[1].resize(n * 4, 0);
        parameters[2].resize(n * 3, std::log(0.01f));
        parameters[3].resize(n * 3);
        parameters[4].resize(n, std::log(0.1f / 0.9f));
        for (size_t g = 0; g < n; ++g) {
            parameters[1][g * 4] = 1;
            for (size_t c = 0; c < 3; ++c)
                parameters[3][g * 3 + c] = (colors[g * 3 + c] / divisor - 0.5f) / 0.28209479177387814f;
        }
        std::puts("[info] Initialization: sparse SfM points, scale=0.01, opacity=0.1. Fresh Adam; not a resumed application checkpoint.");
    } else {
        std::shared_ptr<SplatData> model;
        const auto seed_path = std::filesystem::absolute(argv[7]);
        if (seed_path.extension() == ".licht") {
            auto document = lfs::io::project::ProjectDocument::open(seed_path, {.defer_geometry_payloads = true});
            if (!document)
                throw std::runtime_error(std::string(document.error().user_message()));
            const auto bound = document->bound_checkpoint_uuid();
            if (!bound || !*bound)
                throw std::invalid_argument("Project seed requires a bound training checkpoint");
            const auto* chapter = document->find_checkpoint(**bound);
            if (!chapter)
                throw std::invalid_argument("Bound project checkpoint is missing");
            const auto read = chapter->visit_stream([&](std::istream& stream, uint64_t bytes) -> lfs::Result<void> {
                auto splats = load_checkpoint_splat_data(stream, bytes, options.splat_tensor_allocator);
                if (!splats)
                    throw std::runtime_error(splats.error());
                model = std::make_shared<SplatData>(std::move(*splats));
                return {};
            });
            if (!read)
                throw std::runtime_error(std::string(read.error().user_message()));
            std::puts("[info] Using bound project checkpoint parameters only; fresh HIP Adam, no checkpoint optimizer restore.");
        } else {
            auto seed = loader->load(seed_path, options);
            if (!seed)
                throw std::runtime_error(seed.error().format());
            if (auto* splats = std::get_if<std::shared_ptr<SplatData>>(&seed->data))
                model = *splats;
        }
        if (!model || model->get_max_sh_degree() != 0 || model->has_frozen_ranges() || model->deleted_count() != 0)
            throw std::invalid_argument("Seed requires SH0 with no frozen or deleted rows; unsupported features are not silently discarded");
        const std::array<Tensor, 5> tensors{model->means(), model->rotation_raw(), model->scaling_raw(), model->sh0(), model->opacity_raw()};
        for (size_t f = 0; f < 5; ++f)
            parameters[f] = tensors[f].cpu().contiguous().to_vector();
    }
    const size_t n = parameters[0].size() / 3;
    if (!n || n > 1000000)
        throw std::invalid_argument("Experimental runner requires 1..1000000 Gaussians; no automatic truncation");
    const size_t attributes[]{3, 4, 3, 3, 1};
    auto export_model = [&](const char* name) {
        auto tensor = [&](size_t f) { return Tensor::from_vector(parameters[f], {n, attributes[f]}, Device::CPU); };
        SplatData model(0, tensor(0), tensor(3).reshape({static_cast<int>(n), 1, 3}), Tensor::empty({n, 0, 3}, Device::CPU), tensor(2), tensor(1), tensor(4), 1.f);
        const auto saved = lfs::io::save_ply(model, {.output_path = output / name});
        if (!saved)
            throw std::runtime_error(saved.error().format());
    };
    export_model("initial.ply");
    const auto layout = Session::model_layout(module, n);
    auto allocation = std::make_shared<Tensor>(Tensor::empty({size_t(layout.required_bytes)}, Device::GPU, DataType::UInt8));
    using Storage = SplatExportableStorage;
    auto storage = std::make_shared<Storage::Control>();
    storage->block = std::make_shared<ExportableBlock>();
    storage->block->device_ptr = const_cast<void*>(std::as_const(*allocation).data_ptr());
    storage->block->reserved_bytes = layout.required_bytes;
    storage->capacity = n;
    storage->generation = 1;
    const size_t family[]{Storage::Means, Storage::Rotation, Storage::Scaling, Storage::Sh0, Storage::Opacity};
    for (size_t f = 0; f < 5; ++f) {
        if (parameters[f].size() != n * attributes[f])
            throw std::invalid_argument("Seed parameter dimensions are inconsistent");
        storage->region_offsets[family[f]] = layout.offsets_bytes[f];
        storage->region_bytes[family[f]] = n * attributes[f] * sizeof(float);
        auto view = Tensor::view_sharing_storage(*allocation, layout.offsets_bytes[f], {n, attributes[f]}, n, DataType::Float32, "hip.dataset.model");
        view.copy_from(Tensor::from_vector(parameters[f], {n, attributes[f]}, Device::CPU));
    }
    TensorModel tensor_model(allocation, storage, n, 0);
    LfsHipTrainConfig config{sizeof(config), 0, n, 2ull * 1024 * 1024 * 1024};
    auto session = tensor_model.open(module, config);
    const bool fast_raster = std::getenv("LFS_HIP_FAST_RASTER") != nullptr;
    if (fast_raster)
        session->set_rasterizer(1);
    const bool fused_optimizer = std::getenv("LFS_HIP_FUSED_ADAM") != nullptr;
    if (fused_optimizer)
        session->set_optimizer(1);
    std::printf("[info] HIP rasterizer: %s (explicit selection, no training backend fallback).\n", fast_raster ? "FastGS" : "3DGUT");
    std::printf("[info] HIP optimizer: %s.\n", fused_optimizer ? "fused joint moments (16-bit base, 8-bit SH-rest), FP32 parameters" : "FP32 reference");
    session->set_loss({sizeof(LfsHipTrainLoss), LFS_HIP_LOSS_L1_SSIM, 0.2f, 1});
    // A bounded explicit subset for initial experiments; no implied holdout.
    const size_t count = std::min<size_t>(scene->cameras.size(), max_views);
    std::vector<CameraFrame> frames;
    size_t target_bytes = 0;
    constexpr size_t cpu_target_budget = 512ull * 1024 * 1024;
    std::unique_ptr<HostTargets> host_targets;
    if (const char* enabled = std::getenv("LFS_HIP_HOST_TARGETS"); enabled && std::string_view(enabled) == "1") {
        size_t budget = cpu_target_budget;
        try {
            if (const char* limit = std::getenv("LFS_HIP_HOST_TARGET_BUDGET"))
                budget = std::min<size_t>(budget, integer(limit));
            host_targets = std::make_unique<HostTargets>(module, config.device_ordinal, budget);
            std::printf("[info] HIP owned host targets enabled; pinned budget %zu bytes.\n", budget);
        } catch (const std::exception& error) { /* LFS-CENSUS-OK(empty-catch): Report optional host-target fallback or exit with an explicit failure status. */
            std::fprintf(stderr, "[warning] HIP host target fallback to ordinary RAM: %s\n", error.what());
        }
    }
    std::ofstream preparation(output / "preparation.csv", std::ios::binary);
    preparation.exceptions(std::ios::badbit | std::ios::failbit);
    preparation << "view,calibration_ms,decode_resize_ms,planar_ms,undistort_ms,ownership_ms,width,height,bytes,source_upload_ms,warp_readback_ms,gpu_live_before,gpu_live_after,gpu_pool_after,validation_ms,reference_max_error,reference_mse,image_prepare_backend\n";
    double preparation_ms = 0;
    size_t image_prep_fallbacks = 0, cpu_prepared_views = 0, gpu_prepared_views = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto prepare_start = std::chrono::steady_clock::now();
        auto stage_start = prepare_start;
        auto& camera = *scene->cameras[i * scene->cameras.size() / count];
        camera.prepare_undistortion();
        const double calibration_ms = elapsed(stage_start);
        stage_start = std::chrono::steady_clock::now();
        const auto [source_width, source_height, source_channels] = get_image_info(camera.image_path());
        // Keep both transient native-byte buffers bounded before decode. The
        // persistent target budget below is separate from source preparation.
        constexpr size_t native_source_budget = 256ull * 1024 * 1024;
        if (source_width <= 0 || source_height <= 0 || size_t(source_width) * source_height > native_source_budget / 3)
            throw std::runtime_error("Native RGB source exceeds experimental 256 MiB preparation budget");
        const auto destination = camera.is_undistort_prepared()
                                     ? prepare_undistort_params(camera.undistort_params(), source_width, source_height, 1, max_width)
                                     : UndistortParams{};
        const auto [output_width, output_height] = camera.is_undistort_prepared()
                                                       ? std::pair{destination.dst_width, destination.dst_height}
                                                       : resized_image_dimensions(source_width, source_height, 1, max_width);
        if (output_width <= 0 || output_height <= 0 || size_t(output_width) * output_height > (cpu_target_budget - target_bytes) / 12)
            throw std::runtime_error("Prepared CPU image subset exceeds explicit 512 MiB budget; reduce VIEWS or MAX_WIDTH");
        auto [raw, width, height, channels] = load_image(camera.image_path(), 1, camera.is_undistort_prepared() ? 0 : max_width);
        std::unique_ptr<void, decltype(&free_image)> pixels(raw, free_image);
        if (!raw || channels != 3 || width <= 0 || height <= 0)
            throw std::runtime_error("CPU RGB decode failed");
        const double decode_resize_ms = elapsed(stage_start);
        stage_start = std::chrono::steady_clock::now();
        // Match Media Ingest: retain the native byte source for a single
        // combined resize/undistort, never resize and quantize it first.
        auto target = Tensor::empty({3, size_t(height), size_t(width)}, Device::CPU, DataType::UInt8);
        auto* values = target.ptr<uint8_t>();
        const size_t area = size_t(width) * height;
        for (size_t p = 0; p < area; ++p)
            for (size_t c = 0; c < 3; ++c)
                values[c * area + p] = raw[p * 3 + c];
        const double planar_ms = elapsed(stage_start);
        stage_start = std::chrono::steady_clock::now();
        double source_upload_ms = 0, warp_readback_ms = 0, validation_ms = 0, reference_max_error = 0, reference_mse = 0;
        size_t gpu_live_before = 0, gpu_live_after = 0, gpu_pool_after = 0;
        bool prepared_on_gpu = false;
        if (camera.is_undistort_prepared()) {
            const auto params = prepare_undistort_params(camera.undistort_params(), width, height, 1, max_width);
            const float footprint = std::max(params.src_fx / params.dst_fx, params.src_fy / params.dst_fy);
            const double quadrature = std::max(8.0, std::ceil(double(footprint)));
            const bool bounded_work_fallback = !cpu_image_prep && std::isfinite(footprint) &&
                                               (footprint > 64 || double(params.dst_width) * quadrature * quadrature > 2 * 1024 * 1024);
            if (bounded_work_fallback) {
                ++image_prep_fallbacks;
                std::fprintf(stderr, "[warn] HIP image preparation fallback for view %zu: native sampling work limit exceeded (footprint %.3f); using CPU area/Lanczos. Training remains HIP.\n", i, footprint);
            }
            if (cpu_image_prep || bounded_work_fallback) {
                target = undistort_cpu_target(target, params);
            } else {
                prepared_on_gpu = true;
                const auto reference_source = validate_image_prep ? target : Tensor{};
                gpu_live_before = gpu_backend_memory_info(GpuBackend::HIP, true).allocated_bytes;
                {
                    const auto upload_start = std::chrono::steady_clock::now();
                    const auto gpu_source = target.to(Device::GPU);
                    source_upload_ms = elapsed(upload_start);
                    const auto warp_start = std::chrono::steady_clock::now();
                    target = undistort_image(gpu_source, params, nullptr).cpu();
                    warp_readback_ms = elapsed(warp_start);
                }
                const auto memory = gpu_backend_memory_info(GpuBackend::HIP, true);
                gpu_live_after = memory.allocated_bytes;
                gpu_pool_after = memory.pool_reserved_current;
                if (gpu_live_after != gpu_live_before)
                    throw std::runtime_error("HIP image preparation retained a transient GPU allocation after readback");
                if (validate_image_prep) {
                    const auto validation_start = std::chrono::steady_clock::now();
                    const auto reference = undistort_cpu_target(reference_source, params);
                    const auto* expected = reference.ptr<float>();
                    const auto* actual = std::as_const(target).ptr<float>();
                    for (size_t p = 0; p < target.numel(); ++p) {
                        if (!std::isfinite(actual[p]))
                            throw std::runtime_error("HIP image preparation produced a nonfinite target");
                        const double delta = double(actual[p]) - expected[p];
                        reference_max_error = std::max(reference_max_error, std::abs(delta));
                        reference_mse += delta * delta / target.numel();
                    }
                    validation_ms = elapsed(validation_start);
                    std::printf("[info] Prepared target vs CPU: max %.9g, MSE %.9g; diagnostic CPU reference %.3f ms.\n",
                                reference_max_error, reference_mse, validation_ms);
                    if (reference_max_error > 4e-5 || reference_mse > 1e-10)
                        throw std::runtime_error("HIP image preparation differs from the CPU reference");
                }
            }
            // Portable ingest's non-float decode contract is the 16-bit grid.
            auto* samples = target.ptr<float>();
            for (size_t p = 0; p < target.numel(); ++p) {
                if (!std::isfinite(samples[p]))
                    throw std::runtime_error("Image preparation produced a nonfinite target");
                samples[p] = float(int(std::clamp(samples[p], 0.f, 1.f) * 65535.f + 0.5f)) / 65535.f;
            }
        } else {
            target = target.to(DataType::Float32) / 255.f;
        }
        if (prepared_on_gpu)
            ++gpu_prepared_views;
        else
            ++cpu_prepared_views;
        const double undistort_ms = elapsed(stage_start);
        stage_start = std::chrono::steady_clock::now();
        if (target.bytes() > cpu_target_budget - target_bytes)
            throw std::runtime_error("Prepared CPU image subset exceeds explicit 512 MiB budget; reduce VIEWS or MAX_WIDTH");
        target_bytes += target.bytes();
        frames.emplace_back(camera, target, 0.01f, 1000.f, camera.is_undistort_prepared(), host_targets.get());
        const double ownership_ms = elapsed(stage_start);
        preparation_ms += elapsed(prepare_start);
        const auto descriptor = frames.back().descriptor();
        preparation << i << ',' << calibration_ms << ',' << decode_resize_ms << ',' << planar_ms << ','
                    << undistort_ms << ',' << ownership_ms << ',' << descriptor.width << ',' << descriptor.height
                    << ',' << target.bytes() << ',' << source_upload_ms << ',' << warp_readback_ms
                    << ',' << gpu_live_before << ',' << gpu_live_after << ',' << gpu_pool_after
                    << ',' << validation_ms << ',' << reference_max_error << ',' << reference_mse << ',' << (prepared_on_gpu ? "hip" : "cpu") << '\n';
        preparation.flush();
        std::printf("[info] Prepared view %zu/%zu (%u x %u); CPU targets %zu bytes\n", i + 1, count, frames.back().descriptor().width, frames.back().descriptor().height, target_bytes);
        std::fflush(stdout);
    }
    std::ofstream csv(output / "steps.csv", std::ios::binary);
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "step,view,frame_ms,step_ms,allocated_bytes,intersections\n";
    std::mt19937 random(42);
    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::printf("[info] Ready: %zu Gaussians, %zu views, CPU target cache %zu bytes; seed=42, L1/SSIM weight=0.2, valid padding, session budget=2 GiB.\n", n, count, target_bytes);
    auto evaluate = [&](const char* tag) {
        double mse = 0;
        for (size_t v = 0; v < count; ++v) {
            frames[v].bind(*session);
            const auto frame = frames[v].descriptor();
            std::vector<float> rendered(size_t(frame.width) * frame.height * 3);
            session->render(rendered);
            double error = 0;
            for (size_t p = 0; p < rendered.size(); ++p) {
                const double delta = double(rendered[p]) - frame.target_chw[p];
                error += delta * delta;
            }
            error /= rendered.size();
            mse += error / count;
            image(output / (std::string(tag) + "-" + std::to_string(v) + ".ppm"), frame, rendered);
            if (std::string_view(tag) == "before")
                image(output / ("target-" + std::to_string(v) + ".ppm"), frame, std::vector<float>(frame.target_chw, frame.target_chw + rendered.size()));
        }
        std::printf("[info] %s training-view MSE %.9g PSNR %.6f; not holdout evaluation\n", tag, mse, -10 * std::log10(std::max(mse, 1e-30)));
        return mse;
    };
    const double before = evaluate("before");
    unsigned completed = 0;
    // evaluate() leaves the final immutable owner bound. Reusing that exact
    // owner requires neither validation nor another host-to-device transfer.
    // Keep this optimization at the runner level: the borrowed-pointer C ABI
    // must continue to accept callers mutating a target in place.
    size_t bound_view = count - 1;
    size_t frame_uploads = 0;
    for (unsigned step = 0; step < steps && !interrupted; ++step) {
        if (step % count == 0)
            std::shuffle(order.begin(), order.end(), random);
        const size_t view = order[step % count];
        auto start = std::chrono::steady_clock::now();
        if (view != bound_view) {
            frames[view].bind(*session);
            bound_view = view;
            ++frame_uploads;
        }
        const double frame_ms = elapsed(start);
        const float decay = std::exp(std::log(0.01f) * float(step) / steps);
        const LfsHipTrainRates rates{0.00016f * decay, 0.001f, 0.005f, 0.0025f, 0.05f};
        start = std::chrono::steady_clock::now();
        session->step(rates);
        const double step_ms = elapsed(start);
        const auto info = session->info();
        csv << step + 1 << ',' << view << ',' << frame_ms << ',' << step_ms << ',' << info.allocated_bytes << ',' << info.intersection_count << '\n';
        completed = step + 1;
        if (completed == 1 || completed % 10 == 0 || completed == steps) {
            std::printf("[info] HIP step %u/%u: %.3f ms, frame %.3f ms, session %llu bytes\n", completed, steps, step_ms, frame_ms, static_cast<unsigned long long>(info.allocated_bytes));
            std::fflush(stdout);
            csv.flush();
        }
    }
    const double after = evaluate("after");
    for (size_t f = 0; f < 5; ++f)
        session->read_parameter(f, parameters[f]);
    export_model("trained.ply");
    auto roundtrip = loader->load(output / "trained.ply", options);
    if (!roundtrip)
        throw std::runtime_error(roundtrip.error().format());
    auto* restored = std::get_if<std::shared_ptr<SplatData>>(&roundtrip->data);
    if (!restored || !*restored || (*restored)->size() != n)
        throw std::runtime_error("Exported HIP model did not round-trip through the application loader");
    const std::array<Tensor, 5> restored_tensors{(*restored)->means(), (*restored)->rotation_raw(), (*restored)->scaling_raw(), (*restored)->sh0(), (*restored)->opacity_raw()};
    for (size_t f = 0; f < 5; ++f) {
        auto expected = parameters[f];
        // The application's SplatData PLY exporter normalizes quaternions.
        // Preserve that existing contract and compare against its mathematical
        // result, rather than requiring the raw optimizer quaternion norm.
        if (f == 1)
            for (size_t g = 0; g < n; ++g) {
                double norm = 0;
                for (size_t c = 0; c < 4; ++c)
                    norm += double(expected[g * 4 + c]) * expected[g * 4 + c];
                const double inverse = 1 / std::max(std::sqrt(norm), 1e-12);
                for (size_t c = 0; c < 4; ++c)
                    expected[g * 4 + c] = float(expected[g * 4 + c] * inverse);
            }
        const auto actual = restored_tensors[f].cpu().contiguous().to_vector();
        if (actual.size() != expected.size())
            throw std::runtime_error("Exported HIP parameter dimensions changed after PLY reload");
        for (size_t i = 0; i < actual.size(); ++i)
            if (!std::isfinite(actual[i]) || (f == 1 ? std::abs(actual[i] - expected[i]) > 3e-7f : actual[i] != expected[i]))
                throw std::runtime_error("Exported HIP model parameters differ after application PLY reload");
    }
    if (gpu_backend_live(GpuBackend::Vulkan))
        throw std::runtime_error("Experimental HIP runner unexpectedly initialized a Vulkan Tensor context");
    std::puts("[info] PLY round-trip: four parameter families exact; quaternions normalized as in the application exporter (tolerance 3e-7). No Vulkan Tensor context initialized.");
    const auto final_info = session->info();
    std::printf("[info] Native session memory: live=%llu bytes, peak=%llu bytes, budget=%llu bytes.\n",
                static_cast<unsigned long long>(final_info.allocated_bytes),
                static_cast<unsigned long long>(final_info.peak_allocated_bytes),
                static_cast<unsigned long long>(final_info.budget_bytes));
    std::ofstream summary(output / "summary.json", std::ios::binary);
    summary.exceptions(std::ios::badbit | std::ios::failbit);
    summary << std::setprecision(12) << "{\"experimental_fixed_topology\":true,\"sh_degree\":0,\"views\":" << count << ",\"gaussians\":" << n
            << ",\"rasterizer\":\"" << (fast_raster ? "FastGS" : "3DGUT") << "\",\"optimizer\":\"" << (fused_optimizer ? "joint-fused" : "fp32") << "\""
            << ",\"steps\":" << completed << ",\"interrupted\":" << (interrupted ? "true" : "false")
            << ",\"image_prepare_backend\":\"" << (gpu_prepared_views ? (cpu_prepared_views ? "mixed" : "hip") : "cpu") << "\""
            << ",\"image_prep_fallbacks\":" << image_prep_fallbacks
            << ",\"gpu_prepared_views\":" << gpu_prepared_views << ",\"cpu_prepared_views\":" << cpu_prepared_views
            << ",\"image_validation\":" << (validate_image_prep ? "true" : "false")
            << ",\"preparation_ms\":" << preparation_ms << ",\"training_frame_uploads\":" << frame_uploads
            << ",\"max_width\":" << max_width << ",\"cpu_target_bytes\":" << target_bytes
            << ",\"hip_host_target_bytes\":" << (host_targets ? host_targets->allocated_bytes() : 0)
            << ",\"session_live_bytes\":" << final_info.allocated_bytes << ",\"session_peak_bytes\":" << final_info.peak_allocated_bytes
            << ",\"shuffle_seed\":42,\"ssim_weight\":0.2,\"session_budget_bytes\":2147483648,\"training_view_mse_before\":" << before << ",\"training_view_mse_after\":" << after << "}\n";
    session->close();
    std::puts("[info] Saved trained.ply, steps.csv, summary.json and before/after/target PPM images. Fresh optimizer only; PLY is not a resumable optimizer checkpoint.");
    return interrupted ? 130 : 0;
} catch (const std::exception& error) { /* LFS-CENSUS-OK(empty-catch): Report optional host-target fallback or exit with an explicit failure status. */
    std::fprintf(stderr, "[error] Experimental HIP dataset training stopped without fallback: %s\n", error.what());
    return 1;
}

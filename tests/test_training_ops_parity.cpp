/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Cross-backend parity for the training ops tables.
//
// Each family builds the deterministic inputs used by that family's exact-byte
// fixture, runs the CUDA table, runs the second backend's table on a copy of
// those inputs, and compares outputs. Integer and bookkeeping results are
// exact. Float tolerances below are the cross-backend budget; the self-check
// ignores them and requires identical bytes.
//
// Every family runs against Vulkan and Metal. An unfilled slot or a missing
// device skips with the family or backend name. TrainingOpsParity.SelfCheck
// compares CUDA with itself.

#include "core/camera.hpp"
#include "core/error.hpp"
#include "core/event_bridge/control_boundary.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"
#include "core/tensor_image.hpp"
#include "cuda_backend_test.hpp"
#include "io/dataset_scene_import.hpp"
#include "lfs/training/idle_arena_scratch.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/ops/fast_services.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/refine_scratch.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

extern "C" int stbi_write_png(char const* filename, int w, int h, int comp, const void* data, int stride_in_bytes);

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace ops = lfs::gpu_ops;
    namespace joint = lfs::training::joint_adam;
    namespace quant = lfs::core::sh_value_quant;

    // Cross-backend budgets. 0/0 is exact, including integer bookkeeping.
    struct Tol {
        double abs = 0;
        double rel = 0;
    };

    constexpr Tol kExact{0, 0};
    constexpr Tol kReduce{1e-5, 1e-5}; // photometric, geometry, mask, extra-loss reductions
    constexpr Tol kRaster{1e-4, 1e-4}; // blended raster outputs
    constexpr Tol kAdam{1e-6, 1e-5};   // optimizer parameters after a short step
    constexpr Tol kConv{1e-4, 1e-4};   // lpips / bilateral / ppisp float math
    // 200-step synthetic loss. Two CUDA runs drift by a few thousandths; the
    // raster blend is an atomic reduction, so the limit is abs + rel*|expected|.
    constexpr Tol kLoss{1e-2, 1e-2};

    enum class Kind { Missing,
                      Exact,
                      Float };

    struct Field {
        std::string name;
        Kind kind = Kind::Missing;
        Tol tol{};
        std::vector<uint8_t> bytes;
        std::vector<float> values;
    };

    struct Snapshot {
        std::vector<Field> fields;

        void missing(const std::string& name) { fields.push_back({name, Kind::Missing, {}}); }

        void exact_bytes(const std::string& name, const void* data, size_t size) {
            Field field{name, Kind::Exact, kExact, {}, {}};
            const auto* bytes = static_cast<const uint8_t*>(data);
            field.bytes.assign(bytes, bytes + size);
            fields.push_back(std::move(field));
        }

        void exact_i(const std::string& name, int64_t value) {
            exact_bytes(name, &value, sizeof(value));
        }

        void exact_u(const std::string& name, uint64_t value) {
            exact_bytes(name, &value, sizeof(value));
        }
    };

    bool same_float(float actual, float expected, Tol tol) {
        if (std::isnan(actual) || std::isnan(expected)) {
            return std::isnan(actual) && std::isnan(expected);
        }
        const double diff = std::abs(static_cast<double>(actual) - static_cast<double>(expected));
        const double limit = tol.abs + tol.rel * std::abs(static_cast<double>(expected));
        return diff <= limit;
    }

    // Empty when the snapshots agree. exact forces identical bytes on every field.
    std::string compare_snapshots(const Snapshot& actual, const Snapshot& expected, bool exact) {
        if (actual.fields.size() != expected.fields.size()) {
            return "field count " + std::to_string(actual.fields.size()) + " vs " +
                   std::to_string(expected.fields.size());
        }
        for (size_t i = 0; i < actual.fields.size(); ++i) {
            const Field& a = actual.fields[i];
            const Field& b = expected.fields[i];
            if (a.name != b.name) {
                return "field name " + a.name + " vs " + b.name;
            }
            if (a.kind != b.kind) {
                return a.name + " presence/kind mismatch";
            }
            if (a.kind == Kind::Missing) {
                continue;
            }
            if (exact || a.kind == Kind::Exact || (a.tol.abs == 0 && a.tol.rel == 0)) {
                if (a.bytes != b.bytes) {
                    return a.name + " bytes differ";
                }
                continue;
            }
            if (a.values.size() != b.values.size()) {
                return a.name + " length mismatch";
            }
            for (size_t j = 0; j < a.values.size(); ++j) {
                if (!same_float(a.values[j], b.values[j], a.tol)) {
                    return a.name + " index " + std::to_string(j) + " " + std::to_string(a.values[j]) +
                           " vs " + std::to_string(b.values[j]);
                }
            }
        }
        return {};
    }

    void keep(Snapshot& snapshot, GpuBackend backend, const std::string& name, const Tensor& tensor, Tol tol) {
        if (!tensor.is_valid() || tensor.numel() == 0) {
            snapshot.missing(name);
            return;
        }
        if (backend == GpuBackend::CUDA) {
            const cudaError_t status = cudaDeviceSynchronize();
            if (status != cudaSuccess) {
                snapshot.exact_bytes(name + ".sync", cudaGetErrorString(status), std::strlen(cudaGetErrorString(status)));
                return;
            }
        }
        Tensor view = tensor;
        if (view.dtype() == DataType::Float16) {
            view = view.to(DataType::Float32);
        }
        auto cpu = view.cpu().contiguous();
        const auto* raw = static_cast<const uint8_t*>(cpu.data_ptr());
        Field field{name, Kind::Exact, tol, {raw, raw + cpu.bytes()}, {}};
        if (cpu.dtype() == DataType::Float32 && (tol.abs != 0 || tol.rel != 0)) {
            field.kind = Kind::Float;
            field.values.resize(cpu.numel());
            std::memcpy(field.values.data(), cpu.data_ptr(), cpu.bytes());
        }
        snapshot.fields.push_back(std::move(field));
    }

    void keep_f(Snapshot& snapshot, const std::string& name, float value, Tol tol) {
        Field field{name, Kind::Float, tol, {}, {value}};
        const auto bits = std::bit_cast<uint32_t>(value);
        field.bytes.assign(reinterpret_cast<const uint8_t*>(&bits), reinterpret_cast<const uint8_t*>(&bits) + sizeof(bits));
        if (tol.abs == 0 && tol.rel == 0) {
            field.kind = Kind::Exact;
        }
        snapshot.fields.push_back(std::move(field));
    }

    struct Capture {
        Snapshot snapshot;
        std::string error;
    };

    const char* backend_name(const GpuBackend backend) {
        switch (backend) {
        case GpuBackend::CUDA: return "Cuda";
        case GpuBackend::Vulkan: return "Vulkan";
        case GpuBackend::Metal: return "Metal";
        }
        return "Unknown";
    }

    bool family_present(const lfs::training::TrainingOps& table, lfs::training::Family family) {
        switch (family) {
        case lfs::training::Family::Session: return table.session != nullptr;
        case lfs::training::Family::Fast: return table.fast != nullptr;
        case lfs::training::Family::Gsplat: return table.gsplat != nullptr;
        case lfs::training::Family::Photometric: return table.photometric != nullptr;
        case lfs::training::Family::Geometry: return table.geometry != nullptr;
        case lfs::training::Family::Masks: return table.masks != nullptr;
        case lfs::training::Family::ExtraLoss: return table.extra_loss != nullptr;
        case lfs::training::Family::Adam: return table.adam != nullptr;
        case lfs::training::Family::Mcmc: return table.mcmc != nullptr;
        case lfs::training::Family::Mrnf: return table.mrnf != nullptr;
        case lfs::training::Family::Refine: return table.refine != nullptr;
        case lfs::training::Family::Bilateral: return table.bilateral != nullptr;
        case lfs::training::Family::PPISP: return table.ppisp != nullptr;
        case lfs::training::Family::Controller: return table.controller != nullptr;
        case lfs::training::Family::Morton: return table.morton != nullptr;
        case lfs::training::Family::Sh: return table.sh != nullptr;
        case lfs::training::Family::TrainingImage: return table.training_image != nullptr;
        case lfs::training::Family::SharedImage: return table.shared_image != nullptr;
        case lfs::training::Family::Lpips: return table.lpips != nullptr;
        case lfs::training::Family::Count: return false;
        }
        return false;
    }

    // Fixture generators. Seeds and shapes match the family's exact-byte test.
    std::vector<float> pattern_values(size_t count, float scale, int seed) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            const auto k = static_cast<int>((i * 7919u + static_cast<size_t>(seed) * 104729u) % 2003u);
            values[i] = scale * (static_cast<float>(k) / 1001.0f - 1.0f);
        }
        return values;
    }

    Tensor pattern(const lfs::core::TensorShape& shape, float scale, int seed) {
        return Tensor::from_vector(pattern_values(shape.elements(), scale, seed), shape, Device::GPU);
    }

    Tensor pattern_short(const lfs::core::TensorShape& shape, float offset = 0.f) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = offset + static_cast<float>((i * 17 + 3) % 101) / 101.f;
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor pattern_mrnf(const lfs::core::TensorShape& shape, float scale, int seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            const auto k = static_cast<int>((i * 17u + static_cast<size_t>(seed) * 13u) % 97u);
            values[i] = scale * (static_cast<float>(k) / 48.f - 1.f);
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor bool_mask(size_t count, size_t period) {
        std::vector<bool> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = i % period == 0;
        }
        return Tensor::from_vector(values, {count}, Device::GPU);
    }

    Tensor i64_rows(const std::vector<int64_t>& values) {
        auto cpu = Tensor::empty({values.size()}, Device::CPU, DataType::Int64);
        for (size_t i = 0; i < values.size(); ++i) {
            cpu.ptr<int64_t>()[i] = values[i];
        }
        return cpu.gpu();
    }

    Tensor i32_rows(const std::vector<int>& values) {
        auto cpu = Tensor::empty({values.size()}, Device::CPU, DataType::Int32);
        for (size_t i = 0; i < values.size(); ++i) {
            cpu.ptr<int>()[i] = values[i];
        }
        return cpu.gpu();
    }

    Tensor image_bytes(const lfs::core::TensorShape& shape) {
        auto cpu = Tensor::empty(shape, Device::CPU, DataType::UInt8);
        auto* bytes = static_cast<uint8_t*>(cpu.data_ptr());
        for (size_t i = 0; i < cpu.bytes(); ++i) {
            bytes[i] = static_cast<uint8_t>((i * 71 + 149) % 256);
        }
        return cpu.gpu();
    }

    Tensor image_floats(const lfs::core::TensorShape& shape) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>((i * 7919 + 104729) % 2003) / 2002.f;
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    float bc1(int step) { return static_cast<float>(1.0 / (1.0 - std::pow(0.9, step))); }
    float bc2(int step) { return static_cast<float>(1.0 / std::sqrt(1.0 - std::pow(0.999, step))); }

    Capture capture(lfs::training::Family family, GpuBackend backend);

    Capture capture_photometric(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).photometric;
        constexpr int H = 32;
        constexpr int W = 40;
        constexpr float kWeight = 0.2f;
        auto fixed = [](float bias) {
            auto cpu = Tensor::empty({1, 3, 32, 40}, Device::CPU);
            float* values = cpu.ptr<float>();
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < 32; ++y) {
                    for (int x = 0; x < 40; ++x) {
                        values[(c * 32 + y) * 40 + x] = bias + 0.01f * static_cast<float>(c) + 0.001f * static_cast<float>(x + y);
                    }
                }
            }
            return cpu.gpu();
        };
        const Tensor corrected = fixed(0.2f);
        const Tensor raw = fixed(0.05f);
        const Tensor target = fixed(0.4f);
        auto mask_cpu = Tensor::empty({32, 40}, Device::CPU);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                mask_cpu.ptr<float>()[y * W + x] = ((x + y) % 5 == 0) ? 0.f : 1.f;
            }
        }
        const Tensor mask = mask_cpu.gpu();
        const Tensor predicted = fixed(0.3f);
        const Tensor metric_target = fixed(0.6f);
        ops::PhotoSaved saved{.backend = table->create()};
        if (!saved.backend) {
            out.error = "photometric create returned null";
            return out;
        }
        struct Path {
            const char* name;
            ops::PhotoPath path;
            float weight;
            bool raw;
            bool mask;
        };
        const Path paths[] = {
            {"l1", ops::PhotoPath::L1, 0.f, false, false},
            {"ssim", ops::PhotoPath::SSIM, 1.f, false, false},
            {"fused", ops::PhotoPath::Fused, kWeight, false, false},
            {"decoupled", ops::PhotoPath::Decoupled, kWeight, true, false},
            {"masked_fused", ops::PhotoPath::MaskedFused, kWeight, false, true},
            {"masked_decoupled", ops::PhotoPath::MaskedDecoupled, kWeight, true, true},
        };
        Tensor absent;
        for (const Path& path : paths) {
            Tensor loss, grad_corrected, grad_raw;
            table->evaluate(saved, corrected, path.raw ? raw : absent, target, path.mask ? mask : absent,
                            {.path = path.path, .ssim_weight = path.weight, .valid_padding = true},
                            loss, grad_corrected, grad_raw);
            keep(out.snapshot, backend, std::string("photometric.evaluate.") + path.name + ".loss", loss, kReduce);
            keep(out.snapshot, backend, std::string("photometric.evaluate.") + path.name + ".grad", grad_corrected, kReduce);
            keep(out.snapshot, backend, std::string("photometric.evaluate.") + path.name + ".grad_raw", grad_raw, kReduce);
        }
        const Tensor metric = table->metric(saved, predicted, metric_target, false, true);
        keep(out.snapshot, backend, "photometric.metric", metric, kReduce);
        const Tensor mapped = table->metric(saved, predicted, metric_target, true, false);
        keep(out.snapshot, backend, "photometric.metric_maps", mapped, kReduce);
        keep(out.snapshot, backend, "photometric.ssim_map", saved.ssim_map, kReduce);
        keep(out.snapshot, backend, "photometric.cs_map", saved.cs_map, kReduce);
        Tensor error;
        table->error_map(saved, predicted, metric_target, error, true);
        keep(out.snapshot, backend, "photometric.error_map", error, kReduce);
        Tensor from_map = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::GPU);
        table->map_to_error(saved.ssim_map, from_map);
        keep(out.snapshot, backend, "photometric.map_to_error", from_map, kReduce);
        const ops::PhotoWorkspaceBytes bytes = table->workspace_bytes(saved);
        out.snapshot.exact_u("photometric.workspace.required", bytes.required);
        out.snapshot.exact_u("photometric.workspace.allocated", bytes.allocated);
        out.snapshot.exact_u("photometric.workspace.error_map", bytes.error_map);
        table->shrink_to_required(saved);
        const ops::PhotoWorkspaceBytes shrunk = table->workspace_bytes(saved);
        out.snapshot.exact_u("photometric.workspace.shrunk", shrunk.allocated);
        table->reset(saved);
        const ops::PhotoWorkspaceBytes reset = table->workspace_bytes(saved);
        out.snapshot.exact_u("photometric.workspace.reset", reset.allocated);
        return out;
    }

    Capture capture_adam(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).adam;
        constexpr size_t n = 700;
        constexpr ops::AdamModifiers modifiers{
            .frozen_lr_scale = 0.25f,
            .cropbox_lr_scale = 0.5f,
            .median_extent = 1.5f,
            .r_min = 1.f,
            .r_max = 300.f,
            .screen_share_limit = 0.3f,
            .screen_share_penalty = 0.05f,
        };
        constexpr ops::AdamHyper hyper{.beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-15f};
        const Tensor frozen = bool_mask(n, 5);
        const Tensor crop = bool_mask(n, 7);
        const Tensor raw_scales = pattern({n, 3}, 2.f, 17);
        const Tensor far = bool_mask(n, 3);
        const Tensor share = pattern({n}, 0.4f, 23).abs();
        const ops::AdamMasks masks{frozen, crop, raw_scales, far, share};
        table->validate_far_mask(far.ptr<bool>());
        out.snapshot.exact_i("adam.validate_far_mask", 1);

        struct Group {
            Tensor parameter, packed, bounds, gradient;
        };
        auto make_group = [](size_t count, size_t attrs, int bits, int seed) {
            Group group;
            group.parameter = pattern({count, attrs}, 0.5f, seed);
            group.packed = Tensor::zeros(
                {count, attrs * static_cast<size_t>(joint::bytes_per_cell(bits))}, Device::GPU, DataType::UInt8);
            group.bounds = Tensor::zeros({joint::n_bounds_for_prims(count), size_t{4}}, Device::GPU);
            group.gradient = pattern({count, attrs}, 1e-3f, seed + 1);
            return group;
        };
        constexpr std::array<size_t, 5> attrs{3, 3, 3, 4, 1};
        std::vector<Group> groups;
        for (size_t i = 0; i < attrs.size(); ++i) {
            groups.push_back(make_group(n - 40 * i, attrs[i], 16, static_cast<int>(10 * i)));
        }
        std::vector<ops::JointStep> steps;
        for (size_t i = 0; i < groups.size(); ++i) {
            Group& group = groups[i];
            steps.push_back({
                .parameter = group.parameter,
                .packed = group.packed,
                .bounds = group.bounds,
                .gradient = group.gradient,
                .primitives = static_cast<int>(group.parameter.shape()[0]),
                .attributes = static_cast<int>(attrs[i]),
                .bits = 16,
                .lr = 0.01f * static_cast<float>(i + 1),
                .bc1_rcp = bc1(1),
                .bc2_sqrt_rcp = bc2(1),
                .apply_mean_step = i == 0,
                .apply_screen_share = i == 2,
            });
        }
        table->step_batch(steps, masks, hyper, modifiers);
        for (size_t i = 0; i < groups.size(); ++i) {
            const std::string prefix = "adam.step_batch." + std::to_string(i);
            keep(out.snapshot, backend, prefix + ".parameter", groups[i].parameter, kAdam);
            keep(out.snapshot, backend, prefix + ".packed", groups[i].packed, kExact);
            keep(out.snapshot, backend, prefix + ".bounds", groups[i].bounds, kAdam);
        }

        constexpr uint32_t rest = 15;
        const size_t floats = lfs::core::sh_swizzled_float_count(n, rest);
        Tensor sh_parameter = pattern({floats}, 0.2f, 43);
        Tensor sh_packed = Tensor::zeros({floats * static_cast<size_t>(joint::bytes_per_cell(8))}, Device::GPU, DataType::UInt8);
        Tensor sh_bounds = Tensor::zeros({joint::n_bounds_for_prims(n), size_t{4}}, Device::GPU);
        Tensor sh_gradient = pattern({floats}, 1e-3f, 41);
        Tensor absent;
        table->step_sh(sh_parameter, sh_packed, sh_bounds, absent, sh_gradient, masks, hyper, modifiers,
                       {.primitives = static_cast<int>(n),
                        .layout_slots = static_cast<int>(lfs::core::sh_float4_slots_for_rest(rest)),
                        .active_bases = 4,
                        .value_bits = 0,
                        .value_cells = 0,
                        .step_size = 0.005f * bc1(1),
                        .bc2_sqrt_rcp = bc2(1)});
        keep(out.snapshot, backend, "adam.step_sh.parameter", sh_parameter, kAdam);
        keep(out.snapshot, backend, "adam.step_sh.packed", sh_packed, kExact);
        keep(out.snapshot, backend, "adam.step_sh.bounds", sh_bounds, kAdam);

        const Tensor indices = i64_rows({0, 5, 5, 255, 256, 300, 511, 512, 699, 3});
        Group rows = make_group(n, 4, 16, 3);
        rows.packed.copy_from(pattern(rows.packed.shape(), 100.f, 63).abs().to(DataType::UInt8));
        rows.bounds.copy_from(pattern(rows.bounds.shape(), 0.5f, 64));
        table->encode_zero(rows.packed, rows.bounds, indices,
                           {.layout = ops::JointLayout::Rows, .primitives = static_cast<int>(n), .attributes_or_slots = 4, .bits = 16});
        keep(out.snapshot, backend, "adam.encode_zero.packed", rows.packed, kExact);
        keep(out.snapshot, backend, "adam.encode_zero.bounds", rows.bounds, kAdam);
        return out;
    }

    Capture capture_mcmc(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).mcmc;
        constexpr size_t n = 257;
        table->initialize(51);
        out.snapshot.exact_i("mcmc.initialize", 51);
        auto opacity = pattern_short({n}, 0.01f).clamp(0.01f, 0.99f);
        auto scales = pattern_short({n, 3}, 0.1f);
        auto ratios = Tensor::empty({n}, Device::CPU, DataType::Int32);
        for (size_t i = 0; i < n; ++i) {
            ratios.ptr<int32_t>()[i] = static_cast<int>(1 + i % 50);
        }
        ratios = ratios.gpu();
        auto new_opacity = Tensor::zeros({n}, Device::GPU);
        auto new_scales = Tensor::zeros({n, 3}, Device::GPU);
        table->relocate(opacity, scales, ratios, new_opacity, new_scales, 0.005f);
        keep(out.snapshot, backend, "mcmc.relocate.opacity", new_opacity, kExact);
        keep(out.snapshot, backend, "mcmc.relocate.scales", new_scales, kExact);

        auto raw_opacity = pattern_short({n}, -4.f);
        auto raw_scales = pattern_short({n, 3}, -2.f);
        auto quats = pattern_short({n, 4});
        auto means = pattern_short({n, 3});
        const Tensor frozen = bool_mask(n, 3);
        table->noise(raw_opacity, raw_scales, quats, frozen, means, 12345, 0.2f);
        keep(out.snapshot, backend, "mcmc.noise", means, kExact);

        const auto src = i64_rows({0, 3, 5});
        const auto dst = i64_rows({9, 11, 16});
        constexpr size_t rows = 17;
        auto row_means = pattern_short({rows, 3});
        auto sh0 = pattern_short({rows, 1, 3});
        auto row_scales = pattern_short({rows, 3}, -2.f);
        auto row_quats = pattern_short({rows, 4});
        auto row_opacity = pattern_short({rows});
        table->copy_rows(src, dst, {row_means, sh0, row_scales, row_quats, row_opacity});
        keep(out.snapshot, backend, "mcmc.copy.means", row_means, kExact);
        auto updated_scales = pattern_short({3, 3}, -5.f);
        auto updated_opacity = pattern_short({3}, -3.f);
        table->update_rows(dst, updated_scales, updated_opacity, row_scales, row_opacity);
        keep(out.snapshot, backend, "mcmc.update.scales", row_scales, kExact);
        keep(out.snapshot, backend, "mcmc.update.opacity", row_opacity, kExact);

        constexpr size_t count = 513;
        auto weights = pattern_short({n}, 0.01f);
        auto sample_opacity = pattern_short({n});
        auto sample_scales = pattern_short({n, 3}, -2.f);
        const auto alive = i64_rows({0, 4, 7, 31, 42, 128, 256});
        auto indices = Tensor::empty({count}, Device::GPU, DataType::Int64);
        auto sampled_opacity = Tensor::zeros({count}, Device::GPU);
        auto sampled_scales = Tensor::zeros({count, 3}, Device::GPU);
        table->sample(weights, sample_opacity, sample_scales, alive, indices, sampled_opacity, sampled_scales,
                      ops::SampleDomain::AliveIndices, 98765);
        keep(out.snapshot, backend, "mcmc.sample.indices", indices, kExact);
        keep(out.snapshot, backend, "mcmc.sample.opacity", sampled_opacity, kExact);
        keep(out.snapshot, backend, "mcmc.sample.scales", sampled_scales, kExact);

        auto error_max = pattern_short({n});
        auto densification = pattern_short({2, n}, 0.5f);
        table->fold_error(error_max, densification);
        keep(out.snapshot, backend, "mcmc.fold_error.max", error_max, kExact);
        keep(out.snapshot, backend, "mcmc.fold_error.rows", densification, kExact);
        return out;
    }

    Capture capture_refine(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).refine;
        constexpr size_t n = 257;
        auto means = pattern_short({n, 3});
        auto rotations = pattern_short({n, 4}, 0.1f);
        auto scales = pattern_short({n, 3}, -2.f);
        auto sh0 = pattern_short({n, 1, 3});
        auto opacity = pattern_short({n, 1}, -0.5f);
        auto child_means = pattern_short({3, 3}, 0.2f);
        auto child_rotations = pattern_short({3, 4}, 0.3f);
        auto child_scales = pattern_short({3, 3}, -1.f);
        auto child_sh0 = pattern_short({3, 1, 3}, 0.4f);
        auto child_opacity = pattern_short({3, 1}, -0.2f);
        const auto split_ids = i64_rows({0, 128, 256});
        table->split({means, rotations, scales, sh0, opacity},
                     {child_means, child_rotations, child_scales, child_sh0, child_opacity}, split_ids);
        keep(out.snapshot, backend, "refine.split.means", means, kExact);
        keep(out.snapshot, backend, "refine.split.children", child_means, kExact);
        const auto destinations = i64_rows({2, 127, 254});
        auto free = Tensor::ones({n}, Device::GPU, DataType::Bool);
        table->fill_slots(destinations, {child_means, child_rotations, child_scales, child_sh0, child_opacity},
                          {means, rotations, scales, sh0, opacity}, free);
        keep(out.snapshot, backend, "refine.fill.means", means, kExact);
        keep(out.snapshot, backend, "refine.fill.free", free, kExact);

        auto b0 = pattern_short({n}) > 0.5f;
        auto f0 = pattern_short({513}, -0.5f);
        auto counts = Tensor::full({4}, -1.f, Device::GPU, DataType::Int64);
        Tensor absent;
        table->counts(b0, absent, f0, absent, counts);
        keep(out.snapshot, backend, "refine.counts", counts, kExact);

        const float nan = std::numeric_limits<float>::quiet_NaN();
        auto median = Tensor::from_vector(std::vector<float>{nan, 0.f, -2.f, 9.f, 1.f, 4.f, 4.f, 100.f}, {8}, Device::GPU);
        table->normalize_positive_median(median);
        keep(out.snapshot, backend, "refine.median", median, kExact);

        auto shares = pattern_short({n});
        auto log_scales = pattern_short({n, 3}, -2.f);
        auto frozen = pattern_short({129}) > 0.5f;
        table->clip_scales(log_scales, shares, frozen, 0.3f);
        keep(out.snapshot, backend, "refine.clip", log_scales, kExact);
        auto error = pattern_short({n}, -0.2f);
        auto scores = Tensor::full({n}, -1.f, Device::GPU);
        table->oversize_scores(error, shares, frozen, scores, 0.3f);
        keep(out.snapshot, backend, "refine.oversize", scores, kReduce);

        auto dead_opacity = pattern_short({n});
        auto dead_rotations = pattern_short({n, 4});
        dead_rotations.slice(0, 0, 1).zero_();
        auto dead = Tensor::zeros({n}, Device::GPU, DataType::Bool);
        auto rotation = dead.clone();
        table->dead_mask(dead_opacity, dead_rotations, dead, 0.2f);
        table->rotation_mask(dead_rotations, rotation);
        keep(out.snapshot, backend, "refine.dead", dead, kExact);
        keep(out.snapshot, backend, "refine.rotation", rotation, kExact);
        return out;
    }

    Capture capture_masks(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).masks;
        constexpr size_t h = 33;
        constexpr size_t w = 35;
        auto cpu = Tensor::empty({h, w}, Device::CPU, DataType::UInt8);
        for (size_t i = 0; i < h * w; ++i) {
            cpu.ptr<uint8_t>()[i] = static_cast<uint8_t>((i * 79 + 31) % 256);
        }
        const auto u8 = cpu.gpu();
        const auto mask = u8.to(DataType::Float32) / 255.f;
        const auto alpha = Tensor::full({h, w}, 0.43f, Device::GPU);
        const auto roi = Tensor::full({h, w}, 0.37f, Device::GPU);
        auto weight = Tensor::full({h, w}, -9.f, Device::GPU);
        table->photometric_weight(mask, roi, weight, ops::MaskPhotoMode::SegmentAndIgnore);
        keep(out.snapshot, backend, "masks.photometric", weight, kExact);
        auto grad = Tensor::full({h, w}, -9.f, Device::GPU);
        auto temp = Tensor::zeros({1024}, Device::GPU);
        auto loss = Tensor::full({1}, -9.f, Device::GPU);
        table->opacity_penalty(alpha, mask, roi, grad, temp, loss, ops::MaskOpacityMode::SegmentAndIgnore, 1.7f, 0.13f);
        keep(out.snapshot, backend, "masks.opacity.grad", grad, kReduce);
        keep(out.snapshot, backend, "masks.opacity.loss", loss, kReduce);
        table->alpha_consistency(alpha, mask, roi, grad, temp, loss, 10.f);
        keep(out.snapshot, backend, "masks.alpha.grad", grad, kReduce);
        keep(out.snapshot, backend, "masks.alpha.loss", loss, kReduce);
        return out;
    }

    Capture capture_extra(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).extra_loss;
        constexpr size_t n = 1027;
        auto values = [](size_t count, size_t attrs) {
            std::vector<float> data(count * attrs);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<float>(static_cast<int>((i * 79 + 31) % 257) - 128) / 64.f;
            }
            return data;
        };
        for (const auto kind : {ops::Regularizer::Scale, ops::Regularizer::Opacity}) {
            const size_t attrs = kind == ops::Regularizer::Scale ? 3 : 1;
            const auto raw = Tensor::from_vector(values(n, attrs), {n, attrs}, Device::GPU);
            auto gradient = Tensor::full({n, attrs}, 0.17f, Device::GPU);
            auto loss = Tensor::full({1}, -9.f, Device::GPU);
            auto temp = Tensor::zeros({1024}, Device::GPU);
            table->regularize(raw, gradient, loss, temp, kind, 0.013f);
            const char* name = kind == ops::Regularizer::Scale ? "scale" : "opacity";
            keep(out.snapshot, backend, std::string("extra.") + name + ".grad", gradient, kReduce);
            keep(out.snapshot, backend, std::string("extra.") + name + ".loss", loss, kReduce);
        }
        std::vector<float> unit(n);
        for (size_t i = 0; i < n; ++i) {
            unit[i] = static_cast<float>((i * 79 + 31) % 257) / 256.f;
        }
        const auto sigmoid = Tensor::from_vector(unit, {n, 1}, Device::GPU);
        const auto z = sigmoid * 0.7f;
        const auto u = sigmoid * 0.03f;
        auto admm_grad = Tensor::full({n, 1}, 0.17f, Device::GPU);
        table->admm(sigmoid, z, u, admm_grad, 0.03f, 0.7f, true);
        keep(out.snapshot, backend, "extra.admm", admm_grad, kReduce);
        return out;
    }

    Capture capture_geometry(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).geometry;
        constexpr size_t H = 40;
        constexpr size_t W = 40;
        constexpr size_t N = H * W;
        std::vector<float> depth(N), alpha(N, 0.875f), normal(3 * N), target(N), weight(N);
        for (size_t y = 0; y < H; ++y) {
            for (size_t x = 0; x < W; ++x) {
                const size_t i = y * W + x;
                depth[i] = alpha[i] * (2.f + static_cast<float>(x) / 1024.f);
                target[i] = 0.2f + static_cast<float>((i * 7919u + 42u) % 2003u) / 4000.f;
                normal[i] = 0.125f;
                normal[N + i] = 0.25f;
                normal[2 * N + i] = -0.875f;
                weight[i] = x % 3 == 1 && y % 3 == 1 ? 1.f : 0.f;
            }
        }
        const auto depth_t = Tensor::from_vector(depth, {H, W}, Device::GPU);
        const auto alpha_t = Tensor::from_vector(alpha, {H, W}, Device::GPU);
        const auto normal_t = Tensor::from_vector(normal, {3, H, W}, Device::GPU);
        const auto target_t = Tensor::from_vector(target, {H, W}, Device::GPU);
        const auto weight_t = Tensor::from_vector(weight, {H, W}, Device::GPU);
        const ops::Intrinsics intr{40, 40, 20, 20};
        lfs::training::kernels::DepthAnchor anchor;
        anchor.valid = true;
        anchor.scale = 1.f;
        anchor.shift = 0.1f;
        anchor.floor = 0.01f;
        auto run = [&](const char* name, auto&& invoke, size_t partials_n, bool normal_grad, bool depth_grad) {
            auto gd = Tensor::zeros({H, W}, Device::GPU);
            auto ga = Tensor::zeros({H, W}, Device::GPU);
            auto gn = Tensor::zeros({3, H, W}, Device::GPU);
            auto loss = Tensor::zeros({1}, Device::GPU);
            auto partials = Tensor::zeros({partials_n}, Device::GPU);
            invoke(gd, ga, gn, loss, partials);
            if (depth_grad) {
                keep(out.snapshot, backend, std::string(name) + ".depth", gd, kReduce);
            }
            keep(out.snapshot, backend, std::string(name) + ".alpha", ga, kReduce);
            if (normal_grad) {
                keep(out.snapshot, backend, std::string(name) + ".normal", gn, kReduce);
            }
            keep(out.snapshot, backend, std::string(name) + ".loss", loss, kReduce);
            keep(out.snapshot, backend, std::string(name) + ".partials", partials, kExact);
        };
        namespace k = lfs::training::kernels;
        run("geometry.depth", [&](Tensor& gd, Tensor& ga, Tensor&, Tensor& loss, Tensor& partials) { table->depth(depth_t, alpha_t, target_t, weight_t, gd, ga, loss, partials, {0.5f, 0.25f, 0.f, &anchor}); }, k::depth_loss_partial_count(N), false, true);
        const auto normal_target = normal_t + 0.25f;
        run("geometry.normal", [&](Tensor&, Tensor&, Tensor& gn, Tensor& loss, Tensor& partials) { table->normal(normal_t, alpha_t, normal_target, weight_t, gn, loss, partials, 0.5f); }, k::normal_loss_partial_count(N), true, false);
        run("geometry.consistency", [&](Tensor& gd, Tensor& ga, Tensor& gn, Tensor& loss, Tensor& partials) { table->consistency(normal_t, depth_t, alpha_t, weight_t, gn, gd, ga, loss, partials, intr, 0.5f); }, k::normal_loss_partial_count(N), true, true);
        run("geometry.prior", [&](Tensor& gd, Tensor& ga, Tensor&, Tensor& loss, Tensor& partials) { table->prior_depth(normal_t, depth_t, alpha_t, weight_t, gd, ga, loss, partials, intr, 0.5f); }, k::normal_loss_partial_count(N), false, true);

        std::vector<float> xyz(1024 * 3);
        for (size_t i = 0; i < 1024; ++i) {
            xyz[3 * i] = (static_cast<float>(i % 32) - 16.f) / 40.f;
            xyz[3 * i + 1] = (static_cast<float>(i / 32) - 16.f) / 40.f;
            xyz[3 * i + 2] = 2.f;
        }
        const auto points = Tensor::from_vector(xyz, {1024, 3}, Device::GPU);
        const auto view = Tensor::from_vector(
            std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4}, Device::GPU);
        const ops::AnchorParams anchor_params{{40, 40, 20, 20}, 0.01f, {-10, -10, -10}, {10, 10, 10}};
        auto samples = table->collect_anchor_samples(points, view, target_t, anchor_params);
        std::sort(samples.begin(), samples.end(), [](const ops::AnchorSample& a, const ops::AnchorSample& b) {
            return std::tie(a.x, a.y) < std::tie(b.x, b.y);
        });
        out.snapshot.exact_i("geometry.anchors.count", static_cast<int64_t>(samples.size()));
        if (!samples.empty()) {
            out.snapshot.exact_bytes("geometry.anchors", samples.data(), samples.size() * sizeof(ops::AnchorSample));
        }
        return out;
    }

    Capture capture_mrnf(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).mrnf;
        constexpr size_t n = 257;
        constexpr uint64_t kSeed = 0x4d524e46ull;
        auto opacity = pattern_mrnf({n}, 1.f, 3) - 8.f;
        auto scales = pattern_mrnf({n, 3}, 1.f, 5);
        auto visibility = pattern_mrnf({n}, 1.f, 7).abs() + 1.f;
        auto means = pattern_mrnf({n, 3}, 0.5f, 1);
        const Tensor frozen = bool_mask(n, 4);
        table->noise(means, opacity, visibility, frozen, {.seed = kSeed, .lr_mean = 1.f, .noise_weight = 4.f, .median_scale = 1.f});
        keep(out.snapshot, backend, "mrnf.noise", means, kExact);
        auto raw = pattern_mrnf({n}, 1.5f, 9);
        auto log_scales = pattern_mrnf({n, 3}, 0.4f, 11);
        const Tensor far = bool_mask(n, 3);
        table->decay(raw, log_scales, frozen, far, {.opacity_decay = 0.02f, .scale_decay = 0.01f, .far_decay_scale = 0.25f, .train_t = 0.4f});
        keep(out.snapshot, backend, "mrnf.decay.opacity", raw, kExact);
        keep(out.snapshot, backend, "mrnf.decay.scales", log_scales, kExact);

        constexpr size_t bounds_n = 129;
        auto bound_means = pattern_mrnf({bounds_n, 3}, 3.f, 2);
        const ops::Bounds bounds = table->percentile_bounds(bound_means, 0.8f);
        keep_f(out.snapshot, "mrnf.bounds.cx", bounds.center[0], kExact);
        keep_f(out.snapshot, "mrnf.bounds.cy", bounds.center[1], kExact);
        keep_f(out.snapshot, "mrnf.bounds.cz", bounds.center[2], kExact);
        keep_f(out.snapshot, "mrnf.bounds.median", bounds.median_size, kExact);
        keep_f(out.snapshot, "mrnf.bounds.max", bounds.max_extent, kExact);
        auto extent_scales = pattern_mrnf({bounds_n, 3}, 0.8f, 4);
        const ops::ScalarValidity extent = table->median_extent(extent_scales);
        keep_f(out.snapshot, "mrnf.extent", extent.value, kExact);
        out.snapshot.exact_i("mrnf.extent.valid", extent.valid ? 1 : 0);

        auto weights = pattern_mrnf({n}, 1.f, 8).abs();
        auto host = weights.cpu();
        for (size_t i = 0; i < n; i += 5) {
            host.ptr<float>()[i] = 0.f;
        }
        weights = host.gpu();
        constexpr size_t k = 17;
        auto top = Tensor::empty({k}, Device::GPU, DataType::Int64);
        lfs::training::GumbelTopKScratch scratch;
        table->gumbel(&scratch, weights, top, {.seed = kSeed, .known_nnz = 0, .compact_sparse = true});
        keep(out.snapshot, backend, "mrnf.gumbel", top, kExact);

        constexpr size_t fold_n = 64;
        auto vis = pattern_mrnf({fold_n}, 1.f, 1).abs();
        auto weight = pattern_mrnf({fold_n}, 0.2f, 2);
        auto dens = pattern_mrnf({2, fold_n}, 0.5f, 3);
        auto ratio = pattern_mrnf({fold_n}, 0.1f, 4);
        table->fold(vis, weight, dens, ratio, 0.75f);
        keep(out.snapshot, backend, "mrnf.fold.dens", dens, kExact);
        auto max_error = pattern_mrnf({fold_n}, 0.3f, 6);
        auto err = pattern_mrnf({2, fold_n}, 0.8f, 7);
        table->fold_error(max_error, err);
        keep(out.snapshot, backend, "mrnf.fold_error", err, kExact);

        auto project_means = pattern_mrnf({fold_n, 3}, 1.f, 12);
        const auto w2c = Tensor::from_vector(
            std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1}, {4, 4}, Device::GPU);
        auto means2d = Tensor::zeros({fold_n, 2}, Device::GPU);
        auto radii = Tensor::zeros({fold_n}, Device::GPU);
        const ops::ProjectParams project{.image = {.h = 8, .w = 12}, .intrinsics = {.fx = 20.f, .fy = 18.f, .cx = 6.f, .cy = 4.f}, .near_plane = 0.01f};
        table->project_centers(project_means, w2c, means2d, radii, project);
        keep(out.snapshot, backend, "mrnf.project.means2d", means2d, kExact);
        keep(out.snapshot, backend, "mrnf.project.radii", radii, kExact);
        auto image_error = pattern_mrnf({8, 12}, 1.f, 15).abs();
        auto scores = Tensor::zeros({fold_n}, Device::GPU);
        table->gather_center_error(means2d, radii, image_error, scores);
        keep(out.snapshot, backend, "mrnf.center_error", scores, kReduce);
        auto far_mask = Tensor::zeros({fold_n}, Device::GPU, DataType::Bool);
        table->far_mask(project_means, far_mask, {0.1f, -0.2f, 0.3f}, 1.5f);
        keep(out.snapshot, backend, "mrnf.far", far_mask, kExact);

        constexpr int height = 6;
        constexpr int width = 8;
        constexpr size_t hw = static_cast<size_t>(height * width);
        auto predicted = pattern_mrnf({3, height, width}, 1.f, 1);
        auto target = pattern_mrnf({3, height, width}, 0.7f, 2);
        auto pixel_error = Tensor::zeros({height, width}, Device::GPU);
        table->mean_abs_error(predicted, target, pixel_error);
        keep(out.snapshot, backend, "mrnf.mae", pixel_error, kReduce);
        auto alpha = pattern_mrnf({hw}, 0.5f, 3).abs().clamp(0.f, 1.f);
        auto seed_weights = pixel_error.clone().reshape({hw});
        table->seed_weights(seed_weights, alpha, seed_weights);
        keep(out.snapshot, backend, "mrnf.seeds", seed_weights, kReduce);
        const auto pixel_indices = i64_rows({1, 4, 7, 20, 40});
        auto depth = pattern_mrnf({hw}, 2.f, 6).abs();
        auto rgb = Tensor::zeros({5, 3}, Device::GPU);
        auto out_alpha = Tensor::zeros({5}, Device::GPU);
        auto out_depth = Tensor::zeros({5}, Device::GPU);
        table->gather_seeds(pixel_indices, target, alpha, depth, rgb, out_alpha, out_depth);
        keep(out.snapshot, backend, "mrnf.gather.rgb", rgb, kExact);
        keep(out.snapshot, backend, "mrnf.gather.depth", out_depth, kExact);
        auto median_values = pattern_mrnf({33}, 3.f, 9).abs();
        const float median = table->sorted_median(median_values);
        keep_f(out.snapshot, "mrnf.median", median, kExact);
        auto starved = pattern_mrnf({33}, 1.f, 10).abs();
        auto vis_starve = pattern_mrnf({33}, 2.f, 11).abs();
        table->starvation_weights(starved, vis_starve, median);
        keep(out.snapshot, backend, "mrnf.starvation", starved, kExact);

        auto prune = bool_mask(fold_n, 3);
        auto compact = Tensor::empty({fold_n}, Device::GPU, DataType::Int64);
        const size_t compacted = table->compact_bool_indices(prune, compact, fold_n);
        out.snapshot.exact_u("mrnf.compact.count", compacted);
        keep(out.snapshot, backend, "mrnf.compact.indices", compact, kExact);
        auto scale_max = pattern_mrnf({fold_n}, 0.5f, 4);
        auto prune_mask = Tensor::zeros({fold_n}, Device::GPU, DataType::Bool);
        table->prune_bounds(project_means, scale_max, prune_mask, {0.f, 0.f, 0.f}, 2.f, 0.5f);
        keep(out.snapshot, backend, "mrnf.prune", prune_mask, kExact);
        auto parent = Tensor::zeros({fold_n}, Device::GPU);
        auto active = bool_mask(fold_n, 2);
        auto trainable = bool_mask(fold_n, 5);
        auto edge = pattern_mrnf({fold_n}, 0.2f, 6).abs();
        table->replace_parent_weights(opacity.slice(0, 0, fold_n), visibility.slice(0, 0, fold_n), active, trainable, edge, parent);
        keep(out.snapshot, backend, "mrnf.replace", parent, kReduce);
        return out;
    }

    Capture capture_bilateral(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).bilateral;
        auto grid = pattern({1, 12, 2, 2, 2}, 1.f, 3);
        auto rgb = pattern({5, 7, 3}, 1.f, 11);
        auto offset = pattern({12}, 1.f, 17);
        auto output = Tensor::zeros(rgb.shape(), Device::GPU);
        const ops::GridSliceParams params{ops::Layout::HWC, ops::GridTransform::Affine, true};
        table->slice_forward(grid, rgb, offset, output, params);
        keep(out.snapshot, backend, "bilateral.forward", output, kConv);
        std::vector<float> gradient(rgb.numel(), 0.f);
        for (size_t c = 0; c < 3; ++c) {
            gradient[17 * 3 + c] = static_cast<float>(c + 1) / 8.f;
        }
        auto grad_output = Tensor::from_vector(gradient, rgb.shape(), Device::GPU);
        auto grad_grid = Tensor::zeros(grid.shape(), Device::GPU);
        auto grad_rgb = Tensor::zeros(rgb.shape(), Device::GPU);
        table->slice_backward(grid, rgb, grad_output, offset, grad_grid, grad_rgb, params);
        keep(out.snapshot, backend, "bilateral.backward.grid", grad_grid, kConv);
        keep(out.snapshot, backend, "bilateral.backward.rgb", grad_rgb, kConv);

        auto tv_grid = pattern({2, 9, 3, 4, 5}, 1.f, 5);
        auto loss = Tensor::zeros({1}, Device::GPU);
        auto temp = Tensor::empty({2048}, Device::GPU);
        table->tv_forward(tv_grid, loss, temp, 7);
        keep(out.snapshot, backend, "bilateral.tv.loss", loss, kReduce);
        auto tv_grad = pattern(tv_grid.shape(), 1.f, 11);
        table->tv_backward(tv_grid, tv_grad, 0.25f, 7);
        keep(out.snapshot, backend, "bilateral.tv.grad", tv_grad, kConv);

        auto project = pattern({2, 12, 2, 3, 4}, 1.f, 2);
        auto mean = pattern({12}, 1.f, 11);
        auto identity = pattern({12}, 1.f, 19);
        table->project_mean(project, mean, identity, 0);
        keep(out.snapshot, backend, "bilateral.project", project, kConv);
        auto sum = pattern({12}, 1.f, 1);
        auto shared = pattern({12}, 1.f, 11);
        auto old_mean = pattern({12}, 1.f, 23);
        auto new_mean = pattern({12}, 1.f, 29);
        table->update_offset(sum, shared, identity, old_mean, new_mean, 24.f, 1.f / 48.f);
        keep(out.snapshot, backend, "bilateral.offset", shared, kConv);

        auto adam_grid = pattern({1, 12, 2, 3, 4}, 1.f, 4);
        auto m1 = pattern(adam_grid.shape(), 1.f, 11);
        auto m2 = pattern(adam_grid.shape(), 1.f, 17);
        auto adam_grad = pattern(adam_grid.shape(), 1.f, 19);
        const ops::AdamUpdateParams adam{0.002f, 0.9f, 0.999f, 10.f, 31.622776f, 1e-15f};
        table->adam(adam_grid, m1, m2, adam_grad, adam);
        table->scale_moments(m1, m2, 0.81f, 0.998001f);
        keep(out.snapshot, backend, "bilateral.adam.grid", adam_grid, kAdam);
        keep(out.snapshot, backend, "bilateral.adam.m1", m1, kAdam);

        auto host = Tensor::from_vector(pattern_values(8, 0.5f, 7), {8}, Device::CPU);
        auto device = Tensor::zeros({8}, Device::GPU);
        table->upload_slice(host, device, 0, 0, 8);
        auto downloaded = Tensor::zeros({8}, Device::CPU);
        table->download_slice(downloaded, device, 0, 0, 8);
        if (backend == GpuBackend::CUDA) {
            cudaDeviceSynchronize();
        }
        keep(out.snapshot, backend, "bilateral.upload", device, kExact);
        keep(out.snapshot, backend, "bilateral.download", downloaded, kExact);
        return out;
    }

    Capture capture_ppisp(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).ppisp;
        struct Params {
            Tensor exposure = pattern({3}, 1.f, 1);
            Tensor vignetting = pattern({30}, 1.f, 2);
            Tensor color = pattern({24}, 1.f, 3);
            Tensor crf = pattern({24}, 1.f, 4);
            ops::PPISPInputs inputs() const { return {exposure, vignetting, color, crf}; }
            ops::PPISPOutputs outputs() { return {exposure, vignetting, color, crf}; }
        };
        Params identity;
        table->initialize(identity.outputs());
        keep(out.snapshot, backend, "ppisp.init.exposure", identity.exposure, kExact);
        keep(out.snapshot, backend, "ppisp.init.crf", identity.crf, kExact);
        Params values;
        const auto rgb = pattern({3, 2, 7}, 1.f, 7).add(0.5f);
        auto corrected = Tensor::empty_like(rgb);
        table->forward(values.inputs(), rgb, corrected, {0, 8, 2, 3, 1, 2});
        keep(out.snapshot, backend, "ppisp.forward", corrected, kConv);
        auto pixel = pattern({3, 1, 1}, 1.f, 7).add(0.5f);
        auto grad = pattern({3, 1, 1}, 1.f, 9);
        Params grads;
        auto grad_rgb = Tensor::empty_like(pixel);
        table->backward(values.inputs(), pixel, grad, grads.outputs(), grad_rgb, 2, 3, 1, 2);
        keep(out.snapshot, backend, "ppisp.backward.rgb", grad_rgb, kConv);
        keep(out.snapshot, backend, "ppisp.backward.exposure", grads.exposure, kConv);

        const ops::PPISPAdamUpdateParams hyper{0.003f, 0.9f, 0.999f, 10.f, 31.622776f, 1e-8f};
        Tensor parameter = pattern({257}, 1.f, 1);
        Tensor moment1 = pattern({257}, 1.f, 2);
        Tensor moment2 = pattern({257}, 1.f, 3).abs().add(0.1f);
        Tensor step_grad = pattern({257}, 1.f, 4);
        table->adam({parameter, moment1, moment2, step_grad}, hyper);
        keep(out.snapshot, backend, "ppisp.adam", parameter, kAdam);
        std::array<Tensor, 4> p{pattern({3}, 1.f, 1), pattern({30}, 1.f, 2), pattern({24}, 1.f, 3), pattern({24}, 1.f, 4)};
        std::array<Tensor, 4> m{pattern({3}, 1.f, 5), pattern({30}, 1.f, 6), pattern({24}, 1.f, 7), pattern({24}, 1.f, 8)};
        std::array<Tensor, 4> v{pattern({3}, 1.f, 9).abs().add(0.1f), pattern({30}, 1.f, 10).abs().add(0.1f),
                                pattern({24}, 1.f, 11).abs().add(0.1f), pattern({24}, 1.f, 12).abs().add(0.1f)};
        std::array<Tensor, 4> g{pattern({3}, 1.f, 13), pattern({30}, 1.f, 14), pattern({24}, 1.f, 15), pattern({24}, 1.f, 16)};
        table->adam_batch({ops::PPISPAdamGroup{p[0], m[0], v[0], g[0]}, ops::PPISPAdamGroup{p[1], m[1], v[1], g[1]},
                           ops::PPISPAdamGroup{p[2], m[2], v[2], g[2]}, ops::PPISPAdamGroup{p[3], m[3], v[3], g[3]}},
                          hyper);
        keep(out.snapshot, backend, "ppisp.adam_batch.exposure", p[0], kAdam);
        auto vignette = pattern({15}, 1.f, 8);
        auto vignette_grad = pattern({15}, 1.f, 3);
        auto vignette_loss = Tensor::zeros({1}, Device::GPU);
        table->vignetting_regularization(vignette, vignette_grad, vignette_loss, 0.01f, 0.02f, 0.03f);
        keep(out.snapshot, backend, "ppisp.vignette.grad", vignette_grad, kConv);
        keep(out.snapshot, backend, "ppisp.vignette.loss", vignette_loss, kReduce);
        Params projected;
        table->project_mean(projected.exposure, projected.color);
        keep(out.snapshot, backend, "ppisp.project.exposure", projected.exposure, kExact);
        keep(out.snapshot, backend, "ppisp.project.color", projected.color, kExact);
        return out;
    }

    Capture capture_controller(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).controller;
        auto fc = Tensor::zeros({1, 1601}, Device::GPU);
        auto features = pattern({1, 1600}, 1.f, 2);
        table->prepare_input(features, fc, 0.7f);
        keep(out.snapshot, backend, "controller.prepare", fc, kExact);
        constexpr int m = 9;
        constexpr int n = 128;
        auto grad = pattern({1, m}, 1.f, 3);
        auto activation = pattern({1, n}, 1.f, 2);
        auto weight = pattern({m, n}, 1.f, 3);
        auto weight_grad = pattern({m, n}, 1.f, 4);
        auto bias_grad = pattern({m}, 1.f, 5);
        auto input_grad = Tensor::empty({1, n}, Device::GPU);
        table->backward_layer(grad, activation, weight, weight_grad, bias_grad, input_grad);
        keep(out.snapshot, backend, "controller.weight_grad", weight_grad, kConv);
        keep(out.snapshot, backend, "controller.bias_grad", bias_grad, kConv);
        keep(out.snapshot, backend, "controller.input_grad", input_grad, kConv);
        return out;
    }

    Capture capture_training_image(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).training_image;
        auto loss = Tensor::full({1}, 0.375f, Device::GPU);
        auto latest = Tensor::full({7}, -1.f, Device::GPU);
        auto ema = latest.clone();
        table->heatmap(loss, latest, ema, 0, 0.2f);
        loss.mul_(1.5f);
        table->heatmap(loss, latest, ema, 6, 0.2f);
        keep(out.snapshot, backend, "image.heatmap.latest", latest, kExact);
        keep(out.snapshot, backend, "image.heatmap.ema", ema, kExact);

        const auto view = Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4}, Device::GPU);
        const auto position = Tensor::zeros({3}, Device::GPU);
        ops::RoiParams roi{
            .image = {7, 9},
            .intrinsics = {6, 6, 4.5f, 3.5f},
            .world_to_cropbox = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.2f, 0.3f, 0, 1},
            .minimum = {-1, -1, 2},
            .maximum = {1, 1, 4},
            .outside_weight = 0.2f,
            .inverse = false};
        auto weights = Tensor::full({7, 9}, -1.f, Device::GPU);
        table->roi(view, position, weights, roi);
        keep(out.snapshot, backend, "image.roi", weights, kReduce);

        const auto source = pattern({3, 13, 19}, 1.f, 4);
        auto resized = Tensor::full({3, 7, 31}, -1.f, Device::GPU);
        table->resize_background(source, resized);
        keep(out.snapshot, backend, "image.resize", resized, kConv);
        auto random = Tensor::full({3, 17, 33}, -1.f, Device::GPU);
        table->random_background(random, 42);
        keep(out.snapshot, backend, "image.random", random, kExact);
        const auto canny_src = pattern({3, 35, 37}, 1.f, 6);
        auto edges = Tensor::full({35, 37}, -1.f, Device::GPU);
        table->canny(canny_src, edges);
        keep(out.snapshot, backend, "image.canny", edges, kExact);
        auto normalized = pattern({257}, 1.f, 8);
        auto scalar = Tensor::full({1}, 2.f, Device::GPU);
        table->normalize_scalar(normalized, scalar, 1e-6f);
        keep(out.snapshot, backend, "image.normalize", normalized, kReduce);
        return out;
    }

    Capture capture_shared_image(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).shared_image;
        constexpr size_t H = 13;
        constexpr size_t W = 17;
        auto filled = Tensor::empty({1031}, Device::GPU, DataType::UInt8);
        auto unchanged = Tensor::full({1}, 1.f, Device::GPU, DataType::UInt32);
        table->sentinel_fill(filled, 0x932abe71u);
        table->sentinel_check(filled, unchanged, 0x932abe71u);
        keep(out.snapshot, backend, "shared.sentinel", filled, kExact);
        keep(out.snapshot, backend, "shared.sentinel_flag", unchanged, kExact);

        const ops::NormalPriorTransform transform{.srgb = true, .flip_yz = true, .world_to_camera = true, .w2c = {0, -1, 0, 1, 0, 0, 0, 0, 1}};
        struct Convert {
            const char* name;
            ops::ImageConversion kind;
            lfs::core::TensorShape source_shape;
            DataType source_type;
            lfs::core::TensorShape dest_shape;
            DataType dest_type;
            bool prior;
        };
        const Convert converts[] = {
            {"u8_chw", ops::ImageConversion::U8HWCToF32CHW, {H, W, 3}, DataType::UInt8, {H, W, 3}, DataType::Float32, false},
            {"u16_chw", ops::ImageConversion::U16HWCToF32CHW, {H, W, 6}, DataType::UInt8, {H, W, 3}, DataType::Float32, false},
            {"f32_u16", ops::ImageConversion::F32HWCToU16HWC, {H, W, 3}, DataType::Float32, {H, W, 6}, DataType::UInt8, false},
            {"u16_hwc", ops::ImageConversion::U16HWCToF32HWC, {H, W, 6}, DataType::UInt8, {H, W, 3}, DataType::Float32, false},
            {"normal_j2k", ops::ImageConversion::NormalCHWToJ2KHWC, {H, W, 3}, DataType::Float32, {H, W, 3}, DataType::Float32, false},
            {"j2k_normal", ops::ImageConversion::J2KHWCToNormalCHW, {H, W, 3}, DataType::Float32, {H, W, 3}, DataType::Float32, false},
            {"prior_u8", ops::ImageConversion::NormalPriorU8, {H, W, 3}, DataType::UInt8, {H, W, 3}, DataType::Float32, true},
            {"prior_u16", ops::ImageConversion::NormalPriorU16, {H, W, 6}, DataType::UInt8, {H, W, 3}, DataType::Float32, true},
            {"u8_u8", ops::ImageConversion::U8HWCToU8CHW, {H, W, 3}, DataType::UInt8, {H, W, 3}, DataType::UInt8, false},
            {"u16_u8", ops::ImageConversion::U16HWCToU8CHW, {H, W, 6}, DataType::UInt8, {H, W, 3}, DataType::UInt8, false},
            {"f32_u8", ops::ImageConversion::F32CHWToU8CHW, {3, H, W}, DataType::Float32, {H, W, 3}, DataType::UInt8, false},
            {"u8_f32", ops::ImageConversion::U8HWToF32HW, {H, W}, DataType::UInt8, {H, W}, DataType::Float32, false},
        };
        for (const Convert& item : converts) {
            Tensor source = item.source_type == DataType::UInt8 ? image_bytes(item.source_shape) : image_floats(item.source_shape);
            Tensor destination = item.dest_type == DataType::UInt8 ? image_bytes(item.dest_shape) : image_floats(item.dest_shape);
            table->convert(source, destination, item.kind, H, W, item.source_shape.rank() == 2 ? 1 : 3, item.prior ? transform : ops::NormalPriorTransform{});
            keep(out.snapshot, backend, std::string("shared.convert.") + item.name, destination, item.dest_type == DataType::Float32 ? kConv : kExact);
        }

        auto rgba = image_bytes({H, W, 4});
        auto rgb = Tensor::empty({3, H, W}, Device::GPU);
        auto alpha = Tensor::empty({H, W}, Device::GPU);
        table->rgba_split(rgba, rgb, alpha);
        keep(out.snapshot, backend, "shared.rgba.rgb", rgb, kExact);
        keep(out.snapshot, backend, "shared.rgba.alpha", alpha, kExact);
        auto mask = image_floats({H, W});
        table->mask(mask, ops::MaskTransform::Threshold, 0.61f);
        keep(out.snapshot, backend, "shared.mask", mask, kExact);
        auto resized = table->resize(image_bytes({H, W, 3}), 7, 9, ops::Resample::LanczosRGB, 2);
        keep(out.snapshot, backend, "shared.resize", resized, kConv);
        lfs::core::UndistortParams undistort{};
        undistort.src_fx = undistort.dst_fx = 15.f;
        undistort.src_fy = undistort.dst_fy = 14.f;
        undistort.src_cx = undistort.dst_cx = 8.f;
        undistort.src_cy = undistort.dst_cy = 6.f;
        undistort.src_width = static_cast<int>(W);
        undistort.src_height = static_cast<int>(H);
        undistort.dst_width = 15;
        undistort.dst_height = 11;
        undistort.model_type = lfs::core::CameraModelType::PINHOLE;
        undistort.num_distortion = 4;
        undistort.distortion[0] = 0.08f;
        undistort.distortion[1] = -0.03f;
        undistort.distortion[2] = 0.001f;
        undistort.distortion[3] = -0.002f;
        auto undistorted = table->undistort(image_floats({3, H, W}), undistort, false);
        keep(out.snapshot, backend, "shared.undistort", undistorted, kConv);
        return out;
    }

    Capture capture_lpips(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).lpips;
        auto weight = pattern({128, 64, 3, 3}, 1.f, 7);
        auto taps = Tensor::empty({9, 128, 64}, Device::GPU, DataType::Float16);
        table->weight_taps(weight, taps);
        keep(out.snapshot, backend, "lpips.taps", taps, kConv);
        auto rgb = pattern({1, 3, 7, 9}, 1.f, 1);
        auto conv_w = pattern({64, 3, 3, 3}, 1.f, 2);
        auto bias = pattern({64}, 1.f, 3);
        auto conv = Tensor::empty({1, 64, 7, 9}, Device::GPU, DataType::Float16);
        const ops::RGBConvParams rgb_params{{-.030f, -.088f, -.188f}, {.458f, .448f, .450f}, true};
        table->rgb_conv(rgb, conv_w, bias, conv, rgb_params);
        keep(out.snapshot, backend, "lpips.rgb", conv, kConv);
        auto x = pattern({1, 64, 7, 9}, 1.f, 1);
        auto w = pattern({64, 64, 3, 3}, 1.f, 2);
        auto b = pattern({64}, 1.f, 3);
        auto y = Tensor::empty({1, 64, 7, 9}, Device::GPU, DataType::Float16);
        auto scratch = Tensor::empty({9, 64, 64}, Device::GPU, DataType::Float16);
        Tensor absent;
        ops::ConvParams conv_params;
        conv_params.pad_h = conv_params.pad_w = 1;
        conv_params.activation = lfs::core::nn::Activation::Relu;
        table->convolution(x, w, absent, b, y, scratch, conv_params);
        keep(out.snapshot, backend, "lpips.conv", y, kConv);
        auto px = pattern({1, 64, 2, 8}, 1.f, 1);
        auto py = pattern({1, 64, 2, 8}, 1.f, 2);
        auto pw = pattern({1, 64, 1, 1}, 1.f, 3);
        auto score = Tensor::zeros({1}, Device::GPU);
        auto pooled_x = Tensor::empty({1, 64, 1, 4}, Device::GPU, DataType::Float16);
        auto pooled_y = Tensor::empty(pooled_x.shape(), Device::GPU, pooled_x.dtype());
        table->pool_reduce(px, py, pw, score, pooled_x, pooled_y, {0, 2, 1, 7, 1.f / 12.f});
        keep(out.snapshot, backend, "lpips.pool", score, kConv);
        keep(out.snapshot, backend, "lpips.pool_x", pooled_x, kConv);
        return out;
    }

    Capture capture_sh(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).sh;
        constexpr size_t kN = 8;
        constexpr uint32_t kRest = 3;
        const auto indices = i64_rows({1, 5, 2});
        const auto indices_i32 = i32_rows({1, 5, 2});
        const auto src = pattern({lfs::core::sh_swizzled_float_count(kN, kRest)}, 0.25f, 3);
        auto codes = Tensor::zeros({quant::sh_value_u16_count(kN, kRest)}, Device::GPU, DataType::Float16);
        auto bounds = Tensor::zeros({quant::n_bounds_for_prims(kN) * 2}, Device::GPU);
        table->encode_q16(src, codes, bounds, kN, kRest, 0, 0);
        keep(out.snapshot, backend, "sh.encode.codes", codes, kExact);
        keep(out.snapshot, backend, "sh.encode.bounds", bounds, kAdam);
        auto decoded = Tensor::zeros({lfs::core::sh_swizzled_float_count(kN, kRest)}, Device::GPU);
        table->decode_q16(codes, bounds, decoded, kN, kRest);
        keep(out.snapshot, backend, "sh.decode", decoded, kAdam);

        const auto dest = i64_rows({0, 256, 257, 512, 256});
        auto ids = Tensor::zeros({dest.numel()}, Device::GPU);
        table->block_ids(dest, ids);
        keep(out.snapshot, backend, "sh.block_ids", ids, kExact);
        auto sorted = ids.sort(0, false).first.contiguous();
        const size_t runs_n = sorted.numel();
        auto unique = Tensor::zeros({runs_n}, Device::GPU, DataType::Int32);
        auto offsets = Tensor::zeros({runs_n}, Device::GPU, DataType::Int32);
        auto runs = Tensor::zeros({size_t{1}}, Device::GPU, DataType::Int32);
        auto state = table->create_run_scratch();
        if (!state) {
            out.error = "sh create_run_scratch returned null";
            return out;
        }
        table->block_runs(*state, sorted, unique, offsets, runs);
        keep(out.snapshot, backend, "sh.block_runs.count", runs, kExact);
        keep(out.snapshot, backend, "sh.block_runs.ids", unique, kExact);

        auto canonical = pattern({indices.numel(), size_t{kRest}, size_t{3}}, 0.8f, 21);
        auto touch_ids = Tensor::zeros({indices.numel()}, Device::GPU);
        table->block_ids(indices, touch_ids);
        auto ordered = touch_ids.sort(0, false);
        Tensor order = std::move(ordered.second);
        Tensor sorted_ids = ordered.first.contiguous();
        Tensor sorted_dest = indices.index_select(0, order).contiguous();
        Tensor sorted_can = canonical.index_select(0, order).contiguous();
        auto touch_unique = Tensor::zeros({indices.numel()}, Device::GPU, DataType::Int32);
        auto touch_offsets = Tensor::zeros({indices.numel()}, Device::GPU, DataType::Int32);
        auto touch_runs = Tensor::zeros({size_t{1}}, Device::GPU, DataType::Int32);
        auto touch_state = table->create_run_scratch();
        table->block_runs(*touch_state, sorted_ids, touch_unique, touch_offsets, touch_runs);
        auto touched_codes = codes.clone();
        auto touched_bounds = bounds.clone();
        const ops::Q16TouchParams touch{.sorted_count = indices.numel(), .primitives = kN, .decode_source_rows = kN, .rest = kRest};
        Tensor no_order;
        table->reencode_touched(touched_codes, touched_bounds, sorted_can, sorted_dest, touch_unique, touch_offsets, touch_runs, no_order, touch);
        keep(out.snapshot, backend, "sh.reencode.codes", touched_codes, kExact);

        constexpr uint64_t kCount = 6;
        auto range = Tensor::empty({size_t{kCount}}, Device::GPU);
        const ops::ShRangeParams q16_range{.canonical_float_offset = 3, .float_count = kCount, .primitives = kN, .destination_rest = kRest, .layout_rest = kRest, .storage = ops::ShStorage::Q16};
        table->decode_range(codes, bounds, range, q16_range);
        keep(out.snapshot, backend, "sh.range.q16", range, kAdam);
        auto f32_range_params = q16_range;
        f32_range_params.storage = ops::ShStorage::Float32;
        auto f32_range = Tensor::empty({size_t{kCount}}, Device::GPU);
        table->decode_range(src, Tensor{}, f32_range, f32_range_params);
        keep(out.snapshot, backend, "sh.range.f32", f32_range, kExact);
        auto half = src.to(DataType::Float16);
        auto f16_range_params = q16_range;
        f16_range_params.storage = ops::ShStorage::IeeeFloat16;
        auto f16_range = Tensor::empty({size_t{kCount}}, Device::GPU);
        table->decode_range(half, Tensor{}, f16_range, f16_range_params);
        keep(out.snapshot, backend, "sh.range.f16", f16_range, kAdam);

        const ops::ShRowsParams row_params{.source_rows = kN, .count = indices.numel(), .destination_offset = 0, .source_rest = kRest, .destination_rest = kRest};
        auto zeroed = src.clone();
        table->zero_rows(zeroed, indices_i32, kRest);
        keep(out.snapshot, backend, "sh.zero", zeroed, kExact);
        auto gathered = Tensor::zeros({lfs::core::sh_swizzled_float_count(indices.numel(), kRest)}, Device::GPU);
        table->gather_swizzled(src, indices_i32, gathered, row_params);
        keep(out.snapshot, backend, "sh.gather_swizzled", gathered, kExact);
        auto canonical_out = Tensor::zeros({indices.numel(), size_t{kRest}, size_t{3}}, Device::GPU);
        table->gather_canonical(src, indices, canonical_out, row_params);
        keep(out.snapshot, backend, "sh.gather_canonical", canonical_out, kExact);
        auto appended = Tensor::zeros({lfs::core::sh_swizzled_float_count(kN, kRest)}, Device::GPU);
        const ops::ShRowsParams append_at{.source_rows = indices.numel(), .count = indices.numel(), .destination_offset = 2, .source_rest = kRest, .destination_rest = kRest};
        table->append_canonical(canonical_out, appended, append_at);
        keep(out.snapshot, backend, "sh.append", appended, kExact);
        auto scattered = src.clone();
        table->scatter_canonical(canonical_out, indices_i32, scattered, row_params);
        keep(out.snapshot, backend, "sh.scatter", scattered, kExact);

        constexpr size_t kCapacity = 8;
        auto storage = Tensor::zeros({kCapacity}, Device::GPU, DataType::UInt8);
        table->fill_bytes(storage, kCapacity, 0xA5);
        keep(out.snapshot, backend, "sh.fill", storage, kExact);
        auto part_a = pattern({4}, 0.2f, 1);
        auto part_b = pattern({4}, 0.3f, 2);
        auto arena = Tensor::empty({8}, Device::GPU);
        const std::array<Tensor, 2> parts{part_a, part_b};
        auto concatenated = table->concatenate_into_arena(parts, static_cast<char*>(arena.data_ptr()), {8}, DataType::Float32, lfs::core::TensorExecutionTarget::current());
        (void)concatenated;
        keep(out.snapshot, backend, "sh.concatenate", arena, kExact);
        return out;
    }

    Tensor shuffled_indices(size_t count) {
        auto cpu = Tensor::empty({count}, Device::CPU, DataType::Int64);
        std::iota(cpu.ptr<int64_t>(), cpu.ptr<int64_t>() + count, int64_t{0});
        std::mt19937 rng(731);
        std::shuffle(cpu.ptr<int64_t>(), cpu.ptr<int64_t>() + count, rng);
        return cpu.gpu();
    }

    lfs::core::SplatData make_degree_splat(size_t count, int degree) {
        const size_t rest = degree > 0 ? static_cast<size_t>(degree * (degree + 2)) : 0;
        auto means = pattern({count, 3}, 1.f, 1);
        auto sh0 = pattern({count, 1, 3}, 0.5f, 2);
        auto shN = rest == 0 ? Tensor::zeros({count, size_t{0}, 3}, Device::GPU)
                             : pattern({count, rest, 3}, 0.2f, 3);
        auto scaling = pattern({count, 3}, 1.f, 4);
        auto rotation = pattern({count, 4}, 1.f, 5);
        auto opacity = pattern({count, 1}, 1.f, 6);
        return lfs::core::SplatData(degree, std::move(means), std::move(sh0), std::move(shN), std::move(scaling),
                                    std::move(rotation), std::move(opacity), 1.f);
    }

    Capture capture_morton(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).morton;
        constexpr size_t n = 513;
        std::mt19937 rng(12345);
        std::vector<float> xyz(n * 3);
        for (size_t i = 0; i < n; ++i) {
            xyz[i * 3] = static_cast<float>(rng() % 17) - 8.f;
            xyz[i * 3 + 1] = static_cast<float>(rng() % 17) - 8.f;
            xyz[i * 3 + 2] = 2.f;
        }
        const auto means = Tensor::from_vector(xyz, {n, 3}, Device::GPU);
        const Tensor permutation = table->permutation(means);
        keep(out.snapshot, backend, "morton.permutation", permutation, kExact);

        const auto perm = shuffled_indices(n);
        std::vector<float> bound_values(joint::n_bounds_for_prims(n) * 4);
        for (size_t block = 0; block < bound_values.size() / 4; ++block) {
            bound_values[block * 4] = -0.3f - static_cast<float>(block) * 0.1f;
            bound_values[block * 4 + 1] = 0.5f + static_cast<float>(block) * 0.1f;
            bound_values[block * 4 + 2] = 0.01f;
            bound_values[block * 4 + 3] = 0.1f + static_cast<float>(block) * 0.01f;
        }
        const auto bounds = Tensor::from_vector(bound_values, {bound_values.size()}, Device::GPU);
        constexpr int width = 5;
        const size_t cells = lfs::core::sh_swizzled_padded_n(n) * width * 4;
        auto packed_cpu = Tensor::empty({cells * static_cast<size_t>(joint::bytes_per_cell(16))}, Device::CPU, DataType::UInt8);
        std::mt19937 bytes_rng(991);
        for (size_t i = 0; i < packed_cpu.numel(); ++i) {
            packed_cpu.ptr<uint8_t>()[i] = static_cast<uint8_t>(bytes_rng());
        }
        const auto packed = packed_cpu.gpu();
        auto permuted = Tensor::zeros(packed.shape(), Device::GPU, DataType::UInt8);
        auto permuted_bounds = Tensor::zeros(bounds.shape(), Device::GPU);
        const ops::JointCodecParams codec{ops::JointLayout::SwizzledSH, static_cast<int>(n), width, 16};
        table->permute_joint(packed, bounds, perm, permuted, permuted_bounds, codec);
        keep(out.snapshot, backend, "morton.joint.packed", permuted, kExact);
        keep(out.snapshot, backend, "morton.joint.bounds", permuted_bounds, kExact);
        auto grouped = packed.clone();
        auto grouped_bounds = Tensor::zeros(bounds.shape(), Device::GPU);
        const size_t slot_bytes = cells / width * static_cast<size_t>(joint::bytes_per_cell(16));
        auto scratch = Tensor::empty({slot_bytes * 3}, Device::GPU, DataType::UInt8);
        table->permute_joint_grouped(grouped, bounds, perm, grouped_bounds, scratch, codec);
        keep(out.snapshot, backend, "morton.grouped.packed", grouped, kExact);

        constexpr size_t sh_n = 32;
        constexpr uint32_t rest = 3;
        const auto sh_perm = shuffled_indices(sh_n);
        const auto target = lfs::core::TensorExecutionTarget::current();
        lfs::training::IdleArenaScratch arena(8u << 20, 0, target);
        auto fp32 = make_degree_splat(sh_n, 1);
        table->permute_sh_fp32(fp32, sh_perm, target, arena);
        keep(out.snapshot, backend, "morton.sh_fp32", fp32.shN(), kExact);

        lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(true);
        auto q16 = make_degree_splat(sh_n, 1);
        const bool quantized = lfs::training::sh_value::apply_shN_value_quant(q16);
        lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
        if (!quantized) {
            out.error = "sh value quant did not apply";
            return out;
        }
        table->permute_sh_q16(q16, sh_perm, target, arena);
        keep(out.snapshot, backend, "morton.sh_q16", q16.shN(), kExact);

        auto live = pattern({16}, 0.1f, 1);
        auto source = pattern({16}, 0.2f, 2);
        table->copy_back(live, source.data_ptr(), source.bytes(), target);
        keep(out.snapshot, backend, "morton.copy_back", live, kExact);
        auto gradient = pattern({lfs::core::sh_swizzled_float_count(sh_n, rest)}, 0.2f, 3);
        auto gathered = Tensor::zeros(gradient.shape(), Device::GPU);
        table->gather_gradient(gradient, sh_perm, gathered, rest, target);
        keep(out.snapshot, backend, "morton.gather_gradient", gathered, kExact);
        return out;
    }

    lfs::core::Camera make_camera(int width, int height) {
        std::vector<float> rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> translation = {0, 0, 4};
        auto r = Tensor::from_blob(rotation.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto t = Tensor::from_blob(translation.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
        return lfs::core::Camera(r, t, 100.f, 100.f, width * 0.5f, height * 0.5f, Tensor(), Tensor(),
                                 lfs::core::CameraModelType::PINHOLE, "test", "", std::filesystem::path{}, width, height, 0);
    }

    lfs::core::SplatData make_fast_splat(int count) {
        auto means = Tensor::zeros({static_cast<size_t>(count), 3}, Device::GPU);
        auto cpu = means.to(Device::CPU);
        float* values = cpu.ptr<float>();
        for (int i = 0; i < count; ++i) {
            values[i * 3 + 0] = (i % 5) * 0.3f - 0.6f;
            values[i * 3 + 1] = (i / 5) * 0.3f - 0.6f;
            values[i * 3 + 2] = 0.f;
        }
        means = cpu.to(Device::GPU);
        auto sh0 = Tensor::full({static_cast<size_t>(count), 1, 3}, 0.5f, Device::GPU);
        auto shN = Tensor::zeros({static_cast<size_t>(count), 0, 3}, Device::GPU);
        auto scaling = Tensor::full({static_cast<size_t>(count), 3}, -2.f, Device::GPU);
        std::vector<float> rotation(static_cast<size_t>(count) * 4, 0.f);
        for (int i = 0; i < count; ++i) {
            rotation[static_cast<size_t>(i) * 4] = 1.f;
        }
        auto quat = Tensor::from_blob(rotation.data(), {static_cast<size_t>(count), 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto opacity = Tensor::full({static_cast<size_t>(count)}, 2.f, Device::GPU);
        return lfs::core::SplatData(0, std::move(means), std::move(sh0), std::move(shN), std::move(scaling),
                                    std::move(quat), std::move(opacity), 1.f);
    }

    Capture capture_fast(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).fast;
        table->warmup();
        out.snapshot.exact_i("fast.warmup", 1);
        auto camera = make_camera(32, 24);
        auto splat = make_fast_splat(12);
        auto background = Tensor::zeros({3}, Device::GPU);
        ops::FastSaved saved{.backend = table->create()};
        if (!saved.backend) {
            out.error = "fast create returned null";
            return out;
        }
        lfs::training::RenderOutput output;
        const ops::RasterResult forward = lfs::training::fast_render(
            *table, saved, camera, splat, background, 0, 0, 0, 0, false, {}, false, true, output);
        out.snapshot.exact_i("fast.forward.code", static_cast<int64_t>(forward.code));
        out.snapshot.exact_i("fast.forward.work", forward.has_work ? 1 : 0);
        if (forward.code != ops::RasterResult::Code::Success) {
            out.error = std::string("fast forward failed: ") + std::string(forward.message);
            table->release(saved);
            return out;
        }
        keep(out.snapshot, backend, "fast.image", output.image, kRaster);
        keep(out.snapshot, backend, "fast.alpha", output.alpha, kRaster);
        table->record_vram(saved, output.image, output.alpha, true, static_cast<size_t>(splat.size()));
        out.snapshot.exact_i("fast.record_vram", 1);

        lfs::training::AdamConfig config{.lr = 1e-2f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        lfs::training::AdamOptimizer optimizer(splat, config);
        optimizer.allocate_gradients();
        optimizer.zero_grad(1);
        const ops::BackwardAdam adam = optimizer.prepare_fastgs_fused_adam(1, lfs::core::TensorExecutionTarget::current());
        if (!lfs::training::fastgs_adam_enabled(adam)) {
            out.error = "fast fused adam was not enabled";
            table->release(saved);
            return out;
        }
        // A zero image gradient keeps the fused update deterministic. A nonzero
        // blend accumulates with atomics, so two CUDA launches are not byte-identical.
        auto grad = Tensor::zeros_like(output.image);
        Tensor none;
        table->backward(saved, {.image = grad, .alpha = none, .depth = none, .normal = none},
                        splat._densification_info, none, none, none, adam, DensificationType::None);
        keep(out.snapshot, backend, "fast.backward.means", splat.means(), kAdam);
        keep(out.snapshot, backend, "fast.backward.scales", splat.scaling_raw(), kAdam);
        keep(out.snapshot, backend, "fast.backward.opacity", splat.opacity_raw(), kAdam);
        table->release_caches(saved);
        out.snapshot.exact_i("fast.release_caches", 1);
        table->release(saved);
        out.snapshot.exact_i("fast.release", 1);
        return out;
    }

    Capture capture_gsplat(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).gsplat;
        auto means = Tensor::from_vector(std::vector<float>{0.f, 0.f, 3.f}, {1, 3}, Device::GPU);
        auto scales = Tensor::full({1, 3}, -1.f, Device::GPU);
        auto rotations = Tensor::from_vector(std::vector<float>{1.f, 0.f, 0.f, 0.f}, {1, 4}, Device::GPU);
        auto opacity = Tensor::full({1, 1}, 1.f, Device::GPU);
        auto sh0 = Tensor::full({1, 1, 3}, 0.25f, Device::GPU);
        auto shN = Tensor::zeros({384}, Device::GPU);
        auto view = Tensor::eye(4, Device::GPU);
        auto background = Tensor::full({3}, 0.1f, Device::GPU);
        Tensor empty;
        const ops::SplatInputs inputs{means, scales, rotations, opacity, sh0, shN, empty};
        ops::GsplatParams params{.full_image = {1, 1}, .intrinsics = {1.f, 1.f, 0.5f, 0.5f}, .sh = {.active_bases = 1, .layout_bases = 1}};
        ops::GsplatSaved saved{.backend = table->create()};
        if (!saved.backend) {
            out.error = "gsplat create returned null";
            return out;
        }
        Tensor image, alpha, depth, normal;
        const ops::RasterResult forward = table->forward(saved, inputs, view, empty, empty, background, empty, params, {image, alpha, depth, normal});
        out.snapshot.exact_i("gsplat.forward.code", static_cast<int64_t>(forward.code));
        if (forward.code != ops::RasterResult::Code::Success) {
            out.error = std::string("gsplat forward failed: ") + std::string(forward.message);
            table->release(saved);
            return out;
        }
        keep(out.snapshot, backend, "gsplat.image", image, kRaster);
        keep(out.snapshot, backend, "gsplat.alpha", alpha, kRaster);
        table->record_vram(saved, image, alpha, empty, empty, empty);
        out.snapshot.exact_i("gsplat.record_vram", 1);
        auto image_grad = Tensor::full({3, 1, 1}, 0.125f, Device::GPU);
        auto alpha_grad = Tensor::full({1, 1, 1}, 0.25f, Device::GPU);
        auto gm = Tensor::zeros_like(means);
        auto gs = Tensor::zeros_like(scales);
        auto gr = Tensor::zeros_like(rotations);
        auto go = Tensor::zeros_like(opacity);
        auto g0 = Tensor::zeros_like(sh0);
        auto gn = Tensor::zeros_like(shN);
        struct GradientBuffers {
            std::array<Tensor*, 6> tensors;
        };
        GradientBuffers buffers{{&gm, &gs, &gr, &go, &g0, &gn}};
        const ops::GsplatGradients gradients{&buffers, [](void* owner, ops::AdamSlot slot) -> Tensor& {
                                                 return *static_cast<GradientBuffers*>(owner)->tensors[static_cast<size_t>(slot)];
                                             }};
        auto densification = Tensor::zeros({2, 1}, Device::GPU);
        auto errors = Tensor::full({1, 1}, 0.3f, Device::GPU);
        auto edges = Tensor::full({1, 1}, 0.7f, Device::GPU);
        auto scores = Tensor::zeros({1}, Device::GPU);
        auto share = Tensor::zeros({1}, Device::GPU);
        table->backward(saved, image_grad, alpha_grad, gradients, densification, errors, edges, scores, share);
        keep(out.snapshot, backend, "gsplat.grad.means", gm, kRaster);
        keep(out.snapshot, backend, "gsplat.grad.sh0", g0, kRaster);
        keep(out.snapshot, backend, "gsplat.densification", densification, kReduce);
        const bool caches_released = table->release_caches(saved);
        out.snapshot.exact_i("gsplat.release_caches", caches_released ? 1 : 0);
        table->release(saved);
        out.snapshot.exact_i("gsplat.release", 1);
        return out;
    }

    Capture capture_session(GpuBackend backend) {
        Capture out;
        const auto* table = lfs::training::training_ops(backend).session;
        // A previous caller may have left a live arena. Reset first so the borrow
        // size does not depend on that residue; full_reset is idempotent.
        table->reset_arena();
        for (const size_t bytes : {size_t{0}, size_t{1}, size_t{255}, size_t{256}, size_t{4096}, size_t{1} << 20}) {
            out.snapshot.exact_u("session.allocation." + std::to_string(bytes), table->allocation_bytes(bytes));
        }
        const uint32_t previous = table->set_arena_timeout(17);
        const uint32_t observed = table->set_arena_timeout(previous);
        out.snapshot.exact_u("session.timeout.previous", previous);
        out.snapshot.exact_u("session.timeout.roundtrip", observed);
        out.snapshot.exact_u("session.baseline", table->device_baseline_bytes());
        table->sample_memory();
        table->dump_arena_statistics();
        table->log_arena_failure("training-ops-parity");
        table->profile(true);
        table->profile(false);
        if (backend == GpuBackend::CUDA) {
            cudaGetLastError();
        }
        (void)table->arena_memory_info();
        const auto target = lfs::core::TensorExecutionTarget::current();
        const ops::IdleArenaBorrow borrow = table->borrow_idle_arena(4096, 0, target);
        if (borrow.data != nullptr && borrow.bytes > 0) {
            table->zero_idle_arena(borrow, std::min(borrow.bytes, size_t{64}), target);
        }
        table->release_idle_arena(borrow, target);
        out.snapshot.exact_u("session.borrow", borrow.bytes);
        (void)table->direct_storage_live_bytes();
        const auto error = table->last_error();
        out.snapshot.exact_i("session.last_error", error ? 1 : 0);
        table->resize_arena("training-ops-parity", false);
        table->reset_arena();
        out.snapshot.exact_i("session.reset", 1);
        return out;
    }

    Capture capture(lfs::training::Family family, GpuBackend backend) {
        lfs::core::GpuBackendScope scope(backend);
        switch (family) {
        case lfs::training::Family::Photometric: return capture_photometric(backend);
        case lfs::training::Family::Adam: return capture_adam(backend);
        case lfs::training::Family::Mcmc: return capture_mcmc(backend);
        case lfs::training::Family::Refine: return capture_refine(backend);
        case lfs::training::Family::Masks: return capture_masks(backend);
        case lfs::training::Family::ExtraLoss: return capture_extra(backend);
        case lfs::training::Family::Geometry: return capture_geometry(backend);
        case lfs::training::Family::Mrnf: return capture_mrnf(backend);
        case lfs::training::Family::Bilateral: return capture_bilateral(backend);
        case lfs::training::Family::PPISP: return capture_ppisp(backend);
        case lfs::training::Family::Controller: return capture_controller(backend);
        case lfs::training::Family::TrainingImage: return capture_training_image(backend);
        case lfs::training::Family::SharedImage: return capture_shared_image(backend);
        case lfs::training::Family::Lpips: return capture_lpips(backend);
        case lfs::training::Family::Sh: return capture_sh(backend);
        case lfs::training::Family::Morton: return capture_morton(backend);
        case lfs::training::Family::Fast: return capture_fast(backend);
        case lfs::training::Family::Gsplat: return capture_gsplat(backend);
        case lfs::training::Family::Session: return capture_session(backend);
        case lfs::training::Family::Count: break;
        }
        Capture out;
        out.error = "no capture for family";
        return out;
    }

    void expect_match(const Capture& actual, const Capture& expected, bool exact) {
        EXPECT_TRUE(actual.error.empty()) << actual.error;
        EXPECT_TRUE(expected.error.empty()) << expected.error;
        if (!actual.error.empty() || !expected.error.empty()) {
            return;
        }
        const std::string diff = compare_snapshots(actual.snapshot, expected.snapshot, exact);
        EXPECT_TRUE(diff.empty()) << diff;
    }

    TEST(TrainingOpsParity, ComparatorRejectsPerturbation) {
        Snapshot original;
        keep_f(original, "value", 1.f, Tol{1e-6, 0});
        original.exact_i("index", 7);
        Snapshot within = original;
        within.fields[0].values[0] = 1.f + 1e-7f;
        within.fields[0].bytes.clear();
        EXPECT_TRUE(compare_snapshots(within, original, false).empty());

        Snapshot perturbed = original;
        perturbed.fields[0].values[0] = 1.f + 1e-3f;
        const std::string rejected = compare_snapshots(perturbed, original, false);
        EXPECT_FALSE(rejected.empty()) << "perturbed float must fail the tolerance";

        Snapshot flipped = original;
        flipped.fields[1].bytes[0] ^= 0x1;
        EXPECT_FALSE(compare_snapshots(flipped, original, true).empty());
        EXPECT_TRUE(compare_snapshots(original, original, true).empty());
    }

    TEST(TrainingOpsParity, SelfCheck) {
        if (!lfs::core::gpu_backend_available(GpuBackend::CUDA)) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto previous = lfs::core::default_gpu_backend();
        lfs::test::reset_gpu_backend_for_testing();
        ASSERT_TRUE(lfs::core::set_default_gpu_backend(GpuBackend::CUDA));
        const lfs::core::GpuBackendScope scope(GpuBackend::CUDA);
        for (size_t index = 0; index < static_cast<size_t>(lfs::training::Family::Count); ++index) {
            const auto family = static_cast<lfs::training::Family>(index);
            SCOPED_TRACE(lfs::training::training_family_name(family));
            expect_match(capture(family, GpuBackend::CUDA), capture(family, GpuBackend::CUDA), true);
        }
        lfs::test::reset_gpu_backend_for_testing();
        EXPECT_TRUE(lfs::core::set_default_gpu_backend(previous));
    }

    class TrainingOpsFamilyParity
        : public ::testing::TestWithParam<std::tuple<lfs::training::Family, GpuBackend>> {};

    TEST_P(TrainingOpsFamilyParity, MatchesCuda) {
        const auto [family, backend] = GetParam();
        if (!family_present(lfs::training::training_ops(backend), family)) {
            GTEST_SKIP() << lfs::training::training_family_name(family);
        }
        if (!lfs::core::gpu_backend_available(GpuBackend::CUDA) || !lfs::core::gpu_backend_available(backend)) {
            GTEST_SKIP() << backend_name(backend) << " or CUDA device unavailable";
        }
        expect_match(capture(family, backend), capture(family, GpuBackend::CUDA), false);
    }

    INSTANTIATE_TEST_SUITE_P(
        Backends, TrainingOpsFamilyParity,
        ::testing::Combine(::testing::Values(
                               lfs::training::Family::Session,
                               lfs::training::Family::Fast,
                               lfs::training::Family::Gsplat,
                               lfs::training::Family::Photometric,
                               lfs::training::Family::Geometry,
                               lfs::training::Family::Masks,
                               lfs::training::Family::ExtraLoss,
                               lfs::training::Family::Adam,
                               lfs::training::Family::Mcmc,
                               lfs::training::Family::Mrnf,
                               lfs::training::Family::Refine,
                               lfs::training::Family::Bilateral,
                               lfs::training::Family::PPISP,
                               lfs::training::Family::Controller,
                               lfs::training::Family::Morton,
                               lfs::training::Family::Sh,
                               lfs::training::Family::TrainingImage,
                               lfs::training::Family::SharedImage,
                               lfs::training::Family::Lpips),
                           ::testing::Values(GpuBackend::Vulkan, GpuBackend::Metal)),
        [](const ::testing::TestParamInfo<TrainingOpsFamilyParity::ParamType>& info) {
            return std::string(lfs::training::training_family_name(std::get<0>(info.param))) + "_" +
                   backend_name(std::get<1>(info.param));
        });

    std::filesystem::path write_synthetic_scene() {
        const auto root = std::filesystem::temp_directory_path() / "lfs_training_ops_parity_scene";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root / "images");
        std::filesystem::create_directories(root / "sparse" / "0");
        std::ofstream(root / "sparse/0/cameras.txt") << "1 PINHOLE 32 32 40 40 16 16\n";
        std::ofstream(root / "sparse/0/images.txt")
            << "1 1 0 0 0 0 0 2 1 a.png\n\n"
            << "2 1 0 0 0 0.2 0 2 1 b.png\n\n";
        std::ofstream points(root / "sparse/0/points3D.txt");
        for (int i = 0; i < 16; ++i) {
            const float x = (static_cast<float>(i % 4) - 1.5f) * 0.15f;
            const float y = (static_cast<float>(i / 4) - 1.5f) * 0.15f;
            points << (i + 1) << ' ' << x << ' ' << y << " 0 128 140 120 0.1 1 0 2 0\n";
        }
        std::vector<uint8_t> rgb(32 * 32 * 3);
        for (const char* name : {"a.png", "b.png"}) {
            for (int y = 0; y < 32; ++y) {
                for (int x = 0; x < 32; ++x) {
                    const int pixel = (y * 32 + x) * 3;
                    const int shift = name[0] == 'b' ? 40 : 0;
                    rgb[static_cast<size_t>(pixel)] = static_cast<uint8_t>(40 + x * 4);
                    rgb[static_cast<size_t>(pixel + 1)] = static_cast<uint8_t>(20 + y * 3 + shift / 4);
                    rgb[static_cast<size_t>(pixel + 2)] = static_cast<uint8_t>(80 + ((x + y + shift) % 50));
                }
            }
            if (stbi_write_png((root / "images" / name).string().c_str(), 32, 32, 3, rgb.data(), 32 * 3) == 0) {
                return {};
            }
        }
        return root;
    }

    std::vector<float> train_synthetic(const std::filesystem::path& dataset, const std::filesystem::path& output, std::string& error) {
        lfs::core::param::TrainingParameters params;
        params.dataset.data_path = dataset;
        params.dataset.images = "images";
        params.dataset.output_path = output;
        params.optimization.iterations = 200;
        params.optimization.strategy = "mcmc";
        params.optimization.sh_degree = 0;
        params.optimization.headless = true;
        params.optimization.max_cap = 5000;
        params.optimization.refine_every = 1000;
        params.optimization.start_refine = 1000;
        params.optimization.stop_refine = 1000;
        params.optimization.morton_reorder_interval = 0;
        params.optimization.save_steps = {};
        params.optimization.eval_steps = {};
        params.optimization.enable_eval = false;
        std::filesystem::create_directories(output);

        std::vector<float> losses;
        auto& boundary = lfs::training::ControlBoundary::instance();
        const auto handle = boundary.register_callback(lfs::training::ControlHook::PostStep, [&](const lfs::training::HookContext& ctx) {
            losses.push_back(ctx.loss);
        });
        lfs::core::Scene scene;
        const auto loaded = lfs::training::loadTrainingDataIntoScene(params, scene);
        if (!loaded) {
            error = loaded.error();
            boundary.unregister_callback(lfs::training::ControlHook::PostStep, handle);
            return {};
        }
        const auto inited = lfs::training::initializeTrainingModel(params, scene);
        if (!inited) {
            error = inited.error();
            boundary.unregister_callback(lfs::training::ControlHook::PostStep, handle);
            return {};
        }
        lfs::training::Trainer trainer(scene);
        const auto ready = trainer.initialize(params);
        if (!ready) {
            error = ready.error();
            boundary.unregister_callback(lfs::training::ControlHook::PostStep, handle);
            return {};
        }
        const auto trained = trainer.train();
        trainer.shutdown();
        boundary.unregister_callback(lfs::training::ControlHook::PostStep, handle);
        if (!trained) {
            error = lfs::format_for_developer(trained.error());
            return {};
        }
        return losses;
    }

    class TrainingOpsLossCurveParity : public ::testing::TestWithParam<GpuBackend> {};

    TEST_P(TrainingOpsLossCurveParity, SyntheticLossCurve) {
        const auto backend = std::optional<GpuBackend>(GetParam());
        lfs::core::param::TrainingParameters params;
        params.optimization.iterations = 200;
        params.optimization.strategy = "mcmc";
        params.optimization.sh_degree = 0;
        params.optimization.headless = true;
        params.optimization.max_cap = 5000;
        params.optimization.refine_every = 1000;
        params.optimization.start_refine = 1000;
        params.optimization.stop_refine = 1000;
        params.optimization.morton_reorder_interval = 0;
        params.optimization.enable_eval = false;
        const auto required = lfs::training::required_training_families(params, lfs::training::training_loader_dependencies(params));
        const auto missing = lfs::training::missing_training_families(lfs::training::training_ops(*backend), required);
        if (!missing.empty()) {
            std::string names;
            for (size_t i = 0; i < missing.size(); ++i) {
                if (i != 0) {
                    names += ", ";
                }
                names += missing[i];
            }
            GTEST_SKIP() << names;
            return;
        }
        if (!lfs::core::gpu_backend_available(GpuBackend::CUDA) || !lfs::core::gpu_backend_available(*backend)) {
            GTEST_SKIP() << backend_name(*backend) << " or CUDA device unavailable";
        }

        const auto dataset = write_synthetic_scene();
        ASSERT_FALSE(dataset.empty());
        const auto previous = lfs::core::default_gpu_backend();
        auto run = [&](GpuBackend which, const char* tag) {
            lfs::test::reset_gpu_backend_for_testing();
            const auto switched = lfs::core::set_default_gpu_backend(which);
            if (!switched) {
                ADD_FAILURE() << lfs::format_for_developer(switched.error());
            }
            lfs::core::GpuBackendScope scope(which);
            std::string error;
            auto losses = train_synthetic(dataset, dataset / tag, error);
            EXPECT_TRUE(error.empty()) << error;
            return losses;
        };
        const auto reference = run(GpuBackend::CUDA, "cuda");
        const auto other = run(*backend, "second");
        lfs::test::reset_gpu_backend_for_testing();
        const auto restored = lfs::core::set_default_gpu_backend(previous);
        if (!restored) {
            ADD_FAILURE() << lfs::format_for_developer(restored.error());
        }
        ASSERT_FALSE(reference.empty());
        ASSERT_EQ(reference.size(), other.size());
        float max_abs = 0.f;
        size_t worst = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            EXPECT_TRUE(std::isfinite(reference[i]));
            EXPECT_TRUE(std::isfinite(other[i]));
            const float delta = std::fabs(other[i] - reference[i]);
            if (delta > max_abs) {
                max_abs = delta;
                worst = i;
            }
            EXPECT_TRUE(same_float(other[i], reference[i], kLoss))
                << "step " << i << " cuda " << reference[i] << " second " << other[i]
                << " max_abs " << max_abs << " at " << worst;
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, TrainingOpsLossCurveParity,
                             ::testing::Values(GpuBackend::CUDA, GpuBackend::Vulkan, GpuBackend::Metal),
                             [](const ::testing::TestParamInfo<GpuBackend>& info) {
                                 return std::string(backend_name(info.param));
                             });

} // namespace

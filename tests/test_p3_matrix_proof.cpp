/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_document.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::Uuid;
    using Json = lfs::io::JsonChapterDom::Json;
    using namespace lfs::io::project;

    constexpr auto P3_MATRIX_ROWS = std::to_array<std::string_view>({
        "PROJ-39",
        "PROJ-40",
        "PROJ-41",
        "PROJ-42",
        "PROJ-43",
        "PROJ-44",
        "PROJ-45",
        "PRMS-53",
        "PRMS-54",
        "PRMS-55",
        "PRMS-56",
        "SCNG-64",
        "SCNG-65",
        "SCNG-66",
        "SCNG-67",
        "SCNG-68",
        "SCNG-69",
        "SCNG-70",
        "SCNG-71",
        "SCNG-72",
        "SCNG-73",
        "SCNG-74",
        "SCNG-75",
        "SCNG-76",
        "SCNG-77",
        "SELM-83",
        "SELM-84",
        "SELM-85",
        "SELM-86",
        "REFS-96",
        "REFS-97",
        "REFS-98",
        "REFS-99",
        "REFS-100",
        "REFS-101",
        "REFS-102",
        "SPLT-110",
        "SPLT-111",
        "SPLT-112",
        "SPLT-113",
        "SPLT-114",
        "SPLT-115",
        "SPLT-116",
    });

    struct PendingParameterExclusion {
        std::string_view field;
        std::string_view phase;
    };

    constexpr auto PENDING_PARAMETER_EXCLUSIONS =
        std::to_array<PendingParameterExclusion>({
            {"headless", "P6/process launch"},
            {"auto_train", "P6/process launch"},
            {"no_splash", "P6/process launch"},
            {"debug_python", "P6/process launch"},
            {"debug_python_port", "P6/process launch"},
            {"config_file", "P6/process launch"},
        });

    constexpr auto PENDING_OPTIMIZATION_PARAMETER_FIELDS =
        std::to_array<std::string_view>({
            "iterations",
            "means_lr",
            "means_lr_end",
            "shs_lr",
            "opacity_lr",
            "scaling_lr",
            "scaling_lr_end",
            "rotation_lr",
            "cropbox_lr_scale",
            "cropbox_loss_weight",
            "lambda_dssim",
            "min_opacity",
            "refine_every",
            "start_refine",
            "stop_refine",
            "grad_threshold",
            "sh_degree",
            "opacity_reg",
            "scale_reg",
            "init_opacity",
            "init_scaling",
            "max_cap",
            "eval_steps",
            "save_steps",
            "enable_eval",
            "enable_save_eval_images",
            "strategy",
            "mip_filter",
            "use_bilateral_grid",
            "bilateral_grid_X",
            "bilateral_grid_Y",
            "bilateral_grid_W",
            "bilateral_grid_lr",
            "tv_loss_weight",
            "use_ppisp",
            "ppisp_lr",
            "ppisp_reg_weight",
            "ppisp_warmup_steps",
            "ppisp_freeze_from_sidecar",
            "ppisp_reference_uuid",
            "ppisp_use_controller",
            "ppisp_freeze_gaussians_on_distill",
            "ppisp_controller_activation_step",
            "ppisp_controller_lr",
            "prune_opacity",
            "grow_scale3d",
            "grow_scale2d",
            "prune_scale3d",
            "prune_scale2d",
            "reset_every",
            "pause_refine_after_reset",
            "revised_opacity",
            "gut",
            "undistort",
            "steps_scaler",
            "sh_degree_interval",
            "random",
            "init_num_pts",
            "init_extent",
            "enable_sparsity",
            "sparsify_steps",
            "init_rho",
            "prune_ratio",
            "bg_modulation",
            "bg_mode",
            "bg_color",
            "background_image_reference_uuid",
            "mask_mode",
            "invert_masks",
            "mask_opacity_penalty_weight",
            "mask_opacity_penalty_power",
            "mask_threshold",
            "use_alpha_as_mask",
            "use_depth_loss",
            "depth_loss_weight",
            "depth_loss_mode",
            "use_normal_loss",
            "normal_loss_weight",
            "normal_consistency_weight",
            "normal_flatten_weight",
            "normal_loss_space",
            "growth_grad_threshold",
            "grow_fraction",
            "grow_until_iter",
            "opacity_decay",
            "scale_decay",
            "means_noise_weight",
            "bounds_percentile",
            "use_error_map",
            "use_edge_map",
        });

    Json pending_parameter_field_map(
        const lfs::core::param::OptimizationParameters& parameters,
        const ParameterManagerSnapshot::ReferenceBindings&
            references) {
        Json result =
            Json::parse(parameters.to_json().dump());
        for (const auto& exclusion :
             PENDING_PARAMETER_EXCLUSIONS) {
            result.erase(exclusion.field);
        }
        result.erase("bg_image_path");
        result.erase("ppisp_sidecar_path");
        if (references.background_image_reference) {
            result["background_image_reference_uuid"] =
                references.background_image_reference->to_string();
        }
        if (references.ppisp_reference) {
            result["ppisp_reference_uuid"] =
                references.ppisp_reference->to_string();
        }
        return result;
    }

    lfs::core::param::OptimizationParameters
    distinct_pending_parameters(
        const std::string_view strategy,
        const std::uint32_t tag) {
        using lfs::core::param::BackgroundMode;
        using lfs::core::param::MaskMode;

        const float delta =
            static_cast<float>(tag) * 0.001f;
        lfs::core::param::OptimizationParameters result;
        result.iterations = 31'000 + tag;
        result.sh_degree_interval = 900 + tag;
        result.means_lr = 0.010f + delta;
        result.means_lr_end = 0.001f + delta;
        result.shs_lr = 0.020f + delta;
        result.opacity_lr = 0.030f + delta;
        result.scaling_lr = 0.040f + delta;
        result.scaling_lr_end = 0.050f + delta;
        result.rotation_lr = 0.060f + delta;
        result.cropbox_lr_scale = 0.21f + delta;
        result.cropbox_loss_weight = 0.22f + delta;
        result.lambda_dssim = 0.23f + delta;
        result.min_opacity = 0.024f + delta;
        result.refine_every = 70 + tag;
        result.start_refine = 700 + tag;
        result.stop_refine = 15'000 + tag;
        result.grad_threshold = 0.070f + delta;
        result.sh_degree = 2;
        result.opacity_reg = 0.080f + delta;
        result.scale_reg = 0.090f + delta;
        result.init_opacity = 0.41f + delta;
        result.init_scaling = 0.12f + delta;
        result.max_cap = 900'000 + static_cast<int>(tag);
        result.eval_steps = {1'111 + tag, 2'222 + tag};
        result.save_steps = {3'333 + tag, 4'444 + tag};
        result.bg_modulation = true;
        result.enable_eval = true;
        result.enable_save_eval_images = false;
        result.headless = true;
        result.auto_train = true;
        result.no_splash = true;
        result.debug_python = true;
        result.debug_python_port =
            10'000 + static_cast<int>(tag);
        result.strategy = strategy;

        result.mask_mode = MaskMode::AlphaConsistent;
        result.invert_masks = true;
        result.mask_threshold = 0.61f + delta;
        result.mask_opacity_penalty_weight = 0.71f + delta;
        result.mask_opacity_penalty_power = 1.51f + delta;
        result.use_alpha_as_mask = false;

        result.use_depth_loss = true;
        result.depth_loss_weight = 1.71f + delta;
        result.depth_loss_mode = "ssi-depth";
        result.use_normal_loss = true;
        result.normal_loss_weight = 0.31f + delta;
        result.normal_consistency_weight = 0.32f + delta;
        result.normal_flatten_weight = 0.33f + delta;
        result.normal_loss_space = "world";
        result.mip_filter = true;

        result.bg_mode = BackgroundMode::Image;
        result.bg_color = {
            0.11f + delta,
            0.22f + delta,
            0.33f + delta,
        };
        result.bg_image_path =
            std::format("excluded-background-{}.png", tag);

        result.use_bilateral_grid = true;
        result.bilateral_grid_X = 17 + static_cast<int>(tag);
        result.bilateral_grid_Y = 18 + static_cast<int>(tag);
        result.bilateral_grid_W = 9 + static_cast<int>(tag);
        result.bilateral_grid_lr = 0.014f + delta;
        result.tv_loss_weight = 7.1f + delta;

        result.use_ppisp = true;
        result.ppisp_lr = 0.015f + delta;
        result.ppisp_reg_weight = 0.016f + delta;
        result.ppisp_warmup_steps =
            600 + static_cast<int>(tag);
        result.ppisp_freeze_from_sidecar = true;
        result.ppisp_sidecar_path =
            std::format("excluded-ppisp-{}.bin", tag);
        result.ppisp_use_controller = true;
        result.ppisp_freeze_gaussians_on_distill = false;
        result.ppisp_controller_activation_step =
            2'000 + static_cast<int>(tag);
        result.ppisp_controller_lr = 0.017f + delta;

        result.prune_opacity = 0.18f + delta;
        result.grow_scale3d = 0.19f + delta;
        result.grow_scale2d = 0.20f + delta;
        result.prune_scale3d = 0.21f + delta;
        result.prune_scale2d = 0.22f + delta;
        result.reset_every = 2'500 + tag;
        result.pause_refine_after_reset = 200 + tag;
        result.revised_opacity = true;
        result.gut =
            strategy != lfs::core::param::kStrategyIGSPlus;
        result.undistort = true;
        result.steps_scaler = 1.2f + delta;

        result.growth_grad_threshold = 0.023f + delta;
        result.grow_fraction = 0.24f + delta;
        result.grow_until_iter = 12'000 + tag;
        result.opacity_decay = 0.025f + delta;
        result.scale_decay = 0.026f + delta;
        result.means_noise_weight = 40.0f + delta;
        result.bounds_percentile = 0.72f + delta;
        result.use_error_map = false;
        result.use_edge_map = false;

        result.random = true;
        result.init_num_pts = 80'000 + static_cast<int>(tag);
        result.init_extent = 4.0f + delta;
        result.enable_sparsity = true;
        result.sparsify_steps = 12'000 + static_cast<int>(tag);
        result.init_rho = 0.027f + delta;
        result.prune_ratio = 0.55f + delta;
        result.config_file =
            std::format("excluded-config-{}.json", tag);
        return result;
    }

    Uuid matrix_uuid(const std::uint64_t tag) {
        const auto parsed = Uuid::from_string(
            std::format("73000000-0000-4000-8000-{:012x}", tag));
        EXPECT_TRUE(parsed);
        return parsed.value_or(Uuid{});
    }

    class MatrixTemporaryDirectory {
    public:
        MatrixTemporaryDirectory()
            : path(fs::temp_directory_path() /
                   ("lfs-p3-matrix-" +
                    lfs::core::generate_uuid_v4().to_string())) {
            fs::create_directories(path);
        }

        ~MatrixTemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }

        fs::path path;
    };

    void require_ok(lfs::Result<void> result) {
        ASSERT_TRUE(result)
            << (result ? std::string{}
                       : lfs::format_for_developer(result.error()));
    }

    ProjectDocumentSaveOptions matrix_save_options(
        const std::uint64_t tag, const std::uint64_t wallclock) {
        return ProjectDocumentSaveOptions{
            .commit =
                {
                    .kind = CommitKind::Explicit,
                    .commit_uuid = matrix_uuid(tag),
                    .snapshot_uuid = matrix_uuid(tag + 1),
                    .wallclock_unix_ns = wallclock,
                },
            .file_uuid = matrix_uuid(tag + 2),
            .index_compression =
                IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    ReferenceFingerprint matrix_fingerprint(
        const std::uint8_t tag,
        const FingerprintKind kind = FingerprintKind::File) {
        ReferenceFingerprint result;
        result.kind = kind;
        result.size = 10'000 + tag;
        result.mtime_unix_ns = 20'000 + tag;
        result.head_xxh3.bytes.fill(tag);
        result.tail_xxh3.bytes.fill(
            static_cast<std::uint8_t>(tag + 1));
        Hash128 full;
        full.bytes.fill(static_cast<std::uint8_t>(tag + 2));
        result.full_xxh3 = full;
        return result;
    }

    std::vector<std::byte> byte_values(
        const std::initializer_list<std::uint8_t> values) {
        std::vector<std::byte> result;
        result.reserve(values.size());
        for (const std::uint8_t value : values) {
            result.push_back(static_cast<std::byte>(value));
        }
        return result;
    }

    Tensor uint8_tensor(
        const std::initializer_list<std::uint8_t> values,
        const lfs::core::TensorShape& shape) {
        Tensor result =
            Tensor::empty(shape, Device::CPU, DataType::UInt8);
        std::ranges::copy(values, result.ptr<std::uint8_t>());
        return result;
    }

    std::unique_ptr<lfs::core::SplatData> make_matrix_splat() {
        constexpr std::size_t count = 4;
        std::vector<float> means{
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            6.0f,
            7.0f,
            8.0f,
            9.0f,
            10.0f,
            11.0f,
            12.0f,
        };
        std::vector<float> sh0(count * 3);
        std::vector<float> shn(count * 3 * 3);
        std::vector<float> scaling(count * 3);
        std::vector<float> rotation(count * 4, 0.0f);
        std::vector<float> opacity(count);
        for (std::size_t index = 0; index < count; ++index) {
            sh0[index * 3] = static_cast<float>(index) + 0.1f;
            sh0[index * 3 + 1] = static_cast<float>(index) + 0.2f;
            sh0[index * 3 + 2] = static_cast<float>(index) + 0.3f;
            scaling[index * 3] = -1.0f - static_cast<float>(index);
            scaling[index * 3 + 1] = -2.0f;
            scaling[index * 3 + 2] = -3.0f;
            rotation[index * 4] = 1.0f;
            opacity[index] = -0.5f + static_cast<float>(index) * 0.1f;
        }
        for (std::size_t index = 0; index < shn.size(); ++index) {
            shn[index] = static_cast<float>(index) / 100.0f;
        }

        auto result = std::make_unique<lfs::core::SplatData>(
            1,
            Tensor::from_vector(
                means, {count, std::size_t{3}}, Device::CPU),
            Tensor::from_vector(
                sh0, {count, std::size_t{1}, std::size_t{3}},
                Device::CPU),
            Tensor::from_vector(
                shn, {count, std::size_t{3}, std::size_t{3}},
                Device::CPU),
            Tensor::from_vector(
                scaling, {count, std::size_t{3}}, Device::CPU),
            Tensor::from_vector(
                rotation, {count, std::size_t{4}}, Device::CPU),
            Tensor::from_vector(
                opacity, {count, std::size_t{1}}, Device::CPU),
            2.5f);
        result->set_active_sh_degree(0);
        result->deleted() =
            uint8_tensor({0, 1, 0, 1}, {count});
        result->_densification_info = Tensor::from_vector(
            std::vector<float>{
                0.1f,
                0.2f,
                0.3f,
                0.4f,
                1.1f,
                1.2f,
                1.3f,
                1.4f,
            },
            {std::size_t{2}, count}, Device::CPU);
        result->set_frozen_ranges({
            {.start = 1, .count = 2},
        });
        return result;
    }

    PointCloudPayload make_matrix_point_cloud() {
        auto point_cloud = std::make_shared<lfs::core::PointCloud>();
        point_cloud->means = Tensor::from_vector(
            std::vector<float>{
                0.0f,
                1.0f,
                2.0f,
                -3.0f,
                4.5f,
                6.0f,
            },
            {std::size_t{2}, std::size_t{3}}, Device::CPU);
        point_cloud->colors = uint8_tensor(
            {255, 128, 64, 0, 192, 255},
            {std::size_t{2}, std::size_t{3}});
        point_cloud->normals = Tensor::from_vector(
            std::vector<float>{
                0.0f,
                0.0f,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
            },
            {std::size_t{2}, std::size_t{3}}, Device::CPU);
        point_cloud->sh0 = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6},
            {std::size_t{2}, std::size_t{3}, std::size_t{1}},
            Device::CPU);
        point_cloud->shN = Tensor::from_vector(
            std::vector<float>{
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
            },
            {std::size_t{2}, std::size_t{3}, std::size_t{2}},
            Device::CPU);
        point_cloud->opacity = Tensor::from_vector(
            std::vector<float>{0.1f, 0.9f},
            {std::size_t{2}, std::size_t{1}}, Device::CPU);
        point_cloud->scaling = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6},
            {std::size_t{2}, std::size_t{3}}, Device::CPU);
        point_cloud->rotation = Tensor::from_vector(
            std::vector<float>{1, 0, 0, 0, 0.5f, 0.5f, 0.5f, 0.5f},
            {std::size_t{2}, std::size_t{4}}, Device::CPU);
        point_cloud->attribute_names = {
            "x", "y", "z", "red", "green", "blue", "opacity"};

        PointCloudPayload result(std::move(point_cloud));
        const auto status = result.add_opaque_property(
            GeometryPropertyPlane{
                .name = "vendor_score",
                .components = 1,
                .dtype = GeometryDtype::UInt16,
                .encoding = 77,
                .bytes = byte_values({0x10, 0x20, 0x30, 0x40}),
            });
        EXPECT_TRUE(status)
            << (status ? std::string{}
                       : lfs::format_for_developer(status.error()));
        return result;
    }

    MeshPayload make_matrix_mesh() {
        auto mesh = std::make_shared<lfs::core::MeshData>();
        mesh->vertices = Tensor::from_vector(
            std::vector<float>{
                -1.0f,
                -1.0f,
                0.0f,
                1.0f,
                -1.0f,
                0.0f,
                1.0f,
                1.0f,
                0.0f,
                -1.0f,
                1.0f,
                0.0f,
            },
            {std::size_t{4}, std::size_t{3}}, Device::CPU);
        mesh->normals = Tensor::from_vector(
            std::vector<float>{
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
            },
            {std::size_t{4}, std::size_t{3}}, Device::CPU);
        mesh->tangents = Tensor::from_vector(
            std::vector<float>{
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
            },
            {std::size_t{4}, std::size_t{4}}, Device::CPU);
        mesh->texcoords = Tensor::from_vector(
            std::vector<float>{0, 0, 1, 0, 1, 1, 0, 1},
            {std::size_t{4}, std::size_t{2}}, Device::CPU);
        mesh->colors = Tensor::from_vector(
            std::vector<float>{
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
            },
            {std::size_t{4}, std::size_t{4}}, Device::CPU);
        mesh->indices = Tensor::from_vector(
            std::vector<std::int32_t>{0, 1, 2, 0, 2, 3},
            {std::size_t{2}, std::size_t{3}}, Device::CPU);
        mesh->texture_images.push_back(lfs::core::TextureImage{
            .pixels = {1, 2, 3, 4, 5, 6},
            .width = 2,
            .height = 1,
            .channels = 3,
        });
        lfs::core::Material material;
        material.name = "matrix-plane";
        material.base_color = {0.25f, 0.5f, 0.75f, 1.0f};
        material.emissive = {0.1f, 0.2f, 0.3f};
        material.metallic = 0.4f;
        material.roughness = 0.6f;
        material.ao = 0.8f;
        material.albedo_tex = 1;
        material.normal_tex = 1;
        material.metallic_roughness_tex = 1;
        material.emissive_tex = 1;
        material.ao_tex = 1;
        material.albedo_tex_path = "textures/albedo.png";
        material.normal_tex_path = "textures/normal.png";
        material.metallic_roughness_tex_path = "textures/mr.png";
        material.double_sided = true;
        mesh->materials.push_back(material);
        mesh->submeshes.push_back(lfs::core::Submesh{
            .start_index = 0,
            .index_count = 6,
            .material_index = 0,
        });

        MeshPayload result(std::move(mesh));
        const auto status = result.add_opaque_property(
            GeometryPropertyPlane{
                .name = "vendor_ids",
                .components = 1,
                .dtype = GeometryDtype::UInt32,
                .encoding = 91,
                .bytes = byte_values({
                    0x01,
                    0x00,
                    0x00,
                    0x00,
                    0x02,
                    0x00,
                    0x00,
                    0x00,
                    0x03,
                    0x00,
                    0x00,
                    0x00,
                    0x04,
                    0x00,
                    0x00,
                    0x00,
                }),
            });
        EXPECT_TRUE(status)
            << (status ? std::string{}
                       : lfs::format_for_developer(status.error()));
        return result;
    }

    std::vector<std::string> split_markdown_row(
        const std::string_view line) {
        std::vector<std::string> cells;
        std::size_t begin = 0;
        while (begin <= line.size()) {
            const std::size_t end = line.find('|', begin);
            std::string cell(line.substr(
                begin, end == std::string_view::npos
                           ? line.size() - begin
                           : end - begin));
            const auto first = cell.find_first_not_of(" \t");
            const auto last = cell.find_last_not_of(" \t");
            cells.push_back(
                first == std::string::npos
                    ? std::string{}
                    : cell.substr(first, last - first + 1));
            if (end == std::string_view::npos) {
                break;
            }
            begin = end + 1;
        }
        return cells;
    }

    std::string phase_for_authority(const std::string_view authority) {
        if (authority == "PROJ" || authority == "PRMS" ||
            authority == "SCNG" || authority == "SELM" ||
            authority == "REFS" || authority == "SPLT") {
            return "P3";
        }
        if (authority == "CKPT" || authority == "PPIS") {
            return "P4";
        }
        if (authority == "GUIL" || authority == "EDTR" ||
            authority == "VIEW" || authority == "SEQR" ||
            authority == "METR") {
            return "P5";
        }
        return {};
    }

    struct MatrixRows {
        std::set<std::string> p3;
        std::map<std::string, std::string> deferred;
    };

    MatrixRows read_matrix_rows() {
        const fs::path path =
            fs::path(PROJECT_ROOT_PATH) / "docs/licht_ownership_matrix.md";
        std::ifstream stream(path);
        EXPECT_TRUE(stream.is_open()) << path;
        MatrixRows result;
        std::string line;
        std::size_t line_number = 0;
        while (std::getline(stream, line)) {
            ++line_number;
            if (line.empty() || line.front() != '|') {
                continue;
            }
            const auto cells = split_markdown_row(line);
            if (cells.size() < 6 || cells[4].size() < 3 ||
                cells[4].front() != '`' || cells[4].back() != '`') {
                continue;
            }
            const std::string authority =
                cells[4].substr(1, cells[4].size() - 2);
            const std::string phase = phase_for_authority(authority);
            if (phase.empty()) {
                continue;
            }
            const std::string row_id =
                std::format("{}-{}", authority, line_number);
            if (phase == "P3") {
                result.p3.insert(row_id);
            } else {
                result.deferred.emplace(row_id, phase);
            }
        }
        return result;
    }

    TEST(P3MatrixProof, EveryAssignedRowSurvivesSaveLoadSave) {
        MatrixTemporaryDirectory temporary;
        const fs::path path = temporary.path / "matrix-proof.licht";

        const Uuid project_uuid = matrix_uuid(1);
        const Uuid dataset_ref = matrix_uuid(2);
        const Uuid colmap_ref = matrix_uuid(3);
        const Uuid rad_ref = matrix_uuid(4);
        const Uuid rad_meta_ref = matrix_uuid(5);
        const Uuid background_ref = matrix_uuid(6);
        const Uuid environment_ref = matrix_uuid(7);
        const Uuid sequence_ref = matrix_uuid(8);
        const Uuid ppisp_ref = matrix_uuid(9);
        const Uuid root_node = matrix_uuid(20);
        const Uuid dataset_node = matrix_uuid(21);
        const Uuid camera_group_node = matrix_uuid(22);
        const Uuid image_group_node = matrix_uuid(23);
        const Uuid image_node = matrix_uuid(24);
        const Uuid sequence_node = matrix_uuid(25);
        const Uuid training_node = matrix_uuid(26);
        const Uuid imported_node = matrix_uuid(27);
        const Uuid point_node = matrix_uuid(28);
        const Uuid mesh_node = matrix_uuid(29);
        const Uuid live_rad_node = matrix_uuid(30);
        const Uuid crop_node = matrix_uuid(31);
        const Uuid ellipsoid_node = matrix_uuid(32);
        const Uuid camera_node = matrix_uuid(33);
        const Uuid checkpoint_uuid = matrix_uuid(34);

        auto document = ProjectDocument::create(project_uuid, 100);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(document.error());

        ProjectManifest manifest{
            .application_name = "LichtFeld Matrix Proof",
            .application_version = {2, 7, 3},
            .schema_version = {2, 0, 0},
            .minimum_reader_version = {1, 4, 0},
            .minimum_safe_writer_version = {1, 8, 0},
            .required_capabilities = {"p3-core", "retained-json"},
            .optional_capabilities = {"future-capability"},
        };
        auto& project = document->edit_project();
        require_ok(project.set_manifest(manifest));
        const std::array lineage{matrix_uuid(40), matrix_uuid(41)};
        require_ok(project.set_project_lineage(lineage));
        require_ok(project.set_dataset_reference(dataset_ref));
        const ProjectGeoreference georeference{
            .crs = "EPSG:2056",
            .world_origin = {2'600'000.25, 1'200'000.5, 450.125},
            .world_unit_scale = 0.01,
            .world_origin_provenance =
                WorldOriginProvenance::CentralizeByPointCloud,
        };
        require_ok(project.set_georeference(georeference));

        std::vector<ReferenceRecord> expected_references;
        const auto add_reference =
            [&](const Uuid& uuid, std::string key, std::string kind,
                std::string preferred, const LocatorBase base,
                const FingerprintKind fingerprint_kind,
                const std::uint8_t tag, const bool unresolved = false) {
                ReferenceRecord record{
                    .uuid = uuid,
                    .key = std::move(key),
                    .kind = std::move(kind),
                    .locator =
                        {
                            .preferred = std::move(preferred),
                            .base = base,
                            .absolute_fallback =
                                std::format("/fallback/{}", tag),
                        },
                    .fingerprint =
                        matrix_fingerprint(tag, fingerprint_kind),
                    .unresolved = unresolved,
                };
                expected_references.push_back(record);
                require_ok(document->edit_references().upsert(record));
            };
        add_reference(dataset_ref, "dataset.root", "dataset",
                      "../dataset", LocatorBase::Project,
                      FingerprintKind::Directory, 1);
        add_reference(colmap_ref, "dataset.colmap", "colmap",
                      "sparse/0", LocatorBase::Dataset,
                      FingerprintKind::Directory, 2);
        add_reference(rad_ref, "splat.live_rad", "rad",
                      "assets/live.rad", LocatorBase::Project,
                      FingerprintKind::File, 3);
        add_reference(rad_meta_ref, "splat.live_rad.meta",
                      "rad_meta_cache", "assets/live.rad.meta",
                      LocatorBase::Project, FingerprintKind::File, 4,
                      true);
        add_reference(background_ref, "training.background",
                      "background_image", "images/background.png",
                      LocatorBase::Dataset, FingerprintKind::File, 5);
        add_reference(environment_ref, "view.environment",
                      "environment_map", "assets/studio.hdr",
                      LocatorBase::Project, FingerprintKind::File, 6);
        add_reference(sequence_ref, "sequence.directory",
                      "ply_sequence", "sequences/review",
                      LocatorBase::Project, FingerprintKind::Directory, 7);
        add_reference(ppisp_ref, "training.ppisp",
                      "ppisp_sidecar", "appearance/model.ppisp",
                      LocatorBase::Project, FingerprintKind::File, 8);

        const std::array<float, 16> edited_transform{
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            3,
            4,
            5,
            1,
        };
        std::vector<SceneNodeRecord> expected_nodes;
        const auto add_node = [&](SceneNodeRecord node) {
            expected_nodes.push_back(node);
            require_ok(document->edit_scene_graph().upsert_node(node));
        };
        add_node(SceneNodeRecord{
            .uuid = root_node,
            .type = "group",
            .name = "Root group",
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = dataset_node,
            .type = "dataset",
            .name = "Dataset",
            .parent_uuid = root_node,
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = camera_group_node,
            .type = "camera_group",
            .name = "Cameras",
            .parent_uuid = dataset_node,
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = image_group_node,
            .type = "image_group",
            .name = "Train images",
            .parent_uuid = camera_group_node,
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = image_node,
            .type = "image",
            .name = "Image placeholder",
            .parent_uuid = image_group_node,
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = sequence_node,
            .type = "ply_sequence",
            .name = "Review sequence",
            .parent_uuid = root_node,
            .child_order = 1,
        });
        add_node(SceneNodeRecord{
            .uuid = training_node,
            .type = "splat",
            .name = "Training model",
            .parent_uuid = root_node,
            .child_order = 2,
            .payload =
                PayloadBinding{
                    .fourcc = "CKPT",
                    .instance_uuid = checkpoint_uuid,
                    .source_kind = "checkpoint",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = imported_node,
            .type = "splat",
            .name = "Edited imported splat",
            .parent_uuid = root_node,
            .child_order = 3,
            .local_transform = edited_transform,
            .visible = false,
            .locked = true,
            .payload_diverged = true,
            .payload =
                PayloadBinding{
                    .fourcc = "SPLT",
                    .instance_uuid = imported_node,
                    .source_kind = "spz",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = point_node,
            .type = "pointcloud",
            .name = "Survey points",
            .parent_uuid = root_node,
            .child_order = 4,
            .georef_pose =
                GeorefPose{
                    .rotation = {0.9238795325, 0.0, 0.3826834324, 0.0},
                    .translation = {1000.25, 2000.5, -10.75},
                },
            .payload =
                PayloadBinding{
                    .fourcc = "PCLD",
                    .instance_uuid = point_node,
                    .source_kind = "ply",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = mesh_node,
            .type = "mesh",
            .name = "Textured mesh",
            .parent_uuid = root_node,
            .child_order = 5,
            .payload =
                PayloadBinding{
                    .fourcc = "MESH",
                    .instance_uuid = mesh_node,
                    .source_kind = "obj",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = live_rad_node,
            .type = "splat",
            .name = "Live RAD",
            .parent_uuid = root_node,
            .child_order = 6,
            .payload =
                PayloadBinding{
                    .fourcc = "REFS",
                    .instance_uuid = rad_ref,
                    .reference_uuid = rad_ref,
                    .source_kind = "rad",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = crop_node,
            .type = "cropbox",
            .name = "Crop region",
            .parent_uuid = root_node,
            .child_order = 7,
            .cropbox =
                CropBoxRecord{
                    .min = {-2.0f, -3.0f, -4.0f},
                    .max = {5.0f, 6.0f, 7.0f},
                    .inverse = true,
                    .enabled = true,
                    .color = {0.1f, 0.2f, 0.3f},
                    .line_width = 3.5f,
                },
        });
        add_node(SceneNodeRecord{
            .uuid = ellipsoid_node,
            .type = "ellipsoid",
            .name = "Ellipsoid region",
            .parent_uuid = root_node,
            .child_order = 8,
            .ellipsoid =
                EllipsoidRecord{
                    .radii = {2.0f, 3.0f, 4.0f},
                    .inverse = true,
                    .enabled = true,
                    .color = {0.4f, 0.5f, 0.6f},
                    .line_width = 4.5f,
                },
        });
        const CameraRecord camera{
            .uid = 71,
            .camera_id = 72,
            .rotation = {1, 0, 0, 0, 0, -1, 0, 1, 0},
            .translation = {1.25f, 2.5f, 3.75f},
            .focal_x = 1200.5f,
            .focal_y = 1199.5f,
            .center_x = 640.25f,
            .center_y = 360.75f,
            .radial_distortion = {0.1f, -0.01f, 0.001f},
            .tangential_distortion = {0.002f, -0.003f},
            .camera_model_type = 4,
            .camera_width = 1280,
            .camera_height = 720,
            .image_width = 1920,
            .image_height = 1080,
            .image_name = "frame_0071.png",
            .image_path = "images/frame_0071.png",
            .mask_path = "masks/frame_0071.png",
            .depth_path = "depth/frame_0071.png",
            .normal_path = "normals/frame_0071.png",
            .has_alpha = true,
            .split = "eval",
        };
        add_node(SceneNodeRecord{
            .uuid = camera_node,
            .type = "camera",
            .name = "Evaluation camera",
            .parent_uuid = image_group_node,
            .child_order = 1,
            .training_enabled = false,
            .camera = camera,
        });
        require_ok(
            document->edit_scene_graph().set_training_model_uuid(
                training_node));

        auto splat = SplatChapterPayload::capture(
            *make_matrix_splat(), SplatSourceKind::ImportedSpz, false);
        ASSERT_TRUE(splat)
            << lfs::format_for_developer(splat.error());
        require_ok(document->set_splat(imported_node, std::move(*splat)));
        require_ok(document->set_point_cloud(
            point_node, make_matrix_point_cloud()));
        require_ok(document->set_mesh(mesh_node, make_matrix_mesh()));

        for (const auto& [node_uuid, fourcc, locator, tag] :
             std::array{
                 std::tuple{imported_node, std::string{"SPLT"},
                            std::string{"imports/source.spz"},
                            std::uint8_t{11}},
                 std::tuple{point_node, std::string{"PCLD"},
                            std::string{"imports/points.ply"},
                            std::uint8_t{12}},
                 std::tuple{mesh_node, std::string{"MESH"},
                            std::string{"imports/mesh.obj"},
                            std::uint8_t{13}},
             }) {
            require_ok(project.upsert_embed_decision(EmbedDecision{
                .uuid = node_uuid,
                .node_uuid = node_uuid,
                .payload_fourcc = fourcc,
                .decision = "embedded",
                .reason = "matrix proof embedded payload",
            }));
            require_ok(project.upsert_embedded_payload_provenance(
                EmbeddedPayloadProvenance{
                    .uuid = node_uuid,
                    .node_uuid = node_uuid,
                    .fourcc = fourcc,
                    .import_locator =
                        {
                            .preferred = locator,
                            .base = LocatorBase::Project,
                        },
                    .import_fingerprint =
                        matrix_fingerprint(tag),
                    .content_xxh3_128 = {},
                }));
        }
        require_ok(project.upsert_embed_decision(EmbedDecision{
            .uuid = live_rad_node,
            .node_uuid = live_rad_node,
            .payload_fourcc = "REFS",
            .decision = "external",
            .reference_uuid = rad_ref,
            .reason = "live RAD remains external",
        }));
        require_ok(project.upsert_provenance(ProvenanceRecord{
            .uuid = matrix_uuid(50),
            .kind = "legacy_view_path",
            .value = "imports/source.spz",
        }));
        require_ok(project.upsert_provenance(ProvenanceRecord{
            .uuid = matrix_uuid(51),
            .kind = "mesh_texture_sources",
            .value =
                "textures/albedo.png;textures/normal.png;textures/mr.png",
        }));

        auto& selection = document->edit_selection();
        require_ok(selection.set_groups(
            {
                lfs::core::SelectionGroup{
                    .id = 3,
                    .name = "Review",
                    .color = {0.2f, 0.4f, 0.8f},
                    .locked = false,
                },
                lfs::core::SelectionGroup{
                    .id = 7,
                    .name = "Protected",
                    .color = {0.9f, 0.1f, 0.3f},
                    .locked = true,
                },
            },
            7, 8));
        require_ok(selection.upsert_slice(SelectionMaskSlice{
            .node_uuid = imported_node,
            .domain = lfs::core::SelectionDomain::Splat,
            .encoding = SelectionMaskEncoding::DeltaBitpack,
            .mask = {3, 0, 7, 3},
        }));
        require_ok(selection.upsert_slice(SelectionMaskSlice{
            .node_uuid = point_node,
            .domain = lfs::core::SelectionDomain::PointCloud,
            .encoding = SelectionMaskEncoding::DeltaBitpack,
            .mask = {0, 7},
        }));
        require_ok(selection.set_selected_node_uuids(
            {point_node, imported_node, camera_node}));

        ParameterManagerSnapshot parameters;
        parameters.active_strategy =
            std::string(lfs::core::param::kStrategyIGSPlus);
        parameters.mcmc_session =
            distinct_pending_parameters(
                lfs::core::param::kStrategyMCMC, 1);
        parameters.mrnf_session =
            distinct_pending_parameters(
                lfs::core::param::kStrategyMRNF, 2);
        parameters.igs_session =
            distinct_pending_parameters(
                lfs::core::param::kStrategyIGSPlus, 3);
        parameters.mcmc_current =
            distinct_pending_parameters(
                lfs::core::param::kStrategyMCMC, 4);
        parameters.mrnf_current =
            distinct_pending_parameters(
                lfs::core::param::kStrategyMRNF, 5);
        parameters.igs_current =
            distinct_pending_parameters(
                lfs::core::param::kStrategyIGSPlus, 6);
        parameters.mcmc_session_references
            .background_image_reference = background_ref;
        parameters.mcmc_session_references.ppisp_reference =
            ppisp_ref;
        parameters.mrnf_session_references
            .background_image_reference = background_ref;
        parameters.mrnf_session_references.ppisp_reference =
            ppisp_ref;
        parameters.igs_session_references
            .background_image_reference = background_ref;
        parameters.igs_session_references.ppisp_reference =
            ppisp_ref;
        parameters.mcmc_current_references.ppisp_reference =
            ppisp_ref;
        parameters.mrnf_current_references
            .background_image_reference = background_ref;
        parameters.mrnf_current_references.ppisp_reference =
            ppisp_ref;
        parameters.igs_current_references
            .background_image_reference = background_ref;
        parameters.igs_current_references.ppisp_reference =
            ppisp_ref;
        parameters.mcmc_current_references
            .background_image_reference = background_ref;
        parameters.dataset.images = "images_matrix";
        parameters.dataset.resize_factor = 4;
        parameters.dataset.test_every = 11;
        parameters.dataset.timelapse_images = {"frame_a.png", "frame_b.png"};
        parameters.dataset.timelapse_every = 77;
        parameters.dataset.max_width = 2048;
        parameters.dataset.min_track_length = 5;
        parameters.dataset.invert_masks = true;
        parameters.dataset.mask_threshold = 0.625f;
        parameters.dataset.centralize_dataset = "cameras";
        parameters.dataset.loading_params.use_cpu_memory = false;
        parameters.dataset.loading_params.min_cpu_free_memory_ratio = 0.25f;
        parameters.dataset.loading_params.min_cpu_free_GB = 3.5f;
        parameters.dataset.loading_params.use_fs_cache = false;
        parameters.dataset.loading_params.print_cache_status = false;
        parameters.dataset.loading_params.print_status_freq_num = 123;
        parameters.dataset.loading_params.use_16bit_color = true;
        require_ok(document->edit_parameters().set_snapshot(parameters));

        auto first_save =
            document->save(path, matrix_save_options(100, 200));
        ASSERT_TRUE(first_save)
            << lfs::format_for_developer(first_save.error());

        auto first_open = ProjectDocument::open(path);
        ASSERT_TRUE(first_open)
            << lfs::format_for_developer(first_open.error());
        const auto first_refs_bytes = first_open->references().to_bytes();
        const auto first_scng_bytes = first_open->scene_graph().to_bytes();
        const auto first_selm_bytes =
            encode_selection_chapter(first_open->selection());
        ASSERT_TRUE(first_selm_bytes);
        const auto first_prms_bytes = first_open->parameters().to_bytes();
        const auto* first_splat = first_open->find_splat(imported_node);
        const auto* first_point =
            first_open->find_point_cloud(point_node);
        const auto* first_mesh = first_open->find_mesh(mesh_node);
        ASSERT_NE(first_splat, nullptr);
        ASSERT_NE(first_point, nullptr);
        ASSERT_NE(first_mesh, nullptr);
        const std::vector<std::byte> first_splat_bytes(
            first_splat->bytes().begin(), first_splat->bytes().end());
        const auto first_point_bytes =
            encode_point_cloud_payload(*first_point);
        const auto first_mesh_bytes = encode_mesh_payload(*first_mesh);
        ASSERT_TRUE(first_point_bytes);
        ASSERT_TRUE(first_mesh_bytes);

        auto second_save =
            first_open->save(path, matrix_save_options(110, 300));
        ASSERT_TRUE(second_save)
            << lfs::format_for_developer(second_save.error());
        EXPECT_EQ(second_save->rewritten_chunks, 1u);
        EXPECT_EQ(second_save->reused_chunks, 7u);

        auto second_open = ProjectDocument::open(path);
        ASSERT_TRUE(second_open)
            << lfs::format_for_developer(second_open.error());
        std::set<std::string> proven;
        const auto prove = [&](const std::string_view row) {
            EXPECT_TRUE(proven.emplace(row).second) << row;
        };

        auto second_manifest = second_open->project().manifest();
        ASSERT_TRUE(second_manifest);
        EXPECT_EQ(*second_manifest, manifest);
        prove("PROJ-39");
        auto second_project_uuid = second_open->project().project_uuid();
        ASSERT_TRUE(second_project_uuid);
        EXPECT_EQ(*second_project_uuid, project_uuid);
        prove("PROJ-40");
        auto created = second_open->project().created_at_unix_ns();
        auto modified = second_open->project().modified_at_unix_ns();
        ASSERT_TRUE(created);
        ASSERT_TRUE(modified);
        EXPECT_EQ(*created, 100u);
        EXPECT_EQ(*modified, 300u);
        prove("PROJ-41");
        auto reopened_dataset =
            second_open->project().dataset_reference();
        ASSERT_TRUE(reopened_dataset);
        EXPECT_EQ(*reopened_dataset, dataset_ref);
        prove("PROJ-42");
        auto decisions = second_open->project().embed_decisions();
        ASSERT_TRUE(decisions);
        EXPECT_EQ(decisions->size(), 4u);
        EXPECT_EQ(std::ranges::count_if(
                      *decisions,
                      [](const EmbedDecision& item) {
                          return item.decision == "embedded";
                      }),
                  3);
        EXPECT_EQ(std::ranges::count_if(
                      *decisions,
                      [](const EmbedDecision& item) {
                          return item.decision == "external";
                      }),
                  1);
        prove("PROJ-43");
        auto source_provenance = second_open->project().provenance();
        auto embedded_provenance =
            second_open->project().embedded_payload_provenance();
        ASSERT_TRUE(source_provenance);
        ASSERT_TRUE(embedded_provenance);
        EXPECT_EQ(source_provenance->size(), 2u);
        ASSERT_EQ(embedded_provenance->size(), 3u);
        EXPECT_TRUE(std::ranges::all_of(
            *embedded_provenance,
            [](const EmbeddedPayloadProvenance& item) {
                return !item.import_locator.preferred.empty() &&
                       item.import_fingerprint.full_xxh3.has_value() &&
                       !item.content_xxh3_128.is_zero();
            }));
        prove("PROJ-44");
        auto reopened_georeference =
            second_open->project().georeference();
        ASSERT_TRUE(reopened_georeference);
        EXPECT_EQ(*reopened_georeference, georeference);
        prove("PROJ-45");

        EXPECT_EQ(second_open->references().to_bytes(),
                  first_refs_bytes);
        auto reopened_references =
            second_open->references().records();
        ASSERT_TRUE(reopened_references);
        EXPECT_EQ(*reopened_references, expected_references);
        prove("REFS-96");
        const auto prove_reference_records =
            [&](const std::string_view row,
                const std::initializer_list<std::string_view>
                    kinds) {
                const auto selected =
                    [&](const ReferenceRecord& record) {
                        return std::ranges::find(
                                   kinds, record.kind) !=
                               kinds.end();
                    };
                std::vector<ReferenceRecord> expected;
                std::vector<ReferenceRecord> actual;
                std::ranges::copy_if(
                    expected_references,
                    std::back_inserter(expected), selected);
                std::ranges::copy_if(
                    *reopened_references,
                    std::back_inserter(actual), selected);
                EXPECT_FALSE(expected.empty()) << row;
                EXPECT_EQ(actual, expected) << row;
                prove(row);
            };
        prove_reference_records("REFS-97", {"dataset"});
        prove_reference_records("REFS-98", {"colmap"});
        prove_reference_records(
            "REFS-99", {"rad", "rad_meta_cache"});
        prove_reference_records(
            "REFS-100", {"background_image"});
        prove_reference_records(
            "REFS-101", {"environment_map"});
        prove_reference_records(
            "REFS-102", {"ply_sequence"});

        EXPECT_EQ(second_open->scene_graph().to_bytes(),
                  first_scng_bytes);
        auto reopened_nodes = second_open->scene_graph().nodes();
        ASSERT_TRUE(reopened_nodes);
        EXPECT_EQ(*reopened_nodes, expected_nodes);
        EXPECT_TRUE(std::ranges::all_of(
            *reopened_nodes,
            [](const SceneNodeRecord& node) {
                return !node.uuid.is_nil();
            }));
        prove("SCNG-64");
        EXPECT_TRUE(std::ranges::all_of(
            *reopened_nodes,
            [](const SceneNodeRecord& node) {
                return !node.type.empty() && !node.name.empty();
            }));
        prove("SCNG-65");
        const auto reopened_imported =
            second_open->scene_graph().find(imported_node);
        const auto reopened_point =
            second_open->scene_graph().find(point_node);
        const auto reopened_crop =
            second_open->scene_graph().find(crop_node);
        const auto reopened_ellipsoid =
            second_open->scene_graph().find(ellipsoid_node);
        const auto reopened_camera =
            second_open->scene_graph().find(camera_node);
        ASSERT_TRUE(reopened_imported && *reopened_imported);
        ASSERT_TRUE(reopened_point && *reopened_point);
        ASSERT_TRUE(reopened_crop && *reopened_crop);
        ASSERT_TRUE(reopened_ellipsoid && *reopened_ellipsoid);
        ASSERT_TRUE(reopened_camera && *reopened_camera);
        EXPECT_EQ((*reopened_imported)->parent_uuid, root_node);
        EXPECT_EQ((*reopened_imported)->child_order, 3u);
        prove("SCNG-66");
        EXPECT_EQ((*reopened_imported)->local_transform,
                  edited_transform);
        prove("SCNG-67");
        ASSERT_TRUE((*reopened_point)->georef_pose);
        EXPECT_EQ((*reopened_point)->georef_pose->translation,
                  (std::array<double, 3>{1000.25, 2000.5, -10.75}));
        prove("SCNG-68");
        EXPECT_FALSE((*reopened_imported)->visible);
        EXPECT_TRUE((*reopened_imported)->locked);
        prove("SCNG-69");
        EXPECT_TRUE((*reopened_imported)->payload_diverged);
        prove("SCNG-70");
        auto reopened_training =
            second_open->scene_graph().training_model_uuid();
        ASSERT_TRUE(reopened_training);
        EXPECT_EQ(*reopened_training, training_node);
        prove("SCNG-71");
        EXPECT_FALSE((*reopened_camera)->training_enabled);
        prove("SCNG-72");
        ASSERT_TRUE((*reopened_crop)->cropbox);
        EXPECT_EQ((*reopened_crop)->cropbox,
                  expected_nodes[11].cropbox);
        prove("SCNG-73");
        ASSERT_TRUE((*reopened_ellipsoid)->ellipsoid);
        EXPECT_EQ((*reopened_ellipsoid)->ellipsoid,
                  expected_nodes[12].ellipsoid);
        prove("SCNG-74");
        ASSERT_TRUE((*reopened_camera)->camera);
        EXPECT_EQ((*reopened_camera)->camera->uid, 71);
        EXPECT_EQ((*reopened_camera)->camera->camera_id, 72);
        EXPECT_EQ((*reopened_camera)->camera->rotation,
                  camera.rotation);
        EXPECT_FLOAT_EQ((*reopened_camera)->camera->focal_x,
                        1200.5f);
        EXPECT_EQ((*reopened_camera)->camera->radial_distortion,
                  camera.radial_distortion);
        prove("SCNG-75");
        EXPECT_EQ((*reopened_camera)->camera->image_name,
                  "frame_0071.png");
        EXPECT_EQ((*reopened_camera)->camera->image_path,
                  "images/frame_0071.png");
        EXPECT_EQ((*reopened_camera)->camera->normal_path,
                  "normals/frame_0071.png");
        EXPECT_TRUE((*reopened_camera)->camera->has_alpha);
        EXPECT_EQ((*reopened_camera)->camera->split, "eval");
        prove("SCNG-76");
        const std::set<std::string> structural_types{
            "group", "dataset", "camera_group", "image_group",
            "image", "ply_sequence"};
        EXPECT_TRUE(std::ranges::all_of(
            structural_types,
            [&](const std::string& type) {
                return std::ranges::any_of(
                    *reopened_nodes,
                    [&](const SceneNodeRecord& node) {
                        return node.type == type;
                    });
            }));
        EXPECT_EQ((*reopened_imported)->payload->fourcc, "SPLT");
        EXPECT_EQ((*reopened_point)->payload->fourcc, "PCLD");
        prove("SCNG-77");

        auto second_selm_bytes =
            encode_selection_chapter(second_open->selection());
        ASSERT_TRUE(second_selm_bytes);
        EXPECT_EQ(*second_selm_bytes, *first_selm_bytes);
        ASSERT_EQ(second_open->selection().groups().size(), 2u);
        EXPECT_EQ(second_open->selection().groups()[1].name,
                  "Protected");
        EXPECT_TRUE(second_open->selection().groups()[1].locked);
        prove("SELM-83");
        EXPECT_EQ(second_open->selection().active_group_id(), 7u);
        EXPECT_EQ(second_open->selection().next_group_id(), 8u);
        prove("SELM-84");
        ASSERT_EQ(second_open->selection().slices().size(), 2u);
        EXPECT_TRUE(std::ranges::any_of(
            second_open->selection().slices(),
            [](const SelectionMaskSlice& slice) {
                return slice.domain ==
                           lfs::core::SelectionDomain::Splat &&
                       slice.mask == std::vector<std::uint8_t>{3, 0, 7, 3};
            }));
        EXPECT_TRUE(std::ranges::any_of(
            second_open->selection().slices(),
            [](const SelectionMaskSlice& slice) {
                return slice.domain ==
                           lfs::core::SelectionDomain::PointCloud &&
                       slice.mask == std::vector<std::uint8_t>{0, 7};
            }));
        prove("SELM-85");
        EXPECT_EQ(second_open->selection().selected_node_uuids(),
                  (std::vector<Uuid>{
                      point_node, imported_node, camera_node}));
        prove("SELM-86");

        EXPECT_EQ(second_open->parameters().to_bytes(),
                  first_prms_bytes);
        auto reopened_parameters =
            second_open->parameters().snapshot();
        ASSERT_TRUE(reopened_parameters);
        EXPECT_EQ(reopened_parameters->active_strategy,
                  lfs::core::param::kStrategyIGSPlus);
        prove("PRMS-53");
        struct PendingRoleProof {
            std::string_view path;
            const lfs::core::param::OptimizationParameters*
                expected_parameters;
            const lfs::core::param::OptimizationParameters*
                actual_parameters;
            const ParameterManagerSnapshot::ReferenceBindings*
                expected_references;
            const ParameterManagerSnapshot::ReferenceBindings*
                actual_references;
        };
        const std::array pending_roles{
            PendingRoleProof{
                "presets.mcmc.session",
                &parameters.mcmc_session,
                &reopened_parameters->mcmc_session,
                &parameters.mcmc_session_references,
                &reopened_parameters->mcmc_session_references,
            },
            PendingRoleProof{
                "presets.mrnf.session",
                &parameters.mrnf_session,
                &reopened_parameters->mrnf_session,
                &parameters.mrnf_session_references,
                &reopened_parameters->mrnf_session_references,
            },
            PendingRoleProof{
                "presets.igs+.session",
                &parameters.igs_session,
                &reopened_parameters->igs_session,
                &parameters.igs_session_references,
                &reopened_parameters->igs_session_references,
            },
            PendingRoleProof{
                "presets.mcmc.current",
                &parameters.mcmc_current,
                &reopened_parameters->mcmc_current,
                &parameters.mcmc_current_references,
                &reopened_parameters->mcmc_current_references,
            },
            PendingRoleProof{
                "presets.mrnf.current",
                &parameters.mrnf_current,
                &reopened_parameters->mrnf_current,
                &parameters.mrnf_current_references,
                &reopened_parameters->mrnf_current_references,
            },
            PendingRoleProof{
                "presets.igs+.current",
                &parameters.igs_current,
                &reopened_parameters->igs_current,
                &parameters.igs_current_references,
                &reopened_parameters->igs_current_references,
            },
        };
        for (const auto& role : pending_roles) {
            SCOPED_TRACE(role.path);
            const auto serialized =
                second_open->parameters().dom().get_json(
                    role.path);
            ASSERT_TRUE(serialized);
            const auto expected_fields =
                pending_parameter_field_map(
                    *role.expected_parameters,
                    *role.expected_references);
            EXPECT_EQ(*serialized, expected_fields);
            std::set<std::string> serialized_fields;
            for (const auto& [field, ignored] :
                 serialized->items()) {
                (void)ignored;
                serialized_fields.insert(field);
            }
            const std::set<std::string> expected_field_inventory(
                PENDING_OPTIMIZATION_PARAMETER_FIELDS.begin(),
                PENDING_OPTIMIZATION_PARAMETER_FIELDS.end());
            EXPECT_EQ(serialized_fields,
                      expected_field_inventory);
            EXPECT_EQ(
                pending_parameter_field_map(
                    *role.actual_parameters,
                    *role.actual_references),
                expected_fields);
            EXPECT_EQ(*role.actual_references,
                      *role.expected_references);
            for (const auto& exclusion :
                 PENDING_PARAMETER_EXCLUSIONS) {
                SCOPED_TRACE(exclusion.phase);
                EXPECT_FALSE(
                    serialized->contains(exclusion.field))
                    << exclusion.field;
            }
        }
        auto reverse_index =
            second_open->reverse_reference_index();
        ASSERT_TRUE(reverse_index);
        EXPECT_TRUE(std::ranges::any_of(
            reverse_index->at(background_ref),
            [](const ReferenceOwnerBinding& binding) {
                return binding.chapter == "PRMS" &&
                       binding.field ==
                           "presets.mrnf.current."
                           "background_image_reference_uuid";
            }));
        EXPECT_TRUE(std::ranges::any_of(
            reverse_index->at(ppisp_ref),
            [](const ReferenceOwnerBinding& binding) {
                return binding.chapter == "PRMS" &&
                       binding.field ==
                           "presets.mrnf.session."
                           "ppisp_reference_uuid";
            }));
        prove("PRMS-54");
        EXPECT_EQ(reopened_parameters->dataset.images,
                  "images_matrix");
        EXPECT_EQ(reopened_parameters->dataset.timelapse_images,
                  (std::vector<std::string>{
                      "frame_a.png", "frame_b.png"}));
        EXPECT_EQ(reopened_parameters->dataset.centralize_dataset,
                  "cameras");
        EXPECT_TRUE(reopened_parameters->dataset.invert_masks);
        prove("PRMS-55");
        EXPECT_FALSE(
            reopened_parameters->dataset.loading_params.use_cpu_memory);
        EXPECT_FLOAT_EQ(
            reopened_parameters->dataset.loading_params.min_cpu_free_GB,
            3.5f);
        EXPECT_FALSE(
            reopened_parameters->dataset.loading_params.use_fs_cache);
        EXPECT_EQ(
            reopened_parameters->dataset.loading_params
                .print_status_freq_num,
            123);
        EXPECT_TRUE(
            reopened_parameters->dataset.loading_params.use_16bit_color);
        prove("PRMS-56");

        const auto* second_splat =
            second_open->find_splat(imported_node);
        ASSERT_NE(second_splat, nullptr);
        EXPECT_TRUE(std::ranges::equal(
            second_splat->bytes(), first_splat_bytes));
        auto hydrated_splat = second_splat->hydrate();
        ASSERT_TRUE(hydrated_splat)
            << lfs::format_for_developer(hydrated_splat.error());
        EXPECT_EQ(second_splat->lfsp_version(), 4u);
        EXPECT_EQ((*hydrated_splat)->get_active_sh_degree(), 0);
        EXPECT_EQ((*hydrated_splat)->get_max_sh_degree(), 1);
        EXPECT_FLOAT_EQ((*hydrated_splat)->get_scene_scale(), 2.5f);
        prove("SPLT-110");
        EXPECT_EQ((*hydrated_splat)->means().shape().str(), "[4, 3]");
        EXPECT_FLOAT_EQ(
            (*hydrated_splat)->means().cpu().ptr<float>()[0], 1.0f);
        prove("SPLT-111");
        EXPECT_EQ((*hydrated_splat)->sh0().shape().str(),
                  "[4, 1, 3]");
        EXPECT_TRUE((*hydrated_splat)->shN().is_valid());
        prove("SPLT-112");
        EXPECT_EQ((*hydrated_splat)->scaling_raw().shape().str(),
                  "[4, 3]");
        EXPECT_EQ((*hydrated_splat)->rotation_raw().shape().str(),
                  "[4, 4]");
        EXPECT_EQ((*hydrated_splat)->opacity_raw().shape().str(),
                  "[4, 1]");
        prove("SPLT-113");
        EXPECT_TRUE((*hydrated_splat)->deleted().is_valid());
        EXPECT_EQ(
            (*hydrated_splat)->deleted().cpu().to_vector_uint8(),
            (std::vector<std::uint8_t>{0, 1, 0, 1}));
        prove("SPLT-114");
        EXPECT_EQ(
            (*hydrated_splat)->_densification_info.shape().str(),
            "[2, 4]");
        prove("SPLT-115");
        ASSERT_EQ((*hydrated_splat)->frozen_ranges().size(), 1u);
        EXPECT_EQ((*hydrated_splat)->frozen_ranges()[0].start, 1u);
        EXPECT_EQ((*hydrated_splat)->frozen_ranges()[0].count, 2u);
        prove("SPLT-116");

        const auto* second_point =
            second_open->find_point_cloud(point_node);
        const auto* second_mesh =
            second_open->find_mesh(mesh_node);
        ASSERT_NE(second_point, nullptr);
        ASSERT_NE(second_mesh, nullptr);
        auto second_point_bytes =
            encode_point_cloud_payload(*second_point);
        auto second_mesh_bytes = encode_mesh_payload(*second_mesh);
        ASSERT_TRUE(second_point_bytes);
        ASSERT_TRUE(second_mesh_bytes);
        EXPECT_EQ(*second_point_bytes, *first_point_bytes);
        EXPECT_EQ(second_point->point_cloud()->means.shape().str(),
                  "[2, 3]");
        EXPECT_TRUE(second_point->point_cloud()->normals.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->sh0.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->shN.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->opacity.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->scaling.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->rotation.is_valid());
        EXPECT_EQ(second_point->point_cloud()->attribute_names.size(),
                  7u);
        const auto point_vendor = std::ranges::find(
            second_point->retained_properties(),
            std::string_view{"vendor_score"},
            &GeometryPropertyPlane::name);
        ASSERT_NE(point_vendor,
                  second_point->retained_properties().end());
        EXPECT_EQ(point_vendor->bytes,
                  byte_values({0x10, 0x20, 0x30, 0x40}));

        EXPECT_EQ(*second_mesh_bytes, *first_mesh_bytes);
        EXPECT_TRUE(second_mesh->mesh()->normals.is_valid());
        EXPECT_TRUE(second_mesh->mesh()->tangents.is_valid());
        EXPECT_TRUE(second_mesh->mesh()->texcoords.is_valid());
        EXPECT_TRUE(second_mesh->mesh()->colors.is_valid());
        ASSERT_EQ(second_mesh->mesh()->submeshes.size(), 1u);
        EXPECT_EQ(second_mesh->mesh()->submeshes[0].index_count, 6u);
        ASSERT_EQ(second_mesh->mesh()->materials.size(), 1u);
        EXPECT_EQ(second_mesh->mesh()->materials[0].name,
                  "matrix-plane");
        EXPECT_TRUE(second_mesh->mesh()->materials[0].double_sided);
        ASSERT_EQ(second_mesh->mesh()->texture_images.size(), 1u);
        EXPECT_EQ(second_mesh->mesh()->texture_images[0].pixels,
                  (std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6}));
        const auto mesh_vendor = std::ranges::find(
            second_mesh->retained_properties(),
            std::string_view{"vendor_ids"},
            &GeometryPropertyPlane::name);
        ASSERT_NE(mesh_vendor,
                  second_mesh->retained_properties().end());

        const MatrixRows matrix_rows = read_matrix_rows();
        const std::set<std::string> registered(
            P3_MATRIX_ROWS.begin(), P3_MATRIX_ROWS.end());
        EXPECT_EQ(matrix_rows.p3, registered)
            << "P3_MATRIX_ROWS must be updated whenever the normative "
               "ownership matrix gains or loses a P3 table row";
        EXPECT_EQ(proven, registered)
            << "Every registered P3 row must reach an explicit assertion "
               "after save->load->save";

        ASSERT_FALSE(matrix_rows.deferred.empty());
        for (const auto& [row_id, phase] : matrix_rows.deferred) {
            EXPECT_TRUE(phase == "P4" || phase == "P5")
                << row_id << " has no explicit later-phase tag";
        }

        // Gap #12 is normative prose rather than a Markdown table row. These
        // assertions are its machine-checkable coverage entries.
        EXPECT_EQ(point_vendor->encoding, 77u)
            << "PCLD-GAP12-242";
        EXPECT_EQ(mesh_vendor->encoding, 91u)
            << "MESH-GAP12-242";
    }

} // namespace

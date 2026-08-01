/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/uuid.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/session_chapters.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lfs::vis::project {
    struct PanelCameraProjectState;
}

namespace lfs::core {
    class MeshData;
    class PointCloud;
    class SplatData;
} // namespace lfs::core

namespace lfs::io::project {
    class ProjectDocument;
    struct ProjectDocumentSaveOptions;
} // namespace lfs::io::project

namespace lfs::test::licht {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;

    inline constexpr std::string_view P3_MATRIX_ROW_DATA =
        "PROJ-39 PROJ-40 PROJ-41 PROJ-42 PROJ-43 PROJ-44 PROJ-45 "
        "PRMS-53 PRMS-54 PRMS-55 PRMS-56 SCNG-64 SCNG-65 SCNG-66 SCNG-67 SCNG-68 "
        "SCNG-69 SCNG-70 SCNG-71 SCNG-72 SCNG-73 SCNG-74 SCNG-75 SCNG-76 SCNG-77 "
        "SELM-83 SELM-84 SELM-85 SELM-86 REFS-96 REFS-97 REFS-98 REFS-99 REFS-100 "
        "REFS-101 REFS-102 SPLT-110 SPLT-111 SPLT-112 SPLT-113 SPLT-114 SPLT-115 SPLT-116";
    inline constexpr std::string_view P4_CKPT_MATRIX_ROW_DATA =
        "CKPT-126 CKPT-127 CKPT-128 CKPT-129 CKPT-130 CKPT-131 CKPT-132 CKPT-133 "
        "CKPT-134 CKPT-135 CKPT-136 CKPT-137 CKPT-138 CKPT-139 CKPT-140 CKPT-141 "
        "CKPT-142 CKPT-143 CKPT-144 CKPT-145 CKPT-146 CKPT-147 CKPT-148 CKPT-149";
    inline constexpr std::string_view P4_PPIS_MATRIX_ROW_DATA =
        "PPIS-157 PPIS-158 PPIS-159 PPIS-160";
    inline constexpr std::string_view P5_MATRIX_ROW_DATA =
        "GUIL-166 GUIL-167 GUIL-168 GUIL-169 GUIL-170 GUIL-171 EDTR-179 EDTR-180 "
        "EDTR-181 EDTR-182 EDTR-183 EDTR-184 VIEW-192 VIEW-193 VIEW-194 VIEW-195 "
        "VIEW-196 VIEW-197 VIEW-198 VIEW-199 VIEW-200 VIEW-201 VIEW-202 VIEW-203 "
        "VIEW-204 VIEW-205 VIEW-206 VIEW-207 VIEW-208 VIEW-209 VIEW-210 SEQR-218 "
        "SEQR-219 SEQR-220 SEQR-221 SEQR-222 METR-230 METR-231 METR-232 METR-233";
    inline constexpr std::string_view OPTIMIZATION_PARAMETER_FIELD_DATA =
        "iterations means_lr means_lr_end shs_lr opacity_lr scaling_lr scaling_lr_end "
        "rotation_lr cropbox_lr_scale cropbox_loss_weight lambda_dssim min_opacity "
        "refine_every start_refine stop_refine grad_threshold sh_degree opacity_reg scale_reg "
        "init_opacity init_scaling max_cap eval_steps save_steps enable_eval "
        "enable_save_eval_images strategy mip_filter use_bilateral_grid bilateral_grid_X "
        "bilateral_grid_Y bilateral_grid_W bilateral_grid_lr tv_loss_weight use_ppisp ppisp_lr "
        "ppisp_reg_weight ppisp_warmup_steps ppisp_freeze_from_sidecar ppisp_reference_uuid "
        "ppisp_use_controller ppisp_freeze_gaussians_on_distill ppisp_controller_activation_step "
        "ppisp_controller_lr prune_opacity grow_scale3d grow_scale2d prune_scale3d prune_scale2d "
        "reset_every pause_refine_after_reset revised_opacity gut undistort steps_scaler "
        "sh_degree_interval random init_num_pts init_extent enable_sparsity sparsify_steps init_rho "
        "prune_ratio bg_modulation bg_mode bg_color background_image_reference_uuid mask_mode "
        "invert_masks mask_opacity_penalty_weight mask_opacity_penalty_power mask_threshold "
        "use_alpha_as_mask use_depth_loss depth_loss_weight depth_loss_mode use_normal_loss "
        "normal_loss_weight normal_consistency_weight normal_flatten_weight normal_loss_space "
        "growth_grad_threshold grow_fraction grow_until_iter opacity_decay scale_decay "
        "means_noise_weight bounds_percentile use_error_map use_edge_map";

    [[nodiscard]] inline std::set<std::string> word_set(const std::string_view words) {
        std::set<std::string> result;
        for (std::size_t begin = 0; begin < words.size();) {
            while (begin < words.size() && words[begin] == ' ') {
                ++begin;
            }
            const std::size_t end = words.find(' ', begin);
            if (begin < words.size()) {
                result.emplace(words.substr(begin, end - begin));
            }
            begin = end == std::string_view::npos ? words.size() : end + 1;
        }
        return result;
    }

    class TemporaryDirectory {
    public:
        explicit TemporaryDirectory(const std::string_view prefix = "lfs-licht-test") {
            static std::atomic_uint64_t counter{0};
            path = fs::temp_directory_path() /
                   std::format("{}-{}-{}", prefix,
                               std::chrono::steady_clock::now().time_since_epoch().count(),
                               counter.fetch_add(1));
            fs::create_directories(path);
        }

        ~TemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }

        TemporaryDirectory(const TemporaryDirectory&) = delete;
        TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

        fs::path path;
    };

    [[nodiscard]] inline core::Uuid fixed_uuid(const std::uint64_t tag) {
        const auto parsed = core::Uuid::from_string(
            std::format("{:08x}-0000-4000-8000-{:012x}", tag, tag));
        if (!parsed) {
            throw std::runtime_error("invalid deterministic test UUID");
        }
        return *parsed;
    }

    [[nodiscard]] inline core::Uuid fixed_uuid_in_namespace(
        const std::uint32_t namespace_tag, const std::uint64_t tag) {
        const auto parsed = core::Uuid::from_string(
            std::format("{:08x}-0000-4000-8000-{:012x}", namespace_tag, tag));
        if (!parsed) {
            throw std::runtime_error("invalid namespaced deterministic test UUID");
        }
        return *parsed;
    }

    [[nodiscard]] inline core::Uuid uuid_literal(const std::string_view text) {
        const auto parsed = core::Uuid::from_string(text);
        if (!parsed) {
            throw std::runtime_error("invalid UUID test literal");
        }
        return *parsed;
    }

    [[nodiscard]] inline ChunkKey fixed_key(const std::string_view fourcc,
                                            const std::uint64_t tag) {
        const auto parsed = Fourcc::from_string(fourcc);
        if (!parsed) {
            throw std::runtime_error("invalid test fourcc");
        }
        return {.fourcc = *parsed, .instance_uuid = fixed_uuid(tag)};
    }

    template <typename T>
    [[nodiscard]] T require_result(lfs::Result<T> result) {
        if (!result) {
            throw std::runtime_error(lfs::format_for_developer(result.error()));
        }
        return std::move(*result);
    }

    template <typename T>
    [[nodiscard]] std::unique_ptr<T> require_result_ptr(lfs::Result<T> result) {
        return std::make_unique<T>(require_result(std::move(result)));
    }

    inline void require_status(lfs::Result<void> result) {
        if (!result) {
            throw std::runtime_error(lfs::format_for_developer(result.error()));
        }
    }

    [[nodiscard]] inline std::vector<std::byte> byte_vector(const std::string_view text) {
        const auto view = std::as_bytes(std::span(text.data(), text.size()));
        return {view.begin(), view.end()};
    }

    [[nodiscard]] inline std::vector<std::byte> read_file_bytes(const fs::path& path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error(std::format("cannot read {}", path.string()));
        }
        const std::vector<char> raw{std::istreambuf_iterator<char>(stream),
                                    std::istreambuf_iterator<char>()};
        std::vector<std::byte> result(raw.size());
        if (!raw.empty()) {
            std::memcpy(result.data(), raw.data(), raw.size());
        }
        return result;
    }

    [[nodiscard]] inline io::JsonChapterDom::Json json_root(
        const io::JsonChapterDom& dom) {
        return io::JsonChapterDom::Json::parse(dom.dump());
    }

    inline void write_file_bytes(const fs::path& path,
                                 const std::span<const std::byte> contents) {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(contents.data()),
                     static_cast<std::streamsize>(contents.size()));
        if (!stream) {
            throw std::runtime_error(std::format("cannot write {}", path.string()));
        }
    }

    inline void write_u32_le(const std::span<std::byte> bytes, const std::size_t offset,
                             const std::uint32_t value) {
        if (offset + sizeof(value) > bytes.size()) {
            throw std::out_of_range("test u32 write exceeds payload");
        }
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            bytes[offset + index] = static_cast<std::byte>(value >> (index * 8));
        }
    }

    inline void write_u64_le(const std::span<std::byte> bytes, const std::size_t offset,
                             const std::uint64_t value) {
        if (offset + sizeof(value) > bytes.size()) {
            throw std::out_of_range("test u64 write exceeds payload");
        }
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            bytes[offset + index] = static_cast<std::byte>(value >> (index * 8));
        }
    }

    [[nodiscard]] inline std::vector<std::byte> hex_bytes(const std::string_view text) {
        const auto nibble = [](const char value) -> std::uint8_t {
            if (value >= '0' && value <= '9') {
                return static_cast<std::uint8_t>(value - '0');
            }
            if (value >= 'a' && value <= 'f') {
                return static_cast<std::uint8_t>(value - 'a' + 10);
            }
            throw std::invalid_argument("invalid test hex digit");
        };
        if (text.size() % 2 != 0) {
            throw std::invalid_argument("odd test hex string");
        }
        std::vector<std::byte> result(text.size() / 2);
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] = static_cast<std::byte>(
                (nibble(text[index * 2]) << 4) | nibble(text[index * 2 + 1]));
        }
        return result;
    }

    [[nodiscard]] inline std::vector<std::byte> one_pixel_png() {
        return hex_bytes(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c6360000000020001e527d4a20000000049454e44ae426082");
    }

    [[nodiscard]] inline std::vector<std::byte> read_file_range(
        const fs::path& path, const std::uint64_t offset, const std::size_t count) {
        std::ifstream input(path, std::ios::binary);
        input.seekg(static_cast<std::streamoff>(offset));
        std::vector<std::byte> result(count);
        input.read(reinterpret_cast<char*>(result.data()),
                   static_cast<std::streamsize>(result.size()));
        if (!input) {
            throw std::runtime_error(std::format("cannot read {} bytes from {} at 0x{:x}",
                                                 count, path.string(), offset));
        }
        return result;
    }

    inline void write_file_range(const fs::path& path, const std::uint64_t offset,
                                 const std::span<const std::byte> contents) {
        std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
        output.seekp(static_cast<std::streamoff>(offset));
        output.write(reinterpret_cast<const char*>(contents.data()),
                     static_cast<std::streamsize>(contents.size()));
        if (!output) {
            throw std::runtime_error(
                std::format("cannot write {} at 0x{:x}", path.string(), offset));
        }
    }

    template <typename WorkItem>
    void drain_work_queue(std::mutex& mutex, std::vector<WorkItem>& queue) {
        std::vector<WorkItem> pending;
        {
            std::lock_guard lock(mutex);
            pending.swap(queue);
        }
        for (auto& item : pending) {
            if (item.run) {
                item.run();
            }
        }
    }

    template <typename Predicate, typename Action>
    [[nodiscard]] bool wait_until(const std::chrono::milliseconds timeout,
                                  Predicate&& predicate, Action&& action) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!predicate() && std::chrono::steady_clock::now() < deadline) {
            action();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return predicate();
    }

#ifndef _WIN32
    template <typename ChildWork>
    [[nodiscard]] int run_child_process(ChildWork&& child_work,
                                        const int exception_exit = 126,
                                        const int normal_exit = 0) {
        const pid_t child = ::fork();
        if (child < 0) {
            throw std::runtime_error("fork failed in test driver");
        }
        if (child == 0) {
            try {
                child_work();
            } catch (...) {
                ::_exit(exception_exit);
            }
            ::_exit(normal_exit);
        }
        int status = 0;
        if (::waitpid(child, &status, 0) != child) {
            throw std::runtime_error("waitpid failed in test driver");
        }
        return status;
    }
#endif

    [[nodiscard]] inline ReferenceFingerprint fingerprint(
        const std::uint8_t tag, const FingerprintKind kind = FingerprintKind::File,
        const std::uint64_t size_base = 1000, const std::uint64_t time_base = 2000) {
        ReferenceFingerprint result;
        result.kind = kind;
        result.size = size_base + tag;
        result.mtime_unix_ns = time_base + tag;
        result.head_xxh3.bytes.fill(tag);
        result.tail_xxh3.bytes.fill(static_cast<std::uint8_t>(tag + 1));
        return result;
    }

    [[nodiscard]] inline CreateOptions deterministic_create_options(
        const std::uint64_t file_tag, const std::uint64_t creation_time_unix_ns,
        const ContainerRole role = ContainerRole::Master) {
        return {
            .project_uuid = fixed_uuid(1),
            .file_uuid = fixed_uuid(file_tag),
            .role = role,
            .creation_time_unix_ns = creation_time_unix_ns,
            .index_compression = IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    [[nodiscard]] inline CommitOptions deterministic_commit_options(
        const std::uint64_t commit_tag, const std::uint64_t snapshot_tag,
        const std::uint64_t wallclock_unix_ns,
        const CommitKind kind = CommitKind::Explicit) {
        return {
            .kind = kind,
            .commit_uuid = fixed_uuid(commit_tag),
            .snapshot_uuid = fixed_uuid(snapshot_tag),
            .wallclock_unix_ns = wallclock_unix_ns,
        };
    }

    [[nodiscard]] inline AppendOptions deterministic_append_options() {
        return {
            .compatibility = {},
            .index_compression = IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    [[nodiscard]] vis::project::PanelCameraProjectState rolled_panel_camera(float tag);
    [[nodiscard]] ProjectSessionChapters make_populated_session_chapters();

    struct PopulatedProjectFixture {
        PopulatedProjectFixture();
        ~PopulatedProjectFixture();
        PopulatedProjectFixture(PopulatedProjectFixture&&) noexcept;
        PopulatedProjectFixture& operator=(PopulatedProjectFixture&&) noexcept;

        std::unique_ptr<ProjectDocument> document;
        core::Uuid project_uuid;
        core::Uuid dataset_reference;
        core::Uuid background_reference;
        core::Uuid ppisp_reference;
        core::Uuid root_node;
        core::Uuid training_node;
        core::Uuid imported_node;
        core::Uuid point_node;
        core::Uuid mesh_node;
        core::Uuid crop_node;
        core::Uuid ellipsoid_node;
        core::Uuid camera_node;
        core::Uuid checkpoint_uuid;
        ProjectManifest manifest;
        ProjectGeoreference georeference;
        std::array<float, 16> edited_transform{};
        CameraRecord camera;
        std::vector<ReferenceRecord> references;
        std::vector<SceneNodeRecord> nodes;
        ParameterManagerSnapshot parameters;
    };

    [[nodiscard]] PopulatedProjectFixture make_populated_project_fixture();
    [[nodiscard]] std::unique_ptr<ProjectDocument>
    make_empty_document(core::Uuid project_uuid, std::uint64_t created_at_unix_ns = 100);
    [[nodiscard]] std::unique_ptr<core::SplatData> make_matrix_splat(bool cuda = false);
    [[nodiscard]] std::unique_ptr<core::SplatData> make_splat(std::size_t count);
    [[nodiscard]] std::shared_ptr<core::PointCloud> make_point_cloud(std::size_t count);
    [[nodiscard]] std::shared_ptr<core::MeshData> make_triangle_mesh();
    [[nodiscard]] ProjectDocumentSaveOptions deterministic_document_save_options(
        std::uint32_t uuid_namespace, std::uint64_t identity_tag,
        std::uint64_t wallclock_unix_ns);
    [[nodiscard]] std::vector<std::byte>
    byte_values(std::initializer_list<std::uint8_t> values);

} // namespace lfs::test::licht

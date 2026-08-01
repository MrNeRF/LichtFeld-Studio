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
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
            throw std::runtime_error("test u32 write exceeds payload");
        }
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            bytes[offset + index] = static_cast<std::byte>(value >> (index * 8));
        }
    }

    inline void write_u64_le(const std::span<std::byte> bytes, const std::size_t offset,
                             const std::uint64_t value) {
        if (offset + sizeof(value) > bytes.size()) {
            throw std::runtime_error("test u64 write exceeds payload");
        }
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            bytes[offset + index] = static_cast<std::byte>(value >> (index * 8));
        }
    }

    [[nodiscard]] std::vector<std::byte> one_pixel_png();

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

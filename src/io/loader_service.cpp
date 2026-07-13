/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loader_service.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "io/error.hpp"
#include "io/formats/rad.hpp"
#include "io/loaders/blender_loader.hpp"
#include "io/loaders/checkpoint_loader.hpp"
#include "io/loaders/colmap_loader.hpp"
#include "io/loaders/mesh_loader.hpp"
#include "io/loaders/ply_loader.hpp"
#include "io/loaders/rad_loader.hpp"
#include "io/loaders/sogs_loader.hpp"
#include "io/loaders/spz_loader.hpp"
#include "io/loaders/usd_loader.hpp"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <string>

namespace lfs::io {

    LoaderService::LoaderService()
        : registry_(std::make_unique<DataLoaderRegistry>()) {

        // Register default loaders
        registry_->registerLoader(std::make_unique<PLYLoader>());
        registry_->registerLoader(std::make_unique<SogLoader>());
        registry_->registerLoader(std::make_unique<SpzLoader>());
        registry_->registerLoader(std::make_unique<USDLoader>());
        registry_->registerLoader(std::make_unique<RadLoader>());
        registry_->registerLoader(std::make_unique<CheckpointLoader>());
        registry_->registerLoader(std::make_unique<ColmapLoader>());
        registry_->registerLoader(std::make_unique<BlenderLoader>());
        registry_->registerLoader(std::make_unique<MeshLoader>());

        LOG_DEBUG("LoaderService initialized with {} loaders", registry_->size());
    }

    namespace {
        [[nodiscard]] bool splat_tensor_renderer_ready(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid() || tensor.numel() == 0) {
                return true; // empty/absent — nothing to migrate
            }
            // Match what the Vulkan splat renderer actually binds (vksplat requires this kind).
            return tensor.is_external_storage() &&
                   tensor.external_storage_kind() == "vulkan_external_buffer";
        }

        [[nodiscard]] bool pagedRadGpuResidencyRequested(const lfs::core::SplatData& model) {
            return lfs::io::rad_paged_load_recommended(model);
        }
    } // namespace

    Result<void> migrateSplatTensorsToAllocator(lfs::core::SplatData& model,
                                                const SplatTensorAllocator& allocator) {
        if (!allocator) {
            return {};
        }
        if (pagedRadGpuResidencyRequested(model)) {
            model.set_tensor_allocator(allocator);
            LOG_INFO("RAD paged LOD active: skipping full renderer-storage migration (chunks={})",
                     model.lod_tree->chunk_count());
            return {};
        }
        if (splat_tensor_renderer_ready(model.means_raw()) &&
            splat_tensor_renderer_ready(model.sh0_raw()) &&
            splat_tensor_renderer_ready(model.scaling_raw()) &&
            splat_tensor_renderer_ready(model.rotation_raw()) &&
            splat_tensor_renderer_ready(model.opacity_raw()) &&
            splat_tensor_renderer_ready(model.shN_raw())) {
            model.set_tensor_allocator(allocator);
            return {};
        }

        try {
            const auto copy_to_allocator =
                [&](const lfs::core::Tensor& source, const std::string_view name) -> lfs::core::Tensor {
                lfs::core::Tensor source_contiguous = source.is_contiguous() ? source : source.contiguous();
                const auto& shape = source_contiguous.shape();
                const size_t capacity = shape.rank() > 0 ? shape[0] : source_contiguous.numel();
                lfs::core::Tensor dst = allocator(shape, capacity, source_contiguous.dtype(), name);
                dst.set_name(std::string{name});

                // Viewer splat tensors are read directly by Vulkan. Match the PLY
                // loader's known-good host-to-external upload path instead of
                // relying on CUDA device-to-device copies into imported Vk memory.
                if (dst.is_external_storage() &&
                    dst.external_storage_kind() == "vulkan_external_buffer" &&
                    source_contiguous.device() == lfs::core::Device::CUDA) {
                    dst.copy_from(source_contiguous.cpu());
                } else {
                    dst.copy_from(source_contiguous);
                }
                return dst;
            };

            const int max_sh = model.get_max_sh_degree();
            const int active_sh = model.get_active_sh_degree();
            const float scene_scale = model.get_scene_scale();
            lfs::core::Tensor deleted = model.has_deleted_mask() ? model.deleted() : lfs::core::Tensor{};

            lfs::core::Tensor shN;
            if (model.shN_raw().is_valid() && model.shN_raw().numel() > 0) {
                shN = copy_to_allocator(model.shN_raw(), "SplatData.shN");
            }
            lfs::core::SplatData migrated(max_sh,
                                          copy_to_allocator(model.means_raw(), "SplatData.means"),
                                          copy_to_allocator(model.sh0_raw(), "SplatData.sh0"),
                                          std::move(shN),
                                          copy_to_allocator(model.scaling_raw(), "SplatData.scaling"),
                                          copy_to_allocator(model.rotation_raw(), "SplatData.rotation"),
                                          copy_to_allocator(model.opacity_raw(), "SplatData.opacity"),
                                          scene_scale,
                                          lfs::core::SplatData::ShNLayout::Swizzled);
            migrated.set_active_sh_degree(active_sh);
            if (deleted.is_valid()) {
                migrated.deleted() = std::move(deleted);
            }
            auto lod_tree = std::move(model.lod_tree);
            model = std::move(migrated);
            model.lod_tree = std::move(lod_tree);
            model.set_tensor_allocator(allocator);
            lfs::core::Tensor::trim_memory_pool();
        } catch (const std::exception& e) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to migrate splat tensors to renderer storage: {}", e.what()));
        }
        return {};
    }

    Result<LoadResult> LoaderService::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        // Check if path exists first
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            return make_error(ErrorCode::PATH_NOT_FOUND, "", path);
        }

        // Find appropriate loader
        auto* loader = registry_->findLoader(path);
        if (!loader) {
            // Build user-friendly error message
            const bool is_directory = std::filesystem::is_directory(path, ec);
            const std::string path_type = is_directory ? "folder" : "file";
            const std::string filename = lfs::core::path_to_utf8(path.filename());

            std::string message;
            if (is_directory) {
                message = std::format(
                    "The folder '{}' is not a recognized dataset.\n\n"
                    "For COLMAP datasets, ensure the folder contains:\n"
                    "  - cameras.bin (or cameras.txt)\n"
                    "  - images.bin (or images.txt)\n"
                    "  - An 'images' folder with your photos\n\n"
                    "For NeRF/Blender datasets, ensure the folder contains:\n"
                    "  - transforms.json (or transforms_train.json)",
                    filename);
            } else {
                auto ext = path.extension().string();
                message = std::format(
                    "Cannot open '{}' - unsupported file format.\n\n"
                    "Supported formats:\n"
                    "  - Gaussian Splat files: .ply, .sog, .spz, .rad, .usd, .usda, .usdc, .usdz\n"
                    "  - Mesh files: .obj, .fbx, .gltf, .glb, .stl, .dae\n"
                    "  - Training checkpoints: .resume\n"
                    "  - NeRF transforms: .json",
                    filename);
            }

            LOG_ERROR("Unsupported format: {} ({})", lfs::core::path_to_utf8(path), path_type);
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, message, path);
        }

        LOG_INFO("Using {} loader for: {}", loader->name(), lfs::core::path_to_utf8(path));

        // Perform the load - let the loader return proper errors
        auto result = loader->load(path, options);

        // Guarantee a renderer-ready model regardless of the format's decoder: formats that
        // don't honor splat_tensor_allocator (SOG, SPZ, ...) land in plain CUDA storage, which
        // the Vulkan splat renderer rejects. This migrates them in one place; PLY/checkpoint are
        // already external, so it's a no-op for them.
        if (result && options.splat_tensor_allocator) {
            if (auto* splat = std::get_if<std::shared_ptr<SplatData>>(&result->data); splat && *splat) {
                if (auto migrated = migrateSplatTensorsToAllocator(**splat, options.splat_tensor_allocator);
                    !migrated) {
                    return std::unexpected(migrated.error());
                }
            }
        }
        return result;
    }

    std::vector<std::string> LoaderService::getAvailableLoaders() const {
        std::vector<std::string> names;
        for (const auto& info : registry_->getLoaderInfo()) {
            names.push_back(info.name);
        }
        return names;
    }

    std::vector<std::string> LoaderService::getSupportedExtensions() const {
        return registry_->getAllSupportedExtensions();
    }

} // namespace lfs::io

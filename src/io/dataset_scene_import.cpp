#include "core/tensor_backend.hpp"
#include "dataset_scene_import_internal.hpp"
namespace lfs::training {
    namespace {
        int effectiveMinTrackLengthForLoad(const lfs::core::param::TrainingParameters& params) {
            if (params.dataset.min_track_length > 0 &&
                params.init_path.has_value() &&
                !params.init_path->empty()) {
                LOG_WARN(
                    "min-track-length cannot be used with --init-ply; COLMAP sparse point filtering will not be applied because initialization uses '{}'",
                    *params.init_path);
                return 0;
            }
            return params.dataset.min_track_length;
        }

        constexpr float kShC0 = 0.28209479177387814f;

        std::shared_ptr<lfs::core::PointCloud> pointCloudPreviewFromSplat(
            const lfs::core::SplatData& splat) {
            const auto n = static_cast<size_t>(splat.size());
            auto means = splat.means_raw().is_valid()
                             ? splat.means_raw().cpu().contiguous()
                             : lfs::core::Tensor::zeros({n, 3}, lfs::core::Device::CPU);
            auto colors = lfs::core::Tensor::zeros({n, 3}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            if (n > 0 && colors.data_ptr() != nullptr) {
                std::memset(colors.data_ptr(), 255, n * 3);
            }

            const auto& sh0 = splat.sh0_raw();
            if (n > 0 && sh0.is_valid() && sh0.numel() > 0) {
                try {
                    auto rgb = (sh0.cpu().slice(1, 0, 1).squeeze(1) * kShC0 + 0.5f)
                                   .clamp(0.0f, 1.0f)
                                   .contiguous();
                    if (rgb.is_valid() && rgb.ndim() == 2 &&
                        static_cast<size_t>(rgb.size(0)) == n && rgb.size(1) == 3 &&
                        rgb.dtype() == lfs::core::DataType::Float32 && rgb.ptr<float>() != nullptr) {
                        const float* src = rgb.ptr<float>();
                        auto* dst = static_cast<uint8_t*>(colors.data_ptr());
                        for (size_t i = 0; i < n * 3; ++i) {
                            const float scaled = src[i] * 255.0f;
                            dst[i] = static_cast<uint8_t>(
                                scaled < 0.0f ? 0.0f : (scaled > 255.0f ? 255.0f : scaled + 0.5f));
                        }
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Could not derive PointCloud colors from SH0: {}", e.what());
                }
            }

            return std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors));
        }

        lfs::Result<std::shared_ptr<lfs::core::PointCloud>>
        loadInitReplacementPointCloud(const std::filesystem::path& init_file) {
            const auto filename = lfs::core::path_to_utf8(init_file.filename());
            const auto path_utf8 = lfs::core::path_to_utf8(init_file);

            if (isPlainPointCloudPly(init_file)) {
                auto pc_result = lfs::io::load_ply_point_cloud(init_file);
                if (!pc_result) {
                    return initFileError(
                        std::format("Failed to load '{}': {}", path_utf8, pc_result.error()));
                }
                if (pc_result->size() <= 0) {
                    return initFileError(std::format("'{}' contains no points", path_utf8));
                }
                LOG_INFO("Init {} points from {} (replaces points3D)", pc_result->size(), filename);
                return std::make_shared<lfs::core::PointCloud>(std::move(*pc_result));
            }

            auto loader = lfs::io::Loader::create();
            auto init_result = loader->load(init_file);
            if (!init_result) {
                return initFileError(std::format(
                    "Failed to load '{}': {}", path_utf8, init_result.error().format()));
            }

            auto* splat_ptr = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&init_result->data);
            if (!splat_ptr || !*splat_ptr) {
                return initFileError(std::format("'{}': invalid SplatData", path_utf8));
            }

            auto preview = pointCloudPreviewFromSplat(**splat_ptr);
            if (!preview || preview->size() <= 0) {
                return initFileError(std::format("'{}' contains no points", path_utf8));
            }
            LOG_INFO("Init {} points from splat {} (preview)", preview->size(), filename);
            return preview;
        }

        glm::vec3 centralizedDatasetOrigin(const std::optional<lfs::io::ImportGeoreference>& georeference) {
            if (!georeference) {
                return glm::vec3{0.0f};
            }
            using Provenance = lfs::io::ImportWorldOriginProvenance;
            if (georeference->world_origin_provenance != Provenance::CentralizeByCameras &&
                georeference->world_origin_provenance != Provenance::CentralizeByPointCloud) {
                return glm::vec3{0.0f};
            }
            const auto& origin = georeference->world_origin;
            return {static_cast<float>(origin[0]), static_cast<float>(origin[1]), static_cast<float>(origin[2])};
        }

        lfs::Result<void> attachDatasetPointCloud(
            const lfs::core::param::TrainingParameters& params,
            lfs::core::Scene& scene,
            const lfs::core::NodeId dataset_id,
            const lfs::io::LoadedScene& data,
            const bool verbose) {
            std::shared_ptr<lfs::core::PointCloud> point_cloud;
            if (params.init_path.has_value() && !params.init_path->empty()) {
                auto loaded = loadInitReplacementPointCloud(lfs::core::utf8_to_path(*params.init_path));
                if (!loaded) {
                    return lfs::Status::failure(loaded.error());
                }
                point_cloud = std::move(*loaded);
                centerInitializationMeans(point_cloud->means, scene.getTrainingDataOrigin());
            } else if (data.point_cloud && data.point_cloud->size() > 0) {
                point_cloud = data.point_cloud;
                if (verbose) {
                    LOG_INFO("Adding {} points to scene", point_cloud->size());
                }
            } else {
                if (verbose) {
                    LOG_INFO("No point cloud, using random initialization");
                }
                point_cloud = createRandomPointCloud();
                if (verbose) {
                    LOG_INFO("Adding {} random points to scene", point_cloud->size());
                }
            }

            scene.setInitialPointCloud(point_cloud);
            scene.addPointCloud("PointCloud", point_cloud, dataset_id);
            return {};
        }

        [[nodiscard]] bool isAllocatorBackedTrainingTensorReady(const lfs::core::Tensor& tensor,
                                                                const size_t required_capacity) {
            if (!tensor.is_valid() || tensor.numel() == 0) {
                return required_capacity == 0;
            }
            if (!tensor.is_external_storage() || tensor.capacity() < required_capacity) {
                return false;
            }
            const auto kind = tensor.external_storage_kind();
            return kind == "vulkan_external_buffer" || kind == "splat.exportable";
        }

        std::expected<void, std::string> installLoadedDataset(
            const lfs::core::param::TrainingParameters& params,
            lfs::core::Scene& scene,
            lfs::io::LoadResult& load_result,
            const bool direct_load) {
            return std::visit([&](auto&& data) -> std::expected<void, std::string> {
                using T = std::decay_t<decltype(data)>;

                if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                    auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                    applyTrainingSHDegree(*model, params.optimization.sh_degree);
                    const auto model_id = scene.addSplat("loaded_model", std::move(model));
                    if (model_id == lfs::core::NULL_NODE) {
                        return std::unexpected("Failed to add loaded training model to scene");
                    }
                    scene.setTrainingModelNode(model_id);
                    if (direct_load) {
                        LOG_INFO("Loaded PLY directly into scene");
                    }
                    return {};

                } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                    scene.setSceneCenter(load_result.scene_center);
                    scene.setTrainingDataOrigin(centralizedDatasetOrigin(load_result.georeference));
                    scene.setImagesHaveAlpha(load_result.images_have_alpha);

                    std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                    if (dataset_name.empty()) {
                        dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                    }
                    if (dataset_name.empty()) {
                        dataset_name = "Dataset";
                    }

                    const auto dataset_id = scene.addDataset(dataset_name);
                    if (auto attached = attachDatasetPointCloud(params, scene, dataset_id, data, direct_load);
                        !attached) {
                        return std::unexpected(std::string(attached.error().user_message()));
                    }

                    const auto& cameras = data.cameras;
                    const bool hold_out = params.optimization.holds_out_eval_images();
                    const int test_every = params.dataset.test_every;
                    size_t train_count = 0;
                    size_t val_count = 0;
                    size_t mask_count = 0;
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        const bool is_eval = hold_out && (i % test_every) == 0;
                        cameras[i]->set_split(is_eval ? lfs::core::CameraSplit::Eval : lfs::core::CameraSplit::Train);
                        if (is_eval) {
                            ++val_count;
                        } else {
                            ++train_count;
                        }
                        if (cameras[i]->has_mask()) {
                            ++mask_count;
                        }
                    }

                    const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);
                    const auto train_cameras_id = scene.addCameraGroup(
                        "Training", cameras_group_id, train_count);
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if (!hold_out || (i % test_every) != 0) {
                            scene.addCamera(cameras[i]->image_name(), train_cameras_id, cameras[i]);
                        }
                    }
                    if (hold_out && val_count > 0) {
                        const auto val_cameras_id = scene.addCameraGroup(
                            "Validation", cameras_group_id, val_count);
                        for (size_t i = 0; i < cameras.size(); ++i) {
                            if ((i % test_every) == 0) {
                                scene.addCamera(cameras[i]->image_name(), val_cameras_id, cameras[i]);
                            }
                        }
                    }

                    const auto val_suffix = hold_out ? std::format(" + {} val", val_count) : std::string{};
                    const std::string mask_suffix = mask_count == 0 ? std::string{}
                                                    : direct_load   ? std::format(" ({} with masks)", mask_count)
                                                                    : std::format(" ({} masked)", mask_count);
                    if (direct_load) {
                        LOG_INFO("Loaded dataset '{}' into scene: {} train{} cameras{}",
                                 dataset_name, train_count, val_suffix, mask_suffix);
                    } else {
                        LOG_INFO("Dataset '{}': {} train{} cameras{}",
                                 dataset_name, train_count, val_suffix, mask_suffix);
                    }
                    return {};

                } else if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::MeshData>>) {
                    assert(data && "MeshData must not be null");
                    std::string mesh_name = lfs::core::path_to_utf8(params.dataset.data_path.stem());
                    if (mesh_name.empty())
                        mesh_name = "mesh";
                    scene.addMesh(mesh_name, data);
                    LOG_INFO("Loaded mesh '{}' into scene", mesh_name);
                    return {};

                } else {
                    return std::unexpected(direct_load ? "Unknown data type returned from loader"
                                                       : "Unknown data type from loader");
                }
            },
                              load_result.data);
        }

    } // namespace

    std::expected<void, std::string> migrateTrainingModelToAllocator(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::SplatData& model,
        const lfs::core::SplatTensorAllocator& tensor_allocator,
        const bool force_reallocation) {
        if (!tensor_allocator) {
            return {};
        }

        const size_t n = static_cast<size_t>(model.size());

        // Exportable and vulkan_external_buffer blocks commit live-N plus headroom.
        // Readiness uses that committed capacity when it is below max_cap.
        const size_t configured_max =
            params.optimization.max_cap > 0
                ? std::max<size_t>(static_cast<size_t>(params.optimization.max_cap), n)
                : 0;
        const auto means_kind =
            model.means_raw().is_valid() ? model.means_raw().external_storage_kind()
                                         : std::string{};
        const bool exportable_or_interop =
            means_kind == "splat.exportable" || means_kind == "vulkan_external_buffer";
        const bool exportable_live_n =
            model.means_raw().is_valid() && model.means_raw().is_external_storage() &&
            exportable_or_interop && model.means_raw().capacity() > 0 &&
            (configured_max == 0 || model.means_raw().capacity() < configured_max);
        // Exportable live-N keeps its committed capacity; other storage targets max_cap.
        const size_t target_capacity =
            exportable_live_n
                ? std::max<size_t>(model.means_raw().capacity(), n)
                : (configured_max > 0
                       ? configured_max
                       : std::max<size_t>(model.means_raw().capacity(), n));
        const auto layout_rest = static_cast<std::uint32_t>(model.max_sh_coeffs_rest());
        // q16 is the committed shN encoding. A transient float workspace is already migrate-ready.
        const bool shN_is_q16 = model.shN_value_quantized();
        const bool shN_float_densify_workspace =
            layout_rest > 0 && model.shN_raw().is_valid() &&
            model.shN_raw().dtype() == lfs::core::DataType::Float32 && !shN_is_q16;
        const size_t target_shN_capacity =
            layout_rest == 0
                ? 0
                : (shN_is_q16
                       ? lfs::core::sh_value_quant::sh_value_u16_count(target_capacity, layout_rest)
                       : lfs::core::sh_swizzled_float_count(target_capacity, layout_rest));
        const size_t target_bounds_capacity =
            shN_is_q16 ? lfs::core::sh_value_quant::n_bounds_for_prims(target_capacity) * 2u
                       : 0;

        const bool shN_ready =
            target_shN_capacity == 0 || shN_float_densify_workspace ||
            isAllocatorBackedTrainingTensorReady(model.shN_raw(), target_shN_capacity);
        const bool bounds_ready =
            target_bounds_capacity == 0 || shN_float_densify_workspace ||
            isAllocatorBackedTrainingTensorReady(model.shN_value_bounds(),
                                                 target_bounds_capacity);

        const bool already_allocator_backed =
            isAllocatorBackedTrainingTensorReady(model.means_raw(), target_capacity) &&
            isAllocatorBackedTrainingTensorReady(model.sh0_raw(), target_capacity) &&
            isAllocatorBackedTrainingTensorReady(model.scaling_raw(), target_capacity) &&
            isAllocatorBackedTrainingTensorReady(model.rotation_raw(), target_capacity) &&
            isAllocatorBackedTrainingTensorReady(model.opacity_raw(), target_capacity) &&
            shN_ready && bounds_ready;
        if (already_allocator_backed && !force_reallocation) {
            model.set_tensor_allocator(tensor_allocator);
            return {};
        }

        try {
            auto capacity_ensure = model.release_capacity_ensure();

            const int max_sh = model.get_max_sh_degree();
            const int active_sh = model.get_active_sh_degree();
            const float scene_scale = model.get_scene_scale();
            auto frozen_ranges = model.frozen_ranges();
            lfs::core::Tensor deleted = model.has_deleted_mask() ? model.deleted() : lfs::core::Tensor{};
            lfs::core::Tensor densification_info = model._densification_info;
            lfs::core::Tensor max_screen_share = model._max_screen_share;

            const auto copy_param =
                [&](const lfs::core::Tensor& source,
                    const lfs::core::TensorShape& shape,
                    const size_t capacity,
                    const std::string_view name) -> lfs::core::Tensor {
                lfs::core::Tensor source_cuda = source.device() == lfs::core::Device::GPU
                                                    ? source
                                                    : source.gpu();
                if (!source_cuda.is_contiguous()) {
                    source_cuda = source_cuda.contiguous();
                }
                lfs::core::Tensor dst = tensor_allocator(
                    shape,
                    capacity,
                    source_cuda.dtype(),
                    name);
                dst.set_name(std::string{name});
                if (const auto backend = lfs::core::gpu_backend_of(dst);
                    backend && lfs::core::gpu_backend_of(source_cuda) != backend) {
                    auto migrated = source_cuda.to(*backend);
                    std::swap(source_cuda, migrated);
                }
                dst.copy_from(source_cuda);
                return dst;
            };

            lfs::core::Tensor means = copy_param(
                model.means_raw(), model.means_raw().shape(), target_capacity, "SplatData.means");
            lfs::core::Tensor sh0 = copy_param(
                model.sh0_raw(), model.sh0_raw().shape(), target_capacity, "SplatData.sh0");
            lfs::core::Tensor scaling = copy_param(
                model.scaling_raw(), model.scaling_raw().shape(), target_capacity, "SplatData.scaling");
            lfs::core::Tensor rotation = copy_param(
                model.rotation_raw(), model.rotation_raw().shape(), target_capacity, "SplatData.rotation");
            lfs::core::Tensor opacity = copy_param(
                model.opacity_raw(), model.opacity_raw().shape(), target_capacity, "SplatData.opacity");

            lfs::core::Tensor shN;
            lfs::core::Tensor shN_bounds;
            bool need_q16_encode = false;
            if (layout_rest > 0 && model.shN_raw().is_valid() && model.shN_raw().numel() > 0) {
                const size_t float_cap =
                    lfs::core::sh_swizzled_float_count(target_capacity, layout_rest);
                const size_t q16_cap =
                    lfs::core::sh_value_quant::sh_value_u16_count(target_capacity, layout_rest);
                const size_t bounds_cap =
                    lfs::core::sh_value_quant::n_bounds_for_prims(target_capacity) * 2u;

                if (model.shN_value_quantized()) {
                    // Pad-dropped q16 codes + bounds → exportable/view target.
                    shN = copy_param(
                        model.shN_raw(), model.shN_raw().shape(), q16_cap, "SplatData.shN");
                    if (model.shN_value_bounds().is_valid() &&
                        model.shN_value_bounds().numel() > 0) {
                        shN_bounds = copy_param(
                            model.shN_value_bounds(),
                            model.shN_value_bounds().shape(),
                            bounds_cap,
                            "SplatData.shN_value_bounds");
                    }
                } else {
                    // A rejected or clamped allocation is re-encoded to q16.
                    lfs::core::Tensor src_float = model.shN_raw();
                    if (src_float.device() != lfs::core::Device::GPU) {
                        src_float = src_float.gpu();
                    }
                    if (!src_float.is_contiguous()) {
                        src_float = src_float.contiguous();
                    }
                    lfs::core::Tensor installed;
                    try {
                        installed = tensor_allocator(src_float.shape(),
                                                     float_cap,
                                                     lfs::core::DataType::Float32,
                                                     "SplatData.shN");
                    } catch (const lfs::core::ShareableAllocationLimitError& error) {
                        LOG_INFO("Float shN install rejected by shareable allocation limit ({}); "
                                 "re-encoding to q16 instead",
                                 error.what());
                    } catch (const std::exception& error) {
                        LOG_DEBUG("Float shN install rejected by allocator ({}); "
                                  "re-encoding to q16 instead",
                                  error.what());
                    }
                    const bool landed_in_q16_exportable =
                        !installed.is_valid() ||
                        installed.dtype() == lfs::core::DataType::Float16 ||
                        installed.capacity() < float_cap;
                    if (landed_in_q16_exportable) {
                        shN = std::move(src_float);
                        need_q16_encode = true;
                    } else {
                        installed.set_name("SplatData.shN");
                        if (const auto backend = lfs::core::gpu_backend_of(installed);
                            backend && lfs::core::gpu_backend_of(src_float) != backend) {
                            auto migrated = src_float.to(*backend);
                            std::swap(src_float, migrated);
                        }
                        installed.copy_from(src_float);
                        shN = std::move(installed);
                    }
                }
            }

            if (const auto backend = lfs::core::gpu_backend_of(means)) {
                for (auto* tensor : {&shN, &shN_bounds, &deleted, &densification_info, &max_screen_share}) {
                    if (tensor->is_valid() && tensor->device() == lfs::core::Device::GPU &&
                        lfs::core::gpu_backend_of(*tensor) != backend) {
                        auto migrated = (*tensor).to(*backend);
                        std::swap(*tensor, migrated);
                    }
                }
            }

            lfs::core::SplatData migrated(max_sh,
                                          std::move(means),
                                          std::move(sh0),
                                          std::move(shN),
                                          std::move(scaling),
                                          std::move(rotation),
                                          std::move(opacity),
                                          scene_scale,
                                          lfs::core::SplatData::ShNLayout::Swizzled);
            migrated.set_active_sh_degree(active_sh, std::move(shN_bounds));
            if (deleted.is_valid()) {
                migrated.deleted() = std::move(deleted);
            }
            if (densification_info.is_valid()) {
                migrated._densification_info = std::move(densification_info);
            }
            if (max_screen_share.is_valid()) {
                migrated._max_screen_share = std::move(max_screen_share);
            }
            migrated.set_frozen_ranges(std::move(frozen_ranges));
            std::swap(model, migrated);
            model.set_tensor_allocator(tensor_allocator);
            if (capacity_ensure) {
                model.set_capacity_ensure(std::move(capacity_ensure));
            }
            // A clamped target stores q16.
            if (need_q16_encode && model.shN_raw().is_valid() &&
                !model.shN_value_quantized()) {
                (void)model.apply_shN_value_quant();
            }
            lfs::core::Tensor::trim_memory_pool();

            LOG_INFO("Migrated training SplatData tensors to Vulkan-external storage "
                     "(gaussians={}, capacity={}, shN_q16={}, shN_capacity_cells={})",
                     n,
                     model.means_raw().capacity(),
                     model.shN_value_quantized(),
                     model.shN_raw().is_valid() ? model.shN_raw().capacity() : 0);
        } catch (const std::exception& e) {
            return std::unexpected(std::format(
                "Failed to migrate training SplatData to Vulkan-external storage: {}",
                e.what()));
        }

        return {};
    }

    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene) {

        auto data_loader = lfs::io::Loader::create();

        const auto& data_path = params.dataset.data_path;
        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .min_track_length = effectiveMinTrackLengthForLoad(params),
            .validate_only = false,
            .load_masks = params.optimization.mask_mode != lfs::core::param::MaskMode::None,
            .load_depths = params.optimization.use_depth_loss &&
                           params.optimization.depth_loss_weight > 0.0f,
            .load_normals = training_normal_priors_enabled(params.optimization) ||
                            (!params.optimization.gut && params.optimization.enable_eval),
            .normal_auto_generate = params.optimization.normal_auto_generate,
            .centralize = parse_centralize(params.dataset.centralize_dataset),
            .progress = [&data_path](float percentage, const std::string& message) {
                LOG_DEBUG("[{:5.1f}%] {}", percentage, message);
                lfs::core::events::state::DatasetLoadProgress{
                    .path = data_path,
                    .progress = percentage,
                    .step = message}
                    .emit();
            }};

        LOG_INFO("Loading dataset from: {}", lfs::core::path_to_utf8(params.dataset.data_path));
        auto load_result = data_loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error().format()));
        }

        LOG_INFO("Dataset loaded successfully using {} loader", load_result->loader_used);

        return installLoadedDataset(params, scene, *load_result, true);
    }

    std::expected<void, std::string> validateDatasetPath(
        const lfs::core::param::TrainingParameters& params) {

        auto data_loader = lfs::io::Loader::create();

        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .min_track_length = params.dataset.min_track_length,
            .validate_only = true,
            .load_masks = params.optimization.mask_mode != lfs::core::param::MaskMode::None,
            .load_depths = params.optimization.use_depth_loss &&
                           params.optimization.depth_loss_weight > 0.0f,
            .load_normals = training_normal_priors_enabled(params.optimization),
            .normal_auto_generate = params.optimization.normal_auto_generate};

        auto result = data_loader->load(params.dataset.data_path, load_options);
        if (!result) {
            return std::unexpected(result.error().format());
        }
        return {};
    }

    std::expected<void, std::string> applyLoadResultToScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::io::LoadResult&& load_result) {

        return installLoadedDataset(params, scene, load_result, false);
    }

} // namespace lfs::training

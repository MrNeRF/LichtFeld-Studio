/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"

#include "core/assert.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "popspa_cameras.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/gsplat_rasterizer.hpp"
#include "strategies/postprocess_compaction.hpp"
#include "strategies/strategy_utils.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <limits>
#include <thread>

namespace lfs::training {

    namespace {
        std::unique_ptr<AdamOptimizer> make_popspa_optimizer(core::SplatData& model, const core::param::OptimizationParameters& opt) {
            AdamConfig config;
            config.eps = 1e-15;
            config.initial_capacity = model.size();
            config.param_lrs = {
                {"means", opt.popspa_means_lr * static_cast<double>(model.get_scene_scale())},
                {"scaling", opt.popspa_scales_lr},
                {"rotation", opt.popspa_quaternions_lr},
                {"opacity", opt.popspa_opacities_lr},
                {"sh0", opt.popspa_sh0_lr},
                {"shN", opt.popspa_shn_lr},
            };
            auto optimizer = std::make_unique<AdamOptimizer>(model, config);
            optimizer->allocate_gradients(model.size());
            // Adam allocates gradients lazily. Empty views still need a zero
            // gradient for every resident parameter so momentum and age advance.
            for (const auto type : {ParamType::Means, ParamType::Sh0, ParamType::ShN,
                                    ParamType::Scaling, ParamType::Rotation, ParamType::Opacity}) {
                if (type != ParamType::ShN || model.max_sh_coeffs_rest() > 0)
                    (void)optimizer->get_grad(type);
            }
            apply_frozen_ranges_to_optimizer(model, *optimizer, 0.0f);
            return optimizer;
        }

    } // namespace

    uint64_t sort_and_fingerprint_popspa_cameras(std::vector<core::Camera*>& cameras) {
        std::sort(cameras.begin(), cameras.end(), [](const auto* a, const auto* b) { return a->uid() < b->uid(); });
        uint64_t hash = 14695981039346656037ULL;
        const auto append = [&hash](const void* data, size_t size) {
            const auto* bytes = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                hash ^= bytes[i];
                hash *= 1099511628211ULL;
            }
        };
        for (const auto* cam : cameras) {
            const std::string identity = std::format("{}:{}:{}:{}:{}:{};", cam->uid(), cam->camera_id(),
                                                     core::path_to_utf8(cam->image_path()), cam->image_width(), cam->image_height(),
                                                     static_cast<int>(cam->camera_model_type()));
            append(identity.data(), identity.size());
            const auto [fx, fy, cx, cy] = cam->get_intrinsics();
            const float intrinsics[] = {fx, fy, cx, cy};
            append(intrinsics, sizeof(intrinsics));
            for (const auto& tensor : {cam->R(), cam->T(), cam->radial_distortion(), cam->tangential_distortion()}) {
                if (tensor.is_valid() && tensor.numel() > 0) {
                    const auto host = tensor.cpu().contiguous();
                    append(host.data_ptr(), host.numel() * core::dtype_size(host.dtype()));
                }
            }
        }
        return hash;
    }

    bool Trainer::popspa_enabled() const {
        const auto params = getParams();
        return params.optimization.enable_sparsity &&
               params.optimization.sparsity_method == core::param::SparsityMethod::POPSpa;
    }

    bool Trainer::popspa_optimizing() const noexcept {
        return popspa_controller_.is_initialized() &&
               (popspa_controller_.phase() == POPSpaPhase::Sparsify ||
                popspa_controller_.phase() == POPSpaPhase::Recover);
    }

    AdamOptimizer& Trainer::active_gaussian_optimizer() {
        return popspa_optimizing() ? *popspa_optimizer_ : strategy_->get_optimizer();
    }

    core::param::OptimizationParameters Trainer::popspa_step_parameters() const {
        auto opt = params_.optimization;
        opt.lambda_dssim = opt.popspa_ssim_weight;
        opt.scale_reg = 0.0f;
        opt.opacity_reg = 0.0f;
        opt.use_depth_loss = false;
        opt.normal_loss_weight = 0.0f;
        opt.normal_consistency_weight = 0.0f;
        opt.normal_flatten_weight = 0.0f;
        opt.ppisp_use_controller = false;
        opt.ppisp_freeze_gaussians_on_distill = false;
        return opt;
    }

    void Trainer::preserve_active_popspa_parameters(core::param::TrainingParameters& update) const {
        if (!popspa_controller_.is_initialized())
            return;
        auto& opt = update.optimization;
        const auto& current = params_.optimization;
        bool changed = false;
        const auto preserve = [&](auto member) {
            changed |= opt.*member != current.*member;
            opt.*member = current.*member;
        };
        using Opt = core::param::OptimizationParameters;
        preserve(&Opt::iterations);
        preserve(&Opt::enable_sparsity);
        preserve(&Opt::sparsity_method);
        preserve(&Opt::popspa_target_count);
        preserve(&Opt::popspa_first_prune_count);
        preserve(&Opt::popspa_sparsify_steps);
        preserve(&Opt::popspa_refine_steps);
        preserve(&Opt::popspa_rho);
        preserve(&Opt::popspa_projection_interval);
        preserve(&Opt::popspa_erank_weight);
        preserve(&Opt::popspa_thin_scale_weight);
        preserve(&Opt::popspa_erank_epsilon);
        preserve(&Opt::popspa_means_lr);
        preserve(&Opt::popspa_scales_lr);
        preserve(&Opt::popspa_opacities_lr);
        preserve(&Opt::popspa_quaternions_lr);
        preserve(&Opt::popspa_sh0_lr);
        preserve(&Opt::popspa_shn_lr);
        preserve(&Opt::popspa_ssim_weight);
        preserve(&Opt::popspa_seed);
        preserve(&Opt::gut);
        preserve(&Opt::mip_filter);
        preserve(&Opt::bg_mode);
        preserve(&Opt::bg_color);
        preserve(&Opt::bg_image_path);
        preserve(&Opt::bg_modulation);
        if (changed)
            LOG_WARN("POPSpa settings are fixed after its first score pass starts; restart training to change them");
    }

    void Trainer::initialize_popspa_optimizer() {
        popspa_optimizer_ = make_popspa_optimizer(strategy_->get_model(), params_.optimization);
    }

    lfs::Status Trainer::advance_popspa_boundaries(const int completed_iteration, std::stop_token stop_token) {
        if (!popspa_enabled() || completed_iteration < get_regular_iterations())
            return {};
        if (popspa_dataset_validated_ && popspa_controller_.is_initialized() &&
            (popspa_optimizing() || popspa_controller_.phase() == POPSpaPhase::Complete))
            return {};
        const size_t rows_before = strategy_->get_model().size();
        const auto publish_changed_topology_on_failure = [&]() noexcept {
            if (!scene_ || strategy_->get_model().size() == rows_before)
                return;
            try {
                // The topology may already be committed if external storage
                // publication failed. Keep scene counts/cache coherent for saving.
                scene_->syncTrainingModelTopology(strategy_->get_model().size());
            } catch (const std::exception& e) {
                LOG_ERROR("POPSpa topology publication after failure: {}", e.what());
            }
        };
        try {
            LFS_ASSERT_MSG(train_dataset_ && train_dataset_->size() > 0,
                           "POPSpa scoring requires a nonempty training dataset (views=0)");
            std::vector<core::Camera*> cameras;
            cameras.reserve(train_dataset_->size());
            // Use the dataset's split indices: get_cameras() also contains held-out views.
            for (size_t i = 0; i < train_dataset_->size(); ++i) {
                auto* cam = train_dataset_->get_camera(i);
                if (!cam->image_size_loaded())
                    cam->load_image_size(params_.dataset.resize_factor, params_.dataset.max_width);
                cameras.push_back(cam);
            }
            const uint64_t fingerprint = sort_and_fingerprint_popspa_cameras(cameras);
            auto& model = strategy_->get_model();
            if (!popspa_controller_.is_initialized()) {
                sparsity_optimizer_.reset();
                const auto& opt = params_.optimization;
                auto active = compute_near_zero_rotation_mask(model.rotation_raw()).logical_not();
                if (model.has_deleted_mask())
                    active = active.logical_and(model.deleted().logical_not());
                const POPSpaController::Config config{
                    .target_count = static_cast<size_t>(opt.popspa_target_count),
                    .first_prune_count = static_cast<size_t>(opt.popspa_first_prune_count),
                    .sparsify_steps = static_cast<uint32_t>(opt.popspa_sparsify_steps),
                    .refine_steps = static_cast<uint32_t>(opt.popspa_refine_steps),
                    .rho = opt.popspa_rho,
                    .projection_interval = static_cast<uint32_t>(opt.popspa_projection_interval),
                    .erank_weight = opt.popspa_erank_weight,
                    .thin_scale_weight = opt.popspa_thin_scale_weight,
                    .erank_epsilon = opt.popspa_erank_epsilon,
                    .camera_fingerprint = fingerprint,
                };
                POPSpaController staged_controller;
                auto initialized = staged_controller.initialize(config, model.opacity_raw(), cameras.size(), active,
                                                                make_frozen_mask(model, model.size(), model.means().device()));
                if (!initialized)
                    return initialized;
                auto staged_optimizer = make_popspa_optimizer(model, opt);
                popspa_controller_.adopt_checkpoint_state(staged_controller);
                popspa_optimizer_ = std::move(staged_optimizer);
                popspa_completed_iteration_.store(completed_iteration, std::memory_order_release);
                LOG_INFO("POPSpa: {} rows, target {}, {} score views", model.size(), config.target_count, cameras.size());
            } else {
                LFS_ASSERT_MSG(popspa_controller_.config().camera_fingerprint == fingerprint &&
                                   popspa_controller_.score_view_count() == cameras.size() &&
                                   popspa_controller_.state_size() == model.size(),
                               std::format("POPSpa resume must match camera identity and model rows (saved_views={}, views={}, saved_rows={}, rows={})",
                                           popspa_controller_.score_view_count(), cameras.size(), popspa_controller_.state_size(), model.size()));
            }
            popspa_dataset_validated_ = true;
            while (!popspa_optimizing() && popspa_controller_.phase() != POPSpaPhase::Complete) {
                handle_control_requests(completed_iteration, stop_token);
                while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                    consume_requested_project_snapshot(completed_iteration);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    handle_control_requests(completed_iteration, stop_token);
                }
                if (stop_requested_.load() || stop_token.stop_requested())
                    return {};
                const auto phase = popspa_controller_.phase();
                if (phase == POPSpaPhase::FirstScore || phase == POPSpaPhase::SecondScore) {
                    auto* cam = cameras.at(popspa_controller_.score_view());
                    const int background_step = static_cast<int>((static_cast<uint64_t>(params_.optimization.popspa_seed) +
                                                                  popspa_controller_.score_view()) %
                                                                 static_cast<uint64_t>(std::numeric_limits<int>::max()));
                    auto& bg = background_for_step(background_step);
                    core::Tensor image_bg;
                    if (params_.optimization.bg_mode == core::param::BackgroundMode::Image)
                        image_bg = get_background_image_for_camera(cam->image_width(), cam->image_height());
                    else if (params_.optimization.bg_mode == core::param::BackgroundMode::Random)
                        image_bg = get_random_background_for_camera(cam->image_width(), cam->image_height(), background_step);
                    if (params_.optimization.gut) {
                        auto rendered = gsplat_rasterize_forward(*cam, model, bg, 0, 0, 0, 0,
                                                                 1.0f, false, GsplatRenderMode::RGB, true, image_bg);
                        if (!rendered)
                            throw std::runtime_error(rendered.error());
                        const auto scored = gsplat_accumulate_pop_scores(rendered->second, popspa_controller_.scores());
                        core::GlobalArenaManager::instance().get_arena().end_frame(rendered->second.frame_id, rendered->second.stream);
                        if (!scored)
                            return scored;
                    } else {
                        auto rendered = fast_rasterize_forward(*cam, model, bg, 0, 0, 0, 0,
                                                               params_.optimization.mip_filter, image_bg);
                        if (!rendered)
                            return lfs::Status::failure(std::move(rendered).error());
                        auto scored = fast_accumulate_pop_scores(rendered->second, popspa_controller_.scores());
                        if (!scored)
                            return scored;
                    }
                    if (auto advanced = popspa_controller_.finish_score_view(); !advanced)
                        return advanced;
                    const auto view = popspa_controller_.score_view();
                    if (view % 25 == 0 || view == cameras.size())
                        LOG_INFO("POPSpa score pass {}: {}/{} views", phase == POPSpaPhase::FirstScore ? 1 : 2, view, cameras.size());
                    consume_requested_project_snapshot(completed_iteration);
                    continue;
                }

                auto indices = popspa_controller_.prune_indices();
                if (!indices)
                    return lfs::Status::failure(std::move(indices).error());
                const size_t old_size = model.size();
                {
                    std::unique_lock render_lock(render_mutex_);
                    std::unique_lock<std::mutex> combined_lock;
                    if (scene_)
                        combined_lock = scene_->acquireCombinedModelExclusive();
                    struct MutationLockMark {
                        MutationLockMark() { mark_live_model_mutation_lock_held(true); }
                        ~MutationLockMark() { mark_live_model_mutation_lock_held(false); }
                    } mutation_lock_mark;
                    waitForModelReaders();
                    const auto barrier = beginExportableDensifyBarrier();
                    if (barrier == ExportableDensifyBarrierBegin::Failed)
                        throw std::runtime_error("POPSpa could not acquire the exportable topology barrier");
                    try {
                        POPSpaController staged_controller = popspa_controller_;
                        std::unique_ptr<AdamOptimizer> staged_optimizer;
                        auto compacted = compact_gaussians_for_postprocess(
                            *strategy_, get_runtime_optimization_params(), *indices,
                            [&](core::SplatData& staged_model) -> lfs::Status {
                                auto accepted = staged_controller.accept_prune(staged_model.opacity_raw(), {},
                                                                               make_frozen_mask(staged_model, staged_model.size(), staged_model.means().device()));
                                if (!accepted)
                                    return accepted;
                                staged_optimizer = make_popspa_optimizer(staged_model, params_.optimization);
                                return {};
                            });
                        if (!compacted)
                            throw lfs::Exception(std::move(compacted).error());
                        // Algorithm state is prepared before committing topology.
                        // Publish into the shared external storage only afterward:
                        // its allocator can alias the old model's physical buffers.
                        popspa_controller_.adopt_checkpoint_state(staged_controller);
                        popspa_optimizer_->adopt_checkpoint_state(*staged_optimizer);
                        popspa_optimizer_->set_frozen_mask(staged_optimizer->frozen_mask());
                        popspa_optimizer_->set_frozen_lr_scale(0.0f);
                        ++mutation_epoch_;
                        if (auto storage = ensureModelTensorAllocatorStorage(model, "POPSpa compaction"); !storage)
                            throw std::runtime_error(storage.error());
                        recordParamsReady();
                    } catch (...) {
                        if (barrier == ExportableDensifyBarrierBegin::Acquired)
                            (void)endExportableDensifyBarrier();
                        throw;
                    }
                    if (barrier == ExportableDensifyBarrierBegin::Acquired && !endExportableDensifyBarrier())
                        throw std::runtime_error("POPSpa could not publish compacted exportable storage");
                }
                if (scene_)
                    scene_->syncTrainingModelTopology(model.size());
                LOG_INFO("POPSpa prune: {} -> {} Gaussians", old_size, model.size());
                consume_requested_project_snapshot(completed_iteration);
            }
            return {};
        } catch (const lfs::Exception& e) {
            publish_changed_topology_on_failure();
            return lfs::Status::failure(e.error());
        } catch (const std::exception& e) {
            publish_changed_topology_on_failure();
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "POPSpa phase transition failed.",
                .detail = e.what(),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
    }

} // namespace lfs::training

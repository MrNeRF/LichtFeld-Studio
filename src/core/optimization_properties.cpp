/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/optimization_properties.hpp"

#include "core/parameters.hpp"
#include "core/property_registry.hpp"

#include <limits>
#include <mutex>

namespace lfs::core::param {

    using namespace lfs::core::prop;

    void register_optimization_properties() {
        // Registry defaults match the default strategy (MRNF), not bare struct {}.
        const OptimizationParameters d = OptimizationParameters::mrnf_defaults();
        PropertyGroupBuilder<OptimizationParameters>("optimization", "Optimization")
            // Training control
            .size_prop(&OptimizationParameters::iterations,
                       "iterations", "Max Iterations", d.iterations, 1, 1000000,
                       "Maximum number of training iterations")
            .json_required()
            .locale("training_params.iterations")
            .tooltip("training.tooltip.iterations")
            .precision(0)
            .ui_step(100)
            .all_strategies()
            .int_prop(&OptimizationParameters::sh_degree,
                      "sh_degree", "SH Degree", d.sh_degree, 0, 3,
                      "Spherical harmonics degree (0-3)")
            .json_required()
            .all_strategies()
            .size_prop(&OptimizationParameters::sh_degree_interval,
                       "sh_degree_interval", "SH Interval", d.sh_degree_interval, 100, 10000,
                       "Iterations between SH degree increases")
            .locale("training.refinement.sh_upgrade_every")
            .tooltip("training.tooltip.sh_upgrade_every")
            .precision(0)
            .ui_step(100)
            .all_strategies()
            .int_prop(&OptimizationParameters::max_cap,
                      "max_cap", "Max Gaussians", d.max_cap, 1000, 200000000,
                      "Maximum number of gaussians")
            .locale("training_params.max_gaussians")
            .tooltip("training.tooltip.max_gaussians")
            .precision(0)
            .ui_step(100000)

            // Learning rates
            .all_strategies()
            .float_prop(&OptimizationParameters::means_lr,
                        "means_lr", "Position LR", d.means_lr, 0.0f, 0.001f,
                        "Learning rate for gaussian positions")
            .json_required()
            .locale("training.opt.lr.position")
            .tooltip("training.tooltip.lr_position")
            .precision(6)
            .ui_step(1e-6)
            .flags(PROP_LIVE_UPDATE)
            .all_strategies()
            .float_prop(&OptimizationParameters::means_lr_end,
                        "means_lr_end", "Position LR End", d.means_lr_end, 0.0f, 0.001f,
                        "Target end learning rate for gaussian positions")
            .locale("training.advanced.means_lr_end")
            .tooltip("training.tooltip.means_lr_end")
            .precision(8)
            .ui_step(1e-8)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::shs_lr,
                        "shs_lr", "SH LR", d.shs_lr, 0.0f, 0.1f,
                        "Learning rate for spherical harmonics")
            .json_required()
            .locale("training.opt.lr.sh_coeff")
            .tooltip("training.tooltip.lr_sh_coeff")
            .precision(4)
            .ui_step(1e-4)
            .flags(PROP_LIVE_UPDATE)
            .all_strategies()
            .float_prop(&OptimizationParameters::opacity_lr,
                        "opacity_lr", "Opacity LR", d.opacity_lr, 0.0f, 1.0f,
                        "Learning rate for opacity")
            .json_required()
            .locale("training.opt.lr.opacity")
            .tooltip("training.tooltip.lr_opacity")
            .precision(4)
            .ui_step(0.001)
            .flags(PROP_LIVE_UPDATE)
            .all_strategies()
            .float_prop(&OptimizationParameters::scaling_lr,
                        "scaling_lr", "Scale LR", d.scaling_lr, 0.0f, 0.1f,
                        "Learning rate for gaussian scales")
            .json_required()
            .locale("training.opt.lr.scaling")
            .tooltip("training.tooltip.lr_scaling")
            .precision(4)
            .ui_step(1e-4)
            .flags(PROP_LIVE_UPDATE)
            .all_strategies()
            .float_prop(&OptimizationParameters::scaling_lr_end,
                        "scaling_lr_end", "Scale LR End", d.scaling_lr_end, 0.0f, 0.1f,
                        "Target end learning rate for gaussian scales")
            .locale("training.advanced.scaling_lr_end")
            .tooltip("training.tooltip.scaling_lr_end")
            .precision(6)
            .ui_step(1e-5)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::rotation_lr,
                        "rotation_lr", "Rotation LR", d.rotation_lr, 0.0f, 0.1f,
                        "Learning rate for rotations")
            .json_required()
            .locale("training.opt.lr.rotation")
            .tooltip("training.tooltip.lr_rotation")
            .precision(4)
            .ui_step(1e-4)
            .flags(PROP_LIVE_UPDATE)
            .all_strategies()
            .float_prop(&OptimizationParameters::cropbox_lr_scale,
                        "cropbox_lr_scale", "Rejected splat LR scale", d.cropbox_lr_scale, 0.0f, 1.0f,
                        "Scales Adam steps and refinement signals for rejected splats; strategy noise, decay, and resets remain active")
            .locale("training.advanced.cropbox_lr_scale")
            .tooltip("training.tooltip.cropbox_lr_scale")
            .precision(3)
            .ui_step(0.01)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::cropbox_loss_weight,
                        "cropbox_loss_weight", "Outside ROI loss weight", d.cropbox_loss_weight, 0.0f, 1.0f,
                        "Scales pixel losses for camera rays outside the active crop box")
            .locale("training.advanced.cropbox_loss_weight")
            .tooltip("training.tooltip.cropbox_loss_weight")
            .precision(3)
            .ui_step(0.01)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)

            // Loss parameters
            .all_strategies()
            .float_prop(&OptimizationParameters::lambda_dssim,
                        "lambda_dssim", "DSSIM Weight", d.lambda_dssim, 0.0f, 1.0f,
                        "Weight for structural similarity loss")
            .json_required()
            .all_strategies()
            .float_prop(&OptimizationParameters::opacity_reg,
                        "opacity_reg", "Opacity Reg", d.opacity_reg, 0.0f, 1.0f,
                        "Opacity regularization weight")
            .locale("training.losses.opacity_reg")
            .tooltip("training.tooltip.opacity_reg")
            .precision(4)
            .ui_step(0.001)
            .all_strategies()
            .float_prop(&OptimizationParameters::scale_reg,
                        "scale_reg", "Scale Reg", d.scale_reg, 0.0f, 1.0f,
                        "Scale regularization weight")
            .locale("training.losses.scale_reg")
            .tooltip("training.tooltip.scale_reg")
            .precision(4)
            .ui_step(0.001)

            // Refinement
            .all_strategies()
            .size_prop(&OptimizationParameters::refine_every,
                       "refine_every", "Refine Every", d.refine_every, 1, 1000,
                       "Interval for adaptive density control")
            .json_required()
            .locale("training.refinement.refine_every")
            .tooltip("training.tooltip.refine_every")
            .precision(0)
            .ui_step(10)
            .all_strategies()
            .size_prop(&OptimizationParameters::start_refine,
                       "start_refine", "Start Refine", d.start_refine, 0, 10000,
                       "Iteration to start refinement")
            .json_required()
            .locale("training.refinement.start_refine")
            .tooltip("training.tooltip.start_refine")
            .precision(0)
            .ui_step(100)
            .all_strategies()
            .size_prop(&OptimizationParameters::stop_refine,
                       "stop_refine", "Stop Refine", d.stop_refine, 0, 100000,
                       "Iteration to stop refinement")
            .json_required()
            .locale("training.refinement.stop_refine")
            .tooltip("training.tooltip.stop_refine")
            .precision(0)
            .ui_step(1000)
            .all_strategies()
            .size_prop(&OptimizationParameters::morton_reorder_interval,
                       "morton_reorder_interval", "Morton Reorder", d.morton_reorder_interval, 0, 30000,
                       "Reorder Gaussians by 3D Morton code every N iterations (0 disables)")
            .locale("training.refinement.morton_reorder_interval")
            .tooltip("training.tooltip.morton_reorder_interval")
            .precision(0)
            .ui_step(1000)
            .all_strategies()
            .float_prop(&OptimizationParameters::min_opacity,
                        "min_opacity", "Min Opacity", d.min_opacity, 0.0f, std::numeric_limits<float>::infinity(),
                        "Minimum opacity for pruning")
            .json_required()
            .locale("training.thresholds.min_opacity")
            .tooltip("training.tooltip.min_opacity")
            .precision(4)
            .ui_step(0.001)
            .flags(PROP_ADVANCED)
            .strategies({"mcmc"})
            .all_strategies()
            .float_prop(&OptimizationParameters::init_opacity,
                        "init_opacity", "Init Opacity", d.init_opacity, 0.0f, 1.0f,
                        "Initial opacity for new gaussians")
            .all_strategies()
            .float_prop(&OptimizationParameters::init_scaling,
                        "init_scaling", "Init Scale", d.init_scaling, 0.0f, 1.0f,
                        "Initial scale for new gaussians")
            .locale("training.init.init_scaling")
            .tooltip("training.tooltip.init_scaling")
            .precision(3)
            .ui_step(0.01)

            // Mask parameters
            .all_strategies()
            .enum_prop(&OptimizationParameters::mask_mode,
                       "mask_mode", "Mask Mode", d.mask_mode,
                       {{"None", MaskMode::None, "training.options.mask.none", "none"},
                        {"Segment", MaskMode::Segment, "training.options.mask.segment", "segment"},
                        {"Ignore", MaskMode::Ignore, "training.options.mask.ignore", "ignore"},
                        {"SegmentAndIgnore", MaskMode::SegmentAndIgnore, "training.options.mask.segment_and_ignore", "segment_and_ignore"},
                        {"AlphaConsistent", MaskMode::AlphaConsistent, "training.options.mask.alpha_consistent", "alpha_consistent"}},
                       "Attention mask behavior during training")
            .locale("training_params.mask_mode")
            .tooltip("training.tooltip.mask_mode")
            .all_strategies()
            .bool_prop(&OptimizationParameters::invert_masks,
                       "invert_masks", "Invert Masks", d.invert_masks,
                       "Swap object and background in masks")
            .locale("training_params.invert_masks")
            .tooltip("training.tooltip.invert_masks")
            .all_strategies()
            .float_prop(&OptimizationParameters::mask_threshold,
                        "mask_threshold", "Mask Threshold", d.mask_threshold, 0.0f, 1.0f,
                        "Threshold for mask binarization")
            .locale("training.masking.threshold")
            .tooltip("training.tooltip.mask_threshold")
            .precision(3)
            .ui_step(0.05)
            .all_strategies()
            .float_prop(&OptimizationParameters::mask_opacity_penalty_weight,
                        "mask_opacity_penalty_weight", "Penalty Weight", d.mask_opacity_penalty_weight, 0.0f, 10.0f,
                        "Opacity penalty weight for segment mode")
            .locale("training.masking.penalty_weight")
            .tooltip("training.tooltip.penalty_weight")
            .precision(3)
            .ui_step(0.1)
            .all_strategies()
            .float_prop(&OptimizationParameters::mask_opacity_penalty_power,
                        "mask_opacity_penalty_power", "Penalty Power", d.mask_opacity_penalty_power, 0.5f, 4.0f,
                        "Power for opacity penalty in segment mode")
            .locale("training.masking.penalty_power")
            .tooltip("training.tooltip.penalty_power")
            .precision(3)
            .ui_step(0.1)
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_alpha_as_mask,
                       "use_alpha_as_mask", "Use Alpha as Mask", d.use_alpha_as_mask,
                       "Use alpha channel from RGBA images as mask source")
            .locale("training_params.use_alpha_as_mask")
            .tooltip("training.tooltip.use_alpha_as_mask")
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_depth_loss,
                       "use_depth_loss", "Use Depth Loss", d.use_depth_loss,
                       "Use dataset depth maps for depth supervision")
            .locale("training_params.use_depth_loss")
            .tooltip("training.tooltip.use_depth_loss")
            .all_strategies()
            .float_prop(&OptimizationParameters::depth_loss_weight,
                        "depth_loss_weight", "Depth Loss Weight", d.depth_loss_weight, 0.0f, 100.0f,
                        "Weight for depth supervision")
            .locale("training_params.depth_loss_weight")
            .tooltip("training.tooltip.depth_loss_weight")
            .precision(3)
            .ui_step(0.1)
            .all_strategies()
            .string_prop(&OptimizationParameters::depth_loss_mode,
                         "depth_loss_mode", "Depth Loss Mode", d.depth_loss_mode,
                         "Depth prior convention: ssi (auto-detect), ssi-disparity, or ssi-depth")
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_normal_loss,
                       "use_normal_loss", "Use Normal Loss", d.use_normal_loss,
                       "Use dataset normal maps for normal supervision")
            .locale("training_params.use_normal_loss")
            .tooltip("training.tooltip.use_normal_loss")
            .all_strategies()
            .float_prop(&OptimizationParameters::normal_loss_weight,
                        "normal_loss_weight", "Normal Loss Weight", d.normal_loss_weight, 0.0f, 100.0f,
                        "Weight for prior normal supervision")
            .locale("training_params.normal_loss_weight")
            .tooltip("training.tooltip.normal_loss_weight")
            .precision(3)
            .ui_step(0.01)
            .all_strategies()
            .float_prop(&OptimizationParameters::normal_consistency_weight,
                        "normal_consistency_weight", "Normal Consistency Weight", d.normal_consistency_weight, 0.0f, 100.0f,
                        "Weight for depth-normal consistency")
            .locale("training_params.normal_consistency_weight")
            .tooltip("training.tooltip.normal_consistency_weight")
            .precision(3)
            .ui_step(0.01)
            .all_strategies()
            .float_prop(&OptimizationParameters::normal_flatten_weight,
                        "normal_flatten_weight", "Normal Flatten Weight", d.normal_flatten_weight, 0.0f, 1000.0f,
                        "Min-axis scale flattening weight while normal supervision is active")
            .locale("training_params.normal_flatten_weight")
            .tooltip("training.tooltip.normal_flatten_weight")
            .precision(3)
            .ui_step(0.1)
            .all_strategies()
            .enum_prop(&OptimizationParameters::normal_loss_space,
                       "normal_loss_space", "Normal Loss Space", d.normal_loss_space,
                       {{"Auto", NormalLossSpace::Auto, "training.options.normal_loss_space.auto", "auto"},
                        {"Camera (OpenCV)", NormalLossSpace::CameraOpenCV,
                         "training.options.normal_loss_space.camera_opencv", "camera-opencv"},
                        {"Camera (OpenGL)", NormalLossSpace::CameraOpenGL,
                         "training.options.normal_loss_space.camera_opengl", "camera-opengl"},
                        {"World", NormalLossSpace::World, "training.options.normal_loss_space.world", "world"}},
                       "Normal prior coordinate space: auto, camera-opencv, camera-opengl, or world")
            .locale("training.advanced.normal_loss_space")
            .tooltip("training.tooltip.normal_loss_space")
            .flags(PROP_ADVANCED)
            .all_strategies()

            // Bilateral grid
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_bilateral_grid,
                       "use_bilateral_grid", "Bilateral Grid", d.use_bilateral_grid,
                       "Enable bilateral grid color correction")
            .locale("training_params.bilateral_grid")
            .tooltip("training.tooltip.bilateral_grid")
            .flags(PROP_NEEDS_RESTART)
            .all_strategies()
            .int_prop(&OptimizationParameters::bilateral_grid_X,
                      "bilateral_grid_x", "Grid X", d.bilateral_grid_X, 4, 64,
                      "Bilateral grid X resolution")
            .json_key("bilateral_grid_X")
            .locale("training.bilateral.grid_x")
            .tooltip("training.tooltip.bilateral_grid_x")
            .precision(0)
            .ui_step(1)
            .all_strategies()
            .int_prop(&OptimizationParameters::bilateral_grid_Y,
                      "bilateral_grid_y", "Grid Y", d.bilateral_grid_Y, 4, 64,
                      "Bilateral grid Y resolution")
            .json_key("bilateral_grid_Y")
            .locale("training.bilateral.grid_y")
            .tooltip("training.tooltip.bilateral_grid_y")
            .precision(0)
            .ui_step(1)
            .all_strategies()
            .int_prop(&OptimizationParameters::bilateral_grid_W,
                      "bilateral_grid_w", "Grid W", d.bilateral_grid_W, 2, 32,
                      "Bilateral grid intensity bins")
            .json_key("bilateral_grid_W")
            .locale("training.bilateral.grid_w")
            .tooltip("training.tooltip.bilateral_grid_w")
            .precision(0)
            .ui_step(1)
            .all_strategies()
            .float_prop(&OptimizationParameters::bilateral_grid_lr,
                        "bilateral_grid_lr", "Grid LR", d.bilateral_grid_lr, 0.0f, 0.1f,
                        "Bilateral grid learning rate")
            .locale("training.bilateral.learning_rate")
            .tooltip("training.tooltip.bilateral_grid_lr")
            .precision(6)
            .ui_step(0.00001)
            .all_strategies()
            .float_prop(&OptimizationParameters::tv_loss_weight,
                        "tv_loss_weight", "TV Loss Weight", d.tv_loss_weight, 0.0f, 100.0f,
                        "Total variation loss weight")
            .locale("training.losses.tv_loss_weight")
            .tooltip("training.tooltip.tv_loss_weight")
            .precision(1)
            .ui_step(0.5)

            // Strategy
            .all_strategies()
            .string_prop(&OptimizationParameters::strategy,
                         "strategy", "Strategy", d.strategy,
                         "Optimization strategy: mcmc, mrnf, or igs+")
            .flags(PROP_NEEDS_RESTART)

            // Shared densification parameters
            .all_strategies()
            .float_prop(&OptimizationParameters::prune_opacity,
                        "prune_opacity", "Prune Opacity", d.prune_opacity, 0.0f, std::numeric_limits<float>::infinity(),
                        "Opacity threshold for pruning")
            .locale("training.thresholds.prune_opacity")
            .tooltip("training.tooltip.prune_opacity")
            .precision(4)
            .ui_step(0.001)
            .strategies({"igs+"})
            .all_strategies()
            .size_prop(&OptimizationParameters::reset_every,
                       "reset_every", "Reset Every", d.reset_every, 100, 10000,
                       "Iteration interval for opacity reset")
            .locale("training.refinement.reset_every")
            .tooltip("training.tooltip.reset_every")
            .precision(0)
            .ui_step(100)
            .strategies({"igs+"})
            // MRNF strategy parameters
            .all_strategies()
            .float_prop(&OptimizationParameters::growth_grad_threshold,
                        "growth_grad_threshold", "Growth Grad Threshold", d.growth_grad_threshold, 0.0f, 1.0f,
                        "Min refine weight for growth candidacy (MRNF)")
            .locale("training.advanced.growth_grad_threshold")
            .tooltip("training.tooltip.growth_grad_threshold")
            .precision(4)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::grow_fraction,
                        "grow_fraction", "Grow Fraction", d.grow_fraction, 0.0f, 1.0f,
                        "Fraction of above-threshold splats to grow (MRNF)")
            .locale("training.advanced.grow_fraction")
            .tooltip("training.tooltip.grow_fraction")
            .precision(3)
            .ui_step(0.01)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .size_prop(&OptimizationParameters::grow_until_iter,
                       "grow_until_iter", "Grow Until Iter", d.grow_until_iter, 0, 100000,
                       "Stop MRNF growth after this iteration")
            .locale("training.refinement.grow_until_iter")
            .tooltip("training.tooltip.grow_until_iter")
            .precision(0)
            .ui_step(1000)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::opacity_decay,
                        "opacity_decay", "Opacity Decay", d.opacity_decay, 0.0f, 0.1f,
                        "Opacity decay rate per refine (MRNF)")
            .locale("training.advanced.opacity_decay")
            .tooltip("training.tooltip.opacity_decay")
            .precision(4)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::scale_decay,
                        "scale_decay", "Scale Decay", d.scale_decay, 0.0f, 0.1f,
                        "Scale decay rate per refine (MRNF)")
            .locale("training.advanced.scale_decay")
            .tooltip("training.tooltip.scale_decay")
            .precision(4)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::means_noise_weight,
                        "means_noise_weight", "Means Noise Weight", d.means_noise_weight, 0.0f, 200.0f,
                        "Exploration noise multiplier for means updates (MRNF)")
            .locale("training.advanced.means_noise_weight")
            .tooltip("training.tooltip.means_noise_weight")
            .precision(2)
            .ui_step(1.0)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::bounds_percentile,
                        "bounds_percentile", "Bounds Percentile", d.bounds_percentile, 0.5f, 1.0f,
                        "Percentile for bounds computation (MRNF)")
            .locale("training.advanced.bounds_percentile")
            .tooltip("training.tooltip.bounds_percentile")
            .precision(3)
            .ui_step(0.01)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_error_map,
                       "use_error_map", "Error Map", d.use_error_map,
                       "Weight MRNF refine signal by per-pixel SSIM error map")
            .locale("training.advanced.use_error_map")
            .tooltip("training.tooltip.use_error_map")
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_edge_map,
                       "use_edge_map", "Edge Map", d.use_edge_map,
                       "Weight MRNF refine signal by Sobel edge map on GT images")
            .locale("training.advanced.use_edge_map")
            .tooltip("training.tooltip.use_edge_map")
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})

            // Flags
            .all_strategies()
            .bool_prop(&OptimizationParameters::mip_filter,
                       "mip_filter", "Mip Filter", d.mip_filter,
                       "Enable mip filtering (anti-aliasing)")
            .locale("training_params.mip_filter")
            .tooltip("training.tooltip.mip_filter")
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_ppisp,
                       "ppisp", "PPISP", d.use_ppisp,
                       "Enable per-camera physically plausible image signal processing")
            .json_key("use_ppisp")
            .locale("training_params.ppisp")
            .tooltip("training.tooltip.ppisp")
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_exposure_from_exif,
                       "ppisp_exposure_from_exif", "EXIF Exposure", d.ppisp_exposure_from_exif,
                       "Seed per-frame PPISP exposure from image EXIF")
            .locale("training_params.ppisp_exposure_from_exif")
            .tooltip("training.tooltip.ppisp_exposure_from_exif")
            .all_strategies()
            .float_prop(&OptimizationParameters::ppisp_lr,
                        "ppisp_lr", "PPISP Learning Rate", d.ppisp_lr, 0.0f, 0.1f,
                        "Learning rate for PPISP parameters")
            .locale("training_params.ppisp_lr")
            .tooltip("training.tooltip.ppisp_lr")
            .precision(5)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::ppisp_reg_weight,
                        "ppisp_reg_weight", "PPISP Regularization", d.ppisp_reg_weight, 0.0f, 0.1f,
                        "Regularization weight for PPISP parameters")
            .locale("training_params.ppisp_reg")
            .tooltip("training.tooltip.ppisp_reg")
            .precision(5)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .int_prop(&OptimizationParameters::ppisp_warmup_steps,
                      "ppisp_warmup_steps", "PPISP Warmup Steps", d.ppisp_warmup_steps, 0, 100000,
                      "Steps before PPISP training starts")
            .locale("training_params.ppisp_warmup")
            .tooltip("training.tooltip.ppisp_warmup")
            .precision(0)
            .ui_step(100)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_use_controller,
                       "ppisp_use_controller", "Controller", d.ppisp_use_controller,
                       "Enable PPISP controller for novel view synthesis")
            .locale("training_params.ppisp_controller")
            .tooltip("training.tooltip.ppisp_controller")
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_freeze_from_sidecar,
                       "ppisp_freeze_from_sidecar", "Freeze From Sidecar", d.ppisp_freeze_from_sidecar,
                       "Load PPISP weights from a sidecar and freeze PPISP learning during training")
            .locale("training_params.ppisp_freeze_from_sidecar")
            .tooltip("training.tooltip.ppisp_freeze_from_sidecar")
            .all_strategies()
            .int_prop(&OptimizationParameters::ppisp_controller_activation_step,
                      "ppisp_controller_activation_step", "Controller Step", d.ppisp_controller_activation_step, -1, 100000,
                      "Iteration to start controller distillation (negative = final 5000 planned steps)")
            .all_strategies()
            .float_prop(&OptimizationParameters::ppisp_controller_lr,
                        "ppisp_controller_lr", "Controller LR", d.ppisp_controller_lr, 1e-5f, 1e-1f,
                        "Learning rate for PPISP controller")
            .locale("training_params.ppisp_controller_lr")
            .tooltip("training.tooltip.ppisp_controller_lr")
            .precision(5)
            .ui_step(0.0001)
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_freeze_gaussians_on_distill,
                       "ppisp_freeze_gaussians", "Freeze Gaussians", d.ppisp_freeze_gaussians_on_distill,
                       "Freeze Gaussians during controller distillation")
            .json_key("ppisp_freeze_gaussians_on_distill")
            .locale("training_params.ppisp_freeze_gaussians")
            .tooltip("training.tooltip.ppisp_freeze_gaussians")
            .all_strategies()
            .bool_prop(&OptimizationParameters::bg_modulation,
                       "bg_modulation", "BG Modulation", d.bg_modulation,
                       "Enable sinusoidal background modulation")
            .all_strategies()
            .bool_prop(&OptimizationParameters::headless,
                       "headless", "Headless", d.headless,
                       "Run without visualization")
            .flags(PROP_READONLY)
            .all_strategies()
            .bool_prop(&OptimizationParameters::enable_eval,
                       "enable_eval", "Enable Eval", d.enable_eval,
                       "Run evaluation at specified steps")
            .locale("training_params.enable_eval")
            .tooltip("training.tooltip.enable_eval")

            // Random initialization
            .all_strategies()
            .bool_prop(&OptimizationParameters::random,
                       "random", "Random Init", d.random,
                       "Use random initialization instead of SfM")
            .locale("training.init.random_init")
            .tooltip("training.tooltip.random_init")
            .flags(PROP_NEEDS_RESTART)
            .all_strategies()
            .int_prop(&OptimizationParameters::init_num_pts,
                      "init_num_pts", "Init Points", d.init_num_pts, 1000, 1000000,
                      "Number of random points to initialize")
            .locale("training.init.num_points")
            .tooltip("training.tooltip.num_points")
            .precision(0)
            .ui_step(10000)
            .all_strategies()
            .float_prop(&OptimizationParameters::init_extent,
                        "init_extent", "Init Extent", d.init_extent, 0.1f, 10.0f,
                        "Extent of random point cloud")
            .locale("training.init.extent")
            .tooltip("training.tooltip.extent")
            .precision(1)
            .ui_step(0.5)

            // Sparsity
            .all_strategies()
            .bool_prop(&OptimizationParameters::enable_sparsity,
                       "enable_sparsity", "Enable Sparsity", d.enable_sparsity,
                       "Enable sparsity optimization")
            .locale("training_params.sparsity")
            .tooltip("training.tooltip.sparsity")
            .all_strategies()
            .int_prop(&OptimizationParameters::sparsify_steps,
                      "sparsify_steps", "Sparsify Steps", d.sparsify_steps, 1000, 50000,
                      "Number of sparsification steps to run after regular training")
            .locale("training_params.sparsify_steps")
            .tooltip("training.tooltip.sparsify_steps")
            .precision(0)
            .ui_step(1000)
            .all_strategies()
            .float_prop(&OptimizationParameters::prune_ratio,
                        "prune_ratio", "Prune Ratio", d.prune_ratio, 0.0f, 1.0f,
                        "Target pruning ratio for sparsification")
            .all_strategies()
            .float_prop(&OptimizationParameters::init_rho,
                        "init_rho", "Init Rho", d.init_rho, 0.0f, 0.01f,
                        "Initial ADMM penalty rho for sparsity optimization")
            .locale("training_params.init_rho")
            .tooltip("training.tooltip.init_rho")
            .precision(4)
            .ui_step(0.001)
            .all_strategies()
            .float_prop(&OptimizationParameters::steps_scaler,
                        "steps_scaler", "Steps Scaler", d.steps_scaler, 0.0f, 10.0f,
                        "Scale training step counts")
            .locale("training_params.steps_scaler")
            .tooltip("training.tooltip.steps_scaler")
            .precision(2)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::gut,
                       "gut", "GUT", d.gut,
                       "Gaussian Unscented Transform")
            .locale("training_params.gut")
            .tooltip("training.tooltip.gut")
            .all_strategies()
            .bool_prop(&OptimizationParameters::undistort,
                       "undistort", "Undistort", d.undistort,
                       "Undistort images on-the-fly before training")
            .locale("training_params.undistort")
            .tooltip("training.tooltip.undistort")
            .flags(PROP_NEEDS_RESTART)
            .all_strategies()
            .enum_prop(&OptimizationParameters::bg_mode,
                       "bg_mode", "Background Mode", d.bg_mode,
                       {{"SolidColor", BackgroundMode::SolidColor, "training.options.bg.color", "solid_color"},
                        {"Modulation", BackgroundMode::Modulation, "training.options.bg.modulation", "modulation"},
                        {"Image", BackgroundMode::Image, "training.options.bg.image", "image"},
                        {"Random", BackgroundMode::Random, "training.options.bg.random", "random"}},
                       "Background mode")
            .locale("training_params.bg_mode")
            .tooltip("training.tooltip.bg_modulation")
            .all_strategies()
            .build();
    }

    namespace {
        std::mutex optimization_registration_mutex;
    }

    void ensure_optimization_properties_registered() {
        auto& registry = PropertyRegistry::instance();
        if (registry.get_group_snapshot("optimization"))
            return;

        std::lock_guard lock(optimization_registration_mutex);
        if (!registry.get_group_snapshot("optimization"))
            register_optimization_properties();
    }

} // namespace lfs::core::param

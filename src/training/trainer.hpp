/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "checkpoint.hpp"
#include "components/bilateral_grid.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "components/sparsity_optimizer.hpp"
#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "dataset.hpp"
#include "kernels/depth_loss.hpp"
#include "lfs/kernels/ssim.cuh"
#include "losses/photometric_loss.hpp"
#include "metrics/metrics.hpp"
#include "optimizer/scheduler.hpp"
#include "progress.hpp"
#include "strategies/istrategy.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stop_token>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace lfs::core {
    class Scene;
}

namespace lfs::training {
    class AdamOptimizer;
    struct PPISPFileMetadata;

    struct PPISPViewportOverrides {
        // Exposure
        float exposure_offset = 0.0f;

        // Vignetting
        bool vignette_enabled = true;
        float vignette_strength = 1.0f;

        // Color correction
        float wb_temperature = 0.0f;
        float wb_tint = 0.0f;
        float color_red_x = 0.0f;
        float color_red_y = 0.0f;
        float color_green_x = 0.0f;
        float color_green_y = 0.0f;
        float color_blue_x = 0.0f;
        float color_blue_y = 0.0f;

        // CRF
        float gamma_multiplier = 1.0f;
        float gamma_red = 0.0f;
        float gamma_green = 0.0f;
        float gamma_blue = 0.0f;
        float crf_toe = 0.0f;
        float crf_shoulder = 0.0f;

        [[nodiscard]] bool isIdentity() const {
            return exposure_offset == 0.0f && vignette_enabled && vignette_strength == 1.0f &&
                   wb_temperature == 0.0f && wb_tint == 0.0f && color_red_x == 0.0f && color_red_y == 0.0f &&
                   color_green_x == 0.0f && color_green_y == 0.0f && color_blue_x == 0.0f && color_blue_y == 0.0f &&
                   gamma_multiplier == 1.0f && gamma_red == 0.0f && gamma_green == 0.0f && gamma_blue == 0.0f &&
                   crf_toe == 0.0f && crf_shoulder == 0.0f;
        }
    };

    class Trainer {
    public:
        struct GTLoadConfigSnapshot {
            int resize_factor = 1;
            int max_width = 0;
            bool undistort = false;
        };

        struct CameraMetricsAppearanceConfig {
            bool enabled = false;
            PPISPViewportOverrides overrides{};
            bool use_controller = true;
        };

        struct CameraMetricsSnapshot {
            float psnr = 0.0f;
            std::optional<float> ssim;
            bool used_mask = false;
        };

        // Legacy constructor - takes ownership of strategy and shares datasets
        Trainer(std::shared_ptr<CameraDataset> dataset,
                std::unique_ptr<IStrategy> strategy,
                std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits);

        /**
         * @brief Constructor - takes Scene reference (Scene owns all data)
         *
         * Scene provides:
         * - Training model via getTrainingModel() (SplatData)
         * - Cameras via getAllCameras() (from CAMERA nodes)
         */
        Trainer(lfs::core::Scene& scene);

        // Delete copy operations
        Trainer(const Trainer&) = delete;

        Trainer& operator=(const Trainer&) = delete;

        // Allow move operations
        Trainer(Trainer&&) = default;

        Trainer& operator=(Trainer&&) = default;

        ~Trainer();

        // Initialize trainer - must be called before training
        std::expected<void, std::string> initialize(const lfs::core::param::TrainingParameters& params);

        // Check if trainer is initialized
        bool isInitialized() const { return initialized_.load(); }

        // Main training method with stop token support
        std::expected<void, std::string> train(std::stop_token stop_token = {});

        // Control methods for GUI interaction
        void request_pause() { pause_requested_ = true; }
        void request_resume() { pause_requested_ = false; }
        void request_save() { save_requested_ = true; }
        void request_stop() { stop_requested_ = true; }

        bool is_paused() const { return is_paused_.load(); }
        bool is_running() const { return is_running_.load(); }
        bool is_training_complete() const { return training_complete_.load(); }
        bool has_stopped() const { return stop_requested_.load(); }

        // Set Python script paths to execute once before training; scripts register per-iteration callbacks.
        void set_python_scripts(std::vector<std::filesystem::path> scripts) {
            python_scripts_ = std::move(scripts);
        }

        // Get current training state
        int get_current_iteration() const { return current_iteration_.load(); }
        int get_total_iterations() const;
        std::filesystem::path get_output_path() const { return getParams().dataset.output_path; }
        float get_current_loss() const { return current_loss_.load(); }
        bool fillCameraLossColors(const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
                                  std::vector<std::array<float, 3>>& colors) const;

        // just for viewer to get model
        const IStrategy& get_strategy() const { return *strategy_; }

        // Mutable access for controlled callbacks (e.g., Python control layer)
        IStrategy& get_strategy_mutable() { return *strategy_; }

        // Allow viewer to lock for rendering
        std::shared_mutex& getRenderMutex() const { return render_mutex_; }
        // Held shared by viewer/metric readers around their model read and
        // exclusive by the trainer around the non-refining in-place optimizer
        // step, so a reader can never observe a half-written model. Distinct from
        // render_mutex_ (taken only on refining/topology steps) and never held
        // across a synchronous readback, so it avoids the startup deadlock that
        // gating render_mutex_ every step would hit. The GPU edges still order the
        // actual reads/writes; this lock just makes their setup mutually
        // exclusive.
        std::shared_mutex& getModelAccessMutex() const { return model_access_mutex_; }

        // GPU-side model-read handshake. Call both under a shared lock on
        // getRenderMutex(), bracketing every GPU read of the live model enqueued
        // on reader_stream: beginModelRead orders the reads after the last
        // consistent parameter state; endModelRead records the reads so the next
        // optimizer step waits for them (GPU-side, no CPU blocking).
        void beginModelRead(cudaStream_t reader_stream);
        void endModelRead(cudaStream_t reader_stream);

        cudaStream_t trainingStream() const { return training_stream_; }

        // Reverse edge for the zero-copy viewport: the viewer's render-complete
        // timeline imported into CUDA, plus the latest timeline value covering
        // submits that bound live training storage. The trainer waits the value
        // on its stream before the next step's in-place writes.
        void setViewerReleaseFence(cudaExternalSemaphore_t semaphore);
        void publishViewerBorrow(uint64_t value);

        lfs::core::param::TrainingParameters getParams() const {
            std::lock_guard<std::mutex> lock(params_mutex_);
            return params_;
        }
        void setParams(const lfs::core::param::TrainingParameters& params);
        void setSplatTensorAllocator(lfs::core::SplatTensorAllocator allocator) {
            splat_tensor_allocator_ = std::move(allocator);
        }

        void setOnIterationStart(std::function<void()> cb) { on_iteration_start_ = std::move(cb); }

        lfs::core::Scene* getScene() const { return scene_; }
        std::shared_ptr<lfs::io::PipelinedImageLoader> getActiveImageLoader() const;
        GTLoadConfigSnapshot getGTLoadConfigSnapshot() const;
        std::expected<CameraMetricsSnapshot, std::string> computeCameraMetrics(
            const lfs::core::Camera& camera,
            bool include_ssim,
            CameraMetricsAppearanceConfig appearance);

        /// Apply PPISP correction to a rendered image for viewport display
        /// @param rgb rendered image [C,H,W] or [H,W,C]
        /// @param camera_uid camera UID (-1 for novel view)
        /// @param overrides user-controlled adjustments (exposure, vignette, WB, gamma)
        /// @param use_controller if true, use controller for novel views; if false, use learned params
        /// @return corrected image, or input if PPISP not enabled
        lfs::core::Tensor applyPPISPForViewport(const lfs::core::Tensor& rgb, int camera_uid,
                                                const PPISPViewportOverrides& overrides = {},
                                                bool use_controller = true) const;

        /// Check if PPISP is enabled, initialized, and ready for rendering
        bool hasPPISP() const {
            const auto params = getParams();
            return ppisp_ != nullptr && params.optimization.use_ppisp && ppisp_->isFinalized();
        }

        /// Check if PPISP controller is enabled and ready for novel views
        bool hasPPISPController() const {
            const auto params = getParams();
            return ppisp_controller_pool_ != nullptr && params.optimization.ppisp_use_controller;
        }

        PPISPControllerPool* getPPISPControllerPool() const { return ppisp_controller_pool_.get(); }
        PPISP* getPPISP() const { return ppisp_.get(); }
        std::unique_ptr<PPISP> takePPISP() { return std::move(ppisp_); }
        std::unique_ptr<PPISPControllerPool> takePPISPControllerPool() { return std::move(ppisp_controller_pool_); }

        // Checkpoint methods
        std::expected<void, std::string> save_checkpoint(int iteration);
        std::expected<void, std::string> save_checkpoint_to(const std::filesystem::path& output_path, int iteration);
        std::expected<int, std::string> load_checkpoint(const std::filesystem::path& checkpoint_path);
        void save_final_ply_and_checkpoint(int iteration);

        // Orderly shutdown - GPU sync, wait for async saves, release resources. Idempotent.
        void shutdown();

    private:
        // Helper for deferred event emission to prevent deadlocks
        struct DeferredEvents {
            std::vector<std::function<void()>> events;

            template <typename Event>
            void add(Event&& e) {
                events.push_back([e = std::move(e)]() { e.emit(); });
            }

            ~DeferredEvents() {
                for (auto& e : events)
                    e();
            }
        };

        // Training step result
        enum class StepResult {
            Continue,
            Stop,
            RetryAfterOom
        };

        // Returns the background color to use at a given iteration
        lfs::core::Tensor& background_for_step(int iter);

        // Returns the resized background image for the given camera dimensions
        // Returns empty tensor if no background image is set
        lfs::core::Tensor get_background_image_for_camera(int width, int height);
        void clearBackgroundImageCache();

        lfs::core::Tensor get_random_background_for_camera(int width, int height, int iteration);

        // Protected method for processing a single training step
        std::expected<StepResult, std::string> train_step(
            int iter,
            lfs::core::Camera* cam,
            lfs::core::Tensor gt_image,
            RenderMode render_mode,
            std::stop_token stop_token = {});

        void setActiveImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader> loader);
        int get_regular_iterations() const;
        int get_active_sparsify_steps() const;
        int get_sparsity_boundary_iteration() const;
        lfs::core::param::OptimizationParameters get_runtime_optimization_params() const;
        void sync_strategy_optimization_params();
        std::expected<void, std::string> initialize_camera_loss_heatmap(
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras);
        void update_camera_loss_heatmap(const lfs::core::Camera& camera,
                                        const lfs::core::Tensor& image_loss);
        void maybe_publish_camera_loss_heatmap(int iter, bool force = false);
        void publish_camera_loss_heatmap_snapshot();

        struct PhotometricLossResult {
            lfs::core::Tensor loss;
            lfs::core::Tensor grad_corrected;
            lfs::core::Tensor grad_raw;
        };

        // Compute photometric loss AND gradient manually (no autograd)
        // Returns GPU tensors for loss and gradients (avoid sync!)
        std::expected<PhotometricLossResult, std::string> compute_photometric_loss_with_gradient(
            const lfs::core::Tensor& corrected,
            const lfs::core::Tensor& gt_image,
            const lfs::core::param::OptimizationParameters& opt_params,
            const lfs::core::Tensor& raw_rendered);

        struct MaskLossResult {
            lfs::core::Tensor loss;
            lfs::core::Tensor grad_corrected;
            lfs::core::Tensor grad_raw;
            lfs::core::Tensor grad_alpha;
        };

        // Masked photometric loss with optional alpha gradient
        std::expected<MaskLossResult, std::string> compute_photometric_loss_with_mask(
            const lfs::core::Tensor& corrected,
            const lfs::core::Tensor& gt_image,
            const lfs::core::Tensor& mask,
            const lfs::core::Tensor& alpha,
            const lfs::core::param::OptimizationParameters& opt_params,
            const lfs::core::Tensor& raw_rendered);

        // Validate masks exist for all cameras when mask mode is enabled
        std::expected<void, std::string> validate_masks();

        // Returns GPU tensor for loss (avoid sync!)
        std::expected<lfs::core::Tensor, std::string> compute_scale_reg_loss(
            lfs::core::SplatData& splatData,
            AdamOptimizer& optimizer,
            const lfs::core::param::OptimizationParameters& opt_params);

        // Returns GPU tensor for loss (avoid sync!)
        std::expected<lfs::core::Tensor, std::string> compute_opacity_reg_loss(
            lfs::core::SplatData& splatData,
            AdamOptimizer& optimizer,
            const lfs::core::param::OptimizationParameters& opt_params);

        // Sparsity optimization - returns GPU tensor (no CPU sync)
        std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string> compute_sparsity_loss_forward(
            const int iter, const lfs::core::SplatData& splat_data);

        std::expected<void, std::string> handle_sparsity_update(const int iter, lfs::core::SplatData& splat_data);
        std::expected<void, std::string> apply_sparsity_pruning(const int iter, lfs::core::SplatData& splat_data);

        // Cleanup method for re-initialization
        void cleanup();

        std::expected<void, std::string> initialize_bilateral_grid();
        std::expected<void, std::string> initialize_ppisp();
        std::expected<void, std::string> initialize_ppisp_controller();
        std::expected<void, std::string> apply_ppisp_sidecar_if_configured();
        std::expected<PPISPFileMetadata, std::string> build_ppisp_sidecar_metadata() const;
        struct PPISPSidecarMappings {
            std::vector<int> frame_mapping;
            std::vector<int> camera_mapping;
        };
        std::expected<PPISPSidecarMappings, std::string> build_ppisp_sidecar_mappings(
            const PPISP& loaded_ppisp,
            const PPISPFileMetadata& metadata,
            const std::filesystem::path& sidecar_path) const;
        [[nodiscard]] bool is_ppisp_frozen() const {
            const auto params = getParams();
            return params.optimization.use_ppisp &&
                   params.optimization.ppisp_freeze_from_sidecar;
        }
        [[nodiscard]] bool should_apply_ppisp_sidecar_on_init() const {
            const auto params = getParams();
            return params.optimization.use_ppisp &&
                   params.optimization.ppisp_freeze_from_sidecar &&
                   !params.resume_checkpoint.has_value() &&
                   !params.optimization.ppisp_sidecar_path.empty();
        }
        [[nodiscard]] PPISPControllerPool* controller_pool_for_save(int iteration) const;
        [[nodiscard]] lfs::core::param::TrainingParameters params_for_checkpoint_save() const;
        [[nodiscard]] TrainingProgress::Phase get_progress_phase(
            int iter,
            bool in_controller_phase = false) const;

        // Handle control requests
        void handle_control_requests(int iter, std::stop_token stop_token = {});
        void apply_pending_params_at_safe_point();
        void apply_param_side_effects(
            const lfs::core::param::TrainingParameters& params,
            bool background_image_path_changed);

        std::expected<void, std::string> save_ply(const std::filesystem::path& save_path,
                                                  const std::string& filename,
                                                  int iter_num,
                                                  bool join_threads = true,
                                                  bool save_checkpoint = true);
        void updateGTLoadConfigSnapshot();
        void clearActiveImageLoader();

        struct CameraLossHeatmapState {
            std::vector<int> camera_uids;
            std::unordered_map<int, std::size_t> uid_to_slot;
            lfs::core::Tensor latest_loss_gpu;
            lfs::core::Tensor ema_loss_gpu;
            lfs::core::Tensor ema_loss_stage_cpu;
            std::vector<std::array<float, 3>> published_colors;
            std::vector<uint8_t> published_valid;
            mutable std::shared_mutex snapshot_mutex;
            cudaStream_t copy_stream = nullptr;
            cudaEvent_t ready_event = nullptr;
            cudaEvent_t done_event = nullptr;
            cudaStream_t producer_stream = nullptr;
            bool copy_in_flight = false;
            bool dirty = false;

            ~CameraLossHeatmapState();
        };

        std::shared_ptr<CameraLossHeatmapState> getCameraLossHeatmap() const;
        void setCameraLossHeatmap(std::shared_ptr<CameraLossHeatmapState> heatmap);
        std::expected<void, std::string> ensureModelTensorAllocatorStorage(
            lfs::core::SplatData& model,
            std::string_view reason);

        lfs::core::Scene* scene_ = nullptr;
        std::shared_ptr<CameraDataset> base_dataset_;
        std::shared_ptr<CameraDataset> train_dataset_;
        std::shared_ptr<CameraDataset> val_dataset_;
        std::shared_ptr<lfs::io::PipelinedImageLoader> active_image_loader_;
        std::unique_ptr<IStrategy> strategy_;
        // Hot-loop reads use params_ without locking. Active updates therefore
        // coalesce here and are installed only by the worker at safe boundaries.
        mutable std::mutex params_mutex_;
        lfs::core::param::TrainingParameters params_;
        std::optional<lfs::core::param::TrainingParameters> pending_params_;
        lfs::core::SplatTensorAllocator splat_tensor_allocator_;
        std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits_;

        lfs::core::Tensor background_{};
        lfs::core::Tensor bg_mix_buffer_;
        lfs::core::Tensor bg_image_base_{}; // Original background image [C, H, W]
        struct BackgroundImageCacheEntry {
            lfs::core::Tensor tensor;
            size_t allocation_bytes = 0;
            uint64_t last_used = 0;
        };
        // Resized backgrounds are bounded by physical bucket size, not entry count.
        static constexpr size_t BG_IMAGE_CACHE_BUDGET_BYTES = 256ULL * 1024 * 1024;
        std::unordered_map<uint64_t, BackgroundImageCacheEntry> bg_image_cache_;
        size_t bg_image_cache_bytes_ = 0;
        uint64_t bg_image_cache_clock_ = 0;
        lfs::core::Tensor random_bg_buffer_{}; // Reusable buffer for random background
        std::unique_ptr<TrainingProgress> progress_;
        size_t train_dataset_size_ = 0;
        size_t total_cameras_count_ = 0;
        std::shared_ptr<CameraLossHeatmapState> camera_loss_heatmap_;

        // Pre-loaded mask from pipelined dataloader (used in train_step)
        lfs::core::Tensor pipelined_mask_;
        lfs::core::Tensor pipelined_depth_;
        lfs::core::Tensor pipelined_normal_;

        // Bilateral grid for appearance modeling (optional)
        std::unique_ptr<BilateralGrid> bilateral_grid_;

        // PPISP for physically-plausible ISP appearance modeling (optional)
        std::unique_ptr<PPISP> ppisp_;

        // PPISP controller pool for novel view synthesis (Phase 2 distillation)
        // Shared CNN and per-camera FC weights for memory efficiency
        std::unique_ptr<PPISPControllerPool> ppisp_controller_pool_;

        std::unique_ptr<ISparsityOptimizer> sparsity_optimizer_;

        // Persistent photometric loss (workspace reuse across iterations)
        lfs::training::losses::PhotometricLoss photometric_loss_;

        // Cached GPU scalar to avoid per-iteration allocation
        core::Tensor loss_accumulator_;
        core::Tensor depth_loss_scalar_;
        core::Tensor depth_loss_grad_;
        core::Tensor depth_loss_grad_alpha_;
        core::Tensor depth_loss_partials_;
        core::Tensor normal_loss_scalar_;
        core::Tensor normal_loss_grad_;
        core::Tensor normal_loss_partials_;
        core::Tensor normal_consistency_scalar_;
        core::Tensor normal_consistency_partials_;
        core::Tensor normal_prior_depth_scalar_;
        // Dataset-level normal-prior convention, resolved once at startup
        bool normal_prior_flip_yz_ = false;
        bool normal_prior_world_space_ = false;
        bool normal_prior_usable_ = true;
        bool normal_prior_srgb_ = false;
        // Prior-world -> reconstruction-world rotation (row-major), identity
        // for camera-space priors.
        std::array<float, 9> normal_prior_world_rotation_{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        std::unordered_map<int, lfs::training::kernels::DepthAnchor> depth_anchors_;
        bool depth_anchor_fit_attempted_ = false;
        lfs::training::kernels::DepthPriorType resolved_depth_prior_ =
            lfs::training::kernels::DepthPriorType::Auto;

        // Pre-allocated SSIM-map workspace for densification error maps.
        lfs::training::kernels::SSIMMapWorkspace densification_ssim_workspace_;
        lfs::training::kernels::MaskedFusedL1SSIMWorkspace masked_fused_workspace_;
        lfs::training::kernels::DecoupledFusedL1SSIMWorkspace decoupled_fused_workspace_;
        lfs::training::kernels::MaskedDecoupledFusedL1SSIMWorkspace masked_decoupled_fused_workspace_;

        // Pre-allocated error map buffer for densification (avoids per-iteration allocation)
        core::Tensor densification_error_map_;

        // Reusable buffer for Sobel edge map (lfs edge-importance densification)
        core::Tensor edge_map_buffer_;

        // Metrics evaluator - handles all evaluation logic
        std::unique_ptr<lfs::training::MetricsEvaluator> evaluator_;

        // Single mutex that protects the model during training
        mutable std::shared_mutex render_mutex_;
        mutable std::shared_mutex model_access_mutex_;

        // Mutex for initialization to ensure thread safety
        mutable std::mutex init_mutex_;
        mutable std::mutex active_image_loader_mutex_;
        mutable std::mutex camera_loss_heatmap_mutex_;
        mutable std::mutex gt_load_config_mutex_;

        // Control flags for thread communication
        std::atomic<bool> pause_requested_{false};
        std::atomic<bool> save_requested_{false};
        std::atomic<bool> stop_requested_{false};
        std::atomic<bool> is_paused_{false};
        std::atomic<bool> is_running_{false};
        std::atomic<bool> training_complete_{false};
        std::atomic<bool> ready_to_start_{false};
        std::atomic<bool> initialized_{false};
        std::atomic<bool> shutdown_complete_{false};

        // Current training state
        std::atomic<int> current_iteration_{0};
        std::atomic<float> current_loss_{0.0f};

        // Async callback system
        std::function<void()> callback_;
        std::atomic<bool> callback_busy_{false};
        cudaStream_t callback_stream_ = nullptr;

        // Dedicated stream for all training-thread GPU work, installed as the
        // thread's current stream in train().
        cudaStream_t training_stream_ = nullptr;

        // Non-blocking stream for on-demand GUI metric renders
        // (computeCameraMetrics, called from the UI thread). Lets PSNR/SSIM
        // tensor work overlap training and keeps the metric render's arena
        // frame off a device-sync fallback.
        cudaStream_t metrics_stream_ = nullptr;

        // Trainer↔viewer GPU handshake. Forward edge: params_ready_event_ marks
        // a consistent end-of-step parameter state; readers wait on it before
        // enqueuing reads (beginModelRead). Reverse edges: reader_done ring
        // events plus the viewer's exported release fence gate the next step's
        // in-place writes behind in-flight reads — GPU-side only, no CPU stall.
        // Lock order: render_mutex_ → stream_sync_mutex_ (leaf; only CUDA
        // record/wait calls under it).
        static constexpr size_t READER_DONE_RING = 4;
        cudaEvent_t params_ready_event_ = nullptr;
        bool params_ready_recorded_ = false;
        std::array<cudaEvent_t, READER_DONE_RING> reader_done_events_{};
        uint32_t reader_done_head_ = 0;
        uint32_t reader_done_pending_ = 0;
        cudaExternalSemaphore_t viewer_release_semaphore_ = nullptr;
        std::atomic<uint64_t> viewer_borrow_value_{0};
        uint64_t viewer_borrow_waited_ = 0;
        mutable std::mutex stream_sync_mutex_;

        void createCudaResources();
        void createSyncPrimitives();
        void destroySyncPrimitives();
        void recordParamsReady();
        void waitForModelReaders();
        void fitDepthAnchors(size_t cameras_with_depth);

        // Async loss readback: the periodic loss sample is copied D2H into a
        // small pinned ring and polled on later iterations instead of stalling
        // the pipeline with .item(). NaN/Inf detection lags by at most
        // LOSS_RING * LOSS_SYNC_INTERVAL iterations.
        static constexpr size_t LOSS_RING = 4;
        struct LossReadbackSlot {
            float* pinned = nullptr;
            cudaEvent_t done = nullptr;
            int iter = 0;
            bool in_flight = false;
        };
        std::array<LossReadbackSlot, LOSS_RING> loss_slots_{};
        size_t loss_slot_head_ = 0;

        void submitLossReadback(const lfs::core::Tensor& total_loss, int iter);
        std::expected<void, std::string> harvestLossReadbacks(bool drain, bool in_controller_phase);

        // Python control scripts (file paths) to execute before training starts
        std::vector<std::filesystem::path> python_scripts_;

        std::function<void()> on_iteration_start_;
        GTLoadConfigSnapshot gt_load_config_snapshot_;
    };
} // namespace lfs::training

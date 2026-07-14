/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/sequencer_ui_manager.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/rml_sequencer_overlay.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/string_keys.hpp"
#include "gui/translation_gizmo.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "io/loader.hpp"
#include "io/video/video_export_options.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/interpolation.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/timeline_view_math.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace lfs::vis::gui {

    namespace {
        constexpr size_t MIN_PATH_RENDER_SAMPLES = 128;
        constexpr size_t MAX_PATH_RENDER_SAMPLES = 4096;
        constexpr float PATH_SAMPLES_PER_VIEWPORT_PIXEL = 2.0f;
        constexpr uint32_t PLY_SEQUENCE_CACHE_MAGIC = 0x4C465351; // "LFSQ"
        constexpr uint32_t PLY_SEQUENCE_CACHE_VERSION = 1;

        struct PlySequenceCacheHeader {
            uint32_t magic = PLY_SEQUENCE_CACHE_MAGIC;
            uint32_t version = PLY_SEQUENCE_CACHE_VERSION;
            uint64_t source_size = 0;
            int64_t source_mtime_ns = 0;
            uint64_t source_key = 0;
        };

        [[nodiscard]] uint64_t fnv1a64(std::string_view text) {
            uint64_t hash = 14695981039346656037ull;
            for (const unsigned char c : text) {
                hash ^= static_cast<uint64_t>(c);
                hash *= 1099511628211ull;
            }
            return hash;
        }

        [[nodiscard]] int64_t fileTimeNs(const std::filesystem::path& path) {
            std::error_code ec;
            const auto write_time = std::filesystem::last_write_time(path, ec);
            if (ec)
                return 0;
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       write_time.time_since_epoch())
                .count();
        }

        [[nodiscard]] std::filesystem::path stableAbsolutePath(const std::filesystem::path& path) {
            std::error_code ec;
            auto absolute = std::filesystem::weakly_canonical(path, ec);
            if (!ec && !absolute.empty())
                return absolute;
            absolute = std::filesystem::absolute(path, ec);
            return ec ? path : absolute;
        }

        [[nodiscard]] std::filesystem::path plySequenceCacheRoot() {
#ifdef _WIN32
            if (const char* local_app_data = std::getenv("LOCALAPPDATA");
                local_app_data && *local_app_data) {
                return lfs::core::utf8_to_path(local_app_data) / "LichtFeld" / "ply_sequence_cache";
            }
            if (const char* temp = std::getenv("TEMP"); temp && *temp) {
                return lfs::core::utf8_to_path(temp) / "LichtFeld" / "ply_sequence_cache";
            }
            return std::filesystem::path("C:/Temp/LichtFeld/ply_sequence_cache");
#else
            if (const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
                xdg_cache && *xdg_cache) {
                return lfs::core::utf8_to_path(xdg_cache) / "lichtfeld" / "ply_sequence_cache";
            }
            if (const char* home = std::getenv("HOME"); home && *home) {
                return lfs::core::utf8_to_path(home) / ".cache" / "lichtfeld" / "ply_sequence_cache";
            }
            return std::filesystem::temp_directory_path() / "lichtfeld" / "ply_sequence_cache";
#endif
        }

        [[nodiscard]] std::optional<PlySequenceCacheHeader> makePlySequenceCacheHeader(
            const std::filesystem::path& source_path) {
            std::error_code ec;
            const uint64_t source_size = std::filesystem::file_size(source_path, ec);
            if (ec)
                return std::nullopt;

            const auto absolute = stableAbsolutePath(source_path);
            const int64_t source_mtime_ns = fileTimeNs(source_path);
            const std::string key_text = std::format("{}|{}|{}",
                                                     lfs::core::path_to_utf8(absolute),
                                                     source_size,
                                                     source_mtime_ns);
            return PlySequenceCacheHeader{
                .magic = PLY_SEQUENCE_CACHE_MAGIC,
                .version = PLY_SEQUENCE_CACHE_VERSION,
                .source_size = source_size,
                .source_mtime_ns = source_mtime_ns,
                .source_key = fnv1a64(key_text)};
        }

        [[nodiscard]] std::filesystem::path plySequenceCachePath(
            const PlySequenceCacheHeader& header) {
            return plySequenceCacheRoot() / std::format("{:016x}.lfs_splat", header.source_key);
        }

        [[nodiscard]] bool cacheHeaderMatches(const PlySequenceCacheHeader& actual,
                                              const PlySequenceCacheHeader& expected) {
            return actual.magic == PLY_SEQUENCE_CACHE_MAGIC &&
                   actual.version == PLY_SEQUENCE_CACHE_VERSION &&
                   actual.source_size == expected.source_size &&
                   actual.source_mtime_ns == expected.source_mtime_ns &&
                   actual.source_key == expected.source_key;
        }

        [[nodiscard]] std::unique_ptr<lfs::core::SplatData> loadPlySequenceCache(
            const std::filesystem::path& source_path,
            const lfs::io::SplatTensorAllocator& allocator,
            std::string& error) {
            const auto expected_header = makePlySequenceCacheHeader(source_path);
            if (!expected_header) {
                error = "source metadata unavailable";
                return nullptr;
            }

            const auto cache_path = plySequenceCachePath(*expected_header);
            std::ifstream file;
            if (!lfs::core::open_file_for_read(cache_path, std::ios::binary, file)) {
                error = "cache miss";
                return nullptr;
            }

            PlySequenceCacheHeader actual_header{};
            file.read(reinterpret_cast<char*>(&actual_header), sizeof(actual_header));
            if (!file || !cacheHeaderMatches(actual_header, *expected_header)) {
                error = "stale cache";
                std::error_code ec;
                std::filesystem::remove(cache_path, ec);
                return nullptr;
            }

            try {
                auto model = std::make_unique<lfs::core::SplatData>();
                model->deserialize(file, allocator);
                if (!file) {
                    error = "truncated cache";
                    std::error_code ec;
                    std::filesystem::remove(cache_path, ec);
                    return nullptr;
                }
                return model;
            } catch (const std::exception& e) {
                error = e.what();
                std::error_code ec;
                std::filesystem::remove(cache_path, ec);
                return nullptr;
            }
        }

        [[nodiscard]] bool writePlySequenceCache(const std::filesystem::path& source_path,
                                                 const lfs::core::SplatData& model,
                                                 std::string& error) {
            const auto header = makePlySequenceCacheHeader(source_path);
            if (!header) {
                error = "source metadata unavailable";
                return false;
            }

            const auto cache_path = plySequenceCachePath(*header);
            std::error_code ec;
            std::filesystem::create_directories(cache_path.parent_path(), ec);
            if (ec) {
                error = ec.message();
                return false;
            }

            const auto tmp_path = cache_path.parent_path() /
                                  (cache_path.filename().string() + ".tmp");
            std::ofstream file;
            if (!lfs::core::open_file_for_write(tmp_path,
                                                std::ios::binary | std::ios::trunc,
                                                file)) {
                error = "open failed";
                return false;
            }

            try {
                file.write(reinterpret_cast<const char*>(&*header), sizeof(*header));
                model.serialize(file);
            } catch (const std::exception& e) {
                error = e.what();
                std::filesystem::remove(tmp_path, ec);
                return false;
            }
            file.close();
            if (!file) {
                error = "write failed";
                std::filesystem::remove(tmp_path, ec);
                return false;
            }

            std::filesystem::remove(cache_path, ec);
            ec.clear();
            std::filesystem::rename(tmp_path, cache_path, ec);
            if (ec) {
                error = ec.message();
                std::filesystem::remove(tmp_path, ec);
                return false;
            }
            return true;
        }

    } // namespace

    SequencerUIManager::SequencerUIManager(VisualizerImpl* viewer, panels::SequencerUIState& ui_state,
                                           gui::RmlUIManager* rml_manager)
        : viewer_(viewer),
          ui_state_(ui_state),
          panel_(std::make_unique<RmlSequencerPanel>(controller_, ui_state_, rml_manager)),
          overlay_(std::make_unique<RmlSequencerOverlay>(controller_, rml_manager)),
          scene_sync_(std::make_unique<KeyframeSceneSync>(controller_, viewer)) {}

    SequencerUIManager::~SequencerUIManager() {
        stopPlySequenceStreaming();
    }

    void SequencerUIManager::destroyGraphicsResources() {
        stopPlySequenceStreaming();
        last_ply_sequence_frame_ = std::nullopt;
        loaded_ply_sequence_frames_.clear();
        if (panel_)
            panel_->destroyGraphicsResources();
        if (overlay_)
            overlay_->destroyGraphicsResources();
        pip_texture_.reset();
        pip_initialized_ = false;
        line_renderer_.destroyResources();
        film_strip_.destroyGraphicsResources();
    }

    void SequencerUIManager::reloadRmlResources() {
        if (panel_)
            panel_->reloadResources();
        if (overlay_)
            overlay_->reloadResources();
        pip_needs_update_ = true;
    }

    void SequencerUIManager::setSequencerEnabled(const bool enabled) {
        if (enabled)
            return;

        if (panel_)
            panel_->clearPendingComposite();

        if (ui_state_.show_pip_preview)
            ui_state_.show_pip_preview = false;

        viewport_edit_mode_ = SequencerViewportEditMode::None;
        keyframe_gizmo_active_ = false;
        pip_last_keyframe_ = std::nullopt;
        pip_needs_update_ = true;
        last_panel_frame_time_ = std::chrono::steady_clock::now();
        endViewportKeyframeEdit();
    }

    void SequencerUIManager::beginViewportKeyframeEdit(const size_t keyframe_index) {
        const auto* const keyframe = controller_.timeline().getKeyframe(keyframe_index);
        if (!keyframe || keyframe->is_loop_point)
            return;

        controller_.selectKeyframe(keyframe_index);
        if (auto* sm = viewer_->getSceneManager())
            sm->clearSelection();
        viewport_keyframe_edit_snapshot_ = *keyframe;
        viewport_edit_mode_ = SequencerViewportEditMode::None;
        keyframe_gizmo_active_ = false;
        edit_entered_mouse_down_ = true;
    }

    void SequencerUIManager::endViewportKeyframeEdit() {
        viewport_keyframe_edit_snapshot_ = std::nullopt;
        if (overlay_)
            overlay_->hideEditOverlay();
    }

    sequencer::CameraState SequencerUIManager::currentViewportCameraState() const {
        const auto& cam = viewer_->getViewport().camera;
        auto* const rm = viewer_->getRenderingManager();

        return {
            .position = cam.t,
            .rotation = glm::quat_cast(cam.R),
            .focal_length_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM};
    }

    void SequencerUIManager::restoreViewportCameraState(const sequencer::CameraState& state) const {
        auto& vp = viewer_->getViewport();
        vp.setViewMatrix(glm::mat3_cast(state.rotation), state.position);

        if (auto* const rm = viewer_->getRenderingManager()) {
            rm->setFocalLength(state.focal_length_mm);
            rm->markCameraPoseChanged();
        }
    }

    void SequencerUIManager::setupEvents() {
        using namespace lfs::core::events;

        ui::RenderSettingsChanged::when([this](const auto& event) {
            if (event.equirectangular)
                ui_state_.equirectangular = *event.equirectangular;
        });

        cmd::SequencerAddKeyframe::when([this](const auto&) {
            const auto& cam = viewer_->getViewport().camera;
            const float time = controller_.playhead();
            const glm::vec3 position = cam.t;
            const glm::quat rotation = glm::quat_cast(cam.R);

            auto* const rm = viewer_->getRenderingManager();
            const float focal_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;

            // Match Blender / After Effects / Maya: clicking + at a time that already has a
            // keyframe overwrites that keyframe with the current pose instead of stacking.
            constexpr float REPLACE_EPSILON_S = 0.01f;
            const auto& keyframes = controller_.timeline().keyframes();
            const auto existing = std::find_if(keyframes.begin(), keyframes.end(),
                                               [time](const lfs::sequencer::Keyframe& kf) {
                                                   return !kf.is_loop_point &&
                                                          std::abs(kf.time - time) < REPLACE_EPSILON_S;
                                               });
            if (existing != keyframes.end()) {
                controller_.updateKeyframeById(existing->id, position, rotation, focal_mm);
            } else {
                lfs::sequencer::Keyframe kf;
                kf.time = time;
                kf.position = position;
                kf.rotation = rotation;
                kf.focal_length_mm = focal_mm;
                controller_.addKeyframeAtTime(kf, time);
            }
            state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
        });

        cmd::SequencerUpdateKeyframe::when([this](const auto&) {
            if (!controller_.hasSelection())
                return;
            const auto& cam = viewer_->getViewport().camera;
            auto* const rm = viewer_->getRenderingManager();
            const float focal_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
            controller_.updateSelectedKeyframe(
                cam.t,
                glm::quat_cast(cam.R),
                focal_mm);
            if (viewport_keyframe_edit_snapshot_.has_value() &&
                controller_.selectedKeyframeId().has_value() &&
                *controller_.selectedKeyframeId() ==
                    viewport_keyframe_edit_snapshot_->id) {
                viewport_keyframe_edit_snapshot_->position = cam.t;
                viewport_keyframe_edit_snapshot_->rotation =
                    glm::quat_cast(cam.R);
                viewport_keyframe_edit_snapshot_->focal_length_mm = focal_mm;
            } else {
                endViewportKeyframeEdit();
            }
            state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
        });

        cmd::SequencerGoToKeyframe::when([this](const auto&) {
            endViewportKeyframeEdit();
        });

        cmd::SequencerPlayPause::when([this](const auto&) {
            controller_.togglePlayPause();
        });

        cmd::SequencerLoadPlySequence::when([this](const auto& event) {
            if (event.fps > 0.0f)
                ui_state_.sequence_fps = std::clamp(event.fps, MIN_SEQUENCE_FPS, MAX_SEQUENCE_FPS);
            loadPlySequenceFromDirectory(std::filesystem::path(event.directory));
        });

        state::KeyframeListChanged::when([this](const auto&) {
            film_strip_.invalidateAll();
        });

        // File → New Project (and any dataset swap) emits SceneCleared. Drop the camera path
        // and film-strip thumbs by wiping the timeline; the keyframes belong to the old scene.
        state::SceneCleared::when([this](const auto&) {
            if (controller_.timeline().realKeyframeCount() == 0 &&
                !controller_.timeline().hasAnimationClip() &&
                !controller_.hasPlySequence())
                return;
            stopPlySequenceStreaming();
            controller_.clear();
            last_ply_sequence_frame_ = std::nullopt;
            loaded_ply_sequence_frames_.clear();
            film_strip_.invalidateAll();
            state::KeyframeListChanged{.count = 0}.emit();
        });

        ui::NodeSelected::when([this](const auto& e) {
            if (e.type != "KEYFRAME") {
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
            }
        });

        scene_sync_->setupEvents();
    }

    void SequencerUIManager::setFloating(const bool floating) {
        if (panel_)
            panel_->setFloating(floating);
    }

    float SequencerUIManager::preferredFloatingHeight() const {
        const float dp = std::max(1.0f, getThemeDpiScale());
        const float strip_h = ui_state_.show_film_strip ? FilmStripRenderer::STRIP_HEIGHT : 0.0f;
        return (panel_config::HEIGHT + panel_config::EASING_STRIPE_HEIGHT) * dp + strip_h;
    }

    void SequencerUIManager::render(const UIContext& ctx, const ViewportLayout& viewport,
                                    const float panel_x, const float panel_y,
                                    const float panel_width, const float panel_height,
                                    const PanelInputState& panel_input) {
        const auto* const gui = viewer_->getGuiManager();
        const bool sequencer_enabled = gui && gui->panelLayout().isShowSequencer();
        if (!sequencer_enabled) {
            setSequencerEnabled(false);
            return;
        }

        if (ui_state_.equirectangular != last_equirectangular_) {
            last_equirectangular_ = ui_state_.equirectangular;
            pip_needs_update_ = true;
            film_strip_.invalidateAll();
        }

        const auto& sdl_buf = viewer_->getWindowManager()->frameInput();
        lfs::vis::PanelInputState overlay_input;
        overlay_input.mouse_x = sdl_buf.mouse_x;
        overlay_input.mouse_y = sdl_buf.mouse_y;
        overlay_input.mouse_down[0] = sdl_buf.mouse_down[0];
        overlay_input.mouse_down[1] = sdl_buf.mouse_down[1];
        overlay_input.mouse_clicked[0] = sdl_buf.mouse_clicked[0];
        overlay_input.mouse_clicked[1] = sdl_buf.mouse_clicked[1];
        overlay_input.mouse_released[0] = sdl_buf.mouse_released[0];
        overlay_input.mouse_released[1] = sdl_buf.mouse_released[1];
        overlay_input.key_ctrl = (sdl_buf.key_mods & SDL_KMOD_CTRL) != 0;
        overlay_input.key_shift = (sdl_buf.key_mods & SDL_KMOD_SHIFT) != 0;
        overlay_input.key_alt = (sdl_buf.key_mods & SDL_KMOD_ALT) != 0;
        overlay_input.key_super = (sdl_buf.key_mods & SDL_KMOD_GUI) != 0;
        for (auto sc : sdl_buf.keys_pressed)
            overlay_input.keys_pressed.push_back(static_cast<int>(sc));
        for (auto sc : sdl_buf.keys_released)
            overlay_input.keys_released.push_back(static_cast<int>(sc));
        overlay_input.text_codepoints = sdl_buf.text_codepoints;
        overlay_input.text_inputs = sdl_buf.text_inputs;
        overlay_input.text_editing = sdl_buf.text_editing;
        overlay_input.text_editing_start = sdl_buf.text_editing_start;
        overlay_input.text_editing_length = sdl_buf.text_editing_length;
        overlay_input.has_text_editing = sdl_buf.has_text_editing;

        renderKeyframeEditOverlay(viewport);
        overlay_->processInput(overlay_input);
        handleOverlayActions();

        const bool overlay_active = overlay_->wantsInput() ||
                                    overlay_->isMouseOverEditOverlay(sdl_buf.mouse_x, sdl_buf.mouse_y);
        if (edit_entered_mouse_down_ && !sdl_buf.mouse_down[0])
            edit_entered_mouse_down_ = false;
        if (overlay_active || edit_entered_mouse_down_)
            guiFocusState().want_capture_mouse = true;

        const bool actively_following =
            ui_state_.follow_playback &&
            controller_.timeline().realKeyframeCount() > 0;

        if (ui_state_.show_camera_path && !actively_following) {
            renderCameraPath(viewport);
            renderKeyframeGizmo(ctx, viewport);
        }
        renderKeyframePreview(ctx);
        renderSequencerPanel(ctx, viewport, panel_x, panel_y, panel_width, panel_height, panel_input);
        syncPipPreviewWindow(viewport);

        overlay_->render(sdl_buf.window_w, sdl_buf.window_h);
    }

    void SequencerUIManager::compositeOverlays(const int screen_w, const int screen_h) {
        if (panel_)
            panel_->compositeToScreen(screen_w, screen_h);
        if (overlay_)
            overlay_->compositeToScreen(screen_w, screen_h);
    }

    bool SequencerUIManager::blocksPointer(const double x, const double y) const {
        return overlay_ &&
               (overlay_->wantsInput() || overlay_->isMouseOverEditOverlay(static_cast<float>(x),
                                                                           static_cast<float>(y)));
    }

    bool SequencerUIManager::blocksKeyboard() const {
        return overlay_ && (overlay_->isContextMenuOpen() || overlay_->isPopupOpen());
    }

    bool SequencerUIManager::needsAnimationFrame() const {
        return controller_.isPlaying() ||
               controller_.state() == PlaybackState::SCRUBBING ||
               plySequenceStreamHasWork() ||
               keyframe_gizmo_active_ ||
               viewport_keyframe_edit_snapshot_.has_value() ||
               (ui_state_.show_pip_preview && pip_needs_update_) ||
               (overlay_ && (overlay_->wantsInput() ||
                             overlay_->isContextMenuOpen() ||
                             overlay_->isPopupOpen()));
    }

    bool SequencerUIManager::plySequenceStreamHasWork() const {
        std::lock_guard lock(ply_stream_mutex_);
        return ply_stream_inflight_ ||
               !ply_stream_requests_.empty() ||
               !ply_stream_completed_.empty();
    }

    size_t SequencerUIManager::plySequenceFrameDistance(const size_t lhs,
                                                        const size_t rhs,
                                                        const size_t frame_count) const {
        if (frame_count == 0)
            return 0;

        const size_t direct = lhs > rhs ? lhs - rhs : rhs - lhs;
        if (controller_.loopMode() != LoopMode::LOOP)
            return direct;
        return std::min(direct, frame_count - direct);
    }

    bool SequencerUIManager::isPlySequenceFrameInWindow(const size_t frame_index,
                                                        const size_t center_frame,
                                                        const size_t frame_count) const {
        return isPlySequenceFrameInWindow(frame_index,
                                          center_frame,
                                          frame_count,
                                          controller_.loopMode() == LoopMode::LOOP);
    }

    bool SequencerUIManager::isPlySequenceFrameInWindow(const size_t frame_index,
                                                        const size_t center_frame,
                                                        const size_t frame_count,
                                                        const bool loop) const {
        if (frame_count == 0 || frame_index >= frame_count || center_frame >= frame_count)
            return false;
        if (frame_count <= MAX_STREAM_RESIDENT_FRAMES)
            return true;
        if (frame_index == center_frame)
            return true;

        if (loop) {
            const size_t forward = (frame_index + frame_count - center_frame) % frame_count;
            const size_t backward = (center_frame + frame_count - frame_index) % frame_count;
            return forward <= STREAM_PREFETCH_AHEAD || backward <= STREAM_PREFETCH_BEHIND;
        }

        if (frame_index > center_frame)
            return frame_index - center_frame <= STREAM_PREFETCH_AHEAD;
        return center_frame - frame_index <= STREAM_PREFETCH_BEHIND;
    }

    std::optional<size_t> SequencerUIManager::selectPlySequenceDisplayFrame(const size_t requested_frame) const {
        const auto* const sequence = controller_.plySequence();
        if (!sequence || requested_frame >= sequence->frames.size())
            return std::nullopt;

        std::lock_guard lock(ply_stream_mutex_);
        if (requested_frame < ply_stream_states_.size() &&
            ply_stream_states_[requested_frame] == PlyStreamFrameState::Resident) {
            return requested_frame;
        }

        const size_t frame_count = sequence->frames.size();
        const bool playing = controller_.isPlaying();
        std::optional<size_t> best_frame;
        size_t best_score = 0;
        for (size_t frame = 0; frame < ply_stream_states_.size() && frame < frame_count; ++frame) {
            if (ply_stream_states_[frame] != PlyStreamFrameState::Resident)
                continue;

            size_t score = 0;
            if (!playing) {
                score = plySequenceFrameDistance(frame, requested_frame, frame_count);
            } else if (controller_.loopMode() == LoopMode::LOOP) {
                score = (frame + frame_count - requested_frame) % frame_count;
            } else if (frame >= requested_frame) {
                score = frame - requested_frame;
            } else {
                score = STREAM_PREFETCH_AHEAD + 1 + requested_frame - frame;
            }

            if (!best_frame.has_value() || score < best_score) {
                best_frame = frame;
                best_score = score;
            }
        }

        return best_frame;
    }

    void SequencerUIManager::requestPlySequenceFrame(const size_t frame_index, const bool priority) {
        std::lock_guard lock(ply_stream_mutex_);
        if (frame_index >= ply_stream_states_.size())
            return;

        auto& state = ply_stream_states_[frame_index];
        if (state == PlyStreamFrameState::Resident || state == PlyStreamFrameState::Loading)
            return;

        if (state == PlyStreamFrameState::Queued) {
            if (priority) {
                std::erase(ply_stream_requests_, frame_index);
                ply_stream_requests_.push_front(frame_index);
            }
            return;
        }

        state = PlyStreamFrameState::Queued;
        if (priority)
            ply_stream_requests_.push_front(frame_index);
        else
            ply_stream_requests_.push_back(frame_index);
        ply_stream_cv_.notify_one();
    }

    void SequencerUIManager::prunePlySequenceRequests(const size_t frame_index) {
        const auto* const sequence = controller_.plySequence();
        if (!sequence || sequence->frames.size() <= MAX_STREAM_RESIDENT_FRAMES)
            return;

        const size_t count = sequence->frames.size();
        std::lock_guard lock(ply_stream_mutex_);
        for (auto it = ply_stream_requests_.begin(); it != ply_stream_requests_.end();) {
            const size_t queued_frame = *it;
            if (isPlySequenceFrameInWindow(queued_frame, frame_index, count)) {
                ++it;
                continue;
            }
            if (queued_frame < ply_stream_states_.size() &&
                ply_stream_states_[queued_frame] == PlyStreamFrameState::Queued) {
                ply_stream_states_[queued_frame] = PlyStreamFrameState::Empty;
            }
            it = ply_stream_requests_.erase(it);
            ++ply_stream_stale_request_drop_count_;
        }
    }

    void SequencerUIManager::requestPlySequenceWindow(const size_t frame_index) {
        const auto* const sequence = controller_.plySequence();
        if (!sequence || sequence->frames.empty())
            return;

        const size_t count = sequence->frames.size();
        ply_stream_target_frame_.store(frame_index, std::memory_order_release);
        ply_stream_target_loop_.store(controller_.loopMode() == LoopMode::LOOP, std::memory_order_release);
        prunePlySequenceRequests(frame_index);
        requestPlySequenceFrame(frame_index, true);

        if (count <= MAX_STREAM_RESIDENT_FRAMES) {
            for (size_t frame = 0; frame < count; ++frame)
                requestPlySequenceFrame(frame, frame == frame_index);
            return;
        }

        for (size_t offset = 1; offset <= STREAM_PREFETCH_AHEAD && offset < count; ++offset) {
            const size_t next = controller_.loopMode() == LoopMode::LOOP
                                    ? (frame_index + offset) % count
                                    : frame_index + offset;
            if (next >= count)
                break;
            requestPlySequenceFrame(next, false);
        }

        for (size_t offset = 1; offset <= STREAM_PREFETCH_BEHIND && offset < count; ++offset) {
            if (controller_.loopMode() == LoopMode::LOOP) {
                requestPlySequenceFrame((frame_index + count - offset) % count, false);
            } else if (offset <= frame_index) {
                requestPlySequenceFrame(frame_index - offset, false);
            } else {
                break;
            }
        }
    }

    void SequencerUIManager::plySequenceStreamWorker(const uint64_t generation) {
        auto loader = lfs::io::Loader::create();

        while (!ply_stream_stop_.load(std::memory_order_acquire) &&
               ply_stream_generation_.load(std::memory_order_acquire) == generation) {
            size_t frame_index = 0;
            std::filesystem::path path;
            lfs::io::SplatTensorAllocator allocator;

            {
                std::unique_lock lock(ply_stream_mutex_);
                ply_stream_cv_.wait(lock, [this, generation] {
                    return ply_stream_stop_.load(std::memory_order_acquire) ||
                           ply_stream_generation_.load(std::memory_order_acquire) != generation ||
                           !ply_stream_requests_.empty();
                });

                if (ply_stream_stop_.load(std::memory_order_acquire) ||
                    ply_stream_generation_.load(std::memory_order_acquire) != generation)
                    return;

                frame_index = ply_stream_requests_.front();
                ply_stream_requests_.pop_front();
                if (frame_index >= ply_stream_states_.size() ||
                    frame_index >= ply_stream_paths_.size() ||
                    ply_stream_states_[frame_index] != PlyStreamFrameState::Queued) {
                    continue;
                }

                ply_stream_states_[frame_index] = PlyStreamFrameState::Loading;
                ply_stream_inflight_ = true;
                ply_stream_inflight_frame_ = frame_index;
                path = ply_stream_paths_[frame_index];
                allocator = ply_stream_allocator_;
            }

            const auto start = std::chrono::steady_clock::now();
            PlyStreamResult completed{};
            completed.generation = generation;
            completed.frame_index = frame_index;

            try {
                std::string cache_error;
                completed.model = loadPlySequenceCache(path, allocator, cache_error);
                completed.cache_hit = completed.model != nullptr;
                completed.cache_miss = !completed.cache_hit;

                if (!completed.model) {
                    const lfs::io::LoadOptions load_options{
                        .resize_factor = -1,
                        .max_width = 0,
                        .images_folder = "images",
                        .validate_only = false,
                        .cancel_requested = [this, generation, frame_index, frame_count = ply_stream_paths_.size()] {
                            if (ply_stream_stop_.load(std::memory_order_acquire) ||
                                ply_stream_generation_.load(std::memory_order_acquire) != generation) {
                                return true;
                            }
                            if (frame_count <= MAX_STREAM_RESIDENT_FRAMES)
                                return false;

                            const size_t target = ply_stream_target_frame_.load(std::memory_order_acquire);
                            if (target >= frame_count)
                                return false;
                            const bool loop = ply_stream_target_loop_.load(std::memory_order_acquire);
                            return !isPlySequenceFrameInWindow(frame_index, target, frame_count, loop);
                        },
                        .splat_tensor_allocator = allocator};

                    auto load_result = loader->load(path, load_options);
                    if (!load_result)
                        throw std::runtime_error(load_result.error().format());

                    auto* splat_data = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&load_result->data);
                    if (!splat_data || !*splat_data)
                        throw std::runtime_error("sequence frame is not a gaussian splat PLY");

                    completed.model = std::make_unique<lfs::core::SplatData>(std::move(**splat_data));
                    std::string write_error;
                    if (writePlySequenceCache(path, *completed.model, write_error)) {
                        completed.cache_written = true;
                    } else {
                        completed.cache_write_failed = true;
                        LOG_DEBUG("Failed to write PLY sequence cache for '{}': {}",
                                  lfs::core::path_to_utf8(path),
                                  write_error);
                    }
                }
            } catch (const lfs::io::LoadCancelledError& e) {
                completed.cancelled = true;
                completed.error = e.what();
            } catch (const std::exception& e) {
                completed.error = e.what();
            }

            completed.load_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

            {
                std::lock_guard lock(ply_stream_mutex_);
                if (ply_stream_generation_.load(std::memory_order_acquire) == generation) {
                    ply_stream_completed_.push_back(std::move(completed));
                }
                ply_stream_inflight_ = false;
                ply_stream_cv_.notify_one();
            }
        }
    }

    void SequencerUIManager::startPlySequenceStreaming(std::vector<std::filesystem::path> paths,
                                                       lfs::io::SplatTensorAllocator allocator) {
        stopPlySequenceStreaming();

        const uint64_t generation = ply_stream_generation_.fetch_add(1, std::memory_order_acq_rel) + 1;
        {
            std::lock_guard lock(ply_stream_mutex_);
            ply_stream_paths_ = std::move(paths);
            ply_stream_allocator_ = std::move(allocator);
            ply_stream_states_.assign(ply_stream_paths_.size(), PlyStreamFrameState::Empty);
            ply_stream_requests_.clear();
            ply_stream_completed_.clear();
            ply_stream_inflight_ = false;
            ply_stream_inflight_frame_ = 0;
            ply_stream_last_load_ms_ = 0.0;
            ply_stream_failed_count_ = 0;
            ply_stream_miss_count_ = 0;
            ply_stream_fallback_count_ = 0;
            ply_stream_eviction_count_ = 0;
            ply_stream_stale_request_drop_count_ = 0;
            ply_stream_cache_hit_count_ = 0;
            ply_stream_cache_miss_count_ = 0;
            ply_stream_cache_write_count_ = 0;
            ply_stream_cache_write_fail_count_ = 0;
        }
        ply_stream_target_frame_.store(0, std::memory_order_release);
        ply_stream_target_loop_.store(controller_.loopMode() == LoopMode::LOOP, std::memory_order_release);

        ply_stream_stop_.store(false, std::memory_order_release);
        ply_stream_thread_ = std::thread([this, generation] {
            plySequenceStreamWorker(generation);
        });

        requestPlySequenceWindow(0);
    }

    void SequencerUIManager::stopPlySequenceStreaming() {
        ply_stream_stop_.store(true, std::memory_order_release);
        ply_stream_generation_.fetch_add(1, std::memory_order_acq_rel);
        ply_stream_cv_.notify_all();
        if (ply_stream_thread_.joinable())
            ply_stream_thread_.join();

        std::lock_guard lock(ply_stream_mutex_);
        ply_stream_paths_.clear();
        ply_stream_allocator_ = {};
        ply_stream_states_.clear();
        ply_stream_requests_.clear();
        ply_stream_completed_.clear();
        ply_stream_inflight_ = false;
        ply_stream_inflight_frame_ = 0;
        ply_stream_last_load_ms_ = 0.0;
        ply_stream_failed_count_ = 0;
        ply_stream_miss_count_ = 0;
        ply_stream_fallback_count_ = 0;
        ply_stream_eviction_count_ = 0;
        ply_stream_stale_request_drop_count_ = 0;
        ply_stream_cache_hit_count_ = 0;
        ply_stream_cache_miss_count_ = 0;
        ply_stream_cache_write_count_ = 0;
        ply_stream_cache_write_fail_count_ = 0;
        ply_stream_target_frame_.store(0, std::memory_order_release);
        ply_stream_target_loop_.store(false, std::memory_order_release);
    }

    void SequencerUIManager::drainPlySequenceStream() {
        auto* const scene_manager = viewer_->getSceneManager();
        const auto* const sequence = controller_.plySequence();
        if (!scene_manager || !sequence)
            return;

        std::deque<PlyStreamResult> completed;
        {
            std::lock_guard lock(ply_stream_mutex_);
            completed.swap(ply_stream_completed_);
        }

        if (completed.empty())
            return;

        auto& scene = scene_manager->getScene();
        const uint64_t active_generation = ply_stream_generation_.load(std::memory_order_acquire);
        bool current_frame_loaded = false;
        const auto current_frame = controller_.currentPlySequenceFrameIndex();

        while (!completed.empty()) {
            auto result = std::move(completed.front());
            completed.pop_front();
            if (result.generation != active_generation ||
                result.frame_index >= sequence->frames.size()) {
                continue;
            }

            const std::string& node_name = sequence->frames[result.frame_index].node_name;
            if (!result.model) {
                if (result.cancelled) {
                    std::lock_guard lock(ply_stream_mutex_);
                    if (result.frame_index < ply_stream_states_.size())
                        ply_stream_states_[result.frame_index] = PlyStreamFrameState::Empty;
                    ply_stream_last_load_ms_ = result.load_ms;
                    continue;
                }

                LOG_ERROR("Failed to stream PLY sequence frame {}: {}",
                          result.frame_index,
                          result.error.empty() ? "unknown error" : result.error);
                std::lock_guard lock(ply_stream_mutex_);
                if (result.frame_index < ply_stream_states_.size())
                    ply_stream_states_[result.frame_index] = PlyStreamFrameState::Failed;
                ++ply_stream_failed_count_;
                ply_stream_last_load_ms_ = result.load_ms;
                continue;
            }

            const size_t gaussian_count = result.model->size();
            auto old_model = scene.swapNodeModel(node_name, std::move(result.model));
            old_model.reset();
            scene.setNodeVisibility(node_name, false);
            scene_manager->setPlyPath(node_name, sequence->frames[result.frame_index].path);

            {
                std::lock_guard lock(ply_stream_mutex_);
                if (result.frame_index < ply_stream_states_.size())
                    ply_stream_states_[result.frame_index] = PlyStreamFrameState::Resident;
                std::erase(loaded_ply_sequence_frames_, result.frame_index);
                loaded_ply_sequence_frames_.push_back(result.frame_index);
                ply_stream_last_load_ms_ = result.load_ms;
                if (result.cache_hit)
                    ++ply_stream_cache_hit_count_;
                if (result.cache_miss)
                    ++ply_stream_cache_miss_count_;
                if (result.cache_written)
                    ++ply_stream_cache_write_count_;
                if (result.cache_write_failed)
                    ++ply_stream_cache_write_fail_count_;
            }

            LOG_INFO("Streamed PLY sequence frame {} '{}' ({} gaussians, {:.1f}ms, cache={})",
                     result.frame_index,
                     node_name,
                     gaussian_count,
                     result.load_ms,
                     result.cache_hit ? "hit" : "miss");
            current_frame_loaded = current_frame.has_value() && *current_frame == result.frame_index;
        }

        if (current_frame_loaded)
            last_ply_sequence_frame_ = std::nullopt;
        if (auto* const rm = viewer_->getRenderingManager())
            rm->markDirty(DirtyFlag::SPLATS);
        if (current_frame.has_value())
            evictPlySequenceFrames(*current_frame);
    }

    void SequencerUIManager::evictPlySequenceFrames(const size_t keep_frame_index) {
        const auto* const sequence = controller_.plySequence();
        auto* const scene_manager = viewer_->getSceneManager();
        if (!sequence || !scene_manager)
            return;

        auto& scene = scene_manager->getScene();
        const size_t frame_count = sequence->frames.size();
        const size_t budget = std::min(MAX_STREAM_RESIDENT_FRAMES, frame_count);
        while (loaded_ply_sequence_frames_.size() > budget) {
            auto victim_it = loaded_ply_sequence_frames_.end();
            bool victim_outside_window = false;
            size_t victim_distance = 0;

            for (auto it = loaded_ply_sequence_frames_.begin(); it != loaded_ply_sequence_frames_.end(); ++it) {
                const size_t candidate = *it;
                if (candidate == keep_frame_index ||
                    (last_ply_sequence_frame_.has_value() && candidate == *last_ply_sequence_frame_) ||
                    candidate >= frame_count) {
                    continue;
                }

                const bool outside_window = !isPlySequenceFrameInWindow(candidate, keep_frame_index, frame_count);
                const size_t distance = plySequenceFrameDistance(candidate, keep_frame_index, frame_count);
                if (victim_it == loaded_ply_sequence_frames_.end() ||
                    (outside_window && !victim_outside_window) ||
                    (outside_window == victim_outside_window && distance > victim_distance)) {
                    victim_it = it;
                    victim_outside_window = outside_window;
                    victim_distance = distance;
                }
            }

            if (victim_it == loaded_ply_sequence_frames_.end())
                return;

            const size_t victim = *victim_it;
            loaded_ply_sequence_frames_.erase(victim_it);
            if (victim >= frame_count)
                continue;

            const std::string& victim_name = sequence->frames[victim].node_name;
            auto old_model = scene.swapNodeModel(victim_name, nullptr);
            old_model.reset();
            scene.setNodeVisibility(victim_name, false);
            std::lock_guard lock(ply_stream_mutex_);
            if (victim < ply_stream_states_.size())
                ply_stream_states_[victim] = PlyStreamFrameState::Empty;
            ++ply_stream_eviction_count_;
        }
    }

    float SequencerUIManager::advancePanelClock() {
        const auto now = std::chrono::steady_clock::now();
        float delta_time = std::chrono::duration<float>(now - last_panel_frame_time_).count();
        last_panel_frame_time_ = now;
        if (!std::isfinite(delta_time) || delta_time < 0.0f)
            delta_time = 0.0f;
        delta_time = std::min(delta_time, 0.1f);
        panel_elapsed_time_ += delta_time;
        last_panel_delta_time_ = delta_time;
        return delta_time;
    }

    float SequencerUIManager::advancePlaybackClock() {
        const auto now = std::chrono::steady_clock::now();
        if (!last_playback_tick_time_.has_value()) {
            last_playback_tick_time_ = now;
            return 0.0f;
        }

        float delta_time = std::chrono::duration<float>(now - *last_playback_tick_time_).count();
        last_playback_tick_time_ = now;
        if (!std::isfinite(delta_time) || delta_time < 0.0f)
            delta_time = 0.0f;
        if (!controller_.hasPlySequence())
            delta_time = std::min(delta_time, 0.1f);
        return delta_time;
    }

    float SequencerUIManager::playbackDelta(const float delta_time) const {
        return delta_time;
    }

    void SequencerUIManager::applyPlaybackCameraFollow() {
        auto* const rm = viewer_->getRenderingManager();
        if (!rm)
            return;

        const bool is_playing = controller_.isPlaying() && controller_.timeline().realKeyframeCount() > 0;
        rm->setOverlayAnimationActive(is_playing);
        if (ui_state_.follow_playback && controller_.timeline().realKeyframeCount() > 0) {
            const auto state = controller_.currentCameraState();
            auto& vp = viewer_->getViewport();
            vp.setViewMatrix(glm::mat3_cast(state.rotation), state.position);
            rm->setFocalLength(state.focal_length_mm);
            rm->markCameraPoseChanged();
        }
    }

    void SequencerUIManager::advancePlayback(const float delta_time) {
        controller_.update(playbackDelta(delta_time));
        applyPlaybackCameraFollow();
        applyPlySequenceFrame();
    }

    void SequencerUIManager::tickPlaybackBeforeSceneRender() {
        if (!controller_.isPlaying()) {
            last_playback_tick_time_ = std::nullopt;
            drainPlySequenceStream();
            applyPlySequenceFrame();
            return;
        }

        advancePanelClock();
        advancePlayback(advancePlaybackClock());
        playback_ticked_before_scene_ = true;
    }

    std::string SequencerUIManager::plyPlayerStatusJson() const {
        const auto* const sequence = controller_.plySequence();
        if (!sequence)
            return {};
        const auto current_frame = controller_.currentPlySequenceFrameIndex();

        size_t resident = 0;
        size_t queued = 0;
        size_t failed = 0;
        bool inflight = false;
        double last_load_ms = 0.0;
        size_t misses = 0;
        size_t fallbacks = 0;
        size_t evictions = 0;
        size_t stale_drops = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
        size_t cache_writes = 0;
        size_t cache_write_failures = 0;
        {
            std::lock_guard lock(ply_stream_mutex_);
            for (const auto state : ply_stream_states_) {
                switch (state) {
                case PlyStreamFrameState::Resident: ++resident; break;
                case PlyStreamFrameState::Queued: ++queued; break;
                case PlyStreamFrameState::Loading: break;
                case PlyStreamFrameState::Failed: ++failed; break;
                case PlyStreamFrameState::Empty: break;
                }
            }
            queued = std::max(queued, ply_stream_requests_.size());
            inflight = ply_stream_inflight_;
            failed = std::max(failed, ply_stream_failed_count_);
            last_load_ms = ply_stream_last_load_ms_;
            misses = ply_stream_miss_count_;
            fallbacks = ply_stream_fallback_count_;
            evictions = ply_stream_eviction_count_;
            stale_drops = ply_stream_stale_request_drop_count_;
            cache_hits = ply_stream_cache_hit_count_;
            cache_misses = ply_stream_cache_miss_count_;
            cache_writes = ply_stream_cache_write_count_;
            cache_write_failures = ply_stream_cache_write_fail_count_;
        }

        return std::format(
            "{{\"frame_count\":{},\"displayed_frame\":{},\"requested_frame\":{},\"on_target\":{},"
            "\"resident\":{},\"slots\":{},\"max_slots\":{},\"decode_queue\":{},"
            "\"inflight\":{},\"failed\":{},\"last_swap_ms\":{:.3f},"
            "\"last_load_ms\":{:.3f},\"misses\":{},\"fallbacks\":{},"
            "\"evictions\":{},\"stale_queue_drops\":{},\"cache_hits\":{},"
            "\"cache_misses\":{},\"cache_writes\":{},\"cache_write_failures\":{},"
            "\"streaming\":true}}",
            sequence->frames.size(),
            last_ply_sequence_frame_.has_value() ? static_cast<long long>(*last_ply_sequence_frame_) : -1ll,
            current_frame.has_value() ? static_cast<long long>(*current_frame) : -1ll,
            last_ply_sequence_frame_.has_value() && current_frame.has_value() &&
                    *last_ply_sequence_frame_ == *current_frame
                ? "true"
                : "false",
            resident,
            resident,
            std::min(MAX_STREAM_RESIDENT_FRAMES, sequence->frames.size()),
            queued,
            inflight ? 1 : 0,
            failed,
            0.0,
            last_load_ms,
            misses,
            fallbacks,
            evictions,
            stale_drops,
            cache_hits,
            cache_misses,
            cache_writes,
            cache_write_failures);
    }

    void SequencerUIManager::renderSequencerPanel(const UIContext& /*ctx*/, const ViewportLayout& viewport,
                                                  const float panel_x, const float panel_y,
                                                  const float panel_width, const float panel_height,
                                                  const PanelInputState& panel_input) {
        (void)viewport;
        const bool already_ticked = playback_ticked_before_scene_;
        const float delta_time = already_ticked ? last_panel_delta_time_ : advancePanelClock();
        playback_ticked_before_scene_ = false;
        if (!already_ticked) {
            if (controller_.isPlaying()) {
                advancePlayback(advancePlaybackClock());
            } else {
                last_playback_tick_time_ = std::nullopt;
                advancePlayback(delta_time);
            }
        } else if (!controller_.isPlaying()) {
            last_playback_tick_time_ = std::nullopt;
        }

        panel_->setFilmStripAttached(ui_state_.show_film_strip);

        panel_input_ = toSequencerPanelInput(panel_input);
        panel_input_.time = panel_elapsed_time_;
        panel_input_.delta_time = delta_time;
        panel_input_.want_capture_mouse = guiFocusState().want_capture_mouse;

        panel_->render(panel_x, panel_y, panel_width, panel_height, panel_input_,
                       viewer_->getRenderingManager(), viewer_->getSceneManager(), film_strip_);

        if (panel_->isHovered())
            guiFocusState().want_capture_mouse = true;
        if (panel_->wantsKeyboard())
            guiFocusState().want_capture_keyboard = true;

        const auto timeline_menu = panel_->consumeContextMenu();
        if (timeline_menu.open) {
            overlay_->showContextMenu(panel_input_.mouse_x, panel_input_.mouse_y,
                                      timeline_menu.keyframe, timeline_menu.time, viewport_edit_mode_);
        }

        const auto time_req = panel_->consumeTimeEditRequest();
        if (time_req.active)
            overlay_->showTimeEdit(time_req.keyframe_index, time_req.current_time);

        const auto focal_req = panel_->consumeFocalEditRequest();
        if (focal_req.active)
            overlay_->showFocalEdit(focal_req.keyframe_index, focal_req.current_focal_mm);

        if (panel_->consumeSavePathRequest()) {
            const auto path = gui::SaveJsonFileDialog("camera_path");
            if (!path.empty()) {
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                if (controller_.saveToJson(path_utf8))
                    LOG_INFO("Camera path saved to {}", path_utf8);
                else
                    LOG_ERROR("Failed to save camera path to {}", path_utf8);
            }
        }

        if (panel_->consumeLoadPathRequest()) {
            const auto path = gui::OpenJsonFileDialog();
            if (!path.empty()) {
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                if (controller_.loadFromJson(path_utf8)) {
                    LOG_INFO("Camera path loaded from {}", path_utf8);
                    lfs::core::events::state::KeyframeListChanged{
                        .count = controller_.timeline().realKeyframeCount()}
                        .emit();
                    pip_needs_update_ = true;
                } else {
                    LOG_ERROR("Failed to load camera path from {}", path_utf8);
                }
            }
        }

        if (panel_->consumeLoadSequenceRequest()) {
            const auto path = gui::PickFolderDialog();
            if (!path.empty())
                loadPlySequenceFromDirectory(path);
        }

        if (panel_->consumeDockToggleRequest()) {
            const PanelSpace target = panel_->isFloating() ? PanelSpace::BottomDock : PanelSpace::Floating;
            if (!PanelRegistry::instance().set_panel_space("native.sequencer", target)) {
                LOG_ERROR("Failed to move sequencer panel to {}",
                          target == PanelSpace::Floating ? "floating" : "bottom dock");
            }
        }

        if (panel_->consumeClosePanelRequest()) {
            if (auto* const gui = viewer_->getGuiManager())
                gui->panelLayout().setShowSequencer(false);
            setSequencerEnabled(false);
        }

        if (panel_->consumeExportRequest() && controller_.timeline().realKeyframeCount() > 0) {
            const auto info = lfs::io::video::getPresetInfo(ui_state_.preset);
            const int w = ui_state_.preset == lfs::io::video::VideoPreset::CUSTOM
                              ? ui_state_.custom_width
                              : info.width;
            const int h = ui_state_.preset == lfs::io::video::VideoPreset::CUSTOM
                              ? ui_state_.custom_height
                              : info.height;
            lfs::core::events::cmd::SequencerExportVideo{
                .width = w,
                .height = h,
                .framerate = ui_state_.framerate,
                .crf = ui_state_.quality}
                .emit();
        }

        if (panel_->consumeClearRequest() &&
            (controller_.timeline().realKeyframeCount() > 0 || controller_.timeline().hasAnimationClip() ||
             controller_.hasPlySequence())) {
            stopPlySequenceStreaming();
            controller_.clear();
            last_ply_sequence_frame_ = std::nullopt;
            loaded_ply_sequence_frames_.clear();
            lfs::core::events::state::KeyframeListChanged{.count = 0}.emit();
            LOG_INFO("Sequencer cleared");
        }

        auto ctx_req = panel_->consumeTransportContextMenu();
        if (ctx_req.target != TransportContextMenuRequest::Target::NONE) {
            auto& cm = viewer_->getGuiManager()->globalContextMenu();
            std::vector<gui::ContextMenuItem> items;

            using Target = TransportContextMenuRequest::Target;
            switch (ctx_req.target) {
            case Target::SNAP: {
                items.push_back({LOC("context_menu.snap_interval"), "", false, true});
                constexpr std::array<float, 4> snap_values = {0.25f, 0.5f, 1.0f, 2.0f};
                constexpr std::array<const char*, 4> snap_labels = {"0.25s", "0.5s", "1s", "2s"};
                for (size_t i = 0; i < snap_values.size(); ++i) {
                    bool active = std::abs(ui_state_.snap_interval - snap_values[i]) < 0.01f;
                    items.push_back({snap_labels[i],
                                     std::format("snap_{}", snap_values[i]),
                                     false, false, false, active});
                }
                break;
            }
            case Target::PREVIEW: {
                items.push_back({LOC("context_menu.preview_scale"), "", false, true});
                constexpr std::array<float, 5> scale_values = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f};
                constexpr std::array<const char*, 5> scale_labels = {"0.5x", "0.75x", "1.0x", "1.5x", "2.0x"};
                for (size_t i = 0; i < scale_values.size(); ++i) {
                    bool active = std::abs(ui_state_.pip_preview_scale - scale_values[i]) < 0.01f;
                    items.push_back({scale_labels[i],
                                     std::format("scale_{}", scale_values[i]),
                                     false, false, false, active});
                }
                break;
            }
            case Target::FORMAT: {
                items.push_back({LOC("context_menu.video_format"), "", false, true});
                using lfs::io::video::VideoPreset;
                for (int p = 0; p <= static_cast<int>(VideoPreset::CUSTOM); ++p) {
                    const auto preset = static_cast<VideoPreset>(p);
                    const auto info = lfs::io::video::getPresetInfo(preset);
                    bool active = ui_state_.preset == preset;
                    items.push_back({info.name,
                                     std::format("preset_{}", p),
                                     false, false, false, active});
                }
                break;
            }
            case Target::CLEAR: {
                items.push_back({LOC("context_menu.clear_confirm"), "", false, true});
                items.push_back({LOC("context_menu.confirm"), "clear_confirm"});
                items.push_back({LOC("context_menu.cancel"), "clear_cancel"});
                break;
            }
            default:
                break;
            }

            if (!items.empty()) {
                const auto target = ctx_req.target;
                cm.request(std::move(items), ctx_req.screen_x, ctx_req.screen_y,
                           [this, target](std::string_view action) {
                               switch (target) {
                               case Target::SNAP:
                                   if (action.starts_with("snap_"))
                                       ui_state_.snap_interval = std::stof(std::string(action.substr(5)));
                                   break;
                               case Target::PREVIEW:
                                   if (action.starts_with("scale_"))
                                       ui_state_.pip_preview_scale = std::stof(std::string(action.substr(6)));
                                   break;
                               case Target::FORMAT:
                                   if (action.starts_with("preset_")) {
                                       using lfs::io::video::VideoPreset;
                                       const int idx = std::stoi(std::string(action.substr(7)));
                                       ui_state_.preset = static_cast<VideoPreset>(idx);
                                       const auto info = lfs::io::video::getPresetInfo(ui_state_.preset);
                                       ui_state_.custom_width = info.width;
                                       ui_state_.custom_height = info.height;
                                       ui_state_.framerate = info.framerate;
                                   }
                                   break;
                               case Target::CLEAR:
                                   if (action == "clear_confirm" &&
                                       (controller_.timeline().realKeyframeCount() > 0 || controller_.timeline().hasAnimationClip() ||
                                        controller_.hasPlySequence())) {
                                       stopPlySequenceStreaming();
                                       controller_.clear();
                                       last_ply_sequence_frame_ = std::nullopt;
                                       loaded_ply_sequence_frames_.clear();
                                       lfs::core::events::state::KeyframeListChanged{.count = 0}.emit();
                                       LOG_INFO("Sequencer cleared");
                                   }
                                   break;
                               case Target::NONE:
                                   break;
                               }
                           });
            }
        }

        applyPlySequenceFrame();
    }

    void SequencerUIManager::renderCameraPath(const ViewportLayout& viewport) {
        constexpr float PATH_THICKNESS = 2.0f;
        constexpr float PATH_SAMPLE_RADIUS = 2.5f;
        constexpr float FRUSTUM_THICKNESS = 1.5f;
        constexpr float NDC_CULL_MARGIN = 1.5f;
        constexpr size_t MAX_PATH_SAMPLE_MARKERS = 2000;
        constexpr float FRUSTUM_DEPTH = 0.25f;
        constexpr float SENSOR_ASPECT = rendering::SENSOR_WIDTH_35MM / rendering::SENSOR_HEIGHT_35MM;
        constexpr float HIT_RADIUS = 15.0f;

        const auto& timeline = controller_.timeline();
        if (timeline.empty())
            return;

        const auto& vp = viewer_->getViewport();
        auto* const rm = viewer_->getRenderingManager();
        if (!rm)
            return;
        const auto& settings = rm->getSettings();
        const glm::ivec2 vp_size(static_cast<int>(viewport.size.x), static_cast<int>(viewport.size.y));
        const auto* const rendering_manager = static_cast<const RenderingManager*>(rm);

        struct CameraPathPanel {
            SplitViewPanelId panel_id = SplitViewPanelId::Left;
            const Viewport* viewport = nullptr;
            glm::vec2 projection_pos{0.0f};
            glm::vec2 projection_size{0.0f};
            glm::ivec2 render_size{0};
            gui::ClipRect clip_rect{};

            [[nodiscard]] bool valid() const {
                return viewport != nullptr &&
                       projection_size.x > 0.0f &&
                       projection_size.y > 0.0f &&
                       render_size.x > 0 &&
                       render_size.y > 0 &&
                       clip_rect.width > 0 &&
                       clip_rect.height > 0;
            }

            [[nodiscard]] bool contains(const float x, const float y) const {
                return x >= static_cast<float>(clip_rect.x) &&
                       x <= static_cast<float>(clip_rect.x + clip_rect.width) &&
                       y >= static_cast<float>(clip_rect.y) &&
                       y <= static_cast<float>(clip_rect.y + clip_rect.height);
            }
        };

        std::vector<CameraPathPanel> panels;
        panels.reserve(2);

        const auto add_viewer_panel = [&](const std::optional<RenderingManager::ViewerPanelInfo>& info_opt) {
            if (!info_opt || !info_opt->valid())
                return;
            const auto& info = *info_opt;
            panels.push_back(CameraPathPanel{
                .panel_id = info.panel,
                .viewport = info.viewport,
                .projection_pos = {info.x, info.y},
                .projection_size = {info.width, info.height},
                .render_size = {info.render_width, info.render_height},
                .clip_rect = {
                    static_cast<int>(std::round(info.x)),
                    static_cast<int>(std::round(info.y)),
                    static_cast<int>(std::round(info.width)),
                    static_cast<int>(std::round(info.height)),
                },
            });
        };

        if (rm->isIndependentSplitViewActive()) {
            add_viewer_panel(rendering_manager->resolveViewerPanel(
                vp, viewport.pos, viewport.size, std::nullopt, SplitViewPanelId::Left));
            add_viewer_panel(rendering_manager->resolveViewerPanel(
                vp, viewport.pos, viewport.size, std::nullopt, SplitViewPanelId::Right));
        }

        if (panels.empty()) {
            const int clip_x = static_cast<int>(std::round(viewport.pos.x));
            const int clip_y = static_cast<int>(std::round(viewport.pos.y));
            const int clip_w = static_cast<int>(std::round(viewport.size.x));
            const int clip_h = static_cast<int>(std::round(viewport.size.y));
            std::vector<gui::ClipRect> clip_rects;
            clip_rects.reserve(2);

            if (const auto divider_x = rm->getSplitDividerScreenX(viewport.pos, viewport.size);
                divider_x.has_value()) {
                const int divider =
                    std::clamp(static_cast<int>(std::round(*divider_x)), clip_x, clip_x + clip_w);
                if (divider > clip_x)
                    clip_rects.push_back({clip_x, clip_y, divider - clip_x, clip_h});
                if (divider < clip_x + clip_w)
                    clip_rects.push_back({divider, clip_y, clip_x + clip_w - divider, clip_h});
            }

            if (clip_rects.empty())
                clip_rects.push_back({clip_x, clip_y, clip_w, clip_h});

            for (size_t i = 0; i < clip_rects.size(); ++i) {
                panels.push_back(CameraPathPanel{
                    .panel_id = (i == 0) ? SplitViewPanelId::Left : SplitViewPanelId::Right,
                    .viewport = &vp,
                    .projection_pos = viewport.pos,
                    .projection_size = viewport.size,
                    .render_size = vp_size,
                    .clip_rect = clip_rects[i],
                });
            }
        }

        if (panels.empty())
            return;

        if (viewport_edit_mode_ != SequencerViewportEditMode::None) {
            const auto selected = controller_.selectedKeyframe();
            const auto* const selected_keyframe =
                selected.has_value() && *selected < timeline.size()
                    ? timeline.getKeyframe(*selected)
                    : nullptr;
            if (!selected_keyframe || selected_keyframe->is_loop_point) {
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
            }
        }

        const auto projectToScreen = [&](const CameraPathPanel& panel,
                                         const glm::vec3& pos) -> glm::vec2 {
            const auto projected = lfs::rendering::projectWorldPoint(
                panel.viewport->getRotationMatrix(),
                panel.viewport->getTranslation(),
                panel.render_size,
                pos,
                settings.focal_length_mm,
                settings.orthographic,
                settings.ortho_scale);
            if (!projected)
                return {-10000.0f, -10000.0f};
            const float scale_x =
                panel.projection_size.x / static_cast<float>(std::max(panel.render_size.x, 1));
            const float scale_y =
                panel.projection_size.y / static_cast<float>(std::max(panel.render_size.y, 1));
            return {
                panel.projection_pos.x + projected->x * scale_x,
                panel.projection_pos.y + projected->y * scale_y,
            };
        };

        const auto isVisible = [&](const CameraPathPanel& panel,
                                   const glm::vec3& pos) -> bool {
            const auto projected = lfs::rendering::projectWorldPoint(
                panel.viewport->getRotationMatrix(),
                panel.viewport->getTranslation(),
                panel.render_size,
                pos,
                settings.focal_length_mm,
                settings.orthographic,
                settings.ortho_scale);
            if (!projected)
                return false;
            const float margin_x =
                (NDC_CULL_MARGIN - 1.0f) * 0.5f * static_cast<float>(panel.render_size.x);
            const float margin_y =
                (NDC_CULL_MARGIN - 1.0f) * 0.5f * static_cast<float>(panel.render_size.y);
            return projected->x >= -margin_x &&
                   projected->x <= static_cast<float>(panel.render_size.x) + margin_x &&
                   projected->y >= -margin_y &&
                   projected->y <= static_cast<float>(panel.render_size.y) + margin_y;
        };

        const auto toColor = [](const ThemeColor& c, const float alpha) -> glm::vec4 {
            return {c.x, c.y, c.z, alpha};
        };

        const auto& t = theme();
        const auto* const wm = viewer_->getWindowManager();
        const glm::ivec2 screen_size = wm ? wm->getWindowSize() : glm::ivec2{};
        const glm::ivec2 framebuffer_size = wm ? wm->getFramebufferSize() : glm::ivec2{};
        const int screen_w = screen_size.x;
        const int screen_h = screen_size.y;
        const int fb_w = framebuffer_size.x;
        const int fb_h = framebuffer_size.y;

        const int path_framerate = std::max(ui_state_.framerate, 1);
        const float base_path_time_step = 1.0f / static_cast<float>(path_framerate);
        const float path_duration = std::max(timeline.endTime() - timeline.startTime(), 0.0f);
        const size_t target_render_samples = std::clamp<size_t>(
            static_cast<size_t>(
                std::ceil(std::max(viewport.size.x, 1.0f) * PATH_SAMPLES_PER_VIEWPORT_PIXEL)),
            MIN_PATH_RENDER_SAMPLES, MAX_PATH_RENDER_SAMPLES);
        const float capped_path_time_step =
            (path_duration > 0.0f && target_render_samples > 1)
                ? path_duration / static_cast<float>(target_render_samples - 1)
                : base_path_time_step;
        const float path_time_step = std::max(base_path_time_step, capped_path_time_step);
        const auto path_points = timeline.generatePathAtTimeStep(path_time_step);

        const auto& input = viewer_->getWindowManager()->frameInput();
        const float mouse_x = input.mouse_x;
        const float mouse_y = input.mouse_y;
        const CameraPathPanel* mouse_panel = nullptr;
        for (const auto& panel : panels) {
            if (panel.contains(mouse_x, mouse_y)) {
                mouse_panel = &panel;
                break;
            }
        }

        std::optional<size_t> hovered_keyframe;
        float closest_dist = HIT_RADIUS;

        const glm::vec4 frustum_color = toColor(t.palette.primary, 0.7f);
        const glm::vec4 hovered_frustum_color = toColor(lighten(t.palette.primary, 0.15f), 0.85f);
        const glm::vec4 selected_frustum_color = toColor(lighten(t.palette.primary, 0.3f), 0.9f);

        if (mouse_panel) {
            for (size_t i = 0; i < timeline.keyframes().size(); ++i) {
                const auto& kf = timeline.keyframes()[i];
                if (kf.is_loop_point)
                    continue;
                if (!isVisible(*mouse_panel, kf.position))
                    continue;

                const glm::vec2 s_apex = projectToScreen(*mouse_panel, kf.position);
                const float dx = mouse_x - s_apex.x;
                const float dy = mouse_y - s_apex.y;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist < closest_dist) {
                    closest_dist = dist;
                    hovered_keyframe = i;
                }
            }
        }

        const auto drawOverlay = [&](const CameraPathPanel& panel) {
            if (path_points.size() >= 2) {
                const glm::vec4 path_color = toColor(t.palette.primary, 0.8f);
                const glm::vec4 sample_color = toColor(t.palette.primary, 0.45f);
                for (size_t i = 0; i + 1 < path_points.size(); ++i) {
                    if (!isVisible(panel, path_points[i]) && !isVisible(panel, path_points[i + 1]))
                        continue;
                    line_renderer_.addLine(projectToScreen(panel, path_points[i]),
                                           projectToScreen(panel, path_points[i + 1]),
                                           path_color, PATH_THICKNESS);
                }

                const size_t marker_stride =
                    std::max<size_t>(path_points.size() / MAX_PATH_SAMPLE_MARKERS, 1);
                for (size_t i = 0; i < path_points.size(); i += marker_stride) {
                    if (!isVisible(panel, path_points[i]))
                        continue;
                    line_renderer_.addCircleFilled(projectToScreen(panel, path_points[i]),
                                                   PATH_SAMPLE_RADIUS,
                                                   sample_color, 10);
                }
            }

            for (size_t i = 0; i < timeline.keyframes().size(); ++i) {
                const auto& kf = timeline.keyframes()[i];
                if (kf.is_loop_point)
                    continue;
                if (!isVisible(panel, kf.position))
                    continue;

                const glm::vec2 s_apex = projectToScreen(panel, kf.position);
                const bool selected = controller_.selectedKeyframe() == i;
                const bool hovered = mouse_panel == &panel && hovered_keyframe == i;
                glm::vec4 color = frustum_color;
                if (selected)
                    color = selected_frustum_color;
                else if (hovered)
                    color = hovered_frustum_color;
                const float thickness = selected ? FRUSTUM_THICKNESS * 1.5f : FRUSTUM_THICKNESS;

                const float half_vfov = rendering::focalLengthToVFovRad(kf.focal_length_mm) * 0.5f;
                const float half_h = std::tan(half_vfov) * FRUSTUM_DEPTH;
                const float half_w = half_h * SENSOR_ASPECT;

                const glm::mat3 rot_mat = glm::mat3_cast(kf.rotation);
                const glm::vec3 forward = rendering::cameraForward(rot_mat);
                const glm::vec3 up = rendering::cameraUp(rot_mat);
                const glm::vec3 right = rendering::cameraRight(rot_mat);

                const glm::vec3 apex = kf.position;
                const glm::vec3 base_center = apex + forward * FRUSTUM_DEPTH;
                const glm::vec3 tl = base_center + up * half_h - right * half_w;
                const glm::vec3 tr = base_center + up * half_h + right * half_w;
                const glm::vec3 bl = base_center - up * half_h - right * half_w;
                const glm::vec3 br = base_center - up * half_h + right * half_w;

                const glm::vec2 s_tl = projectToScreen(panel, tl);
                const glm::vec2 s_tr = projectToScreen(panel, tr);
                const glm::vec2 s_bl = projectToScreen(panel, bl);
                const glm::vec2 s_br = projectToScreen(panel, br);

                line_renderer_.addLine(s_apex, s_tl, color, thickness);
                line_renderer_.addLine(s_apex, s_tr, color, thickness);
                line_renderer_.addLine(s_apex, s_bl, color, thickness);
                line_renderer_.addLine(s_apex, s_br, color, thickness);

                line_renderer_.addLine(s_tl, s_tr, color, thickness);
                line_renderer_.addLine(s_tr, s_br, color, thickness);
                line_renderer_.addLine(s_br, s_bl, color, thickness);
                line_renderer_.addLine(s_bl, s_tl, color, thickness);

                const glm::vec3 up_tip = base_center + up * half_h * 1.3f;
                const glm::vec2 s_up = projectToScreen(panel, up_tip);
                line_renderer_.addTriangleFilled(s_up, s_tl, s_tr, color);
            }

            if (!controller_.isStopped()) {
                const auto state = controller_.currentCameraState();
                if (isVisible(panel, state.position)) {
                    const glm::vec4 playhead_color = toColor(t.palette.error, 1.0f);
                    constexpr float PLAYHEAD_FRUSTUM_DEPTH = 0.20f;

                    const float ph_half_vfov = rendering::focalLengthToVFovRad(state.focal_length_mm) * 0.5f;
                    const float ph_half_h = std::tan(ph_half_vfov) * PLAYHEAD_FRUSTUM_DEPTH;
                    const float ph_half_w = ph_half_h * SENSOR_ASPECT;

                    const glm::mat3 rot_mat = glm::mat3_cast(state.rotation);
                    const glm::vec3 forward = rendering::cameraForward(rot_mat);
                    const glm::vec3 up = rendering::cameraUp(rot_mat);
                    const glm::vec3 right = rendering::cameraRight(rot_mat);

                    const glm::vec3 apex = state.position;
                    const glm::vec3 base_center = apex + forward * PLAYHEAD_FRUSTUM_DEPTH;
                    const glm::vec3 tl = base_center + up * ph_half_h - right * ph_half_w;
                    const glm::vec3 tr = base_center + up * ph_half_h + right * ph_half_w;
                    const glm::vec3 bl = base_center - up * ph_half_h - right * ph_half_w;
                    const glm::vec3 br = base_center - up * ph_half_h + right * ph_half_w;

                    const glm::vec2 s_apex = projectToScreen(panel, apex);
                    const glm::vec2 s_tl = projectToScreen(panel, tl);
                    const glm::vec2 s_tr = projectToScreen(panel, tr);
                    const glm::vec2 s_bl = projectToScreen(panel, bl);
                    const glm::vec2 s_br = projectToScreen(panel, br);

                    line_renderer_.addLine(s_apex, s_tl, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_apex, s_tr, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_apex, s_bl, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_apex, s_br, playhead_color, FRUSTUM_THICKNESS);

                    line_renderer_.addLine(s_tl, s_tr, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_tr, s_br, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_br, s_bl, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_bl, s_tl, playhead_color, FRUSTUM_THICKNESS);

                    const glm::vec3 up_tip = base_center + up * ph_half_h * 1.3f;
                    const glm::vec2 s_up = projectToScreen(panel, up_tip);
                    line_renderer_.addTriangleFilled(s_up, s_tl, s_tr, playhead_color);
                }
            }
        };

        for (const auto& panel : panels) {
            line_renderer_.begin(screen_w, screen_h, fb_w, fb_h, panel.clip_rect);
            drawOverlay(panel);
            line_renderer_.end();
        }

        const bool overlay_blocks_mouse =
            overlay_->wantsInput() || overlay_->isMouseOverEditOverlay(mouse_x, mouse_y);
        const bool mouse_blocked_by_ui =
            overlay_blocks_mouse ||
            guiFocusState().want_capture_mouse;

        if (mouse_panel && !mouse_blocked_by_ui && hovered_keyframe.has_value()) {
            const auto* const hovered = timeline.getKeyframe(*hovered_keyframe);
            if (hovered && !hovered->is_loop_point) {
                if (input.mouse_clicked[0] &&
                    !isRotationGizmoHovered() &&
                    !isTranslationGizmoHovered()) {
                    beginViewportKeyframeEdit(*hovered_keyframe);
                    guiFocusState().want_capture_mouse = true;
                }
                if (input.mouse_clicked[1]) {
                    overlay_->showContextMenu(mouse_x, mouse_y, hovered_keyframe,
                                              hovered->time, viewport_edit_mode_);
                    guiFocusState().want_capture_mouse = true;
                }
            }
        }
    }

    void SequencerUIManager::renderKeyframeGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        if (viewport_edit_mode_ == SequencerViewportEditMode::None)
            return;

        const auto selected = controller_.selectedKeyframe();
        const auto selected_id = controller_.selectedKeyframeId();
        if (!selected.has_value() || !selected_id.has_value()) {
            viewport_edit_mode_ = SequencerViewportEditMode::None;
            keyframe_gizmo_active_ = false;
            return;
        }

        const auto& timeline = controller_.timeline();
        const auto* const kf = timeline.getKeyframe(*selected);
        if (!kf || kf->is_loop_point) {
            viewport_edit_mode_ = SequencerViewportEditMode::None;
            keyframe_gizmo_active_ = false;
            return;
        }

        auto* const rendering_manager = viewer_ ? viewer_->getRenderingManager() : nullptr;
        if (!rendering_manager)
            return;

        auto& primary_viewport = ctx.viewer ? ctx.viewer->getViewport() : viewer_->getViewport();
        const auto& settings = rendering_manager->getSettings();

        const auto& input = viewer_->getWindowManager()->frameInput();
        std::optional<glm::vec2> screen_point;
        if (input.mouse_x >= viewport.pos.x &&
            input.mouse_x <= viewport.pos.x + viewport.size.x &&
            input.mouse_y >= viewport.pos.y &&
            input.mouse_y <= viewport.pos.y + viewport.size.y) {
            screen_point = glm::vec2(input.mouse_x, input.mouse_y);
        }

        const Viewport* gizmo_viewport = &primary_viewport;
        glm::vec2 rect_pos = viewport.pos;
        glm::vec2 rect_size = viewport.size;
        glm::ivec2 render_size(static_cast<int>(std::round(viewport.size.x)),
                               static_cast<int>(std::round(viewport.size.y)));

        if (rendering_manager->isIndependentSplitViewActive()) {
            auto panel = rendering_manager->resolveViewerPanel(
                primary_viewport, viewport.pos, viewport.size, screen_point, std::nullopt);
            if (!panel || !panel->valid()) {
                panel = rendering_manager->resolveViewerPanel(
                    primary_viewport,
                    viewport.pos,
                    viewport.size,
                    std::nullopt,
                    rendering_manager->getFocusedSplitPanel());
            }
            if (panel && panel->valid()) {
                gizmo_viewport = panel->viewport;
                rect_pos = {panel->x, panel->y};
                rect_size = {panel->width, panel->height};
                render_size = {panel->render_width, panel->render_height};
            }
        }

        if (!gizmo_viewport || rect_size.x <= 0.0f || rect_size.y <= 0.0f ||
            render_size.x <= 0 || render_size.y <= 0) {
            return;
        }

        const glm::mat4 view = gizmo_viewport->getViewMatrix();
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            render_size,
            settings.focal_length_mm,
            settings.orthographic,
            settings.ortho_scale);

        const glm::mat3 rot_mat = glm::mat3_cast(kf->rotation);
        glm::mat4 gizmo_matrix(rot_mat);
        gizmo_matrix[3] = glm::vec4(kf->position, 1.0f);

        const bool rotate_mode = viewport_edit_mode_ == SequencerViewportEditMode::Rotate;

        NativeOverlayDrawList draw_list;
        const glm::vec2 clip_min(rect_pos.x, rect_pos.y);
        const glm::vec2 clip_max(rect_pos.x + rect_size.x, rect_pos.y + rect_size.y);
        draw_list.PushClipRect(clip_min, clip_max, true);
        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        const NativeGizmoInput gizmo_input{
            .mouse_pos = {frame_input.mouse_x, frame_input.mouse_y},
            .mouse_left_down = frame_input.mouse_down[0],
            .mouse_left_clicked = frame_input.mouse_clicked[0],
        };
        const bool snap_modifier = (frame_input.key_mods & SDL_KMOD_CTRL) != 0;

        bool changed = false;
        bool is_using = false;
        glm::mat3 rotation_delta(1.0f);

        if (rotate_mode) {
            RotationGizmoConfig rotation_config;
            rotation_config.id = 4000;
            rotation_config.viewport_pos = rect_pos;
            rotation_config.viewport_size = rect_size;
            rotation_config.view = view;
            rotation_config.projection = projection;
            rotation_config.pivot_world = kf->position;
            rotation_config.orientation_world = rot_mat;
            rotation_config.draw_list = &draw_list;
            rotation_config.input = gizmo_input;
            rotation_config.snap = snap_modifier;
            rotation_config.snap_degrees = 5.0f;

            const auto rotation_result = drawRotationGizmo(rotation_config);
            changed = rotation_result.changed;
            is_using = rotation_result.active;
            rotation_delta = rotation_result.delta_rotation;
            if (rotation_result.hovered || rotation_result.active)
                guiFocusState().want_capture_mouse = true;
        } else {
            TranslationGizmoConfig translation_config;
            translation_config.id = 4000;
            translation_config.viewport_pos = rect_pos;
            translation_config.viewport_size = rect_size;
            translation_config.view = view;
            translation_config.projection = projection;
            translation_config.pivot_world = kf->position;
            translation_config.orientation_world = glm::mat3(1.0f);
            translation_config.draw_list = &draw_list;
            translation_config.input = gizmo_input;
            translation_config.snap = snap_modifier;
            translation_config.snap_units = 0.1f;

            const auto translation_result = drawTranslationGizmo(translation_config);
            changed = translation_result.changed;
            is_using = translation_result.active;
            if (translation_result.active) {
                const glm::vec3 translated_position = kf->position + translation_result.delta_translation;
                gizmo_matrix[3] = glm::vec4(translated_position, 1.0f);
            }
            if (translation_result.hovered || translation_result.active)
                guiFocusState().want_capture_mouse = true;
        }

        if (is_using)
            guiFocusState().want_capture_mouse = true;

        if (is_using && !keyframe_gizmo_active_)
            keyframe_gizmo_active_ = true;

        if (changed) {
            const glm::vec3 new_pos(gizmo_matrix[3]);
            const glm::quat new_rot = rotate_mode
                                          ? glm::normalize(glm::quat_cast(rotation_delta * rot_mat))
                                          : glm::normalize(glm::quat_cast(glm::mat3(gizmo_matrix)));
            if (controller_.updateKeyframeById(
                    *selected_id,
                    new_pos,
                    new_rot,
                    kf->focal_length_mm)) {
                pip_needs_update_ = true;
                rendering_manager->markDirty(DirtyFlag::OVERLAY);
            }
        }

        if (!is_using && keyframe_gizmo_active_) {
            keyframe_gizmo_active_ = false;
            lfs::core::events::state::KeyframeListChanged{
                .count = controller_.timeline().realKeyframeCount()}
                .emit();
        }

        draw_list.PopClipRect();
    }

    void SequencerUIManager::loadPlySequenceFromDirectory(const std::filesystem::path& directory) {
        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager)
            return;

        std::error_code ec;
        if (!std::filesystem::is_directory(directory, ec)) {
            LOG_ERROR("PLY sequence path is not a directory: {}", lfs::core::path_to_utf8(directory));
            return;
        }
        last_ply_sequence_frame_ = std::nullopt;
        loaded_ply_sequence_frames_.clear();

        std::vector<std::filesystem::path> paths;
        const std::filesystem::directory_iterator entries(directory, ec);
        if (ec) {
            LOG_ERROR("Failed to read PLY sequence directory {}: {}",
                      lfs::core::path_to_utf8(directory),
                      ec.message());
            return;
        }
        for (const auto& entry : entries) {
            if (ec) {
                LOG_ERROR("Failed to read PLY sequence directory {}: {}",
                          lfs::core::path_to_utf8(directory),
                          ec.message());
                return;
            }
            if (!entry.is_regular_file(ec))
                continue;
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".ply")
                paths.push_back(entry.path());
        }

        std::sort(paths.begin(), paths.end());
        if (paths.empty()) {
            LOG_WARN("No PLY files found in sequence directory: {}", lfs::core::path_to_utf8(directory));
            return;
        }

        // A PLY sequence replaces the scene. Each file gets a real scene-graph
        // child immediately; the heavy SplatData is streamed into those nodes.
        if (scene_manager->getContentType() != SceneManager::ContentType::Empty) {
            if (!scene_manager->clear())
                return;
        }

        const std::string sequence_prefix = lfs::core::path_to_utf8(directory.filename().empty()
                                                                        ? directory.parent_path().filename()
                                                                        : directory.filename());
        const std::string sequence_name = sequence_prefix.empty() ? "PLY Sequence" : sequence_prefix;
        const std::string sequence_node = scene_manager->addPlySequenceNode(sequence_name, "", paths.size());
        if (sequence_node.empty()) {
            LOG_ERROR("Failed to create PLY sequence node for {}",
                      lfs::core::path_to_utf8(directory));
            return;
        }

        std::vector<std::filesystem::path> loaded_paths;
        std::vector<std::string> node_names;
        loaded_paths.reserve(paths.size());
        node_names.reserve(paths.size());
        auto splat_allocator = scene_manager->makeExternalSplatAllocator();
        auto& scene = scene_manager->getScene();
        scene.setCombinedModelAllocator(splat_allocator);
        const core::NodeId sequence_id = scene.getNodeIdByName(sequence_node);
        if (sequence_id == core::NULL_NODE) {
            LOG_ERROR("Failed to resolve PLY sequence node '{}'", sequence_node);
            return;
        }

        for (size_t i = 0; i < paths.size(); ++i) {
            const std::string stem = lfs::core::path_to_utf8(paths[i].stem());
            const std::string base_name = std::format("{}_{:04}_{}", sequence_node, i, stem);
            std::string node_name = base_name;
            for (int suffix = 1; scene.getNode(node_name); ++suffix)
                node_name = std::format("{}_{}", base_name, suffix);

            if (scene.addSplatPlaceholder(node_name, sequence_id) == core::NULL_NODE) {
                LOG_ERROR("Failed to create PLY sequence frame placeholder '{}'", node_name);
                if (!scene_manager->clear())
                    LOG_WARN("Failed to clear partial PLY sequence after placeholder failure");
                return;
            }
            scene.setNodeVisibility(node_name, false);
            scene_manager->setPlyPath(node_name, paths[i]);
            loaded_paths.push_back(paths[i]);
            node_names.push_back(node_name);
            LOG_DEBUG("Added PLY sequence placeholder '{}'", node_name);
        }

        ui_state_.sequence_fps = std::clamp(ui_state_.sequence_fps, MIN_SEQUENCE_FPS, MAX_SEQUENCE_FPS);
        controller_.setPlySequence(directory, sequence_node, std::move(loaded_paths), std::move(node_names), ui_state_.sequence_fps);
        ui_state_.sequence_fps = controller_.plySequenceFps();
        last_ply_sequence_frame_ = std::nullopt;
        startPlySequenceStreaming(paths, std::move(splat_allocator));
        applyPlySequenceFrame();

        if (const auto* sequence = controller_.plySequence()) {
            scene_manager->selectNode(sequence->node_name);
            LOG_INFO("Registered PLY sequence '{}' with {} frames at {} fps",
                     lfs::core::path_to_utf8(directory),
                     sequence->frames.size(),
                     sequence->fps);
        }

        lfs::core::events::state::KeyframeListChanged{
            .count = controller_.timeline().realKeyframeCount()}
            .emit();
    }

    void SequencerUIManager::applyPlySequenceFrame() {
        drainPlySequenceStream();
        auto* const scene_manager = viewer_->getSceneManager();
        const auto* const sequence = controller_.plySequence();
        const auto frame_index = controller_.currentPlySequenceFrameIndex();
        if (!scene_manager || !sequence || !frame_index.has_value()) {
            last_ply_sequence_frame_ = std::nullopt;
            return;
        }
        if (*frame_index >= sequence->frames.size())
            return;

        const size_t requested_frame = *frame_index;
        requestPlySequenceWindow(requested_frame);
        const auto display_frame = selectPlySequenceDisplayFrame(requested_frame);
        if (!display_frame.has_value()) {
            std::lock_guard lock(ply_stream_mutex_);
            ++ply_stream_miss_count_;
            return;
        }
        if (*display_frame != requested_frame) {
            std::lock_guard lock(ply_stream_mutex_);
            ++ply_stream_miss_count_;
            ++ply_stream_fallback_count_;
        }
        if (last_ply_sequence_frame_ == display_frame)
            return;

        auto& scene = scene_manager->getScene();
        if (last_ply_sequence_frame_.has_value() &&
            *last_ply_sequence_frame_ < sequence->frames.size()) {
            const std::string& previous = sequence->frames[*last_ply_sequence_frame_].node_name;
            if (scene.getNode(previous))
                scene.setNodeVisibility(previous, false);
        } else {
            for (const size_t loaded_frame : loaded_ply_sequence_frames_) {
                if (loaded_frame == *display_frame || loaded_frame >= sequence->frames.size())
                    continue;
                const std::string& loaded_name = sequence->frames[loaded_frame].node_name;
                if (scene.getNode(loaded_name))
                    scene.setNodeVisibility(loaded_name, false);
            }
        }

        const std::string& active = sequence->frames[*display_frame].node_name;
        if (scene.getNode(active))
            scene.setNodeVisibility(active, true);
        last_ply_sequence_frame_ = display_frame;
        if (auto* const rm = viewer_->getRenderingManager())
            rm->markDirty(DirtyFlag::SPLATS);
    }

    void SequencerUIManager::handleOverlayActions() {
        using Action = RmlSequencerOverlay::Action;
        using namespace lfs::core::events;

        while (auto action = overlay_->consumeAction()) {
            switch (action->action) {
            case Action::ADD_KEYFRAME: {
                const auto& cam = viewer_->getViewport().camera;
                auto* const rm = viewer_->getRenderingManager();
                const float focal_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
                const float time = std::max(0.0f, action->time);

                lfs::sequencer::Keyframe kf;
                kf.position = cam.t;
                kf.rotation = glm::quat_cast(cam.R);
                kf.focal_length_mm = focal_mm;
                controller_.addKeyframeAtTime(kf, time);
                controller_.seek(time);
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                pip_needs_update_ = true;
            } break;
            case Action::UPDATE_KEYFRAME:
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                cmd::SequencerUpdateKeyframe{}.emit();
                break;
            case Action::GOTO_KEYFRAME:
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerGoToKeyframe{.keyframe_index = action->keyframe_index}.emit();
                break;
            case Action::EDIT_FOCAL_LENGTH:
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                panel_->openFocalLengthEdit(
                    action->keyframe_index,
                    controller_.timeline().keyframes()[action->keyframe_index].focal_length_mm);
                break;
            case Action::SET_TRANSLATE:
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                viewport_edit_mode_ = (viewport_edit_mode_ == SequencerViewportEditMode::Translate)
                                          ? SequencerViewportEditMode::None
                                          : SequencerViewportEditMode::Translate;
                if (auto* const rm = viewer_->getRenderingManager())
                    rm->markDirty(DirtyFlag::OVERLAY);
                break;
            case Action::SET_ROTATE:
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                viewport_edit_mode_ = (viewport_edit_mode_ == SequencerViewportEditMode::Rotate)
                                          ? SequencerViewportEditMode::None
                                          : SequencerViewportEditMode::Rotate;
                if (auto* const rm = viewer_->getRenderingManager())
                    rm->markDirty(DirtyFlag::OVERLAY);
                break;
            case Action::SET_EASING: {
                const auto easing = static_cast<sequencer::EasingType>(action->easing_value);
                controller_.setKeyframeEasing(action->keyframe_index, easing);
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                break;
            }
            case Action::DELETE_KEYFRAME:
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                controller_.removeSelectedKeyframe();
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                break;
            case Action::CLOSE_EDIT_PANEL:
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                break;
            case Action::APPLY_EDIT:
                if (viewport_keyframe_edit_snapshot_.has_value()) {
                    const auto view_state = currentViewportCameraState();
                    if (controller_.updateKeyframeById(
                            viewport_keyframe_edit_snapshot_->id,
                            view_state.position,
                            view_state.rotation,
                            view_state.focal_length_mm)) {
                        state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                        pip_needs_update_ = true;
                    }
                    if (const auto* const keyframe =
                            controller_.timeline().getKeyframeById(
                                viewport_keyframe_edit_snapshot_->id)) {
                        viewport_keyframe_edit_snapshot_ = *keyframe;
                    }
                }
                break;
            case Action::REVERT_EDIT: {
                if (viewport_keyframe_edit_snapshot_.has_value()) {
                    restoreViewportCameraState({.position = viewport_keyframe_edit_snapshot_->position,
                                                .rotation = viewport_keyframe_edit_snapshot_->rotation,
                                                .focal_length_mm = viewport_keyframe_edit_snapshot_->focal_length_mm});
                }
                break;
            }
            }
        }

        if (auto time_result = overlay_->consumeTimeEdit()) {
            if (controller_.setKeyframeTime(time_result->index, time_result->value)) {
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                pip_needs_update_ = true;
            }
        }

        if (auto focal_result = overlay_->consumeFocalEdit()) {
            if (controller_.setKeyframeFocalLength(focal_result->index, focal_result->value)) {
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                pip_needs_update_ = true;
            }
        }
    }

    void SequencerUIManager::initPipPreview() {
        pip_initialized_ = true;
    }

    void SequencerUIManager::renderKeyframePreview(const UIContext& ctx) {
        if (!ui_state_.show_pip_preview)
            return;

        const bool is_playing = !controller_.isStopped();
        const auto selected = controller_.selectedKeyframe();

        const auto now = std::chrono::steady_clock::now();
        if (is_playing) {
            const float elapsed = std::chrono::duration<float>(now - pip_last_render_time_).count();
            if (elapsed < 1.0f / PREVIEW_TARGET_FPS)
                return;
        }

        auto* const rm = ctx.viewer->getRenderingManager();
        auto* const sm = ctx.viewer->getSceneManager();
        if (!rm || !sm)
            return;

        if (!pip_initialized_)
            initPipPreview();

        glm::mat3 cam_rot;
        glm::vec3 cam_pos;
        float cam_focal_length_mm;
        auto& vp = ctx.viewer->getViewport();

        if (is_playing) {
            const auto state = controller_.currentCameraState();
            cam_rot = glm::mat3_cast(state.rotation);
            cam_pos = state.position;
            cam_focal_length_mm = state.focal_length_mm;
        } else {
            if (selected.has_value()) {
                if (pip_last_keyframe_ == selected && !pip_needs_update_)
                    return;

                const auto& timeline = controller_.timeline();
                if (*selected >= timeline.size())
                    return;

                const auto* const kf = timeline.getKeyframe(*selected);
                if (!kf)
                    return;

                cam_rot = glm::mat3_cast(kf->rotation);
                cam_pos = kf->position;
                cam_focal_length_mm = kf->focal_length_mm;
            } else {
                if (pip_last_keyframe_.has_value()) {
                    pip_needs_update_ = true;
                    pip_last_keyframe_ = std::nullopt;
                }
                if (!pip_needs_update_)
                    return;

                cam_rot = vp.camera.R;
                cam_pos = vp.camera.t;
                cam_focal_length_mm = rm ? rm->getFocalLengthMm()
                                         : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
            }
        }

        const auto image = rm->renderPreviewImage(sm, cam_rot, cam_pos, cam_focal_length_mm,
                                                  PREVIEW_WIDTH, PREVIEW_HEIGHT);
        // Vulkan preview readback is already in the orientation RmlUi samples.
        if (image && pip_texture_.upload(*image, PREVIEW_WIDTH, PREVIEW_HEIGHT, /*flip_y=*/false)) {
            pip_last_render_time_ = now;
            if (!is_playing) {
                pip_last_keyframe_ = selected;
                pip_needs_update_ = false;
            }
        }
    }

    void SequencerUIManager::syncPipPreviewWindow(const ViewportLayout& viewport) {
        if (!overlay_)
            return;

        if (!ui_state_.show_pip_preview) {
            overlay_->hidePreviewWindow();
            return;
        }

        const bool is_playing = !controller_.isStopped();
        const auto selected = controller_.selectedKeyframe();

        if (!pip_initialized_ || !pip_texture_.valid()) {
            overlay_->hidePreviewWindow();
            return;
        }

        if (!is_playing && selected.has_value()) {
            const auto& timeline = controller_.timeline();
            if (*selected >= timeline.size()) {
                overlay_->hidePreviewWindow();
                return;
            }
            const auto* const kf = timeline.getKeyframe(*selected);
            if (!kf || kf->is_loop_point) {
                overlay_->hidePreviewWindow();
                return;
            }
        }

        const float scale = ui_state_.pip_preview_scale;
        constexpr float MARGIN = 16.0f;
        constexpr float TITLE_HEIGHT = 18.0f;
        const float scaled_width = static_cast<float>(PREVIEW_WIDTH) * scale;
        const float scaled_height = static_cast<float>(PREVIEW_HEIGHT) * scale;
        const float total_height = scaled_height + TITLE_HEIGHT + 8.0f;

        const float left = viewport.pos.x + MARGIN;
        const float top = panel_->cachedPanelY() - total_height - MARGIN;

        const float playhead = controller_.playhead();
        const std::string title = (is_playing || !selected.has_value())
                                      ? std::vformat(LOC(lichtfeld::Strings::Sequencer::PLAYBACK_TIME),
                                                     std::make_format_args(playhead))
                                      : [&selected]() {
                                            const size_t kf_num = *selected + 1;
                                            return std::vformat(LOC(lichtfeld::Strings::Sequencer::KEYFRAME_PREVIEW),
                                                                std::make_format_args(kf_num));
                                        }();

        overlay_->showPreviewWindow(left, top, scaled_width, scaled_height,
                                    title, is_playing,
                                    pip_texture_.rmlSrcUrl(PREVIEW_WIDTH, PREVIEW_HEIGHT));
    }

    void SequencerUIManager::renderKeyframeEditOverlay(const ViewportLayout& viewport) {
        if (!viewport_keyframe_edit_snapshot_.has_value()) {
            overlay_->hideEditOverlay();
            return;
        }

        const auto& timeline = controller_.timeline();
        const auto selected = controller_.selectedKeyframe();
        if (!selected.has_value() || !controller_.selectedKeyframeId().has_value() ||
            *controller_.selectedKeyframeId() != viewport_keyframe_edit_snapshot_->id) {
            endViewportKeyframeEdit();
            overlay_->hideEditOverlay();
            return;
        }

        const auto keyframe_index = timeline.findKeyframeIndex(viewport_keyframe_edit_snapshot_->id);
        if (!keyframe_index.has_value()) {
            endViewportKeyframeEdit();
            overlay_->hideEditOverlay();
            return;
        }

        const auto* const keyframe = timeline.getKeyframe(*keyframe_index);
        if (!keyframe || keyframe->is_loop_point) {
            endViewportKeyframeEdit();
            overlay_->hideEditOverlay();
            return;
        }

        const auto cam = currentViewportCameraState();
        const float pos_delta = glm::length(cam.position - viewport_keyframe_edit_snapshot_->position);
        const float dot = std::clamp(std::abs(glm::dot(cam.rotation, viewport_keyframe_edit_snapshot_->rotation)), 0.0f, 1.0f);
        const float rot_delta = glm::degrees(2.0f * std::acos(dot));

        overlay_->updateEditOverlay(*keyframe_index, pos_delta, rot_delta,
                                    viewport.pos.x + viewport.size.x, viewport.pos.y);
    }

} // namespace lfs::vis::gui

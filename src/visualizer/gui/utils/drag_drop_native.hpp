/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <string>
#include <vector>

struct SDL_Window;
#ifdef __linux__
union _XEvent;
#endif

namespace lfs::vis::gui {

    // Forward declaration for Windows COM drop target
#ifdef _WIN32
    class DropTarget;
#endif

    // Native drag-and-drop handler for visual feedback during file drags
    // Provides DragEnter/DragLeave visual feedback during file drags
    class NativeDragDrop {
#ifdef _WIN32
        friend class DropTarget;
#endif

    public:
        using FileDropCallback = std::function<void(const std::vector<std::string>& paths)>;

        ~NativeDragDrop();

        // Initialize native drag-drop handling for the window
        // Returns true if platform supports drag hover detection
        bool init(SDL_Window* window);

        // Shutdown and cleanup
        void shutdown();

        // Check if drag is currently hovering over window
        [[nodiscard]] bool isDragHovering() const { return drag_hovering_; }

        // Set callbacks
        void setFileDropCallback(FileDropCallback cb) { on_file_drop_ = std::move(cb); }

        // Force-reset hovering state (called when file drop is confirmed via SDL callback)
        void resetHovering() { setDragHovering(false); }

        // Per-frame hook for platforms that need polling (no-op on X11 and Windows)
        void pollEvents();

    private:
        SDL_Window* window_ = nullptr;
        bool drag_hovering_ = false;
        bool initialized_ = false;

        FileDropCallback on_file_drop_;

        // Platform-specific implementation data
        struct PlatformData;
        PlatformData* platform_data_ = nullptr;

        void setDragHovering(bool hovering);
        void handleFileDrop(const std::vector<std::string>& paths);
#ifdef __linux__
        static bool x11EventHook(void* userdata, union _XEvent* xevent);
#endif
    };

} // namespace lfs::vis::gui

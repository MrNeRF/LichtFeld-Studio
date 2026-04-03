#pragma once

#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "zep_config.h"

#include "zep/mcommon/animation/timer.h"
#include "zep/mcommon/file/cpptoml.h"
#include "zep/mcommon/math/math.h"
#include "zep/mcommon/threadpool.h"

#include "zep/keymap.h"

#include "splits.h"

// Basic Architecture

// Editor
//      Buffers
//      Modes -> (Active BufferRegion)
// Display
//      BufferRegions (->Buffers)
//
// A buffer is just an array of chars in a gap buffer, with simple operations to insert, delete and search
// A display is something that can display a collection of regions and the editor controls in a window
// A buffer region is a single view onto a buffer inside the main display
//
// The editor has a list of ZepBuffers.
// The editor has different editor modes (vim/standard)
// ZepDisplay can render the editor (with imgui or something else)
// The display has multiple BufferRegions, each a window onto a buffer.
// Multiple regions can refer to the same buffer (N Regions : N Buffers)
// The Modes receive key presses and act on a buffer region
namespace Zep {

    namespace fs = std::filesystem;

    class ZepBuffer;
    class ZepMode;
    class ZepMode_Vim;
    class ZepMode_Standard;
    class ZepEditor;
    class ZepSyntax;
    class ZepTabWindow;
    class ZepWindow;
    class ZepTheme;
    class ZepDisplay;
    class IZepFileSystem;
    class Indexer;

    struct Region;

#define ZEP_UNUSED(var) (void)var;

    // Helpers
    inline bool ZTestFlags(const uint32_t& flags, uint32_t value) {
        return ((flags & value) ? true : false);
    }
    inline uint32_t ZSetFlags(const uint32_t& flags, uint32_t value, bool set = true) {
        if (set) {
            return flags | value;
        } else
            return flags;
    }
    inline uint32_t ZClearFlags(const uint32_t& flags, uint32_t value) {
        return flags & ~value;
    }

    namespace ZepEditorFlags {
        enum {
            None = (0),
            DisableThreads = (1 << 0),
            FastUpdate = (1 << 1)
        };
    };

    enum class ZepMouseButton {
        Left,
        Middle,
        Right,
        Unknown
    };

    enum class Msg {
        ModifyCommand,
        HandleCommand,
        RequestQuit,
        GetClipBoard,
        SetClipBoard,
        MouseMove,
        MouseDown,
        MouseUp,
        MouseWheel,
        Buffer,
        ComponentChanged,
        Tick,
        ConfigChanged,
        ToolTip
    };

    struct IZepComponent;
    class ZepMessage {
    public:
        ZepMessage(Msg id, const std::string& strIn = std::string())
            : messageId(id), str(strIn) {
        }

        ZepMessage(Msg id, const NVec2f& p, ZepMouseButton b = ZepMouseButton::Unknown)
            : messageId(id), pos(p), button(b) {
        }

        ZepMessage(Msg id, const NVec2f& p, std::string strIn)
            : messageId(id), pos(p), str(strIn) {
        }

        ZepMessage(Msg id, IZepComponent* pComp)
            : messageId(id), pComponent(pComp) {
        }

        Msg messageId;        // Message ID
        std::string str;      // Generic string for simple messages
        bool handled = false; // If the message was handled
        NVec2f pos;
        ZepMouseButton button = ZepMouseButton::Unknown;
        IZepComponent* pComponent = nullptr;
    };

    struct IZepComponent {
        virtual void Notify(std::shared_ptr<ZepMessage> message) {
            ZEP_UNUSED(message);
        };
        virtual ZepEditor& GetEditor() const = 0;
    };

    class ZepComponent : public IZepComponent {
    public:
        ZepComponent(ZepEditor& editor);
        virtual ~ZepComponent();
        ZepEditor& GetEditor() const override {
            return m_editor;
        }

    private:
        ZepEditor& m_editor;
    };

    // Registers are used by the editor to store/retrieve text fragments
    struct Register {
        Register()
            : text(""), lineWise(false) {
        }
        Register(const char* ch, bool lw = false)
            : text(ch), lineWise(lw) {
        }
        Register(uint8_t* ch, bool lw = false)
            : text((const char*)ch), lineWise(lw) {
        }
        Register(const std::string& str, bool lw = false)
            : text(str), lineWise(lw) {
        }

        std::string text;
        bool lineWise = false;
    };

    using tRegisters = std::map<std::string, Register>;
    using tBuffers = std::deque<std::shared_ptr<ZepBuffer>>;
    using tSyntaxFactory = std::function<std::shared_ptr<ZepSyntax>(ZepBuffer*)>;

    struct SyntaxProvider {
        std::string syntaxID;
        tSyntaxFactory factory = nullptr;
    };

    const float bottomBorder = 2.0f;
    const float textBorder = 2.0f;
    const float tabSelectLine = 8.0f;
    const float tabSpacing = 1.0f;
    const float leftBorderChars = 4;

#define DPI_VEC2(value) (value * GetEditor().GetDisplay().GetPixelScale())
#define DPI_Y(value)    (GetEditor().GetDisplay().GetPixelScale().y * value)
#define DPI_X(value)    (GetEditor().GetDisplay().GetPixelScale().x * value)
#define DPI_RECT(value) (value * GetEditor().GetDisplay().GetPixelScale())

    inline float FontHeightPixelsFromPointSize(float pointSize, float pixelScaleY) {
        const auto fontDotsPerInch = 72.0f;
        auto inches = pointSize / fontDotsPerInch;
        return inches * (pixelScaleY * 96.0f);
    }

    enum class EditorStyle {
        Normal = 0,
        Minimal
    };

    struct EditorConfig {
        uint32_t showScrollBar = 1;
        float scrollBarSize = 12.0f;
        float scrollBarMinSize = 16.0f;
        EditorStyle style = EditorStyle::Normal;
        NVec2f lineMargins = NVec2f(1.0f);
        NVec2f widgetMargins = NVec2f(1.0f);
        NVec2f inlineWidgetMargins = NVec2f(2.0f);
        float underlineHeight = 3.0f;
        bool showLineNumbers = true;
        bool shortTabNames = true;
        bool tabToneColors = false;
        bool showIndicatorRegion = true;
        bool autoHideCommandRegion = true;
        bool cursorLineSolid = false;
        bool showNormalModeKeyStrokes = false;
        bool searchGitRoot = true;
        float backgroundFadeTime = 60.0f;
        float backgroundFadeWait = 60.0f;
    };

    class ZepExCommand : public ZepComponent {
    public:
        ZepExCommand(ZepEditor& editor)
            : ZepComponent(editor) {
        }
        virtual ~ZepExCommand() {}
        virtual void Run(const std::vector<std::string>& args = {}) = 0;
        virtual const char* ExCommandName() const = 0;
        virtual StringId ExCommandId() const {
            return StringId(ExCommandName());
        }
        virtual void Init() {};
        virtual const KeyMap* GetKeyMappings(ZepMode&) const {
            return nullptr;
        };
    };

    struct TabRegionTab : public Region {
        NVec4f color;
        ZepTabWindow* pTabWindow = nullptr;
    };

    class ZepEditor {
    public:
        // Root path is the path to search for a config file
        ZepEditor(ZepDisplay* pDisplay, const fs::path& root, uint32_t flags = 0, IZepFileSystem* pFileSystem = nullptr);
        ~ZepEditor();

        void LoadConfig(const fs::path& config_path);
        void LoadConfig(std::shared_ptr<cpptoml::table> spConfig);
        void SaveConfig(std::shared_ptr<cpptoml::table> spConfig);
        void RequestQuit();

        void Reset();
        ZepBuffer* InitWithFileOrDir(const std::string& str);
        ZepBuffer* InitWithFile(const std::string& str);
        ZepBuffer* InitWithText(const std::string& strName, const std::string& strText);

        ZepMode* GetGlobalMode();
        void RegisterGlobalMode(std::shared_ptr<ZepMode> spMode);
        void RegisterExCommand(std::shared_ptr<ZepExCommand> spMode);
        ZepExCommand* FindExCommand(const std::string& strName);
        ZepExCommand* FindExCommand(const StringId& strName);
        void SetGlobalMode(const std::string& currentMode);
        ZepMode* GetSecondaryMode() const;
        const fs::path& GetConfigRoot() const;

        std::vector<const KeyMap*> GetGlobalKeyMaps(ZepMode& mode);

        void RegisterBufferMode(const std::string& strExtension, std::shared_ptr<ZepMode> spMode);

        void Display();

        void RegisterSyntaxFactory(const std::vector<std::string>& mappings, SyntaxProvider factory);
        bool Broadcast(std::shared_ptr<ZepMessage> payload);
        void RegisterCallback(IZepComponent* pClient) {
            m_notifyClients.insert(pClient);
        }
        void UnRegisterCallback(IZepComponent* pClient) {
            m_notifyClients.erase(pClient);
        }

        const tBuffers& GetBuffers() const;
        ZepBuffer* GetMRUBuffer() const;
        void SaveBuffer(ZepBuffer& buffer);
        void SaveBufferAs(ZepBuffer& buffer, fs::path filePath);
        ZepBuffer* GetFileBuffer(const fs::path& filePath, uint32_t fileFlags = 0, bool create = true);
        ZepBuffer* GetEmptyBuffer(const std::string& name, uint32_t fileFlags = 0);
        void RemoveBuffer(ZepBuffer* pBuffer);
        std::vector<ZepWindow*> FindBufferWindows(const ZepBuffer* pBuffer) const;
        ZepBuffer* GetActiveBuffer() const;
        ZepBuffer* FindFileBuffer(const fs::path& filePath);
        ZepWindow* EnsureWindow(ZepBuffer& buffer);

        void SetRegister(const std::string& reg, const Register& val);
        void SetRegister(const char reg, const Register& val);
        void SetRegister(const std::string& reg, const char* pszText);
        void SetRegister(const char reg, const char* pszText);
        const Register& GetRegister(const std::string& reg);
        const Register& GetRegister(const char reg);
        const tRegisters& GetRegisters();

        void ReadClipboard();
        void WriteClipboard();

        void Notify(std::shared_ptr<ZepMessage> message);
        uint32_t GetFlags() const;
        void SetFlags(uint32_t flags);

        // Tab windows
        using tTabWindows = std::vector<ZepTabWindow*>;
        void NextTabWindow();
        void PreviousTabWindow();
        void SetCurrentTabWindow(ZepTabWindow* pTabWindow);
        void SetCurrentWindow(ZepWindow* pWindow);
        ZepTabWindow* GetActiveTabWindow() const;
        ZepTabWindow* AddTabWindow();
        void RemoveTabWindow(ZepTabWindow* pTabWindow);
        const tTabWindows& GetTabWindows() const;
        ZepWindow* GetActiveWindow() const;

        void UpdateTabs();

        ZepWindow* AddTree();
        ZepWindow* AddSearch();

        void ResetCursorTimer();
        bool GetCursorBlinkState() const;

        void ResetLastEditTimer();
        float GetLastEditElapsedTime() const;

        void RequestRefresh();
        bool RefreshRequired();

        void SetCommandText(const std::string& strCommand);

        std::string GetCommandText() const;
        const std::vector<std::string>& GetCommandLines() {
            return m_commandLines;
        }

        void UpdateWindowState();

        // Setup the display fixed_size for the editor
        void SetDisplayRegion(const NVec2f& topLeft, const NVec2f& bottomRight);
        void UpdateSize();

        ZepDisplay& GetDisplay() const {
            return *m_pDisplay;
        }

        IZepFileSystem& GetFileSystem() const {
            return *m_pFileSystem;
        }

        ZepTheme& GetTheme() const;

        bool OnMouseMove(const NVec2f& mousePos);
        bool OnMouseDown(const NVec2f& mousePos, ZepMouseButton button);
        bool OnMouseUp(const NVec2f& mousePos, ZepMouseButton button);
        bool OnMouseWheel(const NVec2f& mousePos, float scrollDistance);
        const NVec2f GetMousePos() const;

        void SetBufferSyntax(ZepBuffer& buffer) const;
        void SetBufferMode(ZepBuffer& buffer) const;

        EditorConfig& GetConfig() {
            return m_config;
        }

        // Helper for macros
        ZepEditor& GetEditor() {
            return *this;
        }

        ThreadPool& GetThreadPool() const;

        // Used to inform when a file changes - called from outside zep by the platform specific code, if possible
        virtual void OnFileChanged(const fs::path& path);

        ZepBuffer* GetBufferFromHandle(uint64_t handle);

        NVec4f ModifyBackgroundColor(const NVec4f& col) const;
        float GetBackgroundAlpha() const;

    private:
        // Call GetBuffer publicly, to stop creation of duplicate buffers refering to the same file
        ZepBuffer* CreateNewBuffer(const std::string& bufferName);
        ZepBuffer* CreateNewBuffer(const fs::path& path);

        void InitBuffer(ZepBuffer& buffer);
        void InitDataGrid(ZepBuffer& buffer, const NVec2i& dimensions);

        // Ensure there is a valid tab window and return it
        ZepTabWindow* EnsureTab();

    public:
        bool isFocused = true;

    private:
        ZepDisplay* m_pDisplay;
        IZepFileSystem* m_pFileSystem;

        std::set<IZepComponent*> m_notifyClients;
        mutable tRegisters m_registers;

        std::shared_ptr<ZepTheme> m_spTheme;
        std::map<std::string, SyntaxProvider> m_mapSyntax;
        std::map<std::string, std::shared_ptr<ZepMode>> m_mapGlobalModes;
        std::map<std::string, std::shared_ptr<ZepMode>> m_mapBufferModes;
        std::map<std::string, std::shared_ptr<ZepExCommand>> m_mapExCommands;

        // Blinking cursor
        timer m_cursorTimer;

        // Last edit
        timer m_lastEditTimer;

        // Active mode
        ZepMode* m_pCurrentMode = nullptr;

        // Tab windows
        tTabWindows m_tabWindows;
        ZepTabWindow* m_pActiveTabWindow = nullptr;

        // List of buffers that the editor is managing
        // May or may not be visible
        tBuffers m_buffers;
        uint32_t m_flags = 0;

        mutable std::atomic_bool m_bPendingRefresh = {true};
        mutable bool m_lastCursorBlink = false;

        std::vector<std::string> m_commandLines; // Command information, shown under the buffer

        std::shared_ptr<Region> m_editorRegion;
        std::shared_ptr<Region> m_tabContentRegion;
        std::shared_ptr<Region> m_commandRegion;
        std::shared_ptr<Region> m_tabRegion;
        bool m_bRegionsChanged = false;

        float m_tabOffsetX = 0.0f;

        NVec2f m_mousePos = NVec2f(0.0f);

        // Config
        EditorConfig m_config;
        fs::path m_configRoot;

        std::unique_ptr<ThreadPool> m_threadPool;

        std::shared_ptr<Indexer> m_indexer;
    };

} // namespace Zep

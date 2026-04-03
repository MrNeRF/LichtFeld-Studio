#include "zep/editor.h"
#include "zep/filesystem.h"
#include "zep/indexer.h"
#include "zep/mode_search.h"
#include "zep/mode_standard.h"
#include "zep/mode_tree.h"
#include "zep/regress.h"
#include "zep/syntax.h"
#include "zep/syntax_providers.h"
#include "zep/tab_window.h"

#include "config_app.h"

#include <unordered_set>

namespace Zep {
#ifdef _DEBUG
    Zep::ZLogger logger = {true, Zep::ZLT::DBG};
#else
    Zep::ZLogger logger = {false, Zep::ZLT::INFO};
#endif
    bool Zep::ZLog::disabled = false;
} // namespace Zep

namespace Zep {

    ZepComponent::ZepComponent(ZepEditor& editor)
        : m_editor(editor) {
        m_editor.RegisterCallback(this);
    }

    ZepComponent::~ZepComponent() {
        m_editor.UnRegisterCallback(this);
    }

    float ZepEditor::GetBackgroundAlpha() const {
        // TODO: I should be able to make some kind of global functions.
        // Rewriting zep, it would be much flatter and less class based ;)
        if (m_config.style == EditorStyle::Minimal) {
            float lastEdit = GetLastEditElapsedTime();
            if (lastEdit > m_config.backgroundFadeWait) {
                // lastEdit -= m_config.backgroundFadeWait;
                return std::max(0.0f, 1.0f - (lastEdit - m_config.backgroundFadeWait) / m_config.backgroundFadeTime);
            }
        }
        return 1.0f;
    }

    NVec4f ZepEditor::ModifyBackgroundColor(const NVec4f& c) const {
        // TODO: I should be able to make some kind of global functions.
        // Rewriting zep, it would be much flatter and less class based ;)
        if (m_config.style == EditorStyle::Minimal) {
            float alpha = GetBackgroundAlpha();
            return NVec4f(c.x, c.y, c.z, alpha);
        }
        return c;
    }

    ZepEditor::ZepEditor(ZepDisplay* pDisplay, const fs::path& configRoot, uint32_t flags, IZepFileSystem* pFileSystem)
        : m_pDisplay(pDisplay), m_pFileSystem(pFileSystem), m_flags(flags), m_configRoot(configRoot) {

#if defined(ZEP_FEATURE_CPP_FILE_SYSTEM)
        if (m_pFileSystem == nullptr) {
            m_pFileSystem = new ZepFileSystemCPP(configRoot);
        }
#else
        if (m_pFileSystem == nullptr) {
            assert(!"Must supply a file system - no default available on this platform!");
            throw std::invalid_argument("pFileSystem");
        }
#endif

        if (m_flags & ZepEditorFlags::DisableThreads) {
            m_threadPool = std::make_unique<ThreadPool>(1);
        } else {
            m_threadPool = std::make_unique<ThreadPool>();
        }

        LoadConfig(m_pFileSystem->GetConfigPath() / "zep.cfg");

        m_spTheme = std::make_shared<ZepTheme>();

        assert(m_pDisplay != nullptr);
        RegisterGlobalMode(std::make_shared<ZepMode_Vim>(*this));
        RegisterGlobalMode(std::make_shared<ZepMode_Standard>(*this));
        SetGlobalMode(ZepMode_Vim::StaticName());

        timer_restart(m_cursorTimer);
        timer_restart(m_lastEditTimer);
        m_commandLines.push_back("");

        RegisterSyntaxProviders(*this);

        m_editorRegion = std::make_shared<Region>();
        m_editorRegion->layoutType = RegionLayoutType::VBox;

        m_tabRegion = std::make_shared<Region>();
        m_tabRegion->layoutType = RegionLayoutType::HBox;
        m_tabRegion->margin = NVec4f(0, textBorder, 0, textBorder);

        m_tabContentRegion = std::make_shared<Region>();
        m_commandRegion = std::make_shared<Region>();

        m_editorRegion->children.push_back(m_tabRegion);
        m_editorRegion->children.push_back(m_tabContentRegion);
        m_editorRegion->children.push_back(m_commandRegion);

#ifdef IMPLEMENTED_INDEXER
        m_indexer = std::make_shared<Indexer>(*this);
        m_indexer->StartIndexing();
#endif

        Reset();
    }

    ZepEditor::~ZepEditor() {
        std::for_each(m_tabWindows.begin(), m_tabWindows.end(), [](ZepTabWindow* w) { delete w; });
        m_tabWindows.clear();
        delete m_pDisplay;
        delete m_pFileSystem;
    }

    ThreadPool& ZepEditor::GetThreadPool() const {
        return *m_threadPool;
    }

    void ZepEditor::OnFileChanged(const fs::path& path) {
        if (path.filename() == "zep.cfg") {
            ZLOG(INFO, "Reloading config");
            LoadConfig(path);
            Broadcast(std::make_shared<ZepMessage>(Msg::ConfigChanged));
        }
    }

    // If you pass a valid path to a 'zep.cfg' file, then editor settings will serialize from that
    // You can even edit it inside zep for immediate changes :)
    void ZepEditor::LoadConfig(const fs::path& config_path) {
        if (!GetFileSystem().Exists(config_path)) {
            return;
        }

        try {
            std::shared_ptr<cpptoml::table> spConfig;
            spConfig = cpptoml::parse_file(config_path.string());
            if (spConfig == nullptr)
                return;

            LoadConfig(spConfig);
        } catch (cpptoml::parse_exception& ex) {
            std::ostringstream str;
            str << config_path.filename().string() << " : Failed to parse. " << ex.what();
            SetCommandText(str.str());
        } catch (...) {
            std::ostringstream str;
            str << config_path.filename().string() << " : Failed to parse. ";
            SetCommandText(str.str());
        }
    }

    void ZepEditor::LoadConfig(std::shared_ptr<cpptoml::table> spConfig) {
        try {
            m_config.showNormalModeKeyStrokes = spConfig->get_qualified_as<bool>("editor.show_normal_mode_keystrokes").value_or(false);
            m_config.showIndicatorRegion = spConfig->get_qualified_as<bool>("editor.show_indicator_region").value_or(true);
            m_config.showLineNumbers = spConfig->get_qualified_as<bool>("editor.show_line_numbers").value_or(true);
            m_config.autoHideCommandRegion = spConfig->get_qualified_as<bool>("editor.autohide_command_region").value_or(false);
            m_config.cursorLineSolid = spConfig->get_qualified_as<bool>("editor.cursor_line_solid").value_or(true);
            m_config.backgroundFadeTime = (float)spConfig->get_qualified_as<double>("editor.background_fade_time").value_or(60.0f);
            m_config.backgroundFadeWait = (float)spConfig->get_qualified_as<double>("editor.background_fade_wait").value_or(60.0f);
            m_config.showScrollBar = spConfig->get_qualified_as<uint32_t>("editor.show_scrollbar").value_or(1);
            m_config.scrollBarSize = (float)spConfig->get_qualified_as<double>("editor.scrollbar_size").value_or(m_config.scrollBarSize);
            m_config.scrollBarMinSize = (float)spConfig->get_qualified_as<double>("editor.scrollbar_min_size").value_or(m_config.scrollBarMinSize);
            m_config.lineMargins.x = (float)spConfig->get_qualified_as<double>("editor.line_margin_top").value_or(1);
            m_config.lineMargins.y = (float)spConfig->get_qualified_as<double>("editor.line_margin_bottom").value_or(1);
            m_config.widgetMargins.x = (float)spConfig->get_qualified_as<double>("editor.widget_margin_top").value_or(1);
            m_config.widgetMargins.y = (float)spConfig->get_qualified_as<double>("editor.widget_margin_bottom").value_or(1);
            m_config.shortTabNames = spConfig->get_qualified_as<bool>("editor.short_tab_names").value_or(false);
            m_config.tabToneColors = spConfig->get_qualified_as<bool>("editor.tab_tone_colors").value_or(false);
            m_config.searchGitRoot = spConfig->get_qualified_as<bool>("search.search_git_root").value_or(true);
            auto styleStr = string_tolower(spConfig->get_qualified_as<std::string>("editor.style").value_or("normal"));
            if (styleStr == "normal") {
                m_config.style = EditorStyle::Normal;
            } else if (styleStr == "minimal") {
                m_config.style = EditorStyle::Minimal;
            }

            // Forward settings to file system
            GetFileSystem().SetFlags(m_config.searchGitRoot ? ZepFileSystemFlags::SearchGitRoot : 0);
        } catch (...) {
        }
    }

    void ZepEditor::SaveConfig(std::shared_ptr<cpptoml::table> spConfig) {
        auto table = spConfig->get_table("editor");
        if (!table) {
            table = cpptoml::make_table();
            spConfig->insert("editor", table);
        }

        table->insert("autohide_command_region", m_config.autoHideCommandRegion);
        table->insert("background_fade_time", (double)m_config.backgroundFadeTime);
        table->insert("background_fade_wait", (double)m_config.backgroundFadeWait);
        table->insert("cursor_line_solid", m_config.cursorLineSolid);
        table->insert("line_margin_bottom", m_config.lineMargins.y);
        table->insert("line_margin_top", m_config.lineMargins.x);
        table->insert("short_tab_names", m_config.shortTabNames);
        table->insert("tab_tone_colors", m_config.tabToneColors);
        table->insert("search_git_root", m_config.searchGitRoot);
        table->insert("show_indicator_region", m_config.showIndicatorRegion);
        table->insert("show_line_numbers", m_config.showLineNumbers);
        table->insert("show_normal_mode_keystrokes", m_config.showNormalModeKeyStrokes);
        table->insert("show_scrollbar", m_config.showScrollBar);
        table->insert("scrollbar_size", (double)m_config.scrollBarSize);
        table->insert("scrollbar_min_size", (double)m_config.scrollBarMinSize);
        table->insert("widget_margin_bottom", m_config.widgetMargins.y);
        table->insert("widget_margin_top", m_config.widgetMargins.x);

        table->insert("style", m_config.style == EditorStyle::Minimal ? "minimal" : "normal");

        /*
        Example Write:
        std::ofstream stream("d:/dev/out.txt");
        cpptoml::toml_writer writer(stream, "");
        writer.visit(*spConfig);
        */
    }

    void ZepEditor::SaveBufferAs(ZepBuffer& buffer, fs::path path) {
        buffer.SetFilePath(path);
        SaveBuffer(buffer);
    }

    void ZepEditor::SaveBuffer(ZepBuffer& buffer) {
        // TODO:
        // - What if the buffer has no associated file?  Prompt for one.
        // - We don't check for outside modification yet either, meaning this could overwrite
        std::ostringstream strText;

        if (buffer.HasFileFlags(FileFlags::ReadOnly)) {
            strText << "Failed to save, Read Only: " << buffer.GetDisplayName();
        } else if (buffer.HasFileFlags(FileFlags::Locked)) {
            strText << "Failed to save, Locked: " << buffer.GetDisplayName();
        } else if (buffer.GetFilePath().empty()) {
            strText << "Error: No file name";
        } else {
            int64_t size;
            if (!buffer.Save(size)) {
                strText << "Failed to save: " << buffer.GetDisplayName() << " at: " << buffer.GetFilePath().string();
            } else {
                strText << "Wrote " << buffer.GetFilePath().string() << ", " << size << " bytes";
            }
        }
        SetCommandText(strText.str());
    }

    std::vector<ZepWindow*> ZepEditor::FindBufferWindows(const ZepBuffer* pBuffer) const {
        std::vector<ZepWindow*> bufferWindows;
        for (auto& tab : m_tabWindows) {
            for (auto& win : tab->GetWindows()) {
                if (&win->GetBuffer() == pBuffer) {
                    bufferWindows.push_back(win);
                }
            }
        }
        return bufferWindows;
    }

    void ZepEditor::RemoveBuffer(ZepBuffer* pBuffer) {
        auto bufferWindows = FindBufferWindows(pBuffer);
        for (auto& window : bufferWindows) {
            window->GetTabWindow().RemoveWindow(window);
        }

        // Find the buffer in the list of buffers owned by the editor and remove it
        auto itr = std::find_if(m_buffers.begin(), m_buffers.end(), [pBuffer](std::shared_ptr<ZepBuffer> spBuffer) {
            return spBuffer.get() == pBuffer;
        });

        if (itr != m_buffers.end()) {
            m_buffers.erase(itr);
        }
    }

    ZepBuffer* ZepEditor::GetEmptyBuffer(const std::string& name, uint32_t fileFlags) {
        auto pBuffer = CreateNewBuffer(name);
        pBuffer->SetFileFlags(fileFlags);
        return pBuffer;
    }

    ZepBuffer* ZepEditor::FindFileBuffer(const fs::path& filePath) {
        auto& buffers = GetBuffers();
        for (auto& buff : buffers) {
            if (m_pFileSystem->Equivalent(buff->GetFilePath(), filePath)) {
                return buff.get();
            }
        }
        return nullptr;
    }

    ZepBuffer* ZepEditor::GetFileBuffer(const fs::path& filePath, uint32_t fileFlags, bool create) {
        auto path = GetFileSystem().Exists(filePath) ? GetFileSystem().Canonical(filePath) : filePath;
        if (!path.empty()) {
            for (auto& pBuffer : m_buffers) {
                if (!pBuffer->GetFilePath().empty()) {
                    if (GetFileSystem().Equivalent(pBuffer->GetFilePath(), path)) {
                        return pBuffer.get();
                    }
                }
            }
        }

        if (!create) {
            return nullptr;
        }

        // Create buffer, try to load even if not present, the buffer represents the save path (it just isn't saved yet)
        auto pBuffer = CreateNewBuffer(filePath);
        pBuffer->SetFileFlags(fileFlags);

        return pBuffer;
    }

    ZepWindow* ZepEditor::AddTree() {
        auto pTree = GetEmptyBuffer("Tree.tree", FileFlags::Locked | FileFlags::ReadOnly);
        auto pTreeWindow = GetActiveTabWindow()->AddWindow(pTree, nullptr, RegionLayoutType::HBox);

        auto pActiveWindow = GetActiveTabWindow()->GetActiveWindow();
        pActiveWindow->SetBuffer(pTree);

        auto pTreeModel = std::make_shared<ZepFileTree>();
        auto pRoot = pTreeModel->GetRoot();

        pRoot->AddChild(std::make_shared<ZepFileNode>("Child1", ZepTreeNodeFlags::IsFolder));
        auto pChild2 = pRoot->AddChild(std::make_shared<ZepFileNode>("Child2", ZepTreeNodeFlags::IsFolder));
        pChild2->AddChild(std::make_shared<ZepFileNode>("Child2_1"));

        pRoot->ExpandAll(true);

        auto pMode = std::make_shared<ZepMode_Tree>(*this, pTreeModel, *pActiveWindow, *pTreeWindow);
        pTree->SetMode(pMode);
        pMode->Begin(pActiveWindow);
        return pActiveWindow;
    }

    ZepWindow* ZepEditor::AddSearch() {
        if (!GetActiveTabWindow()) {
            return nullptr;
        }

        static std::unordered_set<std::string> search_keywords = {};
        static std::unordered_set<std::string> search_identifiers = {};

        auto pSearchBuffer = GetEmptyBuffer("Search", FileFlags::Locked | FileFlags::ReadOnly);
        pSearchBuffer->SetBufferType(BufferType::Search);
        pSearchBuffer->SetSyntax(std::make_shared<ZepSyntax>(*pSearchBuffer, search_keywords, search_identifiers, ZepSyntaxFlags::CaseInsensitive));

        auto pActiveWindow = GetActiveTabWindow()->GetActiveWindow();
        bool hasGit = false;
        auto searchPath = GetFileSystem().GetSearchRoot(pActiveWindow->GetBuffer().GetFilePath(), hasGit);

        auto pSearchWindow = GetActiveTabWindow()->AddWindow(pSearchBuffer, nullptr, RegionLayoutType::VBox);
        pSearchWindow->SetWindowFlags(pSearchWindow->GetWindowFlags() | WindowFlags::Modal);

        auto pMode = std::make_shared<ZepMode_Search>(*this, *pActiveWindow, *pSearchWindow, searchPath);
        pSearchBuffer->SetMode(pMode);
        pMode->Begin(pSearchWindow);
        return pSearchWindow;
    }

    ZepTabWindow* ZepEditor::EnsureTab() {
        if (m_tabWindows.empty()) {
            return AddTabWindow();
        }

        if (m_pActiveTabWindow) {
            return m_pActiveTabWindow;
        }
        return m_tabWindows[0];
    }

    // Nothing here currently; we used to create a default tab, now we don't.  It is up to the client
    void ZepEditor::Reset() {
    }

    // TODO fix for directory startup; it won't work
    ZepBuffer* ZepEditor::InitWithFileOrDir(const std::string& str) {
        fs::path startPath(str);

        auto& fs = GetFileSystem();
        if (fs.Exists(startPath)) {
            startPath = fs.Canonical(startPath);

            // If a directory, just return the default already created buffer.
            if (fs.IsDirectory(startPath)) {
                // Remember the working directory
                fs.SetWorkingDirectory(startPath);
                return &GetActiveTabWindow()->GetActiveWindow()->GetBuffer();
            } else {
                // Try to get the working directory from the parent path of the passed file
                auto parentDir = startPath.parent_path();
                if (fs.Exists(parentDir) && fs.IsDirectory(parentDir)) {
                    fs.SetWorkingDirectory(startPath.parent_path());
                }
            }
        }

        return InitWithFile(str);
    }

    ZepBuffer* ZepEditor::InitWithFile(const std::string& str) {
        fs::path startPath(str);
        // Get a buffer for the start file; even if the path is not valid; it can be created but not saved
        auto pFileBuffer = GetFileBuffer(startPath);
        auto pTab = EnsureTab();
        pTab->AddWindow(pFileBuffer, nullptr, RegionLayoutType::HBox);

        return pFileBuffer;
    }

    ZepBuffer* ZepEditor::InitWithText(const std::string& strName, const std::string& strText) {
        auto pTab = EnsureTab();

        auto pBuffer = GetEmptyBuffer(strName);
        pBuffer->SetText(strText);

        pTab->AddWindow(pBuffer, nullptr, RegionLayoutType::HBox);

        return pBuffer;
    }

    // Here we ensure that the editor is in a valid state, and cleanup Default buffers
    void ZepEditor::UpdateWindowState() {
        // If there is no active tab window, and we have one, set it.
        if (!m_pActiveTabWindow) {
            if (!m_tabWindows.empty()) {
                SetCurrentTabWindow(m_tabWindows.back());
            }
        }

        // If the tab window doesn't contain an active window, and there is one, set it
        if (m_pActiveTabWindow && !m_pActiveTabWindow->GetActiveWindow()) {
            if (!m_pActiveTabWindow->GetWindows().empty()) {
                m_pActiveTabWindow->SetActiveWindow(m_pActiveTabWindow->GetWindows().back());
                m_bRegionsChanged = true;
            }
        }

        // Clean up any default buffers
        std::vector<ZepBuffer*> victims;
        for (auto& buffer : m_buffers) {
            if (!buffer->HasFileFlags(FileFlags::DefaultBuffer) || buffer->HasFileFlags(FileFlags::Dirty)) {
                continue;
            }

            auto windows = FindBufferWindows(buffer.get());
            if (windows.empty()) {
                victims.push_back(buffer.get());
            }
        }

        for (auto& victim : victims) {
            RemoveBuffer(victim);
        }

        // If the display says we need a layout update, force it on all the windows
        if (GetDisplay().LayoutDirty()) {
            for (auto& tabWindow : GetTabWindows()) {
                for (auto& window : tabWindow->GetWindows()) {
                    window->DirtyLayout();
                }
            }
            GetDisplay().SetLayoutDirty(false);
        }
    }

    void ZepEditor::ResetCursorTimer() {
        timer_restart(m_cursorTimer);
    }

    void ZepEditor::ResetLastEditTimer() {
        timer_restart(m_lastEditTimer);
    }

    float ZepEditor::GetLastEditElapsedTime() const {
        return (float)timer_get_elapsed_seconds(m_lastEditTimer);
    }

    void ZepEditor::NextTabWindow() {
        auto itr = std::find(m_tabWindows.begin(), m_tabWindows.end(), m_pActiveTabWindow);
        if (itr != m_tabWindows.end())
            itr++;

        if (itr == m_tabWindows.end()) {
            itr = m_tabWindows.end() - 1;
        }
        SetCurrentTabWindow(*itr);
    }

    void ZepEditor::PreviousTabWindow() {
        auto itr = std::find(m_tabWindows.begin(), m_tabWindows.end(), m_pActiveTabWindow);
        if (itr == m_tabWindows.end()) {
            return;
        }

        if (itr != m_tabWindows.begin()) {
            itr--;
        }

        SetCurrentTabWindow(*itr);
    }

    void ZepEditor::SetCurrentWindow(ZepWindow* pWindow) {
        if (!pWindow) {
            return;
        }
        auto& tab = pWindow->GetTabWindow();
        tab.SetActiveWindow(pWindow);

        SetCurrentTabWindow(&tab);
    }

    void ZepEditor::SetCurrentTabWindow(ZepTabWindow* pTabWindow) {
        // Sanity
        auto itr = std::find(m_tabWindows.begin(), m_tabWindows.end(), pTabWindow);
        if (itr != m_tabWindows.end()) {
            m_pActiveTabWindow = pTabWindow;

            // Force a reactivation of the active window to ensure buffer setup is correct
            m_pActiveTabWindow->SetActiveWindow(m_pActiveTabWindow->GetActiveWindow());
        }
    }

    ZepTabWindow* ZepEditor::GetActiveTabWindow() const {
        return m_pActiveTabWindow;
    }

    void ZepEditor::UpdateTabs() {
        m_tabRegion->children.clear();
        if (GetTabWindows().size() > 1) {
            // Tab region
            for (auto& window : GetTabWindows()) {
                if (window->GetActiveWindow() == nullptr)
                    continue;

                auto& buffer = window->GetActiveWindow()->GetBuffer();

                auto tabColor = GetTheme().GetColor(ThemeColor::TabActive);

                auto name = window->GetName();
                auto tabLength = m_pDisplay->GetFont(ZepTextType::UI).GetTextSize((const uint8_t*)name.c_str()).x + DPI_X(textBorder) * 2;

                auto spTabRegionTab = std::make_shared<TabRegionTab>();

                spTabRegionTab->color = tabColor;
                spTabRegionTab->debugName = name;
                spTabRegionTab->pTabWindow = window;
                spTabRegionTab->fixed_size = NVec2f(tabLength, 0.0f);
                spTabRegionTab->layoutType = RegionLayoutType::HBox;
                spTabRegionTab->padding = DPI_VEC2(NVec2f(textBorder, textBorder));
                spTabRegionTab->flags = RegionFlags::Fixed;
                m_tabRegion->children.push_back(spTabRegionTab);
                spTabRegionTab->pParent = m_tabRegion.get();
            }
        }
        LayoutRegion(*m_tabRegion);
    }

    ZepTabWindow* ZepEditor::AddTabWindow() {
        auto pTabWindow = new ZepTabWindow(*this);
        m_tabWindows.push_back(pTabWindow);
        m_pActiveTabWindow = pTabWindow;

        auto pEmpty = GetEmptyBuffer("[Default]", FileFlags::DefaultBuffer);
        pTabWindow->AddWindow(pEmpty, nullptr, RegionLayoutType::HBox);

        return pTabWindow;
    }

    void ZepEditor::RequestQuit() {
        Broadcast(std::make_shared<ZepMessage>(Msg::RequestQuit, "RequestQuit"));
    }

    void ZepEditor::RemoveTabWindow(ZepTabWindow* pTabWindow) {
        assert(pTabWindow);
        if (!pTabWindow)
            return;

        auto itrFound = std::find(m_tabWindows.begin(), m_tabWindows.end(), pTabWindow);
        if (itrFound == m_tabWindows.end()) {
            assert(!"Not found?");
            return;
        }

        delete pTabWindow;
        m_tabWindows.erase(itrFound);

        if (m_tabWindows.empty()) {
            m_pActiveTabWindow = nullptr;

            // Reset the window state, but request a quit
            Reset();
            RequestQuit();
        } else {
            if (m_pActiveTabWindow == pTabWindow) {
                assert(!m_tabWindows.empty());
                m_pActiveTabWindow = m_tabWindows[m_tabWindows.size() - 1];

                // Force a reset of active to initialize the mode
                m_pActiveTabWindow->SetActiveWindow(m_pActiveTabWindow->GetActiveWindow());
            }
        }

        GetEditor().UpdateTabs();
    }

    const ZepEditor::tTabWindows& ZepEditor::GetTabWindows() const {
        return m_tabWindows;
    }

    void ZepEditor::RegisterGlobalMode(std::shared_ptr<ZepMode> spMode) {
        m_mapGlobalModes[spMode->Name()] = spMode;
        spMode->Init();
    }

    void ZepEditor::RegisterExCommand(std::shared_ptr<ZepExCommand> spCommand) {
        m_mapExCommands[spCommand->ExCommandName()] = spCommand;
    }

    ZepExCommand* ZepEditor::FindExCommand(const std::string& strName) {
        auto itr = m_mapExCommands.find(strName);
        if (itr != m_mapExCommands.end()) {
            return itr->second.get();
        }
        return nullptr;
    }

    ZepExCommand* ZepEditor::FindExCommand(const StringId& Id) {
        if (Id.id == 0) {
            return nullptr;
        }

        for (auto& [name, pEx] : m_mapExCommands) {
            if (pEx->ExCommandId() == Id) {
                return pEx.get();
            }
        }
        return nullptr;
    }

    void ZepEditor::RegisterBufferMode(const std::string& extension, std::shared_ptr<ZepMode> spMode) {
        m_mapBufferModes[extension] = spMode;
        spMode->Init();
    }

    void ZepEditor::SetGlobalMode(const std::string& currentMode) {
        auto itrMode = m_mapGlobalModes.find(currentMode);
        if (itrMode != m_mapGlobalModes.end()) {
            ZepWindow* pWindow = nullptr;
            if (m_pCurrentMode) {
                pWindow = m_pCurrentMode->GetCurrentWindow();
            }

            m_pCurrentMode = itrMode->second.get();

            if (pWindow) {
                m_pCurrentMode->Begin(pWindow);
            }
        }
    }

    ZepMode* ZepEditor::GetGlobalMode() {
        // The 'Mode' is typically vim or normal and determines how editing is done in a panel
        if (!m_pCurrentMode && !m_mapGlobalModes.empty()) {
            m_pCurrentMode = m_mapGlobalModes.begin()->second.get();
        }

        return m_pCurrentMode;
    }

    void ZepEditor::SetBufferMode(ZepBuffer& buffer) const {
        // Reset it in case we are changing the text in a buffer
        buffer.SetMode(nullptr);

        std::string ext;
        std::string fileName;
        if (buffer.GetFilePath().has_filename() && buffer.GetFilePath().filename().has_extension()) {
            ext = string_tolower(buffer.GetFilePath().filename().extension().string());
            fileName = string_tolower(buffer.GetFilePath().filename().string());
        } else {
            auto str = buffer.GetName();
            size_t dot_pos = str.find_last_of(".");
            if (dot_pos != std::string::npos) {
                ext = string_tolower(str.substr(dot_pos, str.length() - dot_pos));
            }
        }

        auto itr = m_mapBufferModes.find(ext);
        if (itr != m_mapBufferModes.end()) {
            buffer.SetMode(itr->second);
        }
    }

    void ZepEditor::SetBufferSyntax(ZepBuffer& buffer) const {
        std::string ext;
        std::string fileName;
        if (buffer.GetFilePath().has_filename() && buffer.GetFilePath().filename().has_extension()) {
            ext = string_tolower(buffer.GetFilePath().filename().extension().string());
            fileName = string_tolower(buffer.GetFilePath().filename().string());
        } else {
            auto str = buffer.GetName();
            size_t dot_pos = str.find_last_of(".");
            if (dot_pos != std::string::npos) {
                ext = string_tolower(str.substr(dot_pos, str.length() - dot_pos));
            }
        }

        // first check file name
        if (!fileName.empty()) {
            auto itr = m_mapSyntax.find(fileName);
            if (itr != m_mapSyntax.end()) {
                buffer.SetSyntaxProvider(itr->second);
                return;
            }
        }

        auto itr = m_mapSyntax.find(ext);
        if (itr != m_mapSyntax.end()) {
            buffer.SetSyntaxProvider(itr->second);
        } else {
            itr = m_mapSyntax.find(string_tolower(buffer.GetName()));
            if (itr != m_mapSyntax.end()) {
                buffer.SetSyntaxProvider(itr->second);
            } else {
                buffer.SetSyntaxProvider(SyntaxProvider{});
            }
        }
    }

    void ZepEditor::RegisterSyntaxFactory(const std::vector<std::string>& mappings, SyntaxProvider provider) {
        for (auto& m : mappings) {
            m_mapSyntax[string_tolower(m)] = provider;
        }
    }

    // Inform clients of an event in the buffer
    bool ZepEditor::Broadcast(std::shared_ptr<ZepMessage> message) {
        Notify(message);
        if (message->handled)
            return true;

        for (auto& client : m_notifyClients) {
            client->Notify(message);
            if (message->handled)
                break;
        }
        return message->handled;
    }

    const std::deque<std::shared_ptr<ZepBuffer>>& ZepEditor::GetBuffers() const {
        return m_buffers;
    }

    void ZepEditor::InitDataGrid(ZepBuffer& buffer, const NVec2i& dimensions) {
        std::ostringstream str;
        for (int column = 0; column < dimensions.y; column++) {
            for (int row = 0; row < dimensions.x; row++) {
                str << ".";
            }
            str << "\n";
        }
        buffer.SetText(str.str());
    }

    // Do any special buffer processing
    void ZepEditor::InitBuffer(ZepBuffer& buffer) {
        SetBufferMode(buffer);
    }

    ZepBuffer* ZepEditor::CreateNewBuffer(const std::string& str) {
        auto pBuffer = std::make_shared<ZepBuffer>(*this, str);

        // For a new buffer, set the syntax based on the string name
        SetBufferSyntax(*pBuffer);

        m_buffers.push_front(pBuffer);

        InitBuffer(*pBuffer);

        return pBuffer.get();
    }

    ZepBuffer* ZepEditor::CreateNewBuffer(const fs::path& path) {
        auto pBuffer = std::make_shared<ZepBuffer>(*this, path);
        m_buffers.push_front(pBuffer);

        InitBuffer(*pBuffer);

        return pBuffer.get();
    }

    ZepBuffer* ZepEditor::GetMRUBuffer() const {
        return m_buffers.front().get();
    }

    void ZepEditor::ReadClipboard() {
        auto pMsg = std::make_shared<ZepMessage>(Msg::GetClipBoard);
        Broadcast(pMsg);
        if (pMsg->handled) {
            if (pMsg->str != m_registers["+"].text) {
                m_registers["+"] = pMsg->str;
                m_registers["*"] = pMsg->str;
                m_registers["\""] = pMsg->str;
            }
        }
    }

    void ZepEditor::WriteClipboard() {
        auto pMsg = std::make_shared<ZepMessage>(Msg::SetClipBoard);
        pMsg->str = m_registers["+"].text;
        Broadcast(pMsg);
    }

    void ZepEditor::SetRegister(const std::string& reg, const Register& val) {
        m_registers[reg] = val;
        if (reg == "+" || reg == "*") {
            WriteClipboard();
        }
    }

    void ZepEditor::SetRegister(const char reg, const Register& val) {
        std::string str({reg});
        m_registers[str] = val;
        if (reg == '+' || reg == '*') {
            WriteClipboard();
        }
    }

    void ZepEditor::SetRegister(const std::string& reg, const char* pszText) {
        m_registers[reg] = Register(pszText);
        if (reg == "+" || reg == "*") {
            WriteClipboard();
        }
    }

    void ZepEditor::SetRegister(const char reg, const char* pszText) {
        std::string str({reg});
        m_registers[str] = Register(pszText);
        if (reg == '+' || reg == '*') {
            WriteClipboard();
        }
    }

    const Register& ZepEditor::GetRegister(const std::string& reg) {
        if (reg == "+" || reg == "*") {
            ReadClipboard();
        }
        return m_registers[reg];
    }

    const Register& ZepEditor::GetRegister(const char reg) {
        if (reg == '+' || reg == '*') {
            ReadClipboard();
        }
        std::string str({reg});
        return m_registers[str];
    }
    const tRegisters& ZepEditor::GetRegisters() {
        ReadClipboard();
        return m_registers;
    }

    void ZepEditor::Notify(std::shared_ptr<ZepMessage> pMsg) {
        if (pMsg->messageId == Msg::MouseDown) {
            for (auto& tab : m_tabRegion->children) {
                if (tab->rect.Contains(pMsg->pos)) {
                    auto pTabRegionTab = std::static_pointer_cast<TabRegionTab>(tab);
                    SetCurrentTabWindow(pTabRegionTab->pTabWindow);
                }
            }
        }
    }

    std::string ZepEditor::GetCommandText() const {
        std::ostringstream str;
        bool start = true;
        for (auto& line : m_commandLines) {
            if (!start)
                str << "\n";
            start = false;
            str << line;
        }
        return str.str();
    }

    void ZepEditor::SetCommandText(const std::string& strCommand) {
        m_commandLines = string_split(strCommand, "\n\r");
        if (m_commandLines.empty()) {
            m_commandLines.push_back("");
        }
        m_bRegionsChanged = true;
    }

    void ZepEditor::RequestRefresh() {
        m_bPendingRefresh = true;
    }

    bool ZepEditor::RefreshRequired() {
        // Allow any components to update themselves
        Broadcast(std::make_shared<ZepMessage>(Msg::Tick));

        auto lastBlink = m_lastCursorBlink;
        if (m_bPendingRefresh || lastBlink != GetCursorBlinkState()) {
            if (!ZTestFlags(m_flags, ZepEditorFlags::FastUpdate)) {
                m_bPendingRefresh = false;
            }
            return true;
        }

        return false;
    }

    bool ZepEditor::GetCursorBlinkState() const {
        if (!isFocused)
            return false;
        m_lastCursorBlink = (int(timer_get_elapsed_seconds(m_cursorTimer) * 1.75f) & 1) ? true : false;
        return m_lastCursorBlink;
    }

    void ZepEditor::SetDisplayRegion(const NVec2f& topLeft, const NVec2f& bottomRight) {
        m_editorRegion->rect.topLeftPx = topLeft;
        m_editorRegion->rect.bottomRightPx = bottomRight;
        UpdateSize();
    }

    void ZepEditor::UpdateSize() {
        auto& uiFont = m_pDisplay->GetFont(ZepTextType::UI);
        auto commandCount = GetCommandLines().size();
        const float commandSize = uiFont.GetPixelHeight() * commandCount + DPI_X(textBorder) * 2.0f;
        auto displaySize = m_editorRegion->rect.Size();

        // Regions
        m_commandRegion->fixed_size = NVec2f(0.0f, commandSize);
        m_commandRegion->flags = RegionFlags::Fixed;

        const auto selectHeight = int(DPI_Y(tabSelectLine));

        // Add tabs for extra windows
        if (GetTabWindows().size() > 1) {
            m_tabRegion->fixed_size = NVec2f(0.0f, uiFont.GetPixelHeight() + DPI_X(textBorder) * 2 + selectHeight);
            m_tabRegion->flags = RegionFlags::Fixed;
        } else {
            m_tabRegion->fixed_size = NVec2f(0.0f);
            m_tabRegion->flags = RegionFlags::Fixed;
        }

        m_tabContentRegion->flags = RegionFlags::Expanding;

        LayoutRegion(*m_editorRegion);

        if (GetActiveTabWindow()) {
            GetActiveTabWindow()->SetDisplayRegion(m_tabContentRegion->rect);
        }
    }

    void ZepEditor::Display() {
        UpdateWindowState();

        if (m_bRegionsChanged) {
            m_bRegionsChanged = false;
            UpdateSize();
        }

        // Command plus output
        auto& commandLines = GetCommandLines();

        long commandCount = long(commandLines.size());

        auto& uiFont = m_pDisplay->GetFont(ZepTextType::UI);
        const float commandSize = uiFont.GetPixelHeight() * commandCount + DPI_X(textBorder) * 2.0f;

        auto displaySize = m_editorRegion->rect.Size();

        auto commandSpace = commandCount;
        commandSpace = std::max(commandCount, 0l);

        // This fill will effectively fill the region around the tabs in Normal mode
        if (GetConfig().style == EditorStyle::Normal) {
            GetDisplay().DrawRectFilled(m_editorRegion->rect, ModifyBackgroundColor(GetTheme().GetColor(ThemeColor::Background)));
        }

        // Background rect for CommandLine
        if (!GetCommandText().empty() || (GetConfig().autoHideCommandRegion == false)) {
            m_pDisplay->DrawRectFilled(m_commandRegion->rect, ModifyBackgroundColor(GetTheme().GetColor(ThemeColor::Background)));
        }

        // Draw command text
        auto screenPosYPx = m_commandRegion->rect.topLeftPx + NVec2f(0.0f, DPI_X(textBorder));
        for (int i = 0; i < commandSpace; i++) {
            if (!commandLines[i].empty()) {
                auto textSize = uiFont.GetTextSize((const uint8_t*)commandLines[i].c_str(), (const uint8_t*)commandLines[i].c_str() + commandLines[i].size());
                m_pDisplay->DrawChars(uiFont, screenPosYPx, GetTheme().GetColor(ThemeColor::Text), (const uint8_t*)commandLines[i].c_str());
            }

            screenPosYPx.y += uiFont.GetPixelHeight();
            screenPosYPx.x = m_commandRegion->rect.topLeftPx.x;
        }

        if (GetConfig().style == EditorStyle::Normal) {
            // A line along the bottom of the tab region
            m_pDisplay->DrawRectFilled(
                NRectf(NVec2f(m_tabRegion->rect.Left(), m_tabRegion->rect.Bottom() - DPI_Y(1)), NVec2f(m_tabRegion->rect.Right(), m_tabRegion->rect.Bottom())), GetTheme().GetColor(ThemeColor::TabInactive));
        }

        // Figure out the active region
        auto pActiveTabWindow = GetActiveTabWindow();
        NRectf tabRect;
        for (auto& tab : m_tabRegion->children) {
            if (std::static_pointer_cast<TabRegionTab>(tab)->pTabWindow == pActiveTabWindow) {
                tabRect = tab->rect;
                break;
            }
        }

        // Figure out the virtual vs real page size of the tabs
        float virtualSize = 0.0f;
        float tabRegionSize = m_tabRegion->rect.Width();
        if (!m_tabRegion->children.empty()) {
            virtualSize = m_tabRegion->children.back()->rect.Right();
        }

        const auto selectHeight = int(DPI_Y(tabSelectLine));

        // Clamp it
        m_tabOffsetX = std::min(m_tabOffsetX, 0.0f);

        NRectf currentTabRc;
        for (auto& tab : m_tabRegion->children) {
            auto spTabRegionTab = std::static_pointer_cast<TabRegionTab>(tab);
            currentTabRc = spTabRegionTab->rect;
            currentTabRc.Adjust(m_tabOffsetX, 0);

            if (spTabRegionTab->pTabWindow == GetActiveTabWindow()) {
                // Move back to ensure the tab is shown
                if (currentTabRc.Left() < m_tabRegion->rect.Left()) {
                    m_tabOffsetX += (m_tabRegion->rect.Left() - currentTabRc.Left());
                } else if (currentTabRc.Right() > m_tabRegion->rect.Right()) {
                    m_tabOffsetX -= (currentTabRc.Right() - m_tabRegion->rect.Right());
                }
            }
        }

        // If there is extra space and tabOffset is < 0, then adjust it.
        if (currentTabRc.Right() < m_tabRegion->rect.Right()) {
            m_tabOffsetX += m_tabRegion->rect.Right() - currentTabRc.Right();
        }

        m_tabOffsetX = std::min(m_tabOffsetX, 0.0f);

        // Now display the tabs
        for (auto& tab : m_tabRegion->children) {
            auto spTabRegionTab = std::static_pointer_cast<TabRegionTab>(tab);

            auto rc = spTabRegionTab->rect;
            rc.Adjust(m_tabOffsetX, 0);
            rc.SetHeight(rc.Height() - selectHeight);

            auto toneColor = spTabRegionTab->color;

            // I don't think tab window will ever be NULL!
            // Adding a check here for the time being
            std::string text;
            assert(spTabRegionTab->pTabWindow);

            if (GetEditor().GetConfig().tabToneColors) {
                if (spTabRegionTab->pTabWindow) {
                    text = spTabRegionTab->pTabWindow->GetName();
                    if (spTabRegionTab->pTabWindow->GetActiveWindow() && spTabRegionTab->pTabWindow->GetActiveWindow()->GetBuffer().GetToneColor().w != 0.0f) {
                        toneColor = spTabRegionTab->pTabWindow->GetActiveWindow()->GetBuffer().GetToneColor();
                    }
                }
            }

            if (spTabRegionTab->pTabWindow != GetActiveTabWindow()) {
                toneColor *= NVec4f(0.4f, 0.4f, 0.4f, 1.0f);
            }

            auto backColor = ModifyBackgroundColor(toneColor);

            auto& buffer = spTabRegionTab->pTabWindow->GetActiveWindow()->GetBuffer();
            if (buffer.HasFileFlags(FileFlags::HasWarnings) || buffer.HasFileFlags(FileFlags::HasErrors)) {
                NVec4f overrideColor;
                if (buffer.HasFileFlags(FileFlags::HasWarnings)) {
                    overrideColor = GetTheme().GetColor(ThemeColor::Warning);
                }

                // Errors win for coloring
                if (buffer.HasFileFlags(FileFlags::HasErrors)) {
                    overrideColor = GetTheme().GetColor(ThemeColor::Error);
                }

                m_pDisplay->DrawRectFilled(rc, backColor);

                auto rcError = rc;
                rcError.SetSize(NVec2f(uiFont.GetDefaultCharSize().x + textBorder, rc.Height()));

                m_pDisplay->DrawRectFilled(rcError, overrideColor);
            } else {
                m_pDisplay->DrawRectFilled(rc, backColor);
            }

            auto lum = Luminosity(backColor);
            auto textCol = NVec4f(1.0f);
            if (lum > .5f) {
                textCol.x = 0.0f;
                textCol.y = 0.0f;
                textCol.z = 0.0f;
            }

            // Tab background rect
            // Tab text
            m_pDisplay->DrawChars(uiFont, rc.topLeftPx + DPI_VEC2(NVec2f(textBorder, textBorder)), textCol, (const uint8_t*)text.c_str());

            auto drawTabLine = [&](auto yPos, auto col, auto height) {
                m_pDisplay->DrawRectFilled(NRectf(rc.Left(), yPos, rc.Width(), float(height / 2)), ModifyBackgroundColor(NVec4f(0.0f, 0.0f, 0.0f, 1.0f)));
                m_pDisplay->DrawRectFilled(NRectf(rc.Left(), yPos + height / 2, rc.Width(), float(height / 2)), ModifyBackgroundColor(col));
            };

            if (spTabRegionTab->pTabWindow == GetActiveTabWindow()) {
                drawTabLine(rc.Bottom(), GetTheme().GetColor(ThemeColor::TabActive), selectHeight);
            }
        }

        // Display the tab
        if (GetActiveTabWindow()) {
            GetActiveTabWindow()->Display();
        }
    } // namespace Zep

    ZepTheme& ZepEditor::GetTheme() const {
        return *m_spTheme;
    }

    bool ZepEditor::OnMouseMove(const NVec2f& mousePos) {
        m_mousePos = mousePos;
        bool handled = Broadcast(std::make_shared<ZepMessage>(Msg::MouseMove, mousePos));
        m_bPendingRefresh = true;
        return handled;
    }

    bool ZepEditor::OnMouseDown(const NVec2f& mousePos, ZepMouseButton button) {
        m_mousePos = mousePos;
        bool handled = Broadcast(std::make_shared<ZepMessage>(Msg::MouseDown, mousePos, button));
        m_bPendingRefresh = true;
        return handled;
    }

    bool ZepEditor::OnMouseUp(const NVec2f& mousePos, ZepMouseButton button) {
        m_mousePos = mousePos;
        bool handled = Broadcast(std::make_shared<ZepMessage>(Msg::MouseUp, mousePos, button));
        m_bPendingRefresh = true;
        return handled;
    }

    bool ZepEditor::OnMouseWheel(const NVec2f& mousePos, float scrollDistance) {
        m_mousePos = mousePos;
        bool handled = Broadcast(std::make_shared<ZepMessage>(Msg::MouseWheel, mousePos, std::to_string(scrollDistance)));
        m_bPendingRefresh = true;
        return handled;
    }

    const NVec2f ZepEditor::GetMousePos() const {
        return m_mousePos;
    }

    uint32_t ZepEditor::GetFlags() const {
        return m_flags;
    }

    void ZepEditor::SetFlags(uint32_t flags) {
        m_flags = flags;
        if (ZTestFlags(m_flags, ZepEditorFlags::FastUpdate)) {
            RequestRefresh();
        }
    }

    std::vector<const KeyMap*> ZepEditor::GetGlobalKeyMaps(ZepMode& mode) {
        std::vector<const KeyMap*> maps;
        for (auto& [id, spMap] : m_mapExCommands) {
            auto pMap = spMap->GetKeyMappings(mode);
            if (pMap) {
                maps.push_back(pMap);
            }
        }
        return maps;
    }

    ZepBuffer* ZepEditor::GetBufferFromHandle(uint64_t handle) {
        for (auto& buffer : m_buffers) {
            if (uint64_t(buffer.get()) == handle) {
                return buffer.get();
            }
        }
        return nullptr;
    }

    ZepWindow* ZepEditor::GetActiveWindow() const {
        auto pTabWindow = GetActiveTabWindow();
        if (pTabWindow) {
            return pTabWindow->GetActiveWindow();
        }
        return nullptr;
    }

    ZepBuffer* ZepEditor::GetActiveBuffer() const {
        auto pWindow = GetActiveWindow();
        if (pWindow) {
            // A window always has an associated buffer
            return &pWindow->GetBuffer();
        }
        return nullptr;
    }

    ZepWindow* ZepEditor::EnsureWindow(ZepBuffer& buffer) {
        auto windows = FindBufferWindows(&buffer);
        if (!windows.empty()) {
            return windows[0];
        }

        auto pTab = EnsureTab();
        return pTab->AddWindow(&buffer);
    }

    const fs::path& ZepEditor::GetConfigRoot() const {
        return m_configRoot;
    }

} // namespace Zep

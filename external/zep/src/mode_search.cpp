#include "zep/mode_search.h"
#include "zep/filesystem.h"
#include "zep/tab_window.h"
#include "zep/window.h"

#include "zep/mcommon/logger.h"
#include "zep/mcommon/threadutils.h"

#include "zep/mcommon/file/fnmatch.h"

namespace Zep {

    ZepMode_Search::ZepMode_Search(ZepEditor& editor, ZepWindow& launchWindow, ZepWindow& window, const fs::path& path)
        : ZepMode(editor), m_launchWindow(launchWindow), m_window(window), m_startPath(path) {
    }

    ZepMode_Search::~ZepMode_Search() {
        // Ensure threads have finished
        if (m_indexResult.valid()) {
            m_indexResult.wait();
        }

        if (m_searchResult.valid()) {
            m_searchResult.wait();
        }
    }

    void ZepMode_Search::AddKeyPress(uint32_t key, uint32_t modifiers) {
        (void)modifiers;
        if (key == ExtKeys::ESCAPE) {
            // CM TODO:
            // Note that the Repl represents the new way to do these commands; and this mode should be ported
            // to do the same thing.  It should also use the keymapper

            // We choose to rearrange the window and return to the previous order here.
            // If we just delete the buffer, it would have the same effect, but the editor
            // does not currently maintain a list of window orderings; so this is a second best for now.
            // TODO: Add window order tracking along with cursors for CTRL+i/o support
            auto& buffer = m_window.GetBuffer();
            GetEditor().GetActiveTabWindow()->RemoveWindow(&m_window);
            GetEditor().GetActiveTabWindow()->SetActiveWindow(&m_launchWindow);
            GetEditor().RemoveBuffer(&buffer);
            return;
        } else if (key == ExtKeys::RETURN) {
            OpenSelection(OpenType::Replace);
            return;
        } else if (key == ExtKeys::BACKSPACE) {
            if (m_searchTerm.length() > 0) {
                m_searchTerm = m_searchTerm.substr(0, m_searchTerm.length() - 1);
                UpdateTree();
            }
        } else {
            if (modifiers & ModifierKey::Ctrl) {
                if (key == 'j' || key == ExtKeys::DOWN) {
                    m_window.MoveCursorY(1);
                } else if (key == 'k' || key == ExtKeys::UP) {
                    m_window.MoveCursorY(-1);
                } else if (key == 'v') {
                    OpenSelection(OpenType::VSplit);
                    return;
                } else if (key == 'x') {
                    OpenSelection(OpenType::HSplit);
                    return;
                } else if (key == 't') {
                    OpenSelection(OpenType::Tab);
                    return;
                }
            } else if (key == ExtKeys::DOWN) {
                m_window.MoveCursorY(1);
            } else if (key == ExtKeys::UP) {
                m_window.MoveCursorY(-1);
            }
            // TODO: UTF8
            else if (std::isgraph(key)) {
                m_searchTerm += char(key);
                UpdateTree();
            }
        }

        std::ostringstream str;
        str << ">>> " << m_searchTerm;

        if (!m_indexTree.empty()) {
            str << " (" << m_indexTree[m_indexTree.size() - 1]->indices.size() << " / " << m_indexTree[0]->indices.size() << ")";
        }

        GetEditor().SetCommandText(str.str());
    }

    void ZepMode_Search::Begin(ZepWindow* pWindow) {
        ZepMode::Begin(pWindow);

        m_searchTerm = "";
        GetEditor().SetCommandText(">>> ");

        m_indexResult = Indexer::IndexPaths(GetEditor(), m_startPath);
        m_window.GetBuffer().SetText(std::string("Indexing: ") + m_startPath.string());

        fileSearchActive = true;
    }

    void ZepMode_Search::Notify(std::shared_ptr<ZepMessage> message) {
        ZepMode::Notify(message);
        if (message->messageId == Msg::Tick) {
            if (fileSearchActive) {
                if (!is_future_ready(m_indexResult)) {
                    return;
                }

                fileSearchActive = false;

                m_spFilePaths = m_indexResult.get();
                if (!m_spFilePaths->errors.empty()) {
                    GetEditor().SetCommandText(m_spFilePaths->errors);
                    return;
                }

                InitSearchTree();
                ShowTreeResult();
                UpdateTree();

                GetEditor().RequestRefresh();
            }

            if (treeSearchActive) {
                UpdateTree();
            }
        }
    }

    void ZepMode_Search::InitSearchTree() {
        m_indexTree.clear();
        auto pInitSet = std::make_shared<IndexSet>();
        for (uint32_t i = 0; i < (uint32_t)m_spFilePaths->paths.size(); i++) {
            pInitSet->indices.insert(std::make_pair(0, SearchResult{i, 0}));
        }
        m_indexTree.push_back(pInitSet);
    }

    void ZepMode_Search::ShowTreeResult() {
        std::ostringstream str;
        bool start = true;
        for (auto& index : m_indexTree.back()->indices) {
            if (!start) {
                str << std::endl;
            }
            str << m_spFilePaths->paths[index.second.index].string();
            start = false;
        }
        m_window.GetBuffer().SetText(str.str());
        m_window.SetBufferCursor(m_window.GetBuffer().Begin());
    }

    void ZepMode_Search::OpenSelection(OpenType type) {
        if (m_indexTree.empty())
            return;

        auto cursor = m_window.GetBufferCursor();
        auto line = m_window.GetBuffer().GetBufferLine(cursor);
        auto paths = m_indexTree[m_indexTree.size() - 1];

        auto& buffer = m_window.GetBuffer();

        GetEditor().GetActiveTabWindow()->SetActiveWindow(&m_launchWindow);

        long count = 0;
        for (auto& index : m_indexTree.back()->indices) {
            if (count == line) {
                auto path = m_spFilePaths->paths[index.second.index];
                auto full_path = m_spFilePaths->root / path;

                auto pBuffer = GetEditor().GetFileBuffer(full_path, 0, true);
                if (pBuffer != nullptr) {
                    switch (type) {
                    case OpenType::Replace: {
                        auto win = GetEditor().FindBufferWindows(pBuffer);
                        // If they just hit enter, then jump to existing if possible.
                        if (!win.empty()) {
                            GetEditor().SetCurrentTabWindow(&win[0]->GetTabWindow());
                            win[0]->GetTabWindow().SetActiveWindow(win[0]);
                        } else {
                            m_launchWindow.SetBuffer(pBuffer);
                        }
                    } break;
                    case OpenType::VSplit:
                        GetEditor().GetActiveTabWindow()->AddWindow(pBuffer, &m_launchWindow, RegionLayoutType::HBox);
                        break;
                    case OpenType::HSplit:
                        GetEditor().GetActiveTabWindow()->AddWindow(pBuffer, &m_launchWindow, RegionLayoutType::VBox);
                        break;
                    case OpenType::Tab:
                        GetEditor().AddTabWindow()->AddWindow(pBuffer, nullptr, RegionLayoutType::HBox);
                        break;
                    }
                }
            }
            count++;
        }

        // Removing the buffer will also kill this mode and its window; this is the last thing we can do here
        GetEditor().RemoveBuffer(&buffer);
    }

    void ZepMode_Search::UpdateTree() {
        if (fileSearchActive) {
            return;
        }

        if (treeSearchActive) {
            if (!is_future_ready(m_searchResult)) {
                return;
            }

            m_indexTree.push_back(m_searchResult.get());
            treeSearchActive = false;
        }

        // If the user is typing capitals, he cares about them in the search!
        m_caseImportant = string_tolower(m_searchTerm) != m_searchTerm;

        assert(!m_indexTree.empty());

        uint32_t treeDepth = uint32_t(m_indexTree.size() - 1);
        if (m_searchTerm.size() < treeDepth) {
            while (m_searchTerm.size() < treeDepth) {
                m_indexTree.pop_back();
                treeDepth--;
            };
        } else if (m_searchTerm.size() > treeDepth) {
            std::shared_ptr<IndexSet> spStartSet;
            spStartSet = m_indexTree[m_indexTree.size() - 1];
            char startChar = m_searchTerm[m_indexTree.size() - 1];

            // Search for a match at the next level of the search tree
            m_searchResult = GetEditor().GetThreadPool().enqueue([&](std::shared_ptr<IndexSet> spStartSet, const char startChar) {
                auto spResult = std::make_shared<IndexSet>();
                for (auto& searchPair : spStartSet->indices) {
                    auto index = searchPair.second.index;
                    auto loc = searchPair.second.location;
                    auto dist = searchPair.first;

                    size_t pos = 0;
                    if (m_caseImportant) {
                        auto str = m_spFilePaths->paths[index].string();
                        pos = str.find_first_of(startChar, loc);
                    } else {
                        auto str = m_spFilePaths->lowerPaths[index];
                        pos = str.find_first_of(startChar, loc);
                    }

                    if (pos != std::string::npos) {
                        // this approach 'clumps things together'
                        // It rewards more for strings of subsequent characters
                        uint32_t newDist = ((uint32_t)pos - loc);
                        if (dist == 0) {
                            newDist = 1;
                        } else if (newDist == 1) {
                            newDist = dist;
                        } else {
                            newDist = dist + 1;
                        }

                        spResult->indices.insert(std::make_pair(newDist, SearchResult{index, (uint32_t)pos}));
                    }
                }
                return spResult;
            },
                                                                 spStartSet, startChar);

            treeSearchActive = true;
        }

        ShowTreeResult();
        GetEditor().RequestRefresh();
    }

    CursorType ZepMode_Search::GetCursorType() const {
        return CursorType::LineMarker;
    }

} // namespace Zep

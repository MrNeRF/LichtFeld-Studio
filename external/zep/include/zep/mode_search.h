#pragma once

#include "mode.h"
#include <future>
#include <memory>
#include <regex>

#include <zep/indexer.h>

namespace Zep {

    class ZepWindow;
    class ZepMode_Search : public ZepMode {
    public:
        ZepMode_Search(ZepEditor& editor, ZepWindow& previousWindow, ZepWindow& window, const fs::path& startPath);
        ~ZepMode_Search();

        virtual void AddKeyPress(uint32_t key, uint32_t modifiers = 0) override;
        virtual void Begin(ZepWindow* pWindow) override;
        virtual void Notify(std::shared_ptr<ZepMessage> message) override;
        virtual EditorMode DefaultMode() const override {
            return EditorMode::Normal;
        }

        static const char* StaticName() {
            return "Search";
        }
        virtual const char* Name() const override {
            return StaticName();
        }

        virtual CursorType GetCursorType() const override;

    private:
        void GetSearchPaths(const fs::path& path, std::vector<std::string>& ignore, std::vector<std::string>& include) const;
        void InitSearchTree();
        void ShowTreeResult();
        void UpdateTree();

        enum class OpenType {
            Replace,
            VSplit,
            HSplit,
            Tab
        };
        void OpenSelection(OpenType type);

    private:
        // List of lines in the file result, with last found char
        struct SearchResult {
            uint32_t index = 0;
            uint32_t location = 0;
        };

        // A mapping from character distance to a list of lines
        struct IndexSet {
            std::multimap<uint32_t, SearchResult> indices;
        };

        bool fileSearchActive = false;
        bool treeSearchActive = false;

        // Results of the file search and the indexing threads
        std::future<std::shared_ptr<FileIndexResult>> m_indexResult;
        std::future<std::shared_ptr<IndexSet>> m_searchResult;

        // All files that can potentially match
        std::shared_ptr<FileIndexResult> m_spFilePaths;

        // A hierarchy of index results.
        // The 'top' of the tree is the most narrow finding from a set of 'n' characters
        // index a,b,c -> index b,c -> index c
        std::vector<std::shared_ptr<IndexSet>> m_indexTree;

        // What we are searching for
        std::string m_searchTerm;
        bool m_caseImportant = false;

        ZepWindow& m_launchWindow;
        ZepWindow& m_window;
        fs::path m_startPath;
    };

} // namespace Zep

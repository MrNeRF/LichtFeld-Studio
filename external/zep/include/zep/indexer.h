#pragma once

#include <future>
#include <memory>
#include <regex>
#include <thread>

#include "zep/editor.h"

namespace Zep {

    class ZepEditor;

    // List of files found in the directory search
    struct FileIndexResult {
        fs::path root;
        std::vector<fs::path> paths;
        std::vector<std::string> lowerPaths;
        std::string errors;
    };

    struct SymbolDetails {
        int line = 0;
        int column = 0;
    };

    using SymbolContainer = std::map<std::string, SymbolDetails>;

    class Indexer : public ZepComponent {
    public:
        Indexer(ZepEditor& editor);

        virtual void Notify(std::shared_ptr<ZepMessage> message) override;

        bool StartIndexing();
        void StartSymbolSearch();

        static void GetSearchPaths(ZepEditor& editor, const fs::path& path, std::vector<std::string>& ignore_patterns, std::vector<std::string>& include_patterns, std::string& errors);
        static std::future<std::shared_ptr<FileIndexResult>> IndexPaths(ZepEditor& editor, const fs::path& startPath);

    private:
        bool m_fileSearchActive = false;
        std::future<std::shared_ptr<FileIndexResult>> m_indexResult;
        std::shared_ptr<FileIndexResult> m_spFilePaths;

        std::mutex m_queueMutex;
        std::deque<fs::path> m_searchQueue;

        std::mutex m_symbolMutex;
        SymbolContainer m_symbols;
        fs::path m_searchRoot;
    };

} // namespace Zep

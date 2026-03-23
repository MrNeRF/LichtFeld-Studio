#include "zep/mcommon/file/fnmatch.h"
#include "zep/mcommon/file/path.h"
#include "zep/mcommon/logger.h"
#include "zep/mcommon/string/stringutils.h"
#include "zep/mcommon/threadutils.h"

#include "zep/filesystem.h"
#include "zep/indexer.h"

namespace Zep {

    enum TypeName {
        t_class,
        t_void,
        t_byte,
        t_char,
        t_int,
        t_long,
        t_float,
        t_double,
        t_uint32_t,
        t_uint8_t,
        t_uint64_t,
        t_int32_t,
        t_int64_t,
        t_int8_t
    };

    std::map<std::string, TypeName> MapToType = {
        {"class", t_class},
        {"void", t_void},
        {"byte", t_byte},
        {"char", t_char},
        {"int", t_int},
        {"long", t_long},
        {"float", t_float},
        {"double", t_double},
        {"uint32_t", t_uint32_t},
        {"uint8_t", t_uint8_t},
        {"uint64_t", t_uint64_t},
        {"int32_t", t_int32_t},
        {"int64_t", t_int64_t},
        {"int8_t", t_int8_t}};

    Indexer::Indexer(ZepEditor& editor)
        : ZepComponent(editor) {
    }

    void Indexer::GetSearchPaths(ZepEditor& editor, const fs::path& path, std::vector<std::string>& ignore_patterns, std::vector<std::string>& include_patterns, std::string& errors) {
        fs::path config = path / ".zep" / "project.cfg";
        if (!editor.GetFileSystem().Exists(config)) {
            config = editor.GetConfigRoot() / "zep.cfg";
        }

        if (editor.GetFileSystem().Exists(config)) {
            try {
                auto spConfig = cpptoml::parse_file(config.string());
                if (spConfig != nullptr) {
                    ignore_patterns = spConfig->get_qualified_array_of<std::string>("search.ignore").value_or(std::vector<std::string>{});
                    include_patterns = spConfig->get_qualified_array_of<std::string>("search.include").value_or(std::vector<std::string>{});
                }
            } catch (cpptoml::parse_exception& ex) {
                std::ostringstream str;
                str << config.filename().string() << " : Failed to parse. " << ex.what();
                errors = str.str();
            } catch (...) {
                std::ostringstream str;
                str << config.filename().string() << " : Failed to parse. ";
                errors = str.str();
            }
        }

        if (ignore_patterns.empty()) {
            ignore_patterns = {
                "[Bb]uild/*",
                "**/[Oo]bj/**",
                "**/[Bb]in/**",
                "[Bb]uilt*"};
        }
        if (include_patterns.empty()) {
            include_patterns = {
                "*.cpp",
                "*.c",
                "*.hpp",
                "*.h",
                "*.lsp",
                "*.scm",
                "*.cs",
                "*.cfg",
                "*.orca"};
        }
    } // namespace Zep

    std::future<std::shared_ptr<FileIndexResult>> Indexer::IndexPaths(ZepEditor& editor, const fs::path& startPath) {
        std::vector<std::string> ignorePaths;
        std::vector<std::string> includePaths;
        std::string errors;
        GetSearchPaths(editor, startPath, ignorePaths, includePaths, errors);

        auto spResult = std::make_shared<FileIndexResult>();
        if (!errors.empty()) {
            spResult->errors = errors;
            return make_ready_future(spResult);
        }

        auto pFileSystem = &editor.GetFileSystem();
        return editor.GetThreadPool().enqueue([=](fs::path root) {
            spResult->root = root;

            try {
                // Index the whole subtree, ignoring any patterns supplied to us
                pFileSystem->ScanDirectory(root, [&](const fs::path& p, bool& recurse) -> bool {
                    recurse = true;

                    auto bDir = pFileSystem->IsDirectory(p);

                    // Add this one to our list
                    auto targetZep = pFileSystem->Canonical(p);
                    auto rel = path_get_relative(root, targetZep);

                    bool matched = true;
                    for (auto& proj : ignorePaths) {
                        auto res = fnmatch(proj.c_str(), rel.string().c_str(), 0);
                        if (res == 0) {
                            matched = false;
                            break;
                        }
                    }

                    if (!matched) {
                        if (bDir) {
                            recurse = false;
                        }
                        return true;
                    }

                    matched = false;
                    for (auto& proj : includePaths) {
                        auto res = fnmatch(proj.c_str(), rel.string().c_str(), 0);
                        if (res == 0) {
                            matched = true;
                            break;
                        }
                    }

                    if (!matched) {
                        return true;
                    }

                    // Not adding directories to the search list
                    if (bDir) {
                        return true;
                    }

                    spResult->paths.push_back(rel);
                    spResult->lowerPaths.push_back(string_tolower(rel.string()));

                    return true;
                });
            } catch (std::exception&) {
            }
            return spResult;
        },
                                              startPath);
    }

    void Indexer::Notify(std::shared_ptr<ZepMessage> message) {
        if (message->messageId == Msg::Tick) {
            if (m_fileSearchActive) {
                if (!is_future_ready(m_indexResult)) {
                    return;
                }

                m_fileSearchActive = false;

                m_spFilePaths = m_indexResult.get();
                if (!m_spFilePaths->errors.empty()) {
                    GetEditor().SetCommandText(m_spFilePaths->errors);
                    return;
                }

                // Queue the files to be searched
                {
                    std::lock_guard<std::mutex> guard(m_queueMutex);
                    for (auto& p : m_spFilePaths->paths) {
                        m_searchQueue.push_back(p);
                    }
                }

                // Kick off the thread
                StartSymbolSearch();
            }
        }
    }

    void Indexer::StartSymbolSearch() {
        GetEditor().GetThreadPool().enqueue([=]() {
            for (;;) {
                fs::path path;
                {
                    std::lock_guard<std::mutex> lock(m_queueMutex);
                    if (m_searchQueue.empty()) {
                        return true;
                    }

                    path = m_searchQueue.front();
                    m_searchQueue.pop_front();
                }

                auto& fs = GetEditor().GetFileSystem();

                std::string strLast;

                auto fullPath = m_searchRoot / path;
                if (fs.Exists(fullPath)) {
                    ZLOG(DBG, "Parsing: " << fullPath.string());
                    auto strFile = GetEditor().GetFileSystem().Read(fullPath);

                    std::vector<std::string> tokens;
                    string_split(strFile, ";()[] \t\n\r&!\"\'*:,<>", tokens);
                }
            }
        });
    }

    bool Indexer::StartIndexing() {
        bool foundGit = false;
        m_searchRoot = GetEditor().GetFileSystem().GetSearchRoot(GetEditor().GetFileSystem().GetWorkingDirectory(), foundGit);
        if (!foundGit) {
            ZLOG(INFO, "Not a git project");
            return false;
        }

        auto& fs = GetEditor().GetFileSystem();

        auto indexDBRoot = m_searchRoot / ".zep";
        if (!fs.IsDirectory(indexDBRoot)) {
            if (!fs.MakeDirectories(indexDBRoot)) {
                ZLOG(ERROR, "Can't get the index folder");
                return false;
            }
        }

        int v = 0;
        fs.Write(indexDBRoot / "indexdb", &v, 1);

        m_fileSearchActive = true;
        m_indexResult = Indexer::IndexPaths(GetEditor(), m_searchRoot);

        return true;
    }

} // namespace Zep

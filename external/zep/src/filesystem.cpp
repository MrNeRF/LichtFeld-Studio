#include "zep/filesystem.h"
#include "zep/editor.h" // FOR ZEP_UNUSED

#include <fstream>

#include "zep/mcommon/logger.h"
#include "zep/mcommon/string/stringutils.h"

#undef ERROR

#if defined(ZEP_FEATURE_CPP_FILE_SYSTEM)

#include <filesystem>

#ifdef __APPLE__
namespace cpp_fs = std::__fs::filesystem;
#else
namespace cpp_fs = std::filesystem;
#endif

namespace Zep {
    ZepFileSystemCPP::ZepFileSystemCPP(const fs::path& configPath) {
        // Use the config path
        m_configPath = configPath;

        m_workingDirectory = fs::path(cpp_fs::current_path().string());

        // Didn't find the config path, try the working directory
        if (!Exists(m_configPath)) {
            m_configPath = m_workingDirectory;
        }

        ZLOG(INFO, "Config Dir: " << m_configPath.string());
        ZLOG(INFO, "Working Dir: " << m_workingDirectory.string());
    }

    ZepFileSystemCPP::~ZepFileSystemCPP() {
    }

    void ZepFileSystemCPP::SetWorkingDirectory(const fs::path& path) {
        if (!IsDirectory(path)) {
            return;
        }
        // Set the file system's current working directory too.
        cpp_fs::current_path(path.string());
        m_workingDirectory = path;
    }

    const fs::path& ZepFileSystemCPP::GetWorkingDirectory() const {
        return m_workingDirectory;
    }

    fs::path ZepFileSystemCPP::GetConfigPath() const {
        return m_configPath;
    }

    bool ZepFileSystemCPP::MakeDirectories(const fs::path& path) {
        return cpp_fs::create_directories(path.c_str());
    }

    bool ZepFileSystemCPP::IsDirectory(const fs::path& path) const {
        if (!Exists(path)) {
            return false;
        }
        return cpp_fs::is_directory(path.string());
    }

    bool ZepFileSystemCPP::IsReadOnly(const fs::path& path) const {
        auto perms = cpp_fs::status(path.string()).permissions();
        if ((perms & cpp_fs::perms::owner_write) == cpp_fs::perms::owner_write) {
            return false;
        }
        return true;
    }

    std::string ZepFileSystemCPP::Read(const fs::path& fileName) {
        std::ifstream in(fileName, std::ios::in | std::ios::binary);
        if (in) {
            std::string contents;
            in.seekg(0, std::ios::end);
            contents.resize(size_t(in.tellg()));
            in.seekg(0, std::ios::beg);
            in.read(&contents[0], contents.size());
            in.close();
            return (contents);
        } else {
            ZLOG(ERROR, "File Not Found: " << fileName.string());
        }
        return std::string();
    }

    bool ZepFileSystemCPP::Write(const fs::path& fileName, const void* pData, size_t size) {
        FILE* pFile;
        pFile = fopen(fileName.string().c_str(), "wb");
        if (!pFile) {
            return false;
        }
        // Valid to open/close with size == 0, which will truncate
        if (size != 0) {
            fwrite(pData, sizeof(uint8_t), size, pFile);
        }
        fclose(pFile);
        return true;
    }

    void ZepFileSystemCPP::ScanDirectory(const fs::path& path, std::function<bool(const fs::path& path, bool& dont_recurse)> fnScan) const {
        for (auto itr = cpp_fs::recursive_directory_iterator(path.string());
             itr != cpp_fs::recursive_directory_iterator();
             itr++) {
            auto p = fs::path(itr->path().string());

            bool recurse = true;
            if (!fnScan(p, recurse))
                return;

            if (!recurse && itr.recursion_pending()) {
                itr.disable_recursion_pending();
            }
        }
    }

    bool ZepFileSystemCPP::Exists(const fs::path& path) const {
        try {
            return cpp_fs::exists(path.string());
        } catch (cpp_fs::filesystem_error& err) {
            ZEP_UNUSED(err);
            ZLOG(ERROR, "Exception: " << err.what());
            return false;
        }
    }

    bool ZepFileSystemCPP::Equivalent(const fs::path& path1, const fs::path& path2) const {
        try {
            // The below API expects existing files!  Best we can do is direct compare of paths
            if (!cpp_fs::exists(path1.string()) || !cpp_fs::exists(path2.string())) {
                return Canonical(path1).string() == Canonical(path2).string();
            }
            return cpp_fs::equivalent(path1.string(), path2.string());
        } catch (cpp_fs::filesystem_error& err) {
            ZEP_UNUSED(err);
            ZLOG(ERROR, "Exception: " << err.what());
            return path1 == path2;
        }
    }

    fs::path ZepFileSystemCPP::Canonical(const fs::path& path) const {
        try {
#ifdef __unix__
            // TODO: Remove when unix doesn't need <experimental/filesystem>
            // I can't remember why weakly_connical is used....
            return fs::path(cpp_fs::canonical(path.string()).string());
#else
            return fs::path(cpp_fs::weakly_canonical(path.string()).string());
#endif
        } catch (cpp_fs::filesystem_error& err) {
            ZEP_UNUSED(err);
            ZLOG(ERROR, "Exception: " << err.what());
            return path;
        }
    }

    fs::path ZepFileSystemCPP::GetSearchRoot(const fs::path& start, bool& foundGit) const {
        foundGit = false;
        auto findStartPath = [&](const fs::path& startPath) {
            if (!startPath.empty()) {
                auto testPath = startPath;
                if (!IsDirectory(testPath)) {
                    testPath = testPath.parent_path();
                }

                if (!(m_flags & ZepFileSystemFlags::SearchGitRoot)) {
                    return testPath;
                }

                while (!testPath.empty() && IsDirectory(testPath)) {
                    foundGit = false;

                    // Look in this dir
                    ScanDirectory(testPath, [&](const fs::path& p, bool& recurse) -> bool {
                        // Not looking at sub folders
                        recurse = false;

                        // Found the .git repo
                        if (p.extension() == ".git" && IsDirectory(p)) {
                            foundGit = true;

                            // Quit search
                            return false;
                        }
                        return true;
                    });

                    // If found,  return it as the path we need
                    if (foundGit) {
                        return testPath;
                    }

                    if (!testPath.has_relative_path()) {
                        break;
                    }
                    testPath = testPath.parent_path();
                }

                // Didn't find a sensible search root, so start at the parent folder of the start path if it is a file
                if (!IsDirectory(startPath)) {
                    if (IsDirectory(startPath.parent_path())) {
                        return startPath.parent_path();
                    }
                }
            }
            return startPath;
        };

        fs::path workingDir = GetWorkingDirectory();

        auto startPath = findStartPath(start);
        if (startPath.empty()) {
            startPath = findStartPath(workingDir);
            if (startPath.empty()) {
                startPath = GetWorkingDirectory();
            }
        }

        // Failure case, just use current path
        if (startPath.empty()) {
            startPath = start;
        }
        return startPath;
    }

    void ZepFileSystemCPP::SetFlags(uint32_t flags) {
        m_flags = flags;
    }

} // namespace Zep

#endif // CPP_FILESYSTEM

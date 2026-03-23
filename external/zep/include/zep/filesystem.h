#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "zep_config.h"

namespace Zep {

    namespace fs = std::filesystem;

    enum ZepFileSystemFlags {
        SearchGitRoot = (1 << 0)
    };

    // Zep's view of the outside world in terms of files
    // Below there is a version of this that will work on most platforms using std's <filesystem> for file operations
    // If you want to expose your app's view of the world, you need to implement this minimal set of functions
    class IZepFileSystem {
    public:
        virtual ~IZepFileSystem(){};
        virtual std::string Read(const fs::path& filePath) = 0;
        virtual bool Write(const fs::path& filePath, const void* pData, size_t size) = 0;

        // This is the application config path, where the executable configuration files live
        // (and most likely the .exe too).
        virtual fs::path GetConfigPath() const = 0;

        // The rootpath is either the git working directory or the app current working directory
        virtual fs::path GetSearchRoot(const fs::path& start, bool& foundGit) const = 0;

        // The working directory is typically the root of the current project that is being edited;
        // i.e. it is set to the path of the first thing that is passed to zep, or is the zep startup folder
        virtual const fs::path& GetWorkingDirectory() const = 0;
        virtual void SetWorkingDirectory(const fs::path& path) = 0;
        virtual bool MakeDirectories(const fs::path& path) = 0;

        virtual bool IsDirectory(const fs::path& path) const = 0;
        virtual bool IsReadOnly(const fs::path& path) const = 0;
        virtual bool Exists(const fs::path& path) const = 0;

        // A callback API for scaning
        virtual void ScanDirectory(const fs::path& path, std::function<bool(const fs::path& path, bool& dont_recurse)> fnScan) const = 0;

        // Equivalent means 'the same file'
        virtual bool Equivalent(const fs::path& path1, const fs::path& path2) const = 0;
        virtual fs::path Canonical(const fs::path& path) const = 0;
        virtual void SetFlags(uint32_t flags) = 0;
    };

// CPP File system - part of the standard C++ libraries
#if defined(ZEP_FEATURE_CPP_FILE_SYSTEM)

    // A generic file system using cross platform fs:: and tinydir for searches
    // This is typically the only one that is used for normal desktop usage.
    // But you could make your own if your files were stored in a compressed folder, or the target system didn't have a traditional file system...
    class ZepFileSystemCPP : public IZepFileSystem {
    public:
        ZepFileSystemCPP(const fs::path& configPath);
        ~ZepFileSystemCPP();
        virtual std::string Read(const fs::path& filePath) override;
        virtual bool Write(const fs::path& filePath, const void* pData, size_t size) override;
        virtual void ScanDirectory(const fs::path& path, std::function<bool(const fs::path& path, bool& dont_recurse)> fnScan) const override;
        virtual void SetWorkingDirectory(const fs::path& path) override;
        virtual bool MakeDirectories(const fs::path& path) override;
        virtual const fs::path& GetWorkingDirectory() const override;
        virtual fs::path GetConfigPath() const override;
        virtual fs::path GetSearchRoot(const fs::path& start, bool& foundGit) const override;
        virtual bool IsDirectory(const fs::path& path) const override;
        virtual bool IsReadOnly(const fs::path& path) const override;
        virtual bool Exists(const fs::path& path) const override;
        virtual bool Equivalent(const fs::path& path1, const fs::path& path2) const override;
        virtual fs::path Canonical(const fs::path& path) const override;
        virtual void SetFlags(uint32_t flags) override;

    private:
        fs::path m_workingDirectory;
        fs::path m_configPath;
        uint32_t m_flags = ZepFileSystemFlags::SearchGitRoot;
    };
#endif // CPP File system

} // namespace Zep

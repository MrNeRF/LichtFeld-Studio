#pragma once
#include <filesystem>
namespace lfs::training {
    inline void absolutize_dataset_path_for_snapshot(
        std::filesystem::path& path) {
        if (path.empty() || path.is_absolute())
            return;
        std::error_code error;
        auto absolute = std::filesystem::absolute(path, error);
        if (error)
            return;
        path = absolute.lexically_normal();
    }

} // namespace lfs::training

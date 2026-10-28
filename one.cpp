#include "core/native_fs.hpp"
#include <cstdio>
#include <fstream>
#include <string>

namespace fs = lfs::core::native_fs;
using std::filesystem::path;

// Same steps as NativeFilesystemTest.UncOperationsWorkWhenTheLocalShareIsAvailable: deep directories, ofstream, file_size, rename.
int wmain(int argc, wchar_t** argv) {
    if (argc < 3)
        return 2;
    const int depth = std::stoi(argv[2]);
    auto dir = path(argv[1]);
    for (int i = 0; i < depth; ++i)
        dir /= path(u8"unicode_\u6E2C\u8A66_\u00E0_component_01234567890123456789");
    std::error_code e;
    fs::create_directories(dir, e);
    const auto file = dir / "network.bin", renamed = dir / "renamed.bin";
    {
        std::ofstream out(fs::native_path(file), std::ios::binary);
        out << "network";
    }
    std::error_code size_error;
    (void)fs::file_size(file, size_error);
    std::error_code rename_error;
    fs::rename(file, renamed, rename_error);
    std::printf("depth=%d len=%zu mkdir=%d size=%d rename=%d\n", depth, fs::native_path(file, true).native().size(), e.value(),
                size_error.value(), rename_error.value());
    return rename_error.value();
}

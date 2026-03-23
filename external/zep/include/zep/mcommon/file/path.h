#pragma once

#include "zep/mcommon/string/stringutils.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

namespace Zep {

    namespace fs = std::filesystem;
    fs::path path_get_relative(const fs::path& from, const fs::path& to);

} // namespace Zep

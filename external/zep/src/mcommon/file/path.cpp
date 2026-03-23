#include "zep/mcommon/file/path.h"

namespace Zep {

    // http://stackoverflow.com/a/29221546/18942
    fs::path path_get_relative(const fs::path& from, const fs::path& to) {
        // Start at the root path and while they are the same then do nothing then when they first
        // diverge take the remainder of the two path and replace the entire from path with ".."
        // segments.
        fs::path::const_iterator fromIter = from.begin();
        fs::path::const_iterator toIter = to.begin();

        // Loop through both
        while (fromIter != from.end() && toIter != to.end() && (*toIter) == (*fromIter)) {
            ++toIter;
            ++fromIter;
        }

        fs::path finalPath;
        while (fromIter != from.end()) {
            finalPath = finalPath / "..";
            ++fromIter;
        }

        while (toIter != to.end()) {
            finalPath = finalPath / *toIter;
            ++toIter;
        }

        return finalPath;
    }

} // namespace Zep

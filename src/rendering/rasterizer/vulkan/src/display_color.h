#pragma once
#include <cstdint>
#include <glm/glm.hpp>

namespace lfs::rendering {
#define LFS_COLOR_INLINE inline
#define LFS_COLOR_VEC3   glm::vec3
#define LFS_COLOR_UINT   std::uint32_t
#define LFS_COLOR_MIX    glm::mix
    using glm::clamp;
    using glm::dot;
    using glm::max;
    using glm::min;
    using glm::pow;
#include "../shader/src/slang/display_color.inc"
#undef LFS_COLOR_INLINE
#undef LFS_COLOR_VEC3
#undef LFS_COLOR_UINT
#undef LFS_COLOR_MIX
} // namespace lfs::rendering

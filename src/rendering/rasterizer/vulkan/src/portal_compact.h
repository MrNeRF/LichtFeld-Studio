#pragma once
#include <glm/glm.hpp>

namespace lfs::rendering {
#define LFS_PORTAL_INLINE inline
#define LFS_PORTAL_VEC3   glm::vec3
#define LFS_PORTAL_VEC4   glm::vec4
    using glm::clamp;
    using glm::dot;
    using glm::exp;
    using glm::floor;
    using glm::log;
    using glm::max;
    using glm::sqrt;
#include "../shader/src/slang/portal_compact.inc"
#undef LFS_PORTAL_INLINE
#undef LFS_PORTAL_VEC3
#undef LFS_PORTAL_VEC4
} // namespace lfs::rendering

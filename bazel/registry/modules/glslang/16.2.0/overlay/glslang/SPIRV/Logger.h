// Forwarding header added by the Bazel overlay of glslang.
// The CMake install tree ships SPIRV/Logger.h as <glslang/SPIRV/Logger.h>;
// forwarding (instead of copying) keeps a single physical header, so
// <SPIRV/Logger.h> and <glslang/SPIRV/Logger.h> can be mixed safely.
#include "../../SPIRV/Logger.h"

// Forwarding header added by the Bazel overlay of glslang.
// The CMake install tree ships SPIRV/SpvTools.h as <glslang/SPIRV/SpvTools.h>;
// forwarding (instead of copying) keeps a single physical header, so
// <SPIRV/SpvTools.h> and <glslang/SPIRV/SpvTools.h> can be mixed safely.
#include "../../SPIRV/SpvTools.h"

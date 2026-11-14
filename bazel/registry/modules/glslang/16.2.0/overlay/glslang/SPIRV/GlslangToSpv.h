// Forwarding header added by the Bazel overlay of glslang.
// The CMake install tree ships SPIRV/GlslangToSpv.h as <glslang/SPIRV/GlslangToSpv.h>;
// forwarding (instead of copying) keeps a single physical header, so
// <SPIRV/GlslangToSpv.h> and <glslang/SPIRV/GlslangToSpv.h> can be mixed safely.
#include "../../SPIRV/GlslangToSpv.h"

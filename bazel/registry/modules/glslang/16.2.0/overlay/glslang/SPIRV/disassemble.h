// Forwarding header added by the Bazel overlay of glslang.
// The CMake install tree ships SPIRV/disassemble.h as <glslang/SPIRV/disassemble.h>;
// forwarding (instead of copying) keeps a single physical header, so
// <SPIRV/disassemble.h> and <glslang/SPIRV/disassemble.h> can be mixed safely.
#include "../../SPIRV/disassemble.h"

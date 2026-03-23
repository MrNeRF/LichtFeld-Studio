#include "zep/buffer.h"
#include "zep/editor.h"
#include "zep/syntax.h"
#include "zep/syntax_markdown.h"
#include "zep/syntax_python.h"
#include "zep/syntax_tree.h"

namespace Zep {

    // Most of these keyword values came from earlier editor keyword lists.
    // I'll fill these out when I get time.  At the moment the syntax code matches on these, comments and numbers.
    static std::unordered_set<std::string> cpp_keywords = {
        "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch", "char", "char16_t", "char32_t", "class",
        "compl", "concept", "const", "constexpr", "const_cast", "continue", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float",
        "for", "friend", "goto", "if", "import", "inline", "int", "long", "module", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected", "public",
        "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch", "synchronized", "template", "this", "thread_local",
        "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", "#define", "#include",
        "uint32_t", "int32_t", "uint64_t", "int64_t", "size_t", "uint8_t", "int8_t", "int16_t", "uint16_t"};

    static std::unordered_set<std::string> cpp_identifiers = {
        "abort", "abs", "acos", "asin", "atan", "atexit", "atof", "atoi", "atol", "ceil", "clock", "cosh", "ctime", "div", "exit", "fabs", "floor", "fmod", "getchar", "getenv", "isalnum", "isalpha", "isdigit", "isgraph",
        "ispunct", "isspace", "isupper", "kbhit", "log10", "log2", "log", "memcmp", "modf", "pow", "printf", "sprintf", "snprintf", "putchar", "putenv", "puts", "rand", "remove", "rename", "sinh", "sqrt", "srand", "strcat", "strcmp", "strerror", "time", "tolower", "toupper",
        "std", "string", "vector", "map", "unordered_map", "set", "unordered_set", "min", "max"};

    static std::unordered_set<std::string> toml_keywords = {};
    static std::unordered_set<std::string> toml_identifiers = {};

    static std::unordered_set<std::string> hlsl_keywords = {
        "CompileShader", "const", "continue", "ComputeShader", "ConsumeStructuredBuffer", "default", "DepthStencilState", "DepthStencilView", "discard", "do", "double", "DomainShader", "dword", "else", "export", "extern",
        "false", "float", "for", "fxgroup", "GeometryShader", "groupshared", "half", "Hullshader", "if", "in", "inline", "inout", "InputPatch", "int", "interface", "line", "lineadj", "linear", "LineStream", "matrix", "min16float",
        "min10float", "min16int", "min12int", "min16uint", "namespace", "nointerpolation", "noperspective", "NULL", "out", "OutputPatch", "packoffset", "pass", "pixelfragment", "PixelShader", "point", "PointStream", "precise",
        "RasterizerState", "RenderTargetView", "return", "register", "row_major", "RWBuffer", "RWByteAddressBuffer", "RWStructuredBuffer", "RWTexture1D", "RWTexture1DArray", "RWTexture2D", "RWTexture2DArray", "RWTexture3D",
        "sample", "sampler", "SamplerState", "SamplerComparisonState", "shared", "snorm", "stateblock", "stateblock_state", "static", "string", "struct", "switch", "StructuredBuffer", "tbuffer", "technique", "technique10",
        "technique11", "texture", "Texture1D", "Texture1DArray", "Texture2D", "Texture2DArray", "Texture2DMS", "Texture2DMSArray", "Texture3D", "TextureCube", "TextureCubeArray", "true", "typedef", "triangle", "triangleadj",
        "TriangleStream", "uint", "uniform", "unorm", "unsigned", "vector", "vertexfragment", "VertexShader", "void", "volatile", "while", "bool1", "bool2", "bool3", "bool4", "double1", "double2", "double3", "double4", "float1",
        "float2", "float3", "float4", "int1", "int2", "int3", "int4", "in", "out", "inout", "uint1", "uint2", "uint3", "uint4", "dword1", "dword2", "dword3", "dword4", "half1", "half2", "half3", "half4", "float1x1", "float2x1",
        "float3x1", "float4x1", "float1x2", "float2x2", "float3x2", "float4x2", "float1x3", "float2x3", "float3x3", "float4x3", "float1x4", "float2x4", "float3x4", "float4x4", "half1x1", "half2x1", "half3x1", "half4x1", "half1x2",
        "half2x2", "half3x2", "half4x2", "half1x3", "half2x3", "half3x3", "half4x3", "half1x4", "half2x4", "half3x4", "half4x4"};

    static std::unordered_set<std::string> hlsl_identifiers = {
        "abort", "abs", "acos", "all", "AllMemoryBarrier", "AllMemoryBarrierWithGroupSync", "any", "asdouble", "asfloat", "asin", "asint", "asint", "asuint", "asuint", "atan", "atan2", "ceil", "CheckAccessFullyMapped", "clamp",
        "clip", "cos", "cosh", "countbits", "cross", "D3DCOLORtoUBYTE4", "ddx", "ddx_coarse", "ddx_fine", "ddy", "ddy_coarse", "ddy_fine", "degrees", "determinant", "DeviceMemoryBarrier", "DeviceMemoryBarrierWithGroupSync",
        "distance", "dot", "dst", "errorf", "EvaluateAttributeAtCentroid", "EvaluateAttributeAtSample", "EvaluateAttributeSnapped", "exp", "exp2", "f16tof32", "f32tof16", "faceforward", "firstbithigh", "firstbitlow", "floor",
        "fma", "fmod", "frac", "frexp", "fwidth", "GetRenderTargetSampleCount", "GetRenderTargetSamplePosition", "GroupMemoryBarrier", "GroupMemoryBarrierWithGroupSync", "InterlockedAdd", "InterlockedAnd", "InterlockedCompareExchange",
        "InterlockedCompareStore", "InterlockedExchange", "InterlockedMax", "InterlockedMin", "InterlockedOr", "InterlockedXor", "isfinite", "isinf", "isnan", "ldexp", "length", "lerp", "lit", "log", "log10", "log2", "mad", "max", "min",
        "modf", "msad4", "mul", "noise", "normalize", "pow", "printf", "Process2DQuadTessFactorsAvg", "Process2DQuadTessFactorsMax", "Process2DQuadTessFactorsMin", "ProcessIsolineTessFactors", "ProcessQuadTessFactorsAvg", "ProcessQuadTessFactorsMax",
        "ProcessQuadTessFactorsMin", "ProcessTriTessFactorsAvg", "ProcessTriTessFactorsMax", "ProcessTriTessFactorsMin", "radians", "rcp", "reflect", "refract", "reversebits", "round", "rsqrt", "saturate", "sign", "sin", "sincos", "sinh", "smoothstep",
        "sqrt", "step", "tan", "tanh", "tex1D", "tex1D", "tex1Dbias", "tex1Dgrad", "tex1Dlod", "tex1Dproj", "tex2D", "tex2D", "tex2Dbias", "tex2Dgrad", "tex2Dlod", "tex2Dproj", "tex3D", "tex3D", "tex3Dbias", "tex3Dgrad", "tex3Dlod", "tex3Dproj",
        "texCUBE", "texCUBE", "texCUBEbias", "texCUBEgrad", "texCUBElod", "texCUBEproj", "transpose", "trunc"};

    // From here: https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.00.pdf
    static std::unordered_set<std::string> glsl_keywords{
        "#version", "#extension", "#include", "attribute", "const", "uniform", "varying", "layout", "centroid", "flat", "smooth", "noperspective", "patch", "sample", "break", "continue", "do", "for", "while", "switch", "case", "default",
        "if", "else", "subroutine", "in", "out", "inout", "float", "double", "int", "void", "bool", "true", "false", "invariant", "discard", "return", "mat2", "mat3", "mat4", "dmat2", "dmat3", "dmat4",
        "mat2x2", "mat2x3", "mat2x4", "dmat2x2", "dmat2x3", "dmat2x4", "mat3x2", "mat3x3", "mat3x4", "dmat3x2", "dmat3x3", "dmat3x4", "mat4x2", "mat4x3", "mat4x4", "dmat4x2", "dmat4x3", "dmat4x4",
        "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "bvec2", "bvec3", "bvec4", "dvec2", "dvec3", "dvec4", "uint", "uvec2", "uvec3", "uvec4", "lowp", "mediump", "highp", "precision",
        "sampler1D", "sampler2D", "sampler3D", "samplerCube", "sampler1DShadow", "sampler2DShadow", "samplerCubeShadow", "sampler1DArray", "sampler2DArray", "sampler1DArrayShadow", "sampler2DArrayShadow",
        "isampler1D", "isampler2D", "isampler3D", "isamplerCube", "isampler1DArray", "isampler2DArray", "usampler1D", "usampler2D", "usampler3D", "usamplerCube", "usampler1DArray", "usampler2DArray",
        "sampler2DRect", "sampler2DRectShadow", "isampler2DRect", "usampler2DRect", "samplerBuffer", "isamplerBuffer", "usamplerBuffer", "sampler2DMS", "isampler2DMS", "usampler2DMS",
        "sampler2DMSArray", "isampler2DMSArray", "usampler2DMSArray", "samplerCubeArray", "samplerCubeArrayShadow", "isamplerCubeArray", "usamplerCubeArray", "struct",
        "gl_Position", "binding", "location", "EmitVertex", "EndPrimitive", "gl_in", "triangles", "line_strip", "max_vertices"};

    static std::unordered_set<std::string> glsl_identifiers = {
        "abort", "abs", "acos", "asin", "atan", "atexit", "atof", "atoi", "atol", "ceil", "clock", "cosh", "ctime", "div", "exit", "fabs", "floor", "fmod", "getchar", "getenv", "isalnum",
        "isalpha", "isdigit", "isgraph", "ispunct", "isspace", "isupper", "kbhit", "log10", "log2", "log", "memcmp", "modf", "pow", "putchar", "putenv", "puts", "rand", "remove", "rename",
        "sinh", "sqrt", "srand", "strcat", "strcmp", "strerror", "time", "tolower", "toupper"};

    static std::unordered_set<std::string> c_keywords = {
        "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short",
        "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary",
        "_Noreturn", "_Static_assert", "_Thread_local"};

    static std::unordered_set<std::string> c_identifiers = {
        "abort", "abs", "acos", "asin", "atan", "atexit", "atof", "atoi", "atol", "ceil", "clock", "cosh", "ctime", "div", "exit", "fabs", "floor", "fmod", "getchar", "getenv", "isalnum", "isalpha", "isdigit", "isgraph",
        "ispunct", "isspace", "isupper", "kbhit", "log10", "log2", "log", "memcmp", "modf", "pow", "putchar", "putenv", "puts", "rand", "remove", "rename", "sinh", "sqrt", "srand", "strcat", "strcmp", "strerror", "time", "tolower", "toupper"};
    static std::unordered_set<std::string> sql_keywords = {
        "ADD", "EXCEPT", "PERCENT", "ALL", "EXEC", "PLAN", "ALTER", "EXECUTE", "PRECISION", "AND", "EXISTS", "PRIMARY", "ANY", "EXIT", "PRINT", "AS", "FETCH", "PROC", "ASC", "FILE", "PROCEDURE",
        "AUTHORIZATION", "FILLFACTOR", "PUBLIC", "BACKUP", "FOR", "RAISERROR", "BEGIN", "FOREIGN", "READ", "BETWEEN", "FREETEXT", "READTEXT", "BREAK", "FREETEXTTABLE", "RECONFIGURE",
        "BROWSE", "FROM", "REFERENCES", "BULK", "FULL", "REPLICATION", "BY", "FUNCTION", "RESTORE", "CASCADE", "GOTO", "RESTRICT", "CASE", "GRANT", "RETURN", "CHECK", "GROUP", "REVOKE",
        "CHECKPOINT", "HAVING", "RIGHT", "CLOSE", "HOLDLOCK", "ROLLBACK", "CLUSTERED", "IDENTITY", "ROWCOUNT", "COALESCE", "IDENTITY_INSERT", "ROWGUIDCOL", "COLLATE", "IDENTITYCOL", "RULE",
        "COLUMN", "IF", "SAVE", "COMMIT", "IN", "SCHEMA", "COMPUTE", "INDEX", "SELECT", "CONSTRAINT", "INNER", "SESSION_USER", "CONTAINS", "INSERT", "SET", "CONTAINSTABLE", "INTERSECT", "SETUSER",
        "CONTINUE", "INTO", "SHUTDOWN", "CONVERT", "IS", "SOME", "CREATE", "JOIN", "STATISTICS", "CROSS", "KEY", "SYSTEM_USER", "CURRENT", "KILL", "TABLE", "CURRENT_DATE", "LEFT", "TEXTSIZE",
        "CURRENT_TIME", "LIKE", "THEN", "CURRENT_TIMESTAMP", "LINENO", "TO", "CURRENT_USER", "LOAD", "TOP", "CURSOR", "NATIONAL", "TRAN", "DATABASE", "NOCHECK", "TRANSACTION",
        "DBCC", "NONCLUSTERED", "TRIGGER", "DEALLOCATE", "NOT", "TRUNCATE", "DECLARE", "NULL", "TSEQUAL", "DEFAULT", "NULLIF", "UNION", "DELETE", "OF", "UNIQUE", "DENY", "OFF", "UPDATE",
        "DESC", "OFFSETS", "UPDATETEXT", "DISK", "ON", "USE", "DISTINCT", "OPEN", "USER", "DISTRIBUTED", "OPENDATASOURCE", "VALUES", "DOUBLE", "OPENQUERY", "VARYING", "DROP", "OPENROWSET", "VIEW",
        "DUMMY", "OPENXML", "WAITFOR", "DUMP", "OPTION", "WHEN", "ELSE", "OR", "WHERE", "END", "ORDER", "WHILE", "ERRLVL", "OUTER", "WITH", "ESCAPE", "OVER", "WRITETEXT"};

    static std::unordered_set<std::string> cmake_keywords = {
        "option", "add_compile_options", "cmake_minimum_required", "project", "message", "add_dependencies", "add_test", "find_package", "include_directories", "configure_file", "target_link_libraries", "source_group", "set", "set_property", "include", "add_executable", "add_library", "if", "elseif", "endif", "find", "glob"};

    static std::unordered_set<std::string> cmake_identifiers = {};

    static std::unordered_set<std::string> lua_keywords = {
        "and", "break", "do", "", "else", "elseif", "end", "false", "for", "function", "if", "in", "", "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while"};

    static std::unordered_set<std::string> lua_identifiers = {
        "assert", "collectgarbage", "dofile", "error", "getmetatable", "ipairs", "loadfile", "load", "loadstring", "next", "pairs", "pcall", "print", "rawequal", "rawlen", "rawget", "rawset",
        "select", "setmetatable", "tonumber", "tostring", "type", "xpcall", "_G", "_VERSION", "arshift", "band", "bnot", "bor", "bxor", "btest", "extract", "lrotate", "lshift", "replace",
        "rrotate", "rshift", "create", "resume", "running", "status", "wrap", "yield", "isyieldable", "debug", "getuservalue", "gethook", "getinfo", "getlocal", "getregistry", "getmetatable",
        "getupvalue", "upvaluejoin", "upvalueid", "setuservalue", "sethook", "setlocal", "setmetatable", "setupvalue", "traceback", "close", "flush", "input", "lines", "open", "output", "popen",
        "read", "tmpfile", "type", "write", "close", "flush", "lines", "read", "seek", "setvbuf", "write", "__gc", "__tostring", "abs", "acos", "asin", "atan", "ceil", "cos", "deg", "exp", "tointeger",
        "floor", "fmod", "ult", "log", "max", "min", "modf", "rad", "random", "randomseed", "sin", "sqrt", "string", "tan", "type", "atan2", "cosh", "sinh", "tanh",
        "pow", "frexp", "ldexp", "log10", "pi", "huge", "maxinteger", "mininteger", "loadlib", "searchpath", "seeall", "preload", "cpath", "path", "searchers", "loaded", "module", "require", "clock",
        "date", "difftime", "execute", "exit", "getenv", "remove", "rename", "setlocale", "time", "tmpname", "byte", "char", "dump", "find", "format", "gmatch", "gsub", "len", "lower", "match", "rep",
        "reverse", "sub", "upper", "pack", "packsize", "unpack", "concat", "maxn", "insert", "pack", "unpack", "remove", "move", "sort", "offset", "codepoint", "char", "len", "codes", "charpattern",
        "coroutine", "table", "io", "os", "string", "uint8_t", "bit32", "math", "debug", "package"};

    static std::unordered_set<std::string> lisp_keywords = {
        "+", "-", "eval"};

    static std::unordered_set<std::string> lisp_identifiers = {
        "cdr", "car"};

    static std::unordered_set<std::string> janet_keywords = {
        "if", "do", "fn", "while", "def", "var", "quote", "quasiquote", "unquote", "splice", "set", "break"};

    static std::unordered_set<std::string> janet_identifiers = {
        "%", "*", "+", "-", "/", "<", "<=", "=", ">", ">=", "abstract?", "accumulate", "accumulate2", "all", "all-bindings", "all-dynamics", "apply", "array", "array/concat", "array/ensure", "array/fill", "array/insert", "array/new", "array/new-filled", "array/peek", "array/pop", "array/push", "array/remove", "array/slice", "array?", "asm", "assert", "bad-compile", "bad-parse", "band", "blshift", "bnot", "boolean?", "bor", "brshift", "brushift", "buffer", "buffer/bit", "buffer/bit-clear", "buffer/bit-set", "buffer/bit-toggle", "buffer/blit", "buffer/clear", "buffer/fill", "buffer/format", "buffer/new", "buffer/new-filled", "buffer/popn", "buffer/push-byte", "buffer/push-string", "buffer/push-word", "buffer/slice", "buffer?", "bxor", "bytes?", "cfunction?", "cli-main", "comp", "compile", "complement", "count", "debug", "debug/arg-stack", "debug/break", "debug/fbreak", "debug/lineage", "debug/stack", "debug/stacktrace", "debug/step", "debug/unbreak", "debug/unfbreak", "debugger-env", "dec", "deep-not=", "deep=", "default-peg-grammar", "defglobal", "describe", "dictionary?", "disasm", "distinct", "doc*", "doc-format", "dofile", "drop", "drop-until", "drop-while", "dyn", "eflush", "empty?", "env-lookup", "eprin", "eprinf", "eprint", "eprintf", "error", "eval", "eval-string", "even?", "every?", "extreme", "false?", "fiber/can-resume?", "fiber/current", "fiber/getenv", "fiber/maxstack", "fiber/new", "fiber/root", "fiber/setenv", "fiber/setmaxstack", "fiber/status", "fiber?", "file/close", "file/flush", "file/open", "file/popen", "file/read", "file/seek", "file/temp", "file/write", "filter", "find", "find-index", "first", "flatten", "flatten-into", "flush", "freeze", "frequencies", "function?", "gccollect", "gcinterval", "gcsetinterval", "gensym", "get", "get-in", "getline", "hash", "idempotent?", "identity", "import*", "in", "inc", "indexed?", "int/s64", "int/u64", "int?", "interleave", "interpose", "invert", "janet/build", "janet/config-bits", "janet/version", "juxt*", "keep", "keys", "keyword", "keyword?", "kvs", "last", "length", "load-image", "load-image-dict", "macex", "macex1", "make-env", "make-image", "make-image-dict", "map", "mapcat", "marshal", "math/-inf", "math/abs", "math/acos", "math/acosh", "math/asin", "math/asinh", "math/atan", "math/atan2", "math/atanh", "math/cbrt", "math/ceil", "math/cos", "math/cosh", "math/e", "math/erf", "math/erfc", "math/exp", "math/exp2", "math/expm1", "math/floor", "math/gamma", "math/hypot", "math/inf", "math/log", "math/log10", "math/log1p", "math/log2", "math/next", "math/pi", "math/pow", "math/random", "math/rng", "math/rng-buffer", "math/rng-int", "math/rng-uniform", "math/round", "math/seedrandom", "math/sin", "math/sinh", "math/sqrt", "math/tan", "math/tanh", "math/trunc", "max", "mean", "merge", "merge-into", "min", "mod", "module/add-paths", "module/cache", "module/expand-path", "module/find", "module/loaders", "module/loading", "module/paths", "nan?", "nat?", "native", "neg?", "net/chunk", "net/close", "net/connect", "net/read", "net/server", "net/write", "next", "nil?", "not", "not=", "number?", "odd?", "one?", "os/arch", "os/cd", "os/chmod", "os/clock", "os/cryptorand", "os/cwd", "os/date", "os/dir", "os/environ", "os/execute", "os/exit", "os/getenv", "os/link", "os/lstat", "os/mkdir", "os/mktime", "os/perm-int", "os/perm-string", "os/readlink", "os/realpath", "os/rename", "os/rm", "os/rmdir", "os/setenv", "os/shell", "os/sleep", "os/stat", "os/symlink", "os/time", "os/touch", "os/umask", "os/which", "pairs", "parse", "parser/byte", "parser/clone", "parser/consume", "parser/eof", "parser/error", "parser/flush", "parser/has-more", "parser/insert", "parser/new", "parser/produce", "parser/state", "parser/status", "parser/where", "partial", "partition", "peg/compile", "peg/match", "pos?", "postwalk", "pp", "prewalk", "prin", "prinf", "print", "printf", "product", "propagate", "put", "put-in", "quit", "range", "reduce", "reduce2", "repl", "require", "resume", "return", "reverse", "root-env", "run-context", "scan-number", "setdyn", "signal", "slice", "slurp", "some", "sort", "sort-by", "sorted", "sorted-by", "spit", "stderr", "stdin", "stdout", "string", "string/ascii-lower", "string/ascii-upper", "string/bytes", "string/check-set", "string/find", "string/find-all", "string/format", "string/from-bytes", "string/has-prefix?", "string/has-suffix?", "string/join", "string/repeat", "string/replace", "string/replace-all", "string/reverse", "string/slice", "string/split", "string/trim", "string/triml", "string/trimr", "string?", "struct", "struct?", "sum", "symbol", "symbol?", "table", "table/clone", "table/getproto", "table/new", "table/rawget", "table/setproto", "table/to-struct", "table?", "take", "take-until", "take-while", "tarray/buffer", "tarray/copy-bytes", "tarray/length", "tarray/new", "tarray/properties", "tarray/slice", "tarray/swap-bytes", "thread/close", "thread/current", "thread/new", "thread/receive", "thread/send", "trace", "true?", "truthy?", "tuple", "tuple/brackets", "tuple/setmap", "tuple/slice", "tuple/sourcemap", "tuple/type", "tuple?", "type", "unmarshal", "untrace", "update", "update-in", "values", "varglobal", "walk", "yield", "zero?", "zipcoll",
        "%=", "*=", "++", "+=", "--", "-=", "->", "->>", "-?>", "-?>>", "/=", "and", "as->", "as?->", "case", "chr", "comment", "comptime", "cond", "coro", "def-", "default", "defer", "defmacro", "defmacro-", "defn", "defn-", "doc", "each", "eachk", "eachp", "edefer", "for", "generate", "if-let", "if-not", "if-with", "import", "juxt", "label", "let", "loop", "match", "or", "prompt", "protect", "seq", "short-fn", "try", "unless", "use", "var-", "varfn", "when", "when-let", "when-with", "with", "with-dyns", "with-syms", "with-vars"};

    static std::unordered_set<std::string> scenegraph_keywords = {
        "pass", "post_2d", "geometry", "script", "camera", "position", "look_at", "field_of_view", "near_far", "build_as", "vs", "gs", "fs", "model", "scale", "path", "surface", "size", "format", "targets", "clear", "ray_group_triangles", "ray_group_general", "ray_group_procedural", "ray_gen", "closest_hit", "miss", "any_hit", "callable", "intersection"};

    static std::unordered_set<std::string> py_keywords = {
        "false", "none", "true", "and", "as", "assert", "async", "await", "break", "case",
        "class", "continue", "def", "del", "elif", "else", "except", "finally", "for",
        "from", "global", "if", "import", "in", "is", "lambda", "match", "nonlocal",
        "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"};

    static std::unordered_set<std::string> py_identifiers = {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "frozenset",
        "int", "isinstance", "issubclass", "len", "list", "map", "max", "min", "open",
        "print", "range", "reversed", "set", "sorted", "str", "sum", "tuple", "type", "zip"};

    static std::unordered_set<std::string> scenegraph_identifiers = {
        ":", "=", "default", "default_color", "default_depth", "screen_rect"};

    static std::unordered_set<std::string> tree_keywords = {};
    static std::unordered_set<std::string> tree_identifiers = {};
    static std::unordered_set<std::string> markdown_keywords = {};
    static std::unordered_set<std::string> markdown_identifiers = {};

    void RegisterSyntaxProviders(ZepEditor& editor) {
        editor.RegisterSyntaxFactory({".scenegraph"}, SyntaxProvider{"scenegraph", tSyntaxFactory([](ZepBuffer* pBuffer) {
                                                                         return std::make_shared<ZepSyntax>(*pBuffer, scenegraph_keywords, scenegraph_identifiers);
                                                                     })});
        editor.RegisterSyntaxFactory({".vert", ".frag", ".geom", ".rchit", ".rgen", ".rmiss", ".glsl"}, SyntaxProvider{"gl_shader", tSyntaxFactory([](ZepBuffer* pBuffer) {
                                                                                                                           return std::make_shared<ZepSyntax>(*pBuffer, glsl_keywords, glsl_identifiers);
                                                                                                                       })});

        editor.RegisterSyntaxFactory({".hlsl", ".hlsli", ".vs", ".ps", ".gs"}, SyntaxProvider{"hlsl_shader", tSyntaxFactory([](ZepBuffer* pBuffer) {
                                                                                                  return std::make_shared<ZepSyntax>(*pBuffer, hlsl_keywords, hlsl_identifiers);
                                                                                              })});

        editor.RegisterSyntaxFactory({".cpp", ".cxx", ".hpp", ".h", ".c"}, SyntaxProvider{"cpp", tSyntaxFactory([](ZepBuffer* pBuffer) {
                                                                                              return std::make_shared<ZepSyntax>(*pBuffer, cpp_keywords, cpp_identifiers);
                                                                                          })});

        editor.RegisterSyntaxFactory({".lisp", ".lsp"}, SyntaxProvider{"lisp", tSyntaxFactory([](ZepBuffer* pBuffer) {
                                                                           return std::make_shared<ZepSyntax>(*pBuffer, lisp_keywords, lisp_identifiers, ZepSyntaxFlags::LispLike);
                                                                       })});

        editor.RegisterSyntaxFactory({".janet"}, SyntaxProvider{"janet", tSyntaxFactory([](ZepBuffer* pBuffer) {
                                                                    return std::make_shared<ZepSyntax>(*pBuffer, janet_keywords, janet_identifiers, ZepSyntaxFlags::LispLike);
                                                                })});

        editor.RegisterSyntaxFactory({".scm", ".scheme", ".sps", ".sls", ".sld", ".ss", ".sch"}, SyntaxProvider{"lisp", tSyntaxFactory([](ZepBuffer* pBuffer) {
                                                                                                                    return std::make_shared<ZepSyntax>(*pBuffer, lisp_keywords, lisp_identifiers, ZepSyntaxFlags::LispLike);
                                                                                                                })});

        editor.RegisterSyntaxFactory({".cmake", "CMakeLists.txt", "CMakeCache.txt"}, SyntaxProvider{"cmake", tSyntaxFactory([](ZepBuffer* pBuffer) {
                                                                                                        return std::make_shared<ZepSyntax>(*pBuffer, cmake_keywords, cmake_identifiers, ZepSyntaxFlags::CaseInsensitive);
                                                                                                    })});

        editor.RegisterSyntaxFactory(
            {".toml"},
            SyntaxProvider{"cpp", tSyntaxFactory([](ZepBuffer* pBuffer) {
                               return std::make_shared<ZepSyntax>(*pBuffer, toml_keywords, toml_identifiers, ZepSyntaxFlags::CaseInsensitive);
                           })});

        editor.RegisterSyntaxFactory(
            {".tree"},
            SyntaxProvider{"tree", tSyntaxFactory([](ZepBuffer* pBuffer) {
                               return std::make_shared<ZepSyntax_Tree>(*pBuffer, tree_keywords, tree_identifiers, ZepSyntaxFlags::CaseInsensitive);
                           })});

        editor.RegisterSyntaxFactory(
            {".md", ".markdown"},
            SyntaxProvider{"markdown", tSyntaxFactory([](ZepBuffer* pBuffer) {
                               return std::make_shared<ZepSyntax_Markdown>(*pBuffer, markdown_keywords, markdown_identifiers, ZepSyntaxFlags::CaseInsensitive);
                           })});

        editor.RegisterSyntaxFactory(
            {".py", ".python"},
            SyntaxProvider{"python", tSyntaxFactory([](ZepBuffer* pBuffer) {
                               return std::make_shared<ZepSyntax_Python>(*pBuffer, py_keywords, py_identifiers, ZepSyntaxFlags::CaseInsensitive);
                           })});
    }

} // namespace Zep

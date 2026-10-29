#!/usr/bin/env python3
"""Build historical default keymaps from their original C++ implementations.

Run --check in CI. Before changing defaults, bump PROFILE_VERSION and run
python3 tools/generate_keymap_defaults.py --record-previous HEAD.
"""
import argparse, base64, json, re, subprocess, tempfile, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/visualizer/input/generated_keymap_defaults.hpp"
SOURCE_ROWS = '''
v0.5.0 v0.5.0
v0.5.1 v0.5.1
v0.5.2 v0.5.2
v0.5.3 v0.5.3
v0.5.4 v0.5.4
v2 a14554398
v3 3d9772412
v4 f7bc72875
v5 0cb2f99d8
v10 b07736740
v12 7da59cc8d
v13 be1031c6e
v14 10d115a8d
v15 8e398da0c
v16 316f5f660
v17 9673429de
v18 ec6013c86
v19 43d29983d
v20 9169a2a00
v21 97191fe8e
v22 de67a4dc5
v23 3c17c1d9f
v25 ad83ef3ef
v26_window 424a41c36
v27 950710539
v28 7fb6b317ba8b
v26_group 95770df08
v29 27dd365aa
v29_master 11926fd5f2f8968514e1378f2e1212ffe25a6cfb
v30 11926fd5f2f8968514e1378f2e1212ffe25a6cfb
v31 2f4f139745259857dbfca51f644b55d66705cdaf
v32 6537262420834b1a7ff1a705cc2d6aa39b96c490
v33 28290f8891d96cb23fd50593ec1d4f5c594e81f8
v34 f3a2b375b7162e1eff44f6413b46f32854b5386a
v35 c9bbcaca31935de68e96dd5c008ad379ac86d4ff
v36 32e64ebd10458cac81f012579e83f13d2f60b4f9
v37 e8ed4816389d1b8e5b205e88802730e15db587ae
v38 d939cd94f9200bec5b867fa250342ded1e512ba6
v39 3152c3f4709a92873ceb5a674ff937817b1f9a4e
'''
SOURCES = dict(row.split() for row in SOURCE_ROWS.splitlines() if row.strip())
VERSION_OVERRIDES = {"v30": 30}  # v30 changed storage only; defaults match v29_master.
DECODER = 'inline std::string decodeHistoricalKeymapDefaults() {\n    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";\n    std::vector<unsigned char> compressed;\n    unsigned int accumulator = 0;\n    int bits = 0;\n    for (const char ch : kHistoricalKeymapDefaultsBase64) {\n        if (ch == \'=\') break;\n        const auto value = alphabet.find(ch);\n        if (value == std::string_view::npos) continue;\n        accumulator = (accumulator << 6) | static_cast<unsigned int>(value);\n        bits += 6;\n        if (bits >= 8) {\n            bits -= 8;\n            compressed.push_back(static_cast<unsigned char>((accumulator >> bits) & 0xff));\n            accumulator &= (1u << bits) - 1u;\n        }\n    }\n    z_stream stream{};\n    stream.next_in = compressed.data();\n    stream.avail_in = static_cast<uInt>(compressed.size());\n    if (inflateInit(&stream) != Z_OK)\n        throw std::runtime_error("Could not initialize historical keymap decoder");\n    std::string result;\n    std::vector<char> chunk(4096);\n    int status = Z_OK;\n    while (status == Z_OK) {\n        stream.next_out = reinterpret_cast<Bytef*>(chunk.data());\n        stream.avail_out = static_cast<uInt>(chunk.size());\n        status = inflate(&stream, Z_NO_FLUSH);\n        result.append(chunk.data(), chunk.size() - stream.avail_out);\n    }\n    inflateEnd(&stream);\n    if (status != Z_STREAM_END)\n        throw std::runtime_error("Could not decode historical keymap defaults");\n    return result;\n}\ninline const nlohmann::json& historicalKeymapDefaults() {\n    static const nlohmann::json table = nlohmann::json::parse(decodeHistoricalKeymapDefaults());\n    return table;\n}\n// clang-format on\n'

def show(ref, path):
    if ref == "WORKTREE": return (ROOT / path).read_text()
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)

def block(source, marker):
    start, opening, depth = source.index(marker), source.index("{", source.index(marker)), 0
    for i in range(opening, len(source)):
        depth += (source[i] == "{") - (source[i] == "}")
        if depth == 0: return source[start:i + 1]
    raise ValueError(marker)

def enum_values(header, kind):
    body = block(header, f"enum class {kind}").split("{", 1)[1].rsplit("}", 1)[0]
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
    result, number = {}, -1
    for entry in body.split(","):
        if not (entry := entry.strip()): continue
        match = re.fullmatch(r"(\w+)(?:\s*=\s*(\d+))?", entry)
        if not match: raise ValueError(f"unsupported {kind} entry: {entry}")
        number = int(match.group(2)) if match.group(2) else number + 1
        result[number] = match.group(1)
    return result

CPP = r'''#include <array>
#include <iostream>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include "input/key_codes.hpp"
namespace lfs::vis::input { using namespace std;
ENUMS
enum Modifier { MODIFIER_NONE=KEYMOD_NONE, MODIFIER_SHIFT=KEYMOD_SHIFT, MODIFIER_CTRL=KEYMOD_CTRL, MODIFIER_ALT=KEYMOD_ALT, MODIFIER_SUPER=KEYMOD_SUPER };
enum class MouseButton { LEFT=0, RIGHT=1, MIDDLE=2 };
struct KeyTrigger { int key; int modifiers=0; bool on_repeat=false; };
struct MouseButtonTrigger { MouseButton button; int modifiers=0; bool double_click=false; };
struct MouseScrollTrigger { int modifiers=0; optional<int> chord_key={}; };
struct MouseDragTrigger { MouseButton button; int modifiers=0; optional<int> chord_key={}; };
using InputTrigger=variant<KeyTrigger,MouseButtonTrigger,MouseScrollTrigger,MouseDragTrigger>;
struct Binding { ToolMode mode; InputTrigger trigger; Action action; string description; };
struct Profile { string name, description; vector<Binding> bindings; };
struct InputBindings { static Profile createDefaultProfile(); };
ARRAYS
ACTION_NAME
FUNCTION
}
int main() { using namespace lfs::vis::input; for (const auto& b : InputBindings::createDefaultProfile().bindings) { cout << int(b.mode) << '\t' << int(b.action) << '\t' << b.description << '\t'; visit([](const auto& t) { using T=decay_t<decltype(t)>; if constexpr(is_same_v<T,KeyTrigger>) cout << "key\t" << t.key << '\t' << t.modifiers << '\t' << t.on_repeat; else if constexpr(is_same_v<T,MouseButtonTrigger>) cout << "mouse_button\t" << int(t.button) << '\t' << t.modifiers << '\t' << t.double_click; else if constexpr(is_same_v<T,MouseScrollTrigger>) cout << "scroll\t" << t.modifiers << '\t' << t.chord_key.value_or(-1); else cout << "drag\t" << int(t.button) << '\t' << t.modifiers << '\t' << t.chord_key.value_or(-1); }, b.trigger); cout << '\n'; } }
'''

def generate(ref):
    source = show(ref, "src/visualizer/input/input_bindings.cpp")
    header = show(ref, "src/visualizer/input/input_bindings.hpp")
    try: key_codes = show(ref, "src/visualizer/input/key_codes.hpp")
    except subprocess.CalledProcessError: key_codes = (ROOT / "src/visualizer/input/key_codes.hpp").read_text()
    actions = enum_values(header, "Action")
    version = int(re.search(r"PROFILE_VERSION\s*=\s*(\d+)", source).group(1))
    arrays = []
    before_defaults = source[:source.index("Profile InputBindings::createDefaultProfile()")]
    for name in ("ALL_MODES", "NODE_PICK_MODES", "DELETE_NODE_MODES", "DELETE_GAUSSIANS_MODES"):
        match = re.search(r"constexpr (?:std::array<ToolMode,\s*\d+>|ToolMode)\s+" + name + r"(?:\[\])?\s*=\s*\{.*?\};", before_defaults, re.S)
        if match: arrays.append(match.group(0))
    cpp = CPP.replace("ENUMS", "\n".join(block(header, f"enum class {kind}") + ";" for kind in ("ToolMode", "Action")))
    cpp = cpp.replace("ARRAYS", "\n".join(arrays)).replace("ACTION_NAME", block(source, "std::string getActionName("))
    cpp = cpp.replace("FUNCTION", block(source, "Profile InputBindings::createDefaultProfile()")).replace("GLFW_KEY_", "KEY_")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "defaults.cpp"; path.write_text(cpp)
        (Path(tmp) / "input").mkdir(); (Path(tmp) / "input/key_codes.hpp").write_text(key_codes)
        exe = Path(tmp) / "defaults"
        subprocess.run(["c++", "-std=c++20", "-I", tmp, "-I", str(ROOT / "src/visualizer"), str(path), "-o", str(exe)], check=True)
        rows = subprocess.check_output([str(exe)], text=True)
    bindings = []
    for row in rows.splitlines():
        mode, action, desc, kind, *args = row.split("\t")
        item = dict(mode=int(mode), action=int(action), action_symbol=actions[int(action)], description=desc, trigger_type=kind)
        if kind == "key": item.update(key=int(args[0]), modifiers=int(args[1]), on_repeat=bool(int(args[2])))
        elif kind == "mouse_button": item.update(button=int(args[0]), modifiers=int(args[1]), double_click=bool(int(args[2])))
        elif kind == "scroll":
            item["modifiers"] = int(args[0])
            if int(args[1]) >= 0: item["chord_key"] = int(args[1])
        else:
            item.update(button=int(args[0]), modifiers=int(args[1]))
            if int(args[2]) >= 0: item["chord_key"] = int(args[2])
        bindings.append(item)
    return dict(name="Default", version=version, source=ref,
                actions={str(k): v for k, v in actions.items()}, bindings=bindings)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true")
    parser.add_argument("--record-previous", metavar="REF", help="record defaults before a profile version bump")
    args = parser.parse_args()
    if args.record_previous:
        ref = subprocess.check_output(["git", "rev-parse", args.record_previous], cwd=ROOT, text=True).strip()
        previous, current = generate(ref), generate("WORKTREE")
        if current["version"] != previous["version"] + 1: raise SystemExit("Bump PROFILE_VERSION by one before recording defaults")
        name = f'v{previous["version"]}'
        if name in SOURCES and SOURCES[name] != ref: raise SystemExit(f"{name} is already recorded from a different revision")
        if name not in SOURCES:
            script = Path(__file__); source = script.read_text(); marker = "\n'''\nSOURCES = dict"
            if marker not in source: raise SystemExit("Could not update historical source list")
            source = source.replace(marker, f"\n{name} {ref}" + marker, 1)
            script.write_text(source)
            SOURCES[name] = ref
    profiles, data, indices = [], [], {}
    for name, ref in SOURCES.items():
        item = generate(ref)
        if name in VERSION_OVERRIDES:
            if item["version"] + 1 != VERSION_OVERRIDES[name]: raise SystemExit(f"invalid unchanged-default alias: {name}")
            item["version"] = VERSION_OVERRIDES[name]
        canonical = json.dumps({"actions": item["actions"], "bindings": item["bindings"]}, separators=(",", ":"), sort_keys=True)
        index = indices.get(canonical)
        if index is None: index = len(data); indices[canonical] = index; data.append(json.loads(canonical))
        profiles.append(dict(version=item["version"], name=name, data=index)); print(f"{name}: profile version {item['version']}, table {index}")
    packed = base64.b64encode(zlib.compress(json.dumps(dict(profiles=profiles, data=data), separators=(",", ":"), sort_keys=True).encode(), 9)).decode("ascii")
    chunks = [packed[i:i + 992] for i in range(0, len(packed), 992)]
    lines = ["// Generated by tools/generate_keymap_defaults.py; do not edit.\n// clang-format off\n", "#pragma once\n#include <array>\n#include <stdexcept>\n#include <string>\n#include <string_view>\n#include <utility>\n#include <vector>\n#include <nlohmann/json.hpp>\n#include <zlib.h>\n", "namespace lfs::vis::input {\n", f"inline constexpr int kLatestHistoricalProfileVersion = {max(p['version'] for p in profiles)};\n", "inline constexpr std::string_view kHistoricalKeymapDefaultsBase64 =\n"]
    lines.extend(f'    "{chunk}"\n' for chunk in chunks); lines.append("    ;\n")
    lines.append(DECODER.replace('\n// clang-format on\n', '\n    // clang-format on\n'))
    symbols = enum_values((ROOT / "src/visualizer/input/input_bindings.hpp").read_text(), "Action")
    lines.append(f"    inline constexpr std::array<std::pair<std::string_view, int>, {len(symbols)}> kCurrentActionSymbols = {{{{\n")
    lines.extend(f'        {{"{symbol}", static_cast<int>(Action::{symbol})}},\n' for symbol in symbols.values())
    lines.append("    }};\n} // namespace lfs::vis::input\n"); generated = "".join(lines)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != generated: raise SystemExit(f"stale keymap defaults: {OUTPUT}")
    else: OUTPUT.write_text(generated)
    current = generate("WORKTREE")
    bindings = [{k:v for k,v in b.items() if k != "action_symbol"} for b in current["bindings"]]
    baseline = ROOT / f'tests/data/keymap_contract/current_v{current["version"]}.digest'; baseline.parent.mkdir(parents=True, exist_ok=True)
    digest = 14695981039346656037
    for byte in json.dumps(bindings, separators=(",", ":"), sort_keys=True).encode("utf-8"):
        digest = ((digest ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    content = f'version {current["version"]}\nfnv1a64 {digest:016x}\n'; print(f'current defaults v{current["version"]} fnv1a64: {digest:016x}')
    if args.check:
        if not baseline.exists() or baseline.read_text() != content:
            raise SystemExit("Default bindings changed without recording the previous version. Bump PROFILE_VERSION, then run: python3 tools/generate_keymap_defaults.py --record-previous HEAD")
    else: baseline.write_text(content)

if __name__ == "__main__": main()

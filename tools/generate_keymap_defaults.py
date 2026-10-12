#!/usr/bin/env python3
"""Build historical default keymaps from their original C++ implementations.

Run with --check in CI. A new profile version needs a snapshot of the previous
version before its defaults are edited; release tags are kept as separate rows.
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests/data/keymap_defaults"
SOURCES = {
    "v0.5.0": "v0.5.0",
    "v0.5.1": "v0.5.1",
    "v0.5.2": "v0.5.2",
    "v0.5.3": "v0.5.3",
    "v0.5.4": "v0.5.4",
    "v2": "a14554398",
    "v3": "3d9772412",
    "v4": "f7bc72875",
    "v5": "0cb2f99d8",
    "v10": "b07736740",
    "v12": "7da59cc8d",
    "v13": "be1031c6e",
    "v14": "10d115a8d",
    "v15": "8e398da0c",
    "v16": "316f5f660",
    "v17": "9673429de",
    "v18": "ec6013c86",
    "v19": "43d29983d",
    "v20": "9169a2a00",
    "v21": "97191fe8e",
    "v22": "de67a4dc5",
    "v23": "3c17c1d9f",
    "v25": "ad83ef3ef",
    "v26_window": "424a41c36",
    "v27": "950710539",
    "v26_group": "95770df08",
    "v29": "27dd365aa",
    "v29_master": "11926fd5f2f8968514e1378f2e1212ffe25a6cfb",
}


def show(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True)


def block(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for i in range(opening, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise ValueError(marker)


def enum_values(header, kind):
    body = block(header, f"enum class {kind}").split("{", 1)[1].rsplit("}", 1)[0]
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
    result = {}
    number = -1
    for entry in body.split(","):
        entry = entry.strip()
        if not entry:
            continue
        match = re.fullmatch(r"(\w+)(?:\s*=\s*(\d+))?", entry)
        if not match:
            raise ValueError(f"unsupported {kind} entry: {entry}")
        number = int(match.group(2)) if match.group(2) else number + 1
        result[number] = match.group(1)
    return result


def generate(name, ref):
    source = show(ref, "src/visualizer/input/input_bindings.cpp")
    header = show(ref, "src/visualizer/input/input_bindings.hpp")
    action_symbols = enum_values(header, "Action")
    version = int(re.search(r"PROFILE_VERSION\s*=\s*(\d+)", source).group(1))
    enums = "\n".join(block(header, f"enum class {kind}") + ";" for kind in ("ToolMode", "Action"))
    function = block(source, "Profile InputBindings::createDefaultProfile()")
    action_name = block(source, "std::string getActionName(")
    arrays = []
    for array in ("ALL_MODES", "NODE_PICK_MODES", "DELETE_NODE_MODES", "DELETE_GAUSSIANS_MODES"):
        match = re.search(r"constexpr (?:std::array<ToolMode,\s*\d+>|ToolMode)\s+" + array + r"(?:\[\])?\s*=\s*\{.*?\};", source[: source.index("Profile InputBindings::createDefaultProfile()")], re.S)
        if match:
            arrays.append(match.group(0))
    cpp = r'''#include <array>
#include <iostream>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include "input/key_codes.hpp"
namespace lfs::vis::input {
using namespace std;
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
int main() {
    using namespace lfs::vis::input;
    for (const auto& b : InputBindings::createDefaultProfile().bindings) {
        cout << int(b.mode) << '\t' << int(b.action) << '\t' << b.description << '\t';
        visit([](const auto& t) {
            using T=decay_t<decltype(t)>;
            if constexpr(is_same_v<T,KeyTrigger>) cout << "key\t" << t.key << '\t' << t.modifiers << '\t' << t.on_repeat;
            else if constexpr(is_same_v<T,MouseButtonTrigger>) cout << "mouse_button\t" << int(t.button) << '\t' << t.modifiers << '\t' << t.double_click;
            else if constexpr(is_same_v<T,MouseScrollTrigger>) cout << "scroll\t" << t.modifiers << '\t' << t.chord_key.value_or(-1);
            else cout << "drag\t" << int(t.button) << '\t' << t.modifiers << '\t' << t.chord_key.value_or(-1);
        }, b.trigger);
        cout << '\n';
    }
}
'''
    cpp = cpp.replace("ENUMS", enums).replace("ARRAYS", "\n".join(arrays)).replace("ACTION_NAME", action_name).replace("FUNCTION", function)
    cpp = cpp.replace("GLFW_KEY_", "KEY_")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "defaults.cpp"
        path.write_text(cpp)
        executable = Path(tmp) / "defaults"
        subprocess.run(["c++", "-std=c++20", "-I", str(ROOT / "src/visualizer"), str(path), "-o", str(executable)], check=True)
        rows = subprocess.check_output([str(executable)], text=True)
    bindings = []
    for row in rows.splitlines():
        mode, action, desc, kind, *args = row.split("\t")
        item = {"mode": int(mode), "action": int(action), "action_symbol": action_symbols[int(action)],
                "description": desc, "trigger_type": kind}
        if kind == "key":
            item.update(key=int(args[0]), modifiers=int(args[1]), on_repeat=bool(int(args[2])))
        elif kind == "mouse_button":
            item.update(button=int(args[0]), modifiers=int(args[1]), double_click=bool(int(args[2])))
        elif kind == "scroll":
            item.update(modifiers=int(args[0]))
            if int(args[1]) >= 0:
                item["chord_key"] = int(args[1])
        else:
            item.update(button=int(args[0]), modifiers=int(args[1]))
            if int(args[2]) >= 0:
                item["chord_key"] = int(args[2])
        bindings.append(item)
    return {"name": "Default", "version": version, "source": ref, "bindings": bindings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    embedded = []
    for name, ref in SOURCES.items():
        data = generate(name, ref)
        saved_profile = {"name": data["name"], "version": data["version"],
                         "bindings": [{key: value for key, value in binding.items()
                                       if key != "action_symbol"} for binding in data["bindings"]]}
        content = json.dumps(saved_profile, indent=2) + "\n"
        path = OUTPUT / f"{name}.json"
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"stale keymap defaults: {path}")
        else:
            path.write_text(content)
        embedded.append((data["version"], name, json.dumps(data, separators=(",", ":"))))
        print(f"{name}: {path.name}")
    header = ROOT / "src/visualizer/input/generated_keymap_defaults.hpp"
    lines = ["// Generated by tools/generate_keymap_defaults.py; do not edit.\n",
             "#pragma once\n#include <array>\n#include <string_view>\n#include <utility>\n",
             "namespace lfs::vis::input {\n",
             "struct HistoricalKeymapDefaults { int version; std::string_view name; std::string_view json; };\n",
             f"inline constexpr int kLatestHistoricalProfileVersion = {max(row[0] for row in embedded)};\n",
             f"inline constexpr std::array<HistoricalKeymapDefaults, {len(embedded)}> kHistoricalKeymapDefaults = {{{{\n"]
    for version, name, raw in embedded:
        lines.append(f'    {{{version}, "{name}", R"keymap({raw})keymap"}},\n')
    lines.append("}};\n")
    current_symbols = enum_values((ROOT / "src/visualizer/input/input_bindings.hpp").read_text(), "Action")
    lines.append(f"inline constexpr std::array<std::pair<std::string_view, int>, {len(current_symbols)}> kCurrentActionSymbols = {{{{\n")
    for number, symbol in current_symbols.items():
        lines.append(f'    {{"{symbol}", static_cast<int>(Action::{symbol})}},\n')
    lines.append("}};\n}\n")
    generated = "".join(lines)
    if args.check:
        if not header.exists() or header.read_text() != generated:
            raise SystemExit(f"stale keymap defaults: {header}")
    else:
        header.write_text(generated)


if __name__ == "__main__":
    main()

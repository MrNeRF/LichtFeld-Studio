#!/usr/bin/env python3
"""
String Extraction Tool for LichtFeld Studio

Finds hardcoded strings in C++ code that should be localized.

Usage:
    python scripts/extract_strings.py                    # Scan all GUI code
    python scripts/extract_strings.py --file menu_bar.cpp # Scan specific file
    python scripts/extract_strings.py --suggest          # Suggest key names
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def find_project_root() -> Path:
    """Find the project root directory"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'CMakeLists.txt').exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def extract_imgui_strings(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Extract hardcoded strings from ImGui calls

    Returns: List of (line_number, function_name, string_content)
    """
    results = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Pattern for ImGui calls with string literals
        # Matches: ImGui::Function("string", ...)
        pattern = r'ImGui::(\w+)\s*\(\s*"([^"]+)"'

        for line_num, line in enumerate(lines, 1):
            # Skip lines that use LOC()
            if 'LOC(' in line:
                continue

            # Skip comments
            if line.strip().startswith('//'):
                continue

            matches = re.finditer(pattern, line)
            for match in matches:
                func_name = match.group(1)
                string_content = match.group(2)

                # Skip certain patterns
                if should_skip_string(string_content, func_name):
                    continue

                results.append((line_num, func_name, string_content))

    except Exception as e:
        print(f"{Colors.RED}Error reading {file_path}: {e}{Colors.END}")

    return results


def should_skip_string(content: str, func_name: str) -> bool:
    """Determine if a string should be skipped"""

    # Skip empty strings
    if not content.strip():
        return True

    # Skip format specifiers only
    if content in ['%s', '%d', '%f', '%zu', '%u', '%llu']:
        return True

    # Skip internal IDs (start with ##)
    if content.startswith('##'):
        return True

    # Skip single characters or symbols
    if len(content) == 1:
        return True

    # Skip URLs
    if content.startswith('http://') or content.startswith('https://'):
        return True

    # Skip file paths
    if '/' in content or '\\' in content:
        return True

    # Skip things that look like C++ format strings
    if content.count('%') > 1:
        return True

    # PushID, PopID typically use internal identifiers
    if func_name in ['PushID', 'PopID']:
        return True

    return False


def suggest_key_name(string: str, func_name: str, context: str) -> str:
    """Suggest a localization key name for a string"""

    # Clean the string
    clean = re.sub(r'[^\w\s]', '', string.lower())
    clean = re.sub(r'\s+', '_', clean.strip())

    # Determine category from function name
    category = ""
    if func_name in ['MenuItem', 'BeginMenu']:
        category = "menu"
    elif func_name in ['Begin', 'BeginChild']:
        category = "window"
    elif func_name in ['Button']:
        category = "button"
    elif func_name in ['Text', 'TextColored', 'TextWrapped']:
        category = "label"
    else:
        category = "ui"

    # Truncate if too long
    if len(clean) > 30:
        words = clean.split('_')
        clean = '_'.join(words[:3])

    return f"{category}.{clean}"


def scan_directory(gui_dir: Path, specific_file: str = None) -> Dict[str, List]:
    """Scan directory for hardcoded strings"""

    results = defaultdict(list)

    if specific_file:
        files = [gui_dir / specific_file]
    else:
        files = list(gui_dir.rglob('*.cpp'))

    for file_path in files:
        if not file_path.exists():
            print(f"{Colors.YELLOW}⚠ File not found: {file_path}{Colors.END}")
            continue

        strings = extract_imgui_strings(file_path)
        if strings:
            rel_path = file_path.relative_to(gui_dir.parent.parent.parent)
            results[str(rel_path)] = strings

    return results


def print_results(results: Dict[str, List], suggest: bool = False):
    """Print extraction results"""

    if not results:
        print(f"{Colors.GREEN}✓ No hardcoded strings found!{Colors.END}")
        return

    total_strings = sum(len(strings) for strings in results.values())

    print(f"{Colors.BOLD}Found {total_strings} hardcoded string(s) in {len(results)} file(s):{Colors.END}\n")

    for file_path, strings in sorted(results.items()):
        print(f"{Colors.CYAN}{file_path}{Colors.END}")

        for line_num, func_name, content in strings:
            print(f"  {Colors.YELLOW}Line {line_num}{Colors.END}: "
                  f"{Colors.BLUE}{func_name}{Colors.END}(\"{content}\")")

            if suggest:
                key = suggest_key_name(content, func_name, file_path)
                print(f"    {Colors.MAGENTA}→ Suggested key: {key}{Colors.END}")
                print(f"    {Colors.MAGENTA}→ Replace with: LOC(Strings::???){Colors.END}")

        print()


def generate_todo_list(results: Dict[str, List]) -> str:
    """Generate a TODO checklist"""

    if not results:
        return ""

    todo = "# Localization TODOs\n\n"

    for file_path, strings in sorted(results.items()):
        todo += f"## {file_path}\n\n"

        for line_num, func_name, content in strings:
            todo += f"- [ ] Line {line_num}: `{func_name}(\"{content}\")`\n"

        todo += "\n"

    return todo


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract hardcoded strings from GUI code'
    )
    parser.add_argument(
        '--file',
        help='Specific file to scan (relative to gui/)'
    )
    parser.add_argument(
        '--suggest',
        action='store_true',
        help='Suggest localization key names'
    )
    parser.add_argument(
        '--todo',
        action='store_true',
        help='Generate TODO checklist'
    )

    args = parser.parse_args()

    try:
        project_root = find_project_root()
        gui_dir = project_root / 'src/visualizer/gui'

        if not gui_dir.exists():
            print(f"{Colors.RED}✗ GUI directory not found: {gui_dir}{Colors.END}")
            sys.exit(1)

        print(f"{Colors.BOLD}Scanning for hardcoded strings...{Colors.END}\n")

        results = scan_directory(gui_dir, args.file)

        print_results(results, suggest=args.suggest)

        if args.todo and results:
            todo = generate_todo_list(results)
            todo_path = project_root / 'LOCALIZATION_TODO.md'
            with open(todo_path, 'w') as f:
                f.write(todo)
            print(f"{Colors.GREEN}✓ TODO list saved to: {todo_path}{Colors.END}")

        if results:
            print(f"{Colors.YELLOW}💡 Tip: Use LOC() with string_keys.hpp constants{Colors.END}")
            print(f"{Colors.YELLOW}💡 Run with --suggest to get key name suggestions{Colors.END}")
            sys.exit(1)

    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

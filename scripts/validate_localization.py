#!/usr/bin/env python3
"""
Localization Validation Script for LichtFeld Studio

This script validates that all localization files are complete and consistent.

Features:
- Checks all languages have the same keys as English
- Finds missing translations
- Finds unused translations
- Validates JSON syntax
- Compares with string_keys.hpp and C++ LOC() calls
- --fix: auto-adds missing keys and reorders to match en.json key order
- --lang: scope to specific language(s)

Usage:
    python scripts/validate_localization.py                          # Check all
    python scripts/validate_localization.py --fix                    # Fix all
    python scripts/validate_localization.py --lang zh                # Check only zh
    python scripts/validate_localization.py --lang zh --lang ja      # Check zh & ja
    python scripts/validate_localization.py --lang zh,ja --fix       # Fix zh & ja
"""

import json
import sys
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Set


class Colors:
    """ANSI color codes for terminal output"""
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


def load_json_file(filepath: Path) -> OrderedDict:
    """Load and parse a JSON file, preserving key order."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}✗ JSON syntax error in {filepath.name}:{Colors.END}")
        print(f"  {e}")
        return None
    except Exception as e:
        print(f"{Colors.RED}✗ Error loading {filepath.name}:{Colors.END}")
        print(f"  {e}")
        return None


def save_json_file(data: OrderedDict, filepath: Path) -> None:
    """Write JSON with 2-space indent, trailing newline."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def flatten_dict(d: Dict, prefix: str = '') -> Dict[str, str]:
    """Flatten nested dictionary into dot-notation keys (for reporting only)."""
    result = {}
    for key, value in d.items():
        if key.startswith('_'):  # Skip metadata keys
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, full_key))
        elif isinstance(value, str):
            result[full_key] = value
    return result


def sync_dict(en_dict: OrderedDict, target_dict: OrderedDict) -> OrderedDict:
    """Recursively align target_dict keys and order with en_dict.

    Missing keys are filled from en_dict. Target-only keys are dropped.
    """
    result = OrderedDict()
    for key in en_dict:
        if key not in target_dict:
            result[key] = deepcopy(en_dict[key])
        elif isinstance(en_dict[key], dict) and isinstance(target_dict[key], dict):
            result[key] = sync_dict(en_dict[key], target_dict[key])
        else:
            result[key] = target_dict[key]
    return result


def extract_keys_from_cpp_header(header_path: Path) -> Set[str]:
    """Extract string keys from string_keys.hpp"""
    keys = set()
    try:
        with open(header_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Match: inline constexpr const char* NAME = "key.path.here";
            pattern = r'inline\s+constexpr\s+const\s+char\*\s+\w+\s*=\s*"([^"]+)"'
            matches = re.findall(pattern, content)
            keys.update(matches)
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ Could not parse string_keys.hpp: {e}{Colors.END}")
    return keys


def extract_keys_from_code(code_dir: Path) -> Set[str]:
    """Extract localization keys referenced from all consumer surfaces.

    Keys are consumed from C++ (LOC), Python/JS plugins (tr), and RML markup
    (data-tooltip), so scanning only C++ LOC() calls misses most of them.
    """
    keys = set()
    patterns = [
        r'LOC\s*\(\s*"([^"]+)"\s*\)',
        r'\btr\s*\(\s*["\']([^"\']+)["\']',
        r'(?<![\w-])data-tooltip\s*=\s*"([^"]+)"',
    ]
    source_exts = {'.cpp', '.hpp', '.h', '.cc', '.cu', '.cuh', '.py', '.rml', '.rcss', '.js'}

    for src_file in code_dir.rglob('*'):
        if src_file.suffix not in source_exts:
            continue
        try:
            content = src_file.read_text(encoding='utf-8')
        except Exception:
            continue  # Skip unreadable files silently
        for pattern in patterns:
            keys.update(re.findall(pattern, content))

    return keys


def validate_localization(project_root: Path, fix: bool = False, langs: list|None = None) -> int:
    """
    Validate localization files.

    Args:
        fix: If True, add missing keys and reorder to match en.json.
        langs: Optional list of language codes to check (e.g. ['zh', 'ja']).
               If None, all non-en languages are checked.

    Returns:
        0 if validation passed, 1 if errors found
    """
    locale_dir = project_root / 'src/visualizer/gui/resources/locales'
    header_path = project_root / 'src/visualizer/gui/string_keys.hpp'
    code_dir = project_root / 'src'

    if not locale_dir.exists():
        print(f"{Colors.RED}✗ Locale directory not found: {locale_dir}{Colors.END}")
        return 1

    print(f"{Colors.BOLD}Validating LichtFeld Studio Localization{Colors.END}\n")

    # Load all locale files
    locale_files = sorted(locale_dir.glob('*.json'))
    if not locale_files:
        print(f"{Colors.RED}✗ No locale files found in {locale_dir}{Colors.END}")
        return 1

    locales = {}
    for locale_file in locale_files:
        lang_code = locale_file.stem
        data = load_json_file(locale_file)
        if data is None:
            return 1
        locales[lang_code] = {
            'file': locale_file,
            'data': data,
            'flat': flatten_dict(data)
        }

    print(f"{Colors.GREEN}✓ Found {len(locales)} language(s):{Colors.END} {', '.join(locales.keys())}\n")

    # Use English as reference
    if 'en' not in locales:
        print(f"{Colors.RED}✗ English (en.json) not found - cannot validate{Colors.END}")
        return 1

    en_data = locales['en']['data']
    reference_keys = set(locales['en']['flat'].keys())
    print(f"{Colors.CYAN}Reference (English): {len(reference_keys)} keys{Colors.END}")

    # Filter target languages
    target_langs = [lc for lc in sorted(locales.keys()) if lc != 'en']
    if langs:
        unknown = [lc for lc in langs if lc not in target_langs]
        if unknown:
            print(f"\n{Colors.RED}✗ Unknown language code(s): {', '.join(unknown)}. "
                  f"Available: {', '.join(target_langs)}{Colors.END}\n")
            return 1
        target_langs = [lc for lc in target_langs if lc in langs]
    print()

    # Validate each language
    has_errors = False

    for lang_code in target_langs:
        locale_data = locales[lang_code]

        lang_keys = set(locale_data['flat'].keys())
        lang_name = locale_data['data'].get('_language_name', lang_code)

        print(f"{Colors.BOLD}Checking {lang_name} ({lang_code}):{Colors.END}")

        missing = reference_keys - lang_keys
        extra = lang_keys - reference_keys

        if missing:
            has_errors = True
            print(f"  {Colors.RED}✗ Missing {len(missing)} translation(s):{Colors.END}")
            for key in sorted(missing)[:10]:  # Show first 10
                print(f"    - {key}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

            if fix:
                print(f"  {Colors.YELLOW}→ Syncing: adding missing keys + reordering to en.json order...{Colors.END}")
                synced = sync_dict(en_data, locale_data['data'])
                save_json_file(synced, locale_data['file'])
                # Rebuild flat dict from synced data for accurate reporting
                locale_data['data'] = synced
                locale_data['flat'] = flatten_dict(synced)

        if extra:
            has_errors = True
            print(f"  {Colors.YELLOW}⚠ Extra {len(extra)} unused key(s) (dropped if --fix was used):{Colors.END}")
            for key in sorted(extra)[:5]:
                print(f"    - {key}")
            if len(extra) > 5:
                print(f"    ... and {len(extra) - 5} more")

        if not missing and not extra:
            print(f"  {Colors.GREEN}✓ Complete ({len(lang_keys)} keys){Colors.END}")

        print()

    # Check string_keys.hpp consistency
    header_keys = set()
    if header_path.exists():
        print(f"{Colors.BOLD}Checking string_keys.hpp consistency:{Colors.END}")
        header_keys = extract_keys_from_cpp_header(header_path)

        if header_keys:
            print(f"{Colors.CYAN}Found {len(header_keys)} keys in header{Colors.END}")

            missing_in_json = header_keys - reference_keys
            missing_in_header = reference_keys - header_keys

            if missing_in_json:
                has_errors = True
                print(f"{Colors.RED}✗ Keys in header but not in en.json:{Colors.END}")
                for key in sorted(missing_in_json)[:10]:
                    print(f"  - {key}")
                if len(missing_in_json) > 10:
                    print(f"  ... and {len(missing_in_json) - 10} more")

            if missing_in_header:
                print(f"{Colors.CYAN}ℹ {len(missing_in_header)} en.json key(s) not in header "
                      f"(header is a C++-only convenience subset; Python/RML/JS keys live only in en.json){Colors.END}")

            if not missing_in_json:
                print(f"{Colors.GREEN}✓ Every header constant resolves to an en.json key{Colors.END}")
        print()

    # Check for keys used in code
    print(f"{Colors.BOLD}Checking code usage:{Colors.END}")
    code_keys = extract_keys_from_code(code_dir)
    all_used_keys = code_keys | header_keys
    if all_used_keys:
        print(f"  {Colors.CYAN}{len(code_keys)} code reference(s) + {len(header_keys)} header = "
              f"{len(all_used_keys)} referenced keys{Colors.END}")

        unused = reference_keys - all_used_keys
        # A referenced key that is a strict prefix of a real key is a dynamically
        # built key (e.g. "histogram.metric." + name), not an undefined one.
        dynamic_prefixes = {rk.rsplit('.', 1)[0] for rk in reference_keys}
        undefined = {k for k in (all_used_keys - reference_keys)
                     if k not in dynamic_prefixes
                     and not any(rk.startswith(k + '.') for rk in reference_keys)}

        if undefined:
            has_errors = True
            print(f"  {Colors.RED}✗ {len(undefined)} undefined key(s) in code:{Colors.END}")
            for key in sorted(undefined)[:10]:
                print(f"    - {key}")
            if len(undefined) > 10:
                print(f"    ... and {len(undefined) - 10} more")
        else:
            print(f"  {Colors.GREEN}✓ All code references are defined{Colors.END}")

        if unused:
            print(f"  {Colors.YELLOW}⚠ {len(unused)} keys in en.json not referenced in code{Colors.END}")
    print()

    # Summary
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    if has_errors:
        print(f"{Colors.RED}✗ Validation FAILED - please fix the issues above{Colors.END}")
        if not fix:
            print(f"\n{Colors.CYAN}Tip: Run with --fix to auto-add missing keys and reorder{Colors.END}")
        return 1
    else:
        print(f"{Colors.GREEN}✓ Validation PASSED - all localizations are complete{Colors.END}")
        return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate LichtFeld Studio localization files')
    parser.add_argument('--fix', action='store_true',
                       help='Automatically add missing keys and reorder to match en.json')
    parser.add_argument('--lang', action='append', default=None,
                       help='Language code(s) to check (e.g. --lang zh --lang ja). '
                            'Repeatable or comma-separated. If omitted, checks all.')
    args = parser.parse_args()

    # Flatten --lang values and split comma-separated entries
    langs = None
    if args.lang:
        langs = []
        for entry in args.lang:
            langs.extend(s.strip() for s in entry.split(',') if s.strip())

    try:
        project_root = find_project_root()
        exit_code = validate_localization(project_root, fix=args.fix, langs=langs)
        sys.exit(exit_code)
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.END}")
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Localization Validation Script for LichtFeld Studio

This script validates that all localization files are complete and consistent.

Features:
- Checks all languages have the same keys as English
- Finds missing translations
- Finds unused translations
- Validates JSON syntax
- Compares with string_keys.hpp for consistency

Usage:
    python scripts/validate_localization.py
    python scripts/validate_localization.py --fix  # Auto-add missing keys with TODO markers
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict


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


def load_json_file(filepath: Path) -> Dict:
    """Load and parse a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}✗ JSON syntax error in {filepath.name}:{Colors.END}")
        print(f"  {e}")
        return None
    except Exception as e:
        print(f"{Colors.RED}✗ Error loading {filepath.name}:{Colors.END}")
        print(f"  {e}")
        return None


def flatten_dict(d: Dict, prefix: str = '') -> Dict[str, str]:
    """Flatten nested dictionary into dot-notation keys"""
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
    """Extract LOC() usage from C++ source files"""
    keys = set()
    pattern = r'LOC\s*\(\s*"([^"]+)"\s*\)'

    for cpp_file in code_dir.rglob('*.cpp'):
        try:
            with open(cpp_file, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = re.findall(pattern, content)
                keys.update(matches)
        except Exception:
            pass  # Skip unreadable files silently

    return keys


def validate_localization(project_root: Path, fix: bool = False) -> int:
    """
    Validate all localization files

    Returns:
        0 if validation passed, 1 if errors found
    """
    locale_dir = project_root / 'src/visualizer/gui/resources/locales'
    header_path = project_root / 'src/visualizer/gui/string_keys.hpp'
    code_dir = project_root / 'src/visualizer/gui'

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

    reference_keys = set(locales['en']['flat'].keys())
    print(f"{Colors.CYAN}Reference (English): {len(reference_keys)} keys{Colors.END}\n")

    # Validate each language
    has_errors = False

    for lang_code, locale_data in locales.items():
        if lang_code == 'en':
            continue

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
                print(f"  {Colors.YELLOW}→ Adding missing keys with TODO markers...{Colors.END}")
                # Add missing keys
                for key in missing:
                    locale_data['flat'][key] = f"TODO: {locales['en']['flat'][key]}"

        if extra:
            has_errors = True
            print(f"  {Colors.YELLOW}⚠ Extra {len(extra)} unused key(s):{Colors.END}")
            for key in sorted(extra)[:5]:
                print(f"    - {key}")
            if len(extra) > 5:
                print(f"    ... and {len(extra) - 5} more")

        if not missing and not extra:
            print(f"  {Colors.GREEN}✓ Complete ({len(lang_keys)} keys){Colors.END}")

        print()

    # Check string_keys.hpp consistency
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
                has_errors = True
                print(f"{Colors.YELLOW}⚠ Keys in en.json but not in header:{Colors.END}")
                for key in sorted(missing_in_header)[:10]:
                    print(f"  - {key}")
                if len(missing_in_header) > 10:
                    print(f"  ... and {len(missing_in_header) - 10} more")

            if not missing_in_json and not missing_in_header:
                print(f"{Colors.GREEN}✓ Header and JSON are in sync{Colors.END}")
        print()

    # Check for keys used in code
    print(f"{Colors.BOLD}Checking code usage:{Colors.END}")
    code_keys = extract_keys_from_code(code_dir)
    if code_keys:
        print(f"{Colors.CYAN}Found {len(code_keys)} LOC() calls in code{Colors.END}")

        unused = reference_keys - code_keys
        undefined = code_keys - reference_keys

        if undefined:
            has_errors = True
            print(f"{Colors.RED}✗ Undefined keys used in code:{Colors.END}")
            for key in sorted(undefined):
                print(f"  - {key}")

        if unused:
            print(f"{Colors.YELLOW}⚠ {len(unused)} defined but unused keys{Colors.END}")

        if not undefined:
            print(f"{Colors.GREEN}✓ All code references are defined{Colors.END}")
    print()

    # Summary
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    if has_errors:
        print(f"{Colors.RED}✗ Validation FAILED - please fix the issues above{Colors.END}")
        if not fix:
            print(f"\n{Colors.CYAN}Tip: Run with --fix to auto-add missing keys{Colors.END}")
        return 1
    else:
        print(f"{Colors.GREEN}✓ Validation PASSED - all localizations are complete{Colors.END}")
        return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate LichtFeld Studio localization files')
    parser.add_argument('--fix', action='store_true',
                       help='Automatically add missing keys with TODO markers')
    args = parser.parse_args()

    try:
        project_root = find_project_root()
        exit_code = validate_localization(project_root, fix=args.fix)
        sys.exit(exit_code)
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.END}")
        sys.exit(1)


if __name__ == '__main__':
    main()

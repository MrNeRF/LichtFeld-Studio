#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate shipped locale keys and std::format placeholders against en.json."""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCALES_DIR = PROJECT_ROOT / "src" / "visualizer" / "gui" / "resources" / "locales"


def flatten_strings(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Return every string leaf using the same dotted-key convention as LocalizationManager."""
    flattened: dict[str, str] = {}
    for key, value in data.items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, str):
            flattened[dotted_key] = value
        elif isinstance(value, dict):
            flattened.update(flatten_strings(value, dotted_key))
    return flattened


def load_locale(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as locale_file:
        data = json.load(locale_file)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return flatten_strings(data)


def placeholders(value: str) -> tuple[str, ...]:
    """Return normalized format placeholders, raising ValueError for malformed braces."""
    return tuple(sorted(
        f"{{{field_name}{f'!{conversion}' if conversion else ''}{f':{format_spec}' if format_spec else ''}}}"
        for _, field_name, format_spec, conversion in string.Formatter().parse(value)
        if field_name is not None
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Validate shipped locale keys and std::format placeholders against en.json.

en.json is the canonical locale. The default check validates every shipped
locale and exits with status 1 when a locale:
  - omits a key from en.json;
  - contains an unexpected or obsolete key; or
  - has malformed or mismatched std::format placeholders.

Named placeholders may be reordered to suit the target language. Identical
non-empty values are allowed as temporary English fallbacks by default; use an
audit option below to report or reject them explicitly.""",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Required CI-equivalent validation.\n"
            "  python tools/check_locale_completeness.py\n\n"
            "  # Count non-empty values still equal to English. Does not fail.\n"
            "  python tools/check_locale_completeness.py --report-identical\n\n"
            "  # Print every value counted by the identical-value audit. Does not fail.\n"
            "  python tools/check_locale_completeness.py --list-identical\n\n"
            "  # Translation audit: fail if a non-empty value still equals English.\n"
            "  python tools/check_locale_completeness.py --fail-on-identical"
        ),
    )
    parser.add_argument(
        "--report-identical",
        action="store_true",
        help="report non-empty values that still match en.json without failing",
    )
    parser.add_argument(
        "--list-identical",
        action="store_true",
        help="list the keys counted by --report-identical without failing",
    )
    parser.add_argument(
        "--fail-on-identical",
        action="store_true",
        help="treat non-empty values matching en.json as failures for a manual translation audit",
    )
    args = parser.parse_args()
    if args.list_identical or args.fail_on_identical:
        args.report_identical = True
    return args


def main() -> int:
    args = parse_args()
    english_path = LOCALES_DIR / "en.json"
    english = load_locale(english_path)
    english_keys = set(english)
    failures: list[str] = []
    identical_reports: list[tuple[str, list[str]]] = []

    for locale_path in sorted(LOCALES_DIR.glob("*.json")):
        if locale_path == english_path:
            continue
        locale = load_locale(locale_path)
        missing = sorted(english_keys - set(locale))
        extra = sorted(set(locale) - english_keys)
        if missing:
            failures.append(f"{locale_path.name} is missing {len(missing)} key(s):")
            failures.extend(f"  {key}" for key in missing)
        if extra:
            failures.append(f"{locale_path.name} has {len(extra)} unexpected key(s):")
            failures.extend(f"  {key}" for key in extra)
        for key in sorted(english_keys & set(locale)):
            try:
                english_placeholders = placeholders(english[key])
                locale_placeholders = placeholders(locale[key])
            except ValueError as error:
                failures.append(f"{locale_path.name} has an invalid format string for {key}: {error}")
                continue
            if locale_placeholders != english_placeholders:
                failures.append(
                    f"{locale_path.name} has mismatched placeholders for {key}: "
                    f"expected {english_placeholders}, got {locale_placeholders}"
                )
        if args.report_identical:
            identical = sorted(
                key
                for key in english_keys & set(locale)
                if english[key].strip() and english[key] == locale[key]
            )
            identical_reports.append((locale_path.name, identical))

    if failures:
        print("Locale completeness check failed:")
        print("\n".join(failures))
    if identical_reports:
        print("Untranslated-value report (identical to en.json):")
        for locale_name, identical in identical_reports:
            print(f"{locale_name}: {len(identical)} value(s)")
            if args.list_identical:
                print("\n".join(f"  {key}" for key in identical))

    if failures:
        return 1
    if args.fail_on_identical and any(identical for _, identical in identical_reports):
        return 1

    print(f"Locale completeness check passed for {len(english_keys)} English keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

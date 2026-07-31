#!/usr/bin/env python3
"""Update the machine-gated runtime ownership inventory after explicit review."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


BEGIN = "<!-- P8-RUNTIME-INVENTORY-BEGIN -->"
END = "<!-- P8-RUNTIME-INVENTORY-END -->"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    matrix_path = repository / "docs/licht_ownership_matrix.md"
    markers = args.inventory.read_text(encoding="utf-8").strip().splitlines()
    if not markers or any(not line.startswith("<!-- P8-RUNTIME ") for line in markers):
        print("ownership-inventory FAIL: input is not a runtime marker list", file=sys.stderr)
        return 1
    if markers != sorted(set(markers)):
        print("ownership-inventory FAIL: markers must be unique and sorted", file=sys.stderr)
        return 1
    matrix = matrix_path.read_text(encoding="utf-8")
    if BEGIN not in matrix or END not in matrix:
        print("ownership-inventory FAIL: matrix lacks inventory boundaries", file=sys.stderr)
        return 1
    before, remainder = matrix.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    updated = before + BEGIN + "\n" + "\n".join(markers) + "\n" + END + after
    if not args.write:
        if updated != matrix:
            print("ownership-inventory FAIL: document differs from runtime inventory")
            return 1
        print(f"ownership-inventory PASS: items={len(markers)}")
        return 0
    if os.environ.get("LICHT_OWNERSHIP_UPDATE") != "1":
        print("ownership-inventory FAIL: --write requires LICHT_OWNERSHIP_UPDATE=1", file=sys.stderr)
        return 1
    matrix_path.write_text(updated, encoding="utf-8")
    print(f"ownership-inventory UPDATED: items={len(markers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

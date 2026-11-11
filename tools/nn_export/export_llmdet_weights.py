#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert the official LLMDet Tiny safetensors checkpoint without Torch."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/python"))
from _lfs_llmdet_weights import SOURCE, convert
from _lfs_nn_weights import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.out)
    with args.out.open("rb") as stream:
        print(f"wrote {args.out} sha256={sha256_file(stream)}")


if __name__ == "__main__":
    main()

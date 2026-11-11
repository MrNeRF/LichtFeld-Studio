#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare BERT lexical data from the local, pinned official LLMDet vocabulary.

Source: iSEE-Laboratory/llmdet_tiny at d05199165a19320a9236396e20fca0a5e065189c.
No network, model weights or model framework is needed.
"""
import argparse
import gzip
import hashlib
import io
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, required=True)
    args = parser.parse_args()
    raw = args.vocab.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3":
        raise ValueError("Vocabulary does not match the pinned BERT source")
    output = Path(__file__).resolve().parents[1] / "src/python/lfs_plugins/assets/bert/vocab.txt.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    packed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=packed, mtime=0) as stream:
        stream.write(raw)
    output.write_bytes(packed.getvalue())


if __name__ == "__main__":
    main()

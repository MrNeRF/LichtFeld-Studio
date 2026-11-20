# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""BERT lexical parity and bounded LLMDet input contracts; no model runtime."""
import gzip
from importlib import import_module
import json
from pathlib import Path

import pytest
from test_rmlui_image_sources import panel_modules


@pytest.fixture
def tokenizer_module(panel_modules):
    return import_module("lfs_plugins.bert_tokenizer")


def test_matches_independent_bert_wordpiece_reference(tokenizer_module):
    tokenizer = tokenizer_module.BertTokenizer.load()
    cases = json.loads((Path(__file__).parents[1] / "data/nn/bert_text_ref_fixture.json").read_text(encoding="utf-8"))
    for text, ids in cases["tokenizer_cases"].items():
        assert tokenizer.encode(text) == tuple(ids), text


@pytest.mark.parametrize("text", ["", "\u0000\u200b", "[CLS] person", "[MASK]", "person " * 254, "x" * 1025])
def test_rejects_empty_special_and_overlong_input(tokenizer_module, text):
    with pytest.raises(ValueError):
        tokenizer_module.BertTokenizer.load().encode(text)


def test_corrupt_vocabulary_is_rejected(tokenizer_module, monkeypatch, tmp_path):
    path = tmp_path / "assets/bert/vocab.txt.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(b"corrupt"))
    monkeypatch.setattr(tokenizer_module, "__file__", str(tmp_path / "bert_tokenizer.py"))
    with pytest.raises(ValueError):
        tokenizer_module.BertTokenizer.load()

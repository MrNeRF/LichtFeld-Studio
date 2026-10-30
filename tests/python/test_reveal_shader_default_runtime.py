# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare every deployed Slang shader recipe with the PR base using the same compiler."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / 'src/rendering/rasterizer/vulkan/shader/src/slang'


def _recipes():
    # slang_recipes.json lists every recipe the rasterizer build compiles.
    manifest = os.environ.get('LFS_SHADER_RECIPES')
    if not manifest or not Path(manifest).is_file():
        return [pytest.param(None, None, marks=pytest.mark.skip(reason='requires LFS_SHADER_RECIPES'))]
    data = json.loads(Path(manifest).read_text())
    return [pytest.param(data['flags'], recipe, id=recipe['name']) for recipe in data['recipes']]


@pytest.mark.parametrize("flags,recipe", _recipes())
def test_deployed_output_matches_base(tmp_path, flags, recipe):
    compiler = os.environ.get('LFS_SLANGC')
    base = os.environ.get('LFS_SHADER_BASE_DIR')
    if not compiler or not base:
        pytest.skip('requires LFS_SLANGC and LFS_SHADER_BASE_DIR')
    outputs = []
    for source_dir in (Path(base), CURRENT):
        output = tmp_path / f'{len(outputs)}.spv'
        subprocess.run([compiler, str(source_dir / recipe['source']), *flags,
                        *[f'-D{definition}' for definition in recipe['defines']], f'-I{source_dir}',
                        '-o', str(output)], check=True)
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1], f"{recipe['name']} output differs from the PR base"

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare deployed default shader output with the PR base using the same compiler."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / 'src/rendering/rasterizer/vulkan/shader/src/slang'


@pytest.mark.parametrize("source,definitions", [
    ("vertex_shader.slang", ["EXPORT_MODE=0"]),
    ("vertex_shader.slang", ["EXPORT_MODE=0", "LFS_3DGUT_PROJECTION=1"]),
    ("cull_splats.slang", ["EXPORT_MODE=0", "CULL_ENTRY=1"]),
    ("alphablend_shader.slang", ["EXPORT_MODE=0", "LFS_3DGUT_RASTERIZE=1"]),
])
def test_default_projection_matches_base(tmp_path, source, definitions):
    compiler = os.environ.get('LFS_SLANGC')
    base = os.environ.get('LFS_SHADER_BASE_DIR')
    if not compiler or not base:
        pytest.skip('requires LFS_SLANGC and LFS_SHADER_BASE_DIR')
    outputs = []
    for source_dir in (Path(base), CURRENT):
        output = tmp_path / f'{len(outputs)}.spv'
        subprocess.run([compiler, str(source_dir / source), '-stage', 'compute', '-O',
                        '-fp-mode', 'fast', '-line-directive-mode', 'none', '-target', 'spirv',
                        *[f'-D{definition}' for definition in definitions], f'-I{source_dir}', '-o', str(output)], check=True)
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1], f'default {source} output differs from the PR base'

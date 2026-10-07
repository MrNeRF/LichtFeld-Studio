# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Selection pipeline stages reject nonpositive iteration counts."""

import pytest


@pytest.mark.parametrize("stage_name", ["grow", "shrink"])
@pytest.mark.parametrize("iterations", [0, -1])
def test_selection_stages_reject_nonpositive_iterations(lf, stage_name, iterations):
    stage_factory = getattr(lf.pipeline.select, stage_name)
    with pytest.raises(ValueError, match="iterations must be positive"):
        stage_factory(iterations=iterations)


def test_selection_stages_accept_positive_iterations(lf):
    assert lf.pipeline.select.grow(iterations=1) is not None
    assert lf.pipeline.select.shrink(iterations=1) is not None

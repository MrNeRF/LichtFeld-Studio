// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../../src/rendering/rasterizer/slang/spark_alpha.inc"

struct Parameters {
    Ptr<float, Access.ReadWrite> input;
    Ptr<float, Access.ReadWrite> output;
    uint count;
    float unused0;
    float unused1;
    uint unused2;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < p.count)
        p.output[tid.x] = lfsSparkAlphaFromDensity(p.input[2 * tid.x], p.input[2 * tid.x + 1]);
}

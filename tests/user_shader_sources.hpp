/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::test {
    inline constexpr char kIdentity[] = "import lfs;\nLfsElement effect(LfsElement e, LfsContext ctx) { return e; }\n";
    inline constexpr char kGain[] = R"(import lfs;
struct Params { [LfsParam(0.0, 2.0, 1.0)] float gain; };
LfsElement effect(LfsElement e, LfsContext ctx, Params p) { e.color *= p.gain; return e; })";
    inline constexpr char kTint[] = R"(import lfs;
struct Params { [LfsParam(0.0, 1.0, 0.5)] float3 tint; };
LfsElement effect(LfsElement e, LfsContext ctx, Params p) { e.color += p.tint; return e; })";
    inline constexpr char kEdit[] = R"(import lfs;
LfsElement effect(LfsElement e, LfsContext ctx) {
    e.position += ctx.param(0).xyz + ctx.param(1).xyz * e.color.r;
    if (ctx.param(2).w > 0.0) e.color = ctx.param(2).xyz;
    if (ctx.param(3).w > 0.0) e.opacity = ctx.param(3).x;
    if (ctx.param(4).w > 0.0) e.scale = ctx.param(4).xyz;
    if (any(ctx.param(5) != float4(0.0))) e.rotation = ctx.param(5);
    if (ctx.param(6).x > 0.0) e.position.x = ctx.param(6).x / 0.0;
    e.index = 7u;
    return e;
}
)";
    inline constexpr char kTinyEffect[] =
        "import lfs;\nLfsElement effect(LfsElement e, LfsContext ctx) {\n"
        "    e.color = float3(0.25, 0.5, 0.75);\n    e.opacity = 0.5;\n    return e;\n}\n";
    inline constexpr char kShift[] = R"(import lfs;
struct Params { [LfsParam(-1.0, 1.0, 0.0)] float3 shift; };
LfsElement effect(LfsElement e, LfsContext ctx, Params p) {
    e.color += p.shift;
    return e;
})";
    inline constexpr char kGreen[] = R"(import lfs;
struct Params { [LfsParam(0.0, 1.0, 0.0)] float green; };
LfsElement effect(LfsElement e, LfsContext ctx, Params p) {
    e.color.g += p.green;
    return e;
})";
    inline constexpr char kObjectBounds[] = R"(import lfs;
LfsElement effect(LfsElement e, LfsContext ctx) {
    float extent = max(ctx.boundsMax.x - ctx.boundsMin.x, 0.001);
    e.color = float3(0.3 * float(e.objectId), (e.position.x - ctx.boundsMin.x) / extent, 0.5);
    return e;
})";
    inline constexpr char kBroken[] = "import lfs;\nLfsElement effect(LfsElement e, LfsContext ctx) { return f; }\n";
    inline constexpr char kCameraSide[] = R"(import lfs;
LfsElement effect(LfsElement e, LfsContext ctx) {
    e.color = ctx.cameraPosition.x > 0.0 ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 1.0);
    return e;
})";
} // namespace lfs::test

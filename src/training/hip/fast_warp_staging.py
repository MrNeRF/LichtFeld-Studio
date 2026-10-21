# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Private wave32 staging adaptation; preserves the CUDA per-pixel derivative."""
import re


def adjacent_pixels(text: str) -> str:
    """Map warps to pairs (0,2), (1,3), (4,6), (5,7), each covering 8x8 pixels."""
    replacements = {
        "const uint st0 = warp_id;": "const uint st0 = (warp_id & 1u) + ((warp_id >> 1) * 4u);",
        "const uint st1 = warp_id + 4;": "const uint st1 = st0 + 2u;",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f"HIP adjacent subtile seam changed: {old}")
        text = text.replace(old, new)
    return text


def adapt(text: str) -> str:
    start = text.index("        // BWD-A:")
    end = text.index("        float pixel_error0", start)
    reduction_end = text.index("        constexpr int kBwdWarps", start)
    reduction = text[start:reduction_end]
    result = text[:start] + reduction + """        const int T_eff = min(tile_n_primitives, static_cast<int>(local_max_contrib));
        if (T_eff <= 0) return;

""" + text[end:]
    start = result.index("        // Fetch batch size:")
    fetch = result.index("            // Collaborative fetch", start)
    result = result[:start] + """        // Each full wave advances its own reverse prefix in register batches.
        // The accepted contribution order of each pixel is unchanged.
        const bool stage_depth = grad_depth_map != nullptr;
        for (int batch_base = 0; batch_base < T_eff; batch_base += 32) {
            const int n_batch = min(32, T_eff - batch_base);
            uint hip_fetch_valid = 0, hip_fetch_prim_idx = 0, hip_fetch_work_idx = 0;
            float2 hip_fetch_mean2d = make_float2(0.0f);
            ushort4 hip_fetch_pixel_bbox = make_ushort4(0, 0, 0, 0);
            float4 hip_fetch_conic_opacity = make_float4(0.0f), hip_fetch_color = make_float4(0.0f);
            float hip_fetch_depth = 0;
            float3 hip_fetch_normal = make_float3(0.0f);
""" + result[fetch:]
    begin = result.index("            // Collaborative fetch")
    end = result.index("            // WARP_BWD_WALK_BEGIN", begin)
    stage = result[begin:end].replace("thread_rank", "lane_id")
    fields = {"s_valid_splat": "valid", "s_prim_idx": "prim_idx", "s_work_idx": "work_idx",
              "s_mean2d": "mean2d", "s_pixel_bbox": "pixel_bbox", "s_conic_opacity": "conic_opacity",
              "s_color": "color", "s_depth": "depth", "s_normal": "normal"}
    for old, new in fields.items():
        stage = stage.replace(old + "[lane_id]", "hip_fetch_" + new)
    stage = stage.replace("            block.sync();", "")
    result = result[:begin] + stage + result[end:]
    for old, new in fields.items():
        result = result.replace(old + "[j_test]", "hip_fetch_" + new)
        if old in ("s_mean2d", "s_conic_opacity", "s_color", "s_normal"):
            result = result.replace(old + "[j]", "lfs_fast_uniform_broadcast(hip_fetch_" + new + ", k)")
        else:
            result = result.replace(old + "[j]", "lfs_fast_uniform_shuffle(hip_fetch_" + new + ", k)")
    result = result.replace("            block.sync(); // end of batch: safe re-use of shared staging", "")
    if "block.sync()" in result or "__syncthreads" in re.sub(r"//[^\n]*", "", result):
        raise RuntimeError("Unexpected block synchronization in wave-staged backward")
    if re.search(r"\bs_(?:valid_splat|prim_idx|work_idx|mean2d|pixel_bbox|conic_opacity|color|depth|normal)\b", result):
        raise RuntimeError("Unadapted shared staging reference in wave-staged backward")
    return result

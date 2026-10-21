# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Private RGB forward with one wave covering a complete 16x16 tile."""
import re
from fast_four_pixels import brace_end, shifted


def adapt(text: str) -> str:
    start = text.index("        static_assert(config::tile_width")
    end = text.index("        const uint tile_idx", start)
    prefix = text[:start] + """
        static_assert(config::tile_width == 16 && config::tile_height == 16,
                      "Eight-pixel forward requires 16x16 tiles");
        static_assert(!kRenderNormal && !kRenderDepth, "Eight-pixel forward is RGB-only");
        const uint lane_id = thread_rank & 31u;
        const uint local_x = lane_id & 7u, local_y = lane_id >> 3;
        const uint2 start_pixel_coords = {group_index.x * 16u, group_index.y * 16u};
"""
    for p in range(8):
        prefix += f"""
        const uint tlx{p} = {p % 2 * 8}u + local_x;
        const uint tly{p} = {p // 2 * 4}u + local_y;
        const uint pixel_rank{p} = tly{p} * 16u + tlx{p};
        const uint2 pix{p} = {{start_pixel_coords.x + tlx{p}, start_pixel_coords.y + tly{p}}};
        const bool inside{p} = pix{p}.x < width && pix{p}.y < height;
        const float2 pixel{p} = make_float2(__uint2float_rn(pix{p}.x), __uint2float_rn(pix{p}.y)) + 0.5f;
        const uint sub{p}_ix0 = start_pixel_coords.x + {p % 2 * 8}u;
        const uint sub{p}_iy0 = start_pixel_coords.y + {p // 2 * 4}u;
        const uint sub{p}_ix1 = sub{p}_ix0 + 8u, sub{p}_iy1 = sub{p}_iy0 + 4u;
"""
    tail = text[end:]
    start = tail.index("        float3 color_pixel0")
    end = tail.index("        for (int n_points_remaining", start)
    states = tail[start:end]
    tail = tail[:end] + "".join(shifted(states, offset) for offset in (2, 4, 6)) + tail[end:]
    old = "__syncthreads_count(done0 && done1) == config::block_size_blend_forward"
    assert tail.count(old) == 1
    tail = tail.replace(old, "__syncthreads_count(" + " && ".join(f"done{p}" for p in range(8)) + ") == 32")
    start = tail.index("            if (static_cast<int>(thread_rank) < batch_size)")
    end = brace_end(tail, start)
    fetch = tail[start:end].replace("thread_rank", "fetch_lane")
    fetch = fetch.replace("if (static_cast<int>(fetch_lane) < batch_size)",
                          "for (uint fetch_lane = thread_rank; static_cast<int>(fetch_lane) < batch_size; fetch_lane += 32u)")
    tail = tail[:start] + fetch + tail[end:]
    declarations = "                bool intersects0 = false;\n                bool intersects1 = false;"
    assert tail.count(declarations) == 1
    tail = tail.replace(declarations, "\n".join(f"                bool intersects{p} = false;" for p in range(8)))
    start = tail.index("                    intersects0 =")
    end = tail.index("                }", start)
    tests = tail[start:end]
    tail = tail[:end] + "".join(shifted(tests, offset) for offset in (2, 4, 6)) + tail[end:]
    start = tail.index("                unsigned mask0")
    end = tail.index("                if (warp_cull_mode", start)
    masks = tail[start:end]
    tail = tail[:end] + "".join(shifted(masks, offset) for offset in (2, 4, 6)) + tail[end:]
    tail = tail.replace("mask0 = mask1 =", " = ".join(f"mask{p}" for p in range(8)) + " =")
    start = tail.index("                    const bool walk0")
    end = tail.index("                    if (!hit0 && !hit1)", start)
    walks = tail[start:end]
    tail = tail[:end] + "".join(shifted(walks, offset) for offset in (2, 4, 6)) + tail[end:]
    tail = tail.replace("if (!hit0 && !hit1)", "if (" + " && ".join(f"!hit{p}" for p in range(8)) + ")")
    blocks = []
    for p in range(2):
        start = tail.index(f"                    if (hit{p})")
        blocks.append(tail[start:brace_end(tail, start)])
    end = tail.index(blocks[1]) + len(blocks[1])
    tail = tail[:end] + "\n" + "\n".join(shifted("\n".join(blocks), offset) for offset in (2, 4, 6)) + tail[end:]
    start = tail.index("        if (inside0)")
    epilogue = tail[start:]
    tail += "".join(shifted(epilogue, offset) for offset in (2, 4, 6))
    # A full tile belongs to one wave. Stage exactly one primitive per lane,
    # preserving the original per-pixel order without duplicate wave fetches.
    tail = tail.replace("config::block_size_blend_forward", "32")
    tail = re.sub(r"        __shared__ [^\n]+\n", "", tail)
    early = "if (" + " && ".join(f"!hit{p}" for p in range(8)) + ")\n                        continue;"
    assert tail.count(early) == 1
    tail = tail.replace(early, "")
    boundary = "                    const float log_alpha_pt = c.w;"
    assert tail.count(boundary) == 1
    # Keep every source lane active for all register broadcasts. Pixel-local
    # early exits may only occur after those full-wave collectives complete.
    tail = tail.replace(boundary, boundary + "\n                    " + early)
    done = " && ".join(f"done{p}" for p in range(8))
    tail = tail.replace("__syncthreads_count(" + done + ") == 32",
                        "lfs_fast_wave_ballot(" + done + ") == 0xffffffffu")
    tail = tail.replace("            block.sync();", "")
    declarations = """
            float2 hip_forward_mean = make_float2(0.0f);
            ushort4 hip_forward_bbox = make_ushort4(0, 0, 0, 0);
            float4 hip_forward_conic = make_float4(0.0f), hip_forward_color = make_float4(0.0f);
            float hip_forward_depth = 0.0f;
            float3 hip_forward_normal = make_float3(0.0f);
"""
    fetch = "            for (uint fetch_lane = thread_rank;"
    assert tail.count(fetch) == 1
    tail = tail.replace(fetch, declarations + fetch)
    for source, target, vector in (
        ("collected_mean2d", "hip_forward_mean", True),
        ("collected_conic_opacity", "hip_forward_conic", True),
        ("collected_color", "hip_forward_color", True),
        ("collected_depth", "hip_forward_depth", False),
        ("collected_normal", "hip_forward_normal", True),
        ("s_bbox", "hip_forward_bbox", False),
    ):
        tail = tail.replace(source + "[fetch_lane]", target)
        tail = tail.replace(source + "[j_test]", target)
        broadcast = "lfs_fast_uniform_broadcast" if vector else "lfs_fast_uniform_shuffle"
        tail = tail.replace(source + "[j]", broadcast + "(" + target + ", k)")
    if "__shared__" in tail or "block.sync()" in tail or "__syncthreads" in tail:
        raise RuntimeError("Unexpected block-wide staging in single-wave forward")
    return prefix + tail

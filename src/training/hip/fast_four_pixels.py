# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Private multi-pixel RGB backward, preserving per-quad clipping."""
import re


def shifted(text: str, offset: int = 2) -> str:
    def replace(match):
        if match[1] is not None:
            return "sub" + str(int(match[1]) + offset) + "_" + match[2]
        return match[3] + str(int(match[4]) + offset)
    return re.sub(r"\bsub([01])_(\w+)\b|\b([A-Za-z_]\w*?)([01])\b", replace, text)


def brace_end(text: str, start: int) -> int:
    pos = text.index("{", start) + 1
    depth = 1
    while depth:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return pos


def adapt(text: str, pixels: int = 4) -> str:
    """Input is the independent wave32 body, before adjacent-pair duplication."""
    if pixels not in (4, 8):
        raise ValueError("Expected four or eight pixels per lane")
    pairs = pixels // 2
    offsets = range(2, pixels, 2)
    start = text.index("        static_assert(config::tile_width")
    state_end = text.index("        float pixel_error0")
    prefix = text[:start] + """
        static_assert(config::tile_width == 16 && config::tile_height == 16 &&
                      config::block_size_blend == 256, "Four-pixel HIP layout requires 16x16 tiles");
        static_assert((DENSIFICATION_TYPE == DensificationType::None || DENSIFICATION_TYPE == DensificationType::MCMC) && !NORMAL_CHANNEL && !IMPACT_SCORE &&
                      WARP_CULL_MODE == 0 && !HIP_DEPTH_OR_EDGE, "Four-pixel HIP kernel is RGB-only");
        // Each wave covers 16x8 pixels. Quads (0,2) and (1,3) retain the
        // original adjacent-wave 8x8 clipping domains.
        const uint lane_id = thread_rank & 31u;
        const uint warp_id = thread_rank >> 5;
        const uint local_x = lane_id & 7u;
        const uint local_y = lane_id >> 3;
        const uint n_pixels = width * height;
        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * 16u, tile_coords.y * 16u};
        const float sub_w = 8.0f, sub_h = 4.0f;
"""
    for p in range(pixels):
        prefix += f"""
        const uint sub{p}_ix0 = start_pixel_coords.x + {p % 2 * 8}u;
        const uint sub{p}_iy0 = start_pixel_coords.y + warp_id * 8u + {p // 2 * 4}u;
        const uint sub{p}_ix1 = sub{p}_ix0 + 8u, sub{p}_iy1 = sub{p}_iy0 + 4u;
        const float sub{p}_x0 = static_cast<float>(sub{p}_ix0), sub{p}_y0 = static_cast<float>(sub{p}_iy0);
        const uint2 pix{p} = {{sub{p}_ix0 + local_x, sub{p}_iy0 + local_y}};
        const bool inside{p} = pix{p}.x < width && pix{p}.y < height;
        const uint pixel_rank{p} = (warp_id * 8u + {p // 2 * 4}u + local_y) * 16u + {p % 2 * 8}u + local_x;
        const uint pixel_idx{p} = inside{p} ? width * pix{p}.y + pix{p}.x : 0u;
        const float2 pixel{p} = make_float2(__uint2float_rn(pix{p}.x), __uint2float_rn(pix{p}.y)) + 0.5f;
        const uint last{p} = inside{p} ? tile_n_contributions[pixel_idx{p}] : 0u;
        float T{p} = inside{p} ? tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank{p}] : 1.0f;
        float3 grad_c{p} = inside{p} ? make_float3(grad_image[pixel_idx{p}], grad_image[n_pixels + pixel_idx{p}], grad_image[2 * n_pixels + pixel_idx{p}]) : make_float3(0.0f);
        float grad_T{p} = inside{p} ? background_transmittance_grad(background_grad, grad_c{p}, pixel_idx{p}, n_pixels) : 0.0f;
        float3 behind{p} = make_float3(0.0f); // Impact scoring uses the upstream two-pixel kernel.
        const float grad_d{p} = (inside{p} && grad_depth_map != nullptr) ? grad_depth_map[pixel_idx{p}] : 0.0f;
        float3 grad_n{p} = make_float3(0.0f);
        if constexpr (NORMAL_CHANNEL) {{
            if (inside{p} && grad_normal_map != nullptr)
                grad_n{p} = make_float3(grad_normal_map[pixel_idx{p}], grad_normal_map[n_pixels + pixel_idx{p}], grad_normal_map[2 * n_pixels + pixel_idx{p}]);
        }}
"""
    if pixels == 8:
        prefix = prefix.replace("Four-pixel", "Eight-pixel").replace("Each wave covers 16x8", "Each wave covers 16x16")
        prefix = prefix.replace("Quads (0,2) and (1,3)", "Quads (0,2), (1,3), (4,6), (5,7)")
    maxima = [f"last{p}" for p in range(pixels)]
    while len(maxima) > 1:
        maxima = [f"max({maxima[i]}, {maxima[i + 1]})" for i in range(0, len(maxima), 2)]
    prefix += "\n        uint local_max_contrib = " + maxima[0] + ";"
    prefix += """
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            local_max_contrib = max(local_max_contrib, lfs_fast_wave_shuffle_xor(local_max_contrib, offset));
        const int T_eff = min(tile_n_primitives, static_cast<int>(local_max_contrib));
        if (T_eff <= 0) return;
"""
    tail = text[state_end:]
    end = tail.index("        // Each full wave")
    tail = tail[:end] + "".join(shifted(tail[:end], offset) for offset in offsets) + tail[end:]
    tail = tail.replace("bool hit0 = false, hit1 = false;", "bool " + ", ".join(f"hit{p} = false" for p in range(pixels)) + ";")
    begin = tail.index("                    const bool aabb0")
    end = tail.index("                    const float4 co", begin)
    tail = tail[:end] + "".join(shifted(tail[begin:end], offset) for offset in offsets) + tail[end:]
    blocks = []
    for p in range(2):
        begin = tail.index(f"                    if (aabb{p})")
        blocks.append(tail[begin:brace_end(tail, begin)])
    end = tail.index(blocks[1]) + len(blocks[1])
    tail = tail[:end] + "\n" + "\n".join(shifted("\n".join(blocks), offset) for offset in offsets) + tail[end:]
    for first, last in [("                unsigned mask0", "                if constexpr (WARP_CULL_MODE"),
                        ("                    const bool use0", "                    if constexpr (WARP_CULL_MODE")]:
        begin = tail.index(first)
        end = tail.index(last, begin)
        tail = tail[:end] + "".join(shifted(tail[begin:end], offset) for offset in offsets) + tail[end:]
    tail = tail.replace("mask0 = mask1 =", " = ".join(f"mask{p}" for p in range(pixels)) + " =")
    tail = tail.replace("unsigned live = mask0 | mask1;", "unsigned live = " + " | ".join(f"mask{p}" for p in range(pixels)) + ";")
    accum_anchor = "BlendBackwardAccum accum = make_zero_blend_backward_accum();"
    contribution_anchor = "bool has_contribution = false;"
    assert tail.count(accum_anchor) == tail.count(contribution_anchor) == 1
    tail = tail.replace(accum_anchor, f"BlendBackwardAccum pair_accum[{pairs}] = {{" + ", ".join("make_zero_blend_backward_accum()" for _ in range(pairs)) + "};")
    tail = tail.replace(contribution_anchor, f"bool pair_contribution[{pairs}] = {{" + ", ".join("false" for _ in range(pairs)) + "};")
    anchor = "auto accumulate_pixel = [&](const bool use"
    assert tail.count(anchor) == 1
    tail = tail.replace(anchor, "auto accumulate_pixel = [&](BlendBackwardAccum& accum, bool& has_contribution, const bool use")
    pattern = r"                    accumulate_pixel\(use0,.*?;\s*accumulate_pixel\(use1,.*?;"
    matches = list(re.finditer(pattern, tail, re.S))
    assert len(matches) == 1
    match = matches[0]
    calls = match[0] + "\n" + "\n".join(shifted(match[0], offset) for offset in offsets)
    for p in range(pixels):
        pair = (p // 4) * 2 + p % 2
        calls = calls.replace(f"accumulate_pixel(use{p},", f"accumulate_pixel(pair_accum[{pair}], pair_contribution[{pair}], use{p},")
    tail = tail[:match.start()] + calls + tail[match.end():]
    # Reuse the existing field reduction and destination mapping. Reduce and
    # clamp the original 8x8 quads independently before combining fields.
    start = tail.index("#ifdef LFS_HIP_FAST_SCATTER_REDUCE")
    reduce_start = tail.index("                    // Warp-reduce", start)
    scatter_start = tail.index("                        if (lane_id < 16 && value != 0.0f)", reduce_start)
    scatter_end = brace_end(tail, scatter_start)
    emit = tail[tail.index("{", scatter_start) + 1:scatter_end - 1]
    emit = emit.replace("lane_id < 13 ? clamp_grad(value) : value", "value")
    reduce = tail[reduce_start:scatter_start]
    fields = ["mean_x", "mean_y", "conic_x", "conic_y", "conic_z", "compensated_opacity", "color_x", "color_y", "color_z", "depth", "normal_x", "normal_y", "normal_z", "densification_weight", "densification_error_weighted", "edge_weighted_contribution"]
    # Sum all absolute fields from the old quads per lane. If every lane is
    # below clamp/64, each old 32-lane reduction is below clamp/2 (including a
    # generous FP32 rounding margin). Their clamps are therefore identities.
    # NaN/Inf propagates through addition and makes the strict comparison false;
    # do not replace this bound with fmax, which could discard a NaN operand.
    bound = "                    float merge_bound = 0.0f;\n"
    for field in fields:
        bound += "                    merge_bound += " + " + ".join(f"fabsf(pair_accum[{i}].{field})" for i in range(pairs)) + ";\n"
    bound += "                    const bool merge_safe = lfs_fast_wave_ballot(merge_bound < GRAD_CLAMP_MAX / 64.0f) == 0xffffffffu;\n"
    merged = "                        BlendBackwardAccum merged = make_zero_blend_backward_accum();\n"
    for field in fields:
        merged += f"                        merged.{field} = " + " + ".join(f"pair_accum[{i}].{field}" for i in range(pairs)) + ";\n"
    replacement = bound + """                    float combined_field = 0.0f;
                    auto reduce_pair = [&](const BlendBackwardAccum& accum, const bool has_contribution) {
""" + reduce + """
                        if (lane_id < 16)
                            combined_field += lane_id < 13 ? clamp_grad(value) : value;
                        }
                    };
                    if (merge_safe) {
""" + merged + "                        reduce_pair(merged, " + " || ".join(f"pair_contribution[{i}]" for i in range(pairs)) + ");" + """
                    } else {
""" + "".join(f"                        reduce_pair(pair_accum[{i}], pair_contribution[{i}]);\n" for i in range(pairs)).rstrip("\n") + """
                    }
                    const float value = combined_field;
                    if (lane_id < 16 && value != 0.0f && primitive_idx < n_primitives) {
""" + emit + "\n                    }\n"
    end = tail.index("\n                }\n            }\n            // WARP_BWD_WALK_END", start)
    tail = tail[:start] + replacement + tail[end:]
    result = prefix + tail
    if pixels == 8:
        start = result.index("                        const float mean_x = reduce_field(accum.mean_x);")
        end = result.index("                        if (lane_id < 16)", start)
        reference = result[start:end].replace("const float value =", "value =")
        fields = ["mean_x", "mean_y", "conic_x", "conic_y", "conic_z", "compensated_opacity", "color_x", "color_y", "color_z", "depth", "normal_x", "normal_y", "normal_z", "densification_weight", "densification_error_weighted", "edge_weighted_contribution"]
        alternative = "                        float value;\n#ifdef LFS_HIP_FAST_BUTTERFLY_FIELDS\n                        if (n_contrib > 1u) {\n                            value = lfs_fast_reduce_scatter16(" + ", ".join("accum." + field for field in fields) + ", lane_id);\n                        } else\n#endif\n                        {\n" + reference + "                        }\n"
        result = result[:start] + alternative + result[end:]
    return result

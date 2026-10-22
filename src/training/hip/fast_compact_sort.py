# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Private HIP tile keys after exact visible-depth ordering.

No depth bits are discarded: the stable visible ordering already carries them.
The original 64-bit path remains available for reference and small worksets.
"""


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"HIP compact sort seam changed: {old}")
    return text.replace(old, new)


def kernels(text: str) -> str:
    start = text.index("    __device__ __forceinline__ void emit_ellipse_tile_span(")
    end = text.index("    // Dual-pixel forward blend:", start)
    body = text[start:end]
    for prefix, name in (("__device__ __forceinline__", "emit_ellipse_tile_span"),
                         ("__device__ __noinline__", "create_instances_coop_warp"),
                         ("__global__", "create_instances_cu"),
                         ("__global__", "extract_instance_ranges_cu")):
        body = replace_once(body, f"    {prefix} void {name}(",
                            f"    template<class HipInstanceKey>\n    {prefix} void {name}(")
    body = body.replace("InstanceKey*", "HipInstanceKey*")
    body = replace_once(body,
                        "            instance_keys[write_at] = make_instance_key(tile_key, depth_key);",
                        "            if constexpr (sizeof(HipInstanceKey) == sizeof(uint))\n"
                        "                instance_keys[write_at] = tile_key;\n"
                        "            else\n"
                        "                instance_keys[write_at] = make_instance_key(tile_key, depth_key);")
    if body.count(">> kInstanceDepthBits") != 2:
        raise RuntimeError("HIP compact range extraction seam changed")
    body = body.replace(">> kInstanceDepthBits",
                        ">> (sizeof(HipInstanceKey) == sizeof(uint) ? 0 : kInstanceDepthBits)")
    return text[:start] + body + text[end:]


def forward(text: str) -> str:
    start = text.index("    struct FastGSSortWorkspace {")
    end = text.index("    // test hooks", start)
    body = text[start:end].replace("raster::InstanceKey", "HipInstanceKey")
    body = "    template<class HipInstanceKey>\n" + body
    text = text[:start] + body + text[end:]
    text = replace_once(text, "    FastGSSortWorkspace sort_workspace;\n", "")
    start = text.index("        if (n_instances > 0) {")
    end = text.index("\n        }\n    }", start)
    body = text[start + len("        if (n_instances > 0) {"):end]
    body = body.replace("InstanceKey", "HipInstanceKey")
    body = body.replace("key_end_bit", "hip_sort_end_bit")
    body = body.replace("lfs_fast_sort_pairs(", "lfs_fast_instance_sort_pairs(")
    replacement = """        if (n_instances > 0) {
            auto sort_instances = [&]<class HipInstanceKey>() {
                FastGSSortWorkspace<HipInstanceKey> sort_workspace;
                const int hip_sort_end_bit = key_end_bit -
                    (sizeof(HipInstanceKey) == sizeof(uint) ? kInstanceDepthBits : 0);
""" + body + """
            };
            static const bool compact_keys = [] {
                const char* value = std::getenv("LFS_HIP_FAST_COMPACT_TILE_KEYS");
                const bool enabled = !value || std::string_view(value) != "0";
                if (enabled) std::fprintf(stderr, "[info] HIP compact stable tile keys enabled\\n");
                return enabled;
            }();
            if (compact_keys && hip_depth_order.indices)
                sort_instances.template operator()<uint>();
            else
                sort_instances.template operator()<InstanceKey>();"""
    text = text[:start] + replacement + text[end:]
    seam = '#include "fast_depth_sort.hpp"'
    helper = r'''
template<class Key, class Value, class Count>
hipError_t lfs_fast_instance_sort_pairs(void* temporary, size_t& bytes,
    hipcub::DoubleBuffer<Key>& keys, hipcub::DoubleBuffer<Value>& values,
    Count count, int begin_bit, int end_bit, hipStream_t stream,
    bool depth_ordered) {
    if constexpr (sizeof(Key) == sizeof(unsigned)) {
        if (end_bit == 0) {
            if (!temporary) bytes = 1;
            return hipSuccess;
        }
        return lfs_fast_full_sort_pairs(temporary, bytes, keys, values,
                                       count, 0, end_bit, stream);
    } else {
        return lfs_fast_sort_pairs(temporary, bytes, keys, values, count,
                                  begin_bit, end_bit, stream, depth_ordered);
    }
}
'''
    return replace_once(text, seam, seam + helper)

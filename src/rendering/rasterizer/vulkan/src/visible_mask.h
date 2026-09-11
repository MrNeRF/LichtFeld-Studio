#pragma once

#include "../shader/src/slang/visible_mask.inc"

#include <cstddef>

namespace lfs::rendering::vulkan::visible_mask {

    inline constexpr std::size_t kWorkgroupSize = LFS_VK_VISIBLE_MASK_WORKGROUP_SIZE;
    inline constexpr std::size_t kWordBits = LFS_VK_VISIBLE_MASK_WORD_BITS;
    inline constexpr std::size_t kWordsPerWorkgroup = LFS_VK_VISIBLE_MASK_WORDS_PER_WORKGROUP;

    [[nodiscard]] constexpr std::size_t workgroupCount(const std::size_t num_splats) {
        return (num_splats + kWorkgroupSize - 1u) / kWorkgroupSize;
    }

    [[nodiscard]] constexpr std::size_t maskWordCount(const std::size_t num_splats) {
        return workgroupCount(num_splats) * kWordsPerWorkgroup;
    }

    static_assert(kWorkgroupSize % kWordBits == 0);
    static_assert(kWorkgroupSize == kWordsPerWorkgroup * kWordBits);

} // namespace lfs::rendering::vulkan::visible_mask

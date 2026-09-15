#ifndef LFS_VK_VISIBLE_MASK_INC
#define LFS_VK_VISIBLE_MASK_INC

// Geometry shared by the Vulkan host and the Slang visibility passes. Keep
// these as preprocessor constants so the two translation units cannot drift.
#define LFS_VK_VISIBLE_MASK_WORKGROUP_SIZE 128u
#define LFS_VK_VISIBLE_MASK_WORD_BITS      32u
#define LFS_VK_VISIBLE_MASK_WORDS_PER_WORKGROUP \
    (LFS_VK_VISIBLE_MASK_WORKGROUP_SIZE / LFS_VK_VISIBLE_MASK_WORD_BITS)

#endif // LFS_VK_VISIBLE_MASK_INC

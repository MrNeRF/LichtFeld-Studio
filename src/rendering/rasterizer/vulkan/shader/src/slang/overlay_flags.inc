#ifndef LFS_VK_OVERLAY_FLAGS_INC
#define LFS_VK_OVERLAY_FLAGS_INC

// One byte per splat, accessed as packed 32-bit words on every Vulkan device.
// Shared by the host allocation/dispatch and the shader writer.
#define LFS_VK_OVERLAY_WORD_BYTES 4u
#define LFS_VK_OVERLAY_WRITE_BIT 64u

#endif // LFS_VK_OVERLAY_FLAGS_INC

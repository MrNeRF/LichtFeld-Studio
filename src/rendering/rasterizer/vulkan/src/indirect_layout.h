#pragma once

#include "config.h"

#include "../shader/src/slang/indirect_layout.inc"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vulkan/vulkan.h>

namespace lfs::rendering::vulkan::indirect_layout {

    struct Layout {
        std::string_view word_count_constant;
        std::size_t word_count;
    };

    inline constexpr std::size_t kCommandWordCount = LFS_VK_DISPATCH_COMMAND_WORD_COUNT;
    inline constexpr std::size_t kCommandXWordOffset = LFS_VK_DISPATCH_X_WORD_OFFSET;
    inline constexpr std::size_t kCommandYWordOffset = LFS_VK_DISPATCH_Y_WORD_OFFSET;
    inline constexpr std::size_t kCommandZWordOffset = LFS_VK_DISPATCH_Z_WORD_OFFSET;

    [[nodiscard]] constexpr VkDeviceSize byteOffset(const std::size_t word_offset) {
        return static_cast<VkDeviceSize>(word_offset * sizeof(std::uint32_t));
    }

    [[nodiscard]] constexpr std::size_t byteSize(const Layout layout) {
        return layout.word_count * sizeof(std::uint32_t);
    }

    [[nodiscard]] constexpr std::size_t depthWaveRecordUpperBound(
        const std::size_t raw_instances,
        const std::size_t max_rank_emission,
        const std::size_t wave_capacity = HIGS_DEPTH_WAVE_INSTANCES) {
        return wave_capacity > max_rank_emission
                   ? 1u + raw_instances / (wave_capacity - max_rank_emission)
                   : 0u;
    }

    struct VisibleSortDispatch {
        inline static constexpr Layout kLayout{
            "LFS_VK_VISIBLE_SORT_DISPATCH_WORD_COUNT",
            LFS_VK_VISIBLE_SORT_DISPATCH_WORD_COUNT};
        inline static constexpr std::size_t kRadixWordOffset =
            LFS_VK_VISIBLE_SORT_DISPATCH_RADIX_WORD_OFFSET;
    };

    struct TileBatchDispatch {
        inline static constexpr Layout kLayout{
            "LFS_VK_TILE_BATCH_DISPATCH_WORD_COUNT",
            LFS_VK_TILE_BATCH_DISPATCH_WORD_COUNT};
        inline static constexpr std::size_t kRasterWordOffset =
            LFS_VK_TILE_BATCH_DISPATCH_RASTER_WORD_OFFSET;
    };

    struct VisibleChainDispatch {
        inline static constexpr Layout kLayout{
            "LFS_VK_VISIBLE_CHAIN_DISPATCH_WORD_COUNT",
            LFS_VK_VISIBLE_CHAIN_DISPATCH_WORD_COUNT};
        inline static constexpr std::size_t kRadixWordOffset =
            LFS_VK_VISIBLE_CHAIN_DISPATCH_RADIX_WORD_OFFSET;
        inline static constexpr std::size_t kPerElementWordOffset =
            LFS_VK_VISIBLE_CHAIN_DISPATCH_PER_ELEMENT_WORD_OFFSET;
        inline static constexpr std::size_t kCumsumLevel0WordOffset =
            LFS_VK_VISIBLE_CHAIN_DISPATCH_CUMSUM_L0_WORD_OFFSET;
        inline static constexpr std::size_t kCumsumLevel1WordOffset =
            LFS_VK_VISIBLE_CHAIN_DISPATCH_CUMSUM_L1_WORD_OFFSET;
    };

    struct SurvivorState {
        inline static constexpr Layout kLayout{
            "LFS_VK_SURVIVOR_STATE_WORD_COUNT",
            LFS_VK_SURVIVOR_STATE_WORD_COUNT};
        inline static constexpr std::size_t kCountWordOffset =
            LFS_VK_SURVIVOR_STATE_COUNT_WORD_OFFSET;
        inline static constexpr std::size_t kProjectionWordOffset =
            LFS_VK_SURVIVOR_STATE_PROJECTION_WORD_OFFSET;
    };

    struct DepthWave {
        inline static constexpr Layout kRecordLayout{
            "LFS_VK_DEPTH_WAVE_RECORD_STRIDE_WORDS",
            LFS_VK_DEPTH_WAVE_RECORD_STRIDE_WORDS};
        inline static constexpr std::size_t kRecordStrideWords =
            LFS_VK_DEPTH_WAVE_RECORD_STRIDE_WORDS;
        inline static constexpr std::size_t kHeaderNeededWord =
            LFS_VK_DEPTH_WAVE_HEADER_NEEDED_WORD;
        inline static constexpr std::size_t kCountWordOffset =
            LFS_VK_DEPTH_WAVE_COUNT_WORD_OFFSET;
        inline static constexpr std::size_t kKeygenWordOffset =
            LFS_VK_DEPTH_WAVE_KEYGEN_WORD_OFFSET;
        inline static constexpr std::size_t kRadixWordOffset =
            LFS_VK_DEPTH_WAVE_RADIX_WORD_OFFSET;
        inline static constexpr std::size_t kRangeWordOffset =
            LFS_VK_DEPTH_WAVE_RANGE_WORD_OFFSET;
        inline static constexpr std::size_t kPerTileWordOffset =
            LFS_VK_DEPTH_WAVE_PER_TILE_WORD_OFFSET;
        inline static constexpr std::size_t kFullscreenWordOffset =
            LFS_VK_DEPTH_WAVE_FULLSCREEN_WORD_OFFSET;
        inline static constexpr std::size_t kRankBaseWord =
            LFS_VK_DEPTH_WAVE_RANK_BASE_WORD;
        inline static constexpr std::size_t kRankCountWord =
            LFS_VK_DEPTH_WAVE_RANK_COUNT_WORD;
        inline static constexpr std::size_t kInstanceBaseWord =
            LFS_VK_DEPTH_WAVE_INSTANCE_BASE_WORD;

        [[nodiscard]] static constexpr Layout layout(const std::size_t armed) {
            return {"DepthWave", (1u + armed) * kRecordStrideWords};
        }

        [[nodiscard]] static constexpr std::size_t recordWordOffset(const std::size_t wave) {
            return (1u + wave) * kRecordStrideWords;
        }

#define LFS_DEPTH_WAVE_ACCESSOR(name, member)                                 \
    [[nodiscard]] static constexpr std::size_t name(const std::size_t wave) { \
        return recordWordOffset(wave) + member;                               \
    }
        LFS_DEPTH_WAVE_ACCESSOR(countWordOffset, kCountWordOffset)
        LFS_DEPTH_WAVE_ACCESSOR(keygenWordOffset, kKeygenWordOffset)
        LFS_DEPTH_WAVE_ACCESSOR(radixWordOffset, kRadixWordOffset)
        LFS_DEPTH_WAVE_ACCESSOR(rangeWordOffset, kRangeWordOffset)
        LFS_DEPTH_WAVE_ACCESSOR(perTileWordOffset, kPerTileWordOffset)
        LFS_DEPTH_WAVE_ACCESSOR(fullscreenWordOffset, kFullscreenWordOffset)
        LFS_DEPTH_WAVE_ACCESSOR(rankBaseWord, kRankBaseWord)
        LFS_DEPTH_WAVE_ACCESSOR(rankCountWord, kRankCountWord)
        LFS_DEPTH_WAVE_ACCESSOR(instanceBaseWord, kInstanceBaseWord)
#undef LFS_DEPTH_WAVE_ACCESSOR
    };

    struct MacroWaveDispatch {
        inline static constexpr Layout kLayout{
            "LFS_VK_MACRO_WAVE_DISPATCH_WORD_COUNT",
            LFS_VK_MACRO_WAVE_DISPATCH_WORD_COUNT};
        inline static constexpr std::size_t kWaveStrideWords =
            LFS_VK_MACRO_WAVE_DISPATCH_WAVE_STRIDE_WORDS;
        inline static constexpr std::size_t kRasterBaseWordOffset =
            LFS_VK_MACRO_WAVE_DISPATCH_RASTER_BASE_WORD_OFFSET;
        inline static constexpr std::size_t kComposeBaseWordOffset =
            LFS_VK_MACRO_WAVE_DISPATCH_COMPOSE_BASE_WORD_OFFSET;

        [[nodiscard]] static constexpr std::size_t rasterWordOffset(const std::size_t wave) {
            return kRasterBaseWordOffset + wave * kWaveStrideWords;
        }

        [[nodiscard]] static constexpr std::size_t composeWordOffset(const std::size_t wave) {
            return kComposeBaseWordOffset + wave * kWaveStrideWords;
        }
    };

    static_assert(sizeof(VkDispatchIndirectCommand) == kCommandWordCount * sizeof(std::uint32_t));
    static_assert(kCommandXWordOffset == 0);
    static_assert(kCommandYWordOffset == kCommandXWordOffset + 1);
    static_assert(kCommandZWordOffset == kCommandYWordOffset + 1);
    static_assert(VisibleSortDispatch::kRadixWordOffset + kCommandWordCount ==
                  VisibleSortDispatch::kLayout.word_count);
    static_assert(TileBatchDispatch::kRasterWordOffset + kCommandWordCount ==
                  TileBatchDispatch::kLayout.word_count);
    static_assert(VisibleChainDispatch::kRadixWordOffset + kCommandWordCount ==
                  VisibleChainDispatch::kPerElementWordOffset);
    static_assert(VisibleChainDispatch::kPerElementWordOffset + kCommandWordCount ==
                  VisibleChainDispatch::kCumsumLevel0WordOffset);
    static_assert(VisibleChainDispatch::kCumsumLevel0WordOffset + kCommandWordCount ==
                  VisibleChainDispatch::kCumsumLevel1WordOffset);
    static_assert(VisibleChainDispatch::kCumsumLevel1WordOffset + kCommandWordCount ==
                  VisibleChainDispatch::kLayout.word_count);
    static_assert(SurvivorState::kCountWordOffset == 0);
    static_assert(SurvivorState::kProjectionWordOffset ==
                  SurvivorState::kCountWordOffset + 1);
    static_assert(SurvivorState::kProjectionWordOffset + kCommandWordCount ==
                  SurvivorState::kLayout.word_count);
    static_assert(DepthWave::kRecordStrideWords * sizeof(std::uint32_t) == 256u);
    static_assert(DepthWave::kHeaderNeededWord == 0u);
    static_assert(DepthWave::kCountWordOffset + 2u <= DepthWave::kKeygenWordOffset);
    static_assert(DepthWave::kKeygenWordOffset + kCommandWordCount ==
                  DepthWave::kRadixWordOffset);
    static_assert(DepthWave::kRadixWordOffset + kCommandWordCount ==
                  DepthWave::kRangeWordOffset);
    static_assert(DepthWave::kRangeWordOffset + kCommandWordCount ==
                  DepthWave::kPerTileWordOffset);
    static_assert(DepthWave::kPerTileWordOffset + kCommandWordCount ==
                  DepthWave::kFullscreenWordOffset);
    static_assert(DepthWave::kInstanceBaseWord < DepthWave::kRecordStrideWords);
    static_assert(MacroWaveDispatch::kComposeBaseWordOffset ==
                  HIGS_RASTER_MAX_WAVES * MacroWaveDispatch::kWaveStrideWords);
    static_assert(MacroWaveDispatch::kComposeBaseWordOffset +
                      HIGS_RASTER_MAX_WAVES * MacroWaveDispatch::kWaveStrideWords ==
                  MacroWaveDispatch::kLayout.word_count);

} // namespace lfs::rendering::vulkan::indirect_layout

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lfs::rendering::metal {
    enum class GpuStage : uint32_t { Projection, Instances, Sort, Blend, Present, Count };

    // Opt-in diagnostics only. Each frame owns its sample buffer until its
    // command completes; resolving never participates in ordinary rendering.
    class GpuProfile {
    public:
        explicit GpuProfile(id<MTLDevice> device) {
            if (@available(macOS 11.0, iOS 14.0, *)) {
                if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
                    return;
                for (id<MTLCounterSet> set in device.counterSets) {
                    if (![set.name isEqualToString:MTLCommonCounterSetTimestamp])
                        continue;
                    auto descriptor = [MTLCounterSampleBufferDescriptor new];
                    descriptor.counterSet = set;
                    descriptor.sampleCount = kSamples;
                    descriptor.storageMode = MTLStorageModeShared;
                    descriptor.label = @"LichtFeld opt-in viewer GPU timestamps";
                    NSError* error = nil;
                    buffer_ = [device newCounterSampleBufferWithDescriptor:descriptor error:&error];
                    if (!buffer_)
                        throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal timestamp allocation failed");
                    break;
                }
            }
        }
        bool available() const { return buffer_ != nil; }
        void reset() { stages_.clear(); }
        id<MTLComputeCommandEncoder> begin(id<MTLCommandBuffer> command, GpuStage stage) {
            if (!available())
                return [command computeCommandEncoder];
            if (stages_.size() * 2 + 2 > kSamples)
                throw std::length_error("Metal viewer GPU timestamp reservation exceeded");
            if (@available(macOS 11.0, iOS 14.0, *)) {
                auto descriptor = [MTLComputePassDescriptor computePassDescriptor];
                auto attachment = descriptor.sampleBufferAttachments[0];
                attachment.sampleBuffer = buffer_;
                attachment.startOfEncoderSampleIndex = stages_.size() * 2;
                attachment.endOfEncoderSampleIndex = stages_.size() * 2 + 1;
                auto encoder = [command computeCommandEncoderWithDescriptor:descriptor];
                if (!encoder)
                    throw std::runtime_error("Metal profiled compute encoder failed");
                stages_.push_back(stage);
                return encoder;
            }
            return [command computeCommandEncoder];
        }
        // Caller must have observed successful completion of the owning command.
        std::array<double, size_t(GpuStage::Count)> resolve() const {
            std::array<double, size_t(GpuStage::Count)> result{};
            if (!available() || stages_.empty())
                return result;
            const auto data = [buffer_ resolveCounterRange:NSMakeRange(0, stages_.size() * 2)];
            if (!data || data.length != stages_.size() * 2 * sizeof(MTLCounterResultTimestamp))
                throw std::runtime_error("Metal GPU timestamp resolution failed");
            const auto timestamps = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
            for (size_t i = 0; i < stages_.size(); ++i) {
                const uint64_t start = timestamps[i * 2].timestamp, end = timestamps[i * 2 + 1].timestamp;
                if (!start || end < start || end == UINT64_MAX)
                    throw std::runtime_error("Invalid Metal GPU timestamp sample");
                result[size_t(stages_[i])] += double(end - start) / 1e6;
            }
            return result;
        }

    private:
        static constexpr NSUInteger kSamples = 256;
        id<MTLCounterSampleBuffer> buffer_ = nil;
        std::vector<GpuStage> stages_;
    };
    inline id<MTLComputeCommandEncoder> profiledCompute(id<MTLCommandBuffer> command,
                                                      GpuProfile* profile, GpuStage stage) {
        return profile ? profile->begin(command, stage) : [command computeCommandEncoder];
    }
} // namespace lfs::rendering::metal

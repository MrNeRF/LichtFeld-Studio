/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../tensor_completion.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "metal_context.hpp"

namespace lfs::core::internal {
    namespace {
        // A Metal-only work queue: the consumer is a native Metal shared event,
        // and completion is the Metal batch serial.
        class API_AVAILABLE(macos(26.0)) MetalWorkQueue final : public MetalVulkanQueue {
        public:
            explicit MetalWorkQueue(void* const consumer)
                : consumer_((__bridge id<MTLSharedEvent>)consumer) {}

            void* timeline() const override { return nullptr; }

            void wait(const uint64_t consumer_value) override {
                if (consumer_)
                    metal::acquire_context()->queue_wait(consumer_, consumer_value);
            }

            TensorCompletion signal() override {
                return TensorCompletionAccess::metal(metal::acquire_context()->flush());
            }

        private:
            id<MTLSharedEvent> consumer_;
        };
    } // namespace

    std::unique_ptr<MetalVulkanQueue> make_metal_work_queue(void* const consumer_event) {
        if (@available(macOS 26.0, *))
            return std::make_unique<MetalWorkQueue>(consumer_event);
        throw TensorError("Metal tensors require macOS 26");
    }
} // namespace lfs::core::internal

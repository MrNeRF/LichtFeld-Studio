/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_module.hpp"

#include "metal_context.hpp"

#include "core/assert.hpp"
#include "core/error.hpp"

#include <format>
#include <map>
#include <mutex>
#include <vector>

namespace lfs::core::internal {
    namespace {
        API_AVAILABLE_BEGIN(macos(26.0))

        class Module final : public MetalModule {
        public:
            Module(std::string source, const bool fast_math)
                : source_(std::move(source)),
                  fast_math_(fast_math) {}

            uint64_t address(const StorageRef& storage) override {
                const auto at = metal::acquire_context()->locate(storage);
                return at.address + at.offset;
            }

            void launch(const std::string_view function, const std::span<const std::pair<uint32_t, uint32_t>> constants,
                        const std::span<const StorageRef> uses, const std::span<const std::byte> params,
                        const std::array<uint32_t, 3> groups, const std::array<uint32_t, 3> group) override {
                if (groups[0] == 0 || groups[1] == 0 || groups[2] == 0)
                    return;
                const auto context = metal::acquire_context();
                const auto state = pipeline(context, function, constants);
                context->dispatch(uses, {.pipeline = state,
                                          .buffers = {},
                                          .params = params,
                                          .grid = MTLSizeMake(groups[0], groups[1], groups[2]),
                                          .group_size = MTLSizeMake(group[0], group[1], group[2])});
            }

        private:
            id<MTLComputePipelineState> pipeline(const std::shared_ptr<metal::Context>& context,
                                                 const std::string_view function,
                                                 const std::span<const std::pair<uint32_t, uint32_t>> constants) {
                std::vector<std::pair<uint32_t, uint32_t>> key_constants(constants.begin(), constants.end());
                auto key = std::pair{std::string(function), std::move(key_constants)};
                std::lock_guard lock(mutex_);
                // Cached trainer modules outlive backend shutdown; their launches
                // must follow the reopened tensor context's submission timeline.
                if (context_id_ != context->context_id()) {
                    pipelines_.clear();
                    library_ = nil;
                    context_id_ = context->context_id();
                }
                if (const auto found = pipelines_.find(key); found != pipelines_.end())
                    return found->second;
                if (!library_) {
                    MTLCompileOptions* const options = [MTLCompileOptions new];
                    options.languageVersion = MTLLanguageVersion4_0;
                    options.mathMode = fast_math_ ? MTLMathModeFast : MTLMathModeSafe;
                    NSError* error = nil;
                    library_ = [context->device() newLibraryWithSource:@(source_.c_str()) options:options error:&error];
                    if (!library_)
                        throw TensorError(std::format("Metal kernel module failed to compile: {}",
                                                      error ? error.localizedDescription.UTF8String : "unknown error"));
                }
                MTLFunctionConstantValues* const values = [MTLFunctionConstantValues new];
                for (const auto& [index, value] : key.second)
                    [values setConstantValue:&value type:MTLDataTypeUInt atIndex:index];
                NSError* error = nil;
                NSString* const name = [[NSString alloc] initWithBytes:function.data()
                                                                length:function.size()
                                                              encoding:NSUTF8StringEncoding];
                id<MTLFunction> const kernel = [library_ newFunctionWithName:name constantValues:values error:&error];
                id<MTLComputePipelineState> const state =
                    kernel ? [context->device() newComputePipelineStateWithFunction:kernel error:&error] : nil;
                if (!state)
                    throw TensorError(std::format("Metal kernel '{}' failed: {}", function,
                                                  error ? error.localizedDescription.UTF8String : "unknown error"));
                pipelines_.emplace(std::move(key), state);
                return state;
            }

            uint64_t context_id_ = 0;
            std::string source_;
            bool fast_math_;
            std::mutex mutex_;
            id<MTLLibrary> library_;
            std::map<std::pair<std::string, std::vector<std::pair<uint32_t, uint32_t>>>, id<MTLComputePipelineState>>
                pipelines_;
        };

        API_AVAILABLE_END
    } // namespace

    std::unique_ptr<MetalModule> make_metal_module(std::string source, const bool fast_math) {
        if (@available(macOS 26.0, *)) {
            if (metal_backend_available())
                return std::make_unique<Module>(std::move(source), fast_math);
        }
        throw TensorError("Metal kernel modules need macOS 26 and a Metal 4 GPU");
    }

} // namespace lfs::core::internal

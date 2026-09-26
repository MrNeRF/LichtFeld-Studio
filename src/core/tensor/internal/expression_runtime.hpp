/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "expression_program.hpp"

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lfs::core::internal {

    enum class ExprStride : uint8_t { Zero,
                                      Unit,
                                      Strided };

    // Code generation and the cache key read only this launch class. Sizes, strides and
    // addresses are arguments, except for the bounded short-fold variants.
    struct ExpressionSignature {
        struct Binding {
            DataType dtype = DataType::Float32;
            bool host = false;
            // Indexed by the output element id: contiguous over the domain without the fold
            // dimension and constant along it.
            bool linear = false;
            // CUDA: the device input four bytes after this one, loaded with it as one pair.
            int8_t pair = -1;
            std::array<ExprStride, ExpressionProgram::max_rank> strides{};
        };
        uint32_t rank = 1;
        int32_t fold_dim = -1;
        // 1..64 is part of the layout class so the fold unrolls; 0 keeps a runtime trip count.
        uint32_t fold_length = 0;
        // Every gather index is proven in range for this binding, so the kernel omits bounds.
        bool gather_in_range = false;
        uint32_t inputs = 0, outputs = 0;
        std::array<Binding, ExpressionProgram::max_inputs> input{};
        std::array<Binding, ExpressionProgram::max_outputs> output{};
    };

    // Word offsets into the launch argument block shared by every emitter and launcher. The
    // 64-bit addresses come first. Host inputs are packed densely into the bank in binding
    // order; the first starts at the bank and each later one has a word holding its start.
    // Strides are present for Strided classes, bounds for gathered dimensions and magic
    // divisors for every domain dimension decomposed from the element id.
    struct ExpressionLayout {
        static constexpr uint32_t none = ~0u;
        static constexpr uint32_t max_words = 256;
        uint32_t words = 0;
        uint32_t count = none;
        uint32_t bank = none;
        std::array<uint32_t, ExpressionProgram::max_rank> dims{none, none, none, none};
        std::array<uint32_t, ExpressionProgram::max_rank> magic{none, none, none, none};
        std::array<uint32_t, ExpressionProgram::max_rank> shift{none, none, none, none};
        std::array<uint32_t, ExpressionProgram::max_inputs> input{};
        std::array<std::array<uint32_t, ExpressionProgram::max_rank>, ExpressionProgram::max_inputs> input_stride{};
        std::array<std::array<uint32_t, ExpressionProgram::max_rank>, ExpressionProgram::max_inputs> bound{};
        std::array<uint32_t, ExpressionProgram::max_outputs> output{};
        std::array<std::array<uint32_t, ExpressionProgram::max_rank>, ExpressionProgram::max_outputs> output_stride{};
    };

    static_assert(ExpressionLayout::max_words * sizeof(uint32_t) <= 4096);
    static_assert(ExpressionLayout::max_words * sizeof(uint32_t) <= 65536);

    LFS_CORE_API ExpressionLayout expression_layout(const ExpressionProgram& program,
                                                    const ExpressionSignature& signature);
    // True when integer interval analysis proves every gather index lies inside its bound.
    LFS_CORE_API bool expression_gather_in_range(const ExpressionProgram& program,
                                                 const ExprShape& shape,
                                                 const std::array<std::array<uint32_t, ExpressionProgram::max_rank>,
                                                                  ExpressionProgram::max_inputs>& bounds);
    // Output elements per 32-bit word of the narrowest output.
    uint32_t expression_packing(const ExpressionSignature& signature);

    // Invariant values are constant along the fold dimension, Variant values change along it
    // and Folded values depend on a fold result.
    enum class ExprPhase : uint8_t { Invariant,
                                     Variant,
                                     Folded };
    // The instruction defining each operand register of every instruction.
    std::span<const std::array<uint32_t, 3>> expression_sources(const ExpressionProgram& program);
    // Throws when a value depends on the fold both inside and after it, or a store varies
    // along the fold dimension.
    std::vector<ExprPhase> expression_phases(const ExpressionProgram& program,
                                             const ExpressionSignature& signature);

    struct ExpressionLaunch {
        const ExpressionProgram* program = nullptr;
        bool prepare_only = false;
        ExpressionSignature signature;
        uint32_t count = 0;
        std::array<uint32_t, ExpressionLayout::max_words> arguments{};
        uint32_t words = 0;
        std::array<StorageRef, ExpressionProgram::max_inputs> reads{};
        uint32_t read_count = 0;
        std::array<StorageRef, ExpressionProgram::max_outputs> writes{};
        uint32_t write_count = 0;
    };

    std::string expression_key(const ExpressionProgram& program, const ExpressionSignature& signature);

    struct CompiledExpression {
        virtual ~CompiledExpression() = default;
        size_t bytes = 0;
    };

    struct ExpressionCacheStats {
        uint64_t hits = 0, disk_hits = 0, compilations = 0, evictions = 0, loads = 0;
        double compile_ms = 0;
        size_t entries = 0, bytes = 0;
    };

    class LFS_CORE_API ExpressionCache {
    public:
        using Compile = std::function<std::vector<char>()>;
        using Load = std::function<std::shared_ptr<CompiledExpression>(std::span<const char>)>;

        std::shared_ptr<CompiledExpression> get(const std::string& key,
                                                const Compile& compile, const Load& load);
        [[nodiscard]] ExpressionCacheStats stats() const;
        void clear();

    private:
        struct Entry {
            std::shared_ptr<CompiledExpression> kernel;
            std::list<std::string>::iterator position;
        };
        mutable std::mutex mutex_;
        std::list<std::string> lru_;
        std::unordered_map<std::string, Entry> entries_;
        ExpressionCacheStats stats_;
    };

    class RuntimeLibrary {
    public:
        explicit RuntimeLibrary(const char* name);
        ~RuntimeLibrary();
        RuntimeLibrary(const RuntimeLibrary&) = delete;
        RuntimeLibrary& operator=(const RuntimeLibrary&) = delete;
        [[nodiscard]] void* symbol(const char* name) const;

    private:
        void* handle_;
    };

    std::string_view expression_emitter_hash();

    LFS_CORE_API ExpressionCacheStats expression_cache_stats(GpuBackend backend);
    ExpressionCache& cuda_expression_cache();
    // Loading a new CUDA program synchronizes the context, so specializations must stay few.
    // Runs a device-wide synchronization, then unloads modules evicted before it.
    void synchronize_and_unload_cuda_expressions(const std::function<void()>& synchronize);

} // namespace lfs::core::internal

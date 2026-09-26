/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#define SPV_ENABLE_UTILITY_CODE
#include <array>
#include <cstdint>
#include <set>
#include <span>
#include <spirv/unified1/spirv.hpp>
#include <string>
#include <vector>

namespace lfs::core::internal::spirv {
    struct Instruction {
        spv::Op opcode;
        std::vector<uint32_t> operands;
    };

    struct Module {
        std::array<uint32_t, 5> header{};
        std::vector<Instruction> instructions;
    };

    struct Analysis {
        std::vector<spv::Capability> capabilities;
        std::vector<std::string> extensions;
        uint32_t entry_id = 0;
        std::string entry_name;
        spv::ExecutionModel execution_model = spv::ExecutionModelMax;
        std::array<uint32_t, 3> local_size{};
        std::set<uint32_t> float_widths;
        std::set<uint32_t> preserve_widths;
        uint32_t push_constant_size = 0;
    };

    Module parse(const std::vector<uint32_t>& words);
    std::vector<uint32_t> serialize(const Module& module);
    Analysis analyze(const Module& module);
    std::vector<uint32_t> finalize(std::span<const uint32_t> words);
} // namespace lfs::core::internal::spirv

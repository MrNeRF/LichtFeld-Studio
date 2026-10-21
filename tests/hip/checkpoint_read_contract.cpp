/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../checkpoint_fixture.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/strategies/mcmc.hpp"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace {
    using namespace lfs;
    using namespace lfs::core;

    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    std::unique_ptr<SplatData> model(size_t count) {
        std::vector<float> means(count * 3, 0.f), rotations(count * 4, 0.f);
        for (size_t i = 0; i < count; ++i) {
            means[i * 3] = float(i);
            rotations[i * 4] = 1.f;
        }
        return std::make_unique<SplatData>(0,
            Tensor::from_vector(means, {count, size_t{3}}, Device::CPU),
            Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CPU),
            Tensor::zeros({size_t{0}}, Device::CPU),
            Tensor::zeros({count, size_t{3}}, Device::CPU),
            Tensor::from_vector(rotations, {count, size_t{4}}, Device::CPU),
            Tensor::zeros({count, size_t{1}}, Device::CPU), 1.f);
    }

    // Same LFAS fixture format exercised by CheckpointLegacySparsityTest upstream.
    // Product checkpoint readers and migration code are not changed by this test.
    void make_old_checkpoint(const std::filesystem::path& path, size_t rows, int iteration) {
        std::ifstream input(path, std::ios::binary);
        require(bool(input), "Cannot open checkpoint fixture");
        const std::string bytes((std::istreambuf_iterator<char>(input)), {});
        input.close();
        require(bytes.size() >= sizeof(CheckpointHeader), "Checkpoint fixture is truncated");
        CheckpointHeader header;
        std::memcpy(&header, bytes.data(), sizeof(header));
        auto params = nlohmann::json::parse(bytes.substr(header.params_json_offset, header.params_json_size));
        auto& opt = params["optimization"];
        opt["enable_sparsity"] = true;
        opt["sparsity_method"] = "opacity_admm";
        for (auto it = opt.begin(); it != opt.end();)
            it = it.key().starts_with("popspa_") ? opt.erase(it) : std::next(it);
        opt["sparsify_steps"] = 15000;
        opt["init_rho"] = .0005f;
        opt["prune_ratio"] = .6f;
        std::string out = bytes.substr(0, header.params_json_offset);
        if (rows) {
            std::ostringstream payload(std::ios::binary);
            constexpr uint32_t magic = 0x4C464153, version = 1;
            payload.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            payload.write(reinterpret_cast<const char*>(&version), sizeof(version));
            const auto state = Tensor::full({rows, size_t{1}}, .25f, Device::CPU);
            payload << state << state << state;
            out += payload.str();
            header.flags = header.flags | CheckpointFlags::HAS_SPARSITY;
        }
        const auto json = params.dump();
        header.iteration = iteration;
        header.params_json_offset = out.size();
        header.params_json_size = json.size();
        out += json;
        std::memcpy(out.data(), &header, sizeof(header));
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(out.data(), std::streamsize(out.size()));
        output.close();
        require(bool(output), "Cannot finish old-format fixture");
    }

    void check_case(const std::filesystem::path& root, const char* name, size_t rows,
                    int iteration, bool expected_success, bool expected_sparsity) {
        const auto directory = root / name;
        require(!std::filesystem::exists(directory), "Test output must be fresh");
        param::TrainingParameters params;
        params.dataset.output_path = directory;
        params.optimization.strategy = "mcmc";
        params.optimization.iterations = 20;
        params.optimization.sh_degree = 0;
        params.optimization.max_cap = 16;
        auto source = model(4);
        training::MCMC source_strategy(*source);
        auto written = test::write_checkpoint_fixture(directory, 7, source_strategy, params, nullptr, nullptr, nullptr);
        if (!written)
            throw std::runtime_error(written.error());
        const auto path = test::checkpoint_fixture_path(directory);
        make_old_checkpoint(path, rows, iteration);

        auto destination = model(1);
        training::MCMC strategy(*destination);
        auto loaded_params = params;
        const auto loaded = training::load_checkpoint(path, strategy, loaded_params, nullptr, nullptr, nullptr);
        if (expected_success) {
            if (!loaded)
                throw std::runtime_error(loaded.error());
            require(*loaded == iteration && strategy.get_model().size() == 4, "Old checkpoint model/iteration changed");
            require(loaded_params.optimization.enable_sparsity == expected_sparsity, "Old sparsity settings migrated incorrectly");
            const auto means = strategy.get_model().means().cpu().to_vector();
            require(means.size() == 12, "Loaded model shape changed");
            for (size_t i = 0; i < means.size(); ++i)
                require(means[i] == (i % 3 == 0 ? float(i / 3) : 0.f), "Old checkpoint model values changed");
            if (expected_sparsity) {
                require(loaded_params.optimization.resolved_total_iterations() == 10020, "Old pre-tail settings did not migrate to POPSpa");
                const auto unavailable = training::unavailable_training_reason(loaded_params, GpuBackend::HIP, {});
                if (unavailable)
                    throw std::runtime_error("Migrated POPSpa checkpoint is missing native HIP training families: " + *unavailable);
            }
            written = test::write_checkpoint_fixture(directory, iteration, strategy, loaded_params, nullptr, nullptr, nullptr);
            if (!written)
                throw std::runtime_error(written.error());
            const auto header = load_checkpoint_header(path);
            require(header.has_value() && !has_flag(header->flags, CheckpointFlags::HAS_SPARSITY), "Rewritten checkpoint retained old ADMM state");
        } else {
            require(!loaded && loaded.error().find("does not match model count") != std::string::npos,
                    "Invalid old checkpoint row count was accepted or failed for a different reason");
            require(strategy.get_model().size() == 1, "Failed read mutated destination model");
        }
        require(configured_gpu_backend() == GpuBackend::HIP, "Checkpoint reading changed backend selection");
        std::printf("PASS HIP checkpoint read: %s\n", name);
    }
}

int main(int argc, char** argv) try {
    require(argc == 2 && std::filesystem::path(argv[1]).is_absolute(), "Pass a fresh absolute result directory");
    require(bool(set_default_gpu_backend(GpuBackend::HIP)), "Cannot select explicit HIP backend");
    const std::filesystem::path root(argv[1]);
    check_case(root, "old-state-consumed", 4, 7, true, false);
    check_case(root, "pre-tail-migrates-popspa", 0, 7, true, true);
    check_case(root, "wrong-row-count-rejected", 3, 7, false, false);
    check_case(root, "during-tail-sparsity-disabled", 0, 21, true, false);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP CHECKPOINT READ FAILED: %s\n", error.what());
    return 1;
}

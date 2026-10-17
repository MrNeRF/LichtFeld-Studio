/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "formats/ply.hpp"
#include "gui/utils/splat_transfer.hpp"
#include <cmath>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <stdexcept>
using namespace lfs::core;
static void require(bool yes, const char* message) {
    if (!yes)
        throw std::runtime_error(message);
}
static void checkNear(float a, float b) { require(std::abs(a - b) < 2e-5f, "Numeric mismatch"); }
static SplatData fixture(int degree) {
    const size_t k = (degree + 1) * (degree + 1) - 1;
    auto means = Tensor::from_vector(std::vector<float>{.3f, 1.2f, -.7f, 7, 8, 9, 2.4f, -.5f, 1.8f}, {3, 3}, Device::CPU);
    auto dc = Tensor::from_vector(std::vector<float>{.1f, .2f, -.3f, .4f, .5f, .6f, .7f, .8f, .9f}, {3, 1, 3}, Device::CPU);
    std::vector<float> coefficients(3 * k * 3);
    for (size_t i = 0; i < coefficients.size(); ++i)
        coefficients[i] = .01f * float(int(i % 17) - 8);
    auto rest = k ? Tensor::from_vector(coefficients, {3, k, 3}, Device::CPU) : Tensor{};
    auto scale = Tensor::from_vector(std::vector<float>{-2, -3, -4, -2, -3, -4, -2, -3, -4}, {3, 3}, Device::CPU);
    auto rotation = Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {3, 4}, Device::CPU);
    auto opacity = Tensor::from_vector(std::vector<float>{-1, 0, 1}, {3, 1}, Device::CPU);
    SplatData model(degree, std::move(means), std::move(dc), std::move(rest), std::move(scale), std::move(rotation), std::move(opacity), 1.0f);
    model.deleted() = Tensor::from_vector(std::initializer_list<bool>{false, true, false}, {3}, Device::CPU);
    model.notify_deleted_mask_changed();
    model.refresh_deleted_count();
    return model;
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Pass output manifest path");
        lfs::io::LoadOptions options;
        options.splat_tensor_allocator = [](TensorShape shape, size_t, DataType type, std::string_view) { return Tensor::empty(std::move(shape), Device::CPU, type); };
        std::ofstream manifest(argv[1]);
        manifest << "{\"exports\":[";
        for (int degree = 0; degree <= 3; ++degree) {
            auto model = fixture(degree);
            const auto original_rest = model.shN_canonical_cpu();
            model.set_active_sh_degree(0); // Export all stored SH even if the viewport displays only DC.
            auto world = glm::translate(glm::mat4(1), glm::vec3(1, 2, 3)) * glm::scale(glm::mat4(1), glm::vec3(2));
            auto future = lfs::vis::gui::prepareSplatTransfer(model, world, std::make_shared<std::atomic<bool>>(false));
            model.means().fill_(999);
            const auto path = future.get();
            require(std::filesystem::is_regular_file(path), "No committed PLY");
            for (const auto& entry : std::filesystem::directory_iterator(path.parent_path()))
                require(entry.path().filename().string().find(".partial.") == std::string::npos, "Partial remains");
            auto loaded = lfs::io::load_ply(path, options);
            if (!loaded)
                throw std::runtime_error(lfs::format_for_developer(loaded.error()));
            auto& out = loaded->value;
            require(out.size() == 2, "Deleted mask not preserved");
            auto pos = out.means().to(Device::CPU).contiguous();
            const auto* p = pos.ptr<float>();
            const float expected[] = {1.6f, 1.6f, -4.4f, 5.8f, 6.6f, -1.0f};
            for (int i = 0; i < 6; ++i)
                checkNear(p[i], expected[i]);
            auto opacity = out.opacity_raw().to(Device::CPU).contiguous();
            checkNear(opacity.ptr<float>()[0], -1);
            checkNear(opacity.ptr<float>()[1], 1);
            auto dc = out.sh0().to(Device::CPU).contiguous();
            checkNear(dc.ptr<float>()[0], .1f);
            checkNear(dc.ptr<float>()[3], .7f);
            require(out.get_active_sh_degree() == degree, "SH degree changed");
            auto scales = out.scaling_raw().to(Device::CPU).contiguous();
            auto rotations = out.rotation_raw().to(Device::CPU).contiguous();
            const auto* q = rotations.ptr<float>();
            const auto r = glm::mat3_cast(glm::normalize(glm::quat(q[0], q[1], q[2], q[3])));
            glm::mat3 diag(0);
            for (int i = 0; i < 3; ++i)
                diag[i][i] = std::exp(2 * scales.ptr<float>()[i]);
            const auto covariance = r * diag * glm::transpose(r);
            checkNear(covariance[0][0], 4 * std::exp(-4.0f));
            checkNear(covariance[1][1], 4 * std::exp(-8.0f));
            checkNear(covariance[2][2], 4 * std::exp(-6.0f));
            if (degree == 1) {
                auto sh = out.shN_canonical_cpu();
                const float* a = sh.ptr<float>();
                const float* b = original_rest.ptr<float>();
                for (int c = 0; c < 3; ++c) {
                    checkNear(a[c], -b[3 + c]);
                    checkNear(a[3 + c], b[c]);
                    checkNear(a[6 + c], b[6 + c]);
                }
            }
            if (degree)
                manifest << ',';
            manifest << "{\"degree\":" << degree << ",\"path\":\"" << path.generic_string() << "\",\"points\":2}";
            auto cancelled = std::make_shared<std::atomic<bool>>(true);
            bool rejected = false;
            try {
                lfs::vis::gui::prepareSplatTransfer(fixture(degree), glm::mat4(1), cancelled).get();
            } catch (const std::exception&) { rejected = true; }
            require(rejected, "Cancellation ignored");
        }
        {
            auto lod = fixture(0);
            lod.lod_tree = std::make_unique<SplatLodTree>();
            lod.lod_tree->child_count = {2, 0, 0};
            lod.lod_tree->child_start = {1, 0, 0};
            lod.lod_tree->lod_level = {1, 0, 0};
            const auto path = lfs::vis::gui::prepareSplatTransfer(lod, glm::mat4(1), std::make_shared<std::atomic<bool>>(false)).get();
            auto loaded = lfs::io::load_ply(path, options);
            require(bool(loaded), "LOD leaf PLY failed");
            require(loaded->value.size() == 1, "LOD interior nodes or deleted leaves were exported");
            const auto means = loaded->value.means().to(Device::CPU).contiguous();
            checkNear(means.ptr<float>()[0], 2.4f);
            checkNear(means.ptr<float>()[1], 1.8f);
            checkNear(means.ptr<float>()[2], .5f);
        }
        {
            auto model = fixture(3);
            auto cancel = std::make_shared<std::atomic<bool>>(false);
            const auto path = lfs::vis::gui::prepareSplatTransfer(model, glm::mat4(1), cancel,
                                                                  {.format = "ply", .name = "../../CON:hello", .sh_degree = 1, .z_up = false})
                                  .get();
            require(path.parent_path().parent_path().filename() == "lichtfeld-transfers", "Name escaped transfer folder");
            auto loaded = lfs::io::load_ply(path, options);
            require(bool(loaded), "Custom PLY failed");
            require(loaded->value.get_active_sh_degree() == 1, "Custom SH degree ignored");
            auto pos = loaded->value.means().to(Device::CPU).contiguous();
            checkNear(pos.ptr<float>()[1], 1.2f);
            const auto compressed = lfs::vis::gui::prepareSplatTransfer(fixture(0), glm::mat4(1), cancel,
                                                                        {.format = "spz", .name = "test", .z_up = false})
                                        .get();
            require(std::filesystem::file_size(compressed) > 32, "No SPZ exported");
            require(compressed.extension() == ".spz", "Wrong SPZ extension");
            const auto sog = lfs::vis::gui::prepareSplatTransfer(fixture(0), glm::mat4(1), cancel,
                                                                 {.format = "sog", .name = "test", .sog_iterations = 2, .z_up = false})
                                 .get();
            require(std::filesystem::file_size(sog) > 32, "No SOG exported");
        }
        manifest << "],\"snapshot_deleted_world_covariance_sh1_cancel_lod\":\"passed\"}";
        manifest.close();
        std::cout << "PASS snapshot, deleted mask, world axes, covariance, opacity, SH0-3 and cancellation\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

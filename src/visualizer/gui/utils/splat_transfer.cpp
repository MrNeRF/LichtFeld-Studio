/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_transfer.hpp"
#include <algorithm>
#include <cmath>
#include <core/path_utils.hpp>
#include <core/splat_data_transform.hpp>
#include <core/uuid.hpp>
#include <io/exporter.hpp>
#include <stdexcept>
namespace lfs::vis::gui {
    std::future<std::filesystem::path> prepareSplatTransfer(
        const core::SplatData& model, glm::mat4 world,
        std::shared_ptr<std::atomic<bool>> cancelled, SplatTransferOptions options) {
        if (!cancelled)
            throw std::runtime_error("Missing cancellation state");
        if (cancelled->load())
            throw std::runtime_error("Splat preparation cancelled");
        if (model.lod_tree && model.lod_tree->has_tree()) {
            auto leaves = core::extract_lod_leaves(model);
            if (!leaves)
                throw std::runtime_error(lfs::format_for_developer(leaves.error()));
            return prepareSplatTransfer(*leaves, world, cancelled, options);
        }
        if (model.size() == 0)
            throw std::runtime_error("The splat object is empty");
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                if (!std::isfinite(world[col][row]))
                    throw std::runtime_error("Non-finite object transform");
        const glm::mat3 linear(world);
        const float s2 = glm::dot(linear[0], linear[0]);
        if (!std::isfinite(s2) || s2 <= 0 || std::abs(glm::determinant(linear)) < 1e-12f)
            throw std::runtime_error("Invalid or singular object transform");
        if (model.get_max_sh_degree() > 0) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (std::abs(glm::dot(linear[i], linear[j]) - (i == j ? s2 : 0)) > 1e-6f * s2)
                        throw std::runtime_error("SH splat drag currently requires positive uniform object scale");
            if (glm::determinant(linear) <= 0)
                throw std::runtime_error("SH splat drag currently requires a proper rotation");
        }
        const auto cpu = [](const core::Tensor& t) {
            return t.is_valid() ? t.to(core::Device::CPU).contiguous().clone() : core::Tensor{};
        };
        const int degree = options.sh_degree < 0 ? model.get_max_sh_degree() : std::clamp(options.sh_degree, 0, model.get_max_sh_degree());
        auto sh = cpu(model.shN_canonical_cpu());
        const int coefficients = (degree + 1) * (degree + 1) - 1;
        if (sh.is_valid() && coefficients > 0)
            sh = sh.slice(1, 0, coefficients).contiguous();
        else
            sh = core::Tensor::empty({static_cast<size_t>(model.size()), 0, 3}, core::Device::CPU, core::DataType::Float32);
        auto snapshot = core::SplatData::from_raw_tensors(
            degree, degree, model.get_scene_scale(),
            cpu(model.means()), cpu(model.sh0()), std::move(sh),
            cpu(model.scaling_raw()), cpu(model.rotation_raw()), cpu(model.opacity_raw()),
            cpu(model.deleted()), {}, {});
        if (!snapshot)
            throw std::runtime_error(lfs::format_for_developer(snapshot.error()));
        if (options.format != "ply" && options.format != "spz" && options.format != "sog")
            throw std::runtime_error("Unsupported splat drag format");
        const auto id = core::generate_uuid_v4().to_string();
        // A portable file name: never interpret node labels as directories.
        for (auto& c : options.name)
            if (static_cast<unsigned char>(c) < 32 || std::string_view("<>:\"/\\|?*").find(c) != std::string_view::npos)
                c = '_';
        while (!options.name.empty() && (options.name.back() == '.' || options.name.back() == ' '))
            options.name.pop_back();
        if (options.name.empty())
            options.name = "splat";
        if (options.name.size() > 120)
            options.name = id.substr(0, 12);
        // Prefix also avoids Windows reserved device names such as CON and AUX.
        options.name = "splat-" + options.name;
        return std::async(std::launch::async,
                          [model = std::move(*snapshot), world, cancelled, id, options] {
                              const auto folder = std::filesystem::temp_directory_path() / "lichtfeld-transfers" / id;
                              std::filesystem::create_directories(folder);
                              const auto target = folder / core::utf8_to_path(options.name + "." + options.format);
                              const auto staged = folder / core::utf8_to_path(options.name + ".partial." + options.format);
                              try {
                                  if (cancelled->load())
                                      throw std::runtime_error("Splat preparation cancelled");
                                  // Raw scene data uses RDF axes. Blender uses +Z up: (x, y, z) -> (x, z, -y).
                                  // Transform covariance and SH together with positions.
                                  glm::mat4 axes(1.0f);
                                  axes[1] = {0, 0, -1, 0};
                                  axes[2] = {0, 1, 0, 0};
                                  core::transform(*model, (options.z_up ? axes : glm::mat4(1.0f)) * world);
                                  const auto progress = [cancelled](float, const std::string&) { return !cancelled->load(); };
                                  io::Result<void> result;
                                  if (options.format == "ply")
                                      result = io::save_ply(*model, {.output_path = staged, .binary = true, .progress_callback = progress});
                                  else if (options.format == "spz")
                                      result = io::save_spz(*model, {.output_path = staged, .version = 4, .progress_callback = progress});
                                  else
                                      result = io::save_sog(*model, {.output_path = staged,
                                                                     .kmeans_iterations = std::clamp(options.sog_iterations, 1, 100),
                                                                     .use_gpu = false,
                                                                     .progress_callback = progress});
                                  if (!result)
                                      throw std::runtime_error(result.error().format());
                                  if (cancelled->load())
                                      throw std::runtime_error("Splat preparation cancelled");
                                  std::filesystem::rename(staged, target);
                                  return target;
                              } catch (...) {
                                  std::error_code ec;
                                  std::filesystem::remove(staged, ec);
                                  std::filesystem::remove(folder, ec);
                                  throw;
                              }
                          });
    }
} // namespace lfs::vis::gui

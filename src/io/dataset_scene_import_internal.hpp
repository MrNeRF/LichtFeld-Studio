#pragma once
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/error.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/provenance.hpp"
#include "core/scene.hpp"
#include "core/sh_value_quant.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "core/source_site.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/training_normal_priors.hpp"
#include "io/dataset_scene_import.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "io/project_document.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <variant>

namespace lfs::training {
    namespace {
        std::shared_ptr<lfs::core::PointCloud> createRandomPointCloud() {
            constexpr size_t N = 10000;
            auto positions = lfs::core::Tensor::rand({N, 3}, lfs::core::Device::CPU) * 2.0f - 1.0f;
            auto colors = lfs::core::Tensor::randint({N, 3}, 0, 256, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            return std::make_shared<lfs::core::PointCloud>(positions, colors);
        }

        void applyTrainingSHDegree(lfs::core::SplatData& splat, const int target_degree) {
            const int before = splat.get_max_sh_degree();
            if (splat.set_sh_degree(target_degree)) {
                LOG_INFO("Adjusted training model SH degree: {} -> {}", before, splat.get_max_sh_degree());
            }
            if (splat.get_max_sh_degree() > 0 && splat.get_active_sh_degree() != 0) {
                const int active_before = splat.get_active_sh_degree();
                splat.set_active_sh_degree(0);
                LOG_INFO("Training SH schedule active degree: {} -> 0 (max {})",
                         active_before, splat.get_max_sh_degree());
            }
        }

        lfs::Error initFileError(std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Training,
                .user_message = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        bool isPlainPointCloudPly(const std::filesystem::path& path) {
            return path.extension().string() == ".ply" && !lfs::io::is_gaussian_splat_ply(path);
        }

        void centerInitializationMeans(lfs::core::Tensor& means, const glm::vec3& origin) {
            if (origin == glm::vec3{0.0f}) {
                return;
            }
            const auto shift = lfs::core::Tensor::from_vector(
                std::vector<float>{origin.x, origin.y, origin.z}, {3}, means.device());
            means = means - shift;
        }

    } // namespace
} // namespace lfs::training

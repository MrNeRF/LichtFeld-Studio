/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/sidecar_discovery.hpp"

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/filesystem_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <iterator>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace lfs::io {

    namespace {

        [[nodiscard]] bool equals_ascii_ci(const std::string_view a, const std::string_view b) {
            if (a.size() != b.size()) {
                return false;
            }
            for (size_t i = 0; i < a.size(); ++i) {
                const auto ca = static_cast<unsigned char>(a[i]);
                const auto cb = static_cast<unsigned char>(b[i]);
                const char la = (ca >= 'A' && ca <= 'Z') ? static_cast<char>(ca - 'A' + 'a') : static_cast<char>(ca);
                const char lb = (cb >= 'A' && cb <= 'Z') ? static_cast<char>(cb - 'A' + 'a') : static_cast<char>(cb);
                if (la != lb) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] std::string to_generic_utf8(const std::filesystem::path& p) {
            const auto utf8 = p.generic_u8string();
            return {utf8.begin(), utf8.end()};
        }

        [[nodiscard]] bool is_within_root(const std::filesystem::path& rel) {
            if (rel.empty()) {
                return false;
            }
            const auto first = *rel.begin();
            return first != ".." && first != ".";
        }

        [[nodiscard]] std::filesystem::path relative_image_subpath(
            const std::filesystem::path& image_path,
            const std::filesystem::path& normalized_root,
            const std::filesystem::path& canonical_root) {
            if (image_path.empty() || normalized_root.empty()) {
                return {};
            }

            std::error_code ec;
            const std::filesystem::path abs_image = image_path.is_absolute()
                                                        ? image_path.lexically_normal()
                                                        : std::filesystem::absolute(image_path, ec).lexically_normal();
            if (!ec) {
                if (auto rel = abs_image.lexically_relative(normalized_root); is_within_root(rel)) {
                    return rel;
                }
            }

            if (!canonical_root.empty()) {
                std::error_code canon_ec;
                const auto canon_image = std::filesystem::weakly_canonical(abs_image, canon_ec);
                if (!canon_ec) {
                    if (auto rel = canon_image.lexically_relative(canonical_root); is_within_root(rel)) {
                        return rel;
                    }
                }
            }
            return {};
        }

        [[nodiscard]] std::filesystem::path drop_leading_component(const std::filesystem::path& rel) {
            std::filesystem::path stripped;
            for (auto it = std::next(rel.begin()); it != rel.end(); ++it) {
                stripped /= *it;
            }
            return stripped;
        }

        [[nodiscard]] std::vector<std::string> build_camera_lookup_candidates(
            const lfs::core::Camera& cam,
            const std::filesystem::path& normalized_root,
            const std::filesystem::path& canonical_root) {
            std::vector<std::string> candidates;
            std::unordered_set<std::string> seen;

            const auto append_candidate = [&](const std::string& candidate) {
                if (!candidate.empty() && seen.insert(candidate).second) {
                    candidates.push_back(candidate);
                }
            };

            const std::string& image_name = cam.image_name();
            const std::filesystem::path rel =
                relative_image_subpath(cam.image_path(), normalized_root, canonical_root);

            std::string rel_str;
            std::string rel_without_images;
            std::string rel_without_first;
            if (!rel.empty()) {
                rel_str = to_generic_utf8(rel);
                const auto components = static_cast<size_t>(std::distance(rel.begin(), rel.end()));
                if (components >= 2) {
                    const std::string first_part = lfs::core::path_to_utf8(*rel.begin());
                    const std::filesystem::path stripped = drop_leading_component(rel);
                    const std::string stripped_str = to_generic_utf8(stripped);
                    if (equals_ascii_ci(first_part, "images")) {
                        rel_without_images = stripped_str;
                    }
                    // Only drop a non-"images" root folder when at least one
                    // subfolder remains so nested names do not collapse to a
                    // bare basename before full-path lookups.
                    if (components >= 3) {
                        rel_without_first = stripped_str;
                    }
                }
            }

            const bool image_name_has_dir =
                image_name.find('/') != std::string::npos || image_name.find('\\') != std::string::npos;

            if (image_name_has_dir) {
                append_candidate(image_name);
                append_candidate(rel_without_images);
                append_candidate(rel_without_first);
                append_candidate(rel_str);
            } else {
                append_candidate(rel_without_images);
                append_candidate(rel_without_first);
                append_candidate(rel_str);
                append_candidate(image_name);
            }

            return candidates;
        }

        template <typename Cache>
        [[nodiscard]] FileLookupResult lookup_with_candidates(
            const Cache& cache,
            const std::vector<std::string>& candidates) {
            bool saw_ambiguous = false;
            for (const auto& candidate : candidates) {
                if (auto result = cache.lookup(candidate); result.found()) {
                    return result;
                } else if (result.ambiguous()) {
                    saw_ambiguous = true;
                }
            }
            if (saw_ambiguous) {
                return FileLookupResult{LookupStatus::Ambiguous, {}};
            }
            return {};
        }

        [[nodiscard]] std::string format_ambiguity_message(
            const std::string_view kind,
            const std::string_view folder_kind,
            const lfs::core::Camera& cam,
            const std::filesystem::path& dataset_root) {
            const std::string display_name = !cam.image_name().empty()
                                                 ? cam.image_name()
                                                 : lfs::core::path_to_utf8(cam.image_path());
            return std::format(
                "{} for image '{}' is ambiguous across the dataset {} folders ({}). "
                "Keep {} in the same relative subdirectories as the images or rename them uniquely.",
                kind,
                display_name,
                folder_kind,
                lfs::core::path_to_utf8(dataset_root),
                folder_kind);
        }

        [[nodiscard]] bool needs_mask_discovery(const lfs::core::Camera& cam) {
            return !cam.has_mask() &&
                   !cam.mask_asset_uuid().has_value() &&
                   (!cam.image_name().empty() || !cam.image_path().empty());
        }

        [[nodiscard]] bool needs_depth_discovery(const lfs::core::Camera& cam) {
            return !cam.has_depth() &&
                   (!cam.image_name().empty() || !cam.image_path().empty());
        }

        [[nodiscard]] bool needs_normal_discovery(const lfs::core::Camera& cam) {
            return !cam.has_normal() &&
                   (!cam.image_name().empty() || !cam.image_path().empty());
        }

        [[nodiscard]] bool sidecar_matches_camera_contract(
            const lfs::core::Camera& cam,
            const std::filesystem::path& sidecar_path,
            const int required_channels) {
            int sc_w = 0, sc_h = 0, sc_c = 0;
            try {
                std::tie(sc_w, sc_h, sc_c) = lfs::core::get_image_info(sidecar_path);
            } catch (...) {
                return false;
            }
            if (sc_w <= 0 || sc_h <= 0) {
                return false;
            }
            if (required_channels > 0 && sc_c != required_channels) {
                return false;
            }

            if (cam.image_path().empty() || !safe_is_regular_file(cam.image_path())) {
                return true;
            }
            int img_w = 0, img_h = 0, img_c = 0;
            try {
                std::tie(img_w, img_h, img_c) = lfs::core::get_image_info(cam.image_path());
            } catch (...) {
                return true;
            }
            if (img_w <= 0 || img_h <= 0) {
                return true;
            }
            return sidecar_dimensions_match_contract(sc_w, sc_h, img_w, img_h);
        }

        template <typename Cache, typename Setter>
        [[nodiscard]] std::expected<void, std::string> bind_camera_sidecar(
            const std::optional<Cache>& cache,
            const bool needed,
            const std::vector<std::string>& candidates,
            const lfs::core::Camera& cam,
            const std::filesystem::path& dataset_root,
            const std::string_view kind_label,
            const std::string_view folder_label,
            const bool fail_on_ambiguous,
            const bool validate_contract,
            const int required_channels,
            size_t& bound_count,
            size_t& ambiguous_count,
            bool& warned_ambiguous,
            Setter&& setter) {
            if (!cache || !needed) {
                return {};
            }
            if (auto lookup = lookup_with_candidates(*cache, candidates); lookup.found()) {
                if (validate_contract &&
                    !sidecar_matches_camera_contract(cam, lookup.path, required_channels)) {
                    LOG_WARN("Ignoring '{}' for '{}' because it does not match image dimensions/channels contract",
                             lfs::core::path_to_utf8(lookup.path),
                             !cam.image_name().empty() ? cam.image_name()
                                                       : lfs::core::path_to_utf8(cam.image_path()));
                    return {};
                }
                setter(std::move(lookup.path));
                ++bound_count;
            } else if (lookup.ambiguous()) {
                const std::string msg =
                    format_ambiguity_message(kind_label, folder_label, cam, dataset_root);
                if (fail_on_ambiguous) {
                    return std::unexpected(msg);
                }
                ++ambiguous_count;
                if (!warned_ambiguous) {
                    LOG_WARN("{}", msg);
                    warned_ambiguous = true;
                }
            }
            return {};
        }

    } // namespace

    std::filesystem::path resolve_dataset_root(const std::filesystem::path& data_path) {
        if (data_path.empty()) {
            return {};
        }
        if (safe_is_directory(data_path)) {
            return data_path;
        }
        std::string ext = lfs::core::path_to_utf8(data_path.extension());
        detail::ascii_lower_inplace(ext);
        if (safe_is_regular_file(data_path) || ext == ".json") {
            const auto parent = data_path.parent_path();
            if (!parent.empty() && safe_is_directory(parent)) {
                return parent;
            }
        }
        return {};
    }

    std::expected<SidecarAssociationResult, std::string> associate_dataset_sidecars(
        const std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
        const std::filesystem::path& dataset_path,
        const SidecarAssociationOptions& options) {
        SidecarAssociationResult result;
        if ((!options.masks && !options.depths && !options.normals) || cameras.empty()) {
            return result;
        }

        const std::filesystem::path dataset_root = resolve_dataset_root(dataset_path);
        if (dataset_root.empty()) {
            return result;
        }

        const auto any_need = [&](auto pred) {
            return std::ranges::any_of(cameras, [&](const auto& cam) {
                return cam && pred(*cam);
            });
        };
        const bool any_need_mask = options.masks && any_need(needs_mask_discovery);
        const bool any_need_depth = options.depths && any_need(needs_depth_discovery);
        const bool any_need_normal = options.normals && any_need(needs_normal_discovery);
        if (!any_need_mask && !any_need_depth && !any_need_normal) {
            return result;
        }

        std::optional<MaskDirCache> mask_cache;
        if (any_need_mask) {
            MaskDirCache cache(dataset_root, options.cancel_requested);
            if (cache.has_mask_dirs()) {
                mask_cache = std::move(cache);
            }
        }

        std::optional<DepthDirCache> depth_cache;
        if (any_need_depth) {
            DepthDirCache cache(dataset_root, options.cancel_requested);
            if (cache.has_depth_dirs()) {
                depth_cache = std::move(cache);
            }
        }

        std::optional<NormalDirCache> normal_cache;
        if (any_need_normal) {
            NormalDirCache cache(dataset_root, options.cancel_requested);
            if (cache.has_normal_dirs()) {
                normal_cache = std::move(cache);
            }
        }

        if (!mask_cache && !depth_cache && !normal_cache) {
            return result;
        }

        std::error_code ec;
        const std::filesystem::path normalized_root = dataset_root.is_absolute()
                                                          ? dataset_root.lexically_normal()
                                                          : std::filesystem::absolute(dataset_root, ec).lexically_normal();
        std::error_code canon_ec;
        const std::filesystem::path canonical_root =
            std::filesystem::weakly_canonical(normalized_root, canon_ec);
        const std::filesystem::path effective_canon_root = canon_ec ? std::filesystem::path{} : canonical_root;

        bool warned_ambiguous_mask = false;
        bool warned_ambiguous_depth = false;
        bool warned_ambiguous_normal = false;

        for (size_t i = 0; i < cameras.size(); ++i) {
            if ((i % detail::CANCEL_POLL_INTERVAL) == 0 && options.cancel_requested && options.cancel_requested()) {
                return std::unexpected("Sidecar discovery cancelled");
            }

            const auto& cam = cameras[i];
            if (!cam) {
                continue;
            }

            const bool find_mask = mask_cache && needs_mask_discovery(*cam);
            const bool find_depth = depth_cache && needs_depth_discovery(*cam);
            const bool find_normal = normal_cache && needs_normal_discovery(*cam);
            if (!find_mask && !find_depth && !find_normal) {
                continue;
            }

            const std::vector<std::string> candidates =
                build_camera_lookup_candidates(*cam, normalized_root, effective_canon_root);

            if (auto ok = bind_camera_sidecar(
                    mask_cache, find_mask, candidates, *cam, dataset_root,
                    "Mask", "mask", options.fail_on_ambiguous_mask,
                    false, 0,
                    result.masks_bound, result.ambiguous_masks, warned_ambiguous_mask,
                    [&](std::filesystem::path path) { cam->set_mask_path(std::move(path)); });
                !ok) {
                return std::unexpected(ok.error());
            }
            if (auto ok = bind_camera_sidecar(
                    depth_cache, find_depth, candidates, *cam, dataset_root,
                    "Depth map", "depth", options.fail_on_ambiguous_depth,
                    options.skip_mismatched_depths, 0,
                    result.depths_bound, result.ambiguous_depths, warned_ambiguous_depth,
                    [&](std::filesystem::path path) { cam->set_depth_path(std::move(path)); });
                !ok) {
                return std::unexpected(ok.error());
            }
            if (auto ok = bind_camera_sidecar(
                    normal_cache, find_normal, candidates, *cam, dataset_root,
                    "Normal map", "normal", options.fail_on_ambiguous_normal,
                    options.skip_mismatched_normals, 3,
                    result.normals_bound, result.ambiguous_normals, warned_ambiguous_normal,
                    [&](std::filesystem::path path) { cam->set_normal_path(std::move(path)); });
                !ok) {
                return std::unexpected(ok.error());
            }
        }

        return result;
    }

} // namespace lfs::io

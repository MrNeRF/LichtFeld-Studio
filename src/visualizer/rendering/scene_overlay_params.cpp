/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "scene_overlay_params.hpp"
#include "rendering/coordinate_conventions.hpp"
#include <algorithm>
#include <format>
namespace lfs::vis {
    namespace {
        void writeVec4(float* dst, const std::size_t index, const glm::vec4& value) {
            dst[index * 4 + 0] = value.x;
            dst[index * 4 + 1] = value.y;
            dst[index * 4 + 2] = value.z;
            dst[index * 4 + 3] = value.w;
        }

        void writeMat4Rows(float* dst, const std::size_t index, const glm::mat4& matrix) {
            for (int row = 0; row < 4; ++row) {
                writeVec4(dst,
                          index + static_cast<std::size_t>(row),
                          glm::vec4(matrix[0][row],
                                    matrix[1][row],
                                    matrix[2][row],
                                    matrix[3][row]));
            }
        }

        void writeMat4AffineRows(float* dst, const std::size_t index, const glm::mat4& matrix) {
            for (int row = 0; row < 3; ++row) {
                writeVec4(dst,
                          index + static_cast<std::size_t>(row),
                          glm::vec4(matrix[0][row],
                                    matrix[1][row],
                                    matrix[2][row],
                                    matrix[3][row]));
            }
        }

    } // namespace

    namespace detail {

        // Builds the overlay parameter table on CPU only. The H2D transfer is
        // performed at the call site, conditionally on an output-bytes diff.
        [[nodiscard]] std::expected<std::vector<float>, std::string> buildOverlayParamsCpuFloats(
            const lfs::rendering::ViewportRenderRequest& request,
            const bool selection_enabled,
            const bool preview_enabled,
            const bool transform_indices_enabled,
            const std::size_t node_mask_count,
            const bool node_visibility_cull) {
            try {
                std::vector<float> cpu(static_cast<std::size_t>(ParamCount) * 4u, 0.0f);
                float* const dst = cpu.data();

                const auto write_crop = [&](const lfs::rendering::GaussianScopedBoxFilter& crop,
                                            const std::size_t flags_index) {
                    writeVec4(dst,
                              flags_index,
                              glm::vec4(1.0f,
                                        crop.inverse ? 1.0f : 0.0f,
                                        crop.desaturate ? 1.0f : 0.0f,
                                        static_cast<float>(crop.parent_node_index)));
                    writeVec4(dst, flags_index + 1, glm::vec4(crop.bounds.min, 0.0f));
                    writeVec4(dst, flags_index + 2, glm::vec4(crop.bounds.max, 0.0f));
                    writeMat4Rows(dst, flags_index + 3, crop.bounds.transform);
                };
                const auto write_ellipsoid = [&](const lfs::rendering::GaussianScopedEllipsoidFilter& ellipsoid,
                                                 const std::size_t flags_index) {
                    writeVec4(dst,
                              flags_index,
                              glm::vec4(1.0f,
                                        ellipsoid.inverse ? 1.0f : 0.0f,
                                        ellipsoid.desaturate ? 1.0f : 0.0f,
                                        static_cast<float>(ellipsoid.parent_node_index)));
                    writeVec4(dst, flags_index + 1, glm::vec4(ellipsoid.bounds.radii, 0.0f));
                    writeMat4AffineRows(dst, flags_index + 2, ellipsoid.bounds.transform);
                };

                const auto& crop_regions = request.filters.crop_regions;
                if (!crop_regions.empty()) {
                    write_crop(crop_regions.front(), CropFlags);
                    const std::size_t extra_count = std::min<std::size_t>(crop_regions.size() - 1u, CropExtraCount);
                    for (std::size_t i = 0; i < extra_count; ++i) {
                        write_crop(crop_regions[i + 1u], CropExtraBase + i * CropParamStride);
                    }
                } else if (request.filters.crop_region) {
                    write_crop(*request.filters.crop_region, CropFlags);
                }

                const auto& ellipsoid_regions = request.filters.ellipsoid_regions;
                if (!ellipsoid_regions.empty()) {
                    write_ellipsoid(ellipsoid_regions.front(), EllipsoidFlags);
                    const std::size_t extra_count = std::min<std::size_t>(ellipsoid_regions.size() - 1u, EllipsoidExtraCount);
                    for (std::size_t i = 0; i < extra_count; ++i) {
                        write_ellipsoid(ellipsoid_regions[i + 1u], EllipsoidExtraBase + i * EllipsoidParamStride);
                    }
                } else if (request.filters.ellipsoid_region) {
                    write_ellipsoid(*request.filters.ellipsoid_region, EllipsoidFlags);
                }

                if (request.filters.view_volume) {
                    const auto& screen_window = request.filters.screen_window;
                    writeVec4(dst,
                              ViewFlags,
                              glm::vec4(screen_window ? 1.0f : 0.0f,
                                        request.filters.cull_outside_view_volume ? 1.0f : 0.0f,
                                        request.filters.dim_outside_view_volume ? 1.0f : 0.0f,
                                        0.0f)); // retired isotropic lane; window scale now in ViewWindow
                    writeVec4(dst,
                              ViewWindow,
                              glm::vec4(screen_window ? screen_window->scale_x : 0.0f,
                                        screen_window ? screen_window->scale_y : 0.0f,
                                        (screen_window && screen_window->drag_preview) ? 1.0f : 0.0f,
                                        0.0f)); // z: drag-preview live-reveal flag
                    writeVec4(dst,
                              ViewMin,
                              glm::vec4(request.filters.view_volume->min,
                                        screen_window ? screen_window->offset_x : 0.0f));
                    writeVec4(dst,
                              ViewMax,
                              glm::vec4(request.filters.view_volume->max,
                                        screen_window ? screen_window->offset_y : 0.0f));
                    // Containment intrinsics for the shader's screen-window test. Written only
                    // for a non-orthographic, non-equirect request that carries them; the table
                    // is zero-filled at construction, so slot 12 stays (0,0,0,0) otherwise and the
                    // shader falls back to cam's own focals and the image centre.
                    if (!request.frame_view.orthographic && !request.equirectangular) {
                        if (const auto& containment = request.frame_view.containment_intrinsics) {
                            writeVec4(dst,
                                      ViewIntrinsics,
                                      glm::vec4(containment->focal_x,
                                                containment->focal_y,
                                                containment->center_x,
                                                containment->center_y));
                        }
                    }
                    writeMat4Rows(dst, ViewTransform, request.filters.view_volume->transform);
                }

                writeVec4(dst,
                          VisibilityFlags,
                          glm::vec4(node_visibility_cull ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f));
                writeVec4(dst,
                          EmphasisFlags,
                          glm::vec4(request.overlay.emphasis.dim_non_emphasized ? 1.0f : 0.0f,
                                    transform_indices_enabled ? 1.0f : 0.0f,
                                    static_cast<float>(node_mask_count),
                                    0.0f));
                writeVec4(dst,
                          CursorFlags,
                          glm::vec4(request.overlay.cursor.enabled ? 1.0f : 0.0f,
                                    request.overlay.cursor.saturation_preview ? 1.0f : 0.0f,
                                    request.overlay.cursor.saturation_amount,
                                    request.overlay.markers.ring_width));
                writeVec4(dst,
                          MarkerFlags,
                          glm::vec4(request.overlay.markers.show_rings ? 1.0f : 0.0f,
                                    request.overlay.markers.show_center_markers ? 1.0f : 0.0f,
                                    0.0f,
                                    0.0f));
                const bool cursor_selection_enabled =
                    !request.overlay.cursor.saturation_preview &&
                    request.overlay.cursor.enabled &&
                    request.overlay.cursor.radius > 0.0f;
                writeVec4(dst,
                          SelectionCursor,
                          glm::vec4(request.overlay.cursor.cursor.x,
                                    request.overlay.cursor.cursor.y,
                                    std::max(request.overlay.cursor.radius, 0.0f),
                                    cursor_selection_enabled ? 1.0f : 0.0f));
                writeVec4(dst,
                          SelectionFlags,
                          glm::vec4(selection_enabled ? 1.0f : 0.0f,
                                    preview_enabled ? 1.0f : 0.0f,
                                    request.overlay.emphasis.transient_mask.additive ? 1.0f : 0.0f,
                                    request.overlay.emphasis.focused_gaussian_id >= 0
                                        ? static_cast<float>(request.overlay.emphasis.focused_gaussian_id)
                                        : -1.0f));

                return cpu;
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Scene renderer failed to stage overlay parameters: {}", e.what()));
            }
        }

    } // namespace detail
} // namespace lfs::vis

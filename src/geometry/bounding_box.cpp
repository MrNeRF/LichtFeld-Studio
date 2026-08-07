/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "geometry/bounding_box.hpp"

#include <stdexcept>

namespace lfs {
    namespace geometry {

        BoundingBox::BoundingBox()
            : min_bounds_(-1.0f, -1.0f, -1.0f),
              max_bounds_(1.0f, 1.0f, 1.0f),
              world2BBox_(EuclideanTransform()) {}

        BoundingBox::~BoundingBox() {}

        void BoundingBox::setBounds(const glm::vec3& min, const glm::vec3& max) {
            // Validate bounds
            if (min.x > max.x || min.y > max.y || min.z > max.z) {
                throw std::runtime_error("Warning: Invalid bounding box bounds (min > max)");
            }

            min_bounds_ = min;
            max_bounds_ = max;
        }

        void BoundingBox::setworld2BBox(const geometry::EuclideanTransform& transform) {
            world2BBox_ = transform;
            world2BBox_mat4_ = transform.toMat4();
            has_full_transform_ = false;
        }

        void BoundingBox::setworld2BBox(const glm::mat4& transform) {
            world2BBox_mat4_ = transform;
            has_full_transform_ = true;
            // Also set EuclideanTransform for backwards compatibility (loses scale)
            world2BBox_ = EuclideanTransform(transform);
        }

    } // namespace geometry
} // namespace lfs

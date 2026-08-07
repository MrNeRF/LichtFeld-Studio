/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lfs {
    namespace geometry {

        class EuclideanTransform {
        private:
            glm::quat m_rotation{};  // Quaternion for rotation
            glm::vec3 m_translation; // Vector for translation

        public:
            // Default constructor - identity transformation
            EuclideanTransform();

            // Constructor from quaternion and translation vector
            EuclideanTransform(const glm::quat& rot, const glm::vec3& trans);

            // Constructor from 4x4 transformation matrix
            explicit EuclideanTransform(const glm::mat4& matrix);

            // Convert to 4x4 transformation matrix
            glm::mat4 toMat4() const;

            // Composition operator - combines two transformations
            EuclideanTransform operator*(const EuclideanTransform& other) const;

            // Inverse transformation
            EuclideanTransform inv() const;

            // Getters
            const glm::quat& getRotation() const { return m_rotation; }
            const glm::vec3& getTranslation() const { return m_translation; }

        private:
            // Orthonormalize rotation matrix
            static glm::mat4 OrthonormalizeRotation(const glm::mat4& matrix);
        };
    } // namespace geometry
} // namespace lfs

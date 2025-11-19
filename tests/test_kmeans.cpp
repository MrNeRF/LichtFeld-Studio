/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

#include "kernels/kmeans.cuh"

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-4f;

// Helper to check CUDA errors
#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t error = call;                                                     \
        ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error); \
    } while (0)

    // Helper to create random test data
    torch::Tensor create_random_data(int n, int d, float min_val = 0.0f, float max_val = 1.0f) {
        auto data = torch::rand({n, d}, torch::kCUDA) * (max_val - min_val) + min_val;
        return data;
    }

    // Helper to create clustered test data
    torch::Tensor create_clustered_data(int n, int k, int d, float separation = 5.0f) {
        auto data = torch::zeros({n, d}, torch::kCUDA);
        int points_per_cluster = n / k;

        for (int c = 0; c < k; ++c) {
            int start = c * points_per_cluster;
            int end = (c == k - 1) ? n : (c + 1) * points_per_cluster;

            // Create cluster centered at (c * separation, c * separation, ...)
            auto cluster_center = torch::full({d}, c * separation, torch::kCUDA);
            auto noise = torch::randn({end - start, d}, torch::kCUDA) * 0.5f;

            data.slice(0, start, end) = cluster_center + noise;
        }

        return data;
    }

    // Helper function to compute inertia (sum of squared distances to centroids)
    float compute_inertia(const torch::Tensor& data,
                          const torch::Tensor& centroids,
                          const torch::Tensor& labels) {
        auto n = data.size(0);
        auto d = data.size(1);

        float inertia = 0.0f;
        auto data_cpu = data.cpu();
        auto centroids_cpu = centroids.cpu();
        auto labels_cpu = labels.cpu();

        for (int i = 0; i < n; ++i) {
            int cluster = labels_cpu[i].item<int>();
            float dist = 0.0f;
            for (int j = 0; j < d; ++j) {
                float diff = data_cpu[i][j].item<float>() - centroids_cpu[cluster][j].item<float>();
                dist += diff * diff;
            }
            inertia += dist;
        }

        return inertia;
    }

    // Helper to verify all labels are valid
    bool verify_labels_valid(const torch::Tensor& labels, int k) {
        auto labels_cpu = labels.cpu();
        for (int i = 0; i < labels.size(0); ++i) {
            int label = labels_cpu[i].item<int>();
            if (label < 0 || label >= k) {
                return false;
            }
        }
        return true;
    }

    // Helper to count unique labels
    int count_unique_labels(const torch::Tensor& labels) {
        auto labels_cpu = labels.cpu();
        std::vector<int> label_vec;
        for (int i = 0; i < labels.size(0); ++i) {
            label_vec.push_back(labels_cpu[i].item<int>());
        }
        std::sort(label_vec.begin(), label_vec.end());
        auto last = std::unique(label_vec.begin(), label_vec.end());
        return std::distance(label_vec.begin(), last);
    }

} // anonymous namespace

class KMeansTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";

        // Set random seed for reproducibility
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(KMeansTest, BasicClustering2D) {
    const int n_points = 1000;
    const int n_dims = 2;
    const int k = 3;

    auto data = create_clustered_data(n_points, k, n_dims, 10.0f);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 50, 1e-4f);

    EXPECT_EQ(centroids.size(0), k);
    EXPECT_EQ(centroids.size(1), n_dims);
    EXPECT_EQ(labels.size(0), n_points);

    // Verify all labels are valid
    EXPECT_TRUE(verify_labels_valid(labels, k));

    // Verify we found k clusters (or fewer if data permits)
    int unique_clusters = count_unique_labels(labels);
    EXPECT_LE(unique_clusters, k);
    EXPECT_GT(unique_clusters, 0);
}

TEST_F(KMeansTest, BasicClustering1D) {
    const int n_points = 500;
    const int k = 256;

    auto data = create_random_data(n_points, 1, 0.0f, 100.0f);

    auto [centroids, labels] = gs::cuda::kmeans_1d(data, k, 20);

    EXPECT_EQ(centroids.size(0), k);
    EXPECT_EQ(centroids.size(1), 1);
    EXPECT_EQ(labels.size(0), n_points);

    // Verify centroids are sorted
    auto centroids_cpu = centroids.squeeze(1).cpu();
    for (int i = 1; i < k; ++i) {
        EXPECT_GE(centroids_cpu[i].item<float>(), centroids_cpu[i - 1].item<float>())
            << "Centroids should be sorted in 1D k-means";
    }

    // Verify all labels are valid
    EXPECT_TRUE(verify_labels_valid(labels, k));
}

TEST_F(KMeansTest, EmptyInput) {
    const int k = 3;
    auto empty_data = torch::empty({0, 2}, torch::kCUDA);

    // Should handle empty input gracefully
    auto [centroids, labels] = gs::cuda::kmeans(empty_data, k, 10, 1e-4f);

    // Empty input should return empty results
    EXPECT_EQ(centroids.size(0), 0);
    EXPECT_EQ(labels.size(0), 0);
}

TEST_F(KMeansTest, FewerPointsThanClusters) {
    const int n_points = 5;
    const int n_dims = 2;
    const int k = 10;

    auto data = create_random_data(n_points, n_dims);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 10, 1e-4f);

    // Should return at most n_points centroids
    EXPECT_LE(centroids.size(0), n_points);
    EXPECT_EQ(labels.size(0), n_points);

    auto unique = count_unique_labels(labels);
    EXPECT_LE(unique, n_points);
}

// ============================================================================
// Input Validation Tests
// ============================================================================

TEST_F(KMeansTest, WrongDimensionality) {
    auto data_3d = torch::rand({10, 5, 3}, torch::kCUDA);

    EXPECT_THROW({ gs::cuda::kmeans(data_3d, 3, 10, 1e-4f); }, c10::Error);
}

TEST_F(KMeansTest, CPUTensor) {
    auto data_cpu = torch::rand({100, 2});

    EXPECT_THROW({ gs::cuda::kmeans(data_cpu, 3, 10, 1e-4f); }, c10::Error);
}

TEST_F(KMeansTest, WrongDtype) {
    auto data_double = torch::rand({100, 2}, torch::kDouble).cuda();

    EXPECT_THROW({ gs::cuda::kmeans(data_double, 3, 10, 1e-4f); }, c10::Error);
}

// ============================================================================
// Convergence Tests
// ============================================================================

TEST_F(KMeansTest, ConvergenceWithIterations) {
    const int n_points = 500;
    const int n_dims = 2;
    const int k = 4;

    auto data = create_clustered_data(n_points, k, n_dims, 8.0f);

    // Run with different iteration counts
    auto [centroids1, labels1] = gs::cuda::kmeans(data, k, 5, 1e-4f);
    auto [centroids2, labels2] = gs::cuda::kmeans(data, k, 50, 1e-4f);

    float inertia1 = compute_inertia(data, centroids1, labels1);
    float inertia2 = compute_inertia(data, centroids2, labels2);

    // More iterations should give equal or better inertia
    EXPECT_LE(inertia2, inertia1 + FLOAT_TOLERANCE);
}

TEST_F(KMeansTest, ToleranceStopsEarly) {
    const int n_points = 300;
    const int n_dims = 2;
    const int k = 3;

    auto data = create_clustered_data(n_points, k, n_dims, 20.0f); // Very separated clusters

    // With high tolerance, should converge quickly (still produces valid results)
    auto [centroids, labels] = gs::cuda::kmeans(data, k, 100, 1.0f); // High tolerance

    // Should still produce valid clustering
    EXPECT_TRUE(verify_labels_valid(labels, k));
}

// ============================================================================
// 1D Specific Tests
// ============================================================================

TEST_F(KMeansTest, OneDimensionalOrdering) {
    const int n_points = 256;
    const int k = 16;

    // Create linearly spaced data
    auto data = torch::linspace(0, 255, n_points, torch::kCUDA).unsqueeze(1);

    auto [centroids, labels] = gs::cuda::kmeans_1d(data, k, 20);

    auto centroids_cpu = centroids.squeeze(1).cpu();

    // Verify centroids are strictly increasing
    for (int i = 1; i < k; ++i) {
        EXPECT_GT(centroids_cpu[i].item<float>(), centroids_cpu[i - 1].item<float>());
    }

    // Verify points are assigned to nearest centroid
    auto data_cpu = data.squeeze(1).cpu();
    auto labels_cpu = labels.cpu();

    for (int i = 0; i < n_points; ++i) {
        int assigned = labels_cpu[i].item<int>();
        float point = data_cpu[i].item<float>();
        float dist_to_assigned = std::abs(point - centroids_cpu[assigned].item<float>());

        // Check no other centroid is significantly closer
        for (int c = 0; c < k; ++c) {
            float dist = std::abs(point - centroids_cpu[c].item<float>());
            EXPECT_GE(dist, dist_to_assigned - FLOAT_TOLERANCE);
        }
    }
}

TEST_F(KMeansTest, OneDimensionalWithDuplicates) {
    const int n_points = 100;
    const int k = 10;

    // Create data with many duplicates
    auto data = torch::floor(torch::arange(0, n_points, torch::kCUDA) / 10.0f).unsqueeze(1);

    auto [centroids, labels] = gs::cuda::kmeans_1d(data, k, 20);

    // Should handle duplicates gracefully
    EXPECT_TRUE(verify_labels_valid(labels, k));
}

TEST_F(KMeansTest, OneDimensional1DInput) {
    const int n_points = 300;
    const int k = 10;

    // Test with 1D input (not 2D with second dim = 1)
    auto data_1d = create_random_data(n_points, 1, 0.0f, 100.0f).squeeze(1);

    auto [centroids, labels] = gs::cuda::kmeans_1d(data_1d, k, 20);

    EXPECT_EQ(centroids.size(0), k);
    EXPECT_TRUE(verify_labels_valid(labels, k));
}

// ============================================================================
// Dimension Tests
// ============================================================================

TEST_F(KMeansTest, HighDimensionalClustering) {
    const int n_points = 200;
    const int n_dims = 50;
    const int k = 5;

    auto data = create_random_data(n_points, n_dims);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 30, 1e-4f);

    EXPECT_EQ(centroids.size(0), k);
    EXPECT_EQ(centroids.size(1), n_dims);
    EXPECT_TRUE(verify_labels_valid(labels, k));
}

TEST_F(KMeansTest, ThreeDimensionalClustering) {
    const int n_points = 500;
    const int n_dims = 3;
    const int k = 5;

    auto data = create_clustered_data(n_points, k, n_dims, 15.0f);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 30, 1e-4f);

    EXPECT_EQ(centroids.size(0), k);
    EXPECT_EQ(centroids.size(1), n_dims);
    EXPECT_TRUE(verify_labels_valid(labels, k));

    // Should find most or all clusters
    int unique = count_unique_labels(labels);
    EXPECT_GE(unique, k - 1); // Allow for one cluster to be missed due to randomness
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(KMeansTest, SingleCluster) {
    const int n_points = 100;
    const int n_dims = 3;
    const int k = 1;

    auto data = create_random_data(n_points, n_dims);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 10, 1e-4f);

    EXPECT_EQ(centroids.size(0), 1);
    auto labels_cpu = labels.cpu();

    // All points should be in cluster 0
    for (int i = 0; i < n_points; ++i) {
        EXPECT_EQ(labels_cpu[i].item<int>(), 0);
    }
}

TEST_F(KMeansTest, SinglePoint) {
    const int n_points = 1;
    const int n_dims = 2;
    const int k = 3;

    auto data = create_random_data(n_points, n_dims);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 10, 1e-4f);

    // Single point should result in single centroid
    EXPECT_EQ(centroids.size(0), 1);
    EXPECT_EQ(labels.size(0), 1);

    auto labels_cpu = labels.cpu();
    EXPECT_GE(labels_cpu[0].item<int>(), 0);
}

TEST_F(KMeansTest, TwoPoints) {
    const int n_points = 2;
    const int n_dims = 2;
    const int k = 3;

    auto data = torch::tensor({{0.0f, 0.0f}, {10.0f, 10.0f}}, torch::kCUDA);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 10, 1e-4f);

    // Should handle gracefully
    EXPECT_EQ(labels.size(0), 2);
    EXPECT_TRUE(verify_labels_valid(labels, centroids.size(0)));
}

TEST_F(KMeansTest, VeryTightCluster) {
    const int n_points = 100;
    const int n_dims = 2;
    const int k = 3;

    // Create a tight cluster with reasonable variance
    auto data = torch::randn({n_points, n_dims}, torch::kCUDA) * 0.1f + 42.0f;

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 10, 1e-4f);

    auto centroids_cpu = centroids.cpu();

    // Should have returned some centroids
    EXPECT_GT(centroids.size(0), 0);
    EXPECT_LE(centroids.size(0), k);

    // All centroids should be near 42.0 (within reasonable range)
    for (int i = 0; i < centroids.size(0); ++i) {
        for (int d = 0; d < n_dims; ++d) {
            EXPECT_NEAR(centroids_cpu[i][d].item<float>(), 42.0f, 1.0f);
        }
    }

    // All labels should be valid
    EXPECT_TRUE(verify_labels_valid(labels, centroids.size(0)));
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(KMeansTest, LargeDataset) {
    const int n_points = 10000;
    const int n_dims = 3;
    const int k = 8;

    auto data = create_random_data(n_points, n_dims);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 20, 1e-4f);

    EXPECT_EQ(centroids.size(0), k);
    EXPECT_EQ(labels.size(0), n_points);
    EXPECT_TRUE(verify_labels_valid(labels, k));
}

TEST_F(KMeansTest, ManyClusters) {
    const int n_points = 5000;
    const int n_dims = 2;
    const int k = 256;

    auto data = create_random_data(n_points, n_dims);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 30, 1e-4f);

    EXPECT_EQ(centroids.size(0), k);
    EXPECT_TRUE(verify_labels_valid(labels, k));
}

TEST_F(KMeansTest, LargeOneDimensionalDataset) {
    const int n_points = 50000;
    const int k = 256;

    auto data = create_random_data(n_points, 1, 0.0f, 10000.0f);

    auto [centroids, labels] = gs::cuda::kmeans_1d(data, k, 20);

    auto centroids_cpu = centroids.squeeze(1).cpu();

    // Verify centroids are sorted
    for (int i = 1; i < k; ++i) {
        EXPECT_GE(centroids_cpu[i].item<float>(), centroids_cpu[i - 1].item<float>());
    }

    EXPECT_TRUE(verify_labels_valid(labels, k));
}

TEST_F(KMeansTest, VeryHighDimensional) {
    const int n_points = 100;
    const int n_dims = 128;
    const int k = 5;

    auto data = create_random_data(n_points, n_dims);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 20, 1e-4f);

    EXPECT_EQ(centroids.size(0), k);
    EXPECT_EQ(centroids.size(1), n_dims);
    EXPECT_TRUE(verify_labels_valid(labels, k));
}

// ============================================================================
// Quality Tests
// ============================================================================

TEST_F(KMeansTest, WellSeparatedClusters) {
    const int n_points = 600;
    const int n_dims = 2;
    const int k = 3;

    auto data = create_clustered_data(n_points, k, n_dims, 50.0f); // Very well separated

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 50, 1e-4f);

    // Should find all k clusters
    int unique = count_unique_labels(labels);
    EXPECT_EQ(unique, k);

    // Each cluster should have roughly equal number of points
    auto labels_cpu = labels.cpu();
    std::vector<int> counts(k, 0);
    for (int i = 0; i < n_points; ++i) {
        counts[labels_cpu[i].item<int>()]++;
    }

    int expected = n_points / k;
    for (int count : counts) {
        // Within 20% of expected
        EXPECT_NEAR(count, expected, expected * 0.2);
    }
}

TEST_F(KMeansTest, ImprovesWithIterations) {
    const int n_points = 400;
    const int n_dims = 2;
    const int k = 4;

    auto data = create_clustered_data(n_points, k, n_dims, 5.0f);

    // Run with 1 iteration
    auto [centroids1, labels1] = gs::cuda::kmeans(data, k, 1, 0.0f);
    float inertia1 = compute_inertia(data, centroids1, labels1);

    // Run with 20 iterations
    auto [centroids20, labels20] = gs::cuda::kmeans(data, k, 20, 1e-4f);
    float inertia20 = compute_inertia(data, centroids20, labels20);

    // More iterations should improve or maintain inertia
    EXPECT_LE(inertia20, inertia1);
}

TEST_F(KMeansTest, CentroidsAreInDataRange) {
    const int n_points = 500;
    const int n_dims = 3;
    const int k = 5;

    float min_val = 10.0f;
    float max_val = 50.0f;
    auto data = create_random_data(n_points, n_dims, min_val, max_val);

    auto [centroids, labels] = gs::cuda::kmeans(data, k, 30, 1e-4f);

    auto centroids_cpu = centroids.cpu();

    // Centroids should be within or close to the data range
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < n_dims; ++d) {
            float val = centroids_cpu[i][d].item<float>();
            EXPECT_GE(val, min_val - 1.0f); // Allow small margin
            EXPECT_LE(val, max_val + 1.0f);
        }
    }
}

TEST_F(KMeansTest, DifferentTolerance) {
    const int n_points = 300;
    const int n_dims = 2;
    const int k = 4;

    auto data = create_clustered_data(n_points, k, n_dims, 8.0f);

    // Strict tolerance
    auto [centroids_strict, labels_strict] = gs::cuda::kmeans(data, k, 100, 1e-6f);

    // Loose tolerance
    auto [centroids_loose, labels_loose] = gs::cuda::kmeans(data, k, 100, 1e-2f);

    // Both should produce valid results
    EXPECT_TRUE(verify_labels_valid(labels_strict, k));
    EXPECT_TRUE(verify_labels_valid(labels_loose, k));
}

// ============================================================================
// Specialized Tests
// ============================================================================

TEST_F(KMeansTest, SOGTypical256Clusters) {
    // Typical use case for SOG quantization
    const int n_points = 10000;
    const int k = 256;

    auto data = create_random_data(n_points, 1, 0.0f, 1.0f);

    auto [centroids, labels] = gs::cuda::kmeans_1d(data, k, 20);

    EXPECT_EQ(centroids.size(0), 256);
    EXPECT_TRUE(verify_labels_valid(labels, 256));

    // Verify sorted
    auto centroids_cpu = centroids.squeeze(1).cpu();
    for (int i = 1; i < 256; ++i) {
        EXPECT_GE(centroids_cpu[i].item<float>(), centroids_cpu[i - 1].item<float>());
    }
}

TEST_F(KMeansTest, CompareKmeansVsKmeans1D) {
    const int n_points = 1000;
    const int k = 16;

    auto data = create_random_data(n_points, 1, 0.0f, 100.0f);

    // Run general k-means
    auto [centroids_nd, labels_nd] = gs::cuda::kmeans(data, k, 30, 1e-4f);
    float inertia_nd = compute_inertia(data, centroids_nd, labels_nd);

    // Run 1D specialized k-means
    auto [centroids_1d, labels_1d] = gs::cuda::kmeans_1d(data, k, 30);
    float inertia_1d = compute_inertia(data, centroids_1d, labels_1d);

    // Both should produce valid and comparable results
    EXPECT_TRUE(verify_labels_valid(labels_nd, k));
    EXPECT_TRUE(verify_labels_valid(labels_1d, k));

    // Inertias should be similar (1D might be better due to optimization)
    EXPECT_NEAR(inertia_nd, inertia_1d, std::max(inertia_nd, inertia_1d) * 0.1f);
}

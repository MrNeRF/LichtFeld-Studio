/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace lfs::core;

class AllocatorBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Warmup
        auto warmup = Tensor::rand({100, 100}, Device::CUDA);
        cudaDeviceSynchronize();
    }
};

TEST_F(AllocatorBenchmarkTest, SmallAllocationThroughput) {
    // Benchmark small allocations that should hit the slab allocator
    const int iterations = 10000;
    const size_t sizes[] = {256, 1024, 4096, 16384, 65536, 262144};

    std::cout << "\n=== Small Allocation Throughput ===\n";

    for (size_t alloc_size : sizes) {
        std::vector<void*> ptrs;
        ptrs.reserve(iterations);

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            void* ptr = CudaMemoryPool::instance().allocate(alloc_size);
            ptrs.push_back(ptr);
        }

        auto mid = std::chrono::high_resolution_clock::now();

        for (void* ptr : ptrs) {
            CudaMemoryPool::instance().deallocate(ptr, alloc_size);
        }

        auto end = std::chrono::high_resolution_clock::now();

        double alloc_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mid - start).count() / (double)iterations;
        double free_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - mid).count() / (double)iterations;

        std::cout << "  " << (alloc_size / 1024) << " KB: alloc=" << alloc_ns << " ns, free=" << free_ns << " ns\n";
    }
}

TEST_F(AllocatorBenchmarkTest, TensorCreationThroughput) {
    // Benchmark actual tensor creation (includes allocation + initialization)
    const int iterations = 1000;

    std::cout << "\n=== Tensor Creation Throughput ===\n";

    // Small tensors (should use slab)
    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            auto t = Tensor::empty({64, 64}, Device::CUDA); // 16 KB
        }
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)iterations;
        std::cout << "  Tensor::empty({64,64}): " << us << " us/iter\n";
    }

    // Medium tensors (should use cudaMallocAsync)
    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            auto t = Tensor::empty({512, 512}, Device::CUDA); // 1 MB
        }
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)iterations;
        std::cout << "  Tensor::empty({512,512}): " << us << " us/iter\n";
    }

    // Large tensors
    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            auto t = Tensor::empty({2048, 2048}, Device::CUDA); // 16 MB
        }
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)iterations;
        std::cout << "  Tensor::empty({2048,2048}): " << us << " us/iter\n";
    }
}

TEST_F(AllocatorBenchmarkTest, MixedSizeWorkload) {
    // Simulate realistic workload with mixed tensor sizes
    const int iterations = 5000;

    std::cout << "\n=== Mixed Size Workload ===\n";

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        // Small gradients
        auto grad1 = Tensor::empty({32, 32}, Device::CUDA);
        auto grad2 = Tensor::empty({64, 64}, Device::CUDA);

        // Medium features
        auto feat = Tensor::empty({256, 256}, Device::CUDA);

        // Small parameters
        auto param = Tensor::empty({128}, Device::CUDA);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)iterations;
    std::cout << "  Mixed workload (4 tensors): " << us << " us/iter\n";
    std::cout << "  Throughput: " << (iterations * 4) / (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0) << " tensors/sec\n";
}

TEST_F(AllocatorBenchmarkTest, SlabAllocatorStats) {
    std::cout << "\n=== Slab Allocator Statistics ===\n";
    GPUSlabAllocator::instance().print_stats();
    CudaMemoryPool::instance().print_stats();
}

TEST_F(AllocatorBenchmarkTest, CompareWithCudaMalloc) {
    // Direct comparison with raw cudaMalloc
    const int iterations = 1000;
    const size_t alloc_size = 16384; // 16 KB

    std::cout << "\n=== Direct cudaMalloc Comparison (16 KB) ===\n";

    // Our allocator
    {
        std::vector<void*> ptrs;
        ptrs.reserve(iterations);

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            void* ptr = CudaMemoryPool::instance().allocate(alloc_size);
            ptrs.push_back(ptr);
        }
        cudaDeviceSynchronize();

        auto mid = std::chrono::high_resolution_clock::now();

        for (void* ptr : ptrs) {
            CudaMemoryPool::instance().deallocate(ptr, alloc_size);
        }
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();

        double alloc_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mid - start).count() / (double)iterations;
        double free_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - mid).count() / (double)iterations;
        std::cout << "  CudaMemoryPool: alloc=" << alloc_ns << " ns, free=" << free_ns << " ns\n";
    }

    // Raw cudaMallocAsync
    {
        std::vector<void*> ptrs;
        ptrs.reserve(iterations);

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            void* ptr;
            cudaMallocAsync(&ptr, alloc_size, nullptr);
            ptrs.push_back(ptr);
        }
        cudaDeviceSynchronize();

        auto mid = std::chrono::high_resolution_clock::now();

        for (void* ptr : ptrs) {
            cudaFreeAsync(ptr, nullptr);
        }
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();

        double alloc_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mid - start).count() / (double)iterations;
        double free_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - mid).count() / (double)iterations;
        std::cout << "  cudaMallocAsync: alloc=" << alloc_ns << " ns, free=" << free_ns << " ns\n";
    }
}

TEST_F(AllocatorBenchmarkTest, LargeTensorBucketing) {
    // Test size bucketing for large tensors like (20M, 3, 15)
    std::cout << "\n=== Large Tensor Size Bucketing ===\n";

    // Simulate (20M, 3, 15) float tensor = 3.6 GB
    const size_t tensor_size = 20000000ULL * 3 * 15 * sizeof(float); // ~3.6 GB
    std::cout << "  Tensor size: " << (tensor_size / (1024.0 * 1024.0 * 1024.0)) << " GB\n";

    // Check bucket size
    size_t bucket_size = SizeBucketedPool::get_bucket_size(tensor_size);
    std::cout << "  Bucket size: " << (bucket_size / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  Waste: " << ((bucket_size - tensor_size) / (1024.0 * 1024.0)) << " MB ("
              << (100.0 * (bucket_size - tensor_size) / bucket_size) << "%)\n";

    // Test various sizes around this tensor
    std::cout << "\n  Size bucketing examples:\n";
    size_t sizes[] = {
        100ULL * 1024 * 1024,      // 100 MB
        500ULL * 1024 * 1024,      // 500 MB
        1024ULL * 1024 * 1024,     // 1 GB
        2ULL * 1024 * 1024 * 1024, // 2 GB
        3ULL * 1024 * 1024 * 1024, // 3 GB
        tensor_size,               // 3.6 GB
        4ULL * 1024 * 1024 * 1024, // 4 GB
    };

    for (size_t size : sizes) {
        size_t bucket = SizeBucketedPool::get_bucket_size(size);
        double waste_pct = 100.0 * (bucket - size) / bucket;
        std::cout << "    " << (size / (1024.0 * 1024.0)) << " MB -> "
                  << (bucket / (1024.0 * 1024.0)) << " MB bucket ("
                  << waste_pct << "% waste)\n";
    }
}

TEST_F(AllocatorBenchmarkTest, BucketCacheHitRate) {
    // Simulate repeated allocation pattern (like training iterations)
    std::cout << "\n=== Bucket Cache Hit Rate (Repeated Allocations) ===\n";

    const int iterations = 100;
    // Simulate common tensor sizes in training
    const size_t sizes[] = {
        1024 * 1024,       // 1 MB (features)
        4 * 1024 * 1024,   // 4 MB (gradients)
        16 * 1024 * 1024,  // 16 MB (activations)
        64 * 1024 * 1024,  // 64 MB (large batch)
        256 * 1024 * 1024, // 256 MB (image tensors)
    };

    // First pass: allocate and free (populate cache)
    for (size_t size : sizes) {
        void* ptr = CudaMemoryPool::instance().allocate(size);
        CudaMemoryPool::instance().deallocate(ptr, size);
    }
    cudaDeviceSynchronize();

    // Second pass: should hit cache
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        for (size_t size : sizes) {
            void* ptr = CudaMemoryPool::instance().allocate(size);
            CudaMemoryPool::instance().deallocate(ptr, size);
        }
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                (double)(iterations * 5); // 5 sizes per iteration
    std::cout << "  Average alloc+free time: " << us << " us\n";

    // Print cache stats
    SizeBucketedPool::instance().print_stats();
}

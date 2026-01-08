/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#endif

namespace lfs::core {

// Enable/disable profiling at compile time (controlled by CMake)
#ifndef ENABLE_ALLOCATION_PROFILING
#define ENABLE_ALLOCATION_PROFILING 0 // Default to disabled if not set by CMake
#endif

    struct TensorAllocation {
        std::vector<size_t> shape;
        size_t bytes = 0;
        std::string dtype;
        int alloc_iteration = -1; // Iteration when allocated (-1 = unknown)

        std::string shape_str() const {
            if (shape.empty())
                return "[]";
            std::string s = "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0)
                    s += ", ";
                s += std::to_string(shape[i]);
            }
            s += "]";
            return s;
        }

        double mb() const {
            return bytes / (1024.0 * 1024.0);
        }
    };

    // Metadata for tracking active allocations (for lifetime analysis)
    struct AllocationMetadata {
        void* ptr = nullptr;
        std::vector<size_t> shape;
        size_t bytes = 0;
        std::string dtype;
        int alloc_iteration = -1;
        std::string location; // Stack trace

        std::string shape_str() const {
            if (shape.empty())
                return "[]";
            std::string s = "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0)
                    s += ", ";
                s += std::to_string(shape[i]);
            }
            s += "]";
            return s;
        }
    };

    struct AllocationSite {
        size_t total_bytes = 0;
        size_t count = 0;
        size_t peak_bytes = 0;
        std::vector<TensorAllocation> tensors; // Track individual tensor allocations

        void record(size_t bytes) {
            total_bytes += bytes;
            count++;
            if (bytes > peak_bytes) {
                peak_bytes = bytes;
            }
        }

        void record_tensor(const std::vector<size_t>& shape, size_t bytes, const std::string& dtype, int iteration = -1) {
            record(bytes);
            TensorAllocation tensor;
            tensor.shape = shape;
            tensor.bytes = bytes;
            tensor.dtype = dtype;
            tensor.alloc_iteration = iteration;
            tensors.push_back(tensor);
        }
    };

    class AllocationProfiler {
    public:
        static AllocationProfiler& instance() {
            static AllocationProfiler profiler;
            return profiler;
        }

        // Capture stack trace and record allocation
        void record_allocation(size_t bytes, int skip_frames = 2) {
            if constexpr (!ENABLE_ALLOCATION_PROFILING) {
                return;
            }

#ifdef __linux__
            std::string location = capture_stack_trace(skip_frames);

            // Record allocation
            std::lock_guard<std::mutex> lock(mutex_);
            sites_[location].record(bytes);
            total_allocs_++;
            total_bytes_ += bytes;
#endif
        }

        // Set current iteration number (call from training loop)
        void set_iteration(int iteration) {
            if constexpr (!ENABLE_ALLOCATION_PROFILING) {
                return;
            }
            current_iteration_.store(iteration, std::memory_order_relaxed);
        }

        // Record tensor allocation with shape and dtype (with pointer tracking for lifetime analysis)
        void record_tensor_allocation(void* ptr, const std::vector<size_t>& shape, size_t bytes,
                                      const std::string& dtype, int skip_frames = 2) {
            if constexpr (!ENABLE_ALLOCATION_PROFILING) {
                return;
            }

#ifdef __linux__
            int iteration = current_iteration_.load(std::memory_order_relaxed);
            std::string location = capture_stack_trace(skip_frames);

            // Record allocation
            std::lock_guard<std::mutex> lock(mutex_);
            sites_[location].record_tensor(shape, bytes, dtype, iteration);
            total_allocs_++;
            total_bytes_ += bytes;

            // Track active allocation for lifetime analysis
            if (ptr) {
                AllocationMetadata metadata;
                metadata.ptr = ptr;
                metadata.shape = shape;
                metadata.bytes = bytes;
                metadata.dtype = dtype;
                metadata.alloc_iteration = iteration;
                metadata.location = location;
                active_allocations_[ptr] = metadata;
            }
#endif
        }

        // Backward compatibility: without pointer tracking
        void record_tensor_allocation(const std::vector<size_t>& shape, size_t bytes,
                                      const std::string& dtype, int skip_frames = 2) {
            record_tensor_allocation(nullptr, shape, bytes, dtype, skip_frames);
        }

        // Record tensor deallocation (for lifetime tracking)
        void record_deallocation(void* ptr) {
            if constexpr (!ENABLE_ALLOCATION_PROFILING) {
                return;
            }

#ifdef __linux__
            if (!ptr)
                return;

            int current_iter = current_iteration_.load(std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(mutex_);

            auto it = active_allocations_.find(ptr);
            if (it != active_allocations_.end()) {
                const auto& metadata = it->second;
                int lifetime = current_iter - metadata.alloc_iteration;

                // Only track lifetimes if we have valid iteration numbers
                if (metadata.alloc_iteration >= 0 && lifetime >= 0) {
                    // Track by shape+dtype only (for summary)
                    std::string key = metadata.shape_str() + "|" + metadata.dtype;
                    lifetime_stats_[key].push_back(lifetime);

                    // Also track by shape+dtype+origin (for detailed investigation)
                    std::string detailed_key = key + "|" + metadata.location;
                    lifetime_stats_by_origin_[detailed_key].push_back(lifetime);
                }

                active_allocations_.erase(it);
            }
#endif
        }

    private:
        std::string capture_stack_trace(int skip_frames) {
#ifdef __linux__
            // Capture stack trace
            constexpr int MAX_FRAMES = 20;
            void* callstack[MAX_FRAMES];
            int frames = backtrace(callstack, MAX_FRAMES);

            if (frames <= skip_frames) {
                return "";
            }

            // Get symbols for the entire stack
            char** symbols = backtrace_symbols(callstack, frames);
            if (!symbols) {
                return "";
            }

            // Build location string from multiple frames to get context
            std::string location;
            std::vector<std::string> app_frames; // Collect application-level frames

            // Start from skip_frames and collect up to 10 frames (full call chain)
            for (int i = skip_frames; i < std::min(frames, skip_frames + 10); ++i) {
                std::string frame = symbols[i];

                // Extract function name between '(' and '+' or ')'
                size_t start = frame.find('(');
                size_t end = frame.find('+', start);
                if (end == std::string::npos) {
                    end = frame.find(')', start);
                }

                if (start != std::string::npos && end != std::string::npos && end > start + 1) {
                    std::string mangled = frame.substr(start + 1, end - start - 1);

                    // Try to demangle
                    int status = 0;
                    char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
                    if (status == 0 && demangled) {
                        // Shorten long template names
                        std::string func = demangled;
                        free(demangled);

                        // Remove template noise for readability
                        size_t template_start = func.find('<');
                        if (template_start != std::string::npos) {
                            func = func.substr(0, template_start);
                        }

                        // Filter out tensor library internals, libc frames, and boilerplate entry points
                        // We want to see APPLICATION code, not tensor/libc/framework implementation details
                        if (func.find("lfs::core::Tensor") != std::string::npos ||
                            func.find("lfs::core::tensor") != std::string::npos ||
                            func.find("lfs::core::TensorImpl") != std::string::npos ||
                            func.find("lfs::core::internal") != std::string::npos ||
                            func.find("lfs::core::Application::run") != std::string::npos ||
                            func.find("lfs::core::run_headless_app") != std::string::npos ||
                            func.find("__libc") != std::string::npos ||
                            func.find("_start") == 0 ||
                            func == "main") {
                            // Skip these frames - they're too low-level or boilerplate
                            continue;
                        }

                        // Try to resolve file:line using dladdr
                        Dl_info info;
                        if (dladdr(callstack[i], &info) && info.dli_fname && info.dli_saddr) {
                            // Calculate offset from function start
                            ptrdiff_t offset = (char*)callstack[i] - (char*)info.dli_saddr;
                            char addr_buf[128];
                            snprintf(addr_buf, sizeof(addr_buf), " [%s+%#tx]",
                                     info.dli_fname, offset);
                            func += addr_buf;
                        }

                        app_frames.push_back(func);
                    } else if (!mangled.empty()) {
                        app_frames.push_back(mangled);
                    }
                }
            }

            // Build location from application-level frames (show up to 5 most relevant frames)
            int num_frames_to_show = std::min(5, (int)app_frames.size());
            for (int i = 0; i < num_frames_to_show; ++i) {
                if (!location.empty())
                    location += " <- ";
                location += app_frames[i];
            }

            free(symbols);

            // Fallback if we couldn't extract anything
            if (location.empty()) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%p", callstack[skip_frames]);
                location = buf;
            }

            return location;
#else
            return "";
#endif
        }

    public:
        // Print top N allocation sites
        void print_top_allocators(int top_n = 20) {
            if constexpr (!ENABLE_ALLOCATION_PROFILING) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            // Convert map to vector for sorting
            std::vector<std::pair<std::string, AllocationSite>> sites_vec(sites_.begin(), sites_.end());

            // Sort by total bytes
            std::sort(sites_vec.begin(), sites_vec.end(),
                      [](const auto& a, const auto& b) {
                          return a.second.total_bytes > b.second.total_bytes;
                      });

            printf("\n========== TOP %d ALLOCATION SITES ==========\n", top_n);
            printf("Total allocations: %zu, Total bytes: %.2f GB\n\n",
                   total_allocs_.load(), total_bytes_.load() / (1024.0 * 1024.0 * 1024.0));

            printf("%12s | %8s | %12s | %s\n", "Total (MB)", "Count", "Avg (KB)", "Origin");
            printf("%s\n", std::string(160, '-').c_str());

            int count = 0;
            for (const auto& [location, site] : sites_vec) {
                if (count++ >= top_n)
                    break;

                double total_mb = site.total_bytes / (1024.0 * 1024.0);
                double avg_kb = (site.total_bytes / site.count) / 1024.0;

                // Use the full location string - NO truncation, let it display fully
                std::string origin = location;

                printf("%12.2f | %8zu | %12.2f | %s\n",
                       total_mb,
                       site.count,
                       avg_kb,
                       origin.c_str());

                // If we have tensor allocations, print them
                if (!site.tensors.empty()) {
                    // Group tensors by shape for better display
                    std::map<std::string, std::vector<const TensorAllocation*>> shape_groups;
                    for (const auto& tensor : site.tensors) {
                        std::string key = tensor.shape_str() + " " + tensor.dtype;
                        shape_groups[key].push_back(&tensor);
                    }

                    // Print tensor details
                    for (const auto& [shape_dtype, tensors] : shape_groups) {
                        double total_tensor_mb = 0;
                        for (const auto* t : tensors) {
                            total_tensor_mb += t->mb();
                        }

                        // Extract shape and dtype
                        size_t dtype_pos = shape_dtype.rfind(' ');
                        std::string shape = shape_dtype.substr(0, dtype_pos);
                        std::string dtype = shape_dtype.substr(dtype_pos + 1);

                        printf("  └─ %zu x %s %-10s | %12.2f MB total | %.2f MB each\n",
                               tensors.size(),
                               shape.c_str(),
                               dtype.c_str(),
                               total_tensor_mb,
                               total_tensor_mb / tensors.size());
                    }
                }
            }
            printf("\n");
        }

        // Print detailed tensor allocations grouped by shape
        void print_tensor_allocations(int limit = 50) {
            if constexpr (!ENABLE_ALLOCATION_PROFILING) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            // Group tensors by origin + shape + dtype
            struct TensorGroup {
                std::string origin;
                std::string shape;
                std::string dtype;
                size_t count = 0;
                size_t total_bytes = 0;
            };

            std::map<std::string, TensorGroup> groups;

            for (const auto& [location, site] : sites_) {
                for (const auto& tensor : site.tensors) {
                    // Use the full location string - NO truncation
                    std::string origin = location;

                    // Create unique key for grouping
                    std::string key = origin + "|" + tensor.shape_str() + "|" + tensor.dtype;

                    auto& group = groups[key];
                    if (group.count == 0) {
                        group.origin = origin;
                        group.shape = tensor.shape_str();
                        group.dtype = tensor.dtype;
                    }
                    group.count++;
                    group.total_bytes += tensor.bytes;
                }
            }

            // Convert to vector for sorting
            std::vector<TensorGroup> sorted_groups;
            for (const auto& [key, group] : groups) {
                sorted_groups.push_back(group);
            }

            // Sort by total bytes
            std::sort(sorted_groups.begin(), sorted_groups.end(),
                      [](const auto& a, const auto& b) {
                          return a.total_bytes > b.total_bytes;
                      });

            printf("\n========== TENSOR ALLOCATIONS (GROUPED BY SHAPE, TOP %d) ==========\n", limit);
            printf("%-30s | %-10s | %10s | %8s | %s\n", "Shape", "DType", "Size (MB)", "Count", "Origin");
            printf("%s\n", std::string(160, '-').c_str());

            int count = 0;
            for (const auto& group : sorted_groups) {
                if (count++ >= limit)
                    break;

                double total_mb = group.total_bytes / (1024.0 * 1024.0);
                printf("%-30s | %-10s | %10.2f | %8zu | %s\n",
                       group.shape.c_str(),
                       group.dtype.c_str(),
                       total_mb,
                       group.count,
                       group.origin.c_str());
            }
            printf("\n");
        }

        // Print lifetime statistics (how long tensors live before being freed)
        void print_lifetime_stats(int limit = 20) {
            if constexpr (!ENABLE_ALLOCATION_PROFILING) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            if (lifetime_stats_.empty()) {
                return; // No lifetime data collected yet
            }

            printf("\n========== TENSOR LIFETIME STATISTICS (TOP %d) ==========\n", limit);
            printf("%-30s | %-10s | %8s | %8s | %8s | %8s\n",
                   "Shape | DType", "Count", "Min", "Avg", "Max", "Median");
            printf("%s\n", std::string(100, '-').c_str());

            // Convert to vector for sorting
            struct LifetimeSummary {
                std::string key;
                int count;
                int min_lifetime;
                double avg_lifetime;
                int max_lifetime;
                int median_lifetime;
            };

            std::vector<LifetimeSummary> summaries;
            for (const auto& [key, lifetimes] : lifetime_stats_) {
                if (lifetimes.empty())
                    continue;

                std::vector<int> sorted = lifetimes;
                std::sort(sorted.begin(), sorted.end());

                int min_lt = sorted.front();
                int max_lt = sorted.back();
                double sum = 0;
                for (int lt : sorted)
                    sum += lt;
                double avg_lt = sum / sorted.size();
                int median_lt = sorted[sorted.size() / 2];

                summaries.push_back({key, (int)sorted.size(), min_lt, avg_lt, max_lt, median_lt});
            }

            // Sort by count (most frequent first)
            std::sort(summaries.begin(), summaries.end(),
                      [](const auto& a, const auto& b) { return a.count > b.count; });

            int count = 0;
            for (const auto& summary : summaries) {
                if (count++ >= limit)
                    break;

                // Truncate key if too long
                std::string display_key = summary.key;
                if (display_key.length() > 30) {
                    display_key = "..." + display_key.substr(display_key.length() - 27);
                }

                printf("%-30s | %-10d | %8d | %8.1f | %8d | %8d\n",
                       display_key.c_str(),
                       summary.count,
                       summary.min_lifetime,
                       summary.avg_lifetime,
                       summary.max_lifetime,
                       summary.median_lifetime);
            }
            printf("\n");
        }

        // Print detailed lifetime statistics by origin (where allocations come from)
        void print_lifetime_stats_by_origin(int limit = 20) {
            if constexpr (!ENABLE_ALLOCATION_PROFILING) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            if (lifetime_stats_by_origin_.empty()) {
                return; // No lifetime data collected yet
            }

            printf("\n========== ALLOCATION ORIGINS (HIGH FREQUENCY, TOP %d) ==========\n", limit);
            printf("%-25s | %-10s | %8s | %-60s\n",
                   "Shape | DType", "Count", "Lifetime", "Origin (Code Location)");
            printf("%s\n", std::string(130, '-').c_str());

            // Convert to vector for sorting
            struct OriginSummary {
                std::string shape_dtype;
                std::string origin;
                int count;
                double avg_lifetime;
            };

            std::vector<OriginSummary> summaries;
            for (const auto& [detailed_key, lifetimes] : lifetime_stats_by_origin_) {
                if (lifetimes.empty())
                    continue;

                // Parse detailed_key: "shape|dtype|origin"
                size_t first_pipe = detailed_key.find('|');
                size_t second_pipe = detailed_key.find('|', first_pipe + 1);
                if (first_pipe == std::string::npos || second_pipe == std::string::npos)
                    continue;

                std::string shape_dtype = detailed_key.substr(0, second_pipe);
                std::string origin = detailed_key.substr(second_pipe + 1);

                double sum = 0;
                for (int lt : lifetimes)
                    sum += lt;
                double avg_lt = sum / lifetimes.size();

                summaries.push_back({shape_dtype, origin, (int)lifetimes.size(), avg_lt});
            }

            // Sort by count (most frequent first)
            std::sort(summaries.begin(), summaries.end(),
                      [](const auto& a, const auto& b) { return a.count > b.count; });

            int count = 0;
            for (const auto& summary : summaries) {
                if (count++ >= limit)
                    break;

                // Truncate shape/dtype if too long
                std::string display_key = summary.shape_dtype;
                if (display_key.length() > 25) {
                    display_key = "..." + display_key.substr(display_key.length() - 22);
                }

                // Extract first function from origin (most relevant)
                std::string short_origin = summary.origin;
                size_t arrow_pos = short_origin.find(" <- ");
                if (arrow_pos != std::string::npos) {
                    short_origin = short_origin.substr(0, arrow_pos);
                }
                if (short_origin.length() > 60) {
                    short_origin = short_origin.substr(0, 57) + "...";
                }

                printf("%-25s | %-10d | %8.1f | %-60s\n",
                       display_key.c_str(),
                       summary.count,
                       summary.avg_lifetime,
                       short_origin.c_str());
            }
            printf("\n");
        }

        void print_active_allocations(int limit = 50) {
            if constexpr (!ENABLE_ALLOCATION_PROFILING) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            if (active_allocations_.empty()) {
                printf("\n========== ACTIVE ALLOCATIONS: NONE ==========\n");
                return;
            }

            struct ActiveGroup {
                std::string shape;
                std::string dtype;
                std::string origin;
                size_t count = 0;
                size_t total_bytes = 0;
                int oldest_iteration = INT_MAX;
                int newest_iteration = -1;
            };

            std::map<std::string, ActiveGroup> groups;
            size_t total_active_bytes = 0;

            for (const auto& [ptr, meta] : active_allocations_) {
                std::string key = meta.shape_str() + "|" + meta.dtype + "|" + meta.location;
                auto& g = groups[key];
                if (g.count == 0) {
                    g.shape = meta.shape_str();
                    g.dtype = meta.dtype;
                    g.origin = meta.location;
                }
                g.count++;
                g.total_bytes += meta.bytes;
                total_active_bytes += meta.bytes;
                if (meta.alloc_iteration >= 0) {
                    g.oldest_iteration = std::min(g.oldest_iteration, meta.alloc_iteration);
                    g.newest_iteration = std::max(g.newest_iteration, meta.alloc_iteration);
                }
            }

            std::vector<ActiveGroup> sorted;
            for (const auto& [k, g] : groups)
                sorted.push_back(g);
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.total_bytes > b.total_bytes; });

            printf("\n========== ACTIVE ALLOCATIONS (TOP %d) ==========\n", limit);
            printf("Total active: %zu allocations, %.2f GB\n\n",
                   active_allocations_.size(), total_active_bytes / (1024.0 * 1024.0 * 1024.0));

            printf("%-25s | %-10s | %10s | %6s | %8s | %s\n",
                   "Shape", "DType", "Size (MB)", "Count", "Iter Age", "Origin");
            printf("%s\n", std::string(130, '-').c_str());

            int current_iter = current_iteration_.load();
            int count = 0;
            for (const auto& g : sorted) {
                if (count++ >= limit)
                    break;
                double mb = g.total_bytes / (1024.0 * 1024.0);
                int age = (g.oldest_iteration >= 0 && g.oldest_iteration != INT_MAX && current_iter > 0)
                              ? current_iter - g.oldest_iteration
                              : -1;

                std::string short_origin = g.origin;
                size_t arrow_pos = short_origin.find(" <- ");
                if (arrow_pos != std::string::npos) {
                    short_origin = short_origin.substr(0, arrow_pos);
                }
                if (short_origin.length() > 50) {
                    short_origin = short_origin.substr(0, 47) + "...";
                }

                printf("%-25s | %-10s | %10.2f | %6zu | %8d | %s\n",
                       g.shape.c_str(), g.dtype.c_str(), mb, g.count, age, short_origin.c_str());
            }
            printf("\n");
        }

        void reset() {
            std::lock_guard<std::mutex> lock(mutex_);
            sites_.clear();
            total_allocs_ = 0;
            total_bytes_ = 0;
            active_allocations_.clear();
            lifetime_stats_.clear();
            lifetime_stats_by_origin_.clear();
        }

    private:
        AllocationProfiler() = default;

        std::mutex mutex_;
        std::map<std::string, AllocationSite> sites_;
        std::atomic<size_t> total_allocs_{0};
        std::atomic<size_t> total_bytes_{0};

        // Lifetime tracking
        std::atomic<int> current_iteration_{0};                            // Current training iteration
        std::unordered_map<void*, AllocationMetadata> active_allocations_; // Ptr -> allocation metadata
        std::map<std::string, std::vector<int>> lifetime_stats_;           // Shape|dtype -> list of lifetimes (in iterations)
        std::map<std::string, std::vector<int>> lifetime_stats_by_origin_; // Shape|dtype|origin -> list of lifetimes
    };

} // namespace lfs::core

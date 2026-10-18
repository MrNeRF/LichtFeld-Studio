/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_transfer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <core/logger.hpp>
#include <core/native_fs.hpp>
#include <core/path_utils.hpp>
#include <core/splat_data_transform.hpp>
#include <core/user_paths.hpp>
#include <core/uuid.hpp>
#include <deque>
#include <fstream>
#include <functional>
#include <io/exporter.hpp>
#include <mutex>
#include <stdexcept>
#include <thread>
namespace lfs::vis::gui {
    namespace {
        // A process-owned worker: futures never join on the UI thread. One export
        // runs at a time, and at most one further snapshot may wait in the queue.
        class TransferWorker {
            std::mutex mutex_;
            std::condition_variable_any ready_;
            std::deque<std::packaged_task<void()>> jobs_;
            std::jthread thread_;

        public:
            TransferWorker() : thread_([this](std::stop_token stop) {
                                   while (true) {
                                       std::packaged_task<void()> task;
                                       {
                                           std::unique_lock lock(mutex_);
                                           ready_.wait(lock, stop, [this] { return !jobs_.empty(); });
                                           if (stop.stop_requested() && jobs_.empty())
                                               return;
                                           task = std::move(jobs_.front());
                                           jobs_.pop_front();
                                       }
                                       task(); // packaged_task captures failures in its future.
                                   }
                               }) {}
            ~TransferWorker() {
                thread_.request_stop();
                ready_.notify_all();
            }
            template <class F>
            auto submit(F&& work) {
                using T = std::invoke_result_t<F>;
                std::packaged_task<T()> task(std::forward<F>(work));
                auto result = task.get_future();
                {
                    std::lock_guard lock(mutex_);
                    if (jobs_.size() >= 1)
                        throw std::runtime_error("Splat transfer queue is busy");
                    jobs_.emplace_back([task = std::move(task)]() mutable { task(); });
                }
                ready_.notify_one();
                return result;
            }
        };
        TransferWorker& transferWorker() {
            static TransferWorker worker;
            return worker;
        }
    } // namespace

    namespace {
        // The app's cache folder, not the system temp: /tmp is RAM-backed on many Linux systems and a
        // batch can be gigabytes.
        std::filesystem::path transferRoot(const SplatTransferOptions& options) {
            if (!options.staging_root.empty())
                return core::native_fs::absolute(options.staging_root);
            if (const auto paths = core::UserPaths::resolve())
                return paths->cacheDir() / "transfers";
            return core::native_fs::temp_directory_path() / "lichtfeld-transfers";
        }
    } // namespace

    void cleanupSplatTransfers(const std::filesystem::path& root, std::chrono::hours age) {
        namespace fs = core::native_fs;
        std::error_code ec;
        if (!fs::is_directory(root, ec) || fs::is_symlink(fs::symlink_status(root, ec)))
            return;
        const auto now = std::filesystem::file_time_type::clock::now();
        for (auto it = fs::directory_iterator(root, ec); !ec && it != std::filesystem::directory_iterator{}; it.increment(ec)) {
            const auto path = it->path();
            const auto name = path.filename().u8string();
            if (name.size() != 36 || name[8] != '-' || name[13] != '-' || name[18] != '-' || name[23] != '-')
                continue;
            if (name.find_first_not_of(u8"0123456789abcdef-") != std::u8string::npos)
                continue;
            if (fs::is_symlink(it->symlink_status(ec)) || !it->is_directory(ec)) {
                ec.clear();
                continue;
            }
            const auto marker = path / ".lichtfeld-transfer";
            const auto status = fs::symlink_status(marker, ec);
            if (ec || !fs::is_regular_file(status) || fs::is_symlink(status)) {
                ec.clear();
                continue;
            }
            const auto modified = fs::last_write_time(marker, ec);
            if (ec) {
                ec.clear();
                continue;
            }
            if (now - modified < age)
                continue;
            // Only our marked UUID directories under the caller's transfer root.
            fs::remove_all(path, ec);
            if (ec) {
                LOG_WARN("Cannot expire splat transfer: {}", ec.message());
                ec.clear();
            }
        }
    }

    static std::packaged_task<std::filesystem::path()> captureTransfer(
        const core::SplatData& model, glm::mat4 world,
        std::shared_ptr<std::atomic<bool>> cancelled, SplatTransferOptions options) {
        if (!cancelled)
            throw std::runtime_error("Missing cancellation state");
        if (cancelled->load())
            throw std::runtime_error("Splat preparation cancelled");
        if (!options.axes.valid())
            throw std::runtime_error("Invalid splat transfer axes");
        if (model.lod_tree && model.lod_tree->has_tree()) {
            auto leaves = core::extract_lod_leaves(model);
            if (!leaves)
                throw std::runtime_error(lfs::format_for_developer(leaves.error()));
            return captureTransfer(*leaves, world, cancelled, options);
        }
        if (model.size() == 0)
            throw std::runtime_error("The splat object is empty");
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                if (!std::isfinite(world[col][row]))
                    throw std::runtime_error("Non-finite object transform");
        if (std::abs(world[0][3]) > 1e-6f || std::abs(world[1][3]) > 1e-6f || std::abs(world[2][3]) > 1e-6f || std::abs(world[3][3] - 1.0f) > 1e-6f)
            throw std::runtime_error("Splat transfer requires an affine transform");
        const glm::mat3 linear(world);
        const float s2 = glm::dot(linear[0], linear[0]);
        if (!std::isfinite(s2) || s2 <= 0 || std::abs(glm::determinant(linear)) < 1e-12f)
            throw std::runtime_error("Invalid or singular object transform");
        const int degree = options.sh_degree < 0 ? model.get_max_sh_degree() : std::clamp(options.sh_degree, 0, model.get_max_sh_degree());
        if (degree > 0) {
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
        if (options.format != "ply" && options.format != "spz")
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
            options.name = id;
        if (options.training_step && *options.training_step >= 0)
            options.name += "_step_" + std::to_string(*options.training_step);
        // Prefix also avoids Windows reserved device names such as CON and AUX.
        options.name = "splat-" + options.name;
        return std::packaged_task<std::filesystem::path()>(
            [model = std::move(*snapshot), world, cancelled, id, options] {
                const auto root = transferRoot(options);
                const auto folder = root / id;
                const auto target = folder / core::utf8_to_path(options.name + "." + options.format);
                const auto staged = folder / core::utf8_to_path(options.name + ".partial." + options.format);
                core::native_fs::create_directories(root);
                // Reserve the UUID directory exclusively. Even an unexpected
                // UUID collision must not touch or clean somebody else's data.
                if (!core::native_fs::create_directory(folder))
                    throw std::runtime_error("Transfer folder already exists");
                try {
                    {
                        std::ofstream marker(lfs::core::native_fs::native_path(folder / ".lichtfeld-transfer"), std::ios::binary);
                        if (!marker)
                            throw std::runtime_error("Cannot mark transfer folder");
                        marker << "LichtFeld transfer v1\n";
                        marker.flush();
                        if (!marker)
                            throw std::runtime_error("Cannot write transfer marker");
                    }
                    if (cancelled->load())
                        throw std::runtime_error("Splat preparation cancelled");
                    // Transform positions, covariance and SH through one basis change.
                    core::transform(*model, options.axes.matrix() * world);
                    const auto progress = [cancelled, &options](float value, const std::string&) { return !cancelled->load() && (!options.progress || options.progress(value)); };
                    io::Result<void> result;
                    if (options.format == "ply")
                        result = io::save_ply(*model, {.output_path = staged, .binary = true, .progress_callback = progress});
                    else
                        result = io::save_spz(*model, {.output_path = staged, .version = 4, .progress_callback = progress});
                    if (!result)
                        throw std::runtime_error(result.error().format());
                    if (cancelled->load())
                        throw std::runtime_error("Splat preparation cancelled");
                    core::native_fs::rename(staged, target);
                    return target;
                } catch (...) {
                    std::error_code ec;
                    core::native_fs::remove(staged, ec);
                    core::native_fs::remove(folder / ".lichtfeld-transfer", ec);
                    core::native_fs::remove(folder, ec);
                    throw;
                }
            });
    }
    std::future<std::vector<std::filesystem::path>> prepareSplatTransferBatch(
        std::span<const SplatTransferSource> sources, std::shared_ptr<std::atomic<bool>> cancelled) {
        if (sources.empty())
            throw std::runtime_error("Empty transfer batch");
        // Bound the owned CPU snapshot before allocating any of it. Canonical
        // float SH can be larger than the compressed source tensor.
        constexpr size_t budget = size_t(2) * 1024 * 1024 * 1024;
        size_t bytes = 0;
        for (size_t index = 0; index < sources.size(); ++index) {
            const auto& source = sources[index];
            if (!source.model)
                throw SplatTransferError(index, "Missing splat source");
            const auto degree = source.model->get_max_sh_degree();
            const size_t per_point = sizeof(float) * (14 + 3 * (degree + 1) * (degree + 1)) + 1;
            if (source.model->size() > (budget - bytes) / per_point)
                throw SplatTransferError(index, "Drag snapshot exceeds 2 GiB; use the normal export panel");
            bytes += source.model->size() * per_point;
        }
        auto tasks = std::make_unique<std::vector<std::packaged_task<std::filesystem::path()>>>();
        tasks->reserve(sources.size());
        for (size_t index = 0; index < sources.size(); ++index) {
            const auto& source = sources[index];
            try {
                tasks->push_back(captureTransfer(*source.model, source.world, cancelled, source.options));
            } catch (const std::exception& error) {
                throw SplatTransferError(index, error.what());
            } catch (...) {
                throw SplatTransferError(index, "Unknown snapshot failure");
            }
        }
        return transferWorker().submit([tasks = std::move(tasks), cancelled, root = transferRoot(sources.front().options)]() mutable {
            // Receivers read on drop; an hour is ample for delayed readers and bounds what piles up on disk.
            cleanupSplatTransfers(root);
            std::vector<std::filesystem::path> paths;
            try {
                paths.reserve(tasks->size()); // Allocate bookkeeping before any file commits.
                for (size_t index = 0; index < tasks->size(); ++index) {
                    try {
                        auto& task = (*tasks)[index];
                        auto result = task.get_future();
                        task();
                        paths.push_back(result.get());
                    } catch (const std::exception& error) {
                        throw SplatTransferError(index, error.what());
                    } catch (...) {
                        throw SplatTransferError(index, "Unknown writer failure");
                    }
                }
                if (cancelled->load())
                    throw std::runtime_error("Splat batch cancelled before delivery");
                return paths;
            } catch (...) {
                cancelled->store(true);
                // These files have never been handed to a receiver; release all
                // completed siblings too if one item in the batch fails.
                for (const auto& path : paths) {
                    std::error_code ec;
                    core::native_fs::remove(path, ec);
                    core::native_fs::remove(path.parent_path() / ".lichtfeld-transfer", ec);
                    core::native_fs::remove(path.parent_path(), ec);
                }
                throw;
            }
        });
    }
} // namespace lfs::vis::gui

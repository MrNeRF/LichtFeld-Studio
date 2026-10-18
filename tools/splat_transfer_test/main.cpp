/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "formats/ply.hpp"
#include "formats/spz.hpp"
#include "gui/utils/file_transfer_progress.hpp"
#include "gui/utils/splat_transfer.hpp"
#include <algorithm>
#include <cmath>
#include <core/uuid.hpp>
#include <cuda_runtime_api.h>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
using namespace lfs::core;
static void require(bool yes, const char* message) {
    if (!yes)
        throw std::runtime_error(message);
}
static void checkNear(float a, float b) { require(std::abs(a - b) < 2e-5f, "Numeric mismatch"); }
static SplatData fixture(int degree) {
    const size_t k = (degree + 1) * (degree + 1) - 1;
    auto means = Tensor::from_vector(std::vector<float>{.3f, 1.2f, -.7f, 7, 8, 9, 2.4f, -.5f, 1.8f}, {3, 3}, Device::CPU);
    auto dc = Tensor::from_vector(std::vector<float>{.1f, .2f, -.3f, .4f, .5f, .6f, .7f, .8f, .9f}, {3, 1, 3}, Device::CPU);
    std::vector<float> coefficients(3 * k * 3);
    for (size_t i = 0; i < coefficients.size(); ++i)
        coefficients[i] = .01f * float(int(i % 17) - 8);
    auto rest = k ? Tensor::from_vector(coefficients, {3, k, 3}, Device::CPU) : Tensor{};
    auto scale = Tensor::from_vector(std::vector<float>{-2, -3, -4, -2, -3, -4, -2, -3, -4}, {3, 3}, Device::CPU);
    auto rotation = Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {3, 4}, Device::CPU);
    auto opacity = Tensor::from_vector(std::vector<float>{-1, 0, 1}, {3, 1}, Device::CPU);
    SplatData model(degree, std::move(means), std::move(dc), std::move(rest), std::move(scale), std::move(rotation), std::move(opacity), 1.0f);
    model.deleted() = Tensor::from_vector(std::initializer_list<bool>{false, true, false}, {3}, Device::CPU);
    model.notify_deleted_mask_changed();
    model.refresh_deleted_count();
    return model;
}
// The contract stages in its own folder, never in the user's transfer cache.
static const auto kStagingRoot = std::filesystem::temp_directory_path() / ("lichtfeld-transfer-contract-" + generate_uuid_v4().to_string());
// One-source batch: snapshot now, file from the worker.
static std::future<std::filesystem::path> transfer(const SplatData& model, glm::mat4 world,
                                                   std::shared_ptr<std::atomic<bool>> cancelled,
                                                   lfs::vis::gui::SplatTransferOptions options = {}) {
    if (options.staging_root.empty())
        options.staging_root = kStagingRoot;
    const lfs::vis::gui::SplatTransferSource source[] = {{&model, world, std::move(options)}};
    auto batch = lfs::vis::gui::prepareSplatTransferBatch(source, std::move(cancelled));
    return std::async(std::launch::deferred, [batch = std::move(batch)]() mutable { return batch.get().front(); });
}
int main(int argc, char** argv) {
    try {
        const bool require_cuda = argc == 2 && std::string_view(argv[1]) == "--require-cuda";
        require(argc == 1 || require_cuda, "Optional argument: --require-cuda");
        // CPU input tensors do not make this an accelerator-free contract:
        // snapshot hydration and resident SH storage use the core CUDA path.
        int device_count = 0;
        const auto cuda_status = cudaGetDeviceCount(&device_count);
        const bool unavailable = (cuda_status == cudaSuccess && device_count == 0) ||
                                 cuda_status == cudaErrorNoDevice ||
                                 cuda_status == cudaErrorInsufficientDriver ||
                                 cuda_status == cudaErrorInitializationError ||
                                 cuda_status == cudaErrorStubLibrary;
        if (unavailable) {
            std::cerr << (require_cuda ? "FAIL" : "SKIP")
                      << ": splat transfer contract requires a working CUDA device: "
                      << cudaGetErrorString(cuda_status) << ", device_count=" << device_count << '\n';
            return require_cuda ? 1 : 77;
        }
        require(cuda_status == cudaSuccess, "Unexpected CUDA error while probing export test availability");
        lfs::io::LoadOptions options;
        options.splat_tensor_allocator = [](TensorShape shape, size_t, DataType type, std::string_view) { return Tensor::empty(std::move(shape), Device::CPU, type); };
        for (const std::string format : {"ply", "spz"}) {
            for (const int step : {0, 43453}) {
                const auto path = transfer(fixture(0), glm::mat4(1), std::make_shared<std::atomic<bool>>(false),
                                           {.format = format, .name = "Model", .training_step = step})
                                      .get();
                require(path.filename() == "splat-Model_step_" + std::to_string(step) + "." + format,
                        "Training step missing from exported file name");
            }
        }
        {
            const auto path = transfer(fixture(0), glm::mat4(1), std::make_shared<std::atomic<bool>>(false),
                                       {.name = "Model"})
                                  .get();
            require(path.filename() == "splat-Model.ply", "Non-training name changed");
            auto loaded = lfs::io::load_ply(path, options);
            require(bool(loaded), "Default-axis PLY failed");
            const auto means = loaded->value.means().to(Device::CPU).contiguous();
            checkNear(means.ptr<float>()[0], .3f);
            checkNear(means.ptr<float>()[1], 1.2f);
            checkNear(means.ptr<float>()[2], -.7f);
            const auto long_name = transfer(fixture(0), glm::mat4(1), std::make_shared<std::atomic<bool>>(false),
                                            {.name = std::string(200, 'x'), .training_step = 43453})
                                       .get();
            require(long_name.filename().string().ends_with("_step_43453.ply"), "Long-name fallback lost training step");
        }
        for (const int up : {-2, 2, -3, 3}) {
            for (int degree = 0; degree <= 3; ++degree) {
                auto model = fixture(degree);
                const auto initial_sh = model.shN_canonical_cpu();
                auto path = transfer(model, glm::mat4(1), std::make_shared<std::atomic<bool>>(false), {.axes = {up}}).get();
                auto loaded = lfs::io::load_ply(path, options);
                require(bool(loaded), "Axis conversion PLY failed");
                const auto matrix = lfs::vis::SplatDragAxes{up}.matrix();
                const auto expected = matrix * glm::vec4(.3f, 1.2f, -.7f, 1);
                const auto positions = loaded->value.means().to(Device::CPU).contiguous();
                for (int component = 0; component < 3; ++component)
                    checkNear(positions.ptr<float>()[component], expected[component]);
                const int inverse_up = up == 3 ? -3 : up == -3 ? 3
                                                               : up;
                path = transfer(loaded->value, glm::mat4(1), std::make_shared<std::atomic<bool>>(false), {.axes = {inverse_up}}).get();
                auto restored = lfs::io::load_ply(path, options);
                require(bool(restored), "Inverse axis conversion PLY failed");
                const auto restored_positions = restored->value.means().to(Device::CPU).contiguous();
                const auto source_positions = model.means().to(Device::CPU).contiguous();
                const auto sh = restored->value.shN_canonical_cpu();
                const size_t coefficients = (degree + 1) * (degree + 1) - 1;
                for (size_t item = 0; item < 2; ++item) {
                    const size_t source = item == 0 ? 0 : 2;
                    for (size_t component = 0; component < 3; ++component)
                        checkNear(restored_positions.ptr<float>()[item * 3 + component], source_positions.ptr<float>()[source * 3 + component]);
                    for (size_t coefficient = 0; coefficient < coefficients * 3; ++coefficient)
                        checkNear(sh.ptr<float>()[item * coefficients * 3 + coefficient], initial_sh.ptr<float>()[source * coefficients * 3 + coefficient]);
                }
            }
        }
        for (int degree = 0; degree <= 3; ++degree) {
            auto model = fixture(degree);
            const auto original_rest = model.shN_canonical_cpu();
            model.set_active_sh_degree(0); // Export all stored SH even if the viewport displays only DC.
            auto world = glm::translate(glm::mat4(1), glm::vec3(1, 2, 3)) * glm::scale(glm::mat4(1), glm::vec3(2));
            auto future = transfer(model, world, std::make_shared<std::atomic<bool>>(false), {.name = degree == 2 ? "scena-è-東京" : "splat", .axes = lfs::vis::SplatDragAxes::blender()});
            model.means().fill_(999);
            const auto path = future.get();
            require(std::filesystem::is_regular_file(path), "No committed PLY");
            for (const auto& entry : std::filesystem::directory_iterator(path.parent_path()))
                require(entry.path().filename().u8string().find(u8".partial.") == std::u8string::npos, "Partial remains");
            auto loaded = lfs::io::load_ply(path, options);
            if (!loaded)
                throw std::runtime_error(lfs::format_for_developer(loaded.error()));
            auto& out = loaded->value;
            require(out.size() == 2, "Deleted mask not preserved");
            auto pos = out.means().to(Device::CPU).contiguous();
            const auto* p = pos.ptr<float>();
            const float expected[] = {1.6f, 1.6f, -4.4f, 5.8f, 6.6f, -1.0f};
            for (int i = 0; i < 6; ++i)
                checkNear(p[i], expected[i]);
            auto opacity = out.opacity_raw().to(Device::CPU).contiguous();
            checkNear(opacity.ptr<float>()[0], -1);
            checkNear(opacity.ptr<float>()[1], 1);
            auto dc = out.sh0().to(Device::CPU).contiguous();
            checkNear(dc.ptr<float>()[0], .1f);
            checkNear(dc.ptr<float>()[3], .7f);
            require(out.get_active_sh_degree() == degree, "SH degree changed");
            auto scales = out.scaling_raw().to(Device::CPU).contiguous();
            auto rotations = out.rotation_raw().to(Device::CPU).contiguous();
            const auto* q = rotations.ptr<float>();
            const auto r = glm::mat3_cast(glm::normalize(glm::quat(q[0], q[1], q[2], q[3])));
            glm::mat3 diag(0);
            for (int i = 0; i < 3; ++i)
                diag[i][i] = std::exp(2 * scales.ptr<float>()[i]);
            const auto covariance = r * diag * glm::transpose(r);
            checkNear(covariance[0][0], 4 * std::exp(-4.0f));
            checkNear(covariance[1][1], 4 * std::exp(-8.0f));
            checkNear(covariance[2][2], 4 * std::exp(-6.0f));
            if (degree == 1) {
                auto sh = out.shN_canonical_cpu();
                const float* a = sh.ptr<float>();
                const float* b = original_rest.ptr<float>();
                for (int c = 0; c < 3; ++c) {
                    checkNear(a[c], -b[3 + c]);
                    checkNear(a[3 + c], b[c]);
                    checkNear(a[6 + c], b[6 + c]);
                }
            }
            auto cancelled = std::make_shared<std::atomic<bool>>(true);
            bool rejected = false;
            try {
                transfer(fixture(degree), glm::mat4(1), cancelled).get();
            } catch (const std::exception&) { rejected = true; }
            require(rejected, "Cancellation ignored");
        }
        {
            auto lod = fixture(0);
            lod.lod_tree = std::make_unique<SplatLodTree>();
            lod.lod_tree->child_count = {2, 0, 0};
            lod.lod_tree->child_start = {1, 0, 0};
            lod.lod_tree->lod_level = {1, 0, 0};
            const auto path = transfer(lod, glm::mat4(1), std::make_shared<std::atomic<bool>>(false), {.axes = lfs::vis::SplatDragAxes::blender()}).get();
            auto loaded = lfs::io::load_ply(path, options);
            require(bool(loaded), "LOD leaf PLY failed");
            require(loaded->value.size() == 1, "LOD interior nodes or deleted leaves were exported");
            const auto means = loaded->value.means().to(Device::CPU).contiguous();
            checkNear(means.ptr<float>()[0], 2.4f);
            checkNear(means.ptr<float>()[1], 1.8f);
            checkNear(means.ptr<float>()[2], .5f);
        }
        {
            auto model = fixture(3);
            auto cancel = std::make_shared<std::atomic<bool>>(false);
            bool rejected_format = false;
            const auto path = transfer(model, glm::mat4(1), cancel,
                                       {.format = "ply", .name = "../../CON:hello", .sh_degree = 1, .axes = {}})
                                  .get();
            require(path.parent_path().parent_path() == kStagingRoot, "Name escaped transfer folder");
            auto loaded = lfs::io::load_ply(path, options);
            require(bool(loaded), "Custom PLY failed");
            require(loaded->value.get_active_sh_degree() == 1, "Custom SH degree ignored");
            auto pos = loaded->value.means().to(Device::CPU).contiguous();
            checkNear(pos.ptr<float>()[1], 1.2f);
            const auto compressed = transfer(fixture(0), glm::mat4(1), cancel,
                                             {.format = "spz", .name = "test", .axes = {}})
                                        .get();
            require(std::filesystem::file_size(compressed) > 32, "No SPZ exported");
            require(compressed.extension() == ".spz", "Wrong SPZ extension");
            rejected_format = false;
            try {
                transfer(fixture(0), glm::mat4(1), cancel, {.format = "sog", .name = "test"}).get();
            } catch (const std::exception&) { rejected_format = true; }
            require(rejected_format, "Drag export accepted SOG");
        }

        {
            const auto rejected = [](auto&& work) {
                bool failed = false;
                try {
                    work();
                } catch (const std::exception&) { failed = true; }
                require(failed, "Invalid transfer accepted");
            };
            rejected([] { transfer(fixture(0), glm::mat4(1), nullptr); });
            rejected([] { auto m = glm::mat4(1); m[0][0] = std::numeric_limits<float>::quiet_NaN(); transfer(fixture(0), m, std::make_shared<std::atomic<bool>>(false)); });
            rejected([] { transfer(fixture(0), glm::mat4(0), std::make_shared<std::atomic<bool>>(false)); });
            rejected([] { transfer(fixture(1), glm::scale(glm::mat4(1), glm::vec3(1, 2, 3)), std::make_shared<std::atomic<bool>>(false)); });
            rejected([] { transfer(fixture(0), glm::mat4(1), std::make_shared<std::atomic<bool>>(false), {.format = "bad"}); });
            rejected([] { lfs::vis::gui::prepareSplatTransferBatch({}, std::make_shared<std::atomic<bool>>(false)); });
            auto model = fixture(0);
            const lfs::vis::gui::SplatTransferSource sources[] = {{&model, glm::mat4(1), {.name = "first", .staging_root = kStagingRoot}},
                                                                  {&model, glm::mat4(1), {.name = "second", .staging_root = kStagingRoot}}};
            const auto paths = lfs::vis::gui::prepareSplatTransferBatch(sources, std::make_shared<std::atomic<bool>>(false)).get();
            require(paths.size() == 2 && paths[0] != paths[1], "Batch files collided");
            for (const auto& path : paths)
                require(std::filesystem::is_regular_file(path), "Batch did not commit every file");
            const lfs::vis::gui::SplatTransferSource invalid_sources[] = {
                {&model, glm::mat4(1), {}},
                {&model, glm::mat4(0), {}}};
            bool indexed_snapshot_error = false;
            try {
                lfs::vis::gui::prepareSplatTransferBatch(invalid_sources, std::make_shared<std::atomic<bool>>(false));
            } catch (const lfs::vis::gui::SplatTransferError& error) { indexed_snapshot_error = error.item_index == 1; }
            require(indexed_snapshot_error, "Snapshot failure lost its node index");
            const lfs::vis::gui::SplatTransferSource invalid_single[] = {{&model, glm::mat4(0), {}}};
            indexed_snapshot_error = false;
            try {
                lfs::vis::gui::prepareSplatTransferBatch(invalid_single, std::make_shared<std::atomic<bool>>(false));
            } catch (const lfs::vis::gui::SplatTransferError& error) { indexed_snapshot_error = error.item_index == 0; }
            require(indexed_snapshot_error, "Single-item failure lost its node index");
            auto cancel = std::make_shared<std::atomic<bool>>(false);
            bool reached = false;
            rejected([&] { transfer(fixture(0), glm::mat4(1), cancel,
                                    {.format = "spz", .progress = [&](float fraction) { if (fraction >= .4f) { reached = true; cancel->store(true); return false; } return true; }})
                               .get(); });
            require(reached && cancel->load(), "Cancellation during export was not exercised");
        }
        {
            // A forgotten future must not join a still-running export on the UI
            // thread; queue admission must remain bounded while it is running.
            std::promise<void> entered, release;
            auto gate = release.get_future().share();
            std::atomic<bool> signalled{false};
            std::atomic<bool> released{false};
            auto active = transfer(fixture(0), glm::mat4(1), std::make_shared<std::atomic<bool>>(false),
                                   {.progress = [&](float) { if (!signalled.exchange(true)) entered.set_value(); gate.wait(); return true; }});
            require(entered.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready, "Worker did not run");
            auto queued = transfer(fixture(0), glm::mat4(1), std::make_shared<std::atomic<bool>>(false));
            bool busy = false;
            try {
                transfer(fixture(0), glm::mat4(1), std::make_shared<std::atomic<bool>>(false));
            } catch (const std::exception&) { busy = true; }
            std::jthread watchdog([&] { if (gate.wait_for(std::chrono::seconds(1)) != std::future_status::ready && !released.exchange(true)) release.set_value(); });
            const auto before = std::chrono::steady_clock::now();
            active = {};
            const auto elapsed = std::chrono::steady_clock::now() - before;
            if (!released.exchange(true))
                release.set_value();
            queued.get();
            require(busy, "Unbounded transfer queue");
            require(elapsed < std::chrono::milliseconds(500), "Future destruction blocked the UI thread");
        }
        {
            const auto root = std::filesystem::temp_directory_path() / ("lichtfeld-transfer-test-" + generate_uuid_v4().to_string());
            const auto old = root / generate_uuid_v4().to_string();
            const auto fresh = root / generate_uuid_v4().to_string();
            const auto unmarked = root / generate_uuid_v4().to_string();
            std::filesystem::create_directories(root / std::filesystem::path(std::u8string(u8"東京")));
            for (const auto& dir : {old, fresh, unmarked})
                std::filesystem::create_directories(dir);
            std::ofstream(old / ".lichtfeld-transfer") << "owned";
            std::ofstream(fresh / ".lichtfeld-transfer") << "owned";
            std::ofstream(old / "file.ply") << "data";
            std::ofstream(unmarked / "user-file") << "preserve";
            // Delivered transfers expire after an hour; receivers read them on drop.
            std::filesystem::last_write_time(old / ".lichtfeld-transfer", std::filesystem::file_time_type::clock::now() - std::chrono::hours(2));
            lfs::vis::gui::cleanupSplatTransfers(root);
            require(!std::filesystem::exists(old), "Expired transfer retained");
            require(std::filesystem::exists(fresh), "Fresh transfer removed");
            require(std::filesystem::exists(unmarked / "user-file"), "Unmarked data removed");
            // This is a unique test-owned root created above, never a user path.
            std::filesystem::remove_all(root);
        }

        {
            const auto root = std::filesystem::temp_directory_path() / ("lichtfeld-rollback-test-" + generate_uuid_v4().to_string());
            std::filesystem::create_directories(root);
            auto model = fixture(0);
            const lfs::vis::gui::SplatTransferSource sources[] = {
                {&model, glm::mat4(1), {.name = "first", .staging_root = root}},
                {&model, glm::mat4(1), {.name = "second", .progress = [](float) -> bool { throw std::runtime_error("Synthetic writer failure"); }, .staging_root = root}}};
            bool rejected = false;
            try {
                lfs::vis::gui::prepareSplatTransferBatch(sources, std::make_shared<std::atomic<bool>>(false)).get();
            } catch (const lfs::vis::gui::SplatTransferError& error) { rejected = error.item_index == 1; }
            require(rejected && std::filesystem::is_empty(root), "Failed batch leaked partial or committed siblings");
            const lfs::vis::gui::SplatTransferSource single_failure[] = {
                {&model, glm::mat4(1), {.name = "single", .progress = [](float) -> bool { throw 42; }, .staging_root = root}}};
            rejected = false;
            try {
                lfs::vis::gui::prepareSplatTransferBatch(single_failure, std::make_shared<std::atomic<bool>>(false)).get();
            } catch (const lfs::vis::gui::SplatTransferError& error) { rejected = error.item_index == 0; }
            require(rejected && std::filesystem::is_empty(root), "Single unknown writer failure lost node identity or leaked files");

            const auto blocker = root / "not-a-directory";
            std::ofstream(blocker) << "preserve";
            rejected = false;
            try {
                transfer(model, glm::mat4(1), std::make_shared<std::atomic<bool>>(false), {.staging_root = blocker}).get();
            } catch (const std::exception&) { rejected = true; }
            require(rejected && std::filesystem::is_regular_file(blocker), "IO failure damaged an existing file");
            std::filesystem::remove_all(root); // Unique test-owned root above.
        }
        {
            const auto root = std::filesystem::temp_directory_path() / ("lichtfeld-sh-cap-" + generate_uuid_v4().to_string());
            for (int available = 0; available <= 3; ++available) {
                auto model = fixture(available);
                for (int cap = 0; cap <= 3; ++cap) {
                    for (const auto* format : {"ply", "spz"}) {
                        const auto path = transfer(model, glm::mat4(1), std::make_shared<std::atomic<bool>>(false),
                                                   {.format = format, .sh_degree = cap, .staging_root = root})
                                              .get();
                        const int expected = std::min(available, cap);
                        if (std::string_view(format) == "ply") {
                            auto loaded = lfs::io::load_ply(path, options);
                            require(bool(loaded) && loaded->value.get_max_sh_degree() == expected, "PLY SH cap increased or lost source degree");
                            require(loaded->value.size() == 2, "SH cap removed points");
                        } else {
                            auto loaded = lfs::io::load_spz(path);
                            require(bool(loaded) && loaded->get_max_sh_degree() == expected, "SPZ SH cap increased or lost source degree");
                            require(loaded->size() == 2, "SPZ SH cap removed points");
                        }
                    }
                }
            }
            const auto path = transfer(fixture(3), glm::scale(glm::mat4(1), glm::vec3(1, 2, 3)),
                                       std::make_shared<std::atomic<bool>>(false), {.sh_degree = 0, .staging_root = root})
                                  .get();
            auto loaded = lfs::io::load_ply(path, options);
            require(bool(loaded) && loaded->value.get_max_sh_degree() == 0, "Discarded SH still rejected an SH0 affine export");
            std::filesystem::remove_all(root); // Unique test-owned root.
        }

        {
            // Staging never overwrites an existing user file. Identical export
            // names live in distinct, exclusively reserved UUID directories.
            const auto root = std::filesystem::temp_directory_path() / ("lichtfeld-name-test-" + generate_uuid_v4().to_string());
            std::filesystem::create_directories(root);
            const auto existing = root / "splat-same.ply";
            std::ofstream(existing) << "user content";
            auto model = fixture(0);
            const auto token = std::make_shared<std::atomic<bool>>(false);
            const auto first = transfer(model, glm::mat4(1), token, {.name = "same", .staging_root = root}).get();
            const auto second = transfer(model, glm::mat4(1), token, {.name = "same", .staging_root = root}).get();
            require(first != second && first.filename() == second.filename(), "Readable names or isolated transfer roots changed");
            const auto long_first = transfer(model, glm::mat4(1), token, {.name = std::string(200, 'x'), .staging_root = root}).get();
            const auto long_second = transfer(model, glm::mat4(1), token, {.name = std::string(200, 'x'), .staging_root = root}).get();
            require(long_first.filename() != long_second.filename(), "Long batch names collided after fallback");
            require(long_first.stem().string() == "splat-" + long_first.parent_path().filename().string(), "Long filename lost its full unique staging identity");
            std::ifstream input(existing);
            std::string preserved;
            std::getline(input, preserved);
            input.close();
            require(preserved == "user content", "Staging overwrote a preexisting file");
            for (const auto& name : {"CON", "AUX. ", "../bad:<|name", "control\nname"}) {
                const auto path = transfer(model, glm::mat4(1), token, {.name = name, .staging_root = root}).get();
                const auto leaf = path.filename().u8string();
                require(leaf.starts_with(u8"splat-") && leaf.find_first_of(u8"<>:\"/\\|?*\n\r") == std::u8string::npos, "Unsafe exported leaf name");
                require(path.parent_path().parent_path() == root, "Node name escaped its transfer root");
            }
            std::filesystem::remove_all(root); // Unique test-owned root above.
        }
        {
            using namespace lfs::vis::gui;
            const auto root = std::filesystem::temp_directory_path() / ("lichtfeld-batch-progress-" + generate_uuid_v4().to_string());
            std::filesystem::create_directories(root);
            auto model = fixture(0);
            auto cancelled = std::make_shared<std::atomic<bool>>(false);
            FileTransferProgressState progress(cancelled);
            std::vector<int> observed;
            std::vector<SplatTransferSource> sources;
            for (size_t i = 0; i < 3; ++i) {
                sources.push_back({&model, glm::translate(glm::mat4(1), glm::vec3(float(i), 0, 0)), {.name = "member-" + std::to_string(i), .staging_root = root}});
                sources.back().options.progress = [&, i](float value) {
                    const bool keep = progress.report(i, 3, value);
                    observed.push_back(progress.completed.load());
                    return keep;
                };
            }
            const auto files = prepareSplatTransferBatch(sources, cancelled).get();
            require(files.size() == 3 && progress.completed.load() == 1000, "Batch progress did not finish at 100 percent");
            for (size_t i = 0; i < files.size(); ++i) {
                auto loaded = lfs::io::load_ply(files[i], options);
                require(bool(loaded) && loaded->value.size() == 2, "Batch lost an object or point");
                const auto positions = loaded->value.means().to(Device::CPU).contiguous();
                const auto source = model.means().to(Device::CPU).contiguous();
                checkNear(positions.ptr<float>()[0], source.ptr<float>()[0] + float(i));
            }
            require(std::is_sorted(observed.begin(), observed.end()), "Batch progress moved backwards");
            require(std::ranges::any_of(observed, [](int value) { return value > 0 && value < 500; }) && std::ranges::any_of(observed, [](int value) { return value >= 500 && value < 1000; }), "Batch progress omitted intermediate members");
            std::filesystem::remove_all(root); // Unique test-owned root.
            std::filesystem::create_directories(root);
            progress.completed.store(0);
            observed.clear();
            sources[1].options.progress = [&](float value) {
                if (value > 0) {
                    progress.requestCancel();
                    return false;
                }
                return true;
            };
            bool rejected = false;
            try {
                prepareSplatTransferBatch(sources, cancelled).get();
            } catch (const std::exception&) { rejected = true; }
            require(rejected && progress.user_cancelled.load() && std::filesystem::is_empty(root), "Mid-batch cancellation left exported siblings");
            std::filesystem::remove_all(root); // Unique test-owned root.
        }

        std::filesystem::remove_all(kStagingRoot); // Unique test-owned root.
        std::cout << "PASS snapshot, deleted mask, world axes, covariance, opacity, SH0-3 and cancellation\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

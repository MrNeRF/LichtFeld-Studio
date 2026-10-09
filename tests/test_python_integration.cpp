/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python/python_compat.hpp"
#include "visualizer/rendering/passes/vulkan_split_view_pass.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include <gtest/gtest.h>

#include <torch/torch.h>

#include "core/camera.hpp"
#include "core/error_bus.hpp"
#include "core/event_bridge/command_api.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/control_boundary.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/property_registry.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"
#include "gui/line_renderer.hpp"
#include "io/loader.hpp"
#include "operation/undo_history.hpp"
#include "python/gil.hpp"
#include "python/python_buffer_analysis.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "python_test_support.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "scene/scene_manager.hpp"
#include "screen/screen_service.hpp"
#include "tensor_test_support.hpp"
#include "visualizer/ipc/render_settings_convert.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/rendering/depth_window_state.hpp"
#include "visualizer/rendering/dirty_flags.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/rendering/rendering_types.hpp"
#include "visualizer/visualizer.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <numbers>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

class PythonIntegrationTest : public ::testing::Test {
protected:
    static void TearDownTestSuite() {
        // Release retained UI callbacks while Python and its GIL are alive.
        // Windows DLL detach must not destroy these objects after process exit.
        if (Py_IsInitialized()) {
            const lfs::python::GilAcquire gil;
            lfs::python::invoke_python_cleanup();
        }
    }

    void SetUp() override {
        const auto module_dir = lfs::test::findPythonModuleDir();
        ASSERT_FALSE(module_dir.empty()) << "Could not locate built lichtfeld module for Python tests";
        lfs::test::prependPythonPath(module_dir);
        (void)lfs::python::ensure_initialized();
    }

    std::filesystem::path createTempScript(const std::string& content) {
        auto temp_dir = std::filesystem::temp_directory_path();
        auto script_path = temp_dir / "test_script.py";
        std::ofstream ofs(script_path);
        ofs << content;
        ofs.close();
        return script_path;
    }
};

namespace {
    struct PythonTensorResult {
        std::vector<int64_t> shape;
        std::vector<float> values;
    };

    struct ScopedViewCallback {
        explicit ScopedViewCallback(lfs::vis::GetViewCallback callback) {
            lfs::vis::set_view_callback(std::move(callback));
        }

        ~ScopedViewCallback() {
            lfs::vis::set_view_callback(nullptr);
        }
    };

    struct ScopedRenderSettingsCallbacks {
        ScopedRenderSettingsCallbacks(lfs::vis::GetRenderSettingsCallback get_cb,
                                      lfs::vis::SetRenderSettingsCallback set_cb) {
            lfs::vis::set_render_settings_callbacks(std::move(get_cb), std::move(set_cb));
        }

        ~ScopedRenderSettingsCallbacks() {
            lfs::vis::set_render_settings_callbacks(nullptr, nullptr);
        }
    };

    struct ScopedCaptureViewportRenderCallback {
        explicit ScopedCaptureViewportRenderCallback(lfs::vis::CaptureViewportRenderCallback callback) {
            lfs::vis::set_capture_viewport_render_callback(std::move(callback));
        }

        ~ScopedCaptureViewportRenderCallback() {
            lfs::vis::set_capture_viewport_render_callback(nullptr);
        }
    };

    struct ScopedVisualizer {
        explicit ScopedVisualizer(lfs::vis::Visualizer* viewer)
            : previous_(lfs::python::get_visualizer()) {
            lfs::python::set_visualizer(viewer);
        }

        ~ScopedVisualizer() {
            lfs::python::set_visualizer(previous_);
        }

    private:
        lfs::vis::Visualizer* previous_;
    };

    class TestVisualizer final : public lfs::vis::Visualizer {
    public:
        void run() override {}
        void setParameters(const lfs::core::param::TrainingParameters&) override {}
        std::expected<void, std::string> loadPLY(const std::filesystem::path&) override {
            return std::unexpected("not implemented");
        }
        std::expected<void, std::string> addSplatFile(const std::filesystem::path&) override {
            return std::unexpected("not implemented");
        }
        std::expected<void, std::string> loadDataset(const std::filesystem::path&) override {
            return std::unexpected("not implemented");
        }
        std::expected<void, std::string> loadCheckpointForTraining(const std::filesystem::path&) override {
            return std::unexpected("not implemented");
        }
        void consolidateModels() override {}
        std::expected<void, std::string> clearScene() override {
            ++clear_calls;
            if (!clear_error.empty())
                return std::unexpected(clear_error);
            return {};
        }
        int clear_calls = 0;
        std::string clear_error;
        lfs::core::Scene& getScene() override { return scene_; }
        lfs::vis::SceneManager* getSceneManager() override { return nullptr; }
        lfs::vis::RenderingManager* getRenderingManager() override { return nullptr; }

        bool postWork(WorkItem work) override {
            ++post_work_calls;
            if (!accepts_posted_work) {
                if (work.cancel) {
                    work.cancel();
                }
                return false;
            }
            if (queue_posted_work) {
                {
                    std::lock_guard lock(queued_work_mutex);
                    queued_work.push_back(std::move(work));
                }
                queued_work_cv.notify_all();
                return true;
            }
            if (work.run) {
                work.run();
            }
            return true;
        }

        [[nodiscard]] bool isOnViewerThread() const override { return on_viewer_thread; }
        [[nodiscard]] bool acceptsPostedWork() const override { return accepts_posted_work; }
        void setShutdownRequestedCallback(std::function<void()>) override {}
        std::expected<void, std::string> startTraining() override {
            return std::unexpected("not implemented");
        }
        lfs::Result<void> projectSave(bool) override {
            return {};
        }
        lfs::Result<void> projectSaveAs(
            const std::filesystem::path&, bool) override {
            return {};
        }
        lfs::Result<void> projectCreateAt(
            const std::filesystem::path&,
            lfs::vis::ProjectSwitchDisposition,
            bool = false) override {
            return {};
        }
        lfs::Result<lfs::vis::ProjectOpenOutcome> projectOpen(
            const std::filesystem::path&,
            lfs::vis::ProjectSwitchDisposition) override {
            return project_open_outcome;
        }
        lfs::Result<void> projectCompact() override {
            return {};
        }
        lfs::Result<bool> projectIsDirty() override {
            return false;
        }
        lfs::Result<bool> projectHasPath() override {
            return false;
        }
        lfs::Result<lfs::vis::ProjectInfo>
        projectGetInfo() override {
            return lfs::vis::ProjectInfo{};
        }
        lfs::Result<std::optional<lfs::io::project::ProjectLicense>>
        projectGetLicense() override {
            return project_license_;
        }
        lfs::Result<void> projectSetLicense(
            const lfs::io::project::ProjectLicense& license) override {
            project_license_ = license;
            return {};
        }
        lfs::Result<void> projectClearLicense() override {
            project_license_.reset();
            return {};
        }

        lfs::Result<lfs::vis::ProjectWritePoll> projectPollWrite() override {
            poll_thread = std::this_thread::get_id();
            ++poll_calls;
            return lfs::vis::ProjectWritePoll{.generation = 42};
        }
        void projectWaitWrite() override {
            wait_thread = std::this_thread::get_id();
        }
        std::thread::id poll_thread;
        std::thread::id wait_thread;
        int poll_calls = 0;
        bool project_save_started = false;
        bool consumeProjectSaveStarted() override {
            return std::exchange(project_save_started, false);
        }

        [[nodiscard]] bool waitForQueuedWork(const std::chrono::milliseconds timeout, const size_t count = 1) {
            std::unique_lock lock(queued_work_mutex);
            return queued_work_cv.wait_for(lock, timeout, [this, count]() { return queued_work.size() >= count; });
        }

        bool runNextQueuedWork() {
            WorkItem work;
            {
                std::lock_guard lock(queued_work_mutex);
                if (queued_work.empty()) {
                    return false;
                }
                work = std::move(queued_work.front());
                queued_work.pop_front();
            }
            if (work.run) {
                work.run();
            }
            return true;
        }

        bool cancelNextQueuedWork() {
            WorkItem work;
            {
                std::lock_guard lock(queued_work_mutex);
                if (queued_work.empty())
                    return false;
                work = std::move(queued_work.front());
                queued_work.pop_front();
            }
            if (work.cancel)
                work.cancel();
            return true;
        }

        bool on_viewer_thread = false;
        bool accepts_posted_work = true;
        bool queue_posted_work = false;
        int post_work_calls = 0;
        lfs::vis::ProjectOpenOutcome
            project_open_outcome =
                lfs::vis::ProjectOpenOutcome::Opened;

    private:
        lfs::core::Scene scene_;
        std::optional<lfs::io::project::ProjectLicense> project_license_;
        std::mutex queued_work_mutex;
        std::condition_variable queued_work_cv;
        std::deque<WorkItem> queued_work;
    };

    struct ViewSettingCallbacks {
        lfs::vis::screen::ScreenService views;
        lfs::vis::RenderingManager manager{views};
        lfs::vis::ViewId first = views.activeView();
        lfs::vis::ViewId second = views.screen().split(lfs::vis::screen::AreaId{first}, lfs::vis::screen::SplitAxis::Columns, 0.5f).value;
        int commits = 0;
        ViewSettingCallbacks() {
            views.screen().setActiveView({first});
            lfs::vis::set_render_settings_callbacks([&]() { return std::optional{lfs::vis::to_proxy(manager.getSettings())}; }, nullptr);
            lfs::vis::set_view_render_settings_callbacks(
                [&]() -> std::optional<lfs::vis::RenderSettingsTarget> {
                    return views.read([&](const auto& screen) { return lfs::vis::RenderSettingsTarget{screen.activeView().value, views.screenEpoch()}; });
                },
                [&](const lfs::vis::RenderSettingsTarget target) -> std::optional<lfs::vis::RenderSettingsProxy> {
                    if (target.screen_epoch != views.screenEpoch())
                        return std::nullopt;
                    const auto settings = manager.trySettingsForView(target.view);
                    return settings ? std::optional{lfs::vis::to_proxy(*settings)} : std::nullopt;
                },
                [&](const lfs::vis::RenderSettingsTarget target, const lfs::vis::RenderSettingsProxy& proxy,
                    lfs::vis::RenderSettingsUpdateIntent) -> std::optional<lfs::vis::RenderSettingsProxy> {
                    if (target.screen_epoch != views.screenEpoch())
                        return std::nullopt;
                    auto settings = manager.trySettingsForView(target.view);
                    if (!settings)
                        return std::nullopt;
                    lfs::vis::apply_proxy(*settings, proxy);
                    if (!manager.updateSettingsForView(target.view, *settings))
                        return std::nullopt;
                    ++commits;
                    return lfs::vis::to_proxy(manager.settingsForView(target.view));
                });
        }
        ~ViewSettingCallbacks() {
            lfs::vis::set_view_render_settings_callbacks(nullptr, nullptr, nullptr);
            lfs::vis::set_render_settings_callbacks(nullptr, nullptr);
        }
    };

    std::unique_ptr<lfs::core::SplatData> makeSingleWhiteSplat(const float x, const float y, const float z) {
        using lfs::core::Device;
        using lfs::core::Tensor;

        return std::make_unique<lfs::core::SplatData>(
            0,
            Tensor::from_vector({x, y, z}, {size_t{1}, size_t{3}}, Device::CPU),
            Tensor::from_vector({1.0f, 1.0f, 1.0f}, {size_t{1}, size_t{1}, size_t{3}}, Device::CPU),
            Tensor::zeros({size_t{1}, size_t{0}, size_t{3}}, Device::CPU, lfs::core::DataType::Float32),
            Tensor::from_vector({1.5f, 1.5f, 1.5f}, {size_t{1}, size_t{3}}, Device::CPU),
            Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{4}}, Device::CPU),
            Tensor::from_vector({8.0f}, {size_t{1}, size_t{1}}, Device::CPU),
            1.0f);
    }

    std::shared_ptr<lfs::core::Tensor> makeViewportReadbackLikeImage() {
        auto image = lfs::core::Tensor::zeros(
            {size_t{3}, size_t{2}, size_t{2}},
            lfs::core::Device::CPU,
            lfs::core::DataType::Float32);
        auto* const ptr = image.ptr<float>();
        constexpr size_t width = 2;
        constexpr size_t height = 2;

        // Simulate raw bottom-left-origin readback: the logical top row is stored last.
        for (size_t col = 0; col < width; ++col) {
            ptr[0 * height * width + (height - 1) * width + col] = 10.0f;
        }

        return std::make_shared<lfs::core::Tensor>(std::move(image));
    }

    std::string consumePythonError() {
        if (!PyErr_Occurred()) {
            return "unknown Python error";
        }

        PyObject* type = nullptr;
        PyObject* value = nullptr;
        PyObject* traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);

        std::string message = "unknown Python error";
        if (value) {
            PyObject* as_str = PyObject_Str(value);
            if (as_str) {
                if (const char* utf8 = PyUnicode_AsUTF8(as_str)) {
                    message = utf8;
                }
                Py_DECREF(as_str);
            }
        }

        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        return message;
    }

    // Executes a snippet against a caller-owned globals dict so several snippets
    // can share retained Python objects across C++ state changes.
    void execPythonInGlobals(PyObject* globals, const std::string& script) {
        PyObject* exec_result = PyRun_String(script.c_str(), Py_file_input, globals, globals);
        if (!exec_result) {
            throw std::runtime_error(consumePythonError());
        }
        Py_DECREF(exec_result);
    }

    PythonTensorResult runPythonTensorSnippet(const std::string& script) {
        const lfs::python::GilAcquire gil;

        PyObject* globals = PyDict_New();
        PyObject* locals = PyDict_New();
        if (!globals || !locals) {
            Py_XDECREF(globals);
            Py_XDECREF(locals);
            throw std::runtime_error("Failed to allocate Python dictionaries");
        }

        PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());

        PyObject* exec_result = PyRun_String(script.c_str(), Py_file_input, globals, locals);
        if (!exec_result) {
            const auto error = consumePythonError();
            Py_DECREF(globals);
            Py_DECREF(locals);
            throw std::runtime_error(error);
        }
        Py_DECREF(exec_result);

        auto* const shape_obj = PyDict_GetItemString(locals, "result_shape");
        auto* const values_obj = PyDict_GetItemString(locals, "result_values");
        if (!shape_obj || !PyTuple_Check(shape_obj) || !values_obj || !PyList_Check(values_obj)) {
            Py_DECREF(globals);
            Py_DECREF(locals);
            throw std::runtime_error("Python snippet did not populate result_shape/result_values");
        }

        PythonTensorResult result;
        result.shape.reserve(static_cast<size_t>(PyTuple_Size(shape_obj)));
        for (Py_ssize_t i = 0; i < PyTuple_Size(shape_obj); ++i) {
            result.shape.push_back(PyLong_AsLongLong(PyTuple_GetItem(shape_obj, i)));
        }

        result.values.reserve(static_cast<size_t>(PyList_Size(values_obj)));
        for (Py_ssize_t i = 0; i < PyList_Size(values_obj); ++i) {
            result.values.push_back(static_cast<float>(PyFloat_AsDouble(PyList_GetItem(values_obj, i))));
        }

        Py_DECREF(globals);
        Py_DECREF(locals);
        return result;
    }

    std::vector<long long> runPythonHookContextSnippet(const std::string& registration_script,
                                                       const lfs::training::HookContext& snapshot_ctx,
                                                       const lfs::training::HookContext& callback_ctx) {
        lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());
        lfs::training::ControlBoundary::instance().clear_all();
        lfs::training::CommandCenter::instance().update_snapshot(
            snapshot_ctx,
            /*max_iterations=*/5000,
            /*is_paused=*/false,
            /*is_running=*/true,
            /*stop_requested=*/false,
            lfs::training::TrainingPhase::SafeControl);

        PyObject* globals = nullptr;
        {
            const lfs::python::GilAcquire gil;
            globals = PyDict_New();
            if (!globals) {
                throw std::runtime_error("Failed to allocate Python globals");
            }

            PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());
            PyObject* exec_result = PyRun_String(registration_script.c_str(), Py_file_input, globals, globals);
            if (!exec_result) {
                const auto error = consumePythonError();
                Py_DECREF(globals);
                throw std::runtime_error(error);
            }
            Py_DECREF(exec_result);
        }

        lfs::training::ControlBoundary::instance().notify(
            lfs::training::ControlHook::PostStep,
            callback_ctx);
        lfs::training::ControlBoundary::instance().drain_callbacks();

        std::vector<long long> result;
        {
            const lfs::python::GilAcquire gil;
            auto* records_obj = PyDict_GetItemString(globals, "records");
            if (!records_obj || !PyList_Check(records_obj) || PyList_Size(records_obj) != 1) {
                lfs::training::ControlBoundary::instance().clear_all();
                Py_DECREF(globals);
                throw std::runtime_error("Hook script did not record exactly one callback invocation");
            }

            auto* record = PyList_GetItem(records_obj, 0);
            if (!record || !PyTuple_Check(record)) {
                lfs::training::ControlBoundary::instance().clear_all();
                Py_DECREF(globals);
                throw std::runtime_error("Recorded hook result is not a tuple");
            }

            result.reserve(static_cast<size_t>(PyTuple_Size(record)));
            for (Py_ssize_t i = 0; i < PyTuple_Size(record); ++i) {
                result.push_back(PyLong_AsLongLong(PyTuple_GetItem(record, i)));
            }

            // Drop retained Python callbacks before releasing the globals dict.
            lfs::training::ControlBoundary::instance().clear_all();
            Py_DECREF(globals);
        }
        return result;
    }

    void comparePythonResultToTorch(const PythonTensorResult& custom,
                                    const torch::Tensor& reference,
                                    const std::string& context,
                                    const float rtol = 1e-5f,
                                    const float atol = 1e-7f) {
        const auto reference_cpu = reference.cpu().contiguous();

        ASSERT_EQ(custom.shape.size(), static_cast<size_t>(reference_cpu.dim())) << context << ": rank mismatch";
        for (size_t i = 0; i < custom.shape.size(); ++i) {
            ASSERT_EQ(custom.shape[i], reference_cpu.size(i))
                << context << ": shape mismatch at dim " << i;
        }

        const auto* const reference_values = reference_cpu.data_ptr<float>();
        ASSERT_EQ(custom.values.size(), static_cast<size_t>(reference_cpu.numel())) << context << ": numel mismatch";

        for (size_t i = 0; i < custom.values.size(); ++i) {
            const float diff = std::abs(custom.values[i] - reference_values[i]);
            const float threshold = atol + rtol * std::abs(reference_values[i]);
            EXPECT_LE(diff, threshold) << context << ": mismatch at index " << i;
        }
    }

    bool formatterUnavailable(const lfs::python::FormatResult& result) {
        return !result.success &&
               (result.error.find("uv not found") != std::string::npos ||
                result.error.find("Failed to create venv for black") != std::string::npos ||
                result.error.find("ImportError") != std::string::npos);
    }
} // namespace

TEST_F(PythonIntegrationTest, InitializationSucceeds) {
    // Just verify that initialization doesn't throw
    EXPECT_NO_THROW((void)lfs::python::ensure_initialized());
}

TEST_F(PythonIntegrationTest, SceneTimeCallbackSupportsSetTickClearAndUnsetTick) {
    struct SceneTimeCallbackReset {
        ~SceneTimeCallbackReset() { lfs::python::clear_scene_time_callback(); }
    } reset;

    lfs::python::clear_scene_time_callback();
    EXPECT_FALSE(lfs::python::has_scene_time_callback());
    EXPECT_NO_THROW(lfs::python::tick_scene_time_callback(-1.0f));

    std::vector<float> clip_times;
    lfs::python::set_scene_time_callback(
        [&clip_times](const float clip_time) { clip_times.push_back(clip_time); });
    EXPECT_TRUE(lfs::python::has_scene_time_callback());

    lfs::python::tick_scene_time_callback(0.0f);
    lfs::python::tick_scene_time_callback(1.25f);
    EXPECT_EQ(clip_times, (std::vector<float>{0.0f, 1.25f}));

    lfs::python::clear_scene_time_callback();
    EXPECT_FALSE(lfs::python::has_scene_time_callback());
    EXPECT_NO_THROW(lfs::python::tick_scene_time_callback(2.0f));
    EXPECT_EQ(clip_times, (std::vector<float>{0.0f, 1.25f}));
}

TEST_F(PythonIntegrationTest, SceneTimeCallbackIsSharedWithPythonExtension) {
    const lfs::python::GilAcquire gil;
    struct SceneTimeCallbackReset {
        ~SceneTimeCallbackReset() { lfs::python::clear_scene_time_callback(); }
    } reset;

    ASSERT_EQ(PyRun_SimpleString(
                  "import lichtfeld as lf\n"
                  "_scene_time_contract_ticks = []\n"
                  "lf.on_scene_time(_scene_time_contract_ticks.append)\n"),
              0);
    ASSERT_TRUE(lfs::python::has_scene_time_callback());
    lfs::python::tick_scene_time_callback(0.0f);
    lfs::python::tick_scene_time_callback(0.5f);
    lfs::python::tick_scene_time_callback(1.0f);
    ASSERT_EQ(PyRun_SimpleString(
                  "assert _scene_time_contract_ticks == [0.0, 0.5, 1.0]\n"
                  "lf.clear_scene_time()\n"
                  "del _scene_time_contract_ticks\n"),
              0);
    EXPECT_FALSE(lfs::python::has_scene_time_callback());
}

TEST_F(PythonIntegrationTest, ForcedInitFailureLatchesQueryableError) {
    // The forced-failure latch is terminal; the RAII guard restores the real
    // (Ready) latch even if an ASSERT below returns early, so sibling tests in
    // the shuffle are unaffected.
    struct InitLatchResetGuard {
        ~InitLatchResetGuard() {
            lfs::python::force_python_init_failure_for_testing(false);
            lfs::python::reset_python_init_state_for_testing();
        }
    } reset_guard;

    lfs::python::reset_python_init_state_for_testing();

    lfs::python::force_python_init_failure_for_testing(true);
    const lfs::Status first = lfs::python::ensure_initialized();
    ASSERT_FALSE(first.has_value());
    EXPECT_EQ(first.error().code(), lfs::ErrorCode::Unavailable);
    EXPECT_EQ(first.error().domain(), lfs::ErrorDomain::Python);

    const auto latched = lfs::python::init_state();
    EXPECT_EQ(latched.state, lfs::python::PyInitState::Failed);
    ASSERT_TRUE(latched.error.has_value());
    EXPECT_EQ(latched.error->code(), lfs::ErrorCode::Unavailable);

    // Idempotent: a second call re-returns the same latched failure.
    const lfs::Status second = lfs::python::ensure_initialized();
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code(), lfs::ErrorCode::Unavailable);
    EXPECT_EQ(second.error().domain(), lfs::ErrorDomain::Python);
}

TEST_F(PythonIntegrationTest, ReadyStateAfterSuccessfulInit) {
    lfs::python::reset_python_init_state_for_testing();
    EXPECT_TRUE(lfs::python::ensure_initialized().has_value());
    EXPECT_EQ(lfs::python::init_state().state, lfs::python::PyInitState::Ready);
}

TEST_F(PythonIntegrationTest, OutputCallbackCanBeSet) {
    bool callback_set = false;
    lfs::python::set_output_callback([&](const std::string&, bool) { callback_set = true; });
    EXPECT_TRUE(true); // If we got here, setting the callback didn't crash
}

TEST_F(PythonIntegrationTest, OutputRedirectCanBeInstalled) {
    // This should not throw
    EXPECT_NO_THROW(lfs::python::install_output_redirect());
}

TEST_F(PythonIntegrationTest, EmptyScriptListSucceeds) {
    auto result = lfs::python::run_scripts({});
    EXPECT_TRUE(result.has_value()) << "Empty script list should succeed";
}

TEST_F(PythonIntegrationTest, FormatPythonCodePreservesValidBlockIndentation) {
    const auto result = lfs::python::format_python_code("if True:\n    print('x')\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.code, "if True:\n    print(\"x\")\n");
}

TEST_F(PythonIntegrationTest, PythonBufferAnalysisReportsCleanCode) {
    const auto analysis = lfs::python::analyze_python_buffer("if True:\n    print('x')\n");
    EXPECT_TRUE(analysis.clean()) << analysis.summary;
    EXPECT_EQ(analysis.status, lfs::python::PythonBufferStatus::Clean);
}

TEST_F(PythonIntegrationTest, PythonBufferAnalysisReportsDiagnosticByteRange) {
    constexpr std::string_view code = "import os\nif True print('x')\n";
    const auto analysis = lfs::python::analyze_python_buffer(code);

    ASSERT_EQ(analysis.status, lfs::python::PythonBufferStatus::SyntaxError);
    ASSERT_FALSE(analysis.issues.empty());
    const auto& issue = analysis.issues.front();
    EXPECT_LE(issue.start_byte, issue.end_byte);
    EXPECT_LE(issue.end_byte, code.size());
    EXPECT_FALSE(issue.message.empty());
}

TEST_F(PythonIntegrationTest, PythonSyntaxDocumentExtractsSymbolsAndScope) {
    constexpr std::string_view code =
        "import os\n"
        "from pathlib import Path\n"
        "CONFIG = Path('x')\n"
        "\n"
        "@decorator\n"
        "class Tool:\n"
        "    def run(self):\n"
        "        pass\n"
        "\n"
        "@decorator\n"
        "def helper():\n"
        "    return CONFIG\n";

    lfs::python::PythonSyntaxDocument document;
    ASSERT_TRUE(document.reset(code));
    ASSERT_EQ(document.analysis().status, lfs::python::PythonBufferStatus::Clean);

    int imports = 0;
    int classes = 0;
    int functions = 0;
    int variables = 0;
    for (const auto& symbol : document.symbols()) {
        switch (symbol.kind) {
        case lfs::python::PythonSymbolKind::Import:
            ++imports;
            break;
        case lfs::python::PythonSymbolKind::Class:
            ++classes;
            EXPECT_EQ(symbol.name, "Tool");
            break;
        case lfs::python::PythonSymbolKind::Function:
            ++functions;
            EXPECT_TRUE(symbol.name == "run" || symbol.name == "helper");
            break;
        case lfs::python::PythonSymbolKind::Variable:
            ++variables;
            EXPECT_EQ(symbol.name, "CONFIG");
            break;
        }
    }

    EXPECT_EQ(imports, 2);
    EXPECT_EQ(classes, 1);
    EXPECT_EQ(functions, 2);
    EXPECT_EQ(variables, 1);
    EXPECT_FALSE(document.foldRanges().empty());
    EXPECT_FALSE(document.highlights().empty());
    EXPECT_EQ(document.scopeAt(code.find("pass")), "Tool.run");
}

TEST_F(PythonIntegrationTest, PythonSyntaxDocumentAppliesIncrementalEdits) {
    constexpr std::string_view original =
        "def run():\n"
        "    pass\n";
    std::string updated(original);
    const size_t replace_start = updated.find("pass");
    ASSERT_NE(replace_start, std::string::npos);
    constexpr std::string_view replacement = "if True pass";
    updated.replace(replace_start, std::string_view("pass").size(), replacement);

    lfs::python::PythonSyntaxDocument document;
    ASSERT_TRUE(document.reset(original));

    const lfs::python::PythonBufferEdit edit{
        .start_byte = replace_start,
        .old_end_byte = replace_start + std::string_view("pass").size(),
        .new_end_byte = replace_start + replacement.size(),
        .start_point = lfs::python::python_buffer_point_at_byte(original, replace_start),
        .old_end_point =
            lfs::python::python_buffer_point_at_byte(original, replace_start + std::string_view("pass").size()),
        .new_end_point = lfs::python::python_buffer_point_at_byte(updated, replace_start + replacement.size()),
    };
    const std::array edits{edit};

    ASSERT_TRUE(document.applyEditsAndReparse(updated, edits));
    EXPECT_EQ(document.analysis().status, lfs::python::PythonBufferStatus::SyntaxError);
    ASSERT_FALSE(document.analysis().issues.empty());
}

TEST_F(PythonIntegrationTest, PythonSyntaxDocumentExtractsFallbackHighlightCaptures) {
    constexpr std::string_view code =
        "@decorator\n"
        "def run():\n"
        "    return 42\n";

    lfs::python::PythonSyntaxDocument document;
    ASSERT_TRUE(document.reset(code));
    ASSERT_EQ(document.analysis().status, lfs::python::PythonBufferStatus::Clean);

    bool has_keyword = false;
    bool has_decorator = false;
    bool has_function = false;
    bool has_number = false;
    for (const auto& highlight : document.highlights()) {
        has_keyword |= highlight.kind == lfs::python::PythonHighlightKind::Keyword;
        has_decorator |= highlight.kind == lfs::python::PythonHighlightKind::Decorator;
        has_function |= highlight.kind == lfs::python::PythonHighlightKind::Function;
        has_number |= highlight.kind == lfs::python::PythonHighlightKind::Number;
        EXPECT_LT(highlight.start_byte, highlight.end_byte);
        EXPECT_LE(highlight.end_byte, code.size());
    }

    EXPECT_TRUE(has_keyword);
    EXPECT_TRUE(has_decorator);
    EXPECT_TRUE(has_function);
    EXPECT_TRUE(has_number);
}

TEST_F(PythonIntegrationTest, PythonSyntaxDocumentKeepsStructureDuringSyntaxError) {
    constexpr std::string_view valid =
        "class Tool:\n"
        "    def run(self):\n"
        "        pass\n";
    constexpr std::string_view invalid = "if True print('x')\n";

    lfs::python::PythonSyntaxDocument document;
    ASSERT_TRUE(document.reset(valid));
    ASSERT_EQ(document.analysis().status, lfs::python::PythonBufferStatus::Clean);
    ASSERT_FALSE(document.symbols().empty());
    ASSERT_FALSE(document.foldRanges().empty());

    ASSERT_TRUE(document.reset(invalid));
    EXPECT_EQ(document.analysis().status, lfs::python::PythonBufferStatus::SyntaxError);
    ASSERT_FALSE(document.symbols().empty());
    ASSERT_FALSE(document.foldRanges().empty());

    bool retained_class = false;
    bool retained_function = false;
    for (const auto& symbol : document.symbols()) {
        retained_class |= symbol.kind == lfs::python::PythonSymbolKind::Class && symbol.name == "Tool";
        retained_function |= symbol.kind == lfs::python::PythonSymbolKind::Function && symbol.name == "run";
    }
    EXPECT_TRUE(retained_class);
    EXPECT_TRUE(retained_function);
}

TEST_F(PythonIntegrationTest, FormatPythonCodeRejectsIndentedSnippetBeforeBlack) {
    const auto result = lfs::python::format_python_code("    if True:\n        print('x')\n");

    ASSERT_FALSE(result.success);
    EXPECT_NE(result.error.find("Python syntax error"), std::string::npos);
}

TEST_F(PythonIntegrationTest, FormatPythonCodeRejectsUnexpectedTopLevelIndentBeforeBlack) {
    const auto result = lfs::python::format_python_code(
        "import lichtfeld as lf\n    scene = lf.get_scene()\nprint('hello world')\n");

    ASSERT_FALSE(result.success);
    EXPECT_NE(result.error.find("Python syntax error"), std::string::npos);
}

TEST_F(PythonIntegrationTest, FormatPythonCodeRejectsLeadingPreambleBulletsBeforeBlack) {
    const auto result = lfs::python::format_python_code(
        "1. SOURCE_NAME if set\n"
        "2. currently selected node\n"
        "3. first splat node in the scene\n"
        "\n"
        "from pathlib import Path\n"
        "import lichtfeld as lf\n");

    ASSERT_FALSE(result.success);
    EXPECT_NE(result.error.find("Python syntax error"), std::string::npos);
}

TEST_F(PythonIntegrationTest, CleanPythonCodeRepairsUnindentedFunctionBlock) {
    const auto result = lfs::python::clean_python_code(
        "def _safe_path_component(text):\n"
        "stripped = str(text or \"\").strip()\n"
        "if not stripped:\n"
        "    return \"splat\"\n"
        "safe = \"\".join(ch if ch.isalnum() or ch in (\"-\", \"_\", \".\") else \"_\" for ch in stripped)\n"
        "safe = safe.strip(\"_\")\n"
        "return safe or \"splat\"\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_NE(result.code.find("def _safe_path_component(text):"), std::string::npos);
    EXPECT_NE(result.code.find("    stripped = str(text or \"\").strip()"), std::string::npos);
    EXPECT_NE(result.code.find("    if not stripped:"), std::string::npos);
    EXPECT_NE(result.code.find("        return \"splat\""), std::string::npos);
    EXPECT_NE(result.code.find("    return safe or \"splat\""), std::string::npos);
}

// Catches Format commenting out valid leading statements that the preamble heuristic does not recognize.
TEST_F(PythonIntegrationTest, FormatPythonCodeKeepsLeadingStatements) {
    const auto result = lfs::python::format_python_code("x=[1,2 ,3];print( 'a' ,x)\nfirst, second = 1, 2\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.code, "x = [1, 2, 3]\nprint(\"a\", x)\nfirst, second = 1, 2\n");
}

TEST_F(PythonIntegrationTest, CleanPythonCodeCommentsProsePreambleButKeepsAssignment) {
    const auto result = lfs::python::clean_python_code("Here is the script:\nscene = lf.get_scene()\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.code, "# Here is the script:\nscene = lf.get_scene()\n");
}

TEST_F(PythonIntegrationTest, FormatPythonCodeReportsSyntaxErrorWithoutUnexpectedResultFallback) {
    const auto result = lfs::python::format_python_code("import os\nif True print('x')\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
    EXPECT_EQ(result.error.find("unexpected result"), std::string::npos);
}

TEST_F(PythonIntegrationTest, LookAtReturnsVisualizerPoseTranslation) {
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
_rotation, translation = lf.look_at((1.0, 2.0, 3.0), (1.0, 2.0, 2.0))
result_shape = tuple(translation.shape)
result_values = translation.flatten().tolist()
)PY");

    ASSERT_EQ(result.shape.size(), static_cast<size_t>(1));
    EXPECT_EQ(result.shape[0], 3);
    ASSERT_EQ(result.values.size(), static_cast<size_t>(3));
    EXPECT_FLOAT_EQ(result.values[0], 1.0f);
    EXPECT_FLOAT_EQ(result.values[1], 2.0f);
    EXPECT_FLOAT_EQ(result.values[2], 3.0f);
}

TEST_F(PythonIntegrationTest, GetCurrentViewComputesHorizontalFovFromViewportAspect) {
    const ScopedViewCallback callback([]() -> std::optional<lfs::vis::ViewInfo> {
        lfs::vis::ViewInfo info{};
        info.rotation = {1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f};
        info.translation = {0.0f, 0.0f, 0.0f};
        info.width = 200;
        info.height = 100;
        info.fov = 60.0f;
        return info;
    });

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
view = lf.get_current_view()
result_shape = (2,)
result_values = [float(view.fov_x), float(view.fov_y)]
)PY");

    const float expected_fov_x =
        std::atan(std::tan(60.0f * std::numbers::pi_v<float> / 360.0f) * 2.0f) * 360.0f / std::numbers::pi_v<float>;
    ASSERT_EQ(result.shape.size(), static_cast<size_t>(1));
    EXPECT_EQ(result.shape[0], 2);
    ASSERT_EQ(result.values.size(), static_cast<size_t>(2));
    EXPECT_NEAR(result.values[0], expected_fov_x, 1e-4f);
    EXPECT_FLOAT_EQ(result.values[1], 60.0f);
}

TEST_F(PythonIntegrationTest, RenderViewRequiresActiveVisualizerRenderer) {
    lfs::core::Scene scene;
    scene.addSplat("single", makeSingleWhiteSplat(0.0f, 0.0f, 2.0f));
    const lfs::python::SceneContextGuard scene_guard(&scene);

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
rotation, translation = lf.look_at((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
img = lf.render_view(rotation, translation, 64, 64, fov=60.0)
result_shape = (1,)
result_values = [1.0 if img is None else 0.0]
)PY");

    ASSERT_EQ(result.shape.size(), static_cast<size_t>(1));
    EXPECT_EQ(result.shape[0], 1);
    ASSERT_EQ(result.values.size(), static_cast<size_t>(1));
    EXPECT_FLOAT_EQ(result.values[0], 1.0f);
}

TEST_F(PythonIntegrationTest, CaptureViewportMatchesViewportVerticalOrientation) {
    const ScopedCaptureViewportRenderCallback callback([]() -> std::optional<lfs::vis::ViewportRender> {
        return lfs::vis::ViewportRender{makeViewportReadbackLikeImage(), nullptr};
    });

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
viewport = lf.capture_viewport()
img = viewport.image.cpu().tolist()
top = sum(pixel[0] for pixel in img[0])
bottom = sum(pixel[0] for pixel in img[1])
result_shape = (2,)
result_values = [float(top), float(bottom)]
)PY");

    ASSERT_EQ(result.shape.size(), static_cast<size_t>(1));
    EXPECT_EQ(result.shape[0], 2);
    ASSERT_EQ(result.values.size(), static_cast<size_t>(2));
    EXPECT_GT(result.values[0], result.values[1]);
}

TEST_F(PythonIntegrationTest, CaptureSplitComparisonPreservesPresentedOrientation) {
    for (const bool flip_y : {false, true}) {
        const ScopedCaptureViewportRenderCallback callback([flip_y]() -> std::optional<lfs::vis::ViewportRender> {
            constexpr size_t width = 64;
            constexpr size_t height = 8;
            std::vector<float> pixels(3 * width * height, 0.0f);
            for (size_t x = 0; x < width; ++x) {
                pixels[(flip_y ? height - 1 : 0) * width + x] = 1.0f;
            }
            const auto image = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::from_vector(
                pixels, {3, height, width}, lfs::core::Device::CPU));
            lfs::vis::SplitViewCpuDesc params;
            params.left.image = params.right.image = image;
            params.left.flip_y = params.right.flip_y = flip_y;
            params.content_rect = {0, 0, width, height};
            return lfs::vis::ViewportRender{
                lfs::vis::RenderingManager::composeSplitViewCpu(params, {width, height}), nullptr};
        });
        const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
image = lf.capture_viewport().image.cpu().tolist()
result_shape = (4,)
result_values = [image[0][0][0], image[-1][0][0], image[0][-1][0], image[-1][-1][0]]
)PY");
        ASSERT_EQ(result.values.size(), 4u);
        EXPECT_FLOAT_EQ(result.values[0], 1.0f);
        EXPECT_FLOAT_EQ(result.values[1], 0.0f);
        EXPECT_FLOAT_EQ(result.values[2], 1.0f);
        EXPECT_FLOAT_EQ(result.values[3], 0.0f);
    }
}

TEST_F(PythonIntegrationTest, CaptureViewportPostsToViewerThreadWhenOffThread) {
    TestVisualizer viewer;
    const ScopedVisualizer scoped_viewer(&viewer);
    const ScopedCaptureViewportRenderCallback callback([]() -> std::optional<lfs::vis::ViewportRender> {
        return lfs::vis::ViewportRender{makeViewportReadbackLikeImage(), nullptr};
    });

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
viewport = lf.capture_viewport()
result_shape = (1,)
result_values = [float(viewport.image.cpu().sum().item())]
)PY");

    ASSERT_EQ(result.shape.size(), static_cast<size_t>(1));
    EXPECT_EQ(result.shape[0], 1);
    ASSERT_EQ(result.values.size(), static_cast<size_t>(1));
    EXPECT_GT(result.values[0], 0.0f);
    EXPECT_EQ(viewer.post_work_calls, 1);
}

TEST_F(PythonIntegrationTest, ScreenMutationsPostToViewerThreadWhenOffThread) {
    TestVisualizer viewer;
    const ScopedVisualizer scoped_viewer(&viewer);

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
lf.ui.screen.reset()
lf.toggle_split_viewport()
result_shape = (1,)
result_values = [1.0]
)PY");

    ASSERT_EQ(result.values.size(), 1u);
    EXPECT_FLOAT_EQ(result.values[0], 1.0F);
    EXPECT_EQ(viewer.post_work_calls, 2);
}

TEST_F(PythonIntegrationTest,
       ProjectOpenSurfacesRecoveryPromptPendingOutcome) {
    TestVisualizer viewer;
    viewer.project_open_outcome =
        lfs::vis::ProjectOpenOutcome::
            RecoveryPromptPending;
    const ScopedVisualizer scoped_viewer(&viewer);

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
outcome = lf.project_open("pending.licht", discard_changes=True)
result_shape = (1,)
result_values = [1.0 if outcome is lf.ProjectOpenOutcome.RECOVERY_PROMPT_PENDING else 0.0]
)PY");

    ASSERT_EQ(result.values.size(), 1u);
    EXPECT_FLOAT_EQ(result.values[0], 1.0F);
    EXPECT_EQ(viewer.post_work_calls, 1);
}

TEST_F(PythonIntegrationTest, ProjectWritePollRunsOnViewerThread) {
    using namespace std::chrono_literals;
    TestVisualizer viewer;
    viewer.queue_posted_work = true;
    const ScopedVisualizer scoped_viewer(&viewer);
    std::jthread viewer_worker([&]() {
        if (viewer.waitForQueuedWork(1s)) {
            // Acquiring the GIL here also checks that the caller releases it
            // while waiting for the viewer to settle the write.
            const lfs::python::GilAcquire gil;
            viewer.runNextQueuedWork();
        }
    });
    const auto viewer_thread = viewer_worker.get_id();
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
state = lf.project_poll_write()
result_shape = (1,)
result_values = [float(state['generation'])]
)PY");
    viewer_worker.join();
    ASSERT_EQ(result.values.size(), 1u);
    EXPECT_FLOAT_EQ(result.values[0], 42.0F);
    EXPECT_EQ(viewer.poll_calls, 1);
    EXPECT_EQ(viewer.poll_thread, viewer_thread);
    EXPECT_EQ(viewer.wait_thread, std::thread::id{});
}

TEST_F(PythonIntegrationTest, ProjectPreviewWaitPollsOnViewerThread) {
    using namespace std::chrono_literals;
    TestVisualizer viewer;
    viewer.queue_posted_work = true;
    const ScopedVisualizer scoped_viewer(&viewer);
    std::jthread viewer_worker([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            if (viewer.waitForQueuedWork(100ms)) {
                const lfs::python::GilAcquire gil;
                viewer.runNextQueuedWork();
            }
        }
    });
    const auto viewer_thread = viewer_worker.get_id();
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
result_shape = (1,)
result_values = [float(lf.project_set_preview(b'preview', wait=True))]
)PY");
    viewer_worker.request_stop();
    viewer_worker.join();
    ASSERT_EQ(result.values.size(), 1u);
    EXPECT_FLOAT_EQ(result.values[0], 1.0F);
    EXPECT_EQ(viewer.poll_calls, 1);
    EXPECT_EQ(viewer.poll_thread, viewer_thread);
    EXPECT_EQ(viewer.wait_thread, viewer_thread);
}

TEST_F(PythonIntegrationTest, ProjectSaveWaitPollsOnViewerThread) {
    using namespace std::chrono_literals;
    TestVisualizer viewer;
    viewer.queue_posted_work = true;
    viewer.project_save_started = true;
    const ScopedVisualizer scoped_viewer(&viewer);
    std::jthread viewer_worker([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            if (viewer.waitForQueuedWork(100ms)) {
                const lfs::python::GilAcquire gil;
                viewer.runNextQueuedWork();
            }
        }
    });
    const auto viewer_thread = viewer_worker.get_id();
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
result_shape = (1,)
result_values = [float(lf.project_save(wait=True))]
)PY");
    viewer_worker.request_stop();
    viewer_worker.join();
    ASSERT_EQ(result.values.size(), 1u);
    EXPECT_FLOAT_EQ(result.values[0], 1.0F);
    EXPECT_EQ(viewer.poll_calls, 1);
    EXPECT_EQ(viewer.poll_thread, viewer_thread);
    EXPECT_EQ(viewer.wait_thread, viewer_thread);
}

TEST_F(PythonIntegrationTest, ProjectWritePollRunsInlineOnViewerThread) {
    TestVisualizer viewer;
    viewer.on_viewer_thread = true;
    const ScopedVisualizer scoped_viewer(&viewer);
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
result_shape = (1,)
result_values = [float(lf.project_poll_write()['generation'])]
)PY");
    ASSERT_EQ(result.values.size(), 1u);
    EXPECT_FLOAT_EQ(result.values[0], 42.0F);
    EXPECT_EQ(viewer.poll_thread, std::this_thread::get_id());
    EXPECT_EQ(viewer.post_work_calls, 0);
    EXPECT_EQ(viewer.wait_thread, std::thread::id{});
}

TEST_F(PythonIntegrationTest, ProjectWritePollRejectsViewerShutdown) {
    TestVisualizer viewer;
    viewer.accepts_posted_work = false;
    const ScopedVisualizer scoped_viewer(&viewer);
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
result_shape = (1,)
try:
    lf.project_poll_write()
    result_values = [0.0]
except RuntimeError:
    result_values = [1.0]
)PY");
    ASSERT_EQ(result.values.size(), 1u);
    EXPECT_FLOAT_EQ(result.values[0], 1.0F);
    EXPECT_EQ(viewer.poll_calls, 0);
}

TEST_F(PythonIntegrationTest, SceneClearPreservesTypedShutdownError) {
    for (const bool inline_call : {false, true}) {
        TestVisualizer viewer;
        viewer.on_viewer_thread = inline_call;
        viewer.accepts_posted_work = false;
        const ScopedVisualizer scoped_viewer(&viewer);
        const lfs::python::SceneContextGuard scene_guard(&viewer.getScene());
        const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
result_shape = (2,)
result_values = []
for clear in (lf.clear_scene, lf.get_scene().clear):
    try:
        clear()
        result_values.append(0.0)
    except lf.CancelledError as error:
        assert error.code == 'Cancelled'
        assert error.domain == 'Python'
        result_values.append(1.0)
)PY");
        EXPECT_EQ(result.values, (std::vector<float>{1.0F, 1.0F}));
        EXPECT_EQ(viewer.clear_calls, 0);
    }
}

TEST_F(PythonIntegrationTest, SceneClearPreservesLegacyFailureContext) {
    for (const bool inline_call : {false, true}) {
        TestVisualizer viewer;
        viewer.on_viewer_thread = inline_call;
        viewer.clear_error = "Scene is busy";
        const ScopedVisualizer scoped_viewer(&viewer);
        const lfs::python::SceneContextGuard scene_guard(&viewer.getScene());
        const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
result_shape = (2,)
result_values = []
for clear in (lf.clear_scene, lf.get_scene().clear):
    try:
        clear()
        result_values.append(0.0)
    except lf.Error as error:
        assert error.code == 'FailedPrecondition'
        assert error.domain == 'Rendering'
        assert str(error) == 'Scene is busy'
        assert error.context
        result_values.append(1.0)
)PY");
        EXPECT_EQ(result.values, (std::vector<float>{1.0F, 1.0F}));
        EXPECT_EQ(viewer.clear_calls, 2);
    }
}

TEST_F(PythonIntegrationTest, ProjectLicenseRoundTripsThroughBinding) {
    TestVisualizer viewer;
    const ScopedVisualizer scoped_viewer(&viewer);

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
assert lf.project_get_license() is None
lf.project_set_license("CC BY-NC", "Use with attribution")
license = lf.project_get_license()
set_ok = license == {"identifier": "CC BY-NC", "notice": "Use with attribution"}
lf.project_clear_license()
clear_ok = lf.project_get_license() is None
result_shape = (2,)
result_values = [1.0 if set_ok else 0.0, 1.0 if clear_ok else 0.0]
)PY");

    ASSERT_EQ(result.values.size(), 2u);
    EXPECT_FLOAT_EQ(result.values[0], 1.0F);
    EXPECT_FLOAT_EQ(result.values[1], 1.0F);
}

TEST_F(PythonIntegrationTest, GTComparisonActionsRunOnViewerThread) {
    using namespace std::chrono_literals;
    lfs::vis::screen::ScreenService screens;
    lfs::vis::SceneManager scene_manager;
    lfs::vis::RenderingManager rendering(screens);
    scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::Dataset);
    auto& scene = scene_manager.getScene();
    const auto group = scene.addCameraGroup("Cameras", scene.addGroup("Dataset"), 1);
    scene.addCamera("source", group, std::make_shared<lfs::core::Camera>(lfs::core::Tensor::eye(3, lfs::core::Device::CPU), lfs::core::Tensor::zeros({size_t{3}}, lfs::core::Device::CPU), 100.0f, 100.0f, 32.0f, 32.0f, lfs::core::Tensor(), lfs::core::Tensor(), lfs::core::CameraModelType::PINHOLE, "source", std::filesystem::path("source.png"), std::filesystem::path{}, 64, 64, 1));
    struct RestoreManagers {
        lfs::vis::SceneManager* scene = lfs::vis::services().sceneOrNull();
        lfs::vis::RenderingManager* rendering = lfs::python::get_rendering_manager();
        ~RestoreManagers() {
            lfs::vis::services().set(scene);
            lfs::python::set_rendering_manager(rendering);
        }
    } restore_managers;
    lfs::vis::services().set(&scene_manager);
    lfs::python::set_rendering_manager(&rendering);
    TestVisualizer viewer;
    viewer.queue_posted_work = true;
    const ScopedVisualizer scoped_viewer(&viewer);
    const auto viewer_thread = std::this_thread::get_id();
    std::vector<int> actions;
    std::vector<std::thread::id> action_threads;
    lfs::event::ScopedHandler handlers;
    handlers.subscribe<lfs::core::events::cmd::GoToCamView>([&](const auto& event) {
        actions.push_back(event.cam_id);
        action_threads.push_back(std::this_thread::get_id());
    });
    handlers.subscribe<lfs::core::events::cmd::ToggleGTComparison>([&](const auto&) {
        actions.push_back(-1);
        action_threads.push_back(std::this_thread::get_id());
    });

    std::exception_ptr python_error;
    PythonTensorResult result;
    std::thread python_worker([&] {
        try {
            result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
for uid in (7, 13, 7):
    lf.ui.go_to_camera_view(uid)
    lf.ui.toggle_gt_comparison()
result_shape = (1,)
result_values = [1.0]
)PY");
        } catch (...) {
            python_error = std::current_exception();
        }
    });
    for (int i = 0; i < 6; ++i) {
        if (!viewer.waitForQueuedWork(5s)) {
            ADD_FAILURE() << "Camera action was not posted to the viewer";
            break;
        }
        EXPECT_TRUE(viewer.runNextQueuedWork());
    }
    python_worker.join();
    if (python_error) {
        std::rethrow_exception(python_error);
    }
    EXPECT_EQ(result.values, std::vector<float>{1.0f});
    EXPECT_EQ(actions, (std::vector<int>{7, -1, 13, -1, 7, -1}));
    EXPECT_EQ(action_threads, std::vector<std::thread::id>(6, viewer_thread));
    EXPECT_EQ(viewer.post_work_calls, 6);
}

TEST_F(PythonIntegrationTest, CaptureViewportReleasesGilWhileWaitingForViewerThread) {
    using namespace std::chrono_literals;

    TestVisualizer viewer;
    viewer.queue_posted_work = true;
    const ScopedVisualizer scoped_viewer(&viewer);
    const ScopedCaptureViewportRenderCallback callback([]() -> std::optional<lfs::vis::ViewportRender> {
        return lfs::vis::ViewportRender{makeViewportReadbackLikeImage(), nullptr};
    });

    std::atomic_bool python_call_completed = false;
    std::atomic_bool gil_acquired_during_call = false;
    std::atomic_bool probe_started = false;
    std::atomic_bool queued_work_seen = false;
    std::atomic_bool viewer_work_ran = false;

    std::thread gil_probe([&]() {
        queued_work_seen.store(viewer.waitForQueuedWork(1s), std::memory_order_release);
        if (!queued_work_seen.load(std::memory_order_acquire)) {
            return;
        }
        probe_started.store(true, std::memory_order_release);
        const lfs::python::GilAcquire gil;
        gil_acquired_during_call.store(!python_call_completed.load(std::memory_order_acquire),
                                       std::memory_order_release);
    });

    std::thread viewer_worker([&]() {
        if (!viewer.waitForQueuedWork(1s)) {
            return;
        }
        const auto deadline = std::chrono::steady_clock::now() + 500ms;
        while (std::chrono::steady_clock::now() < deadline &&
               (!probe_started.load(std::memory_order_acquire) ||
                !gil_acquired_during_call.load(std::memory_order_acquire))) {
            std::this_thread::sleep_for(10ms);
        }
        viewer_work_ran.store(viewer.runNextQueuedWork(), std::memory_order_release);
    });

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
viewport = lf.capture_viewport()
result_shape = (1,)
result_values = [float(viewport.image.cpu().sum().item())]
)PY");
    python_call_completed.store(true, std::memory_order_release);

    viewer_worker.join();
    gil_probe.join();

    ASSERT_EQ(result.shape.size(), static_cast<size_t>(1));
    EXPECT_EQ(result.shape[0], 1);
    ASSERT_EQ(result.values.size(), static_cast<size_t>(1));
    EXPECT_GT(result.values[0], 0.0f);
    EXPECT_TRUE(queued_work_seen.load(std::memory_order_acquire));
    EXPECT_TRUE(viewer_work_ran.load(std::memory_order_acquire));
    EXPECT_TRUE(gil_acquired_during_call.load(std::memory_order_acquire));
}

TEST_F(PythonIntegrationTest, SceneCameraExposesVisualizerRenderContract) {
    const auto dataset_dir = std::filesystem::path(PROJECT_ROOT_PATH) / "data" / "bicycle";
    if (!std::filesystem::exists(dataset_dir / "sparse")) {
        GTEST_SKIP() << "bicycle sparse data not available";
    }

    auto loader = lfs::io::Loader::create();
    lfs::io::LoadOptions options;
    // The committed masks are quarter-resolution and intentionally pair with images_4.
    options.resize_factor = 4;
    options.images_folder = "images_4";

    auto load_result = loader->load(dataset_dir, options);
    ASSERT_TRUE(load_result.has_value()) << "Failed to load dataset: " << load_result.error().format();
    ASSERT_TRUE(std::holds_alternative<lfs::io::LoadedScene>(load_result->data));
    const auto& loaded_scene = std::get<lfs::io::LoadedScene>(load_result->data);
    ASSERT_FALSE(loaded_scene.cameras.empty());
    const auto& raw_camera = *loaded_scene.cameras.front();

    const auto expected_pose = lfs::rendering::visualizerCameraPoseFromDataWorldToCamera(
        lfs::rendering::mat3FromRowMajor3x3(static_cast<const float*>(raw_camera.R().cpu().contiguous().data_ptr())),
        [&]() {
            const auto translation = raw_camera.T().cpu().contiguous();
            const auto* ptr = static_cast<const float*>(translation.data_ptr());
            return glm::vec3(ptr[0], ptr[1], ptr[2]);
        }());
    const auto expected_view = lfs::rendering::makeViewMatrix(expected_pose.rotation, expected_pose.translation);
    const float expected_fov_x = raw_camera.FoVx() * 180.0f / std::numbers::pi_v<float>;
    const float expected_fov_y = raw_camera.FoVy() * 180.0f / std::numbers::pi_v<float>;
    const auto expected_raw_rotation = raw_camera.R().cpu().contiguous();
    const auto expected_raw_translation = raw_camera.T().cpu().contiguous();
    const auto expected_raw_view = raw_camera.world_view_transform().cpu().contiguous();
    const auto expected_raw_position = raw_camera.cam_position().cpu().contiguous();

    const auto script = std::string(R"PY(
import lichtfeld as lf
import warnings
result = lf.io.load(r")PY") +
                        dataset_dir.string() + R"PY(", resize_factor=4, images_folder="images_4")
camera = result.cameras[0]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    raw_rotation = camera.R.cpu().flatten().tolist()
    raw_translation = camera.T.cpu().flatten().tolist()
    raw_view = camera.world_view_transform.cpu().flatten().tolist()
    raw_position = camera.cam_position.cpu().flatten().tolist()
result_shape = (64,)
result_values = (
    camera.rotation.cpu().flatten().tolist()
    + camera.translation.cpu().flatten().tolist()
    + camera.view_matrix.cpu().flatten().tolist()
    + [float(camera.fov_x), float(camera.fov_y)]
    + raw_rotation
    + raw_translation
    + raw_view
    + raw_position
    + [
        float(len(caught)),
        1.0 if all(issubclass(w.category, DeprecationWarning) for w in caught) else 0.0,
        1.0 if all("deprecated" in str(w.message) for w in caught) else 0.0,
    ]
)
)PY";
    const auto result = runPythonTensorSnippet(script);

    ASSERT_EQ(result.shape.size(), static_cast<size_t>(1));
    EXPECT_EQ(result.shape[0], 64);
    ASSERT_EQ(result.values.size(), static_cast<size_t>(64));

    size_t index = 0;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            EXPECT_NEAR(result.values[index++], expected_pose.rotation[col][row], 1e-5f);
        }
    }
    EXPECT_NEAR(result.values[index++], expected_pose.translation.x, 1e-5f);
    EXPECT_NEAR(result.values[index++], expected_pose.translation.y, 1e-5f);
    EXPECT_NEAR(result.values[index++], expected_pose.translation.z, 1e-5f);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            EXPECT_NEAR(result.values[index++], expected_view[col][row], 1e-5f);
        }
    }
    EXPECT_NEAR(result.values[index++], expected_fov_x, 1e-4f);
    EXPECT_NEAR(result.values[index++], expected_fov_y, 1e-4f);
    for (int i = 0; i < expected_raw_rotation.numel(); ++i) {
        EXPECT_NEAR(result.values[index++], expected_raw_rotation.ptr<float>()[i], 1e-5f);
    }
    for (int i = 0; i < expected_raw_translation.numel(); ++i) {
        EXPECT_NEAR(result.values[index++], expected_raw_translation.ptr<float>()[i], 1e-5f);
    }
    for (int i = 0; i < expected_raw_view.numel(); ++i) {
        EXPECT_NEAR(result.values[index++], expected_raw_view.ptr<float>()[i], 1e-5f);
    }
    for (int i = 0; i < expected_raw_position.numel(); ++i) {
        EXPECT_NEAR(result.values[index++], expected_raw_position.ptr<float>()[i], 1e-5f);
    }
    EXPECT_FLOAT_EQ(result.values[index++], 4.0f);
    EXPECT_FLOAT_EQ(result.values[index++], 1.0f);
    EXPECT_FLOAT_EQ(result.values[index++], 1.0f);
}

TEST_F(PythonIntegrationTest, RestoredTensorOperations) {
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
x = lf.Tensor.normal([2, 2], mean=2, std=0, device="gpu")
assert x.backend in ("cuda", "vulkan", "metal")
assert x.all_close(lf.Tensor.ones([2, 2], device="gpu") * 2)
assert x.allclose(x.clone())
assert lf.Tensor.bernoulli([4], p=1, device="gpu").sum().item() == 4
weights = lf.Tensor.ones([4], device="gpu")
assert sorted(lf.Tensor.multinomial(weights, 4, seed=42).tolist()) == [0, 1, 2, 3]
assert lf.Tensor.diag(weights).sum().item() == 4
assert x.cdist(x).sum().item() == 0
assert x.normalize().sum().item() == 0
assert x.mod(x).sum().item() == 0
assert x.reduce(lf.ReduceOp.SUM).item() == 8
assert x.nonzero_split()[0].numel == 4
assert x.linear(x).sum().item() == 32
assert x.reshape([1, 2, 1, 2]).conv1x1(x).sum().item() == 32
x[0, 1] = -3
assert x[0, 1].item() == -3
half = x.to("float16")
assert half.clamp(-1, 1).to("float32").sum().item() == 2
assert half.clamp_min_(-1) is half
assert half.clamp_(-1, 1) is half
assert half.to("float32").sum().item() == 2
assert x.where_into_(x < 0, 5, x) is x
assert x[0, 1].item() == 5
indices = lf.Tensor.zeros([2], device="gpu", dtype="int32")
assert x.gather_lazy(indices).tolist() == [2, 2]
assert x.validate()["is_valid"]
assert x.diff(x.clone())["num_different"] == 0
assert x.stats()["backend"] == x.backend
assert x.reserved_allocation_bytes >= 16
assert lf.nn.softmax(x).sum().item() == 2
assert lf.nn.silu(x).all_close(x.swish())
assert lf.nn.rms_norm(x, lf.Tensor.ones([2], device="gpu")).shape == (2, 2)
assert lf.nn.residual_scale(x, x, lf.Tensor.ones([2], device="gpu")).all_close(x * 2)
sequence = half.reshape([1, 1, 2, 2])
windows = lf.nn.window_partition(sequence, 3)
assert lf.nn.window_unpartition(windows, 3, 2).to("float32").all_close(sequence.to("float32"))
result_shape = (1,)
result_values = [1.0]
)PY");
    ASSERT_EQ(result.values.size(), 1u);
    EXPECT_FLOAT_EQ(result.values[0], 1.0F);
}

TEST_F(PythonIntegrationTest, PyTensorSyncWaitsForItsVulkanBackendWithCudaDefault) {
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP();
    GpuBackendScope vulkan(GpuBackend::Vulkan);
    const lfs::python::GilAcquire gil;
    const auto decref = [](PyObject* object) { Py_XDECREF(object); };
    std::unique_ptr<PyObject, decltype(decref)> globals(PyDict_New(), decref);
    ASSERT_NE(globals.get(), nullptr);
    PyDict_SetItemString(globals.get(), "__builtins__", PyEval_GetBuiltins());
    execPythonInGlobals(globals.get(), R"PY(
import lichtfeld as lf
t = lf.Tensor.ones([257], device="gpu")
assert t.backend == "vulkan", t.backend
)PY");
    auto sentinel = Tensor::zeros({257}, Device::GPU);
    sentinel.fill_(7.f, nullptr);
    const auto pending = lfs::test::vulkan_pending_value(sentinel);
    ASSERT_GT(pending, lfs::test::vulkan_completed_value());
    {
        GpuBackendScope opposite(GpuBackend::CUDA);
        execPythonInGlobals(globals.get(), "t.sync()\n");
    }
    EXPECT_GE(lfs::test::vulkan_completed_value(), pending);
    // Drain even on a failed expectation, so the test leaves no queued work.
    TensorCompletion completion;
    completion.include(GpuBackend::Vulkan);
    completion.wait();
    EXPECT_EQ(sentinel.cpu().to_vector(), std::vector<float>(257, 7.f));
    for (const auto& message : lfs::test::vulkan_validation_messages()) {
        ADD_FAILURE() << message;
    }
}

TEST_F(PythonIntegrationTest, PyTensorBooleanRowMaskIndexingMatchesTorch) {
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
t = lf.Tensor.arange(1, 13, 1, device="cpu", dtype="float32").reshape([4, 3])
mask = lf.Tensor.arange(0, 4, 1, device="cpu", dtype="float32") != 1
selected = t[mask]
result_shape = tuple(selected.shape)
result_values = selected.flatten().tolist()
)PY");

    const auto torch_tensor = torch::arange(1, 13, torch::kFloat32).reshape({4, 3});
    const auto torch_mask = torch::tensor(std::vector<int>{1, 0, 1, 1}, torch::kInt32).to(torch::kBool);
    const auto torch_result = torch_tensor.index({torch_mask});

    comparePythonResultToTorch(result, torch_result, "PyTensor row mask");
}

TEST_F(PythonIntegrationTest, PyTensorElementwiseBooleanMaskIndexingMatchesTorch) {
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
t = lf.Tensor.arange(1, 13, 1, device="cpu", dtype="float32").reshape([4, 3])
mask = (t == 1) | (t == 4) | (t == 6) | (t == 9) | (t == 10)
selected = t[mask]
result_shape = tuple(selected.shape)
result_values = selected.tolist()
)PY");

    const auto torch_tensor = torch::arange(1, 13, torch::kFloat32).reshape({4, 3});
    const auto torch_mask =
        (torch_tensor == 1) |
        (torch_tensor == 4) |
        (torch_tensor == 6) |
        (torch_tensor == 9) |
        (torch_tensor == 10);
    const auto torch_result = torch_tensor.masked_select(torch_mask);

    comparePythonResultToTorch(result, torch_result, "PyTensor elementwise mask");
}

TEST_F(PythonIntegrationTest, DecoratorHookContextUsesLiveHookSnapshot) {
    const lfs::training::HookContext stale_snapshot{
        .iteration = 0,
        .loss = 0.0f,
        .num_gaussians = 0,
        .is_refining = false,
        .trainer = nullptr,
    };
    const lfs::training::HookContext live_callback{
        .iteration = 1047,
        .loss = 0.125f,
        .num_gaussians = 98765,
        .is_refining = true,
        .trainer = nullptr,
    };

    const auto result = runPythonHookContextSnippet(
        R"PY(
import lichtfeld as lf
records = []

@lf.on_post_step
def _hook(hook):
    ctx = lf.context()
    records.append((
        hook["iter"],
        hook["iteration"],
        hook["num_splats"],
        hook["num_gaussians"],
        ctx.iteration,
        ctx.num_gaussians,
    ))
)PY",
        stale_snapshot,
        live_callback);

    ASSERT_EQ(result.size(), 6u);
    EXPECT_EQ(result[0], live_callback.iteration);
    EXPECT_EQ(result[1], live_callback.iteration);
    EXPECT_EQ(result[2], static_cast<long long>(live_callback.num_gaussians));
    EXPECT_EQ(result[3], static_cast<long long>(live_callback.num_gaussians));
    EXPECT_EQ(result[4], live_callback.iteration);
    EXPECT_EQ(result[5], static_cast<long long>(live_callback.num_gaussians));
}

TEST_F(PythonIntegrationTest, ScopedHandlerHookContextUsesLiveHookSnapshot) {
    const lfs::training::HookContext stale_snapshot{
        .iteration = 0,
        .loss = 0.0f,
        .num_gaussians = 0,
        .is_refining = false,
        .trainer = nullptr,
    };
    const lfs::training::HookContext live_callback{
        .iteration = 1008,
        .loss = 0.25f,
        .num_gaussians = 54321,
        .is_refining = false,
        .trainer = nullptr,
    };

    const auto result = runPythonHookContextSnippet(
        R"PY(
import lichtfeld as lf
records = []
handler = lf.ScopedHandler()

def _hook(hook):
    ctx = lf.context()
    records.append((
        hook["iter"],
        hook["iteration"],
        hook["num_splats"],
        hook["num_gaussians"],
        ctx.iteration,
        ctx.num_gaussians,
    ))

handler.on_post_step(_hook)
)PY",
        stale_snapshot,
        live_callback);

    ASSERT_EQ(result.size(), 6u);
    EXPECT_EQ(result[0], live_callback.iteration);
    EXPECT_EQ(result[1], live_callback.iteration);
    EXPECT_EQ(result[2], static_cast<long long>(live_callback.num_gaussians));
    EXPECT_EQ(result[3], static_cast<long long>(live_callback.num_gaussians));
    EXPECT_EQ(result[4], live_callback.iteration);
    EXPECT_EQ(result[5], static_cast<long long>(live_callback.num_gaussians));
}

namespace {
    class CountingPythonInitConsumer final : public lfs::NativeErrorConsumer {
    public:
        void on_error(const lfs::ErrorNotification& notification,
                      const lfs::ErrorDeliveryInfo&) noexcept override {
            if (notification.error.domain() == lfs::ErrorDomain::Python &&
                notification.error.code() == lfs::ErrorCode::Unavailable)
                count.fetch_add(1, std::memory_order_relaxed);
        }
        std::atomic<int> count{0};
    };
} // namespace

TEST_F(PythonIntegrationTest, ConcurrentEnsureInitializedLatchesOnceUnderRace) {
    struct InitLatchResetGuard {
        ~InitLatchResetGuard() {
            lfs::python::force_python_init_failure_for_testing(false);
            lfs::python::reset_python_init_state_for_testing();
        }
    } reset_guard;

    lfs::python::reset_python_init_state_for_testing();

    CountingPythonInitConsumer consumer;
    auto subscription = lfs::ErrorBus::instance().subscribe(consumer);

    lfs::python::force_python_init_failure_for_testing(true);

    constexpr int kThreads = 8;
    std::vector<lfs::Status> results(kThreads);
    std::barrier start(kThreads);
    {
        std::vector<std::jthread> threads;
        threads.reserve(kThreads);
        for (int i = 0; i < kThreads; ++i) {
            threads.emplace_back([&, i] {
                start.arrive_and_wait();
                results[i] = lfs::python::ensure_initialized();
            });
        }
    }

    for (const lfs::Status& result : results) {
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::Unavailable);
        EXPECT_EQ(result.error().domain(), lfs::ErrorDomain::Python);
    }
    EXPECT_EQ(lfs::python::init_state().state, lfs::python::PyInitState::Failed);
    // The race-safety invariant: 8 concurrent callers never double-publish the
    // failure (the duplicate-toast hazard). Exact-once liveness depends on a
    // fresh process-global publish latch, which the test-only reset seam cannot
    // guarantee against sibling tests in a shared process; it is covered by the
    // single-threaded forced-failure path instead.
    EXPECT_LE(consumer.count.load(), 1);
}

// A retained Python settings proxy must re-read live settings before applying its named
// property after a focus change. Otherwise DirtyFlag::ALL back-routes the proxy's stale
// depth window into the newly focused panel.
namespace {
    // The real viewer registers the manager used by the compiled binding.
    // No window loop or SelectionTool is initialized: this is its no-tool lane.
} // namespace
// NOTE: Tests that actually execute Python scripts require the lichtfeld module
// to be importable, which depends on the CommandCenter and training infrastructure.
// These are better tested via integration tests (running training with --python-script).

TEST_F(PythonIntegrationTest, LoadConfirmationLegacyCallbackOpensDialog) {
    const lfs::python::GilAcquire gil;
    std::unique_ptr<PyObject, decltype(&Py_DecRef)> globals(PyDict_New(), Py_DecRef);
    ASSERT_NE(globals, nullptr);
    PyDict_SetItemString(globals.get(), "__builtins__", PyEval_GetBuiltins());
    const auto run = [&](const char* code) {
        auto* result = PyRun_String(code, Py_file_input, globals.get(), globals.get());
        if (!result)
            ADD_FAILURE() << consumePythonError();
        const bool success = result != nullptr;
        Py_XDECREF(result);
        return success;
    };
    ASSERT_TRUE(run(R"PY(
import lichtfeld as lf
calls = []
dialogs = []
original_dialog = lf.ui.confirm_dialog
lf.ui.confirm_dialog = lambda *args: dialogs.append(args)
def callback(paths, is_dataset, replace):
    calls.append((paths, is_dataset, replace))
    lf.ui.confirm_dialog('Load files', 'Confirm replacement', ['Load', 'Cancel'])
lf.ui.on_show_load_file_confirmation(callback)
)PY"));
    lfs::core::events::cmd::ShowLoadFileConfirmation{
        .paths = {"first.ply", "second.ply"},
        .is_dataset = false,
        .replace = true,
        .user_batch = true}
        .emit();
    EXPECT_TRUE(run(R"PY(
try:
    assert calls == [(['first.ply', 'second.ply'], False, True)], calls
    assert dialogs == [('Load files', 'Confirm replacement', ['Load', 'Cancel'])], dialogs
finally:
    lf.ui.on_show_load_file_confirmation(lambda paths, is_dataset, replace: None)
    lf.ui.confirm_dialog = original_dialog
)PY"));
}

TEST_F(PythonIntegrationTest, LoadConfirmationBatchCallbackRetainsProvenance) {
    const lfs::python::GilAcquire gil;
    std::unique_ptr<PyObject, decltype(&Py_DecRef)> globals(PyDict_New(), Py_DecRef);
    ASSERT_NE(globals, nullptr);
    PyDict_SetItemString(globals.get(), "__builtins__", PyEval_GetBuiltins());
    const auto run = [&](const char* code) {
        auto* result = PyRun_String(code, Py_file_input, globals.get(), globals.get());
        if (!result)
            ADD_FAILURE() << consumePythonError();
        const bool success = result != nullptr;
        Py_XDECREF(result);
        return success;
    };
    ASSERT_TRUE(run(R"PY(
import lichtfeld as lf
calls = []
lf.ui.on_show_load_file_confirmation(lambda *args: calls.append(('old', args)))
def callback(paths, is_dataset, replace, user_batch):
    calls.append((paths, is_dataset, replace, user_batch))
lf.ui.on_show_load_file_confirmation_with_batch(callback)
lf.ui.on_show_load_file_confirmation_with_batch(callback)
)PY"));
    lfs::core::events::cmd::ShowLoadFileConfirmation{
        .paths = {"first.ply", "second.ply"},
        .is_dataset = false,
        .replace = true,
        .user_batch = true}
        .emit();
    EXPECT_TRUE(run(R"PY(
try:
    assert calls == [(['first.ply', 'second.ply'], False, True, True)], calls
finally:
    lf.ui.on_show_load_file_confirmation_with_batch(lambda paths, is_dataset, replace, user_batch: None)
)PY"));
}

TEST_F(PythonIntegrationTest, CustomGizmoOverlayDispatchesPerViewAndRetainsInstance) {
    const lfs::python::GilAcquire gil;
    std::unique_ptr<PyObject, decltype(&Py_DecRef)> globals(PyDict_New(), Py_DecRef);
    PyDict_SetItemString(globals.get(), "__builtins__", PyEval_GetBuiltins());
    const auto run = [&](const char* code) {
        auto* result = PyRun_String(code, Py_file_input, globals.get(), globals.get());
        if (!result)
            ADD_FAILURE() << consumePythonError();
        const bool success = result != nullptr;
        Py_XDECREF(result);
        return success;
    };
    ASSERT_TRUE(run(R"PY(
import lichtfeld as lf
calls = []
instances = []
class CustomOverlay:
    gizmo_id = "test.custom.overlay"
    enabled = True
    def __init__(self):
        self.frames = 0
        instances.append(self)
    @classmethod
    def poll(cls, ctx):
        return cls.enabled
    def draw(self, ctx):
        self.frames += 1
        point = ctx.world_to_screen((0, 0, 0))
        calls.append((self.frames, point, ctx.camera_position, ctx.camera_forward,
                      ctx.screen_to_world_ray(point)))
        ctx.draw_line_3d((0, 0, 0), (1, 0, 0), (1, 0, 0, 1), 2)
        ctx.draw_filled_circle(point, 4, (0, 1, 0, 1))
lf.register_gizmo(CustomOverlay)
)PY"));
    EXPECT_TRUE(lfs::python::has_viewport_draw_handlers());
    const glm::mat4 proj = glm::perspective(glm::radians(60.0f), 2.0f, 0.1f, 100.0f);
    const glm::vec2 size(400, 200);
    lfs::vis::gui::NativeOverlayDrawList draw_list;
    lfs::rendering::ScreenOverlayRenderer overlay;
    for (int i = 0; i < 2; ++i) {
        const glm::vec2 pos(100 + i * 400, 50);
        const glm::vec3 camera(i == 0 ? 0 : 5, 0, i == 0 ? 5 : 0);
        const glm::vec3 forward = glm::normalize(-camera);
        const glm::mat4 view = glm::lookAt(camera, glm::vec3(0), glm::vec3(0, 1, 0));
        overlay.beginFrame();
        lfs::python::invoke_viewport_overlay(glm::value_ptr(view), glm::value_ptr(proj),
                                             glm::value_ptr(pos), glm::value_ptr(size),
                                             glm::value_ptr(camera), glm::value_ptr(forward),
                                             &overlay, &draw_list);
        overlay.endFrame();
        const auto commands = overlay.consumeCommands();
        EXPECT_EQ(commands.size(), 2u);
        for (const auto& command : commands) {
            EXPECT_NEAR(command.p0.x, pos.x + size.x / 2, 0.001f);
            EXPECT_NEAR(command.p0.y, pos.y + size.y / 2, 0.001f);
            EXPECT_TRUE(command.clip.has_value());
            if (command.clip) {
                EXPECT_EQ(command.clip->min, pos);
                EXPECT_EQ(command.clip->max, pos + size);
            }
        }
    }
    EXPECT_TRUE(run(R"PY(
assert len(instances) == 1, len(instances)
assert [c[0] for c in calls] == [1, 2], calls
for c, point, camera, forward in zip(calls, [(300, 150), (700, 150)],
                                    [(0, 0, 5), (5, 0, 0)], [(0, 0, -1), (-1, 0, 0)]):
    for got, want in zip(c[1:], [point, camera, forward, forward]):
        assert all(abs(a-b) < 0.001 for a, b in zip(got, want)), (got, want)
)PY"));
    EXPECT_TRUE(run("CustomOverlay.enabled = False"));
    const glm::mat4 view(1);
    const glm::vec2 pos(0);
    const glm::vec3 camera(0), forward(0, 0, -1);
    overlay.beginFrame();
    lfs::python::invoke_viewport_overlay(glm::value_ptr(view), glm::value_ptr(proj),
                                         glm::value_ptr(pos), glm::value_ptr(size),
                                         glm::value_ptr(camera), glm::value_ptr(forward),
                                         &overlay, &draw_list);
    EXPECT_TRUE(overlay.consumeCommands().empty());
    EXPECT_TRUE(run("lf.unregister_gizmo(CustomOverlay.gizmo_id)"));
    EXPECT_FALSE(lfs::python::has_viewport_draw_handlers());
    overlay.endFrame();
}
TEST_F(PythonIntegrationTest, RenderSettingsQueuedPropertiesSurviveFocusChangeWithoutLostUpdates) {
    using namespace std::chrono_literals;
    TestVisualizer viewer;
    viewer.queue_posted_work = true;
    const ScopedVisualizer scoped_viewer(&viewer);
    ViewSettingCallbacks callbacks;
    std::array<std::exception_ptr, 2> errors;
    auto run = [&](int index, const char* statement) {
        try {
            (void)runPythonTensorSnippet(std::string("import lichtfeld as lf\ns=lf.get_render_settings()\n") + statement +
                                         "\nresult_shape=(1,)\nresult_values=[1.0]");
        } catch (...) { errors[index] = std::current_exception(); }
    };
    std::thread first([&] { run(0, "s.focal_length_mm=73.0"); });
    const bool first_queued = viewer.waitForQueuedWork(5s);
    std::thread second([&] { run(1, "s.show_grid=False"); });
    const bool both_queued = viewer.waitForQueuedWork(5s, 2);
    callbacks.views.edit([&](auto& screen) { screen.setActiveView({callbacks.second}); });
    while (viewer.runNextQueuedWork()) {}
    first.join();
    second.join();
    ASSERT_TRUE(first_queued);
    ASSERT_TRUE(both_queued);
    for (const auto& error : errors)
        if (error)
            std::rethrow_exception(error);
    EXPECT_FLOAT_EQ(callbacks.manager.settingsForView(callbacks.first).focal_length_mm, 73.0f);
    EXPECT_FALSE(callbacks.manager.settingsForView(callbacks.first).show_grid);
    EXPECT_FLOAT_EQ(callbacks.manager.settingsForView(callbacks.second).focal_length_mm, 35.0f);
    EXPECT_TRUE(callbacks.manager.settingsForView(callbacks.second).show_grid);
    EXPECT_EQ(callbacks.views.activeView(), callbacks.second);
    EXPECT_EQ(callbacks.commits, 2);
}

TEST_F(PythonIntegrationTest, RenderSettingsDeletedViewAndReplacedScreenFailWithoutRedirecting) {
    using namespace std::chrono_literals;
    for (const bool replace_screen : {false, true}) {
        TestVisualizer viewer;
        viewer.queue_posted_work = true;
        const ScopedVisualizer scoped_viewer(&viewer);
        ViewSettingCallbacks callbacks;
        PythonTensorResult result;
        std::exception_ptr error;
        std::thread worker([&] {
            try {
                result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
s=lf.get_render_settings()
failed=False
try:
    s.focal_length_mm=73.0
except RuntimeError:
    failed=True
result_shape=(1,)
result_values=[float(failed)]
)PY");
            } catch (...) { error = std::current_exception(); }
        });
        const bool queued = viewer.waitForQueuedWork(5s);
        if (replace_screen)
            callbacks.views.resetToDefault();
        else
            callbacks.views.edit([&](auto& screen) { EXPECT_TRUE(screen.close({callbacks.first})); });
        while (viewer.runNextQueuedWork()) {}
        worker.join();
        ASSERT_TRUE(queued);
        if (error)
            std::rethrow_exception(error);
        EXPECT_EQ(result.values, std::vector<float>{1.0f});
        EXPECT_EQ(callbacks.commits, 0);
        EXPECT_FLOAT_EQ(callbacks.manager.getSettings().focal_length_mm, 35.0f);
    }
}

TEST_F(PythonIntegrationTest, RenderSettingsShutdownAndCancelledWorkReportFailure) {
    using namespace std::chrono_literals;
    TestVisualizer viewer;
    const ScopedVisualizer scoped_viewer(&viewer);
    ViewSettingCallbacks callbacks;
    const std::string script = R"PY(
import lichtfeld as lf
s=lf.get_render_settings()
failed=False
try:
    s.show_grid=False
except RuntimeError:
    failed=True
result_shape=(1,)
result_values=[float(failed)]
)PY";
    viewer.accepts_posted_work = false;
    EXPECT_EQ(runPythonTensorSnippet(script).values, std::vector<float>{1.0f});
    viewer.accepts_posted_work = true;
    viewer.queue_posted_work = true;
    PythonTensorResult result;
    std::exception_ptr error;
    std::thread worker([&] { try { result=runPythonTensorSnippet(script); } catch (...) { error=std::current_exception(); } });
    const bool queued = viewer.waitForQueuedWork(5s);
    EXPECT_TRUE(viewer.cancelNextQueuedWork());
    worker.join();
    ASSERT_TRUE(queued);
    if (error)
        std::rethrow_exception(error);
    EXPECT_EQ(result.values, std::vector<float>{1.0f});
    EXPECT_EQ(callbacks.commits, 0);
}

TEST_F(PythonIntegrationTest, RenderSettingsNormalizationSceneScopeAndNotificationsArePreserved) {
    TestVisualizer viewer;
    viewer.on_viewer_thread = true;
    const ScopedVisualizer scoped_viewer(&viewer);
    ViewSettingCallbacks callbacks;
    int notifications = 0;
    bool notified_with_gil = true;
    auto& registry = lfs::core::prop::PropertyRegistry::instance();
    // Import registers the render metadata before subscribing.
    (void)runPythonTensorSnippet("import lichtfeld as lf\nresult_shape=(1,)\nresult_values=[1.0]");
    const auto subscription = registry.subscribe("render_settings", "equirectangular", [&](const auto&, const auto&, const auto&, const auto&) {
        ++notifications;
        notified_with_gil = notified_with_gil && PyGILState_Check();
    });
    PythonTensorResult result;
    try {
        result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
s=lf.get_render_settings()
s.gt_comparison_actual_size=True
s.equirectangular=True
normalized=not s.gt_comparison_actual_size
s.equirectangular=True
s.color_exposure=1.7
invalid=False
try:
    s.focal_length_mm=object()
except (TypeError,RuntimeError):
    invalid=True
result_shape=(2,)
result_values=[float(normalized),float(invalid)]
)PY");
    } catch (...) {
        registry.unsubscribe(subscription);
        throw;
    }
    registry.unsubscribe(subscription);
    EXPECT_EQ(result.values, (std::vector<float>{1.0f, 1.0f}));
    EXPECT_EQ(notifications, 1);
    EXPECT_TRUE(notified_with_gil);
    EXPECT_TRUE(callbacks.manager.settingsForView(callbacks.first).equirectangular);
    EXPECT_FALSE(callbacks.manager.settingsForView(callbacks.second).equirectangular);
    EXPECT_FLOAT_EQ(callbacks.manager.settingsForView(callbacks.first).color_exposure, 1.7f);
    EXPECT_FLOAT_EQ(callbacks.manager.settingsForView(callbacks.second).color_exposure, 1.7f);
}

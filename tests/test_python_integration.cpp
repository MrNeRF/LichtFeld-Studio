/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python/python_compat.hpp"
#include <gtest/gtest.h>

#include <torch/torch.h>

#include "core/camera.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/control_boundary.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "io/loader.hpp"
#include "python/gil.hpp"
#include "python/python_buffer_analysis.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "training/control/command_api.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/visualizer.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <numbers>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
    std::filesystem::path findPythonModuleDir();
    void prependPythonPath(const std::filesystem::path& path);
} // namespace

class PythonIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto module_dir = findPythonModuleDir();
        ASSERT_FALSE(module_dir.empty()) << "Could not locate built lichtfeld module for Python tests";
        prependPythonPath(module_dir);
        lfs::python::ensure_initialized();
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
        std::expected<void, std::string> clearScene() override { return {}; }
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
        std::expected<std::filesystem::path, std::string> saveCheckpoint(
            const std::optional<std::filesystem::path>&) override {
            return std::unexpected("not implemented");
        }

        [[nodiscard]] bool waitForQueuedWork(const std::chrono::milliseconds timeout) {
            std::unique_lock lock(queued_work_mutex);
            return queued_work_cv.wait_for(lock, timeout, [this]() { return !queued_work.empty(); });
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

        bool on_viewer_thread = false;
        bool accepts_posted_work = true;
        bool queue_posted_work = false;
        int post_work_calls = 0;

    private:
        lfs::core::Scene scene_;
        std::mutex queued_work_mutex;
        std::condition_variable queued_work_cv;
        std::deque<WorkItem> queued_work;
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

        // Simulate raw bottom-left-origin readback: the logical bottom row is bright.
        for (size_t col = 0; col < width; ++col) {
            ptr[0 * height * width + col * height + 0] = 10.0f;
        }

        return std::make_shared<lfs::core::Tensor>(std::move(image));
    }

    bool containsLichtfeldModule(const std::filesystem::path& dir) {
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec)) {
            return false;
        }

        for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec) || file_ec) {
                continue;
            }

            const auto filename = it->path().filename().string();
            const auto ext = it->path().extension().string();
            if ((ext == ".so" || ext == ".pyd") && filename.rfind("lichtfeld", 0) == 0) {
                return true;
            }
        }

        return false;
    }

    std::filesystem::path findPythonModuleDir() {
        std::error_code ec;
        const auto cwd = std::filesystem::current_path(ec);
        const auto project_root = std::filesystem::path(PROJECT_ROOT_PATH);

        for (const auto& candidate : {
                 cwd / "src" / "python",
                 cwd.parent_path() / "src" / "python",
                 project_root / "build" / "src" / "python",
             }) {
            if (containsLichtfeldModule(candidate)) {
                return candidate;
            }
        }

        return {};
    }

    void prependPythonPath(const std::filesystem::path& path) {
        const auto value = path.string();
        const char* existing = std::getenv("PYTHONPATH");
#ifdef _WIN32
        const char separator = ';';
#else
        const char separator = ':';
#endif
        const std::string combined =
            existing && *existing ? value + separator + std::string(existing) : value;

#ifdef _WIN32
        _putenv_s("PYTHONPATH", combined.c_str());
#else
        setenv("PYTHONPATH", combined.c_str(), 1);
#endif
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
    EXPECT_NO_THROW(lfs::python::ensure_initialized());
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
    EXPECT_TRUE(document.structureCurrent());
    EXPECT_FALSE(document.foldRanges().empty());
    EXPECT_FALSE(document.highlights().empty());
    EXPECT_EQ(document.scopeAt(code.find("pass")), "Tool.run");

    const auto block = document.enclosingBlockRange(code.find("pass"));
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(code.substr(block->start_byte, block->end_byte - block->start_byte).find("def run"),
              std::string_view::npos);

    const auto ranges = document.enclosingBlockRanges(code.find("pass"));
    ASSERT_GE(ranges.size(), 2);
    EXPECT_NE(code.substr(ranges[0].start_byte, ranges[0].end_byte - ranges[0].start_byte).find("def run"),
              std::string_view::npos);
    EXPECT_NE(code.substr(ranges[1].start_byte, ranges[1].end_byte - ranges[1].start_byte).find("class Tool"),
              std::string_view::npos);
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
    ASSERT_TRUE(document.structureCurrent());
    ASSERT_FALSE(document.symbols().empty());
    ASSERT_FALSE(document.foldRanges().empty());

    ASSERT_TRUE(document.reset(invalid));
    EXPECT_EQ(document.analysis().status, lfs::python::PythonBufferStatus::SyntaxError);
    EXPECT_FALSE(document.structureCurrent());
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

TEST_F(PythonIntegrationTest, RenderViewAppliesVisualizerSceneTransforms) {
    lfs::core::Scene scene;
    scene.addSplat("single", makeSingleWhiteSplat(0.0f, 0.0f, 2.0f));
    const lfs::python::SceneContextGuard scene_guard(&scene);

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
rotation, translation = lf.look_at((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
img = lf.render_view(rotation, translation, 64, 64, fov=60.0)
result_shape = (1,)
result_values = [float(img.max().cpu().item())]
)PY");

    ASSERT_EQ(result.shape.size(), static_cast<size_t>(1));
    EXPECT_EQ(result.shape[0], 1);
    ASSERT_EQ(result.values.size(), static_cast<size_t>(1));
    EXPECT_GT(result.values[0], 0.01f);
}

TEST_F(PythonIntegrationTest, RenderViewMatchesViewportVerticalOrientation) {
    lfs::core::Scene scene;
    scene.addSplat("single", makeSingleWhiteSplat(0.0f, -0.75f, 2.0f));
    const lfs::python::SceneContextGuard scene_guard(&scene);

    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
rotation, translation = lf.look_at((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
img = lf.render_view(rotation, translation, 64, 64, fov=60.0).cpu().tolist()
top = sum(pixel[0] for row in img[:32] for pixel in row)
bottom = sum(pixel[0] for row in img[32:] for pixel in row)
result_shape = (2,)
result_values = [float(top), float(bottom)]
)PY");

    ASSERT_EQ(result.shape.size(), static_cast<size_t>(1));
    EXPECT_EQ(result.shape[0], 2);
    ASSERT_EQ(result.values.size(), static_cast<size_t>(2));
    EXPECT_GT(result.values[0], result.values[1]);
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
    options.resize_factor = 8;
    options.images_folder = "images_8";

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
                        dataset_dir.string() + R"PY(", resize_factor=8, images_folder="images_8")
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

// NOTE: Tests that actually execute Python scripts require the lichtfeld module
// to be importable, which depends on the CommandCenter and training infrastructure.
// These are better tested via integration tests (running training with --python-script).

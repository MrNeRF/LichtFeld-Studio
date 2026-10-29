/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// harness adapter: the merged support entry retains the same fixture interfaces.
#define LFS_USER_SHADER_CUSTOM_FIXTURE
#define LFS_USER_SHADER_PROGRAM_FIXTURE
#define LFS_USER_SHADER_SERVICE_FIXTURE
#include "rendering/rasterizer/vulkan/src/perf_timer.h"
#include "user_shader_measure.hpp"
#include "user_shader_test_support.hpp"

using namespace lfs::test::measure;
static std::unique_ptr<Options> options;
template <class Fixture>
class BenchmarkFixture : public Fixture {
protected:
    void TestBody() override = 0;

public:
    void run() {
        this->SetUpTestSuite();
        this->SetUp();
        if (!this->HasFatalFailure() && !this->IsSkipped())
            this->TestBody();
        this->TearDown();
        this->TearDownTestSuite();
    }
};

namespace lfs::test::program_set {
    class Benchmark : public BenchmarkFixture<ProgramSetTest> {
        // harness adapter: one compile, pipeline worker completion, then viewer publication.
        void measure(const char* label, const char* tint, const std::vector<std::string>& hosts,
                     lfs::vis::shaders::SlangCompiler& compiler,
                     const std::shared_ptr<const lfs::vis::shaders::ShaderPackage>& package) {
            const auto source = std::format("import lfs;\nLfsElement effect(LfsElement e, LfsContext ctx) "
                                            "{{ e.color = float3({}); return e; }}",
                                            tint);
            const auto start = Clock::now();
            auto sources = compileHosts<TestRenderer::ComputeProgramSource>(compiler, package, source, hosts);
            ASSERT_FALSE(sources.empty());
            const auto compiled = Clock::now();
            ASSERT_TRUE(renderer_->requestProgramSet(std::move(sources)));
            renderer_->waitForProgramSetBuilds();
            const auto built = Clock::now();
            ASSERT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
            const auto published = Clock::now();
            if (options->matches(std::string("edit_latency/") + label))
                emit({{"metric", "edit_latency"}, {"case", label}, {"link_ms", ms(compiled - start)}, {"pipeline_ms", ms(built - compiled)}, {"publish_ms", ms(published - built)}});
        }
        void TestBody() override {
            initializeShipped();
            ASSERT_FALSE(HasFailure());
            const auto package = loadPackage(LFS_SHADER_PACKAGE_DIR);
            ASSERT_TRUE(package);
            lfs::vis::shaders::SlangCompiler compiler;
            measure("cold_first_family", "0.25", {"projection_forward_custom"}, compiler, package);
            measure("changed_source", "0.5", {"projection_forward_custom"}, compiler, package);
            measure("3dgut", "0.25", {"projection_forward_3dgut_custom"}, compiler, package);
            measure("all_hosts", "0.75", {}, compiler, package);
        }
    };
} // namespace lfs::test::program_set

namespace lfs::test::custom_effect {
    class Benchmark : public BenchmarkFixture<CustomEffectTest> {
        // harness adapter: app loader plus its Vulkan external allocation contract.
        std::shared_ptr<SplatData> load(const char* path) {
            lfs::io::LoadOptions load_options;
            load_options.splat_tensor_allocator = [&](auto shape, auto capacity, auto dtype, auto label) {
                auto t = lfs::vis::makeVulkanExternalTensor(*context_, std::move(shape), dtype, capacity,
                                                            std::string(label).c_str());
                if (!t)
                    throw std::runtime_error("benchmark splat allocation failed");
                return std::move(*t);
            };
            auto result = lfs::io::Loader::create()->load(path, load_options);
            if (!result || !std::holds_alternative<std::shared_ptr<SplatData>>(result->data))
                throw std::runtime_error("benchmark PLY load failed");
            return std::get<std::shared_ptr<SplatData>>(std::move(result->data));
        }
        // harness adapter: submit the same projection paths as dispatchCases; no image readback.
        void submit(const SplatData& model, const ViewportRenderRequest& request, Path path) {
            const auto result = viewport_->render(*context_, model, request, true, Slot::Main, path == Path::Legacy);
            ASSERT_TRUE(result) << result.error();
            ASSERT_FALSE(result->custom_pending);
            ASSERT_EQ(Access::customProjection(*viewport_), request.custom_effect.has_value());
        }
        // harness adapter: retain dynamic host discovery; do not assume a fixed host count.
        void publishEffect(const char* source) {
            const auto& compiled = programs(source);
            ASSERT_FALSE(compiled.empty());
            Access::publish(*viewport_, compiled);
        }
        void scene(const char* label, const char* path) {
            const auto a = generate(256, 128);
            auto splats = path ? load(path) : std::make_shared<SplatData>(model(a));
            ASSERT_GT(splats->size(), 0);
            transforms_ = path ? std::vector<glm::mat4>{glm::mat4(1)} : nodeTransforms();
            transform_indices_ = path ? std::make_shared<Tensor>(Tensor::zeros({static_cast<std::size_t>(splats->size())}, Device::CUDA, DataType::Int32)) : indices(a);
            const auto means = splats->means().to(Device::CPU).contiguous();
            glm::vec3 lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
            for (std::size_t i = 0; i < static_cast<std::size_t>(splats->size()); ++i) {
                const auto* p = means.ptr<float>() + 3 * i;
                const auto v = glm::vec3(transforms_[path ? 0 : a.nodes[i]] * glm::vec4(p[0], p[1], p[2], 1));
                if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
                    continue;
                lo = glm::min(lo, v);
                hi = glm::max(hi, v);
            }
            ASSERT_LE(lo.x, hi.x);
            const auto center = (lo + hi) * 0.5f;
            const float radius = std::max(glm::length(hi - lo) * 0.5f, 0.01f);
            struct Timings {
                std::array<std::pair<std::size_t, double>, PerfTimer::END> stages{};
                bool active = false;
            };
            auto times = std::make_shared<Timings>();
            auto& stages = times->stages;
            auto& timing = times->active;
            auto initial = request(Path::Legacy);
            initial.frame_view.size = {1280, 720};
            initial.sh_degree = splats->get_active_sh_degree();
            submit(*splats, initial, Path::Legacy);
            initialized_ = true;
            auto& renderer = Access::renderer(*viewport_);
            renderer.waitForPendingBatch();
            renderer.addTimerCallback([times](const auto& updates) {
                for (std::size_t i = 0; times->active && i < updates.size(); ++i) {
                    times->stages[i].first += updates[i].first;
                    times->stages[i].second += updates[i].second;
                }
            });
            for (const auto path_mode : {Path::Survivors, Path::Legacy, Path::Gut}) {
                // Rotate order across repetitions; every repetition includes every state.
                for (int order = 0; order < 4; ++order) {
                    const int state = (order + options->sample) % 4;
                    const char* state_name = std::array{"default", "unused", "identity", "moving"}[state];
                    if (!options->matches(std::string("gpu_frame/") + label + "/" + name(path_mode) + "/" + state_name))
                        continue;
                    Access::release(*viewport_);
                    if (state != 0)
                        publishEffect(state == 3 ? kEdit : kIdentity);
                    auto r = request(path_mode);
                    r.frame_view.size = {1280, 720};
                    r.sh_degree = splats->get_active_sh_degree();
                    r.deterministic_export = false;
                    const float distance = 1.2f * radius / std::sin(lfs::rendering::focalLengthToVFovRad(r.frame_view.focal_length_mm) * 0.5f);
                    r.frame_view.near_plane = std::max(radius * 0.001f, 1e-5f);
                    r.frame_view.far_plane = distance + 2.0f * radius;
                    r.frame_view.translation = center + glm::vec3(0, 0, distance);
                    r.frame_view.rotation = lfs::rendering::makeVisualizerLookAtRotation(r.frame_view.translation, center);
                    if (state >= 2) {
                        r.custom_effect = global();
                        r.custom_effect->objects.resize(transforms_.size());
                        r.custom_effect->bounds_min = lo;
                        r.custom_effect->bounds_max = hi;
                        if (state == 3)
                            r.custom_effect->bindings[0].values[Offset].x = 0.01f * radius;
                    }
                    // One frame in flight, as the app paces frames; pass timestamps are unaffected.
                    for (int f = 0; f < 16; ++f) {
                        submit(*splats, r, path_mode);
                        renderer.waitForPendingBatch();
                    }
                    stages = {};
                    timing = true;
                    for (int f = 0; f < 64; ++f) {
                        submit(*splats, r, path_mode);
                        renderer.waitForPendingBatch();
                    }
                    timing = false;
                    const auto projection = path_mode == Path::Survivors ? PerfTimer::ProjectionSurvivors : PerfTimer::ProjectionForward;
                    ASSERT_GE(stages[projection].first, 64u);
                    ASSERT_GE(stages[PerfTimer::RasterizeForward].first, 64u);
                    if (path_mode == Path::Survivors)
                        ASSERT_GE(stages[PerfTimer::CullSplats].first, 64u);
                    const double cull = 1e6 * stages[PerfTimer::CullSplats].second / 64;
                    const double proj = 1e6 * stages[projection].second / 64;
                    const double raster = 1e6 * stages[PerfTimer::RasterizeForward].second / 64;
                    emit({{"metric", "gpu_frame"}, {"scene", label}, {"scene_source", path ? "environment" : "fixture"}, {"mode", name(path_mode)}, {"state", state_name}, {"cull_us", cull}, {"proj_us", proj}, {"raster_us", raster}, {"total_us", cull + proj + raster}});
                }
            }
            viewport_.reset(); // callbacks die before their captured state
            viewport_ = std::make_unique<VksplatViewportRenderer>();
            initialized_ = false;
        }
        void TestBody() override {
            for (const auto& [label, variable] : std::array<std::pair<const char*, const char*>, 3>{{{"fixture", ""}, {"large", "LFS_BENCH_PLY_LARGE"}, {"small", "LFS_BENCH_PLY_SMALL"}}}) {
                bool selected = false;
                for (const char* mode : {"survivors", "legacy", "3dgut"})
                    for (const char* state : {"default", "unused", "identity", "moving"})
                        selected |= options->matches(std::string("gpu_frame/") + label + "/" + mode + "/" + state);
                if (!selected)
                    continue;
                const char* path = std::getenv(variable);
                scene(label, path && *path ? path : nullptr);
                if (HasFailure())
                    return;
            }
        }
    };
} // namespace lfs::test::custom_effect

namespace lfs::test::service_rendering {
    class Benchmark : public BenchmarkFixture<ShaderServiceRenderingTest> {
        std::string metric_;

    public:
        explicit Benchmark(std::string metric) : metric_(std::move(metric)) {}

    private:
        // harness adapter: capture, state/request copy, and the renderer's actual overlay packing.
        std::size_t cpuDraw(std::size_t nodes) {
            const auto snapshot = ServiceAccess::snapshot(service());
            lfs::vis::SceneRenderState state;
            state.custom_effect = shaders::makeCustomEffectState(*snapshot, shaders::captureEffectScene(scene(), nodes), 0.5f);
            lfs::rendering::ViewportRenderRequest request;
            request.custom_effect = state.custom_effect;
            auto packed = lfs::vis::vksplat_overlay::buildParams(request, false, false, true, nodes, false);
            auto tail = lfs::vis::vksplat_overlay::buildCustomEffectTail(request, VulkanGSRendererUniforms{});
            if (!tail)
                throw std::runtime_error("custom parameter packing failed");
            packed.insert(packed.end(), tail->begin(), tail->end());
            return packed.size();
        }
        void cpu() {
            manager_->setShaderService(nullptr);
            service_.reset();
            service_ = std::make_unique<ShaderService>(scene(), [&] { return std::make_unique<ScheduledLinker>(schedule_); }, [] {});
            for (const int count : {1, 64, 512}) {
                if (!options->matches("cpu_per_draw/" + std::to_string(count)))
                    continue;
                ServiceAccess::clear(service());
                scene().clear();
                nodes_.clear();
                const auto a = generate(64, 32, 0.0f); // exactly 2048 splats per node
                const std::size_t n = a.size();
                for (int i = 0; i < count; ++i) {
                    const auto external = [&](const auto& data, lfs::core::TensorShape shape) {
                        return externalTensor(*context_, Tensor::from_vector(data, shape, Device::CPU), n, "measure-node");
                    };
                    auto model = std::make_unique<SplatData>(0, external(a.means, {n, 3}), external(a.sh0, {n, 1, 3}),
                                                             Tensor::zeros({n, 0, 3}, Device::CUDA), external(a.scales, {n, 3}),
                                                             external(a.rotations, {n, 4}), external(a.opacities, {n, 1}), 1.0f);
                    Uuid id;
                    scene().addSplat(std::format("node{}", i), std::move(model), lfs::core::NULL_NODE, &id);
                    nodes_.push_back(id);
                }
                const auto global_asset = ServiceAccess::create(service(), "Global effect", kShift);
                ASSERT_TRUE(global_asset);
                ASSERT_TRUE(ServiceAccess::bind(service(), {}, *global_asset));
                const auto local = ServiceAccess::create(service(), "Node effect", kShift);
                ASSERT_TRUE(local);
                for (const auto& id : nodes_)
                    ASSERT_TRUE(ServiceAccess::bind(service(), {Kind::Node, id}, *local));
                // CPU metric: real scene/service/packing, fake link and publication avoid any GPU draw of 512 nodes.
                ServiceAccess::demand(service(), "projection_forward_custom");
                const auto deadline = Clock::now() + 5s;
                while (ServiceAccess::status(service(), *global_asset)->links == 0 && Clock::now() < deadline)
                    std::this_thread::sleep_for(1ms);
                ASSERT_GT(ServiceAccess::status(service(), *global_asset)->links, 0u);
                ASSERT_TRUE(ServiceAccess::adoptReady(service()));
                ASSERT_EQ(ServiceAccess::snapshot(service())->bindings.size(), static_cast<std::size_t>(count + 1));
                volatile std::size_t sink = cpuDraw(count);
                std::vector<double> samples;
                for (int i = 0; i < 500; ++i) {
                    const auto start = Clock::now();
                    sink = cpuDraw(count);
                    samples.push_back(ms(Clock::now() - start) * 1000);
                }
                (void)sink;
                emit({{"metric", "cpu_per_draw"}, {"nodes", count}, {"us", median(samples)}});
            }
        }
        // harness adapter: real video snapshot, closed export preflight and draw callback, with completed pixel readback.
        void exportFrames() {
            viewport_.frameBufferSize = {1280, 720};
            ASSERT_TRUE(settle([&] { return redraw().rendered; }));
            const auto asset = ServiceAccess::create(service(), "Export effect", kIdentity);
            ASSERT_TRUE(asset);
            ASSERT_TRUE(ServiceAccess::bind(service(), {}, *asset));
            ASSERT_EQ(ServiceAccess::status(service(), *asset)->links, 0u);
            const auto start = Clock::now();
            auto snapshot = lfs::vis::gui::captureVideoExportSceneSnapshot(*scene_);
            ASSERT_TRUE(snapshot);
            ASSERT_TRUE(lfs::io::migrateSplatTensorsToAllocator(*snapshot->combined_model, manager_->makeSplatTensorAllocator()));
            auto pin = ServiceAccess::pin(*manager_, *snapshot->gaussianModel(), settings_, {1280, 720});
            OffViewer viewer(*this, [&] { redraw(); });
            int attempts = 0, first_attempts = 0;
            std::vector<int> attempts_per_frame;
            double first_ms = 0, probe_ms = 0;
            bool live = false;
            std::vector<float> baseline;
            const bool complete = viewer.run([&] {
                for (int i = 0; i < 30; ++i) {
                    const int previous_attempts = attempts;
                    auto pixels = ServiceAccess::draw(*manager_, pin, viewer.post(), [&]() -> Image {
                        ++attempts;
                        const auto shaders = ServiceAccess::frozen(pin);
                        auto state = lfs::vis::gui::makeVideoExportGaussianSceneState(*snapshot, {}, shaders.get(), i / 30.0f);
                        return rgbOf(manager_->renderPreviewImage(*snapshot->gaussianModel(), std::move(state),
                                                                  viewport_.getRotationMatrix(), viewport_.getTranslation(), settings_.focal_length_mm, 1280, 720));
                    });
                    if (!pixels || pixels->empty())
                        return false;
                    attempts_per_frame.push_back(attempts - previous_attempts);
                    if (i == 0) {
                        first_ms = ms(Clock::now() - start);
                        first_attempts = attempts;
                    }
                    if (i == 1)
                        viewer.onViewer([&] {
                            baseline = drawn();
                            EXPECT_TRUE(ServiceAccess::edit(service(), *asset, kTinyEffect));
                        });
                    if (i == 15) {
                        // Fixed opportunity window, excluded from throughput time, while frames remain to export.
                        const auto probe = Clock::now();
                        while (Clock::now() - probe < 5s) {
                            viewer.onViewer([&] {
                                const auto current = drawn();
                                live |= !baseline.empty() && current.size() == baseline.size() &&
                                        ServiceAccess::status(service(), *asset)->displayed_revision == 2 &&
                                        compareImages(current, baseline).max > 0.05f;
                            });
                            std::this_thread::sleep_for(10ms);
                        }
                        probe_ms = ms(Clock::now() - probe);
                    }
                }
                return true;
            });
            ASSERT_TRUE(complete);
            const double wall = ms(Clock::now() - start);
            emit({{"metric", "export"}, {"first_frame_ms", first_ms}, {"total_ms", wall}, {"total_excluding_probe_ms", wall - probe_ms}, {"viewport_probe_ms", probe_ms}, {"draw_attempts_first", first_attempts}, {"draw_attempts_total", attempts}, {"draw_attempts_per_frame", attempts_per_frame}, {"viewport_live_during_export", live}});
        }
        void TestBody() override {
            if (metric_ == "idle_sync") {
                ASSERT_FALSE(ServiceAccess::snapshot(service()));
                emit({{"metric", "idle_sync"}, {"locks_per_iteration", nullptr}, {"ns_per_iteration", idleSync(*manager_)}, {"method", "syncCustomPrograms_100k_fallback"}});
            }
            if (metric_ == "cpu_per_draw")
                cpu();
            if (metric_ == "export")
                exportFrames();
        }
    };
} // namespace lfs::test::service_rendering

// harness adapter: identical workload on direct-setter BEFORE and command-only AFTER.
static void measureSetParameter() {
    using namespace lfs::vis::shaders;
    for (double mib : {0.01, 1.0, 16.0}) {
        lfs::core::Scene scene;
        lfs::test::Schedule schedule;
        ShaderService service(scene, [&] { return std::make_unique<lfs::test::ScheduledLinker>(schedule); }, [] {});
        std::size_t remaining = static_cast<std::size_t>(mib * 1024 * 1024);
        lfs::core::Uuid bound;
        while (remaining) {
            const auto bytes = std::min(remaining, kMaxShaderSourceBytes);
            auto id = ServiceAccess::create(service, "Parameter measurement", std::string(bytes, ' '));
            ASSERT_TRUE(id);
            if (bound.is_nil())
                bound = *id;
            remaining -= bytes;
        }
        ASSERT_TRUE(ServiceAccess::bind(service, {}, bound));
        ServiceAccess::demand(service, "projection_forward_custom");
        const auto deadline = Clock::now() + std::chrono::seconds(10);
        while (!ServiceAccess::status(service, bound)->links && Clock::now() < deadline)
            std::this_thread::yield();
        ASSERT_GT(ServiceAccess::status(service, bound)->links, 0u);
        ASSERT_TRUE(ServiceAccess::adoptReady(service));
        const auto parameter = [&]<class Service>(Service& target, float value) {
            if constexpr (requires { target.setParameter(bound, "gain", std::span(&value, 1)); })
                return bool(target.setParameter(bound, "gain", std::span(&value, 1)));
            else
                return bool(target.command(SetShaderParameter{bound, "gain", {value}, {}}));
        };
        std::vector<double> samples;
        for (int i = 0; i < 500; ++i) {
            const auto start = Clock::now();
            const bool changed = parameter(service, (i % 2) ? 0.5f : 1.5f);
            samples.push_back(ms(Clock::now() - start) * 1000);
            ASSERT_TRUE(changed);
        }
        emit({{"metric", "set_parameter"}, {"case", std::to_string(mib)}, {"library_mib", mib}, {"us", median(samples)}});
    }
}

int main(int argc, char** argv) {
    try {
        options = std::make_unique<Options>(argc, argv);
        if (options->sample < 0)
            return repetitions(*options);
        ::testing::InitGoogleTest(&argc, argv);
        if (options->matches("set_parameter"))
            measureSetParameter();
        // Edit latency goes first, before the driver or compiler has seen custom programs.
        bool edit = false, gpu = false;
        for (const char* label : {"cold_first_family", "changed_source", "3dgut", "all_hosts"})
            edit |= options->matches(std::string("edit_latency/") + label);
        for (const char* scene : {"fixture", "large", "small"})
            for (const char* mode : {"survivors", "legacy", "3dgut"})
                for (const char* state : {"default", "unused", "identity", "moving"})
                    gpu |= options->matches(std::string("gpu_frame/") + scene + "/" + mode + "/" + state);
        if (edit)
            lfs::test::program_set::Benchmark{}.run();
        if (gpu)
            lfs::test::custom_effect::Benchmark{}.run();
        for (const char* metric : {"cpu_per_draw", "idle_sync", "export"}) {
            bool selected = options->matches(metric);
            if (std::string(metric) == "cpu_per_draw")
                for (int n : {1, 64, 512})
                    selected |= options->matches(std::string(metric) + "/" + std::to_string(n));
            if (selected)
                lfs::test::service_rendering::Benchmark{metric}.run();
        }
        return ::testing::Test::HasFailure() || ::testing::Test::IsSkipped() ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "../../src/training/hip/session.hpp"
#include "../../src/training/hip/training_api.h"
#include "photometric_oracle.hpp"
#include "training_fixture.hpp"
#include "training_session_fixture.hpp"
#include "training_sh_session.hpp"
#ifdef LFS_HIP_SESSION_WITH_CUDA
#include <cuda_runtime_api.h>
#endif
#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
    using namespace hip_fixture;
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
#ifdef LFS_HIP_SESSION_WITH_CUDA
    struct CudaGuard {
        void* pointer = nullptr;
        std::vector<unsigned char> pattern = std::vector<unsigned char>(1024 * 1024);
        static void check_cuda(cudaError_t status) {
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
        }
        CudaGuard() {
            check_cuda(cudaSetDevice(0));
            for (size_t i = 0; i < pattern.size(); ++i)
                pattern[i] = static_cast<unsigned char>((i * 37 + 11) % 251);
            check_cuda(cudaMalloc(&pointer, pattern.size()));
            check_cuda(cudaMemcpy(pointer, pattern.data(), pattern.size(), cudaMemcpyHostToDevice));
        }
        ~CudaGuard() {
            if (pointer)
                (void)cudaFree(pointer);
        }
        void verify() {
            std::vector<unsigned char> result(pattern.size());
            check_cuda(cudaMemcpy(result.data(), pointer, result.size(), cudaMemcpyDeviceToHost));
            require(result == pattern, "HIP session changed the live NVIDIA CUDA allocation");
            check_cuda(cudaMemset(pointer, 0x5a, pattern.size()));
            check_cuda(cudaMemcpy(result.data(), pointer, result.size(), cudaMemcpyDeviceToHost));
            for (auto value : result)
                require(value == 0x5a, "CUDA failed after HIP training");
            std::puts("PASS: live NVIDIA CUDA allocation unchanged and writable after HIP training");
        }
    };
#endif
    template <class F>
    F symbol(HMODULE module, const char* name) {
        auto address = GetProcAddress(module, name);
        require(address, name);
        return reinterpret_cast<F>(address);
    }
    struct Api {
        HMODULE module;
#define MEMBER(name) decltype(&::name) name = symbol<decltype(&::name)>(module, #name)
        MEMBER(lfs_hip_train_abi);
        MEMBER(lfs_hip_train_create);
        MEMBER(lfs_hip_train_destroy);
        MEMBER(lfs_hip_train_set_frame);
        MEMBER(lfs_hip_train_step);
        MEMBER(lfs_hip_train_render);
        MEMBER(lfs_hip_train_read_parameter);
        MEMBER(lfs_hip_train_info);
        MEMBER(lfs_hip_train_set_budget);
        MEMBER(lfs_hip_train_set_cancel);
#undef MEMBER
        explicit Api(const char* path) : module(LoadLibraryExW(std::filesystem::absolute(path).make_preferred().c_str(), nullptr,
                                                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) { require(module, "Cannot load HIP training module"); }
        ~Api() {
            if (module)
                FreeLibrary(module);
        }
        void close() {
            require(FreeLibrary(module), "Cannot unload HIP training module");
            module = nullptr;
        }
    };
    struct Session {
        Api& api;
        LfsHipTraining* handle = nullptr;
        explicit Session(Api& api) : api(api) {}
        ~Session() {
            if (handle) {
                LfsHipError error{};
                if (api.lfs_hip_train_destroy(handle, &error))
                    std::fprintf(stderr, "Session teardown failed: %s\n", error.message);
            }
        }
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        void close() {
            LfsHipError error{};
            const int status = api.lfs_hip_train_destroy(handle, &error);
            require(status == LFS_HIP_TRAIN_OK, error.message);
            handle = nullptr;
        }
    };
    void check(int status, const LfsHipError& error) { require(status == LFS_HIP_TRAIN_OK, error.message); }
    void rejected(int status, int expected, const LfsHipError& error) {
        if (status != expected)
            std::fprintf(stderr, "Expected status %d, got %d: %s\n", expected, status, error.message);
        require(status == expected && error.message[0], "Wrong failure status or missing diagnostic");
    }
    std::vector<float> read(Api& api, LfsHipTraining* session, unsigned family, size_t count) {
        std::vector<float> value(count);
        LfsHipError error{};
        check(api.lfs_hip_train_read_parameter(session, family, value.data(), value.size(), &error), error);
        return value;
    }
    double mse(const std::vector<float>& image, const std::vector<float>& target) {
        double loss = 0;
        for (size_t i = 0; i < image.size(); ++i) {
            require(std::isfinite(image[i]), "Nonfinite session render");
            const double difference = double(image[i]) - target[i];
            loss += difference * difference;
        }
        return loss / image.size();
    }
    void save(const char* name, const std::vector<float>& image) {
        std::ofstream file(name, std::ios::binary);
        file << "P6\n"
             << width << ' ' << height << "\n255\n";
        for (unsigned i = 0; i < pixels; ++i)
            for (unsigned c = 0; c < 3; ++c) {
                unsigned char value = static_cast<unsigned char>(std::clamp(image[c * pixels + i], 0.0f, 1.0f) * 255 + 0.5f);
                file.write(reinterpret_cast<const char*>(&value), 1);
            }
        require(bool(file), "Cannot save session render");
    }

    void large_model(Api& api) {
        constexpr size_t count = 1000000;
        Fixture fixture;
        std::array<std::vector<float>, 5> values{std::vector<float>(count * 3), std::vector<float>(count * 4),
                                                 std::vector<float>(count * 3, -1.5f), std::vector<float>(count * 3), std::vector<float>(count, -4)};
        for (size_t i = 0; i < count; ++i) {
            values[0][i * 3 + 2] = 2000;
            values[1][i * 4] = 1;
        }
        for (unsigned family = 0; family < 5; ++family)
            std::copy(fixture.values[family].begin(), fixture.values[family].end(), values[family].begin());
        LfsHipTrainModel model{values[0].data(), values[1].data(), values[2].data(), values[3].data(), values[4].data()};
        auto frame = fixture.frame();
        LfsHipTrainConfig config{sizeof(config), 0, count, 128 * 1024 * 1024};
        LfsHipError error{};
        Session session(api);
        rejected(api.lfs_hip_train_create(&config, &model, &session.handle, &error), LFS_HIP_TRAIN_RESOURCE_EXHAUSTED, error);
        require(!session.handle, "Large failed allocation published a session");
        config.allocation_budget_bytes = 512 * 1024 * 1024;
        check(api.lfs_hip_train_create(&config, &model, &session.handle, &error), error);
        check(api.lfs_hip_train_set_frame(session.handle, &frame, &error), error);
        const LfsHipTrainRates rates{0.002f, 0.002f, 0.003f, 0.05f, 0.01f};
        check(api.lfs_hip_train_step(session.handle, &rates, &error), error);
        LfsHipTrainInfo info{};
        check(api.lfs_hip_train_info(session.handle, &info, &error), error);
        require(info.completed_steps == 1 && !info.poisoned && info.allocated_bytes < 384 * 1024 * 1024 &&
                    info.peak_allocated_bytes <= config.allocation_budget_bytes && info.intersection_capacity < 10000,
                "Large model allocation or intersection capacity mismatch");
        const auto means = read(api, session.handle, 0, count * 3);
        require(means[count * 3 - 3] == 0 && means[count * 3 - 2] == 0 && means[count * 3 - 1] == 2000,
                "An invisible Gaussian changed in the large model");
        std::printf("PASS 1M model allocation/step: live %llu bytes, peak %llu, intersections %llu, capacity %llu\n",
                    static_cast<unsigned long long>(info.allocated_bytes), static_cast<unsigned long long>(info.peak_allocated_bytes),
                    static_cast<unsigned long long>(info.intersection_count), static_cast<unsigned long long>(info.intersection_capacity));
        std::puts("Memory/scaling test: only 263 Gaussians visible, 35x29 pixels; not a 1M-visible training benchmark.");
        session.close();
    }
    void host_adapter(const char* path) {
        using lfs::training::hip::TrainingError;
        using HostSession = lfs::training::hip::Session;
        Fixture fixture;
        fixture.perturb();
        const auto absolute = std::filesystem::absolute(path);
        const LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
        const LfsHipTrainModel model{fixture.values[0].data(), fixture.values[1].data(), fixture.values[2].data(),
                                     fixture.values[3].data(), fixture.values[4].data()};
        bool relative_rejected = false;
        try {
            HostSession invalid(absolute.filename(), config, model);
        } catch (const std::invalid_argument&) {
            relative_rejected = true;
        }
        require(relative_rejected, "Host adapter accepted a module through the working directory");
        HostSession session(absolute, config, model);
        session.set_frame(fixture.frame());
        std::vector<float> image(channels);
        session.render(image);
        const double initial = mse(image, fixture.target);
        const auto live = session.info().allocated_bytes;
        const LfsHipTrainRates rates{0.002f, 0.002f, 0.003f, 0.015f / sh_c0, 0.01f};
        session.set_budget(live);
        auto larger = fixture.frame();
        larger.width *= 2;
        larger.height *= 2;
        std::vector<float> larger_target(size_t(larger.width) * larger.height * 3);
        larger.target_chw = larger_target.data();
        bool budget_rejected = false;
        try {
            session.set_frame(larger);
        } catch (const TrainingError& error) {
            budget_rejected = error.status == LFS_HIP_TRAIN_RESOURCE_EXHAUSTED && error.what()[0];
        }
        require(budget_rejected && session.info().completed_steps == 0, "Host adapter lost a recoverable budget error");
        session.set_budget(config.allocation_budget_bytes);
        session.set_cancel(true);
        bool cancelled = false;
        try {
            session.step(rates);
        } catch (const TrainingError& error) {
            cancelled = error.status == LFS_HIP_TRAIN_CANCELLED;
        }
        require(cancelled && session.info().completed_steps == 0, "Host cancellation changed the optimizer");
        session.set_cancel(false);
        std::exception_ptr failure;
        std::thread worker([&] {
            try {
                for (unsigned i = 0; i < 100; ++i)
                    session.step(rates);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        for (unsigned i = 0; i < 30; ++i)
            (void)session.info();
        worker.join();
        if (failure)
            std::rethrow_exception(failure);
        session.render(image);
        require(session.info().completed_steps == 100 && mse(image, fixture.target) < initial * 0.1,
                "Host adapter training did not converge");
        std::vector<float> means(n * 3);
        session.read_parameter(0, means);
        require(means != fixture.values[0], "Host adapter failed to read updated model");
        session.close();
        session.close();
        bool closed_rejected = false;
        try {
            session.set_cancel(true);
        } catch (const std::logic_error&) {
            closed_rejected = true;
        }
        require(closed_rejected, "Host adapter called a destroyed session");
        // Independent session remains valid after another session is destroyed.
        HostSession survivor(absolute, config, model);
        survivor.set_frame(fixture.frame());
        survivor.step(rates);
        require(survivor.info().completed_steps == 1, "Host adapter module lifetime failed");
        std::printf("PASS host adapter: recovery, cancellation, serialized access, 100 steps MSE %.9g -> %.9g, teardown\n",
                    initial, mse(image, fixture.target));
    }
    void photometric_session(const char* path) {
        using HostSession = lfs::training::hip::Session;
        using lfs::training::hip::TrainingError;
        const auto absolute = std::filesystem::absolute(path);
        Fixture fixture;
        fixture.perturb();
        const LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
        HostSession session(absolute, config, fixture.model());
        const auto runtime = GetModuleHandleA("amdhip64_7.dll");
        int version = 0;
        require(symbol<int (*)(int*)>(runtime, "hipRuntimeGetVersion")(&version) == 0, "Cannot read HIP runtime version");
        std::printf("Photometric session runtime %d\n", version);
        session.set_frame(fixture.frame());
        std::vector<float> image(channels);
        session.render(image);
        require(std::abs(session.loss() - mse(image, fixture.target)) < 1e-7, "Default MSE loss changed");
        LfsHipTrainLoss loss{sizeof(loss), LFS_HIP_LOSS_L1_SSIM, 0.2f, 1};
        for (unsigned bad = 0; bad < 7; ++bad) {
            auto invalid = loss;
            switch (bad) {
            case 0: invalid.kind = 2; break;
            case 1: invalid.ssim_weight = -0.1f; break;
            case 2: invalid.ssim_weight = std::numeric_limits<float>::quiet_NaN(); break;
            case 3: invalid.ssim_weight = 1.1f; break;
            case 4: invalid.valid_padding = 2; break;
            case 5: invalid.struct_bytes = 0; break;
            case 6: invalid.kind = LFS_HIP_LOSS_MSE; break;
            }
            bool rejected = false;
            try {
                session.set_loss(invalid);
            } catch (const TrainingError& error) {
                rejected = error.status == LFS_HIP_TRAIN_INVALID;
            }
            require(rejected, "Invalid HIP photometric configuration accepted");
        }
        const auto base_live = session.info().allocated_bytes;
        session.set_loss(loss);
        const LfsHipTrainRates rates{0.002f, 0.002f, 0.003f, 0.015f / sh_c0, 0.01f};
        std::vector<float> means(n * 3);
        for (uint64_t budget : {base_live, base_live + pixels * 18}) {
            session.set_budget(budget);
            bool exhausted = false;
            try {
                session.step(rates);
            } catch (const TrainingError& error) {
                exhausted = error.status == LFS_HIP_TRAIN_RESOURCE_EXHAUSTED;
            }
            session.read_parameter(0, means);
            require(exhausted && session.info().completed_steps == 0 && !session.info().poisoned &&
                        session.info().allocated_bytes == base_live && means == fixture.values[0],
                    "SSIM workspace budget failure changed model or leaked storage");
        }
        session.set_budget(config.allocation_budget_bytes);
        (void)session.loss();
        require(session.info().allocated_bytes == base_live + pixels * 22, "Unexpected SSIM workspace size");
        loss.ssim_weight = 0;
        session.set_loss(loss);
        require(session.info().allocated_bytes == base_live, "Pure L1 retained unnecessary SSIM workspace");

        const unsigned dimensions[][2]{{1, 1}, {9, 7}, {11, 11}, {17, 13}, {35, 29}, {65, 37}, {9, 17}, {17, 9}};
        unsigned cases = 0;
        for (const auto& shape : dimensions) {
            auto frame = fixture.frame();
            frame.width = shape[0];
            frame.height = shape[1];
            frame.cx = frame.width / 2.0f;
            frame.cy = frame.height / 2.0f;
            std::vector<float> target(size_t(frame.width) * frame.height * 3), rendered(target.size());
            for (size_t i = 0; i < target.size(); ++i)
                target[i] = 0.15f + 0.6f * float((i * 17 + 3) % 101) / 100;
            frame.target_chw = target.data();
            session.set_frame(frame);
            session.render(rendered);
            for (bool valid : {false, true})
                for (float weight : {0.f, 0.2f, 1.f}) {
                    loss = {sizeof(loss), LFS_HIP_LOSS_L1_SSIM, weight, uint32_t(valid)};
                    session.set_loss(loss);
                    const double actual = session.loss();
                    const double expected = photo_oracle::evaluate(frame.width, frame.height, rendered, target, weight, valid, false).loss;
                    require(std::abs(actual - expected) < 5e-5, "HIP session GPU loss reduction differs from CPU oracle");
                    require(session.info().completed_steps == 0, "Diagnostic loss advanced optimizer");
                    ++cases;
                }
        }
        session.read_parameter(0, means);
        require(means == fixture.values[0], "Photometric diagnostics changed parameters");
        session.close();
        std::printf("PASS %u session GPU loss values, workspace budget failure/recovery, resize and diagnostics\n", cases);

        {
            HostSession exact(absolute, config, fixture.model());
            auto frame = fixture.frame();
            exact.set_frame(frame);
            exact.render(image);
            frame.target_chw = image.data();
            exact.set_frame(frame);
            loss = {sizeof(loss), LFS_HIP_LOSS_L1_SSIM, 0, 1};
            exact.set_loss(loss);
            require(exact.loss() == 0, "Exact L1 match has nonzero loss");
            exact.step(rates);
            for (unsigned family = 0; family < 5; ++family) {
                std::vector<float> actual(fixture.values[family].size());
                exact.read_parameter(family, actual);
                require(actual == fixture.values[family], "Pure L1 zero gradient changed model");
            }
            exact.close();
            std::puts("PASS pure L1 exact-match subgradient preserves all model parameters");
        }

        for (float weight : {0.f, 0.2f, 1.f}) {
            HostSession trained(absolute, config, fixture.model());
            trained.set_frame(fixture.frame());
            loss = {sizeof(loss), LFS_HIP_LOSS_L1_SSIM, weight, 1};
            trained.set_loss(loss);
            const float initial = trained.loss();
            for (unsigned i = 0; i < 300; ++i)
                trained.step(rates);
            const float final = trained.loss();
            trained.render(image);
            require(std::isfinite(final) && final < initial * 0.1f && mse(image, fixture.target) < 0.001 &&
                        trained.info().completed_steps == 300 && !trained.info().poisoned,
                    "HIP L1/SSIM session did not converge");
            std::printf("PASS session weight %.1f: 300 steps loss %.9g -> %.9g, image MSE %.9g\n",
                        weight, initial, final, mse(image, fixture.target));
            trained.close();
        }
    }
} // namespace

int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(argc == 2 || argc == 3, "Expected training DLL path and optional --large-model");
#ifdef LFS_HIP_SESSION_WITH_CUDA
    CudaGuard cuda_guard;
#endif
    if (argc == 3 && std::string(argv[2]) == "--sh") {
        Fixture fixture;
        hip_sh_test::session(std::filesystem::absolute(argv[1]), fixture);
#ifdef LFS_HIP_SESSION_WITH_CUDA
        cuda_guard.verify();
#endif
        return 0;
    }
    if (argc == 3 && std::string(argv[2]) == "--large-sh") {
        Fixture fixture;
        hip_sh_test::large_session(std::filesystem::absolute(argv[1]), fixture);
#ifdef LFS_HIP_SESSION_WITH_CUDA
        cuda_guard.verify();
#endif
        return 0;
    }
    if (argc == 3 && std::string(argv[2]) == "--photometric") {
        photometric_session(argv[1]);
#ifdef LFS_HIP_SESSION_WITH_CUDA
        cuda_guard.verify();
#endif
        return 0;
    }
    if (argc == 3 && std::string(argv[2]) == "--host-adapter") {
        host_adapter(argv[1]);
#ifdef LFS_HIP_SESSION_WITH_CUDA
        cuda_guard.verify();
#endif
        return 0;
    }
    Api api(argv[1]);
    require(api.lfs_hip_train_abi() == 1, "Unexpected HIP training ABI");
    char runtime_path[32768]{};
    const auto runtime = GetModuleHandleA("amdhip64_7.dll");
    require(runtime && GetModuleFileNameA(runtime, runtime_path, sizeof(runtime_path)), "Cannot identify HIP training runtime");
    int runtime_version = 0;
    require(symbol<int (*)(int*)>(runtime, "hipRuntimeGetVersion")(&runtime_version) == 0, "Cannot read HIP training runtime version");
    std::printf("HIP training runtime: %d, %s\n", runtime_version, runtime_path);
    if (argc == 3) {
        require(std::string(argv[2]) == "--large-model", "Unknown HIP training test option");
        large_model(api);
        api.close();
#ifdef LFS_HIP_SESSION_WITH_CUDA
        cuda_guard.verify();
#endif
        return 0;
    }
    Fixture fixture;
    LfsHipError error{};
    LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
    auto model = fixture.model();
    auto frame = fixture.frame();
    const LfsHipTrainRates rates{0.002f, 0.002f, 0.003f, 0.015f / sh_c0, 0.01f};
    Session session(api);
    auto bad_config = config;
    bad_config.device_ordinal = INT32_MAX;
    rejected(api.lfs_hip_train_create(&bad_config, &model, &session.handle, &error), LFS_HIP_TRAIN_INVALID, error);
    require(!session.handle, "Invalid device published a session");
    bad_config = config;
    bad_config.allocation_budget_bytes = 1;
    rejected(api.lfs_hip_train_create(&bad_config, &model, &session.handle, &error), LFS_HIP_TRAIN_RESOURCE_EXHAUSTED, error);
    require(!session.handle, "Budget failure published a session");
    const float saved_rotation = fixture.values[1][0];
    fixture.values[1][0] = 0;
    rejected(api.lfs_hip_train_create(&config, &model, &session.handle, &error), LFS_HIP_TRAIN_INVALID, error);
    require(!session.handle, "Invalid quaternion published a session");
    fixture.values[1][0] = saved_rotation;
    check(api.lfs_hip_train_create(&config, &model, &session.handle, &error), error);
    std::vector<float> image(channels);
    rejected(api.lfs_hip_train_render(session.handle, image.data(), image.size(), &error), LFS_HIP_TRAIN_INVALID, error);
    check(api.lfs_hip_train_set_frame(session.handle, &frame, &error), error);
    check(api.lfs_hip_train_render(session.handle, image.data(), image.size(), &error), error);
    float max_error = 0;
    for (size_t i = 0; i < image.size(); ++i)
        max_error = std::max(max_error, std::abs(image[i] - fixture.target[i]));
    require(max_error < 3e-5f, "C ABI session forward differs from CPU reference");
    std::printf("PASS C ABI SH0 render vs CPU: max error %.9g\n", max_error);
    check(api.lfs_hip_train_destroy(session.handle, &error), error);
    session.handle = nullptr;
    fixture.perturb();
    model = fixture.model();
    check(api.lfs_hip_train_create(&config, &model, &session.handle, &error), error);
    check(api.lfs_hip_train_set_frame(session.handle, &frame, &error), error);
    rejected(api.lfs_hip_train_render(session.handle, image.data(), image.size() - 1, &error), LFS_HIP_TRAIN_INVALID, error);
    for (unsigned family = 0; family < 5; ++family)
        require(read(api, session.handle, family, fixture.values[family].size()) == fixture.values[family], "Model upload changed parameter values");
    LfsHipTrainInfo info{};
    check(api.lfs_hip_train_info(session.handle, &info, &error), error);
    const auto live_before = info.allocated_bytes;
    check(api.lfs_hip_train_set_budget(session.handle, live_before, &error), error);
    rejected(api.lfs_hip_train_step(session.handle, &rates, &error), LFS_HIP_TRAIN_RESOURCE_EXHAUSTED, error);
    check(api.lfs_hip_train_info(session.handle, &info, &error), error);
    require(!info.completed_steps && !info.poisoned && info.allocated_bytes == live_before, "Budget failure changed session state or leaked allocations");
    for (unsigned family = 0; family < 5; ++family)
        require(read(api, session.handle, family, fixture.values[family].size()) == fixture.values[family], "Budget failure modified model");
    auto bigger = frame;
    std::vector<float> bigger_target(64 * 64 * 3, 0.5f);
    bigger.width = bigger.height = 64;
    bigger.target_chw = bigger_target.data();
    check(api.lfs_hip_train_set_budget(session.handle, live_before + bigger_target.size() * sizeof(float), &error), error);
    rejected(api.lfs_hip_train_set_frame(session.handle, &bigger, &error), LFS_HIP_TRAIN_RESOURCE_EXHAUSTED, error);
    check(api.lfs_hip_train_info(session.handle, &info, &error), error);
    require(info.width == width && info.height == height && info.allocated_bytes == live_before, "Failed frame replacement lost old frame or memory");
    check(api.lfs_hip_train_set_budget(session.handle, config.allocation_budget_bytes, &error), error);
    Session control(api);
    check(api.lfs_hip_train_create(&config, &model, &control.handle, &error), error);
    check(api.lfs_hip_train_set_frame(control.handle, &frame, &error), error);
    check(api.lfs_hip_train_step(session.handle, &rates, &error), error);
    check(api.lfs_hip_train_step(control.handle, &rates, &error), error);
    for (unsigned family = 0; family < 5; ++family) {
        const auto recovered = read(api, session.handle, family, fixture.values[family].size());
        const auto fresh = read(api, control.handle, family, recovered.size());
        for (size_t i = 0; i < recovered.size(); ++i)
            require(std::abs(recovered[i] - fresh[i]) < 1e-5f, "Budget failure altered first optimizer update");
    }
    std::puts("PASS: budget rejection preserves model, step, old frame, live allocations and first optimizer update");
    auto invalid_frame = frame;
    invalid_frame.fx = std::numeric_limits<float>::quiet_NaN();
    rejected(api.lfs_hip_train_set_frame(session.handle, &invalid_frame, &error), LFS_HIP_TRAIN_INVALID, error);
    invalid_frame = frame;
    invalid_frame.world_to_camera[0] = 2;
    rejected(api.lfs_hip_train_set_frame(session.handle, &invalid_frame, &error), LFS_HIP_TRAIN_INVALID, error);
    invalid_frame = frame;
    invalid_frame.world_to_camera[0] = -1;
    rejected(api.lfs_hip_train_set_frame(session.handle, &invalid_frame, &error), LFS_HIP_TRAIN_INVALID, error);
    auto invalid_target = fixture.target;
    invalid_target[0] = std::numeric_limits<float>::quiet_NaN();
    invalid_frame = frame;
    invalid_frame.target_chw = invalid_target.data();
    rejected(api.lfs_hip_train_set_frame(session.handle, &invalid_frame, &error), LFS_HIP_TRAIN_INVALID, error);
    auto invalid_rates = rates;
    invalid_rates.sh0 = -1;
    rejected(api.lfs_hip_train_step(session.handle, &invalid_rates, &error), LFS_HIP_TRAIN_INVALID, error);
    const auto before_cancel = read(api, session.handle, 0, n * 3);
    api.lfs_hip_train_set_cancel(session.handle, 1);
    rejected(api.lfs_hip_train_step(session.handle, &rates, &error), LFS_HIP_TRAIN_CANCELLED, error);
    require(read(api, session.handle, 0, n * 3) == before_cancel, "Cancelled step changed parameters");
    api.lfs_hip_train_set_cancel(session.handle, 0);
    // Start the quality run from exactly the same perturbed model, not from
    // the first-step recovery validation above.
    check(api.lfs_hip_train_destroy(session.handle, &error), error);
    session.handle = nullptr;
    check(api.lfs_hip_train_create(&config, &model, &session.handle, &error), error);
    check(api.lfs_hip_train_set_frame(session.handle, &frame, &error), error);
    check(api.lfs_hip_train_render(session.handle, image.data(), image.size(), &error), error);
    const double initial_loss = mse(image, fixture.target);
    save("session-target.ppm", fixture.target);
    save("session-initial.ppm", image);
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < 400; ++i)
        check(api.lfs_hip_train_step(session.handle, &rates, &error), error);
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    check(api.lfs_hip_train_render(session.handle, image.data(), image.size(), &error), error);
    const double final_loss = mse(image, fixture.target);
    require(final_loss < initial_loss * 0.02, "C ABI training did not converge");
    for (unsigned family = 0; family < 5; ++family)
        for (float value : read(api, session.handle, family, fixture.values[family].size()))
            require(std::isfinite(value), "Nonfinite session parameter");
    save("session-final.ppm", image);
    check(api.lfs_hip_train_info(session.handle, &info, &error), error);
    require(info.completed_steps == 400 && info.allocated_bytes <= info.budget_bytes && info.peak_allocated_bytes <= info.budget_bytes,
            "Session step count or allocation accounting mismatch");
    std::printf("PASS C ABI training: MSE %.9g -> %.9g, 400 steps %.3f ms (%.3f ms/step), live %llu, peak %llu, intersections %llu, capacity %llu\n",
                initial_loss, final_loss, elapsed, elapsed / 400, static_cast<unsigned long long>(info.allocated_bytes),
                static_cast<unsigned long long>(info.peak_allocated_bytes), static_cast<unsigned long long>(info.intersection_count), static_cast<unsigned long long>(info.intersection_capacity));
    // Two independent sessions on two host threads, sharing only HIP's module.
    std::atomic_int failures = 0;
    const auto train = [&](LfsHipTraining* handle) {
        LfsHipError thread_error{};
        for (unsigned i = 0; i < 10; ++i)
            if (api.lfs_hip_train_step(handle, &rates, &thread_error)) {
                ++failures;
                break;
            }
    };
    std::thread one(train, session.handle), two(train, control.handle);
    one.join();
    two.join();
    require(!failures, "Concurrent independent HIP sessions failed");
    std::atomic_bool started = false;
    std::atomic_int worker_status = 0;
    std::thread worker([&] {
        LfsHipError thread_error{};
        started.store(true);
        for (unsigned i = 0; i < 10000; ++i) {
            const int result = api.lfs_hip_train_step(control.handle, &rates, &thread_error);
            if (result) {
                worker_status.store(result);
                break;
            }
        }
    });
    while (!started.load())
        std::this_thread::yield();
    api.lfs_hip_train_set_cancel(control.handle, 1);
    worker.join();
    require(worker_status == LFS_HIP_TRAIN_CANCELLED, "Cross-thread cancellation was not observed");
    api.lfs_hip_train_set_cancel(control.handle, 0);
    check(api.lfs_hip_train_step(control.handle, &rates, &error), error);
    std::puts("PASS: independent sessions, cross-thread cancellation and resume");
    auto culled_frame = frame;
    culled_frame.world_to_camera[11] = -2000;
    check(api.lfs_hip_train_set_frame(control.handle, &culled_frame, &error), error);
    check(api.lfs_hip_train_render(control.handle, image.data(), image.size(), &error), error);
    for (float value : image)
        require(value == 0, "C ABI empty frame did not clear RGB");
    check(api.lfs_hip_train_set_frame(control.handle, &frame, &error), error);
    check(api.lfs_hip_train_step(control.handle, &rates, &error), error);
    std::puts("PASS: translated camera, all-culled frame and frame restore");
    {
        Session clamped(api);
        const float means[] = {0, 0, 3}, quaternion[] = {1, 0, 0, 0}, scales[] = {-1.2f, -1.2f, -1.2f}, coefficients[] = {-4, -4, -4}, opacity[] = {0};
        LfsHipTrainModel black{means, quaternion, scales, coefficients, opacity};
        auto one_config = config;
        one_config.gaussian_count = 1;
        auto white_frame = frame;
        std::vector<float> white(channels, 1);
        white_frame.target_chw = white.data();
        const LfsHipTrainRates sh_only{0, 0, 0, 0.05f, 0};
        check(api.lfs_hip_train_create(&one_config, &black, &clamped.handle, &error), error);
        check(api.lfs_hip_train_set_frame(clamped.handle, &white_frame, &error), error);
        for (unsigned i = 0; i < 5; ++i)
            check(api.lfs_hip_train_step(clamped.handle, &sh_only, &error), error);
        for (float coefficient : read(api, clamped.handle, 3, 3))
            require(coefficient == -4, "SH0 clamp did not suppress its gradient");
        std::puts("PASS: SH0 clamp derivative and packed single-Gaussian layout");
        clamped.close();
    }
    control.close();
    session.close();
    api.close();
#ifdef LFS_HIP_SESSION_WITH_CUDA
    cuda_guard.verify();
#endif
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}

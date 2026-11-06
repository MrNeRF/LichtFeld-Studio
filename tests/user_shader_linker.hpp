/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "shaders/shader_service.hpp"
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace lfs::vis {
    // Fixture commands use production validation and installation without adding UI history or routing a timeline.
    struct ShaderCommands {
        shaders::ShaderService& service;
        auto command(const shaders::ShaderCommand& cmd) {
            return service.applyCommand(cmd, {}, false);
        }
        auto createAsset(std::string name, std::string source) { return command(shaders::SetShaderSource{{}, std::move(source), std::move(name), {}}); }
        auto updateSource(core::Uuid id, std::string source) { return command(shaders::SetShaderSource{id, std::move(source), {}, {}}); }
        auto renameAsset(core::Uuid id, std::string name) { return command(shaders::RenameShader{id, std::move(name)}); }
        auto removeAsset(core::Uuid id) { return command(shaders::RemoveShader{id}); }
        auto setParameter(core::Uuid id, std::string_view name, std::span<const float> values) { return command(shaders::SetShaderParameter{id, std::string(name), {values.begin(), values.end()}, {}}); }
        auto bind(shaders::ShaderSlot slot, core::Uuid id) { return command(shaders::BindShader{slot, id}); }
        auto unbind(shaders::ShaderSlot slot) { return command(shaders::UnbindShader{slot, {}}); }
    };
} // namespace lfs::vis

namespace lfs::test {
    namespace shaders = lfs::vis::shaders;
    // Links on the service worker in the order the test releases them.
    // A composition with sources containing "broken" fails with a
    // diagnostic located in each of their files.
    struct Schedule {
        std::mutex mutex;
        std::condition_variable changed;
        bool gated = false;
        int released = 0; // links that may return since the gate closed
        int returned = 0;
        int linkers = 0;
        std::string failed_host;
        std::vector<std::string> started; // "<source>[+<source>...]@<host>"

        void gate(const bool on) {
            const std::lock_guard lock(mutex);
            gated = on;
            released = returned = 0;
            changed.notify_all();
        }
        void release(const int count = 1) {
            const std::lock_guard lock(mutex);
            released += count;
            changed.notify_all();
        }
        // Waits for `count` links to have started.
        bool waitStarted(const std::size_t count, const std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
            std::unique_lock lock(mutex);
            return changed.wait_for(lock, timeout, [&] { return started.size() >= count; });
        }
    };

    class ScheduledLinker final : public shaders::HostLinker {
    public:
        explicit ScheduledLinker(Schedule& schedule, bool editor = false) : schedule_(schedule), editor_(editor) {}

        ScheduledLinker(Schedule& schedule, std::unique_ptr<shaders::HostLinker> inner)
            : schedule_(schedule), inner_(std::move(inner)) {}

        lfs::Result<shaders::LinkedHost> link(const std::span<const shaders::EffectSource> effects,
                                              const std::string& host,
                                              shaders::CompileDiagnostics& diagnostics) override {
            std::string sources;
            for (const auto& effect : effects)
                sources += (sources.empty() ? "" : "+") + effect.source;
            std::unique_lock lock(schedule_.mutex);
            schedule_.started.push_back(sources + "@" + host);
            schedule_.changed.notify_all();
            schedule_.changed.wait(lock, [&] { return !schedule_.gated || schedule_.released > schedule_.returned; });
            ++schedule_.returned;
            if (inner_ && host != schedule_.failed_host) {
                lock.unlock();
                return inner_->link(effects, host, diagnostics);
            }
            for (const auto& effect : editor_ ? effects.first(1) : effects) {
                if (effect.source.find("broken") == std::string::npos && host != schedule_.failed_host)
                    continue;
                diagnostics.raw += editor_ ? "fixture.slang(2, 3): error E1: <broken> & identifier" : effect.name + "(2:3): error E1: broken effect\n";
                diagnostics.entries.push_back(
                    {shaders::DiagnosticSeverity::Error, "E1", effect.name, 2, 3, editor_ ? "<broken> & identifier" : "broken effect"});
            }
            if (!diagnostics.entries.empty()) {
                return lfs::make_error(lfs::ErrorInit{.code = lfs::ErrorCode::InvalidArgument,
                                                      .domain = lfs::ErrorDomain::Rendering,
                                                      .user_message = editor_ ? "Generated shader failed" : "Shader compilation failed.",
                                                      .detection = LFS_SOURCE_SITE_CURRENT()});
            }
            shaders::LinkedHost linked;
            linked.program.label = host;
            linked.program.spirv = editor_ ? std::vector<std::uint32_t>{0x07230203u, 0, 0, 0, 0} : std::vector<std::uint32_t>{static_cast<std::uint32_t>(std::hash<std::string>{}(sources))};
            linked.schemas.assign(
                effects.size(),
                {{.name = "gain", .slot = 0, .components = 1, .minimum = 0.0f, .maximum = 2.0f, .initial = 1.0f},
                 {.name = "tint", .slot = 1, .components = 3, .minimum = 0.0f, .maximum = 1.0f, .initial = 0.5f}});
            for (std::size_t i = 0; i < effects.size(); ++i) {
                if (effects[i].source == "reordered") {
                    std::swap(linked.schemas[i][0], linked.schemas[i][1]);
                    linked.schemas[i][0].slot = 0;
                    linked.schemas[i][1].slot = 1;
                }
            }
            if (editor_)
                linked.schemas.assign(effects.size(), {{"strength", 0, 1, 0.0f, 1.0f, 0.5f}});
            return linked;
        }

    private:
        Schedule& schedule_;
        bool editor_ = false;
        std::unique_ptr<shaders::HostLinker> inner_;
    };

} // namespace lfs::test

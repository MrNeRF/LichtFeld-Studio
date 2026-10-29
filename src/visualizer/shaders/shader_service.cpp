/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "shaders/shader_service.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "io/project/chapter_binary_utils.hpp"
#include "operation/undo_history.hpp"
#include "scene/visible_splat_bounds.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <source_location>
#include <utility>

namespace lfs::vis::shaders {

    namespace {
        using Values = std::unordered_map<std::string, std::vector<float>>;

        Error serviceError(const ErrorCode code, std::string message,
                           const std::source_location location = std::source_location::current()) {
            return make_error(ErrorInit{
                .code = code,
                .domain = ErrorDomain::Rendering,
                .user_message = std::move(message),
                .detection = core::source_site_from_location(location),
            });
        }

        // A failure the compiler did not describe, as its single diagnostic.
        CompileDiagnostics describe(const Error& error) {
            std::string message(error.detail().empty() ? error.user_message() : error.detail());
            return {.raw = message, .entries = {{DiagnosticSeverity::Error, "LFS", {}, 0, 0, message}}};
        }

        // The virtual file of an asset's source in a composition.
        std::string effectFile(const core::Uuid& asset) { return asset.to_string() + ".slang"; }

        // Slots follow `schema`; a value set for another schema keeps its
        // name but not its component count, and those slots take the initial value.
        rendering::GaussianCustomEffectState::Parameters parameterBlock(const std::vector<Parameter>& schema,
                                                                        const Values& values) {
            rendering::GaussianCustomEffectState::Parameters block{};
            for (const auto& parameter : schema) {
                if (parameter.slot >= block.size())
                    continue;
                auto& slot = block[parameter.slot] = glm::vec4(parameter.initial);
                if (const auto found = values.find(parameter.name);
                    found != values.end() && found->second.size() == parameter.components) {
                    for (std::uint32_t i = 0; i < parameter.components; ++i)
                        slot[static_cast<glm::length_t>(i)] = found->second[i];
                }
            }
            return block;
        }

        class SlangLinker final : public HostLinker {
        public:
            explicit SlangLinker(std::filesystem::path manifest) : manifest_(std::move(manifest)) {}

            Result<LinkedHost> link(const std::span<const EffectSource> effects, const std::string& host,
                                    CompileDiagnostics& diagnostics) override {
                if (!package_) {
                    auto loaded = SlangCompiler::load_package(manifest_);
                    if (!loaded) {
                        diagnostics = describe(loaded.error());
                        return std::move(loaded).error();
                    }
                    package_ = std::move(*loaded);
                }
                auto output = compiler_.compile(
                    {.effects = {effects.begin(), effects.end()}, .package = package_, .host = host}, diagnostics);
                if (!output)
                    return std::move(output).error();
                // compile() accepted the host, so the package holds it.
                const auto& stage = *std::ranges::find(package_->hosts, host,
                                                       [](const HostStage& s) { return s.module.name; });
                std::vector<int> bindings;
                for (const auto& binding : stage.bindings)
                    bindings.push_back(static_cast<int>(binding.binding));
                return LinkedHost{{.label = host,
                                   .bindings = std::move(bindings),
                                   .workgroup = stage.workgroup,
                                   .subgroup_size = stage.subgroup_size,
                                   .spirv = std::move(output->spirv)},
                                  std::move(output->parameters)};
            }

        private:
            std::filesystem::path manifest_;
            std::shared_ptr<const ShaderPackage> package_;
            SlangCompiler compiler_;
        };
    } // namespace

    HostLinkerFactory slangHostLinker(std::filesystem::path manifest) {
        return [manifest = std::move(manifest)] { return std::make_unique<SlangLinker>(manifest); };
    }

    EffectScene captureEffectScene(const core::Scene& scene, const std::size_t slots) {
        EffectScene captured;
        captured.objects.resize(slots);
        std::uint32_t id = 0;
        for (const auto& slot : scene.getVisibleSplatNodeSlots()) {
            if (slot.node && slot.slot_index < slots)
                captured.objects[slot.slot_index] = {slot.node->uuid, id++};
        }
        captured.bounded = visibleSplatBounds(scene, captured.bounds_min, captured.bounds_max);
        return captured;
    }

    std::optional<rendering::GaussianCustomEffectState>
    makeCustomEffectState(const ShaderFrameSnapshot& snapshot, const EffectScene& scene, const float time) {
        using State = rendering::GaussianCustomEffectState;
        State state;
        state.global_binding = snapshot.global;
        bool applies = state.global_binding != State::kNone;
        state.objects.resize(scene.objects.size());
        for (std::size_t i = 0; i < scene.objects.size(); ++i) {
            const auto& object = scene.objects[i];
            if (object.node.is_nil())
                continue;
            state.objects[i].id = object.id;
            if (const auto bound = snapshot.nodes.find(object.node);
                bound != snapshot.nodes.end()) {
                state.objects[i].binding = bound->second;
                applies = true;
            }
        }
        if (!applies)
            return std::nullopt;
        state.time = time;
        state.bounds_min = scene.bounds_min;
        state.bounds_max = scene.bounds_max;
        state.shared_bindings = snapshot.parameters;
        if (!state.shared_bindings)
            state.bindings = snapshot.bindings;
        return state;
    }

    struct ShaderService::AssetContent {
        std::string name;
        std::shared_ptr<const std::string> source = std::make_shared<const std::string>();
        Values values{};
        std::string language = "slang";
        io::JsonChapterDom::Json pending = io::JsonChapterDom::Json::object();
        std::size_t escaped_size = 0;
        void cacheSize() {
            using Json = io::JsonChapterDom::Json;
            escaped_size = Json(name).dump().size() + Json(language).dump().size() + Json(*source).dump().size();
        }
    };
    struct ShaderService::Asset : AssetContent {
        std::uint64_t revision = 1, schema_revision = 0, links = 0;
        std::vector<Parameter> schema;
    };

    // One composition and the hosts draws asked for.
    struct ShaderService::Build : CompositionContent {
        enum class HostState : std::uint8_t { Queued,
                                              Linking,
                                              Linked,
                                              Failed };
        struct Host {
            std::string label;
            HostState state = HostState::Queued;
            std::shared_ptr<const HostProgram> program{};
            bool submitted = false;
        };
        SnapshotPtr frozen;
        std::weak_ptr<int> owner;
        std::vector<std::pair<core::Uuid, Values>> values;
        std::vector<Host> hosts{}; // in the order draws asked for them
        // Reflected by the first link, per member.
        std::vector<std::vector<Parameter>> schemas{};
        CompileDiagnostics diagnostics{};
        // The linked hosts once none waits; a later host adds a new set.
        std::shared_ptr<const ShaderPrograms> programs{};
        bool set_failed = false; // the program set of `programs` cannot be built

        [[nodiscard]] bool waiting() const {
            return std::ranges::any_of(hosts, [](const Host& host) {
                return host.state == HostState::Queued || host.state == HostState::Linking;
            });
        }
        [[nodiscard]] bool linkFailed() const {
            return std::ranges::contains(hosts, HostState::Failed, &Host::state);
        }
        [[nodiscard]] bool anyFailed() const { return set_failed || linkFailed(); }
        // The linked hosts as a new set once none waits, if they changed.
        void publish() {
            if (hosts.empty() || waiting() || anyFailed())
                return;
            std::vector<std::shared_ptr<const HostProgram>> linked;
            for (const auto& host : hosts) {
                linked.push_back(host.program);
            }
            if (programs && programs->hosts == linked)
                return;
            ShaderPrograms set;
            set.hosts = std::move(linked);
            for (std::size_t i = 0; i < members.size(); ++i)
                set.effects.push_back({members[i].asset, members[i].revision, schemas[i]});
            programs = std::make_shared<const ShaderPrograms>(std::move(set));
        }
    };

    struct ShaderService::WorkerMessage {
        std::shared_ptr<Build> build;
        std::uint64_t epoch;
        std::string host;
        std::optional<Result<LinkedHost>> result;
        CompileDiagnostics diagnostics;
    };

    void ShaderChanges::notify(const bool close, const bool cancel) {
        {
            const std::lock_guard lock(mutex);
            closed |= close;
            const auto next = generation.fetch_add(1, std::memory_order_release) + 1;
            if (cancel)
                cancelled_at = next;
        }
        changed.notify_all();
    }
    bool ShaderChanges::cancelled(const std::uint64_t origin) const {
        const std::lock_guard lock(mutex);
        return closed || cancelled_at > origin;
    }
    bool ShaderChanges::wait(const std::uint64_t observed, const std::chrono::steady_clock::time_point deadline,
                             const std::stop_token stop) {
        std::unique_lock lock(mutex);
        return changed.wait_until(lock, stop, deadline, [&] { return closed || generation.load(std::memory_order_acquire) != observed; });
    }

    std::vector<core::Uuid> ShaderService::Bindings::assets() const {
        std::vector<core::Uuid> assets;
        if (global)
            assets.push_back(*global);
        for (const auto& entry : nodes)
            assets.push_back(entry.second);
        std::ranges::sort(assets, {}, &core::Uuid::bytes);
        const auto [first, last] = std::ranges::unique(assets);
        assets.erase(first, last);
        return assets;
    }

    bool ShaderService::Bindings::binds(const core::Uuid& asset) const {
        return global == asset ||
               std::ranges::any_of(nodes, [&](const auto& entry) { return entry.second == asset; });
    }

    std::optional<core::Uuid> ShaderService::Bindings::get(const ShaderSlot& slot) const {
        switch (slot.kind) {
        case ShaderSlot::Kind::Global: return global;
        case ShaderSlot::Kind::Node:
            if (const auto found = nodes.find(slot.node); found != nodes.end())
                return found->second;
        }
        return std::nullopt;
    }

    void ShaderService::Bindings::set(const ShaderSlot& slot, std::optional<core::Uuid> asset) {
        if (slot.kind == ShaderSlot::Kind::Global)
            global = asset;
        else if (asset)
            nodes[slot.node] = *asset;
        else
            nodes.erase(slot.node);
    }

    namespace {
        // A slot of `bindings` whose asset is in `effects` binds that effect
        // with the parameter block `block(effect index)`.
        template <class Effects, class Asset, class Block, class Bindings>
        std::shared_ptr<ShaderFrameSnapshot> layOut(const Effects& effects, const Asset asset_of, const Block& block,
                                                    const Bindings& bindings) {
            ShaderFrameSnapshot snapshot;
            const auto bind = [&](ShaderSlot slot, const std::optional<core::Uuid>& asset) {
                const auto found = asset ? std::ranges::find(effects, *asset, asset_of) : effects.end();
                if (found == effects.end())
                    return ShaderFrameSnapshot::kNone;
                const auto effect = static_cast<std::uint32_t>(found - effects.begin());
                snapshot.slots.push_back({slot, *asset});
                snapshot.bindings.push_back({block(effect), effect});
                return static_cast<std::uint32_t>(snapshot.bindings.size() - 1);
            };
            snapshot.global = bind({}, bindings.global);
            for (const auto& [node, asset] : bindings.nodes) {
                if (const auto binding = bind({ShaderSlot::Kind::Node, node}, asset); binding != ShaderFrameSnapshot::kNone)
                    snapshot.nodes.emplace(node, binding);
            }
            snapshot.parameters = std::make_shared<const decltype(snapshot.bindings)>(snapshot.bindings);
            return std::make_shared<ShaderFrameSnapshot>(std::move(snapshot));
        }
    } // namespace

    ShaderService::ShaderService(const core::Scene& scene, HostLinkerFactory linker, std::function<void()> wake)
        : scene_(scene),
          linker_(std::move(linker)),
          wake_(std::move(wake)),
          owner_(std::this_thread::get_id()) {
        using namespace core::events::state;
        events_.subscribe<PLYRemoved>([this](const auto&) { reconcileNodes(); });
        events_.subscribe<PLYAdded>([this](const auto&) { reconcileNodes(); });
        events_.subscribe<SceneCleared>([this](const auto&) { reconcileNodes(); });
    }

    ShaderService::~ShaderService() {
        stop();
        if (worker_.joinable())
            worker_.join();
    }

    void ShaderService::stop() {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
        changes_->notify(true);
        changed_.notify_all();
    }

    void ShaderService::wake() const {
        changes_->notify();
        if (wake_)
            wake_();
    }

    Status ShaderService::validateTarget(const ShaderSlot& slot) const {
        if ((slot.kind != ShaderSlot::Kind::Global && slot.kind != ShaderSlot::Kind::Node) ||
            (slot.kind != ShaderSlot::Kind::Node && !slot.node.is_nil()))
            return Status::failure(serviceError(ErrorCode::InvalidArgument, "Invalid shader target."));
        if (slot.kind == ShaderSlot::Kind::Node) {
            const auto* const node = scene_.getNodeByUuid(slot.node);
            if (!node)
                return Status::failure(serviceError(ErrorCode::NotFound, "No such scene node."));
            if (node->type != core::NodeType::SPLAT) {
                return Status::failure(
                    serviceError(ErrorCode::InvalidArgument, std::format("'{}' is not a splat node.", node->name)));
            }
        }
        return {};
    }

    Status ShaderService::validateBinding(const ShaderSlot& slot, const core::Uuid& asset) const {
        const std::lock_guard lock(mutex_);
        if (stopping_)
            return Status::failure(serviceError(ErrorCode::Unavailable, "Shader service has stopped."));
        return validateBindingLocked(slot, asset);
    }

    Status ShaderService::validateBindingLocked(const ShaderSlot& slot, const core::Uuid& asset) const {
        if (auto valid = validateTarget(slot); !valid)
            return valid;
        if (!asset.is_nil() && !assets_.contains(asset))
            return Status::failure(serviceError(ErrorCode::NotFound, "No such shader."));
        if (slot.kind == ShaderSlot::Kind::Node && !bindings_.nodes.contains(slot.node) && bindings_.nodes.size() + bindings_.global.has_value() >= io::project::kMaxShaderBindings)
            return Status::failure(serviceError(ErrorCode::ResourceExhausted, "Shader binding limit exceeded."));
        Bindings next = bindings_;
        next.set(slot, asset);
        if (next.nodes.size() + next.global.has_value() > io::project::kMaxShaderBindings || !libraryFitsLocked({}, nullptr, &next))
            return Status::failure(serviceError(ErrorCode::ResourceExhausted, "Shader binding or chapter size limit exceeded."));
        if (next.assets().size() > kMaxBoundShaders) {
            return Status::failure(serviceError(
                ErrorCode::FailedPrecondition,
                std::format("At most {} shaders can be bound at once; unbind one first.", kMaxBoundShaders)));
        }
        return {};
    }

    std::vector<ShaderAsset> ShaderService::assets(const bool include_source) const {
        const std::lock_guard lock(mutex_);
        std::vector<ShaderAsset> result;
        for (const auto& [id, asset] : assets_)
            result.push_back({id, asset.name, include_source ? *asset.source : std::string{}, asset.revision});
        std::ranges::sort(result, {}, &ShaderAsset::name);
        return result;
    }

    std::optional<core::Uuid> ShaderService::boundAsset(const ShaderSlot& slot, const bool include_detached) const {
        const std::lock_guard lock(mutex_);
        const auto asset = bindings_.get(slot);
        if (!asset && include_detached && slot.kind == ShaderSlot::Kind::Node)
            if (const auto found = detached_.find(slot.node); found != detached_.end())
                return found->second;
        return asset;
    }

    bool ShaderService::libraryFitsLocked(const core::Uuid id, const AssetContent* replacement, const Bindings* bindings) const {
        using Json = io::JsonChapterDom::Json;
        // Reserve JSON punctuation, UUIDs and field names; text uses its escaped size.
        std::size_t bytes = 256 + 160 * ((bindings ? bindings : &bindings_)->nodes.size() + 1);
        const auto add = [&](const AssetContent& asset) {
            std::size_t parameters = 2, count = 0;
            const auto field = [&](const std::string& name, const Json& value) {
                parameters += (count++ ? 1 : 0) + Json(name).dump().size() + 1 + value.dump().size();
            };
            for (const auto& [name, value] : asset.pending.items())
                field(name, value);
            for (const auto& [name, value] : asset.values)
                if (!asset.pending.contains(name))
                    field(name, value.size() == 1 ? Json(value.front()) : Json(value));
            bytes += 160 + asset.escaped_size + parameters;
        };
        for (const auto& [key, asset] : assets_)
            if (key != id)
                add(asset);
        if (replacement)
            add(*replacement);
        return bytes <= io::project::kMaxShaderChapterBytes;
    }

    bool ShaderService::sourceFitsLocked(std::string_view source, const std::optional<core::Uuid> replacing) const {
        if (source.size() > kMaxShaderSourceBytes || !io::project::chapter_binary::valid_utf8(source))
            return false;
        std::size_t total = source.size();
        for (const auto& [id, asset] : assets_)
            if (id != replacing)
                total += asset.source->size();
        return total <= io::project::kMaxShaderTotalSourceBytes;
    }

    ShaderProjectToken ShaderService::projectToken() const {
        if (std::this_thread::get_id() == owner_)
            return project_token_;
        const std::lock_guard lock(mutex_);
        return project_token_;
    }

    Result<ShaderStatus> ShaderService::status(const core::Uuid& asset) const {
        const_cast<ShaderService*>(this)->acceptResults();
        const std::lock_guard lock(mutex_);
        return statusLocked(asset);
    }

    Result<ShaderStatus> ShaderService::statusLocked(const core::Uuid& asset) const {
        const auto found = assets_.find(asset);
        if (found == assets_.end())
            return serviceError(ErrorCode::NotFound, "No such shader.");
        const auto& entry = found->second;
        ShaderStatus status;
        status.schema_revision = entry.schema_revision;
        status.revision = entry.revision;
        status.parameters = entry.schema;
        status.links = entry.links;
        const auto block = parameterBlock(entry.schema, entry.values);
        for (const auto& parameter : entry.schema)
            status.values.push_back(block[parameter.slot]);
        const auto display = display_.lock();
        const auto shown = display && display->custom && display->snapshot ? display->snapshot->programs : nullptr;
        if (shown && std::ranges::contains(display->snapshot->slots, asset, &ShaderBinding::asset)) {
            if (const auto effect = std::ranges::find(shown->effects, asset, &ShaderPrograms::Effect::asset);
                effect != shown->effects.end())
                status.displayed_revision = effect->revision;
        }
        if (entry.language != "slang") {
            status.state = ShaderState::Failed;
            status.diagnostics_revision = entry.revision;
            status.diagnostics = describe(serviceError(ErrorCode::Unsupported, "Unsupported shader language: " + entry.language));
            return status;
        }
        if (!bindings_.binds(asset))
            return status;
        const Build& build = *desired_;
        if (!build.diagnostics.entries.empty() || !build.diagnostics.raw.empty() || !build.schemas.empty())
            status.diagnostics_revision = entry.revision;
        for (const auto& host : build.hosts) {
            if (host.state == Build::HostState::Failed)
                status.failed.push_back(host.label);
            else if (host.state != Build::HostState::Linked)
                status.pending.push_back(host.label);
            status.compiling |= host.state == Build::HostState::Linking;
        }
        const auto file = effectFile(asset);
        status.diagnostics.raw = build.diagnostics.raw;
        for (const auto& diagnostic : build.diagnostics.entries) {
            if (diagnostic.file.empty() || diagnostic.file == file)
                status.diagnostics.entries.push_back(diagnostic);
        }
        // A failure that names no member's file is every member's.
        std::vector<core::Uuid> blamed;
        for (const auto& member : build.members) {
            if (std::ranges::any_of(build.diagnostics.entries, [&](const Diagnostic& d) {
                    return d.severity == DiagnosticSeverity::Error && d.file == effectFile(member.asset);
                }))
                blamed.push_back(member.asset);
        }
        if (build.anyFailed() && (blamed.empty() || std::ranges::contains(blamed, asset)))
            status.state = ShaderState::Failed;
        else if (status.displayed_revision == entry.revision && display->complete)
            status.state = ShaderState::Applied;
        else if (build.anyFailed()) {
            status.state = ShaderState::Blocked;
            status.blocked_by = std::move(blamed);
        } else
            status.state = ShaderState::Pending;
        return status;
    }

    void ShaderService::clear() {
        {
            const std::lock_guard lock(mutex_);
            ++project_token_[0];
            project_token_[1] = 0;
            assets_.clear();
            removed_revisions_.clear();
            history_lifetime_ = std::make_shared<int>(0);
            changes_->notify(false, true);
            bindings_ = {};
            has_bindings_.store(false, std::memory_order_release);
            detached_.clear();
            desired_.reset();
            linking_.reset();
            display_.reset();
            retained_.reset();
            linked_.reset();
            requests_.clear();
            demanded_.clear();
            snapshot_.reset();
            changed_.notify_all();
        }
        bindingsChanged();
        wake();
    }

    bool ShaderService::hasApplicableBindings() const {
        const std::lock_guard lock(mutex_);
        if (bindings_.global)
            return true;
        for (const auto& slot : scene_.getVisibleSplatNodeSlots())
            if (slot.node && bindings_.nodes.contains(slot.node->uuid))
                return true;
        return false;
    }

    SnapshotPtr ShaderService::candidate() const {
        const_cast<ShaderService*>(this)->acceptResults();
        const std::lock_guard lock(mutex_);
        return snapshot_;
    }

    void ShaderService::observeDisplay(std::shared_ptr<const DisplayRecord> display) {
        ShaderEvent event{.kind = ShaderEvent::Kind::Applied};
        {
            const std::lock_guard lock(mutex_);
            if (display && display->snapshot && display->snapshot->epoch != project_token_[0])
                return;
            if (display_.lock() == display)
                return;
            display_ = display;
            if (display && display->custom && display->snapshot) {
                event.members = display->snapshot->content->members;
                event.bindings = display->snapshot->slots;
            }
        }
        changes_->notify();
        lfs::event::emit(event);
    }

    void ShaderService::failCandidate(SnapshotPtr snapshot, const Error& error) {
        if (!snapshot)
            return;
        {
            const std::lock_guard lock(mutex_);
            if (!desired_ || snapshot->content != desired_ || snapshot->epoch != project_token_[0])
                return;
            desired_->set_failed = true;
            desired_->diagnostics = describe(error);
            snapshot_.reset();
        }
        lfs::event::emit(ShaderEvent{.kind = ShaderEvent::Kind::Failed, .members = snapshot->content->members, .diagnostics = describe(error)});
        wake();
    }

    void ShaderService::deviceReset() {
        const std::lock_guard lock(mutex_);
        if (desired_)
            desired_->set_failed = false;
        refreshSnapshotLocked();
        changes_->notify();
    }

    void ShaderService::demandLive(const std::span<const std::string> hosts, SnapshotPtr retained) {
        const std::lock_guard lock(mutex_);
        demanded_.assign(hosts.begin(), hosts.end());
        if (retained && retained->epoch == project_token_[0] && retained->content != desired_) {
            // The composition itself, so a host it is still linking is not linked again; it links after the current one.
            if (retained_ != retained->content) {
                retained_.reset();
                if (const auto found = std::ranges::find_if(requests_, [&](const auto& request) { return request.lock() == retained->content; }); found != requests_.end()) {
                    retained_ = found->lock();
                    std::rotate(found, found + 1, requests_.end());
                }
            }
        } else
            retained_.reset();
        for (const auto& build : {desired_, retained_})
            if (build)
                for (const auto& host : demanded_)
                    queueHostLocked(*build, host);
        refreshSnapshotLocked();
    }

    SnapshotPtr ShaderService::restrictFallback(SnapshotPtr retained) const {
        const std::lock_guard lock(mutex_);
        if (!retained || retained->epoch != project_token_[0] || !hasBindings())
            return {};
        const auto programs = retained_ && retained_->epoch == retained->epoch && retained_->revision == retained->content->revision && !retained_->waiting() && !retained_->anyFailed() ? retained_->programs : retained->programs;
        const auto removed = [&](const ShaderBinding& slot) {
            return slot.target.kind == ShaderSlot::Kind::Global ? !bindings_.global : !bindings_.nodes.contains(slot.target.node);
        };
        if (std::ranges::all_of(retained->slots, removed))
            return {};
        if (programs == retained->programs && std::ranges::none_of(retained->slots, removed))
            return retained;
        auto next = std::make_shared<ShaderFrameSnapshot>(*retained);
        next->programs = programs;
        std::erase_if(next->slots, removed);
        if (!bindings_.global)
            next->global = ShaderFrameSnapshot::kNone;
        std::erase_if(next->nodes, [&](const auto& node) { return !bindings_.nodes.contains(node.first); });
        ++next->revision;
        return next;
    }

    void ShaderService::queueHostLocked(Build& build, const std::string& host) {
        // Nothing links once stopped.
        if (stopping_ || std::ranges::contains(build.hosts, host, &Build::Host::label))
            return;
        // The composition's sources already failed to link; this host would too.
        if (build.linkFailed()) {
            build.hosts.push_back({.label = host, .state = Build::HostState::Failed});
            return;
        }
        build.hosts.push_back({.label = host});
        if (!worker_.joinable())
            worker_ = std::thread([this] { runWorker(); });
        changed_.notify_all();
    }

    void ShaderService::retargetLocked() {
        has_bindings_.store(bindings_.global || !bindings_.nodes.empty(), std::memory_order_release);
        std::vector<ShaderMember> members;
        for (const auto& asset : bindings_.assets())
            if (assets_.at(asset).language == "slang")
                members.push_back({asset, assets_.at(asset).revision});
        if (desired_ && desired_->members == members)
            return;
        // The composition left keeps its linked hosts, not its program set: binding it again links nothing.
        if (desired_ && desired_->programs && !desired_->linkFailed()) {
            auto linked = std::make_shared<Build>(*desired_);
            linked->programs.reset();
            linked->set_failed = false;
            std::erase_if(linked->hosts, [](const auto& host) { return host.state != Build::HostState::Linked; });
            linked_ = std::move(linked);
        }
        if (members.empty()) {
            desired_.reset();
            retained_.reset();
            return;
        }
        // Reuse a complete composition the renderer still holds instead of linking it again.
        for (const auto& request : requests_)
            if (auto build = request.lock(); build && !build->frozen && build->epoch == project_token_[0] && build->members == members && build->programs && !build->waiting() && !build->anyFailed()) {
                desired_ = std::move(build);
                return;
            }
        const bool relink = !linked_ || linked_->epoch != project_token_[0] || linked_->members != members;
        desired_ = relink ? std::make_shared<Build>() : std::make_shared<Build>(*linked_);
        desired_->revision = ++composition_revision_;
        desired_->epoch = project_token_[0];
        std::erase_if(requests_, [](const auto& request) { return request.expired(); });
        requests_.push_back(desired_);
        if (relink) {
            desired_->members = std::move(members);
            for (const auto& member : desired_->members)
                desired_->effects.push_back({*assets_.at(member.asset).source, effectFile(member.asset)});
        }
        desired_->publish();
        // The hosts the latest effect frame ran link before draws ask.
        for (const auto& host : demanded_)
            queueHostLocked(*desired_, host);
    }

    void ShaderService::refreshSnapshotLocked() {
        if (!desired_ || desired_->waiting() || desired_->anyFailed() || !desired_->programs) {
            snapshot_.reset();
            return;
        }
        if (snapshot_ && snapshot_->revision == project_token_[1] && snapshot_->programs == desired_->programs && snapshot_->content == desired_)
            return;
        const auto& effects = desired_->programs->effects;
        auto snapshot = layOut(effects, &ShaderPrograms::Effect::asset, [&](std::size_t i) { return parameterBlock(effects[i].schema, assets_.at(effects[i].asset).values); }, bindings_);
        snapshot->epoch = project_token_[0];
        snapshot->revision = project_token_[1];
        snapshot->content = desired_;
        snapshot->programs = desired_->programs;
        snapshot_ = std::move(snapshot);
    }

    void ShaderService::reconcileNodes() {
        {
            const std::lock_guard lock(mutex_);
            bool changed = false;
            for (auto it = bindings_.nodes.begin(); it != bindings_.nodes.end();) {
                if (scene_.getNodeByUuid(it->first)) {
                    ++it;
                    continue;
                }
                detached_.insert(*it);
                it = bindings_.nodes.erase(it);
                changed = true;
            }
            for (auto it = detached_.begin(); it != detached_.end();) {
                const auto* const node = scene_.getNodeByUuid(it->first);
                if (!node) {
                    ++it;
                    continue;
                }
                // Its node is back: the binding returns within the cap, else it is gone.
                Bindings next = bindings_;
                next.nodes.insert(*it);
                if (node->type == core::NodeType::SPLAT && next.assets().size() <= kMaxBoundShaders &&
                    next.nodes.size() + next.global.has_value() <= io::project::kMaxShaderBindings && libraryFitsLocked({}, nullptr, &next)) {
                    bindings_ = std::move(next);
                    changed = true;
                } else if (node->type == core::NodeType::SPLAT) {
                    LOG_INFO("The shader binding of '{}' is not restored: at most {} shaders can be bound at once.",
                             node->name, kMaxBoundShaders);
                }
                it = detached_.erase(it);
            }
            if (!changed)
                return;
            markDirtyLocked();
            retargetLocked();
            refreshSnapshotLocked();
        }
        bindingsChanged();
        wake();
    }

    std::shared_ptr<ShaderService::Build> ShaderService::nextBuild() const {
        for (const auto& request : requests_) {
            const auto build = request.lock();
            if (build && (build == desired_ || build == retained_ || !build->owner.expired()) &&
                std::ranges::none_of(build->hosts, [](const auto& h) { return h.submitted && (h.state == Build::HostState::Queued || h.state == Build::HostState::Linking); }) &&
                std::ranges::any_of(build->hosts, [](const auto& h) { return h.state == Build::HostState::Queued && !h.submitted; }))
                return build;
        }
        return {};
    }

    bool ShaderService::currentLocked(const WorkerMessage& message) const {
        return !stopping_ && message.epoch == project_token_[0] &&
               (message.build == desired_ || message.build == retained_ || !message.build->owner.expired());
    }

    void ShaderService::acceptResults() {
        if (std::this_thread::get_id() != owner_ || accepting_results_)
            return;
        accepting_results_ = true;
        struct Reset {
            bool& flag;
            ~Reset() { flag = false; }
        } reset{accepting_results_};
        for (;;) {
            std::unique_lock lock(mutex_);
            if (messages_.empty())
                return;
            auto message = std::move(messages_.front());
            messages_.erase(messages_.begin());
            if (stopping_ || message.epoch != project_token_[0])
                continue;
            // A replaced composition keeps its result quietly: the renderer may still fall back to it.
            const bool current = currentLocked(message);
            auto& build = *message.build;
            ShaderEvent event{.kind = ShaderEvent::Kind::Started, .host = message.host};
            event.members = build.members;
            if (!message.result)
                std::ranges::find(build.hosts, message.host, &Build::Host::label)->state = Build::HostState::Linking;
            if (message.result) {
                auto& linked = *message.result;
                auto& entry = *std::ranges::find(build.hosts, message.host, &Build::Host::label);
                entry.state = linked ? Build::HostState::Linked : Build::HostState::Failed;
                event.kind = linked ? ShaderEvent::Kind::Completed : ShaderEvent::Kind::Failed;
                event.diagnostics = !linked && message.diagnostics.entries.empty() ? describe(linked.error()) : message.diagnostics;
                if (linked) {
                    entry.program = std::make_shared<const HostProgram>(std::move(linked->program));
                    if (build.schemas.empty())
                        build.schemas = std::move(linked->schemas);
                } else {
                    for (auto& host : build.hosts)
                        if (host.state == Build::HostState::Queued)
                            host.state = Build::HostState::Failed;
                }
                if (!linked || !build.anyFailed())
                    build.diagnostics = event.diagnostics;
                for (std::size_t i = 0; i < build.members.size(); ++i) {
                    const auto& member = build.members[i];
                    const auto found = assets_.find(member.asset);
                    // Retained/export work belongs to its old composition, not to edited live assets.
                    if (message.build != desired_ || found == assets_.end() || found->second.revision != member.revision)
                        continue;
                    auto& asset = found->second;
                    ++asset.links;
                    if (linked) {
                        asset.schema = build.schemas[i];
                        asset.schema_revision = member.revision;
                    }
                }
                build.publish();
                refreshSnapshotLocked();
                changed_.notify_all();
            }
            if (!current)
                continue;
            changes_->notify();
            lock.unlock();
            lfs::event::emit(event);
        }
    }

    void ShaderService::runWorker() {
        const core::TaskContext context{.name = "user shader compiler", .domain = ErrorDomain::Rendering, .operation_id = {}, .site = LFS_SOURCE_SITE_CURRENT()};
        core::run_guarded<void>(context, [&]() -> Status {
            std::unique_ptr<HostLinker> linker;
            std::unique_lock lock(mutex_);
            for (;;) {
                changed_.wait(lock, [&] { return stopping_ || nextBuild(); });
                if (stopping_) return {};
                WorkerMessage message{nextBuild(), project_token_[0], {}, {}, {}};
                std::erase_if(requests_, [&](const auto& request) { return request.expired() || request.lock() == message.build; });
                requests_.push_back(message.build);
                auto host = std::ranges::find_if(message.build->hosts, [](const auto& host) { return host.state == Build::HostState::Queued && !host.submitted; });
                host->submitted = true;
                message.host = host->label;
                linking_ = ShaderWork{message.build->members, message.host, true};
                messages_.push_back(message);
                lock.unlock();
                wake();
                core::run_guarded<LinkedHost>(context, [&]() -> Result<LinkedHost> {
                    if (!linker) linker = linker_();
                    if (!linker) return serviceError(ErrorCode::Unavailable, "Shader linker is unavailable.");
                    return linker->link(message.build->effects, message.host, message.diagnostics);
                }, [&](Result<LinkedHost>&& linked) { message.result.emplace(std::move(linked)); });
                lock.lock();
                linking_.reset();
                messages_.push_back(std::move(message));
                lock.unlock();
                wake();
                lock.lock();
            } }, [&](Status&& result) {
            if (result) return;
            stop();
            wake(); });
    }

    std::vector<ShaderBinding> ShaderService::bindingsLocked() const {
        std::vector<ShaderBinding> result;
        if (bindings_.global)
            result.push_back({{}, *bindings_.global});
        for (const auto& [node, asset] : bindings_.nodes)
            result.push_back({{ShaderSlot::Kind::Node, node}, asset});
        return result;
    }

    ShaderInspection ShaderService::inspect(const bool include_source) const {
        const_cast<ShaderService*>(this)->acceptResults();
        const std::lock_guard lock(mutex_);
        return inspectLocked(include_source, true);
    }
    ShaderInspection ShaderService::work() const {
        const_cast<ShaderService*>(this)->acceptResults();
        const std::lock_guard lock(mutex_);
        return inspectLocked(false, false);
    }
    ShaderInspection ShaderService::inspectLocked(const bool source, const bool include_assets) const {
        ShaderInspection result;
        const auto add = [&](const core::Uuid& id, const Asset& asset) {
            result.assets.push_back({{id, asset.name, source ? *asset.source : std::string{}, asset.revision}, *statusLocked(id)});
        };
        if (include_assets)
            for (const auto& [id, asset] : assets_)
                add(id, asset);
        std::ranges::sort(result.assets, {}, [](const auto& item) { return item.asset.id.bytes; });
        if (include_assets) {
            result.bindings = bindingsLocked();
            if (const auto display = display_.lock(); display && display->custom && display->snapshot)
                result.displayed_bindings = display->snapshot->slots;
        }
        if (stopping_)
            return result;
        for (const auto& request : requests_) {
            const auto build = request.lock();
            if (!build)
                continue;
            for (const auto& host : build->hosts) {
                if (build != desired_ && build != retained_ && build->owner.expired())
                    continue;
                if (host.state != Build::HostState::Queued && host.state != Build::HostState::Linking)
                    continue;
                if (!linking_ || linking_->host != host.label || linking_->members != build->members)
                    result.work.push_back({build->members, host.label, host.submitted});
            }
        }
        if (linking_)
            result.work.push_back(*linking_);
        return result;
    }

    void ShaderService::bindingsChanged() const {
        ShaderEvent event{.kind = ShaderEvent::Kind::BindingsChanged};
        {
            const std::lock_guard lock(mutex_);
            event.bindings = bindingsLocked();
        }
        lfs::event::emit(event);
    }

    struct ShaderService::EditState {
        core::Uuid id;
        std::optional<AssetContent> asset;
        std::optional<ShaderSlot> slot;
        std::optional<core::Uuid> binding;
        bool same(const EditState& other) const {
            return id == other.id && slot == other.slot && binding == other.binding &&
                   asset.has_value() == other.asset.has_value() &&
                   (!asset || (asset->name == other.asset->name && (asset->source == other.asset->source || *asset->source == *other.asset->source) &&
                               asset->values == other.asset->values && asset->language == other.asset->language && asset->pending == other.asset->pending));
        }
        std::size_t bytes() const {
            std::size_t size = sizeof(*this);
            if (asset) {
                size += asset->name.size() + asset->source->size();
                for (const auto& [name, values] : asset->values)
                    size += name.size() + values.size() * sizeof(float);
            }
            return size;
        }
    };

    ShaderService::EditState ShaderService::editStateLocked(const core::Uuid& id,
                                                            const std::optional<ShaderSlot>& slot) const {
        EditState state{.id = id, .slot = slot};
        if (const auto it = assets_.find(id); it != assets_.end())
            state.asset = it->second;
        if (slot)
            state.binding = bindings_.get(*slot);
        return state;
    }

    class ShaderService::UndoEdit final : public op::UndoEntry {
    public:
        UndoEdit(ShaderService& service, EditState before, EditState after, std::optional<ShaderSlot> redo_binding)
            : service_(service),
              lifetime_(service.history_lifetime_),
              before_(std::move(before)),
              after_(std::move(after)),
              redo_binding_(redo_binding) {}

        void undo() override { apply(after_, before_); }

        void redo() override {
            if (lifetime_.expired())
                throw op::HistoryStaleEntryError("Shader history belongs to a closed project.");
            if (redo_binding_) {
                const auto asset = service_.status(after_.id) ? after_.id : core::Uuid{};
                if (auto valid = service_.validateBinding(*redo_binding_, asset); !valid)
                    throw op::HistoryStaleEntryError(std::string(valid.error().user_message()));
            }
            apply(before_, after_);
        }

        std::string name() const override { return LOC("editor.shader"); }

        std::size_t estimatedBytes() const override { return sizeof(*this) + before_.bytes() + after_.bytes(); }

    private:
        void apply(const EditState& expected, const EditState& desired) {
            if (lifetime_.expired())
                throw op::HistoryStaleEntryError("Shader history belongs to a closed project.");
            if (auto result = service_.restoreEdit(expected, desired); !result)
                throw op::HistoryStaleEntryError(std::string(result.error().user_message()));
        }
        ShaderService& service_;
        std::weak_ptr<void> lifetime_;
        EditState before_, after_;
        std::optional<ShaderSlot> redo_binding_;
    };

    Status ShaderService::installEdit(const EditState& expected, const EditState& desired) {
        {
            if (desired.asset && (((!expected.asset || desired.asset->source != expected.asset->source) && !sourceFitsLocked(*desired.asset->source, desired.id)) ||
                                  (!assets_.contains(desired.id) && assets_.size() >= io::project::kMaxShaderAssets)))
                return Status::failure(serviceError(ErrorCode::ResourceExhausted, "Shader library limit exceeded."));
            if (!editStateLocked(expected.id, expected.slot).same(expected))
                return Status::failure(serviceError(ErrorCode::FailedPrecondition, "Shader history is stale."));
            Bindings next = bindings_;
            if (desired.slot) {
                if (desired.binding && !assets_.contains(*desired.binding) &&
                    !(desired.asset && desired.id == *desired.binding))
                    return Status::failure(serviceError(ErrorCode::NotFound, "The bound shader no longer exists."));
                const auto slot = *desired.slot;
                if (slot.kind == ShaderSlot::Kind::Node) {
                    const auto* node = scene_.getNodeByUuid(slot.node);
                    if (!node || node->type != core::NodeType::SPLAT)
                        return Status::failure(
                            serviceError(ErrorCode::FailedPrecondition, "The shader target no longer exists."));
                }
                next.set(slot, desired.binding);
                if (next.nodes.size() + (next.global ? 1 : 0) > io::project::kMaxShaderBindings)
                    return Status::failure(serviceError(ErrorCode::ResourceExhausted, "Shader binding limit exceeded."));
                if (next.assets().size() > kMaxBoundShaders)
                    return Status::failure(serviceError(ErrorCode::FailedPrecondition, "Too many bound shaders."));
            }
            if (!libraryFitsLocked(desired.id, desired.asset ? &*desired.asset : nullptr, &next))
                return Status::failure(serviceError(ErrorCode::ResourceExhausted, "Shader chapter size limit exceeded."));
            if (expected.asset && !desired.asset &&
                (next.binds(desired.id) ||
                 std::ranges::any_of(detached_, [&](const auto& item) { return item.second == desired.id; })))
                return Status::failure(
                    serviceError(ErrorCode::FailedPrecondition, "Unbind the shader before removing it."));
            if (desired.asset) {
                installContentLocked(desired.id, *desired.asset);
            } else if (expected.asset) {
                removed_revisions_[desired.id] = assets_.at(desired.id).revision;
                assets_.erase(desired.id);
            }
            bindings_ = std::move(next);
            markDirtyLocked();
            retargetLocked();
            refreshSnapshotLocked();
        }
        return {};
    }

    void ShaderService::installContentLocked(const core::Uuid id, AssetContent content) {
        auto found = assets_.find(id);
        if (found == assets_.end()) {
            Asset asset;
            static_cast<AssetContent&>(asset) = std::move(content);
            asset.revision = removed_revisions_[id] + 1;
            assets_.emplace(id, std::move(asset));
        } else {
            auto& current = found->second;
            current.revision += (current.source != content.source && *current.source != *content.source) || current.language != content.language;
            static_cast<AssetContent&>(current) = std::move(content);
        }
    }

    Status ShaderService::restoreEdit(const EditState& expected, const EditState& desired) {
        {
            const std::lock_guard lock(mutex_);
            if (auto result = installEdit(expected, desired); !result)
                return result;
        }
        if (desired.slot)
            bindingsChanged();
        wake();
        return {};
    }

    Result<core::Uuid> ShaderService::command(const ShaderCommand& command,
                                              const std::optional<ShaderSlot> redo_binding) {
        return applyCommand(command, redo_binding, true);
    }
    Result<core::Uuid> ShaderService::applyCommand(const ShaderCommand& command,
                                                   const std::optional<ShaderSlot> redo_binding, const bool authoring) {
        acceptResults();
        if (std::this_thread::get_id() != owner_)
            return serviceError(ErrorCode::ContractViolation, "Shader commands require the viewer thread.");
        std::optional<ShaderSlot> slot;
        std::visit([&](const auto& cmd) {
            using T = std::decay_t<decltype(cmd)>;
            if constexpr (std::is_same_v<T, BindShader> || std::is_same_v<T, UnbindShader> || std::is_same_v<T, RunShader>)
                slot = cmd.target;
        },
                   command);
        if (const auto* unbind = std::get_if<UnbindShader>(&command);
            unbind && slot->kind == ShaderSlot::Kind::Node) {
            std::unique_lock lock(mutex_);
            if (stopping_)
                return serviceError(ErrorCode::Unavailable, "Shader service has stopped.");
            if (const auto found = detached_.find(slot->node); found != detached_.end()) {
                if (unbind->asset && unbind->asset != found->second)
                    return serviceError(ErrorCode::FailedPrecondition, "Shader binding does not match.");
                const auto asset = found->second;
                // Forget the node undo reference; undoing deletion must not restore a stopped binding.
                detached_.erase(found);
                markDirtyLocked();
                lock.unlock();
                bindingsChanged();
                wake();
                return asset;
            }
        }
        if (slot) {
            if (auto valid = validateTarget(*slot); !valid)
                return valid.error();
        }
        EditState before, after;
        core::Uuid id;
        {
            const std::lock_guard lock(mutex_);
            if (stopping_)
                return serviceError(ErrorCode::Unavailable, "Shader service has stopped.");
            auto prepared = prepareEdit(command);
            if (!prepared)
                return prepared.error();
            after = std::move(*prepared);
            id = after.id;
            before = editStateLocked(id, slot);
            if (!before.same(after) || slot || std::holds_alternative<SetShaderParameter>(command)) {
                if (auto result = installEdit(before, after); !result)
                    return result.error();
            }
        }
        if (std::holds_alternative<UnbindShader>(command))
            id = before.binding.value_or(id);
        auto& history = op::undoHistory();
        if (authoring && !before.same(after) && !history.isPlaybackActive())
            history.push(std::make_unique<UndoEdit>(*this, std::move(before), std::move(after), redo_binding));
        if (slot)
            bindingsChanged();
        wake();
        if (const auto* bind = std::get_if<BindShader>(&command))
            return bind->asset;
        return id;
    }

    Result<ShaderService::EditState> ShaderService::prepareEdit(const ShaderCommand& command) const {
        return std::visit([&](const auto& cmd) -> Result<EditState> {
            using T = std::decay_t<decltype(cmd)>;
            core::Uuid id;
            std::optional<ShaderSlot> slot;
            if constexpr (std::is_same_v<T, SetShaderSource> || std::is_same_v<T, RunShader>)
                id = cmd.asset.value_or(core::generate_uuid_v4());
            else if constexpr (!std::is_same_v<T, BindShader> && !std::is_same_v<T, UnbindShader>)
                id = cmd.asset;
            if constexpr (std::is_same_v<T, RunShader> || std::is_same_v<T, BindShader> || std::is_same_v<T, UnbindShader>)
                slot = cmd.target;
            auto next = editStateLocked(id, slot);
            if constexpr (std::is_same_v<T, BindShader> || std::is_same_v<T, RunShader>) {
                const auto bound = [&] { if constexpr (std::is_same_v<T, BindShader>) return cmd.asset; else return id; }();
                if (auto valid = validateBindingLocked(*slot, assets_.contains(bound) ? bound : core::Uuid{}); !valid)
                    return valid.error();
                if constexpr (std::is_same_v<T, BindShader>) {
                    if (!assets_.contains(bound))
                        return serviceError(ErrorCode::NotFound, "No such shader.");
                }
                next.binding = bound;
            } else if constexpr (std::is_same_v<T, UnbindShader>) {
                if (!next.binding || (cmd.asset && cmd.asset != next.binding))
                    return serviceError(ErrorCode::FailedPrecondition, "Shader binding does not match.");
                next.binding.reset();
            }
            if constexpr (std::is_same_v<T, SetShaderSource> || std::is_same_v<T, SetShaderParameter>) {
                if (cmd.expected_revision && (!assets_.contains(id) || assets_.at(id).revision != *cmd.expected_revision))
                    return serviceError(ErrorCode::FailedPrecondition, "Shader source revision does not match.");
            }
            if constexpr (std::is_same_v<T, SetShaderSource> || std::is_same_v<T, RunShader>) {
                if (cmd.asset && !next.asset)
                    return serviceError(ErrorCode::NotFound, "No such shader.");
                if (cmd.source.size() > kMaxShaderSourceBytes)
                    return serviceError(ErrorCode::InvalidArgument, "Shader source exceeds 1 MiB.");
                if (!sourceFitsLocked(cmd.source, id))
                    return serviceError(ErrorCode::ResourceExhausted, "Shader library limit exceeded or source is not UTF-8.");
                if (!next.asset)
                    next.asset = AssetContent{.name = LOC("shader.untitled")};
                if (*next.asset->source != cmd.source)
                    next.asset->source = std::make_shared<const std::string>(cmd.source);
                next.asset->language = "slang";
                if constexpr (std::is_same_v<T, SetShaderSource>)
                    if (cmd.name)
                        next.asset->name = *cmd.name;
            } else if constexpr (std::is_same_v<T, RenameShader> || std::is_same_v<T, RemoveShader> || std::is_same_v<T, SetShaderParameter>) {
                if (!next.asset)
                    return serviceError(ErrorCode::NotFound, "No such shader.");
                if constexpr (std::is_same_v<T, RenameShader>) {
                    next.asset->name = cmd.name;
                } else if constexpr (std::is_same_v<T, RemoveShader>)
                    next.asset.reset();
                else {
                    const auto& schema = assets_.at(id).schema;
                    const auto parameter = std::ranges::find(schema, cmd.name, &Parameter::name);
                    if (parameter == schema.end())
                        return serviceError(ErrorCode::NotFound, std::format("Shader '{}' has no parameter '{}'.", next.asset->name, cmd.name));
                    if (cmd.value.size() != parameter->components)
                        return serviceError(ErrorCode::InvalidArgument, std::format("Parameter '{}' takes {} values, not {}.", cmd.name, parameter->components, cmd.value.size()));
                    for (float v : cmd.value)
                        if (!std::isfinite(v) || v < parameter->minimum || v > parameter->maximum)
                            return serviceError(ErrorCode::InvalidArgument, std::format("Parameter '{}' takes values from {} to {}.", cmd.name, parameter->minimum, parameter->maximum));
                    if (cmd.name.size() > io::project::kMaxShaderNames || (!next.asset->values.contains(cmd.name) && next.asset->values.size() >= io::project::kMaxShaderNames))
                        return serviceError(ErrorCode::ResourceExhausted, "Shader parameter limit exceeded.");
                    next.asset->pending.erase(cmd.name);
                    next.asset->values[cmd.name] = cmd.value;
                }
            }
            if (next.asset && (next.asset->name.empty() || next.asset->name.size() > io::project::kMaxShaderNames || !io::project::chapter_binary::valid_utf8(next.asset->name)))
                return serviceError(ErrorCode::InvalidArgument, "A shader needs a name.");
            if constexpr (std::is_same_v<T, SetShaderSource> || std::is_same_v<T, RunShader> || std::is_same_v<T, RenameShader>)
                if (const auto old = assets_.find(id); old == assets_.end() || old->second.name != next.asset->name ||
                                                       old->second.language != next.asset->language || old->second.source != next.asset->source)
                    next.asset->cacheSize();
            if (next.asset && !assets_.contains(id) && assets_.size() >= io::project::kMaxShaderAssets)
                return serviceError(ErrorCode::ResourceExhausted, "Shader library limit exceeded or source is not UTF-8.");
            auto bindings = bindings_;
            if (slot)
                bindings.set(*slot, next.binding);
            if (!libraryFitsLocked(id, next.asset ? &*next.asset : nullptr, &bindings))
                return serviceError(ErrorCode::ResourceExhausted, "Shader chapter size limit exceeded.");
            return next;
        },
                          command);
    }

    ShaderService::ExportLease ShaderService::prepareExport(const std::span<const std::string> hosts) {
        acceptResults();
        const std::lock_guard lock(mutex_);
        ExportLease lease{.changes = changes_, .origin = generation()};
        if (!desired_)
            return lease;
        auto build = std::make_shared<Build>();
        static_cast<CompositionContent&>(*build) = *desired_;
        build->frozen = layOut(build->members, &ShaderMember::asset, [](std::size_t) { return rendering::GaussianCustomEffectState::Parameters{}; }, bindings_);
        for (const auto& member : build->members)
            build->values.emplace_back(member.asset, assets_.at(member.asset).values);
        lease.lifetime = std::make_shared<int>(0);
        build->owner = lease.lifetime;
        lease.request = build;
        requests_.push_back(build);
        if (desired_->programs && !desired_->waiting() && !desired_->anyFailed()) {
            build->programs = desired_->programs;
            build->schemas = desired_->schemas;
            for (const auto& host : build->programs->hosts)
                build->hosts.push_back({host->label, Build::HostState::Linked, host});
        }
        for (const auto& host : hosts)
            queueHostLocked(*build, host);
        return lease;
    }

    Result<SnapshotPtr> ShaderService::exportSnapshot(const ExportLease& lease, const std::span<const std::string> hosts) {
        acceptResults();
        const std::lock_guard lock(mutex_);
        if (lease.ended() || lease.changes != changes_ || (lease.request && lease.request->epoch != project_token_[0]))
            return serviceError(ErrorCode::Cancelled, "The export was cancelled.");
        if (!lease.request)
            return SnapshotPtr{};
        auto& build = *lease.request;
        for (const auto& host : hosts)
            queueHostLocked(build, host);
        if (build.anyFailed()) {
            const auto error = std::ranges::find(build.diagnostics.entries, DiagnosticSeverity::Error, &Diagnostic::severity);
            if (error == build.diagnostics.entries.end())
                return serviceError(ErrorCode::FailedPrecondition, build.diagnostics.raw);
            std::string file = error->file;
            for (const auto& member : build.members)
                if (const auto asset = assets_.find(member.asset); asset != assets_.end() && file == effectFile(member.asset))
                    file = asset->second.name;
            return serviceError(ErrorCode::FailedPrecondition, std::format("{}({}:{}): {}", file, error->line, error->column, error->message));
        }
        if (build.waiting() || !build.programs)
            return SnapshotPtr{};
        if (lease.snapshot && lease.snapshot->programs == build.programs)
            return lease.snapshot;
        auto snapshot = std::make_shared<ShaderFrameSnapshot>(*build.frozen);
        snapshot->epoch = build.epoch;
        snapshot->revision = build.revision;
        snapshot->content = lease.request;
        snapshot->programs = build.programs;
        for (auto& binding : snapshot->bindings)
            binding.values = parameterBlock(build.schemas[binding.effect], build.values[binding.effect].second);
        snapshot->parameters = std::make_shared<const decltype(snapshot->bindings)>(snapshot->bindings);
        return SnapshotPtr(std::move(snapshot));
    }

} // namespace lfs::vis::shaders

#ifdef LFS_USER_SHADER_MEASURE
#include "user_shader_measure.hpp"
namespace lfs::vis::shaders {
    // harness adapter: restore a fresh, unbound asset under a previous project's UUID.
    void UserShaderMeasureAccess::replaceId(ShaderService& service, const core::Uuid& from, const core::Uuid& to) {
        const std::lock_guard lock(service.mutex_);
        if (service.bindings_.binds(from) || service.assets_.contains(to))
            throw std::runtime_error("measurement UUID replacement requires an unbound, fresh asset");
        auto node = service.assets_.extract(from);
        if (node.empty())
            throw std::runtime_error("measurement asset missing");
        node.key() = to;
        service.assets_.insert(std::move(node));
    }
} // namespace lfs::vis::shaders
#endif

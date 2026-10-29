/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/export.hpp"
#include "core/uuid.hpp"
#include "io/session_chapters.hpp"
#include "rendering/rendering.hpp"
#include "shaders/slang_compiler.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace lfs::core {
    class Scene;
} // namespace lfs::core

namespace lfs::vis {
    class SequencerController;
}

namespace lfs::vis {
    struct ShaderCommands;
}

namespace lfs::vis::shaders {

    // At most this many assets fill the binding slots at once; a program set
    // runs each of them.
    inline constexpr std::size_t kMaxBoundShaders = 8;
    inline constexpr std::size_t kMaxShaderSourceBytes = io::project::kMaxShaderSourceBytes;

    // One linked projection host.
    struct HostProgram {
        std::string label;         // host module name, e.g. "projection_forward_custom"
        std::vector<int> bindings; // storage buffers of descriptor set 0
        std::array<std::uint32_t, 3> workgroup{};
        std::uint32_t subgroup_size = 0;
        std::vector<std::uint32_t> spirv;
    };

    struct LinkedHost {
        HostProgram program;
        std::vector<std::vector<Parameter>> schemas; // reflected, per effect
    };

    // Links effect sources into package hosts. ShaderService creates it on its
    // compiler worker and calls it only there.
    class HostLinker {
    public:
        virtual ~HostLinker() = default;
        [[nodiscard]] virtual Result<LinkedHost> link(std::span<const EffectSource> effects, const std::string& host,
                                                      CompileDiagnostics& diagnostics) = 0;
    };
    using HostLinkerFactory = std::function<std::unique_ptr<HostLinker>()>;

    // Links through SlangCompiler, which loads the package at `manifest` on
    // its first link.
    [[nodiscard]] LFS_VIS_API HostLinkerFactory slangHostLinker(std::filesystem::path manifest);

    // Immutable executable content shared by binding-only publications.
    struct ShaderPrograms {
        struct Effect {
            core::Uuid asset;
            std::uint64_t revision = 0;
            std::vector<Parameter> schema; // its parameter block follows it
        };
        std::vector<Effect> effects;
        std::vector<std::shared_ptr<const HostProgram>> hosts;
    };

    // A binding slot: the global slot, the reveal slot or a splat node's own.
    struct ShaderSlot {
        enum class Kind : std::uint8_t { Global,
                                         Reveal,
                                         Node };
        Kind kind = Kind::Global;
        core::Uuid node; // Kind::Node only
        bool operator==(const ShaderSlot&) const = default;
    };

    struct ShaderBinding {
        ShaderSlot target;
        core::Uuid asset;
        bool operator==(const ShaderBinding&) const = default;
    };

    // One complete publication: executable identity, bindings and matching parameters.
    struct CompositionContent;
    struct ShaderFrameSnapshot {
        std::uint64_t revision = 0, epoch = 0;
        std::shared_ptr<const CompositionContent> content;
        std::vector<ShaderBinding> slots;
        std::shared_ptr<const ShaderPrograms> programs;
        static constexpr std::uint32_t kNone = rendering::GaussianCustomEffectState::kNone;
        std::vector<rendering::GaussianCustomEffectState::Binding> bindings;
        std::shared_ptr<const std::vector<rendering::GaussianCustomEffectState::Binding>> parameters;
        std::uint32_t global = kNone; // binding of the global slot
        std::uint32_t reveal = kNone;
        std::optional<core::Uuid> reveal_asset;
        std::unordered_map<core::Uuid, std::uint32_t> nodes; // splat nodes and the binding of their slot
    };

    struct CustomRevealContext {
        core::Uuid shader_id;
        float progress = 0.0f;
        float softness = 0.1f;
        bool operator==(const CustomRevealContext&) const = default;
    };

    // Draw-local object ids and original visible bounds, in model-slot order.
    struct EffectScene {
        struct Object {
            core::Uuid node; // nil for a slot without a visible splat node
            std::uint32_t id = 0;
        };
        std::vector<Object> objects; // per model slot
        bool bounded = false;        // visible geometry has bounds
        glm::vec3 bounds_min{0.0f};  // the origin without them
        glm::vec3 bounds_max{0.0f};
    };
    [[nodiscard]] LFS_VIS_API EffectScene captureEffectScene(const core::Scene& scene, std::size_t slots);

    // Apply the snapshot only to visible slots and the matching active reveal.
    [[nodiscard]] LFS_VIS_API std::optional<rendering::GaussianCustomEffectState>
    makeCustomEffectState(const ShaderFrameSnapshot& snapshot, const EffectScene& scene, float time,
                          const std::optional<CustomRevealContext>& reveal = {});

    using ShaderProjectToken = std::array<std::uint64_t, 2>;
    struct ShaderProjectState {
        io::project::ShaderLibraryChapter chapter;
        ShaderProjectToken token;
    };

    struct ShaderAsset {
        core::Uuid id;
        std::string name;
        std::string source;
        std::uint64_t revision = 0;
    };

    enum class ShaderState : std::uint8_t {
        Unbound, // no slot binds the asset
        Pending, // bound; draws do not run its revision yet
        Applied, // bound; draws run its revision
        Blocked, // bound; another asset of its composition fails, so an earlier composition or the built-in programs draw
        Failed,  // bound; its revision cannot draw, so an earlier one or the built-in programs do
    };

    struct ShaderStatus {
        ShaderState state = ShaderState::Unbound;
        std::uint64_t revision = 0;           // of the current source
        std::uint64_t displayed_revision = 0; // the revision draws run; 0 for none
        bool compiling = false;               // the worker links a host of the revision
        std::vector<std::string> pending;     // hosts of the revision still to link
        std::vector<std::string> failed;      // hosts of the revision that cannot be built
        std::vector<core::Uuid> blocked_by;   // Blocked: the assets whose failure blocks it
        // Of its latest failure, else its latest link: the entries of its own
        // file and of none; `raw` holds the whole composition's output.
        CompileDiagnostics diagnostics;
        std::vector<Parameter> parameters; // schema of the latest linked revision
        std::vector<glm::vec4> values;     // per parameter
        std::uint64_t schema_revision = 0;
        std::uint64_t diagnostics_revision = 0;
        std::uint64_t links = 0; // hosts the worker linked or tried to link with it
    };

    [[nodiscard]] LFS_VIS_API std::string_view shaderOutcome(const ShaderStatus&, std::uint64_t revision, bool applied);

    struct ShaderMember {
        core::Uuid asset;
        std::uint64_t revision = 0;
        bool operator==(const ShaderMember&) const = default;
    };

    struct CompositionContent {
        std::uint64_t epoch = 0, revision = 0;
        std::vector<ShaderMember> members;
        std::vector<EffectSource> effects;
    };
    using SnapshotPtr = std::shared_ptr<const ShaderFrameSnapshot>;
    struct DisplayRecord {
        SnapshotPtr snapshot;
        bool custom = false;
        rendering::ProgramSetHandle programs;
        bool complete = true;
    };

    struct ShaderWork {
        std::vector<ShaderMember> members;
        std::string host;
        bool compiling = false;
    };

    struct ShaderAssetView {
        ShaderAsset asset;
        ShaderStatus status;
    };

    struct ShaderInspection {
        std::vector<ShaderAssetView> assets;
        std::vector<ShaderBinding> bindings, displayed_bindings;
        std::vector<ShaderWork> work;
    };

    struct ShaderEvent {
        enum class Kind { Started,
                          Completed,
                          Failed,
                          BindingsChanged,
                          Applied };
        Kind kind;
        std::vector<ShaderMember> members;
        std::string host;
        CompileDiagnostics diagnostics;
        std::vector<ShaderBinding> bindings;
    };

    struct SetShaderSource {
        std::optional<core::Uuid> asset;
        std::string source;
        std::optional<std::string> name;
        std::optional<std::uint64_t> expected_revision;
    };

    struct RunShader {
        std::optional<core::Uuid> asset;
        std::string source;
        ShaderSlot target;
    };

    struct RenameShader {
        core::Uuid asset;
        std::string name;
    };

    struct BindShader {
        ShaderSlot target;
        core::Uuid asset;
    };

    struct UnbindShader {
        ShaderSlot target;
        std::optional<core::Uuid> asset;
    };

    struct SetShaderParameter {
        core::Uuid asset;
        std::string name;
        std::vector<float> value;
        std::optional<std::uint64_t> expected_revision;
    };

    struct RemoveShader {
        core::Uuid asset;
    };
    using ShaderCommand = std::variant<SetShaderSource, RunShader, RenameShader, BindShader, UnbindShader,
                                       SetShaderParameter, RemoveShader>;

    // Waiters retain this signal independently of the service and renderer.
    struct LFS_VIS_API ShaderChanges {
        static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
        std::atomic<std::uint64_t> generation{1};
        mutable std::mutex mutex;
        std::condition_variable_any changed;
        bool closed = false;
        std::uint64_t cancelled_at = 0;
        void notify(bool close = false, bool cancel = false);
        [[nodiscard]] bool cancelled(std::uint64_t origin) const;
        bool wait(std::uint64_t observed, std::chrono::steady_clock::time_point deadline, std::stop_token stop = {});
    };

    class LFS_VIS_API ShaderService {
    public:
        struct Build;
        // The previews a lease's draws render: its plan links exactly their hosts.
        struct ExportFrame {
            int width = 0;
            int height = 0;
            bool depth = false; // a depth capture
            bool bands = false; // also banded splat depth under composited meshes
        };
        struct ExportLease {
            mutable SnapshotPtr snapshot;
            mutable rendering::ProgramSetHandle programs;
            std::shared_ptr<Build> request;
            std::shared_ptr<int> lifetime;
            std::shared_ptr<ShaderChanges> changes;
            std::uint64_t origin = 0;
            bool live_request = false;
            bool per_frame = false;
            ExportFrame frame;
            std::optional<Error> error;
            // Hosts a draw needed beyond its plan; the next preflight links them.
            mutable std::vector<std::string> missing;
            // Its build links hosts its draws found missing; a frameless lease waits for them.
            mutable bool discovered = false;
            [[nodiscard]] bool ended() const { return changes && changes->cancelled(origin); }
        };
        [[nodiscard]] std::uint64_t generation() const { return changes_->generation.load(std::memory_order_acquire); }
        [[nodiscard]] std::shared_ptr<ShaderChanges> changes() const { return changes_; }
        // Viewer-side acceptance of tagged worker messages, before external callbacks.
        void acceptResults();
        void setRevealController(SequencerController* controller) { reveal_controller_ = controller; }
        [[nodiscard]] Status validateRevealTarget() const;
        [[nodiscard]] Status validateBinding(const ShaderSlot& slot, const core::Uuid& asset) const;
        [[nodiscard]] ShaderInspection inspect(bool include_source = true) const;
        [[nodiscard]] Result<ShaderInspection> asset(const core::Uuid&, bool include_source = false) const;
        [[nodiscard]] ShaderInspection work() const;
        [[nodiscard]] bool hasApplicableBindings(const std::optional<CustomRevealContext>& reveal) const;
        [[nodiscard]] bool hasBindings() const { return has_bindings_.load(std::memory_order_acquire); }
        [[nodiscard]] Result<core::Uuid> command(const ShaderCommand& command,
                                                 std::optional<ShaderSlot> redo_binding = {});

        // `wake` is called from any thread, outside the service lock.
        ShaderService(const core::Scene& scene, HostLinkerFactory linker, std::function<void()> wake);
        ~ShaderService();
        ShaderService(const ShaderService&) = delete;
        ShaderService& operator=(const ShaderService&) = delete;

        [[nodiscard]] Status validateTarget(const ShaderSlot& slot) const;
        [[nodiscard]] std::vector<ShaderAsset> assets(bool include_source = true) const;
        // Include undo-held node bindings when observing document Stop targets.
        [[nodiscard]] std::optional<core::Uuid> boundAsset(const ShaderSlot& slot, bool include_detached = false) const;
        [[nodiscard]] bool dirty() const;
        [[nodiscard]] Result<ShaderProjectState> captureProject(const io::project::ShaderLibraryChapter& retained,
                                                                std::span<const core::Uuid> omitted = {}) const;
        [[nodiscard]] std::vector<std::string> restoreProject(const io::project::ShaderLibraryChapter& chapter,
                                                              std::optional<core::Uuid> reveal);
        void clearDirtyIfUnchanged(ShaderProjectToken token);
        [[nodiscard]] ShaderProjectToken projectToken() const;
        [[nodiscard]] Result<ShaderStatus> status(const core::Uuid& asset) const;

        // Project switch: drops every asset and binding. A running link ends
        // on its own and its result is dropped.
        void clear();
        // Viewer shutdown: export waits end as cancelled and nothing links anymore.
        void stop();

        [[nodiscard]] bool onOwnerThread() const { return std::this_thread::get_id() == owner_; }
        // Complete CPU publications; GPU readiness and presentation belong to the renderer.
        void demandLive(std::span<const std::string> hosts, SnapshotPtr retained = {});
        [[nodiscard]] SnapshotPtr candidate() const;
        [[nodiscard]] SnapshotPtr restrictFallback(SnapshotPtr retained) const;
        void observeDisplay(std::shared_ptr<const DisplayRecord> display);
        void failCandidate(SnapshotPtr snapshot, const Error& error);
        void deviceReset();
        [[nodiscard]] ExportLease prepareExport(std::span<const std::string> hosts);
        [[nodiscard]] Result<SnapshotPtr> exportSnapshot(const ExportLease& lease, std::span<const std::string> hosts = {});

    private:
        friend struct UserShaderMeasureAccess;
        friend struct lfs::vis::ShaderCommands;
        Result<core::Uuid> applyCommand(const ShaderCommand&, std::optional<ShaderSlot>, bool authoring);
        SequencerController* reveal_controller_ = nullptr;
        friend class lfs::vis::SequencerController;
        [[nodiscard]] Status setRevealBinding(std::optional<core::Uuid>);
        struct AssetContent;
        struct Asset;
        struct EditState;
        class UndoEdit;
        [[nodiscard]] Result<EditState> prepareEdit(const ShaderCommand&) const;
        [[nodiscard]] Status installEdit(const EditState&, const EditState&);
        void installContentLocked(core::Uuid, AssetContent);
        [[nodiscard]] EditState editStateLocked(const core::Uuid&, const std::optional<ShaderSlot>&) const;
        [[nodiscard]] Status restoreEdit(const EditState& expected, const EditState& desired);
        [[nodiscard]] Status validateBindingLocked(const ShaderSlot&, const core::Uuid&) const;
        [[nodiscard]] Result<ShaderStatus> statusLocked(const core::Uuid&) const;
        [[nodiscard]] std::vector<ShaderBinding> bindingsLocked() const;
        void bindingsChanged() const;

        struct WorkerMessage;
        ShaderInspection inspectLocked(std::optional<core::Uuid>, bool include_source, bool include_assets) const;
        bool currentLocked(const WorkerMessage&) const;
        struct Bindings {
            std::optional<core::Uuid> global;
            std::optional<core::Uuid> reveal;
            std::unordered_map<core::Uuid, core::Uuid> nodes; // node -> asset
            // The distinct bound assets in composition order.
            [[nodiscard]] std::vector<core::Uuid> assets() const;
            [[nodiscard]] bool binds(const core::Uuid& asset) const;
            [[nodiscard]] std::optional<core::Uuid> get(const ShaderSlot&) const;
            void set(const ShaderSlot&, std::optional<core::Uuid>);
        };

        void runWorker();
        // The build with a queued host the worker links next.
        [[nodiscard]] std::shared_ptr<Build> nextBuild() const;
        void queueHostLocked(Build& build, const std::string& host);
        // Moves bindings of removed nodes aside and restores those of nodes back again.
        void reconcileNodes();
        // Follows the bindings to a build of the bound assets' current revisions.
        void retargetLocked();
        void refreshSnapshotLocked();
        void wake() const;

        const core::Scene& scene_;
        const HostLinkerFactory linker_;
        const std::function<void()> wake_;
        const std::thread::id owner_;

        std::shared_ptr<ShaderChanges> changes_ = std::make_shared<ShaderChanges>();
        mutable std::mutex mutex_;
        std::vector<WorkerMessage> messages_;
        bool accepting_results_ = false;
        std::atomic<bool> has_bindings_{false};
        // Compiler work queue signal.
        mutable std::condition_variable_any changed_;
        std::thread worker_;
        bool stopping_ = false;

        std::unordered_map<core::Uuid, Asset> assets_;
        Bindings bindings_;
        bool dirty_ = false;
        ShaderProjectToken project_token_{1, 0};
        void markDirtyLocked() {
            dirty_ = true;
            ++project_token_[1];
            changes_->notify();
        }
        [[nodiscard]] bool libraryFitsLocked(core::Uuid id, const AssetContent* replacement, const Bindings* bindings = nullptr) const;
        [[nodiscard]] bool sourceFitsLocked(std::string_view source, std::optional<core::Uuid> replacing = {}) const;
        // Bindings of removed nodes, back when an undo restores the node.
        std::unordered_map<core::Uuid, core::Uuid> detached_;
        std::vector<std::string> demanded_;
        std::vector<std::weak_ptr<Build>> requests_;
        std::shared_ptr<Build> retained_;
        std::shared_ptr<const Build> linked_; // the last complete live composition's linked hosts, without a program set
        std::optional<ShaderWork> linking_;
        std::weak_ptr<const DisplayRecord> display_;
        std::shared_ptr<void> history_lifetime_ = std::make_shared<int>(0);
        std::unordered_map<core::Uuid, std::uint64_t> removed_revisions_;
        std::uint64_t composition_revision_ = 0;
        std::shared_ptr<Build> desired_; // the bound assets' current revisions
        SnapshotPtr snapshot_;
        event::ScopedHandler events_;
    };

} // namespace lfs::vis::shaders

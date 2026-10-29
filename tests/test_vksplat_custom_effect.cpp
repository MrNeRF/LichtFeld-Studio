/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define LFS_USER_SHADER_CUSTOM_FIXTURE
#include "user_shader_test_support.hpp"
namespace lfs::test::custom_effect {
    TEST_F(CustomEffectTest, IdentityRendersLikeTheBuiltInPrograms) {
        publish(kIdentity);
        for (const auto encoding : kEncodings) {
            const auto scene = model(arrays_, encoding);
            for (const auto path : kPaths) {
                SCOPED_TRACE(std::format("{} {}", name(encoding), name(path)));
                auto r = request(path);
                const auto expected = draw(scene, r, path);
                EXPECT_FALSE(expected.custom);
                EXPECT_EQ(Access::encodedSh(*viewport_),
                          (std::array{encoding == Encoding::Float16, encoding == Encoding::Q16}));
                for (const auto& state : {perNode({}), global()}) {
                    r.custom_effect = state;
                    const auto custom = draw(scene, r, path);
                    expectSame(custom, expected);
                    EXPECT_TRUE(custom.custom);
                    EXPECT_FALSE(custom.missing);
                    EXPECT_EQ(custom.geometry, gut(path));
                }
                // A custom draw that binds no element runs the hosts unchanged too.
                r.custom_effect = GaussianCustomEffectState{};
                expectSame(draw(scene, r, path), expected);
                // The default draw after it drops the derived geometry. Legacy
                // draws free their scratch anyway, so only the others show it.
                r.custom_effect.reset();
                if (path != Path::Legacy)
                    EXPECT_FALSE(draw(scene, r, path).geometry);
            }
        }
    }

    TEST_F(CustomEffectTest, PlannedHostsMatchDispatchedChains) {
        publish(kIdentity);
        const auto scene = model(arrays_);
        struct Row {
            const char* name;
            bool warm, capture, training, deterministic, preview, guides, depth;
            int transparency;
        };
        const Row rows[] = {
            {"warmup", true, false, false, false, false, false, false, 0},
            {"settled", false, false, false, false, false, false, false, 0},
            {"capture", false, true, false, false, true, false, false, 0},
            {"training", false, false, true, false, false, false, false, 0},
            {"deterministic preview", true, false, false, true, true, false, false, 2},
            {"stochastic", false, false, false, false, false, false, false, 1},
            {"guides", false, false, false, false, false, true, false, 2},
            {"depth", false, false, false, false, false, false, true, 2},
        };
        for (const auto row : rows) {
            SCOPED_TRACE(row.name);
            auto r = request(Path::Survivors);
            r.custom_effect = global();
            r.deterministic_export = row.deterministic;
            r.transparency_mode = static_cast<lfs::rendering::GaussianTransparencyMode>(row.transparency);
            r.overlay.markers.show_center_markers = row.guides;
            r.depth_view = row.depth;
            const auto slot = row.preview ? Slot::Preview : Slot::Main;
            Access::warmup(*viewport_, row.warm);
            viewport_->setDepthCaptureMode(row.capture);
            const auto planned = viewport_->prepare(*context_, scene, r, slot, row.training);
            ASSERT_TRUE(planned) << planned.error();
            const bool stochastic = row.transparency != 0 && !row.capture && !row.deterministic && !row.preview && !row.guides && !row.depth;
            EXPECT_EQ(planned->stochastic, stochastic);
            EXPECT_EQ(planned->inputs.survivors, !stochastic && !row.capture && !row.training &&
                                                     Access::renderer(*viewport_).supportsFloat16Storage() && (!row.warm || row.deterministic));
            const auto drawn = render(scene, r, slot, false, row.training);
            EXPECT_FALSE(drawn.missing);
            EXPECT_EQ(Access::renderer(*viewport_).planned_custom_projection_, planned->hosts.programs[0]);
        }
        viewport_->setDepthCaptureMode(false);
    }

    TEST_F(CustomEffectTest, LinkedBuiltInRevealMatchesTheRevealPrograms) {
        publish(kIdentity);
        const auto scene = model(arrays_);
        const auto plain = draw(scene, request(Path::Legacy), Path::Legacy);
        std::size_t changed = 0;
        for (int effect = 0; effect < 11; ++effect) {
            for (const float progress : {0.0f, 0.45f, 0.95f, 1.0f}) {
                for (const auto path : {Path::Legacy, Path::Survivors, Path::Gut, Path::BandedGut}) {
                    SCOPED_TRACE(std::format("reveal {} at {} {}", effect, progress, name(path)));
                    auto r = request(path);
                    r.reveal = lfs::rendering::GaussianRevealState{
                        .progress = progress,
                        .softness = 0.2f,
                        .center_x = 0.1f,
                        .center_z = -4.4f,
                        .radius = 2.0f,
                        .effect = effect,
                        .min_y = -1.0f,
                        .max_y = 1.0f};
                    const auto expected = draw(scene, r, path);
                    changed += path == Path::Legacy && progress == 0.45f &&
                               compareImages(expected.pixels, plain.pixels).max > 0.05f;
                    r.custom_effect = global();
                    expectSame(draw(scene, r, path), expected);
                }
            }
        }
        EXPECT_GE(changed, 10u) << "most reveal effects must change the image mid-way";
    }

    // The global binding runs before a node's own, each with its parameters:
    // the global binding recolors, then node 1 moves by the red it sees. The
    // edits do not commute, so the order shows.
    TEST_F(CustomEffectTest, GlobalNodeRevealOrderMatchesIndependentOracle) {
        publish(kEdit);
        const glm::vec3 color(0.9f, 0.2f, 0.1f), per_red(0.0f, 0.3f, 0.0f);
        auto state = perNode({});
        state.bindings[1].values[OffsetPerRed] = glm::vec4(per_red, 0.0f);
        state.bindings.push_back({});
        state.bindings.back().values[Color] = glm::vec4(color, 1.0f);
        state.global_binding = static_cast<std::uint32_t>(state.bindings.size() - 1);
        Arrays edited = arrays_;
        for (std::size_t i = 0; i < edited.size(); ++i) {
            for (int c = 0; c < 3; ++c)
                edited.sh0[3 * i + c] = (color[c] - 0.5f) / kC0;
            if (edited.nodes[i] == 1)
                for (int axis = 0; axis < 3; ++axis)
                    edited.means[3 * i + axis] += per_red[axis] * color.r;
        }
        expectEdited(edited, state);
        {
            Parameters recolor{}, move{};
            recolor[Color] = glm::vec4(0.9f, 0.2f, 0.1f, 1.0f);
            move[OffsetPerRed] = glm::vec4(0.0f, 0.3f, 0.0f, 0.0f);
            // Global recolor, then the move as every node's binding ...
            GaussianCustomEffectState nodes = global();
            nodes.bindings = {{recolor}, {move}};
            nodes.global_binding = 0;
            for (auto& object : nodes.objects)
                object.binding = 1;
            // ... or as the reveal binding.
            GaussianCustomEffectState reveal = global();
            reveal.bindings = {{recolor}, {move}};
            reveal.global_binding = 0;
            reveal.reveal_binding = 1;
            auto reversed = reveal;
            reversed.global_binding = 1;
            reversed.reveal_binding = 0;
            auto without = reveal;
            without.reveal_binding = GaussianCustomEffectState::kNone;
            const auto scene = model(arrays_);
            for (const auto path : {Path::Legacy, Path::Survivors, Path::Gut}) {
                SCOPED_TRACE(name(path));
                auto r = request(path);
                r.sh_degree = 0;
                r.reveal = lfs::rendering::GaussianRevealState{.progress = 0.6f, .softness = 0.2f, .center_x = 0.1f, .center_z = -4.4f, .radius = 2.5f, .effect = 1, .min_y = -1.0f, .max_y = 1.0f};
                r.custom_effect = nodes;
                const auto expected = draw(scene, r, path);
                r.custom_effect = reveal;
                expectSame(draw(scene, r, path), expected);
                r.custom_effect = reversed;
                EXPECT_GT(compareImages(draw(scene, r, path).pixels, expected.pixels).max, 0.05f) << "order must show";
                r.reveal.reset();
                r.custom_effect = without;
                const auto recolored = draw(scene, r, path);
                r.custom_effect = nodes;
                const auto custom_only = draw(scene, r, path);
                r.custom_effect = reveal;
                expectSame(draw(scene, r, path), custom_only);
                EXPECT_GT(compareImages(custom_only.pixels, recolored.pixels).max, 0.05f);
            }
        }
    }

    // While a reveal is set, the reveal binding runs after the node's own; it
    // does not run without one.

    TEST_F(CustomEffectTest, EditsMatchTheEditedScene) {
        publish(kEdit);
        std::array<Parameters, kNodes> values{};
        const glm::vec3 offset(0.25f, -0.15f, 0.0f), color(0.9f, 0.2f, 0.1f), scale(0.06f, 0.025f, 0.04f);
        const glm::vec4 rotation = glm::normalize(glm::vec4(0.8f, 0.1f, -0.5f, 0.3f));
        values[0][Offset] = glm::vec4(offset, 0.0f);
        values[0][Scale] = glm::vec4(scale, 1.0f);
        values[1][Color] = glm::vec4(color, 1.0f);
        values[1][Rotation] = rotation;
        // Node 2 has a rotated, nonuniformly scaled transform; scale and
        // rotation are source-space.
        const glm::vec3 scale2(0.03f, 0.05f, 0.02f);
        const glm::vec4 rotation2 = glm::normalize(glm::vec4(0.6f, -0.2f, 0.7f, 0.1f));
        values[2][Opacity] = glm::vec4(0.5f, 0.0f, 0.0f, 1.0f);
        values[2][Scale] = glm::vec4(scale2, 1.0f);
        values[2][Rotation] = rotation2;
        Arrays edited = arrays_;
        for (std::size_t i = 0; i < edited.size(); ++i) {
            if (edited.nodes[i] == 0) {
                for (int axis = 0; axis < 3; ++axis) {
                    edited.means[3 * i + axis] += offset[axis];
                    edited.scales[3 * i + axis] = std::log(scale[axis]);
                }
            } else if (edited.nodes[i] == 1) {
                for (int c = 0; c < 3; ++c)
                    edited.sh0[3 * i + c] = (color[c] - 0.5f) / kC0;
                for (int k = 0; k < 4; ++k)
                    edited.rotations[4 * i + k] = rotation[k];
            } else {
                edited.opacities[i] = 0.0f;
                for (int axis = 0; axis < 3; ++axis)
                    edited.scales[3 * i + axis] = std::log(scale2[axis]);
                for (int k = 0; k < 4; ++k)
                    edited.rotations[4 * i + k] = rotation2[k];
            }
        }
        for (const auto encoding : kEncodings) {
            const auto scene = model(arrays_, encoding);
            const auto oracle = model(edited, encoding);
            for (const auto path : kPaths) {
                SCOPED_TRACE(std::format("{} {}", name(encoding), name(path)));
                auto r = request(path);
                r.sh_degree = 0; // the edited scene evaluates SH at the moved centers
                const auto unedited = draw(scene, r, path);
                const auto expected = draw(oracle, r, path);
                r.custom_effect = perNode(values);
                expectClose(draw(scene, r, path), expected, unedited);
            }
        }
    }

    TEST_F(CustomEffectTest, ElementMovedIntoViewSurvivesCull) {
        publish(kEdit);
        constexpr float far = 40.0f;
        Arrays outside = arrays_;
        for (std::size_t i = 0; i < outside.size(); ++i)
            if (outside.nodes[i] == 0)
                outside.means[3 * i] += far;
        std::array<Parameters, kNodes> values{};
        values[0][Offset] = glm::vec4(-far, 0.0f, 0.0f, 0.0f);
        const auto scene = model(outside);
        auto r = request(Path::Survivors);
        r.sh_degree = 0; // SH is evaluated toward the stored center
        const auto culled = draw(scene, r, Path::Survivors);
        const auto expected = draw(model(arrays_), r, Path::Survivors);
        r.custom_effect = perNode(values);
        expectClose(draw(scene, r, Path::Survivors), expected, culled);
    }

    TEST_F(CustomEffectTest, ColorDrivenMovementUsesTheColorTheElementShows) {
        publish(kEdit);
        // Moves by the shaded red, then sets a new color: the move must use the
        // element's own color, not one evaluated again after it moved.
        std::array<Parameters, kNodes> values{};
        const glm::vec3 per_red(0.0f, 0.4f, 0.0f), color(0.2f, 0.7f, 0.9f);
        for (auto& node : values) {
            node[OffsetPerRed] = glm::vec4(per_red, 0.0f);
            node[Color] = glm::vec4(color, 1.0f);
        }
        Arrays edited = arrays_;
        for (std::size_t i = 0; i < edited.size(); ++i) {
            const float red = std::max(kC0 * arrays_.sh0[3 * i] + 0.5f, 0.0f);
            for (int axis = 0; axis < 3; ++axis)
                edited.means[3 * i + axis] += per_red[axis] * red;
            for (int c = 0; c < 3; ++c)
                edited.sh0[3 * i + c] = (color[c] - 0.5f) / kC0;
        }
        // Moving in world space equals moving the stored centers only for nodes
        // whose transform has an identity linear part.
        transforms_[2] = glm::translate(glm::mat4(1.0f), glm::vec3(-0.1f, 0.05f, 0.2f));
        expectEdited(edited, perNode(values));
    }

    TEST_F(CustomEffectTest, GutDrawsUseMovedGeometryAtLodSlots) {
        publish(kEdit);
        // Every other element in reverse order, with logical ids of their
        // neighbours, so slot, physical and logical indices all differ, with
        // transition weights below one.
        std::vector<std::uint32_t> indices, logical;
        std::vector<float> weights;
        for (std::size_t i = arrays_.size(); i-- > 0;) {
            if (i % 2 == 0) {
                indices.push_back(static_cast<std::uint32_t>(i));
                logical.push_back(static_cast<std::uint32_t>(i ^ 1u));
                weights.push_back(0.4f + 0.6f * static_cast<float>(i % 7) / 6.0f);
            }
        }
        std::array<Parameters, kNodes> values{};
        const glm::vec3 offset(-0.2f, 0.1f, 0.05f);
        const glm::vec4 rotation = glm::normalize(glm::vec4(0.7f, -0.3f, 0.4f, 0.2f));
        values[0][Offset] = glm::vec4(offset, 0.0f);
        values[0][Rotation] = rotation;
        values[0][Scale] = glm::vec4(0.07f, 0.02f, 0.03f, 1.0f);
        // The same reshape on node 2's rotated, nonuniformly scaled transform.
        values[2][Rotation] = rotation;
        values[2][Scale] = values[0][Scale];
        Arrays edited = arrays_;
        for (std::size_t i = 0; i < edited.size(); ++i) {
            if (edited.nodes[i] == 1)
                continue;
            if (edited.nodes[i] == 0)
                for (int axis = 0; axis < 3; ++axis)
                    edited.means[3 * i + axis] += offset[axis];
            for (int k = 0; k < 4; ++k)
                edited.rotations[4 * i + k] = rotation[k];
            edited.scales[3 * i] = std::log(0.07f);
            edited.scales[3 * i + 1] = std::log(0.02f);
            edited.scales[3 * i + 2] = std::log(0.03f);
        }
        const auto scene = model(arrays_);
        const auto oracle = model(edited);
        for (const auto path : {Path::Gut, Path::GutOutline}) {
            SCOPED_TRACE(name(path));
            auto r = request(path);
            r.sh_degree = 0;
            r.lod_indices = indices.data();
            r.lod_logical_indices = logical.data();
            r.lod_weights = weights.data();
            r.lod_count = indices.size();
            const auto unedited = draw(scene, r, path);
            const auto expected = draw(oracle, r, path);
            r.custom_effect = global();
            expectSame(draw(scene, r, path), unedited);
            r.custom_effect = perNode(values);
            expectClose(draw(scene, r, path), expected, unedited);
        }
    }

    TEST_F(CustomEffectTest, NonFiniteElementsAreDropped) {
        publish(kEdit);
        std::array<Parameters, kNodes> values{};
        values[1][NonFinite] = glm::vec4(1.0f);
        const auto scene = model(arrays_);
        for (const auto path : {Path::Legacy, Path::Survivors, Path::Gut}) {
            SCOPED_TRACE(name(path));
            auto r = request(path);
            r.scene.node_visibility_mask = {true, false, true};
            const auto expected = draw(scene, r, path);
            r.scene.node_visibility_mask.clear();
            r.custom_effect = perNode(values);
            expectSame(draw(scene, r, path), expected);
        }
    }

    TEST_F(CustomEffectTest, CustomResourcesAreReleasedAfterTheirLastDraw) {
        publish(kIdentity);
        const auto scene = model(arrays_);
        auto r = request(Path::Gut);
        const auto plain = draw(scene, r, Path::Gut);
        r.custom_effect = global();
        EXPECT_TRUE(draw(scene, r, Path::Gut).geometry);
        // A non-GUT custom draw does not keep the geometry of the last GUT
        // one. Survivor draws keep their scratch, unlike live training ones.
        auto survivors = request(Path::Survivors);
        survivors.custom_effect = global();
        const auto non_gut = draw(scene, survivors, Path::Survivors);
        EXPECT_TRUE(non_gut.custom);
        EXPECT_FALSE(non_gut.geometry);
        EXPECT_TRUE(draw(scene, r, Path::Gut).geometry);
        for (const auto path : {Path::Gut, Path::Survivors}) {
            auto r = request(Path::Gut);
            r.custom_effect = global();
            ASSERT_TRUE(draw(scene, r, Path::Gut).geometry);
            Access::renderer(*viewport_).waitForPendingBatch();
            auto plain = request(path);
            EXPECT_FALSE(draw(scene, plain, path).geometry) << name(path);
        }
        EXPECT_TRUE(draw(scene, r, Path::Gut).custom);
        Access::release(*viewport_);
        const auto refused = viewport_->rerenderSelectionOverlay(*context_, scene, r, Slot::Main, false);
        ASSERT_FALSE(refused);
        EXPECT_NE(refused.error().find("custom effect"), std::string::npos) << refused.error();
        // A request for a custom effect without a published set draws the built-in programs.
        const auto after = draw(scene, r, Path::Gut);
        EXPECT_FALSE(after.geometry);
        expectSame(after, plain);
        r.custom_effect.reset();
        EXPECT_FALSE(draw(scene, r, Path::Gut).custom);
        const auto reused = viewport_->rerenderSelectionOverlay(*context_, scene, r, Slot::Main, false);
        EXPECT_TRUE(reused) << reused.error();
        ASSERT_TRUE(context_->deviceWaitIdle());
        EXPECT_EQ(Access::retiredSets(*viewport_), 0u);
    }

    // Dropping the geometry of the last GUT custom draw once every batch
    // completed must not free it inside the batch being recorded.

    // A set without the host of a draw's chain leaves the draw built-in and
    // names the host; the draws it holds run their host.
    TEST_F(CustomEffectTest, DrawsWithoutTheirHostRenderAsDefault) {
        publish(kEdit, {"projection_forward_custom"});
        std::array<Parameters, kNodes> values{};
        values[0][Offset] = glm::vec4(0.3f, 0.0f, 0.0f, 0.0f);
        auto r = request(Path::Legacy);
        const auto q16 = model(arrays_, Encoding::Q16);
        const auto plain_q16 = draw(q16, r, Path::Legacy);
        r.custom_effect = perNode(values);
        const auto missing = draw(q16, r, Path::Legacy);
        expectSame(missing, plain_q16);
        EXPECT_FALSE(missing.custom);
        EXPECT_TRUE(missing.missing);
        const auto fp32 = model(arrays_);
        const auto moved = draw(fp32, r, Path::Legacy);
        EXPECT_TRUE(moved.custom);
        EXPECT_FALSE(moved.missing);
        EXPECT_GT(compareImages(moved.pixels, draw(fp32, request(Path::Legacy), Path::Legacy).pixels).max, 0.05f);
    }

    // The selection overlay pass redraws from the last projection; a custom
    // one, whose hosts may already be retired, is drawn again in full.

    // Stochastic transparency accumulates custom draws as it does default ones
    // and starts over when the program set, time or parameters change: such a
    // frame equals one drawn after an explicit history reset.
    TEST_F(CustomEffectTest, StochasticHistoryRestartsWithTheEffect) {
        using Mode = lfs::rendering::GaussianTransparencyMode;
        const auto scene = model(arrays_);
        constexpr int kFrames = 4;
        std::array<Parameters, kNodes> moving{};
        moving[0][Offset] = glm::vec4(0.1f, 0.0f, 0.0f, 0.0f);
        auto later = global();
        later.time = 1.0f;
        for (const auto mode : {Mode::Stochastic, Mode::SteadySplats}) {
            SCOPED_TRACE(static_cast<int>(mode));
            auto base = request(Path::Legacy);
            base.transparency_mode = mode;
            base.stochastic_max_frames = 64;
            // kFrames draws of `before` on a fresh viewport, then one of `after`.
            const auto run = [&](const std::optional<GaussianCustomEffectState>& before,
                                 const std::optional<GaussianCustomEffectState>& after, const bool republish,
                                 const bool reset) {
                viewport_ = std::make_unique<VksplatViewportRenderer>();
                initialized_ = false;
                publish(kEdit);
                auto r = base;
                r.custom_effect = before;
                for (int frame = 0; frame < kFrames; ++frame)
                    (void)render(scene, r, Slot::Main, frame == 0, false);
                if (republish)
                    publish(kEdit);
                if (reset)
                    viewport_->invalidateStochasticHistory();
                r.custom_effect = after;
                return render(scene, r, Slot::Main, false, false);
            };
            const auto identity = run(perNode({}), perNode({}), false, false);
            EXPECT_TRUE(identity.custom);
            expectSame(identity, run(std::nullopt, std::nullopt, false, false));
            EXPECT_GT(compareImages(identity.pixels, run(perNode({}), perNode({}), false, true).pixels).count, 0u)
                << "accumulated frames must differ from a first one";
            expectSame(run(perNode({}), perNode(moving), false, false), run(perNode({}), perNode(moving), false, true));
            expectSame(run(global(), later, false, false), run(global(), later, false, true));
            expectSame(run(global(), global(), true, false), run(global(), global(), true, true));
        }
    }

    // The quantized LOD page pool: identity draws match the default ones and a
    // moving edit matches the moved scene on every chain that draws it, with
    // reveals, and through the RAD 3DGUT rasters and outlines.
    TEST_F(CustomEffectTest, QuantizedPagePoolDrawsMatch) {
        std::vector<std::uint32_t> slots(arrays_.size());
        std::iota(slots.begin(), slots.end(), 0u);
        std::array<Parameters, kNodes> values{};
        const glm::vec3 offset(0.2f, -0.1f, 0.0f);
        values[0][Offset] = glm::vec4(offset, 0.0f);
        Arrays edited = arrays_;
        for (std::size_t i = 0; i < edited.size(); ++i)
            if (edited.nodes[i] == 0)
                for (int axis = 0; axis < 3; ++axis)
                    edited.means[3 * i + axis] += offset[axis];
        struct Case {
            bool rad;
            Path path;
            bool reveal;
        };
        std::vector<Case> cases;
        for (const auto path : kPaths)
            cases.push_back({false, path, false});
        for (const auto path : {Path::Legacy, Path::Survivors, Path::Gut, Path::BandedGut})
            cases.push_back({false, path, true});
        for (const auto path : {Path::Gut, Path::GutOutline, Path::BandedGut})
            cases.push_back({true, path, false});
        const std::array scenes{lodModel(arrays_, false), lodModel(arrays_, true)};
        const std::array oracles{lodModel(edited, false), lodModel(edited, true)};
        const auto lod_request = [&](const Case& c) {
            auto r = request(c.path);
            r.sh_degree = 0; // the moved scene evaluates SH at the moved centers
            r.lod_indices = slots.data();
            r.lod_count = slots.size();
            if (c.reveal)
                r.reveal = lfs::rendering::GaussianRevealState{.progress = 0.45f, .softness = 0.2f, .center_x = 0.1f, .center_z = -4.4f, .radius = 2.0f, .effect = 3, .min_y = -1.0f, .max_y = 1.0f};
            return r;
        };
        // A model's first draw makes its pages resident.
        const auto settled = [&](const SplatData& model, const ViewportRenderRequest& r, const Path path) {
            (void)draw(model, r, path);
            auto frame = draw(model, r, path);
            EXPECT_TRUE(Access::quantPool(*viewport_));
            return frame;
        };
        const auto label = [](const Case& c) {
            return std::format("{}{}{}", name(c.path), c.rad ? " rad" : "", c.reveal ? " reveal" : "");
        };
        std::vector<Frame> plain, moved;
        for (const auto& c : cases) {
            SCOPED_TRACE(label(c));
            const auto r = lod_request(c);
            plain.push_back(settled(scenes[c.rad], r, c.path));
            moved.push_back(c.reveal ? Frame{} : settled(oracles[c.rad], r, c.path));
        }
        // RAD opacity encoding changes the image, so those draws take the RAD path.
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto same = std::ranges::find_if(
                cases, [&](const Case& c) { return !c.rad && !c.reveal && c.path == cases[i].path; });
            if (cases[i].rad)
                EXPECT_GT(compareImages(plain[i].pixels, plain[same - cases.begin()].pixels).max, 0.05f) << label(cases[i]);
        }
        publish(kIdentity);
        for (std::size_t i = 0; i < cases.size(); ++i) {
            SCOPED_TRACE(label(cases[i]));
            auto r = lod_request(cases[i]);
            r.custom_effect = global();
            const auto custom = settled(scenes[cases[i].rad], r, cases[i].path);
            EXPECT_TRUE(custom.custom);
            expectSame(custom, plain[i]);
        }
        publish(kEdit);
        for (std::size_t i = 0; i < cases.size(); ++i) {
            if (cases[i].reveal)
                continue; // the built-in reveal runs at the stored centers
            SCOPED_TRACE(label(cases[i]));
            auto r = lod_request(cases[i]);
            r.custom_effect = perNode(values);
            expectClose(settled(scenes[cases[i].rad], r, cases[i].path), moved[i], plain[i]);
        }
    }

    // Writes the context field parameter slot 0 selects to the color. Bound
    // globally with slot 1 set, it tries to rewrite the read-only ids.
    constexpr char kContext[] = R"(import lfs;
LfsElement effect(LfsElement e, LfsContext ctx) {
    if (ctx.param(1).x > 0.0) {
        e.index = 999u;
        e.objectId = 77u;
        e.kind = 5u;
        return e;
    }
    const float4 p = float4(1.0, 2.0, 3.0, 1.0);
    switch (uint(ctx.param(0).x)) {
    case 0: e.color = float3(ctx.time, ctx.progress, ctx.radius); break;
    case 1: e.color = ctx.cameraPosition; break;
    case 2: e.color = ctx.cameraIntrinsics.xyz; break;
    case 3: e.color = float3(ctx.cameraIntrinsics.w, float(ctx.imageSize.x), float(ctx.imageSize.y)); break;
    case 4: e.color = float3(float(ctx.cameraModel), ctx.softness, 0.0); break;
    case 5: e.color = mul(ctx.worldToView, p).xyz; break;
    case 6: e.color = ctx.worldToView[1].xyw; break;
    case 7: e.color = mul(p, ctx.worldToView).xyz; break;
    case 8: e.color = mul(ctx.objectToWorld, p).xyz; break;
    case 9: e.color = ctx.objectToWorld[0].xyw; break;
    case 10: e.color = mul(p, ctx.objectToWorld).xyz; break;
    default: e.color = float3(float(e.index), float(e.objectId), float(e.kind)); break;
    }
    return e;
}
)";

    class CustomEffectContextTest : public CustomEffectTest, public ::testing::WithParamInterface<Path> {};

    // The context a host gives the effect, read back from the projected color
    // of one element on node 2 (rotated, nonuniformly scaled), in 2D survivor
    // and 3DGUT projection, which keep their scratch; the same base camera
    // model in both.
    TEST_P(CustomEffectContextTest, HostsFillEveryContextField) {
        const Path path = GetParam();
        publish(kContext);
        Arrays one;
        one.nodes = {2};
        one.means = {0.0f, 0.0f, -4.0f};
        one.sh0 = {0.0f, 0.0f, 0.0f};
        one.shN.assign(9, 0.0f);
        one.scales = {-2.5f, -2.5f, -2.5f};
        one.rotations = {1.0f, 0.0f, 0.0f, 0.0f};
        one.opacities = {2.0f};
        const auto scene = model(one);
        auto r = request(path);
        r.scene.transform_indices = indices(one);
        r.frame_view.translation = glm::vec3(0.1f, -0.2f, 0.3f);
        r.frame_view.rotation = glm::mat3(glm::rotate(glm::mat4(1.0f), glm::radians(10.0f), glm::vec3(0.0f, 1.0f, 0.0f)));
        r.frame_view.intrinsics_override = lfs::rendering::CameraIntrinsics{
            .focal_x = 80.0f,
            .focal_y = 70.0f,
            .center_x = 50.0f,
            .center_y = 30.0f};
        GaussianCustomEffectState state;
        state.time = 2.5f;
        state.bounds_min = glm::vec3(-1.0f, -2.0f, -6.0f);
        state.bounds_max = glm::vec3(1.0f, 2.0f, -2.0f);
        state.bindings.resize(2);
        state.bindings[0].values[1] = glm::vec4(1.0f);
        state.global_binding = 0;
        state.objects.resize(kNodes);
        state.objects[2] = {.binding = 1, .id = 9};

        const glm::vec4 p(1.0f, 2.0f, 3.0f, 1.0f);
        const glm::mat3 world_to_camera =
            glm::transpose(lfs::rendering::dataCameraToWorldFromVisualizerRotation(r.frame_view.rotation));
        glm::mat4 view(world_to_camera);
        view[3] = glm::vec4(-world_to_camera * r.frame_view.translation, 1.0f);
        const glm::mat4& object = transforms_[2];
        const float radius = 0.5f * glm::length(state.bounds_max - state.bounds_min);
        const std::vector<glm::vec3> expected = {
            {2.5f, 1.0f, radius},
            r.frame_view.translation,
            {80.0f, 70.0f, 50.0f},
            {30.0f, 96.0f, 64.0f},
            {0.0f, 0.0f, 0.0f}, // pinhole in both projections
            glm::vec3(view * p),
            {view[0][1], view[1][1], view[3][1]},
            glm::vec3(p * view),
            glm::vec3(object * p),
            {object[0][0], object[1][0], object[3][0]},
            glm::vec3(p * object),
            {0.0f, 9.0f, 0.0f}, // logical index, object id and kind, unchanged by the global binding
        };
        for (std::size_t field = 0; field < expected.size(); ++field) {
            SCOPED_TRACE(field);
            state.bindings[1].values[0] = glm::vec4(static_cast<float>(field));
            r.custom_effect = state;
            const auto frame = draw(scene, r, path);
            ASSERT_TRUE(frame.custom);
            const auto color = Access::colors(*viewport_, context_->allocator(), 1);
            ASSERT_EQ(color.size(), 3u);
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(color[c], expected[field][c], 1e-4f * std::max(1.0f, std::abs(expected[field][c]))) << c;
        }
    }
    INSTANTIATE_TEST_SUITE_P(Projections, CustomEffectContextTest, ::testing::Values(Path::Survivors, Path::Gut),
                             [](const auto& info) { return info.param == Path::Gut ? "Gut" : "Survivors"; });

    TEST_F(CustomEffectTest, DispatchCountsAndStageTimings) {
        dispatchCases([](Path, std::size_t, auto) {});
    }

    // The table the hosts read: header, context, parameter blocks, then object records.
    TEST(CustomEffectTail, PacksTheLayoutTheHostsRead) {
        namespace overlay = lfs::vis::detail;
        VulkanGSRendererUniforms camera{};
        camera.fx = 100.0f;
        camera.fy = 90.0f;
        camera.cx = 48.0f;
        camera.cy = 32.0f;
        camera.camera_width = 96;
        camera.camera_height = 64;
        camera.camera_model = 3u | (1u << 8); // fisheye, drawn through 3DGUT
        camera.depth_offset = 2.0f;
        for (int i = 0; i < 16; ++i)
            camera.world_view_transform[i] = static_cast<float>(i);
        ViewportRenderRequest r;
        r.frame_view.translation = glm::vec3(1.0f, 2.0f, 3.0f);
        EXPECT_FALSE(overlay::buildCustomEffectTail(r, camera));
        GaussianCustomEffectState state;
        state.time = 2.5f;
        state.bounds_min = glm::vec3(-1.0f);
        state.bounds_max = glm::vec3(3.0f);
        state.bindings.resize(2);
        state.bindings[1].values[15] = glm::vec4(7.0f);
        state.bindings[1].effect = 3;
        state.objects = {{.binding = 1, .id = 9}, {}};
        state.global_binding = 0;
        r.custom_effect = state;
        const auto tail = overlay::buildCustomEffectTail(r, camera);
        ASSERT_TRUE(tail) << lfs::format_for_developer(tail.error());
        const auto slot = [&](const std::vector<float>& t, const std::size_t index) {
            return glm::vec4(t[4 * index], t[4 * index + 1], t[4 * index + 2], t[4 * index + 3]);
        };
        const auto bits = [&](const std::vector<float>& t, const std::size_t index) {
            const auto v = slot(t, index);
            return glm::uvec4(std::bit_cast<std::uint32_t>(v.x), std::bit_cast<std::uint32_t>(v.y),
                              std::bit_cast<std::uint32_t>(v.z), std::bit_cast<std::uint32_t>(v.w));
        };
        constexpr std::size_t parameters = LFS_VK_CUSTOM_HEADER_SLOTS, effects = parameters + 32, objects = effects + 1;
        ASSERT_EQ(tail->size(), 4 * (objects + 2));
        // Without a reveal the reveal binding is absent.
        EXPECT_EQ(bits(*tail, LFS_VK_CUSTOM_HEADER_SLOT), glm::uvec4(2, 2, 0, GaussianCustomEffectState::kNone));
        EXPECT_EQ(slot(*tail, LFS_VK_CUSTOM_TIMING_SLOT), glm::vec4(2.5f, 1.0f, 0.0f, 0.5f * std::sqrt(48.0f)));
        EXPECT_EQ(slot(*tail, LFS_VK_CUSTOM_BOUNDS_MIN_SLOT), glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f));
        EXPECT_EQ(slot(*tail, LFS_VK_CUSTOM_BOUNDS_MAX_SLOT), glm::vec4(3.0f, 3.0f, 3.0f, 0.0f));
        EXPECT_EQ(slot(*tail, LFS_VK_CUSTOM_CENTER_SLOT), glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
        EXPECT_EQ(slot(*tail, LFS_VK_CUSTOM_CAMERA_POSITION_SLOT), glm::vec4(1.0f, 2.0f, 3.0f, 0.0f));
        EXPECT_EQ(slot(*tail, LFS_VK_CUSTOM_INTRINSICS_SLOT), glm::vec4(100.0f, 90.0f, 48.0f, 32.0f));
        // The base camera model, without the renderer's 3DGUT bit.
        EXPECT_EQ(bits(*tail, LFS_VK_CUSTOM_IMAGE_SLOT), glm::uvec4(96, 64, 3, 0));
        // Rows of the column-major view, with the orthographic setback removed.
        EXPECT_EQ(slot(*tail, LFS_VK_CUSTOM_WORLD_TO_VIEW_SLOT), glm::vec4(0.0f, 4.0f, 8.0f, 12.0f));
        EXPECT_EQ(slot(*tail, LFS_VK_CUSTOM_WORLD_TO_VIEW_SLOT + 2), glm::vec4(2.0f, 6.0f, 10.0f, 12.0f));
        EXPECT_EQ(slot(*tail, parameters + 31), glm::vec4(7.0f));
        EXPECT_EQ(bits(*tail, effects), glm::uvec4(0, 3, 0, 0));
        EXPECT_EQ(bits(*tail, objects), glm::uvec4(1, 9, 0, 0));
        EXPECT_EQ(bits(*tail, objects + 1), glm::uvec4(GaussianCustomEffectState::kNone, 0, 0, 0));

        r.custom_effect->reveal_binding = 1;
        r.custom_effect->reveal_progress = .5f;
        r.custom_effect->reveal_softness = .2f;
        const auto custom = overlay::buildCustomEffectTail(r, camera);
        ASSERT_TRUE(custom);
        EXPECT_EQ(bits(*custom, LFS_VK_CUSTOM_HEADER_SLOT), glm::uvec4(2, 2, 0, 1));
        EXPECT_EQ(slot(*custom, LFS_VK_CUSTOM_TIMING_SLOT), glm::vec4(2.5f, .5f, .2f, std::sqrt(18.0f)));
        EXPECT_EQ(slot(*custom, LFS_VK_CUSTOM_CENTER_SLOT), glm::vec4(0.0f));
        r.reveal = lfs::rendering::GaussianRevealState{.progress = 2.0f, .softness = 0.25f, .center_x = 1.0f, .center_z = -2.0f, .radius = 5.0f};
        const auto revealed = overlay::buildCustomEffectTail(r, camera);
        ASSERT_TRUE(revealed);
        EXPECT_EQ(bits(*revealed, LFS_VK_CUSTOM_HEADER_SLOT), glm::uvec4(2, 2, 0, 1));
        EXPECT_EQ(slot(*revealed, LFS_VK_CUSTOM_TIMING_SLOT), glm::vec4(2.5f, 1.0f, 0.25f, 5.0f));
        EXPECT_EQ(slot(*revealed, LFS_VK_CUSTOM_CENTER_SLOT), glm::vec4(1.0f, 0.0f, -2.0f, 0.0f));
        r.custom_effect->objects[1].binding = 2;
        EXPECT_FALSE(overlay::buildCustomEffectTail(r, camera));
    }

} // namespace lfs::test::custom_effect

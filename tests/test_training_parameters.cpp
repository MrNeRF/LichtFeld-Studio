/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/optimization_properties.hpp"
#include "core/parameters.hpp"
#include "core/property_registry.hpp"
#include "python/lfs/py_params.hpp"

#include <algorithm>
#include <any>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using lfs::core::param::apply_explicit_training_overrides;
using lfs::core::param::OptimizationParameters;
using lfs::core::prop::PropertyMeta;
using lfs::core::prop::PropertyObjectRef;
using lfs::core::prop::PropertyRegistry;
using lfs::core::prop::PropType;

namespace {

    PropertyMeta optimization_meta(const std::string& id) {
        auto meta = PropertyRegistry::instance().get_property("optimization", id);
        if (!meta)
            throw std::runtime_error("Missing registered optimization property: " + id);
        return *meta;
    }

    template <typename T>
    T resolved(const OptimizationParameters& source, const std::string& id) {
        return std::any_cast<T>(lfs::python::resolve_optimization_default(optimization_meta(id), source));
    }

    std::filesystem::path eval_config_path(const std::string_view filename) {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "eval" / filename;
    }

    std::uint64_t frozen_config_fingerprint(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("Cannot read frozen config: " + path.string());

        std::uint64_t hash = 0xcbf29ce484222325ULL;
        char byte = 0;
        while (input.get(byte)) {
            if (byte == '\r')
                continue;
            hash ^= static_cast<unsigned char>(byte);
            hash *= 0x100000001b3ULL;
        }
        return hash;
    }

    std::string read_file_bytes(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("Cannot read fixture: " + path.string());
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    void expect_same_value(const PropertyMeta& meta, const std::any& actual, const std::any& expected) {
        switch (meta.type) {
        case PropType::Float:
            EXPECT_FLOAT_EQ(std::any_cast<float>(actual), std::any_cast<float>(expected));
            break;
        case PropType::Int:
            EXPECT_EQ(std::any_cast<int>(actual), std::any_cast<int>(expected));
            break;
        case PropType::SizeT:
            EXPECT_EQ(std::any_cast<size_t>(actual), std::any_cast<size_t>(expected));
            break;
        case PropType::Bool:
            EXPECT_EQ(std::any_cast<bool>(actual), std::any_cast<bool>(expected));
            break;
        case PropType::String:
            EXPECT_EQ(std::any_cast<std::string>(actual), std::any_cast<std::string>(expected));
            break;
        case PropType::Enum:
            EXPECT_EQ(std::any_cast<int>(actual), std::any_cast<int>(expected));
            break;
        default:
            ADD_FAILURE() << "unsupported registered property type";
            break;
        }
    }

    class TrainingParametersTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::core::param::register_optimization_properties();
        }

        void TearDown() override {
            PropertyRegistry::instance().unregister_group("optimization");
        }
    };

    // The MRNF sentinels cross several member types and factory overrides, catching wrong-member
    // getter wiring, MRNF factory drift, and a resolver that falls back to stored constants.
    TEST_F(TrainingParametersTest, ResolvesMrnfFactorySentinels) {
        const auto defaults = OptimizationParameters::defaults_for_strategy("mrnf");

        EXPECT_FLOAT_EQ(resolved<float>(defaults, "opacity_lr"), 0.012f);
        EXPECT_EQ(resolved<int>(defaults, "max_cap"), 5'000'000);
        EXPECT_FLOAT_EQ(resolved<float>(defaults, "min_opacity"), 1.0f / 255.0f);
        EXPECT_FLOAT_EQ(resolved<float>(defaults, "opacity_reg"), 0.003f);
        EXPECT_EQ(resolved<size_t>(defaults, "refine_every"), 200u);
    }

    TEST_F(TrainingParametersTest, StrategyApplicabilityTagsAreCanonicalAndKnown) {
        const std::set<std::string> canonical = {"mcmc", "mrnf", "igs+"};
        const std::map<std::string, std::vector<std::string>> expected = {
            {"means_lr_end", {"mrnf"}},
            {"scaling_lr_end", {"mrnf"}},
            {"growth_grad_threshold", {"mrnf"}},
            {"grow_fraction", {"mrnf"}},
            {"grow_until_iter", {"mrnf"}},
            {"opacity_decay", {"mrnf"}},
            {"scale_decay", {"mrnf"}},
            {"means_noise_weight", {"mrnf"}},
            {"bounds_percentile", {"mrnf"}},
            {"use_error_map", {"mrnf"}},
            {"use_edge_map", {"mrnf"}},
            {"background_improvements", {"mrnf"}},
            {"far_scene_min_fraction", {"mrnf"}},
            {"growth_ratio_rank", {"mrnf"}},
            {"growth_ratio_pow", {"mrnf"}},
            {"fill_pacing_iter", {"mrnf"}},
            {"far_seed_dose", {"mrnf"}},
            {"prune_opacity", {"igs+"}},
            {"reset_every", {"igs+"}},
            {"min_opacity", {"mcmc"}},
        };

        const auto group = PropertyRegistry::instance().get_group_snapshot("optimization");
        ASSERT_TRUE(group.has_value());
        std::set<std::string> tagged;
        for (const auto& meta : group->properties) {
            EXPECT_TRUE(meta.strategy_applicability_explicit) << meta.id;
            for (const auto& strategy : meta.strategies) {
                EXPECT_TRUE(canonical.contains(strategy)) << meta.id << ": " << strategy;
                tagged.insert(meta.id);
            }
            const auto expected_it = expected.find(meta.id);
            if (expected_it == expected.end())
                EXPECT_TRUE(meta.strategies.empty()) << meta.id;
            else
                EXPECT_EQ(meta.strategies, expected_it->second);
        }
        for (const auto& [prop_id, strategies] : expected) {
            const auto meta = optimization_meta(prop_id);
            EXPECT_EQ(meta.strategies, strategies);
            EXPECT_TRUE(tagged.contains(prop_id));
        }
    }

    // These values discriminate MCMC from MRNF and pin both strategy dispatch and MCMC factory
    // values while continuing to exercise the production resolver seam.
    TEST_F(TrainingParametersTest, ResolvesMcmcFactorySentinels) {
        const auto defaults = OptimizationParameters::defaults_for_strategy("mcmc");

        EXPECT_FLOAT_EQ(resolved<float>(defaults, "opacity_lr"), 0.025f);
        EXPECT_EQ(resolved<int>(defaults, "max_cap"), 1'000'000);
        EXPECT_EQ(resolved<size_t>(defaults, "refine_every"), 100u);
    }

    // The IGS+ block pins its distinct override set, catching IGS+ factory drift and resolution
    // that incorrectly uses either the bare struct or another strategy's source instance.
    TEST_F(TrainingParametersTest, ResolvesIgsPlusFactorySentinels) {
        const auto defaults = OptimizationParameters::defaults_for_strategy("igs+");

        EXPECT_FLOAT_EQ(resolved<float>(defaults, "scaling_lr"), 0.02f);
        EXPECT_FLOAT_EQ(resolved<float>(defaults, "shs_lr"), 0.005f);
        EXPECT_EQ(resolved<int>(defaults, "max_cap"), 4'000'000);
        EXPECT_FLOAT_EQ(resolved<float>(defaults, "tv_loss_weight"), 5.0f);
        EXPECT_FLOAT_EQ(resolved<float>(defaults, "init_opacity"), 0.1f);
        EXPECT_EQ(resolved<size_t>(defaults, "stop_refine"), 15'000u);
    }

    TEST_F(TrainingParametersTest, ResolvesEveryRegisteredPropertyFromStrategySource) {
        const auto group = PropertyRegistry::instance().get_group_snapshot("optimization");
        ASSERT_TRUE(group.has_value());
        ASSERT_FALSE(group->properties.empty());

        std::array<std::pair<std::string_view, OptimizationParameters>, 3> direct_factories = {{
            {"mcmc", OptimizationParameters::mcmc_defaults()},
            {"mrnf", OptimizationParameters::mrnf_defaults()},
            {"igs+", OptimizationParameters::igs_plus_defaults()},
        }};

        for (auto& [strategy, direct_factory] : direct_factories) {
            const auto dispatched = OptimizationParameters::defaults_for_strategy(strategy);
            for (const auto& meta : group->properties) {
                SCOPED_TRACE(std::string(strategy) + ":" + meta.id);
                ASSERT_TRUE(meta.getter);

                const auto actual = lfs::python::resolve_optimization_default(meta, dispatched);
                auto direct_ref = PropertyObjectRef::cpp(&direct_factory);
                const auto expected = meta.getter(direct_ref);
                expect_same_value(meta, actual, expected);
            }
        }
    }

    TEST_F(TrainingParametersTest, DefaultsForStrategyCanonicalizesAliasesAndFallbacks) {
        EXPECT_FLOAT_EQ(OptimizationParameters::defaults_for_strategy("mnrf").opacity_lr, 0.012f);
        EXPECT_FLOAT_EQ(OptimizationParameters::defaults_for_strategy("lfs").opacity_lr, 0.012f);
        EXPECT_FLOAT_EQ(OptimizationParameters::defaults_for_strategy("").opacity_lr, 0.012f);
        EXPECT_FLOAT_EQ(OptimizationParameters::defaults_for_strategy("garbage").opacity_lr, 0.012f);
    }

    // This checks the serialized surface, not C++ struct completeness: a member absent from both
    // serialization and registration is intentionally out of scope. bg_image_path is emitted only
    // when non-empty, so the probe sets it explicitly before enumerating JSON keys.
    TEST_F(TrainingParametersTest, SerializedSurfaceHasRegistryCoverage) {
        OptimizationParameters serialization_probe{};
        serialization_probe.bg_image_path = "coverage-background.png";
        const auto serialized = serialization_probe.to_json();

        const auto group = PropertyRegistry::instance().get_group_snapshot("optimization");
        ASSERT_TRUE(group.has_value());

        std::set<std::string> registered_ids;
        std::map<std::string, std::string> json_to_property;
        for (const auto& meta : group->properties) {
            ASSERT_TRUE(registered_ids.insert(meta.id).second) << "duplicate property id: " << meta.id;
            const auto& json_key = meta.json_key.empty() ? meta.id : meta.json_key;
            ASSERT_TRUE(json_to_property.emplace(json_key, meta.id).second)
                << "duplicate optimization JSON key: " << json_key;
        }

        const std::map<std::string, std::string> allowlist = {
            {"bg_color", "background color uses its dedicated Python binding"},
            {"bg_image_path", "background image path uses its dedicated Python binding"},
            {"enable_save_eval_images", "evaluation image output is not a registry property"},
            {"eval_steps", "vector-valued evaluation schedule is managed separately"},
            {"ppisp_sidecar_path", "PPISP sidecar path uses its dedicated Python binding"},
            {"save_steps", "vector-valued save schedule is managed separately"},
        };

        for (const auto& [json_key, reason] : allowlist) {
            SCOPED_TRACE(json_key);
            EXPECT_TRUE(serialized.contains(json_key));
            EXPECT_FALSE(reason.empty());
            EXPECT_FALSE(registered_ids.contains(json_key));
            EXPECT_FALSE(json_to_property.contains(json_key));
        }

        for (auto it = serialized.begin(); it != serialized.end(); ++it) {
            const std::string& json_key = it.key();
            EXPECT_TRUE(json_to_property.contains(json_key) || allowlist.contains(json_key))
                << "serialized key has no declaration-carried property mapping or allow-list reason: " << json_key;
        }

        for (const auto& [json_key, property_id] : json_to_property) {
            EXPECT_TRUE(serialized.contains(json_key))
                << "registered property has no serialized key: " << property_id;
        }
    }

    TEST_F(TrainingParametersTest, DeclarationPinsRequiredJsonKeys) {
        const auto group = PropertyRegistry::instance().get_group_snapshot("optimization");
        ASSERT_TRUE(group.has_value());

        std::set<std::string> required;
        for (const auto& meta : group->properties) {
            if (meta.json_required)
                required.insert(meta.json_key.empty() ? meta.id : meta.json_key);
        }

        EXPECT_EQ(required, (std::set<std::string>{
                                "iterations",
                                "means_lr",
                                "shs_lr",
                                "opacity_lr",
                                "scaling_lr",
                                "rotation_lr",
                                "lambda_dssim",
                                "min_opacity",
                                "refine_every",
                                "start_refine",
                                "stop_refine",
                                "sh_degree"}));
    }

    TEST_F(TrainingParametersTest, CoreSerializationEnsuresOptimizationRegistration) {
        PropertyRegistry::instance().unregister_group("optimization");
        ASSERT_FALSE(PropertyRegistry::instance().get_group_snapshot("optimization"));

        const auto json = OptimizationParameters::mrnf_defaults().to_json();

        EXPECT_TRUE(json.contains("iterations"));
        EXPECT_TRUE(PropertyRegistry::instance().get_group_snapshot("optimization"));
    }

    TEST_F(TrainingParametersTest, MissingRequiredJsonKeyStillThrows) {
        auto json = OptimizationParameters::mrnf_defaults().to_json();
        json.erase("iterations");
        EXPECT_THROW((void)OptimizationParameters::from_json(json), nlohmann::json::out_of_range);
    }

    TEST_F(TrainingParametersTest, PpispExposureFromExifRoundTripsThroughJson) {
        auto params = OptimizationParameters::mrnf_defaults();
        EXPECT_TRUE(params.ppisp_exposure_from_exif);
        EXPECT_TRUE(params.to_json().at("ppisp_exposure_from_exif").get<bool>());

        params.ppisp_exposure_from_exif = false;
        const auto json = params.to_json();
        EXPECT_FALSE(json.at("ppisp_exposure_from_exif").get<bool>());
        EXPECT_FALSE(OptimizationParameters::from_json(json).ppisp_exposure_from_exif);
    }

    TEST_F(TrainingParametersTest, MissingOptionalJsonValuesUseStrategyDefaults) {
        auto mrnf_json = OptimizationParameters::mrnf_defaults().to_json();
        mrnf_json.erase("max_cap");
        EXPECT_EQ(OptimizationParameters::from_json(mrnf_json).max_cap, 5'000'000);

        auto igs_json = OptimizationParameters::igs_plus_defaults().to_json();
        igs_json.erase("tv_loss_weight");
        EXPECT_FLOAT_EQ(OptimizationParameters::from_json(igs_json).tv_loss_weight, 5.0f);
    }

    TEST_F(TrainingParametersTest, RemovedOptimizationJsonKeysAreIgnored) {
        auto json = OptimizationParameters::mrnf_defaults().to_json();
        json["grad_threshold"] = 0.003f;
        json["grow_scale3d"] = 0.01f;
        json["grow_scale2d"] = 0.05f;
        json["prune_scale3d"] = 0.1f;
        json["prune_scale2d"] = 0.15f;
        json["pause_refine_after_reset"] = 100u;
        json["revised_opacity"] = true;

        const auto parsed = OptimizationParameters::from_json(json);
        EXPECT_EQ(parsed.strategy, "mrnf");
        EXPECT_TRUE(parsed.validate().empty());
    }

    TEST_F(TrainingParametersTest, NormalLossSpaceKeepsStringWireFormat) {
        using lfs::core::param::NormalLossSpace;
        const std::array<std::pair<NormalLossSpace, std::string_view>, 4> values = {{
            {NormalLossSpace::Auto, "auto"},
            {NormalLossSpace::CameraOpenCV, "camera-opencv"},
            {NormalLossSpace::CameraOpenGL, "camera-opengl"},
            {NormalLossSpace::World, "world"},
        }};

        for (const auto& [space, wire] : values) {
            auto params = OptimizationParameters::mrnf_defaults();
            params.normal_loss_space = space;
            const auto json = params.to_json();
            EXPECT_EQ(json.at("normal_loss_space").get<std::string>(), wire);

            auto old_json = json;
            old_json["normal_loss_space"] = wire;
            EXPECT_EQ(OptimizationParameters::from_json(old_json).normal_loss_space, space);
        }
    }

    TEST_F(TrainingParametersTest, NormalAutoGenerateRoundTripAndDefault) {
        auto params = OptimizationParameters::mrnf_defaults();
        EXPECT_TRUE(params.normal_auto_generate);
        EXPECT_TRUE(params.validate().empty());

        params.normal_auto_generate = false;
        const auto json = params.to_json();
        EXPECT_FALSE(json.at("normal_auto_generate").get<bool>());

        const auto restored = OptimizationParameters::from_json(json);
        EXPECT_FALSE(restored.normal_auto_generate);

        auto missing = json;
        missing.erase("normal_auto_generate");
        const auto defaults = OptimizationParameters::from_json(missing);
        EXPECT_TRUE(defaults.normal_auto_generate);
    }

    TEST_F(TrainingParametersTest, NormalSupervisionScheduleRoundTripAndValidation) {
        auto params = OptimizationParameters::mrnf_defaults();
        EXPECT_FLOAT_EQ(params.normal_start_fraction, 0.2f);
        EXPECT_FLOAT_EQ(params.normal_end_fraction, 1.0f);
        EXPECT_TRUE(params.validate().empty());

        params.normal_start_fraction = 0.3f;
        params.normal_end_fraction = 0.9f;
        const auto json = params.to_json();
        EXPECT_FLOAT_EQ(json.at("normal_start_fraction").get<float>(), 0.3f);
        EXPECT_FLOAT_EQ(json.at("normal_end_fraction").get<float>(), 0.9f);

        const auto restored = OptimizationParameters::from_json(json);
        EXPECT_FLOAT_EQ(restored.normal_start_fraction, 0.3f);
        EXPECT_FLOAT_EQ(restored.normal_end_fraction, 0.9f);
        EXPECT_TRUE(restored.validate().empty());

        auto missing = json;
        missing.erase("normal_start_fraction");
        missing.erase("normal_end_fraction");
        const auto defaults = OptimizationParameters::from_json(missing);
        EXPECT_FLOAT_EQ(defaults.normal_start_fraction, 0.2f);
        EXPECT_FLOAT_EQ(defaults.normal_end_fraction, 1.0f);

        params.normal_start_fraction = 0.8f;
        params.normal_end_fraction = 0.4f;
        const auto order_error = params.validate();
        EXPECT_NE(order_error.find("normal_start_fraction must not exceed normal_end_fraction"),
                  std::string::npos)
            << order_error;

        params.normal_start_fraction = 0.4f;
        params.normal_end_fraction = 0.4f;
        EXPECT_TRUE(params.validate().empty());

        params.normal_start_fraction = 1.1f;
        params.normal_end_fraction = 1.0f;
        EXPECT_NE(params.validate().find("normal_start_fraction"), std::string::npos);
        params.normal_start_fraction = 0.2f;
        params.normal_end_fraction = -0.1f;
        EXPECT_NE(params.validate().find("normal_end_fraction"), std::string::npos);
    }

    TEST_F(TrainingParametersTest, NormalSupervisionActiveRespectsStartEndAndStepsScaler) {
        OptimizationParameters params;
        params.use_normal_loss = true;
        params.iterations = 30'000;
        params.normal_start_fraction = 0.2f;
        params.normal_end_fraction = 1.0f;

        const int total = params.resolved_total_iterations();
        ASSERT_EQ(total, 30'000);
        const int start_iter = static_cast<int>(
            params.normal_start_fraction * static_cast<float>(std::max(1, total)));
        EXPECT_EQ(start_iter, 6'000);

        EXPECT_FALSE(params.normal_supervision_active(start_iter - 1));
        EXPECT_TRUE(params.normal_supervision_active(start_iter));
        EXPECT_TRUE(params.normal_supervision_active(total));

        params.use_normal_loss = false;
        EXPECT_FALSE(params.normal_supervision_active(start_iter));
        params.use_normal_loss = true;

        params.normal_end_fraction = 0.8f;
        EXPECT_TRUE(params.normal_supervision_active(start_iter));
        EXPECT_TRUE(params.normal_supervision_active(23'999));
        EXPECT_FALSE(params.normal_supervision_active(24'000));
        EXPECT_FALSE(params.normal_supervision_active(total));

        params.normal_end_fraction = 1.0f;
        params.steps_scaler = 0.5f;
        params.apply_step_scaling();
        const int scaled_total = params.resolved_total_iterations();
        EXPECT_EQ(scaled_total, 15'000);
        const int scaled_start = static_cast<int>(
            params.normal_start_fraction * static_cast<float>(std::max(1, scaled_total)));
        EXPECT_EQ(scaled_start, 3'000);
        EXPECT_FALSE(params.normal_supervision_active(scaled_start - 1));
        EXPECT_TRUE(params.normal_supervision_active(scaled_start));
        EXPECT_TRUE(params.normal_supervision_active(scaled_total));
    }

    TEST_F(TrainingParametersTest, OldNewToJsonParity) {
        const std::array<std::pair<std::string_view, OptimizationParameters>, 3> factories = {{
            {"mcmc", OptimizationParameters::mcmc_defaults()},
            {"mrnf", OptimizationParameters::mrnf_defaults()},
            {"igs_plus", OptimizationParameters::igs_plus_defaults()},
        }};

        for (const auto& [name, params] : factories) {
            SCOPED_TRACE(name);
            const auto fixture_path = std::filesystem::path(PROJECT_ROOT_PATH) /
                                      "tests" / "data" / "param_json_golden" /
                                      (std::string(name) + ".json");
            const auto fixture_bytes = read_file_bytes(fixture_path);
            const auto expected = nlohmann::json::parse(fixture_bytes);
            const auto actual = params.to_json();

            EXPECT_TRUE(actual == expected);
            EXPECT_EQ(actual.dump(2), expected.dump(2));
            EXPECT_EQ(actual.dump(2) + '\n', fixture_bytes);

            auto mutated = params;
            auto mutated_expected = expected;
            if (name == "mcmc") {
                mutated.opacity_lr = 0.03125f;
                mutated_expected["opacity_lr"] = mutated.opacity_lr;
            } else if (name == "mrnf") {
                mutated.max_cap = 123'456'789;
                mutated_expected["max_cap"] = mutated.max_cap;
            } else {
                mutated.init_extent = 4.25f;
                mutated_expected["init_extent"] = mutated.init_extent;
            }

            const auto mutated_actual = mutated.to_json();
            EXPECT_TRUE(mutated_actual == mutated_expected);
            EXPECT_EQ(mutated_actual.dump(2), mutated_expected.dump(2));
        }
    }

    TEST_F(TrainingParametersTest, EvalBenchmarkConfigsParseAsIs) {
        const auto mcmc_path = eval_config_path("mcmc_optimization_params.json");
        EXPECT_EQ(frozen_config_fingerprint(mcmc_path), 0x627939cba4bdc0bbULL);
        const auto mcmc_result = lfs::core::param::read_optim_params_from_json(mcmc_path);
        ASSERT_TRUE(mcmc_result.has_value()) << mcmc_result.error();
        EXPECT_FLOAT_EQ(mcmc_result->opacity_lr, 0.0335f);
        EXPECT_FLOAT_EQ(mcmc_result->shs_lr, 0.0024f);
        EXPECT_FLOAT_EQ(mcmc_result->opacity_reg, 0.0042f);
        EXPECT_EQ(mcmc_result->strategy, "mcmc");
        EXPECT_EQ(mcmc_result->max_cap, 1'000'000);

        const auto mrnf_path = eval_config_path("mrnf_optimization_params.json");
        EXPECT_EQ(frozen_config_fingerprint(mrnf_path), 0xd673eeb0fe318eeULL);
        const auto mrnf_result = lfs::core::param::read_optim_params_from_json(mrnf_path);
        ASSERT_TRUE(mrnf_result.has_value()) << mrnf_result.error();
        EXPECT_FLOAT_EQ(mrnf_result->means_lr, 2e-05f);
        EXPECT_FLOAT_EQ(mrnf_result->means_lr_end, 2e-07f);
        EXPECT_EQ(mrnf_result->start_refine, 0u);
        EXPECT_EQ(mrnf_result->stop_refine, 28'500u);
        EXPECT_FLOAT_EQ(mrnf_result->min_opacity, 0.0039215689f);

        const auto igs_path = eval_config_path("improvedGSplus_optimization_params.json");
        EXPECT_EQ(frozen_config_fingerprint(igs_path), 0xf86e40494df20d22ULL);
        const auto igs_result = lfs::core::param::read_optim_params_from_json(igs_path);
        ASSERT_TRUE(igs_result.has_value()) << igs_result.error();
        EXPECT_FLOAT_EQ(igs_result->init_opacity, 0.3f);
        EXPECT_FLOAT_EQ(igs_result->init_scaling, 0.2f);
        EXPECT_EQ(igs_result->refine_every, 500u);
        EXPECT_FLOAT_EQ(igs_result->tv_loss_weight, 5.0f);
        EXPECT_EQ(igs_result->strategy, "igs+");
    }

    TEST_F(TrainingParametersTest, ExploreStarvationWeightingIsConfigResidue) {
        const auto defaults = OptimizationParameters::mrnf_defaults();
        EXPECT_TRUE(defaults.explore_starvation_weighting);

        const auto default_json = defaults.to_json();
        EXPECT_FALSE(default_json.contains("explore_starvation_weighting"));
        EXPECT_FALSE(PropertyRegistry::instance().get_property("optimization", "explore_starvation_weighting"));

        auto json = defaults.to_json();
        json["explore_starvation_weighting"] = false;
        const auto parsed = OptimizationParameters::from_json(json);
        EXPECT_FALSE(parsed.explore_starvation_weighting);
        EXPECT_TRUE(parsed.validate().empty());
        EXPECT_FALSE(parsed.to_json().at("explore_starvation_weighting").get<bool>());
    }

    TEST_F(TrainingParametersTest, SaveLoadRoundTripPreservesParameters) {
        std::array<std::pair<std::string_view, OptimizationParameters>, 3> factories = {{
            {"mcmc", OptimizationParameters::mcmc_defaults()},
            {"mrnf", OptimizationParameters::mrnf_defaults()},
            {"igs_plus", OptimizationParameters::igs_plus_defaults()},
        }};
        factories[0].second.opacity_lr = 0.03125f;
        factories[1].second.max_cap = 123'456'789;
        factories[2].second.init_extent = 4.25f;

        const auto group = PropertyRegistry::instance().get_group_snapshot("optimization");
        ASSERT_TRUE(group.has_value());

        for (auto& [strategy, expected] : factories) {
            SCOPED_TRACE(strategy);
            const auto path = std::filesystem::temp_directory_path() /
                              ("lfs_training_parameters_roundtrip_" + std::string(strategy) + ".json");
            std::error_code ec;
            std::filesystem::remove(path, ec);

            lfs::core::param::TrainingParameters training;
            training.optimization = expected;
            const auto save_result = lfs::core::param::save_training_parameters_to_json(training, path);
            ASSERT_TRUE(save_result.has_value()) << save_result.error();

            auto load_result = lfs::core::param::read_optim_params_from_json(path);
            std::filesystem::remove(path, ec);
            ASSERT_TRUE(load_result.has_value()) << load_result.error();

            auto actual_ref = PropertyObjectRef::cpp(&*load_result);
            auto expected_ref = PropertyObjectRef::cpp(&expected);
            for (const auto& meta : group->properties) {
                SCOPED_TRACE(meta.id);
                ASSERT_TRUE(meta.getter);
                expect_same_value(meta, meta.getter(actual_ref), meta.getter(expected_ref));
            }
        }
    }

    TEST(ExplicitTrainingOverridesTest, ResumeKeepsRestoredUnlessKeyPresent) {
        lfs::core::param::TrainingParameters restored;
        restored.optimization.iterations = 30'000;
        restored.optimization.enable_eval = false;
        restored.optimization.enable_save_eval_images = false;
        restored.optimization.eval_steps = {7'000, 30'000};
        restored.optimization.save_steps = {7'000, 30'000};
        restored.optimization.max_cap = 1'234'567;
        restored.dataset.test_every = 8;

        lfs::core::param::ExplicitTrainingOverrides overrides;
        apply_explicit_training_overrides(restored, overrides);
        EXPECT_EQ(restored.optimization.iterations, 30'000u);
        EXPECT_FALSE(restored.optimization.enable_eval);
        EXPECT_EQ(restored.dataset.test_every, 8);
        EXPECT_EQ(restored.optimization.max_cap, 1'234'567);

        overrides.optimization_json = nlohmann::json{
            {"iterations", 30100},
            {"enable_eval", true},
            {"enable_save_eval_images", true},
            {"eval_steps", {30100}},
            {"save_steps", {30100}},
        }
                                          .dump();
        overrides.dataset_json = nlohmann::json{{"test_every", 64}}.dump();
        apply_explicit_training_overrides(restored, overrides);

        EXPECT_EQ(restored.optimization.iterations, 30100u);
        EXPECT_TRUE(restored.optimization.enable_eval);
        EXPECT_TRUE(restored.optimization.enable_save_eval_images);
        EXPECT_EQ(restored.optimization.eval_steps, std::vector<size_t>({30100}));
        EXPECT_EQ(restored.optimization.save_steps, std::vector<size_t>({30100}));
        EXPECT_EQ(restored.dataset.test_every, 64);
        EXPECT_EQ(restored.optimization.max_cap, 1'234'567);
        EXPECT_TRUE(overrides.has_optimization_key("iterations"));
        EXPECT_TRUE(overrides.has_optimization_key("eval_steps"));
        EXPECT_TRUE(overrides.has_dataset_key("test_every"));
        EXPECT_FALSE(overrides.has_optimization_key("max_cap"));
    }

    TEST(ExplicitTrainingOverridesTest, CliKeysWinOverConfigKeys) {
        lfs::core::param::TrainingParameters restored;
        restored.optimization.iterations = 30'000;
        restored.optimization.eval_steps = {7'000, 30'000};
        restored.dataset.test_every = 8;

        lfs::core::param::ExplicitTrainingOverrides overrides;
        lfs::core::param::merge_explicit_json_overlay(
            overrides.optimization_json,
            nlohmann::json{{"iterations", 30100}, {"eval_steps", {30100}}}.dump());
        lfs::core::param::merge_explicit_json_overlay(
            overrides.dataset_json, nlohmann::json{{"test_every", 32}}.dump());
        lfs::core::param::merge_explicit_json_overlay(
            overrides.optimization_json, nlohmann::json{{"iterations", 40000}}.dump());
        lfs::core::param::merge_explicit_json_overlay(
            overrides.dataset_json, nlohmann::json{{"test_every", 64}}.dump());

        apply_explicit_training_overrides(restored, overrides);
        EXPECT_EQ(restored.optimization.iterations, 40000u);
        EXPECT_EQ(restored.optimization.eval_steps, std::vector<size_t>({30100}));
        EXPECT_EQ(restored.dataset.test_every, 64);
    }

} // namespace

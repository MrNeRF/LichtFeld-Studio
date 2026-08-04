/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/property_registry.hpp"
#include "python/lfs/py_params.hpp"

#include <any>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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
            lfs::python::register_optimization_properties();
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
        EXPECT_TRUE(resolved<bool>(defaults, "revised_opacity"));
        EXPECT_FLOAT_EQ(resolved<float>(defaults, "opacity_reg"), 0.0f);
        EXPECT_EQ(resolved<size_t>(defaults, "refine_every"), 200u);
        EXPECT_FLOAT_EQ(resolved<float>(defaults, "grad_threshold"), 0.003f);
    }

    // These values discriminate MCMC from MRNF and pin both strategy dispatch and MCMC factory
    // values while continuing to exercise the production resolver seam.
    TEST_F(TrainingParametersTest, ResolvesMcmcFactorySentinels) {
        const auto defaults = OptimizationParameters::defaults_for_strategy("mcmc");

        EXPECT_FLOAT_EQ(resolved<float>(defaults, "opacity_lr"), 0.025f);
        EXPECT_EQ(resolved<int>(defaults, "max_cap"), 1'000'000);
        EXPECT_EQ(resolved<size_t>(defaults, "refine_every"), 100u);
        EXPECT_FALSE(resolved<bool>(defaults, "revised_opacity"));
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
        const auto* group = PropertyRegistry::instance().get_group("optimization");
        ASSERT_NE(group, nullptr);
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

        const auto* group = PropertyRegistry::instance().get_group("optimization");
        ASSERT_NE(group, nullptr);

        std::set<std::string> registered_ids;
        for (const auto& meta : group->properties)
            ASSERT_TRUE(registered_ids.insert(meta.id).second) << "duplicate property id: " << meta.id;

        const std::map<std::string, std::string> json_to_property = {
            {"bilateral_grid_W", "bilateral_grid_w"},
            {"bilateral_grid_X", "bilateral_grid_x"},
            {"bilateral_grid_Y", "bilateral_grid_y"},
            {"ppisp_freeze_gaussians_on_distill", "ppisp_freeze_gaussians"},
            {"use_ppisp", "ppisp"},
        };

        const std::map<std::string, std::string> allowlist = {
            {"bg_color", "background color uses its dedicated Python binding"},
            {"bg_image_path", "background image path uses its dedicated Python binding"},
            {"enable_save_eval_images", "evaluation image output is not a registry property"},
            {"eval_steps", "vector-valued evaluation schedule is managed separately"},
            {"ppisp_lr", "PPISP optimizer tuning is not registered yet"},
            {"ppisp_reg_weight", "PPISP optimizer tuning is not registered yet"},
            {"ppisp_sidecar_path", "PPISP sidecar path uses its dedicated Python binding"},
            {"ppisp_warmup_steps", "PPISP optimizer tuning is not registered yet"},
            {"save_steps", "vector-valued save schedule is managed separately"},
        };

        std::map<std::string, std::string> property_to_json;
        for (const auto& [json_key, property_id] : json_to_property) {
            SCOPED_TRACE(json_key);
            EXPECT_TRUE(serialized.contains(json_key));
            EXPECT_TRUE(registered_ids.contains(property_id));
            EXPECT_TRUE(property_to_json.emplace(property_id, json_key).second);
        }

        for (const auto& [json_key, reason] : allowlist) {
            SCOPED_TRACE(json_key);
            EXPECT_TRUE(serialized.contains(json_key));
            EXPECT_FALSE(reason.empty());
            EXPECT_FALSE(registered_ids.contains(json_key));
            EXPECT_FALSE(json_to_property.contains(json_key));
        }

        for (auto it = serialized.begin(); it != serialized.end(); ++it) {
            const std::string& json_key = it.key();
            const auto renamed = json_to_property.find(json_key);
            const std::string& property_id = renamed == json_to_property.end() ? json_key : renamed->second;
            EXPECT_TRUE(registered_ids.contains(property_id) || allowlist.contains(json_key))
                << "serialized key has no registered property or allow-list reason: " << json_key;
        }

        for (const auto& property_id : registered_ids) {
            const auto renamed = property_to_json.find(property_id);
            const std::string& json_key = renamed == property_to_json.end() ? property_id : renamed->second;
            EXPECT_TRUE(serialized.contains(json_key))
                << "registered property has no serialized key: " << property_id;
        }
    }

    TEST_F(TrainingParametersTest, EvalBenchmarkConfigsParseAsIs) {
        const auto mcmc_path = eval_config_path("mcmc_optimization_params.json");
        EXPECT_EQ(frozen_config_fingerprint(mcmc_path), 0x5296bbd8725d137eULL);
        const auto mcmc_result = lfs::core::param::read_optim_params_from_json(mcmc_path);
        ASSERT_TRUE(mcmc_result.has_value()) << mcmc_result.error();
        EXPECT_FLOAT_EQ(mcmc_result->opacity_lr, 0.0335f);
        EXPECT_FLOAT_EQ(mcmc_result->shs_lr, 0.0024f);
        EXPECT_FLOAT_EQ(mcmc_result->opacity_reg, 0.0042f);
        EXPECT_EQ(mcmc_result->strategy, "mcmc");
        EXPECT_EQ(mcmc_result->max_cap, 1'000'000);

        const auto mrnf_path = eval_config_path("mrnf_optimization_params.json");
        EXPECT_EQ(frozen_config_fingerprint(mrnf_path), 0x40c65afdecde5828ULL);
        const auto mrnf_result = lfs::core::param::read_optim_params_from_json(mrnf_path);
        ASSERT_TRUE(mrnf_result.has_value()) << mrnf_result.error();
        EXPECT_FLOAT_EQ(mrnf_result->means_lr, 0.000128f);
        EXPECT_EQ(mrnf_result->start_refine, 500u);
        EXPECT_EQ(mrnf_result->stop_refine, 28'500u);
        EXPECT_FLOAT_EQ(mrnf_result->min_opacity, 0.0039215689f);
        EXPECT_TRUE(mrnf_result->revised_opacity);

        const auto igs_path = eval_config_path("improvedGSplus_optimization_params.json");
        EXPECT_EQ(frozen_config_fingerprint(igs_path), 0x2cf8daf2e3da1198ULL);
        const auto igs_result = lfs::core::param::read_optim_params_from_json(igs_path);
        ASSERT_TRUE(igs_result.has_value()) << igs_result.error();
        EXPECT_FLOAT_EQ(igs_result->init_opacity, 0.3f);
        EXPECT_FLOAT_EQ(igs_result->init_scaling, 0.2f);
        EXPECT_EQ(igs_result->refine_every, 500u);
        EXPECT_FLOAT_EQ(igs_result->tv_loss_weight, 5.0f);
        EXPECT_EQ(igs_result->strategy, "igs+");
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

        const auto* group = PropertyRegistry::instance().get_group("optimization");
        ASSERT_NE(group, nullptr);

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

} // namespace

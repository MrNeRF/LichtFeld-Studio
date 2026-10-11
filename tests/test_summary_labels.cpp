/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/utils/summary_labels.hpp"
#include <array>
#include <filesystem>
#include <gtest/gtest.h>

namespace lfs::vis::gui {
    namespace {
        class SummaryLabelsTest : public ::testing::Test {
        protected:
            void SetUp() override {
                ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().initialize(
                    (std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/resources/locales").string()));
            }
            void TearDown() override { lfs::event::LocalizationManager::getInstance().reset(); }
        };

        TEST_F(SummaryLabelsTest, TranslatesSceneCountsOnLanguageSwitch) {
            struct Labels {
                const char* language;
                const char* model;
                const char* node;
                const char* models;
                const char* nodes;
            };
            constexpr std::array labels{
                Labels{"de", "1 Modell", "1 Knoten", "2 Modelle", "2 Knoten"},
                Labels{"es", "1 modelo", "1 nodo", "2 modelos", "2 nodos"},
                Labels{"fr", "1 modèle", "1 nœud", "2 modèles", "2 nœuds"},
                Labels{"it", "1 modello", "1 nodo", "2 modelli", "2 nodi"},
                Labels{"ja", "1 モデル", "1 ノード", "2 モデル", "2 ノード"},
                Labels{"ko", "1 모델", "1 노드", "2 모델", "2 노드"},
                Labels{"nl", "1 model", "1 knooppunt", "2 modellen", "2 knooppunten"},
                Labels{"pl", "1 model", "1 węzeł", "2 modele", "2 węzły"},
                Labels{"zh", "1 个模型", "1 个节点", "2 个模型", "2 个节点"},
                Labels{"en", "1 model", "1 node", "2 models", "2 nodes"}};
            for (const auto& label : labels) {
                SCOPED_TRACE(label.language);
                ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().setLanguage(label.language));
                EXPECT_EQ(formatSceneModelCount(1), label.model);
                EXPECT_EQ(formatSceneNodeCount(1), label.node);
                EXPECT_EQ(formatSceneModelCount(2), label.models);
                EXPECT_EQ(formatSceneNodeCount(2), label.nodes);
            }
        }

        TEST_F(SummaryLabelsTest, PreservesEnglishCounts) {
            ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().setLanguage("en"));
            for (size_t count : {0, 1, 2, 5, 1000, 1000000}) {
                const auto suffix = count == 1 ? "" : "s";
                EXPECT_EQ(formatSceneModelCount(count), std::format("{} model{}", lfs::core::format_count(count), suffix));
                EXPECT_EQ(formatSceneNodeCount(count), std::format("{} node{}", lfs::core::format_count(count), suffix));
            }
        }

        TEST_F(SummaryLabelsTest, PolishCountInflection) {
            ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().setLanguage("pl"));
            for (size_t count : {2, 3, 4, 22, 23, 24, 102}) {
                EXPECT_EQ(formatSceneModelCount(count), std::format("{} modele", count));
                EXPECT_EQ(formatSceneNodeCount(count), std::format("{} węzły", count));
            }
            for (size_t count : {0, 5, 11, 12, 13, 14, 21, 101, 112}) {
                EXPECT_EQ(formatSceneModelCount(count), std::format("{} modeli", count));
                EXPECT_EQ(formatSceneNodeCount(count), std::format("{} węzłów", count));
            }
        }

        TEST_F(SummaryLabelsTest, RemovesTrailingColonInEveryLocale) {
            auto& locale = lfs::event::LocalizationManager::getInstance();
            for (const auto& language : locale.getAvailableLanguages()) {
                SCOPED_TRACE(language);
                ASSERT_TRUE(locale.setLanguage(language));
                const auto label = stripLabelColon(LOC("status.gaussians"));
                EXPECT_FALSE(label.ends_with(':'));
                EXPECT_FALSE(label.ends_with("："));
                EXPECT_FALSE(label.ends_with(' '));
                EXPECT_FALSE(label.empty());
                if (language == "zh")
                    EXPECT_EQ(label, "高斯");
            }
        }

        TEST_F(SummaryLabelsTest, PreservesAsciiAndInteriorPunctuation) {
            for (const std::string input : {"", ": ", "Gaussians:", "Gaussians:  ", "Steps", "WASD: move:", "缩放", "A：B:"}) {
                const auto end = input.find_last_not_of(": ");
                EXPECT_EQ(stripLabelColon(input), end == std::string::npos ? input : input.substr(0, end + 1));
            }
            EXPECT_EQ(stripLabelColon("高斯：  "), "高斯");
            EXPECT_EQ(stripLabelColon("A：B："), "A：B");
        }
    } // namespace
} // namespace lfs::vis::gui

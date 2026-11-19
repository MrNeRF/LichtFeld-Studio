/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <nvimgcodec.h>

#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "imgproc/exception.h"

namespace nvimgcodec {

class PluginSelection
{
  public:
    struct Rule
    {
        std::optional<std::string> codec_name;
        std::vector<std::string> plugin_ids;
    };

    explicit PluginSelection(const nvimgcodecExecutionParams_t* exec_params)
    {
        if (exec_params->num_selections < 0) {
            throw Exception(INVALID_PARAMETER, "num_selections must not be negative");
        }
        if (exec_params->num_selections == 0) {
            if (exec_params->selections != nullptr) {
                throw Exception(INVALID_PARAMETER, "selections must be null when num_selections is zero");
            }
            return;
        }
        if (exec_params->selections == nullptr) {
            throw Exception(INVALID_PARAMETER, "selections must not be null when num_selections is positive");
        }

        rules_.reserve(exec_params->num_selections);
        std::unordered_set<std::string> codec_names;
        bool has_global_rule = false;
        for (int selection_idx = 0; selection_idx < exec_params->num_selections; ++selection_idx) {
            const auto& selection = exec_params->selections[selection_idx];
            if (selection.struct_type != NVIMGCODEC_STRUCTURE_TYPE_PLUGIN_SELECTION ||
                selection.struct_size != sizeof(nvimgcodecPluginSelection_t)) {
                throw Exception(INVALID_PARAMETER, "Invalid plugin selection structure");
            }
            if (selection.num_plugin_ids <= 0 || selection.plugin_ids == nullptr) {
                throw Exception(INVALID_PARAMETER, "Each plugin selection must contain at least one plugin identifier");
            }

            Rule rule;
            if (selection.codec_name == nullptr) {
                has_global_rule = true;
            } else {
                if (selection.codec_name[0] == '\0') {
                    throw Exception(INVALID_PARAMETER, "codec_name must not be empty");
                }
                rule.codec_name = selection.codec_name;
                if (!codec_names.insert(*rule.codec_name).second) {
                    throw Exception(INVALID_PARAMETER, "Duplicate codec selection: " + *rule.codec_name);
                }
            }

            std::unordered_set<std::string> plugin_ids;
            rule.plugin_ids.reserve(selection.num_plugin_ids);
            for (int plugin_idx = 0; plugin_idx < selection.num_plugin_ids; ++plugin_idx) {
                const char* plugin_id = selection.plugin_ids[plugin_idx];
                if (plugin_id == nullptr || plugin_id[0] == '\0') {
                    throw Exception(INVALID_PARAMETER, "Plugin identifiers must not be null or empty");
                }
                if (!plugin_ids.insert(plugin_id).second) {
                    throw Exception(INVALID_PARAMETER, "Duplicate plugin identifier: " + std::string(plugin_id));
                }
                rule.plugin_ids.emplace_back(plugin_id);
            }
            rules_.push_back(std::move(rule));
        }

        if (has_global_rule && rules_.size() != 1) {
            throw Exception(INVALID_PARAMETER, "A global plugin selection cannot be combined with codec selections");
        }
    }

    bool specified() const { return !rules_.empty(); }

    bool global() const { return specified() && !rules_[0].codec_name.has_value(); }

    const std::vector<Rule>& rules() const { return rules_; }

  private:
    std::vector<Rule> rules_;
};

} // namespace nvimgcodec

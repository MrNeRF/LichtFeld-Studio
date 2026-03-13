/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lfs::mcp {

    enum class LLMProvider { Anthropic,
                             OpenAI };

    struct ImageAttachment {
        std::string base64_data;
        std::string media_type = "image/png";
    };

    struct TextAttachment {
        std::string content;
        std::string label;
    };

    using Attachment = std::variant<ImageAttachment, TextAttachment>;

    struct LLMRequest {
        std::string prompt;
        std::vector<Attachment> attachments;
        std::optional<std::string> system_prompt;
        int max_tokens = 4096;
        float temperature = 0.7f;
    };

    struct LLMResponse {
        std::string content;
        std::string model;
        int input_tokens = 0;
        int output_tokens = 0;
        bool success = true;
        std::string error;
    };

    class LFS_MCP_API LLMClient {
    public:
        LLMClient();
        ~LLMClient();

        void set_provider(LLMProvider provider);
        void set_api_key(const std::string& key);
        void set_model(const std::string& model);
        void set_base_url(const std::string& url);

        std::expected<LLMResponse, std::string> complete(const LLMRequest& request);

        std::future<LLMResponse> complete_async(const LLMRequest& request);

        bool is_configured() const;

        static std::expected<std::string, std::string> load_api_key_from_env();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    LFS_MCP_API std::expected<LLMResponse, std::string> ask_training_advisor(
        LLMClient& client,
        int iteration,
        float loss,
        std::size_t num_gaussians,
        const std::string& base64_render,
        const std::string& problem_description);

} // namespace lfs::mcp

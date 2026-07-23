/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "llm_client.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib/httplib.h"

#include <nlohmann/json.hpp>

#include <cstdlib>

namespace lfs::mcp {

    using json = nlohmann::json;

    struct LLMClient::Impl {
        LLMProvider provider = LLMProvider::Anthropic;
        std::string api_key;
        std::string model = "claude-sonnet-4-20250514";
        std::string base_url;

        std::string get_base_url() const {
            if (!base_url.empty()) {
                return base_url;
            }
            switch (provider) {
            case LLMProvider::Anthropic:
                return "https://api.anthropic.com";
            case LLMProvider::OpenAI:
                return "https://api.openai.com";
            }
            return "https://api.anthropic.com";
        }

        json build_anthropic_request(const LLMRequest& req) const {
            json messages = json::array();
            json user_content = json::array();

            for (const auto& attachment : req.attachments) {
                if (std::holds_alternative<ImageAttachment>(attachment)) {
                    const auto& img = std::get<ImageAttachment>(attachment);
                    user_content.push_back(json{
                        {"type", "image"},
                        {"source", json{
                                       {"type", "base64"},
                                       {"media_type", img.media_type},
                                       {"data", img.base64_data}}}});
                } else if (std::holds_alternative<TextAttachment>(attachment)) {
                    const auto& txt = std::get<TextAttachment>(attachment);
                    std::string formatted = txt.label.empty()
                                                ? txt.content
                                                : "[" + txt.label + "]\n" + txt.content;
                    user_content.push_back(json{{"type", "text"}, {"text", formatted}});
                }
            }

            user_content.push_back(json{{"type", "text"}, {"text", req.prompt}});

            messages.push_back(json{{"role", "user"}, {"content", user_content}});

            json body;
            body["model"] = model;
            body["max_tokens"] = req.max_tokens;
            body["messages"] = messages;

            if (req.system_prompt) {
                body["system"] = *req.system_prompt;
            }

            if (req.temperature >= 0.0f && req.temperature <= 1.0f) {
                body["temperature"] = req.temperature;
            }

            return body;
        }

        json build_openai_request(const LLMRequest& req) const {
            json messages = json::array();

            if (req.system_prompt) {
                messages.push_back(json{{"role", "system"}, {"content", *req.system_prompt}});
            }

            json user_content = json::array();

            for (const auto& attachment : req.attachments) {
                if (std::holds_alternative<ImageAttachment>(attachment)) {
                    const auto& img = std::get<ImageAttachment>(attachment);
                    std::string data_url = "data:" + img.media_type + ";base64," + img.base64_data;
                    user_content.push_back(json{
                        {"type", "image_url"},
                        {"image_url", json{{"url", data_url}}}});
                } else if (std::holds_alternative<TextAttachment>(attachment)) {
                    const auto& txt = std::get<TextAttachment>(attachment);
                    std::string formatted = txt.label.empty()
                                                ? txt.content
                                                : "[" + txt.label + "]\n" + txt.content;
                    user_content.push_back(json{{"type", "text"}, {"text", formatted}});
                }
            }

            user_content.push_back(json{{"type", "text"}, {"text", req.prompt}});

            messages.push_back(json{{"role", "user"}, {"content", user_content}});

            json body;
            body["model"] = model;
            body["max_tokens"] = req.max_tokens;
            body["messages"] = messages;

            if (req.temperature >= 0.0f && req.temperature <= 2.0f) {
                body["temperature"] = req.temperature;
            }

            return body;
        }

        LLMResponse parse_anthropic_response(const std::string& body) const {
            LLMResponse resp;
            try {
                auto j = json::parse(body);

                if (j.contains("error")) {
                    resp.success = false;
                    resp.error = j["error"].value("message", "Unknown error");
                    return resp;
                }

                resp.model = j.value("model", "");
                resp.input_tokens = j.value("usage", json{}).value("input_tokens", 0);
                resp.output_tokens = j.value("usage", json{}).value("output_tokens", 0);

                if (j.contains("content") && j["content"].is_array()) {
                    for (const auto& block : j["content"]) {
                        if (block.value("type", "") == "text") {
                            resp.content += block.value("text", "");
                        }
                    }
                }

                resp.success = true;
            } catch (const std::exception& e) {
                resp.success = false;
                resp.error = std::string("Failed to parse response: ") + e.what();
            }
            return resp;
        }

        LLMResponse parse_openai_response(const std::string& body) const {
            LLMResponse resp;
            try {
                auto j = json::parse(body);

                if (j.contains("error")) {
                    resp.success = false;
                    resp.error = j["error"].value("message", "Unknown error");
                    return resp;
                }

                resp.model = j.value("model", "");
                resp.input_tokens = j.value("usage", json{}).value("prompt_tokens", 0);
                resp.output_tokens = j.value("usage", json{}).value("completion_tokens", 0);

                if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                    const auto& choice = j["choices"][0];
                    if (choice.contains("message")) {
                        resp.content = choice["message"].value("content", "");
                    }
                }

                resp.success = true;
            } catch (const std::exception& e) {
                resp.success = false;
                resp.error = std::string("Failed to parse response: ") + e.what();
            }
            return resp;
        }

        std::expected<LLMResponse, std::string> do_request(const LLMRequest& req) {
            if (api_key.empty()) {
                return std::unexpected("API key not set");
            }

            std::string url = get_base_url();
            std::string path;
            json body;
            httplib::Headers headers;

            switch (provider) {
            case LLMProvider::Anthropic:
                path = "/v1/messages";
                body = build_anthropic_request(req);
                headers = {
                    {"x-api-key", api_key},
                    {"anthropic-version", "2023-06-01"},
                    {"content-type", "application/json"}};
                break;

            case LLMProvider::OpenAI:
                path = "/v1/chat/completions";
                body = build_openai_request(req);
                headers = {
                    {"Authorization", "Bearer " + api_key},
                    {"content-type", "application/json"}};
                break;
            }

            httplib::Client client(url);
            client.set_connection_timeout(30);
            client.set_read_timeout(120);

            std::string body_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
            auto result = client.Post(path, headers, body_str, "application/json");

            if (!result) {
                return std::unexpected("HTTP request failed: " + httplib::to_string(result.error()));
            }

            if (result->status != 200) {
                return std::unexpected("HTTP " + std::to_string(result->status) + ": " + result->body);
            }

            switch (provider) {
            case LLMProvider::Anthropic:
                return parse_anthropic_response(result->body);
            case LLMProvider::OpenAI:
                return parse_openai_response(result->body);
            }

            return std::unexpected("Unknown provider");
        }
    };

    LLMClient::LLMClient()
        : impl_(std::make_unique<Impl>()) {}

    LLMClient::~LLMClient() = default;

    void LLMClient::set_provider(LLMProvider provider) {
        impl_->provider = provider;
        if (provider == LLMProvider::Anthropic && impl_->model.find("gpt") != std::string::npos) {
            impl_->model = "claude-sonnet-4-20250514";
        } else if (provider == LLMProvider::OpenAI && impl_->model.find("claude") != std::string::npos) {
            impl_->model = "gpt-4o";
        }
    }

    void LLMClient::set_api_key(const std::string& key) { impl_->api_key = key; }

    void LLMClient::set_model(const std::string& model) { impl_->model = model; }

    void LLMClient::set_base_url(const std::string& url) { impl_->base_url = url; }

    std::expected<LLMResponse, std::string> LLMClient::complete(const LLMRequest& request) {
        return impl_->do_request(request);
    }

    std::future<LLMResponse> LLMClient::complete_async(const LLMRequest& request) {
        return std::async(std::launch::async, [this, request]() {
            auto result = impl_->do_request(request);
            if (result) {
                return *result;
            }
            LLMResponse error_resp;
            error_resp.success = false;
            error_resp.error = result.error();
            return error_resp;
        });
    }

    bool LLMClient::is_configured() const { return !impl_->api_key.empty(); }

    std::expected<std::string, std::string> LLMClient::load_api_key_from_env() {
        if (const char* key = std::getenv("ANTHROPIC_API_KEY")) {
            return std::string(key);
        }
        if (const char* key = std::getenv("OPENAI_API_KEY")) {
            return std::string(key);
        }
        return std::unexpected("No API key found in environment (ANTHROPIC_API_KEY or OPENAI_API_KEY)");
    }

    std::expected<LLMResponse, std::string> ask_training_advisor(
        LLMClient& client,
        int iteration,
        float loss,
        std::size_t num_gaussians,
        const std::string& base64_render,
        const std::string& problem_description) {

        LLMRequest req;
        req.system_prompt = R"(You are an expert in 3D Gaussian Splatting training. You help users optimize their training runs by analyzing training state, rendered output, and suggesting parameter adjustments.

When giving advice:
1. Be specific about what parameters to change and by how much
2. Explain why the change should help
3. Consider the current training iteration and loss trend
4. Look at the rendered image for artifacts that indicate specific problems

Common issues and solutions:
- Floaters: Lower opacity learning rate, increase opacity reset frequency
- Blurry regions: Increase densification gradient threshold, check SH degree
- Over-splitting: Reduce split threshold, increase cull threshold
- Loss plateau: Adjust learning rate schedule, check if densification is working
- Memory issues: Reduce max gaussians, increase culling frequency)";

        req.attachments.push_back(TextAttachment{
            .content = "Iteration: " + std::to_string(iteration) + "\n"
                                                                   "Loss: " +
                       std::to_string(loss) + "\n"
                                              "Gaussians: " +
                       std::to_string(num_gaussians),
            .label = "Training State"});

        if (!base64_render.empty()) {
            req.attachments.push_back(ImageAttachment{
                .base64_data = base64_render,
                .media_type = "image/png"});
        }

        req.prompt = problem_description.empty()
                         ? "Please analyze the current training state and rendered output. What adjustments would you recommend?"
                         : problem_description;

        req.temperature = 0.3f;

        return client.complete(req);
    }

} // namespace lfs::mcp

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_document_utils.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/rmlui/rml_path_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "internal/resource_paths.hpp"

#include <RmlUi/Core.h>

#include <format>
#include <fstream>
#include <iterator>
#include <regex>

namespace lfs::vis::gui::rml_documents {

    namespace {
        constexpr std::string_view kFallbackFontResource = "rmlui/font_fallback.rcss";

        std::string injectParseTimeFontFallback(std::string document_rml,
                                                const std::filesystem::path& fallback_rcss_path) {
            const std::string fallback_href = rml_paths::filesystemPathToFileUri(fallback_rcss_path);
            if (document_rml.find(fallback_href) != std::string::npos ||
                document_rml.find(kFallbackFontResource) != std::string::npos) {
                return document_rml;
            }

            const std::string fallback_link = std::format(
                "\n  <link type=\"text/rcss\" href=\"{}\"/>\n", fallback_href);
            if (const auto head_pos = document_rml.find("<head>"); head_pos != std::string::npos) {
                document_rml.insert(head_pos + std::string_view("<head>").size(), fallback_link);
                return document_rml;
            }

            const std::string head_block = std::format(
                "<head>\n  <link type=\"text/rcss\" href=\"{}\"/>\n</head>\n", fallback_href);
            if (const auto body_pos = document_rml.find("<body"); body_pos != std::string::npos) {
                document_rml.insert(body_pos, head_block);
                return document_rml;
            }

            if (const auto rml_pos = document_rml.find("<rml>"); rml_pos != std::string::npos) {
                document_rml.insert(rml_pos + std::string_view("<rml>").size(),
                                    std::string("\n") + head_block);
            }
            return document_rml;
        }

        std::optional<std::string> resolveRelativeImageSource(
            const std::string_view source,
            const std::filesystem::path& document_path) {
            if (source.empty() || rml_paths::hasUriScheme(source) ||
                rml_paths::pathReferenceToFilesystemPath(source)) {
                return std::nullopt;
            }

            const auto resolved_path =
                (document_path.parent_path() / lfs::core::utf8_to_path(std::string(source)))
                    .lexically_normal();
            if (!std::filesystem::exists(resolved_path)) {
                std::string asset_source(source);
                while (asset_source.rfind("../", 0) == 0)
                    asset_source.erase(0, 3);
                if (!asset_source.empty()) {
                    try {
                        const auto asset_path = lfs::vis::getAssetPath(asset_source);
                        if (std::filesystem::exists(asset_path))
                            return rml_theme::pathToRmlImageSource(asset_path);
                    } catch (const std::exception& e) {
                        LOG_DEBUG("RmlUI image source fallback could not resolve asset '{}': {}",
                                  asset_source,
                                  e.what());
                    }
                }
                return std::nullopt;
            }

            return rml_theme::pathToRmlImageSource(resolved_path);
        }

        std::optional<std::filesystem::path> resolveStylesheetHref(
            const std::string_view href,
            const std::filesystem::path& document_path) {
            if (href.empty())
                return std::nullopt;

            std::string reference(href);
            if (const auto suffix_pos = reference.find_first_of("?#");
                suffix_pos != std::string::npos) {
                reference.erase(suffix_pos);
            }
            if (reference.empty())
                return std::nullopt;

            if (const auto filesystem_path =
                    rml_paths::pathReferenceToFilesystemPath(reference)) {
                return filesystem_path->lexically_normal();
            }

            if (rml_paths::hasUriScheme(reference))
                return std::nullopt;

            return (document_path.parent_path() /
                    lfs::core::utf8_to_path(rml_paths::percentDecode(reference)))
                .lexically_normal();
        }
    } // namespace

    std::string rewriteRelativeImageSources(std::string document_rml,
                                            const std::filesystem::path& document_path) {
        static const std::regex kImgSrcPattern(
            R"((<img\b[^>]*\bsrc\s*=\s*)(["'])([^"']*)(["']))",
            std::regex_constants::icase);

        std::string rewritten;
        std::size_t last_pos = 0;
        for (std::sregex_iterator it(document_rml.begin(), document_rml.end(), kImgSrcPattern), end;
             it != end; ++it) {
            const auto& match = *it;
            const auto match_pos = static_cast<std::size_t>(match.position());
            const auto match_end = match_pos + static_cast<std::size_t>(match.length());
            rewritten.append(document_rml, last_pos, match_pos - last_pos);

            rewritten += match[1].str();
            rewritten += match[2].str();

            const std::string original_source = match[3].str();
            if (const auto resolved_source = resolveRelativeImageSource(original_source, document_path)) {
                rewritten += *resolved_source;
            } else {
                rewritten += original_source;
            }

            rewritten += match[4].str();
            last_pos = match_end;
        }

        if (rewritten.empty())
            return document_rml;

        rewritten.append(document_rml, last_pos, std::string::npos);
        return rewritten;
    }

    std::vector<std::filesystem::path> linkedStylesheetPaths(
        const std::string& document_rml,
        const std::filesystem::path& document_path) {
        static const std::regex kLinkTagPattern(R"(<\s*link\b[^>]*>)",
                                                std::regex_constants::icase);
        static const std::regex kRcssTypePattern(R"(\btype\s*=\s*(["'])text/rcss\1)",
                                                 std::regex_constants::icase);
        static const std::regex kHrefPattern(R"(\bhref\s*=\s*(["'])([^"']*)\1)",
                                             std::regex_constants::icase);

        std::vector<std::filesystem::path> paths;
        for (std::sregex_iterator it(document_rml.begin(), document_rml.end(), kLinkTagPattern),
             end;
             it != end; ++it) {
            const std::string tag = it->str();
            if (!std::regex_search(tag, kRcssTypePattern))
                continue;

            std::smatch href_match;
            if (!std::regex_search(tag, href_match, kHrefPattern))
                continue;

            if (const auto resolved_path =
                    resolveStylesheetHref(href_match[2].str(), document_path)) {
                paths.push_back(*resolved_path);
            }
        }

        return paths;
    }

    std::vector<std::filesystem::path> loadLinkedStylesheetPaths(
        const std::filesystem::path& document_path) {
        std::ifstream input(document_path, std::ios::binary);
        if (!input)
            return {};

        std::string document_rml{std::istreambuf_iterator<char>(input),
                                 std::istreambuf_iterator<char>()};
        return linkedStylesheetPaths(document_rml, document_path);
    }

    namespace {
        std::string preprocessDocumentSource(std::string document_rml,
                                             const std::filesystem::path& document_path) {
            document_rml = rewriteRelativeImageSources(std::move(document_rml), document_path);
            const auto fallback_rcss_path =
                lfs::vis::getAssetPath(std::string(kFallbackFontResource));
            return injectParseTimeFontFallback(std::move(document_rml), fallback_rcss_path);
        }

        std::optional<std::string> loadDocumentSource(const std::filesystem::path& document_path) {
            std::ifstream input(document_path, std::ios::binary);
            if (!input)
                return std::nullopt;

            std::string document_rml{std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>()};
            return preprocessDocumentSource(std::move(document_rml), document_path);
        }
    } // namespace

    Rml::ElementDocument* loadDocument(Rml::Context* const context,
                                       const std::filesystem::path& document_path) {
        if (!context)
            return nullptr;

        if (auto document_source = loadDocumentSource(document_path)) {
            return context->LoadDocumentFromMemory(*document_source,
                                                   rml_paths::filesystemPathToFileUri(document_path));
        }

        return context->LoadDocument(rml_paths::filesystemPathToFileUri(document_path));
    }

} // namespace lfs::vis::gui::rml_documents

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis::gui::rml_documents {

    LFS_VIS_API std::string rewriteRelativeImageSources(std::string document_rml,
                                                        const std::filesystem::path& document_path);
    LFS_VIS_API std::vector<std::filesystem::path> linkedStylesheetPaths(
        const std::string& document_rml,
        const std::filesystem::path& document_path);
    LFS_VIS_API std::vector<std::filesystem::path> loadLinkedStylesheetPaths(
        const std::filesystem::path& document_path);
    LFS_VIS_API Rml::ElementDocument* loadDocument(Rml::Context* context,
                                                   const std::filesystem::path& document_path);

} // namespace lfs::vis::gui::rml_documents

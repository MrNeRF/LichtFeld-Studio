// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "image_metadata.hpp"
#include "media/media_ingest.hpp"
#include "media_backends.hpp"
#include <nlohmann/json.hpp>

namespace lfs::media::detail {
    // Extraction checkpoint policy, separate from manifest publication. Codecs,
    // selection and atomic publication remain owned by their existing services.
    class ResumableFileSink final : public FrameSink {
    public:
        ResumableFileSink(const IngestRequest&, const FileExtraction&);
        SinkResult begin(const SinkSession&) override;
        SinkResult write(const FrameView&) override;
        SinkResult prepareComplete(const SinkSummary&) override;
        std::string_view imageEncoderBackend() const noexcept override { return gpu_jpeg_ ? "nvimagecodec_cuda" : "cpu"; }
        SinkResult complete(const SinkSummary&) override;
        void abort(const SinkSummary&) noexcept override;

    private:
        const IngestRequest& request_;
        FileExtraction output_;
        std::function<bool()> cancelled_;
        std::filesystem::path checkpoint_;
        nlohmann::json identity_;
        std::unique_ptr<GpuJpegEncoder> gpu_jpeg_;
        std::size_t accepted_ = 0;
        std::size_t recorded_ = 0;
        std::unordered_set<std::filesystem::path> filenames_;
        bool active_ = false;
        int applied_rotation_ = 0;
        std::shared_ptr<const SourceMetadataCatalog> source_metadata_;
        std::shared_ptr<const EmbeddedImageMetadata> embedded_metadata_;
    };
} // namespace lfs::media::detail

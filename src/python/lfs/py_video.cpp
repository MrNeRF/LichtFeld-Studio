/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_video.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/video/video_metadata.hpp"
#include "io/video_frame_extractor.hpp"

#include <nanobind/stl/string.h>

#include <exception>
#include <stdexcept>
#include <string>

namespace lfs::python {

    namespace {

        // Progress and cancel callbacks run on the extractor's thread while the
        // calling Python thread has released the GIL, so each call reacquires it.
        struct PyProgress {
            nb::object callback;

            [[nodiscard]] bool valid() const { return callback.is_valid() && !callback.is_none(); }

            void operator()(const int current, const int total, const int discarded) const {
                if (!valid())
                    return;
                nb::gil_scoped_acquire gil;
                try {
                    callback(current, total, discarded);
                } catch (const std::exception& e) {
                    LOG_ERROR("lf.video progress callback error: {}", e.what());
                }
            }
        };

        struct PyCancel {
            nb::object callback;

            [[nodiscard]] bool valid() const { return callback.is_valid() && !callback.is_none(); }

            bool operator()() const {
                if (!valid())
                    return false;
                nb::gil_scoped_acquire gil;
                try {
                    return nb::cast<bool>(callback());
                } catch (const std::exception& e) {
                    LOG_ERROR("lf.video cancel callback error: {}", e.what());
                    return true;
                }
            }
        };

        const char* outcome_name(const lfs::io::ExtractionOutcome outcome) {
            switch (outcome) {
            case lfs::io::ExtractionOutcome::Completed: return "completed";
            case lfs::io::ExtractionOutcome::Cancelled: return "cancelled";
            case lfs::io::ExtractionOutcome::Failed: return "failed";
            }
            return "failed";
        }

    } // namespace

    void register_video(nb::module_& m) {
        m.def(
            "probe",
            [](const std::string& path) {
                lfs::io::video::VideoMetadata meta;
                std::string error;
                bool ok = false;
                {
                    nb::gil_scoped_release release;
                    ok = lfs::io::video::probeVideoFileMetadata(lfs::core::utf8_to_path(path), meta, error);
                }
                if (!ok)
                    throw std::runtime_error(error);
                nb::dict result;
                result["width"] = meta.width;
                result["height"] = meta.height;
                result["fps"] = meta.fps;
                result["duration"] = meta.duration;
                result["total_frames"] = static_cast<long long>(meta.total_frames);
                result["rotation"] = meta.rotation;
                result["spherical"] = meta.spherical;
                result["is_hdr"] = meta.is_hdr;
                return result;
            },
            nb::arg("path"),
            "Read container metadata without decoding: width, height, fps, duration, total_frames, rotation, "
            "spherical, is_hdr. Raises RuntimeError when the file cannot be opened.");

        m.def(
            "extract_frames",
            [](const std::string& video_path, const std::string& output_dir, const double fps,
               const int jpg_quality, const float scale, const bool sharpness_enabled,
               const double sharpness_threshold, const int rotation, const double start_time,
               const double end_time, const bool convert_hdr_to_sdr, const std::string& filename_pattern,
               nb::object on_progress, nb::object should_cancel) {
                lfs::io::VideoFrameExtractor::Params params;
                params.video_path = lfs::core::utf8_to_path(video_path);
                params.output_dir = lfs::core::utf8_to_path(output_dir);
                params.mode = lfs::io::ExtractionMode::FPS;
                params.fps = fps;
                params.format = lfs::io::ImageFormat::JPG;
                params.jpg_quality = jpg_quality;
                if (scale > 0.0f && scale < 1.0f) {
                    params.resolution_mode = lfs::io::ResolutionMode::Scale;
                    params.scale = scale;
                }
                params.sharpness.enabled = sharpness_enabled;
                params.sharpness.threshold = sharpness_threshold;
                params.sharpness.algorithm = lfs::io::SharpnessAlgorithm::COMBINED;
                params.rotation = rotation;
                params.start_time = start_time;
                params.end_time = end_time;
                params.convert_hdr_to_sdr = convert_hdr_to_sdr;
                params.filename_pattern = filename_pattern;
                params.generate_metadata = true;

                const PyProgress progress{on_progress};
                const PyCancel cancel{should_cancel};
                params.progress_callback = [progress](const int c, const int t, const int d) { progress(c, t, d); };
                params.cancel_requested = [cancel]() { return cancel(); };

                std::string error;
                lfs::io::ExtractionOutcome outcome = lfs::io::ExtractionOutcome::Failed;
                {
                    nb::gil_scoped_release release;
                    lfs::io::VideoFrameExtractor extractor;
                    extractor.extract(params, error);
                    outcome = extractor.lastOutcome();
                }
                nb::dict result;
                result["outcome"] = outcome_name(outcome);
                result["error"] = error;
                return result;
            },
            nb::arg("video_path"), nb::arg("output_dir"), nb::arg("fps") = 2.0, nb::arg("jpg_quality") = 95,
            nb::arg("scale") = 1.0f, nb::arg("sharpness_enabled") = true, nb::arg("sharpness_threshold") = 10.0,
            nb::arg("rotation") = 0, nb::arg("start_time") = 0.0, nb::arg("end_time") = -1.0,
            nb::arg("convert_hdr_to_sdr") = true, nb::arg("filename_pattern") = "frame_%06d",
            nb::arg("on_progress") = nb::none(), nb::arg("should_cancel") = nb::none(),
            "Extract JPEG frames at a fixed rate. Blocks with the GIL released; on_progress(current, total, "
            "discarded) and should_cancel() are called from the extraction thread. Returns {outcome, error} "
            "with outcome one of completed, cancelled, failed.");
    }

} // namespace lfs::python

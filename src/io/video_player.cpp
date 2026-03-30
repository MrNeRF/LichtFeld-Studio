/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_player.hpp"
#include "core/include/core/logger.hpp"
#include "core/path_utils.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace lfs::io {

    namespace {

        constexpr size_t FRAME_QUEUE_SIZE = 16;
        constexpr int MAX_PREVIEW_HEIGHT = 720;
        constexpr size_t MIN_BUFFERED_FRAMES = 4;
        constexpr int MAX_SW_DECODE_THREADS = 4;

        const char* getHwDecoderName(const AVCodecID codec_id) {
            switch (codec_id) {
            case AV_CODEC_ID_H264:
                return "h264_cuvid";
            case AV_CODEC_ID_HEVC:
                return "hevc_cuvid";
            case AV_CODEC_ID_VP8:
                return "vp8_cuvid";
            case AV_CODEC_ID_VP9:
                return "vp9_cuvid";
            case AV_CODEC_ID_AV1:
                return "av1_cuvid";
            case AV_CODEC_ID_MPEG1VIDEO:
                return "mpeg1_cuvid";
            case AV_CODEC_ID_MPEG2VIDEO:
                return "mpeg2_cuvid";
            case AV_CODEC_ID_MPEG4:
                return "mpeg4_cuvid";
            case AV_CODEC_ID_VC1:
                return "vc1_cuvid";
            default:
                return nullptr;
            }
        }

        AVPixelFormat getHwFormat(AVCodecContext*, const AVPixelFormat* pix_fmts) {
            for (const AVPixelFormat* p = pix_fmts; *p != -1; p++) {
                if (*p == AV_PIX_FMT_CUDA) {
                    return *p;
                }
            }
            return AV_PIX_FMT_NONE;
        }

    } // namespace

    struct DecodedFrame {
        std::vector<uint8_t> data;
        double pts = 0;
        int64_t frame_number = 0;
    };

    class VideoPlayer::Impl {
    public:
        ~Impl() { close(); }

        bool open(const std::filesystem::path& path) {
            close();

            const std::string path_utf8 = lfs::core::path_to_utf8(path);

            if (avformat_open_input(&fmt_ctx_, path_utf8.c_str(), nullptr, nullptr) < 0) {
                LOG_WARN("VideoPlayer: Failed to open {}", path_utf8);
                return false;
            }

            if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
                close();
                return false;
            }

            for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
                if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                    video_stream_idx_ = static_cast<int>(i);
                    break;
                }
            }

            if (video_stream_idx_ < 0) {
                close();
                return false;
            }

            AVStream* const stream = fmt_ctx_->streams[video_stream_idx_];
            const AVCodecID codec_id = stream->codecpar->codec_id;

            const char* hw_decoder_name = getHwDecoderName(codec_id);
            const AVCodec* codec = nullptr;

            if (hw_decoder_name) {
                codec = avcodec_find_decoder_by_name(hw_decoder_name);
                if (codec &&
                    av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) ==
                        0) {
                    using_hw_decode_ = true;
                    LOG_INFO("VideoPlayer: NVDEC decoder: {}", hw_decoder_name);
                } else {
                    codec = nullptr;
                }
            }

            if (!codec) {
                codec = avcodec_find_decoder(codec_id);
                if (!codec) {
                    close();
                    return false;
                }
                using_hw_decode_ = false;
                LOG_INFO("VideoPlayer: CPU decoder");
            }

            codec_ctx_ = avcodec_alloc_context3(codec);
            avcodec_parameters_to_context(codec_ctx_, stream->codecpar);

            if (using_hw_decode_) {
                codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
                codec_ctx_->get_format = getHwFormat;
            } else {
                codec_ctx_->thread_count =
                    std::min(MAX_SW_DECODE_THREADS, static_cast<int>(std::thread::hardware_concurrency()));
                codec_ctx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
            }

            if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
                close();
                return false;
            }

            src_width_ = codec_ctx_->width;
            src_height_ = codec_ctx_->height;
            fps_ = av_q2d(stream->r_frame_rate);
            time_base_ = av_q2d(stream->time_base);

            if (src_height_ > MAX_PREVIEW_HEIGHT) {
                const float scale = static_cast<float>(MAX_PREVIEW_HEIGHT) / src_height_;
                width_ = static_cast<int>(src_width_ * scale) & ~1;
                height_ = MAX_PREVIEW_HEIGHT;
            } else {
                width_ = src_width_;
                height_ = src_height_;
            }
            frame_size_ = static_cast<size_t>(width_) * height_ * 3;

            total_frames_ = stream->nb_frames;
            if (total_frames_ == 0) {
                duration_ = static_cast<double>(fmt_ctx_->duration) / AV_TIME_BASE;
                total_frames_ = static_cast<int64_t>(duration_ * fps_);
            } else {
                duration_ = static_cast<double>(total_frames_) / fps_;
            }

            frame_ = av_frame_alloc();
            sw_frame_ = av_frame_alloc();
            packet_ = av_packet_alloc();
            display_buffer_.resize(frame_size_);

            if (!using_hw_decode_) {
                sws_ctx_ = sws_getContext(src_width_, src_height_, codec_ctx_->pix_fmt, width_, height_,
                                          AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            }

            if (decodeNextFrame()) {
                display_buffer_ = std::move(decoded_frame_.data);
                current_time_ = decoded_frame_.pts;
                current_frame_ = decoded_frame_.frame_number;
            }

            is_open_ = true;
            stop_decode_thread_ = false;
            decode_thread_ = std::thread(&Impl::decodeThreadFunc, this);

            return true;
        }

        void close() {
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                stop_decode_thread_ = true;
            }
            queue_cv_.notify_all();

            if (decode_thread_.joinable()) {
                decode_thread_.join();
            }

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                while (!frame_queue_.empty()) {
                    frame_queue_.pop();
                }
            }

            if (sws_ctx_) {
                sws_freeContext(sws_ctx_);
                sws_ctx_ = nullptr;
            }
            if (frame_) {
                av_frame_free(&frame_);
                frame_ = nullptr;
            }
            if (sw_frame_) {
                av_frame_free(&sw_frame_);
                sw_frame_ = nullptr;
            }
            if (packet_) {
                av_packet_free(&packet_);
                packet_ = nullptr;
            }
            if (codec_ctx_) {
                avcodec_free_context(&codec_ctx_);
                codec_ctx_ = nullptr;
            }
            if (hw_device_ctx_) {
                av_buffer_unref(&hw_device_ctx_);
                hw_device_ctx_ = nullptr;
            }
            if (fmt_ctx_) {
                avformat_close_input(&fmt_ctx_);
                fmt_ctx_ = nullptr;
            }

            video_stream_idx_ = -1;
            is_open_ = false;
            is_playing_ = false;
            using_hw_decode_ = false;
            current_time_ = 0;
            current_frame_ = 0;
        }

        [[nodiscard]] bool isOpen() const { return is_open_; }

        void play() {
            if (!is_open_) {
                return;
            }

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                    return frame_queue_.size() >= MIN_BUFFERED_FRAMES || eof_reached_ ||
                           stop_decode_thread_;
                });
            }

            is_playing_ = true;
            playback_start_time_ = -1;
        }

        void pause() { is_playing_ = false; }

        void togglePlayPause() {
            if (is_playing_) {
                pause();
            } else {
                play();
            }
        }

        [[nodiscard]] bool isPlaying() const { return is_playing_; }

        void seek(double seconds) {
            if (!is_open_) {
                return;
            }

            pause();
            seconds = std::clamp(seconds, 0.0, duration_);

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                while (!frame_queue_.empty()) {
                    frame_queue_.pop();
                }
                seek_target_ = seconds;
                seek_requested_ = true;
            }
            queue_cv_.notify_all();

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !seek_requested_ || stop_decode_thread_; });

                if (!frame_queue_.empty()) {
                    auto frame = std::move(frame_queue_.front());
                    frame_queue_.pop();
                    display_buffer_ = std::move(frame.data);
                    current_time_ = frame.pts;
                    current_frame_ = frame.frame_number;
                }
            }
        }

        void seekFrame(const int64_t frame_number) { seek(static_cast<double>(frame_number) / fps_); }

        void stepForward() {
            if (!is_open_) {
                return;
            }
            pause();

            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (!frame_queue_.empty()) {
                auto frame = std::move(frame_queue_.front());
                frame_queue_.pop();
                lock.unlock();
                queue_cv_.notify_all();

                display_buffer_ = std::move(frame.data);
                current_time_ = frame.pts;
                current_frame_ = frame.frame_number;
            }
        }

        void stepBackward() {
            if (!is_open_) {
                return;
            }
            pause();
            const int64_t target = std::max<int64_t>(0, current_frame_ - 1);
            seekFrame(target);
        }

        bool update(double /*current_wall_time*/) {
            if (!is_open_ || !is_playing_) {
                return false;
            }

            std::unique_lock<std::mutex> lock(queue_mutex_);

            if (frame_queue_.empty()) {
                lock.unlock();
                if (eof_reached_) {
                    is_playing_ = false;
                }
                return false;
            }

            auto& front = frame_queue_.front();
            display_buffer_ = std::move(front.data);
            current_time_ = front.pts;
            current_frame_ = front.frame_number;
            frame_queue_.pop();

            lock.unlock();
            queue_cv_.notify_all();

            return true;
        }

        [[nodiscard]] const uint8_t* currentFrameData() const {
            return display_buffer_.empty() ? nullptr : display_buffer_.data();
        }

        [[nodiscard]] int width() const { return width_; }
        [[nodiscard]] int height() const { return height_; }
        [[nodiscard]] double currentTime() const { return current_time_; }
        [[nodiscard]] double duration() const { return duration_; }
        [[nodiscard]] int64_t currentFrameNumber() const { return current_frame_; }
        [[nodiscard]] int64_t totalFrames() const { return total_frames_; }
        [[nodiscard]] double fps() const { return fps_; }

        std::vector<uint8_t> getThumbnail(const double time, const int max_width) {
            if (!is_open_) {
                return {};
            }

            seek(time);

            if (display_buffer_.empty()) {
                return {};
            }

            if (max_width > 0 && width_ > max_width) {
                const int thumb_height = height_ * max_width / width_;
                std::vector<uint8_t> thumb(static_cast<size_t>(max_width) * thumb_height * 3);

                SwsContext* const thumb_sws =
                    sws_getContext(width_, height_, AV_PIX_FMT_RGB24, max_width, thumb_height,
                                   AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);

                if (thumb_sws) {
                    uint8_t* src_data[1] = {display_buffer_.data()};
                    int src_linesize[1] = {width_ * 3};
                    uint8_t* dst_data[1] = {thumb.data()};
                    int dst_linesize[1] = {max_width * 3};

                    sws_scale(thumb_sws, src_data, src_linesize, 0, height_, dst_data, dst_linesize);
                    sws_freeContext(thumb_sws);
                }
                return thumb;
            }

            return display_buffer_;
        }

    private:
        void decodeThreadFunc() {
            while (true) {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                queue_cv_.wait(lock, [this] {
                    return stop_decode_thread_ || seek_requested_ ||
                           frame_queue_.size() < FRAME_QUEUE_SIZE;
                });

                if (stop_decode_thread_) {
                    break;
                }

                if (seek_requested_) {
                    lock.unlock();
                    performSeek(seek_target_);
                    lock.lock();
                    seek_requested_ = false;
                    lock.unlock();
                    queue_cv_.notify_all();
                    continue;
                }

                lock.unlock();

                if (decodeNextFrame()) {
                    std::lock_guard<std::mutex> qlock(queue_mutex_);
                    frame_queue_.push(std::move(decoded_frame_));
                } else {
                    eof_reached_ = true;
                }
            }
        }

        void performSeek(const double seconds) {
            const int64_t timestamp = static_cast<int64_t>(seconds / time_base_);
            avcodec_flush_buffers(codec_ctx_);
            av_seek_frame(fmt_ctx_, video_stream_idx_, timestamp, AVSEEK_FLAG_BACKWARD);
            eof_reached_ = false;

            while (decodeNextFrame()) {
                if (decoded_frame_.pts >= seconds - 0.5 / fps_) {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    frame_queue_.push(std::move(decoded_frame_));
                    break;
                }
            }
        }

        bool decodeNextFrame() {
            while (av_read_frame(fmt_ctx_, packet_) >= 0) {
                if (packet_->stream_index == video_stream_idx_) {
                    if (avcodec_send_packet(codec_ctx_, packet_) == 0) {
                        if (avcodec_receive_frame(codec_ctx_, frame_) == 0) {
                            convertFrameToBuffer();
                            av_packet_unref(packet_);
                            return true;
                        }
                    }
                }
                av_packet_unref(packet_);
            }

            avcodec_send_packet(codec_ctx_, nullptr);
            if (avcodec_receive_frame(codec_ctx_, frame_) == 0) {
                convertFrameToBuffer();
                return true;
            }

            return false;
        }

        void convertFrameToBuffer() {
            AVFrame* src_frame = frame_;

            if (using_hw_decode_ && frame_->format == AV_PIX_FMT_CUDA) {
                if (av_hwframe_transfer_data(sw_frame_, frame_, 0) < 0) {
                    return;
                }
                src_frame = sw_frame_;
            }

            if (!sws_ctx_) {
                sws_ctx_ = sws_getContext(src_width_, src_height_,
                                          static_cast<AVPixelFormat>(src_frame->format), width_, height_,
                                          AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            }

            decoded_frame_.data.resize(frame_size_);
            uint8_t* dst_data[1] = {decoded_frame_.data.data()};
            int dst_linesize[1] = {width_ * 3};
            sws_scale(sws_ctx_, src_frame->data, src_frame->linesize, 0, src_height_, dst_data,
                      dst_linesize);

            decoded_frame_.pts = static_cast<double>(frame_->pts) * time_base_;
            decoded_frame_.frame_number = static_cast<int64_t>(decoded_frame_.pts * fps_);
        }

        AVFormatContext* fmt_ctx_ = nullptr;
        AVCodecContext* codec_ctx_ = nullptr;
        AVBufferRef* hw_device_ctx_ = nullptr;
        SwsContext* sws_ctx_ = nullptr;
        AVFrame* frame_ = nullptr;
        AVFrame* sw_frame_ = nullptr;
        AVPacket* packet_ = nullptr;
        bool using_hw_decode_ = false;

        int video_stream_idx_ = -1;
        int src_width_ = 0;
        int src_height_ = 0;
        int width_ = 0;
        int height_ = 0;
        double fps_ = 0;
        double time_base_ = 0;
        double duration_ = 0;
        int64_t total_frames_ = 0;
        size_t frame_size_ = 0;

        std::vector<uint8_t> display_buffer_;
        double current_time_ = 0;
        int64_t current_frame_ = 0;

        double playback_start_time_ = -1;

        bool is_open_ = false;
        bool is_playing_ = false;

        std::thread decode_thread_;
        std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
        std::queue<DecodedFrame> frame_queue_;
        DecodedFrame decoded_frame_;
        std::atomic<bool> stop_decode_thread_{false};
        std::atomic<bool> eof_reached_{false};
        bool seek_requested_ = false;
        double seek_target_ = 0;
    };

    VideoPlayer::VideoPlayer() : impl_(std::make_unique<Impl>()) {}
    VideoPlayer::~VideoPlayer() = default;

    bool VideoPlayer::open(const std::filesystem::path& path) { return impl_->open(path); }
    void VideoPlayer::close() { impl_->close(); }
    bool VideoPlayer::isOpen() const { return impl_->isOpen(); }

    void VideoPlayer::play() { impl_->play(); }
    void VideoPlayer::pause() { impl_->pause(); }
    void VideoPlayer::togglePlayPause() { impl_->togglePlayPause(); }
    bool VideoPlayer::isPlaying() const { return impl_->isPlaying(); }

    void VideoPlayer::seek(const double seconds) { impl_->seek(seconds); }
    void VideoPlayer::seekFrame(const int64_t frame_number) { impl_->seekFrame(frame_number); }
    void VideoPlayer::stepForward() { impl_->stepForward(); }
    void VideoPlayer::stepBackward() { impl_->stepBackward(); }

    bool VideoPlayer::update(const double current_wall_time) { return impl_->update(current_wall_time); }

    const uint8_t* VideoPlayer::currentFrameData() const { return impl_->currentFrameData(); }
    int VideoPlayer::width() const { return impl_->width(); }
    int VideoPlayer::height() const { return impl_->height(); }

    double VideoPlayer::currentTime() const { return impl_->currentTime(); }
    double VideoPlayer::duration() const { return impl_->duration(); }
    int64_t VideoPlayer::currentFrameNumber() const { return impl_->currentFrameNumber(); }
    int64_t VideoPlayer::totalFrames() const { return impl_->totalFrames(); }
    double VideoPlayer::fps() const { return impl_->fps(); }

    std::vector<uint8_t> VideoPlayer::getThumbnail(const double time, const int max_width) {
        return impl_->getThumbnail(time, max_width);
    }

} // namespace lfs::io

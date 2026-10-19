/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/sfm_observation_chapter.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <span>

namespace lfs::io::project {
    namespace {
        static_assert(std::endian::native == std::endian::little,
                      "SFMO is stored little endian and copied without byte swapping");
        static_assert(sizeof(SfmObservation) == 5 * sizeof(float));

        constexpr std::uint32_t SFMO_MAGIC = 0x4F4D4653u; // "SFMO"
        constexpr std::uint32_t SFMO_VERSION = 1;

        struct Header {
            std::uint32_t magic = SFMO_MAGIC;
            std::uint32_t version = SFMO_VERSION;
            std::uint32_t camera_count = 0;
            std::uint32_t reserved = 0;
            std::uint64_t observation_count = 0;
            std::uint64_t content_hash = 0;
        };
        static_assert(sizeof(Header) == 32);

        struct IndexEntry {
            std::array<std::uint8_t, 16> uuid{};
            std::uint64_t count = 0;
        };
        static_assert(sizeof(IndexEntry) == 24);

        // FNV-1a over little-endian 64-bit words, zero-padding the final partial word.
        class ContentHasher {
        public:
            void write(const void* data, const std::size_t size) {
                const auto* bytes = static_cast<const std::uint8_t*>(data);
                for (std::size_t i = 0; i < size; ++i) {
                    pending_ |= static_cast<std::uint64_t>(bytes[i]) << (8 * pending_bytes_);
                    if (++pending_bytes_ == 8)
                        flush();
                }
            }
            [[nodiscard]] std::uint64_t finish() {
                if (pending_bytes_ > 0)
                    flush();
                return hash_;
            }

        private:
            void flush() {
                hash_ = (hash_ ^ pending_) * 1099511628211ull;
                pending_ = 0;
                pending_bytes_ = 0;
            }
            std::uint64_t hash_ = 14695981039346656037ull;
            std::uint64_t pending_ = 0;
            int pending_bytes_ = 0;
        };

        class ByteSink {
        public:
            explicit ByteSink(std::vector<std::byte>& bytes) : bytes_(bytes) {}
            void write(const void* data, const std::size_t size) {
                const auto* begin = static_cast<const std::byte*>(data);
                bytes_.insert(bytes_.end(), begin, begin + size);
            }

        private:
            std::vector<std::byte>& bytes_;
        };

        template <typename Sink>
        void emit_body(const SfmObservationTable& table, Sink& sink) {
            for (const auto& [uuid, observations] : table) {
                const IndexEntry entry{.uuid = uuid.bytes, .count = observations->size()};
                sink.write(&entry, sizeof(entry));
            }
            for (const auto& [uuid, observations] : table)
                sink.write(observations->data(), observations->size() * sizeof(SfmObservation));
        }

        std::uint64_t total_observations(const SfmObservationTable& table) {
            std::uint64_t total = 0;
            for (const auto& [uuid, observations] : table)
                total += observations->size();
            return total;
        }

        lfs::Error sfmo_error(std::string detail) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::DataLoss,
                .domain = lfs::ErrorDomain::IO,
                .user_message = "The SfM observation chapter is invalid.",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        template <typename T>
        lfs::Result<void> read_object(const LazyChunkValue& chapter, const std::uint64_t offset, T& value) {
            return chapter.read_at(offset, std::as_writable_bytes(std::span(&value, 1)));
        }
    } // namespace

    std::uint64_t sfm_observation_content_hash(const SfmObservationTable& table) {
        ContentHasher hasher;
        emit_body(table, hasher);
        return hasher.finish();
    }

    std::vector<std::byte> encode_sfm_observations(const SfmObservationTable& table) {
        assert(table.size() <= std::numeric_limits<std::uint32_t>::max());
        const Header header{
            .camera_count = static_cast<std::uint32_t>(table.size()),
            .observation_count = total_observations(table),
            .content_hash = sfm_observation_content_hash(table),
        };
        std::vector<std::byte> bytes;
        bytes.reserve(sizeof(Header) + table.size() * sizeof(IndexEntry) +
                      header.observation_count * sizeof(SfmObservation));
        ByteSink sink(bytes);
        sink.write(&header, sizeof(header));
        emit_body(table, sink);
        return bytes;
    }

    lfs::Result<SfmObservationIndex> read_sfm_observation_index(const LazyChunkValue& chapter) {
        Header header;
        if (chapter.size() < sizeof(Header))
            return sfmo_error(std::format("SFMO has {} bytes, the header needs {}", chapter.size(), sizeof(Header)));
        if (auto read = read_object(chapter, 0, header); !read)
            return std::move(read).error();
        if (header.magic != SFMO_MAGIC || header.version != SFMO_VERSION || header.reserved != 0)
            return sfmo_error(std::format("SFMO header magic={:#x} version={} reserved={}",
                                          header.magic, header.version, header.reserved));
        const std::uint64_t index_bytes = std::uint64_t{header.camera_count} * sizeof(IndexEntry);
        const std::uint64_t max_observations = (std::numeric_limits<std::uint64_t>::max() - sizeof(Header) - index_bytes) /
                                               sizeof(SfmObservation);
        if (header.observation_count > max_observations ||
            chapter.size() != sizeof(Header) + index_bytes + header.observation_count * sizeof(SfmObservation))
            return sfmo_error(std::format("SFMO size {} does not match {} cameras and {} observations",
                                          chapter.size(), header.camera_count, header.observation_count));

        std::vector<IndexEntry> entries(header.camera_count);
        if (auto read = chapter.read_at(sizeof(Header), std::as_writable_bytes(std::span(entries))); !read)
            return std::move(read).error();
        SfmObservationIndex index{.observation_count = header.observation_count, .content_hash = header.content_hash};
        std::uint64_t first = 0;
        for (const auto& entry : entries) {
            lfs::core::Uuid uuid;
            uuid.bytes = entry.uuid;
            if (uuid.is_nil() || entry.count > header.observation_count - first ||
                (!index.cameras.empty() && !UuidLess{}(std::prev(index.cameras.end())->first, uuid)))
                return sfmo_error("SFMO index entries must be non-nil, ascending and within the observation count");
            index.cameras.emplace(uuid, SfmObservationIndex::Entry{.first = first, .count = entry.count});
            first += entry.count;
        }
        if (first != header.observation_count)
            return sfmo_error(std::format("SFMO index covers {} of {} observations", first, header.observation_count));
        return index;
    }

    lfs::Result<std::shared_ptr<const SfmObservationChapterSource>>
    SfmObservationChapterSource::open(const LazyChunkValue& chapter) {
        auto index = read_sfm_observation_index(chapter);
        if (!index)
            return std::move(index).error();
        auto shared = chapter.share();
        if (!shared)
            return std::move(shared).error();
        return std::shared_ptr<const SfmObservationChapterSource>(
            new SfmObservationChapterSource(std::move(*shared), std::move(*index)));
    }

    SfmObservationChapterSource::SfmObservationChapterSource(LazyChunkValue chapter, SfmObservationIndex index)
        : chapter_(std::move(chapter)),
          chapter_uuid_(chapter_.snapshot_uuid()),
          index_(std::move(index)) {}

    const std::vector<SfmObservation>& SfmObservationChapterSource::observations(
        const lfs::core::Uuid& camera_node) const {
        std::call_once(decoded_, [this] {
            std::vector<SfmObservation> all(index_.observation_count);
            const std::uint64_t data_offset = sizeof(Header) + index_.cameras.size() * sizeof(IndexEntry);
            if (auto read = chapter_.read_at(data_offset, std::as_writable_bytes(std::span(all))); !read) {
                LOG_ERROR("SfM observations unavailable: {}", lfs::format_for_developer(read.error()));
                return;
            }
            for (const auto& [uuid, entry] : index_.cameras) {
                const auto begin = all.begin() + static_cast<std::ptrdiff_t>(entry.first);
                observations_.emplace(uuid, std::vector<SfmObservation>(
                                                begin, begin + static_cast<std::ptrdiff_t>(entry.count)));
            }
        });
        static const std::vector<SfmObservation> none;
        const auto found = observations_.find(camera_node);
        return found == observations_.end() ? none : found->second;
    }

    SfmObservationCameras capture_sfm_observation_cameras(const lfs::core::Scene& scene) {
        SfmObservationCameras cameras;
        for (const lfs::core::SceneNode* node : scene.getNodes()) {
            if (node->camera && node->camera->sfm_observation_count() > 0)
                cameras.emplace_back(node->uuid, node->camera);
        }
        return cameras;
    }

    lfs::Result<void> sync_sfm_observations(ProjectDocument& document, const SfmObservationCameras& cameras) {
        const LazyChunkValue* const current = document.find_sfm_observations();
        if (cameras.empty())
            return current ? document.set_sfm_observations(std::nullopt) : lfs::Result<void>{};
        const bool backed_by_current =
            current && std::ranges::all_of(cameras, [current](const auto& entry) {
                const auto* source =
                    dynamic_cast<const SfmObservationChapterSource*>(entry.second->sfm_observation_source());
                return source && source->chapter_uuid() == current->snapshot_uuid();
            });
        if (backed_by_current)
            return {};

        SfmObservationTable table;
        for (const auto& [uuid, camera] : cameras)
            table.emplace(uuid, &camera->sfm_observations());
        if (current) {
            const auto index = read_sfm_observation_index(*current);
            if (index && index->content_hash == sfm_observation_content_hash(table))
                return {};
        }
        auto payload = LazyChunkValue::from_owned(encode_sfm_observations(table), lfs::core::generate_uuid_v4());
        if (!payload)
            return lfs::Result<void>::failure(std::move(payload).error());
        return document.set_sfm_observations(std::move(*payload));
    }

} // namespace lfs::io::project

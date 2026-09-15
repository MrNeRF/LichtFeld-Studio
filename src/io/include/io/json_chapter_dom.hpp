/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

#include <nlohmann/json.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace lfs::io {

    namespace json_chapter_detail {

        template <typename T>
        concept ReadableScalar =
            std::same_as<T, std::remove_cvref_t<T>> &&
            (std::same_as<T, bool> || std::same_as<T, std::string> ||
             (std::integral<T> && !std::same_as<T, bool>) || std::floating_point<T>);

        template <typename T>
        concept WritableScalar =
            ReadableScalar<std::remove_cvref_t<T>> || std::convertible_to<T, std::string_view>;

    } // namespace json_chapter_detail

    // Retained, insertion-ordered JSON for one .licht chapter. There is no
    // parallel typed representation: every accessor resolves and edits this
    // DOM, and dump() serializes this DOM directly.
    //
    // Paths use dot-separated object keys. Escaping is deliberately not
    // supported: empty segments and keys containing '.' cannot be addressed
    // through this API. Such keys are still retained and serialized unchanged.
    class LFS_IO_API JsonChapterDom {
    public:
        using Json = nlohmann::ordered_json;

        class ConstElement;

        // Locator-backed mutable view of one UUID-addressed array element.
        // The element is resolved again for each operation, so other array
        // edits do not invalidate the view. Moving or destroying the owning
        // JsonChapterDom invalidates it; changing its "uuid" field makes this
        // locator stop resolving it.
        class LFS_IO_API Element {
        public:
            template <json_chapter_detail::ReadableScalar T>
            [[nodiscard]] std::optional<T> get(const std::string_view path) const {
                return owner_->get_from_element<T>(array_path_, uuid_, path);
            }

            template <typename T>
                requires json_chapter_detail::WritableScalar<T>
            [[nodiscard]] lfs::Result<void> set(const std::string_view path, T&& value) {
                return owner_->set_element_value(
                    array_path_, uuid_, path, JsonChapterDom::scalar_to_json(std::forward<T>(value)));
            }

            [[nodiscard]] lfs::Result<bool> remove(std::string_view path);
            [[nodiscard]] bool exists() const;
            [[nodiscard]] std::optional<Json> get_json(std::string_view path) const;
            [[nodiscard]] lfs::Result<void> set_json(std::string_view path, Json value);

        private:
            friend class JsonChapterDom;

            Element(JsonChapterDom& owner, std::string array_path, std::string uuid)
                : owner_(&owner),
                  array_path_(std::move(array_path)),
                  uuid_(std::move(uuid)) {}

            JsonChapterDom* owner_;
            std::string array_path_;
            std::string uuid_;
        };

        // Read-only counterpart returned when finding an element through a
        // const JsonChapterDom.
        class LFS_IO_API ConstElement {
        public:
            template <json_chapter_detail::ReadableScalar T>
            [[nodiscard]] std::optional<T> get(const std::string_view path) const {
                return owner_->get_from_element<T>(array_path_, uuid_, path);
            }

            [[nodiscard]] bool exists() const;
            [[nodiscard]] std::optional<Json> get_json(std::string_view path) const;

        private:
            friend class JsonChapterDom;

            ConstElement(const JsonChapterDom& owner, std::string array_path, std::string uuid)
                : owner_(&owner),
                  array_path_(std::move(array_path)),
                  uuid_(std::move(uuid)) {}

            const JsonChapterDom* owner_;
            std::string array_path_;
            std::string uuid_;
        };

        // New chapters start as an empty object.
        JsonChapterDom();
        JsonChapterDom(const JsonChapterDom&) = default;
        JsonChapterDom(JsonChapterDom&&) noexcept = default;
        JsonChapterDom& operator=(const JsonChapterDom&) = default;
        JsonChapterDom& operator=(JsonChapterDom&&) noexcept = default;
        ~JsonChapterDom() = default;

        [[nodiscard]] static lfs::Result<JsonChapterDom> parse(std::string_view bytes);
        [[nodiscard]] static lfs::Result<JsonChapterDom> from_bytes(std::span<const std::byte> bytes);

        // Two-space indentation, insertion-ordered object keys, no trailing
        // newline. parse(dump(x)).dump() is stable.
        [[nodiscard]] std::string dump() const;
        [[nodiscard]] std::vector<std::byte> to_bytes() const;

        // Missing paths and incompatible JSON scalar types both return
        // std::nullopt. Mutators report invalid paths and structural conflicts
        // as typed lfs::Error values.
        template <json_chapter_detail::ReadableScalar T>
        [[nodiscard]] std::optional<T> get(const std::string_view path) const {
            return read_scalar<T>(find_node_from(root_, path));
        }

        template <json_chapter_detail::ReadableScalar T>
        [[nodiscard]] static std::optional<T> read(const Json& node,
                                                   std::string_view path) {
            return read_scalar<T>(find_node_from(node, path));
        }
        [[nodiscard]] static std::optional<Json> read_json(const Json& node,
                                                           std::string_view path);
        [[nodiscard]] static const Json* read_json_ref(const Json& node,
                                                       std::string_view path);

        template <typename T>
            requires json_chapter_detail::WritableScalar<T>
        [[nodiscard]] lfs::Result<void> set(const std::string_view path, T&& value) {
            return set_value(path, scalar_to_json(std::forward<T>(value)));
        }

        [[nodiscard]] lfs::Result<bool> remove(std::string_view path);
        [[nodiscard]] std::optional<Json> get_json(std::string_view path) const;
        [[nodiscard]] const Json* get_json_ref(std::string_view path) const;
        [[nodiscard]] lfs::Result<void> set_json(std::string_view path, Json value);

        // UUIDs use the exact canonical textual form
        // xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx with lowercase hex digits.
        [[nodiscard]] std::optional<Element> array_find(std::string_view path, std::string_view uuid);
        [[nodiscard]] std::optional<ConstElement> array_find(std::string_view path,
                                                             std::string_view uuid) const;
        [[nodiscard]] lfs::Result<Element> array_upsert(std::string_view path, std::string_view uuid);
        [[nodiscard]] lfs::Result<bool> array_remove(std::string_view path, std::string_view uuid);
        [[nodiscard]] lfs::Result<std::vector<std::string>>
        array_uuids(std::string_view path) const;

        // One-pass enumeration of a UUID-addressed object array: each element's
        // canonical uuid together with a copy of its JSON, in array order. Same
        // validation and error taxonomy as array_uuids; a missing path yields an
        // empty vector.
        [[nodiscard]] lfs::Result<std::vector<std::pair<std::string, Json>>>
        array_items(std::string_view path) const;
        [[nodiscard]] lfs::Result<std::vector<std::pair<std::string, const Json*>>>
        array_item_refs(std::string_view path) const;

        // Copy of one element's JSON, or nullopt when the array or element is absent
        // (same resolution semantics as const array_find).
        [[nodiscard]] std::optional<Json> array_get(std::string_view path,
                                                    std::string_view uuid) const;

        // Replace-or-insert one element. value must be an object whose "uuid" member
        // equals uuid (canonical form); a replaced element keeps its array position.
        [[nodiscard]] lfs::Result<void> array_put(std::string_view path,
                                                  std::string_view uuid, Json value);

    private:
        explicit JsonChapterDom(Json root)
            : root_(std::move(root)) {}

        template <typename T>
            requires json_chapter_detail::WritableScalar<T>
        [[nodiscard]] static Json scalar_to_json(T&& value) {
            if constexpr (std::convertible_to<T, std::string_view>) {
                return Json(std::string(std::string_view(std::forward<T>(value))));
            } else {
                return Json(std::forward<T>(value));
            }
        }

        template <json_chapter_detail::ReadableScalar T>
        [[nodiscard]] static std::optional<T> read_scalar(const Json* value) {
            if (value == nullptr || value->is_null() || value->is_structured()) {
                return std::nullopt;
            }

            try {
                if constexpr (std::same_as<T, bool>) {
                    if (!value->is_boolean()) {
                        return std::nullopt;
                    }
                } else if constexpr (std::same_as<T, std::string>) {
                    if (!value->is_string()) {
                        return std::nullopt;
                    }
                } else if constexpr (std::integral<T>) {
                    if (value->is_number_unsigned()) {
                        const auto raw = value->template get<std::uint64_t>();
                        if (raw > static_cast<std::uint64_t>(std::numeric_limits<T>::max())) {
                            return std::nullopt;
                        }
                        return static_cast<T>(raw);
                    }
                    if (!value->is_number_integer()) {
                        return std::nullopt;
                    }
                    const auto raw = value->template get<std::int64_t>();
                    if constexpr (std::signed_integral<T>) {
                        if (raw < static_cast<std::int64_t>(std::numeric_limits<T>::min()) ||
                            raw > static_cast<std::int64_t>(std::numeric_limits<T>::max())) {
                            return std::nullopt;
                        }
                    } else if (raw < 0 ||
                               static_cast<std::uint64_t>(raw) >
                                   static_cast<std::uint64_t>(std::numeric_limits<T>::max())) {
                        return std::nullopt;
                    }
                    return static_cast<T>(raw);
                } else if constexpr (std::floating_point<T>) {
                    if (!value->is_number()) {
                        return std::nullopt;
                    }
                }
                return value->template get<T>();
            } catch (const nlohmann::json::exception&) {
                return std::nullopt;
            }
        }

        template <json_chapter_detail::ReadableScalar T>
        [[nodiscard]] std::optional<T> get_from_element(const std::string_view array_path,
                                                        const std::string_view uuid,
                                                        const std::string_view path) const {
            const Json* element = find_array_element_node(root_, array_path, uuid);
            return element == nullptr ? std::nullopt : read_scalar<T>(find_node_from(*element, path));
        }

        [[nodiscard]] static const Json* find_node_from(const Json& root, std::string_view path);
        [[nodiscard]] static Json* find_array_element_node(Json& root, std::string_view path,
                                                           std::string_view uuid);
        [[nodiscard]] static const Json* find_array_element_node(const Json& root,
                                                                 std::string_view path,
                                                                 std::string_view uuid);
        [[nodiscard]] static bool is_canonical_uuid(std::string_view uuid);

        [[nodiscard]] lfs::Result<void> set_value(std::string_view path, Json value);
        [[nodiscard]] lfs::Result<void> set_element_value(std::string_view array_path,
                                                          std::string_view uuid,
                                                          std::string_view path, Json value);
        [[nodiscard]] lfs::Result<bool> remove_element_value(std::string_view array_path,
                                                             std::string_view uuid,
                                                             std::string_view path);

        Json root_;
    };

} // namespace lfs::io

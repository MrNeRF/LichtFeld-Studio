/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/json_chapter_dom.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace lfs::io {

    namespace {

        using Json = nlohmann::ordered_json;
        constexpr std::size_t MAX_JSON_CHAPTER_BYTES =
            64ull * 1024 * 1024;
        constexpr std::size_t MAX_JSON_NESTING_DEPTH = 512;

        bool exceeds_json_nesting_limit(
            const std::string_view bytes) noexcept {
            std::size_t depth = 0;
            bool in_string = false;
            bool escaped = false;
            for (const char character : bytes) {
                if (in_string) {
                    if (escaped) {
                        escaped = false;
                    } else if (character == '\\') {
                        escaped = true;
                    } else if (character == '"') {
                        in_string = false;
                    }
                    continue;
                }
                if (character == '"') {
                    in_string = true;
                } else if (character == '{' || character == '[') {
                    ++depth;
                    if (depth > MAX_JSON_NESTING_DEPTH) {
                        return true;
                    }
                } else if ((character == '}' || character == ']') &&
                           depth != 0) {
                    --depth;
                }
            }
            return false;
        }

        struct PathSegments {
            std::vector<std::string_view> values;
        };

        lfs::Error make_dom_error(const lfs::ErrorCode code, std::string user_message,
                                  std::string detail, const std::string_view path = {},
                                  const std::string_view uuid = {}) {
            lfs::SmallFields fields;
            if (!path.empty()) {
                fields.add("path", path);
            }
            if (!uuid.empty()) {
                fields.add("uuid", uuid);
            }
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .user_message = std::move(user_message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
            });
        }

        lfs::Result<PathSegments> split_path(const std::string_view path) {
            if (path.empty()) {
                return make_dom_error(lfs::ErrorCode::InvalidArgument,
                                      "The JSON chapter path is invalid.",
                                      "JSON chapter paths must not be empty");
            }

            PathSegments result;
            std::size_t begin = 0;
            while (begin < path.size()) {
                const std::size_t end = path.find('.', begin);
                const std::size_t length =
                    (end == std::string_view::npos ? path.size() : end) - begin;
                if (length == 0) {
                    return make_dom_error(
                        lfs::ErrorCode::InvalidArgument, "The JSON chapter path is invalid.",
                        std::format("JSON chapter path '{}' contains an empty key segment", path), path);
                }
                result.values.emplace_back(path.substr(begin, length));
                if (end == std::string_view::npos) {
                    break;
                }
                begin = end + 1;
                if (begin == path.size()) {
                    return make_dom_error(
                        lfs::ErrorCode::InvalidArgument, "The JSON chapter path is invalid.",
                        std::format("JSON chapter path '{}' ends with an empty key segment", path), path);
                }
            }
            return result;
        }

        std::string path_prefix(const PathSegments& path, const std::size_t inclusive_index) {
            std::string result;
            for (std::size_t i = 0; i <= inclusive_index; ++i) {
                if (!result.empty()) {
                    result.push_back('.');
                }
                result.append(path.values[i]);
            }
            return result;
        }

        lfs::Error non_object_path_error(const std::string_view action,
                                         const std::string_view whole_path,
                                         const std::string_view prefix, const Json& value) {
            return make_dom_error(
                lfs::ErrorCode::FailedPrecondition, "The JSON chapter has an incompatible structure.",
                std::format("Cannot {} JSON path '{}': '{}' is {}, expected an object", action,
                            whole_path, prefix, value.type_name()),
                whole_path);
        }

        template <typename JsonType>
        lfs::Result<JsonType*> resolve_node(JsonType& root, const std::string_view path,
                                            const std::string_view action) {
            auto split = split_path(path);
            if (!split) {
                return std::move(split).error();
            }

            JsonType* current = &root;
            for (std::size_t i = 0; i < split->values.size(); ++i) {
                if (!current->is_object()) {
                    const std::string prefix = i == 0 ? "<root>" : path_prefix(*split, i - 1);
                    return non_object_path_error(action, path, prefix, *current);
                }
                const auto found = current->find(std::string(split->values[i]));
                if (found == current->end()) {
                    return static_cast<JsonType*>(nullptr);
                }
                current = &*found;
            }
            return current;
        }

        lfs::Result<void> set_value_at(Json& root, const std::string_view path, Json value) {
            auto split = split_path(path);
            if (!split) {
                return lfs::Result<void>::failure(std::move(split).error());
            }

            const Json* current = &root;
            for (std::size_t i = 0; i + 1 < split->values.size(); ++i) {
                if (!current->is_object()) {
                    const std::string prefix = i == 0 ? "<root>" : path_prefix(*split, i - 1);
                    return lfs::Result<void>::failure(non_object_path_error(
                        "set", path, prefix, *current));
                }
                const auto found = current->find(std::string(split->values[i]));
                if (found == current->end()) {
                    current = nullptr;
                    break;
                }
                current = &*found;
            }
            if (current != nullptr && !current->is_object()) {
                const std::string prefix = split->values.size() == 1
                                               ? "<root>"
                                               : path_prefix(*split, split->values.size() - 2);
                return lfs::Result<void>::failure(non_object_path_error(
                    "set", path, prefix, *current));
            }

            Json* destination = &root;
            for (std::size_t i = 0; i + 1 < split->values.size(); ++i) {
                auto found = destination->find(std::string(split->values[i]));
                if (found == destination->end()) {
                    (*destination)[std::string(split->values[i])] = Json::object();
                    found = destination->find(std::string(split->values[i]));
                }
                destination = &*found;
            }
            (*destination)[std::string(split->values.back())] = std::move(value);
            return {};
        }

        lfs::Result<bool> remove_value_at(Json& root, const std::string_view path) {
            auto split = split_path(path);
            if (!split) {
                return std::move(split).error();
            }

            Json* parent = &root;
            for (std::size_t i = 0; i + 1 < split->values.size(); ++i) {
                if (!parent->is_object()) {
                    const std::string prefix = i == 0 ? "<root>" : path_prefix(*split, i - 1);
                    return non_object_path_error("remove", path, prefix, *parent);
                }
                const auto found = parent->find(std::string(split->values[i]));
                if (found == parent->end()) {
                    return false;
                }
                parent = &*found;
            }
            if (!parent->is_object()) {
                const std::string prefix = split->values.size() == 1
                                               ? "<root>"
                                               : path_prefix(*split, split->values.size() - 2);
                return non_object_path_error("remove", path, prefix, *parent);
            }
            return parent->erase(std::string(split->values.back())) != 0;
        }

        template <typename T>
        lfs::Result<T> operation_failure(lfs::Error error) {
            if constexpr (std::same_as<T, void>) {
                return lfs::Result<void>::failure(std::move(error));
            } else {
                return error;
            }
        }

        template <typename T, typename Function>
        lfs::Result<T> guarded_dom_operation(const std::string_view path, Function&& function) {
            try {
                return std::forward<Function>(function)();
            } catch (const std::bad_alloc&) {
                return operation_failure<T>(make_dom_error(
                    lfs::ErrorCode::ResourceExhausted, "Not enough memory to edit the JSON chapter.",
                    std::format("Memory allocation failed while editing JSON path '{}'", path), path));
            } catch (const nlohmann::json::exception& error) {
                return operation_failure<T>(make_dom_error(
                    lfs::ErrorCode::Internal, "The JSON chapter could not be edited.",
                    std::format("JSON library failure while editing path '{}': {}", path, error.what()),
                    path));
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): foreign JSON/standard-library failures are
                // normalized into a typed lfs::Error and returned to the caller below.
                return operation_failure<T>(make_dom_error(
                    lfs::ErrorCode::Internal, "The JSON chapter could not be edited.",
                    std::format("Unexpected failure while editing JSON path '{}': {}", path, error.what()),
                    path));
            }
        }

    } // namespace

    JsonChapterDom::JsonChapterDom()
        : root_(Json::object()) {}

    lfs::Result<JsonChapterDom> JsonChapterDom::parse(const std::string_view bytes) {
        if (bytes.size() > MAX_JSON_CHAPTER_BYTES) {
            return make_dom_error(
                lfs::ErrorCode::ResourceExhausted,
                "The project JSON chapter is too large to load.",
                std::format(
                    "JSON chapter has {} bytes; maximum is {} bytes",
                    bytes.size(), MAX_JSON_CHAPTER_BYTES));
        }
        if (exceeds_json_nesting_limit(bytes)) {
            return make_dom_error(
                lfs::ErrorCode::ResourceExhausted,
                "The project JSON chapter is nested too deeply to load.",
                std::format(
                    "JSON nesting exceeds the maximum depth of {}",
                    MAX_JSON_NESTING_DEPTH));
        }
        try {
            bool duplicate_key = false;
            std::string duplicate_name;
            std::unordered_map<int, std::unordered_set<std::string>>
                object_keys;
            const auto reject_duplicate =
                [&](const int depth, const Json::parse_event_t event,
                    Json& parsed) {
                    if (event == Json::parse_event_t::object_start) {
                        object_keys[depth + 1].clear();
                    } else if (event == Json::parse_event_t::key) {
                        auto& keys = object_keys[depth];
                        const auto key = parsed.get<std::string>();
                        if (!keys.insert(key).second) {
                            duplicate_key = true;
                            duplicate_name = key;
                        }
                    } else if (event == Json::parse_event_t::object_end) {
                        object_keys.erase(depth + 1);
                    }
                    return true;
                };
            auto parsed = Json::parse(
                bytes.begin(), bytes.end(), reject_duplicate);
            if (duplicate_key) {
                return make_dom_error(
                    lfs::ErrorCode::DataLoss,
                    "The project JSON chapter contains a duplicate key.",
                    std::format(
                        "Duplicate JSON object key '{}' is forbidden; both parsers use reject semantics",
                        duplicate_name),
                    duplicate_name);
            }
            return JsonChapterDom(std::move(parsed));
        } catch (const nlohmann::json::parse_error& error) {
            return make_dom_error(
                lfs::ErrorCode::DataLoss, "The project contains malformed JSON.",
                std::format("Malformed JSON chapter at byte {}: {}", error.byte, error.what()));
        } catch (const std::bad_alloc&) {
            return make_dom_error(lfs::ErrorCode::ResourceExhausted,
                                  "Not enough memory to read the JSON chapter.",
                                  std::format("Memory allocation failed while parsing {} JSON bytes",
                                              bytes.size()));
        } catch (const nlohmann::json::exception& error) {
            return make_dom_error(lfs::ErrorCode::DataLoss,
                                  "The project contains invalid JSON.",
                                  std::format("Invalid JSON chapter: {}", error.what()));
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): this parser boundary converts the foreign
            // exception into a typed IO/DataLoss error instead of swallowing it.
            return make_dom_error(lfs::ErrorCode::DataLoss,
                                  "The project JSON could not be read.",
                                  std::format("JSON chapter parse failed: {}", error.what()));
        }
    }

    lfs::Result<JsonChapterDom> JsonChapterDom::from_bytes(const std::span<const std::byte> bytes) {
        if (bytes.empty()) {
            return parse({});
        }
        return parse(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    }

    std::string JsonChapterDom::dump() const {
        return root_.dump(2);
    }

    std::vector<std::byte> JsonChapterDom::to_bytes() const {
        const std::string serialized = dump();
        std::vector<std::byte> bytes(serialized.size());
        std::memcpy(bytes.data(), serialized.data(), serialized.size());
        return bytes;
    }

    const JsonChapterDom::Json* JsonChapterDom::find_node_from(const Json& root,
                                                               const std::string_view path) {
        if (path.empty()) {
            return nullptr;
        }

        const Json* current = &root;
        std::size_t begin = 0;
        while (begin < path.size()) {
            const std::size_t end = path.find('.', begin);
            const std::size_t length =
                (end == std::string_view::npos ? path.size() : end) - begin;
            if (length == 0 || !current->is_object()) {
                return nullptr;
            }
            const auto found = current->find(std::string(path.substr(begin, length)));
            if (found == current->end()) {
                return nullptr;
            }
            current = &*found;
            if (end == std::string_view::npos) {
                return current;
            }
            begin = end + 1;
            if (begin == path.size()) {
                return nullptr;
            }
        }
        return nullptr;
    }

    bool JsonChapterDom::is_canonical_uuid(const std::string_view uuid) {
        if (uuid.size() != 36) {
            return false;
        }
        for (std::size_t i = 0; i < uuid.size(); ++i) {
            if (i == 8 || i == 13 || i == 18 || i == 23) {
                if (uuid[i] != '-') {
                    return false;
                }
                continue;
            }
            if (!((uuid[i] >= '0' && uuid[i] <= '9') || (uuid[i] >= 'a' && uuid[i] <= 'f'))) {
                return false;
            }
        }
        return true;
    }

    JsonChapterDom::Json* JsonChapterDom::find_array_element_node(Json& root,
                                                                  const std::string_view path,
                                                                  const std::string_view uuid) {
        return const_cast<Json*>(find_array_element_node(std::as_const(root), path, uuid));
    }

    const JsonChapterDom::Json* JsonChapterDom::find_array_element_node(
        const Json& root, const std::string_view path, const std::string_view uuid) {
        const Json* array = find_node_from(root, path);
        if (array == nullptr || !array->is_array()) {
            return nullptr;
        }
        const auto found = std::find_if(array->begin(), array->end(), [uuid](const Json& element) {
            if (!element.is_object()) {
                return false;
            }
            const auto member = element.find("uuid");
            return member != element.end() && member->is_string() &&
                   member->get_ref<const std::string&>() == uuid;
        });
        return found == array->end() ? nullptr : &*found;
    }

    lfs::Result<void> JsonChapterDom::set_value(const std::string_view path, Json value) {
        return guarded_dom_operation<void>(path, [&]() -> lfs::Result<void> {
            Json candidate = root_;
            auto result = set_value_at(candidate, path, std::move(value));
            if (!result) {
                return result;
            }
            root_ = std::move(candidate);
            return {};
        });
    }

    lfs::Result<bool> JsonChapterDom::remove(const std::string_view path) {
        return guarded_dom_operation<bool>(path, [&]() -> lfs::Result<bool> {
            Json candidate = root_;
            auto removed = remove_value_at(candidate, path);
            if (!removed || !*removed) {
                return removed;
            }
            root_ = std::move(candidate);
            return true;
        });
    }

    std::optional<JsonChapterDom::Json> JsonChapterDom::get_json(
        const std::string_view path) const {
        const Json* value = find_node_from(root_, path);
        return value == nullptr ? std::nullopt : std::optional<Json>(*value);
    }

    lfs::Result<void> JsonChapterDom::set_json(const std::string_view path, Json value) {
        return set_value(path, std::move(value));
    }

    std::optional<JsonChapterDom::Element> JsonChapterDom::array_find(
        const std::string_view path, const std::string_view uuid) {
        if (!is_canonical_uuid(uuid) || find_array_element_node(root_, path, uuid) == nullptr) {
            return std::nullopt;
        }
        return Element(*this, std::string(path), std::string(uuid));
    }

    std::optional<JsonChapterDom::ConstElement> JsonChapterDom::array_find(
        const std::string_view path, const std::string_view uuid) const {
        if (!is_canonical_uuid(uuid) || find_array_element_node(root_, path, uuid) == nullptr) {
            return std::nullopt;
        }
        return ConstElement(*this, std::string(path), std::string(uuid));
    }

    lfs::Result<JsonChapterDom::Element> JsonChapterDom::array_upsert(
        const std::string_view path, const std::string_view uuid) {
        if (!is_canonical_uuid(uuid)) {
            return make_dom_error(
                lfs::ErrorCode::InvalidArgument, "The JSON chapter UUID is invalid.",
                std::format("UUID '{}' is not canonical lowercase xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                            uuid),
                path, uuid);
        }

        return guarded_dom_operation<Element>(path, [&]() -> lfs::Result<Element> {
            auto existing_array = resolve_node(root_, path, "upsert into");
            if (!existing_array) {
                return std::move(existing_array).error();
            }
            if (*existing_array != nullptr) {
                if (!(*existing_array)->is_array()) {
                    return make_dom_error(
                        lfs::ErrorCode::FailedPrecondition,
                        "The JSON chapter has an incompatible structure.",
                        std::format("Cannot upsert UUID '{}' at '{}': value is {}, expected an array",
                                    uuid, path, (*existing_array)->type_name()),
                        path, uuid);
                }
                if (find_array_element_node(root_, path, uuid) != nullptr) {
                    return Element(*this, std::string(path), std::string(uuid));
                }
            }

            Json candidate = root_;
            auto candidate_array = resolve_node(candidate, path, "upsert into");
            if (!candidate_array) {
                return std::move(candidate_array).error();
            }
            if (*candidate_array == nullptr) {
                auto created = set_value_at(candidate, path, Json::array());
                if (!created) {
                    return std::move(created).error();
                }
                candidate_array = resolve_node(candidate, path, "upsert into");
                if (!candidate_array) {
                    return std::move(candidate_array).error();
                }
            }
            (*candidate_array)->push_back(Json{{"uuid", std::string(uuid)}});
            root_ = std::move(candidate);
            return Element(*this, std::string(path), std::string(uuid));
        });
    }

    lfs::Result<bool> JsonChapterDom::array_remove(const std::string_view path,
                                                   const std::string_view uuid) {
        if (!is_canonical_uuid(uuid)) {
            return make_dom_error(
                lfs::ErrorCode::InvalidArgument, "The JSON chapter UUID is invalid.",
                std::format("UUID '{}' is not canonical lowercase xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                            uuid),
                path, uuid);
        }

        return guarded_dom_operation<bool>(path, [&]() -> lfs::Result<bool> {
            auto current_array = resolve_node(root_, path, "remove from");
            if (!current_array) {
                return std::move(current_array).error();
            }
            if (*current_array == nullptr) {
                return false;
            }
            if (!(*current_array)->is_array()) {
                return make_dom_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "The JSON chapter has an incompatible structure.",
                    std::format("Cannot remove UUID '{}' at '{}': value is {}, expected an array", uuid,
                                path, (*current_array)->type_name()),
                    path, uuid);
            }
            if (find_array_element_node(root_, path, uuid) == nullptr) {
                return false;
            }

            Json candidate = root_;
            auto candidate_array = resolve_node(candidate, path, "remove from");
            if (!candidate_array) {
                return std::move(candidate_array).error();
            }
            const auto found = std::find_if((*candidate_array)->begin(), (*candidate_array)->end(),
                                            [uuid](const Json& element) {
                                                if (!element.is_object()) {
                                                    return false;
                                                }
                                                const auto member = element.find("uuid");
                                                return member != element.end() && member->is_string() &&
                                                       member->get_ref<const std::string&>() == uuid;
                                            });
            (*candidate_array)->erase(found);
            root_ = std::move(candidate);
            return true;
        });
    }

    lfs::Result<std::vector<std::string>> JsonChapterDom::array_uuids(
        const std::string_view path) const {
        return guarded_dom_operation<std::vector<std::string>>(
            path, [&]() -> lfs::Result<std::vector<std::string>> {
                auto resolved = resolve_node(root_, path, "enumerate");
                if (!resolved) {
                    return std::move(resolved).error();
                }
                if (*resolved == nullptr) {
                    return std::vector<std::string>{};
                }
                if (!(*resolved)->is_array()) {
                    return make_dom_error(
                        lfs::ErrorCode::FailedPrecondition,
                        "The JSON chapter has an incompatible structure.",
                        std::format("Cannot enumerate '{}': value is {}, expected an array",
                                    path, (*resolved)->type_name()),
                        path);
                }

                std::vector<std::string> uuids;
                uuids.reserve((*resolved)->size());
                for (std::size_t i = 0; i < (*resolved)->size(); ++i) {
                    const Json& element = (**resolved)[i];
                    if (!element.is_object()) {
                        return make_dom_error(
                            lfs::ErrorCode::DataLoss,
                            "The JSON chapter contains an invalid UUID array.",
                            std::format("Element {} of '{}' is {}, expected an object",
                                        i, path, element.type_name()),
                            path);
                    }
                    const auto member = element.find("uuid");
                    if (member == element.end() || !member->is_string() ||
                        !is_canonical_uuid(member->get_ref<const std::string&>())) {
                        return make_dom_error(
                            lfs::ErrorCode::DataLoss,
                            "The JSON chapter contains an invalid UUID array.",
                            std::format("Element {} of '{}' has no canonical UUID", i, path),
                            path);
                    }
                    const std::string& uuid = member->get_ref<const std::string&>();
                    if (std::ranges::find(uuids, uuid) != uuids.end()) {
                        return make_dom_error(
                            lfs::ErrorCode::DataLoss,
                            "The JSON chapter contains a duplicate UUID.",
                            std::format("UUID '{}' occurs more than once in '{}'", uuid, path),
                            path, uuid);
                    }
                    uuids.push_back(uuid);
                }
                return uuids;
            });
    }

    lfs::Result<void> JsonChapterDom::set_element_value(
        const std::string_view array_path, const std::string_view uuid, const std::string_view path,
        Json value) {
        return guarded_dom_operation<void>(path, [&]() -> lfs::Result<void> {
            Json candidate = root_;
            Json* element = find_array_element_node(candidate, array_path, uuid);
            if (element == nullptr) {
                return lfs::Result<void>::failure(make_dom_error(
                    lfs::ErrorCode::NotFound, "The JSON chapter element no longer exists.",
                    std::format("UUID '{}' was not found in JSON array '{}'", uuid, array_path),
                    array_path, uuid));
            }
            auto result = set_value_at(*element, path, std::move(value));
            if (!result) {
                return result;
            }
            root_ = std::move(candidate);
            return {};
        });
    }

    lfs::Result<bool> JsonChapterDom::remove_element_value(
        const std::string_view array_path, const std::string_view uuid, const std::string_view path) {
        return guarded_dom_operation<bool>(path, [&]() -> lfs::Result<bool> {
            Json candidate = root_;
            Json* element = find_array_element_node(candidate, array_path, uuid);
            if (element == nullptr) {
                return make_dom_error(
                    lfs::ErrorCode::NotFound, "The JSON chapter element no longer exists.",
                    std::format("UUID '{}' was not found in JSON array '{}'", uuid, array_path),
                    array_path, uuid);
            }
            auto removed = remove_value_at(*element, path);
            if (!removed || !*removed) {
                return removed;
            }
            root_ = std::move(candidate);
            return true;
        });
    }

    lfs::Result<bool> JsonChapterDom::Element::remove(const std::string_view path) {
        return owner_->remove_element_value(array_path_, uuid_, path);
    }

    bool JsonChapterDom::Element::exists() const {
        return owner_->find_array_element_node(owner_->root_, array_path_, uuid_) != nullptr;
    }

    std::optional<JsonChapterDom::Json> JsonChapterDom::Element::get_json(
        const std::string_view path) const {
        const Json* element = owner_->find_array_element_node(owner_->root_, array_path_, uuid_);
        if (element == nullptr) {
            return std::nullopt;
        }
        const Json* value = owner_->find_node_from(*element, path);
        return value == nullptr ? std::nullopt : std::optional<Json>(*value);
    }

    lfs::Result<void> JsonChapterDom::Element::set_json(
        const std::string_view path, Json value) {
        return owner_->set_element_value(array_path_, uuid_, path, std::move(value));
    }

    bool JsonChapterDom::ConstElement::exists() const {
        return owner_->find_array_element_node(owner_->root_, array_path_, uuid_) != nullptr;
    }

    std::optional<JsonChapterDom::Json> JsonChapterDom::ConstElement::get_json(
        const std::string_view path) const {
        const Json* element = owner_->find_array_element_node(owner_->root_, array_path_, uuid_);
        if (element == nullptr) {
            return std::nullopt;
        }
        const Json* value = owner_->find_node_from(*element, path);
        return value == nullptr ? std::nullopt : std::optional<Json>(*value);
    }

} // namespace lfs::io

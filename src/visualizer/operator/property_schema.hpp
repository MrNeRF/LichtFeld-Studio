/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace lfs::vis::op {

    enum class PropertyType : uint8_t {
        BOOL,
        INT,
        FLOAT,
        STRING,
        ENUM,
        FLOAT_VECTOR,
        INT_VECTOR,
        TENSOR
    };

    struct PropertySchema {
        std::string name;
        std::string description;
        PropertyType type = PropertyType::STRING;
        std::string subtype;

        std::optional<double> min;
        std::optional<double> max;
        std::optional<int> maxlen;
        std::optional<int> size;
        std::optional<int> step;
        std::optional<int> precision;
        std::vector<std::tuple<std::string, std::string, std::string>> enum_items;
    };

    class LFS_VIS_API PropertySchemaRegistry {
    public:
        static PropertySchemaRegistry& instance();

        void registerSchema(const std::string& operator_id, std::vector<PropertySchema> properties);
        void unregisterSchema(const std::string& operator_id);

        [[nodiscard]] const std::vector<PropertySchema>* getSchema(const std::string& operator_id) const;
        [[nodiscard]] const PropertySchema* getPropertySchema(const std::string& operator_id,
                                                              const std::string& prop_name) const;

    private:
        PropertySchemaRegistry() = default;
        ~PropertySchemaRegistry() = default;
        PropertySchemaRegistry(const PropertySchemaRegistry&) = delete;
        PropertySchemaRegistry& operator=(const PropertySchemaRegistry&) = delete;

        mutable std::mutex mutex_;
        std::unordered_map<std::string, std::vector<PropertySchema>> schemas_;
    };

    inline PropertySchemaRegistry& propertySchemas() {
        return PropertySchemaRegistry::instance();
    }

} // namespace lfs::vis::op

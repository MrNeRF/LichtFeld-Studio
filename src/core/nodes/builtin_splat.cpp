/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
namespace lfs::nodes::builtin {
    namespace {
        constexpr glm::vec3 kRec709Luma(0.2126f, 0.7152f, 0.0722f);

        glm::mat3 hue_rotation(float degrees) {
            const float cosine = std::cos(glm::radians(degrees));
            const float sine = std::sin(glm::radians(degrees));
            constexpr float red = 0.2126f;
            constexpr float green = 0.7152f;
            constexpr float blue = 0.0722f;
            constexpr float green_from_red = (red * red + blue * (1 - red)) / green;
            constexpr float green_from_green = red - blue;
            constexpr float green_from_blue = -(red * (1 - blue) + blue * blue) / green;
            const float rows[3][3] = {
                {red + cosine * (1 - red) - sine * red,
                 green - cosine * green - sine * green,
                 blue - cosine * blue + sine * (1 - blue)},
                {red - cosine * red + sine * green_from_red,
                 green + cosine * (1 - green) + sine * green_from_green,
                 blue - cosine * blue + sine * green_from_blue},
                {red - cosine * red - sine * (1 - red),
                 green - cosine * green + sine * green,
                 blue + cosine * (1 - blue) + sine * blue}};
            glm::mat3 result(0);
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    result[column][row] = rows[row][column];
            return result;
        }

        Tensor rec709_luma(const Tensor& colour) {
            return channel(colour, 0) * kRec709Luma.r + channel(colour, 1) * kRec709Luma.g +
                   channel(colour, 2) * kRec709Luma.b;
        }

        std::pair<float, float> measured_luma_range(const Tensor& colour) {
            const size_t count = colour.shape()[0];
            if (count == 0)
                return {0, 1};
            constexpr size_t kMaximumSamples = 200'000;
            const size_t stride = std::max<size_t>((count + kMaximumSamples - 1) / kMaximumSamples, 1);
            const size_t samples = (count + stride - 1) / stride;
            const auto indices = (Tensor::linspace(0, static_cast<float>(samples - 1), samples, colour.device()) *
                                  static_cast<float>(stride))
                                     .to(DataType::Int32);
            const auto luma = rec709_luma(colour.index_select(0, indices)).to(DataType::Float32);
            const auto sorted = luma.sort(0, false).first;
            const size_t last = sorted.numel() - 1;
            const float low = sorted.slice(0, last / 100, last / 100 + 1).item<float>();
            const size_t high_index = last * 99 / 100;
            const float high = sorted.slice(0, high_index, high_index + 1).item<float>();
            return high - low < 1e-3f ? std::pair{0.0f, 1.0f} : std::pair{low, high};
        }
    } // namespace

    void evaluate_set_colour(NodeContext& context) {
        Geometry geometry = geometry_input(context);
        if (geometry.splats) {
            auto& s = *geometry.splats;
            auto fc = field_context(s);
            auto w = selection(context, "Selection", fc);
            auto color = context.evaluate_field("Colour", fc, COLOUR_SOCKET);
            auto base = s.sh0 * kShC0 + 0.5f;
            s.sh0 = (blend(base, color, w) - 0.5f) / kShC0;
            if (property_bool(context, "clear_view_dependent", true)) {
                auto expanded = w.unsqueeze(1).unsqueeze(2);
                s.shN = s.shN * (expanded.neg() + 1.0f);
            }
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_set_opacity(NodeContext& context) {
        Geometry geometry = geometry_input(context);
        if (geometry.splats) {
            auto& s = *geometry.splats;
            auto fc = field_context(s);
            auto w = selection(context, "Selection", fc);
            auto value = context.evaluate_field("Opacity", fc, FLOAT_SOCKET).clamp(1e-6f, 1.0f - 1e-6f);
            s.opacity = blend(s.opacity.sigmoid(), value, w).clamp(1e-6f, 1.0f - 1e-6f).logit();
        }
        context.set_output("Geometry", std::move(geometry));
    }
    void evaluate_set_scale(NodeContext& context) {
        Geometry geometry = geometry_input(context);
        if (geometry.splats) {
            auto& s = *geometry.splats;
            auto fc = field_context(s);
            auto w = selection(context, "Selection", fc);
            auto value = context.evaluate_field("Scale", fc, VECTOR_SOCKET).clamp_min(1e-8f);
            s.scaling = blend(s.scaling.exp(), value, w).clamp_min(1e-8f).log();
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_set_degree(NodeContext& context) {
        Geometry geometry = geometry_input(context);
        if (geometry.splats) {
            auto& s = *geometry.splats;
            int degree = std::clamp(input_int(context, "Degree", 0), 0, 3);
            size_t coeff = (degree + 1) * (degree + 1) - 1;
            if (coeff == 0)
                s.shN = core::Tensor::zeros({s.means.shape()[0], 0, 3}, s.means.device());
            else if (s.shN.shape()[1] >= coeff)
                s.shN = s.shN.slice(1, 0, coeff).contiguous();
            else {
                auto padded = core::Tensor::zeros({s.means.shape()[0], coeff, 3}, s.means.device());
                if (s.shN.shape()[1])
                    padded.slice(1, 0, s.shN.shape()[1]).copy_from(s.shN);
                s.shN = std::move(padded);
            }
            s.sh_degree = degree;
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_sharpen(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.splats) {
            auto& splats = *geometry.splats;
            const auto weight = selection(context, "Selection", field_context(splats));
            const float shrink = 1 - input_float(context, "Amount");
            splats.scaling = blend(splats.scaling, splats.scaling + std::log(shrink), weight);
            if (property_bool(context, "keep_coverage", true)) {
                const auto opacity = splats.opacity.sigmoid();
                splats.opacity = blend(opacity, (opacity / shrink).clamp(1e-6f, 1 - 1e-6f), weight)
                                     .clamp(1e-6f, 1 - 1e-6f)
                                     .logit();
            }
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_scale_clamp(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.splats) {
            auto& splats = *geometry.splats;
            const auto weight = selection(context, "Selection", field_context(splats));
            const float limit = std::log(input_float(context, "Max Aspect", 10));
            const auto centre = (splats.scaling.max(1, true) + splats.scaling.min(1, true)) * 0.5f;
            const auto clamped = splats.scaling.maximum(centre - limit * 0.5f).minimum(centre + limit * 0.5f);
            splats.scaling = blend(splats.scaling, clamped, weight);
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_colour_correct(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.splats) {
            auto& splats = *geometry.splats;
            const float exposure = input_float(context, "Exposure");
            float black = input_float(context, "Black Point");
            float white = input_float(context, "White Point", 1);
            const float midpoint = input_float(context, "Midpoint", 0.5f);
            const float temperature = input_float(context, "Temperature");
            const float tint = input_float(context, "Tint");
            const float contrast = input_float(context, "Contrast", 1);
            const float saturation = input_float(context, "Saturation", 1);
            const float hue = input_float(context, "Hue Shift");
            const float gamma = input_float(context, "Gamma", 1);
            const glm::vec3 shadows = input_vector(context, "Shadows");
            const glm::vec3 midtones = input_vector(context, "Midtones");
            const glm::vec3 highlights = input_vector(context, "Highlights");
            const bool midpoint_identity = midpoint == 0.5f;
            const bool grading_identity = shadows == glm::vec3(0) && midtones == glm::vec3(0) &&
                                          highlights == glm::vec3(0);
            const auto base = splats.sh0 * kShC0 + 0.5f;
            const float exposure_scale = std::exp2(exposure);
            if (property_bool(context, "auto_range", false)) {
                const auto [low, high] = measured_luma_range(base * exposure_scale);
                const float span = high - low;
                black = low + black * span;
                white = low + white * span;
            }
            const float level_range = std::max(white - black, 1e-4f);
            const bool affine_identity = exposure == 0 && black == 0 && white == 1 && temperature == 0 &&
                                         tint == 0 && contrast == 1 && saturation == 1 && hue == 0;
            if (affine_identity && midpoint_identity && grading_identity && gamma == 1) {
                context.set_output("Geometry", std::move(geometry));
                return;
            }
            glm::mat3 matrix(0);
            const float affine_scale = exposure_scale / level_range;
            matrix[0][0] = affine_scale;
            matrix[1][1] = affine_scale;
            matrix[2][2] = affine_scale;
            glm::vec3 offset(-black / level_range);
            glm::mat3 white_balance(0);
            white_balance[0][0] = 1 + temperature;
            white_balance[1][1] = 1 + tint;
            white_balance[2][2] = 1 - temperature;
            matrix = white_balance * matrix;
            offset = white_balance * offset;
            matrix *= contrast;
            offset = (offset - glm::vec3(0.5f)) * contrast + glm::vec3(0.5f);
            glm::mat3 saturation_matrix(0);
            for (int column = 0; column < 3; ++column)
                for (int row = 0; row < 3; ++row)
                    saturation_matrix[column][row] =
                        (1 - saturation) * kRec709Luma[column] + (row == column ? saturation : 0);
            const auto hue_matrix = hue_rotation(hue);
            matrix = hue_matrix * saturation_matrix * matrix;
            offset = hue_matrix * saturation_matrix * offset;
            const auto weight = selection(context, "Selection", field_context(splats));
            const auto affine = matrix_tensor(matrix, splats.means.device());
            auto corrected =
                affine_identity ? base : base.matmul(affine) + vector_tensor(offset, base.device());
            if (!midpoint_identity) {
                const float exponent = std::log(0.5f) / std::log(midpoint);
                corrected = corrected.sign() * corrected.abs().pow(exponent);
            }
            if (!grading_identity) {
                const auto luma = rec709_luma(corrected).clamp(0, 1);
                const auto shadow_weight = (luma.neg() * 2 + 1).clamp(0, 1).pow(2).unsqueeze(1);
                const auto highlight_weight = (luma * 2 - 1).clamp(0, 1).pow(2).unsqueeze(1);
                const auto midtone_weight = ((luma * 2 - 1).abs().neg() + 1).clamp(0, 1).pow(2).unsqueeze(1);
                corrected = corrected + vector_tensor(shadows * 0.35f, corrected.device()) * shadow_weight +
                            vector_tensor(midtones * 0.35f, corrected.device()) * midtone_weight +
                            vector_tensor(highlights * 0.35f, corrected.device()) * highlight_weight;
            }
            if (gamma != 1)
                corrected = corrected.maximum(0).pow(1 / gamma);
            splats.sh0 = (blend(base, corrected, weight) - 0.5f) / kShC0;
            if (!affine_identity && splats.shN.numel()) {
                const auto transformed =
                    splats.shN.reshape({-1, 3}).matmul(affine).reshape(splats.shN.shape());
                splats.shN = blend(splats.shN, transformed, weight);
            }
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_recolour(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.splats) {
            auto& splats = *geometry.splats;
            const float amount = input_float(context, "Weight", 1);
            if (amount > 0) {
                const auto selected = selection(context, "Selection", field_context(splats));
                const auto weight = selected * amount;
                const auto base = splats.sh0 * kShC0 + 0.5f;
                const glm::vec3 target_value = input_vector(context, "Colour");
                auto target = vector_tensor(target_value, base.device());
                if (property_bool(context, "keep_shading", true)) {
                    const float target_luma = std::max(glm::dot(target_value, kRec709Luma), 1e-4f);
                    target = target * (rec709_luma(base) / target_luma).unsqueeze(1);
                }
                splats.sh0 = (blend(base, target, weight) - 0.5f) / kShC0;
                if (property_bool(context, "fade_view_dependent", true) && splats.shN.numel())
                    splats.shN = splats.shN * (weight.neg() + 1).unsqueeze(1).unsqueeze(2);
            }
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_invert(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.splats) {
            auto& splats = *geometry.splats;
            const auto weight = selection(context, "Selection", field_context(splats));
            const auto base = splats.sh0 * kShC0 + 0.5f;
            splats.sh0 = (blend(base, base.neg() + 1, weight) - 0.5f) / kShC0;
            splats.shN = blend(splats.shN, -splats.shN, weight);
        }
        context.set_output("Geometry", std::move(geometry));
    }
    void register_splat(NodeTypeRegistry& registry) {
        const auto geo = std::string(GEOMETRY_SOCKET);
        const auto f = std::string(FLOAT_SOCKET);
        const auto i = std::string(INT_SOCKET);
        const auto v = std::string(VECTOR_SOCKET);
        const auto c = std::string(COLOUR_SOCKET);
        register_type(registry, type("lfs.set_colour", "Splat",
                                     geometry_inputs({in("Selection", f, 1.0f, true)
                                                          .range(0, 1)
                                                          .step_size(0.01),
                                                      in("Colour", c, glm::vec3(0.5f), true).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_set_colour,
                                     {prop("clear_view_dependent", PropertyKind::Bool, true)}));
        register_type(registry,
                      type("lfs.set_opacity", "Splat",
                           geometry_inputs({in("Selection", f, 1.0f, true)
                                                .range(0, 1)
                                                .step_size(0.01),
                                            in("Opacity", f, 1.0f, true).range(0, 1).step_size(0.01)}),
                           {out("Geometry", geo)}, evaluate_set_opacity));
        register_type(registry, type("lfs.set_scale", "Splat",
                                     geometry_inputs({in("Selection", f, 1.0f, true)
                                                          .range(0, 1)
                                                          .step_size(0.01),
                                                      in("Scale", v, glm::vec3(1), true)
                                                          .minimum(0)
                                                          .step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_set_scale));
        register_type(registry, type("lfs.set_sh_degree", "Splat",
                                     geometry_inputs({in("Degree", i, std::int64_t(0)).range(0, 3).step_size(1)}),
                                     {out("Geometry", geo)}, evaluate_set_degree));
        register_type(registry, type("lfs.sharpen", "Splat",
                                     geometry_inputs({in("Selection", f, 1.0f, true)
                                                          .range(0, 1)
                                                          .step_size(0.01),
                                                      in("Amount", f, 0.25f).range(0, 0.95).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_sharpen,
                                     {prop("keep_coverage", PropertyKind::Bool, true)}));
        register_type(registry,
                      type("lfs.scale_clamp", "Splat",
                           geometry_inputs({in("Selection", f, 1.0f, true)
                                                .range(0, 1)
                                                .step_size(0.01),
                                            in("Max Aspect", f, 10.0f).minimum(1).step_size(0.1)}),
                           {out("Geometry", geo)}, evaluate_scale_clamp));
        register_type(registry,
                      type("lfs.colour_correct", "Colour",
                           geometry_inputs({in("Selection", f, 1.0f, true)
                                                .range(0, 1)
                                                .step_size(0.01),
                                            in("Exposure", f, 0.0f).soft_range(-10, 10).step_size(0.1),
                                            in("Black Point", f, 0.0f).step_size(0.01),
                                            in("White Point", f, 1.0f).step_size(0.01),
                                            in("Midpoint", f, 0.5f).range(0.01, 0.99).step_size(0.01),
                                            in("Contrast", f, 1.0f).minimum(0).step_size(0.01),
                                            in("Saturation", f, 1.0f).minimum(0).step_size(0.01),
                                            in("Hue Shift", f, 0.0f).step_size(1),
                                            in("Temperature", f, 0.0f).range(-1, 1).step_size(0.01),
                                            in("Tint", f, 0.0f).range(-1, 1).step_size(0.01),
                                            in("Shadows", c, glm::vec3(0)).range(-1, 1).step_size(0.01),
                                            in("Midtones", c, glm::vec3(0)).range(-1, 1).step_size(0.01),
                                            in("Highlights", c, glm::vec3(0)).range(-1, 1).step_size(0.01),
                                            in("Gamma", f, 1.0f).minimum(0.001).step_size(0.01)}),
                           {out("Geometry", geo)}, evaluate_colour_correct,
                           {prop("auto_range", PropertyKind::Bool, false)}));
        register_type(registry,
                      type("lfs.recolour", "Colour",
                           geometry_inputs({in("Selection", f, 1.0f, true)
                                                .range(0, 1)
                                                .step_size(0.01),
                                            in("Colour", c, glm::vec3(0.15f, 0.30f, 0.85f)).step_size(0.01),
                                            in("Weight", f, 1.0f).range(0, 1).step_size(0.01)}),
                           {out("Geometry", geo)}, evaluate_recolour,
                           {prop("keep_shading", PropertyKind::Bool, true),
                            prop("fade_view_dependent", PropertyKind::Bool, true)}));
        register_type(registry, type("lfs.invert_colour", "Colour",
                                     geometry_inputs(
                                         {in("Selection", f, 1.0f, true).range(0, 1).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_invert));
    }

} // namespace lfs::nodes::builtin

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error_typed.hpp"
#include "core/image_io.hpp"
#include "core/nn/models/bert_text.hpp"
#include "core/nn/models/llmdet.hpp"
#include "core/nn/models/llmdet_encoder.hpp"
#include "core/nn/models/swin_tiny.hpp"
#include "core/path_utils.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "preprocessing/preprocess.hpp"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
#include <optional>

using namespace lfs::core;
using nn::models::BertText;
using nn::models::LlmDetEncoder;
using nn::models::SwinTiny;

namespace {
    void expect_reference(const Tensor& feature, const nlohmann::json& expected, double& largest_error) {
        ASSERT_EQ(feature.shape(), TensorShape(expected.at("shape").get<std::vector<size_t>>()));
        const auto values = feature.to(Device::CPU).to_vector();
        for (size_t i = 0; i < expected.at("indices").size(); ++i) {
            const auto index = expected.at("indices")[i].get<size_t>();
            ASSERT_LT(index, values.size());
            const auto reference = expected.at("values")[i].get<double>();
            largest_error = std::max(largest_error, std::abs(values[index] - reference));
            EXPECT_NEAR(values[index], reference, 1e-3);
        }
        const double sum = std::accumulate(values.begin(), values.end(), 0.0);
        const double squared = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
        EXPECT_NEAR(sum / values.size(), expected.at("mean").get<double>(), 1e-4);
        EXPECT_NEAR(squared, expected.at("squared_norm").get<double>(), squared * 1e-4);
    }
} // namespace

TEST(BertTextContract, RejectsMissingWeights) {
    const BertText unloaded;
    const nn::WeightFile missing;
    EXPECT_FALSE(BertText::load(missing, "", Device::CPU));
    EXPECT_FALSE(BertText::load(missing, "", Device::CUDA));
    const std::vector<int> tokens{101, 2711, 102}, positions{0, 0, 0};
    std::vector<float> mask(9, 0);
    EXPECT_FALSE(unloaded.encode(tokens, positions, mask));
}

TEST(SwinTinyContract, RejectsMissingWeights) {
    const SwinTiny unloaded;
    const nn::WeightFile missing;
    EXPECT_FALSE(SwinTiny::load(missing, Device::CPU));
    EXPECT_FALSE(SwinTiny::load(missing, Device::CUDA));
    EXPECT_FALSE(unloaded.encode(Tensor{}));
}

TEST(LlmDetEncoderContract, RejectsMissingWeights) {
    const LlmDetEncoder unloaded;
    const nn::WeightFile missing;
    EXPECT_FALSE(LlmDetEncoder::load(missing, Device::CPU));
    EXPECT_FALSE(LlmDetEncoder::load(missing, Device::CUDA));
    EXPECT_FALSE(unloaded.encode({}, Tensor{}, {}, {}));
}

TEST(LlmDetContract, ResizeGeometryMatchesHuggingFaceAndRejectsTooFewProposals) {
    using nn::models::LlmDet;
    EXPECT_EQ(LlmDet::input_size(1600, 3840), (std::array{555, 1333}));
    EXPECT_EQ(LlmDet::input_size(3840, 1600), (std::array{1333, 555}));
    EXPECT_EQ(LlmDet::input_size(3000, 4000), (std::array{800, 1066}));
    EXPECT_EQ(LlmDet::input_size(800, 1066), (std::array{800, 1066}));
    EXPECT_FALSE(LlmDet::input_size(0, 800));
    EXPECT_FALSE(LlmDet::input_size(60, 3000));
    EXPECT_FALSE(LlmDet::input_size(3000, 60));
    EXPECT_TRUE(LlmDet::input_size(74, 3000));
}

TEST(LlmDetContract, RejectsMissingWeightsAndUnboundedDescriptions) {
    using nn::models::LlmDet;
    const LlmDet unloaded;
    const nn::WeightFile missing;
    EXPECT_FALSE(LlmDet::load(missing, Device::CPU));
    EXPECT_FALSE(nn::models::LlmDetDecoder::load(missing, Device::CUDA));
    EXPECT_FALSE(unloaded.forward(Tensor{}, {}));
    std::vector<std::vector<int>> phrases{{101, 2711, 1012, 102}};
    EXPECT_TRUE(LlmDet::valid_phrases(phrases));
    EXPECT_FALSE(unloaded.encode(phrases));
    phrases.assign(17, phrases.front());
    EXPECT_FALSE(LlmDet::valid_phrases(phrases));
    phrases = {{101, 1012, 1012, 102}};
    EXPECT_FALSE(LlmDet::valid_phrases(phrases));
    phrases = {std::vector<int>(255, 2711), std::vector<int>{101, 2711, 1012, 102}};
    phrases[0].front() = 101;
    phrases[0][253] = 1012;
    phrases[0].back() = 102;
    EXPECT_FALSE(LlmDet::valid_phrases(phrases)); // Context cap is shared across descriptions.
}

class LlmDetBackboneParity : public ::testing::Test {
public:
    ~LlmDetBackboneParity() override {
        if (stream) {
            LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "Backbone upload drain");
            LFS_CUDA_LOG_STREAM_DESTROY(stream, "Backbone stream teardown");
        }
    }

protected:
    std::optional<nn::WeightFile> archive;
    cudaStream_t stream = nullptr;
    void SetUp() override {
        const auto cached = lfs::preprocessing::inspect_cached_model(lfs::preprocessing::CachedModel::LlmDet);
        ASSERT_TRUE(cached);
        if (!cached->present)
            GTEST_SKIP() << "LLMDet model is not installed";
        auto opened = nn::WeightFile::open(cached->path);
        ASSERT_TRUE(opened);
        archive.emplace(std::move(*opened));
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    }
    void TearDown() override {
        if (stream)
            CudaMemoryPool::instance().release_stream(stream);
    }
};

TEST_F(LlmDetBackboneParity, RealPeopleImageDistinguishesPresentAndAbsentCategories) {
    CUDAStreamGuard guard(stream);
    auto model = nn::models::LlmDet::load(*archive, Device::CUDA);
    ASSERT_TRUE(model);
    auto [pixels, width, height, channels] = load_image(std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/nn/semantic_people.png");
    ASSERT_EQ(channels, 3);
    std::vector<float> rgb(size_t(width) * height * 3);
    for (size_t i = 0; i < rgb.size(); ++i)
        rgb[i] = pixels[i] / 255.f;
    free_image(pixels);
    const auto image = Tensor::from_vector(rgb, {1, size_t(height), size_t(width), 3}, Device::CPU)
                           .to(Device::CUDA)
                           .permute({0, 3, 1, 2})
                           .contiguous();
    for (const int token : {2711, 2482, 3899, 3712}) {
        const std::vector<std::vector<int>> phrases{{101, token, 1012, 102}};
        auto text = model->encode(phrases);
        ASSERT_TRUE(text);
        auto predictions = model->forward(image, *text);
        ASSERT_TRUE(predictions);
        const auto logits = predictions->logits.to(Device::CPU).to_vector();
        const size_t length = text->positions.size();
        float score = 0;
        for (size_t i = 0; i < 900; ++i)
            score = std::max(score, 1.f / (1.f + std::exp(-logits[i * length + 1])));
        SCOPED_TRACE("BERT category token " + std::to_string(token));
        // Official LLMDet Tiny detection threshold; absent classes must not
        // generate image-wide SAM proposals on this real two-person frame.
        if (token == 2711)
            EXPECT_GT(score, .4f);
        else
            EXPECT_LT(score, .4f);
    }
}

TEST_F(LlmDetBackboneParity, BertMatchesFloat64ReferenceForIsolatedCategorySpans) {
    CUDAStreamGuard guard(stream);
    std::ifstream input(std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/nn/bert_text_ref_fixture.json");
    ASSERT_TRUE(input);
    nlohmann::json fixture;
    input >> fixture;
    ASSERT_EQ(archive->meta().at("source_sha256"), fixture.at("checkpoint_sha256"));
    auto model = BertText::load(*archive, "model.text_backbone.", Device::CUDA);
    ASSERT_TRUE(model);
    for (const auto& item : fixture.at("cases")) {
        const auto tokens = item.at("tokens").get<std::vector<int>>();
        const auto positions = item.at("positions").get<std::vector<int>>();
        std::vector<float> mask(tokens.size() * tokens.size(), -INFINITY);
        for (const auto& block : item.at("blocks"))
            for (size_t row = block[0]; row < block[1]; ++row)
                for (size_t col = block[0]; col < block[1]; ++col)
                    mask[row * tokens.size() + col] = 0;
        auto result = model->encode(tokens, positions, mask);
        ASSERT_TRUE(result);
        const auto values = result->to(Device::CPU).to_vector();
        ASSERT_EQ(values.size(), tokens.size() * 768);
        for (size_t i = 0; i < item.at("indices").size(); ++i) {
            const auto index = item.at("indices")[i].get<size_t>();
            ASSERT_LT(index, values.size());
            EXPECT_NEAR(values[index], item.at("values")[i].get<double>(), 3e-4);
        }
        const double sum = std::accumulate(values.begin(), values.end(), 0.0);
        const double squared = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
        EXPECT_NEAR(sum / values.size(), item.at("mean").get<double>(), 1e-5);
        EXPECT_NEAR(squared, item.at("squared_norm").get<double>(), squared * 1e-4);
        // An entirely masked row must be rejected before attention produces NaN.
        mask[0] = -INFINITY;
        EXPECT_FALSE(model->encode(tokens, positions, mask));
        mask[0] = 0;
        auto invalid_positions = positions;
        invalid_positions[0] = 512;
        EXPECT_FALSE(model->encode(tokens, invalid_positions, mask));
    }
}

TEST_F(LlmDetBackboneParity, SwinMatchesFloat64ReferenceWithOddPaddingAndShiftedBorders) {
    CUDAStreamGuard guard(stream);
    std::ifstream input(std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/nn/swin_tiny_ref_fixture.json");
    ASSERT_TRUE(input);
    nlohmann::json fixture;
    input >> fixture;
    ASSERT_EQ(archive->meta().at("source_sha256"), fixture.at("checkpoint_sha256"));
    auto model = SwinTiny::load(*archive, Device::CUDA);
    ASSERT_TRUE(model);
    double largest_error = 0;
    for (const auto& item : fixture.at("cases")) {
        const auto height = item.at("height").get<size_t>(), width = item.at("width").get<size_t>();
        std::vector<float> pixels(3 * height * width);
        for (size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = float(int(i * 37 % 251) - 125) / 127.f;
        const auto image = Tensor::from_vector(pixels, {1, 3, height, width}, Device::CPU).to(Device::CUDA);
        auto result = model->encode(image);
        ASSERT_TRUE(result);
        ASSERT_EQ(item.at("outputs").size(), result->size());
        for (size_t stage = 0; stage < result->size(); ++stage) {
            const auto& expected = item.at("outputs")[stage];
            const auto& feature = (*result)[stage];
            expect_reference(feature, expected, largest_error);
        }
    }
    RecordProperty("maximum_sample_error", largest_error);
}

TEST_F(LlmDetBackboneParity, EncoderMatchesFloat64MultimodalReferenceAcrossSixLayers) {
    CUDAStreamGuard guard(stream);
    std::ifstream input(std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/nn/llmdet_encoder_ref_fixture.json");
    ASSERT_TRUE(input);
    nlohmann::json fixture;
    input >> fixture;
    ASSERT_EQ(archive->meta().at("source_sha256"), fixture.at("checkpoint_sha256"));
    auto model = LlmDetEncoder::load(*archive, Device::CUDA);
    ASSERT_TRUE(model);
    const auto synthetic = [](const TensorShape& shape) {
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = float(int(i * 37 % 251) - 125) / 128.f;
        return Tensor::from_vector(data, shape, Device::CPU).to(Device::CUDA);
    };
    const std::array<Tensor, 3> features{synthetic({1, 192, 3, 5}), synthetic({1, 384, 2, 3}), synthetic({1, 768, 1, 2})};
    const auto text = synthetic({1, 6, 768});
    const std::vector<int> positions{0, 0, 1, 0, 1, 0};
    std::vector<float> mask(36, -INFINITY);
    for (const auto [first, last] : {std::pair{0, 1}, {1, 3}, {3, 5}, {5, 6}})
        for (int row = first; row < last; ++row)
            for (int col = first; col < last; ++col)
                mask[row * 6 + col] = 0;
    auto result = model->encode(features, text, positions, mask);
    ASSERT_TRUE(result);
    for (size_t i = 0; i < result->spatial_shapes.size(); ++i) {
        EXPECT_EQ(result->spatial_shapes[i].first, fixture.at("spatial_shapes")[i][0].get<int>());
        EXPECT_EQ(result->spatial_shapes[i].second, fixture.at("spatial_shapes")[i][1].get<int>());
    }
    double largest_error = 0;
    expect_reference(result->vision, fixture.at("vision"), largest_error);
    expect_reference(result->text, fixture.at("text"), largest_error);
    RecordProperty("maximum_sample_error", largest_error);
    mask[0] = -INFINITY;
    EXPECT_FALSE(model->encode(features, text, positions, mask));
}

TEST_F(LlmDetBackboneParity, DecoderMatchesIndependentReferenceThroughFinalBoxRefinement) {
    CUDAStreamGuard guard(stream);
    std::ifstream input(std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/nn/llmdet_decoder_ref_fixture.json");
    ASSERT_TRUE(input);
    nlohmann::json fixture;
    input >> fixture;
    ASSERT_EQ(archive->meta().at("source_sha256"), fixture.at("checkpoint_sha256"));
    auto model = nn::models::LlmDetDecoder::load(*archive, Device::CUDA);
    ASSERT_TRUE(model);
    nn::models::LlmDetEncoded encoded;
    size_t count = 0;
    for (size_t i = 0; i < 4; ++i) {
        encoded.spatial_shapes[i] = {fixture.at("spatial_shapes")[i][0].get<int>(), fixture.at("spatial_shapes")[i][1].get<int>()};
        count += size_t(encoded.spatial_shapes[i].first) * encoded.spatial_shapes[i].second;
    }
    const auto source_rows = fixture.at("source_rows").get<std::vector<size_t>>();
    ASSERT_EQ(source_rows.size(), count);
    const auto synthetic = [](const std::vector<size_t>& rows) {
        std::vector<float> values(rows.size() * 256);
        for (size_t row = 0; row < rows.size(); ++row)
            for (size_t c = 0; c < 256; ++c) {
                const auto bits = uint32_t((rows[row] * 256 + c) * 1664525ull + 1013904223ull);
                values[row * 256 + c] = float(int((bits >> 8) & 65535) - 32768) / 32768.f;
            }
        return Tensor::from_vector(values, {1, rows.size(), 256}, Device::CPU).to(Device::CUDA);
    };
    encoded.vision = synthetic(source_rows);
    encoded.text = synthetic({0, 1, 2, 3, 4, 5});
    auto result = model->forward(encoded);
    ASSERT_TRUE(result);
    double largest_error = 0;
    expect_reference(result->boxes, fixture.at("boxes"), largest_error);
    expect_reference(result->logits, fixture.at("logits"), largest_error);
    RecordProperty("maximum_sample_error", largest_error);
    encoded.spatial_shapes[0].first += 1;
    EXPECT_FALSE(model->forward(encoded));
}

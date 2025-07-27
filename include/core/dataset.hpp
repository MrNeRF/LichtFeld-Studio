#pragma once

#include "core/camera.hpp"
#include "core/colmap_reader.hpp"
#include "core/parameters.hpp"
#include "core/transforms_reader.hpp"
#include <expected>
#include <format>
#include <memory>
#include <torch/torch.h>
#include <vector>

// Camera with loaded image
struct CameraWithImage {
    Camera* camera;
    torch::Tensor image;
};

using CameraExample = torch::data::Example<CameraWithImage, torch::Tensor>;

class CameraDataset : public torch::data::Dataset<CameraDataset, CameraExample> {
public:
    enum class Split {
        TRAIN,
        VAL,
        ALL
    };

    CameraDataset(std::vector<std::shared_ptr<Camera>> cameras,
                  const gs::param::DatasetConfig& params,
                  Split split = Split::ALL)
        : _cameras(std::move(cameras)),
          _datasetConfig(params),
          _split(split) {

        // Create indices based on split
        _indices.clear();
        for (size_t i = 0; i < _cameras.size(); ++i) {
            const bool is_test = (i % params.test_every) == 0;

            if (_split == Split::ALL ||
                (_split == Split::TRAIN && !is_test) ||
                (_split == Split::VAL && is_test)) {
                _indices.push_back(i);
            }
        }

        std::cout << "Dataset created with " << _indices.size()
                  << " images (split: " << static_cast<int>(_split) << ")" << std::endl;
    }
    // Default copy constructor works with shared_ptr
    CameraDataset(const CameraDataset&) = default;
    CameraDataset(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(const CameraDataset&) = default;

    void preload_data() {
        if (!_image_cache.empty()) {
            std::cout << "Dataset already preloaded." << std::endl;
            return;
        }

        std::cout << "Preloading dataset into RAM... This may take a moment." << std::endl;
        _image_cache.reserve(_indices.size());

        for (size_t i = 0; i < _indices.size(); ++i) {
            size_t camera_idx = _indices[i];
            auto& cam = _cameras[camera_idx];
            torch::Tensor image = cam->load_and_get_image(_datasetConfig.resolution);
            _image_cache.push_back(image.clone());
        }
        std::cout << "Dataset preloading complete." << std::endl;
    }

    CameraExample get(size_t index) override {
        if (index >= _indices.size()) {
            throw std::out_of_range("Dataset index out of range");
        }

        size_t camera_idx = _indices[index];
        auto& cam = _cameras[camera_idx];

        if (!_image_cache.empty() && _image_cache.size() > index) {
            // Get tensors directly from RAM cache
            return {{cam.get(), _image_cache[index]}, torch::empty({})};
        } else {
            // Fallback to loading from disk if not preloaded
            torch::Tensor image = cam->load_and_get_image(_datasetConfig.resolution);
            return {{cam.get(), std::move(image)}, torch::empty({})};
        }
    }

    torch::optional<size_t> size() const override {
        return _indices.size();
    }

    const std::vector<std::shared_ptr<Camera>>& get_cameras() const {
        return _cameras;
    }

    Split get_split() const { return _split; }

private:
    std::vector<std::shared_ptr<Camera>> _cameras;
    const gs::param::DatasetConfig& _datasetConfig;
    Split _split;
    std::vector<size_t> _indices;
    std::vector<torch::Tensor> _image_cache;
};

inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
create_dataset_from_colmap(const gs::param::DatasetConfig& datasetConfig) {

    try {
        if (!std::filesystem::exists(datasetConfig.data_path)) {
            return std::unexpected(std::format("Data path does not exist: {}",
                                               datasetConfig.data_path.string()));
        }

        // Read COLMAP data with specified images folder
        auto [camera_infos, scene_center] = read_colmap_cameras_and_images(
            datasetConfig.data_path, datasetConfig.images);

        std::vector<std::shared_ptr<Camera>> cameras;
        cameras.reserve(camera_infos.size());

        for (size_t i = 0; i < camera_infos.size(); ++i) {
            const auto& info = camera_infos[i];

            auto cam = std::make_shared<Camera>(
                info._R,
                info._T,
                info._focal_x,
                info._focal_y,
                info._center_x,
                info._center_y,
                info._radial_distortion,
                info._tangential_distortion,
                info._camera_model_type,
                info._image_name,
                info._image_path,
                info._width,
                info._height,
                static_cast<int>(i));

            cameras.push_back(std::move(cam));
        }

        // Create dataset with ALL images
        auto dataset = std::make_shared<CameraDataset>(
            std::move(cameras), datasetConfig, CameraDataset::Split::ALL);

        return std::make_tuple(dataset, scene_center);

    } catch (const std::exception& e) {
        return std::unexpected(std::format("Failed to create dataset from COLMAP: {}", e.what()));
    }
}

inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
create_dataset_from_transforms(const gs::param::DatasetConfig& datasetConfig) {

    try {
        if (!std::filesystem::exists(datasetConfig.data_path)) {
            return std::unexpected(std::format("Data path does not exist: {}",
                                               datasetConfig.data_path.string()));
        }

        // Read COLMAP data with specified images folder
        auto [camera_infos, scene_center] = read_transforms_cameras_and_images(
            datasetConfig.data_path);

        std::vector<std::shared_ptr<Camera>> cameras;
        cameras.reserve(camera_infos.size());

        for (size_t i = 0; i < camera_infos.size(); ++i) {
            const auto& info = camera_infos[i];

            auto cam = std::make_shared<Camera>(
                info._R,
                info._T,
                info._focal_x,
                info._focal_y,
                info._center_x,
                info._center_y,
                info._radial_distortion,
                info._tangential_distortion,
                info._camera_model_type,
                info._image_name,
                info._image_path,
                info._width,
                info._height,
                static_cast<int>(i));

            cameras.push_back(std::move(cam));
        }

        // Create dataset with ALL images
        auto dataset = std::make_shared<CameraDataset>(
            std::move(cameras), datasetConfig, CameraDataset::Split::ALL);

        return std::make_tuple(dataset, scene_center);

    } catch (const std::exception& e) {
        return std::unexpected(std::format("Failed to create dataset from COLMAP: {}", e.what()));
    }
}

inline auto create_dataloader_from_dataset(
    std::shared_ptr<CameraDataset> dataset,
    int num_workers = 4) {

    const size_t dataset_size = dataset->size().value();

    return torch::data::make_data_loader(
        *dataset,
        torch::data::samplers::RandomSampler(dataset_size),
        torch::data::DataLoaderOptions()
            .batch_size(1)
            .workers(num_workers)
            .enforce_ordering(false));
}
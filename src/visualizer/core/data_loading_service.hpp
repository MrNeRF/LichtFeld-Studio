#pragma once

#include "core/events.hpp"
#include "core/parameters.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace gs {
    class SceneManager;
}

namespace gs::visualizer {
    class ViewerStateManager;

    class DataLoadingService {
    public:
        DataLoadingService(SceneManager* scene_manager,
                           ViewerStateManager* state_manager);
        ~DataLoadingService();

        // Set parameters for dataset loading
        void setParameters(const param::TrainingParameters& params) { params_ = params; }
        const param::TrainingParameters& getParameters() const { return params_; }

        // Loading operations
        std::expected<void, std::string> loadPLY(const std::filesystem::path& path);
        std::expected<void, std::string> loadDataset(const std::filesystem::path& path);
        void clearScene();

    private:
        void setupEventHandlers();
        void handleLoadFileCommand(const events::cmd::LoadFile& cmd);

        SceneManager* scene_manager_;
        ViewerStateManager* state_manager_;
        param::TrainingParameters params_;
    };

} // namespace gs::visualizer

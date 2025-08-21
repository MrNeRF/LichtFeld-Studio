#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/parameters.hpp"

namespace gs::management {

    // Version structure for semantic versioning
    struct Version {
        int major;
        int minor;
        int patch;

        Version(int maj = 0, int min = 0, int p = 1) : major(maj),
                                                       minor(min),
                                                       patch(p) {}
        Version(const std::string& versionStr);

        std::string toString() const;
        bool operator>=(const Version& other) const;
        bool operator<(const Version& other) const;
        bool operator<=(const Version& other) const;
        bool operator>(const Version& other) const;
        bool operator==(const Version& other) const;
        bool operator!=(const Version& other) const;
    };

    // Data structures
    struct PlyData {
        bool is_imported = false;
        std::filesystem::path ply_path;
        int ply_training_iter_number = 0;
        std::string ply_name;

        // Constructor for easy initialization
        PlyData() = default;
        PlyData(bool imported, const std::filesystem::path& path, int iter, const std::string& _ply_name)
            : is_imported(imported),
              ply_path(path),
              ply_training_iter_number(iter),
              ply_name(_ply_name) {}
    };

    struct DataSetInfo : public param::DatasetConfig {
        std::string data_type;
        DataSetInfo() = default;
        explicit DataSetInfo(const DatasetConfig& data_config);
    };

    struct OutputsInfo {
        std::vector<PlyData> plys;
    };

    // Main project data structure
    struct ProjectData {
        Version version;
        std::string project_name;
        std::string project_creation_time;
        std::string project_last_update_time;
        DataSetInfo data_set_info;
        OutputsInfo outputs;

        // optimization params
        param::OptimizationParameters optimization;
        // Additional fields for future versions can be added here
        nlohmann::json additional_fields; // For storing unknown fields during migration
    };

    // Migration interface for backward compatibility
    class ProjectMigrator {
    public:
        virtual ~ProjectMigrator() = default;
        virtual bool canMigrate(const Version& from, const Version& to) const = 0;
        virtual nlohmann::json migrate(const nlohmann::json& oldData, const Version& from, const Version& to) const = 0;
    };

    // Concrete migrator implementations
    class MigratorRegistry {
    private:
        std::vector<std::unique_ptr<ProjectMigrator>> migrators_;

    public:
        void registerMigrator(std::unique_ptr<ProjectMigrator> migrator);
        nlohmann::json migrateToVersion(const nlohmann::json& data, const Version& from, const Version& to) const;
    };

    // Main project file manager class
    // if you add additonal fields - add test add them in test_management too!
    class Project {
    private:
        ProjectData project_data_;
        MigratorRegistry migrator_registry_;

        // Internal helper methods
        void initializeMigrators();
        bool validateJsonStructure(const nlohmann::json& json) const;
        ProjectData parseProjectData(const nlohmann::json& json) const;
        nlohmann::ordered_json serializeProjectData(const ProjectData& data) const;

    public:
        static const Version CURRENT_VERSION;
        static const std::string FILE_HEADER;
        static const std::string EXTENSION;

        explicit Project(bool update_file_on_change = false);
        explicit Project(const ProjectData& initialData, bool update_file_on_change = false);

        void setProjectOutputFolder(const std::filesystem::path& path) { project_data_.data_set_info.output_path = path; }
        std::filesystem::path getProjectOutputFolder() const { return project_data_.data_set_info.output_path; }

        // project file name
        void setProjectFileName(const std::filesystem::path& path);
        std::filesystem::path getProjectFileName() const { return output_file_name_; }

        // Main interface methods
        bool readFromFile(const std::filesystem::path& filepath);
        // if the user gave a path - use path else use the one that was given in setOutputFileName
        bool writeToFile(const std::filesystem::path& filepath = {});

        // Data access methods
        const ProjectData& getProjectData() const { return project_data_; }
        ProjectData& getProjectData() { return project_data_; }

        void setProjectData(const ProjectData& data) { project_data_ = data; }
        void setOptimizationParams(const param::OptimizationParameters& opt) { project_data_.optimization = opt; }
        [[nodiscard]] param::OptimizationParameters getOptimizationParams() const { return project_data_.optimization; }

        // Convenience methods
        void setProjectName(const std::string& name);
        void setDataInfo(const param::DatasetConfig& data_config);
        void addPly(const PlyData& ply);
        void removePly(size_t index);
        [[nodiscard]] std::vector<PlyData> getPlys() const;

        // Version and compatibility methods
        Version getFileVersion() const { return project_data_.version; }
        static Version getCurrentVersion() { return CURRENT_VERSION; }
        bool isCompatible(const Version& fileVersion) const;

        // Utility methods
        std::string generateCurrentTimeStamp() const;
        bool validateProjectData() const;

    private:
        std::filesystem::path output_file_name_;
        bool update_file_on_change_ = false; // if true update file on every change
        mutable std::mutex io_mutex_;
    };

    std::shared_ptr<Project> CreateNewProject(const gs::param::DatasetConfig& data, const param::OptimizationParameters& opt,
                                              const std::string& project_name = "LichtFeldStudioProject");

    // find the
    std::filesystem::path FindProjectFile(const std::filesystem::path& directory);

} // namespace gs::management
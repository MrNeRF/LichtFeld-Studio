/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tcp_responder.hpp"

namespace lfs::tcp {

    ResponderServer::ResponderServer(int port, std::shared_ptr<lfs::vis::TrainerManager> trainer_manager_)
        : TCPServer(port, std::move(trainer_manager_), zmq::socket_type::rep),
          running_(false) {
        // Wake up every 200ms so recv doesn't hold the program
        socket_.set(zmq::sockopt::rcvtimeo, 200);
    }

    ResponderServer::~ResponderServer() {
        ResponderServer::stop();
    }

    void ResponderServer::start() {
        running_ = true;
        response_thread_ = std::thread(&ResponderServer::run, this);
    }

    void ResponderServer::stop() {
        running_ = false;
        if (response_thread_.joinable()) {
            response_thread_.join();
        }
    }

    void ResponderServer::join() {
        if (response_thread_.joinable()) {
            response_thread_.join();
        }
    }

    void ResponderServer::run() {
        nlohmann::json request;
        while (running_) {
            bool received = false;
            try {
                received = receive(request); // Returns false on timeout
            } catch (const std::exception& e) {
                // Malformed / unreadable message
                nlohmann::json error_response{{"success", false}, {"error", e.what()}};
                try {
                    send(error_response);
                } catch (...) {}
                continue;
            }

            if (!received) {
                continue;
            }

            nlohmann::json response;
            try {
                response = generateResponse(request);
            } catch (const std::exception& e) {
                response = {{"success", false}, {"error", e.what()}};
            }
            try {
                send(response);
            } catch (const std::exception&) {
                // Send failure leaves a REP socket unable to continue its
                // receive/send transaction, so stop the server explicitly.
                running_ = false;
            }
        }
    }

    nlohmann::json ResponderServer::generateResponse(const nlohmann::json& request) {
        if (!request.is_object()) {
            return {{"success", false}, {"error", "Request must be a JSON object"}};
        }
        auto command = request.value("command", "");
        nlohmann::json response;
        response["command"] = command;

        if (command == "get") {
            auto parameter = request.value("parameter", "");
            bool success = false;
            response["parameter"] = parameter;
            response["value"] = getValue(parameter, success);
            response["success"] = success;
        } else if (command == "start") {
            response["success"] = trainer_manager_->startTraining();
        } else if (command == "pause") {
            trainer_manager_->pauseTraining();
            response["success"] = trainer_manager_->isPaused();
        } else if (command == "resume") {
            trainer_manager_->resumeTraining();
            response["success"] = !trainer_manager_->isPaused();
        } else if (command == "stop") {
            trainer_manager_->stopTraining();
            response["success"] = true;
        } else if (command == "save_checkpoint") {
            trainer_manager_->requestSaveCheckpoint();
            response["success"] = true;
        } else {
            response["success"] = false;
        }
        return response;
    }

    nlohmann::json ResponderServer::getValue(std::string_view parameter, bool& success) {
        success = true;
        if (parameter == "state") {
            return std::string(lfs::vis::TrainingStateMachine::stateName(trainer_manager_->getState()));
        }
        if (parameter == "is_running") {
            return trainer_manager_->isRunning();
        }
        if (parameter == "is_paused") {
            return trainer_manager_->isPaused();
        }
        if (parameter == "is_finished") {
            return trainer_manager_->isFinished();
        }
        if (parameter == "is_training_active") {
            return trainer_manager_->isTrainingActive();
        }
        if (parameter == "can_start") {
            return trainer_manager_->canStart();
        }
        if (parameter == "can_pause") {
            return trainer_manager_->canPause();
        }
        if (parameter == "can_resume") {
            return trainer_manager_->canResume();
        }
        if (parameter == "can_stop") {
            return trainer_manager_->canStop();
        }
        if (parameter == "can_reset") {
            return trainer_manager_->canReset();
        }
        if (parameter == "current_iteration") {
            return trainer_manager_->getCurrentIteration();
        }
        if (parameter == "current_loss") {
            return trainer_manager_->getCurrentLoss();
        }
        if (parameter == "total_iterations") {
            return trainer_manager_->getTotalIterations();
        }
        if (parameter == "num_splats") {
            return trainer_manager_->getNumSplats();
        }
        if (parameter == "max_gaussians") {
            return trainer_manager_->getMaxGaussians();
        }
        if (parameter == "strategy_type") {
            return std::string(trainer_manager_->getStrategyType());
        }
        if (parameter == "is_gut_enabled") {
            return trainer_manager_->isGutEnabled();
        }
        if (parameter == "elapsed_seconds") {
            return trainer_manager_->getElapsedSeconds();
        }
        if (parameter == "estimated_remaining_seconds") {
            return trainer_manager_->getEstimatedRemainingSeconds();
        }
        if (parameter == "last_error") {
            return std::string(trainer_manager_->getLastError());
        }
        success = false;
        return "";
    }
} // namespace lfs::tcp

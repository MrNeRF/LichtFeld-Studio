/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tcp_responder.hpp"

#include "core/error.hpp"
#include "core/error_envelope.hpp"
#include "core/error_reporter.hpp"
#include "core/guarded_task.hpp"

namespace lfs::tcp {

    namespace {

        nlohmann::json failure_response(const lfs::Error& error) {
            nlohmann::json envelope = lfs::core::to_wire_envelope(error);
            std::string message = envelope.value("message", std::string{});
            return nlohmann::json{
                {"success", false},
                {"error", std::move(envelope)},
                {"error_message", std::move(message)}};
        }

    } // namespace

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
        response_thread_ = std::thread([this] {
            lfs::core::run_guarded<void>(
                lfs::core::TaskContext{
                    .name = "tcp.responder-thread",
                    .domain = lfs::ErrorDomain::TCP,
                    .operation_id = lfs::OperationId::generate(),
                    .site = LFS_SOURCE_SITE_CURRENT(),
                },
                [this]() -> lfs::Result<void> {
                    run();
                    return {};
                },
                [](lfs::Result<void>&& result) {
                    if (!result) {
                        lfs::core::ErrorReporter::get().report(result.error(), lfs::core::ReportChannel::OwnerLog);
                    }
                });
        });
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
        // Placeholder seed: lfs::Error has no public default constructor, and
        // receive() overwrites this only on the MalformedJson/Transport
        // branches — the only ones that read it back.
        lfs::Error receive_error = lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::TCP,
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
        while (running_) {
            switch (receive(request, &receive_error)) {
            case TcpReceiveStatus::Timeout:
                continue;
            case TcpReceiveStatus::Transport:
                lfs::core::ErrorReporter::get().report(receive_error, lfs::core::ReportChannel::OwnerLog);
                continue;
            case TcpReceiveStatus::MalformedJson:
                if (const lfs::Status sent = send(failure_response(receive_error)); !sent) {
                    lfs::core::ErrorReporter::get().report(sent.error(), lfs::core::ReportChannel::OwnerLog);
                    running_ = false;
                }
                continue;
            case TcpReceiveStatus::Message:
                break;
            }

            nlohmann::json response;
            try {
                response = generateResponse(request);
            } catch (...) {
                const lfs::Error error = lfs::core::detail::normalize_current_exception(lfs::core::TaskContext{
                    .name = "tcp.request",
                    .domain = lfs::ErrorDomain::TCP,
                    .operation_id = lfs::OperationId::generate(),
                    .site = LFS_SOURCE_SITE_CURRENT(),
                });
                lfs::core::ErrorReporter::get().report(error, lfs::core::ReportChannel::OwnerLog);
                response = failure_response(error);
            }
            if (const lfs::Status sent = send(response); !sent) {
                // A REP socket cannot continue its receive/send transaction
                // after a failed send, so report and stop the server.
                lfs::core::ErrorReporter::get().report(sent.error(), lfs::core::ReportChannel::OwnerLog);
                running_ = false;
            }
        }
    }

    nlohmann::json ResponderServer::generateResponse(const nlohmann::json& request) {
        if (!request.is_object()) {
            return failure_response(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::TCP,
                .user_message = "Request must be a JSON object",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        auto command = request.value("command", "");
        nlohmann::json response;
        response["command"] = command;

        if (command == "get") {
            auto parameter = request.value("parameter", "");
            response["parameter"] = parameter;
            bool success = false;
            nlohmann::json value = getValue(parameter, success);
            if (success) {
                response["value"] = std::move(value);
                response["success"] = true;
            } else {
                lfs::Error error = lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::NotFound,
                    .domain = lfs::ErrorDomain::TCP,
                    .user_message = "Unknown parameter: " + parameter,
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                    .fields = lfs::SmallFields{}.add("parameter", parameter),
                });
                response["error"] = lfs::core::to_wire_envelope(error);
                response["error_message"] = "Unknown parameter: " + parameter;
                response["value"] = "";
                response["success"] = false;
            }
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
            lfs::Error error = lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::NotFound,
                .domain = lfs::ErrorDomain::TCP,
                .user_message = "Unknown command: " + command,
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = lfs::SmallFields{}.add("command", command),
            });
            response["error"] = lfs::core::to_wire_envelope(error);
            response["error_message"] = "Unknown command: " + command;
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
        if (parameter == "last_training_error") {
            if (auto latched = trainer_manager_->lastTrainingError()) {
                return lfs::core::to_wire_envelope(*latched);
            }
            return nullptr;
        }
        success = false;
        return "";
    }
} // namespace lfs::tcp

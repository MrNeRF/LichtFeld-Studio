/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/editor/python_lsp_client.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

    class ScopedEnvVar {
    public:
        ScopedEnvVar(const char* key, std::string value)
            : key_(key) {
            if (const char* current = std::getenv(key_)) {
                had_previous_ = true;
                previous_ = current;
            }

#ifdef _WIN32
            _putenv_s(key_, value.c_str());
#else
            setenv(key_, value.c_str(), 1);
#endif
        }

        ~ScopedEnvVar() {
#ifdef _WIN32
            if (had_previous_) {
                _putenv_s(key_, previous_.c_str());
            } else {
                _putenv_s(key_, "");
            }
#else
            if (had_previous_) {
                setenv(key_, previous_.c_str(), 1);
            } else {
                unsetenv(key_);
            }
#endif
        }

    private:
        const char* key_;
        bool had_previous_ = false;
        std::string previous_;
    };

#ifndef _WIN32
    std::filesystem::path write_mock_lsp_server() {
        const auto temp_dir =
            std::filesystem::temp_directory_path() / "lichtfeld_python_lsp_test";
        std::filesystem::create_directories(temp_dir);

        const auto script = temp_dir / "mock_python_lsp.py";
        std::ofstream file(script);
        file << R"PY(#!/usr/bin/env python3
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, value = line.decode("utf-8").split(":", 1)
        headers[name.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    payload = sys.stdin.buffer.read(length)
    return json.loads(payload.decode("utf-8"))


def send_message(payload):
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    if message is None:
        break

    method = message.get("method")
    if method == "initialize":
        send_message(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "capabilities": {
                        "completionProvider": {"triggerCharacters": ["."]},
                        "semanticTokensProvider": {
                            "legend": {
                                "tokenTypes": ["namespace", "function", "property"],
                                "tokenModifiers": ["defaultLibrary"],
                            },
                            "full": True,
                        },
                    }
                },
            }
        )
    elif method == "textDocument/completion":
        send_message(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "isIncomplete": False,
                    "items": [
                        {
                            "label": "get_scene",
                            "detail": "mock detail",
                            "sortText": "001",
                            "textEdit": {
                                "range": {
                                    "start": {"line": 0, "character": 3},
                                    "end": {"line": 0, "character": 6},
                                },
                                "newText": "get_scene()",
                            },
                            "additionalTextEdits": [
                                {
                                    "range": {
                                        "start": {"line": 0, "character": 0},
                                        "end": {"line": 0, "character": 0},
                                    },
                                    "newText": "import lichtfeld as lf\n",
                                }
                            ],
                        }
                    ],
                },
            }
        )
    elif method == "textDocument/semanticTokens/full":
        send_message(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "data": [
                        0, 7, 9, 0, 0,
                        1, 11, 9, 1, 0,
                        1, 14, 19, 2, 0,
                    ]
                },
            }
        )
)PY";
        file.close();

        chmod(script.c_str(), 0755);
        return script;
    }
#endif

} // namespace

TEST(PythonLspClientTest, ReceivesCompletionItemsFromLanguageServer) {
#ifdef _WIN32
    GTEST_SKIP() << "Mock LSP server fixture is only implemented on POSIX.";
#else
    const auto script = write_mock_lsp_server();
    ScopedEnvVar server_override("LFS_PYTHON_LSP", script.string());
    ScopedEnvVar workspace_override("LFS_PYTHON_LSP_WORKSPACE",
                                    (script.parent_path() / "workspace").string());

    lfs::vis::editor::PythonLspClient client;
    const int version = client.updateDocument("lf.get");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!client.isReady() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(client.isReady());

    client.requestCompletion(version, 0, 6, false);

    std::optional<lfs::vis::editor::PythonLspClient::CompletionList> completion;
    while (std::chrono::steady_clock::now() < deadline) {
        completion = client.takeLatestCompletion();
        if (completion.has_value()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ASSERT_TRUE(completion.has_value());
    ASSERT_EQ(completion->document_version, version);
    ASSERT_EQ(completion->items.size(), 1u);
    EXPECT_EQ(completion->items[0].label, "get_scene");
    ASSERT_TRUE(completion->items[0].text_edit.has_value());
    EXPECT_EQ(completion->items[0].text_edit->new_text, "get_scene()");
    ASSERT_EQ(completion->items[0].additional_text_edits.size(), 1u);
    EXPECT_EQ(completion->items[0].additional_text_edits[0].new_text,
              "import lichtfeld as lf\n");
#endif
}

TEST(PythonLspClientTest, ReceivesSemanticTokensFromLanguageServer) {
#ifdef _WIN32
    GTEST_SKIP() << "Mock LSP server fixture is only implemented on POSIX.";
#else
    const auto script = write_mock_lsp_server();
    ScopedEnvVar server_override("LFS_PYTHON_LSP", script.string());
    ScopedEnvVar workspace_override("LFS_PYTHON_LSP_WORKSPACE",
                                    (script.parent_path() / "workspace").string());

    lfs::vis::editor::PythonLspClient client;
    const int version = client.updateDocument(
        "import lichtfeld as lf\nscene = lf.get_scene()\ncount = scene.active_camera_count\n");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!client.isReady() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(client.isReady());

    client.requestSemanticTokens(version);

    std::optional<lfs::vis::editor::PythonLspClient::SemanticTokenList> semantic_tokens;
    while (std::chrono::steady_clock::now() < deadline) {
        semantic_tokens = client.takeLatestSemanticTokens();
        if (semantic_tokens.has_value()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ASSERT_TRUE(semantic_tokens.has_value());
    ASSERT_EQ(semantic_tokens->document_version, version);
    ASSERT_EQ(semantic_tokens->tokens.size(), 3u);
    EXPECT_EQ(semantic_tokens->tokens[0].type, "namespace");
    EXPECT_EQ(semantic_tokens->tokens[1].type, "function");
    EXPECT_EQ(semantic_tokens->tokens[2].type, "property");
#endif
}

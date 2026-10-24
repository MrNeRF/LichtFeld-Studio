/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_serialization.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
using namespace lfs::core;
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: bench_collision_queries <native volume.json>\n";
        return 1;
    }
    nlohmann::json j;
    std::ifstream(argv[1]) >> j;
    const auto v = j.get<CollisionVolume>();
    std::mt19937 random(55319);
    std::uniform_real_distribution<float> xz(-5, 5), y(-1, 4);
    std::vector<glm::vec3> cases;
    for (int i = 0; i < 1000; ++i)
        cases.push_back({xz(random), y(random), xz(random)});
    std::vector<double> times;
    size_t hits = 0;
    for (int round = -3; round < 30; ++round) {
        auto start = std::chrono::steady_clock::now();
        for (auto p : cases)
            hits += v.capsuleIntersects(p, {0, 1, 0}, 1.8f, .2f);
        auto dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (round >= 0)
            times.push_back(dt);
    }
    std::sort(times.begin(), times.end());
    std::cout << "cases=" << cases.size() << " rounds=30 hits_with_warmup=" << hits << " median_batch_ms=" << times[15] << " median_query_us=" << times[15] * 1000 / cases.size() << " p90_batch_ms=" << times[27] << std::endl;
}

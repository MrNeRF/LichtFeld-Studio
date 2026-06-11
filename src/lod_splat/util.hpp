/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace lfs::lod {

    inline double nowMs() {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    struct ScopedTimer {
        double t0;
        double& out;
        explicit ScopedTimer(double& accum) : t0(nowMs()), out(accum) {}
        ~ScopedTimer() { out += nowMs() - t0; }
    };

    inline void parallelFor(size_t begin, size_t end, unsigned threads,
                            const std::function<void(size_t, size_t)>& fn) {
        const size_t n = end - begin;
        if (threads <= 1 || n < 4096) {
            fn(begin, end);
            return;
        }
        std::vector<std::thread> pool;
        const size_t chunk = (n + threads - 1) / threads;
        for (unsigned t = 0; t < threads; ++t) {
            const size_t b = begin + t * chunk;
            if (b >= end)
                break;
            const size_t e = std::min(end, b + chunk);
            pool.emplace_back(fn, b, e);
        }
        for (auto& th : pool)
            th.join();
    }

    // LSD radix sort of (key, value) pairs, 8-bit digits.
    template <typename Key>
    void radixSortPairs(std::vector<Key>& keys, std::vector<uint32_t>& vals) {
        const size_t n = keys.size();
        std::vector<Key> kbuf(n);
        std::vector<uint32_t> vbuf(n);
        Key* kin = keys.data();
        Key* kout = kbuf.data();
        uint32_t* vin = vals.data();
        uint32_t* vout = vbuf.data();
        constexpr int passes = (int)sizeof(Key);
        size_t hist[256];
        for (int p = 0; p < passes; ++p) {
            const int shift = p * 8;
            // skip pass if all keys share this byte
            for (size_t& h : hist)
                h = 0;
            for (size_t i = 0; i < n; ++i)
                ++hist[(kin[i] >> shift) & 0xFF];
            bool trivial = false;
            for (int b = 0; b < 256; ++b)
                if (hist[b] == n) {
                    trivial = true;
                    break;
                }
            if (trivial)
                continue;
            size_t sum = 0;
            for (int b = 0; b < 256; ++b) {
                const size_t c = hist[b];
                hist[b] = sum;
                sum += c;
            }
            for (size_t i = 0; i < n; ++i) {
                const size_t dst = hist[(kin[i] >> shift) & 0xFF]++;
                kout[dst] = kin[i];
                vout[dst] = vin[i];
            }
            std::swap(kin, kout);
            std::swap(vin, vout);
        }
        if (kin != keys.data()) {
            std::copy(kin, kin + n, keys.data());
            std::copy(vin, vin + n, vals.data());
        }
    }

} // namespace lfs::lod

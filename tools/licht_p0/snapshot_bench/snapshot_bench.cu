// Training-snapshot benchmark prototype (P0 packet c).
// Standalone CUDA runtime + std only. WRITE/COMPILE-CHECK artifact; orchestrator runs GPU.

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n  call: %s\n",        \
                         __FILE__, __LINE__, cudaGetErrorString(_err), #call);   \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

namespace {

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;
using Sec   = std::chrono::duration<double>;

constexpr size_t kMiB = 1024ull * 1024ull;
constexpr size_t kGiB = 1024ull * kMiB;

// One uint64 stamp word at the head of every device buffer; mutations skip it.
constexpr size_t kStampWords = 1;
constexpr size_t kStampBytes = kStampWords * sizeof(uint64_t);

// Pinned→pageable drain copy strategy.
// nt  = AVX non-temporal streaming stores (default; avoids LLC pollution)
// std = libc memcpy
enum class DrainCopy { Nt, Std };

struct Config {
    size_t gib              = 10;
    int    cycles           = 20;
    size_t band_mib         = 128;
    int    bands            = 3;
    int    disk_throttle_mbps = 0; // 0 = unthrottled
    std::string json_path;
    bool   quick            = false;
    std::string disk_path   = "/dev/null";
    int    post_resume_steps = 100;
    int    baseline_steps    = 50;
    size_t buffer_mib        = 256; // per device tensor
    int    drain_threads    = 1;    // pinned→pageable H2H drain workers
    DrainCopy drain_copy    = DrainCopy::Nt;
    bool   stress_overlap   = false; // if true, overlap prev writer with next pause
};

const char* drain_copy_name(DrainCopy m) {
    return m == DrainCopy::Nt ? "nt" : "std";
}

struct Percentiles {
    double p50 = 0;
    double p95 = 0;
    double max = 0;
    double mean = 0;
};

struct GateResult {
    const char* name;
    bool        pass;
    std::string detail;
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --gib N                 Total device tensor size in GiB (default 10)\n"
        "  --cycles N              Snapshot cycles (default 20)\n"
        "  --band-mib N            Pinned ring band size in MiB (default 128)\n"
        "  --bands N               Pinned ring band count (default 3)\n"
        "  --disk-throttle-mbps N  Rate-limit background disk writes (0=none)\n"
        "  --json PATH             Write machine-readable results JSON\n"
        "  --disk PATH             Background write target (default /dev/null)\n"
        "  --drain-threads N       Parallel pinned→pageable drain workers (default 1)\n"
        "  --drain-copy nt|std     Drain memcpy path: nt=AVX streaming stores (default),\n"
        "                          std=libc memcpy\n"
        "  --stress-overlap        Overlap prev writer with next pause (violates design;\n"
        "                          reference measurement only — not gated)\n"
        "  --quick                 Smoke: 1 GiB, 3 cycles\n"
        "  -h, --help              This help\n",
        argv0);
}

Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--gib") {
            c.gib = static_cast<size_t>(std::stoull(need("--gib")));
        } else if (a == "--cycles") {
            c.cycles = std::stoi(need("--cycles"));
        } else if (a == "--band-mib") {
            c.band_mib = static_cast<size_t>(std::stoull(need("--band-mib")));
        } else if (a == "--bands") {
            c.bands = std::stoi(need("--bands"));
        } else if (a == "--disk-throttle-mbps") {
            c.disk_throttle_mbps = std::stoi(need("--disk-throttle-mbps"));
        } else if (a == "--json") {
            c.json_path = need("--json");
        } else if (a == "--disk") {
            c.disk_path = need("--disk");
        } else if (a == "--drain-threads") {
            c.drain_threads = std::stoi(need("--drain-threads"));
        } else if (a == "--drain-copy") {
            const std::string v = need("--drain-copy");
            if (v == "nt") {
                c.drain_copy = DrainCopy::Nt;
            } else if (v == "std") {
                c.drain_copy = DrainCopy::Std;
            } else {
                std::fprintf(stderr, "invalid --drain-copy=%s (want nt|std)\n", v.c_str());
                std::exit(2);
            }
        } else if (a == "--stress-overlap") {
            c.stress_overlap = true;
        } else if (a == "--quick") {
            c.quick = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    if (c.quick) {
        c.gib    = 1;
        c.cycles = 3;
    }
    if (c.cycles < 1 || c.bands < 1 || c.band_mib == 0 || c.gib == 0) {
        std::fprintf(stderr, "invalid config: gib/cycles/bands/band-mib must be positive\n");
        std::exit(2);
    }
    if (c.drain_threads < 1) {
        std::fprintf(stderr, "invalid config: --drain-threads must be >= 1\n");
        std::exit(2);
    }
    return c;
}

// ---------------------------------------------------------------------------
// Host utilities
// ---------------------------------------------------------------------------

Percentiles compute_percentiles(std::vector<double> v) {
    Percentiles p;
    if (v.empty()) return p;
    std::sort(v.begin(), v.end());
    p.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    p.max  = v.back();
    auto at = [&](double q) {
        const double idx = q * static_cast<double>(v.size() - 1);
        const size_t lo  = static_cast<size_t>(idx);
        const size_t hi  = std::min(lo + 1, v.size() - 1);
        const double t   = idx - static_cast<double>(lo);
        return v[lo] * (1.0 - t) + v[hi] * t;
    };
    p.p50 = at(0.50);
    p.p95 = at(0.95);
    return p;
}

// Linux RSS in bytes; returns 0 if unavailable.
size_t read_rss_bytes() {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            // VmRSS:   12345 kB
            size_t kb = 0;
            if (std::sscanf(line.c_str(), "VmRSS: %zu", &kb) == 1) {
                return kb * 1024ull;
            }
        }
    }
    return 0;
}

// Non-temporal pinned→pageable copy: 32-byte AVX streaming stores to pageable so
// staging (write-once, read-later by CRC/serialization) does not thrash the LLC
// against the training kernel's host paths. Scalar head/tail for unaligned edges;
// _mm_sfence after the band body so NT stores are globally visible before reuse.
// target("avx2") lets the default nvcc host line (no -mavx2) still emit the
// always_inline AVX intrinsics without a global -march change.
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void nt_memcpy(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    auto* d = static_cast<uint8_t*>(dst);
    const auto* s = static_cast<const uint8_t*>(src);

    // Scalar head until dst is 32-byte aligned (required by _mm256_stream_si256).
    while (bytes > 0 && (reinterpret_cast<uintptr_t>(d) & 31u) != 0) {
        *d++ = *s++;
        --bytes;
    }

    const size_t nvec = bytes >> 5; // 32-byte vectors
    auto* dv = reinterpret_cast<__m256i*>(d);
    const auto* sv = reinterpret_cast<const __m256i*>(s);
    if ((reinterpret_cast<uintptr_t>(s) & 31u) == 0) {
        for (size_t i = 0; i < nvec; ++i) {
            const __m256i v = _mm256_load_si256(sv + i);
            _mm256_stream_si256(dv + i, v);
        }
    } else {
        for (size_t i = 0; i < nvec; ++i) {
            const __m256i v = _mm256_loadu_si256(sv + i);
            _mm256_stream_si256(dv + i, v);
        }
    }
    d += nvec << 5;
    s += nvec << 5;
    bytes -= nvec << 5;

    // Scalar tail.
    while (bytes > 0) {
        *d++ = *s++;
        --bytes;
    }
    _mm_sfence();
}
#else
void nt_memcpy(void* dst, const void* src, size_t bytes) {
    // Non-x86_64: fall back to libc (host path only; CUDA targets here are x86_64).
    std::memcpy(dst, src, bytes);
}
#endif

void drain_memcpy(void* dst, const void* src, size_t bytes, DrainCopy mode) {
    if (mode == DrainCopy::Nt) {
        nt_memcpy(dst, src, bytes);
    } else {
        std::memcpy(dst, src, bytes);
    }
}

// Parallel pinned→pageable (or any host) drain. n_threads==1 is a single-threaded call.
// Used to keep the ring from being H2H-bound below PCIe D2H throughput.
void parallel_memcpy(void* dst, const void* src, size_t bytes, int n_threads,
                     DrainCopy mode) {
    if (bytes == 0) return;
    if (n_threads <= 1 || bytes < 1 * kMiB) {
        drain_memcpy(dst, src, bytes, mode);
        return;
    }
    const int n = n_threads;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n));
    // Round chunk size up to 32 bytes so NT workers start on aligned dst when possible.
    size_t chunk = (bytes + static_cast<size_t>(n) - 1) / static_cast<size_t>(n);
    chunk = (chunk + 31u) & ~size_t{31};
    for (int t = 0; t < n; ++t) {
        const size_t off = static_cast<size_t>(t) * chunk;
        if (off >= bytes) break;
        const size_t len = std::min(chunk, bytes - off);
        workers.emplace_back([=] {
            drain_memcpy(static_cast<char*>(dst) + off,
                         static_cast<const char*>(src) + off, len, mode);
        });
    }
    for (auto& w : workers) w.join();
}

// Prefault pageable host staging: malloc + touch every page once.
void* alloc_prefaulted(size_t bytes) {
    void* p = std::malloc(bytes);
    if (!p) {
        std::fprintf(stderr, "malloc(%zu) failed for host staging\n", bytes);
        std::exit(1);
    }
    volatile char* v = static_cast<volatile char*>(p);
    for (size_t off = 0; off < bytes; off += 4096) {
        v[off] = 0;
    }
    if (bytes > 0) {
        v[bytes - 1] = 0;
    }
    return p;
}

// CRC32 (ISO polynomial 0xEDB88320), reflected.
uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) {
            const uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

uint32_t crc32_buffer(const void* data, size_t len) {
    return crc32_update(0, static_cast<const uint8_t*>(data), len);
}

// Rate-limited chunked write. throttle_mbps == 0 → no limit.
// Returns wall ms spent in this function.
double write_throttled(const void* data, size_t bytes, const std::string& path,
                       int throttle_mbps) {
    const auto t0 = Clock::now();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "failed to open disk target: %s\n", path.c_str());
        std::exit(1);
    }
    constexpr size_t kChunk = 4 * kMiB;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t remaining = bytes;
    size_t written   = 0;
    const auto start = Clock::now();

    while (remaining > 0) {
        const size_t n = std::min(remaining, kChunk);
        out.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
        if (!out) {
            std::fprintf(stderr, "disk write failed at offset %zu\n", written);
            std::exit(1);
        }
        p += n;
        remaining -= n;
        written += n;

        if (throttle_mbps > 0) {
            const double allowed_bytes =
                Sec(Clock::now() - start).count() * static_cast<double>(throttle_mbps) * 1e6;
            if (static_cast<double>(written) > allowed_bytes) {
                const double need_s =
                    (static_cast<double>(written) /
                     (static_cast<double>(throttle_mbps) * 1e6)) -
                    Sec(Clock::now() - start).count();
                if (need_s > 0) {
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(need_s));
                }
            }
        }
    }
    out.flush();
    return Ms(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Optimizer-step stand-in: mutate payload after stamp word so stamps stay intact
// until an explicit stamp kernel runs.
__global__ void optimizer_step_kernel(float* data, size_t n_floats, float delta) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // Skip first sizeof(uint64_t)/sizeof(float) elements reserved for the stamp.
    const size_t payload_start = (kStampBytes + sizeof(float) - 1) / sizeof(float);
    if (i < n_floats && i >= payload_start) {
        data[i] += delta;
    }
}

__global__ void stamp_kernel(uint64_t* header, uint64_t snapshot_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        header[0] = snapshot_id;
    }
}

// ---------------------------------------------------------------------------
// Device buffers + streams
// ---------------------------------------------------------------------------

struct DeviceBuffer {
    void*  ptr  = nullptr;
    size_t bytes = 0;
};

struct BenchState {
    Config cfg;
    size_t total_bytes   = 0;
    size_t band_bytes    = 0;
    size_t pinned_total  = 0;
    size_t n_buffers     = 0;

    std::vector<DeviceBuffer> device_bufs;
    cudaStream_t opt_stream  = nullptr;
    cudaStream_t d2h_stream  = nullptr;

    // Pinned ring
    std::vector<void*>          pinned;   // bands
    std::vector<cudaEvent_t>    band_done;

    // Prefaulted pageable staging (full snapshot)
    void*  host_staging = nullptr;
    size_t host_staging_bytes = 0;

    // Baseline pinned buffer for efficiency reference (also counts toward peak pinned
    // only during the baseline measurement; freed before the cycle loop).
    void*  baseline_pinned = nullptr;
    size_t baseline_pinned_bytes = 0;
    double baseline_d2h_gib_s = 0;

    size_t peak_pinned_bytes = 0;
    size_t rss_before_staging = 0;
    size_t rss_after_staging  = 0;
};

void free_bench(BenchState& s) {
    for (auto& b : s.device_bufs) {
        if (b.ptr) CHECK_CUDA(cudaFree(b.ptr));
        b.ptr = nullptr;
    }
    s.device_bufs.clear();
    for (void* p : s.pinned) {
        if (p) CHECK_CUDA(cudaFreeHost(p));
    }
    s.pinned.clear();
    for (cudaEvent_t e : s.band_done) {
        if (e) CHECK_CUDA(cudaEventDestroy(e));
    }
    s.band_done.clear();
    if (s.baseline_pinned) {
        CHECK_CUDA(cudaFreeHost(s.baseline_pinned));
        s.baseline_pinned = nullptr;
    }
    if (s.opt_stream)  CHECK_CUDA(cudaStreamDestroy(s.opt_stream));
    if (s.d2h_stream)  CHECK_CUDA(cudaStreamDestroy(s.d2h_stream));
    s.opt_stream = s.d2h_stream = nullptr;
    if (s.host_staging) {
        std::free(s.host_staging);
        s.host_staging = nullptr;
    }
}

void init_bench(BenchState& s) {
    const Config& c = s.cfg;
    s.total_bytes = c.gib * kGiB;
    s.band_bytes  = c.band_mib * kMiB;
    s.pinned_total = static_cast<size_t>(c.bands) * s.band_bytes;

    if (s.pinned_total > 512 * kMiB) {
        std::fprintf(stderr,
            "warning: pinned ring is %zu MiB (> 512 MiB gate ceiling)\n",
            s.pinned_total / kMiB);
    }

    const size_t buf_bytes = c.buffer_mib * kMiB;
    s.n_buffers = (s.total_bytes + buf_bytes - 1) / buf_bytes;
    // Adjust so sum == total_bytes exactly (last buffer may be smaller).
    s.device_bufs.resize(s.n_buffers);
    size_t remaining = s.total_bytes;
    for (size_t i = 0; i < s.n_buffers; ++i) {
        const size_t n = std::min(remaining, buf_bytes);
        s.device_bufs[i].bytes = n;
        CHECK_CUDA(cudaMalloc(&s.device_bufs[i].ptr, n));
        CHECK_CUDA(cudaMemset(s.device_bufs[i].ptr, 0, n));
        remaining -= n;
    }

    CHECK_CUDA(cudaStreamCreateWithFlags(&s.opt_stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&s.d2h_stream, cudaStreamNonBlocking));

    s.pinned.resize(static_cast<size_t>(c.bands), nullptr);
    s.band_done.resize(static_cast<size_t>(c.bands), nullptr);
    for (int i = 0; i < c.bands; ++i) {
        CHECK_CUDA(cudaMallocHost(&s.pinned[static_cast<size_t>(i)], s.band_bytes));
        CHECK_CUDA(cudaEventCreateWithFlags(&s.band_done[static_cast<size_t>(i)],
                                            cudaEventDisableTiming));
        // Prefault pinned pages.
        std::memset(s.pinned[static_cast<size_t>(i)], 0, s.band_bytes);
    }
    s.peak_pinned_bytes = s.pinned_total;

    s.rss_before_staging = read_rss_bytes();
    s.host_staging_bytes = s.total_bytes;
    s.host_staging = alloc_prefaulted(s.host_staging_bytes);
    s.rss_after_staging = read_rss_bytes();
}

// Pinned-direct baseline: one large D2H into a ≤512 MiB pinned buffer (freed before cycles).
// Peak pinned for the gate is the ring only; baseline pin is temporary and not counted.
void measure_pinned_baseline(BenchState& s) {
    const size_t xfer = std::min(s.total_bytes, 512 * kMiB);
    s.baseline_pinned_bytes = xfer;
    CHECK_CUDA(cudaMallocHost(&s.baseline_pinned, xfer));
    std::memset(s.baseline_pinned, 0, xfer);

    void* src = nullptr;
    bool  own = false;
    if (s.device_bufs[0].bytes >= xfer) {
        src = s.device_bufs[0].ptr;
    } else {
        CHECK_CUDA(cudaMalloc(&src, xfer));
        own = true;
        CHECK_CUDA(cudaMemset(src, 0, xfer));
    }

    for (int w = 0; w < 3; ++w) {
        CHECK_CUDA(cudaMemcpy(s.baseline_pinned, src, xfer, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));
    constexpr int kIters = 10;
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < kIters; ++i) {
        CHECK_CUDA(cudaMemcpy(s.baseline_pinned, src, xfer, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));

    const double gib = static_cast<double>(xfer) * kIters / static_cast<double>(kGiB);
    const double sec = static_cast<double>(ms) / 1000.0;
    s.baseline_d2h_gib_s = gib / sec;

    if (own) CHECK_CUDA(cudaFree(src));
    CHECK_CUDA(cudaFreeHost(s.baseline_pinned));
    s.baseline_pinned = nullptr;
    s.peak_pinned_bytes = s.pinned_total;
}

void launch_optimizer_step(BenchState& s, float delta) {
    constexpr int kThreads = 256;
    for (auto& b : s.device_bufs) {
        const size_t n_floats = b.bytes / sizeof(float);
        if (n_floats == 0) continue;
        const int blocks = static_cast<int>((n_floats + kThreads - 1) / kThreads);
        optimizer_step_kernel<<<blocks, kThreads, 0, s.opt_stream>>>(
            static_cast<float*>(b.ptr), n_floats, delta);
        CHECK_CUDA(cudaGetLastError());
    }
}

void stamp_all(BenchState& s, uint64_t snapshot_id) {
    for (auto& b : s.device_bufs) {
        if (b.bytes < kStampBytes) continue;
        stamp_kernel<<<1, 1, 0, s.opt_stream>>>(
            static_cast<uint64_t*>(b.ptr), snapshot_id);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(s.opt_stream));
}

// Mean step time in ms over n_steps (includes launch + stream sync per step).
double measure_step_time_ms(BenchState& s, int n_steps, float delta) {
    // Warm one step
    launch_optimizer_step(s, delta);
    CHECK_CUDA(cudaStreamSynchronize(s.opt_stream));

    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventRecord(e0, s.opt_stream));
    for (int i = 0; i < n_steps; ++i) {
        launch_optimizer_step(s, delta);
    }
    CHECK_CUDA(cudaEventRecord(e1, s.opt_stream));
    CHECK_CUDA(cudaEventSynchronize(e1));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    return static_cast<double>(ms) / static_cast<double>(n_steps);
}

// Banded D2H through pinned ring into host_staging.
// Returns {pause_ms, d2h_gib_s} where pause is safe-point-enter → last D2H event done.
// safe_point_enter_already: caller clocks pause_start before calling if it needs
// stream-sync inside; this function starts pause_clock at entry (before syncs).
struct SnapshotTiming {
    double pause_ms   = 0;
    double d2h_gib_s  = 0;
    double d2h_ms     = 0;
};

SnapshotTiming run_snapshot_d2h(BenchState& s, uint64_t snapshot_id) {
    SnapshotTiming t;
    const auto pause_start = Clock::now();
    const int drain_threads = s.cfg.drain_threads;
    const DrainCopy drain_copy = s.cfg.drain_copy;

    // (a) optimizer-enters-safe-point: sync all streams that mutate persisted state.
    CHECK_CUDA(cudaStreamSynchronize(s.opt_stream));
    // d2h_stream should be idle between cycles; sync for completeness.
    CHECK_CUDA(cudaStreamSynchronize(s.d2h_stream));

    // (b) stamp one snapshot id over all tensors.
    stamp_all(s, snapshot_id);

    // (c) banded D2H through pinned ring → pageable staging (H2H after each band event).
    // Linearize all device bytes as a flat address sequence.
    const size_t n_bands = static_cast<size_t>(s.cfg.bands);
    std::vector<size_t> band_host_off(n_bands, 0);
    std::vector<size_t> band_valid   (n_bands, 0);
    std::vector<bool>   band_inflight(n_bands, false);

    size_t global_off = 0;
    size_t buf_i = 0;
    size_t buf_off = 0;
    size_t band_seq = 0; // absolute band issue index

    auto issue_one = [&](size_t slot, size_t host_off, void* dev_ptr, size_t n) {
        CHECK_CUDA(cudaMemcpyAsync(s.pinned[slot], dev_ptr, n,
                                   cudaMemcpyDeviceToHost, s.d2h_stream));
        CHECK_CUDA(cudaEventRecord(s.band_done[slot], s.d2h_stream));
        band_host_off[slot] = host_off;
        band_valid[slot]    = n;
        band_inflight[slot] = true;
    };

    auto retire_slot = [&](size_t slot) {
        if (!band_inflight[slot]) return;
        CHECK_CUDA(cudaEventSynchronize(s.band_done[slot]));
        parallel_memcpy(static_cast<char*>(s.host_staging) + band_host_off[slot],
                        s.pinned[slot], band_valid[slot], drain_threads, drain_copy);
        band_inflight[slot] = false;
    };

    const auto d2h_start = Clock::now();

    while (global_off < s.total_bytes) {
        const size_t slot = band_seq % n_bands;
        // Wait for this ring slot (and drain prior content into staging).
        retire_slot(slot);

        // One D2H per ring slot: min(band_bytes, remaining total, remaining in current buffer).
        while (buf_i < s.device_bufs.size() &&
               buf_off >= s.device_bufs[buf_i].bytes) {
            ++buf_i;
            buf_off = 0;
        }
        if (buf_i >= s.device_bufs.size()) break;

        const size_t piece = std::min({
            s.band_bytes,
            s.total_bytes - global_off,
            s.device_bufs[buf_i].bytes - buf_off,
        });
        void* dev = static_cast<char*>(s.device_bufs[buf_i].ptr) + buf_off;
        issue_one(slot, global_off, dev, piece);
        buf_off    += piece;
        global_off += piece;
        ++band_seq;
    }

    // Sync every outstanding D2H event; pause ends when the last one completes.
    for (size_t k = 0; k < n_bands; ++k) {
        if (!band_inflight[k]) continue;
        CHECK_CUDA(cudaEventSynchronize(s.band_done[k]));
    }
    t.pause_ms = Ms(Clock::now() - pause_start).count();

    // Final H2H into staging may run after optimizer-may-mutate (caller resumes training
    // after this function returns pause_ms). Staging is complete before CRC/serialization.
    // Drain remaining bands; with multiple outstanding slots, run H2H in parallel across
    // slots as well as within each band via drain_threads.
    {
        std::vector<std::thread> final_drain;
        for (size_t k = 0; k < n_bands; ++k) {
            if (!band_inflight[k]) continue;
            const size_t slot = k;
            final_drain.emplace_back([&, slot] {
                parallel_memcpy(static_cast<char*>(s.host_staging) + band_host_off[slot],
                                s.pinned[slot], band_valid[slot], drain_threads, drain_copy);
            });
            band_inflight[slot] = false;
        }
        for (auto& w : final_drain) w.join();
    }

    const double d2h_ms = Ms(Clock::now() - d2h_start).count();
    t.d2h_ms = d2h_ms;
    const double gib = static_cast<double>(s.total_bytes) / static_cast<double>(kGiB);
    t.d2h_gib_s = gib / (d2h_ms / 1000.0);
    return t;
}

// Verify every host buffer carries the same snapshot stamp.
bool verify_consistency(const BenchState& s, uint64_t expected_id,
                        std::string& err) {
    size_t off = 0;
    for (size_t i = 0; i < s.device_bufs.size(); ++i) {
        const size_t n = s.device_bufs[i].bytes;
        if (n < kStampBytes) {
            off += n;
            continue;
        }
        uint64_t stamp = 0;
        std::memcpy(&stamp, static_cast<const char*>(s.host_staging) + off, sizeof(stamp));
        if (stamp != expected_id) {
            err = "buffer " + std::to_string(i) + " stamp " + std::to_string(stamp) +
                  " != expected " + std::to_string(expected_id);
            return false;
        }
        off += n;
    }
    return true;
}

// Confirm device stamps were mutated after resume while host staging still has old stamp.
bool verify_no_leak(BenchState& s, uint64_t host_id, uint64_t device_id_after,
                    std::string& err) {
    // Host staging must still show host_id.
    if (!verify_consistency(s, host_id, err)) {
        err = "leak into staging: " + err;
        return false;
    }
    // Device should now carry device_id_after (we re-stamp after mutations for a clear proof).
    std::vector<uint64_t> dev_stamps(s.device_bufs.size(), 0);
    for (size_t i = 0; i < s.device_bufs.size(); ++i) {
        if (s.device_bufs[i].bytes < kStampBytes) continue;
        CHECK_CUDA(cudaMemcpy(&dev_stamps[i], s.device_bufs[i].ptr, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        if (dev_stamps[i] != device_id_after) {
            err = "device buffer " + std::to_string(i) + " stamp " +
                  std::to_string(dev_stamps[i]) + " != post-resume " +
                  std::to_string(device_id_after);
            return false;
        }
    }
    return true;
}

// Background serialization: CRC32 over staging + throttled disk write.
struct SerialResult {
    uint32_t crc = 0;
    double   wall_ms = 0;
};

SerialResult run_serialization(const void* staging, size_t bytes,
                               const std::string& disk_path, int throttle_mbps) {
    SerialResult r;
    const auto t0 = Clock::now();
    r.crc = crc32_buffer(staging, bytes);
    write_throttled(staging, bytes, disk_path, throttle_mbps);
    r.wall_ms = Ms(Clock::now() - t0).count();
    return r;
}

// One in-flight writer job (production: one snapshot + one writer; newer requests coalesce).
struct WriterJob {
    std::thread  th;
    SerialResult result{};
    bool         running = false;

    void join() {
        if (!running) return;
        if (th.joinable()) th.join();
        running = false;
    }

    void launch(const void* staging, size_t bytes, const std::string& path,
                int throttle_mbps) {
        // Caller must ensure any previous job is joined and staging is exclusive.
        if (running) {
            std::fprintf(stderr, "WriterJob::launch while still running\n");
            std::exit(1);
        }
        result = {};
        running = true;
        th = std::thread([this, staging, bytes, path, throttle_mbps] {
            result = run_serialization(staging, bytes, path, throttle_mbps);
        });
    }
};

// ---------------------------------------------------------------------------
// One full cycle: snapshot + resume training concurrent with serialization
// ---------------------------------------------------------------------------

struct CycleResult {
    double pause_ms = 0;
    double d2h_gib_s = 0;
    double post_step_ms = 0;
    bool   consistency_ok = false;
    bool   no_leak_ok = false;
    std::string err;
    uint32_t crc = 0;
};

// Contract mode (default): caller joins `writer` *before* calling (outside pause clock).
// Stress-overlap: previous writer may still run during the pause; we join after pause
// before launching the new job (staging exclusivity for CRC/disk).
CycleResult run_cycle(BenchState& s, uint64_t snapshot_id, int throttle_mbps,
                      bool measure_post_steps, WriterJob& writer, bool stress_overlap) {
    CycleResult cr;

    // Pre-snapshot training activity so stamps aren't already set.
    launch_optimizer_step(s, 0.001f);
    CHECK_CUDA(cudaStreamSynchronize(s.opt_stream));

    SnapshotTiming st = run_snapshot_d2h(s, snapshot_id);
    cr.pause_ms  = st.pause_ms;
    cr.d2h_gib_s = st.d2h_gib_s;

    // Stress-overlap only: reclaim staging after the (intentionally polluted) pause.
    if (stress_overlap) {
        writer.join();
    }

    std::string verr;
    cr.consistency_ok = verify_consistency(s, snapshot_id, verr);
    if (!cr.consistency_ok) {
        cr.err = verr;
        return cr;
    }

    // (d) optimizer-may-mutate: resume training on opt_stream while (e) serializes.
    writer.launch(s.host_staging, s.host_staging_bytes, s.cfg.disk_path, throttle_mbps);

    // Mutations after resume — must not appear in staging.
    if (measure_post_steps) {
        cr.post_step_ms = measure_step_time_ms(s, s.cfg.post_resume_steps, 0.01f);
    } else {
        for (int i = 0; i < 10; ++i) {
            launch_optimizer_step(s, 0.01f);
        }
        CHECK_CUDA(cudaStreamSynchronize(s.opt_stream));
    }

    // Distinct post-resume stamp on device for leak proof.
    const uint64_t post_id = snapshot_id + 0x9e3779b97f4a7c15ull;
    stamp_all(s, post_id);

    // Leak check while writer may still be reading staging (read-only).
    cr.no_leak_ok = verify_no_leak(s, snapshot_id, post_id, verr);
    if (!cr.no_leak_ok) {
        cr.err = verr;
    }

    // Contract mode: join writer before returning so the next cycle's pre-join is a
    // no-op and CRC is available for logging. This wait is still outside any pause
    // clock (production equivalent: coalesce — do not start the next snapshot yet).
    // Stress-overlap: leave the writer running so it contends with the next pause.
    if (!stress_overlap) {
        writer.join();
        cr.crc = writer.result.crc;
    }
    return cr;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

void print_table(const std::string& title,
                 const std::vector<std::pair<std::string, std::string>>& rows) {
    std::cout << "\n=== " << title << " ===\n";
    size_t w = 0;
    for (const auto& r : rows) w = std::max(w, r.first.size());
    for (const auto& r : rows) {
        std::cout << "  " << std::left << std::setw(static_cast<int>(w) + 2)
                  << r.first << r.second << "\n";
    }
}

std::string fmt_d(double v, int prec = 3) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

void write_json(const std::string& path,
                const Config& cfg,
                const std::vector<GateResult>& gates,
                double cold_pause_ms,
                const Percentiles& pause,
                double d2h_eff_pct,
                double banded_d2h_gib_s,
                double baseline_d2h_gib_s,
                size_t peak_pinned,
                size_t staging_bytes,
                size_t rss_delta,
                double baseline_step_ms,
                double post_step_ms,
                double step_reg_pct,
                double disk_throttle_delta_ms,
                bool all_consistency_ok,
                bool overall_pass) {
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "failed to write JSON: %s\n", path.c_str());
        std::exit(1);
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"gib\": " << cfg.gib << ",\n";
    out << "    \"cycles\": " << cfg.cycles << ",\n";
    out << "    \"band_mib\": " << cfg.band_mib << ",\n";
    out << "    \"bands\": " << cfg.bands << ",\n";
    out << "    \"disk_throttle_mbps\": " << cfg.disk_throttle_mbps << ",\n";
    out << "    \"drain_threads\": " << cfg.drain_threads << ",\n";
    out << "    \"drain_copy\": \"" << drain_copy_name(cfg.drain_copy) << "\",\n";
    out << "    \"stress_overlap\": " << (cfg.stress_overlap ? "true" : "false") << ",\n";
    out << "    \"quick\": " << (cfg.quick ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"metrics\": {\n";
    out << "    \"cold_pause_ms\": " << cold_pause_ms << ",\n";
    out << "    \"pause_ms\": {\"p50\": " << pause.p50 << ", \"p95\": " << pause.p95
        << ", \"max\": " << pause.max << ", \"mean\": " << pause.mean << "},\n";
    out << "    \"d2h_efficiency_pct\": " << d2h_eff_pct << ",\n";
    out << "    \"banded_d2h_gib_s\": " << banded_d2h_gib_s << ",\n";
    out << "    \"baseline_d2h_gib_s\": " << baseline_d2h_gib_s << ",\n";
    out << "    \"pinned_bytes_peak\": " << peak_pinned << ",\n";
    out << "    \"staging_bytes\": " << staging_bytes << ",\n";
    out << "    \"rss_delta_bytes\": " << rss_delta << ",\n";
    out << "    \"baseline_step_ms\": " << baseline_step_ms << ",\n";
    out << "    \"post_resume_step_ms\": " << post_step_ms << ",\n";
    out << "    \"step_time_regression_pct\": " << step_reg_pct << ",\n";
    out << "    \"disk_throttle_delta_ms\": " << disk_throttle_delta_ms << ",\n";
    out << "    \"consistency_all_ok\": " << (all_consistency_ok ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"gates\": {\n";
    for (size_t i = 0; i < gates.size(); ++i) {
        out << "    \"" << gates[i].name << "\": {\"pass\": "
            << (gates[i].pass ? "true" : "false") << ", \"detail\": \"";
        for (char ch : gates[i].detail) {
            if (ch == '"' || ch == '\\') out << '\\';
            out << ch;
        }
        out << "\"}" << (i + 1 < gates.size() ? "," : "") << "\n";
    }
    out << "  },\n";
    out << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << "\n";
    out << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cout << "snapshot_bench | device=" << prop.name
              << " | gib=" << cfg.gib
              << " | cycles=" << cfg.cycles
              << " | bands=" << cfg.bands << "x" << cfg.band_mib << " MiB"
              << " | disk_throttle_mbps=" << cfg.disk_throttle_mbps
              << " | drain_threads=" << cfg.drain_threads
              << " | drain_copy=" << drain_copy_name(cfg.drain_copy)
              << (cfg.stress_overlap ? " | stress-overlap" : " | contract-mode")
              << (cfg.quick ? " | quick" : "")
              << "\n";

    BenchState state;
    state.cfg = cfg;
    init_bench(state);
    WriterJob writer;

    const size_t rss_delta =
        (state.rss_after_staging >= state.rss_before_staging)
            ? (state.rss_after_staging - state.rss_before_staging)
            : 0;

    std::cout << "device tensors: " << state.n_buffers << " buffers, "
              << (state.total_bytes / kMiB) << " MiB total\n";
    std::cout << "pinned ring: " << (state.pinned_total / kMiB) << " MiB; "
              << "host staging: " << (state.host_staging_bytes / kMiB) << " MiB (prefaulted)\n";
    std::cout << "RSS delta after staging alloc: " << (rss_delta / kMiB) << " MiB\n";

    measure_pinned_baseline(state);
    std::cout << "pinned-direct baseline D2H: " << fmt_d(state.baseline_d2h_gib_s)
              << " GiB/s\n";

    // Baseline optimizer step time (no snapshot activity).
    const double baseline_step_ms =
        measure_step_time_ms(state, cfg.baseline_steps, 0.001f);
    std::cout << "baseline step time: " << fmt_d(baseline_step_ms, 4) << " ms\n";

    // Primary pass: configured throttle (default unthrottled).
    std::vector<double> pause_all;
    std::vector<double> d2h_rates;
    std::vector<double> post_steps;
    pause_all.reserve(static_cast<size_t>(cfg.cycles));
    bool all_consistency = true;
    double cold_pause_ms = 0;

    for (int cyc = 0; cyc < cfg.cycles; ++cyc) {
        // Design contract (default): one writer in flight; join previous before the
        // next snapshot begins. This wait is NOT part of the pause clock — in
        // production it is request coalescing. --stress-overlap skips this join so
        // the previous CRC+disk job contends with the next pause (reference only).
        if (!cfg.stress_overlap) {
            writer.join();
        }

        const uint64_t snap_id = 0x1000ull + static_cast<uint64_t>(cyc);
        CycleResult cr = run_cycle(state, snap_id, cfg.disk_throttle_mbps,
                                   /*measure_post_steps=*/true, writer,
                                   cfg.stress_overlap);
        if (cyc == 0) cold_pause_ms = cr.pause_ms;
        pause_all.push_back(cr.pause_ms);
        d2h_rates.push_back(cr.d2h_gib_s);
        post_steps.push_back(cr.post_step_ms);
        if (!cr.consistency_ok || !cr.no_leak_ok) {
            all_consistency = false;
            std::fprintf(stderr, "cycle %d consistency failure: %s\n", cyc,
                         cr.err.c_str());
        }
        // Under stress-overlap, CRC may still be in flight; join only for display
        // would defeat the overlap — print 0 until a later reclaim join.
        if (cfg.stress_overlap && cr.crc == 0 && !writer.running) {
            cr.crc = writer.result.crc;
        }
        std::cout << "  cycle " << std::setw(3) << cyc
                  << "  pause_ms=" << fmt_d(cr.pause_ms)
                  << "  d2h_GiB/s=" << fmt_d(cr.d2h_gib_s)
                  << "  post_step_ms=" << fmt_d(cr.post_step_ms, 4)
                  << "  crc=0x" << std::hex << cr.crc << std::dec
                  << (cr.consistency_ok && cr.no_leak_ok ? "  OK" : "  FAIL")
                  << "\n";
    }
    // Drain any writer left in flight (stress-overlap last cycle).
    writer.join();

    const Percentiles pause_pct = compute_percentiles(pause_all);
    // Max gate includes cold cycle (already in pause_all[0]).
    const double mean_banded_d2h =
        std::accumulate(d2h_rates.begin(), d2h_rates.end(), 0.0) /
        static_cast<double>(d2h_rates.size());
    const double d2h_eff_pct =
        (state.baseline_d2h_gib_s > 0)
            ? (100.0 * mean_banded_d2h / state.baseline_d2h_gib_s)
            : 0.0;

    const double mean_post_step =
        std::accumulate(post_steps.begin(), post_steps.end(), 0.0) /
        static_cast<double>(post_steps.size());
    const double step_reg_pct =
        (baseline_step_ms > 0)
            ? (100.0 * (mean_post_step - baseline_step_ms) / baseline_step_ms)
            : 0.0;

    // Disk throttle delta: compare pause under unthrottled vs 500 MB/s.
    // Uses a short comparison pass (same cycle count, or min(cycles, 5) for --quick).
    const int cmp_cycles = cfg.quick ? cfg.cycles : std::min(cfg.cycles, 10);
    auto mean_pause_at = [&](int throttle) {
        std::vector<double> ps;
        ps.reserve(static_cast<size_t>(cmp_cycles));
        for (int cyc = 0; cyc < cmp_cycles; ++cyc) {
            if (!cfg.stress_overlap) {
                writer.join();
            }
            const uint64_t snap_id =
                0x8000ull + static_cast<uint64_t>(throttle) * 1000ull +
                static_cast<uint64_t>(cyc);
            CycleResult cr = run_cycle(state, snap_id, throttle,
                                       /*measure_post_steps=*/false, writer,
                                       cfg.stress_overlap);
            ps.push_back(cr.pause_ms);
            if (!cr.consistency_ok || !cr.no_leak_ok) {
                all_consistency = false;
                std::fprintf(stderr, "throttle-cmp cycle fail: %s\n", cr.err.c_str());
            }
        }
        writer.join();
        return std::accumulate(ps.begin(), ps.end(), 0.0) /
               static_cast<double>(ps.size());
    };

    std::cout << "\nThrottle comparison (" << cmp_cycles
              << " cycles each, unthrottled vs 500 MB/s)...\n";
    const double pause_unthrottled = mean_pause_at(0);
    const double pause_500         = mean_pause_at(500);
    const double disk_throttle_delta_ms = std::abs(pause_500 - pause_unthrottled);
    std::cout << "  mean pause unthrottled: " << fmt_d(pause_unthrottled) << " ms\n";
    std::cout << "  mean pause 500 MB/s:    " << fmt_d(pause_500) << " ms\n";
    std::cout << "  |delta|:                " << fmt_d(disk_throttle_delta_ms) << " ms\n";

    // Gates
    const size_t pinned_gate = 512 * kMiB;
    // Plan: extra host RAM beyond staging — RSS delta ≤ snapshot + 768 MiB.
    const size_t rss_gate = state.host_staging_bytes + 768 * kMiB;
    const size_t extra_host =
        (rss_delta > state.host_staging_bytes)
            ? (rss_delta - state.host_staging_bytes)
            : 0;

    std::vector<GateResult> gates;
    gates.push_back({"pause_p95_ms",
                     pause_pct.p95 <= 750.0,
                     "p95=" + fmt_d(pause_pct.p95) + " (gate ≤ 750)"});
    gates.push_back({"pause_max_ms",
                     pause_pct.max <= 1000.0,
                     "max=" + fmt_d(pause_pct.max) + " incl cold=" +
                         fmt_d(cold_pause_ms) + " (gate ≤ 1000)"});
    gates.push_back({"d2h_efficiency_pct",
                     d2h_eff_pct >= 80.0,
                     fmt_d(d2h_eff_pct) + "% (gate ≥ 80%)"});
    gates.push_back({"pinned_bytes_peak",
                     state.peak_pinned_bytes <= pinned_gate,
                     std::to_string(state.peak_pinned_bytes) + " (gate ≤ " +
                         std::to_string(pinned_gate) + ")"});
    gates.push_back({"extra_host_ram",
                     rss_delta <= rss_gate,
                     "rss_delta=" + std::to_string(rss_delta) +
                         " extra_beyond_staging=" + std::to_string(extra_host) +
                         " staging=" + std::to_string(state.host_staging_bytes) +
                         " (gate rss_delta ≤ snapshot+768MiB)"});
    gates.push_back({"step_time_regression_pct",
                     step_reg_pct <= 10.0,
                     fmt_d(step_reg_pct) + "% (gate ≤ 10%)"});
    gates.push_back({"disk_throttle_delta_ms",
                     disk_throttle_delta_ms <= 100.0,
                     fmt_d(disk_throttle_delta_ms) + " (gate ≤ 100)"});
    gates.push_back({"consistency_proof",
                     all_consistency,
                     all_consistency ? "all cycles OK" : "one or more cycles failed"});

    bool overall = true;
    for (const auto& g : gates) overall = overall && g.pass;

    print_table("Results", {
        {"cold_pause_ms", fmt_d(cold_pause_ms)},
        {"pause_ms p50",  fmt_d(pause_pct.p50)},
        {"pause_ms p95",  fmt_d(pause_pct.p95)},
        {"pause_ms max",  fmt_d(pause_pct.max)},
        {"pause_ms mean", fmt_d(pause_pct.mean)},
        {"banded D2H GiB/s (mean)", fmt_d(mean_banded_d2h)},
        {"baseline D2H GiB/s", fmt_d(state.baseline_d2h_gib_s)},
        {"d2h_efficiency %", fmt_d(d2h_eff_pct)},
        {"pinned_bytes peak", std::to_string(state.peak_pinned_bytes)},
        {"staging_bytes", std::to_string(state.host_staging_bytes)},
        {"rss_delta_bytes", std::to_string(rss_delta)},
        {"extra_host_beyond_staging", std::to_string(extra_host)},
        {"baseline_step_ms", fmt_d(baseline_step_ms, 4)},
        {"post_resume_step_ms (mean)", fmt_d(mean_post_step, 4)},
        {"step_time_regression_pct", fmt_d(step_reg_pct)},
        {"disk_throttle_delta_ms", fmt_d(disk_throttle_delta_ms)},
    });

    std::cout << "\n=== Gates ===\n";
    for (const auto& g : gates) {
        std::cout << "  [" << (g.pass ? "PASS" : "FAIL") << "] " << g.name
                  << "  " << g.detail << "\n";
    }
    std::cout << "\nOverall: " << (overall ? "PASS" : "FAIL") << "\n";
    std::cout.flush();
    std::fflush(stdout);
    std::fflush(stderr);

    // Capture exit status before any teardown so a later failure cannot drop it.
    // Overall: FAIL must yield process exit code 1 (gates already computed above).
    const int exit_code = overall ? 0 : 1;

    if (!cfg.json_path.empty()) {
        write_json(cfg.json_path, cfg, gates, cold_pause_ms, pause_pct, d2h_eff_pct,
                   mean_banded_d2h, state.baseline_d2h_gib_s, state.peak_pinned_bytes,
                   state.host_staging_bytes, rss_delta, baseline_step_ms, mean_post_step,
                   step_reg_pct, disk_throttle_delta_ms, all_consistency, overall);
        std::cout << "Wrote " << cfg.json_path << "\n";
        std::cout.flush();
    }

    writer.join();
    free_bench(state);
    // Use std::exit so the status cannot be lost if a static destructor misbehaves.
    std::exit(exit_code);
}

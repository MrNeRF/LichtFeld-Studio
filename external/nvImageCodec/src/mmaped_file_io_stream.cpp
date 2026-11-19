// # SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// # SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <errno.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#if !defined(__AARCH64_QNX__) && !defined(__AARCH64_GNU__) && !defined(__aarch64__)
#include <linux/sysctl.h>
#include <sys/syscall.h>
#endif
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <algorithm>
#include <tuple>

#include "mmaped_file_io_stream.h"

static void *file_map(const char *path, size_t *length, bool read_ahead) {
  int fd = -1;
  struct stat s;
  void *p = nullptr;
  int flags = MAP_PRIVATE;
  if (read_ahead) {
#if !defined(__AARCH64_QNX__) && !defined(__AARCH64_GNU__) && !defined(__aarch64__)
    flags |= MAP_POPULATE;
#endif
  }

  if ((fd = open(path, O_RDONLY)) < 0) {
    goto fail;
  }

  if (fstat(fd, &s) < 0) {
    goto fail;
  }

  *length = (size_t)s.st_size;

  if ((p = mmap(nullptr, *length, PROT_READ, flags, fd, 0)) == MAP_FAILED) {
    p = nullptr;
    goto fail;
  }

fail:
  if (p == nullptr) {
    throw std::runtime_error("File mapping failed: " + std::string(path));
  }
  if (fd > -1) {
    close(fd);
  }
  return p;
}

namespace nvimgcodec {

using MappedFile = std::tuple<std::weak_ptr<void>, size_t>;


// Cache the already opened files: this avoids mapping the same file N times in memory.
// Doing so is wasteful since the file is read-only: we can share the underlying buffer,
// with different pos_.
std::mutex mapped_files_mutex;
std::map<std::string, MappedFile> mapped_files;

MmapedFileIoStream::MmapedFileIoStream(const std::string& path, bool read_ahead) :
  FileIoStream(path), length_(0), pos_(0), read_ahead_whole_file_(read_ahead) {
  std::lock_guard<std::mutex> lock(mapped_files_mutex);
  std::weak_ptr<void> mapped_memory;
  std::tie(mapped_memory, length_) = mapped_files[path];

  if (!(p_ = mapped_memory.lock())) {
    void *p = file_map(path.c_str(), &length_, read_ahead_whole_file_);
    size_t length_tmp = length_;
    p_ = std::shared_ptr<void>(p, [=](void*) {
      // we are not touching mapped_files, weak_ptr is enough to check if
      // memory is valid or not
      munmap(p, length_tmp);
     });
    mapped_files[path] = std::make_tuple(p_, length_);
  }

  path_ = path;

  if(p_ == nullptr)
    std::runtime_error("Could not open file " + path + ": " + std::strerror(errno));
}

MmapedFileIoStream::~MmapedFileIoStream() {
  close();
}


void MmapedFileIoStream::close() {
  // Not doing any munmap right now, since Buffer objects might still
  // reference the memory range of the mapping.
  // When last instance of p_ in cease to exist memory will be unmapped
  p_ = nullptr;
  length_ = 0;
  pos_ = 0;
}


inline uint8_t* readAheadHelper(std::shared_ptr<void> &p, size_t &pos,
                                 size_t &n_bytes, bool read_ahead) {
  auto tmp = static_cast<uint8_t*>(p.get()) + pos;
  // Ask OS to load memory content to RAM to avoid sluggish page fault during actual access to
  // mmaped memory
  if (read_ahead) {
#if !defined(__AARCH64_QNX__) && !defined(__AARCH64_GNU__) && !defined(__aarch64__)
    madvise(tmp, n_bytes, MADV_WILLNEED);
#endif
  }
  return tmp;
}

void MmapedFileIoStream::seek(size_t pos, int whence) {
  if (whence == SEEK_CUR)
    pos += pos_;
  else if (whence == SEEK_END)
    pos += length_;
  if(pos > (size_t)length_)
    throw std::runtime_error("Invalid seek");
  pos_ = pos;
}

size_t MmapedFileIoStream::tell() const {
  return pos_;
}

// This method saves a memcpy
std::shared_ptr<void> MmapedFileIoStream::get(size_t n_bytes) {
  if (pos_ + n_bytes > length_) {
    return nullptr;
  }
  auto tmp = p_;
  std::shared_ptr<void> p(readAheadHelper(p_, pos_, n_bytes, !read_ahead_whole_file_),
    [tmp](void*) {
    // This is an empty lambda, which is a custom deleter for
    // std::shared_ptr.
    // While instantiating shared_ptr, also lambda is instantiated,
    // making a copy of p_. This way, reference counter
    // of p_ is incremented. Therefore, for the duration
    // of life cycle of underlying memory in shared_ptr, file that was
    // mapped creating p_ won't be unmapped
    // It will be freed, when last shared_ptr is deleted.
  });
  pos_ += n_bytes;
  return p;
}

size_t MmapedFileIoStream::read(void *buffer, size_t n_bytes) {
  n_bytes = std::min(n_bytes, length_ - pos_);
  memcpy(buffer, readAheadHelper(p_, pos_, n_bytes, !read_ahead_whole_file_), n_bytes);
  pos_ += n_bytes;
  return n_bytes;
}

size_t MmapedFileIoStream::size() const {
  return length_;
}

void* MmapedFileIoStream::map(size_t offset, size_t size) const {
  assert(offset + size <= length_);
  return (void*)(reinterpret_cast<unsigned char*>(p_.get()) + offset);
}

}  // namespace dali

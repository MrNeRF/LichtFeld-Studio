#pragma once

#include <cstdint>

uint32_t murmur_hash(const void* key, int len, uint32_t seed);
uint64_t murmur_hash_64(const void* key, uint32_t len, uint64_t seed);

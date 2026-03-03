#pragma once

#include <cstdint>
#include <cstddef>

#define GOLD_RATIO_CONSTANT 0x9e3779b97f4a7c15ull

namespace sveht {

struct HashParams64 {
  uint64_t hash_factor;
  uint8_t shift_right; // number of bits to shift right
  uint64_t buckets_mask; // buckets-1 (power of two size)
};

}
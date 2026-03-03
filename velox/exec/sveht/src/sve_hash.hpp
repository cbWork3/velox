#pragma once

#include <arm_sve.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "hash.hpp"
#include "macro.hpp"

namespace sveht {

// struct alignas(64) KeyValue { // 对象地址必须是 64 的整数倍，代价：内存膨胀 4 倍，实际只用 16B，占 64B
//   uint64_t key;
//   uint64_t value;
//   // Total: 16 bytes
// };

struct KeyValue { // TODO 需要alignas(64)吗？
  uint64_t key; // TODO scalar2 this key is 8-byte normalizedKey computed by Velox
  char* value;
  // Total: 16 bytes
};

inline __attribute__((always_inline)) int build_single_key(
    uint64_t key,
    char* value,
    const HashParams64& p,
    KeyValue* table) {
  uint64_t hash = ((key * p.hash_factor) >> p.shift_right) & p.buckets_mask;
  uint64_t start = hash;
  while (true) {
    if (table[hash].key == UINT64_MAX) {
      table[hash].key = key;
      table[hash].value = value;
      debug_cout("Inserted key " << key << "at hash " << hash << std::endl);
      return 0;
    }
    if (table[hash].key == key) {
      return 1; // duplicate key
    }
    hash = (hash + 1) & p.buckets_mask;
    if (hash == start) {
      return 2; // table full
    }
  }
}

inline void build_scalar(
    const uint64_t* build_keys,
    char** build_values,
    size_t num_keys,
    const HashParams64& p,
    KeyValue* table) {
  for (size_t i = 0; i < num_keys; ++i) {
    int ret = build_single_key(build_keys[i], build_values[i], p, table);
    if (ret != 0) {
      if (ret == 1) {
        throw std::runtime_error(
            "Duplicate key " +
            std::to_string(static_cast<int64_t>(build_keys[i])) +
            " found in table");
      } else if (ret == 2) {
        throw std::runtime_error(
            "Table full, failed to insert key " +
            std::to_string(static_cast<int64_t>(build_keys[i])) +
            " into table");
      }
    }
  }
}

// inline __attribute__((always_inline)) int probe_single_key(
//     const uint64_t key,
//     const uint64_t payload,
//     const uint64_t hash,
//     const KeyValue* table,
//     const uint64_t buckets_mask,
//     uint64_t* output_keys,
//     uint64_t* output_values,
//     uint64_t* output_payloads,
//     int& output_index) {
//   uint64_t current_hash = hash;
//
//   while (true) {
//     const KeyValue& entry = table[current_hash];
//
//     if (entry.key == key) {
//       output_keys[output_index] = key;
//       output_values[output_index] = entry.value;
//       if (output_payloads != NULL) {
//         output_payloads[output_index] = payload;
//       }
//       output_index++;
//       return 0;
//     }
//     if (entry.key == UINT64_MAX) {
//       return 1; // Key not found
//     }
//
//     current_hash = (current_hash + 1) & buckets_mask;
//     if (current_hash == hash) {
//       return 2; // Prevent infinite loop if table is full
//     };
//   }
// }

// int probe_scalar(
//     const uint64_t* query_keys,
//     const uint64_t* query_payloads,
//     const KeyValue* table,
//     const HashParams64& param,
//     size_t num_keys,
//     uint64_t* output_keys,
//     uint64_t* output_values,
//     uint64_t* output_payloads) {
//   int output_count = 0;
//
//   for (size_t i = 0; i < num_keys; ++i) {
//     uint64_t key = query_keys[i];
//     uint64_t payload = query_payloads ? query_payloads[i] : 0;
//
//     // Compute hash for this key
//     uint64_t hash =
//         ((key * param.hash_factor) >> param.shift_right) & param.buckets_mask;
//
//     probe_single_key(
//         key,
//         payload,
//         hash,
//         table,
//         param.buckets_mask,
//         output_keys,
//         output_values,
//         output_payloads,
//         output_count);
//   }
//   return output_count;
// }

} // namespace sveht

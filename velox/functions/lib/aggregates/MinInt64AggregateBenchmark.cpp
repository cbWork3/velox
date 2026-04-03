/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Standalone micro-benchmark: naive scalar min(int64) hash-agg update vs
 * SVE path (same structure as MinInt64AggregateSVE.inc.h /
 * hashAggUpdateSVEWithCharForNormal).
 *
 * Build (example AArch64 + SVE):
 *   c++ -O3 -std=c++17 -march=armv8.2-a+sve MinInt64AggregateBenchmark.cpp -o min_int64_bench
 *
 * Or from a Velox build tree (target velox_min_int64_aggregate_benchmark):
 *   cmake --build . --target velox_min_int64_aggregate_benchmark
 *
 * Run:
 *   ./min_int64_bench [rows] [unique_groups] [iterations] [seed]
 *   ./min_int64_bench --all-scenarios [iterations] [seed]
 *   ./min_int64_bench --list-scenarios
 *
 * No Velox / Folly dependencies — only libc++ and (on AArch64) arm_sve.h.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__aarch64__)
#include <arm_sve.h>
#endif

namespace min_int64_aggregate_bench {

constexpr int32_t kNullByte = 0;
constexpr uint8_t kNullMask = 1;
constexpr int32_t kAccOffset = 8;
constexpr int64_t kInitialAcc = INT64_MAX;

template <typename T, typename U>
constexpr inline T roundUp(T value, U factor) {
  return (value + (factor - 1)) / factor * factor;
}

template <typename T>
inline bool isBitSet(const T* bits, uint64_t idx) {
  return bits[idx / (sizeof(bits[0]) * 8)] &
      (static_cast<T>(1) << (idx & ((sizeof(bits[0]) * 8) - 1)));
}

inline bool isBitNull(const uint64_t* bits, int32_t index) {
  return isBitSet(bits, index) == false;
}

// Row-major bit layout matching Velox/Sum SVE: 4 bytes per 32 rows, index
// arr8Index = (row_chunk_start / 8) for ldr of 32 row bits.
inline void setRowBit(std::vector<uint8_t>& bits, int32_t row, bool selected) {
  if (selected) {
    bits[row / 8] |= static_cast<uint8_t>(1u << (row % 8));
  }
}

inline bool rowBit(const std::vector<uint8_t>& bits, int32_t row) {
  return (bits[row / 8] >> (row % 8)) & 1;
}

struct GroupRow {
  uint8_t bytes[16]{};

  void reset() {
    std::memset(bytes, 0, sizeof(bytes));
    *reinterpret_cast<int64_t*>(bytes + kAccOffset) = kInitialAcc;
  }

  bool operator==(const GroupRow& o) const {
    return std::memcmp(bytes, o.bytes, sizeof(bytes)) == 0;
  }
};

// --- Naive reference (one row at a time) ---------------------------------

void naiveMinHashAgg(
    char** rowGroupPtr,
    const std::vector<uint8_t>& bitmap1,
    const std::vector<uint8_t>& bitmap2,
    const int64_t* values,
    int32_t begin,
    int32_t end,
    uint64_t* numNulls /* unused when we keep nulls cleared; kept for API parity */) {
  (void)numNulls;
  for (int32_t r = begin; r < end; ++r) {
    if (!rowBit(bitmap1, r)) {
      continue;
    }
    if (!rowBit(bitmap2, r)) {
      continue;
    }
    char* g = rowGroupPtr[r];
    int64_t* acc = reinterpret_cast<int64_t*>(g + kAccOffset);
    int64_t v = values[r];
    if (*acc > v) {
      *acc = v;
    }
  }
}

#if defined(__aarch64__)

// --- SVE kernel (logic aligned with MinInt64AggregateSVE.inc.h) ------------

struct SveMinKernel {
  int32_t nullByte_ = kNullByte;
  uint8_t nullMask_ = kNullMask;
  int32_t offset_ = kAccOffset;
  uint64_t numNulls_ = 0; // 0 => clearNullSVE is a no-op (bench focuses on min)

  int32_t getOffsetFromAgg() const {
    return offset_;
  }

  svbool_t getBitMask(
      uint8_t* nulls_,
      int32_t index,
      int mode,
      uint32_t* dic,
      int32_t length) {
    svbool_t pg;
    if (mode == 0) {
      pg = svptrue_b8();
      return pg;
    }
    if (mode == 1) {
      __asm__ __volatile__("ldr %0, [%1]"
                           : "=Upl"(pg)
                           : "r"(&(nulls_[index]))
                           : "memory");
      return pg;
    }
    if (mode == 2) {
      if (!isBitNull(reinterpret_cast<uint64_t*>(nulls_), 0)) {
        pg = svptrue_b8();
      } else {
        pg = svpfalse();
      }
      return pg;
    }
    if (mode == 3) {
      svuint32_t onc = svdup_u32(1);
      svuint32_t inv = svindex_u32(0, 1);
      svuint32_t pow = svlsl_m(svptrue_b32(), onc, inv);
      uint8_t tmpNulls[4] = {0};
      uint32_t* null32ptr = reinterpret_cast<uint32_t*>(nulls_);

      svuint32_t posv, idxbufv, bufv, offsetv;
      svbool_t nullvec, pg1;

      pg1 = svwhilelt_b32(index * 8, length);
      posv = svld1(pg1, dic + index * 8);
      idxbufv = svlsr_x(pg1, posv, 5);
      bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
      offsetv = svand_m(pg1, posv, 0b11111);
      bufv = svlsr_m(pg1, bufv, offsetv);
      bufv = svand_m(pg1, bufv, 0x1);
      nullvec = svcmpgt(pg1, bufv, 0);
      if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
        tmpNulls[0] = svaddv(nullvec, pow);
      } else {
        tmpNulls[0] = 0;
      }

      pg1 = svwhilelt_b32(index * 8 + 8, length);
      posv = svld1(pg1, dic + index * 8 + 8);
      idxbufv = svlsr_x(pg1, posv, 5);
      bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
      offsetv = svand_m(pg1, posv, 0b11111);
      bufv = svlsr_m(pg1, bufv, offsetv);
      bufv = svand_m(pg1, bufv, 0x1);
      nullvec = svcmpgt(pg1, bufv, 0);
      if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
        tmpNulls[1] = svaddv(nullvec, pow);
      } else {
        tmpNulls[1] = 0;
      }

      pg1 = svwhilelt_b32(index * 8 + 16, length);
      posv = svld1(pg1, dic + index * 8 + 16);
      idxbufv = svlsr_x(pg1, posv, 5);
      bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
      offsetv = svand_m(pg1, posv, 0b11111);
      bufv = svlsr_m(pg1, bufv, offsetv);
      bufv = svand_m(pg1, bufv, 0x1);
      nullvec = svcmpgt(pg1, bufv, 0);
      if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
        tmpNulls[2] = svaddv(nullvec, pow);
      } else {
        tmpNulls[2] = 0;
      }

      pg1 = svwhilelt_b32(index * 8 + 24, length);
      posv = svld1(pg1, dic + index * 8 + 24);
      idxbufv = svlsr_x(pg1, posv, 5);
      bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
      offsetv = svand_m(pg1, posv, 0b11111);
      bufv = svlsr_m(pg1, bufv, offsetv);
      bufv = svand_m(pg1, bufv, 0x1);
      nullvec = svcmpgt(pg1, bufv, 0);
      if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
        tmpNulls[3] = svaddv(nullvec, pow);
      } else {
        tmpNulls[3] = 0;
      }

      __asm__ __volatile__("ldr %0, [%1]"
                           : "=Upl"(pg)
                           : "r"(tmpNulls)
                           : "memory");
      return pg;
    }
    pg = svpfalse();
    return pg;
  }

  bool clearNullSVE(svuint64_t ptr, svbool_t pg) {
    if (numNulls_) {
      svint64_t group = svld1sb_gather_u64base_offset_s64(
          pg, ptr, nullByte_);
      svuint8_t group8 = svreinterpret_u8(group);

      svuint8_t tmp = svand_n_u8_z(pg, group8, nullMask_);
      svbool_t test = svcmpne_n_u8(svptrue_b8(), tmp, 0);
      if (svptest_any(svptrue_b8(), test)) {
        uint8_t negNull = ~nullMask_;

        svuint8_t adjust = svand_n_u8_m(test, group8, negNull);
        svst1b_scatter_u64base_offset_s64(
            pg, ptr, nullByte_, svreinterpret_s64(adjust));

        int num = svcntp_b8(test, test);
        numNulls_ -= num;
        return true;
      }
    }
    return false;
  }

  inline __attribute__((always_inline)) svbool_t
  getUinqMask(svbool_t pg, const svuint64_t val) {
    svuint64_t s1 = svext_u64(val, val, 1);
    svbool_t mask2 = svcmpeq(svwhilelt_b64(0, 3), val, s1);

    svuint64_t s2 = svext_u64(val, val, 2);
    svbool_t mask3 = svcmpeq(svwhilelt_b64(0, 2), val, s2);
    svbool_t mask12 = svorr_b_z(pg, mask2, mask3);

    svuint64_t s3 = svext_u64(val, val, 3);
    svbool_t mask4 = svcmpeq(svwhilelt_b64(0, 1), val, s3);

    svbool_t mask = svorr_b_z(pg, mask4, mask12);
    mask = svnot_b_z(pg, mask);

    return mask;
  }

  static inline void minAssignScalarAt(
      char* groupRow,
      int64_t incoming,
      int64_t offsetFromAgg) {
    int64_t* acc = reinterpret_cast<int64_t*>(groupRow + offsetFromAgg);
    if (*acc > incoming) {
      *acc = incoming;
    }
  }

  void hashAggUpdateSVEWithCharForNormal(
      char** result,
      uint64_t* bitmap1,
      uint64_t* bitmap2,
      int64_t* value,
      int32_t begin,
      int32_t end,
      int mode1,
      int /*mode2*/,
      uint32_t* dic) {
    uint8_t* bitmap1_8 = reinterpret_cast<uint8_t*>(bitmap1);
    uint8_t* bitmap2_8 = reinterpret_cast<uint8_t*>(bitmap2);

    int32_t firstWord =
        roundUp(begin, 32) == begin ? begin : roundUp(begin, 32) - 32;
    int32_t lastWord = roundUp(end, 32);
    svbool_t mask, mask1, mask2{};
    const int64_t off = getOffsetFromAgg();

    for (int32_t count = firstWord; count + 32 <= lastWord; count += 32) {
      int32_t arr8Index = count / 8;
      if (bitmap2_8 != nullptr) {
        mask2 = getBitMask(bitmap2_8, arr8Index, mode1, dic, end);
      }
      __asm__ __volatile__("ldr %0, [%1]"
                           : "=Upl"(mask1)
                           : "r"(&bitmap1_8[arr8Index])
                           : "memory");
      mask = svand_b_z(svptrue_b8(), mask1, mask2);
      mask = svand_b_z(svptrue_b8(), mask, svwhilelt_b8(count, end));
      if (!svptest_any(svptrue_b8(), mask)) {
        continue;
      }

      svbool_t mask00 = svunpklo(mask);
      svbool_t mask01 = svunpkhi(mask);
      if (svptest_any(svptrue_b16(), mask00)) {
        svbool_t mask10 = svunpklo(mask00);
        if (svptest_any(svptrue_b32(), mask10)) {
          svbool_t mask20 = svunpklo(mask10);
          svbool_t mask21 = svunpkhi(mask10);
          if (svptest_any(svptrue_b64(), mask20)) {
            svuint64_t ptr =
                svld1(mask20, reinterpret_cast<uint64_t*>(result + count));
            svbool_t m20 = getUinqMask(mask20, ptr);
            clearNullSVE(ptr, m20);
            uint8_t flag0[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]" : : "r"(&flag0[0]), "Upl"(mask20) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag0[i] != 0) {
                minAssignScalarAt(result[count + i], value[count + i], off);
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask21)) {
            svuint64_t ptr =
                svld1(mask21, reinterpret_cast<uint64_t*>(result + count + 4));
            svbool_t m21 = getUinqMask(mask21, ptr);
            clearNullSVE(ptr, m21);
            uint8_t flag1[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]" : : "r"(&flag1[0]), "Upl"(mask21) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag1[i] != 0) {
                minAssignScalarAt(
                    result[count + 4 + i], value[count + 4 + i], off);
              }
            }
          }
        }
        svbool_t mask11 = svunpkhi(mask00);
        if (svptest_any(svptrue_b32(), mask11)) {
          svbool_t mask22 = svunpklo(mask11);
          svbool_t mask23 = svunpkhi(mask11);
          if (svptest_any(svptrue_b64(), mask22)) {
            svuint64_t ptr =
                svld1(mask22, reinterpret_cast<uint64_t*>(result + count + 8));
            svbool_t m22 = getUinqMask(mask22, ptr);
            clearNullSVE(ptr, m22);
            uint8_t flag2[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]" : : "r"(&flag2[0]), "Upl"(mask22) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag2[i] != 0) {
                minAssignScalarAt(
                    result[count + 8 + i], value[count + 8 + i], off);
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask23)) {
            svuint64_t ptr =
                svld1(mask23, reinterpret_cast<uint64_t*>(result + count + 12));
            svbool_t m23 = getUinqMask(mask23, ptr);
            clearNullSVE(ptr, m23);
            uint8_t flag3[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]" : : "r"(&flag3[0]), "Upl"(mask23) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag3[i] != 0) {
                minAssignScalarAt(
                    result[count + 12 + i], value[count + 12 + i], off);
              }
            }
          }
        }
      }

      svbool_t mask12 = svunpklo(mask01);

      if (svptest_any(svptrue_b16(), mask01)) {
        svbool_t mask24 = svunpklo(mask12);
        svbool_t mask25 = svunpkhi(mask12);
        if (svptest_any(svptrue_b32(), mask12)) {
          if (svptest_any(svptrue_b64(), mask24)) {
            svuint64_t ptr =
                svld1(mask24, reinterpret_cast<uint64_t*>(result + count + 16));
            svbool_t m24 = getUinqMask(mask24, ptr);
            clearNullSVE(ptr, m24);
            uint8_t flag4[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]" : : "r"(&flag4[0]), "Upl"(mask24) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag4[i] != 0) {
                minAssignScalarAt(
                    result[count + 16 + i], value[count + 16 + i], off);
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask25)) {
            svuint64_t ptr =
                svld1(mask25, reinterpret_cast<uint64_t*>(result + count + 20));
            svbool_t m25 = getUinqMask(mask25, ptr);
            clearNullSVE(ptr, m25);
            uint8_t flag5[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]" : : "r"(&flag5[0]), "Upl"(mask25) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag5[i] != 0) {
                minAssignScalarAt(
                    result[count + 20 + i], value[count + 20 + i], off);
              }
            }
          }
        }
        svbool_t mask13 = svunpkhi(mask01);

        if (svptest_any(svptrue_b32(), mask13)) {
          svbool_t mask26 = svunpklo(mask13);
          svbool_t mask27 = svunpkhi(mask13);
          if (svptest_any(svptrue_b64(), mask26)) {
            svuint64_t ptr =
                svld1(mask26, reinterpret_cast<uint64_t*>(result + count + 24));
            svbool_t m26 = getUinqMask(mask26, ptr);
            clearNullSVE(ptr, m26);
            uint8_t flag6[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]" : : "r"(&flag6[0]), "Upl"(mask26) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag6[i] != 0) {
                minAssignScalarAt(
                    result[count + 24 + i], value[count + 24 + i], off);
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask27)) {
            svuint64_t ptr =
                svld1(mask27, reinterpret_cast<uint64_t*>(result + count + 28));
            svbool_t m27 = getUinqMask(mask27, ptr);
            clearNullSVE(ptr, m27);
            uint8_t flag7[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]" : : "r"(&flag7[0]), "Upl"(mask27) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag7[i] != 0) {
                minAssignScalarAt(
                    result[count + 28 + i], value[count + 28 + i], off);
              }
            }
          }
        }
      }
    }
  }
};

void sveMinHashAgg(
    SveMinKernel& kernel,
    char** rowGroupPtr,
    uint64_t* bitmap1_u64,
    uint64_t* bitmap2_u64,
    int64_t* values,
    int32_t begin,
    int32_t end) {
  kernel.hashAggUpdateSVEWithCharForNormal(
      rowGroupPtr,
      bitmap1_u64,
      bitmap2_u64,
      values,
      begin,
      end,
      /*mode1=*/1,
      /*mode2=*/0,
      /*dic=*/nullptr);
}

#endif // __aarch64__

struct BenchState {
  int32_t numRows = 0;
  int32_t numGroups = 0;
  std::vector<GroupRow> groupPool;
  std::vector<char*> rowGroupPtr;
  std::vector<int64_t> values;
  std::vector<uint8_t> bitmap1;
  std::vector<uint8_t> bitmap2;
  std::vector<uint64_t> bitmap1AsU64;
  std::vector<uint64_t> bitmap2AsU64;

  void init(int32_t rows, int32_t groups, uint32_t seed) {
    numRows = rows;
    numGroups = std::max(1, groups);
    groupPool.resize(numGroups);
    rowGroupPtr.resize(numRows);
    values.resize(numRows);
    // SVE loads 4 bytes at arr8Index = chunk_start/8; pad past logical row bits.
    const int32_t bmBytes = roundUp(rows, 32) / 8 + 4;
    bitmap1.assign(bmBytes, 0);
    bitmap2.assign(bmBytes, 0);
    bitmap1AsU64.resize((bmBytes + 7) / 8);
    bitmap2AsU64.resize((bmBytes + 7) / 8);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> gid(0, numGroups - 1);
    std::uniform_int_distribution<int64_t> val(-1LL << 40, 1LL << 40);

    for (int32_t r = 0; r < numRows; ++r) {
      rowGroupPtr[r] = reinterpret_cast<char*>(&groupPool[gid(rng)]);
      values[r] = val(rng);
      setRowBit(bitmap1, r, true);
      setRowBit(bitmap2, r, true);
    }

    std::memcpy(bitmap1AsU64.data(), bitmap1.data(), bmBytes);
    std::memcpy(bitmap2AsU64.data(), bitmap2.data(), bmBytes);
  }

  void resetGroups() {
    for (auto& g : groupPool) {
      g.reset();
    }
  }

  std::vector<GroupRow> snapshotGroups() const {
    return groupPool;
  }

  bool groupsEqual(const std::vector<GroupRow>& a, const std::vector<GroupRow>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (!(a[i] == b[i])) {
        return false;
      }
    }
    return true;
  }
};

double secondsSince(std::chrono::steady_clock::time_point t0) {
  using namespace std::chrono;
  return duration<double>(steady_clock::now() - t0).count();
}

struct ScenarioSpec {
  const char* label;
  int32_t rows;
  int32_t groups;
};

// Preset workloads: tiny chunks, L1/L2/L3-friendly sizes, high/low cardinality.
static const ScenarioSpec kScenarioTable[] = {
    {"1_chunk_32x4", 32, 4},
    {"2_chunks_64x8", 64, 8},
    {"small_1k_x_32", 1024, 32},
    {"small_4k_x_256", 4096, 256},
    {"small_16k_x_1k", 16384, 1024},
    {"med_64k_x_4k", 65536, 4096},
    {"med_256k_x_16k", 262144, 16384},
    {"large_1m_x_64k", 1048576, 65536},
    {"xlarge_4m_x_256k", 4194304, 262144},
    {"conflict_heavy_1m_x_64", 1048576, 64},
    {"conflict_heavy_256k_x_16", 262144, 16},
    {"high_card_256k_x_200k", 262144, 200000},
    {"skew_rows_99999_x_8k", 99999, 8192},
    {"odd_rows_100001_x_5k", 100001, 5000},
};

static constexpr size_t kScenarioCount =
    sizeof(kScenarioTable) / sizeof(kScenarioTable[0]);

void printUsage(const char* argv0) {
  std::fprintf(
      stderr,
      "Usage:\n"
      "  %s [rows] [unique_groups] [iterations] [seed]\n"
      "      Single scenario (defaults: 262144 rows, 4096 groups, 50 iters).\n"
      "  %s --all-scenarios [iterations] [seed]\n"
      "      Run correctness + timing for all preset (rows, groups) pairs.\n"
      "  %s --list-scenarios\n"
      "      Print preset scenario table only.\n"
      "\n"
      "Options:\n"
      "  rows / unique_groups   hash-agg size; groups <= rows typical.\n"
      "  iterations             timed repetitions per impl (default 50).\n"
      "  seed                   RNG seed (default 1).\n",
      argv0,
      argv0,
      argv0);
}

void printScenarioList() {
  std::printf("Preset scenarios (%zu total):\n", kScenarioCount);
  for (size_t i = 0; i < kScenarioCount; ++i) {
    const auto& s = kScenarioTable[i];
    std::printf(
        "  %2zu  %-28s  rows=%10d  groups=%10d  avg_rows/group ~%.1f\n",
        i,
        s.label,
        s.rows,
        s.groups,
        static_cast<double>(s.rows) / std::max(1, s.groups));
  }
}

// Returns 0 on success, 2 on naive vs SVE mismatch (AArch64), 1 on bad params.
int runOneBenchmark(
    int32_t numRows,
    int32_t numGroups,
    int iterations,
    uint32_t seed,
    const char* scenarioTag) {
  if (numRows <= 0 || numGroups <= 0 || iterations <= 0) {
    return 1;
  }

  BenchState state;
  state.init(numRows, numGroups, seed);

  uint64_t nullsDummy = 0;

  state.resetGroups();
  naiveMinHashAgg(
      state.rowGroupPtr.data(),
      state.bitmap1,
      state.bitmap2,
      state.values.data(),
      0,
      numRows,
      &nullsDummy);
  const std::vector<GroupRow> afterNaive = state.snapshotGroups();

#if defined(__aarch64__)
  state.resetGroups();
  SveMinKernel sveKernel;
  sveMinHashAgg(
      sveKernel,
      state.rowGroupPtr.data(),
      state.bitmap1AsU64.data(),
      state.bitmap2AsU64.data(),
      state.values.data(),
      0,
      numRows);
  if (!state.groupsEqual(afterNaive, state.groupPool)) {
    std::fprintf(
        stderr,
        "FAIL [%s]: naive and SVE group states differ (%d rows, %d groups).\n",
        scenarioTag != nullptr ? scenarioTag : "custom",
        numRows,
        numGroups);
    return 2;
  }
  if (scenarioTag != nullptr) {
    std::printf(
        "OK [%s] naive vs SVE match: rows=%d groups=%d\n",
        scenarioTag,
        numRows,
        numGroups);
  } else {
    std::printf(
        "OK: naive vs SVE outputs match (%d rows, %d groups).\n",
        numRows,
        numGroups);
  }
#else
  (void)afterNaive;
  if (scenarioTag != nullptr) {
    std::printf(
        "[%s] Skip SVE (not AArch64); naive ref only. rows=%d groups=%d\n",
        scenarioTag,
        numRows,
        numGroups);
  } else {
    std::printf("Skip SVE check (not AArch64). Naive reference ran alone.\n");
  }
#endif

  const int warmup = std::min(5, std::max(1, iterations / 10));
  for (int w = 0; w < warmup; ++w) {
    state.resetGroups();
    naiveMinHashAgg(
        state.rowGroupPtr.data(),
        state.bitmap1,
        state.bitmap2,
        state.values.data(),
        0,
        numRows,
        &nullsDummy);
  }

  double tNaive = 0;
  for (int i = 0; i < iterations; ++i) {
    state.resetGroups();
    auto t0 = std::chrono::steady_clock::now();
    naiveMinHashAgg(
        state.rowGroupPtr.data(),
        state.bitmap1,
        state.bitmap2,
        state.values.data(),
        0,
        numRows,
        &nullsDummy);
    tNaive += secondsSince(t0);
  }

#if defined(__aarch64__)
  for (int w = 0; w < warmup; ++w) {
    state.resetGroups();
    SveMinKernel k;
    sveMinHashAgg(
        k,
        state.rowGroupPtr.data(),
        state.bitmap1AsU64.data(),
        state.bitmap2AsU64.data(),
        state.values.data(),
        0,
        numRows);
  }

  double tSve = 0;
  for (int i = 0; i < iterations; ++i) {
    state.resetGroups();
    SveMinKernel k;
    auto t0 = std::chrono::steady_clock::now();
    sveMinHashAgg(
        k,
        state.rowGroupPtr.data(),
        state.bitmap1AsU64.data(),
        state.bitmap2AsU64.data(),
        state.values.data(),
        0,
        numRows);
    tSve += secondsSince(t0);
  }

  const double nsRowNaive = tNaive * 1e9 / (iterations * numRows);
  const double nsRowSve = tSve * 1e9 / (iterations * numRows);
  const double speedup = tSve > 0 ? tNaive / tSve : 0.0;

  if (scenarioTag != nullptr) {
    std::printf(
        "TIME [%s] iters=%d seed=%u | naive %8.4fs (%6.2f ns/row) | SVE %8.4fs "
        "(%6.2f ns/row) | speedup %.3fx\n",
        scenarioTag,
        iterations,
        seed,
        tNaive,
        nsRowNaive,
        tSve,
        nsRowSve,
        speedup);
  } else {
    std::printf(
        "Timing (%d iterations, %d rows, %d groups, seed=%u):\n"
        "  naive: %8.4f s total, %8.2f ns/row\n"
        "  SVE:   %8.4f s total, %8.2f ns/row\n"
        "  speedup (naive/SVE): %.3fx\n",
        iterations,
        numRows,
        numGroups,
        seed,
        tNaive,
        nsRowNaive,
        tSve,
        nsRowSve,
        speedup);
  }
#else
  if (scenarioTag != nullptr) {
    std::printf(
        "TIME [%s] iters=%d seed=%u | naive only %8.4fs (%6.2f ns/row)\n",
        scenarioTag,
        iterations,
        seed,
        tNaive,
        tNaive * 1e9 / (iterations * numRows));
  } else {
    std::printf(
        "Timing naive only (%d iterations, %d rows, %d groups, seed=%u):\n"
        "  naive: %8.4f s total, %8.2f ns/row\n",
        iterations,
        numRows,
        numGroups,
        seed,
        tNaive,
        tNaive * 1e9 / (iterations * numRows));
  }
#endif

  return 0;
}

int runAllScenarios(int iterations, uint32_t seed) {
  std::printf(
      "=== Running %zu preset scenarios (iterations=%d, seed=%u) ===\n\n",
      kScenarioCount,
      iterations,
      seed);
  int worstRc = 0;
  for (size_t i = 0; i < kScenarioCount; ++i) {
    const auto& s = kScenarioTable[i];
    std::printf("\n--- [%zu/%zu] %s ---\n", i + 1, kScenarioCount, s.label);
    const int rc =
        runOneBenchmark(s.rows, s.groups, iterations, seed, s.label);
    if (rc != 0) {
      worstRc = rc;
    }
  }
  std::printf(
      "\n=== Finished all scenarios (worst exit code %d) ===\n", worstRc);
  return worstRc;
}

} // namespace min_int64_aggregate_bench

int main(int argc, char** argv) {
  using namespace min_int64_aggregate_bench;

  if (argc > 1) {
    const std::string a1 = argv[1];
    if (a1 == "-h" || a1 == "--help") {
      printUsage(argv[0]);
      return 0;
    }
    if (a1 == "--list-scenarios") {
      printScenarioList();
      return 0;
    }
    if (a1 == "--all-scenarios" || a1 == "-a") {
      const int iterations = argc > 2 ? std::atoi(argv[2]) : 50;
      const uint32_t seed =
          argc > 3
              ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10))
              : 1;
      if (iterations <= 0) {
        printUsage(argv[0]);
        return 1;
      }
      return runAllScenarios(iterations, seed);
    }
  }

  int32_t numRows = 262144;
  int32_t numGroups = 4096;
  int iterations = 50;
  uint32_t seed = 1;

  if (argc > 1) {
    numRows = std::atoi(argv[1]);
  }
  if (argc > 2) {
    numGroups = std::atoi(argv[2]);
  }
  if (argc > 3) {
    iterations = std::atoi(argv[3]);
  }
  if (argc > 4) {
    seed = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
  }

  if (numRows <= 0 || numGroups <= 0 || iterations <= 0) {
    printUsage(argv[0]);
    return 1;
  }

  return runOneBenchmark(numRows, numGroups, iterations, seed, nullptr);
}

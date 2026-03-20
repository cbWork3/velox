/*
 * Benchmark: SVE-optimized hash aggregate SUM(int32 -> int64)
 *
 * Compares 6 implementations:
 *   1. Scalar baseline       - per-row isBitSet + accumulate
 *   2. SVE-Original          - mask-unpack tree from SumAggregateBase.h
 *   3. WordCTZ               - word-AND bitmap + ctz scalar accumulate
 *   4. WordCTZ-Prefetch      - same as 3 + software prefetch of group pointers
 *   5. WordCTZ-Unroll4       - same as 3 + 4x unrolled ctz inner loop
 *   6. CTZ+Unroll4+Prefetch  - extract all positions first, 4x unroll + look-ahead prefetch
 *
 * Build (ARM64 with SVE):
 *   g++ -O2 -march=armv8-a+sve -o bench_sve_sum_int32 bench_sve_sum_int32.cpp
 *
 * Run:
 *   ./bench_sve_sum_int32
 */

#include <arm_sve.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ---------- simulated group memory layout ----------
static constexpr int64_t ACCUMULATOR_OFFSET = 8;
static constexpr int64_t NULL_BYTE_OFFSET   = 0;
static constexpr uint8_t NULL_MASK          = 0x01;
static constexpr int     GROUP_SIZE         = 32;

static inline int64_t* groupAccumulator(char* group) {
  return reinterpret_cast<int64_t*>(group + ACCUMULATOR_OFFSET);
}

template <typename T>
static inline bool isBitSet(const T* bits, uint64_t idx) {
  return bits[idx / (sizeof(bits[0]) * 8)] &
         (static_cast<T>(1) << (idx & ((sizeof(bits[0]) * 8) - 1)));
}

template <typename T, typename U>
static constexpr inline T roundUp(T value, U factor) {
  return (value + (factor - 1)) / factor * factor;
}

// =====================================================================
//  1. Scalar baseline
// =====================================================================

static void __attribute__((noinline)) scalarUpdate(
    char** result,
    uint64_t* bitmap1,
    uint64_t* bitmap2,
    int32_t* value,
    int32_t begin,
    int32_t end) {
  for (int32_t i = begin; i < end; ++i) {
    if (!isBitSet(bitmap1, i))
      continue;
    if (bitmap2 != nullptr && !isBitSet(bitmap2, i))
      continue;
    char* group = result[i];
    *reinterpret_cast<uint8_t*>(group + NULL_BYTE_OFFSET) &= ~NULL_MASK;
    *groupAccumulator(group) += static_cast<int64_t>(value[i]);
  }
}

// =====================================================================
//  2. Original SVE (mask-unpack tree) - from SumAggregateBase.h line 653
// =====================================================================

static int g_numNulls = 0;

static inline bool clearNullSVE_orig(svuint64_t ptr, svbool_t pg) {
  if (g_numNulls) {
    svint64_t group =
        svld1sb_gather_u64base_offset_s64(pg, ptr, NULL_BYTE_OFFSET);
    svuint8_t group8 = svreinterpret_u8(group);
    svuint8_t tmp = svand_n_u8_z(pg, group8, NULL_MASK);
    svbool_t test = svcmpne_n_u8(svptrue_b8(), tmp, 0);
    if (svptest_any(svptrue_b8(), test)) {
      uint8_t negNull = ~NULL_MASK;
      svuint8_t adjust = svand_n_u8_m(test, group8, negNull);
      svst1b_scatter_u64base_offset_s64(pg, ptr, NULL_BYTE_OFFSET,
                                        svreinterpret_s64(adjust));
      g_numNulls -= svcntp_b8(test, test);
      return true;
    }
  }
  return false;
}

static inline __attribute__((always_inline)) svbool_t
getUinqMask(svbool_t pg, const svuint64_t val) {
  svuint64_t s1 = svext_u64(val, val, 1);
  svbool_t mask2 = svcmpeq(svwhilelt_b64(0, 3), val, s1);
  svuint64_t s2 = svext_u64(val, val, 2);
  svbool_t mask3 = svcmpeq(svwhilelt_b64(0, 2), val, s2);
  svbool_t mask12 = svorr_b_z(pg, mask2, mask3);
  svuint64_t s3 = svext_u64(val, val, 3);
  svbool_t mask4 = svcmpeq(svwhilelt_b64(0, 1), val, s3);
  svbool_t mask = svorr_b_z(pg, mask4, mask12);
  return svnot_b_z(pg, mask);
}

static void __attribute__((noinline)) sveOrigUpdate(
    char** result,
    uint64_t* bitmap1,
    uint64_t* bitmap2,
    int32_t* value,
    int32_t begin,
    int32_t end) {
  uint8_t* bitmap1_8 = reinterpret_cast<uint8_t*>(bitmap1);
  uint8_t* bitmap2_8 = reinterpret_cast<uint8_t*>(bitmap2);
  int32_t firstWord =
      roundUp(begin, 32) == begin ? begin : roundUp(begin, 32) - 32;
  int32_t lastWord = roundUp(end, 32);
  svbool_t mask, mask1, mask2;

  for (int32_t count = firstWord; count + 32 <= lastWord; count += 32) {
    int32_t arr8Index = count / 8;
    if (bitmap2_8 != nullptr) {
      __asm__ __volatile__("ldr %0, [%1]"
                           : "=Upl"(mask2)
                           : "r"(&bitmap2_8[arr8Index])
                           : "memory");
    } else {
      mask2 = svptrue_b8();
    }
    __asm__ __volatile__("ldr %0, [%1]"
                         : "=Upl"(mask1)
                         : "r"(&bitmap1_8[arr8Index])
                         : "memory");
    mask = svand_b_z(svptrue_b8(), mask1, mask2);
    mask = svand_b_z(svptrue_b8(), mask, svwhilelt_b8(count, end));
    if (!svptest_any(svptrue_b8(), mask))
      continue;

#define ORIG_LANE(lane_mask, offset)                                           \
  do {                                                                         \
    if (svptest_any(svptrue_b64(), lane_mask)) {                               \
      svuint64_t ptr =                                                         \
          svld1(lane_mask,                                                     \
                reinterpret_cast<uint64_t*>(result + count + (offset)));        \
      svbool_t m = getUinqMask(lane_mask, ptr);                                \
      clearNullSVE_orig(ptr, m);                                               \
      uint8_t flag[4] = {0, 0, 0, 0};                                         \
      __asm__ __volatile__("str %1, [%0]"                                      \
                           :                                                   \
                           : "r"(&flag[0]), "Upl"(lane_mask)                   \
                           : "memory");                                        \
      for (int i = 0; i < 4; i++) {                                            \
        if (flag[i] != 0)                                                      \
          *groupAccumulator(*(result + count + (offset) + i)) +=               \
              value[count + (offset) + i];                                     \
      }                                                                        \
    }                                                                          \
  } while (0)

    svbool_t mask00 = svunpklo(mask);
    svbool_t mask01 = svunpkhi(mask);
    if (svptest_any(svptrue_b16(), mask00)) {
      svbool_t mask10 = svunpklo(mask00);
      if (svptest_any(svptrue_b32(), mask10)) {
        ORIG_LANE(svunpklo(mask10), 0);
        ORIG_LANE(svunpkhi(mask10), 4);
      }
      svbool_t mask11 = svunpkhi(mask00);
      if (svptest_any(svptrue_b32(), mask11)) {
        ORIG_LANE(svunpklo(mask11), 8);
        ORIG_LANE(svunpkhi(mask11), 12);
      }
    }
    if (svptest_any(svptrue_b16(), mask01)) {
      svbool_t mask12 = svunpklo(mask01);
      if (svptest_any(svptrue_b32(), mask12)) {
        ORIG_LANE(svunpklo(mask12), 16);
        ORIG_LANE(svunpkhi(mask12), 20);
      }
      svbool_t mask13 = svunpkhi(mask01);
      if (svptest_any(svptrue_b32(), mask13)) {
        ORIG_LANE(svunpklo(mask13), 24);
        ORIG_LANE(svunpkhi(mask13), 28);
      }
    }
#undef ORIG_LANE
  }
}

// =====================================================================
//  Shared: inline bitmap word masking for [begin, end)
// =====================================================================

static inline uint64_t getMaskedWord(
    uint64_t* bitmap1, uint64_t* bitmap2,
    int32_t w, int32_t rowBase, int32_t begin, int32_t end) {
  uint64_t bits = bitmap1[w];
  if (bitmap2 != nullptr)
    bits &= bitmap2[w];
  if (rowBase < begin)
    bits &= ~((1ULL << (begin - rowBase)) - 1);
  if (rowBase + 64 > end) {
    int shift = end - rowBase;
    if (shift < 64)
      bits &= (1ULL << shift) - 1;
  }
  return bits;
}

// =====================================================================
//  3. WordCTZ - word-level AND + ctz scan + scalar accumulate
// =====================================================================

static void __attribute__((noinline)) wordCtzUpdate(
    char** result,
    uint64_t* bitmap1,
    uint64_t* bitmap2,
    int32_t* value,
    int32_t begin,
    int32_t end) {
  int32_t wordBegin = begin / 64;
  int32_t wordEnd = (end + 63) / 64;

  for (int32_t w = wordBegin; w < wordEnd; ++w) {
    int32_t rowBase = w * 64;
    uint64_t bits = getMaskedWord(bitmap1, bitmap2, w, rowBase, begin, end);

    while (bits != 0) {
      int pos = __builtin_ctzll(bits);
      int32_t row = rowBase + pos;
      char* group = result[row];
      *reinterpret_cast<uint8_t*>(group + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *groupAccumulator(group) += static_cast<int64_t>(value[row]);
      bits &= bits - 1;
    }
  }
}

// =====================================================================
//  4. WordCTZ-Prefetch - same + prefetch next word's group pointers
// =====================================================================

static void __attribute__((noinline)) wordCtzPrefetchUpdate(
    char** result,
    uint64_t* bitmap1,
    uint64_t* bitmap2,
    int32_t* value,
    int32_t begin,
    int32_t end) {
  int32_t wordBegin = begin / 64;
  int32_t wordEnd = (end + 63) / 64;

  for (int32_t w = wordBegin; w < wordEnd; ++w) {
    int32_t rowBase = w * 64;
    uint64_t bits = getMaskedWord(bitmap1, bitmap2, w, rowBase, begin, end);

    // Prefetch group pointers and values for next word
    if (w + 1 < wordEnd) {
      int32_t nextBase = (w + 1) * 64;
      __builtin_prefetch(&result[nextBase], 0, 1);
      __builtin_prefetch(&value[nextBase], 0, 1);
      __builtin_prefetch(&bitmap1[w + 1], 0, 3);
      if (bitmap2)
        __builtin_prefetch(&bitmap2[w + 1], 0, 3);
    }

    // Prefetch the first few group targets in this word
    {
      uint64_t peek = bits;
      for (int p = 0; p < 4 && peek != 0; ++p) {
        int pos = __builtin_ctzll(peek);
        __builtin_prefetch(result[rowBase + pos], 1, 1);
        peek &= peek - 1;
      }
    }

    while (bits != 0) {
      int pos = __builtin_ctzll(bits);
      int32_t row = rowBase + pos;
      char* group = result[row];
      *reinterpret_cast<uint8_t*>(group + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *groupAccumulator(group) += static_cast<int64_t>(value[row]);
      bits &= bits - 1;

      // Prefetch next group target
      if (bits != 0) {
        int nextPos = __builtin_ctzll(bits);
        __builtin_prefetch(result[rowBase + nextPos], 1, 1);
      }
    }
  }
}

// =====================================================================
//  5. WordCTZ-Unroll4 - extract 4 positions at once, batch process
// =====================================================================

static void __attribute__((noinline)) wordCtzUnroll4Update(
    char** result,
    uint64_t* bitmap1,
    uint64_t* bitmap2,
    int32_t* value,
    int32_t begin,
    int32_t end) {
  int32_t wordBegin = begin / 64;
  int32_t wordEnd = (end + 63) / 64;

  for (int32_t w = wordBegin; w < wordEnd; ++w) {
    int32_t rowBase = w * 64;
    uint64_t bits = getMaskedWord(bitmap1, bitmap2, w, rowBase, begin, end);

    // Process 4 set bits at a time
    while (__builtin_popcountll(bits) >= 4) {
      int p0 = __builtin_ctzll(bits); bits &= bits - 1;
      int p1 = __builtin_ctzll(bits); bits &= bits - 1;
      int p2 = __builtin_ctzll(bits); bits &= bits - 1;
      int p3 = __builtin_ctzll(bits); bits &= bits - 1;

      int32_t r0 = rowBase + p0, r1 = rowBase + p1;
      int32_t r2 = rowBase + p2, r3 = rowBase + p3;

      char* g0 = result[r0]; char* g1 = result[r1];
      char* g2 = result[r2]; char* g3 = result[r3];

      *reinterpret_cast<uint8_t*>(g0 + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *reinterpret_cast<uint8_t*>(g1 + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *reinterpret_cast<uint8_t*>(g2 + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *reinterpret_cast<uint8_t*>(g3 + NULL_BYTE_OFFSET) &= ~NULL_MASK;

      *groupAccumulator(g0) += static_cast<int64_t>(value[r0]);
      *groupAccumulator(g1) += static_cast<int64_t>(value[r1]);
      *groupAccumulator(g2) += static_cast<int64_t>(value[r2]);
      *groupAccumulator(g3) += static_cast<int64_t>(value[r3]);
    }

    // Remainder
    while (bits != 0) {
      int pos = __builtin_ctzll(bits);
      int32_t row = rowBase + pos;
      char* group = result[row];
      *reinterpret_cast<uint8_t*>(group + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *groupAccumulator(group) += static_cast<int64_t>(value[row]);
      bits &= bits - 1;
    }
  }
}

// =====================================================================
//  6. WordCTZ+Unroll4+Prefetch - best of both: 4x unroll + prefetch
//     Avoids popcountll in loop condition (use simple bit-clear count).
//     Prefetches group memory for upcoming rows to hide cache latency.
// =====================================================================

static void __attribute__((noinline)) wordCtzUnroll4PfUpdate(
    char** result,
    uint64_t* bitmap1,
    uint64_t* bitmap2,
    int32_t* value,
    int32_t begin,
    int32_t end) {
  int32_t wordBegin = begin / 64;
  int32_t wordEnd = (end + 63) / 64;

  for (int32_t w = wordBegin; w < wordEnd; ++w) {
    int32_t rowBase = w * 64;
    uint64_t bits = getMaskedWord(bitmap1, bitmap2, w, rowBase, begin, end);
    if (bits == 0)
      continue;

    // Prefetch next word's metadata
    if (w + 1 < wordEnd) {
      __builtin_prefetch(&bitmap1[w + 1], 0, 3);
      if (bitmap2)
        __builtin_prefetch(&bitmap2[w + 1], 0, 3);
    }

    // Extract all positions first into a compact array (branchless inner loop)
    int32_t rows[64];
    int cnt = 0;
    {
      uint64_t tmp = bits;
      while (tmp != 0) {
        rows[cnt++] = rowBase + __builtin_ctzll(tmp);
        tmp &= tmp - 1;
      }
    }

    // Prefetch first batch of group pointers
    int pfEnd = cnt < 8 ? cnt : 8;
    for (int p = 0; p < pfEnd; ++p)
      __builtin_prefetch(result[rows[p]], 1, 1);

    // Process 4 at a time with look-ahead prefetch
    int i = 0;
    for (; i + 3 < cnt; i += 4) {
      // Prefetch groups 4 ahead
      if (i + 7 < cnt) {
        __builtin_prefetch(result[rows[i + 4]], 1, 1);
        __builtin_prefetch(result[rows[i + 5]], 1, 1);
        __builtin_prefetch(result[rows[i + 6]], 1, 1);
        __builtin_prefetch(result[rows[i + 7]], 1, 1);
      }

      char* g0 = result[rows[i]];
      char* g1 = result[rows[i + 1]];
      char* g2 = result[rows[i + 2]];
      char* g3 = result[rows[i + 3]];

      *reinterpret_cast<uint8_t*>(g0 + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *reinterpret_cast<uint8_t*>(g1 + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *reinterpret_cast<uint8_t*>(g2 + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *reinterpret_cast<uint8_t*>(g3 + NULL_BYTE_OFFSET) &= ~NULL_MASK;

      *groupAccumulator(g0) += static_cast<int64_t>(value[rows[i]]);
      *groupAccumulator(g1) += static_cast<int64_t>(value[rows[i + 1]]);
      *groupAccumulator(g2) += static_cast<int64_t>(value[rows[i + 2]]);
      *groupAccumulator(g3) += static_cast<int64_t>(value[rows[i + 3]]);
    }

    // Remainder
    for (; i < cnt; ++i) {
      char* group = result[rows[i]];
      *reinterpret_cast<uint8_t*>(group + NULL_BYTE_OFFSET) &= ~NULL_MASK;
      *groupAccumulator(group) += static_cast<int64_t>(value[rows[i]]);
    }
  }
}

// =====================================================================
//  Benchmark harness
// =====================================================================

struct BenchData {
  int                   numRows;
  int                   numGroups;
  std::vector<char*>    groups;
  std::vector<char>     groupStorage;
  std::vector<uint64_t> bitmap1;
  std::vector<uint64_t> bitmap2;
  std::vector<int32_t>  values;
};

static void initBenchData(BenchData& d, int numRows, int numGroups,
                           double selectivity, double nullRate,
                           unsigned seed) {
  d.numRows   = numRows;
  d.numGroups = numGroups;
  d.groupStorage.resize(static_cast<size_t>(numGroups) * GROUP_SIZE, 0);
  d.groups.resize(numRows);
  d.values.resize(numRows);

  int bitmapWords = (numRows + 63) / 64;
  d.bitmap1.assign(bitmapWords, 0);
  d.bitmap2.assign(bitmapWords, 0);

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> groupDist(0, numGroups - 1);
  std::uniform_int_distribution<int32_t> valDist(-10000, 10000);
  std::uniform_real_distribution<double> prob(0.0, 1.0);

  for (int i = 0; i < numRows; ++i) {
    int gid = groupDist(rng);
    d.groups[i] =
        d.groupStorage.data() + static_cast<size_t>(gid) * GROUP_SIZE;
    d.values[i] = valDist(rng);
    if (prob(rng) < selectivity)
      d.bitmap1[i / 64] |= (1ULL << (i % 64));
    if (prob(rng) < (1.0 - nullRate))
      d.bitmap2[i / 64] |= (1ULL << (i % 64));
  }
}

static void resetGroups(BenchData& d) {
  std::memset(d.groupStorage.data(), 0,
              static_cast<size_t>(d.numGroups) * GROUP_SIZE);
  for (int g = 0; g < d.numGroups; ++g) {
    char* base = d.groupStorage.data() + static_cast<size_t>(g) * GROUP_SIZE;
    reinterpret_cast<uint8_t*>(base + NULL_BYTE_OFFSET)[0] |= NULL_MASK;
  }
  g_numNulls = d.numGroups;
}

using Clock = std::chrono::high_resolution_clock;

typedef void (*UpdateFn)(char**, uint64_t*, uint64_t*, int32_t*, int32_t, int32_t);

static double bench(BenchData& d, UpdateFn fn, int iters) {
  // Warmup
  for (int it = 0; it < 3; ++it) {
    resetGroups(d);
    fn(d.groups.data(), d.bitmap1.data(), d.bitmap2.data(),
       d.values.data(), 0, d.numRows);
  }
  auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    resetGroups(d);
    fn(d.groups.data(), d.bitmap1.data(), d.bitmap2.data(),
       d.values.data(), 0, d.numRows);
  }
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static bool verify(BenchData& d, UpdateFn fn, const char* name) {
  std::vector<char> refStorage(d.groupStorage.size());
  std::vector<char> testStorage(d.groupStorage.size());

  resetGroups(d);
  scalarUpdate(d.groups.data(), d.bitmap1.data(), d.bitmap2.data(),
               d.values.data(), 0, d.numRows);
  std::memcpy(refStorage.data(), d.groupStorage.data(), d.groupStorage.size());

  resetGroups(d);
  fn(d.groups.data(), d.bitmap1.data(), d.bitmap2.data(),
     d.values.data(), 0, d.numRows);
  std::memcpy(testStorage.data(), d.groupStorage.data(), d.groupStorage.size());

  for (int g = 0; g < d.numGroups; ++g) {
    size_t off = static_cast<size_t>(g) * GROUP_SIZE + ACCUMULATOR_OFFSET;
    int64_t ref  = *reinterpret_cast<int64_t*>(refStorage.data() + off);
    int64_t test = *reinterpret_cast<int64_t*>(testStorage.data() + off);
    if (ref != test) {
      fprintf(stderr, "[FAIL] %s group %d: expected=%ld got=%ld\n",
              name, g, ref, test);
      return false;
    }
  }
  return true;
}

struct BenchConfig {
  const char* label;
  int         numRows;
  int         numGroups;
  double      selectivity;
  double      nullRate;
};

struct Impl {
  const char* name;
  UpdateFn    fn;
};

int main() {
  printf("================================================================\n");
  printf("  Benchmark: SUM(int32->int64) hash aggregate\n");
  printf("  SVE vector length: %u bits\n", (unsigned)svcntb() * 8);
  printf("================================================================\n\n");

  Impl impls[] = {
      {"Scalar",         scalarUpdate},
      {"SVE-Original",   sveOrigUpdate},
      {"WordCTZ",        wordCtzUpdate},
      {"WordCTZ+PF",     wordCtzPrefetchUpdate},
      {"WordCTZ+Unrl4",  wordCtzUnroll4Update},
      {"CTZ+Unrl4+PF",   wordCtzUnroll4PfUpdate},
  };
  int numImpls = sizeof(impls) / sizeof(impls[0]);

  BenchConfig configs[] = {
      {"1K/64g/100%/0%n",       1024,    64,    1.0, 0.0},
      {"1K/64g/100%/20%n",      1024,    64,    1.0, 0.2},
      {"1K/64g/80%/10%n",       1024,    64,    0.8, 0.1},
      {"8K/256g/100%/0%n",      8192,    256,   1.0, 0.0},
      {"8K/256g/100%/20%n",     8192,    256,   1.0, 0.2},
      {"8K/256g/80%/10%n",      8192,    256,   0.8, 0.1},
      {"64K/1Kg/100%/0%n",      65536,   1024,  1.0, 0.0},
      {"64K/1Kg/80%/10%n",      65536,   1024,  0.8, 0.1},
      {"256K/4Kg/100%/0%n",     262144,  4096,  1.0, 0.0},
      {"256K/4Kg/80%/10%n",     262144,  4096,  0.8, 0.1},
      {"1M/4Kg/100%/0%n",       1048576, 4096,  1.0, 0.0},
      {"1M/4Kg/80%/10%n",       1048576, 4096,  0.8, 0.1},
      {"1M/64Kg/100%/0%n",      1048576, 65536, 1.0, 0.0},
  };
  int numConfigs = sizeof(configs) / sizeof(configs[0]);

  // Print header
  printf("%-22s", "Scenario");
  for (int j = 0; j < numImpls; ++j)
    printf(" %13s", impls[j].name);
  printf("  BestSpeedup\n");
  for (int k = 0; k < 22 + numImpls * 14 + 13; ++k)
    putchar('-');
  putchar('\n');

  for (int c = 0; c < numConfigs; ++c) {
    auto& cfg = configs[c];
    BenchData d;
    initBenchData(d, cfg.numRows, cfg.numGroups, cfg.selectivity,
                  cfg.nullRate, 42 + c);

    bool allOk = true;
    for (int j = 1; j < numImpls; ++j) {
      if (!verify(d, impls[j].fn, impls[j].name)) {
        fprintf(stderr, "  -> FAILED for scenario: %s\n", cfg.label);
        allOk = false;
      }
    }
    if (!allOk)
      continue;

    int iters = cfg.numRows <= 8192   ? 5000
                : cfg.numRows <= 65536 ? 1000
                                       : 200;

    double times[16];
    for (int j = 0; j < numImpls; ++j)
      times[j] = bench(d, impls[j].fn, iters);

    double scalarTime = times[0];
    double bestOpt = 1e18;
    for (int j = 1; j < numImpls; ++j)
      if (times[j] < bestOpt)
        bestOpt = times[j];

    printf("%-22s", cfg.label);
    for (int j = 0; j < numImpls; ++j)
      printf(" %10.4fms", times[j]);
    printf("  %7.2fx\n", scalarTime / bestOpt);
  }

  printf("\nDone.\n");
  return 0;
}

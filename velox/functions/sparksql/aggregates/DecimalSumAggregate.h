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
#pragma once

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/type/DecimalUtil.h"
#include <arm_sve.h>

namespace facebook::velox::functions::aggregate::sparksql {

/// @tparam TInputType The raw input data type.
/// @tparam TSumType The type of sum in the output of partial aggregation or the
/// final output type of final aggregation.
/// @tparam ResultPrecision The precision of the result type, used for checking
/// overflow.
template <typename TInputType, typename TSumType, uint8_t ResultPrecision>
class DecimalSumAggregate {
 public:
  using InputType = Row<TInputType>;

  using IntermediateType =
      Row</*sum*/ TSumType,
          /*isEmpty*/ bool>;

  using OutputType = TSumType;

  /// Spark's decimal sum doesn't have the concept of a null group, each group
  /// is initialized with an initial value, where sum = 0 and isEmpty = true.
  /// The final agg may fallback to being executed in Spark, so the meaning of
  /// the intermediate data should be consistent with Spark. Therefore, we need
  /// to use the parameter nonNullGroup in writeIntermediateResult to output a
  /// null group as sum = 0, isEmpty = true. nonNullGroup is only available when
  /// default-null behavior is disabled.
  static constexpr bool default_null_behavior_ = false;

  static bool toIntermediate(
      exec::out_type<Row<TSumType, bool>>& out,
      exec::optional_arg_type<TInputType> in) {
    if (in.has_value()) {
      out.copy_from(std::make_tuple(static_cast<TSumType>(in.value()), false));
    } else {
      out.copy_from(std::make_tuple(static_cast<TSumType>(0), true));
    }
    return true;
  }

  /// This struct stores the sum of input values, overflow during accumulation,
  /// and a bool value isEmpty used to indicate whether all inputs are null. The
  /// initial value of sum is 0. We need to keep sum unchanged if the input is
  /// null, as sum function ignores null input. If the isEmpty is true, then it
  /// means there were no values to begin with or all the values were null, so
  /// the result will be null. If the isEmpty is false, then if sum is nullopt
  /// that means an overflow has happened, it returns null.
  struct AccumulatorType {
    std::optional<int128_t> sum{0};
    int64_t overflow{0};
    bool isEmpty{true};

    static constexpr bool is_aligned_ = true;

    AccumulatorType() = delete;

    explicit AccumulatorType(HashStringAllocator* /*allocator*/) {}

    std::optional<int128_t> computeFinalResult() const {
      if (!sum.has_value()) {
        return std::nullopt;
      }
      auto const adjustedSum =
          DecimalUtil::adjustSumForOverflow(sum.value(), overflow);
      if (adjustedSum.has_value() &&
          DecimalUtil::valueInPrecisionRange(adjustedSum, ResultPrecision)) {
        return adjustedSum;
      } else {
        // Found overflow during computing adjusted sum.
        return std::nullopt;
      }
    }

    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<TInputType> data) {
      if (!data.has_value()) {
        return false;
      }
      if (!sum.has_value()) {
        // sum is initialized to 0. When it is nullopt, it implies that the
        // input data must not be empty.
        VELOX_CHECK(!isEmpty);
        return true;
      }
      int128_t result;
      overflow +=
          DecimalUtil::addWithOverflow(result, data.value(), sum.value());
      sum = result;
      isEmpty = false;
      return true;
    }

    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<Row<TSumType, bool>> other) {
      if (!other.has_value()) {
        return false;
      }
      auto const otherSum = other.value().template at<0>();
      auto const otherIsEmpty = other.value().template at<1>();

      // isEmpty is never null.
      VELOX_CHECK(otherIsEmpty.has_value());
      if (isEmpty && otherIsEmpty.value()) {
        // Both accumulators are empty, no need to do the combination.
        return false;
      }

      bool currentOverflow = !isEmpty && !sum.has_value();
      bool otherOverflow = !otherIsEmpty.value() && !otherSum.has_value();
      if (currentOverflow || otherOverflow) {
        sum = std::nullopt;
        isEmpty = false;
      } else {
        int128_t result;
        overflow +=
            DecimalUtil::addWithOverflow(result, otherSum.value(), sum.value());
        sum = result;
        isEmpty &= otherIsEmpty.value();
      }
      return true;
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<IntermediateType>& out) {
      if (!nonNullGroup) {
        // If a group is null, all values in this group are null. In Spark, this
        // group will be the initial value, where sum is 0 and isEmpty is true.
        out = std::make_tuple(static_cast<TSumType>(0), true);
      } else {
        auto finalResult = computeFinalResult();
        if (finalResult.has_value()) {
          out = std::make_tuple(
              static_cast<TSumType>(finalResult.value()), isEmpty);
        } else {
          // Sum should be set to null on overflow,
          // and isEmpty should be set to false.
          out.template set_null_at<0>();
          out.template get_writer_at<1>() = false;
        }
      }
      return true;
    }

    bool writeFinalResult(bool nonNullGroup, exec::out_type<OutputType>& out) {
      if (!nonNullGroup || isEmpty) {
        // If isEmpty is true, we should set null.
        return false;
      }
      auto finalResult = computeFinalResult();
      if (finalResult.has_value()) {
        out = static_cast<TSumType>(finalResult.value());
        return true;
      } else {
        // Sum should be set to null on overflow.
        return false;
      }
    }
  };
};

/// Optimized version that overrides addRawInput with SVE-accelerated bitmap
/// scanning + density-based dispatch: dense batches use SVE vectorized int128
/// accumulation (lo/hi split + carry propagation), sparse batches use CTZ
/// extraction + 4x unrolled scalar accumulation with prefetch.
/// Assumes VL=256 (svptrue_b8 handles 32 elements per iteration).
template <typename TInputType, typename TSumType, uint8_t ResultPrecision>
class OptimizedSparkDecimalSumAggregate
    : public exec::SimpleAggregateAdapter<
          DecimalSumAggregate<TInputType, TSumType, ResultPrecision>> {
  using Base = exec::SimpleAggregateAdapter<
      DecimalSumAggregate<TInputType, TSumType, ResultPrecision>>;
  using AccumulatorType = typename DecimalSumAggregate<
      TInputType, TSumType, ResultPrecision>::AccumulatorType;

  // When popcount(bits) >= this threshold within a 32-row block, take the SVE
  // path; otherwise fall back to scalar CTZ+unroll. Tunable via benchmarking.
  static constexpr int kSVEDensityThreshold = 8;

 public:
  explicit OptimizedSparkDecimalSumAggregate(TypePtr resultType)
      : Base(std::move(resultType)) {
    // The SVE path relies on a stable AccumulatorType memory layout.
    // std::optional<int128_t> is expected to be 24 bytes: 16 bytes for value
    // then a has_value flag (with padding). The int128_t value sits at the
    // beginning of the optional on common implementations (GCC/Clang aarch64).
    static_assert(
        sizeof(std::optional<int128_t>) <= 32,
        "Unexpected std::optional<int128_t> size");
    static_assert(
        alignof(AccumulatorType) >= alignof(int128_t),
        "AccumulatorType must be aligned to int128_t");
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedInput_.decode(*args[0], rows);
    hashAggUpdateDecimal(
        groups,
        rows.getBits(),
        decodedInput_.getNulls(),
        reinterpret_cast<const TInputType*>(decodedInput_.getData()),
        rows.getBegin(),
        rows.getEnd(),
        decodedInput_.getMode1(),
        decodedInput_.getmode2(),
        reinterpret_cast<uint32_t*>(decodedInput_.getDic()));
  }

 private:
  template <typename T>
  inline bool isBitSet(const T* bits, uint64_t idx) {
    return bits[idx / (sizeof(bits[0]) * 8)] &
        (static_cast<T>(1) << (idx & ((sizeof(bits[0]) * 8) - 1)));
  }

  inline bool isBitNull(const uint64_t* bits, int32_t index) {
    return isBitSet(bits, index) == false;
  }

  template <typename U>
  constexpr inline U roundUp(U val, U factor) {
    return (val + (factor - 1)) / factor * factor;
  }

  // -----------------------------------------------------------------------
  // SVE bitmap helpers (adapted from SumAggregateBase int64 SVE)
  // -----------------------------------------------------------------------

  /// Load 32 rows of null bitmap as an SVE predicate, respecting mode.
  svbool_t getBitMaskSVE128(
      uint8_t* nulls,
      int32_t byteIndex,
      int mode,
      uint32_t* dic,
      int32_t length) {
    svbool_t pg;
    if (mode == 0) {
      return svptrue_b8();
    } else if (mode == 1) {
      __asm__ __volatile__("ldr %0, [%1]"
                           : "=Upl"(pg)
                           : "r"(&nulls[byteIndex])
                           : "memory");
      return pg;
    } else if (mode == 2) {
      if (!isBitNull(reinterpret_cast<uint64_t*>(nulls), 0)) {
        return svptrue_b8();
      }
      return svpfalse();
    } else if (mode == 3) {
      svuint32_t onc = svdup_u32(1);
      svuint32_t inv = svindex_u32(0, 1);
      svuint32_t pow = svlsl_m(svptrue_b32(), onc, inv);
      uint8_t tmpNulls[4] = {0, 0, 0, 0};
      uint32_t* null32ptr = reinterpret_cast<uint32_t*>(nulls);

      for (int blk = 0; blk < 4; ++blk) {
        int32_t base = byteIndex * 8 + blk * 8;
        svbool_t pg1 = svwhilelt_b32(base, length);
        svuint32_t posv = svld1(pg1, dic + base);
        svuint32_t idxbufv = svlsr_x(pg1, posv, 5);
        svuint32_t bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
        svuint32_t offsetv = svand_m(pg1, posv, 0b11111);
        bufv = svlsr_m(pg1, bufv, offsetv);
        bufv = svand_m(pg1, bufv, 0x1);
        svbool_t nullvec = svcmpgt(pg1, bufv, 0u);
        if (__builtin_expect(svptest_any(pg1, nullvec), 0)) {
          tmpNulls[blk] = static_cast<uint8_t>(svaddv(nullvec, pow));
        }
      }

      __asm__ __volatile__("ldr %0, [%1]"
                           : "=Upl"(pg)
                           : "r"(tmpNulls)
                           : "memory");
      return pg;
    }
    return svpfalse();
  }

  /// SVE batch clearNull: gather null bytes, mask off nullMask, scatter back.
  inline void clearNullSVE128(svuint64_t groupPtrs, svbool_t pg) {
    if (this->numNulls_) {
      svint64_t group = svld1sb_gather_u64base_offset_s64(
          pg, groupPtrs, this->nullByte_);
      svuint8_t group8 = svreinterpret_u8(group);
      svuint8_t tmp = svand_n_u8_z(pg, group8, this->nullMask_);
      svbool_t test = svcmpne_n_u8(svptrue_b8(), tmp, 0);
      if (svptest_any(svptrue_b8(), test)) {
        uint8_t negNull = ~this->nullMask_;
        svuint8_t adjust = svand_n_u8_m(test, group8, negNull);
        svst1b_scatter_u64base_offset_s64(
            pg, groupPtrs, this->nullByte_, svreinterpret_s64(adjust));
        this->numNulls_ -= svcntp_b8(test, test);
      }
    }
  }

  /// Load int128 input values as (lo64, hi64) pair using SVE gather.
  /// Each int128 is 16 bytes: lo64 at offset 0, hi64 at offset 8.
  /// With VL=256, svint64_t holds 4 x int64, so we process 2 int128 per call
  /// when working at b64 granularity (2 active lanes).
  inline void getValueSVE128(
      const TInputType* rawValue,
      int mode2,
      svbool_t pg,
      uint32_t rowIndex,
      uint32_t* dic,
      svint64_t& outLo,
      svint64_t& outHi) {
    if constexpr (sizeof(TInputType) == 16) {
      // int128_t input: each element is 16 bytes, load lo and hi halves.
      if (mode2 == 3) {
        // Dictionary: gather by dic[rowIndex..]
        svbool_t pg64to32 = svuzp1_b8(pg, svpfalse());
        svuint32_t dicIdx = svld1(pg64to32, dic + rowIndex);
        svuint64_t dicIdx64 = svunpklo(dicIdx);
        // Scale index by 16 (sizeof(int128_t)) to get byte offset, then load
        // lo64 at byteOff+0 and hi64 at byteOff+8.
        svuint64_t byteOff = svlsl_x(pg, dicIdx64, 4);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(rawValue);
        outLo = svld1_gather_offset_s64(
            pg, reinterpret_cast<const int64_t*>(base), byteOff);
        svuint64_t byteOffHi = svadd_x(pg, byteOff, 8);
        outHi = svld1_gather_offset_s64(
            pg, reinterpret_cast<const int64_t*>(base), byteOffHi);
      } else {
        // Identity/constant: contiguous load.
        const int64_t* p =
            reinterpret_cast<const int64_t*>(rawValue + rowIndex);
        // Interleaved memory: [lo0, hi0, lo1, hi1, ...]
        // Use svld2 to de-interleave.
        svint64x2_t pair = svld2(pg, p);
        outLo = svget2(pair, 0);
        outHi = svget2(pair, 1);
      }
    } else {
      // int64_t input: sign-extend to int128 (hi = arith shift right 63).
      svint64_t val64;
      if (mode2 == 3) {
        svbool_t pg64to32 = svuzp1_b8(pg, svpfalse());
        svuint32_t dicIdx = svld1(pg64to32, dic + rowIndex);
        svuint64_t dicIdx64 = svunpklo(dicIdx);
        val64 = svld1_gather_index(
            pg, reinterpret_cast<const int64_t*>(rawValue), dicIdx64);
      } else if (mode2 == 2) {
        val64 = svdup_n_s64(
            static_cast<int64_t>(rawValue[0]));
      } else {
        val64 = svld1(pg, reinterpret_cast<const int64_t*>(rawValue) + rowIndex);
      }
      outLo = val64;
      outHi = svasr_x(pg, val64, 63);
    }
  }

  /// Vectorized int128 addWithOverflow using SVE.
  /// Operates on (lo, hi) representation. Returns overflow per lane as int64.
  ///
  /// Algorithm mirrors DecimalUtil::addWithOverflow:
  ///   - If signs differ: result = lhs + rhs (no overflow possible)
  ///   - If both positive: unsigned add, overflow = (unsignedSum >> 127)
  ///   - If both negative: negate both, unsigned add, negate result,
  ///     overflow = -(unsignedSum >> 127)
  ///
  /// For SVE we simplify: always do unsigned 128-bit add of abs values when
  /// same-sign, using lo/hi split with carry propagation.
  inline void sveAddInt128WithOverflow(
      svbool_t pg,
      svint64_t lhsLo,
      svint64_t lhsHi,
      svint64_t rhsLo,
      svint64_t rhsHi,
      svint64_t& resultLo,
      svint64_t& resultHi,
      svint64_t& overflow) {
    svuint64_t uLhsLo = svreinterpret_u64(lhsLo);
    svuint64_t uLhsHi = svreinterpret_u64(lhsHi);
    svuint64_t uRhsLo = svreinterpret_u64(rhsLo);
    svuint64_t uRhsHi = svreinterpret_u64(rhsHi);

    // Sign detection: negative if hi < 0
    svbool_t lhsNeg = svcmplt(pg, lhsHi, 0);
    svbool_t rhsNeg = svcmplt(pg, rhsHi, 0);

    // Same sign mask
    svbool_t sameSign = svnot_b_z(pg, sveor_b_z(pg, lhsNeg, rhsNeg));
    // Different sign: just add directly (no overflow)
    svbool_t diffSign = sveor_b_z(pg, lhsNeg, rhsNeg);

    // --- Different sign path: result = lhs + rhs, overflow = 0 ---
    svuint64_t diffLo = svadd_m(diffSign, uLhsLo, uRhsLo);
    // Carry: if result < either operand (unsigned), there was a carry
    svbool_t diffCarry = svcmplt(diffSign, diffLo, uLhsLo);
    svuint64_t diffHi = svadd_m(diffSign, uLhsHi, uRhsHi);
    diffHi = svadd_m(diffCarry, diffHi, 1);

    // --- Same sign path: need overflow detection ---
    // Both positive: unsigned add directly
    svbool_t bothPos = svbic_b_z(pg, sameSign, lhsNeg);
    // Both negative: negate both, add, negate result
    svbool_t bothNeg = svand_b_z(pg, sameSign, lhsNeg);

    // For both-negative: abs(x) = ~x + 1 (two's complement negate 128-bit)
    // svnot_u64_x: inactive lanes are don't-care (we select via svsel later)
    svuint64_t notLhsLo = svnot_u64_x(bothNeg, uLhsLo);
    svuint64_t negLhsLo = svadd_m(bothNeg, notLhsLo, 1);
    svbool_t negLhsCarry = svcmpeq(bothNeg, negLhsLo, (uint64_t)0);
    svuint64_t negLhsHi = svnot_u64_x(bothNeg, uLhsHi);
    negLhsHi = svadd_m(negLhsCarry, negLhsHi, 1);

    svuint64_t notRhsLo = svnot_u64_x(bothNeg, uRhsLo);
    svuint64_t negRhsLo = svadd_m(bothNeg, notRhsLo, 1);
    svbool_t negRhsCarry = svcmpeq(bothNeg, negRhsLo, (uint64_t)0);
    svuint64_t negRhsHi = svnot_u64_x(bothNeg, uRhsHi);
    negRhsHi = svadd_m(negRhsCarry, negRhsHi, 1);

    // Select abs values: for bothNeg use negated, for bothPos use original
    svuint64_t aLo = svsel(bothNeg, negLhsLo, uLhsLo);
    svuint64_t aHi = svsel(bothNeg, negLhsHi, uLhsHi);
    svuint64_t bLo = svsel(bothNeg, negRhsLo, uRhsLo);
    svuint64_t bHi = svsel(bothNeg, negRhsHi, uRhsHi);

    // Unsigned 128-bit add of abs values
    svuint64_t sumLo = svadd_m(sameSign, aLo, bLo);
    svbool_t carry = svcmplt(sameSign, sumLo, aLo);
    svuint64_t sumHi = svadd_m(sameSign, aHi, bHi);
    sumHi = svadd_m(carry, sumHi, 1);

    // Overflow = bit 127 of the unsigned sum
    svuint64_t overflowBit = svlsr_m(sameSign, sumHi, 63);

    // Mask out bit 127 from result (keep lower 127 bits)
    svuint64_t mask127 = svdup_u64(0x7FFFFFFFFFFFFFFFULL);
    svuint64_t maskedHi = svand_m(sameSign, sumHi, mask127);

    // For both-negative: negate the result back and negate overflow
    svuint64_t finalSameLo = sumLo;
    svuint64_t finalSameHi = maskedHi;

    svuint64_t notResLo = svnot_u64_x(bothNeg, finalSameLo);
    svuint64_t reNegLo = svadd_m(bothNeg, notResLo, 1);
    svbool_t reNegCarry = svcmpeq(bothNeg, reNegLo, (uint64_t)0);
    svuint64_t reNegHi = svnot_u64_x(bothNeg, finalSameHi);
    reNegHi = svadd_m(reNegCarry, reNegHi, 1);

    finalSameLo = svsel(bothNeg, reNegLo, finalSameLo);
    finalSameHi = svsel(bothNeg, reNegHi, finalSameHi);

    // Combine paths: select same-sign or diff-sign results
    resultLo = svreinterpret_s64(svsel(sameSign, finalSameLo, diffLo));
    resultHi = svreinterpret_s64(svsel(sameSign, finalSameHi, diffHi));

    // Overflow: +1 for both-positive, -1 for both-negative, 0 for diff-sign
    svint64_t posOvf = svreinterpret_s64(overflowBit);
    svint64_t negOvf = svneg_s64_m(posOvf, bothNeg, posOvf);
    overflow = svsel(sameSign, negOvf, svdup_s64(0));
  }

  /// Process a group of 2 int128 rows via SVE at b64 granularity.
  /// pg is a b64 predicate with up to 4 active lanes (=2 int128 elements).
  /// groupPtrs holds group base addresses for each lane pair.
  inline void processSVEGroup(
      svbool_t pg,
      svuint64_t groupPtrs,
      const TInputType* rawValue,
      int mode2,
      uint32_t rowIndex,
      uint32_t* dic) {
    const auto offset = this->offset_;

    clearNullSVE128(groupPtrs, pg);

    // We need to determine the offset of the raw int128_t value within
    // std::optional<int128_t>. On GCC/Clang aarch64, the value is at offset 0
    // and the has_value flag follows.
    constexpr int64_t kSumLoOffset = 0;
    constexpr int64_t kSumHiOffset = 8;
    constexpr int64_t kOverflowOffset =
        static_cast<int64_t>(sizeof(std::optional<int128_t>));
    constexpr int64_t kIsEmptyOffset =
        kOverflowOffset + static_cast<int64_t>(sizeof(int64_t));

    // Load accumulator sum lo/hi via gather
    svint64_t accLo = svld1_gather_u64base_offset_s64(
        pg, groupPtrs, offset + kSumLoOffset);
    svint64_t accHi = svld1_gather_u64base_offset_s64(
        pg, groupPtrs, offset + kSumHiOffset);

    // Load input values as lo/hi
    svint64_t inLo, inHi;
    getValueSVE128(rawValue, mode2, pg, rowIndex, dic, inLo, inHi);

    // Vectorized int128 add with overflow detection
    svint64_t resLo, resHi, ovf;
    sveAddInt128WithOverflow(pg, inLo, inHi, accLo, accHi, resLo, resHi, ovf);

    // Store results back
    svst1_scatter_u64base_offset_s64(pg, groupPtrs, offset + kSumLoOffset, resLo);
    svst1_scatter_u64base_offset_s64(pg, groupPtrs, offset + kSumHiOffset, resHi);

    // Update overflow: acc->overflow += ovf
    svint64_t curOvf = svld1_gather_u64base_offset_s64(
        pg, groupPtrs, offset + kOverflowOffset);
    curOvf = svadd_m(pg, curOvf, ovf);
    svst1_scatter_u64base_offset_s64(
        pg, groupPtrs, offset + kOverflowOffset, curOvf);

    // Set isEmpty = false (0) for active lanes
    svst1b_scatter_u64base_offset_s64(
        pg, groupPtrs, offset + kIsEmptyOffset, svdup_s64(0));
  }

  /// SVE path: process a 32-row block via SVE predicate pipeline.
  /// Unpacks the b8 mask down to b64 in stages, processing 2 int128
  /// elements (4 int64 lanes) per leaf call.
  void processSVEBlock(
      char** groups,
      svbool_t mask,
      int32_t count,
      const TInputType* rawValue,
      int mode2,
      uint32_t* dic) {
    // Level 0: b8 -> b16
    svbool_t mask00 = svunpklo(mask);
    svbool_t mask01 = svunpkhi(mask);

    auto processHalf16 = [&](svbool_t m16, int32_t base16) {
      if (!svptest_any(svptrue_b16(), m16))
        return;
      svbool_t m32lo = svunpklo(m16);
      svbool_t m32hi = svunpkhi(m16);

      auto processQuarter32 = [&](svbool_t m32, int32_t base32) {
        if (!svptest_any(svptrue_b32(), m32))
          return;
        svbool_t m64lo = svunpklo(m32);
        svbool_t m64hi = svunpkhi(m32);

        if (svptest_any(svptrue_b64(), m64lo)) {
          svuint64_t ptrs = svld1(
              m64lo, reinterpret_cast<uint64_t*>(groups + base32));
          processSVEGroup(
              m64lo, ptrs, rawValue, mode2,
              static_cast<uint32_t>(base32), dic);
        }
        if (svptest_any(svptrue_b64(), m64hi)) {
          svuint64_t ptrs = svld1(
              m64hi, reinterpret_cast<uint64_t*>(groups + base32 + 4));
          processSVEGroup(
              m64hi, ptrs, rawValue, mode2,
              static_cast<uint32_t>(base32 + 4), dic);
        }
      };

      processQuarter32(m32lo, base16);
      processQuarter32(m32hi, base16 + 8);
    };

    processHalf16(mask00, count);
    processHalf16(mask01, count + 16);
  }

  // -----------------------------------------------------------------------
  // Scalar path (existing implementation, for sparse data)
  // -----------------------------------------------------------------------

  void scalarProcessRows(
      char** groups,
      const int32_t* rows,
      int cnt,
      const TInputType* rawValue,
      int mode2,
      uint32_t* dic) {
    auto getValue = [&](int32_t idx) -> TInputType {
      return (mode2 == 3) ? rawValue[dic[idx]] : rawValue[idx];
    };

    const auto offset = this->offset_;
    int i = 0;
    for (; i + 3 < cnt; i += 4) {
      if (i + 7 < cnt) {
        __builtin_prefetch(groups[rows[i + 4]] + offset, 1, 1);
        __builtin_prefetch(groups[rows[i + 5]] + offset, 1, 1);
        __builtin_prefetch(groups[rows[i + 6]] + offset, 1, 1);
        __builtin_prefetch(groups[rows[i + 7]] + offset, 1, 1);
      }

      char* g0 = groups[rows[i]];
      char* g1 = groups[rows[i + 1]];
      char* g2 = groups[rows[i + 2]];
      char* g3 = groups[rows[i + 3]];

      auto* acc0 = this->template value<AccumulatorType>(g0);
      auto* acc1 = this->template value<AccumulatorType>(g1);
      auto* acc2 = this->template value<AccumulatorType>(g2);
      auto* acc3 = this->template value<AccumulatorType>(g3);

      if (__builtin_expect(
              acc0->sum.has_value() & acc1->sum.has_value() &
                  acc2->sum.has_value() & acc3->sum.has_value(),
              1)) {
        int128_t r0, r1, r2, r3;
        acc0->overflow += DecimalUtil::addWithOverflow(
            r0, static_cast<int128_t>(getValue(rows[i])),
            acc0->sum.value());
        acc1->overflow += DecimalUtil::addWithOverflow(
            r1, static_cast<int128_t>(getValue(rows[i + 1])),
            acc1->sum.value());
        acc2->overflow += DecimalUtil::addWithOverflow(
            r2, static_cast<int128_t>(getValue(rows[i + 2])),
            acc2->sum.value());
        acc3->overflow += DecimalUtil::addWithOverflow(
            r3, static_cast<int128_t>(getValue(rows[i + 3])),
            acc3->sum.value());
        acc0->sum = r0;
        acc1->sum = r1;
        acc2->sum = r2;
        acc3->sum = r3;
        acc0->isEmpty = false;
        acc1->isEmpty = false;
        acc2->isEmpty = false;
        acc3->isEmpty = false;
      } else {
        auto accumOne = [&](auto* acc, int32_t row) {
          if (acc->sum.has_value()) {
            int128_t r;
            acc->overflow += DecimalUtil::addWithOverflow(
                r, static_cast<int128_t>(getValue(row)),
                acc->sum.value());
            acc->sum = r;
            acc->isEmpty = false;
          }
        };
        accumOne(acc0, rows[i]);
        accumOne(acc1, rows[i + 1]);
        accumOne(acc2, rows[i + 2]);
        accumOne(acc3, rows[i + 3]);
      }

      this->clearNull(g0);
      this->clearNull(g1);
      this->clearNull(g2);
      this->clearNull(g3);
    }
    for (; i < cnt; ++i) {
      if (i + 4 < cnt) {
        __builtin_prefetch(groups[rows[i + 4]] + offset, 1, 1);
      }
      char* g = groups[rows[i]];
      auto* acc = this->template value<AccumulatorType>(g);
      if (__builtin_expect(acc->sum.has_value(), 1)) {
        int128_t result;
        acc->overflow += DecimalUtil::addWithOverflow(
            result, static_cast<int128_t>(getValue(rows[i])),
            acc->sum.value());
        acc->sum = result;
        acc->isEmpty = false;
      }
      this->clearNull(g);
    }
  }

  /// Scalar fallback for a 32-row block: CTZ extract + 4x unrolled.
  void scalarProcessBlock(
      char** groups,
      uint64_t bits,
      int32_t rowBase,
      const TInputType* rawValue,
      int mode2,
      uint32_t* dic) {
    int32_t extractedRows[64];
    int cnt = 0;
    uint64_t tmp = bits;
    while (tmp != 0) {
      extractedRows[cnt++] = rowBase + __builtin_ctzll(tmp);
      tmp &= tmp - 1;
    }
    scalarProcessRows(groups, extractedRows, cnt, rawValue, mode2, dic);
  }

  /// Scalar fallback with per-row null filtering (for mode3).
  void scalarProcessBlockWithNullCheck(
      char** groups,
      uint64_t bits,
      int32_t rowBase,
      const TInputType* rawValue,
      int mode2,
      uint32_t* dic,
      uint64_t* bitmap2) {
    auto getNullBitMode3 = [&](int32_t idx) -> bool {
      if (bitmap2 == nullptr)
        return true;
      return isBitSet(bitmap2, static_cast<uint64_t>(dic[idx]));
    };

    int32_t extractedRows[64];
    int cnt = 0;
    uint64_t tmp = bits;
    while (tmp != 0) {
      int32_t row = rowBase + __builtin_ctzll(tmp);
      if (getNullBitMode3(row))
        extractedRows[cnt++] = row;
      tmp &= tmp - 1;
    }
    scalarProcessRows(groups, extractedRows, cnt, rawValue, mode2, dic);
  }

  // -----------------------------------------------------------------------
  // Main entry: density-based dispatch between SVE and scalar paths
  // -----------------------------------------------------------------------

  void hashAggUpdateDecimal(
      char** groups,
      uint64_t* bitmap1,
      uint64_t* bitmap2,
      const TInputType* value,
      int32_t begin,
      int32_t end,
      int mode1,
      int mode2,
      uint32_t* dic) {
    uint8_t* bitmap1_8 = reinterpret_cast<uint8_t*>(bitmap1);
    uint8_t* bitmap2_8 = reinterpret_cast<uint8_t*>(bitmap2);

    // Align iteration to 32-row blocks for SVE (VL=256, b8 processes 32 rows)
    int32_t firstBlock =
        roundUp(begin, 32) == begin ? begin : roundUp(begin, 32) - 32;
    int32_t lastBlock = roundUp(end, 32);

    if (mode1 == 2) {
      // Constant null mapping: check once and return if null
      if (bitmap2 != nullptr &&
          isBitNull(reinterpret_cast<uint64_t*>(bitmap2), 0)) {
        return;
      }
    }

    for (int32_t count = firstBlock; count + 32 <= lastBlock; count += 32) {
      int32_t arr8Index = count / 8;

      // SVE load selection bitmap
      svbool_t selMask;
      __asm__ __volatile__("ldr %0, [%1]"
                           : "=Upl"(selMask)
                           : "r"(&bitmap1_8[arr8Index])
                           : "memory");

      // SVE load and merge null bitmap
      svbool_t nullMask;
      if (mode1 == 0 || mode1 == 2) {
        nullMask = svptrue_b8();
      } else if (mode1 == 1) {
        if (bitmap2_8 != nullptr) {
          nullMask = getBitMaskSVE128(
              bitmap2_8, arr8Index, mode1, dic, end);
        } else {
          nullMask = svptrue_b8();
        }
      } else {
        nullMask = getBitMaskSVE128(
            bitmap2_8, arr8Index, mode1, dic, end);
      }

      svbool_t mask = svand_b_z(svptrue_b8(), selMask, nullMask);

      // Apply range bounds: mask off lanes outside [begin, end)
      mask = svand_b_z(svptrue_b8(), mask, svwhilelt_b8(count, end));
      if (count < begin) {
        // svwhilelt_b8(count, begin) is true for lanes [count, begin),
        // i.e. the lanes we want to exclude. Invert and AND.
        svbool_t excludeMask = svwhilelt_b8(count, begin);
        svbool_t startMask = svnot_b_z(svptrue_b8(), excludeMask);
        mask = svand_b_z(svptrue_b8(), mask, startMask);
      }

      if (!svptest_any(svptrue_b8(), mask))
        continue;

      // Count active rows to decide SVE vs scalar
      int activeCnt = svcntp_b8(svptrue_b8(), mask);

      if (activeCnt >= kSVEDensityThreshold) {
        // --- SVE path: process via predicate pipeline ---
        // For mode3, the null filtering is already in the mask from
        // getBitMaskSVE128, so we can use the same SVE path.
        processSVEBlock(groups, mask, count, value, mode2, dic);
      } else {
        // --- Scalar path: CTZ extract + 4x unroll ---
        // Convert SVE predicate back to a uint64_t bitmask for scalar loop.
        // Store predicate to memory and reinterpret as uint32_t (32 bits).
        uint8_t predBytes[4] = {0, 0, 0, 0};
        __asm__ __volatile__("str %1, [%0]"
                             : : "r"(predBytes), "Upl"(mask) : "memory");
        uint32_t bits32 = *reinterpret_cast<uint32_t*>(predBytes);

        // Process each bit in the 32-bit mask
        int32_t extractedRows[32];
        int cnt = 0;
        uint32_t tmp = bits32;
        while (tmp != 0) {
          extractedRows[cnt++] = count + __builtin_ctz(tmp);
          tmp &= tmp - 1;
        }
        scalarProcessRows(groups, extractedRows, cnt, value, mode2, dic);
      }
    }
  }

  DecodedVector decodedInput_;
};

} // namespace facebook::velox::functions::aggregate::sparksql

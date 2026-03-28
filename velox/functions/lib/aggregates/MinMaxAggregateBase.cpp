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

#include "velox/functions/lib/aggregates/MinMaxAggregateBase.h"

#include <limits>
#include <arm_sve.h>
#include "velox/exec/AggregationHook.h"
#include "velox/functions/lib/CheckNestedNulls.h"
#include "velox/functions/lib/aggregates/Compare.h"
#include "velox/functions/lib/aggregates/SimpleNumericAggregate.h"
#include "velox/functions/lib/aggregates/SingleValueAccumulator.h"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::functions::aggregate {

namespace {

template <typename T>
inline bool isBitSet(const T* bits, uint64_t idx) {
  return bits[idx / (sizeof(bits[0]) * 8)] &
      (static_cast<T>(1) << (idx & ((sizeof(bits[0]) * 8) - 1)));
}

inline bool isBitNull(const uint64_t* bits, int32_t index) {
  return !isBitSet(bits, index);
}

template <typename T, typename U>
constexpr inline T roundUp(T value, U factor) {
  return (value + (factor - 1)) / factor * factor;
}

template <typename T>
struct MinMaxTrait : public std::numeric_limits<T> {};

template <typename T>
class SimpleNumericMinMaxAggregate : public SimpleNumericAggregate<T, T, T> {
  using BaseAggregate = SimpleNumericAggregate<T, T, T>;

 public:
  explicit SimpleNumericMinMaxAggregate(
      TypePtr resultType,
      TimestampPrecision precision)
      : BaseAggregate(resultType), timestampPrecision_(precision) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(T);
  }

  int32_t accumulatorAlignmentSize() const override {
    if constexpr (std::is_same_v<T, int128_t>) {
      // Override 'accumulatorAlignmentSize' for UnscaledLongDecimal values as
      // it uses int128_t type. Some CPUs don't support misaligned access to
      // int128_t type.
      return static_cast<int32_t>(sizeof(int128_t));
    } else {
      return 1;
    }
  }

  bool supportsToIntermediate() const override {
    return true;
  }

  void toIntermediate(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      VectorPtr& result) const override {
    this->singleInputAsIntermediate(rows, args, result);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    if constexpr (std::is_same_v<T, Timestamp>) {
      // Truncate timestamps to corresponding precision.
      BaseAggregate::template doExtractValues<Timestamp>(
          groups, numGroups, result, [&](char* group) {
            auto ts =
                *BaseAggregate::Aggregate::template value<Timestamp>(group);
            return Timestamp::truncate(ts, timestampPrecision_);
          });
    } else {
      BaseAggregate::template doExtractValues<T>(
          groups, numGroups, result, [&](char* group) {
            return *BaseAggregate::Aggregate::template value<T>(group);
          });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::template doExtractValues<T>(
        groups, numGroups, result, [&](char* group) {
          return *BaseAggregate::Aggregate::template value<T>(group);
        });
  }

 private:
  const TimestampPrecision timestampPrecision_;
};

template <typename T>
class SimpleNumericMaxAggregate : public SimpleNumericMinMaxAggregate<T> {
  using BaseAggregate = SimpleNumericAggregate<T, T, T>;

 public:
  explicit SimpleNumericMaxAggregate(
      TypePtr resultType,
      TimestampPrecision precision = TimestampPrecision::kMilliseconds)
      : SimpleNumericMinMaxAggregate<T>(resultType, precision) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    if constexpr (BaseAggregate::template kMayPushdown<T>) {
      if (!args[0]->type()->isDecimal()) {
        if (mayPushdown && args[0]->isLazy()) {
          BaseAggregate::template pushdown<
              velox::aggregate::MinMaxHook<T, false>>(groups, rows, args[0]);
          return;
        }
      } else {
        mayPushdown = false;
      }
    } else {
      mayPushdown = false;
    }
    BaseAggregate::template updateGroups<true, T>(
        groups, rows, args[0], updateGroup, mayPushdown);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::updateOneGroup(
        group,
        rows,
        args[0],
        updateGroup,
        [](T& result, T value, int /* unused */) { result = value; },
        mayPushdown,
        kInitialValue_);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      *exec::Aggregate::value<T>(groups[i]) = kInitialValue_;
    }
  }

  static inline void updateGroup(T& result, T value) {
    if constexpr (std::is_floating_point_v<T>) {
      if (util::floating_point::NaNAwareLessThan<T>{}(result, value)) {
        result = value;
      }
    } else {
      if (result < value) {
        result = value;
      }
    }
  }

 private:
  static const T kInitialValue_;
};

template <typename T>
const T SimpleNumericMaxAggregate<T>::kInitialValue_ = MinMaxTrait<T>::lowest();

// Negative INF is the smallest value of floating point type.
template <>
const float SimpleNumericMaxAggregate<float>::kInitialValue_ =
    -1 * MinMaxTrait<float>::infinity();

template <>
const double SimpleNumericMaxAggregate<double>::kInitialValue_ =
    -1 * MinMaxTrait<double>::infinity();

template <typename T>
class SimpleNumericMinAggregate : public SimpleNumericMinMaxAggregate<T> {
  using BaseAggregate = SimpleNumericAggregate<T, T, T>;

 public:
  explicit SimpleNumericMinAggregate(
      TypePtr resultType,
      TimestampPrecision precision = TimestampPrecision::kMilliseconds)
      : SimpleNumericMinMaxAggregate<T>(resultType, precision) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    if constexpr (BaseAggregate::template kMayPushdown<T>) {
      if (!args[0]->type()->isDecimal()) {
        if (mayPushdown && args[0]->isLazy()) {
          BaseAggregate::template pushdown<
              velox::aggregate::MinMaxHook<T, true>>(groups, rows, args[0]);
          return;
        }
      } else {
        mayPushdown = false;
      }
    } else {
      mayPushdown = false;
    }

    if constexpr (std::is_same_v<T, int64_t>) {
      if (this->numNulls_) {
        DecodedVector decoded(*args[0], rows, !mayPushdown);
        if (decoded.mayHaveNulls()) {
          hashAggUpdateSVEForMinInt64(
              groups,
              rows.getBits(),
              decoded.getNulls(),
              reinterpret_cast<int64_t*>(decoded.getData()),
              rows.getBegin(),
              rows.getEnd(),
              decoded.getMode1(),
              decoded.getmode2(),
              reinterpret_cast<uint32_t*>(decoded.getDic()));
          return;
        }
      }
    }

    BaseAggregate::template updateGroups<true, T>(
        groups, rows, args[0], updateGroup, mayPushdown);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::updateOneGroup(
        group,
        rows,
        args[0],
        updateGroup,
        [](T& result, T value, int /* unused */) { result = value; },
        mayPushdown,
        kInitialValue_);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }

 protected:
  static inline void updateGroup(T& result, T value) {
    if constexpr (std::is_floating_point_v<T>) {
      if (util::floating_point::NaNAwareGreaterThan<T>{}(result, value)) {
        result = value;
      }
    } else {
      if (result > value) {
        result = value;
      }
    }
  }

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      *exec::Aggregate::value<T>(groups[i]) = kInitialValue_;
    }
  }

 private:
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
    } else if (mode == 1) {
      __asm__ __volatile__("ldr %0, [%1]"
                           : "=Upl"(pg)
                           : "r"(&(nulls_[index]))
                           : "memory");
      return pg;
    } else if (mode == 2) {
      if (!isBitNull(reinterpret_cast<uint64_t*>(nulls_), 0)) {
        pg = svptrue_b8();
      } else {
        pg = svpfalse();
      }
      return pg;
    } else if (mode == 3) {
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
        uint8_t nullsres = svaddv(nullvec, pow);
        tmpNulls[0] = nullsres;
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
        uint8_t nullsres = svaddv(nullvec, pow);
        tmpNulls[1] = nullsres;
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
        uint8_t nullsres = svaddv(nullvec, pow);
        tmpNulls[2] = nullsres;
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
        uint8_t nullsres = svaddv(nullvec, pow);
        tmpNulls[3] = nullsres;
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
    if (this->numNulls_) {
      svint64_t group = svld1sb_gather_u64base_offset_s64(
          pg, ptr, this->nullByte_);
      svuint8_t group8 = svreinterpret_u8(group);

      svuint8_t tmp = svand_n_u8_z(pg, group8, this->nullMask_);
      svbool_t test = svcmpne_n_u8(svptrue_b8(), tmp, 0);
      if (svptest_any(svptrue_b8(), test)) {
        uint8_t negNull = ~this->nullMask_;
        svuint8_t adjust = svand_n_u8_m(test, group8, negNull);
        svst1b_scatter_u64base_offset_s64(
            pg, ptr, this->nullByte_, svreinterpret_s64(adjust));
        int num = svcntp_b8(test, test);
        this->numNulls_ -= num;
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

    svbool_t maskResult = svorr_b_z(pg, mask4, mask12);
    maskResult = svnot_b_z(pg, maskResult);
    return maskResult;
  }

  svint64_t getValueSVE(
      int64_t* value,
      int32_t mode,
      svbool_t pg,
      uint32_t index,
      uint32_t* dic) {
    svint64_t result;
    if (mode == 0 || mode == 1) {
      result = svld1_s64(pg, value + index);
    } else if (mode == 2) {
      result = svdup_n_s64(value[0]);
    } else if (mode == 3) {
      svbool_t pg64to32 = svuzp1_b8(pg, svpfalse());
      svuint32_t offset = svld1(pg64to32, dic + index);
      svuint64_t offsetLow = svunpklo(offset);
      result = svld1_gather_index(pg, value, offsetLow);
    }
    return result;
  }

  // SVE-accelerated path for int64_t MIN.
  // Uses SVE predicate registers for bitmap loading/combining/unpacking,
  // SVE gather for null clearing, and scalar min comparison to avoid
  // scatter conflicts when multiple rows map to the same group.
  //
  // mode1: null bitmap encoding (0=none, 1=flat, 2=constant, 3=dictionary)
  // mode2: value encoding (0/1=flat, 2=constant, 3=dictionary)
  void hashAggUpdateSVEForMinInt64(
      char** result,
      uint64_t* bitmap1,
      uint64_t* bitmap2,
      int64_t* value,
      int32_t begin,
      int32_t end,
      int mode1,
      int mode2,
      uint32_t* dic) {
    uint8_t* bitmap1_8 = reinterpret_cast<uint8_t*>(bitmap1);
    uint8_t* bitmap2_8 = reinterpret_cast<uint8_t*>(bitmap2);

    int32_t firstWord =
        roundUp(begin, 32) == begin ? begin : roundUp(begin, 32) - 32;
    int32_t lastWord = roundUp(end, 32);
    svbool_t mask, mask1, mask2;

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
      if (!svptest_any(svptrue_b8(), mask))
        continue;

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
                "str %1, [%0]"
                :
                : "r"(&flag0[0]), "Upl"(mask20)
                : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag0[i] != 0) {
                int64_t v = value[count + i];
                int64_t& c = *exec::Aggregate::value<int64_t>(
                    *(result + count + i));
                if (v < c)
                  c = v;
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask21)) {
            svuint64_t ptr = svld1(
                mask21, reinterpret_cast<uint64_t*>(result + count + 4));
            svbool_t m21 = getUinqMask(mask21, ptr);
            clearNullSVE(ptr, m21);
            uint8_t flag1[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]"
                :
                : "r"(&flag1[0]), "Upl"(mask21)
                : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag1[i] != 0) {
                int64_t v = value[count + 4 + i];
                int64_t& c = *exec::Aggregate::value<int64_t>(
                    *(result + count + 4 + i));
                if (v < c)
                  c = v;
              }
            }
          }
        }

        svbool_t mask11 = svunpkhi(mask00);
        if (svptest_any(svptrue_b32(), mask11)) {
          svbool_t mask22 = svunpklo(mask11);
          svbool_t mask23 = svunpkhi(mask11);
          if (svptest_any(svptrue_b64(), mask22)) {
            svuint64_t ptr = svld1(
                mask22, reinterpret_cast<uint64_t*>(result + count + 8));
            svbool_t m22 = getUinqMask(mask22, ptr);
            clearNullSVE(ptr, m22);
            uint8_t flag2[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]"
                :
                : "r"(&flag2[0]), "Upl"(mask22)
                : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag2[i] != 0) {
                int64_t v = value[count + 8 + i];
                int64_t& c = *exec::Aggregate::value<int64_t>(
                    *(result + count + 8 + i));
                if (v < c)
                  c = v;
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask23)) {
            svuint64_t ptr = svld1(
                mask23, reinterpret_cast<uint64_t*>(result + count + 12));
            svbool_t m23 = getUinqMask(mask23, ptr);
            clearNullSVE(ptr, m23);
            uint8_t flag3[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]"
                :
                : "r"(&flag3[0]), "Upl"(mask23)
                : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag3[i] != 0) {
                int64_t v = value[count + 12 + i];
                int64_t& c = *exec::Aggregate::value<int64_t>(
                    *(result + count + 12 + i));
                if (v < c)
                  c = v;
              }
            }
          }
        }
      }

      if (svptest_any(svptrue_b16(), mask01)) {
        svbool_t mask12 = svunpklo(mask01);
        if (svptest_any(svptrue_b32(), mask12)) {
          svbool_t mask24 = svunpklo(mask12);
          svbool_t mask25 = svunpkhi(mask12);
          if (svptest_any(svptrue_b64(), mask24)) {
            svuint64_t ptr = svld1(
                mask24, reinterpret_cast<uint64_t*>(result + count + 16));
            svbool_t m24 = getUinqMask(mask24, ptr);
            clearNullSVE(ptr, m24);
            uint8_t flag4[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]"
                :
                : "r"(&flag4[0]), "Upl"(mask24)
                : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag4[i] != 0) {
                int64_t v = value[count + 16 + i];
                int64_t& c = *exec::Aggregate::value<int64_t>(
                    *(result + count + 16 + i));
                if (v < c)
                  c = v;
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask25)) {
            svuint64_t ptr = svld1(
                mask25, reinterpret_cast<uint64_t*>(result + count + 20));
            svbool_t m25 = getUinqMask(mask25, ptr);
            clearNullSVE(ptr, m25);
            uint8_t flag5[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]"
                :
                : "r"(&flag5[0]), "Upl"(mask25)
                : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag5[i] != 0) {
                int64_t v = value[count + 20 + i];
                int64_t& c = *exec::Aggregate::value<int64_t>(
                    *(result + count + 20 + i));
                if (v < c)
                  c = v;
              }
            }
          }
        }

        svbool_t mask13 = svunpkhi(mask01);
        if (svptest_any(svptrue_b32(), mask13)) {
          svbool_t mask26 = svunpklo(mask13);
          svbool_t mask27 = svunpkhi(mask13);
          if (svptest_any(svptrue_b64(), mask26)) {
            svuint64_t ptr = svld1(
                mask26, reinterpret_cast<uint64_t*>(result + count + 24));
            svbool_t m26 = getUinqMask(mask26, ptr);
            clearNullSVE(ptr, m26);
            uint8_t flag6[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]"
                :
                : "r"(&flag6[0]), "Upl"(mask26)
                : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag6[i] != 0) {
                int64_t v = value[count + 24 + i];
                int64_t& c = *exec::Aggregate::value<int64_t>(
                    *(result + count + 24 + i));
                if (v < c)
                  c = v;
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask27)) {
            svuint64_t ptr = svld1(
                mask27, reinterpret_cast<uint64_t*>(result + count + 28));
            svbool_t m27 = getUinqMask(mask27, ptr);
            clearNullSVE(ptr, m27);
            uint8_t flag7[4] = {0, 0, 0, 0};
            __asm__ __volatile__(
                "str %1, [%0]"
                :
                : "r"(&flag7[0]), "Upl"(mask27)
                : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag7[i] != 0) {
                int64_t v = value[count + 28 + i];
                int64_t& c = *exec::Aggregate::value<int64_t>(
                    *(result + count + 28 + i));
                if (v < c)
                  c = v;
              }
            }
          }
        }
      }
    }
  }

  static const T kInitialValue_;
};

template <typename T>
const T SimpleNumericMinAggregate<T>::kInitialValue_ = MinMaxTrait<T>::max();

// In velox, NaN is considered larger than infinity for floating point types.
template <>
const float SimpleNumericMinAggregate<float>::kInitialValue_ =
    MinMaxTrait<float>::quiet_NaN();

template <>
const double SimpleNumericMinAggregate<double>::kInitialValue_ =
    MinMaxTrait<double>::quiet_NaN();

class MinMaxAggregateBase : public exec::Aggregate {
 public:
  explicit MinMaxAggregateBase(
      const TypePtr& resultType,
      bool throwOnNestedNulls)
      : exec::Aggregate(resultType), throwOnNestedNulls_(throwOnNestedNulls) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(SingleValueAccumulator);
  }

  bool supportsToIntermediate() const override {
    return true;
  }

  void toIntermediate(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      VectorPtr& result) const override {
    const auto& input = args[0];

    if (throwOnNestedNulls_) {
      DecodedVector decoded(*input, rows, true);
      auto indices = decoded.indices();
      rows.applyToSelected([&](vector_size_t i) {
        velox::functions::checkNestedNulls(
            decoded, indices, i, throwOnNestedNulls_);
      });
    }

    if (rows.isAllSelected()) {
      result = input;
      return;
    }

    auto* pool = allocator_->pool();

    // Set result to NULL for rows that are masked out.
    BufferPtr nulls = allocateNulls(rows.size(), pool, bits::kNull);
    rows.clearNulls(nulls);

    BufferPtr indices = allocateIndices(rows.size(), pool);
    auto* rawIndices = indices->asMutable<vector_size_t>();
    std::iota(rawIndices, rawIndices + rows.size(), 0);

    result = BaseVector::wrapInDictionary(nulls, indices, rows.size(), input);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    (*result)->resize(numGroups);

    uint64_t* rawNulls = nullptr;
    if ((*result)->mayHaveNulls()) {
      BufferPtr& nulls = (*result)->mutableNulls((*result)->size());
      rawNulls = nulls->asMutable<uint64_t>();
    }

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto accumulator = value<SingleValueAccumulator>(group);
      if (!accumulator->hasValue()) {
        (*result)->setNull(i, true);
      } else {
        if (rawNulls) {
          bits::clearBit(rawNulls, i);
        }
        accumulator->read(*result, i);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    // partial and final aggregations are the same
    extractValues(groups, numGroups, result);
  }

 protected:
  template <
      typename TCompareTest,
      CompareFlags::NullHandlingMode nullHandlingMode>
  void doUpdate(
      char** groups,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      TCompareTest compareTest) {
    DecodedVector decoded(*arg, rows, true);
    auto indices = decoded.indices();
    auto baseVector = decoded.base();

    if (decoded.isConstantMapping() && decoded.isNullAt(0)) {
      // nothing to do; all values are nulls
      return;
    }

    rows.applyToSelected([&](vector_size_t i) {
      if (velox::functions::checkNestedNulls(
              decoded, indices, i, throwOnNestedNulls_)) {
        return;
      }

      auto accumulator = value<SingleValueAccumulator>(groups[i]);
      if (!accumulator->hasValue() ||
          compareTest(compare(accumulator, decoded, i, nullHandlingMode))) {
        accumulator->write(baseVector, indices[i], allocator_);
      }
    });
  }

  template <
      typename TCompareTest,
      CompareFlags::NullHandlingMode nullHandlingMode>
  void doUpdateSingleGroup(
      char* group,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      TCompareTest compareTest) {
    DecodedVector decoded(*arg, rows, true);
    auto indices = decoded.indices();
    auto baseVector = decoded.base();

    if (decoded.isConstantMapping()) {
      if (velox::functions::checkNestedNulls(
              decoded, indices, 0, throwOnNestedNulls_)) {
        return;
      }

      auto accumulator = value<SingleValueAccumulator>(group);
      if (!accumulator->hasValue() ||
          compareTest(compare(accumulator, decoded, 0, nullHandlingMode))) {
        accumulator->write(baseVector, indices[0], allocator_);
      }
      return;
    }

    auto accumulator = value<SingleValueAccumulator>(group);
    rows.applyToSelected([&](vector_size_t i) {
      if (velox::functions::checkNestedNulls(
              decoded, indices, i, throwOnNestedNulls_)) {
        return;
      }
      if (!accumulator->hasValue() ||
          compareTest(compare(accumulator, decoded, i, nullHandlingMode))) {
        accumulator->write(baseVector, indices[i], allocator_);
      }
    });
  }

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) SingleValueAccumulator();
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto group : groups) {
      if (isInitialized(group)) {
        value<SingleValueAccumulator>(group)->destroy(allocator_);
      }
    }
  }

 private:
  const bool throwOnNestedNulls_;
};

template <CompareFlags::NullHandlingMode nullHandlingMode>
class MaxAggregate : public MinMaxAggregateBase {
 public:
  explicit MaxAggregate(const TypePtr& resultType, bool throwOnNestedNulls)
      : MinMaxAggregateBase(resultType, throwOnNestedNulls) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdate<std::function<bool(int32_t)>, nullHandlingMode>(
        groups, rows, args[0], [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdateSingleGroup<std::function<bool(int32_t)>, nullHandlingMode>(
        group, rows, args[0], [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }
};

template <CompareFlags::NullHandlingMode nullHandlingMode>
class MinAggregate : public MinMaxAggregateBase {
 public:
  explicit MinAggregate(const TypePtr& resultType, bool throwOnNestedNulls)
      : MinMaxAggregateBase(resultType, throwOnNestedNulls) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdate<std::function<bool(int32_t)>, nullHandlingMode>(
        groups, rows, args[0], [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdateSingleGroup<std::function<bool(int32_t)>, nullHandlingMode>(
        group, rows, args[0], [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }
};

template <
    template <typename T>
    class TSimpleNumericAggregate,
    template <CompareFlags::NullHandlingMode nullHandlingMode>
    typename TAggregate>
exec::AggregateFunctionFactory getMinMaxFunctionFactoryInternal(
    const std::string& name,
    CompareFlags::NullHandlingMode nullHandlingMode,
    TimestampPrecision precision) {
  auto factory = [name, nullHandlingMode, precision](
                     core::AggregationNode::Step step,
                     std::vector<TypePtr> argTypes,
                     const TypePtr& resultType,
                     const core::QueryConfig& /*config*/)
      -> std::unique_ptr<exec::Aggregate> {
    auto inputType = argTypes[0];

    if (inputType->providesCustomComparison()) {
      return std::make_unique<
          TAggregate<CompareFlags::NullHandlingMode::kNullAsIndeterminate>>(
          inputType, false);
    }

    switch (inputType->kind()) {
      case TypeKind::BOOLEAN:
        return std::make_unique<TSimpleNumericAggregate<bool>>(resultType);
      case TypeKind::TINYINT:
        return std::make_unique<TSimpleNumericAggregate<int8_t>>(resultType);
      case TypeKind::SMALLINT:
        return std::make_unique<TSimpleNumericAggregate<int16_t>>(resultType);
      case TypeKind::INTEGER:
        return std::make_unique<TSimpleNumericAggregate<int32_t>>(resultType);
      case TypeKind::BIGINT:
        return std::make_unique<TSimpleNumericAggregate<int64_t>>(resultType);
      case TypeKind::REAL:
        return std::make_unique<TSimpleNumericAggregate<float>>(resultType);
      case TypeKind::DOUBLE:
        return std::make_unique<TSimpleNumericAggregate<double>>(resultType);
      case TypeKind::TIMESTAMP:
        return std::make_unique<TSimpleNumericAggregate<Timestamp>>(
            resultType, precision);
      case TypeKind::HUGEINT:
        return std::make_unique<TSimpleNumericAggregate<int128_t>>(resultType);
      case TypeKind::VARBINARY:
        [[fallthrough]];
      case TypeKind::VARCHAR:
        return std::make_unique<
            TAggregate<CompareFlags::NullHandlingMode::kNullAsIndeterminate>>(
            inputType, false);
      case TypeKind::ARRAY:
        [[fallthrough]];
      case TypeKind::ROW:
        if (nullHandlingMode == CompareFlags::NullHandlingMode::kNullAsValue) {
          return std::make_unique<
              TAggregate<CompareFlags::NullHandlingMode::kNullAsValue>>(
              inputType, false);
        } else {
          return std::make_unique<
              TAggregate<CompareFlags::NullHandlingMode::kNullAsIndeterminate>>(
              inputType, true);
        }
      case TypeKind::UNKNOWN:
        return std::make_unique<TSimpleNumericAggregate<UnknownValue>>(
            resultType);
      default:
        VELOX_UNREACHABLE(
            "Unknown input type for {} aggregation {}",
            name,
            inputType->kindName());
    }
  };
  return factory;
}

} // namespace

exec::AggregateFunctionFactory getMinFunctionFactory(
    const std::string& name,
    CompareFlags::NullHandlingMode nullHandlingMode,
    TimestampPrecision precision) {
  return getMinMaxFunctionFactoryInternal<
      SimpleNumericMinAggregate,
      MinAggregate>(name, nullHandlingMode, precision);
}

exec::AggregateFunctionFactory getMaxFunctionFactory(
    const std::string& name,
    CompareFlags::NullHandlingMode nullHandlingMode,
    TimestampPrecision precision) {
  return getMinMaxFunctionFactoryInternal<
      SimpleNumericMaxAggregate,
      MaxAggregate>(name, nullHandlingMode, precision);
}
} // namespace facebook::velox::functions::aggregate

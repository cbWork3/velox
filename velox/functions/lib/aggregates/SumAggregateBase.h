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

#include "folly/CPortability.h"

#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/CheckedArithmeticImpl.h"
#include "velox/functions/lib/aggregates/DecimalAggregate.h"
#include "velox/functions/lib/aggregates/SimpleNumericAggregate.h"
#include <arm_sve.h>

namespace facebook::velox::functions::aggregate {

template <
    typename TInput,
    typename TAccumulator,
    typename ResultType,
    bool Overflow>
class SumAggregateBase
    : public SimpleNumericAggregate<TInput, TAccumulator, ResultType> {
  using BaseAggregate =
      SimpleNumericAggregate<TInput, TAccumulator, ResultType>;

 public:
  explicit SumAggregateBase(TypePtr resultType) : BaseAggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(TAccumulator);
  }

  int32_t accumulatorAlignmentSize() const override {
    return 1;
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::template doExtractValues<ResultType>(
        groups, numGroups, result, [&](char* group) {
          // 'ResultType' and 'TAccumulator' might not be same such as sum(real)
          // and we do an explicit type conversion here.
          return (ResultType)(*BaseAggregate::Aggregate::template value<
                              TAccumulator>(group));
        });
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::template doExtractValues<TAccumulator>(
        groups, numGroups, result, [&](char* group) {
          return *BaseAggregate::Aggregate::template value<TAccumulator>(group);
        });
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    updateInternal<TAccumulator>(groups, rows, args, mayPushdown);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    updateInternal<TAccumulator, TAccumulator>(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::template updateOneGroup<TAccumulator>(
        group,
        rows,
        args[0],
        &updateSingleValue<TAccumulator>,
        &updateDuplicateValues<TAccumulator>,
        mayPushdown,
        TAccumulator(0));
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::template updateOneGroup<TAccumulator, TAccumulator>(
        group,
        rows,
        args[0],
        &updateSingleValue<TAccumulator>,
        &updateDuplicateValues<TAccumulator>,
        mayPushdown,
        TAccumulator(0));
  }

 protected:
  template <typename T>
  static constexpr bool kMayPushdown = !std::is_same_v<T, int128_t> &&
      !std::is_same_v<T, Timestamp> && !std::is_same_v<T, UnknownValue>;

#define UNLIKELY(x) (__builtin_expect((x), 0))

  template <typename T>
  inline bool isBitSet(const T* bits, uint64_t idx) {
    return bits[idx / (sizeof(bits[0]) * 8)] &
        (static_cast<T>(1) << (idx & ((sizeof(bits[0]) * 8) - 1)));
  }

  inline bool isBitNull(const uint64_t* bits, int32_t index) {
    return isBitSet(bits, index) == false;
  }
  template <typename T, typename U>
  constexpr inline T roundUp(T value, U factor) {
    return (value + (factor - 1)) / factor * factor;
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
    } else if (mode == 1) {
      __asm__ __volatile__("ldr %0, [%1]"
                           : "=Upl"(pg)
                           : "r"(&(nulls_[index]))
                           : "memory");
      return pg;
    } else if (mode == 2) {
      if (!isBitNull(
              reinterpret_cast<uint64_t*>(nulls_),
              0))
      {
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

      // 处理第一个8元素块
      pg1 = svwhilelt_b32(index*8, length);
      // 使用安全的方式加载字典值，确保不会越界
      posv = svld1(pg1, dic + index * 8);
      idxbufv = svlsr_x(pg1, posv, 5); // div 32 得到uint32 对应下标
      bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
      offsetv = svand_m(pg1, posv, 0b11111); // uint32内偏移，mod 32
      bufv = svlsr_m(pg1, bufv, offsetv); // 右移，取第offsettv位
      bufv = svand_m(pg1, bufv, 0x1); // 将其他位置0
      nullvec = svcmpgt(pg1, bufv, 0);
      if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
        uint8_t nullsres = svaddv(nullvec, pow);
        tmpNulls[0] = nullsres;
      } else {
        tmpNulls[0] = 0;
      }

      // 处理第二个8元素块
      pg1 = svwhilelt_b32(index*8 + 8, length);
      posv = svld1(pg1, dic + index*8 + 8);
      idxbufv = svlsr_x(pg1, posv, 5); // div 32 得到uint32 对应下标
      bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
      offsetv = svand_m(pg1, posv, 0b11111); // uint32内偏移，mod 32
      bufv = svlsr_m(pg1, bufv, offsetv); // 右移，取第offsettv位
      bufv = svand_m(pg1, bufv, 0x1); // 将其他位置0
      nullvec = svcmpgt(pg1, bufv, 0);
      if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
        uint8_t nullsres = svaddv(nullvec, pow);
        tmpNulls[1] = nullsres;
      } else {
        tmpNulls[1] = 0;
      }

      // 处理第三个8元素块
      pg1 = svwhilelt_b32(index*8 + 16, length);
      posv = svld1(pg1, dic + index*8 + 16);
      idxbufv = svlsr_x(pg1, posv, 5); // div 32 得到uint32 对应下标
      bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
      offsetv = svand_m(pg1, posv, 0b11111); // uint32内偏移，mod 32
      bufv = svlsr_m(pg1, bufv, offsetv); // 右移，取第offsettv位
      bufv = svand_m(pg1, bufv, 0x1); // 将其他位置0
      nullvec = svcmpgt(pg1, bufv, 0);
      if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
        uint8_t nullsres = svaddv(nullvec, pow);
        tmpNulls[2] = nullsres;
      } else {
        tmpNulls[2] = 0;
      }

      // 处理第四个8元素块
      pg1 = svwhilelt_b32(index*8 + 24, length);
      posv = svld1(pg1, dic + index*8 + 24);
      idxbufv = svlsr_x(pg1, posv, 5); // div 32 得到uint32 对应下标
      bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
      offsetv = svand_m(pg1, posv, 0b11111); // uint32内偏移，mod 32
      bufv = svlsr_m(pg1, bufv, offsetv); // 右移，取第offsettv位
      bufv = svand_m(pg1, bufv, 0x1); // 将其他位置0
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
    // 默认返回全false掩码
    pg = svpfalse();
    return pg;
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
      // return;
    } else if (mode == 2) {
      result = svdup_n_s64(value[0]);
      // return;
    } else if (mode == 3) {
      // pg进来是对应int64(value)的如果要取对应dic，要调整p寄存器，原来可能是
      // 0001 0001 0001 0001，调整后0000 0000 0101 0101
      svbool_t pg64to32 = svuzp1_b8(pg, svpfalse());
      svuint32_t offset = svld1(pg64to32, dic + index);
      svuint64_t offsetLow = svunpklo(offset);
      result = svld1_gather_index(pg, value, offsetLow);
      // return;
    }
    return result;
  }

  bool clearNullSVE(svuint64_t ptr, svbool_t pg) //
  {
    if (this->numNulls_) {
      svint64_t group = svld1sb_gather_u64base_offset_s64(
          pg, ptr, this->nullByte_); // 这里要变
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

  svint64_t loadGatherResult(svuint64_t ptr, svbool_t mask, int64_t idx) {
    svint64_t value =
        svld1_gather_u64base_offset_s64(mask, ptr, idx); // 这里要变
    return value;
  }

  void storeScatterResult(svuint64_t ptr, svbool_t mask, svint64_t value, int64_t idx) {
    svst1_scatter_u64base_offset_s64(mask, ptr, idx, value);
  }

  void hashAggUpdateSVEWithChar(
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
    svint64_t tmpValue;
        // 注意这里的count是统计第几个元素，svbool_t去load，bitmap，一次性可以处理32个元素
            for (int32_t count = firstWord; count + 32 <= lastWord; count += 32) {
      int32_t arr8Index = count / 8;
      if (bitmap2_8 != nullptr) {
        mask2 = getBitMask(bitmap2_8, arr8Index, mode1, dic, end); // 一次取32个
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
            svint64_t tmpValue0;
            tmpValue0 = getValueSVE(value, mode2, mask20, count, dic);
            svuint64_t ptr =
                svld1(mask20, reinterpret_cast<uint64_t*>(result + count));
            clearNullSVE(ptr, mask20);
            svint64_t tmpResult0 = loadGatherResult(ptr, mask20, this->getOffsetFromAgg());
            tmpResult0 = svadd_m(mask20, tmpResult0, tmpValue0);
            storeScatterResult(ptr, mask20, tmpResult0, this->getOffsetFromAgg());
          }

          if (svptest_any(svptrue_b64(), mask21)) {
            svint64_t tmpValue1;
            tmpValue1 = getValueSVE(value, mode2, mask21, 4 + count, dic);
            svuint64_t ptr =
                svld1(mask21, reinterpret_cast<uint64_t*>(result + count + 4));
            clearNullSVE(ptr, mask21);
            svint64_t tmpResult1 = loadGatherResult(ptr, mask21, this->getOffsetFromAgg());
            tmpResult1 = svadd_m(mask21, tmpResult1, tmpValue1);
            storeScatterResult(ptr, mask21, tmpResult1, this->getOffsetFromAgg());
          }
        }
        svbool_t mask11 = svunpkhi(mask00);
        if (svptest_any(svptrue_b32(), mask11)) {
          svbool_t mask22 = svunpklo(mask11);
          svbool_t mask23 = svunpkhi(mask11);
          if (svptest_any(svptrue_b64(), mask22)) {
            svint64_t tmpValue2;
            tmpValue2 = getValueSVE(value, mode2, mask22, 8 + count, dic);
            svuint64_t ptr =
                svld1(mask22, reinterpret_cast<uint64_t*>(result + count + 8));
            clearNullSVE(ptr, mask22);
            svint64_t tmpResult2 = loadGatherResult(ptr, mask22, this->getOffsetFromAgg());
            tmpResult2 = svadd_m(mask22, tmpResult2, tmpValue2);
            storeScatterResult(ptr, mask22, tmpResult2, this->getOffsetFromAgg());
          }

          if (svptest_any(svptrue_b64(), mask23)) {
            svint64_t tmpValue3;
            tmpValue3 = getValueSVE(value, mode2, mask23, 12 + count, dic);
            svuint64_t ptr =
                svld1(mask23, reinterpret_cast<uint64_t*>(result + count + 12));
            clearNullSVE(ptr, mask23);
            svint64_t tmpResult3 = loadGatherResult(ptr, mask23, this->getOffsetFromAgg());
            tmpResult3 = svadd_m(mask23, tmpResult3, tmpValue3);
            storeScatterResult(ptr, mask23, tmpResult3, this->getOffsetFromAgg());
          }
        }
      }

      svbool_t mask12 = svunpklo(mask01);

      if (svptest_any(svptrue_b16(), mask01)) {
        svbool_t mask24 = svunpklo(mask12);
        svbool_t mask25 = svunpkhi(mask12);
        if (svptest_any(svptrue_b32(), mask12)) {
          if (svptest_any(svptrue_b64(), mask24)) {
            svint64_t tmpValue4;
            tmpValue4 = getValueSVE(value, mode2, mask24, 16 + count, dic);
            svuint64_t ptr =
                svld1(mask24, reinterpret_cast<uint64_t*>(result + count + 16));
            clearNullSVE(ptr, mask24);
            svint64_t tmpResult4 = loadGatherResult(ptr, mask24, this->getOffsetFromAgg());
            tmpResult4 = svadd_m(mask24, tmpResult4, tmpValue4);
            storeScatterResult(ptr, mask24, tmpResult4, this->getOffsetFromAgg());
          }

          if (svptest_any(svptrue_b64(), mask25)) {
            svint64_t tmpValue5;
            tmpValue5 = getValueSVE(value, mode2, mask25, 20 + count, dic);
            svuint64_t ptr =
                svld1(mask25, reinterpret_cast<uint64_t*>(result + count + 20));
            clearNullSVE(ptr, mask25);
            svint64_t tmpResult5 = loadGatherResult(ptr, mask25, this->getOffsetFromAgg());
            tmpResult5 = svadd_m(mask25, tmpResult5, tmpValue5);
            storeScatterResult(ptr, mask25, tmpResult5, this->getOffsetFromAgg());
          }
        }
        svbool_t mask13 = svunpkhi(mask01);

        if (svptest_any(svptrue_b32(), mask13)) {
          svbool_t mask26 = svunpklo(mask13);
          svbool_t mask27 = svunpkhi(mask13);
          if (svptest_any(svptrue_b64(), mask26)) {
            svint64_t tmpValue6;
            tmpValue6 = getValueSVE(value,  mode2, mask26, 24 + count, dic);
            svuint64_t ptr =
                svld1(mask26, reinterpret_cast<uint64_t*>(result + count + 24));
            clearNullSVE(ptr, mask26);
            svint64_t tmpResult6 = loadGatherResult(ptr, mask26, this->getOffsetFromAgg());
            tmpResult6 = svadd_m(mask26, tmpResult6, tmpValue6);
            storeScatterResult(ptr, mask26, tmpResult6, this->getOffsetFromAgg());
          }

          if (svptest_any(svptrue_b64(), mask27)) {
            svint64_t tmpValue7;
            tmpValue7 = getValueSVE(value, mode2, mask27, 28 + count, dic);
            svuint64_t ptr =
                svld1(mask27, reinterpret_cast<uint64_t*>(result + count + 28));
            clearNullSVE(ptr, mask27);
            svint64_t tmpResult7 = loadGatherResult(ptr, mask27, this->getOffsetFromAgg());
            tmpResult7 = svadd_m(mask27, tmpResult7, tmpValue7);
            storeScatterResult(ptr, mask27, tmpResult7, this->getOffsetFromAgg());
          }
        }
      }
    }
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

    void hashAggUpdateSVEWithCharForNormal(
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
    svint64_t tmpValue;
        // 注意这里的count是统计第几个元素，svbool_t去load，bitmap，一次性可以处理32个元素
            for (int32_t count = firstWord; count + 32 <= lastWord; count += 32) {
      int32_t arr8Index = count / 8;
      if (bitmap2_8 != nullptr) {
        mask2 = getBitMask(bitmap2_8, arr8Index, mode1, dic, end); // 一次取32个
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
            __asm__ __volatile__("str %1, [%0]": : "r" (&flag0[0]), "Upl" (mask20) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag0[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + i)) += value[count + i];
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask21)) {
            svuint64_t ptr =
                svld1(mask21, reinterpret_cast<uint64_t*>(result + count + 4));
            svbool_t m21 = getUinqMask(mask21, ptr);
            clearNullSVE(ptr, m21);
            uint8_t flag1[4] = {0, 0, 0, 0};
            __asm__ __volatile__("str %1, [%0]": : "r" (&flag1[0]), "Upl" (mask21) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag1[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 4 + i)) += value[count + 4 + i];
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
            __asm__ __volatile__("str %1, [%0]": : "r" (&flag2[0]), "Upl" (mask22) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag2[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 8 + i)) += value[count + 8 + i];
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask23)) {
            svuint64_t ptr =
                svld1(mask23, reinterpret_cast<uint64_t*>(result + count + 12));
            svbool_t m23 = getUinqMask(mask23, ptr);
            clearNullSVE(ptr, m23);
            uint8_t flag3[4] = {0, 0, 0, 0};
            __asm__ __volatile__("str %1, [%0]": : "r" (&flag3[0]), "Upl" (mask23) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag3[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 12 + i)) += value[count + 12 + i];
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
            __asm__ __volatile__("str %1, [%0]": : "r" (&flag4[0]), "Upl" (mask24) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag4[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 16 + i)) += value[count + 16 + i];
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask25)) {
            svuint64_t ptr =
                svld1(mask25, reinterpret_cast<uint64_t*>(result + count + 20));
            svbool_t m25 = getUinqMask(mask25, ptr);
            clearNullSVE(ptr, m25);
            uint8_t flag5[4] = {0, 0, 0, 0};
            __asm__ __volatile__("str %1, [%0]": : "r" (&flag5[0]), "Upl" (mask25) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag5[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 20 + i)) += value[count + 20 + i];
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
            __asm__ __volatile__("str %1, [%0]": : "r" (&flag6[0]), "Upl" (mask26) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag6[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 24 + i)) += value[count + 24 + i];
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask27)) {
            svuint64_t ptr =
                svld1(mask27, reinterpret_cast<uint64_t*>(result + count + 28));
            svbool_t m27 = getUinqMask(mask27, ptr);
            clearNullSVE(ptr, m27);
            uint8_t flag7[4] = {0, 0, 0, 0};
            __asm__ __volatile__("str %1, [%0]": : "r" (&flag7[0]), "Upl" (mask27) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag7[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 28 + i)) += value[count + 28 + i];
              }
            }
          }
        }
      }
    }
  }

  // SVE-optimized path for int32 SUM: loads int32 values and accumulates to
  // int64. Uses same structure as hashAggUpdateSVEWithCharForNormal but with
  // int32_t* value for correct memory layout.
  void hashAggUpdateSVEWithCharForNormalInt32(
      char** result,
      uint64_t* bitmap1,
      uint64_t* bitmap2,
      int32_t* value,
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
      } else {
        mask2 = svptrue_b8();
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
            __asm__ __volatile__("str %1, [%0]" : : "r"(&flag0[0]), "Upl"(mask20) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag0[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + i)) += value[count + i];
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask21)) {
            svuint64_t ptr =
                svld1(mask21, reinterpret_cast<uint64_t*>(result + count + 4));
            svbool_t m21 = getUinqMask(mask21, ptr);
            clearNullSVE(ptr, m21);
            uint8_t flag1[4] = {0, 0, 0, 0};
            __asm__ __volatile__("str %1, [%0]" : : "r"(&flag1[0]), "Upl"(mask21) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag1[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 4 + i)) += value[count + 4 + i];
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
            __asm__ __volatile__("str %1, [%0]" : : "r"(&flag2[0]), "Upl"(mask22) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag2[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 8 + i)) += value[count + 8 + i];
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask23)) {
            svuint64_t ptr =
                svld1(mask23, reinterpret_cast<uint64_t*>(result + count + 12));
            svbool_t m23 = getUinqMask(mask23, ptr);
            clearNullSVE(ptr, m23);
            uint8_t flag3[4] = {0, 0, 0, 0};
            __asm__ __volatile__("str %1, [%0]" : : "r"(&flag3[0]), "Upl"(mask23) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag3[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 12 + i)) += value[count + 12 + i];
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
            __asm__ __volatile__("str %1, [%0]" : : "r"(&flag4[0]), "Upl"(mask24) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag4[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 16 + i)) += value[count + 16 + i];
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask25)) {
            svuint64_t ptr =
                svld1(mask25, reinterpret_cast<uint64_t*>(result + count + 20));
            svbool_t m25 = getUinqMask(mask25, ptr);
            clearNullSVE(ptr, m25);
            uint8_t flag5[4] = {0, 0, 0, 0};
            __asm__ __volatile__("str %1, [%0]" : : "r"(&flag5[0]), "Upl"(mask25) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag5[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 20 + i)) += value[count + 20 + i];
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
            __asm__ __volatile__("str %1, [%0]" : : "r"(&flag6[0]), "Upl"(mask26) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag6[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 24 + i)) += value[count + 24 + i];
              }
            }
          }

          if (svptest_any(svptrue_b64(), mask27)) {
            svuint64_t ptr =
                svld1(mask27, reinterpret_cast<uint64_t*>(result + count + 28));
            svbool_t m27 = getUinqMask(mask27, ptr);
            clearNullSVE(ptr, m27);
            uint8_t flag7[4] = {0, 0, 0, 0};
            __asm__ __volatile__("str %1, [%0]" : : "r"(&flag7[0]), "Upl"(mask27) : "memory");
            for (int i = 0; i < 4; i++) {
              if (flag7[i] != 0) {
                *exec::Aggregate::value<int64_t>(*(result + count + 28 + i)) += value[count + 28 + i];
              }
            }
          }
        }
      }
    }
  }

  template <
      bool tableHasNulls,
      typename TData = ResultType,
      typename TValue = TInput,
      typename UpdateSingleValue>
  void updateGroups(
      char** groups,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      UpdateSingleValue updateSingleValue,
      bool mayPushdown,
      DecodedVector& decoded) {

    if constexpr (kMayPushdown<TData>) {
      auto encoding = decoded.base()->encoding();
      if (encoding == VectorEncoding::Simple::LAZY &&
          !arg->type()->isDecimal()) {
        velox::aggregate::SimpleCallableHook<TData, UpdateSingleValue> hook(
            exec::Aggregate::offset_,
            exec::Aggregate::nullByte_,
            exec::Aggregate::nullMask_,
            groups,
            &this->exec::Aggregate::numNulls_,
            updateSingleValue);

        auto indices = decoded.indices();
        decoded.base()->as<const LazyVector>()->load(
            RowSet(indices, arg->size()), &hook);
        return;
      }
    }
    // groups
    // rows.bits
    uint64_t* bitmask1 = rows.getBits();
    // decode.bits
    uint64_t* bitmask2 = decoded.getNulls();
    // decode value
    int64_t* value = reinterpret_cast<int64_t*>(decoded.getData());
    // begin, end
    vector_size_t begin = rows.getBegin();
    vector_size_t end = rows.getEnd();

    // mode1, mode2
    int mode1 = decoded.getMode1();
    int mode2 = decoded.getmode2();

    // decode dic
    vector_size_t* dic = decoded.getDic();

    if constexpr (std::is_same_v<TValue, int32_t>) {
      int32_t* value32 = reinterpret_cast<int32_t*>(decoded.getData());
      hashAggUpdateSVEWithCharForNormalInt32(
          groups,
          bitmask1,
          bitmask2,
          value32,
          begin,
          end,
          mode1,
          mode2,
          reinterpret_cast<uint32_t*>(dic));
    } else {
      hashAggUpdateSVEWithCharForNormal(
          groups,
          bitmask1,
          bitmask2,
          value,
          begin,
          end,
          mode1,
          mode2,
          reinterpret_cast<uint32_t*>(dic));
    }
  }

  template <
      bool tableHasNulls,
      typename TDataType = TAccumulator,
      typename Update>
  inline void
  updateNonNullValue(char* group, TDataType value, Update updateValue) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    updateValue(*exec::Aggregate::value<TDataType>(group), value);
  }
  // TData is used to store the updated sum state. It can be either
  // TAccumulator or TResult, which in most cases are the same, but for
  // sum(real) can differ. TValue is used to decode the sum input 'args'.
  // It can be either TAccumulator or TInput, which is most cases are the same
  // but for sum(real) can differ.
  template <typename TData, typename TValue = TInput>
  void updateInternal(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) {
    const auto& arg = args[0];

    if (mayPushdown && arg->isLazy()) {
      BaseAggregate::template pushdown<
          facebook::velox::aggregate::SumHook<TData, Overflow>>(
          groups, rows, arg);
      return;
    }

    if (exec::Aggregate::numNulls_) {
      DecodedVector decoded(*arg, rows, !mayPushdown);
      if (std::is_same_v<TData, int64_t> &&
          (std::is_same_v<TValue, int64_t> || std::is_same_v<TValue, int32_t>) &&
          decoded.mayHaveNulls() && Overflow) {
        updateGroups<true, TData, TValue>(
            groups, rows, arg, &updateSingleValue<TData>, false, decoded);
      } else {
        BaseAggregate::template updateGroups<true, TData, TValue>(
            groups, rows, arg, &updateSingleValue<TData>, false);
      }
    } else {
      BaseAggregate::template updateGroups<false, TData, TValue>(
          groups, rows, arg, &updateSingleValue<TData>, false);
    }
  }

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      *exec::Aggregate::value<TAccumulator>(groups[i]) = 0;
    }
  }

 private:
  template <typename TData>
  static void updateSingleValue(TData& result, TData value) {
    velox::aggregate::SumHook<TData, Overflow>::add(result, value);
  }

  // Disable undefined behavior sanitizer to not fail on signed integer
  // overflow.
  template <typename TData>
#if defined(FOLLY_DISABLE_UNDEFINED_BEHAVIOR_SANITIZER)
  FOLLY_DISABLE_UNDEFINED_BEHAVIOR_SANITIZER("signed-integer-overflow")
#endif
  static void updateDuplicateValues(TData& result, TData value, int n) {
    if constexpr (
        (std::is_same_v<TData, int64_t> && Overflow) ||
        std::is_same_v<TData, double> || std::is_same_v<TData, float>) {
      result += n * value;
    } else {
      result = functions::checkedPlus<TData>(
          result, functions::checkedMultiply<TData>(TData(n), value));
    }
  }
};

template <typename TInputType>
class DecimalSumAggregate
    : public functions::aggregate::DecimalAggregate<int128_t, TInputType> {
 public:
  explicit DecimalSumAggregate(TypePtr resultType)
      : functions::aggregate::DecimalAggregate<int128_t, TInputType>(
            resultType) {}

  virtual int128_t computeFinalValue(
      functions::aggregate::LongDecimalWithOverflowState* accumulator) final {
    auto sum = DecimalUtil::adjustSumForOverflow(
        accumulator->sum, accumulator->overflow);
    VELOX_USER_CHECK(sum.has_value(), "Decimal overflow");
    DecimalUtil::valueInRange(sum.value());
    return sum.value();
  }
};

} // namespace facebook::velox::functions::aggregate

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

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregationHook.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/LazyVector.h"

#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

namespace facebook::velox::functions::aggregate {


template <typename TInput, typename TAccumulator, typename TResult>
class SimpleNumericAggregate : public exec::Aggregate {
 protected:
  explicit SimpleNumericAggregate(TypePtr resultType) : Aggregate(resultType) {}

 public:
  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

 protected:
  template <typename T>
  static constexpr bool kMayPushdown = !std::is_same_v<T, int128_t> &&
      !std::is_same_v<T, Timestamp> && !std::is_same_v<T, UnknownValue>;

  // TData is either TAccumulator or TResult, which in most cases are the same,
  // but for sum(real) can differ.
  template <typename TData = TResult, typename ExtractOneValue>
  void doExtractValues(
      char** groups,
      int32_t numGroups,
      VectorPtr* result,
      ExtractOneValue extractOneValue) {
    VELOX_CHECK_EQ((*result)->encoding(), VectorEncoding::Simple::FLAT);
    auto vector = (*result)->as<FlatVector<TData>>();
    VELOX_CHECK(
        vector,
        "Unexpected type of the result vector: {}",
        (*result)->type()->toString());
    VELOX_CHECK_EQ(vector->elementSize(), sizeof(TData));
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);
    if constexpr (std::is_same_v<TData, bool>) {
      uint64_t* rawValues = vector->template mutableRawValues<uint64_t>();
      for (int32_t i = 0; i < numGroups; ++i) {
        char* group = groups[i];
        if (isNull(group)) {
          vector->setNull(i, true);
        } else {
          clearNull(rawNulls, i);
          bits::setBit(rawValues, i, extractOneValue(group));
        }
      }
    } else {
      TData* rawValues = vector->mutableRawValues();
      for (int32_t i = 0; i < numGroups; ++i) {
        char* group = groups[i];
        if (isNull(group)) {
          vector->setNull(i, true);
        } else {
          clearNull(rawNulls, i);
          rawValues[i] = extractOneValue(group);
        }
      }
    }
  }

  // TData is used to store the updated group states. It can be either
  // TAccumulator or TResult, which in most cases are the same, but for
  // sum(real) can differ. TValue is used to decode the update input 'args'.
  // It can be either TAccumulator or TInput, which is most cases are the same
  // but for sum(real) can differ.
  template <
      bool tableHasNulls,
      typename TData = TResult,
      typename TValue = TInput,
      typename UpdateSingleValue>
  void updateGroups(
      char** groups,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      UpdateSingleValue updateSingleValue,
      bool mayPushdown) {
    DecodedVector decoded(*arg, rows, !mayPushdown);
    auto encoding = decoded.base()->encoding();
    if constexpr (kMayPushdown<TData>) {
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

#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE
    // Spark SUM(INTEGER)->BIGINT (e.g. SUM(ss_quantity) GROUP BY ss_store_sk).
    // SVE batch path for perf profiling on AArch64+SVE builds.
    if constexpr (std::is_same_v<TData, int64_t> &&
                  std::is_same_v<TValue, int32_t>) {
      if (!decoded.isConstantMapping() && decoded.isIdentityMapping() &&
          !decoded.mayHaveNulls()) {
        updateGroupsSveSumInt32Impl(
            groups,
            rows.getBits(),
            decoded.data<int32_t>(),
            static_cast<int32_t>(rows.begin()),
            static_cast<int32_t>(rows.end()));
        return;
      }
    }
#endif

    if (decoded.isConstantMapping()) {
      if (!decoded.isNullAt(0)) {
        auto value = decoded.valueAt<TValue>(0);
        rows.applyToSelected([&](vector_size_t i) {
          updateNonNullValue<tableHasNulls, TData>(
              groups[i], TData(value), updateSingleValue);
        });
      }
    } else if (decoded.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decoded.isNullAt(i)) {
          return;
        }
        updateNonNullValue<tableHasNulls, TData>(
            groups[i], TData(decoded.valueAt<TValue>(i)), updateSingleValue);
      });
    } else if (decoded.isIdentityMapping() && !std::is_same_v<TValue, bool>) {
      auto data = decoded.data<TValue>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<tableHasNulls, TData>(
            groups[i], TData(data[i]), updateSingleValue);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<tableHasNulls, TData>(
            groups[i], TData(decoded.valueAt<TValue>(i)), updateSingleValue);
      });
    }
  }

  // TData is used to store the updated group state. It can be either
  // TAccumulator or TResult, which in most cases are the same, but for
  // sum(real) can differ. TValue is used to decode the update input 'args'.
  // It can be either TAccumulator or TInput, which is most cases are the same
  // but for sum(real) can differ.
  template <
      typename TData = TResult,
      typename TValue = TInput,
      typename UpdateSingle,
      typename UpdateDuplicate>
  void updateOneGroup(
      char* group,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      UpdateSingle updateSingleValue,
      UpdateDuplicate updateDuplicateValues,
      bool /*mayPushdown*/,
      TData initialValue) {
    DecodedVector decoded(*arg, rows);

    // Do row by row if not all rows are selected.
    if (decoded.isConstantMapping()) {
      if (!decoded.isNullAt(0)) {
        updateDuplicateValues(
            initialValue,
            TData(decoded.valueAt<TValue>(0)),
            rows.countSelected());
        updateNonNullValue<true, TData>(group, initialValue, updateSingleValue);
      }
    } else if (decoded.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decoded.isNullAt(i)) {
          return;
        }
        updateNonNullValue<true, TData>(
            group, TData(decoded.valueAt<TValue>(i)), updateSingleValue);
      });
    } else if (decoded.isIdentityMapping() && !std::is_same_v<TValue, bool>) {
      auto data = decoded.data<TValue>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<true, TData>(
            group, TData(data[i]), updateSingleValue);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<true, TData>(
            group, TData(decoded.valueAt<TValue>(i)), updateSingleValue);
      });
    }
  }

  template <typename THook>
  void
  pushdown(char** groups, const SelectivityVector& rows, const VectorPtr& arg) {
    DecodedVector decoded(*arg, rows, false);
    const vector_size_t* indices = decoded.indices();
    THook hook(
        exec::Aggregate::offset_,
        exec::Aggregate::nullByte_,
        exec::Aggregate::nullMask_,
        groups,
        &this->exec::Aggregate::numNulls_);
    // The decoded vector does not really keep the info from the 'rows', except
    // for the 'upper bound' of it. In case not all rows are selected we need to
    // generate proper indices, which we 'indirect' through the ones we got from
    // the decoded vector.
    vector_size_t numIndices{arg->size()};
    if (not rows.isAllSelected()) {
      const auto numSelected = rows.countSelected();
      if (numSelected != arg->size()) {
        pushdownCustomIndices_.resize(numSelected);
        vector_size_t tgtIndex{0};
        rows.template applyToSelected([&](vector_size_t i) {
          pushdownCustomIndices_[tgtIndex++] = indices[i];
        });
        indices = pushdownCustomIndices_.data();
        numIndices = numSelected;
      }
    }

    decoded.base()->as<const LazyVector>()->load(
        RowSet(indices, numIndices), &hook);
  }

#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE
 private:
  static constexpr int32_t roundUpSve(int32_t value, int32_t factor) {
    return (value + (factor - 1)) / factor * factor;
  }

  static svbool_t svmGetUinqMask(svbool_t pg, svuint64_t val) {
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

  void clearNullSveGather(svuint64_t ptr, svbool_t pg) {
    if (!numNulls_) {
      return;
    }
    svint64_t g = svld1sb_gather_u64base_offset_s64(pg, ptr, nullByte_);
    svuint8_t g8 = svreinterpret_u8(g);
    svuint8_t tmp = svand_n_u8_z(pg, g8, nullMask_);
    svbool_t test = svcmpne_n_u8(svptrue_b8(), tmp, 0);
    if (svptest_any(svptrue_b8(), test)) {
      uint8_t negNull = ~nullMask_;
      svuint8_t adjust = svand_n_u8_m(test, g8, negNull);
      svst1b_scatter_u64base_offset_s64(
          pg, ptr, nullByte_, svreinterpret_s64(adjust));
      numNulls_ -= svcntp_b8(test, test);
    }
  }

  void updateGroupsSveSumInt32Impl(
      char** groups,
      uint64_t* bitmap1,
      const int32_t* data,
      int32_t begin,
      int32_t end) {
    auto* bitmap1_8 = reinterpret_cast<uint8_t*>(bitmap1);
    const int32_t firstWord =
        roundUpSve(begin, 32) == begin ? begin : roundUpSve(begin, 32) - 32;
    const int32_t lastWord = roundUpSve(end, 32);
    const svbool_t maskAllRows = svptrue_b8();

    for (int32_t count = firstWord; count + 32 <= lastWord; count += 32) {
      const int32_t arr8Index = count / 8;
      svbool_t mask1;
      __asm__ __volatile__(
          "ldr %0, [%1]" : "=Upl"(mask1) : "r"(&bitmap1_8[arr8Index]) : "memory");
      svbool_t mask = svand_b_z(svptrue_b8(), mask1, maskAllRows);
      mask = svand_b_z(svptrue_b8(), mask, svwhilelt_b8(count, end));
      if (!svptest_any(svptrue_b8(), mask)) {
        continue;
      }

      const svbool_t mask00 = svunpklo(mask);
      const svbool_t mask01 = svunpkhi(mask);

      auto addFour = [&](int32_t base, svbool_t mask64) {
        if (!svptest_any(svptrue_b64(), mask64)) {
          return;
        }
        svuint64_t ptr =
            svld1(mask64, reinterpret_cast<uint64_t*>(groups + base));
        svbool_t m = svmGetUinqMask(mask64, ptr);
        clearNullSveGather(ptr, m);
        uint8_t flag[4] = {0, 0, 0, 0};
        __asm__ __volatile__(
            "str %1, [%0]" : : "r"(&flag[0]), "Upl"(m) : "memory");
        for (int i = 0; i < 4; ++i) {
          if (flag[i] != 0) {
            *this->template value<int64_t>(*(groups + base + i)) +=
                static_cast<int64_t>(data[base + i]);
          }
        }
      };

      if (svptest_any(svptrue_b16(), mask00)) {
        const svbool_t mask10 = svunpklo(mask00);
        if (svptest_any(svptrue_b32(), mask10)) {
          addFour(count, svunpklo(mask10));
          addFour(count + 4, svunpkhi(mask10));
        }
        const svbool_t mask11 = svunpkhi(mask00);
        if (svptest_any(svptrue_b32(), mask11)) {
          addFour(count + 8, svunpklo(mask11));
          addFour(count + 12, svunpkhi(mask11));
        }
      }
      if (svptest_any(svptrue_b16(), mask01)) {
        const svbool_t mask12 = svunpklo(mask01);
        if (svptest_any(svptrue_b32(), mask12)) {
          addFour(count + 16, svunpklo(mask12));
          addFour(count + 20, svunpkhi(mask12));
        }
        const svbool_t mask13 = svunpkhi(mask01);
        if (svptest_any(svptrue_b32(), mask13)) {
          addFour(count + 24, svunpklo(mask13));
          addFour(count + 28, svunpkhi(mask13));
        }
      }
    }
  }

#endif

 private:
  // TData is either TAccumulator or TResult, which in most cases are the same,
  // but for sum(real) can differ.
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
};

} // namespace facebook::velox::functions::aggregate

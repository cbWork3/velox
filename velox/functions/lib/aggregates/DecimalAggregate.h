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

#include "velox/common/base/IOUtils.h"
#include "velox/exec/Aggregate.h"
#include "velox/type/HugeInt.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::aggregate {

/**
 *  LongDecimalWithOverflowState has the following fields:
 *    SUM: Total sum so far.
 *    COUNT: Total number of rows so far.
 *    OVERFLOW: Total count of net overflow or underflow so far.
 */
struct LongDecimalWithOverflowState {
 public:
  void mergeWith(const StringView& serializedData) {
    VELOX_CHECK_EQ(serializedData.size(), serializedSize());
    auto serialized = serializedData.data();
    common::InputByteStream stream(serialized);
    count += stream.read<int64_t>();
    overflow += stream.read<int64_t>();
    uint64_t lowerSum = stream.read<uint64_t>();
    int64_t upperSum = stream.read<int64_t>();
    overflow += DecimalUtil::addWithOverflow(
        this->sum, HugeInt::build(upperSum, lowerSum), this->sum);
  }

  void serialize(StringView& serialized) {
    VELOX_CHECK_EQ(serialized.size(), serializedSize());
    char* outputBuffer = const_cast<char*>(serialized.data());
    common::OutputByteStream outStream(outputBuffer);
    outStream.append((char*)&count, sizeof(int64_t));
    outStream.append((char*)&overflow, sizeof(int64_t));
    uint64_t lower = HugeInt::lower(sum);
    int64_t upper = HugeInt::upper(sum);
    outStream.append((char*)&lower, sizeof(int64_t));
    outStream.append((char*)&upper, sizeof(int64_t));
  }

  /*
   * Total size = sizeOf(count) + sizeOf(overflow) + sizeOf(sum)
   *            = 8 + 8 + 16 = 32.
   */
  inline static size_t serializedSize() {
    return sizeof(int64_t) * 4;
  }

  int128_t sum{0};
  int64_t count{0};
  int64_t overflow{0};
};

template <typename TResultType, typename TInputType = TResultType>
class DecimalAggregate : public exec::Aggregate {
 public:
  explicit DecimalAggregate(TypePtr resultType) : exec::Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(LongDecimalWithOverflowState);
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(LongDecimalWithOverflowState);
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedRaw_.decode(*args[0], rows);
    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        auto value = decodedRaw_.valueAt<TInputType>(0);
        rows.applyToSelected([&](vector_size_t i) {
          updateNonNullValue(groups[i], TResultType(value));
        });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      hashAggUpdateInt128(
          groups,
          rows.getBits(),
          decodedRaw_.getNulls(),
          reinterpret_cast<const TInputType*>(decodedRaw_.getData()),
          rows.getBegin(),
          rows.getEnd(),
          decodedRaw_.getMode1(),
          decodedRaw_.getmode2(),
          reinterpret_cast<uint32_t*>(decodedRaw_.getDic()));
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<TInputType>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<false>(groups[i], TResultType(data[i]));
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue(
            groups[i], TResultType(decodedRaw_.valueAt<TInputType>(i)));
      });
    }
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedRaw_.decode(*args[0], rows);
    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        auto value = decodedRaw_.valueAt<TInputType>(0);
        rows.template applyToSelected([&](vector_size_t i) {
          updateNonNullValue(group, TResultType(value));
        });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(
              group, TResultType(decodedRaw_.valueAt<TInputType>(i)));
        }
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      const TInputType* data = decodedRaw_.data<TInputType>();
      LongDecimalWithOverflowState accumulator;
      rows.applyToSelected([&](vector_size_t i) {
        accumulator.overflow += DecimalUtil::addWithOverflow(
            accumulator.sum, data[i], accumulator.sum);
      });
      accumulator.count = rows.countSelected();
      char rawData[LongDecimalWithOverflowState::serializedSize()];
      StringView serialized(
          rawData, LongDecimalWithOverflowState::serializedSize());
      accumulator.serialize(serialized);
      mergeAccumulators<false>(group, serialized);
    } else {
      LongDecimalWithOverflowState accumulator;
      rows.applyToSelected([&](vector_size_t i) {
        accumulator.overflow += DecimalUtil::addWithOverflow(
            accumulator.sum,
            decodedRaw_.valueAt<TInputType>(i),
            accumulator.sum);
      });
      accumulator.count = rows.countSelected();
      char rawData[LongDecimalWithOverflowState::serializedSize()];
      StringView serialized(
          rawData, LongDecimalWithOverflowState::serializedSize());
      accumulator.serialize(serialized);
      mergeAccumulators(group, serialized);
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto intermediateFlatVector =
        dynamic_cast<const FlatVector<StringView>*>(decodedPartial_.base());
    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto serializedAccumulator =
            intermediateFlatVector->valueAt(decodedIndex);
        rows.applyToSelected([&](vector_size_t i) {
          clearNull(groups[i]);
          auto accumulator = decimalAccumulator(groups[i]);
          accumulator->mergeWith(serializedAccumulator);
        });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        auto serializedAccumulator =
            intermediateFlatVector->valueAt(decodedIndex);
        auto accumulator = decimalAccumulator(groups[i]);
        accumulator->mergeWith(serializedAccumulator);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        auto serializedAccumulator =
            intermediateFlatVector->valueAt(decodedIndex);
        auto accumulator = decimalAccumulator(groups[i]);
        accumulator->mergeWith(serializedAccumulator);
      });
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto intermediateFlatVector =
        dynamic_cast<const FlatVector<StringView>*>(decodedPartial_.base());

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto serializedAccumulator =
            intermediateFlatVector->valueAt(decodedIndex);
        if (rows.hasSelections()) {
          clearNull(group);
        }
        rows.applyToSelected([&](vector_size_t i) {
          mergeAccumulators(group, serializedAccumulator);
        });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        clearNull(group);
        auto decodedIndex = decodedPartial_.index(i);
        auto serializedAccumulator =
            intermediateFlatVector->valueAt(decodedIndex);
        mergeAccumulators(group, serializedAccumulator);
      });
    } else {
      if (rows.hasSelections()) {
        clearNull(group);
      }
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        auto serializedAccumulator =
            intermediateFlatVector->valueAt(decodedIndex);
        mergeAccumulators(group, serializedAccumulator);
      });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto stringViewVector = (*result)->as<FlatVector<StringView>>();
    stringViewVector->resize(numGroups);
    uint64_t* rawNulls = nullptr;
    rawNulls = getRawNulls(stringViewVector);
    for (auto i = 0; i < numGroups; ++i) {
      auto accumulator = decimalAccumulator(groups[i]);
      if (isNull(groups[i])) {
        stringViewVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto size = accumulator->serializedSize();
        char* rawBuffer = stringViewVector->getRawStringBufferWithSpace(size);
        StringView serialized(rawBuffer, size);
        accumulator->serialize(serialized);
        stringViewVector->setNoCopy(i, serialized);
      }
    }
  }

  virtual TResultType computeFinalValue(
      LongDecimalWithOverflowState* accumulator) {
    return 0;
  };

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<TResultType>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);

    TResultType* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        vector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto accumulator = decimalAccumulator(group);
        rawValues[i] = computeFinalValue(accumulator);
      }
    }
  }

  template <bool tableHasNulls = true>
  void mergeAccumulators(char* group, const StringView& serialized) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = decimalAccumulator(group);
    accumulator->mergeWith(serialized);
  }

  template <bool tableHasNulls = true>
  void updateNonNullValue(char* group, TResultType value) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = decimalAccumulator(group);
    accumulator->overflow +=
        DecimalUtil::addWithOverflow(accumulator->sum, value, accumulator->sum);
    accumulator->count += 1;
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) LongDecimalWithOverflowState();
    }
  }

  inline LongDecimalWithOverflowState* decimalAccumulator(char* group) {
    return exec::Aggregate::value<LongDecimalWithOverflowState>(group);
  }

 private:
  template <typename T>
  inline bool isBitSet(const T* bits, uint64_t idx) {
    return bits[idx / (sizeof(bits[0]) * 8)] &
        (static_cast<T>(1) << (idx & ((sizeof(bits[0]) * 8) - 1)));
  }

  template <typename U>
  constexpr inline U roundUp(U value, U factor) {
    return (value + (factor - 1)) / factor * factor;
  }

  void hashAggUpdateInt128(
      char** groups,
      uint64_t* bitmap1,
      uint64_t* bitmap2,
      const TInputType* value,
      int32_t begin,
      int32_t end,
      int mode1,
      int mode2,
      uint32_t* dic) {
    auto getValue = [&](int32_t idx) -> TInputType {
      return (mode2 == 3) ? value[dic[idx]] : value[idx];
    };

    auto getNullBit = [&](int32_t idx) -> bool {
      if (bitmap2 == nullptr)
        return true;
      switch (mode1) {
        case 0:
          return true;
        case 1:
          return isBitSet(bitmap2, static_cast<uint64_t>(idx));
        case 2:
          return isBitSet(bitmap2, static_cast<uint64_t>(0));
        case 3:
          return isBitSet(bitmap2, static_cast<uint64_t>(dic[idx]));
        default:
          return false;
      }
    };

    auto processRows = [&](const int32_t* rows, int cnt) {
      int i = 0;
      for (; i + 3 < cnt; i += 4) {
        char* g0 = groups[rows[i]];
        char* g1 = groups[rows[i + 1]];
        char* g2 = groups[rows[i + 2]];
        char* g3 = groups[rows[i + 3]];

        exec::Aggregate::clearNull(g0);
        exec::Aggregate::clearNull(g1);
        exec::Aggregate::clearNull(g2);
        exec::Aggregate::clearNull(g3);

        auto* acc0 = decimalAccumulator(g0);
        auto* acc1 = decimalAccumulator(g1);
        auto* acc2 = decimalAccumulator(g2);
        auto* acc3 = decimalAccumulator(g3);

        acc0->overflow += DecimalUtil::addWithOverflow(
            acc0->sum, TResultType(getValue(rows[i])), acc0->sum);
        acc1->overflow += DecimalUtil::addWithOverflow(
            acc1->sum, TResultType(getValue(rows[i + 1])), acc1->sum);
        acc2->overflow += DecimalUtil::addWithOverflow(
            acc2->sum, TResultType(getValue(rows[i + 2])), acc2->sum);
        acc3->overflow += DecimalUtil::addWithOverflow(
            acc3->sum, TResultType(getValue(rows[i + 3])), acc3->sum);

        acc0->count += 1;
        acc1->count += 1;
        acc2->count += 1;
        acc3->count += 1;
      }
      for (; i < cnt; ++i) {
        char* g = groups[rows[i]];
        exec::Aggregate::clearNull(g);
        auto* acc = decimalAccumulator(g);
        acc->overflow += DecimalUtil::addWithOverflow(
            acc->sum, TResultType(getValue(rows[i])), acc->sum);
        acc->count += 1;
      }
    };

    int32_t wordBegin = begin / 64;
    int32_t wordEnd = (end + 63) / 64;

    if (mode1 == 0 || mode1 == 1) {
      for (int32_t w = wordBegin; w < wordEnd; ++w) {
        int32_t rowBase = w * 64;
        uint64_t bits = bitmap1[w];
        if (mode1 == 1 && bitmap2 != nullptr)
          bits &= bitmap2[w];

        if (rowBase < begin)
          bits &= ~((1ULL << (begin - rowBase)) - 1);
        if (rowBase + 64 > end) {
          int shift = end - rowBase;
          if (shift < 64)
            bits &= (1ULL << shift) - 1;
        }
        if (bits == 0)
          continue;

        int32_t rows[64];
        int cnt = 0;
        uint64_t tmp = bits;
        while (tmp != 0) {
          rows[cnt++] = rowBase + __builtin_ctzll(tmp);
          tmp &= tmp - 1;
        }

        processRows(rows, cnt);
      }
    } else if (mode1 == 2) {
      if (!getNullBit(0))
        return;

      for (int32_t w = wordBegin; w < wordEnd; ++w) {
        int32_t rowBase = w * 64;
        uint64_t bits = bitmap1[w];

        if (rowBase < begin)
          bits &= ~((1ULL << (begin - rowBase)) - 1);
        if (rowBase + 64 > end) {
          int shift = end - rowBase;
          if (shift < 64)
            bits &= (1ULL << shift) - 1;
        }
        if (bits == 0)
          continue;

        int32_t rows[64];
        int cnt = 0;
        uint64_t tmp = bits;
        while (tmp != 0) {
          rows[cnt++] = rowBase + __builtin_ctzll(tmp);
          tmp &= tmp - 1;
        }

        processRows(rows, cnt);
      }
    } else {
      for (int32_t w = wordBegin; w < wordEnd; ++w) {
        int32_t rowBase = w * 64;
        uint64_t bits = bitmap1[w];

        if (rowBase < begin)
          bits &= ~((1ULL << (begin - rowBase)) - 1);
        if (rowBase + 64 > end) {
          int shift = end - rowBase;
          if (shift < 64)
            bits &= (1ULL << shift) - 1;
        }
        if (bits == 0)
          continue;

        int32_t rows[64];
        int cnt = 0;
        uint64_t tmp = bits;
        while (tmp != 0) {
          int32_t row = rowBase + __builtin_ctzll(tmp);
          if (getNullBit(row))
            rows[cnt++] = row;
          tmp &= tmp - 1;
        }

        processRows(rows, cnt);
      }
    }
  }

  DecodedVector decodedRaw_;
  DecodedVector decodedPartial_;
};

} // namespace facebook::velox::functions::aggregate

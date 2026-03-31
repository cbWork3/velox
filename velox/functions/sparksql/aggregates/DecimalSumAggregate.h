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

/// Optimized version that overrides addRawInput with word-level bitmap
/// scanning + ctz extraction + 4x unrolled accumulation, bypassing the
/// generic SimpleAggregateAdapter per-row iteration.
template <typename TInputType, typename TSumType, uint8_t ResultPrecision>
class OptimizedSparkDecimalSumAggregate
    : public exec::SimpleAggregateAdapter<
          DecimalSumAggregate<TInputType, TSumType, ResultPrecision>> {
  using Base = exec::SimpleAggregateAdapter<
      DecimalSumAggregate<TInputType, TSumType, ResultPrecision>>;
  using AccumulatorType = typename DecimalSumAggregate<
      TInputType, TSumType, ResultPrecision>::AccumulatorType;

 public:
  explicit OptimizedSparkDecimalSumAggregate(TypePtr resultType)
      : Base(std::move(resultType)) {}

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

    const auto offset = this->offset_;

    auto processRows = [&](const int32_t* rows, int cnt) {
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

        int32_t extractedRows[64];
        int cnt = 0;
        uint64_t tmp = bits;
        while (tmp != 0) {
          extractedRows[cnt++] = rowBase + __builtin_ctzll(tmp);
          tmp &= tmp - 1;
        }

        processRows(extractedRows, cnt);
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

        int32_t extractedRows[64];
        int cnt = 0;
        uint64_t tmp = bits;
        while (tmp != 0) {
          extractedRows[cnt++] = rowBase + __builtin_ctzll(tmp);
          tmp &= tmp - 1;
        }

        processRows(extractedRows, cnt);
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

        int32_t extractedRows[64];
        int cnt = 0;
        uint64_t tmp = bits;
        while (tmp != 0) {
          int32_t row = rowBase + __builtin_ctzll(tmp);
          if (getNullBit(row))
            extractedRows[cnt++] = row;
          tmp &= tmp - 1;
        }

        processRows(extractedRows, cnt);
      }
    }
  }

  DecodedVector decodedInput_;
};

} // namespace facebook::velox::functions::aggregate::sparksql

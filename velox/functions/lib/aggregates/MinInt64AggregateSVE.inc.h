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

// SVE hash-aggregation update for min(bigint). Included from
// MinMaxAggregateBase.cpp under #if defined(__aarch64__) inside anonymous
// namespace. Mirrors SumAggregateBase.h SVE layout; combines with svmins_m.
// DecodedVector / LazyVector come from SimpleNumericAggregate.h in the .cpp.

class SimpleNumericMinAggregateInt64SVE : public SimpleNumericMinMaxAggregate<int64_t> {
  using BaseMinMax = SimpleNumericMinMaxAggregate<int64_t>;
  using BaseAggregate = SimpleNumericAggregate<int64_t, int64_t, int64_t>;

 public:
  explicit SimpleNumericMinAggregateInt64SVE(
      TypePtr resultType,
      TimestampPrecision precision = TimestampPrecision::kMilliseconds)
      : BaseMinMax(resultType, precision) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    if constexpr (BaseAggregate::template kMayPushdown<int64_t>) {
      if (!args[0]->type()->isDecimal()) {
        if (mayPushdown && args[0]->isLazy()) {
          BaseAggregate::template pushdown<
              velox::aggregate::MinMaxHook<int64_t, true>>(
              groups, rows, args[0]);
          return;
        }
      } else {
        mayPushdown = false;
      }
    } else {
      mayPushdown = false;
    }

    if (exec::Aggregate::numNulls_) {
      DecodedVector decoded(*args[0], rows, !mayPushdown);
      if (decoded.mayHaveNulls()) {
        updateGroupsWithDecoded(groups, rows, args[0], mayPushdown, decoded);
      } else {
        BaseAggregate::template updateGroups<true, int64_t>(
            groups, rows, args[0], updateGroup, mayPushdown);
      }
    } else {
      BaseAggregate::template updateGroups<false, int64_t>(
          groups, rows, args[0], updateGroup, mayPushdown);
    }
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
        [](int64_t& result, int64_t value, int /* unused */) { result = value; },
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
  static inline void updateGroup(int64_t& result, int64_t value) {
    if (result > value) {
      result = value;
    }
  }

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      *exec::Aggregate::value<int64_t>(groups[i]) = kInitialValue_;
    }
  }

 private:
  static const int64_t kInitialValue_;

  template <typename T>
  static inline bool isBitSet(const T* bits, uint64_t idx) {
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

  // Hash buckets may map multiple rows to the same group pointer in one SVE
  // chunk; getUinqMask + scalar min matches SumAggregateBase
  // hashAggUpdateSVEWithCharForNormal.
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
    svbool_t mask, mask1, mask2;
    const int64_t off = this->getOffsetFromAgg();

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
                minAssignScalarAt(
                    result[count + i], value[count + i], off);
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

  void updateGroupsWithDecoded(
      char** groups,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      bool mayPushdown,
      DecodedVector& decoded) {
    if constexpr (BaseAggregate::template kMayPushdown<int64_t>) {
      auto encoding = decoded.base()->encoding();
      if (encoding == VectorEncoding::Simple::LAZY &&
          !arg->type()->isDecimal()) {
        velox::aggregate::SimpleCallableHook<int64_t, void (*)(int64_t&, int64_t)>
            hook(
                exec::Aggregate::offset_,
                exec::Aggregate::nullByte_,
                exec::Aggregate::nullMask_,
                groups,
                &this->exec::Aggregate::numNulls_,
                &SimpleNumericMinAggregateInt64SVE::updateGroup);

        auto indices = decoded.indices();
        decoded.base()->as<const LazyVector>()->load(
            RowSet(indices, arg->size()), &hook);
        return;
      }
    }

    uint64_t* bitmask1 = rows.getBits();
    uint64_t* bitmask2 = decoded.getNulls();
    int64_t* value = reinterpret_cast<int64_t*>(decoded.getData());
    vector_size_t begin = rows.getBegin();
    vector_size_t end = rows.getEnd();
    int mode1 = decoded.getMode1();
    int mode2 = decoded.getmode2();
    vector_size_t* dic = decoded.getDic();

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
};

const int64_t SimpleNumericMinAggregateInt64SVE::kInitialValue_ =
    MinMaxTrait<int64_t>::max();

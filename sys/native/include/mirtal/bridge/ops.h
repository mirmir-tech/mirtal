#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace mirtal {

struct Array;
struct Arrays;
struct Stream;

std::shared_ptr<Array> subtract(const Array&, const Array&, const Stream&);
std::shared_ptr<Array> negative(const Array&, const Stream&);
std::shared_ptr<Array> exp(const Array&, const Stream&);
std::shared_ptr<Array> erf(const Array&, const Stream&);
std::shared_ptr<Array> cos(const Array&, const Stream&);
std::shared_ptr<Array> sin(const Array&, const Stream&);
std::shared_ptr<Array> reciprocal(const Array&, const Stream&);
std::shared_ptr<Array> minimum(const Array&, const Array&, const Stream&);
std::shared_ptr<Array> maximum(const Array&, const Array&, const Stream&);
std::shared_ptr<Array> power(const Array&, const Array&, const Stream&);
std::shared_ptr<Array> floor_divide(const Array&, const Array&, const Stream&);
std::shared_ptr<Array> less(const Array&, const Array&, const Stream&);
std::shared_ptr<Array> greater_equal(const Array&, const Array&, const Stream&);
std::shared_ptr<Array> logical_and(const Array&, const Array&, const Stream&);
std::shared_ptr<Array> clip(const Array&, const Array&, const Array&, const Stream&);
std::shared_ptr<Array> matmul(const Array&, const Array&, const Stream&);
std::shared_ptr<Array> layer_norm(
    const Array&, const Array&, const Array&, float, const Stream&);
std::shared_ptr<Array> arange(float, float, float, std::uint8_t, const Stream&);
std::shared_ptr<Array> full(
    rust::Slice<const std::int32_t>, float, std::uint8_t, const Stream&);
std::shared_ptr<Array> concatenate(const Arrays&, std::int32_t, const Stream&);
std::shared_ptr<Array> stack(const Arrays&, std::int32_t, const Stream&);
std::shared_ptr<Array> repeat(const Array&, std::int32_t, std::int32_t, const Stream&);
std::shared_ptr<Array> conv1d(
    const Array&,
    const Array&,
    std::int32_t,
    std::int32_t,
    std::int32_t,
    std::int32_t,
    const Stream&);
std::shared_ptr<Array> slice(
    const Array&, rust::Slice<const std::int32_t>, rust::Slice<const std::int32_t>, const Stream&);
std::shared_ptr<Array> slice_update(
    const Array&,
    const Array&,
    rust::Slice<const std::int32_t>,
    rust::Slice<const std::int32_t>,
    const Stream&);
std::shared_ptr<Array> depends(const Array&, const Arrays&, const Stream&);
std::shared_ptr<Array> argpartition(const Array&, std::int32_t, std::int32_t, const Stream&);
std::shared_ptr<Array> argsort(const Array&, std::int32_t, const Stream&);
std::shared_ptr<Array> take(const Array&, const Array&, std::int32_t, const Stream&);
std::shared_ptr<Array> take_along_axis(
    const Array&, const Array&, std::int32_t, const Stream&);
std::shared_ptr<Array> softmax(const Array&, std::int32_t, bool, const Stream&);
std::shared_ptr<Array> logsumexp(const Array&, std::int32_t, bool, const Stream&);
std::shared_ptr<Array> cumulative_sum(
    const Array&, std::int32_t, bool, bool, const Stream&);
std::shared_ptr<Array> reduce_max(const Array&, std::int32_t, bool, const Stream&);
std::shared_ptr<Array> reduce_sum(const Array&, std::int32_t, bool, const Stream&);
std::shared_ptr<Array> argmax_axis(const Array&, std::int32_t, bool, const Stream&);

}  // namespace mirtal

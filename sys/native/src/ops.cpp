#include "mirtal/bridge.h"

#include <stdexcept>
#include <utility>

namespace mirtal {
namespace {
std::shared_ptr<Array> wrap_op(mx::array value) {
  return std::make_shared<Array>(std::move(value));
}

mx::Dtype op_dtype(std::uint8_t value) {
  switch (value) {
    case 0: return mx::bool_;
    case 1: return mx::uint32;
    case 2: return mx::int32;
    case 3: return mx::float16;
    case 4: return mx::bfloat16;
    case 5: return mx::float32;
    default: throw std::invalid_argument("unsupported operation dtype");
  }
}
}  // namespace

std::shared_ptr<Array> subtract(const Array& a, const Array& b, const Stream& s) {
  return wrap_op(mx::subtract(a.value, b.value, s.value));
}
std::shared_ptr<Array> negative(const Array& x, const Stream& s) {
  return wrap_op(mx::negative(x.value, s.value));
}
std::shared_ptr<Array> exp(const Array& x, const Stream& s) {
  return wrap_op(mx::exp(x.value, s.value));
}
std::shared_ptr<Array> erf(const Array& x, const Stream& s) {
  return wrap_op(mx::erf(x.value, s.value));
}
std::shared_ptr<Array> cos(const Array& x, const Stream& s) {
  return wrap_op(mx::cos(x.value, s.value));
}
std::shared_ptr<Array> sin(const Array& x, const Stream& s) {
  return wrap_op(mx::sin(x.value, s.value));
}
std::shared_ptr<Array> reciprocal(const Array& x, const Stream& s) {
  return wrap_op(mx::reciprocal(x.value, s.value));
}
std::shared_ptr<Array> minimum(const Array& a, const Array& b, const Stream& s) {
  return wrap_op(mx::minimum(a.value, b.value, s.value));
}
std::shared_ptr<Array> maximum(const Array& a, const Array& b, const Stream& s) {
  return wrap_op(mx::maximum(a.value, b.value, s.value));
}
std::shared_ptr<Array> power(const Array& a, const Array& b, const Stream& s) {
  return wrap_op(mx::power(a.value, b.value, s.value));
}
std::shared_ptr<Array> floor_divide(const Array& a, const Array& b, const Stream& s) {
  return wrap_op(mx::floor_divide(a.value, b.value, s.value));
}
std::shared_ptr<Array> less(const Array& a, const Array& b, const Stream& s) {
  return wrap_op(mx::less(a.value, b.value, s.value));
}
std::shared_ptr<Array> greater_equal(const Array& a, const Array& b, const Stream& s) {
  return wrap_op(mx::greater_equal(a.value, b.value, s.value));
}
std::shared_ptr<Array> logical_and(const Array& a, const Array& b, const Stream& s) {
  return wrap_op(mx::logical_and(a.value, b.value, s.value));
}
std::shared_ptr<Array> clip(
    const Array& input, const Array& minimum, const Array& maximum, const Stream& s) {
  return wrap_op(mx::clip(
      input.value,
      std::optional<mx::array>{minimum.value},
      std::optional<mx::array>{maximum.value},
      s.value));
}
std::shared_ptr<Array> matmul(const Array& a, const Array& b, const Stream& s) {
  return wrap_op(mx::matmul(a.value, b.value, s.value));
}
std::shared_ptr<Array> layer_norm(
    const Array& input,
    const Array& weight,
    const Array& bias,
    float eps,
    const Stream& s) {
  return wrap_op(mx::fast::layer_norm(
      input.value,
      std::optional<mx::array>{weight.value},
      std::optional<mx::array>{bias.value},
      eps,
      s.value));
}
std::shared_ptr<Array> arange(
    float start, float stop, float step, std::uint8_t dtype, const Stream& s) {
  return wrap_op(mx::arange(start, stop, step, op_dtype(dtype), s.value));
}
std::shared_ptr<Array> full(
    rust::Slice<const std::int32_t> shape,
    float value,
    std::uint8_t dtype,
    const Stream& s) {
  return wrap_op(
      mx::full(mx::Shape(shape.begin(), shape.end()), value, op_dtype(dtype), s.value));
}
std::shared_ptr<Array> concatenate(const Arrays& inputs, std::int32_t axis, const Stream& s) {
  return wrap_op(mx::concatenate(inputs.values, axis, s.value));
}
std::shared_ptr<Array> stack(const Arrays& inputs, std::int32_t axis, const Stream& s) {
  return wrap_op(mx::stack(inputs.values, axis, s.value));
}
std::shared_ptr<Array> repeat(
    const Array& input, std::int32_t repeats, std::int32_t axis, const Stream& s) {
  return wrap_op(mx::repeat(input.value, repeats, axis, s.value));
}
std::shared_ptr<Array> conv1d(
    const Array& input,
    const Array& weight,
    std::int32_t stride,
    std::int32_t padding,
    std::int32_t dilation,
    std::int32_t groups,
    const Stream& s) {
  return wrap_op(mx::conv1d(
      input.value, weight.value, stride, padding, dilation, groups, s.value));
}
std::shared_ptr<Array> slice(
    const Array& x,
    rust::Slice<const std::int32_t> start,
    rust::Slice<const std::int32_t> stop,
    const Stream& s) {
  return wrap_op(mx::slice(
      x.value,
      mx::Shape(start.begin(), start.end()),
      mx::Shape(stop.begin(), stop.end()),
      s.value));
}
std::shared_ptr<Array> slice_update(
    const Array& input,
    const Array& update,
    rust::Slice<const std::int32_t> start,
    rust::Slice<const std::int32_t> stop,
    const Stream& s) {
  return wrap_op(mx::slice_update(
      input.value,
      update.value,
      mx::Shape(start.begin(), start.end()),
      mx::Shape(stop.begin(), stop.end()),
      s.value));
}
std::shared_ptr<Array> depends(const Array& input, const Arrays& dependencies, const Stream&) {
  return wrap_op(mx::depends({input.value}, dependencies.values).front());
}
std::shared_ptr<Array> argpartition(
    const Array& x, std::int32_t kth, std::int32_t axis, const Stream& s) {
  return wrap_op(mx::argpartition(x.value, kth, axis, s.value));
}
std::shared_ptr<Array> argsort(const Array& x, std::int32_t axis, const Stream& s) {
  return wrap_op(mx::argsort(x.value, axis, s.value));
}
std::shared_ptr<Array> take(
    const Array& x, const Array& indices, std::int32_t axis, const Stream& s) {
  return wrap_op(mx::take(x.value, indices.value, axis, s.value));
}
std::shared_ptr<Array> take_along_axis(
    const Array& x, const Array& indices, std::int32_t axis, const Stream& s) {
  return wrap_op(mx::take_along_axis(x.value, indices.value, axis, s.value));
}
std::shared_ptr<Array> softmax(
    const Array& x, std::int32_t axis, bool precise, const Stream& s) {
  return wrap_op(mx::softmax(x.value, axis, precise, s.value));
}
std::shared_ptr<Array> logsumexp(
    const Array& x, std::int32_t axis, bool keepdims, const Stream& s) {
  return wrap_op(mx::logsumexp(x.value, axis, keepdims, s.value));
}
std::shared_ptr<Array> cumulative_sum(
    const Array& x,
    std::int32_t axis,
    bool reverse,
    bool inclusive,
    const Stream& s) {
  return wrap_op(mx::cumsum(x.value, axis, reverse, inclusive, s.value));
}
std::shared_ptr<Array> reduce_max(
    const Array& x, std::int32_t axis, bool keepdims, const Stream& s) {
  return wrap_op(mx::max(x.value, axis, keepdims, s.value));
}
std::shared_ptr<Array> reduce_sum(
    const Array& x, std::int32_t axis, bool keepdims, const Stream& s) {
  return wrap_op(mx::sum(x.value, axis, keepdims, s.value));
}
std::shared_ptr<Array> argmax_axis(
    const Array& x, std::int32_t axis, bool keepdims, const Stream& s) {
  return wrap_op(mx::argmax(x.value, axis, keepdims, s.value));
}
}  // namespace mirtal

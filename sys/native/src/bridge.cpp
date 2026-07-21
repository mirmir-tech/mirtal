#include "mirtal/bridge.h"

#include "mlx/memory.h"

#include <algorithm>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mirtal {
namespace {
std::atomic<std::uint64_t> next_stream_id{1};

mx::Shape checked_shape(
    std::size_t elements,
    rust::Slice<const std::int32_t> shape) {
  std::size_t expected = 1;
  for (auto dimension : shape) {
    if (dimension < 0) throw std::runtime_error("shape contains a negative dimension");
    expected *= static_cast<std::size_t>(dimension);
  }
  if (expected != elements) throw std::runtime_error("shape does not match data length");
  return mx::Shape(shape.begin(), shape.end());
}

mx::Dtype dtype(std::uint8_t value) {
  switch (value) {
    case 0: return mx::bool_;
    case 1: return mx::uint32;
    case 2: return mx::int32;
    case 3: return mx::float16;
    case 4: return mx::bfloat16;
    case 5: return mx::float32;
    case 6: return mx::uint8;
    default: throw std::runtime_error("unsupported MLX dtype");
  }
}

std::uint8_t dtype(const mx::Dtype& value) {
  if (value == mx::bool_) return 0;
  if (value == mx::uint32) return 1;
  if (value == mx::int32) return 2;
  if (value == mx::float16) return 3;
  if (value == mx::bfloat16) return 4;
  if (value == mx::float32) return 5;
  if (value == mx::uint8) return 6;
  return 255;
}

std::size_t recommended_working_set() {
  const auto& info = mx::device_info(mx::Device(mx::Device::gpu, 0));
  auto value = info.find("max_recommended_working_set_size");
  if (value == info.end()) return 0;
  auto bytes = std::get_if<std::size_t>(&value->second);
  return bytes == nullptr ? 0 : *bytes;
}

template <typename T>
void copy(const Array& array, const Stream& stream, rust::Slice<T> output, mx::Dtype target) {
  if (array.value.size() != output.size()) {
    throw std::runtime_error("array output length does not match");
  }
  auto converted = mx::astype(array.value, target, stream.value);
  auto evaluated = mx::contiguous(converted, false, stream.value);
  evaluated.eval();
  std::copy_n(evaluated.data<T>(), output.size(), output.data());
}

std::shared_ptr<Array> wrap(mx::array value) {
  return std::make_shared<Array>(std::move(value));
}
}  // namespace

rust::String version() { return rust::String(mx::version()); }
void clear_memory_cache() { mx::clear_cache(); }
bool configure_recommended_wired_limit() {
  auto bytes = recommended_working_set();
  if (bytes == 0) return false;
  static_cast<void>(mx::set_wired_limit(bytes));
  return true;
}
std::size_t active_memory() { return mx::get_active_memory(); }
std::size_t cache_memory() { return mx::get_cache_memory(); }
std::size_t peak_memory() { return mx::get_peak_memory(); }
std::size_t memory_limit() { return mx::get_memory_limit(); }
std::size_t recommended_memory() { return recommended_working_set(); }

std::unique_ptr<Stream> new_stream(std::uint8_t kind, std::int32_t index) {
  auto type = kind == 0 ? mx::Device::cpu : mx::Device::gpu;
  if (kind > 1 || index < 0) throw std::runtime_error("invalid MLX device");
  auto value = mx::new_stream(mx::Device(type, index));
  return std::make_unique<Stream>(value, next_stream_id.fetch_add(1));
}
std::size_t stream_native_value(const Stream& stream) noexcept {
  return reinterpret_cast<std::size_t>(&stream.value);
}
std::uint64_t stream_id(const Stream& stream) noexcept { return stream.id; }
void synchronize(const Stream& stream) { mx::synchronize(stream.value); }

std::shared_ptr<Array> array_from_f32(
    rust::Slice<const float> data,
    rust::Slice<const std::int32_t> shape) {
  return wrap(mx::array(data.data(), checked_shape(data.size(), shape), mx::float32));
}
std::shared_ptr<Array> array_from_u32(
    rust::Slice<const std::uint32_t> data,
    rust::Slice<const std::int32_t> shape) {
  return wrap(mx::array(data.data(), checked_shape(data.size(), shape), mx::uint32));
}
std::shared_ptr<Array> array_from_owned_native_handle(std::size_t address) {
  if (address == 0) throw std::runtime_error("owned native MLX array handle is null");
  return std::shared_ptr<Array>(reinterpret_cast<Array*>(address));
}
std::size_t array_native_handle(const Array& array) noexcept {
  return reinterpret_cast<std::size_t>(&array);
}
rust::Vec<std::int32_t> array_shape(const Array& array) {
  rust::Vec<std::int32_t> output;
  output.reserve(array.value.ndim());
  for (auto dimension : array.value.shape()) output.push_back(dimension);
  return output;
}
std::uint8_t array_dtype(const Array& array) { return dtype(array.value.dtype()); }
std::size_t array_len(const Array& array) noexcept { return array.value.size(); }
void array_eval(const Array& array) { mx::async_eval(array.value); }
void array_copy_f32(const Array& array, const Stream& stream, rust::Slice<float> output) {
  copy(array, stream, output, mx::float32);
}
void array_copy_u32(
    const Array& array,
    const Stream& stream,
    rust::Slice<std::uint32_t> output) {
  copy(array, stream, output, mx::uint32);
}
std::uint32_t item_u32(const Array& input, const Stream& stream) {
  if (input.value.dtype() != mx::uint32 || input.value.size() != 1) {
    throw std::runtime_error("expected a single uint32 MLX array value");
  }
  static_cast<void>(stream);
  mx::eval(input.value);
  return input.value.data<std::uint32_t>()[0];
}

std::shared_ptr<Array> add(const Array& left, const Array& right, const Stream& stream) {
  return wrap(mx::add(left.value, right.value, stream.value));
}
std::shared_ptr<Array> add_scalar(const Array& input, float value, const Stream& stream) {
  return wrap(mx::add(input.value, mx::array(value, input.value.dtype()), stream.value));
}
std::shared_ptr<Array> multiply(
    const Array& left,
    const Array& right,
    const Stream& stream) {
  return wrap(mx::multiply(left.value, right.value, stream.value));
}
std::shared_ptr<Array> multiply_scalar(
    const Array& input,
    float value,
    const Stream& stream) {
  return wrap(mx::multiply(input.value, mx::array(value, input.value.dtype()), stream.value));
}
std::shared_ptr<Array> divide(
    const Array& left,
    const Array& right,
    const Stream& stream) {
  return wrap(mx::divide(left.value, right.value, stream.value));
}
std::shared_ptr<Array> power_scalar(
    const Array& input,
    float exponent,
    const Stream& stream) {
  return wrap(mx::power(
      input.value, mx::array(exponent, input.value.dtype()), stream.value));
}
std::shared_ptr<Array> rms_norm(
    const Array& input,
    const Array& weight,
    float eps,
    const Stream& stream) {
  return wrap(mx::fast::rms_norm(
      input.value, std::optional<mx::array>{weight.value}, eps, stream.value));
}
std::shared_ptr<Array> rms_norm_unit(
    const Array& input,
    float eps,
    const Stream& stream) {
  return wrap(mx::fast::rms_norm(input.value, std::nullopt, eps, stream.value));
}
std::shared_ptr<Array> astype(
    const Array& input,
    std::uint8_t target,
    const Stream& stream) {
  return input.value.dtype() == dtype(target)
      ? wrap(input.value)
      : wrap(mx::astype(input.value, dtype(target), stream.value));
}
std::shared_ptr<Array> reshape(
    const Array& input,
    rust::Slice<const std::int32_t> shape,
    const Stream& stream) {
  return wrap(mx::reshape(input.value, mx::Shape(shape.begin(), shape.end()), stream.value));
}
std::shared_ptr<Array> transpose(
    const Array& input,
    rust::Slice<const std::int32_t> axes,
    const Stream& stream) {
  return wrap(mx::transpose(input.value, std::vector<int>(axes.begin(), axes.end()), stream.value));
}
std::shared_ptr<Array> expand_dims(
    const Array& input,
    rust::Slice<const std::int32_t> axes,
    const Stream& stream) {
  return wrap(mx::expand_dims(
      input.value, std::vector<int>(axes.begin(), axes.end()), stream.value));
}
std::shared_ptr<Array> squeeze_axis(
    const Array& input,
    std::int32_t axis,
    const Stream& stream) {
  return wrap(mx::squeeze(input.value, axis, stream.value));
}
std::shared_ptr<Array> sigmoid(const Array& input, const Stream& stream) {
  return wrap(mx::sigmoid(input.value, stream.value));
}
std::shared_ptr<Array> sigmoid_multiply(
    const Array& gate,
    const Array& input,
    const Stream& stream) {
  auto activated = mx::sigmoid(gate.value, stream.value);
  return wrap(mx::multiply(activated, input.value, stream.value));
}
std::shared_ptr<Array> silu(const Array& input, const Stream& stream) {
  auto gate = mx::sigmoid(input.value, stream.value);
  return wrap(mx::multiply(input.value, gate, stream.value));
}
std::shared_ptr<Array> tanh(const Array& input, const Stream& stream) {
  return wrap(mx::tanh(input.value, stream.value));
}
}  // namespace mirtal

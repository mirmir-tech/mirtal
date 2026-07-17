#include "mirtal-sys/src/lib.rs.h"

#include <optional>
#include <stdexcept>
#include <utility>

namespace mirtal {

std::unique_ptr<Arrays> quantize(
    const Array& input,
    std::int32_t group_size,
    std::int32_t bits,
    const Stream& stream) {
  auto values = mx::quantize(
      input.value, group_size, bits, "affine", std::nullopt, stream.value);
  if (values.size() != 3) {
    throw std::runtime_error("affine quantize did not return weight/scales/biases");
  }
  return std::make_unique<Arrays>(std::move(values), &stream);
}

std::shared_ptr<Array> quantized_matmul(
    const Array& input,
    const Array& weight,
    const Array& scales,
    const Array& biases,
    const QuantizationOptions& options,
    const Stream& stream) {
  return std::make_shared<Array>(mx::quantized_matmul(
      input.value,
      weight.value,
      scales.value,
      biases.value,
      options.transpose,
      options.group_size,
      options.bits,
      "affine",
      stream.value));
}

std::shared_ptr<Array> gather_qmm(
    const Array& input,
    const Array& weight,
    const Array& scales,
    const Array& biases,
    const Array& rhs_indices,
    const QuantizationOptions& options,
    const Stream& stream) {
  return std::make_shared<Array>(mx::gather_qmm(
      input.value,
      weight.value,
      scales.value,
      biases.value,
      std::nullopt,
      rhs_indices.value,
      options.transpose,
      options.group_size,
      options.bits,
      "affine",
      options.sorted_indices,
      stream.value));
}

std::shared_ptr<Array> dequantize(
    const Array& weight,
    const Array& scales,
    const Array& biases,
    const QuantizationOptions& options,
    const Stream& stream) {
  return std::make_shared<Array>(mx::dequantize(
      weight.value,
      scales.value,
      std::optional<mx::array>{biases.value},
      options.group_size,
      options.bits,
      "affine",
      std::nullopt,
      std::nullopt,
      stream.value));
}

}  // namespace mirtal

#include "mirtal-sys/src/quantized.rs.h"
#include "mlx/version.h"

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

std::unique_ptr<Arrays> quantize_mxfp8(
    const Array& input,
    const Stream& stream) {
  auto values = mx::quantize(
      input.value, 32, 8, "mxfp8", std::nullopt, stream.value);
  if (values.size() != 2) {
    throw std::runtime_error("MXFP8 quantize did not return weight/scales");
  }
  return std::make_unique<Arrays>(std::move(values), &stream);
}

std::unique_ptr<Arrays> quantize_mxfp4(
    const Array& input,
    const Stream& stream) {
  auto values = mx::quantize(
      input.value, 32, 4, "mxfp4", std::nullopt, stream.value);
  if (values.size() != 2) {
    throw std::runtime_error("MXFP4 quantize did not return weight/scales");
  }
  return std::make_unique<Arrays>(std::move(values), &stream);
}

std::shared_ptr<Array> mxfp4_matmul(
    const Array& input,
    const Array& weight,
    const Array& scales,
    bool transpose,
    const Stream& stream) {
  return std::make_shared<Array>(mx::quantized_matmul(
      input.value,
      weight.value,
      scales.value,
      std::nullopt,
      transpose,
      32,
      4,
      "mxfp4",
      stream.value));
}

std::shared_ptr<Array> gather_mxfp4(
    const Array& input,
    const Array& weight,
    const Array& scales,
    const NativeGatherIndices& indices,
    bool transpose,
    bool sorted_indices,
    const Stream& stream) {
  return std::make_shared<Array>(mx::gather_qmm(
      input.value,
      weight.value,
      scales.value,
      std::nullopt,
      indices.lhs ? std::optional<mx::array>{indices.lhs->value} : std::nullopt,
      indices.rhs ? std::optional<mx::array>{indices.rhs->value} : std::nullopt,
      transpose,
      32,
      4,
      "mxfp4",
#if MLX_VERSION_NUMERIC >= 32003
      std::nullopt,
#endif
      sorted_indices,
      stream.value));
}

std::shared_ptr<Array> mxfp8_matmul(
    const Array& input,
    const Array& weight,
    const Array& scales,
    bool transpose,
    const Stream& stream) {
  return std::make_shared<Array>(mx::quantized_matmul(
      input.value,
      weight.value,
      scales.value,
      std::nullopt,
      transpose,
      32,
      8,
      "mxfp8",
      stream.value));
}

std::shared_ptr<Array> dequantize_mxfp8(
    const Array& weight,
    const Array& scales,
    const Stream& stream) {
  return std::make_shared<Array>(mx::dequantize(
      weight.value,
      scales.value,
      std::nullopt,
      32,
      8,
      "mxfp8",
      std::nullopt,
      std::nullopt,
      stream.value));
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
    const NativeGatherIndices& indices,
    const QuantizationOptions& options,
    const Stream& stream) {
  return std::make_shared<Array>(mx::gather_qmm(
      input.value,
      weight.value,
      scales.value,
      biases.value,
      indices.lhs ? std::optional<mx::array>{indices.lhs->value} : std::nullopt,
      indices.rhs ? std::optional<mx::array>{indices.rhs->value} : std::nullopt,
      options.transpose,
      options.group_size,
      options.bits,
      "affine",
#if MLX_VERSION_NUMERIC >= 32003
      std::nullopt,
#endif
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

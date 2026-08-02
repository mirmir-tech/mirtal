#pragma once

#include <cstdint>
#include <memory>

namespace mirtal {

struct Array;
struct Arrays;
struct QuantizationOptions;
struct Stream;

std::unique_ptr<Arrays> quantize(
    const Array& input,
    std::int32_t group_size,
    std::int32_t bits,
    const Stream& stream);
std::unique_ptr<Arrays> quantize_mxfp8(
    const Array& input,
    const Stream& stream);
std::shared_ptr<Array> mxfp8_matmul(
    const Array& input,
    const Array& weight,
    const Array& scales,
    bool transpose,
    const Stream& stream);
std::shared_ptr<Array> dequantize_mxfp8(
    const Array& weight,
    const Array& scales,
    const Stream& stream);
std::shared_ptr<Array> quantized_matmul(
    const Array& input,
    const Array& weight,
    const Array& scales,
    const Array& biases,
    const QuantizationOptions& options,
    const Stream& stream);
std::shared_ptr<Array> gather_qmm(
    const Array& input,
    const Array& weight,
    const Array& scales,
    const Array& biases,
    const Array& rhs_indices,
    const QuantizationOptions& options,
    const Stream& stream);
std::shared_ptr<Array> dequantize(
    const Array& weight,
    const Array& scales,
    const Array& biases,
    const QuantizationOptions& options,
    const Stream& stream);

}  // namespace mirtal

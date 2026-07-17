#pragma once

#include "mlx/mlx.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mirtal {

namespace mx = mlx::core;

struct Array {
  explicit Array(mx::array value) : value(std::move(value)) {}
  mx::array value;
};

struct Stream {
  Stream(mx::Stream value, std::uint64_t id) : value(value), id(id) {}
  mx::Stream value;
  std::uint64_t id;
};

std::vector<mx::array> call_compiled_native(
    std::size_t address,
    const std::vector<mx::array>& inputs);
std::vector<mx::array> call_metal_kernel_native(
    std::size_t address,
    const std::vector<mx::array>& inputs,
    const std::vector<mx::Shape>& output_shapes,
    const std::vector<mx::Dtype>& output_dtypes,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    std::vector<std::pair<std::string, mx::fast::TemplateArg>> templates,
    std::optional<float> init_value,
    bool verbose,
    const mx::Stream& stream);
void* metal_pipeline_native(std::size_t address, const std::string& function);
mx::array sdpa_native(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    float scale,
    bool causal,
    const mx::Stream& stream);
mx::array rope_native(
    const mx::array& input,
    std::int32_t dimensions,
    bool traditional,
    std::optional<float> base,
    float scale,
    std::int32_t offset,
    std::optional<mx::array> frequencies,
    const mx::Stream& stream);

}  // namespace mirtal

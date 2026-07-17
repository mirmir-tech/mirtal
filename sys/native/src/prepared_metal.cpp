#include "mirtal/bridge.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mirtal {
struct PreparedMetal::Impl {
  Impl(const MetalKernel& kernel, const MetalLaunch& launch)
      : kernel(kernel), launch(launch) {
    inputs.reserve(kernel.inputs);
  }

  MetalKernel kernel;
  MetalLaunch launch;
  std::vector<mx::array> inputs;
};

PreparedMetal::PreparedMetal(std::unique_ptr<Impl> implementation)
    : implementation(std::move(implementation)) {}
PreparedMetal::~PreparedMetal() = default;

std::unique_ptr<PreparedMetal> new_prepared_metal(
    const MetalKernel& kernel,
    const MetalLaunch& launch) {
  if (launch.output_shapes.size() != kernel.outputs ||
      launch.output_dtypes.size() != kernel.outputs) {
    throw std::runtime_error("prepared Metal launch output arity is invalid");
  }
  return std::make_unique<PreparedMetal>(
      std::make_unique<PreparedMetal::Impl>(kernel, launch));
}

void prepared_metal_set_input(PreparedMetal& prepared, std::size_t index, const Array& input) {
  auto& implementation = *prepared.implementation;
  auto& values = implementation.inputs;
  if (index > values.size() || index >= implementation.kernel.inputs) {
    throw std::runtime_error("prepared Metal input index is out of bounds");
  }
  if (index == values.size()) {
    values.push_back(input.value);
  } else {
    values[index] = input.value;
  }
}

std::unique_ptr<Arrays> prepared_metal_dispatch(
    PreparedMetal& prepared,
    const Stream& stream) {
  auto& implementation = *prepared.implementation;
  auto& kernel = implementation.kernel;
  auto& launch = implementation.launch;
  if (implementation.inputs.size() != kernel.inputs) {
    throw std::runtime_error("prepared Metal dispatch input arity changed");
  }
  auto outputs = kernel.function(
      implementation.inputs,
      launch.output_shapes,
      launch.output_dtypes,
      launch.grid,
      launch.threadgroup,
      launch.templates,
      launch.init_value,
      launch.verbose,
      stream.value);
  return std::make_unique<Arrays>(std::move(outputs), &stream);
}
}  // namespace mirtal

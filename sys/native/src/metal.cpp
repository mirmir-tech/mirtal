#include "mirtal/bridge.h"
#include "mlx/backend/metal/device.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mirtal {
namespace {
constexpr char kSeparator = '\x1f';

std::string string(rust::Str value) {
  return std::string(value.data(), value.size());
}

std::vector<std::string> names(rust::Str value) {
  std::vector<std::string> output;
  std::stringstream input(string(value));
  std::string name;
  while (std::getline(input, name, kSeparator)) {
    if (!name.empty()) output.push_back(std::move(name));
  }
  return output;
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
    default: throw std::runtime_error("unsupported Metal output dtype");
  }
}
}  // namespace

std::unique_ptr<MetalKernel> new_metal_kernel(
    rust::Str name,
    rust::Str input_names,
    rust::Str output_names,
    rust::Str source,
    rust::Str header,
    bool row_contiguous,
    bool atomic_outputs) {
  auto inputs = names(input_names);
  auto outputs = names(output_names);
  auto function = mx::fast::metal_kernel(
      string(name),
      inputs,
      outputs,
      string(source),
      string(header),
      row_contiguous,
      atomic_outputs);
  return std::make_unique<MetalKernel>(
      std::move(function), inputs.size(), outputs.size());
}

std::size_t metal_kernel_native_handle(const MetalKernel& kernel) noexcept {
  return reinterpret_cast<std::size_t>(&kernel);
}

std::unique_ptr<MetalLibrary> new_metal_library(rust::Str name, rust::Str source) {
  return std::make_unique<MetalLibrary>(string(name), string(source));
}

std::size_t metal_library_native_handle(const MetalLibrary& library) noexcept {
  return reinterpret_cast<std::size_t>(&library);
}

void* metal_pipeline_native(std::size_t address, const std::string& function) {
  if (address == 0) throw std::runtime_error("Metal library handle is null");
  const auto& library = *reinterpret_cast<const MetalLibrary*>(address);
  auto& metal = mx::metal::device(mx::Device(mx::Device::gpu, 0));
  auto* native = metal.get_library(library.name, [&library] { return library.source; });
  return metal.get_kernel(function, native);
}

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
    const mx::Stream& stream) {
  if (address == 0) throw std::runtime_error("Metal kernel handle is null");
  const auto& kernel = *reinterpret_cast<const MetalKernel*>(address);
  if (inputs.size() != kernel.inputs || output_shapes.size() != kernel.outputs ||
      output_dtypes.size() != kernel.outputs) {
    throw std::runtime_error("Metal kernel native call has invalid arity");
  }
  return kernel.function(
      inputs,
      output_shapes,
      output_dtypes,
      grid,
      threadgroup,
      std::move(templates),
      init_value,
      verbose,
      stream);
}

std::unique_ptr<MetalLaunch> new_metal_launch(
    std::int32_t grid_x,
    std::int32_t grid_y,
    std::int32_t grid_z,
    std::int32_t group_x,
    std::int32_t group_y,
    std::int32_t group_z,
    bool verbose) {
  return std::make_unique<MetalLaunch>(MetalLaunch{
      {},
      {},
      {},
      {grid_x, grid_y, grid_z},
      {group_x, group_y, group_z},
      std::nullopt,
      verbose});
}

void metal_launch_add_output(
    MetalLaunch& launch,
    rust::Slice<const std::int32_t> shape,
    std::uint8_t value) {
  launch.output_shapes.emplace_back(shape.begin(), shape.end());
  launch.output_dtypes.push_back(dtype(value));
}

void metal_launch_add_template_int(
    MetalLaunch& launch,
    rust::Str name,
    std::int32_t value) {
  launch.templates.emplace_back(string(name), value);
}

void metal_launch_add_template_bool(MetalLaunch& launch, rust::Str name, bool value) {
  launch.templates.emplace_back(string(name), value);
}

void metal_launch_add_template_dtype(
    MetalLaunch& launch,
    rust::Str name,
    std::uint8_t value) {
  launch.templates.emplace_back(string(name), dtype(value));
}

void metal_launch_set_init(MetalLaunch& launch, float value) noexcept {
  launch.init_value = value;
}

std::unique_ptr<Arrays> metal_dispatch(
    const MetalKernel& kernel,
    const Arrays& inputs,
    const MetalLaunch& launch,
    const Stream& stream) {
  if (inputs.values.size() != kernel.inputs ||
      launch.output_shapes.size() != kernel.outputs) {
    throw std::runtime_error("Metal kernel input or output arity differs from its descriptor");
  }
  auto outputs = kernel.function(
      inputs.values,
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

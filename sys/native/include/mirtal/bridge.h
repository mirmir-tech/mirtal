#pragma once

#include "mirtal/native.h"
#include "rust/cxx.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "mirtal/bridge/attention.h"
#include "mirtal/bridge/ops.h"
#include "mirtal/bridge/aliasing.h"
#include "mirtal/bridge/prepared_metal.h"
#include "mirtal/bridge/quantized.h"
#include "mirtal/bridge/read.h"
#include "mirtal/bridge/rope.h"

namespace mirtal {

struct GraphCallback;
struct QuantizationOptions;

struct Arrays {
  Arrays() = default;
  Arrays(std::vector<mx::array> values, const Stream* stream)
      : values(std::move(values)), stream(stream) {}
  std::vector<mx::array> values;
  const Stream* stream = nullptr;
};

using CompiledFunction =
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

struct Compiled {
  Compiled(Stream stream, CompiledFunction function)
      : stream(std::move(stream)), function(std::move(function)) {}
  Stream stream;
  CompiledFunction function;
};

struct MetalKernel {
  MetalKernel(mx::fast::CustomKernelFunction function, std::size_t inputs, std::size_t outputs)
      : function(std::move(function)), inputs(inputs), outputs(outputs) {}
  mx::fast::CustomKernelFunction function;
  std::size_t inputs;
  std::size_t outputs;
};

struct MetalLibrary {
  MetalLibrary(std::string name, std::string source)
      : name(std::move(name)), source(std::move(source)) {}
  std::string name;
  std::string source;
};

struct TensorMap {
  std::unordered_map<std::string, mx::array> values;
};

struct MetalLaunch {
  std::vector<mx::Shape> output_shapes;
  std::vector<mx::Dtype> output_dtypes;
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> templates;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> threadgroup;
  std::optional<float> init_value;
  bool verbose;
};

rust::String version();
void clear_memory_cache();
bool configure_recommended_wired_limit();
std::size_t active_memory();
std::size_t cache_memory();
std::size_t peak_memory();
std::size_t memory_limit();
std::size_t recommended_memory();

std::unique_ptr<Stream> new_stream(std::uint8_t kind, std::int32_t index);
std::size_t stream_native_value(const Stream& stream) noexcept;
std::uint64_t stream_id(const Stream& stream) noexcept;
void synchronize(const Stream& stream);

std::shared_ptr<Array> array_from_f32(
    rust::Slice<const float> data,
    rust::Slice<const std::int32_t> shape);
std::shared_ptr<Array> array_from_u32(
    rust::Slice<const std::uint32_t> data,
    rust::Slice<const std::int32_t> shape);
std::shared_ptr<Array> array_from_owned_native_handle(std::size_t address);
std::size_t array_native_handle(const Array& array) noexcept;
rust::Vec<std::int32_t> array_shape(const Array& array);
std::uint8_t array_dtype(const Array& array);
std::size_t array_len(const Array& array) noexcept;
void array_eval(const Array& array);
void array_copy_f32(
    const Array& array,
    const Stream& stream,
    rust::Slice<float> output);
void array_copy_u32(
    const Array& array,
    const Stream& stream,
    rust::Slice<std::uint32_t> output);

std::shared_ptr<Array> add(const Array& left, const Array& right, const Stream& stream);
std::shared_ptr<Array> add_scalar(const Array& input, float value, const Stream& stream);
std::shared_ptr<Array> multiply(
    const Array& left,
    const Array& right,
    const Stream& stream);
std::shared_ptr<Array> multiply_scalar(
    const Array& input,
    float value,
    const Stream& stream);
std::shared_ptr<Array> divide(
    const Array& left,
    const Array& right,
    const Stream& stream);
std::shared_ptr<Array> power_scalar(
    const Array& input,
    float exponent,
    const Stream& stream);
std::shared_ptr<Array> rms_norm(
    const Array& input,
    const Array& weight,
    float eps,
    const Stream& stream);
std::shared_ptr<Array> rms_norm_unit(
    const Array& input,
    float eps,
    const Stream& stream);
std::shared_ptr<Array> astype(
    const Array& input,
    std::uint8_t dtype,
    const Stream& stream);
std::shared_ptr<Array> from_fp8(
    const Array& input,
    std::uint8_t dtype,
    const Stream& stream);
std::shared_ptr<Array> to_fp8(const Array& input, const Stream& stream);
std::shared_ptr<Array> view_dtype(
    const Array& input,
    std::uint8_t dtype,
    const Stream& stream);
std::shared_ptr<Array> reshape(
    const Array& input,
    rust::Slice<const std::int32_t> shape,
    const Stream& stream);
std::shared_ptr<Array> transpose(
    const Array& input,
    rust::Slice<const std::int32_t> axes,
    const Stream& stream);
std::shared_ptr<Array> expand_dims(
    const Array& input,
    rust::Slice<const std::int32_t> axes,
    const Stream& stream);
std::shared_ptr<Array> squeeze_axis(
    const Array& input,
    std::int32_t axis,
    const Stream& stream);
std::shared_ptr<Array> sigmoid(const Array& input, const Stream& stream);
std::shared_ptr<Array> sigmoid_multiply(
    const Array& gate,
    const Array& input,
    const Stream& stream);
std::shared_ptr<Array> silu(const Array& input, const Stream& stream);
std::shared_ptr<Array> tanh(const Array& input, const Stream& stream);

std::uint32_t item_u32(const Array& input, const Stream& stream);
std::unique_ptr<Arrays> new_arrays();
void arrays_push(Arrays& arrays, const Array& array);
std::size_t arrays_len(const Arrays& arrays) noexcept;
std::shared_ptr<Array> arrays_get(const Arrays& arrays, std::size_t index);
const Stream& arrays_stream(const Arrays& arrays);
std::unique_ptr<Compiled> new_compiled(
    rust::Box<GraphCallback> callback,
    bool shapeless,
    const Stream& stream);
std::unique_ptr<Arrays> compiled_call(const Compiled& compiled, const Arrays& inputs);
std::size_t compiled_native_handle(const Compiled& compiled) noexcept;
std::unique_ptr<MetalKernel> new_metal_kernel(
    rust::Str name,
    rust::Str input_names,
    rust::Str output_names,
    rust::Str source,
    rust::Str header,
    bool row_contiguous,
    bool atomic_outputs);
std::size_t metal_kernel_native_handle(const MetalKernel& kernel) noexcept;
std::unique_ptr<MetalLibrary> new_metal_library(rust::Str name, rust::Str source);
std::size_t metal_library_native_handle(const MetalLibrary& library) noexcept;
std::unique_ptr<TensorMap> load_safetensors(rust::Str path, const Stream& stream);
std::size_t tensor_map_len(const TensorMap& tensors) noexcept;
void tensor_map_eval(const TensorMap& tensors);
bool tensor_map_contains(const TensorMap& tensors, rust::Str name);
std::shared_ptr<Array> tensor_map_get(const TensorMap& tensors, rust::Str name);
void export_graph_dot(const Array& array, rust::Str path);
std::unique_ptr<MetalLaunch> new_metal_launch(
    std::int32_t grid_x,
    std::int32_t grid_y,
    std::int32_t grid_z,
    std::int32_t group_x,
    std::int32_t group_y,
    std::int32_t group_z,
    bool verbose);
void metal_launch_add_output(
    MetalLaunch& launch,
    rust::Slice<const std::int32_t> shape,
    std::uint8_t dtype);
void metal_launch_add_template_int(MetalLaunch& launch, rust::Str name, std::int32_t value);
void metal_launch_add_template_bool(MetalLaunch& launch, rust::Str name, bool value);
void metal_launch_add_template_dtype(MetalLaunch& launch, rust::Str name, std::uint8_t value);
void metal_launch_set_init(MetalLaunch& launch, float value) noexcept;
std::unique_ptr<Arrays> metal_dispatch(
    const MetalKernel& kernel,
    const Arrays& inputs,
    const MetalLaunch& launch,
    const Stream& stream);

}  // namespace mirtal

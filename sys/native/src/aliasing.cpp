#include "mirtal/bridge.h"

#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mirtal {
namespace {
std::vector<std::uint32_t> vector(rust::Slice<const std::uint32_t> values) {
  return {values.begin(), values.end()};
}

struct AliasingState {
  AliasingState(
      std::size_t input_arity,
      std::vector<std::uint32_t> aliases,
      std::vector<std::uint32_t> stride_inputs,
      std::vector<std::uint32_t> stride_axes,
      std::string function,
      MetalLibrary library)
      : aliases(std::move(aliases)), stride_inputs(std::move(stride_inputs)),
        stride_axes(std::move(stride_axes)), function(std::move(function)),
        library(std::move(library)), output_by_input(input_arity, -1) {
    for (std::size_t output = 0; output < this->aliases.size(); ++output) {
      if (output_by_input.at(this->aliases[output]) >= 0) {
        throw std::runtime_error("aliasing outputs must refer to distinct inputs");
      }
      output_by_input.at(this->aliases[output]) = static_cast<std::int32_t>(output);
    }
    pipeline = static_cast<MTL::ComputePipelineState*>(metal_pipeline_native(
        reinterpret_cast<std::size_t>(&this->library), this->function));
  }

  std::vector<std::uint32_t> aliases, stride_inputs, stride_axes;
  std::string function;
  MetalLibrary library;
  std::vector<std::int32_t> output_by_input;
  MTL::ComputePipelineState* pipeline = nullptr;
};

class AliasingKernel final : public mx::Primitive {
 public:
  AliasingKernel(
      mx::Stream stream,
      std::shared_ptr<const AliasingState> state,
      rust::Slice<const std::uint32_t> constants,
      MTL::Size grid,
      MTL::Size group)
      : Primitive(stream), state_(std::move(state)), parameters_(constants.begin(), constants.end()),
        constant_count_(constants.size()), grid_(grid), group_(group) {
    parameters_.resize(constant_count_ + state_->stride_inputs.size());
  }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("aliasing Metal kernels require a GPU stream");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    if (outputs.size() != state_->aliases.size()) {
      throw std::runtime_error("aliasing Metal kernel output arity changed");
    }
    for (std::size_t index = 0; index < state_->aliases.size(); ++index) {
      outputs[index].copy_shared_buffer(inputs.at(state_->aliases[index]));
    }
    for (std::size_t index = 0; index < state_->stride_inputs.size(); ++index) {
      auto stride = inputs.at(state_->stride_inputs[index]).strides(state_->stride_axes[index]);
      if (stride < 0 || stride > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("aliasing Metal kernel stride exceeds uint32");
      }
      parameters_[constant_count_ + index] = static_cast<std::uint32_t>(stride);
    }
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(state_->pipeline);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      auto output = state_->output_by_input[index];
      if (output < 0) {
        encoder.set_input_array(inputs[index], index);
      } else {
        encoder.set_output_array(outputs[static_cast<std::size_t>(output)], index);
      }
    }
    encoder.set_vector_bytes(parameters_, inputs.size());
    encoder.dispatch_threads(grid_, group_);
  }

  const char* name() const override { return state_->function.c_str(); }

 private:
  std::shared_ptr<const AliasingState> state_;
  std::vector<std::uint32_t> parameters_;
  std::size_t constant_count_;
  MTL::Size grid_, group_;
};

std::unique_ptr<Arrays> dispatch(
    const std::shared_ptr<const AliasingState>& state,
    const std::vector<mx::array>& inputs,
    rust::Slice<const std::uint32_t> constants,
    MTL::Size grid,
    MTL::Size group,
    const Stream& stream,
    std::vector<mx::Shape>& shapes,
    std::vector<mx::Dtype>& dtypes) {
  shapes.clear();
  dtypes.clear();
  for (auto alias : state->aliases) {
    shapes.push_back(inputs.at(alias).shape());
    dtypes.push_back(inputs.at(alias).dtype());
  }
  auto primitive = std::make_shared<AliasingKernel>(
      stream.value, state, constants, grid, group);
  return std::make_unique<Arrays>(
      mx::array::make_arrays(shapes, dtypes, std::move(primitive), inputs), &stream);
}

std::shared_ptr<const AliasingState> state(
    std::size_t input_arity,
    rust::Slice<const std::uint32_t> aliases,
    rust::Slice<const std::uint32_t> stride_inputs,
    rust::Slice<const std::uint32_t> stride_axes,
    rust::Str function,
    const MetalLibrary& library) {
  if (input_arity == 0 || aliases.empty() || stride_inputs.size() != stride_axes.size()) {
    throw std::runtime_error("invalid aliasing Metal plan");
  }
  for (auto alias : aliases) {
    if (alias >= input_arity) throw std::runtime_error("aliasing output is out of bounds");
  }
  for (auto input : stride_inputs) {
    if (input >= input_arity) throw std::runtime_error("aliasing stride input is out of bounds");
  }
  return std::make_shared<AliasingState>(
      input_arity, vector(aliases), vector(stride_inputs), vector(stride_axes),
      std::string(function.data(), function.size()), library);
}
}  // namespace

struct AliasingPlan::Impl {
  Impl(
      std::size_t input_arity,
      std::size_t constant_arity,
      std::shared_ptr<const AliasingState> state)
      : input_arity(input_arity), constant_arity(constant_arity), state(std::move(state)) {
    inputs.reserve(input_arity);
    shapes.reserve(this->state->aliases.size());
    dtypes.reserve(this->state->aliases.size());
  }

  std::size_t input_arity, constant_arity;
  std::shared_ptr<const AliasingState> state;
  std::vector<mx::array> inputs;
  std::vector<mx::Shape> shapes;
  std::vector<mx::Dtype> dtypes;
};

AliasingPlan::AliasingPlan(std::unique_ptr<Impl> implementation)
    : implementation(std::move(implementation)) {}
AliasingPlan::~AliasingPlan() = default;

std::unique_ptr<AliasingPlan> new_aliasing_plan(
    std::size_t input_arity,
    std::size_t constant_arity,
    rust::Slice<const std::uint32_t> aliases,
    rust::Slice<const std::uint32_t> stride_inputs,
    rust::Slice<const std::uint32_t> stride_axes,
    rust::Str function,
    const MetalLibrary& library) {
  auto shared = state(input_arity, aliases, stride_inputs, stride_axes, function, library);
  return std::make_unique<AliasingPlan>(
      std::make_unique<AliasingPlan::Impl>(input_arity, constant_arity, std::move(shared)));
}

void aliasing_plan_set_input(AliasingPlan& plan, std::size_t index, const Array& input) {
  auto& values = plan.implementation->inputs;
  if (index > values.size() || index >= plan.implementation->input_arity) {
    throw std::runtime_error("prepared aliasing input index is out of bounds");
  }
  if (index == values.size()) {
    values.push_back(input.value);
  } else {
    values[index] = input.value;
  }
}

std::unique_ptr<Arrays> aliasing_plan_dispatch(
    AliasingPlan& plan,
    rust::Slice<const std::uint32_t> constants,
    std::uint32_t gx,
    std::uint32_t gy,
    std::uint32_t gz,
    std::uint32_t tx,
    std::uint32_t ty,
    std::uint32_t tz,
    const Stream& stream) {
  auto& implementation = *plan.implementation;
  if (implementation.inputs.size() != implementation.input_arity ||
      constants.size() != implementation.constant_arity) {
    throw std::runtime_error("prepared aliasing dispatch arity changed");
  }
  return dispatch(
      implementation.state, implementation.inputs, constants, MTL::Size(gx, gy, gz),
      MTL::Size(tx, ty, tz), stream, implementation.shapes, implementation.dtypes);
}

std::unique_ptr<Arrays> aliasing_dispatch(
    const Arrays& inputs,
    rust::Slice<const std::uint32_t> aliases,
    rust::Slice<const std::uint32_t> constants,
    rust::Slice<const std::uint32_t> stride_inputs,
    rust::Slice<const std::uint32_t> stride_axes,
    rust::Str function,
    const MetalLibrary& library,
    std::uint32_t gx,
    std::uint32_t gy,
    std::uint32_t gz,
    std::uint32_t tx,
    std::uint32_t ty,
    std::uint32_t tz,
    const Stream& stream) {
  auto shared = state(
      inputs.values.size(), aliases, stride_inputs, stride_axes, function, library);
  std::vector<mx::Shape> shapes;
  std::vector<mx::Dtype> dtypes;
  shapes.reserve(shared->aliases.size());
  dtypes.reserve(shared->aliases.size());
  return dispatch(
      shared, inputs.values, constants, MTL::Size(gx, gy, gz), MTL::Size(tx, ty, tz), stream,
      shapes, dtypes);
}
}  // namespace mirtal

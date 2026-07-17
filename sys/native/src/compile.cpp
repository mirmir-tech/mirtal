#include "mirtal-sys/src/lib.rs.h"

#include "mlx/compile.h"

#include <stdexcept>
#include <utility>

namespace mirtal {
namespace {
struct CallbackState {
  CallbackState(rust::Box<GraphCallback> callback, const Stream& source)
      : callback(std::move(callback)), stream(source.value, source.id) {}
  rust::Box<GraphCallback> callback;
  Stream stream;
};
}  // namespace

std::unique_ptr<Arrays> new_arrays() { return std::make_unique<Arrays>(); }

void arrays_push(Arrays& arrays, const Array& array) {
  arrays.values.push_back(array.value);
}

std::size_t arrays_len(const Arrays& arrays) noexcept { return arrays.values.size(); }

std::shared_ptr<Array> arrays_get(const Arrays& arrays, std::size_t index) {
  if (index >= arrays.values.size()) throw std::runtime_error("array index is out of bounds");
  return std::make_shared<Array>(arrays.values[index]);
}

const Stream& arrays_stream(const Arrays& arrays) {
  if (arrays.stream == nullptr) throw std::runtime_error("array list has no execution stream");
  return *arrays.stream;
}

std::unique_ptr<Compiled> new_compiled(
    rust::Box<GraphCallback> callback,
    bool shapeless,
    const Stream& stream) {
  auto state = std::make_shared<CallbackState>(std::move(callback), stream);
  auto graph = [state](const std::vector<mx::array>& inputs) {
    auto values = std::make_unique<Arrays>(inputs, &state->stream);
    auto outputs = invoke_graph(*state->callback, std::move(values));
    return std::move(outputs->values);
  };
  auto function = mx::compile(
      CompiledFunction{std::move(graph)},
      shapeless);
  return std::make_unique<Compiled>(Stream(stream.value, stream.id), std::move(function));
}

std::unique_ptr<Arrays> compiled_call(const Compiled& compiled, const Arrays& inputs) {
  return std::make_unique<Arrays>(compiled.function(inputs.values), &compiled.stream);
}

std::size_t compiled_native_handle(const Compiled& compiled) noexcept {
  return reinterpret_cast<std::size_t>(&compiled);
}

std::vector<mx::array> call_compiled_native(
    std::size_t address,
    const std::vector<mx::array>& inputs) {
  if (address == 0) throw std::runtime_error("native compiled graph is null");
  return reinterpret_cast<const Compiled*>(address)->function(inputs);
}
}  // namespace mirtal

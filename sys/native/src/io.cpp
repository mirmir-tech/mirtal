#include "mirtal/bridge.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mirtal {
namespace {
std::string string(rust::Str value) {
  return std::string(value.data(), value.size());
}
}  // namespace

std::unique_ptr<TensorMap> load_safetensors(rust::Str path, const Stream& stream) {
  auto loaded = mx::load_safetensors(string(path), stream.value);
  return std::make_unique<TensorMap>(TensorMap{std::move(loaded.first)});
}

std::size_t tensor_map_len(const TensorMap& tensors) noexcept {
  return tensors.values.size();
}

void tensor_map_eval(const TensorMap& tensors) {
  std::vector<mx::array> values;
  values.reserve(tensors.values.size());
  for (const auto& [name, value] : tensors.values) values.push_back(value);
  mx::eval(std::move(values));
}

bool tensor_map_contains(const TensorMap& tensors, rust::Str name) {
  return tensors.values.contains(string(name));
}

std::shared_ptr<Array> tensor_map_get(const TensorMap& tensors, rust::Str name) {
  auto key = string(name);
  auto found = tensors.values.find(key);
  if (found == tensors.values.end()) {
    throw std::runtime_error("missing safetensor: " + key);
  }
  return std::make_shared<Array>(found->second);
}
}  // namespace mirtal

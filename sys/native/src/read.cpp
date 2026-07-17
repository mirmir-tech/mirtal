#include "mirtal/bridge.h"

#include <algorithm>
#include <stdexcept>

namespace mirtal {
void array_read_f32(const Array& array, rust::Slice<float> output) {
  if (array.value.dtype() != mx::float32 || array.value.size() != output.size()) {
    throw std::runtime_error("float32 array output is incompatible");
  }
  auto evaluated = mx::contiguous(array.value);
  evaluated.eval();
  std::copy_n(evaluated.data<float>(), output.size(), output.data());
}

std::uint32_t array_read_u32_scalar(const Array& array) {
  if (array.value.dtype() != mx::uint32 || array.value.size() != 1) {
    throw std::runtime_error("expected one uint32 array value");
  }
  mx::eval(array.value);
  return array.value.data<std::uint32_t>()[0];
}
}  // namespace mirtal

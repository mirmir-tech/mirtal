#pragma once

#include <cstdint>

#include "rust/cxx.h"

namespace mirtal {

struct Array;

void array_read_f32(const Array&, rust::Slice<float>);
std::uint32_t array_read_u32_scalar(const Array&);

}  // namespace mirtal

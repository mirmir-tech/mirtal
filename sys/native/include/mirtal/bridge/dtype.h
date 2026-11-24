#pragma once

#include <cstdint>

#include "mirtal/native.h"

namespace mirtal {

// Element types across the bridge. The Rust bridge declares the same enum,
// and cxx asserts at compile time that every discriminant matches.
enum class DType : std::uint8_t {
  Bool,
  Uint32,
  Int32,
  Float16,
  Bfloat16,
  Float32,
  Uint8,
};

// The MLX type of `value`; throws for a value no enumerator names.
mx::Dtype native_dtype(DType value);
// The bridge type of an MLX type; throws for a type the bridge does not carry.
DType bridge_dtype(const mx::Dtype& value);

}  // namespace mirtal

#pragma once

#include <memory>

namespace mirtal {

struct Array;
struct Arrays;
struct MetalKernel;
struct MetalLaunch;
struct Stream;

struct PreparedMetal {
  struct Impl;

  explicit PreparedMetal(std::unique_ptr<Impl> implementation);
  ~PreparedMetal();

  std::unique_ptr<Impl> implementation;
};

std::unique_ptr<PreparedMetal> new_prepared_metal(
    const MetalKernel&,
    const MetalLaunch&);
void prepared_metal_set_input(PreparedMetal&, std::size_t, const Array&);
std::unique_ptr<Arrays> prepared_metal_dispatch(PreparedMetal&, const Stream&);

}  // namespace mirtal

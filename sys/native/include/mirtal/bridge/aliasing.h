#pragma once

#include <memory>

namespace mirtal {

struct Array;
struct Arrays;
struct MetalLibrary;
struct Stream;

struct AliasingPlan {
  struct Impl;

  explicit AliasingPlan(std::unique_ptr<Impl> implementation);
  ~AliasingPlan();

  std::unique_ptr<Impl> implementation;
};

std::unique_ptr<AliasingPlan> new_aliasing_plan(
    std::size_t,
    std::size_t,
    rust::Slice<const std::uint32_t>,
    rust::Slice<const std::uint32_t>,
    rust::Slice<const std::uint32_t>,
    rust::Str,
    const MetalLibrary&);

void aliasing_plan_set_input(AliasingPlan&, std::size_t, const Array&);

std::unique_ptr<Arrays> aliasing_plan_dispatch(
    AliasingPlan&,
    rust::Slice<const std::uint32_t>,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    const Stream&);

std::unique_ptr<Arrays> aliasing_dispatch(
    const Arrays&,
    rust::Slice<const std::uint32_t>,
    rust::Slice<const std::uint32_t>,
    rust::Slice<const std::uint32_t>,
    rust::Slice<const std::uint32_t>,
    rust::Str,
    const MetalLibrary&,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    const Stream&);

}  // namespace mirtal

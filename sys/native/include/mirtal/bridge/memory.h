#pragma once
#include "mirtal/native.h"
#include <memory>

namespace mirtal {
struct Allocation {
  explicit Allocation(std::shared_ptr<mx::array::Data> data)
      : data(std::move(data)) {}
  std::shared_ptr<mx::array::Data> data;
};
std::shared_ptr<Allocation> array_allocation(const Array& array);
std::size_t allocation_identity(const Allocation& allocation);
std::size_t allocation_bytes(const Allocation& allocation);
}

#include "mirtal/bridge/memory.h"

namespace mirtal {
std::shared_ptr<Allocation> array_allocation(const Array& array) {
  if (!array.value.is_available()) return nullptr;
  auto data = array.value.data_shared_ptr();
  if (!data || !data->buffer().ptr()) return nullptr;
  return std::make_shared<Allocation>(std::move(data));
}
std::size_t allocation_identity(const Allocation& allocation) {
  return reinterpret_cast<std::size_t>(allocation.data->buffer().ptr());
}
std::size_t allocation_bytes(const Allocation& allocation) {
  return mx::allocator::allocator().size(allocation.data->buffer());
}
}

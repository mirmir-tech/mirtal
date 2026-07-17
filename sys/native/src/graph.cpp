#include "mirtal/bridge.h"

#include "mlx/graph_utils.h"

#include <fstream>
#include <stdexcept>
#include <string>

namespace mirtal {
void export_graph_dot(const Array& array, rust::Str path) {
  auto native_path = std::string(path.data(), path.size());
  std::ofstream output(native_path);
  if (!output) throw std::runtime_error("MLX graph dump path cannot be opened");
  mx::export_to_dot(output, array.value);
}
}  // namespace mirtal

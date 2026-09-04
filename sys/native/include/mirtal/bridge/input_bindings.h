#pragma once

#include "mlx/array.h"
#include <vector>

namespace mirtal {
// Lazy outputs own their graph inputs. Keep the reusable vector's capacity,
// but never keep per-dispatch array owners in a prepared launch plan.
struct InputBindingsScope {
  explicit InputBindingsScope(std::vector<mlx::core::array>& inputs) : inputs(inputs) {}
  ~InputBindingsScope() { inputs.clear(); }
  InputBindingsScope(const InputBindingsScope&) = delete;
  InputBindingsScope& operator=(const InputBindingsScope&) = delete;
  std::vector<mlx::core::array>& inputs;
};
} // namespace mirtal

#include "mirtal-sys/src/lib.rs.h"

#include <optional>
#include <utility>

namespace mirtal {
mx::array rope_native(
    const mx::array& input,
    std::int32_t dimensions,
    bool traditional,
    std::optional<float> base,
    float scale,
    std::int32_t offset,
    std::optional<mx::array> frequencies,
    const mx::Stream& stream) {
  return mx::fast::rope(
      input,
      dimensions,
      traditional,
      base,
      scale,
      offset,
      frequencies,
      stream);
}

std::shared_ptr<Array> rope(
    const Array& input,
    const NativeRopeOptions& options,
    const Stream& stream) {
  auto optional_base = options.has_base
      ? std::make_optional(options.base)
      : std::optional<float>{};
  return std::make_shared<Array>(rope_native(
      input.value,
      options.dimensions,
      options.traditional,
      optional_base,
      options.scale,
      options.offset,
      std::nullopt,
      stream.value));
}

std::shared_ptr<Array> rope_with_frequencies(
    const Array& input,
    std::int32_t dimensions,
    bool traditional,
    const Array& frequencies,
    std::int32_t offset,
    const Stream& stream) {
  return std::make_shared<Array>(rope_native(
      input.value,
      dimensions,
      traditional,
      std::nullopt,
      1.0f,
      offset,
      frequencies.value,
      stream.value));
}

}  // namespace mirtal

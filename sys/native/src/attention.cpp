#include "mirtal-sys/src/lib.rs.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace mirtal {
namespace {
mx::array sdpa_impl(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    float scale,
    const std::string& mask_mode,
    std::optional<mx::array> mask,
    std::optional<mx::array> sinks,
    const mx::Stream& stream) {
  return mx::fast::scaled_dot_product_attention(
      queries, keys, values, scale, mask_mode, mask, sinks, stream);
}
}  // namespace

mx::array sdpa_native(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    float scale,
    bool causal,
    const mx::Stream& stream) {
  return sdpa_impl(
      queries,
      keys,
      values,
      scale,
      causal ? "causal" : "",
      std::nullopt,
      std::nullopt,
      stream);
}

std::shared_ptr<Array> sdpa(
    const Array& queries,
    const Array& keys,
    const Array& values,
    const NativeAttentionOptions& options,
    std::shared_ptr<Array> mask,
    std::shared_ptr<Array> sinks,
    const Stream& stream) {
  std::string mask_mode;
  std::optional<mx::array> mask_array;
  switch (options.mask_kind) {
    case 0: break;
    case 1: mask_mode = "causal"; break;
    case 2:
      if (mask == nullptr) throw std::invalid_argument("attention mask is null");
      mask_array = mask->value;
      break;
    default: throw std::invalid_argument("invalid attention mask kind");
  }
  auto sink_array = sinks == nullptr
      ? std::optional<mx::array>{}
      : std::make_optional(sinks->value);
  return std::make_shared<Array>(sdpa_impl(
      queries.value,
      keys.value,
      values.value,
      options.scale,
      mask_mode,
      mask_array,
      sink_array,
      stream.value));
}
}  // namespace mirtal

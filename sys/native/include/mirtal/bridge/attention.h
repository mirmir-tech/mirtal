#pragma once

#include <cstdint>
#include <memory>

namespace mirtal {

struct Array;
struct NativeAttentionOptions;
struct Stream;

std::shared_ptr<Array> sdpa(
    const Array& queries,
    const Array& keys,
    const Array& values,
    const NativeAttentionOptions& options,
    std::shared_ptr<Array> mask,
    std::shared_ptr<Array> sinks,
    const Stream& stream);

}  // namespace mirtal

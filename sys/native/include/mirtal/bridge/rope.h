#pragma once

#include <cstdint>
#include <memory>

namespace mirtal {

struct Array;
struct NativeRopeOptions;
struct Stream;

std::shared_ptr<Array> rope(
    const Array& input,
    const NativeRopeOptions& options,
    const Stream& stream);
std::shared_ptr<Array> rope_with_frequencies(
    const Array& input,
    std::int32_t dimensions,
    bool traditional,
    const Array& frequencies,
    std::int32_t offset,
    const Stream& stream);
}  // namespace mirtal

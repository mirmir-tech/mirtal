#include "mirtal-sys/src/io.rs.h"
#include "mlx/allocator.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace mirtal {
namespace {
std::string string(rust::Str value) {
  return std::string(value.data(), value.size());
}

// Positional pread reader over a whole file; MLX's Load primitive reads each
// array's bytes from it lazily on evaluation.
class FileReader : public mx::io::Reader {
 public:
  explicit FileReader(std::string path)
      : file_(std::fopen(path.c_str(), "rb"), &std::fclose),
        path_(std::move(path)) {
    if (!good()) throw std::runtime_error("failed to open tensor file " + path_);
    struct stat info{};
    if (fstat(fileno(file_.get()), &info) != 0 || info.st_size < 0) {
      throw std::runtime_error("failed to inspect tensor file " + path_);
    }
    size_ = static_cast<std::size_t>(info.st_size);
  }

  bool is_open() const override { return file_ != nullptr; }
  bool good() const override { return is_open(); }
  std::size_t tell() override { return position_; }

  void seek(std::int64_t offset, std::ios_base::seekdir way) override {
    const auto base = way == std::ios_base::beg
                          ? 0
                          : static_cast<std::int64_t>(
                                way == std::ios_base::end ? size_ : position_);
    if (offset < -base) {
      throw std::runtime_error("tensor file seek precedes file start");
    }
    position_ = static_cast<std::size_t>(base + offset);
  }

  void read(char* data, std::size_t count) override {
    read(data, count, position_);
    position_ += count;
  }

  void read(char* data, std::size_t count, std::size_t offset) override {
    read_file(data, count, offset);
  }

  std::string label() const override { return "file " + path_; }
  std::size_t size() const noexcept { return size_; }

 protected:
  void read_file(char* data, std::size_t count, std::size_t offset) const {
    if (offset > size_ || count > size_ - offset) {
      throw std::runtime_error("tensor read exceeds file size");
    }
    // macOS refuses a single read above INT_MAX bytes (EINVAL), so large
    // tensors are read a gibibyte at a time.
    constexpr std::size_t max_read = std::size_t{1} << 30;
    while (count > 0) {
      const auto read =
          pread(fileno(file_.get()), data, std::min(count, max_read), offset);
      if (read < 0 && errno == EINTR) continue;
      if (read <= 0) {
        // The file shrank since its ranges were checked, or the read failed.
        std::ostringstream message;
        message << "failed to read " << count << " bytes at offset " << offset << " from "
                << path_ << ": "
                << (read == 0 ? "unexpected end of file" : std::strerror(errno));
        throw std::runtime_error(message.str());
      }
      data += read;
      count -= static_cast<std::size_t>(read);
      offset += static_cast<std::size_t>(read);
    }
  }

 private:
  std::unique_ptr<std::FILE, int (*)(std::FILE*)> file_;
  std::string path_;
  std::size_t position_{0};
  std::size_t size_{0};
};

class SafeTensorReader final : public FileReader {
 public:
  explicit SafeTensorReader(std::string path) : FileReader(std::move(path)) {
    const auto size_ = size();
    std::uint64_t length = 0;
    read_file(reinterpret_cast<char*>(&length), sizeof(length), 0);
    constexpr std::uint64_t kMaximumHeader = 100000000;
    if (length == 0 || length >= kMaximumHeader) {
      throw std::runtime_error("invalid SafeTensors header length");
    }
    if (length > size_ - std::min(size_, sizeof(length))) {
      throw std::runtime_error("SafeTensors header exceeds file size");
    }
    header_.resize(length);
    read_file(header_.data(), header_.size(), sizeof(length));
    replace_e5m2();
  }

  using FileReader::read;

  void read(char* data, std::size_t count, std::size_t offset) override {
    read_file(data, count, offset);
    constexpr std::size_t kHeaderOffset = sizeof(std::uint64_t);
    const auto start = std::max(offset, kHeaderOffset);
    const auto end = std::min(offset + count, kHeaderOffset + header_.size());
    if (start < end) {
      std::memcpy(data + start - offset, header_.data() + start - kHeaderOffset,
                  end - start);
    }
  }

  void release_header() {
    header_.clear();
    header_.shrink_to_fit();
  }

 private:
  void replace_e5m2() {
    constexpr std::string_view source = "\"F8_E5M2\"";
    constexpr std::string_view replacement = "\"U8\"     ";
    auto begin = header_.begin();
    while ((begin = std::search(begin, header_.end(), source.begin(),
                                source.end())) != header_.end()) {
      if (is_dtype_value(static_cast<std::size_t>(begin - header_.begin()))) {
        std::copy(replacement.begin(), replacement.end(), begin);
      }
      begin += replacement.size();
    }
  }

  bool is_dtype_value(std::size_t value_offset) const {
    auto cursor = value_offset;
    while (cursor > 0 && std::isspace(static_cast<unsigned char>(header_[cursor - 1]))) {
      --cursor;
    }
    if (cursor == 0 || header_[--cursor] != ':') return false;
    while (cursor > 0 && std::isspace(static_cast<unsigned char>(header_[cursor - 1]))) {
      --cursor;
    }
    constexpr std::string_view key = "\"dtype\"";
    return cursor >= key.size()
        && std::equal(key.begin(), key.end(), header_.begin() + cursor - key.size());
  }

  std::vector<char> header_;
};

// One array read from `offset` of a file on evaluation, as MLX's `Load`
// reads a SafeTensors tensor; libmlx does not export `Load` itself.
//
// The read runs in `eval_cpu` itself, on the thread evaluating the array,
// not dispatched to the CPU stream's worker: a failed read throws back
// through the evaluation to the caller (MLX's worker has no handler and
// would terminate the process), and the buffer is filled before the
// evaluation returns, so dropping the array right after cannot leave a read
// writing into freed memory. A range has no inputs to wait for.
class RangeRead final : public mx::Primitive {
 public:
  RangeRead(mx::Stream stream, std::shared_ptr<FileReader> reader, std::size_t offset)
      : Primitive(stream), reader_(std::move(reader)), offset_(offset) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>& outputs) override {
    auto& out = outputs.at(0);
    out.set_data(mx::allocator::malloc(out.nbytes()));
    reader_->read(out.data<char>(), out.nbytes(), offset_);
  }

  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("tensor ranges read on a CPU stream");
  }

  const char* name() const override { return "RangeRead"; }

 private:
  std::shared_ptr<FileReader> reader_;
  std::size_t offset_;
};
}  // namespace

std::unique_ptr<TensorMap> load_safetensors(rust::Str path, const Stream& stream) {
  auto reader = std::make_shared<SafeTensorReader>(string(path));
  auto loaded = mx::load_safetensors(reader, stream.value);
  reader->release_header();
  return std::make_unique<TensorMap>(TensorMap{std::move(loaded.first)});
}

// Each range becomes an array of one RangeRead over a shared reader, as
// mx::load_safetensors builds its arrays: evaluation preads the range into a
// newly allocated buffer, one copy from the file. Rust validated the ranges;
// the reader still bounds every read, should the file change.
std::unique_ptr<TensorMap> load_tensor_ranges(
    rust::Str path, rust::Slice<const TensorRange> ranges, const Stream& stream) {
  if (stream.value.device != mx::Device::cpu) {
    throw std::invalid_argument("tensor ranges load on a CPU stream");
  }
  auto reader = std::make_shared<FileReader>(string(path));
  TensorMap tensors;
  tensors.values.reserve(ranges.size());
  for (const auto& range : ranges) {
    mx::Shape shape;
    shape.reserve(range.shape.size());
    for (const auto dimension : range.shape) {
      shape.push_back(static_cast<mx::ShapeElem>(dimension));
    }
    auto load = std::make_shared<RangeRead>(stream.value, reader, range.offset);
    tensors.values.insert_or_assign(
        std::string(range.name),
        mx::array(std::move(shape), native_dtype(range.dtype), std::move(load), {}));
  }
  return std::make_unique<TensorMap>(std::move(tensors));
}

std::size_t tensor_map_len(const TensorMap& tensors) noexcept {
  return tensors.values.size();
}

void tensor_map_eval(const TensorMap& tensors) {
  std::vector<mx::array> values;
  values.reserve(tensors.values.size());
  for (const auto& [name, value] : tensors.values) values.push_back(value);
  mx::eval(std::move(values));
}

bool tensor_map_contains(const TensorMap& tensors, rust::Str name) {
  return tensors.values.contains(string(name));
}

std::shared_ptr<Array> tensor_map_get(const TensorMap& tensors, rust::Str name) {
  auto key = string(name);
  auto found = tensors.values.find(key);
  if (found == tensors.values.end()) {
    throw std::runtime_error("missing safetensor: " + key);
  }
  return std::make_shared<Array>(found->second);
}

std::shared_ptr<Array> tensor_map_take(TensorMap& tensors, rust::Str name) {
  auto key = string(name);
  auto node = tensors.values.extract(key);
  if (node.empty()) throw std::runtime_error("missing safetensor: " + key);
  return std::make_shared<Array>(std::move(node.mapped()));
}
}  // namespace mirtal

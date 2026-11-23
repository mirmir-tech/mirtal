#include "mirtal-sys/src/io.rs.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
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
                                way == std::ios_base::end ? stream_size() : position_);
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
  // Size of the byte stream presented to MLX; subclasses that prepend
  // synthesized bytes extend it.
  virtual std::size_t stream_size() const noexcept { return size_; }

  void read_file(char* data, std::size_t count, std::size_t offset) const {
    if (offset > size_ || count > size_ - offset) {
      throw std::runtime_error("tensor read exceeds file size");
    }
    while (count > 0) {
      const auto read = pread(fileno(file_.get()), data, count, offset);
      if (read <= 0) {
        std::ostringstream message;
        message << "failed to read " << count << " bytes from " << path_;
        throw std::runtime_error(message.str());
      }
      data += read;
      count -= read;
      offset += read;
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

// Presents named ranges of an arbitrary file as a SafeTensors stream: a
// synthesized header precedes the file, and every tensor's data offsets
// address its bytes in place, so MLX loads each range lazily without copies.
class RangeReader final : public FileReader {
 public:
  RangeReader(std::string path, rust::Slice<const TensorRange> ranges)
      : FileReader(std::move(path)) {
    std::unordered_set<std::string> names;
    std::ostringstream header;
    header << '{';
    for (const auto& range : ranges) {
      auto name = std::string(range.name);
      if (!names.insert(name).second) {
        throw std::runtime_error("duplicate tensor range: " + name);
      }
      const auto [label, element] = safetensors_dtype(range.dtype);
      std::uint64_t bytes = element;
      std::ostringstream shape;
      for (const auto dimension : range.shape) {
        if (dimension == 0) throw std::runtime_error("empty tensor range: " + name);
        if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
          throw std::runtime_error("tensor range dimension exceeds int32: " + name);
        }
        if (bytes > std::numeric_limits<std::uint64_t>::max() / dimension) {
          throw std::runtime_error("tensor range size overflows: " + name);
        }
        bytes *= dimension;
        if (shape.tellp() > 0) shape << ',';
        shape << dimension;
      }
      if (range.shape.empty()) throw std::runtime_error("scalar tensor range: " + name);
      if (range.offset > size() || bytes > size() - range.offset) {
        throw std::runtime_error("tensor range exceeds file size: " + name);
      }
      if (header.tellp() > 1) header << ',';
      header << '"' << escape(name) << R"(":{"dtype":")" << label << R"(","shape":[)"
             << shape.str() << R"(],"data_offsets":[)" << range.offset << ','
             << range.offset + bytes << "]}";
    }
    header << '}';
    header_ = header.str();
    while (header_.size() % 8 != 0) header_.push_back(' ');
    prefix_ = sizeof(std::uint64_t) + header_.size();
  }

  using FileReader::read;

  void read(char* data, std::size_t count, std::size_t offset) override {
    if (offset < prefix_) {
      const auto head = std::min(count, prefix_ - offset);
      for (std::size_t index = 0; index < head; ++index) {
        data[index] = prefix_byte(offset + index);
      }
      data += head;
      count -= head;
      offset += head;
    }
    if (count > 0) read_file(data, count, offset - prefix_);
  }

  void release_header() { header_.clear(); header_.shrink_to_fit(); }

 protected:
  std::size_t stream_size() const noexcept override { return prefix_ + size(); }

 private:
  char prefix_byte(std::size_t offset) const {
    if (offset >= sizeof(std::uint64_t)) return header_[offset - sizeof(std::uint64_t)];
    const auto length = static_cast<std::uint64_t>(header_.size());
    return static_cast<char>((length >> (8 * offset)) & 0xff);
  }

  // Codes follow mirtal::DType.
  static std::pair<const char*, std::uint64_t> safetensors_dtype(std::uint8_t code) {
    switch (code) {
      case 0: return {"BOOL", 1};
      case 1: return {"U32", 4};
      case 2: return {"I32", 4};
      case 3: return {"F16", 2};
      case 4: return {"BF16", 2};
      case 5: return {"F32", 4};
      case 6: return {"U8", 1};
      default: throw std::runtime_error("unsupported tensor range dtype");
    }
  }

  static std::string escape(const std::string& name) {
    std::string escaped;
    for (const char value : name) {
      if (value == '"' || value == '\\') escaped.push_back('\\');
      if (static_cast<unsigned char>(value) < 0x20) {
        throw std::runtime_error("tensor range name has a control character");
      }
      escaped.push_back(value);
    }
    return escaped;
  }

  std::string header_;
  std::size_t prefix_{0};
};
}  // namespace

std::unique_ptr<TensorMap> load_safetensors(rust::Str path, const Stream& stream) {
  auto reader = std::make_shared<SafeTensorReader>(string(path));
  auto loaded = mx::load_safetensors(reader, stream.value);
  reader->release_header();
  return std::make_unique<TensorMap>(TensorMap{std::move(loaded.first)});
}

std::unique_ptr<TensorMap> load_tensor_ranges(
    rust::Str path, rust::Slice<const TensorRange> ranges, const Stream& stream) {
  auto reader = std::make_shared<RangeReader>(string(path), ranges);
  auto loaded = mx::load_safetensors(reader, stream.value);
  reader->release_header();
  return std::make_unique<TensorMap>(TensorMap{std::move(loaded.first)});
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
}  // namespace mirtal

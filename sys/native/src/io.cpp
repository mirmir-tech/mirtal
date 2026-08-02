#include "mirtal/bridge.h"

#include <algorithm>
#include <cctype>
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

class SafeTensorReader final : public mx::io::Reader {
 public:
  explicit SafeTensorReader(std::string path)
      : file_(std::fopen(path.c_str(), "rb"), &std::fclose),
        path_(std::move(path)) {
    if (!good()) throw std::runtime_error("failed to open SafeTensors file");
    struct stat info{};
    if (fstat(fileno(file_.get()), &info) != 0 || info.st_size < 0) {
      throw std::runtime_error("failed to inspect SafeTensors file");
    }
    size_ = static_cast<std::size_t>(info.st_size);
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

  bool is_open() const override { return file_ != nullptr; }
  bool good() const override { return is_open(); }
  std::size_t tell() override { return position_; }

  void seek(std::int64_t offset, std::ios_base::seekdir way) override {
    const auto base = way == std::ios_base::beg
                          ? 0
                          : static_cast<std::int64_t>(
                                way == std::ios_base::end ? size_ : position_);
    if (offset < -base) {
      throw std::runtime_error("SafeTensors seek precedes file start");
    }
    position_ = static_cast<std::size_t>(base + offset);
  }

  void read(char* data, std::size_t count) override {
    read(data, count, position_);
    position_ += count;
  }

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

  std::string label() const override { return "file " + path_; }

  void release_header() {
    header_.clear();
    header_.shrink_to_fit();
  }

 private:
  void read_file(char* data, std::size_t count, std::size_t offset) const {
    if (offset > size_ || count > size_ - offset) {
      throw std::runtime_error("SafeTensors read exceeds file size");
    }
    while (count > 0) {
      const auto read = pread(fileno(file_.get()), data, count, offset);
      if (read <= 0) {
        std::ostringstream message;
        message << "failed to read " << count << " SafeTensors bytes";
        throw std::runtime_error(message.str());
      }
      data += read;
      count -= read;
      offset += read;
    }
  }

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

  std::unique_ptr<std::FILE, int (*)(std::FILE*)> file_;
  std::string path_;
  std::vector<char> header_;
  std::size_t position_{0};
  std::size_t size_{0};
};
}  // namespace

std::unique_ptr<TensorMap> load_safetensors(rust::Str path, const Stream& stream) {
  auto reader = std::make_shared<SafeTensorReader>(string(path));
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

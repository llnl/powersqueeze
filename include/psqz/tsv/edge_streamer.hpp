// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/tsv/reader.hpp>

#include <sstream>

namespace psqz::tsv::detail {

template <typename ParametersType>
std::string map_name(std::string &&name, ParametersType &params) {
  std::stringstream ss;
  ss << name << "_local_map_" << params.vertex_count();
  return ss.str();
}

template <typename HandlerType,
          template <typename, template <typename> class> class BaseType>
struct edge_streamer : public BaseType<HandlerType, psqz::tsv::reader> {
  using handler_type = HandlerType;
  using base_type    = BaseType<handler_type, psqz::tsv::reader>;
  using reader_type  = typename base_type::reader_type;

 public:
  edge_streamer(handler_type &handler) : base_type(handler) {}

  std::string local_map_name() const override {
    return map_name(this->name(), this->_params);
  }

  reader_type spawn_reader() override {
    return {this->_comm, this->_params.index_filename()};
  }
};

template <typename HandlerType, template <typename> class BaseType>
struct normalize : public BaseType<HandlerType> {
  using handler_type = HandlerType;
  using base_type    = BaseType<handler_type>;

 public:
  normalize(handler_type &handler) : base_type(handler) {}

  std::string local_map_name() const override {
    return map_name(this->name(), this->_params);
  }
};
}  // namespace psqz::tsv::detail

// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/tsv/reader.hpp>

#include <sstream>
#include <type_traits>

namespace psqz::tsv::detail {

template <typename HandlerType,
          template <typename, template <typename> class> class BaseType>
struct community_streamer : public BaseType<HandlerType, psqz::tsv::reader> {
  using handler_type = HandlerType;
  using base_type    = BaseType<handler_type, psqz::tsv::reader>;
  using reader_type  = typename base_type::reader_type;

 public:
  community_streamer(handler_type &handler) : base_type(handler) {}

  std::string local_map_name() const override {
    return map_name(this->name(), this->_params);
  }

  reader_type spawn_reader() override {
    return {this->_comm, this->_params.truth_filename()};
  }
};

template <typename T, typename = int>
struct has_truth : std::false_type {};

template <typename T>
struct has_truth<T, decltype((void)T::truth_filename, 0)> : std::true_type {};

template <typename HandlerType,
          template <typename, template <typename> class> class BaseType>
struct queries : public BaseType<HandlerType, psqz::tsv::query_reader> {
  using handler_type    = HandlerType;
  using parameters_type = typename handler_type::parameters_type;

  using base_type   = BaseType<handler_type, psqz::tsv::query_reader>;
  using reader_type = typename base_type::reader_type;

 public:
  queries(handler_type &handler) : base_type(handler) {}

  // std::string local_map_name() const override {
  //   return map_name(this->name(), this->_params);
  // }

  reader_type spawn_reader() override {
    if (this->_params.query_filename().empty()) {
      if constexpr (has_truth<parameters_type>::value) {
        return {this->_comm, this->_params.truth_filename()};
      } else {
        throw std::logic_error("query-only workflow has no query file!");
      }
    } else {
      return {this->_comm, this->_params.query_filename()};
    }
  }
};
}  // namespace psqz::tsv::detail

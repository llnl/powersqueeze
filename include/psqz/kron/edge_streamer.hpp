// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/kron/reader.hpp>

#include <sstream>

namespace psqz::kron::detail {

template <typename ParametersType>
std::string map_name(std::string &&name, ParametersType &params) {
  std::stringstream ss;
  ss << name << "_local_map_" << params.vertex_count() << "_a"
     << params.target_degree_power() << "_b" << params.noise_ratio() << "_s"
     << params.random_seed();
  return ss.str();
}

template <typename HandlerType,
          template <typename, template <typename> class> class BaseType>
struct edge_streamer : public BaseType<HandlerType, psqz::kron::reader> {
  using handler_type    = HandlerType;
  using parameters_type = typename handler_type::parameters_type;
  using base_type       = BaseType<HandlerType, psqz::kron::reader>;
  using reader_type     = typename base_type::reader_type;

 public:
  edge_streamer(handler_type &handler) : base_type(handler) {}

  std::string local_map_name() const override {
    return map_name(this->name(), this->_params);
  }

  reader_type spawn_reader() override {
    reader_type reader(this->_comm, this->_params.left_index_filename(),
                       this->_params.right_index_filename(),
                       this->_params.left_truth_filename(),
                       this->_params.right_truth_filename(),
                       this->_params.left_truth_filename(),
                       this->_params.right_truth_filename(),
                       this->_params.target_degree_power(),
                       this->_params.noise_ratio(),
                       this->_params.random_seed());
    reader.set_mode(reader_type::mode_type::index);
    return reader;
  }
};

template <typename HandlerType, template <typename> class BaseType>
struct adjacency_normalizer : public BaseType<HandlerType> {
  using handler_type = HandlerType;
  using base_type    = BaseType<handler_type>;

 public:
  adjacency_normalizer(handler_type &handler) : base_type(handler) {}

  std::string local_map_name() const override {
    return map_name(this->name(), this->_params);
  }
};
}  // namespace psqz::kron::detail

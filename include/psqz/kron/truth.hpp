// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/kron/reader.hpp>

#include <sstream>

namespace psqz::kron::detail {

template <typename HandlerType,
          template <typename, template <typename> class> class BaseType>
struct truth : public BaseType<HandlerType, psqz::kron::reader> {
  using handler_type    = HandlerType;
  using parameters_type = typename handler_type::parameters_type;
  using base_type       = BaseType<HandlerType, psqz::kron::reader>;
  using reader_type     = typename base_type::reader_type;

 public:
  truth(handler_type &handler) : base_type(handler) {}

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
    reader.set_mode(reader_type::mode_type::truth);
    return reader;
  }
};

template <typename HandlerType,
          template <typename, template <typename> class> class BaseType>
struct queries : public BaseType<HandlerType, psqz::kron::reader> {
  using handler_type    = HandlerType;
  using parameters_type = typename handler_type::parameters_type;
  using base_type       = BaseType<HandlerType, psqz::kron::reader>;
  using reader_type     = typename base_type::reader_type;

 public:
  queries(handler_type &handler) : base_type(handler) {}

  // std::string local_map_name() const override {
  //   return map_name(this->name(), this->_params);
  // }

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
    reader.set_mode(reader_type::mode_type::truth);
    return reader;
  }
};

}  // namespace psqz::kron::detail

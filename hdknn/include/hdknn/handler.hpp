// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/handler.hpp>

namespace hdknn {

template <typename ParametersType, std::size_t RangeSize,
          std::size_t ReplicationCount, typename AdjacencyType,
          typename FeatureType = float, typename CmtyType = std::size_t,
          typename DistType = float>
class handler
    : public psqz::handler<ParametersType, RangeSize, ReplicationCount,
                           AdjacencyType, FeatureType, CmtyType> {
 public:
  using base_type = psqz::handler<ParametersType, RangeSize, ReplicationCount,
                                  AdjacencyType, FeatureType, CmtyType>;

  using parameters_type       = typename base_type::parameters_type;
  using index_type            = typename base_type::index_type;
  using feature_type          = typename base_type::feature_type;
  using index_vec_type        = typename base_type::index_vec_type;
  using feature_vec_type      = typename base_type::feature_vec_type;
  using cmty_type             = typename base_type::cmty_type;
  using weight_type           = typename base_type::weight_type;
  using adjacency_elt_type    = typename base_type::adjacency_elt_type;
  using adjacency_vec_type    = typename base_type::adjacency_vec_type;
  using adjacency_type        = typename base_type::adjacency_type;
  using truth_type            = typename base_type::truth_type;
  using query_type            = typename base_type::query_type;
  using sketch_container_type = typename base_type::sketch_container_type;
  using degree_type           = typename base_type::degree_type;
  using degree_container_type = typename base_type::degree_container_type;
  using metrics_type          = typename base_type::metrics_type;
  using timer_type            = typename base_type::timer_type;

  using dist_type         = DistType;
  using neighbor_type     = std::pair<index_type, dist_type>;
  using neighborhood_type = std::vector<neighbor_type>;
  using neighborhood_container_type =
      ygm::container::map<index_type, neighborhood_type>;

 public:
  handler(ygm::comm &comm, const parameters_type &params)
      : base_type(comm, params) {}
};

}  // namespace hdknn

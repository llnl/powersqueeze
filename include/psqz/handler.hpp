// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/utils/metrics.hpp>
#include <psqz/utils/spawner.hpp>

#include <ygm/container/set.hpp>
#include <ygm/utility/timer.hpp>

#include <krowkee/cereal/eigen.hpp>

#include <Eigen/Dense>

#include <iomanip>

namespace psqz {
template <typename ParametersType, std::size_t RangeSize,
          std::size_t ReplicationCount, typename AdjacencyType,
          typename FeatureType = float, typename CmtyType = std::size_t>
class handler {
 public:
  using parameters_type = ParametersType;
  using feature_type    = FeatureType;
  using cmty_type       = CmtyType;

  using adjacency_type = AdjacencyType;

  using index_type         = typename adjacency_type::index_type;
  using weight_type        = typename adjacency_type::weight_type;
  using index_vec_type     = typename adjacency_type::index_vec_type;
  using adjacency_elt_type = std::pair<index_type, weight_type>;
  using adjacency_vec_type =
      typename adjacency_type::vector_type<adjacency_elt_type>;

  using truth_type            = ygm::container::map<index_type, cmty_type>;
  using query_type            = ygm::container::set<index_type>;
  using sketch_container_type = typename adjacency_type::container_type<
      index_type, Eigen::Vector<feature_type, Eigen::Dynamic>>;
  using feature_vec_type = typename sketch_container_type::mapped_type;
  using degree_type      = float;
  using degree_container_type =
      typename adjacency_type::container_type<index_type, degree_type>;
  using metrics_type = Metrics;
  using timer_type   = ygm::utility::timer;

  static constexpr std::size_t range_size        = RangeSize;
  static constexpr std::size_t replication_count = ReplicationCount;
  static constexpr std::size_t register_count = range_size * replication_count;

 protected:
  ygm::comm             &_comm;
  const parameters_type &_params;
  metrics_type           _metrics;
  timer_type             _timer;

 public:
  handler(ygm::comm &comm, const parameters_type &params)
      : _comm(comm), _params(params), _metrics(), _timer() {}

  ygm::comm             &comm() { return _comm; }
  const parameters_type &params() { return _params; }

  template <typename T>
  void chirp_line(const std::string &name, const T val) {
    if (_params.verbose()) {
      std::stringstream ss;
      ss << std::setw(50) << std::right << name << " -- " << val;
      _comm.cout0(ss.str());
    }
  }

  void chirp_metric(const std::string name) {
    auto time = _timer.elapsed();
    _metrics.set(name, time);
    chirp_line(name, time);
    _timer.reset();
  }

  template <typename T>
  void chirp_metric(const std::string name, const T val) {
    _metrics.set(name, val);
    chirp_line(name, val);
  }

  void repeat_metrics() {
    std::stringstream ss;
    ss << _params.csv_names() << "," << csv_metric_names(_metrics) << std::endl
       << _params.csv_values() << "," << csv_metric_vals(_metrics);
    _comm.cout0(ss.str());
  }

  void reset_timer() { _timer.reset(); }

  template <typename Point>
  static typename adjacency_type::container_type<index_type, Point> spawn(
      ygm::comm &comm, const Point &dummy, const index_type size) {
    return psqz::spawn<typename adjacency_type::container_type, index_type,
                       Point>(comm, dummy, size);
  }
};

}  // namespace psqz

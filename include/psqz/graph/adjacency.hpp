// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#if __has_include(<metall/metall.hpp>)
#include <psqz/metall.hpp>
#endif

#include <psqz/graph/graph.hpp>
#include <psqz/utils/spawner.hpp>

#include <ygm/comm.hpp>

#include <cmath>
#include <sstream>

namespace psqz::graph {

template <template <typename, typename> class ContainerType = psqz::ygm_array,
          template <typename> class VecType                 = std::vector,
          typename IndexType = std::size_t, typename WeightType = float>
struct square_undirected_adjacency {
  template <typename T>
  using vector_type = VecType<T>;

  template <typename IndexT, typename VecT>
  using container_type = ContainerType<IndexT, VecT>;

  using index_type         = IndexType;
  using weight_type        = WeightType;
  using index_vec_type     = vector_type<index_type>;
  using edge_type          = edge<index_type, weight_type>;
  using adjacency_elt_type = std::pair<index_type, weight_type>;
  using adjacency_vec_type = vector_type<adjacency_elt_type>;

  using row_container_type = container_type<index_type, adjacency_vec_type>;

 private:
  row_container_type _row_container;

  static constexpr auto insert_lambda =
      [](const index_type &row_idx, adjacency_vec_type &row_adj,
         const adjacency_elt_type &elt) { row_adj.push_back(elt); };

  static constexpr adjacency_vec_type _get_default(
      const std::size_t vertex_count) {
    adjacency_vec_type default_value{};
    default_value.reserve(
        static_cast<std::size_t>(__builtin_clz(vertex_count)));
    return default_value;
  };

 public:
  square_undirected_adjacency(ygm::comm &comm, std::size_t vertex_count)
      : _row_container(
            psqz::spawn<container_type, index_type, adjacency_vec_type>(
                comm, _get_default(vertex_count), vertex_count)) {}

  ygm::comm &comm() { return _row_container.comm(); }

  std::size_t size() { return _row_container.size(); }

  template <typename... Args>
  void for_all_rows(Args &&...args) {
    _row_container.for_all(args...);
  }

  template <typename... Args>
  void for_all_cols(Args &&...args) {
    _row_container.for_all_cols(args...);
  }

  template <typename... Args>
  void async_insert_edge(const edge_type &edge, Args &...args) {
    _row_container.async_visit(edge.src, insert_lambda,
                               adjacency_elt_type{edge.dst, edge.wgt});
    _row_container.async_visit(edge.dst, insert_lambda,
                               adjacency_elt_type{edge.src, edge.wgt});
  }

  template <typename... Args>
  void local_row_visit(const index_type idx, Args &&...args) {
    _row_container.local_visit(idx, args...);
  }

  template <typename... Args>
  void local_col_visit(const index_type idx, Args &&...args) {
    _row_container.local_visit(idx, args...);
  }
};
}  // namespace psqz::graph

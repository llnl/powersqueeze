// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <type_traits>

namespace psqz::sketch::interleaved {

template <typename AdjacencyType, typename SketchContainerType>
void spMV(AdjacencyType &adjacency, SketchContainerType &current_sketch,
          SketchContainerType &next_sketch) {
  using index_type         = typename AdjacencyType::index_type;
  using adjacency_vec_type = typename AdjacencyType::adjacency_vec_type;
  using adjacency_elt_type = typename AdjacencyType::adjacency_elt_type;
  using weight_type        = typename AdjacencyType::weight_type;
  using feature_vec_type   = typename SketchContainerType::mapped_type;
  using feature_type       = typename feature_vec_type::value_type;
  static_assert(
      std::is_same<index_type, typename SketchContainerType::key_type>());
  static_assert(
      std::is_same<index_type, typename adjacency_elt_type::first_type>());

  auto next_sketch_ptr = next_sketch.get_ygm_ptr();

  current_sketch.for_all(
      [&adjacency, &next_sketch](const index_type       &col_idx,
                                 const feature_vec_type &col_sketch) {
        // This assumes that A and current_sketch share the same partitioning
        // scheme.
        auto adj_visitor = [&next_sketch](const index_type         &col_idx,
                                          const adjacency_vec_type &col_adj,
                                          const feature_vec_type &col_sketch) {
          for (const adjacency_elt_type &row : col_adj) {
            const index_type row_idx = row.first;
            next_sketch.async_visit(
                row_idx,
                [](const index_type &row_idx, feature_vec_type &row_sketch,
                   const feature_vec_type &col_sketch) {
                  row_sketch += col_sketch;
                },
                col_sketch);
          }
        };
        adjacency.local_row_visit(col_idx, adj_visitor, col_sketch);
      });

  current_sketch.comm().barrier();
}
}  // namespace psqz::sketch::interleaved

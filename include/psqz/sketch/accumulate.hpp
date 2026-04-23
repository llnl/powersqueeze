// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <krowkee/sketch.hpp>

#include <type_traits>

namespace psqz::sketch {

template <std::size_t RangeSize, std::size_t ReplicationCount,
          typename AdjacencyType, typename SketchContainerType>
void accumulate(AdjacencyType &adjacency, SketchContainerType &SAp1,
                const std::uint64_t &random_seed) {
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

  using sketch_type =
      krowkee::sketch::SparseJLT<feature_type, RangeSize, ReplicationCount,
                                 std::shared_ptr>;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  // For reasons unknown, there has to be seed passed to this
  // constructor or fails stealthily
  transform_ptr_type transform_ptr(
      std::make_shared<transform_type>(random_seed));
  sketch_type col_sketch(transform_ptr);

  adjacency.for_all_rows([&col_sketch, &SAp1](
                             const index_type         &col_idx,
                             const adjacency_vec_type &col_adj) {
    col_sketch.clear();
    for (const adjacency_elt_type &row : col_adj) {
      const index_type  &row_idx = row.first;
      const weight_type &wgt     = row.second;
      col_sketch.insert(row_idx, wgt);
    }

    // assumes that adjacency and SAp1 share the same partitioning scheme.
    // induces unnecessary copies, but not clear if there is an easy way to
    // avoid them.
    auto update_lambda = [](const index_type &col_idx, feature_vec_type &sketch,
                            const feature_vec_type &col_sketch) {
      std::transform(std::begin(sketch), std::end(sketch),
                     std::begin(col_sketch), std::begin(sketch),
                     std::plus<feature_type>());
    };
    SAp1.local_visit(col_idx, update_lambda, col_sketch.scaled_registers());
  });

  SAp1.comm().barrier();
}

}  // namespace psqz::sketch

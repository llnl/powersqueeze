// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/sketch/accumulate.hpp>

#include <krowkee/sketch.hpp>

#include <Eigen/Dense>

#include <type_traits>

namespace psqz::sketch {
template <std::size_t RangeSize, std::size_t ReplicationCount,
          typename MatrixType, typename AdjacencyType,
          typename SketchContainerType, std::size_t FinalRangeSize = RangeSize,
          std::size_t FinalReplicationCount = ReplicationCount>
std::vector<MatrixType> accumulate_matrices(AdjacencyType       &adjacency,
                                            SketchContainerType &SAp1,
                                            const std::uint64_t &random_seed,
                                            const int transform_count) {
  using index_type         = typename AdjacencyType::key_type;
  using adjacency_vec_type = typename AdjacencyType::mapped_type;
  using adjacency_elt_type = typename adjacency_vec_type::value_type;
  using weight_type        = typename adjacency_elt_type::second_type;
  using feature_vec_type   = typename SketchContainerType::mapped_type;
  using feature_type       = typename feature_vec_type::value_type;
  static_assert(
      std::is_same<index_type, typename SketchContainerType::key_type>());
  static_assert(
      std::is_same<index_type, typename adjacency_elt_type::first_type>());

  using single_sketch_type =
      krowkee::sketch::SparseJLT<feature_type, RangeSize, ReplicationCount,
                                 std::shared_ptr>;
  using single_transform_type = typename single_sketch_type::transform_type;
  using single_transform_ptr_type =
      typename single_sketch_type::transform_ptr_type;

  using double_sketch_type =
      krowkee::sketch::DoubleSparseJLT<feature_type, RangeSize,
                                       ReplicationCount, std::shared_ptr>;
  using double_transform_type = typename double_sketch_type::transform_type;
  using double_transform_ptr_type =
      typename double_sketch_type::transform_ptr_type;

  using final_double_sketch_type =
      krowkee::sketch::DoubleSparseJLT<feature_type, RangeSize,
                                       ReplicationCount, std::shared_ptr,
                                       FinalRangeSize, FinalReplicationCount>;
  using final_double_transform_type =
      typename final_double_sketch_type::transform_type;
  using final_double_transform_ptr_type =
      typename final_double_sketch_type::transform_ptr_type;
  using final_single_transform_type =
      typename final_double_transform_type::col_transform_type;
  using final_single_transform_ptr_type =
      typename final_double_transform_type::col_transform_ptr_type;

  static_assert(
      std::is_same<single_transform_type,
                   typename double_transform_type::row_transform_type>::value);
  static_assert(
      std::is_same<single_transform_type,
                   typename double_transform_type::col_transform_type>::value);

  static_assert(
      std::is_same<
          typename double_transform_type::col_transform_type,
          typename final_double_transform_type::row_transform_type>::value);

  YGM_ASSERT_RELEASE(transform_count > 0 && transform_count < 10);

  const int single_transform_count = transform_count - 1;
  const int double_transform_count = transform_count - 2;

  ygm::comm &comm = SAp1.comm();

  // We create a vector of shared pointers for each of the individual
  // sketch transforms.
  std::vector<single_transform_ptr_type> single_transform_ptrs;
  for (int i(0); i < single_transform_count; ++i) {
    single_transform_ptrs.push_back(
        std::make_shared<single_transform_type>(random_seed + i));
  }
  // Using these shared pointers, we now create a vector of pointers to all of
  // the two-sided sketch transforms.
  std::vector<double_transform_ptr_type> double_transform_ptrs;
  for (int i(0); i < double_transform_count; ++i) {
    double_transform_ptrs.push_back(std::make_shared<double_transform_type>(
        single_transform_ptrs[i], single_transform_ptrs[i + 1]));
  }

  final_single_transform_ptr_type final_single_transform(
      std::make_shared<final_single_transform_type>(random_seed +
                                                    transform_count));
  final_double_transform_ptr_type final_double_transform(
      std::make_shared<final_double_transform_type>(
          single_transform_ptrs.back(), final_single_transform));

  // We also create local double-sided sketches that will hold the double
  // sided embeddings.
  std::vector<double_sketch_type> double_sketches;
  for (const double_transform_ptr_type &double_transform_ptr :
       double_transform_ptrs) {
    double_sketches.emplace_back(double_transform_ptr);
  }
  final_double_sketch_type final_double_sketch(final_double_transform);

  single_sketch_type col_sketch(single_transform_ptrs[0]);

  adjacency.for_all([&col_sketch, &SAp1, &double_sketches,
                     &final_double_sketch](const index_type         &col_idx,
                                           const adjacency_vec_type &col_adj) {
    col_sketch.clear();
    for (const adjacency_elt_type &row : col_adj) {
      const index_type  &row_idx = row.first;
      const weight_type &wgt     = row.second;
      col_sketch.insert(row_idx, wgt);
      for (double_sketch_type &double_sketch : double_sketches) {
        double_sketch.insert({row_idx, col_idx}, wgt);
      }
      final_double_sketch.insert({row_idx, col_idx}, wgt);
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
  comm.barrier();

  // We create the parallel two-sided matrices
  std::vector<MatrixType> double_matrices;

  // We dump the contents of the S^tAR embeddings to Eigen matrices. Each rank
  // will hold their local updates, and then we allreduce the matrices so that
  // each rank holds the fully sketched matrices.
  for (int i(0); i < double_sketches.size(); ++i) {
    // using a const reference to avoid an extra copy
    const MatrixType &double_matrix =
        double_sketches[i].container().registers();
    // it is very important that the dummy matrix have the the correct shapes!
    double_matrices.push_back(
        MatrixType::Zero(double_matrix.rows(), double_matrix.cols()));
    YGM_ASSERT_MPI(MPI_Allreduce(
        double_matrix.data(), double_matrices[i].data(),
        double_matrix.rows() * double_matrix.cols(),
        ygm::detail::mpi_typeof(feature_type()), MPI_SUM, comm.get_mpi_comm()));
    // apply scaling factor
    double_matrices[i] /= double_transform_type::scaling_factor;
  }
  // repeat for final matrix
  const MatrixType &final_double_matrix =
      final_double_sketch.container().registers();
  double_matrices.push_back(
      MatrixType::Zero(final_double_matrix.rows(), final_double_matrix.cols()));
  YGM_ASSERT_MPI(MPI_Allreduce(
      final_double_matrix.data(), double_matrices.back().data(),
      final_double_matrix.rows() * final_double_matrix.cols(),
      ygm::detail::mpi_typeof(feature_type()), MPI_SUM, comm.get_mpi_comm()));
  // apply scaling factor
  double_matrices.back() /= final_double_transform_type::scaling_factor;

  return double_matrices;
}
}  // namespace psqz::sketch

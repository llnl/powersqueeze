// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/utils/parameters.hpp>

#include <ygm/comm.hpp>
#include <ygm/detail/collective.hpp>

#include <sstream>

namespace example {
template <typename ParametersType>
struct parameters : public ParametersType {
  using base_type = ParametersType;

  psqz::parameter<std::string> file_directory;
  psqz::parameter<bool>        all_exponents;

  parameters()
      : base_type(),
        file_directory("file_directory", "Path for all outfiles", 'f', true,
                       ""),
        all_exponents("all_exponents?",
                      "Indicates whether to print all exponent sketches", 'E',
                      false, false) {
    this->_params.push_back(&file_directory);
    this->_params.push_back(&all_exponents);
  }

  bool _help_needed() const { return base_type::_help_needed(); }
};
}  // namespace example

template <typename HandlerType, typename SketchContainerType>
void sketch_stats(HandlerType &handler, SketchContainerType &sketch_container,
                  const std::string name) {
  using index_type       = typename SketchContainerType::key_type;
  using feature_vec_type = typename SketchContainerType::mapped_type;
  using feature_type     = typename feature_vec_type::value_type;

  std::size_t nonzeros{0};
  std::size_t empties{0};
  sketch_container.for_all(
      [&nonzeros, &empties](const index_type       &idx,
                            const feature_vec_type &sketch) {
        bool empty{true};
        for (const feature_type &feature : sketch) {
          if (feature != 0.0) {
            ++nonzeros;
            empty = false;
          }
        }
        if (empty) {
          ++empties;
        }
      });
  handler.comm().barrier();

  std::stringstream ss;
  ss << name << " density";
  handler.chirp_metric(ss.str(), float(ygm::sum(nonzeros, handler.comm())) /
                                     (handler.params().vertex_count() *
                                      handler.params().range_size() *
                                      handler.params().replication_count()));
  std::stringstream se;
  se << name << " empty rows";
  handler.chirp_metric(se.str(), ygm::sum(empties, handler.comm()));
}

std::string sketch_name(int exponent) {
  std::stringstream ss;
  ss << "SAp" << exponent;
  return ss.str();
}

template <typename HandlerType, typename SketchContainerType,
          typename ParametersType>
void sketch_accounting(HandlerType          &handler,
                       SketchContainerType  &sketch_container,
                       const ParametersType &params, const int exponent,
                       const int all_exponents) {
  std::string name(sketch_name(exponent));
  handler.chirp_metric(name + " accumulate time");
  sketch_stats(handler, sketch_container, name);
  if (!params.file_directory().empty()) {
    if (all_exponents || (!all_exponents && exponent == params.exponent())) {
      std::filesystem::path path(params.file_directory());
      psqz::create_directory_if_not_exists(path);
      psqz::create_directory_if_not_exists(path / "features");
      psqz::write_feature_files(sketch_container, path, name);
      handler.chirp_metric(name + " write time");
    }
  }
}

template <typename HandlerType, typename SketchContainerType,
          typename ParametersType>
void sketch_accounting(HandlerType          &handler,
                       SketchContainerType  &sketch_container,
                       const ParametersType &params, const int exponent) {
  sketch_accounting(handler, sketch_container, params, exponent,
                    params.all_exponents());
}

template <typename HandlerType, typename AdjacencyType, typename TruthType>
void adjacency_report(HandlerType &handler, AdjacencyType &adjacency,
                      TruthType &truth) {
  using index_type         = typename AdjacencyType::index_type;
  using adjacency_vec_type = typename AdjacencyType::adjacency_vec_type;
  using adjacency_elt_type = typename AdjacencyType::adjacency_elt_type;
  using weight_type        = typename AdjacencyType::weight_type;
  using cmty_type          = typename TruthType::mapped_type;
  static_assert(std::is_same<index_type, typename TruthType::key_type>());
  static_assert(
      std::is_same<index_type, typename adjacency_elt_type::first_type>());

  ygm::comm &world = adjacency.comm();

  static index_type intra_edge_count{0};
  static index_type inter_edge_count{0};
  std::size_t       edge_count{0};
  std::size_t       adj_empties{0};
  adjacency.for_all_rows(
      [&truth, &edge_count, &adj_empties](const index_type         &col_idx,
                                          const adjacency_vec_type &col_adj) {
        edge_count += col_adj.size();
        if (col_adj.size() == 0) {
          ++adj_empties;
        }
        for (const adjacency_elt_type &row : col_adj) {
          const index_type &row_idx = row.first;
          truth.async_visit(
              row_idx,
              [](auto map_ptr, const index_type &row_idx,
                 const cmty_type &row_cmty, const index_type &col_idx) {
                map_ptr->async_visit(
                    col_idx,
                    [](const index_type &col_idx, const cmty_type &col_cmty,
                       const cmty_type row_cmty) {
                      if (row_cmty == col_cmty) {
                        ++intra_edge_count;
                      } else {
                        ++inter_edge_count;
                      }
                    },
                    row_cmty);
              },
              col_idx);
        }
      });
  world.barrier();

  std::size_t vertices          = adjacency.size();
  std::size_t edges             = ygm::sum(edge_count, world);
  std::size_t intra_edges       = ygm::sum(intra_edge_count, world);
  std::size_t inter_edges       = ygm::sum(inter_edge_count, world);
  std::size_t adjacency_empties = ygm::sum(adj_empties, world);

  ygm::container::map<cmty_type, std::size_t> cmty_counts(world);
  truth.for_all([&cmty_counts](const index_type &idx, const cmty_type &cmty) {
    cmty_counts.async_visit(
        cmty, [](const cmty_type &cmty, std::size_t &count) { ++count; });
  });
  world.barrier();
  std::size_t communities = cmty_counts.size();
  world.barrier();

  std::size_t min_cmty_size  = -1;
  std::size_t max_cmty_size  = 0;
  double      mean_cmty_size = 0;
  cmty_counts.for_all([&min_cmty_size, &max_cmty_size, &mean_cmty_size](
                          const cmty_type &cmty, const std::size_t &count) {
    if (count < min_cmty_size) {
      min_cmty_size = count;
    }
    if (count > max_cmty_size) {
      max_cmty_size = count;
    }
    mean_cmty_size += count;
  });
  world.barrier();
  min_cmty_size  = ygm::min(min_cmty_size, world);
  max_cmty_size  = ygm::max(max_cmty_size, world);
  mean_cmty_size = ygm::sum(mean_cmty_size, world) / communities;
  world.barrier();

  handler.chirp_metric("vertices", vertices);
  handler.chirp_metric("edges", edges);
  handler.chirp_metric("communities", communities);
  handler.chirp_metric("minimum community size", min_cmty_size);
  handler.chirp_metric("maximum community size", max_cmty_size);
  handler.chirp_metric("mean community size", mean_cmty_size);
  handler.chirp_metric("adjacency empty rows", adjacency_empties);
  handler.chirp_metric("intra-community edges", intra_edges);
  handler.chirp_metric("inter-community edges", inter_edges);
  handler.chirp_metric("vertex/community ratio", float(vertices) / communities);
  handler.chirp_metric("edge/vertex ratio", float(edges) / vertices);
  handler.chirp_metric("intra/inter ratio", float(intra_edges) / inter_edges);
}

template <typename HandlerType, typename AdjacencyType>
void adjacency_report(HandlerType &handler, AdjacencyType &adjacency) {
  using index_type         = typename AdjacencyType::index_type;
  using adjacency_vec_type = typename AdjacencyType::adjacency_vec_type;
  using adjacency_elt_type = typename AdjacencyType::adjacency_elt_type;
  using weight_type        = typename AdjacencyType::weight_type;
  static_assert(
      std::is_same<index_type, typename adjacency_elt_type::first_type>());

  ygm::comm &world = adjacency.comm();

  std::size_t edge_count{0};
  std::size_t adj_empties{0};
  adjacency.for_all_rows(
      [&edge_count, &adj_empties](const index_type         &col_idx,
                                  const adjacency_vec_type &col_adj) {
        edge_count += col_adj.size();
        if (col_adj.size() == 0) {
          ++adj_empties;
        }
      });
  world.barrier();

  std::size_t vertices          = adjacency.size();
  std::size_t edges             = ygm::sum(edge_count, world);
  std::size_t adjacency_empties = ygm::sum(adj_empties, world);

  handler.chirp_metric("vertices", vertices);
  handler.chirp_metric("edges", edges);
  handler.chirp_metric("adjacency empty rows", adjacency_empties);
}
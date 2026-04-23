// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#if __has_include(<metall/metall.hpp>)
#include <psqz/metall.hpp>
#endif

#include <ygm/comm.hpp>

#include <cmath>
#include <sstream>

namespace psqz::graph {

namespace detail {
template <typename HandlerType, template <typename> class ReaderType>
struct edge_streamer {
  using handler_type       = HandlerType;
  using parameters_type    = typename handler_type::parameters_type;
  using index_type         = typename handler_type::index_type;
  using index_vec_type     = typename handler_type::index_vec_type;
  using cmty_type          = typename handler_type::cmty_type;
  using weight_type        = typename handler_type::weight_type;
  using adjacency_elt_type = typename handler_type::adjacency_elt_type;
  using adjacency_vec_type = typename handler_type::adjacency_vec_type;
  using adjacency_type     = typename handler_type::adjacency_type;
  using truth_type         = typename handler_type::truth_type;
  using container_type     = adjacency_type;
  using element_type       = adjacency_elt_type;
  using vector_type        = adjacency_vec_type;
  using edge_type          = typename adjacency_type::edge_type;
  using reader_type        = ReaderType<edge_type>;

 protected:
  handler_type          &_handler;
  ygm::comm             &_comm;
  const parameters_type &_params;

 public:
  edge_streamer(handler_type &handler)
      : _handler(handler), _comm(_handler.comm()), _params(_handler.params()) {}

  ygm::comm             &comm() { return _comm; }
  handler_type          &handler() { return _handler; }
  const parameters_type &params() { return _params; }

  virtual std::string name() const { return "adjacency"; }

  virtual std::string local_map_name() const = 0;

  virtual reader_type spawn_reader() = 0;

  template <typename... Args>
  adjacency_type operator()(Args &...args) {
    return load_adjacency();
  }

  virtual adjacency_type load_adjacency() {
    reader_type reader = spawn_reader();

    adjacency_type adjacency(_comm, _params.vertex_count());

    reader.for_all([&adjacency](const edge_type &edge) {
      adjacency.async_insert_edge(edge);
    });
    _comm.barrier();
    _handler.chirp_metric(name() + " read time");
    return adjacency;
  }
};

template <typename HandlerType>
struct adjacency_normalizer {
  using handler_type          = HandlerType;
  using parameters_type       = typename handler_type::parameters_type;
  using index_type            = typename handler_type::index_type;
  using feature_type          = typename handler_type::feature_type;
  using index_vec_type        = typename handler_type::index_vec_type;
  using cmty_type             = typename handler_type::cmty_type;
  using weight_type           = typename handler_type::weight_type;
  using adjacency_elt_type    = typename handler_type::adjacency_elt_type;
  using adjacency_vec_type    = typename handler_type::adjacency_vec_type;
  using adjacency_type        = typename handler_type::adjacency_type;
  using truth_type            = typename handler_type::truth_type;
  using degree_type           = typename handler_type::degree_type;
  using degree_container_type = typename handler_type::degree_container_type;
  using container_type        = adjacency_type;
  using element_type          = adjacency_elt_type;
  using vector_type           = adjacency_vec_type;

 protected:
  handler_type          &_handler;
  ygm::comm             &_comm;
  const parameters_type &_params;

 public:
  adjacency_normalizer(handler_type &handler)
      : _handler(handler), _comm(_handler.comm()), _params(_handler.params()) {}

  ygm::comm             &comm() { return _comm; }
  handler_type          &handler() { return _handler; }
  const parameters_type &params() { return _params; }

  constexpr std::string name() const { return "normalized adjacency"; }

  virtual std::string local_map_name() const = 0;

  // A stand-in for loading from metall. Should only be used in that context.
  adjacency_type operator()() {
    return adjacency_type{_comm, _params.vertex_count()};
  }

  // This uses O(m) communication, which could be a problem for big graphs.
  adjacency_type operator()(adjacency_type &adjacency_hat) {
    adjacency_hat.for_all([&adjacency_hat](const index_type   &col_idx,
                                           adjacency_vec_type &col_adj) {
      degree_type col_degree{0.0};
      for (const adjacency_elt_type &elt : col_adj) {
        col_degree += elt.second;
      }
      degree_type col_dhinv = 1.0 / std::sqrt(col_degree);
      for (adjacency_elt_type &elt : col_adj) {
        index_type  &row_idx = elt.first;
        weight_type &wgt     = elt.second;
        elt.second *= col_degree;
        adjacency_hat.async_visit(
            row_idx,
            [](const index_type &row_idx, adjacency_vec_type &row_adj,
               const index_type &col_idx, const degree_type &col_dhinv) {
              for (adjacency_elt_type &elt : row_adj) {
                index_type &idx = elt.first;
                if (idx == col_idx) {
                  weight_type &wgt = elt.second;
                  wgt *= col_dhinv;
                  break;
                }
              }
            },
            col_idx, col_dhinv);
      }
    });
    this->_comm.barrier();
    this->_handler.chirp_metric(name() + " normalize time");

    return adjacency_hat;
  }
};

}  // namespace detail

template <typename HandlerType, template <typename> class EdgeStreamerType>
struct adjacency_streamer {
  using handler_type   = HandlerType;
  using adjacency_type = typename handler_type::adjacency_type;
#if __has_include(<metall/metall.hpp>)
  using edge_streamer_func =
      psqz::mtl::vectorized_wrapper<EdgeStreamerType<handler_type>>;
#else
  using edge_streamer_func = EdgeStreamerType<handler_type>;
#endif

 protected:
  edge_streamer_func _edge_streamer_fn;

 public:
  adjacency_streamer(handler_type &handler) : _edge_streamer_fn(handler) {}

  adjacency_type operator()() { return _edge_streamer_fn(); }
};

template <typename HandlerType, template <typename> class EdgeStreamerType,
          template <typename> class AdjacencyNormalizerType>
struct normalized_adjacency_streamer {
  using handler_type   = HandlerType;
  using adjacency_type = typename handler_type::adjacency_type;
#if __has_include(<metall/metall.hpp>)
  using adjacency_func =
      psqz::mtl::vectorized_wrapper<EdgeStreamerType<handler_type>>;
  using normalize_func =
      psqz::mtl::vectorized_wrapper<AdjacencyNormalizerType<handler_type>>;
  using mtl_map_type = typename adjacency_func::mtl_map_type;
#else
  using edge_streamer_func        = EdgeStreamerType<handler_type>;
  using adjacency_normalizer_func = AdjacencyNormalizerType<handler_type>;
#endif

 protected:
  edge_streamer_func        _edge_streamer_fn;
  adjacency_normalizer_func _adjacency_normalizer_fn;

 public:
  normalized_adjacency_streamer(handler_type &handler)
      : _edge_streamer_fn(handler), _adjacency_normalizer_fn(handler) {}

  adjacency_type operator()() {
#if __has_include(<metall/metall.hpp>)
    if (_adjacency_normalizer_fn.exists()) {
      return _adjacency_normalizer_fn();
    }
#endif
    adjacency_type adjacency = _edge_streamer_fn();
    return _adjacency_normalizer_fn(adjacency);
  }
};

}  // namespace psqz::graph

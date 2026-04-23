// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/graph/graph.hpp>

#include <ygm/comm.hpp>

namespace psqz::graph {

namespace detail {

template <typename HandlerType, template <typename> class ReaderType>
struct truth {
  using handler_type    = HandlerType;
  using parameters_type = typename handler_type::parameters_type;
  using index_type      = typename handler_type::index_type;
  using index_vec_type  = typename handler_type::index_vec_type;
  using cmty_type       = typename handler_type::cmty_type;
  using weight_type     = typename handler_type::weight_type;
  using truth_type      = typename handler_type::truth_type;
  using container_type  = truth_type;
  using element_type    = cmty_type;
  using edge_type       = edge<index_type, weight_type>;
  using reader_type     = ReaderType<edge_type>;

 protected:
  handler_type          &_handler;
  ygm::comm             &_comm;
  const parameters_type &_params;

 public:
  truth(handler_type &handler)
      : _handler(handler), _comm(_handler.comm()), _params(_handler.params()) {}

  ygm::comm             &comm() { return _comm; }
  handler_type          &handler() { return _handler; }
  const parameters_type &params() { return _params; }

  constexpr std::string name() const { return "truth"; }

  virtual std::string local_map_name() const = 0;

  template <typename... Args>
  truth_type spawn(Args... args) {
    return {_comm};
  }

  virtual reader_type spawn_reader() = 0;

  truth_type operator()() {
    truth_type  truth  = spawn();
    reader_type reader = spawn_reader();

    reader.for_all([&truth](const edge_type &edge) {
      truth.async_insert(edge.src, edge.dst);
    });

    _comm.barrier();
    _handler.chirp_metric(name() + " read time");
    return truth;
  }
};

template <typename HandlerType, template <typename> class ReaderType>
struct queries {
  using handler_type    = HandlerType;
  using parameters_type = typename handler_type::parameters_type;
  using query_type      = typename handler_type::query_type;
  using index_type      = typename handler_type::index_type;
  using cmty_type       = typename handler_type::cmty_type;
  using weight_type     = typename handler_type::weight_type;
  using container_type  = query_type;
  using element_type    = cmty_type;
  using edge_type       = edge<index_type, weight_type>;
  using reader_type     = ReaderType<edge_type>;

 protected:
  handler_type          &_handler;
  ygm::comm             &_comm;
  const parameters_type &_params;

 public:
  queries(handler_type &handler)
      : _handler(handler), _comm(_handler.comm()), _params(_handler.params()) {}

  ygm::comm             &comm() { return _comm; }
  handler_type          &handler() { return _handler; }
  const parameters_type &params() { return _params; }

  constexpr std::string name() const { return "query"; }

  template <typename... Args>
  query_type spawn(Args... args) {
    return {_comm};
  }

  virtual reader_type spawn_reader() = 0;

  query_type operator()() {
    query_type  queries = spawn();
    reader_type reader  = spawn_reader();

    reader.for_all(
        [&queries](const edge_type &edge) { queries.async_insert(edge.src); });

    _comm.barrier();
    _handler.chirp_metric(name() + " read time");
    return queries;
  }
};
}  // namespace detail

template <typename HandlerType, template <typename> class TruthFunc>
struct truth {
  using handler_type = HandlerType;
  using truth_type   = typename handler_type::truth_type;
#if __has_include(<metall/metall.hpp>)
  using truth_func = psqz::mtl::wrapper<TruthFunc<handler_type>>;
#else
  using truth_func = TruthFunc<handler_type>;
#endif

 protected:
  truth_func _truth_fn;

 public:
  truth(handler_type &handler) : _truth_fn(handler) {}

  truth_type operator()() { return _truth_fn(); }
};

template <typename HandlerType, template <typename> class QueryFunc>
struct queries {
  using handler_type = HandlerType;
  using query_type   = typename handler_type::query_type;
  using query_func   = QueryFunc<handler_type>;

 protected:
  query_func _query_fn;

 public:
  queries(handler_type &handler) : _query_fn(handler) {}

  query_type operator()() { return _query_fn(); }
};

}  // namespace psqz::graph

// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#include <hdknn/dnnd/parameters.hpp>
#include <hdknn/handler.hpp>
#include <hdknn/metrics/precision.hpp>
#include <hdknn/metrics/recall.hpp>
#include <hdknn/metrics/search_depth.hpp>

#include <psqz/tsv/graph.hpp>
#include <psqz/utils/reader.hpp>

#include <saltatlas/dnnd/dnnd.hpp>
#include <saltatlas/dnnd/utility.hpp>

#include <ygm/detail/collective.hpp>

struct parameters_type : public hdknn::dnnd::parameters<psqz::parameters> {
  using base_type = hdknn::dnnd::parameters<psqz::parameters>;

  psqz::parameter<int>         embedding_size;
  psqz::parameter<std::string> truth_index;
  psqz::parameter<std::string> features_index;
  psqz::parameter<std::string> dump_path;
  psqz::parameter<std::string> query_filename;

  parameters_type()
      : base_type(),
        embedding_size("embedding_size", "Size of the embedding", 'C', true, 0),
        truth_index("truth_index", "Path to truth index file", 'f', true, ""),
        features_index("features_index", "Path to features index file", 'F',
                       true, ""),
        query_filename("query_filename",
                       "Optional file containing indices to query", 'q', true,
                       ""),
        dump_path("dump_path", "Optional path to dump output files.", 'd', true,
                  "") {
    this->_params.push_back(&embedding_size);
    this->_params.push_back(&truth_index);
    this->_params.push_back(&features_index);
    this->_params.push_back(&query_filename);
    this->_params.push_back(&dump_path);
  }

  bool _help_needed() const {
    if (!std::filesystem::exists(truth_index()) &&
        !std::filesystem::exists(this->query_filename())) {
      std::cout << "DNND_only requires at least one of a truth index file and "
                   "a query index file, whereas specified truth file "
                << truth_index() << " and query file " << this->query_filename()
                << " do not exist." << std::endl;
      return true;
    }
    if (!std::filesystem::exists(features_index())) {
      std::cout << "DNND_only requires a truth index file, whereas "
                << features_index() << " does not exist." << std::endl;
      return true;
    }
    if (embedding_size() < 1) {
      std::cout << "DNND only requires a valid embedding size, not "
                << embedding_size() << std::endl;
      return true;
    }
    return base_type::_help_needed();
  }
};

constexpr auto parse_cmd_line = psqz::parse_cmd_line<parameters_type>;

struct dnnd_only {
  using handler_type =
      hdknn::handler<parameters_type, 8, 1, psqz::ygm_map, std::vector, float,
                     std::size_t, float, std::size_t>;

  using query_fn = psqz::tsv::queries<handler_type>;

  using precision_fn    = hdknn::metric::precision<handler_type>;
  using recall_fn       = hdknn::metric::recall<handler_type>;
  using search_depth_fn = hdknn::metric::search_depth<handler_type>;

  using index_type            = handler_type::index_type;
  using feature_type          = handler_type::feature_type;
  using feature_vec_type      = handler_type::feature_vec_type;
  using cmty_type             = handler_type::cmty_type;
  using dist_type             = handler_type::dist_type;
  using adjacency_elt_type    = handler_type::adjacency_elt_type;
  using adjacency_vec_type    = handler_type::adjacency_vec_type;
  using adjacency_type        = handler_type::adjacency_type;
  using truth_type            = handler_type::truth_type;
  using query_type            = handler_type::query_type;
  using sketch_container_type = handler_type::sketch_container_type;

  using dnnd_type = saltatlas::dnnd<index_type, feature_vec_type, dist_type>;
  using neighbor_store_type = typename dnnd_type::neighbor_store_type;
  using neighbor_type       = typename handler_type::neighbor_type;
  using neighborhood_type   = typename handler_type::neighborhood_type;
  using neighborhood_container_type =
      typename handler_type::neighborhood_container_type;

  void operator()(ygm::comm &world, const parameters_type &params) const {
    // the `handler`is a convenience struct that holds all of the relevant
    // types and serves as "glue" that holds powersqueeze workflows together.
    // A `handler_type` object interfaces between the ygm::comm, recorded
    // metrics, the parameters, and all powersqueeze and hdknn functors.
    handler_type handler{world, params};

    truth_type truth(world);
    if (std::filesystem::exists(params.truth_index())) {
      psqz::read_truth(truth, params.truth_index());
      world.barrier();
      handler.chirp_metric("truth load time");
      handler.chirp_metric("truth count", truth.size());
    }

    // collect the queries - might want to make this configurable in the future.
    query_type queries(world);
    if (std::filesystem::exists(params.query_filename())) {
      queries = query_fn{handler}();
    } else {
      truth.for_all([&queries](const index_type &idx, const cmty_type &cmty) {
        queries.async_insert(idx);
      });
    }
    world.barrier();
    handler.chirp_metric("queries load time");
    handler.chirp_metric("queries count", queries.size());

    sketch_container_type sketch_container(world);
    psqz::read_features(sketch_container, params.embedding_size(),
                        params.features_index());
    world.barrier();
    handler.chirp_metric("features load time");

    handler.chirp_metric("feature vector count", sketch_container.size());

    // perform dnnd workflow
    //
    // this workflow is encapsulated in the `hdknn::dnnd::knn_index` functor
    // found in `hdknn/dnnd/knn_index.hpp`. however, the workflow has been
    // replicated explicitly here to illustrate how to interface between
    // powersqueeze objects and downstream analytics.
    //
    // note that throughout we are using an old version of the dnnd API.

    dnnd_type dnnd{saltatlas::distance::id::sql2, world};

    // here we create a point store and populate it with the elements in the
    // sketch container.
    handler.reset_timer();
    dnnd.add_points(sketch_container);
    world.barrier();
    handler.chirp_metric("point store population time");

    // here we construct the dnnd index using the populated point stores.
    handler.reset_timer();
    dnnd.build(params.nn_count(), params.rho(), params.delta(),
               params.batch_size());
    world.barrier();
    handler.chirp_metric("index build time");

    // here we optimize the index.
    handler.reset_timer();
    dnnd.optimize(params.make_index_undirected(),
                  params.pruning_degree_multiplier());
    world.barrier();
    handler.chirp_metric("index optimize time");

    // now we construct the query stores.
    handler.reset_timer();
    std::vector<index_type> query_indices;
    queries.for_all([&query_indices](const index_type &idx) {
      query_indices.push_back(idx);
    });
    world.barrier();
    handler.chirp_metric("query populate time");

    // now we perform the queries.
    handler.reset_timer();
    const std::unordered_map<index_type,
                             std::vector<typename dnnd_type::neighbor_type>>
        query_neighborhoods =
            dnnd.get_neighbors(query_indices.begin(), query_indices.end());
    world.barrier();
    handler.chirp_metric("query dnnd time");

    YGM_ASSERT_RELEASE(ygm::sum(query_indices.size(), handler.comm()) ==
                       ygm::sum(query_neighborhoods.size(), handler.comm()));

    // finally, we convert the neighbor store into a ygm container mapping
    // indices to their neighbor's indices and distances.
    handler.reset_timer();
    neighborhood_container_type nbhd_map{world};
    for (const std::pair<index_type,
                         std::vector<typename dnnd_type::neighbor_type>>
             &query_neighborhood : query_neighborhoods) {
      const index_type &query = query_neighborhood.first;
      const std::vector<typename dnnd_type::neighbor_type> &neighborhood =
          query_neighborhood.second;
      nbhd_map.async_visit(
          query,
          [](const index_type &qry_idx, neighborhood_type &nbhd,
             const auto &nns, const int &k) {
            int counter{0};
            for (const auto &nb : nns) {
              if (counter++ >= k) {
                break;
              }
              const dist_type  &distance = nb.distance;
              const index_type &nbr_idx  = nb.id;
              nbhd.push_back({nbr_idx, distance});
            }
          },
          neighborhood, params.nn_query());
    }
    world.barrier();
    handler.chirp_metric("neighborhood collect time");

    // at long last, compute knn quality metrics using hdknn implementations
    //
    // note that, internally, these metrics ignore the "self" neighbor, as in
    // this setting we are always querying the neighborhoods of points
    // (vertices) that are already in the data structure. if a future use case
    // queries for novel points, then these implementations will need to be
    // modified.

    if (std::filesystem::exists(params.truth_index())) {
      precision_fn{handler}(truth, queries, nbhd_map);
      recall_fn{handler}(truth, queries, nbhd_map);
      search_depth_fn{handler}(truth, queries, nbhd_map);
    }

    if (!params.dump_path().empty()) {
      // we have to re-localize the neighborhoods in order to use saltaltas dump
      // utilities. This is brittle and should be replaced.
      ygm::container::map<index_type, index_type> query_orders(handler.comm());
      int                                         order{0};
      queries.for_all([&query_orders, &order](const index_type &idx) {
        query_orders.async_insert(idx, order++);
      });
      static neighbor_store_type neighbor_store{query_indices.size()};
      static const int           nn_query = params.nn_query();
      world.barrier();
      for (const std::pair<index_type,
                           std::vector<typename dnnd_type::neighbor_type>>
               &query_neighborhood : query_neighborhoods) {
        const index_type &query = query_neighborhood.first;
        const std::vector<typename dnnd_type::neighbor_type> &neighborhood =
            query_neighborhood.second;
        query_orders.async_visit(
            query,
            [](const index_type &query, const index_type &order,
               const std::vector<typename dnnd_type::neighbor_type>
                   &neighborhood) {
              for (int i{0}; i < neighborhood.size(); ++i) {
                if (i >= nn_query) {
                  break;
                }
                neighbor_store[order].push_back(neighborhood[i]);
              }
            },
            neighborhood);
      }
      world.barrier();
      handler.chirp_metric("neighborhood collect time");

      // at long last, dump neighborhoods to file.
      saltatlas::utility::gather_and_dump_neighbors(
          neighbor_store, params.dump_path(), handler.comm());
      handler.chirp_metric("data dump time");
    }

    handler.repeat_metrics();
  }
};

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    if (world.rank0()) {
      psqz::repeat_cmd_line(argc, argv);
    }

    // The parameters are a simple struct that hold values specified on the
    // command line at runtime that affect execution decisions.
    // this specific type, `hdknn::dnnd::kron::sketch::parameters`, is
    // tailored to kronecker product two tsv inputs in the graph challenge
    // style and includes power iteration parameters, such as the range size
    // and the exponent, as well as dnnd-specific parameters.
    //
    // if you have a different input data type or have additional parameters
    // to read from the CLI in your workflow, you will need to create your own
    // `parameters_type` that inherits from this or `psqz::parameters`.
    parameters_type params = parse_cmd_line(argc, argv);

    if (params.verbose()) {
      world.cout0(params);
    }

    dnnd_only{}(world, params);
  }
}

// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#include <hdknn/dhnsw/parameters.hpp>
#include <hdknn/handler.hpp>
#include <hdknn/metrics/precision.hpp>
#include <hdknn/metrics/recall.hpp>
#include <hdknn/metrics/search_depth.hpp>

#include <saltatlas/dhnsw/dhnsw.hpp>

#include <psqz/graph/adjacency.hpp>
#include <psqz/utils/reader.hpp>

#include <ygm/detail/collective.hpp>

struct parameters_type : public hdknn::dhnsw::parameters<psqz::parameters> {
  using base_type = hdknn::dhnsw::parameters<psqz::parameters>;

  psqz::parameter<int>         embedding_size;
  psqz::parameter<std::string> truth_index;
  psqz::parameter<std::string> features_index;

  parameters_type()
      : base_type(),
        embedding_size("embedding_size", "Size of the embedding", 'C', true, 0),
        truth_index("truth_index", "Path to truth index file", 'f', true, ""),
        features_index("features_index", "Path to features index file", 'F',
                       true, "") {
    this->_params.push_back(&embedding_size);
    this->_params.push_back(&truth_index);
    this->_params.push_back(&features_index);
  }

  bool _help_needed() const {
    if (!std::filesystem::exists(truth_index())) {
      std::cout << "DNND only requires a truth index file, whereas "
                << truth_index() << " does not exist." << std::endl;
      return true;
    }
    if (!std::filesystem::exists(features_index())) {
      std::cout << "DNND only requires a truth index file, whereas "
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

float my_l2_sqr(const Eigen::Vector<float, Eigen::Dynamic> &x,
                const Eigen::Vector<float, Eigen::Dynamic> &y) {
  if (x.size() != y.size()) {
    std::cerr << "Size mismatch for l2 distance (" << x.size() << " vs "
              << y.size() << ")" << std::endl;
    exit(-1);
  }

  return (x - y).squaredNorm();
}

struct dhnsw_only {
  using adjacency_type =
      psqz::graph::square_undirected_adjacency<psqz::ygm_map, std::vector,
                                               std::size_t, float>;
  using handler_type = hdknn::handler<parameters_type, 8, 1, adjacency_type,
                                      float, std::size_t, float>;

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
  using truth_type            = handler_type::truth_type;
  using query_type            = handler_type::query_type;
  using sketch_container_type = handler_type::sketch_container_type;

  using dhnsw_type = saltatlas::dhnsw<index_type, feature_vec_type, dist_type>;
  using neighbor_type     = typename handler_type::neighbor_type;
  using neighborhood_type = typename handler_type::neighborhood_type;
  using neighborhood_container_type =
      typename handler_type::neighborhood_container_type;

  void operator()(ygm::comm &world, const parameters_type &params) const {
    // the `handler`is a convenience struct that holds all of the relevant
    // types and serves as "glue" that holds powersqueeze workflows together.
    // A `handler_type` object interfaces between the ygm::comm, recorded
    // metrics, the parameters, and all powersqueeze and hdknn functors.
    handler_type handler{world, params};

    truth_type truth(world);
    psqz::read_truth(truth, params.truth_index());
    world.barrier();
    handler.chirp_metric("truth load time");
    handler.chirp_metric("truth count", truth.size());

    // collect the queries - might want to make this configurable in the future.
    query_type queries(world);
    truth.for_all([&queries](const index_type &idx, const cmty_type &cmty) {
      queries.async_insert(idx);
    });
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

    dhnsw_type dhnsw{my_l2_sqr, world};

    // here we create a point store and populate it with the elements in the
    // sketch container.
    handler.reset_timer();
    dhnsw.add_points(sketch_container);
    world.barrier();
    handler.chirp_metric("point store population time");

    // here we construct the dnnd index using the populated point stores.
    handler.reset_timer();
    dhnsw.build();
    world.barrier();
    handler.chirp_metric("index build time");

    // now we construct the query stores.
    handler.reset_timer();
    static std::vector<index_type>       query_indices;
    static std::vector<feature_vec_type> query_features;
    queries.for_all([&sketch_container](const index_type &idx) {
      sketch_container.async_visit(
          idx, [](const index_type &idx, const feature_vec_type &sketch) {
            query_indices.push_back(idx);
            query_features.push_back(sketch);
          });
    });
    world.barrier();
    handler.chirp_metric("query populate time");

    const std::vector<std::vector<typename dhnsw_type::neighbor_type>>
        query_neighborhoods = dhnsw.query(
            query_features.begin(), query_features.end(), params.nn_query());
    world.barrier();
    handler.chirp_metric("query dhnsw time");

    YGM_ASSERT_RELEASE(query_indices.size() == query_neighborhoods.size());
    YGM_ASSERT_RELEASE(ygm::sum(query_indices.size(), world) ==
                       ygm::sum(query_neighborhoods.size(), world));

    neighborhood_container_type nbhd_map(world);

    for (std::size_t i{0}; i < query_neighborhoods.size(); ++i) {
      nbhd_map.async_visit(
          query_indices[i],
          [](const index_type &qry_idx, neighborhood_type &nbhd,
             const auto &nns) {
            for (const auto &nb : nns) {
              const dist_type  &distance = nb.distance;
              const index_type &nbr_idx  = nb.id;
              nbhd.push_back({nbr_idx, distance});
            }
          },
          query_neighborhoods[i]);
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

    precision_fn{handler}(truth, queries, nbhd_map);
    recall_fn{handler}(truth, queries, nbhd_map);
    search_depth_fn{handler}(truth, queries, nbhd_map);

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

    dhnsw_only{}(world, params);
  }
}

// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

// comment out if files are 0-indexed.
#define TSV_DECREMENT

#include <psqz/handler.hpp>
#include <psqz/sketch/accumulate.hpp>
#include <psqz/sketch/interleaved.hpp>
#include <psqz/sketch/parameters.hpp>
#include <psqz/tsv/graph.hpp>
#include <psqz/tsv/truth.hpp>
#include <psqz/utils/writer.hpp>

#include <krowkee/util/runtime.hpp>

#include <common.hpp>

// Add a output prefix parameter.
struct parameters_type : public psqz::sketch::tsv::parameters {
  using base_type = psqz::sketch::tsv::parameters;

  psqz::parameter<std::string> file_directory;

  parameters_type()
      : base_type(),
        file_directory("file_directory", "Path for all outfiles", 'f', true,
                       "") {
    this->_params.push_back(&file_directory);
  }

  bool _help_needed() const { return base_type::_help_needed(); }
};

constexpr auto parse_cmd_line = psqz::parse_cmd_line<parameters_type>;

// This is an ad-hoc implementation of the Jaccard index that treats the last
// dimension of the points as its size.
template <typename point_type, typename distance_type>
inline distance_type approx_jaccard_index(const point_type &p0,
                                          const point_type &p1) {
  assert(p0.size() == p1.size());
  std::size_t   size = p0.size();
  distance_type ip   = 0.0;
  distance_type ss0  = p0[size - 1] * p0[size - 1];
  distance_type ss1  = p1[size - 1] * p1[size - 1];
  for (std::size_t i = 0; i < size - 1; ++i) {
    ip += p0[i] * p1[i];
  }

  distance_type              approx_jaccard = (ip / (ss0 + ss1 - ip));
  static const distance_type eps            = 1e-10;
  if (approx_jaccard < eps) {
    approx_jaccard = eps;
  } else if (approx_jaccard > 1.0) {
    return 0.0;
  }

  return 1.0 - approx_jaccard;
}

// In this example we contain the workflow into the functor
// `jaccard_dnnd_kron` because it is a convenient way to instrument the
// workflows for different compile-time sketch parameters, such as the
// `RangeSize` and `ReplicationCount` that is set by the CLI flag `-r` and `-R`.
// For fixed sketch size parameters, one could simple write
// `jaccard_dnnd_kron::operator()` as the body of the main function.
template <std::size_t RangeSize, std::size_t ReplicationCount>
struct jaccard_tsv {
  using handler_type =
      psqz::handler<parameters_type, RangeSize, ReplicationCount,
                    psqz::ygm_array, std::vector, float, std::size_t, float,
                    std::size_t>;

  using adjacency_fn = psqz::tsv::adjacency<handler_type>;
  using truth_fn     = psqz::tsv::truth<handler_type>;
  using query_fn     = psqz::tsv::queries<handler_type>;

  using index_type            = handler_type::index_type;
  using feature_type          = handler_type::feature_type;
  using feature_vec_type      = handler_type::feature_vec_type;
  using cmty_type             = handler_type::cmty_type;
  using adjacency_elt_type    = handler_type::adjacency_elt_type;
  using adjacency_vec_type    = handler_type::adjacency_vec_type;
  using adjacency_type        = handler_type::adjacency_type;
  using truth_type            = handler_type::truth_type;
  using query_type            = handler_type::query_type;
  using sketch_container_type = handler_type::sketch_container_type;

  void operator()(ygm::comm &world, const parameters_type &params) const {
    // the `handler`is a convenience struct that holds all of the relevant types
    // and serves as "glue" that holds powersqueeze workflows together.
    // A `handler_type` object interfaces between the ygm::comm, recorded
    // metrics, the parameters, and all powersqueeze and hdknn functors.
    handler_type handler{world, params};

    // collect the symmetric adjacency matrix from file.
    //
    // this implementation assumes that the data is formatted like HPEC graph
    // challenge data.
    //
    // `adjacency_fn` is a functor that takes the `handler` and, upon
    // invocation, returns an `adjacency_type` object, which is a ygm container
    // with `index_type` keys and `adjacency_vec_type` values. Internally, it
    // reads from a tsv file listing edges in an SBM graph.
    //
    // This is currently a dense representation of the adjacency matrix, so for
    // very large, very dense, or very skewed degree distribution graphs this
    // could be a bottleneck in the workflow.
    //
    // if you have a different input data type, you need only create a new I/O
    // wrapped in an `adjacency` class that inherits from
    // `psqz::graph::adjacency` to use this same workflow, possibly in addition
    // to a new `parameters` class to handle new parameters of your I/O.
    adjacency_type adjacency = adjacency_fn{handler}();

    // report information
    //
    // Here we use some relatively uninteresting (and therefore obfuscated)
    // boilerplate to record statistics of the adjacency matrix.
    ygm::utility::timer timer{};
    adjacency_report(handler, adjacency);
    handler.chirp_metric("reporting time", timer.elapsed());
    timer.reset();
    handler.reset_timer();

    // Perform the sketch workflow.
    //
    // Here we accumulate a one-sided sketch of the adjacency matrix. In this
    // case we use a ygm::container::array, so we create dummy vector to
    // populate the array with zeros upon creation to eliminate memory
    // reallocation during the accumulation. the values of this container are
    // the truncated, power-iteration-embedded vectors that can be used in
    // downstream metric applications.
    feature_vec_type dummy(handler_type::register_count *
                           handler_type::replication_count);

    sketch_container_type SAp1(world, params.vertex_count(), dummy);
    psqz::sketch::accumulate<RangeSize, ReplicationCount>(adjacency, SAp1,
                                                          params.random_seed());
    handler.chirp_metric("SAp1 accumulate time");

    // This is an ad-hoc solution where we append the size to the end of each
    // sketch vector in order to approximate the Jaccard index.
    adjacency.for_all(
        [&SAp1](const index_type &idx, const adjacency_vec_type &adj_vec) {
          SAp1.async_visit(
              idx,
              [](const index_type &idx, feature_vec_type &sketch,
                 const feature_type &size) { sketch.push_back(size); },
              adj_vec.size());
        });
    handler.comm().barrier();

    // report information

    sketch_stats(handler, SAp1, "SAp1");

    if (!params.file_directory().empty()) {
      std::filesystem::path path(params.file_directory());
      psqz::create_directory_if_not_exists(path);
      psqz::create_directory_if_not_exists(path / "features");
      psqz::write_feature_files(SAp1, path, "SAp1");
      if (std::filesystem::exists(params.truth_filename())) {
        truth_type truth = truth_fn{handler}();
        world.barrier();
        psqz::write_truth_files(truth, path);
      }
    }

    handler.repeat_metrics();
  }
};

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    world.welcome();
    if (world.rank0()) {
      psqz::repeat_cmd_line(argc, argv);
    }

    // The parameters are a simple struct that hold values specified on the
    // command line at runtime that affect execution decisions.
    // this specific type, `hdknn::dnnd::tsv::sketch::parameters`, is tailored
    // to tsv inputs in the graph challenge style and includes power iteration
    // parameters, such as the range size and the exponent, as well as
    // dnnd-specific parameters.
    //
    // if you have a different input data type or have additional parameters to
    // read from the CLI in your workflow, you will need to create your own
    // `parameters_type` that inherits from this or `psqz::parameters`.
    parameters_type params = parse_cmd_line(argc, argv);

    if (params.verbose()) {
      world.cout0(params);
    }

    // `krowkee::dispatch` is a convenience function that dispatches a
    // compile-time sized version of the sketch workflow using runtime
    // parameters.
    krowkee::dispatch<jaccard_tsv, void>{
        params.range_size(), params.replication_count()}(world, params);
  }
}

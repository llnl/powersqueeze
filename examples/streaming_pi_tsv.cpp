// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

// comment out if files are 0-indexed.
#define TSV_DECREMENT

#include <psqz/graph/adjacency.hpp>
#include <psqz/handler.hpp>
#include <psqz/sketch/accumulate_matrices.hpp>
#include <psqz/sketch/interleaved.hpp>
#include <psqz/sketch/parameters.hpp>
#include <psqz/tsv/graph.hpp>
#include <psqz/tsv/truth.hpp>
#include <psqz/utils/writer.hpp>

#include <psqz/sketch/chebyshev.hpp>

#include <krowkee/util/runtime.hpp>

#include <common.hpp>

#include <filesystem>
#include <unordered_map>

// Add a output prefix parameter.
using parameters_type =
    example::parameters<psqz::sketch::streaming::tsv::parameters>;

constexpr auto parse_cmd_line = psqz::parse_cmd_line<parameters_type>;

// In this example we contain the workflow into the functor
// `streaming_pi_tsv` because it is a convenient way to instrument the
// workflows for different compile-time sketch parameters, such as the
// `RangeSize` and `ReplicationCount` that are set by the CLI flags `-r` and
// `-R`. For fixed sketch size parameters, one could simple write
// `streaming_pi_tsv::operator()` as the body of the main function.
template <std::size_t RangeSize, std::size_t ReplicationCount,
          std::size_t FinalRangeSize        = RangeSize,
          std::size_t FinalReplicationCount = ReplicationCount>
struct streaming_pi_tsv {
  using adjacency_type =
      psqz::graph::square_undirected_adjacency<psqz::ygm_array, std::vector,
                                               std::size_t, float>;
  using handler_type =
      psqz::handler<parameters_type, RangeSize, ReplicationCount,
                    adjacency_type, float, std::size_t>;

  using adjacency_streamer_fn = psqz::tsv::adjacency_streamer<handler_type>;
  using truth_fn              = psqz::tsv::truth<handler_type>;

  using index_type            = handler_type::index_type;
  using feature_type          = handler_type::feature_type;
  using feature_vec_type      = handler_type::feature_vec_type;
  using cmty_type             = handler_type::cmty_type;
  using adjacency_vec_type    = handler_type::adjacency_vec_type;
  using adjacency_elt_type    = handler_type::adjacency_elt_type;
  using truth_type            = handler_type::truth_type;
  using sketch_container_type = handler_type::sketch_container_type;

  using matrix_type =
      Eigen::Matrix<feature_type, Eigen::Dynamic, Eigen::Dynamic>;
  using vector_type = Eigen::Vector<feature_type, Eigen::Dynamic>;
  using vector_container_type =
      typename adjacency_type::container_type<index_type, vector_type>;

  void operator()(ygm::comm &world, const parameters_type &params) const {
    // the `handler`is a convenience struct that holds all of the relevant types
    // and serves as "glue" that holds powersqueeze workflows together.
    // A `handler_type` object interfaces between the ygm::comm, recorded
    // metrics, the parameters, and all powersqueeze functors.
    handler_type handler{world, params};

    // collect the symmetric adjacency matrix from file.
    //
    // this implementation assumes that the data is formatted like HPEC graph
    // challenge data in tsv file(s).
    //
    // `adjacency_streamer_fn` is a functor that takes the `handler` and, upon
    // invocation, returns an `adjacency_type` object, which is a ygm container
    // with `index_type` keys and `adjacency_vec_type` values.
    //
    // if you have a different input data type, you need only create a new I/O
    // wrapped in an `adjacency` class that inherits from
    // `psqz::graph::adjacency` to use this same workflow, possibly in addition
    // to a new `parameters_type` class to handle parameters of your I/O.
    adjacency_type adjacency = adjacency_streamer_fn{handler}();

    // collect the ground truth
    //
    // this implementation assumes that the data is formatted like HPEC graph
    // challenge data tsv file(s).
    //
    // `truth_fn` is a functor that takes the `handler` and, upon invocation,
    // returns a `truth_type` object, which is a ygm container with `index_type`
    // keys and `cmty_type` values. this container maps each vertex to its
    // ground truth community.
    //
    // if you have a different input data type, you need only create a new I/O
    // wrapped in a `truth` class that inherits from `psqz::graph::truth` to use
    // this same workflow, possibly in addition to a new `parameters` class to
    // hander new parameters of your I/O.
    truth_type truth = truth_fn{handler}();
    if (!params.file_directory().empty()) {
      std::filesystem::path path(params.file_directory());
      psqz::create_directory_if_not_exists(path);
      psqz::write_truth_files(truth, path);
    }

    // report information
    //
    // Here we use some relatively uninteresting (and therefore obfuscated)
    // boilerplate to record statistics of the adjacency matrix.
    ygm::utility::timer timer{};
    adjacency_report(handler, adjacency, truth);
    handler.chirp_metric("reporting time", timer.elapsed());
    timer.reset();
    handler.reset_timer();

    // Perform the sketch workflow
    //
    // Here we accumulate a one-sided sketch of the adjacency matrix. In this
    // case we use a ygm::container::array, so we create dummy vector to
    // populate the array with zeros upon creation to eliminate memory
    // reallocation during the accumulation.

    feature_vec_type dummy(handler_type::register_count);

    sketch_container_type this_sketch(world, params.vertex_count(), dummy);
    psqz::sketch::accumulate<RangeSize, ReplicationCount>(
        adjacency, this_sketch, params.random_seed());
    sketch_accounting(handler, this_sketch, params, 1, false);

    std::vector<matrix_type> matrices = psqz::sketch::accumulate_matrices<
        RangeSize, ReplicationCount, matrix_type, adjacency_type,
        sketch_container_type, FinalRangeSize, FinalReplicationCount>(
        adjacency, this_sketch, params.random_seed(), params.exponent());

    // Here we perform the in-place matrix multiplications to compute streaming
    // sketches of the powers of the adjacency matrix to the desired target
    // power. Unlike `power_iteration_tsv`, this does not support printing
    // intermediate powers.
    matrix_type partial_product = matrices.back();
    for (int i = matrices.size() - 2; i >= 0; --i) {
      partial_product = matrices[i] * partial_product;
    }

    vector_container_type print_embedding(
        world, params.vertex_count(),
        vector_type::Zero(FinalRangeSize * FinalReplicationCount));
    this_sketch.for_all(
        [&print_embedding, &partial_product, &params](
            const index_type &idx, const feature_vec_type &sketch) {
          vector_type first_embedding =
              vector_type::Zero(handler_type::register_count);
          for (int i(0); i < sketch.size(); ++i) {
            first_embedding(static_cast<Eigen::Index>(i)) = (sketch[i]);
          }
          vector_type final_embedding =
              first_embedding.transpose() * partial_product;
          print_embedding.local_insert(idx, final_embedding);
        });
    world.barrier();
    sketch_accounting(handler, print_embedding, params, params.exponent(),
                      false);

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
    // this specific type, `psqz::sketch::tsv::parameters`, is tailored to tsv
    // inputs in the graph challenge style and includes power iteration
    // parameters, such as the range size and the exponent.
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
    // krowkee::dispatch<streaming_pi_tsv, void>(params.range_size(),
    // params.replication_count()}(world, params);
    krowkee::dispatch_rectangular<streaming_pi_tsv, void>{
        params.range_size(), params.replication_count(),
        params.final_range_size(),
        params.final_replication_count()}(world, params);
  }
}

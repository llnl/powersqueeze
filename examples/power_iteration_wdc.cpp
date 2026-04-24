// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#include <psqz/graph/adjacency.hpp>
#include <psqz/handler.hpp>
#include <psqz/sketch/accumulate.hpp>
#include <psqz/sketch/interleaved.hpp>
#include <psqz/sketch/parameters.hpp>
#include <psqz/tsv/graph.hpp>
#include <psqz/tsv/truth.hpp>
#include <psqz/utils/writer.hpp>

#include <krowkee/util/runtime.hpp>

#include <common.hpp>

#include <filesystem>
#include <unordered_map>

namespace wdc {

template <typename IndexType, typename CmtyType>
bool try_read_line(std::pair<IndexType, CmtyType> &pair, std::ifstream &ifs) {
  std::string line;
  IndexType   idx;
  CmtyType    tld;
  if (std::getline(ifs, line)) {
    std::stringstream ss(line);
    ss >> tld >> idx;
    pair = {idx, tld};
    return true;
  } else {
    return false;
  }
}

template <typename IndexType, typename CmtyType>
struct reader {
  using input_type = std::string;
  using index_type = IndexType;
  using cmty_type  = CmtyType;

  reader(ygm::comm &comm, const input_type &filename,
         const bool verbose = false)
      : _comm(comm), _bag(comm), _filled(false), _verbose(verbose) {
    if (!filename.empty()) {
      fill_bag(_bag, filename);
      _filled = true;
    }
  }

  template <typename Function>
  void for_all(Function fn) {
    if (!_filled) {
      throw std::logic_error("reader has no filenmae!");
    }
    auto read_file_lambda = [&fn](const input_type &filename) {
      std::ifstream ifs(filename);
      if (!ifs.good()) {
        std::cerr << "error opening filename: " << filename << std::endl;
      }
      std::pair<index_type, cmty_type> pair;
      while (try_read_line(pair, ifs)) {
        fn(pair);
      }
    };
    _bag.for_all(read_file_lambda);
  }

  void fill_bag(ygm::container::bag<input_type> &bag,
                const input_type                &filename) const {
    if (bag.comm().rank0()) {
      if (psqz::tsv::is_tsv(filename) == false) {
        if (_verbose) {
          std::cout << "File " << filename << " is not a tsv. Assuming it "
                    << "contains a list of tsv files." << std::endl;
        }

        std::ifstream ifs(filename.c_str());
        std::string   line;
        while (std::getline(ifs, line)) {
          std::stringstream ss(line);
          std::string       fname;
          ss >> fname;
          bag.async_insert(fname);
        }
      } else {
        bag.async_insert(filename);
      }
    }
    bag.comm().barrier();
  }

  ygm::comm &comm() { return _comm; }

 private:
  ygm::comm                      &_comm;
  ygm::container::bag<input_type> _bag;
  bool                            _filled;
  bool                            _verbose;
};

template <typename HandlerType>
struct truth {
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
  using container_type     = truth_type;
  using reader_type        = reader<index_type, cmty_type>;

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

  std::string local_map_name() const {
    return map_name(this->name(), this->_params);
  }

  template <typename... Args>
  truth_type spawn(Args... args) {
    return {_comm};
  }

  reader_type spawn_reader() {
    return {this->_comm, this->_params.truth_filename()};
  };

  truth_type operator()() {
    truth_type  truth  = spawn();
    reader_type reader = spawn_reader();

    reader.for_all([&truth](const std::pair<index_type, cmty_type> &pair) {
      const index_type &idx  = pair.first;
      const cmty_type  &cmty = pair.second;
      truth.async_insert(idx, cmty);
    });

    _comm.barrier();
    _handler.chirp_metric(name() + " read time");
    return truth;
  }
};
}  // namespace wdc

// Add a output prefix parameter.
using parameters_type = example::parameters<psqz::sketch::tsv::parameters>;

constexpr auto parse_cmd_line = psqz::parse_cmd_line<parameters_type>;

// In this example we contain the workflow into the functor
// `power_iteration_tsv` because it is a convenient way to instrument the
// workflows for different compile-time sketch parameters, such as the
// `RangeSize` and `ReplicationCount` that are set by the CLI flags `-r` and
// `-R`. For fixed sketch size parameters, one could simple write
// `power_iteration_tsv::operator()` as the body of the main function.
template <std::size_t RangeSize, std::size_t ReplicationCount>
struct power_iteration_tsv {
  using adjacency_type =
      psqz::graph::square_undirected_adjacency<psqz::ygm_array, std::vector,
                                               std::size_t, float>;
  using handler_type =
      psqz::handler<parameters_type, RangeSize, ReplicationCount,
                    adjacency_type, float, std::size_t>;

  using adjacency_streamer_fn = psqz::tsv::adjacency_streamer<handler_type>;
  using truth_fn              = wdc::truth<handler_type>;

  using index_type            = handler_type::index_type;
  using feature_type          = handler_type::feature_type;
  using feature_vec_type      = handler_type::feature_vec_type;
  using cmty_type             = handler_type::cmty_type;
  using adjacency_vec_type    = handler_type::adjacency_vec_type;
  using adjacency_elt_type    = handler_type::adjacency_elt_type;
  using truth_type            = handler_type::truth_type;
  using sketch_container_type = handler_type::sketch_container_type;

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
    // `truth_fn` is a functor that takes the `hander` and, upon invocation,
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
    feature_vec_type dummy(
        feature_vec_type::Zero(handler_type::register_count));

    sketch_container_type this_sketch(world, params.vertex_count(), dummy);
    psqz::sketch::accumulate<RangeSize, ReplicationCount>(
        adjacency, this_sketch, params.random_seed());
    sketch_accounting(handler, this_sketch, params, 1);

    // Here we perform iterative sparse matrix-multivector multiplications to
    // accumulate sketches of the powers of the adjacency matrix, repeating up
    // to the desired target power.
    int exponent{1};
    while (++exponent <= params.exponent()) {
      sketch_container_type next_sketch(world, params.vertex_count(), dummy);
      handler.reset_timer();
      psqz::sketch::interleaved::spMV(adjacency, this_sketch, next_sketch);
      sketch_accounting(handler, next_sketch, params, exponent);
      this_sketch.local_swap(next_sketch);
      handler.chirp_metric(sketch_name(exponent) + " swap time");
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
    krowkee::dispatch<power_iteration_tsv, void>{
        params.range_size(), params.replication_count()}(world, params);
  }
}

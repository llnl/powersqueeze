# powersqueeze examples

This directory includes examples illustrating the use of `powersqueeze` to
construct a low-dimensional vector representation of graph vertices.
Most features in the `psqz` namespace are functors organized around a specific
operation, such as loading an adjacency matrix from inputs or computing the
power iteration given the adjacency matrix.
See the code in, e.g., `power_iteration_tsv.cpp` for inline descriptions of the
types and operations.

## power_iteration_tsv.cpp

This file illustrates `powersqueeze`'s basic functionality on tsv data that is
organized like the streaming partition challenge datasets from the HPEC
[graph challenge dataset](https://graphchallenge.mit.edu/data-sets).
There are really only two functors at play: and `adjacency_fn` functor that
constructs the adjacency matrix and a `sketch_fn` functor that performs the
power iteration.
There are many implementations of both in the `psqz` package, depending on the
specific format of the data and the desired power iteration specifics.
This example expects graph challenge-style tsv inputs, so it uses
`psqz::tsv::adjacency`, which inherits from `psqz::graph::adjacency`.
Supporting a different input file format is as simple as defining a new class
that inherits from `psqz::graph::adjacency` and fills in the I/O details.

### CLI

You can view the CLI of the `power_iteration_tsv` executable by running

```
$ ./examples/power_iteration_tsv -h
Usage:
 -h             Indicates whether to print usage string
 -c <arg>       Number of vertices in the input graph
 -z <arg>       Random seed
 -B <arg>       Buffer size used during exponentiation (default \infty)
 -M <arg>       Metall datastore path (optional)
 -u             Indicates that the graph stream is undirected
 -m             Indicates ygm::map use (ygm::array otherwise)
 -V             Indicates whether to print verbose output
 -i <arg>       File containing data to build index (required)
 -q <arg>       Optional file containing indices to query
 -g <arg>       File containing true communities
 -r <arg>       Power of 2 size of projection range
 -R <arg>       Power of 2 number of replicated projections
 -e <arg>       Integral power of data matrix
 -C             Perform chebyshev polynomial embedding
 -f <arg>       Path for all outfiles
 -E             Indicates whether to print all exponent sketches
```

The `-i` flag indicates the file containing the input.
This file can consist of either a single tsv file, or a file whose lines each
list the fully-qualified path of a tsv file.
The latter case is useful for parallel I/O in situations where the graph stream
is very large.

Similarly, the `-g` flag indicates the file containing the ground truth
assignments of vertex indices to communities.
This code does not make use of `-g`, but some argument is still required due to
how the `parameters` struct was implemented.

The `-u` flag instructs the I/O to replicate the reverse edge complementing each
edge that is read, i.e. on reading the edge `(i, j)` also insert the edge
`(j, i)`.
This flag should be used for Graph Challenge data.

The `-r` flag must be a power of two, and represents the number of dimensions
to be used for each countsketch projection. 
The `-R` flag indicates the number of countsketch tiles to be used, so `-r` *
`-R` is the number of dimensions into which the vertices will be projected.
The `-e` flag is the exponent to which the adjacency matrix will be raised
during power iteration.
The cost of additional powers becomes constant after two or three iterations
once the projection vectors become dense, and in practice it is not helpful
(and is sometimes even detrimental) to go above 4.

The `-c` argument is required if you want `powersqueeze` to use
`ygm::container::array`s as its distributed data structure for the power
iteration workflow.
This example is hard-coded to use arrays, but it is also possible to use 
`ygm::container::map`, which case `-c` is no longer required.
The map-based workflow is more realistic for online applications where you
don't know the size of your dataset ahead of time, but it makes the linear
algebra noticeably slower than using arrays.

The `-f` and `-l` arguments, if specified, indicate path prefixex to which the
outfiles containing the created features and the true labels will be written,
respectively.
`-E` tells the code to print outfiles for each approximate exponent of the
adjacency matrix, not just the final power.

The `-B`, `-M`, `-q`, `-m`, and `-C` arguments are not used by this experiment,
as the choices that they represent have been hard-coded in the source for the
sake of clarity.

### Outfiles

Using the `-f` argument will induce the example to write the feature
representation (points) to the provided path prefix.
The `-l` argument is similar for the true communities.
For example, if you pass `-f path/to/points/` and `-l path/to/labels/` to an
instance running on `N` mpi ranks, each rank `i` will write two files:
`path/to/points/i_features.txt` and `path/to/labels/i_labels.txt`.

Each row of a `*_labels.txt` file has the format `index label`.
Each row of a `*_features.txt` file has the format
`index feature1 feature2 feature3...`.

NOTE: Presently, the `*_labels.txt` and `*_features.txt` files do not share an
index partitioning scheme for reasons that are mysterious to me.
I am confident, however that the indices are correct.

### Example invocation

An example invocation on LC, using the prepared saltaltas benchmarks, might look
like

```
srun -N 1 --tasks-per-node 36 -p pbatch -A seq ./examples/power_iteration_tsv \
  -i /p/lustre1/salta/benchmarks/graphchallenge/2017/5000/parts.txt \
  -g /p/lustre1/salta/benchmarks/graphchallenge/2017/5000/gt_parts.txt \
  -c 5000 -r 8 -e 3 -u -V \
  -f /path/to/points/ -l /path/to/labels/
```

This can be made a bit simpler using some convenience scripts within the repo,
namely

```
srun -N 1 --tasks-per-node 36 -p pbatch -A seq \
  $(../scripts/gc_parts.sh ./examples/power_iteration_tsv 5000) \
  -r 8 -e 3 -u -V \
  -f /path/to/points/ -l /path/to/labels/
```

This script simply auto-fills the `-i`, `-g`, `-q`, and `-c` fields based upon
the hard-coded paths of the saltatlas benchmarks.
You need only specify a valid number of vertices (e.g. 5000, 50000, 200000, etc)
that occurs in the graph challenge dataset stored on LC.

## power_iteration_kron.cpp

## power_iteration_dnnd_kron.cpp

This file illustrates `powersqueeze`'s basic functionality on Kronecker graphs
- graphs whose topology is the result of a Kronecker product of two smaller
graph matrices.
This implementation performs Kronecker products of tsv data that is
organized like the streaming partition challenge datasets from the HPEC
[graph challenge dataset](https://graphchallenge.mit.edu/data-sets).

The `adjacency_fn` functor in this example consumes two input files and computes
a Kronecker product adjacency matrix, implemented in `psqz::kron::adjacency`.
In order to avoid making this matrix too dense, it stochastically thins the
matrix by sampling intra-community product edges and inter-community product
edges with user-specified probabilities that are given in the CLI.
It is possible to make these probabilities too small, which will result in
empty rows/columns of the adjacency matrix and will disrupt any downstream
metrics.
This code checks and notifies the user if there are any empty rows of the
adjacency matrix (or the sketch representations).

### CLI

You can view the CLI of the `power_iteration_kron` executable by running

```
$ ./examples/power_iteration_kron -h
Usage:
 -h             Indicates whether to print usage string
 -c <arg>       Number of vertices in the input graph
 -z <arg>       Random seed
 -B <arg>       Buffer size used during exponentiation (default \infty)
 -M <arg>       Metall datastore path (optional)
 -u             Indicates that the graph stream is undirected
 -m             Indicates ygm::map use (ygm::array otherwise)
 -V             Indicates whether to print verbose output
 -i <arg>       File containing data for left kronecker graph (required)
 -I <arg>       File containing data for right kronecker graph (required)
 -g <arg>       File containing community data for left kronecker graph (required)
 -G <arg>       File containing community data for right kronecker graph (required)
 -a <arg>       power (a>1.0) of expected mean degree log(|V(A)| * |V(B)|)^a
 -b <arg>       ratio (b>1.0) of intra-community to inter-community preservation
 -r <arg>       Power of 2 size of projection range
 -R <arg>       Power of 2 number of replicated projections
 -e <arg>       Integral power of data matrix
 -C             Perform chebyshev polynomial embedding
 -f <arg>       Path for all outfiles
 -E             Indicates whether to print all exponent sketches
```

Many of these arguments are explained in the description of
`power_iteration_tsv.cpp`.
Those explanations are not repeated here.
 
Here, the `-i` argument indicates the (single) file containing the left vertex's
edge stream, while the `-I` flag indicates the (single) file containing the
right vertex's edge stream.
The `-g` and `-G` arguments similarly indicate the left and right ground truth
files.

The `-a` and `-b` flags affect the probability of preserving intra- and
inter-cluster edges.
Both must be between > 1.0.
`-a` sets a target mean degree of the product graph at $(\log V_A * V_B)^a$,
where $a$ is the passed exponent.
The default value of `-a` is 1.4, and is probably fine for most use cases.
This target degree is used to compute the probability of preserving
intra community edges in the product graph. 
`-b` sets the factor by which inter community edges are preserved less than the
intra community edges, and the default value is 4.0.
If these values are too small, the resulting adjacency matrix could have empty
rows/columns.
Unfortunately, the threshold for "too small" is data dependent and so it depends
on experimentation.
The defaults appear fine in testing so far.

### Outfiles

Writing outfiles for `power_iteration_kron.cpp` uses the `-f` and `-l` flags in
the same way as `power_iteration_tsv.cpp`.

### Example invocation

An example invocation on LC, using the prepared saltaltas benchmarks, might look
like

```
srun -N 1 --tasks-per-node 36 -p pbatch -A seq ./examples/power_iteration_kron \
  -i /p/lustre1/salta/benchmarks/graphchallenge/2017/100/simulated_blockmodel_graph_100_nodes.tsv \
  -I /p/lustre1/salta/benchmarks/graphchallenge/2017/100/simulated_blockmodel_graph_100_nodes.tsv \
  -g /p/lustre1/salta/benchmarks/graphchallenge/2017/100/simulated_blockmodel_graph_100_nodes_truePartition.tsv \
  -G /p/lustre1/salta/benchmarks/graphchallenge/2017/100/simulated_blockmodel_graph_100_nodes_truePartition.tsv \
  -c 10000 -r 8 -e 3 -V -a 1.4 -b 4.0 \
  -f /path/to/points/ -l /path/to/labels/
```

This can be made a bit simpler using some convenience scripts within the repo,
namely

```
srun -N 1 --tasks-per-node 36 -p pbatch -A seq \
  $(../scripts/gc_kron.sh ./examples/hdknn/power_iteration_dnnd_kron 100 100) \
  -r 8 -e 3 -V -a 1.4 -b 4.0 \
  -f /path/to/points/ -l /path/to/labels/
```

This script simply auto-fills the `-i`, `-I`, `-g`, `-G`, and `-c` fields based
upon the hard-coded paths of the saltatlas benchmarks.
You need only specify two valid numbers of vertices
(e.g. 5000, 50000, 200000, etc) indicating the left and right graph challenge
graphs to use.
It is possible to compute products of a graph with itself, such as in the above
example.


## streaming_pi_tsv.cpp

This file illustrates `powersqueeze`'s fully streaming functionality on tsv data
that is organized like the streaming partition challenge datasets from the HPEC
[graph challenge dataset](https://graphchallenge.mit.edu/data-sets).
This workflow is more complex than that in `power_iteration_tsv.cpp`, and
involves computing moderately large square sketch matrices and multiplying them
together to obtain the final embedding.

### CLI

You can view the CLI of the `streaming_pi_tsv` executable by running

```
$ ./examples/streaming_pi_tsv -h
Usage:
 -h             Indicates whether to print usage string
 -c <arg>       Number of vertices in the input graph
 -z <arg>       Random seed
 -B <arg>       Buffer size used during exponentiation (default \infty)
 -M <arg>       Metall datastore path (optional)
 -u             Indicates that the graph stream is undirected
 -m             Indicates ygm::map use (ygm::array otherwise)
 -V             Indicates whether to print verbose output
 -i <arg>       File containing data to build index (required)
 -q <arg>       Optional file containing indices to query
 -g <arg>       File containing true communities
 -r <arg>       Power of 2 size of projection range
 -R <arg>       Power of 2 number of replicated projections
 -e <arg>       Integral power of data matrix
 -C             Perform chebyshev polynomial embedding
 -j <arg>       Power of 2 size of final projection range
 -J <arg>       Power of 2 number of replicated final projections
 -f <arg>       Path for all outfiles
 -E             Indicates whether to print all exponent sketches
```

The `-j` and `-J` flags are the only new additions, which indicate the range
size and replication count of the final projection dimension.
In general, one will want to set `-r` and `-R` to be fairly large and `-j` and
`-J` fairly small to maintain a sufficiently expressive sketch with small memory
footprint.
All of these flags must be a power of two.
Also, the `-E` flag is always ignored.

### Outfiles

The outfiles are the same as `power_iteration_tsv.cpp`, with the exception that
`streaming_pi_tsv` does not support the printing of intermediate powers. Thus,
only the final embedding will be returned.

### Example invocation

Invocation is very similar to `power_iteration_tsv.cpp`, except that `-E` is
ignored.

## jaccard_tsv.hpp

Perhaps the simplest example, `jaccard_tsv.hpp` simply estimates the Jaccard
similarity using a simple inclusion-exclusion subtraction.
However, it is not possible to provide error guarantees in general, so sets with
small true Jaccard similarity (very small intersection relative to the set
dimesionality) will have large error.

### CLI

You can view the CLI of the `jaccard_tsv.hpp` executable by running

```
# ./examples/jaccard_tsv -h
 -h             Indicates whether to print usage string
 -c <arg>       Number of vertices in the input graph
 -z <arg>       Random seed
 -B <arg>       Buffer size used during exponentiation (default \infty)
 -M <arg>       Metall datastore path (optional)
 -u             Indicates that the graph stream is undirected
 -m             Indicates ygm::map use (ygm::array otherwise)
 -V             Indicates whether to print verbose output
 -i <arg>       File containing data to build index (required)
 -q <arg>       Optional file containing indices to query
 -g <arg>       File containing true communities
 -r <arg>       Power of 2 size of projection range
 -R <arg>       Power of 2 number of replicated projections
 -e <arg>       Integral power of data matrix
 -C             Perform chebyshev polynomial embedding
 -f <arg>       Path for all outfiles
```

These flags have the same interpretation as the above, although note that many
are ignored, such as `-e` and `-C`.

### Outfiles

The outfiles are the same as `power_iteration_tsv.cpp`, with the exception that
`jaccard_tsv.hpp` has no powers and so only a single. Thus,
only the final embedding will be returned.

### Example invocation

Invocation is very similar to `power_iteration_tsv.cpp`, with no `-e`
invocation.

```
$ mpirun -n 4 ./examples/jaccard_tsv \
  -i /Users/priest2/workspace/nisenemarks/data/2017/5000/parts.txt \
  -g /Users/priest2/workspace/nisenemarks/data/2017/5000/gt_parts.txt \
  -c 5000 -r 32 -R 1 -e 1 -V -f tmp
```
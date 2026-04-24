// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

namespace psqz::sketch {

template <typename SketchContainerType>
void chebyshev_p2(SketchContainerType &SAp1, SketchContainerType &SAp2,
                  SketchContainerType &polynomial) {
  using index_type       = typename SketchContainerType::key_type;
  using feature_vec_type = typename SketchContainerType::mapped_type;
  using feature_type     = typename feature_vec_type::value_type;

  auto add_lambda = [](const index_type &idx, feature_vec_type &sketch,
                       const feature_vec_type &incoming) {
    sketch += incoming;
  };

  SAp1.for_all([&polynomial, &add_lambda](const index_type &idx,
                                          feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg *= 16.0;
      reg -= 7.0;
    }
    polynomial.async_visit(idx, add_lambda, sketch);
  });
  SAp2.for_all([&polynomial, &add_lambda](const index_type &idx,
                                          feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg *= 32.0;
    }
    polynomial.async_visit(idx, add_lambda, sketch);
  });

  polynomial.comm().barrier();

  polynomial.for_all([](const index_type &idx, feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg /= 41.0;
    }
  });

  polynomial.comm().barrier();
}

template <typename SketchContainerType>
void chebyshev_p3(SketchContainerType &SAp1, SketchContainerType &SAp2,
                  SketchContainerType &SAp3, SketchContainerType &polynomial) {
  using index_type       = typename SketchContainerType::key_type;
  using feature_vec_type = typename SketchContainerType::mapped_type;
  using feature_type     = typename feature_vec_type::value_type;

  auto add_lambda = [](const index_type &idx, feature_vec_type &sketch,
                       const feature_vec_type &incoming) {
    sketch += incoming;
  };

  SAp1.for_all([&polynomial, &add_lambda](const index_type &idx,
                                          feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg *= -60.0;
      reg -= 23.0;
    }
    polynomial.async_visit(idx, add_lambda, sketch);
  });
  SAp2.for_all([&polynomial, &add_lambda](const index_type &idx,
                                          feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg *= 192.0;
    }
    polynomial.async_visit(idx, add_lambda, sketch);
  });
  SAp3.for_all([&polynomial, &add_lambda](const index_type &idx,
                                          feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg *= 256.0;
    }
    polynomial.async_visit(idx, add_lambda, sketch);
  });

  polynomial.comm().barrier();

  polynomial.for_all([](const index_type &idx, feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg /= 365.0;
    }
  });

  polynomial.comm().barrier();
}

template <typename SketchContainerType>
void chebyshev_p4(SketchContainerType &SAp1, SketchContainerType &SAp2,
                  SketchContainerType &SAp3, SketchContainerType &SAp4,
                  SketchContainerType &polynomial) {
  using index_type       = typename SketchContainerType::key_type;
  using feature_vec_type = typename SketchContainerType::mapped_type;
  using feature_type     = typename feature_vec_type::value_type;

  auto add_lambda = [](const index_type &idx, feature_vec_type &sketch,
                       const feature_vec_type &incoming) {
    sketch += incoming;
  };

  SAp1.for_all([&polynomial, &add_lambda](const index_type &idx,
                                          feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg *= -448.0;
      reg += 17.0;
    }
    polynomial.async_visit(idx, add_lambda, sketch);
  });
  SAp2.for_all([&polynomial, &add_lambda](const index_type &idx,
                                          feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg *= 384.0;
    }
    polynomial.async_visit(idx, add_lambda, sketch);
  });
  SAp3.for_all([&polynomial, &add_lambda](const index_type &idx,
                                          feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg *= 2048.0;
    }
    polynomial.async_visit(idx, add_lambda, sketch);
  });
  SAp4.for_all([&polynomial, &add_lambda](const index_type &idx,
                                          feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg *= 2048.0;
    }
    polynomial.async_visit(idx, add_lambda, sketch);
  });

  polynomial.comm().barrier();

  polynomial.for_all([](const index_type &idx, feature_vec_type &sketch) {
    for (feature_type &reg : sketch) {
      reg /= 3281.0;
    }
  });

  polynomial.comm().barrier();
}
}  // namespace psqz::sketch

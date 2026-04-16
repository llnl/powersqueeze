// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/kron/parameters.hpp>
#include <psqz/tsv/parameters.hpp>

#include <unistd.h>

namespace psqz::sketch {

namespace detail {
template <typename BaseType>
struct parameters : public BaseType {
  using base_type = BaseType;

  parameter<std::uint64_t> range_size;
  parameter<std::uint64_t> replication_count;
  parameter<int>           exponent;
  parameter<bool>          chebyshev;

  parameters()
      : base_type(),
        range_size("range_size", "Power of 2 size of projection range", 'r',
                   true, 8),
        replication_count("replication_count",
                          "Power of 2 number of replicated projections", 'R',
                          true, 1),
        exponent("exponent", "Integral power of data matrix", 'e', true, 1),
        chebyshev("chebyshev?", "Perform chebyshev polynomial embedding", 'C',
                  false, false) {
    this->_params.push_back(&range_size);
    this->_params.push_back(&replication_count);
    this->_params.push_back(&exponent);
    this->_params.push_back(&chebyshev);
  }

  bool _help_needed() const override {
    bool ret = base_type::_help_needed();
    if (range_size() < 1) {
      std::cout << "Must specify positive range size, not " << range_size()
                << std::endl;
      return true;
    }
    if (replication_count() < 1) {
      std::cout << "Must specify positive replication count, not "
                << replication_count() << std::endl;
      return true;
    }
    if (exponent() < 1) {
      std::cout << "Must specify positive exponent, not " << exponent()
                << std::endl;
      return true;
    }
    return ret;
  }
};

namespace streaming {
template <typename BaseType>
struct parameters : public psqz::sketch::detail::parameters<BaseType> {
  using base_type = psqz::sketch::detail::parameters<BaseType>;

  parameter<std::uint64_t> final_range_size;
  parameter<std::uint64_t> final_replication_count;

  parameters()
      : base_type(),
        final_range_size("final range_size",
                         "Power of 2 size of final projection range", 'j', true,
                         8),
        final_replication_count(
            "replication_count",
            "Power of 2 number of replicated final projections", 'J', true, 1) {
    this->_params.push_back(&final_range_size);
    this->_params.push_back(&final_replication_count);
  }

  bool _help_needed() const override {
    bool ret = base_type::_help_needed();
    if (final_range_size() < 1) {
      std::cout << "Must specify positive range size, not "
                << final_range_size() << std::endl;
      return true;
    }
    if (final_replication_count() < 1) {
      std::cout << "Must specify positive replication count, not "
                << final_replication_count() << std::endl;
      return true;
    }
    return ret;
  }
};
}  // namespace streaming
}  // namespace detail

namespace tsv {
using parameters = psqz::sketch::detail::parameters<psqz::tsv::parameters>;

namespace query_only {
using parameters =
    psqz::sketch::detail::parameters<psqz::tsv::query_only::parameters>;
}
}  // namespace tsv

namespace kron {
using parameters = psqz::sketch::detail::parameters<psqz::kron::parameters>;
}

namespace streaming {
namespace tsv {
using parameters =
    psqz::sketch::detail::streaming::parameters<psqz::tsv::parameters>;

namespace query_only {
using parameters = psqz::sketch::detail::streaming::parameters<
    psqz::tsv::query_only::parameters>;
}
}  // namespace tsv

namespace kron {
using parameters =
    psqz::sketch::detail::streaming::parameters<psqz::kron::parameters>;
}
}  // namespace streaming

}  // namespace psqz::sketch

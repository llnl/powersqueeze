// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <psqz/graph/edge_streamer.hpp>
#include <psqz/graph/truth.hpp>
#include <psqz/tsv/edge_streamer.hpp>
#include <psqz/tsv/truth.hpp>

namespace psqz::tsv {

namespace core {
template <typename HandlerType>
using edge_streamer =
    psqz::tsv::detail::edge_streamer<HandlerType,
                                     psqz::graph::detail::edge_streamer>;
template <typename HandlerType>
using adjacency_normalizer = psqz::tsv::detail::adjacency_normalizer<
    HandlerType, psqz::graph::detail::adjacency_normalizer>;
template <typename HandlerType>
using truth = psqz::tsv::detail::truth<HandlerType, psqz::graph::detail::truth>;
template <typename HandlerType>
using queries =
    psqz::tsv::detail::queries<HandlerType, psqz::graph::detail::queries>;
}  // namespace core

template <typename HandlerType>
using adjacency_streamer =
    psqz::graph::adjacency_streamer<HandlerType, core::edge_streamer>;
template <typename HandlerType>
using normalized_edge_streamer =
    psqz::graph::normalized_adjacency_streamer<HandlerType, core::edge_streamer,
                                               core::adjacency_normalizer>;
template <typename HandlerType>
using truth = psqz::graph::truth<HandlerType, core::truth>;
template <typename HandlerType>
using queries = psqz::graph::queries<HandlerType, core::queries>;
}  // namespace psqz::tsv
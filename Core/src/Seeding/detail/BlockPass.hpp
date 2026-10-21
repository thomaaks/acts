// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace Acts::detail {

/// Collects, in order and without a branch, the values of the items whose
/// flag is set.
///
/// @param flags whether each item is kept, 1 or 0
/// @param valueOf the value collected, called with an item's position
/// @param out receives the values; room for one per item
/// @return the part of @p out holding the kept values
template <typename value_of_t, typename out_t, std::size_t kExtent>
std::span<out_t> collectFlagged(std::span<const std::uint32_t> flags,
                                const value_of_t& valueOf,
                                std::span<out_t, kExtent> out) {
  std::size_t nKept = 0;
  for (std::size_t i = 0; i < flags.size(); ++i) {
    // written every time; the position only moves on if the flag is set
    out[nKept] = valueOf(i);
    nKept += flags[i];
  }

  return out.first(nKept);
}

/// Calls @p process with the positions in [@p begin, @p end) that pass a cut,
/// one block of positions at a time. For each block, the cut is evaluated for
/// every position and the passing ones are collected, both without a branch.
///
/// @tparam kBlockSize the number of positions per block
/// @tparam kChunkSize the number of positions per step of the cut's loop; a
///   length known at compile time lets each step be vectorised
/// @param passes the cut, called with a position in [@p begin, @p end),
///   returning 1 if it passes and 0 if not
/// @param process called with the passing positions of a block, in
///   [@p begin, @p end) and in order, for each block that has any
template <std::size_t kBlockSize, std::size_t kChunkSize, typename passes_t,
          typename process_t>
void forEachPassingBlock(std::uint32_t begin, std::uint32_t end,
                         const passes_t& passes, const process_t& process) {
  std::array<std::uint32_t, kBlockSize> flags{};
  std::array<std::uint32_t, kBlockSize> passing{};

  for (std::size_t block = begin; block < end; block += kBlockSize) {
    const std::size_t n = std::min(kBlockSize, end - block);

    std::uint32_t nPassed = 0;
    std::size_t i = 0;
    for (; i + kChunkSize <= n; i += kChunkSize) {
      for (std::size_t j = i; j < i + kChunkSize; ++j) {
        flags[j] = passes(block + j);
        nPassed += flags[j];
      }
    }
    for (; i < n; ++i) {
      flags[i] = passes(block + i);
      nPassed += flags[i];
    }

    if (nPassed == 0) {
      continue;
    }

    process(collectFlagged(
        std::span<const std::uint32_t>(flags).first(n),
        [&](std::size_t k) { return static_cast<std::uint32_t>(block + k); },
        std::span(passing)));
  }
}

}  // namespace Acts::detail

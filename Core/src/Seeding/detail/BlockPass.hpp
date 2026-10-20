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
#include <ranges>
#include <span>

namespace Acts::detail {

/// One value for each candidate of a block.
template <typename value_t, std::size_t kBlockSize>
using PerCandidate = std::array<value_t, kBlockSize>;

/// Calls @p function with each i in [0, @p n).
///
/// @tparam kChunkSize the number of calls per step of the loop; a length
///   known at compile time lets each step be vectorised
template <std::size_t kChunkSize, typename function_t>
void forEach(std::size_t n, const function_t& function) {
  std::size_t i = 0;

  for (; i + kChunkSize <= n; i += kChunkSize) {
    for (std::size_t j = i; j < i + kChunkSize; ++j) {
      function(j);
    }
  }

  for (; i < n; ++i) {
    function(i);
  }
}

/// A block of consecutive candidate positions.
struct Block {
  /// The position of the block's first candidate
  std::size_t first = 0;
  /// The number of candidates in the block
  std::size_t size = 0;

  /// Calls @p function with each candidate's index in the block, i, and its
  /// position, first + i, in steps of @p kChunkSize as detail::forEach.
  template <std::size_t kChunkSize, typename function_t>
  void forEach(const function_t& function) const {
    detail::forEach<kChunkSize>(size,
                                [&](std::size_t i) { function(i, first + i); });
  }
};

/// The positions in [@p begin, @p end) as blocks of up to @p kBlockSize.
template <std::size_t kBlockSize>
auto blocks(std::size_t begin, std::size_t end) {
  const std::size_t nBlocks = (end - begin + kBlockSize - 1) / kBlockSize;
  return std::views::iota(std::size_t{0}, nBlocks) |
         std::views::transform([=](std::size_t b) {
           const std::size_t first = begin + b * kBlockSize;
           return Block{first, std::min(kBlockSize, end - first)};
         });
}

/// Whether each of up to @p kBlockSize candidates passes a cut, 1 or 0.
template <std::size_t kBlockSize>
class PassFlags {
 public:
  /// The flag of candidate @p i
  std::uint32_t& operator[](std::size_t i) { return m_flags[i]; }

  /// The i in [0, @p n) whose flag is set, in order, collected without a
  /// branch.
  std::span<const std::uint32_t> passing(std::size_t n) {
    std::size_t nPassing = 0;
    for (std::size_t i = 0; i < n; ++i) {
      // written every time; the position only moves on if the flag is set
      m_passing[nPassing] = static_cast<std::uint32_t>(i);
      nPassing += m_flags[i];
    }

    return std::span<const std::uint32_t>(m_passing).first(nPassing);
  }

  /// The indices in @p block of its candidates whose flag is set
  std::span<const std::uint32_t> passing(const Block& block) {
    return passing(block.size);
  }

 private:
  // Not cleared: a flag is always written before it is read.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
  PerCandidate<std::uint32_t, kBlockSize> m_flags;
  PerCandidate<std::uint32_t, kBlockSize> m_passing;
  // NOLINTEND(cppcoreguidelines-pro-type-member-init)
};

}  // namespace Acts::detail

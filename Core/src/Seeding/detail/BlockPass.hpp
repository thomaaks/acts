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
#include <tuple>
#include <type_traits>

namespace Acts::detail {

/// One value for each candidate of a block.
template <typename value_t, std::size_t kBlockSize>
using PerCandidate = std::array<value_t, kBlockSize>;

/// A pass that is not made: every candidate is kept.
struct NoPass {};

/// @p pass if @p kEnabled, otherwise a pass that is not made.
template <bool kEnabled, typename pass_t>
auto passIf(const pass_t& pass) {
  if constexpr (kEnabled) {
    return pass;
  } else {
    return NoPass{};
  }
}

/// A pass made only for the candidates for which a condition holds; the
/// others are kept.
template <typename condition_t, typename pass_t>
struct OnlyWhere {
  condition_t condition;
  pass_t pass;
};

/// @p pass, made only for the candidates for which @p condition holds.
template <typename condition_t, typename pass_t>
OnlyWhere<condition_t, pass_t> onlyWhere(const condition_t& condition,
                                         const pass_t& pass) {
  return {condition, pass};
}

/// The passes, in the order they are made.
template <typename... passes_t>
std::tuple<passes_t...> passes(const passes_t&... list) {
  return {list...};
}

/// Calls @p function with each i in [0, @p n), @p kChunkSize at a time: a
/// length known at compile time lets each step be vectorised.
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

template <typename pass_t>
constexpr bool kIsOnlyWhere = false;
template <typename condition_t, typename pass_t>
constexpr bool kIsOnlyWhere<OnlyWhere<condition_t, pass_t>> = true;

/// Makes @p pass for the @p survivors of a block, given as indices in the
/// block whose first position is @p first, and keeps those that pass it in
/// place, in order and without a branch.
///
/// @return the number of survivors kept
template <std::size_t kBlockSize, typename pass_t>
std::size_t makePass(const pass_t& pass, std::size_t first,
                     std::span<std::uint32_t> survivors,
                     PerCandidate<std::uint32_t, kBlockSize>& flags,
                     PerCandidate<std::uint32_t, kBlockSize>& subset) {
  if constexpr (std::is_same_v<pass_t, NoPass>) {
    return survivors.size();
  } else if constexpr (kIsOnlyWhere<pass_t>) {
    // the survivors for which the condition holds; the others are kept
    std::size_t nSubset = 0;
    for (const std::uint32_t i : survivors) {
      subset[nSubset] = i;
      nSubset += static_cast<std::size_t>(pass.condition(i, first + i));
      flags[i] = 1;
    }
    for (const std::uint32_t i : std::span(subset).first(nSubset)) {
      flags[i] = static_cast<std::uint32_t>(pass.pass(i, first + i));
    }

    std::size_t nKept = 0;
    for (const std::uint32_t i : survivors) {
      survivors[nKept] = i;
      nKept += flags[i];
    }
    return nKept;
  } else if constexpr (std::is_void_v<std::invoke_result_t<
                           const pass_t&, std::size_t, std::size_t>>) {
    // a pass that only computes keeps every survivor
    for (const std::uint32_t i : survivors) {
      pass(i, first + i);
    }
    return survivors.size();
  } else {
    std::size_t nKept = 0;
    for (const std::uint32_t i : survivors) {
      // written every time; the position only moves on if i is kept
      survivors[nKept] = i;
      nKept += static_cast<std::size_t>(pass(i, first + i));
    }
    return nKept;
  }
}

/// Makes @p passes for the candidates at the positions [@p begin, @p end),
/// one block of up to @p kBlockSize at a time, and calls @p keep for each
/// candidate that passes them all, in order.
///
/// A pass is called with a candidate's index in its block, i, and its
/// position, k, and returns whether the candidate is kept; a pass that
/// returns nothing keeps every candidate. The first pass is made for every
/// candidate of a block, @p kChunkSize at a time as in forEach, and each
/// further pass for the candidates that passed the ones before it. The
/// candidates that pass are collected without a branch, and a block where no
/// candidate passes the first pass is skipped.
template <std::size_t kBlockSize, std::size_t kChunkSize, typename... passes_t,
          typename keep_t>
void runBlockPass(std::size_t begin, std::size_t end,
                  const std::tuple<passes_t...>& passes, const keep_t& keep) {
  // Not cleared: each entry is written before it is read.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
  PerCandidate<std::uint32_t, kBlockSize> flags;
  PerCandidate<std::uint32_t, kBlockSize> survivors;
  PerCandidate<std::uint32_t, kBlockSize> subset;
  // NOLINTEND(cppcoreguidelines-pro-type-member-init)

  for (std::size_t first = begin; first < end; first += kBlockSize) {
    const std::size_t n = std::min(kBlockSize, end - first);

    std::apply(
        [&](const auto& firstPass, const auto&... furtherPasses) {
          std::uint32_t nPassed = 0;
          forEach<kChunkSize>(n, [&](std::size_t i) {
            flags[i] = static_cast<std::uint32_t>(firstPass(i, first + i));
            nPassed += flags[i];
          });
          if (nPassed == 0) {
            return;
          }

          std::size_t nSurvivors = 0;
          for (std::size_t i = 0; i < n; ++i) {
            survivors[nSurvivors] = static_cast<std::uint32_t>(i);
            nSurvivors += flags[i];
          }

          ((nSurvivors = makePass<kBlockSize>(
                furtherPasses, first, std::span(survivors).first(nSurvivors),
                flags, subset)),
           ...);

          for (const std::uint32_t i : std::span(survivors).first(nSurvivors)) {
            keep(i, first + i);
          }
        },
        passes);
  }
}

}  // namespace Acts::detail

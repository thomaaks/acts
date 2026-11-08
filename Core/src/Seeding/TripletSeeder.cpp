// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/TripletSeeder.hpp"

#include "Acts/EventData/SpacePointContainer.hpp"
#include "Acts/Seeding/DoubletSeedFinder.hpp"
#include "Acts/Seeding/TripletSeedFinder.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <span>

#include <Eigen/Dense>

namespace Acts {

namespace {

/// Advance a space point group past the candidates that can no longer pair
/// with the middle space point, or with any of the ones after it. Groups are
/// sorted by radius and the middle space points are swept in ascending radius,
/// so the cut point only ever moves forward and the group is never rewound.
///
/// This is the incremental form of the binary search that primes the groups in
/// `createSeedsFromGroups`; `isInRange` has to spell the radius cut exactly the
/// way `DoubletSeedFinder` does, so that the two agree bit for bit.
template <typename SpacePointGroup, typename Predicate>
std::uint32_t advanceGroup(SpacePointGroup& group, Predicate isInRange) {
  std::uint32_t offset = 0;
  for (ConstSpacePointProxy candidateSp : group) {
    if (isInRange(candidateSp)) {
      break;
    }
    ++offset;
  }
  group = group.subrange(offset);
  return offset;
}

/// Advance a space point group's end past the candidates in the middle space
/// point's radial range, from `end`, that end for the middle space point
/// before: like the start, it only moves forward as the middle space points
/// do. `isInRange` has to spell the radius cut exactly as `DoubletSeedFinder`
/// does.
template <typename SpacePointGroup, typename Predicate>
std::uint32_t advanceGroupEnd(const SpacePointGroup& group, std::uint32_t end,
                              Predicate isInRange) {
  const auto size = static_cast<std::uint32_t>(group.size());
  while (end < size && isInRange(group[end])) {
    ++end;
  }
  return end;
}

template <typename SpacePointCollections>
void createSeedsFromGroupsImpl(
    const Logger& logger, TripletSeeder::Cache& cache,
    const DoubletSeedFinder& bottomFinder, const DoubletSeedFinder& topFinder,
    const TripletSeedFinder& tripletFinder, const ITripletSeedFilter& filter,
    const SpacePointContainer& spacePoints,
    SpacePointCollections& bottomSpGroups, const ConstSpacePointProxy& middleSp,
    SpacePointCollections& topSpGroups, bool spacePointsSortedByRadius,
    SeedContainer& outputSeeds, std::span<std::uint32_t> bottomEnds,
    std::span<std::uint32_t> topEnds) {
  MiddleSpInfo middleSpInfo = DoubletSeedFinder::computeMiddleSpInfo(middleSp);
  const float rM = middleSp.zr()[1];
  const float bottomDeltaRMin = bottomFinder.config().deltaRMin;
  const float bottomDeltaRMax = bottomFinder.config().deltaRMax;
  const float topDeltaRMin = topFinder.config().deltaRMin;
  const float topDeltaRMax = topFinder.config().deltaRMax;

  // the doublets of one side, each group advanced to the first space point in
  // the middle space point's radial range and cut at its end
  const auto collectDoublets =
      [&](auto& spGroups, std::span<std::uint32_t> ends,
          const DoubletSeedFinder& finder, DoubletsForMiddleSp& doublets,
          auto reachesWindow, auto withinWindow) {
        assert(ends.size() == spGroups.size());
        doublets.clear();
        for (std::size_t g = 0; g < spGroups.size(); ++g) {
          auto& spGroup = spGroups[g];
          if (!spacePointsSortedByRadius) {
            finder.createDoublets(middleSp, middleSpInfo, spGroup, doublets);
            continue;
          }
          const std::uint32_t offset = advanceGroup(spGroup, reachesWindow);
          // the start may have passed the old end of an empty window
          ends[g] = advanceGroupEnd(
              spGroup, ends[g] - std::min(ends[g], offset), withinWindow);
          finder.createDoublets(middleSp, middleSpInfo,
                                spGroup.subrange(0, ends[g]), doublets);
        }
      };

  // create middle-top doublets
  collectDoublets(
      topSpGroups, topEnds, topFinder, cache.topDoublets,
      [&](const ConstSpacePointProxy& sp) {
        return sp.zr()[1] - rM >= topDeltaRMin;
      },
      [&](const ConstSpacePointProxy& sp) {
        return sp.zr()[1] - rM <= topDeltaRMax;
      });

  // no top SP found -> cannot form any triplet
  if (cache.topDoublets.empty()) {
    ACTS_VERBOSE("No compatible Tops, returning");
    return;
  }

  if (!filter.sufficientTopDoublets(spacePoints, middleSp, cache.topDoublets)) {
    return;
  }

  // create middle-bottom doublets
  collectDoublets(
      bottomSpGroups, bottomEnds, bottomFinder, cache.bottomDoublets,
      [&](const ConstSpacePointProxy& sp) {
        return rM - sp.zr()[1] <= bottomDeltaRMax;
      },
      [&](const ConstSpacePointProxy& sp) {
        return rM - sp.zr()[1] >= bottomDeltaRMin;
      });

  // no bottom SP found -> cannot form any triplet
  if (cache.bottomDoublets.empty()) {
    ACTS_VERBOSE("No compatible Bottoms, returning");
    return;
  }

  ACTS_VERBOSE("Candidates: " << cache.bottomDoublets.size() << " bottoms and "
                              << cache.topDoublets.size()
                              << " tops for middle candidate indexed "
                              << middleSp.index());

  // the triplet top candidates of each bottom doublet go to the filter
  const auto filterCandidates =
      [&](const DoubletsForMiddleSp::Proxy& bottomDoublet,
          const TripletTopCandidates& candidates) {
        filter.filterTripletTopCandidates(spacePoints, middleSp, bottomDoublet,
                                          candidates);
      };
  TripletSeedFinder::CandidatesCallback filterCallback;
  filterCallback.connect(filterCandidates);

  // combine doublets to triplets
  if (tripletFinder.config().sortedByCotTheta) {
    cache.bottomDoublets.sortByCotTheta({0, cache.bottomDoublets.size()},
                                        cache.sortedBottoms, cache.sortStorage);
    cache.topDoublets.sortByCotTheta({0, cache.topDoublets.size()},
                                     cache.sortedTops, cache.sortStorage);

    tripletFinder.createTripletTopCandidates(
        spacePoints, middleSp, cache.bottomDoublets.subset(cache.sortedBottoms),
        cache.topDoublets.subset(cache.sortedTops), cache.tripletTopCandidates,
        filterCallback);
  } else {
    tripletFinder.createTripletTopCandidates(
        spacePoints, middleSp, cache.bottomDoublets.range(),
        cache.topDoublets.range(), cache.tripletTopCandidates, filterCallback);
  }

  filter.filterTripletsMiddleFixed(spacePoints, outputSeeds);
}

}  // namespace

TripletSeeder::TripletSeeder(std::unique_ptr<const Logger> logger_)
    : m_logger(std::move(logger_)) {
  if (m_logger == nullptr) {
    throw std::invalid_argument("TripletSeeder: logger cannot be null");
  }
}

void TripletSeeder::createSeedsFromGroup(
    Cache& cache, const DoubletSeedFinder& bottomFinder,
    const DoubletSeedFinder& topFinder, const TripletSeedFinder& tripletFinder,
    const ITripletSeedFilter& filter, const SpacePointContainer& spacePoints,
    SpacePointContainer::ConstSubset bottomSps,
    const ConstSpacePointProxy& middleSp,
    SpacePointContainer::ConstSubset topSps, SeedContainer& outputSeeds) const {
  assert((bottomFinder.config().spacePointsSortedByRadius ==
          topFinder.config().spacePointsSortedByRadius) &&
         "Inconsistent space point sorting");

  // the cursors live and die with this call, so every middle space point
  // starts its sweep at the front of the subset again
  std::array<SpacePointContainer::ConstSubset, 1> bottomSpGroups{bottomSps};
  std::array<SpacePointContainer::ConstSubset, 1> topSpGroups{topSps};
  std::array<std::uint32_t, 1> bottomEnds{0};
  std::array<std::uint32_t, 1> topEnds{0};

  createSeedsFromGroupsImpl(*m_logger, cache, bottomFinder, topFinder,
                            tripletFinder, filter, spacePoints, bottomSpGroups,
                            middleSp, topSpGroups,
                            bottomFinder.config().spacePointsSortedByRadius,
                            outputSeeds, bottomEnds, topEnds);
}

void TripletSeeder::createSeedsFromGroups(
    Cache& cache, const DoubletSeedFinder& bottomFinder,
    const DoubletSeedFinder& topFinder, const TripletSeedFinder& tripletFinder,
    const ITripletSeedFilter& filter, const SpacePointContainer& spacePoints,
    std::span<const SpacePointContainer::ConstRange> bottomSpGroups,
    const SpacePointContainer::ConstRange& middleSpGroup,
    std::span<const SpacePointContainer::ConstRange> topSpGroups,
    const std::pair<float, float>& radiusRangeForMiddle,
    SeedContainer& outputSeeds) const {
  assert((bottomFinder.config().spacePointsSortedByRadius ==
          topFinder.config().spacePointsSortedByRadius) &&
         "Inconsistent space point sorting");
  const bool spacePointsSortedByRadius =
      bottomFinder.config().spacePointsSortedByRadius;
  const float bottomDeltaRMax = bottomFinder.config().deltaRMax;
  const float topDeltaRMin = topFinder.config().deltaRMin;

  if (middleSpGroup.empty()) {
    return;
  }

  // the cursors are ours, so the caller's groups stay untouched and can be
  // handed to another call unchanged
  cache.bottomSpGroups.assign(bottomSpGroups.begin(), bottomSpGroups.end());
  cache.topSpGroups.assign(topSpGroups.begin(), topSpGroups.end());
  cache.bottomSpGroupEnds.assign(bottomSpGroups.size(), 0);
  cache.topSpGroupEnds.assign(topSpGroups.size(), 0);

  if (spacePointsSortedByRadius) {
    // Initialize initial offsets for bottom and top space points with binary
    // search. This requires at least one middle space point to be present which
    // is already checked above.
    const ConstSpacePointProxy firstMiddleSp = middleSpGroup.front();
    const float firstMiddleSpR = firstMiddleSp.zr()[1];

    for (auto& bottomSpGroup : cache.bottomSpGroups) {
      // Find the first bottom space point that is within the deltaRMax of the
      // first middle space point.
      const auto low = std::ranges::lower_bound(
          bottomSpGroup, firstMiddleSpR - bottomDeltaRMax, {},
          [&](const ConstSpacePointProxy& sp) { return sp.zr()[1]; });
      bottomSpGroup = bottomSpGroup.subrange(low - bottomSpGroup.begin());
    }

    for (auto& topSpGroup : cache.topSpGroups) {
      // Find the first top space point that is within the deltaRMin of the
      // first middle space point.
      const auto low = std::ranges::lower_bound(
          topSpGroup, firstMiddleSpR + topDeltaRMin, {},
          [&](const ConstSpacePointProxy& sp) { return sp.zr()[1]; });
      topSpGroup = topSpGroup.subrange(low - topSpGroup.begin());
    }
  }

  for (ConstSpacePointProxy spM : middleSpGroup) {
    const float rM = spM.zr()[1];

    if (spacePointsSortedByRadius) {
      // check if spM is outside our radial region of interest
      if (rM < radiusRangeForMiddle.first) {
        continue;
      }
      if (rM > radiusRangeForMiddle.second) {
        // break because SPs are sorted in r
        break;
      }
    }

    createSeedsFromGroupsImpl(*m_logger, cache, bottomFinder, topFinder,
                              tripletFinder, filter, spacePoints,
                              cache.bottomSpGroups, spM, cache.topSpGroups,
                              spacePointsSortedByRadius, outputSeeds,
                              cache.bottomSpGroupEnds, cache.topSpGroupEnds);
  }
}

}  // namespace Acts

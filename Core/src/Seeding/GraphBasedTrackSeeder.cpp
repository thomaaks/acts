// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/GraphBasedTrackSeeder.hpp"

#include "Acts/Seeding/GbtsGraphBuilder.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Acts::Experimental {

GraphBasedTrackSeeder::DerivedConfig::DerivedConfig(const Config& config)
    : Config(config),
      phiSliceWidth(2 * std::numbers::pi_v<float> / config.nMaxPhiSlice) {}

GraphBasedTrackSeeder::GraphBasedTrackSeeder(
    const DerivedConfig& config, std::shared_ptr<GbtsGeometry> geometry,
    std::unique_ptr<const Acts::Logger> logger)
    : m_cfg(config),
      m_geometry(std::move(geometry)),
      m_logger(std::move(logger)) {
  if (m_cfg.phiSortBuckets > GbtsNodeStorage::kMaxPhiSortBuckets) {
    throw std::invalid_argument(
        "GraphBasedTrackSeeder: phiSortBuckets exceeds the maximum");
  }

  if (m_cfg.useClusterWidthCuts && m_cfg.tauLookupTable.empty()) {
    throw std::invalid_argument(
        "GraphBasedTrackSeeder: the cluster width cuts need a tau lookup "
        "table");
  }
}

GbtsNodeStorage GraphBasedTrackSeeder::makeNodeStorage() const {
  return GbtsNodeStorage(
      {m_cfg.useClusterWidthCuts, m_cfg.maxEndcapClusterWidth,
       m_cfg.moduleHalfLengthY, m_cfg.moduleEdgeTolerance, m_cfg.phiSliceWidth,
       m_cfg.phiIndexMargin, m_cfg.phiSortBuckets, m_cfg.tauLutBinWidth},
      m_geometry, m_cfg.tauLookupTable);
}

void GraphBasedTrackSeeder::createSeeds(const SpacePointContainer& spacePoints,
                                        const GbtsRoiDescriptor& roi,
                                        const GbtsGraphBuilder& graphBuilder,
                                        const GbtsTrackingFilter& filter,
                                        const Options& options,
                                        SeedContainer& outputSeeds) const {
  GbtsNodeStorage nodeStorage = makeNodeStorage();

  nodeStorage.extend(spacePoints,
                     spacePoints.column<GbtsLayerIndex>("gbtsLayerIndex"),
                     spacePoints.column<float>("clusterWidth"),
                     spacePoints.column<float>("localPositionY"));

  nodeStorage.finalize();

  createSeeds(nodeStorage, roi, graphBuilder, filter, options, outputSeeds);
}

void GraphBasedTrackSeeder::createSeeds(
    GbtsNodeStorage& nodeStorage, const GbtsRoiDescriptor& roi,
    const GbtsGraphBuilder& graphBuilder, const GbtsTrackingFilter& filter,
    const Options& options, SeedContainer& outputSeeds,
    std::vector<std::optional<GbtsSeedFit>>* outputFits) const {
  ACTS_DEBUG("Loaded " << nodeStorage.numberOfNodes() << " graph nodes");

  GbtsGraph graph =
      graphBuilder.buildTheGraph(roi, nodeStorage, options.bFieldInZ);

  ACTS_DEBUG("Created graph with " << graph.edgeStorage.size() << " edges and "
                                   << graph.nConnections << " edge links");

  if (graph.edgeStorage.empty() || graph.nConnections == 0) {
    ACTS_WARNING("Missing edges or edge connections");
  }

  const auto seedsBefore = outputSeeds.size();
  extractSeedsFromTheGraph(nodeStorage, graph, outputSeeds, filter,
                           outputFits);

  ACTS_DEBUG("GBTS created " << outputSeeds.size() - seedsBefore << " seeds");
  if (outputSeeds.size() == seedsBefore) {
    ACTS_WARNING("No Seed Candidates");
  }
}

void GraphBasedTrackSeeder::extractSeedsFromTheGraph(
    const GbtsNodeStorage& nodeStorage, const GbtsGraph& graph,
    SeedContainer& outputSeeds, const GbtsTrackingFilter& filter,
    std::vector<std::optional<GbtsSeedFit>>* outputFits) const {
  const detail::GbtsNodeView nodeView = nodeStorage.nodeView();

  std::vector<SeedCandidateProperties> vSeedCandidates;

  // Turn the fitted path of every head edge into a seed candidate.
  for (const detail::GbtsEdgeState& rs :
       extractBestPathPerEdge(nodeView, graph, filter)) {
    const detail::GbtsEdge& head = *rs.vs.front();
    const float seedAbsEta = std::abs(-std::log(head.p[0]));

    std::vector<SpacePointIndex> vN;

    for (auto sIt = rs.vs.rbegin(); sIt != rs.vs.rend(); ++sIt) {
      if (sIt == rs.vs.rbegin()) {
        vN.push_back((*sIt)->n1);
      }

      vN.push_back((*sIt)->n2);
    }

    // a triplet is accepted if it makes it up to this point
    if (vN.size() < 3) {
      continue;
    }

    const auto origSeedSize = static_cast<std::uint32_t>(vN.size());

    const float origSeedQuality = -rs.j / origSeedSize;

    bool seedSplitFlag = (seedAbsEta < m_cfg.maxSeedSplitEta) &&
                         (origSeedSize >= m_cfg.minSplitSeedSize) &&
                         (origSeedSize <= m_cfg.maxSplitSeedSize);

    // split the seed by dropping spacepoints
    if (seedSplitFlag) {
      // the seed's own triplet, and those of the seed without its first node
      // and without its middle one, read in place
      const std::array<std::array<SpacePointIndex, 3>, 3> triplets = {{
          {vN[0], vN[origSeedSize / 2], vN[origSeedSize - 1]},
          {vN[1], vN[(origSeedSize - 1) / 2 + 1], vN[origSeedSize - 1]},
          {vN[0], vN[(origSeedSize - 1) / 2 + origSeedSize % 2],
           vN[origSeedSize - 1]},
      }};

      // triplet parameter estimate: the inverse radius of each triplet's
      // circle, mapped conformally about its last node; zero where it cannot
      // be
      std::array<float, 3> invRads{};
      for (std::uint32_t k = 0; k < invRads.size(); k++) {
        const std::array<SpacePointIndex, 3>& t = triplets[k];
        if (const auto line =
                detail::conformalLine(nodeView, t[2], {t[0], t[1]})) {
          const auto [A, B] = *line;
          invRads[k] = B / std::sqrt(1 + A * A);
        }
      }

      seedSplitFlag =
          !(std::abs(invRads[1] - invRads[0]) < m_cfg.maxInvRadDiff &&
            std::abs(invRads[2] - invRads[0]) < m_cfg.maxInvRadDiff &&
            std::abs(invRads[2] - invRads[1]) < m_cfg.maxInvRadDiff);
    }

    vSeedCandidates.push_back(
        {origSeedQuality, std::move(vN), seedSplitFlag, std::nullopt});

    if (outputFits != nullptr) {
      // filter state at the innermost node, x' along the first edge (inwards)
      const double slope2 = 1. + static_cast<double>(rs.x[1]) * rs.x[1];
      vSeedCandidates.back().fit = GbtsSeedFit{
          .position = Vector3(rs.refX * rs.c - rs.x[0] * rs.s,
                              rs.refX * rs.s + rs.x[0] * rs.c, rs.y[0]),
          // outwards: along -x', with the slope of the state
          .phi = std::atan2(-rs.s - rs.c * rs.x[1], -rs.c + rs.s * rs.x[1]),
          .cotTheta = rs.y[1],
          .curvature = rs.x[2] / (slope2 * std::sqrt(slope2))};
    }
  }

  // clone removal code goes below ...

  // best first, and in the order found where equal
  std::ranges::stable_sort(vSeedCandidates, {},
                           &SeedCandidateProperties::seedQuality);

  // whether a better candidate took each hit, indexed by graph node index
  std::vector<std::uint32_t> h2t(nodeStorage.numberOfNodes(), 0);

  // the space points of the seed being written out
  std::vector<SpacePointIndex> spacePoints;

  // The candidates best first, so each hit goes to the first to take it, and
  // whether one shares too many hits with better ones is known as it comes.
  // The clones dropped, and split seeds split if need be.
  for (const auto& seed : vSeedCandidates) {
    // the hits taken by a better candidate
    const auto nOther = std::ranges::count_if(
        seed.nodes, [&](const SpacePointIndex node) { return h2t[node] != 0; });
    for (const SpacePointIndex node : seed.nodes) {
      h2t[node] = 1;
    }

    if (nOther > m_cfg.hitShareThreshold * seed.nodes.size()) {
      continue;  // identified as a clone of a better candidate
    }

    // the seed without the node at skipIdx, or whole when it is past the end
    const auto addSeed = [&](const std::size_t skipIdx) {
      spacePoints.clear();
      for (std::size_t k = 0; k < seed.nodes.size(); k++) {
        if (k != skipIdx) {
          spacePoints.push_back(nodeStorage.spacePointIndex(seed.nodes[k]));
        }
      }
      auto newSeed = outputSeeds.createSeed();
      newSeed.assignSpacePointIndices(spacePoints);
      newSeed.quality() = seed.seedQuality;
      // a drop-out has no fit of its own
      if (outputFits != nullptr) {
        outputFits->push_back(seed.forSeedSplitting ? std::nullopt : seed.fit);
      }
    };

    if (!seed.forSeedSplitting) {
      addSeed(seed.nodes.size());
      continue;
    }

    // a split seed as "drop-out" seeds: the first and the middle dropped
    addSeed(0);
    addSeed(seed.nodes.size() / 2);
  }
}

std::vector<detail::GbtsEdgeState>
GraphBasedTrackSeeder::extractBestPathPerEdge(
    const detail::GbtsNodeView& nodeView, const GbtsGraph& graph,
    const GbtsTrackingFilter& filter) const {
  const std::vector<detail::GbtsEdge>& edgeStorage = graph.edgeStorage;
  const auto nEdges = static_cast<std::uint32_t>(edgeStorage.size());
  const auto maxLength = static_cast<std::uint32_t>(m_cfg.maxChainLength);

  // The level of an edge is the length of the longest chain of edges going
  // outwards from it. `vNei` only holds edges further in, which are created
  // later, so one pass in edge order settles it. Edges in chains longer than
  // `maxLength` stay above it and are left out.
  std::vector<std::uint32_t> level(nEdges, 1);
  for (std::uint32_t e = 0; e < nEdges; ++e) {
    const detail::GbtsEdge& edge = edgeStorage[e];
    const std::uint32_t next = std::min(level[e] + 1, maxLength + 1);
    for (std::uint32_t k = 0; k < edge.nNei; ++k) {
      std::uint32_t& innerLevel = level[edge.vNei[k]];
      innerLevel = std::max(innerLevel, next);
    }
  }

  // The most edges on a chain from an edge inwards to a root, the edge
  // included, stepping one level up at a time; zero for an edge left out.
  // The edges further in come later, so one pass backwards settles it.
  std::vector<std::uint32_t> edgesToRoot(nEdges, 0);
  for (std::uint32_t e = nEdges; e-- > 0;) {
    if (level[e] > maxLength) {
      continue;
    }
    // A root has no settled edge further in exactly one level up: an edge
    // further in with a longer chain outwards elsewhere does not end the chain
    const std::uint32_t nextLevel = level[e] + 1;
    std::uint32_t length = 1;
    if (nextLevel <= maxLength) {
      const detail::GbtsEdge& edge = edgeStorage[e];
      for (std::uint32_t k = 0; k < edge.nNei; ++k) {
        const std::uint32_t inner = edge.vNei[k];
        if (level[inner] == nextLevel) {
          length = std::max(length, edgesToRoot[inner] + 1);
        }
      }
    }
    edgesToRoot[e] = length;
  }

  // From every edge, search all paths inwards to a root, fitting on the way.
  // The complete path with the best score per space point is the candidate
  // of the edge.
  std::vector<detail::GbtsEdgeState> candidates;
  std::uint32_t minLength = 0;
  std::vector<std::uint32_t> path;
  std::vector<std::uint32_t> bestPath;
  float bestQuality = 0;
  detail::GbtsEdgeState bestState;

  const auto search = [&](const auto& self, const std::uint32_t e,
                          const detail::GbtsEdgeState& state) -> void {
    path.push_back(e);
    // every step goes one level up, so the depth is bounded by `maxLength`
    assert(path.size() <= maxLength);

    if (edgesToRoot[e] == 1) {
      // a root completes the chain
      const float quality = -state.j / static_cast<float>(path.size() + 1);
      if (path.size() >= minLength && quality < bestQuality) {
        bestQuality = quality;
        bestState = state;
        bestPath = path;
      }
    } else {
      const detail::GbtsEdge& edge = edgeStorage[e];
      for (std::uint32_t k = 0; k < edge.nNei; ++k) {
        const std::uint32_t inner = edge.vNei[k];
        if (level[inner] != level[e] + 1 ||
            path.size() + edgesToRoot[inner] < minLength) {
          continue;
        }
        detail::GbtsEdgeState innerState = state;
        if (filter.update(nodeView, edgeStorage[inner], innerState)) {
          self(self, inner, innerState);
        }
      }
    }

    path.pop_back();
  };

  for (std::uint32_t head = 0; head < nEdges; ++head) {
    const detail::GbtsEdge& headEdge = edgeStorage[head];

    // `addTriplets` accepts a path one edge short at low eta
    minLength = m_cfg.minSeedLevel;
    if (m_cfg.addTriplets && minLength > 0 &&
        std::abs(-std::log(headEdge.p[0])) <= m_cfg.maxAbsEtaAddTriplets) {
      --minLength;
    }

    if (edgesToRoot[head] == 0 || edgesToRoot[head] < minLength) {
      continue;
    }

    detail::GbtsEdgeState state;
    state.initialize(headEdge, nodeView, filter.m_cfg.initialVarianceX,
                     filter.m_cfg.initialVarianceY);
    if (!filter.update(nodeView, headEdge, state)) {
      continue;
    }

    bestQuality = std::numeric_limits<float>::max();
    bestPath.clear();
    search(search, head, state);
    if (bestPath.empty()) {
      continue;
    }

    bestState.vs.clear();
    for (const std::uint32_t e : bestPath) {
      bestState.vs.push_back(&edgeStorage[e]);
    }
    candidates.push_back(bestState);
  }

  return candidates;
}

}  // namespace Acts::Experimental

// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/GbtsNodeStorage.hpp"

#include "Acts/Seeding/GbtsGeometry.hpp"
#include "Acts/SpacePointFormation/detail/StripSpacePointCalibrationImpl.hpp"
#include "Acts/Utilities/MathHelpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <tuple>
#include <utility>

namespace Acts::Experimental {

GbtsNodeStorage::GbtsNodeStorage(const Config& config,
                                 std::shared_ptr<const GbtsGeometry> geometry,
                                 GbtsTauLookupTable tauLut)
    : m_cfg(config),
      m_geometry(std::move(geometry)),
      m_tauLut(std::move(tauLut)),
      m_etaBins(m_geometry->numBins()),
      m_stagedPerBin(m_geometry->numBins()) {}

std::optional<std::uint32_t> GbtsNodeStorage::insert(
    const SpacePointIndex index, const float x, const float y, const float z,
    const GbtsLayerIndex layerIndex, const float clusterWidth,
    const float localPositionY) {
  const float r = fastHypot(x, y);
  const float phi = std::atan2(y, x);
  return insert(index, x, y, z, r, phi, layerIndex, clusterWidth,
                localPositionY);
}

std::optional<std::uint32_t> GbtsNodeStorage::insert(
    const SpacePointIndex index, const float x, const float y, const float z,
    const float r, const float phi, const GbtsLayerIndex layerIndex,
    const float clusterWidth, const float localPositionY,
    const OuterStripSpacePointCalibrationDetails* strip) {
  const detail::GbtsLayer& layer = m_geometry->layerByIndex(layerIndex);
  const GbtsLayerDescription& description = layer.layerDescription();

  // wide pixel endcap clusters are dropped when the width cuts are on
  if (m_cfg.useClusterWidthCuts && description.type == GbtsLayerType::Endcap &&
      description.technology == GbtsLayerTechnology::Pixel &&
      clusterWidth > m_cfg.maxEndcapClusterWidth) {
    return std::nullopt;
  }

  const std::uint32_t bin = layer.getEtaBin(z, r);

  std::uint32_t stripIndex = detail::kNoStrip;
  // A strip pair on a pixel layer is never read: the seeder takes the strip
  // path per bin.
  if (strip != nullptr &&
      description.technology == GbtsLayerTechnology::Strip) {
    stripIndex = static_cast<std::uint32_t>(m_strips.size());
    // Derived once here rather than once per pair in the graph: it is six
    // cross products and a node takes part in many pairs.
    m_strips.push_back(
        Acts::detail::deriveOuterStripSpacePointCalibrationDetails(*strip));
  }

  m_stagedPerBin.at(bin).emplace_back(index, x, y, z, r, phi, clusterWidth,
                                      localPositionY, layerIndex, stripIndex);

  return bin;
}

void GbtsNodeStorage::extend(
    const SpacePointContainer& spacePoints,
    const ConstSpacePointColumnProxy<GbtsLayerIndex>& layerColumn,
    const ConstSpacePointColumnProxy<float>& clusterWidthColumn,
    const ConstSpacePointColumnProxy<float>& localPositionYColumn) {
  const bool strips =
      spacePoints.hasColumns(SpacePointColumns::StripCalibrationDetails);
  for (const auto& sp : spacePoints) {
    insert(sp, layerColumn, clusterWidthColumn, localPositionYColumn, strips);
  }
}

std::vector<std::pair<float, std::uint32_t>> GbtsNodeStorage::sortBinByPhi(
    const std::vector<StagedNode>& staged) const {
  const std::uint32_t nBuckets = m_cfg.phiSortBuckets;
  const auto bucketOf = [&](const float phi) {
    return static_cast<std::uint32_t>(
        0.5 * nBuckets * (phi / std::numbers::pi_v<float> + 1.0f));
  };

  // The buckets counted first and laid out in one array, rather than each
  // grown on its own: the same buckets, in the same order.
  std::array<std::uint32_t, kMaxPhiSortBuckets + 2> bucketStart{};
  for (const StagedNode& node : staged) {
    ++bucketStart[bucketOf(node.phi) + 1];
  }
  for (std::uint32_t bucket = 1; bucket <= nBuckets + 1; ++bucket) {
    bucketStart[bucket] += bucketStart[bucket - 1];
  }

  std::array<std::uint32_t, kMaxPhiSortBuckets + 2> bucketEnd = bucketStart;
  std::vector<std::pair<float, std::uint32_t>> phiOrder(staged.size());
  for (std::uint32_t stagedIdx = 0; stagedIdx < staged.size(); ++stagedIdx) {
    const float phi = staged[stagedIdx].phi;
    phiOrder[bucketEnd[bucketOf(phi)]++] = {phi, stagedIdx};
  }

  // Sort key (phi, z, r, staged index). Nodes with identical phi are ordered
  // by position, so the order does not depend on the input order. Only nodes
  // at the same position fall back to the insertion index.
  const auto byKey = [&staged](const std::pair<float, std::uint32_t>& a,
                               const std::pair<float, std::uint32_t>& b) {
    const StagedNode& na = staged[a.second];
    const StagedNode& nb = staged[b.second];
    return std::tie(a.first, na.z, na.r, a.second) <
           std::tie(b.first, nb.z, nb.r, b.second);
  };
  for (std::uint32_t bucket = 0; bucket <= nBuckets; ++bucket) {
    std::sort(phiOrder.begin() + bucketStart[bucket],
              phiOrder.begin() + bucketStart[bucket + 1], byKey);
  }

  return phiOrder;
}

void GbtsNodeStorage::finalize() {
  std::uint32_t nNodes = 0;
  for (const std::vector<StagedNode>& staged : m_stagedPerBin) {
    nNodes += static_cast<std::uint32_t>(staged.size());
  }

  m_positions.reserve(nNodes);
  m_spacePointIndex.reserve(nNodes);
  m_layers.reserve(nNodes);
  m_params.resize(nNodes);
  m_edgeInfo.resize(nNodes);

  const float phiMargin = m_cfg.phiIndexMargin * m_cfg.phiSliceWidth;
  const float twoPi = 2 * std::numbers::pi_v<float>;

  // The strip pairs reordered into node order, so that the pairs a bin reads
  // sit together; left empty when nothing carries one, which is what
  // `hasStrips` reports.
  std::vector<OuterStripSpacePointCalibrationDetailsDerived> strips;
  if (!m_strips.empty()) {
    m_stripIndex.assign(nNodes, detail::kNoStrip);
    strips.reserve(m_strips.size());
  }

  for (std::uint32_t bin = 0; bin < m_etaBins.size(); ++bin) {
    detail::GbtsEtaBinInfo& binInfo = m_etaBins[bin];

    binInfo.nodes = {numberOfNodes(), numberOfNodes()};

    const std::vector<StagedNode>& staged = m_stagedPerBin[bin];
    if (staged.empty()) {
      continue;
    }

    const auto phiOrder = sortBinByPhi(staged);

    for (const auto& [phi, stagedIdx] : phiOrder) {
      const StagedNode& node = staged[stagedIdx];
      const std::uint32_t nodeIdx = numberOfNodes();

      m_positions.push_back({node.x, node.y, node.z, node.r});
      m_spacePointIndex.push_back(node.spacePointIndex);

      m_layers.push_back(node.layer);

      if (node.strip != detail::kNoStrip) {
        m_stripIndex[nodeIdx] = static_cast<std::uint32_t>(strips.size());
        strips.push_back(m_strips[node.strip]);
      }

      detail::GbtsNodeParams& params = m_params[nodeIdx];
      params = {.phi = node.phi, .r = node.r, .z = node.z};

      // minTau and maxTau keep their "do not cut" defaults unless the lookup
      // table narrows them
      if (m_cfg.useClusterWidthCuts) {
        applyTauCuts(node, params);
      }

      binInfo.minRadius = std::min(binInfo.minRadius, node.r);
      binInfo.maxRadius = std::max(binInfo.maxRadius, node.r);
      binInfo.minZ = std::min(binInfo.minZ, node.z);
      binInfo.maxZ = std::max(binInfo.maxZ, node.z);
    }

    binInfo.nodes.second = numberOfNodes();
    // every node in a bin is on the same layer, so any of them will do
    const GbtsLayerDescription& description =
        m_geometry->layerDescription(staged.front().layer);
    binInfo.barrelOrder = description.barrelOrder;
    binInfo.technology = description.technology;

    // The bin's (phi, node) pairs in phi order, those within the margin of the
    // wrap-around repeated shifted by 2pi, so that the phi sliding window never
    // has to wrap: the last nodes before them and the first ones after, sized
    // up front.
    const auto n = static_cast<std::uint32_t>(phiOrder.size());
    std::uint32_t head = 0;
    while (head < n &&
           phiOrder[head].first < -std::numbers::pi_v<float> + phiMargin) {
      ++head;
    }
    std::uint32_t tail = n;
    while (tail > 0 &&
           phiOrder[tail - 1].first > std::numbers::pi_v<float> - phiMargin) {
      --tail;
    }
    const SpacePointIndex first = binInfo.nodes.first;
    binInfo.phiNodes.reserve(n - tail + n + head);
    for (std::uint32_t k = tail; k < n; ++k) {
      binInfo.phiNodes.emplace_back(phiOrder[k].first - twoPi, first + k);
    }
    for (std::uint32_t k = 0; k < n; ++k) {
      binInfo.phiNodes.emplace_back(phiOrder[k].first, first + k);
    }
    for (std::uint32_t k = 0; k < head; ++k) {
      binInfo.phiNodes.emplace_back(phiOrder[k].first + twoPi, first + k);
    }
  }

  m_strips = std::move(strips);

  m_stagedPerBin.clear();
  m_stagedPerBin.shrink_to_fit();
}

void GbtsNodeStorage::applyTauCuts(const StagedNode& staged,
                                   detail::GbtsNodeParams& params) const {
  const GbtsLayerDescription& description =
      m_geometry->layerDescription(staged.layer);

  // the table is trained on pixel barrel clusters
  if (description.technology != GbtsLayerTechnology::Pixel ||
      description.type != GbtsLayerType::Barrel) {
    return;
  }

  // by the reciprocal, not the division: 1/0.05f is exactly 20, the division
  // is not, and the difference lands on the bin edges
  const auto lutBinIdx =
      static_cast<std::int32_t>(
          std::floor(staged.clusterWidth * (1.0f / m_cfg.tauLutBinWidth))) -
      1;

  if (lutBinIdx < 0 ||
      lutBinIdx >= static_cast<std::int32_t>(m_tauLut.size())) {
    return;
  }

  const GbtsTauBounds& bounds = m_tauLut[static_cast<std::size_t>(lutBinIdx)];

  // close to the edge the cluster may be shortened, which the lookup table
  // covers with a separate pair of bounds
  const float dist2border =
      m_cfg.moduleHalfLengthY - std::abs(staged.localPositionY);
  const bool nearEdge = dist2border <= m_cfg.moduleEdgeTolerance;

  params.minTau = nearEdge ? bounds.minTauNearEdge : bounds.minTau;
  params.maxTau = nearEdge ? bounds.maxTauNearEdge : bounds.maxTau;

  if (params.maxTau < 0) {
    // insufficient training data, do not cut on tau
    params.maxTau = std::numeric_limits<float>::infinity();
  }
}

}  // namespace Acts::Experimental

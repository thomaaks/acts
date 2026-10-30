// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/StripSpacePointCalibrationDetails.hpp"
#include "Acts/EventData/Types.hpp"
#include "Acts/Seeding/GbtsLayerDescription.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Acts::Experimental::detail {

/// Maximum number of neighbouring edges recorded per graph edge
static constexpr std::uint32_t kGbtsMaxEdgeNeighbours = 6;

/// Bins of the per-node z0 histogram, which is kept as a bit mask in
/// GbtsNodeEdgeInfo::isConnected and so may not exceed its width.
static constexpr std::int32_t kGbtsZ0HistogramBins = 16;
static_assert(kGbtsZ0HistogramBins <=
              std::numeric_limits<std::uint16_t>::digits);

//! [gbts node params]
/// Per-node parameters used while building the graph.
///
/// All five values are read together in the innermost doublet loop, so they sit
/// in one record rather than in separate arrays.
struct GbtsNodeParams final {
  /// Minimum accepted |cot(theta)|. The infinite defaults mean "do not cut on
  /// tau"; only the tau lookup table narrows them.
  float minTau{-std::numeric_limits<float>::infinity()};
  /// Maximum accepted |cot(theta)|. See @ref minTau.
  float maxTau{std::numeric_limits<float>::infinity()};
  /// Azimuthal angle in the xy plane.
  float phi{};
  /// Transverse distance from the beamline.
  float r{};
  /// Global z coordinate.
  float z{};
};
//! [gbts node params]

//! [gbts node edge info]
/// Per-node graph bookkeeping, written while the graph is built.
///
/// The three fields are read together in the innermost doublet loop, so they
/// sit in one record rather than in separate arrays.
struct GbtsNodeEdgeInfo final {
  /// Index of the first incoming graph edge attached to the node.
  std::uint32_t firstEdge{};
  /// Total number of incoming graph edges attached to the node.
  std::uint16_t numEdges{};
  /// z0 histogram bitmask. Non-zero marks the node's outer neighbourhood as
  /// connected to the previously built graph.
  std::uint16_t isConnected{};
};
//! [gbts node edge info]

//! [gbts eta bin info]
/// Constant per-eta-bin data.
struct GbtsEtaBinInfo final {
  /// Range of node indices belonging to this bin.
  SpacePointIndexRange nodes{0, 0};

  /// (phi, node index) pairs covering this bin, with duplicated entries shifted
  /// by +-2pi at the wrap-around so the phi sliding window never has to wrap.
  std::vector<std::pair<float, SpacePointIndex>> phiNodes;

  /// Radial and z extent of the bin's nodes, empty until a node widens it.
  float minRadius{std::numeric_limits<float>::max()};
  float maxRadius{std::numeric_limits<float>::lowest()};
  float minZ{std::numeric_limits<float>::max()};
  float maxZ{std::numeric_limits<float>::lowest()};
  /// Inside-out pixel barrel ordinal of the bin's layer, -1 for the rest.
  std::int32_t barrelOrder{-1};

  /// Technology of the layer this bin belongs to.
  GbtsLayerTechnology technology{};

  /// Check if bin is empty
  /// @return True if bin has no nodes
  bool empty() const { return nodes.first == nodes.second; }
};
//! [gbts eta bin info]

/// What a node's entry in `GbtsNodeView::stripIndex` holds when it carries no
/// stereo pair, i.e. when it is a pixel node.
constexpr std::uint32_t kNoStrip = std::numeric_limits<std::uint32_t>::max();

/// Read-only view of the node attributes needed outside the graph builder.
///
/// Positions are packed as (x, y, z, r) because every consumer reads them
/// together, by node index.
struct GbtsNodeView final {
  /// Packed (x, y, z, r) per node.
  std::span<const std::array<float, 4>> positions;
  /// Dense layer index per node.
  std::span<const GbtsLayerIndex> layers;
  /// Stereo pairs of the strip nodes, reached through `stripIndex`.
  std::span<const OuterStripSpacePointCalibrationDetailsDerived> strips;
  /// Index into `strips` per node, `kNoStrip` where a node carries none.
  /// Empty when nothing does.
  std::span<const std::uint32_t> stripIndex;

  /// The stereo pair a node was formed from.
  /// @param index Index of the node
  /// @return The pair, or null for a pixel node
  const OuterStripSpacePointCalibrationDetailsDerived* strip(
      SpacePointIndex index) const {
    if (stripIndex.empty() || stripIndex[index] == kNoStrip) {
      return nullptr;
    }
    return &strips[stripIndex[index]];
  }
};

/// The circle through three nodes, mapped conformally about one of them to the
/// line v = A u + B through the other two. Its curvature is
/// B / sqrt(1 + A * A).
/// @param view View of the node positions
/// @param center The node the mapping is centred on
/// @param others The other two nodes
/// @return A and B, or nothing where the two nodes map to the same u
inline std::optional<std::array<float, 2>> conformalLine(
    const GbtsNodeView& view, const SpacePointIndex center,
    const std::array<SpacePointIndex, 2>& others) {
  // (x, y, z, r)
  const std::array<float, 4>& n0 = view.positions[center];

  const float x0 = n0[0];
  const float y0 = n0[1];
  const float cosA = x0 / n0[3];
  const float sinA = y0 / n0[3];

  std::array<float, 2> u{};
  std::array<float, 2> v{};
  for (std::uint32_t k = 0; k < 2; k++) {
    const std::array<float, 4>& nk = view.positions[others[k]];

    const float dx = nk[0] - x0;
    const float dy = nk[1] - y0;
    const float r2Inv = 1.0f / (dx * dx + dy * dy);

    u[k] = (dx * cosA + dy * sinA) * r2Inv;
    v[k] = (-dx * sinA + dy * cosA) * r2Inv;
  }

  const float du = u[0] - u[1];
  if (du == 0.0) {
    return std::nullopt;
  }

  const float A = (v[0] - v[1]) / du;
  return std::array<float, 2>{A, v[1] - A * u[1]};
}

//! [gbts edge]
/// Edge between two GBTS nodes with fit parameters.
struct GbtsEdge final {
  /// Constructor
  /// @param n1_ Inner node index
  /// @param n2_ Outer node index
  /// @param tauRatioCut_ Tau ratio cut of this doublet as an outer pair
  /// @param n2PixelBarrel_ Whether the outer node is on the pixel barrel
  /// @param p1_ First fit parameter
  /// @param p2_ Second fit parameter
  /// @param p3_ Third fit parameter
  GbtsEdge(SpacePointIndex n1_, SpacePointIndex n2_, float tauRatioCut_,
           bool n2PixelBarrel_, float p1_, float p2_, float p3_)
      : n1{n1_},
        n2{n2_},
        n2PixelBarrel{n2PixelBarrel_},
        p{p1_, p2_, p3_},
        tauRatioCut{tauRatioCut_} {}

  /// Inner node of the edge
  SpacePointIndex n1{kSpacePointIndexInvalid};
  /// Outer node of the edge
  SpacePointIndex n2{kSpacePointIndexInvalid};

  std::uint8_t nNei{0};

  /// Whether the outer node is on the pixel barrel. Sits in padding.
  bool n2PixelBarrel{};

  std::array<float, 3> p{};

  /// GbtsLayerConnection::tauRatioCutOuter of this doublet's layer pair,
  /// cached for the neighbour loop.
  float tauRatioCut{};

  std::array<std::uint32_t, kGbtsMaxEdgeNeighbours> vNei{};
};
//! [gbts edge]

}  // namespace Acts::Experimental::detail

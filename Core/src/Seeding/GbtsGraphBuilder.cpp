// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/GbtsGraphBuilder.hpp"

#include "Acts/SpacePointFormation/detail/StripSpacePointCalibrationImpl.hpp"
#include "Acts/Utilities/MathHelpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Acts::Experimental {

namespace {

/// Sliding window in phi used to define range used for edge creation.
///
/// Covers one non-empty source eta bin, whose phi-ordered node list it holds
/// directly so that the innermost loop does not reach through the bin.
struct SlidingWindow {
  /// phi-ordered nodes of the bin, including the wrap-around duplicates
  const std::pair<float, SpacePointIndex>* phiNodes{};
  /// number of entries in @c phiNodes
  std::uint32_t numPhiNodes{};
  /// sliding window position
  std::uint32_t firstIt{};
  /// window half-width;
  float deltaPhi{};
  /// Inside-out pixel barrel ordinal of the bin's layer, -1 for the rest.
  std::int32_t barrelOrder{-1};
  /// Tau ratio cuts of the bin's layer pair, as inner and as outer pair
  float tauRatioCutInner{};
  float tauRatioCutOuter{};
  /// Technology of the bin's layer.
  GbtsLayerTechnology technology{};
  /// Radial and z extent of the bin's nodes.
  float minR{};
  float maxR{};
  float minZ{};
  float maxZ{};
};

/// Check a triplet against the pT and d0 cuts.
/// @param cfg Graph builder configuration
/// @param nodeView View of the node positions and layers
/// @param candidateTriplet The three graph nodes
/// @param tripletMinPt Minimum transverse momentum
/// @param tauRatio Tau ratio of the triplet
/// @param bFieldInZ Magnetic field in z, in GeV/(e*mm)
/// @return Whether the triplet is accepted
bool validateTriplet(const GbtsGraphBuilder::Config& cfg,
                     const detail::GbtsNodeView& nodeView,
                     const std::array<SpacePointIndex, 3>& candidateTriplet,
                     const float tripletMinPt, const float tauRatio,
                     const float bFieldInZ) {
  // conformal mapping with the center at the middle spacepoint
  const std::optional<std::array<float, 2>> line =
      detail::conformalLine(nodeView, candidateTriplet[1],
                            {candidateTriplet[0], candidateTriplet[2]});
  if (!line) {
    return false;
  }

  const auto [A, B] = *line;

  const float r0 = nodeView.positions[candidateTriplet[1]][3];

  // the transverse impact parameter
  if (std::abs(r0 * (B * r0 - A)) > cfg.d0Max) {
    return false;
  }

  // a straight-line track is OK
  if (B == 0.0) {
    return true;
  }

  const float R = std::sqrt(1 + A * A) / B;

  // 1T magnetic field used
  const float pT = std::abs(bFieldInZ * R / 2);

  // a relatively high-pT track takes a tighter tau ratio
  return !(pT < tripletMinPt ||
           (pT > 5 * tripletMinPt && tauRatio > 0.9f * cfg.tauRatioCut));
}

/// Check to see if z0 of segment is within the expected z range of the
/// beamspot
/// @param cfg Graph builder configuration
/// @param z0BitMask Sets allowed bins of allowed z value
/// @param z0 Estimated z0 of segments z value at beamspot
/// @param z0HistoCoeff Scalfactor that converts z coodindate into bin index
/// @return Whether segment is within beamspot range; never for an empty mask
///
/// Computed as a value, without a branch, so that the caller folds it into
/// one branch with its other doublet cuts: whether a pair points at a filled
/// bin is data the predictor cannot learn.
bool checkZ0BitMask(const GbtsGraphBuilder::Config& cfg,
                    const std::uint16_t z0BitMask, const float z0,
                    const float z0HistoCoeff) {
  // z0 is not yet range checked here -- doubletFilterRZ is folded in beside
  // it -- so a bin outside the histogram, below it too as a wrapped unsigned,
  // is reachable. It is never set, and the shift is clamped only to be defined.
  constexpr auto kBins =
      static_cast<std::uint32_t>(detail::kGbtsZ0HistogramBins);
  const auto isSet = [z0BitMask](const std::int32_t bin) {
    const auto u = static_cast<std::uint32_t>(bin);
    return (u < kBins) & (((z0BitMask >> std::min(u, kBins - 1)) & 1) != 0);
  };

  // the bins of z0 -/+ its resolution; z0's own bin is one of the two, the
  // resolution being under half a bin
  const float dz = z0 - cfg.minZ0;
  const auto binDown =
      static_cast<std::int32_t>(z0HistoCoeff * (dz - cfg.z0Resolution));
  const auto binUp =
      static_cast<std::int32_t>(z0HistoCoeff * (dz + cfg.z0Resolution));

  return isSet(binDown) | isSet(binUp);
}

}  // namespace

GbtsGraphBuilder::GbtsGraphBuilder(const Config& config,
                                   std::shared_ptr<const GbtsGeometry> geometry,
                                   std::unique_ptr<const Acts::Logger> logger)
    : m_cfg(config),
      m_geometry(std::move(geometry)),
      m_logger(std::move(logger)) {}

GbtsGraph GbtsGraphBuilder::buildTheGraph(const GbtsRoiDescriptor& roi,
                                          GbtsNodeStorage& nodeStorage,
                                          const float bFieldInZ) const {
  GbtsGraph graph;
  std::vector<detail::GbtsEdge>& edgeStorage = graph.edgeStorage;

  // used to calculate Z cut on doublets
  const float cutZMinU =
      m_cfg.minZ0 + m_cfg.maxOuterRadius * static_cast<float>(roi.dzdrMin());
  const float cutZMaxU =
      m_cfg.maxZ0 + m_cfg.maxOuterRadius * static_cast<float>(roi.dzdrMax());

  const float tripletPtMin = m_cfg.tripletPtFraction * m_cfg.minPt;

  const float ptScale = m_cfg.tuningPt / m_cfg.minPt;

  const float curvatureCutHighEta = m_cfg.maxCurvatureHighEta * ptScale;
  const float curvatureCutLowEta = m_cfg.maxCurvatureLowEta * ptScale;

  // the default sliding window along phi. Taken from the node storage so that
  // the windows and the phi indexing they slide over cannot disagree.
  const float deltaPhi0 = 0.5f * nodeStorage.m_cfg.phiSliceWidth;

  std::uint32_t nConnections = 0;

  // good matches dropped because the outer edge held the maximum number of
  // neighbours
  std::uint32_t nOverflow = 0;

  // Scales the maximum number of edges by the number of space points using the
  // factor maxEdgesPerSP
  const auto maxEdges = static_cast<std::size_t>(
      m_cfg.maxEdgesPerSP * static_cast<float>(nodeStorage.numberOfNodes()));

  edgeStorage.reserve(maxEdges);
  graph.linked.assign(maxEdges / 64 + 1, 0);

  // scale factor to get indexes of binned beamspot
  const float z0HistoCoeff =
      detail::kGbtsZ0HistogramBins / (m_cfg.maxZ0 - m_cfg.minZ0 + 1e-6f);

  const detail::GbtsNodeView nodeView = nodeStorage.nodeView();
  const std::span<const detail::GbtsNodeParams> params = nodeStorage.m_params;
  const std::span<detail::GbtsNodeEdgeInfo> edgeInfo = nodeStorage.m_edgeInfo;

  // reused across bin groups so that the windows are allocated once
  std::vector<SlidingWindow> phiSlidingWindow;

  // A bin's connected nodes alone, for the bin groups that skip the rest
  // (useZ0Histogram). Which they are changes only as the bin's own group
  // runs, so a list is kept across the groups that read it, and made again
  // only where that group ran after it: by the count of groups begun when it
  // was made, and when the bin's own group last ended.
  const std::uint32_t numBins = m_geometry->numBins();
  std::vector<std::vector<std::pair<float, SpacePointIndex>>> connectedPhiNodes(
      numBins);
  std::vector<std::uint32_t> connectedMadeAt(numBins, 0);
  std::vector<std::uint32_t> connectionsChangedAt(numBins, 0);
  std::uint32_t groupCount = 0;

  // the window positions of the n2 nodes that pass the doublet cuts
  std::vector<std::uint32_t> passingN2;
  // and their tau and curvature, as the doublet cuts made them
  std::vector<float> passingTau;
  std::vector<float> passingCurv;

  const bool calibrate = m_cfg.calibrateStrips && nodeStorage.hasStrips();

  // Put a strip node back where a direction says it crossed. Both ends, since
  // the nominal position sits on the inner strip and the calibrated one on the
  // outer. A direction that misses either strip is no crossing at all.
  const auto calibrateNode = [&](const SpacePointIndex node,
                                 const std::array<float, 3>& direction,
                                 float& r, float& z) {
    std::array<float, 3> point{};
    if (!Acts::detail::calibrateOuterStripSpacePoint(
            direction, nodeStorage.strip(node), point,
            m_cfg.maxStripLengthFraction)) {
      return false;
    }
    r = fastHypot(point[0], point[1]);
    z = point[2];
    return true;
  };

  // loop over bin groups
  for (const auto& bg : m_geometry->binGroups()) {
    ++groupCount;
    const detail::GbtsEtaBinInfo& B1 = nodeStorage.m_etaBins[bg.bin];

    if (B1.empty()) {
      continue;
    }

    const std::int32_t barrelOrder1 = B1.barrelOrder;

    const bool isPixel1 = B1.technology == GbtsLayerTechnology::Pixel;
    const bool isPixelBarrel1 = barrelOrder1 >= 0;

    const bool useZ0Histogram =
        barrelOrder1 >= 0 && barrelOrder1 <= m_cfg.z0HistogramMaxBarrelOrder;
    const bool useMatchBeforeCreate =
        m_cfg.matchBeforeCreate && barrelOrder1 >= 0 &&
        barrelOrder1 <= m_cfg.matchBeforeCreateMaxBarrelOrder;

    // prepare a sliding window for each non-empty bin2 in the group

    phiSlidingWindow.clear();

    // loop over n2 eta-bins in L2 layers
    for (const GbtsBinLink& link : bg.links) {
      const std::uint32_t b2Idx = link.bin;
      const detail::GbtsEtaBinInfo& B2 = nodeStorage.m_etaBins[b2Idx];

      if (B2.empty()) {
        continue;
      }

      // the default window width, or one that grows with the radial gap
      const float absDr = std::fabs(B2.maxRadius - B1.minRadius);
      const float deltaPhi =
          !m_cfg.useEtaBinning ? deltaPhi0
          : absDr < m_cfg.phiWindowSplitDeltaRadius
              ? m_cfg.phiWindowNearOffset +
                    m_cfg.phiWindowNearSlope * ptScale * absDr
              : m_cfg.phiWindowFarOffset +
                    m_cfg.phiWindowFarSlope * ptScale * absDr;

      SlidingWindow& window = phiSlidingWindow.emplace_back();
      window.phiNodes = B2.phiNodes.data();
      window.numPhiNodes = static_cast<std::uint32_t>(B2.phiNodes.size());
      window.deltaPhi = deltaPhi;
      window.barrelOrder = B2.barrelOrder;
      window.tauRatioCutInner = link.tauRatioCutInner;
      window.tauRatioCutOuter = link.tauRatioCutOuter;
      window.technology = B2.technology;
      window.minR = B2.minRadius;
      window.maxR = B2.maxRadius;
      window.minZ = B2.minZ;
      window.maxZ = B2.maxZ;

      if (passingN2.size() < B2.phiNodes.size()) {
        passingN2.resize(B2.phiNodes.size());
        passingTau.resize(B2.phiNodes.size());
        passingCurv.resize(B2.phiNodes.size());
      }

      // The isolated nodes of another bin are skipped for every n1 of this
      // group below, and no n1 of it changes which they are: leave them out
      // of its window once, rather than test each for each n1, where whether
      // a node is isolated is data the predictor cannot learn. A bin of the
      // group itself is left as it is, its nodes still being connected.
      if (useZ0Histogram && b2Idx != bg.bin) {
        std::vector<std::pair<float, SpacePointIndex>>& connected =
            connectedPhiNodes[b2Idx];
        if (connectedMadeAt[b2Idx] <= connectionsChangedAt[b2Idx]) {
          connected.clear();
          for (const auto& phiNode : B2.phiNodes) {
            if (edgeInfo[phiNode.second].isConnected != 0) {
              connected.push_back(phiNode);
            }
          }
          connectedMadeAt[b2Idx] = groupCount;
        }
        window.phiNodes = connected.data();
        window.numPhiNodes = static_cast<std::uint32_t>(connected.size());
      }
    }

    // in GBTSv3 the outer loop goes over n1 nodes in the Layer 1 bin
    for (SpacePointIndex n1Idx = B1.nodes.first; n1Idx < B1.nodes.second;
         ++n1Idx) {
      // n1's edges start at the top watermark of the edge storage
      const auto firstEdge = static_cast<std::uint32_t>(edgeStorage.size());

      // the counter for the incoming graph edges created for n1
      std::uint16_t numCreatedEdges = 0;

      // the z0 bins of n1's edges that found a neighbour; non-zero marks n1
      // as connected
      std::uint16_t z0BitMask = 0;

      const detail::GbtsNodeParams& n1pars = params[n1Idx];

      const float phi1 = n1pars.phi;
      const float r1 = n1pars.r;
      const float z1 = n1pars.z;

      // n1's part of the tau window, the same for every window and every n2
      const float hi1 = std::min(m_cfg.maxAbsTau, n1pars.maxTau);
      const float lo1 = n1pars.minTau;

      // The tau a pair of n1 can pass the cuts below with: n1's window, and
      // z0 and zOuter in range where doubletFilterRZ asks for them. A window
      // whose bin it cannot reach with any of it is skipped whole below.
      // The r-z cuts as a tau window of n1's alone: z0 = z1 - r1 * tau and
      // zOuter = z1 + (maxOuterRadius - r1) * tau, each in its range, so that
      // a pair below tests its tau against two bounds.
      float rzLo = std::numeric_limits<float>::lowest();
      float rzHi = std::numeric_limits<float>::max();
      if (m_cfg.doubletFilterRZ) {
        rzLo = (z1 - m_cfg.maxZ0) / r1;
        rzHi = (z1 - m_cfg.minZ0) / r1;
        const float outerDr = m_cfg.maxOuterRadius - r1;
        if (outerDr != 0) {
          const auto [outerLo, outerHi] = std::minmax(
              {(cutZMinU - z1) / outerDr, (cutZMaxU - z1) / outerDr});
          rzLo = std::max(rzLo, outerLo);
          rzHi = std::min(rzHi, outerHi);
        } else if (z1 < cutZMinU || z1 > cutZMaxU) {
          rzLo = std::numeric_limits<float>::max();
          rzHi = std::numeric_limits<float>::lowest();
        }
      }
      const float reachLo = std::max(-hi1, rzLo);
      const float reachHi = std::min(hi1, rzHi);

      // the intermediate loop over sliding windows
      for (auto& slw : phiSlidingWindow) {
        const std::int32_t barrelOrder2 = slw.barrelOrder;

        const bool isPixel2 = slw.technology == GbtsLayerTechnology::Pixel;
        const bool isPixelBarrel2 = barrelOrder2 >= 0;

        const bool stripPair = calibrate && (!isPixel1 || !isPixel2);

        // No pair of the window is in reach: every one would fail the radius,
        // tau or r-z cuts below. A strip pair's ends move, so it is not judged
        // here.
        if (!stripPair) {
          if (slw.maxR - r1 < m_cfg.minDeltaRadius || reachLo > reachHi) {
            continue;
          }
          // Where a pair with tau in reach lands, z2 - z1 being tau * dr.
          // Generous by a millimetre, against rounding: it only ever skips
          // pairs the exact cuts below would reject.
          const float drLo = std::max(slw.minR - r1, m_cfg.minDeltaRadius);
          const float drHi = slw.maxR - r1;
          const float zLo =
              z1 + std::min(reachLo * drLo, reachLo * drHi) - 1.0f;
          const float zHi =
              z1 + std::max(reachHi * drLo, reachHi * drHi) + 1.0f;
          if (zHi < slw.minZ || zLo > slw.maxZ) {
            continue;
          }
        }

        // sliding window phi1 +/- deltaPhi
        const float minPhi = phi1 - slw.deltaPhi;
        const float maxPhi = phi1 + slw.deltaPhi;

        // The window's nodes that pass the doublet cuts below first, kept in
        // order without a branch on whether each does: which cut a pair
        // misses is data the predictor cannot learn, while for the nodes kept
        // every cut below passes. The radius cut stays a branch, as it is
        // predictable. A strip pair is kept for the cuts below alone.
        std::uint32_t numPassing = 0;
        for (std::uint32_t k = slw.firstIt; k < slw.numPhiNodes; ++k) {
          const float phi2 = slw.phiNodes[k].first;

          if (phi2 < minPhi) {
            // update the window position
            slw.firstIt = k;
            continue;
          }
          if (phi2 > maxPhi) {
            // break and go to the next window
            break;
          }

          const SpacePointIndex n2Idx = slw.phiNodes[k].second;
          const detail::GbtsNodeParams& n2pars = params[n2Idx];
          const float dr = n2pars.r - r1;
          if (dr < m_cfg.minDeltaRadius) {
            continue;
          }
          const std::uint16_t nodeInfo = edgeInfo[n2Idx].isConnected;
          const float tau = (n2pars.z - z1) / dr;
          const float ftau = std::fabs(tau);
          const float z0 = z1 - r1 * tau;
          const float curv = (phi2 - phi1) / dr;
          const float curvCut = ftau < m_cfg.curvatureSplitAbsTau
                                    ? curvatureCutLowEta
                                    : curvatureCutHighEta;
          const bool inZ0Histogram =
              !useZ0Histogram ||
              checkZ0BitMask(m_cfg, nodeInfo, z0, z0HistoCoeff);
          const bool inRZ = (tau >= rzLo) & (tau <= rzHi);
          passingN2[numPassing] = k;
          passingTau[numPassing] = tau;
          passingCurv[numPassing] = curv;
          numPassing += static_cast<std::uint32_t>(
              stripPair | ((ftau >= std::max(lo1, n2pars.minTau)) &
                           (ftau <= std::min(hi1, n2pars.maxTau)) &
                           inZ0Histogram & inRZ &
                           (std::abs(curv) <= curvCut)));
        }

        // the inner loop over the n2 nodes kept
        for (std::uint32_t i = 0; i < numPassing; ++i) {
          const std::uint32_t n2PhiIdx = passingN2[i];
          const float phi2 = slw.phiNodes[n2PhiIdx].first;

          const SpacePointIndex n2Idx = slw.phiNodes[n2PhiIdx].second;

          const detail::GbtsNodeEdgeInfo& n2Info = edgeInfo[n2Idx];

          const std::uint16_t nodeInfo = n2Info.isConnected;

          const std::uint32_t n2FirstEdge = n2Info.firstEdge;
          const std::uint16_t n2NumEdges = n2Info.numEdges;
          const std::uint32_t n2LastEdge = n2FirstEdge + n2NumEdges;

          // the ends as the pair puts them: nominal, or slid along the strip
          // when resolved
          float r1c = r1;
          float z1c = z1;
          float r2c = 0.f;
          float tau = 0.f;
          float curv = 0.f;

          if (!stripPair) {
            // The doublet cuts above made tau and the curvature from the same
            // nominal ends, as below, and the pair passed every one of them.
            r2c = params[n2Idx].r;
            tau = passingTau[i];
            curv = passingCurv[i];
          } else {
            // skip isolated nodes as their incoming edges lead to nowhere;
            // the cuts above left out the others' already
            if (useZ0Histogram && (nodeInfo == 0)) {
              continue;
            }

            const detail::GbtsNodeParams& n2pars = params[n2Idx];

            const float r2 = n2pars.r;

            float dr = r2 - r1;

            // On the nominal radii, so an endcap pair the slide would have
            // opened up is lost here. The cheap reject is worth more.
            if (dr < m_cfg.minDeltaRadius) {
              continue;
            }

            const float z2 = n2pars.z;

            // Azimuth is kept as it was -- exactly right in the barrel, and to
            // the stereo angle in the endcap, where the strip slid along is a
            // hair off radial.
            r2c = r2;
            float z2c = z2;

            // A circle's chord bisects the tangent directions, so turning the
            // chord by the pair's own azimuth difference gives each tangent.
            const std::array<float, 4>& p1 = nodeView.positions[n1Idx];
            const std::array<float, 4>& p2 = nodeView.positions[n2Idx];
            const float dx = p2[0] - p1[0];
            const float dy = p2[1] - p1[1];
            const float dzc = z2 - z1;
            const float turn = phi2 - phi1;

            bool resolved = true;
            if (!isPixel1) {
              resolved = calibrateNode(
                  n1Idx, {dx + dy * turn, dy - dx * turn, dzc}, r1c, z1c);
            }
            if (resolved && !isPixel2) {
              resolved = calibrateNode(
                  n2Idx, {dx - dy * turn, dy + dx * turn, dzc}, r2c, z2c);
            }
            if (!resolved) {
              continue;
            }

            dr = r2c - r1c;
            if (dr < m_cfg.minDeltaRadius) {
              continue;
            }

            const float dz = z2c - z1c;
            tau = dz / dr;
            const float ftau = std::fabs(tau);
            // The five tau windows as one branch: a pair is either in all of
            // them or out, and which one it misses is data the predictor cannot
            // learn. tau is never NaN here, since dr >= minDeltaRadius > 0, so
            // each !(a > b) is exactly a <= b.
            if (!((ftau >= std::max(lo1, n2pars.minTau)) &
                  (ftau <= std::min(hi1, n2pars.maxTau)))) {
              continue;
            }

            const float z0s = z1c - r1c * tau;

            // The z0 histogram, the r-z cuts and the curvature cut as one
            // branch, as above.
            const bool inZ0Histogram =
                !useZ0Histogram ||
                checkZ0BitMask(m_cfg, nodeInfo, z0s, z0HistoCoeff);

            const float zOuter = z0s + m_cfg.maxOuterRadius * tau;
            const bool inRZ = !m_cfg.doubletFilterRZ ||
                              ((z0s >= m_cfg.minZ0) & (z0s <= m_cfg.maxZ0) &
                               (zOuter >= cutZMinU) & (zOuter <= cutZMaxU));

            curv = (phi2 - phi1) / dr;
            const float curvCut = ftau < m_cfg.curvatureSplitAbsTau
                                      ? curvatureCutLowEta
                                      : curvatureCutHighEta;

            if (!(inZ0Histogram & inRZ & (std::abs(curv) <= curvCut))) {
              continue;
            }
          }

          const float z0 = z1c - r1c * tau;

          const float hypotTau = fastHypot(1, tau);
          const float expEta = hypotTau - tau;
          // 1 / expEta, since (hypotTau - tau) * (hypotTau + tau) == 1. The
          // sum is also the better conditioned form for large tau.
          const float invExpEta = hypotTau + tau;

          // match edge candidate against edges incoming to n2
          if (useMatchBeforeCreate) {
            // we must have enough incoming edges to decide
            bool isGood = n2NumEdges <= m_cfg.matchBeforeCreateMaxEdges;

            if (!isGood) {
              for (std::uint32_t n2InIdx = n2FirstEdge; n2InIdx < n2LastEdge;
                   ++n2InIdx) {
                const float tau2 = edgeStorage[n2InIdx].p[0];
                const float tauRatio = tau2 * invExpEta - 1.0f;

                if (std::abs(tauRatio) > m_cfg.tauRatioPrecut) {  // bad match
                  continue;
                }
                isGood = true;  // good match found
                break;
              }
            }

            if (!isGood) {  // no match found, skip creating [n1 <- n2] edge
              continue;
            }
          }

          if (edgeStorage.size() < maxEdges) {
            const auto outEdgeIdx =
                static_cast<std::uint32_t>(edgeStorage.size());

            edgeStorage.emplace_back(n1Idx, n2Idx, slw.tauRatioCutOuter,
                                     isPixelBarrel2, expEta, curv,
                                     phi1 + curv * r1c);

            ++numCreatedEdges;

            const float phi2u = phi2 + curv * r2c;

            // looking for neighbours of the new edge
            for (std::uint32_t inEdgeIdx = n2FirstEdge; inEdgeIdx < n2LastEdge;
                 ++inEdgeIdx) {
              detail::GbtsEdge* pS = &edgeStorage[inEdgeIdx];

              const float absTauRatio = std::abs(pS->p[0] * invExpEta - 1.0f);

              // the looser of the cuts the two layer pairs carry
              if (absTauRatio >
                  std::max(slw.tauRatioCutInner, pS->tauRatioCut)) {
                continue;  // bad match
              }

              const bool isPixelBarrel3 = pS->n2PixelBarrel;

              float dPhi = phi2u - pS->p[2];

              if (dPhi < -std::numbers::pi_v<float>) {
                dPhi += 2 * std::numbers::pi_v<float>;
              } else if (dPhi > std::numbers::pi_v<float>) {
                dPhi -= 2 * std::numbers::pi_v<float>;
              }

              if (std::abs(dPhi) > m_cfg.cutDPhiMax) {
                continue;
              }

              const float dcurv = curv - pS->p[1];

              // |dcurv| > cut is exactly dcurv < -cut || dcurv > cut, NaN too
              if (std::abs(dcurv) > m_cfg.cutDCurvMax) {
                continue;
              }

              // final check: cuts on pT and d0
              if (m_cfg.validateTriplets && isPixelBarrel1 && isPixelBarrel2 &&
                  isPixelBarrel3) {
                const std::array<SpacePointIndex, 3> candidateTriplet = {
                    n1Idx, n2Idx, pS->n2};

                if (!validateTriplet(m_cfg, nodeView, candidateTriplet,
                                     tripletPtMin, absTauRatio, bFieldInZ)) {
                  continue;
                }
              }

              // the match is good: it is lost only if the outer edge holds
              // the maximum number of neighbours already
              if (pS->nNei >= detail::kGbtsMaxEdgeNeighbours) {
                ++nOverflow;
                continue;
              }

              pS->vNei[pS->nNei] = outEdgeIdx;
              ++pS->nNei;
              graph.linked[inEdgeIdx / 64] |= std::uint64_t{1}
                                              << (inEdgeIdx % 64);

              // edge confirmed - mark its z0 bin. Only doubletFilterRZ holds
              // z0 to the histogram range, and it is optional.
              if (const auto z0BinIndex = static_cast<std::int32_t>(
                      z0HistoCoeff * (z0 - m_cfg.minZ0));
                  z0BinIndex >= 0 &&
                  z0BinIndex < detail::kGbtsZ0HistogramBins) {
                z0BitMask |= static_cast<std::uint16_t>(1u << z0BinIndex);
              }

              nConnections++;
            }
          }
        }  // loop over n2 (outer) nodes inside a sliding window on n2 bin
      }  // loop over sliding windows associated with n2 bins

      // updating the n1 node attributes

      edgeInfo[n1Idx] = {firstEdge, numCreatedEdges, z0BitMask};

    }  // loop over n1 (inner) nodes

    connectionsChangedAt[bg.bin] = groupCount;
  }  // loop over bin groups: a single n1 bin and multiple n2 bins

  if (edgeStorage.size() >= maxEdges) {
    ACTS_WARNING(
        "Maximum number of graph edges exceeded - possible efficiency loss "
        << edgeStorage.size());
  }
  if (nOverflow > 0) {
    ACTS_DEBUG(nOverflow << " edge links dropped: the outer edge already held "
                         << detail::kGbtsMaxEdgeNeighbours
                         << " neighbours - possible efficiency loss");
  }
  graph.nConnections = nConnections;
  graph.linked.resize(edgeStorage.size() / 64 + 1);
  return graph;
}

}  // namespace Acts::Experimental

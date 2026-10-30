// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/GbtsGeometry.hpp"

#include "Acts/Utilities/MathHelpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Acts::Experimental::detail {

float GbtsLayer::LayerPoint::eta() const {
  const float t = z / r;
  return -std::log(fastHypot(1, t) - t);
}

GbtsLayer::LayerPoint GbtsLayer::pointAt(const float bound) const {
  const GbtsLayerDescription& d = m_layerDescription;
  const float ref =
      d.refCoord + d.slope * (bound - 0.5f * (d.minBound + d.maxBound));
  return d.type == GbtsLayerType::Barrel ? LayerPoint{ref, bound}
                                         : LayerPoint{bound, ref};
}

float GbtsLayer::boundAt(const float eta) const {
  const GbtsLayerDescription& d = m_layerDescription;
  const float sh = std::sinh(eta);
  // reference coordinate the line would have at boundary coordinate zero
  const float ref0 = d.refCoord - d.slope * 0.5f * (d.minBound + d.maxBound);
  // barrel: r = ref0 + slope * z, endcap: z = ref0 + slope * r; and
  // z = r * sinh(eta)
  return d.type == GbtsLayerType::Barrel ? ref0 * sh / (1.f - d.slope * sh)
                                         : ref0 / (sh - d.slope);
}

GbtsLayer::GbtsLayer(const GbtsLayerDescription& layerDescription,
                     const float etaBinWidth, const std::uint32_t bin0)
    : m_layerDescription(layerDescription) {
  if (m_layerDescription.type != GbtsLayerType::Barrel &&
      m_layerDescription.type != GbtsLayerType::Endcap) {
    throw std::runtime_error("invalid layer type");
  }

  const float eta1 = pointAt(m_layerDescription.minBound).eta();
  const float eta2 = pointAt(m_layerDescription.maxBound).eta();

  // increasing them slightly to avoid range_check exceptions
  const auto minEta = static_cast<float>(std::min(eta1, eta2) - 1e-6);
  const auto maxEta = static_cast<float>(std::max(eta1, eta2) + 1e-6);

  const float deltaEta = maxEta - minEta;

  // a layer thinner than one bin width keeps a single bin spanning it
  std::uint32_t numBins = 1;
  if (deltaEta >= etaBinWidth) {
    numBins = static_cast<std::uint32_t>(deltaEta / etaBinWidth);
    if (deltaEta - etaBinWidth * numBins > 0.5f * etaBinWidth) {
      ++numBins;
    }
  }

  m_binning = {bin0, numBins, minEta, deltaEta / static_cast<float>(numBins)};

  if (numBins == 1) {
    // the single bin spans the layer, so it takes the layer's own bounds
    m_minBinCoord.push_back(m_layerDescription.minBound);
    m_maxBinCoord.push_back(m_layerDescription.maxBound);
    return;
  }

  for (std::uint32_t i = 1; i < numBins; ++i) {
    m_tauEdges.push_back(static_cast<float>(
        std::sinh(static_cast<double>(minEta) +
                  i * static_cast<double>(m_binning.etaBinWidth))));
  }

  float eta = minEta + 0.5f * m_binning.etaBinWidth;

  for (std::uint32_t i = 0; i < numBins; ++i) {
    // for the positive endcap larger eta corresponds to smaller radius
    const float c1 = boundAt(eta - 0.5f * m_binning.etaBinWidth);
    const float c2 = boundAt(eta + 0.5f * m_binning.etaBinWidth);
    m_minBinCoord.push_back(std::min(c1, c2));
    m_maxBinCoord.push_back(std::max(c1, c2));

    eta += m_binning.etaBinWidth;
  }
}

bool GbtsLayer::checkCompatibility(const GbtsLayer& otherLayer,
                                   const std::uint32_t b1,
                                   const std::uint32_t b2, const float minZ0,
                                   const float maxZ0,
                                   const float z0Tolerance) const {
  const std::array<LayerPoint, 2> ends1 = {pointAt(m_minBinCoord.at(b1)),
                                           pointAt(m_maxBinCoord.at(b1))};
  std::array<LayerPoint, 2> ends2 = {
      otherLayer.pointAt(otherLayer.m_minBinCoord.at(b2)),
      otherLayer.pointAt(otherLayer.m_maxBinCoord.at(b2))};

  if (this == &otherLayer) {
    // same layer link: the outer point is a layer thickness further out
    for (LayerPoint& p : ends2) {
      if (m_layerDescription.type == GbtsLayerType::Barrel) {
        p.r += m_layerDescription.layerThickness;
      } else {
        p.z += std::copysign(m_layerDescription.layerThickness, p.z);
      }
    }
  }

  constexpr float inf = std::numeric_limits<float>::infinity();
  float z0Min = inf;
  float z0Max = -inf;
  bool outwards = false;

  for (const LayerPoint& p1 : ends1) {
    for (const LayerPoint& p2 : ends2) {
      const float dr = p2.r - p1.r;
      const float dz = p2.z - p1.z;
      if (dr <= 0) {
        // pole in between: z0 diverges with the sign of z1 - z2
        z0Min = dz >= 0 ? -inf : z0Min;
        z0Max = dz <= 0 ? inf : z0Max;
        continue;
      }
      outwards = true;
      const float z0 = p1.z - p1.r * dz / dr;
      z0Min = std::min(z0Min, z0);
      z0Max = std::max(z0Max, z0);
    }
  }

  return outwards &&
         !(z0Max < minZ0 - z0Tolerance || z0Min > maxZ0 + z0Tolerance);
}

std::uint32_t GbtsLayer::getEtaBin(const float zh, const float rh) const {
  if (m_binning.numBins == 1) {
    return m_binning.firstBin;
  }

  const float t1 = zh / rh;

  // eta = asinh(t1) rises with t1, so away from the bin edges its bin is the
  // number of edges at or below t1. Within a hair of one, or where the
  // cancellation below costs eta its precision, the formula decides, as ever.
  if (std::abs(t1) < 8.0f) {
    const auto above = std::ranges::upper_bound(m_tauEdges, t1);
    const float tolerance = 1e-4f * (1.0f + std::abs(t1));
    const bool nearBelow =
        above != m_tauEdges.begin() && t1 - *(above - 1) < tolerance;
    const bool nearAbove = above != m_tauEdges.end() && *above - t1 < tolerance;
    if (!nearBelow && !nearAbove) {
      return m_binning.firstBin +
             static_cast<std::uint32_t>(above - m_tauEdges.begin());
    }
  }

  const float eta = -std::log(fastHypot(1, t1) - t1);

  // signed: eta below the layer's range truncates negative, before the clamp
  const auto rawIdx = static_cast<std::int32_t>((eta - m_binning.minEta) /
                                                m_binning.etaBinWidth);
  const auto idx = static_cast<std::uint32_t>(std::clamp<std::int32_t>(
      rawIdx, 0, static_cast<std::int32_t>(m_binning.numBins) - 1));

  // index in the global storage
  return m_binning.firstBin + idx;
}

}  // namespace Acts::Experimental::detail

namespace Acts::Experimental {

namespace {
// Fill in the inside-out pixel barrel order where the layers leave it unset.
std::vector<GbtsLayerDescription> orderPixelBarrel(
    std::span<const GbtsLayerDescription> layerDescriptions) {
  std::vector<GbtsLayerDescription> layers(layerDescriptions.begin(),
                                           layerDescriptions.end());

  std::vector<GbtsLayerDescription*> pixelBarrel;
  for (GbtsLayerDescription& layer : layers) {
    if (layer.type == GbtsLayerType::Barrel &&
        layer.technology == GbtsLayerTechnology::Pixel) {
      pixelBarrel.push_back(&layer);
    }
  }

  const auto numOrdered = std::ranges::count_if(
      pixelBarrel,
      [](const GbtsLayerDescription* l) { return l->barrelOrder >= 0; });
  if (numOrdered == 0) {
    // ties broken by id so the result does not depend on the input order
    std::ranges::sort(pixelBarrel, {}, [](const GbtsLayerDescription* l) {
      return std::pair{l->refCoord, l->id};
    });
    for (std::size_t i = 0; i < pixelBarrel.size(); ++i) {
      pixelBarrel[i]->barrelOrder = static_cast<std::int32_t>(i);
    }
  } else if (numOrdered != std::ssize(pixelBarrel)) {
    throw std::invalid_argument(
        "GbtsGeometry: barrelOrder must be set on every pixel barrel layer or "
        "on none of them");
  }

  return layers;
}
}  // namespace

void assignAdaptiveTauRatioCuts(std::span<GbtsLayerConnection> connections,
                                std::span<const GbtsLayerDescription> layers,
                                const GbtsAdaptiveTauRatioCuts& cuts) {
  std::unordered_map<GbtsExperimentLayerId, const GbtsLayerDescription*> byId;
  const std::vector<GbtsLayerDescription> ordered = orderPixelBarrel(layers);
  for (const GbtsLayerDescription& layer : ordered) {
    byId.emplace(layer.id, &layer);
  }
  const auto find = [&](GbtsExperimentLayerId id) {
    const auto it = byId.find(id);
    if (it == byId.end()) {
      throw std::invalid_argument(
          "assignAdaptiveTauRatioCuts: connection to an unknown layer");
    }
    return it->second;
  };

  for (GbtsLayerConnection& connection : connections) {
    const GbtsLayerDescription* outer = find(connection.src);
    const GbtsLayerDescription* inner = find(connection.dst);

    const bool outerInBarrel = outer->barrelOrder >= 0;
    const bool innerInBarrel = inner->barrelOrder >= 0;
    const bool skipsLayer = outerInBarrel && innerInBarrel &&
                            outer->barrelOrder - inner->barrelOrder != 1;
    const bool leavesBarrel = innerInBarrel && !outerInBarrel;
    const float strip = (outer->technology == GbtsLayerTechnology::Strip ||
                         inner->technology == GbtsLayerTechnology::Strip)
                            ? cuts.tauRatioCorrStrip
                            : 0.f;

    // leaving the barrel widens only the triplet that leaves after its middle
    connection.tauRatioCutInner =
        cuts.tauRatioCut + (skipsLayer ? cuts.tauRatioCorr : 0.f) + strip;
    connection.tauRatioCutOuter =
        cuts.tauRatioCut +
        (skipsLayer || leavesBarrel ? cuts.tauRatioCorr : 0.f) + strip;
  }
}

GbtsGeometry::GbtsGeometry(
    std::span<const GbtsLayerDescription> layerDescriptions,
    std::span<const GbtsLayerConnection> layerConnections,
    const float etaBinWidth, const GbtsZ0Range& z0Range, const Logger& logger) {
  // the innermost-layer cuts key on where a pixel barrel layer sits radially
  const std::vector<GbtsLayerDescription> layers =
      orderPixelBarrel(layerDescriptions);

  for (const GbtsLayerDescription& layer : layers) {
    const auto index = static_cast<GbtsLayerIndex>(m_layers.size());
    m_nEtaBins +=
        m_layers.emplace_back(layer, etaBinWidth, m_nEtaBins).binning().numBins;
    m_layerFromUserIdMap.try_emplace(layer.id, index);
  }

  // calculating bin tables in the connector...
  // calculate bin pairs for graph edge building

  std::vector<const detail::GbtsLayer*> binLayerMap(m_nEtaBins);

  // 1. a map of bin-to-bin connections, for the stages found below with the
  // graph ablation algorithm. key: bin. value: (outgoing, incoming) bins it
  // connects to.
  std::unordered_map<std::uint32_t, std::pair<std::vector<std::uint32_t>,
                                              std::vector<GbtsBinLink>>>
      binMap;

  const auto layerById = [this](const GbtsExperimentLayerId id) {
    const std::optional<GbtsLayerIndex> idx = layerIndex(id);
    return idx ? &m_layers[*idx] : nullptr;
  };

  for (const GbtsLayerConnection& connection : layerConnections) {
    if (connection.tauRatioCutInner < 0.f ||
        connection.tauRatioCutOuter < 0.f) {
      throw std::invalid_argument(
          "GbtsGeometry: layer connection with a negative tau ratio cut");
    }

    const detail::GbtsLayer* pL1 = layerById(connection.dst);  // n1
    const detail::GbtsLayer* pL2 = layerById(connection.src);  // n2
    if (pL1 == nullptr) {
      ACTS_WARNING("Skipping invalid dst layer " << connection.dst);
      continue;
    }
    if (pL2 == nullptr) {
      ACTS_WARNING("Skipping invalid src layer " << connection.src);
      continue;
    }

    // loop over bins in Layer 1
    for (std::uint32_t b1 = 0; b1 < pL1->binning().numBins; ++b1) {
      // loop over bins in Layer 2
      for (std::uint32_t b2 = 0; b2 < pL2->binning().numBins; ++b2) {
        if (!pL1->checkCompatibility(*pL2, b1, b2, z0Range.min, z0Range.max,
                                     z0Range.z0Tolerance)) {
          continue;
        }

        const std::uint32_t bin1Idx = pL1->binning().firstBin + b1;
        const std::uint32_t bin2Idx = pL2->binning().firstBin + b2;

        binLayerMap[bin1Idx] = pL1;
        binLayerMap[bin2Idx] = pL2;

        // incoming bin1 <- bin2
        binMap[bin1Idx].second.push_back(
            {bin2Idx, connection.tauRatioCutInner, connection.tauRatioCutOuter});
        binMap[bin2Idx].first.push_back(bin1Idx);  // outgoing bin2 -> bin1
      }
    }
  }

  // 2. find stages starting from the last one (i.e. bin1 with no outgoing
  // connections), and fill binGroups with them in reverse order: each stage
  // goes before the ones found already

  while (!binMap.empty()) {
    std::vector<std::uint32_t> exitBins;

    // 2a. find all bins with zero outgoing links if
    // remaining links form an intra-layer link circle

    for (const auto& bl : binMap) {
      const detail::GbtsLayer* layer1 = binLayerMap[bl.first];
      const bool isBarrel =
          layer1->layerDescription().type == GbtsLayerType::Barrel;
      // A bin with no outgoing links exits. A bin will never have empty
      // outLinks if it links back to itself.
      // The barrel can link its bins both ways. 1->2->1 will survive as a
      // pair, exit both together after all other outgoing links.
      // The endcap separates its bins in radius. Only a self link will
      // survive, exit it after all other outgoing links.
      if (std::ranges::all_of(bl.second.first, [&](const std::uint32_t bin2) {
            return binLayerMap[bin2] == layer1 &&
                   (bin2 == bl.first || isBarrel);
          })) {
        exitBins.push_back(bl.first);
      }
    }

    // order exit bins (important for graph building)
    std::ranges::sort(exitBins);

    // 2b. remove links : graph ablation, and the exit bins themselves; an
    // exit bin already removed needs no link taken out of it

    // every exit bin is a key of the map, taken from it above; the incoming
    // links are never changed by the ablation, which takes out outgoing ones
    std::vector<GbtsBinGroup> stage;
    for (const std::uint32_t& bin1Key : exitBins) {
      std::vector<GbtsBinLink>& incoming = binMap.at(bin1Key).second;
      for (const GbtsBinLink& bin2Link : incoming) {
        if (const auto p2 = binMap.find(bin2Link.bin); p2 != binMap.end()) {
          std::erase(p2->second.first, bin1Key);
        }
      }
      stage.push_back({bin1Key, std::move(incoming)});
      binMap.erase(bin1Key);
    }
    m_binGroups.insert(m_binGroups.begin(), stage.begin(), stage.end());
  }
}

std::optional<GbtsLayerIndex> GbtsGeometry::layerIndex(
    GbtsExperimentLayerId id) const {
  const auto it = m_layerFromUserIdMap.find(id);
  return it != m_layerFromUserIdMap.end() ? std::optional(it->second)
                                          : std::nullopt;
}

}  // namespace Acts::Experimental

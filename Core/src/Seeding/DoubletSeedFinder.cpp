// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/DoubletSeedFinder.hpp"

#include "Acts/EventData/SpacePointContainer.hpp"
#include "Acts/SpacePointFormation/detail/StripSpacePointCalibrationImpl.hpp"
#include "Acts/Utilities/MathHelpers.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>

#include <boost/mp11.hpp>
#include <boost/mp11/algorithm.hpp>

#include "detail/BlockPass.hpp"

namespace Acts {

namespace {

/// How the cuts on deltaZ and on the collision region are made, chosen in
/// create(). Where a range is symmetric about zero, it is tested with one
/// comparison on the absolute value; an unbounded deltaZ range rejects no
/// candidate, so its cut is not made.
enum class ZCuts {
  /// two comparisons each
  eGeneral,
  /// symmetric deltaZ range and collision region
  eSymmetric,
  /// symmetric collision region, unbounded deltaZ range
  eSymmetricNoDeltaZ,
};

template <bool isBottomCandidate, bool interactionPointCut, bool sortedByR,
          bool experimentCuts, bool useTime, bool stripCalibration, ZCuts zCuts>
class Impl final : public DoubletSeedFinder {
 public:
  explicit Impl(const DerivedConfig& config) : m_cfg(config) {}

  const DerivedConfig& config() const override { return m_cfg; }

  /// Iterates over dublets and tests the compatibility by applying a series of
  /// cuts that can be tested with only two SPs.
  ///
  /// @param config Doublet cuts that define the compatibility of space points
  /// @param middleSp Space point candidate to be used as middle SP in a seed
  /// @param middleSpInfo Information about the middle space point
  /// @param candidateSps Range or subet of space points to be used as candidates
  ///   for middle SP in a seed
  /// @param compatibleDoublets Output container for compatible doublets
  template <typename CandidateSps>
  void createDoubletsImpl(const ConstSpacePointProxy& middleSp,
                          const MiddleSpInfo& middleSpInfo,
                          CandidateSps candidateSps,
                          DoubletsForMiddleSp& compatibleDoublets) const {
    const float impactMax =
        isBottomCandidate ? -m_cfg.impactMax : m_cfg.impactMax;

    const float xM = middleSp.xy()[0];
    const float yM = middleSp.xy()[1];
    const float zM = middleSp.zr()[0];
    const float rM = middleSp.zr()[1];
    const float varianceZM = middleSp.varianceZ();
    const float varianceRM = middleSp.varianceR();

    // time of the middle space point and its variance. only filled when the
    // time cut is enabled
    [[maybe_unused]] float tM = 0;
    [[maybe_unused]] float varianceTM = 0;
    if constexpr (useTime) {
      tM = middleSp.time();
      varianceTM = middleSp.varianceT();
    }

    // equivalent to impactMax / (rM * rM);
    const float vIPAbs = impactMax * middleSpInfo.uIP2;

    // only filled with the strip calibration
    [[maybe_unused]] float calibrationTolerance = 0;
    [[maybe_unused]] OuterStripSpacePointCalibrationDetailsDerived
        calibrationM{};
    if constexpr (stripCalibration) {
      calibrationTolerance = *m_cfg.stripCalibrationTolerance;
      calibrationM = detail::deriveOuterStripSpacePointCalibrationDetails(
          middleSp.outerStripCalibrationDetails());
    }

    const auto outsideRangeCheck = [](const float value, const float min,
                                      const float max) {
      // intentionally using `|` after profiling. faster due to better branch
      // prediction
      return static_cast<bool>(static_cast<int>(value < min) |
                               static_cast<int>(value > max));
    };
    // Same result as outsideRangeCheck(value, -max, max) for every input, but
    // one comparison instead of two.
    const auto outsideSymmetricRange = [](const float value, const float max) {
      return std::abs(value) > max;
    };

    const auto calculateError = [&](float varianceZO, float varianceRO,
                                    float iDeltaR2, float cotTheta) {
      return iDeltaR2 * ((varianceZM + varianceZO) +
                         (cotTheta * cotTheta) * (varianceRM + varianceRO));
    };

    const SpacePointContainer& container = candidateSps.container();
    const auto xy = container.xyColumn().data();
    const auto zr = container.zrColumn().data();
    const auto varianceZ = container.varianceZColumn().data();
    const auto varianceR = container.varianceRColumn().data();

    // The candidates by their position in candidateSps: a range of them is
    // contiguous, a subset lists them.
    constexpr bool kContiguous =
        std::is_same_v<CandidateSps, SpacePointContainer::ConstRange>;
    const auto candidateIndex = [&](std::size_t k) -> SpacePointIndex {
      if constexpr (kContiguous) {
        return candidateSps.range().first + k;
      } else {
        return candidateSps.subset()[k];
      }
    };

    // Sorted by radius, the candidates inside deltaR end at the first one
    // outside it.
    std::size_t end = candidateSps.size();
    if constexpr (sortedByR) {
      const auto rOf = [&](std::size_t k) { return zr[candidateIndex(k)][1]; };
      assert(std::ranges::is_sorted(std::views::iota(std::size_t{0}, end), {},
                                    rOf));
      end = *std::ranges::partition_point(
          std::views::iota(std::size_t{0}, end), [&](std::size_t k) {
            return isBottomCandidate ? !(rM - rOf(k) < m_cfg.deltaRMin)
                                     : !(rOf(k) - rM > m_cfg.deltaRMax);
          });
    }

    // The config values the cuts on z and r read, as locals, so that they are
    // loaded once rather than for every candidate.
    const float deltaRMin = m_cfg.deltaRMin;
    const float deltaRMax = m_cfg.deltaRMax;
    [[maybe_unused]] const float deltaZMin = m_cfg.deltaZMin;
    [[maybe_unused]] const float deltaZMax = m_cfg.deltaZMax;
    [[maybe_unused]] const float collisionRegionMin = m_cfg.collisionRegionMin;
    const float collisionRegionMax = m_cfg.collisionRegionMax;

    // The candidates go through in blocks: the cuts on z and r are made for
    // every candidate of a block without a branch, and the rest of the loop
    // only for those that pass them.
    constexpr std::size_t kBlockSize = 256;
    // The candidates per step of the cuts on z and r: a subset's candidates
    // are scattered, so they go one at a time
    constexpr std::size_t kChunkSize = kContiguous ? 8 : 1;

    // One value per candidate of a block, or per survivor of its cuts on z
    // and r. Not cleared: each entry is written before it is read.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
    detail::PerCandidate<float, kBlockSize> deltaR;
    detail::PerCandidate<float, kBlockSize> deltaZ;
    [[maybe_unused]] detail::PerCandidate<float, kBlockSize> xNewFrame;
    [[maybe_unused]] detail::PerCandidate<float, kBlockSize> yNewFrame;
    [[maybe_unused]] detail::PerCandidate<float, kBlockSize> iDeltaR2;
    [[maybe_unused]] detail::PerCandidate<float, kBlockSize> uT;
    [[maybe_unused]] detail::PerCandidate<float, kBlockSize> vT;
    // NOLINTEND(cppcoreguidelines-pro-type-member-init)
    detail::PassFlags<kBlockSize> passes;
    [[maybe_unused]] detail::PassFlags<kBlockSize> keep;
    [[maybe_unused]] detail::PassFlags<kBlockSize> needsCurvatureCut;

    for (const detail::Block block : detail::blocks<kBlockSize>(0, end)) {
      block.forEach<kChunkSize>([&](std::size_t i, std::size_t k) {
        const std::array<float, 2>& zrO = zr[candidateIndex(k)];

        deltaR[i] = isBottomCandidate ? rM - zrO[1] : zrO[1] - rM;
        deltaZ[i] = isBottomCandidate ? zM - zrO[0] : zrO[0] - zM;

        bool outside = false;
        if constexpr (!sortedByR) {
          outside = outsideRangeCheck(deltaR[i], deltaRMin, deltaRMax);
        }

        if constexpr (zCuts == ZCuts::eGeneral) {
          outside |= outsideRangeCheck(deltaZ[i], deltaZMin, deltaZMax);
        } else if constexpr (zCuts == ZCuts::eSymmetric) {
          // deltaZMin is -deltaZMax
          outside |= outsideSymmetricRange(deltaZ[i], deltaZMax);
        }

        // the longitudinal impact parameter zOrigin is defined as (zM - rM *
        // cotTheta) where cotTheta is the ratio Z/R (forward angle) of space
        // point duplet but instead we calculate (zOrigin * deltaR) and multiply
        // collisionRegion by deltaR to avoid divisions
        const float zOriginTimesDeltaR = zM * deltaR[i] - rM * deltaZ[i];
        // check if duplet origin on z axis within collision region
        if constexpr (zCuts == ZCuts::eGeneral) {
          outside |= outsideRangeCheck(zOriginTimesDeltaR,
                                       collisionRegionMin * deltaR[i],
                                       collisionRegionMax * deltaR[i]);
        } else {
          // collisionRegionMin * deltaR is exactly -(collisionRegionMax *
          // deltaR), since negation is exact
          outside |= outsideSymmetricRange(zOriginTimesDeltaR,
                                           collisionRegionMax * deltaR[i]);
        }

        passes[i] = static_cast<std::uint32_t>(!outside);
      });

      const std::span<const std::uint32_t> survivors = passes.passing(block);

      // With the interaction point cut, the cuts that follow are made for
      // every survivor of a block without a branch too: the time cut, the
      // impact parameter test and the cotTheta cut, then the curvature cut
      // only for the survivors it can still reject.
      if constexpr (interactionPointCut && !stripCalibration) {
        for (std::size_t s = 0; s < survivors.size(); ++s) {
          const std::uint32_t i = survivors[s];
          const SpacePointIndex indexO = candidateIndex(block.first + i);
          const float xO = xy[indexO][0];
          const float yO = xy[indexO][1];

          // check the time compatibility of the two space points. the time
          // difference is corrected for the time of flight along the straight
          // line between them, assuming an outgoing particle at the speed of
          // light (which is 1 in ACTS units)
          bool inTime = true;
          if constexpr (useTime) {
            const ConstSpacePointProxy otherSp = container[indexO];
            const float distance =
                fastHypot(xO - xM, yO - yM, zr[indexO][0] - zM);
            const float dt = isBottomCandidate ? tM - otherSp.time() - distance
                                               : otherSp.time() - tM - distance;
            const float varianceDt = varianceTM + otherSp.varianceT();
            inTime = !(dt * dt > m_cfg.timeCutNSigma2 * varianceDt);
          }

          // transform SP coordinates to the u-v reference frame
          const float deltaX = xO - xM;
          const float deltaY = yO - yM;

          xNewFrame[s] =
              deltaX * middleSpInfo.cosPhiM + deltaY * middleSpInfo.sinPhiM;
          yNewFrame[s] =
              deltaY * middleSpInfo.cosPhiM - deltaX * middleSpInfo.sinPhiM;

          const float deltaR2 = deltaX * deltaX + deltaY * deltaY;
          iDeltaR2[s] = 1 / deltaR2;

          uT[s] = xNewFrame[s] * iDeltaR2[s];
          vT[s] = yNewFrame[s] * iDeltaR2[s];

          // We check the interaction point by evaluating the minimal distance
          // between the origin and the straight line connecting the two points
          // in the doublets. Using a geometric similarity, the Im is given by
          // yNewFrame * rM / deltaR > config.impactMax
          // However, we make here an approximation of the impact parameter
          // which is valid under the assumption yNewFrame / xNewFrame is small
          // The correct computation would be:
          // yNewFrame * yNewFrame * rM * rM > config.impactMax *
          // config.impactMax * deltaR2
          const bool ipTest =
              std::abs(rM * yNewFrame[s]) > impactMax * xNewFrame[s];

          // check if duplet cotTheta is within the region of interest
          // cotTheta is defined as (deltaZ / deltaR) but instead we multiply
          // cotThetaMax by deltaR to avoid division
          const bool cutCotTheta =
              outsideSymmetricRange(deltaZ[i], m_cfg.cotThetaMax * deltaR[i]);

          // kept unless cut on time or cotTheta; where ipTest holds, the
          // curvature cut below decides
          keep[s] =
              static_cast<std::uint32_t>(inTime && !cutCotTheta && !ipTest);
          needsCurvatureCut[s] =
              static_cast<std::uint32_t>(inTime && !cutCotTheta && ipTest);
        }

        for (const std::uint32_t s :
             needsCurvatureCut.passing(survivors.size())) {
          // in the rotated frame the interaction point is positioned at x =
          // -rM and y ~= impactParam
          const float vIP = (yNewFrame[s] > 0) ? -vIPAbs : vIPAbs;

          // we can obtain aCoef as the slope dv/du of the linear function,
          // estimated using du and dv between the two SP bCoef is obtained by
          // inserting aCoef into the linear equation
          const float aCoef = (vT[s] - vIP) / (uT[s] - middleSpInfo.uIP);
          const float bCoef = vIP - aCoef * middleSpInfo.uIP;
          // the distance of the straight line from the origin (radius of the
          // circle) is related to aCoef and bCoef by d^2 = bCoef^2 / (1 +
          // aCoef^2) = 1 / (radius^2) and we can apply the cut on the
          // curvature
          keep[s] = static_cast<std::uint32_t>(
              !((bCoef * bCoef) * m_cfg.minHelixDiameter2 > 1 + aCoef * aCoef));
        }

        for (const std::uint32_t s : keep.passing(survivors.size())) {
          const std::uint32_t i = survivors[s];
          const SpacePointIndex indexO = candidateIndex(block.first + i);

          const float iDeltaR = std::sqrt(iDeltaR2[s]);
          const float cotTheta = deltaZ[i] * iDeltaR;

          // discard doublets based on experiment specific cuts
          if constexpr (experimentCuts) {
            if (!m_cfg.experimentCuts(middleSp, container[indexO], cotTheta,
                                      isBottomCandidate)) {
              continue;
            }
          }

          const float er = calculateError(varianceZ[indexO], varianceR[indexO],
                                          iDeltaR2[s], cotTheta);

          // fill output vectors
          compatibleDoublets.emplace_back(indexO, cotTheta, iDeltaR, er, uT[s],
                                          vT[s], xNewFrame[s], yNewFrame[s]);
        }

        continue;
      }

      for (const std::uint32_t i : survivors) {
        const SpacePointIndex indexO = candidateIndex(block.first + i);
        float xO = xy[indexO][0];
        float yO = xy[indexO][1];
        float zO = zr[indexO][0];

        // check the time compatibility of the two space points. the time
        // difference is corrected for the time of flight along the straight
        // line between them, assuming an outgoing particle at the speed of
        // light (which is 1 in ACTS units). placed after the cheap range
        // checks above to avoid the square root for candidates that are
        // rejected anyway
        if constexpr (useTime) {
          const ConstSpacePointProxy otherSp = container[indexO];
          const float distance = fastHypot(xO - xM, yO - yM, zO - zM);
          float dt = 0;
          if constexpr (isBottomCandidate) {
            dt = tM - otherSp.time() - distance;
          } else {
            dt = otherSp.time() - tM - distance;
          }
          const float varianceDt = varianceTM + otherSp.varianceT();
          if (dt * dt > m_cfg.timeCutNSigma2 * varianceDt) {
            continue;
          }
        }

        // middle space point of this doublet, calibrated if enabled
        float xMd = xM;
        float yMd = yM;
        float rMd = rM;
        float uIP = middleSpInfo.uIP;
        float vIPAbsD = vIPAbs;

        // calibrate both space points against their chord
        if constexpr (stripCalibration) {
          const std::array<float, 3> chord{xO - xM, yO - yM, zO - zM};
          std::array<float, 3> calibratedM{};
          std::array<float, 3> calibratedO{};
          if (!detail::calibrateOuterStripSpacePoint(
                  chord, calibrationM, calibratedM, calibrationTolerance) ||
              !detail::calibrateOuterStripSpacePoint(
                  chord,
                  detail::deriveOuterStripSpacePointCalibrationDetails(
                      container[indexO].outerStripCalibrationDetails()),
                  calibratedO, calibrationTolerance)) {
            continue;
          }

          xMd = calibratedM[0];
          yMd = calibratedM[1];
          rMd = fastHypot(xMd, yMd);
          uIP = -1 / rMd;
          vIPAbsD = impactMax * uIP * uIP;

          xO = calibratedO[0];
          yO = calibratedO[1];
          zO = calibratedO[2];
          const float rOd = fastHypot(xO, yO);

          deltaR[i] = isBottomCandidate ? rMd - rOd : rOd - rMd;
          deltaZ[i] =
              isBottomCandidate ? calibratedM[2] - zO : zO - calibratedM[2];
        }

        // if interactionPointCut is false we apply z cuts before coordinate
        // transformation to avoid unnecessary calculations. If
        // interactionPointCut is true we apply the curvature cut first because
        // it is more frequent but requires the coordinate transformation
        if constexpr (!interactionPointCut) {
          // check if duplet cotTheta is within the region of interest
          // cotTheta is defined as (deltaZ / deltaR) but instead we multiply
          // cotThetaMax by deltaR to avoid division
          if (outsideSymmetricRange(deltaZ[i], m_cfg.cotThetaMax * deltaR[i])) {
            continue;
          }

          // transform SP coordinates to the u-v reference frame
          const float deltaX = xO - xMd;
          const float deltaY = yO - yMd;

          const float xNewFrame =
              deltaX * middleSpInfo.cosPhiM + deltaY * middleSpInfo.sinPhiM;
          const float yNewFrame =
              deltaY * middleSpInfo.cosPhiM - deltaX * middleSpInfo.sinPhiM;

          const float deltaR2 = deltaX * deltaX + deltaY * deltaY;
          const float iDeltaR2 = 1 / deltaR2;

          const float uT = xNewFrame * iDeltaR2;
          const float vT = yNewFrame * iDeltaR2;

          const float iDeltaR = std::sqrt(iDeltaR2);
          const float cotTheta = deltaZ[i] * iDeltaR;

          // discard doublets based on experiment specific cuts (also applied
          // here, in the interactionPointCut == false branch, not only in the
          // branch below -- otherwise the cut would silently never run when
          // the IP cut is disabled)
          if constexpr (experimentCuts) {
            if (!m_cfg.experimentCuts(middleSp, container[indexO], cotTheta,
                                      isBottomCandidate)) {
              continue;
            }
          }

          const float er = calculateError(varianceZ[indexO], varianceR[indexO],
                                          iDeltaR2, cotTheta);

          // fill output vectors
          compatibleDoublets.emplace_back(indexO, cotTheta, iDeltaR, er, uT, vT,
                                          xNewFrame, yNewFrame);
          continue;
        }

        // transform SP coordinates to the u-v reference frame
        const float deltaX = xO - xMd;
        const float deltaY = yO - yMd;

        const float xNewFrame =
            deltaX * middleSpInfo.cosPhiM + deltaY * middleSpInfo.sinPhiM;
        const float yNewFrame =
            deltaY * middleSpInfo.cosPhiM - deltaX * middleSpInfo.sinPhiM;

        const float deltaR2 = deltaX * deltaX + deltaY * deltaY;
        const float iDeltaR2 = 1 / deltaR2;

        const float uT = xNewFrame * iDeltaR2;
        const float vT = yNewFrame * iDeltaR2;

        // We check the interaction point by evaluating the minimal distance
        // between the origin and the straight line connecting the two points in
        // the doublets. Using a geometric similarity, the Im is given by
        // yNewFrame * rM / deltaR > config.impactMax
        // However, we make here an approximation of the impact parameter
        // which is valid under the assumption yNewFrame / xNewFrame is small
        // The correct computation would be:
        // yNewFrame * yNewFrame * rM * rM > config.impactMax *
        // config.impactMax * deltaR2
        if (std::abs(rMd * yNewFrame) > impactMax * xNewFrame) {
          // in the rotated frame the interaction point is positioned at x =
          // -rM and y ~= impactParam
          const float vIP = (yNewFrame > 0) ? -vIPAbsD : vIPAbsD;

          // we can obtain aCoef as the slope dv/du of the linear function,
          // estimated using du and dv between the two SP bCoef is obtained by
          // inserting aCoef into the linear equation
          const float aCoef = (vT - vIP) / (uT - uIP);
          const float bCoef = vIP - aCoef * uIP;
          // the distance of the straight line from the origin (radius of the
          // circle) is related to aCoef and bCoef by d^2 = bCoef^2 / (1 +
          // aCoef^2) = 1 / (radius^2) and we can apply the cut on the
          // curvature
          if ((bCoef * bCoef) * m_cfg.minHelixDiameter2 > 1 + aCoef * aCoef) {
            continue;
          }
        }

        // check if duplet cotTheta is within the region of interest
        // cotTheta is defined as (deltaZ / deltaR) but instead we multiply
        // cotThetaMax by deltaR to avoid division
        if (outsideSymmetricRange(deltaZ[i], m_cfg.cotThetaMax * deltaR[i])) {
          continue;
        }

        const float iDeltaR = std::sqrt(iDeltaR2);
        const float cotTheta = deltaZ[i] * iDeltaR;

        // discard doublets based on experiment specific cuts
        if constexpr (experimentCuts) {
          if (!m_cfg.experimentCuts(middleSp, container[indexO], cotTheta,
                                    isBottomCandidate)) {
            continue;
          }
        }

        const float er = calculateError(varianceZ[indexO], varianceR[indexO],
                                        iDeltaR2, cotTheta);

        // fill output vectors
        compatibleDoublets.emplace_back(indexO, cotTheta, iDeltaR, er, uT, vT,
                                        xNewFrame, yNewFrame);
      }
    }
  }

  void createDoublets(const ConstSpacePointProxy& middleSp,
                      const MiddleSpInfo& middleSpInfo,
                      SpacePointContainer::ConstSubset candidateSps,
                      DoubletsForMiddleSp& compatibleDoublets) const override {
    createDoubletsImpl(middleSp, middleSpInfo, candidateSps,
                       compatibleDoublets);
  }

  void createDoublets(const ConstSpacePointProxy& middleSp,
                      const MiddleSpInfo& middleSpInfo,
                      SpacePointContainer::ConstRange candidateSps,
                      DoubletsForMiddleSp& compatibleDoublets) const override {
    createDoubletsImpl(middleSp, middleSpInfo, candidateSps,
                       compatibleDoublets);
  }

 private:
  DerivedConfig m_cfg;
};

}  // namespace

std::unique_ptr<DoubletSeedFinder> DoubletSeedFinder::create(
    const DerivedConfig& config) {
  using BooleanOptions =
      boost::mp11::mp_list<std::bool_constant<false>, std::bool_constant<true>>;

  using IsBottomCandidateOptions = BooleanOptions;
  using InteractionPointCutOptions = BooleanOptions;
  using SortedByROptions = BooleanOptions;
  using ExperimentCutsOptions = BooleanOptions;
  using UseTimeOptions = BooleanOptions;
  using StripCalibrationOptions = BooleanOptions;
  using ZCutsOptions = boost::mp11::mp_list<
      std::integral_constant<ZCuts, ZCuts::eGeneral>,
      std::integral_constant<ZCuts, ZCuts::eSymmetric>,
      std::integral_constant<ZCuts, ZCuts::eSymmetricNoDeltaZ>>;

  using DoubletOptions =
      boost::mp11::mp_product<boost::mp11::mp_list, IsBottomCandidateOptions,
                              InteractionPointCutOptions, SortedByROptions,
                              ExperimentCutsOptions, UseTimeOptions,
                              StripCalibrationOptions, ZCutsOptions>;

  // Compared exactly: only an exactly symmetric range gives the same cut.
  const bool symmetricCollisionRegion =
      config.collisionRegionMin == -config.collisionRegionMax;
  const bool unboundedDeltaZ =
      config.deltaZMin == -std::numeric_limits<float>::infinity() &&
      config.deltaZMax == std::numeric_limits<float>::infinity();
  ZCuts configZCuts = ZCuts::eGeneral;
  if (symmetricCollisionRegion && unboundedDeltaZ) {
    configZCuts = ZCuts::eSymmetricNoDeltaZ;
  } else if (symmetricCollisionRegion &&
             config.deltaZMin == -config.deltaZMax) {
    configZCuts = ZCuts::eSymmetric;
  }

  std::unique_ptr<DoubletSeedFinder> result;
  boost::mp11::mp_for_each<DoubletOptions>([&](auto option) {
    using OptionType = decltype(option);

    using IsBottomCandidate = boost::mp11::mp_at_c<OptionType, 0>;
    using InteractionPointCut = boost::mp11::mp_at_c<OptionType, 1>;
    using SortedByR = boost::mp11::mp_at_c<OptionType, 2>;
    using ExperimentCuts = boost::mp11::mp_at_c<OptionType, 3>;
    using UseTime = boost::mp11::mp_at_c<OptionType, 4>;
    using StripCalibration = boost::mp11::mp_at_c<OptionType, 5>;
    using ZCutsOption = boost::mp11::mp_at_c<OptionType, 6>;

    const bool configIsBottomCandidate =
        config.candidateDirection == Direction::Backward();

    if (configIsBottomCandidate != IsBottomCandidate::value ||
        config.interactionPointCut != InteractionPointCut::value ||
        config.spacePointsSortedByRadius != SortedByR::value ||
        config.experimentCuts.connected() != ExperimentCuts::value ||
        config.useTime != UseTime::value ||
        config.stripCalibrationTolerance.has_value() !=
            StripCalibration::value ||
        configZCuts != ZCutsOption::value) {
      return;  // skip if the configuration does not match
    }

    // check if we already have an implementation for this configuration
    if (result != nullptr) {
      throw std::runtime_error(
          "DoubletSeedFinder: Multiple implementations found for one "
          "configuration");
    }

    // create the implementation for the given configuration
    result = std::make_unique<
        Impl<IsBottomCandidate::value, InteractionPointCut::value,
             SortedByR::value, ExperimentCuts::value, UseTime::value,
             StripCalibration::value, ZCutsOption::value>>(config);
  });
  if (result == nullptr) {
    throw std::runtime_error(
        "DoubletSeedFinder: No implementation found for the given "
        "configuration");
  }
  return result;
}

DoubletSeedFinder::DerivedConfig::DerivedConfig(const Config& config,
                                                float bFieldInZ_)
    : Config(config), bFieldInZ(bFieldInZ_) {
  // bFieldInZ is in (pT/radius) natively, no need for conversion
  const float pTPerHelixRadius = bFieldInZ;
  minHelixDiameter2 = square(minPt * 2 / pTPerHelixRadius) * helixCutTolerance;
  timeCutNSigma2 = square(timeCutNSigma);
}

MiddleSpInfo DoubletSeedFinder::computeMiddleSpInfo(
    const ConstSpacePointProxy& spM) {
  const float rM = spM.zr()[1];
  const float uIP = -1 / rM;
  const float cosPhiM = -spM.xy()[0] * uIP;
  const float sinPhiM = -spM.xy()[1] * uIP;
  const float uIP2 = uIP * uIP;

  return {uIP, uIP2, cosPhiM, sinPhiM};
}

}  // namespace Acts

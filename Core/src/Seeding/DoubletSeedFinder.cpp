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

#include <algorithm>
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

    // The cut values the cuts on z and r read.
    const float deltaRMin = m_cfg.deltaRMin;
    const float deltaRMax = m_cfg.deltaRMax;
    [[maybe_unused]] const float deltaZMin = m_cfg.deltaZMin;
    [[maybe_unused]] const float deltaZMax = m_cfg.deltaZMax;
    [[maybe_unused]] const float collisionRegionMin = m_cfg.collisionRegionMin;
    const float collisionRegionMax = m_cfg.collisionRegionMax;

    // The cuts that need only the candidate's z and r: deltaR, deltaZ and the
    // collision region.
    const auto passesZAndRCuts = [=](float zO, float rO) {
      const float deltaR = isBottomCandidate ? rM - rO : rO - rM;
      const float deltaZ = isBottomCandidate ? zM - zO : zO - zM;
      bool outside = false;
      if constexpr (!sortedByR) {
        outside = outsideRangeCheck(deltaR, deltaRMin, deltaRMax);
      }
      if constexpr (zCuts == ZCuts::eGeneral) {
        outside |= outsideRangeCheck(deltaZ, deltaZMin, deltaZMax);
      } else if constexpr (zCuts == ZCuts::eSymmetric) {
        // deltaZMin is -deltaZMax
        outside |= outsideSymmetricRange(deltaZ, deltaZMax);
      }
      // the longitudinal impact parameter zOrigin is defined as (zM - rM *
      // cotTheta) where cotTheta is the ratio Z/R (forward angle) of space
      // point duplet but instead we calculate (zOrigin * deltaR) and multiply
      // collisionRegion by deltaR to avoid divisions
      const float zOriginTimesDeltaR = zM * deltaR - rM * deltaZ;
      // check if duplet origin on z axis within collision region
      if constexpr (zCuts == ZCuts::eGeneral) {
        outside |=
            outsideRangeCheck(zOriginTimesDeltaR, collisionRegionMin * deltaR,
                              collisionRegionMax * deltaR);
      } else {
        // collisionRegionMin * deltaR is exactly -(collisionRegionMax *
        // deltaR), since negation is exact
        outside |= outsideSymmetricRange(zOriginTimesDeltaR,
                                         collisionRegionMax * deltaR);
      }
      return static_cast<std::uint32_t>(!outside);
    };

    const SpacePointContainer& container = candidateSps.container();
    const auto xy = container.xyColumn().data();
    const auto zr = container.zrColumn().data();
    const auto varianceZ = container.varianceZColumn().data();
    const auto varianceR = container.varianceRColumn().data();

    // The candidates as indices into the container: a range of them is
    // contiguous, a subset lists them.
    constexpr bool kContiguous =
        std::is_same_v<CandidateSps, SpacePointContainer::ConstRange>;
    const auto candidateIndex = [&](std::uint32_t k) -> SpacePointIndex {
      if constexpr (kContiguous) {
        return candidateSps.range().first + k;
      } else {
        return candidateSps.subset()[k];
      }
    };

    // Sorted by radius, the candidates inside deltaR end at the first one
    // outside it.
    std::uint32_t end = static_cast<std::uint32_t>(candidateSps.size());
    if constexpr (sortedByR) {
      assert(std::ranges::is_sorted(
          std::views::iota(std::uint32_t{0}, end), {},
          [&](std::uint32_t k) { return zr[candidateIndex(k)][1]; }));
      end = *std::ranges::partition_point(
          std::views::iota(std::uint32_t{0}, end), [&](std::uint32_t k) {
            const float rO = zr[candidateIndex(k)][1];
            return isBottomCandidate ? !(rM - rO < deltaRMin)
                                     : !(rO - rM > deltaRMax);
          });
    }

    // The middle space point as the cuts after the strip calibration see it:
    // calibrated against the doublet's chord if enabled, as it is otherwise.
    struct MiddleFrame {
      float x;
      float y;
      float r;
      float uIP;
      float vIPAbs;
    };
    const MiddleFrame middle{xM, yM, rM, middleSpInfo.uIP, vIPAbs};

    // The candidate in the u-v reference frame of the middle space point.
    struct UvFrame {
      float xNewFrame;
      float yNewFrame;
      float uT;
      float vT;
      float iDeltaR2;
    };
    const auto transform = [&](float xO, float yO, const MiddleFrame& mid) {
      const float deltaX = xO - mid.x;
      const float deltaY = yO - mid.y;

      const float xNewFrame =
          deltaX * middleSpInfo.cosPhiM + deltaY * middleSpInfo.sinPhiM;
      const float yNewFrame =
          deltaY * middleSpInfo.cosPhiM - deltaX * middleSpInfo.sinPhiM;

      const float deltaR2 = deltaX * deltaX + deltaY * deltaY;
      const float iDeltaR2 = 1 / deltaR2;

      return UvFrame{xNewFrame, yNewFrame, xNewFrame * iDeltaR2,
                     yNewFrame * iDeltaR2, iDeltaR2};
    };

    // The experiment cuts, and the doublet if it passes them, for a candidate
    // that passed all other cuts.
    const auto addDoublet = [&](SpacePointIndex indexO, float deltaZ,
                                const UvFrame& uv) {
      const float iDeltaR = std::sqrt(uv.iDeltaR2);
      const float cotTheta = deltaZ * iDeltaR;

      // discard doublets based on experiment specific cuts
      if constexpr (experimentCuts) {
        if (!m_cfg.experimentCuts(middleSp, container[indexO], cotTheta,
                                  isBottomCandidate)) {
          return;
        }
      }

      const float er = calculateError(varianceZ[indexO], varianceR[indexO],
                                      uv.iDeltaR2, cotTheta);

      // fill output vectors
      compatibleDoublets.emplace_back(indexO, cotTheta, iDeltaR, er, uv.uT,
                                      uv.vT, uv.xNewFrame, uv.yNewFrame);
    };

    // We check the interaction point by evaluating the minimal distance
    // between the origin and the straight line connecting the two points in
    // the doublets. Using a geometric similarity, the Im is given by
    // yNewFrame * rM / deltaR > config.impactMax
    // However, we make here an approximation of the impact parameter
    // which is valid under the assumption yNewFrame / xNewFrame is small
    // The correct computation would be:
    // yNewFrame * yNewFrame * rM * rM > config.impactMax *
    // config.impactMax * deltaR2
    [[maybe_unused]] const auto ipTest = [&](const UvFrame& uv,
                                             const MiddleFrame& mid) {
      return std::abs(mid.r * uv.yNewFrame) > impactMax * uv.xNewFrame;
    };

    // The curvature cut, for a candidate in the u-v reference frame.
    [[maybe_unused]] const auto failsCurvatureCut =
        [&](const UvFrame& uv, const MiddleFrame& mid) {
          // in the rotated frame the interaction point is positioned at x = -rM
          // and y ~= impactParam
          const float vIP = (uv.yNewFrame > 0) ? -mid.vIPAbs : mid.vIPAbs;

          // we can obtain aCoef as the slope dv/du of the linear function,
          // estimated using du and dv between the two SP bCoef is obtained by
          // inserting aCoef into the linear equation
          const float aCoef = (uv.vT - vIP) / (uv.uT - mid.uIP);
          const float bCoef = vIP - aCoef * mid.uIP;
          // the distance of the straight line from the origin (radius of the
          // circle) is related to aCoef and bCoef by d^2 = bCoef^2 / (1 +
          // aCoef^2) = 1 / (radius^2) and we can apply the cut on the curvature
          return (bCoef * bCoef) * m_cfg.minHelixDiameter2 > 1 + aCoef * aCoef;
        };

    // The cuts after those on z, r and time, and the doublet, for one
    // candidate.
    const auto complete = [&](SpacePointIndex indexO, float xO, float yO,
                              float deltaR, float deltaZ,
                              const MiddleFrame& mid) {
      // check if duplet cotTheta is within the region of interest
      // cotTheta is defined as (deltaZ / deltaR) but instead we multiply
      // cotThetaMax by deltaR to avoid division
      if (outsideSymmetricRange(deltaZ, m_cfg.cotThetaMax * deltaR)) {
        return;
      }
      const UvFrame uv = transform(xO, yO, mid);
      if constexpr (interactionPointCut) {
        if (ipTest(uv, mid) && failsCurvatureCut(uv, mid)) {
          return;
        }
      }
      addDoublet(indexO, deltaZ, uv);
    };

    // The strip calibration of both space points against their chord, then
    // the remaining cuts and the doublet with the calibrated positions.
    [[maybe_unused]] const auto calibrateAndComplete =
        [&](SpacePointIndex indexO) {
          const std::array<float, 3> chord{
              xy[indexO][0] - xM, xy[indexO][1] - yM, zr[indexO][0] - zM};
          std::array<float, 3> calibratedM{};
          std::array<float, 3> calibratedO{};
          if (!detail::calibrateOuterStripSpacePoint(
                  chord, calibrationM, calibratedM, calibrationTolerance) ||
              !detail::calibrateOuterStripSpacePoint(
                  chord,
                  detail::deriveOuterStripSpacePointCalibrationDetails(
                      container[indexO].outerStripCalibrationDetails()),
                  calibratedO, calibrationTolerance)) {
            return;
          }

          const float rMd = fastHypot(calibratedM[0], calibratedM[1]);
          const float uIP = -1 / rMd;
          const MiddleFrame calibrated{calibratedM[0], calibratedM[1], rMd, uIP,
                                       impactMax * uIP * uIP};
          const float rOd = fastHypot(calibratedO[0], calibratedO[1]);
          complete(indexO, calibratedO[0], calibratedO[1],
                   isBottomCandidate ? rMd - rOd : rOd - rMd,
                   isBottomCandidate ? calibratedM[2] - calibratedO[2]
                                     : calibratedO[2] - calibratedM[2],
                   calibrated);
        };

    // The candidates go through in blocks: the cuts on z and r are made for a
    // whole block without a branch, and only the candidates that pass them go
    // on to the remaining cuts.
    constexpr std::size_t kBlockSize = 256;
    // The candidates per step of the cuts on z and r: a subset's candidates
    // are scattered, so they go one at a time
    constexpr std::size_t kChunkSize = kContiguous ? 8 : 1;

    const auto passesZAndRCutsAt = [&](std::size_t k) {
      if constexpr (kContiguous) {
        // by the offset from the range's first candidate, which cannot wrap
        // around
        const std::array<float, 2>& zrO = zr[candidateSps.range().first + k];
        return passesZAndRCuts(zrO[0], zrO[1]);
      } else {
        const std::array<float, 2>& zrO =
            zr[candidateIndex(static_cast<std::uint32_t>(k))];
        return passesZAndRCuts(zrO[0], zrO[1]);
      }
    };

    // The time cut, for the candidates that pass the cuts on z and r: the time
    // difference is corrected for the time of flight along the straight line
    // between the two space points, assuming an outgoing particle at the
    // speed of light (which is 1 in ACTS units)
    [[maybe_unused]] std::array<std::uint32_t, kBlockSize> inTime{};
    const auto keepInTime = [&](std::span<const std::uint32_t> passing) {
      if constexpr (!useTime) {
        return passing;
      } else {
        std::size_t nInTime = 0;
        for (const std::uint32_t k : passing) {
          const SpacePointIndex indexO = candidateIndex(k);
          const ConstSpacePointProxy otherSp = container[indexO];
          const float distance = fastHypot(
              xy[indexO][0] - xM, xy[indexO][1] - yM, zr[indexO][0] - zM);
          const float dt = isBottomCandidate ? tM - otherSp.time() - distance
                                             : otherSp.time() - tM - distance;
          const float varianceDt = varianceTM + otherSp.varianceT();
          // written every time; the position only moves on if in time
          inTime[nInTime] = k;
          nInTime += static_cast<std::size_t>(
              !(dt * dt > m_cfg.timeCutNSigma2 * varianceDt));
        }
        return std::span<const std::uint32_t>(inTime).first(nInTime);
      }
    };

    // Without the interaction point cut, or with the strip calibration, the
    // survivors are completed one at a time.
    if constexpr (!interactionPointCut || stripCalibration) {
      detail::forEachPassingBlock<kBlockSize, kChunkSize>(
          0, end, passesZAndRCutsAt,
          [&](std::span<const std::uint32_t> passingZAndR) {
            for (const std::uint32_t k : keepInTime(passingZAndR)) {
              const SpacePointIndex indexO = candidateIndex(k);
              if constexpr (stripCalibration) {
                calibrateAndComplete(indexO);
              } else {
                complete(
                    indexO, xy[indexO][0], xy[indexO][1],
                    isBottomCandidate ? rM - zr[indexO][1] : zr[indexO][1] - rM,
                    isBottomCandidate ? zM - zr[indexO][0] : zr[indexO][0] - zM,
                    middle);
              }
            }
          });
      return;
    }

    // With the interaction point cut, the remaining cuts are made for the
    // survivors of a block in the same way: what they compute for each
    // survivor is kept for its completion. Not cleared, as it is large and
    // each entry is written before it is read.
    struct Survivor {
      SpacePointIndex index;
      float deltaZ;
      UvFrame uv;
    };
    // NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
    std::array<Survivor, kBlockSize> survivors;
    std::array<std::uint32_t, kBlockSize> keep;
    std::array<std::uint32_t, kBlockSize> needsCurvatureCut;
    std::array<std::uint32_t, kBlockSize> kept;
    // NOLINTEND(cppcoreguidelines-pro-type-member-init)

    detail::forEachPassingBlock<kBlockSize, kChunkSize>(
        0, end, passesZAndRCutsAt,
        [&](std::span<const std::uint32_t> passingZAndR) {
          const std::span<const std::uint32_t> passing =
              keepInTime(passingZAndR);

          // the transformation and the cuts on the impact parameter and
          // cotTheta; the curvature cut is left for the survivors it can still
          // reject
          std::uint32_t nCurvature = 0;
          for (std::uint32_t k = 0; k < passing.size(); ++k) {
            const SpacePointIndex indexO = candidateIndex(passing[k]);
            const float deltaR =
                isBottomCandidate ? rM - zr[indexO][1] : zr[indexO][1] - rM;
            const float deltaZ =
                isBottomCandidate ? zM - zr[indexO][0] : zr[indexO][0] - zM;
            const UvFrame uv = transform(xy[indexO][0], xy[indexO][1], middle);
            const bool ipTestHolds = ipTest(uv, middle);

            // check if duplet cotTheta is within the region of interest
            // cotTheta is defined as (deltaZ / deltaR) but instead we multiply
            // cotThetaMax by deltaR to avoid division
            const bool cutCotTheta =
                outsideSymmetricRange(deltaZ, m_cfg.cotThetaMax * deltaR);

            survivors[k] = {indexO, deltaZ, uv};
            // kept unless cut on cotTheta; where ipTest holds, the curvature
            // cut below decides
            keep[k] = static_cast<std::uint32_t>(!cutCotTheta && !ipTestHolds);
            needsCurvatureCut[nCurvature] = k;
            nCurvature +=
                static_cast<std::uint32_t>(ipTestHolds && !cutCotTheta);
          }

          for (const std::uint32_t k :
               std::span(needsCurvatureCut).first(nCurvature)) {
            keep[k] = static_cast<std::uint32_t>(
                !failsCurvatureCut(survivors[k].uv, middle));
          }

          for (const std::uint32_t k : detail::collectFlagged(
                   std::span<const std::uint32_t>(keep).first(passing.size()),
                   [](std::size_t k) { return static_cast<std::uint32_t>(k); },
                   std::span(kept))) {
            addDoublet(survivors[k].index, survivors[k].deltaZ,
                       survivors[k].uv);
          }
        });
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

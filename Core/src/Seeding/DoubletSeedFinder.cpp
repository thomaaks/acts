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
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>

#include <boost/mp11.hpp>
#include <boost/mp11/algorithm.hpp>

#include "detail/BlockPass.hpp"
#include "detail/SortByKey.hpp"

namespace Acts {

void DoubletsForMiddleSp::sortByCotTheta(
    const IndexRange& range,
    std::vector<IndexAndCotTheta>& indexAndCotTheta) const {
  SortStorage storage;
  sortByCotTheta(range, indexAndCotTheta, storage);
}

void DoubletsForMiddleSp::sortByCotTheta(
    const IndexRange& range, std::vector<IndexAndCotTheta>& indexAndCotTheta,
    SortStorage& storage) const {
  indexAndCotTheta.clear();
  indexAndCotTheta.reserve(range.second - range.first);
  for (Index i = range.first; i < range.second; ++i) {
    indexAndCotTheta.emplace_back(i, m_cotTheta[i]);
  }

  const auto cotThetaOf = [](const IndexAndCotTheta& item) {
    return item.cotTheta;
  };

  // The limit up to which a quadratic sort is used: the rank sort for the whole
  // list instead of the linear bucket sort, and the insertion pass for one
  // bucket instead of std::ranges::stable_sort
  constexpr std::size_t kQuadraticLinearCrossover = 32;
  detail::sortByKey<kQuadraticLinearCrossover>(std::span(indexAndCotTheta),
                                               storage, cotThetaOf);
}

namespace {

template <bool isBottomCandidate, bool interactionPointCut, bool sortedByR,
          bool experimentCuts, bool useTime, bool stripCalibration>
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

    // A candidate's index in the container, by its position in candidateSps
    const auto candidateIndex = [&](std::size_t k) {
      return candidateSps[static_cast<SpacePointIndex>(k)].index();
    };
    // A candidate's z and r by its position; a range's by the offset from its
    // first candidate, without which the first pass is not vectorised
    const auto zrAt = [&](std::size_t k) -> const std::array<float, 2>& {
      if constexpr (std::is_same_v<CandidateSps,
                                   SpacePointContainer::ConstRange>) {
        return zr[candidateSps.range().first + k];
      } else {
        return zr[candidateIndex(k)];
      }
    };

    // Sorted by radius, the candidates inside deltaR end at the first one
    // outside it.
    std::size_t end = candidateSps.size();
    if constexpr (sortedByR) {
      const auto rOf = [&](std::size_t k) { return zrAt(k)[1]; };
      assert(std::ranges::is_sorted(std::views::iota(std::size_t{0}, end), {},
                                    rOf));
      end = *std::ranges::partition_point(
          std::views::iota(std::size_t{0}, end), [&](std::size_t k) {
            return isBottomCandidate ? !(rM - rOf(k) < m_cfg.deltaRMin)
                                     : !(rOf(k) - rM > m_cfg.deltaRMax);
          });
    }

    // ---- The cuts ----------------------------------------------------------

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

    // The cuts that need only the candidate's z and r: deltaR, deltaZ and the
    // collision region. The config values they read are copied, so that they
    // are loaded once rather than for every candidate.
    const float deltaRMin = m_cfg.deltaRMin;
    const float deltaRMax = m_cfg.deltaRMax;
    const float deltaZMin = m_cfg.deltaZMin;
    const float deltaZMax = m_cfg.deltaZMax;
    const float collisionRegionMin = m_cfg.collisionRegionMin;
    const float collisionRegionMax = m_cfg.collisionRegionMax;
    const auto passesZAndRCuts = [=](float deltaR, float deltaZ) {
      bool outside = false;
      if constexpr (!sortedByR) {
        outside = outsideRangeCheck(deltaR, deltaRMin, deltaRMax);
      }

      outside |= outsideRangeCheck(deltaZ, deltaZMin, deltaZMax);

      // the longitudinal impact parameter zOrigin is defined as (zM - rM *
      // cotTheta) where cotTheta is the ratio Z/R (forward angle) of space
      // point duplet but instead we calculate (zOrigin * deltaR) and multiply
      // collisionRegion by deltaR to avoid divisions
      const float zOriginTimesDeltaR = zM * deltaR - rM * deltaZ;
      // check if duplet origin on z axis within collision region
      outside |=
          outsideRangeCheck(zOriginTimesDeltaR, collisionRegionMin * deltaR,
                            collisionRegionMax * deltaR);

      return !outside;
    };

    // check the time compatibility of the two space points. the time
    // difference is corrected for the time of flight along the straight line
    // between them, assuming an outgoing particle at the speed of light
    // (which is 1 in ACTS units)
    const auto inTime = [&](SpacePointIndex indexO) {
      const ConstSpacePointProxy otherSp = container[indexO];
      const float distance =
          fastHypot(xy[indexO][0] - xM, xy[indexO][1] - yM, zr[indexO][0] - zM);
      float dt = 0;
      if constexpr (isBottomCandidate) {
        dt = tM - otherSp.time() - distance;
      } else {
        dt = otherSp.time() - tM - distance;
      }
      const float varianceDt = varianceTM + otherSp.varianceT();
      return !(dt * dt > m_cfg.timeCutNSigma2 * varianceDt);
    };

    // calibrate both space points against their chord; the cuts after it and
    // the doublet use the calibrated positions
    struct Calibrated {
      MiddleFrame middle;
      std::array<float, 2> xyO;
      float deltaR;
      float deltaZ;
    };
    [[maybe_unused]] const auto calibrate =
        [&](SpacePointIndex indexO) -> std::optional<Calibrated> {
      const std::array<float, 3> chord{xy[indexO][0] - xM, xy[indexO][1] - yM,
                                       zr[indexO][0] - zM};
      std::array<float, 3> calibratedM{};
      std::array<float, 3> calibratedO{};
      if (!detail::calibrateOuterStripSpacePoint(
              chord, calibrationM, calibratedM, calibrationTolerance) ||
          !detail::calibrateOuterStripSpacePoint(
              chord,
              detail::deriveOuterStripSpacePointCalibrationDetails(
                  container[indexO].outerStripCalibrationDetails()),
              calibratedO, calibrationTolerance)) {
        return std::nullopt;
      }

      const float rMd = fastHypot(calibratedM[0], calibratedM[1]);
      const float uIP = -1 / rMd;
      const float rOd = fastHypot(calibratedO[0], calibratedO[1]);
      return Calibrated{
          {calibratedM[0], calibratedM[1], rMd, uIP, impactMax * uIP * uIP},
          {calibratedO[0], calibratedO[1]},
          isBottomCandidate ? rMd - rOd : rOd - rMd,
          isBottomCandidate ? calibratedM[2] - calibratedO[2]
                            : calibratedO[2] - calibratedM[2]};
    };

    // check if duplet cotTheta is within the region of interest
    // cotTheta is defined as (deltaZ / deltaR) but instead we multiply
    // cotThetaMax by deltaR to avoid division
    const auto passesCotThetaCut = [&](float deltaR, float deltaZ) {
      return !outsideRangeCheck(deltaZ, -m_cfg.cotThetaMax * deltaR,
                                m_cfg.cotThetaMax * deltaR);
    };

    // transform SP coordinates to the u-v reference frame
    const auto transform = [&](const std::array<float, 2>& xyO,
                               const MiddleFrame& mid) {
      const float deltaX = xyO[0] - mid.x;
      const float deltaY = xyO[1] - mid.y;

      const float xNewFrame =
          deltaX * middleSpInfo.cosPhiM + deltaY * middleSpInfo.sinPhiM;
      const float yNewFrame =
          deltaY * middleSpInfo.cosPhiM - deltaX * middleSpInfo.sinPhiM;

      const float deltaR2 = deltaX * deltaX + deltaY * deltaY;
      const float iDeltaR2 = 1 / deltaR2;

      return UvFrame{xNewFrame, yNewFrame, xNewFrame * iDeltaR2,
                     yNewFrame * iDeltaR2, iDeltaR2};
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
    const auto ipTest = [&](const UvFrame& uv, const MiddleFrame& mid) {
      return std::abs(mid.r * uv.yNewFrame) > impactMax * uv.xNewFrame;
    };

    // the curvature cut, for a doublet whose impact parameter is too large
    const auto passesCurvatureCut = [&](const UvFrame& uv,
                                        const MiddleFrame& mid) {
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
      return !((bCoef * bCoef) * m_cfg.minHelixDiameter2 > 1 + aCoef * aCoef);
    };

    // discard doublets based on experiment specific cuts
    const auto passesExperimentCut = [&](SpacePointIndex indexO, float deltaZ,
                                         const UvFrame& uv) {
      const float cotTheta = deltaZ * std::sqrt(uv.iDeltaR2);
      return m_cfg.experimentCuts(middleSp, container[indexO], cotTheta,
                                  isBottomCandidate);
    };

    // fill output vectors
    const auto addDoublet = [&](SpacePointIndex indexO, float deltaZ,
                                const UvFrame& uv) {
      const float iDeltaR = std::sqrt(uv.iDeltaR2);
      const float cotTheta = deltaZ * iDeltaR;
      const float er = calculateError(varianceZ[indexO], varianceR[indexO],
                                      uv.iDeltaR2, cotTheta);
      compatibleDoublets.emplace_back(indexO, cotTheta, iDeltaR, er, uv.uT,
                                      uv.vT, uv.xNewFrame, uv.yNewFrame);
    };

    // ---- The block pass ----------------------------------------------------

    // The candidates go through in blocks, and each cut is a pass over the
    // candidates of a block that are left (see detail/BlockPass.hpp).
    constexpr std::size_t kBlockSize = 256;
    // The candidates per step of the cuts on z and r
    constexpr std::size_t kChunkSize = 8;

    // What the cuts compute for each candidate of a block, for the cuts after
    // them. Not cleared: each entry is written before it is read. The
    // calibrated positions are only kept with the strip calibration.
    constexpr std::size_t kCalibrated = stripCalibration ? kBlockSize : 0;
    // NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
    detail::PerCandidate<float, kBlockSize> deltaR;
    detail::PerCandidate<float, kBlockSize> deltaZ;
    detail::PerCandidate<UvFrame, kBlockSize> uv;
    [[maybe_unused]] std::array<MiddleFrame, kCalibrated> calibratedMiddle;
    [[maybe_unused]] std::array<std::array<float, 2>, kCalibrated> calibratedXY;
    // NOLINTEND(cppcoreguidelines-pro-type-member-init)

    // The middle space point and the candidate's x and y as the cuts after
    // the strip calibration see them
    const auto middleOf = [&](std::size_t i) -> const MiddleFrame& {
      if constexpr (stripCalibration) {
        return calibratedMiddle[i];
      } else {
        return middle;
      }
    };
    const auto xyOf = [&](std::size_t i,
                          std::size_t k) -> const std::array<float, 2>& {
      if constexpr (stripCalibration) {
        return calibratedXY[i];
      } else {
        return xy[candidateIndex(k)];
      }
    };

    // Each pass is called with a candidate's index in its block, i, and its
    // position, k.
    detail::runBlockPass<kBlockSize, kChunkSize>(
        0, end,
        detail::passes(
            [&](std::size_t i, std::size_t k) {
              const std::array<float, 2>& zrO = zrAt(k);
              deltaR[i] = isBottomCandidate ? rM - zrO[1] : zrO[1] - rM;
              deltaZ[i] = isBottomCandidate ? zM - zrO[0] : zrO[0] - zM;
              return passesZAndRCuts(deltaR[i], deltaZ[i]);
            },
            detail::passIf<useTime>([&](std::size_t /*i*/, std::size_t k) {
              return inTime(candidateIndex(k));
            }),
            detail::passIf<stripCalibration>([&](std::size_t i, std::size_t k) {
              if constexpr (stripCalibration) {
                const std::optional<Calibrated> calibrated =
                    calibrate(candidateIndex(k));
                if (!calibrated) {
                  return false;
                }
                calibratedMiddle[i] = calibrated->middle;
                calibratedXY[i] = calibrated->xyO;
                deltaR[i] = calibrated->deltaR;
                deltaZ[i] = calibrated->deltaZ;
              }
              return true;
            }),
            [&](std::size_t i, std::size_t /*k*/) {
              return passesCotThetaCut(deltaR[i], deltaZ[i]);
            },
            [&](std::size_t i, std::size_t k) {
              uv[i] = transform(xyOf(i, k), middleOf(i));
            },
            detail::passIf<interactionPointCut>(detail::onlyWhere(
                [&](std::size_t i, std::size_t /*k*/) {
                  return ipTest(uv[i], middleOf(i));
                },
                [&](std::size_t i, std::size_t /*k*/) {
                  return passesCurvatureCut(uv[i], middleOf(i));
                })),
            detail::passIf<experimentCuts>([&](std::size_t i, std::size_t k) {
              return passesExperimentCut(candidateIndex(k), deltaZ[i], uv[i]);
            })),
        [&](std::size_t i, std::size_t k) {
          addDoublet(candidateIndex(k), deltaZ[i], uv[i]);
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

  using DoubletOptions =
      boost::mp11::mp_product<boost::mp11::mp_list, IsBottomCandidateOptions,
                              InteractionPointCutOptions, SortedByROptions,
                              ExperimentCutsOptions, UseTimeOptions,
                              StripCalibrationOptions>;

  std::unique_ptr<DoubletSeedFinder> result;
  boost::mp11::mp_for_each<DoubletOptions>([&](auto option) {
    using OptionType = decltype(option);

    using IsBottomCandidate = boost::mp11::mp_at_c<OptionType, 0>;
    using InteractionPointCut = boost::mp11::mp_at_c<OptionType, 1>;
    using SortedByR = boost::mp11::mp_at_c<OptionType, 2>;
    using ExperimentCuts = boost::mp11::mp_at_c<OptionType, 3>;
    using UseTime = boost::mp11::mp_at_c<OptionType, 4>;
    using StripCalibration = boost::mp11::mp_at_c<OptionType, 5>;

    const bool configIsBottomCandidate =
        config.candidateDirection == Direction::Backward();

    if (configIsBottomCandidate != IsBottomCandidate::value ||
        config.interactionPointCut != InteractionPointCut::value ||
        config.spacePointsSortedByRadius != SortedByR::value ||
        config.experimentCuts.connected() != ExperimentCuts::value ||
        config.useTime != UseTime::value ||
        config.stripCalibrationTolerance.has_value() !=
            StripCalibration::value) {
      return;  // skip if the configuration does not match
    }

    // check if we already have an implementation for this configuration
    if (result != nullptr) {
      throw std::runtime_error(
          "DoubletSeedFinder: Multiple implementations found for one "
          "configuration");
    }

    // create the implementation for the given configuration
    result = std::make_unique<Impl<
        IsBottomCandidate::value, InteractionPointCut::value, SortedByR::value,
        ExperimentCuts::value, UseTime::value, StripCalibration::value>>(
        config);
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

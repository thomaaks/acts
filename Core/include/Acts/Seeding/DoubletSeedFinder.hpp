// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Direction.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/SpacePointContainer.hpp"
#include "Acts/Utilities/Delegate.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace Acts {

/// One doublet of a middle space point: the index of its other space point and
/// the parameters the triplet finder reads.
class DoubletForMiddleSp {
 public:
  /// Constructor from the doublet's parameters
  /// @param sp Space point index for the doublet
  /// @param cotTheta Cotangent of polar angle
  /// @param iDeltaR Inverse delta R parameter
  /// @param er Error in R coordinate
  /// @param u U coordinate parameter
  /// @param v V coordinate parameter
  /// @param x X coordinate
  /// @param y Y coordinate
  DoubletForMiddleSp(SpacePointIndex sp, float cotTheta, float iDeltaR,
                     float er, float u, float v, float x, float y)
      : m_spacePoint(sp),
        m_cotTheta(cotTheta),
        m_iDeltaR(iDeltaR),
        m_er(er),
        m_u(u),
        m_v(v),
        m_x(x),
        m_y(y) {}

  /// Get space point index pair
  /// @return The space point index
  SpacePointIndex spacePointIndex() const { return m_spacePoint; }

  /// Get cotangent of theta
  /// @return The cotTheta value
  float cotTheta() const { return m_cotTheta; }
  /// Get er value
  /// @return The er value
  float er() const { return m_er; }
  /// Get inverse delta r
  /// @return The inverse delta r value
  float iDeltaR() const { return m_iDeltaR; }
  /// Get u coordinate
  /// @return The u value
  float u() const { return m_u; }
  /// Get v coordinate
  /// @return The v value
  float v() const { return m_v; }
  /// Get x coordinate
  /// @return The x value
  float x() const { return m_x; }
  /// Get y coordinate
  /// @return The y value
  float y() const { return m_y; }

 private:
  SpacePointIndex m_spacePoint{};
  // parameters required to calculate a circle with linear equation
  float m_cotTheta{};
  float m_iDeltaR{};
  float m_er{};
  float m_u{};
  float m_v{};
  float m_x{};
  float m_y{};
};

/// Container for doublets found by the doublet seed finder, one record per
/// doublet.
class DoubletsForMiddleSp {
 public:
  /// Type alias for index type used in doublets container
  using Index = std::uint32_t;
  /// Type alias for range of indices in doublets container
  using IndexRange = std::pair<Index, Index>;
  /// Type alias for subset of indices in doublets container
  using IndexSubset = std::span<const Index>;

  /// Check if the doublets container is empty
  /// @return True if container has no doublets
  [[nodiscard]] bool empty() const { return m_doublets.empty(); }
  /// Get the number of doublets in container
  /// @return Number of doublets stored
  [[nodiscard]] Index size() const {
    return static_cast<Index>(m_doublets.size());
  }

  /// Clear all stored doublets and associated data
  void clear() { m_doublets.clear(); }

  /// Add a new doublet with associated parameters
  /// @param sp space point index for the doublet
  /// @param cotTheta Cotangent of polar angle
  /// @param iDeltaR Inverse delta R parameter
  /// @param er Error in R coordinate
  /// @param u U coordinate parameter
  /// @param v V coordinate parameter
  /// @param x X coordinate
  /// @param y Y coordinate
  void emplace_back(SpacePointIndex sp, float cotTheta, float iDeltaR, float er,
                    float u, float v, float x, float y) {
    m_doublets.emplace_back(sp, cotTheta, iDeltaR, er, u, v, x, y);
  }

  /// Pair of doublet index and cotTheta value.
  struct IndexAndCotTheta {
    /// Doublet index
    Index index{};
    /// Cotangent of theta
    float cotTheta{};
  };

  /// Type alias for subset of index and cotTheta pairs
  using IndexAndCotThetaSubset = std::span<const IndexAndCotTheta>;

  /// Storage for sortByCotTheta, kept by the caller so that sorting does not
  /// allocate
  struct SortStorage {
    /// Doublets being sorted
    std::vector<IndexAndCotTheta> scratch;
    /// Bucket counts and indices
    std::vector<std::uint32_t> buckets;
  };

  /// Sort doublets by cotTheta within given range. Doublets with equal
  /// cotTheta keep their index order.
  /// @param range Index range to sort within
  /// @param indexAndCotTheta Output vector containing sorted index and cotTheta pairs
  void sortByCotTheta(const IndexRange& range,
                      std::vector<IndexAndCotTheta>& indexAndCotTheta) const;

  /// Sort doublets by cotTheta within given range, with storage the caller
  /// keeps between calls. Doublets with equal cotTheta keep their index order.
  /// @param range Index range to sort within
  /// @param indexAndCotTheta Output vector containing sorted index and cotTheta pairs
  /// @param storage Storage for the sort, of any content
  void sortByCotTheta(const IndexRange& range,
                      std::vector<IndexAndCotTheta>& indexAndCotTheta,
                      SortStorage& storage) const;

  /// Access doublet by index
  /// @param index Index of the doublet to access
  /// @return The doublet
  const DoubletForMiddleSp& operator[](Index index) const {
    return m_doublets[index];
  }
  /// Access doublet by index and cotTheta pair
  /// @param indexAndCotTheta Index and cotTheta pair for the doublet
  /// @return The doublet
  const DoubletForMiddleSp& operator[](
      IndexAndCotTheta indexAndCotTheta) const {
    return m_doublets[indexAndCotTheta.index];
  }

  /// Type alias for const iterator over doublets in container
  using const_iterator = std::vector<DoubletForMiddleSp>::const_iterator;

  /// Get iterator to beginning of doublets container
  /// @return Const iterator to first doublet
  const_iterator begin() const { return m_doublets.begin(); }
  /// Get iterator to end of doublets container
  /// @return Const iterator past the last doublet
  const_iterator end() const { return m_doublets.end(); }

  /// Range view over doublets in the container.
  using Range = std::span<const DoubletForMiddleSp>;

  /// Get range view of all doublets
  /// @return Range object covering all doublets
  Range range() const noexcept { return m_doublets; }
  /// Get range view of doublets within specified index range
  /// @param range Index range to create view for
  /// @return Range object covering specified doublets
  Range range(const IndexRange& range) const noexcept {
    return Range(m_doublets).subspan(range.first, range.second - range.first);
  }

  /// View of doublets in the order of a list of their indices, read by
  /// reference.
  /// @tparam Order Index, or index and cotTheta pair
  template <typename Order>
  class InOrder : public std::ranges::view_interface<InOrder<Order>> {
   public:
    /// @param doublets The doublets
    /// @param order Their indices, in the order to visit them
    InOrder(const DoubletsForMiddleSp& doublets, std::span<const Order> order)
        : InOrder(doublets.m_doublets.data(), order) {}

    /// @return Iterator to the first doublet
    auto begin() const { return m_view.begin(); }
    /// @return Iterator past the last doublet
    auto end() const { return m_view.end(); }
    /// @param offset Number of doublets to skip
    /// @return The doublets from offset on
    InOrder subspan(std::size_t offset) const {
      return InOrder(m_doublets, m_view.base().subspan(offset));
    }

   private:
    struct At {
      const DoubletForMiddleSp* doublets{};
      const DoubletForMiddleSp& operator()(Index index) const {
        return doublets[index];
      }
      const DoubletForMiddleSp& operator()(const IndexAndCotTheta& item) const {
        return doublets[item.index];
      }
    };

    InOrder(const DoubletForMiddleSp* doublets, std::span<const Order> order)
        : m_doublets(doublets), m_view(order, At{doublets}) {}

    const DoubletForMiddleSp* m_doublets{};
    std::ranges::transform_view<std::span<const Order>, At> m_view;
  };

  /// Subset view of doublets addressed by indices.
  using Subset = InOrder<Index>;
  /// Subset view of doublets addressed by index and cotTheta pairs.
  using Subset2 = InOrder<IndexAndCotTheta>;

  /// Create subset view from index subset
  /// @param subset Span of indices to include in subset
  /// @return Subset object for the specified indices
  Subset subset(const IndexSubset& subset) const noexcept {
    return Subset(*this, subset);
  }
  /// Create subset view from index and cotTheta subset
  /// @param subset Span of index and cotTheta pairs to include
  /// @return Subset2 object for the specified pairs
  Subset2 subset(const IndexAndCotThetaSubset& subset) const noexcept {
    return Subset2(*this, subset);
  }

 private:
  std::vector<DoubletForMiddleSp> m_doublets;
};

/// Derived quantities for the middle space point in a doublet.
struct MiddleSpInfo {
  /// minus one over radius of middle SP
  float uIP{};
  /// square of uIP
  float uIP2{};
  /// ratio between middle SP x position and radius
  float cosPhiM{};
  /// ratio between middle SP y position and radius
  float sinPhiM{};
};

/// Interface and a collection of standard implementations for a doublet seed
/// finder. Given a starting space point and a collection of candidates, it
/// finds all doublets that satisfy the selection criteria. For the standard
/// implementations the criteria are given by interaction point cuts.
///
/// @note The standard implementations rely on virtual function dispatch which
/// did not turn out to affect the performance after measurement.
class DoubletSeedFinder {
 public:
  /// Collection of configuration parameters for the doublet seed finder. This
  /// includes doublet cuts, steering switches, and assumptions about the space
  /// points.
  struct Config {
    /// Whether the input space points are sorted by radius
    bool spacePointsSortedByRadius = false;

    /// Direction of the doublet candidate space points. Either forward, also
    /// called top doublet, or backward, also called bottom doublet.
    Direction candidateDirection = Direction::Forward();

    /// Minimum radial distance between two doublet components
    float deltaRMin = 5 * UnitConstants::mm;
    /// Maximum radial distance between two doublet components
    float deltaRMax = 270 * UnitConstants::mm;

    /// Minimal z distance between two doublet components
    float deltaZMin = -std::numeric_limits<float>::infinity();
    /// Maximum z distance between two doublet components
    float deltaZMax = std::numeric_limits<float>::infinity();

    /// Maximum value of impact parameter estimation of the seed candidates
    float impactMax = 20 * UnitConstants::mm;

    /// Enable cut on the compatibility between interaction point and doublet,
    /// this is an useful approximation to speed up the seeding
    bool interactionPointCut = false;
    /// Enable the cut on the time compatibility
    /// The space point container has to provide the
    /// `SpacePointColumns::Time` and `SpacePointColumns::VarianceT` columns
    bool useTime = false;

    /// Limiting location of collision region in z-axis used to check if doublet
    /// origin is within reasonable bounds
    float collisionRegionMin = -150 * UnitConstants::mm;
    /// Maximum collision region boundary in z-axis for doublet origin checks
    float collisionRegionMax = +150 * UnitConstants::mm;

    /// Maximum allowed cotTheta between two space-points in doublet, used to
    /// check if forward angle is within bounds
    float cotThetaMax = 10.01788;  // equivalent to eta = 3 (pseudorapidity)

    /// Minimum transverse momentum (pT) used to check the r-z slope
    /// compatibility of triplets with maximum multiple scattering effect
    /// (produced by the minimum allowed pT particle) + a certain uncertainty
    /// term. Check the documentation for more information
    /// https://acts.readthedocs.io/en/latest/core/reconstruction/pattern_recognition/seeding.html
    float minPt = 400 * UnitConstants::MeV;
    /// Parameter which can loosen the tolerance of the track seed to form a
    /// helix. This is useful for e.g. misaligned seeding.
    float helixCutTolerance = 1;
    /// Maximum allowed difference of the times of the two space points,
    /// corrected for the time of flight between them, in units of their
    /// combined time resolution. Only used when `useTime` is enabled.
    float timeCutNSigma = 5;

    /// If set, calibrate both strip space points against their chord after the
    /// z origin cut, with this tolerance in strip half lengths. The following
    /// cuts and the doublet use the calibrated positions.
    /// Requires `SpacePointColumns::StripCalibrationDetails`.
    std::optional<float> stripCalibrationTolerance;

    /// Type alias for delegate to apply experiment specific cuts during doublet
    /// finding
    using ExperimentCuts =
        Delegate<bool(const ConstSpacePointProxy& /*middle*/,
                      const ConstSpacePointProxy& /*other*/, float /*cotTheta*/,
                      bool /*isBottomCandidate*/)>;

    /// Delegate to apply experiment specific cuts during doublet finding
    ExperimentCuts experimentCuts;
  };

  /// Derived configuration for the doublet seed finder using a magnetic field.
  struct DerivedConfig : public Config {
    /// Constructor for derived configuration with magnetic field
    /// @param config Base configuration to derive from
    /// @param bFieldInZ Magnetic field strength in z-direction
    DerivedConfig(const Config& config, float bFieldInZ);

    /// Magnetic field strength in z-direction for helix calculation
    float bFieldInZ = std::numeric_limits<float>::quiet_NaN();
    /// Squared minimum helix diameter derived from magnetic field and minimum
    /// pT
    float minHelixDiameter2 = std::numeric_limits<float>::quiet_NaN();
    /// Squared time cut in units of sigma, allowing the cut to be applied on
    /// the combined time variance
    float timeCutNSigma2 = std::numeric_limits<float>::quiet_NaN();
  };

  /// Computes additional quantities from the middle space point which can be
  /// reused during doublet finding.
  /// @param spM Middle space point for doublet computation
  /// @return MiddleSpInfo structure with computed quantities
  static MiddleSpInfo computeMiddleSpInfo(const ConstSpacePointProxy& spM);

  /// Creates a new doublet seed finder instance given the configuration.
  /// @param config Configuration for the doublet seed finder
  /// @return Unique pointer to new DoubletSeedFinder instance
  static std::unique_ptr<DoubletSeedFinder> create(const DerivedConfig& config);

  virtual ~DoubletSeedFinder() = default;

  /// Returns the configuration of the doublet seed finder.
  /// @return Reference to the configuration object
  virtual const DerivedConfig& config() const = 0;

  /// Creates compatible dublets by applying a series of cuts that can be
  /// tested with only two SPs.
  ///
  /// @param middleSp Space point candidate to be used as middle SP in a seed
  /// @param middleSpInfo Information about the middle space point
  /// @param candidateSps Subset of space points to be used as candidates for
  ///   middle SP in a seed
  /// @param compatibleDoublets Output container for compatible doublets
  virtual void createDoublets(
      const ConstSpacePointProxy& middleSp, const MiddleSpInfo& middleSpInfo,
      SpacePointContainer::ConstSubset candidateSps,
      DoubletsForMiddleSp& compatibleDoublets) const = 0;

  /// Creates compatible dublets by applying a series of cuts that can be
  /// tested with only two SPs.
  ///
  /// @param middleSp Space point candidate to be used as middle SP in a seed
  /// @param middleSpInfo Information about the middle space point
  /// @param candidateSps Range of space points to be used as candidates for
  ///   middle SP in a seed
  /// @param compatibleDoublets Output container for compatible doublets
  virtual void createDoublets(
      const ConstSpacePointProxy& middleSp, const MiddleSpInfo& middleSpInfo,
      SpacePointContainer::ConstRange candidateSps,
      DoubletsForMiddleSp& compatibleDoublets) const = 0;
};

}  // namespace Acts

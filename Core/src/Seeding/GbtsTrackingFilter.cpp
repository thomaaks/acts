// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/GbtsTrackingFilter.hpp"

#include "Acts/Seeding/GbtsGeometry.hpp"
#include "Acts/Utilities/MathHelpers.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <utility>

namespace Acts::Experimental::detail {

void detail::GbtsEdgeState::initialize(const detail::GbtsEdge& pS,
                                       const detail::GbtsNodeView& nodeView,
                                       const std::array<float, 3>& varianceX,
                                       const std::array<float, 2>& varianceY) {
  j = 0;
  vs.clear();

  // n2->n1

  // (x, y, z, r)
  const std::array<float, 4>& n1 = nodeView.positions[pS.n1];
  const std::array<float, 4>& n2 = nodeView.positions[pS.n2];

  const float dx = n1[0] - n2[0];
  const float dy = n1[1] - n2[1];
  const float L = std::sqrt(dx * dx + dy * dy);

  s = dy / L;
  c = dx / L;

  // transform for extrapolation and update
  //  x' =  x*c + y*s
  //  y' = -x*s + y*c

  refY = n2[3];
  refX = n2[0] * c + n2[1] * s;

  // X-state: y, dy/dx, d2y/dx2

  x = {-n2[0] * s + n2[1] * c, 0, 0};

  // Y-state: z, dz/dr

  y = {n2[2], (n1[2] - n2[2]) / (n1[3] - n2[3])};

  cx = {{{varianceX[0], 0, 0}, {0, varianceX[1], 0}, {0, 0, varianceX[2]}}};
  cy = {{{varianceY[0], 0}, {0, varianceY[1]}}};
}

}  // namespace Acts::Experimental::detail

namespace Acts::Experimental {

GbtsTrackingFilter::GbtsTrackingFilter(
    const Config& config, const std::shared_ptr<const GbtsGeometry>& geometry,
    std::unique_ptr<const Logger> logger)
    : m_cfg(config), m_geometry(geometry), m_logger(std::move(logger)) {}

bool GbtsTrackingFilter::update(const detail::GbtsNodeView& nodeView,
                                const detail::GbtsEdge& pS,
                                detail::GbtsEdgeState& ts) const {
  // add ms.

  const float tau2 = ts.y[1] * ts.y[1];
  const float invSin2 = 1 + tau2;

  // (x, y, z, r)
  const std::array<float, 4>& n1 = nodeView.positions[pS.n1];

  const float lenCorr =
      m_geometry->layerDescription(nodeView.layers[pS.n2]).type ==
              GbtsLayerType::Barrel
          ? invSin2
          : invSin2 / tau2;

  // the curvature as a fraction of the largest, as the minimum pT's
  const float corrMS = m_cfg.sigmaMS * (std::abs(ts.x[2]) / m_cfg.maxCurvature);

  const float sigma2 = m_cfg.radLen * lenCorr * corrMS * corrMS;  // /invSin2

  ts.cx[1][1] += sigma2;

  ts.cy[1][1] += sigma2;

  // extrapolation

  std::array<std::array<float, 3>, 3> Cx{};
  std::array<std::array<float, 2>, 2> Cy{};

  const float x = n1[0];
  const float y = n1[1];
  const float r = n1[3];

  const float refX = x * ts.c + y * ts.s;
  const float mx = -x * ts.s + y * ts.c;  // measured X[0]; Y[0]'s is z

  const float A = refX - ts.refX;
  const float B = 0.5 * A * A;
  const float dr = r - ts.refY;

  const std::array<float, 3> X = {ts.x[0] + ts.x[1] * A + ts.x[2] * B,
                                  ts.x[1] + ts.x[2] * A, ts.x[2]};

  Cx[0][0] = ts.cx[0][0] + 2 * ts.cx[0][1] * A + 2 * ts.cx[0][2] * B +
             A * A * ts.cx[1][1] + 2 * A * B * ts.cx[1][2] +
             B * B * ts.cx[2][2];
  Cx[0][1] = Cx[1][0] = ts.cx[0][1] + ts.cx[1][1] * A + ts.cx[1][2] * B +
                        ts.cx[0][2] * A + A * A * ts.cx[1][2] +
                        A * B * ts.cx[2][2];
  Cx[0][2] = Cx[2][0] = ts.cx[0][2] + ts.cx[1][2] * A + ts.cx[2][2] * B;

  Cx[1][1] = ts.cx[1][1] + 2 * A * ts.cx[1][2] + A * A * ts.cx[2][2];
  Cx[1][2] = Cx[2][1] = ts.cx[1][2] + ts.cx[2][2] * A;

  Cx[2][2] = ts.cx[2][2];

  const std::array<float, 2> Y = {ts.y[0] + ts.y[1] * dr, ts.y[1]};

  Cy[0][0] = ts.cy[0][0] + 2 * ts.cy[0][1] * dr + dr * dr * ts.cy[1][1];
  Cy[0][1] = Cy[1][0] = ts.cy[0][1] + dr * ts.cy[1][1];
  Cy[1][1] = ts.cy[1][1];

  // chi2 test

  const float resid_x = mx - X[0];
  const float resid_y = n1[2] - Y[0];

  const GbtsLayerType type =
      m_geometry->layerDescription(nodeView.layers[pS.n1]).type;

  // Across the strips a pair is sharper than a pixel, along them far worse:
  // the crossing is unconstrained over the strip, a half length over sqrt(3)
  // off its half vector -- the outer strip's, a stereo angle from the inner.
  float sigmaX = m_cfg.sigmaX;
  float sigmaY = m_cfg.sigmaY;
  if (const auto* strip = nodeView.strip(pS.n1); strip != nullptr) {
    constexpr float invSqrt3 = std::numbers::inv_sqrt3_v<float>;
    const std::array<float, 3>& half = strip->outerHalfVector;
    // the walk is along the strip, so project its reach onto the fit's axes
    const float alongX = -half[0] * ts.s + half[1] * ts.c;
    sigmaX = fastHypot(m_cfg.sigmaXStrip, alongX * invSqrt3);
    sigmaY = (type == GbtsLayerType::Barrel ? std::abs(half[2])
                                            : fastHypot(half[0], half[1])) *
             invSqrt3;
  }

  const float sigma_rz =
      square(type == GbtsLayerType::Barrel ? sigmaY : sigmaY * ts.y[1]);

  const float Dx = 1.0f / (Cx[0][0] + sigmaX * sigmaX);

  const float Dy = 1.0f / (Cy[0][0] + sigma_rz);

  const float dchi2_x = resid_x * resid_x * Dx;
  const float dchi2_y = resid_y * resid_y * Dy;

  if (dchi2_x > m_cfg.maxDChi2X || dchi2_y > m_cfg.maxDChi2Y) {
    return false;
  }

  ts.j += m_cfg.addHit - dchi2_x * m_cfg.weightX - dchi2_y * m_cfg.weightY;

  // state update

  const std::array<float, 3> Kx = {Dx * Cx[0][0], Dx * Cx[0][1], Dx * Cx[0][2]};
  const std::array<float, 2> Ky = {Dy * Cy[0][0], Dy * Cy[0][1]};

  for (std::uint32_t i = 0; i < 3; ++i) {
    ts.x[i] = X[i] + Kx[i] * resid_x;
  }

  if (std::abs(ts.x[2]) > m_cfg.maxCurvature) {
    return false;
  }

  for (std::uint32_t i = 0; i < 2; ++i) {
    ts.y[i] = Y[i] + Ky[i] * resid_y;
  }

  const float z0 = ts.y[0] - r * ts.y[1];

  if (std::abs(z0) > m_cfg.maxZ0) {
    return false;
  }

  for (std::uint32_t i = 0; i < 3; ++i) {
    for (std::uint32_t j = 0; j < 3; ++j) {
      ts.cx[i][j] = Cx[i][j] - Kx[i] * Cx[0][j];
    }
  }

  for (std::uint32_t i = 0; i < 2; ++i) {
    for (std::uint32_t j = 0; j < 2; ++j) {
      ts.cy[i][j] = Cy[i][j] - Ky[i] * Cy[0][j];
    }
  }
  ts.refX = refX;
  ts.refY = r;
  return true;
}

}  // namespace Acts::Experimental

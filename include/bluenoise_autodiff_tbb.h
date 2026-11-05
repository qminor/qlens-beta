#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <limits>
#include <utility>
#include <numeric>
#include <functional>
#include <iostream>
#include <Eigen/Dense>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/enumerable_thread_specific.h>

#include "types.h"
#include "mathexpr.h"
#include "pixelgrid.h"
#include "LBFGSmod.h"

using namespace LBFGSpp;

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

struct BlueNoiseParameters {
	int densityK = 10;
	double spacingScale = 0.75;
	double repulsionStrength = 1.0;
	double tetherStrength = 0.05;
	double stepSize = 0.001;
	double maxStep = 0.0;
	int iterations = 68;
	double site_displacement = 0;
	double repulsion_epsilon = 1e-12;
	double minStepSize = 1e-8;
	double maxStepSize = 1.0;
	double backtrackFactor = 0.5;
	double armijoConstant = 1e-4;
	int maxLineSearchIterations = 20;
	double energy_tolerance = 1e-4;
	double gradient_tolerance = 1e-4;
	double gradient_tolerance_rel = 1e-3;
};

template <typename VecType, typename Scalar>
class BlueNoiseRelaxation
{
public:
	using SiteVector = Eigen::VectorX<Scalar>;
	using SiteMatrix = Eigen::MatrixX<Scalar>;

#ifdef USE_STAN
	using GradientType = std::conditional_t<stan::is_autodiff_v<Scalar>, stan::math::var_value<Eigen::MatrixXd>, Eigen::MatrixXd>;
#else
	using GradientType = Eigen::MatrixXd;
#endif

private:
	BlueNoiseParameters params_;
	bool use_nn_interpolation = false;
	int n_datapts = 0;
	int n_sites = 0;
	int n_fixed_sites = 0;
	VecType data_x;
	VecType data_y;
	SiteMatrix sites_;
	SiteMatrix originalSites_;
	Eigen::VectorX<bool> fixedBoundary_;
	Eigen::VectorXd movableMask_;
	SiteVector localSpacing_;
	SiteVector localSpacing_interp;
	SiteVector localSpacing_xder;
	SiteVector localSpacing_yder;
	std::vector<int> site_ivals;
	std::vector<int> site_jvals;
	std::vector<std::pair<int, int>> neighborEdges_;
	DelaunaySourceGrid* delaunay_grid = nullptr;

public:
	BlueNoiseRelaxation(const VecType& dataX, const VecType& dataY, const SiteVector& siteX, const SiteVector& siteY, const Eigen::VectorX<bool>& fixedBoundary, const bool use_nn) : BlueNoiseRelaxation(dataX, dataY, siteX, siteY, fixedBoundary, BlueNoiseParameters(), use_nn) {}

	BlueNoiseRelaxation(const VecType& dataX, const VecType& dataY, const SiteVector& siteX, const SiteVector& siteY, const Eigen::VectorX<bool>& fixedBoundary, const BlueNoiseParameters& params, const bool use_nn) : params_(params), use_nn_interpolation(use_nn), data_x(dataX), data_y(dataY)
	{
		if (dataX.size() != dataY.size()) throw std::invalid_argument("dataX and dataY must have the same size.");
		if (siteX.size() != siteY.size()) throw std::invalid_argument("siteX and siteY must have the same size.");
		if (siteX.size() != fixedBoundary.size()) throw std::invalid_argument("fixedBoundary must have one entry per site.");
		if (dataX.size() == 0) throw std::invalid_argument("At least one data point is required.");
		if (siteX.size() == 0) throw std::invalid_argument("At least one site is required.");
		if (params_.densityK <= 0) throw std::invalid_argument("densityK must be positive.");
		if (params_.iterations < 0) throw std::invalid_argument("iterations must be non-negative.");
		n_datapts = static_cast<int>(dataX.size());
		n_sites = static_cast<int>(siteX.size());
		fixedBoundary_ = fixedBoundary;
		sites_.resize(n_sites, 2);
		sites_.col(0) = siteX;
		sites_.col(1) = siteY;
		originalSites_ = sites_;
		site_ivals.resize(n_sites);
		site_jvals.resize(n_sites);
		movableMask_.resize(n_sites);
		n_fixed_sites = 0;
		for (int i = 0; i < n_sites; ++i) {
			movableMask_[i] = fixedBoundary_[i] ? 0.0 : 1.0;
			if (fixedBoundary_[i]) ++n_fixed_sites;
		}
		localSpacing_.resize(n_sites);
		localSpacing_interp.resize(n_sites);
		localSpacing_xder.resize(n_sites);
		localSpacing_yder.resize(n_sites);
		computeLocalSpacing();
		for (Eigen::Index i = 0; i < n_sites; ++i) {
			if (!fixedBoundary_[i]) {
				sites_(i, 0) += params_.site_displacement;
				sites_(i, 1) += params_.site_displacement;
			}
		}

	}

private:

	struct LocalSpacingCache {
		std::vector<std::vector<int>> nearest;
	};

	double dataXValue(int i) const {
#ifdef USE_STAN
		if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) return data_x.val()(i);
		else
#endif
		return data_x(i);
	}

	double dataYValue(int i) const {
#ifdef USE_STAN
		if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) return data_y.val()(i);
		else
#endif
		return data_y(i);
	}

	static double scalarValue(const Scalar& x) {
#ifdef USE_STAN
		if constexpr (stan::is_autodiff_v<Scalar>) return stan::math::value_of(x);
		else
#endif
		return static_cast<double>(x);
	}

	void computeLocalSpacing()
	{
		const int k = std::min(params_.densityK, n_datapts);

		LocalSpacingCache cache;
		cache.nearest.resize(n_sites);

		Eigen::VectorXd spacing(n_sites);
		Eigen::VectorXd dxsum = Eigen::VectorXd::Zero(n_sites);
		Eigen::VectorXd dysum = Eigen::VectorXd::Zero(n_sites);
		Eigen::VectorXd derivativeScale = Eigen::VectorXd::Zero(n_sites);

		tbb::parallel_for(tbb::blocked_range<int>(0, n_sites), [&](const tbb::blocked_range<int>& range) {
			for (int s = range.begin(); s != range.end(); ++s) {
				const double px = scalarValue(sites_(s, 0));
				const double py = scalarValue(sites_(s, 1));

				std::vector<std::pair<double, int>> distances;
				distances.reserve(n_datapts);

				for (int i = 0; i < n_datapts; ++i) {
					const double dx = px - dataXValue(i);
					const double dy = py - dataYValue(i);
					const double d2 = dx * dx + dy * dy;
					if (d2 > 0.0) distances.emplace_back(d2, i);
				}

				const int nkeep = std::min(k, static_cast<int>(distances.size()));

				std::partial_sort(distances.begin(), distances.begin() + nkeep, distances.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

				double sum = 0.0;
				auto& nearest = cache.nearest[s];
				nearest.reserve(nkeep);

				for (int j = 0; j < nkeep; ++j) {
					sum += distances[j].first;
					nearest.push_back(distances[j].second);
				}

				spacing[s] = nkeep > 0 ? params_.spacingScale * std::sqrt(sum / nkeep) : 0.0;

				if (nkeep > 0 && sum > 0.0) {
					derivativeScale[s] = params_.spacingScale / (nkeep * std::sqrt(sum / nkeep));
					for (int idx : nearest) {
						dxsum[s] += px - dataXValue(idx);
						dysum[s] += py - dataYValue(idx);
					}
				}
			}
		});

#ifdef USE_STAN
		const int n = n_sites;
		const SiteMatrix sitesHandle = sites_;
		const VecType dataXHandle = data_x;
		const VecType dataYHandle = data_y;

		if constexpr (stan::is_autodiff_v<Scalar> && std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
			stan::math::var_value<Eigen::VectorXd> spacing_var = stan::math::make_callback_var(spacing, [cache = std::move(cache), dxsum = std::move(dxsum), dysum = std::move(dysum), derivativeScale = std::move(derivativeScale), n, sitesHandle, dataXHandle, dataYHandle](const auto& res) mutable {
				const Eigen::VectorXd& spacing_adj = res.adj();
				Eigen::VectorXd siteXAdj = Eigen::VectorXd::Zero(n);
				Eigen::VectorXd siteYAdj = Eigen::VectorXd::Zero(n);
				Eigen::VectorXd dataXAdj = Eigen::VectorXd::Zero(dataXHandle.size());
				Eigen::VectorXd dataYAdj = Eigen::VectorXd::Zero(dataYHandle.size());

				for (int s = 0; s < n; ++s) {
					const double adj = spacing_adj[s] * derivativeScale[s];
					if (adj == 0.0) continue;

					siteXAdj[s] += adj * dxsum[s];
					siteYAdj[s] += adj * dysum[s];

					for (int idx : cache.nearest[s]) {
						dataXAdj[idx] -= adj * (scalarValue(sitesHandle(s, 0)) - dataXHandle.val()(idx));
						dataYAdj[idx] -= adj * (scalarValue(sitesHandle(s, 1)) - dataYHandle.val()(idx));
					}
				}

				for (int s = 0; s < n; ++s) {
					sitesHandle(s, 0).vi_->adj_ += siteXAdj[s];
					sitesHandle(s, 1).vi_->adj_ += siteYAdj[s];
				}

				dataXHandle.adj() += dataXAdj;
				dataYHandle.adj() += dataYAdj;
			});

			for (int s = 0; s < n; ++s) localSpacing_[s] = spacing_var(s);
		} else
#endif
		{
			for (int s = 0; s < n_sites; ++s) localSpacing_[s] = Scalar(spacing[s]);
		}
	}



	/*
	void computeLocalSpacing()
	{
		const int k = std::min(params_.densityK, n_datapts);
		LocalSpacingCache cache;
		cache.nearest.resize(n_sites);
		Eigen::VectorXd spacing(n_sites);
		Eigen::VectorXd dxsum = Eigen::VectorXd::Zero(n_sites);
		Eigen::VectorXd dysum = Eigen::VectorXd::Zero(n_sites);
		Eigen::VectorXd derivativeScale = Eigen::VectorXd::Zero(n_sites);

		//std::chrono::steady_clock::time_point mytime0 = std::chrono::steady_clock::now();
		//std::chrono::duration<double> mytime;

		for (int s = 0; s < n_sites; ++s) {
			const double px = scalarValue(sites_(s, 0));
			const double py = scalarValue(sites_(s, 1));
			std::vector<std::pair<double, int>> distances;
			distances.reserve(n_datapts);

			for (int i = 0; i < n_datapts; ++i) {
				const double dx = px - dataXValue(i);
				const double dy = py - dataYValue(i);
				const double d2 = dx * dx + dy * dy;
				if (d2 > 0.0) distances.emplace_back(d2, i);
			}

			const int nkeep = std::min(k, static_cast<int>(distances.size()));
			std::partial_sort(distances.begin(), distances.begin() + nkeep, distances.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

			double sum = 0.0;
			for (int j = 0; j < nkeep; ++j) {
				sum += distances[j].first;
				cache.nearest[s].push_back(distances[j].second);
			}

			spacing[s] = nkeep > 0 ? params_.spacingScale * std::sqrt(sum / nkeep) : 0.0;

			if (nkeep > 0 && sum > 0.0) {
				derivativeScale[s] = params_.spacingScale / (nkeep * std::sqrt(sum / nkeep));
				for (int idx : cache.nearest[s]) {
					dxsum[s] += px - dataXValue(idx);
					dysum[s] += py - dataYValue(idx);
				}
			}
		}

		//mytime = std::chrono::steady_clock::now() - mytime0;
		//cout << "Wall time for computing local spacing (forward part): " << mytime.count() << endl;

#ifdef USE_STAN
		const int n = n_sites;

		const SiteMatrix sitesHandle = sites_;
		const VecType dataXHandle = data_x;
		const VecType dataYHandle = data_y;

		if constexpr ((stan::is_autodiff_v<Scalar>) && (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>)) {
			stan::math::var_value<Eigen::VectorXd> spacing_var = stan::math::make_callback_var(spacing, [cache = std::move(cache), dxsum = std::move(dxsum), dysum = std::move(dysum), derivativeScale = std::move(derivativeScale), n, sitesHandle, dataXHandle, dataYHandle](const auto& res) mutable {
				const Eigen::VectorXd& spacing_adj = res.adj();
				Eigen::VectorXd siteXAdj = Eigen::VectorXd::Zero(n);
				Eigen::VectorXd siteYAdj = Eigen::VectorXd::Zero(n);
				Eigen::VectorXd dataXAdj = Eigen::VectorXd::Zero(dataXHandle.size());
				Eigen::VectorXd dataYAdj = Eigen::VectorXd::Zero(dataYHandle.size());

		std::chrono::steady_clock::time_point mytime0 = std::chrono::steady_clock::now();
		std::chrono::duration<double> mytime;


				for (int s = 0; s < n; ++s) {
					const double adj = spacing_adj[s] * derivativeScale[s];
					if (adj == 0.0) continue;

					siteXAdj[s] += adj * dxsum[s];
					siteYAdj[s] += adj * dysum[s];

					for (int idx : cache.nearest[s]) {
						dataXAdj[idx] -= adj * (scalarValue(sitesHandle(s, 0)) - dataXHandle.val()(idx));
						dataYAdj[idx] -= adj * (scalarValue(sitesHandle(s, 1)) - dataYHandle.val()(idx));
					}
				}

				for (int s = 0; s < n; ++s) {
					sitesHandle(s, 0).vi_->adj_ += siteXAdj[s];
					sitesHandle(s, 1).vi_->adj_ += siteYAdj[s];
				}

				dataXHandle.adj() += dataXAdj;
				dataYHandle.adj() += dataYAdj;
			mytime = std::chrono::steady_clock::now() - mytime0;
			cout << "Wall time for local spacing callback: " << mytime.count() << endl;
			});

			for (int s = 0; s < n; ++s) localSpacing_[s] = spacing_var(s);
		} else
#endif
		{
			for (int s = 0; s < n; ++s) localSpacing_[s] = Scalar(spacing[s]);
		}
	}
	*/


	/*
	void interpolateLocalSpacing()
	{
		if (delaunay_grid == nullptr) throw std::logic_error("BlueNoiseRelaxation: Delaunay grid has not been set.");

		// This doens't work yet; not sure if it's worth exploring further or not
		tbb::parallel_for(tbb::blocked_range<int>(0, n_sites), [&](const tbb::blocked_range<int>& range) {
			for (int i = range.begin(); i != range.end(); ++i) {
				if (!fixedBoundary_[i]) {
					if (use_nn_interpolation) {
						localSpacing_interp[i] = delaunay_grid->find_lensed_surface_brightness_direct_nn(sites_(i, 0), sites_(i, 1), site_ivals[i], site_jvals[i], localSpacing_xder[i], localSpacing_yder[i]);
					} else {
						localSpacing_interp[i] = delaunay_grid->find_lensed_surface_brightness_direct_3pt(sites_(i, 0), sites_(i, 1), site_ivals[i], site_jvals[i], localSpacing_xder[i], localSpacing_yder[i]);
					}
				} else {
					localSpacing_interp[i] = localSpacing_[i];
					localSpacing_xder[i] = Scalar(0);
					localSpacing_yder[i] = Scalar(0);
				}
			}
		});
	}
	*/


	void interpolateLocalSpacing()
	{
		for (int i = 0; i < n_sites; ++i) {
			if (!fixedBoundary_[i]) {
				if (delaunay_grid == nullptr) throw std::logic_error("BlueNoiseRelaxation: Delaunay grid has not been set.");
				if (use_nn_interpolation) localSpacing_interp[i] = delaunay_grid->find_lensed_surface_brightness_direct_nn(sites_(i, 0), sites_(i, 1), site_ivals[i], site_jvals[i], localSpacing_xder[i], localSpacing_yder[i]);
				else localSpacing_interp[i] = delaunay_grid->find_lensed_surface_brightness_direct_3pt(sites_(i, 0), sites_(i, 1), site_ivals[i], site_jvals[i], localSpacing_xder[i], localSpacing_yder[i]);
			} else {
				localSpacing_interp[i] = localSpacing_[i];
				localSpacing_xder[i] = 0.0;
				localSpacing_yder[i] = 0.0;
			}
		}
	}

	/*
	SiteMatrix getGradients() const
	{
		SiteMatrix gradient = SiteMatrix::Zero(n_sites, 2);

		for (const auto& edge : neighborEdges_) {
			const int i = edge.first;
			const int j = edge.second;
			const Scalar dx = sites_(i, 0) - sites_(j, 0);
			const Scalar dy = sites_(i, 1) - sites_(j, 1);
			Scalar d2 = dx * dx + dy * dy;
			if (scalarValue(d2) < params_.repulsion_epsilon) d2 = Scalar(params_.repulsion_epsilon);
			using std::sqrt;
			const Scalar d = sqrt(d2);
			const Scalar targetSpacing = Scalar(0.5) * (localSpacing_interp[i] + localSpacing_interp[j]);
			const Scalar force = Scalar(params_.repulsionStrength) * (d - targetSpacing);
			const Scalar invD = Scalar(1) / d;
			const Scalar gx = force * dx * invD;
			const Scalar gy = force * dy * invD;
			gradient(i, 0) += gx - Scalar(0.5) * force * localSpacing_xder[i];
			gradient(i, 1) += gy - Scalar(0.5) * force * localSpacing_yder[i];
			gradient(j, 0) -= gx;
			gradient(j, 1) -= gy;
			gradient(j, 0) -= Scalar(0.5) * force * localSpacing_xder[j];
			gradient(j, 1) -= Scalar(0.5) * force * localSpacing_yder[j];
		}

		// Now add tether force 
		for (int i = 0; i < n_sites; ++i) {
			if (fixedBoundary_[i]) continue;
			gradient(i, 0) += Scalar(2.0 * params_.tetherStrength) * (sites_(i, 0) - originalSites_(i, 0));
			gradient(i, 1) += Scalar(2.0 * params_.tetherStrength) * (sites_(i, 1) - originalSites_(i, 1));
		}
		return gradient;
	}
	*/


	GradientType getGradients() const
	{
		Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n_sites, 2);

#ifdef USE_STAN
		if constexpr (stan::is_autodiff_v<Scalar>)
		{
			struct EdgeData {
				int i, j;
				double Hxx, Hxy, Hyy;
				double nx, ny;
				double force;
				double k;
				double qix, qiy, qjx, qjy;
				bool clamped;
			};

			struct SiteData {
				Scalar x;
				Scalar y;
				Scalar ox;
				Scalar oy;
			};

			std::vector<SiteData> siteVars;
			siteVars.reserve(n_sites);

			std::vector<Scalar> spacingVars(n_sites);
			std::vector<Scalar> spacingXderVars(n_sites);
			std::vector<Scalar> spacingYderVars(n_sites);

			std::vector<bool> movable(n_sites);

			for (int i = 0; i < n_sites; ++i)
			{
				siteVars.push_back({sites_(i, 0), sites_(i, 1), originalSites_(i, 0), originalSites_(i, 1)});
				spacingVars[i] = localSpacing_interp[i];
				spacingXderVars[i] = localSpacing_xder[i];
				spacingYderVars[i] = localSpacing_yder[i];
				movable[i] = !fixedBoundary_[i];
			}

			const double k = params_.repulsionStrength;
			const double tether = params_.tetherStrength;

			std::vector<EdgeData> cache(neighborEdges_.size());

			tbb::enumerable_thread_specific<Eigen::MatrixXd> threadGradients([&]() {
				return Eigen::MatrixXd::Zero(n_sites, 2);
			});

			const int numEdges = static_cast<int>(neighborEdges_.size());

			tbb::parallel_for(tbb::blocked_range<int>(0, numEdges), [&](const tbb::blocked_range<int>& range) {
				Eigen::MatrixXd& localG = threadGradients.local();

				for (int eidx = range.begin(); eidx != range.end(); ++eidx) {
					const int i = neighborEdges_[eidx].first;
					const int j = neighborEdges_[eidx].second;

					const double dx = scalarValue(sites_(i, 0)) - scalarValue(sites_(j, 0));
					const double dy = scalarValue(sites_(i, 1)) - scalarValue(sites_(j, 1));

					const double d2 = dx * dx + dy * dy;
					const bool clamped = d2 < params_.repulsion_epsilon;
					const double r = std::sqrt(clamped ? params_.repulsion_epsilon : d2);

					const double invR = 1.0 / r;
					const double nx = dx * invR;
					const double ny = dy * invR;

					const double si = scalarValue(localSpacing_interp[i]);
					const double sj = scalarValue(localSpacing_interp[j]);
					const double targetSpacing = 0.5 * (si + sj);

					const double force = k * (r - targetSpacing);

					const double qix = scalarValue(localSpacing_xder[i]);
					const double qiy = scalarValue(localSpacing_yder[i]);
					const double qjx = scalarValue(localSpacing_xder[j]);
					const double qjy = scalarValue(localSpacing_yder[j]);

					const double gx = force * nx;
					const double gy = force * ny;

					localG(i, 0) += gx - 0.5 * force * qix;
					localG(i, 1) += gy - 0.5 * force * qiy;
					localG(j, 0) -= gx + 0.5 * force * qjx;
					localG(j, 1) -= gy + 0.5 * force * qjy;

					EdgeData edgeData;
					edgeData.i = i;
					edgeData.j = j;
					edgeData.nx = nx;
					edgeData.ny = ny;
					edgeData.force = force;
					edgeData.k = k;
					edgeData.clamped = clamped;
					edgeData.qix = qix;
					edgeData.qiy = qiy;
					edgeData.qjx = qjx;
					edgeData.qjy = qjy;

					if (clamped) {
						const double a = force * invR;
						edgeData.Hxx = a;
						edgeData.Hxy = 0.0;
						edgeData.Hyy = a;
					} else {
						edgeData.Hxx = k * nx * nx + force * invR * ny * ny;
						edgeData.Hxy = k * nx * ny - force * invR * nx * ny;
						edgeData.Hyy = k * ny * ny + force * invR * nx * nx;
					}

					cache[eidx] = edgeData;
				}
			});

			for (const Eigen::MatrixXd& localG : threadGradients) G += localG;


			/*
			for (const auto& edge : neighborEdges_)
			{
				const int i = edge.first;
				const int j = edge.second;

				const double dx = scalarValue(sites_(i, 0)) - scalarValue(sites_(j, 0));
				const double dy = scalarValue(sites_(i, 1)) - scalarValue(sites_(j, 1));

				const double d2 = dx * dx + dy * dy;
				const bool clamped = d2 < params_.repulsion_epsilon;
				const double r = std::sqrt(clamped ? params_.repulsion_epsilon : d2);

				const double invR = 1.0 / r;
				const double nx = dx * invR;
				const double ny = dy * invR;

				const double si = scalarValue(localSpacing_interp[i]);
				const double sj = scalarValue(localSpacing_interp[j]);
				const double targetSpacing = 0.5 * (si + sj);

				const double force = k * (r - targetSpacing);

				const double qix = scalarValue(localSpacing_xder[i]);
				const double qiy = scalarValue(localSpacing_yder[i]);
				const double qjx = scalarValue(localSpacing_xder[j]);
				const double qjy = scalarValue(localSpacing_yder[j]);

				const double gx = force * nx;
				const double gy = force * ny;

				G(i, 0) += gx - 0.5 * force * qix;
				G(i, 1) += gy - 0.5 * force * qiy;
				G(j, 0) -= gx + 0.5 * force * qjx;
				G(j, 1) -= gy + 0.5 * force * qjy;

				EdgeData e;
				e.i = i;
				e.j = j;
				e.nx = nx;
				e.ny = ny;
				e.force = force;
				e.k = k;
				e.clamped = clamped;

				e.qix = qix;
				e.qiy = qiy;
				e.qjx = qjx;
				e.qjy = qjy;

				if (clamped)
				{
					const double a = force * invR;
					e.Hxx = a;
					e.Hxy = 0.0;
					e.Hyy = a;
				}
				else
				{
					const double invR3 = invR * invR * invR;
					e.Hxx = k * nx * nx + force * invR * ny * ny;
					e.Hxy = k * nx * ny - force * invR * nx * ny;
					e.Hyy = k * ny * ny + force * invR * nx * nx;
				}

				cache.push_back(e);
			}
			*/

			// Add tether gradient.
			for (int i = 0; i < n_sites; ++i)
			{
				if (!movable[i]) continue;

				G(i, 0) += 2.0 * tether * (scalarValue(sites_(i, 0)) - scalarValue(originalSites_(i, 0)));
				G(i, 1) += 2.0 * tether * (scalarValue(sites_(i, 1)) - scalarValue(originalSites_(i, 1)));
			}

			return stan::math::make_callback_var(G, [siteVars = std::move(siteVars), spacingVars = std::move(spacingVars), spacingXderVars = std::move(spacingXderVars), spacingYderVars = std::move(spacingYderVars), movable = std::move(movable), cache = std::move(cache), tether](auto& vi) {
				const auto& adj = vi.adj();

		//std::chrono::steady_clock::time_point mytime0 = std::chrono::steady_clock::now();
		//std::chrono::duration<double> mytime;


				// Reverse pass through repulsion.
				for (const auto& e : cache)
				{
					const int i = e.i;
					const int j = e.j;

					const double aix = adj(i, 0);
					const double aiy = adj(i, 1);
					const double ajx = adj(j, 0);
					const double ajy = adj(j, 1);

					const double ax = aix - ajx;
					const double ay = aiy - ajy;

					const double c = -0.5 * (aix * e.qix + aiy * e.qiy + ajx * e.qjx + ajy * e.qjy);

					double vx = e.Hxx * ax + e.Hxy * ay;
					double vy = e.Hxy * ax + e.Hyy * ay;

					if (!e.clamped)
					{
						vx += c * e.k * e.nx;
						vy += c * e.k * e.ny;
					}

					siteVars[i].x.adj() += vx;
					siteVars[i].y.adj() += vy;
					siteVars[j].x.adj() -= vx;
					siteVars[j].y.adj() -= vy;

					const double spacingAdj = -0.5 * e.k * (ax * e.nx + ay * e.ny + c);

					spacingVars[i].adj() += spacingAdj;
					spacingVars[j].adj() += spacingAdj;

					spacingXderVars[i].adj() -= 0.5 * e.force * aix;
					spacingYderVars[i].adj() -= 0.5 * e.force * aiy;
					spacingXderVars[j].adj() -= 0.5 * e.force * ajx;
					spacingYderVars[j].adj() -= 0.5 * e.force * ajy;
				}

				// Reverse pass through tether.
				for (int i = 0; i < static_cast<int>(siteVars.size()); ++i)
				{
					if (!movable[i]) continue;

					const double ax = adj(i, 0);
					const double ay = adj(i, 1);

					siteVars[i].x.adj() += 2.0 * tether * ax;
					siteVars[i].y.adj() += 2.0 * tether * ay;

					siteVars[i].ox.adj() -= 2.0 * tether * ax;
					siteVars[i].oy.adj() -= 2.0 * tether * ay;
				}
			//mytime = std::chrono::steady_clock::now() - mytime0;
			//cout << "Wall time for force gradient callback: " << mytime.count() << endl;
			});
		}
		else
#endif
		{
			const int numEdges = static_cast<int>(neighborEdges_.size());

			tbb::enumerable_thread_specific<Eigen::MatrixXd> threadGradients([&]() {
				return Eigen::MatrixXd::Zero(n_sites, 2);
			});

			tbb::parallel_for(tbb::blocked_range<int>(0, numEdges), [&](const tbb::blocked_range<int>& range) {
				Eigen::MatrixXd& localG = threadGradients.local();

				for (int e = range.begin(); e != range.end(); ++e) {
					const int i = neighborEdges_[e].first;
					const int j = neighborEdges_[e].second;

					const double dx = scalarValue(sites_(i, 0)) - scalarValue(sites_(j, 0));
					const double dy = scalarValue(sites_(i, 1)) - scalarValue(sites_(j, 1));

					double d2 = dx * dx + dy * dy;
					if (d2 < params_.repulsion_epsilon) d2 = params_.repulsion_epsilon;

					const double d = std::sqrt(d2);
					const double nx = dx / d;
					const double ny = dy / d;

					const double targetSpacing = 0.5 * (scalarValue(localSpacing_interp[i]) + scalarValue(localSpacing_interp[j]));
					const double force = params_.repulsionStrength * (d - targetSpacing);

					localG(i, 0) += force * nx - 0.5 * force * scalarValue(localSpacing_xder[i]);
					localG(i, 1) += force * ny - 0.5 * force * scalarValue(localSpacing_yder[i]);
					localG(j, 0) -= force * nx + 0.5 * force * scalarValue(localSpacing_xder[j]);
					localG(j, 1) -= force * ny + 0.5 * force * scalarValue(localSpacing_yder[j]);
				}
			});

			for (const Eigen::MatrixXd& localG : threadGradients) G += localG;


			/*
			for (const auto& edge : neighborEdges_)
			{
				const int i = edge.first;
				const int j = edge.second;

				const double dx = scalarValue(sites_(i, 0)) - scalarValue(sites_(j, 0));
				const double dy = scalarValue(sites_(i, 1)) - scalarValue(sites_(j, 1));

				double d2 = dx * dx + dy * dy;
				if (d2 < params_.repulsion_epsilon) d2 = params_.repulsion_epsilon;

				const double d = std::sqrt(d2);
				const double nx = dx / d;
				const double ny = dy / d;

				const double targetSpacing = 0.5 * (scalarValue(localSpacing_interp[i]) + scalarValue(localSpacing_interp[j]));
				const double force = params_.repulsionStrength * (d - targetSpacing);

				G(i, 0) += force * nx - 0.5 * force * scalarValue(localSpacing_xder[i]);
				G(i, 1) += force * ny - 0.5 * force * scalarValue(localSpacing_yder[i]);
				G(j, 0) -= force * nx + 0.5 * force * scalarValue(localSpacing_xder[j]);
				G(j, 1) -= force * ny + 0.5 * force * scalarValue(localSpacing_yder[j]);
			}
			*/

			for (int i = 0; i < n_sites; ++i)
			{
				if (fixedBoundary_[i]) continue;

				G(i, 0) += 2.0 * params_.tetherStrength * (scalarValue(sites_(i, 0)) - scalarValue(originalSites_(i, 0)));
				G(i, 1) += 2.0 * params_.tetherStrength * (scalarValue(sites_(i, 1)) - scalarValue(originalSites_(i, 1)));
			}

			return G;
		}
	}

public:
	void setNeighborEdges(const std::vector<std::pair<int, int>>& edges) {
		neighborEdges_ = edges;
	}

	int relax(const bool verbal)
	{
		double step = params_.stepSize;
		int iteration = 0;
		std::chrono::steady_clock::time_point mytime0;
		std::chrono::duration<double> mytime;
		mytime0=std::chrono::steady_clock::now();

		for (; iteration < params_.iterations; ++iteration) {
			interpolateLocalSpacing();
			auto gradient = getGradients();
#ifdef USE_STAN
			const auto& gradient_val = [&]() -> const auto& {
				 if constexpr (stan::is_autodiff_v<Scalar>)
					  return gradient.val();
				 else
					  return gradient;
			}();
#else
			const auto& gradient_val = gradient;
#endif
			const double currentEnergy = energyValue();
			double gradNorm2 = 0.0;
			for (int i = 0; i < n_sites; ++i) if (!fixedBoundary_[i]) gradNorm2 += gradient_val(i, 0) * gradient_val(i, 0) + gradient_val(i, 1) * gradient_val(i, 1);
			if (gradNorm2 < 1e-8) break;
			const SiteMatrix oldSites = sites_;
			double trialStep = std::min(step, params_.maxStepSize);
			bool accepted = false;
			int ls = 0;
			for (; ls < params_.maxLineSearchIterations; ++ls) {
				sites_ = oldSites;
				for (int i = 0; i < n_sites; ++i) {
					if (!fixedBoundary_[i]) {
						sites_(i, 0) -= Scalar(trialStep) * gradient(i, 0);
						sites_(i, 1) -= Scalar(trialStep) * gradient(i, 1);
					}
				}
				interpolateLocalSpacing();
				const double trialEnergy = energyValue();
				const double requiredEnergy = currentEnergy - params_.armijoConstant * trialStep * gradNorm2;
				if (std::isfinite(trialEnergy) and (trialEnergy <= requiredEnergy)) {
					accepted = true;
					break;
				}
				trialStep *= params_.backtrackFactor;
				if (trialStep < params_.minStepSize) break;
			}
			if (!accepted) {
				sites_ = oldSites;
				interpolateLocalSpacing();
				break;
			}
			step = std::min(1.5 * trialStep, params_.maxStepSize);
			if (verbal) std::cout << "Iteration " << iteration << ", energy = " << energyValue() << ", step = " << trialStep << ", gradnorm2 = " << gradNorm2 << ", trial_steps (ls) = " << ls << std::endl;
		}
		//mytime = std::chrono::steady_clock::now() - mytime0;
		//cout << "Wall time for blue noise relaxation: " << mytime.count() << endl;

		return iteration;
	}

	int relax_LBFGS(const bool verbal)
	{
		class Objective_Func
		{
			private:
			bool verbal;
			int counter;
			int n, n_sites;
			BlueNoiseRelaxation* bn;
			public:
			Objective_Func(const int n_sites_in, const int n_fixed, const bool verbal_in, BlueNoiseRelaxation* bn_in) : n_sites(n_sites_in), verbal(verbal_in), bn(bn_in) {
				n = 2*(n_sites_in-n_fixed);
				counter = 0;
			}
			~Objective_Func() { }
			Scalar operator()(const Eigen::VectorX<Scalar>& params, Eigen::VectorX<Scalar>& grad)
			{
				int i,j;
				for (i=0, j=0; i < n_sites; i++) {
					if (!bn->fixedBoundary_[i]) {
						bn->sites_(i,0) = params[j++];
						bn->sites_(i,1) = params[j++];
					}
				}
				if (j != n) die("UH-OH j != n");

				bn->interpolateLocalSpacing();
				auto gradient = bn->getGradients();
#ifdef USE_STAN
				const auto& gradient_val = [&]() -> const auto& {
					 if constexpr (stan::is_autodiff_v<Scalar>)
						  return gradient.val();
					 else
						  return gradient;
				}();
#else
				const auto& gradient_val = gradient;
#endif
				double gradNorm2 = 0;
				for (i=0, j=0; i < n_sites; i++) {
					if (!bn->fixedBoundary_[i]) {
						grad(j++) = gradient(i,0);
						grad(j++) = gradient(i,1);
						gradNorm2 += gradient_val(i,0)*gradient_val(i,0) + gradient_val(i,1)*gradient_val(i,1);
					}
				}
				Scalar energy = bn->energy();
				if (verbal) std::cout << "Func evaluation " << counter++ << ", energy = " << scalarValue(energy) << ", gradnorm2=" << gradNorm2 << std::endl;
				return energy;
			}
		};

		LBFGSParam<Scalar> lbfgs_param;
		lbfgs_param.past = 5;
		lbfgs_param.delta = params_.energy_tolerance;
		lbfgs_param.epsilon = params_.gradient_tolerance;
		lbfgs_param.epsilon_rel = params_.gradient_tolerance_rel;
		lbfgs_param.max_iterations = params_.iterations;

		Objective_Func objective_func(n_sites,n_fixed_sites,verbal,this);

		int n_params = 2*(n_sites-n_fixed_sites);
		Eigen::VectorX<Scalar> paramvec(n_params);
		int i,j;
		for (i=0, j=0; i < n_sites; i++) {
			if (!fixedBoundary_[i]) {
				paramvec(j++) = sites_(i,0);
				paramvec(j++) = sites_(i,1);
			}
		}

		LBFGSSolver<Scalar> solver(lbfgs_param);
		Scalar min_energy;
		int convergence_status = -1;
		int n_iterations = solver.minimize(objective_func,paramvec,min_energy,convergence_status,verbal);
		if (verbal) {
			if (convergence_status==0) cout << "Gradient converged below threshold " << lbfgs_param.epsilon << endl;
			else if (convergence_status==1) cout << "Gradient converged to within relative threshold " << lbfgs_param.epsilon_rel << "*grad" << endl;
			else if (convergence_status==2) cout << "Energy converged to within threshold (before gradient converged) " << lbfgs_param.delta << endl;
			else if (convergence_status==3) cout << "Max number of iterations reached (" << lbfgs_param.max_iterations << ")" << endl;
			else if (convergence_status==4) cout << "Line search failure; best point returned after " << lbfgs_param.max_iterations << " iterations" << endl;
		}

		for (i=0, j=0; i < n_sites; i++) {
			if (!fixedBoundary_[i]) {
				sites_(i,0) = paramvec(j++);
				sites_(i,1) = paramvec(j++);
			}
		}
		return n_iterations;
	}

	double repulsionEnergyValue() const
	{
		return tbb::parallel_reduce(tbb::blocked_range<int>(0, static_cast<int>(neighborEdges_.size())), 0.0, [&](const tbb::blocked_range<int>& range, double sum) {
			for (int e = range.begin(); e != range.end(); ++e) {
				const int i = neighborEdges_[e].first;
				const int j = neighborEdges_[e].second;

				const double dx = scalarValue(sites_(i, 0)) - scalarValue(sites_(j, 0));
				const double dy = scalarValue(sites_(i, 1)) - scalarValue(sites_(j, 1));

				double d2 = dx * dx + dy * dy;
				if (d2 < params_.repulsion_epsilon) d2 = params_.repulsion_epsilon;

				const double d = std::sqrt(d2);
				const double targetSpacing = 0.5 * (scalarValue(localSpacing_interp[i]) + scalarValue(localSpacing_interp[j]));
				const double error = d - targetSpacing;

				sum += 0.5 * params_.repulsionStrength * error * error;
			}
			return sum;
		}, std::plus<double>());
	}

	double tetherEnergyValue() const
	{
		double sum = 0.0;

		for (int i = 0; i < n_sites; ++i) {
			if (fixedBoundary_[i]) continue;

			const double dx = scalarValue(sites_(i, 0)) - scalarValue(originalSites_(i, 0));
			const double dy = scalarValue(sites_(i, 1)) - scalarValue(originalSites_(i, 1));

			sum += dx * dx + dy * dy;
		}

		return params_.tetherStrength * sum;
	}

	double energyValue() const
	{
		return repulsionEnergyValue() + tetherEnergyValue();
	}

	Scalar energy() const {
		return repulsionEnergy() + tetherEnergy();
	}

	Scalar repulsionEnergy() const
	{
		Scalar sum = Scalar(0);
		for (const auto& edge : neighborEdges_) {
			const int i = edge.first;
			const int j = edge.second;
			const Scalar dx = sites_(i, 0) - sites_(j, 0);
			const Scalar dy = sites_(i, 1) - sites_(j, 1);
			Scalar d2 = dx * dx + dy * dy;
			if (scalarValue(d2) < params_.repulsion_epsilon) d2 = Scalar(params_.repulsion_epsilon);
			using std::sqrt;
			const Scalar d = sqrt(d2);
			const Scalar targetSpacing = Scalar(0.5) * (localSpacing_interp[i] + localSpacing_interp[j]);
			const Scalar error = d - targetSpacing;
			sum += Scalar(0.5 * params_.repulsionStrength) * error * error;
		}
		return sum;
	}

	Scalar tetherEnergy() const
	{
		Scalar sum = Scalar(0);
		for (int i = 0; i < n_sites; ++i) {
			if (!fixedBoundary_[i]) {
				const Scalar dx = sites_(i, 0) - originalSites_(i, 0);
				const Scalar dy = sites_(i, 1) - originalSites_(i, 1);
				sum += dx * dx + dy * dy;
			}
		}
		return Scalar(params_.tetherStrength) * sum;
	}


	SiteVector siteX() const {
		return sites_.col(0);
	}

	SiteVector siteY() const {
		return sites_.col(1);
	}

	const SiteVector& localSpacing() const {
		return localSpacing_;
	}

	void set_delaunay_grid(DelaunaySourceGrid* grid_in, int* ivals_in, int* jvals_in) {
		delaunay_grid = grid_in;
		const bool status = delaunay_grid->template assign_surface_brightness<Scalar>(localSpacing_.data(), n_sites);
		if (!status) throw std::runtime_error("size of Delaunay grid does not match number of sites");
		for (int i = 0; i < n_sites; ++i) {
			site_ivals[i] = ivals_in[i];
			site_jvals[i] = jvals_in[i];
		}
		interpolateLocalSpacing();
	}
};

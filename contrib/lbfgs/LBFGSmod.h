// Copyright (C) 2016-2026 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_LBFGS_H
#define LBFGSPP_LBFGS_H

#include <Eigen/Core>
#include <iostream>
#include "types.h"
#include "LBFGSpp_mod/Param.h"
#include "LBFGSpp_mod/BFGSMat.h"
#include "LBFGSpp_mod/LineSearchBacktracking.h"
#include "LBFGSpp_mod/LineSearchBracketing.h"
#include "LBFGSpp_mod/LineSearchNocedalWright.h"
#include "LBFGSpp_mod/LineSearchMoreThuente.h"


namespace LBFGSpp {

///
/// L-BFGS solver for unconstrained numerical optimization
///
template <typename Scalar, template <class> class LineSearch = LineSearchNocedalWright>
class LBFGSSolver
{
private:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MapVec = Eigen::Map<Vector>;

    const LBFGSParam<Scalar>& m_param;
    BFGSMat<Scalar> m_bfgs;
    Vector m_fx;
    Vector m_xp;
    Vector m_grad;
    Scalar m_gnorm;
    Vector m_gradp;
    Vector m_drt;

    // Reset internal variables
    inline void reset(int n)
    {
        const int m = m_param.m;
        m_bfgs.reset(n, m);
        m_xp.resize(n);
        m_grad.resize(n);
        m_gradp.resize(n);
        m_drt.resize(n);
        if (m_param.past > 0) m_fx.resize(m_param.past);
    }

public:
    ///
    /// Constructor for the L-BFGS solver.
    ///
    LBFGSSolver(const LBFGSParam<Scalar>& param) : m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function using the L-BFGS algorithm.
    ///
    /// f(x, grad) returns the objective function value and overwrites grad
    /// with the explicitly calculated gradient.
    ///
    template <typename Foo>
    inline int minimize(Foo& f, Vector& x, Scalar& fx, int& convergence_status, const bool verbal)
    {
        const int n = x.size();
        reset(n);

        const int fpast = m_param.past;

        // Evaluate function and compute gradient.
        fx = f(x, m_grad);
        m_gnorm = m_grad.norm();

        // Track the globally best point encountered.
        Vector best_x = x;
        Vector best_grad = m_grad;
        Scalar best_fx = fx;

        if (fpast > 0) m_fx[0] = fx;

        // Early exit if the initial x is already a minimizer.
        if (value_of(m_gnorm) <= value_of(m_param.epsilon))
        {
            convergence_status = 0;
            return 1;
        }

        if (value_of(m_gnorm) <= value_of(m_param.epsilon_rel) * value_of(x.norm()))
        {
            convergence_status = 1;
            return 1;
        }

        // Initial direction.
        m_drt.noalias() = -m_grad;

        // Initial step size.
        Scalar step = Scalar(1) / m_drt.norm();

        // Tolerance for s'y >= eps * (y'y).
        const double eps = std::numeric_limits<double>::epsilon();

        // s and y vectors.
        Vector vecs(n), vecy(n);

        int k = 1;
        for (;;)
        {
            // Save the current x and gradient.
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;
            Scalar dg = m_grad.dot(m_drt);
            const Scalar step_max = m_param.max_step;

            // Line search to update x, fx and gradient.
            const bool linesearch_success =
                LineSearch<Scalar>::LineSearch(
                    f, m_param, m_xp, m_drt, step_max,
                    step, fx, m_grad, dg, x, verbal);

            // Update the globally best point.
            if (value_of(fx) < value_of(best_fx))
            {
                best_fx = fx;
                best_x = x;
                best_grad = m_grad;
            }

            // New gradient norm.
            m_gnorm = m_grad.norm();

            // Convergence test -- gradient.
            if (value_of(m_gnorm) <= value_of(m_param.epsilon))
            {
                convergence_status = 0;
                return k;
            }

            if (value_of(m_gnorm) <= value_of(m_param.epsilon_rel) * value_of(x.norm()))
            {
                convergence_status = 1;
                return k;
            }

            // Convergence test -- objective function value.
            if (fpast > 0)
            {
                const Scalar fxd = m_fx[k % fpast];
                const double fxd_value = value_of(fxd);
                const double fx_value = value_of(fx);
                const double delta_value = value_of(m_param.delta);

                if (k >= fpast &&
                    std::abs(fxd_value - fx_value) <=
                    delta_value * std::max(
                        std::max(std::abs(fx_value), std::abs(fxd_value)),
                        1.0))
                {
                    convergence_status = 2;
                    return k;
                }

                m_fx[k % fpast] = fx;
            }

            // Recoverable line search failure.
            // Return the globally best point encountered.
            if (!linesearch_success)
            {
                x = best_x;
                fx = best_fx;
                m_grad = best_grad;
                m_gnorm = m_grad.norm();

                convergence_status = 4;
                return k;
            }

            // Maximum number of iterations.
            if (m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                convergence_status = 3;
                return k;
            }

            // Update s and y.
            vecs.noalias() = x - m_xp;
            vecy.noalias() = m_grad - m_gradp;

            if (value_of(vecs.dot(vecy)) > eps * value_of(vecy.squaredNorm()))
                m_bfgs.add_correction(vecs, vecy);

            // Recursive formula to compute d = -H * g.
            m_bfgs.apply_Hv(m_grad, -Scalar(1), m_drt);

            // Reset step = 1.0 as initial guess for the next line search.
            step = Scalar(1);
            k++;
        }

        return k;
    }

    ///
    /// Returning the gradient vector on the last iterate.
    ///
    const Vector& final_grad() const { return m_grad; }

    ///
    /// Returning the Euclidean norm of the final gradient.
    ///
    Scalar final_grad_norm() const { return m_gnorm; }

    ///
    /// Returning the approximate Hessian matrix on the last iterate.
    ///
    Matrix final_approx_hessian() const { return m_bfgs.get_Bmat(); }

    ///
    /// Returning the approximate inverse Hessian matrix on the last iterate.
    ///
    Matrix final_approx_inverse_hessian() const { return m_bfgs.get_Hmat(); }
};

}  // namespace LBFGSpp

#endif  // LBFGSPP_LBFGS_H

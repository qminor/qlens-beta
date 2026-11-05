// Copyright (C) 2016-2026 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_LINE_SEARCH_BACKTRACKING_H
#define LBFGSPP_LINE_SEARCH_BACKTRACKING_H

#include <Eigen/Core>
#include <stdexcept>  // std::runtime_error
#include "Param.h"

namespace LBFGSpp {

///
/// The backtracking line search algorithm for L-BFGS. Mainly for internal use.
///
template <typename Scalar>
class LineSearchBacktracking
{
private:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

public:
    ///
    /// Line search by backtracking.
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param param    Parameters for the L-BFGS algorithm.
    /// \param xp       The current point.
    /// \param drt      The current moving direction.
    /// \param step_max The upper bound for the step size that makes x feasible.
    ///                 Can be ignored for the L-BFGS solver.
    /// \param step     In: The initial step length.
    ///                 Out: The calculated step length.
    /// \param fx       In: The objective function value at the current point.
    ///                 Out: The function value at the new point.
    /// \param grad     In: The current gradient vector.
    ///                 Out: The gradient at the new point.
    /// \param dg       In: The inner product between drt and grad.
    ///                 Out: The inner product between drt and the new gradient.
    /// \param x        Out: The new point moved to.
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, const LBFGSParam<Scalar>& param,
                           const Vector& xp, const Vector& drt, const Scalar& step_max,
                           Scalar& step, Scalar& fx, Vector& grad, Scalar& dg, Vector& x)
    {
        // Decreasing and increasing factors
        const double dec = 0.5;
        const double inc = 2.1;

        // Check the value of step
        if (value_of(step) <= 0.0)
            throw std::invalid_argument("'step' must be positive");

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if (value_of(dg_init) > 0.0)
            throw std::logic_error("the moving direction increases the objective function value");

        const Scalar test_decr = param.ftol * dg_init;
        double width = 0.0;

        int iter;
        for (iter = 0; iter < param.max_linesearch; iter++)
        {
            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * drt;
            // Evaluate this candidate
            fx = f(x, grad);

            const double fx_value = value_of(fx);
            const double fx_init_value = value_of(fx_init);
            const double test_decr_value = value_of(test_decr);

            if (fx_value > fx_init_value + value_of(step) * test_decr_value || std::isnan(fx_value))
            {
                width = dec;
            }
            else
            {
                dg = grad.dot(drt);

                // Armijo condition is met
                if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
                    break;

                if (value_of(dg) < value_of(param.wolfe) * value_of(dg_init))
                {
                    width = inc;
                }
                else
                {
                    // Regular Wolfe condition is met
                    if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE)
                        break;

                    if (value_of(dg) > -value_of(param.wolfe) * value_of(dg_init))
                    {
                        width = dec;
                    }
                    else
                    {
                        // Strong Wolfe condition is met
                        break;
                    }
                }
            }

            if (value_of(step) < value_of(param.min_step))
                throw std::runtime_error("the line search step became smaller than the minimum value allowed");

            if (value_of(step) > value_of(param.max_step))
                throw std::runtime_error("the line search step became larger than the maximum value allowed");

            step *= width;
        }

        if (iter >= param.max_linesearch)
            throw std::runtime_error("the line search routine reached the maximum number of iterations");
    }
};

}  // namespace LBFGSpp

#endif  // LBFGSPP_LINE_SEARCH_BACKTRACKING_H

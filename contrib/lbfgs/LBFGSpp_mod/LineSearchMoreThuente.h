// Copyright (C) 2020-2026 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_LINE_SEARCH_MORE_THUENTE_H
#define LBFGSPP_LINE_SEARCH_MORE_THUENTE_H

#include <Eigen/Core>
#include <stdexcept>  // std::invalid_argument, std::runtime_error
#include "Param.h"

namespace LBFGSpp {

///
/// The line search algorithm by Moré and Thuente (1994).
/// Can be used for both L-BFGS and L-BFGS-B.
///
/// The algorithm finds a step size alpha satisfying the strong Wolfe
/// conditions using double-precision arithmetic. The objective function
/// may use stan::math::var internally, provided it returns a double
/// objective value and writes the gradient into a double-valued vector.
///
template <typename Scalar = double>
class LineSearchMoreThuente
{
private:
    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    // Minimizer of a quadratic function q(x) = c0 + c1*x + c2*x^2
    // interpolating fa, ga, and fb.
    static double quadratic_minimizer(
        const double& a,
        const double& b,
        const double& fa,
        const double& ga,
        const double& fb)
    {
        const double ba = b - a;
        const double w = 0.5 * ba * ga / (fa - fb + ba * ga);
        return a + w * ba;
    }

    // Minimizer of a quadratic function interpolating ga and gb.
    static double quadratic_minimizer(
        const double& a,
        const double& b,
        const double& ga,
        const double& gb)
    {
        const double w = ga / (ga - gb);
        return a + w * (b - a);
    }

    // Local minimizer of a cubic function interpolating
    // fa, ga, fb, and gb.
    static double cubic_minimizer(
        const double& a,
        const double& b,
        const double& fa,
        const double& fb,
        const double& ga,
        const double& gb,
        bool& exists)
    {
        using std::abs;
        using std::sqrt;

        const double apb = a + b;
        const double ba = b - a;
        const double ba2 = ba * ba;
        const double fba = fb - fa;
        const double gba = gb - ga;

        const double z3 = (ga + gb) * ba - 2.0 * fba;
        const double z2 = 0.5 * (gba * ba2 - 3.0 * apb * z3);
        const double z1 =
            fba * ba2 - apb * z2 - (a * apb + b * b) * z3;

        const double eps = std::numeric_limits<double>::epsilon();

        if (abs(z3) < eps * abs(z2) ||
            abs(z3) < eps * abs(z1))
        {
            exists = (z2 * ba > 0.0);
            return exists ? (-0.5 * z1 / z2) : b;
        }

        const double u = z2 / (3.0 * z3);
        const double v = z1 / z2;
        const double vu = v / u;

        exists = (vu <= 1.0);

        if (!exists)
            return b;

        double r1 = 0.0;
        double r2 = 0.0;

        if (abs(u) >= abs(v))
        {
            const double w = 1.0 + sqrt(1.0 - vu);
            r1 = -u * w;
            r2 = -v / w;
        }
        else
        {
            const double sqrtd =
                sqrt(abs(u)) * sqrt(abs(v)) * sqrt(1.0 - u / v);

            r1 = -u - sqrtd;
            r2 = -u + sqrtd;
        }

        return (z3 * ba > 0.0)
            ? (std::max)(r1, r2)
            : (std::min)(r1, r2);
    }

    // Select the next step using interpolation and safeguarding.
    static double step_selection(
        const double& al,
        const double& au,
        const double& at,
        const double& fl,
        const double& fu,
        const double& ft,
        const double& gl,
        const double& gu,
        const double& gt)
    {
        using std::abs;

        if (al == au)
            return al;

        if (!std::isfinite(ft) || !std::isfinite(gt))
            return (al + at) / 2.0;

        bool ac_exists;

        const double ac =
            cubic_minimizer(al, at, fl, ft, gl, gt, ac_exists);

        const double aq = quadratic_minimizer(al, at, fl, gl, ft);

        // Case 1: ft > fl.
        if (ft > fl)
        {
            if (!ac_exists)
                return aq;

            return (abs(ac - al) < abs(aq - al))
                ? ac
                : (aq + ac) / 2.0;
        }

        const double as = quadratic_minimizer(al, at, gl, gt);

        // Case 2: opposite derivative signs.
        if (gt * gl < 0.0)
            return (abs(ac - at) >= abs(as - at)) ? ac : as;

        // Case 3: derivative magnitude decreases.
        const double deltal = 1.1;
        const double deltau = 0.66;

        if (abs(gt) < abs(gl))
        {
            const double res =
                (ac_exists &&
                 (ac - at) * (at - al) > 0.0 &&
                 abs(ac - at) < abs(as - at))
                    ? ac
                    : as;

            return (at > al)
                ? (std::min)(at + deltau * (au - at), res)
                : (std::max)(at + deltau * (au - at), res);
        }

        // Extrapolation when the upper endpoint is infinite.
        if (!std::isfinite(au) ||
            !std::isfinite(fu) ||
            !std::isfinite(gu))
        {
            return at + deltal * (at - al);
        }

        bool ae_exists;

        const double ae =
            cubic_minimizer(at, au, ft, fu, gt, gu, ae_exists);

        return (at > al)
            ? (std::min)(at + deltau * (au - at), ae)
            : (std::max)(at + deltau * (au - at), ae);
    }

public:
    ///
    /// Line search by Moré and Thuente (1994).
    ///
    /// The function object f(x, grad) returns the objective value
    /// as a double and writes the gradient into grad.
    ///
    /// All line-search bookkeeping is performed using double.
    ///
    template <typename Foo, typename SolverParam>
    static void LineSearch(
        Foo& f,
        const SolverParam& param,
        const Vector& xp,
        const Vector& drt,
        const double& step_max,
        double& step,
        double& fx,
        Vector& grad,
        double& dg,
        Vector& x)
    {
        using std::abs;

        const double step_min = static_cast<double>(param.min_step);

        if (step <= 0.0)
            throw std::invalid_argument("'step' must be positive");

        if (step < step_min)
            throw std::invalid_argument(
                "'step' is smaller than 'param.min_step'");

        if (step > step_max)
            throw std::invalid_argument("'step' exceeds 'step_max'");

        const double fx_init = fx;
        const double dg_init = dg;

        if (dg_init >= 0.0)
            throw std::logic_error(
                "the moving direction does not decrease the objective function value");

        const double test_decr =
            static_cast<double>(param.ftol) * dg_init;

        const double test_curv =
            -static_cast<double>(param.wolfe) * dg_init;

        const double Inf = std::numeric_limits<double>::infinity();

        double I_lo = 0.0;
        double I_hi = Inf;

        double fI_lo = 0.0;
        double fI_hi = Inf;

        double gI_lo =
            (1.0 - static_cast<double>(param.ftol)) * dg_init;

        double gI_hi = Inf;

        double psiI_lo = fI_lo;

        Vector x_lo = xp;
        Vector grad_lo = grad;

        double fx_lo = fx_init;
        double dg_lo = dg_init;

        bool bracketed = false;
        bool f_is_psi = true;
        bool use_step_min_safeguard = (step_min > 0.0);

        double I_width = Inf;
        double I_width_prev = Inf;

        int I_shrink_fail_count = 0;

        const double delta_max = 1.1;
        const double delta_min = 7.0 / 12.0;
        const double shrink = 0.66;

        int iter;

        for (iter = 0; iter < param.max_linesearch; ++iter)
        {
            // Evaluate the objective and gradient at the trial point.
            x.noalias() = xp + step * drt;

            fx = f(x, grad);
            dg = grad.dot(drt);

            const double psit =
                fx - fx_init - step * test_decr;

            const double dpsit = dg - test_decr;

            // Strong Wolfe convergence test.
            if (psit <= 0.0 && abs(dg) <= test_curv)
                return;

            // Lower-bound exit.
            if (step <= step_min &&
                (psit > 0.0 || dpsit >= 0.0))
            {
                return;
            }

            // Upper-bound exit.
            if (step >= step_max &&
                psit <= 0.0 && dpsit < 0.0)
            {
                return;
            }

            // Keep the auxiliary function psi for interpolation,
            // matching the behavior of the original implementation.
            const double ft = f_is_psi ? psit : fx;
            const double gt = f_is_psi ? dpsit : dg;

            // Disable the lower-bound safeguard once a valid descent
            // trial has been found.
            if (use_step_min_safeguard &&
                psit <= 0.0 && dpsit < 0.0)
            {
                use_step_min_safeguard = false;
            }

            double new_step;

            const bool in_case_2 =
                (psit <= psiI_lo) &&
                (dpsit * (I_lo - step) > 0.0);

            if (in_case_2)
            {
                new_step =
                    (std::min)(step_max,
                               step + delta_max * (step - I_lo));
            }
            else
            {
                new_step = step_selection(
                    I_lo, I_hi, step,
                    fI_lo, fI_hi, ft,
                    gI_lo, gI_hi, gt);

                new_step = (std::max)(new_step, step_min);
                new_step = (std::min)(new_step, step_max);

                if (use_step_min_safeguard)
                {
                    const double lower = step_min;
                    const double upper =
                        (std::max)(step_min, delta_min * step);

                    new_step = (std::max)(new_step, lower);
                    new_step = (std::min)(new_step, upper);
                }
            }

            // Update the bracketing interval.
            if (psit > psiI_lo)
            {
                // Case 1.
                I_hi = step;
                fI_hi = ft;
                gI_hi = gt;
            }
            else if (in_case_2)
            {
                // Case 2.
                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                psiI_lo = psit;

                x_lo.swap(x);
                grad_lo.swap(grad);

                fx_lo = fx;
                dg_lo = dg;
            }
            else
            {
                // Case 3.
                I_hi = I_lo;
                fI_hi = fI_lo;
                gI_hi = gI_lo;

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                psiI_lo = psit;

                x_lo.swap(x);
                grad_lo.swap(grad);

                fx_lo = fx;
                dg_lo = dg;
            }

            // Detect whether a valid bracket has been established.
            if (!bracketed && !in_case_2)
            {
                const double I_left = (std::min)(I_lo, I_hi);
                const double I_right = (std::max)(I_lo, I_hi);

                bracketed =
                    (I_left >= step_min && I_right <= step_max);
            }

            // Enforce interval shrinkage.
            if (bracketed)
            {
                I_width_prev = I_width;
                I_width = abs(I_hi - I_lo);

                if (I_width_prev < Inf &&
                    I_width > shrink * I_width_prev)
                {
                    ++I_shrink_fail_count;
                }
                else
                {
                    I_shrink_fail_count = 0;
                }

                if (I_shrink_fail_count >= 2)
                {
                    new_step = (I_lo + I_hi) / 2.0;
                    I_shrink_fail_count = 0;
                }
            }

            step = new_step;
        }

        // If the iteration limit is reached, return the best stored
        // lower-endpoint point, as in the original implementation.
        if (iter >= param.max_linesearch)
        {
            step = I_lo;
            fx = fx_lo;
            dg = dg_lo;

            x.swap(x_lo);
            grad.swap(grad_lo);
        }
    }
};

}  // namespace LBFGSpp

#endif  // LBFGSPP_LINE_SEARCH_MORE_THUENTE_H

// Copyright (C) 2020-2026 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_BK_LDLT_H
#define LBFGSPP_BK_LDLT_H

#include <vector>
#include <stdexcept>
#include <Eigen/Core>

/// \cond

namespace LBFGSpp {

enum COMPUTATION_INFO
{
    SUCCESSFUL = 0,
    NOT_COMPUTED,
    NUMERICAL_ISSUE
};

// Bunch-Kaufman LDLT decomposition
// References:
// 1. Bunch, J. R., & Kaufman, L. (1977). Some stable methods for calculating inertia and solving symmetric linear systems.
//    Mathematics of computation, 31(137), 163-179.
// 2. Golub, G. H., & Van Loan, C. F. (2012). Matrix computations (Vol. 3). JHU press. Section 4.4.
// 3. Bunch-Parlett diagonal pivoting <http://oz.nthu.edu.tw/~d947207/Chap13_GE3.ppt>
// 4. Ashcraft, C., Grimes, R. G., & Lewis, J. G. (1998). Accurate symmetric indefinite linear equation solvers.
//    SIAM Journal on Matrix Analysis and Applications, 20(2), 513-561.
template <typename Scalar = double>
class BKLDLT
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using MapVec = Eigen::Map<Vector>;
    using MapConstVec = Eigen::Map<const Vector>;

    using IntVector = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
    using GenericVector = Eigen::Ref<Vector>;
    using GenericMatrix = Eigen::Ref<Matrix>;
    using ConstGenericMatrix = const Eigen::Ref<const Matrix>;
    using ConstGenericVector = const Eigen::Ref<const Vector>;

    Index m_n;
    Vector m_data;
    std::vector<double*> m_colptr;
    IntVector m_perm;
    std::vector<std::pair<Index, Index> > m_permc;

    bool m_computed;
    int m_info;

    double* col_pointer(Index k) { return m_colptr[k]; }
    double& coeff(Index i, Index j) { return m_colptr[j][i - j]; }
    const double& coeff(Index i, Index j) const { return m_colptr[j][i - j]; }
    double& diag_coeff(Index i) { return m_colptr[i][0]; }
    const double& diag_coeff(Index i) const { return m_colptr[i][0]; }

    void compute_pointer()
    {
        m_colptr.clear();
        m_colptr.reserve(m_n);
        double* head = m_data.data();

        for (Index i = 0; i < m_n; i++)
        {
            m_colptr.push_back(head);
            head += (m_n - i);
        }
    }

    void copy_data(ConstGenericMatrix& mat, int uplo, const double& shift)
    {
        if (uplo == Eigen::Lower)
        {
            for (Index j = 0; j < m_n; j++)
            {
                const double* begin = &mat.coeffRef(j, j);
                const Index len = m_n - j;
                std::copy(begin, begin + len, col_pointer(j));
                diag_coeff(j) -= shift;
            }
        }
        else
        {
            double* dest = m_data.data();
            for (Index i = 0; i < m_n; i++)
            {
                for (Index j = i; j < m_n; j++, dest++)
                {
                    *dest = mat.coeff(i, j);
                }
                diag_coeff(i) -= shift;
            }
        }
    }

    void compress_permutation()
    {
        for (Index i = 0; i < m_n; i++)
        {
            const Index perm = (m_perm[i] >= 0) ? m_perm[i] : (-m_perm[i] - 1);
            if (perm != i)
                m_permc.push_back(std::make_pair(i, perm));
        }
    }

    void pivoting_1x1(Index k, Index r)
    {
        if (k == r)
        {
            m_perm[k] = r;
            return;
        }

        std::swap(diag_coeff(k), diag_coeff(r));
        std::swap_ranges(&coeff(r + 1, k), col_pointer(k + 1), &coeff(r + 1, r));

        double* src = &coeff(k + 1, k);
        for (Index j = k + 1; j < r; j++, src++)
        {
            std::swap(*src, coeff(r, j));
        }

        m_perm[k] = r;
    }

    void pivoting_2x2(Index k, Index r, Index p)
    {
        pivoting_1x1(k, p);
        pivoting_1x1(k + 1, r);

        std::swap(coeff(k + 1, k), coeff(r, k));

        m_perm[k] = -m_perm[k] - 1;
        m_perm[k + 1] = -m_perm[k + 1] - 1;
    }

    void interchange_rows(Index r1, Index r2, Index c1, Index c2)
    {
        if (r1 == r2)
            return;

        for (Index j = c1; j <= c2; j++)
        {
            std::swap(coeff(r1, j), coeff(r2, j));
        }
    }

    double find_lambda(Index k, Index& r)
    {
        using std::abs;

        const double* head = col_pointer(k);
        const double* end = col_pointer(k + 1);

        r = k + 1;
        double lambda = abs(head[1]);

        for (const double* ptr = head + 2; ptr < end; ptr++)
        {
            const double abs_elem = abs(*ptr);
            if (lambda < abs_elem)
            {
                lambda = abs_elem;
                r = k + (ptr - head);
            }
        }

        return lambda;
    }

    double find_sigma(Index k, Index r, Index& p)
    {
        using std::abs;

        double sigma = -1.0;
        if (r < m_n - 1)
            sigma = find_lambda(r, p);

        for (Index j = k; j < r; j++)
        {
            const double abs_elem = abs(coeff(r, j));
            if (sigma < abs_elem)
            {
                sigma = abs_elem;
                p = j;
            }
        }

        return sigma;
    }

    bool permutate_mat(Index k, const double& alpha)
    {
        using std::abs;

        Index r = k, p = k;
        const double lambda = find_lambda(k, r);

        if (lambda > 0.0)
        {
            const double abs_akk = abs(diag_coeff(k));

            if (abs_akk < alpha * lambda)
            {
                const double sigma = find_sigma(k, r, p);

                if (sigma * abs_akk < alpha * lambda * lambda)
                {
                    if (abs_akk >= alpha * sigma)
                    {
                        pivoting_1x1(k, r);
                        interchange_rows(k, r, 0, k - 1);
                        return true;
                    }
                    else
                    {
                        p = k;

                        pivoting_2x2(k, r, p);
                        interchange_rows(k, p, 0, k - 1);
                        interchange_rows(k + 1, r, 0, k - 1);
                        return false;
                    }
                }
            }
        }

        return true;
    }

    void inverse_inplace_2x2(double& e11, double& e21, double& e22) const
    {
        const double delta = e11 * e22 - e21 * e21;
        std::swap(e11, e22);
        e11 /= delta;
        e22 /= delta;
        e21 = -e21 / delta;
    }

    int gaussian_elimination_1x1(Index k)
    {
        const double akk = diag_coeff(k);
        if (akk == 0.0)
            return NUMERICAL_ISSUE;

        diag_coeff(k) = 1.0 / akk;

        double* lptr = col_pointer(k) + 1;
        const Index ldim = m_n - k - 1;
        MapVec l(lptr, ldim);

        for (Index j = 0; j < ldim; j++)
        {
            MapVec(col_pointer(j + k + 1), ldim - j).noalias()
                -= (lptr[j] / akk) * l.tail(ldim - j);
        }

        l /= akk;
        return SUCCESSFUL;
    }

    int gaussian_elimination_2x2(Index k)
    {
        double& e11 = diag_coeff(k);
        double& e21 = coeff(k + 1, k);
        double& e22 = diag_coeff(k + 1);

        if (e11 * e22 - e21 * e21 == 0.0)
            return NUMERICAL_ISSUE;

        inverse_inplace_2x2(e11, e21, e22);

        double* l1ptr = &coeff(k + 2, k);
        double* l2ptr = &coeff(k + 2, k + 1);
        const Index ldim = m_n - k - 2;
        MapVec l1(l1ptr, ldim), l2(l2ptr, ldim);

        Eigen::Matrix<double, Eigen::Dynamic, 2> X(ldim, 2);
        X.col(0).noalias() = l1 * e11 + l2 * e21;
        X.col(1).noalias() = l1 * e21 + l2 * e22;

        for (Index j = 0; j < ldim; j++)
        {
            MapVec(col_pointer(j + k + 2), ldim - j).noalias()
                -= (X.col(0).tail(ldim - j) * l1ptr[j]
                    + X.col(1).tail(ldim - j) * l2ptr[j]);
        }

        l1.noalias() = X.col(0);
        l2.noalias() = X.col(1);

        return SUCCESSFUL;
    }

public:
    BKLDLT() :
        m_n(0), m_computed(false), m_info(NOT_COMPUTED)
    {}

    BKLDLT(ConstGenericMatrix& mat, int uplo = Eigen::Lower, const double& shift = 0.0) :
        m_n(mat.rows()), m_computed(false), m_info(NOT_COMPUTED)
    {
        compute(mat, uplo, shift);
    }

    void compute(ConstGenericMatrix& mat, int uplo = Eigen::Lower, const double& shift = 0.0)
    {
        m_n = mat.rows();
        if (m_n != mat.cols())
            throw std::invalid_argument("BKLDLT: matrix must be square");

        m_perm.setLinSpaced(m_n, 0, m_n - 1);
        m_permc.clear();

        m_data.resize((m_n * (m_n + 1)) / 2);
        compute_pointer();
        copy_data(mat, uplo, shift);

        const double alpha = (1.0 + std::sqrt(17.0)) / 8.0;
        Index k = 0;

        for (k = 0; k < m_n - 1; k++)
        {
            const bool is_1x1 = permutate_mat(k, alpha);

            if (is_1x1)
            {
                m_info = gaussian_elimination_1x1(k);
            }
            else
            {
                m_info = gaussian_elimination_2x2(k);
                k++;
            }

            if (m_info != SUCCESSFUL)
                break;
        }

        if (k == m_n - 1)
        {
            const double akk = diag_coeff(k);
            if (akk == 0.0)
                m_info = NUMERICAL_ISSUE;

            diag_coeff(k) = 1.0 / diag_coeff(k);
        }

        compress_permutation();
        m_computed = true;
    }

    void solve_inplace(GenericVector b) const
    {
        if (!m_computed)
            throw std::logic_error("BKLDLT: need to call compute() first");

        double* x = b.data();
        MapVec res(x, m_n);
        Index npermc = m_permc.size();

        for (Index i = 0; i < npermc; i++)
            std::swap(x[m_permc[i].first], x[m_permc[i].second]);

        const Index end = (m_perm[m_n - 1] < 0) ? (m_n - 3) : (m_n - 2);
        for (Index i = 0; i <= end; i++)
        {
            const Index b1size = m_n - i - 1;
            const Index b2size = b1size - 1;

            if (m_perm[i] >= 0)
            {
                MapConstVec l(&coeff(i + 1, i), b1size);
                res.segment(i + 1, b1size).noalias() -= l * x[i];
            }
            else
            {
                MapConstVec l1(&coeff(i + 2, i), b2size);
                MapConstVec l2(&coeff(i + 2, i + 1), b2size);
                res.segment(i + 2, b2size).noalias() -= (l1 * x[i] + l2 * x[i + 1]);
                i++;
            }
        }

        for (Index i = 0; i < m_n; i++)
        {
            const double e11 = diag_coeff(i);
            if (m_perm[i] >= 0)
            {
                x[i] *= e11;
            }
            else
            {
                const double e21 = coeff(i + 1, i), e22 = diag_coeff(i + 1);
                const double wi = x[i] * e11 + x[i + 1] * e21;
                x[i + 1] = x[i] * e21 + x[i + 1] * e22;
                x[i] = wi;
                i++;
            }
        }

        Index i = (m_perm[m_n - 1] < 0) ? (m_n - 3) : (m_n - 2);
        for (; i >= 0; i--)
        {
            const Index ldim = m_n - i - 1;
            MapConstVec l(&coeff(i + 1, i), ldim);
            x[i] -= res.segment(i + 1, ldim).dot(l);

            if (m_perm[i] < 0)
            {
                MapConstVec l2(&coeff(i + 1, i - 1), ldim);
                x[i - 1] -= res.segment(i + 1, ldim).dot(l2);
                i--;
            }
        }

        for (i = npermc - 1; i >= 0; i--)
            std::swap(x[m_permc[i].first], x[m_permc[i].second]);
    }

    Vector solve(ConstGenericVector& b) const
    {
        Vector res = b;
        solve_inplace(res);
        return res;
    }

    int info() const { return m_info; }
};

}  // namespace LBFGSpp

/// \endcond

#endif  // LBFGSPP_BK_LDLT_H

#ifndef TYPES_H
#define TYPES_H


#include <Eigen/Core>
#include <Eigen/Sparse>

#ifdef USE_STAN
#include <stan/math.hpp>
#include <stan/math/mix/functor/hessian.hpp>
#endif

// Eventually, I'd like to be able to instantiate EigenTypes with fvar (forward-mode autodif variable), which will be useful for getting the Hessian.
// But for now, it's only being instantiated with doubles.
template <typename Scalar>
struct EigenTypes {
	using QScalar = Scalar;
	using VecType = Eigen::VectorX<Scalar>;
	using MatType = Eigen::MatrixX<Scalar>;
	using SparseMatType = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
};

using PlainTypes = EigenTypes<double>;

#ifdef USE_STAN
struct VarmatTypes {
	using QScalar = stan::math::var;
	using VecType = stan::math::var_value<Eigen::VectorXd>;
	using MatType = stan::math::var_value<Eigen::MatrixXd>;
	using SparseMatType = stan::math::var_value<Eigen::SparseMatrix<double, Eigen::ColMajor>>;
};
#endif

// this function is for assigning value that sorts out whether it's autodiff or not. (This should not add overhead as long as compiling with -O2 or -O3)
template <typename T>
inline auto value_of(const T& x)
{
#ifdef USE_STAN
    return stan::math::value_of(x);
#else
    return x;
#endif
}
#endif // TYPES_H

#ifndef HYP2F1_H
#define HYP2F1_H
#include <complex>
using namespace std;

//#include "complex_functions.H"

complex<double> Gamma_ratio_diff_small_eps (const complex<double> &z,const complex<double> &eps);
complex<double> Gamma_inv_diff_eps (const complex<double> &z,const complex<double> &eps);
complex<double> A_sum_init (const int m,const complex<double> &eps,const complex<double> &Gamma_inv_one_meps);
complex<double> log_A_sum_init (const int m,const complex<double> &eps);
complex<double> B_sum_init_PS_one (const complex<double> &a,const complex<double> &b,const complex<double> &c, const complex<double> &Gamma_c,const complex<double> &Gamma_inv_one_meps, const complex<double> &Gamma_inv_eps_pa_pm,const complex<double> &Gamma_inv_eps_pb_pm, const complex<double> &one_minus_z,const int m,const complex<double> &eps);

complex<double> B_sum_init_PS_infinity (const complex<double> &a,const complex<double> &c, const complex<double> &Gamma_c,const complex<double> &Gamma_inv_cma, const complex<double> &Gamma_inv_one_meps,const complex<double> &Gamma_inv_eps_pa_pm, const complex<double> &z,const int m,const complex<double> &eps);

void cv_poly_der_tab_calc (const complex<double> &a,const complex<double> &b,const complex<double> &c,const complex<double> &z,double cv_poly_der_tab[]);
double cv_poly_der_calc (const double cv_poly_der_tab[],const double x);
int min_n_calc (const double cv_poly_der_tab[]);
complex<double> hyp_PS_zero (const complex<double> &a,const complex<double> &b,const complex<double> &c,const complex<double> &z);
complex<double> hyp_PS_one (const complex<double> &a,const complex<double> &b,const complex<double> &c,const complex<double> &one_minus_z);
complex<double> hyp_PS_infinity (const complex<double> &a,const complex<double> &b,const complex<double> &c,const complex<double> &z);
complex<double> hyp_PS_complex_plane_rest (const complex<double> &a,const complex<double> &b,const complex<double> &c,const complex<double> &z);
complex<double> hyp_2F1 (const complex<double> &a,const complex<double> &b,const complex<double> &c,const complex<double> &z);
double test_2F1 (const complex<double> &a,const complex<double> &b,const complex<double> &c,const complex<double> &z,const complex<double> &F);

#endif //HYP2F1_H

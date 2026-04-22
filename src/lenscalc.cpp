#include "qlens.h"
#include "mathexpr.h"
#include "pixelgrid.h"

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

//template <>
//lensvector<double> *LensCalc<double>::defs, **LensCalc<double>::defs_subtot, *LensCalc<double>::defs_i, *LensCalc<double>::xvals_i;
//template <>
//lensmatrix<double> *LensCalc<double>::jacs, *LensCalc<double>::hesses, **LensCalc<double>::hesses_subtot, *LensCalc<double>::hesses_i, *LensCalc<double>::Amats_i;
//#ifdef USE_STAN
//template <>
//lensvector<stan::math::var> *LensCalc<stan::math::var>::defs, **LensCalc<stan::math::var>::defs_subtot, *LensCalc<stan::math::var>::defs_i, *LensCalc<stan::math::var>::xvals_i;
//template <>
//lensmatrix<stan::math::var> *LensCalc<stan::math::var>::jacs, *LensCalc<stan::math::var>::hesses, **LensCalc<stan::math::var>::hesses_subtot, *LensCalc<stan::math::var>::hesses_i, *LensCalc<stan::math::var>::Amats_i;
//#endif
template <>
lensvector<double> *LensCalc<double>::defs;
template <>
lensvector<double> **LensCalc<double>::defs_subtot;
template <>
lensvector<double> *LensCalc<double>::defs_i;
template <>
lensvector<double> *LensCalc<double>::xvals_i;
template <>
lensmatrix<double> *LensCalc<double>::jacs;
template <>
lensmatrix<double> *LensCalc<double>::hesses;
template <>
lensmatrix<double> **LensCalc<double>::hesses_subtot;
template <>
lensmatrix<double> *LensCalc<double>::hesses_i;
template <>
lensmatrix<double> *LensCalc<double>::Amats_i;
#ifdef USE_STAN
template <>
lensvector<stan::math::var> *LensCalc<stan::math::var>::defs;
template <>
lensvector<stan::math::var> **LensCalc<stan::math::var>::defs_subtot;
template <>
lensvector<stan::math::var> *LensCalc<stan::math::var>::defs_i;
template <>
lensvector<stan::math::var> *LensCalc<stan::math::var>::xvals_i;
template <>
lensmatrix<stan::math::var> *LensCalc<stan::math::var>::jacs;
template <>
lensmatrix<stan::math::var> *LensCalc<stan::math::var>::hesses;
template <>
lensmatrix<stan::math::var> **LensCalc<stan::math::var>::hesses_subtot;
template <>
lensmatrix<stan::math::var> *LensCalc<stan::math::var>::hesses_i;
template <>
lensmatrix<stan::math::var> *LensCalc<stan::math::var>::Amats_i;
#endif


template<typename QScalar>
QScalar QLens::kappa(const QScalar& x, const QScalar& y, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	QScalar kappa;
	if (n_lens_redshifts==1) {
		int j;
		kappa=0;
		for (j=0; j < nlens; j++) {
			kappa += lens_list[j]->kappa(x,y);
		}
		for (j=0; j < n_pixellated_lens; j++) {
			//if (lensgrids[j]->include_in_lensing_calculations) kappa += lensgrids[j]->kappa(x,y,0);
		}
		kappa *= zfacs[0];
	} else {
		lensmatrix<QScalar> *jac = &lc.jacs[0];
		hessian<QScalar>(x,y,(*jac),0,zfacs,betafacs);
		kappa = ((*jac)[0][0] + (*jac)[1][1])/2;
	}

	return kappa;
}
template double QLens::kappa<double>(const double& x, const double& y, double* zfacs, double** betafacs);
#ifdef USE_STAN
template stan::math::var QLens::kappa<stan::math::var>(const stan::math::var& x, const stan::math::var& y, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
QScalar QLens::potential(const double& x, const double& y, double* zfacs, double** betafacs)
{
	QScalar pot=0, pot_subtot;
	// This is not really sensical for multiplane lensing, and time delays need to be treated as in Schneider's textbook. Fix later
	int i,j;
	for (i=0; i < n_lens_redshifts; i++) {
		if (zfacs[i] != 0.0) {
			pot_subtot=0;
			for (j=0; j < zlens_group_size[i]; j++) {
				pot_subtot += lens_list[zlens_group_lens_indx[i][j]]->potential(x,y);
			}
			for (j=0; j < n_pixellated_lens; j++) {
				//if ((lensgrids[j]->include_in_lensing_calculations) and (pixellated_lens_redshift_idx[j]==i)) pot_subtot += lensgrids[j]->interpolate_potential(x,y,0);
			}
			pot += zfacs[i]*pot_subtot;
		}
	}
	return pot;
}
template double QLens::potential<double>(const double& x, const double& y, double* zfacs, double** betafacs);
#ifdef USE_STAN
template stan::math::var QLens::potential<stan::math::var>(const double& x, const double& y, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::deflection(const QScalar& x, const QScalar& y, lensvector<QScalar>& def_tot, const int &thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensvector<QScalar> *x_i = &lc.xvals_i[thread];
	lensvector<QScalar> *def = &lc.defs_i[thread];
	lensvector<QScalar> **def_i = &lc.defs_subtot[thread];

	int i,j;
	def_tot[0] = 0;
	def_tot[1] = 0;
	//std::cout << "n_redshifts=" << n_lens_redshifts << std::endl;
	//for (i=0; i < n_lens_redshifts; i++) {
		//std::cout << "Lens redshift" << i << " (z=" << lens_redshifts[i] << "): zfac=" << zfacs[i] << std::endl;
	//}
	for (i=0; i < n_lens_redshifts; i++) {
		if (zfacs[i] != 0.0) {
			//std::cout << "redshift " << i << ":\n";
			(*def_i)[i][0] = 0;
			(*def_i)[i][1] = 0;
			(*x_i)[0] = x;
			(*x_i)[1] = y;
			for (j=0; j < i; j++) {
				//std::cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
				(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
				(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
			}
			for (j=0; j < zlens_group_size[i]; j++) {
				lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
				//std::cout << "Lens redshift " << i << ", lens " << zlens_group_lens_indx[i][j] << " def=" << (*def)[0] << " " << (*def)[1] << std::endl;
				(*def_i)[i][0] += (*def)[0];
				(*def_i)[i][1] += (*def)[1];
			}
			for (j=0; j < n_pixellated_lens; j++) {
				if ((lensgrids[j]->include_in_lensing_calculations) and (pixellated_lens_redshift_idx[j]==i)) {
					//lensgrids[j]->deflection((*x_i)[0],(*x_i)[1],(*def),thread);
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
			}
			//std::cout << "Lens redshift" << i << " (z=" << lens_redshifts[i] << "): xi=" << (*x_i)[0] << " " << (*x_i)[1] << std::endl;
			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			def_tot[0] += (*def_i)[i][0];
			def_tot[1] += (*def_i)[i][1];
		}
	}
}
template void QLens::deflection<double>(const double& x, const double& y, lensvector<double>& def_tot, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::deflection<stan::math::var>(const stan::math::var& x, const stan::math::var& y, lensvector<stan::math::var>& def_tot, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::deflection(const QScalar& x, const QScalar& y, QScalar& def_tot_x, QScalar& def_tot_y, const int &thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensvector<QScalar> *x_i = &lc.xvals_i[thread];
	lensvector<QScalar> *def = &lc.defs_i[thread];
	lensvector<QScalar> **def_i = &lc.defs_subtot[thread];
	int i,j;
	def_tot_x = 0;
	def_tot_y = 0;
	//std::cout << "n_redshifts=" << n_lens_redshifts << std::endl;
	for (i=0; i < n_lens_redshifts; i++) {
		if (zfacs[i] != 0.0) {
			//std::cout << "redshift " << i << ":\n";
			(*def_i)[i][0] = 0;
			(*def_i)[i][1] = 0;
			(*x_i)[0] = x;
			(*x_i)[1] = y;
			for (j=0; j < i; j++) {
				//std::cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
				(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
				(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
			}
			for (j=0; j < zlens_group_size[i]; j++) {
				lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
				(*def_i)[i][0] += (*def)[0];
				(*def_i)[i][1] += (*def)[1];
			}
			for (j=0; j < n_pixellated_lens; j++) {
				if ((lensgrids[j]->include_in_lensing_calculations) and (pixellated_lens_redshift_idx[j]==i)) {
					//lensgrids[j]->deflection((*x_i)[0],(*x_i)[1],(*def),thread);
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
			}
			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			def_tot_x += (*def_i)[i][0];
			def_tot_y += (*def_i)[i][1];
		}
	}
}
template void QLens::deflection<double>(const double& x, const double& y, double& def_tot_x, double& def_tot_y, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::deflection<stan::math::var>(const stan::math::var& x, const stan::math::var& y, stan::math::var& def_tot_x, stan::math::var& def_tot_y, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::deflection_exclude(const double& x, const double& y, bool* exclude, QScalar& def_tot_x, QScalar& def_tot_y, const int &thread, double* zfacs, double** betafacs)
{
	bool skip_lens_plane = false;
	int skip_i = -1;
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensvector<QScalar> *x_i = &lc.xvals_i[thread];
	lensvector<QScalar> *def = &lc.defs_i[thread];
	lensvector<QScalar> **def_i = &lc.defs_subtot[thread];
	int i,j;
	def_tot_x = 0;
	def_tot_y = 0;

	for (i=0; i < n_lens_redshifts; i++) {
		if ((zlens_group_size[i]==1) and (exclude[zlens_group_lens_indx[i][0]])) {
			skip_lens_plane = true;
			skip_i = i;
			// should allow for multiple redshifts to be excluded...fix later
		}
	}

	//std::cout << "n_redshifts=" << n_lens_redshifts << std::endl;
	for (i=0; i < n_lens_redshifts; i++) {
		//std::cout << "redshift " << i << ":\n";
		if ((!skip_lens_plane) or (skip_i != i)) {
			(*def_i)[i][0] = 0;
			(*def_i)[i][1] = 0;
			(*x_i)[0] = x;
			(*x_i)[1] = y;
			for (j=0; j < i; j++) {
				//std::cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
				if ((!skip_lens_plane) or (skip_i != j)) {
					(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
					(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
				}
			}
			for (j=0; j < zlens_group_size[i]; j++) {
				if (exclude[zlens_group_lens_indx[i][j]]) ;
				else {
					lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
			}
			for (j=0; j < n_pixellated_lens; j++) {
				if ((lensgrids[j]->include_in_lensing_calculations) and (pixellated_lens_redshift_idx[j]==i)) {
					//lensgrids[j]->deflection((*x_i)[0],(*x_i)[1],(*def),thread);
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
			}
			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			def_tot_x += (*def_i)[i][0];
			def_tot_y += (*def_i)[i][1];
		}
	}
}
template void QLens::deflection_exclude<double>(const double& x, const double& y, bool* exclude, double& def_tot_x, double& def_tot_y, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::deflection_exclude<stan::math::var>(const double& x, const double& y, bool* exclude, stan::math::var& def_tot_x, stan::math::var& def_tot_y, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::lens_equation(const lensvector<double>& x, lensvector<QScalar>& f, const int& thread, double *zfacs, double** betafacs)
{
	deflection<QScalar>(x[0],x[1],f,thread,zfacs,betafacs);
	f[0] = source[0] - x[0] + f[0]; // finding root of lens equation, i.e. f(x) = beta - theta + alpha = 0   (where alpha is the deflection)
	f[1] = source[1] - x[1] + f[1];
}
template void QLens::lens_equation<double>(const lensvector<double>& x, lensvector<double>& f, const int& thread, double *zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::lens_equation<stan::math::var>(const lensvector<double>& x, lensvector<stan::math::var>& f, const int& thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::map_to_lens_plane(const int& redshift_i, const QScalar& x, const QScalar& y, lensvector<QScalar>& xi, const int &thread, double* zfacs, double** betafacs)
{
	if (redshift_i >= n_lens_redshifts) die("lens redshift index does not exist");
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensvector<QScalar> *x_i = &lc.xvals_i[thread];
	lensvector<QScalar> *def = &lc.defs_i[thread];
	lensvector<QScalar> **def_i = &lc.defs_subtot[thread];

	int i,j;
	//std::cout << "n_redshifts=" << n_lens_redshifts << std::endl;
	for (i=0; i <= redshift_i; i++) {
		//std::cout << "redshift " << i << ":\n";
		(*def_i)[i][0] = 0;
		(*def_i)[i][1] = 0;
		(*x_i)[0] = x;
		(*x_i)[1] = y;
		for (j=0; j < i; j++) {
			//std::cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
			(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
			(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
		}
		if (i==redshift_i) break;
		for (j=0; j < zlens_group_size[i]; j++) {
			lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
			(*def_i)[i][0] += (*def)[0];
			(*def_i)[i][1] += (*def)[1];
		}
		for (j=0; j < n_pixellated_lens; j++) {
			if ((lensgrids[j]->include_in_lensing_calculations) and (pixellated_lens_redshift_idx[j]==i)) {
				//lensgrids[j]->deflection((*x_i)[0],(*x_i)[1],(*def),thread);
				(*def_i)[i][0] += (*def)[0];
				(*def_i)[i][1] += (*def)[1];
			}
		}
		(*def_i)[i][0] *= zfacs[i];
		(*def_i)[i][1] *= zfacs[i];
	}
	xi[0] = (*x_i)[0];
	xi[1] = (*x_i)[1];
}
template void QLens::map_to_lens_plane<double>(const int& redshift_i, const double& x, const double& y, lensvector<double>& xi, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::map_to_lens_plane<stan::math::var>(const int& redshift_i, const stan::math::var& x, const stan::math::var& y, lensvector<stan::math::var>& xi, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::hessian(const QScalar& x, const QScalar& y, lensmatrix<QScalar>& hess_tot, const int &thread, double* zfacs, double** betafacs) // calculates the Hessian of the lensing potential
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	if (n_lens_redshifts > 1) {
		lensvector<QScalar> *x_i = &lc.xvals_i[thread];
		lensmatrix<QScalar> *A_i = &lc.Amats_i[thread];
		lensvector<QScalar> *def = &lc.defs_i[thread];
		lensvector<QScalar> **def_i = &lc.defs_subtot[thread];
		lensmatrix<QScalar> *hess = &lc.hesses_i[thread];
		lensmatrix<QScalar> **hess_i = &lc.hesses_subtot[thread];

		int i,j;
		hess_tot[0][0] = 0;
		hess_tot[1][1] = 0;
		hess_tot[0][1] = 0;
		hess_tot[1][0] = 0;
		for (i=0; i < n_lens_redshifts; i++) {
			if (zfacs[i] != 0.0) {
				(*hess_i)[i][0][0] = 0;
				(*hess_i)[i][1][1] = 0;
				(*hess_i)[i][0][1] = 0;
				(*hess_i)[i][1][0] = 0;
				(*A_i)[0][0] = 1;
				(*A_i)[1][1] = 1;
				(*A_i)[0][1] = 0;
				(*A_i)[1][0] = 0;
				(*def_i)[i][0] = 0;
				(*def_i)[i][1] = 0;
				(*x_i)[0] = x;
				(*x_i)[1] = y;
				for (j=0; j < i; j++) {
					//std::cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
					(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
					(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
					(*A_i)[0][0] -= (betafacs[i-1][j])*((*hess_i)[j])[0][0];
					(*A_i)[1][1] -= (betafacs[i-1][j])*((*hess_i)[j])[1][1];
					(*A_i)[1][0] -= (betafacs[i-1][j])*((*hess_i)[j])[1][0];
					(*A_i)[0][1] -= (betafacs[i-1][j])*((*hess_i)[j])[0][1];
				}
				for (j=0; j < zlens_group_size[i]; j++) {
					lens_list[zlens_group_lens_indx[i][j]]->potential_derivatives((*x_i)[0],(*x_i)[1],(*def),(*hess));
					(*hess_i)[i][0][0] += (*hess)[0][0];
					(*hess_i)[i][1][1] += (*hess)[1][1];
					(*hess_i)[i][0][1] += (*hess)[0][1];
					(*hess_i)[i][1][0] += (*hess)[1][0];
					if (i < n_lens_redshifts-1) {
						(*def_i)[i][0] += (*def)[0];
						(*def_i)[i][1] += (*def)[1];
					}
				}
				for (j=0; j < n_pixellated_lens; j++) {
					if ((lensgrids[j]->include_in_lensing_calculations) and (pixellated_lens_redshift_idx[j]==i)) {
						//lensgrids[j]->hessian((*x_i)[0],(*x_i)[1],(*hess),thread);
						(*hess_i)[i][0][0] += (*hess)[0][0];
						(*hess_i)[i][1][1] += (*hess)[1][1];
						(*hess_i)[i][0][1] += (*hess)[0][1];
						(*hess_i)[i][1][0] += (*hess)[1][0];
						if (i < n_lens_redshifts-1) {
							//lensgrids[j]->deflection((*x_i)[0],(*x_i)[1],(*def),thread);
							(*def_i)[i][0] += (*def)[0];
							(*def_i)[i][1] += (*def)[1];
						}
					}
				}

				if (i < n_lens_redshifts-1) {
					(*def_i)[i][0] *= zfacs[i];
					(*def_i)[i][1] *= zfacs[i];
				}
				(*hess_i)[i][0][0] *= zfacs[i];
				(*hess_i)[i][1][1] *= zfacs[i];
				(*hess_i)[i][0][1] *= zfacs[i];
				(*hess_i)[i][1][0] *= zfacs[i];

				(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
				(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
				(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
				(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
				(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
				(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];

				hess_tot[0][0] += (*hess_i)[i][0][0];
				hess_tot[1][1] += (*hess_i)[i][1][1];
				hess_tot[1][0] += (*hess_i)[i][1][0];
				hess_tot[0][1] += (*hess_i)[i][0][1];
			}
		}
	} else {
		lensmatrix<QScalar> *hess = &lc.hesses_i[thread];
		int j;
		hess_tot[0][0] = 0;
		hess_tot[1][1] = 0;
		hess_tot[0][1] = 0;
		hess_tot[1][0] = 0;
		for (j=0; j < nlens; j++) {
			lens_list[j]->hessian(x,y,(*hess));
			hess_tot[0][0] += (*hess)[0][0];
			hess_tot[1][1] += (*hess)[1][1];
			hess_tot[0][1] += (*hess)[0][1];
			hess_tot[1][0] += (*hess)[1][0];
		}
		for (j=0; j < n_pixellated_lens; j++) {
			if (lensgrids[j]->include_in_lensing_calculations) {
				//lensgrids[j]->hessian(x,y,(*hess),thread);
				hess_tot[0][0] += (*hess)[0][0];
				hess_tot[1][1] += (*hess)[1][1];
				hess_tot[0][1] += (*hess)[0][1];
				hess_tot[1][0] += (*hess)[1][0];
			}
		}
		//std::cout << "hess_00=" << hess_tot[0][0] << " hess11=" << hess_tot[1][1] << " hess01=" << hess_tot[0][1] << std::endl;
		hess_tot[0][0] *= zfacs[0];
		hess_tot[1][1] *= zfacs[0];
		hess_tot[0][1] *= zfacs[0];
		hess_tot[1][0] *= zfacs[0];
	}
}
template void QLens::hessian<double>(const double& x, const double& y, lensmatrix<double>& hess_tot, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::hessian<stan::math::var>(const stan::math::var& x, const stan::math::var& y, lensmatrix<stan::math::var>& hess_tot, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::hessian_weak(const double& x, const double& y, lensmatrix<QScalar>& hess_tot, const int &thread, double* zfacs) // calculates the Hessian of the lensing potential, but ignores multiplane recursive lensing since it's assumed we're in the weak regime
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *hess = &lc.hesses_i[thread];
	int j;
	hess_tot[0][0] = 0;
	hess_tot[1][1] = 0;
	hess_tot[0][1] = 0;
	hess_tot[1][0] = 0;
	for (j=0; j < nlens; j++) {
		lens_list[j]->hessian(x,y,(*hess));
		hess_tot[0][0] += (*hess)[0][0];
		hess_tot[1][1] += (*hess)[1][1];
		hess_tot[0][1] += (*hess)[0][1];
		hess_tot[1][0] += (*hess)[1][0];
	}
	for (j=0; j < n_pixellated_lens; j++) {
		if (lensgrids[j]->include_in_lensing_calculations) {
			//lensgrids[j]->hessian(x,y,(*hess),thread);
			hess_tot[0][0] += (*hess)[0][0];
			hess_tot[1][1] += (*hess)[1][1];
			hess_tot[0][1] += (*hess)[0][1];
			hess_tot[1][0] += (*hess)[1][0];
		}
	}
	hess_tot[0][0] *= zfacs[0];
	hess_tot[1][1] *= zfacs[0];
	hess_tot[0][1] *= zfacs[0];
	hess_tot[1][0] *= zfacs[0];
}
template void QLens::hessian_weak<double>(const double& x, const double& y, lensmatrix<double>& hess_tot, const int &thread, double* zfacs);
#ifdef USE_STAN
template void QLens::hessian_weak<stan::math::var>(const double& x, const double& y, lensmatrix<stan::math::var>& hess_tot, const int &thread, double* zfacs);
#endif

template<typename QScalar>
void QLens::find_sourcept(const lensvector<double>& x, lensvector<QScalar>& srcpt, const int& thread, double* zfacs, double** betafacs)
{
	deflection<QScalar>(x[0],x[1],srcpt,thread,zfacs,betafacs);
	srcpt[0] = x[0] - srcpt[0]; // this uses the lens equation, beta = theta - alpha (except without defining an intermediate lensvector<double> alpha, which would be an extra memory operation)
	srcpt[1] = x[1] - srcpt[1];
}
template void QLens::find_sourcept<double>(const lensvector<double>& x, lensvector<double>& srcpt, const int& thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::find_sourcept<stan::math::var>(const lensvector<double>& x, lensvector<stan::math::var>& srcpt, const int& thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::find_sourcept(const lensvector<double>& x, QScalar& srcpt_x, QScalar& srcpt_y, const int& thread, double* zfacs, double** betafacs)
{
	deflection<QScalar>(x[0],x[1],srcpt_x,srcpt_y,thread,zfacs,betafacs);
	srcpt_x = x[0] - srcpt_x; // this uses the lens equation, beta = theta - alpha (except without defining an intermediate lensvector<double> alpha, which would be an extra memory operation)
	srcpt_y = x[1] - srcpt_y;
}
template void QLens::find_sourcept<double>(const lensvector<double>& x, double& srcpt_x, double& srcpt_y, const int& thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::find_sourcept<stan::math::var>(const lensvector<double>& x, stan::math::var& srcpt_x, stan::math::var& srcpt_y, const int& thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
QScalar QLens::inverse_magnification(const lensvector<double>& x, const int &thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *jac = &lc.jacs[thread];
	hessian<QScalar>(x[0],x[1],(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	return determinant((*jac));
}
template double QLens::inverse_magnification<double>(const lensvector<double>& x, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template stan::math::var QLens::inverse_magnification<stan::math::var>(const lensvector<double>& x, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
QScalar QLens::magnification(const lensvector<double> &x, const int &thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *jac = &lc.jacs[thread];
	hessian<QScalar>(x[0],x[1],(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	return 1.0/determinant((*jac));
}
template double QLens::magnification<double>(const lensvector<double> &x, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template stan::math::var QLens::magnification<stan::math::var>(const lensvector<double> &x, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
QScalar QLens::shear(const lensvector<QScalar> &x, const int &thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *hess = &lc.hesses[thread];
	hessian<QScalar>(x[0],x[1],(*hess),thread,zfacs,betafacs);
	QScalar shear1, shear2;
	shear1 = 0.5*((*hess)[0][0]-(*hess)[1][1]);
	shear2 = (*hess)[0][1];
	return sqrt(shear1*shear1+shear2*shear2);
}
template double QLens::shear<double>(const lensvector<double> &x, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template stan::math::var QLens::shear<stan::math::var>(const lensvector<stan::math::var> &x, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::shear(const lensvector<QScalar> &x, QScalar& shear_tot, QScalar& angle, const int &thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *hess = &lc.hesses[thread];
	hessian<QScalar>(x[0],x[1],(*hess),thread,zfacs,betafacs);
	QScalar shear1, shear2;
	shear1 = 0.5*((*hess)[0][0]-(*hess)[1][1]);
	shear2 = (*hess)[0][1];
	shear_tot = sqrt(shear1*shear1+shear2*shear2);
	if (shear1==0) {
		if (shear2 > 0) angle = M_HALFPI;
		else angle = -M_HALFPI;
	} else {
		angle = atan(abs(shear2/shear1));
		if (shear1 < 0) {
			if (shear2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (shear2 < 0) {
			angle = -angle;
		}
	}
	angle = 0.5*radians_to_degrees(angle);
}
template void QLens::shear<double>(const lensvector<double> &x, double& shear_tot, double& angle, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::shear<stan::math::var>(const lensvector<stan::math::var> &x, stan::math::var& shear_tot, stan::math::var& angle, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::reduced_shear_components(const lensvector<double> &x, QScalar& g1, QScalar& g2, const int &thread, double* zfacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *hess = &lc.hesses[thread];
	hessian_weak<QScalar>(x[0],x[1],(*hess),thread,zfacs);
	QScalar kap_denom = 1 - ((*hess)[0][0] + (*hess)[1][1])/2;
	g1 = 0.5*((*hess)[0][0]-(*hess)[1][1]) / kap_denom;
	g2 = (*hess)[0][1] / kap_denom;
}
template void QLens::reduced_shear_components<double>(const lensvector<double> &x, double& g1, double& g2, const int &thread, double* zfacs);
#ifdef USE_STAN
template void QLens::reduced_shear_components<stan::math::var>(const lensvector<double> &x, stan::math::var& g1, stan::math::var& g2, const int &thread, double* zfacs);
#endif

// the following functions find the shear, kappa and magnification at the position where a perturber is placed;
// this information is used to determine the optimal subgrid size and resolution

template<typename QScalar>
void QLens::hessian_exclude(const double& x, const double& y, bool* exclude, lensmatrix<QScalar>& hess_tot, const int& thread, double* zfacs, double** betafacs)
{
	bool skip_lens_plane = false;
	int skip_i = -1;
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensvector<QScalar> *x_i = &lc.xvals_i[thread];
	lensmatrix<QScalar> *A_i = &lc.Amats_i[thread];
	lensvector<QScalar> *def = &lc.defs_i[thread];
	lensvector<QScalar> **def_i = &lc.defs_subtot[thread];
	lensmatrix<QScalar> *hess = &lc.hesses_i[thread];
	lensmatrix<QScalar> **hess_i = &lc.hesses_subtot[thread];

	int i,j;
	for (i=0; i < n_lens_redshifts; i++) {
		if ((zlens_group_size[i]==1) and (exclude[zlens_group_lens_indx[i][0]])) {
			skip_lens_plane = true;
			skip_i = i;
			// should allow for multiple redshifts to be excluded...fix later
		}
	}
	hess_tot[0][0] = 0;
	hess_tot[1][1] = 0;
	hess_tot[0][1] = 0;
	hess_tot[1][0] = 0;
	if (n_lens_redshifts > 1) {
		for (i=0; i < n_lens_redshifts; i++) {
			if ((!skip_lens_plane) or (skip_i != i)) {
				(*hess_i)[i][0][0] = 0;
				(*hess_i)[i][1][1] = 0;
				(*hess_i)[i][0][1] = 0;
				(*hess_i)[i][1][0] = 0;
				(*A_i)[0][0] = 1;
				(*A_i)[1][1] = 1;
				(*A_i)[0][1] = 0;
				(*A_i)[1][0] = 0;
				(*def_i)[i][0] = 0;
				(*def_i)[i][1] = 0;
				(*x_i)[0] = x;
				(*x_i)[1] = y;
				for (j=0; j < i; j++) {
					//std::cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << " " << (*def_i)[j][0] << " " << (*def_i)[j][1] << "...\n";
					if ((!skip_lens_plane) or (skip_i != j)) {
						(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
						(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
						(*A_i) -= (betafacs[i-1][j])*((*hess_i)[j]);
					}
				}
				for (j=0; j < zlens_group_size[i]; j++) {
					// if this is only lens in the lens plane, we still want to include in hessian/deflection until the very
					// end when we add up hessian, because we want the nonlinear effects taken into account here
					if (exclude[zlens_group_lens_indx[i][j]]) ;
					else {
						lens_list[zlens_group_lens_indx[i][j]]->hessian((*x_i)[0],(*x_i)[1],(*hess));
						//std::cout << "lens " << zlens_group_lens_indx[i][j] << ", x=" << (*x_i)[0] << ", y=" << (*x_i)[1] << ", hess: " << (*hess)[0][0] << " " << (*hess)[1][1] << " " << (*hess)[0][1] << std::endl;
						(*hess_i)[i][0][0] += (*hess)[0][0];
						(*hess_i)[i][1][1] += (*hess)[1][1];
						(*hess_i)[i][0][1] += (*hess)[0][1];
						(*hess_i)[i][1][0] += (*hess)[1][0];
						if (i < n_lens_redshifts-1) {
							lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
							(*def_i)[i][0] += (*def)[0];
							(*def_i)[i][1] += (*def)[1];
						}
					}
				}
				for (j=0; j < n_pixellated_lens; j++) {
					if ((lensgrids[j]->include_in_lensing_calculations) and (pixellated_lens_redshift_idx[j]==i)) {
						//lensgrids[j]->hessian((*x_i)[0],(*x_i)[1],(*hess),thread);
						(*hess_i)[i][0][0] += (*hess)[0][0];
						(*hess_i)[i][1][1] += (*hess)[1][1];
						(*hess_i)[i][0][1] += (*hess)[0][1];
						(*hess_i)[i][1][0] += (*hess)[1][0];
						if (i < n_lens_redshifts-1) {
							//lensgrids[j]->deflection((*x_i)[0],(*x_i)[1],(*def),thread);
							(*def_i)[i][0] += (*def)[0];
							(*def_i)[i][1] += (*def)[1];
						}
					}
				}
				if (i < n_lens_redshifts-1) {
					(*def_i)[i][0] *= zfacs[i];
					(*def_i)[i][1] *= zfacs[i];
				}
				(*hess_i)[i][0][0] *= zfacs[i];
				(*hess_i)[i][1][1] *= zfacs[i];
				(*hess_i)[i][0][1] *= zfacs[i];
				(*hess_i)[i][1][0] *= zfacs[i];

				//std::cout << "lens plane " << i << ", hess before: " << (*hess_i)[i][0][0] << " " << (*hess_i)[i][1][1] << " " << (*hess_i)[i][0][1] << std::endl;
				(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
				(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
				(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
				(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
				(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
				(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];
				//std::cout << "lens plane " << i << ", hess after: " << (*hess_i)[i][0][0] << " " << (*hess_i)[i][1][1] << " " << (*hess_i)[i][0][1] << std::endl;

				hess_tot += (*hess_i)[i];
			}
		}
	} else {
		if (use_perturber_flags) {
			for (i=0; i < nlens; i++) {
				if ((!exclude[i]) and (lens_list[i]->perturber==false)) {
					lens_list[i]->hessian(x,y,(*hess));
					hess_tot[0][0] += (*hess)[0][0];
					hess_tot[1][1] += (*hess)[1][1];
					hess_tot[0][1] += (*hess)[0][1];
					hess_tot[1][0] += (*hess)[1][0];
				}
			}
		} else {
			for (i=0; i < nlens; i++) {
				if (!exclude[i]) {
					lens_list[i]->hessian(x,y,(*hess));
					hess_tot[0][0] += (*hess)[0][0];
					hess_tot[1][1] += (*hess)[1][1];
					hess_tot[0][1] += (*hess)[0][1];
					hess_tot[1][0] += (*hess)[1][0];
				}
			}
		}
		for (j=0; j < n_pixellated_lens; j++) {
			if (lensgrids[j]->include_in_lensing_calculations) {
				//lensgrids[j]->hessian(x,y,(*hess),thread);
				hess_tot[0][0] += (*hess)[0][0];
				hess_tot[1][1] += (*hess)[1][1];
				hess_tot[0][1] += (*hess)[0][1];
				hess_tot[1][0] += (*hess)[1][0];
			}
		}
		hess_tot[0][0] *= zfacs[0];
		hess_tot[1][1] *= zfacs[0];
		hess_tot[0][1] *= zfacs[0];
		hess_tot[1][0] *= zfacs[0];
	}
}
template void QLens::hessian_exclude<double>(const double& x, const double& y, bool* exclude, lensmatrix<double>& hess_tot, const int& thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::hessian_exclude<stan::math::var>(const double& x, const double& y, bool* exclude, lensmatrix<stan::math::var>& hess_tot, const int& thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
QScalar QLens::magnification_exclude(const lensvector<double> &x, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *jac = &lc.jacs[thread];
	hessian_exclude<QScalar>(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];

	return 1.0/determinant((*jac));
}
template double QLens::magnification_exclude<double>(const lensvector<double> &x, bool* exclude, const int& thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template stan::math::var QLens::magnification_exclude<stan::math::var>(const lensvector<double> &x, bool* exclude, const int& thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
QScalar QLens::inverse_magnification_exclude(const lensvector<double> &x, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *jac = &lc.jacs[thread];
	hessian_exclude<QScalar>(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];

	return determinant((*jac));
}
template double QLens::inverse_magnification_exclude<double>(const lensvector<double> &x, bool* exclude, const int& thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template stan::math::var QLens::inverse_magnification_exclude<stan::math::var>(const lensvector<double> &x, bool* exclude, const int& thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
QScalar QLens::shear_exclude(const lensvector<double> &x, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *jac = &lc.jacs[thread];
	hessian_exclude<QScalar>(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	QScalar shear1, shear2;
	shear1 = 0.5*((*jac)[1][1]-(*jac)[0][0]);
	shear2 = -(*jac)[0][1];
	return sqrt(shear1*shear1+shear2*shear2);
}
template double QLens::shear_exclude<double>(const lensvector<double> &x, bool* exclude, const int& thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template stan::math::var QLens::shear_exclude<stan::math::var>(const lensvector<double> &x, bool* exclude, const int& thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::shear_exclude(const lensvector<double> &x, QScalar &shear, QScalar &angle, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	lensmatrix<QScalar> *jac = &lc.jacs[thread];
	hessian_exclude<QScalar>(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	QScalar shear1, shear2;
	shear1 = 0.5*((*jac)[1][1]-(*jac)[0][0]);
	shear2 = -(*jac)[0][1];
	shear = sqrt(shear1*shear1+shear2*shear2);
	if (shear1==0) {
		if (shear2 > 0) angle = M_HALFPI;
		else angle = -M_HALFPI;
	} else {
		angle = atan(abs(shear2/shear1));
		if (shear1 < 0) {
			if (shear2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (shear2 < 0) {
			angle = -angle;
		}
	}
	angle = 0.5*radians_to_degrees(angle);
}
template void QLens::shear_exclude<double>(const lensvector<double> &x, double &shear, double &angle, bool* exclude, const int& thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::shear_exclude<stan::math::var>(const lensvector<double> &x, stan::math::var &shear, stan::math::var &angle, bool* exclude, const int& thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
QScalar QLens::kappa_exclude(const lensvector<double> &x, bool* exclude, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	QScalar kappa;
	if (n_lens_redshifts==1) {
		int j;
		kappa=0;
		if (use_perturber_flags) {
			for (j=0; j < nlens; j++) {
				if ((!exclude[j]) and (lens_list[j]->perturber==false))
					kappa += lens_list[j]->kappa(x[0],x[1]);
			}
		} else {
			for (j=0; j < nlens; j++) {
				if (!exclude[j])
					kappa += lens_list[j]->kappa(x[0],x[1]);
			}
		}
		for (j=0; j < n_pixellated_lens; j++) {
			//if (lensgrids[j]->include_in_lensing_calculations) kappa += lensgrids[j]->kappa(x[0],x[1],0);
		}
		kappa *= zfacs[0];
	} else {
		lensmatrix<QScalar> *jac = &lc.jacs[0];
		hessian_exclude<QScalar>(x[0],x[1],exclude,(*jac),0,zfacs,betafacs);
		kappa = ((*jac)[0][0] + (*jac)[1][1])/2;
	}
	return kappa;
}
template double QLens::kappa_exclude<double>(const lensvector<double> &x, bool* exclude, double* zfacs, double** betafacs);
#ifdef USE_STAN
template stan::math::var QLens::kappa_exclude<stan::math::var>(const lensvector<double> &x, bool* exclude, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::kappa_inverse_mag_sourcept(const lensvector<QScalar>& xvec, lensvector<QScalar>& srcpt, QScalar &kap_tot, QScalar &invmag, const int &thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	QScalar x = xvec[0], y = xvec[1];
	lensmatrix<QScalar> *jac = &lc.jacs[thread];
	lensvector<QScalar> *def_tot = &lc.defs[thread];

	if (n_lens_redshifts > 1) {
		lensvector<QScalar> *x_i = &lc.xvals_i[thread];
		lensmatrix<QScalar> *A_i = &lc.Amats_i[thread];
		lensvector<QScalar> *def = &lc.defs_i[thread];
		lensvector<QScalar> **def_i = &lc.defs_subtot[thread];
		lensmatrix<QScalar> *hess = &lc.hesses_i[thread];
		lensmatrix<QScalar> **hess_i = &lc.hesses_subtot[thread];

		int i,j;
		(*jac)[0][0] = 0;
		(*jac)[1][1] = 0;
		(*jac)[0][1] = 0;
		(*jac)[1][0] = 0;
		(*def_tot)[0] = 0;
		(*def_tot)[1] = 0;
		for (i=0; i < n_lens_redshifts; i++) {
			(*hess_i)[i][0][0] = 0;
			(*hess_i)[i][1][1] = 0;
			(*hess_i)[i][0][1] = 0;
			(*hess_i)[i][1][0] = 0;
			(*A_i)[0][0] = 1;
			(*A_i)[1][1] = 1;
			(*A_i)[0][1] = 0;
			(*A_i)[1][0] = 0;
			(*def_i)[i][0] = 0;
			(*def_i)[i][1] = 0;
			(*x_i)[0] = x;
			(*x_i)[1] = y;
			for (j=0; j < i; j++) {
				//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
				(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
				(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
				(*A_i)[0][0] -= (betafacs[i-1][j])*((*hess_i)[j])[0][0];
				(*A_i)[1][1] -= (betafacs[i-1][j])*((*hess_i)[j])[1][1];
				(*A_i)[1][0] -= (betafacs[i-1][j])*((*hess_i)[j])[1][0];
				(*A_i)[0][1] -= (betafacs[i-1][j])*((*hess_i)[j])[0][1];
			}
			for (j=0; j < zlens_group_size[i]; j++) {
				lens_list[zlens_group_lens_indx[i][j]]->potential_derivatives((*x_i)[0],(*x_i)[1],(*def),(*hess));
				(*hess_i)[i][0][0] += (*hess)[0][0];
				(*hess_i)[i][1][1] += (*hess)[1][1];
				(*hess_i)[i][0][1] += (*hess)[0][1];
				(*hess_i)[i][1][0] += (*hess)[1][0];
				(*def_i)[i][0] += (*def)[0];
				(*def_i)[i][1] += (*def)[1];
			}
			for (j=0; j < n_pixellated_lens; j++) {
				if ((lensgrids[j]->include_in_lensing_calculations) and (pixellated_lens_redshift_idx[j]==i)) {
					//lensgrids[j]->hessian((*x_i)[0],(*x_i)[1],(*hess),thread);
					(*hess_i)[i][0][0] += (*hess)[0][0];
					(*hess_i)[i][1][1] += (*hess)[1][1];
					(*hess_i)[i][0][1] += (*hess)[0][1];
					(*hess_i)[i][1][0] += (*hess)[1][0];
					if (i < n_lens_redshifts-1) {
						//lensgrids[j]->deflection((*x_i)[0],(*x_i)[1],(*def),thread);
						(*def_i)[i][0] += (*def)[0];
						(*def_i)[i][1] += (*def)[1];
					}
				}
			}

			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			(*def_tot)[0] += (*def_i)[i][0];
			(*def_tot)[1] += (*def_i)[i][1];

			(*hess_i)[i][0][0] *= zfacs[i];
			(*hess_i)[i][1][1] *= zfacs[i];
			(*hess_i)[i][0][1] *= zfacs[i];
			(*hess_i)[i][1][0] *= zfacs[i];

			(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
			(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
			(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
			(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
			(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
			(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];

			(*jac)[0][0] += (*hess_i)[i][0][0];
			(*jac)[1][1] += (*hess_i)[i][1][1];
			(*jac)[1][0] += (*hess_i)[i][1][0];
			(*jac)[0][1] += (*hess_i)[i][0][1];
		}
		kap_tot = ((*jac)[0][0] + (*jac)[1][1])/2;
	} else {
		(*jac)[0][0] = 0;
		(*jac)[1][1] = 0;
		(*jac)[0][1] = 0;
		(*jac)[1][0] = 0;
		(*def_tot)[0] = 0;
		(*def_tot)[1] = 0;
		kap_tot = 0;

		if ((nthreads==1) or (!multithread_perturber_deflections)) {
			int j;
			QScalar kap;
			(*jac)[0][0] = 0;
			(*jac)[1][1] = 0;
			(*jac)[0][1] = 0;
			(*jac)[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;
			kap_tot = 0;
			lensvector<QScalar> *def = &lc.defs_i[0];
			lensmatrix<QScalar> *hess = &lc.hesses_i[0];
			for (j=0; j < nlens; j++) {
				lens_list[j]->kappa_and_potential_derivatives(x,y,kap,(*def),(*hess));
				(*jac)[0][0] += (*hess)[0][0];
				(*jac)[1][1] += (*hess)[1][1];
				(*jac)[0][1] += (*hess)[0][1];
				(*jac)[1][0] += (*hess)[1][0];
				(*def_tot)[0] += (*def)[0];
				(*def_tot)[1] += (*def)[1];
				kap_tot += kap;
			}
			for (j=0; j < n_pixellated_lens; j++) {
				if (lensgrids[j]->include_in_lensing_calculations) {
					//lensgrids[j]->kappa_and_potential_derivatives(x,y,kap,(*def),(*hess),thread);
					(*jac)[0][0] += (*hess)[0][0];
					(*jac)[1][1] += (*hess)[1][1];
					(*jac)[0][1] += (*hess)[0][1];
					(*jac)[1][0] += (*hess)[1][0];
					(*def_tot)[0] += (*def)[0];
					(*def_tot)[1] += (*def)[1];
					kap_tot += kap;
				}
			}
		} else {
			// The following parallel scheme is useful for clusters when LOTS of perturbers are present
			QScalar *hess00 = new QScalar[nthreads];
			QScalar *hess11 = new QScalar[nthreads];
			QScalar *hess01 = new QScalar[nthreads];
			QScalar *def0 = new QScalar[nthreads];
			QScalar *def1 = new QScalar[nthreads];
			QScalar *kapi = new QScalar[nthreads];

			//cout << "Starting new deflection calculation..." << endl << flush;
			#pragma omp parallel
			{
				int thread2;
#ifdef USE_OPENMP
				thread2 = omp_get_thread_num();
#else
				thread2 = 0;
#endif
				lensvector<QScalar> *def = &lc.defs_i[thread2];
				lensmatrix<QScalar> *hess = &lc.hesses_i[thread2];
				//QScalar hess00=0, hess11=0, hess01=0, def0=0, def1=0, kapi=0;
				int j;
				QScalar kap;
				hess00[thread2] = 0;
				hess11[thread2] = 0;
				hess01[thread2] = 0;
				def0[thread2] = 0;
				def1[thread2] = 0;
				kapi[thread2] = 0;

				#pragma omp for schedule(dynamic)
				for (j=0; j < nlens; j++) {
					lens_list[j]->kappa_and_potential_derivatives(x,y,kap,(*def),(*hess));
					hess00[thread2] += (*hess)[0][0];
					hess11[thread2] += (*hess)[1][1];
					hess01[thread2] += (*hess)[0][1];
					def0[thread2] += (*def)[0];
					def1[thread2] += (*def)[1];
					kapi[thread2] += kap;
				}
				//#pragma omp critical
				//{
					//cout << "Thread " << thread2 << " finished" << endl << flush;
				//}
				//#pragma omp critical
				//{
					//(*jac)[0][0] += hess00;
					//(*jac)[1][1] += hess11;
					//(*jac)[0][1] += hess01;
					//(*jac)[1][0] += hess01;
					//(*def_tot)[0] += def0;
					//(*def_tot)[1] += def1;
					//kap_tot += kapi;
				//}
			}
			//cout << "Finished parallel part" << endl << flush;
			for (int j=0; j < nthreads; j++) {
				(*jac)[0][0] += hess00[j];
				(*jac)[1][1] += hess11[j];
				(*jac)[0][1] += hess01[j];
				(*jac)[1][0] += hess01[j];
				(*def_tot)[0] += def0[j];
				(*def_tot)[1] += def1[j];
				kap_tot += kapi[j];
			}
			delete[] hess00;
			delete[] hess11;
			delete[] hess01;
			delete[] def0;
			delete[] def1;
			delete[] kapi;
			if (n_pixellated_lens > 0) {
				QScalar kap;
				lensvector<QScalar> *def = &lc.defs_i[thread];
				lensmatrix<QScalar> *hess = &lc.hesses_i[thread];
				for (int j=0; j < n_pixellated_lens; j++) {
					if (lensgrids[j]->include_in_lensing_calculations) {
						//lensgrids[j]->kappa_and_potential_derivatives(x,y,kap,(*def),(*hess),thread);
						(*jac)[0][0] += (*hess)[0][0];
						(*jac)[1][1] += (*hess)[1][1];
						(*jac)[0][1] += (*hess)[0][1];
						(*jac)[1][0] += (*hess)[1][0];
						(*def_tot)[0] += (*def)[0];
						(*def_tot)[1] += (*def)[1];
						kap_tot += kap;
					}
				}
			}
		}
		//double defx = (*def_tot)[0];
		//double defy = (*def_tot)[1];
		//double jac00 = (*jac)[0][0];

		(*jac)[0][0] *= zfacs[0];
		(*jac)[1][1] *= zfacs[0];
		(*jac)[0][1] *= zfacs[0];
		(*jac)[1][0] *= zfacs[0];
		(*def_tot)[0] *= zfacs[0];
		(*def_tot)[1] *= zfacs[0];
		kap_tot *= zfacs[0];
		//cout << "Finished def calc" << endl << flush;
	}
	srcpt[0] = x - (*def_tot)[0]; // this uses the lens equation, beta = theta - alpha
	srcpt[1] = y - (*def_tot)[1];

	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	invmag = determinant((*jac));
	//cout << "Finished def calc for real; invmag = " << invmag << ", srcpt0=" << srcpt[0] << " srcpt1=" << srcpt[1] << " kap=" << kap_tot << " " << endl << flush;
}
template void QLens::kappa_inverse_mag_sourcept<double>(const lensvector<double>& xvec, lensvector<double>& srcpt, double &kap_tot, double &invmag, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::kappa_inverse_mag_sourcept<stan::math::var>(const lensvector<stan::math::var>& xvec, lensvector<stan::math::var>& srcpt, stan::math::var &kap_tot, stan::math::var &invmag, const int &thread, double* zfacs, double** betafacs);
#endif

template<typename QScalar>
void QLens::sourcept_jacobian(const lensvector<double>& xvec, lensvector<QScalar>& srcpt, lensmatrix<QScalar>& jac_tot, const int &thread, double* zfacs, double** betafacs)
{
	LensCalc<QScalar>& lc = assign_lenscalc_object<QScalar>();
	QScalar x = xvec[0], y = xvec[1];
	lensvector<QScalar> *def_tot = &lc.defs[thread];

	if (n_lens_redshifts > 1) {
		lensvector<QScalar> *x_i = &lc.xvals_i[thread];
		lensmatrix<QScalar> *A_i = &lc.Amats_i[thread];
		lensvector<QScalar> *def = &lc.defs_i[thread];
		lensvector<QScalar> **def_i = &lc.defs_subtot[thread];
		lensmatrix<QScalar> *hess = &lc.hesses_i[thread];
		lensmatrix<QScalar> **hess_i = &lc.hesses_subtot[thread];

		int i,j;
		jac_tot[0][0] = 0;
		jac_tot[1][1] = 0;
		jac_tot[0][1] = 0;
		jac_tot[1][0] = 0;
		(*def_tot)[0] = 0;
		(*def_tot)[1] = 0;
		for (i=0; i < n_lens_redshifts; i++) {
			(*hess_i)[i][0][0] = 0;
			(*hess_i)[i][1][1] = 0;
			(*hess_i)[i][0][1] = 0;
			(*hess_i)[i][1][0] = 0;
			(*A_i)[0][0] = 1;
			(*A_i)[1][1] = 1;
			(*A_i)[0][1] = 0;
			(*A_i)[1][0] = 0;
			(*def_i)[i][0] = 0;
			(*def_i)[i][1] = 0;
			(*x_i)[0] = x;
			(*x_i)[1] = y;
			for (j=0; j < i; j++) {
				//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
				(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
				(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
				(*A_i)[0][0] -= (betafacs[i-1][j])*((*hess_i)[j])[0][0];
				(*A_i)[1][1] -= (betafacs[i-1][j])*((*hess_i)[j])[1][1];
				(*A_i)[1][0] -= (betafacs[i-1][j])*((*hess_i)[j])[1][0];
				(*A_i)[0][1] -= (betafacs[i-1][j])*((*hess_i)[j])[0][1];
			}
			for (j=0; j < zlens_group_size[i]; j++) {
				lens_list[zlens_group_lens_indx[i][j]]->potential_derivatives((*x_i)[0],(*x_i)[1],(*def),(*hess));
				(*hess_i)[i][0][0] += (*hess)[0][0];
				(*hess_i)[i][1][1] += (*hess)[1][1];
				(*hess_i)[i][0][1] += (*hess)[0][1];
				(*hess_i)[i][1][0] += (*hess)[1][0];
				(*def_i)[i][0] += (*def)[0];
				(*def_i)[i][1] += (*def)[1];
			}
			for (j=0; j < n_pixellated_lens; j++) {
				if ((lensgrids[j]->include_in_lensing_calculations) and (pixellated_lens_redshift_idx[j]==i)) {
					//lensgrids[j]->potential_derivatives((*x_i)[0],(*x_i)[1],(*def),(*hess),thread);
					(*hess_i)[i][0][0] += (*hess)[0][0];
					(*hess_i)[i][1][1] += (*hess)[1][1];
					(*hess_i)[i][0][1] += (*hess)[0][1];
					(*hess_i)[i][1][0] += (*hess)[1][0];
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
			}

			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			(*def_tot)[0] += (*def_i)[i][0];
			(*def_tot)[1] += (*def_i)[i][1];

			(*hess_i)[i][0][0] *= zfacs[i];
			(*hess_i)[i][1][1] *= zfacs[i];
			(*hess_i)[i][0][1] *= zfacs[i];
			(*hess_i)[i][1][0] *= zfacs[i];

			(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
			(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
			(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
			(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
			(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
			(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];

			jac_tot[0][0] += (*hess_i)[i][0][0];
			jac_tot[1][1] += (*hess_i)[i][1][1];
			jac_tot[1][0] += (*hess_i)[i][1][0];
			jac_tot[0][1] += (*hess_i)[i][0][1];
		}
	} else {
		jac_tot[0][0] = 0;
		jac_tot[1][1] = 0;
		jac_tot[0][1] = 0;
		jac_tot[1][0] = 0;
		(*def_tot)[0] = 0;
		(*def_tot)[1] = 0;

		if ((nthreads==1) or (!multithread_perturber_deflections)) {
			lensvector<QScalar> *def = &lc.defs_i[0];
			lensmatrix<QScalar> *hess = &lc.hesses_i[0];
			int j;
			jac_tot[0][0] = 0;
			jac_tot[1][1] = 0;
			jac_tot[0][1] = 0;
			jac_tot[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;
			for (j=0; j < nlens; j++) {
				lens_list[j]->potential_derivatives(x,y,(*def),(*hess));
				jac_tot[0][0] += (*hess)[0][0];
				jac_tot[1][1] += (*hess)[1][1];
				jac_tot[0][1] += (*hess)[0][1];
				jac_tot[1][0] += (*hess)[1][0];
				(*def_tot)[0] += (*def)[0];
				(*def_tot)[1] += (*def)[1];
			}
			for (j=0; j < n_pixellated_lens; j++) {
				if (lensgrids[j]->include_in_lensing_calculations) {
					//lensgrids[j]->potential_derivatives(x,y,(*def),(*hess),thread);
					jac_tot[0][0] += (*hess)[0][0];
					jac_tot[1][1] += (*hess)[1][1];
					jac_tot[0][1] += (*hess)[0][1];
					jac_tot[1][0] += (*hess)[1][0];
					(*def_tot)[0] += (*def)[0];
					(*def_tot)[1] += (*def)[1];
				}
			}
		} else {
			// The following parallel scheme is useful for clusters when LOTS of perturbers are present
			QScalar *hess00 = new QScalar[nthreads];
			QScalar *hess11 = new QScalar[nthreads];
			QScalar *hess01 = new QScalar[nthreads];
			QScalar *def0 = new QScalar[nthreads];
			QScalar *def1 = new QScalar[nthreads];

			#pragma omp parallel
			{
				int thread2;
#ifdef USE_OPENMP
				thread2 = omp_get_thread_num();
#else
				thread2 = 0;
#endif
				lensvector<QScalar> *def = &lc.defs_i[thread2];
				lensmatrix<QScalar> *hess = &lc.hesses_i[thread2];
				//QScalar hess00=0, hess11=0, hess01=0, def0=0, def1=0, kapi=0;
				int j;
				//QScalar kap;
					hess00[thread2] = 0;
					hess11[thread2] = 0;
					hess01[thread2] = 0;
					def0[thread2] = 0;
					def1[thread2] = 0;

				#pragma omp for schedule(dynamic)
				for (j=0; j < nlens; j++) {
					lens_list[j]->potential_derivatives(x,y,(*def),(*hess));
					hess00[thread2] += (*hess)[0][0];
					hess11[thread2] += (*hess)[1][1];
					hess01[thread2] += (*hess)[0][1];
					def0[thread2] += (*def)[0];
					def1[thread2] += (*def)[1];
				}
				//#pragma omp critical
				//{
					//jac_tot[0][0] += hess00;
					//jac_tot[1][1] += hess11;
					//jac_tot[0][1] += hess01;
					//jac_tot[1][0] += hess01;
					//(*def_tot)[0] += def0;
					//(*def_tot)[1] += def1;
				//}
			}
			for (int j=0; j < nthreads; j++) {
				jac_tot[0][0] += hess00[j];
				jac_tot[1][1] += hess11[j];
				jac_tot[0][1] += hess01[j];
				jac_tot[1][0] += hess01[j];
				(*def_tot)[0] += def0[j];
				(*def_tot)[1] += def1[j];
			}
			delete[] hess00;
			delete[] hess11;
			delete[] hess01;
			delete[] def0;
			delete[] def1;
			if (n_pixellated_lens > 0) {
				QScalar kap;
				lensvector<QScalar> *def = &lc.defs_i[thread];
				lensmatrix<QScalar> *hess = &lc.hesses_i[thread];
				for (int j=0; j < n_pixellated_lens; j++) {
					if (lensgrids[j]->include_in_lensing_calculations) {
						//lensgrids[j]->potential_derivatives(x,y,(*def),(*hess),thread);
						jac_tot[0][0] += (*hess)[0][0];
						jac_tot[1][1] += (*hess)[1][1];
						jac_tot[0][1] += (*hess)[0][1];
						jac_tot[1][0] += (*hess)[1][0];
						(*def_tot)[0] += (*def)[0];
						(*def_tot)[1] += (*def)[1];
					}
				}
			}
		}
		jac_tot[0][0] *= zfacs[0];
		jac_tot[1][1] *= zfacs[0];
		jac_tot[0][1] *= zfacs[0];
		jac_tot[1][0] *= zfacs[0];
		(*def_tot)[0] *= zfacs[0];
		(*def_tot)[1] *= zfacs[0];
	}
	srcpt[0] = x - (*def_tot)[0]; // this uses the lens equation, beta = theta - alpha
	srcpt[1] = y - (*def_tot)[1];

	jac_tot[0][0] = 1 - jac_tot[0][0];
	jac_tot[1][1] = 1 - jac_tot[1][1];
	jac_tot[0][1] = -jac_tot[0][1];
	jac_tot[1][0] = -jac_tot[1][0];
}
template void QLens::sourcept_jacobian<double>(const lensvector<double>& xvec, lensvector<double>& srcpt, lensmatrix<double>& jac_tot, const int &thread, double* zfacs, double** betafacs);
#ifdef USE_STAN
template void QLens::sourcept_jacobian<stan::math::var>(const lensvector<double>& xvec, lensvector<stan::math::var>& srcpt, lensmatrix<stan::math::var>& jac_tot, const int &thread, double* zfacs, double** betafacs);
#endif


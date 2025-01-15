#include "qlens.h"
#include "mathexpr.h"

double QLens::kappa(const double& x, const double& y, double* zfacs, double** betafacs)
{
	double kappa;
	if (n_lens_redshifts==1) {
		int j;
		kappa=0;
		for (j=0; j < nlens; j++) {
			kappa += lens_list[j]->kappa(x,y);
		}
		kappa *= zfacs[0];
	} else {
		lensmatrix *jac = &jacs[0];
		hessian(x,y,(*jac),0,zfacs,betafacs);
		kappa = ((*jac)[0][0] + (*jac)[1][1])/2;
	}

	return kappa;
}


double QLens::potential(const double& x, const double& y, double* zfacs, double** betafacs)
{
	double pot=0, pot_subtot;
	// This is not really sensical for multiplane lensing, and time delays need to be treated as in Schneider's textbook. Fix later
	int i,j;
	for (i=0; i < n_lens_redshifts; i++) {
		if (zfacs[i] != 0.0) {
			pot_subtot=0;
			for (j=0; j < zlens_group_size[i]; j++) {
				pot_subtot += lens_list[zlens_group_lens_indx[i][j]]->potential(x,y);
			}
			pot += zfacs[i]*pot_subtot;
		}
	}
	return pot;
}

void QLens::deflection(const double& x, const double& y, lensvector& def_tot, const int &thread, double* zfacs, double** betafacs)
{
	lensvector *x_i = &xvals_i[thread];
	lensvector *def = &defs_i[thread];
	lensvector **def_i = &defs_subtot[thread];

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
			//std::cout << "Lens redshift" << i << " (z=" << lens_redshifts[i] << "): xi=" << (*x_i)[0] << " " << (*x_i)[1] << std::endl;
			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			def_tot[0] += (*def_i)[i][0];
			def_tot[1] += (*def_i)[i][1];
		}
	}
}

void QLens::deflection(const double& x, const double& y, double& def_tot_x, double& def_tot_y, const int &thread, double* zfacs, double** betafacs)
{
	lensvector *x_i = &xvals_i[thread];
	lensvector *def = &defs_i[thread];
	lensvector **def_i = &defs_subtot[thread];
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
			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			def_tot_x += (*def_i)[i][0];
			def_tot_y += (*def_i)[i][1];
		}
	}
}

void QLens::deflection_exclude(const double& x, const double& y, bool* exclude, double& def_tot_x, double& def_tot_y, const int &thread, double* zfacs, double** betafacs)
{
	bool skip_lens_plane = false;
	int skip_i = -1;
	lensvector *x_i = &xvals_i[thread];
	lensvector *def = &defs_i[thread];
	lensvector **def_i = &defs_subtot[thread];
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
			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			def_tot_x += (*def_i)[i][0];
			def_tot_y += (*def_i)[i][1];
		}
	}
}

void QLens::lens_equation(const lensvector& x, lensvector& f, const int& thread, double *zfacs, double** betafacs)
{
	deflection(x[0],x[1],f,thread,zfacs,betafacs);
	f[0] = source[0] - x[0] + f[0]; // finding root of lens equation, i.e. f(x) = beta - theta + alpha = 0   (where alpha is the deflection)
	f[1] = source[1] - x[1] + f[1];
}

void QLens::map_to_lens_plane(const int& redshift_i, const double& x, const double& y, lensvector& xi, const int &thread, double* zfacs, double** betafacs)
{
	if (redshift_i >= n_lens_redshifts) die("lens redshift index does not exist");
	lensvector *x_i = &xvals_i[thread];
	lensvector *def = &defs_i[thread];
	lensvector **def_i = &defs_subtot[thread];

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
		(*def_i)[i][0] *= zfacs[i];
		(*def_i)[i][1] *= zfacs[i];
	}
	xi[0] = (*x_i)[0];
	xi[1] = (*x_i)[1];
}

void QLens::hessian(const double& x, const double& y, lensmatrix& hess_tot, const int &thread, double* zfacs, double** betafacs) // calculates the Hessian of the lensing potential
{
	if (n_lens_redshifts > 1) {
		lensvector *x_i = &xvals_i[thread];
		lensmatrix *A_i = &Amats_i[thread];
		lensvector *def = &defs_i[thread];
		lensvector **def_i = &defs_subtot[thread];
		lensmatrix *hess = &hesses_i[thread];
		lensmatrix **hess_i = &hesses_subtot[thread];

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
		lensmatrix *hess = &hesses_i[thread];
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
		hess_tot[0][0] *= zfacs[0];
		hess_tot[1][1] *= zfacs[0];
		hess_tot[0][1] *= zfacs[0];
		hess_tot[1][0] *= zfacs[0];
	}
}

void QLens::hessian_weak(const double& x, const double& y, lensmatrix& hess_tot, const int &thread, double* zfacs) // calculates the Hessian of the lensing potential, but ignores multiplane recursive lensing since it's assumed we're in the weak regime
{
	lensmatrix *hess = &hesses_i[thread];
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
	hess_tot[0][0] *= zfacs[0];
	hess_tot[1][1] *= zfacs[0];
	hess_tot[0][1] *= zfacs[0];
	hess_tot[1][0] *= zfacs[0];
}

void QLens::find_sourcept(const lensvector& x, lensvector& srcpt, const int& thread, double* zfacs, double** betafacs)
{
	deflection(x[0],x[1],srcpt,thread,zfacs,betafacs);
	srcpt[0] = x[0] - srcpt[0]; // this uses the lens equation, beta = theta - alpha (except without defining an intermediate lensvector alpha, which would be an extra memory operation)
	srcpt[1] = x[1] - srcpt[1];
}

void QLens::find_sourcept(const lensvector& x, double& srcpt_x, double& srcpt_y, const int& thread, double* zfacs, double** betafacs)
{
	deflection(x[0],x[1],srcpt_x,srcpt_y,thread,zfacs,betafacs);
	srcpt_x = x[0] - srcpt_x; // this uses the lens equation, beta = theta - alpha (except without defining an intermediate lensvector alpha, which would be an extra memory operation)
	srcpt_y = x[1] - srcpt_y;
}

double QLens::inverse_magnification(const lensvector& x, const int &thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian(x[0],x[1],(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	return determinant((*jac));
}

double QLens::magnification(const lensvector &x, const int &thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian(x[0],x[1],(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	return 1.0/determinant((*jac));
}

double QLens::shear(const lensvector &x, const int &thread, double* zfacs, double** betafacs)
{
	lensmatrix *hess = &hesses[thread];
	hessian(x[0],x[1],(*hess),thread,zfacs,betafacs);
	double shear1, shear2;
	shear1 = 0.5*((*hess)[0][0]-(*hess)[1][1]);
	shear2 = (*hess)[0][1];
	return sqrt(shear1*shear1+shear2*shear2);
}

void QLens::shear(const lensvector &x, double& shear_tot, double& angle, const int &thread, double* zfacs, double** betafacs)
{
	lensmatrix *hess = &hesses[thread];
	hessian(x[0],x[1],(*hess),thread,zfacs,betafacs);
	double shear1, shear2;
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

void QLens::reduced_shear_components(const lensvector &x, double& g1, double& g2, const int &thread, double* zfacs)
{
	lensmatrix *hess = &hesses[thread];
	hessian_weak(x[0],x[1],(*hess),thread,zfacs);
	double kap_denom = 1 - ((*hess)[0][0] + (*hess)[1][1])/2;
	g1 = 0.5*((*hess)[0][0]-(*hess)[1][1]) / kap_denom;
	g2 = (*hess)[0][1] / kap_denom;
}

// the following functions find the shear, kappa and magnification at the position where a perturber is placed;
// this information is used to determine the optimal subgrid size and resolution

void QLens::hessian_exclude(const double& x, const double& y, bool* exclude, lensmatrix& hess_tot, const int& thread, double* zfacs, double** betafacs)
{
	bool skip_lens_plane = false;
	int skip_i = -1;
	lensvector *x_i = &xvals_i[thread];
	lensmatrix *A_i = &Amats_i[thread];
	lensvector *def = &defs_i[thread];
	lensvector **def_i = &defs_subtot[thread];
	lensmatrix *hess = &hesses_i[thread];
	lensmatrix **hess_i = &hesses_subtot[thread];

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
		hess_tot[0][0] *= zfacs[0];
		hess_tot[1][1] *= zfacs[0];
		hess_tot[0][1] *= zfacs[0];
		hess_tot[1][0] *= zfacs[0];
	}
}

double QLens::magnification_exclude(const lensvector &x, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];

	return 1.0/determinant((*jac));
}

double QLens::inverse_magnification_exclude(const lensvector &x, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];

	return determinant((*jac));
}

double QLens::shear_exclude(const lensvector &x, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	double shear1, shear2;
	shear1 = 0.5*((*jac)[1][1]-(*jac)[0][0]);
	shear2 = -(*jac)[0][1];
	return sqrt(shear1*shear1+shear2*shear2);
}

void QLens::shear_exclude(const lensvector &x, double &shear, double &angle, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	double shear1, shear2;
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

double QLens::kappa_exclude(const lensvector &x, bool* exclude, double* zfacs, double** betafacs)
{

	double kappa;
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
		kappa *= zfacs[0];
	} else {
		lensmatrix *jac = &jacs[0];
		hessian_exclude(x[0],x[1],exclude,(*jac),0,zfacs,betafacs);
		kappa = ((*jac)[0][0] + (*jac)[1][1])/2;
	}
	return kappa;
}

void QLens::kappa_inverse_mag_sourcept(const lensvector& xvec, lensvector& srcpt, double &kap_tot, double &invmag, const int &thread, double* zfacs, double** betafacs)
{
	double x = xvec[0], y = xvec[1];
	lensmatrix *jac = &jacs[thread];
	lensvector *def_tot = &defs[thread];

	if (n_lens_redshifts > 1) {
		lensvector *x_i = &xvals_i[thread];
		lensmatrix *A_i = &Amats_i[thread];
		lensvector *def = &defs_i[thread];
		lensvector **def_i = &defs_subtot[thread];
		lensmatrix *hess = &hesses_i[thread];
		lensmatrix **hess_i = &hesses_subtot[thread];

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
			double kap;
			(*jac)[0][0] = 0;
			(*jac)[1][1] = 0;
			(*jac)[0][1] = 0;
			(*jac)[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;
			kap_tot = 0;
			lensvector *def = &defs_i[0];
			lensmatrix *hess = &hesses_i[0];
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
		} else {
			// The following parallel scheme is useful for clusters when LOTS of perturbers are present
			double *hess00 = new double[nthreads];
			double *hess11 = new double[nthreads];
			double *hess01 = new double[nthreads];
			double *def0 = new double[nthreads];
			double *def1 = new double[nthreads];
			double *kapi = new double[nthreads];

			//cout << "Starting new deflection calculation..." << endl << flush;
			#pragma omp parallel
			{
				int thread2;
#ifdef USE_OPENMP
				thread2 = omp_get_thread_num();
#else
				thread2 = 0;
#endif
				lensvector *def = &defs_i[thread2];
				lensmatrix *hess = &hesses_i[thread2];
				//double hess00=0, hess11=0, hess01=0, def0=0, def1=0, kapi=0;
				int j;
				double kap;
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

void QLens::sourcept_jacobian(const lensvector& xvec, lensvector& srcpt, lensmatrix& jac_tot, const int &thread, double* zfacs, double** betafacs)
{
	double x = xvec[0], y = xvec[1];
	lensvector *def_tot = &defs[thread];

	if (n_lens_redshifts > 1) {
		lensvector *x_i = &xvals_i[thread];
		lensmatrix *A_i = &Amats_i[thread];
		lensvector *def = &defs_i[thread];
		lensvector **def_i = &defs_subtot[thread];
		lensmatrix *hess = &hesses_i[thread];
		lensmatrix **hess_i = &hesses_subtot[thread];

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
			lensvector *def = &defs_i[0];
			lensmatrix *hess = &hesses_i[0];
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
		} else {
			// The following parallel scheme is useful for clusters when LOTS of perturbers are present
			double *hess00 = new double[nthreads];
			double *hess11 = new double[nthreads];
			double *hess01 = new double[nthreads];
			double *def0 = new double[nthreads];
			double *def1 = new double[nthreads];

			#pragma omp parallel
			{
				int thread2;
#ifdef USE_OPENMP
				thread2 = omp_get_thread_num();
#else
				thread2 = 0;
#endif
				lensvector *def = &defs_i[thread2];
				lensmatrix *hess = &hesses_i[thread2];
				//double hess00=0, hess11=0, hess01=0, def0=0, def1=0, kapi=0;
				int j;
				//double kap;
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


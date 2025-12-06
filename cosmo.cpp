#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "modelparams.h"
#include "vector.h"
#include "spline.h"
#include "romberg.h"
#include "mathexpr.h"
#include "errors.h"
#include "brent.h"
#include "cosmo.h"
#include "qlens.h"
using namespace std;

const double Cosmology::default_omega_baryon = 0.0488;
const double Cosmology::default_del_R = 2.215;
const double Cosmology::default_k_pivot = 0.05;
const double Cosmology::default_spectral_index = 0.96;
const double Cosmology::default_running = 0;
const double Cosmology::default_n_massive_neutrinos = 1.015; // effective neutrino number; following Planck convention
const double Cosmology::default_neutrino_mass = 0.06; // in eV; this is the minimal mass required to explain neutrino oscillation experiments
const double Cosmology::min_tophat_mass = 1e2;
const double Cosmology::max_tophat_mass = 1e18;
const double Cosmology::default_sigma8 = 0.82; // Only used if normalizing by sigma8 rather than A_s


void CosmologyParams::remove_comments(string& instring)
{
	string instring_copy(instring);
	instring.clear();
	size_t comment_pos = instring_copy.find("#");
	if (comment_pos != string::npos) {
		instring = instring_copy.substr(0,comment_pos);
	} else instring = instring_copy;
}

bool CosmologyParams::datastring_convert(const string& instring, double& outvar)
{
	datastream.clear(); // resets the error flags
	datastream.str(string()); // clears the stringstream
	datastream << instring;
	if (datastream >> outvar) return true;
	else return false;
}

bool CosmologyParams::read_data_line(ifstream& data_infile, vector<string>& datawords, int &n_datawords)
{
	static const int n_characters = 256;
	string word;
	n_datawords = 0;
	datawords.clear();
	do {
		char dataline[n_characters];
		data_infile.getline(dataline,n_characters);
		if ((data_infile.rdstate() & ifstream::eofbit) != 0) return false;
		string linestring(dataline);
		remove_comments(linestring);
		istringstream datastream0(linestring.c_str());
		while (datastream0 >> word) datawords.push_back(word);
		n_datawords = datawords.size();
	} while (n_datawords==0); // skip lines that are blank or only have comments
	return true;
}

bool CosmologyParams::load_params(string filename)
{
	ifstream params_file(filename.c_str());
	int n_datawords;
	vector<string> datawords;

	if (!params_file.is_open()) {
		warn("cosmology parameter file '%s' could not be opened",filename.c_str());
		return false;
	}
	double A_s_num;
	if (read_data_line(params_file,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	if (n_datawords != 1) { warn("input data file has incorrect format; first line should specify Hubble parameter h"); return false; }
	if (datastring_convert(datawords[0],hubble)==false) { warn("data file has incorrect format; could not read Hubble parameter"); return false; }

	if (read_data_line(params_file,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	if (n_datawords != 1) { warn("input data file has incorrect format; second line should specify omega_m parameter"); return false; }
	if (datastring_convert(datawords[0],omega_m)==false) { warn("data file has incorrect format; could not read omega_m parameter"); return false; }

	if (read_data_line(params_file,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	if (n_datawords != 1) { warn("input data file has incorrect format; third line should specify omega_b parameter"); return false; }
	if (datastring_convert(datawords[0],omega_b)==false) { warn("data file has incorrect format; could not read omega_b parameter"); return false; }

	if (read_data_line(params_file,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	if (n_datawords != 1) { warn("input data file has incorrect format; fourth line should specify omega_lambda parameter"); return false; }
	if (datastring_convert(datawords[0],omega_lambda)==false) { warn("data file has incorrect format; could not read omega_lambda parameter"); return false; }

	if (read_data_line(params_file,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	if (n_datawords != 1) { warn("input data file has incorrect format; fifth line should specify A_s parameter"); return false; }
	if (datastring_convert(datawords[0],A_s_num)==false) { warn("data file has incorrect format; could not read A_s parameter"); return false; }

	if (read_data_line(params_file,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	if (n_datawords != 1) { warn("input data file has incorrect format; sixth line should specify spectral index parameter"); return false; }
	if (datastring_convert(datawords[0],spectral_index)==false) { warn("data file has incorrect format; could not read spectral index parameter"); return false; }

	if (read_data_line(params_file,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	if (n_datawords != 1) { warn("input data file has incorrect format; seventh line should specify running of spectral index parameter"); return false; }
	if (datastring_convert(datawords[0],running)==false) { warn("data file has incorrect format; could not read running of spectral index parameter"); return false; }

	if (read_data_line(params_file,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	if (n_datawords != 1) { warn("input data file has incorrect format; eighth line should specify tensor-to-scalar ratio parameter"); return false; }
	if (datastring_convert(datawords[0],tensor_scalar_ratio)==false) { warn("data file has incorrect format; could not read tensor-to-scalar ratio parameter"); return false; }

	A_s = A_s_num * 1e-9;
	return true;
}

/***************************************** Cosmology class *****************************************/

void Cosmology::setup_parameters(const bool initial_setup)
{
	if (initial_setup) {
		// default initial values
		omega_m = 0.3;
		hubble = 0.7;

		setup_parameter_arrays(2);
	} else {
		// always reset the active parameter flags, since the active ones will be determined below
		// NOTE: if (initial_setup==true), active params are reset in setup_parameter_arrays(..) above
		n_active_params = 0;
		for (int i=0; i < n_params; i++) {
			active_params[i] = false; // default
		}
	}

	int indx = 0;

	if (initial_setup) {
		param[indx] = &hubble;
		paramnames[indx] = "hubble"; latex_paramnames[indx] = "H"; latex_param_subscripts[indx] = "0";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	active_params[indx] = true; 
	n_active_params++;
	indx++;

	if (initial_setup) {
		param[indx] = &omega_m;
		paramnames[indx] = "omega_m"; latex_paramnames[indx] = "\\Omega"; latex_param_subscripts[indx] = "M";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	active_params[indx] = true; 
	n_active_params++;
	indx++;
}

void Cosmology::copy_cosmo_data(const Cosmology* cosmo_in)
{
	hubble = cosmo_in->hubble;
	omega_m = cosmo_in->omega_m;
	omega_b = cosmo_in->omega_b;
	A_s = cosmo_in->A_s;
	copy_param_arrays(cosmo_in);
	set_cosmology(omega_m,omega_b,hubble,A_s);
}

void Cosmology::update_meta_parameters(const bool varied_only_fitparams)
{
	set_cosmology(omega_m,omega_b,hubble,A_s);
	if ((qlens) and (n_vary_params > 0)) {
		qlens->update_zfactors_and_betafactors();
		for (int i=0; i < qlens->nlens; i++) {
			qlens->lens_list[i]->update_meta_parameters(); // if the cosmology has changed, update cosmology info and any parameters that depend on them (unless there are anchored parameters, in which case it will be done below)
		}
	}
}

int Cosmology::set_cosmology(double omega_matter, double omega_baryon, double neutrino_mass, double n_massive_neutrinos, double omega_lamb, double hub, double A_s_in, bool normalize_by_sigma8)
{
	omega_m = omega_matter;
	omega_b = omega_baryon;
	omega_lambda = omega_lamb;
	hubble = hub;
	hubble_length = 2.99792458e3/hubble; // in Mpc
	dcrit0 = 2.775e11*hubble*hubble;  // units are solar masses per Mpc^3
	A_s = A_s_in;
	k_pivot = default_k_pivot;

	double omega_hdm = n_massive_neutrinos*neutrino_mass/93.04/SQR(hubble);

	// fitting formula from Carroll, Press & Turner 1992
	double growth_factor = (5.0/2.0)*omega_m/(pow(omega_m, 4.0/7.0) - omega_lambda + (1 + 0.5*omega_m)*(1 + omega_lambda/70.0));

	//double growth_factor = growth_function(1);
	variance_normalization = (4.0/25.0)*A_s*QUARTIC(hubble_length)*SQR(growth_factor/omega_m);
	power_k_normalization = variance_normalization*(2*M_PI*M_PI);
		// Now set up the transfer function. Fitting Formulae are for CDM + Baryon + Massive Neutrino (MDM) cosmologies.
	// Daniel J. Eisenstein & Wayne Hu, Institute for Advanced Study
	
	theta_cmb = 2.7255/2.7;	// Assuming T_cmb = 2.7255 K

	int qwarn = 0;
	 // Look for strange input
	if (omega_baryon<0.0) {
		cerr << "set_cosmology(): Negative omega_baryon set to trace amount.\n";
		qwarn = 1;
	}
	if (omega_hdm<0.0) {
		cerr << "set_cosmology(): Negative omega_hdm set to trace amount.\n";
		qwarn = 1;
	}
	if (hubble<=0.0) {
		cerr <<"set_cosmology(): Negative Hubble constant illegal.\n";
		exit(1);  // Can't recover
	}
	//else if (hubble>2.0) {
		//cerr <<"Warning: Hubble constant should be in units of 100 km/s/Mpc, which means it should not be significantly larger than 1.\n";
		//qwarn = 1;
	//}
	if (n_massive_neutrinos<1) n_massive_neutrinos=1;
	num_degen_hdm = n_massive_neutrinos;	

	if (omega_baryon<=0) die("omega_baryon must be greater than zero");

	omega_curv = 1.0-omega_matter-omega_lambda;
	omhh = omega_matter*SQR(hubble);
	obhh = omega_baryon*SQR(hubble);
	onhh = omega_hdm*SQR(hubble);
	f_baryon = omega_baryon/omega_matter;
	f_hdm = omega_hdm/omega_matter;
	f_cdm = 1.0-f_baryon-f_hdm;
	f_cb = f_cdm+f_baryon;
	f_bnu = f_baryon+f_hdm;

	// Compute the equality scale.
	z_equality = 25000.0*omhh/SQR(SQR(theta_cmb));	// Actually 1+z_eq
	k_equality = 0.0746*omhh/SQR(theta_cmb);

	// Compute the drag epoch and sound horizon
	double z_drag_b1, z_drag_b2;
	z_drag_b1 = 0.313*pow(omhh,-0.419)*(1+0.607*pow(omhh,0.674));
	z_drag_b2 = 0.238*pow(omhh,0.223);
	z_drag = 1291*pow(omhh,0.251)/(1.0+0.659*pow(omhh,0.828))*
		(1.0+z_drag_b1*pow(obhh,z_drag_b2));
	y_drag = z_equality/(1.0+z_drag);

	sound_horizon_fit = 44.5*log(9.83/omhh)/sqrt(1.0+10.0*pow(obhh,0.75));

	// Set up for the free-streaming & infall growth function
	p_c = 0.25*(5.0-sqrt(1+24.0*f_cdm));
	p_cb = 0.25*(5.0-sqrt(1+24.0*f_cb));

	growth_small_k = z_equality*2.5*omega_m / (pow(omega_m,4.0/7.0)-omega_lambda+(1.0+omega_m/2.0)*(1.0+omega_lambda/70.0));
	 
	// Compute small-scale suppression
	alpha_nu = f_cdm/f_cb*(5.0-2.*(p_c+p_cb))/(5.-4.*p_cb)*
		pow(1+y_drag,p_cb-p_c)* (1+f_bnu*(-0.553+0.126*f_bnu*f_bnu))/
		(1-0.193*sqrt(f_hdm*num_degen_hdm)+0.169*f_hdm*pow(num_degen_hdm,0.2))*
		(1+(p_c-p_cb)/2*(1+1/(3.-4.*p_c)/(7.-4.*p_cb))/(1+y_drag));
	alpha_gamma = sqrt(alpha_nu);
	beta_c = 1/(1-0.949*f_bnu);

	if (normalize_by_sigma8) {
		double sig8 = rms_sigma8();
		variance_normalization *= SQR(default_sigma8/sig8); // enforce sigma8 = 0.8 as temporary fix... need to understand why A_s giving crazy sigma8
		power_k_normalization = variance_normalization*(2*M_PI*M_PI);
	}

	 spline_comoving_distance();
	 //rms_tophat_spline();

	 return qwarn;
}

double Cosmology::transfer_function(double kk)
{
	// Fitting Formulae for CDM + Baryon + Massive Neutrino (MDM) cosmologies.
	// Daniel J. Eisenstein & Wayne Hu, Institute for Advanced Study
	// 
	// Given a wavenumber in Mpc^-1, return the transfer function for the cosmology held in the global variables.
	// The following are set:
	// growth_cb -- the transfer function for density-weighted CDM + Baryon perturbations. 
	// growth_cbnu -- the transfer function for density-weighted CDM + Baryon + Massive Neutrino perturbations. */

	 double tf_sup_L, tf_sup_C;
	 double temp1, temp2;

	 qq = kk/omhh*SQR(theta_cmb);
	 //cout << kk << " " << omhh << " " << theta_cmb << " " << qq << endl;

	 // Compute the scale-dependent growth functions
	 y_freestream = 17.2*f_hdm*(1+0.488*pow(f_hdm,-7.0/6.0))*SQR(num_degen_hdm*qq/f_hdm);
	 //cout << f_hdm << " " << num_degen_hdm << " " << qq << " " << theta_cmb << " " << omhh << " " << kk << endl;
	 temp1 = pow(growth_small_k, 1.0-p_cb);
	 temp2 = pow(growth_small_k/(1+y_freestream),0.7);
	 growth_cb = pow(1.0+temp2, p_cb/0.7)*temp1;
	 growth_cbnu = pow(pow(f_cb,0.7/p_cb)+temp2, p_cb/0.7)*temp1;

	 // Compute the master function
	 gamma_eff =omhh*(alpha_gamma+(1-alpha_gamma)/(1+SQR(SQR(kk*sound_horizon_fit*0.43))));
	 qq_eff = qq*omhh/gamma_eff;

	 tf_sup_L = log(EULER+1.84*beta_c*alpha_gamma*qq_eff);
	 tf_sup_C = 14.4+325/(1+60.5*pow(qq_eff,1.11));
	 tf_sup = tf_sup_L/(tf_sup_L+tf_sup_C*SQR(qq_eff));

	 qq_nu = 3.92*qq*sqrt(num_degen_hdm/f_hdm);
	 max_fs_correction = 1+1.2*pow(f_hdm,0.64)*pow(num_degen_hdm,0.3+0.6*f_hdm)/(pow(qq_nu,-1.6)+pow(qq_nu,0.8));
	 tf_master = tf_sup*max_fs_correction;

	 // Now compute the CDM+HDM+baryon transfer functions
	 tf_cb = tf_master*growth_cb/growth_small_k;
	 tf_cbnu = tf_master*growth_cbnu/growth_small_k;
	 //cout << growth_small_k << " " << y_freestream << " " << growth_cb << " " << growth_cbnu << " " << alpha_nu << " " << alpha_gamma << " " << gamma_eff << " " << qq_eff << " " << tf_sup_L << " " << tf_sup_C << endl;
	 return tf_cb;
}

double Cosmology::transfer_function(double kk, double zz)
{
	// Fitting Formulae for CDM + Baryon + Massive Neutrino (MDM) cosmologies.
	// Daniel J. Eisenstein & Wayne Hu, Institute for Advanced Study
	// 
	// Given a wavenumber in Mpc^-1, return the transfer function for the cosmology held in the global variables.
	// The following are set:
	// growth_cb -- the transfer function for density-weighted CDM + Baryon perturbations. 
	// growth_cbnu -- the transfer function for density-weighted CDM + Baryon + Massive Neutrino perturbations. */

	double omega_lambda_z, omega_matter_z;
	double omega_denom = omega_lambda+SQR(1.0+zz)*(omega_curv+omega_m*(1.0+zz));
	omega_lambda_z = omega_lambda/omega_denom;
	omega_matter_z = omega_m*SQR(1.0+zz)*(1.0+zz)/omega_denom;
	double growth_small_k_z = z_equality/(1.0+zz)*2.5*omega_matter_z / (pow(omega_matter_z,4.0/7.0)-omega_lambda_z+(1.0+omega_matter_z/2.0)*(1.0+omega_lambda_z/70.0));
	//growth_z_to_z0 = z_equality*2.5*omega_m/(pow(omega_m,4.0/7.0) - omega_lambda + (1.0+omega_m/2.0)*(1.0+omega_lambda/70.0));
	//growth_z_to_z0 = growth_small_k/growth_z_to_z0;	

	 double tf_sup_L, tf_sup_C;
	 double temp1, temp2;

	 qq = kk/omhh*SQR(theta_cmb);

	 // Compute the scale-dependent growth functions
	 y_freestream = 17.2*f_hdm*(1+0.488*pow(f_hdm,-7.0/6.0))*SQR(num_degen_hdm*qq/f_hdm);
	 temp1 = pow(growth_small_k_z, 1.0-p_cb);
	 temp2 = pow(growth_small_k_z/(1+y_freestream),0.7);
	 growth_cb = pow(1.0+temp2, p_cb/0.7)*temp1;
	 growth_cbnu = pow(pow(f_cb,0.7/p_cb)+temp2, p_cb/0.7)*temp1;

	 // Compute the master function
	 gamma_eff =omhh*(alpha_gamma+(1-alpha_gamma)/(1+SQR(SQR(kk*sound_horizon_fit*0.43))));
	 qq_eff = qq*omhh/gamma_eff;

	 tf_sup_L = log(EULER+1.84*beta_c*alpha_gamma*qq_eff);
	 tf_sup_C = 14.4+325/(1+60.5*pow(qq_eff,1.11));
	 tf_sup = tf_sup_L/(tf_sup_L+tf_sup_C*SQR(qq_eff));

	 qq_nu = 3.92*qq*sqrt(num_degen_hdm/f_hdm);
	 max_fs_correction = 1+1.2*pow(f_hdm,0.64)*pow(num_degen_hdm,0.3+0.6*f_hdm)/(pow(qq_nu,-1.6)+pow(qq_nu,0.8));
	 tf_master = tf_sup*max_fs_correction;

	 // Now compute the CDM+HDM+baryon transfer functions
	 tf_cb = tf_master*growth_cb/growth_small_k_z;
	 tf_cbnu = tf_master*growth_cbnu/growth_small_k_z;
	 return tf_cb;
}

double Cosmology::growth_function(double a)
{
	double (Romberg::*growth_ptr)(const double);
	growth_ptr = static_cast<double (Romberg::*)(const double)> (&Cosmology::growth_function_integrand);
	double integral = romberg_open(growth_ptr,0,a,1e-6,5);
	return 2.5*omega_m*h_over_h0(a)*integral;
}

double Cosmology::growth_function_integrand(double a)
{
	return 1.0/CUBE(a*h_over_h0(a));
}

double Cosmology::scaled_curvature_perturbation(double logkappa)
{
	return pow(exp(logkappa),ns-1 + 0.5*running*logkappa);
}

double Cosmology::curvature_power_spectrum(double k)
{
	double logkappa = log(k/k_pivot);
	return A_s*scaled_curvature_perturbation(logkappa);
}

double Cosmology::matter_power_spectrum(double k)
{
	double logkappa = log(k/k_pivot);
	return power_k_normalization*k*SQR(transfer_function(k))*scaled_curvature_perturbation(logkappa);
}

double Cosmology::matter_power_spectrum(double k, double z)
{
	double logkappa = log(k/k_pivot);
	double redshift_factor = growth_function(1.0/(1+z))/growth_factor;
	return power_k_normalization*SQR(redshift_factor)*k*SQR(transfer_function(k))*scaled_curvature_perturbation(logkappa);
}

double Cosmology::variance(double k)
{
	double logkappa = log(k/k_pivot);
	return variance_normalization*QUARTIC(k)*SQR(transfer_function(k))*scaled_curvature_perturbation(logkappa);
}

double Cosmology::variance(double k, double z)
{
	double logkappa = log(k/k_pivot);
	double redshift_factor = growth_function(1.0/(1+z))/growth_factor;
	return variance_normalization*SQR(redshift_factor)*QUARTIC(k)*SQR(transfer_function(k))*scaled_curvature_perturbation(logkappa);
}

double Cosmology::rms_sigma_tophat(const double mass, const double z) // dM/M: rms mass fluctuation of CDM halos
{
	double mass_fluctuation;
	tophat_window_R = pow(3 * mass / (M_4PI*omega_m*dcrit0*CUBE(1+z)), (1.0/3.0)); // unit of length is Mpc

	//tophat_window_R = 8.0/hubble;
	double (Romberg::*tophat_ptr)(const double);
	tophat_ptr = static_cast<double (Romberg::*)(const double)> (&Cosmology::tophat_window_k);
	mass_fluctuation = romberg_open(tophat_ptr, 0, 10, 1.0e-6, 5) + romberg_improper(tophat_ptr, 10, 1e30, 1.0e-6, 5);
	return sqrt(mass_fluctuation);
}

double Cosmology::rms_sigma8() // dM/M: rms mass fluctuation of CDM halos
{
	double mass_fluctuation;
	tophat_window_R = 8.0/hubble;
	double (Romberg::*tophat_ptr)(const double);
	tophat_ptr = static_cast<double (Romberg::*)(const double)> (&Cosmology::tophat_window_k);
	mass_fluctuation = romberg_open(tophat_ptr, 0, 10, 1.0e-6, 5) + romberg_improper(tophat_ptr, 10, 1e30, 1.0e-6, 5);
	return sqrt(mass_fluctuation);
}

double Cosmology::tophat_window_k(const double k)
{
	double x, W_k; /* W_k here is actually (W_k/V_k)^2 (see fig.9.2, K&T p.332) */
	x = k*tophat_window_R;
	W_k = SQR(sin(x)/(x*x*x) - cos(x)/(x*x));
	return (9.0*variance(k)*W_k/k);
}

void Cosmology::rms_tophat_spline()
{
	const int sigma_nn = 100;
	dvector mass_table(sigma_nn), sigma_table(sigma_nn);
	double mass, mass_step;
	mass_step = pow(max_tophat_mass/min_tophat_mass, 1.0/(sigma_nn-1));
	int i;
	for (i=0, mass = min_tophat_mass; i < sigma_nn; i++, mass *= mass_step) {
		mass_table[i] = mass;
		sigma_table[i] = rms_sigma_tophat(mass, 0); // rms_sigma is evaluated at z=0 for the mass function
	}
	rms_sigma.input(mass_table, sigma_table);
	//inverse_sigma.input(sigma_table, mass_table);
	//rms_sigma.output("tophat.spl");
	//inverse_sigma.output("tophatin.spl");
	return;
}

void Cosmology::spline_comoving_distance(void)
{
	const int zsteps = 200;
	dvector z_table(zsteps), d_table(zsteps);
	int i;
	double z, zmin, zmax, zstep;
	zmin=0, zmax=10, zstep=(zmax-zmin)/(zsteps-1);

	double (Romberg::*comoving_dist_ptr)(const double);
	comoving_dist_ptr = static_cast<double (Romberg::*)(const double)> (&Cosmology::comoving_distance_derivative);
	for (i=0, z=zmin; i < zsteps; i++, z += zstep) {
		z_table[i] = z;
		d_table[i] = romberg(comoving_dist_ptr, 0, z, 1e-6, 5);
	}

	comoving_distance_spline.input(z_table, d_table);
}

void Cosmology::redshift_distribution(void)
{
	const int zsteps = 200;
	int i;
	double z, zmin, zmax, zstep;
	double chi, nz;
	zmin=0, zmax=10, zstep=(zmax-zmin)/(zsteps-1);

	ofstream nzout("nz.dat");
	double (Romberg::*comoving_dist_ptr)(const double);
	comoving_dist_ptr = static_cast<double (Romberg::*)(const double)> (&Cosmology::comoving_distance_derivative);
	for (i=0, z=zmin; i < zsteps; i++, z += zstep) {
		chi = romberg(comoving_dist_ptr, 0, z, 1e-6, 5)/hubble_length;
		nz = chi*chi/pow(1+z,5);
		nzout << z << " " << chi << " " << nz << endl;
	}
}

double Cosmology::comoving_distance_exact(const double z)
{
	double (Romberg::*comoving_dist_ptr)(const double);
	comoving_dist_ptr = static_cast<double (Romberg::*)(const double)> (&Cosmology::comoving_distance_derivative);
	double dist = romberg(comoving_dist_ptr, 0, z, 1e-6, 5);
	return dist;
}

double Cosmology::angular_radius(double chi)
{
	double omega_curv = 1-omega_m-omega_lambda;
	double hubble_radius = hubble_length/sqrt(abs(omega_curv));
	if (omega_curv > 0)
		return hubble_radius*(sinh(chi/hubble_radius));
	else if (omega_curv < 0)
		return hubble_radius*(sin(chi/hubble_radius));
	else return chi;
}

double Cosmology::comoving_distance_derivative(const double z)
{
	return (hubble_length*pow(omega_m*CUBE(1+z)+(1-omega_m-omega_lambda)*SQR(1+z) + omega_lambda, -0.5));
	//return pow(1+z,-1.5);
}

double Cosmology::critical_density(const double z)
{
	return dcrit0*(omega_m*CUBE(1+z)+1-omega_m);
}

void Cosmology::get_halo_parameters_from_rs_ds(const double z, const double rs, const double ds, double &mvir, double &rvir)
{
	static const double virial_ratio = 200.0;
	double (Brent::*croot)(const double);
	croot = static_cast<double (Brent::*)(const double)> (&Cosmology::concentration_root_equation);
	croot_const = virial_ratio*critical_density(z)/(3*ds*1e9);
	double c;
	c = BrentsMethod(croot, 0.01, 1000, 1e-4);
	rvir = c * rs;
	mvir = 4.0*M_PI/3.0*CUBE(rvir)*1e-9*virial_ratio*critical_density(z);
}

double Cosmology::concentration_root_equation(const double c)
{
	return croot_const*c*c*c - log(1+c) + c/(1+c);
}

void Cosmology::get_cored_halo_parameters_from_rs_ds(const double z, const double rs, const double ds, const double beta, double &mvir, double &rvir)
{
	static const double virial_ratio = 200.0;
	double (Brent::*croot)(const double);
	croot = static_cast<double (Brent::*)(const double)> (&Cosmology::cored_concentration_root_equation);
	croot_const = virial_ratio*critical_density(z)*SQR(1-beta)/(3*ds*1e9);
	beta_const = beta;
	double c;
	c = BrentsMethod(croot, 0.01, 1000, 1e-4);
	rvir = c * rs;
	mvir = 4.0*M_PI/3.0*CUBE(rvir)*1e-9*virial_ratio*critical_density(z);
}

double Cosmology::cored_concentration_root_equation(const double c)
{
	if (beta_const==0)
		return croot_const*c*c*c - log(1+c) + c/(1+c);
	else
		return croot_const*c*c*c - (1-2*beta_const)*log(1+c) - beta_const*beta_const*log(1+c/beta_const) + (1-beta_const)*c/(1+c);
}

double Cosmology::time_delay_factor_arcsec(double zl, double zs) // for lensing
{
	double dc_s, dc_l, dc_ls;    // comoving distances
	dc_s = comoving_distance_spline.splint(zs) * 1e-3;   // The 1e3 factor converts from Mpc to Gpc 
	dc_l = comoving_distance_spline.splint(zl) * 1e-3;
	if (dc_l >= dc_s) {
		//warn("source is further away than the lensing object (zlens = %f, zsource = %f)", zl, zs);
		return 0.0;
	}
	dc_ls = dc_s - dc_l;

	// NOTE: although strictly speaking, angular diameter distances should be used, there is an extra (1+z) time delay factor and it
	// works out the same if we just use comoving distances.

	static const double speed_of_light_Gpc_per_day = 8.3998e-13;
	static const double radians_to_arcsec = 2.06265e5; // we are assuming the scaled time delay surface is in units of arcsec^2
	return dc_l*dc_s/(speed_of_light_Gpc_per_day*dc_ls)/SQR(radians_to_arcsec);
}

double Cosmology::time_delay_factor_kpc(double zl, double zs) // for lensing
{
	double kpc_to_arcsec = 206.264806/angular_diameter_distance(zl);
	return (time_delay_factor_arcsec(zl,zs) * SQR(kpc_to_arcsec));
}

double Cosmology::sigma_crit_kpc(double zl, double zs) // for lensing
{
	double dc_s, dc_l, dc_ls;    // comoving distances
	double da_s, da_l, da_ls;    // angular diameter distances
	dc_s = comoving_distance_spline.splint(zs) * 1e-3;   // The 1e3 factor converts from Mpc to Gpc 
	dc_l = comoving_distance_spline.splint(zl) * 1e-3;
	if (dc_l >= dc_s) die("source must be further away than the lensing object (zlens = %f, zsource = %f)", zl, zs);
	dc_ls = dc_s - dc_l;

	da_s = dc_s / (1 + zs);       // convert from comoving to angular diameter distance
	da_l = dc_l / (1 + zl);
	da_ls = dc_ls / (1 + zs);
	
	return 1.66477e9*da_s/(da_l*da_ls); // Units are in solar masses/kpc^2
}

double Cosmology::deflection_scale_factor(double zl, double zs) // for lensing
{
	double dc_s, dc_l, dc_ls;    // comoving distances
	dc_s = comoving_distance_spline.splint(zs) * 1e-3;   // The 1e3 factor converts from Mpc to Gpc 
	dc_l = comoving_distance_spline.splint(zl) * 1e-3;
	if (dc_l >= dc_s) die("source must be further away than the lensing object (zlens = %f, zsource = %f)", zl, zs);
	dc_ls = dc_s - dc_l;

	return dc_s/dc_ls;
}

double Cosmology::kappa_ratio(double zl, double zs, double zs0) // for lensing
{
	double dc_s, dc_l, dc_ls;    // comoving distances
	double da_s, da_ls;    // angular diameter distances
	double dc_s0, dc_ls0;    // comoving distances
	double da_s0, da_ls0;    // angular diameter distances
	dc_s = comoving_distance_spline.splint(zs);
	dc_s0 = comoving_distance_spline.splint(zs0);
	dc_l = comoving_distance_spline.splint(zl);
	if (dc_l >= dc_s) {
		//warn("source is further away than the lensing object (zlens = %f, zsource = %f)", zl, zs);
		return 0.0;
	}
	if (dc_l >= dc_s0) die("reference source must be further away than the lensing object (zlens = %f, zsource = %f)", zl, zs0);
	dc_ls = dc_s - dc_l;
	dc_ls0 = dc_s0 - dc_l;

	da_s = dc_s / (1 + zs);       // convert from comoving to angular diameter distance
	da_s0 = dc_s0 / (1 + zs0);       // convert from comoving to angular diameter distance
	da_ls = dc_ls / (1 + zs);
	da_ls0 = dc_ls0 / (1 + zs0);
	
	return ((da_ls/da_ls0) * (da_s0/da_s));
}

double Cosmology::calculate_beta_factor(double zl1, double zl2, double zs) // for multi-plane lensing
{
	if (zl1 > zl2) die("zl2 must be greater than zl1");
	double dc_l1, dc_l2, dc_s;
	double da_12, da_l2, da_s, da_l1s;
	dc_l1 = comoving_distance_spline.splint(zl1);
	dc_l2 = comoving_distance_spline.splint(zl2);
	dc_s = comoving_distance_spline.splint(zs);
	//if (dc_l1 >= dc_s) die("source must be further away than lens 1 (zlens1 = %f, zsource = %f)", zl1, zs);
	if (dc_l1 >= dc_s) return 0.0;
	//if (dc_l2 >= dc_s) die("source must be further away than lens 2 (zlens2 = %f, zsource = %f)", zl2, zs);
	da_12 = (dc_l2-dc_l1) / (1 + zl2);
	da_l2 = dc_l2 / (1 + zl2);

	//double da_l1 = dc_l1 / (1 + zl1); // test only!! THIS IS WRONG!!!
	//da_12 = da_l2 - da_l1; // test only!! DELETE AFTTER!!!!! IS WRONG!!!

	da_s = dc_s / (1 + zs);
	da_l1s = (dc_s - dc_l1) / (1 + zs);
	//double betafac = ((da_12/da_l2) * (da_s/da_l1s));
	//cout << "BETA: " << betafac << endl;
	return ((da_12/da_l2) * (da_s/da_l1s));
}

double Cosmology::calculate_sigpert_scale_factor(double zl1, double zl2, double zs, double rp, double al, double tp) // for multi-plane lensing perturbations
{
	// NOTE: here, zl2 is the perturber's redshift, z1 is the primary lens redshift
	double dc_l1, dc_l2, dc_s;
	double da_l2, da_l1, da_l1s, da_l2s;
	dc_l1 = comoving_distance_spline.splint(zl1);
	dc_l2 = comoving_distance_spline.splint(zl2);
	dc_s = comoving_distance_spline.splint(zs);
	//if (dc_l1 >= dc_s) die("source must be further away than lens 1 (zlens1 = %f, zsource = %f)", zl1, zs);
	if (dc_l1 >= dc_s) return 0.0;
	//if (dc_l2 >= dc_s) die("source must be further away than lens 2 (zlens2 = %f, zsource = %f)", zl2, zs);
	da_l1 = dc_l1 / (1 + zl1);
	da_l2 = dc_l2 / (1 + zl2);
	da_l1s = (dc_s - dc_l1) / (1 + zs);
	da_l2s = (dc_s - dc_l2) / (1 + zs);
	double fac = da_l1*da_l1s / (da_l2*da_l2s);
	double zfore, zback, beta;
	zfore = dmin(zl1,zl2);
	zback = dmax(zl1,zl2);
	beta = calculate_beta_factor(zfore,zback,zs);
	double term = al*rp/(tp+rp);
	if (zl2 < zl1) term *= 2;
	//cout << "BETA=" << beta << " FAC=" << fac << endl;
	return (fac / (1 - beta*(1-term)));
}

double Cosmology::calculate_menc_scale_factor(double zl1, double zl2, double zs, double rp, double al, double tp) // for multi-plane lensing perturbations
{
	double dc_l1, dc_l2, dc_s;
	double da_l2, da_l1, da_l1s, da_l2s;
	dc_l1 = comoving_distance_spline.splint(zl1);
	dc_l2 = comoving_distance_spline.splint(zl2);
	dc_s = comoving_distance_spline.splint(zs);
	//if (dc_l1 >= dc_s) die("source must be further away than lens 1 (zlens1 = %f, zsource = %f)", zl1, zs);
	if (dc_l1 >= dc_s) return 0.0;
	//if (dc_l2 >= dc_s) die("source must be further away than lens 2 (zlens2 = %f, zsource = %f)", zl2, zs);
	da_l1 = dc_l1 / (1 + zl1);
	da_l2 = dc_l2 / (1 + zl2);
	da_l1s = (dc_s - dc_l1) / (1 + zs);
	da_l2s = (dc_s - dc_l2) / (1 + zs);
	//double fac = da_l1*da_l1s / (da_l2*da_l2s);
	double fac = da_l2*da_l1s / (da_l1*da_l2s); // this is the mass scale factor
	double zfore, zback, beta;
	zfore = dmin(zl1,zl2);
	zback = dmax(zl1,zl2);
	beta = calculate_beta_factor(zfore,zback,zs);
	//cout << "BETA=" << beta << " FAC=" << fac << endl;
	double term = al*rp/(tp+rp);
	if (zl2 < zl1) term *= 2;
	//cout << "BETA=" << beta << " FAC=" << fac << endl;
	return (fac / (1 - beta*(1-term)));
}



double Cosmology::sigma_crit_arcsec(double zl, double zs) // for lensing
{
	double kpc_to_arcsec = 206.2648/angular_diameter_distance(zl);
	return (sigma_crit_kpc(zl,zs) / SQR(kpc_to_arcsec));
}

void Cosmology::plot_power_k(int nsteps, const double log10k_min, const double log10k_max, const string filename)
{
	double power, relative_power, logk_pivot, kappa, logkappa, logkappa_step, k, log10k, logkappa_min, logkappa_max;
	logk_pivot = log(k_pivot);
	// Here, kappa = k/k_pivot, and logkappa - ln(k/k_pivot)
	logkappa_max = ln10*log10k_max - logk_pivot;
	logkappa_min = ln10*log10k_min - logk_pivot;
	logkappa_step = (logkappa_max-logkappa_min)/(nsteps-1);
	int i;

	// now we find amplitude of perturbations when they cross the horizon (just to check)
	//k=hubble_length;
	//logkappa = log(k/k_pivot);
	//double dhsq = variance_normalization*QUARTIC(k)*SQR(transfer_function(k))*pow(k/k_pivot,ns-1);
	//double dh = sqrt(dhsq)*omega_m/growth_factor;
	//cout << dh << endl;
	//die();

	ofstream pout(filename.c_str());
	for (i=0, logkappa=logkappa_min; i < nsteps; i++, logkappa += logkappa_step) {
		kappa = exp(logkappa);
		k = k_pivot*kappa;
		power = CUBE(k)*matter_power_spectrum(k); // plots k^3*P(k)
		relative_power = scaled_curvature_perturbation(logkappa) / pow(kappa,ns-1);
		pout << k << " " << power << " " << relative_power << endl;
	}
}

void Cosmology::plot_primordial_power_spectrum(int nsteps, const double log10k_min, const double log10k_max, const string filename)
{
	double power, relative_power, logk_pivot, kappa, logkappa, logkappa_step, k, log10k, logkappa_min, logkappa_max;
	logk_pivot = log(k_pivot);
	// Here, kappa = k/k_pivot, and logkappa - ln(k/k_pivot)
	logkappa_max = ln10*log10k_max - logk_pivot;
	logkappa_min = ln10*log10k_min - logk_pivot;
	logkappa_step = (logkappa_max-logkappa_min)/(nsteps-1);
	int i;

	// now we find amplitude of perturbations when they cross the horizon (just to check)
	//k=hubble_length;
	//logkappa = log(k/k_pivot);
	//double dhsq = variance_normalization*QUARTIC(k)*SQR(transfer_function(k))*pow(k/k_pivot,ns-1);
	//double dh = sqrt(dhsq)*omega_m/growth_factor;
	//cout << dh << endl;
	//die();

	ofstream pout;
	if (filename != "stdout") pout.open(filename.c_str());
	for (i=0, logkappa=logkappa_min; i < nsteps; i++, logkappa += logkappa_step) {
		kappa = exp(logkappa);
		k = k_pivot*kappa;
		power = curvature_power_spectrum(k); // plots k^3*P(k)
		if (filename=="stdout")
			cout << k << " " << power << endl;
		else
			pout << k << " " << power << endl;
	}
}
void Cosmology::plot_angular_power_spectrum(int nsteps, const double log10k_min, const double log10k_max, const string filename)
{
	double power, relative_power, logk_pivot, kappa, logkappa, logkappa_step, k, log10k, logkappa_min, logkappa_max;
	logk_pivot = log(k_pivot);
	// Here, kappa = k/k_pivot, and logkappa - ln(k/k_pivot)
	logkappa_max = ln10*log10k_max - logk_pivot;
	logkappa_min = ln10*log10k_min - logk_pivot;
	logkappa_step = (logkappa_max-logkappa_min)/(nsteps-1);
	int i;

	ofstream pout;
	if (filename != "stdout") pout.open(filename.c_str());
	double x_c=13100; // comoving distance to last scattering
	double l;
	for (i=0, logkappa=logkappa_min; i < nsteps; i++, logkappa += logkappa_step) {
		kappa = exp(logkappa);
		k = k_pivot*kappa;
		l = x_c*k; //approx formula, more accurate for large k
		power = curvature_power_spectrum(k); // plots k^3*P(k)
		if (filename=="stdout")
			cout << l << " " << power << endl;
		else
			pout << l << " " << power << endl;
	}
}

void Cosmology::plot_mc_relation_dutton_moline(const double z, const double xsub)
{
	// This uses the Dutton mass-concentration relation, with the (1+b*log(xsub)) factor for subhalos introduced
	// by Moline et al where xsub = rsub/rvir; note that xsub=1 is the same as field halos
	int i,n_logm=300;
	double c200, logm, logmi=8.0, logmf=12.0;
	double logmstep = (logmf-logmi)/(n_logm-1);
	for (i=0, logm=logmi; i < n_logm; i++, logm += logmstep) {
		c200 = median_concentration_dutton(pow(10,logm),z);
		if (xsub != 1.0) c200 *= (1-0.54*log(xsub)/ln10);
		cout << logm << " " << log(c200)/ln10 << " " << log(c200/1.66)/ln10 << " " << log(c200*1.66)/ln10 << endl; // the 1.66 is twice the 1-sigma scatter in c200 (1.29^2)
	}
}

double Cosmology::median_concentration_dutton(const double mass, const double z)
{
	double a, b, logc;
	a = 0.52 + (0.901 - 0.520)*exp(-0.617*pow(z,1.21));
	b = -0.101 + 0.026*z;
	//double logccheck = 2.148 - 0.11*log(mass)/ln10;

	logc = a + b*log(mass*hubble*1e-12)/ln10;
	//cout << "LOGC: " << logc << " " << logccheck << endl;
	return pow(10,logc);
}

double Cosmology::median_concentration_bullock(const double mass, const double z)
{
	double ans;
	ans = (9/(1.0 + z))*pow(mass/mstar(0), -0.13); // For c_Bullock we take mstar at z=0, since we know evolution goes like 1/(1+z)
	return ans;
}

double Cosmology::mstar(const double z)
{
	double mstar_root;
	zroot = z;
	double (Brent::*mroot_eq)(const double);
	mroot_eq = static_cast<double (Brent::*)(const double)> (&Cosmology::sigma_root);
	mstar_root = BrentsMethod(mroot_eq, min_tophat_mass, 1e15, 1e-6*min_tophat_mass);
	return mstar_root;
}

double Cosmology::sigma_root(const double mass)
{
	//double ans = (rms_sigma.splint(mass)-delta_z(zroot));
	//cout << mass << " " << ans << endl;
	return (rms_sigma_tophat(mass,0)-delta_z(zroot));
}

double Cosmology::delta_z(const double z)
{
	return 1.686 * (d_plus(0)/d_plus(z)); // delta_c = 1.686
}

double Cosmology::d_plus(const double z)   // empirical fitting function for growth function (oguri p. 112)
{
	double g, omega_m_z, omega_lambda_z;

	omega_m_z = omega_m*CUBE(1+z)/(omega_m*CUBE(1+z)+1-omega_m);
	omega_lambda_z = 1 - omega_m_z;
	g = (5.0/2.0)*omega_m_z*(1.0/(pow(omega_m_z, 4.0/7.0) - omega_lambda_z + (1 + omega_m_z/2)*(1 + omega_lambda_z/70)));
	return g/(1+z);
}

double Cosmology::rms_lsig(const double rad)
{
	double sigma_spline;
	double mass = omega_m*dcrit0*M_PI*(4.0/3.0)*CUBE(rad);
	sigma_spline = rms_sigma.splint(mass);
	return sigma_spline;
}

double Cosmology::mass_function_ST(const double mass, const double z)
{
	double dsigma_dlogm, dr_dm, matter_density, sig, rad, der, nu, ans;
	matter_density = omega_m*dcrit0;
	sig = rms_sigma.splint(mass);
	rad = pow(3 * mass / (M_4PI*omega_m*dcrit0), (1.0/3.0)); // unit of length is Mpc

	dr_dm = -(1.0/3.0)*pow(3/(4*M_PI*omega_m*dcrit0), 1.0/3.0)*pow(mass, -2.0/3.0);  // dr/dm (chain rule)

	// this is d(ln(sigma^-1))/dln(M), which goes roughly like M^(-1/2)
	//der = derivative(rms_lsig, rad, 1e-6);
	double h=1e-6;
	der = (rms_lsig(rad+h) - rms_lsig(rad-h)) / (2*h);
	dsigma_dlogm = (mass/sig)*dr_dm*der;

	nu = (SQR(delta_z(z)) * 0.707) / (sig*sig);
	ans = ((matter_density/(mass*mass)) * 0.322 * dsigma_dlogm * sqrt(2*nu/M_PI) * (1 + pow(nu,-0.3)) * exp(-nu/2));
	return ans;
}


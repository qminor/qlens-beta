// NOTE: Only flat cosmologies are currently supported
#ifndef COSMO_H
#define COSMO_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "spline.h"
#include "romberg.h"
#include "brent.h"
#include "mathexpr.h"

struct CosmologyParams
{
	double hubble;
	double omega_m, omega_b, omega_lambda;
	double A_s;
	double spectral_index, running;
	double tensor_scalar_ratio;
	std::stringstream datastream;

	CosmologyParams() { }
	CosmologyParams(CosmologyParams &cosmo_in)
	{
		tensor_scalar_ratio=cosmo_in.tensor_scalar_ratio;
		hubble=cosmo_in.hubble; omega_m=cosmo_in.omega_m; omega_b=cosmo_in.omega_b; omega_lambda=cosmo_in.omega_lambda;
		A_s=cosmo_in.A_s; spectral_index=cosmo_in.spectral_index; running=cosmo_in.running;
	}
	CosmologyParams(double hub, double omega_matter, double omega_baryon, double omega_l, double del_R, double ns, double alpha, double r)
	{
		hubble=hub; omega_m=omega_matter; omega_b=omega_baryon; omega_lambda=omega_l; A_s = del_R; spectral_index=ns; running=alpha; tensor_scalar_ratio=r;
	}
	CosmologyParams(std::string filename) { load_params(filename); }
	bool load_params(std::string filename);
	bool read_data_line(std::ifstream& data_infile, std::vector<std::string>& datawords, int &n_datawords);
	bool datastring_convert(const std::string& instring, double& outvar);
	void remove_comments(std::string& instring);
};

class Cosmology : public Spline, public Romberg, public Brent
{
	private:
	double omega_m, omega_b, omega_lambda, hubble, hubble_length, growth_factor, dcrit0;
	double power_k_normalization, variance_normalization;
	double A_s; // dimensionless curvature power spectrum at the pivot scale
	double ns, running;
	double zroot;

	double k_pivot; // CMB pivot scale
	static const double default_k_pivot;
	static const double default_spectral_index;
	static const double default_running;
	static const double default_n_massive_neutrinos; // effective neutrino number; following Planck convention
	static const double default_neutrino_mass; // in eV; this is the minimal mass required to explain neutrino oscillation experiments
	static const double min_tophat_mass;
	static const double max_tophat_mass;
	static const double default_sigma8;
	double croot_const, beta_const; // used for solving for the concentration of NFW (or cored NFW) halo using root finder

	private:
	double tophat_window_R;
	Spline comoving_distance_spline;
	Spline rms_sigma;

	// See bottom of this file for a description of the following variables, used in the transfer function
	double alpha_gamma, alpha_nu, beta_c, num_degen_hdm, f_baryon, f_bnu, f_cb, f_cdm, f_hdm, growth_small_k, growth_to_z0, k_equality,
		obhh, omega_curv, omhh, onhh, p_c, p_cb, sound_horizon_fit, theta_cmb, y_drag, z_drag, z_equality;
	double gamma_eff, growth_cb, growth_cbnu, max_fs_correction, qq, qq_eff, qq_nu, tf_master, tf_sup, y_freestream;
	double tf_cb, tf_cbnu;

	public:
	Cosmology() { k_pivot = default_k_pivot; ns = default_spectral_index; running = default_running; }
	int set_cosmology(double omega_matter, double omega_baryon, double neutrino_mass, double degen_hdm, double omega_lamb, double hub, double del_R, bool normalize_by_sigma8);
	Cosmology(CosmologyParams &cosmo) {
		k_pivot = default_k_pivot;
		ns = cosmo.spectral_index;
		running = cosmo.running;
		set_cosmology(cosmo.omega_m,cosmo.omega_b,default_neutrino_mass,default_n_massive_neutrinos,cosmo.omega_lambda,cosmo.hubble,cosmo.A_s,true);
	}
	Cosmology(double omega_matter, double omega_baryon, double hub, double del_R) {
		k_pivot = default_k_pivot; ns = default_spectral_index; running = default_running;
		set_cosmology(omega_matter,omega_baryon,default_neutrino_mass,default_n_massive_neutrinos,1-omega_matter,hub,del_R,true);
	}
	Cosmology(double omega_matter, double omega_baryon, double omega_lamb, double hub, double del_R) {
		k_pivot = default_k_pivot; ns = default_spectral_index; running = default_running;
		set_cosmology(omega_matter,omega_baryon,default_neutrino_mass,default_n_massive_neutrinos,omega_lamb,hub,del_R,true);
	}
	Cosmology(double omega_matter, double omega_baryon, double omega_hdm, int degen_hdm, double omega_lamb, double hub, double del_R) {
		k_pivot = default_k_pivot; ns = default_spectral_index; running = default_running;
		set_cosmology(omega_matter,omega_baryon,omega_hdm,degen_hdm,omega_lamb,hub,del_R,true);
	}
	void set_cosmology(double omega_matter, double omega_baryon, double hub, double del_R) {
	// if this function is called, we assume a flat universe, three massless neutrinos, and find transfer function at redshift zero
		k_pivot = default_k_pivot; ns = default_spectral_index; running = default_running;
		set_cosmology(omega_matter,omega_baryon,default_neutrino_mass,default_n_massive_neutrinos,1-omega_matter,hub,del_R,true);
	}
	void set_cosmology(double omega_matter, double omega_baryon, double omega_lamb, double hub, double del_R) {
		// similar to above, except universe is not necessarily flat
		k_pivot = default_k_pivot; ns = default_spectral_index; running = default_running;
		set_cosmology(omega_matter,omega_baryon,default_neutrino_mass,default_n_massive_neutrinos,omega_lamb,hub,del_R,true);
	}
	void set_cosmology(CosmologyParams &cosmo) {
		k_pivot = default_k_pivot; ns = default_spectral_index; running = default_running;
		set_cosmology(cosmo.omega_m,cosmo.omega_b,default_neutrino_mass,default_n_massive_neutrinos,1-cosmo.omega_m,cosmo.hubble,cosmo.A_s,true);
	}

	void set_pivot_scale(double pscale) { k_pivot = pscale; }
	void set_power_spectrum_scale_params(double index, double run) { ns = index; running = run; }

	double transfer_function(double kk);
	double transfer_function(double kk, double zz);

	virtual double scaled_curvature_perturbation(double logkappa);
	double curvature_power_spectrum(double logkappa);

	double matter_power_spectrum(double k);
	double matter_power_spectrum(double k, double z);
	double variance(double k);
	double variance(double k, double z);
	void rms_tophat_spline();
	double rms_sigma_tophat(const double mass, const double z);
	double rms_sigma8();
	void plot_power_k(int nsteps, const double log10k_min, const double log10k_max, const std::string filename);
	void plot_primordial_power_spectrum(int nsteps, const double log10k_min, const double log10k_max, const std::string filename);
	void plot_angular_power_spectrum(int nsteps, const double log10k_min, const double log10k_max, const std::string filename);

	void plot_mc_relation_dutton_moline(const double z, const double xsub);
	double median_concentration_bullock(const double mass, const double z);
	double median_concentration_dutton(const double mass, const double z);
	double mstar(const double z);
	double sigma_root(const double mass);
	double delta_z(const double z);
	double d_plus(const double z);
	double rms_lsig(const double rad);
	double mass_function_ST(const double mass, const double z);


	// note: the following distance functions assume a flat universe
	void spline_comoving_distance(void);
	void redshift_distribution(void);
	double comoving_distance_derivative(const double z);
	double angular_radius(double chi);
	double comoving_distance(const double z) { return (comoving_distance_spline.splint(z)); }
	double angular_diameter_distance(const double z) { return (angular_radius(comoving_distance_spline.splint(z)) / (1+z)); }
	double luminosity_distance(const double z) { return (angular_radius(comoving_distance_spline.splint(z)) * (1+z)); }
	double comoving_distance_exact(const double z);
	double angular_diameter_distance_exact(const double z) { return (angular_radius(comoving_distance_exact(z)) / (1+z)); }
	double luminosity_distance_exact(const double z) { return (angular_radius(comoving_distance_exact(z)) * (1+z)); }

	double critical_density(const double z);
	double matter_density(const double z) { return omega_m*dcrit0*CUBE(1+z); }
	void get_halo_parameters_from_rs_ds(const double z, const double rs, const double ds, double &mvir, double &rvir);
	void get_cored_halo_parameters_from_rs_ds(const double z, const double rs, const double ds, const double beta, double &mvir, double &rvir);
	double cored_concentration_root_equation(const double c);

	double concentration_root_equation(const double c);
	double sigma_crit_kpc(double zl, double zs); // for lensing
	double sigma_crit_arcsec(double zl, double zs); // for lensing
	double time_delay_factor_arcsec(double zl, double zs); // for lensing
	double time_delay_factor_kpc(double zl, double zs); // for lensing
	double deflection_scale_factor(double zl, double zs); // for lensing
	double kappa_ratio(double zl, double zs, double zs0); // for lensing
	double calculate_beta_factor(double zl1, double zl2, double zs); // for multi-plane lensing
	double calculate_sigpert_scale_factor(double zl1, double zl2, double zs, double rp, double al, double tp); // for multi-plane lensing perturbations
	double calculate_menc_scale_factor(double zl1, double zl2, double zs, double rp, double al, double tp); // for multi-plane lensing perturbations

	double growth_function(double a);
	double dt_dz(const double z) { return (hubble_length*pow(omega_m*CUBE(1+z)+1-omega_m, -0.5)/(1+z)); }
	double h_over_h0(const double a) { return sqrt(omega_m/(a*a*a) + (1-omega_m-omega_lambda)/(a*a) + omega_lambda); }

	private:
	double growth_function_integrand(double a);
	double tophat_window_k(double k);
};


/* Description of variables relevant to transfer function:

	alpha_nu,					 The small-scale suppression
	alpha_gamma,				 sqrt(alpha_nu)
	beta_c,						 The correction to the log in the small-scale
	num_degen_hdm,				 Number of degenerate massive neutrino species
	f_baryon,					 Baryon fraction
	f_bnu,						 Baryon + Massive Neutrino fraction
	f_cb,							 Baryon + CDM fraction
	f_cdm,						 CDM fraction
	f_hdm,						 Massive Neutrino fraction
	growth_small_k,					 D_1(z) -- the growth function as k->0
	growth_to_z0,				 D_1(z)/D_1(0) -- the growth relative to z=0
	k_equality,					 The comoving wave number of the horizon at equality
	obhh,							 Omega_baryon * hubble^2
	omega_curv,					 = 1 - omega_matter - omega_lambda
	omhh,							 Omega_matter * hubble^2
	onhh,							 Omega_hdm * hubble^2
	p_c,							 The correction to the exponent before drag epoch
	p_cb,							 The correction to the exponent after drag epoch
	sound_horizon_fit,  		 The sound horizon at the drag epoch
	theta_cmb,					 The temperature of the CMB, in units of 2.7 K
	y_drag,						 Ratio of z_equality to z_drag
	z_drag,						 Redshift of the drag epoch
	z_equality					 Redshift of matter-radiation equality

	 The following are set in transfer_function(k):
	gamma_eff,			 Effective \Gamma
	growth_cb,					 Growth factor for CDM+Baryon perturbations
	growth_cbnu,				 Growth factor for CDM+Baryon+Neutrino pert.
	max_fs_correction,  		 Correction near maximal free streaming
	qq,							 Wavenumber rescaled by \Gamma
	qq_eff,						 Wavenumber rescaled by effective Gamma
	qq_nu,						 Wavenumber compared to maximal free streaming
	tf_master,					 Master TF
	tf_sup,						 Suppressed TF
	y_freestream 				 The epoch of free-streaming for a given scale

	 Finally, transfer_function(k) gives its answers as:
	tf_cb,			 The transfer function for density-weighted CDM + Baryon perturbations. This is returned by transfer_function(k)
	tf_cbnu						 The transfer function for density-weighted CDM + Baryon + Massive Neutrino perturbations. 
*/

#endif

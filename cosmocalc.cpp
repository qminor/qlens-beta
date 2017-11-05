#include "cosmo.h"
#include "errors.h"
#include "mathexpr.h"
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

char *advance(char*);
void usage_error(void);

enum Mode {
	Display_Distances,
	Display_Sigma_Crit,
	Convert_Length_To_Arcsec,
	Convert_Arcsec_To_Length,
	Testing
} mode;

int main(int argc, char *argv[])
{
	int i;

	CosmologyParams cosmology;
	mode = Display_Distances; // default mode

	char params_file[30] = "";
	string cosmology_filename = "planck.csm";
	double redshift=0;
	double redshift_source = 2.0; // for lensing
	double object_size=0;
	bool display_sigma_crit_in_kpc = false; // for lensing
	if (argc==1) usage_error();
	for (i = 1; i < argc; i++)    /* Process command-line arguments */
	{
		if ((*argv[i] == '-') && (isalpha(*(argv[i]+1)))) {
			int c;
			while (c = *++argv[i]) {
				switch (c) {
				case 'z':
					if (sscanf(argv[i], "z%lf", &redshift)==0) die("invalid redshift");
					argv[i] = advance(argv[i]);
					break;
				case 'Z':
					if (sscanf(argv[i], "Z%lf", &redshift_source)==0) die("invalid source redshift");
					argv[i] = advance(argv[i]);
					break;
				case 'l': mode = Display_Sigma_Crit; break;
				case 's':
					if (sscanf(argv[i], "s%lf", &object_size)==0) die("invalid length");
					argv[i] = advance(argv[i]);
					mode = Convert_Length_To_Arcsec;
					break;
				case 'S':
					if (sscanf(argv[i], "S%lf", &object_size)==0) die("invalid angular size");
					argv[i] = advance(argv[i]);
					mode = Convert_Arcsec_To_Length;
					break;
				case 'k': display_sigma_crit_in_kpc = true; break;
				case 'c':
					if (sscanf(argv[i], "c:%s", params_file)==1) {
						argv[i] += (1 + strlen(params_file));
						cosmology_filename.assign(params_file);
						cosmology_filename += ".csm";
					}
					break;
				default: usage_error(); break;
				}
			}
		}
	}
	if (cosmology.load_params(cosmology_filename)==false) die();

	Cosmology cosmo(cosmology);
	if (mode==Display_Distances) {
		double comoving_distance, angular_diameter_distance, luminosity_distance;
		const double Mpc_to_Gpc = 1e-3;
		comoving_distance = Mpc_to_Gpc*cosmo.comoving_distance(redshift);
		angular_diameter_distance = Mpc_to_Gpc*cosmo.angular_diameter_distance(redshift);
		luminosity_distance = Mpc_to_Gpc*cosmo.luminosity_distance(redshift);
		cout << "z = " << redshift << " (assuming flat Universe)" << endl;
		cout << "Comoving distance: " << comoving_distance << " Gpc" << endl;
		cout << "Angular diameter distance: " << angular_diameter_distance << " Gpc" << endl;
		cout << "Luminosity distance: " << luminosity_distance << " Gpc" << endl;
	} else if (mode==Display_Sigma_Crit) {
		cout << "z_lens = " << redshift << ", z_source = " << redshift_source << endl;
		if (redshift==0) cout << "Cannot find sigma_crit if z_lens = 0\n";
		else {
			if (display_sigma_crit_in_kpc) {
				double sigcr = cosmo.sigma_crit_kpc(redshift,redshift_source);
				cout << "sigma_crit = " << sigcr << " solar masses/kpc^2\n";
				double td_factor = cosmo.time_delay_factor_kpc(redshift,redshift_source);
				cout << "time delay factor = " << td_factor << " days/(kpc^2)\n";
			} else {
				double sigcr = cosmo.sigma_crit_arcsec(redshift,redshift_source);
				cout << "sigma_crit = " << sigcr << " solar masses/arcsec^2\n";
				double td_factor = cosmo.time_delay_factor_arcsec(redshift,redshift_source);
				cout << "time delay factor = " << td_factor << " days/(arcsec^2)\n";
			}
		}
	} else if (mode==Convert_Length_To_Arcsec) {
		cout << "z = " << redshift << " (assuming flat Universe)" << endl;
		double kpc_to_arcsec = 1e-3*(180/M_PI)*3600/cosmo.angular_diameter_distance(redshift);
		double angular_size = object_size * kpc_to_arcsec;
		cout << "object size: " << object_size << " kpc\n";
		cout << "angular size: " << angular_size << " arcsec\n";
	} else if (mode==Convert_Arcsec_To_Length) {
		cout << "z = " << redshift << " (assuming flat Universe)" << endl;
		double kpc_to_arcsec = 1e-3*(180/M_PI)*3600/cosmo.angular_diameter_distance(redshift);
		double object_size_kpc = object_size / kpc_to_arcsec;
		cout << "angular size: " << object_size << " arcsec\n";
		cout << "object size: " << object_size_kpc << " kpc\n";
	} else if (mode==Testing) {
	}
	return 0;
}

char *advance(char *p)
{
	while ((*++p) && ((!isalpha(*p)) || (*p=='e'))) /* This advances to the next flag (if there is one) */
		;
	p--;
	return p;
}

void usage_error(void)
{
	cout << "Usage: cosmocalc <-z -Z -d -s -l -k -c>" << endl << endl;
	cout <<  "COSMOLOGY CALCULATOR: displays information relevant to cosmological and lensing calculations.\n"
				"Options:\n"
				"  -z##     Set redshift of object (for lensing, this is redshift of lens; default z=0)\n"
				"  -Z##     Set redshift of source object (for lensing only; default Z=2)\n"
				"  -l       Show lensing information (critical surface mass density and time delay factor)\n"
				"  -k       Set lensing units to kpc (in the lens plane) rather than arcseconds (for -s option)\n"
				"  -s##     Convert physical size of an object (at redshift z) to angular size (in arcsec)\n"
				"  -S##     Convert angular size of an object (at redshift z, in arcsec) to physical size\n"
				"  -c:<file> Load cosmology parameters from input file (default: 'planck.csm')\n\n";
	exit(1);
}

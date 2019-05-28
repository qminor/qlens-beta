#include "qlens.h"
#include "profile.h"
#include "errors.h"
#include "mathexpr.h"
#include "pixelgrid.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <readline/readline.h>
#include <readline/history.h>
#include <unistd.h>
using namespace std;

#define Complain(errmsg) do { if (mpi_id==0) cerr << "Error: " << errmsg << endl; if ((read_from_file) and (quit_after_error)) die(); goto next_line; } while (false) // the while(false) is a trick to make the macro syntax behave like a function
#define display_switch(setting) ((setting) ? "on" : "off")
#define set_switch(setting,setword) do { if ((setword)=="on") setting = true; else if ((setword)=="off") setting = false; else Complain("invalid argument; must specify 'on' or 'off'"); } while (false)
#define LENS_AXIS_DIR ((LensProfile::orient_major_axis_north==true) ? "y-axis" : "x-axis")

void Lens::process_commands(bool read_file)
{
	bool show_cc = true; // if true, plots critical curves along with image positions (via plotlens script)
	bool plot_srcplane = true;
	plot_key_outside = false;
	show_plot_key = true;
	show_colorbar = true;
	colorbar_min = -1e30;
	colorbar_max = 1e30;
	plot_square_axes = false;
	show_imgsrch_grid = false;
	plot_title = "";
	post_title = ""; // Title for posterior triangle plots
	data_info = "";
	chain_info = "";
	param_markers = "";
	n_param_markers = 10000;
	fontsize = 14;
	linewidth = 1;
	string setword; // used for toggling boolean settings

	read_from_file = read_file;
	paused_while_reading_file = false; // used to keep track of whether qlens opened a script while it was paused
	ws = NULL;
	buffer = NULL;

	for (;;)
   {
		next_line:
		if (use_scientific_notation) cout << setiosflags(ios::scientific);
		else {
			cout << resetiosflags(ios::scientific);
			cout.unsetf(ios_base::floatfield);
		}

		if (read_command(true)==false) return;
		if (line.size()==0) continue;

		// if a demo is specified, change command to read the appropriate input files
		if (!read_from_file) {
			if (line=="demo1") { line = "read demo1.in"; cout << "Reading input file 'demo1.in'..." << endl << endl; }
			if (line=="demo2") { line = "read demo2.in"; cout << "Reading input file 'demo2.in'..." << endl << endl; }
			words.clear();
			istringstream linestream(line);
			string word;
			while (linestream >> word) words.push_back(word);
			nwords = words.size();
		}
		remove_equal_sign();

		if ((words[0]=="quit") or (words[0]=="q") or (words[0]=="exit")) return;
		else if (words[0]=="help")
		{
			if (mpi_id==0) {
				if (nwords==1) {
					cout << "Available commands:   (type 'help <command>' for usage information)\n\n"
						"read -- execute commands from input file\n"
						"write -- save command history to text file\n"
						"settings -- display all settings (type 'help settings' for list of settings)\n"
						"lens -- add a lens from the list of lens models (type 'help lens' for list of models)\n"
						"fit -- commands for lens model fitting (type 'help fit' for list of subcommands)\n"
						"imgdata -- commands for loading point image data ('help imgdata' for list of subcommands)\n"
						"wldata -- commands for loading weak lensing data ('help wldata' for list of subcommands)\n"
						"sbmap -- commands for surface brightness maps ('help sbmap' for list of subcommands)\n"
						"source -- add a source from the list of surface brightness models ('help source' for list)\n"
						"cosmology -- display cosmological information, including physical properties of lenses\n"
						"lensinfo -- display kappa, deflection, magnification, shear, and potential at a specific point\n"
						"mass_r -- display both projected (2D) mass and 3D mass enclosed within a given radius\n"
						"plotlensinfo -- plot pixel maps of kappa, deflection, magnification, shear, and potential\n"
						"grid -- specify coordinates of Cartesian grid on image plane\n"
						"autogrid -- determine optimal grid size from critical curves\n"
						"mkgrid -- create new grid for searching for point images\n"
						"findimg -- find set of images corresponding to a given source position\n"
						"plotimg -- find and plot image positions corresponding to a given source position\n"
						"mksrctab -- create file containing Cartesian grid of point sources to be read by 'findimgs'\n"
						"mksrcgal -- create file containing elliptical grid of point sources to be read by 'findimgs'\n"
						"findimgs -- find sets of images from source positions listed in a given input file\n"
						"plotimgs -- find and plot image positions from sources listed in a given input file\n"
						"replotimgs -- replot image positions previously found by 'plotimgs' command\n"
						"plotcrit -- plot critical curves and caustics to data files (or to an image)\n"
						"plotgrid -- plot recursive grid, if exists; plots corners of all grid cells\n"
						"plotlogkappa -- plot log(kappa) as a colormap along with isokappa contours\n"
						"plotlogmag -- plot log(magnification) as a colormap along with magnification contours\n"
						"plotkappa -- plot radial kappa profile for each lens model and the total kappa profile\n"
						"plotmass -- plot radial mass profile\n"
						"defspline -- create a bicubic spline of the deflection field over the range of the grid\n"
						"einstein -- find Einstein radius of a given lens model\n"
						"\n";
						//"FEATURES THAT REQUIRE CCSPLINE MODE:\n"
						//"cc_reset -- delete the current critical curve spline and create a new one\n"              // These are obsolete features, probably should remove
						//"auto_defspline -- spline deflection over an optimized grid determined by critical curves\n" // Although the 'mkrandsrc' and 'printcs' should be generalized
						//"mkrandsrc -- create file containing randomly plotted sources to be read with 'findimgs'\n" // so they don't require ccspline mode (then get rid of ccspline)
						//"printcs -- print total (unbiased) cross section\n\n";
				} else if (words[1]=="settings") {
					cout << "Type 'help <category_name>' to display a list of settings in each category, and 'help <setting>'\n"
						"for a detailed description of each individual setting. Type 'settings' to display all current\n"
						"settings, '<category_name>' to display settings in a specific category, or enter a specific\n"
						"setting name to display its current value.\n"
						"\n"
						"SETTING CATEGORIES:    (type 'help <category_name>' for list of settings in each category)\n"
						"\n"
						"plot_settings -- preferences for graphical plotting\n"
						"imgsrch_settings -- settings for image searching and grid configuration\n"
						"fit_settings -- settings for model fitting, esp. chi-square options and parameter exploration\n"
						"sbmap_settings -- surface brightness maps and source pixel inversion\n"
						"cosmo_settings -- cosmology and redshift settings\n"
						"lens_settings -- settings for lens model parametrizations and tabulated models\n"
						"misc_settings -- miscellaneous settings, including terminal display preferences\n"
						"\n";
				} else if (words[1]=="plot_settings") {
					cout <<
						"terminal -- set file output mode when plotting to files (text, postscript, or PDF)\n"
						"show_cc -- display critical curves when plotting images (on/off)\n"
						"show_srcplane -- show source plane plots when plotting images (on/off)\n"
						"ptsize/ps -- set point size for plotting image positions with 'plotimg'\n"
						"pttype/pt -- set point marker type for plotting image positions with 'plotimg'\n"
						"plot_title -- set title to display on plots\n"
						"plot_square_axes -- enforce x, y axes to follow the same length scale in plots\n"
						"fontsize -- set font size for title and key in plots\n"
						"linewidth -- set line width in plots\n"
						"plot_key -- specify whether to display key in plots (on/off)\n"
						"plot_key_outside -- specify whether to display key inside plots or next to plots (on/off)\n"
						"colorbar -- specify whether to show color bar scale in pixel map plots (on/off)\n"
						"\n";
				} else if (words[1]=="imgsrch_settings") {
					cout << 
						"gridtype -- set grid to radial or Cartesian\n"
						"autogrid_from_Re -- automatically set grid size from Einstein radius of primary lens (on/off)\n"
						"autogrid_before_mkgrid -- automatically set grid size from critical curves using 'autogrid'\n"
						"autocenter -- if on, automatically center grid on the primary lens, as set by 'primary_lens'\n"
						"central_image -- include central images when fitting to data (on/off)\n"
						"time_delays -- calculate time delays for all images (on/off)\n"
						"min_cellsize -- minimum allowed (average) length of subgrid cells used for image searching\n"
						<< ((radial_grid) ? "rsplit -- set initial number of grid rows in the radial direction\n"
						"thetasplit -- set initial number of grid columns in the angular direction\n"
						: "xsplit -- set initial number of grid rows in the x-direction\n"
						"ysplit -- set initial number of grid columns in the y-direction\n") <<
						"cc_splitlevels -- set # of times grid squares are split when containing critical curve\n"
						"cc_split_neighbors -- when splitting cells that contain critical curves, also split neighbors\n"
						"imgpos_accuracy -- required accuracy in the calculated image positions\n"
						"imgsep_threshold -- if image distance to other images < threshold, discard one as 'duplicate'\n"
						"imgsrch_mag_threshold -- warn if images have mag > threshold (or reject if reject_himag = on)\n"
						"reject_himag -- reject images found that have magnification higher than imgsrch_mag_threshold\n"
						"rmin_frac -- set minimum radius of innermost cells in radial grid (fraction of max radius)\n"
						"galsubgrid -- subgrid around perturbing lenses not co-centered with primary lens (on/off)\n"
						"galsub_radius -- scale factor for the optimal radius of perturber subgridding\n"
						"galsub_min_cellsize -- minimum perturber subgrid cell length (units of Einstein radius)\n"
						"galsub_cc_splitlevels -- number of critical curve splittings around perturbing galaxies\n"
						"ccspline -- set critical curve spline mode on/off\n"
						"auto_ccspline -- spline critical curves only if elliptical symmetry is present (on/off)\n"
						"\n";
				} else if (words[1]=="sbmap_settings") {
					cout <<
						"\033[4mSurface brightness pixel map settings\033[0m\n"
						"fits_format -- load surface brightness maps from FITS files (if on) or text files (if off)\n"
						"img_npixels -- set number of pixels for images produced by 'sbmap plotimg'\n"
						"src_npixels -- set # of source grid pixels for plotting or inverting lensed pixel images\n"
						"srcgrid -- set source grid size and location for inverting or plotting lensed pixel images\n"
						"raytrace_method -- set method for ray tracing image pixels to source pixels\n"
						"sim_pixel_noise -- simulated pixel noise added to images produced by 'sbmap plotimg'\n"
						"psf_width -- width of Gaussian point spread function (PSF) along x- and y-axes\n"
						"psf_threshold -- when loading PSF from FITS file, approximate PSF as zero if below threshold\n"
						"psf_mpi -- parallelize PSF convolution using MPI (on/off)\n"
						"\n"
						"\033[4mSource pixel reconstruction settings\033[0m\n"
						"inversion_method -- set method for image matrix inversion (mumps, umfpack, or cg)\n"
						"adaptive_grid -- use adaptive source grid that splits source pixels recursively (on/off)\n"
						"auto_src_npixels -- automatically determine # of source pixels from lens model/data (on/off)\n"
						"auto_srcgrid -- automatically choose source grid size/location from lens model/data (on/off)\n"
						"noise_threshold -- threshold (multiple of pixel noise) for automatic source pixel grid sizing\n"
						"data_pixel_size -- specify the pixel size to assume for pixel data files\n"
						"data_pixel_noise -- pixel noise in data pixel images (loaded using 'sbmap loadimg')\n"
						"inversion_nthreads -- number of OpenMP threads to use specifically for matrix inversion\n"
						"pixel_fraction -- fraction of srcpixels/imgpixels used to determine number of source pixels\n"
						"vary_pixel_fraction -- allow pixel fraction to vary as a free parameter during a fit (on/off)\n"
						"regparam -- value of regularization parameter used for inverting lensed pixel images\n"
						"vary_regparam -- vary the regularization as a free parameter during a fit (on/off)\n"
						"outside_sb_prior -- impose penalty if model produces large surface brightness beyond pixel mask\n"
						"outside_sb_noise_threshold -- max s.b. allowed beyond mask by outside_sb_prior (times pixel noise)\n"
						"nimg_prior -- impose penalty if # of images produced at max surface brightness < nimg_threshold\n"
						"nimg_threshold -- threshold on # of images near max surface brightness (used if nimg_prior is on)\n"
						"nimg_sb_threshold -- for nimg_prior, include only pixels brighter than threshold times max s.b.\n"
						"subhalo_prior -- restrict subhalo position to lie within pixel mask (pjaffe/corecusp only)\n"
						"activate_unmapped_srcpixels -- when inverting, include srcpixels that don't map to any imgpixels\n"
						"exclude_srcpixels_outside_mask -- when inverting, exclude srcpixels that map beyond pixel mask\n"
						"remove_unmapped_subpixels -- when inverting, exclude *sub*pixels that don't map to any imgpixels\n"
						"sb_threshold -- minimum surface brightness to include when determining image centroids\n"
						"parallel_mumps -- run MUMPS matrix inversion using parallel analysis mode (if on)\n"
						"show_mumps_info -- show MUMPS information output after a lensing matrix inversion\n"
						"srcpixel_mag_threshold -- split srcpixels if magnification > threshold (if adaptive_grid=on)\n"
						"vary_srcpixel_mag_threshold -- vary srcpixel_mag_threshold as a free parameter (if on)\n"
						"\n";
				} else if (words[1]=="fit_settings") {
					cout <<
						"\033[4mChi-square function settings\033[0m\n"
						"imgplane_chisq -- use chi-square defined in image plane (if on) or source plane (if off)\n"
						"chisqmag -- use magnification in source plane chi-square function for image positions\n"
						"chisqflux -- include flux information in chi-square fit (on/off)\n"
						"chisq_time_delays -- include time delay information in chi-square fit (on/off)\n"
						"chisq_parity -- include parity information in flux chi-square fit (on/off)\n"
						"analytic_bestfit_src -- find (approx) best-fit source coordinates automatically during fit\n"
						"chisq_mag_threshold -- exclude images from chi-square whose magnification is below threshold\n"
						"chisq_imgsep_threshold -- if any image pairs are closer than threshold, exclude one from chisq\n"
						"chisq_imgplane_threshold -- switch to imgplane_chisq if below threshold (if imgplane_chisq off)\n"
						"nimg_penalty -- penalize chi-square if too many images are produced (if imgplane_chisq on)\n"
						"chisqtol -- chi-square required accuracy during fit\n"
						"srcflux -- flux of point source (for producing or fitting image flux data)\n"
						"fix_srcflux -- fix source flux to specified value during fit rather than find analytically\n"
						"syserr_pos -- Systematic error parameter, added in quadrature to all position errors\n"
						"vary_syserr_pos -- specify whether to vary syserr_pos during a fit (on/off)\n"
						"wl_shearfac -- Weak lensing scale factor parameter, which scales the reduced shear in chi-square\n"
						"vary_wl_shearfac -- specify whether to vary wl_shearfac during a fit (on/off)\n"
						"\n"
						"\033[4mOptimization and Monte Carlo sampler settings\033[0m\n"
						"nrepeat -- number of repeat chi-square optimizations after original run\n"
						"find_errors -- calculate and show marginalized error in each parameter after chi-square fit\n"
						"simplex_nmax -- max number of iterations allowed when using downhill simplex (at temp=0)\n"
						"simplex_nmax_anneal -- number of iterations at given temperature during simulated annealing\n"
						"simplex_minchisq -- downhill simplex finishes immediately if chisq falls below this value\n"
						"simplex_minchisq_anneal -- when annealing, skip to temp=0 if chisq falls below this value\n"
						"simplex_temp0 -- initial annealing temperature for downhill simplex (zero --> no annealing)\n"
						"simplex_tfac -- \"cooling factor\" controls how quickly temp is reduced during annealing\n"
						"simplex_show_bestfit -- show the current best-fit parameters during annealing (if on)\n"
						"data_info -- description of data that is stored in FITS file header and chain file headers\n"
						"chain_info -- description of chain that can be stored using 'fit mkposts' command\n"
						"param_markers -- parameter values to be marked in posteriors plotted by mkdist tool\n"
						"subplot_params -- subset of parameters to make a triangle subplot using 'fit mkposts' command\n"
						"n_livepts -- number of live points used in nested sampling runs\n"
						"polychord_nrepeats -- num_repeats per parameter for PolyChord nested sampler\n"
						"mcmc_chains -- number of chains used in MCMC routines (e.g. T-Walk)\n"
						"mcmctol -- during MCMC, stop chains if Gelman-Rubin R-statistic falls below this threshold\n"
						"mcmclog -- output MCMC convergence, accept ratio etc. to log file while running (if on)\n"
						"random_seed -- random number generator seed for Monte Carlo samplers and simulated annealing\n"
						"chisqlog -- output chi-square, parameter information to log file for each chisq evaluation\n"
						"\n";
				} else if (words[1]=="cosmo_settings") {
					cout <<
						"hubble -- Hubble parameter given by h, where H_0 = 100*h km/s/Mpc (default = 0.7)\n"
						"omega_m -- Matter density today divided by critical density of Universe (default = 0.3)\n"
						"zlens -- default redshift of new lenses that get created\n"
						"zsrc -- redshift of source plane used for 'einstein', findimg', 'plotimg' and 'plotcrit'\n"
						"zsrc_ref -- source redshift used to define lensing quantities (kappa, etc.) (default=zsrc)\n"
						"auto_zsrc_scaling -- if on, automatically set zsrc_ref=zsrc until lens is created (default=on)\n"
						"vary_hubble -- specify whether to vary the Hubble parameter during a fit (on/off)\n"
						"vary_omega_m -- specify whether to vary the omega_m parameter during a fit (on/off)\n"
						"\n";
				} else if (words[1]=="lens_settings") {
					cout << 
						"emode -- controls how ellipticity is introduced into kappa (0,1,2) or potential (3)\n"
						"pmode -- sets the default pmode (for lens models with alternate parametrizations)\n"
						"primary_lens -- sets which lens number is considered the 'primary' (set to 'auto' by default)\n"
						"integral_method -- set numerical integration method (patterson/romberg/gauss)\n"
						"integral_tolerance -- set tolerance for numerical integration (for romberg/patterson)\n"
						"major_axis_along_y -- orient major axis of lenses along y-direction when theta = 0 (on/off)\n"
						"ellipticity_components -- if on, use components of ellipticity e=1-q instead of (q,theta)\n"
						"shear_components -- if on, use components of external shear instead of (shear,theta)\n"
						"tab_rmin -- set minimum radius for interpolation grid in tabulated model\n"
						"tab_r_N -- set number of points along r (spaced logarithmically) for grid in tabulated model\n"
						"tab_phi_N -- set number of points along angular direction for grid in tabulated model\n"
						//"auto_srcgrid_set_pixelsize -- determine size of source pixels based on magnifications\n"  // This is not working well yet
						"\n";
				} else if (words[1]=="misc_settings") {
					cout << 
						"verbal_mode -- if on, display detailed output and input script text to terminal (default=on)\n"
						"warnings -- set warning flags on/off\n"
						"imgsrch_warnings -- set warning flags related to image searching on/off\n"
						"show_wtime -- show time required for executing commands (e.g. mkgrid); requires OpenMP\n"
						"sci_notation -- display numbers in scientific notation (on/off)\n"
						"sim_err_pos -- random error in image positions, added when producing simulated image data\n"
						"sim_err_flux -- random error in image fluxes, added when producing simulated image data\n"
						"sim_err_td -- random error in time delays, added when producing simulated image data\n"
						"\n";
				} else if (words[1]=="read")
					cout << "read <filename>\n\n"
						"Execute commands from file <filename>. After commands have been executed, returns to\n"
						"command prompt mode.\n";
				else if (words[1]=="write")
					cout << "write <filename>\n\n"
						"Save all commands that have been entered to a text file named <filename>.\n";
				else if (words[1]=="grid")
					cout << "grid <xmax> <ymax>\n"
						"grid <xmin> <xmax> <ymin> <ymax>\n"
						"grid center <xc> <yc>\n\n"
						"Sets the grid size for image searching as (xmin,xmax) x (ymin,ymax) or centers the grid on\n"
						"<xc>, <yc> if 'grid center' is specified. If no arguments are given, simply outputs the\n"
						"present grid size.\n";
				else if (words[1]=="srcgrid")
					cout << "srcgrid <xmin> <xmax> <ymin> <ymax>\n\n"
						"Sets the size and location for a pixellated source grid as (xmin,xmax) x (ymin,ymax). Note that\n"
						"if the source grid is changed manually, 'auto_srcgrid' will be automatically turned off.\n"
						"If no arguments are given, simply outputs the present grid size.\n";
				else if (words[1]=="autogrid")
					cout << "autogrid <rmin> <rmax> [gridfrac]\n"
						"autogrid\n\n"
						"Automatically determines the optimal grid size, as long as the critical curves are\n"
						"bracketed by <rmin> and <rmax> (or the default rmin/rmax values if no arguments are\n"
						"given). The optional parameter [gridfrac] specifies the ratio of grid size vs. the\n"
						"radius of the outmost critical curve; this defaults to " << autogrid_frac << ".\n";
				else if (words[1]=="mkgrid")
					cout << "autogrid\n\n"
						"Create grid with the current grid dimensions and splitting parameters.\n";
				else if (words[1]=="einstein")
					cout << "einstein\n"
						"einstein <lens_number>\n\n"
						"Calculates and displays the Einstein radius of the specified lens model. The specified\n"
						"lens <lens_number> should correspond to the number displayed in the list of\n"
						"lens models when the 'lens' command is entered with no arguments.\n"
						"If no lens number is given, calculates the Einstein radius of the primary lens (lens 0)\n"
						"combined with all other lenses that are co-centered with the primary lens.\n";
				else if (words[1]=="major_axis_along_y")
					cout << "major_axis_along_y <on/off>\n\n"
						"Specifies whether to orient the major axis of each lens model along y (if on) or x (if off)\n"
						"for theta=0. (default=on)\n";
				else if (words[1]=="lens") {
					if (nwords==2)
						cout << "lens <lensmodel> <lens_parameter##> ... [z=#] [emode=#] [pmode=#]\n"
							"lens update <lens_number> ...\n"
							"lens clear [lens_number]\n"
							"lens savetab <lens_number> <filename>   (applies to 'tab' models only)\n\n"
							"Creates (or updates) a lens from given model and parameters. If other lenses are present, the new\n"
							"lens will be superimposed with the others. The optional 'z=#' argument sets the lens redshift; if\n"
							"omitted, its redshift is set to the current value of 'zlens'. Likewise, the optional 'emode=#' sets\n"
							"the ellipticity mode for elliptical mass models; if omitted, the default emode is used (see 'help\n"
							"emode' for more info). Some lens models (e.g. nfw) have alternate parametrizations, which can be\n"
							"chosen using 'pmode=#' (see 'help lens ...' for a description of each parametrization mode.)\n"
							"Type 'lens' with no arguments to see a numbered list of the current lens models that have been\n"
							"created, along with their parameter values. To remove a lens or delete the entire configuration,\n"
							"type 'lens clear'. To update parameters of an existing lens, use 'lens update' (see 'help lens\n" 
							"update' for usage info). Finally, it is possible to anchor the parameters of one lens to another\n"
							"lens; type 'help lens anchoring' for info on how to do this.\n\n"
							"Available lens models:   (type 'help lens <lensmodel>' for usage information)\n\n"
							"\033[4mElliptical mass models:\033[0m\n"  // ASCII codes are for underlining
							"alpha -- softened power-law profile\n"
							"pjaffe -- Pseudo-Jaffe profile (smoothly truncated isothermal profile with core)\n"
							"nfw -- NFW model\n"
							"tnfw -- Truncated NFW model\n"
							"cnfw -- NFW model with core\n"
							"hern -- Hernquist model\n"
							"expdisk -- exponential disk\n"
							"sersic -- Sersic profile\n"
							"corecusp -- generalized profile with core, scale radius, inner & outer log-slope\n"
							"kspline -- splined kappa profile (generated from an input file)\n"
							"tab -- generic tabulated model, used to interpolate lensing properties in (r,phi) for above models\n"
							"qtab -- generic tabulated model used to interpolate lensing properties in (r,phi,q) for above models\n"
							"\n"
							"\033[4mNon-elliptical mass models:\033[0m\n"
							"ptmass -- point mass\n"
							"shear -- external shear\n"
							"sheet -- mass sheet\n"
							"mpole -- multipole term in lensing potential\n"
							"kmpole -- multipole term in kappa\n\n";
					else if (words[2]=="clear")
						cout << "lens clear\n"
							"lens clear [lens_number]\n"
							"lens clear [min_lens#]-[max_lens#]\n\n"
							"Removes one (or some subset) of the lens galaxies in the current configuration. If no argument is\n"
							"given, all lenses are removed; if a single argument <lens_number> is given, removes only the lens\n"
							"assigned to the given number in the list of lenses generated by the 'lens' command. To remove more\n"
							"than one lens, you can use a hyphen; for example, 'lens clear 2-5' removes lenses #2 through #5 on\n"
							"the list.\n";
					else if (words[2]=="anchoring")
						cout << "There are a few different ways that you can anchor a lens parameter to another lens parameter.\n"
							"To demonstrate this, suppose our first lens is entered as follows:\n\n"
							"fit lens alpha 5 1 0 0.8 30 0 0\n"
							"1 1 0 1 1 1 1\n\n"
							"so that this now becomes listed as lens '0'. There are two ways to anchor parameters to this lens:\n\n"
							"a) Anchor type 1: Now suppose you add another model, e.g. a kappa multipole, where I want\n"
							"the angle to always be equal to that of lens 0. Then I enter this as\n\n"
							"fit lens kmpole 0.1 2 anchor=0,4 0 0\n"
							"1 0 0 1 1\n\n"
							"The 'anchor=0,4' means we are anchoring this parameter (the angle) to lens 0, parameter 4\n"
							"which is the angle of the first lens (remember the first parameter is indexed as zero!). The vary\n"
							"flag must be turned off for the parameter being anchored, or else qlens will complain.\n\n"
							"NOTE: Keep in mind that as long as you use the correct format, qlens will not complain no matter\n"
							"how absurd the choice of anchoring is; so make sure you have indexed it correctly. To test it out,\n"
							"you can use 'lens update ...' to update the lens you are anchoring to, and make sure that the\n"
							"anchored parameter changes accordingly.\n\n"
							"b) Anchor type 2: Suppose I want to add a model where I want a parameter to keep the same *ratio*\n"
							"with a parameter in another lens that I started with. You can do this using the following format:\n\n"
							"fit lens alpha 2.5/anchor=0,0 1 0 0.8 30 0 0\n"
							"1 0 0 1 1 1 1\n\n"
							"The '2.5/anchor=0,0' enters the initial value in as 2.5, and since this is half of the parameter we\n"
							"are anchoring to (b=5 for lens 0), they will always keep this ratio. It is even possible to anchor a\n"
							"parameter to another parameter in the *same* lens, if you use the lens number that will be assigned\n"
							"to the lens you are creating. Again, the vary flag *must* be off for the parameter being anchored.\n\n"
							"To anchor the center of a lens to another lens, you can use 'anchor_center=...' as a shortcut. So\n"
							"in the previous example, if we wanted to also anchor the center of the lens to lens 0, we do\n\n"
							"fit lens alpha 2.5/anchor=0,0 1 0 0.8 30 anchor_center=0\n"
							"1 0 0 1 1 0 0\n\n"
							"The vary flags for center coordinates must be entered as zeroes, or they can be omitted altogether.\n";
					else if (words[2]=="anchor")
						cout << "lens anchor <lens1> <lens2>\n\n"
							"Fix the center of <lens1> to that of <lens2> for fitting purposes, where <lens1>\n"
							"and <lens2> are the numbers assigned to the corresponding lens models in the current\n"
							"lens list (type 'lens' to see the list).\n";
					else if (words[2]=="update")
						cout << "lens update <lens_number> ...\n"
							"lens update <lens_number> <param1>=# <param2>=# ...\n\n"
							"Update the parameters in a current lens model. The first argument specifies the number of the lens in the\n"
							"list produced by the 'lens' command, while the remaining arguments should be the same arguments as when you\n"
							"add a new lens model (type 'help lens <model> for a refresher on the arguments for a specific lens model).\n"
							"Alternatively, you can update specific parameters with arguments '<param>=#'. For example:\n\n"
							"lens update 0 b=5 q=0.8 theta=20\n\n"
							"With the above command, qlens will update only the specific parameters above for lens 0, while leaving the\n"
							"other parameters unchanged.\n";
					else if (words[2]=="savetab")
						cout << "lens savetab <lens_number> <filename>\n\n"
							"Save the interpolation tables from a 'tab' lens model to the specified file '<filename>.tab'. Note\n"
							"that <lens_number> must correspond to a 'tab' model in the list of existing lens models generated by\n"
							"the 'lens' command.\n";
					else if (words[2]=="alpha")
						cout << "lens alpha <b> <alpha> <s> <q/e> [theta] [x-center] [y-center]\n\n"
							"where <b> is the mass parameter, <alpha> is the exponent of the radial power law (alpha=1 for\n"
							"isothermal), <s> is the core radius, <q/e> is the axis ratio or ellipticity (depending on the\n"
							"ellipticity mode) , and [theta] is the angle of rotation (counterclockwise, in degrees) about the\n"
							"center (defaults=0).\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="ptmass")
						cout << "lens ptmass <b> [x-center] [y-center]\n\n"
							"where <b> is the Einstein radius of the point mass. If center coordinates are not\n"
							"specified, the lens is centered at the origin by default.\n";
					else if (words[2]=="sheet")
						cout << "lens sheet <kappa> [x-center] [y-center]\n\n"
							"where <kappa> is the external convergence of the mass sheet. Although center coordinates might appear\n"
							"meaningless for a uniform mass sheet, they define the point of zero deflection by the sheet. In reality\n"
							"this point is determined by the perturbing object generating the (local) mass sheet, but it can be\n"
							"arbitrarily chosen during lens model fitting. The simplest choice is to anchor the mass sheet to the\n"
							"primary lens object during the fit, thereby ensuring that the caustic region will not be significantly\n"
							"offcenter from the critical curves. If center coordinates are not specified, the lens is centered at\n"
							"the origin by default.\n";
					else if (words[2]=="shear")
						if (Shear::use_shear_component_params)
							cout << "lens shear <shear_1> <shear_2> [x-center] [y-center]\n\n"
								"where <shear_1> and <shear_2> are the components of the shear.\n";
						else
							cout << "lens shear <shear> [theta] [x-center] [y-center]\n\n"
								"where <shear> is the magnitude of external shear, and [theta] gives the direction of the (hypothetical)\n"
								"perturber that generates the shear (counterclockwise, in degrees) about the center (defaults=0).\n"
								"Note that theta is the angle of the perturber (assuming tangential shear), *NOT* the angle of the shear\n"
								"itself (which differs from theta by 90 degrees).\n\n"
								"Also note that for theta=0, the shear term has a phase shift " << ((LensProfile::orient_major_axis_north) ? "90" : "0") << " degrees, analogous to having\n"
								"the the perturber along the " << LENS_AXIS_DIR << ". However, the direction (x vs. y) for theta=0\n"
								"can be toggled by setting major_axis_along_y on/off.\n";
					else if (words[2]=="mpole")
						cout << "lens mpole [sin/cos] [m=#] <A_m> <n> [theta] [x-center] [y-center]\n\n"
							"Adds a multipole term to the lensing potential, where the optional argument [sin/cos] specifies\n"
							"whether it is a sine or cosine multipole term (default is cosine), [m=#] specifies the order of\n"
							"the multipole term (which must be an integer; default=0), <A_m> is the coefficient of the monopole\n"
							"term, <n> is the power law index and [theta] is the angle of rotation (counterclockwise, in\n"
							"degrees) about the center (defaults=0). For example,\n\n"
							"lens mpole sin m=3 0.05 2 45 0 0\n\n"
							"specifies a sine multipole term of order 3. (For sine terms, the coefficient is labeled as B_m\n"
							"instead of A_m, so in this example B_m=0.05.) The multipole is normalized such that a cosine\n"
							"term with m=2, n=2 is equivalent to an external shear term with A_m = gamma.\n\n"
							"Keep in mind that the order of the multiple (m) cannot be varied as a parameter, so only the\n"
							"remaining five parameters can be varied during a fit or using the 'lens update' command.\n"
							"Note that for theta=0, the sinusoidal term has a phase shift " << ((LensProfile::orient_major_axis_north) ? "90" : "0") << " degrees, analogous to having\n"
							"the major axis of the lens along the " << LENS_AXIS_DIR << " (the direction of the major axis (x/y) for theta=0\n"
							"is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="kmpole")
						cout << "lens kmpole [sin/cos] [m=#] <A_m> <beta> [theta] [x-center] [y-center]\n\n"
							"Adds a multipole term to kappa, with a radial dependence r^(-beta). The optional argument\n"
							"[sin/cos] specifies whether it is a sine or cosine multipole term (default is cosine), [m=#]\n"
							"specifies the order of the multipole term (which must be an integer; default=0), <A_m> is the\n"
							"coefficient of the monopole term, <beta> is the power law index and [theta] is the angle of\n"
							"rotation (counterclockwise, in degrees) about the center (defaults=0). For example,\n\n"
							"lens kmpole cos m=2 0.05 2 45 0 0\n\n"
							"specifies a cosine multipole term of order 2. (For sine terms, the coefficient is labeled as B_m\n"
							"instead of A_m, so in this example B_m=0.05.)\n\n"
							"An important caveat is that beta cannot be larger than 2-m (or else deflections become infinite).\n"
							"Also keep in mind that the order of the multiple (m) cannot be varied as a parameter, so only the\n"
							"remaining five parameters can be varied during a fit or using the 'lens update' command.\n"
							"Note that for theta=0, the sinusoidal term has a phase shift " << ((LensProfile::orient_major_axis_north) ? "90" : "0") << " degrees, analogous to having\n"
							"the major axis of the lens along the " << LENS_AXIS_DIR << " (the direction of the major axis (x/y) for theta=0\n"
							"is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="pjaffe")
						cout << "lens pjaffe <b> <a> <s> <q/e> [theta] [x-center] [y-center]                 (pmode=0)\n"
								  "lens pjaffe <sigma0> <a_kpc> <s_kpc> <q/e> [theta] [x-center] [y-center]    (pmode=1)\n"
								  "lens pjaffe <mtot> <a_kpc> <s_kpc> <q/e> [theta] [x-center] [y-center]      (pmode=2)\n\n"
							"where <b> is the default mass parameter (it is the Einstein radius in the limit of large a, s=0); <a> is\n"
							"the tidal radius, <s> is the core radius, <q/e> is the axis ratio or ellipticity (depending on the ellip-\n"
							"ticity mode), and [theta] is the angle of rotation (counterclockwise, in degrees) about the center\n"
							"(defaults=0). For pmode=1, the mass parameter is the central velocity dispersion, with the truncation and\n"
							"core radii in kpc rather than arcsec; for pmode=2, the mass parameter <mtot> is the total mass.\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="nfw")
						cout << "lens nfw <ks> <rs> <q/e> [theta] [x-center] [y-center]           (pmode=0)\n"
								  "lens nfw <mvir> <c> <q/e> [theta] [x-center] [y-center]          (pmode=1)\n"
								  "lens nfw <mvir> <rs_kpc> <q/e> [theta] [x-center] [y-center]     (pmode=2)\n\n"
							"where <ks/mvir> is the mass parameter, <rs> is the scale radius (or <c>=concentration for pmode=1),\n"
							"<q/e> is the axis ratio or ellipticity (depending on the ellipticity mode), and [theta] is the angle\n"
							"of rotation (counterclockwise, in degrees) about the center (all defaults = 0).\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="tnfw")
						cout << "lens tnfw <ks> <rs> <rt> <q/e> [theta] [x-center] [y-center]\n\n"
							"Truncated NFW profile from Baltz et al. (2008), which is produced by multiplying the NFW density\n"
							"profile by a factor (1+(r/rt)^2)^-2, where rt acts as the truncation/tidal radius.\n\n"
							"Here, <ks> is the mass parameter, <rs> is the scale radius, <rt> is the tidal radius, <q/e> is the\n"
							"axis ratio or ellipticity (depending on the ellipticity mode), and [theta] is the angle of rotation\n"
							"(counterclockwise, in degrees) about the center (all defaults = 0).\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="cnfw")
						cout << "lens cnfw <ks> <rs> <rc> <q/e> [theta] [x-center] [y-center]          (pmode=0)\n"
								  "lens cnfw <mvir> <c> <beta> <q/e> [theta] [x-center] [y-center]       (pmode=1)\n"
								  "lens cnfw <mvir> <rs_kpc> <beta> <q/e> [theta] [x-center] [y-center]  (pmode=2)\n"
								  "lens cnfw <mvir> <c> <rc_kpc> <q/e> [theta] [x-center] [y-center]  (pmode=3)\n\n"
							"Cored NFW profile with 3d density given by rho = rho_s/((r+rc)*(r+rs)^2) Here, <ks/mvir> is the mass\n"
							"parameter, <rs> is the scale radius (or <c>=concentration for pmode=1), <rc> is the core radius (or\n"
							"<beta> = rc/rs for pmodes 1,2), <q/e> is the axis ratio or ellipticity (depending on the ellipticity\n"
							"mode), and [theta] is the angle of rotation (counterclockwise, in degrees) about the center.\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="expdisk")
						cout << "lens expdisk <k0> <rs> <q/e> [theta] [x-center] [y-center]\n\n"
							"where <k0> is the mass parameter, <R_d> is the scale radius, <q/e> is the axis ratio or ellipticity\n"
							"(depending on the ellipticity mode), and [theta] is the angle of rotation (counterclockwise, in degrees)\n"
							"about the center (all defaults = 0).\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="hern")
						cout << "lens hern <ks> <rs> <q/e> [theta] [x-center] [y-center]\n\n"
							"where <ks> is the mass parameter, <rs> is the scale radius, <q/e> is the axis ratio or ellipticity\n"
							"(depending on the ellipticity mode), and [theta] is the angle of rotation (counterclockwise, in degrees)\n"
							"about the center (all defaults = 0).\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="corecusp")
						cout << "lens corecusp [re_param] <k0/R_e> <gamma> <n> <a> <s> <q/e> [theta] [x-center] [y-center]\n\n"
							"This is a cored version of the halo model of Munoz et al. (2001), where <a> is the scale/tidal\n"
							"radius, <k0> = 2*pi*rho_0*a/sigma_crit, <s> is the core radius, and <gamma>/<n> are the inner/outer\n"
							"(3D) log-slopes respectively. To use the Einstein radius as a parameter instead of k0, include the\n"
							"argument 're_param'. (The pseudo-Jaffe profile corresponds to gamma=2, n=4 and b=k0*a/(1-(s/a)^2).)\n"
							"As with the other models, <q/e> is the axis ratio or ellipticity (depending on the ellipticity mode),\n"
							"and [theta] is the angle of rotation (counter-clockwise, in degrees) about the center (all defaults = 0).\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="sersic")
						cout << "lens sersic <kappa_e> <R_eff> <n> <q/e> [theta] [x-center] [y-center]\n\n"
							"The sersic profile is defined by kappa = kappa_e * exp(-b*((R/R_eff)^(1/n)-1)), where kappa_e is the\n"
							"kappa value at the effective (half-mass) radius R_eff, and b is a factor automatically determined from\n"
							"the value for n to ensure that R_eff contains half the total mass (from Cardone et al. 2003). Here,\n"
							"[theta] is the angle of rotation (counterclockwise, in degrees) about the center (defaults=0). Note that\n"
							"for theta=0, the major axis of the source is along the " << LENS_AXIS_DIR << " (the direction of the major\n"
							"axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="kspline")
						cout << "lens kspline <filename> [qx] [f] [q/e] [theta] [x-center] [y-center]\n\n"
							"where <filename> gives the input file containing the tabulated radial profile, <q/e> is the axis ratio\n"
							"or ellipticity (depending on the ellipticity mode), and [theta] is the angle of rotation\n"
							"(counterclockwise, in degrees) about the center. [qx] scales the x-axis, [f] scales kappa for all r.\n"
							"(defaults: qx=f=q=1, theta=xc=yc=0)\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="tab")
						cout << "lens tab <lens=#>/<filename> <kscale> <rscale> [theta] [x-center] [y-center]\n\n"
							"Tabulates the lensing properties over a grid in (log(r),phi) from an existing lens model, corresponding to\n"
							"the # in the 'lens=#' argument, so that lensing properties can be found by linear interpolation; this speeds up\n"
							"calculations for models where numerical integration is required. When generating a 'tab' model from an existing\n"
							"lens, the original lens is replaced by the tabulated version. Alternatively, if the interpolation tables have\n"
							"already been generated and saved to a file using 'lens savetab', the file can be loaded by giving the filename\n"
							"instead of the 'lens=#' argument. The interpolated values can be scaled by a mass parameter <kscale> and along\n"
							"the radial direction using <rscale>. If generating from an existing lens, the specified values of kscale, rscale\n"
							"do not actually alter the lens being tabulated, but rather sets the value for these scale parameters such that\n"
							"the original lens is reproduced if the scale parameters equal these input values. Additionally, the tabulation\n"
							"is done in the (non-rotated) coordinate frame of the lens; therefore, the settings for theta, x-center and\n"
							"y-center for the original lens are ignored in favor of the values entered when generating the 'tab' model.\n\n"
							"The number of points being tabulated are controlled by the parameters 'tab_r_N' and 'tab_phi_N' where phi is\n"
							"the azimuthal angle. The minimum radius is set by 'tab_rmin', while the maximum radius is set by the size of the\n"
							"grid used for image searching (which is set automatically from the Einstein radius if 'autogrid_from_Re' is on).\n"
							"When generating a 'tab' model, it is recommended to test the accuracy of the lensing properties by using the\n"
							"'lensinfo' command for both the original and the tabulated model.\n"
							"Finally, to save the interpolation tables to a file, use 'lens savetab <lens#> <filename>' where <lens#> is the\n"
							"number corresponding to the tabulated model.\n\n";
					else if (words[2]=="qtab")
						cout << "lens qtab <lens=#>/<filename> <kscale> <rscale> <q> [theta] [x-center] [y-center]\n\n"
							"Tabulates the lensing properties over a grid in (log(r),phi,q) from an existing lens model, corresponding to\n"
							"the # in the 'lens=#' argument, so that lensing properties can be found by linear interpolation; this speeds up\n"
							"calculations for models where numerical integration is required. When generating a 'qtab' model from an existing\n"
							"lens, the original lens is replaced by the tabulated version. Alternatively, if the interpolation tables have\n"
							"already been generated and saved to a file using 'lens savetab', the file can be loaded by giving the filename\n"
							"instead of the 'lens=#' argument. The interpolated values can be scaled by a mass parameter <kscale> and along\n"
							"the radial direction using <rscale>. If generating from an existing lens, the specified values of kscale, rscale\n"
							"do not actually alter the lens being tabulated, but rather sets the value for these scale parameters such that\n"
							"the original lens is reproduced if the scale parameters equal these input values. Additionally, the tabulation\n"
							"is done in the (non-rotated) coordinate frame of the lens; therefore, the settings for theta, x-center and\n"
							"y-center for the original lens are ignored in favor of the values entered when generating the 'qtab' model.\n\n"
							"The number of points being tabulated are controlled by parameters 'tab_r_N', 'tab_q_N', 'tab_phi_N' where phi is\n"
							"the azimuthal angle. The minimum radius is set by 'tab_rmin', while the maximum radius is set by the size of the\n"
							"grid used for image searching (which is set automatically from the Einstein radius if 'autogrid_from_Re' is on).\n"
							"Likewise, the minimum q-value being tabulated is set by 'tab_qmin'. When generating a 'qtab' model, it is\n"
							"recommended to test the accuracy of the lensing properties by using the 'lensinfo' command for both the original\n"
							"and the tabulated model.\n\n"
							"Finally, to save the interpolation tables to a file, use 'lens savetab <lens#> <filename>' where <lens#> is the\n"
							"number corresponding to the tabulated model.\n\n";
					else Complain("lens model not recognized");
				}
				else if (words[1]=="fit") {
					if (nwords==2)
						cout << "fit\n"
							"fit lens ...\n"
							"fit sourcept ...\n"
							"fit source_mode <mode>\n"
							"fit run\n"
							"fit chisq\n"
							"fit findimg [sourcept_num]\n"
							"fit plotimg [src=#]\n"
							"fit plotsrc [src=#]\n"
							"fit data_imginfo       (NEED TO WRITE HELP DOCS FOR THIS)\n"
							"fit method <method>\n"
							"fit label <label>\n"
							"fit output_dir <dirname>\n"
							"fit use_bestfit\n"
							"fit save_bestfit\n"
							"fit load_bestfit ...\n"
							"fit add_chain_dparams\n"
							"fit mkposts ...\n"
							"fit plimits ...\n"
							"fit stepsizes ...\n"
							"fit dparams ...\n"
							"fit priors ...\n"
							"fit transform ...\n"
							"fit changevary <lens_number>\n"
							"fit vary_sourcept ...\n"
							"fit regularization <method>\n\n"
							"Commands needed to fit lens models. If the 'fit' command is entered with no arguments, the\n"
							"current fit model (including lens and source) is listed along with the free parameters.\n"
							"For help with the specific fit commands, type 'help fit <command>'. To run the chi-square fit,\n"
							"routine, use the 'fit run' command.\n\n";
					else if (words[2]=="lens") 
						cout << "fit lens ...\n\n"
							"Identical to the 'lens' command, except that after entering the lens model, flags\n"
							"must be entered for each parameter (either 0 or 1) to specify whether it will be\n"
							"varied or not. For example, consider the following command:\n\n"
							"fit lens alpha 10 1 0.5 0.9 0 0 0\n"
							"1 0 0 1 0 0 0\n"
							"0.1 20                             #limits for b  (only needed for MCMC/nested sampling)\n"
							"0.1 1                              #limits for q  (only needed for MCMC/nested sampling)\n\n"
							"This specifies that parameters b, and q will be varied, while the others are fixed; the\n"
							"arguments specified on the first line give their initial values. The total number of flags\n"
							"entered must match the total number of parameters for the given lens model.\n"
							"If you are using T-Walk or nested sampling, you may also need to enter upper/lower bounds\n"
							"for each parameter to be varied, as shown above; otherwise, if you are doing a chi-square\n"
							"optimization (the default qlens mode), those lines are omitted.\n\n"
							"If you would like the center of the lens model to be fixed to that of another lens\n"
							"model that has already been specified, you can type 'anchor_center=<##>' at the end\n"
							"of the fit command in place of the x/y center coordinates, where <##> is the number of\n"
							"the lens model given in the current lens list (to see the list simply type 'lens'). If\n"
							"you anchor to another lens, you must omit the vary flags for the lens center x/y\n"
							"coordinates (always the last two parameters). For example:\n\n"
							"fit lens alpha 10 1 0.5 0.9 0 anchor_center=0\n"
							"1 0 0 1 1\n\n"
							"This is similar to the above example, except that this lens is anchored to lens 0\n"
							"so that their centers will always coincide when the parameters are varied.\n\n"
							"Since it is common to anchor external shear to a lens model, this can be done in\n" <<
							((Shear::use_shear_component_params) ?
								"one line by adding 'shear=<shear_1> <shear_2>' to the end of the line. For example,\n"
								"to add external shear with shear_1=0.1 and shear_2=0.05, one can enter:\n\n"
								"fit lens alpha 10 1 0 0.9 0 0.3 0.5 shear=0.1 0.05\n" :
								"one line by adding 'shear=<shear> <theta>' to the end of the line. For example,\n"
								"to add external shear with shear=0.1 and theta=30 degrees, one can enter:\n\n"
								"fit lens alpha 10 1 0 0.9 0 0.3 0.5 shear=0.1 30\n") <<
							"1 0 0 1 1 1 1 1 1     # vary flags for the alpha model + external shear parameters\n\n"
							"You can also vary the lens redshift by adding the argument 'varyz=1' at the end of the line\n"
							"containing the vary flags. Finally, it is possible to anchor a specific parameter to another\n"
							"parameter in another lens model (or within the same lens model). For more info on this, type\n"
							"'help lens anchoring'.\n";
					else if (words[2]=="sourcept")
						cout << "fit sourcept\n"
							"fit sourcept <sourcept_num>\n"
							"fit sourcept <x0> <y0>                      (for a single source point)\n"
							"fit sourcept auto                           (adopts best fit from source plane chi-sq)\n\n"
							"Specify the initial fit coordinates for the source points, one for each set of images loaded as\n"
							"data (via the 'imgdata' command). If only one source point is being used, these can be given as\n"
							"arguments on the same line (e.g. 'fit sourcept 1 2.5'); for more than one source point, the\n"
							"coordinates must be entered on separate lines. For example, if two image sets are loaded, then\n"
							"to specify the initial source data points one would type:\n\n"
							"fit sourcept\n"
							"1 2.5\n"
							"3.2 -1\n\n"
							"This sets the initial source point coordinates to (1,2.5) and (3.2,-1). To change a specific source\n"
							"point, enter 'fit sourcept #', where # is the index the source point is listed under (which you can\n"
							"see using the 'fit' command), then enter the coordinates on the following line. If the fit method\n"
							"being used requires lower and upper limits on parameters (e.g. for nested sampling or MCMC),\n"
							"the user will be prompted to give lower and upper limits on x and then y for each source point\n"
							"(the format for entering limits is similar to 'fit lens'; see 'help fit lens' for description).\n\n";
					else if (words[2]=="source_mode")
						cout << "fit source_mode <mode>\n\n"
							"Specify the type of source/lens fitting to use. If no argument is given, prints the current source\n"
							"mode. Options are:\n\n"
							"ptsource -- the data is in the form of point images (set using 'imgdata' command) and one\n"
							"            or more point sources are used as fit parameters.\n"
							"pixel -- the data is in the form of a surface brightness map (set using the 'sbmap' command\n"
							"         and the source galaxy is pixellated and inferred by a linear inversion.\n"
							"sbprofile -- similar to 'pixel', except that an analytic model is used for the source galaxy\n"
							"             rather than a pixellated source. (This option has not been implemented yet)\n";
					else if (words[2]=="findimg")
						cout << "fit findimg [sourcept_num]\n\n"
							"Find the images produced by the current lens model and source point(s) and output their positions,\n"
							"magnifications, and (optional) time delays, using the same output format as the 'findimg' command.\n"
							"if no argument is given, image data is output for all the source points being modeled; otherwise,\n"
							"sourcept_num corresponds to the number assigned to a given source point which is listed by the 'fit'\n"
							"command (to set the initial values for the model source points, use the 'fit sourcept' command).\n";
					else if (words[2]=="plotimg")
						cout << "fit plotimg [src=#]\n"
							"fit plotimg [src=#] <sourcepic_file> <imagepic_file>\n\n"
							"Plot the images produced by the current lens model and source point(s) specified by [src=#] to the\n"
							"screen, along with the image data that is being fit (listed by the 'imgdata' command). If no\n"
							"argument is given, images from all source points are plotted; otherwise, # corresponds to the\n"
							"number assigned to a given source point listed by the 'imgdata' command. To plot for a subset of\n"
							"sources, you can use a hyphen, e.g. 'fit plotimg src=3-5' plots for source points 3 through 5\n"
							"only. If all the sources are being plotted (i.e. no 'src=' argument given), critical curves are\n"
							"plotted that correspond to the source redshift given by 'zsrc'; otherwise if plotting for a subset\n"
							"of sources, the critical curves and caustics shown correspond to the source redshift of the first\n"
							"source point being plotted. So in the above example, critical curves are plotted for the source\n"
							"redshift corresponding to image set 3.\n";
					else if (words[2]=="run")
						cout << "fit run [-resume]\n\n"
							"Run the selected model fit routine, which can either be a minimization (e.g. Powell's method)\n"
							"or a Monte Carlo sampler (e.g. MCMC or nested sampling method) depending on the fit method that\n"
							"has been selected. If using MultiNest, you can add the argument '-resume' to continue a prev-\n"
							"ious run that had been interrupted. For more information about the output produced by these\n"
							"methods, type 'help fit method <method>' for whichever fit method is selected.\n";
					else if (words[2]=="chisq")
						cout << "fit chisq [diag]\n\n"
							"Output the chi-square value for the current model and data. If using more than one chi-square\n"
							"component (e.g. fluxes and time delays), each chi-square will be printed separately in addition\n"
							"to the total chi-square value. If argument 'diag' is added, diagnostic information is printed\n"
							"for image plane chi-square; for image plane chi-square, the diagnostics include the chi-square\n"
							"contribution from each data image, the model image it matches to, as well as extra model images\n"
							"that aren't matched to any data image.\n";
					else if (words[2]=="method") {
						if (nwords==3)
							cout << "fit method <fit_method>\n\n"
								"Specify the fitting method, which can be either a chi-square optimization or Monte Carlo\n"
								"sampling routine. Available fit methods are:\n\n"
								"simplex -- minimize chi-square using downhill simplex method (+ optional simulated annealing)\n"
								"powell -- minimize chi-square using Powell's method\n"
								"nest -- basic nested sampling\n"
								"twalk -- T-Walk MCMC algorithm\n"
#ifdef USE_POLYCHORD
								"polychord -- PolyChord nested sampler\n"
#else
								"polychord -- PolyChord nested sampler (*must be compiled with qlens to use)\n"
#endif
#ifdef USE_MULTINEST
								"multinest -- MultiNest nested sampler\n\n"
#else
								"multinest -- MultiNest nested sampler (*must be compiled with qlens to use)\n\n"
#endif
								"For more information on a given fitting method and the output it produces, type\n"
								"'help fit method <fit_method>'.\n";
						else if (words[3]=="simplex")
							cout << "fit method simplex\n\n"
								"The downhill simplex method uses the Nelder-Mead algorithm to minimize the chi-square function\n"
								"and returns the best-fit parameter values. If 'find_errors' is on, the Fisher matrix is then\n"
								"calculated numerically and marginalized error estimates are displayed for each parameter. The\n"
								"maximum allowed number of iterations is controlled by the variable 'simplex_nmax', while the\n"
								"convergence criterion is set by 'chisqtol'.\n\n"
								"Optional simulated annealing can also be used, where the temperature is exponentially reduced\n"
								"by a specified factor until reaching a desired final temperature. The initial temperature is set\n"
								"by 'simplex_temp0' (which is zero by default, in which case no annealing is used), the\n"
								"temperature cooling factor is set by 'simplex_tfac', and final temperature set by 'simplex_tempf'.\n"
								"After the final temperature is reached, a final run is performed with the temperature set to zero.\n"
								"If annealing is used, note that 'simplex_nmax_anneal' is the max iterations allowed at each\n"
								"temperature setting, not the total allowed number of iterations. By contrast, 'simplex_nmax' is\n"
								"the max allowed iterations when the temperature is set to zero, which is the case for the final\n"
								"iteration after annealing. Finally, one may also specify a chi-square threshold 'simplex_minchisq'\n"
								"such that we skip to zero temperature if the chi-square falls below the given threshold.\n\n";
						else if (words[3]=="powell")
							cout << "fit method powell\n\n"
								"Powell's method minimizes the chi-square function using Powell's conjugate direction method\n"
								"and returns the best-fit parameter values. If 'find_errors' is on, the Fisher matrix is then\n"
								"calculated numerically and marginalized error estimates are displayed for each parameter. The\n"
								"convergence criterion is set by 'chisqtol'.\n\n";
						else if (words[3]=="nest")
							cout << "fit method nest\n\n"
								"The nested sampling algorithm outputs points that sample the parameter space, which can then\n"
								"be marginalized by binning the points in the parameter(s) of interest, weighting them according\n"
								"to the supplied weights. The resulting points and weights are output to the file '<label>', while\n"
								"the parameters that maximize the space are output to the file <label>.max', where the label is set\n"
								"by the 'fit label' command. The number of initial 'active' points is set by n_livepts.\n";
						else if (words[3]=="twalk")
							cout << "fit method twalk\n\n"
								"T-Walk is a Markov Chain Monte Carlo (MCMC) algorithm that samples the parameter space using a\n"
								"Metropolis-Hastings step and outputs the resulting chain(s) of points, which can then be marginalized\n"
								"by binning in the parameter(s) of interest. Data points are output to the file '<label>', where the\n"
								"label is set by the 'fit label' command. The algorithm uses the Gelman-Rubin R-statistic to determine\n"
								"convergence and terminates after R reaches the value set by mcmctol.\n";
						else Complain("unknown fit method");
					} else if (words[2]=="label")
						cout << "fit label <label>\n\n"
							"Specify label for output files produced by the chosen fit method (see 'help fit method' for information\n"
							"on the output format for the chosen fit method). By default, the output directory is automatically set\n"
							"to 'chains_<label>' unless the output directory is specified using the 'fit output_dir' command.\n";
					else if (words[2]=="output_dir")
						cout << "fit output_dir <dirname>\n\n"
							"Specify output directory for data produced by the chosen fit method. If not set explicitly, the output\n"
							"directory defaults to 'chains_<label>' where <label> is set using the 'fit label' command (or if the\n"
							"label is not set, defaults to the qlens directory).\n";
					else if (words[2]=="output_dir")
						cout << "fit output_dir <dirname>\n\n"
							"Specify output directory for output files produced by chosen fit method (see 'help fit method' for\n"
							"information on the output format for the chosen fit method).\n";
					else if (words[2]=="use_bestfit")
						cout << "fit use_bestfit\n\n"
							"Adopt the current best-fit lens model (this can only be done after a chi-square fit has been performed).\n";
					else if (words[2]=="save_bestfit")
						cout << "fit save_bestfit\n"
							"fit save_bestfit <label>\n\n"
							"Save information on the best-fit model obtained after fitting to data to the file '<label>.bestfit',\n"
							"where <label> is set by the 'fit label' command, and the output directory is set by 'fit output_dir'.\n"
							"You can also set the fit label in the same line by adding it as an extra argument. In addition to\n"
							"the best fit parameters, the parameter covariance matrix is output to the file '<label>.pcov'.\n"
							"Note that if 'auto_save_bestfit' is set to 'on', the best-fit model is saved automatically after a\n"
							"fit is performed. If one uses T-Walk or nested Sampling, data from the chains is saved automatically\n"
							"regardless of whether 'save_bestfit' is invoked.\n";
					else if (words[2]=="load_bestfit")
						cout << "fit load_bestfit [fit_label]\n\n"
							"Read the best-fit lens model '<fit_label>_bf.in' contained in the fit output directory (which by default\n"
							"is 'chains_<label>'). If [fit_label] is omitted, the current fit label is used (which is set by the 'fit\n"
							"label' command).\n";
					else if (words[2]=="add_chain_dparams")
						cout << "fit add_chain_dparams\n\n"
							"Loads a chain that has already been created, and adds any derived parameters defined using 'fit dparams'.\n"
							"The fit label should match that of the chain, and the lens model should be the same as in the chain\n"
							"(to be safe, run 'fit load_bestfit' beforehand). Any derived parameters that were used in the original\n"
							"chain will still be included.\n";
					else if (words[2]=="mkposts")
						cout << "fit mkposts <dirname> [-n#] [-N#] [-subonly]\n\n"
							"After a chain has been generated using MCMC or nested sampling, 'fit mkposts' will run the mkdist tool\n"
							"from QLens and generate 1d and 2d posteriors, copying the resulting PDF files from the chains directory\n"
							"to directory <dirname> (which is created if it doesn't already exist; otherwise if <dirname> is omitted,\n"
							"the files are not copied to another directory). In addition, the chain and data descriptions given by\n"
							"'chain_info' and 'data_info', the best-fit point and 95\% credible intervals are all output to the file\n"
							"<label>.chain_info and also copied to directory <dirname>. The 'fit mkposts' command is useful if many\n"
							"jobs are being run, so all the posteriors can be automatically generated and placed into the same\n"
							"directory for comparison. In addition, a triangle subplot can be made using a subset of all the fit\n"
							"parameters defined using the 'subplot_params' command (and if '-subonly' is entered, only the subplot\n"
							"is copied to the directory <dirname>).\n"
							"(NOTE: by default, histograms are made using 50 and 40 bins for 1D and 2D posteriors, respectively; if\n"
							"desired, these numbers can be changed using '-n#' and '-N#' arguments, where # is the number of bins.)\n";
					else if (words[2]=="plimits")
						cout << "fit plimits\n"
						"fit plimits <param_num/name> <lower_limit> <upper_limit>\n"
						"fit plimits <param_num/name> none\n\n"
							"Define limits on the allowed parameter space by imposing a steep penalty in the chi-square if the\n"
							"fit strays outside the limits. Type 'fit plimits' to see the current list of fit parameters.\n"
							"For example, 'fit plimits 0 0.5 1.5' limits parameter 0 to the range [0.5:1.5]. To disable the\n"
							"limits for a specific parameter, type 'none' instead of giving limits. (To revert to the default parameter\n"
							"limits, type 'fit plimits reset'.) Setting parameter limits this way is useful when doing a chi-square\n"
							"minimization with downhill simplex or Powell's method, but is unnecessary for the Monte Carlo samplers\n"
							"(twalk or nest) since limits must be entered for all parameters when the fit model is defined.\n";
					else if (words[2]=="stepsizes")
						cout << "fit stepsizes\n"
							"fit stepsizes <param_num/name> <stepsize>\n"
							"fit stepsizes scale <factor>\n\n"
							"Define initial stepsizes for the chi-square minimization methods (simplex or powell). Values are\n"
							"automatically chosen for the stepsizes by default, but can be customized. Type 'fit stepsizes' to\n"
							"see the currest list of fit parameters. For example, 'fit stepsizes 5 0.3' sets the initial stepsize\n"
							"for parameter 5 to 0.3. You can also scale all the stepsizes by a certain factor, e.g. 'fit\n"
							"stepsize scale 5' multiplies all stepsizes by 5. To reset all stepsizes to their automatic values,\n"
							"type 'fit stepsize reset'.\n";
					else if (words[2]=="priors")
						cout << "fit priors\n"
							"fit priors <param_num/name> <prior_type> [prior_params]\n\n"
							"Define prior probability distributions in each fit parameter, which are used by the T-Walk and nested\n"
							"sampling routines. Type 'fit priors' to see current list of fit parameters and corresponding priors.\n"
							"For the first argument you can put the parameter number as displayed in the list, or you can enter the\n"
							"parameter name (the latter approach is less bug-prone since numbers may change if the model is changed).\n"
							"The list of available priors is given below. Regardless of the prior chosen, the upper and lower bounds\n"
							"defined when you create lens models (in twalk or nest mode) still apply. Keep in mind that the initial\n"
							"sampling of the parameter space will not follow the prior chosen, but rather will draw uniform deviates\n"
							"within the specified limits. If you want the initial sampling to follow the prior, consider doing a\n"
							"parameter transformation instead using 'fit transform'.\n\n"
							"Available prior types:\n\n"
							"uniform  -- this is the default prior for all parameters\n"
							"log      -- prior 1/p, where p is the parameter; this is equivalent to a uniform prior in log(p)\n"
							"gaussian -- Gaussian prior with two parameter arguments, mean value <mean> and dispersion <sig>\n"
							"              (e.g., 'fit priors # gaussian 0.2 0.5' will be Gaussian with mean 0.2 and dispersion 0.5)\n"
							"gauss2 -- Gaussian prior in two parameters, with arguments (p2,mean1,mean2,sig1,sig2,sig12), where p2 gives\n"
							"            the second parameter and sig1,sig2,sig12 are the square root of the elements of the covariance\n"
							"            matrix between the two parameters.\n"
							"            (e.g., 'fit priors 0 gauss2 1 0.5 0.5 0.1 0.2 0.05' is a Gaussian prior in parameters 0,1.)\n\n";
					else if (words[2]=="transform")
						cout << "fit transform\n"
							"fit transform <param_num/name> <transform_type> [transform_params] ... [include_jac]\n\n"
							"Define coordinate transformation of one or more fit parameters. Type 'fit transform' to see current list of\n"
							"fit parameters and corresponding transformations/priors being used. The list of available transformations is\n"
							"given below. If upper/lower bounds were defined on the original parameter while in twalk/nest mode, these\n"
							"bounds will also be transformed. Transforming parameters is a useful alternative to defining non-uniform\n"
							"priors, since the transformation also changes the way the parameter space is initially sampled; for example,\n"
							"if a Gaussian transformation is made, then the initial sampling will follow a Gaussian distribution in the\n"
							"original parameter. If a transformation is made to better sample the prior space, then the corresponding\n"
							"prior should not also be selected; for example, you can either define a log-prior in the parameter p, or\n"
							"else transform to log(p) and use a uniform prior in log(p). These two approaches are equivalent, the only\n"
							"difference is in how the parameter space is initially explored. If you want to transform the parameter but\n"
							"still have a uniform prior in the *original* parameter, add the argument 'include_jac'.\n\n"
							"Available transformation types:\n\n"
							"none     -- no transformation (the default for all parameters)\n"
							"log      -- transform to log(p) using the base 10 logarithm\n"
							"linear   -- transform to L{p} = A*p + b. The two parameter arguments are <A> and <b>, so e.g. 'fit transform\n"
							"              linear 2 5' will transform p --> 2*p + 5.\n"
							"ratio   -- transform p1 --> p1/p2, that is, to the ratio of two parameters. There is one argument, which is\n"
							"              the parameter number for p2. e.g., 'fit transform 3 ratio 4' will transform to the ratio of\n"
							"              parameter 3 over parameter 4.\n"
							"gaussian -- transformation whose Jacobian is Gaussian, and thus is equivalent to having a Gaussian prior in\n"
							"              the original parameter. There are two arguments, mean value <mean> and dispersion <sig>\n"
							"              (e.g., 'fit transform # gaussian 0.2 0.5' will be Gaussian with mean 0.2 and dispersion 0.5)\n\n";
					else if (words[2]=="dparams")
						cout << "fit dparams\n"
							"fit dparams add <param_type> [param_arg] [lens#]     (lens# and/or param_arg are optional for some dparams)\n"
							"fit dparams rename <param_type> <text_name> <latex_name>\n"
							"fit dparams clear [param_num]\n\n"
							"Define derived parameters whose values will be output along with the primary parameters after running\n"
							"nested sampling or T-Walk. Type 'fit dparams' to list all the derived parameters that have been defined,\n"
							"'fit dparams clear' to remove one or all of the derived parameters from the list, and 'fit dparams add'\n"
							"to add a new derived parameter. All derived parameters are defined by a type and (usually) an argument;\n"
							"in the list below, the parameter argument is denoted in brackets <...>, otherwise no argument is required.\n"
							"The available derived parameter types are:  ('*' means if lens_number is omitted, all lenses are included)\n\n"
							"raw_chisq -- The chi-square value (not including priors; no arguments required)\n"
							"kappa_r  -- *The kappa at radius <r>, averaged over all angles (where <r> in arcseconds)\n"
							"dkappa_r -- *The derivative of kappa at radius <r>, averaged over all angles (where <r> is in arcseconds)\n"
							"mass2d_r -- The projected mass enclosed within elliptical radius <r> (in arcsec) for a specific lens [lens#]\n"
							"mass3d_r -- The 3d mass enclosed within elliptical radius <r> (in arcsec) for a specific lens [lens#]\n"
							"re_zsrc -- The (spherically averaged) Einstein radius of lens [lens#] for a source redshift <zsrc>\n"
							"mass_re -- The projected mass enclosed within Einstein radius of lens [lens#] for a source redshift <zsrc>\n"
							"lensparam -- The value of parameter <paramnum> for lens [lens#] using default pmode\n"
							"r_perturb -- The critical curve perturbation radius of perturbing lens [lens#]; assumes lens 0 is primary\n"
							"                  (See Minor et al. 2017 for definition of perturbation radius for subhalos)\n"
							"mass_perturb -- The projected mass enclosed within r_perturb (see above) of perturbing lens [lens#]\n"
							"sigma_perturb -- The average projected density within r_perturb (see above) of perturbing lens [lens#]\n"
							"r_perturb_rel -- Same as r_perturb, except it's subtracted from the unperturbed critical curve location\n"
							"\n";
					else if (words[2]=="changevary")
						cout << "fit changevary <lens_number>\n\n"
							"Change the parameter vary flags for a specific lens model that has already been created. After specifying\n"
							"the lens, on the next lens vary flags are entered just as you do when creating the lens model. Note that\n"
							"the number of vary flags must exactly match the number of parameters for the given lens (except that vary\n"
							"flags for the center coordinates can be omitted if the lens in question is anchored to another lens).\n";
					else if (words[2]=="vary_sourcept")
						cout << "fit vary_sourcept\n"
							"fit vary_sourcept <vary_srcx> <vary_srcy>\n\n"
							"Specify whether to vary each component of the source point being fit (0=fixed, 1=vary). If only one\n"
							"source point is being fit to, the vary flags can be given as the third and fourth arguments. If\n"
							"more than one source point is being fit to, type 'fit vary_sourcept' and you will be prompted for\n"
							"the vary flags for each source point separately.\n";
					else if (words[2]=="regularization")
						cout << "fit regularization <method>\n\n"
							"Specify the type of regularization that will be used when fitting to a surface brightness pixel\n"
							"map using source pixel reconstruction. Regularization imposes smoothness on the reconstructed\n"
							"source and is vital if significant noise is present in the image. If no argument is given,\n"
							"prints the current regularization method. Available methods are:\n\n"
							"none -- no regularization\n"
							"norm -- regularization matrix built from (squared) surface brightness of each pixel\n"
							"gradient -- regularization matrix built from the derivative between neighboring pixels\n"
							"curvature -- regularization matrix built from the curvature between neighboring pixels (default)\n\n";
					else Complain("unknown fit command");
				}
				else if (words[1]=="imgdata") {
					if (nwords==2) {
						cout << "imgdata\n"
							"imgdata read <filename>\n"
							"imgdata add <x_coord> <y_coord>\n"
							"imgdata write <filename>\n"
							"imgdata plot [dataset_number]\n"
							"imgdata clear [dataset_number]\n"
							"imgdata add_from_centroid ...\n"
							"imgdata use_in_chisq ...\n\n"
							"Commands for loading (or simulating) point image data for lens model fitting. For help\n"
							"on each command type 'help imgdata <command>'. If 'imgdata' is typed with no argument,\n"
							"displays the current image data that has been loaded.\n";
					} else {
						if (words[2]=="read")
							cout << "imgdata read <filename>\n\n"
								"Read image data from file '<filename>'. On the first line, the lens redshift can be\n"
								"specified using 'zlens = <#>', but this is optional. Next, the file must specify the\n"
								"number of image sets on line 1; for each image set, the number of images is specified,\n"
								"followed by the source redshift (also optional). Then follows a line of data for each\n"
								"image. The image data consists of seven fields: the x/y coordinates, the position error,\n"
								"the flux and flux error, and the time delay and time delay error. If flux and/or time\n"
								"delays are not given, the corresponding errors should be given a value of zero. An example\n"
								"data file looks as follows (note comments can be added to file using '#'):\n\n"
								"zlens = 0.5  # optional lens redshift\n"
								"1 # number of image sets\n"
								"4 2.0 # number of images for set 1, followed by source redshift (optional)\n"
								"#pos_x    pos_y    sig_pos  flux      sig_flux td       sig_td\n"
								"-3.296107 0.398733 1.000000 -7.080211 1.000000 1.820612 1.000000\n"
								"-2.765528 -2.229838 1.000000 7.148270 1.000000 1.616719 1.000000\n"
								"3.631658 -1.031247 1.000000 -2.010732 1.000000 5.377263 1.000000\n"
								"1.102283 4.876925 1.000000 4.109006 1.000000 0.000000 1.000000\n\n"
								"Note, comments (marked with #) can be placed at the end of any line or at the start\n"
								"of a line.\n";
						else if (words[2]=="write")
							cout << "imgdata write <filename>\n\n"
								"Outputs the current image data set to file, with the same format as required by\n"
								"'imgdata read' command. (For a description of the required format, type 'help imgdata\n"
								"read'.)\n";
						else if (words[2]=="add")
							cout << "imgdata add <x-coord> <y-coord>\n\n"
								"Add simulated image data set defined by the source point (<x-coord>,<y-coord>) lensed\n"
								"by the current lens configuration. The data errors are given by the variables sim_err_pos,\n"
								"sim_err_flux, and sim_err_td for the position, flux and time delay (if using) of each image.\n"
								"A corresponding random Gaussian error is added to each value.\n";
						else if (words[2]=="plot")
							cout << "imgdata plot [dataset_number]\n"
								"imgdata plot sbmap\n\n"
								"Plots the image positions corresponding to the specified set of images loaded as data,\n"
								"corresponding to the number given by the 'imgdata' command (default=0 if no argument given).\n"
								"To plot alongside image positions from the current lens model being fit, use 'fit plotimg' instead.\n"
								"If 'sbmap' is added as a third argument, superimposes image data points with the surface brightness\n"
								"pixel data (if loaded).\n";
						else if (words[2]=="clear")
							cout << "imgdata clear [dataset_number]\n"
								"imgdata clear [min_dataset]-[max_dataset]\n\n"
								"Removes one (or all) of the image data sets in the current configuration. If no argument is given,\n"
								"all data sets are removed; if a single argument <dataset_number> is given, removes only the data set\n"
								"assigned to the given number in the list generated by the 'imgdata' command. To remove more than one\n"
								"image dataset, you can use a hyphen; for example, 'imgdata clear 2-5' removes datasets #2 through #5\n"
								"on the list.\n";
						else if (words[2]=="add_from_centroid")
							cout << "imgdata add_from_centroid <xmin> <xmax> <ymin> <ymax>\n\n"
								"Finds the centroid of the image surface brightness map (if loaded), within the range of\n"
								"coordinates supplied by the arguments (xmin,xmax,ymin,ymax). The resulting centroid is then\n"
								"added as a point image. A minimum surface brightness threshold for inclusion in the calculation\n"
								"may be specified by setting the 'sb_threshold' variable, which is zero by default. The total\n"
								"flux is also calculated (note however that the image is large enough, the total flux may be\n"
								"unreliable as an approximation to the flux at the centroid point). The position error is simply\n"
								"the pixel width, whereas the flux error is determined from the pixel noise (data_pixel_noise).\n";
						else if (words[2]=="use_in_chisq")
							cout << "imgdata use_in_chisq <n_imgset> <nimg> [on/off]\n\n"
								"Specify whether to include a particular data image in the position chi-square. By default, all of the\n"
								"images are included in the chi-square, unless a particular image is set to 'off'. Note that if an\n"
								"image is set to 'off' and the image plane chi-square is being used, the data image is still matched to\n"
								"the closest model image, but the matching pair is then excluded from the chi-square (however, a penalty\n"
								"is still incurred if there is no matching image at all). If the source positions are being solved for\n"
								"analytically (using 'analytic_bestfit_src'), the images set to 'off' are not included in the calculation\n"
								"of the best-fit source points. When the list of images is plotted using the 'imgdata' command, the\n"
								"message 'excluded from chisq' is printed next to any image that is being excluded from the chi-square.\n";
						else Complain("imgdata command not recognized");
					}
				}
				else if (words[1]=="wldata")
				{
					if (nwords==2) {
						cout << "wldata\n"
							"wldata read <filename>\n"
							"wldata add <x_coord> <y_coord>\n"
							"wldata write <filename>\n"
							"wldata clear [dataset_number]\n\n"
							"Commands for loading (or simulating) weak lensing data for lens model fitting. For help\n"
							"on each command type 'help wldata <command>'. If 'wldata' is typed with no argument,\n"
							"displays the current weak lensing data that has been loaded.\n";
					} else {
						if (words[2]=="read")
							cout << "wldata read <filename>\n\n"
								"Read weak lensing data from file '<filename>'. The variable 'chisq_weak_lensing' is automatically\n"
								"turned on after the data is loaded in, so the data will be included in the chi-square. The data\n"
								"file should have the following format:\n\n"
								"#ID   x    y    g1   g2   err_g1   err_g2    zsrc\n\n"
								"where ID is an identifier name for a given source, g1 and g2 are the reduced shear values, with\n"
								"corresponding errors err_g1 and err_g2, and zsrc is the redshift of the lensed source.\n"
								"Note, comments (marked with #) can be placed at the end of any line or at the start of a line.\n";
						else if (words[2]=="write")
							cout << "wldata write <filename>\n\n"
								"Outputs the current weak lensing data set to file, with the same format as required by\n"
								"'wldata read' command. (For a description of the required format, type 'help wldata read')\n";
						else if (words[2]=="add")
							cout << "wldata add <x-coord> <y-coord> [z=#]\n\n"
								"Add simulated weak lensing source defined by the source point (<x-coord>,<y-coord>) lensed by the\n"
								"current lens configuration. The redshift source can be entered via the optional argument 'z=#'; if\n"
								"this argument is not entered, the source has a default redshift given by the value of 'zsrc'.\n"
								"The errors in the reduced shear are given by the variable sim_err_shear, so that a corresponding\n"
								"random Gaussian error is added to the reduced shear components g1 and g2.\n";
						else if (words[2]=="clear")
							cout << "wldata clear\n\n"
								"Removes all the weak lensing data from the current configuration.\n";
						else Complain("wldata command not recognized");
					}
				}
				else if (words[1]=="sbmap")
				{
					if (nwords==2) {
						cout << "sbmap\n"
							"sbmap plotimg [...]\n"
							"sbmap makesrc\n"
							"sbmap savesrc [output_file]\n"
							"sbmap plotsrc [output_file]\n"
							"sbmap loadsrc <source_file>\n"
							"sbmap loadimg <image_file>\n"
							"sbmap loadpsf <psf_file>\n"        // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap unloadpsf\n"                 // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap plotdata\n"
							"sbmap invert\n"
							"sbmap loadmask <mask_file>\n"      // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap set_all_pixels\n"
							"sbmap unset_all_pixels\n"
							"sbmap set_data_annulus [...]\n"
							"sbmap set_data_window [...]\n"
							"sbmap unset_data_annulus [...]\n"
							"sbmap unset_data_window [...]\n"
							"sbmap unset_low_sn_pixels [...]\n"
							"sbmap trim_mask_windows [...]\n"
							"sbmap set_neighbor_pixels\n"
							"sbmap find_noise [...]\n\n"
							"Commands for loading, simulating, plotting and inverting surface brightness pixel maps. For\n"
							"help on individual subcommands, type 'help sbmap <command>'. If 'sbmap' is typed with no\n"
							"arguments, shows the dimensions of current image/source surface brightness maps, if loaded.\n";
					} else {
						if (words[2]=="loadimg")
							cout << "sbmap loadimg <image_filename>\n\n"
								"Load an image surface brightness map from file <image_file>. If 'fits_format' is off, then\n"
								"loads a text file with name '<image_filename>.dat' where pixel values are arranged in matrix\n"
								"form. The x-values are loaded in from file '<image_filename>.x' and likewise for y-values.\n"
								"If 'fits_format' is on, then the filename must be in FITS format, and the size of each\n"
								"pixel must be specified beforehand in the variable 'data_pixel_size'. If no pixel size\n"
								"has been specified, than the grid dimensions specified by the 'grid' command are used to\n"
								"set the pixel size. After loading the pixel image, the number of image pixels for plotting\n"
								"(set by 'img_npixels') is automatically set to be identical to those of the data image.\n\n"
								"Currently the pixel noise is specified by setting 'data_pixel_noise', and is assumed to have\n"
								"the same dispersion for all pixels (in future versions there will be an option to load a noise\n"
								"map for uncorrelated pixel noise, or a full covariance matrix for correlated pixel noise).\n";
						else if (words[2]=="loadsrc")
							cout << "sbmap loadsrc <source_filename>\n\n"
								"Load a source surface brightness pixel map that was previously saved in qlens (using 'sbmap\n"
								"savesrc').\n";
						else if (words[2]=="plotimg")
							cout << "sbmap plotimg [-...] (optional arguments: [-fits] [-residual] [-replot] [-nosrc] [-reduce2/4/8])\n"
								"sbmap plotimg [-...] [image_file]\n"
								"sbmap plotimg [-...] [source_file] [image_file]\n\n"
								"Plot a lensed pixel image from a pixellated source surface brightness distribution under\n"
								"the assumed lens model. If the terminal is set to 'text', the surface brightness values are\n"
								"output in matrix form in the file '<image_file>.dat', whereas the x-values are plotted as\n"
								"'<image_file>.x', and likewise for the y-values (if no arguments are given, the file labels\n"
								"default to 'src_pixel' and 'img_pixel' for the source and image pixel maps, respectively, and\n"
								"these are plotted to the screen in a separate window; if only one file argument is given, the\n"
								"source pixel map is not plotted). The source file is plotted in the same format. The critical\n"
								"curves and caustics are also plotted, unless show_cc is set to 'off'. If simulated pixel noise\n"
								"is specified ('sim_pixel_noise'), then random noise is added to each pixel with dispersion set\n"
								"by sim_pixel_noise.\n\n"
								"Optional arguments:\n"
								"  [-fits] plots to FITS files; filename(s) must be specified with this option\n"
								"  [-residual] plots residual image by subtracting from the data image\n"
								"  [-nosrc] omit the source plane plot (equivalent having 'show_srcplane' off)\n"
								"  [-reduce2/4/8] generate higher resolution image first, then reduces number of pixels by aver-\n"
								"               aging 2x2 or 4x4 or 8x8 pixel groups to generate low-res pixel surface brightness\n"
								"               values.\n"
								"  [-replot] plots image that was previously found and plotted by the 'sbmap plotimg' command.\n"
								"     This allows one to tweak plot parameters (range, show_cc etc.) without having to calculate\n"
								"     the lensed pixel images again.\n\n"
								"OPTIONAL: arguments to 'sbmap plotimg' can be followed with terms in brackets [#:#][#:#]\n"
								"specifying the plotting range for the x and y axes, respectively. A range is allowed\n"
								"for both the source and image plots. Two examples in postscript mode:\n\n"
								"sbmap plotimg source.ps image.ps [-5:5][-5:5] [-15:15][-15:15]\n"
								"sbmap plotimg source.ps image.ps [][] [0:15][0:15]\n\n"
								"In the first example a range is specified for both the x/y axes for both plots,\n"
								"whereas in the second example a range is specified only for the image plot.\n";
						else if (words[2]=="makesrc")
							cout << "makesrc\n"
								"Create a pixellated source surface brightness map from one or more analytic surface brightness\n"
								"profiles, which are specified using the 'source' command. The number of source pixels and size\n"
								"of the source pixel grid are specified using the 'src_npixels' and 'srcgrid' commands.\n";
						else if (words[2]=="savesrc")
							cout << "sbmap savesrc [output_file]\n\n"
								"Output the source surface brightness map to files. The surface brightness values are\n"
								"output in matrix form in the file '<output_file>.dat', and also in a more efficient form as\n"
								"'<output_file>.sb', which saves information for splitting source pixels into subpixels (only\n"
								"important if adaptive_grid is on); this file is read when loading the source later using 'sbmap\n"
								"loadsrc'. The x-values are plotted as '<output_file>.x', and likewise for the y-values.\n";
						else if (words[2]=="plotsrc")
							cout << "sbmap plotsrc [output_file]\n\n"
								"Plot the pixellated source surface brightness distribution (which has been generated either\n"
								"using the makesrc, loadsrc, or invert commands). If the terminal is set to 'text', the surface\n"
								"brightness values are output in matrix form in the file '<source_file>.dat', whereas the\n"
								"x-values are plotted as '<source_file>.x', and likewise for the y-values. If no arguments are\n"
								"given, the file label defaults to 'src_pixel', and these are plotted to the screen in a\n"
								"separate window.  If a lens model has been created, the caustics are also plotted, unless\n"
								"show_cc is set to 'off'.\n\n"
								"OPTIONAL: arguments to 'sbmap plotsrc' can be followed with terms in brackets [#:#][#:#]\n"
								"specifying the plotting range for the x and y axes, respectively. Example in postscript mode:\n\n"
								"sbmap plotsrc source.ps [-5:5][-5:5]\n\n";
						else if (words[2]=="plotdata")
							cout << "sbmap plotdata\n"
								"sbmap plotdata [output_file]\n\n"
								"Plot the image surface brightness pixel data, which is loaded using 'sbmap loadimg'. If no\n"
								"file argument is given, uses file label 'img_pixel' for plotting, and the result is plotted\n"
								"to the screen (the output file conventions are the same as in 'sbmap plotimg').\n\n"
								"OPTIONAL: arguments to 'sbmap plotdata' can be followed with terms in brackets [#:#][#:#]\n"
								"specifying the plotting range for the x and y axes, respectively. A range is allowed\n"
								"for both the source and image plots. An example in postscript mode:\n\n"
								"sbmap plotdata image.ps [-5:5][-4:4]\n";
						else if (words[2]=="invert")
							cout << "sbmap invert\n\n"
								"Invert the image surface brightness map under the assumed lens model using linear inversion. The\n"
								"method used for the linear inversion is specified in 'inversion_method', which can be set to either\n"
								"'cg' (conjugate gradient method), 'mumps' or 'umfpack'. The latter two options require qlens to be\n"
								"compiled with the MUMPS or UMFPACK software packages, respectively.\n";
						else if (words[2]=="set_all_pixels")
							cout << "sbmap set_all_pixels\n\n"
								"Activates all pixels in the image data so they are used in fitting and plotting. This command can only\n"
								"be used after 'sbmap loadimg ...' has been used to load the image data.\n";
						else if (words[2]=="unset_all_pixels")
							cout << "sbmap unset_all_pixels\n\n"
								"Deactivates all pixels in the image data so none are used in fitting and plotting. This command is\n"
								"used before selecting specific regions to activate using 'sbmap set_data_annulus' or 'sbmap\n"
								"set_data_window' commands. Also note that this command can only be used after 'sbmap loadimg ...' has\n"
								"been used to load the image data.\n";
						else if (words[2]=="set_data_annulus")
							cout << "sbmap set_data_annulus <xcenter> <ycenter> <rmin> <rmax> [theta_min] [theta_max]\n\n"
								"Activates all pixels within the specified annulus. In addition to the center coordinates and min/max\n"
								"radii, the user may also specify an angular range with the additional arguments theta_min, theta_max\n"
								"(where the angles are in degrees and must be in the range 0 to 360). The defaults are theta_min=0,\n"
								"theta_max=360 (in other words, a complete annulus). Also note that this command does not deactivate\n"
								"pixels, so it is recommended to use 'sbmap unset_all_pixels' before running this command.\n\n";
						else if (words[2]=="set_data_window")
							cout << "sbmap set_data_window <xmin> <xmax> <ymin> <ymax>\n\n"
								"Activates all pixels within the specified rectangular window. Note that this command does not deactivate\n"
								"pixels, so it is recommended to use 'sbmap unset_all_pixels' before running this command.\n\n";
						else if (words[2]=="unset_data_annulus")
							cout << "sbmap unset_data_annulus <xcenter> <ycenter> <rmin> <rmax> [theta_min] [theta_max]\n\n"
								"This command is identical to 'set_data_annulus' except that it deactivates the pixels, rather than\n"
								"activating them. See 'help sbmap set_data_annulus' for usage information.\n\n";
						else if (words[2]=="unset_data_window")
							cout << "sbmap unset_data_window <xmin> <xmax> <ymin> <ymax>\n\n"
								"This command is identical to 'set_data_window' except that it deactivates the pixels, rather than\n"
								"activating them. See 'help sbmap set_data_window' for usage information.\n\n";
						else if (words[2]=="unset_low_sn_pixels")
							cout << "sbmap unset_low_sn_pixels <sb_threshold>\n\n"
								"Deactivates all pixels with surface brightness values lower than the stated threshold <sb_threshold>.\n"
								"This can be used in conjunction with the 'trim_mask_windows' command, if one uses a conservatively\n"
								"threshold but only wants to keep mask regions that contain high surface brightness.\n\n";
						else if (words[2]=="trim_mask_windows")
							cout << "sbmap trim_mask_windows <threshold>\n\n"
								"Removes pixel 'windows' (regions with continguous pixels) that do not have any pixels with surface\n"
								"brightness greater than <threshold>*pixel_noise. Typically, thresholds of 4 or 5 are sufficient\n"
								"assuming Gaussian noise.\n\n";
						else if (words[2]=="set_neighbor_pixels")
							cout << "sbmap set_neighbor_pixels\n\n"
								"Activate all pixels that neighbor a pixel already included within the mask. This effectively enlarges\n"
								"the pixel mask outward by one pixel length.\n\n";
						else if (words[2]=="find_noise")
							cout << "sbmap find_noise <xmin> <xmax> <ymin> <ymax>\n\n"
								"Calculates the mean and dispersion of the surface brightness values for pixels within the specified\n"
								"rectangular window. When calculating the dispersion, iterative 3-sigma clipping is used to exclude\n"
								"outlier pixels which can bias the estimated dispersion.\n\n";
						else Complain("command not recognized");
					}
				}
				else if (words[1]=="source")
				{
					if (nwords==2)
						cout << "source <sourcemodel> <source_parameter##> ...\n"
							"source clear\n\n"
							"Creates a source object from specified source surface brightness profile model and parameters.\n"
							"If other sources are present, the new source will be superimposed with the others. If no arguments\n"
							"are given, a numbered list of the current source models being used is printed along with their\n"
							"parameter values. To remove a source or delete the entire configuration, use 'source clear'.\n\n"
							"Available source models:    (type 'help source '<sourcemodel>' for usage information)\n\n"
							"gaussian -- Gaussian with dispersion <sig> and q*<sig> along major/minor axes respectively\n"
							"sersic -- Sersic profile S = S0*exp(k*r^(1/n))\n"
							"tophat -- ellipsoidal 'top hat' profile\n"
							"spline -- splined surface brightness profile (generated from an input file)\n\n";
					else if (words[2]=="clear")
						cout << "source clear <#>\n\n"
							"Remove source model # from the list (the list of source objects can be printed using the\n"
							"'source' command). If no arguments given, delete entire source configuration and start over.\n";
					else if (words[2]=="gaussian")
						cout << "source gaussian <max_sb> <sigma> <q> [theta] [x-center] [y-center]\n\n"
							"where <max_sb> is the peak value of the surface brightness, <sigma> is the dispersion of\n"
							"the surface brightness along the major axis of the profile, <q> is the axis ratio (so that\n"
							"the dispersion along the minor axis is q*max_sb), and [theta] is the angle of rotation\n"
							"(counterclockwise, in degrees) about the center (defaults=0). Note that for theta=0, the\n"
							"major axis of the source is along the " << LENS_AXIS_DIR << " (the direction of the major axis (x/y) for\n"
							"theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="sersic")
						cout << "source sersic <s0> <R_eff> <n> <q> [theta] [x-center] [y-center]\n\n"
							"The sersic profile is defined by s = s0 * exp(-b*(R/R_eff)^(1/n)), where b is a factor automatically\n"
							"determined from the value for n (enforces the half-light radius Re). For an elliptical model, we make\n"
							"the replacement R --> sqrt(q*x^2 + (y^2/q), analogous to the elliptical radius defined in the lens\n"
							"models. Here, [theta] is the angle of rotation (counterclockwise, in degrees) about the center\n"
							"(defaults=0). Note that for theta=0, the major axis of the source is along the " << LENS_AXIS_DIR << " (the\n"
							"direction of the major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="tophat")
						cout << "source tophat <sb> <R> <q> [theta] [x-center] [y-center]\n\n"
							"The tophat profile is defined by a constant surface brightness <sb> within an ellipsoidal region with\n"
							"major axis <R> and axis ratio <q>, and zero surface brightness outside this region. Here, [theta] is the\n"
							"angle of rotation (counterclockwise, in degrees) about the center (defaults=0). Note that for theta=0,\n"
							"the major axis of the source is along the " << LENS_AXIS_DIR << " (the direction of the major axis (x/y) for theta=0\n"
							"is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="spline")
						cout << "source spline <filename> <q> [theta] [qx] [f] [x-center] [y-center]\n\n"
							"where <filename> gives the input file containing the tabulated radial surface brightness profile,\n"
							"<q> is the axis ratio, and [theta] is the angle of rotation (counterclockwise, in degrees)\n"
							"about the center. [qx] scales the major axis, [f] scales the surface brightness for all r.\n"
							"(default: qx=f=1, theta=xc=yc=0)\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else Complain("source model not recognized");
				}
				else if (words[1]=="defspline")
					cout << "defspline <N> [xmax] [ymax]   (creates deflection field spline)\n"
							  "defspline off                 (deletes current spline)\n\n"
						"Creates a bicubic spline of the deflection field with NxN steps over the grid\n"
						"(-xmax,xmax) x (-ymax,ymax). If [xmax] and [ymax] are not specified, defaults to\n"
						"current gridsize. If no argument is given, outputs the size and number of steps of\n"
						"current spline. If 'off' is specified, deletes current deflection spline.\n";
				else if (words[1]=="auto_defspline")  // obsolete--should probably remove
					cout << "auto_defspline <N>\n\n"  // obsolete--should probably remove
						"Creates a bicubic spline of the deflection field with NxN steps over an optimized\n"  // obsolete--should probably remove
						"grid determined by the points where the outer critical curve crosses x/y axes.\n";  // obsolete--should probably remove
				else if (words[1]=="lensinfo")
					cout << "lensinfo <x> <y>\n\n"
						"Displays the total kappa, deflection, potential, magnification, shear magnitude/direction, and\n"
						"corresponding source position for the point (x,y).\n";
				else if (words[1]=="plotlensinfo")
					cout << "plotlensinfo [file_root]\n\n"
						"Plot a pixel map of the kappa, magnification, shear, and potential at each pixel, which are output to\n"
						"'<file_root>.kappa', '<file_root>.mag', and so on (if no file label is specified, <file_root> defaults\n"
						"to 'lensmap'). The number of pixels set using the 'img_npixels' command, and grid dimensions are defined\n"
						" by the 'grid' command. All files are output to the directory set by the 'set_output_dir' command.\n";
				else if (words[1]=="plotcrit")
					cout << "plotcrit\n"
						"plotcrit <file>                      (text mode)\n"
						"plotcrit <cc_file> <caustics_file>   (postscript/PDF mode)\n\n"
						"The critical curves and corresponding caustics are plotted to <file>. If no filenames\n"
						"are specified, critical curves and caustics are plotted graphically using Gnuplot.\n"
						"In postscript/PDF mode, two filenames are required (for c.c's and caustics separately).\n"
						"In text mode, the data is written to the file as follows:\n"
						"<critical_curve_x> <critical_curve_y> <caustic_x> <caustic_y>\n";
				else if (words[1]=="plotgrid")
					cout << "plotgrid [file]\n\n"
						"Plots recursive grid (if one has been created) to <file>, or to the screen using gnuplot if no filename\n"
						"is specified. The (x,y) coordinates of the corners of grid cells are plotted, along with the critical\n"
						"curves.\n";
				else if (words[1]=="plotlogkappa")
					cout << "plotlogkappa [file]\n\n"
						"Plots a colormap of the total log(kappa), along with log(kappa) contours, either to the screen (if no\n"
						"filename is specified) or to a file. If terminal is set to 'text' and a filename is given, outputs\n"
						"log(kappa) map to file. The number of pixels is set by 'img_npixels'. (Note: in pseudo-elliptical\n"
						"models, it is possible for kappa to have negative values in some places. In this case, a warning is\n"
						"produced, and then the log of the absolute value of kappa is plotted.\n";
				else if (words[1]=="plotlogmag")
					cout << "plotlogmag [file]\n\n"
						"Plots a colormap of log(magnification), along with contours, either to the screen (if no filename is\n"
						"specified) or to a file. If terminal is set to 'text' and a filename is given, outputs log(mag) map\n"
						"to file. The number of pixels is set by 'img_npixels'.\n";
				else if (words[1]=="findimg")
					cout << "findimg <source_x> <source_y>\n\n"
						"Finds the set of images corresponding to given source position. The image data is\n"
						"written as follows:\n"
						"<x_position>  <y_position>  <magnification>  <time delay (optional)>\n";
				else if (words[1]=="plotimg")
					cout << "plotimg <source_x> <source_y> [imagepic_file] [sourcepic_file] [grid]\n\n"
						"Plots the set of images corresponding to given source position, together with the critical\n"
						"curves. If no filename arguments are given, plots to the screen in a separate window.\n"
						"The resulting image information is also\n"
						"written to the screen as follows:\n"
						"<x_position>  <y_position>  <magnification>  <time delay (optional)>\n\n"
						"OPTIONAL: the arguments to plotimgs can be followed with terms in brackets [#:#][#:#]\n"
						"specifying the plotting range for the x and y axes, respectively. A range is allowed\n"
						"for both the source and image plots. Two examples in postscript mode:\n\n"
						"plotimg source.ps images.ps [-5:5][-5:5] [-15:15][-15:15]\n"
						"plotimg source.ps images.ps [][] [0:15][0:15]\n\n"
						"In the first example a range is specified for both the x/y axes for both plots,\n"
						"whereas in the second example a range is specified only for the image plot.\n\n"
						"If the argument 'grid' is added, plots the grid used for searching along with the images.\n\n";
				else if (words[1]=="findimgs")
					cout << "findimgs [source_file] [images_file]\n"
						"findimgs [source_file]\n\n"
						"Finds images corresponding to source positions specified in [source_file] (default = \n"
						"'sourcexy.in') and writes images to [images_file] (default = 'images.dat').\n"
						"The image data is written as follows:\n"
						"<x_position>  <y_position>  <magnification>  <time delay (optional)>\n";
				else if (words[1]=="plotimgs")
					cout << "plotimgs [source_infile]\n"
						"plotimgs <sourcepic_outfile> <imagepic_outfile>                 (postscript/PDF mode)\n"
						"plotimgs <source_infile> <sourcepic_outfile> <imagepic_outfile> (postscript/PDF mode)\n\n"
						"Plots images graphically from source positions specified in [source_infile] (default = \n"
						"'sourcexy.in'). If no output filename arguments are given, the sources and images are\n"
						"plotted to the screen (along with the caustics and critical curves if show_cc is on).\n\n"
						"OPTIONAL: the arguments to plotimgs can be followed with terms in brackets [#:#][#:#]\n"
						"specifying the plotting range for the x and y axes, respectively. A range is allowed\n"
						"for both the source and image plots. Two examples in postscript mode:\n\n"
						"plotimgs sources.ps images.ps [-5:5][-5:5] [-15:15][-15:15]\n"
						"plotimgs sources.ps images.ps [][] [0:15][0:15]\n\n"
						"In the first example a range is specified for both the x/y axes for both plots,\n"
						"whereas in the second example a range is specified only for the image plot.\n\n";
				else if (words[1]=="replotimgs")
					cout << "replotimgs\n"
						"replotimgs <sourcepic_outfile> <imagepic_outfile>               (postscript/PDF mode)\n"
						"Plots images graphically that were previously found and plotted by the 'plotimgs'\n"
						"command. This allows one to tweak plot parameters (point size, range, show_cc etc.)\n"
						"without having to calculate all the image positions again.\n\n"
						"OPTIONAL: the arguments to replotimgs can be followed with terms in brackets [#:#][#:#]\n"
						"specifying the plotting range for the x and y axes, respectively. A range is allowed\n"
						"for both the source and image plots. Two examples in postscript mode:\n\n"
						"replotimgs sources.ps images.ps [-5:5][-5:5] [-15:15][-15:15]\n"
						"replotimgs sources.ps images.ps [] [0:15][0:15]\n\n"
						"In the first example a range is specified for both the x/y axes for both plots,\n"
						"whereas in the second example a range is specified only for the image plot.\n\n";
				else if ((words[1]=="ptsize") or (words[1]=="ps"))
					cout << "ptsize <size> (or, shorter form: ps <size>)\n\n"
						"Sets point size for plotting in gnuplot.\n";
				else if ((words[1]=="pttype") or (words[1]=="pt"))
					cout << "pttype <type> (or, shorter form: pt <size>)\n\n"
						"Sets point marker type (given by an integer) for plotting in gnuplot.\n";
				else if (words[1]=="show_cc")
					cout << "show_cc <on/off>\n\n"
						"If on, plots critical curves along with image positions; otherwise, omits critical curves.\n"
						"(default=on)\n";
				else if (words[1]=="show_srcplane")
					cout << "show_srcplane <on/off>\n\n"
						"If on, plots the source along with the lensed images when the 'plotimg' or 'plotimgs' commands\n"
						"are used. (default=on)\n";
				else if (words[1]=="plot_title")
					cout << "plot_title <title>\n\n"
						"Set the title of a plot produced by the 'plotimg' or 'plotimgs' command. You can enter the title\n"
						"with surrounding single or double quotes if desired; these will simply be removed in the title\n"
						"that gets printed. (For example: plot_title \"Hello here is a title\")\n";
				else if (words[1]=="chain_info")
					cout << "chain_info <info>\n\n"
						"Sets a description of the job being run that will be stored as a comment at the top of the chain\n"
						"data file produced by nested sampling or MCMC. This description will also be stored in the file\n"
						"<label>.chain_info in the chains directory when the command 'fit mkposts' is executed. The info can\n"
						"be surrounded by optional quotation marks. (For example: chain_info \"Hello here is info\")\n";
				else if (words[1]=="data_info")
					cout << "data_info <info>\n\n"
						"Contains a description of the data being fit to that is stored in the FITS file header for mock\n"
						"data images that are created in qlens. When the FITS file is loaded later in qlens, the data\n"
						"description is automatically stored into 'data_info'. The data_info is also stored as a comment in\n"
						"the header of chain data files created by qlens. When entering data_info manually, the text can be\n"
						"be surrounded by optional quotation marks. (For example: data_info \"Hello here is info\")\n";
				else if (words[1]=="param_markers")
					cout << "param_markers <marker1> <marker2> ...\n"
						"param_markers allparams\n\n"
						"Define parameter values to be marked in posteriors plotted by mkdist tool (using the '-m' option in\n"
						"mkdist). The marker values will be saved to the file '<label>.markers' in the chains directory when a\n"
						"chain is produced, where <label> is set by the 'fit label' command. The marker values will also be\n"
						"saved to this file and plotted automatically when executing the 'fit mkposts' command. If a FITS file\n"
						"is created, the parameter markers are stored in the FITS file header and will be automatically loaded\n"
						"back into 'param_markers' if the FITS file is loaded into qlens later using 'sbmap loadimg'.\n\n"
						"If the argument is entered as 'allparams', current values of all the fit parameters and derived\n"
						"parameters are used. If parameter values are entered in manually, keep in mind that qlens does not\n"
						"check whether the number of marker values entered matches the number of parameters. In addition, the\n"
						"names of specific parameters (whether fit or derived parameters) can be entered, in which case the\n"
						"names will be replaced by their current numeric values.\n"
						"(For example: 'param_markers b q theta', or 'param_markers 1 0.7 30 xc yc'.)\n";
				else if (words[1]=="subplot_params")
					cout << "subplot_params <param_names> ...\n"
						"subplot_params reset\n\n"
						"Define a subset of parameters that will be used to make a sub-triangle plot when the 'fit mkposts'\n"
						"command is run. Simply list the parameter names in the arguments or 'reset' to clear the list.\n"
						"If no argument is given, the current list of subplot parameters is printed.\n";
				else if (words[1]=="mksrctab")
					cout << "mksrctab <xmin> <xmax> <xpoints> <ymin> <ymax> <ypoints> [source_outfile]\n\n"
						"Creates a regular grid of sources and plots to [source_outfile] (default='sourcexy.in')\n";
				else if (words[1]=="mksrcgal")
					cout << "mksrcgal <xcenter> <ycenter> <a> <q> <angle> <n_ellipses> <pts_per_ellipse> [outfile]\n\n"
						"Creates an elliptical grid of sources (representing a galaxy) with major axis a, axis ratio q.\n"
						"The grid is a series of concentric ellipses, plotted to [outfile] (default='sourcexy.in').\n";
				else if (words[1]=="mkrandsrc")
					cout << "mkrandsrc <N> [source_outfile]   (supported only in ccspline mode)\n\n"
						"Plots N sources randomly within the curve circumscribing the outermost caustic(s).\n"
						"Sources are plotted to [source_outfile] (default='sourcexy.in')\n";
				else if (words[1]=="plotkappa")
					cout << "plotkappa <rmin> <rmax> <steps> <kappa_file> [dkappa_file] [lens=#]\n\n"
						"Plots the radial kappa profile and its (optional) derivative to <kappa_file> and [dkappa_file]\n"
						"respectively. If the 'lens=#' argument is not given, the total kappa profile (averaged over all\n"
						"angles) is plotted, with r=0 chosen to be at the center of the primary lens (which is set by\n"
						"'primary_lens'). Any lens models which are not centered on the origin are ignored. In addition\n"
						"to the kappa, the radius in kpc and density in solar masses per kpc^2 are given in the third and\n"
						"fourth columns, i.e. the output file has the following format:\n\n"
						"r(arcsec) kappa r(kpc) Sigma(M_sol/kpc^2)\n\n"
						"If a specific lens is chosen by the 'lens=#' argument, only the kappa profile\n"
						"for that lens is plotted; however, in addition, the average kappa, deflection, and enclosed mass\n"
						"are plotted (as additional columns in the file <kappa_file>) for that lens.\n"
						"(NOTE: Only text mode plotting is supported for this command.)\n";
				else if (words[1]=="plotmass")
					cout << "plotmass <rmin> <rmax> <steps> <mass_file>\n\n"
						"Plots the radial mass profile to <mass_file>. The columns in the output file are radius in\n"
						"arcseconds, mass in m_solar, and radius in kpc respectively.\n"
						"NOTE: Only text mode plotting is supported for this feature.\n";
				else if (words[1]=="printcs")
					cout << "printcs [filename]\n\n"
						"Prints the total (unbiased) cross section. If [filename] is specified, saves to file.\n";
				else if (words[1]=="clear")
					cout << "clear <#>\n\n"
						"Remove lens galaxy # from the list (the list of galaxies can be printed using the\n"
						"'lens' command). If no arguments given, delete entire lens configuration and start over.\n";
				else if (words[1]=="cc_reset")  // obsolete--should probably remove
					cout << "cc_reset -- (no arguments) delete the current critical curve spline\n"; // obsolete--should probably remove
				else if (words[1]=="integral_method")
					cout << "integral_method <method> [npoints]\n\n"
						"Set integration method (either 'patterson', 'romberg' or 'gauss') which is used for lens models\n"
						"where numerical quadrature is required for the lensing calculations. See below for a description of\n"
						"of each method. If gauss is selected, can set number of points = [npoints]. For patterson or romberg,\n"
						"which are both adaptive quadrature methods, the required error tolerance is controlled by the setting\n"
						"'integral_tolerance'. If no arguments are given, prints current integration method.\n\n"
						"\033[4mpatterson\033[0m: Gauss-Patterson quadrature is the fastest integration method in Qlens that is adapt-\n"
						"   ive, meaning the integral can be refined iteratively as more points are added until the desired\n"
						"   tolerance is achieved. It is based on the Kronrod-Patterson rules which start from Gauss-Legendre\n"
						"   quadrature and successively add points to achieve higher order, giving nested quadrature rules.\n"
						"   The maximum number of allowed points is 512, so Qlens will give a warning if the tolerance has\n"
						"   not been reached after reaching 512 points (this may happen for very steep density profiles).\n\n"
						"\033[4mromberg\033[0m: Romberg integration uses Newton-Cotes rules of successively higher orders and\n"
						"   generates an estimated value and error using Richardson extrapolation. Generally a lot more\n"
						"   points are required compared to 'patterson' or 'gauss' to achieve the same accuracy, but in\n"
						"   principle there is no limit to the number of points that can be used to achieve the desired\n"
						"   integral tolerance.\n\n"
						"\033[4mgauss\033[0m: Gaussian quadrature is a non-adaptive method that uses a fixed number of points, and hence\n"
						"   is not recommended in general for lens model fitting although it can be useful for diagnostic\n"
						"   purposes, e.g. testing a new lens model.\n\n";
				else if (words[1]=="integral_tolerance")
					cout << "integral_tolerance <tolerance>\n\n"
						"Set tolerance limit for numerical integration, specifically for the adaptive quadrature methods\n"
						"'patterson' (Gauss-Patterson quadrature) and 'romberg' (Romberg integration). For patterson, the\n"
						"integration stops when the difference between successive quadrature estimates is smaller than the\n"
						"specified tolerance. Note that the tolerance is thus the estimated error in second-to-last iteration,\n"
						"and hence the error in the final estimate is usually a great deal smaller than the specified tol-\n"
						"erance. For romberg, the error returned using Richardson extrapolation is required to be less than\n"
						"the specified tolerance.\n";
				else if (words[1]=="major_axis_along_y")
					cout << "major_axis_along_y <on/off>\n\n"
						"Specify whether to orient major axis of lenses along the y-direction (if on) or x-direction\n"
						"if off (this is on by default, in accordance with the usual convention).\n";
				else if (words[1]=="warnings")
					cout << "warnings <on/off>\n"
						"Set warnings on/off. Warnings are given if, e.g.:\n"
						"  --an image of the wrong parity is discovered\n"
						"  --critical curves could not be located, etc.\n";
				else if (words[1]=="imgsrch_warnings")
					cout << "imgsrch_warnings <on/off>\n"
						"imgsrch_warnings <on/off>\n\n"
						"Warnings related to the root finding routine (Newton's Method) used to locate images. This includes warnings\n"
						"if a duplicate image is found (where the condition for duplicates is set by 'imgsep_threshold', if the root\n"
						"finder converges to a local minimum rather than a root, if a probable false root was found, etc.\n";
				else if (words[1]=="sci_notation")
					cout << "sci_notation <on/off>\n\n"
						"Display results in scientific notation (default=on).\n\n";
				else if (words[1]=="gridtype")
					cout << "gridtype [radial/cartesian]\n\n"
						"Set the grid for image searches to radial or Cartesian (default=radial). If the gridtype is\n"
						"Cartesian, the the number of initial splittings is set by the xsplit and ysplit commands,\n"
						"rather than rsplit and thetasplit.\n";
				else if (words[1]=="emode")
					cout << "emode [#]\n\n"
						"The ellipticity mode (which can be either 0,1,2, or 3) controls how ellipticity is introduced\n"
						"into the lens models. In modes 0-2, ellipticity is introduced into the projected density (kappa),\n"
						"whereas in mode 3, ellipticity is introduced into the potential (called a pseudo-elliptic\n"
						"model). The ellipticity is parameterized by the axis ratio q in modes 0,1, and by the\n"
						"ellipticity parameter in modes 2,3. The 'emode' command changes the default ellipticity mode for\n"
						"lens models that get created, but you can also specify the ellipticity mode of a specific lens\n"
						"you create by adding the argument 'emode=#' to the line (e.g., 'lens nfw emode=3 0.8 20 0.3').\n"
						"Below we describe each mode in detail.\n\n"
						"Mode 0: in kappa(R), we let R^2 --> x^2 + (y/q)^2. Ellipticity parameter: q\n"
						"Mode 1: in kappa(R), we let R^2 --> qx^2 + y^2/q. Ellipticity parameter: q  (qlens default mode)\n"
						"Mode 2: in kappa(R), we let R^2 --> (1-e)*x^2 + (1+e)*y^2. Ellipticity parameter: e (epsilon)\n"
						"Mode 3: in potential(R), we let R^2 --> (1-e)*x^2 + (1+e)*y^2. Ellipticity parameter: e (epsilon)\n\n"
						"If a lens is created using mode 3, the prefix 'pseudo-' is added to the lens model name. The\n"
						"pseudo-elliptical model can do lens calculations significantly faster in most of the elliptical\n"
						"mass models, since analytic formulas are used for the deflection. Exceptions are the pjaffe\n"
						"model and the alpha model in the case where s=0 or alpha=1, since in these cases the formulas\n"
						"are analytic regardless. Keep in mind however that the pseudo-elliptical models can lead to\n"
						"unphysical density contours when the ellipticity is high enough (you can check this using the\n"
						"command 'plotlogkappa').\n";
				else if (words[1]=="ccspline")
					cout << "ccspline <on/off>\n\n"
						"Set critical curve spline mode on/off (if no arguments given, prints current setting.\n"
						"Critical curve spline mode is appropriate for lenses with circular or elliptical\n"
						"symmetry (or close to it), where exactly two critical curves exist with roughly the\n"
						"same symmetry; it is not appropriate when substructure is included and should be\n"
						"turned off in this case.\n";
				else if (words[1]=="auto_ccspline")
					cout << "auto_ccspline <on/off>\n\n"
						"Set automatic critical curve spline mode on/off (default=on). When on, the critical\n"
						"curves are splined (i.e. ccspline is turned on) only if elliptical symmetry is present,\n"
						"i.e. all lenses are centered at the origin. If elliptical symmetry is not present, the\n"
						"critical curves are not splined.\n";
				else if (words[1]=="autocenter")
					cout << "autocenter [on/off]\n\n"
						"Automatically center the grid on the center of the 'primary' lens (if on), which is set by\n"
						"the command 'primary_lens'.\n"
						"in the list of specified lenses shown by typing 'lens'). By default this is set to 'on'. If\n"
						"instead set to 'off', the grid does not set its center automatically and defaults to (0,0)\n"
						"unless changed manually by the 'grid center' command. 'autocenter' is also automatically\n"
						"turned off if the grid dimensions are set manually.\n\n";
				else if (words[1]=="primary_lens")
					cout << "primary_lens <lens_number>\n"
						"primary_lens auto\n\n"
						"Set the primary lens to correspond to the specified lens number (as displayed in the list of\n"
						"specified lenses shown by typing 'lens'). If set to 'auto' (which is the default), the\n"
						"primary lens is automatically chosen to be the lens with the largest Einstein radius. Among\n"
						"other things, the primary lens determines the grid center if 'autocenter' is on; its center\n"
						"is chosen as the origin when the total radial kappa profile (averaged over angles) is plotted.\n\n";
				else if (words[1]=="autogrid_from_Re")
					cout << "autogrid_from_Re <on/off>\n\n"
						"Automatically set the grid size from the Einstein radius of primary lens before grid\n"
						"is created (if on). This is much faster than the more generic 'autogrid' function and is\n"
						"therefore recommended during model fitting ('autogrid' is slower because it searches for\n"
						"the critical curves of the entire lens configuration), but may not work well if the lens\n"
						"has a very low axis ratio or is very strongly perturbed by external shear.\n\n";
				else if (words[1]=="autogrid_before_mkgrid")
					cout << "autogrid_before_mkgrid <on/off>\n\n"
						"Automatically set the grid size from critical curves using the 'autogrid' function before\n"
						"grid is created (if on). This is quite robust, but is much slower compared to the\n"
						"alternative 'autogrid_from_Re' function. Therefore, during model fitting it is recommended\n"
						"to use 'autogrid_from_Re' (if the lens is not too asymmetric) or else fix the grid size based\n"
						"on the observed image configuration.\n\n";
				else if (words[1]=="chisqmag")
					cout << "chisqmag <on/off>\n\n"
						"Use magnification in chi-square function for image positions (if on). Note that this is\n"
						"only relevant for the source plane chi-square (i.e. imgplane_chisq is set to 'off').\n";
				else if (words[1]=="chisqflux")
					cout << "chisqflux <on/off>\n\n"
						"Include the image fluxes in the chi-square function (if on). Note that this is only relevant\n"
						"for point source searches (i.e. fit source_mode = ptsource). (default=off)\n";
				else if (words[1]=="chisqtol")
					cout << "chisqtol <##>\n\n"
						"Set the required accuracy for chi-square minimization (if Powell's method or the downhill\n"
						"simplex methods are used.\n";
				else if (words[1]=="chisq_parity")
					cout << "chisq_parity <on/off>\n\n"
						"Include parity information in addition to flux in the chi-square function (if on). Note\n"
						"that this is only relevant for point source searches and if chisqflux is on. (default=off)\n";
				else if (words[1]=="chisq_time_delays")
					cout << "chisq_time_delays <on/off>\n\n"
						"Include time delays in the chi-square function (if on). Note that this is only relevant for\n"
						"point source searches (i.e. fit source_mode = ptsource). (default=off)\n";
				else if (words[1]=="imgplane_chisq")
					cout << "imgplane_chisq <on/off>\n\n"
						"Use the lensed image positions in the chi-square function for fitting (if on); otherwise,\n"
						"use the source positions obtained by mapping the data images to the source plane.\n";
				else if (words[1]=="fix_srcflux")
					cout << "fix_srcflux <on/off>\n\n"
						"Fix source flux to specified value 'srcflux' during fit, rather than optimizing it during\n"
						"the chi-square evaluation. (default=off)\n";
				else if (words[1]=="srcflux")
					cout << "srcflux <#>\n\n"
						"Flux of source point, used when creating simulated data using 'imgdata add' command. Also, if\n"
						"fix_srcflux is set to 'on', source flux is fixed to this value during model fitting, rather\n"
						"than optimizing it during the chi-square evaluation. (default=1.0)\n";
				else if (words[1]=="central_image")
					cout << "central_image <on/off>\n\n"
						"Include central images when fitting to image data (if on).\n";
				else if (words[1]=="time_delays")
					cout << "time_delays <on/off>\n\n"
						"Calculate time delay for each image (with earliest image given a time delay of zero).\n";
				else if ((words[1]=="rsplit") and (radial_grid))
					cout << "rsplit <#>\n\n"
						"Sets initial number of grid rows in the radial direction. If no arguments are given,\n"
						"prints the current rsplit value.\n";
				else if ((words[1]=="xsplit") and (!radial_grid))
					cout << "xsplit <#>\n\n"
						"Sets initial number of grid rows in the x-direction. If no arguments are given,\n"
						"prints the current xsplit value.\n";
				else if ((words[1]=="thetasplit") and (radial_grid))
					cout << "thetasplit <#>\n\n"
						"Sets initial number of grid columns in the angular direction. If no arguments are given,\n"
						"prints the current thetasplit value.\n";
				else if ((words[1]=="ysplit") and (!radial_grid))
					cout << "ysplit <#>\n\n"
						"Sets initial number of grid columns in the y-direction. If no arguments are given,\n"
						"prints the current ysplit value.\n";
				else if (words[1]=="cc_splitlevels")
					cout << "cc_splitlevels <#>\n\n"
						"Sets the number of times that cells containing the critical curves are recursively\n"
						"split. If no arguments are given, prints the current cc_splitlevels value.\n";
				else if (words[1]=="imgpos_accuracy")
					cout << "imgpos_accuracy <#>\n\n"
						"Sets the accuracy in the image position (x- and y-coordinates) required for Newton's\n"
						"method. If no arguments are given, prints the current accuracy setting.\n";
				else if (words[1]=="imgsrch_mag_threshold")
					cout << "imgsrch_mag_threshold <#>\n\n"
						"Sets the magnification threshold, such that a warning is printed is images are found with\n"
						"magnifications greater than this threshold. If 'reject_himag' is turned on, any images found\n"
						"above the threshold are discarded during the image searching. In most cases, magnifications\n"
						"above ~1000 carry a high risk of phantom images or inaccuracies in the root finder algorithm,\n"
						"often creating duplicate images, due to their proximity to a critical curve. Turning on\n"
						"'reject_himag' can alleviate this issue, however if the lens is very symmetric and the source's\n"
						"project position is near the center of the lens, imgsrch_mag_threshold must be raised since\n"
						"the expected magnifications are high (increasing cc_splitlevels will reduce the possibility\n"
						"of missing an image in this situation). During fitting, duplicate images can also be dealt\n"
						"with by using 'chisq_imgsep_threshold' to discard duplicates.\n";
				else if (words[1]=="reject_himag")
					cout << "reject_himag <on/off>\n"
						"If is turned on, any images found with magnifications higher than 'imgsrch_mag_threshold' are\n"
						"discarded during the image searching. See 'help imgsrch_mag_threshold' for more information\n"
						"on the use of these commands and their pros/cons in lens modeling.\n";
				else if (words[1]=="min_cellsize")
					cout << "min_cellsize <#>\n\n"
						"Specifies the minimum (average) length a cell can have and still be split (e.g. around\n"
						"critical curves or for perturber subgridding). For lens modelling, this should be\n"
						"comparable to or smaller than the resolution (in terms of area) of the image in question.\n";
				else if (words[1]=="galsub_radius")
					cout << "galsub_radius <#>\n\n"
						"When subgridding around perturbing galaxies, galsub_radius scales the maximum radius away from\n"
						"the center of each perturber within which cells are split. For each perturber, its Einstein\n"
						"radius along with shear, kappa, and parity information are used to determine the optimal\n"
						"subgridding radius for each perturber; galsub_radius can be used to scale these subgridding\n"
						"radii by the specified factor.\n";
				else if (words[1]=="galsub_min_cellsize")
					cout << "galsub_min_cellsize <#>\n\n"
						"When subgridding around perturbing galaxies, galsub_min_cellsize specifies the\n"
						"minimum allowed (average) length of subgrid cells in terms of the fraction of\n"
						"the Einstein radius of a given perturber. Note that this does *not* include\n"
						"the subsequent splittings around critical curves (see galsub_cc_splitlevels).\n";
				else if (words[1]=="galsub_cc_splitlevels")
					cout << "galsub_cc_splitlevels <#>\n\n"
						"For subgrid cells that are created around perturbing galaxies, this sets the number\n"
						"of times that cells containing critical curves are recursively split. Note that not\n"
						"all splittings may occur if the minimum cell size (set by min_cellsize) is reached.\n";
				else if (words[1]=="rmin_frac")
					cout << "rmin_frac <#>\n\n"
						"Set minimum radius of innermost cells in grid (in terms of fraction of max radius);\n"
						"this defaults to " << default_rmin_frac << ".\n";
				else if (words[1]=="galsubgrid")
					cout << "galsubgrid <on/off>   (default=on)\n\n"
						"When turned on, subgrids around perturbing galaxies (defined as galaxies with Einstein\n"
						"radii less than " << perturber_einstein_radius_fraction << " times the largest Einstein radius) when a new grid is created.\n";
				else if (words[1]=="fits_format")
					cout << "fits_format <on/off> (default=on)\n\n"
						"It on, surface brightness maps are loaded from FITS files (using 'sbmap loadimg'). If off,\n"
						"maps are loaded from files in text format. See 'help sbmap loadimg' for more information on\n"
						"the required format for loading surface brightness maps in text format.\n";
				else if (words[1]=="data_pixel_size")
					cout << "data_pixel_size <##>\n\n"
						"Set the pixel size for files in FITS format. If no size is specified, the pixel grid will\n"
						"assume the same dimensions that are set by the 'grid' command.\n";
				else if (words[1]=="raytrace_method")
					cout << "raytrace_method <method>\n\n"
						"Set the method for ray tracing image pixels to source pixels (using the 'sbmap' commands.\n"
						"Available methods are:\n\n"
						"interpolation -- interpolate surface brightness using linear interpolation in the three nearest\n"
						"                 source pixels.\n"
						"direct -- surface brightness of nearest source pixel is used.\n"
						"overlap -- after ray-tracing a pixel to the source plane, find the overlap area with all source\n"
						"           pixels it overlaps with and weight the surface brightness accordingly.\n";
				else if (words[1]=="img_npixels")
					cout << "img_npixels <npixels_x> <npixels_y>\n\n"
						"Set the number of pixels, along x and y, for plotting image surface brightness maps (using 'sbmap\n"
						"plotimg'). If no arguments are given, prints the current image pixel dimensions.\n"
						"(To set the image grid size and location, use the 'grid' command.)\n";
				else if (words[1]=="src_npixels")
					cout << "src_npixels <npixels_x> <npixels_y>\n\n"
						"Set the number of pixels, along x and y, for producing a pixellated source grid using the 'sbmap\n"
						"makesrc' or 'sbmap invert' command, or doing a model fit with a pixellated source. Note that if\n"
						"the number of source pixels is changed manually, 'auto_src_npixels' is automatically turned off.\n"
						"If no arguments are given, prints the current source pixel setting. (To set the source grid size\n"
						"and location, use the 'srcgrid' command.)\n";
				else if (words[1]=="data_pixel_noise")
					cout << "data_pixel_noise <noise>\n\n"
						"Set the pixel noise expected in the image pixel data (loaded using 'sbmap loadimg'), which is\n"
						"the dispersion in the surface brightness of each pixel. At present, QLens assumes the noise to be\n"
						"uncorrelated and the same for all pixels; in a later version it will be possible to input a full\n"
						"covariance matrix.\n";
				else if (words[1]=="sim_pixel_noise")
					cout << "sim_pixel_noise <noise>\n\n"
						"Sets simulated pixel noise to be added to lensed pixel images (produced by the 'sbmap plotimg'\n"
						"command), which is the dispersion in surface brightness of each pixel.\n";
				else if (words[1]=="psf_width")
					cout << "psf_width <width>\n"
						"psf_width <x_width> <y_width>\n\n"
						"Set width of point spread function (PSF) along x- and y-axes, where the PSF is modeled as\n"
						"Gaussian and the width is defined here as the dispersion along a given axis. When producing\n"
						"lensed pixel images (e.g. 'sbmap plotimg' or 'sbmap invert'), the resulting image is convolved\n"
						"with this PSF. If only one argument is given, the PSF is assumed to be symmetric and the same\n"
						"width is given along both axes.\n";
				else if (words[1]=="regparam")
					cout << "regparam <R0>\n"
						"regparam <Rmin> <R0> <Rmax>\n\n"
						"Value of regularization parameter for inverting lensed pixel images. If no argument is given, prints\n"
						"the current value. If the regularization parameter will be varied when model fitting ('vary_regparam'\n"
						"set to on), <R0> gives the initial regularization parameter. If the fit method requires upper and\n"
						"lower limits (e.g. nested sampling or MCMC), then <Rmin>, <R0>, and <Rmax> must all be specified\n"
						"before performing a fit.\n";
				else if (words[1]=="vary_regparam")
					cout << "vary_regparam <on/off>\n\n"
						"Specify whether to vary the regularization parameter when fitting to a pixel image. This option can only be\n"
						"set to 'on' if the source mode is set to 'pixel' (using 'fit source_mode') and if a regularization method is\n"
						"set using the 'fit regularization' command. If set to 'on', the initial value and upper/lower limits (if\n"
						"relevant) are listed by the 'fit' command. (default=off)\n";
				else if (words[1]=="sb_threshold")
					cout << "sb_threshold <threshold>\n\n"
						"Set minimum surface brightness required to include in calculation of image centroid (see the 'imgdata\n"
						"add_from_centroid' command for information on how to do this). (default=0)\n";
				else if ((words[1]=="terminal") or (words[1]=="term"))
					cout << "terminal <text/ps/pdf>\n\n"
						"Set plot output mode when plotting to files (text, postscript or PDF). (Shorthand: 'term').\n";
				else if ((words[1]=="quit") or (words[1]=="exit") or (words[1]=="q"))
					cout << words[1] << " -- (no arguments) exit qlens.\n";
				else Complain("command not recognized");
			}
		}
		else if ((words[0]=="settings") or (words[0]=="plot_settings") or (words[0]=="imgsrch_settings") or (words[0]=="sbmap_settings") or (words[0]=="fit_settings") or (words[0]=="cosmo_settings") or (words[0]=="lens_settings") or (words[0]=="misc_settings"))
		{
			if (mpi_id==0) {
				bool show_plot_settings = false;
				bool show_imgsrch_settings = false;
				bool show_sbmap_settings = false;
				bool show_fit_settings = false;
				bool show_cosmo_settings = false;
				bool show_lens_settings = false;
				bool show_misc_settings = false;
				if (words[0]=="plot_settings") show_plot_settings = true;
				else if (words[0]=="imgsrch_settings") show_imgsrch_settings = true;
				else if (words[0]=="sbmap_settings") show_sbmap_settings = true;
				else if (words[0]=="fit_settings") show_fit_settings = true;
				else if (words[0]=="cosmo_settings") show_cosmo_settings = true;
				else if (words[0]=="lens_settings") show_lens_settings = true;
				else if (words[0]=="misc_settings") show_misc_settings = true;
				else if (words[0]=="settings") {
					show_plot_settings = true;
					show_imgsrch_settings = true;
					show_sbmap_settings = true;
					show_fit_settings = true;
					show_cosmo_settings = true;
					show_lens_settings = true;
					show_misc_settings = true;
				}
				bool setting;
				int intval;
				double doubleval;

				cout << endl;
				if (show_plot_settings) {
					cout << "\033[4mGraphical plot settings\033[0m\n";
					cout << "terminal: " << ((terminal==TEXT) ? "text\n" : (terminal==POSTSCRIPT) ? "postscript\n" : (terminal==PDF) ? "PDF\n" : "Unknown terminal\n");
					cout << "show_cc: " << display_switch(show_cc) << endl;
					cout << "show_srcplane: " << display_switch(plot_srcplane) << endl;
					cout << resetiosflags(ios::scientific);
					cout << "ptsize = " << plot_ptsize << endl;
					cout << "pttype = " << plot_pttype << endl;
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
					if (plot_title.empty()) cout << "plot_title: none\n";
					else cout << "plot_title: '" << plot_title << "'\n";
					cout << "plot_square_axes: " << display_switch(plot_square_axes) << endl;
					if (mpi_id==0) {
						cout << resetiosflags(ios::scientific);
						cout << "fontsize = " << fontsize << endl;
						if (use_scientific_notation) cout << setiosflags(ios::scientific);
					}
					cout << resetiosflags(ios::scientific);
					cout << "linewidth = " << linewidth << endl;
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
					cout << "plot_key: " << display_switch(show_plot_key) << endl;
					cout << "plot_key_outside: " << display_switch(plot_key_outside) << endl;
					cout << "colorbar: " << display_switch(show_colorbar) << endl;
					cout << endl;
				}
				if (show_imgsrch_settings) {
					cout << "\033[4mImage search grid settings\033[0m\n";
					if (radial_grid) cout << "gridtype: radial" << endl;
					else cout << "gridtype: Cartesian" << endl;
					cout << "autogrid_from_Re: " << display_switch(auto_gridsize_from_einstein_radius) << endl;
					cout << "autogrid_before_mkgrid: " << display_switch(autogrid_before_grid_creation) << endl;
					cout << "autocenter: ";
					if (autocenter==false) cout << "off\n";
					else cout << "centers on lens " << primary_lens_number << endl;
					cout << "central_image: " << display_switch(include_central_image) << endl;
					cout << "time_delays: " << display_switch(include_time_delays) << endl;
					cout << "min_cellsize = " << sqrt(min_cell_area) << endl;
					cout << resetiosflags(ios::scientific);
					if (radial_grid) {
						cout << "rsplit = " << usplit_initial << endl;
						cout << "thetasplit = " << wsplit_initial << endl;
					} else {
						cout << "xsplit = " << usplit_initial << endl;
						cout << "ysplit = " << wsplit_initial << endl;
					}
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
					cout << "cc_splitlevels = " << cc_splitlevels << endl;
					cout << "cc_split_neighbors: " << display_switch(cc_neighbor_splittings) << endl;
					cout << "imgpos_accuracy = " << Grid::image_pos_accuracy << endl;
					cout << "imgsep_threshold = " << redundancy_separation_threshold << endl;
					cout << "imgsrch_mag_threshold = " << newton_magnification_threshold << endl;
					cout << "reject_himag: " << display_switch(reject_himag_images) << endl;
					cout << "rmin_frac = " << rmin_frac << endl;
					cout << "galsubgrid: " << display_switch(subgrid_around_perturbers) << endl;
					cout << "galsub_radius = " << galsubgrid_radius_fraction << endl;
					cout << "galsub_min_cellsize = " << galsubgrid_min_cellsize_fraction << " (units of Einstein radius)" << endl;
					cout << "galsub_cc_splitlevels = " << galsubgrid_cc_splittings << endl;
					cout << "ccspline: " << display_switch(use_cc_spline) << endl;
					cout << "auto_ccspline: " << display_switch(auto_ccspline) << endl;
					cout << endl;
				}
				if (show_sbmap_settings) {
					cout << "\033[4mSurface brightness pixel map settings\033[0m\n";
					cout << "fits_format: " << display_switch(fits_format) << endl;
					cout << "img_npixels: (" << n_image_pixels_x << "," << n_image_pixels_y << ")\n";
					cout << "src_npixels: (" << srcgrid_npixels_x << "," << srcgrid_npixels_y << ")";
					if (auto_srcgrid_npixels) cout << " (auto_src_npixels on)";
					cout << endl;
					cout << "srcgrid: (" << sourcegrid_xmin << "," << sourcegrid_xmax << ") x (" << sourcegrid_ymin << "," << sourcegrid_ymax << ")";
					if (auto_sourcegrid) cout << " (auto_srcgrid on)";
					cout << endl;
					cout << "raytrace_method: " << ((ray_tracing_method==Area_Overlap) ? "area overlap\n" : ((ray_tracing_method==Interpolate) and (interpolate_sb_3pt)) ? "linear 3-point interpolation\n" : ((ray_tracing_method==Interpolate) and (!interpolate_sb_3pt)) ? "direct (nearest source pixel used)\n" : "unknown\n");
					cout << "sim_pixel_noise = " << sim_pixel_noise << endl;
					cout << "psf_width: (" << psf_width_x << "," << psf_width_y << ")\n";
					cout << "psf_threshold = " << psf_threshold << endl;
					cout << "psf_mpi: " << display_switch(psf_convolution_mpi) << endl;
					cout << endl;
					cout << "\033[4mSource pixel reconstruction settings\033[0m\n";
					cout << "inversion_method: " << ((inversion_method==MUMPS) ? "LDL factorization (MUMPS)\n" : (inversion_method==UMFPACK) ? "LU factorization (UMFPACK)\n" : (inversion_method==CG_Method) ? "conjugate gradient method\n" : "unknown\n");
					cout << "adaptive_grid: " << display_switch(adaptive_grid) << endl;
					cout << "auto_src_npixels: " << display_switch(auto_srcgrid_npixels) << endl;
					cout << "auto_srcgrid: " << display_switch(auto_sourcegrid) << endl;
					cout << "noise_threshold = " << noise_threshold << endl;
					if (data_pixel_size < 0) cout << "data_pixel_size: not specified\n";
					else cout << "data_pixel_size: " << data_pixel_size << endl;
					cout << "data_pixel_noise = " << data_pixel_noise << endl;
					cout << "inversion_nthreads = " << inversion_nthreads << endl;
					cout << "pixel_fraction = " << pixel_fraction << endl;
					cout << "vary_pixel_fraction: " << display_switch(vary_pixel_fraction) << endl;
					cout << "srcgrid_xshift = " << srcgrid_xshift << endl;
					cout << "vary_srcgrid_xshift: " << display_switch(vary_srcgrid_xshift) << endl;
					cout << "srcgrid_yshift = " << srcgrid_xshift << endl;
					cout << "vary_srcgrid_yshift: " << display_switch(vary_srcgrid_yshift) << endl;
					cout << "srcgrid_scale = " << srcgrid_size_scale << endl;
					cout << "vary_srcgrid_scale: " << display_switch(vary_srcgrid_size_scale) << endl;
					cout << "regparam = " << regularization_parameter << endl;
					cout << "vary_regparam: " << display_switch(vary_regularization_parameter) << endl;
					cout << "outside_sb_prior: " << display_switch(outside_sb_prior) << endl;
					cout << "outside_sb_noise_threshold = " << outside_sb_prior_noise_frac << endl;
					cout << "nimg_prior: " << display_switch(n_image_prior) << endl;
					cout << "nimg_threshold = " << n_image_threshold << endl;
					cout << "nimg_sb_threshold = " << n_image_prior_sb_frac << endl;
					cout << "subhalo_prior: " << display_switch(subhalo_prior) << endl;
					cout << "activate_unmapped_srcpixels: " << display_switch(activate_unmapped_source_pixels) << endl;
					cout << "exclude_srcpixels_outside_mask: " << display_switch(exclude_source_pixels_beyond_fit_window) << endl;
					cout << "remove_unmapped_subpixels: " << display_switch(regrid_if_unmapped_source_subpixels) << endl;
					cout << "sb_threshold = " << sb_threshold << endl;
					cout << "parallel_mumps: " << display_switch(parallel_mumps) << endl;
					cout << "show_mumps_info: " << display_switch(show_mumps_info) << endl;
					cout << "srcpixel_mag_threshold = " << pixel_magnification_threshold << endl;
					cout << "vary_srcpixel_mag_threshold: " << display_switch(vary_magnification_threshold) << endl;
					cout << endl;
				}
				if (show_fit_settings) {
					cout << "\033[4mChi-square function settings\033[0m\n";
					cout << "imgplane_chisq: " << display_switch(use_image_plane_chisq) << endl;
					cout << "chisqmag: " << display_switch(use_magnification_in_chisq) << endl;
					cout << "chisqflux: " << display_switch(include_flux_chisq) << endl;
					cout << "chisq_time_delays: " << display_switch(include_time_delay_chisq) << endl;
					cout << "chisq_parity: " << display_switch(include_parity_in_chisq) << endl;
					cout << "analytic_bestfit_src: " << display_switch(use_analytic_bestfit_src) << endl;
					cout << "chisq_mag_threshold = " << chisq_magnification_threshold << endl;
					cout << "chisq_imgsep_threshold = " << chisq_imgsep_threshold << endl;
					cout << "chisq_imgplane_threshold = " << chisq_imgplane_substitute_threshold << endl;
					cout << "nimg_penalty: " << display_switch(n_images_penalty) << endl;
					cout << "chisqtol = " << chisq_tolerance << endl;
					cout << "srcflux = " << source_flux << endl;
					cout << "fix_srcflux: " << display_switch(fix_source_flux) << endl;
					cout << "syserr_pos = " << syserr_pos << endl;
					cout << "vary_syserr_pos: " << display_switch(vary_syserr_pos_parameter) << endl;
					cout << "wl_shearfac = " << wl_shear_factor << endl;
					cout << "vary_wl_shearfac: " << display_switch(vary_wl_shear_factor_parameter) << endl;
					cout << endl;
					cout << "\033[4mOptimization and Monte Carlo sampler settings\033[0m\n";
					cout << "fit method: " << ((fitmethod==POWELL) ? "powell\n" : (fitmethod==SIMPLEX) ? "simplex\n" : (fitmethod==NESTED_SAMPLING) ? "nest\n" : (fitmethod==TWALK) ? "twalk" : (fitmethod==POLYCHORD) ? "polychord" : (fitmethod==MULTINEST) ? "multinest" : "Unknown fitmethod\n");
					cout << "fit source_mode: " << ((source_fit_mode==Point_Source) ? "ptsource\n" : (source_fit_mode==Pixellated_Source) ? "pixel\n" : (source_fit_mode==Parameterized_Source) ? "sbprofile\n" : "unknown\n");
					cout << "nrepeat = " << n_repeats << endl;
					cout << "find_errors: " << display_switch(calculate_parameter_errors) << endl;
					cout << "simplex_nmax = " << simplex_nmax << endl;
					cout << "simplex_nmax_anneal = " << simplex_nmax_anneal << endl;
					cout << "simplex_minchisq = " << simplex_minchisq << endl;
					cout << "simplex_minchisq_anneal = " << simplex_minchisq_anneal << endl;
					cout << "simplex_temp0 = " << simplex_temp_initial << endl;
					cout << "simplex_tempf = " << simplex_temp_final << endl;
					cout << "simplex_cooling_factor = " << simplex_cooling_factor << endl;
					cout << "simplex_show_bestfit: " << display_switch(simplex_show_bestfit) << endl;
					if (data_info.empty()) cout << "data_info: none\n";
					else cout << "data_info: '" << data_info << "'\n";
					if (chain_info.empty()) cout << "chain_info: none\n";
					else cout << "chain_info: '" << chain_info << "'\n";
					if (param_markers.empty()) cout << "param_markers: none\n";
					else cout << "param_markers: '" << param_markers << "'\n";
					if (!param_settings->subplot_params_defined()) cout << "subplot_params: none\n";
					else cout << "subplot_params: " << param_settings->print_subplot_params() << endl;
					cout << "n_livepts = " << n_livepts << endl;
					cout << "polychord_nrepeats = " << polychord_nrepeats << endl;
					cout << "mcmc_chains = " << mcmc_threads << endl;
					cout << "mcmctol = " << mcmc_tolerance << endl;
					cout << "mcmc_logfile: " << display_switch(mcmc_logfile) << endl;
					cout << "random_seed = " << get_random_seed() << endl;
					cout << "chisqlog: " << display_switch(open_chisq_logfile) << endl;
					cout << endl;
				}
				if (show_cosmo_settings) {
					cout << "\033[4mCosmology settings\033[0m\n";
					cout << "hubble = " << hubble << endl;
					cout << "omega_m = " << omega_matter << endl;
					cout << "zlens = " << lens_redshift << endl;
					cout << "zsrc = " << source_redshift << endl;
					cout << "zsrc_ref = " << reference_source_redshift << endl;
					cout << "auto_zsrc_scaling = " << auto_zsource_scaling << endl;
					cout << "vary_hubble: " << display_switch(vary_hubble_parameter) << endl;
					cout << "vary_omega_m: " << display_switch(vary_omega_matter_parameter) << endl;
					cout << endl;
				}
				if (show_lens_settings) {
					cout << "\033[4mLens model settings\033[0m\n";
					cout << "emode = " << LensProfile::default_ellipticity_mode << endl;
					cout << "pmode = " << default_parameter_mode << endl;
					cout << "primary_lens = " << primary_lens_number << ((auto_set_primary_lens==true) ? " (auto)": "") << endl;
					if (LensProfile::integral_method==Romberg_Integration) cout << "integral_method: Romberg integration" << endl;
					else if (LensProfile::integral_method==Gauss_Patterson_Quadrature) cout << "integral_method: Gauss-Patterson quadrature" << endl;
					else if (LensProfile::integral_method==Gaussian_Quadrature) cout << "integral_method: Gaussian quadrature with " << Gauss_NN << " points" << endl;
					cout << "integral_tolerance = " << integral_tolerance << endl;
					cout << "major_axis_along_y: " << display_switch(LensProfile::orient_major_axis_north) << endl;
					cout << "ellipticity_components: " << display_switch(LensProfile::use_ellipticity_components) << endl;
					cout << "shear_components: " << display_switch(Shear::use_shear_component_params) << endl;
					cout << "tab_rmin = " << tabulate_rmin << endl;
					cout << "tab_r_N = " << tabulate_logr_N << endl;
					cout << "tab_phi_N = " << tabulate_phi_N << endl;
					cout << endl;
				}
				if (show_misc_settings) {
					cout << "\033[4mMiscellaneous settings\033[0m\n";
					cout << "verbal_mode: " << display_switch(verbal_mode) << endl;
					cout << "warnings: " << display_switch(warnings) << endl;
					cout << "imgsrch_warnings: " << display_switch(newton_warnings) << endl;
					cout << "show_wtime: " << display_switch(show_wtime) << endl;
					cout << "sci_notation: " << display_switch(use_scientific_notation) << endl;
					cout << "sim_err_pos = " << sim_err_pos << endl;
					cout << "sim_err_flux = " << sim_err_flux << endl;
					cout << "sim_err_td = " << sim_err_td << endl;
					cout << endl;
				}
			}
		}
		else if (words[0]=="read")
		{
			if (nwords == 2) {
				if (infile->is_open()) {
					if (n_infiles==10) Complain("cannot open more than 10 files at once");
					infile++;
				}
				infile->open(words[1].c_str());
				if (infile->is_open()) {
					if ((n_infiles > 0) and (!read_from_file)) paused_while_reading_file = true;
					read_from_file = true;
					n_infiles++;
				}
				else {
					cerr << "Error: input file '" << words[1] << "' could not be opened" << endl;
					if (n_infiles > 0) infile--;
				}
			} else if (nwords == 1) {
				Complain("must specify filename for input file to be read");
			} else Complain("invalid number of arguments; must specify one filename to be read");
		}
		else if (words[0]=="write")
		{
			if (nwords == 2) {
				ofstream outfile(words[1].c_str());
				if (outfile.is_open()) 
				{
					for (int i=0; i < lines.size()-1; i++) {
						outfile << lines[i] << endl;
					}
					outfile.close();
				}
				else cerr << "Error: output file '" << words[1] << "' could not be opened" << endl;
			} else if (nwords == 1) {
				Complain("must specify filename for output file to be written to");
			} else Complain("invalid number of arguments; must specify one filename to be read");
		}
		else if (words[0]=="integral_method")
		{
			IntegrationMethod method;
			if ((nwords == 2) or (nwords == 3)) {
				string method_name;
				if (!(ws[1] >> method_name)) Complain("invalid integration method");
				if (method_name=="patterson") {
					if (nwords > 2) Complain("no arguments are allowed for 'integral_method patterson'");
					method = Gauss_Patterson_Quadrature;
				} else if (method_name=="romberg") {
					if (nwords > 2) Complain("no arguments are allowed for 'integral_method romberg'");
					method = Romberg_Integration;
				}
				else if (method_name=="gauss") {
					method = Gaussian_Quadrature;
					if (nwords == 3) {
						int pts;
						if (!(ws[2] >> pts)) Complain("invalid number of points");
						set_Gauss_NN(pts);
					}
				} else Complain("unknown integration method");
				set_integration_method(method);
			} else if (nwords==1) {
				if (mpi_id==0) {
					method = LensProfile::integral_method;
					if (method==Romberg_Integration) cout << "Integration method: Romberg integration with tolerance = " << integral_tolerance << endl;
					else if (method==Gauss_Patterson_Quadrature) cout << "Integration method: Gauss-Patterson quadrature with tolerance = " << integral_tolerance << endl;
					else if (method==Gaussian_Quadrature) cout << "Integration method: Gaussian quadrature with " << Gauss_NN << " points" << endl;
				}
			} else Complain("no more than two arguments are allowed for 'integral_method' (method,npoints)");
		}
		else if (words[0]=="gridtype")
		{
			if (nwords >= 2) {
				string gridtype_name;
				if (!(ws[1] >> gridtype_name)) Complain("invalid grid type");
				if (gridtype_name=="radial") radial_grid = true;
				else if ((gridtype_name=="cartesian") or (gridtype_name=="Cartesian")) {
					if (radial_grid) {
						int rsp, thetasp, xysp;
						double xysp_approx;
						rsp = usplit_initial;
						thetasp = wsplit_initial;
						xysp_approx = sqrt(usplit_initial*wsplit_initial);
						xysp = (int) xysp_approx;
						usplit_initial = wsplit_initial = xysp;
						radial_grid = false;
					}
				}
				else Complain("unknown grid type");
			} else {
				if (mpi_id==0) {
					if (radial_grid) cout << "Grid type: radial" << endl;
					else cout << "Grid type: Cartesian" << endl;
				}
			}
		}
		else if (words[0]=="grid")
		{
			if (nwords==1) {
				if (mpi_id==0) {
					double xlh, ylh;
					xlh = grid_xlength/2.0;
					ylh = grid_ylength/2.0;
					if ((xlh < 1e3) and (ylh < 1e3)) cout << resetiosflags(ios::scientific);
					cout << "grid = (" << grid_xcenter-xlh << "," << grid_xcenter+xlh << ") x (" << grid_ycenter-ylh << "," << grid_ycenter+ylh << ")" << endl;
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
				}
			} else if (nwords == 3) {
				double xlh, ylh;
				if (!(ws[1] >> xlh)) Complain("invalid grid x-coordinate");
				if (!(ws[2] >> ylh)) Complain("invalid grid y-coordinate");
				set_gridsize(xlh*2,ylh*2);
				if (mpi_id==0) {
					if ((xlh < 1e3) and (ylh < 1e3)) cout << resetiosflags(ios::scientific);
					cout << "set grid = (" << -xlh << "," << xlh << ") x (" << -ylh << "," << ylh << ")" << endl;
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
				}
			} else if (nwords == 4) {
				if (words[1]=="center") {
					double xc,yc;
					if (!(ws[2] >> xc)) Complain("invalid grid x-coordinate");
					if (!(ws[3] >> yc)) Complain("invalid grid y-coordinate");
					set_gridcenter(xc,yc);
				} else Complain("invalid arguments to 'grid' (type 'help grid' for usage information)");
			} else if (nwords == 5) {
				double xmin,xmax,ymin,ymax;
				if (!(ws[1] >> xmin)) Complain("invalid grid xmin");
				if (!(ws[2] >> xmax)) Complain("invalid grid ymin");
				if (!(ws[3] >> ymin)) Complain("invalid grid xmax");
				if (!(ws[4] >> ymax)) Complain("invalid grid ymax");
				set_grid_corners(xmin,xmax,ymin,ymax);
			} else Complain("invalid arguments to 'grid' (type 'help grid' for usage information)");
		}
		else if (words[0]=="img_npixels")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Number of image pixels = (" << n_image_pixels_x << "," << n_image_pixels_y << ")\n";
			} else if (nwords == 3) {
				int npx, npy;
				if (!(ws[1] >> npx)) Complain("invalid number of pixels");
				if (!(ws[2] >> npy)) Complain("invalid number of pixels");
				n_image_pixels_x = npx;
				n_image_pixels_y = npy;
			} else Complain("two arguments required to set 'img_npixels' (npixels_x, npixels_y)");
		}
		else if (words[0]=="src_npixels")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Number of source pixels = (" << srcgrid_npixels_x << "," << srcgrid_npixels_y << ")";
				if (auto_srcgrid_npixels) cout << " (auto_src_npixels on)";
				cout << endl;
			} else if (nwords == 3) {
				int npx, npy;
				if (!(ws[1] >> npx)) Complain("invalid number of pixels");
				if (!(ws[2] >> npy)) Complain("invalid number of pixels");
				srcgrid_npixels_x = npx;
				srcgrid_npixels_y = npy;
				auto_srcgrid_npixels = false;
			} else Complain("invalid arguments to 'src_npixels' (type 'help src_npixels' for usage information)");
		}
		else if (words[0]=="autogrid")
		{
			if (nlens == 0) Complain("cannot autogrid; no lens model has been specified");
			if (nwords >= 3) {
				double ccrmin, ccrmax, gridfrac;
				if (!(ws[1] >> ccrmin)) Complain("invalid grid x-coordinate");
				if (!(ws[2] >> ccrmax)) Complain("invalid grid y-coordinate");
				if (ccrmin > ccrmax) Complain("first parameter (rmin) must be less than second (rmax)");
				if (ccrmin <= 0.0) Complain("rmin parameter must be greater than zero (e.g. 1e-3)");
				if (nwords==4) {
					double gridfrac;
					if (!(ws[3] >> gridfrac)) Complain("invalid grid size fraction");
					if (gridfrac <= 1)
						Complain("autogrid: grid must be larger than tangential critical curve (autogrid_frac > 1)");
					autogrid(ccrmin,ccrmax,gridfrac);
				} else if (nwords > 4) {
					Complain("autogrid requires only two (rmin, rmax) or three (rmin, rmax, gridfrac) parameters");
				} else
					autogrid(ccrmin,ccrmax);
			} else if (nwords == 1) {
				autogrid();
			} else Complain("must specify two parameters, rmin and rmax, to determine gridsize automatically");
		}
		else if (words[0]=="cosmology") {
			if (nwords==1) {
				if (mpi_id==0) print_lens_cosmology_info(0,nlens-1);
			} else if (nwords==2) {
				int lnum;
				if (!(ws[1] >> lnum)) Complain("invalid lens number");
				if (lnum >= nlens) Complain("specified lens number does not exist");
				if (mpi_id==0) print_lens_cosmology_info(lnum,lnum);
			} else Complain("either zero or one argument required for cosmology command (lens_number)");
		}
		else if (words[0]=="mass_r") {
			if (nwords != 3) Complain("exactly two arguments required for mass_r command (lens_number,radius_arcsec)");
			int lnum;
			double r_arcsec;
			if (!(ws[1] >> lnum)) Complain("invalid lens number");
			if (lnum >= nlens) Complain("specified lens number does not exist");
			if (!(ws[2] >> r_arcsec)) Complain("invalid radius");
			if (mpi_id==0) output_mass_r(r_arcsec,lnum);
		}
		else if ((words[0]=="lens") or ((words[0]=="fit") and (nwords > 1) and (words[1]=="lens")))
		{
			bool update_parameters = false;
			bool update_specific_parameters = false; // option for user to update one (or more) specific parameters rather than update all of them at once
			bool vary_parameters = false;
			bool anchor_lens_center = false;
			bool add_shear = false;
			int emode = -1; // if set, then specifies the ellipticity mode for the lens being created
			boolvector vary_flags, shear_vary_flags;
			vector<string> specific_update_params;
			vector<double> specific_update_param_vals;
			dvector param_vals;
			double shear_param_vals[2];
			int nparams_to_vary, tot_nparams_to_vary;
			int anchornum; // in case new lens is being anchored to existing lens
			int lens_number;
			bool update_zl = false;
			double zl_in = lens_redshift;
			bool vary_zl = false;
			int pmode = default_parameter_mode;
			bool is_perturber = false;

			struct ParamAnchor {
				bool anchor_param;
				int paramnum;
				int anchor_paramnum;
				bool use_anchor_ratio;
				int anchor_lens_number;
				ParamAnchor() {
					anchor_param = false;
					use_anchor_ratio = false;
				}
				void shift(const int np) { if (paramnum > np) paramnum--; }
			};
			ParamAnchor parameter_anchors[20]; // number of anchors per lens can't exceed 20 (which will never happen!)
			int parameter_anchor_i = 0;

			if (words[0]=="fit") {
				vary_parameters = true;
				// now remove the "fit" word from the line so we can add lenses the same way,
				// but with vary_parameters==true so we'll prompt for an extra line to vary parameters
				stringstream* new_ws = new stringstream[nwords-1];
				for (int i=0; i < nwords-1; i++) {
					words[i] = words[i+1];
					new_ws[i] << words[i];
				}
				words.pop_back();
				nwords--;
				delete[] ws;
				ws = new_ws;
			}
			LensProfileName profile_name;
			if ((nwords > 1) and (words[1]=="update")) {
				if (nwords > 2) {
					if (!(ws[2] >> lens_number)) Complain("invalid lens number");
					if ((nlens <= lens_number) or (lens_number < 0)) Complain("specified lens number does not exist");
					update_parameters = true;
					profile_name = lens_list[lens_number]->get_lenstype();
					// Now we'll remove the "update" word and replace the lens number with the lens name
					// so it follows the format of the usual "lens" command, but with update_parameters==true
					stringstream* new_ws = new stringstream[nwords-1];
					words[1] = (profile_name==KSPLINE) ? "kspline" :
									(profile_name==ALPHA) ? "alpha" :
									(profile_name==PJAFFE) ? "pjaffe" :
									(profile_name==MULTIPOLE) ? "mpole" :
									(profile_name==nfw) ? "nfw" :
									(profile_name==TRUNCATED_nfw) ? "tnfw" :
									(profile_name==CORED_nfw) ? "cnfw" :
									(profile_name==HERNQUIST) ? "hern" :
									(profile_name==EXPDISK) ? "expdisk" :
									(profile_name==CORECUSP) ? "corecusp" :
									(profile_name==SERSIC_LENS) ? "sersic" :
									(profile_name==SHEAR) ? "shear" :
									(profile_name==TABULATED) ? "tab" :
									(profile_name==QTABULATED) ? "qtab" :
									(profile_name==PTMASS) ? "ptmass" : "test";
					for (int i=2; i < nwords-1; i++)
						words[i] = words[i+1];
					for (int i=0; i < nwords-1; i++)
						new_ws[i] << words[i];
					words.pop_back();
					nwords--;
					delete[] ws;
					ws = new_ws;
				} else Complain("must specify a lens number to update, followed by parameters");
			} else {
				// check for words that specify ellipticity mode, shear anchoring, or parameter anchoring
				for (int i=1; i < nwords; i++) {
					if (words[i]=="perturber") {
						is_perturber = true;
						remove_word(i);
						i = nwords; // breaks out of this loop, without breaking from outer loop
					}
				}
				for (int i=2; i < nwords; i++) {
					int pos;
					if ((pos = words[i].find("emode=")) != string::npos) {
						string enumstring = words[i].substr(pos+6);
						stringstream enumstr;
						enumstr << enumstring;
						if (!(enumstr >> emode)) Complain("incorrect format for ellipticity mode; must specify 0, 1, 2, or 3");
						if ((emode < 0) or (emode > 3)) Complain("ellipticity mode must be either 0, 1, 2, or 3");
						remove_word(i);
						i = nwords; // breaks out of this loop, without breaking from outer loop
					}
				}	

				for (int i=2; i < nwords; i++) {
					int pos;
					if ((pos = words[i].find("pmode=")) != string::npos) {
						string pnumstring = words[i].substr(pos+6);
						stringstream pnumstr;
						pnumstr << pnumstring;
						if (!(pnumstr >> pmode)) Complain("incorrect format for parameter mode; must specify 0, 1, or 2");
						remove_word(i);
						i = nwords; // breaks out of this loop, without breaking from outer loop
					}
				}

				for (int i=2; i < nwords; i++) {
					if (words[i].find("shear=")==0) {
						if (i==nwords-1) Complain("adding external shear via 'shear=# #' requires two arguments (shear1,shear2)");
						add_shear = true;
						string shearstr = words[i].substr(6);
						stringstream shearstream;
						shearstream << shearstr;
						if (!(shearstream >> shear_param_vals[0])) {
							if (Shear::use_shear_component_params) Complain("invalid shear_1 value");
							Complain("invalid shear value");
						}
						if (!(ws[i+1] >> shear_param_vals[1])) {
							if (Shear::use_shear_component_params) Complain("invalid shear_2 value");
							Complain("invalid shear angle");
						}
						remove_word(i+1);
						remove_word(i);
						break;
					}
				}

				for (int i=2; i < nwords; i++) {
					int pos0;
					if ((pos0 = words[i].find("/anchor=")) != string::npos) {
						string pvalstring, astr;
						pvalstring = words[i].substr(0,pos0);
						astr = words[i].substr(pos0+8);
						int pos, lnum, pnum;
						if ((pos = astr.find(",")) != string::npos) {
							string lnumstring, pnumstring;
							lnumstring = astr.substr(0,pos);
							pnumstring = astr.substr(pos+1);
							stringstream lnumstr, pnumstr;
							lnumstr << lnumstring;
							if (!(lnumstr >> lnum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<lens_number>,<param_number>' in place of parameter");
							pnumstr << pnumstring;
							if (!(pnumstr >> pnum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<lens_number>,<param_number>' in place of parameter");
							if (lnum > nlens) Complain("specified lens number to anchor to does not exist");
							if ((lnum != nlens) and (pnum >= lens_list[lnum]->get_n_params())) Complain("specified parameter number to anchor to does not exist for given lens");
							parameter_anchors[parameter_anchor_i].anchor_param = true;
							parameter_anchors[parameter_anchor_i].use_anchor_ratio = true;
							parameter_anchors[parameter_anchor_i].paramnum = i-2;
							parameter_anchors[parameter_anchor_i].anchor_lens_number = lnum;
							parameter_anchors[parameter_anchor_i].anchor_paramnum = pnum;
							parameter_anchor_i++;
							words[i] = pvalstring;
							ws[i].str(""); ws[i].clear();
							ws[i] << words[i];
						} else Complain("incorrect format for anchoring parameter; must type 'anchor=<lens_number>,<param_number>' in place of parameter");
					}
				}	

				for (int i=2; i < nwords; i++) {
					if (words[i].find("anchor=")==0) {
						string astr = words[i].substr(7);
						int pos, lnum, pnum;
						if ((pos = astr.find(",")) != string::npos) {
							string lnumstring, pnumstring;
							lnumstring = astr.substr(0,pos);
							pnumstring = astr.substr(pos+1);
							stringstream lnumstr, pnumstr;
							lnumstr << lnumstring;
							if (!(lnumstr >> lnum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<lens_number>,<param_number>' in place of parameter");
							pnumstr << pnumstring;
							if (!(pnumstr >> pnum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<lens_number>,<param_number>' in place of parameter");
							if (lnum > nlens) Complain("specified lens number to anchor to does not exist");
							if ((lnum != nlens) and (pnum >= lens_list[lnum]->get_n_params())) Complain("specified parameter number to anchor to does not exist for given lens");
							parameter_anchors[parameter_anchor_i].anchor_param = true;
							parameter_anchors[parameter_anchor_i].paramnum = i-2;
							parameter_anchors[parameter_anchor_i].anchor_lens_number = lnum;
							parameter_anchors[parameter_anchor_i].anchor_paramnum = pnum;
							parameter_anchor_i++;
							words[i] = "0";
							ws[i].str(""); ws[i].clear();
							ws[i] << words[i];
						} else Complain("incorrect format for anchoring parameter; must type 'anchor=<lens_number>,<param_number>' in place of parameter");
					}
				}	
			}

			if (update_parameters) {
				int pos, n_updates = 0;
				double pval;
				for (int i=2; i < nwords; i++) {
					if ((pos = words[i].find("="))!=string::npos) {
						if (words[i].find("anchor=")!=string::npos) Complain("parameter anchorings cannot be updated");
						n_updates++;
						specific_update_params.push_back(words[i].substr(0,pos));
						stringstream pvalstr;
						pvalstr << words[i].substr(pos+1);
						pvalstr >> pval;
						specific_update_param_vals.push_back(pval);
					} else if (i==2) break;
				}
				if (n_updates > 0) {
					if (n_updates < nwords-2) Complain("lens parameters must all be updated at once, or else specific parameters using '<param>=...'");
					update_specific_parameters = true;
					for (int i=0; i < n_updates; i++)
						if (lens_list[lens_number]->update_specific_parameter(specific_update_params[i],specific_update_param_vals[i])==false) Complain("could not find parameter '" << specific_update_params[i] << "' in lens " << lens_number);
					update_anchored_parameters_and_redshift_data();
					reset();
				}
			}
			if (!update_specific_parameters) {
				for (int i=3; i < nwords; i++) {
					int pos;
					if ((pos = words[i].find("z=")) != string::npos) {
						string znumstring = words[i].substr(pos+2);
						stringstream znumstr;
						znumstr << znumstring;
						if (!(znumstr >> zl_in)) Complain("incorrect format for lens redshift");
						if (zl_in < 0) Complain("lens redshift cannot be negative");
						remove_word(i);
						i = nwords; // breaks out of this loop, without breaking from outer loop
						update_zl = true;
					}
				}	
			}

			if (update_specific_parameters) ;
			else if (nwords==1) {
				if (mpi_id==0) print_lens_list(vary_parameters);
			}
			else if (words[1]=="clear")
			{
				if (nwords==2) {
					clear_lenses();
					clear_source_objects();
				} else if (nwords==3) {
					int lensnumber, min_lensnumber, max_lensnumber, pos;
					if ((pos = words[2].find("-")) != string::npos) {
						string lminstring, lmaxstring;
						lminstring = words[2].substr(0,pos);
						lmaxstring = words[2].substr(pos+1);
						stringstream lmaxstream, lminstream;
						lminstream << lminstring;
						lmaxstream << lmaxstring;
						if (!(lminstream >> min_lensnumber)) Complain("invalid min lens number");
						if (!(lmaxstream >> max_lensnumber)) Complain("invalid max lens number");
						if (max_lensnumber >= nlens) Complain("specified max lens number exceeds number of lenses in list");
						if ((min_lensnumber > max_lensnumber) or (min_lensnumber < 0)) Complain("specified min lens number cannot exceed max lens number");
						int pi, pf;
						for (int i=max_lensnumber; i >= min_lensnumber; i--) {
							remove_lens(i);
						}
					} else {
						if (!(ws[2] >> lensnumber)) Complain("invalid lens number");
						int pi, pf;
						remove_lens(lensnumber);
					}
				} else Complain("'lens clear' command requires either one or zero arguments");
			}
			else if (words[1]=="anchor")
			{
				if (nwords != 4) Complain("must specify two arguments for 'lens anchor': lens1 --> lens2");
				int lens1, lens2;
				if (!(ws[2] >> lens1)) Complain("invalid lens number for lens to be anchored");
				if (!(ws[3] >> lens2)) Complain("invalid lens number for lens to anchor to");
				if (lens1 >= nlens) Complain("lens1 number does not exist");
				if (lens2 >= nlens) Complain("lens2 number does not exist");
				if (lens1 == lens2) Complain("lens1, lens2 must be different");
				lens_list[lens1]->anchor_center_to_lens(lens_list,lens2);
			}
			else if (words[1]=="savetab")
			{
				if (nwords != 4) Complain("must specify two arguments for 'lens savetab': lens number and output file");
				int lnum;
				if (!(ws[2] >> lnum)) Complain("invalid lens number for saving tabulated lens");
				if (lnum >= nlens) Complain("lens number does not exist");
				if (!save_tabulated_lens_to_file(lnum,words[3])) Complain("specified lens is not a tabulated lens model");
			}
			else
			{
				if (zl_in > reference_source_redshift) Complain("lens redshift cannot be greater than reference source redshift (zsrc_ref)");
				if (words[1]=="alpha")
				{
					if (nwords > 9) Complain("more than 7 parameters not allowed for model alpha");
					if (nwords >= 6) {
						double b, alpha, s;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> b)) Complain("invalid b parameter for model alpha");
						if (!(ws[3] >> alpha)) Complain("invalid alpha parameter for model alpha");
						if (!(ws[4] >> s)) Complain("invalid s (core) parameter for model alpha");
						if (!(ws[5] >> q)) Complain("invalid q parameter for model alpha");
						if (alpha <= 0) Complain("alpha cannot be less than or equal to zero (or else the mass diverges near r=0)");
						if (nwords >= 7) {
							if (!(ws[6] >> theta)) Complain("invalid theta parameter for model alpha");
							if (nwords == 8) {
								if (words[7].find("anchor_center=")==0) {
									string anchorstr = words[7].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								}
							}
							if (nwords == 9) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model alpha");
								if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model alpha");
							}
						}
						param_vals.input(8);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=b; param_vals[1]=alpha; param_vals[2]=s; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 5 : 7;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[5] != "0") or (words[6] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (b,alpha,s,q,theta) in model alpha";
								}
								else complain_str = "Must specify vary flags for seven parameters (b,alpha,s,q,theta,xc,yc) in model alpha";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(ALPHA, emode, zl_in, reference_source_redshift, b, alpha, s, q, theta, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("alpha requires at least 4 parameters (b, alpha, s, q)");
				}
				else if (words[1]=="pjaffe")
				{
					bool set_tidal_host = false;
					int hostnum;
					if ((pmode < 0) or (pmode > 2)) Complain("parameter mode must be either 0, 1, or 2");
					if (nwords > 9) Complain("more than 7 parameters not allowed for model pjaffe");
					if (nwords >= 6) {
						double p1, p2, p3;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> p1)) {
							if (pmode==0) Complain("invalid b parameter for model pjaffe");
							else Complain("invalid sigma_v parameter for model pjaffe");
						}
						if (words[3].find("host=")==0) {
							string hoststr = words[3].substr(5);
							stringstream hoststream;
							hoststream << hoststr;
							if (!(hoststream >> hostnum)) Complain("invalid lens number for tidal host");
							if (hostnum >= nlens) Complain("lens number does not exist");
							set_tidal_host = true;
							p2 = 0;
						} else if (!(ws[3] >> p2)) Complain("invalid a parameter for model pjaffe");
						if (!(ws[4] >> p3)) Complain("invalid s (core) parameter for model pjaffe");
						if (!(ws[5] >> q)) Complain("invalid q parameter for model pjaffe");
						if (nwords >= 7) {
							if (!(ws[6] >> theta)) Complain("invalid theta parameter for model pjaffe");
							if (nwords == 8) {
								if (words[7].find("anchor_center=")==0) {
									string anchorstr = words[7].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								}
							}
							else if (nwords == 9) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model pjaffe");
								if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model pjaffe");
							}
						}
						param_vals.input(8);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=p2; param_vals[2]=p3; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 5 : 7;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[5] != "0") or (words[6] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (b,a,s,q,theta) in model pjaffe";
								}
								else complain_str = "Must specify vary flags for seven parameters (b,a,s,q,theta,xc,yc) in model pjaffe";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							if ((set_tidal_host==true) and (vary_flags[1]==true)) Complain("parameter a cannot be varied if calculated from tidal host");

							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(PJAFFE, emode, zl_in, reference_source_redshift, p1, p2, p3, q, theta, xc, yc, 0, 0, pmode);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (set_tidal_host) {
								lens_list[nlens-1]->assign_special_anchored_parameters(lens_list[hostnum],1,true);
								if ((vary_parameters) and (vary_flags[1])) lens_list[nlens-1]->unassign_special_anchored_parameter(); // we're only setting the initial value for a
							}
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("pjaffe requires at least 4 parameters (b, a, s, q)");
				}
				else if ((words[1]=="mpole") or (words[1]=="kmpole"))
				{
					bool kappa_multipole = false;
					bool sine_term = false;
					int primary_lens_num;
					if (words[1]=="kmpole") kappa_multipole = true;
					if (nwords > 9) Complain("more than 8 arguments not allowed for model " << words[1]);
					if (nwords >= 4) {
						double a_m, n;
						int m=0;
						if ((words[2].find("sin")==0) or (words[2].find("cos")==0)) {
							if (update_parameters) Complain("sine/cosine argument cannot be specified when updating " << words[1]);
							if (words[2].find("sin")==0) sine_term = true;
							stringstream* new_ws = new stringstream[nwords-1];
							words.erase(words.begin()+2);
							for (int i=0; i < nwords-1; i++) {
								new_ws[i] << words[i];
							}
							delete[] ws;
							ws = new_ws;
							nwords--;
							for (int i=0; i < parameter_anchor_i; i++) parameter_anchors[i].shift(2);
						}
						if (words[2].find("m=")==0) {
							if (update_parameters) Complain("m=# argument cannot be specified when updating " << words[1]);
							string mstr = words[2].substr(2);
							stringstream mstream;
							mstream << mstr;
							if (!(mstream >> m)) Complain("invalid m value");
							stringstream* new_ws = new stringstream[nwords-1];
							words.erase(words.begin()+2);
							for (int i=0; i < nwords-1; i++) {
								new_ws[i] << words[i];
							}
							delete[] ws;
							ws = new_ws;
							nwords--;
							for (int i=0; i < parameter_anchor_i; i++) parameter_anchors[i].shift(2);
						}
						double theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> a_m)) Complain("invalid a_m parameter for model " << words[1]);
						if (!(ws[3] >> n)) {
							if (kappa_multipole) Complain("invalid beta parameter for model " << words[1]);
							else Complain("invalid n parameter for model " << words[1]);
						}
						if ((kappa_multipole) and (n == 2-m)) Complain("for kmpole, beta cannot be equal to 2-m (or else deflections become infinite)");
						if (nwords >= 5) {
							if (!(ws[4] >> theta)) Complain("invalid theta parameter for model " << words[1]);
							if (nwords == 6) {
								if (words[5].find("anchor_center=")==0) {
									string anchorstr = words[5].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == 7) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[5] >> xc)) Complain("invalid x-center parameter for model " << words[1]);
								if (!(ws[6] >> yc)) Complain("invalid y-center parameter for model " << words[1]);
							}
						}
						param_vals.input(6);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=a_m; param_vals[1]=n; param_vals[2]=theta; param_vals[3]=xc; param_vals[4]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[5]=zl_in;
						else param_vals[5]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 3 : 5;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[3] != "0") or (words[4] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for three parameters (A_m,n,theta) in model " + words[1];
								}
								else complain_str = "Must specify vary flags for six parameters (A_m,n,theta,xc,yc) in model " + words[1];
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_multipole_lens(zl_in, reference_source_redshift, m, a_m, n, theta, xc, yc, kappa_multipole, sine_term);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("mpole requires at least 2 parameters (a_m, n)");
				}
				else if (words[1]=="nfw")
				{
					bool set_median_concentration = false;
					double cmed_factor = 1.0;
					if ((update_parameters) and (lens_list[lens_number]->anchor_special_parameter)) {
						set_median_concentration = true;
						pmode = 1; // you should generalize the parameter choice option so it's in the LensProfile class; then you can check for different parametrizations directly
					}
					if ((pmode < 0) or (pmode > 2)) Complain("parameter mode must be either 0, 1, or 2");

					if (nwords > 8) Complain("more than 6 parameters not allowed for model nfw");
					if (nwords >= 5) {
						double p1, p2;
						double q, theta = 0, xc = 0, yc = 0;
						if (pmode==2) {
							if (!(ws[2] >> p1)) Complain("invalid mvir parameter for model nfw");
							if (!(ws[3] >> p2)) Complain("invalid rs parameter for model nfw");
						} else if (pmode==1) {
							if (!(ws[2] >> p1)) Complain("invalid mvir parameter for model nfw");
							int pos;
							if ((pos = words[3].find("*cmed")) != string::npos) {
								set_median_concentration = true;
								string facstring;
								facstring = words[3].substr(0,pos);
								stringstream facstream;
								facstream << facstring;
								if (!(facstream >> cmed_factor)) Complain("invalid factor of median concentration");
							} else if (words[3]=="cmed") set_median_concentration = true;
							else if (!(ws[3] >> p2)) Complain("invalid c parameter for model nfw");
						} else {
							if (!(ws[2] >> p1)) Complain("invalid ks parameter for model nfw");
							if (!(ws[3] >> p2)) Complain("invalid rs parameter for model nfw");
						}
						if (!(ws[4] >> q)) Complain("invalid q parameter for model nfw");
						if (nwords >= 6) {
							if (!(ws[5] >> theta)) Complain("invalid theta parameter for model nfw");
							if (nwords == 7) {
								if (words[6].find("anchor_center=")==0) {
									string anchorstr = words[6].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == 8) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[6] >> xc)) Complain("invalid x-center parameter for model nfw");
								if (!(ws[7] >> yc)) Complain("invalid y-center parameter for model nfw");
							}
						}
						param_vals.input(7);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=p2; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 4 : 6;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[4] != "0") or (words[5] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for four parameters (p1,p2,q,theta) in model nfw";
								}
								else complain_str = "Must specify vary flags for six parameters (p1,p2,q,theta,xc,yc) in model nfw";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(nfw, emode, zl_in, reference_source_redshift, p1, p2, 0.0, q, theta, xc, yc, 0, 0, pmode);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (set_median_concentration) {
								lens_list[nlens-1]->assign_special_anchored_parameters(lens_list[nlens-1],cmed_factor,true);
								if ((vary_parameters) and (vary_flags[1])) lens_list[nlens-1]->unassign_special_anchored_parameter(); // we're only setting the initial value for c
							}
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("nfw requires at least 3 parameters (ks, rs, q)");
				}
				else if (words[1]=="tnfw")
				{
					if (nwords > 9) Complain("more than 7 parameters not allowed for model tnfw");
					if (nwords >= 6) {
						double ks, rs, rt;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> ks)) Complain("invalid ks parameter for model tnfw");
						if (!(ws[3] >> rs)) Complain("invalid rs parameter for model tnfw");
						if (!(ws[4] >> rt)) Complain("invalid rt parameter for model tnfw");
						if (!(ws[5] >> q)) Complain("invalid q parameter for model tnfw");
						if (nwords >= 7) {
							if (!(ws[6] >> theta)) Complain("invalid theta parameter for model tnfw");
							if (nwords == 8) {
								if (words[7].find("anchor_center=")==0) {
									string anchorstr = words[7].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == 9) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model tnfw");
								if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model tnfw");
							}
						}
						param_vals.input(8);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=ks; param_vals[1]=rs; param_vals[2]=rt; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 5 : 7;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[5] != "0") or (words[6] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (ks,rs,rt,q,theta) in model tnfw";
								}
								else complain_str = "Must specify vary flags for seven parameters (ks,rs,rt,q,theta,xc,yc) in model tnfw";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(TRUNCATED_nfw, emode, zl_in, reference_source_redshift, ks, rs, rt, q, theta, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("tnfw requires at least 4 parameters (ks, rs, rt, q)");
				}
				else if (words[1]=="cnfw")
				{
					bool set_median_concentration = false;
					if ((update_parameters) and (lens_list[lens_number]->anchor_special_parameter)) {
						set_median_concentration = true;
						pmode = 1; // you should generalize the parameter choice option so it's in the LensProfile class; then you can check for different parametrizations directly
					}
					if ((pmode < 0) or (pmode > 3)) Complain("parameter mode must be either 0, 1, 2 or 3");

					if (nwords > 9) Complain("more than 7 parameters not allowed for model cnfw");
					if (nwords >= 6) {
						double p1, p2, p3;
						double q, theta = 0, xc = 0, yc = 0;
						if (pmode==3) {
							if (!(ws[2] >> p1)) Complain("invalid m200 parameter for model cnfw");
							if (words[3]=="cmed") set_median_concentration = true;
							else if (!(ws[3] >> p2)) Complain("invalid c parameter for model cnfw");
							if (!(ws[4] >> p3)) Complain("invalid rs_kpc parameter for model cnfw");
						} else if (pmode==2) {
							if (!(ws[2] >> p1)) Complain("invalid m200 parameter for model cnfw");
							if (!(ws[3] >> p2)) Complain("invalid rs parameter for model cnfw");
							if (!(ws[4] >> p3)) Complain("invalid beta parameter for model cnfw");
						} else if (pmode==1) {
							if (!(ws[2] >> p1)) Complain("invalid m200 parameter for model cnfw");
							if (words[3]=="cmed") set_median_concentration = true;
							else if (!(ws[3] >> p2)) Complain("invalid c parameter for model cnfw");
							if (!(ws[4] >> p3)) Complain("invalid beta parameter for model cnfw");
						} else {
							if (!(ws[2] >> p1)) Complain("invalid ks parameter for model cnfw");
							if (!(ws[3] >> p2)) Complain("invalid rs parameter for model cnfw");
							if (!(ws[4] >> p3)) Complain("invalid rc parameter for model cnfw");
						}
						if (!(ws[5] >> q)) Complain("invalid q parameter for model cnfw");
						if (nwords >= 7) {
							if (!(ws[6] >> theta)) Complain("invalid theta parameter for model cnfw");
							if (nwords == 8) {
								if (words[7].find("anchor_center=")==0) {
									string anchorstr = words[7].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == 9) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model cnfw");
								if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model cnfw");
							}
						}
						//if (p3 >= p2) Complain("core radius (p3) must be smaller than scale radius (p2) for model cnfw");
						param_vals.input(8);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=p2; param_vals[2]=p3; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 5 : 7;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[5] != "0") or (words[6] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (p1,p2,p3,q,theta) in model cnfw";
								}
								else complain_str = "Must specify vary flags for seven parameters (p1,p2,p3,q,theta,xc,yc) in model cnfw";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(CORED_nfw, emode, zl_in, reference_source_redshift, p1, p2, p3, q, theta, xc, yc, 0, 0, pmode);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (set_median_concentration) {
								lens_list[nlens-1]->assign_special_anchored_parameters(lens_list[nlens-1],1,true);
								if ((vary_parameters) and (vary_flags[1])) lens_list[nlens-1]->unassign_special_anchored_parameter(); // we're only setting the initial value for c
							}
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("cnfw requires at least 4 parameters (ks, rs, rc, q)");
				}
				else if (words[1]=="expdisk")
				{
					if (nwords > 8) Complain("more than 6 parameters not allowed for model expdisk");
					if (nwords >= 5) {
						double k0, R_d;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> k0)) Complain("invalid k0 parameter for model expdisk");
						if (!(ws[3] >> R_d)) Complain("invalid R_d parameter for model expdisk");
						if (!(ws[4] >> q)) Complain("invalid q parameter for model expdisk");
						if (nwords >= 6) {
							if (!(ws[5] >> theta)) Complain("invalid theta parameter for model expdisk");
							if (nwords == 7) {
								if (words[6].find("anchor_center=")==0) {
									string anchorstr = words[6].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == 8) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[6] >> xc)) Complain("invalid x-center parameter for model expdisk");
								if (!(ws[7] >> yc)) Complain("invalid y-center parameter for model expdisk");
							}
						}
						param_vals.input(7);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=k0; param_vals[1]=R_d; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 4 : 6;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[4] != "0") or (words[5] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for four parameters (k0,R_d,q,theta) in model expdisk";
								}
								else complain_str = "Must specify vary flags for six parameters (k0,R_d,q,theta,xc,yc) in model expdisk";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(EXPDISK, emode, zl_in, reference_source_redshift, k0, R_d, 0.0, q, theta, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("expdisk requires at least 3 parameteR_d (k0, R_d, q)");
				}
				else if (words[1]=="kspline")
				{
					if (nwords > 9) Complain("more than 7 parameters not allowed for model kspline");
					string filename;
					if (nwords >= 3) {
						double q = 1, theta = 0, xc = 0, yc = 0, qx = 1, f = 1;
						int index=2;
						if (!update_parameters)
							filename = words[index++];
						if ((nwords >= index+1) and (!(ws[index] >> qx))) Complain("invalid qx parameter for model kspline");
						else index++;
						if ((nwords >= index+1) and (!(ws[index] >> f))) Complain("invalid f parameter for model kspline");
						else index++;
						if (nwords >= index+1) {
							if (!(ws[index] >> q)) Complain("invalid q parameter");
							else index++;
							if (!(ws[index] >> theta)) Complain("invalid theta parameter");
							else index++;
							if (nwords == index+1) {
								if (words[index].find("anchor_center=")==0) {
									string anchorstr = words[index].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == index+2) {
							if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[index++] >> xc)) Complain("invalid x-center parameter for model kspline");
								if (!(ws[index] >> yc)) Complain("invalid y-center parameter for model kspline");
							}
						}
						param_vals.input(7);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=qx; param_vals[1]=f; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 4 : 6;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[2] != "0") or (words[3] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for four parameters (qx,f,q,theta) in model kspline";
								}
								else complain_str = "Must specify vary flags for six parameters (qx,f,q,theta,xc,yc) in model kspline";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(filename.c_str(), emode, zl_in, reference_source_redshift, q, theta, qx, f, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("kspline requires at least 1 argument (filename)");
				}
				else if (words[1]=="hern")
				{
					if (nwords > 9) Complain("more than 7 parameters not allowed for model hern");
					if (nwords >= 5) {
						double ks, rs;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> ks)) Complain("invalid ks parameter for model hern");
						if (!(ws[3] >> rs)) Complain("invalid rs parameter for model hern");
						if (!(ws[4] >> q)) Complain("invalid q parameter for model hern");
						if (nwords >= 6) {
							if (!(ws[5] >> theta)) Complain("invalid theta parameter for model hern");
							if (nwords == 7) {
								if (words[6].find("anchor_center=")==0) {
									string anchorstr = words[6].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == 8) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[6] >> xc)) Complain("invalid x-center parameter for model hern");
								if (!(ws[7] >> yc)) Complain("invalid y-center parameter for model hern");
							}
						}
						param_vals.input(7);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=ks; param_vals[1]=rs; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 4 : 6;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[4] != "0") or (words[5] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for four parameters (ks,rs,q,theta) in model hern";
								}
								else complain_str = "Must specify vary flags for six parameters (ks,rs,q,theta,xc,yc) in model hern";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(HERNQUIST, emode, zl_in, reference_source_redshift, ks, rs, 0.0, q, theta, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("hern requires at least 3 parameters (ks, rs, q)");
				}
				else if (words[1]=="corecusp")
				{
					bool set_tidal_host = false;
					// should change format so 'pmode=1' uses the re_param
					int pmode = 0;
					int hostnum;
					if (nwords > 12) Complain("more than 10 parameters not allowed for model corecusp");
					if ((nwords >= 8) and (words[2].find("re_param")==0)) {
						if (update_parameters) Complain("Einstein radius parameterization cannot be changed when updating corecusp");
						pmode = 1;
						stringstream* new_ws = new stringstream[nwords-1];
						words.erase(words.begin()+2);
						for (int i=0; i < nwords-1; i++) {
							new_ws[i] << words[i];
						}
						delete[] ws;
						ws = new_ws;
						nwords--;
						for (int i=0; i < parameter_anchor_i; i++) parameter_anchors[i].shift(2);
					}
					if (nwords >= 8) {
						double k0, gamma, n, a, s;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> k0)) Complain("invalid k0 parameter for model corecusp");
						if (!(ws[3] >> gamma)) Complain("invalid gamma parameter for model corecusp");
						if (!(ws[4] >> n)) Complain("invalid n parameter for model corecusp");
						if (words[5].find("host=")==0) {
							string hoststr = words[5].substr(5);
							stringstream hoststream;
							hoststream << hoststr;
							if (!(hoststream >> hostnum)) Complain("invalid lens number for tidal host");
							if (hostnum >= nlens) Complain("lens number does not exist");
							set_tidal_host = true;
							a = 0;
						} else {
							if (!(ws[5] >> a)) Complain("invalid a parameter for model corecusp");
						}
						if (!(ws[6] >> s)) Complain("invalid s (core) parameter for model corecusp");
						if (a < s) Complain("scale radius a cannot be smaller than s");
						if (!(ws[7] >> q)) Complain("invalid q parameter for model corecusp");
						if (nwords >= 9) {
							if (!(ws[8] >> theta)) Complain("invalid theta parameter for model corecusp");
							if (nwords == 10) {
								if (words[9].find("anchor_center=")==0) {
									string anchorstr = words[7].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								}
							}
							if (nwords == 11) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[9] >> xc)) Complain("invalid x-center parameter for model corecusp");
								if (!(ws[10] >> yc)) Complain("invalid y-center parameter for model corecusp");
							}
						}
						param_vals.input(10);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=k0; param_vals[1]=gamma; param_vals[2]=n; param_vals[3]=a; param_vals[4]=s; param_vals[5]=q; param_vals[6]=theta; param_vals[7]=xc; param_vals[8]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[9]=zl_in;
						else param_vals[9]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 7 : 9;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[7] != "0") or (words[8] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for seven parameters (k0,gamma,n,a,s,q,theta) in model corecusp";
								}
								else complain_str = "Must specify vary flags for nine parameters (k0,gamma,n,a,s,q,theta,xc,yc) in model corecusp";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(CORECUSP, emode, zl_in, reference_source_redshift, k0, a, s, q, theta, xc, yc, gamma, n, pmode);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (set_tidal_host) {
								lens_list[nlens-1]->assign_special_anchored_parameters(lens_list[hostnum],1,true);
								if ((vary_parameters) and (vary_flags[3])) lens_list[nlens-1]->unassign_special_anchored_parameter(); // we're only setting the initial value for a
							}
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("corecusp requires at least 6 parameters (k0, gamma, n, a, s, q)");
				}
				else if (words[1]=="ptmass")
				{
					if (nwords > 5) Complain("more than 3 parameters not allowed for model ptmass");
					if (nwords >= 3) {
						double b, xc = 0, yc = 0;
						if (!(ws[2] >> b)) Complain("invalid b parameter for model ptmass");
						if (nwords >= 4) {
							if (nwords == 4) {
								if (words[3].find("anchor_center=")==0) {
									string anchorstr = words[3].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								}
							}
							else if (nwords == 5) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[3] >> xc)) Complain("invalid x-center parameter for model ptmass");
								if (!(ws[4] >> yc)) Complain("invalid y-center parameter for model ptmass");
							}
						}
						param_vals.input(4);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=b; param_vals[1]=xc; param_vals[2]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[3]=zl_in;
						else param_vals[3]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 1 : 3;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[1] != "0") or (words[2] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for one parameter (b) in model ptmass";
								}
								else complain_str = "Must specify vary flags for three parameters (b,xc,yc) in model ptmass";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_ptmass_lens(zl_in, reference_source_redshift, b, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("ptmass requires at least 1 parameter, b (Einstein radius)");
				}
				else if (words[1]=="sersic")
				{
					int primary_lens_num;
					if (nwords > 9) Complain("more than 7 parameters not allowed for model sersic");
					if (nwords >= 6) {
						double kappe_e, re, n;
						double q, theta = 0, xc = 0, yc = 0;
						int pos;
						if (!(ws[2] >> kappe_e)) Complain("invalid kappe_e parameter for model sersic");
						if (!(ws[3] >> re)) Complain("invalid sersic parameter for model sersic");
						if (!(ws[4] >> n)) Complain("invalid n (core) parameter for model sersic");
						if (!(ws[5] >> q)) Complain("invalid q parameter for model sersic");
						if (re <= 0) Complain("re cannot be less than or equal to zero");
						if (nwords >= 7) {
							if (!(ws[6] >> theta)) Complain("invalid theta parameter for model sersic");
							if (nwords == 8) {
								if (words[7].find("anchor_center=")==0) {
									string anchorstr = words[7].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate entered for center, but not y-coordinate");
							}
							if (nwords == 9) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model sersic");
								if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model sersic");
							}
						}
						param_vals.input(8);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=kappe_e; param_vals[1]=re; param_vals[2]=n; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 5 : 7;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[5] != "0") or (words[6] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (kappe_e,R_eff,n,q,theta) in model sersic";
								}
								else complain_str = "Must specify vary flags for seven parameters (kappe_e,R_eff,n,q,theta,xc,yc) in model sersic";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_lens(SERSIC_LENS, emode, zl_in, reference_source_redshift, kappe_e, re, n, q, theta, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("sersic requires at least 4 parameters (kappe_e, R_eff, n, q)");
				}
				else if (words[1]=="sheet")
				{
					if (nwords > 5) Complain("more than 3 parameters not allowed for model sheet");
					if (nwords >= 3) {
						double kappa, xc = 0, yc = 0;
						if (!(ws[2] >> kappa)) Complain("invalid kappa parameter for model sheet");
						if (nwords >= 4) {
							if (nwords == 4) {
								if (words[3].find("anchor_center=")==0) {
									string anchorstr = words[3].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								}
							}
							else if (nwords == 5) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[3] >> xc)) Complain("invalid x-center parameter for model sheet");
								if (!(ws[4] >> yc)) Complain("invalid y-center parameter for model sheet");
							}
						}
						param_vals.input(4);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=kappa; param_vals[1]=xc; param_vals[2]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[3]=zl_in;
						else param_vals[3]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 1 : 3;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[1] != "0") or (words[2] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for one parameter (kappa) in model sheet";
								}
								else complain_str = "Must specify vary flags for three parameters (kappa,xc,yc) in model sheet";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_mass_sheet_lens(zl_in, reference_source_redshift, kappa, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
						}
					}
					else Complain("sheet requires at least 1 parameter, kappa (Einstein radius)");
				}
				else if (words[1]=="shear")
				{
					if (add_shear==true) Complain("cannot add shear to shear model");
					if (nwords > 6) Complain("more than 7 parameters not allowed for model shear");
					if (nwords >= 3) {
						double shear_p1, shear_p2 = 0, xc = 0, yc = 0;
						if (!(ws[2] >> shear_p1)) {
							if (Shear::use_shear_component_params) Complain("invalid shear_1 parameter for model shear");
							else Complain("invalid shear parameter for model shear");
						}
						if (nwords >= 4) {
							if (!(ws[3] >> shear_p2)) {
								if (Shear::use_shear_component_params) Complain("invalid shear_2 parameter for model shear");
								else Complain("invalid theta parameter for model shear");
							}
							if (nwords == 5) {
								if (words[4].find("anchor_center=")==0) {
									string anchorstr = words[4].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate entered for center, but not y-coordinate");
							}
							if (nwords == 6) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[4] >> xc)) Complain("invalid x-center parameter for model shear");
								if (!(ws[5] >> yc)) Complain("invalid y-center parameter for model shear");
							}
						}
						param_vals.input(5);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=shear_p1; param_vals[1]=shear_p2; param_vals[2]=xc; param_vals[3]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[4]=zl_in;
						else param_vals[4]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 2 : 4;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != nparams_to_vary) {
								if (anchor_lens_center) {
									if (nwords==4) {
										if ((words[2] != "0") or (words[3] != "0")) Complain("center coordinates cannot be varied as free parameters if anchored to another lens");
									} else Complain("Must specify vary flags for two parameters (shear,shear_p2) in model shear");
								}
								else Complain("Must specify vary flags for four parameters (shear,shear_p2,xc,yc) in model shear");
							}
							vary_flags.input(nparams_to_vary+1);
							bool invalid_params = false;
							int i;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							add_shear_lens(zl_in, reference_source_redshift, shear_p1, shear_p2, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
						}
					}
					else {
						if (Shear::use_shear_component_params) Complain("shear requires 4 parameters (shear_1, shear_2, xc, yc)");
						else Complain("shear requires 4 parameters (shear, q, xc, yc)");
					}
				}
				else if (words[1]=="tab")
				{
					string filename;
					int lnum;
					bool tabulate_existing_lens = false;
					if ((nwords >= 3) and (!update_parameters)) {
						if (words[2].find("lens=")==0) {
							string lstr = words[2].substr(5);
							stringstream lstream;
							lstream << lstr;
							if (!(lstream >> lnum)) Complain("invalid lens number to tabulate for model tab");
							if (lnum >= nlens) Complain("lens number to tabulate does not exist");
							tabulate_existing_lens = true;
						}
						else {
							if (!(ws[2] >> filename)) Complain("invalid file name for loading interpolation tables for model tab");
						}
						stringstream* new_ws = new stringstream[nwords-1];
						words.erase(words.begin()+2);
						for (int i=0; i < nwords-1; i++) {
							new_ws[i] << words[i];
						}
						delete[] ws;
						ws = new_ws;
						nwords--;
					}

					if (nwords > 7) Complain("more than 5 parameters not allowed for model tab");
					if (nwords >= 4) {
						double kscale, rscale, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> kscale)) Complain("invalid kscale parameter for model tab");
						if (!(ws[3] >> rscale)) Complain("invalid rscale parameter for model tab");
						if (nwords >= 5) {
							if (!(ws[4] >> theta)) Complain("invalid theta parameter for model tab");
							if (nwords == 6) {
								if (words[5].find("anchor_center=")==0) {
									string anchorstr = words[5].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == 7) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[5] >> xc)) Complain("invalid x-center parameter for model tab");
								if (!(ws[6] >> yc)) Complain("invalid y-center parameter for model tab");
							}
						}
						param_vals.input(6);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=kscale; param_vals[1]=rscale; param_vals[2]=theta; param_vals[3]=xc; param_vals[4]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[5]=zl_in;
						else param_vals[5]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 3 : 5;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[tot_nparams_to_vary] != "0") or (words[tot_nparams_to_vary+1] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for three parameters (kscale,rscale,theta) in model tab";
								}
								else complain_str = "Must specify vary flags for five parameters (kscale,rscale,theta,xc,yc) in model tab";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							if (tabulate_existing_lens) {
								add_tabulated_lens(zl_in, reference_source_redshift, lnum, kscale, rscale, theta, xc, yc);
							} else {
								if (!add_tabulated_lens_from_file(zl_in, reference_source_redshift, kscale, rscale, theta, xc, yc, filename)) Complain("input file for tabulated model either does not exist, or is in incorrect format");
							}
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (tabulate_existing_lens) remove_lens(lnum);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("tab requires at least 2 parameters (kscale, rscale)");
				}
				else if (words[1]=="qtab")
				{
					string filename;
					int lnum;
					bool qtabulate_existing_lens = false;
					if ((nwords >= 3) and (!update_parameters)) {
						if (words[2].find("lens=")==0) {
							string lstr = words[2].substr(5);
							stringstream lstream;
							lstream << lstr;
							if (!(lstream >> lnum)) Complain("invalid lens number to qtabulate for model qtab");
							if (lnum >= nlens) Complain("lens number to qtabulate does not exist");
							qtabulate_existing_lens = true;
						}
						else {
							if (!(ws[2] >> filename)) Complain("invalid file name for loading interpolation qtables for model qtab");
						}
						stringstream* new_ws = new stringstream[nwords-1];
						words.erase(words.begin()+2);
						for (int i=0; i < nwords-1; i++) {
							new_ws[i] << words[i];
						}
						delete[] ws;
						ws = new_ws;
						nwords--;
					}

					if (nwords > 8) Complain("more than 6 parameters not allowed for model qtab");
					if (nwords >= 5) {
						double kscale, rscale, q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> kscale)) Complain("invalid kscale parameter for model qtab");
						if (!(ws[3] >> rscale)) Complain("invalid rscale parameter for model qtab");
						if (!(ws[4] >> q)) Complain("invalid q parameter for model qtab");
						if (nwords >= 6) {
							if (!(ws[5] >> theta)) Complain("invalid theta parameter for model qtab");
							if (nwords == 7) {
								if (words[6].find("anchor_center=")==0) {
									string anchorstr = words[6].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == 8) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[6] >> xc)) Complain("invalid x-center parameter for model qtab");
								if (!(ws[7] >> yc)) Complain("invalid y-center parameter for model qtab");
							}
						}
						param_vals.input(7);
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_lens_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=kscale; param_vals[1]=rscale; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? 4 : 6;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[tot_nparams_to_vary] != "0") or (words[tot_nparams_to_vary+1] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for three parameters (kscale,rscale,theta) in model qtab";
								}
								else complain_str = "Must specify vary flags for five parameters (kscale,rscale,theta,xc,yc) in model qtab";
								if ((add_shear) and (nwords != tot_nparams_to_vary)) {
									complain_str += ",\n     plus two shear parameters ";
									complain_str += ((Shear::use_shear_component_params) ? "(shear1,shear2)" : "(shear,angle)");
								}
								if (complain_str != "") Complain(complain_str);
							}
							vary_flags.input(nparams_to_vary+1);
							if (add_shear) shear_vary_flags.input(2);
							bool invalid_params = false;
							int i,j;
							for (i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
							for (i=nparams_to_vary, j=0; i < tot_nparams_to_vary; i++, j++) if (!(ws[i] >> shear_vary_flags[j])) invalid_params = true;
							if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
							for (i=0; i < parameter_anchor_i; i++) if (vary_flags[parameter_anchors[i].paramnum]==true) Complain("Vary flag for anchored parameter must be set to 0");
							vary_flags[nparams_to_vary] = vary_zl;
						}
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
							update_anchored_parameters_and_redshift_data();
							reset();
							if (auto_ccspline) automatically_determine_ccspline_mode();
						} else {
							if (qtabulate_existing_lens) {
								add_qtabulated_lens(zl_in, reference_source_redshift, lnum, kscale, rscale, q, theta, xc, yc);
							} else {
								if (!add_qtabulated_lens_from_file(zl_in, reference_source_redshift, kscale, rscale, q, theta, xc, yc, filename)) Complain("input file for qtabulated model either does not exist, or is in incorrect format");
							}
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(lens_list,anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_anchor_ratio,lens_list[parameter_anchors[i].anchor_lens_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (qtabulate_existing_lens) remove_lens(lnum);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("qtab requires at least 3 parameters (ks, rs, q)");
				}
				else if (words[1]=="testmodel")
				{
					if (nwords > 6) Complain("more than 7 parameters not allowed for model testmodel");
					if (nwords >= 3) {
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> q)) Complain("invalid q parameter for model testmodel");
						if (nwords >= 4) {
							if (!(ws[3] >> theta)) Complain("invalid theta parameter for model testmodel");
							if (nwords == 5) Complain("x-coordinate specified for center, but not y-coordinate");
							if (nwords == 6) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[4] >> xc)) Complain("invalid x-center parameter for model testmodel");
								if (!(ws[5] >> yc)) Complain("invalid y-center parameter for model testmodel");
							}
						}
						add_lens(TESTMODEL, emode, zl_in, reference_source_redshift, 0, 0, 0, q, theta, xc, yc);
						if (vary_parameters) Complain("vary parameters not supported for testmodel");
						if (is_perturber) lens_list[nlens-1]->set_perturber(true);
						if (auto_set_primary_lens) set_primary_lens();
					}
					else Complain("testmodel requires 4 parameters (q, theta, xc, yc)");
				}
				else Complain("unrecognized lens model");
				if ((vary_parameters) and ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST))) {
					int nvary=0;
					if (vary_zl) nparams_to_vary++;
					for (int i=0; i < nparams_to_vary; i++) if (vary_flags[i]==true) nvary++;
					if (nvary != 0) {
						dvector lower(nvary), upper(nvary), lower_initial(nvary), upper_initial(nvary);
						vector<string> paramnames;
						lens_list[nlens-1]->get_fit_parameter_names(paramnames);
						int i,j;
						for (i=0, j=0; j < nparams_to_vary; j++) {
							if (vary_flags[j]) {
								if ((mpi_id==0) and (verbal_mode)) cout << "limits for parameter " << paramnames[i] << ":\n";
								if (read_command(false)==false) { remove_lens(nlens-1); Complain("parameter limits could not be read"); }
								if (nwords >= 2) {
									if (!(ws[0] >> lower[i])) { remove_lens(nlens-1); Complain("invalid lower limit"); }
									if (!(ws[1] >> upper[i])) { remove_lens(nlens-1); Complain("invalid upper limit"); }
									if (nwords == 2) {
										lower_initial[i] = lower[i];
										upper_initial[i] = upper[i];
									} else if (nwords == 3) {
										double width;
										if (!(ws[2] >> width)) { remove_lens(nlens-1); Complain("invalid initial parameter width"); }
										lower_initial[i] = param_vals[j] - width;
										upper_initial[i] = param_vals[j] + width;
									} else if (nwords == 4) {
										if (!(ws[2] >> lower_initial[i])) { remove_lens(nlens-1); Complain("invalid initial lower limit"); }
										if (!(ws[3] >> upper_initial[i])) { remove_lens(nlens-1); Complain("invalid initial upper limit"); }
									} else {
										remove_lens(nlens-1); Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
									}
								} else {
									remove_lens(nlens-1); Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
								}
								if (lower_initial[i] < lower[i]) lower_initial[i] = lower[i];
								if (upper_initial[i] > upper[i]) upper_initial[i] = upper[i];
								i++;
							}
						}
						lens_list[nlens-1]->set_limits(lower,upper,lower_initial,upper_initial);
					}
				}
				if (add_shear) {
					add_shear_lens(zl_in, reference_source_redshift, shear_param_vals[0], shear_param_vals[1], 0, 0);
					lens_list[nlens-1]->anchor_center_to_lens(lens_list,nlens-2);
					if (vary_parameters) {
						boolvector shear_vary_flags_extended; // extra field for redshift, which we don't vary by default for external shear
						shear_vary_flags_extended.input(3);
						shear_vary_flags_extended[0] = shear_vary_flags[0];
						shear_vary_flags_extended[1] = shear_vary_flags[1];
						shear_vary_flags_extended[2] = false;
						set_lens_vary_parameters(nlens-1,shear_vary_flags_extended);
					}
					if ((vary_parameters) and ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod==POLYCHORD) or (fitmethod==MULTINEST))) {
						int nvary_shear=0;
						for (int i=0; i < 2; i++) if (shear_vary_flags[i]==true) nvary_shear++;
						if (nvary_shear==0) continue;
						dvector lower(nvary_shear), upper(nvary_shear), lower_initial(nvary_shear), upper_initial(nvary_shear);
						vector<string> paramnames;
						lens_list[nlens-1]->get_fit_parameter_names(paramnames);
						int i,j;
						for (j=0, i=0; j < 2; j++) {
							if (shear_vary_flags[j]) {
								if ((mpi_id==0) and (verbal_mode)) cout << "Limits for parameter " << paramnames[i] << ":\n";
								if (read_command(false)==false) { remove_lens(nlens-1); Complain("parameter limits could not be read"); }
								if (nwords >= 2) {
									if (!(ws[0] >> lower[i])) { remove_lens(nlens-1); Complain("invalid lower limit"); }
									if (!(ws[1] >> upper[i])) { remove_lens(nlens-1); Complain("invalid upper limit"); }
									if (nwords == 2) {
										lower_initial[i] = lower[i];
										upper_initial[i] = upper[i];
									} else if (nwords == 3) {
										double width;
										if (!(ws[2] >> width)) { remove_lens(nlens-1); Complain("invalid initial parameter width"); }
										lower_initial[i] = shear_param_vals[j] - width;
										upper_initial[i] = shear_param_vals[j] + width;
									} else if (nwords == 4) {
										if (!(ws[2] >> lower_initial[i])) { remove_lens(nlens-1); Complain("invalid initial lower limit"); }
										if (!(ws[3] >> upper_initial[i])) { remove_lens(nlens-1); Complain("invalid initial upper limit"); }
									} else {
										remove_lens(nlens-1); Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
									}
								} else {
									remove_lens(nlens-1); Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
								}
								if (lower_initial[i] < lower[i]) lower_initial[i] = lower[i];
								if (upper_initial[i] > upper[i]) upper_initial[i] = upper[i];
								i++;
							}
						}
						lens_list[nlens-1]->set_limits(lower,upper,lower_initial,upper_initial);
					}
				}
			}
		}
		else if ((words[0]=="source") or ((words[0]=="fit") and (nwords > 1) and (words[1]=="source")))
		{
			bool update_parameters = false;
			bool update_specific_parameters = false; // option for user to update one (or more) specific parameters rather than update all of them at once
			bool vary_parameters = false;
			boolvector vary_flags;
			vector<string> specific_update_params;
			vector<double> specific_update_param_vals;
			dvector param_vals;
			int nparams_to_vary;
			int src_number;

			if (words[0]=="fit") {
				if (source_fit_mode != Parameterized_Source) Complain("cannot vary parameters for source object unless 'fit source_mode' is set to 'sbprofile'");
				vary_parameters = true;
				// now remove the "fit" word from the line so we can add sources the same way,
				// but with vary_parameters==true so we'll prompt for an extra line to vary parameters
				stringstream* new_ws = new stringstream[nwords-1];
				for (int i=0; i < nwords-1; i++) {
					words[i] = words[i+1];
					new_ws[i] << words[i];
				}
				words.pop_back();
				nwords--;
				delete[] ws;
				ws = new_ws;
			}
			SB_ProfileName profile_name;
			if ((nwords > 1) and (words[1]=="update")) {
				if (nwords > 2) {
					if (!(ws[2] >> src_number)) Complain("invalid source number");
					if ((n_sb <= src_number) or (src_number < 0)) Complain("specified source number does not exist");
					update_parameters = true;
					profile_name = sb_list[src_number]->get_sbtype();
					// Now we'll remove the "update" word and replace the source number with the source name
					// so it follows the format of the usual "source" command, but with update_parameters==true
					stringstream* new_ws = new stringstream[nwords-1];
					words[1] = (profile_name==SB_SPLINE) ? "sbspline" :
									(profile_name==GAUSSIAN) ? "gaussian" :
									(profile_name==SERSIC) ? "sersic" : "tophat";
					for (int i=2; i < nwords-1; i++)
						words[i] = words[i+1];
					for (int i=0; i < nwords-1; i++)
						new_ws[i] << words[i];
					words.pop_back();
					nwords--;
					delete[] ws;
					ws = new_ws;
				} else Complain("must specify a source number to update, followed by parameters");
			}

			if (update_parameters) {
				int pos, n_updates = 0;
				double pval;
				for (int i=2; i < nwords; i++) {
					if ((pos = words[i].find("="))!=string::npos) {
						n_updates++;
						specific_update_params.push_back(words[i].substr(0,pos));
						stringstream pvalstr;
						pvalstr << words[i].substr(pos+1);
						pvalstr >> pval;
						specific_update_param_vals.push_back(pval);
					} else if (i==2) break;
				}
				if (n_updates > 0) {
					if (n_updates < nwords-2) Complain("source parameters must all be updated at once, or else specific parameters using '<param>=...'");
					update_specific_parameters = true;
					for (int i=0; i < n_updates; i++)
						if (sb_list[src_number]->update_specific_parameter(specific_update_params[i],specific_update_param_vals[i])==false) Complain("could not find parameter '" << specific_update_params[i] << "' in source " << src_number);
				}
			}

			if (update_specific_parameters) ;
			else if (nwords==1) {
				print_source_list(false);
			}
			else if (words[1]=="clear")
			{
				if (nwords==2) {
					clear_source_objects();
				} else if (nwords==3) {
					int source_number;
					if (!(ws[2] >> source_number)) Complain("invalid source number");
					remove_source_object(source_number);
				} else Complain("source clear command requires either one or zero arguments");
			}
			else if (words[1]=="gaussian")
			{
				if (nwords > 8) Complain("more than 7 parameters not allowed for model gaussian");
				if (nwords >= 5) {
					double sbnorm, sig;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> sbnorm)) Complain("invalid surface brightness normalization parameter for model gaussian");
					if (!(ws[3] >> sig)) Complain("invalid sigma parameter for model gaussian");
					if (!(ws[4] >> q)) Complain("invalid q parameter for model gaussian");
					if (nwords >= 6) {
						if (!(ws[5] >> theta)) Complain("invalid theta parameter for model gaussian");
						if (nwords == 8) {
							if (!(ws[6] >> xc)) Complain("invalid x-center parameter for model gaussian");
							if (!(ws[7] >> yc)) Complain("invalid y-center parameter for model gaussian");
						}
					}
					nparams_to_vary = 6;
					param_vals.input(nparams_to_vary);
					param_vals[0]=sbnorm; param_vals[1]=sig; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;

					if (vary_parameters) {
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for six parameters (sbmax,sigma,q,theta,xc,yc) in model gaussian");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						add_source_object(GAUSSIAN, sbnorm, sig, 0, q, theta, xc, yc);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
					}
				}
				else Complain("gaussian requires at least 3 parameters (max_sb, sig, q)");
			}
			else if (words[1]=="sersic")
			{
				if (nwords > 9) Complain("more than 8 parameters not allowed for model sersic");
				if (nwords >= 6) {
					double s0, reff, n;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> s0)) Complain("invalid s0 parameter for model sersic");
					if (!(ws[3] >> reff)) Complain("invalid R_eff parameter for model sersic");
					if (!(ws[4] >> n)) Complain("invalid n parameter for model sersic");
					if (!(ws[5] >> q)) Complain("invalid q parameter for model sersic");
					if (nwords >= 7) {
						if (!(ws[6] >> theta)) Complain("invalid theta parameter for model sersic");
						if (nwords == 9) {
							if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model sersic");
							if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model sersic");
						}
					}

					nparams_to_vary = 7;
					param_vals.input(nparams_to_vary);
					param_vals[0]=s0; param_vals[1]=reff; param_vals[2] = n; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;

					if (vary_parameters) {
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for six parameters (s0,Reff,q,theta,xc,yc) in model sersic");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						add_source_object(SERSIC, s0, reff, n, q, theta, xc, yc);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
					}
				}
				else Complain("sersic requires at least 4 parameters (max_sb, k, n, q)");
			}
			else if (words[1]=="tophat")
			{
				if (nwords > 8) Complain("more than 7 parameters not allowed for model tophat");
				if (nwords >= 5) {
					double sb, rad;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> sb)) Complain("invalid surface brightness normalization parameter for model tophat");
					if (!(ws[3] >> rad)) Complain("invalid radius parameter for model tophat");
					if (!(ws[4] >> q)) Complain("invalid q parameter for model tophat");
					if (nwords >= 6) {
						if (!(ws[5] >> theta)) Complain("invalid theta parameter for model tophat");
						if (nwords == 8) {
							if (!(ws[6] >> xc)) Complain("invalid x-center parameter for model tophat");
							if (!(ws[7] >> yc)) Complain("invalid y-center parameter for model tophat");
						}
					}

					nparams_to_vary = 6;
					param_vals.input(nparams_to_vary);
					param_vals[0]=sb; param_vals[1]=rad; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;

					if (vary_parameters) {
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for six parameters (sb,radius,q,theta,xc,yc) in model tophat");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						add_source_object(TOPHAT, sb, rad, 0, q, theta, xc, yc);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
					}
				}
				else Complain("tophat requires at least 3 parameters (sb, radius, q)");
			}
			else if (words[1]=="spline")
			{
				if (nwords > 9) Complain("more than 7 parameters not allowed for model spline");
				if (nwords >= 4) {
					double q, theta = 0, xc = 0, yc = 0, qx = 1, f = 1;
					if (!(ws[3] >> q)) Complain("invalid q parameter");
					if (nwords >= 5) {
						if (!(ws[4] >> theta)) Complain("invalid theta parameter");
						if (nwords >= 7) {
							if (!(ws[5] >> qx)) Complain("invalid qx parameter for model spline");
							if (!(ws[6] >> f)) Complain("invalid f parameter for model spline");
							if (nwords == 8) {
								 Complain("x-coordinate specified for center, but not y-coordinate");
							}
							else if (nwords == 9) {
								if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model spline");
								if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model spline");
							}
						} else if (nwords == 6) Complain("must specify qx and f parameters together");
					}
					add_source_object(words[2].c_str(), q, theta, qx, f, xc, yc);
				}
				else Complain("spline requires at least 2 parameters (filename, q)");
			}
			else Complain("source model not recognized");
			if ((vary_parameters) and ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST))) {
				int nvary=0;
				for (int i=0; i < nparams_to_vary; i++) if (vary_flags[i]==true) nvary++;
				if (nvary != 0) {
					dvector lower(nvary), upper(nvary), lower_initial(nvary), upper_initial(nvary);
					vector<string> paramnames;
					sb_list[n_sb-1]->get_fit_parameter_names(paramnames);
					int i,j;
					for (i=0, j=0; j < nparams_to_vary; j++) {
						if (vary_flags[j]) {
							if ((mpi_id==0) and (verbal_mode)) cout << "limits for parameter " << paramnames[i] << ":\n";
							if (read_command(false)==false) { remove_source_object(n_sb-1); Complain("parameter limits could not be read"); }
							if (nwords >= 2) {
								if (!(ws[0] >> lower[i])) { remove_source_object(n_sb-1); Complain("invalid lower limit"); }
								if (!(ws[1] >> upper[i])) { remove_source_object(n_sb-1); Complain("invalid upper limit"); }
								if (nwords == 2) {
									lower_initial[i] = lower[i];
									upper_initial[i] = upper[i];
								} else if (nwords == 3) {
									double width;
									if (!(ws[2] >> width)) { remove_source_object(n_sb-1); Complain("invalid initial parameter width"); }
									lower_initial[i] = param_vals[j] - width;
									upper_initial[i] = param_vals[j] + width;
								} else if (nwords == 4) {
									if (!(ws[2] >> lower_initial[i])) { remove_source_object(n_sb-1); Complain("invalid initial lower limit"); }
									if (!(ws[3] >> upper_initial[i])) { remove_source_object(n_sb-1); Complain("invalid initial upper limit"); }
								} else {
									remove_source_object(n_sb-1); Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
								}
							} else {
								remove_source_object(n_sb-1); Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
							}
							if (lower_initial[i] < lower[i]) lower_initial[i] = lower[i];
							if (upper_initial[i] > upper[i]) upper_initial[i] = upper[i];
							i++;
						}
					}
					sb_list[n_sb-1]->set_limits(lower,upper,lower_initial,upper_initial);
				}
			}
		}
		else if (words[0]=="fit")
		{
			// Note: the "fit lens" command is handled along with the "lens" command above (not here), since the two commands overlap extensively
			if (nwords == 1) {
				if (mpi_id==0) print_fit_model();
			} else {
				if (words[1]=="method") {
					if (nwords==2) {
						if (mpi_id==0) {
							if (fitmethod==POWELL) cout << "Fit method: powell" << endl;
							else if (fitmethod==SIMPLEX) cout << "Fit method: simplex" << endl;
							else if (fitmethod==NESTED_SAMPLING) cout << "Fit method: nest" << endl;
							else if (fitmethod==POLYCHORD) cout << "Fit method: polychord" << endl;
							else if (fitmethod==MULTINEST) cout << "Fit method: multinest" << endl;
							else if (fitmethod==TWALK) cout << "Fit method: twalk" << endl;
							else {
								cout << "Unknown fit method" << endl;
							}
						}
					} else if (nwords==3) {
						if (!(ws[2] >> setword)) Complain("invalid argument to 'fit method' command; must specify valid fit method");
						if (setword=="powell") set_fitmethod(POWELL);
						else if (setword=="simplex") set_fitmethod(SIMPLEX);
						else if (setword=="nest") set_fitmethod(NESTED_SAMPLING);
						else if (setword=="twalk") set_fitmethod(TWALK);
						else if (setword=="polychord") {
#ifdef USE_POLYCHORD
							set_fitmethod(POLYCHORD);
#else
							Complain("qlens code needs to be compiled with PolyChord to use this fit method");
#endif
						}
						else if (setword=="multinest") {
#ifdef USE_MULTINEST
							set_fitmethod(MULTINEST);
#else
							Complain("qlens code needs to be compiled with MultiNest to use this fit method");
#endif
						}
						else Complain("invalid argument to 'fit method' command; must specify valid fit method");
					} else Complain("invalid number of arguments; can only specify fit method type");
				}
				else if (words[1]=="regularization") {
					if (nwords==2) {
						if (mpi_id==0) {
							if (regularization_method==None) cout << "Regularization method: none" << endl;
							else if (regularization_method==Norm) cout << "Regularization method: norm" << endl;
							else if (regularization_method==Gradient) cout << "Regularization method: gradient" << endl;
							else if (regularization_method==Curvature) cout << "Regularization method: curvature" << endl;
							else if (regularization_method==Image_Plane_Curvature) cout << "Regularization method: image plane curvature" << endl;
							else cout << "Unknown regularization method" << endl;
						}
					} else if (nwords==3) {
						if (!(ws[2] >> setword)) Complain("invalid argument to 'fit regularization' command; must specify valid regularization method");
						if (setword=="none") regularization_method = None;
						else if (setword=="norm") regularization_method = Norm;
						else if (setword=="gradient") regularization_method = Gradient;
						else if (setword=="curvature") regularization_method = Curvature;
						else if (setword=="image_plane_curvature") regularization_method = Image_Plane_Curvature;
						else Complain("invalid argument to 'fit regularization' command; must specify valid regularization method");
					} else Complain("invalid number of arguments; can only specify regularization method");
				}
				else if (words[1]=="sourcept")
				{
					if (n_sourcepts_fit==0) Complain("No image data has been loaded");
					if ((nwords==3) and (words[2]=="auto")) {
						if (nlens==0) Complain("No lens model has been defined; cannot determine optimal source point");
						lensvector *srcpts = new lensvector[n_sourcepts_fit];
						output_analytic_srcpos(srcpts);
						for (int i=0; i < n_sourcepts_fit; i++) {
							sourcepts_fit[i][0] = srcpts[i][0];
							sourcepts_fit[i][1] = srcpts[i][1];
						}
						delete[] srcpts;
					}
					else if (nwords==4)
					{
						if (n_sourcepts_fit > 1) Complain("with more than one source point, coordinates must be entered on separate lines after 'fit sourcept'");
						double xs, ys;
						if (!(ws[2] >> xs)) Complain("Invalid x-coordinate for initial source point");
						if (!(ws[3] >> ys)) Complain("Invalid y-coordinate for initial source point");
						sourcepts_fit[0][0] = xs;
						sourcepts_fit[0][1] = ys;
						if ((fitmethod != POWELL) and (fitmethod != SIMPLEX))
						{
							if (mpi_id==0) cout << "Limits for x-coordinate of source point:\n";
							if (read_command(false)==false) return;
							double xmin,xmax,ymin,ymax;
							if (nwords != 2) Complain("Must specify two arguments for source point x-coordinate limits: xmin, xmax");
							if (!(ws[0] >> xmin)) Complain("Invalid lower limit for x-coordinate of source point");
							if (!(ws[1] >> xmax)) Complain("Invalid upper limit for x-coordinate of source point");
							if (xmin > xmax) Complain("lower limit cannot be greater than upper limit");
							if (mpi_id==0) cout << "Limits for y-coordinate of source point:\n";
							if (read_command(false)==false) return;
							if (nwords != 2) Complain("Must specify two arguments for source point y-coordinate limits: ymin, ymax");
							if (!(ws[0] >> ymin)) Complain("Invalid lower limit for y-coordinate of source point");
							if (!(ws[1] >> ymax)) Complain("Invalid upper limit for y-coordinate of source point");
							if (ymin > ymax) Complain("lower limit cannot be greater than upper limit");
							sourcepts_lower_limit[0][0] = xmin;
							sourcepts_upper_limit[0][0] = xmax;
							sourcepts_lower_limit[0][1] = ymin;
							sourcepts_upper_limit[0][1] = ymax;
						}
						if (use_analytic_bestfit_src) {
							if (mpi_id==0) warn(warnings,"with 'analytic_bestfit_src' turned on, the source position(s) are not varied as free parameters since\nthe best-fit values are solved for analytically (to disable this, turn 'analytic_bestfit_src' off).\n");
						}
					} else if ((nwords==2) or (nwords==3)) {
						int imin = 0, imax = n_sourcepts_fit;
						if (nwords==3) {
							int srcpt_num;
							if (!(ws[2] >> srcpt_num)) Complain("Invalid index number for source point");
							if ((srcpt_num >= n_sourcepts_fit) or (srcpt_num < 0)) Complain("Source point number " << srcpt_num << " does not exist");
							imin=srcpt_num;
							imax=srcpt_num+1;
						}
						double xs, ys;
						for (int i=imin; i < imax; i++) {
							if ((verbal_mode) and (mpi_id==0)) cout << "Source point " << i << ":\n";
							if (read_command(false)==false) return;
							if (nwords != 2) Complain("Must specify two coordinates for initial source point");
							if (!(ws[0] >> xs)) Complain("Invalid x-coordinate for initial source point");
							if (!(ws[1] >> ys)) Complain("Invalid y-coordinate for initial source point");
							if ((fitmethod != POWELL) and (fitmethod != SIMPLEX))
							{
								if (mpi_id==0) cout << "Limits for x-coordinate of source point " << i << ":\n";
								if (read_command(false)==false) return;
								double xmin,xmax,ymin,ymax;
								if (nwords != 2) Complain("Must specify two arguments for source point x-coordinate limits: xmin, xmax");
								if (!(ws[0] >> xmin)) Complain("Invalid lower limit for x-coordinate of source point");
								if (!(ws[1] >> xmax)) Complain("Invalid upper limit for x-coordinate of source point");
								if (mpi_id==0) cout << "Limits for y-coordinate of source point " << i << ":\n";
								if (read_command(false)==false) return;
								if (nwords != 2) Complain("Must specify two arguments for source point y-coordinate limits: ymin, ymax");
								if (!(ws[0] >> ymin)) Complain("Invalid lower limit for y-coordinate of source point");
								if (!(ws[1] >> ymax)) Complain("Invalid upper limit for y-coordinate of source point");
								sourcepts_lower_limit[i][0] = xmin;
								sourcepts_upper_limit[i][0] = xmax;
								sourcepts_lower_limit[i][1] = ymin;
								sourcepts_upper_limit[i][1] = ymax;
							}
							sourcepts_fit[i][0] = xs;
							sourcepts_fit[i][1] = ys;
						}
						if (use_analytic_bestfit_src) {
							if (mpi_id==0) warn(warnings,"with 'analytic_bestfit_src' turned on, the source position(s) are not varied as free parameters since\nthe best-fit values are solved for analytically (to disable this, turn 'analytic_bestfit_src' off).\n");
						}
					} else Complain("Must specify either zero or two arguments (sourcept_x, sourcept_y)");
				}
				else if (words[1]=="vary_sourcept")
				{
					if (n_sourcepts_fit==0) Complain("No image data has been loaded");
					if (nwords==4) {
						bool vary_xs, vary_ys;
						if (!(ws[2] >> vary_xs)) Complain("Invalid vary flag for source point");
						if (!(ws[3] >> vary_ys)) Complain("Invalid vary flag for source point");
						vary_sourcepts_x[0] = vary_xs;
						vary_sourcepts_y[0] = vary_ys;
						update_parameter_list();
					} else if (nwords==2) { 
						bool vary_xs, vary_ys;
						for (int i=0; i < n_sourcepts_fit; i++) {
							if ((verbal_mode) and (mpi_id==0)) cout << "Source " << i << ":\n";
							if (read_command(false)==false) return;
							if (nwords != 2) Complain("Must specify two vary flags for source point");
							if (!(ws[0] >> vary_xs)) Complain("Invalid vary flag for source point");
							if (!(ws[1] >> vary_ys)) Complain("Invalid vary flag for source point");
							vary_sourcepts_x[i] = vary_xs;
							vary_sourcepts_y[i] = vary_ys;
						}
						update_parameter_list();
					} else Complain("Invalid number of arguments for source vary flags");
				}
				else if (words[1]=="changevary")
				{
					// At the moment, there is no error checking for changing vary flags of anchored parameters. This should be done from within
					// set_lens_vary_parameters(...), and an integer error code should be returned so specific errors can be printed. Then you should
					// simplify all the error checking in the above code for adding lens models so that errors are printed using the same interface.
					if (nwords != 3) Complain("one argument required for 'fit changevary' (lens number)");
					int lensnum;
					if (!(ws[2] >> lensnum)) Complain("Invalid lens number to change vary parameters");
					if (lensnum >= nlens) Complain("specified lens number does not exist");
					if (read_command(false)==false) return;
					bool vary_zl = check_vary_z(); // this looks for the 'varyz=#' arg. If it finds it, removes it and sets 'vary_zl' to true; if not, sets vary_zl to 'false'
					int nparams_to_vary = nwords;
					boolvector vary_flags(nparams_to_vary+1);
					for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
					vary_flags[nparams_to_vary] = vary_zl;
					int nparam;
					if (set_lens_vary_parameters(lensnum,vary_flags)==false) {
						int npar = lens_list[lensnum]->n_params;
						Complain("number of vary flags does not match number of parameters (" << npar << ") for specified lens");
					}
				}
				else if (words[1]=="source_mode")
				{
					if (nwords==2) {
						if (mpi_id==0) {
							if (source_fit_mode==Point_Source) cout << "Source mode for fitting: ptsource" << endl;
							else if (source_fit_mode==Pixellated_Source) cout << "Source mode for fitting: pixel" << endl;
							else if (source_fit_mode==Parameterized_Source) cout << "Source mode for fitting: sbprofile" << endl;
							else cout << "Unknown fit method" << endl;
						}
					} else if (nwords==3) {
						if (!(ws[2] >> setword)) Complain("invalid argument; must specify valid fit method (ptsource, pixel, sbprofile)");
						if (setword=="ptsource") source_fit_mode = Point_Source;
						else if (setword=="pixel") source_fit_mode = Pixellated_Source;
						else if (setword=="sbprofile") source_fit_mode = Parameterized_Source;
						else Complain("invalid argument; must specify valid source mode (ptsource, pixel, sbprofile)");
					} else Complain("invalid number of arguments; can only specify fit source mode (ptsource, pixel, sbprofile)");
				}
				else if (words[1]=="findimg")
				{
					bool show_all = true;
					int dataset = 0;
					if (n_sourcepts_fit==0) Complain("No data source points have been specified");
					if (sourcepts_fit==NULL) Complain("No initial source point has been specified");
					if (nwords==3) {
						// if using the "src=#" notation (as in 'fit plotimg'), remove the "src=" part
						if (words[2].find("src=") == 0) {
							words[2] = words[2].substr(4);
							ws[2].str(""); ws[2].clear();
							ws[2] << words[2];
						}
						if (!(ws[2] >> dataset)) Complain("invalid image dataset");
						if (dataset >= n_sourcepts_fit) Complain("specified image dataset has not been loaded");
						show_all = false;
					} else if (nwords > 3) Complain("invalid number of arguments; can only specify sourcept number");

					double* srcflux = new double[n_sourcepts_fit];
					lensvector *srcpts = new lensvector[n_sourcepts_fit];
					if (include_flux_chisq) {
						output_model_source_flux(srcflux);
						if (fix_source_flux) srcflux[0] = source_flux;
					} else {
						for (int i=0; i < n_sourcepts_fit; i++) srcflux[i] = -1; // -1 tells it to not print fluxes
					}
					if (use_analytic_bestfit_src) {
						output_analytic_srcpos(srcpts);
					} else {
						for (int i=0; i < n_sourcepts_fit; i++) srcpts[i] = sourcepts_fit[i];
					}

					if (mpi_id==0) cout << endl;
					if (show_all) {
						bool different_zsrc = false;
						for (int i=0; i < n_sourcepts_fit; i++) if (source_redshifts[i] != source_redshift) different_zsrc = true;
						for (int i=0; i < n_sourcepts_fit; i++) {
							if (different_zsrc) {
								if ((i == 0) or (source_redshifts[i] != source_redshifts[i-1])) {
									create_grid(false,zfactors[i],beta_factors[i]);
								}
							}
							if (mpi_id==0) cout << "# Source " << i << ":" << endl;
							output_images_single_source(srcpts[i][0], srcpts[i][1], verbal_mode, srcflux[i], true);
						}
						if (different_zsrc) {
							reset();
							//create_grid(false);
						}
					} else {
						if (source_redshifts[dataset] != source_redshift) {
							reset();
							create_grid(false,zfactors[dataset],beta_factors[dataset]);
						}
						output_images_single_source(srcpts[dataset][0], srcpts[dataset][1], verbal_mode, srcflux[dataset], true);
						if (source_redshifts[dataset] != source_redshift) {
							reset();
							//create_grid(false);
						}
					}
					double rms_err;
					int nmatched;
					// This gets rms error info, but at the cost of having to find all the images again. It would be much better for
					// the chisq_diagnostic function to just return all the image info and run it above, instead of finding all the images separately.
					// You would have to handle the output from here, and do away with the output_images_single_source(...) function which is a bad
					// way to do it anyway.
					// MAKE THIS UPGRADE LATER!!!!!!!!!!!
					chisq_pos_image_plane_diagnostic(false,rms_err,nmatched);
					if (mpi_id==0) cout << "# matched image pairs = " << nmatched << ", rms_imgpos_error = " << rms_err << endl << endl;
					delete[] srcflux;
					delete[] srcpts;
				}
				else if (words[1]=="data_imginfo")
				{
					if (nlens==0) Complain("No lens model has been specified");
					if (n_sourcepts_fit==0) Complain("No data source points have been specified");
					if (sourcepts_fit==NULL) Complain("No initial source point has been specified");
					if ((show_cc) and (plotcrit("crit.dat")==false)) Complain("could not plot critical curves");
					if (nwords != 2) Complain("command 'fit imginfo' does not require any arguments");
					double* srcflux = new double[n_sourcepts_fit];
					lensvector *srcpts = new lensvector[n_sourcepts_fit];
					if (include_flux_chisq) {
						output_model_source_flux(srcflux);
					} else {
						for (int i=0; i < n_sourcepts_fit; i++) srcflux[i] = -1; // -1 tells it to not print fluxes
					}

					if (fix_source_flux) srcflux[0] = source_flux;
					if (use_analytic_bestfit_src) {
						output_analytic_srcpos(srcpts);
					} else {
						for (int i=0; i < n_sourcepts_fit; i++) srcpts[i] = sourcepts_fit[i];
					}
					if (mpi_id==0) cout << endl;
					for (int i=0; i < n_sourcepts_fit; i++) {
						plot_srcpts_from_image_data(i,NULL,srcpts[i][0],srcpts[i][1],srcflux[i]);
					}
					delete[] srcflux;
					delete[] srcpts;
				}
				else if (words[1]=="plotsrc")
				{
					if (nlens==0) Complain("No lens model has been specified");
					if (n_sourcepts_fit==0) Complain("No data source points have been specified");
					if (sourcepts_fit==NULL) Complain("No initial source point has been specified");
					int dataset;
					bool show_multiple = false;
					int min_dataset = 0, max_dataset = n_sourcepts_fit - 1;
					if (nwords==2) show_multiple = true;
					else {
						int pos0, pos;
						if ((pos0 = words[2].find("src="))==0) {
							string srcstring = words[2].substr(pos0+4);
							if ((pos = srcstring.find("-")) != string::npos) {
								string dminstring, dmaxstring;
								dminstring = srcstring.substr(0,pos);
								dmaxstring = srcstring.substr(pos+1);
								stringstream dmaxstream, dminstream;
								dminstream << dminstring;
								dmaxstream << dmaxstring;
								if (!(dminstream >> min_dataset)) Complain("invalid dataset");
								if (!(dmaxstream >> max_dataset)) Complain("invalid dataset");
								if (max_dataset >= n_sourcepts_fit) Complain("specified max image dataset exceeds number of image sets in data");
								show_multiple = true;
							} else {
								string dstr = words[2].substr(4);
								stringstream dstream;
								dstream << dstr;
								if (!(dstream >> dataset)) Complain("invalid dataset");
								if (dataset >= n_sourcepts_fit) Complain("specified image dataset has not been loaded");
							}
							stringstream* new_ws = new stringstream[nwords-1];
							new_ws[0] << words[0];
							new_ws[1] << words[1];
							for (int i=2; i < nwords-1; i++) {
								new_ws[i] << words[i+1];
							}
							words.erase(words.begin()+2);
							delete[] ws;
							ws = new_ws;
							nwords--;
						}
						else show_multiple = true;
					}
					if (show_multiple) {
						reset();
						create_grid(false,reference_zfactors,default_zsrc_beta_factors); // even though we're not finding images, still need to plot caustics
					} else {
						reset();
						create_grid(false,zfactors[dataset],default_zsrc_beta_factors); // even though we're not finding images, still need to plot caustics
					}
					if ((show_cc) and (plotcrit("crit.dat")==false)) Complain("could not plot critical curves and caustics");
					if ((nwords != 3) and (nwords != 2)) Complain("command 'fit plotsrc' requires either zero or one argument (source_filename)");
					double* srcflux = new double[n_sourcepts_fit];
					lensvector *srcpts = new lensvector[n_sourcepts_fit];
					if (include_flux_chisq) {
						output_model_source_flux(srcflux);
					} else {
						for (int i=0; i < n_sourcepts_fit; i++) srcflux[i] = -1; // -1 tells it to not print fluxes
					}

					if (fix_source_flux) srcflux[0] = source_flux;
					if (use_analytic_bestfit_src) {
						output_analytic_srcpos(srcpts);
					} else {
						for (int i=0; i < n_sourcepts_fit; i++) srcpts[i] = sourcepts_fit[i];
					}
					if (mpi_id==0) cout << endl;
					string srcname="srcs.dat";
					bool output_to_text_files = false;
					if ((terminal==TEXT) and (nwords==3)) {
						output_to_text_files = true;
						srcname = words[2];
					}
					ofstream srcfit;
					ofstream srcfile;
					if (mpi_id==0) {
						open_output_file(srcfit,"srcfit.dat");
						open_output_file(srcfile,srcname);
					}
					if (!show_multiple) {
						if (mpi_id==0) {
							if (output_to_text_files) { srcfile << "# "; }
							srcfile << "\"dataset " << dataset << "\"" << endl;
						}
						if (plot_srcpts_from_image_data(dataset,&srcfile,srcpts[dataset][0],srcpts[dataset][1],srcflux[dataset])==true) {
							if (mpi_id==0) {
								srcfit << "\"fit srcpt " << dataset << " (z_{s}=" << source_redshifts[dataset] << ")\"" << endl;
								srcfit << srcpts[dataset][0] << "\t" << srcpts[dataset][1] << endl << endl << endl;
								srcfile << endl << endl;
							}
						}
					} else {
						for (int i=min_dataset; i <= max_dataset; i++) {
							if (mpi_id==0) {
								if (output_to_text_files) { srcfile << "# "; }
								srcfile << "\"dataset " << i << "\"" << endl;
								srcfile << srcpts[i][0] << "\t" << srcpts[i][1] << " # from fit" << endl;
							}
							if (plot_srcpts_from_image_data(i,&srcfile,srcpts[i][0],srcpts[i][1],srcflux[i])==true) {
								if (mpi_id==0) {
									srcfit << "\"fit srcpt " << i << " (z_{s}=" << source_redshifts[i] << ")\"" << endl;
									srcfit << srcpts[i][0] << "\t" << srcpts[i][1] << endl << endl << endl;
									srcfile << endl << endl;
								}
							}
						}
					}
					if (nwords==3) {
						if (terminal != TEXT) {
							if ((show_multiple) and (n_sourcepts_fit > 1)) run_plotter("srcptfits",words[2],"");
							else run_plotter("srcptfit",words[2],"");
						}
					} else {
						if ((show_multiple) and (n_sourcepts_fit > 1)) run_plotter("srcptfits");
						else run_plotter("srcptfit");
					}
					reset();
					create_grid(false,reference_zfactors,default_zsrc_beta_factors);
					delete[] srcflux;
					delete[] srcpts;
				}
				else if (words[1]=="plotimg")
				{
					// this needs to be redone a bit, along with "fit findimg"--it should just call get_images(...) directly, so that it can
					// add up the total number of images, specify how many images are above the stated magnification threshold, etc.
					// also, you should put this entire thing in a function -- also true for many commands in this file! Critical before public release.
					if (nlens==0) Complain("No lens model has been specified");
					if (n_sourcepts_fit==0) Complain("No data source points have been specified");
					if (sourcepts_fit==NULL) Complain("No initial source point has been specified");
					int dataset;
					bool show_multiple = false;
					bool show_grid = false;
					int min_dataset = 0, max_dataset = n_sourcepts_fit - 1;
					if ((nwords > 2) and (words[nwords-1]=="grid")) {
						show_grid = true;
						remove_word(nwords-1);
					}
					if (nwords==2) show_multiple = true;
					else {
						int pos0, pos;
						if ((pos0 = words[2].find("src="))==0) {
							string srcstring = words[2].substr(pos0+4);
							if ((pos = srcstring.find("-")) != string::npos) {
								string dminstring, dmaxstring;
								dminstring = srcstring.substr(0,pos);
								dmaxstring = srcstring.substr(pos+1);
								stringstream dmaxstream, dminstream;
								dminstream << dminstring;
								dmaxstream << dmaxstring;
								if (!(dminstream >> min_dataset)) Complain("invalid dataset");
								if (!(dmaxstream >> max_dataset)) Complain("invalid dataset");
								if (max_dataset >= n_sourcepts_fit) Complain("specified max image dataset exceeds number of image sets in data");
								show_multiple = true;
							} else {
								string dstr = words[2].substr(4);
								stringstream dstream;
								dstream << dstr;
								if (!(dstream >> dataset)) Complain("invalid dataset");
								if (dataset >= n_sourcepts_fit) Complain("specified image dataset has not been loaded");
							}
							stringstream* new_ws = new stringstream[nwords-1];
							new_ws[0] << words[0];
							new_ws[1] << words[1];
							for (int i=2; i < nwords-1; i++) {
								new_ws[i] << words[i+1];
							}
							words.erase(words.begin()+2);
							delete[] ws;
							ws = new_ws;
							nwords--;
						}
						else show_multiple = true;
					}
					// If showing multiple sources, plot critical curves using zsrc
					if ((show_multiple) and (show_cc) and (plotcrit("crit.dat")==false)) Complain("could not plot critical curves");
					if (!show_multiple) {
						reset();
						int zgroup = -1;
						for (int k=0; k < source_redshift_groups.size()-1; k++) { if ((dataset >= source_redshift_groups[k]) and (dataset < source_redshift_groups[k+1])) zgroup = k; }
						create_grid(false,zfactors[dataset],beta_factors[dataset],zgroup);
						if (show_grid) plot_recursive_grid("xgrid.dat");
						// Plot critical curves corresponding to the particular source redshift being plotted
						if ((show_cc) and (plotcrit("crit.dat")==false)) Complain("could not plot critical curves");
					} else {
						reset();
						create_grid(false,zfactors[min_dataset],beta_factors[min_dataset]);
					}
					if ((nwords != 4) and (nwords != 2)) Complain("command 'fit plotimg' requires either zero or two arguments (source_filename, image_filename)");
					double* srcflux = new double[n_sourcepts_fit];
					lensvector *srcpts = new lensvector[n_sourcepts_fit];
					if (include_flux_chisq) {
						output_model_source_flux(srcflux);
					} else {
						for (int i=0; i < n_sourcepts_fit; i++) srcflux[i] = -1; // -1 tells it to not print fluxes
					}

					if (fix_source_flux) srcflux[0] = source_flux;
					if (use_analytic_bestfit_src) {
						output_analytic_srcpos(srcpts);
					} else {
						for (int i=0; i < n_sourcepts_fit; i++) srcpts[i] = sourcepts_fit[i];
					}
					if (mpi_id==0) cout << endl;
					string imgname="imgs.dat", srcname="srcs.dat";
					bool output_to_text_files = false;
					if ((terminal==TEXT) and (nwords==4)) {
						output_to_text_files = true;
						srcname = words[2];
						imgname = words[3];
					}
					ofstream imgout;
					ofstream imgfile;
					ofstream srcfile;
					if (mpi_id==0) {
						open_output_file(imgout,"imgdat.dat");
						open_output_file(imgfile,imgname);
						open_output_file(srcfile,srcname);
					}
					if (!show_multiple) {
						if (mpi_id==0) {
							if (output_to_text_files) { imgfile << "# "; srcfile << "# "; }
							imgfile << "\"image set " << dataset << "\"" << endl;
							srcfile << "\"source " << dataset << "\"" << endl;
						}
						if (plot_images_single_source(srcpts[dataset][0], srcpts[dataset][1], verbal_mode, imgfile, srcfile, srcflux[dataset], true)==true) {
							if (mpi_id==0) {
								imgout << "\"dataset " << dataset << " (z_{s}=" << source_redshifts[dataset] << ")\"" << endl;
								image_data[dataset].write_to_file(imgout);
								imgout << endl << endl;
								imgfile << endl << endl;
								srcfile << endl << endl;
							}
						}
					} else {
						reset();
						for (int i=min_dataset; i <= max_dataset; i++) {
							if ((i == min_dataset) or (zfactors[i] != zfactors[i-1]))
								create_grid(false,zfactors[i],beta_factors[i]);
							if (mpi_id==0) {
								if (output_to_text_files) { imgfile << "# "; srcfile << "# "; }
								imgfile << "\"image set " << i << "\"" << endl;
								srcfile << "\"source " << i << "\"" << endl;
							}
							if (plot_images_single_source(srcpts[i][0], srcpts[i][1], verbal_mode, imgfile, srcfile, srcflux[i], true)==true) {
								if (mpi_id==0) {
									imgout << "\"dataset " << i << " (z_{s}=" << source_redshifts[i] << ")\"" << endl;
									image_data[i].write_to_file(imgout);
									imgout << endl << endl;
									imgfile << endl << endl;
									srcfile << endl << endl;
								}
							}
						}
					}
					if (show_grid) show_imgsrch_grid = true; // show the grid along with the images
					if (nwords==4) {
						if (terminal != TEXT) {
							if (show_cc) {
								if ((show_multiple) and (n_sourcepts_fit > 1)) run_plotter("imgfits",words[3],"");
								else run_plotter("imgfit",words[3],"");
							}
							else run_plotter("imgfit_nocc",words[3],"");
							if (plot_srcplane) {
								if ((show_multiple) and (n_sourcepts_fit > 1)) run_plotter("srcfits",words[2],"");
								else run_plotter("srcfit",words[2],"");
							}
						}
					} else {
						if (show_cc) {
							if ((show_multiple) and (n_sourcepts_fit > 1)) run_plotter("imgfits");
							else run_plotter("imgfit");
						}
						else run_plotter("imgfit_nocc");
						if (plot_srcplane) {
							if ((show_multiple) and (n_sourcepts_fit > 1)) run_plotter("srcfits");
							else run_plotter("srcfit");
						}
					}
					show_imgsrch_grid = false;
					reset();
					create_grid(false,reference_zfactors,default_zsrc_beta_factors);
					delete[] srcflux;
					delete[] srcpts;
				}
				else if (words[1]=="plot_chisq1d")
				{
					if ((nwords >= 6) and (nwords < 8)) {
						int n,p;
						double ip,fp;
						string filename;
						if (!(ws[2] >> p)) Complain("invalid parameter number");
						if (!(ws[3] >> n)) Complain("invalid number of points");
						if (!(ws[4] >> ip)) Complain("invalid initial point");
						if (!(ws[5] >> fp)) Complain("invalid final point");
						if (nwords==7) filename = words[6];
						else filename = "chisq1d.dat";
						plot_chisq_1d(p,n,ip,fp,filename);
					} else Complain("invalid number of parameters for command 'fit plot_chisq1d' (need parameter#,npoints,initial,final)");
				}
				else if (words[1]=="mkposts")
				{
					bool copy_subplot_only = false;
					bool resampled_posts = false;
					bool no2dposts = false;
					int nbins1d = 50, nbins2d = 40;
					if (nwords > 2) {
						for (int i=2; i < nwords; i++) {
							if (words[i]=="-subonly") {
								copy_subplot_only = true;
								remove_word(i);
								break;
							}
						}
						for (int i=2; i < nwords; i++) {
							if (words[i]=="-no2d") {
								no2dposts = true;
								remove_word(i);
								break;
							}
						}
						for (int i=2; i < nwords; i++) {
							if (words[i]=="-new") {
								resampled_posts = true;
								remove_word(i);
							}
						}
						int pos = -1;
						for (int i=2; i < nwords; i++) {
							if (((pos = words[i].find("-n")) != string::npos) and (pos==0)) {
								string nbinstring = words[i].substr(pos+2);
								stringstream nbinstr;
								nbinstr << nbinstring;
								if (!(nbinstr >> nbins1d)) Complain("incorrect format for number of 1d bins for mkdist; should be entered as '-n#'");;
								remove_word(i);
								pos = -1;
								break;
							}
						}
						for (int i=2; i < nwords; i++) {
							if (((pos = words[i].find("-N")) != string::npos) and (pos==0)) {
								string nbinstring = words[i].substr(pos+2);
								stringstream nbinstr;
								nbinstr << nbinstring;
								if (!(nbinstr >> nbins2d)) Complain("incorrect format for number of 2d bins for mkdist; should be entered as '-N#'");;
								remove_word(i);
								pos = -1;
								break;
							}
						}
					}
					if (nwords==2) {
						run_mkdist(false,"",nbins1d,nbins2d,copy_subplot_only,resampled_posts,no2dposts);
					} else if (nwords==3) {
						run_mkdist(true,words[2],nbins1d,nbins2d,copy_subplot_only,resampled_posts,no2dposts);
					} else Complain("either zero/one argument allowed for 'fit mkposts' (directory name, plus optional '-n' or '-N' args)");
				}
				else if (words[1]=="run")
				{
					bool resume = false;
					if (nwords > 2) {
						if ((nwords==3) and (words[2]=="-resume")) resume = true;
						else Complain("invalid arguments after 'fit run'");
					}
					if (fitmethod==POWELL) chi_square_fit_powell();
					else if (fitmethod==SIMPLEX) chi_square_fit_simplex();
					else if (fitmethod==NESTED_SAMPLING) nested_sampling();
					else if (fitmethod==POLYCHORD) polychord();
					else if (fitmethod==MULTINEST) multinest(resume);
					else if (fitmethod==TWALK) chi_square_twalk();
					else Complain("unsupported fit method");
				}
				else if (words[1]=="chisq")
				{
					bool showdiag = false;
					if (nwords == 3) {
						if (words[2]=="diag") showdiag = true;
						else Complain("invalid argument to 'fit chisq'");
					}
					chisq_single_evaluation(showdiag,true);
					clear_raw_chisq(); // in case raw chi-square is being used as a derived parameter
				}
				else if (words[1]=="label")
				{
					if ((nwords == 2) and (mpi_id==0)) cout << "Fit label: " << fit_output_filename << endl;
					else {
						if (nwords != 3) Complain("a single filename must be specified after 'fit label'");
						if (!(ws[2] >> fit_output_filename)) Complain("Invalid fit label");
						if (auto_fit_output_dir) fit_output_dir = "chains_" + fit_output_filename;
					}
				}
				else if (words[1]=="priors")
				{
					int nparams;
					get_n_fit_parameters(nparams);
					if (nparams==0) Complain("no fit parameters have been defined");
					if (nwords==2) { if (mpi_id==0) param_settings->print_priors(); }
					else if (nwords >= 4) {
						int param_num;
						if (!(ws[2] >> param_num)) {
							if ((param_num = param_settings->lookup_param_number(words[2])) == -1)
							Complain("Invalid parameter number/name");
						}
						if (param_num >= nparams) Complain("Parameter number does not exist (see parameter list with 'fit priors')");
						if (words[3]=="uniform") param_settings->priors[param_num]->set_uniform();
						else if (words[3]=="log") param_settings->priors[param_num]->set_log();
						else if (words[3]=="gaussian") {
							if (nwords != 6) Complain("'fit priors gaussian' requires two additional arguments (mean,sigma)");
							double sig, pos;
							if (!(ws[4] >> pos)) Complain("Invalid mean value for Gaussian prior");
							if (!(ws[5] >> sig)) Complain("Invalid dispersion value for Gaussian prior");
							param_settings->priors[param_num]->set_gaussian(pos,sig);
						}
						else if (words[3]=="gauss2") {
							if (nwords != 10) Complain("'fit priors gauss2' requires six additional arguments (pnum,mean1,mean2,sig1,sig2,sig12)");
							int param_num2;
							double mean1, mean2;
							double sig1, sig2, sig12;
							if (!(ws[4] >> param_num2)) Complain("Invalid value for second parameter for multivariate Gaussian prior");
							if (!(ws[5] >> mean1)) Complain("Invalid mean value for Gaussian prior");
							if (!(ws[6] >> mean2)) Complain("Invalid mean value for Gaussian prior");
							if (!(ws[7] >> sig1)) Complain("Invalid sigma1 value for Gaussian prior");
							if (!(ws[8] >> sig2)) Complain("Invalid sigma2 value for Gaussian prior");
							if (!(ws[9] >> sig12)) Complain("Invalid sigma12 value for Gaussian prior");
							param_settings->priors[param_num]->set_gauss2(param_num,param_num2,mean1,mean2,sig1,sig2,sig12);
							param_settings->priors[param_num2]->set_gauss2_secondary(param_num,param_num2);
						}
						else Complain("prior type not recognized");
					}
					else Complain("command 'fit priors' requires either zero or two arguments (param_number,prior_type)");
				}
				else if (words[1]=="transform")
				{
					bool include_jac=false;
					if (words[nwords-1].find("include_jac")==0) {
						include_jac = true;
						stringstream* new_ws = new stringstream[nwords-1];
						words.erase(words.begin()+nwords-1);
						for (int i=0; i < nwords-1; i++) {
							new_ws[i] << words[i];
						}
						delete[] ws;
						ws = new_ws;
						nwords--;
					}
					int nparams;
					get_n_fit_parameters(nparams);
					if (nparams==0) Complain("no fit parameters have been defined");
					if (nwords==2) { if (mpi_id==0) param_settings->print_priors(); }
					else if (nwords >= 4) {
						int param_num;
						if (!(ws[2] >> param_num)) {
							if ((param_num = param_settings->lookup_param_number(words[2])) == -1)
							Complain("Invalid parameter number/name");
						}
						if (param_num >= nparams) Complain("Parameter number does not exist (see parameter list with 'fit transform'");
						if (words[3]=="none") param_settings->transforms[param_num]->set_none();
						else if (words[3]=="log") param_settings->transforms[param_num]->set_log();
						else if (words[3]=="gaussian") {
							if (nwords != 6) Complain("'fit transform gaussian' requires two parameter arguments (mean,sigma)");
							double sig, pos;
							if (!(ws[4] >> pos)) Complain("Invalid mean value for Gaussian transformation");
							if (!(ws[5] >> sig)) Complain("Invalid dispersion value for Gaussian transformation");
							param_settings->transforms[param_num]->set_gaussian(pos,sig);
						}
						else if (words[3]=="linear") {
							if (nwords != 6) Complain("'fit transform linear' requires two additional arguments (A,b)");
							double a,b;
							if (!(ws[4] >> a)) Complain("Invalid A value for linear transformation");
							if (!(ws[5] >> b)) Complain("Invalid b value for Gaussian transformation");
							param_settings->transforms[param_num]->set_linear(a,b);
						}
						else if (words[3]=="ratio") {
							if (nwords != 5) Complain("'fit transform ratio' requires one additional argument (paramnum)");
							int ratio_pnum;
							if (!(ws[4] >> ratio_pnum)) Complain("Invalid parameter number for ratio transformation");
							if ((ratio_pnum >= nparams) or (ratio_pnum < 0)) Complain("Parameter number specified for ratio transformation does not exist");
							param_settings->transforms[param_num]->set_ratio(ratio_pnum);
						}
						else Complain("transformation type not recognized");
						param_settings->transforms[param_num]->set_include_jacobian(include_jac);
						param_settings->transform_stepsizes();
					}
					else Complain("command 'fit transform' requires either zero or two arguments (param_number,transformation_type)");
				}
				else if (words[1]=="stepsizes")
				{
					int nparams;
					get_n_fit_parameters(nparams);
					if (nwords==2) { if (mpi_id==0) param_settings->print_stepsizes(); }
					else if (nwords == 3) {
						if (words[2]=="double") param_settings->scale_stepsizes(2);
						else if (words[2]=="half") param_settings->scale_stepsizes(0.5);
						else if (words[2]=="reset") {
							dvector stepsizes(nparams);
							get_automatic_initial_stepsizes(stepsizes);
							param_settings->reset_stepsizes(stepsizes.array());
						}
						else Complain("argument to 'fit stepsizes' not recognized");
					}
					else if (nwords == 4) {
						if (words[2]=="scale") {
							double fac;
							if (!(ws[3] >> fac)) Complain("Invalid scale factor");
							param_settings->scale_stepsizes(fac);
						} else {
							int param_num;
							double stepsize;
							if (!(ws[2] >> param_num)) {
								if ((param_num = param_settings->lookup_param_number(words[2])) == -1)
								Complain("Invalid parameter number/name");
							}
							if (param_num >= nparams) Complain("Parameter number does not exist (see parameter list with 'fit stepsizes'");
							if (!(ws[3] >> stepsize)) Complain("Invalid stepsize");
							param_settings->set_stepsize(param_num,stepsize);
						}
					}
					else Complain("command 'fit stepsizes' requires up to two arguments (see 'help fit stepsizes')");
				}
				else if (words[1]=="plimits")
				{
					int nparams;
					get_n_fit_parameters(nparams);
					set_default_plimits();
					if (nwords==2) { if (mpi_id==0) param_settings->print_penalty_limits(); }
					else if (nwords == 3) {
						if (words[2]=="reset") {
							param_settings->clear_penalty_limits();
							set_default_plimits();
						}
						else Complain("command not recognized for 'fit plimits'");
					}
					else if (nwords == 4) {
						int param_num;
						if (!(ws[2] >> param_num)) {
							if ((param_num = param_settings->lookup_param_number(words[2])) == -1)
							Complain("Invalid parameter number/name");
						}
						if (!(ws[2] >> param_num)) Complain("Invalid parameter number");
						if (param_num >= nparams) Complain("Parameter number does not exist (see parameter list with 'fit plimits'");
						if (words[3]=="none") param_settings->clear_penalty_limit(param_num);
						else Complain("Setting for penalty limit not recognized; either give limits (low and high) or else 'none'");
					}
					else if (nwords == 5) {
						int param_num;
						double lo, hi;
						if (!(ws[2] >> param_num)) Complain("Invalid parameter number");
						if (param_num >= nparams) Complain("Parameter number does not exist (see parameter list with 'fit plimits'");
						if (words[3]=="-inf") lo=-1e30;
						else if (!(ws[3] >> lo)) Complain("Invalid lower limit");
						if (words[4]=="inf") hi=1e30;
						else if (!(ws[4] >> hi)) Complain("Invalid higher limit");
						param_settings->set_penalty_limit(param_num,lo,hi);
					}
					else Complain("command 'fit plimits' requires either zero or three arguments (see 'help fit plimits')");
				}
				else if (words[1]=="dparams")
				{
					if (nwords==2) { if (mpi_id==0) print_derived_param_list(); }
					else {
						if (words[2]=="clear") {
							if (nwords==3) clear_derived_params();
							else if (nwords==4) {
								int dparam_number;
								if (!(ws[3] >> dparam_number)) Complain("invalid dparam number");
								remove_derived_param(dparam_number);
							} else Complain("only one argument allowed for 'fit dparams clear' (number of derived parameter to remove)");
						}
						else if (words[2]=="add") {
							if (nwords < 4) Complain("at least one additional argument required for 'fit dparams add' (dparam_type)");
							double dparam_arg;
							int lensnum;
							if (words[3]=="kappa_r") {
								if (nwords < 5) Complain("at least one additional argument required for 'fit dparams add kappa_r' (param_arg)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (nwords==6) {
									if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
									if (lensnum >= nlens) Complain("specified lens number does not exist");
								} else lensnum = -1;
								add_derived_param(KappaR,dparam_arg,lensnum);
							} else if (words[3]=="dkappa_r") {
								if (nwords < 5) Complain("at least one additional argument required for 'fit dparams add dkappa_r' (param_arg)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (nwords==6) {
									if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
									if (lensnum >= nlens) Complain("specified lens number does not exist");
								} else lensnum = -1;
								add_derived_param(DKappaR,dparam_arg,lensnum);
							} else if (words[3]=="mass2d_r") {
								if (nwords != 6) Complain("derived parameter mass2d_r requires two arguments (r_arcsec,lens_number)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(Mass2dR,dparam_arg,lensnum);
							} else if (words[3]=="mass3d_r") {
								if (nwords != 6) Complain("derived parameter mass3d_r requires two arguments (r_arcsec,lens_number)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(Mass3dR,dparam_arg,lensnum);
							} else if (words[3]=="re_zsrc") {
								if (nwords != 6) Complain("derived parameter re_zsrc requires two arguments (zsrc,lens_number)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(Einstein,dparam_arg,lensnum);
							} else if (words[3]=="mass_re") {
								if (nwords != 6) Complain("derived parameter mass_re requires two arguments (zsrc,lens_number)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(Einstein_Mass,dparam_arg,lensnum);
							} else if (words[3]=="lensparam") {
								int paramnum;
								if (nwords != 6) Complain("derived parameter lensparam requires two arguments (paramnum,lens_number)");
								if (!(ws[4] >> paramnum)) Complain("invalid derived parameter argument--must be integer (parameter number)");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(LensParam,paramnum,lensnum);
							} else if (words[3]=="r_perturb") {
								if (nwords != 5) Complain("derived parameter r_perturb requires only one arguments (lens_number)");
								if (!(ws[4] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								if (lensnum == 0) Complain("specified lens number cannot be 0 (since lens 0 is assumed to be primary lens)");
								add_derived_param(Perturbation_Radius,0.0,lensnum);
							} else if (words[3]=="r_perturb_rel") {
								if (nwords != 5) Complain("derived parameter r_perturb_rel requires only one arguments (lens_number)");
								if (!(ws[4] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								if (lensnum == 0) Complain("specified lens number cannot be 0 (since lens 0 is assumed to be primary lens)");
								add_derived_param(Relative_Perturbation_Radius,0.0,lensnum);
							} else if (words[3]=="mass_perturb") {
								if (nwords != 5) Complain("derived parameter mass_perturb requires only one arguments (lens_number)");
								if (!(ws[4] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								if (lensnum == 0) Complain("specified lens number cannot be 0 (since lens 0 is assumed to be primary lens)");
								add_derived_param(Robust_Perturbation_Mass,0.0,lensnum);
							} else if (words[3]=="sigma_perturb") {
								if (nwords != 5) Complain("derived parameter sigma_perturb requires only one arguments (lens_number)");
								if (!(ws[4] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								if (lensnum == 0) Complain("specified lens number cannot be 0 (since lens 0 is assumed to be primary lens)");
								add_derived_param(Robust_Perturbation_Density,0.0,lensnum);
							} else if (words[3]=="raw_chisq") {
								if (nwords != 4) Complain("no arguments required for derived param raw_chisq");
								add_derived_param(Chi_Square,0.0,-1);
							} else Complain("derived parameter type not recognized");
						}
						else if (words[2]=="rename") {
							if (nwords != 6) Complain("three arguments required for 'fit dparams rename' (param_number,name,latex_name)");
							int dparam_number;
							if (!(ws[3] >> dparam_number)) Complain("invalid dparam number");
							if ((dparam_number >= n_derived_params) or (n_derived_params == 0)) Complain("Specified derived parameter does not exist");
							rename_derived_param(dparam_number,words[4],words[5]);
						}
						else if (nwords==3) {
							int dpnum;
							if (!(ws[2] >> dpnum)) Complain("invalid dparam number");
							if ((dpnum < 0) or (dpnum >= n_derived_params)) Complain("specified derived parameter has not been defined");
							double dparam_out = dparam_list[dpnum]->get_derived_param(this);
							cout << dparam_out << endl;
						}
						else Complain("unrecognized argument to 'fit dparams'");
					}
				}
				else if (words[1]=="output_dir")
				{
					if (nwords == 2) {
						if (mpi_id==0) cout << "Fit output directory: " << fit_output_dir << endl;
					} else {
						remove_word(0);
						remove_word(0);
						fit_output_dir = "";
						for (int i=0; i < nwords-1; i++) fit_output_dir += words[i] + " ";
						fit_output_dir += words[nwords-1];
						int pos;
						while ((pos = fit_output_dir.find('"')) != string::npos) fit_output_dir.erase(pos,1);
						while ((pos = fit_output_dir.find('\'')) != string::npos) fit_output_dir.erase(pos,1);
						if (fit_output_dir.at(fit_output_dir.length()-1)=='/') fit_output_dir.erase(fit_output_dir.length()-1);
						auto_fit_output_dir = false;
					}
				} else if (words[1]=="use_bestfit") {
					if (nwords > 2) Complain("no arguments allowed for 'use_bestfit' command");
					if (use_bestfit_model()==false) Complain("could not adopt best-fit model");
				} else if (words[1]=="save_bestfit") {
					if (nwords > 3) Complain("no more than one argument allowed for 'save_bestfit' command (filename)");
					if (nwords==3) {
						if (!(ws[2] >> fit_output_filename)) Complain("Invalid fit label");
						if (auto_fit_output_dir) fit_output_dir = "chains_" + fit_output_filename;
					}
					if (mpi_id==0) output_bestfit_model();
				} else if (words[1]=="load_bestfit") {
					if (nwords <= 3) {
						string scriptfile_str;
						if (nwords==3) scriptfile_str = "chains_" + words[2] + "/" + words[2] + "_bf.in";
						else {
							if (auto_fit_output_dir) fit_output_dir = "chains_" + fit_output_filename;
							scriptfile_str = fit_output_dir + "/" + fit_output_filename + "_bf.in";
						}
						ifstream testbf(scriptfile_str.c_str());
						if (!testbf.is_open()) Complain("Error: best-fit lens model file '" << scriptfile_str << "' could not be opened");
						string checklimits;
						testbf >> checklimits;
						if ((checklimits=="#limits") and ((fitmethod==SIMPLEX) or (fitmethod==POWELL))) {
							if (nwords==3) scriptfile_str = "chains_" + words[2] + "/" + words[2] + "_bf_nolimits.in";
							else scriptfile_str = fit_output_dir + "/" + fit_output_filename + "_bf_nolimits.in";
						} else if ((checklimits=="#nolimits") and ((fitmethod != SIMPLEX) and (fitmethod != POWELL))) {
							Complain("The best-fit model did not have parameter limits defined. Switch to simplex or powell and try again");
						}
						testbf.close();
						// the following lines are redundant from the "read" command. Should be put in a separate function to reduce redundancies
						if (infile->is_open()) {
							if (n_infiles==10) Complain("cannot open more than 10 files at once");
							infile++;
						}
						infile->open(scriptfile_str.c_str());
						if (infile->is_open()) {
							if ((n_infiles > 0) and (!read_from_file)) paused_while_reading_file = true;
							read_from_file = true;
							n_infiles++;
						}
						else {
							if (n_infiles > 0) infile--;
							Complain("Error: best-fit lens model file '" << scriptfile_str << "' could not be opened");
						}
						getline((*infile),line); // skip the first comment line
						clear_lenses();
						clear_source_objects();
						update_parameter_list();
						// Clear any existing lens models so the new one can be loaded in
					} else Complain("at most one argument allowed for 'load_bestfit' (fit_label)");
				} else if (words[1]=="add_chain_dparams") {
					if (add_dparams_to_chain()==false) Complain("could not process chain data");
				}
				else Complain("unknown fit command");
			}
		}
		else if (words[0]=="imgdata")
		{
			if (nwords == 1) {
				if (n_sourcepts_fit==0) Complain("no image data has been loaded");
				print_image_data(true); // The boolean argument should be removed (it says to print errors...should ALWAYS print errors!)
				// print the image data that is being used
			} else if (nwords >= 2) {
				if (words[1]=="add") {
					if (nwords < 4) Complain("At least two arguments are required for 'imgdata add' (x,y coordinates of source pt.)");
					// later, add option to include flux of source or measurement errors as extra arguments
					lensvector src;
					if (!(ws[2] >> src[0])) Complain("invalid x-coordinate of source point");
					if (!(ws[3] >> src[1])) Complain("invalid y-coordinate of source point");
					add_simulated_image_data(src);
					update_parameter_list();
				} else if (words[1]=="read") {
					if (nwords != 3) Complain("One argument required for 'imgdata read' (filename)");
					if (load_image_data(words[2])==false) Complain("unable to load image data");
					update_parameter_list();
				} else if (words[1]=="write") {
					if (nwords != 3) Complain("One argument required for 'imgdata write' (filename)");
					write_image_data(words[2]);
				} else if (words[1]=="clear") {
					if (nwords==2) {
						clear_image_data();
					} else if (nwords==3) {
						int imgset_number, min_imgnumber, max_imgnumber, pos;
						if ((pos = words[2].find("-")) != string::npos) {
							string imgminstring, imgmaxstring;
							imgminstring = words[2].substr(0,pos);
							imgmaxstring = words[2].substr(pos+1);
							stringstream imgmaxstream, imgminstream;
							imgminstream << imgminstring;
							imgmaxstream << imgmaxstring;
							if (!(imgminstream >> min_imgnumber)) Complain("invalid min image dataset number");
							if (!(imgmaxstream >> max_imgnumber)) Complain("invalid max image dataset number");
							if (max_imgnumber >= n_sourcepts_fit) Complain("specified max image dataset number exceeds number of data sets in list");
							if ((min_imgnumber > max_imgnumber) or (min_imgnumber < 0)) Complain("specified min image dataset number cannot exceed max image dataset number");
							for (int i=max_imgnumber; i >= min_imgnumber; i--) remove_image_data(i);
						} else {
							if (!(ws[2] >> imgset_number)) Complain("invalid image dataset number");
							remove_image_data(imgset_number);
						}
					} else Complain("'imgdata clear' command requires either one or zero arguments");
				} else if (words[1]=="plot") {
					bool show_sbmap = false;
					bool show_all = false;
					int dataset;
					if (words[nwords-1]=="sbmap") {
						show_sbmap = true;
						stringstream* new_ws = new stringstream[nwords-1];
						words.erase(words.begin()+nwords-1);
						for (int i=0; i < nwords-1; i++) {
							new_ws[i] << words[i];
						}
						delete[] ws;
						ws = new_ws;
						nwords--;
					}
					if (nwords==2) show_all = true;
					else if (nwords==3) {
						if (!(ws[2] >> dataset)) Complain("invalid image dataset");
						if (dataset >= n_sourcepts_fit) Complain("specified image dataset has not been loaded");
					} else Complain("invalid number of arguments to 'imgdata plot'");
					ofstream imgout;
					open_output_file(imgout,"imgdat.dat");
					if (show_all) {
						for (int i=0; i < n_sourcepts_fit; i++) {
							imgout << "\"Dataset " << i << " (z_{s}=" << source_redshifts[i] << ")\"" << endl;
							image_data[i].write_to_file(imgout);
							imgout << endl << endl;
						}
					} else {
						imgout << "\"Dataset " << dataset << " (z_{s}=" << source_redshifts[dataset] << ")\"" << endl;
						image_data[dataset].write_to_file(imgout);
						imgout << endl << endl;
					}
					if (!show_sbmap)
						run_plotter("imgdat");
					else {
						if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
						image_pixel_data->plot_surface_brightness("img_pixel");
						stringstream xminstream, xmaxstream, yminstream, ymaxstream;
						string xminstr, xmaxstr, yminstr, ymaxstr;
						xminstream << image_pixel_data->xvals[0]; xminstream >> xminstr;
						yminstream << image_pixel_data->yvals[0]; yminstream >> yminstr;
						xmaxstream << image_pixel_data->xvals[image_pixel_data->npixels_x]; xmaxstream >> xmaxstr;
						ymaxstream << image_pixel_data->yvals[image_pixel_data->npixels_y]; ymaxstream >> ymaxstr;
						string range = "[" + xminstr + ":" + xmaxstr + "][" + yminstr + ":" + ymaxstr + "]";
						run_plotter_range("imgpixel_imgdat",range);
					}
				} else if (words[1]=="add_from_centroid") {
					if (nwords != 6) Complain("four arguments are required for 'imgdata add_from_centroid' (xmin,xmax,ymin,ymax)");
					double xmin,xmax,ymin,ymax;
					if (!(ws[2] >> xmin)) Complain("invalid minimum x-coordinate");
					if (!(ws[3] >> xmax)) Complain("invalid maximum x-coordinate");
					if (!(ws[4] >> ymin)) Complain("invalid minimum y-coordinate");
					if (!(ws[5] >> ymax)) Complain("invalid maximum y-coordinate");
					if (image_pixel_data==NULL) Complain("cannot add centroid image; image pixel data has not been loaded");
					if (image_data==NULL) {
						if (n_sourcepts_fit != 0) die("number of source points should be zero if image_data has not been allocated");
						n_sourcepts_fit = 1;
						sourcepts_fit = new lensvector[1];
						vary_sourcepts_x = new bool[1];
						vary_sourcepts_y = new bool[1];
						vary_sourcepts_x[0] = true;
						vary_sourcepts_x[1] = true;
						image_data = new ImageData[1];
						update_parameter_list();
					}
					image_pixel_data->add_point_image_from_centroid(image_data+n_sourcepts_fit-1,xmin,xmax,ymin,ymax,sb_threshold,data_pixel_noise); // this assumes centroids are added to last dataset
				} else if (words[1]=="use_in_chisq") {
					if ((nwords < 4) or (nwords > 5)) Complain("Two or three arguments are required for 'imgdata use_in_chisq' (imageset, image_number, on/off)");
					if (image_data == NULL) Complain("no image data has been loaded");
					int n_imgset, n_img;
					bool use_in_chisq;
					if (!(ws[2] >> n_imgset)) Complain("invalid image set number");
					if (!(ws[3] >> n_img)) Complain("invalid image number");
					if (n_imgset >= n_sourcepts_fit) Complain("image set number has not been loaded");
					if (nwords == 5) {
						if (!(ws[4] >> setword)) Complain("invalid argument to 'use_in_chisq' command; must specify 'on' or 'off'");
						set_switch(use_in_chisq,setword);
						if (image_data[n_imgset].set_use_in_chisq(n_img,use_in_chisq) == false) Complain("specified image number does not exist");
					} else {
						if (n_img >= image_data[n_imgset].n_images) Complain("specified image number does not exist");
						if (mpi_id==0) cout << "Include image (" << n_imgset << "," << n_img << ") in chisq: " << display_switch(image_data[n_imgset].use_in_chisq[n_img]) << endl;
					}
				} else Complain("invalid argument to command 'imgdata'");
			}
		}
		else if (words[0]=="wldata")
		{
			if (nwords == 1) {
				if (weak_lensing_data.n_sources==0) Complain("no weak lensing data has been loaded");
				if (mpi_id==0) weak_lensing_data.print_list(use_scientific_notation);
			} else {
				if (words[1]=="add") {
					double zsrc = source_redshift;
					for (int i=2; i < nwords; i++) {
						int pos;
						if ((pos = words[i].find("z=")) != string::npos) {
							string znumstring = words[i].substr(pos+2);
							stringstream znumstr;
							znumstr << znumstring;
							if (!(znumstr >> zsrc)) Complain("incorrect format for source redshift");
							if (zsrc < 0) Complain("source redshift cannot be negative");
							remove_word(i);
							i = nwords; // breaks out of this loop, without breaking from outer loop
						}
					}	
					if (nwords != 4) Complain("Two arguments are required for 'wldata add' (x,y coordinates of source pt., plus optional 'z=' arg)");
					lensvector src;
					if (!(ws[2] >> src[0])) Complain("invalid x-coordinate of source point");
					if (!(ws[3] >> src[1])) Complain("invalid y-coordinate of source point");
					stringstream idstr;
					string id_string;
					idstr << weak_lensing_data.n_sources;
					idstr >> id_string;
					add_simulated_weak_lensing_data(id_string,src,zsrc);
				}
				else if (words[1]=="read") {
					if (nwords != 3) Complain("One argument required for 'wldata read' (filename)");
					if (load_weak_lensing_data(words[2])==false) Complain("unable to load weak lensing data");
				} else if (words[1]=="write") {
					if (nwords != 3) Complain("One argument required for 'wldata write' (filename)");
					weak_lensing_data.write_to_file(words[2]);
				} else if (words[1]=="clear") {
					weak_lensing_data.clear();
				} else Complain("invalid argument to command 'wldata'");
			}
		}
		else if (words[0]=="defspline")
		{
			if (nwords == 1) {
				double xmax, ymax;
				int nsteps;
				if (get_deflection_spline_info(xmax, ymax, nsteps)==false)
					cout << "No deflection field spline has been created" << endl;
				else {
					if (mpi_id==0) {
						if ((xmax < 1e3) and (ymax < 1e3)) cout << resetiosflags(ios::scientific);
						cout << "Deflection splined with " << nsteps << " steps on a "
							"(" << -xmax << "," << xmax << ") x (" << -ymax << "," << ymax << ") grid" << endl;
						if (use_scientific_notation) cout << setiosflags(ios::scientific);
					}
				}
			}
			else if (nwords == 2) {
				if (!islens()) Complain("no lens models have been specified");
				if (words[1]=="off") {
					if (unspline_deflection()==false) Complain("deflection field spline has not been created");
				} else {
					int splinesteps;
					if (!(ws[1] >> splinesteps)) Complain("invalid number of spline steps");
					if ((splinesteps/2)==(splinesteps/2.0))
						splinesteps++;   // Enforce odd number of steps (to avoid the origin)
					create_deflection_spline(splinesteps);
					if (mpi_id==0) cout << "Created " << splinesteps << "x" << splinesteps << " bicubic spline" << endl;
				}
			}
			else if (nwords == 4) {
				if (!islens()) Complain("no lens models have been specified");
				int splinesteps;
				double xmax, ymax;
				if (!(ws[1] >> splinesteps)) Complain("invalid number of spline steps");
				if (!(ws[2] >> xmax)) Complain("invalid xmax value");
				if (!(ws[3] >> ymax)) Complain("invalid ymax value");
				if ((splinesteps/2)==(splinesteps/2.0))
					splinesteps++;   // Enforce odd number of steps (to avoid the origin)
				spline_deflection(xmax,ymax,splinesteps);
				if (mpi_id==0) cout << "Created " << splinesteps << "x" << splinesteps << " bicubic spline" << endl;
			} else Complain("must specify number of steps N and (xmax,ymax) (defaults to current gridsize)");
		}
		else if (words[0]=="auto_defspline")  // obsolete--should probably remove
		{
			if (!use_cc_spline) Complain("auto_defspline is only supported in ccspline mode");
			if (nwords >= 2) {
				if (!islens()) Complain("no lens models have been specified");
				int splinesteps;
				if (!(ws[1] >> splinesteps)) Complain("invalid number of spline steps");
				if ((splinesteps/2)==(splinesteps/2.0))
					splinesteps++;   // Enforce odd number of steps (to avoid the origin)
				if (autospline_deflection(splinesteps)==true) {
					if (mpi_id==0) cout << "Created " << splinesteps << "x" << splinesteps << " bicubic spline" << endl;
				}
			} else Complain("must specify number of points N (will create a NxN bicubic spline)");
		}
		else if (words[0]=="plotcrit")
		{
			if (!islens()) Complain("must specify lens model first");
			string range1, range2;
			extract_word_starts_with('[',2,2,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
			extract_word_starts_with('[',3,3,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
			if ((!plot_srcplane) and (range2.empty())) { range2 = range1; range1 = ""; }
			if (nwords == 3) {
				if (terminal == TEXT) Complain("only one filename is required for text plotting of critical curves");
				if (plotcrit("crit.dat")==true) {
					run_plotter("crit",words[1],range1);
					if (plot_srcplane) run_plotter("caust",words[2],range2);
				} else Complain("No critical curves found");
			} else if (nwords == 2) {
				if (terminal == TEXT) {
					if (plotcrit(words[1].c_str())==false) Complain("No critical curves found");
				}
				else Complain("two filenames must be specified for plotting of critical curves in postscript/PDF mode");
			} else if (nwords == 1) {
				if (plotcrit("crit.dat")==true) {
					run_plotter("crit");
					if (plot_srcplane) run_plotter("caust");
				} else Complain("No critical curves found");
			} else Complain("must specify two filenames, one for each critical curve/caustic");
		}
		else if (words[0]=="plotgrid")
		{
			string range;
			extract_word_starts_with('[',2,2,range); // allow for ranges to be specified (if it's not, then ranges are set to "")
			if (nwords == 2) {
				if (terminal == TEXT) {
					if (plot_recursive_grid(words[1].c_str())==false)
						Complain("could not generate recursive grid");
				} else {
					if (plot_recursive_grid("xgrid.dat")==false)
						Complain("could not generate recursive grid");
					run_plotter("grid",words[1],range);
				}
			} else if (nwords == 1) {
				if (plot_recursive_grid("xgrid.dat")==false)
					Complain("could not generate recursive grid");
				run_plotter("grid");
			} else Complain("invalid number of arguments; must specify one filename for plotting");
		}
		else if (words[0]=="mkgrid")
		{
			if (nwords != 1) {
				Complain("no arguments are allowed for 'mkgrid' command");
			} else {
				if (create_grid(verbal_mode,reference_zfactors,default_zsrc_beta_factors)==false)
					Complain("could not generate recursive grid");
			}
		}
		else if (words[0]=="printcs")
		{
			bool output_to_file;
			string cs_filename;
			double area;
			ofstream cs_out;
			if (!islens()) Complain("must specify lens model first");
			if (nwords == 2) {
				if (!(ws[1] >> cs_filename)) Complain("invalid filename");
				output_to_file = true;
				cs_out.open(cs_filename.c_str(), ofstream::out);
			} else output_to_file = false;
			if (total_cross_section(area)==false) Complain("Could not determine total cross section");
			if (output_to_file) {
				cs_out << "#\n" "source plane: area and number of sources\n" << area << " unknown\n";
			} else {
				if (mpi_id==0) cout << "total cross section: " << area << endl;
			}
		}
		else if (words[0]=="cc_reset")
		{
			if (use_cc_spline) delete_ccspline();
			else Complain("cc_reset can only be used when ccspline is on");
		}
		else if (words[0]=="plotkappa")
		{
			if (terminal != TEXT) Complain("only text plotting supported for plotkappa");
			if (!islens()) Complain("must specify lens model first");
			int lens_number = -1;
			if (words[nwords-1].find("lens=")==0) {
				string lstr = words[nwords-1].substr(5);
				stringstream lstream;
				lstream << lstr;
				if (!(lstream >> lens_number)) Complain("invalid lens number");
				stringstream* new_ws = new stringstream[nwords-1];
				words.erase(words.begin()+nwords-1);
				for (int i=0; i < nwords-1; i++) {
					new_ws[i] << words[i];
				}
				delete[] ws;
				ws = new_ws;
				nwords--;
			}
			if ((nwords == 5) or (nwords == 6)) {
				double rmin, rmax;
				int steps;
				ws[1] >> rmin;
				ws[2] >> rmax;
				ws[3] >> steps;
				if (rmin > rmax) Complain("rmin must be smaller than rmax for plotkappa");
				if (lens_number==-1) {
					if (nwords==5) plot_total_kappa(rmin, rmax, steps, words[4].c_str());
					else plot_total_kappa(rmin, rmax, steps, words[4].c_str(), words[5].c_str());
				} else {
					if (nwords==5) plot_kappa_profile(lens_number, rmin, rmax, steps, words[4].c_str());
					else plot_kappa_profile(lens_number, rmin, rmax, steps, words[4].c_str(), words[5].c_str());
				}
			} else
			  Complain("plotkappa requires 5 parameters (rmin, rmax, steps, kappa_outname, kderiv_outname)");
		}
		else if (words[0]=="plotmass")
		{
			if (terminal != TEXT) Complain("only text plotting supported for plotmass");
			if (!islens()) Complain("must specify lens model first");
			if (nwords == 5) {
				double rmin, rmax;
				int steps;
				ws[1] >> rmin;
				ws[2] >> rmax;
				ws[3] >> steps;
				plot_mass_profile(rmin, rmax, steps, words[4].c_str());
			} else
			  Complain("plotmass requires 4 parameters (rmin, rmax, steps, mass_outname)");
		}
		else if (words[0]=="findimg")
		{
			if (!islens()) Complain("must specify lens model first");
			if (nwords==3) {
				double xsource_in, ysource_in;
				if (!(ws[1] >> xsource_in)) Complain("invalid source x-position");
				if (!(ws[2] >> ysource_in)) Complain("invalid source y-position");
				output_images_single_source(xsource_in, ysource_in, verbal_mode);
			} else Complain("must specify two arguments that give source position (e.g. 'findimg 3.0 1.2')");
		}
		else if (words[0]=="plotimg")
		{
			if (!islens()) Complain("must specify lens model first");
			string range1, range2;
			extract_word_starts_with('[',4,4,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
			extract_word_starts_with('[',5,5,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
			if ((!plot_srcplane) and (range2.empty())) { range2 = range1; range1 = ""; }
			bool show_grid = false;
			if (words[nwords-1]=="grid") {
				show_grid = true;
				stringstream* new_ws = new stringstream[nwords-1];
				for (int i=0; i < nwords-1; i++) {
					new_ws[i] << words[i];
				}
				words.erase(words.begin()+nwords-1);
				delete[] ws;
				ws = new_ws;
				nwords--;
			}
			if (nwords > 5) Complain("max 4 arguments allowed for plotimg: <source_x> <source_y> [imagefile] [sourcefile]");
			if (nwords >= 3) {
				if (show_grid) {
					if (plot_recursive_grid("xgrid.dat")==false)
						Complain("could not generate recursive grid");
				}
				double xsource_in, ysource_in;
				if (!(ws[1] >> xsource_in)) Complain("invalid source x-position");
				if (!(ws[2] >> ysource_in)) Complain("invalid source y-position");
				if ((show_cc) and (plotcrit("crit.dat")==false)) warn("could not plot critical curves");
				if (plot_images_single_source(xsource_in, ysource_in, verbal_mode)==true) {
					if (nwords==5) {
						if (show_cc) {
							if (show_grid) run_plotter("image_grid",words[4],range1);
							else run_plotter("image",words[4],range1);
						} else {
							if (show_grid) run_plotter("image_nocc_grid",words[4],range1);
							else run_plotter("image_nocc",words[4],range1);
						}
						if (plot_srcplane) run_plotter("source",words[3],range2);
					} else if (nwords==4) {
						if (show_cc) {
							if (show_grid) run_plotter("image_grid",words[3],range1);
							else run_plotter("image",words[3],range1);
						} else {
							if (show_grid) run_plotter("image_nocc_grid",words[3],range1);
							else run_plotter("image_nocc",words[3],range1);
						}
					} else {
						// only graphical plotting allowed if filenames not specified
						if (show_cc) {
							if (show_grid) run_plotter("image_grid");
							else run_plotter("image");
						}
						else {
							if (show_grid) run_plotter("image_nocc_grid");
							else run_plotter("image_nocc");
						}
						if (plot_srcplane) run_plotter("source");
					}
				}
			} else Complain("must specify source position (e.g. 'plotimg 3.0 1.2')");
		}
		else if (words[0]=="plotlogkappa")
		{
			if (!islens()) Complain("must specify lens model first");
			if (nwords < 3) {
				if ((nwords==2) and (terminal==TEXT)) {
					plot_logkappa_map(n_image_pixels_x,n_image_pixels_y,words[1]);
				} else {
					plot_logkappa_map(n_image_pixels_x,n_image_pixels_y,"lensmap");
					if (nwords==2) {
						run_plotter("lensmap_kappalog",words[1],"");
					} else {
						run_plotter("lensmap_kappalog");
					}
				}
			} else Complain("only up to one argument is allowed for 'plotlogkappa' (filename)");
		}
		else if (words[0]=="plotlogpot")
		{
			if (!islens()) Complain("must specify lens model first");
			if (nwords < 3) {
				if ((nwords==2) and (terminal==TEXT)) {
					plot_logpot_map(n_image_pixels_x,n_image_pixels_y,words[1]);
				} else {
					plot_logpot_map(n_image_pixels_x,n_image_pixels_y,"lensmap");
					if (nwords==2) {
						run_plotter("lensmap_potlog",words[1],"");
					} else {
						run_plotter("lensmap_potlog");
					}
				}
			} else Complain("only up to one argument is allowed for 'plotlogpot' (filename)");
		}
		else if (words[0]=="plotlogmag")
		{
			if (nwords < 3) {
				if (!islens()) Complain("must specify lens model first");
				if ((!show_cc) or (plotcrit("crit.dat")==true)) {
					if ((nwords==2) and (terminal==TEXT)) {
						plot_logmag_map(n_image_pixels_x,n_image_pixels_y,words[1]);
					} else {
						plot_logmag_map(n_image_pixels_x,n_image_pixels_y,"lensmap");
						if (nwords==2) {
							if (show_cc) {
								run_plotter("lensmap_maglog",words[1],"");
							} else {
								run_plotter("lensmap_maglog_nocc",words[1],"");
							}
						} else {
							if (show_cc) {
								run_plotter("lensmap_maglog");
							} else {
								run_plotter("lensmap_maglog_nocc");
							}
						}
					}
				} else Complain("could not find critical curves");
			} else Complain("only up to one argument is allowed for 'plotlogmag' (filename)");
		}
		else if (words[0]=="einstein")
		{
			if (!islens()) Complain("must specify lens model first");
			if (nwords==1) {
				double re, re_kpc, arcsec_to_kpc, sigma_cr_kpc, m_ein;
				re = einstein_radius_of_primary_lens(reference_zfactors[lens_redshift_idx[primary_lens_number]]);
				arcsec_to_kpc = angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
				re_kpc = re*arcsec_to_kpc;
				sigma_cr_kpc = sigma_crit_kpc(lens_redshift, source_redshift);
				m_ein = sigma_cr_kpc*M_PI*SQR(re_kpc);
				cout << "Einstein radius of primary lens (+ co-centered lenses): r_E = " << re << " arcsec, " << re_kpc << " kpc\n";
				cout << "Mass within Einstein radius: " << m_ein << " solar masses\n";
			} else if (nwords==2) {
				if (mpi_id==0) {
					int lens_number;
					if (!(ws[1] >> lens_number)) Complain("invalid lens number");
					double re_major_axis, re_average;
					if (get_einstein_radius(lens_number,re_major_axis,re_average)==false) Complain("could not calculate Einstein radius");
					double re_kpc, re_major_kpc, arcsec_to_kpc, sigma_cr_kpc, m_ein;
					arcsec_to_kpc = angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
					re_kpc = re_average*arcsec_to_kpc;
					re_major_kpc = re_major_axis*arcsec_to_kpc;
					sigma_cr_kpc = sigma_crit_kpc(lens_redshift, source_redshift);
					m_ein = sigma_cr_kpc*M_PI*SQR(re_kpc);
					if (lens_list[lens_number]->isspherical()) {
						cout << "Einstein radius: r_E = " << re_average << " arcsec, " << re_kpc << " kpc\n";
					} else if (lens_list[lens_number]->ellipticity_mode==-1) {
						// Not an elliptical lens, so major axis Einstein radius has no meaning here
						cout << "Einstein radius averaged over angles: r_{E,avg} = " << re_average << " arcsec, " << re_kpc << " kpc\n";
					} else {
						cout << "Average Einstein radius (r_{E,maj}*sqrt(q)): r_{E,avg} = " << re_average << " arcsec, " << re_kpc << " kpc\n";
						cout << "Einstein radius along major axis: r_{E,maj} = " << re_major_axis << " arcsec, " << re_major_kpc << " kpc\n";
					}
					cout << "Mass within average Einstein radius: " << m_ein << " solar masses\n";

				}
			} else Complain("only one argument allowed for 'einstein' command (lens number)");
		}
		else if (words[0]=="sigma_cr")
		{
			if (nwords > 1) Complain("no arguments accepted for command 'sigma_cr'");
			double sigma_cr_arcsec, sigma_cr_kpc;
			sigma_cr_arcsec = sigma_crit_arcsec(lens_redshift, source_redshift);
			sigma_cr_kpc = sigma_crit_kpc(lens_redshift, source_redshift);
			cout << "sigma_crit = " << sigma_cr_arcsec << " solar masses/arcsec^2\n";
			cout << "           = " << sigma_cr_kpc << " solar masses/kpc^2\n";
		}
		else if (words[0]=="mksrctab")
		{
			if (nwords >= 7) {
				double xmin, xmax, ymin, ymax;
				int xsteps, ysteps;
				if (!(ws[1] >> xmin)) Complain("invalid xmin parameter");
				if (!(ws[2] >> xmax)) Complain("invalid xmax parameter");
				if (!(ws[3] >> xsteps)) Complain("invalid xsteps parameter; must be integral number of steps");
				if (!(ws[4] >> ymin)) Complain("invalid ymin parameter");
				if (!(ws[5] >> ymax)) Complain("invalid ymax parameter");
				if (!(ws[6] >> ysteps)) Complain("invalid ysteps parameter; must be integral number of steps");
				string outfile;
				if (nwords==8) outfile = words[7];
				else if (nwords > 8) { Complain("too many parameters in mksrctab"); }
				else outfile = "sourcexy.in";
				make_source_rectangle(xmin,xmax,xsteps,ymin,ymax,ysteps,outfile);
			} else Complain("mksrctab requires at least 6 parameters: xmin, xmax, xpoints, ymin, ymax, ypoints");
		}
		else if (words[0]=="mksrcgal")
		{
			if (nwords >= 8) {
				double xcenter, ycenter, major_axis, q, angle;
				int n_ellipses, points_per_ellipse;
				if (!(ws[1] >> xcenter)) Complain("invalid xcenter parameter");
				if (!(ws[2] >> ycenter)) Complain("invalid ycenter parameter");
				if (!(ws[3] >> major_axis)) Complain("invalid major_axis parameter; must be integral number of steps");
				if (!(ws[4] >> q)) Complain("invalid q parameter");
				if (!(ws[5] >> angle)) Complain("invalid theta parameter");
				if (!(ws[6] >> n_ellipses)) Complain("invalid n_ellipses parameter; must be integral number of steps");
				if (!(ws[7] >> points_per_ellipse)) Complain("invalid points_per_ellipse parameter; must be integral number of steps");
				string outfile;
				if (nwords==9) outfile = words[8];
				else if (nwords > 9) { Complain("too many parameters in mksrcgal"); }
				else outfile = "sourcexy.in";
				make_source_ellipse(xcenter,ycenter,major_axis,q,angle,n_ellipses,points_per_ellipse,outfile);
			} else Complain("mksrcgal requires at least 7 parameters: xcenter, ycenter, major_axis, q, theta, n_ellipses points_per_ellipse");
		}
		else if (words[0]=="mkrandsrc")
		{
			if (!islens()) Complain("must specify lens model first");
			if (nwords >= 2) {
				int nsources;
				if (!(ws[1] >> nsources)) Complain("invalid number of sources");
				string outfile;
				if (nwords==3) outfile = words[2];
				else if (nwords > 3) { Complain("too many parameters in mkrandsrc"); }
				else outfile = "sourcexy.in";
				make_random_sources(nsources, outfile.c_str());
			} else Complain("mkrandsrc requires at least 1 parameter (number of sources)");
		}
		else if (words[0]=="findimgs")
		{
			if (!islens()) Complain("must specify lens model first");
			if (nwords == 1)
				plot_images("sourcexy.in", "images.dat", verbal_mode);	// default source file
			else if (nwords == 2)
				plot_images(words[1].c_str(), "images.dat", verbal_mode);
			else if (nwords == 3)
				plot_images(words[1].c_str(), words[2].c_str(), verbal_mode);
			else Complain("invalid number of arguments to command 'findimgs'");
		}
		else if (words[0]=="plotimgs")
		{
			if (!islens()) Complain("must specify lens model first");
			string range1, range2;
			extract_word_starts_with('[',1,3,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
			extract_word_starts_with('[',1,4,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
			if ((!plot_srcplane) and (range2.empty())) { range2 = range1; range1 = ""; }
			if (nwords == 1) {
				if (plot_images("sourcexy.in", "imgs.dat", verbal_mode)==true) {	// default source file
					if (plot_srcplane) run_plotter_range("sources",range1);
					if (show_cc) run_plotter_range("images",range2);
					else run_plotter_range("images_nocc",range2);
				}
				else Complain("could not create grid to plot images");
			} else if ((nwords==2) and (words[1]=="sbmap")) {
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				image_pixel_data->plot_surface_brightness("img_pixel");
				if (plot_images("sourcexy.in", "imgs.dat", verbal_mode)==true) {	// default source file
					if (show_cc) run_plotter_range("imgpixel_imgpts_plural",range1);
					else run_plotter_range("imgpixel_imgpts_plural_nocc",range1);
				} else Complain("could not create grid to plot images");
			} else if (nwords == 2) {
				if (plot_images(words[1].c_str(), "imgs.dat", verbal_mode)==true) {
					if (plot_srcplane) run_plotter_range("sources",range1);
					if (show_cc) run_plotter_range("images",range2);
					else run_plotter_range("images_nocc",range2);
				} else Complain("could not create grid to plot images");
			} else if (nwords == 3) {
				if (terminal == TEXT) {
					plot_images(words[1].c_str(), words[2].c_str(), verbal_mode);
				} else if (plot_images("sourcexy.in", "imgs.dat", verbal_mode)==true) {
					if (plot_srcplane) run_plotter("sources",words[1],range1);
					if (show_cc) run_plotter("images",words[2],range2);
					else run_plotter("images_nocc",words[2],range2);
				}
			} else if (nwords == 4) {
				if (terminal == TEXT) Complain("only one filename allowed in text mode (image file)");
				if (plot_images(words[1].c_str(), "imgs.dat", verbal_mode)==true) {
					if (plot_srcplane) run_plotter("sources",words[2],range1);
					if (show_cc) run_plotter("images",words[3],range2);
					else run_plotter("images_nocc",words[3],range2);
				}
			} else Complain("invalid number of arguments to command 'plotimgs'");
		}
		else if (words[0]=="replotimgs")
		{
			string range1, range2;
			extract_word_starts_with('[',1,3,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
			extract_word_starts_with('[',1,4,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
			if ((!plot_srcplane) and (range2.empty())) { range2 = range1; range1 = ""; }
			if (nwords == 1) {
				if (plot_srcplane) run_plotter_range("sources",range1);
				if (show_cc) run_plotter_range("images",range2);
				else run_plotter_range("images_nocc",range2);
			} else if (nwords == 2) {
				if (terminal == TEXT) Complain("for replotting, filename not allowed in text mode");
				Complain("for replotting, must specify both source and image output filenames");
			} else if (nwords == 3) {
				if (terminal == TEXT) Complain("for replotting, filename not allowed in text mode");
				if (plot_srcplane) run_plotter("sources",words[1],range1);
				if (show_cc) run_plotter("images",words[2],range2);
				else run_plotter("images_nocc",words[2],range2);
			} else Complain("invalid number of arguments to command 'replotimgs'");
		}
		else if (words[0]=="srcgrid")
		{
			if (nwords==1) {
				if (mpi_id==0) {
					if (((sourcegrid_xmax-sourcegrid_xmin) < 1e3) and ((sourcegrid_ymax-sourcegrid_ymin) < 1e3)) cout << resetiosflags(ios::scientific);
					cout << "Source grid = (" << sourcegrid_xmin << "," << sourcegrid_xmax << ") x (" << sourcegrid_ymin << "," << sourcegrid_ymax << ")";
					if (auto_sourcegrid) cout << " (auto_srcgrid on)";
					cout << endl;
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
				}
			} else if (nwords == 3) {
				double xlh, ylh;
				if (!(ws[1] >> xlh)) Complain("invalid srcgrid x-coordinate");
				if (!(ws[2] >> ylh)) Complain("invalid srcgrid y-coordinate");
				sourcegrid_xmin = -xlh; sourcegrid_xmax = xlh;
				sourcegrid_ymin = -ylh; sourcegrid_ymax = ylh;
				if ((xlh < 1e3) and (ylh < 1e3)) cout << resetiosflags(ios::scientific);
				cout << "set source grid = (" << -xlh << "," << xlh << ") x (" << -ylh << "," << ylh << ")" << endl;
				if (use_scientific_notation) cout << setiosflags(ios::scientific);
				auto_sourcegrid = false;
			} else if (nwords == 5) {
				double xmin,xmax,ymin,ymax;
				if (!(ws[1] >> xmin)) Complain("invalid srcgrid xmin");
				if (!(ws[2] >> xmax)) Complain("invalid srcgrid ymin");
				if (!(ws[3] >> ymin)) Complain("invalid srcgrid xmax");
				if (!(ws[4] >> ymax)) Complain("invalid srcgrid ymax");
				sourcegrid_xmin = xmin; sourcegrid_xmax = xmax;
				sourcegrid_ymin = ymin; sourcegrid_ymax = ymax;
				auto_sourcegrid = false;
			} else Complain("invalid arguments to 'srcgrid' (type 'help srcgrid' for usage information)");
		}
		else if (words[0]=="srcgrid_limits")
		{
			if (nwords==1) {
				if (mpi_id==0) {
					if (((sourcegrid_limit_xmax-sourcegrid_limit_xmin) < 1e3) and ((sourcegrid_limit_ymax-sourcegrid_limit_ymin) < 1e3)) cout << resetiosflags(ios::scientific);
					cout << "source grid limits = (" << sourcegrid_limit_xmin << "," << sourcegrid_limit_xmax << ") x (" << sourcegrid_limit_ymin << "," << sourcegrid_limit_ymax << ")" << endl;
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
				}
			} else if (nwords == 5) {
				double xmin,xmax,ymin,ymax;
				if (!(ws[1] >> xmin)) Complain("invalid srcgrid limit xmin");
				if (!(ws[2] >> xmax)) Complain("invalid srcgrid limit ymin");
				if (!(ws[3] >> ymin)) Complain("invalid srcgrid limit xmax");
				if (!(ws[4] >> ymax)) Complain("invalid srcgrid limit ymax");
				sourcegrid_limit_xmin = xmin; sourcegrid_limit_xmax = xmax;
				sourcegrid_limit_ymin = ymin; sourcegrid_limit_ymax = ymax;
			} else Complain("invalid arguments to 'srcgrid_limits' (type 'help srcgrid_limits' for usage information)");
		}
		else if (words[0]=="sbmap")
		{
			if (nwords==1) {
				if (source_pixel_grid == NULL) cout << "No source surface brightness map has been loaded\n";
				else cout << "Source surface brightness map is loaded with pixel dimension (" << source_pixel_grid->u_N << "," << source_pixel_grid->w_N << ")\n";
				if (image_pixel_grid == NULL) cout << "No image surface brightness map has been loaded\n";
				else cout << "Image surface brightness map is loaded with pixel dimension (" << image_pixel_grid->x_N << "," << image_pixel_grid->y_N << ")\n";
			}
			else if (words[1]=="savesrc")
			{
				string filename;
				if (nwords==2) filename = "src_pixel";
				else if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for source surface brightness map");
				} else Complain("too many arguments to 'sbmap savesrc'");
				if (source_pixel_grid==NULL) Complain("no source surface brightness map has been created/loaded");
				source_pixel_grid->store_surface_brightness_grid_data(filename);
			}
			else if (words[1]=="loadsrc")
			{
				string filename;
				if (nwords==2) filename = "src_pixel";
				else if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for source surface brightness map");
				} else Complain("too many arguments to 'sbmap loadsrc'");
				load_source_surface_brightness_grid(filename);
			}
			else if (words[1]=="loadimg")
			{
				string filename;
				if (nwords==2) filename = "img_pixel";
				else if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for image surface brightness map");
				} else Complain("too many arguments to 'sbmap loadimg'");
				if (!load_image_surface_brightness_grid(filename)) Complain("could not load image data");
			}
			else if (words[1]=="loadmask")
			{
				string filename;
				if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for mask pixel map");
				} else Complain("too many arguments to 'sbmap loadmask'");
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				image_pixel_data->load_mask_fits(filename);
			}
			else if (words[1]=="loadpsf")
			{
				string filename;
				if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for PSF matrix");
				} else Complain("too many arguments to 'sbmap loadpsf'");
				load_psf_fits(filename,verbal_mode);
			}
			else if (words[1]=="unloadpsf")
			{
				string filename;
				if (nwords != 2) Complain("no arguments are required for 'sbmap unloadpsf'");
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				if (!use_input_psf_matrix) Complain("no psf has been loaded from FITS file");
				use_input_psf_matrix = false;
			}
			else if (words[1]=="makesrc")
			{
				if (nwords==2) {
					create_source_surface_brightness_grid(verbal_mode);
				} else Complain("no arguments are allowed for 'sbmap makesrc'");
			}
			else if (words[1]=="plotsrcgrid")
			{
				string outfile;
				if (nwords==2) {
					outfile = "srcgrid.dat";
					plot_source_pixel_grid(outfile.c_str());
				} else if (nwords==3) {
					if (!(ws[2] >> outfile)) Complain("invalid output filename for source surface brightness map");
					plot_source_pixel_grid(outfile.c_str());
				} else Complain("too many arguments to 'sbmap plotsrcgrid'");
			}
			else if (words[1]=="plotdata")
			{
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				string range;
				extract_word_starts_with('[',1,nwords-1,range); // allow for ranges to be specified (if it's not, then ranges are set to "")
				if (range.empty()) {
					stringstream xminstream, xmaxstream, yminstream, ymaxstream;
					string xminstr, xmaxstr, yminstr, ymaxstr;
					xminstream << image_pixel_data->xvals[0]; xminstream >> xminstr;
					yminstream << image_pixel_data->yvals[0]; yminstream >> yminstr;
					xmaxstream << image_pixel_data->xvals[image_pixel_data->npixels_x]; xmaxstream >> xmaxstr;
					ymaxstream << image_pixel_data->yvals[image_pixel_data->npixels_y]; ymaxstream >> ymaxstr;
					range = "[" + xminstr + ":" + xmaxstr + "][" + yminstr + ":" + ymaxstr + "]";
				}
				if (nwords == 2) {
					image_pixel_data->plot_surface_brightness("data_pixel");
					run_plotter_range("datapixel",range);
				} else if (nwords == 3) {
					if (terminal==TEXT) {
						image_pixel_data->plot_surface_brightness(words[2]);
					}
					else {
						image_pixel_data->plot_surface_brightness("data_pixel");
						run_plotter("datapixel",words[2],range);
					}
				}
			}
			else if (words[1]=="unset_all_pixels")
			{
				if (nwords > 2) Complain("no arguments allowed for command 'sbmap unset_all_pixels'");
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				image_pixel_data->set_no_required_data_pixels();
			}
			else if (words[1]=="set_all_pixels")
			{
				if (nwords > 2) Complain("no arguments allowed for command 'sbmap unset_all_pixels'");
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				image_pixel_data->set_all_required_data_pixels();
			}
			else if (words[1]=="set_neighbor_pixels")
			{
				if (nwords > 2) Complain("no arguments allowed for command 'sbmap set_neighbor_pixels'");
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				image_pixel_data->set_nearest_neighbor_pixels();
			}
			else if (words[1]=="unset_low_sn_pixels")
			{
				if (nwords != 3) Complain("one argument allowed for command 'sbmap unset_low_sn_pixels' (sb_threshold)");
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				double sbthresh;
				if (!(ws[2] >> sbthresh)) Complain("invalid surface brightness threshold");
				image_pixel_data->unset_low_signal_pixels(sbthresh,false);
			}
			else if (words[1]=="trim_mask_windows")
			{
				if (nwords != 3) Complain("one argument allowed for command 'sbmap trim_mask_windows' (noise threshold)");
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				double noise_threshold;
				if (!(ws[2] >> noise_threshold)) Complain("invalid noise threshold for keeping mask windows");
				image_pixel_data->assign_mask_windows(noise_threshold);
			}
			else if (words[1]=="set_data_annulus")
			{
				double xc, yc, rmin, rmax, thetamin=0, thetamax=360, xstretch=1.0, ystretch=1.0;
				if (nwords >= 6) {
					if (!(ws[2] >> xc)) Complain("invalid annulus center x-coordinate");
					if (!(ws[3] >> yc)) Complain("invalid annulus center y-coordinate");
					if (!(ws[4] >> rmin)) Complain("invalid annulas rmin");
					if (!(ws[5] >> rmax)) Complain("invalid annulas rmax");
					if (nwords >= 8) {
						if (!(ws[6] >> thetamin)) Complain("invalid annulus thetamin");
						if (!(ws[7] >> thetamax)) Complain("invalid annulus thetamax");
						if (nwords == 10) {
							if (!(ws[8] >> xstretch)) Complain("invalid annulus xstretch");
							if (!(ws[9] >> ystretch)) Complain("invalid annulus ystretch");
						} 
					} else if (nwords != 6) Complain("must specify 4 args (xc,yc,rmin,rmax) plus optional thetamin,thetamax, and xstretch,ystretch");
				} else Complain("must specify at least 4 args (xc,yc,rmin,rmax)");
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				image_pixel_data->set_required_data_annulus(xc,yc,rmin,rmax,thetamin,thetamax,xstretch,ystretch);
			}
			else if (words[1]=="unset_data_annulus")
			{
				double xc, yc, rmin, rmax, thetamin=0, thetamax=360, xstretch=1.0, ystretch=1.0;
				if (nwords >= 6) {
					if (!(ws[2] >> xc)) Complain("invalid annulus center x-coordinate");
					if (!(ws[3] >> yc)) Complain("invalid annulus center y-coordinate");
					if (!(ws[4] >> rmin)) Complain("invalid annulas rmin");
					if (!(ws[5] >> rmax)) Complain("invalid annulas rmax");
					if (nwords >= 8) {
						if (!(ws[6] >> thetamin)) Complain("invalid annulus thetamin");
						if (!(ws[7] >> thetamax)) Complain("invalid annulus thetamax");
						if (nwords == 10) {
							if (!(ws[8] >> xstretch)) Complain("invalid annulus xstretch");
							if (!(ws[9] >> ystretch)) Complain("invalid annulus ystretch");
						} 
					} else if (nwords != 6) Complain("must specify 4 args (xc,yc,rmin,rmax) plus optional thetamin,thetamax, and xstretch,ystretch");
				} else Complain("must specify at least 4 args (xc,yc,rmin,rmax)");
				if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
				image_pixel_data->set_required_data_annulus(xc,yc,rmin,rmax,thetamin,thetamax,xstretch,ystretch,true); // the 'true' says to deactivate the pixels, instead of activating them
			}
			else if (words[1]=="set_data_window")
			{
				double xmin, xmax, ymin, ymax;
				if (nwords == 6) {
					if (!(ws[2] >> xmin)) Complain("invalid rectangle xmin");
					if (!(ws[3] >> xmax)) Complain("invalid rectangle xmax");
					if (!(ws[4] >> ymin)) Complain("invalid rectangle ymin");
					if (!(ws[5] >> ymax)) Complain("invalid rectangle ymax");
					if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
					image_pixel_data->set_required_data_pixels(xmin,xmax,ymin,ymax);
				} else Complain("must specify 4 arguments (xmin,xmax,ymin,ymax) for 'sbmap set_data_window'");
			}
			else if (words[1]=="unset_data_window")
			{
				double xmin, xmax, ymin, ymax;
				if (nwords == 6) {
					if (!(ws[2] >> xmin)) Complain("invalid rectangle xmin");
					if (!(ws[3] >> xmax)) Complain("invalid rectangle xmax");
					if (!(ws[4] >> ymin)) Complain("invalid rectangle ymin");
					if (!(ws[5] >> ymax)) Complain("invalid rectangle ymax");
					if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
					image_pixel_data->set_required_data_pixels(xmin,xmax,ymin,ymax,true); // the 'true' says to deactivate the pixels, instead of activating them
				} else Complain("must specify 4 arguments (xmin,xmax,ymin,ymax) for 'sbmap unset_data_window'");
			}
			else if (words[1]=="find_noise")
			{
				double xmin, xmax, ymin, ymax, sig_sb, mean_sb;
				if (nwords == 6) {
					if (!(ws[2] >> xmin)) Complain("invalid rectangle xmin");
					if (!(ws[3] >> xmax)) Complain("invalid rectangle xmax");
					if (!(ws[4] >> ymin)) Complain("invalid rectangle ymin");
					if (!(ws[5] >> ymax)) Complain("invalid rectangle ymax");
					if (image_pixel_data == NULL) Complain("no image pixel data has been loaded");
					image_pixel_data->estimate_pixel_noise(xmin,xmax,ymin,ymax,sig_sb,mean_sb);
					cout << "Mean surface brightness: " << mean_sb << endl;
					cout << "Dispersion of surface brightness: " << sig_sb << endl;
					cout << endl;
				} else Complain("must specify 4 arguments (xmin,xmax,ymin,ymax) for 'sbmap find_noise'");
			}
			else if (words[1]=="plotimg")
			{
				bool replot = false;
				bool plot_residual = false;
				bool plot_fits = false;
				bool omit_source = false;
				int reduce_factor = 1;
				bool offload_to_data = false;
				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-replot") replot = true;
						else if (args[i]=="-residual") plot_residual = true;
						else if (args[i]=="-fits") plot_fits = true;
						else if (args[i]=="-nosrc") omit_source = true;
						else if (args[i]=="-reduce2") reduce_factor = 2;
						else if (args[i]=="-reduce4") reduce_factor = 4;
						else if (args[i]=="-reduce8") reduce_factor = 8;
						else if (args[i]=="-mkdata") offload_to_data = true;
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				if ((replot) and (plot_fits)) Complain("Cannot use 'replot' option when plotting to fits files");
				bool old_plot_srcplane = plot_srcplane;
				if (omit_source) plot_srcplane = false;

				if (reduce_factor > 1) {
					n_image_pixels_x *= reduce_factor;
					n_image_pixels_y *= reduce_factor;
				}
				if (!islens()) Complain("must specify lens model first");
				if ((source_fit_mode==Pixellated_Source) and (source_pixel_grid == NULL)) Complain("No source surface brightness map has been loaded");
				string range1, range2;
				extract_word_starts_with('[',1,nwords-1,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
				extract_word_starts_with('[',1,nwords-1,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
				if ((!plot_srcplane) and (range2.empty())) { range2 = range1; range1 = ""; }
				if ((plot_srcplane) and (range1 == "")) {
					stringstream xminstream, xmaxstream, yminstream, ymaxstream;
					string xminstr, xmaxstr, yminstr, ymaxstr;
					xminstream << source_pixel_grid->srcgrid_xmin; xminstream >> xminstr;
					yminstream << source_pixel_grid->srcgrid_ymin; yminstream >> yminstr;
					xmaxstream << source_pixel_grid->srcgrid_xmax; xmaxstream >> xmaxstr;
					ymaxstream << source_pixel_grid->srcgrid_ymax; ymaxstream >> ymaxstr;
					range1 = "[" + xminstr + ":" + xmaxstr + "][" + yminstr + ":" + ymaxstr + "]";
				}
				bool foundcc = true;
				if ((!show_cc) or (plot_fits) or ((foundcc = plotcrit("crit.dat"))==true)) {
					if (nwords == 2) {
						if (plot_fits) Complain("file name for FITS file must be specified");
						if ((replot) or (plot_lensed_surface_brightness("img_pixel",reduce_factor,plot_fits,plot_residual,offload_to_data)==true)) {
							if (!offload_to_data) {
								if ((!replot) and (source_pixel_grid != NULL)) { if (mpi_id==0) source_pixel_grid->plot_surface_brightness("src_pixel"); }
								if (show_cc) {
									if ((plot_srcplane) and (source_pixel_grid != NULL)) run_plotter_range("srcpixel",range1);
									run_plotter_range("imgpixel",range2);
								} else {
									if ((plot_srcplane) and (source_pixel_grid != NULL)) run_plotter_range("srcpixel_nocc",range1);
									run_plotter_range("imgpixel_nocc",range2);
								}
							}
						}
					} else if (nwords == 3) {
						if (terminal==TEXT) {
							if (!replot) plot_lensed_surface_brightness(words[2],reduce_factor,plot_fits,plot_residual,offload_to_data);
						}
						else if ((replot) or (plot_lensed_surface_brightness("img_pixel",reduce_factor,plot_fits,plot_residual,offload_to_data)==true)) {
							if (show_cc) {
								run_plotter("imgpixel",words[2],range1);
							} else {
								run_plotter("imgpixel_nocc",words[2],range1);
							}
						}
					} else if (nwords == 4) {
						if (terminal==TEXT) {
							if (!replot) {
								plot_lensed_surface_brightness(words[3],reduce_factor,plot_fits,plot_residual,offload_to_data);
								if ((source_pixel_grid != NULL) and (mpi_id==0)) source_pixel_grid->plot_surface_brightness(words[2]);
							}
						}
						else if ((replot) or (plot_lensed_surface_brightness("img_pixel",reduce_factor,plot_fits,plot_residual,offload_to_data)==true)) {
							if ((!replot) and (source_pixel_grid != NULL)) { if (mpi_id==0) source_pixel_grid->plot_surface_brightness("src_pixel"); }
							if (show_cc) {
								run_plotter("imgpixel",words[3],range2);
								if ((plot_srcplane) and (source_pixel_grid != NULL)) run_plotter("srcpixel",words[2],range1);
							} else {
								run_plotter("imgpixel_nocc",words[3],range2);
								if ((plot_srcplane) and (source_pixel_grid != NULL)) run_plotter("srcpixel_nocc",words[2],range1);
							}
						}
					} else Complain("invalid number of arguments to 'sbmap plotimg'");
				} else if (!foundcc) Complain("could not find critical curves");
				if (omit_source) plot_srcplane = old_plot_srcplane;
			}
			else if (words[1]=="plotsrc")
			{
				if (source_pixel_grid == NULL) Complain("No source surface brightness map has been loaded");
				string range1 = "";
				extract_word_starts_with('[',1,nwords-1,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
				if (range1 == "") {
				//if ((range1 == "") and ((!show_cc) or (!islens()))) {
					stringstream xminstream, xmaxstream, yminstream, ymaxstream;
					string xminstr, xmaxstr, yminstr, ymaxstr;
					xminstream << source_pixel_grid->srcgrid_xmin; xminstream >> xminstr;
					yminstream << source_pixel_grid->srcgrid_ymin; yminstream >> yminstr;
					xmaxstream << source_pixel_grid->srcgrid_xmax; xmaxstream >> xmaxstr;
					ymaxstream << source_pixel_grid->srcgrid_ymax; ymaxstream >> ymaxstr;
					range1 = "[" + xminstr + ":" + xmaxstr + "][" + yminstr + ":" + ymaxstr + "]";
				}
				if (nwords == 2) {
					if (mpi_id==0) source_pixel_grid->plot_surface_brightness("src_pixel");
					if ((islens()) and (show_cc) and (plotcrit("crit.dat")==true)) {
						run_plotter_range("srcpixel",range1);
					} else {
						run_plotter_range("srcpixel_nocc",range1);
					}
				} else if (nwords == 3) {
					if (terminal==TEXT) {
						if (mpi_id==0) source_pixel_grid->plot_surface_brightness(words[2]);
					} else {
						if (mpi_id==0) source_pixel_grid->plot_surface_brightness("src_pixel");
					}
				} else Complain("invalid number of arguments to 'sbmap plotsrc'");
			}
			else if (words[1]=="invert")
			{
				if (!islens()) Complain("must specify lens model first");
				invert_surface_brightness_map_from_data(verbal_mode);

				//test_fitmodel_invert(); // use this to make sure the fitmodel chi-square returns the same value as doing the inversion directly (runs chi-square twice just to make sure)
			}
			else if (words[1]=="plot_imgpixels")
			{
				if (!islens()) Complain("must specify lens model first");
				plot_image_pixel_grid();
			}
			else Complain("command not recognized");
		}
		else if (words[0]=="lensinfo")
		{
			if (mpi_id==0) {
				if (nwords != 3) Complain("two arguments are required; must specify coordinates (x,y) to display lensing information");
				if (!islens()) Complain("must specify lens model first");
				double x, y;
				if (!(ws[1] >> x)) Complain("invalid x-coordinate");
				if (!(ws[2] >> y)) Complain("invalid y-coordinate");
				lensvector point, alpha, beta;
				double sheartot, shear_angle;
				point[0] = x; point[1] = y;
				deflection(point,alpha,reference_zfactors,default_zsrc_beta_factors);
				shear(point,sheartot,shear_angle,0,reference_zfactors,default_zsrc_beta_factors);
				beta[0] = point[0] - alpha[0];
				beta[1] = point[1] - alpha[1];
				double kappaval = 
				kappaval = kappa(point,reference_zfactors,default_zsrc_beta_factors);
				cout << "kappa = " << kappaval << endl;
				cout << "deflection = (" << alpha[0] << "," << alpha[1] << ")\n";
				cout << "potential = " << potential(point,reference_zfactors,default_zsrc_beta_factors) << endl;
				cout << "magnification = " << magnification(point,0,reference_zfactors,default_zsrc_beta_factors) << endl;
				cout << "shear = " << sheartot << ", shear_angle=" << shear_angle << endl;
				cout << "reduced_shear1 = " << sheartot*cos(2*shear_angle*M_PI/180.0)/(1-kappaval) << " reduced_shear2 = " << sheartot*sin(2*shear_angle*M_PI/180.0)/(1-kappaval) << endl;
				cout << "sourcept = (" << beta[0] << "," << beta[1] << ")\n";

				if (n_lens_redshifts > 1) {
					lensvector xl;
					for (int i=1; i < n_lens_redshifts; i++) {
						map_to_lens_plane(i,x,y,xl,0,reference_zfactors,default_zsrc_beta_factors);
						cout << "x(z=" << lens_redshifts[i] << "): (" << xl[0] << "," << xl[1] << ")" << endl;
					}
				}
				cout << endl;
				//cout << "shear/kappa = " << sheartot/kappa(point) << endl;
			}
		}
		else if (words[0]=="plotlensinfo")
		{
			if (!islens()) Complain("must specify lens model first");
			string file_root;
			if (nwords == 1) {
 				// if the fit label hasn't been set, probably we're not doing a fit anyway, so pick a more generic name
				if (fit_output_filename == "fit") file_root = "lensmap";
				else file_root = fit_output_filename;
			} else if (nwords == 2) {
				file_root = words[1];
			} else Complain("only one argument (file label) allowed for 'sbmap plotlensinfo'");
			plot_lensinfo_maps(file_root,n_image_pixels_x,n_image_pixels_y);
		}
		else if ((words[0]=="ptsize") or (words[0]=="ps"))
		{
			if (mpi_id==0) {
				if (nwords == 1) {
					if (mpi_id==0) {
						cout << resetiosflags(ios::scientific);
						cout << plot_ptsize << endl;
						if (use_scientific_notation) cout << setiosflags(ios::scientific);
					}
				} else if (nwords == 2) {
					double plotps;
					if (!(ws[1] >> plotps)) Complain("invalid point size");
					plot_ptsize = plotps;
				} else Complain("only one argument allowed for command 'ptsize' (point size)");
			}
		}
		else if ((words[0]=="pttype") or (words[0]=="pt"))
		{
			if (mpi_id==0) {
				if (nwords == 1) {
					if (mpi_id==0) cout << plot_pttype << endl;
				} else if (nwords == 2) {
					int plotpt;
					if (!(ws[1] >> plotpt)) Complain("invalid point type (must be integer value)");
					plot_pttype = plotpt;
				} else Complain("only one argument allowed for command 'pttype' (point type)");
			}
		}
		else if (words[0]=="show_wtime")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Display wall time during likelihood evaluations: " << display_switch(show_wtime) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'show_wtime' command; must specify 'on' or 'off'");
				set_switch(show_wtime,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="verbal_mode")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Verbal mode: " << display_switch(verbal_mode) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'verbal_mode' command; must specify 'on' or 'off'");
				else set_switch(verbal_mode,setword);
			}
			else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="warnings")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Warnings: " << display_switch(warnings) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'warnings' command; must specify 'on' or 'off'");
				else set_switch(warnings,setword);
			}
			else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="imgsrch_warnings")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Warnings for image search: " << display_switch(newton_warnings) << endl;
			} else if (nwords==2) {
				string setword;
				if (!(ws[1] >> setword)) Complain("invalid argument to 'warnings' command");
				else set_switch(newton_warnings,setword);
			}
			else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="sci_notation")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Scientific notation: " << display_switch(use_scientific_notation) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'sci_notation' command; must specify 'on' or 'off'");
				set_switch(use_scientific_notation,setword);
				if (use_scientific_notation==true) setiosflags(ios::scientific);
				else {
					cout << resetiosflags(ios::scientific);
					cout << setprecision(6);
					cout << fixed;
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="major_axis_along_y")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Orient major axis along y-direction: " << display_switch(LensProfile::orient_major_axis_north) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'major_axis_along_y' command; must specify 'on' or 'off'");
				bool orient_north;
				set_switch(orient_north,setword);
				toggle_major_axis_along_y(orient_north);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="shear_components")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use shear components as parameters: " << display_switch(Shear::use_shear_component_params) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'shear_components' command; must specify 'on' or 'off'");
				bool use_comps;
				set_switch(use_comps,setword);
				Shear::use_shear_component_params = use_comps;
				reassign_lensparam_pointers_and_names();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="ellipticity_components")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use ellipticity components instead of (q,theta): " << display_switch(LensProfile::use_ellipticity_components) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'ellipticity_components' command; must specify 'on' or 'off'");
				bool use_comps;
				set_switch(use_comps,setword);
				LensProfile::use_ellipticity_components = use_comps;
				reassign_lensparam_pointers_and_names();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="emode")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Ellipticity mode: " << LensProfile::default_ellipticity_mode << endl;
			} else if (nwords==2) {
				int elmode;
				if (!(ws[1] >> elmode)) Complain("invalid argument to 'emode' command; must specify 0, 1, 2, or 3");
				if (elmode > 3) Complain("ellipticity mode cannot be greater than 3");
				LensProfile::default_ellipticity_mode = elmode;
			} else Complain("invalid number of arguments; must specify 0, 1, 2, or 3");
		}
		else if (words[0]=="pmode")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Default parameter mode: " << default_parameter_mode << endl;
			} else if (nwords==2) {
				int pm;
				if (!(ws[1] >> pm)) Complain("invalid argument to 'pmode' command");
				default_parameter_mode = pm;
			} else Complain("invalid number of arguments");
		}
		else if (words[0]=="recursive_lensing")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include recursive lensing effects from multiple lens planes: " << display_switch(include_recursive_lensing) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'recursive_lensing' command; must specify 'on' or 'off'");
				set_switch(include_recursive_lensing,setword);
				recalculate_beta_factors();
				reset();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="show_cc")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Show critical curves: " << display_switch(show_cc) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'show_cc' command; must specify 'on' or 'off'");
				set_switch(show_cc,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="show_srcplane")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Show source plane when plotting: " << display_switch(plot_srcplane) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'show_srcplane' command; must specify 'on' or 'off'");
				set_switch(plot_srcplane,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="plot_key_outside")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Plot key outside figure: " << display_switch(plot_key_outside) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'plot_key_outside' command; must specify 'on' or 'off'");
				set_switch(plot_key_outside,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="plot_key")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Show plot key: " << display_switch(show_plot_key) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'plot_key' command; must specify 'on' or 'off'");
				set_switch(show_plot_key,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="plot_title")
		{
			if ((nwords > 1) and (words[1] == "=")) remove_word(1);
			if (nwords==1) {
				if (plot_title.empty()) Complain("plot title has not been set");
				if (mpi_id==0) cout << "Plot title: '" << plot_title << "'\n";
			} else {
				remove_word(0);
				plot_title = "";
				for (int i=0; i < nwords-1; i++) plot_title += words[i] + " ";
				plot_title += words[nwords-1];
				int pos;
				while ((pos = plot_title.find('"')) != string::npos) plot_title.erase(pos,1);
				while ((pos = plot_title.find('\'')) != string::npos) plot_title.erase(pos,1);
			}
		}
		else if (words[0]=="post_title")
		{
			if ((nwords > 1) and (words[1] == "=")) remove_word(1);
			if (nwords==1) {
				if (post_title.empty()) Complain("posterior triangle plot title has not been set");
				if (mpi_id==0) cout << "Posterior triangle plot title: '" << post_title << "'\n";
			} else {
				remove_word(0);
				post_title = "";
				for (int i=0; i < nwords-1; i++) post_title += words[i] + " ";
				post_title += words[nwords-1];
				int pos;
				while ((pos = post_title.find('"')) != string::npos) post_title.erase(pos,1);
				while ((pos = post_title.find('\'')) != string::npos) post_title.erase(pos,1);
			}
		}
		else if (words[0]=="chain_info")
		{
			if ((nwords > 1) and (words[1] == "=")) remove_word(1);
			if (nwords==1) {
				if (chain_info.empty()) Complain("chain description has not been set");
				if (mpi_id==0) cout << "Chain info: '" << chain_info << "'\n";
			} else {
				remove_word(0);
				chain_info = "";
				for (int i=0; i < nwords-1; i++) chain_info += words[i] + " ";
				chain_info += words[nwords-1];
				int pos;
				while ((pos = chain_info.find('"')) != string::npos) chain_info.erase(pos,1);
				while ((pos = chain_info.find('\'')) != string::npos) chain_info.erase(pos,1);
			}
		}
		else if (words[0]=="data_info")
		{
			if ((nwords > 1) and (words[1] == "=")) remove_word(1);
			if (nwords==1) {
				if (data_info.empty()) Complain("data description has not been set");
				if (mpi_id==0) cout << "Data info: '" << data_info << "'\n";
			} else {
				remove_word(0);
				data_info = "";
				for (int i=0; i < nwords-1; i++) data_info += words[i] + " ";
				data_info += words[nwords-1];
				int pos;
				while ((pos = data_info.find('"')) != string::npos) data_info.erase(pos,1);
				while ((pos = data_info.find('\'')) != string::npos) data_info.erase(pos,1);
			}
		}
		else if (words[0]=="param_markers")
		{
			if ((nwords > 1) and (words[1] == "=")) remove_word(1);
			if ((nwords==2) and (words[1] == "none")) {
				param_markers = "";
			} else if ((nwords==2) and (words[1] == "allparams")) {
				if (n_fit_parameters > 0) create_parameter_value_string(param_markers);
				else Complain("no fit parameters have been defined");
			} else if (nwords==1) {
				if (param_markers.empty()) Complain("parameter markers has not been set");
				if (mpi_id==0) cout << "Parameter marker values: '" << param_markers << "'\n";
			} else {
				remove_word(0);
				param_markers = "";
				for (int i=0; i < nwords; i++) {
					char* p;
					strtod(words[i].c_str(), &p);
					if (*p) {
						double pval;
						if (lookup_parameter_value(words[i],pval)==false) Complain("parameter name '" << words[i] << "' is not listed among the fit parameters");
						stringstream pvalstr;
						string pvalstring;
						pvalstr << pval;
						pvalstr >> pvalstring;
						param_markers += pvalstring;
					} else {
						param_markers += words[i]; // a number has been manually entered in, so just tack it on to the line
					}
					if (i != nwords-1) param_markers += " ";
				}
			}
		}
		else if (words[0]=="n_markers")
		{
			if (nwords==1) {
				if (n_param_markers==10000) {
					if (mpi_id==0) cout << "Number of parameter markers: all" << endl;
				} else {
					if (mpi_id==0) cout << "Number of parameter markers: " << n_param_markers << endl;
				}
			} else if (nwords==2) {
				if (words[1]=="all") n_param_markers = 10000;
				else {
					int npm;
					if (!(ws[1] >> npm)) Complain("invalid argument to 'n_markers' command");
					if ((npm >= 0) and (npm < 10000)) n_param_markers = npm;
					else Complain("invalid number of parameter markers; must be between 0 and 10000 (or enter 'all')");
				}
			} else Complain("only one argument allowed for 'n_markers'");
		}
		else if (words[0]=="subplot_params")
		{
			if (nwords==1) {
				if (!param_settings->subplot_params_defined()) Complain("No subplot parameters have been defined");
				else if (mpi_id==0) cout << "Subplot parameters: " << param_settings->print_subplot_params() << endl;
			}
			else if ((nwords==2) and (words[1]=="reset")) param_settings->reset_subplot_params();
			else {
				param_settings->reset_subplot_params();
				for (int i=1; i < nwords; i++) {
					if (!param_settings->set_subplot_param(words[i])) Complain("Fit parameter '" << words[i] << "' does not exist");
				}
			}
		}
		else if (words[0]=="colorbar")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Show color bar in pixel map plots: " << display_switch(show_colorbar) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'colorbar' command; must specify 'on' or 'off'");
				set_switch(show_colorbar,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="cbmin")
		{
			double cbmin;
			if (nwords == 2) {
				if (words[1]=="auto") {
					if (sbmin != -1e30) cbmin = sbmin; // sbmin was stored from last pixel image generated
				}
				else if (!(ws[1] >> cbmin)) Complain("invalid cbmin setting");
				colorbar_min = cbmin;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "colorbar min surface brightness = " << colorbar_min << endl;
			} else Complain("must specify either zero or one argument");
		}
		else if (words[0]=="cbmax")
		{
			double cbmax;
			if (nwords == 2) {
				if (words[1]=="auto") {
					if (sbmax != 1e30) cbmax = sbmax; // sbmax was stored from last pixel image generated
				}
				else if (!(ws[1] >> cbmax)) Complain("invalid cbmax setting");
				colorbar_max = cbmax;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "colorbar max surface brightness = " << colorbar_max << endl;
			} else Complain("must specify either zero or one argument");
		}
		else if (words[0]=="plot_square_axes")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Plot square axes: " << display_switch(plot_square_axes) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'plot_square_axes' command; must specify 'on' or 'off'");
				set_switch(plot_square_axes,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="fontsize")
		{
			if (nwords==1) {
				if (mpi_id==0) {
					cout << resetiosflags(ios::scientific);
					cout << fontsize << endl;
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
				}
			} else if (nwords==2) {
				if (!(ws[1] >> fontsize)) Complain("invalid font size; must be a real number");
			}
			else Complain("only one argument allowed for fontsize");
		}
		else if (words[0]=="linewidth")
		{
			if (nwords==1) {
				if (mpi_id==0) {
					cout << resetiosflags(ios::scientific);
					cout << linewidth << endl;
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
				}
			} else if (nwords==2) {
				if (!(ws[1] >> linewidth)) Complain("invalid line width; must be a real number");
			}
			else Complain("only one argument allowed for linewidth");
		}
		else if (words[0]=="ccspline")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Critical curve spline: " << display_switch(use_cc_spline) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'ccspline' command; must specify 'on' or 'off'");
				if (setword=="on") set_ccspline_mode(true);
				else if (setword=="off") {
					set_ccspline_mode(false);
					delete_ccspline();
				}
				else Complain("invalid argument to 'ccspline' command; must specify 'on' or 'off'");
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="auto_ccspline")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatic critical curve spline: " << display_switch(auto_ccspline) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_ccspline' command; must specify 'on' or 'off'");
				set_switch(auto_ccspline,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="autocenter")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatic grid center: " << display_switch(autocenter) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'autocenter' command; must specify 'on' or 'off'");
				set_switch(autocenter,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="primary_lens")
		{
			if (nwords==1) {
				if (mpi_id==0) {
					cout << "primary_lens = " << primary_lens_number << ((auto_set_primary_lens==true) ? " (auto)": "") << endl;
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'primary_lens' command; must specify lens number # or 'auto'");
				if (setword=="auto") {
					auto_set_primary_lens = true;
					set_primary_lens();
				} else {
					stringstream nstream;
					int lensnum;
					nstream << setword;
					if (!(nstream >> lensnum)) Complain("invalid argument to 'primary_lens' command; must specify lens number # or 'auto'");
					if ((lensnum < 0) or (lensnum >= nlens)) Complain("specified lens number for 'primary_lens' does not exist");
					primary_lens_number = lensnum;
					auto_set_primary_lens = false;
				}
			} else Complain("invalid number of arguments; can only specify lens number # or 'auto'");
		}
		else if (words[0]=="secondary_lens")
		{
			if (nwords==1) {
				if (mpi_id==0) {
					cout << "secondary_lens = ";
					if (include_secondary_lens==true) cout << secondary_lens_number;
					else cout << "none";
					cout << endl;
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'secondary_lens' command; must specify lens number # or 'auto'");
				if (setword=="none") {
					include_secondary_lens = false;
				} else {
					stringstream nstream;
					int lensnum;
					nstream << setword;
					if (!(nstream >> lensnum)) Complain("invalid argument to 'secondary_lens' command; must specify lens number # or 'auto'");
					if ((lensnum < 0) or (lensnum >= nlens)) Complain("specified lens number for 'secondary_lens' does not exist");
					secondary_lens_number = lensnum;
					include_secondary_lens = true;
				}
			} else Complain("invalid number of arguments; can only specify lens number # or 'auto'");
		}
		else if (words[0]=="auto_save_bestfit")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically save best-fit point after fit: " << display_switch(auto_save_bestfit) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_save_bestfit' command; must specify 'on' or 'off'");
				set_switch(auto_save_bestfit,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="autogrid_from_Re")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically set grid size from Einstein radius before grid creation: " << display_switch(auto_gridsize_from_einstein_radius) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'autogrid_from_Re' command; must specify 'on' or 'off'");
				set_switch(auto_gridsize_from_einstein_radius,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="autogrid_before_mkgrid")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically optimize grid parameters before grid creation: " << display_switch(autogrid_before_grid_creation) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'autogrid_before_mkgrid' command; must specify 'on' or 'off'");
				set_switch(autogrid_before_grid_creation,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="imgplane_chisq")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use image plane chi-square function: " << display_switch(use_image_plane_chisq) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'imgplane_chisq' command; must specify 'on' or 'off'");
				set_switch(use_image_plane_chisq,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
			if (nlens > 0) get_n_fit_parameters(n_fit_parameters); // update number of fit parameters, since source parameters might not have been included for source plane chi-square
		}
		else if (words[0]=="bayes_factor")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Calculate Bayes factor after running two model fits: " << display_switch(calculate_bayes_factor) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'bayes_factor' command; must specify 'on' or 'off'");
				set_switch(calculate_bayes_factor,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="analytic_bestfit_src")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Solve for approximate best-fit source coordinates analytically during fit: " << display_switch(use_analytic_bestfit_src) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'analytic_bestfit_src' command; must specify 'on' or 'off'");
				set_switch(use_analytic_bestfit_src,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="chisqmag")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include magnification in source plane chi-square: " << display_switch(use_magnification_in_chisq) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'chisqmag' command; must specify 'on' or 'off'");
				set_switch(use_magnification_in_chisq,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="chisqmag_on_repeats")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include magnification in chi-square during repeat optimizations: " << display_switch(use_magnification_in_chisq_during_repeats) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'chisqmag_on_repeats' command; must specify 'on' or 'off'");
				set_switch(use_magnification_in_chisq_during_repeats,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="chisq_mag_threshold")
		{
			double magthresh;
			if (nwords == 2) {
				if (!(ws[1] >> magthresh)) Complain("invalid magnification threshold for image plane chi-square");
				chisq_magnification_threshold = magthresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "magnification threshold for including image in chi-square function = " << chisq_magnification_threshold << endl;
			} else Complain("must specify either zero or one argument (chi-square magnification threshold)");
		}
		else if (words[0]=="imgsep_threshold")
		{
			double imgsepthresh;
			if (nwords == 2) {
				if (!(ws[1] >> imgsepthresh)) Complain("invalid image separation threshold for rejecting duplicate images");
				redundancy_separation_threshold = imgsepthresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "image separation threshold for rejecting duplicate images = " << redundancy_separation_threshold << endl;
			} else Complain("must specify either zero or one argument (image separation threshold for redundant images)");
		}
		else if (words[0]=="chisq_imgsep_threshold")
		{
			double imgsepthresh;
			if (nwords == 2) {
				if (!(ws[1] >> imgsepthresh)) Complain("invalid image separation threshold for image plane chi-square");
				chisq_imgsep_threshold = imgsepthresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "image separation threshold for including image in chi-square function = " << chisq_imgsep_threshold << endl;
			} else Complain("must specify either zero or one argument (chi-square image separation threshold)");
		}
		else if (words[0]=="chisq_imgplane_threshold")
		{
			double srcplanethresh;
			if (nwords == 2) {
				if (!(ws[1] >> srcplanethresh)) Complain("invalid threshold for substituting image plane chi-square for source plane chi-square");
				chisq_imgplane_substitute_threshold = srcplanethresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "threshold for substituting image plane chi-square for source plane chi-square function = " << chisq_imgplane_substitute_threshold << endl;
			} else Complain("must specify either zero or one argument (chi-square image plane substitution threshold)");
		}
		else if (words[0]=="chisqpos")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include image positions in chi-square: " << display_switch(include_imgpos_chisq) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'chisqpos' command; must specify 'on' or 'off'");
				set_switch(include_imgpos_chisq,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="chisqflux")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include image fluxes in chi-square: " << display_switch(include_flux_chisq) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'chisqflux' command; must specify 'on' or 'off'");
				set_switch(include_flux_chisq,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="chisq_weak_lensing")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include weak lensing reduced shear in chi-square: " << display_switch(include_weak_lensing_chisq) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'chisq_weak_lensing' command; must specify 'on' or 'off'");
				set_switch(include_weak_lensing_chisq,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="nimg_penalty")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include penalty function in chi-square for producing too many images: " << display_switch(n_images_penalty) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'nimg_penalty' command; must specify 'on' or 'off'");
				set_switch(n_images_penalty,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="fix_srcflux")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Fix source flux to specified value: " << display_switch(fix_source_flux) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'fix_srcflux' command; must specify 'on' or 'off'");
				set_switch(fix_source_flux,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="srcflux")
		{
			double srcflux;
			if (nwords == 2) {
				if (!(ws[1] >> srcflux)) Complain("invalid source flux setting");
				source_flux = srcflux;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "source flux = " << source_flux << endl;
			} else Complain("must specify either zero or one argument (source flux)");
		}
		else if (words[0]=="chisq_parity")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include parity information in flux chi-square: " << display_switch(include_parity_in_chisq) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'chisq_parity' command; must specify 'on' or 'off'");
				set_switch(include_parity_in_chisq,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="chisq_time_delays")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include time delays in chi-square: " << display_switch(include_time_delay_chisq) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'chisq_time_delays' command; must specify 'on' or 'off'");
				set_switch(include_time_delay_chisq,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="find_errors")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Calculate marginalized errors from Fisher matrix: " << display_switch(calculate_parameter_errors) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'find_errors' command; must specify 'on' or 'off'");
				set_switch(calculate_parameter_errors,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="central_image")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include central image in fit: " << display_switch(include_central_image) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'central_image' command; must specify 'on' or 'off'");
				set_switch(include_central_image,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="auto_zsrc_scaling")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically set source scaling of kappa (zsrc_ref) to source redshift: " << display_switch(auto_zsource_scaling) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_zsrc_scaling' command; must specify 'on' or 'off'");
				set_switch(auto_zsource_scaling,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="time_delays")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include time delays: " << display_switch(include_time_delays) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'time_delays' command; must specify 'on' or 'off'");
				set_switch(include_time_delays,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="zlens")
		{
			double zlens;
			if (nwords == 2) {
				if (!(ws[1] >> zlens)) Complain("invalid zlens setting");
				lens_redshift = zlens;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "lens redshift = " << lens_redshift << endl;
			} else Complain("must specify either zero or one argument (redshift of lens galaxy)");
		}
		else if (words[0]=="zsrc")
		{
			double zsource;
			if (nwords == 2) {
				if (!(ws[1] >> zsource)) Complain("invalid zsrc setting");
				set_source_redshift(zsource);
				user_changed_zsource = true; // keeps track of whether redshift has been manually changed; if so, then qlens won't automatically change it to redshift from data
				reset();
			} else if (nwords==1) {
				if (mpi_id==0) cout << "source redshift = " << source_redshift << endl;
			} else Complain("must specify either zero or one argument (redshift of source object)");
		}
		else if (words[0]=="zsrc_ref")
		{
			double zrsource;
			if (nwords == 2) {
				if (nlens > 0) Complain("zsrc_ref cannot be changed if any lenses have already been created");
				int i,j;
				if (!(ws[1] >> zrsource)) Complain("invalid zrsource_ref setting");
				reference_source_redshift = zrsource;
				if (auto_zsource_scaling==true) auto_zsource_scaling = false;
				for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = kappa_ratio(lens_redshifts[i],source_redshift,reference_source_redshift);
				reset();
				if (n_sourcepts_fit > 0) {
					for (i=0; i < n_sourcepts_fit; i++) {
						for (j=0; j < n_lens_redshifts; j++) {
							zfactors[i][j] = kappa_ratio(lens_redshifts[j],source_redshifts[i],reference_source_redshift);
						}
					}
				}

			} else if (nwords==1) {
				if (mpi_id==0) cout << "reference source redshift = " << reference_source_redshift << endl;
			} else Complain("must specify either zero or one argument (reference source redshift)");
		}
		else if (words[0]=="hubble")
		{
			double h0param;
			if (nwords == 2) {
				if (!(ws[1] >> h0param)) Complain("invalid hubble setting");
				hubble = h0param;
				set_cosmology(omega_matter,0.04,hubble,2.215);
				if ((vary_hubble_parameter) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
					if (mpi_id==0) cout << "Limits for Hubble parameter:\n";
					if (read_command(false)==false) return;
					double hmin,hmax;
					if (nwords != 2) Complain("Must specify two arguments for Hubble parameter limits: hmin, hmax");
					if (!(ws[0] >> hmin)) Complain("Invalid lower limit for Hubble parameter");
					if (!(ws[1] >> hmax)) Complain("Invalid upper limit for Hubble parameter");
					if (hmin > hmax) Complain("lower limit cannot be greater than upper limit");
					hubble_lower_limit = hmin;
					hubble_upper_limit = hmax;
				}
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Hubble parameter = " << hubble << endl;
			} else Complain("must specify either zero or one argument (Hubble parameter)");
		}
		else if (words[0]=="vary_hubble")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary Hubble parameter: " << display_switch(vary_hubble_parameter) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_hubble' command; must specify 'on' or 'off'");
				set_switch(vary_hubble_parameter,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="omega_m")
		{
			double om;
			if (nwords == 2) {
				if (!(ws[1] >> om)) Complain("invalid omega_m setting");
				omega_matter = om;
				set_cosmology(omega_matter,0.04,hubble,2.215);
				if ((vary_omega_matter_parameter) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
					if (mpi_id==0) cout << "Limits for omega_m parameter:\n";
					if (read_command(false)==false) return;
					double omin,omax;
					if (nwords != 2) Complain("Must specify two arguments for omega_m parameter limits: omin, omax");
					if (!(ws[0] >> omin)) Complain("Invalid lower limit for omega_m parameter");
					if (!(ws[1] >> omax)) Complain("Invalid upper limit for omega_m parameter");
					if (omin > omax) Complain("lower limit cannot be greater than upper limit");
					omega_matter_lower_limit = omin;
					omega_matter_upper_limit = omax;
				}
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Matter density (omega_m) = " << omega_matter << endl;
			} else Complain("must specify either zero or one argument (omega_m value)");
		}
		else if (words[0]=="vary_omega_m")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary omega_m parameter: " << display_switch(vary_omega_matter_parameter) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_omega_m' command; must specify 'on' or 'off'");
				set_switch(vary_omega_matter_parameter,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="syserr_pos")
		{
			double syserrparam;
			if (nwords == 2) {
				if (!(ws[1] >> syserrparam)) Complain("invalid syserr_pos setting");
				syserr_pos = syserrparam;
				if ((vary_syserr_pos_parameter) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
					if (mpi_id==0) cout << "Limits for systematic error parameter:\n";
					if (read_command(false)==false) return;
					double sigmin,sigmax;
					if (nwords != 2) Complain("Must specify two arguments for systematic error parameter limits: sigmin, sigmax");
					if (!(ws[0] >> sigmin)) Complain("Invalid lower limit for systematic error parameter");
					if (!(ws[1] >> sigmax)) Complain("Invalid upper limit for systematic error parameter");
					if (sigmin > sigmax) Complain("lower limit cannot be greater than upper limit");
					syserr_pos_lower_limit = sigmin;
					syserr_pos_upper_limit = sigmax;
				}
			} else if (nwords==1) {
				if (mpi_id==0) cout << "systematic error parameter = " << syserr_pos << endl;
			} else Complain("must specify either zero or one argument (systematic error parameter)");
		}
		else if (words[0]=="vary_syserr_pos")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary systematic error parameter: " << display_switch(vary_syserr_pos_parameter) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_syserr_pos' command; must specify 'on' or 'off'");
				set_switch(vary_syserr_pos_parameter,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="wl_shearfac")
		{
			double syserrparam;
			if (nwords == 2) {
				if (!(ws[1] >> syserrparam)) Complain("invalid wl_shearfac setting");
				wl_shear_factor = syserrparam;
				if ((vary_wl_shear_factor_parameter) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
					if (mpi_id==0) cout << "Limits for weak lensing scale factor parameter:\n";
					if (read_command(false)==false) return;
					double sigmin,sigmax;
					if (nwords != 2) Complain("Must specify two arguments for weak lensing scale factor parameter limits: sigmin, sigmax");
					if (!(ws[0] >> sigmin)) Complain("Invalid lower limit for weak lensing scale factor parameter");
					if (!(ws[1] >> sigmax)) Complain("Invalid upper limit for weak lensing scale factor parameter");
					if (sigmin > sigmax) Complain("lower limit cannot be greater than upper limit");
					wl_shear_factor_lower_limit = sigmin;
					wl_shear_factor_upper_limit = sigmax;
				}
			} else if (nwords==1) {
				if (mpi_id==0) cout << "weak lensing scale factor parameter = " << wl_shear_factor << endl;
			} else Complain("must specify either zero or one argument (weak lensing scale factor parameter)");
		}
		else if (words[0]=="vary_wl_shearfac")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary weak lensing scale factor parameter: " << display_switch(vary_wl_shear_factor_parameter) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_wl_shearfac' command; must specify 'on' or 'off'");
				set_switch(vary_wl_shear_factor_parameter,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if ((words[0]=="rsplit") or (words[0]=="xsplit"))
		{
			if ((words[0]=="rsplit") and (!radial_grid)) Complain("gridtype is set to 'cartesian'; must use 'xsplit' and 'ysplit' for initial splittings");
			if ((words[0]=="xsplit") and (radial_grid)) Complain("gridtype is set to 'radial'; must use 'rsplit' and 'thetasplit' for initial splittings");
			int split;
			if (nwords == 2) {
				if (!(ws[1] >> split)) Complain("invalid number of splittings");
				set_usplit_initial(split);
			} else if (nwords==1) {
				get_usplit_initial(split);
				if (mpi_id==0) {
					if (radial_grid) cout << "radial splittings = " << split << endl;
					else cout << "x-splittings = " << split << endl;
				}
			} else Complain("must specify one argument (number of splittings)");
		}
		else if ((words[0]=="thetasplit") or (words[0]=="ysplit"))
		{
			if ((words[0]=="thetasplit") and (!radial_grid)) Complain("gridtype is set to 'cartesian'; must use 'xsplit' and 'ysplit' for initial splittings");
			if ((words[0]=="ysplit") and (radial_grid)) Complain("gridtype is set to 'radial'; must use 'rsplit' and 'thetasplit' for initial splittings");

			int split;
			if (nwords == 2) {
				if (!(ws[1] >> split)) Complain("invalid number of splittings");
				set_wsplit_initial(split);
			} else if (nwords==1) {
				get_wsplit_initial(split);
				if (mpi_id==0) {
					if (radial_grid) cout << "angular splittings = " << split << endl;
					else cout << "y-splittings = " << split << endl;
				}
			} else Complain("must specify one argument (number of splittings)");
		}
		else if (words[0]=="cc_splitlevels")
		{
			int levels;
			if (nwords == 2) {
				if (!(ws[1] >> levels)) Complain("invalid number of split levels");
				set_cc_splitlevels(levels);
			} else if (nwords==1) {
				get_cc_splitlevels(levels);
				if (mpi_id==0) cout << "critical curve split levels = " << levels << endl;
			} else Complain("must specify either zero or one argument (number of split levels)");
		}
		else if (words[0]=="cc_split_neighbors")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Split cells adjacent to cells containing critical curves: " << display_switch(cc_neighbor_splittings) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'cc_split_neighbors' command; must specify 'on' or 'off'");
				set_switch(cc_neighbor_splittings,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="splitlevels")
		{
			// this should be kept at zero; in fact, it probably shouldn't even be an option
			int levels;
			if (nwords == 2) {
				if (!(ws[1] >> levels)) Complain("invalid number of split levels");
				splitlevels = levels;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "split levels = " << splitlevels << endl;
			} else Complain("must specify either zero or one argument (number of split levels)");
		}
		else if (words[0]=="imgpos_accuracy")
		{
			double imgpos_accuracy;
			if (nwords == 2) {
				if (!(ws[1] >> imgpos_accuracy)) Complain("invalid imgpos_accuracy setting");
				set_imagepos_accuracy(imgpos_accuracy);
			} else if (nwords==1) {
				imgpos_accuracy = Grid::image_pos_accuracy;
				if (mpi_id==0) cout << "image position imgpos_accuracy = " << imgpos_accuracy << endl;
			} else Complain("must specify either zero or one argument (image position accuracy)");
		}
		else if (words[0]=="imgsrch_mag_threshold")
		{
			double thresh;
			if (nwords == 2) {
				if (!(ws[1] >> thresh)) Complain("invalid imgsrch_mag_threshold setting");
				newton_magnification_threshold = thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Threshold to reject or warn about high magnification images: imgsrch_mag_threshold = " << newton_magnification_threshold << endl;
			} else Complain("must specify either zero or one argument for imgsrch_mag_threshold");
		}
		else if (words[0]=="reject_himag")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Reject images found that have magnification higher than imgsrch_mag_threshold: " << display_switch(reject_himag_images) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'reject_himag' command; must specify 'on' or 'off'");
				set_switch(reject_himag_images,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="reject_img_outside_cell")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Reject images found by Newton's method that lie outside original grid cell: " << display_switch(reject_images_found_outside_cell) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'reject_img_outside_cell' command; must specify 'on' or 'off'");
				set_switch(reject_images_found_outside_cell,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="chisqtol")
		{
			double tol;
			if (nwords == 2) {
				if (!(ws[1] >> tol)) Complain("invalid chisqtol setting");
				chisq_tolerance=tol;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "chi-square required accuracy = " << chisq_tolerance << endl;
			} else Complain("must specify either zero or one argument (required chi-square accuracy)");
		}
		else if (words[0]=="integral_tolerance")
		{
			double itolerance;
			if (nwords == 2) {
				if (!(ws[1] >> itolerance)) Complain("invalid value for integral_tolerance");
				set_integral_tolerance(itolerance);
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Tolerance for numerical integration = " << integral_tolerance << endl;
			} else Complain("must specify either zero or one argument for integral_tolerance");
		}
		else if (words[0]=="nrepeat")
		{
			int nrep;
			if (nwords == 2) {
				if (!(ws[1] >> nrep)) Complain("invalid nrepeat setting");
				n_repeats=nrep;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "number of repeat minimizations = " << n_repeats << endl;
			} else Complain("must specify either zero or one argument (number of repeat minimizations)");
		}
		else if (words[0]=="chisqstat_freq")
		{
			int freq;
			if (nwords == 2) {
				if (!(ws[1] >> freq)) Complain("invalid chisqstat_freq setting");
				chisq_display_frequency=freq;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "chi-square status display frequency = " << chisq_display_frequency << endl;
			} else Complain("must specify either zero or one argument (chi-square status display frequency)");
		}
		else if (words[0]=="min_cellsize")
		{
			double min_cellsize;
			if (nwords == 2) {
				if (!(ws[1] >> min_cellsize)) Complain("invalid value for min_cellsize");
				min_cell_area = SQR(min_cellsize);
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Minimum (average) subcell length allowed in grid = " << sqrt(min_cell_area) << endl;
			} else Complain("must specify either zero or one argument for min_cellsize");
		}
		else if (words[0]=="sim_err_pos")
		{
			double simerr;
			if (nwords == 2) {
				if (!(ws[1] >> simerr)) Complain("invalid position error");
				sim_err_pos = simerr;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "simulated (isotropic) error in image positions = " << sim_err_pos << endl;
			} else Complain("must specify either zero or one argument (error in simulated image positions)");
		}
		else if (words[0]=="sim_err_flux")
		{
			double simerr;
			if (nwords == 2) {
				if (!(ws[1] >> simerr)) Complain("invalid flux error");
				sim_err_flux = simerr;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "simulated error in image flux = " << sim_err_flux << endl;
			} else Complain("must specify either zero or one argument (error in simulated image flux)");
		}
		else if (words[0]=="sim_err_td")
		{
			double simerr;
			if (nwords == 2) {
				if (!(ws[1] >> simerr)) Complain("invalid position error");
				sim_err_td = simerr;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "simulated error in image time delays = " << sim_err_td << endl;
			} else Complain("must specify either zero or one argument (error in simulated image time delays)");
		}
		else if (words[0]=="subhalo_rmax")
		{
			if (nwords==2) {
				int lens_number;
				if (!(ws[1] >> lens_number)) Complain("invalid lens number");
				if (lens_number >= nlens) Complain("specified lens number for subhalo does not exist");
				if (lens_number == 0) Complain("perturber cannot be the primary lens (lens 0)");
				double rmax,menc,avgsig;
				if (nlens==1) Complain("perturber lens has not been defined");
				if (!calculate_critical_curve_perturbation_radius_numerical(lens_number,true,rmax,avgsig,menc)) Complain("could not calculate critical curve perturbation radius");
			} else Complain("one argument required for 'subhalo_rmax' (lens number for subhalo)");
		}
		else if (words[0]=="subhalo_rmax2")
		{
			if (nwords==2) {
				int lens_number;
				if (!(ws[1] >> lens_number)) Complain("invalid lens number");
				if (lens_number >= nlens) Complain("specified lens number for subhalo does not exist");
				if (lens_number == 0) Complain("perturber cannot be the primary lens (lens 0)");
				double rmax,menc,avgsig;
				if (nlens==1) Complain("perturber lens has not been defined");
				if (!calculate_critical_curve_perturbation_radius_numerical(lens_number,true,rmax,avgsig,menc,true)) Complain("could not calculate critical curve perturbation radius");
			} else Complain("one argument required for 'subhalo_rmax' (lens number for subhalo)");
		}
		else if (words[0]=="print_betavals")
		{
			print_beta_matrices();
		}
		else if (words[0]=="galsub_radius")
		{
			double galsub_radius;
			if (nwords == 2) {
				if (!(ws[1] >> galsub_radius)) Complain("invalid value for galsub_radius");
				set_galsubgrid_radius_fraction(galsub_radius);
			} else if (nwords==1) {
				get_galsubgrid_radius_fraction(galsub_radius);
				if (mpi_id==0) cout << "Satellite galaxy subgrid radius scaling = " << galsub_radius << endl;
			} else Complain("must specify either zero or one argument for galsub_radius");
		}
		else if (words[0]=="galsub_min_cellsize")
		{
			double galsub_min_cellsize;
			if (nwords == 2) {
				if (!(ws[1] >> galsub_min_cellsize)) Complain("invalid value for galsub_min_cellsize");
				set_galsubgrid_min_cellsize_fraction(galsub_min_cellsize);
			} else if (nwords==1) {
				get_galsubgrid_min_cellsize_fraction(galsub_min_cellsize);
				if (mpi_id==0) cout << "Satellite galaxy minimum cell length = " << galsub_min_cellsize << " (units of Einstein radius)" << endl;
			} else Complain("must specify either zero or one argument for galsub_min_cellsize");
		}
		else if (words[0]=="galsub_cc_splitlevels")
		{
			if (nwords == 2) {
				if (!(ws[1] >> galsubgrid_cc_splittings)) Complain("invalid value for galsub_cc_splitlevels");
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Number of critical curve split levels in perturbing galaxies = " << galsubgrid_cc_splittings << endl;
			} else Complain("must specify either zero or one argument for galsub_cc_splitlevels");
		}
		else if (words[0]=="rmin_frac")
		{
			double rminfrac;
			if (nwords == 2) {
				if (!(ws[1] >> rminfrac)) Complain("invalid rmin_frac");
				set_rminfrac(rminfrac);
			} else if (nwords==1) {
				get_rminfrac(rminfrac);
				if (mpi_id==0) cout << "rmin_frac = " << rminfrac << endl;
			} else Complain("must specify either zero or one argument (rminfrac)");
		}
		else if (words[0]=="tab_rmin")
		{
			if (nwords == 2) {
				double tab_rmin;
				if (!(ws[1] >> tab_rmin)) Complain("invalid tab_rmin");
				tabulate_rmin = tab_rmin;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "tab_rmin = " << tabulate_rmin << endl;
			} else Complain("must specify either zero or one argument (tab_rmin)");
		}
		else if (words[0]=="tab_qmin")
		{
			if (nwords == 2) {
				double tab_qmin;
				if (!(ws[1] >> tab_qmin)) Complain("invalid tab_qmin");
				tabulate_qmin = tab_qmin;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "tab_qmin = " << tabulate_qmin << endl;
			} else Complain("must specify either zero or one argument (tab_qmin)");
		}
		else if (words[0]=="tab_r_N")
		{
			if (nwords == 2) {
				int tab_r_N;
				if (!(ws[1] >> tab_r_N)) Complain("invalid tab_r_N");
				tabulate_logr_N = tab_r_N;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "tab_r_N = " << tabulate_logr_N << endl;
			} else Complain("must specify either zero or one argument (tab_r_N)");
		}
		else if (words[0]=="tab_phi_N")
		{
			if (nwords == 2) {
				int tab_phi_N;
				if (!(ws[1] >> tab_phi_N)) Complain("invalid tab_phi_N");
				tabulate_phi_N = tab_phi_N;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "tab_phi_N = " << tabulate_phi_N << endl;
			} else Complain("must specify either zero or one argument (tab_phi_N)");
		}
		else if (words[0]=="tab_q_N")
		{
			if (nwords == 2) {
				int tab_q_N;
				if (!(ws[1] >> tab_q_N)) Complain("invalid tab_q_N");
				tabulate_q_N = tab_q_N;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "tab_q_N = " << tabulate_q_N << endl;
			} else Complain("must specify either zero or one argument (tab_q_N)");
		}
		else if (words[0]=="psf_width")
		{
			double psfx, psfy;
			if (nwords == 3) {
				if (!(ws[1] >> psfx)) Complain("invalid PSF x-width");
				if (!(ws[2] >> psfy)) Complain("invalid PSF y-width");
				psf_width_x = psfx;
				psf_width_y = psfy;
			} else if (nwords == 2) {
				if (!(ws[1] >> psfx)) Complain("invalid PSF width");
				psf_width_x = psfx;
				psf_width_y = psfx;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Point spread function (PSF) width = (" << psf_width_x << "," << psf_width_y << ")\n";
			} else Complain("can only specify up to two arguments for PSF width (x-width,y-width)");
		}
		else if (words[0]=="psf_threshold")
		{
			double threshold;
			if (nwords == 2) {
				if (!(ws[1] >> threshold)) Complain("invalid PSF width");
				psf_threshold = threshold;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Point spread function (PSF) input threshold = " << psf_threshold << endl;
			} else Complain("can only specify up to one argument for PSF input threshold");
		}
		else if (words[0]=="pixel_fraction")
		{
			double frac, frac_ll, frac_ul;
 			if (nwords == 4) {
 				if (!(ws[1] >> frac_ll)) Complain("invalid source pixel fraction lower limit");
 				if (!(ws[2] >> frac)) Complain("invalid source pixel fraction value");
 				if (!(ws[3] >> frac_ul)) Complain("invalid source pixel fraction upper limit");
 				if ((frac < frac_ll) or (frac > frac_ul)) Complain("initial source pixel fraction should lie within specified prior limits");
 				pixel_fraction = frac;
 				pixel_fraction_lower_limit = frac_ll;
 				pixel_fraction_upper_limit = frac_ul;
 			} else if (nwords == 2) {
				if (!(ws[1] >> frac)) Complain("invalid firstlevel source pixel fraction");
				pixel_fraction = frac;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Source pixel fraction of srcpixels/imgpixels = " << pixel_fraction << endl;
			} else Complain("must specify one argument (firstlevel source pixel fraction)");
		}
		else if (words[0]=="srcgrid_xshift")
		{
			double xsh, xsh_ll, xsh_ul;
 			if (nwords == 4) {
 				if (!(ws[1] >> xsh_ll)) Complain("invalid source srcgrid_xshift lower limit");
 				if (!(ws[2] >> xsh)) Complain("invalid source srcgrid_xshift value");
 				if (!(ws[3] >> xsh_ul)) Complain("invalid source srcgrid_xshift upper limit");
 				if ((xsh < xsh_ll) or (xsh > xsh_ul)) Complain("initial source srcgrid_xshift should lie within specified prior limits");
 				srcgrid_xshift = xsh;
 				srcgrid_xshift_lower_limit = xsh_ll;
 				srcgrid_xshift_upper_limit = xsh_ul;
 			} else if (nwords == 2) {
				if (!(ws[1] >> xsh)) Complain("invalid firstlevel source srcgrid_xshift");
				srcgrid_xshift = xsh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "srcgrid_xshift = " << srcgrid_xshift << endl;
			} else Complain("must specify one argument (srcgrid_xshift)");
		}
		else if (words[0]=="srcgrid_yshift")
		{
			double ysh, ysh_ll, ysh_ul;
 			if (nwords == 4) {
 				if (!(ws[1] >> ysh_ll)) Complain("invalid source srcgrid_yshift lower limit");
 				if (!(ws[2] >> ysh)) Complain("invalid source srcgrid_yshift value");
 				if (!(ws[3] >> ysh_ul)) Complain("invalid source srcgrid_yshift upper limit");
 				if ((ysh < ysh_ll) or (ysh > ysh_ul)) Complain("initial source srcgrid_yshift should lie within specified prior limits");
 				srcgrid_yshift = ysh;
 				srcgrid_yshift_lower_limit = ysh_ll;
 				srcgrid_yshift_upper_limit = ysh_ul;
 			} else if (nwords == 2) {
				if (!(ws[1] >> ysh)) Complain("invalid firstlevel source srcgrid_yshift");
				srcgrid_yshift = ysh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "srcgrid_yshift = " << srcgrid_yshift << endl;
			} else Complain("must specify one argument (srcgrid_yshift)");
		}
		else if (words[0]=="srcgrid_scale")
		{
			double scale, scale_ll, scale_ul;
 			if (nwords == 4) {
 				if (!(ws[1] >> scale_ll)) Complain("invalid source srcgrid_scale lower limit");
 				if (!(ws[2] >> scale)) Complain("invalid source srcgrid_scale value");
 				if (!(ws[3] >> scale_ul)) Complain("invalid source srcgrid_scale upper limit");
 				if ((scale < scale_ll) or (scale > scale_ul)) Complain("initial source srcgrid_scale should lie within specified prior limits");
 				srcgrid_size_scale = scale;
 				srcgrid_size_scale_lower_limit = scale_ll;
 				srcgrid_size_scale_upper_limit = scale_ul;
 			} else if (nwords == 2) {
				if (!(ws[1] >> scale)) Complain("invalid firstlevel source srcgrid_scale");
				srcgrid_size_scale = scale;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "srcgrid_scale = " << srcgrid_size_scale << endl;
			} else Complain("must specify one argument (srcgrid_scale)");
		}
		else if (words[0]=="fits_format")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use FITS format for input surface brightness pixel files: " << display_switch(fits_format) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'fits_format' command; must specify 'on' or 'off'");
				set_switch(fits_format,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="data_pixel_size")
		{
			if (nwords==1) {
				if (mpi_id==0) {
					if (data_pixel_size < 0) cout << "Pixel size for loaded FITS images (data_pixel_size): not specified\n";
					else cout << "Pixel size for loaded FITS images (data_pixel_size): " << data_pixel_size << endl;
				}
			} else if (nwords==2) {
				double ps;
				if (!(ws[1] >> ps)) Complain("invalid argument to 'data_pixel_size' command");
				data_pixel_size = ps;
			} else Complain("invalid number of arguments to 'data_pixel_size'");
		}
		else if (words[0]=="vary_pixel_fraction")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary firstlevel source pixel fraction: " << display_switch(vary_pixel_fraction) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_pixel_fraction' command; must specify 'on' or 'off'");
				set_switch(vary_pixel_fraction,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="vary_srcgrid_xshift")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary source grid x-shift: " << display_switch(vary_srcgrid_xshift) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_srcgrid_xshift' command; must specify 'on' or 'off'");
				set_switch(vary_srcgrid_xshift,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="vary_srcgrid_yshift")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary source grid y-shift: " << display_switch(vary_srcgrid_yshift) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_srcgrid_yshift' command; must specify 'on' or 'off'");
				set_switch(vary_srcgrid_yshift,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="vary_srcgrid_scale")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary source grid size scaling: " << display_switch(vary_srcgrid_size_scale) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_srcgrid_scale' command; must specify 'on' or 'off'");
				set_switch(vary_srcgrid_size_scale,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="n_livepts")
		{
			int n_lp;
			if (nwords == 2) {
				if (!(ws[1] >> n_lp)) Complain("invalid number of live points for nested sampling");
				n_livepts = n_lp;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Number of live points for nested sampling = " << n_livepts << endl;
			} else Complain("must specify either zero or one argument (number of Monte Carlo points)");
		}
		else if (words[0]=="polychord_nrepeats")
		{
			int n_rp;
			if (nwords == 2) {
				if (!(ws[1] >> n_rp)) Complain("invalid num_repeats per parameter for polychord");
				polychord_nrepeats = n_rp;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "num_repeats per parameter for polychord = " << polychord_nrepeats << endl;
			} else Complain("must specify either zero or one argument (num_repeats per parameter for polychord)");
		}
		else if (words[0]=="simplex_nmax")
		{
			int nmax;
			if (nwords == 2) {
				if (!(ws[1] >> nmax)) Complain("invalid maximum number of iterations for downhill simplex");
				simplex_nmax = nmax;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Maximum number of iterations for downhill simplex = " << simplex_nmax << endl;
			} else Complain("must specify either zero or one argument (maximum number of iterations for downhill simplex)");
		}
		else if (words[0]=="simplex_nmax_anneal")
		{
			int nmax;
			if (nwords == 2) {
				if (!(ws[1] >> nmax)) Complain("invalid maximum number of iterations per temperature for downhill simplex");
				simplex_nmax_anneal = nmax;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Maximum number of iterations per temperature for downhill simplex = " << simplex_nmax_anneal << endl;
			} else Complain("must specify either zero or one argument (maximum number of iterations for downhill simplex)");
		}
		else if (words[0]=="simplex_minchisq")
		{
			int minchisq;
			if (nwords == 2) {
				if (!(ws[1] >> minchisq)) Complain("invalid minimum chi-square threshold for downhill simplex");
				simplex_minchisq = minchisq;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Minimum chi-square threshold for ending downhill simplex = " << simplex_minchisq << endl;
			} else Complain("must specify either zero or one argument (minimum chi-square threshold for downhill simplex)");
		}
		else if (words[0]=="simplex_minchisq_anneal")
		{
			int minchisq_anneal;
			if (nwords == 2) {
				if (!(ws[1] >> minchisq_anneal)) Complain("invalid minimum chi-square threshold for simulated annealing");
				simplex_minchisq_anneal = minchisq_anneal;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Minimum chi-square threshold for ending simulated annealing = " << simplex_minchisq_anneal << endl;
			} else Complain("must specify either zero or one argument (minimum chi-square threshold for simulated_annealing)");
		}
		else if (words[0]=="simplex_temp0")
		{
			double temp0;
			if (nwords == 2) {
				if (!(ws[1] >> temp0)) Complain("invalid initial temperature for downhill simplex");
				simplex_temp_initial = temp0;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Initial temperature for downhill simplex = " << simplex_temp_initial << endl;
			} else Complain("must specify either zero or one argument (initial temperature for downhill simplex)");
		}
		else if (words[0]=="simplex_tempf")
		{
			double tempf;
			if (nwords == 2) {
				if (!(ws[1] >> tempf)) Complain("invalid final temperature for downhill simplex");
				simplex_temp_final = tempf;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Final temperature for downhill simplex = " << simplex_temp_final << endl;
			} else Complain("must specify either zero or one argument (final temperature for downhill simplex)");
		}
		else if (words[0]=="simplex_tfac")
		{
			double tfac;
			if (nwords == 2) {
				if (!(ws[1] >> tfac)) Complain("invalid cooling factor for downhill simplex");
				simplex_cooling_factor = tfac;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Cooling factor for downhill simplex = " << simplex_cooling_factor << endl;
			} else Complain("must specify either zero or one argument (cooling factor for downhill simplex)");
		}
		else if (words[0]=="mcmc_chains")
		{
			int nt;
			if (nwords == 2) {
				if (!(ws[1] >> nt)) Complain("invalid number of MCMC chains");
				if (nt < 1) Complain("invalid number of MCMC chains");
				mcmc_threads = nt;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "number of chains for MCMC = " << mcmc_threads << endl;
			} else Complain("must specify either zero or one argument (number of MCMC chains)");
		}
		else if (words[0]=="mcmctol")
		{
			double tol;
			if (nwords == 2) {
				if (!(ws[1] >> tol)) Complain("invalid tolerance for MCMC");
				mcmc_tolerance = tol;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "MCMC tolerance = " << mcmc_tolerance << endl;
			} else Complain("must specify either zero or one argument (tolerance for MCMC)");
		}
		else if (words[0]=="mcmclog")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Output MCMC progress to logfile: " << display_switch(mcmc_logfile) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'mcmclog' command; must specify 'on' or 'off'");
				set_switch(mcmc_logfile,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="chisqlog")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Output chi-square evaluations to logfile: " << display_switch(open_chisq_logfile) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'chisqlog' command; must specify 'on' or 'off'");
				set_switch(open_chisq_logfile,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="simplex_show_bestfit")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Show current best-fit point while doing simulated annealing: " << display_switch(simplex_show_bestfit) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'anneal_show_bestfit' command; must specify 'on' or 'off'");
				set_switch(simplex_show_bestfit,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="psf_mpi")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use parallel PSF convolution with MPI: " << display_switch(psf_convolution_mpi) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'psf_mpi' command; must specify 'on' or 'off'");
				set_switch(psf_convolution_mpi,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="parallel_mumps")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use parallel analysis phase in MUMPS solver: " << display_switch(parallel_mumps) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'parallel_mumps' command; must specify 'on' or 'off'");
				set_switch(parallel_mumps,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="show_mumps_info")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Display MUMPS inversion information: " << display_switch(show_mumps_info) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'show_mumps_info' command; must specify 'on' or 'off'");
				set_switch(show_mumps_info,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="srcpixel_mag_threshold")
		{
			double threshold, threshold_ul, threshold_ll;
			if (nwords == 4) {
				if (!(ws[1] >> threshold_ll)) Complain("invalid source pixel magnification threshold lower limit");
				if (!(ws[2] >> threshold)) Complain("invalid source pixel magnification threshold value");
				if (!(ws[3] >> threshold_ul)) Complain("invalid source pixel magnification threshold upper limit");
				if ((threshold < threshold_ll) or (threshold > threshold_ul)) Complain("initial source pixel magnification threshold should lie within specified prior limits");
				pixel_magnification_threshold = threshold;
				pixel_magnification_threshold_lower_limit = threshold_ll;
				pixel_magnification_threshold_upper_limit = threshold_ul;
			} else if (nwords == 2) {
				if (!(ws[1] >> threshold)) Complain("invalid magnification threshold for source pixel splitting");
				pixel_magnification_threshold = threshold;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "source pixel magnification threshold = " << pixel_magnification_threshold << endl;
			} else Complain("must specify one argument (magnification threshold for source pixel splitting)");
		}
		else if (words[0]=="vary_srcpixel_mag_threshold")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary source pixel magnification threshold: " << display_switch(vary_magnification_threshold) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_srcpixel_mag_threshold' command; must specify 'on' or 'off'");
				set_switch(vary_magnification_threshold,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="src_imgpixel_ratio")
		{
			double ratio;
			if (nwords == 2) {
				if (!(ws[1] >> ratio)) Complain("invalid ratio");
				base_srcpixel_imgpixel_ratio = ratio;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "src_imgpixel_ratio = " << base_srcpixel_imgpixel_ratio << endl;
			} else Complain("must specify either zero or one argument (ratio)");
		}
		else if (words[0]=="regparam")
		{
			double regparam, regparam_ul, regparam_ll;
			if (nwords == 4) {
				if (!(ws[1] >> regparam_ll)) Complain("invalid regularization parameter lower limit");
				if (!(ws[2] >> regparam)) Complain("invalid regularization parameter value");
				if (!(ws[3] >> regparam_ul)) Complain("invalid regularization parameter upper limit");
				if ((regparam < regparam_ll) or (regparam > regparam_ul)) Complain("initial regularization parameter should lie within specified prior limits");
				regularization_parameter = regparam;
				regularization_parameter_lower_limit = regparam_ll;
				regularization_parameter_upper_limit = regparam_ul;
			} else if (nwords == 2) {
				if (!(ws[1] >> regparam)) Complain("invalid regularization parameter value");
				regularization_parameter = regparam;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "regularization parameter = " << regularization_parameter << endl;
			} else Complain("must specify either zero or one argument (regularization parameter value)");
		}
		else if (words[0]=="vary_regparam")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary regularization parameter: " << display_switch(vary_regularization_parameter) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_regparam' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before regparam can be varied (see 'fit regularization')");
				if ((setword=="on") and (source_fit_mode != Pixellated_Source)) Complain("regparam can only be varied if source mode is set to 'pixel' (see 'fit source_mode')");
				set_switch(vary_regularization_parameter,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="data_pixel_noise")
		{
			double pnoise;
			if (nwords == 2) {
				if (!(ws[1] >> pnoise)) Complain("invalid data pixel surface brightness noise");
				data_pixel_noise = pnoise;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "data pixel surface brightness dispersion = " << data_pixel_noise << endl;
			} else Complain("must specify either zero or one argument (data pixel surface brightness dispersion)");
		}
		else if (words[0]=="sim_pixel_noise")
		{
			double pnoise;
			if (nwords == 2) {
				if (!(ws[1] >> pnoise)) Complain("invalid simulated pixel surface brightness dispersion");
				sim_pixel_noise = pnoise;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Simulated pixel surface brightness dispersion = " << sim_pixel_noise << endl;
			} else Complain("must specify either zero or one argument (simulated pixel surface brightness noise)");
		}
		else if (words[0]=="random_seed")
		{
			long long int random_seed;
			if (nwords == 2) {
				if (!(ws[1] >> random_seed)) Complain("invalid value for random seed");
				set_random_seed(random_seed);
			} else if (nwords==1) {
				random_seed = get_random_seed();
				if (mpi_id==0) cout << "Random number generator seed = " << random_seed << endl;
			} else Complain("must specify either zero or one argument for random_seed");
		}
		else if (words[0]=="sb_threshold")
		{
			double sbthresh;
			if (nwords == 2) {
				if (!(ws[1] >> sbthresh)) Complain("invalid surface brightness threshold for centroid finding");
				sb_threshold = sbthresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "surface brightness threshold = " << sb_threshold << endl;
			} else Complain("must specify either zero or one argument (surface brightness threshold for centroid finding)");
		}
		else if (words[0]=="high_sn_threshold")
		{
			double sbthresh;
			if (nwords == 2) {
				if (!(ws[1] >> sbthresh)) Complain("invalid surface brightness fraction threshold for high S/N window");
				high_sn_frac = sbthresh;
				if (image_pixel_data != NULL) image_pixel_data->assign_high_sn_pixels();
			} else if (nwords==1) {
				if (mpi_id==0) cout << "high signal frac threshold = " << high_sn_frac << endl;
			} else Complain("must specify either zero or one argument");
		}
		else if (words[0]=="noise_threshold")
		{
			double noise_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> noise_thresh)) Complain("invalid noise threshold for automatic source grid sizing");
				noise_threshold = noise_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Noise threshold for automatic srcpixel grid sizing = " << noise_threshold << endl;
			} else Complain("must specify either zero or one argument (noise threshold for automatic source grid sizing)");
		}
		else if (words[0]=="outside_sb_noise_threshold")
		{
			double sb_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> sb_thresh)) Complain("invalid surface brightness noise threshold (should be as multiple of data noise)");
				outside_sb_prior_noise_frac = sb_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "surface brightness fraction threshold for outside_sb_noise_threshold = " << outside_sb_prior_noise_frac << endl;
			} else Complain("must specify either zero or one argument for outside_sb_noise_threshold");
		}
		else if (words[0]=="outside_sb_threshold")
		{
			double sb_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> sb_thresh)) Complain("invalid surface brightness noise threshold (should be as fraction of max s.b.)");
				outside_sb_prior_threshold = sb_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "surface brightness fraction threshold for outside_sb_threshold = " << outside_sb_prior_threshold << endl;
			} else Complain("must specify either zero or one argument for outside_sb_threshold");
		}
		else if (words[0]=="outside_sb_n_neighbors")
		{
			double sb_n;
			if (nwords == 2) {
				if (words[1]=="all") sb_n = -1;
				else if (!(ws[1] >> sb_n)) Complain("invalid number of neighbor pixels to search for outside_sb_prior");
				outside_sb_prior_n_neighbors = sb_n;
			} else if (nwords==1) {
				if (mpi_id==0) {
					if (outside_sb_prior_n_neighbors==-1) cout << "number of neighbor pixels to search: outside_sb_n_neighbors = all" << endl;
					else cout << "number of neighbor pixels to search: outside_sb_n_neighbors = " << outside_sb_prior_n_neighbors << endl;
				}
			} else Complain("must specify either zero or one argument for outside_sb_n_neighbors");
		}
		else if (words[0]=="nimg_sb_threshold")
		{
			double sb_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> sb_thresh)) Complain("invalid surface brightness noise threshold (should be as multiple of data noise)");
				n_image_prior_sb_frac = sb_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "surface brightness fraction threshold for nimg_sb_threshold = " << n_image_prior_sb_frac << endl;
			} else Complain("must specify either zero or one argument for nimg_sb_threshold");
		}
		else if (words[0]=="adaptive_grid")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Adaptive source pixel grid (adaptive_grid): " << display_switch(adaptive_grid) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'adaptive_grid' command; must specify 'on' or 'off'");
				set_switch(adaptive_grid,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="auto_srcgrid_set_pixelsize")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Determine source grid pixel size from magnifications: " << display_switch(auto_srcgrid_set_pixel_size) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_srcgrid_set_pixelsize' command; must specify 'on' or 'off'");
				set_switch(auto_srcgrid_set_pixel_size,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="nimg_prior")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Set prior on number of images: " << display_switch(n_image_prior) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'nimg_prior' command; must specify 'on' or 'off'");
				set_switch(n_image_prior,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="nimg_threshold")
		{
			double nimg_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> nimg_thresh)) Complain("invalid number of images at max surface brightness");
				n_image_threshold = nimg_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "threshold number of images at max surface brightness = " << n_image_threshold << endl;
			} else Complain("must specify either zero or one argument for nimg_threshold");
		}
		else if (words[0]=="outside_sb_prior")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Set prior on maximum surface brightness allowed beyond pixel mask: " << display_switch(outside_sb_prior) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'outside_sb_prior' command; must specify 'on' or 'off'");
				set_switch(outside_sb_prior,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="subhalo_prior")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Constrain subhalos (with pjaffe/corecusp profile) to lie within pixel mask: " << display_switch(subhalo_prior) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'subhalo_prior' command; must specify 'on' or 'off'");
				set_switch(subhalo_prior,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="custom_prior")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use custom prior: " << display_switch(use_custom_prior) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'custom_prior' command; must specify 'on' or 'off'");
				set_switch(use_custom_prior,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="activate_unmapped_srcpixels")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Activate source pixels that do not map to any image pixels: " << display_switch(activate_unmapped_source_pixels) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'activate_unmapped_srcpixels' command; must specify 'on' or 'off'");
				set_switch(activate_unmapped_source_pixels,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="exclude_srcpixels_outside_mask")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Exclude source pixels that fall outside masked region (when mapped to source plane): " << display_switch(exclude_source_pixels_beyond_fit_window) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'exclude_srcpixels_outside_mask' command; must specify 'on' or 'off'");
				set_switch(exclude_source_pixels_beyond_fit_window,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="remove_unmapped_subpixels")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Remove source subpixels that do not map to any image pixels: " << display_switch(regrid_if_unmapped_source_subpixels) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'remove_unmapped_subpixels' command; must specify 'on' or 'off'");
				set_switch(regrid_if_unmapped_source_subpixels,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="split_imgpixels")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Split image pixels when ray tracing: " << display_switch(split_imgpixels) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'split_imgpixels' command; must specify 'on' or 'off'");
				set_switch(split_imgpixels,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="imgpixel_nsplit")
		{
			if (nwords == 2) {
				int nt;
				if (!(ws[1] >> nt)) Complain("invalid number of image pixel splittings");
				default_imgpixel_nsplit = nt;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "default number of image pixel splittings = " << default_imgpixel_nsplit << endl;
			} else Complain("must specify either zero or one argument (default number of image pixel splittings)");
		}
		else if (words[0]=="galsubgrid")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Subgrid around perturbing lenses: " << display_switch(subgrid_around_perturbers) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'galsubgrid' command; must specify 'on' or 'off'");
				set_switch(subgrid_around_perturbers,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="galsubgrid_near_imgs")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Subgrid around perturbing lenses only near data images: " << display_switch(subgrid_only_near_data_images) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'galsubgrid_near_imgs' command; must specify 'on' or 'off'");
				set_switch(subgrid_only_near_data_images,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="use_perturber_flags")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Identify perturbers by flagging them: " << display_switch(use_perturber_flags) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'use_perturber_flags' command; must specify 'on' or 'off'");
				set_switch(use_perturber_flags,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="inversion_nthreads")
		{
			if (nwords == 2) {
				int nt;
				if (!(ws[1] >> nt)) Complain("invalid number of threads");
				inversion_nthreads = nt;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "inversion # of threads = " << inversion_nthreads << endl;
			} else Complain("must specify either zero or one argument (number of threads for inversion)");
		}
		else if (words[0]=="raytrace_method") {
			if (nwords==1) {
				if (mpi_id==0) {
					if (ray_tracing_method==Area_Overlap) cout << "Ray tracing method: area overlap" << endl;
					else if ((ray_tracing_method==Interpolate) and (interpolate_sb_3pt)) cout << "Ray tracing method: linear 3-point interpolation" << endl;
					else if ((ray_tracing_method==Interpolate) and (!interpolate_sb_3pt)) cout << "Ray tracing method: direct (nearest source pixel used)" << endl;
					else cout << "Unknown ray tracing method" << endl;
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'raytrace_method' command; must specify valid ray tracing method");
				if (setword=="overlap") ray_tracing_method = Area_Overlap;
				else if (setword=="interpolate") ray_tracing_method = Interpolate;
				else if (setword=="direct") { ray_tracing_method = Interpolate; interpolate_sb_3pt = false; }
				else Complain("invalid argument to 'raytrace_method' command; must specify valid ray tracing method");
			} else Complain("invalid number of arguments; can only specify ray tracing method");
		}
		else if (words[0]=="raytrace_imgplane_wgt")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Weight subpixel raytracing by image plane area: " << display_switch(weight_interpolation_by_imgplane_area) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'raytrace_imgplane_wgt' command; must specify 'on' or 'off'");
				set_switch(weight_interpolation_by_imgplane_area,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="inversion_method") {
			if (nwords==1) {
				if (mpi_id==0) {
					if (inversion_method==MUMPS) cout << "Lensing inversion method: LDL factorization (MUMPS)" << endl;
					else if (inversion_method==UMFPACK) cout << "Lensing inversion method: LU factorization (UMFPACK)" << endl;
					else if (inversion_method==CG_Method) cout << "Lensing inversion method: conjugate gradient method" << endl;
					else cout << "Unknown inversion method" << endl;
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'inversion_method' command; must specify valid inversion method");
				if (setword=="mumps") inversion_method = MUMPS;
				else if (setword=="umfpack") inversion_method = UMFPACK;
				else if (setword=="cg") inversion_method = CG_Method;
				else Complain("invalid argument to 'inversion_method' command; must specify valid inversion method");
			} else Complain("invalid number of arguments; can only inversion method");
		}
		else if (words[0]=="auto_srcgrid")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically determine source grid dimensions: " << display_switch(auto_sourcegrid) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_srcgrid' command; must specify 'on' or 'off'");
				set_switch(auto_sourcegrid,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="auto_src_npixels")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically determine number of source grid pixels: " << display_switch(auto_srcgrid_npixels) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_src_npixels' command; must specify 'on' or 'off'");
				set_switch(auto_srcgrid_npixels,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="autogrid_frac")
		{
			double frac;
			if (nwords == 2) {
				if (!(ws[1] >> frac)) Complain("invalid autogrid_frac");
				autogrid_frac = frac;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "autogrid_frac = " << autogrid_frac << endl;
			} else Complain("must specify either zero or one argument (autogrid_frac)");
		}
		else if ((words[0]=="terminal") or (words[0]=="term"))
		{
			if (nwords==1) {
				if (mpi_id==0) {
					if (terminal==TEXT) cout << "Plot file output format: text" << endl;
					else if (terminal==POSTSCRIPT) cout << "Plot file output format: postscript" << endl;
					else if (terminal==PDF) cout << "Plot file output format: PDF" << endl;
					else cout << "Unknown terminal" << endl;
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'terminal' command; must specify terminal type");
				if (setword=="text") terminal = TEXT;
				else if ((setword=="ps") or (setword=="postscript")) terminal = POSTSCRIPT;
				else if ((setword=="pdf") or (setword=="PDF")) terminal = PDF;
				else Complain("invalid argument to 'terminal' command; must specify terminal type");
			} else Complain("invalid number of arguments; can only specify terminal type");
		}
		else if (words[0]=="pause")
		{
			if (read_from_file) {
				if (!quit_after_reading_file) {   // if 'q' option is given, skip pauses
					if ((mpi_id==0) and (verbal_mode)) cout << "Pausing script file... (enter 'continue' or simply 'c' to continue)\n";
					read_from_file = false;
				}
			}
			else Complain("script file is not currently being read");
		}
		else if ((words[0]=="continue") or (words[0]=="c"))
		{
			if (read_from_file) Complain("script file is already being read");
			if (!infile->is_open()) Complain("no script file is currently loaded");
			read_from_file = true;
		}
		else if (words[0]=="sleep")
		{
			double time_sec = 1.0;
			if (nwords==2) {
				if (!(ws[1] >> time_sec)) Complain("invalid sleep time (should be number multiple of seconds)");
			}
			usleep(time_sec*1e6);
		}
		else if (words[0]=="test") {
			plot_weak_lensing_shear_field();
			//if (add_dparams_to_chain()==false) Complain("could not process chain data");
			//fitmodel_custom_prior();
			//if (lens_list[0]->update_specific_parameter("theta",60)==false) Complain("could not find specified parameter");
			//output_imgplane_chisq_vals();
			//add_derived_param(KappaR,5.0,-1);
			//add_derived_param(DKappaR,5.0,-1);
			//generate_solution_chain_sdp81();
			// These are experimental functions that I either need to make official, or else remove
			//plot_chisq_1d(0,30,50,450);
			//plot_chisq_2d(3,4,20,-0.1,0.1,20,-0.1,0.1); // implement this as a command later; probably should make a 1d version as well
			//make_perturber_population(0.04444,2500,0.1,0.6);
			//plot_perturber_deflection_vs_area();
			//double rmax,menc;
			//calculate_critical_curve_deformation_radius(nlens-1,true,rmax,menc);
			//calculate_critical_curve_deformation_radius_numerical(nlens-1);
			//plot_shear_field(-3,3,50,-3,3,50);
			//plot_shear_field(1e-3,2,300,1e-3,2,300);
		}
		else if (words[0]=="test2") {
			double xmin,xmax,ymin,ymax;
			xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			plot_shear_field(xmin,xmax,100,ymin,ymax,100);
		}
		else if (mpi_id==0) Complain("command not recognized");
	}
	free(buffer);
}

bool Lens::read_command(bool show_prompt)
{
	if (read_from_file) {
		if (infile->eof()) {
			infile->close();
			if (n_infiles > 1) infile--;
			n_infiles--;
			if (n_infiles == 0) {
				read_from_file = false;
				if (quit_after_reading_file) return false;
			} else if (paused_while_reading_file) {
				read_from_file = false;
				paused_while_reading_file = false;
			}
		} else {
			getline((*infile),line);
			lines.push_back(line);
			if ((verbal_mode) and (mpi_id==0)) cout << line << endl;
		}
	} else {
		if (mpi_id==0) {
			if (show_prompt) buffer = readline("> ");
			else buffer = readline("");
		}
		if (buffer==NULL) nullflag=1;
		else {
			nullflag=0;
			buffer_length = strlen(buffer);
		}
#ifdef USE_MPI
		MPI_Bcast(&nullflag,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
		if (nullflag==1) { if (mpi_id==0) cout << endl; return false; }
#ifdef USE_MPI
		MPI_Bcast(&buffer_length,1,MPI_INT,0,MPI_COMM_WORLD);
		if (mpi_id != 0) {
			if (buffer != NULL) delete[] buffer;
			buffer = new char[buffer_length+1];
		}
		MPI_Bcast(buffer,buffer_length+1,MPI_CHAR,0,MPI_COMM_WORLD);
#endif
		line.assign(buffer);
		if (buffer[0] != 0) {
			add_history(buffer);
			lines.push_back(line);
		}
	}
	words.clear();
	if ((line.empty()) or (line=="\r")) return read_command(show_prompt); // skip to the next line if this one is blank (you get carriage return "\r" from Mac or Windows editors)
	remove_comments(line);
	
	istringstream linestream(line);
	string word;
	while (linestream >> word)
		words.push_back(word);
	nwords = words.size();
	if (nwords==0) return read_command(show_prompt); // if the whole line was a comment, or full of spaces, go to the next line
	if (ws != NULL) delete[] ws;
	ws = new stringstream[nwords];
	for (int i=0; i < nwords; i++) ws[i] << words[i];
	return true;
}

bool Lens::check_vary_z()
{
	int pos, varyz = 0;
	for (int i=0; i < nwords; i++) {
		if ((pos = words[i].find("varyz=")) != string::npos) {
			string znumstring = words[i].substr(pos+6);
			stringstream znumstr;
			znumstr << znumstring;
			if (!(znumstr >> varyz)) { warn("incorrect format for varying lens redshift (should be 0 or 1)"); return false; }
			remove_word(i);
			break;
		}
	}
	if (varyz==0) return false;
	else return true;
}

bool Lens::open_command_file(char *filename)
{
	if (infile->is_open()) {
		if (n_infiles==10) {
			warn("cannot open more than 10 files at once");
			return false;
		}
		infile++;
	}
	infile->open(filename);
	if (infile->is_open()) n_infiles++;
	else if (n_infiles > 0) infile--;
	return (infile->is_open()) ? true : false;
}

void Lens::remove_equal_sign(void)
{
	int pos;
	if ((pos = words[0].find('=')) != string::npos) {
		// there's an equal sign in the first word, so remove it and separate into two words
		words.push_back("");
		for (int i=nwords-1; i > 0; i--) words[i+1] = words[i];
		words[1] = words[0].substr(pos+1);
		words[0] = words[0].substr(0,pos);
		nwords++;
		if (ws != NULL) delete[] ws;
		ws = new stringstream[nwords];
		for (int i=0; i < nwords; i++) ws[i] << words[i];
	}
	else if ((nwords == 3) and (words[1]=="="))
	{
		// there's an equal sign in the second of three words (indicating a parameter assignment), so remove it and reduce to two words
		string word1,word2;
		word1=words[0]; word2=words[2];
		words.clear();
		words.push_back(word1);
		words.push_back(word2);
		nwords = 2;
		delete[] ws;
		ws = new stringstream[nwords];
		ws[0] << words[0]; ws[1] << words[1];
	}
}

void Lens::remove_word(int n_remove)
{
	if (n_remove >= nwords) die("word number to remove is greater than number of words in command");
	string *word_temp = new string[nwords-1];
	int j=0;
	for (int i=0; i < nwords; i++) {
		if (i==n_remove) continue;
		word_temp[j++] = words[i];
	}
	words.clear();
	nwords--;
	for (j=0; j < nwords; j++) words.push_back(word_temp[j]);
	delete[] ws;
	ws = new stringstream[nwords];
	for (j=0; j < nwords; j++) ws[j] << words[j];
	delete[] word_temp;
}

void Lens::extract_word_starts_with(const char initial_character, int starting_word, int ending_word, string& extracted_word)
{
	if (starting_word >= nwords) { extracted_word = ""; return; }
	int end;
	if (ending_word >= nwords) end = nwords-1;
	else end = ending_word;
	int nremove = -1;
	for (int i=starting_word; i <= end; i++) {
		if (words[i][0] == initial_character) { nremove = i; extracted_word = words[i]; break; }
	}
	if (nremove==-1) extracted_word = "";
	else remove_word(nremove);
}

bool Lens::extract_word_starts_with(const char initial_character, int starting_word, int ending_word, vector<string>& extracted_words)
{
	bool extracted = false;
	if (starting_word >= nwords) return false;
	int end;
	if (ending_word >= nwords) end = nwords-1;
	else end = ending_word;
	//vector<int> remove_i;
	for (int i=end; i >= starting_word; i--) {
		if (words[i][0] == initial_character) { if (!extracted) extracted = true; extracted_words.push_back(words[i]); remove_word(i); }
	}
	//for (int i=remove_i.size()-1; i >= 0; i--) remove_word(remove_i[i]);
	return extracted;
}

void Lens::run_plotter(string plotcommand)
{
	if (suppress_plots) return;
	if (mpi_id==0) {
		stringstream psstr, ptstr, fsstr, lwstr;
		string psstring, ptstring, fsstring, lwstring;
		psstr << plot_ptsize;
		psstr >> psstring;
		ptstr << plot_pttype;
		ptstr >> ptstring;
		fsstr << fontsize;
		fsstr >> fsstring;
		lwstr << linewidth;
		lwstr >> lwstring;
		stringstream cbminstr, cbmaxstr;
		string cbmin, cbmax;
		cbminstr << colorbar_min;
		cbminstr >> cbmin;
		cbmaxstr << colorbar_max;
		cbmaxstr >> cbmax;
		string command = "plotlens " + plotcommand + " ps=" + psstring + " pt=" + ptstring + " lw=" + lwstring + " fs=" + fsstring;
		if (fit_output_dir != ".") command += " dir=" + fit_output_dir;
		if (!show_plot_key) command += " key=-1";
		else if (plot_key_outside) command += " key=1";
		if (plot_title != "") command += " title='" + plot_title + "'";
		if (show_colorbar==false) command += " nocb";
		if (colorbar_min != -1e30) command += " cbmin=" + cbmin;
		if (colorbar_max != 1e30) command += " cbmax=" + cbmax;
		if (plot_square_axes==true) command += " square";
		if (show_imgsrch_grid==true) command += " grid";
		system(command.c_str());
	}
}

void Lens::run_plotter_file(string plotcommand, string filename)
{
	if (suppress_plots) return;
	if (mpi_id==0) {
		stringstream psstr, ptstr, fsstr, lwstr;
		string psstring, ptstring, fsstring, lwstring;
		psstr << plot_ptsize;
		psstr >> psstring;
		ptstr << plot_pttype;
		ptstr >> ptstring;
		fsstr << fontsize;
		fsstr >> fsstring;
		lwstr << linewidth;
		lwstr >> lwstring;
		stringstream cbminstr, cbmaxstr;
		string cbmin, cbmax;
		cbminstr << colorbar_min;
		cbminstr >> cbmin;
		cbmaxstr << colorbar_max;
		cbmaxstr >> cbmax;
		string command = "plotlens " + plotcommand + " file=" + filename + " ps=" + psstring + " pt=" + ptstring + " lw=" + lwstring + " fs=" + fsstring;
		if (plot_title != "") command += " title='" + plot_title + "'";
		if (terminal==POSTSCRIPT) command += " term=postscript";
		else if (terminal==PDF) command += " term=pdf";
		if (fit_output_dir != ".") command += " dir=" + fit_output_dir;
		if (!show_plot_key) command += " key=-1";
		else if (plot_key_outside) command += " key=1";
		if (show_colorbar==false) command += " nocb";
		if (colorbar_min != -1e30) command += " cbmin=" + cbmin;
		if (colorbar_max != 1e30) command += " cbmax=" + cbmax;
		if (plot_square_axes==true) command += " square";
		if (show_imgsrch_grid==true) command += " grid";
		system(command.c_str());
	}
}

void Lens::run_plotter_range(string plotcommand, string range)
{
	if (suppress_plots) return;
	if (mpi_id==0) {
		stringstream psstr, ptstr, fsstr, lwstr;
		string psstring, ptstring, fsstring, lwstring;
		psstr << plot_ptsize;
		psstr >> psstring;
		ptstr << plot_pttype;
		ptstr >> ptstring;
		fsstr << fontsize;
		fsstr >> fsstring;
		lwstr << linewidth;
		lwstr >> lwstring;
		stringstream cbminstr, cbmaxstr;
		string cbmin, cbmax;
		cbminstr << colorbar_min;
		cbminstr >> cbmin;
		cbmaxstr << colorbar_max;
		cbmaxstr >> cbmax;
		string command = "plotlens " + plotcommand + " " + range + " ps=" + psstring + " pt=" + ptstring + " lw=" + lwstring + " fs=" + fsstring;
		if (plot_title != "") command += " title='" + plot_title + "'";
		if (fit_output_dir != ".") command += " dir=" + fit_output_dir;
		if (!show_plot_key) command += " key=-1";
		else if (plot_key_outside) command += " key=1";
		if (show_colorbar==false) command += " nocb";
		if (colorbar_min != -1e30) command += " cbmin=" + cbmin;
		if (colorbar_max != 1e30) command += " cbmax=" + cbmax;
		if (plot_square_axes==true) command += " square";
		if (show_imgsrch_grid==true) command += " grid";
		system(command.c_str());
	}
}

void Lens::run_plotter(string plotcommand, string filename, string extra_command)
{
	if (suppress_plots) return;
	if (mpi_id==0) {
		stringstream psstr, ptstr, fsstr, lwstr;
		string psstring, ptstring, fsstring, lwstring;
		psstr << plot_ptsize;
		psstr >> psstring;
		ptstr << plot_pttype;
		ptstr >> ptstring;
		fsstr << fontsize;
		fsstr >> fsstring;
		lwstr << linewidth;
		lwstr >> lwstring;
		stringstream cbminstr, cbmaxstr;
		string cbmin, cbmax;
		cbminstr << colorbar_min;
		cbminstr >> cbmin;
		cbmaxstr << colorbar_max;
		cbmaxstr >> cbmax;
		string command = "plotlens " + plotcommand + " file=" + filename + " " + extra_command + " ps=" + psstring + " pt=" + ptstring + " lw=" + lwstring + " fs=" + fsstring;
		if (fit_output_dir != ".") command += " dir=" + fit_output_dir;
		if (plot_title != "") command += " title='" + plot_title + "'";
		if (!show_plot_key) command += " key=-1";
		else if (plot_key_outside) command += " key=1";
		if (terminal==POSTSCRIPT) command += " term=postscript";
		else if (terminal==PDF) command += " term=pdf";
		if (show_colorbar==false) command += " nocb";
		if (colorbar_min != -1e30) command += " cbmin=" + cbmin;
		if (colorbar_max != 1e30) command += " cbmax=" + cbmax;
		if (plot_square_axes==true) command += " square";
		if (show_imgsrch_grid==true) command += " grid";
		system(command.c_str());
	}
}

void Lens::run_mkdist(bool copy_post_files, string posts_dirname, const int nbins_1d, const int nbins_2d, bool copy_subplot_only, bool resampled_posts, bool no2dposts)
{
	if (mpi_id==0) {
		string filename = fit_output_filename;
		if (resampled_posts) filename += ".new";
		if (posts_dirname == fit_output_dir) {
			cerr << "Error: directory for storing posteriors cannot be the same as the chains directory" << endl;
		} else {
			if (param_markers != "") {
				string marker_str = fit_output_dir + "/" + filename + ".markers";
				ofstream markerfile(marker_str.c_str());
				markerfile << param_markers << endl;
				markerfile.close();
			}
			if (post_title != "") {
				string title_str = fit_output_dir + "/" + filename + ".plot_title";
				ofstream titlefile(title_str.c_str());
				titlefile << post_title << endl;
				titlefile.close();
			}

			bool make_subplot = param_settings->subplot_params_defined();
			if (make_subplot) {
				string subplot_str = fit_output_dir + "/" + filename + ".subplot_params";
				ofstream subplotfile(subplot_str.c_str());
				int nparams_tot = param_settings->nparams + param_settings->n_dparams;
				for (int i=0; i < nparams_tot; i++) {
					string pname;
					bool pflag = param_settings->subplot_param_flag(i,pname);
					subplotfile << pname << " ";
					if (pflag) subplotfile << "1";
					else subplotfile << "0";
					subplotfile << endl;
				}
				subplotfile.close();
			}

			stringstream nbins1d_str, nbins2d_str;
			string nbins1d_string, nbins2d_string;
			nbins1d_str << nbins_1d;
			nbins2d_str << nbins_2d;
			nbins1d_str >> nbins1d_string;
			nbins2d_str >> nbins2d_string;
			string command = "cd " + fit_output_dir + "; ";
			command += "mkdist " + filename + " -n" + nbins1d_string;
			if (!no2dposts) command += " -N" + nbins2d_string; // plot histograms
			if ((make_subplot) and (!no2dposts)) command += " -s";
			if (post_title != "") command += " -t";
			if (param_markers != "") {
				command += " -m:" + filename + ".markers"; // add markers for plotting true values
				if (n_param_markers < 10000) {
					stringstream npmstr;
					string npmstring;
					npmstr << n_param_markers;
					npmstr >> npmstring;
					command += " -M" + npmstring;
				}
			}
			command += "; ";
			command += "mkdist " + filename + " -i -b -E2";
			if (param_markers != "") command += " -m:" + filename + ".markers -v"; // print true values to chain_info file
			command += " >" + filename + ".chain_info; "; // produce best-fit point and credible intervals
			command += "python " + filename + ".py; ";
			if (!no2dposts) command += "python " + filename + "_tri.py; "; // run python scripts to make PDFs
			if ((!no2dposts) and (make_subplot)) command += "python " + filename + "_subtri.py; "; // run python scripts to make PDF for subplot
			if (copy_post_files) {
				command += "if [ ! -d ../" + posts_dirname + " ]; then mkdir ../" + posts_dirname + "; fi; ";
				if ((!copy_subplot_only) or (!make_subplot)) command += "cp " + filename + ".pdf ../" + posts_dirname + "; ";
				if (!no2dposts) command += "cp " + filename + "_tri.pdf ../" + posts_dirname + "; ";
				if ((!no2dposts) and (make_subplot)) command += "cp " + filename + "_subtri.pdf ../" + posts_dirname + "; ";
				command += "cp *.chain_info ../" + posts_dirname + "; ";
			}
			command += "cd ..";
			system(command.c_str());
		}
	}
#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}


#include "qlens.h"
#include "params.h"
#include "profile.h"
#include "sbprofile.h"
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

#ifdef USE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

//#ifdef USE_EIGEN
//#include "Eigen/Cholesky"
//#include "Eigen/Dense"
//#include "Eigen/Sparse"
//#endif

#include <unistd.h>
using namespace std;

#define Complain(errmsg) do { if (mpi_id==0) cerr << "Error: " << errmsg << endl; if ((read_from_file) and (quit_after_error)) die(); goto next_line; } while (false) // the while(false) is a trick to make the macro syntax behave like a function
#define display_switch(setting) ((setting) ? "on" : "off")
#define set_switch(setting,setword) do { if ((setword)=="on") setting = true; else if ((setword)=="off") setting = false; else Complain("invalid argument; must specify 'on' or 'off'"); } while (false)
#define LENS_AXIS_DIR ((LensProfile::orient_major_axis_north==true) ? "y-axis" : "x-axis")

void QLens::process_commands(bool read_file)
{
	bool show_cc = true; // if true, plots critical curves along with image positions (via plotlens script)
	bool plot_srcplane = true;
	plot_key_outside = false;
	show_plot_key = true;
	show_colorbar = true;
	colorbar_min = -1e30;
	colorbar_max = 1e30;
	plot_square_axes = true;
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

	double chisq_pix_last = -1; // for investigating difference between subsquent pixel inversions (using 'sbmap invert')

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
						"source -- add an analytic extended source from the list of models ('help source' for list)\n"
						"ptsrc -- add a point source to the model\n"
						"pixsrc -- add a pixellated source to the model\n"
						"fit -- commands for lens model fitting (type 'help fit' for list of subcommands)\n"
						"imgdata -- commands for loading point image data ('help imgdata' for list of subcommands)\n"
						"wldata -- commands for loading weak lensing data ('help wldata' for list of subcommands)\n"
						"sbmap -- commands for surface brightness pixel maps ('help sbmap' for list of subcommands)\n"
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
						"raytracetab -- create file containing source points ray-traced from grid of image points\n"
						"findimgs -- find sets of images from source positions listed in a given input file\n"
						"plotimgs -- find and plot image positions from sources listed in a given input file\n"
						"replotimgs -- replot image positions previously found by 'plotimgs' command\n"
						"plotcrit -- plot critical curves and caustics to data files (or to an image)\n"
						"plotgrid -- plot recursive grid, if exists; plots corners of all grid cells\n"
						"plotlogkappa -- plot log(kappa) as a colormap along with isokappa contours\n"
						"plotlogmag -- plot log(magnification) as a colormap along with magnification contours\n"
						"plotkappa -- plot radial kappa profile for each lens model and the total kappa profile\n"
						"plotmass -- plot radial mass profile\n"
						"einstein -- find Einstein radius of a given lens model\n"
						"\n";
				} else if (words[1]=="settings") {
					cout << "Type 'help <category_name>' to display a list of settings in each category, and 'help <setting>'\n"
						"for a detailed description of each individual setting. Type 'settings' to display all current\n"
						"settings, '<category_name>' to display settings in a specific category, or enter a specific\n"
						"setting name to display its current value.\n"
						"\n"
						"SETTING CATEGORIES:    (type 'help <category_name>' for list of settings in each category)\n"
						"\n"
						"plot_settings -- preferences for graphical plotting\n"
						"imgsrch_settings -- settings for (point) image searching and grid configuration\n"
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
						"cbmin -- minimum value on color bar scale ('auto' by default)\n"
						"cbmax -- maximum value on color bar scale ('auto' by default)\n"
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
						"\n";
				} else if (words[1]=="sbmap_settings") {
					cout <<
						"\033[4mSurface brightness pixel map settings\033[0m\n"
						"fits_format -- load surface brightness maps from FITS files (if on) or text files (if off)\n"
						"img_npixels -- set number of pixels for images produced by 'sbmap plotimg'\n"
						"src_npixels -- set # of source grid pixels for plotting or inverting lensed pixel images\n"
						"srcgrid -- set source grid size and location for inverting or plotting lensed pixel images\n"
						"interpolation_method -- set method for interpolating over source pixels\n"
						"simulate_pixel_noise -- add simulated pixel noise to images produced by 'sbmap plotimg'\n"
						"psf_width -- width of Gaussian point spread function (PSF) along x- and y-axes\n"
						"psf_threshold -- threshold below which PSF is approximated as zero (sets pixel width of PSF)\n"
						"psf_mpi -- parallelize PSF convolution using MPI (on/off)\n"
						"sb_ellipticity_components -- for sbprofiles, use components e=1-q instead of (q,theta)\n"
						"\n"
						"\033[4mSource pixel reconstruction settings\033[0m\n"
						"inversion_method -- set method for lensing matrix inversion\n"
						"srcgrid_type -- source grid type (cartesian, adaptive_cartesian, adaptive)\n"
						"auto_src_npixels -- automatically determine # of source pixels from lens model/data (on/off)\n"
						"auto_srcgrid -- automatically choose source grid size/location from lens model/data (on/off)\n"
						"auto_shapelet_scale -- automatically choose shapelet center/scale from lens model/data (on/off)\n"
						"noise_threshold -- threshold (multiple of pixel noise) for automatic source pixel grid sizing\n"
						"data_pixel_size -- specify the pixel size to assume for pixel data files\n"
						"bg_pixel_noise -- pixel noise in data pixel images (loaded using 'sbmap loadimg')\n"
						"inversion_nthreads -- number of OpenMP threads to use specifically for matrix inversion\n"
						"pixel_fraction -- fraction of srcpixels/imgpixels used to determine number of source pixels\n"
						"regparam -- value of regularization parameter used for inverting lensed pixel images\n"
						"vary_regparam -- vary the regularization as a free parameter during a fit (on/off)\n"
						"outside_sb_prior -- impose penalty if model produces large surface brightness beyond pixel mask\n"
						"outside_sb_noise_threshold -- max s.b. allowed beyond mask by outside_sb_prior (times pixel noise)\n"
						"emask_n_neighbors -- expand mask by set # of pixel neighbors for outside_sb_prior, nimg_prior eval.\n"
						"nimg_prior -- impose penalty if # of images produced at max surface brightness < nimg_threshold\n"
						"nimg_threshold -- threshold on # of images near max surface brightness (used if nimg_prior is on)\n"
						"nimg_mag_threshold -- threshold on imgpixel magnification to be counted in findimg source pixel n_imgs\n"
						"nimg_sb_frac_threshold -- for nimg_prior, include only pixels brighter than threshold times max s.b.\n"
						"auxgrid_npixels -- # of pixels per side for auxiliary sourcegrid used for point images or nimg_prior\n"
						"include_emask_in_chisq -- include extended mask pixels in inversion and chi-square\n"
						"split_imgpixels -- if set to 'on', split image pixels and ray trace the subpixels, then average them\n"
						"imgpixel_nsplit -- specify number of splittings of each pixel (if 'split_imgpixels' is on)\n"
						"emask_nsplit -- specify number of pixel splittings for the extended mask (if 'split_imgpixels' is on)\n"
						"activate_unmapped_srcpixels -- when inverting, include srcpixels that don't map to any imgpixels\n"
						"exclude_srcpixels_outside_mask -- when inverting, exclude srcpixels that map beyond pixel mask\n"
						"remove_unmapped_subpixels -- when inverting, exclude *sub*pixels that don't map to any imgpixels\n"
						"sb_threshold -- minimum surface brightness to include when determining image centroids\n"
						"parallel_mumps -- run MUMPS matrix inversion using parallel analysis mode (if on)\n"
						"show_mumps_info -- show MUMPS information output after a lensing matrix inversion\n"
						"srcpixel_mag_threshold -- split srcpixels if magnification > threshold (srcgrid_type=adaptive_cartesian)\n"
						"\n";
				} else if (words[1]=="fit_settings") {
					cout <<
						"\033[4mChi-square function settings\033[0m\n"
						"imgplane_chisq -- use chi-square defined in image plane (if on) or source plane (if off)\n"
						"chisqmag -- use magnification in source plane chi-square function for image positions\n"
						"chisqpos -- include image positions in chi-square fit (on/off)\n"
						"chisqflux -- include flux information in chi-square fit (on/off)\n"
						"chisq_time_delays -- include time delay information in chi-square fit (on/off)\n"
						"chisq_weak_lensing -- include weak lensing information in chi-square fit (on/off)\n"
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
						"imgsrch_warnings -- set warning flags related to (point) image searching on/off\n"
						"integral_warnings -- set warning flags related to numerical integral convergence on/off\n"
						"show_wtime -- show time required for executing commands (e.g. mkgrid); requires OpenMP\n"
						"sci_notation -- display numbers in scientific notation (on/off)\n"
						"sim_err_pos -- random error in image positions, added when producing simulated image data\n"
						"sim_err_flux -- random error in image fluxes, added when producing simulated image data\n"
						"sim_err_td -- random error in time delays, added when producing simulated image data\n"
						"sim_err_shear -- random error in shear, added when producing simulated weak lensing data\n"
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
						"grid center <xc> <yc>\n"
						"grid -pxsize=#\n"
						"grid -use_data_pxsize\n\n"
						"Sets the grid size for plotting (or image searching) as (xmin,xmax) x (ymin,ymax), or centers\n"
						"the grid on <xc>, <yc> if 'grid center' is specified. If no arguments are given, simply outputs\n"
						"the present grid size. If argument '-pxsize=#' is used, qlens adopts a grid based on the given\n"
						"pixel size, centered on (0,0). The same procedure is done if '-use_data_pxsize' is used, but\n"
						"the pixel size is taken from the 'data_pixel_size' variable.\n\n";
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
							"lens vary ...\n"
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
							"update' for usage info). To change the vary flags for an existing lens, use 'lens vary\n"
							"<lens_number>' (see 'help lens vary' for usage info). Finally, it is possible to anchor the\n"
							"parameters of one lens to another lens; type 'help lens anchoring' for info on how to do this.\n\n"
							"Available lens models:   (type 'help lens <lensmodel>' for usage information)\n\n"
							"\033[4mElliptical mass models:\033[0m\n"  // ASCII codes are for underlining
							"sple -- softened power-law ellipsoid profile\n"
							"dpie -- dual pseudo-isothermal ellipsoid (smoothly truncated isothermal profile with core)\n"
							"nfw -- NFW model\n"
							"tnfw -- Truncated NFW model\n"
							"cnfw -- NFW model with core\n"
							"hern -- Hernquist model\n"
							"expdisk -- exponential disk\n"
							"sersic -- Sersic profile\n"
							"csersic -- Cored Sersic profile\n"
							"corecusp -- generalized profile with core, scale radius, inner & outer log-slope\n"
							"kspline -- splined kappa profile (generated from an input file)\n"
							"tab -- generic tabulated model, used to interpolate lensing properties in (r,phi) for above models\n"
							"qtab -- generic tabulated model used to interpolate lensing properties in (r,phi,q) for above models\n"
							"\n"
							"\033[4mNon-elliptical mass models:\033[0m\n"
							"ptmass -- point mass\n"
							"shear -- external shear\n"
							"sheet -- mass sheet\n"
							"deflection -- uniform deflection\n"
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
							"fit lens sple 5 1 0 0.8 30 0 0\n"
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
							"fit lens sple 2.5/anchor=0,0 1 0 0.8 30 0 0\n"
							"1 0 0 1 1 1 1\n\n"
							"The '2.5/anchor=0,0' enters the initial value in as 2.5, and since this is half of the parameter we\n"
							"are anchoring to (b=5 for lens 0), they will always keep this ratio. It is even possible to anchor a\n"
							"parameter to another parameter in the *same* lens, if you use the lens number that will be assigned\n"
							"to the lens you are creating. Again, the vary flag *must* be off for the parameter being anchored.\n\n"
							"To anchor the center of a lens to another lens, you can use 'anchor_center=...' as a shortcut. So\n"
							"in the previous example, if we wanted to also anchor the center of the lens to lens 0, we do\n\n"
							"fit lens sple 2.5/anchor=0,0 1 0 0.8 30 anchor_center=0\n"
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
					else if (words[2]=="vary")
						cout << "lens vary <lens_number> [none]\n"
									"lens vary [none]\n\n"
							"Change the parameter vary flags for a specific lens model that has already been created. After specifying\n"
							"the lens, on the next lens vary flags are entered just as you do when creating the lens model. Note that\n"
							"the number of vary flags must exactly match the number of parameters for the given lens (except that vary\n"
							"flags for the center coordinates can be omitted if the lens in question is anchored to another lens).\n"
							"If the optional argument 'none' is given after the lens number, all vary flags are set to '0' for that\n"
							"lens, and you will not be prompted to enter vary flags; if the lens number is omitted ('lens vary\n"
							"none'), vary flags are set to '0' for all lenses.\n\n";
					else if (words[2]=="savetab")
						cout << "lens savetab <lens_number> <filename>\n\n"
							"Save the interpolation tables from a 'tab' lens model to the specified file '<filename>.tab'. Note\n"
							"that <lens_number> must correspond to a 'tab' model in the list of existing lens models generated by\n"
							"the 'lens' command.\n";
					else if (words[2]=="sple")
						cout << "lens sple <b> <alpha> <s> <q/e> [theta] [x-center] [y-center]      (pmode=0)\n"
								  "lens sple <b> <gamma> <s> <q/e> [theta] [x-center] [y-center]      (pmode=1)\n\n"
							"where <b> is the mass parameter, <alpha> is the power law exponent (if s=0, this is the 2d log-slope, so\n"
							"alpha=1 for isothermal; if pmode=1, the 3d log-slope <gamma> is used, where gamma=alpha+1), <s> is the core\n"
							"radius, <q/e> is the axis ratio or ellipticity (depending on the ellipticity mode), and [theta] is the angle\n"
							"of rotation (counterclockwise, in degrees) about the center (defaults=0).\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="ptmass")
						cout << "lens ptmass <b> [x-center] [y-center]        (pmode=0)\n"
								  "lens ptmass <mtot> [x-center] [y-center]     (pmode=1)\n\n"
							"where <b> is the Einstein radius of the point mass (if pmode=1, the mass <mtot> is used instead). If center\n"
							"coordinates are not specified, the lens is centered at the origin by default.\n";
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
					else if (words[2]=="deflection")
						cout << "lens deflection <defx> <defy>\n\n"
							"where <defx> and <defy> are the components of the uniform deflection generated by an external perturber.\n"
							"Note that in this case there are no center coordinates and the lens center cannot be anchored to any\n"
							"other lens (or vice versa).\n";
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
					else if (words[2]=="dpie")
						cout << "lens dpie <b> <a> <s> <q/e> [theta] [x-center] [y-center]                 (pmode=0)\n"
								  "lens dpie <sigma0> <a_kpc> <s_kpc> <q/e> [theta] [x-center] [y-center]    (pmode=1)\n"
								  "lens dpie <mtot> <a_kpc> <s_kpc> <q/e> [theta] [x-center] [y-center]      (pmode=2)\n\n"
							"where <b> is the default mass parameter (it is the Einstein radius in the limit of large a, s=0); <a> is\n"
							"the tidal radius, <s> is the core radius, <q/e> is the axis ratio or ellipticity (depending on the ellip-\n"
							"ticity mode), and [theta] is the angle of rotation (counterclockwise, in degrees) about the center\n"
							"(defaults=0). For pmode=1, the mass parameter is the central velocity dispersion, with the truncation and\n"
							"core radii in kpc rather than arcsec; for pmode=2, the mass parameter <mtot> is the total mass.\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="nfw")
						cout << "lens nfw <ks> <rs> <q/e> [theta] [x-center] [y-center]             (pmode=0)\n"
								  "lens nfw <mvir> <c> <q/e> [theta] [x-center] [y-center] [-cmprior] (pmode=1)\n"
								  "lens nfw <mvir> <rs_kpc> <q/e> [theta] [x-center] [y-center]       (pmode=2)\n\n"
							"where <ks/mvir> is the mass parameter, <rs> is the scale radius (or <c>=concentration for pmode=1),\n"
							"<q/e> is the axis ratio or ellipticity (depending on the ellipticity mode), and [theta] is the angle\n"
							"of rotation (counterclockwise, in degrees) about the center (all defaults = 0).\n"
							"In pmode=1, the concentration can be set to the median c(M,z) (for which we use the relation from Dutton et al.\n"
							"2014) by entering 'cmed' for the parameter, or it can be set to some factor of cmed using 'fac*cmed'; e.g., to\n"
							"set it to twice the median concentration, enter '2*cmed'. The concentration will be anchored, so it is updated\n"
							"automatically if mvir or z are changed, so that it always takes the median value for given mvir,z. To set the\n"
							"initial concentration as a factor of cmed but *without* subsequent anchoring, add a '*' (e.g. '1*cmed*)'.\n"
							"You can also include a concentration-mass prior (using scatter in logc of 0.110 dex) with argument '-cmprior'.\n\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="tnfw")
						cout << "lens tnfw <ks> <rs> <rt> <q/e> [theta] [x-center] [y-center]            (pmode=0)\n"
								  "lens tnfw <mvir> <c> <rt_kpc> <q/e> [theta] [x-center] [y-center]       (pmode=1)\n"
								  "lens tnfw <mvir> <c> <tau> <q/e> [theta] [x-center] [y-center]          (pmode=2)\n"
								  "lens tnfw <mvir> <rs_kpc> <rt_kpc> <q/e> [theta] [x-center] [y-center]  (pmode=3)\n"
								  "lens tnfw <mvir> <rs_kpc> <tau_s> <q/e> [theta] [x-center] [y-center]   (pmode=4)\n\n"
							"Truncated NFW profile from Baltz et al. (2008), which is produced by multiplying the NFW density\n"
							"profile by a factor (1+(r/rt)^2)^-2, where rt acts as the truncation/tidal radius. Here,\n"
							"<ks/mvir> is the mass parameter, <rs> is the scale radius (or <c>=concentration for pmode=1,2),\n"
							"<rt> is the tidal radius (or <tau>=rt/r200 in pmode=2, or <tau_s>=rt/rs in pmode=4), <q/e> is the\n"
							"axis ratio or ellipticity (depending on the ellipticity mode), and [theta] is the angle of rotation\n"
							"(counterclockwise, in degrees) about the center (all defaults = 0).\n"
							"In pmode 1 or 2, the concentration can be set to the median c(M,z) (for which we use the relation from Dutton et al.\n"
							"2014) by entering 'cmed' for the parameter, or it can be set to some factor of cmed using 'fac*cmed'; e.g., to\n"
							"set it to twice the median concentration, enter '2*cmed'. The concentration will be anchored, so it is updated\n"
							"automatically if mvir or z are changed, so that it always takes the median value for given mvir,z. To set the\n"
							"initial concentration as a factor of cmed but *without* subsequent anchoring, add a '*' (e.g. '1*cmed*)'.\n\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="cnfw")
						cout << "lens cnfw <ks> <rs> <rc> <q/e> [theta] [x-center] [y-center]          (pmode=0)\n"
								  "lens cnfw <mvir> <c> <beta> <q/e> [theta] [x-center] [y-center]       (pmode=1)\n"
								  "lens cnfw <mvir> <rs_kpc> <beta> <q/e> [theta] [x-center] [y-center]  (pmode=2)\n"
								  "lens cnfw <mvir> <c> <rc_kpc> <q/e> [theta] [x-center] [y-center]     (pmode=3)\n\n"
							"Cored NFW profile with 3d density given by rho = rho_s/((r+rc)*(r+rs)^2) Here, <ks/mvir> is the mass\n"
							"parameter, <rs> is the scale radius (or <c>=concentration for pmode=1), <rc> is the core radius (or\n"
							"<beta> = rc/rs for pmodes 1,2), <q/e> is the axis ratio or ellipticity (depending on ellipticity\n"
							"mode), and [theta] is the angle of rotation (counterclockwise, in degrees) about the center.\n"
							"In pmode 1 or 3, the concentration can be set to the median c(M,z) (for which we use the relation from Dutton et al.\n"
							"2014) by entering 'cmed' for the parameter, or it can be set to some factor of cmed using 'fac*cmed'; e.g., to\n"
							"set it to twice the median concentration, enter '2*cmed'. The concentration will be anchored, so it is updated\n"
							"automatically if mvir or z are changed, so that it always takes the median value for given mvir,z. To set the\n"
							"initial concentration as a factor of cmed but *without* subsequent anchoring, add a '*' (e.g. '1*cmed*)'.\n\n"
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
						cout << "lens corecusp <k0> <gamma> <n> <a> <s> <q/e> [theta] [x-center] [y-center]    (pmode=0)\n"
									"lens corecusp <R_e> <gamma> <n> <a> <s> <q/e> [theta] [x-center] [y-center]    (pmode=1)\n\n"
							"This is a cored version of the halo model of Munoz et al. (2001), where <a> is the scale/tidal radius,\n"
							"<k0> = 2*pi*rho_0*a/sigma_crit, <s> is the core radius, and <gamma>/<n> are the inner/outer (3D) log-\n"
							"slopes respectively. In pmode=1, the Einstein radius R_e is used instead of k0 (although note that for low\n"
							"enough k0 values you get R_e=0, i.e. it is not strongly lensing, so for weak perturbations k0 should be\n"
							"used). (Incidentally the pseudo-Jaffe profile corresponds to gamma=2, n=4 and b=k0*a/(1-(s/a)^2).)\n"
							"As with the other models, <q/e> is the axis ratio or ellipticity (depending on the ellipticity mode),\n"
							"and [theta] is the angle of rotation (counter-clockwise, in degrees) about the center (all defaults = 0).\n"
							"Note that for theta=0, the major axis of the lens is along the " << LENS_AXIS_DIR << " (the direction of the\n"
							"major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="sersic")
						cout << "lens sersic <kappa_e> <R_eff> <n> <q/e> [theta] [x-center] [y-center]   (pmode=0)\n"
								  "lens sersic <Mstar> <R_eff> <n> <q/e> [theta] [x-center] [y-center]     (pmode=1)\n\n"
							"The sersic profile is defined by kappa = kappa_e * exp(-b*((R/R_eff)^(1/n)-1)), where kappa_e is the\n"
							"kappa value at the effective (half-mass) radius R_eff, and b is a factor automatically determined from\n"
							"the value for n to ensure that R_eff contains half the total mass (from Cardone et al. 2003). Here,\n"
							"[theta] is the angle of rotation (counterclockwise, in degrees) about the center (defaults=0). Note that\n"
							"for theta=0, the major axis of the source is along the " << LENS_AXIS_DIR << " (the direction of the major\n"
							"axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="csersic")
						cout << "lens csersic <kappa_e> <R_eff> <n> <rc> <q/e> [theta] [x-center] [y-center]\n\n"
							"The cored sersic profile is defined by kappa = kappa_e * exp(-b*((sqrt(R^2+rc^2)/R_eff)^(1/n)-1)), where\n"
							"kappa_e is the kappa value at the effective (half-mass) radius R_eff, rc is the core size, and b is a factor\n"
							"automatically determined from the value for n to ensure that R_eff contains half the total mass in the limit\n"
							"of zero core size (from Cardone et al. 2003).\n"
							"Here, [theta] is the angle of rotation (counterclockwise, in degrees) about the center (defaults=0). Note that\n"
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
							"fit plotimg [src=#] [-nosrc]\n"
							"fit plotsrc [src=#]\n"
							"fit plotshear\n"
							"fit data_imginfo       (NEED TO WRITE HELP DOCS FOR THIS)\n"
							"fit method <method>\n"
							"fit label <label>\n"
							"fit output_dir <dirname>\n"
							"fit use_bestfit\n"
							"fit save_bestfit\n"
							"fit load_bestfit ...\n"
							"fit add_chain_dparams\n"
							"fit adopt_chain_point ...\n"
							"fit adopt_point_prange ...\n"
							"fit mkposts ...\n"
							"fit plimits ...\n"
							"fit stepsizes ...\n"
							"fit dparams ...\n"
							"fit params ...\n"
							"fit priors ...\n"
							"fit transform ...\n"
							"fit regularization <method>\n"
							"fit output_img_chivals (WRITE HELP DOCS)\n"
							"fit output_wl_chivals  (WRITE HELP DOCS)\n\n"
							"Commands needed to fit lens models. If the 'fit' command is entered with no arguments, the\n"
							"current fit model (including lens and source) is listed along with the free parameters.\n"
							"For help with the specific fit commands, type 'help fit <command>'. To run the chi-square fit,\n"
							"routine, use the 'fit run' command.\n\n";
					else if (words[2]=="lens") 
						cout << "fit lens ...\n\n"
							"Identical to the 'lens' command, except that after entering the lens model, flags\n"
							"must be entered for each parameter (either 0 or 1) to specify whether it will be\n"
							"varied or not. For example, consider the following command:\n\n"
							"fit lens sple 10 1 0.5 0.9 0 0 0\n"
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
							"fit lens sple 10 1 0.5 0.9 0 anchor_center=0\n"
							"1 0 0 1 1\n\n"
							"This is similar to the above example, except that this lens is anchored to lens 0\n"
							"so that their centers will always coincide when the parameters are varied.\n\n"
							"Since it is common to anchor external shear to a lens model, this can be done in\n" <<
							((Shear::use_shear_component_params) ?
								"one line by adding 'shear=<shear_1> <shear_2>' to the end of the line. For example,\n"
								"to add external shear with shear_1=0.1 and shear_2=0.05, one can enter:\n\n"
								"fit lens sple 10 1 0 0.9 0 0.3 0.5 shear=0.1 0.05\n" :
								"one line by adding 'shear=<shear> <theta>' to the end of the line. For example,\n"
								"to add external shear with shear=0.1 and theta=30 degrees, one can enter:\n\n"
								"fit lens sple 10 1 0 0.9 0 0.3 0.5 shear=0.1 30\n") <<
							"1 0 0 1 1 1 1 1 1     # vary flags for the sple model + external shear parameters\n\n"
							"You can also vary the lens redshift by adding the argument 'varyz=1' at the end of the line\n"
							"containing the vary flags. Finally, it is possible to anchor a specific parameter to another\n"
							"parameter in another lens model (or within the same lens model). For more info on this, type\n"
							"'help lens anchoring'.\n";
					else if (words[2]=="sourcept")
						cout << "fit sourcept\n"
							"fit sourcept update                            (change all source point coords)\n"
							"fit sourcept <sourcept_num>                 (change specific source point coords)\n"
							"fit sourcept <x0> <y0>  [-add]              (shortcut if only one source point)\n"
							"fit sourcept auto                           (adopts best fit from source plane chi-sq)\n\n"
							"Specify (or list) the coordinates for the source points, one for each set of images loaded as\n"
							"data (via the 'imgdata' command). If only one source point is being used, these can be given as\n"
							"arguments on the same line (e.g. 'fit sourcept 1 2.5'); for more than one source point, the\n"
							"coordinates must be entered on separate lines using 'fit sourcept set'. For example, if two image sets are loaded, then\n"
							"to specify the initial source data points one would type:\n\n"
							"fit sourcept set\n"
							"1 2.5\n"
							"3.2 -1\n\n"
							"This sets the initial source point coordinates to (1,2.5) and (3.2,-1). To change a specific source\n"
							"point, enter 'fit sourcept #', where # is the index the source point is listed under (which you can\n"
							"see using the 'fit' command), then enter the coordinates on the following line. If the fit method\n"
							"being used requires lower and upper limits on parameters (e.g. for nested sampling or MCMC),\n"
							"the user will be prompted to give lower and upper limits on x and then y for each source point\n"
							"(the format for entering limits is similar to 'fit lens'; see 'help fit lens' for description).\n"
							"Finally, if modeling pixel images, you can add a source point using '-add' as last argument.\n\n";
					else if (words[2]=="source_mode")
						cout << "fit source_mode <mode>\n\n"
							"Specify the type of source/lens fitting to use. If no argument is given, prints the current source\n"
							"mode. For 'ptsource', the data is in the form of point images, while all the others are extended\n"
							"source models. Options are:\n\n"
							"ptsource -- the data is in the form of point images (set using 'imgdata' command) and one\n"
							"            or more point sources are used as fit parameters.\n"
							"cartesian -- the source galaxy is modeled with a Cartesian pixel grid, whose pixels can be split\n"
							"             into subpixels in regions of high magnification if 'adaptive_subgrid' is on.\n"
							"delaunay -- the source galaxy is modeled with a grid produced from ray-traced image pixels/sub-pixels,\n"
							"            from which a Delaunay triangulation is used to find the lensed surface brightness.\n"
							"sbprofile -- an analytic model is used for the source galaxy's surface brightness profile (via the\n"
							"             'source' commands) rather than a pixellated source.\n"
							"shapelet -- shapelet basis functions are used to represent the source galaxy, whose amplitudes\n"
							"            are found by a linear inversion.\n\n";
					else if (words[2]=="findimg")
						cout << "fit findimg [sourcept_num]\n\n"
							"Find the images produced by the current lens model and source point(s) and output their positions,\n"
							"magnifications, and (optional) time delays, using the same output format as the 'findimg' command.\n"
							"if no argument is given, image data is output for all the source points being modeled; otherwise,\n"
							"sourcept_num corresponds to the number assigned to a given source point which is listed by the 'fit'\n"
							"command (to set the initial values for the model source points, use the 'fit sourcept' command).\n";
					else if (words[2]=="plotimg")
						cout << "fit plotimg [src=#] [-nosrc]\n"
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
							"redshift corresponding to image set 3. If filenames are specified, the image data/fits are\n"
							"plotted to a file (the file type depends on the 'terminal' setting, either 'text', 'pdf' or 'ps').\n";
					else if (words[2]=="plotshear")
						cout << "fit plotshear [filename]\n\n"
							"Plot the reduced shear data (listed by the 'wldata' command) as well as the reduced shear produced\n"
							"by the model at the locations of the shear data. The length of each line segment is proportional to\n"
							"the magnitude of the reduced shear at that location. If [filename] is specified, the shear data/fits\n"
							"are plotted to a file (file type depends on the 'terminal' setting, either 'text', 'pdf' or 'ps').\n";
					else if (words[2]=="run")
						cout << "fit run [-resume/process/noerrs]\n\n"
							"Run the selected model fit routine, which can either be a minimization (e.g. Powell's method)\n"
							"or a Monte Carlo sampler (e.g. MCMC or nested sampling method) depending on the fit method that\n"
							"has been selected. If using MultiNest or PolyChord, you can add the argument '-resume' to continue\n"
							"a previous run that had been interrupted; if a run has already been finished previously, you can\n"
							"add the argument '-process' to process the chains from the previous run. If doing an optimization,\n"
							"you can add the argument '-noerrs' to skip calculating Fisher matrix errors (this can also be\n"
							"toggled using the 'find_errors' variable). For more info about the output produced by these\n"
							"methods, type 'help fit method <method>'.\n";
					else if (words[2]=="chisq")
						cout << "fit chisq [-diag] [-wtime]\n\n"
							"Output the chi-square value for the current model and data. If using more than one chi-square\n"
							"component (e.g. fluxes and time delays), each chi-square will be printed separately in addition\n"
							"to the total chi-square value. If argument '-diag' is added and fit source mode is 'ptsource',\n"
							"diagnostic information is printed for image plane chi-square; for image plane chi-square, the\n"
							"diagnostics include the chi-square contribution from each data image, the model image it matches\n"
							"to, as well as extra model images that aren't matched to any data image. If '-wtime' (or '-wt')\n"
							"is added, wall time information is shown regardless of whether 'show_wtime' is on or not.\n";
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
						else if ((words[3]=="nest") or (words[3]=="multinest") or (words[3]=="polychord"))
							cout << "fit method nest\n"
								"fit method multinest\n"
								"fit method polychord\n\n"
								"The nested sampling algorithms output the Bayesian evidence, as well as points that sample the\n"
								"parameter space, which can then be marginalized by binning the points in the parameter(s) of\n"
								"interest, weighting them according to the supplied weights. The resulting points and weights are\n"
								"output to the file '<label>', while the parameters that maximize the space are output to the file\n"
								"<label>.max', where the label is set by the 'fit label' command. The number of initial 'active'\n"
								"points is set by n_livepts. After a run finishes, posterior histograms can be generated using the\n"
								"'mkdist' tool included with qlens; alternatively, they can be generated from within qlens using\n"
								"the 'fit mkposts' command (which runs the mkdist tool).\n";
						else if (words[3]=="twalk")
							cout << "fit method twalk\n\n"
								"T-Walk is a Markov Chain Monte Carlo (MCMC) algorithm that samples the parameter space using a\n"
								"Metropolis-Hastings step and outputs the resulting chain(s) of points, which can then be marginalized\n"
								"by binning in the parameter(s) of interest. Data points are output to the file '<label>', where the\n"
								"label is set by the 'fit label' command. The algorithm uses the Gelman-Rubin R-statistic to determine\n"
								"convergence and terminates after R reaches the value set by mcmctol. After a run finishes, posterior\n"
								"histograms can be generated using the 'mkdist' tool included with qlens; alternatively, they can be\n"
								"generated from within qlens using the 'fit mkposts' command (which runs the mkdist tool).\n";
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
							"chain will still be included. The new chain data will be safed to the file '<fit_label>.new'.\n";
					else if (words[2]=="adopt_chain_point")
						cout << "fit adopt_chain_point <line_number>\n\n"
							"Loads a chain that has already been created, and adopts the model parameters from the point in the\n"
							"on the specified line number of the chain file.\n";
					else if (words[2]=="adopt_point_prange")
						cout << "fit adopt_point_prange <param_number> <minval> <maxval>\n\n"
							"Loads a chain that has already been created, and adopts the model parameters from the point with the\n"
							"lowest chi-square value within the range (minval,maxval) of the parameter specified. The parameter\n"
							"numbers can be listed using the command 'fit priors'. (NOTE: if you want to do this with a derived\n"
							"parameter, enter the number of parameters plus the derived parameter index. e.g. if you have 16\n"
							"parameters and you want derived parameter #2, then do 'fit adopt_point_prange 19 ...'.\n";
					else if (words[2]=="mkposts")
						cout << "fit mkposts <dirname> [-n#] [-N#] [-no2d] [-nohist] [-subonly]\n\n"
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
							"fit stepsizes from_chain <factor>\n\n"
							"Define initial stepsizes for the chi-square minimization methods (simplex or powell). Values are\n"
							"automatically chosen for the stepsizes by default, but can be customized. Type 'fit stepsizes' to\n"
							"see the currest list of fit parameters. For example, 'fit stepsizes 5 0.3' sets the initial stepsize\n"
							"for parameter 5 to 0.3. You can also scale all the stepsizes by a certain factor, e.g. 'fit\n"
							"stepsize scale 5' multiplies all stepsizes by 5. If you have an MCMC/nested sampling chain, then the\n"
							"argument 'from_chain' will use the 95\% credible intervals in each parameter (divided by two; this\n"
							"corresponds to the 2*sigma values for a Gaussian posterior) to set the stepsizes, times the scale\n"
							"factor [factor]. To reset all stepsizes to their automatic values, type 'fit stepsize reset'.\n";
					else if (words[2]=="params")
						cout << "fit params [-nosci]\n"
							"fit params update <param_num/name> <param_val>\n"
							"fit params update <name>=<val> ...\n"
							"fit params update_all <val1> <val2> ...\n"
							"fit params rename <param_num/name> <new_name>\n\n"
							"If no additional argument is given, outputs the current parameter values for all parameters being varied\n"
							"(to not use scientific notation, add '-nosci' argument; this works even if 'sci_notation' is turned off).\n"
							"To update parameters, you can specify a parameter number/name followed by its new value, or else you can\n"
							"update multiple parameters at once using arguments of the form '<name>=<val>', e.g. 'b=1.5 q=0.7' etc.\n"
							"You can also update parameters in order using 'update_all' (updates will stop at the last parameter entered).\n"
							"To manually rename a parameter, use 'rename <param_num/name> <new_name>'. Note that for the second argument\n"
							"you can put the parameter number as displayed in the list, or you can enter the parameter name (the latter\n"
							"approach is less bug-prone since parameter numbers may change if the model is changed).\n\n";
					else if (words[2]=="priors")
						cout << "fit priors [-nosci]\n"
							"fit priors <param_num/name> <prior_type> [prior_params]\n"
							"fit priors <param_num/name> limits <lower_limit> <upper_limit>\n\n"
							"fit priors limits [param_num/name]\n\n"
							"Define prior probability distributions in each fit parameter, which are used by the T-Walk and nested\n"
							"sampling routines. Type 'fit priors' to see current list of fit parameters and corresponding priors/limits\n"
							"(to not use scientific notation, add '-nosci' argument; this works even if 'sci_notation' is turned off).\n"
							"To define a prior, you can enter the parameter number as displayed in the list, or you can enter the\n"
							"parameter name (the latter approach is less bug-prone since numbers may change if the model is changed).\n"
							"To just change the lower/upper prior limits for that parameter, use 'range <low> <high>'; this manually\n"
							"overrides the prior range, which is especially useful if a transformation is being used via 'fit transform'.\n"
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
							"fit transform <param_num/name> <transform_type> [transform_params] ... [-include_jac] [-name=...]\n\n"
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
							"still have a uniform prior in the *original* parameter, add the argument '-include_jac'.\n"
							"Finally, to rename the transformed parameter, add the '-name=<name>' argument.\n\n"
							"Available transformation types:\n\n"
							"none     -- no transformation (the default for all parameters)\n"
							"log      -- transform to log(p) using the base 10 logarithm\n"
							"linear   -- transform to L{p} = A*p + b. The two parameter arguments are <A> and <b>, so e.g. 'fit transform\n"
							"              linear 2 5' will transform p --> 2*p + 5.\n"
							"ratio   -- transform p1 --> p1/p2, that is, to the ratio of two parameters. There is one argument, which is\n"
							"              the parameter number/name for p2. e.g., 'fit transform 3 ratio 4' will transform to the ratio of\n"
							"              parameter 3 over parameter 4.\n"
							"gaussian -- transformation whose Jacobian is Gaussian, and thus is equivalent to having a Gaussian prior in\n"
							"              the original parameter. There are two arguments, mean value <mean> and dispersion <sig>\n"
							"              (e.g., 'fit transform # gaussian 0.2 0.5' will be Gaussian with mean 0.2 and dispersion 0.5)\n\n";
					else if (words[2]=="dparams")
						cout << "fit dparams\n"
							"fit dparams add <param_type> [param_args...] [lens#] [-kpc] (lens#, param_arg are optional for some dparams)\n"
							"fit dparams rename <param_type> <text_name> <latex_name>\n"
							"fit dparams clear [param_num]\n\n"
							"Define derived parameters whose values will be output along with the primary parameters after running\n"
							"nested sampling or T-Walk. Type 'fit dparams' to list all the derived parameters that have been defined,\n"
							"'fit dparams clear' to remove one or all of the derived parameters from the list, and 'fit dparams add'\n"
							"to add a new derived parameter. All derived parameters are defined by a type and (usually) an argument;\n"
							"in the list below, the parameter argument is denoted in brackets <...>, otherwise no argument is required.\n"
							"If the '-kpc' argument is added, distances entered will be in kpc rather than arcseconds.\n"
							"The available derived parameter types are:  ('*' means if lens_number is omitted, all lenses are included)\n\n"
							"raw_chisq -- The chi-square value (not including priors; no arguments required)\n"
							"kappa_r  -- *The kappa at radius <r>, averaged over all angles (where <r> in arcseconds)\n"
							"dkappa_r -- *The derivative of kappa at radius <r>, averaged over all angles (where <r> is in arcseconds)\n"
							"lambda_r  -- 1-<kappa(r)> where <kappa(r)> is averaged over all angles (where <r> in arcseconds)\n"
							"mass2d_r -- The projected mass enclosed within elliptical radius <r> (in arcsec) for a specific lens [lens#]\n"
							"mass3d_r -- The 3d mass enclosed within elliptical radius <r> (in arcsec) for a specific lens [lens#]\n"
							"re_zsrc -- The (spherically averaged) Einstein radius of lens [lens#] for a source redshift <zsrc>\n"
							"xi -- The xi parameter (from Kochanek 2020) for a source redshift <zsrc> (optional lens# can be specified)\n"
							"mass_re -- The projected mass enclosed within Einstein radius of lens [lens#] for a source redshift <zsrc>\n"
							"kappa_re -- kappa(R_ein) of primary lens (+lenses that are co-centered with primary), averaged over all angles\n"
							"logslope -- The average log-slope of kappa between <r1> and <r2> (in arcsec) for a specific lens [lens#]\n"
							"lensparam -- The value of parameter <paramnum> for lens [lens#] using given parameter mode <pmode>\n"
							"r_perturb -- The critical curve perturbation radius of perturbing lens [lens#]; assumes lens 0 is primary\n"
							"                  (See Minor et al. 2017 for definition of perturbation radius for subhalos)\n"
							"mass_perturb -- The projected mass enclosed within r_perturb (see above) of perturbing lens [lens#]\n"
							"sigma_perturb -- The average projected density within r_perturb (see above) of perturbing lens [lens#]\n"
							"r_perturb_rel -- Same as r_perturb, except it's subtracted from the unperturbed critical curve location\n"
							"qs -- axis ratio derived from source pixel covariance matrix; [param1] gives # of points sampled\n" 
							"phi_s -- orientation derived from source pixel covariance matrix; [param1] gives # of points sampled\n" 
							"sig_s -- source dispersion from source pixel covariance matrix; [param1] gives # of points sampled\n" 
							"xavg_s -- centroid x-coordinate derived from adaptive source grid; [param1] gives # of points sampled\n" 
							"yavg_s -- centroid y-coordinate derived from adaptive source grid; [param1] gives # of points sampled\n" 
							"\n";
					else if (words[2]=="regularization")
						cout << "fit regularization <method>\n\n"
							"Specify the type of regularization that will be used when fitting to a surface brightness pixel\n"
							"map using source pixel reconstruction. Regularization imposes smoothness on the reconstructed\n"
							"source and is vital if significant noise is present in the image. If no argument is given,\n"
							"prints the current regularization method. Available methods are:\n\n"
							"none -- no regularization\n"
							"norm -- regularization matrix built from (squared) surface brightness of each pixel\n"
							"gradient -- regularization matrix built from the derivative between neighboring pixels\n"
							"curvature -- regularization matrix built from the curvature between neighboring pixels (default)\n"
							"sgradient -- gradient regularization using natural neighbor interpolation (Delaunay only)\n"
							"scurvature -- curvature regularization using natural neighbor interpolationl (Delaunay only)\n"
							"exp_kernel -- regularization using exponential covariance kernel (Delaunay only)\n"
							"sqexp_kernel -- regularization using squared exponential covariance kernel (Delaunay only)\n"
							"matern_kernel -- regularization using Matern covariance kernel (Delaunay only)\n\n";
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
							//"imgdata add_from_centroid ...\n"
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
						//else if (words[2]=="add_from_centroid")
							//cout << "imgdata add_from_centroid <xmin> <xmax> <ymin> <ymax>\n\n"
								//"Finds the centroid of the image surface brightness map (if loaded), within the range of\n"
								//"coordinates supplied by the arguments (xmin,xmax,ymin,ymax). The resulting centroid is then\n"
								//"added as a point image. A minimum surface brightness threshold for inclusion in the calculation\n"
								//"may be specified by setting the 'sb_threshold' variable, which is zero by default. The total\n"
								//"flux is also calculated (note however that the image is large enough, the total flux may be\n"
								//"unreliable as an approximation to the flux at the centroid point). The position error is simply\n"
								//"the pixel width, whereas the flux error is determined from the pixel noise (bg_pixel_noise).\n";
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
							"wldata add_random <nsrc> <xmin> <xmax> <ymin> <ymax> <zsrc_min> <zsrc_max> [rmin]\n"
							"wldata write <filename>\n"
							"wldata plot [filename]\n"
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
						else if (words[2]=="plot")
							cout << "wldata plot <filename>\n\n"
								"Plots the reduced shear from the current data set, where the lengths of the line segements are\n"
								"proportional to the magnitude of the reduced shear.\n";
						else if (words[2]=="add")
							cout << "wldata add <x-coord> <y-coord> [z=#]\n\n"
								"Add simulated weak lensing source defined by the source point (<x-coord>,<y-coord>) lensed by the\n"
								"current lens configuration. The redshift source can be entered via the optional argument 'z=#'; if\n"
								"this argument is not entered, the source has a default redshift given by the value of 'zsrc'.\n"
								"The errors in the reduced shear are given by the variable sim_err_shear, so that a corresponding\n"
								"random Gaussian error is added to the reduced shear components g1 and g2.\n";
						else if (words[2]=="add_random")
							cout << "wldata add_random <nsrc> <xmin> <xmax> <ymin> <ymax> <zsrc_min> <zsrc_max> [rmin]\n\n"
								"Add simulated weak lensing data from sources with positions and redshifts drawn from uniform\n"
								"random deviates over the defined grid and redshift range. In order to exclude the strong lensing\n"
								"regime, one may optionally specify [rmin] to exclude source points within the specified radius.\n";
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
							"sbmap mksrc\n"
							"sbmap mkplotsrc [...]\n"						// WRITE HELP DOCS FOR THIS COMMAND
							"sbmap savesrc [output_file]\n"
							"sbmap plotsrc [output_file] [...]\n"  // UPDATE HELP DOCS FOR THIS COMMAND
							"sbmap loadsrc <source_file>\n"
							"sbmap loadimg <image_data_file>\n"
							"sbmap saveimg <image_data_file> [...]\n" // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap loadpsf <psf_file>\n"
							"sbmap savepsf <psf_file>\n"
							"sbmap plotpsf\n"
							"sbmap load_noisemap <noisemap_file>\n"
							"sbmap save_noisemap <noisemap_file> [...]\n" // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap unloadpsf\n"                 // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap mkpsf [-spline]\n"                 // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap spline_psf\n"                 // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap plotdata\n"
							"sbmap invert\n"
							"sbmap loadmask <mask_file> [-add]\n"      // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap savemask <mask_file>\n"      // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap invert_mask\n"      // WRITE HELP DOCS FOR THIS COMMAND
							"sbmap set_all_pixels\n"
							"sbmap unset_all_pixels\n"
							"sbmap set_data_annulus [...]\n"
							"sbmap set_data_window [...]\n"
							"sbmap unset_data_annulus [...]\n"
							"sbmap unset_data_window [...]\n"
							"sbmap unset_low_sn_pixels [...]\n"
							"sbmap trim_mask_windows [...]\n"
							"sbmap set_neighbor_pixels [...]\n"
							"sbmap set_posrg_pixels [...]\n"
							"sbmap activate_partner_imgpixels\n"
							"sbmap find_noise [...]\n\n"
							"Commands for loading, simulating, plotting and inverting surface brightness pixel maps. For\n"
							"help on individual subcommands, type 'help sbmap <command>'. If 'sbmap' is typed with no\n"
							"arguments, shows the dimensions of current image/source surface brightness maps, if loaded.\n";
					} else {
						if (words[2]=="loadimg")
							cout << "sbmap loadimg <image_filename> [-showhead] [-hdu=#]\n\n"
								"Load an image surface brightness map from file <image_file>. If 'fits_format' is off, then\n"
								"loads a text file with name '<image_filename>.dat' where pixel values are arranged in matrix\n"
								"form. The x-values are loaded in from file '<image_filename>.x' and likewise for y-values.\n"
								"If 'fits_format' is on, then the filename must be in FITS format, and the size of each\n"
								"pixel must be specified beforehand in the variable 'data_pixel_size'. If no pixel size\n"
								"has been specified, than the grid dimensions specified by the 'grid' command are used to\n"
								"set the pixel size. After loading the pixel image, the number of image pixels for plotting\n"
								"(set by 'img_npixels') is automatically set to be identical to those of the data image.\n"
								"To jump to a different header data unit (HDU), use '-hdu=#' where # is the HDU index.\n"
								"To show the FITS header for the given HDU index, add the argument '-showhead'.\n";
						else if (words[2]=="load_noisemap")
							cout << "sbmap load_noisemap <noisemap_filename> [-showhead] [-hdu=#]\n\n"
								"Load a noise map from FITS file <noisemap_file>. To jump to a different FITS header data unit\n"
								"(HDU), use '-hdu=#' where # is the HDU index. To show the FITS header for the given HDU index,\n"
								"add the argument '-showhead'.\n";
						else if (words[2]=="loadpsf")
							cout << "sbmap loadpsf <psf_filename> [-showhead] [-hdu=#] [-spline] [-supersampled]\n\n"
								"Load PSF map from FITS file <psf_file>. By default, the PSF map will be truncated according to\n"
								"the 'psf_threshold' setting, such that pixels whose brightness falls below the given threshold\n"
								"are excluded. To jump to a different FITS header data unit (HDU), use '-hdu=#' where # is the\n"
								"HDU index. To show the FITS header for the given HDU index, add the argument '-showhead'.\n";
						else if (words[2]=="loadsrc")
							cout << "sbmap loadsrc <source_filename>\n\n"
								"Load a source surface brightness pixel map that was previously saved in qlens (using 'sbmap\n"
								"savesrc').\n";
						else if (words[2]=="plotimg")
							cout << "sbmap plotimg [-...]\n"
								"sbmap plotimg [-...] [image_file]\n"
								"sbmap plotimg [-...] [source_file] [image_file]\n\n"
								"Plot a lensed pixel image from a pixellated source surface brightness distribution under the\n"
								"assumed lens model, plus any additional point sources that are included in the model. If the\n"
								"terminal is set to 'text', the surface brightness values are output in matrix form in the file\n"
								"'<image_file>.dat', whereas the x-values are plotted as '<image_file>.x', and likewise for the\n"
								"y-values (if no arguments are given, the file labels default to 'src_pixel' and 'img_pixel' for\n"
								"the source and image pixel maps, respectively, and these are plotted to the screen in a separate\n"
								"window; if only one file argument is given, the source pixel map is not plotted). The source\n"
								"file is plotted in the same format. The critical curves and caustics are also plotted, unless\n"
								"show_cc is set to 'off'. If adding simulated pixel noise, (via 'simulate_pixel_noise'), then random\n"
								"noise is added to each pixel with dispersion set by 'bg_pixel_noise' or from a noise map.\n\n"
								"Optional arguments:\n"
								"  [-fits] plots to FITS files; filename(s) must be specified with this option\n"
								"  [-res] plots residual image by subtracting from data (or use '-resns' to also omit source plot)\n"
								"  [-nres] plots normalized residual image by subtracting from the data image and dividing by pixel\n"
								"             sigma values given by noise map (or 'bg_pixel_noise' if no noise map)\n"
								"  [-fg] plots only the 'unlensed' sources (may combine with '-residual' to subtract foreground)\n"
								"  [-nomask] plot image (or residuals) using all pixels, including those outside the chosen mask\n"
								"  [-emask/fgmask] plot image (or residuals) using extended/foreground mask\n"
								"  [-pnoise] add pixel noise to the plot (if not using noise map, can specify 'pnoise=...')\n"
								"  [-noptsrc] do not include the point source(s) when plotting the lensed images\n"
								"  [-nocc] omit critical curves and caustics (equivalent having 'show_cc' off)\n"
								"  [-replot] plots image that was previously found and plotted by the 'sbmap plotimg' command.\n"
								"     This allows one to tweak plot parameters (range, show_cc etc.) without having to calculate\n"
								"     the lensed pixel images again.\n\n"
								"OPTIONAL: arguments to 'sbmap plotimg' can be followed with terms in brackets [#:#][#:#]\n"
								"specifying the plotting range for the x and y axes, respectively. A range is allowed\n"
								"for both the source and image plots. Three examples in postscript mode:\n\n"
								"sbmap plotimg source.ps image.ps [-5:5][-5:5] [-15:15][-15:15]\n"
								"sbmap plotimg source.ps image.ps [][] [0:15][0:15]\n"
								"sbmap plotimg image.ps [0:15][0:15]                      (sbprofile mode only)\n\n"
								"In the first example a range is specified for both the x/y axes for both plots, whereas in the\n"
								"second example a range is specified only for the image plot. In example 3, if the source mode\n"
								"is 'sbprofile' and no pixellated source has been created, only the image plot arg's are given.\n";
						else if ((words[2]=="makesrc") or (words[2]=="mksrc"))
							cout << "mksrc\n"
								"Create a pixellated source surface brightness map from one or more analytic surface brightness\n"
								"profiles, which are specified using the 'source' command. The number of source pixels and size\n"
								"of the source pixel grid are specified using the 'src_npixels' and 'srcgrid' commands.\n";
						else if (words[2]=="savesrc")
							cout << "sbmap savesrc [output_file]\n\n"
								"Output the source surface brightness map to files. The surface brightness values are output in\n"
								"matrix form in the file '<output_file>.dat', and in a more efficient form as '<output_file>.sb',\n"
								"which saves information for splitting source pixels into subpixels (only important if\n"
								"srcgrid_type=adaptive_cartesian); this file is read when loading the source later using 'sbmap\n"
								"loadsrc'. The x-values are plotted as '<output_file>.x', and likewise for the y-values.\n";
						else if (words[2]=="plotsrc")
							cout << "sbmap plotsrc [output_file] [-interp] [-nocaust] [-x2/4/8] [-fits]\n\n"
								"Plot the pixellated source surface brightness distribution (which has been generated either\n"
								"using the mksrc, loadsrc, or invert commands). If using a Delaunay grid, the Delaunay source\n"
								"points are also plotted in addition to the corresponding Voronoi pixels; if '-interp' is\n"
								"entered, the interpolated values are shown using the Delaunay triangulation. If the terminal is\n"
								"set to 'text', the surface brightness values are output in matrix form in the file\n"
								"'<source_file>.dat', whereas the x-values are plotted as '<source_file>.x', and likewise for the\n"
								"y-values. If no arguments are given, the file label defaults to 'src_pixel', and these are plotted\n"
								"to the screen in a separate window.  If a lens model has been created, the caustics are also\n"
								"plotted, unless show_cc is set to 'off' or '-nocaust' argument is given (if omitting caustics, the\n"
								"Delaunay source points are also omitted if using a Delaunay grid).\n\n"
								"OPTIONAL: arguments to 'sbmap plotsrc' can be followed with terms in brackets [#:#][#:#]\n"
								"specifying the plotting range for the x and y axes, respectively. Example in postscript mode:\n\n"
								"sbmap plotsrc source.ps [-5:5][-5:5]\n\n";
						else if (words[2]=="plotdata")
							cout << "sbmap plotdata\n"
								"sbmap plotdata [output_file]\n\n"
								"Plot the image surface brightness pixel data, which is loaded using 'sbmap loadimg'. If no\n"
								"file argument is given, uses file label 'img_pixel' for plotting, and the result is plotted\n"
								"to the screen (the output file conventions are the same as in 'sbmap plotimg').\n\n"
								"Optional arguments:\n"
								"  [-nomask] plot data using all pixels, including those outside the chosen mask\n"
								"  [-emask] plot data using extended mask defined by 'emask_n_neighbors'\n\n"
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
							cout << "sbmap set_data_annulus <xcenter> <ycenter> <rmin> <rmax> [theta_min] [theta_max] [xstretch] [ystretch]\n\n"
								"Activates all pixels within the specified annulus. In addition to the center coordinates and min/max\n"
								"radii, the user may also specify an angular range with the additional arguments theta_min, theta_max\n"
								"(where the angles are in degrees and must be in the range 0 to 360). The defaults are theta_min=0,\n"
								"theta_max=360 (in other words, a complete annulus). In addition, one can stretch the annulus along\n"
								"the x- or y- direction by factors [xstretch] and [ystretch]. Also note that this command does not\n"
								"deactivate pixels, so it is recommended to use 'sbmap unset_all_pixels' before running this command.\n\n";
						else if (words[2]=="set_data_window")
							cout << "sbmap set_data_window <xmin> <xmax> <ymin> <ymax>\n\n"
								"Activates all pixels within the specified rectangular window. Note that this command does not deactivate\n"
								"pixels, so it is recommended to use 'sbmap unset_all_pixels' before running this command.\n\n";
						else if (words[2]=="unset_data_annulus")
							cout << "sbmap unset_data_annulus <xcenter> <ycenter> <rmin> <rmax> [theta_min] [theta_max] [xstretch] [ystretch]\n\n"
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
							cout << "sbmap trim_mask_windows <noise_frac_threshold> [size_threshold]\n\n"
								"Removes pixel 'windows' (regions with contiguous pixels) that do not have any pixels with surface\n"
								"brightness greater than <noise_frac_threshold>*pixel_noise, or if the number of pixels in that window\n"
								"is smaller than <size_threshold> (default=0). Typically, thresholds of 4 or 5 are sufficient\n"
								"assuming Gaussian noise.\n\n";
						else if (words[2]=="set_neighbor_pixels")
							cout << "sbmap set_neighbor_pixels [n_neighbors] [-ext/int]\n\n"
								"Activate all pixels that neighbor a pixel already included within the mask, and repeat the procedure\n"
								"[n_neighbors] times (default=1). This effectively enlarges the pixel mask outward by N pixel lengths.\n"
								"If [-ext] is specified, only the neighbor pixels that lie at a larger r-value are activated,\n"
								"whereas neighbors at smaller r-values are selected if [-int] is specified.\n\n";
						else if (words[2]=="set_posrg_pixels")
							cout << "sbmap set_posrg_pixels\n\n"
								"Activate all pixels that have a positive radial gradient. This typically captures the inner part of\n"
								"the lensed arcs in an image. (Note: the gradient is given by the difference in surface brightness\n"
								"between pixel and its next neighbor along x/y; its dot product with r_hat gives the radial gradient).\n"
								"To mask all of the lensed arcs, one can follow up with 'trim mask_windows ...' to clean up the mask,\n"
								"and then use 'sbmap set_neighbor_pixels -ext' until all of the lensed arcs are included in the mask.\n\n";
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
						cout << "source <sourcemodel> <source_parameter##> ... [emode=#]\n"
							"source update <lens_number> ...\n"
							"source vary ...\n"
							"source clear\n\n"
							"Creates a source object from specified source surface brightness profile model and parameters.\n"
							"If other sources are present, the new source will be superimposed with the others. If no arguments\n"
							"are given, a numbered list of the current source models being used is printed along with their\n"
							"parameter values. To remove a source or delete the entire configuration, use 'source clear'.\n"
							"The optional 'emode=#' sets the ellipticity mode for elliptical source models; the elliptical radius\n"
							"is R_ell = sqrt(x^2 + (y/q)^2) for emode=0, or R_ell = sqrt(q*x^2 + y^2/q) for emode=1.\n"
							"If the emode is omitted, the default emode=1 is used.\n\n"
							"Available source models:    (type 'help source '<sourcemodel>' for usage information)\n\n"
							"gaussian -- Gaussian with dispersion <sig> and q*<sig> along major/minor axes respectively\n"
							"sersic -- Sersic profile S = S0 * exp(-b*(R/R_eff)^(1/n))\n"
							"csersic -- Cored Sersic profile S = S0 * exp(-b*(sqrt(R^+rc^2))/R_eff)^(1/n))\n"
							"dsersic -- Double Sersic profile\n"
							"shapelet -- shapelets of specified order n (whose amplitudes can be inferred by inversion)\n"
							"sbmpole -- multipole term\n"
							"tophat -- ellipsoidal 'top hat' profile\n"
							"spline -- splined surface brightness profile (generated from an input file)\n\n";
					else if (words[2]=="clear")
						cout << "source clear <#>\n\n"
							"Remove source model # from the list (the list of source objects can be printed using the\n"
							"'source' command). If no arguments given, delete entire source configuration and start over.\n";
					else if (words[2]=="vary")
						cout << "source vary <source_number> [none]\n"
									"source evary [none]\n\n"
							"Change the parameter vary flags for a specific source model that has already been created. After specifying\n"
							"the source, on the next source vary flags are entered just as you do when creating the source model. Note that\n"
							"the number of vary flags must exactly match the number of parameters for the given source (except that vary\n"
							"flags for the center coordinates can be omitted if the source in question is anchored to another source).\n"
							"If the optional argument 'none' is given after the source number, all vary flags are set to '0' for that\n"
							"source, and you will not be prompted to enter vary flags; if the source number is omitted ('source vary\n"
							"none'), vary flags are set to '0' for all sourcees.\n\n";
					else if (words[2]=="gaussian")
						cout << "source gaussian <sbmax> <sigma> <q> [theta] [x-center] [y-center]\n\n"
							"where <sbmax> is the peak surface brightness (not the peak), <sigma> is the dispersion of\n"
							"the surface brightness along the major axis of the profile, <q> is the axis ratio (so that\n"
							"the dispersion along the minor axis is q*sigma), and [theta] is the angle of rotation\n"
							"(counterclockwise, in degrees) about the center (defaults=0). Note that for theta=0, the\n"
							"major axis of the source is along the " << LENS_AXIS_DIR << " (the direction of the major axis (x/y) for\n"
							"theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="sersic")
						cout << "source sersic <s0> <R_eff> <n> <q> [theta] [x-center] [y-center]\n\n"
							"The sersic profile is defined by S = S0 * exp(-b*(R/R_eff)^(1/n)), where b is a factor automatically\n"
							"determined from the value for n (enforces the half-light radius Re). For the elliptical model, we make\n"
							"the replacement R --> sqrt(q*x^2 + (y^2/q)), analogous to the elliptical radius defined in the lens\n"
							"models. (Note that if n=0.5, this is equivalent to a Gaussian with sigma=R_eff/(1.1774*sqrt(q)).)\n"
							"Note, in the above, [theta] is the angle of rotation (counterclockwise, in degrees) about the center\n"
							"(defaults=0). Note that for theta=0, the major axis of the source is along the " << LENS_AXIS_DIR << " (the\n"
							"direction of the major axis (x/y) for theta=0 is toggled by setting major_axis_along_y on/off).\n";
					else if (words[2]=="sbmpole")
						cout << "source sbmpole [sin/cos] [m=#] <A_m> <r0> [theta] [x-center] [y-center]\n\n"
							"Adds a multipole-like term to the source surface brightness, where the optional argument [sin/cos]\n"
							"specifies whether it is a sine or cosine multipole term (default is cosine), [m=#] specifies the order\n"
							"of the multipole term (which must be an integer; default=0), <A_m> is the coefficient of the monopole\n"
							"term, <r0> is the exponential scale length and [theta] is the angle of rotation (counterclockwise, in\n"
							"degrees) about the center (defaults=0). The radial function is an exponential with scale length r0.\n"
							"For example,\n\n"
							"source sbmpole sin m=3 0.05 2 45 0 0\n\n"
							"specifies a sine multipole term of order 3. (For sine terms, the coefficient is labeled as B_m\n"
							"instead of A_m, so in this example B_m=0.05.)\n"
							"Keep in mind that the order of the multiple (m) cannot be varied as a parameter, so only the\n"
							"remaining five parameters can be varied during a fit or using the 'source update' command.\n";
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
				else if (words[1]=="lensinfo")
					cout << "lensinfo <x> <y>\n\n"
						"Displays the total kappa, deflection, potential, magnification, shear magnitude/direction, and\n"
						"corresponding source position for the point (x,y).\n";
				else if (words[1]=="plotlensinfo")
					cout << "plotlensinfo [file_root] [residual_lensnumber\n\n"
						"Plot a pixel map of the kappa, magnification, shear, and potential at each pixel, which are output to\n"
						"'<file_root>.kappa', '<file_root>.mag', and so on (if no file label is specified, <file_root> defaults\n"
						"to 'lensmap'). The number of pixels set using the 'img_npixels' command, and grid dimensions are defined\n"
						" by the 'grid' command. All files are output to the directory set by the 'set_output_dir' command.\n"
						"To plot the residual values for a perturbed model (perturbed - smooth), set [residual_lensnumber] to the\n"
						"lens number for the perturber.\n";
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
					cout << "plotlogkappa [file] [-contour=#]\n\n"
						"Plots a colormap of the total log(kappa), either to the screen (if no filename is specified) or to a file.\n"
						"If the argument '-contour=#' is added, contours are overlaid with # specifying the number of contours (the\n"
						"contour stepsize is chosen accordingly). If terminal is set to 'text' and a filename is given, outputs\n"
						"log(kappa) map to file. The number of pixels is set by 'img_npixels'.\n"
						"(Note: in pseudo-elliptical and multipole models, it is possible for kappa to have negative values in some\n"
						"places. In this case, a warning is produced, and then the log of the absolute value of kappa is plotted.\n";
				else if (words[1]=="plotlogmag")
					cout << "plotlogmag [file]\n\n"
						"Plots a colormap of log(magnification), along with contours, either to the screen (if no filename is\n"
						"specified) or to a file. If terminal is set to 'text' and a filename is given, outputs log(mag) map\n"
						"to file. The number of pixels is set by 'img_npixels'.\n";
				else if (words[1]=="findimg")
					cout << "findimg <source_x> <source_y> [-imgpt]\n\n"
						"Finds the set of images corresponding to given source position. The image data is\n"
						"written as follows:\n"
						"<x_position>  <y_position>  <magnification>  <time delay (optional)>\n\n"
						"If the argument '-imgpt' is added, then the numbers you enter are interpreted as a\n"
						"point in the image plane, and the partner images are found (corresponding to a common\n"
						"source point).\n\n";
				else if (words[1]=="plotimg")
					cout << "plotimg <source_x> <source_y> [imagepic_file] [sourcepic_file] [-grid] [-imgpt]\n\n"
						"Plots the set of images corresponding to given source position, together with the critical\n"
						"curves. If no filename arguments are given, plots to the screen in a separate window.\n"
						"The resulting image information is also written to the screen as follows:\n"
						"<x_position>  <y_position>  <magnification>  <time delay (optional)>\n\n"
						"OPTIONAL: the arguments to plotimgs can be followed with terms in brackets [#:#][#:#]\n"
						"specifying the plotting range for the x and y axes, respectively. A range is allowed\n"
						"for both the source and image plots. Two examples in postscript mode:\n\n"
						"plotimg source.ps images.ps [-5:5][-5:5] [-15:15][-15:15]\n"
						"plotimg source.ps images.ps [][] [0:15][0:15]\n\n"
						"In the first example a range is specified for both the x/y axes for both plots,\n"
						"whereas in the second example a range is specified only for the image plot.\n\n"
						"If the argument '-grid' is added, plots the grid used for searching along with the images.\n"
						"If the argument '-imgpt' is added, then the numbers you enter are interpreted as a\n"
						"point in the image plane, and the partner images are found (corresponding to a common\n"
						"source point).\n\n";
				else if (words[1]=="plotshear")
					cout << "plotshear\n"
						"plotshear <xmin> <xmax> <nx> <ymin> <ymax> <ny> [shear_outfile]\n\n"
						"Plots a representation of the shear field over a grid of points in the image plane, where the\n"
						"line segments indicate the direction of the shear at the center of that line segment. (The length\n"
						"of the line segments are meaningless and do not indicate shear magnitude). If no arguments are\n"
						"given, the grid is chosen to be co-centered and twice as large as the image grid specified by\n"
						"the 'grid' command, and by default nx=60, ny=60. The result is plotted to the screen unless\n"
						"all grid parameters are specified along with an output filename.\n";
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
				else if (words[1]=="raytracetab")
					cout << "raytracetab <xmin> <xmax> <xpoints> <ymin> <ymax> <ypoints> [source_outfile]\n\n"
						"Creates a regular grid of images, ray-traces them to the source plane, and plots the resulting\n"
						"source points to [source_outfile] (default='sourcexy.in')\n";
				else if (words[1]=="mksrcgal")
					cout << "mksrcgal <xcenter> <ycenter> <a> <q> <angle> <n_ellipses> <pts_per_ellipse> [outfile]\n\n"
						"Creates an elliptical grid of sources (representing a galaxy) with major axis a, axis ratio q.\n"
						"The grid is a series of concentric ellipses, plotted to [outfile] (default='sourcexy.in').\n";
				else if (words[1]=="plotkappa")
					cout << "plotkappa <rmin> <rmax> <steps> <kappa_file> [dkappa_file] [lens=#]\n\n"
						"Plots the radial kappa profile and its (optional) derivative to <kappa_file> and [dkappa_file]\n"
						"respectively (or to the screen if no files are given). If the 'lens=#' argument is not given,\n"
						"the total kappa profile (averaged over all angles) is plotted, with r=0 chosen to be at the center\n"
						"of the primary lens (which is set by 'primary_lens'). In addition to the kappa, the radius in kpc\n"
						"and density in solar masses per kpc^2 are given in the third and fourth columns, i.e. the output\n"
						"file has the following format:\n\n"
						"r(arcsec) kappa r(kpc) Sigma(M_sun/kpc^2)\n\n"
						"If a specific lens is chosen by the 'lens=#' argument, the following is plotted for that lens:\n\n"
						"r(arcsec) kappa kappa_average deflection enclosed_mass r(kpc) Sigma(M_sun/kpc^2) rho3d(M_sun/kpc^3)\n\n"
						"where Sigma and rho3d are the 2d and 3d density profiles of the lens, respectively.\n"
						"(NOTE: Only text mode plotting is supported for this command.)\n";
				else if (words[1]=="plotmass")
					cout << "plotmass <rmin> <rmax> <steps> <mass_file>\n\n"
						"Plots the radial mass profile to <mass_file>. The columns in the output file are radius in\n"
						"arcseconds, mass in m_solar, and radius in kpc respectively.\n"
						"NOTE: Only text mode plotting is supported for this feature.\n";
				else if (words[1]=="clear")
					cout << "clear <#>\n\n"
						"Remove lens galaxy # from the list (the list of galaxies can be printed using the\n"
						"'lens' command). If no arguments given, delete entire lens configuration and start over.\n";
				else if (words[1]=="cc_reset")  // obsolete--should probably remove
					cout << "cc_reset -- (no arguments) delete the current critical curve spline\n"; // obsolete--should probably remove
				else if (words[1]=="integral_method")
					cout << "integral_method <method> [npoints]\n\n"
						"Set integration method (either 'patterson', 'fejer', 'romberg' or 'gauss') which is used for lens\n"
						"models where numerical quadrature is required for the lensing calculations. See below for a description\n"
						"of each method. If gauss is selected, can set number of points = [npoints]. For patterson or romberg,\n"
						"which are both adaptive quadrature methods, the required error tolerance is controlled by the setting\n"
						"'integral_tolerance'. If no arguments are given, prints current integration method.\n\n"
						"\033[4mpatterson\033[0m: Gauss-Patterson quadrature is the fastest integration method in Qlens that is adapt-\n"
						"   ive, meaning the integral can be refined iteratively as more points are added until the desired\n"
						"   tolerance is achieved. It is based on the Kronrod-Patterson rules which start from Gauss-Legendre\n"
						"   quadrature and successively add points to achieve higher order, giving nested quadrature rules.\n"
						"   The maximum number of allowed points is 512, so QLens will give a warning if tolerance has not\n"
						"   been reached after 512 points (this may happen for density profiles with very steep central cusps).\n\n"
						"\033[4mfejer\033[0m: Fejer quadrature is an open-interval version of Clenshaw-Curtis quadrature,\n"
						"   which can be nearly as fast as Gauss-Patterson but allows for a higher number of function\n"
						"   evaluations (up to ~6000).\n\n"
						"\033[4mromberg\033[0m: Romberg integration uses Newton-Cotes rules of successively higher orders and\n"
						"   generates an estimated value and error using Richardson extrapolation. Generally a lot more\n"
						"   points are required compared to 'patterson', 'gauss' or 'fejer' to achieve the same accuracy, but\n"
						"   in principle there is no limit to the number of points that can be used to achieve the desired\n"
						"   integral tolerance.\n\n"
						"\033[4mgauss\033[0m: Gaussian quadrature is a non-adaptive method that uses a fixed number of points,\n"
						"   which can be specified with an additional argument [npoints] (default=20).\n\n";
				else if (words[1]=="integral_tolerance")
					cout << "integral_tolerance <tolerance>\n\n"
						"Set tolerance limit for numerical integration, specifically for the adaptive quadrature methods\n"
						"'patterson' (Gauss-Patterson quadrature), 'fejer' and 'romberg'. For 'patterson', the integration\n"
						"stops when the difference between successive quadrature estimates is smaller than the specified\n"
						"tolerance. Note that the tolerance is thus the estimated error in second-to-last iteration, and\n"
						"hence the error in the final estimate is usually a great deal smaller than the specified tol-\n"
						"erance. For 'romberg', the error returned using Richardson extrapolation is required to be less\n"
						"than the specified tolerance.\n";
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
						"Mode 2: in kappa(R), we let R^2 --> (1-e)^2*x^2 + (1+e)^2*y^2. Ellipticity parameter: e (epsilon)\n"
						"Mode 3: in potential(R), we let R^2 --> (1-e)*x^2 + (1+e)*y^2. Ellipticity parameter: e (epsilon)\n\n"
						"If a lens is created using mode 3, the prefix 'pseudo-' is added to the lens model name. The\n"
						"pseudo-elliptical model can do lens calculations significantly faster in most of the elliptical\n"
						"mass models, since analytic formulas are used for the deflection. Exceptions are the dpie\n"
						"model and the sple model in the case where s=0 or alpha=1, since in these cases the formulas\n"
						"are analytic regardless. Keep in mind however that the pseudo-elliptical models can lead to\n"
						"unphysical density contours when the ellipticity is high enough (you can check this using the\n"
						"command 'plotlogkappa').\n";
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
				else if (words[1]=="chisqpos")
					cout << "chisqpos <on/off>\n\n"
						"Include the image positions in the chi-square function (if on). Note that this is only relevant\n"
						"for point source searches (i.e. fit source_mode = ptsource). (default=on)\n";
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
				else if (words[1]=="inversion_method")
					cout << "inversion_method <method>\n\n"
						"Set method for storing/inverting lensing matrices. Options are:\n\n"
						"dense -- Lmatrix, Fmatrix constructed as dense matrices; inversion performed by native code or Intel MKL\n"
						"fdense -- Fmatrix constructed/inverted as a dense matrix; however, Lmatrix is constructed as sparse\n"
						"mumps -- matrices are constructed/inverted as sparse matrices; inversion handled by MUMPS package\n"
						"umfpack -- matrices are constructed/inverted as sparse matrices; inversion handled by UMFPACK package\n"
						"cg -- matrices are constructed/inverted as sparse matrices; inversion done by conjugate gradient method\n\n";
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
				else if (words[1]=="interpolation_method")
					cout << "interpolation_method <method>   *** NOTE: deprecated (use 'interpolation_method' instead) *** \n\n"
						"Set method for interpolating over source pixels after ray-tracing image pixels to the source plane.\n"
						"Available methods are:\n\n"
						"3pt -- interpolate surface brightness using linear interpolation in the three nearest source pixels.\n"
						"nn -- interpolate surface brightness using natural neighbor interpolation (for Delaunay grids only.)\n";
				else if (words[1]=="raytrace_method")
					cout << "raytrace_method <method>   *** NOTE: deprecated (use 'interpolation_method' instead) *** \n\n"
						"Set method for ray tracing image pixels to source pixels (for either cartesian or delaunay grids).\n"
						"Available methods are:\n\n"
						"interpolate_3pt -- interpolate surface brightness using linear interpolation in the three nearest\n"
						"                 source pixels.\n"
						"interpolate_nn -- interpolate surface brightness using natural neighbors interpolation (for Delaunay\n"
						"                 source mode only.)\n"
						"overlap -- after ray-tracing a pixel to the source plane, find the overlap area with all source\n"
						"           pixels it overlaps with and weight the surface brightness accordingly (cartesian only).\n";
				else if (words[1]=="img_npixels")
					cout << "img_npixels <npixels_x> <npixels_y>\n\n"
						"Set the number of pixels, along x and y, for plotting image surface brightness maps (using 'sbmap\n"
						"plotimg'). If no arguments are given, prints the current image pixel dimensions.\n"
						"(To set the image grid size and location, use the 'grid' command.)\n";
				else if (words[1]=="src_npixels")
					cout << "src_npixels <npixels_x> <npixels_y>\n\n"
						"Set the number of pixels, along x and y, for producing a pixellated source grid using the 'sbmap\n"
						"mksrc' or 'sbmap invert' command, or doing a model fit with a pixellated source. Note that if\n"
						"the number of source pixels is changed manually, 'auto_src_npixels' is automatically turned off.\n"
						"If no arguments are given, prints the current source pixel setting. (To set the source grid size\n"
						"and location, use the 'srcgrid' command.)\n";
				else if ((words[1]=="data_pixel_noise") or (words[1]=="bg_pixel_noise"))
					cout << "bg_pixel_noise <noise>\n\n"
						"Set the background image pixel noise, which is the dispersion in the surface brightness of each pixel.\n"
						"Note that if a noise map is loaded using 'sbmap load_noisemap', then QLens will use only the noise\n"
						"map and bg_pixel_noise will be ignored.\n";
				else if (words[1]=="simulate_pixel_noise")
					cout << "simulate_pixel_noise <on/off>\n\n"
						"If on, add random Gaussian pixel noise to lensed pixel images (produced by the 'sbmap plotimg' command),\n"
						"where the dispersion in surface brightness of each pixel is given either from 'bg_pixel_noise' or\n"
						"from a noise map (if a noise map has been loaded via the 'sbmap load_noisemap' command).\n";
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

					cout << "colorbar min surface brightness = ";
					if (colorbar_min==-1e30) cout << "auto" << endl;
					else cout << colorbar_min << endl;

					cout << "colorbar max surface brightness = ";
					if (colorbar_max==1e30) cout << "auto" << endl;
					else cout << colorbar_max << endl;

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
					cout << "raytrace_method: " << ((ray_tracing_method==Area_Overlap) ? "overlap (pixel overlap area)\n" : ((ray_tracing_method==Interpolate) and (!natural_neighbor_interpolation)) ? "interpolate_3pt (linear 3-point interpolation)\n" : ((ray_tracing_method==Interpolate) and (natural_neighbor_interpolation)) ? "interpolate_nn (natural neighbors interpolation)\n" : "unknown\n");
					cout << "simulate_pixel_noise = " << display_switch(simulate_pixel_noise) << endl;
					//cout << "psf_width: (" << psf_list[0]->psf_width_x << "," << psf_list[0]->psf_width_y << ")\n";
					cout << "psf_threshold = " << psf_threshold << endl;
					//cout << "psf_mpi: " << display_switch(psf_convolution_mpi) << endl;
					cout << endl;
					cout << "\033[4mSource pixel reconstruction settings\033[0m\n";
					cout << "inversion_method: " << ((inversion_method==MUMPS) ? "LDL factorization (MUMPS)\n" : (inversion_method==UMFPACK) ? "LU factorization (UMFPACK)\n" : (inversion_method==CG_Method) ? "conjugate gradient method\n" : "unknown\n");
					cout << "adaptive_subgrid: " << display_switch(adaptive_subgrid) << endl;
					cout << "auto_src_npixels: " << display_switch(auto_srcgrid_npixels) << endl;
					cout << "auto_srcgrid: " << display_switch(auto_sourcegrid) << endl;
					cout << "auto_shapelet_scale: " << display_switch(auto_shapelet_scaling) << endl;
					cout << "noise_threshold = " << noise_threshold << endl;
					if (default_data_pixel_size < 0) cout << "data_pixel_size: not specified\n";
					else cout << "data_pixel_size: " << default_data_pixel_size << endl;
					cout << "bg_pixel_noise = " << background_pixel_noise << endl;
					cout << "inversion_nthreads = " << inversion_nthreads << endl;
					cout << "lum_weighted_regularization: " << display_switch(use_lum_weighted_regularization) << endl;
					cout << "dist_weighted_regularization: " << display_switch(use_distance_weighted_regularization) << endl;
					cout << "mag_weighted_regularization: " << display_switch(use_mag_weighted_regularization) << endl;
					//cout << "regparam_lhi = " << regparam_lhi << endl;
					//cout << "vary_regparam_lhi: " << display_switch(vary_regparam_lhi) << endl;
					cout << "outside_sb_prior: " << display_switch(outside_sb_prior) << endl;
					cout << "outside_sb_noise_threshold = " << outside_sb_prior_noise_frac << endl;

					cout << "nimg_prior: " << display_switch(n_image_prior) << endl;
					cout << "nimg_threshold = " << n_image_threshold << endl;
					cout << "nimg_mag_threshold = " << srcpixel_nimg_mag_threshold << endl;
					cout << "nimg_sb_frac_threshold = " << n_image_prior_sb_frac << endl;
					cout << "activate_unmapped_srcpixels: " << display_switch(activate_unmapped_source_pixels) << endl;
					cout << "exclude_srcpixels_outside_mask: " << display_switch(exclude_source_pixels_beyond_fit_window) << endl;
					cout << "remove_unmapped_subpixels: " << display_switch(regrid_if_unmapped_source_subpixels) << endl;
					cout << "sb_threshold = " << sb_threshold << endl;
					cout << "parallel_mumps: " << display_switch(parallel_mumps) << endl;
					cout << "show_mumps_info: " << display_switch(show_mumps_info) << endl;
					cout << endl;
				}
				if (show_fit_settings) {
					cout << "\033[4mChi-square function settings\033[0m\n";
					cout << "imgplane_chisq: " << display_switch(imgplane_chisq) << endl;
					cout << "chisqmag: " << display_switch(use_magnification_in_chisq) << endl;
					cout << "chisqpos: " << display_switch(include_imgpos_chisq) << endl;
					cout << "chisqflux: " << display_switch(include_flux_chisq) << endl;
					cout << "chisq_time_delays: " << display_switch(include_time_delay_chisq) << endl;
					cout << "chisq_weak_lensing: " << display_switch(include_weak_lensing_chisq) << endl;
					cout << "chisq_parity: " << display_switch(include_parity_in_chisq) << endl;
					cout << "analytic_bestfit_src: " << display_switch(use_analytic_bestfit_src) << endl;
					cout << "chisq_mag_threshold = " << chisq_magnification_threshold << endl;
					cout << "chisq_imgsep_threshold = " << chisq_imgsep_threshold << endl;
					cout << "chisq_imgplane_threshold = " << chisq_imgplane_substitute_threshold << endl;
					cout << "nimg_penalty: " << display_switch(n_images_penalty) << endl;
					cout << "chisqtol = " << chisq_tolerance << endl;
					cout << "analytic_srcflux: " << display_switch(analytic_source_flux) << endl;
					cout << "syserr_pos = " << syserr_pos << endl;
					cout << "wl_shearfac = " << wl_shear_factor << endl;
					cout << endl;
					cout << "\033[4mOptimization and Monte Carlo sampler settings\033[0m\n";
					cout << "fit method: " << ((fitmethod==POWELL) ? "powell\n" : (fitmethod==SIMPLEX) ? "simplex\n" : (fitmethod==NESTED_SAMPLING) ? "nest\n" : (fitmethod==TWALK) ? "twalk" : (fitmethod==POLYCHORD) ? "polychord" : (fitmethod==MULTINEST) ? "multinest" : "Unknown fitmethod\n");
					cout << "fit source_mode: " << ((source_fit_mode==Point_Source) ? "ptsource\n" : (source_fit_mode==Cartesian_Source) ? "cartesian\n" : (source_fit_mode==Delaunay_Source) ? "delaunay" : (source_fit_mode==Parameterized_Source) ? "sbprofile\n" : (source_fit_mode==Shapelet_Source) ? "shapelet\n" : "unknown\n");
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
					cout << "hubble = " << cosmo.get_hubble() << endl;
					cout << "omega_m = " << cosmo.get_omega_m() << endl;
					cout << "zlens = " << lens_redshift << endl;
					cout << "zsrc = " << source_redshift << endl;
					cout << "zsrc_ref = " << reference_source_redshift << endl;
					cout << "auto_zsrc_scaling = " << auto_zsource_scaling << endl;
					//cout << "vary_hubble: " << display_switch(vary_hubble_parameter) << endl;
					//cout << "vary_omega_m: " << display_switch(vary_omega_matter_parameter) << endl;
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
					cout << "sim_err_shear = " << sim_err_shear << endl;
					cout << endl;
				}
			}
		}
		else if (words[0]=="read")
		{
			if (nwords == 2) {
				if (!open_script_file(words[1])) Complain("input file '" << words[1] << "' could not be opened");
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
				else cerr << "output file '" << words[1] << "' could not be opened" << endl;
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
				} else if (method_name=="fejer") {
					if (nwords > 2) Complain("no arguments are allowed for 'integral_method fejer'");
					method = Fejer_Quadrature;
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
					else if (method==Fejer_Quadrature) cout << "Integration method: Fejer quadrature with tolerance = " << integral_tolerance << endl;
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
			int pos;
			if (nwords==1) {
				if (mpi_id==0) {
					double xlh, ylh;
					xlh = grid_xlength/2.0;
					ylh = grid_ylength/2.0;
					if ((xlh < 1e3) and (ylh < 1e3)) cout << resetiosflags(ios::scientific);
					cout << "grid = (" << grid_xcenter-xlh << "," << grid_xcenter+xlh << ") x (" << grid_ycenter-ylh << "," << grid_ycenter+ylh << ")" << endl;
					if (use_scientific_notation) cout << setiosflags(ios::scientific);
				}
			} else if ((nwords==2) and ((words[1]=="-use_data_pxsize") or (words[1]=="-imgpixel"))) {
				if ((n_data_bands > 0) and (default_data_pixel_size <= 0)) Complain("must have loaded image data or else data_pixel_size > 0 to create grid based on image pixels");
				set_grid_from_pixels();
			} else if ((nwords==2) and (pos = words[1].find("-pxsize=")) != string::npos) {
				double psize;
				string sizestring = words[1].substr(pos+8);
				stringstream sizestr;
				sizestr << sizestring;
				if (!(sizestr >> psize)) Complain("incorrect format for pixel noise");
				if (psize < 0) Complain("pixel size value cannot be negative");
				default_data_pixel_size = psize;
				set_grid_from_pixels();
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
			} else if ((nwords == 2) and (words[1]=="-data")) {
				if (n_data_bands > 0) {
					n_image_pixels_x = imgpixel_data_list[0]->npixels_x;
					n_image_pixels_y = imgpixel_data_list[0]->npixels_y;
					set_img_npixels(imgpixel_data_list[0]->npixels_x,imgpixel_data_list[0]->npixels_y);
					if (fft_convolution) cleanup_FFT_convolution_arrays(); // since number of image pixels has changed, will need to redo FFT setup
				} else Complain("image pixel data has not been loaded");
			} else if (nwords == 3) {
				int npx, npy;
				if (!(ws[1] >> npx)) Complain("invalid number of pixels");
				if (!(ws[2] >> npy)) Complain("invalid number of pixels");
				set_img_npixels(npx,npy);
				if (fft_convolution) cleanup_FFT_convolution_arrays(); // since number of image pixels has changed, will need to redo FFT setup
			} else Complain("two arguments required to set 'img_npixels' (npixels_x, npixels_y), or else use '-data' argument");
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
		else if (words[0]=="n_src_clusters")
		{
			int n_clusters;
			if (nwords == 2) {
				if (words[1]=="all") n_src_clusters = ALL_DATAPIXELS;
				else if (words[1]=="half") n_src_clusters = HALF_DATAPIXELS;
				else {
					if (!(ws[1] >> n_clusters)) Complain("invalid number of source clusters for Delaunay grid");
					n_src_clusters = n_clusters;
					if (n_clusters < 0) n_src_clusters = HALF_DATAPIXELS;
				}
				if (use_f_src_clusters) use_f_src_clusters = false;
			} else if (nwords==1) {
				if (mpi_id==0) {
					if (n_src_clusters == ALL_DATAPIXELS) cout << "Number of source clusters for Delaunay grid = 'all' (number of data pixels within mask)" << endl;
					else if (n_src_clusters == HALF_DATAPIXELS) cout << "Number of source clusters for Delaunay grid = 'half' (half the number of data pixels within mask)" << endl;
					else cout << "Number of source clusters for Delaunay grid = " << n_src_clusters << endl;
				}
			} else Complain("must specify either zero or one argument (number of source clusters)");
		}
		else if (words[0]=="f_src_clusters")
		{
			double f_clusters;
			if (nwords == 2) {
				if (!(ws[1] >> f_clusters)) Complain("invalid fraction of source clusters for Delaunay grid");
				f_src_clusters = f_clusters;
				if (!use_f_src_clusters) use_f_src_clusters = true;
			} else if (nwords==1) {
				if (mpi_id==0) {
					cout << "Fraction of source clusters versus data pixels for Delaunay grid = " << f_src_clusters << endl;
				}
			} else Complain("must specify either zero or one argument (fraction of source clusters)");
		}
		else if (words[0]=="n_cluster_it")
		{
			int n_clusters;
			if (nwords == 2) {
				if (!(ws[1] >> n_clusters)) Complain("invalid number of clustering iterations for Delaunay grid");
				n_cluster_iterations = n_clusters;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Number of clustering iterations for Delaunay grid = " << n_cluster_iterations << endl;
			} else Complain("must specify either zero or one argument (number of clustering iterations)");
		}
		else if (words[0]=="auxgrid_npixels")
		{
			// This is the number of pixels (along x and y) for the sourcegrid created for the nimg_prior; not used if fitting with cartesian sourcegrid
			if (nwords==1) {
				if (mpi_id==0) cout << "Number of source pixels per side for auxiliary srcgrid = " << auxiliary_srcgrid_npixels << endl;
			} else if (nwords == 2) {
				int npix;
				if (!(ws[1] >> npix)) Complain("invalid number of pixels");
				auxiliary_srcgrid_npixels = npix;
			} else Complain("only one argument allowed for 'auxgrid_npixels'");
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
			bool use_kpc = false;
			if ((nwords == 4) and (words[3]=="-kpc")) {
				use_kpc = true;
				remove_word(3);
			}
			if (nwords != 3) Complain("exactly two arguments required for mass_r command (lens_number,radius_arcsec)");
			int lnum;
			double r_arcsec;
			if (!(ws[1] >> lnum)) Complain("invalid lens number");
			if (lnum >= nlens) Complain("specified lens number does not exist");
			if (!(ws[2] >> r_arcsec)) Complain("invalid radius");
			if (mpi_id==0) output_mass_r(r_arcsec,lnum,use_kpc);
		}
		else if ((words[0]=="lens") or ((words[0]=="fit") and (nwords > 1) and (words[1]=="lens")))
		{
			bool update_parameters = false;
			bool update_specific_parameters = false; // option for user to update one (or more) specific parameters rather than update all of them at once
			bool vary_parameters = false;
			bool anchor_lens_center = false;
			bool add_shear = false;
			bool egrad = false;
			int egrad_mode = -1;
			int n_bspline_coefs = 0; // only used for egrad_mode=0 (B-spline mode)
			bool enter_egrad_params_and_varyflags = true;
			bool enter_knots = false;
			bool fgrad = false;
			int egrad_qi, egrad_qf, egrad_theta_i, egrad_theta_f;
			int fgrad_amp_i, fgrad_amp_f;
			double ximin; // only used for egrad_mode=0 (B-spline mode)
			double ximax; // only used for egrad_mode=0 (B-spline mode)
			bool linear_xivals; // only used for egrad_mode=0 (B-spline mode)
			double xiref; // only used for egrad_mode=2 
			int emode = -1; // if set, then specifies the ellipticity mode for the lens being created
			bool lensed_center_coords = false;
			boolvector vary_flags, shear_vary_flags;
			dvector efunc_params, egrad_knots;
			dvector fgrad_params, fgrad_knots;
			vector<string> specific_update_params;
			vector<double> specific_update_param_vals;
			dvector param_vals;
			double shear_param_vals[2];
			int default_nparams, nparams_to_vary, tot_nparams_to_vary;
			int anchornum; // in case new lens is being anchored to existing lens
			int lens_number;
			vector<int> fourier_mvals;
			vector<double> fourier_Amvals;
			vector<double> fourier_Bmvals;
			int fourier_nmodes = 0;
			bool update_zl = false;
			double zl_in = lens_redshift;
			bool vary_zl = false;
			int pmode = default_parameter_mode;
			bool is_perturber = false;
			bool transform_to_pixsrc_frame = false;

			const int nmax_anchor = 100;
			ParamAnchor parameter_anchors[nmax_anchor]; // number of anchors per lens can't exceed 100 (which will never happen!)
			int parameter_anchor_i = 0;
			for (int i=0; i < nmax_anchor; i++) parameter_anchors[i].anchor_object_number = nlens; // by default, param anchors are to parameters within the new lens, unless specified otherwise

			for (int i=nwords-1; i > 1; i--) {
				if (words[i]=="-lensed_center") {
					lensed_center_coords = true;
					remove_word(i);
				}
			}
			for (int i=nwords-1; i > 1; i--) {
				if (words[i]=="-transform_to_pixsrc_frame") {
					transform_to_pixsrc_frame = true;
					remove_word(i);
				}
			}

			for (int i=nwords-1; i > 1; i--) {
				int pos;
				if ((pos = words[i].find("egrad=")) != string::npos) {
					string egradstring = words[i].substr(pos+6);
					stringstream egradstr;
					egradstr << egradstring;
					if (!(egradstr >> egrad_mode)) Complain("incorrect format for ellipticity gradient mode; must specify 0, 1, or 2");
					if ((egrad_mode < 0) or (egrad_mode > 2)) Complain("ellipticity gradient mode must be either 0, 1, or 2");
					egrad = true;
					remove_word(i);
				}
			}	
			for (int i=nwords-1; i > 1; i--) {
				if (words[i]=="-fgrad") {
					fgrad = true;
					remove_word(i);
				}
			}


			vector<int> remove_list;
			for (int i=3; i < nwords; i++) {
				int pos0;
				if ((words[i][0]=='f') and (pos0 = words[i].find("=")) != string::npos) {
					if (i==nwords-1) Complain("must specify both fourier amplitudes A_m and B_m (e.g. 'f1=0.01 0.02')");
					string mvalstring, amstring, bmstring;
					mvalstring = words[i].substr(1,pos0-1);
					amstring = words[i].substr(pos0+1);
					bmstring = words[i+1];
					int mval;
					double Am, Bm;
					stringstream mstr, astr, bstr;
					mstr << mvalstring;
					astr << amstring;
					bstr << bmstring;
					if (!(mstr >> mval)) Complain("invalid fourier m-value");
					if (!(astr >> Am)) Complain("invalid fourier A_m amplitude");
					if (!(bstr >> Bm)) Complain("invalid fourier B_m amplitude");
					fourier_mvals.push_back(mval);
					fourier_Amvals.push_back(Am);
					fourier_Bmvals.push_back(Bm);
					remove_list.push_back(i);
					remove_list.push_back(i+1);
					//remove_word(i+1);
					//remove_word(i);
					//astr = words[i].substr(pos0+8);
					//int pos, lnum, pnum;
					fourier_nmodes++;
				}
			}
			for (int i=remove_list.size()-1; i >= 0; i--) {
				remove_word(remove_list[i]);
			}

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
									(profile_name==sple_LENS) ? "sple" :
									(profile_name==dpie_LENS) ? "dpie" :
									(profile_name==MULTIPOLE) ? "mpole" :
									(profile_name==nfw) ? "nfw" :
									(profile_name==TRUNCATED_nfw) ? "tnfw" :
									(profile_name==CORED_nfw) ? "cnfw" :
									(profile_name==HERNQUIST) ? "hern" :
									(profile_name==EXPDISK) ? "expdisk" :
									(profile_name==CORECUSP) ? "corecusp" :
									(profile_name==SERSIC_LENS) ? "sersic" :
									(profile_name==DOUBLE_SERSIC_LENS) ? "dsersic" :
									(profile_name==CORED_SERSIC_LENS) ? "csersic" :
									(profile_name==SHEAR) ? "shear" :
									(profile_name==SHEET) ? "sheet" :
									(profile_name==DEFLECTION) ? "deflection" :
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
					int pos0, pos1;
					if ((pos0 = words[i].find("/anchor=")) != string::npos) {
						if ((pos1 = words[i].find("x^")) != string::npos) {
							string pvalstring, expstring, astr;
							pvalstring = words[i].substr(0,pos1);
							expstring = words[i].substr(pos1+2,pos0-pos1-2);
							astr = words[i].substr(pos0+8);
							int pos, lnum, pnum;
							double ratio, anchor_exponent;
							stringstream rstr, expstr;
							rstr << pvalstring;
							rstr >> ratio;
							expstr << expstring;
							expstr >> anchor_exponent;
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
								parameter_anchors[parameter_anchor_i].use_exponent = true;
								parameter_anchors[parameter_anchor_i].paramnum = i-2;
								parameter_anchors[parameter_anchor_i].anchor_object_number = lnum;
								parameter_anchors[parameter_anchor_i].anchor_paramnum = pnum;
								parameter_anchors[parameter_anchor_i].ratio = ratio;
								parameter_anchors[parameter_anchor_i].exponent = anchor_exponent;
								parameter_anchor_i++;
								words[i] = pvalstring;
								ws[i].str(""); ws[i].clear();
								ws[i] << words[i];
							} else Complain("incorrect format for anchoring parameter; must type 'anchor=<lens_number>,<param_number>' in place of parameter");
						} else {
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
								parameter_anchors[parameter_anchor_i].use_implicit_ratio = true;
								parameter_anchors[parameter_anchor_i].paramnum = i-2;
								parameter_anchors[parameter_anchor_i].anchor_object_number = lnum;
								parameter_anchors[parameter_anchor_i].anchor_paramnum = pnum;
								parameter_anchor_i++;
								words[i] = pvalstring;
								ws[i].str(""); ws[i].clear();
								ws[i] << words[i];
							} else Complain("incorrect format for anchoring parameter; must type 'anchor=<lens_number>,<param_number>' in place of parameter");
						}
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
							parameter_anchors[parameter_anchor_i].anchor_object_number = lnum;
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
					} 
				}
				if (n_updates > 0) {
					if (n_updates < nwords-2) Complain("lens parameters must all be updated at once, or else specific parameters using '<param>=...'");
					update_specific_parameters = true;
					for (int i=0; i < n_updates; i++)
						if (lens_list[lens_number]->update_specific_parameter(specific_update_params[i],specific_update_param_vals[i])==false) Complain("could not find parameter '" << specific_update_params[i] << "' in lens " << lens_number);
					reset_grid();
				}
			}
			else if ((nwords > 1) and ((words[1]=="vary") or (words[1]=="changevary")))
			{
				// At the moment, there is no error checking for changing vary flags of anchored parameters. This should be done from within
				// set_lens_vary_parameters(...), and an integer error code should be returned so specific errors can be printed. Then you should
				// simplify all the error checking in the above code for adding lens models so that errors are printed using the same interface.
				bool set_vary_none = false;
				bool set_vary_all = false;
				if (words[nwords-1]=="none") {
					set_vary_none=true;
					remove_word(nwords-1);
				}
				if (words[nwords-1]=="all") {
					set_vary_all=true;
					remove_word(nwords-1);
				}
				if ((nwords==2) and (set_vary_none)) {
					for (int lensnum=0; lensnum < nlens; lensnum++) {
						int npar = lens_list[lensnum]->n_params;
						boolvector vary_flags(npar);
						for (int i=0; i < npar; i++) vary_flags[i] = false;
						set_lens_vary_parameters(lensnum,vary_flags);
					}
				} else if ((nwords==2) and (set_vary_all)) {
					for (int lensnum=0; lensnum < nlens; lensnum++) {
						int npar = lens_list[lensnum]->n_params;
						boolvector vary_flags(npar);
						for (int i=0; i < npar; i++) vary_flags[i] = true;
						set_lens_vary_parameters(lensnum,vary_flags);
					}
				} else {
					if (nwords != 3) Complain("one argument required for 'lens vary' (lens number)");
					int lensnum;
					if (!(ws[2] >> lensnum)) Complain("Invalid lens number to change vary parameters");
					if (lensnum >= nlens) Complain("specified lens number does not exist");
					if ((!set_vary_none) and (!set_vary_all)) {
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
					} else if (set_vary_none) {
						int npar = lens_list[lensnum]->n_params;
						boolvector vary_flags(npar);
						for (int i=0; i < npar; i++) vary_flags[i] = false;
						set_lens_vary_parameters(lensnum,vary_flags);
					} else if (set_vary_all) {
						int npar = lens_list[lensnum]->n_params;
						boolvector vary_flags(npar);
						for (int i=0; i < npar; i++) vary_flags[i] = true;
						set_lens_vary_parameters(lensnum,vary_flags);
					}
				}
				update_specific_parameters = true; // this will ensure it skips trying to create a lens model
			}
			else if ((nwords > 1) and (words[1]=="change_pmode")) {
				if (nwords==4) {
					int lensnum, pm;
					if (!(ws[2] >> lensnum)) Complain("Invalid lens number to change pmode");
					if (lensnum >= nlens) Complain("specified lens number does not exist");
					if (!(ws[3] >> pm)) Complain("Invalid parameter mode");
					lens_list[lensnum]->change_pmode(pm);
				} else Complain("'lens change_pmode' should have two arguments (lens#, pmode)");
				update_specific_parameters = true;  // this will ensure it skips trying to create a lens model
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
				vary_parameters = false; // this gets it to skip to the end and not try to prompt for parameter limits
			}
			else if (words[1]=="clear")
			{
				if (nwords==2) {
					clear_lenses();
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
			else if ((nwords > 1) and (words[1]=="output_plates")) {
				if (nwords==4) {
					int lensnum, np;
					if (!(ws[2] >> lensnum)) Complain("Invalid lens number to change pmode");
					if (lensnum >= nlens) Complain("specified lens number does not exist");
					if (!(ws[3] >> np)) Complain("Invalid parameter mode");
					bool status = lens_list[lensnum]->output_plates(np);
					if (!status) Complain("specified lens does not have ellipticity gradient enabled; could not output plates");
				} else Complain("'lens output_plates' should have two arguments (lens#, # of plates)");
			}
			else if (words[1]=="anchor_center")
			{
				if (nwords != 4) Complain("must specify two arguments for 'lens anchor_center': lens1 --> lens2");
				int lens_num1, lens_num2;
				if (!(ws[2] >> lens_num1)) Complain("invalid lens number for lens to be anchored");
				if (!(ws[3] >> lens_num2)) Complain("invalid lens number for lens to anchor to");
				if (lens_num1 >= nlens) Complain("lens1 number does not exist");
				if (lens_num2 >= nlens) Complain("lens2 number does not exist");
				if (lens_num1 == lens_num2) Complain("lens1, lens2 must be different");
				lens_list[lens_num1]->anchor_center_to_lens(lens_num2);
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
				if ((words[1]=="sple") or (words[1]=="alpha"))
				{
					if (nwords > 9) Complain("more than 7 parameters not allowed for model sple");
					if ((pmode < 0) or (pmode > 1)) Complain("parameter mode must be either 0 or 1");
					if (nwords >= 6) {
						double b, slope, slope2d, s;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> b)) Complain("invalid b parameter for model sple");
						if (!(ws[3] >> slope)) {
							if (pmode==0) Complain("invalid alpha parameter for model sple");
							else if (pmode==1) Complain("invalid gamma parameter for model sple");
						}
						if (pmode==0) slope2d = slope;
						else slope2d = slope-1;
						if (!(ws[4] >> s)) Complain("invalid s (core) parameter for model sple");
						if (!(ws[5] >> q)) Complain("invalid q parameter for model sple");
						if (slope2d <= 0) Complain("2D (projected) density log-slope cannot be less than or equal to zero (or else the mass diverges near r=0)");
						if (nwords >= 7) {
							if (!(ws[6] >> theta)) Complain("invalid theta parameter for model sple");
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
								if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model sple");
								if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model sple");
							}
						}
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 7; // this does not include redshift
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=b; param_vals[1]=slope; param_vals[2]=s; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[5] != "0") or (words[6] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (b,slope,s,q,theta) in model sple";
								}
								else complain_str = "Must specify vary flags for seven parameters (b,slope,s,q,theta,xc,yc) in model sple";
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,7,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(sple_LENS, emode, zl_in, reference_source_redshift, b, slope, s, 0, q, theta, xc, yc, 0, 0, pmode);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							//if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),n_bspline_coefs,enter_egrad_params_and_varyflags))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("sple requires at least 4 parameters (b, alpha, s, q)");
				}
				else if ((words[1]=="dpie") or (words[1]=="pjaffe")) // 'pjaffe' is deprecated
				{
					bool set_tidal_host = false;
					int hostnum;
					if ((pmode < 0) or (pmode > 2)) Complain("parameter mode must be either 0, 1, or 2");
					if (nwords > 9) Complain("more than 7 parameters not allowed for model dpie");
					if (nwords >= 6) {
						double p1, p2, p3;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> p1)) {
							if (pmode==0) Complain("invalid b parameter for model dpie");
							else Complain("invalid sigma_v parameter for model dpie");
						}
						if (words[3].find("host=")==0) {
							string hoststr = words[3].substr(5);
							stringstream hoststream;
							hoststream << hoststr;
							if (!(hoststream >> hostnum)) Complain("invalid lens number for tidal host");
							if (hostnum >= nlens) Complain("lens number does not exist");
							set_tidal_host = true;
							p2 = 0;
						} else if (!(ws[3] >> p2)) Complain("invalid a parameter for model dpie");
						if (!(ws[4] >> p3)) Complain("invalid s (core) parameter for model dpie");
						if (!(ws[5] >> q)) Complain("invalid q parameter for model dpie");
						if (nwords >= 7) {
							if (!(ws[6] >> theta)) Complain("invalid theta parameter for model dpie");
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
								if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model dpie");
								if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model dpie");
							}
						}
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 7;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=p2; param_vals[2]=p3; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[5] != "0") or (words[6] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (b,a,s,q,theta) in model dpie";
								}
								else complain_str = "Must specify vary flags for seven parameters (b,a,s,q,theta,xc,yc) in model dpie";
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,7,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(dpie_LENS, emode, zl_in, reference_source_redshift, p1, 0, p2, p3, q, theta, xc, yc, 0, 0, pmode);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (set_tidal_host) {
								lens_list[nlens-1]->assign_special_anchored_parameters(lens_list[hostnum],1,true);
								if ((vary_parameters) and (vary_flags[1])) lens_list[nlens-1]->unassign_special_anchored_parameter(); // we're only setting the initial value for a
							}
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("dpie requires at least 4 parameters (b, a, s, q)");
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
							for (int i=0; i < parameter_anchor_i; i++) parameter_anchors[i].shift_down();
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
							for (int i=0; i < parameter_anchor_i; i++) parameter_anchors[i].shift_down();
						}
						double theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> a_m)) Complain("invalid a_m parameter for model " << words[1]);
						if (!(ws[3] >> n)) {
							if (kappa_multipole) Complain("invalid beta parameter for model " << words[1]);
							else Complain("invalid n parameter for model " << words[1]);
						}
						bool anchoring_slope = false;
						for (int i=0; i < parameter_anchor_i; i++) { if (parameter_anchors[i].paramnum == 1) anchoring_slope = true; }
						if ((kappa_multipole) and (!anchoring_slope) and (n == 2-m)) Complain("for kmpole, beta cannot be equal to 2-m (or else deflections become infinite)");
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
						default_nparams = 5;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=a_m; param_vals[1]=n; param_vals[2]=theta; param_vals[3]=xc; param_vals[4]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[5]=zl_in;
						else param_vals[5]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						} else {
							add_multipole_lens(zl_in, reference_source_redshift, m, a_m, n, theta, xc, yc, kappa_multipole, sine_term);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) {
								lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							}
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("mpole requires at least 2 parameters (a_m, n)");
				}
				else if (words[1]=="nfw")
				{
					bool set_median_concentration = false;
					bool no_cmed_anchoring = false; // if set_median_concentration = true, then this says to only set the initial c200 this way, but not anchor to cmed
					double cmed_factor = 1.0;
					bool use_cm_prior = false;
					for (int i=nwords-1; i > 1; i--) {
						if (words[i]=="-cmprior") {
							use_cm_prior = true;
							remove_word(i);
						}
					}
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
							if (!(ws[3] >> p2)) Complain("invalid rs_kpc parameter for model nfw");
						} else if (pmode==1) {
							if (!(ws[2] >> p1)) Complain("invalid mvir parameter for model nfw");
							int pos;
							if ((pos = words[3].find("*cmed")) != string::npos) {
								set_median_concentration = true;
								if (words[3].find("*cmed*") != string::npos) no_cmed_anchoring = true;
								string facstring;
								facstring = words[3].substr(0,pos);
								stringstream facstream;
								facstream << facstring;
								if (!(facstream >> cmed_factor)) Complain("invalid factor of median concentration");
							} else if (words[3]=="cmed") { set_median_concentration = true; }
							else if (words[3]=="cmed*") { set_median_concentration = true; no_cmed_anchoring = true; }
							else if (!(ws[3] >> p2)) Complain("invalid c parameter for model nfw");
							if ((set_median_concentration) and (use_cm_prior)) Complain("cannot fix c to median c(M,z) if concentration-mass prior is also being used");
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
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 6;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=p2; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,6,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(nfw, emode, zl_in, reference_source_redshift, p1, 0, p2, 0, q, theta, xc, yc, 0, 0, pmode);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (set_median_concentration) {
								lens_list[nlens-1]->assign_special_anchored_parameters(lens_list[nlens-1],cmed_factor,true);
								if (((vary_parameters) and (vary_flags[1])) or (no_cmed_anchoring)) lens_list[nlens-1]->unassign_special_anchored_parameter(); // we're only setting the initial value for c
							} else if (use_cm_prior) {
								lens_list[nlens-1]->use_concentration_prior = true;
								if (!concentration_prior) concentration_prior = true; // this tells qlens that at least one of the lenses has a c(M,z) prior
							}
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("nfw requires at least 3 parameters (ks, rs, q)");
				}
				else if (words[1]=="tnfw")
				{
					bool set_median_concentration = false;
					bool no_cmed_anchoring = false;
					double cmed_factor = 1.0;
					int tmode = 1;
					if ((update_parameters) and (lens_list[lens_number]->anchor_special_parameter)) {
						set_median_concentration = true;
						pmode = 1; // you should generalize the parameter choice option so it's in the LensProfile class; then you can check for different parametrizations directly
					}
					if ((pmode < 0) or (pmode > 4)) Complain("parameter mode must be either 0, 1, 2, 3, or 4");

					for (int i=2; i < nwords; i++) {
						if (words[i].find("tmode=")==0) {
							if (update_parameters) Complain("tmode=# argument cannot be specified when updating " << words[1]);
							string tstr = words[2].substr(6);
							stringstream tstream;
							tstream << tstr;
							if (!(tstream >> tmode)) Complain("invalid tmode value");
							stringstream* new_ws = new stringstream[nwords-1];
							words.erase(words.begin()+2);
							for (int i=0; i < nwords-1; i++) {
								new_ws[i] << words[i];
							}
							delete[] ws;
							ws = new_ws;
							nwords--;
							for (int i=0; i < parameter_anchor_i; i++) parameter_anchors[i].shift_down();
							if ((tmode < 0) or (tmode > 1)) Complain("truncation mode can only be set to 0 or 1");
							break;
						}
					}

					if (nwords > 9) Complain("more than 7 parameters not allowed for model tnfw");
					if (nwords >= 6) {
						double p1, p2, p3;
						double q, theta = 0, xc = 0, yc = 0;
						if (pmode==4) {
							if (!(ws[2] >> p1)) Complain("invalid mvir parameter for model nfw");
							if (!(ws[3] >> p2)) Complain("invalid rs_kpc parameter for model nfw");
							if (!(ws[4] >> p3)) Complain("invalid tau_s parameter for model tnfw");
						} else if (pmode==3) {
							if (!(ws[2] >> p1)) Complain("invalid mvir parameter for model nfw");
							if (!(ws[3] >> p2)) Complain("invalid rs_kpc parameter for model nfw");
							if (!(ws[4] >> p3)) Complain("invalid rt_kpc parameter for model tnfw");
						} else if ((pmode==1) or (pmode==2)) {
							if (!(ws[2] >> p1)) Complain("invalid mvir parameter for model nfw");
							int pos;
							if ((pos = words[3].find("*cmed")) != string::npos) {
								set_median_concentration = true;
								if (words[3].find("*cmed*") != string::npos) no_cmed_anchoring = true;
								string facstring;
								facstring = words[3].substr(0,pos);
								stringstream facstream;
								facstream << facstring;
								if (!(facstream >> cmed_factor)) Complain("invalid factor of median concentration");
							} else if (words[3]=="cmed") { set_median_concentration = true; }
							else if (words[3]=="cmed*") { set_median_concentration = true; no_cmed_anchoring = true; }
							else if (!(ws[3] >> p2)) Complain("invalid c parameter for model nfw");
							if (pmode==1) {
								if (!(ws[4] >> p3)) Complain("invalid rt_kpc parameter for model tnfw");
							} else if (pmode==2) {
								if (!(ws[4] >> p3)) Complain("invalid tau parameter for model tnfw");
							}
						} else {
							if (!(ws[2] >> p1)) Complain("invalid ks parameter for model nfw");
							if (!(ws[3] >> p2)) Complain("invalid rs parameter for model nfw");
							if (!(ws[4] >> p3)) Complain("invalid p3 parameter for model tnfw");
						}
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
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 7;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=p2; param_vals[2]=p3; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[5] != "0") or (words[6] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (p1,p2,p3,q,theta) in model tnfw";
								}
								else complain_str = "Must specify vary flags for seven parameters (p1,p2,p3,q,theta,xc,yc) in model tnfw";
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,7,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(TRUNCATED_nfw, emode, zl_in, reference_source_redshift, p1, 0, p2, p3, q, theta, xc, yc, tmode, 0, pmode);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (set_median_concentration) {
								lens_list[nlens-1]->assign_special_anchored_parameters(lens_list[nlens-1],cmed_factor,true);
								if (((vary_parameters) and (vary_flags[1])) or (no_cmed_anchoring)) lens_list[nlens-1]->unassign_special_anchored_parameter(); // we're only setting the initial value for c
							}
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
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
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 7;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=p2; param_vals[2]=p3; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,7,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(CORED_nfw, emode, zl_in, reference_source_redshift, p1, 0, p2, p3, q, theta, xc, yc, 0, 0, pmode);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (set_median_concentration) {
								lens_list[nlens-1]->assign_special_anchored_parameters(lens_list[nlens-1],1,true);
								if ((vary_parameters) and (vary_flags[1])) lens_list[nlens-1]->unassign_special_anchored_parameter(); // we're only setting the initial value for c
							}
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
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
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 6;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=k0; param_vals[1]=R_d; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,6,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(EXPDISK, emode, zl_in, reference_source_redshift, k0, 0, R_d, 0.0, q, theta, xc, yc);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("expdisk requires at least 3 parameteR_d (k0, R_d, q)");
				}
				else if (words[1]=="tophat")
				{
					if (nwords > 8) Complain("more than 6 parameters not allowed for model tophat");
					if (nwords >= 5) {
						double k0, rad;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> k0)) Complain("invalid k0 parameter for model tophat");
						if (!(ws[3] >> rad)) Complain("invalid rad parameter for model tophat");
						if (!(ws[4] >> q)) Complain("invalid q parameter for model tophat");
						if (nwords >= 6) {
							if (!(ws[5] >> theta)) Complain("invalid theta parameter for model tophat");
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
								if (!(ws[6] >> xc)) Complain("invalid x-center parameter for model tophat");
								if (!(ws[7] >> yc)) Complain("invalid y-center parameter for model tophat");
							}
						}
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 6;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=k0; param_vals[1]=rad; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[4] != "0") or (words[5] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for four parameters (k0,rad,q,theta) in model tophat";
								}
								else complain_str = "Must specify vary flags for six parameters (k0,rad,q,theta,xc,yc) in model tophat";
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,6,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(TOPHAT_LENS, emode, zl_in, reference_source_redshift, k0, 0, rad, 0.0, q, theta, xc, yc);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("tophat requires at least 3 parameterad (k0, rad, q)");
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
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 6;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=qx; param_vals[1]=f; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,6,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(filename.c_str(), emode, zl_in, reference_source_redshift, q, theta, qx, f, xc, yc);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
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
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 6;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=ks; param_vals[1]=rs; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,6,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(HERNQUIST, emode, zl_in, reference_source_redshift, ks, 0, rs, 0.0, q, theta, xc, yc);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("hern requires at least 3 parameters (ks, rs, q)");
				}
				else if (words[1]=="corecusp")
				{
					bool set_tidal_host = false;
					// should change format so 'pmode=1' uses the re_param
					int hostnum;
					if (nwords > 12) Complain("more than 10 parameters not allowed for model corecusp");
					// The following is deprecated, because you can just do 'pmode=1', but I'm keeping it in for backwards compatibility
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
						for (int i=0; i < parameter_anchor_i; i++) parameter_anchors[i].shift_down();
					}
					if (nwords >= 8) {
						double p1, gamma, n, a, s;
						double q, theta = 0, xc = 0, yc = 0;
						if (!(ws[2] >> p1)) {
							if (pmode==0) Complain("invalid k0 parameter for model corecusp");
							else Complain("invalid Re parameter for model corecusp");
						}
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
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 9;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=gamma; param_vals[2]=n; param_vals[3]=a; param_vals[4]=s; param_vals[5]=q; param_vals[6]=theta; param_vals[7]=xc; param_vals[8]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[9]=zl_in;
						else param_vals[9]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,9,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(CORECUSP, emode, zl_in, reference_source_redshift, p1, 0, a, s, q, theta, xc, yc, gamma, n, pmode);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (set_tidal_host) {
								lens_list[nlens-1]->assign_special_anchored_parameters(lens_list[hostnum],1,true);
								if ((vary_parameters) and (vary_flags[3])) lens_list[nlens-1]->unassign_special_anchored_parameter(); // we're only setting the initial value for a
							}
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("corecusp requires at least 6 parameters (k0, gamma, n, a, s, q)");
				}
				else if (words[1]=="ptmass")
				{
					if (nwords > 5) Complain("more than 3 parameters not allowed for model ptmass");
					if (nwords >= 3) {
						double p1, xc = 0, yc = 0;
						if (pmode==1) {
							if (!(ws[2] >> p1)) Complain("invalid mtot parameter for model ptmass");
						} else {
							if (!(ws[2] >> p1)) Complain("invalid b parameter for model ptmass");
						}
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
						default_nparams = 3;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=xc; param_vals[2]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[3]=zl_in;
						else param_vals[3]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						} else {
							add_ptmass_lens(zl_in, reference_source_redshift, p1, xc, yc, pmode);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (auto_set_primary_lens) set_primary_lens();
						}
					}
					else Complain("ptmass requires at least 1 parameter, b (Einstein radius)");
				}
				else if (words[1]=="sersic")
				{
					int primary_lens_num;
					if ((pmode < 0) or (pmode > 1)) Complain("parameter mode must be either 0 or 1");
					if (nwords > 9) Complain("more than 7 parameters not allowed for model sersic");
					if (nwords >= 6) {
						double p1, re, n;
						double q, theta = 0, xc = 0, yc = 0;
						int pos;
						if (pmode==1) {
							if (!(ws[2] >> p1)) Complain("invalid mstar parameter for model sersic");
						} else {
							if (!(ws[2] >> p1)) Complain("invalid kappa_e parameter for model sersic");
						}
						if (!(ws[3] >> re)) Complain("invalid R_eff parameter for model sersic");
						if (!(ws[4] >> n)) Complain("invalid n parameter for model sersic");
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
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 7;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=re; param_vals[2]=n; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[7]=zl_in;
						else param_vals[7]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[5] != "0") or (words[6] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (kappa_e,R_eff,n,q,theta) in model sersic (plus optional Fourier modes)";
								}
								else complain_str = "Must specify vary flags for seven parameters (kappa_e,R_eff,n,q,theta,xc,yc) in model sersic (plus optional Fourier modes)";
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,7,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(SERSIC_LENS, emode, zl_in, reference_source_redshift, p1, n, re, 0, q, theta, xc, yc, 0, 0, pmode);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("sersic requires at least 4 parameters (kappa_e, R_eff, n, q)");
				}
				else if (words[1]=="dsersic")
				{
					int primary_lens_num;
					if ((pmode < 0) or (pmode > 1)) Complain("parameter mode must be either 0 or 1");
					if (nwords > 12) Complain("more than 10 parameters not allowed for model dsersic");
					if (nwords >= 9) {
						double p1, delta_k, re1, n1, re2, n2;
						double q, theta = 0, xc = 0, yc = 0;
						int pos;
						if (pmode==1) {
							if (!(ws[2] >> p1)) Complain("invalid mstar parameter for model dsersic");
						} else {
							if (!(ws[2] >> p1)) Complain("invalid kappa0 parameter for model dsersic");
						}
						if (!(ws[3] >> delta_k)) Complain("invalid delta_k parameter for model dsersic");
						if (!(ws[4] >> re1)) Complain("invalid Reff1 parameter for model dsersic");
						if (!(ws[5] >> n1)) Complain("invalid n1 parameter for model dsersic");
						if (!(ws[6] >> re2)) Complain("invalid Reff2 parameter for model dsersic");
						if (!(ws[7] >> n2)) Complain("invalid n2 parameter for model dsersic");
						if (!(ws[8] >> q)) Complain("invalid q parameter for model dsersic");
						if (re1 <= 0) Complain("Reff1 cannot be less than or equal to zero");
						if (re2 <= 0) Complain("Reff2 cannot be less than or equal to zero");
						if (nwords >= 10) {
							if (!(ws[9] >> theta)) Complain("invalid theta parameter for model dsersic");
							if (nwords == 11) {
								if (words[10].find("anchor_center=")==0) {
									string anchorstr = words[10].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate entered for center, but not y-coordinate");
							}
							if (nwords == 12) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[10] >> xc)) Complain("invalid x-center parameter for model dsersic");
								if (!(ws[11] >> yc)) Complain("invalid y-center parameter for model dsersic");
							}
						}
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 10;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=delta_k; param_vals[2]=re1; param_vals[3]=n1; param_vals[4]=re2; param_vals[5]=n2; param_vals[6]=q; param_vals[7]=theta; param_vals[8]=xc; param_vals[9]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[10]=zl_in;
						else param_vals[10]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[8] != "0") or (words[9] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (kappe_e,R_eff,n,q,theta) in model dsersic (plus optional Fourier modes)";
								}
								else complain_str = "Must specify vary flags for seven parameters (kappe_e,R_eff,n,q,theta,xc,yc) in model dsersic (plus optional Fourier modes)";
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,10,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(DOUBLE_SERSIC_LENS, emode, zl_in, reference_source_redshift, p1, n1, re1, re2, q, theta, xc, yc, delta_k, n2, pmode); // weird ordering of parameters in this function. Dislike...
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("dsersic requires at least 7 parameters (kappa0, delta_k, R_eff1, n1, R_eff2, n2, q)");
				}
				else if (words[1]=="csersic")
				{
					int primary_lens_num;
					if ((pmode < 0) or (pmode > 1)) Complain("parameter mode must be either 0 or 1");
					if (nwords > 10) Complain("more than 7 parameters not allowed for model csersic");
					if (nwords >= 7) {
						double p1, re, n, rc;
						double q, theta = 0, xc = 0, yc = 0;
						int pos;
						if (pmode==1) {
							if (!(ws[2] >> p1)) Complain("invalid mstar parameter for model csersic");
						} else {
							if (!(ws[2] >> p1)) Complain("invalid kappe_e parameter for model csersic");
						}
						if (!(ws[3] >> re)) Complain("invalid csersic parameter for model csersic");
						if (!(ws[4] >> n)) Complain("invalid n (core) parameter for model csersic");
						if (!(ws[5] >> rc)) Complain("invalid rc (core) parameter for model csersic");
						if (!(ws[6] >> q)) Complain("invalid q parameter for model csersic");
						if (re <= 0) Complain("re cannot be less than or equal to zero");
						if (nwords >= 8) {
							if (!(ws[7] >> theta)) Complain("invalid theta parameter for model csersic");
							if (nwords == 9) {
								if (words[8].find("anchor_center=")==0) {
									string anchorstr = words[8].substr(14);
									stringstream anchorstream;
									anchorstream << anchorstr;
									if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
									if (anchornum >= nlens) Complain("lens anchor number does not exist");
									anchor_lens_center = true;
								} else Complain("x-coordinate entered for center, but not y-coordinate");
							}
							if (nwords == 10) {
								if ((update_parameters) and (lens_list[lens_number]->center_anchored==true)) Complain("cannot update center point if lens is anchored to another lens");
								if (!(ws[8] >> xc)) Complain("invalid x-center parameter for model csersic");
								if (!(ws[9] >> yc)) Complain("invalid y-center parameter for model csersic");
							}
						}
						if ((LensProfile::use_ellipticity_components) and ((q > 1)  or (theta > 1))) Complain("ellipticity components cannot be greater than 1");

						default_nparams = 8;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=p1; param_vals[1]=re; param_vals[2]=n; param_vals[3]=rc; param_vals[4]=q; param_vals[5]=theta; param_vals[6]=xc; param_vals[7]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[8]=zl_in;
						else param_vals[8]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();

							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								if (anchor_lens_center) {
									if (nwords==tot_nparams_to_vary+2) {
										if ((words[6] != "0") or (words[7] != "0")) complain_str = "center coordinates cannot be varied as free parameters if anchored to another lens";
										else { nparams_to_vary += 2; tot_nparams_to_vary += 2; }
									} else complain_str = "Must specify vary flags for five parameters (kappe_e,R_eff,n,q,theta) in model csersic (plus optional Fourier modes)";
								}
								else complain_str = "Must specify vary flags for eight parameters (kappe_e,R_eff,n,q,theta,xc,yc) in model csersic (plus optional Fourier modes)";
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,8,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");

						if (update_parameters) {
							lens_list[lens_number]->update_parameters(param_vals.array());
						} else {
							create_and_add_lens(CORED_SERSIC_LENS, emode, zl_in, reference_source_redshift, p1, n, re, rc, q, theta, xc, yc, 0, 0, pmode);
							if (egrad) {
								if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
									remove_lens(nlens-1);
									Complain("could not initialize ellipticity gradient; lens object could not be created");
								}
							}
							for (int i=0; i < fourier_nmodes; i++) {
								lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
							}
							if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,lens_list[nlens-1]->get_lensprofile_nparams()+lens_list[nlens-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read fourier gradient parameters");
							if (fgrad) lens_list[nlens-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);

							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
							else if (transform_to_pixsrc_frame) lens_list[nlens-1]->setup_transform_center_coords_to_pixsrc_frame(xc,yc);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
							if (is_perturber) lens_list[nlens-1]->set_perturber(true);
							if (auto_set_primary_lens) set_primary_lens();
							if ((egrad) and (!enter_egrad_params_and_varyflags)) lens_list[nlens-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
						}
					}
					else Complain("csersic requires at least 5 parameters (kappe_e, R_eff, n, rc, q)");
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
						default_nparams = 3;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=kappa; param_vals[1]=xc; param_vals[2]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[3]=zl_in;
						else param_vals[3]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						} else {
							add_mass_sheet_lens(zl_in, reference_source_redshift, kappa, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
						}
					}
					else Complain("sheet requires at least 1 parameter, kappa");
				}
				else if (words[1]=="deflection")
				{
					if (nwords > 4) Complain("more than 2 parameters not allowed for model deflection");
					if (nwords >= 3) {
						double defx, defy;
						if (!(ws[2] >> defx)) Complain("invalid defx parameter for model deflection");
						if (!(ws[3] >> defy)) Complain("invalid defy parameter for model deflection");
						default_nparams = 2;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=defx; param_vals[1]=defy;
						if ((update_zl) or (!update_parameters)) param_vals[2]=zl_in;
						else param_vals[2]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = default_nparams;
							tot_nparams_to_vary = (add_shear) ? nparams_to_vary+2 : nparams_to_vary;
							if (read_command(false)==false) return;
							vary_zl = check_vary_z();
							if (nwords != tot_nparams_to_vary) {
								string complain_str = "";
								complain_str = "Must specify vary flags for three parameters (kappa,xc,yc) in model deflection";
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
						} else {
							create_and_add_lens(DEFLECTION, emode, zl_in, reference_source_redshift, 0, 0, defx, defy, 0, 0, 0, 0);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
							if (vary_parameters) set_lens_vary_parameters(nlens-1,vary_flags);
						}
					}
					else Complain("deflection requires at least two parameters, defx and defy");
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
						default_nparams = 4;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=shear_p1; param_vals[1]=shear_p2; param_vals[2]=xc; param_vals[3]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[4]=zl_in;
						else param_vals[4]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						} else {
							add_shear_lens(zl_in, reference_source_redshift, shear_p1, shear_p2, xc, yc);
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
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
						default_nparams = 5;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=kscale; param_vals[1]=rscale; param_vals[2]=theta; param_vals[3]=xc; param_vals[4]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[5]=zl_in;
						else param_vals[5]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						} else {
							if (tabulate_existing_lens) {
								add_tabulated_lens(zl_in, reference_source_redshift, lnum, kscale, rscale, theta, xc, yc);
							} else {
								if (!add_tabulated_lens_from_file(zl_in, reference_source_redshift, kscale, rscale, theta, xc, yc, filename)) Complain("input file for tabulated model either does not exist, or is in incorrect format");
							}
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
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
						default_nparams = 6;
						param_vals.input(default_nparams+1); // add one for redshift
						for (int i=0; i < parameter_anchor_i; i++) if ((parameter_anchors[i].anchor_object_number==nlens) and (parameter_anchors[i].anchor_paramnum > param_vals.size())) Complain("specified parameter number to anchor to does not exist for given lens");
						param_vals[0]=kscale; param_vals[1]=rscale; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;
						if ((update_zl) or (!update_parameters)) param_vals[6]=zl_in;
						else param_vals[6]=lens_list[lens_number]->zlens;
						if (vary_parameters) {
							nparams_to_vary = (anchor_lens_center) ? default_nparams-2 : default_nparams;
							nparams_to_vary += fourier_nmodes*2;
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
						} else {
							if (qtabulate_existing_lens) {
								add_qtabulated_lens(zl_in, reference_source_redshift, lnum, kscale, rscale, q, theta, xc, yc);
							} else {
								if (!add_qtabulated_lens_from_file(zl_in, reference_source_redshift, kscale, rscale, q, theta, xc, yc, filename)) Complain("input file for qtabulated model either does not exist, or is in incorrect format");
							}
							if (anchor_lens_center) lens_list[nlens-1]->anchor_center_to_lens(anchornum);
							for (int i=0; i < parameter_anchor_i; i++) lens_list[nlens-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,lens_list[parameter_anchors[i].anchor_object_number]);
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
						create_and_add_lens(TESTMODEL, emode, zl_in, reference_source_redshift, 0, 0, 0, 0, q, theta, xc, yc);
						if (egrad) {
							if (lens_list[nlens-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_lens(nlens-1);
								Complain("could not initialize ellipticity gradient; lens object could not be created");
							}
						}
						for (int i=0; i < fourier_nmodes; i++) {
							lens_list[nlens-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if (lensed_center_coords) lens_list[nlens-1]->set_lensed_center(true);
						if (vary_parameters) Complain("vary parameters not supported for testmodel");
						if (is_perturber) lens_list[nlens-1]->set_perturber(true);
						if (auto_set_primary_lens) set_primary_lens();
					}
					else Complain("testmodel requires 4 parameters (q, theta, xc, yc)");
				}
				else Complain("unrecognized lens model");
				if ((vary_parameters) and ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST))) {
					int nvary=0;
					bool enter_limits = true;
					if (vary_zl) nparams_to_vary++;
					for (int i=0; i < nparams_to_vary; i++) if (vary_flags[i]==true) nvary++;
					if (nvary != 0) {
						dvector lower(nvary), upper(nvary), lower_initial(nvary), upper_initial(nvary);
						vector<string> paramnames;
						lens_list[nlens-1]->get_fit_parameter_names(paramnames);
						int i,j;
						for (i=0, j=0; j < nparams_to_vary; j++) {
							if (vary_flags[j]) {
								enter_limits = true;
								if ((egrad) and (egrad_mode==0) and (!enter_egrad_params_and_varyflags)) {
									// It should be possible for the user to enter the egrad limits on a single line, instead of requiring these default values. Implement!
									if ((j >= egrad_qi) and (j < egrad_qf)) {
										lower[i] = 5e-3;
										upper[i] = 1;
										enter_limits = false;
									}
									else if ((j >= egrad_theta_i) and (j < egrad_theta_f)) {
										lower[i] = -180;
										upper[i] = 180;
										enter_limits = false;
									}
									else if ((fgrad) and (j >= fgrad_amp_i) and (j < fgrad_amp_f)) {
										lower[i] = -0.05;
										upper[i] = 0.05;
										enter_limits = false;
									}
									if (!enter_limits) {
										lower_initial[i] = lower[i];
										upper_initial[i] = upper[i];
										i++;
									}
								}
								if (enter_limits) {
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
						}
						lens_list[nlens-1]->set_limits(lower,upper,lower_initial,upper_initial);
					}
				}
				if (add_shear) {
					add_shear_lens(zl_in, reference_source_redshift, shear_param_vals[0], shear_param_vals[1], 0, 0);
					lens_list[nlens-1]->anchor_center_to_lens(nlens-2);
					if (vary_parameters) {
						boolvector shear_vary_flags_extended; // extra field for redshift, which we don't vary by default for external shear
						shear_vary_flags_extended.input(3);
						shear_vary_flags_extended[0] = shear_vary_flags[0];
						shear_vary_flags_extended[1] = shear_vary_flags[1];
						shear_vary_flags_extended[2] = false;
						set_lens_vary_parameters(nlens-1,shear_vary_flags_extended);
					}
					if ((vary_parameters) and ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST))) {
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
			if ((update_parameters) and (lens_list[lens_number]->ellipticity_gradient) and (lens_list[lens_number]->contours_overlap)) warn("density contours overlap for chosen ellipticity gradient parameters");
		}
		else if ((words[0]=="source") or ((words[0]=="fit") and (nwords > 1) and (words[1]=="source")))
		{
			bool update_parameters = false;
			bool update_specific_parameters = false; // option for user to update one (or more) specific parameters rather than update all of them at once
			bool vary_parameters = false;
			bool anchor_source_center = false;
			bool anchor_center_to_lens = false;
			bool anchor_center_to_ptsrc = false;
			int anchornum; // in case new source is being anchored to existing lens
			int pmode = 0;
			int emode = -1;
			bool lensed_center_coords = false;
			bool egrad = false;
			int egrad_mode = -1;
			int n_bspline_coefs = 0; // only used for egrad_mode=0 (B-spline mode)
			double ximin; // only used for egrad_mode=0 (B-spline mode)
			double ximax; // only used for egrad_mode=0 (B-spline mode)
			bool linear_xivals; // only used for egrad_mode=0 (B-spline mode)
			double xiref; // only used for egrad_mode=2 
			bool enter_egrad_params_and_varyflags = true;
			bool enter_knots = false;
			bool fgrad = false;
			int egrad_qi, egrad_qf, egrad_theta_i, egrad_theta_f;
			int fgrad_amp_i, fgrad_amp_f;
			boolvector vary_flags;
			dvector efunc_params, egrad_knots;
			dvector fgrad_params, fgrad_knots;
			vector<string> specific_update_params;
			vector<double> specific_update_param_vals;
			dvector param_vals;
			int default_nparams, nparams_to_vary;
			int src_number;
			vector<int> fourier_mvals;
			vector<double> fourier_Amvals;
			vector<double> fourier_Bmvals;
			int fourier_nmodes = 0;
			bool include_boxiness_parameter = false;
			bool include_truncation_radius = false;
			bool is_lensed = true;
			double zs_in = source_redshift;
			bool zoom = false;
			double c0val = 0;
			double rtval = 0;
			int nmax = -1; // for shapelets and MGE's
			double sig_i = 0.02; // for MGE's
			double sig_f = 3; // for MGE's

			const int nmax_anchor = 100;
			ParamAnchor parameter_anchors[nmax_anchor]; // number of anchors per source can't exceed 100 (which will never happen!)
			int parameter_anchor_i = 0;
			for (int i=0; i < nmax_anchor; i++) parameter_anchors[i].anchor_object_number = n_sb; // by default, param anchors are to parameters within the new source, unless specified otherwise

			for (int i=nwords-1; i > 1; i--) {
				if (words[i]=="-lensed_center") {
					lensed_center_coords = true;
					remove_word(i);
				}
			}

			if (words[0]=="fit") {
				//if ((source_fit_mode != Parameterized_Source) and (source_fit_mode != Shapelet_Source)) Complain("cannot vary parameters for source object unless 'fit source_mode' is set to 'sbprofile' or 'shapelet'");
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
									(profile_name==SERSIC) ? "sersic" :
									(profile_name==CORE_SERSIC) ? "Csersic" :
									(profile_name==CORED_SERSIC) ? "csersic" :
									(profile_name==DOUBLE_SERSIC) ? "dsersic" :
									(profile_name==sple) ? "sple" :
									(profile_name==dpie) ? "dpie" :
									(profile_name==nfw_SOURCE) ? "nfw" : "tophat";
					for (int i=2; i < nwords-1; i++)
						words[i] = words[i+1];
					for (int i=0; i < nwords-1; i++)
						new_ws[i] << words[i];
					words.pop_back();
					nwords--;
					delete[] ws;
					ws = new_ws;
				} else Complain("must specify a source number to update, followed by parameters");
			} else {
				for (int j=nwords-1; j >= 2; j--) {
					if (words[j].find("n=")==0) {
						if (update_parameters) Complain("n=# argument cannot be specified when updating " << words[1]);
						string nstr = words[j].substr(2);
						stringstream nstream;
						nstream << nstr;
						if (!(nstream >> nmax)) Complain("invalid nmax value");
						remove_word(j);
					} else if (words[j].find("si=")==0) {
						if (update_parameters) Complain("si=# argument cannot be specified when updating " << words[1]);
						string astr = words[j].substr(3);
						stringstream astream;
						astream << astr;
						if (!(astream >> sig_i)) Complain("invalid sig_i value");
						remove_word(j);
					} else if (words[j].find("sf=")==0) {
						if (update_parameters) Complain("sf=# argument cannot be specified when updating " << words[1]);
						string astr = words[j].substr(3);
						stringstream astream;
						astream << astr;
						if (!(astream >> sig_f)) Complain("invalid sig_f value");
						remove_word(j);
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

				for (int i=nwords-1; i > 1; i--) {
					int pos;
					if ((pos = words[i].find("egrad=")) != string::npos) {
						string egradstring = words[i].substr(pos+6);
						stringstream egradstr;
						egradstr << egradstring;
						if (!(egradstr >> egrad_mode)) Complain("incorrect format for ellipticity gradient mode; must specify 0 or 1");
						if ((egrad_mode < 0) or (egrad_mode > 1)) Complain("ellipticity gradient mode must be either 0 or 1");
						egrad = true;
						remove_word(i);
					}
				}	
				for (int i=nwords-1; i > 1; i--) {
					if (words[i]=="-fgrad") {
						fgrad = true;
						remove_word(i);
					}
				}

				for (int i=2; i < nwords; i++) {
					int pos0, pos1;
					if ((pos0 = words[i].find("/anchor=")) != string::npos) {
						if ((pos1 = words[i].find("x^")) != string::npos) {
							string pvalstring, expstring, astr;
							pvalstring = words[i].substr(0,pos1);
							expstring = words[i].substr(pos1+2,pos0-pos1-2);
							astr = words[i].substr(pos0+8);
							int pos, snum, pnum;
							double ratio, anchor_exponent;
							stringstream rstr, expstr;
							rstr << pvalstring;
							rstr >> ratio;
							expstr << expstring;
							expstr >> anchor_exponent;
							if ((pos = astr.find(",")) != string::npos) {
								string snumstring, pnumstring;
								snumstring = astr.substr(0,pos);
								pnumstring = astr.substr(pos+1);
								stringstream snumstr, pnumstr;
								snumstr << snumstring;
								if (!(snumstr >> snum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<sb_number>,<param_number>' in place of parameter");
								pnumstr << pnumstring;
								if (!(pnumstr >> pnum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<sb_number>,<param_number>' in place of parameter");
								if (snum > n_sb) Complain("specified source number to anchor to does not exist");
								if ((snum != n_sb) and (pnum >= sb_list[snum]->get_n_params())) Complain("specified parameter number to anchor to does not exist for given source");
								parameter_anchors[parameter_anchor_i].anchor_param = true;
								parameter_anchors[parameter_anchor_i].use_exponent = true;
								parameter_anchors[parameter_anchor_i].paramnum = i-2;
								parameter_anchors[parameter_anchor_i].anchor_object_number = snum;
								parameter_anchors[parameter_anchor_i].anchor_paramnum = pnum;
								parameter_anchors[parameter_anchor_i].ratio = ratio;
								parameter_anchors[parameter_anchor_i].exponent = anchor_exponent;
								parameter_anchor_i++;
								words[i] = pvalstring;
								ws[i].str(""); ws[i].clear();
								ws[i] << words[i];
							} else Complain("incorrect format for anchoring parameter; must type 'anchor=<sb_number>,<param_number>' in place of parameter");
						} else {
							string pvalstring, astr;
							pvalstring = words[i].substr(0,pos0);
							astr = words[i].substr(pos0+8);
							int pos, snum, pnum;
							if ((pos = astr.find(",")) != string::npos) {
								string snumstring, pnumstring;
								snumstring = astr.substr(0,pos);
								pnumstring = astr.substr(pos+1);
								stringstream snumstr, pnumstr;
								snumstr << snumstring;
								if (!(snumstr >> snum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<sb_number>,<param_number>' in place of parameter");
								pnumstr << pnumstring;
								if (!(pnumstr >> pnum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<sb_number>,<param_number>' in place of parameter");
								if (snum > n_sb) Complain("specified source number to anchor to does not exist");
								if ((snum != n_sb) and (pnum >= sb_list[snum]->get_n_params())) Complain("specified parameter number to anchor to does not exist for given source");
								parameter_anchors[parameter_anchor_i].anchor_param = true;
								parameter_anchors[parameter_anchor_i].use_implicit_ratio = true;
								parameter_anchors[parameter_anchor_i].paramnum = i-2;
								parameter_anchors[parameter_anchor_i].anchor_object_number = snum;
								parameter_anchors[parameter_anchor_i].anchor_paramnum = pnum;
								parameter_anchor_i++;
								words[i] = pvalstring;
								ws[i].str(""); ws[i].clear();
								ws[i] << words[i];
							} else Complain("incorrect format for anchoring parameter; must type 'anchor=<sb_number>,<param_number>' in place of parameter");
						}
					}
				}	

				for (int i=2; i < nwords; i++) {
					if (words[i].find("anchor=")==0) {
						string astr = words[i].substr(7);
						int pos, snum, pnum;
						if ((pos = astr.find(",")) != string::npos) {
							string snumstring, pnumstring;
							snumstring = astr.substr(0,pos);
							pnumstring = astr.substr(pos+1);
							stringstream snumstr, pnumstr;
							snumstr << snumstring;
							if (!(snumstr >> snum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<sb_number>,<param_number>' in place of parameter");
							pnumstr << pnumstring;
							if (!(pnumstr >> pnum)) Complain("incorrect format for anchoring parameter; must type 'anchor=<sb_number>,<param_number>' in place of parameter");
							if (snum > n_sb) Complain("specified source number to anchor to does not exist");
							if ((snum != n_sb) and (pnum >= sb_list[snum]->get_n_params())) Complain("specified parameter number to anchor to does not exist for given source");
							parameter_anchors[parameter_anchor_i].anchor_param = true;
							parameter_anchors[parameter_anchor_i].paramnum = i-2;
							parameter_anchors[parameter_anchor_i].anchor_object_number = snum;
							parameter_anchors[parameter_anchor_i].anchor_paramnum = pnum;
							parameter_anchor_i++;
							words[i] = "0";
							ws[i].str(""); ws[i].clear();
							ws[i] << words[i];
						} else Complain("incorrect format for anchoring parameter; must type 'anchor=<sb_number>,<param_number>' in place of parameter");
					}
				}	
			}

			for (int i=nwords-1; i > 1; i--) {
				if (words[i]=="-unlensed") {
					is_lensed = false;
					remove_word(i);
				} else if (words[i]=="-zoom") {
					zoom = true;
					remove_word(i);
				}
			}

			vector<int> remove_list;
			for (int i=3; i < nwords; i++) {
				int pos0;
				if ((words[i][0]=='f') and (pos0 = words[i].find("=")) != string::npos) {
					if (i==nwords-1) Complain("must specify both fourier amplitudes A_m and B_m (e.g. 'f1=0.01 0.02')");
					string mvalstring, amstring, bmstring;
					mvalstring = words[i].substr(1,pos0-1);
					amstring = words[i].substr(pos0+1);
					bmstring = words[i+1];
					int mval;
					double Am, Bm;
					stringstream mstr, astr, bstr;
					mstr << mvalstring;
					astr << amstring;
					bstr << bmstring;
					if (!(mstr >> mval)) Complain("invalid fourier m-value");
					if (!(astr >> Am)) Complain("invalid fourier A_m amplitude");
					if (!(bstr >> Bm)) Complain("invalid fourier B_m amplitude");
					fourier_mvals.push_back(mval);
					fourier_Amvals.push_back(Am);
					fourier_Bmvals.push_back(Bm);
					remove_list.push_back(i);
					remove_list.push_back(i+1);
					//remove_word(i+1);
					//remove_word(i);
					//astr = words[i].substr(pos0+8);
					//int pos, lnum, pnum;
					fourier_nmodes++;
				}
			}
			for (int i=remove_list.size()-1; i >= 0; i--) {
				remove_word(remove_list[i]);
			}

			for (int i=2; i < nwords; i++) {
				if ((words[i][0]=='c') and (words[i][1]=='0') and (words[i][2]=='=') and (!update_parameters)) {
					string c0string;
					c0string = words[i].substr(3);
					stringstream c0str;
					c0str << c0string;
					if (!(c0str >> c0val)) Complain("invalid c0 value");
					remove_word(i);
					include_boxiness_parameter = true;
				}
			}

			for (int i=2; i < nwords; i++) {
				if ((words[i][0]=='r') and (words[i][1]=='t') and (words[i][2]=='=') and (!update_parameters)) {
					string rtstring;
					rtstring = words[i].substr(3);
					stringstream rtstr;
					rtstr << rtstring;
					if (!(rtstr >> rtval)) Complain("invalid rt value");
					remove_word(i);
					include_truncation_radius = true;
				}
			}

			if ((nwords > 1) and (words[1]=="add_fmodes")) {
				if (nwords == 3) {
					if (!(ws[2] >> src_number)) Complain("invalid source number");
					if ((n_sb <= src_number) or (src_number < 0)) Complain("specified source number does not exist");
					if (fourier_nmodes==0) Complain("no Fourier modes have been specified (e.g. 'f1=0.01 0.02 f2=0.0 -0.01', etc.)");
					update_specific_parameters = true;
					nparams_to_vary = fourier_nmodes*2;
					vary_flags.input(nparams_to_vary);
					for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
					if (vary_parameters) {
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for all fourier modes");
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}
					int j = 0;
					// NOTE: when the vary flags are handled this way, it doesn't actually add these to the general parameter list like set_sb_vary_parameters(...) does.
					// Should probably just get the vary flags for that source object, tack on the new Fourier vary flags and then use set_sb_vary_parameters instead.
					for (int i=0; i < fourier_nmodes; i++) {
						sb_list[src_number]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],vary_flags[j],vary_flags[j+1]); // yes, both j++'s should be on this line!
						j += 2;
					}
				} else Complain("must specify a source number to add Fourier modes to, followed by modes");
			}
			if ((nwords > 1) and (words[1]=="remove_fmodes")) {
				if (nwords == 3) {
					if (!(ws[2] >> src_number)) Complain("invalid source number");
					if ((n_sb <= src_number) or (src_number < 0)) Complain("specified source number does not exist");
					if (fourier_nmodes!=0) Complain("cannot remove specific Fourier modes (all modes are removed)");
					update_specific_parameters = true;
					sb_list[src_number]->remove_fourier_modes();
				} else Complain("must specify a source number to remove Fourier modes from");
			}
			//if ((nwords > 1) and (words[1]=="add_rt")) {
				//// This has BIG problems, because if Fourier modes have already been added, it will screw up the order of parameters
				//// that is assumed by the set_geometric_param_pointers() function in sbprofile. Don't use this command for fitting!!!
				//if (nwords == 4) {
					//if (!(ws[2] >> src_number)) Complain("invalid source number");
					//if ((n_sb <= src_number) or (src_number < 0)) Complain("specified source number does not exist");
					//if (!(ws[3] >> rtval)) Complain("invalid rt value");
					//update_specific_parameters = true;
					//// The following code shows up again and again, and should be put in a separate function to reduce repetition
					//nparams_to_vary = 1;
					//vary_flags.input(nparams_to_vary);
					//for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
					//if (vary_parameters) {
						//if (read_command(false)==false) return;
						//if (nwords != nparams_to_vary) Complain("Must specify vary flag for rt");
						//bool invalid_params = false;
						//for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						//if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					//}
					//// NOTE: when the vary flags are handled this way, it doesn't actually add these to the general parameter list like set_sb_vary_parameters(...) does.
					//// Should probably just get the vary flags for that source object, tack on the new vary flag and then use set_sb_vary_parameters instead.
					//sb_list[src_number]->add_truncation_radius(rtval,vary_flags[0]);
				//} else Complain("must specify a source number to add truncation to, followed by rt value");
			//}
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
					if ((sb_list[src_number]->sbtype==SHAPELET) and (fft_convolution)) cleanup_FFT_convolution_arrays(); // shapelet number of amp's may have changed, so redo FFT convolution setup
				}
			}
			else if ((nwords > 1) and ((words[1]=="vary") or (words[1]=="changevary")))
			{
				// At the moment, there is no error checking for changing vary flags of anchored parameters. This should be done from within
				// set_lens_vary_parameters(...), and an integer error code should be returned so specific errors can be printed. Then you should
				// simplify all the error checking in the above code for adding lens models so that errors are printed using the same interface.
				bool set_vary_none = false;
				bool set_vary_all = false;
				if (words[nwords-1]=="none") {
					set_vary_none=true;
					remove_word(nwords-1);
				} else if (words[nwords-1]=="all") {
					set_vary_all=true;
					remove_word(nwords-1);
				}
				if ((nwords==2) and (set_vary_none)) {
					for (int srcnum=0; srcnum < n_sb; srcnum++) {
						int npar = sb_list[srcnum]->n_params;
						boolvector vary_flags(npar);
						for (int i=0; i < npar; i++) vary_flags[i] = false;
						set_sb_vary_parameters(srcnum,vary_flags);
					}
				} else if ((nwords==2) and (set_vary_all)) {
					for (int srcnum=0; srcnum < n_sb; srcnum++) {
						int npar = sb_list[srcnum]->n_params;
						boolvector vary_flags(npar);
						for (int i=0; i < npar; i++) vary_flags[i] = true;
						set_sb_vary_parameters(srcnum,vary_flags);
					}
				} else {
					if (nwords != 3) Complain("one argument required for 'source vary' (src number)");
					int srcnum;
					if (!(ws[2] >> srcnum)) Complain("Invalid src number to change vary parameters");
					if (srcnum >= n_sb) Complain("specified src number does not exist");
					if ((!set_vary_none) and (!set_vary_all)) {
						if (read_command(false)==false) return;
						int nparams_to_vary = nwords;
						boolvector vary_flags(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
						int nparam;
						if (set_sb_vary_parameters(srcnum,vary_flags)==false) {
							int npar = sb_list[srcnum]->n_params;
							Complain("number of vary flags does not match number of parameters (" << npar << ") for specified src");
						}
					} else if (set_vary_none) {
						int npar = sb_list[srcnum]->n_params;
						boolvector vary_flags(npar);
						for (int i=0; i < npar; i++) vary_flags[i] = false;
						set_sb_vary_parameters(srcnum,vary_flags);
					} else if (set_vary_all) {
						int npar = sb_list[srcnum]->n_params;
						boolvector vary_flags(npar);
						for (int i=0; i < npar; i++) vary_flags[i] = true;
						set_sb_vary_parameters(srcnum,vary_flags);
					}
				}
				update_specific_parameters = true;
			}

			if (!update_specific_parameters) {
				for (int i=3; i < nwords; i++) {
					int pos;
					if ((pos = words[i].find("z=")) != string::npos) {
						string znumstring = words[i].substr(pos+2);
						stringstream znumstr;
						znumstr << znumstring;
						if (!(znumstr >> zs_in)) Complain("incorrect format for source redshift");
						if (zs_in < 0) Complain("source redshift cannot be negative");
						remove_word(i);
						i = nwords; // breaks out of this loop, without breaking from outer loop
					}
				}	
			}

			int band = 0;
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("band=")) != string::npos) {
					if (update_specific_parameters) Complain("source band number cannot be updated (remove it and create a new one)");
					string bstring = words[i].substr(pos+5);
					stringstream bstr;
					bstr << bstring;
					if (!(bstr >> band)) Complain("incorrect format for band number");
					if (band < 0) Complain("band number cannot be negative");
					remove_word(i);
					break;
				}
			}	

			if (update_specific_parameters) ; // the updating has already been done, so just continue to the next command
			else if (nwords==1) {
				if (mpi_id==0) print_source_list(vary_parameters);
				vary_parameters = false; // this makes it skip to the next command so it doesn't try to prompt for parameter limits
			}
			else if (words[1]=="clear")
			{
				if (nwords==2) {
					clear_source_objects();
				} else if (nwords==3) {
					//int source_number;
					//if (!(ws[2] >> source_number)) Complain("invalid source number");
					//remove_source_object(source_number);

					int src_number, min_srcnumber, max_srcnumber, pos;
					if ((pos = words[2].find("-")) != string::npos) {
						string srcminstring, srcmaxstring;
						srcminstring = words[2].substr(0,pos);
						srcmaxstring = words[2].substr(pos+1);
						stringstream srcmaxstream, srcminstream;
						srcminstream << srcminstring;
						srcmaxstream << srcmaxstring;
						if (!(srcminstream >> min_srcnumber)) Complain("invalid min source number");
						if (!(srcmaxstream >> max_srcnumber)) Complain("invalid max source number");
						if (max_srcnumber >= n_sb) Complain("specified max source number exceeds number of data sets in list");
						if ((min_srcnumber > max_srcnumber) or (min_srcnumber < 0)) Complain("specified min source number cannot exceed max source number");
						for (int i=max_srcnumber; i >= min_srcnumber; i--) remove_source_object(i);
					} else {
						if (!(ws[2] >> src_number)) Complain("invalid source number");
						remove_source_object(src_number);
					}
				} else Complain("source clear command requires either one or zero arguments");
			}
			else if (words[1]=="zoom")
			{
				if (nwords != 3) Complain("one argument required for 'source zoom' (source number, or 'all')");
				if (words[2]=="all") {
					for (int i=0; i < n_sb; i++) {
						sb_list[i]->set_zoom_subgridding(true);
					}
				} else {
					int src_number;
					if (!(ws[2] >> src_number)) Complain("invalid source number");
					if (src_number >= n_sb) Complain("source number does not exist");
					sb_list[src_number]->set_zoom_subgridding(true);
				}
			}
			else if (words[1]=="unzoom")
			{
				if (nwords != 3) Complain("one argument required for 'source unzoom' (source number)");
				if (words[2]=="all") {
					for (int i=0; i < n_sb; i++) {
						sb_list[i]->set_zoom_subgridding(false);
					}
				} else {
					int src_number;
					if (!(ws[2] >> src_number)) Complain("invalid source number");
					if (src_number >= n_sb) Complain("source number does not exist");
					sb_list[src_number]->set_zoom_subgridding(false);
				}
			}
			else if ((words[1]=="spawn_lens") or (words[1]=="spawn"))
			{
				if (n_sb==0) Complain("no source objects have been created");
				if (nwords < 3) Complain("need at least one argument to 'source spawn_lens' (source number, optional mass parameter value)");
				int src_number;
				if (!(ws[2] >> src_number)) Complain("invalid source number");
				if (src_number >= n_sb) Complain("source number does not exist");
				double mass_param = -1e30;
				if (nwords > 3) {
					if (!(ws[3] >> mass_param)) Complain("invalid mass parameter value");
				}
				bool include_lims = false;
				double minpar=-1e30, maxpar=1e30;
				if ((vary_parameters) and ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST))) {
					include_lims = true;
					if (mpi_id==0) cout << "Prior limits for mass parameter of spawned lens:" << endl;
					if (read_command(false)==false) Complain("could not read prior limits for mass parameter of spawned lens");
					if (nwords != 2) Complain("must give two arguments: lower prior limit and upper prior limit");
					if (!(ws[0] >> minpar)) Complain("invalid lower limit");
					if (!(ws[1] >> maxpar)) Complain("invalid upper limit");
				}
				if (!spawn_lens_from_source_object(src_number,lens_redshift,source_redshift,pmode,vary_parameters,include_lims,minpar,maxpar)) Complain("Lens spawning failed");
				if (mass_param > 0) lens_list[nlens-1]->update_specific_parameter(0,mass_param);
				vary_parameters = false; // This is so it doesn't try to set limits later down in the code, like it does for regular source models
			}
			else if (words[1]=="gaussian")
			{
				if (nwords > 8) Complain("more than 7 parameters not allowed for model gaussian");
				if (nwords >= 5) {
					double sbmax, sig;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> sbmax)) Complain("invalid max surface brightness parameter for model gaussian");
					if (!(ws[3] >> sig)) Complain("invalid sigma parameter for model gaussian");
					if (!(ws[4] >> q)) Complain("invalid q parameter for model gaussian");
					if (nwords >= 6) {
						if (!(ws[5] >> theta)) Complain("invalid theta parameter for model gaussian");
						if (nwords == 7) {
							if (words[6].find("anchor_center=")==0) {
								string anchorstr = words[6].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[6].find("anchor_lens_center=")==0) {
								string anchorstr = words[6].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[6].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[6].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							}

						} else if (nwords == 8) {
							if (!(ws[6] >> xc)) Complain("invalid x-center parameter for model gaussian");
							if (!(ws[7] >> yc)) Complain("invalid y-center parameter for model gaussian");
						}
					}
					default_nparams = 6;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=sbmax; param_vals[1]=sig; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;

					if (vary_parameters) {
						if (include_boxiness_parameter) nparams_to_vary++;
						if (include_truncation_radius) nparams_to_vary++;
						nparams_to_vary += fourier_nmodes*2;
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for six parameters (sbmax,sigma,q,theta,xc,yc) in model gaussian, plus optional c0/rfsc parameter or fourier modes");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,default_nparams,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						add_source_object(GAUSSIAN, is_lensed, band, zs_in, emode, sbmax, sig, 0, 0, q, theta, xc, yc);
						if (egrad) {
							if (sb_list[n_sb-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_source_object(n_sb-1);
								Complain("could not initialize ellipticity gradient; source object could not be created");
							}
						}
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						for (int i=0; i < fourier_nmodes; i++) {
							sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,sb_list[n_sb-1]->get_sbprofile_nparams()+sb_list[n_sb-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read Fourier gradient parameters");
						if (fgrad) sb_list[n_sb-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
						if (include_boxiness_parameter) sb_list[n_sb-1]->add_boxiness_parameter(c0val,false);
						if (include_truncation_radius) sb_list[n_sb-1]->add_truncation_radius(rtval,false);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (lensed_center_coords) sb_list[n_sb-1]->set_lensed_center(true);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if ((egrad) and (!enter_egrad_params_and_varyflags)) sb_list[n_sb-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
					}
				}
				else Complain("gaussian requires at least 3 parameters (sbmax, sig, q)");
			}
			else if (words[1]=="shapelet")
			{
				bool truncate = false;
				double amp00 = 0.1;
				bool amp_specified = false;
				for (int j=nwords-1; j >= 2; j--) {
					if (words[j]=="-truncate") {
						truncate = true;
						remove_word(j);
					} else if (words[j].find("amp0=")==0) {
						if (update_parameters) Complain("amp0=# argument cannot be specified when updating " << words[1]);
						amp_specified = true;
						string astr = words[j].substr(5);
						stringstream astream;
						astream << astr;
						if (!(astream >> amp00)) Complain("invalid shapelet m=0 amplitude value");
						remove_word(j);
					}
				}
				int pi = 2;
				if (nmax == -1) Complain("must specify nmax via 'n=#' argument");
				if (nwords > 7) Complain("more than 5 parameters not allowed for model shapelet");
				if (nmax <= 0) Complain("nmax cannot be negative");
				if (nwords >= 5) {
					double scale;
					double q, theta = 0, xc = 0, yc = 0;
					if (pmode==0) {
						if (!(ws[pi++] >> scale)) Complain("invalid sigma parameter for model shapelet");
					} else {
						if (!(ws[pi++] >> scale)) Complain("invalid sigfac parameter for model shapelet");
					}
					if (!(ws[pi++] >> q)) Complain("invalid q parameter for model shapelet");
					if (nwords >= (pi+1)) {
						if (!(ws[pi++] >> theta)) Complain("invalid theta parameter for model shapelet");
						if (nwords == (pi+1)) {
							if (words[pi].find("anchor_center=")==0) {
								string anchorstr = words[pi].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[pi].find("anchor_lens_center=")==0) {
								string anchorstr = words[pi].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[pi].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[pi].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							} else Complain("must specify both xc and yc, or 'anchor_center=#' if anchoring, for model Shapelet");
						} else if (nwords == (pi+2)) {
							if (!(ws[pi++] >> xc)) Complain("invalid x-center parameter for model shapelet");
							if (!(ws[pi++] >> yc)) Complain("invalid y-center parameter for model shapelet");
						}
					}
					default_nparams = 5;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					//param_vals[0]=amp00; // currently cannot vary amp00 as a free parameter (it would have to be removed from the source amplitudes when inverting)
					int indx=0;
					param_vals[indx++]=scale;
					param_vals[indx++]=q;
					param_vals[indx++]=theta;
					param_vals[indx++]=xc;
					param_vals[indx++]=yc;

					if (vary_parameters) {
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for five parameters (sigma,q,theta,xc,yc) in model shapelet, plus optional c0/rfsc parameter or fourier modes");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
						if (fft_convolution) cleanup_FFT_convolution_arrays(); // since number of shapelet amplitudes may have changed, will redo FFT setup here
					} else {
						add_shapelet_source(is_lensed, band, zs_in, amp00, scale, q, theta, xc, yc, nmax, truncate, pmode);
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) {
							if (set_sb_vary_parameters(n_sb-1,vary_flags)==false) Complain("could not vary parameters for model shapelet");
						}
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
					}
				}
				else Complain("shapelet requires at least 3 parameters (amp00, sig, q)");
			}
			else if (words[1]=="mge")
			{
				double amp0 = 10.0;
				int pi = 2;
				if (nmax == -1) Complain("must specify nmax via 'n=#' argument");
				if (nwords > 7) Complain("more than 6 parameters not allowed for model mge");
				if (nmax <= 0) Complain("nmax cannot be negative");
				if (nwords >= 5) {
					double reg, q, theta = 0, xc = 0, yc = 0;
					if (!(ws[pi++] >> reg)) Complain("invalid regularization parameter for model mge");
					if (!(ws[pi++] >> q)) Complain("invalid q parameter for model mge");
					if (nwords >= (pi+1)) {
						if (!(ws[pi++] >> theta)) Complain("invalid theta parameter for model mge");
						if (nwords == (pi+1)) {
							if (words[pi].find("anchor_center=")==0) {
								string anchorstr = words[pi].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[pi].find("anchor_lens_center=")==0) {
								string anchorstr = words[pi].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[pi].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[pi].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							} else Complain("must specify both xc and yc, or 'anchor_center=#' if anchoring, for model Shapelet");
						} else if (nwords == (pi+2)) {
							if (!(ws[pi++] >> xc)) Complain("invalid x-center parameter for model mge");
							if (!(ws[pi++] >> yc)) Complain("invalid y-center parameter for model mge");
						}
					}
					default_nparams = 5;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					//param_vals[0]=amp0; // currently cannot vary amp0 as a free parameter (it would have to be removed from the source amplitudes when inverting)
					int indx=0;
					param_vals[indx++]=reg;
					param_vals[indx++]=q;
					param_vals[indx++]=theta;
					param_vals[indx++]=xc;
					param_vals[indx++]=yc;

					if (vary_parameters) {
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for four parameters (q,theta,xc,yc) in model mge, plus optional c0/rfsc parameter or fourier modes");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
						if (fft_convolution) cleanup_FFT_convolution_arrays(); // since number of mge amplitudes may have changed, will redo FFT setup here
					} else {
						add_mge_source(is_lensed, band, zs_in, reg, amp0, sig_i, sig_f, q, theta, xc, yc, nmax, pmode);
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (vary_parameters) {
							if (set_sb_vary_parameters(n_sb-1,vary_flags)==false) Complain("could not vary parameters for model mge");
						}
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if ((!is_lensed) and (!include_fgmask_in_inversion)) {
							include_fgmask_in_inversion = true;
							if (mpi_id==0) cout << "NOTE: Setting 'include_fgmask_in_inversion' to 'on', since foreground MGE is assumed to use foreground mask" << endl;
						}
					}
				}
				else Complain("mge requires at least 5 parameters (sig_i, sig_f, regparam, q)");
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
						if (nwords == 8) {
							if (words[7].find("anchor_center=")==0) {
								string anchorstr = words[7].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[7].find("anchor_lens_center=")==0) {
								string anchorstr = words[7].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[7].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[7].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							}
						} else if (nwords == 9) {
							if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model sersic");
							if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model sersic");
						}
					}

					default_nparams = 7;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=s0; param_vals[1]=reff; param_vals[2] = n; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;

					if (vary_parameters) {
						if (include_boxiness_parameter) nparams_to_vary++;
						if (include_truncation_radius) nparams_to_vary++;
						nparams_to_vary += fourier_nmodes*2;
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for seven parameters (s0,Reff,n,q,theta,xc,yc) in model sersic (plus optional Fourier modes)");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,default_nparams,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						add_source_object(SERSIC, is_lensed, band, zs_in, emode, s0, reff, 0, n, q, theta, xc, yc, 0, 0, pmode);
						if (egrad) {
							if (sb_list[n_sb-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_source_object(n_sb-1);
								Complain("could not initialize ellipticity gradient; source object could not be created");
							}
						}
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						for (int i=0; i < fourier_nmodes; i++) {
							sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,sb_list[n_sb-1]->get_sbprofile_nparams()+sb_list[n_sb-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read Fourier gradient parameters");
						if (fgrad) sb_list[n_sb-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
						if (include_boxiness_parameter) sb_list[n_sb-1]->add_boxiness_parameter(c0val,false);
						if (include_truncation_radius) sb_list[n_sb-1]->add_truncation_radius(rtval,false);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (lensed_center_coords) sb_list[n_sb-1]->set_lensed_center(true);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if ((egrad) and (!enter_egrad_params_and_varyflags)) sb_list[n_sb-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
					}
				}
				else Complain("sersic requires at least 4 parameters (max_sb, Reff, n, q)");
			}
			else if (words[1]=="Csersic")
			{
				if (nwords > 12) Complain("more than 10 parameters not allowed for model Csersic");
				if (nwords >= 9) {
					double s0, reff, n, rc, alpha, gamma;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> s0)) Complain("invalid s0 parameter for model Csersic");
					if (!(ws[3] >> reff)) Complain("invalid R_eff parameter for model Csersic");
					if (!(ws[4] >> n)) Complain("invalid n parameter for model Csersic");
					if (!(ws[5] >> rc)) Complain("invalid rc parameter for model Csersic");
					if (!(ws[6] >> gamma)) Complain("invalid gamma parameter for model Csersic");
					if (!(ws[7] >> alpha)) Complain("invalid alpha parameter for model Csersic");
					if (!(ws[8] >> q)) Complain("invalid q parameter for model Csersic");
					if (nwords >= 10) {
						if (!(ws[9] >> theta)) Complain("invalid theta parameter for model Csersic");
						if (nwords == 11) {
							if (words[10].find("anchor_center=")==0) {
								string anchorstr = words[8].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[10].find("anchor_lens_center=")==0) {
								string anchorstr = words[10].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[10].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[10].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							}
						} else if (nwords == 12) {
							if (!(ws[10] >> xc)) Complain("invalid x-center parameter for model Csersic");
							if (!(ws[11] >> yc)) Complain("invalid y-center parameter for model Csersic");
						}
					}

					default_nparams = 10;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=s0; param_vals[1]=reff; param_vals[2] = n; param_vals[3] = rc; param_vals[4] = gamma; param_vals[5] = alpha; param_vals[6]=q; param_vals[7]=theta; param_vals[8]=xc; param_vals[9]=yc;

					if (vary_parameters) {
						if (include_boxiness_parameter) nparams_to_vary++;
						if (include_truncation_radius) nparams_to_vary++;
						nparams_to_vary += fourier_nmodes*2;
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for ten parameters (s0,Reff,n,rc,gamma,alpha,q,theta,xc,yc) in model Csersic");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,default_nparams,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						add_source_object(CORE_SERSIC, is_lensed, band, zs_in, emode, s0, reff, rc, n, q, theta, xc, yc, gamma, alpha);
						if (egrad) {
							if (sb_list[n_sb-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_source_object(n_sb-1);
								Complain("could not initialize ellipticity gradient; source object could not be created");
							}
						}
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						for (int i=0; i < fourier_nmodes; i++) {
							sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,sb_list[n_sb-1]->get_sbprofile_nparams()+sb_list[n_sb-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read Fourier gradient parameters");
						if (fgrad) sb_list[n_sb-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
						if (include_boxiness_parameter) sb_list[n_sb-1]->add_boxiness_parameter(c0val,false);
						if (include_truncation_radius) sb_list[n_sb-1]->add_truncation_radius(rtval,false);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (lensed_center_coords) sb_list[n_sb-1]->set_lensed_center(true);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if ((egrad) and (!enter_egrad_params_and_varyflags)) sb_list[n_sb-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
					}
				}
				else Complain("Csersic requires at least 5 parameters (max_sb, k, n, rc, q)");
			}
			else if (words[1]=="csersic")
			{
				if (nwords > 10) Complain("more than 8 parameters not allowed for model csersic");
				if (nwords >= 7) {
					double s0, reff, n, rc;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> s0)) Complain("invalid s0 parameter for model csersic");
					if (!(ws[3] >> reff)) Complain("invalid R_eff parameter for model csersic");
					if (!(ws[4] >> n)) Complain("invalid n parameter for model csersic");
					if (!(ws[5] >> rc)) Complain("invalid rc parameter for model csersic");
					if (!(ws[6] >> q)) Complain("invalid q parameter for model csersic");
					if (nwords >= 8) {
						if (!(ws[7] >> theta)) Complain("invalid theta parameter for model csersic");
						if (nwords == 9) {
							if (words[8].find("anchor_center=")==0) {
								string anchorstr = words[8].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[8].find("anchor_lens_center=")==0) {
								string anchorstr = words[8].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[8].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[8].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							}
						} else if (nwords == 10) {
							if (!(ws[8] >> xc)) Complain("invalid x-center parameter for model csersic");
							if (!(ws[9] >> yc)) Complain("invalid y-center parameter for model csersic");
						}
					}

					default_nparams = 8;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=s0; param_vals[1]=reff; param_vals[2] = n; param_vals[3] = rc; param_vals[4]=q; param_vals[5]=theta; param_vals[6]=xc; param_vals[7]=yc;

					if (vary_parameters) {
						if (include_boxiness_parameter) nparams_to_vary++;
						if (include_truncation_radius) nparams_to_vary++;
						nparams_to_vary += fourier_nmodes*2;
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for eight parameters (s0,Reff,n,rc,q,theta,xc,yc) in model csersic (plus optional Fourier modes)");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,default_nparams,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						add_source_object(CORED_SERSIC, is_lensed, band, zs_in, emode, s0, reff, rc, n, q, theta, xc, yc);

						if (egrad) {
							if (sb_list[n_sb-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_source_object(n_sb-1);
								Complain("could not initialize ellipticity gradient; source object could not be created");
							}
						}
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						for (int i=0; i < fourier_nmodes; i++) {
							sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,sb_list[n_sb-1]->get_sbprofile_nparams()+sb_list[n_sb-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read Fourier gradient parameters");
						if (fgrad) sb_list[n_sb-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
						if (include_boxiness_parameter) sb_list[n_sb-1]->add_boxiness_parameter(c0val,false);
						if (include_truncation_radius) sb_list[n_sb-1]->add_truncation_radius(rtval,false);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (lensed_center_coords) sb_list[n_sb-1]->set_lensed_center(true);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if ((egrad) and (!enter_egrad_params_and_varyflags)) sb_list[n_sb-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
					}
				}
				else Complain("csersic requires at least 5 parameters (max_sb, k, n, rc, q)");
			}
			else if (words[1]=="dsersic")
			{
				if (nwords > 12) Complain("more than 10 parameters not allowed for model dsersic");
				if (nwords >= 9) {
					double s0, ds, reff1, n1, reff2, n2;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> s0)) Complain("invalid s0 parameter for model dsersic");
					if (!(ws[3] >> ds)) Complain("invalid delta_s parameter for model dsersic");
					if (!(ws[4] >> reff1)) Complain("invalid R_eff1 parameter for model dsersic");
					if (!(ws[5] >> n1)) Complain("invalid n1 parameter for model dsersic");
					if (!(ws[6] >> reff2)) Complain("invalid R_eff2 parameter for model dsersic");
					if (!(ws[7] >> n2)) Complain("invalid n2 parameter for model dsersic");
					if (!(ws[8] >> q)) Complain("invalid q parameter for model dsersic");
					if (nwords >= 10) {
						if (!(ws[9] >> theta)) Complain("invalid theta parameter for model dsersic");
						if (nwords == 11) {
							if (words[10].find("anchor_center=")==0) {
								string anchorstr = words[8].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[10].find("anchor_lens_center=")==0) {
								string anchorstr = words[10].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[10].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[10].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							}
						} else if (nwords == 12) {
							if (!(ws[10] >> xc)) Complain("invalid x-center parameter for model dsersic");
							if (!(ws[11] >> yc)) Complain("invalid y-center parameter for model dsersic");
						}
					}

					default_nparams = 10;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=s0; param_vals[1]=ds; param_vals[2] = reff1; param_vals[3] = n1; param_vals[4] = reff2; param_vals[5] = n2; param_vals[6]=q; param_vals[7]=theta; param_vals[8]=xc; param_vals[9]=yc;

					if (vary_parameters) {
						if (include_boxiness_parameter) nparams_to_vary++;
						if (include_truncation_radius) nparams_to_vary++;
						nparams_to_vary += fourier_nmodes*2;
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for ten parameters (s0_1,Reff1,n1,s0_2,Reff2,n2,q,theta,xc,yc) in model dsersic");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,default_nparams,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						add_source_object(DOUBLE_SERSIC, is_lensed, band, zs_in, emode, s0, reff1, reff2, ds, q, theta, xc, yc, n1, n2); // super-awkward to use the "add_source_object" function for this....
						if (egrad) {
							if (sb_list[n_sb-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_source_object(n_sb-1);
								Complain("could not initialize ellipticity gradient; source object could not be created");
							}
						}
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						for (int i=0; i < fourier_nmodes; i++) {
							sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,sb_list[n_sb-1]->get_sbprofile_nparams()+sb_list[n_sb-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read Fourier gradient parameters");
						if (fgrad) sb_list[n_sb-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
						if (include_boxiness_parameter) sb_list[n_sb-1]->add_boxiness_parameter(c0val,false);
						if (include_truncation_radius) sb_list[n_sb-1]->add_truncation_radius(rtval,false);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (lensed_center_coords) sb_list[n_sb-1]->set_lensed_center(true);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if ((egrad) and (!enter_egrad_params_and_varyflags)) sb_list[n_sb-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
					}
				}
				else Complain("dsersic requires at least 7 parameters (s0, ds, Reff1, n1, Reff2, n2, q)");
			}
			else if (words[1]=="sple")
			{
				if (nwords > 9) Complain("more than 8 parameters not allowed for model sple");
				if (nwords >= 6) {
					double bs, s, alpha;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> bs)) Complain("invalid bs parameter for model sple");
					if (!(ws[3] >> alpha)) Complain("invalid alpha parameter for model sple");
					if (!(ws[4] >> s)) Complain("invalid s parameter for model sple");
					if (!(ws[5] >> q)) Complain("invalid q parameter for model sple");
					if (nwords >= 7) {
						if (!(ws[6] >> theta)) Complain("invalid theta parameter for model sple");
						if (nwords == 8) {
							if (words[7].find("anchor_center=")==0) {
								string anchorstr = words[7].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[7].find("anchor_lens_center=")==0) {
								string anchorstr = words[7].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[7].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[7].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							}
						} else if (nwords == 9) {
							if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model sple");
							if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model sple");
						}
					}

					default_nparams = 7;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=bs; param_vals[1]=alpha; param_vals[2] = s; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;

					if (vary_parameters) {
						if (include_boxiness_parameter) nparams_to_vary++;
						if (include_truncation_radius) nparams_to_vary++;
						nparams_to_vary += fourier_nmodes*2;
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for seven parameters (s0,Reff,n,q,theta,xc,yc) in model sple (plus optional Fourier modes)");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,default_nparams,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						add_source_object(sple, is_lensed, band, zs_in, emode, bs, s, 0, alpha, q, theta, xc, yc);
						if (egrad) {
							if (sb_list[n_sb-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_source_object(n_sb-1);
								Complain("could not initialize ellipticity gradient; source object could not be created");
							}
						}
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						for (int i=0; i < fourier_nmodes; i++) {
							sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,sb_list[n_sb-1]->get_sbprofile_nparams()+sb_list[n_sb-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read Fourier gradient parameters");
						if (fgrad) sb_list[n_sb-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
						if (include_boxiness_parameter) sb_list[n_sb-1]->add_boxiness_parameter(c0val,false);
						if (include_truncation_radius) sb_list[n_sb-1]->add_truncation_radius(rtval,false);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (lensed_center_coords) sb_list[n_sb-1]->set_lensed_center(true);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if ((egrad) and (!enter_egrad_params_and_varyflags)) sb_list[n_sb-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
					}
				}
				else Complain("sple requires at least 4 parameters (bs, alpha, s, q)");
			}
			else if (words[1]=="dpie")
			{
				if (nwords > 9) Complain("more than 8 parameters not allowed for model dpie");
				if (nwords >= 6) {
					double bs, s, a;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> bs)) Complain("invalid bs parameter for model dpie");
					if (!(ws[3] >> a)) Complain("invalid a parameter for model dpie");
					if (!(ws[4] >> s)) Complain("invalid s parameter for model dpie");
					if (!(ws[5] >> q)) Complain("invalid q parameter for model dpie");
					if (nwords >= 7) {
						if (!(ws[6] >> theta)) Complain("invalid theta parameter for model dpie");
						if (nwords == 8) {
							if (words[7].find("anchor_center=")==0) {
								string anchorstr = words[7].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[7].find("anchor_lens_center=")==0) {
								string anchorstr = words[7].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[7].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[7].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							}
						} else if (nwords == 9) {
							if (!(ws[7] >> xc)) Complain("invalid x-center parameter for model dpie");
							if (!(ws[8] >> yc)) Complain("invalid y-center parameter for model dpie");
						}
					}

					default_nparams = 7;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=bs; param_vals[1]=a; param_vals[2] = s; param_vals[3]=q; param_vals[4]=theta; param_vals[5]=xc; param_vals[6]=yc;

					if (vary_parameters) {
						if (include_boxiness_parameter) nparams_to_vary++;
						if (include_truncation_radius) nparams_to_vary++;
						nparams_to_vary += fourier_nmodes*2;
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for seven parameters (s0,Reff,n,q,theta,xc,yc) in model dpie (plus optional Fourier modes)");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,default_nparams,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						add_source_object(dpie, is_lensed, band, zs_in, emode, bs, a, s, 0, q, theta, xc, yc);
						if (egrad) {
							if (sb_list[n_sb-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_source_object(n_sb-1);
								Complain("could not initialize ellipticity gradient; source object could not be created");
							}
						}
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						for (int i=0; i < fourier_nmodes; i++) {
							sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,sb_list[n_sb-1]->get_sbprofile_nparams()+sb_list[n_sb-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read Fourier gradient parameters");
						if (fgrad) sb_list[n_sb-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
						if (include_boxiness_parameter) sb_list[n_sb-1]->add_boxiness_parameter(c0val,false);
						if (include_truncation_radius) sb_list[n_sb-1]->add_truncation_radius(rtval,false);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (lensed_center_coords) sb_list[n_sb-1]->set_lensed_center(true);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if ((egrad) and (!enter_egrad_params_and_varyflags)) sb_list[n_sb-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
					}
				}
				else Complain("dpie requires at least 4 parameters (bs,a,s,q)");
			}
			else if (words[1]=="nfw")
			{
				if (nwords > 8) Complain("more than 7 parameters not allowed for model nfw");
				if (nwords >= 5) {
					double s0, rs;
					double q, theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> s0)) Complain("invalid s0 parameter for model nfw");
					if (!(ws[3] >> rs)) Complain("invalid rs parameter for model nfw");
					if (!(ws[4] >> q)) Complain("invalid q parameter for model nfw");
					if (nwords >= 6) {
						if (!(ws[5] >> theta)) Complain("invalid theta parameter for model nfw");
						if (nwords == 7) {
							if (words[6].find("anchor_center=")==0) {
								string anchorstr = words[6].substr(14);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid source number for source to anchor to");
								if (anchornum >= n_sb) Complain("source anchor number does not exist");
								anchor_source_center = true;
							} else if (words[6].find("anchor_lens_center=")==0) {
								string anchorstr = words[6].substr(19);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid lens number for lens to anchor to");
								if (anchornum >= nlens) Complain("lens anchor number does not exist");
								anchor_center_to_lens = true;
							} else if (words[6].find("anchor_ptsrc_center=")==0) {
								string anchorstr = words[7].substr(20);
								stringstream anchorstream;
								anchorstream << anchorstr;
								if (!(anchorstream >> anchornum)) Complain("invalid ptsrc number for ptsrc to anchor to");
								if (anchornum >= n_ptsrc) Complain("ptsrc anchor number does not exist");
								anchor_center_to_ptsrc = true;
							}
						} else if (nwords == 8) {
							if (!(ws[6] >> xc)) Complain("invalid x-center parameter for model nfw");
							if (!(ws[7] >> yc)) Complain("invalid y-center parameter for model nfw");
						}
					}

					default_nparams = 6;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=s0; param_vals[1]=rs; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;

					if (vary_parameters) {
						if (include_boxiness_parameter) nparams_to_vary++;
						if (include_truncation_radius) nparams_to_vary++;
						nparams_to_vary += fourier_nmodes*2;
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for seven parameters (s0,Reff,n,q,theta,xc,yc) in model nfw (plus optional Fourier modes)");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,default_nparams,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						add_source_object(nfw_SOURCE, is_lensed, band, zs_in, emode, s0, rs, 0, 0, q, theta, xc, yc);
						if (egrad) {
							if (sb_list[n_sb-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_source_object(n_sb-1);
								Complain("could not initialize ellipticity gradient; source object could not be created");
							}
						}
						if (anchor_source_center) sb_list[n_sb-1]->anchor_center_to_source(sb_list,anchornum);
						else if (anchor_center_to_lens) sb_list[n_sb-1]->anchor_center_to_lens(lens_list,anchornum);
						else if (anchor_center_to_ptsrc) sb_list[n_sb-1]->anchor_center_to_ptsrc(ptsrc_list,anchornum);
						for (int i=0; i < fourier_nmodes; i++) {
							sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,sb_list[n_sb-1]->get_sbprofile_nparams()+sb_list[n_sb-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read Fourier gradient parameters");
						if (fgrad) sb_list[n_sb-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
						if (include_boxiness_parameter) sb_list[n_sb-1]->add_boxiness_parameter(c0val,false);
						if (include_truncation_radius) sb_list[n_sb-1]->add_truncation_radius(rtval,false);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (lensed_center_coords) sb_list[n_sb-1]->set_lensed_center(true);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if ((egrad) and (!enter_egrad_params_and_varyflags)) sb_list[n_sb-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
					}
				}
				else Complain("nfw requires at least 3 parameters (s0,rs,q)");
			}


			else if (words[1]=="sbmpole")
			{
				bool sine_term = false;
				if (nwords > 9) Complain("more than 8 arguments not allowed for model " << words[1]);
				if (nwords >= 4) {
					double a_m, r0;
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
					}
					double theta = 0, xc = 0, yc = 0;
					if (!(ws[2] >> a_m)) Complain("invalid a_m parameter for model " << words[1]);
					if (!(ws[3] >> r0)) {
						Complain("invalid r0 parameter for model " << words[1]);
					}
					if (r0 == 2-m) Complain("for sbmpole, r0 cannot be equal to 2-m"); // check if this is sensible?
					if (nwords >= 5) {
						if (!(ws[4] >> theta)) Complain("invalid theta parameter for model " << words[1]);
						else if (nwords == 7) {
							if (!(ws[5] >> xc)) Complain("invalid x-center parameter for model " << words[1]);
							if (!(ws[6] >> yc)) Complain("invalid y-center parameter for model " << words[1]);
						}
					}
					default_nparams = 5;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=a_m; param_vals[1]=r0; param_vals[2]=theta; param_vals[3]=xc; param_vals[4]=yc;

					if (vary_parameters) {
						if (read_command(false)==false) return;
						if (nwords != nparams_to_vary) Complain("Must specify vary flags for five parameters (A_m,r0,theta,xc,yc) in model sbmpole");
						vary_flags.input(nparams_to_vary);
						bool invalid_params = false;
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) invalid_params = true;
						if (invalid_params==true) Complain("Invalid vary flag (must specify 0 or 1)");
					}

					if (update_parameters) {
						sb_list[src_number]->update_parameters(param_vals.array());
					} else {
						add_multipole_source(is_lensed, band, zs_in, m, a_m, r0, theta, xc, yc, sine_term);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
					}
				}
				else Complain("sbmpole requires at least 2 parameters (a_m, r0)");
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

					default_nparams = 6;
					nparams_to_vary = default_nparams;
					param_vals.input(nparams_to_vary);
					param_vals[0]=sb; param_vals[1]=rad; param_vals[2]=q; param_vals[3]=theta; param_vals[4]=xc; param_vals[5]=yc;

					if (vary_parameters) {
						nparams_to_vary += fourier_nmodes*2;
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
						if ((egrad) and (!read_egrad_params(vary_parameters,egrad_mode,efunc_params,nparams_to_vary,vary_flags,default_nparams,xc,yc,parameter_anchors,parameter_anchor_i,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read ellipticity gradient parameters");
						add_source_object(TOPHAT, is_lensed, band, zs_in, emode, sb, rad, 0, 0, q, theta, xc, yc);
						if (egrad) {
							if (sb_list[n_sb-1]->enable_ellipticity_gradient(efunc_params,egrad_mode,n_bspline_coefs,egrad_knots,ximin,ximax,xiref,linear_xivals)==false) {
								remove_source_object(n_sb-1);
								Complain("could not initialize ellipticity gradient; source object could not be created");
							}
						}
						for (int i=0; i < fourier_nmodes; i++) {
							sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
						}
						if ((fgrad) and (!read_fgrad_params(vary_parameters,egrad_mode,fourier_nmodes,fourier_mvals,fgrad_params,nparams_to_vary,vary_flags,sb_list[n_sb-1]->get_sbprofile_nparams()+sb_list[n_sb-1]->get_egrad_nparams(),parameter_anchors,parameter_anchor_i,n_bspline_coefs,fgrad_knots,enter_egrad_params_and_varyflags,enter_knots))) Complain("could not read Fourier gradient parameters");
						if (fgrad) sb_list[n_sb-1]->enable_fourier_gradient(fgrad_params,fgrad_knots);
						for (int i=0; i < parameter_anchor_i; i++) sb_list[n_sb-1]->assign_anchored_parameter(parameter_anchors[i].paramnum,parameter_anchors[i].anchor_paramnum,parameter_anchors[i].use_implicit_ratio,parameter_anchors[i].use_exponent,parameter_anchors[i].ratio,parameter_anchors[i].exponent,sb_list[parameter_anchors[i].anchor_object_number]);
						if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
						if (vary_parameters) set_sb_vary_parameters(n_sb-1,vary_flags);
						if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
						if (lensed_center_coords) sb_list[n_sb-1]->set_lensed_center(true);
						if ((egrad) and (!enter_egrad_params_and_varyflags)) sb_list[n_sb-1]->find_egrad_paramnums(egrad_qi,egrad_qf,egrad_theta_i,egrad_theta_f,fgrad_amp_i,fgrad_amp_f);
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
					add_source_object(words[2].c_str(), is_lensed, band, zs_in, emode, q, theta, qx, f, xc, yc);
					for (int i=0; i < fourier_nmodes; i++) {
						sb_list[n_sb-1]->add_fourier_mode(fourier_mvals[i],fourier_Amvals[i],fourier_Bmvals[i],false,false);
					}
					if (!is_lensed) sb_list[n_sb-1]->set_lensed(false);
					if (zoom) sb_list[n_sb-1]->set_zoom_subgridding(true);
				}
				else Complain("spline requires at least 2 parameters (filename, q)");
			}
			else Complain("source model not recognized");
			if ((vary_parameters) and ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST))) {
				int nvary=0;
				bool enter_limits = true;
				for (int i=0; i < nparams_to_vary; i++) if (vary_flags[i]==true) nvary++;
				if (nvary != 0) {
					dvector lower(nvary), upper(nvary), lower_initial(nvary), upper_initial(nvary);
					vector<string> paramnames;
					sb_list[n_sb-1]->get_fit_parameter_names(paramnames);
					int i,j;
					for (i=0, j=0; j < nparams_to_vary; j++) {
						if (vary_flags[j]) {
							enter_limits = true;
							if ((egrad) and (egrad_mode==0) and (!enter_egrad_params_and_varyflags)) {
								// It should be possible for the user to enter the egrad limits on a single line, instead of requiring these default values. Implement!
								if ((j >= egrad_qi) and (j < egrad_qf)) {
									lower[i] = 5e-3;
									upper[i] = 1;
									enter_limits = false;
								}
								else if ((j >= egrad_theta_i) and (j < egrad_theta_f)) {
									lower[i] = -180;
									upper[i] = 180;
									enter_limits = false;
								}
								else if ((fgrad) and (j >= fgrad_amp_i) and (j < fgrad_amp_f)) {
									lower[i] = -0.05;
									upper[i] = 0.05;
									enter_limits = false;
								}
								if (!enter_limits) {
									lower_initial[i] = lower[i];
									upper_initial[i] = upper[i];
									i++;
								}
							}
							if (enter_limits) {
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
					}
					sb_list[n_sb-1]->set_limits(lower,upper,lower_initial,upper_initial);
				}
			}
			if ((update_parameters) and (sb_list[src_number]->ellipticity_gradient) and (sb_list[src_number]->contours_overlap)) warn("surface brightness contours overlap for chosen ellipticity gradient parameters");
		}
		else if ((words[0]=="pixsrc") or ((words[0]=="fit") and (nwords > 1) and (words[1]=="pixsrc")))
		{
			bool update_specific_parameters = false; // option for user to update one (or more) specific parameters rather than update all of them at once
			bool vary_parameters = false;
			vector<string> specific_update_params;
			vector<double> specific_update_param_vals;
			int pixsrc_number = -1;
			bool entered_varyflags = false;
			bool added_new_pixsrc = false;
			bool prompt_for_flags = true;
			int nparams_to_vary = nwords;
			boolvector vary_flags;

			if (words[0]=="fit") {
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

			if ((nwords > 1) and (words[1]=="update")) update_specific_parameters = true;

			double zsrc = source_redshift;
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("z=")) != string::npos) {
					if (update_specific_parameters) Complain("pixellated source redshift cannot be updated (remove it and create a new one)");
					string znumstring = words[i].substr(pos+2);
					stringstream znumstr;
					znumstr << znumstring;
					if (!(znumstr >> zsrc)) Complain("incorrect format for source redshift");
					if (zsrc < 0) Complain("source redshift cannot be negative");
					remove_word(i);
					break;
				}
			}	
			int band = 0;
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("band=")) != string::npos) {
					if (update_specific_parameters) Complain("pixellated source band number cannot be updated (remove it and create a new one)");
					string bstring = words[i].substr(pos+5);
					stringstream bstr;
					bstr << bstring;
					if (!(bstr >> band)) Complain("incorrect format for band number");
					if (band < 0) Complain("band number cannot be negative");
					remove_word(i);
					break;
				}
			}	


			if (update_specific_parameters) {
				if (nwords > 2) {
					if (!(ws[2] >> pixsrc_number)) Complain("invalid pixellated source number");
					if ((n_pixellated_src <= pixsrc_number) or (pixsrc_number < 0)) Complain("specified pixellated source number does not exist");
					update_specific_parameters = true;
					// Now we'll remove the "update" word
					stringstream* new_ws = new stringstream[nwords-1];
					for (int i=1; i < nwords-1; i++)
						words[i] = words[i+1];
					for (int i=0; i < nwords-1; i++)
						new_ws[i] << words[i];
					words.pop_back();
					nwords--;
					delete[] ws;
					ws = new_ws;
				} else Complain("must specify a pixsrc number to update, followed by parameters");
			}

			if (update_specific_parameters) {
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
					} 
				}
				if (n_updates > 0) {
					for (int i=0; i < n_updates; i++)
						if (srcgrids[pixsrc_number]->update_specific_parameter(specific_update_params[i],specific_update_param_vals[i])==false) Complain("could not find parameter '" << specific_update_params[i] << "' in pixellated source " << pixsrc_number);
				}
			} else if ((nwords > 1) and ((words[1]=="vary") or (words[1]=="changevary"))) {
				vary_parameters = true;
				bool set_vary_none = false;
				bool set_vary_all = false;
				if (words[nwords-1]=="none") {
					set_vary_none=true;
					remove_word(nwords-1);
				}
				if (words[nwords-1]=="all") {
					set_vary_all=true;
					remove_word(nwords-1);
				}
				if ((nwords==2) and (set_vary_none)) {
					for (int pixsrcnum=0; pixsrcnum < n_pixellated_src; pixsrcnum++) {
						nparams_to_vary = srcgrids[pixsrcnum]->n_active_params;
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
						set_pixellated_src_vary_parameters(pixsrcnum,vary_flags);
					}
					entered_varyflags = true;
					pixsrc_number = -1; // so it prompts for limits for all pixellated sources
				} else if ((nwords==2) and (set_vary_all)) {
					for (int pixsrcnum=0; pixsrcnum < n_pixellated_src; pixsrcnum++) {
						nparams_to_vary = srcgrids[pixsrcnum]->n_active_params;
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = true;
						set_pixellated_src_vary_parameters(pixsrcnum,vary_flags);
					}
					entered_varyflags = true;
					pixsrc_number = -1; // so it prompts for limits for all pixellated sources
				} else if (nwords==2) {
					pixsrc_number = -1; // so it prompts for limits for all pixellated sources
				} else {
					if (nwords != 3) Complain("one argument required for 'pixsrc vary' (pixsrc number)");
					if (!(ws[2] >> pixsrc_number)) Complain("Invalid pixsrc number to change vary parameters");
					if (pixsrc_number >= n_pixellated_src) Complain("specified pixsrc number does not exist");
					nparams_to_vary = srcgrids[pixsrc_number]->n_active_params;
					if ((!set_vary_none) and (!set_vary_all)) {
						if (read_command(false)==false) return;
						int nparams_entered = nwords;
						if (nparams_entered != nparams_to_vary) Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified pixsrc");
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
						if (set_pixellated_src_vary_parameters(pixsrc_number,vary_flags)==false) {
							Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified pixsrc");
						}
						entered_varyflags = true;
					} else if (set_vary_none) {
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
						set_pixellated_src_vary_parameters(pixsrc_number,vary_flags);
						entered_varyflags = true;
					} else if (set_vary_all) {
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = true;
						set_pixellated_src_vary_parameters(pixsrc_number,vary_flags);
						entered_varyflags = true;
					}
				}
			} else if (nwords==1) {
				if (mpi_id==0) print_pixellated_source_list(vary_parameters);
				vary_parameters = false; // don't make it prompt for vary flags if they only put 'fit pixsrc'
			} else {
				if (words[1]=="clear") {
					if (nwords==2) {
						while (n_pixellated_src > 0) {
							remove_pixellated_source(n_pixellated_src-1);
						}
					}
					else if (nwords==3) {
						int pixsrc_number;
						if (!(ws[2] >> pixsrc_number)) Complain("invalid pixsrc number");
						remove_pixellated_source(pixsrc_number);
					} else Complain("only one argument allowed for 'pixsrc clear' (number of pixellated source to remove)");
				}
				else if (words[1]=="add") {
					add_pixellated_source(zsrc,band);
					pixsrc_number = n_pixellated_src-1; // for setting vary flags (below)
					added_new_pixsrc = true;

				}
				else Complain("unrecognized argument to 'pixsrc'");
				update_parameter_list();
			}
			if (vary_parameters) {
				int nvary;
				int pixsrcnum, pixsrcnum_i, pixsrcnum_f;
				bool print_line_for_each_source = false;
				if (pixsrc_number < 0) {
					// in this case, prompt for limits for all pixellated sources
					pixsrcnum_i = 0;
					pixsrcnum_f = n_pixellated_src;
					print_line_for_each_source = true;
				} else {
					pixsrcnum_i = pixsrc_number;
					pixsrcnum_f = pixsrc_number+1;
				}
				for (pixsrcnum=pixsrcnum_i; pixsrcnum < pixsrcnum_f; pixsrcnum++) {
					if ((prompt_for_flags) and (!entered_varyflags)) {
						nparams_to_vary = srcgrids[pixsrcnum]->n_active_params;
						if ((mpi_id==0) and (print_line_for_each_source)) cout << "Vary flags for pixellated source " << pixsrcnum << ":" << endl;
						if (read_command(false)==false) return;
						int nparams_entered = nwords;
						if (nparams_entered != nparams_to_vary) Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified pixsrc");
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
						if (set_pixellated_src_vary_parameters(pixsrcnum,vary_flags)==false) {
							Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified pixsrc");
						}
					}
					if ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST)) {
						nvary = srcgrids[pixsrcnum]->n_vary_params;
						srcgrids[pixsrcnum]->get_varyflags(vary_flags);
						if (nvary != 0) {
							dvector lower(nvary), upper(nvary), lower_initial(nvary), upper_initial(nvary);
							vector<string> paramnames;
							srcgrids[pixsrcnum]->get_fit_parameter_names(paramnames);
							int i,j;
							for (i=0, j=0; j < nparams_to_vary; j++) {
								if (vary_flags[j]) {
									if ((mpi_id==0) and (verbal_mode)) cout << "limits for parameter " << paramnames[i] << ":\n";
									if (read_command(false)==false) { if (added_new_pixsrc) remove_pixellated_source(pixsrcnum); Complain("parameter limits could not be read"); }
									if (nwords >= 2) {
										if (!(ws[0] >> lower[i])) { if (added_new_pixsrc) remove_pixellated_source(pixsrcnum); Complain("invalid lower limit"); }
										if (!(ws[1] >> upper[i])) { if (added_new_pixsrc) remove_pixellated_source(pixsrcnum); Complain("invalid upper limit"); }
										if (nwords == 2) {
											lower_initial[i] = lower[i];
											upper_initial[i] = upper[i];
										} else if (nwords == 4) {
											if (!(ws[2] >> lower_initial[i])) { if (added_new_pixsrc) remove_pixellated_source(pixsrcnum); Complain("invalid initial lower limit"); }
											if (!(ws[3] >> upper_initial[i])) { if (added_new_pixsrc) remove_pixellated_source(pixsrcnum); Complain("invalid initial upper limit"); }
										} else {
											if (added_new_pixsrc) remove_pixellated_source(pixsrcnum);
											Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
										}
									} else {
										if (added_new_pixsrc) remove_pixellated_source(pixsrcnum);
										Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
									}
									if (lower_initial[i] < lower[i]) lower_initial[i] = lower[i];
									if (upper_initial[i] > upper[i]) upper_initial[i] = upper[i];
									i++;
								}
							}
							srcgrids[pixsrcnum]->set_limits(lower,upper,lower_initial,upper_initial);
						}
					}
				}
			}
		}
		else if ((words[0]=="pixlens") or ((words[0]=="fit") and (nwords > 1) and (words[1]=="pixlens")))
		{
			bool update_specific_parameters = false; // option for user to update one (or more) specific parameters rather than update all of them at once
			bool vary_parameters = false;
			vector<string> specific_update_params;
			vector<double> specific_update_param_vals;
			int pixlens_number = -1;
			bool entered_varyflags = false;
			bool added_new_pixlens = false;
			bool prompt_for_flags = true;
			int nparams_to_vary = nwords;
			boolvector vary_flags;

			if (words[0]=="fit") {
				vary_parameters = true;
				// now remove the "fit" word from the line so we can add pixellated lenses the same way,
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

			if ((nwords > 1) and (words[1]=="update")) update_specific_parameters = true;

			double zlens = lens_redshift;
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("z=")) != string::npos) {
					if (update_specific_parameters) Complain("pixellated lens redshift cannot be updated (remove it and create a new one)");
					string znumstring = words[i].substr(pos+2);
					stringstream znumstr;
					znumstr << znumstring;
					if (!(znumstr >> zlens)) Complain("incorrect format for lens redshift");
					if (zlens < 0) Complain("lens redshift cannot be negative");
					remove_word(i);
					break;
				}
			}	

			if (update_specific_parameters) {
				if (nwords > 2) {
					if (!(ws[2] >> pixlens_number)) Complain("invalid pixellated lens number");
					if ((n_pixellated_lens <= pixlens_number) or (pixlens_number < 0)) Complain("specified pixellated lens number does not exist");
					update_specific_parameters = true;
					// Now we'll remove the "update" word
					stringstream* new_ws = new stringstream[nwords-1];
					for (int i=1; i < nwords-1; i++)
						words[i] = words[i+1];
					for (int i=0; i < nwords-1; i++)
						new_ws[i] << words[i];
					words.pop_back();
					nwords--;
					delete[] ws;
					ws = new_ws;
				} else Complain("must specify a pixlens number to update, followed by parameters");
			}

			if (update_specific_parameters) {
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
					} 
				}
				if (n_updates > 0) {
					for (int i=0; i < n_updates; i++)
						if (lensgrids[pixlens_number]->update_specific_parameter(specific_update_params[i],specific_update_param_vals[i])==false) Complain("could not find parameter '" << specific_update_params[i] << "' in pixellated lens " << pixlens_number);
				}
			} else if ((nwords > 1) and ((words[1]=="vary") or (words[1]=="changevary"))) {
				vary_parameters = true;
				bool set_vary_none = false;
				bool set_vary_all = false;
				if (words[nwords-1]=="none") {
					set_vary_none=true;
					remove_word(nwords-1);
				}
				if (words[nwords-1]=="all") {
					set_vary_all=true;
					remove_word(nwords-1);
				}
				if ((nwords==2) and (set_vary_none)) {
					for (int pixlensnum=0; pixlensnum < n_pixellated_lens; pixlensnum++) {
						nparams_to_vary = lensgrids[pixlensnum]->n_active_params;
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
						set_pixellated_lens_vary_parameters(pixlensnum,vary_flags);
					}
					entered_varyflags = true;
					pixlens_number = -1; // so it prompts for limits for all pixellated lenses
				} else if ((nwords==2) and (set_vary_all)) {
					for (int pixlensnum=0; pixlensnum < n_pixellated_lens; pixlensnum++) {
						nparams_to_vary = lensgrids[pixlensnum]->n_active_params;
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = true;
						set_pixellated_lens_vary_parameters(pixlensnum,vary_flags);
					}
					entered_varyflags = true;
					pixlens_number = -1; // so it prompts for limits for all pixellated lenses
				} else if (nwords==2) {
					pixlens_number = -1; // so it prompts for limits for all pixellated lenses
				} else {
					if (nwords != 3) Complain("one argument required for 'pixlens vary' (pixlens number)");
					if (!(ws[2] >> pixlens_number)) Complain("Invalid pixlens number to change vary parameters");
					if (pixlens_number >= n_pixellated_lens) Complain("specified pixlens number does not exist");
					nparams_to_vary = lensgrids[pixlens_number]->n_active_params;
					if ((!set_vary_none) and (!set_vary_all)) {
						if (read_command(false)==false) return;
						int nparams_entered = nwords;
						if (nparams_entered != nparams_to_vary) Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified pixsrc");
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
						if (set_pixellated_lens_vary_parameters(pixlens_number,vary_flags)==false) {
							Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified pixlens");
						}
						entered_varyflags = true;
					} else if (set_vary_none) {
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
						set_pixellated_lens_vary_parameters(pixlens_number,vary_flags);
						entered_varyflags = true;
					} else if (set_vary_all) {
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = true;
						set_pixellated_lens_vary_parameters(pixlens_number,vary_flags);
						entered_varyflags = true;
					}
				}
			} else if (nwords==1) {
				if (mpi_id==0) print_pixellated_lens_list(vary_parameters);
				vary_parameters = false; // don't make it prompt for vary flags if they only put 'fit pixlens'
			} else {
				if (words[1]=="clear") {
					if (nwords==2) {
						while (n_pixellated_lens > 0) {
							remove_pixellated_lens(n_pixellated_lens-1);
						}
					}
					else if (nwords==3) {
						int pixlens_number;
						if (!(ws[2] >> pixlens_number)) Complain("invalid pixlens number");
						remove_pixellated_lens(pixlens_number);
					} else Complain("only one argument allowed for 'pixlens clear' (number of pixellated lens to remove)");
				}
				else if (words[1]=="add") {
					add_pixellated_lens(zlens);
					pixlens_number = n_pixellated_lens-1; // for setting vary flags (below)
					added_new_pixlens = true;
					int nx=100, ny=100; // default
					if (nwords > 2) {
						if (!(ws[2] >> nx)) Complain("invalid nx");
						if (nwords > 3) {
							if (!(ws[3] >> ny)) Complain("invalid ny");
						} else {
							ny = nx;
						}
					}
					lensgrids[n_pixellated_lens-1]->set_cartesian_npixels(nx,ny);
					/*
					// creating grid now for testing purposes
					double grid_xmin, grid_xmax, grid_ymin, grid_ymax;
					grid_xmin = grid_xcenter - grid_xlength/2;
					grid_xmax = grid_xcenter + grid_xlength/2;
					grid_ymin = grid_ycenter - grid_ylength/2;
					grid_ymax = grid_ycenter + grid_ylength/2;
					lensgrids[n_pixellated_lens-1]->create_cartesian_pixel_grid(grid_xmin,grid_xmax,grid_ymin,grid_ymax,source_redshift);
					if (mpi_id==0) cout << "Created cartesian lens pixel grid" << endl;
					*/
				}
				else if ((words[1]=="assign_from_lens") or (words[1]=="add_from_lens")) {
					bool add_potential = false;
					if (words[1]=="add_from_lens") add_potential = true;
					int pixlens_num = 0;
					int lens_num = 0;
					int pos;
					for (int i=2; i < nwords; i++) {
						if ((pos = words[i].find("lens=")) != string::npos) {
							string lensnumstring = words[i].substr(pos+5);
							stringstream lensnumstr;
							lensnumstr << lensnumstring;
							if (!(lensnumstr >> lens_num)) Complain("incorrect format for lens number");
							if (lens_num < 0) Complain("lens index cannot be negative");
							if (lens_num >= nlens) Complain("specified lens has not been created");
							remove_word(i);
							break;
						}
					}
					if (nwords==3) {
						if (!(ws[2] >> pixlens_num)) Complain("invalid pixlens number");
					}
					if (pixlens_num >= n_pixellated_lens) Complain("specified pixlens has not been created");
					if (lensgrids[pixlens_num]->n_gridpts==0) {
						//int nx=100, ny=100;
						//if (nwords > 3) {
							//if (!(ws[3] >> nx)) Complain("invalid nx");
							//if (nwords > 4) {
								//if (!(ws[4] >> ny)) Complain("invalid ny");
							//} else {
								//ny = nx;
							//}
						//}
						//lensgrids[n_pixellated_lens-1]->set_cartesian_npixels(nx,ny);

						// creating grid now for testing purposes
						double grid_xmin, grid_xmax, grid_ymin, grid_ymax;
						grid_xmin = grid_xcenter - grid_xlength/2;
						grid_xmax = grid_xcenter + grid_xlength/2;
						grid_ymin = grid_ycenter - grid_ylength/2;
						grid_ymax = grid_ycenter + grid_ylength/2;
						lensgrids[n_pixellated_lens-1]->create_cartesian_pixel_grid(grid_xmin,grid_xmax,grid_ymin,grid_ymax,0);
						if (mpi_id==0) cout << "Created cartesian lens pixel grid" << endl;
						//create_lensgrid_cartesian(0,0,true);
					}
					lensgrids[pixlens_num]->assign_potential_from_analytic_lens(lens_num,add_potential);
				}
				else if ((words[1]=="plotpot") or (words[1]=="plotkappa")) {
					bool plot_kappa = false;
					if (words[1]=="plotkappa") plot_kappa = true;
					int pixlens_num = 0;
					int npix = 600;
					bool interpolate = false;
					vector<string> args;
					if (extract_word_starts_with('-',2,nwords-1,args)==true)
					{
						for (int i=0; i < args.size(); i++) {
							if (args[i]=="-interp") interpolate = true;
							else if (args[i]=="-p100") npix = 100;
							else if (args[i]=="-p200") npix = 200;
							else if (args[i]=="-p300") npix = 300;
							else if (args[i]=="-p400") npix = 400;
							else if (args[i]=="-p500") npix = 500;
							else if (args[i]=="-p600") npix = 600;
							else if (args[i]=="-p700") npix = 700;
							else if (args[i]=="-p800") npix = 800;
							else if (args[i]=="-p1000") npix = 1000;
							else if (args[i]=="-p2000") npix = 2000;
							else if (args[i]=="-p3000") npix = 3000;
							else if (args[i]=="-p4000") npix = 4000;
							else Complain("argument '" << args[i] << "' not recognized");
						}
					}

					if (nwords==3) {
						if (!(ws[2] >> pixlens_num)) Complain("invalid pixlens number");
					}
					if (pixlens_num >= n_pixellated_lens) Complain("specified pixlens has not been created");
					if (!plot_kappa) lensgrids[pixlens_num]->plot_potential("pot_pixel",npix,interpolate,false,false);
					else lensgrids[pixlens_num]->plot_potential("pot_pixel",npix,interpolate,true,false);
					string range = "";
					if (show_cc) run_plotter_range("potpixel","",range);
					else run_plotter_range("potpixel_nocc","",range);
				}
				else Complain("unrecognized argument to 'pixlens'");
				update_parameter_list();
			}
			if (vary_parameters) {
				int nvary;
				int pixlensnum, pixlensnum_i, pixlensnum_f;
				bool print_line_for_each_lens = false;
				if (pixlens_number < 0) {
					// in this case, prompt for limits for all pixellated lenses
					pixlensnum_i = 0;
					pixlensnum_f = n_pixellated_lens;
					print_line_for_each_lens = true;
				} else {
					pixlensnum_i = pixlens_number;
					pixlensnum_f = pixlens_number+1;
				}
				for (pixlensnum=pixlensnum_i; pixlensnum < pixlensnum_f; pixlensnum++) {
					if ((prompt_for_flags) and (!entered_varyflags)) {
						nparams_to_vary = lensgrids[pixlensnum]->n_active_params;
						if ((mpi_id==0) and (print_line_for_each_lens)) cout << "Vary flags for pixellated source " << pixlensnum << ":" << endl;
						if (read_command(false)==false) return;
						int nparams_entered = nwords;
						if (nparams_entered != nparams_to_vary) Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified pixlens");
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
						if (set_pixellated_lens_vary_parameters(pixlensnum,vary_flags)==false) {
							Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified pixlens");
						}
					}
					if ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST)) {
						nvary = lensgrids[pixlensnum]->n_vary_params;
						lensgrids[pixlensnum]->get_varyflags(vary_flags);
						if (nvary != 0) {
							dvector lower(nvary), upper(nvary), lower_initial(nvary), upper_initial(nvary);
							vector<string> paramnames;
							lensgrids[pixlensnum]->get_fit_parameter_names(paramnames);
							int i,j;
							for (i=0, j=0; j < nparams_to_vary; j++) {
								if (vary_flags[j]) {
									if ((mpi_id==0) and (verbal_mode)) cout << "limits for parameter " << paramnames[i] << ":\n";
									if (read_command(false)==false) { if (added_new_pixlens) remove_pixellated_lens(pixlensnum); Complain("parameter limits could not be read"); }
									if (nwords >= 2) {
										if (!(ws[0] >> lower[i])) { if (added_new_pixlens) remove_pixellated_lens(pixlensnum); Complain("invalid lower limit"); }
										if (!(ws[1] >> upper[i])) { if (added_new_pixlens) remove_pixellated_lens(pixlensnum); Complain("invalid upper limit"); }
										if (nwords == 2) {
											lower_initial[i] = lower[i];
											upper_initial[i] = upper[i];
										} else if (nwords == 4) {
											if (!(ws[2] >> lower_initial[i])) { if (added_new_pixlens) remove_pixellated_lens(pixlensnum); Complain("invalid initial lower limit"); }
											if (!(ws[3] >> upper_initial[i])) { if (added_new_pixlens) remove_pixellated_lens(pixlensnum); Complain("invalid initial upper limit"); }
										} else {
											if (added_new_pixlens) remove_pixellated_lens(pixlensnum);
											Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
										}
									} else {
										if (added_new_pixlens) remove_pixellated_lens(pixlensnum);
										Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
									}
									if (lower_initial[i] < lower[i]) lower_initial[i] = lower[i];
									if (upper_initial[i] > upper[i]) upper_initial[i] = upper[i];
									i++;
								}
							}
							lensgrids[pixlensnum]->set_limits(lower,upper,lower_initial,upper_initial);
						}
					}
				}
			}
		}
		else if ((words[0]=="ptsrc") or ((words[0]=="fit") and (nwords > 1) and ((words[1]=="ptsrc") or (words[1]=="sourcept")))) // "sourcept" is the old command, so included here for backwards compatibility
		{
			bool update_specific_parameters = false; // option for user to update one (or more) specific parameters rather than update all of them at once
			bool vary_parameters = false;
			vector<string> specific_update_params;
			vector<double> specific_update_param_vals;
			int ptsrc_number = -1;
			bool lensed = true;
			bool make_imgdata = false;
			bool entered_varyflags = false;
			bool added_new_ptsrc = false;
			bool prompt_for_flags = true;
			int nparams_to_vary = nwords;
			boolvector vary_flags;

			if (words[0]=="fit") {
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

			vector<string> args;
			if (extract_word_starts_with('-',2,nwords-1,args)==true)
			{
				int pos;
				for (int i=0; i < args.size(); i++) {
					if (args[i]=="-add_unlensed") { lensed = false; }
					if (args[i]=="-limits") { prompt_for_flags = false; }
					else if (args[i]=="-mkimgdata") { make_imgdata = true; }
					else Complain("argument '" << args[i] << "' not recognized");
				}
			}

			if ((nwords > 1) and (words[1]=="update")) update_specific_parameters = true;

			double zsrc = (lensed) ? source_redshift : lens_redshift;
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("z=")) != string::npos) {
					if (update_specific_parameters) Complain("point source redshift cannot be updated (remove it and create a new one)");
					string znumstring = words[i].substr(pos+2);
					stringstream znumstr;
					znumstr << znumstring;
					if (!(znumstr >> zsrc)) Complain("incorrect format for source redshift");
					if (zsrc < 0) Complain("source redshift cannot be negative");
					remove_word(i);
					break;
				}
			}	

			if (update_specific_parameters) {
				if (nwords > 2) {
					if (!(ws[2] >> ptsrc_number)) Complain("invalid point source number");
					if ((n_ptsrc <= ptsrc_number) or (ptsrc_number < 0)) Complain("specified point source number does not exist");
					update_specific_parameters = true;
					// Now we'll remove the "update" word
					stringstream* new_ws = new stringstream[nwords-1];
					for (int i=1; i < nwords-1; i++)
						words[i] = words[i+1];
					for (int i=0; i < nwords-1; i++)
						new_ws[i] << words[i];
					words.pop_back();
					nwords--;
					delete[] ws;
					ws = new_ws;
				} else Complain("must specify a ptsrc number to update, followed by parameters");
			}

			if (make_imgdata) add_image_data_from_unlensed_sourcepts(false);
			else if (update_specific_parameters) {
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
					} 
				}
				if (n_updates > 0) {
					for (int i=0; i < n_updates; i++)
						if (ptsrc_list[ptsrc_number]->update_specific_parameter(specific_update_params[i],specific_update_param_vals[i])==false) Complain("could not find parameter '" << specific_update_params[i] << "' in point source " << ptsrc_number);
				}
			} else if ((nwords > 1) and ((words[1]=="vary") or (words[1]=="changevary"))) {
				vary_parameters = true;
				bool set_vary_none = false;
				bool set_vary_all = false;
				if (words[nwords-1]=="none") {
					set_vary_none=true;
					remove_word(nwords-1);
				}
				if (words[nwords-1]=="all") {
					set_vary_all=true;
					remove_word(nwords-1);
				}
				if ((nwords==2) and (set_vary_none)) {
					for (int ptsrcnum=0; ptsrcnum < n_ptsrc; ptsrcnum++) {
						nparams_to_vary = ptsrc_list[ptsrcnum]->n_active_params;
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
						set_ptsrc_vary_parameters(ptsrcnum,vary_flags);
					}
					entered_varyflags = true;
					ptsrc_number = -1; // so it prompts for limits for all point sources
				} else if ((nwords==2) and (set_vary_all)) {
					for (int ptsrcnum=0; ptsrcnum < n_ptsrc; ptsrcnum++) {
						nparams_to_vary = ptsrc_list[ptsrcnum]->n_active_params;
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = true;
						set_ptsrc_vary_parameters(ptsrcnum,vary_flags);
					}
					entered_varyflags = true;
					ptsrc_number = -1; // so it prompts for limits for all point sources
				} else if (nwords==2) {
					ptsrc_number = -1; // so it prompts for limits for all point sources
				} else {
					if (nwords != 3) Complain("one argument required for 'ptsrc vary' (ptsrc number)");
					if (!(ws[2] >> ptsrc_number)) Complain("Invalid ptsrc number to change vary parameters");
					if (ptsrc_number >= n_ptsrc) Complain("specified ptsrc number does not exist");
					nparams_to_vary = ptsrc_list[ptsrc_number]->n_active_params;

					if ((!set_vary_none) and (!set_vary_all)) {
						if (read_command(false)==false) return;
						int nparams_entered = nwords;
						if (nparams_entered != nparams_to_vary) Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified ptsrc");
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
						if (set_ptsrc_vary_parameters(ptsrc_number,vary_flags)==false) {
							Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified ptsrc");
						}
						entered_varyflags = true;
					} else if (set_vary_none) {
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
						set_ptsrc_vary_parameters(ptsrc_number,vary_flags);
						entered_varyflags = true;
					} else if (set_vary_all) {
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = true;
						set_ptsrc_vary_parameters(ptsrc_number,vary_flags);
						entered_varyflags = true;
					}
				}
			} else if (nwords==1) {
				if (mpi_id==0) print_point_source_list(vary_parameters);
				vary_parameters = false; // don't make it prompt for vary flags if they only put 'fit ptsrc'
			} else {
				if (words[1]=="clear") {
					if (nwords==2) {
						while (n_ptsrc > 0) {
							remove_point_source(n_ptsrc-1);
						}
					}
					else if (nwords==3) {
						int ptsrc_number;
						if (!(ws[2] >> ptsrc_number)) Complain("invalid ptsrc number");
						remove_point_source(ptsrc_number);
					} else Complain("only one argument allowed for 'ptsrc clear' (number of point source to remove)");
				}
				else if (words[1]=="auto") {
					if (nwords > 2) Complain("no arguments allowed to 'ptsrc auto'");
					set_analytic_sourcepts(true);
					if ((include_flux_chisq) and (analytic_source_flux)) set_analytic_srcflux(true);
					vary_parameters = false;
				}
				else if (words[1]=="add") {
					lensvector srcpt;
					if (nwords==2) {
					srcpt[0] = 0; // add options to put in arguments for srcx, srcy
					srcpt[1] = 0; // likewise
					} else if (nwords==4) {
						double xs, ys;
						if (!(ws[2] >> xs)) Complain("Invalid x-coordinate for initial source point");
						if (!(ws[3] >> ys)) Complain("Invalid y-coordinate for initial source point");
						srcpt[0] = xs;
						srcpt[1] = ys;
					} else if (nwords==3) Complain("two source coordinates should be specified (xsrc,ysrc)");
					else Complain("too many arguments to 'ptsrc add'; only source coordinates are required");
					bool vary_source_coords = (use_analytic_bestfit_src) ? false : true;
					add_point_source(zsrc,srcpt,vary_source_coords); // even if user is not specifying vary flags directly, assume source coords will be varied unless 'analytic_bestfit_src' is on
					ptsrc_number = n_ptsrc-1; // for setting vary flags (below)
					added_new_ptsrc = true;
				}
				else Complain("unrecognized argument to 'ptsrc'");
				update_parameter_list();
			}

			if (vary_parameters) {
				int nvary;
				int ptsrcnum, ptsrcnum_i, ptsrcnum_f;
				bool print_line_for_each_source = false;
				if (ptsrc_number < 0) {
					// in this case, prompt for limits for all point sources
					ptsrcnum_i = 0;
					ptsrcnum_f = n_ptsrc;
					print_line_for_each_source = true;
				} else {
					ptsrcnum_i = ptsrc_number;
					ptsrcnum_f = ptsrc_number+1;
				}
				for (ptsrcnum=ptsrcnum_i; ptsrcnum < ptsrcnum_f; ptsrcnum++) {
					if ((prompt_for_flags) and (!entered_varyflags)) {
						nparams_to_vary = ptsrc_list[ptsrcnum]->n_active_params;
						if ((mpi_id==0) and (print_line_for_each_source)) cout << "Vary flags for point source " << ptsrcnum << ":" << endl;
						if (read_command(false)==false) return;
						int nparams_entered = nwords;
						if (nparams_entered != nparams_to_vary) Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified ptsrc");
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
						if (set_ptsrc_vary_parameters(ptsrcnum,vary_flags)==false) {
							Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified ptsrc");
						}
					}
					if ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST)) {
						nvary = ptsrc_list[ptsrcnum]->n_vary_params;
						ptsrc_list[ptsrcnum]->get_varyflags(vary_flags);
						if (nvary != 0) {
							dvector lower(nvary), upper(nvary), lower_initial(nvary), upper_initial(nvary);
							vector<string> paramnames;
							ptsrc_list[ptsrcnum]->get_fit_parameter_names(paramnames);
							int i,j;
							for (i=0, j=0; j < nparams_to_vary; j++) {
								if (vary_flags[j]) {
									if ((mpi_id==0) and (verbal_mode)) cout << "limits for parameter " << paramnames[i] << ":\n";
									if (read_command(false)==false) { if (added_new_ptsrc) remove_point_source(ptsrcnum); Complain("parameter limits could not be read"); }
									if (nwords >= 2) {
										if (!(ws[0] >> lower[i])) { if (added_new_ptsrc) remove_point_source(ptsrcnum); Complain("invalid lower limit"); }
										if (!(ws[1] >> upper[i])) { if (added_new_ptsrc) remove_point_source(ptsrcnum); Complain("invalid upper limit"); }
										if (nwords == 2) {
											lower_initial[i] = lower[i];
											upper_initial[i] = upper[i];
										} else if (nwords == 4) {
											if (!(ws[2] >> lower_initial[i])) { if (added_new_ptsrc) remove_point_source(ptsrcnum); Complain("invalid initial lower limit"); }
											if (!(ws[3] >> upper_initial[i])) { if (added_new_ptsrc) remove_point_source(ptsrcnum); Complain("invalid initial upper limit"); }
										} else {
											if (added_new_ptsrc) remove_point_source(ptsrcnum);
											Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
										}
									} else {
										if (added_new_ptsrc) remove_point_source(ptsrcnum);
										Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
									}
									if (lower_initial[i] < lower[i]) lower_initial[i] = lower[i];
									if (upper_initial[i] > upper[i]) upper_initial[i] = upper[i];
									i++;
								}
							}
							ptsrc_list[ptsrcnum]->set_limits(lower,upper,lower_initial,upper_initial);
						}
					}
				}
			}
		}
		else if ((words[0]=="psf") or ((words[0]=="fit") and (nwords > 1) and (words[1]=="psf")))
		{
			bool update_specific_parameters = false; // option for user to update one (or more) specific parameters rather than update all of them at once
			bool vary_parameters = false;
			vector<string> specific_update_params;
			vector<double> specific_update_param_vals;
			int psf_number = -1;
			bool entered_varyflags = false;
			bool added_new_psf = false;
			bool prompt_for_flags = true;
			int nparams_to_vary = nwords;
			boolvector vary_flags;

			if (words[0]=="fit") {
				vary_parameters = true;
				// now remove the "fit" word from the line so we can add PSF's the same way,
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

			if ((nwords > 1) and (words[1]=="update")) update_specific_parameters = true;

			if (update_specific_parameters) {
				if (nwords > 2) {
					if (!(ws[2] >> psf_number)) Complain("invalid PSF number");
					if ((n_psf <= psf_number) or (psf_number < 0)) Complain("specified PSF number does not exist");
					update_specific_parameters = true;
					// Now we'll remove the "update" word
					stringstream* new_ws = new stringstream[nwords-1];
					for (int i=1; i < nwords-1; i++)
						words[i] = words[i+1];
					for (int i=0; i < nwords-1; i++)
						new_ws[i] << words[i];
					words.pop_back();
					nwords--;
					delete[] ws;
					ws = new_ws;
				} else Complain("must specify a psf number to update, followed by parameters");
			}

			if (update_specific_parameters) {
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
					} 
				}
				if (n_updates > 0) {
					for (int i=0; i < n_updates; i++)
						if (psf_list[psf_number]->update_specific_parameter(specific_update_params[i],specific_update_param_vals[i])==false) Complain("could not find parameter '" << specific_update_params[i] << "' in PSF " << psf_number);
				}
			} else if ((nwords > 1) and ((words[1]=="vary") or (words[1]=="changevary"))) {
				vary_parameters = true;
				bool set_vary_none = false;
				bool set_vary_all = false;
				if (words[nwords-1]=="none") {
					set_vary_none=true;
					remove_word(nwords-1);
				}
				if (words[nwords-1]=="all") {
					set_vary_all=true;
					remove_word(nwords-1);
				}
				if ((nwords==2) and (set_vary_none)) {
					for (int psfnum=0; psfnum < n_psf; psfnum++) {
						nparams_to_vary = psf_list[psfnum]->n_active_params;
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
						set_psf_vary_parameters(psfnum,vary_flags);
					}
					entered_varyflags = true;
					psf_number = -1; // so it prompts for limits for all PSF's
				} else if ((nwords==2) and (set_vary_all)) {
					for (int psfnum=0; psfnum < n_psf; psfnum++) {
						nparams_to_vary = psf_list[psfnum]->n_active_params;
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = true;
						set_psf_vary_parameters(psfnum,vary_flags);
					}
					entered_varyflags = true;
					psf_number = -1; // so it prompts for limits for all PSF's
				} else if (nwords==2) {
					psf_number = -1; // so it prompts for limits for all PSF's
				} else {
					if (nwords != 3) Complain("one argument required for 'psf vary' (psf number)");
					if (!(ws[2] >> psf_number)) Complain("Invalid psf number to change vary parameters");
					if (psf_number >= n_psf) Complain("specified psf number does not exist");
					nparams_to_vary = psf_list[psf_number]->n_active_params;
					if ((!set_vary_none) and (!set_vary_all)) {
						if (read_command(false)==false) return;
						int nparams_entered = nwords;
						if (nparams_entered != nparams_to_vary) Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified pixsrc");
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
						if (set_psf_vary_parameters(psf_number,vary_flags)==false) {
							Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified psf");
						}
						entered_varyflags = true;
					} else if (set_vary_none) {
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = false;
						set_psf_vary_parameters(psf_number,vary_flags);
						entered_varyflags = true;
					} else if (set_vary_all) {
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) vary_flags[i] = true;
						set_psf_vary_parameters(psf_number,vary_flags);
						entered_varyflags = true;
					}
				}
			} else if (nwords==1) {
				if (mpi_id==0) print_psf_list(vary_parameters);
				vary_parameters = false; // don't make it prompt for vary flags if they only put 'fit psf'
			//} else {
				//if (words[1]=="clear") {
					//if (nwords==2) {
						//while (n_psf > 0) {
							//remove_psf(n_psf-1);
						//}
					//}
					//else if (nwords==3) {
						//int psf_number;
						//if (!(ws[2] >> psf_number)) Complain("invalid psf number");
						//remove_psf(psf_number);
					//} else Complain("only one argument allowed for 'psf clear' (number of PSF to remove)");
				//}
				//else if (words[1]=="add") {
					//add_psf(zlens);
					//psf_number = n_psf-1; // for setting vary flags (below)
					//added_new_psf = true;
				//}
			} else {
				Complain("unrecognized argument to 'psf'");
			}

			if (vary_parameters) {
				int nvary;
				int psfnum, psfnum_i, psfnum_f;
				bool print_line_for_each_lens = false;
				if (psf_number < 0) {
					// in this case, prompt for limits for all PSF's
					psfnum_i = 0;
					psfnum_f = n_psf;
					print_line_for_each_lens = true;
				} else {
					psfnum_i = psf_number;
					psfnum_f = psf_number+1;
				}
				for (psfnum=psfnum_i; psfnum < psfnum_f; psfnum++) {
					if ((prompt_for_flags) and (!entered_varyflags)) {
						nparams_to_vary = psf_list[psfnum]->n_active_params;
						if ((mpi_id==0) and (print_line_for_each_lens)) cout << "Vary flags for pixellated source " << psfnum << ":" << endl;
						if (read_command(false)==false) return;
						int nparams_entered = nwords;
						if (nparams_entered != nparams_to_vary) Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified psf");
						vary_flags.input(nparams_to_vary);
						for (int i=0; i < nparams_to_vary; i++) if (!(ws[i] >> vary_flags[i])) Complain("vary flag must be set to 0 or 1");
						if (set_psf_vary_parameters(psfnum,vary_flags)==false) {
							Complain("number of vary flags does not match number of parameters (" << nparams_to_vary << ") for specified psf");
						}
					}
					if ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK) or (fitmethod == POLYCHORD) or (fitmethod == MULTINEST)) {
						nvary = psf_list[psfnum]->n_vary_params;
						psf_list[psfnum]->get_varyflags(vary_flags);
						if (nvary != 0) {
							dvector lower(nvary), upper(nvary), lower_initial(nvary), upper_initial(nvary);
							vector<string> paramnames;
							psf_list[psfnum]->get_fit_parameter_names(paramnames);
							int i,j;
							for (i=0, j=0; j < nparams_to_vary; j++) {
								if (vary_flags[j]) {
									if ((mpi_id==0) and (verbal_mode)) cout << "limits for parameter " << paramnames[i] << ":\n";
									if (read_command(false)==false) { if (added_new_psf) remove_psf(psfnum); Complain("parameter limits could not be read"); }
									if (nwords >= 2) {
										if (!(ws[0] >> lower[i])) { if (added_new_psf) remove_psf(psfnum); Complain("invalid lower limit"); }
										if (!(ws[1] >> upper[i])) { if (added_new_psf) remove_psf(psfnum); Complain("invalid upper limit"); }
										if (nwords == 2) {
											lower_initial[i] = lower[i];
											upper_initial[i] = upper[i];
										} else if (nwords == 4) {
											if (!(ws[2] >> lower_initial[i])) { if (added_new_psf) remove_psf(psfnum); Complain("invalid initial lower limit"); }
											if (!(ws[3] >> upper_initial[i])) { if (added_new_psf) remove_psf(psfnum); Complain("invalid initial upper limit"); }
										} else {
											if (added_new_psf) remove_psf(psfnum);
											Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
										}
									} else {
										if (added_new_psf) remove_psf(psfnum);
										Complain("must specify two/four arguments: lower limit, upper limit, and (optional) initial lower limit, initial upper limit");
									}
									if (lower_initial[i] < lower[i]) lower_initial[i] = lower[i];
									if (upper_initial[i] > upper[i]) upper_initial[i] = upper[i];
									i++;
								}
							}
							psf_list[psfnum]->set_limits(lower,upper,lower_initial,upper_initial);
						}
					}
				}
			}
		}
		else if (words[0]=="fit")
		{
			// Note: the "fit lens" command is handled along with the "lens" command above (not here), since the two commands overlap extensively
			//       same is true for the "fit source" command and "fit pixsrc" command
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
#ifdef USE_MULTINEST
							warn("qlens has not been compiled with PolyChord; switching to multinest");
							set_fitmethod(MULTINEST);
#else
							Complain("qlens code needs to be compiled with PolyChord to use this fit method");
#endif
#endif
						}
						else if (setword=="multinest") {
#ifdef USE_MULTINEST
							set_fitmethod(MULTINEST);
#else
							warn("qlens has not been compiled with MultiNest; switching to 'nest' (native nested sampling code)");
							set_fitmethod(NESTED_SAMPLING);
#endif
						}
						else Complain("invalid argument to 'fit method' command; must specify valid fit method");
					} else Complain("invalid number of arguments; can only specify fit method type");
				}
				else if ((words[1]=="regularization") or (words[1]=="reg")) {
					if (nwords==2) {
						if (mpi_id==0) {
							if (regularization_method==None) cout << "Regularization method: none" << endl;
							else if (regularization_method==Norm) cout << "Regularization method: norm" << endl;
							else if (regularization_method==Gradient) cout << "Regularization method: gradient" << endl;
							else if (regularization_method==SmoothGradient) cout << "Regularization method: sgradient" << endl;
							else if (regularization_method==Curvature) cout << "Regularization method: curvature" << endl;
							else if (regularization_method==SmoothCurvature) cout << "Regularization method: scurvature" << endl;
							else if (regularization_method==Matern_Kernel) cout << "Regularization method: Matern kernel" << endl;
							else if (regularization_method==Exponential_Kernel) cout << "Regularization method: exponential kernel" << endl;
							else if (regularization_method==Squared_Exponential_Kernel) cout << "Regularization method: squared exponential kernel" << endl;
							else cout << "Unknown regularization method" << endl;
						}
					} else if (nwords==3) {
						if (!(ws[2] >> setword)) Complain("invalid argument to 'fit regularization' command; must specify valid regularization method");
						if (((setword=="matern_kernel") or (setword=="exp_kernel") or (setword=="sqexp_kernel")) and (source_fit_mode != Delaunay_Source)) Complain("kernel-based regularization is only available for Delaunay source grids ('fit source_mode delaunay')");
						if ((setword=="none") or (setword=="off")) {
							regularization_method = None;
							if (optimize_regparam) {
								optimize_regparam = false;
								if (mpi_id==0) cout << "NOTE: Turning 'optimize_regparam' off" << endl;
							}
						}
						else if (setword=="norm") regularization_method = Norm;
						else if (setword=="gradient") regularization_method = Gradient;
						else if (setword=="sgradient") regularization_method = SmoothGradient;
						else if (setword=="curvature") regularization_method = Curvature;
						else if (setword=="scurvature") regularization_method = SmoothCurvature;
						else if (setword=="matern_kernel") regularization_method = Matern_Kernel;
						else if (setword=="exp_kernel") regularization_method = Exponential_Kernel;
						else if (setword=="sqexp_kernel") regularization_method = Squared_Exponential_Kernel;
						else Complain("invalid argument to 'fit regularization' command; must specify valid regularization method");
						for (int i=0; i < n_pixellated_src; i++) {
							update_pixsrc_active_parameters(i);
						}
					} else Complain("invalid number of arguments; can only specify regularization method");
				}
				else if (words[1]=="source_mode")
				{
					if (nwords==2) {
						if (mpi_id==0) {
							if (source_fit_mode==Point_Source) cout << "Source mode for fitting: ptsource" << endl;
							else if (source_fit_mode==Cartesian_Source) cout << "Source mode for fitting: cartesian" << endl;
							else if (source_fit_mode==Delaunay_Source) cout << "Source mode for fitting: delaunay" << endl;
							else if (source_fit_mode==Parameterized_Source) cout << "Source mode for fitting: sbprofile" << endl;
							else if (source_fit_mode==Shapelet_Source) cout << "Source mode for fitting: shapelet" << endl;
							else cout << "Unknown fit method" << endl;
						}
					} else if (nwords==3) {
						if (!(ws[2] >> setword)) Complain("invalid argument; must specify valid fit method (ptsource, cartesian, delaunay, sbprofile)");
						if (setword=="ptsource") source_fit_mode = Point_Source;
						else if (setword=="cartesian") source_fit_mode = Cartesian_Source;
						else if (setword=="delaunay") source_fit_mode = Delaunay_Source;
						else if (setword=="sbprofile") source_fit_mode = Parameterized_Source;
						else if (setword=="shapelet") source_fit_mode = Shapelet_Source;
						else Complain("invalid argument; must specify valid source mode (ptsource, cartesian, delaunay, sbprofile, shapelet)");
						if (image_pixel_grids != NULL) {
							for (int i=0; i < n_extended_src_redshifts; i++) {
								if (image_pixel_grids[i] != NULL) image_pixel_grids[i]->source_fit_mode = source_fit_mode;
							}
						}
						update_parameter_list();
						for (int i=0; i < n_pixellated_src; i++) {
							if (source_fit_mode==Delaunay_Source) srcgrids[i] = delaunay_srcgrids[i];
							else srcgrids[i] = cartesian_srcgrids[i];
						}
						for (int i=0; i < n_pixellated_src; i++) {
							update_pixsrc_active_parameters(i);
						}

						if ((ray_tracing_method == Interpolate) and (natural_neighbor_interpolation) and (source_fit_mode == Cartesian_Source)) {
							natural_neighbor_interpolation = false;
							if (mpi_id==0) cout << "NOTE: Natural neighbor interpolation is not available for Cartesian source; switching to 3-point interpolation" << endl;
						}
						if ((ray_tracing_method == Area_Overlap) and (source_fit_mode == Delaunay_Source)) {
							ray_tracing_method = Interpolate;
							if (mpi_id==0) cout << "NOTE: Overlap method not available for delaunay source; switching to 3-point interpolation" << endl;
						}
					} else Complain("invalid number of arguments; can only specify fit source mode (ptsource, cartesian, delaunay, sbprofile, shapelet)");
				}
				else if (words[1]=="findimg")
				{
					bool show_all = true;
					int dataset = 0;
					if (n_ptsrc==0) Complain("No data source points have been specified");
					if (nwords==3) {
						// if using the "src=#" notation (as in 'fit plotimg'), remove the "src=" part
						if (words[2].find("src=") == 0) {
							words[2] = words[2].substr(4);
							ws[2].str(""); ws[2].clear();
							ws[2] << words[2];
						}
						if (!(ws[2] >> dataset)) Complain("invalid image dataset");
						if (dataset >= n_ptsrc) Complain("specified image dataset has not been loaded");
						show_all = false;
					} else if (nwords > 3) Complain("invalid number of arguments; can only specify sourcept number");

					if ((include_flux_chisq) and (analytic_source_flux)) set_analytic_srcflux(false);
					if (use_analytic_bestfit_src) set_analytic_sourcepts(false);

					if (mpi_id==0) cout << endl;
					if (show_all) {
						bool different_zsrc = false;
						for (int i=0; i < n_ptsrc; i++) if (ptsrc_redshifts[ptsrc_redshift_idx[i]] != source_redshift) different_zsrc = true;
						for (int i=0; i < n_ptsrc; i++) {
							if (different_zsrc) {
								if ((i == 0) or (ptsrc_redshifts[ptsrc_redshift_idx[i]] != ptsrc_redshifts[ptsrc_redshift_idx[i-1]])) {
									create_grid(false,ptsrc_zfactors[ptsrc_redshift_idx[i]],ptsrc_beta_factors[ptsrc_redshift_idx[i]]);
								}
							}
							if (mpi_id==0) cout << "# Source " << i << ":" << endl;
							output_images_single_source(ptsrc_list[i]->pos[0], ptsrc_list[i]->pos[1], true, ptsrc_list[i]->srcflux, true);
						}
						if (different_zsrc) {
							reset_grid();
							//create_grid(false);
						}
					} else {
						if (ptsrc_redshifts[ptsrc_redshift_idx[dataset]] != source_redshift) {
							reset_grid();
							create_grid(false,ptsrc_zfactors[ptsrc_redshift_idx[dataset]],ptsrc_beta_factors[ptsrc_redshift_idx[dataset]]);
						}
						output_images_single_source(ptsrc_list[dataset]->pos[0], ptsrc_list[dataset]->pos[1], true, ptsrc_list[dataset]->srcflux, true);
						if (ptsrc_redshifts[dataset] != source_redshift) {
							reset_grid();
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
					chisq_pos_image_plane_diagnostic(false,false,rms_err,nmatched);
					if (mpi_id==0) cout << "# matched image pairs = " << nmatched << ", rms_imgpos_error = " << rms_err << endl << endl;
				}
				else if (words[1]=="data_imginfo")
				{
					if (nlens==0) Complain("No lens model has been specified");
					if (n_ptsrc==0) Complain("No data source points have been specified");
					if ((show_cc) and (plot_critical_curves("crit.dat")==false)) Complain("could not plot critical curves");
					if (nwords != 2) Complain("command 'fit imginfo' does not require any arguments");

					if ((include_flux_chisq) and (analytic_source_flux)) set_analytic_srcflux(false);
					if (use_analytic_bestfit_src) set_analytic_sourcepts(false);
					for (int i=0; i < n_ptsrc; i++) {
						plot_srcpts_from_image_data(i,NULL,ptsrc_list[i]->pos[0],ptsrc_list[i]->pos[1],ptsrc_list[i]->srcflux);
					}
				}
				else if (words[1]=="plotsrc")
				{
					if (nlens==0) Complain("No lens model has been specified");
					if (n_ptsrc==0) Complain("No data source points have been specified");
					int dataset;
					bool show_multiple = false;
					int min_dataset = 0, max_dataset = n_ptsrc - 1;
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
								if (max_dataset >= n_ptsrc) Complain("specified max image dataset exceeds number of image sets in data");
								show_multiple = true;
							} else {
								string dstr = words[2].substr(4);
								stringstream dstream;
								dstream << dstr;
								if (!(dstream >> dataset)) Complain("invalid dataset");
								if (dataset >= n_ptsrc) Complain("specified image dataset has not been loaded");
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
						reset_grid();
						create_grid(false,reference_zfactors,default_zsrc_beta_factors); // even though we're not finding images, still need to plot caustics
					} else {
						reset_grid();
						create_grid(false,ptsrc_zfactors[ptsrc_redshift_idx[dataset]],ptsrc_beta_factors[ptsrc_redshift_idx[dataset]]); // even though we're not finding images, still need to plot caustics
					}
					if ((show_cc) and (plot_critical_curves("crit.dat")==false)) Complain("could not plot critical curves and caustics");
					if ((nwords != 3) and (nwords != 2)) Complain("command 'fit plotsrc' requires either zero or one argument (source_filename)");

					if ((include_flux_chisq) and (analytic_source_flux)) set_analytic_srcflux(false);
					if (use_analytic_bestfit_src) set_analytic_sourcepts(false);

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
						if (plot_srcpts_from_image_data(dataset,&srcfile,ptsrc_list[dataset]->pos[0],ptsrc_list[dataset]->pos[1],ptsrc_list[dataset]->srcflux)==true) {
							if (mpi_id==0) {
								srcfit << "\"fit srcpt " << dataset << " (z_{s}=" << ptsrc_redshifts[ptsrc_redshift_idx[dataset]] << ")\"" << endl;
								srcfit << ptsrc_list[dataset]->pos[0] << "\t" << ptsrc_list[dataset]->pos[1] << endl << endl << endl;
								srcfile << endl << endl;
							}
						}
					} else {
						for (int i=min_dataset; i <= max_dataset; i++) {
							if (mpi_id==0) {
								if (output_to_text_files) { srcfile << "# "; }
								srcfile << "\"dataset " << i << "\"" << endl;
								srcfile << ptsrc_list[i]->pos[0] << "\t" << ptsrc_list[i]->pos[1] << " # from fit" << endl;
							}
							if (plot_srcpts_from_image_data(i,&srcfile,ptsrc_list[i]->pos[0],ptsrc_list[i]->pos[1],ptsrc_list[i]->srcflux)==true) {
								if (mpi_id==0) {
									srcfit << "\"fit srcpt " << i << " (z_{s}=" << ptsrc_redshifts[ptsrc_redshift_idx[i]] << ")\"" << endl;
									srcfit << ptsrc_list[i]->pos[0] << "\t" << ptsrc_list[i]->pos[1] << endl << endl << endl;
									srcfile << endl << endl;
								}
							}
						}
					}
					if (nwords==3) {
						if (terminal != TEXT) {
							if ((show_multiple) and (n_ptsrc > 1)) run_plotter_file("srcptfits",words[2],"");
							else run_plotter_file("srcptfit",words[2],"");
						}
					} else {
						if ((show_multiple) and (n_ptsrc > 1)) run_plotter("srcptfits");
						else run_plotter("srcptfit");
					}
					reset_grid();
					create_grid(false,reference_zfactors,default_zsrc_beta_factors);
				}
				else if (words[1]=="plotimg")
				{
					// this needs to be redone a bit, along with "fit findimg"--it should just call get_images(...) directly, so that it can
					// add up the total number of images, specify how many images are above the stated magnification threshold, etc.
					// also, you should put this entire thing in a function -- also true for many commands in this file! Critical before public release.
					if (nlens==0) Complain("No lens model has been specified");
					if (n_ptsrc==0) Complain("No data source points have been specified");
					bool omit_source = false;
					bool omit_cc = false;
					bool old_cc_setting = show_cc;
					bool old_plot_srcplane = plot_srcplane;
					string range1, range2;
					extract_word_starts_with('[',2,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
					extract_word_starts_with('[',2,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
					if (range1.empty()) { range1 = range2; range2 = ""; } // range is for image plane if only one range argument specified
					bool set_title = false;
					string temp_title;
					for (int i=1; i < nwords-1; i++) {
						if (words[i]=="-t") {
							set_title = true;
							set_plot_title(i+1,temp_title);
							remove_word(i);
							break;
						}
					}
					vector<string> args;
					if (extract_word_starts_with('-',2,nwords-1,args)==true)
					{
						int pos;
						for (int i=0; i < args.size(); i++) {
							if (args[i]=="-nosrc") omit_source = true;
							else if (args[i]=="-nocc") { omit_cc = true; show_cc = false; }
							else Complain("argument '" << args[i] << "' not recognized");
						}
					}
					if (omit_source) plot_srcplane = false;
					if (omit_cc) show_cc = false;

					int dataset;
					bool show_multiple = false;
					bool show_grid = false;
					int min_dataset = 0, max_dataset = n_ptsrc - 1;
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
								if (max_dataset >= n_ptsrc) Complain("specified max image dataset exceeds number of image sets in data");
								show_multiple = true;
							} else {
								string dstr = words[2].substr(4);
								stringstream dstream;
								dstream << dstr;
								if (!(dstream >> dataset)) Complain("invalid dataset");
								if (dataset >= n_ptsrc) Complain("specified image dataset has not been loaded");
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
					if ((show_multiple) and (show_cc) and (plot_critical_curves("crit.dat")==false)) Complain("could not plot critical curves");
					if (!show_multiple) {
						reset_grid();
						int zgroup = -1;
						for (int k=0; k < ptsrc_redshift_groups.size()-1; k++) { if ((dataset >= ptsrc_redshift_groups[k]) and (dataset < ptsrc_redshift_groups[k+1])) zgroup = k; }
						create_grid(false,ptsrc_zfactors[ptsrc_redshift_idx[dataset]],ptsrc_beta_factors[ptsrc_redshift_idx[dataset]],zgroup);
						if (show_grid) plot_recursive_grid("xgrid.dat");
						// Plot critical curves corresponding to the particular source redshift being plotted
						if ((show_cc) and (plot_critical_curves("crit.dat")==false)) Complain("could not plot critical curves");
					} else {
						reset_grid();
						create_grid(false,ptsrc_zfactors[ptsrc_redshift_idx[min_dataset]],ptsrc_beta_factors[ptsrc_redshift_idx[min_dataset]]);
					}
					if ((nwords != 4) and (nwords != 2)) Complain("command 'fit plotimg' requires either zero or two arguments (source_filename, image_filename)");
					if ((include_flux_chisq) and (analytic_source_flux)) set_analytic_srcflux(false);
					if (use_analytic_bestfit_src) set_analytic_sourcepts(false);
					if (mpi_id==0) cout << endl;
					string imgname="imgs.dat", srcname="srcs.dat";
					bool output_to_text_files = false;
					if ((terminal==TEXT) and (nwords==4)) {
						output_to_text_files = true;
						srcname = words[2];
						imgname = words[3];
					}
					if (set_title) plot_title = temp_title;
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
						if (plot_images_single_source(ptsrc_list[dataset]->pos[0], ptsrc_list[dataset]->pos[1], verbal_mode, imgfile, srcfile, ptsrc_list[dataset]->srcflux, true)==true) {
							if (mpi_id==0) {
								imgout << "\"dataset " << dataset << " (z_{s}=" << ptsrc_redshifts[ptsrc_redshift_idx[dataset]] << ")\"" << endl;
								image_data[dataset].write_to_file(imgout);
								imgout << endl << endl;
								imgfile << endl << endl;
								srcfile << endl << endl;
							}
						}
					} else {
						reset_grid();
						for (int i=min_dataset; i <= max_dataset; i++) {
							if ((i == min_dataset) or (ptsrc_zfactors[ptsrc_redshift_idx[i]] != ptsrc_zfactors[ptsrc_redshift_idx[i-1]]))
								create_grid(false,ptsrc_zfactors[ptsrc_redshift_idx[i]],ptsrc_beta_factors[ptsrc_redshift_idx[i]]);
							if (mpi_id==0) {
								if (output_to_text_files) { imgfile << "# "; srcfile << "# "; }
								imgfile << "\"image set " << i << "\"" << endl;
								srcfile << "\"source " << i << "\"" << endl;
							}
							if (plot_images_single_source(ptsrc_list[i]->pos[0], ptsrc_list[i]->pos[1], verbal_mode, imgfile, srcfile, ptsrc_list[i]->srcflux, true)==true) {
								if (mpi_id==0) {
									imgout << "\"dataset " << i << " (z_{s}=" << ptsrc_redshifts[ptsrc_redshift_idx[i]] << ")\"" << endl;
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
								if ((show_multiple) and (n_ptsrc > 1)) run_plotter_file("imgfits",words[3],range1);
								else run_plotter_file("imgfit",words[3],range1);
							}
							else {
								run_plotter_file("imgfit_nocc",words[3],range1);
							}
							if (plot_srcplane) {
								if ((show_multiple) and (n_ptsrc > 1)) run_plotter_file("srcfits",words[2],range2);
								else run_plotter_file("srcfit",words[2],range2);
							}
						}
					} else {
						if (show_cc) {
							if ((show_multiple) and (n_ptsrc > 1)) run_plotter("imgfits",range1);
							else run_plotter("imgfit",range1);
							if (plot_srcplane) {
								if ((show_multiple) and (n_ptsrc > 1)) run_plotter("srcfits",range2);
								else run_plotter("srcfit",range2);
							}
						}
						else {
							run_plotter("imgfit_nocc",range1);
						}
					}
					show_imgsrch_grid = false;
					reset_grid();
					create_grid(false,reference_zfactors,default_zsrc_beta_factors);
					if (omit_source) plot_srcplane = old_plot_srcplane;
					if (omit_cc) show_cc = old_cc_setting;
					if (set_title) plot_title = "";
				}
				else if (words[1]=="plotshear")
				{
					if (weak_lensing_data.n_sources==0) Complain("no weak lensing data has been loaded");
					string range;
					extract_word_starts_with('[',2,range); // allow for ranges to be specified (if it's not, then ranges are set to "")
					if ((nwords != 3) and (nwords != 2)) Complain("command 'fit plotshear' requires either zero or one argument (shear_filename)");
					string filename = "shear.dat";
					if ((terminal==TEXT) and (nwords==3)) filename = words[2];
					plot_weak_lensing_shear_data(true,filename);
					if (nwords==3) {
						if (terminal != TEXT) {
							if (show_cc) run_plotter_file("shearfits",words[2],range);
							else run_plotter_file("shearfits_nocc",words[2],range);
						}
					} else {
						if (show_cc) run_plotter("shearfits",range);
						else run_plotter("shearfits_nocc",range);
					}
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
					bool nohists = false;
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
							if (words[i]=="-nohist") {
								nohists = true;
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
						run_mkdist(false,"",nbins1d,nbins2d,copy_subplot_only,resampled_posts,no2dposts,nohists);
					} else if (nwords==3) {
						run_mkdist(true,words[2],nbins1d,nbins2d,copy_subplot_only,resampled_posts,no2dposts,nohists);
					} else Complain("either zero/one argument allowed for 'fit mkposts' (directory name, plus optional '-n' or '-N' args)");
				}
				else if ((words[1]=="run") or (words[1]=="process_chain"))
				{
					int nparams;
					get_n_fit_parameters(nparams);
					if (nparams==0) Complain("cannot run fit; no parameters are being varied");
					bool resume = false;
					bool skip_run = false;
					bool no_errors = false;
					bool adopt_bestfit = false;
					bool varying_lensparams = true;
					bool make_imgdata = false; // if turned on, will convert unlensed source points being fit into a system of lensed images, with position uncertainties taken from the fit
					if (words[1]=="process_chain") {
						if (nwords > 2) Complain("no arguments allowed for 'fit process_chain'");
						skip_run = true;
					} else if (nwords > 2) { // the following arguments apply only to 'fit run'
						vector<string> args;
						if (extract_word_starts_with('-',2,nwords-1,args)==true)
						{
							for (int i=0; i < args.size(); i++) {
								if (args[i]=="-resume") resume = true;
								else if (args[i]=="-adopt") adopt_bestfit = true;
								else if (args[i]=="-process") skip_run = true;
								else if (args[i]=="-noerrs") no_errors = true;
								else if (args[i]=="-mkimgdata") { make_imgdata = true; adopt_bestfit = true; }
								else Complain("argument '" << args[i] << "' not recognized");
							}
						}
					}
					if ((make_imgdata) and ((!calculate_parameter_errors) or (no_errors))) Complain("parameter uncertainties required to generate image data from best-fit points");
					if ((skip_run) and ((fitmethod != MULTINEST) and (fitmethod != POLYCHORD))) Complain("cannot process chains unless Polychord or Multinest is being used");
					if ((resume) and ((fitmethod != MULTINEST) and (fitmethod != POLYCHORD))) Complain("cannot resume unless Polychord or Multinest is being used");
					if ((make_imgdata) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) Complain("cannot make imgdata unless Powell or Simplex is being used");
					if ((make_imgdata) and (no_errors)) Complain("Errors must be turned on to make image data from fit");
					bool old_error_setting;
					if ((no_errors) and ((fitmethod==POWELL) or (fitmethod==SIMPLEX))) {
						old_error_setting = calculate_parameter_errors;
						calculate_parameter_errors = false;
					}
					if (fitmethod==POWELL) chi_square_fit_powell();
					else if (fitmethod==SIMPLEX) chi_square_fit_simplex();
					else if (fitmethod==NESTED_SAMPLING) nested_sampling();
					else if (fitmethod==POLYCHORD) polychord(resume,skip_run);
					else if (fitmethod==MULTINEST) multinest(resume,skip_run);
					else if (fitmethod==TWALK) chi_square_twalk();
					else Complain("unsupported fit method");
					if ((no_errors) and ((fitmethod==POWELL) or (fitmethod==SIMPLEX))) calculate_parameter_errors = old_error_setting;
					if ((adopt_bestfit) and (adopt_model(bestfitparams)==false)) Complain("could not adopt best-fit model");
					if (make_imgdata) {
						int param_num, npar=0;
						get_n_fit_parameters(npar);
						param_num = lensmodel_fit_parameters + srcmodel_fit_parameters;
						add_image_data_from_unlensed_sourcepts(true,param_num,2);
						if (nlens > 0) set_analytic_sourcepts(); // just to have a starting guess for the source point
					}
				}
				else if (words[1]=="chisq")
				{
					bool showdiag = false;
					bool show_lensinfo = false;
					bool temp_show_wtime = false;
					bool show_total_wtime = false;
					bool old_show_wtime = show_wtime;
					bool init_fitmodel = true; // if set to false, will skip creating a separate "fitmodel" object and just use current qlens object as fitmodel
					vector<string> args;
					if (extract_word_starts_with('-',2,nwords-1,args)==true)
					{
						int pos;
						for (int i=0; i < args.size(); i++) {
							if ((args[i]=="-wtime") or (args[i]=="-w")) temp_show_wtime = true;
							else if (args[i]=="-T") show_total_wtime = true;
							else if (args[i]=="-diag") showdiag = true;
							else if (args[i]=="-info") show_lensinfo = true;
							else if ((args[i]=="-skipfm") or (args[i]=="-s")) init_fitmodel = false;
							else Complain("argument '" << args[i] << "' not recognized");
						}
					}

					if (nwords > 2) Complain("no arguments to 'fit chisq' allowed (except for flags using '-....')");
					int np;
					get_n_fit_parameters(np);
					if ((temp_show_wtime) and (!show_wtime)) show_wtime = true;
					chisq_single_evaluation(init_fitmodel,show_total_wtime,showdiag,true,show_lensinfo);
					clear_raw_chisq(); // in case raw chi-square is being used as a derived parameter
					if ((temp_show_wtime) and (!old_show_wtime)) show_wtime = false;
				}
				else if (words[1]=="output_img_chivals") {
					string filename = "img_chivals.dat";
					if (nwords > 3) Complain("only one argument to 'fit output_img_chivals' allowed (output filename)");
					if (nwords == 3) {
						filename.assign(words[2]);
					}
					if (group_num==0) {
						double rms_err;
						int nmatched;
						chisq_pos_image_plane_diagnostic(false,true,rms_err,nmatched,filename);
					}
				}
				else if (words[1]=="output_wl_chivals") {
					string filename = "wl_chivals.dat";
					if (nwords > 3) Complain("only one argument to 'fit output_wl_chivals' allowed (output filename)");
					if (nwords == 3) {
						filename.assign(words[2]);
					}
					if (group_num==0) {
						output_weak_lensing_chivals(filename);
					}
				}
				else if (words[1]=="label")
				{
					string label;
					if ((nwords == 2) and (mpi_id==0)) cout << "Fit label: " << fit_output_filename << endl;
					else {
						if (nwords != 3) Complain("a single filename must be specified after 'fit label'");
						if (!(ws[2] >> label)) Complain("Invalid fit label");
						set_fit_label(label); // this function is located in the header qlens.h
					}
				}
				else if (words[1]=="params")
				{
					bool no_sci_notation = false;
					for (int i=nwords-1; i > 1; i--) {
						if (words[i]=="-nosci") {
							no_sci_notation = true;
							remove_word(i);
						}
					}
					if (nwords==2) {
						if ((no_sci_notation) and (use_scientific_notation)) {
							cout << resetiosflags(ios::scientific);
							cout.unsetf(ios_base::floatfield);
						}
						if (output_parameter_values()==false) Complain("could not output parameter values");
						if ((no_sci_notation) and (use_scientific_notation)) setiosflags(ios::scientific);
					} else if (words[2]=="rename") {
						if (nwords != 5) Complain("two arguments required for 'fit params rename' (param#, param_name)");
						int param_num;
						if (!(ws[3] >> param_num)) {
							if ((param_num = param_settings->lookup_param_number(words[3])) == -1)
							Complain("Invalid parameter number/name");
						}
						int nparams;
						get_n_fit_parameters(nparams);
						if (param_num >= nparams) Complain("Parameter number does not exist (see parameter list with 'fit params')");
						if (!param_settings->set_override_parameter_name(param_num,words[4])) Complain("parameter name not unique; parameter could not be renamed");
					} else if (words[2]=="update") {
						int pos, n_updates = 0;
						double pval;
						vector<string> update_param_list;
						vector<double> update_param_vals;
						for (int i=3; i < nwords; i++) {
							if ((pos = words[i].find("="))!=string::npos) {
								n_updates++;
								update_param_list.push_back(words[i].substr(0,pos));
								stringstream pvalstr;
								pvalstr << words[i].substr(pos+1);
								pvalstr >> pval;
								update_param_vals.push_back(pval);
							} else if (i==3) break;
						}
						for (int i=nwords-1; i >= 3; i--) {
							if ((pos = words[i].find("="))!=string::npos) remove_word(i); // now that they're stored, remove all the '<name>=<val>' arguments
						}

						if (n_updates > 0) {
							int paramnum_list[n_updates];
							if (nwords != 3) Complain("parameters must be updated using '<param_name>=<val>' arguments or else using '<param_num> <val>'");
							for (int i=0; i < n_updates; i++) {
								if ((paramnum_list[i] = param_settings->lookup_param_number(update_param_list[i])) == -1) Complain("Invalid parameter name '" << update_param_list[i] << "'");
							}
							for (int i=0; i < n_updates; i++)
								if (update_parameter_value(paramnum_list[i],update_param_vals[i])==false) Complain("could not update parameter " << paramnum_list);
						} else {
							if (nwords != 5) Complain("two arguments required for 'fit params update' (param#, value)");
							int param_num;
							double paramval;
							if (!(ws[3] >> param_num)) {
								if ((param_num = param_settings->lookup_param_number(words[3])) == -1)
								Complain("Invalid parameter number/name");
							}
							int nparams;
							get_n_fit_parameters(nparams);
							if (param_num >= nparams) Complain("Parameter number does not exist (see parameter list with 'fit params')");
							if (!(ws[4] >> paramval)) Complain("invalid parameter value");
							if (update_parameter_value(param_num,paramval)==false) Complain("could not update parameter value");
						}
					} else if (words[2]=="update_all") {
						int n_updates = nwords-3;
						int nparams;
						double paramval;
							get_n_fit_parameters(nparams);
						if (n_updates > nparams) Complain("cannot have more arguments than fit parameters");
						for (int i=0; i < n_updates; i++) {
							if (!(ws[i+3] >> paramval)) Complain("invalid value for parameter " << i);
							if (update_parameter_value(i,paramval)==false) Complain("could not update parameter value for parameter" << i);
						}
					} else Complain("argument not recognized for 'fit params'");
				}
				else if (words[1]=="priors")
				{
					int nparams;
					get_n_fit_parameters(nparams);
					if (nparams==0) Complain("no fit parameters have been defined");
					bool no_sci_notation = false;
					for (int i=nwords-1; i > 1; i--) {
						if (words[i]=="-nosci") {
							no_sci_notation = true;
							remove_word(i);
						}
					}
					if (nwords==2) {
						if ((no_sci_notation) and (use_scientific_notation)) {
							cout << resetiosflags(ios::scientific);
							cout.unsetf(ios_base::floatfield);
						}
						output_parameter_prior_ranges();
						if ((no_sci_notation) and (use_scientific_notation)) setiosflags(ios::scientific);
					}
					else if (((nwords==3) or (nwords==4)) and (words[2]=="limits")) {
						int nparams;
						get_n_fit_parameters(nparams);
						if (nparams==0) Complain("no fit parameters have been defined");
						double lo, hi;
						if (nwords==3) {
							for (int i=0; i < nparams; i++) {
								string paramname = param_settings->lookup_param_name(i);
								if ((mpi_id==0) and (verbal_mode)) cout << "limits for parameter " << paramname << ":\n";
								if (read_command(false)==false) Complain("parameter limits could not be read"); 
								if ((nwords==1) and (words[0]=="skip")) continue;
								else if (nwords >= 2) {
									if (!(ws[0] >> lo)) Complain("Invalid lower prior limit");
									if (!(ws[1] >> hi)) Complain("Invalid upper prior limit");
									param_settings->set_override_prior_limit(i,lo,hi);
								} else Complain("require lower and upper limits (or 'skip') for parameter '" << paramname << "'");
							}
						} else {
							int param_num = -1;
							if (!(ws[3] >> param_num)) {
								if ((param_num = param_settings->lookup_param_number(words[2])) == -1)
								Complain("Invalid parameter number/name");
							}
							string paramname = param_settings->lookup_param_name(param_num);
							if ((mpi_id==0) and (verbal_mode)) cout << "limits for parameter " << paramname << ":\n";
							if (read_command(false)==false) Complain("parameter limits could not be read"); 
							else if (nwords >= 2) {
								if (!(ws[0] >> lo)) Complain("Invalid lower prior limit");
								if (!(ws[1] >> hi)) Complain("Invalid upper prior limit");
								param_settings->set_override_prior_limit(param_num,lo,hi);
							}
						}
					}
					else if (nwords >= 4) {
						int param_num;
						if (!(ws[2] >> param_num)) {
							if ((param_num = param_settings->lookup_param_number(words[2])) == -1)
							Complain("Invalid parameter number/name");
						}
						if (param_num >= nparams) Complain("Parameter number does not exist (see parameter list with 'fit priors')");
						if (words[3]=="limits") {
							if (nwords != 6) Complain("require lower and upper limits after 'fit priors <param#> range'");
							double lo, hi;
							if (!(ws[4] >> lo)) Complain("Invalid lower prior limit");
							if (!(ws[5] >> hi)) Complain("Invalid upper prior limit");
							param_settings->set_override_prior_limit(param_num,lo,hi);
						}
						else if (words[3]=="uniform") param_settings->priors[param_num]->set_uniform();
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
					bool include_jac = false;
					bool rename = false;
					string new_name = "";
					vector<string> args;
					if (extract_word_starts_with('-',2,nwords-1,args)==true)
					{
						int pos;
						for (int i=0; i < args.size(); i++) {
							if (args[i]=="-include_jac") include_jac = true;
							else if ((pos = args[i].find("-name=")) != string::npos) {
								rename = true;
								new_name = args[i].substr(pos+6);
							}
							else Complain("argument '" << args[i] << "' not recognized");
						}
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
							if (!(ws[4] >> ratio_pnum)) {
								if ((ratio_pnum = param_settings->lookup_param_number(words[4])) == -1)
								Complain("Invalid parameter number/name for ratio transformation");
							}
							if ((ratio_pnum >= nparams) or (ratio_pnum < 0)) Complain("Parameter number specified for ratio transformation does not exist");
							if (ratio_pnum==param_num) Complain("Parameter number in denominator for ratio transformation must be different from parameter number in numerator");
							param_settings->transforms[param_num]->set_ratio(ratio_pnum);
						}
						else Complain("transformation type not recognized");
						param_settings->transforms[param_num]->set_include_jacobian(include_jac);
						param_settings->transform_stepsizes();
						if ((rename) and (!param_settings->set_override_parameter_name(param_num,new_name))) Complain("parameter name not unique; parameter could not be renamed");
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
						else if (words[2]=="from_chain") Complain("command 'fit stepsizes from_chain' requires an additional argument (scaling factor)");
						else Complain("argument to 'fit stepsizes' not recognized");
					}
					else if (nwords == 4) {
						if (words[2]=="scale") {
							double fac;
							if (!(ws[3] >> fac)) Complain("Invalid scale factor");
							param_settings->scale_stepsizes(fac);
						} else if (words[2]=="from_chain") {
							double fac;
							if (!(ws[3] >> fac)) Complain("Invalid scale factor");
							dvector stepsizes(nparams);
							if (get_stepsizes_from_percentiles(fac,stepsizes)==false) Complain("could not get stepsizes from percentiles");
							param_settings->reset_stepsizes_no_transform(stepsizes.array());
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
					update_parameter_list();
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
					bool use_kpc = false;
					vector<string> args;
					if (extract_word_starts_with('-',2,nwords-1,args)==true)
					{
						for (int i=0; i < args.size(); i++) {
							if (args[i]=="-kpc") use_kpc = true;
							else Complain("argument '" << args[i] << "' not recognized");
						}
					}
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
							double dparam_arg = -1e30;
							int lensnum;
							if (words[3]=="kappa_r") {
								if (nwords < 5) Complain("at least one additional argument required for 'fit dparams add kappa_r' (param_arg)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (nwords==6) {
									if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
									if (lensnum >= nlens) Complain("specified lens number does not exist");
								} else lensnum = -1;
								add_derived_param(KappaR,dparam_arg,lensnum,-1,use_kpc);
							} else if (words[3]=="lambda_r") {
								if (nwords < 5) Complain("at least one additional argument required for 'fit dparams add lambda_r' (param_arg)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								add_derived_param(LambdaR,dparam_arg,-1,-1,use_kpc);
							} else if (words[3]=="dkappa_r") {
								if (nwords < 5) Complain("at least one additional argument required for 'fit dparams add dkappa_r' (param_arg)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (nwords==6) {
									if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
									if (lensnum >= nlens) Complain("specified lens number does not exist");
								} else lensnum = -1;
								add_derived_param(DKappaR,dparam_arg,lensnum,-1,use_kpc);
							} else if (words[3]=="logslope") {
								if (nwords != 7) Complain("derived parameter logslope requires three arguments (rmin,rmax,lens_number)");
								double dparam_arg2;
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (!(ws[5] >> dparam_arg2)) Complain("invalid derived parameter argument");
								if (!(ws[6] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(AvgLogSlope,dparam_arg,lensnum,dparam_arg2,use_kpc);
							} else if (words[3]=="mass2d_r") {
								if (nwords != 6) Complain("derived parameter mass2d_r requires two arguments (r_arcsec,lens_number)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(Mass2dR,dparam_arg,lensnum,-1,use_kpc);
							} else if (words[3]=="mass3d_r") {
								if (nwords != 6) Complain("derived parameter mass3d_r requires two arguments (r_arcsec,lens_number)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(Mass3dR,dparam_arg,lensnum,-1,use_kpc);
							} else if (words[3]=="re_zsrc") {
								if (nwords != 6) Complain("derived parameter re_zsrc requires two arguments (zsrc,lens_number)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(Einstein,dparam_arg,lensnum,-1,use_kpc);
							} else if (words[3]=="mass_re") {
								if (nwords != 6) Complain("derived parameter mass_re requires two arguments (zsrc,lens_number)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(Einstein_Mass,dparam_arg,lensnum,-1,use_kpc);
							} else if (words[3]=="xi") {
								if ((nwords != 5) and (nwords != 6)) Complain("derived parameter xi requires one or two arguments (zsrc, and optional lens number)");
								if (!(ws[4] >> dparam_arg)) Complain("invalid derived parameter argument");
								if (nwords==6) {
									if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								} else {
									lensnum = -1;
								}
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(Xi_Param,dparam_arg,lensnum,-1,use_kpc);
							} else if (words[3]=="kappa_re") {
								if (nwords != 4) Complain("derived parameter mass_re doesn't allow any arguments");
								add_derived_param(Kappa_Re,-1e30,-1,-1,false);
							} else if (words[3]=="lensparam") {
								int paramnum,pmode;
								if (nwords != 7) Complain("derived parameter lensparam requires three arguments (paramnum,lens_number,pmode)");
								if (!(ws[4] >> paramnum)) Complain("invalid derived parameter argument--must be integer (parameter number)");
								if (!(ws[5] >> lensnum)) Complain("invalid lens number argument");
								if (!(ws[6] >> pmode)) Complain("invalid pmode argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								add_derived_param(LensParam,paramnum,lensnum,pmode,use_kpc);
							} else if (words[3]=="r_perturb") {
								if (nwords != 5) Complain("derived parameter r_perturb requires only one argument (lens_number)");
								if (!(ws[4] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								if (lensnum == 0) Complain("specified lens number cannot be 0 (since lens 0 is assumed to be primary lens)");
								add_derived_param(Perturbation_Radius,0.0,lensnum,-1,use_kpc);
							} else if (words[3]=="r_perturb_rel") {
								if (nwords != 5) Complain("derived parameter r_perturb_rel requires only one argument (lens_number)");
								if (!(ws[4] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								if (lensnum == 0) Complain("specified lens number cannot be 0 (since lens 0 is assumed to be primary lens)");
								add_derived_param(Relative_Perturbation_Radius,0.0,lensnum,-1,use_kpc);
							} else if (words[3]=="mass_perturb") {
								if (nwords != 5) Complain("derived parameter mass_perturb requires only one argument (lens_number)");
								if (!(ws[4] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								if (lensnum == 0) Complain("specified lens number cannot be 0 (since lens 0 is assumed to be primary lens)");
								add_derived_param(Robust_Perturbation_Mass,0.0,lensnum,-1,use_kpc);
							} else if (words[3]=="sigma_perturb") {
								if (nwords != 5) Complain("derived parameter sigma_perturb requires only one argument (lens_number)");
								if (!(ws[4] >> lensnum)) Complain("invalid lens number argument");
								if (lensnum >= nlens) Complain("specified lens number does not exist");
								if (lensnum == 0) Complain("specified lens number cannot be 0 (since lens 0 is assumed to be primary lens)");
								add_derived_param(Robust_Perturbation_Density,0.0,lensnum,-1,use_kpc);
							} else if (words[3]=="qs") {
								double npix;
								if (nwords != 5) Complain("derived parameter qs requires only one argument (pixel number)");
								if (!(ws[4] >> npix)) Complain("invalid number of pixels");
								add_derived_param(Adaptive_Grid_qs,-1,npix,-1,false);
							} else if (words[3]=="phi_s") {
								double npix;
								if (nwords != 5) Complain("derived parameter phi_s requires only one argument (pixel number)");
								if (!(ws[4] >> npix)) Complain("invalid number of pixels");
								add_derived_param(Adaptive_Grid_phi_s,-1,npix,-1,false);
							} else if (words[3]=="sig_s") {
								double npix;
								if (nwords != 5) Complain("derived parameter sig_s requires only one argument (pixel number)");
								if (!(ws[4] >> npix)) Complain("invalid number of pixels");
								add_derived_param(Adaptive_Grid_sig_s,-1,npix,-1,false);
							} else if (words[3]=="xavg_s") {
								double npix;
								if (nwords != 5) Complain("derived parameter qs requires only one argument (pixel number)");
								if (!(ws[4] >> npix)) Complain("invalid number of pixels");
								add_derived_param(Adaptive_Grid_xavg,-1,npix,-1,false);
							} else if (words[3]=="yavg_s") {
								double npix;
								if (nwords != 5) Complain("derived parameter qs requires only one argument (pixel number)");
								if (!(ws[4] >> npix)) Complain("invalid number of pixels");
								add_derived_param(Adaptive_Grid_yavg,-1,npix,-1,false);
							} else if (words[3]=="raw_chisq") {
								if (nwords != 4) Complain("no arguments required for derived param raw_chisq");
								add_derived_param(Chi_Square,0.0,-1,-1,use_kpc);
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
					if (adopt_model(bestfitparams)==false) Complain("could not adopt best-fit model");
				} else if (words[1]=="save_bestfit") {
					if (nwords > 3) Complain("no more than one argument allowed for 'save_bestfit' command (filename)");
					if (nwords==3) {
						if (!(ws[2] >> fit_output_filename)) Complain("Invalid fit label");
					}
					if (mpi_id==0) output_bestfit_model();
				} else if (words[1]=="load_bestfit") {
					if (nwords <= 3) {
						bool custom_filename = false;
						string filename_str;
						if (nwords==3) {
							custom_filename = true;
							filename_str = fit_output_dir + "/" + words[2] + "_bf.in";
						}
						if (load_bestfit_model(custom_filename,filename_str)==false) Complain("could not load model from best-fit point file");
					} else Complain("at most one argument allowed for 'load_bestfit' (fit_label)");
				} else if (words[1]=="add_chain_dparams") {
					string file_ext = "";
					if (nwords==3) file_ext = words[2];
					else if (nwords > 3) Complain("too many arguments to 'fit add_chain_dparams'. Only one argument allowed (file extension)");
					if (add_dparams_to_chain(file_ext)==false) Complain("could not process chain data");
				} else if (words[1]=="adopt_chain_point") {
					unsigned long pnum;
					if (nwords != 3) Complain("one argument required for command 'fit adopt_chain_point' (line_number)");
					if (!(ws[2] >> pnum)) Complain("incorrect format for argument to 'fit adopt_chain_point' (line number should be integer)");
					if (adopt_point_from_chain(pnum)==false) Complain("could not load point from chain");
				} else if (words[1]=="adopt_chain_bestfit") {
					if (nwords != 2) Complain("no arguments required for command 'fit adopt_chain_bestfit'");
					if (adopt_bestfit_point_from_chain()==false) Complain("could not load point from chain");
				} else if (words[1]=="adopt_point_prange") {
					int paramnum;
					double minval, maxval;
					if (nwords != 5) Complain("three arguments required for command 'fit adopt_point_prange' (param_number, min_value, max_value)");
					if (!(ws[2] >> paramnum)) Complain("incorrect format for argument to 'fit adopt_point_prange' (param number should be integer)");
					if (!(ws[3] >> minval)) Complain("incorrect format for argument to 'fit adopt_point_prange' (min param val should be real number)");
					if (!(ws[4] >> maxval)) Complain("incorrect format for argument to 'fit adopt_point_prange' (max param val should be real number)");
					if (adopt_point_from_chain_paramrange(paramnum,minval,maxval)==false) Complain("could not load point from chain");
				} else Complain("unknown fit command");
			}
		}
		else if (words[0]=="imgdata")
		{
			if (nwords == 1) {
				if (n_ptsrc==0) Complain("no image data has been loaded");
				print_image_data(true); // The boolean argument should be removed (it says to print errors...should ALWAYS print errors!)
				// print the image data that is being used
			} else if (nwords >= 2) {
				if (words[1]=="add") {
					if (nwords < 4) Complain("At least two arguments are required for 'imgdata add' (x,y coordinates of source pt.)");
					// later, add option to include flux of source or measurement errors as extra arguments
					lensvector src;
					if (!(ws[2] >> src[0])) Complain("invalid x-coordinate of source point");
					if (!(ws[3] >> src[1])) Complain("invalid y-coordinate of source point");
					if (add_simulated_image_data(src))
						update_parameter_list();
				} else if (words[1]=="read") {
					if (nwords != 3) Complain("One argument required for 'imgdata read' (filename)");
					if (load_point_image_data(words[2])==false) Complain("unable to load image data");
					update_parameter_list();
				} else if (words[1]=="write") {
					if (nwords != 3) Complain("One argument required for 'imgdata write' (filename)");
					write_point_image_data(words[2]);
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
							if (max_imgnumber >= n_ptsrc) Complain("specified max image dataset number exceeds number of data sets in list");
							if ((min_imgnumber > max_imgnumber) or (min_imgnumber < 0)) Complain("specified min image dataset number cannot exceed max image dataset number");
							for (int i=max_imgnumber; i >= min_imgnumber; i--) remove_point_source(i);
						} else {
							if (!(ws[2] >> imgset_number)) Complain("invalid image dataset number");
							remove_point_source(imgset_number);
						}
					} else Complain("'imgdata clear' command requires either one or zero arguments");
				} else if (words[1]=="plot") {
					bool show_sbmap = false;
					bool show_all = false;
					int dataset;
					string range;
					extract_word_starts_with('[',1,3,range); // allow for ranges to be specified (if it's not, then ranges are set to "")
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
						if (dataset >= n_ptsrc) Complain("specified image dataset has not been loaded");
					} else Complain("invalid number of arguments to 'imgdata plot'");
					ofstream imgout;
					open_output_file(imgout,"imgdat.dat");
					if (show_all) {
						for (int i=0; i < n_ptsrc; i++) {
							imgout << "\"Dataset " << i << " (z_{s}=" << ptsrc_redshifts[ptsrc_redshift_idx[i]] << ")\"" << endl;
							image_data[i].write_to_file(imgout);
							imgout << endl << endl;
						}
					} else {
						imgout << "\"Dataset " << dataset << " (z_{s}=" << ptsrc_redshifts[ptsrc_redshift_idx[dataset]] << ")\"" << endl;
						image_data[dataset].write_to_file(imgout);
						imgout << endl << endl;
					}
					if (!show_sbmap)
						run_plotter("imgdat");
					else {
						if (n_data_bands==0) Complain("no image pixel data has been loaded");
						imgpixel_data_list[0]->plot_surface_brightness("img_pixel",true);
						if (range=="") {
							stringstream xminstream, xmaxstream, yminstream, ymaxstream;
							string xminstr, xmaxstr, yminstr, ymaxstr;
							xminstream << imgpixel_data_list[0]->xvals[0]; xminstream >> xminstr;
							yminstream << imgpixel_data_list[0]->yvals[0]; yminstream >> yminstr;
							xmaxstream << imgpixel_data_list[0]->xvals[imgpixel_data_list[0]->npixels_x]; xmaxstream >> xmaxstr;
							ymaxstream << imgpixel_data_list[0]->yvals[imgpixel_data_list[0]->npixels_y]; ymaxstream >> ymaxstr;
							range = "[" + xminstr + ":" + xmaxstr + "][" + yminstr + ":" + ymaxstr + "]";
						}
						run_plotter_range("imgpixel_imgdat",range);
					}
				} else if (words[1]=="use_in_chisq") {
					if ((nwords < 4) or (nwords > 5)) Complain("Two or three arguments are required for 'imgdata use_in_chisq' (imageset, image_number, on/off)");
					if (image_data == NULL) Complain("no image data has been loaded");
					int n_imgset, n_img;
					bool use_in_chisq;
					if (!(ws[2] >> n_imgset)) Complain("invalid image set number");
					if (!(ws[3] >> n_img)) Complain("invalid image number");
					if (n_imgset >= n_ptsrc) Complain("image set number has not been loaded");
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
				else if (words[1]=="add_random") {
					if ((nwords != 9) and (nwords != 10)) Complain("7 or 8 arguments required for 'wldata add_random' (nsrc,xmin,xmax,ymin,ymax,zmin,zmax,[rmin])");
					int nsrc;
					double xmin,xmax,ymin,ymax,zmin,zmax;
					double rmin=0;
					if (!(ws[2] >> nsrc)) Complain("invalid number of sources");
					if (!(ws[3] >> xmin)) Complain("invalid xmin value");
					if (!(ws[4] >> xmax)) Complain("invalid xmax value");
					if (!(ws[5] >> ymin)) Complain("invalid ymin value");
					if (!(ws[6] >> ymax)) Complain("invalid ymax value");
					if (!(ws[7] >> zmin)) Complain("invalid zmin value");
					if (!(ws[8] >> zmax)) Complain("invalid zmax value");
					if ((nwords==10) and (!(ws[9] >> rmin))) Complain("invalid rmin value");
					add_weak_lensing_data_from_random_sources(nsrc,xmin,xmax,ymin,ymax,zmin,zmax,rmin);
				}
				else if (words[1]=="read") {
					if (nwords != 3) Complain("One argument required for 'wldata read' (filename)");
					if (load_weak_lensing_data(words[2])==false) Complain("unable to load weak lensing data");
				} else if (words[1]=="write") {
					if (nwords != 3) Complain("One argument required for 'wldata write' (filename)");
					weak_lensing_data.write_to_file(words[2]);
				} else if (words[1]=="plot") {
					if (nwords > 3) Complain("Only 0 or 1 arguments allowed for 'wldata plot'");
					string filename = "shear.dat";
					if (nwords==3) filename = words[2];
					plot_weak_lensing_shear_data(false,filename);
					if (nwords==2) run_plotter("sheardata"); // if no filename is specified, just plot to the screen
				} else if (words[1]=="clear") {
					weak_lensing_data.clear();
				} else Complain("invalid argument to command 'wldata'");
			}
		}
		else if (words[0]=="plotcrit")
		{
			if (nlens==0) Complain("must specify lens model first");
			string range1, range2;
			extract_word_starts_with('[',2,2,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
			extract_word_starts_with('[',3,3,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
			if ((!plot_srcplane) and (range2.empty())) { range2 = range1; range1 = ""; }
			if (nwords == 3) {
				if (terminal == TEXT) Complain("only one filename is required for text plotting of critical curves");
				if (plot_critical_curves("crit.dat")==true) {
					run_plotter_file("crit",words[1],range1);
					if (plot_srcplane) run_plotter_file("caust",words[2],range2);
				} else Complain("No critical curves found");
			} else if (nwords == 2) {
				if (terminal == TEXT) {
					if (plot_critical_curves(words[1].c_str())==false) Complain("No critical curves found");
				}
				else Complain("two filenames must be specified for plotting of critical curves in postscript/PDF mode");
			} else if (nwords == 1) {
				if (plot_critical_curves("crit.dat")==true) {
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
					run_plotter_file("grid",words[1],range);
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
		else if (words[0]=="plotkappa")
		{
			if (terminal != TEXT) Complain("only text plotting supported for plotkappa (switch to 'term text')");
			if (nlens==0) Complain("must specify lens model first");
			int lens_number = -1;
			bool plot_percentiles_from_chain = false;
			for (int i=2; i < nwords; i++) {
				if (words[i]=="-pct_from_chain") {
					plot_percentiles_from_chain = true;
					remove_word(i);
					break;
				}
			}

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
			if ((nwords == 4) or (nwords == 5) or (nwords == 6)) {
				double rmin, rmax;
				int steps;
				ws[1] >> rmin;
				ws[2] >> rmax;
				ws[3] >> steps;
				if (rmin > rmax) Complain("rmin must be smaller than rmax for plotkappa");
				if (lens_number==-1) {
					if (nwords==4) {
						plot_total_kappa(rmin, rmax, steps, "kappaprof.dat");
						run_plotter("kappaprofile");
					}
					else if (nwords==5) plot_total_kappa(rmin, rmax, steps, words[4].c_str());
					else plot_total_kappa(rmin, rmax, steps, words[4].c_str(), words[5].c_str());
				} else {
					if (plot_percentiles_from_chain) {
						plot_kappa_profile_percentiles_from_chain(lens_number, rmin, rmax, steps, words[4]);
					} else {
						if (nwords==4) {
							plot_kappa_profile(lens_number, rmin, rmax, steps, "kappaprof.dat");
							run_plotter("kappaprofile");
						} else if (nwords==5) plot_kappa_profile(lens_number, rmin, rmax, steps, words[4].c_str());
						else plot_kappa_profile(lens_number, rmin, rmax, steps, words[4].c_str(), words[5].c_str());
					}
				}
			} else Complain("plotkappa requires at least 3 parameters (rmin, rmax, steps, (optional) kappa_outname, (optional) kderiv_outname)");
		}
		else if (words[0]=="plot_sbprofile")
		{
			if (terminal != TEXT) Complain("only text plotting supported for plot_sbprofile (switch to 'term text')");
			if (n_sb==0) Complain("must specify sb model first");
			int src_number = -1;
			//bool plot_percentiles_from_chain = false;
			//for (int i=2; i < nwords; i++) {
				//if (words[i]=="-pct_from_chain") {
					//plot_percentiles_from_chain = true;
					//remove_word(i);
					//break;
				//}
			//}

			if (words[nwords-1].find("src=")==0) {
				string lstr = words[nwords-1].substr(4);
				stringstream lstream;
				lstream << lstr;
				if (!(lstream >> src_number)) Complain("invalid lens number");
				stringstream* new_ws = new stringstream[nwords-1];
				words.erase(words.begin()+nwords-1);
				for (int i=0; i < nwords-1; i++) {
					new_ws[i] << words[i];
				}
				delete[] ws;
				ws = new_ws;
				nwords--;
			}
			if ((nwords == 4) or (nwords == 5)) {
				double rmin, rmax;
				int steps;
				ws[1] >> rmin;
				ws[2] >> rmax;
				ws[3] >> steps;
				if (rmin > rmax) Complain("rmin must be smaller than rmax for plot_sbprofile");
				if (src_number==-1) {
					if (nwords==4) {
						plot_total_sbprofile(rmin, rmax, steps, "sbprof.dat");
						run_plotter("sbprofile");
					} else {
						plot_total_sbprofile(rmin, rmax, steps, words[4].c_str());
					}
				} else {
					//if (plot_percentiles_from_chain) {
						//plot_sbprofile_profile_percentiles_from_chain(lens_number, rmin, rmax, steps, words[4]);
					//} else {
						if (nwords==4) {
							plot_sb_profile(src_number, rmin, rmax, steps, "sbprof.dat");
							run_plotter("sbprofile");
						} else {
							plot_sb_profile(src_number, rmin, rmax, steps, words[4].c_str());
						}
					//}
				}
			} else Complain("plot_sbprofile requires at least 3 parameters (rmin, rmax, steps, (optional) src=# or sb_profile_outname)");
		}
		else if (words[0]=="plotshear")
		{
			if (nwords > 8) Complain("too many parameters in plotshear (no more than 7 allowed)");
			if ((nwords==1) or (nwords >= 7)) {
				double xmin, xmax, ymin, ymax;
				int xsteps, ysteps;
				if (nwords==1) {
					xmin = grid_xcenter-grid_xlength; xmax = grid_xcenter+grid_xlength;
					ymin = grid_ycenter-grid_ylength; ymax = grid_ycenter+grid_ylength;
					xsteps = 60;
					ysteps = 60;
				} else {
					if (!(ws[1] >> xmin)) Complain("invalid xmin parameter");
					if (!(ws[2] >> xmax)) Complain("invalid xmax parameter");
					if (!(ws[3] >> xsteps)) Complain("invalid nx parameter; must be integral number of steps");
					if (!(ws[4] >> ymin)) Complain("invalid ymin parameter");
					if (!(ws[5] >> ymax)) Complain("invalid ymax parameter");
					if (!(ws[6] >> ysteps)) Complain("invalid ny parameter; must be integral number of steps");
				}
				string outfile = "shearfield.dat";
				if ((nwords==8) and (terminal==TEXT)) outfile = words[7];
				plot_shear_field(xmin,xmax,xsteps,ymin,ymax,ysteps,outfile);
				if (nwords==8) {
					if (terminal != TEXT) {
						if (show_cc) run_plotter_file("shearfield",words[7],"");
						else run_plotter_file("shearfield_nocc",words[7],"");
					}
				} else {
					if (show_cc) run_plotter("shearfield");
					else run_plotter("shearfield_nocc");
				}
			} else Complain("plotshear requires either 0 or at least 6 arguments: xmin, xmax, nx, ymin, ymax, ny, [filename]");
		}
		else if (words[0]=="plotmass")
		{
			if (terminal != TEXT) Complain("only text plotting supported for plotmass");
			if (nlens==0) Complain("must specify lens model first");
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
		//else if (words[0]=="plotmatern")
		//{
			//if (terminal != TEXT) Complain("only text plotting supported for plotmatern");
			//if (delaunay_srcgrids == NULL) Complain("Dalaunay grid must be present to plot matern function");
			//if (nwords == 5) {
				//double rmin, rmax;
				//int steps;
				//ws[1] >> rmin;
				//ws[2] >> rmax;
				//ws[3] >> steps;
				//plot_matern_function(rmin, rmax, steps, words[4].c_str());
			//} else
			  //Complain("plotmass requires 4 parameters (rmin, rmax, steps, mass_outname)");
		//}
		else if (words[0]=="findimg")
		{
			if (nlens==0) Complain("must specify lens model first");
			bool use_imgpt = false;
			vector<string> args;
			if (extract_word_starts_with('-',3,nwords-1,args)==true)
			{
				for (int i=0; i < args.size(); i++) {
					if (args[i]=="-imgpt") use_imgpt = true;
					else Complain("argument '" << args[i] << "' not recognized");
				}
			}
			if (nwords==3) {
				double xsource_in, ysource_in, x_in, y_in;
				if (!(ws[1] >> x_in)) Complain("invalid source x-position");
				if (!(ws[2] >> y_in)) Complain("invalid source y-position");
				if (use_imgpt) {
					lensvector pos,src;
					pos[0] = x_in;
					pos[1] = y_in;
					find_sourcept(pos,src,0,reference_zfactors,default_zsrc_beta_factors);
					xsource_in = src[0];
					ysource_in = src[1];
				} else {
					xsource_in = x_in;
					ysource_in = y_in;
				}
				output_images_single_source(xsource_in, ysource_in, true);
			} else Complain("must specify two arguments that give source position (e.g. 'findimg 3.0 1.2')");
		}
		else if (words[0]=="plotimg")
		{
			bool plot_from_imgpt = false;
			bool show_grid = false;
			bool set_title = false;
			string temp_title;
			for (int i=1; i < nwords-1; i++) {
				if (words[i]=="-t") {
					set_title = true;
					set_plot_title(i+1,temp_title);
					remove_word(i);
					break;
				}
			}
			vector<string> args;
			if (extract_word_starts_with('-',3,nwords-1,args)==true)
			{
				for (int i=0; i < args.size(); i++) {
					if (args[i]=="-imgpt") plot_from_imgpt = true;
					else if (args[i]=="-grid") show_grid = true;
					else Complain("argument '" << args[i] << "' not recognized");
				}
			}
			if (nlens==0) Complain("must specify lens model first");
			string range1, range2;
			extract_word_starts_with('[',3,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
			extract_word_starts_with('[',3,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
			if (range1.empty()) { range1 = range2; range2 = ""; } // range is for image plane if only one range argument specified
			if (nwords > 5) Complain("max 4 arguments allowed for plotimg: <source_x> <source_y> [imagefile] [sourcefile]");
			if (nwords >= 3) {
				if (show_grid) {
					if (plot_recursive_grid("xgrid.dat")==false)
						Complain("could not generate recursive grid");
				}
				double xsource_in, ysource_in, x_in, y_in;
				if (!(ws[1] >> x_in)) Complain("invalid x-position");
				if (!(ws[2] >> y_in)) Complain("invalid y-position");
				if (plot_from_imgpt) {
					lensvector pos,src;
					pos[0] = x_in;
					pos[1] = y_in;
					find_sourcept(pos,src,0,reference_zfactors,default_zsrc_beta_factors);
					xsource_in = src[0];
					ysource_in = src[1];
				} else {
					xsource_in = x_in;
					ysource_in = y_in;
				}
				if ((show_cc) and (plot_critical_curves("crit.dat")==false)) warn("could not plot critical curves");
				if (set_title) plot_title = temp_title;
				if (plot_images_single_source(xsource_in, ysource_in, verbal_mode)==true) {
					if (nwords==5) {
						if (show_cc) {
							if (show_grid) run_plotter_file("image_grid",words[4],range1);
							else run_plotter_file("image",words[4],range1);
						} else {
							if (show_grid) run_plotter_file("image_nocc_grid",words[4],range1);
							else run_plotter_file("image_nocc",words[4],range1);
						}
						if (plot_srcplane) run_plotter_file("source",words[3],range2);
					} else if (nwords==4) {
						if (show_cc) {
							if (show_grid) run_plotter_file("image_grid",words[3],range1);
							else run_plotter_file("image",words[3],range1);
						} else {
							if (show_grid) run_plotter_file("image_nocc_grid",words[3],range1);
							else run_plotter_file("image_nocc",words[3],range1);
						}
					} else {
						// only graphical plotting allowed if filenames not specified
						if (show_cc) {
							if (show_grid) run_plotter("image_grid",range1);
							else run_plotter("image",range1);
						}
						else {
							if (show_grid) run_plotter("image_nocc_grid",range1);
							else run_plotter("image_nocc",range1);
						}
						if (plot_srcplane) run_plotter("source",range2);
					}
				}
				if (set_title) plot_title = "";
			} else Complain("must specify source position (e.g. 'plotimg 3.0 1.2')");
		}
		else if (words[0]=="plotlogkappa")
		{
			bool plot_contours = false;
			int n_contours = 24;
			if (nlens==0) Complain("must specify lens model first");
			int pos;
			if ((nwords > 1) and (pos = words[nwords-1].find("-contour=")) != string::npos) {
				string ncontstring = words[nwords-1].substr(pos+9);
				stringstream ncontstr;
				ncontstr << ncontstring;
				if (!(ncontstr >> n_contours)) Complain("incorrect format for number of contours");
				if (n_contours < 0) Complain("number of contours cannot be negative");
				remove_word(nwords-1);
				plot_contours = true;
			}
			stringstream ncontstr2;
			string ncontstring2;
			ncontstr2 << n_contours;
			ncontstr2 >> ncontstring2;
			if (nwords < 3) {
				if ((nwords==2) and (terminal==TEXT)) {
					plot_logkappa_map(n_image_pixels_x,n_image_pixels_y,words[1],true); // the last option says to ignore the mask (if "true")...maybe allow user to choose?
				} else {
					plot_logkappa_map(n_image_pixels_x,n_image_pixels_y,"lensmap",true); // the last option says to ignore the mask (if "true")...maybe allow user to choose?
					string contstring;
					if (plot_contours) contstring = "ncont=" + ncontstring2; else contstring = "";
					if (nwords==2) {
						run_plotter_file("lensmap_kappalog",words[1],contstring);
					} else {
						run_plotter("lensmap_kappalog",contstring);
					}
				}
			} else Complain("only up to one argument is allowed for 'plotlogkappa' (filename), plus optional 'contour=#'");
		}
		else if (words[0]=="plotlogpot")
		{
			if (nlens==0) Complain("must specify lens model first");
			if (nwords < 3) {
				if ((nwords==2) and (terminal==TEXT)) {
					plot_logpot_map(n_image_pixels_x,n_image_pixels_y,words[1]);
				} else {
					plot_logpot_map(n_image_pixels_x,n_image_pixels_y,"lensmap");
					if (nwords==2) {
						run_plotter_file("lensmap_potlog",words[1],"");
					} else {
						run_plotter("lensmap_potlog");
					}
				}
			} else Complain("only up to one argument is allowed for 'plotlogpot' (filename)");
		}
		else if (words[0]=="plotlogmag")
		{
			if (nwords < 3) {
				if (nlens==0) Complain("must specify lens model first");
				if ((!show_cc) or (plot_critical_curves("crit.dat")==true)) {
					if ((nwords==2) and (terminal==TEXT)) {
						plot_logmag_map(n_image_pixels_x,n_image_pixels_y,words[1]);
					} else {
						plot_logmag_map(n_image_pixels_x,n_image_pixels_y,"lensmap");
						if (nwords==2) {
							if (show_cc) {
								run_plotter_file("lensmap_maglog",words[1],"");
							} else {
								run_plotter_file("lensmap_maglog_nocc",words[1],"");
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
			if (nlens==0) Complain("must specify lens model first");
			if (nwords==1) {
				double re, reav, re_kpc, reav_kpc, arcsec_to_kpc, sigma_cr_kpc, m_ein;
				re = einstein_radius_of_primary_lens(reference_zfactors[lens_redshift_idx[primary_lens_number]],reav);
				arcsec_to_kpc = cosmo.angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
				re_kpc = re*arcsec_to_kpc;
				reav_kpc = reav*arcsec_to_kpc;
				sigma_cr_kpc = cosmo.sigma_crit_kpc(lens_redshift, source_redshift);
				m_ein = sigma_cr_kpc*M_PI*SQR(reav_kpc);
				if (mpi_id==0) {
					cout << "Einstein radius of primary (+ co-centered and/or secondary) lens:\n";
					cout << "r_E_major = " << re << " arcsec, " << re_kpc << " kpc\n";
					cout << "r_E_avg = " << reav << " arcsec, " << reav_kpc << " kpc\n";
					cout << "Mass within average Einstein radius: " << m_ein << " solar masses\n";
				}
			} else if (nwords==2) {
				if (mpi_id==0) {
					int lens_number;
					if (!(ws[1] >> lens_number)) Complain("invalid lens number");
					double re_major_axis, re_average;
					if (get_einstein_radius(lens_number,re_major_axis,re_average)==false) Complain("could not calculate Einstein radius");
					double re_kpc, re_major_kpc, arcsec_to_kpc, sigma_cr_kpc, m_ein;
					arcsec_to_kpc = cosmo.angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
					re_kpc = re_average*arcsec_to_kpc;
					re_major_kpc = re_major_axis*arcsec_to_kpc;
					sigma_cr_kpc = cosmo.sigma_crit_kpc(lens_redshift, source_redshift);
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
			sigma_cr_arcsec = cosmo.sigma_crit_arcsec(lens_redshift, source_redshift);
			sigma_cr_kpc = cosmo.sigma_crit_kpc(lens_redshift, source_redshift);
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
				else if (nwords > 8) Complain("too many parameters in mksrctab");
				else outfile = "sourcexy.in";
				make_source_rectangle(xmin,xmax,xsteps,ymin,ymax,ysteps,outfile);
			} else Complain("mksrctab requires at least 6 parameters: xmin, xmax, xpoints, ymin, ymax, ypoints");
		}
		else if (words[0]=="mksrcgal")
		{
			if (nwords >= 8) {
				double xcenter, ycenter, major_axis, q, angle;
				int n_ellipses, points_per_ellipse;
				bool draw_in_imgplane = false;
				vector<string> args;
				if (extract_word_starts_with('-',1,nwords-1,args)==true)
				{
					int pos;
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-imgplane") draw_in_imgplane = true;
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
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
				make_source_ellipse(xcenter,ycenter,major_axis,q,angle,n_ellipses,points_per_ellipse,draw_in_imgplane,outfile);
			} else Complain("mksrcgal requires at least 7 parameters: xcenter, ycenter, major_axis, q, theta, n_ellipses points_per_ellipse");
		}
		else if (words[0]=="raytracetab")
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
				else if (nwords > 8) { Complain("too many parameters in raytracetab"); }
				else outfile = "sourcexy.in";
				raytrace_image_rectangle(xmin,xmax,xsteps,ymin,ymax,ysteps,outfile);
			} else Complain("raytracetab requires at least 6 parameters: xmin, xmax, xpoints, ymin, ymax, ypoints");
		}
		else if (words[0]=="findimgs")
		{
			if (nlens==0) Complain("must specify lens model first");
			if (nwords == 1)
				plot_images("sourcexy.in", "images.dat", false, verbal_mode);	// default source file
			else if (nwords == 2)
				plot_images(words[1].c_str(), "images.dat", false, verbal_mode);
			else if (nwords == 3)
				plot_images(words[1].c_str(), words[2].c_str(), false, verbal_mode);
			else Complain("invalid number of arguments to command 'findimgs'");
		}
		else if (words[0]=="plotimgs")
		{
			if (nlens==0) Complain("must specify lens model first");
			bool show_multiplicities = false;
			bool show_pixel_data = false;
			bool omit_source = false;
			vector<string> args;
			if (extract_word_starts_with('-',1,nwords-1,args)==true)
			{
				int pos;
				for (int i=0; i < args.size(); i++) {
					if (args[i]=="-showmults") show_multiplicities = true;
					else if (args[i]=="-nosrc") omit_source = true;
					else if (args[i]=="-sbdata") show_pixel_data = true;
					else Complain("argument '" << args[i] << "' not recognized");
				}
			}
			string showmults="";
			if (show_multiplicities) showmults = "showmults";
			if (show_pixel_data) {
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				imgpixel_data_list[0]->plot_surface_brightness("img_pixel",true);
			}


			string range1, range2;
			extract_word_starts_with('[',1,3,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
			extract_word_starts_with('[',1,4,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
			if (((omit_source) or (!plot_srcplane)) and (range2.empty())) { range2 = range1; range1 = ""; }
			if (nwords == 1) {
				if (plot_images("sourcexy.in", "imgs.dat", show_multiplicities, verbal_mode)==true) {	// default source file
					if ((plot_srcplane) and (!omit_source)) run_plotter_range("sources",range1,showmults);
					if (show_cc) {
						if (show_pixel_data) run_plotter_range("imgpixel_imgpts_plural",range2,showmults);
						else run_plotter_range("images",range2,showmults);
					} else {
						if (show_pixel_data) run_plotter_range("imgpixel_imgpts_plural_nocc",range2,showmults);
						else run_plotter_range("images_nocc",range2,showmults);
					}
				}
				else Complain("could not create grid to plot images");
			} else if (nwords == 2) {
				if (terminal == TEXT) {
					plot_images(words[1].c_str(), words[2].c_str(), show_multiplicities, verbal_mode);
				} else if (plot_images("sourcexy.in", "imgs.dat", show_multiplicities, verbal_mode)==true) {
					if (show_cc) {
						if (show_pixel_data) run_plotter_file("imgpixel_imgpts_plural",words[1],range2,showmults);
						else run_plotter_file("images",words[1],range2,showmults);
					} else {
						if (show_pixel_data) run_plotter_file("imgpixel_imgpts_plural_nocc",words[1],range2,showmults);
						else run_plotter_file("images_nocc",range2,words[1],showmults);
					}
				}
			} else if (nwords == 3) {
				if (terminal == TEXT) {
					plot_images(words[1].c_str(), words[2].c_str(), show_multiplicities, verbal_mode);
				} else if (plot_images("sourcexy.in", "imgs.dat", show_multiplicities, verbal_mode)==true) {
					if ((plot_srcplane) and (!omit_source)) run_plotter_file("sources",words[1],range1);
					if (show_cc) {
						if (show_pixel_data) run_plotter_file("imgpixel_imgpts_plural",words[2],range2,showmults);
						else run_plotter_file("images",words[2],range2,showmults);
					} else {
						if (show_pixel_data) run_plotter_file("imgpixel_imgpts_plural_nocc",words[2],range2,showmults);
						else run_plotter_file("images_nocc",range2,words[2],showmults);
					}
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
				if (plot_srcplane) run_plotter_file("sources",words[1],range1);
				if (show_cc) run_plotter_file("images",words[2],range2);
				else run_plotter_file("images_nocc",words[2],range2);
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
			int mask_i=0;
			int band_i=0;
			bool specified_mask = false;
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("mask=")) != string::npos) {
					string mnumstring = words[i].substr(pos+5);
					stringstream mnumstr;
					mnumstr << mnumstring;
					if (!(mnumstr >> mask_i)) Complain("incorrect format for lens redshift");
					if (mask_i < 0) Complain("lens redshift cannot be negative");
					remove_word(i);
					specified_mask = true;
					break;
				}
			}	
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("band=")) != string::npos) {
					string bnumstring = words[i].substr(pos+5);
					stringstream bnumstr;
					bnumstr << bnumstring;
					if (!(bnumstr >> band_i)) Complain("incorrect format for band number");
					if (band_i < 0) Complain("band number cannot be negative");
					remove_word(i);
					break;
				}
			}	

			if (nwords==1) {
				// this needs to be updated to show info about all image pixel grids (at different redshifts) as well as all pixellated sources
				//if (cartesian_srcgrid0 == NULL) cout << "No source surface brightness map has been loaded\n";
				//else cout << "Source surface brightness map is loaded with pixel dimension (" << cartesian_srcgrid0->u_N << "," << cartesian_srcgrid0->w_N << ")\n";
				if ((image_pixel_grids == NULL) or (image_pixel_grids[0] == NULL)) cout << "No image surface brightness map has been loaded\n";
				else cout << "Image surface brightness map is loaded with pixel dimension (" << image_pixel_grids[0]->x_N << "," << image_pixel_grids[0]->y_N << ")\n";
			}
			else if (words[1]=="assign_masks")
			{
				int znum=0;
				if (nwords==2) {
					if (mpi_id==0) print_mask_assignments();
				} else if (nwords==3) {
					if (!(ws[2] >> znum)) Complain("invalid source redshift index");
					if (!assign_mask(band_i,znum,mask_i)) Complain("could not assign mask");
				}
			}
			//else if (words[1]=="savesrc") // DEPRECATED
			//{
				//string filename;
				//if (nwords==2) filename = "src_pixel";
				//else if (nwords==3) {
					//if (!(ws[2] >> filename)) Complain("invalid filename for source surface brightness map");
				//} else Complain("too many arguments to 'sbmap savesrc'");
				//if (cartesian_srcgrid0==NULL) Complain("no source surface brightness map has been created/loaded");
				//cartesian_srcgrid0->store_surface_brightness_grid_data(filename);
			//}
			//else if (words[1]=="loadsrc") // deprecated...source should be loaded from a FITS file anyway
			//{
				//string filename;
				//if (nwords==2) filename = "src_pixel";
				//else if (nwords==3) {
					//if (!(ws[2] >> filename)) Complain("invalid filename for source surface brightness map");
				//} else Complain("too many arguments to 'sbmap loadsrc'");
				//load_source_surface_brightness_grid(filename);
			//}
			else if (words[1]=="loadimg")
			{
				string filename;
				int hdu_indx = 1;
				bool show_header = false;
				double pixsize = default_data_pixel_size;
				double x_offset = 0.0, y_offset = 0.0;
				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					int pos;
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-showhead") show_header = true;
						else if ((pos = args[i].find("-hdu=")) != string::npos) {
							string hdustring = args[i].substr(pos+5);
							stringstream hdustr;
							hdustr << hdustring;
							if (!(hdustr >> hdu_indx)) Complain("incorrect format for HDU index");
							if (hdu_indx <= 0) Complain("HDU index cannot be zero or negative");
						}
						else if ((pos = args[i].find("-pxsize=")) != string::npos) {
							string pxstring = args[i].substr(pos+8);
							stringstream pxstr;
							pxstr << pxstring;
							if (!(pxstr >> pixsize)) Complain("incorrect format for pixel size");
							if (pixsize <= 0) Complain("pixel size cannot be zero or negative");
						}
						else if ((pos = args[i].find("-x_offset=")) != string::npos) {
							string xostring = args[i].substr(pos+10);
							stringstream xostr;
							xostr << xostring;
							if (!(xostr >> x_offset)) Complain("incorrect format for x-offset");
						}
						else if ((pos = args[i].find("-y_offset=")) != string::npos) {
							string yostring = args[i].substr(pos+10);
							stringstream yostr;
							yostr << yostring;
							if (!(yostr >> y_offset)) Complain("incorrect format for y-offset");
						}
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				if (band_i > n_data_bands) Complain("band index cannot be greater than current n_data_bands; to create a new band, set band_i=n_data_bands");

				if (nwords==2) filename = "img_pixel";
				else if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for image surface brightness map");
				} else Complain("too many arguments to 'sbmap loadimg'");
				if (!load_image_surface_brightness_grid(band_i,filename,pixsize,1.0,x_offset,y_offset,hdu_indx,show_header)) Complain("could not load image data");
			}
			else if (words[1]=="saveimg")
			{
				string filename;
				if (nwords==2) filename = "img_pixel";
				else if (nwords>=3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for image surface brightness map");
					if (nwords==3) {
						imgpixel_data_list[band_i]->save_data_fits(filename);
					} else if (nwords==7) {
						double xmin, xmax, ymin, ymax;
						if (!(ws[3] >> xmin)) Complain("invalid xmin argument for 'sbmap saveimg'");
						if (!(ws[4] >> xmax)) Complain("invalid xmax argument for 'sbmap saveimg'");
						if (!(ws[5] >> ymin)) Complain("invalid ymin argument for 'sbmap saveimg'");
						if (!(ws[6] >> ymax)) Complain("invalid ymax argument for 'sbmap saveimg'");
						imgpixel_data_list[band_i]->save_data_fits(filename,true,xmin,xmax,ymin,ymax);
					} else Complain("too many arguments to 'sbmap saveimg'");
				}
			}
			else if (words[1]=="loadmask")
			{
				bool add_mask = false;
				bool subtract_mask = false;
				bool foreground_mask = false;
				bool emask = false;
				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-add") add_mask = true;
						if (args[i]=="-subtract") subtract_mask = true;
						else if ((args[i]=="-fg") or (args[i]=="-fgmask")) foreground_mask = true;
						else if (args[i]=="-emask") emask = true;
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				if ((emask) and (foreground_mask)) Complain("cannot load both emask and foreground mask at the same time");
				string filename;
				if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for mask pixel map");
				} else Complain("too many arguments to 'sbmap loadmask'");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (imgpixel_data_list[band_i]->load_mask_fits(mask_i,filename,foreground_mask,emask,add_mask,subtract_mask)==false) Complain("could not load mask file");
				if (foreground_mask) {
					if (fgmask_padding > 0) {
						imgpixel_data_list[band_i]->expand_foreground_mask(fgmask_padding);
						if (mpi_id==0) cout << "Padding foreground mask by " << fgmask_padding << " neighbors (for convolutions)" << endl;
					}
				}
				//if (mpi_id==0) {
					//if (!foreground_mask) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
					//else {
						//int nfgpix = imgpixel_data_list[band_i]->get_size_of_foreground_mask();
						//cout << "Number of foreground pixels in mask: " << nfgpix << endl;
					//}
				//}
			}
			else if (words[1]=="savemask")
			{
				bool foreground_mask = false;
				bool emask = false;
				int reduce_npx = -1, reduce_npy = -1;
				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-fg") foreground_mask = true;
						else if (args[i]=="-emask") emask = true;
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				if ((emask) and (foreground_mask)) Complain("cannot save both emask and foreground mask at the same time");
				string filename;
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (nwords==2) Complain("output FITS filename for mask is required");
				else if (nwords>=3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for mask file");
					if (nwords==3) {
						imgpixel_data_list[band_i]->save_mask_fits(filename,foreground_mask,emask,mask_i);
					} else if (nwords==7) {
						double xmin, xmax, ymin, ymax;
						if (!(ws[3] >> xmin)) Complain("invalid xmin argument for 'sbmap savemask'");
						if (!(ws[4] >> xmax)) Complain("invalid xmax argument for 'sbmap savemask'");
						if (!(ws[5] >> ymin)) Complain("invalid ymin argument for 'sbmap savemask'");
						if (!(ws[6] >> ymax)) Complain("invalid ymax argument for 'sbmap savemask'");
						imgpixel_data_list[band_i]->save_mask_fits(filename,foreground_mask,emask,mask_i,true,xmin,xmax,ymin,ymax);
					} else Complain("too many arguments to 'sbmap savemask'");
				}
			}
			else if (words[1]=="load_noisemap")
			{
				if (n_data_bands==0) Complain("image pixel data has not been loaded");
				string filename;
				int hdu_indx = 1;
				bool show_header = false;
				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					int pos;
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-showhead") show_header = true;
						else if ((pos = args[i].find("-hdu=")) != string::npos) {
							string hdustring = args[i].substr(pos+5);
							stringstream hdustr;
							hdustr << hdustring;
							if (!(hdustr >> hdu_indx)) Complain("incorrect format for HDU index");
							if (hdu_indx <= 0) Complain("HDU index cannot be zero or negative");
						}
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}

				if (nwords==2) {
					Complain("filename for noise map in FITS format is required (e.g. 'sbmap load_noisemap file.fits')");
				} else if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for noise map");
				} else Complain("too many arguments to 'sbmap load_noisemap'");
				if (!imgpixel_data_list[band_i]->load_noise_map_fits(filename,hdu_indx,show_header)) Complain("could not load noise map fits file '" << filename << "'");
				use_noise_map = true;
			}
			else if (words[1]=="save_noisemap")
			{
				string filename;
				vector<string> args;
				if (nwords==2) Complain("filename for noise map in FITS format is required (e.g. 'sbmap save_noisemap file.fits')");
				if (n_data_bands==0) Complain("image data/noise map has not been loaded or generated");
				else if (nwords>=3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for noise map file");
					if (nwords==3) {
						if (!imgpixel_data_list[band_i]->save_noise_map_fits(filename)) Complain("noise map has not been loaded or generated");
					} else if (nwords==7) {
						double xmin, xmax, ymin, ymax;
						if (!(ws[3] >> xmin)) Complain("invalid xmin argument for 'sbmap save_noisemap'");
						if (!(ws[4] >> xmax)) Complain("invalid xmax argument for 'sbmap save_noisemap'");
						if (!(ws[5] >> ymin)) Complain("invalid ymin argument for 'sbmap save_noisemap'");
						if (!(ws[6] >> ymax)) Complain("invalid ymax argument for 'sbmap save_noisemap'");
						if (!imgpixel_data_list[band_i]->save_noise_map_fits(filename,true,xmin,xmax,ymin,ymax)) Complain("noise map has not been loaded or generated");
					} else Complain("too many arguments to 'sbmap save_noisemap'");
				}
			}
			else if (words[1]=="generate_uniform_noisemap")
			{
				if (background_pixel_noise <= 0) Complain("bg_pixel_noise should be set to a positive nonzero value to generate uniform noise map");
				if (n_data_bands==0) Complain("must load pixel data before generating noise map");
				imgpixel_data_list[band_i]->set_uniform_pixel_noise(background_pixel_noise);
				use_noise_map = true;
			}
			else if (words[1]=="unload_noisemap")
			{
				string filename;
				if (nwords != 2) Complain("no arguments are required for 'sbmap unload_noisemap'");
				if (!use_noise_map) Complain("no noise map has been generated or loaded from FITS file");
				if (n_data_bands > 0) imgpixel_data_list[band_i]->unload_noise_map();
				use_noise_map = false;
			}
			else if (words[1]=="loadpsf")
			{
				string filename;
				bool load_supersampled_psf = false;
				bool cubic_spline = false;
				bool specify_pixsize = false;
				bool show_header = false;
				int hdu_indx = 1;
				bool downsample = false;
				int downsample_fac = 1;
				double psize = -1;
				vector<string> args;
				int pos;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-showhead") show_header = true;
						else if ((pos = args[i].find("-hdu=")) != string::npos) {
							string hdustring = args[i].substr(pos+5);
							stringstream hdustr;
							hdustr << hdustring;
							if (!(hdustr >> hdu_indx)) Complain("incorrect format for HDU index");
							if (hdu_indx <= 0) Complain("HDU index cannot be zero or negative");
						}
						else if (args[i]=="-spline") cubic_spline = true;
						else if (args[i]=="-supersampled") {
							if (!psf_supersampling) Complain("psf_supersampling must be set to 'on' to load supersampled PSF");
							else load_supersampled_psf = true;
						} else if ((pos = args[i].find("-pxsize=")) != string::npos) {
							string sizestring = args[i].substr(pos+8);
							stringstream sizestr;
							sizestr << sizestring;
							if (!(sizestr >> psize)) Complain("incorrect format for pixel noise");
							if (psize < 0) Complain("pixel size value cannot be negative");
							specify_pixsize = true;
						} else if ((pos = args[i].find("-downsample=")) != string::npos) {
							string sizestring = args[i].substr(pos+12);
							stringstream sizestr;
							sizestr << sizestring;
							if (!(sizestr >> downsample_fac)) Complain("incorrect format for pixel noise");
							if (downsample_fac <= 0) Complain("pixel size value cannot be negative or zero");
							downsample = true;
						} else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				if ((cubic_spline) and (load_supersampled_psf)) Complain("cannot spline supersampled PSF matrix");
				if (nwords==2) {
					Complain("filename for PSF in FITS format is required (e.g. 'sbmap loadpsf file.fits')");
				} else if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for PSF matrix");
				} else Complain("too many arguments to 'sbmap loadpsf'");
				if (band_i > n_psf) Complain("band index is higher than n_psf. To create new PSF, set band_i=n_psf");
				if (band_i==n_psf) add_psf();
				if (!psf_list[band_i]->load_psf_fits(filename,hdu_indx,load_supersampled_psf,show_header,verbal_mode and (mpi_id==0))) Complain("could not load PSF fits file '" << filename << "'");
				if (!load_supersampled_psf) {
					if (psf_list[band_i]->psf_spline.is_splined()) psf_list[band_i]->psf_spline.unspline();
					if (cubic_spline) {
						double pixel_xlength, pixel_ylength;
						if (specify_pixsize) {
							pixel_xlength = psize;
							pixel_ylength = psize;
						} else {
							// this code snippet is ugly and gets repeated in a few places. Consolodate and maybe rewrite!
							int primary_grid_indx = band_i*n_extended_src_redshifts;
							if ((image_pixel_grids != NULL) and (image_pixel_grids[primary_grid_indx] != NULL)) {
								pixel_xlength = image_pixel_grids[primary_grid_indx]->pixel_xlength;
								pixel_ylength = image_pixel_grids[primary_grid_indx]->pixel_ylength;
							} else {
								pixel_xlength = grid_xlength / n_image_pixels_x;
								pixel_ylength = grid_ylength / n_image_pixels_y;
							}
						}
						if (psf_list[band_i]->spline_PSF_matrix(pixel_xlength,pixel_ylength)==false) Complain("PSF matrix has not been generated; could not spline");
					}
					if (psf_supersampling) {
						psf_list[band_i]->generate_supersampled_PSF_matrix(downsample,downsample_fac);
						if (mpi_id==0) cout << "Generated supersampled PSF matrix (dimensions: " << psf_list[band_i]->supersampled_psf_npixels_x << " " << psf_list[band_i]->supersampled_psf_npixels_y << ")" << endl;
					}
				}
				if (fft_convolution) cleanup_FFT_convolution_arrays();
			}
			else if (words[1]=="savepsf")
			{
				string filename;
				bool sup = false;
				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-supersampled") {
							if (!psf_supersampling) Complain("psf_supersampling must be set to 'on' to save supersampled PSF");
							else sup = true;
						}
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				if (nwords==2) {
					Complain("filename for PSF in FITS format is required (e.g. 'sbmap savepsf file.fits')");
				} else if (nwords==3) {
					if (!(ws[2] >> filename)) Complain("invalid filename for PSF matrix");
				} else Complain("too many arguments to 'sbmap savepsf'");
				if (band_i >= n_psf) Complain("specified PSF has not been created");
				if (!psf_list[band_i]->save_psf_fits(filename,sup)) Complain("could not save PSF fits file '" << filename << "'");
			}
			else if (words[1]=="plotpsf")
			{
				string filename;
				bool sup = false;
				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-supersampled") {
							if (!psf_supersampling) Complain("psf_supersampling must be set to 'on' to save supersampled PSF");
							else sup = true;
						}
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				if (band_i >= n_psf) Complain("specified PSF has not been created");
				double xstep=1.0, ystep=1.0;
				if (band_i < n_data_bands) {
					xstep = imgpixel_data_list[band_i]->pixel_size;
					ystep = xstep*imgpixel_data_list[band_i]->pixel_xy_ratio;
				}
				if (sup) {
					xstep /= default_imgpixel_nsplit;
					ystep /= default_imgpixel_nsplit;
				}
				//cout << "BAND=" << band_i << endl;
				if ((mpi_id==0) and (!psf_list[band_i]->plot_psf("psfimg",sup,xstep,ystep))) Complain("could not plot PSF fits file '" << filename << "'");
				run_plotter("psfimg");
			}
			else if (words[1]=="unloadpsf")
			{
				string filename;
				if (nwords != 2) Complain("no arguments are required for 'sbmap unloadpsf'");
				if (band_i >= n_psf) Complain("specified PSF has not been created");
				if (!psf_list[band_i]->use_input_psf_matrix) Complain("no psf has been loaded from FITS file");
				psf_list[band_i]->delete_psf_matrix();
				psf_list[band_i]->psf_filename = "";
				if (fft_convolution) cleanup_FFT_convolution_arrays();
			}
			else if ((words[1]=="mkpsf") or (words[1]=="spline_psf"))
			{
				bool mkpsf = false;
				bool cubic_spline = false;
				bool generated_supersampled_psf = false;
				if (words[1]=="mkpsf") mkpsf = true;	
				else if (words[1]=="spline_psf") cubic_spline = true;
				double pixel_xlength, pixel_ylength;
				int primary_grid_indx = band_i*n_extended_src_redshifts;
				if ((image_pixel_grids != NULL) and (image_pixel_grids[primary_grid_indx] != NULL)) {
					pixel_xlength = image_pixel_grids[primary_grid_indx]->pixel_xlength;
					pixel_ylength = image_pixel_grids[primary_grid_indx]->pixel_ylength;
				} else if (band_i < n_data_bands) {
					pixel_xlength = imgpixel_data_list[band_i]->pixel_size;
					pixel_ylength = pixel_xlength*imgpixel_data_list[band_i]->pixel_xy_ratio;
				} else {
					pixel_xlength = grid_xlength / n_image_pixels_x;
					pixel_ylength = grid_ylength / n_image_pixels_y;
				}
				if ((mkpsf) and (nwords==3) and (words[2]=="-spline")) cubic_spline = true;
				if (band_i >= n_psf) Complain("specified PSF has not been created");
				if (mkpsf) {
					if (psf_list[band_i]->generate_PSF_matrix(pixel_xlength,pixel_ylength,false)==false) Complain("could not generate PSF matrix from analytic model");
				}
				if (cubic_spline) {
					if (psf_list[band_i]->spline_PSF_matrix(pixel_xlength,pixel_ylength)==false) Complain("PSF matrix has not been generated; could not spline");
				}
				if ((mkpsf) or (cubic_spline)) {
					if (psf_supersampling) {
						psf_list[band_i]->generate_supersampled_PSF_matrix();
						generated_supersampled_psf = true;
						//if (generate_PSF_matrix(pixel_xlength,pixel_ylength,true)==false) Complain("could not generate supersampled PSF matrix from analytic model");
					}
				}
				if (mpi_id==0) {
					cout << "PSF matrix dimensions: " << psf_list[band_i]->psf_npixels_x << " " << psf_list[band_i]->psf_npixels_y << endl;
					if (generated_supersampled_psf) cout << "Generated supersampled PSF matrix (dimensions: " << psf_list[band_i]->supersampled_psf_npixels_x << " " << psf_list[band_i]->supersampled_psf_npixels_y << ")" << endl;
				}
				if ((mkpsf) or (generated_supersampled_psf)) {
					if (fft_convolution) cleanup_FFT_convolution_arrays();
				}
			}
			else if ((words[1]=="makesrc") or (words[1]=="mksrc") or (words[1]=="mkplotsrc"))
			{
				vector<string> args;
				bool plot_source = false;
				bool make_delaunay_from_sbprofile = false;
				bool use_mask = true;
				bool zoom_in = false;
				bool interpolate = false;
				bool old_auto_srcgrid_npixels = auto_srcgrid_npixels;
				bool scale_to_srcgrid = false;
				int zsrc_i = 0;
				double old_srcgrid_scale;
				//bool changed_srcgrid = false;
				//bool old_auto_srcgrid = false;
				double zoomfactor = 1;
				double delaunay_grid_scale = 1;
				int set_npix = -1; // if negative, doesn't set npix; other wise, it's npix by npix grid
				bool set_title = false;
				string temp_title;
				for (int i=1; i < nwords-1; i++) {
					if (words[i]=="-t") {
						set_title = true;
						set_plot_title(i+1,temp_title);
						remove_word(i);
						break;
					}
				}
				string range;
				extract_word_starts_with('[',2,range); // allow for range to be specified (if it's not, then range is set to "")

				int pos;
				for (int i=2; i < nwords; i++) {
					if ((pos = words[i].find("src=")) != string::npos) {
						string srcnumstring = words[i].substr(pos+4);
						stringstream srcnumstr;
						srcnumstr << srcnumstring;
						if (!(srcnumstr >> zsrc_i)) Complain("incorrect format for lens redshift");
						if (zsrc_i < 0) Complain("source index cannot be negative");
						if (zsrc_i >= n_extended_src_redshifts) Complain("source redshift index does not exist");
						remove_word(i);
						break;
					}
				}

				if (words[1]=="mkplotsrc") { plot_source = true; set_npix = 600; }
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-plot") plot_source = true;
						else if (args[i]=="-interp") interpolate = true;
						else if (args[i]=="-p100") set_npix = 100;
						else if (args[i]=="-p200") set_npix = 200;
						else if (args[i]=="-p300") set_npix = 300;
						else if (args[i]=="-p400") set_npix = 400;
						else if (args[i]=="-p500") set_npix = 500;
						else if (args[i]=="-p600") set_npix = 600;
						else if (args[i]=="-p700") set_npix = 700;
						else if (args[i]=="-p800") set_npix = 800;
						else if (args[i]=="-p1000") set_npix = 1000;
						else if (args[i]=="-p2000") set_npix = 2000;
						else if (args[i]=="-p3000") set_npix = 3000;
						else if (args[i]=="-p4000") set_npix = 4000;
						else if (args[i]=="-nomask") use_mask = false;
						else if (args[i]=="-srcgrid") scale_to_srcgrid = true;
						else if (args[i]=="-x1.5") { zoom_in = true; zoomfactor = 1.5; }
						else if (args[i]=="-x2") { zoom_in = true; zoomfactor = 2; }
						else if (args[i]=="-x4") { zoom_in = true; zoomfactor = 4; }
						else if (args[i]=="-x8") { zoom_in = true; zoomfactor = 8; }
						else if (args[i]=="-delaunay") { make_delaunay_from_sbprofile = true; }
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				bool at_least_one_lensed_src = false;
				for (int k=0; k < n_sb; k++) {
					if ((sb_list[k]->is_lensed) and (sbprofile_redshift_idx[k]==zsrc_i)) { at_least_one_lensed_src = true; break; }
				}
				if ((!at_least_one_lensed_src) and (plot_source)) Complain("at least one analytic lensed source is required for 'sbmap mkplotsrc'");
				else if (!at_least_one_lensed_src) Complain("at least one analytic lensed source is required for 'sbmap mksrc'");

				if ((source_fit_mode==Delaunay_Source) and (!make_delaunay_from_sbprofile) and (!plot_source)) Complain("to make Delaunay source grid from source profiles, use '-delaunay' argument");
				int src_i = -1;
				do {
					for (int i=0; i < n_pixellated_src; i++) {
						if ((pixellated_src_band[i]==band_i) and (pixellated_src_redshift_idx[i]==zsrc_i)) {
							src_i = i;
							break;
						}
					}
					if (src_i < 0) {
						if (mpi_id==0) cout << "Generating pixellated source at corresponding redshift (zsrc=" << extended_src_redshifts[zsrc_i] << ")" << endl;
						add_pixellated_source(extended_src_redshifts[zsrc_i],band_i);
					}
				} while (src_i < 0);
				int imggrid_i = band_i*n_extended_src_redshifts + zsrc_i;

				if (zoom_in) {
					if (make_delaunay_from_sbprofile) {
						delaunay_grid_scale /= zoomfactor;
					} else {
						old_srcgrid_scale = cartesian_srcgrids[src_i]->srcgrid_size_scale;
						cartesian_srcgrids[src_i]->srcgrid_size_scale = 1.0/zoomfactor;
					}
				}
				if (nwords==2) {
					if ((source_fit_mode==Delaunay_Source) and (delaunay_srcgrids==NULL)) Complain("No pixellated source objects have been added to the model");
					if (set_npix > 0) {
						srcgrid_npixels_x = set_npix;
						srcgrid_npixels_y = set_npix;
						auto_srcgrid_npixels = false;
					}
					if (make_delaunay_from_sbprofile) {
						create_sourcegrid_delaunay(src_i,use_mask,verbal_mode);
						if (auto_sourcegrid) find_optimal_sourcegrid_for_analytic_source();
					} else {
						create_sourcegrid_cartesian(band_i,zsrc_i,verbal_mode,use_mask);
						cartesian_srcgrids[src_i]->assign_surface_brightness_from_analytic_source(imggrid_i);
						if ((source_fit_mode==Delaunay_Source) and (delaunay_srcgrids[src_i] != NULL)) {
							cartesian_srcgrids[src_i]->assign_surface_brightness_from_delaunay_grid(delaunay_srcgrids[src_i],true);
						}
					}
					if (plot_source) {
						if ((!make_delaunay_from_sbprofile) and (scale_to_srcgrid)) {
							double xmin,xmax,ymin,ymax;
							cartesian_srcgrids[src_i]->get_grid_dimensions(xmin,xmax,ymin,ymax);
							if (mpi_id==0) cout << "Source grid dimensions: " << xmin << " " << xmax << " " << ymin << " " << ymax << endl;
							stringstream xminstr,yminstr,xmaxstr,ymaxstr;
							string xminstring,yminstring,xmaxstring,ymaxstring;
							xminstr << xmin;
							yminstr << ymin;
							xmaxstr << xmax;
							ymaxstr << ymax;
							xminstr >> xminstring;
							yminstr >> yminstring;
							xmaxstr >> xmaxstring;
							ymaxstr >> ymaxstring;
							range = "[" + xminstring + ":" + xmaxstring + "][" + yminstring + ":" + ymaxstring + "]";
						}

						if (set_title) plot_title = temp_title;
						if (mpi_id==0) {
							if (!make_delaunay_from_sbprofile) cartesian_srcgrids[src_i]->plot_surface_brightness("src_pixel");
							else {
								delaunay_srcgrids[src_i]->plot_surface_brightness("src_pixel",delaunay_grid_scale,set_npix,interpolate,false);
							}
						}
						if ((nlens > 0) and (show_cc) and (plot_critical_curves("crit.dat")==true)) {
							if (make_delaunay_from_sbprofile) run_plotter_range("srcpixel_delaunay",range);
							else run_plotter_range("srcpixel","",range);
						} else {
							if (make_delaunay_from_sbprofile) run_plotter_range("srcpixel_delaunay_nocc","",range);
							else run_plotter_range("srcpixel_nocc","",range);
						}
						if (set_title) plot_title = "";
					}
				} else Complain("no arguments are allowed for 'sbmap makesrc'");
				//if (changed_srcgrid) auto_sourcegrid = old_auto_srcgrid;
				if ((zoom_in) and (!make_delaunay_from_sbprofile)) cartesian_srcgrids[src_i]->srcgrid_size_scale = old_srcgrid_scale;
				auto_srcgrid_npixels = old_auto_srcgrid_npixels;
			}
			//else if (words[1]=="plotsrcgrid")
			//{
				//string outfile;
				//if (nwords==2) {
					//outfile = "srcgrid.dat";
					//plot_source_pixel_grid(-1,outfile.c_str());
				//} else if (nwords==3) {
					//if (!(ws[2] >> outfile)) Complain("invalid output filename for source surface brightness map");
					//plot_source_pixel_grid(-1,outfile.c_str());
				//} else Complain("too many arguments to 'sbmap plotsrcgrid'");
			//}
			else if (words[1]=="plotdata")
			{
				bool show_mask_only = true;
				bool show_isofit = false;
				bool show_extended_mask = false;
				bool show_foreground_mask = false;
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				bool set_title = false;
				bool plot_contours = false;
				int n_contours = 24;
				string temp_title;
				for (int i=1; i < nwords-1; i++) {
					if (words[i]=="-t") {
						set_title = true;
						set_plot_title(i+1,temp_title);
						remove_word(i);
						break;
					}
				}
				if (band_i >= n_data_bands) Complain("specified data band has not been loaded");
				if (mask_i >= imgpixel_data_list[band_i]->n_masks) Complain("mask index has not been created");
				if ((!specified_mask) and (imgpixel_data_list[band_i]->n_masks > 1)) mask_i = -1; // this will tell the imgpixel_data_list[band_i]->plot_surface_brightness function to include all masks

				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					int pos;
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-nomask") show_mask_only = false;
						else if (args[i]=="-isofit") show_isofit = true;
						else if (args[i]=="-emask") show_extended_mask = true;
						else if (args[i]=="-fgmask") show_foreground_mask = true;
						else if ((pos = args[i].find("-contour=")) != string::npos) {
							string ncontstring = args[i].substr(pos+9);
							stringstream ncontstr;
							ncontstr << ncontstring;
							if (!(ncontstr >> n_contours)) Complain("incorrect format for number of contours");
							if (n_contours < 0) Complain("number of contours cannot be negative");
							plot_contours = true;
							show_mask_only = false;
						}
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				stringstream ncontstr2;
				string ncontstring2;
				ncontstr2 << n_contours;
				ncontstr2 >> ncontstring2;

				string range;
				extract_word_starts_with('[',1,nwords-1,range); // allow for ranges to be specified (if it's not, then ranges are set to "")
				if (range.empty()) {
					stringstream xminstream, xmaxstream, yminstream, ymaxstream;
					string xminstr, xmaxstr, yminstr, ymaxstr;
					xminstream << imgpixel_data_list[band_i]->xvals[0]; xminstream >> xminstr;
					yminstream << imgpixel_data_list[band_i]->yvals[0]; yminstream >> yminstr;
					xmaxstream << imgpixel_data_list[band_i]->xvals[imgpixel_data_list[band_i]->npixels_x]; xmaxstream >> xmaxstr;
					ymaxstream << imgpixel_data_list[band_i]->yvals[imgpixel_data_list[band_i]->npixels_y]; ymaxstream >> ymaxstr;
					range = "[" + xminstr + ":" + xmaxstr + "][" + yminstr + ":" + ymaxstr + "]";
				}
				if (set_title) plot_title = temp_title;
				string contstring;
				if (plot_contours) contstring = "ncont=" + ncontstring2; else contstring = "";

				if (nwords == 2) {
					imgpixel_data_list[band_i]->plot_surface_brightness("data_pixel",show_mask_only,show_extended_mask,show_foreground_mask,mask_i);
					if (show_isofit) run_plotter_range("datapixel_ellfit",range,contstring);
					else run_plotter_range("datapixel",range,contstring);
				} else if (nwords == 3) {
					if (terminal==TEXT) {
						imgpixel_data_list[band_i]->plot_surface_brightness(words[2],show_mask_only,show_extended_mask,show_foreground_mask,mask_i);
					}
					else {
						imgpixel_data_list[band_i]->plot_surface_brightness("data_pixel",show_mask_only,show_extended_mask,show_foreground_mask,mask_i);
						run_plotter_file("datapixel",words[2],range,contstring);
					}
				}
				if (set_title) plot_title = "";
			}
			else if (words[1]=="unset_all_pixels")
			{
				if (nwords > 2) Complain("no arguments allowed for command 'sbmap unset_all_pixels'");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				imgpixel_data_list[band_i]->set_no_mask_pixels(mask_i);
			}
			else if (words[1]=="set_posrg_pixels")
			{
				if (nwords > 2) Complain("no arguments allowed for command 'sbmap set_posrg_pixels'");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				imgpixel_data_list[band_i]->set_positive_radial_gradient_pixels(mask_i);
			}
			else if (words[1]=="reset_emask")
			{
				if (nwords > 2) Complain("no arguments allowed for command 'sbmap reset_emask'");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				imgpixel_data_list[band_i]->reset_extended_mask(mask_i);
			}
			else if (words[1]=="set_all_pixels")
			{
				if (nwords > 2) Complain("no arguments allowed for command 'sbmap set_all_pixels'");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (!imgpixel_data_list[band_i]->set_all_mask_pixels(mask_i)) Complain("could not alter mask");
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="invert_mask")
			{
				if (nwords > 2) Complain("no arguments allowed for command 'sbmap invert_mask'");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (!imgpixel_data_list[band_i]->invert_mask(mask_i)) Complain("could not alter mask");
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="create_new_mask")
			{
				if (nwords > 2) Complain("no arguments allowed for command 'sbmap create_new_mask'");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (!imgpixel_data_list[band_i]->create_new_mask()) Complain("could not create new mask");
				if (mpi_id==0) {
					cout << "Number of masks: " << imgpixel_data_list[band_i]->n_masks << endl;
					cout << "Number of pixels in mask " << (imgpixel_data_list[band_i]->n_masks-1) << ": " << imgpixel_data_list[band_i]->n_mask_pixels[imgpixel_data_list[band_i]->n_masks-1] << endl;
				}
			}
			else if (words[1]=="set_neighbor_pixels")
			{
				int ntimes = 1;
				bool exterior = false;
				bool interior = false;
				if (words[nwords-1]=="-ext") {
					exterior = true;
					remove_word(nwords-1);
				} else if (words[nwords-1]=="-int") {
					interior = true;
					remove_word(nwords-1);
				}
				if (nwords > 3) Complain("only one argument allowed for command 'sbmap set_neighbor_pixels' (besides '-ext'/'-int' option)");
				if (nwords == 3) {
					if (!(ws[2] >> ntimes)) Complain("invalid number of neighbor pixels");
				}
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (mask_i >= imgpixel_data_list[band_i]->n_masks) Complain("mask index has not been created");
				for (int i=0; i < ntimes; i++) imgpixel_data_list[band_i]->set_neighbor_pixels(interior,exterior,mask_i);
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="unset_neighbor_pixels")
			{
				int ntimes = 1;
				if (nwords > 3) Complain("only one argument allowed for command 'sbmap unset_neighbor_pixels'");
				if (nwords == 3) {
					if (!(ws[2] >> ntimes)) Complain("invalid number of neighbor pixels");
				}
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (mask_i >= imgpixel_data_list[band_i]->n_masks) Complain("mask index has not been created");
				imgpixel_data_list[band_i]->invert_mask(mask_i);
				for (int i=0; i < ntimes; i++) imgpixel_data_list[band_i]->set_neighbor_pixels(false,false,mask_i);
				imgpixel_data_list[band_i]->invert_mask(mask_i);
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="unset_low_sn_pixels")
			{
				if (nwords != 3) Complain("one argument allowed for command 'sbmap unset_low_sn_pixels' (sb_threshold)");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				double sbthresh;
				if (!(ws[2] >> sbthresh)) Complain("invalid surface brightness threshold");
				if (!imgpixel_data_list[band_i]->unset_low_signal_pixels(sbthresh,mask_i)) Complain("could not alter mask");
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="trim_mask_windows")
			{
				if ((nwords < 3) or (nwords > 4)) Complain("one argument allowed for command 'sbmap trim_mask_windows' (noise threshold)");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				double noise_threshold;
				int threshold_size = 0;
				if (!(ws[2] >> noise_threshold)) Complain("invalid noise threshold for keeping mask windows");
				if (nwords==4) {
					if (!(ws[3] >> threshold_size)) Complain("invalid window size threshold for keeping mask windows");
				}
				if (!imgpixel_data_list[band_i]->assign_mask_windows(noise_threshold,threshold_size,mask_i)) Complain("could not alter mask");
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="set_data_annulus")
			{
				double xc, yc, rmin, rmax, thetamin=0, thetamax=360, xstretch=1.0, ystretch=1.0;
				bool fgmask = false;

				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if ((args[i]=="-fgmask") or (args[i]=="-fg")) fgmask = true;
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}

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
				} else Complain("must specify at least 4 args (xc,yc,rmin,rmax) plus optional thetamin,thetamax, and xstretch,ystretch");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (!imgpixel_data_list[band_i]->set_mask_annulus(xc,yc,rmin,rmax,thetamin,thetamax,xstretch,ystretch,false,fgmask,mask_i)) Complain("coult not alter mask");
				//imgpixel_data_list[band_i]->plot_surface_brightness("data_pixel",true,false,true);
				//run_plotter_range("datapixel","","");

				if (fgmask) {
					if (fgmask_padding > 0) {
						imgpixel_data_list[band_i]->expand_foreground_mask(fgmask_padding);
						if (mpi_id==0) cout << "Padding foreground mask by " << fgmask_padding << " neighbors (for convolutions)" << endl;
					}
				}
				if (mpi_id==0) {
					if (!fgmask) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
					else {
						int nfgpix = imgpixel_data_list[band_i]->get_size_of_foreground_mask();
						cout << "Number of foreground pixels in mask: " << nfgpix << endl;
					}
				}
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="unset_data_annulus")
			{
				double xc, yc, rmin, rmax, thetamin=0, thetamax=360, xstretch=1.0, ystretch=1.0;
				bool fgmask = false;

				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if ((args[i]=="-fgmask") or (args[i]=="-fg")) fgmask = true;
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}

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
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (!imgpixel_data_list[band_i]->set_mask_annulus(xc,yc,rmin,rmax,thetamin,thetamax,xstretch,ystretch,true,fgmask,mask_i)) Complain("could not alter mask"); // the 'true' says to deactivate the pixels, instead of activating them
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
			}
			else if (words[1]=="set_emask_annulus")
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
				} else Complain("must specify at least 4 args (xc,yc,rmin,rmax) plus optional thetamin,thetamax, and xstretch,ystretch");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (!imgpixel_data_list[band_i]->set_extended_mask_annulus(xc,yc,rmin,rmax,thetamin,thetamax,xstretch,ystretch,mask_i)) Complain("could not alter extended mask");
				if (mpi_id==0) cout << "Number of pixels in extended mask: " << imgpixel_data_list[band_i]->get_size_of_extended_mask(mask_i) << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="remove_mask_overlap")
			{
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				imgpixel_data_list[band_i]->remove_overlapping_pixels_from_other_masks(mask_i);
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="activate_partner_imgpixels")
			{
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				imgpixel_data_list[band_i]->activate_partner_image_pixels(mask_i,false);
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
			}
			else if (words[1]=="unset_emask_annulus")
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
				} else Complain("must specify at least 4 args (xc,yc,rmin,rmax) plus optional thetamin,thetamax, and xstretch,ystretch");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				if (!imgpixel_data_list[band_i]->set_extended_mask_annulus(xc,yc,rmin,rmax,thetamin,thetamax,xstretch,ystretch,true,mask_i)) Complain("could not alter extended mask");
				if (mpi_id==0) cout << "Number of pixels in extended mask: " << imgpixel_data_list[band_i]->get_size_of_extended_mask(mask_i) << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="set_fgmask_to_primary")
			{
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				imgpixel_data_list[band_i]->set_foreground_mask_to_primary_mask();
			}
			else if (words[1]=="set_fgmask_annulus")
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
				} else Complain("must specify at least 4 args (xc,yc,rmin,rmax) plus optional thetamin,thetamax, and xstretch,ystretch");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				imgpixel_data_list[band_i]->set_foreground_mask_annulus(xc,yc,rmin,rmax,thetamin,thetamax,xstretch,ystretch,false);
				if (mpi_id==0) cout << "Number of pixels in foreground mask: " << imgpixel_data_list[band_i]->get_size_of_foreground_mask() << endl;
			}
			else if (words[1]=="unset_fgmask_annulus")
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
				} else Complain("must specify at least 4 args (xc,yc,rmin,rmax) plus optional thetamin,thetamax, and xstretch,ystretch");
				if (n_data_bands==0) Complain("no image pixel data has been loaded");
				imgpixel_data_list[band_i]->set_foreground_mask_annulus(xc,yc,rmin,rmax,thetamin,thetamax,xstretch,ystretch,true);
				if (mpi_id==0) cout << "Number of pixels in foreground mask: " << imgpixel_data_list[band_i]->get_size_of_foreground_mask() << endl;
			}
			else if (words[1]=="set_data_window")
			{
				double xmin, xmax, ymin, ymax;
				if (nwords == 6) {
					if (!(ws[2] >> xmin)) Complain("invalid rectangle xmin");
					if (!(ws[3] >> xmax)) Complain("invalid rectangle xmax");
					if (!(ws[4] >> ymin)) Complain("invalid rectangle ymin");
					if (!(ws[5] >> ymax)) Complain("invalid rectangle ymax");
					if (n_data_bands==0) Complain("no image pixel data has been loaded");
					if (!imgpixel_data_list[band_i]->set_mask_window(xmin,xmax,ymin,ymax,mask_i)) Complain("could not alter mask");
				} else Complain("must specify 4 arguments (xmin,xmax,ymin,ymax) for 'sbmap set_data_window'");
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="unset_data_window")
			{
				double xmin, xmax, ymin, ymax;
				if (nwords == 6) {
					if (!(ws[2] >> xmin)) Complain("invalid rectangle xmin");
					if (!(ws[3] >> xmax)) Complain("invalid rectangle xmax");
					if (!(ws[4] >> ymin)) Complain("invalid rectangle ymin");
					if (!(ws[5] >> ymax)) Complain("invalid rectangle ymax");
					if (n_data_bands==0) Complain("no image pixel data has been loaded");
					if (!imgpixel_data_list[band_i]->set_mask_window(xmin,xmax,ymin,ymax,true,mask_i)) Complain("could not alter mask"); // the 'true' says to deactivate the pixels, instead of activating them
				} else Complain("must specify 4 arguments (xmin,xmax,ymin,ymax) for 'sbmap unset_data_window'");
				if (mpi_id==0) cout << "Number of pixels in mask: " << imgpixel_data_list[band_i]->n_mask_pixels[mask_i] << endl;
				if (n_extended_src_redshifts > 0) update_imggrid_mask_values(mask_i);
			}
			else if (words[1]=="find_noise")
			{
				double xmin, xmax, ymin, ymax, sig_sb, mean_sb;
				if (nwords == 6) {
					if (!(ws[2] >> xmin)) Complain("invalid rectangle xmin");
					if (!(ws[3] >> xmax)) Complain("invalid rectangle xmax");
					if (!(ws[4] >> ymin)) Complain("invalid rectangle ymin");
					if (!(ws[5] >> ymax)) Complain("invalid rectangle ymax");
					if (n_data_bands==0) Complain("no image pixel data has been loaded");
					if (!imgpixel_data_list[band_i]->estimate_pixel_noise(xmin,xmax,ymin,ymax,sig_sb,mean_sb,mask_i)) Complain("could not find pixel noise in mask");;
					if (mpi_id==0) {
						cout << "Mean surface brightness in mask: " << mean_sb << endl;
						cout << "Dispersion of surface brightness in mask: " << sig_sb << endl;
						cout << endl;
					}
				} else Complain("must specify 4 arguments (xmin,xmax,ymin,ymax) for 'sbmap find_noise'");
			}
			else if (words[1]=="plotimg")
			{
				bool replot = false;
				bool plot_residual = false;
				bool normalize_sb = false;
				bool plot_foreground_only = false;
				bool omit_foreground = false;
				bool show_all_pixels = false;
				bool show_extended_mask = false;
				bool show_foreground_mask = false;
				bool exclude_ptimgs = false; // by default, we include point images if n_ptsrc > 0
				//if (include_fgmask_in_inversion) show_foreground_mask = true; // no reason not to show emask if we're including it in the inversion
				bool plot_fits = false;
				bool omit_source_plot = true; // changed this to false because it's annoying to have the source plot come out when you really just want e.g. residuals.
				bool offload_to_data = false;
				bool omit_cc = false;
				bool old_cc_setting = show_cc;
				bool subcomp = false;
				bool plot_log = false;
				bool show_noise_thresh = false;
				bool set_title = false;
				bool plot_contours = false;
				bool add_specific_noise = false;
				bool add_noise = false;
				bool show_only_ptimgs = false;
				bool show_current_sb = false; // if true, will plot the surface brightness that is currently stored in the image_pixel_grids
				bool include_imgpts = false; // this is to overlay additional lensed points from "mksrcgal" or "mksrctab" command
				bool show_only_first_order = false;
				bool no_first_order_potential_corrections = false;
				bool old_first_order_sb_correction = first_order_sb_correction;
				bool simulate_noise_setting = simulate_pixel_noise;
				string cbstring = "";
				double pnoise = 0;
				int n_contours = 24;
				string temp_title;
				int zsrc_i = -1;
				if (n_extended_src_redshifts==1) zsrc_i = 0;
				for (int i=1; i < nwords-1; i++) {
					if (words[i]=="-t") {
						set_title = true;
						set_plot_title(i+1,temp_title);
						remove_word(i);
						break;
					}
				}
				int pos;
				for (int i=2; i < nwords; i++) {
					if ((pos = words[i].find("src=")) != string::npos) {
						string srcnumstring = words[i].substr(pos+4);
						stringstream srcnumstr;
						srcnumstr << srcnumstring;
						if (!(srcnumstr >> zsrc_i)) Complain("incorrect format for source index");
						if (zsrc_i < 0) Complain("source index cannot be negative");
						if (zsrc_i >= n_extended_src_redshifts) Complain("source redshift index does not exist");
						remove_word(i);
						break;
					}
				}

				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-replot") replot = true;
						else if ((args[i]=="-res") or (args[i]=="-residual")) plot_residual = true;
						else if (args[i]=="-nres") { plot_residual = true; normalize_sb = true; }
						else if (args[i]=="-nres6") { plot_residual = true; normalize_sb = true; cbstring = "cb6"; }
						else if (args[i]=="-nres5") { plot_residual = true; normalize_sb = true; cbstring = "cb5"; }
						else if (args[i]=="-nres4") { plot_residual = true; normalize_sb = true; cbstring = "cb4"; }
						else if (args[i]=="-nres3") { plot_residual = true; normalize_sb = true; cbstring = "cb3"; }
						//else if (args[i]=="-resns") { plot_residual = true; omit_source_plot = true; } // shortcut argument to plot residuals but not source
						//else if (args[i]=="-nresns") { plot_residual = true; normalize_sb = true; omit_source_plot = true; } // shortcut argument to plot residuals but not source
						else if (args[i]=="-current") show_current_sb = true;
						else if (args[i]=="-norm") normalize_sb = true;
						else if (args[i]=="-thresh") show_noise_thresh = true;
						else if (args[i]=="-log") plot_log = true;
						else if (args[i]=="-fg") plot_foreground_only = true;
						else if (args[i]=="-nofg") omit_foreground = true;
						else if (args[i]=="-nomask") show_all_pixels = true;
						else if (args[i]=="-fgmask") show_foreground_mask = true;
						else if (args[i]=="-emask") show_extended_mask = true;
						else if (args[i]=="-fits") plot_fits = true;
						else if ((args[i]=="-showsrc") or (args[i]=="-showsrcplot")) omit_source_plot = false;
						else if (args[i]=="-noptsrc") exclude_ptimgs = true;
						else if (args[i]=="-onlyptsrc") show_only_ptimgs = true;
						else if (args[i]=="-no1storder") no_first_order_potential_corrections = true;
						else if (args[i]=="-only1storder") show_only_first_order = true;
						else if (args[i]=="-nocc") { omit_cc = true; show_cc = false; }
						else if (args[i]=="-mkdata") offload_to_data = true;
						else if (args[i]=="-subcomp") subcomp = true;
						else if (args[i]=="-imgpts") include_imgpts = true;
						else if (args[i]=="-pnoise") {
							add_noise = true;
							simulate_pixel_noise = true;
						}
						else if ((pos = args[i].find("-pnoise=")) != string::npos) {
							if (use_noise_map) Complain("specific pixel noise cannot be given if noise map is being used");
							string noisestring = args[i].substr(pos+8);
							if (noisestring=="data") pnoise = background_pixel_noise;
							else {
								stringstream noisestr;
								noisestr << noisestring;
								if (!(noisestr >> pnoise)) Complain("incorrect format for pixel noise");
								if (pnoise < 0) Complain("pixel noise value cannot be negative");
							}
							add_specific_noise = true;
							if (!simulate_pixel_noise) simulate_pixel_noise = true;
						}
						else if ((pos = args[i].find("-contour=")) != string::npos) {
							string ncontstring = args[i].substr(pos+9);
							stringstream ncontstr;
							ncontstr << ncontstring;
							if (!(ncontstr >> n_contours)) Complain("incorrect format for number of contours");
							if (n_contours < 0) Complain("number of contours cannot be negative");
							plot_contours = true;
							show_all_pixels = true;
						}
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				stringstream ncontstr2;
				string ncontstring2;
				ncontstr2 << n_contours;
				ncontstr2 >> ncontstring2;
				if ((replot) and (plot_fits)) Complain("Cannot use 'replot' option when plotting to fits files");

				if ((show_current_sb) and (outside_sb_prior)) Complain("cannot use '-current' option if outside_sb_prior is set to 'on'");

				if ((first_order_sb_correction) and (no_first_order_potential_corrections)) {
					first_order_sb_correction = false;
					if ((n_pixellated_lens > 0) and (lensgrids)) {
						for (int i=0; i < n_pixellated_lens; i++) {
							lensgrids[i]->include_in_lensing_calculations = true; // now allow for potential corrections to be included in ray-tracing, rather than using first order approximation
						}
					}
				}

				if (nlens==0) {
					if ((n_sb==0) and (n_ptsrc==0)) {
						Complain("must specify lens/source model first");
					} else {
						bool all_unlensed = true;
						bool at_least_one_mge = false;
						for (int i=0; i < n_sb; i++) {
							if (sb_list[i]->is_lensed) all_unlensed = false;
							if (sb_list[i]->sbtype==MULTI_GAUSSIAN_EXPANSION) at_least_one_mge = true;
						}
						if (!all_unlensed) {
							Complain("must specify lens model first, since lensed sources are present");
						} else {
							if (!at_least_one_mge) plot_foreground_only = true; // since there are only foreground sources
							omit_cc = true;
							show_cc = false;
						}
					}
				}
				if ((exclude_ptimgs) and (show_only_ptimgs)) Complain("cannot both exclude point images and show only point images");
				//if ((source_fit_mode==Cartesian_Source) and (cartesian_srcgrids == NULL)) Complain("No source surface brightness map has been loaded");
				//if ((source_fit_mode==Delaunay_Source) and (delaunay_srcgrids == NULL)) Complain("No pixellated soure objects have been created");
				if (((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source))) omit_source_plot = true;
				bool old_plot_srcplane = plot_srcplane;
				if (omit_source_plot) plot_srcplane = false;
				string range1, range2;
				extract_word_starts_with('[',1,nwords-1,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
				extract_word_starts_with('[',1,nwords-1,range2); // allow for ranges to be specified (if it's not, then ranges are set to "")
				if ((!plot_srcplane) and (range2.empty())) { range2 = range1; range1 = ""; }
				int src_i = -1;
				if (pixellated_src_redshift_idx != NULL) {
					for (int i=0; i < n_pixellated_src; i++) {
						if ((pixellated_src_band[i]==band_i) and (pixellated_src_redshift_idx[i]==zsrc_i)) {
							src_i = i;
							break;
						}
					}
				}
				if ((nwords != 3) and (plot_srcplane) and (range1 == "")) { // if nwords==3, then source plane isn't being plotted
					double xmin,xmax,ymin,ymax;
					if ((source_fit_mode==Cartesian_Source) and (src_i >= 0)) {
						cartesian_srcgrids[src_i]->get_grid_dimensions(xmin,xmax,ymin,ymax);
					} else {
						double xwidth_adj = sourcegrid_xmax-sourcegrid_xmin;
						double ywidth_adj = sourcegrid_ymax-sourcegrid_ymin;
						double srcgrid_xc, srcgrid_yc;
						//delaunay_srcgrids[zsrc_i]->find_centroid(srcgrid_xc,srcgrid_yc);
						srcgrid_xc = (sourcegrid_xmax + sourcegrid_xmin)/2;
						srcgrid_yc = (sourcegrid_ymax + sourcegrid_ymin)/2;
						xmin = srcgrid_xc - xwidth_adj/2;
						xmax = srcgrid_xc + xwidth_adj/2;
						ymin = srcgrid_yc - ywidth_adj/2;
						ymax = srcgrid_yc + ywidth_adj/2;
					}
					stringstream xminstr,yminstr,xmaxstr,ymaxstr;
					string xminstring,yminstring,xmaxstring,ymaxstring;
					xminstr << xmin;
					yminstr << ymin;
					xmaxstr << xmax;
					ymaxstr << ymax;
					xminstr >> xminstring;
					yminstr >> yminstring;
					xmaxstr >> xmaxstring;
					ymaxstr >> ymaxstring;
					range1 = "[" + xminstring + ":" + xmaxstring + "][" + yminstring + ":" + ymaxstring + "]";
				}
				if ((range2.empty()) and (n_data_bands > 0)) {
					stringstream xminstream, xmaxstream, yminstream, ymaxstream;
					string xminstr, xmaxstr, yminstr, ymaxstr;
					xminstream << imgpixel_data_list[band_i]->xvals[0]; xminstream >> xminstr;
					yminstream << imgpixel_data_list[band_i]->yvals[0]; yminstream >> yminstr;
					xmaxstream << imgpixel_data_list[band_i]->xvals[imgpixel_data_list[band_i]->npixels_x]; xmaxstream >> xmaxstr;
					ymaxstream << imgpixel_data_list[band_i]->yvals[imgpixel_data_list[band_i]->npixels_y]; ymaxstream >> ymaxstr;
					range2 = "[" + xminstr + ":" + xmaxstr + "][" + yminstr + ":" + ymaxstr + "]";
				}

				if (n_extended_src_redshifts==0) Complain("no extended source redshifts have been created (do 'sbmap invert' to create automatically)");

				int imggrid_i = band_i*n_extended_src_redshifts + zsrc_i;
				bool foundcc = true;
				bool plotted_src = false;
				double old_pnoise;
				if (add_specific_noise) {
					old_pnoise = background_pixel_noise;
					background_pixel_noise = pnoise;
					//cout << "CHECKING? " << zsrc_i << " " << imggrid_i << " " << n_image_pixel_grids << endl;
					if ((image_pixel_grids != NULL) and (imggrid_i < n_image_pixel_grids) and (image_pixel_grids[imggrid_i] != NULL)) image_pixel_grids[imggrid_i]->setup_noise_map(this);
				}
				if ((include_fgmask_in_inversion) and (mpi_id==0)) cout << "NOTE: Showing foreground mask by default, since include_fgmask_in_inversion = true" << endl;
				if ((show_cc) and (zsrc_i >= 0)) create_grid(false,extended_src_zfactors[zsrc_i],extended_src_beta_factors[zsrc_i],zsrc_i);
				if (include_imgpts) {
					if (!plot_images("sourcexy.in", "imgs.dat", false, verbal_mode)==true) Complain("could not create grid to plot images");
				}

				if ((!show_cc) or (plot_fits) or ((foundcc = plot_critical_curves("crit.dat"))==true)) {
					string contstring;
					if (plot_contours) contstring = "ncont=" + ncontstring2; else contstring = "";
					if (set_title) plot_title = temp_title;
					if (nwords == 2) {
						if (plot_fits) Complain("file name for FITS file must be specified");
						if ((replot) or (plot_lensed_surface_brightness("img_pixel",band_i,plot_fits,plot_residual,plot_foreground_only,omit_foreground,show_all_pixels,normalize_sb,offload_to_data,show_extended_mask,show_foreground_mask,show_noise_thresh,exclude_ptimgs,show_only_ptimgs,zsrc_i,show_only_first_order,plot_log,show_current_sb)==true)) {
							//if ((subcomp) and (show_cc)) {
								//if (plotcrit_exclude_subhalo("crit0.dat",nlens-1)==false) Complain("could not generate critical curves without subhalo");
							//}
							if (!offload_to_data) {
								if (!replot) {
									if ((source_fit_mode==Cartesian_Source) and (src_i >= 0)) {
										if (cartesian_srcgrids[src_i] != NULL) {
											cartesian_srcgrids[src_i]->plot_surface_brightness("src_pixel");
											plotted_src = true;
										}
									} else if ((source_fit_mode==Delaunay_Source) and (src_i >= 0)) {
										if (delaunay_srcgrids[src_i] != NULL) {
											delaunay_srcgrids[src_i]->plot_surface_brightness("src_pixel",1,600,false);
											plotted_src = true;
										}
									}
								}
								if (show_cc) {
									if ((plot_srcplane) and (plotted_src)) {
										if (source_fit_mode==Delaunay_Source) {
											if (include_imgpts) run_plotter_range("srcpixel_delaunay_srcpts_plural",range1);
											else run_plotter_range("srcpixel_delaunay",range1);
										}
										else run_plotter_range("srcpixel",range1);
									}
									if (subcomp) run_plotter_range("imgpixel_comp",range2,contstring,cbstring);
									else if (include_imgpts) run_plotter_range("imgpixel_imgpts_plural",range2,contstring,cbstring);
									else run_plotter_range("imgpixel",range2,contstring,cbstring);
								} else {
									if ((plot_srcplane) and (plotted_src)) {
										if (source_fit_mode==Delaunay_Source) {
											if (include_imgpts) run_plotter_range("srcpixel_delaunay_srcpts_plural_nocc",range1);
											else run_plotter_range("srcpixel_delaunay_nocc",range1);
										}
										else run_plotter_range("srcpixel_nocc",range1);
									}
									if (include_imgpts) run_plotter_range("imgpixel_imgpts_plural_nocc",range2,contstring,cbstring);
									else run_plotter_range("imgpixel_nocc",range2,contstring,cbstring);
								}
							}
						} else {
							Complain("Plotting failed");
						}
					} else if (nwords == 3) {
						if ((terminal==TEXT) or (plot_fits)) {
							if (!replot) plot_lensed_surface_brightness(words[2],band_i,plot_fits,plot_residual,plot_foreground_only,omit_foreground,show_all_pixels,normalize_sb,offload_to_data,show_extended_mask,show_foreground_mask,show_noise_thresh,exclude_ptimgs,show_only_ptimgs,zsrc_i,show_only_first_order,plot_log,show_current_sb);
						}
						else if ((replot) or (plot_lensed_surface_brightness("img_pixel",band_i,plot_fits,plot_residual,plot_foreground_only,omit_foreground,show_all_pixels,normalize_sb,offload_to_data,show_extended_mask,show_foreground_mask,show_noise_thresh,exclude_ptimgs,show_only_ptimgs,zsrc_i,show_only_first_order,plot_log,show_current_sb)==true)) {
							if (show_cc) {
								if (subcomp) run_plotter_file("imgpixel_comp",words[2],range2,contstring,cbstring);
								else if (include_imgpts) run_plotter_file("imgpixel_imgpts_plural",words[2],range2,contstring,cbstring);
								else run_plotter_file("imgpixel",words[2],range2,contstring,cbstring);
							} else {
								if (include_imgpts) run_plotter_file("imgpixel_imgpts_plural_nocc",words[2],range2,contstring,cbstring);
								else run_plotter_file("imgpixel_nocc",words[2],range2,contstring,cbstring);
							}
						} else Complain("Plotting failed");
					} else if (nwords == 4) {
						if ((terminal==TEXT) or (plot_fits)) {
							if (!replot) {
								plot_lensed_surface_brightness(words[3],band_i,plot_fits,plot_residual,plot_foreground_only,omit_foreground,show_all_pixels,normalize_sb,offload_to_data,show_extended_mask,show_foreground_mask,show_noise_thresh,exclude_ptimgs,show_only_ptimgs,zsrc_i,show_only_first_order,plot_log,show_current_sb);
								if ((plotted_src) and (mpi_id==0) and (src_i >= 0)) cartesian_srcgrids[src_i]->plot_surface_brightness(words[2]);
							}
						}
						else if ((replot) or (plot_lensed_surface_brightness("img_pixel",band_i,plot_fits,plot_residual,plot_foreground_only,omit_foreground,show_all_pixels,normalize_sb,offload_to_data,show_extended_mask,show_foreground_mask,show_noise_thresh,exclude_ptimgs,show_only_ptimgs,zsrc_i,show_only_first_order,plot_log,show_current_sb)==true)) {
							if ((!replot) and (plotted_src) and (mpi_id==0) and (src_i >= 0)) { cartesian_srcgrids[src_i]->plot_surface_brightness("src_pixel"); }
							if (show_cc) {
								if (subcomp) run_plotter_file("imgpixel_comp",words[3],range2,contstring,cbstring);
								else if (include_imgpts) run_plotter_file("imgpixel_imgpts_plural",words[3],range2,contstring,cbstring);
								else run_plotter_file("imgpixel",words[3],range2,contstring,cbstring);
								if ((plot_srcplane) and (plotted_src)) run_plotter_file("srcpixel",words[2],range1,contstring,cbstring);
							} else {
								if (include_imgpts) run_plotter_file("imgpixel_imgpts_plural_nocc",words[3],range2,contstring,cbstring);
								else run_plotter_file("imgpixel_nocc",words[3],range2,contstring,cbstring);
								if ((plot_srcplane) and (plotted_src)) run_plotter_file("srcpixel_nocc",words[2],range1);
							}
						} else Complain("Plotting failed");
					} else Complain("invalid number of arguments to 'sbmap plotimg'");
				} else if (!foundcc) Complain("could not find critical curves");
				reset_grid();
				if (add_specific_noise) {
					background_pixel_noise = old_pnoise;
					if ((image_pixel_grids != NULL) and (imggrid_i < n_image_pixel_grids) and (image_pixel_grids[imggrid_i] != NULL)) image_pixel_grids[imggrid_i]->setup_noise_map(this);
				}
				if ((add_noise) or (add_specific_noise)) {
					if (!simulate_noise_setting) simulate_pixel_noise = false;
				}
				if (omit_source_plot) plot_srcplane = old_plot_srcplane;
				if (omit_cc) show_cc = old_cc_setting;
				if (no_first_order_potential_corrections) {
					first_order_sb_correction = old_first_order_sb_correction;
					if (first_order_sb_correction) {
						if ((n_pixellated_lens > 0) and (lensgrids)) {
							for (int i=0; i < n_pixellated_lens; i++) {
								lensgrids[i]->include_in_lensing_calculations = false;
							}
						}
					}
				}
				if (set_title) plot_title = "";
			}
			else if (words[1]=="plotsrc")
			{
				int set_npix = 600; // this is only relevant for a Delaunay source, for which the plotting resolution needs to be given (since it's plotted to Cartesian pixels)
				bool delaunay = false;
				bool zoom_in = false;
				double zoomfactor = 1;
				bool interpolate = false;
				double old_srcgrid_scale;
				double delaunay_grid_scale = 1;
				bool set_title = false;
				bool plot_fits = false;
				bool omit_caustics = false;
				bool old_caustics_setting = show_cc;
				bool plot_mag = false;
				bool include_srcpts = false;
				bool show_raytraced_pts = false; // if true, show all raytraced points (subpixels, if splitting is on) instead of source pixels
				string temp_title;
				int zsrc_i = 0;
				for (int i=1; i < nwords-1; i++) {
					if (words[i]=="-t") {
						set_title = true;
						set_plot_title(i+1,temp_title);
						remove_word(i);
						break;
					}
				}
				int pos;
				for (int i=2; i < nwords; i++) {
					if ((pos = words[i].find("src=")) != string::npos) {
						string srcnumstring = words[i].substr(pos+4);
						stringstream srcnumstr;
						srcnumstr << srcnumstring;
						if (!(srcnumstr >> zsrc_i)) Complain("incorrect format for lens redshift");
						if (zsrc_i < 0) Complain("source index cannot be negative");
						if (zsrc_i >= n_extended_src_redshifts) Complain("source redshift index does not exist");
						remove_word(i);
						break;
					}
				}

				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if (args[i]=="-interp") interpolate = true;
						else if (args[i]=="-mag") plot_mag = true;
						else if (args[i]=="-n100") set_npix = 100;
						else if (args[i]=="-n200") set_npix = 200;
						else if (args[i]=="-n300") set_npix = 300;
						else if (args[i]=="-n400") set_npix = 400;
						else if (args[i]=="-n500") set_npix = 500;
						else if (args[i]=="-n600") set_npix = 600;
						else if (args[i]=="-n700") set_npix = 700;
						else if (args[i]=="-n800") set_npix = 800;
						else if (args[i]=="-n900") set_npix = 900;
						else if (args[i]=="-n1000") set_npix = 1000;
						else if (args[i]=="-n2000") set_npix = 2000;
						else if (args[i]=="-n3000") set_npix = 3000;
						else if (args[i]=="-n4000") set_npix = 4000;
						else if (args[i]=="-fits") plot_fits = 4000;
						else if (args[i]=="-x1.5") { zoom_in = true; zoomfactor = 1.5; }
						else if (args[i]=="-x2") { zoom_in = true; zoomfactor = 2; }
						else if (args[i]=="-x4") { zoom_in = true; zoomfactor = 4; }
						else if (args[i]=="-x8") { zoom_in = true; zoomfactor = 8; }
						else if (args[i]=="-srcpts") { include_srcpts = true; }
						else if (args[i]=="-show_raytraced_pts") show_raytraced_pts = true;
						else if ((args[i]=="-nocaust") or (args[i]=="-nocc")) { omit_caustics = true; show_cc = false; }
						//else if (args[i]=="-nomask") use_mask = false;
						//else if (args[i]=="-srcgrid") scale_to_srcgrid = true;
						//else if (args[i]=="-delaunay") { delaunay = true; }
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}

				if ((source_fit_mode!=Cartesian_Source) and (source_fit_mode!=Delaunay_Source)) Complain("'sbmap plotsrc' is for pixellated sources (cartesian/delaunay); for sbprofile/shapelet mode, use 'sbmap mkplotsrc'");
				if (source_fit_mode==Delaunay_Source) delaunay = true;
				if (srcgrids == NULL) Complain("No pixellated source objects have been created");

				int src_i = -1;
				if ((source_fit_mode==Delaunay_Source) or (source_fit_mode==Cartesian_Source)) {
					for (int i=0; i < n_pixellated_src; i++) {
						if ((pixellated_src_band[i]==band_i) and (pixellated_src_redshift_idx[i]==zsrc_i)) {
							src_i = i;
							break;
						}
					}
				}
				if (src_i < 0) Complain("specified pixellated source number does not exist");
				if ((source_fit_mode==Delaunay_Source) and (delaunay_srcgrids[src_i]->get_n_srcpts()==0)) Complain("Delaunay grid has not been created for specified source (no inversion has been done)");
				else if ((source_fit_mode==Cartesian_Source) and (cartesian_srcgrids[src_i]->levels==0)) Complain("Cartesian grid has not been constructed for specified source (no inversion has been done)");

				if (zoom_in) {
					if (delaunay) {
						delaunay_grid_scale /= zoomfactor;
					} else {
						old_srcgrid_scale = cartesian_srcgrids[src_i]->srcgrid_size_scale;
						cartesian_srcgrids[src_i]->srcgrid_size_scale = 1.0/zoomfactor;
					}
				}

				if (include_srcpts) {
					if (!plot_images("sourcexy.in", "imgs.dat", false, verbal_mode)==true) Complain("could not create grid to plot images");
				}

				string range1 = "";
				extract_word_starts_with('[',1,nwords-1,range1); // allow for ranges to be specified (if it's not, then ranges are set to "")
				if (range1 == "") {
					double xmin,xmax,ymin,ymax;
					if (source_fit_mode==Cartesian_Source) {
						cartesian_srcgrids[src_i]->get_grid_dimensions(xmin,xmax,ymin,ymax);
					} else {
						double xwidth_adj = delaunay_grid_scale*(delaunay_srcgrids[src_i]->srcgrid_xmax-delaunay_srcgrids[src_i]->srcgrid_xmin);
						double ywidth_adj = delaunay_grid_scale*(delaunay_srcgrids[src_i]->srcgrid_ymax-delaunay_srcgrids[src_i]->srcgrid_ymin);
						double srcgrid_xc, srcgrid_yc;
						//delaunay_srcgrid->find_centroid(srcgrid_xc,srcgrid_yc);
						srcgrid_xc = (delaunay_srcgrids[src_i]->srcgrid_xmax + delaunay_srcgrids[src_i]->srcgrid_xmin)/2;
						srcgrid_yc = (delaunay_srcgrids[src_i]->srcgrid_ymax + delaunay_srcgrids[src_i]->srcgrid_ymin)/2;
						xmin = srcgrid_xc - xwidth_adj/2;
						xmax = srcgrid_xc + xwidth_adj/2;
						ymin = srcgrid_yc - ywidth_adj/2;
						ymax = srcgrid_yc + ywidth_adj/2;
					}
					stringstream xminstr,yminstr,xmaxstr,ymaxstr;
					string xminstring,yminstring,xmaxstring,ymaxstring;
					xminstr << xmin;
					yminstr << ymin;
					xmaxstr << xmax;
					ymaxstr << ymax;
					xminstr >> xminstring;
					yminstr >> yminstring;
					xmaxstr >> xmaxstring;
					ymaxstr >> ymaxstring;
					range1 = "[" + xminstring + ":" + xmaxstring + "][" + yminstring + ":" + ymaxstring + "]";
				}
				if (nwords == 2) {
					if (plot_fits) Complain("file name for FITS file must be specified");
					if (set_title) plot_title = temp_title;
					if (mpi_id==0) {
						if (source_fit_mode==Cartesian_Source) cartesian_srcgrids[src_i]->plot_surface_brightness("src_pixel");
						else if (source_fit_mode==Delaunay_Source) {
							if (delaunay_srcgrids[src_i] != NULL) {
								delaunay_srcgrids[src_i]->plot_surface_brightness("src_pixel",set_npix,interpolate,plot_mag);
							} else Complain("Delaunay grid has not been created");

						}
						if (show_raytraced_pts) image_pixel_grids[zsrc_i]->plot_sourcepts("src_pixel",true); // this will overwrite the source point file to show all ray-traced points
					}
					if ((show_cc) and (zsrc_i >= 0)) create_grid(false,extended_src_zfactors[zsrc_i],extended_src_beta_factors[zsrc_i],zsrc_i);
					if ((nlens > 0) and (show_cc) and (plot_critical_curves("crit.dat")==true)) {
						if (source_fit_mode==Delaunay_Source) {
							if (include_srcpts) run_plotter_range("srcpixel_delaunay_srcpts_plural",range1);
							else run_plotter_range("srcpixel_delaunay",range1);
						}
						else run_plotter_range("srcpixel",range1);
					} else {
						if (source_fit_mode==Delaunay_Source) {
							if (include_srcpts) run_plotter_range("srcpixel_delaunay_srcpts_plural_nocc",range1);
							else run_plotter_range("srcpixel_delaunay_nocc",range1);
						}
						else run_plotter_range("srcpixel_nocc",range1);
					}
				} else if (nwords == 3) {
					if (plot_fits) {
						if (source_fit_mode==Cartesian_Source) cartesian_srcgrids[src_i]->output_fits_file(words[2]);
						else if (source_fit_mode==Delaunay_Source) {
							delaunay_srcgrids[src_i]->plot_surface_brightness(words[2],set_npix,interpolate,plot_mag,plot_fits);
						}
					} else {
						if (set_title) plot_title = temp_title;
						if ((terminal==TEXT) or (plot_fits)) {
							if (mpi_id==0) cartesian_srcgrids[src_i]->plot_surface_brightness(words[2]);
						} else {
							if (source_fit_mode==Cartesian_Source) cartesian_srcgrids[src_i]->plot_surface_brightness("src_pixel");
							else if (source_fit_mode==Delaunay_Source) {
								if (delaunay_srcgrids[src_i] != NULL) {
									delaunay_srcgrids[src_i]->plot_surface_brightness("src_pixel",set_npix,interpolate,plot_mag);
								} else Complain("Delaunay grid has not been created");
							}
							if ((nlens > 0) and (show_cc) and (plot_critical_curves("crit.dat")==true)) {
								if (source_fit_mode==Delaunay_Source) {
									if (include_srcpts) run_plotter_file("srcpixel_delaunay_srcpts_plural",words[2],range1);
									else run_plotter_file("srcpixel_delaunay",words[2],range1);
								}
								else run_plotter_file("srcpixel",words[2],range1);
							} else {
								if (source_fit_mode==Delaunay_Source) {
									if (include_srcpts) run_plotter_file("srcpixel_delaunay_srcpts_plural_nocc",words[2],range1);
									else run_plotter_file("srcpixel_delaunay_nocc",words[2],range1);
								}
								else run_plotter_file("srcpixel_nocc",words[2],range1);
							}
						}
					}
				} else Complain("invalid number of arguments to 'sbmap plotsrc'");
				reset_grid();
				if ((zoom_in) and (!delaunay)) cartesian_srcgrids[src_i]->srcgrid_size_scale = old_srcgrid_scale;
				if (omit_caustics) show_cc = old_caustics_setting;
				if (set_title) plot_title = "";
			}
			else if (words[1]=="invert")
			{
				if (!use_noise_map) Complain("Noise map required; either load from fits file or generate uniform noise map from bg_pixel_noise");
				//use_noise_map = true;
				if (source_fit_mode==Point_Source) Complain("cannot invert pixel image if source_mode is set to 'ptsource'");
				bool regopt = false; // false means it uses whatever the actual setting is for optimize_regparam
				bool verbal = true;
				bool chisqdif = false;
				bool old_regopt;
				bool temp_show_wtime = false;
				bool old_show_wtime = show_wtime;
				vector<string> args;
				if (extract_word_starts_with('-',2,nwords-1,args)==true)
				{
					for (int i=0; i < args.size(); i++) {
						if ((args[i]=="-wtime") or (args[i]=="-wt")) temp_show_wtime = true;
						else if (args[i]=="-regopt") regopt = true;
						else if ((args[i]=="-s") or (args[i]=="-silent")) verbal = false;
						else if ((args[i]=="-d") or (args[i]=="-difff")) chisqdif = true;
						else Complain("argument '" << args[i] << "' not recognized");
					}
				}
				if (regopt) {
					old_regopt = optimize_regparam;
					optimize_regparam = true;
				}
				if (nlens==0) {
					if ((n_sb==0) and (n_ptsrc==0)) {
						Complain("must specify lens/source model first");
					} else {
						bool all_unlensed = true;
						for (int i=0; i < n_sb; i++) {
							if (sb_list[i]->is_lensed) all_unlensed = false;
						}
						if (!all_unlensed) Complain("background source objects have been defined, but no lens models have been defined");
						all_unlensed = true;
						for (int i=0; i < n_ptsrc; i++) {
							if (ptsrc_redshifts[ptsrc_redshift_idx[i]] != lens_redshift) all_unlensed = false;
						}
						if (!all_unlensed) Complain("background source points have been defined, but no lens models have been defined");
					}
				}
				if ((temp_show_wtime) and (!show_wtime)) show_wtime = true;
				double chisq, chisq0;
				chisq = invert_surface_brightness_map_from_data(chisq0, verbal);
				if (regopt) optimize_regparam = old_regopt;
				if ((mpi_id==0) and (chisqdif)) {
					double diff = chisq - chisq_pix_last;
					cout << "chisq_dif = " << diff << endl;
				}
				chisq_pix_last = chisq;
				if ((temp_show_wtime) and (!old_show_wtime)) show_wtime = false;
			}
			else if (words[1]=="save_sbweights")
			{
				if (source_fit_mode==Point_Source) Complain("cannot invert pixel image if source_mode is set to 'ptsource'");
				if (use_saved_sbweights) Complain("cannot save sbweights if 'use_saved_sbweights' is also set to 'on'");
				if (nlens==0) {
					if ((n_sb==0) and (n_ptsrc==0)) {
						Complain("must specify lens/source model first");
					} else {
						bool all_unlensed = true;
						for (int i=0; i < n_sb; i++) {
							if (sb_list[i]->is_lensed) all_unlensed = false;
						}
						if (!all_unlensed) Complain("background source objects have been defined, but no lens models have been defined");
						all_unlensed = true;
						for (int i=0; i < n_ptsrc; i++) {
							if (ptsrc_redshifts[ptsrc_redshift_idx[i]] != lens_redshift) all_unlensed = false;
						}
						if (!all_unlensed) Complain("background source points have been defined, but no lens models have been defined");
					}
				}
				double chisq, chisq0;
				save_sbweights_during_inversion = true;
				chisq = invert_surface_brightness_map_from_data(chisq0, false);
				save_sbweights_during_inversion = false;
			}
			else if (words[1]=="plot_imgpixels")
			{
				int zsrc_i = 0;
				if (nwords==3) {
					if (!(ws[2] >> zsrc_i)) Complain("invalid imggrid index");
				}
				plot_image_pixel_grid(band_i,zsrc_i);
			}
			else Complain("command not recognized");
		}
		else if (words[0]=="lensinfo")
		{
			if (mpi_id==0) {
				if (nwords != 3) Complain("two arguments are required; must specify coordinates (x,y) to display lensing information");
				if (nlens==0) Complain("must specify lens model first");
				double x, y;
				if (!(ws[1] >> x)) Complain("invalid x-coordinate");
				if (!(ws[2] >> y)) Complain("invalid y-coordinate");
				print_lensing_info_at_point(x,y);
			}
		}
		else if (words[0]=="plotlensinfo")
		{
			int pert_resid = -1; // if set to a positive number, then the residuals are plotted after subtracting perturbed - smooth model
			if (nlens==0) Complain("must specify lens model first");
			string file_root;
			if (nwords == 1) {
 				// if the fit label hasn't been set, probably we're not doing a fit anyway, so pick a more generic name
				if (fit_output_filename == "fit") file_root = "lensmap";
				else file_root = fit_output_filename;
			} else if (nwords >= 2) {
				file_root = words[1];
				if (nwords==3) {
					if (!(ws[2] >> pert_resid)) Complain("invalid residual perturber number");
					if (pert_resid >= nlens) Complain("perturber lens number does not exist");
				}
			} else Complain("only one argument (file label) allowed for 'sbmap plotlensinfo'");
			plot_lensinfo_maps(file_root,n_image_pixels_x,n_image_pixels_y,pert_resid);
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
		else if ((words[0]=="integral_warnings") or (words[0]=="integration_warnings"))
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Warnings for integration convergence: " << display_switch(LensProfile::integration_warnings) << endl;
			} else if (nwords==2) {
				string setword;
				bool warn;
				if (!(ws[1] >> setword)) Complain("invalid argument to 'integration_warnings' command");
				else set_switch(warn,setword);
				set_integral_convergence_warnings(warn);
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
		else if (words[0]=="major_axis_along_y_src")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Orient major axis along y-direction for source profiles: " << display_switch(SB_Profile::orient_major_axis_north) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'major_axis_along_y_src' command; must specify 'on' or 'off'");
				bool orient_north;
				set_switch(orient_north,setword);
				toggle_major_axis_along_y_src(orient_north);
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
		else if (words[0]=="shear_angle_towards_perturber")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Shear angle points towards (hypothetical) perturber: " << display_switch(Shear::angle_points_towards_perturber) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'shear_angle_towards_perturber' command; must specify 'on' or 'off'");
				bool towards_perturber;
				set_switch(towards_perturber,setword);
				if (Shear::angle_points_towards_perturber != towards_perturber) {
					Shear::angle_points_towards_perturber = towards_perturber;
					if (Shear::use_shear_component_params==false) {
						// awkward, but otherwise it doesn't change the actual theta_shear parameter
						Shear::use_shear_component_params = true;
						reassign_lensparam_pointers_and_names(false);
						Shear::use_shear_component_params = false;
						reassign_lensparam_pointers_and_names(false);
					}
				}
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
		else if (words[0]=="sb_ellipticity_components")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use ellipticity components for SB profiles instead of (q,theta): " << display_switch(SB_Profile::use_sb_ellipticity_components) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'sb_ellipticity_components' command; must specify 'on' or 'off'");
				bool use_comps;
				set_switch(use_comps,setword);
				SB_Profile::use_sb_ellipticity_components = use_comps;
				reassign_sb_param_pointers_and_names();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="use_scaled_fmode_amps")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use 1/m scaled Fourier mode amplitudes: " << display_switch(SB_Profile::use_fmode_scaled_amplitudes) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'use_scaled_fmode_amps' command; must specify 'on' or 'off'");
				bool use_scaled_amps;
				set_switch(use_scaled_amps,setword);
				SB_Profile::use_fmode_scaled_amplitudes = use_scaled_amps;
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="fourier_sbmode")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use Fourier perturbation to surface brightness: " << display_switch(SB_Profile::fourier_sb_perturbation) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'fourier_sbmode' command; must specify 'on' or 'off'");
				bool use_fsbmode;
				set_switch(use_fsbmode,setword);
				SB_Profile::fourier_sb_perturbation = use_fsbmode;
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="fourier_ecc_anomaly")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use eccentric anomaly for Fourier perturbations: " << display_switch(SB_Profile::fourier_use_eccentric_anomaly) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'fourier_ecc_anomaly' command; must specify 'on' or 'off'");
				bool use_fecc;
				set_switch(use_fecc,setword);
				SB_Profile::fourier_use_eccentric_anomaly = use_fecc;
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="zoom_splitfac")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "source zoom subgridding split factor = " << SB_Profile::zoom_split_factor << endl;
			} else if (nwords==2) {
				double fac;
				if (!(ws[1] >> fac)) Complain("invalid argument to 'zoom_splitfac' command; must be real number");
				SB_Profile::zoom_split_factor = fac;
			} else Complain("only one argument allowed for 'zoom_splitfac' (splitting factor)");
		}
		else if (words[0]=="zoom_scale")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "source zoom subgridding scale = " << SB_Profile::zoom_scale << endl;
			} else if (nwords==2) {
				double fac;
				if (!(ws[1] >> fac)) Complain("invalid argument to 'zoom_scale' command; must be real number");
				SB_Profile::zoom_scale = fac;
			} else Complain("only one argument allowed for 'zoom_scale' (splitting factor)");
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
				SB_Profile::default_ellipticity_mode = elmode;
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
				reset_grid();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="show_cc")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Show critical curves: " << display_switch(show_cc) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'show_cc' command; must specify 'on' or 'off'");
				set_switch(show_cc,setword);
				if (show_cc==true) auto_store_cc_points = true; // just in case it was turned off previously
				if ((show_cc==false) and (!autogrid_before_grid_creation)) auto_store_cc_points = false;
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
				set_plot_title(1,plot_title);
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
		else if (words[0]=="exclude_hist2d_params")
		{
			if (nwords==1) {
				if (!param_settings->hist2d_params_defined()) Complain("No excluded hist2d parameters have been defined");
				else if (mpi_id==0) cout << "Excluded parameters from 2d plots: " << param_settings->print_excluded_hist2d_params() << endl;
			}
			else if ((nwords==2) and (words[1]=="reset")) param_settings->reset_hist2d_params();
			else {
				param_settings->reset_hist2d_params();
				for (int i=1; i < nwords; i++) {
					if (!param_settings->exclude_hist2d_param(words[i])) Complain("Fit parameter '" << words[i] << "' does not exist");
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
				if (words[1]=="auto") cbmin = -1e30; // the lens plotting script recognizes this as "auto"
				else if (!(ws[1] >> cbmin)) Complain("invalid cbmin setting");
				colorbar_min = cbmin;
			} else if (nwords==1) {
				if (mpi_id==0) {
					cout << "colorbar min surface brightness = ";
					if (colorbar_min==-1e30) cout << "auto" << endl;
					else cout << colorbar_min << endl;
				}
			} else Complain("must specify either zero or one argument");
		}
		else if (words[0]=="cbmax")
		{
			double cbmax;
			if (nwords == 2) {
				if (words[1]=="auto") cbmax = 1e30; // the lens plotting script recognizes this as "auto"
				else if (!(ws[1] >> cbmax)) Complain("invalid cbmax setting");
				colorbar_max = cbmax;
			} else if (nwords==1) {
				if (mpi_id==0) {
					cout << "colorbar max surface brightness = ";
					if (colorbar_max==1e30) cout << "auto" << endl;
					else cout << colorbar_max << endl;
				}
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
				if (mpi_id==0) cout << "Use image plane chi-square function: " << display_switch(imgplane_chisq) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'imgplane_chisq' command; must specify 'on' or 'off'");
				set_switch(imgplane_chisq,setword);
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
				bool vary_source_coords = (use_analytic_bestfit_src) ? false : true;
				// Automatically turn source coordinate parameters on/off accordingly
				for (int i=0; i < n_ptsrc; i++) {
					update_ptsrc_varyflag(i,"xsrc",vary_source_coords);
					update_ptsrc_varyflag(i,"ysrc",vary_source_coords);
				}
				for (int i=0; i < n_ptsrc; i++) {
					update_ptsrc_active_parameters(i);
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="include_ptsrc_shift")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Allow shift parameters to correct the estimated source positions from analytic_bestfit_src: " << display_switch(include_ptsrc_shift) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'include_ptsrc_shift' command; must specify 'on' or 'off'");
				bool old_setting = include_ptsrc_shift;
				set_switch(include_ptsrc_shift,setword);
				if (include_ptsrc_shift != old_setting) {
					for (int i=0; i < n_ptsrc; i++) {
						update_ptsrc_active_parameters(i);
					}
					if ((include_ptsrc_shift==true) and (old_setting==false)) {
						// Automatically turn source coordinate parameters on 
						if (mpi_id==0) cout << "NOTE: Turning xshift, yshift vary flags on for all point sources" << endl;
						for (int i=0; i < n_ptsrc; i++) {
							update_ptsrc_varyflag(i,"xshift",true);
							update_ptsrc_varyflag(i,"yshift",true);
						}
					}
					update_parameter_list();
				}
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
		else if (words[0]=="analytic_srcflux")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Calculate source flux analytically (instead of using 'srcflux'): " << display_switch(analytic_source_flux) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'analytic_srcflux' command; must specify 'on' or 'off'");
				set_switch(analytic_source_flux,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="invert_imgflux")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include image flux of point images in inversion: " << display_switch(include_imgfluxes_in_inversion) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'invert_imgflux' command; must specify 'on' or 'off'");
				set_switch(include_imgfluxes_in_inversion,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="invert_srcflux")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include source point flux in inversion: " << display_switch(include_srcflux_in_inversion) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'invert_srcflux' command; must specify 'on' or 'off'");
				set_switch(include_srcflux_in_inversion,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="potential_perturbations")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include pixellated perturbations to the lensing potential: " << display_switch(include_potential_perturbations) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'potential_perturbations' command; must specify 'on' or 'off'");
				set_switch(include_potential_perturbations,setword);
				if ((n_pixellated_lens > 0) and (lensgrids)) {
					for (int i=0; i < n_pixellated_lens; i++) {
						if ((first_order_sb_correction) or (!include_potential_perturbations)) lensgrids[i]->include_in_lensing_calculations = false;
						else lensgrids[i]->include_in_lensing_calculations = true;
					}
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="first_order_sb_correction")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include first-order correction to surface brightness from pixellated perturbations to lensing potential: " << display_switch(first_order_sb_correction) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'first_order_sb_correction' command; must specify 'on' or 'off'");
				set_switch(first_order_sb_correction,setword);
				if ((n_pixellated_lens > 0) and (lensgrids)) {
					for (int i=0; i < n_pixellated_lens; i++) {
						if ((first_order_sb_correction) or (!include_potential_perturbations)) lensgrids[i]->include_in_lensing_calculations = false;
						else lensgrids[i]->include_in_lensing_calculations = true;
					}
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="adopt_final_sbgrad")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Adopt final source SB gradient for first-order SB corrections: " << display_switch(adopt_final_sbgrad) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'adopt_final_sbgrad' command; must specify 'on' or 'off'");
				set_switch(adopt_final_sbgrad,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="potential_corr_it")
		{
			int iter;
			if (nwords == 2) {
				if (!(ws[1] >> iter)) Complain("invalid potential_corr_it setting");
				potential_correction_iterations = iter;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "number of iterations for potential corrections = " << potential_correction_iterations << endl;
			} else Complain("must specify either zero or one argument (number of iterations)");
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
		else if (words[0]=="ignore_fg_in_chisq")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Ignore foreground in chi-square: " << display_switch(ignore_foreground_in_chisq) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'ignore_fg_in_chisq' command; must specify 'on' or 'off'");
				set_switch(ignore_foreground_in_chisq,setword);
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
				reset_grid();
				user_changed_zsource = true; // keeps track of whether redshift has been manually changed; if so, then qlens won't automatically change it to redshift from data
			} else if (nwords==1) {
				if (mpi_id==0) cout << "source redshift = " << source_redshift << endl;
			} else Complain("must specify either zero or one argument (redshift of source object)");
		}
		else if (words[0]=="zsrc_ref")
		{
			double zsrc_ref;
			if (nwords == 2) {
				if (nlens > 0) Complain("zsrc_ref cannot be changed if any lenses have already been created");
				if (!(ws[1] >> zsrc_ref)) Complain("invalid zsrc_ref setting");
				set_reference_source_redshift(zsrc_ref);
			} else if (nwords==1) {
				if (mpi_id==0) cout << "reference source redshift = " << reference_source_redshift << endl;
			} else Complain("must specify either zero or one argument (reference source redshift)");
		}
		else if (words[0]=="hubble")
		{
			double hub;
			bool vary_hub;
			if (nwords == 2) {
				if (!(ws[1] >> hub)) Complain("invalid hubble setting");
				cosmo.update_specific_parameter("hubble",hub);
				update_zfactors_and_betafactors();
				for (int i=0; i < nlens; i++) {
					if (cosmo.get_n_vary_params()==0) lens_list[i]->update_cosmology_meta_parameters(true); // if the cosmology has changed, update cosmology info and any parameters that depend on them (this forces the issue even if cosmo params aren't being varied as fit parameters; otherwise will be done on next line)
					lens_list[i]->update_meta_parameters(); // if the cosmology has changed, update cosmology info and any parameters that depend on them 
				}
				cosmo.get_specific_varyflag("hubble",vary_hub);
				//hubbleatter = hub;
				//cosmo.set_cosmology(hubbleatter,0.04,hubble,2.215);
				if ((vary_hub) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
					if (mpi_id==0) cout << "Limits for hubble parameter:\n";
					if (read_command(false)==false) return;
					double hmin,hmax;
					if (nwords != 2) Complain("Must specify two arguments for hubble parameter limits: hmin, hmax");
					if (!(ws[0] >> hmin)) Complain("Invalid lower limit for hubble parameter");
					if (!(ws[1] >> hmax)) Complain("Invalid upper limit for hubble parameter");
					if (hmin > hmax) Complain("lower limit cannot be greater than upper limit");
					cosmo.set_limits_specific_parameter("hubble",hmin,hmax);
					//hubbleatter_lower_limit = hmin;
					//hubbleatter_upper_limit = hmax;
				}
			} else if (nwords==1) {
				cosmo.get_specific_parameter("hubble",hub);
				if (mpi_id==0) cout << "Matter density (hubble) = " << hub << endl;
			} else Complain("must specify either zero or one argument (hubble value)");
		}
		else if (words[0]=="vary_hubble")
		{
			bool vary_hub;
			if (nwords==1) {
				cosmo.get_specific_varyflag("hubble",vary_hub);
				if (mpi_id==0) cout << "Vary hubble parameter: " << display_switch(vary_hub) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_hubble' command; must specify 'on' or 'off'");
				set_switch(vary_hub,setword);
				update_cosmo_varyflag("hubble",vary_hub);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		//else if (words[0]=="hubble")
		//{
			//double h0param;
			//if (nwords == 2) {
				//if (!(ws[1] >> h0param)) Complain("invalid hubble setting");
				//hubble = h0param;
				//cosmo.set_cosmology(omega_matter,0.04,hubble,2.215);
				//if ((vary_hubble_parameter) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
					//if (mpi_id==0) cout << "Limits for Hubble parameter:\n";
					//if (read_command(false)==false) return;
					//double hmin,hmax;
					//if (nwords != 2) Complain("Must specify two arguments for Hubble parameter limits: hmin, hmax");
					//if (!(ws[0] >> hmin)) Complain("Invalid lower limit for Hubble parameter");
					//if (!(ws[1] >> hmax)) Complain("Invalid upper limit for Hubble parameter");
					//if (hmin > hmax) Complain("lower limit cannot be greater than upper limit");
					//hubble_lower_limit = hmin;
					//hubble_upper_limit = hmax;
				//}
			//} else if (nwords==1) {
				//if (mpi_id==0) cout << "Hubble parameter = " << hubble << endl;
			//} else Complain("must specify either zero or one argument (Hubble parameter)");
		//}
		//else if (words[0]=="vary_hubble")
		//{
			//if (nwords==1) {
				//if (mpi_id==0) cout << "Vary Hubble parameter: " << display_switch(vary_hubble_parameter) << endl;
			//} else if (nwords==2) {
				//if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_hubble' command; must specify 'on' or 'off'");
				//set_switch(vary_hubble_parameter,setword);
				//update_parameter_list();
			//} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		//}
		else if (words[0]=="omega_m")
		{
			double om;
			bool vary_om;
			if (nwords == 2) {
				if (!(ws[1] >> om)) Complain("invalid omega_m setting");
				cosmo.update_specific_parameter("omega_m",om);
				update_zfactors_and_betafactors();
				for (int i=0; i < nlens; i++) {
					if (cosmo.get_n_vary_params()==0) lens_list[i]->update_cosmology_meta_parameters(true); // if the cosmology has changed, update cosmology info and any parameters that depend on them (this forces the issue even if cosmo params aren't being varied as fit parameters; otherwise will be done on next line)
					lens_list[i]->update_meta_parameters(); // if the cosmology has changed, update cosmology info and any parameters that depend on them 
				}
				cosmo.get_specific_varyflag("omega_m",vary_om);
				//omega_matter = om;
				//cosmo.set_cosmology(omega_matter,0.04,hubble,2.215);
				if ((vary_om) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
					if (mpi_id==0) cout << "Limits for omega_m parameter:\n";
					if (read_command(false)==false) return;
					double omin,omax;
					if (nwords != 2) Complain("Must specify two arguments for omega_m parameter limits: omin, omax");
					if (!(ws[0] >> omin)) Complain("Invalid lower limit for omega_m parameter");
					if (!(ws[1] >> omax)) Complain("Invalid upper limit for omega_m parameter");
					if (omin > omax) Complain("lower limit cannot be greater than upper limit");
					cosmo.set_limits_specific_parameter("omega_m",omin,omax);
					//omega_matter_lower_limit = omin;
					//omega_matter_upper_limit = omax;
				}
			} else if (nwords==1) {
				cosmo.get_specific_parameter("omega_m",om);
				if (mpi_id==0) cout << "Matter density (omega_m) = " << om << endl;
			} else Complain("must specify either zero or one argument (omega_m value)");
		}
		else if (words[0]=="vary_omega_m")
		{
			bool vary_om;
			if (nwords==1) {
				cosmo.get_specific_varyflag("omega_m",vary_om);
				if (mpi_id==0) cout << "Vary omega_m parameter: " << display_switch(vary_om) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_omega_m' command; must specify 'on' or 'off'");
				set_switch(vary_om,setword);
				update_cosmo_varyflag("omega_m",vary_om);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="syserr_pos")
		{
			double syserrparam;
			bool vary_syserr_pos;
			if (nwords == 2) {
				if (!(ws[1] >> syserrparam)) Complain("invalid syserr_pos setting");
				update_specific_parameter("syserr_pos",syserrparam);
				get_specific_varyflag("syserr_pos",vary_syserr_pos);
				if ((vary_syserr_pos) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
					if (mpi_id==0) cout << "Limits for systematic error parameter:\n";
					if (read_command(false)==false) return;
					double sigmin,sigmax;
					if (nwords != 2) Complain("Must specify two arguments for systematic error parameter limits: sigmin, sigmax");
					if (!(ws[0] >> sigmin)) Complain("Invalid lower limit for systematic error parameter");
					if (!(ws[1] >> sigmax)) Complain("Invalid upper limit for systematic error parameter");
					if (sigmin > sigmax) Complain("lower limit cannot be greater than upper limit");
					set_limits_specific_parameter("syserr_pos",sigmin,sigmax);
				}
			} else if (nwords==1) {
				if (mpi_id==0) cout << "systematic error parameter = " << syserr_pos << endl;
			} else Complain("must specify either zero or one argument (systematic error parameter)");
		}
		else if (words[0]=="vary_syserr_pos")
		{
			bool vary_syserr_pos = false;
			if (nwords==1) {
				get_specific_varyflag("syserr_pos",vary_syserr_pos);
				if (mpi_id==0) cout << "Vary systematic error parameter: " << display_switch(vary_syserr_pos) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_syserr_pos' command; must specify 'on' or 'off'");
				set_switch(vary_syserr_pos,setword);
				update_misc_varyflag("syserr_pos",vary_syserr_pos);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="wl_shearfac")
		{
			double wlshearfac;
			bool vary_wlshearfac;
			if (nwords == 2) {
				if (!(ws[1] >> wlshearfac)) Complain("invalid wl_shearfac setting");
				update_specific_parameter("wl_shearfac",wlshearfac);
				get_specific_varyflag("wl_shearfac",vary_wlshearfac);
				if ((vary_wlshearfac) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
					if (mpi_id==0) cout << "Limits for weak lensing scale factor parameter:\n";
					if (read_command(false)==false) return;
					double facmin,facmax;
					if (nwords != 2) Complain("Must specify two arguments for weak lensing scale factor parameter limits: facmin, facmax");
					if (!(ws[0] >> facmin)) Complain("Invalid lower limit for weak lensing scale factor parameter");
					if (!(ws[1] >> facmax)) Complain("Invalid upper limit for weak lensing scale factor parameter");
					if (facmin > facmax) Complain("lower limit cannot be greater than upper limit");
					set_limits_specific_parameter("wl_shearfac",facmin,facmax);
				}
			} else if (nwords==1) {
				if (mpi_id==0) cout << "weak lensing scale factor parameter = " << wl_shear_factor << endl;
			} else Complain("must specify either zero or one argument (weak lensing scale factor parameter)");
		}
		else if (words[0]=="vary_wl_shearfac")
		{
			bool vary_wlshearfac = false;
			if (nwords==1) {
				get_specific_varyflag("syserr_pos",vary_wlshearfac);
				if (mpi_id==0) cout << "Vary weak lensing scale factor parameter: " << display_switch(vary_wlshearfac) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_wl_shearfac' command; must specify 'on' or 'off'");
				set_switch(vary_wlshearfac,setword);
				update_misc_varyflag("wl_shearfac",vary_wlshearfac);
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
		else if (words[0]=="skip_newton")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Skip Newton's method during image searching: " << display_switch(cc_neighbor_splittings) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'skip_newton' command; must specify 'on' or 'off'");
				set_switch(skip_newtons_method,setword);
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
			} else Complain("must specify either zero or one argument (required chi-square precision)");
		}
		/*
		else if (words[0]=="chisqtol_lumreg")
		{
			double tol;
			if (nwords == 2) {
				if (!(ws[1] >> tol)) Complain("invalid chisqtol_lumreg setting");
				chisqtol_lumreg=tol;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "chi-square tolerance for convergence of luminosity regularization = " << chisqtol_lumreg << endl;
			} else Complain("must specify either zero or one argument for chisqtol_lumreg");
		}
		*/
		else if (words[0]=="lumreg_max_it")
		{
			int maxit;
			if (nwords == 2) {
				if (!(ws[1] >> maxit)) Complain("invalid lumreg_max_it setting");
				lumreg_max_it = maxit;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "max number of iterations for luminosity regularization = " << lumreg_max_it << endl;
			} else Complain("must specify either zero or one argument for lumreg_max_it");
		}
		/*
		else if (words[0]=="lumreg_max_it_final")
		{
			int maxit;
			if (nwords == 2) {
				if (!(ws[1] >> maxit)) Complain("invalid lumreg_max_it_final setting");
				lumreg_max_it_final=maxit;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "max number of final iterations for luminosity regularization = " << lumreg_max_it_final << endl;
			} else Complain("must specify either zero or one argument for lumreg_max_it_final");
		}
		*/
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
		else if (words[0]=="sim_err_shear")
		{
			double simerr;
			if (nwords == 2) {
				if (!(ws[1] >> simerr)) Complain("invalid position error");
				sim_err_shear = simerr;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "simulated error in weak lensing shear = " << sim_err_shear << endl;
			} else Complain("must specify either zero or one argument (error in simulated weak lensing shear)");
		}
		else if (words[0]=="subhalo_rmax")
		{
			if (nwords==2) {
				int lens_number;
				if (!(ws[1] >> lens_number)) Complain("invalid lens number");
				if (lens_number >= nlens) Complain("specified lens number for subhalo does not exist");
				if (lens_number == 0) Complain("perturber cannot be the primary lens (lens 0)");
				double rmax,menc,avgsig,rmax_z,menc_z;
				if (nlens==1) Complain("perturber lens has not been defined");
				if (!calculate_critical_curve_perturbation_radius_numerical(lens_number,true,rmax,avgsig,menc,rmax_z,menc_z)) Complain("could not calculate critical curve perturbation radius");
			} else Complain("one argument required for 'subhalo_rmax' (lens number for subhalo)");
		}
		else if (words[0]=="subhalo_rmax2")
		{
			if (nwords==2) {
				int lens_number;
				if (!(ws[1] >> lens_number)) Complain("invalid lens number");
				if (lens_number >= nlens) Complain("specified lens number for subhalo does not exist");
				if (lens_number == 0) Complain("perturber cannot be the primary lens (lens 0)");
				double rmax,menc,avgsig,rmax_z,menc_z;
				if (nlens==1) Complain("perturber lens has not been defined");
				if (!calculate_critical_curve_perturbation_radius_numerical(lens_number,true,rmax,avgsig,menc,rmax_z,menc_z,true)) Complain("could not calculate critical curve perturbation radius");
			} else Complain("one argument required for 'subhalo_rmax' (lens number for subhalo)");
		}
		else if (words[0]=="print_zfactors")
		{
			print_zfactors_and_beta_matrices();
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
			int band_i=0;
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("band=")) != string::npos) {
					string bnumstring = words[i].substr(pos+5);
					stringstream bnumstr;
					bnumstr << bnumstring;
					if (!(bnumstr >> band_i)) Complain("incorrect format for band number");
					if (band_i < 0) Complain("band number cannot be negative");
					remove_word(i);
					break;
				}
			}	
			if (nwords == 3) {
				if (!(ws[1] >> psfx)) Complain("invalid PSF x-width");
				if (!(ws[2] >> psfy)) Complain("invalid PSF y-width");
				psf_list[band_i]->psf_width_x = psfx;
				psf_list[band_i]->psf_width_y = psfy;
			} else if (nwords == 2) {
				if (!(ws[1] >> psfx)) Complain("invalid PSF width");
				psf_list[band_i]->psf_width_x = psfx;
				psf_list[band_i]->psf_width_y = psfx;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Point spread function (PSF) width = (" << psf_list[band_i]->psf_width_x << "," << psf_list[band_i]->psf_width_y << ")\n";
			} else Complain("can only specify up to two arguments for PSF width (x-width,y-width)");
		}
		else if (words[0]=="psf_threshold")
		{
			double threshold;
			if (nwords == 2) {
				if (!(ws[1] >> threshold)) Complain("invalid PSF threshold");
				if (threshold >= 1) Complain("psf threshold must be less than 1");
				psf_threshold = threshold;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Point spread function (PSF) input threshold = " << psf_threshold << endl;
			} else Complain("can only specify up to one argument for PSF input threshold");
		}
		else if (words[0]=="psf_ptsrc_threshold")
		{
			double threshold;
			if (nwords == 2) {
				if (!(ws[1] >> threshold)) Complain("invalid PSF threshold");
				if (threshold >= 1) Complain("point source psf threshold must be less than 1");
				psf_ptsrc_threshold = threshold;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Point source PSF input threshold = " << psf_ptsrc_threshold << endl;
			} else Complain("can only specify up to one argument for point source PSF input threshold");
		}
		else if (words[0]=="ptimg_nsplit")
		{
			// NOTE: currently only pixels in the primary mask are split; pixels in extended mask are NOT split (see setup_ray_tracing_arrays() in pixelgrid.cpp)
			if (nwords == 2) {
				int nsp;
				if (!(ws[1] >> nsp)) Complain("invalid number of point image PSF pixel splittings");
				ptimg_nsplit = nsp;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "default number of point image PSF pixel splittings = " << ptimg_nsplit << endl;
			} else Complain("must specify either zero or one argument (default number of point source PSF pixel splittings)");
		}
		else if (words[0]=="fft_convolution")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use FFT to calculate PSF convolutions: " << display_switch(fft_convolution) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'fft_convolution' command; must specify 'on' or 'off'");
				bool old_setting = fft_convolution;
				set_switch(fft_convolution,setword);
				if ((!fft_convolution) and (old_setting==true)) cleanup_FFT_convolution_arrays();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="pixel_fraction")
		{
			double frac, frac_ll, frac_ul;
 			if (nwords == 4) {
 				if (!(ws[1] >> frac_ll)) Complain("invalid source pixel fraction lower limit");
 				if (!(ws[2] >> frac)) Complain("invalid source pixel fraction value");
 				if (!(ws[3] >> frac_ul)) Complain("invalid source pixel fraction upper limit");
 				if ((frac < frac_ll) or (frac > frac_ul)) Complain("initial source pixel fraction should lie within specified prior limits");
				if (n_pixellated_src==0) Complain("no pixellated sources have been created");
				for (int i=0; i < n_pixellated_src; i++) {
					cartesian_srcgrids[i]->update_specific_parameter("pixfrac",frac);
					cartesian_srcgrids[i]->set_limits_specific_parameter("pixfrac",frac_ll,frac_ul);
				}
 			} else if (nwords == 2) {
				if (!(ws[1] >> frac)) Complain("invalid firstlevel source pixel fraction");
				if (n_pixellated_src==0) Complain("no pixellated sources have been created");
				for (int i=0; i < n_pixellated_src; i++) {
					cartesian_srcgrids[i]->update_specific_parameter("pixfrac",frac);
				}
			} else if (nwords==1) {
				if (n_pixellated_src==0) cout << "No pixellated sources have been created yet" << endl;
				for (int i=0; i < n_pixellated_src; i++) {
					if (!cartesian_srcgrids[i]->get_specific_parameter("pixfrac",frac)) Complain("pixfrac is not an active parameter for pixellated sources (auto_srcgrid_npixels is off)");
					cout << "pixsrc " << i << ": pixfrac = " << frac << endl;
				}
			} else Complain("must specify one argument (firstlevel source pixel fraction)");
		}
		else if (words[0]=="srcgrid_scale")
		{
			double scale, scale_ll, scale_ul;
 			if (nwords == 4) {
 				if (!(ws[1] >> scale_ll)) Complain("invalid source srcgrid_scale lower limit");
 				if (!(ws[2] >> scale)) Complain("invalid source srcgrid_scale value");
 				if (!(ws[3] >> scale_ul)) Complain("invalid source srcgrid_scale upper limit");
 				if ((scale < scale_ll) or (scale > scale_ul)) Complain("initial source srcgrid_scale should lie within specified prior limits");
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					cartesian_srcgrids[i]->update_specific_parameter("srcgrid_scale",scale);
					cartesian_srcgrids[i]->set_limits_specific_parameter("srcgrid_scale",scale_ll,scale_ul);
				}
 			} else if (nwords == 2) {
				if (!(ws[1] >> scale)) Complain("invalid firstlevel source srcgrid_scale");
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					cartesian_srcgrids[i]->update_specific_parameter("srcgrid_scale",scale);
				}
			} else if (nwords==1) {
				if (mpi_id==0) {
					if (n_pixellated_src==0) cout << "No pixellated sources have been created yet" << endl;
					for (int i=0; i < n_pixellated_src; i++) {
						if (!cartesian_srcgrids[i]->get_specific_parameter("srcgrid_scale",scale)) Complain("srcgrid_scale is not an active parameter for pixellated sources (auto_srcgrid is off)");
						cout << "pixsrc " << i << ": scale = " << scale << endl;
					}
				}

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
					if (default_data_pixel_size < 0) cout << "Pixel size for loaded FITS images (data_pixel_size): not specified\n";
					else cout << "Pixel size for loaded FITS images (data_pixel_size): " << default_data_pixel_size << endl;
				}
			} else if (nwords==2) {
				double ps;
				if (!(ws[1] >> ps)) Complain("invalid argument to 'data_pixel_size' command");
				default_data_pixel_size = ps;
			} else Complain("invalid number of arguments to 'data_pixel_size'");
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
		else if (words[0]=="multinest_target_eff")
		{
			double target_eff;
			if (nwords == 2) {
				if (!(ws[1] >> target_eff)) Complain("invalid target efficiency for nested sampling");
				multinest_target_efficiency = target_eff;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "target efficiency for nested sampling = " << multinest_target_efficiency << endl;
			} else Complain("must specify either zero or one argument (target efficiency for multinest)");
		}
		else if (words[0]=="multinest_constant_eff_mode")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Constant efficiency mode: " << display_switch(multinest_constant_eff_mode) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'multinest_constant_eff_mode' command; must specify 'on' or 'off'");
				set_switch(multinest_constant_eff_mode,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="multinest_mode_separation")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "mode separation: " << display_switch(multinest_mode_separation) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'multinest_mode_separation' command; must specify 'on' or 'off'");
				set_switch(multinest_mode_separation,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
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
		//else if (words[0]=="psf_mpi")
		//{
			//if (nwords==1) {
				//if (mpi_id==0) cout << "Use parallel PSF convolution with MPI: " << display_switch(psf_convolution_mpi) << endl;
			//} else if (nwords==2) {
				//if (!(ws[1] >> setword)) Complain("invalid argument to 'psf_mpi' command; must specify 'on' or 'off'");
				//set_switch(psf_convolution_mpi,setword);
			//} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		//}
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
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					cartesian_srcgrids[i]->update_specific_parameter("mag_threshold",threshold);
					cartesian_srcgrids[i]->set_limits_specific_parameter("mag_threshold",threshold_ll,threshold_ul);
				}
			} else if (nwords == 2) {
				if (!(ws[1] >> threshold)) Complain("invalid magnification threshold for source pixel splitting");
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					cartesian_srcgrids[i]->update_specific_parameter("mag_threshold",threshold);
				}
			} else if (nwords==1) {
				if (mpi_id==0) {
					if (n_pixellated_src==0) cout << "No pixellated sources have been created yet" << endl;
					for (int i=0; i < n_pixellated_src; i++) {
						if (!cartesian_srcgrids[i]->get_specific_parameter("mag_threshold",threshold)) Complain("mag_threshold is not an active parameter for pixellated sources (auto_srcgrid is off)");
						cout << "pixsrc " << i << ": mag_threshold = " << threshold << endl;
					}
				}
			} else Complain("must specify one argument (magnification threshold for source pixel splitting)");
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
			double regparam, regparam_upper, regparam_lower;
			if (nwords == 4) {
				if (!(ws[1] >> regparam_lower)) Complain("invalid regularization parameter lower limit");
				if (!(ws[2] >> regparam)) Complain("invalid regularization parameter value");
				if (!(ws[3] >> regparam_upper)) Complain("invalid regularization parameter upper limit");
				if ((regparam < regparam_lower) or (regparam > regparam_upper)) Complain("initial regularization parameter should lie within specified prior limits");
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					srcgrids[i]->update_specific_parameter("regparam",regparam);
					srcgrids[i]->set_limits_specific_parameter("regparam",regparam_lower,regparam_upper);
				}
			} else if (nwords == 2) {
				if (!(ws[1] >> regparam)) Complain("invalid regularization parameter value");
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					srcgrids[i]->update_specific_parameter("regparam",regparam);
				}
			} else if (nwords==1) {
				if (mpi_id==0) {
					if (n_pixellated_src==0) cout << "No pixellated sources have been created yet" << endl;
					for (int i=0; i < n_pixellated_src; i++) {
						if (!srcgrids[i]->get_specific_parameter("regparam",regparam)) Complain("regparam is not an active parameter for pixellated sources (regularization turned off)");
						cout << "pixsrc " << i << ": regparam = " << regparam << endl;
					}
				}
			} else Complain("must specify either zero or one argument (regularization parameter value)");
		}
		else if (words[0]=="vary_regparam")
		{
			bool vary_regparam = false;
			if (nwords==1) {
				for (int i=0; i < n_pixellated_src; i++) {
					srcgrids[i]->get_specific_varyflag("regparam",vary_regparam);
					if (vary_regparam) break;
				}
				if (mpi_id==0) cout << "Vary regularization parameter: " << display_switch(vary_regparam) << " (for at least one pixsrc)" << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_regparam' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before regparam can be varied (see 'fit regularization')");
				if ((setword=="on") and (optimize_regparam)) Complain("regparam cannot be varied freely if 'optimize_regparam' is set to 'on'");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("regparam can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				set_switch(vary_regparam,setword);
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					update_pixellated_src_varyflag(i,"regparam",vary_regparam);
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lum_weighted_regularization")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use luminosity-weighted regularization: " << display_switch(use_lum_weighted_regularization) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'lum_weighted_regularization' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before regparam can be varied (see 'fit regularization')");
				if ((setword=="on") and (!optimize_regparam) and (!get_lumreg_from_sbweights)) Complain("lum_weighted_regularization requires 'optimize_regparam' or 'lumreg_from_sbweights' to be set to 'on'");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("regparam can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				if ((setword=="on") and (use_distance_weighted_regularization)) {
					use_distance_weighted_regularization = false;
					if (mpi_id==0) cout << "NOTE: setting 'dist_weighted_regularization' to 'off'" << endl;
				}
				set_switch(use_lum_weighted_regularization,setword);
				for (int i=0; i < n_pixellated_src; i++) {
					update_pixsrc_active_parameters(i);
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lumweight_func")
		{
			int lfunc;
			if (nwords == 2) {
				if (!(ws[1] >> lfunc)) Complain("invalid lumweight_func value");
				if ((lfunc < 0) or (lfunc > 2)) Complain("lumweight_func must be either 0, 1, or 2");
				lum_weight_function = lfunc;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "lumweight_func = " << lum_weight_function << endl;
			} else Complain("must specify either zero or one argument (lumweight_func value)");
		}
		else if (words[0]=="dist_weighted_regularization")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use distance-weighted regularization: " << display_switch(use_distance_weighted_regularization) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'dist_weighted_regularization' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before luminosity weighting can be used (see 'fit regularization')");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("regparam can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				if ((setword=="on") and (use_lum_weighted_regularization)) {
					use_lum_weighted_regularization = false;
					if (mpi_id==0) cout << "NOTE: setting 'lum_weighted_regularization' to 'off'" << endl;
				}
				set_switch(use_distance_weighted_regularization,setword);
				for (int i=0; i < n_pixellated_src; i++) {
					update_pixsrc_active_parameters(i);
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="mag_weighted_regularization")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use magnification-weighted regularization: " << display_switch(use_mag_weighted_regularization) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'mag_weighted_regularization' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before regparam can be varied (see 'fit regularization')");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("regparam can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				set_switch(use_mag_weighted_regularization,setword);
				for (int i=0; i < n_pixellated_src; i++) {
					update_pixsrc_active_parameters(i);
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="dist_weighted_srcpixel_clustering")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use distance-weighted source pixel clustering: " << display_switch(use_dist_weighted_srcpixel_clustering) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'dist_weighted_srcpixel_clustering' command; must specify 'on' or 'off'");
				if ((setword=="on") and (source_fit_mode != Delaunay_Source)) Complain("distance-weighted srcpixel clustering can only be used if source mode is set to 'delaunay' (see 'fit source_mode')");
				if ((setword=="on") and (use_lum_weighted_srcpixel_clustering)) Complain("dist_weighted_srcpixel_clustering and lum_weighted_srcpixel_clustering cannot both be on"); 
				if ((setword=="on") and ((split_imgpixels==false) or (default_imgpixel_nsplit==1))) Complain("split_imgpixels must be turned on (and imgpixel_nsplit > 1) to use source pixel clustering");
				set_switch(use_dist_weighted_srcpixel_clustering,setword);
				for (int i=0; i < n_pixellated_src; i++) {
					update_pixsrc_active_parameters(i);
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
			if ((use_dist_weighted_srcpixel_clustering==true) and (default_imgpixel_nsplit < 3)) warn("source pixel clustering algorithm not recommended unless imgpixel_nsplit >= 3");
		}
		else if (words[0]=="lum_weighted_srcpixel_clustering")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use luminosity-weighted source pixel clustering: " << display_switch(use_lum_weighted_srcpixel_clustering) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'lum_weighted_srcpixel_clustering' command; must specify 'on' or 'off'");
				if ((setword=="on") and (!optimize_regparam) and (!use_saved_sbweights)) Complain("lum_weighted_srcpixel_clustering requires either 'optimize_regparam' or 'use_saved_sbweights' to be set to 'on'");
				if ((setword=="on") and (source_fit_mode != Delaunay_Source)) Complain("luminosity-weighted srcpixel clustering can only be used if source mode is set to 'delaunay' (see 'fit source_mode')");
				if ((setword=="on") and (use_dist_weighted_srcpixel_clustering)) Complain("dist_weighted_srcpixel_clustering and lum_weighted_srcpixel_clustering cannot both be on"); 
				if ((setword=="on") and ((split_imgpixels==false) or (default_imgpixel_nsplit==1))) Complain("split_imgpixels must be turned on (and imgpixel_nsplit > 1) to use source pixel clustering");
				set_switch(use_lum_weighted_srcpixel_clustering,setword);
				for (int i=0; i < n_pixellated_src; i++) {
					update_pixsrc_active_parameters(i);
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
			if ((use_lum_weighted_srcpixel_clustering==true) and (default_imgpixel_nsplit < 3)) warn("source pixel clustering algorithm not recommended unless imgpixel_nsplit >= 3");
		}
		else if (words[0]=="lumreg_from_sbweights")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Get luminosity-weighted regularization from saved subpixel sbweights: " << display_switch(get_lumreg_from_sbweights) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'lumreg_from_sbweights' command; must specify 'on' or 'off'");
				set_switch(get_lumreg_from_sbweights,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		/*
		else if (words[0]=="regparam_lsc")
		{
			double reg_lsc, reg_lsc_upper, reg_lsc_lower;
			if (nwords == 4) {
				if (!(ws[1] >> reg_lsc_lower)) Complain("invalid regparam_lsc lower limit");
				if (!(ws[2] >> reg_lsc)) Complain("invalid regparam_lsc value");
				if (!(ws[3] >> reg_lsc_upper)) Complain("invalid regparam_lsc upper limit");
				if ((reg_lsc < reg_lsc_lower) or (reg_lsc > reg_lsc_upper)) Complain("initial regparam_lsc should lie within specified prior limits");
				regparam_lsc = reg_lsc;
				regparam_lsc_lower_limit = reg_lsc_lower;
				regparam_lsc_upper_limit = reg_lsc_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> reg_lsc)) Complain("invalid regparam_lsc value");
				regparam_lsc = reg_lsc;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "regparam_lsc = " << regparam_lsc << endl;
			} else Complain("must specify either zero or one argument (regparam_lsc value)");
		}
		else if (words[0]=="vary_regparam_lsc")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary regparam_lsc: " << display_switch(vary_regparam_lsc) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_regparam_lsc' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before regparam_lsc can be varied (see 'fit regularization')");
				if ((setword=="on") and (!use_lum_weighted_regularization) and (!use_distance_weighted_regularization)) Complain("either 'lum_weighted_regularization', 'dist_weighted_regularization' or 'use_two_kernels' must be set to 'on' before regparam_lsc can be varied");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("regparam_lsc can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				set_switch(vary_regparam_lsc,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="regparam_lum_index")
		{
			double reg_index;
			double reg_index_upper, reg_index_lower;
			if (nwords == 4) {
				if (!(ws[1] >> reg_index_lower)) Complain("invalid regparam_lum_index lower limit");
				if (!(ws[2] >> reg_index)) Complain("invalid regparam_lum_index value");
				if (!(ws[3] >> reg_index_upper)) Complain("invalid regparam_lum_index upper limit");
				if ((reg_index < reg_index_lower) or (reg_index > reg_index_upper)) Complain("initial regparam_lum_index should lie within specified prior limits");
				regparam_lum_index = reg_index;
				regparam_lum_index_lower_limit = reg_index_lower;
				regparam_lum_index_upper_limit = reg_index_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> reg_index)) Complain("invalid regparam_lum_index value");
				regparam_lum_index = reg_index;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "regparam_lum_index = " << regparam_lum_index << endl;
			} else Complain("must specify either zero or one argument (regparam_lum_index value)");
		}
		else if (words[0]=="vary_regparam_lum_index")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary regparam_lum_index: " << display_switch(vary_regparam_lum_index) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_regparam_lum_index' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before regparam_lum_index can be varied (see 'fit regularization')");
				if ((setword=="on") and (!use_lum_weighted_regularization) and (!use_distance_weighted_regularization)) Complain("either lum_weighted_regularization or dist_weighted_regularization must be set to 'on' before regparam_lum_index can be varied");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("regparam_lum_index can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				set_switch(vary_regparam_lum_index,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}

		else if (words[0]=="mag_weight_index")
		{
			double mag_index;
			double mag_index_upper, mag_index_lower;
			if (nwords == 4) {
				if (!(ws[1] >> mag_index_lower)) Complain("invalid mag_weight_index lower limit");
				if (!(ws[2] >> mag_index)) Complain("invalid mag_weight_index value");
				if (!(ws[3] >> mag_index_upper)) Complain("invalid mag_weight_index upper limit");
				if ((mag_index < mag_index_lower) or (mag_index > mag_index_upper)) Complain("initial mag_weight_index should lie within specified prior limits");
				mag_weight_index = mag_index;
				mag_weight_index_lower_limit = mag_index_lower;
				mag_weight_index_upper_limit = mag_index_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> mag_index)) Complain("invalid mag_weight_index value");
				mag_weight_index = mag_index;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "mag_weight_index = " << mag_weight_index << endl;
			} else Complain("must specify either zero or one argument (mag_weight_index value)");
		}
		else if (words[0]=="vary_mag_weight_index")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary mag_weight_index: " << display_switch(vary_mag_weight_index) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_mag_weight_index' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before mag_weight_index can be varied (see 'fit regularization')");
				if ((setword=="on") and (!use_mag_weighted_regularization)) Complain("mag_weighted_regularization must be set to 'on' before mag_weight_index can be varied");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("mag_weight_index can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				set_switch(vary_mag_weight_index,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="mag_weight_sc")
		{
			double mag_sc, mag_sc_upper, mag_sc_lower;
			if (nwords == 4) {
				if (!(ws[1] >> mag_sc_lower)) Complain("invalid mag_weight_sc lower limit");
				if (!(ws[2] >> mag_sc)) Complain("invalid mag_weight_sc value");
				if (!(ws[3] >> mag_sc_upper)) Complain("invalid mag_weight_sc upper limit");
				if ((mag_sc < mag_sc_lower) or (mag_sc > mag_sc_upper)) Complain("initial mag_weight_sc should lie within specified prior limits");
				mag_weight_sc = mag_sc;
				mag_weight_sc_lower_limit = mag_sc_lower;
				mag_weight_sc_upper_limit = mag_sc_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> mag_sc)) Complain("invalid mag_weight_scale value");
				mag_weight_sc = mag_sc;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "mag_weight_scale = " << mag_weight_sc << endl;
			} else Complain("must specify either zero or one argument (mag_weight_sc value)");
		}
		else if (words[0]=="vary_mag_weight_sc")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary mag_weight_sc: " << display_switch(vary_mag_weight_sc) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_mag_weight_sc' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before mag_weight_sc can be varied (see 'fit regularization')");
				if ((setword=="on") and (!use_mag_weighted_regularization)) Complain("'mag_weighted_regularization' must be set to 'on' before mag_weight_sc can be varied");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("mag_weight_sc can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				set_switch(vary_mag_weight_sc,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lumreg_rc")
		{
			double lrrc, lrrc_upper, lrrc_lower;
			if (nwords == 4) {
				if (!(ws[1] >> lrrc_lower)) Complain("invalid lumreg_rc lower limit");
				if (!(ws[2] >> lrrc)) Complain("invalid lumreg_rc value");
				if (!(ws[3] >> lrrc_upper)) Complain("invalid lumreg_rc upper limit");
				if ((lrrc < lrrc_lower) or (lrrc > lrrc_upper)) Complain("initial lumreg_rc should lie within specified prior limits");
				lumreg_rc = lrrc;
				lumreg_rc_lower_limit = lrrc_lower;
				lumreg_rc_upper_limit = lrrc_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> lrrc)) Complain("invalid lumreg_rc value");
				lumreg_rc = lrrc;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "lumreg_rc = " << lumreg_rc << endl;
			} else Complain("must specify either zero or one argument (lumreg_rc value)");
		}
		else if (words[0]=="vary_lumreg_rc")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary lumreg_rc: " << display_switch(vary_lumreg_rc) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_lumreg_rc' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before lumreg_rc can be varied (see 'fit regularization')");
				if ((setword=="on") and (!use_lum_weighted_regularization) and (!use_distance_weighted_regularization)) Complain("lum_weighted_regularization or dist_weighted_regularization must be set to 'on' before lumreg_rc can be varied");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("lumreg_rc can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				set_switch(vary_lumreg_rc,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		*/
		else if (words[0]=="auto_lumreg_center")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Find center of distance-weighted regularization from centroid of ray-traced points: " << display_switch(auto_lumreg_center) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_lumreg_center' command; must specify 'on' or 'off'");
				set_switch(auto_lumreg_center,setword);
				for (int i=0; i < n_pixellated_src; i++) {
					update_pixsrc_active_parameters(i);
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lumreg_center_from_ptsource")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Find center of distance-weighted regularization from position of point source: " << display_switch(lumreg_center_from_ptsource) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'lumreg_center_from_ptsource' command; must specify 'on' or 'off'");
				if ((setword=="on") and (!auto_lumreg_center)) Complain("'auto_lumreg_center' must be set to 'on' before turning on lumreg_center_from_ptsource");
				set_switch(lumreg_center_from_ptsource,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lensed_lumreg_center")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Make lumreg_xcenter, lumreg_ycenter coordinates in the image plane: " << display_switch(lensed_lumreg_center) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'lensed_lumreg_center' command; must specify 'on' or 'off'");
				set_switch(lensed_lumreg_center,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lensed_lumreg_rc")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Define lumreg_rc in image plane instead of source plane: " << display_switch(lensed_lumreg_rc) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'lensed_lumreg_rc' command; must specify 'on' or 'off'");
				set_switch(lensed_lumreg_rc,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="fix_lumreg_sig")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Fix lumreg sigma value instead of estimating from ray-traced points: " << display_switch(fix_lumreg_sig) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'fix_lumreg_sig' command; must specify 'on' or 'off'");
				set_switch(fix_lumreg_sig,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lumreg_sig")
		{
			double lrsig;
			if (nwords == 2) {
				if (!(ws[1] >> lrsig)) Complain("invalid lumreg_sig value");
				lumreg_sig = lrsig;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "lumreg_sig = " << lumreg_sig << endl;
			} else Complain("must specify either zero or one argument (lumreg_sig value)");
		}
		/*	
		else if (words[0]=="lumreg_xcenter")
		{
			double regxc;
			double regxc_upper, regxc_lower;
			if (nwords == 4) {
				if (!(ws[1] >> regxc_lower)) Complain("invalid lumreg_xcenter lower limit");
				if (!(ws[2] >> regxc)) Complain("invalid lumreg_xcenter value");
				if (!(ws[3] >> regxc_upper)) Complain("invalid lumreg_xcenter upper limit");
				if ((regxc < regxc_lower) or (regxc > regxc_upper)) Complain("initial lumreg_xcenter should lie within specified prior limits");
				lumreg_xcenter = regxc;
				lumreg_xcenter_lower_limit = regxc_lower;
				lumreg_xcenter_upper_limit = regxc_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> regxc)) Complain("invalid lumreg_xcenter value");
				lumreg_xcenter = regxc;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "lumreg_xcenter = " << lumreg_xcenter << endl;
			} else Complain("must specify either zero or one argument (lumreg_xcenter value)");
		}
		else if (words[0]=="vary_lumreg_xcenter")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary lumreg_xcenter: " << display_switch(vary_lumreg_xcenter) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_lumreg_xcenter' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before lumreg_xcenter can be varied (see 'fit regularization')");
				if ((setword=="on") and (!use_distance_weighted_regularization)) Complain("dist_weighted_regularization must be set to 'on' before lumreg_xcenter can be varied");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("lumreg_xcenter can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				if ((setword=="on") and (auto_lumreg_center)) Complain("cannot vary lumreg_xcenter if auto_lumreg_center is set to 'on'");
				set_switch(vary_lumreg_xcenter,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lumreg_ycenter")
		{
			double regyc;
			double regyc_upper, regyc_lower;
			if (nwords == 4) {
				if (!(ws[1] >> regyc_lower)) Complain("invalid lumreg_ycenter lower limit");
				if (!(ws[2] >> regyc)) Complain("invalid lumreg_ycenter value");
				if (!(ws[3] >> regyc_upper)) Complain("invalid lumreg_ycenter upper limit");
				if ((regyc < regyc_lower) or (regyc > regyc_upper)) Complain("initial lumreg_ycenter should lie within specified prior limits");
				lumreg_ycenter = regyc;
				lumreg_ycenter_lower_limit = regyc_lower;
				lumreg_ycenter_upper_limit = regyc_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> regyc)) Complain("invalid lumreg_ycenter value");
				lumreg_ycenter = regyc;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "lumreg_ycenter = " << lumreg_ycenter << endl;
			} else Complain("must specify either zero or one argument (lumreg_ycenter value)");
		}
		else if (words[0]=="vary_lumreg_ycenter")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary lumreg_ycenter: " << display_switch(vary_lumreg_ycenter) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_lumreg_ycenter' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before lumreg_ycenter can be varied (see 'fit regularization')");
				if ((setword=="on") and (!use_distance_weighted_regularization)) Complain("dist_weighted_regularization must be set to 'on' before lumreg_ycenter can be varied");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("lumreg_ycenter can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				if ((setword=="on") and (auto_lumreg_center)) Complain("cannot vary lumreg_ycenter if auto_lumreg_center is set to 'on'");
				set_switch(vary_lumreg_ycenter,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lumreg_e1")
		{
			double rege1;
			double rege1_upper, rege1_lower;
			if (nwords == 4) {
				if (!(ws[1] >> rege1_lower)) Complain("invalid lumreg_e1 lower limit");
				if (!(ws[2] >> rege1)) Complain("invalid lumreg_e1 value");
				if (!(ws[3] >> rege1_upper)) Complain("invalid lumreg_e1 upper limit");
				if ((rege1 < rege1_lower) or (rege1 > rege1_upper)) Complain("initial lumreg_e1 should lie within specified prior limits");
				lumreg_e1 = rege1;
				lumreg_e1_lower_limit = rege1_lower;
				lumreg_e1_upper_limit = rege1_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> rege1)) Complain("invalid lumreg_e1 value");
				lumreg_e1 = rege1;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "lumreg_e1 = " << lumreg_e1 << endl;
			} else Complain("must specify either zero or one argument (lumreg_e1 value)");
		}
		else if (words[0]=="vary_lumreg_e1")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary lumreg_e1: " << display_switch(vary_lumreg_e1) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_lumreg_e1' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before lumreg_e1 can be varied (see 'fit regularization')");
				if ((setword=="on") and (!use_distance_weighted_regularization)) Complain("dist_weighted_regularization must be set to 'on' before lumreg_e1 can be varied");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("lumreg_e1 can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				set_switch(vary_lumreg_e1,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="lumreg_e2")
		{
			double rege2;
			double rege2_upper, rege2_lower;
			if (nwords == 4) {
				if (!(ws[1] >> rege2_lower)) Complain("invalid lumreg_e2 lower limit");
				if (!(ws[2] >> rege2)) Complain("invalid lumreg_e2 value");
				if (!(ws[3] >> rege2_upper)) Complain("invalid lumreg_e2 upper limit");
				if ((rege2 < rege2_lower) or (rege2 > rege2_upper)) Complain("initial lumreg_e2 should lie within specified prior limits");
				lumreg_e2 = rege2;
				lumreg_e2_lower_limit = rege2_lower;
				lumreg_e2_upper_limit = rege2_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> rege2)) Complain("invalid lumreg_e2 value");
				lumreg_e2 = rege2;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "lumreg_e2 = " << lumreg_e2 << endl;
			} else Complain("must specify either zero or one argument (lumreg_e2 value)");
		}
		else if (words[0]=="vary_lumreg_e2")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary lumreg_e2: " << display_switch(vary_lumreg_e2) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_lumreg_e2' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before lumreg_e2 can be varied (see 'fit regularization')");
				if ((setword=="on") and (!use_distance_weighted_regularization)) Complain("dist_weighted_regularization must be set to 'on' before lumreg_e2 can be varied");
				if ((setword=="on") and ((source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Shapelet_Source))) Complain("lumreg_e2 can only be varied if source mode is set to 'cartesian', 'delaunay' or 'shapelet' (see 'fit source_mode')");
				set_switch(vary_lumreg_e2,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="alpha_clus")
		{
			double alpha_cl;
			double alpha_cl_upper, alpha_cl_lower;
			if (nwords == 4) {
				if (!(ws[1] >> alpha_cl_lower)) Complain("invalid alpha_clus lower limit");
				if (!(ws[2] >> alpha_cl)) Complain("invalid alpha_clus value");
				if (!(ws[3] >> alpha_cl_upper)) Complain("invalid alpha_clus upper limit");
				if ((alpha_cl < alpha_cl_lower) or (alpha_cl > alpha_cl_upper)) Complain("initial alpha_clus should lie within specified prior limits");
				alpha_clus = alpha_cl;
				alpha_clus_lower_limit = alpha_cl_lower;
				alpha_clus_upper_limit = alpha_cl_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> alpha_cl)) Complain("invalid alpha_clus value");
				alpha_clus = alpha_cl;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "alpha_clus = " << alpha_clus << endl;
			} else Complain("must specify either zero or one argument (alpha_clus value)");
		}
		else if (words[0]=="vary_alpha_clus")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary alpha_clus: " << display_switch(vary_alpha_clus) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_alpha_clus' command; must specify 'on' or 'off'");
				if ((setword=="on") and ((!use_lum_weighted_srcpixel_clustering) and (!use_dist_weighted_srcpixel_clustering))) Complain("dist_weighted_srcpixel_clustering or lum_weighted_srcpixel_clustering must be set to 'on' before alpha_clus can be varied");
				if ((setword=="on") and (source_fit_mode != Delaunay_Source)) Complain("alpha_clus can only be varied if source mode is set to 'delaunay' (see 'fit source_mode')");
				set_switch(vary_alpha_clus,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="beta_clus")
		{
			double beta_cl;
			double beta_cl_upper, beta_cl_lower;
			if (nwords == 4) {
				if (!(ws[1] >> beta_cl_lower)) Complain("invalid beta_clus lower limit");
				if (!(ws[2] >> beta_cl)) Complain("invalid beta_clus value");
				if (!(ws[3] >> beta_cl_upper)) Complain("invalid beta_clus upper limit");
				if ((beta_cl < beta_cl_lower) or (beta_cl > beta_cl_upper)) Complain("initial beta_clus should lie within specified prior limits");
				beta_clus = beta_cl;
				beta_clus_lower_limit = beta_cl_lower;
				beta_clus_upper_limit = beta_cl_upper;
			} else if (nwords == 2) {
				if (!(ws[1] >> beta_cl)) Complain("invalid beta_clus value");
				beta_clus = beta_cl;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "beta_clus = " << beta_clus << endl;
			} else Complain("must specify either zero or one argument (beta_clus value)");
		}
		else if (words[0]=="vary_beta_clus")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Vary beta_clus: " << display_switch(vary_beta_clus) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_beta_clus' command; must specify 'on' or 'off'");
				if ((setword=="on") and ((!use_lum_weighted_srcpixel_clustering) and (!use_dist_weighted_srcpixel_clustering))) Complain("dist_weighted_srcpixel_clustering or lum_weighted_srcpixel_clustering must be set to 'on' before beta_clus can be varied");
				if ((setword=="on") and (source_fit_mode != Delaunay_Source)) Complain("beta_clus can only be varied if source mode is set to 'delaunay' (see 'fit source_mode')");
				set_switch(vary_beta_clus,setword);
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		*/
		else if (words[0]=="corrlength")
		{
			double corrlength, corrlength_ul, corrlength_ll;
			if (nwords == 4) {
				if (!(ws[1] >> corrlength_ll)) Complain("invalid kernel correlation length lower limit");
				if (!(ws[2] >> corrlength)) Complain("invalid kernel correlation length value");
				if (!(ws[3] >> corrlength_ul)) Complain("invalid kernel correlation length upper limit");
				if ((corrlength < corrlength_ll) or (corrlength > corrlength_ul)) Complain("initial kernel correlation length should lie within specified prior limits");
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				if (source_fit_mode==Delaunay_Source) {
					for (int i=0; i < n_pixellated_src; i++) {
						delaunay_srcgrids[i]->update_specific_parameter("corrlength",corrlength);
						delaunay_srcgrids[i]->set_limits_specific_parameter("corrlength",corrlength_ll,corrlength_ul);
					}
				}
			} else if (nwords == 2) {
				if (!(ws[1] >> corrlength)) Complain("invalid kernel correlation length value");
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					if (source_fit_mode==Delaunay_Source) {
						delaunay_srcgrids[i]->update_specific_parameter("corrlength",corrlength);
					}
				}
			} else if (nwords==1) {
				if (mpi_id==0) {
					if (n_pixellated_src==0) cout << "No pixellated sources have been created yet" << endl;
					for (int i=0; i < n_pixellated_src; i++) {
						if (!delaunay_srcgrids[i]->get_specific_parameter("corrlength",corrlength)) Complain("correlation length is not an active parameter for pixellated sources (wrong regularization method)");
						cout << "pixsrc " << i << ": corrlength = " << corrlength << endl;
					}
				}
			} else Complain("must specify either zero or one argument (kernel correlation length value)");
		}
		else if (words[0]=="vary_corrlength")
		{
			bool vary_corrlength;
			if (nwords==1) {
				if (mpi_id==0) {
					if (n_pixellated_src==0) cout << "No pixellated sources have been created yet" << endl;
					for (int i=0; i < n_pixellated_src; i++) {
						if (source_fit_mode==Delaunay_Source) {
							if (!delaunay_srcgrids[i]->get_specific_varyflag("corrlength",vary_corrlength)) Complain("correleation length is not an active parameter for pixellated sources (wrong regularization method)");
							cout << "Vary correlation length for pixsrc " << i << ": " << display_switch(vary_corrlength) << endl;
						}
					}
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_corrlength' command; must specify 'on' or 'off'");
				if (setword=="on") {
					if (regularization_method==None) Complain("regularization method must be chosen before corrlength can be varied (see 'fit regularization')");
					if (source_fit_mode != Delaunay_Source) Complain("corrlength can only be varied if source mode is set to 'delaunay' (see 'fit source_mode')");
				}
				set_switch(vary_corrlength,setword);
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					if (source_fit_mode==Delaunay_Source) update_pixellated_src_varyflag(i,"corrlength",vary_corrlength);
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="matern_index")
		{
			double mat_index, mat_index_ul, mat_index_ll;
			if (nwords == 4) {
				if (!(ws[1] >> mat_index_ll)) Complain("invalid kernel Matern index lower limit");
				if (!(ws[2] >> mat_index)) Complain("invalid kernel Matern index value");
				if (!(ws[3] >> mat_index_ul)) Complain("invalid kernel Matern index upper limit");
				if ((mat_index < mat_index_ll) or (mat_index > mat_index_ul)) Complain("initial kernel Matern index should lie within specified prior limits");
				//if ((mat_index_ul > 5) and (use_matern_scale_parameter)) Complain("matern indices greater than 5 are not advisable when use_matern_scale is set to 'on'");
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					if (source_fit_mode==Delaunay_Source) {
						delaunay_srcgrids[i]->update_specific_parameter("matern_index",mat_index);
						delaunay_srcgrids[i]->set_limits_specific_parameter("matern_index",mat_index_ll,mat_index_ul);
					}
				}
			} else if (nwords == 2) {
				if (!(ws[1] >> mat_index)) Complain("invalid kernel Matern index value");
				//if ((mat_index > 5) and (use_matern_scale_parameter)) Complain("matern indices greater than 5 are not advisable when use_matern_scale is set to 'on'");
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					if (source_fit_mode==Delaunay_Source) {
						delaunay_srcgrids[i]->update_specific_parameter("matern_index",mat_index);
					}
				}
			} else if (nwords==1) {
				if (mpi_id==0) {
					if (n_pixellated_src==0) cout << "No pixellated sources have been created yet" << endl;
					for (int i=0; i < n_pixellated_src; i++) {
						if (source_fit_mode==Delaunay_Source) {
							if (!delaunay_srcgrids[i]->get_specific_parameter("matern_index",mat_index)) Complain("matern index is undefined for pixellated sources (regularization method should be set to 'matern_kernel'");
							cout << "pixsrc " << i << ": matern_index = " << mat_index << endl;
						}
					}
				}
			} else Complain("must specify either zero or one argument (kernel Matern index value)");
		}
		else if (words[0]=="vary_matern_index")
		{
			bool vary_matern_index;
			if (nwords==1) {
				if (mpi_id==0) {
					if (n_pixellated_src==0) cout << "No pixellated sources have been created yet" << endl;
					for (int i=0; i < n_pixellated_src; i++) {
						if (source_fit_mode==Delaunay_Source) {
							if (!delaunay_srcgrids[i]->get_specific_varyflag("matern_index",vary_matern_index)) Complain("matern index is not an active parameter for pixellated sources (wrong regularization method)");
							cout << "Vary matern index for pixsrc " << i << ": " << display_switch(vary_matern_index) << endl;
						}
					}
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'vary_matern_index' command; must specify 'on' or 'off'");
				if ((setword=="on") and (regularization_method==None)) Complain("regularization method must be chosen before matern_index can be varied (see 'fit regularization')");
				if ((setword=="on") and (source_fit_mode != Delaunay_Source)) Complain("matern_index can only be varied if source mode is set to 'delaunay' (see 'fit source_mode')");
				set_switch(vary_matern_index,setword);
				if (n_pixellated_src==0) {
					if (mpi_id==0) cout << "Creating pixellated source with redshift zsrc=" << source_redshift << endl;
					add_pixellated_source(source_redshift);
				}
				for (int i=0; i < n_pixellated_src; i++) {
					if (source_fit_mode==Delaunay_Source) update_pixellated_src_varyflag(i,"matern_index",vary_matern_index);
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="find_cov_inverse")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Find Rmatrix via taking explicit inverse of covariance matrix (for cov. kernel regularization): " << display_switch(find_covmatrix_inverse) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'find_cov_inverse' command; must specify 'on' or 'off'");
				set_switch(find_covmatrix_inverse,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="covmatrix_penalty")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Penalize defective (not positive definite) covariance matrices: " << display_switch(penalize_defective_covmatrix) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'covmatrix_penalty' command; must specify 'on' or 'off'");
				set_switch(penalize_defective_covmatrix,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="covmatrix_epsilon")
		{
			double eps;
			if (nwords == 2) {
				if (!(ws[1] >> eps)) Complain("invalid data pixel surface brightness noise");
				covmatrix_epsilon = eps;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "covmatrix_epsilon = " << covmatrix_epsilon << endl;
			} else Complain("must specify either zero or one argument (covmatrix_epsilon)");
		}
		else if ((words[0]=="bg_pixel_noise") or (words[0]=="data_pixel_noise")) // Note, 'data_pixel_noise' is deprecated
		{
			double pnoise;
			if (nwords == 2) {
				if (!(ws[1] >> pnoise)) Complain("invalid image pixel surface brightness noise");
				background_pixel_noise = pnoise;
				if ((n_data_bands > 0) and (!use_noise_map)) imgpixel_data_list[0]->set_uniform_pixel_noise(pnoise);
			} else if (nwords==1) {
				if (mpi_id==0) cout << "background image pixel surface brightness dispersion = " << background_pixel_noise << endl;
			} else Complain("must specify either zero or one argument (background image pixel surface brightness dispersion)");
		}
		else if (words[0]=="simulate_pixel_noise")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Add pixel noise to simulated images (simulate_pixel_noise): " << display_switch(simulate_pixel_noise) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'simulate_pixel_noise' command; must specify 'on' or 'off'");
				set_switch(simulate_pixel_noise,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="random_seed")
		{
			long long int seed;
			if (nwords == 2) {
				if (!(ws[1] >> seed)) Complain("invalid value for random seed");
				random_seed = seed;
				set_random_seed(seed);
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Random number generator seed = " << random_seed << endl;
			} else Complain("must specify either zero or one argument for random_seed");
		}
		else if (words[0]=="random_reinit")
		{
			reinitialize_random_generator();
		}
		else if (words[0]=="nchisq")
		{
			long long int nchisq;
			if (nwords == 2) {
				if (!(ws[1] >> nchisq)) Complain("invalid value for random nchisq");
				n_ranchisq = nchisq;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Number of (averaged) likelihood evals nchisq = " << n_ranchisq << endl;
			} else Complain("must specify either zero or one argument for nchisq");
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
				if (n_data_bands > 0) imgpixel_data_list[0]->assign_high_sn_pixels();
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
		else if (words[0]=="outside_sb_frac_threshold")
		{
			double sb_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> sb_thresh)) Complain("invalid surface brightness noise threshold (should be as fraction of max s.b.)");
				outside_sb_prior_threshold = sb_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "surface brightness fraction threshold for outside_sb_frac_threshold = " << outside_sb_prior_threshold << endl;
			} else Complain("must specify either zero or one argument for outside_sb_frac_threshold");
		}
		else if (words[0]=="emask_n_neighbors")
		{
			// This command should be made obsolete. Should just do this with 'set_neighbor_pixels' command together with an -emask option, and an option to reset emask to be same as regular mask. IMPLEMENT THIS!!
			if (n_data_bands==0) Complain("must load image pixel data before setting emask_n_neighbors");
			int mask_i=0;
			int band_i=0;
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("mask=")) != string::npos) {
					string mnumstring = words[i].substr(pos+5);
					stringstream mnumstr;
					mnumstr << mnumstring;
					if (!(mnumstr >> mask_i)) Complain("incorrect format for lens redshift");
					if (mask_i < 0) Complain("lens redshift cannot be negative");
					remove_word(i);
					break;
				}
			}
			for (int i=1; i < nwords; i++) {
				int pos;
				if ((pos = words[i].find("band=")) != string::npos) {
					string bnumstring = words[i].substr(pos+5);
					stringstream bnumstr;
					bnumstr << bnumstring;
					if (!(bnumstr >> band_i)) Complain("incorrect format for band number");
					if (band_i < 0) Complain("band number cannot be negative");
					remove_word(i);
					break;
				}
			}	

			double emask_n;
			bool only_interior_pixels = false;
			bool only_exterior_pixels = false;
			bool add_to_emask = false;
			if (words[nwords-1]=="-add") {
				add_to_emask = true;
				remove_word(nwords-1);
			}
			if (nwords >= 2) {
				if (words[1]=="all") {
					imgpixel_data_list[band_i]->extended_mask_n_neighbors[mask_i] = emask_n = -1;
				}
				else if (words[1]=="interior") {
					only_interior_pixels = true;
					add_to_emask = true;
					emask_n = 1000;
					if (nwords > 2) {
						if (!(ws[2] >> emask_n)) Complain("invalid number of neighbor pixels for extended mask");
					}
				}
				else {
					if (!(ws[1] >> emask_n)) Complain("invalid number of neighbor pixels for extended mask");
					if ((emask_n != -1) and (adaptive_subgrid)) Complain("emask_n_neighbors must be set to 'all' for adaptive Cartesian grid");
					imgpixel_data_list[band_i]->extended_mask_n_neighbors[mask_i] = emask_n;
				}
				imgpixel_data_list[band_i]->set_extended_mask(emask_n,add_to_emask,only_interior_pixels,mask_i);
				int npix;
				npix = imgpixel_data_list[band_i]->get_size_of_extended_mask(mask_i);
				if (mpi_id==0) cout << "number of pixels in extended mask: " << npix << endl;
			} else if (nwords==1) {
				if (mpi_id==0) {
					if (imgpixel_data_list[band_i]->extended_mask_n_neighbors[mask_i]==-1) cout << "number of neighbor pixels for extended mask: emask_n_neighbors = all" << endl;
					else cout << "number of neighbor pixels for extended mask: emask_n_neighbors = " << imgpixel_data_list[band_i]->extended_mask_n_neighbors[mask_i] << endl;
					int npix = imgpixel_data_list[band_i]->get_size_of_extended_mask(mask_i);
					cout << "number of pixels in extended mask: " << npix << endl;
				}
			} else Complain("must specify either zero or one argument for emask_n_neighbors");
		}
		else if (words[0]=="fgmask_padding")
		{
			double padding;
			if (nwords == 2) {
				if (!(ws[1] >> padding)) Complain("invalid number of neighbor pixels for padding foreground mask");
				fgmask_padding = padding;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "# of neighbor pixels for padding foreground mask = " << fgmask_padding << endl;
			} else Complain("must specify either zero or one argument for fgmask_padding");
		}
		else if (words[0]=="include_fgmask_in_inversion")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include foreground mask in inversion (include_fgmask_in_inversion): " << display_switch(include_fgmask_in_inversion) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'include_fgmask_in_inversion' command; must specify 'on' or 'off'");
				set_switch(include_fgmask_in_inversion,setword);
				for (int i=0; i < n_extended_src_redshifts; i++) {
					if ((image_pixel_grids != NULL) and (image_pixel_grids[i] != NULL)) image_pixel_grids[i]->update_mask_values(include_fgmask_in_inversion);
				}
				if (fft_convolution) cleanup_FFT_convolution_arrays();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="include_two_pixsrc_in_inversion")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include two pixellated sources in inversion (include_two_pixsrc_in_inversion): " << display_switch(include_two_pixsrc_in_Lmatrix) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'include_two_pixsrc_in_inversion' command; must specify 'on' or 'off'");
				set_switch(include_two_pixsrc_in_Lmatrix,setword);
				if ((include_two_pixsrc_in_Lmatrix==true) and (include_fgmask_in_inversion==false)) {
					include_fgmask_in_inversion = true;
					for (int i=0; i < n_extended_src_redshifts; i++) {
						if ((image_pixel_grids != NULL) and (image_pixel_grids[i] != NULL)) image_pixel_grids[i]->update_mask_values(include_fgmask_in_inversion);
					}
					if (fft_convolution) cleanup_FFT_convolution_arrays();
					if (mpi_id==0) cout << "NOTE: Setting include_fgmask_in_inversion to 'on'" << endl;
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="include_noise_term_in_loglike")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Include pixel noise term in log-likelihood (include_noise_term_in_loglike): " << display_switch(include_noise_term_in_loglike) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'include_noise_term_in_loglike' command; must specify 'on' or 'off'");
				set_switch(include_noise_term_in_loglike,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="zero_lensed_sb_fgmask_prior")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use zero SB prior for lensed sources in foreground mask pixels: " << display_switch(zero_sb_fgmask_prior) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'zero_lensed_sb_emask_prior' command; must specify 'on' or 'off'");
				set_switch(zero_sb_fgmask_prior,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="zero_outside_border")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Enforce zero surface brightness outside border of Delaunay grid: " << display_switch(DelaunaySourceGrid::zero_outside_border) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'zero_outside_border' command; must specify 'on' or 'off'");
				set_switch(DelaunaySourceGrid::zero_outside_border,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="nimg_sb_frac_threshold")
		{
			double sb_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> sb_thresh)) Complain("invalid surface brightness noise threshold (should be as multiple of data noise)");
				n_image_prior_sb_frac = sb_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "surface brightness fraction threshold for nimg_sb_frac_threshold = " << n_image_prior_sb_frac << endl;
			} else Complain("must specify either zero or one argument for nimg_sb_frac_threshold");
		}
		else if (words[0]=="Re_threshold_low")
		{
			double re_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> re_thresh)) Complain("invalid Einstein radius prior threshold");
				einstein_radius_low_threshold = re_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Einstein radius low threshold = " << einstein_radius_low_threshold << endl;
			} else Complain("must specify either zero or one argument for Re_threshold_low");
		}
		else if (words[0]=="Re_threshold_high")
		{
			double re_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> re_thresh)) Complain("invalid Einstein radius prior threshold");
				einstein_radius_high_threshold = re_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "Einstein radius high threshold = " << einstein_radius_high_threshold << endl;
			} else Complain("must specify either zero or one argument for Re_threshold_high");
		}
		else if ((words[0]=="adaptive_subgrid") or (words[0]=="adaptive_grid"))
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Adaptive splitting of Cartesian source grid (adaptive_subgrid): " << display_switch(adaptive_subgrid) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'adaptive_subgrid' command; must specify 'on' or 'off'");
				//if ((extended_mask_n_neighbors != -1) and (setword=="on")) Complain("adaptive grid cannot reliably be used unless emask_n_neighbors is set to 'all'");
				set_switch(adaptive_subgrid,setword);
				if (source_fit_mode==Cartesian_Source) {
					for (int i=0; i < n_pixellated_src; i++) {
						update_pixsrc_active_parameters(i);
					}
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="srcgrid_maxlevels")
		{
			// This is the maximum number of times that the cartesian source grid cells can be split
			if (nwords==1) {
				if (mpi_id==0) cout << "Number of allowed splittings of Cartesian source pixels = " << SourcePixelGrid::max_levels << endl;
			} else if (nwords == 2) {
				int nlev;
				if (!(ws[1] >> nlev)) Complain("invalid number of splittings");
				SourcePixelGrid::max_levels = nlev;
			} else Complain("only one argument allowed for 'srcgrid_maxlevels'");
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
		else if (words[0]=="nimg_mag_threshold")
		{
			double nimg_mag_thresh;
			if (nwords == 2) {
				if (!(ws[1] >> nimg_mag_thresh)) Complain("invalid magnification threshold for calculating number of images of source pixels");
				srcpixel_nimg_mag_threshold = nimg_mag_thresh;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "threshold pixel magnification for calculating n_images of source pixels = " << srcpixel_nimg_mag_threshold << endl;
			} else Complain("must specify either zero or one argument for nimg_mag_threshold");
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
		else if (words[0]=="Re_prior")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Set prior on minimum Einstein radius allowed for primary (+ optional secondary) lens: " << display_switch(einstein_radius_prior) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'einstein_radius_prior' command; must specify 'on' or 'off'");
				set_switch(einstein_radius_prior,setword);
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
		else if (words[0]=="delaunay_high_sn_mode")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Switch to Delaunay mode 0 for high S/N regions: " << display_switch(delaunay_high_sn_mode) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'delaunay_high_sn_mode' command; must specify 'on' or 'off'");
				if ((!delaunay_high_sn_mode) and (setword=="on") and (n_ptsrc > 0)) Complain("'delaunay_high_sn_mode' cannot be used if point sources are present, since the extended surface brightness cannot be obtained directly from the image data");
				set_switch(delaunay_high_sn_mode,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="delaunay_sbfrac")
		{
			if (nwords == 2) {
				double sbfrac;
				if (!(ws[1] >> sbfrac)) Complain("invalid himag sbfracold for image pixel splittings");
				delaunay_high_sn_sbfrac = sbfrac;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "delaunay_sbfrac = " << delaunay_high_sn_sbfrac << endl;
			} else Complain("must specify either zero or one argument");
		}
		else if (words[0]=="use_srcpixel_clustering")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use clustering algorithm to find adaptive grid source pixels: " << display_switch(use_srcpixel_clustering) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'use_srcpixel_clustering' command; must specify 'on' or 'off'");
				if ((setword=="on") and ((split_imgpixels==false) or (default_imgpixel_nsplit==1))) Complain("split_imgpixels must be turned on (and imgpixel_nsplit > 1) to use source pixel clustering");
				set_switch(use_srcpixel_clustering,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
			if ((use_srcpixel_clustering==true) and (default_imgpixel_nsplit < 3)) warn("source pixel clustering algorithm not recommended unless imgpixel_nsplit >= 3");
		}
		else if (words[0]=="use_f_src_clusters")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Define number of src clusters in terms of fraction of data pixels: " << display_switch(use_f_src_clusters) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'use_f_src_clusters' command; must specify 'on' or 'off'");
				set_switch(use_f_src_clusters,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="use_saved_sbweights")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use saved sbweights for luminosity-weighted clustering of adaptive grid source pixels: " << display_switch(use_saved_sbweights) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'use_saved_sbweights' command; must specify 'on' or 'off'");
				set_switch(use_saved_sbweights,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="random_delaunay_srcgrid")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use random adaptive grid source pixels: " << display_switch(use_random_delaunay_srcgrid) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'random_delaunay_srcgrid' command; must specify 'on' or 'off'");
				set_switch(use_random_delaunay_srcgrid,setword);
				if (use_srcpixel_clustering) {
					use_srcpixel_clustering = false;
					if (mpi_id==0) cout << "Setting 'use_srcpixel_clustering' to 'off'" << endl;
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="random_grid_reinit")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Reinitialize random grid each time: " << display_switch(reinitialize_random_grid) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'random_grid_reinit' command; must specify 'on' or 'off'");
				set_switch(reinitialize_random_grid,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="weight_initial_centroids")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Weight initial centroids by luminosity: " << display_switch(weight_initial_centroids) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'weight_initial_centroids' command; must specify 'on' or 'off'");
				if ((setword=="on") and (!use_lum_weighted_srcpixel_clustering)) Complain("This option requires 'lum_weighted_srcpixel_clustering' to be turned on");
				set_switch(weight_initial_centroids,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="dualtree_kmeans")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use dual-tree k-means algorithm for clustering (instead of naive k-means): " << display_switch(use_dualtree_kmeans) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'dualtree_kmeans' command; must specify 'on' or 'off'");
				set_switch(use_dualtree_kmeans,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="clustering_rand_init")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use random initialization of clustering algorithm to find adaptive grid source pixels: " << display_switch(clustering_random_initialization) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'clustering_rand_init' command; must specify 'on' or 'off'");
				set_switch(clustering_random_initialization,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="split_imgpixels")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Split image pixels when ray tracing: " << display_switch(split_imgpixels) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'split_imgpixels' command; must specify 'on' or 'off'");
				bool old_setting = split_imgpixels;
				if ((setword=="off") and ((use_srcpixel_clustering) or (use_lum_weighted_srcpixel_clustering))) {
					if (mpi_id==0) cout << "NOTE: turning off source pixel clustering" << endl;
					use_srcpixel_clustering = false;
					use_lum_weighted_srcpixel_clustering = false;
				}
				set_switch(split_imgpixels,setword);
				//if (image_pixel_grid) {
					//delete image_pixel_grid;
					//image_pixel_grid = NULL;
				//}
				if (split_imgpixels != old_setting) {
					if (psf_supersampling) {
						psf_supersampling = false;
						if (mpi_id==0) cout << "NOTE: Turning off PSF supersampling" << endl;
					}
					if (image_pixel_grids != NULL) {
						for (int i=0; i < n_extended_src_redshifts; i++) {
							if (image_pixel_grids[i] != NULL) {
								image_pixel_grids[i]->delete_ray_tracing_arrays();
								image_pixel_grids[i]->setup_ray_tracing_arrays();
								if (nlens > 0) image_pixel_grids[i]->calculate_sourcepts_and_areas(true);
							}
						}
					}
					if (fft_convolution) cleanup_FFT_convolution_arrays();
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="psf_supersampling")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "PSF supersampling: " << display_switch(psf_supersampling) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'PSF supersampling' command; must specify 'on' or 'off'");
				if ((setword=="on") and (!split_imgpixels)) Complain("cannot use PSF supersampling unless 'split_imgpixels' is set to 'on'");
				bool ss_orig = psf_supersampling;
				set_switch(psf_supersampling,setword);
				if (psf_supersampling != ss_orig) {
					if (fft_convolution) cleanup_FFT_convolution_arrays();
				}
				for (int band_i=0; band_i < n_model_bands; band_i++) {
					if ((psf_supersampling) and (psf_list[band_i]->use_input_psf_matrix)) {
						psf_list[band_i]->generate_supersampled_PSF_matrix();
						if (mpi_id==0) cout << "Generated supersampled PSF matrix (dimensions: " << psf_list[band_i]->supersampled_psf_npixels_x << " " << psf_list[band_i]->supersampled_psf_npixels_y << ")" << endl;
					}
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="split_himag_imgpixels")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Split only high-magnification image pixels when ray tracing: " << display_switch(split_high_mag_imgpixels) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'split_himag_imgpixels' command; must specify 'on' or 'off'");
				set_switch(split_high_mag_imgpixels,setword);
				//if (image_pixel_grid) {
					//delete image_pixel_grid;
					//image_pixel_grid = NULL;
				//}
				if (image_pixel_grids != NULL) {
					for (int i=0; i < n_extended_src_redshifts; i++) {
						if (image_pixel_grids[i] != NULL) {
							image_pixel_grids[i]->delete_ray_tracing_arrays();
							image_pixel_grids[i]->setup_ray_tracing_arrays();
							if (nlens > 0) image_pixel_grids[i]->calculate_sourcepts_and_areas(true);
						}
					}
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="delaunay_from_pixel_centers")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Define Delaunay grid from pixel centers even if 'split_imgpixels' is turned on: " << display_switch(delaunay_from_pixel_centers) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'delaunay_from_pixel_centers' command; must specify 'on' or 'off'");
				if (use_srcpixel_clustering) Complain("use_srcpixel_clustering must be turned off for this option");
				if (use_lum_weighted_srcpixel_clustering) Complain("lum_weighted_srcpixel_clustering must be turned off for this option");
				set_switch(delaunay_from_pixel_centers,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="raytrace_pixel_centers")
		{
			// this will find surface brightnesses by ray-tracing the pixel centers, even if the Delaunay grid is constructed using k-means
			if (nwords==1) {
				if (mpi_id==0) cout << "find surface brightness by ray-tracing pixel centers even if 'split_imgpixels' is turned on: " << display_switch(raytrace_using_pixel_centers) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'raytrace_pixel_centers' command; must specify 'on' or 'off'");
				set_switch(raytrace_using_pixel_centers,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="imgpixel_nsplit")
		{
			// NOTE: currently only pixels in the primary mask are split; pixels in extended mask are NOT split (see setup_ray_tracing_arrays() in pixelgrid.cpp)
			if (nwords == 2) {
				bool changed_npix = false;
				if (words[1]=="-shapelet") {
					if (source_fit_mode != Shapelet_Source) Complain("must be in shapelet mode to set pixel splitting from shapelets");
					if (!set_shapelet_imgpixel_nsplit(0)) Complain("could not set pixel splitting from shapelets");
					if (mpi_id==0) cout << "optimal imgpixel_nsplit from shapelets = " << default_imgpixel_nsplit << endl;
				} else {
					int nt;
					if (!(ws[1] >> nt)) Complain("invalid number of image pixel splittings");
					if (nt != default_imgpixel_nsplit) {
						default_imgpixel_nsplit = nt;
						changed_npix = true;
					}
				}
				if (changed_npix) {
					if (image_pixel_grids != NULL) {
						for (int i=0; i < n_extended_src_redshifts; i++) {
							if (image_pixel_grids[i] != NULL) {
								image_pixel_grids[i]->delete_ray_tracing_arrays();
								image_pixel_grids[i]->setup_ray_tracing_arrays();
								if (nlens > 0) image_pixel_grids[i]->calculate_sourcepts_and_areas(true);
							}
						}
					}
					for (int band_i=0; band_i < n_model_bands; band_i++) {
						if ((psf_supersampling) and (psf_list[band_i]->use_input_psf_matrix)) {
							psf_list[band_i]->generate_supersampled_PSF_matrix();
							if (mpi_id==0) cout << "Generated supersampled PSF matrix (dimensions: " << psf_list[band_i]->supersampled_psf_npixels_x << " " << psf_list[band_i]->supersampled_psf_npixels_y << ")" << endl;
						}
					}
					if (fft_convolution) cleanup_FFT_convolution_arrays();
				}
			} else if (nwords==1) {
				if (mpi_id==0) cout << "default number of image pixel splittings = " << default_imgpixel_nsplit << endl;
			} else Complain("must specify either zero or one argument (default number of image pixel splittings)");
		}
		else if (words[0]=="emask_nsplit")
		{
			if (nwords == 2) {
				int nt;
				if (!(ws[1] >> nt)) Complain("invalid number of image pixel splittings");
				emask_imgpixel_nsplit = nt;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "number of extended mask image pixel splittings = " << emask_imgpixel_nsplit << endl;
			} else Complain("must specify either zero or one argument (number of extended mask image pixel splittings)");
		}
		else if (words[0]=="imgpixel_mag_threshold")
		{
			// NOTE: currently only pixels in the primary mask are split; pixels in extended mask are NOT split (see setup_ray_tracing_arrays() in pixelgrid.cpp)
			if (nwords == 2) {
				double thresh;
				if (!(ws[1] >> thresh)) Complain("invalid himag threshold for image pixel splittings");
				imgpixel_himag_threshold = thresh;
				// Assuming here the imgpixel_mag_threshold has been changed...
				if (image_pixel_grids != NULL) {
					for (int i=0; i < n_extended_src_redshifts; i++) {
						if (image_pixel_grids[i] != NULL) {
							image_pixel_grids[i]->delete_ray_tracing_arrays();
							image_pixel_grids[i]->setup_ray_tracing_arrays();
							if (nlens > 0) image_pixel_grids[i]->calculate_sourcepts_and_areas(true);
						}
					}
				}
			} else if (nwords==1) {
				if (mpi_id==0) cout << "high magnification threshold for splitting image pixels = " << imgpixel_himag_threshold << endl;
			} else Complain("must specify either zero or one argument (magnification threshold for image pixel splittings)");
		}
		else if (words[0]=="imgpixel_lomag_threshold")
		{
			// NOTE: currently only pixels in the primary mask are split; pixels in extended mask are NOT split (see setup_ray_tracing_arrays() in pixelgrid.cpp)
			if (nwords == 2) {
				double thresh;
				if (!(ws[1] >> thresh)) Complain("invalid lomag threshold for image pixel splittings");
				imgpixel_lomag_threshold = thresh;
				// Assuming here the imgpixel_mag_threshold has been changed...
				if (image_pixel_grids != NULL) {
					for (int i=0; i < n_extended_src_redshifts; i++) {
						if (image_pixel_grids[i] != NULL) {
							image_pixel_grids[i]->delete_ray_tracing_arrays();
							image_pixel_grids[i]->setup_ray_tracing_arrays();
							if (nlens > 0) image_pixel_grids[i]->calculate_sourcepts_and_areas(true);
						}
					}
				}
			} else if (nwords==1) {
				if (mpi_id==0) cout << "high magnification threshold for splitting image pixels = " << imgpixel_lomag_threshold << endl;
			} else Complain("must specify either zero or one argument (magnification threshold for image pixel splittings)");
		}
		else if (words[0]=="imgpixel_sb_threshold")
		{
			// NOTE: currently only pixels in the primary mask are split; pixels in extended mask are NOT split (see setup_ray_tracing_arrays() in pixelgrid.cpp)
			if (nwords == 2) {
				double thresh;
				if (!(ws[1] >> thresh)) Complain("invalid number of image pixel splittings");
				imgpixel_sb_threshold = thresh;
				// Assuming here the imgpixel_sb_threshold has been changed...
				if (image_pixel_grids != NULL) {
					for (int i=0; i < n_extended_src_redshifts; i++) {
						if (image_pixel_grids[i] != NULL) {
							image_pixel_grids[i]->delete_ray_tracing_arrays();
							image_pixel_grids[i]->setup_ray_tracing_arrays();
							if (nlens > 0) image_pixel_grids[i]->calculate_sourcepts_and_areas(true);
						}
					}
				}

			} else if (nwords==1) {
				if (mpi_id==0) cout << "surface brightness threshold for splitting image pixels = " << imgpixel_sb_threshold << endl;
			} else Complain("must specify either zero or one argument (surface brightness threshold for image pixel splittings)");
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
		else if (words[0]=="multithread_perturber_deflections")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Multithread perturber lensing calculations: " << display_switch(multithread_perturber_deflections) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'multithread_perturber_deflections' command; must specify 'on' or 'off'");
				set_switch(multithread_perturber_deflections,setword);
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
					if (ray_tracing_method==Area_Overlap) cout << "Ray tracing method: overlap (pixel overlap area)" << endl;
					else if ((ray_tracing_method==Interpolate) and (!natural_neighbor_interpolation)) cout << "Ray tracing method: interpolate_3pt (linear 3-point interpolation)" << endl;
					else if ((ray_tracing_method==Interpolate) and (natural_neighbor_interpolation)) cout << "Ray tracing method: interpolate_nn (natural neighbors interpolation)" << endl;
					else cout << "Unknown ray tracing method" << endl;
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'raytrace_method' command; must specify valid ray tracing method");
				if (setword=="overlap") {
					if (source_fit_mode != Cartesian_Source) Complain("Overlap method is only available for cartesian source grid");
					ray_tracing_method = Area_Overlap;
				}
				else if (setword=="interpolate_3pt") { ray_tracing_method = Interpolate; natural_neighbor_interpolation = false; }
				else if (setword=="interpolate_nn") {
					if (source_fit_mode != Delaunay_Source) Complain("Natural neighbor interpolation is only allowed for Delaunay source grid");
					ray_tracing_method = Interpolate;
					natural_neighbor_interpolation = true;
				}
				else Complain("invalid argument to 'raytrace_method' command; must specify valid ray tracing method");
			} else Complain("invalid number of arguments; can only specify ray tracing method");
			warn("'raytrace_method' is deprecated; use 'interpolation_method' instead");
		}
		else if (words[0]=="interpolation_method") {
			if (nwords==1) {
				if (mpi_id==0) {
					if (!natural_neighbor_interpolation) cout << "Interpolation method: 3pt (linear 3-point interpolation)" << endl;
					else cout << "Interpolation method: nn (natural neighbor interpolation; for Delaunay grid only)" << endl;
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'interpolation_method' command; must specify valid interpolation method");
				if (setword=="3pt") natural_neighbor_interpolation = false;
				else if (setword=="nn") natural_neighbor_interpolation = true;
				else Complain("invalid argument to 'interpolation_method'; must specify either '3pt' or 'nn'");
			} else Complain("invalid number of arguments; can only specify one argument ('3pt' or 'nn')");
		}
		else if (words[0]=="inversion_method") {
			if (nwords==1) {
				if (mpi_id==0) {
					if (inversion_method==MUMPS) cout << "Lensing inversion method: LDL factorization (MUMPS)" << endl;
					else if (inversion_method==UMFPACK) cout << "Lensing inversion method: LU factorization (UMFPACK)" << endl;
					else if (inversion_method==CG_Method) cout << "Lensing inversion method: conjugate gradient method" << endl;
					else if (inversion_method==DENSE) cout << "Lensing inversion method: Dense Fmatrix inversion (w/ dense Lmatrix)" << endl;
					else if (inversion_method==DENSE_FMATRIX) cout << "Lensing inversion method: Dense Fmatrix inversion (w/ sparse Lmatrix)" << endl;
					else cout << "Unknown inversion method" << endl;
				}
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'inversion_method' command; must specify valid inversion method");
				if (setword=="mumps") inversion_method = MUMPS;
				else if (setword=="umfpack") inversion_method = UMFPACK;
				else if (setword=="cg") inversion_method = CG_Method;
				else if (setword=="dense") inversion_method = DENSE;
				else if (setword=="fdense") {
#ifdef USE_MKL
					inversion_method = DENSE_FMATRIX;
#else
					Complain("currently 'fdense' matrix inversion mode is only supported with MKL");
#endif
				}
				else Complain("invalid argument to 'inversion_method' command; must specify valid inversion method");
			} else Complain("invalid number of arguments; can only inversion method");
		}
		else if (words[0]=="use_nnls")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Use non-negative least squares solver: " << display_switch(use_non_negative_least_squares) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'use_nnls' command; must specify 'on' or 'off'");
#ifndef USE_EIGEN
				if (setword=="on") Complain("must compile qlens with eigen to use non-negative least squares");
#endif
				set_switch(use_non_negative_least_squares,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		//else if (words[0]=="use_fnnls")
		//{
			//if (nwords==1) {
				//if (mpi_id==0) cout << "Use non-negative least squares solver: " << display_switch(use_fnnls) << endl;
			//} else if (nwords==2) {
				//if (!(ws[1] >> setword)) Complain("invalid argument to 'use_fnnls' command; must specify 'on' or 'off'");
				//set_switch(use_fnnls,setword);
			//} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		//}
		else if (words[0]=="nnls_it")
		{
			int param;
			if (nwords == 2) {
				if (!(ws[1] >> param)) Complain("invalid maximum number of NNLS iterations");
				if (param < 0) Complain("nnls_it cannot be negative");
				max_nnls_iterations = param;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "nnls_it = " << max_nnls_iterations << endl;
			} else Complain("must specify either zero or one argument (nnls_it)");
		}
		else if (words[0]=="nnls_tol")
		{
			double tol;
			if (nwords == 2) {
				if (!(ws[1] >> tol)) Complain("invalid NNLS tolerance");
				if (tol < 0) Complain("nnls_tol cannot be negative");
				nnls_tolerance = tol;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "nnls_tol = " << nnls_tolerance << endl;
			} else Complain("must specify either zero or one argument (nnls_tol)");
		}
		else if (words[0]=="auto_srcgrid")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically determine source grid dimensions: " << display_switch(auto_sourcegrid) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_srcgrid' command; must specify 'on' or 'off'");
				set_switch(auto_sourcegrid,setword);
				if (source_fit_mode==Cartesian_Source) {
					for (int i=0; i < n_pixellated_src; i++) {
						update_pixsrc_active_parameters(i);
					}
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="auto_src_npixels")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically determine number of source grid pixels: " << display_switch(auto_srcgrid_npixels) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_src_npixels' command; must specify 'on' or 'off'");
				set_switch(auto_srcgrid_npixels,setword);
				if (source_fit_mode==Cartesian_Source) {
					for (int i=0; i < n_pixellated_src; i++) {
						update_pixsrc_active_parameters(i);
					}
				}
				update_parameter_list();
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="auto_shapelet_scale")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically determine shapelet scale: " << display_switch(auto_shapelet_scaling) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_shapelet_scale' command; must specify 'on' or 'off'");
				set_switch(auto_shapelet_scaling,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="shapelet_max_scale")
		{
			double param;
			if (nwords == 2) {
				if (!(ws[1] >> param)) Complain("invalid scale");
				shapelet_max_scale = param;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "shapelet_max_scale = " << shapelet_max_scale << endl;
			} else Complain("must specify either zero or one argument (shapelet_max_scale)");
		}
		else if (words[0]=="shapelet_window_scalefac")
		{
			double param;
			if (nwords == 2) {
				if (!(ws[1] >> param)) Complain("invalid shapelet_window_scalefac");
				shapelet_window_scaling = param;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "shapelet_window_scalefac = " << shapelet_window_scaling << endl;
			} else Complain("must specify either zero or one argument (shapelet_window_scalefac)");
		}
		else if (words[0]=="delaunay_mode")
		{
			int param;
			if (nwords == 2) {
				if (!(ws[1] >> param)) Complain("invalid scale");
				if (param < 0) Complain("delaunay_mode cannot be negative");
				delaunay_mode = param;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "delaunay_mode = " << delaunay_mode << endl;
			} else Complain("must specify either zero or one argument (delaunay_mode)");
		}
		else if (words[0]=="optimize_regparam")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically determine optimal regularization parameter: " << display_switch(optimize_regparam) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'optimize_regparam' command; must specify 'on' or 'off'");
				if ((setword=="on") and ((source_fit_mode != Shapelet_Source) and (source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source))) Complain("optimize_regparam only available in pixellated (cartesian, delaunay) or Shapelet source mode");
				set_switch(optimize_regparam,setword);
				if (n_pixellated_src > 0) {
					bool vary_regparam = false;
					for (int i=0; i < n_pixellated_src; i++) {
						srcgrids[i]->get_specific_varyflag("regparam",vary_regparam);
						if (vary_regparam) break;
					}
					if (vary_regparam) {
						if (mpi_id==0) cout << "NOTE: setting 'vary_regparam' to 'off' and updating parameters" << endl;
						for (int i=0; i < n_pixellated_src; i++) {
							update_pixellated_src_varyflag(i,"regparam",false);
						}
						update_parameter_list();
					}
				}
				if ((setword=="off") and (use_lum_weighted_regularization) and ((!use_saved_sbweights) or (!get_lumreg_from_sbweights))) {
					if (mpi_id==0) cout << "NOTE: setting 'lum_weighted_regularization' to 'off' (to keep it on, consider using sbweights via 'lumreg_from_sbweights' and 'use_saved_sbweights))" << endl;
					use_lum_weighted_regularization = false;
					for (int i=0; i < n_pixellated_src; i++) {
						if (source_fit_mode==Delaunay_Source) update_pixsrc_active_parameters(i);
					}
					update_parameter_list();
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		/*
		else if (words[0]=="optimize_regparam_lhi")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically determine optimal regularization parameter: " << display_switch(optimize_regparam_lhi) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'optimize_regparam_lhi' command; must specify 'on' or 'off'");
				if ((setword=="on") and ((source_fit_mode != Shapelet_Source) and (source_fit_mode != Cartesian_Source) and (source_fit_mode != Delaunay_Source))) Complain("optimize_regparam_lhi only available in pixellated (cartesian, delaunay) or Shapelet source mode");
				if ((setword=="on") and (!optimize_regparam)) Complain("'optimize_regparam' must also be set to 'on' before turning on optimize_regparam_lhi");
				if ((setword=="on") and (!use_lum_weighted_regularization)) Complain("optimize_regparam_lhi only available if 'lum_weighted_regularization' is set to 'on'");
				set_switch(optimize_regparam_lhi,setword);
				if ((setword=="on") and (vary_regparam_lhi)) {
					if (mpi_id==0) cout << "NOTE: setting 'vary_regparam_lhi' to 'off' and updating parameters" << endl;
					vary_regparam_lhi = false;
					update_parameter_list();
				}
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		*/
		else if (words[0]=="regparam_tol")
		{
			double param;
			if (nwords == 2) {
				if (!(ws[1] >> param)) Complain("invalid regparam_tol");
				optimize_regparam_tol = param;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "regparam_tol = " << optimize_regparam_tol << endl;
			} else Complain("must specify either zero or one argument (regparam_tol)");
		}
		else if (words[0]=="regparam_minlog")
		{
			double param;
			if (nwords == 2) {
				if (!(ws[1] >> param)) Complain("invalid regparam_minlog");
				optimize_regparam_minlog = param;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "regparam_minlog = " << optimize_regparam_minlog << endl;
			} else Complain("must specify either zero or one argument (regparam_minlog)");
		}
		else if (words[0]=="regparam_maxlog")
		{
			double param;
			if (nwords == 2) {
				if (!(ws[1] >> param)) Complain("invalid regparam_maxlog");
				optimize_regparam_maxlog = param;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "regparam_maxlog = " << optimize_regparam_maxlog << endl;
			} else Complain("must specify either zero or one argument (regparam_maxlog)");
		}
		else if ((words[0]=="regparam_max_it") or (words[0]=="regparam_maxit"))
		{
			int maxit;
			if (nwords == 2) {
				if (!(ws[1] >> maxit)) Complain("invalid regparam_max_it");
				if (maxit <= 0) Complain("number of iterations for optimizing regparam must be greater than zero");
				max_regopt_iterations = maxit;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "regparam_max_it = " << max_regopt_iterations << endl;
			} else Complain("must specify either zero or one argument (regparam_max_it)");
		}
		else if (words[0]=="auto_shapelet_center")
		{
			if (nwords==1) {
				if (mpi_id==0) cout << "Automatically determine shapelet center: " << display_switch(auto_shapelet_center) << endl;
			} else if (nwords==2) {
				if (!(ws[1] >> setword)) Complain("invalid argument to 'auto_shapelet_center' command; must specify 'on' or 'off'");
				set_switch(auto_shapelet_center,setword);
			} else Complain("invalid number of arguments; can only specify 'on' or 'off'");
		}
		else if (words[0]=="shapelet_scale_mode")
		{
			double mode;
			if (nwords == 2) {
				if (!(ws[1] >> mode)) Complain("invalid shapelet_scale_mode");
				shapelet_scale_mode = mode;
			} else if (nwords==1) {
				if (mpi_id==0) cout << "shapelet_scale_mode = " << shapelet_scale_mode << endl;
			} else Complain("must specify either zero or one argument (shapelet_scale_mode)");
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
			if (!read_from_file) {
				if (!infile->is_open()) Complain("no script file is currently loaded");
				read_from_file = true;
			}
		}
		else if (words[0]=="sleep")
		{
			double time_sec = 1.0;
			if (nwords==2) {
				if (!(ws[1] >> time_sec)) Complain("invalid sleep time (should be number multiple of seconds)");
			}
			usleep(time_sec*1e6);
		}
		else if (words[0]=="plot_egrad") {
			if (nwords < 5) Complain("need at least 4 arguments (sbprofile number, ximin, ximax, npts), plus optional filename suffix");
			if (nwords > 6) Complain("too many arguments to 'plot_egrad' (just need sbprofile number, ximin, ximax, npts, plus optional filename suffix)");
			double ximin, ximax;
			int nn, srcnum;
			string filename_suffix = "grad";
			if (!(ws[1] >> srcnum)) Complain("could not read source number");
			if (!(ws[2] >> ximin)) Complain("could not read source ximin");
			if (!(ws[3] >> ximax)) Complain("could not read source ximax");
			if (!(ws[4] >> nn)) Complain("could not read source number of xi values");
			if (nwords==6) filename_suffix = words[5];
			if (srcnum >= n_sb) Complain("source number does not exist");
			sb_list[srcnum]->plot_ellipticity_function(ximin,ximax,nn,fit_output_dir,filename_suffix);
			if (sb_list[srcnum]->fourier_gradient) sb_list[srcnum]->plot_fourier_functions(ximin,ximax,nn,fit_output_dir,filename_suffix);
		} else if (words[0]=="output_egrad_params") {
			if (nwords < 2) Complain("at least one argument required to 'output_egrad_params' (src_number), plus optional file suffix");
			if (nwords > 4) Complain("too many parameters to 'output_egrad_params'");
			double srcnum;
			if (!(ws[1] >> srcnum)) Complain("could not read source number");
			string suffix = "";
			if (nwords==3) suffix = words[2];
			if (!(output_egrad_values_and_knots(srcnum,suffix))) Complain("could not output egrad values"); // crude but hopefully works well enough
		} else if (words[0]=="output_coolest") {
			if (nwords != 2) Complain("one argument required: filename to output to <filename>.json");
			if (LensProfile::use_ellipticity_components) Complain("ellipticity components must be turned off before generating COOLEST json file");
			if (Shear::use_shear_component_params) Complain("shear components must be turned off before generating COOLEST json file");
			if (!output_coolest_files(words[1])) Complain("could not output coolest .json file");
		} else if (words[0]=="test") {
			/*
#ifdef USE_EIGEN
			Eigen::MatrixXd A(2,2);
			A(0,0) = 3;
			A(0,1) = -1;
			A(1,0) = 2.5;
			A(1,1) = A(1,0) + A(0,1);
			Eigen::MatrixXd B(2,2);
			B << 1, 5, 2, -4;
			Eigen::MatrixXd C(2,2);
			C = A + B;
			cout << C << endl;
			cout << C(0,1) << " IH" << endl;
			//VectorXd b(2);
			//b << 3, 2;
			//C = A*b;
			//cout << C << endl;
			cout << "DIAGNOAL PROD: " << endl;
			cout << C.diagonal().prod() << endl;
			Eigen::MatrixXd D(3,3);
			D(0,0) = 2;
			D(0,1) = 5;
			D(1,1) = 7;
			D(2,2) = 9;
			D(2,1) = 1e-5;
			D(0,2) = 2;
			Eigen::SparseView<Eigen::MatrixXd> sv = D.sparseView();
			cout << sv << endl;
#endif
			*/
			//if ((delaunay_srcgrids) and (delaunay_srcgrids[0])) {
				//double qs,phi_s,xavg,yavg;
				//delaunay_srcgrids[0]->find_source_moments(200,qs,phi_s,xavg,yavg);
				//double phi_s_deg = phi_s*180.0/M_PI;
				//cout << "qs=" << qs << " phi_s=" << phi_s_deg << " degrees" << endl;
			//}

			//generate_supersampled_PSF_matrix();
			/*
			int iter = 20;
			if (nwords==2) {
				if (!(ws[1] >> iter)) Complain("wtf?");
			}

			int i,j,k,n;
			int *pixptr_i, *pixptr_j;
			int npix=0,npix_in_mask;
			npix_in_mask = image_pixel_grid->ntot_cells;
			pixptr_i = image_pixel_grid->masked_pixels_i;
			pixptr_j = image_pixel_grid->masked_pixels_j;
			int nsubpix;
			for (int n=0; n < npix_in_mask; n++) {
				i = pixptr_i[n];
				j = pixptr_j[n];
				nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
				npix += nsubpix;
			}

			double *srcpts_x = new double[npix];
			double *srcpts_y = new double[npix];
			//int *ivals = new int[npix];
			//int *jvals = new int[npix];

			npix = 0;
			int subcell_i1, subcell_i2;
			for (n=0; n < npix_in_mask; n++) {
				i = pixptr_i[n];
				j = pixptr_j[n];
				if (!split_imgpixels) {
					srcpts_x[npix] = image_pixel_grid->center_sourcepts[i][j][0];
					srcpts_y[npix] = image_pixel_grid->center_sourcepts[i][j][1];
					//ivals[npix] = i;
					//jvals[npix] = j;
				} else {
					nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
					for (int k=0; k < nsubpix; k++) {
						srcpts_x[npix] = image_pixel_grid->subpixel_center_sourcepts[i][j][k][0];
						srcpts_y[npix] = image_pixel_grid->subpixel_center_sourcepts[i][j][k][1];
						//ivals[npix] = i;
						//jvals[npix] = j;
						npix++;
					}
				}
			}
			*/
			
			/*
			int nstart;
			double x,y;
			if (!(ws[1] >> nstart)) Complain("wtf?");
			if (!(ws[2] >> x)) Complain("wtf?");
			if (!(ws[3] >> y)) Complain("wtf?");
			lensvector input_pt;
			input_pt[0]=x; input_pt[1]=y;
			bool inside_triangle;
			int trinum = delaunay_srcgrid->search_grid(nstart,input_pt,inside_triangle);
			if (inside_triangle) cout << "INSIDE!" << endl;
			else cout << "NOT INSIDE!" << endl;
			Triangle *triptr = &delaunay_srcgrid->triangle[trinum];
			cout << "Triangle " << trinum << " has vertices: " << triptr->vertex[0][0] << " " << triptr->vertex[0][1] << " " << triptr->vertex[1][0] << " " << triptr->vertex[1][1] << " " << triptr->vertex[2][0] << " " << triptr->vertex[2][1] << endl;
			*/

			/*
			if (nwords != 3) Complain("need two arguments to 'test2' (nu, x)");
			cout << setprecision(16);
			double x, nu, knu;
			if (!(ws[1] >> nu)) Complain("wtf?");
			if (!(ws[2] >> x)) Complain("wtf?");
			//wtime0 = omp_get_wtime();
			knu = modified_bessel_function(x,nu);
			//wtime = omp_get_wtime()-wtime0;
			//cout << "wall time = " << wtime << endl;
			cout << "K_nu(x) = " << knu << endl;
			int nl = (int) (nu + 0.5);
			double xmu = nu-nl;
			cout << "nl=" << nl << ", xmu=" << xmu << endl;

			double gam1,gam2,gampl,gammi;
			beschb(xmu,gam1,gam2,gampl,gammi);
			double gamplcheck, gammicheck;
			gamplcheck = 1.0/Gamma(1+xmu);
			gammicheck = 1.0/Gamma(1-xmu);
			double gam2check = (gammi+gampl)/2;
			double gam1check = (gammi-gampl)/(2*xmu);

			cout << "gampl=" << gampl << " gammi=" << gammi << endl;
			cout << "gamplcheck=" << gamplcheck << " gammicheck=" << gammicheck << endl;
			cout << "gam1=" << gam1 << " gam2=" << gam2 << endl;
			cout << "gam1check=" << gam1check << " gam2check=" << gam2check << endl;
			*/
			//test_inverts();
		//} else if (words[0]=="test2") {
			//if (n_data_bands==0) Complain("image pixel data not loaded");
			//if (nwords < 8) Complain("need 6 args");
			//double xi0, xistep, qi, theta_i, xc_i, yc_i;
			//int maxit = 100;
			//int max_xi_it = 1000;
			//int emode = SB_Profile::default_ellipticity_mode;
			//bool compare_to_sbprofile = false;
			//bool polar = false;
			//bool ecomp_mode = false;
			//string output_label;
			//int sampling_mode = 2;
			//int ximax = -1; 
			//if (!(ws[1] >> xi0)) Complain("wtf?");
			//if (!(ws[2] >> xistep)) Complain("wtf?");
			//if (!(ws[3] >> qi)) Complain("wtf?");
			//if (!(ws[4] >> theta_i)) Complain("wtf?");
			//if (!(ws[5] >> xc_i)) Complain("wtf?");
			//if (!(ws[6] >> yc_i)) Complain("wtf?");
			//output_label = words[7];

			//if (nwords > 8) {
				//for (int i=8; i < nwords; i++) {
					//if ((words[i]=="-sbcomp")) compare_to_sbprofile = true;
					//else if ((words[i]=="-sector")) sampling_mode = 1;
					//else if ((words[i]=="-interp")) sampling_mode = 0;
					//else if ((words[i]=="-sbprofile")) sampling_mode = 3;
					//else if ((words[i]=="-ecomp")) ecomp_mode = true;
					//else if ((words[i]=="-polar")) polar = true;
					//else if (words[i].find("ximax=")==0) {
						//string dstr = words[i].substr(6);
						//stringstream dstream;
						//dstream << dstr;
						//if (!(dstream >> ximax)) Complain("invalid maximum elliptical radius");
					//} else if (words[i].find("max_it=")==0) {
						//string istr = words[i].substr(7);
						//stringstream istream;
						//istream << istr;
						//if (!(istream >> maxit)) Complain("invalid max number of iterations");
					//} else if (words[i].find("xi_it=")==0) {
						//string istr = words[i].substr(6);
						//stringstream istream;
						//istream << istr;
						//if (!(istream >> max_xi_it)) Complain("invalid max number of isophotes");
					//} else if (words[i].find("emode=")==0) {
						//string istr = words[i].substr(6);
						//stringstream istream;
						//istream << istr;
						//if (!(istream >> emode)) Complain("invalid ellipticity mode");
					//} else Complain("unrecognized argument");
				//}
			//}
			//if ((sampling_mode==3) and (n_sb==0)) Complain("need an sbprofile object to compare isophote fitting results to");
			//if (sampling_mode==3) compare_to_sbprofile = true;
			//SB_Profile *sbptr, *sbptr_comp;
			//if ((n_sb==0) or (!compare_to_sbprofile)) sbptr_comp = NULL;
			//else sbptr_comp = sb_list[0];

			//IsophoteData isodata;
			//if (ecomp_mode) imgpixel_data_list[0]->fit_isophote_ecomp(xi0,xistep,emode,qi,theta_i,xc_i,yc_i,maxit,isodata,polar,true,sbptr_comp,sampling_mode,max_xi_it,ximax);
			//else imgpixel_data_list[0]->fit_isophote(xi0,xistep,emode,qi,theta_i,xc_i,yc_i,maxit,isodata,polar,true,sbptr_comp,sampling_mode,max_xi_it,ximax);
			//isodata.plot_isophote_parameters(fit_output_dir,output_label);
		} else if (words[0]=="test2") {
			double scalefac = 1;
			if (nwords > 1) {
				if (!(ws[1] >> scalefac)) Complain("invalid scalefac");
			}
			//output_scaled_percentiles_from_chain(scalefac);

			//string scriptfile = fit_output_dir + "/scaled_limits.in";
			//open_script_file(scriptfile);

			output_scaled_percentiles_from_egrad_fits(0,-0.035979,-0.0102676,scalefac,5,true,true);

			//if (n_sb == 0) Complain("need a source object to spawn a lens");
			//int pmode = 0;
			//if (nwords > 1) {
				//if (!(ws[1] >> pmode)) Complain("invalid pmode");
			//}
			//spawn_lens_from_source_object(0,lens_redshift,source_redshift,pmode,true,true,0.1,10);
			/*
			dvector efunc_params;
			dvector lower_limits, upper_limits;
			boolvector varyflags;
			sb_list[0]->get_egrad_params(efunc_params);
			sb_list[0]->get_vary_flags(varyflags);
			sb_list[0]->get_limits(lower_limits,upper_limits);
			sb_list[0]->disable_ellipticity_gradient();
			sb_list[0]->enable_ellipticity_gradient(efunc_params,1);
			sb_list[0]->vary_parameters(varyflags);
			sb_list[0]->set_limits(lower_limits,upper_limits);
			*/
		} else if (words[0]=="testpt") {
			/*
			if (nwords==4) {
				double x,y;
				int pix0;
				if ((delaunay_srcgrids != NULL) and (delaunay_srcgrids[0] != NULL)) {
					ws[1] >> x;
					ws[2] >> y;
					ws[3] >> pix0;
					lensvector pt(x,y);
					bool inside_triangle;
					int tri = delaunay_srcgrids[0]->search_grid(pix0,pt,inside_triangle);
					cout << "Triangle number = " << tri << endl;
				} else Complain("delaunay grid hasn't been created");
			} else Complain("need coords, pixel0");
			*/
			//const int nn = 12;
			//double xin[nn] = { 1, 5, 9, -1.2, 12, 18, 3, 5, 8, 4, 7.17, 10.9 };
			//double yin[nn] = { 11, 2, 19, 9, 3, 10, 2, 12, 8, 7, 9.05, 16.9 };
			const int nn = 16;
			double xin[nn] = { 0.0000, 1.0000, 2.0000, 3.0000, 0.0001, 1.0001, 2.0001, 3.0001, 0.0002, 1.0002, 2.0002, 3.0002, 0.0003, 1.0003, 2.0003, 3.0003 };
			double yin[nn] = { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0 };
			DelaunaySourceGrid delaunay_grid(this);
			delaunay_grid.create_pixel_grid(xin,yin,nn);
			for (int i=0; i < nn; i++) delaunay_grid.surface_brightness[i] = SQR(xin[i])+SQR(yin[i]);
			delaunay_grid.plot_surface_brightness("test",100,true,false,false);
			delaunay_grid.plot_voronoi_grid("test");
			// For natural neighbor interpolation, there is a failure mode when points near the border of the grid are nearly colinear, resulting in sliver triangles. (Demonstrate in lines above.) Weird circles appear in the interpolated SB. Fix this at some point!!
			//lensvector pt(3.4,8.8);
			//bool inside;
			//int trinum = delaunay_grid.search_grid(0,pt,inside);
			//lensvector vertex[3];
			//vertex[0] = delaunay_grid.triangle[trinum].vertex[0];
			//vertex[1] = delaunay_grid.triangle[trinum].vertex[1];
			//vertex[2] = delaunay_grid.triangle[trinum].vertex[2];
			//cout << "Found point in triangle " << trinum << endl;
			//cout << "Vertices:" << endl;
			//for (int i=0; i < 3; i++) {
				//cout << vertex[i][0] << " " << vertex[i][1] << endl;
			//}
			//cout << endl;

			/*
         Delaunay *delaunay_triangles = new Delaunay(xin, yin, nn);
			delaunay_triangles->Process();
			int triN = delaunay_triangles->TriNum();
			delaunay_triangles->FindNeighbors();
			Triangle *triangle = new Triangle[triN];
			delaunay_triangles->store_triangles(triangle);
			cout << "Triangles:" << endl;
			int i=0, j=0;
			int k;
			ofstream testout("testdel.dat");
			double ptx, pty;
			for (i=0; i < triN; i++) {
				testout << triangle[i].vertex[0][0] << " " << triangle[i].vertex[0][1] << endl;
				testout << triangle[i].vertex[1][0] << " " << triangle[i].vertex[1][1] << endl;
				testout << triangle[i].vertex[2][0] << " " << triangle[i].vertex[2][1] << endl;
				testout << triangle[i].vertex[0][0] << " " << triangle[i].vertex[0][1] << endl;
				testout << endl;
			}
			ofstream testneb("testneb.dat");
			for (j=0; j < 3; j++) {
				k = triangle[3].neighbor_index[j];
				if (k >= 0) {
					testneb << "#triangle neighbor " << k << ":" << endl;
					testneb << triangle[k].vertex[0][0] << " " << triangle[k].vertex[0][1] << endl;
					testneb << triangle[k].vertex[1][0] << " " << triangle[k].vertex[1][1] << endl;
					testneb << triangle[k].vertex[2][0] << " " << triangle[k].vertex[2][1] << endl;
					testneb << triangle[k].vertex[0][0] << " " << triangle[k].vertex[0][1] << endl;
					testneb << endl;
				} else {
					testneb << "#triangle has no neighbor " << k << endl;
				}
			}
			*/

			/*
			for (i=0; i < triN; i++) {
				testout << "#triangle " << i << endl;
				k = tris[j++];
				testout << xin[k] << " " << yin[k] << endl;
				ptx = xin[k];
				pty = yin[k];
				k = tris[j++];
				testout << xin[k] << " " << yin[k] << endl;
				k = tris[j++];
				testout << xin[k] << " " << yin[k] << endl;
				testout << ptx << " " << pty << endl;
				testout << endl;
			}
			*/
			//cout << "Printing neighbors..." << endl;
			//triangles->print_triangle_neighbors(18);
			//delete delaunay_triangles;
			//delete[] triangle;
		} else if (words[0]=="load_isofit") {
			bool fit_sbprofile = false;
			bool no_optimize = false; // if true, do not switch to downhill simplex mode when varying SB profile params during PSF iterations
			bool nested_sampling_all_profiles = false;
			bool optimize_knots = false;
			bool optimize_knots_before_nest = true;
			bool skip_nested_sampling = false;
			bool include_xcyc = false;
			bool include_a34 = false;
			bool include_a56 = false;
			bool input_profile_errors = true; // if on, read profile errors from input file; if off, generate errors from a specified error fraction
			double errfrac = 0;
			int srcnum = 0;
			int n_sbfit_livepts = 600;
			if (nwords < 2) Complain("need 1 arg (filename)");
			string input_filename = words[1];
			bool plot_isofit_profiles = false;
			string plot_label;

			if (nwords > 2) {
				for (int i=2; i < nwords; i++) {
					if (words[i].find("src=")==0) {
						string sstr = words[i].substr(4);
						stringstream sstream;
						sstream << sstr;
						if (!(sstream >> srcnum)) Complain("invalid source number");
						if (n_sb <= srcnum) Complain("specified source number does not exist in list of sources");
					} else if (words[i].find("N=")==0) {
						string nstr = words[i].substr(2);
						stringstream nstream;
						nstream << nstr;
						if (!(nstream >> n_sbfit_livepts)) Complain("invalid number of live points");
					} else if (words[i].find("-plot:")==0) {
						string nstr = words[i].substr(6);
						stringstream nstream;
						nstream << nstr;
						if (!(nstream >> plot_label)) Complain("invalid plot label");
						plot_isofit_profiles = true;
					}
					else if ((words[i]=="-xcyc")) include_xcyc = true;
					else if ((words[i]=="-a34")) include_a34 = true;
					else if ((words[i]=="-a56")) include_a56 = true;
					else if ((words[i]=="-noopt")) no_optimize = true;
					else if ((words[i]=="-nest_all")) nested_sampling_all_profiles = true;
					else if ((words[i]=="-skipnest")) skip_nested_sampling = true;
					else if ((words[i]=="-nest_all_optknots")) {
						nested_sampling_all_profiles = true;
						optimize_knots_before_nest = true;
					}
					else if ((words[i]=="-fitsb")) fit_sbprofile = true;
					else if ((words[i]=="-optknots")) optimize_knots = true;
					else if (words[i].find("errfrac=")==0) {
						string estr = words[i].substr(8);
						stringstream estream;
						estream << estr;
						if (!(estream >> errfrac)) Complain("invalid maximum elliptical radius");
						if ((errfrac <= 0) or (errfrac > 1)) Complain("invalid error fraction; must be a number greater than 0 and less than 1");
						input_profile_errors = false;
					}
				}
			}
			SB_Profile *sbptr;
			if (fit_sbprofile) sbptr = sb_list[srcnum];
			else sbptr = NULL;

			ifstream isodata_input;
			isodata_input.open(input_filename.c_str());
			if (!isodata_input.is_open()) Complain("could not open isophote data file");
			int n_xivals = 0;
			static const int n_characters = 5000;
			char dataline[n_characters];
			string dum;
			while (!isodata_input.eof()) {
				isodata_input.getline(dataline,n_characters);
				if (dataline[0]=='#') continue;
				istringstream checkempty(dataline);
				if (!(checkempty >> dum)) continue;
				n_xivals++;
			}
			isodata_input.close();
			if (n_xivals==0) Complain("could not read any lines from isophote data file");
			isodata_input.open(input_filename.c_str());
			if (!isodata_input.is_open()) Complain("could not open isophote data file");
				
			IsophoteData isodata(n_xivals);
			if (input_profile_errors) {
				if (!isodata.load_profiles(isodata_input,include_xcyc,include_a34,include_a56)) Complain("could not load isophote fitting profiles");;
			} else {
				if (!isodata.load_profiles_noerrs(isodata_input,errfrac,include_xcyc,include_a34,include_a56)) Complain("could not load isophote fitting profiles");
			}
			int sbprofile_fit_iter = 0; // keeps track of how many sbprofile fits have been done, so it does nested sampling on iter=0 and simplex afterwards
			if (skip_nested_sampling) sbprofile_fit_iter++;
			if (nested_sampling_all_profiles) sbprofile_fit_iter = -1;
			if ((fit_sbprofile) and (sbptr != NULL)) {
				if (!sbptr->fit_sbprofile_data(isodata,sbprofile_fit_iter,n_sbfit_livepts,mpi_np,mpi_id,fit_output_dir)) Complain("sbprofile fit failed");
				if ((sbptr != NULL) and (sbptr->ellipticity_gradient)) {
					if (mpi_id==0) cout << "Fitting q profile from isophote fit..." << endl;
					if (!sbptr->fit_egrad_profile_data(isodata,0,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
					if (mpi_id==0) cout << "Fitting theta profile from isophote fit..." << endl;
					if (!sbptr->fit_egrad_profile_data(isodata,1,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
					if (sbptr->fourier_gradient) {
						if (include_a34) {
							if (mpi_id==0) cout << "Fitting Fourier m=3 profiles from isophote fit:" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,4,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (!sbptr->fit_egrad_profile_data(isodata,5,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (mpi_id==0) cout << "Fitting Fourier m=4 profiles from isophote fit:" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,6,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (!sbptr->fit_egrad_profile_data(isodata,7,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						}
						if (include_a56) {
							if (mpi_id==0) cout << "Fitting Fourier m=5 profiles from isophote fit:" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,8,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (!sbptr->fit_egrad_profile_data(isodata,9,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (mpi_id==0) cout << "Fitting Fourier m=6 profiles from isophote fit:" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,10,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (!sbptr->fit_egrad_profile_data(isodata,11,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						}
					}
				}
			}
			if (plot_isofit_profiles) {
				cout << "Plotting isofit profiles to " << fit_output_dir << "/..._" << plot_label << endl;
				isodata.plot_isophote_parameters(fit_output_dir,plot_label);
			}
			update_anchored_parameters_and_redshift_data(); // For any lens that is anchored to the foreground source
		} else if (words[0]=="isofit") {
			if (n_data_bands==0) Complain("image pixel data not loaded");
			if (nwords < 8) Complain("need 6 args");
			double xi0, xistep, qi, theta_i, xc_i, yc_i;
			int maxit = 100;
			int max_xi_it = 1000;
			int emode = SB_Profile::default_ellipticity_mode;
			bool compare_to_sbprofile = false;
			bool polar = false;
			bool include_fourier_m3_mode = false; // note, the m4 mode must be included so it is not an option here
			string output_label;
			int sampling_mode = 2;
			double ximax = -1; 
			int psf_iterations = 0; // if 0, no PSF correction is made
			double sbgrmax = 1.0;
			double sbgrtrans = 0.3;
			double npts_frac = 0.5;
			int n_higher_harmonics = 2; // should be either 2, 3, or 4
			bool fix_center = false;
			bool avg_center = false;
			double posterior_output_scalefac = 20;
			double fmode_posterior_output_scalefac = 5; // Fourier modes have higher uncertainties already, so scaling should be lower
			bool fit_sbprofile = false; // note that during PSF-correction iterations, sbprofile will be fit regardless
			bool no_optimize = false; // if true, do not switch to downhill simplex mode when varying SB profile params during PSF iterations
			bool nested_sampling_on_final_iter = false;
			bool optimize_knots = false;
			bool optimize_knots_before_nest = true;
			bool skip_nested_sampling = false;
			int n_sbfit_livepts = 600;
			int srcnum = 0;
			if (!(ws[1] >> xi0)) Complain("wtf?");
			if (!(ws[2] >> xistep)) Complain("wtf?");
			if (!(ws[3] >> qi)) Complain("wtf?");
			if (!(ws[4] >> theta_i)) Complain("wtf?");
			if (!(ws[5] >> xc_i)) Complain("wtf?");
			if (!(ws[6] >> yc_i)) Complain("wtf?");
			output_label = words[7];

			if (nwords > 8) {
				for (int i=8; i < nwords; i++) {
					if (words[i].find("src=")==0) {
						string lstr = words[i].substr(4);
						stringstream lstream;
						lstream << lstr;
						if (!(lstream >> srcnum)) Complain("invalid source number");
						if (n_sb <= srcnum) Complain("specified source number does not exist in list of sources");
					} else if (words[i].find("N=")==0) {
						string nstr = words[i].substr(2);
						stringstream nstream;
						nstream << nstr;
						if (!(nstream >> n_sbfit_livepts)) Complain("invalid number of live points");
					}
					else if ((words[i]=="-sbcomp")) compare_to_sbprofile = true;
					else if ((words[i]=="-sector")) sampling_mode = 1;
					else if ((words[i]=="-interp")) sampling_mode = 0;
					else if ((words[i]=="-sbprofile")) sampling_mode = 3;
					else if ((words[i]=="-noopt")) no_optimize = true;
					else if ((words[i]=="-nest_final")) nested_sampling_on_final_iter = true;
					else if ((words[i]=="-skipnest")) skip_nested_sampling = true;
					else if ((words[i]=="-nest_final_optknots")) {
						nested_sampling_on_final_iter = true;
						optimize_knots_before_nest = true;
					}
					else if ((words[i]=="-fitsb")) fit_sbprofile = true;
					else if ((words[i]=="-optknots")) optimize_knots = true;
					else if ((words[i]=="-polar")) polar = true;
					else if ((words[i]=="-fix_center")) fix_center = true;
					else if ((words[i]=="-avg_center")) {
						avg_center = true;
						fix_center = false;
					}
					else if (words[i].find("ximax=")==0) {
						string dstr = words[i].substr(6);
						stringstream dstream;
						dstream << dstr;
						if (!(dstream >> ximax)) Complain("invalid maximum elliptical radius");
					} else if (words[i].find("psf_it=")==0) {
						string istr = words[i].substr(7);
						stringstream istream;
						istream << istr;
						if (!(istream >> psf_iterations)) Complain("invalid number of PSF iterations");
					} else if (words[i].find("fmodes=")==0) {
						string istr = words[i].substr(7);
						stringstream istream;
						istream << istr;
						if (!(istream >> n_higher_harmonics)) Complain("invalid number of higher harmonics");
						if ((n_higher_harmonics < 2) or (n_higher_harmonics > 4)) Complain("number of higher harmonics should be either 2, 3, or 4");
					} else if (words[i].find("max_it=")==0) {
						string istr = words[i].substr(7);
						stringstream istream;
						istream << istr;
						if (!(istream >> maxit)) Complain("invalid max number of iterations");
					} else if (words[i].find("xi_it=")==0) {
						string istr = words[i].substr(6);
						stringstream istream;
						istream << istr;
						if (!(istream >> max_xi_it)) Complain("invalid max number of isophotes");
					} else if (words[i].find("emode=")==0) {
						string istr = words[i].substr(6);
						stringstream istream;
						istream << istr;
						if (!(istream >> emode)) Complain("invalid ellipticity mode");
					} else if (words[i].find("npts_frac=")==0) {
						string dstr = words[i].substr(10);
						stringstream dstream;
						dstream << dstr;
						if (!(dstream >> npts_frac)) Complain("invalid minimum fraction of accepted points");
					} else if (words[i].find("sbgrmax=")==0) {
						string dstr = words[i].substr(8);
						stringstream dstream;
						dstream << dstr;
						if (!(dstream >> sbgrmax)) Complain("invalid maximum relative rms SB gradient");
					} else if (words[i].find("sbgr_trans=")==0) {
						string dstr = words[i].substr(11);
						stringstream dstream;
						dstream << dstr;
						if (!(dstream >> sbgrtrans)) Complain("invalid relative rms SB gradient transition threshold");
					} else if (words[i].find("scalepost=")==0) {
						string dstr = words[i].substr(10);
						stringstream dstream;
						dstream << dstr;
						if (!(dstream >> posterior_output_scalefac)) Complain("invalid posterior output scale factor");
					} else Complain("unrecognized argument");
				}
			}
			if ((psf_iterations > 0) and (n_sb==0)) Complain("need an sbprofile object to generate psf correction");
			if ((sampling_mode==3) and (n_sb==0)) Complain("need an sbprofile object to compare isophote fitting results to");
			if (sampling_mode==3) compare_to_sbprofile = true;
			SB_Profile *sbptr, *sbptr_comp;
			if ((n_sb==0) or (!compare_to_sbprofile)) sbptr_comp = NULL;
			else sbptr_comp = sb_list[0];
			if ((psf_iterations > 0) or (fit_sbprofile)) sbptr = sb_list[0];
			else sbptr = NULL;
			if (psf_iterations > 0) {
				double check;
				//if ((!sbptr->get_specific_parameter("A_4",check)) and (!sbptr->get_specific_parameter("A4_i",check))) Complain("sbprofile object must have Fourier mode m=4 defined for psf iterations"); 
				if (!sbptr->fourier_mode_exists(4)) Complain("sbprofile object must have Fourier mode m=4 defined for psf iterations"); 
			}
			if (sbptr != NULL) {
				double check;
				if (sbptr->fourier_mode_exists(3)) include_fourier_m3_mode = true; 
				if ((n_higher_harmonics >= 3) and (!sbptr->fourier_mode_exists(5))) Complain("sbprofile object must have Fourier mode m=5 if fmodes > 2 is specified"); 
				if ((n_higher_harmonics >= 4) and (!sbptr->fourier_mode_exists(6))) Complain("sbprofile object must have Fourier mode m=6 if fmodes > 2 is specified"); 
				if (!SB_Profile::fourier_sb_perturbation) Complain("fourier_sbmode must be set to 'on' to fit sbprofile object to isofit profiles");
			}
			bool verbal = true;
			if (mpi_id > 0) verbal = false;

			ofstream fitout((fit_output_dir + "/" + output_label + "_isofit.dat").c_str());
			imgpixel_data_list[0]->set_isofit_output_stream(&fitout);
			IsophoteData isodata;
			if (imgpixel_data_list[0]->fit_isophote(xi0,xistep,emode,qi,theta_i,xc_i,yc_i,maxit,isodata,polar,verbal,sbptr_comp,sampling_mode,n_higher_harmonics,fix_center,max_xi_it,ximax,sbgrmax,npts_frac,sbgrtrans) == false) Complain("isofit failed");

			double xc_avg = xc_i, yc_avg = yc_i; // these will be updated if avg_center is set to true
			if (!fix_center) {
				double* skip_outlier = new double[isodata.n_xivals];
				for (int i=0; i < isodata.n_xivals; i++) {
					skip_outlier[i] = false;
				}
				bool at_least_one_outlier;
				double xc_invsqr, xc_err;
				double yc_invsqr, yc_err;
				int it=0;
				do {
					xc_avg=0;
					xc_invsqr=0;
					yc_avg=0;
					yc_invsqr=0;
					for (int i=0; i < isodata.n_xivals; i++) {
						if (!skip_outlier[i]) {
							xc_avg += isodata.xcvals[i]/SQR(isodata.xc_errs[i]);
							xc_invsqr += 1.0/SQR(isodata.xc_errs[i]);
							yc_avg += isodata.ycvals[i]/SQR(isodata.yc_errs[i]);
							yc_invsqr += 1.0/SQR(isodata.yc_errs[i]);
						}
					}
					xc_avg /= xc_invsqr;
					yc_avg /= yc_invsqr;
					if (mpi_id==0) {
						cout << "Iteration " << it << ":" << endl;
						cout << "avg xc: " << xc_avg << endl;
						cout << "avg yc: " << yc_avg << endl;
					}
					at_least_one_outlier = false;
					for (int i=0; i < isodata.n_xivals; i++) {
						if (!skip_outlier[i]) {
							if (abs(isodata.xcvals[i]-xc_avg) > 4*isodata.xc_errs[i]) {
								skip_outlier[i] = true;
								at_least_one_outlier = true;
								warn("at least one xc value (xc=%g, xc_err=%g at xi=%g) differs from the mean by more than 4*sigma_err; this point will be clipped and xc will be recalculated",isodata.xcvals[i],isodata.xc_errs[i],isodata.xivals[i]);
							}
							else if (abs(isodata.ycvals[i]-yc_avg) > 4*isodata.yc_errs[i]) {
								skip_outlier[i] = true;
								at_least_one_outlier = true;
								warn("at least one yc value (yc=%g, yc_err=%g at xi=%g) differs from the mean by more than 4*sigma_err; this point will be clipped and yc will be recalculated",isodata.ycvals[i],isodata.yc_errs[i],isodata.xivals[i]);
							}
						}
					}
					it++;
				} while (at_least_one_outlier);
				xc_err = 3*sqrt(1.0/xc_invsqr);
				yc_err = 3*sqrt(1.0/yc_invsqr);
				if (avg_center) {
					if (sbptr != NULL) {
						sbptr->update_specific_parameter("xc",xc_avg);
						sbptr->update_specific_parameter("yc",yc_avg);
					}
					fix_center = true;
					xc_i = xc_avg;
					yc_i = yc_avg;
					if (imgpixel_data_list[0]->fit_isophote(xi0,xistep,emode,qi,theta_i,xc_i,yc_i,maxit,isodata,polar,verbal,sbptr_comp,sampling_mode,n_higher_harmonics,fix_center,max_xi_it,ximax,sbgrmax,npts_frac,sbgrtrans) == false) Complain("isofit failed");
					if (mpi_id==0) {
						cout << "Centroid estimate:" << endl;
						cout << "avg xc: " << xc_avg << " +/- " << xc_err << " # 3-sigma error" << endl;
						cout << "avg yc: " << yc_avg << " +/- " << yc_err << " # 3-sigma error" << endl;
					}
				}
				delete[] skip_outlier;
			}

			double* skip = new double[isodata.n_xivals];
			for (int i=0; i < isodata.n_xivals; i++) {
				skip[i] = false; // if any isophotes don't have fitted values (i.e. are set to NAN), we'll set skip[i] = true for that isophote so we don't try to use it
				if (isodata.A4vals[i]*0.0 != 0.0) { skip[i] = true; continue; }
				if (isodata.sb_avg_vals[i]*0.0 != 0.0) { skip[i] = true; continue; }
			}

			IsophoteData isodata0(isodata);
			// the remaining code is to apply the PSF correction
			int sbprofile_fit_iter = 0; // keeps track of how many sbprofile fits have been done, so it does nested sampling on iter=0 and simplex afterwards
			if (skip_nested_sampling) sbprofile_fit_iter++;
			bool include_ellipticity_gradient = false;
			bool include_fourier_gradient = false;

			if (psf_iterations > 0) {
				int n_xivals = isodata.n_xivals;
				//double iso_qvals[n_xivals];
				//double iso_A4vals[n_xivals];
				//double iso_thetavals[n_xivals];
				//double iso_sbvals[n_xivals];
				int i,j,k=0;
				double **qcorr = new double*[n_xivals];
				double **qvals = new double*[n_xivals];
				double qmax = -1e30, qmin = 1e30;
				for (i=0; i < n_xivals; i++) {
					if (isodata.qvals[i] > qmax) qmax = isodata.qvals[i];
					if (isodata.qvals[i] < qmin) qmin = isodata.qvals[i];
				}
				qmin -= 0.1;
				double qq, qstep = 0.05;
				int nn_q = (int) ((qmax-qmin)/qstep) + 2; // this will ensure the qstep won't be larger than 0.05
				qstep = (qmax-qmin)/(nn_q-1);

				for (i=0; i < n_xivals; i++) {
					qcorr[i] = new double[nn_q];
					qvals[i] = new double[nn_q];
				}

				include_ellipticity_gradient = sbptr->ellipticity_gradient;
				include_fourier_gradient = sbptr->fourier_gradient;

				double A4update;
				for (i=0; i < n_xivals; i++) {
					if (isodata.A4vals[i]*0.0 != 0.0) { skip[i] = true; continue; }
					if (isodata.sb_avg_vals[i]*0.0 != 0.0) { skip[i] = true; continue; }
					//iso_qvals[i] = isodata.qvals[i];
					//iso_A4vals[i] = isodata.A4vals[i];
					//iso_thetavals[i] = isodata.thetavals[i];
					//iso_sbvals[i] = isodata.sb_avg_vals[i];
				}

				//dvector efunc_params;
				//dvector lower_limits, upper_limits;
				//boolvector varyflags;
				do {
					if (mpi_id==0) cout << "Fitting SB profile from isophote fit:" << endl;
					if (!sbptr->fit_sbprofile_data(isodata,sbprofile_fit_iter,n_sbfit_livepts,mpi_np,mpi_id,fit_output_dir)) Complain("sbprofile fit failed");
					sbptr->print_parameters();
					if (include_ellipticity_gradient) {
						if (mpi_id==0) cout << endl << "Fitting q profile from isophote fit:";
						if ((sbprofile_fit_iter==-1) and (optimize_knots_before_nest)) {
							cout << " (will optimize knots first)" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,0,1,n_sbfit_livepts,true,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						} else {
							if ((sbprofile_fit_iter==1) and (optimize_knots)) cout << " (will optimize knots first)";
							cout << endl;
						}
						if (!sbptr->fit_egrad_profile_data(isodata,0,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit
						if (mpi_id==0) cout << endl << "Fitting theta profile from isophote fit:";
						if ((sbprofile_fit_iter==-1) and (optimize_knots_before_nest)) {
							cout << " (will optimize knots first)" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,1,1,n_sbfit_livepts,true,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						} else {
							if ((sbprofile_fit_iter==1) and (optimize_knots)) cout << " (will optimize knots first)";
							cout << endl;
						}
						if (!sbptr->fit_egrad_profile_data(isodata,1,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit

						if (include_fourier_gradient) {
							if (include_fourier_m3_mode) {
								if (mpi_id==0) cout << endl << "Fitting Fourier m=3 cosine profile from isophote fit:" << endl;
								if (!sbptr->fit_egrad_profile_data(isodata,4,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
								if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit
								if (mpi_id==0) cout << endl << "Fitting Fourier m=3 sine profile from isophote fit:" << endl;
								if (!sbptr->fit_egrad_profile_data(isodata,5,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
								if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit
							}
							if (mpi_id==0) cout << endl << "Fitting Fourier m=4 cosine profile from isophote fit:" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,6,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit
							if (mpi_id==0) cout << endl << "Fitting Fourier m=4 sine profile from isophote fit:" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,7,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit
							if (n_higher_harmonics >= 3) {
								if ((mpi_id==0) and (sbprofile_fit_iter==0)) cout << endl << "Fitting Fourier m=5 cosine profile from isophote fit:" << endl;
								if (!sbptr->fit_egrad_profile_data(isodata,8,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
								if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit
								if ((mpi_id==0) and (sbprofile_fit_iter==0)) cout << endl << "Fitting Fourier m=5 sine profile from isophote fit:" << endl;
								if (!sbptr->fit_egrad_profile_data(isodata,9,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
								if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit
							}
							if (n_higher_harmonics >= 4) {
								if ((mpi_id==0) and (sbprofile_fit_iter==0)) cout << endl << "Fitting Fourier m=6 cosine profile from isophote fit:" << endl;
								if (!sbptr->fit_egrad_profile_data(isodata,10,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
								if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit
								if ((mpi_id==0) and (sbprofile_fit_iter==0)) cout << endl << "Fitting Fourier m=6 sine profile from isophote fit:" << endl;
								if (!sbptr->fit_egrad_profile_data(isodata,11,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
								if (sbprofile_fit_iter==0) sbptr->print_parameters(); // just to see if parameters are in the ballpark after first fit
							}
						}

						//sbptr->get_egrad_params(efunc_params);
						//sbptr->get_vary_flags(varyflags);
						//sbptr->get_limits(lower_limits,upper_limits);
					}
					//for (i=0; i < n_xivals; i++) {
						//if (!skip[i]) {
							//iso_qvals[i] = isodata.qvals[i];
							//iso_A4vals[i] = isodata.A4vals[i];
							//iso_thetavals[i] = isodata.thetavals[i];
						//}
					//}
					//if (k==1) sbptr->update_specific_parameter("rc",0); // this simulates refining the SB profile after the initial correction
					double A4_avg=0, A4_invsqr=0;
					double B4_avg=0, B4_invsqr=0;
					double A3_avg=0, A3_invsqr=0;
					double B3_avg=0, B3_invsqr=0;
					xc_avg = 0;
					yc_avg = 0;
					double xc_invsqr=0;
					double yc_invsqr=0;

					double A5_avg=0, A5_invsqr=0;
					double B5_avg=0, B5_invsqr=0;
					double A6_avg=0, A6_invsqr=0;
					double B6_avg=0, B6_invsqr=0;

					for (i=0; i < n_xivals; i++) {
						if (!skip[i]) {
							A4_avg += isodata.A4vals[i]/SQR(isodata.A4_errs[i]);
							A4_invsqr += 1.0/SQR(isodata.A4_errs[i]);
							B4_avg += isodata.B4vals[i]/SQR(isodata.B4_errs[i]);
							B4_invsqr += 1.0/SQR(isodata.B4_errs[i]);
							A3_avg += isodata.A3vals[i]/SQR(isodata.A3_errs[i]);
							A3_invsqr += 1.0/SQR(isodata.A3_errs[i]);
							B3_avg += isodata.B3vals[i]/SQR(isodata.B3_errs[i]);
							B3_invsqr += 1.0/SQR(isodata.B3_errs[i]);
							if (n_higher_harmonics >= 3) {
								A5_avg += isodata.A5vals[i]/SQR(isodata.A5_errs[i]);
								A5_invsqr += 1.0/SQR(isodata.A5_errs[i]);
								B5_avg += isodata.B5vals[i]/SQR(isodata.B5_errs[i]);
								B5_invsqr += 1.0/SQR(isodata.B5_errs[i]);
							}
							if (n_higher_harmonics >= 4) {
								A6_avg += isodata.A6vals[i]/SQR(isodata.A6_errs[i]);
								A6_invsqr += 1.0/SQR(isodata.A6_errs[i]);
								B6_avg += isodata.B6vals[i]/SQR(isodata.B6_errs[i]);
								B6_invsqr += 1.0/SQR(isodata.B6_errs[i]);
							}

							if (!fix_center) {
								xc_avg += isodata.xcvals[i]/SQR(isodata.xc_errs[i]);
								xc_invsqr += 1.0/SQR(isodata.xc_errs[i]);
								yc_avg += isodata.ycvals[i]/SQR(isodata.yc_errs[i]);
								yc_invsqr += 1.0/SQR(isodata.yc_errs[i]);
							}
						}
					}
					A4_avg /= A4_invsqr;
					if (k==0) A4update = A4_avg;
					B4_avg /= B4_invsqr;
					A3_avg /= A3_invsqr;
					B3_avg /= B3_invsqr;
					if (n_higher_harmonics >= 3) {
						A5_avg /= A5_invsqr;
						B5_avg /= B5_invsqr;
					}
					if (n_higher_harmonics >= 4) {
						A6_avg /= A6_invsqr;
						B6_avg /= B6_invsqr;
					}
					if (!fix_center) {
						xc_avg /= xc_invsqr;
						yc_avg /= yc_invsqr;
					}
					if (mpi_id==0) {
						cout << "avg A4: " << A4_avg << endl;
						cout << "avg B4: " << B4_avg << endl;
						cout << "avg A3: " << A3_avg << endl;
						cout << "avg B3: " << B3_avg << endl;
						if (n_higher_harmonics >= 3) {
							cout << "avg A5: " << A5_avg << endl;
							cout << "avg B5: " << B5_avg << endl;
						}
						if (n_higher_harmonics >= 4) {
							cout << "avg A6: " << A6_avg << endl;
							cout << "avg B6: " << B6_avg << endl;
						}
						if (!fix_center) {
							cout << "avg xc: " << xc_avg << endl;
							cout << "avg yc: " << yc_avg << endl;
						}
					}
					if (!include_fourier_gradient) {
						sbptr->update_specific_parameter("A_4",A4update);
						sbptr->update_specific_parameter("B_4",B4_avg);
					}
					if ((!fix_center) and (k==0)) {
						// We only need to update the center once, because there won't be any PSF correction to the center (right?)
						sbptr->update_specific_parameter("xc",xc_avg);
						sbptr->update_specific_parameter("yc",yc_avg);
					}

					IsophoteData isodata_mock[nn_q];
					//ImagePixelData mockdata[nn_q];

					if (n_sb==0) Complain("No surface brightness profiles have been defined"); 
					double *q0vals = new double[nn_q];

					/*
					if (include_ellipticity_gradient) {
						sbptr->disable_ellipticity_gradient();
					}

					double q0, t0;
					sbptr->get_specific_parameter("q",q0); // to keep things simple
					sbptr->get_specific_parameter("theta",t0); // to keep things simple

					for (i=0, qq=qmin; i < nn_q; i++, qq += qstep) {
						q0vals[i] = qq;
						sbptr->update_specific_parameter("q",qq);
						if (image_pixel_grid != NULL) delete image_pixel_grid;
						image_pixel_grid = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,(*imgpixel_data_list[0]),true);
						image_pixel_grid->find_surface_brightness(true); // the 'true' means it will only plot the foreground SB profile
						vectorize_image_pixel_surface_brightness(); // note that in this case, the image pixel vector also contains the foreground
						PSF_convolution_pixel_vector(image_surface_brightness,false);
						store_image_pixel_surface_brightness();
						clear_pixel_matrices();
						if (mpi_id==0) {
							cout << "Making mock isophotes with q=" << qq << endl;
							//cout << "******************************************************************************" << endl;
						}
						mockdata[i].set_lens(this);
						mockdata[i].load_from_image_grid(image_pixel_grid,background_pixel_noise);
						mockdata[i].copy_mask(imgpixel_data_list[0]);
						//mockdata[i].plot_surface_brightness("data_pixel",true,false);
						//run_plotter("datapixel");
						mockdata[i].fit_isophote(xi0,xistep,emode,qq,t0,xc_i,yc_i,maxit,isodata_mock[i],polar,false,NULL,sampling_mode,n_higher_harmonics,fix_center,max_xi_it,ximax,sbgrmax,npts_frac,sbgrtrans);
					}
					Spline qcorr_spline[n_xivals];
					for (i=0; i < n_xivals; i++) {
						if (!skip[i]) {
							for (j=0; j < nn_q; j++) {
								if (isodata_mock[j].qvals[i]*0.0 != 0.0) {
									warn("BLARGH");
									skip[i] = true;
									break;
								}
								qcorr[i][j] = isodata_mock[j].qvals[i] - q0vals[j];
								qvals[i][j] = isodata_mock[j].qvals[i];
							}
							if (!skip[i]) qcorr_spline[i].input(qvals[i],qcorr[i],nn_q);
						}
					}

					//for (i=0; i < n_xivals; i++) {
						//if (!skip[i]) {
							//if (k==psf_iterations-1) isodata.qvals[i] -= qcorr_spline[i].splint(isodata.qvals[i]);
						//}
					//}

					sbptr->update_specific_parameter("q",q0); // set it back to original value
					if (include_ellipticity_gradient) {
						sbptr->enable_ellipticity_gradient(efunc_params,egrad_mode);
						sbptr->vary_parameters(varyflags);
						sbptr->set_limits(lower_limits,upper_limits);
					}
					*/

					// Now we'll create a mock image with the ellipticity/PA gradient, and find the PSF correction to theta/q. Because we've already applied
					// a correction to q, this is not particularly sensitive to getting the epsilon/PA gradient parameters exactly right; this is important
					// since we won't know the exact values in real life, only the fits we've done to parameters that haven't had the following correction yet.
					IsophoteData isodata_mock_t;
					ImagePixelData mockdata_t;

					if (n_extended_src_redshifts==0) {
						load_pixel_grid_from_data(0);
					}
					if (image_pixel_grids==NULL) Complain("image pixel grids could not be generated from given data and masks");
					if (image_pixel_grids[0] != NULL) delete image_pixel_grids[0];
					image_pixel_grids[0] = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,(*imgpixel_data_list[0]),true);
					image_pixel_grids[0]->find_surface_brightness(true); // the 'true' means it will only plot the foreground SB profile
					vectorize_image_pixel_surface_brightness(true); // note that in this case, the image pixel vector also contains the foreground
					PSF_convolution_pixel_vector(0,false);
					store_image_pixel_surface_brightness(0);
					clear_pixel_matrices();
					mockdata_t.set_lens(this);
					mockdata_t.load_from_image_grid(image_pixel_grids[0]);
					mockdata_t.copy_mask(imgpixel_data_list[0]);
					//mockdata[i].plot_surface_brightness("data_pixel",true,false);
					mockdata_t.fit_isophote(xi0,xistep,emode,qi,theta_i,xc_i,yc_i,maxit,isodata_mock_t,polar,false,NULL,sampling_mode,n_higher_harmonics,fix_center,max_xi_it,ximax,sbgrmax,npts_frac,sbgrtrans);

					double eps, th, A3, B3, A4, B4, sbcorr_t, tcorr_t, qcorr_t, A3corr_t, B3corr_t, A4corr_t, B4corr_t;
					double A5, B5, A6, B6, A5corr_t, B5corr_t, A6corr_t, B6corr_t;
					for (i=0; i < n_xivals; i++) {
						if (!skip[i]) {
							if (include_ellipticity_gradient) {
								sbptr->ellipticity_function(isodata.xivals[i],eps,th);
								qq = sqrt(1-eps);
							} else {
								double q0, t0;
								sbptr->get_specific_parameter("q",q0); // to keep things simple
								sbptr->get_specific_parameter("theta",t0); // to keep things simple
								qq = q0;
								th = t0;
							}
							tcorr_t = isodata_mock_t.thetavals[i] - th;
							isodata.thetavals[i] = isodata0.thetavals[i] - tcorr_t;
							if (include_fourier_gradient) {
								sbptr->fourier_mode_function(isodata.xivals[i],4,A4,B4);
								if (include_fourier_m3_mode) sbptr->fourier_mode_function(isodata.xivals[i],3,A3,B3);
								if (n_higher_harmonics >= 3) sbptr->fourier_mode_function(isodata.xivals[i],5,A5,B5);
								if (n_higher_harmonics >= 4) sbptr->fourier_mode_function(isodata.xivals[i],6,A6,B6);
							} else {
								A4 = A4update;
								B4 = B4_avg;
								if (include_fourier_m3_mode) {
									A3 = A3_avg;
									B3 = B3_avg;
								}
								if (n_higher_harmonics >= 3) {
									A5 = A5_avg;
									B5 = B5_avg;
								}
								if (n_higher_harmonics >= 4) {
									A6 = A6_avg;
									B6 = B6_avg;
								}
							}

							//qcorr_t = isodata_mock_t.qvals[i] - qcorr_spline[i].splint(isodata_mock_t.qvals[i]) - qq;
							//isodata.qvals[i] = isodata0.qvals[i] - qcorr_spline[i].splint(isodata.qvals[i]) - qcorr_t;
							qcorr_t = isodata_mock_t.qvals[i] - qq;
							isodata.qvals[i] = isodata0.qvals[i] - qcorr_t;
							//double qcorr_t2 = isodata_mock_t.qvals[i] - qq;
							//double qcheck = isodata0.qvals[i] - qcorr_t2;
							//cout << "newq: " << isodata.qvals[i] << " " << qcheck << " " << qcorr_t << " " << qcorr_t2 << endl;

							sbcorr_t = isodata_mock_t.sb_avg_vals[i] - sbptr->sb_rsq(SQR(isodata.xivals[i]));
							isodata.sb_avg_vals[i] = isodata0.sb_avg_vals[i] - sbcorr_t;

							A4corr_t = isodata_mock_t.A4vals[i] - A4;
							isodata.A4vals[i] = isodata0.A4vals[i] - A4corr_t;
							B4corr_t = isodata_mock_t.B4vals[i] - B4;
							isodata.B4vals[i] = isodata0.B4vals[i] - B4corr_t;

							if (include_fourier_m3_mode) {
								A3corr_t = isodata_mock_t.A3vals[i] - A3;
								isodata.A3vals[i] = isodata0.A3vals[i] - A3corr_t;
								B3corr_t = isodata_mock_t.B3vals[i] - B3;
								isodata.B3vals[i] = isodata0.B3vals[i] - B3corr_t;
							}
							if (n_higher_harmonics >= 3) {
								A5corr_t = isodata_mock_t.A5vals[i] - A5;
								isodata.A5vals[i] = isodata0.A5vals[i] - A5corr_t;
								B5corr_t = isodata_mock_t.B5vals[i] - B5;
								isodata.B5vals[i] = isodata0.B5vals[i] - B5corr_t;
							}
							if (n_higher_harmonics >= 4) {
								A6corr_t = isodata_mock_t.A6vals[i] - A6;
								isodata.A6vals[i] = isodata0.A6vals[i] - A6corr_t;
								B6corr_t = isodata_mock_t.B6vals[i] - B6;
								isodata.B6vals[i] = isodata0.B6vals[i] - B6corr_t;
							}
						}
					}
					if (mpi_id==0) cout << "original avg A4=" << A4_avg << endl;
					A4_avg=0;
					A4_invsqr=0;
					for (i=0; i < n_xivals; i++) {
						if (!skip[i]) {
							A4_avg += isodata.A4vals[i]/SQR(isodata.A4_errs[i]);
							A4_invsqr += 1.0/SQR(isodata.A4_errs[i]);
						}

					}
					A4_avg /= A4_invsqr;
					B4_avg /= B4_invsqr;
					if (mpi_id==0) cout << "corrected avg A4=" << A4_avg << " (iteration " << k << ")" << endl;
					A4update = A4_avg;
					delete[] q0vals;
					if (!no_optimize) sbprofile_fit_iter++;
					if ((nested_sampling_on_final_iter) and (k>=psf_iterations-2)) sbprofile_fit_iter = -1;
				} while (++k < psf_iterations);

				//for (i=0; i < n_xivals; i++) {
					//if (!skip[i]) {
						//isodata.qvals[i] = iso_qvals[i];
						//isodata.A4vals[i] = iso_A4vals[i];
						//isodata.thetavals[i] = iso_thetavals[i];
					//}
				//}
				/*
				for (i=0; i < n_xivals; i++) {
					delete[] qcorr[i];
					delete[] qvals[i];
				}
				delete[] qcorr;
				delete[] qvals;
				*/
			}
			delete[] skip;

			if (((fit_sbprofile) or (psf_iterations > 0)) and (sbptr != NULL) and (!nested_sampling_on_final_iter)) { // if we finished with nested sampling, we don't need to redo the fits here
				if (!sbptr->fit_sbprofile_data(isodata,sbprofile_fit_iter,n_sbfit_livepts,mpi_np,mpi_id,fit_output_dir)) Complain("sbprofile fit failed");
				if ((sbptr != NULL) and (sbptr->ellipticity_gradient)) {
					if (mpi_id==0) cout << "Fitting q profile from isophote fit..." << endl;
					if (!sbptr->fit_egrad_profile_data(isodata,0,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
					if (mpi_id==0) cout << "Fitting theta profile from isophote fit..." << endl;
					if (!sbptr->fit_egrad_profile_data(isodata,1,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
					if (sbptr->fourier_gradient) {
						if (include_fourier_m3_mode) {
							if (mpi_id==0) cout << "Fitting Fourier m=3 profiles from isophote fit:" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,4,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (!sbptr->fit_egrad_profile_data(isodata,5,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						}
						if (mpi_id==0) cout << "Fitting Fourier m=4 profiles from isophote fit:" << endl;
						if (!sbptr->fit_egrad_profile_data(isodata,6,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						if (!sbptr->fit_egrad_profile_data(isodata,7,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						if (n_higher_harmonics >= 3) {
							if (mpi_id==0) cout << "Fitting Fourier m=5 profiles from isophote fit:" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,8,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (!sbptr->fit_egrad_profile_data(isodata,9,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						}
						if (n_higher_harmonics >= 4) {
							if (mpi_id==0) cout << "Fitting Fourier m=6 profiles from isophote fit:" << endl;
							if (!sbptr->fit_egrad_profile_data(isodata,10,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
							if (!sbptr->fit_egrad_profile_data(isodata,11,sbprofile_fit_iter,n_sbfit_livepts,optimize_knots,mpi_np,mpi_id,fit_output_dir)) Complain("egrad profile fit failed");
						}

					}
				}
			}

			if (mpi_id==0) cout << "Plotting..." << endl;
			isodata.plot_isophote_parameters(fit_output_dir,output_label);
			if (nested_sampling_on_final_iter) {
				bool include_m3_fgrad = false;
				double xcavg, ycavg;
				sbptr->get_specific_parameter("xc",xcavg);		
				sbptr->get_specific_parameter("yc",ycavg);		
				if ((include_fourier_gradient) and (include_fourier_m3_mode)) include_m3_fgrad = true;
				output_scaled_percentiles_from_egrad_fits(srcnum,xcavg,ycavg,posterior_output_scalefac,fmode_posterior_output_scalefac,include_m3_fgrad,include_fourier_gradient);
				if (mpi_id==0) {
					int egmode = sbptr->get_egrad_mode();
					if (egmode==0) {
						cout << "Knot values and B-spline coefficient posterior limits (scaled by " << posterior_output_scalefac << ") have been output to a file" << endl;
					} else {
						cout << "egrad parameter posterior limits (scaled by " << posterior_output_scalefac << ") have been output to a file" << endl;
					}
				}
			}
			update_anchored_parameters_and_redshift_data(); // For any lens that is anchored to the foreground source

			//find_bestfit_smooth_model(2);
			//fit_los_despali();
			//test_lens_functions();
			//double chisq0;
			//calculate_chisq0_from_srcgrid(chisq0, true);

			//plot_weak_lensing_shear_data(true);
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
			//make_perturber_population(0.04444,2100,0.1,0.6);
			//plot_perturber_deflection_vs_area();
			//double rmax,menc;
			//calculate_critical_curve_deformation_radius(nlens-1,true,rmax,menc);
			//calculate_critical_curve_deformation_radius_numerical(nlens-1);
			//plot_shear_field(-10,10,50,-10,10,50);
			//plot_shear_field(1e-3,2,300,1e-3,2,300);
			//lens_list[0]->tryupdate();
		/*
		// The following are some specialized qlens functions that probably won't get used again
		} else if (words[0]=="plotmc") {
			// You should have two extra arguments that specify logm_min and logm_max, and maybe even a third arg for number of points
			// Implement this later!
			if (nwords==1) Complain("must specify which subhalo lens number to plot mc relation for");
			int lensnumber = 0;
			double logmmin, logmmax;
			if (!(ws[1] >> lensnumber)) Complain("invalid lens number");
			if (!(ws[2] >> logmmin)) Complain("invalid min log(m200)");
			if (!(ws[3] >> logmmax)) Complain("invalid max log(m200)");
			string filename;
			if (nwords == 5) filename = words[4];
			else filename = "mcplot.dat";
			plot_mc_curve(lensnumber,logmmin,logmmax,filename);
			//double xmin,xmax,ymin,ymax;
			//xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			//ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			//plot_shear_field(xmin,xmax,100,ymin,ymax,100);
		}
		else if (words[0]=="plotmz") {
			// You should have two extra arguments that specify logm_min and logm_max, and maybe even a third arg for number of points
			// Implement this later!
			if (nwords<4) Complain("must specify which subhalo lens number, zmin, zmax");
			int lensnumber = 0;
			double zmin, zmax;
			double yslope1=0, yslope2=0;
			bool keep_dr_const = false;
			if (!(ws[1] >> lensnumber)) Complain("invalid lens number");
			if (!(ws[2] >> zmin)) Complain("invalid zmin");
			if (!(ws[3] >> zmax)) Complain("invalid zmax");
			string filename;
			if (nwords >= 5) filename = words[4];
			else filename = fit_output_filename;
			if ((nwords>=6) and (!(ws[5] >> yslope1))) Complain("invalid yslope1");
			if ((nwords>=7) and (!(ws[6] >> yslope2))) Complain("invalid yslope2");
			if ((nwords>=8) and (!(ws[7] >> keep_dr_const))) Complain("invalid yslope");
			if (keep_dr_const) cout << "Keeping relative rpert constant..." << endl;
			plot_mz_curve(lensnumber,zmin,zmax,yslope1,yslope2,keep_dr_const,filename);
			//double xmin,xmax,ymin,ymax;
			//xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			//ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			//plot_shear_field(xmin,xmax,100,ymin,ymax,100);
		}
		else if (words[0]=="plotmz2") {
			// You should have two extra arguments that specify logm_min and logm_max, and maybe even a third arg for number of points
			// Implement this later!
			if (nwords<5) Complain("must specify subhalo lens number, zmin, zmax and zstep");
			int lensnumber = 0;
			double zmin, zmax, zstep;
			if (!(ws[1] >> lensnumber)) Complain("invalid lens number");
			if (!(ws[2] >> zmin)) Complain("invalid zmin");
			if (!(ws[3] >> zmax)) Complain("invalid zmax");
			if (!(ws[4] >> zstep)) Complain("invalid zmax");
			string filename;
			if (nwords >= 6) filename = words[5];
			else filename = fit_output_filename;
			plot_mz_bestfit(lensnumber,zmin,zmax,zstep,filename);
			//double xmin,xmax,ymin,ymax;
			//xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			//ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			//plot_shear_field(xmin,xmax,100,ymin,ymax,100);
		}
		else if (words[0]=="nfw_newc") {
			double newc;
			if (nwords > 1) { ws[1] >> newc; }
			else die("you fucked up");
			find_equiv_mvir(newc);
			//double xmin,xmax,ymin,ymax;
			//xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			//ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			//plot_shear_field(xmin,xmax,100,ymin,ymax,100);
		*/
		} else if (words[0]=="update_paramlist") {
			update_parameter_list();
		} else if (words[0]=="echo") {
			bool add_endl = true;
			for (int i=1; i < nwords; i++) {
				if (words[i]=="-noendl") {
					remove_word(i);
					add_endl = false;
				}
			}
			for (int i=1; i < nwords; i++) cout << words[i] << " ";
			if (add_endl) cout << endl;
		}
		else if (mpi_id==0) Complain("command not recognized");
	}
#ifdef USE_READLINE
	free(buffer);
#endif
}

bool QLens::read_command(bool show_prompt)
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
#ifdef USE_READLINE
			add_history(line.c_str());
#endif
			lines.push_back(line);
			if ((verbal_mode) and (mpi_id==0)) cout << line << endl;
		}
	} else {
#ifndef USE_READLINE
		warn("cannot run qlens in interactive mode without GNU readline");
		return false;
#else
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
#endif
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

bool QLens::check_vary_z()
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

bool QLens::read_egrad_params(const bool vary_pars, const int egrad_mode, dvector& efunc_params, int& nparams_to_vary, boolvector& varyflags, const int default_nparams, const double xc, const double yc, ParamAnchor* parameter_anchors, int& parameter_anchor_i, int& n_bspline_coefs, dvector& knots, double& ximin, double& ximax, double &xiref, bool &linear_xivals, bool& enter_params_and_varyflags, bool& enter_knots)
{
	// NOTE: the parameter enter_params_and_varyflags will be plugged in as 'enter_prior_limits', because that's how it will be used for after calling this function
	bool anchor_params = false;
	int n_efunc_params;
	enter_params_and_varyflags = true;
	linear_xivals = false;
	int n_unique_knots = 0;
	enter_knots = false;
	xiref = 1.5; // default value (only relevant for egrad_mode=2 right now)
	if (egrad_mode==0) {
		enter_params_and_varyflags = false;
		if (mpi_id==0) cout << "Enter n_bspline_coefficients, ximin, ximax, paramflag:" << endl;
		if (read_command(false)==false) return false;
		if ((nwords<=2) and (words[0]=="read_isofit_data")) {
			string isofit_data_dir = fit_output_dir;
			if (nwords==2) isofit_data_dir = words[1];
			string scriptfile = isofit_data_dir + "/isofit_knots_limits.in";
			if (!open_script_file(scriptfile)) return false;
			if (read_command(false)==false) return false;
		}
		for (int i=nwords-1; i > 1; i--) {
			if (words[i]=="-linear_knots") {
				linear_xivals = true;
				remove_word(i);
			} else if (words[i]=="-enter_knots") {
				enter_knots = true;
				remove_word(i);
			}
		}
		if ((nwords != 3) and (nwords != 4)) return false;
		bool invalid_params = false;
		if (!(ws[0] >> n_bspline_coefs)) invalid_params = true;
		if (!(ws[1] >> ximin)) invalid_params = true;
		if (!(ws[2] >> ximax)) invalid_params = true;
		if ((nwords==4) and (!(ws[3] >> enter_params_and_varyflags))) invalid_params = true;
		if (invalid_params==true) return false;
		if (n_bspline_coefs==0) return false;
		n_efunc_params = 2*n_bspline_coefs;
		n_unique_knots = n_bspline_coefs - 2; // assuming order 3 spline
		knots.input(2*n_unique_knots);
		if (!enter_knots) {
			for (int i=0; i < n_unique_knots; i++) knots[i] = -1e30;
		} else {
			if (mpi_id==0) cout << "Knot values for q (not counting multiplicities):" << endl;
			if (read_command(false)==false) return false;
			if (nwords != n_unique_knots) {
				warn("wrong number of knots given (%i); expecting %i",nwords,n_unique_knots);
				return false;
			}
			bool invalid_params = false;
			for (int i=0; i < n_unique_knots; i++) {
				if (!(ws[i] >> knots[i])) invalid_params = true;
			}
			if (invalid_params==true) return false;
			if (mpi_id==0) cout << "Knot values for theta (not counting multiplicities):" << endl;
			if (read_command(false)==false) return false;
			if (nwords != n_unique_knots) {
				warn("wrong number of knots given (%i); expecting %i",nwords,n_unique_knots);
				return false;
			}
			invalid_params = false;
			for (int i=0; i < n_unique_knots; i++) {
				if (!(ws[i] >> knots[n_unique_knots+i])) invalid_params = true;
			}
			if (invalid_params==true) return false;
		}
	} else if (egrad_mode==1) {
		n_efunc_params = 8;
	} else if (egrad_mode==2) {
		n_efunc_params = 8;
		//if (mpi_id==0) cout << "Enter xi_ref:" << endl;
		//if (read_command(false)==false) return false;
		//if (nwords != 3) return false;
		//if (!(ws[0] >> xiref)) return false;
	} else {
		warn("only egrad_mode=0, 1, or 2 currently supported");
		return false;
	}
	efunc_params.input(n_efunc_params+2); // there will be two extra parameters (xc, yc); those will be taken care of outside of this function
	if (enter_params_and_varyflags) {
		if (mpi_id==0) cout << "Ellipticity gradient parameters:" << endl;
		if (read_command(false)==false) return false;
		for (int i=nwords-1; i > 1; i--) {
			if (words[i]=="-anchor_scale_params") {
				anchor_params = true;
				remove_word(i);
			}
		}
		for (int i=nwords-1; i > 1; i--) {
			int pos;
			if ((pos = words[i].find("xiref=")) != string::npos) {
				string xirefstring = words[i].substr(pos+6);
				stringstream xirefstr;
				xirefstr << xirefstring;
				if (!(xirefstr >> xiref)) return false;
				remove_word(i);
			}
		}	
		if (nwords != n_efunc_params) {
			warn("wrong number of parameters given (%i); expecting %i params",nwords,n_efunc_params);
			return false;
		}
		bool invalid_params = false;
		for (int i=0; i < n_efunc_params; i++) {
			if (!(ws[i] >> efunc_params[i])) invalid_params = true;
		}
		if (invalid_params==true) return false;
		efunc_params[n_efunc_params] = xc;
		efunc_params[n_efunc_params+1] = yc;
	} else {
		for (int i=0; i < n_efunc_params; i++) {
			efunc_params[i] = -1e30; // just to make it obvious they weren't initialized
		}
	}

	nparams_to_vary += n_efunc_params - 2; // since q and theta no longer exist as parameters that can be varied
	if (vary_pars) {
		int npar_old = varyflags.size();
		int npar_new = npar_old + n_efunc_params - 2;
		//cout << "OLD VARYFLAGS(np=" << npar_old << "): " << endl;
		//for (int j=0; j < npar_old; j++) {
			//cout << varyflags[j] << " ";
		//}
		//cout << endl;
		varyflags.resize(npar_new);
		//cout << "VARYFLAGS after resizing (np=" << npar_new << "): " << endl;
		//for (int j=0; j < npar_new; j++) {
			//cout << "FLAG " << j << ": " << varyflags[j] << endl;
		//}
		//cout << endl;

		// The following are the center coords and redshift flags, which are always the last three parameters
		int insertion_point = default_nparams - 2; // Note that default_nparams does not include the redshift parameter, so this should work regardless of whether z is included as a parameter
		//cout << "INSERTION POINT: " << insertion_point << endl;
		//cout << "insertion point: " << insertion_point << " npar_old=" << npar_old << endl;
		for (int i=npar_old-1; i >= insertion_point; i--) varyflags[i+n_efunc_params-2] = varyflags[i];

		insertion_point -= 2; // now we're going to overwrite q, theta in favor of the new egrad param's

		boolvector egrad_param_anchored(n_efunc_params);
		for (int i=0; i < n_efunc_params; i++) egrad_param_anchored[i] = false;
		if (anchor_params) {
			if (egrad_mode==0) {
				warn("cannot anchor egrad parameters in B-spline mode");
				return false;
			} else if ((egrad_mode==1) or (egrad_mode==2)) {
				// anchor xi0_theta to xi0_q
				parameter_anchors[parameter_anchor_i].anchor_param = true;
				parameter_anchors[parameter_anchor_i].paramnum = insertion_point+6;
				parameter_anchors[parameter_anchor_i].anchor_paramnum = insertion_point+2;
				parameter_anchor_i++;
				egrad_param_anchored[6] = true;
				// anchor dxi_theta to dxi_q
				parameter_anchors[parameter_anchor_i].anchor_param = true;
				parameter_anchors[parameter_anchor_i].paramnum = insertion_point+7;
				parameter_anchors[parameter_anchor_i].anchor_paramnum = insertion_point+3;
				parameter_anchor_i++;
				egrad_param_anchored[7] = true;
			}
		}

		if (enter_params_and_varyflags) {
			if (mpi_id==0) cout << "Ellipticity gradient vary flags:" << endl;
			if (read_command(false)==false) return false;
			if ((nwords==1) and (words[0]=="none")) {
				for (int i=0; i < n_efunc_params; i++) {
					varyflags[insertion_point+i] = false;
				}
			} else if ((nwords==1) and (words[0]=="all")) {
				for (int i=0; i < n_efunc_params; i++) {
					varyflags[insertion_point+i] = true;
				}
			} else {
				if (nwords != n_efunc_params) return false;

				bool invalid_params = false;
				//cout << "NEW VARYFLAGS: " << endl;
				for (int i=0; i < n_efunc_params; i++) {
					if (!(ws[i] >> varyflags[insertion_point+i])) invalid_params = true;
					//if (varyflags[insertion_point+i]) nparams_to_vary++;
					if ((egrad_param_anchored[i]) and (varyflags[insertion_point+i])) {
						warn("cannot vary egrad parameter if it has been anchored to another egrad parameter");
						return false;
					}
				}
				if (invalid_params==true) return false;
			}
		} else {
			for (int i=0; i < n_efunc_params; i++) {
				varyflags[insertion_point+i] = true;
				//nparams_to_vary++;
			}
		}
		
		//cout << "efunc_params:" << endl;
		//for (i=0; i < n_efunc_params+2; i++) {
			//cout << efunc_params[i] << " ";
		//}
		//cout << endl;
		//cout << "npar_new: " << npar_new << ", npar_old: " << npar_old << endl;
		//for (int j=0; j < npar_new; j++) {
			//cout << varyflags[j] << " ";
		//}
		//cout << endl;
		//if (nparams_to_vary != npar_new) die("OH DEAR %i %i %i %i",nparams_to_vary,npar_new,npar_old,n_efunc_params);
		//nparams_to_vary = npar_new;
	}

	return true;
}

bool QLens::read_fgrad_params(const bool vary_pars, const int egrad_mode, const int n_fmodes, const vector<int> fourier_mvals, dvector& fgrad_params, int& nparams_to_vary, boolvector& varyflags, const int default_nparams, ParamAnchor* parameter_anchors, int& parameter_anchor_i, int n_bspline_coefs, dvector& knots, const bool enter_params_and_varyflags, const bool enter_knots)
{
	bool anchor_params = false;
	int n_fgrad_params;
	int n_unique_knots = 0;
	if (egrad_mode==0) {
		n_fgrad_params = 2*n_fmodes*n_bspline_coefs;
		n_unique_knots = n_bspline_coefs - 2; // assuming order 3 spline
		knots.input(2*n_fmodes*n_unique_knots);
		if (!enter_knots) {
			for (int i=0; i < n_unique_knots; i++) knots[i] = -1e30;
		} else {
			for (int j=0; j < n_fmodes; j++) {
				if (mpi_id==0) cout << "Knot values for A_" << fourier_mvals[j] << " (not counting multiplicities):" << endl;
				if (read_command(false)==false) return false;
				if (nwords != n_unique_knots) {
					warn("wrong number of knots given (%i); expecting %i",nwords,n_unique_knots);
					return false;
				}
				bool invalid_params = false;
				for (int i=0; i < n_unique_knots; i++) {
					if (!(ws[i] >> knots[2*j*n_unique_knots+i])) invalid_params = true;
				}
				if (invalid_params==true) return false;
				if (mpi_id==0) cout << "Knot values for B_" << fourier_mvals[j] << " (not counting multiplicities):" << endl;
				if (read_command(false)==false) return false;
				if (nwords != n_unique_knots) {
					warn("wrong number of knots given (%i); expecting %i",nwords,n_unique_knots);
					return false;
				}
				invalid_params = false;
				for (int i=0; i < n_unique_knots; i++) {
					if (!(ws[i] >> knots[(2*j+1)*n_unique_knots+i])) invalid_params = true;
				}
			}
		}
	} else if ((egrad_mode==1) or (egrad_mode==2)) {
		n_fgrad_params = n_fmodes*8; // this is for egrad_mode=0
	} else {
		warn("only egrad_mode=0, 1 or 2 currently supported");
		return false;
	}
	fgrad_params.input(n_fgrad_params); // there will be two extra parameters (xc, yc); those will be taken care of outside of this function
	if (enter_params_and_varyflags) {
		if (mpi_id==0) cout << "Fourier gradient parameters:" << endl;
		if (read_command(false)==false) return false;
		for (int i=nwords-1; i > 1; i--) {
			if (words[i]=="-anchor_scale_params") {
				anchor_params = true;
				remove_word(i);
			}
		}
		if (nwords != n_fgrad_params) {
			warn("wrong number of Fourier gradient parameters given (%i required)",n_fgrad_params);
			return false;
		}
		bool invalid_params = false;
		for (int i=0; i < n_fgrad_params; i++) {
			if (!(ws[i] >> fgrad_params[i])) invalid_params = true;
		}
		if (invalid_params==true) return false;
	} else {
		for (int i=0; i < n_fgrad_params; i++) fgrad_params[i] = 0;
	}

	nparams_to_vary += n_fgrad_params - 2*n_fmodes;
	if (vary_pars) {
		int npar_old = varyflags.size();
		int npar_new = npar_old + n_fgrad_params - n_fmodes*2;
		//cout << "OLD VARYFLAGS(np=" << npar_old << "): " << endl;
		//for (int j=0; j < npar_old; j++) {
			//cout << varyflags[j] << " ";
		//}
		//cout << endl;
		varyflags.resize(npar_new);
		// Note: default_nparams here will include the egrad parameters, but we are still allowing the possibility for extra parameters beyond the
		// Fourier mode parameters (besides the redshift)
		//cout << "default_np=" << default_nparams << " npar_old-2=" << (npar_old-2) << endl;
		int n_amps = n_fmodes*2;
		//int insertion_point = (include_redshift_param) ? default_nparams + n_amps - 1 : default_nparams + n_amps;
		int insertion_point = default_nparams + n_amps;
		for (int i=npar_old-1; i >= insertion_point; i--) varyflags[i+n_fgrad_params-n_amps] = varyflags[i];

		insertion_point -= n_amps; // now we're going to overwrite the current Fourier entries

		boolvector fgrad_param_anchored(n_fgrad_params);
		for (int i=0; i < n_fgrad_params; i++) fgrad_param_anchored[i] = false;
		if (anchor_params) {
			if (egrad_mode==0) {
				warn("cannot anchor egrad parameters in B-spline mode");
				return false;
			} else if ((egrad_mode==1) or (egrad_mode==2)) {
				int fpi;
				for(int i=1; i < n_amps; i++) {
					fpi = 4*i;
					// anchor xi0 params
					parameter_anchors[parameter_anchor_i].anchor_param = true;
					parameter_anchors[parameter_anchor_i].paramnum = insertion_point+2+fpi;
					parameter_anchors[parameter_anchor_i].anchor_paramnum = insertion_point+2;
					parameter_anchor_i++;
					fgrad_param_anchored[2+fpi] = true;
					// anchor dxi params
					parameter_anchors[parameter_anchor_i].anchor_param = true;
					parameter_anchors[parameter_anchor_i].paramnum = insertion_point+3+fpi;
					parameter_anchors[parameter_anchor_i].anchor_paramnum = insertion_point+3;
					parameter_anchor_i++;
					fgrad_param_anchored[3+fpi] = true;
				}
			}
		}
		if (enter_params_and_varyflags) {
			if (mpi_id==0) cout << "Fourier gradient vary flags:" << endl;
			if (read_command(false)==false) return false;
			if (nwords != n_fgrad_params) return false;

			bool invalid_params = false;

			int i;
			for (i=0; i < n_fgrad_params; i++) {
				if (!(ws[i] >> varyflags[insertion_point+i])) invalid_params = true;
				//if (varyflags[insertion_point+i]) nparams_to_vary++;
				if ((fgrad_param_anchored[i]) and (varyflags[insertion_point+i])) {
					warn("cannot vary egrad parameter if it has been anchored to another egrad parameter");
					return false;
				}
			}
			if (invalid_params==true) return false;
		} else {
			for (int i=0; i < n_fgrad_params; i++) {
				varyflags[insertion_point+i] = true;
				//nparams_to_vary++;
			}
		}

		//cout << "NEW VARYFLAGS(np=" << npar_new << "): " << endl;
		//for (int j=0; j < npar_new; j++) {
			//cout << varyflags[j] << " ";
		//}
		//cout << endl;

		//nparams_to_vary = npar_new;
		//cout << "fgrad_params:" << endl;
		//for (i=0; i < n_fgrad_params; i++) {
			//cout << fgrad_params[i] << " ";
		//}
		//cout << endl;
		//cout << "npar_new: " << npar_new << ", npar_old: " << npar_old << endl;
		//cout << "npar_vflg: " << varyflags.size() << endl;
		//for (int j=0; j < npar_new; j++) {
			//cout << varyflags[j] << " ";
		//}
		//cout << endl;
	}

	return true;
}

void QLens::remove_equal_sign(void)
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

void QLens::remove_word(int n_remove)
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

void QLens::extract_word_starts_with(const char initial_character, int starting_word, int ending_word, string& extracted_word)
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

void QLens::set_plot_title(int starting_word, string& title)
{
	title = "";
	if (starting_word >= nwords) return;
	for (int i=starting_word; i < nwords-1; i++) title += words[i] + " ";
	title += words[nwords-1];
	for (int i=nwords-1; i >= starting_word; i--) remove_word(i);
	int pos;
	while ((pos = title.find('"')) != string::npos) title.erase(pos,1);
	while ((pos = title.find('\'')) != string::npos) title.erase(pos,1);
}



bool QLens::extract_word_starts_with(const char initial_character, int starting_word, int ending_word, vector<string>& extracted_words)
{
	bool extracted = false;
	if (starting_word >= nwords) return false;
	int end;
	if (ending_word >= nwords) end = nwords-1;
	else end = ending_word;
	//vector<int> remove_i;
	for (int i=end; i >= starting_word; i--) {
		// we make sure the next character after '-' is a letter, not a number
		if ((words[i][0] == initial_character) and (isalpha(words[i][1]))) { if (!extracted) extracted = true; extracted_words.push_back(words[i]); remove_word(i); }
	}
	//for (int i=remove_i.size()-1; i >= 0; i--) remove_word(remove_i[i]);
	return extracted;
}

void QLens::run_plotter(string plotcommand, string extra_command)
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
		string command = "plotlens " + plotcommand + " " + extra_command + " ps=" + psstring + " pt=" + ptstring + " lw=" + lwstring + " fs=" + fsstring;
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

void QLens::run_plotter_file(string plotcommand, string filename, string range, string extra_command, string extra_command2)
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
		string command = "plotlens " + plotcommand + " file=" + filename + " " + range + " " + extra_command + " " + extra_command2 + " ps=" + psstring + " pt=" + ptstring + " lw=" + lwstring + " fs=" + fsstring;
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

void QLens::run_plotter_range(string plotcommand, string range, string extra_command, string extra_command2)
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
		string command = "plotlens " + plotcommand + " " + range + " " + extra_command + " " + extra_command2 + " ps=" + psstring + " pt=" + ptstring + " lw=" + lwstring + " fs=" + fsstring;
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

bool QLens::open_script_file(const string filename)
{
	if (infile->is_open()) {
		if (n_infiles==10) die("cannot open more than 10 files at once");
		infile++;
	}
	infile->open(filename.c_str());
	if (!infile->is_open()) infile->open(("../inifile/" + filename).c_str());
	if (!infile->is_open()) {
		// Now we look for any directories in the PATH variable that have 'qlens' in the name
		size_t pos = 0;
		size_t pos2 = 0;
		int i, ndirs = 1;
		char *env = getenv("PATH");
		string envstring(env);
		while ((pos = envstring.find(':')) != string::npos) {
			ndirs++;
			envstring.replace(pos,1," ");
		}
		istringstream dirstream(envstring);
		string dirstring[ndirs];
		ndirs=0;
		while (dirstream >> dirstring[ndirs]) ndirs++; // it's possible ndirs will be zero, which is why we recount it here
		for (i=0; i < ndirs; i++) {
			pos=pos2=0;
			if (((pos = dirstring[i].find("qlens")) != string::npos) or ((pos2 = dirstring[i].find("kappa")) != string::npos)) {
				infile->open((dirstring[i] + "/" + filename).c_str());
				if (!infile->is_open()) infile->open((dirstring[i] + "/../inifile/" + filename).c_str());
				if (infile->is_open()) break;
			}
		}
	}
	if (infile->is_open()) {
		if ((n_infiles > 0) and (!read_from_file)) paused_while_reading_file = true;
		read_from_file = true;
		n_infiles++;
	}
	else {
		if (n_infiles > 0) infile--;
		return false;
	}
	return true;
}


void QLens::run_mkdist(bool copy_post_files, string posts_dirname, const int nbins_1d, const int nbins_2d, bool copy_subplot_only, bool resampled_posts, bool no2dposts, bool nohists)
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

			if (!no2dposts) {
				string hist2d_str = fit_output_dir + "/" + filename + ".hist2d_params";
				ofstream hist2dfile(hist2d_str.c_str());
				int nparams_tot = param_settings->nparams + param_settings->n_dparams;
				for (int i=0; i < nparams_tot; i++) {
					string pname;
					bool pflag = param_settings->hist2d_param_flag(i,pname);
					hist2dfile << pname << " ";
					if (pflag) hist2dfile << "1";
					else hist2dfile << "0";
					hist2dfile << endl;
				}
				hist2dfile.close();
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
			if (!nohists) {
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
			}
			if (copy_post_files) {
				command += "if [ ! -d ../" + posts_dirname + " ]; then mkdir ../" + posts_dirname + "; fi; ";
				if (!nohists) {
					if ((!copy_subplot_only) or (!make_subplot)) command += "cp " + filename + ".pdf ../" + posts_dirname + "; ";
					if (!no2dposts) command += "cp " + filename + "_tri.pdf ../" + posts_dirname + "; ";
					if ((!no2dposts) and (make_subplot)) command += "cp " + filename + "_subtri.pdf ../" + posts_dirname + "; ";
				}
				if (!nohists) command += "cp *.chain_info ../" + posts_dirname + "; ";
			}
			command += "cd ..";
			system(command.c_str());
		}
	}
#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}


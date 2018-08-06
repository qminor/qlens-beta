# QLens (beta version)
QLens is a software package for modeling and simulating strong gravitational lens systems. Both pixel image modeling (using pixellated source reconstruction) and point image modeling (with option to include fluxes, time delays, and multiple sources at different redshifts) are supported. QLens includes 14 different analytic lens models to use for model fitting, including 10 different density profiles where ellipticity can be introduced into either the projected density or the lensing potential. For lens models that require numerical integration, adaptive Gauss-Patterson quadrature (or alternatively, Romberg integration) is used to achieve the desired accuracy. For each model, there is also an option to speed up the computations by tabulating the lensing properties over a grid of polar coordinates and ellipticity so that interpolation can be used instead of integration. Chi-square optimization can be performed with the downhill simplex method (plus optional simulated annealing) or Powell's method; for Bayesian parameter estimation, nested sampling or an adaptive Metropolis-Hastings MCMC algorithm (T-Walk) can be used to infer the Bayesian evidence and posteriors. These calculations can be parallelized with MPI and OpenMP to run on a computing cluster. An additional tool, mkdist, generates 1D and 2D posterior plots in each parameter, or if an optimization was done, approximate posteriors can be plotted using the Fisher matrix. The QLens package includes an introductory tutorial [qlens\_tutorial.pdf](qlens_tutorial.pdf) that is meant to be readable for undergraduates and beginning graduate students with little or no experience with gravitational lensing concepts.

Required packages for basic, out-of-the-box configuration:
* GNU Readline &mdash; for command-line interface
* gnuplot &mdash; for generating and viewing plots from within QLens

Optional packages:
* OpenMP &mdash; used for multithreading likelihood evaluations, useful especially for lens models that require numerical integration for lensing calculations, or if source pixel reconstruction is being used.
* MPI &mdash; for running multiple MCMC chains simultaneously using twalk, or increasing acceptance ratio during nested sampling; a subset of MPI processes can also be used to parallelize the likelihood if source pixel reconstruction is used, or if multiple source points are being fit at different redshifts
* CFITSIO library &mdash; for reading and writing FITS files
* MUMPS package &mdash; for sparse matrix inversion in pixel image modeling (often painful to install however)
* UMFPACK &mdash; alternative to MUMPS; much easier to install but not quite as fast or parallelized
* TCMalloc &mdash; recommended if compiling with OpenMP or MPI; speeds up multithreading by reducing lock contention between threads

# change log (Aug. 6, 2018)

Upgrades since Jul. 15:

1. A new command has been added, 'fit changevary', that allows you to change the vary flags of an already existing lens. This is useful if you want to do an initial run with fewer parameters, followed by a fit using more free parameters. See 'help fit changevary' for usage.

2. It is now possible to update specific parameters of a lens model. For example, you can do 'lens update 0 b=1.5' and it will only update parameter b for lens model 0. You can also change multiple parameters using the same format, e.g 'lens update 0 b=1.5 q=0.8 theta=30'. However, you can still update lenses using the old format as well, where you enter in all the lens parameters the way you would to create the lens. 

3. The 'subhalo\_rmax' command now works for perturbers that are either in front of or behind the lens. In these cases, in addition to the mass within the perturbation radius, it will also output the mass scaled to the primary lens plane, i.e. what the mass would have to be in order to generate the same perturbation radius if the perturber is a subhalo in the primary lens plane.

4. It is now possible to change the redshift of a lens, using 'lens update'; just put in the argument 'z=#' and it will update the redshift accordingly. Note that if the lens is defined according to an Einstein radius parameter or kappa scaling, changing the redshift will not affect the lensing (unless the source redshift is not equal to zsrc\_ref, which the Einstein radius is defined by; also it may affect the recursive effect if there are lenses at other redshifts). However, if the lens is defined by mass, e.g. the NFW in pmode=1 or 2, the lensing can change dramatically if the lens redshift is updated.

5. The redshift of a given lens can now be varied as a free parameter if desired. (This can be useful to explore the degeneracy of redshift with other parameters.) To do this, when entering the vary flags, at the end of the line enter 'varyz=1'. If doing an MCMC or nested sampling, the lower/upper limits for z will then need to be entered after doing so for the other parameters being varied.

Upgrades since Apr. 11:

QLens now allows for the creation of lenses at multiple redshifts. To create lenses at different redshifts, you can either change 'zlens' before creating a new lens, or include the argument 'z=#' in the line when creating the lens. When you type 'lens' to list the lenses, it will show the redshift of each lens. Note that you cannot change the redshift of a lens once you have created it. The full nonlinear effect of having multiple lens planes is taken into account when calculating the deflections and magnifications (I have not done this for time delays yet, however; it's on the to-do list). In general, any combination of multiple lens planes and multiple sources are now allowed in qlens.

Upgrades since Mar. 18:

1. There is now help documentation in qlens for adding priors and transforming fit parameters. See 'help fit priors' and 'help fit transform' to learn how to implement these options.

2. I have added a new lens model, 'cnfw' which is an NFW profile modified to include a core. The 3d profile goes like (r+r<sub>c</sub>)<sup>-1</sup>(r+r<sub>s</sub>)<sup>-2</sup> and is used, e.g. in Penarrubia et al. (2012). Just as with the NFW profile, there is an analytic solution for the kappa and, in the spherical q=1 case, the deflection as well. Thus, the lensing calculations are not significantly slower compared to the NFW model and in the limit of zero core, it reduces exactly to NFW. Likewise, the pseudo-elliptical version can be used to speed up the lensing calculations using emode=3. See 'help lens cnfw' for help on creating a cored NFW lens model in qlens.

Upgrades since Feb. 9:

1. There is now a command, 'mass\_r' that allows you to find the 2D mass and 3D mass enclosed within a specified radius. For example, if you want the mass within 20 arcseconds for lens #0, you type "mass\_r 0 20". Qlens will then spit out the 2D and 3D masses enclosed as well as the 3D density at that radius, and will also spit out your radius in kpc along with the masses. Also, when using the "cosmology" command, qlens will spit out the total mass if the lens has a well-defined total mass, as well as the half-mass radius. If you don't see these values, it is because the total mass did not converge and is therefore infinite for that particular lens model--this is true for the NFW model, for example. For some profiles, analytic solutions are available for the masses, while for others (e.g. corecusp and truncated NFW) numerical integration over the 3D density profile is performed. For the Sersic profile, there is no analytic 3D density profile, so the 3D density profile is calculated using the standard deprojection integral. All masses are calculated for the corresponding spherical profile, and hence have no dependence on the axis ratio q (in the case of the 2D mass, you can think of it as the projected mass within an ellipse at the given elliptical radius, defined by the 'emode' in qlens). Both the 2D and 3D masses are now available as derived parameters (type "help fit dparams" for usage).

2. There is a new setting, 'simplex\_show\_bestfit', which is off by default. If you turn it on, qlens will show all the best-fit parameters from the previous temperature during simulated annealing. This allows you to monitor whether the model is getting "stuck" somewhere in parameter space, or whether the parameters change rapidly even at low chi-square which could indicate significant degeneracies. (However if you have a large number of free parameters, all the values might not fit in a single line, in which case the terminal window will look very messy.)

3. QLens has two new lens models, "tab" and "qtab". These are models that tabulate the lensing properties from a specified lens model, so that interpolation is used to calculate deflections, magnification, etc. For models that require numerical integration (e.g. NFW, Sersic, etc.), this can speed up the lensing calculations by an order of magnitude. The "tab" model interpolates over a grid of values in (log(r),&phi;), where &phi; is the azimuthal angle, and is useful for models where the ellipticity is already known, usually from the stellar light profile. Once the tabulation is done, the mass of the lens is scaled by a free parameter which can be varied during a fit, which can serve as a proxy for the stellar mass-to-light ratio. It is also possible to vary the radial scale as well. Here is an example:

		lens sersic 2 5 2 0.7 30 0.3 0.8     # this becomes lens 0
		fit lens tab lens=0 2 5 30 0.3 0.8   # parameters: kscale=2, rscale=5, theta=30, xc=0.3, yc=0.8
		1 0 0 0 0

	If you run the above commands, it will take several seconds to tabulate the lensing properties, after which the original lens is replaced by the tabulated model which is named "tab(sersic)". The tabulation is done in the (non-rotated) coordinate system of the lens, which means the angle and center coordinates of the original model will be ignored in favor of the new values. The first argument after 'lens=0' gives the parameter 'kscale', the kappa scaling; this number does not change anything about the lens, it simply defines the reference scale factor, which we make equal to the 'keff' parameter in the Sersic model. This means if you change kscale to 4, it will double the mass density of the lens everywhere since it is twice the reference value. Likewise, the next argument gives the reference value for the radial scaling, which we set equal to the effective radius parameter 'reff' in the Sersic model. Note that in this example, we are only varying the 'kscale' parameter, since we are simulating a case where the effective radius, ellipticity, position angle and center are determined from the stellar light profile and the stellar mass-to-light ratio is the only parameter that needs to be constrained.

	There are three settings relevant to the tabulation: 'tab\_rmin' which sets the minimum radius of the grid (this should be small, e.g. 1e-3, but cannot be zero since the tabulation is over log(r)); 'tab\_r\_N' which sets the number of points along the radial direction (default=2000) and 'tab\_phi\_N' which is for the angular direction (default=200). Because linear interpolation is used, the number of grid points should be at least as large as these defaults; when in doubt, use more points. The maximum radius being tabulated is determined from the Einstein radius of the original lens (or if you fix the grid using the "grid" command, it uses the grid size to set the maximum radius). It is important to test the accuracy of your tabulated model by comparing lens calculations with the original lens, using the 'findimg' or 'lensinfo' commands. If the deflections or magnifications differ by more than a percent, then you should add more points to the grid.

	Since the tabulated model may take some time to generate, you can save the tabulated values to a file if you plan to use the model again, which can then be loaded next time. For example, continuing the script above:

		lens savetab 0 sersic   # saves the tables from lens 0 (our 'tab' model) to file 'sersic.tab'
		# Now we demonstrate loading the tables...
		lens clear
		fit lens tab sersic 2 5 30 0.3 0.8
		1 0 0 0 0

	The last two lines creates a tabulated model from the values in the file 'sersic.tab' and reproduce the exact same model we had before. It is not necessary to type the '.tab' extension in the filename when you are saving or loading the tables.

	The "qtab" model extends the above model so that it interpolates in the axis ratio q as well, which allows it to capture all the essential properties of elliptical lens models that have a single scale radius and mass parameter--this includes the NFW, Hernquist, and exponential disk models. However this model is still a work in progress; because of the way linear interpolation is done, a very large number of grid points is required for good accuracy, enough to push the typical 16 GB of memory on a laptop. So further work is need on my part to make the interpolation more effective without sacrificing the speed too much.

Upgrades since Jan. 18:

1. A new integration method has been implemented, Gauss-Patterson quadrature, which is now the default for lens models that require numerical integration. This is an extension of Gaussian quadrature developed by Kronrod and Patterson that gives "nested" rules, meaning you can add points to achieve higher order while still using the previous function evaluations. This allows for an adaptive quadrature scheme where you can add points until converging to the desired accuracy, which renders the lensing calculations more robust compared to just using Gaussian quadrature with a fixed number of points (the previous default method). If the profile is very steep, you may get a warning that the maximum number of allowed points (511) has been reached, which indicates that the accuracy of the lensing calculations may suffer in this case; this occurs, for example, in the alpha model if the slope &alpha; > 1.9 and the core is small, since this is approaching the limit of a point mass (&alpha;=2; note however that if the core is set to zero in the alpha model, numerical integration is not required so this issue only arises with a small nonzero core). However if you make the integral tolerance stricter you are more likely to get this warning. The default tolerance is 5e-3, although the actual accuracy achieved is always better than this, often by an order of magnitude or more, since the error estimate being evaluated is always the error for the second-to-last integral evaluation. To change the tolerance, type "integral\_method patterson #" where # is the tolerance.

2. There is now an option to add derived parameters in QLens, so that posteriors in the derived parameters can be plotted after nested sampling or T-Walk. Currently there are four possible derived parameters, the kappa, log-slope of kappa, projected mass and 3D mass, which are calculated at a specific radius specified by the user (the mass parameters have been added as of 3/21). For the kappa/slope parameters, if you do not specify a specific lens it will add up the kappa for all the lenses in your model. (Type "help fit dparams" for more specific info on how to add derived parameters.) Although the derived parameters are not listed as free parameters during the fit, after they are output to the "chains\_..." directory they will be treated just like any other parameter when you run mkdist. If you have a lot of derived parameters and don't want them all the posteriors plotted in the triangle plot, you can edit the python script produced by mkdist to remove the parameters you don't want to show. And if you just want 1-sigma errors in the derived parameters, you can run "mkdist ... -e" and it will spit them out.

3. The "settings" and "help settings" commands have been redone, so that all the settings are listed and organized by category, and you can specify specific categories ("plot\_settings", "fit\_settings", etc.). This should make it much easier to look up settings in different categories.

4. For certain lens models, the lensing calculations have been sped up by eliminating certain redundancies in the lensing calculations when searching for point images. For example, for models where numerical integrals are required, the deflection and hessian of the potential actually share two integrals, and thus time is reduced if they are calculated at the same time. Even for models with analytic solutions, the deflection, hessian and (sometimes) kappa make use of the same intermediate calculations and thus should be calculated simultaneously if they are all needed. This is true for all pseudo-elliptical models, as well as the alpha model with zero core size.

Upgrades since Dec. 28:

One major upgrade: I have implemented the "emode" command, where you can set the ellipticity mode which controls how ellipticity is put into lens models. Among other things, this allows for any ellipsoidal mass model to be converted into a "pseudo-elliptical" model (meaning ellipticity is put into the potential, not the projected density).

There are four different ellipticity modes (0-4). In modes 0-2, ellipticity is introduced into the projected density (kappa), whereas in mode 3 it is introduced into the potential. Lest you get overwhelmed, let me just say that if you are not very experienced with lens models, the differences between modes 0-2 are not especially important, so you can just stick with the default mode (which is 1). It's important to know about mode 3 as an option, however (see below for description).

Here is a brief description of each ellipticity mode. In each example, we use coordinates in the "lens frame" where the major axis lies along x, minor axis along y.

1. Mode 0: in the projected density &Sigma;(R), we let R<sup>2</sup> &rarr; x<sup>2</sup> + (y/q)<sup>2</sup>.

	Ellipticity parameter: q (axis ratio).  This happens to be the mode that gravlens (lensmodel) uses for its lens models (albeit with e=1-q as the parameter), with a few exceptions. In this parameterization, the major/minor axes of the density contours are at (R,Rq) respectively. Typically the major axis of the tangential critical curve is roughly equal to the Einstein radius of the corresponding spherical (q=1) model. It's not ideal for model fitting if the Einstein radius is used as a parameter, because typically the *average* radius of the critical curve is better constrained, not its major axis.

2. Mode 1: in the projected density &Sigma;(R), we let R<sup>2</sup> &rarr; qx<sup>2</sup> + y<sup>2</sup>/q.

	Ellipticity parameter: q (axis ratio).  This is the default ellipticity mode in qlens. In this parameterization, the major/minor axes of the density contours are at (R/sqrt(q),R*sqrt(q)) respectively. The average radius of the critical curve is roughly comparable to the Einstein radius of the corresponding spherical model; the major axis will be larger than the Einstein radius, while the minor axis is smaller. 

3. Mode 2: in the projected density &Sigma;(R), we let R<sup>2</sup> &rarr; (1-&epsilon;)x<sup>2</sup> + (1+&epsilon;)y<sup>2</sup>.

	Ellipticity parameter: &epsilon;. This is the parameterization that lenstool uses for all its (non-pseudo) lens models; here again, for power law models the average radius of the critical curve is roughly equal to the Einstein radius. The axis ratio is given by the formula q = sqrt((1-&epsilon;)/(1+&epsilon;)).

4. Mode 3: in the lensing potential &phi;(R), we let R<sup>2</sup> &rarr; (1-&epsilon;)x<sup>2</sup> + (1+&epsilon;)y<sup>2</sup>.

	Ellipticity parameter: &epsilon;. This is the "pseudo-elliptical" version, where ellipticity is put into the potential, not the projected density; note that the elliptical radius is defined the same way as in mode 2. It's important to note that while emodes 0-2 are essentially just reparametrizations of the same ellipsoidal lens model, emode 3 is fundamentally different. The pseudo-elliptical model can do lens calculations significantly faster in most of the elliptical mass models, since analytic formulas are used for the deflection and magnifications (it simply modifies the spherical deflection formula, which has an analytic solution for all mass models in qlens). It will *not* improve the speed of calculations for the pjaffe model or the alpha model in the special case where alpha=1 or s=0, since in these cases the formulas are analytic regardless. Also keep in mind that the pseudo-elliptical models can lead to unphysical density contours when the ellipticity is high enough&mdash;they might begin to look "boxy" (or in more extreme cases, peanut-shaped) and can even become negative in certain places (you can check this using the command "plotlogkappa").

The "emode" command changes the default ellipticity mode for lens models that get created, but you can also specify the ellipticity mode of a specific lens you create by adding the argument "emode=#" to the line (e.g., "lens nfw emode=3 0.8 20 0.3"). It doesn't matter where you put the "emode=#" argument when creating a lens, as long as it's after the name of the lens model&mdash;it could be at the end of the line, for example. Also note that different lenses don't have to have the same ellipticity mode. If a lens is created using mode 3, the prefix "pseudo-" is added to the lens model name. The pseudo-NFW model "pnfw" is still allowed in qlens, but I will remove it soon since it's redundant; it's just the NFW model with emode=3.

Upgrades since Dec. 4:

1. It is now possible for qlens to automatically solve for the best-fit source point coordinates (instead of varying them as free parameters) even if you are using the image plane chi-square; the source-plane chi-square function is still used to do this, but the image plane chi-square is then evaluated using the inferred source point coordinates. To make things as simple as possible, I have this feature turned off by default, regardless of whether you are using source plane or image plane chi-square (note that previously, it was turned on by default when using the source plane chi-square). To turn it on, type "analytic\_bestfit\_src on" (highly recommended if you are modeling several source points!). One approach you can take is to start with analytic\_bestfit\_src on, run a fit, adopt the best-fit model and then turn it off. The source point coordinates will now be listed as free parameters, but they will automatically take the values obtained from the previous fit. Then you can do another fit where the source coordinates are varied as free parameters.

2. Often it can be useful to enforce the x- and y-axes to have the same scale in the plots. This can now be done with the command "plot\_square\_axes on".

3. There is a new command, "cosmology", that displays cosmological information including physical properties of the lens models. Currently this only displays information for the "nfw/pnfw" and "pjaffe" models (e.g. M\_200, concentration, velocity dispersion, etc.), although I plan to extend this to the power-law ("alpha") model in the near future.

4. There is a new setting called "chisq\_imgplane\_threshold". If you set it to some value greater than zero, then when you run a fit with the image plane chi-square turned on, it will first evaluate the source plane chi-square every time; if its value is smaller than the given threshold, qlens will then evaluate the image plane chi-square, otherwise it will simply return the source plane chi-square value without searching for images. This can save a great deal of time compared to only using the image plane chi-square, because if the chi-square is terrible, then the images are nowhere near being a good match, and it is a waste of time to search for images. The threshold should not be made too low (somewhere between 1e4 and 1e6 is a good choice), and this feature should *not* be used when minimizing the chi-square because it can get stuck around the chi-square threshold where it switches over. However, if used with nested sampling or T-Walk, a great deal of time can be saved. Keep in mind that the resulting points in the chain(s) can only be trusted if they mainly sample regions where the chi-sqaure is below the stated threshold (which is why the threshold cannot be too low!). So far I have gotten good results in the tests I have run using nested sampling, but I would love to hear feedback regarding this option.

5. I upgraded the "fit plotimg" command so that if you just type "fit plotimg" by itself, it plots all the images (model and data) in one plot. The critical curves shown are defined for whatever 'zsrc' is set to, which by default is just the same as the first image dataset. Each image set uses a different marker to help differentiate them. If there's a large number of datasets, the key can be a nuisance, so I added an option "plot\_key\_outside" which moves the key outside the figure if you set it to "on". You can also just turn off the key with "plot\_key off". Using the "src=#" argument, you can plot the images for a specific source point, or you can specify a range of source points; for example, to plot source points 1-3, type "fit plotimg src=1-3".

6. There is a new command, "fit plotsrc". This command works nearly the same way as "fit plotimg", except that it maps the data images to the source plane and plots the resulting source points, along with the model source points for comparison. For this reason, only the source plane is plotted using this command, so if you plot to a file, only one filename needs to be specified. In every other respect, the syntax is the same as for "fit plotimg". There is another new command, "fit data\_imginfo", which simply prints the source point coordinates for all the data images to the screen, but also lists the expected flux and/or time delays at the positions of the data images (if "chisqflux" or "time\_delays" are turned on, respectively).

Upgrades since Nov. 28:

1. There is a new option for picking the initial source point(s), "fit sourcept auto". This adopts a best-fit source point(s) using your starting lens configuration and the resulting source plane chi-square function. This is useful if you want to go straight to fitting using the image plane chi-square, which requires initial parameters for the source point coordinates. I should emphasize that if you start out by doing a fit using the source plane chi-square, you don't need to pick initial source point coordinates *at all*, because the best-fit source point coordinates are solved for analytically during the fit (by default).

2. QLens now organizes image data sets by their redshifts. If two image sets have the same source redshift, QLens will not recompute the lensing quantities over the grid when calculating the chi-square for the second image set (since they're the same for a given redshift). This means you can have many images sets at the same redshift (for example, different "knots" along an arc) with very little extra overhead when calculating the chi-square. If you are MPI'ing the chi-square function, you should set the number of processes per group equal to the number of source planes (so, if there are three different source redshifts, there should be 3 MPI processes per group; see below readme "upgrades since Nov. 12", item 2, for instructions on how to do this).

3. By default, "chisqmag" is now turned on (this option tells qlens to include magnification information from the model in the source plane chi-square for image positions). Generally, if one is using MCMC or simulated annealing, it is better to use magnification in the source plane chi-square since the resulting parameter uncertainties are much more accurate. The only reason to *not* have it on is if you are simply trying to minimize the source plane chi-square (without annealing), since it can become unstable if you're not close to a good fit. Since your odds of success are much greater if you use annealing or MCMC, I realized it's safer to have "chisqmag" on by default.

4. Fixed a bug where the cell sizes (which are compared to min\_cellsize) were not being calculated during the fit. (This meant that the subgridding around cluster members was coarser than it should have been.)

Upgrades since Nov. 15:

1. I have added a new lens model, the pseudo-elliptical NFW model (called "pnfw" in QLens). This is the model of Golse & Kneib (2002) that introduces ellipticity into the potential, rather than the projected density. It has the advantage of having analytic solutions for all the lensing quantities, which makes it roughly 10 times faster than the elliptical NFW model, but the density contours become unphysical for ellipticity e > 0.4 or so. Note that when using the pnfw model, the ellipticity parameter is used rather than the axis ratio q. So e=0 means no ellipticity.

2. Fixed a bug related to the parameter anchoring (using "/anchor=") for Alpha & PJaffe models. The ratios listed when you type "fit lens" should now be accurate for these models.

Upgrades since Nov. 12:

1. There are two new variables that are relevant to using the image plane chi-square: 'chisq\_mag\_threshold' and 'chisq\_imgsep\_threshold'. The latter tells the chi-square to ignore duplicate images with separations smaller than the given threshold, while the former ignores images fainter than the given magnification threshold. Both are zero by default, but probably chisq\_imgsep\_threshold = 1e-3 is a reasonable value in most cases&mdash;I find that the duplicate images always seem to have separations smaller than this. Usually increasing the number of splittings around critical curves (using 'cc\_splitlevels) will get rid of duplicate "phantom" images, but it slows down the image searching. So setting the above threshold may be a useful alternative.

2. The image plane chi-square function (for point image modeling) can now be parallelized with MPI if multiple source points are being fit to. This can significantly reduce the time for each chi-square evaluation if you lens model requires numerical integration to calculate deflections (which is true for the NFW or Sersic models, e.g.). As an example, if you are modeling four source points, you can split the work among up to four MPI processes. To do this, e.g. with four processes, run qlens as follows:

		mpirun -n 4 qlens script.in -g1

	The '-g1' argument tells it to use all four processes for each chi-square evaluation. If you have enough processors for this, it will run 3 or 4 times faster in this example; but even with just two processors, you could run with the '-n 2' argument and it will split up the four source points among the two processes (so two each), nearly doubling the speed of each chi-square evaluation. This gives you a better speedup in comparison to running with multiple OpenMP threads (which speeds up the image searching), but with enough processors, you could combine both approaches to make it even faster. In the above example, if you have eight processors you can use two OpenMP threads (with 'export OMP_NUM_THREADS=2' before you run qlens) and you will get an additional speedup.

	Make sure you are not using too many processes for the machine you're using. If you've compiled QLens with OpenMP, you can test out the time required for each chi-square evaluation by running QLens with the '-w' argument (regardless of whether you're using more than one OpenMP thread or not) and using the 'fit chisq' command; it will spit out the time elapsed. This is highly recommended as it will allow you to experiment with different number of processes/threads until you find the fastest combination, before you run your fit(s).

	If you are using TWalk or nested sampling, you have the option of multiple MPI 'groups', where each group does simultaneous chi-square evaluations (which for T-Walk means you can move multiple chains forward at the same time), while the processes within each group parallelize the chi-square function. For example, if you want two processes per chi-square evaluation, and four MPI groups to move the chains forward, you would then have 8 processes total (again, assuming you have enough processors to do this), so you would run it as

		mpirun -n 8 qlens script.in -g4

	By default, if you don't specify the number of groups with the '-g' argument, QLens will assume the same number of groups as processes (which would be eight in the above example); in other words, it assumes only one process per group unless you tell it otherwise. On a computing cluster you have a lot more processors, hence more freedom to do a combination of all these approaches&mdash;parallelizing the chi-square evaluations, running parallel chains, and multithreading the image searching with OpenMP.

Previous upgrades from Nov. 13 version:

1. General lens parameter anchoring

	General lens parameter anchoring has been implemented, so that you can now anchor a lens parameter to any other lens parameter. To demonstrate this, suppose our first lens is entered as follows:

		fit lens alpha 5 1 0 0.8 30 0 0  
		1 1 0 1 1 1 1

	so that this now becomes listed as lens "0".

	a) Anchor type 1: Now suppose I add another model, e.g. a kappa multipole, where I want the angle to always be equal to that of lens 0. Then I enter this as

		fit lens kmpole 0.1 2 anchor=0,4 0 0  
		1 0 0 1 1

	The "anchor=0,4" means we are anchoring this parameter (the angle) to lens 0, parameter 4 which is the angle of the first lens (remember the first parameter is indexed as zero!). The vary flag must be turned off for the parameter being anchored, or else qlens will complain.

	NOTE: Keep in mind that as long as you use the correct format, qlens will not complain no matter how absurd the choice of anchoring is; so make sure you have indexed it correctly! To test it out, you can use "lens update ..." to update the lens you are anchoring to, and make sure that the anchored parameter changes accordingly.

	b) Anchor type 2: Suppose I want to add a model where I want a parameter to keep the same *ratio* with a parameter in another lens that I started with. You can do this using the following format:

		fit lens alpha 2.5/anchor=0,0 1 0 0.8 30 0 0  
		1 0 0 1 1 1 1

	The "2.5/anchor=0,0" enters the initial value in as 2.5, and since this is half of the parameter we are anchoring to (b=5 for lens 0), they will always keep this ratio. It is even possible to anchor a parameter to another parameter in the *same* lens model, if you use the lens number that will be assigned to the lens you are creating. Again, the vary flag *must* be off for the parameter being anchored.

	We can still anchor the lens's center coordinates to another lens the old way, but in order to distinguish from the above anchoring, now the command is "anchor\_center=...". So in the previous example, if we wanted to also anchor the center of the lens to lens 0, we do

		fit lens alpha 2.5/anchor=0,0 1 0 0.8 30 anchor_center=0  
		1 0 0 1 1 0 0

	The vary flags for the center coordinates must be entered as zeroes, or they can be left off altogether.

2. Fit parameter limits are now assigned default values for certain parameters. For example, in any lens, when you type "fit plimits" you will see that by default, the 'q' parameters have limits from 0 to 1, and so on. This used to be done "under the hood" but now is made explicit using plimits. The plimits are also used to define ranges when plotting approximate posteriors using the Fisher matrix (with 'mkdist ... -fP').

3. You no longer have to load an input script by prefacing with '-f:' before writing the file name. Now you can simply type "qlens script.in" (or whatever you call your script).

4. Bug fix: the vary flags for external shear (when added with "shear=# #" at the end of lens models) were not being set properly for NFW and several other lenses. This has been fixed.

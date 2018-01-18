# QLens (beta version)
QLens is a software package for modeling and simulating strong gravitational lens systems. Both point image modeling (with option to include fluxes and time delays) and pixel image modeling (using pixellated source reconstruction) are supported. QLens includes 13 different analytic lens models to choose from for model fitting, with an option to load a numerically generated kappa profile using interpolation over a table of kappa values. Chi-square optimization can be performed with the downhill simplex method (plus optional simulated annealing) or Powell's method; for Bayesian parameter estimation, nested sampling or an adaptive Metropolis-Hastings MCMC algorithm (T-Walk) can be used to infer the Bayesian evidence and posteriors. An additional tool, mkdist, generates 1d and 2d posterior plots in each parameter, or if an optimization was done, approximate posteriors can be plotted using the Fisher matrix. The QLens package includes an introductory tutorial [qlens\_tutorial.pdf](qlens_tutorial.pdf) that is meant to be readable for undergraduates and beginning graduate students with little or no experience with gravitational lensing concepts.

Required packages for basic, out-of-the-box configuration:
* GNU Readline &mdash; for command-line interface
* gnuplot &mdash; for generating and viewing plots from within QLens

Optional packages:
* OpenMP &mdash; used for multithreading likelihood evaluations, useful especially for lens models that require numerical integration for lensing calculations, or if source pixel reconstruction is being used.
* MPI &mdash; for running multiple MCMC chains simultaneously using twalk, or increasing acceptance ratio during nested sampling; a subset of MPI processes can also be used to parallelize the likelihood if source pixel reconstruction is used, or if multiple source points are being fit to
* CFITSIO library &mdash; for reading and writing FITS files
* MUMPS package &mdash; for sparse matrix inversion in pixel image modeling (often painful to install however)
* UMFPACK &mdash; alternative to MUMPS; much easier to install but not quite as fast or parallelized

# change log (Jan. 18, 2018)

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

4. There is a new setting called "chisq\_srcplane\_threshold". If you set it to some value greater than zero, then when you run a fit with the image plane chi-square turned on, it will first evaluate the source plane chi-square every time; if its value is smaller than the given threshold, qlens will then evaluate the image plane chi-square, otherwise it will simply return the source plane chi-square value without searching for images. This can save a great deal of time compared to only using the image plane chi-square, because if the chi-square is terrible, then the images are nowhere near being a good match, and it is a waste of time to search for images. The threshold should not be made too low (somewhere between 1e4 and 1e6 is a good choice), and this feature should *not* be used when minimizing the chi-square because it can get stuck around the chi-square threshold where it switches over. However, if used with nested sampling or T-Walk, a great deal of time can be saved. Keep in mind that the resulting points in the chain(s) can only be trusted if they mainly sample regions where the chi-sqaure is below the stated threshold (which is why the threshold cannot be too low!). So far I have gotten good results in the tests I have run using nested sampling, but I would love to hear feedback regarding this option.

5. I upgraded the "fit plotimg" command so that if you just type "fit plotimg" by itself, it plots all the images (model and data) in one plot. The critical curves shown are defined for whatever 'zsrc' is set to, which by default is just the same as the first image dataset. Each image set uses a different marker to help differentiate them. If there's a large number of datasets, the key can be a nuisance, so I added an option "plot\_key\_outside" which moves the key outside the figure if you set it to "on". You can also just turn off the key with "plot\_key off". Using the "src=#" argument, you can plot the images for a specific source point, or you can specify a range of source points; for example, to plot source points 1-3, type "fit plotimg src=1-3".

6. There is a new command, "fit plotsrc". This command works nearly the same way as "fit plotimg", except that it maps the data images to the source plane and plots the resulting source points, along with the model source points for comparison. For this reason, only the source plane is plotted using this command, so if you plot to a file, only one filename needs to be specified. In every other respect, the syntax is the same as for "fit plotimg". There is another new command, "fit data\_imginfo", which simply prints the source point coordinates for all the data images to the screen, but also lists the expected flux and/or time delays at the positions of the data images (if "chisqflux" or "time\_delays" are turned on, respectively).

Upgrades since Nov. 28:

1. There is a new option for picking the initial source point(s), "fit sourcept auto". This adopts a best-fit source point(s) using your starting lens configuration and the resulting source plane chi-square function. This is useful if you want to go straight to fitting using the image plane chi-square, which requires initial parameters for the source point coordinates. I should emphasize that if you start out by doing a fit using the source plane chi-square, you don't need to pick initial source point coordinates *at all*, because the best-fit source point coordinates are solved for analytically during the fit (by default).

2. QLens now organizes image data sets by their redshifts. If two image sets have the same source redshift, QLens will not recompute the lensing quantities over the grid when calculating the chi-square for the second image set (since they're the same for a given redshift). This means you can have many images sets at the same redshift (for example, different "knots" along an arc) with very little extra overhead when calculating he chi-square. If you are MPI'ing the chi-square function, you should set the number of processes per group equal to the number of source planes (so, if there are three different source redshifts, there should be 3 MPI processes per group; see below readme "upgrades since Nov. 12", item 2, for instructions on how to do this).

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

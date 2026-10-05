from qlens_helper import *

cosmo = Cosmology(omega_m=0.3,hubble=0.7)
q = QLens(cosmo)

(lens,src,pixsrc,imgdata) = q.pix_objects()   # this is so we can enter 'lens' instead of 'q.lens', 'src' instead of 'q.src', etc.
(params,dparams) = q.param_objects()         # same as above; we can enter 'params' instead of 'q.params', etc.

show_commands()

q.fit_label = 'delaunay_fit'

sim_hst_data = imgdata.load("sim_hst_img.fits",band=0,pixsize=0.049)
sim_hst_data.load_noise_map("demo_noisemap.fits")
sim_hst_data.load_mask("delaunay_demo_mask.fits")

#sim_hst_data.unmask_all_pixels()
#sim_hst_data.mask_low_sn_pixels(threshold=0.01)   # note, the signal threshold for masking out a pixel is given by 'threshold' times the noise dispersion for that pixel
#sim_hst_data.trim_mask_windows(noise_threshold=4,npixel_threshold=20)    # if npixel_threshold is omitted, it is set to zero by default
#sim_hst_data.unmask_neighbor_pixels()
#pause()

plotdata(q,nomask=True,title="Mock data for delaunay_fit_demo.py")

q.psf_threshold = 1e-3
q.sbmap_load_psf("hst_psf.fits")

q.split_imgpixels = True
q.imgpixel_nsplit = 2
q.fft_convolution = True

q.sci_notation = True
q.shear_components=True
q.ellipticity_components=True

q.zlens = 0.5
q.zsrc = 2

Alpha = SPLE({"b": 1.3, "alpha": 1.0, "s": 0.0, "e1": 0.0, "e2": 0.0, "xc": 0.015, "yc": -0.006},qlens=q) # Note: the fit can be sensitive to the initial xc, yc
Alpha.vary([1,1,0,1,1,1,1])

extshear = Shear({"shear1": 0.0, "shear2": 0.0},qlens=q)
extshear.vary([1,1,0,0])

lens.add(Alpha,shear=extshear)

params.set_limits([      # We can define prior limits here if we prefer (instead of defining them for each lens object above).
    ("b",1.1,1.8),       # One advantage is that if we transform a parameter, you can define your limits in terms of the transformed parameter (e.g. log(mass)).
    ("alpha",0.6,1.6),
    ("e1",-0.5,0.5),
    ("e2",-0.5,0.5),
    ("xc",-0.5,0.5),
    ("yc",-0.5,0.5),
    ("shear1",-0.2,0.2),
    ("shear2",-0.2,0.2)
])


# First, we will fit with an analytic source to get us close to the right part of parameter space
q.set_source_mode("sbprofile")

q.sb_ellipticity_components=True

# Note: keyword 'lensed_center_peak_sb' means we're ray-tracing the brightest data pixel to define the source center (which will override the xc, yc, inputs below)
gauss_src = Gaussian({"sbmax": 0.8, "sigma": 0.1, "e1": 0.0, "e2": 0, "xc": 0, "yc": 0}, pmode=0, qlens=q, lensed_center_peak_sb=True)
gauss_src.vary_all()
gauss_src.set_limits([
    ("sbmax", .01, 10),
    ("sigma", .001, 0.2),
    ("e1", -0.5, 0.5),
    ("e2", -0.5, 0.5),
    ("xc_l", -2, 2),
    ("yc_l", -2, 2)
])

src.add(gauss_src)

pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

q.gradtol = 0.001
q.chisqtol = 0
q.nrepeat = 0

#q.run_fit("bfgs",adopt=True,show_errors=False)

pause()

src.clear() # Now we delete the analytic source and switch to a pixellated source grid

q.set_source_mode("delaunay")
q.nimg_prior=True
q.nimg_threshold=1.5
q.outside_sb_prior=True
q.outside_sb_frac_threshold=0.15
q.regularization_method="matern_kernel"
q.optimize_regparam=False    # cannot do optimize_regparam when using autodiff

srcgal = pixsrc.add()
srcgal.update({"regparam":100,"corrlength":0.1,"matern_index":1.0})
srcgal.vary([1,1,1])

params.transform("regparam","log")
params.transform("corrlength","log")

params.set_limits([
    ("log(regparam)",-2,3), 
    ("log(corrlength)",-2,2),
    ("matern_index",0.05,3) 
])
 
q.fitmodel()

#q.sbmap_invert()
#plotimg(q,nres=True,title="Residuals before optimizing")      # NOTE: plotimg, plotsrc, and plotdata all return figures and axes, so you can also do e.g.
#plotsrc(q,interp=False,title="Reconstructed source before optimizing")    # (fig, ax) = plotimg(q,show=False) and modify the figures

pause()

q.nimg_prior=True
q.outside_sb_prior=True
#q.param_covmatrix_scale_fac = 5   # we will scale the uncertainties by this factor because with pixellated sources, fisher matrix uncertainties are often unreliably small
q.run_fit("bfgs",adopt=True,show_errors=True)

q.sbmap_invert()
pause()

plotimg(q,nres=True,title="Residuals from best-fit model")

plotsrc(q,title="Reconstructed source from best-fit model")
plotsrc(q,interp=True,title="Reconstructed source (with interpolation)")

q.param_markers("1.3634 1.17163  0.0347001 -0.0100747 0.0152892 -0.00558392 0.0647257 -0.0575047") # true model
q.mkposts(fisher=True)   # with pixellated sources, fisher matrix uncertainties should be taken with a grain of salt--actual uncertainties are usually higher

#plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing


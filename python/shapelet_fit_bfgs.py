from qlens_helper import *

cosmo = Cosmology(omega_m=0.3,hubble=0.7)
q = QLens(cosmo)

(lens,src,pixsrc,imgdata) = q.pix_objects()   # this is so we can enter 'lens' instead of 'q.lens', 'src' instead of 'q.src', etc.
(params,dparams) = q.param_objects()         # same as above; we can enter 'params' instead of 'q.params', etc.

show_commands()

# in this demo script, the mock source deviates from being perfectly elliptical, so we fit it with a Sersic profile + shapelets
q.fit_label = 'shapelet_fit'

sim_hst_data = imgdata.load("boxsersic_img.fits",band=0,pixsize=0.049)
sim_hst_data.load_noise_map("demo_noisemap.fits")

# note, there is typically no mask needed when using analytic sources with/without shapelets, so we don't load any

plotdata(q,nomask=True,title="Mock data for delaunay_fit_demo.py")

q.psf_threshold = 0
q.sbmap_load_psf("hst_psf.fits")

q.split_imgpixels = True
q.imgpixel_nsplit = 4
q.fft_convolution = True

q.sci_notation = True
q.shear_components=True
q.ellipticity_components=True

# since we will anchor the shapelets to the Sersic profile, we don't need qlens to automatically estimate the optimal shapelet scale/center. 
q.auto_shapelet_center=False
q.auto_shapelet_scale=False

q.zlens = 0.5
q.zsrc = 2

q.set_source_mode("shapelet")
q.regularization_method="none"    # 'none' is ok with shapelet order n < 10 or so; if using n > 10, it is better to regularize with 'norm' regularization

Alpha = SPLE({"b": 1.3, "alpha": 1.0, "s": 0.0, "e1": 0.0, "e2": 0.0, "xc": 0.01, "yc": 0.005},qlens=q) # Note: the fit can be sensitive to the initial xc, yc
Alpha.vary([1,1,0,1,1,1,1])
extshear = Shear({"shear1": 0.0647257, "shear2": -0.0575047},qlens=q)
extshear.vary([1,1,0,0])

lens.add(Alpha,shear=extshear)

params.set_limits([      # We can define prior limits here if we prefer (instead of defining them for each lens object above).
    ("b",1.1,1.8),       # One advantage is that if you transform a parameter, you can define your limits in terms of the transformed parameter (e.g. log(mass)).
    ("alpha",0.6,1.6),
    ("e1",-0.5,0.5),
    ("e2",-0.5,0.5),
    ("xc",-0.5,0.5),
    ("yc",-0.5,0.5),
    ("shear1",-0.2,0.2),
    ("shear2",-0.2,0.2)
])


# First, we will fit with an analytic source to get us close to the right part of parameter space
q.set_source_mode("shapelet")

q.sb_ellipticity_components=True

# Note: keyword 'lensed_center_peak_sb' means we're ray-tracing the brightest data pixel to define the source center (which will override the xc, yc, inputs below)
sersic_src = Sersic({"s_eff": 1, "R_eff": 0.3, "n": 0.5, "e1": 0.0, "e2": 0.0}, pmode=1, qlens=q, lensed_center_peak_sb=True)
sersic_src.vary([1,1,1,1,1,1,1])
sersic_src.set_limits([
    ("s_eff", .01, 10),
    ("Reff", .001, 1.0),
    ("n", .1, 5),
    ("e1", -0.6, 0.6),
    ("e2", -0.6, 0.6),
    ("xc_l", -2, 2),
    ("yc_l", -2, 2)
])

#shapelets = Shapelet({"sigma": 0.1, "e1": 0, "e2": 0}, n=6, pmode=0, qlens=q)  # n is the shapelet order. No need to define center coordinates, we will anchor to sersic
shapelets = Shapelet({"sigma": 0.1}, n=6, pmode=0, qlens=q)  # n is the shapelet order. No need to define center coordinates, we will anchor to sersic
#shapelets.vary(["regparam"]) # only if regularizing

src.add(sersic_src)
src.add(shapelets,anchor_center=0)
shapelets.anchor_param("sigma",sersic_src,"Reff",ratio=0.5)   # shapelets tend to work better if the scale (sigma) is somewhat smaller than Reff, so we choose sigma=0.5*Reff

params.transform("Reff_src","log")
#params.transform("regparam_src","log")  # only use if regularizing the shapelets

#params.set_limits([
    #("log(regparam_src)",0,7)  # only if regularizing the shapelets
#])

pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

q.gradtol = 1.0  # tolerance for convergence of gradient in BFGS method
q.nrepeat = 0

q.run_fit("bfgs",adopt=True,show_errors=True)
q.sbmap_invert()

pause()

plotimg(q,nres=True,title="Residuals from best-fit model")
src.mkpixsrc()
plotsrc(q,title="Reconstructed source from best-fit model")
pause()

q.param_markers("1.36 1.15 0.01 -0.02 0.01 0.0036 0.06 -0.05") # true model lens parameters
q.mkposts(fisher=True)

#plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing


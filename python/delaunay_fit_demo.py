from qlens_helper import *

pause()
cosmo = Cosmology(omega_m=0.3,hubble=0.7)
q = QLens(cosmo)
(lens,src,ptsrc,pixsrc,imgdata,params,dparams) = q.objects();

q.fit_label = 'delaunay_fit'

sim_hst_data = imgdata.load("sim_hst_image.fits",band=0,pixsize=0.049)
sim_hst_data.load_noise_map("demo_noisemap.fits")
sim_hst_data.load_mask("mask.fits")

#sim_hst_data.unmask_all_pixels()
#sim_hst_data.mask_low_sn_pixels(threshold=0.01)   # note, the signal threshold for masking out a pixel is given by 'threshold' times the noise dispersion for that pixel
#sim_hst_data.trim_mask_windows(noise_threshold=4,npixel_threshold=20)    # if npixel_threshold is omitted, it is set to zero by default
#sim_hst_data.unmask_neighbor_pixels()
#dataimg = sim_hst_data.plot()  # keyword arguments for this plotting: nomask, fgmask, or emask (all are 'False' by default)
#plot_sb(dataimg,q)
#pause()

q.sbmap_load_psf("hst_psf.fits")

q.split_imgpixels = False

q.sci_notation = True
q.shear_components=True

q.zlens = 0.5
q.zsrc = 2

#Alpha = SPLE({"b": 1.3634, "alpha": 1.17163, "s": 0.0, "q": 0.963867, "theta": 81.9, "xc": 0.0102892, "yc": 0.00358392}) # true model
Alpha = SPLE({"b": 1.35, "alpha": 1.1, "s": 0.0, "q": 0.90, "theta": 80, "xc": 0.01, "yc": 0})
Alpha.vary([1,1,0,1,1,1,1])
#Alpha.set_limits([     # if you are doing nested sampling, you can define limits within the lens object, *or*
    #("b",4,6),         # you can define limits using the 'params' object instead (see below after lens.add)
    #("q",0.2,1),
    #("theta",20,155),
    #("xc",0.3,1.3),
    #("yc",0,0.6)
#])


#extshear = Shear({"shear1": 0.0647257, "shear2": -0.0575047}) # true model
extshear = Shear({"shear1": 0.05, "shear2": -0.03})
extshear.vary([1,1,0,0])

lens.add(Alpha,shear=extshear)

#params.set_limits([      # We can define prior limits here if we prefer (instead of defining them for each lens object above).
    #("b",0.7,1.8),       # One advantage is that if we transform a parameter, you can define your limits in terms of the transformed parameter (e.g. log(mass)).
    #("alpha",0.6,1.6),
    #("q",0.2,1),
    #("theta",20,155),
    #("xc",0.3,1.3),
    #("yc",0,0.6),
    #("shear1",-0.2,0.2),
    #("shear2",-0.2,0.2)
#])

q.nimg_prior=True
q.nimg_threshold=1.4
q.outside_sb_prior=True
q.set_source_mode("delaunay")
q.regularization_method="gradient"
q.optimize_regparam=True
#pixsrc.add()
#pixsrc.update("regparam",5.19292)

#q.sbmap_invert()
#q.fit_chisq()

q.chisqtol = 1e-5

q.fitmodel()

q.sbmap_invert()
#imgdata[0].unmask_all_pixels()

img = q.plotimg(nres=True)
plot_sb(img,q,include_cc=True)

pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

srcplt = pixsrc[0].plot(interp=True)
plot_sb(srcplt,q)

pause()

#q.n_livepts = 300
q.run_fit("simplex",adopt=True,show_errors=False)

q.sbmap_invert()

img = q.plotimg(nres=True)
plot_sb(img,q,include_cc=True)

srcplt = pixsrc[0].plot()
plot_sb(srcplt,q)
#img = q.plotimg(nres=True)
#plot_sb(img)

#q.run_fit("multinest",adopt=True)
#img = q.plotimg(nres=True)
#plot_sb(img)

#plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing


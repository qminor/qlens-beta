from qlens_helper import *

cosmo = Cosmology(omega_m=0.3,hubble=0.7)
q = QLens(cosmo)
(lens,src,ptsrc,pixsrc,imgdata,params,dparams) = q.objects();

q.fit_label = 'sbprofile_demo'

q.set_img_npixels(81,81)
q.set_grid_from_imgpixels(pixsize=0.05)

q.sbmap_load_psf("hst_psf.fits")

q.split_imgpixels = True
q.imgpixel_nsplit = 4

q.sci_notation = True
q.shear_components=True

q.zlens = 0.5
q.zsrc = 2

Alpha = SPLE({"b": 1.35, "alpha": 1.1, "s": 0.0, "q": 0.70, "theta": 80, "xc": 0.01, "yc": 0})
Alpha.vary([1,1,0,1,1,1,1])

extshear = Shear({"shear1": 0.05, "shear2": -0.03})
extshear.vary([1,1,0,0])

lens.add(Alpha,shear=extshear)
plotcrit(q)

sersic_src = Sersic({"s0": 7, "R_eff": 0.087, "n": 2.5, "q": 0.90, "theta": 80, "xc": 0.06, "yc": -0.03})

src.add(sersic_src)

q.nimg_prior=True
q.nimg_threshold=1.4
q.outside_sb_prior=True
q.set_source_mode("sbprofile")

q.bg_pixel_noise = 0.03
q.simulate_pixel_noise = True

dparams.add("xi",q.zsrc)
dparams.print()

img = q.plotimg()
plot_sb(img,q,include_cc=True)

pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

q.plotimg(output_fits="sbprofile_mock_data.fits")



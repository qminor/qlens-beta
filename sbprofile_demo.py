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
q.shear_components=False

q.zlens = 0.222
q.zsrc = 0.8

stars = SersicLens({"Mstar": 2e11, "R_eff": 0.43, "n": 4, "q": 0.49, "theta": 0, "xc": 0.01, "yc": 0.02},pmode=1,qlens=q) # by passing in q=qlens, it gives lens object access to both the cosmology and also q.zlens and q.zsrc_ref
dm_halo = NFW({"mvir": 1.1e13, "q": 0.35, "theta": -15},pmode=1,c_median=True,qlens=q)
extshear = Shear({"shear": 0.21, "theta": -18},qlens=q)

lens.add(stars,shear=extshear)
lens.add(dm_halo,anchor_center=0)

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



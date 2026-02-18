from qlens_helper import *

q = QLens()

(lens,ptsrc,ptimgdata) = q.ptimg_objects()   # this is so we can enter 'lens' instead of 'q.lens', 'ptsrc' instead of 'q.ptsrc', etc.
(params,dparams) = q.param_objects()         # same as above; we can enter 'params' instead of 'q.params', etc.

ptimgdata.load("doublesrc.dat")
lens.add(SPLE({"b": 4.5, "alpha": 1, "s": 0.0, "q": 0.8, "theta": 30, "xc": 0.7, "yc": 0.3}))
lens[0].vary([1,0,0,1,1,1,1])
lens.add(Shear({"shear": 0.02, "theta": 10, "xc": 0.7, "yc": 0.3}))
lens[1].anchor_center(0)
lens[1].vary([1,1])

q.analytic_bestfit_src = True
q.central_image = False
q.imgplane_chisq = True
q.chisqtol = 1e-6
q.nrepeat = 2
q.fitmodel()
q.imgdata_display()
pause()

q.run_fit("simplex")
q.use_bestfit()

plot_fit_ptimgs(q) # plot_fit_ptimgs returns the source and image figures, so you can also do
                # (srcfig, imgfig) = plot_fit_ptimgs(q,showplot=False) and modify the figures

#plt.show()

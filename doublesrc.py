from qlens_helper import *

q = QLens()

q.imgdata_read("doublesrc.dat")
q.imgdata_display()
q.add_lens(Alpha({"b": 4.5, "alpha": 1, "s": 0.0, "q": 0.8, "theta": 30, "xc": 0.7, "yc": 0.3}))
q.lens[0].setvary([1,0,0,1,1,1,1])
q.add_lens(Shear({"shear": 0.02, "theta": 10, "xc": 0.7, "yc": 0.3}))
q.lens[1].anchor_center(0)
q.lens[1].setvary([1,1])

q.analytic_bestfit_src = True
q.central_image = False
q.imgplane_chisq = True
q.chisqtol = 1e-6
q.nrepeat = 2
q.fitmodel()
pause()

q.run_fit("simplex")
q.use_bestfit()
fit_plotimg(q) # fit_plotimg returns the source and image figures, so you can also do
                # (srcfig, imgfig) = fit_plotimg(q,showplot=False) and modify the figures

#plt.show()

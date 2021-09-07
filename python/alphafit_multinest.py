from qlens_helper import *

q = QLens()

q.sci_notation = False
q.imgdata_read("../data/alphafit.dat")
#q.imgdata_display()
q.add_lens(Alpha({"b": 4.5, "alpha": 1, "s": 0.0, "q": 0.8, "theta": 30, "xc": 0.7, "yc": 0.3}))
q.lens[0].setvary([1,0,0,1,1,1,1])
q.lens[0].set_prior_limits("b",4,6)
q.lens[0].set_prior_limits("q",0.2,1)
q.lens[0].set_prior_limits("theta",20,155)
q.lens[0].set_prior_limits("xc",0.3,1.3)
q.lens[0].set_prior_limits("yc",0,0.6)

q.add_lens(Shear({"shear": 0.02, "theta": 10, "xc": 0.7, "yc": 0.3}))
q.lens[1].anchor_center(0)
q.lens[1].setvary([1,1])
q.lens[1].set_prior_limits("shear",0.01,0.2)
q.lens[1].set_prior_limits("theta_shear",1,100)

q.central_image = False
q.imgplane_chisq = False
q.analytic_bestfit_src = True
q.flux_chisq = True
q.chisqtol = 1e-6
#q.set_sourcepts_auto()
q.nrepeat = 2
#print("Fit model:")
q.fitmodel()
pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

q.n_livepts = 300
q.run_fit("multinest")
q.use_bestfit()
fit_plotimg(q) # fit_plotimg returns the source and image figures, so you can also do
                # (srcfig, imgfig) = fit_plotimg(q,showplot=False) and modify the figures

plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing

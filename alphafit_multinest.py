from qlens_helper import *

cosmo = Cosmology(omega_m=0.3,hubble=0.7)
q = QLens(cosmo)
(lens,src,ptsrc,pixsrc,imgdata,params,dparams) = q.objects();

q.fit_label = 'alpha_multinest'

q.sci_notation = False
q.imgdata_read("../data/alphafit.dat")
#q.imgdata_display()

q.shear_components=True
q.ellipticity_components=False

Alpha = SPLE({"b": 1.35, "gamma": 2, "s": 0.0, "q": 0.8, "theta": 80, "xc": 0.05, "yc": 0.03},pmode=1)
Alpha.vary([1,0,0,1,1,1,1])
Alpha.set_limits([
    ("b",4,6),
    ("q",0.2,1),
    ("theta",20,155),
    ("xc",0.3,1.3),
    ("yc",0,0.6)
])

extshear = Shear({"shear1": 0.03, "shear2": -0.05})
extshear.vary([1,1,0,0])
extshear.set_limits([
    ("shear1",-0.2,0.2),
    ("shear2",-0.2,0.2)
])

lens.add(Alpha,shear=extshear)

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
q.run_fit("multinest",adopt=True,resume=False)
#q.use_bestfit()
#q.adopt_chain_bestfit()
fit_plotimg(q) # fit_plotimg returns the source and image figures, so you can also do
                # (srcfig, imgfig) = fit_plotimg(q,showplot=False) and modify the figures

plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing

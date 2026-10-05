from qlens_helper import *

cosmo = Cosmology(omega_m=0.3,hubble=0.7)
q = QLens(cosmo)
(lens,ptsrc,ptimgdata) = q.ptimg_objects()   # this is so we can enter 'lens' instead of 'q.lens', 'ptsrc' instead of 'q.ptsrc', etc.
(params,dparams) = q.param_objects()         # same as above; we can enter 'params' instead of 'q.params', etc.

show_commands()

q.fit_label = 'alpha_bfgs'
q.sci_notation = False

ptimgdata.load("alphafit.dat") # this function will become obsolete once the ptimgdata class is wrapped
print(ptimgdata)               # Note: when in interactive mode you can just type 'ptimgdata' to print the data

q.shear_components=True
q.ellipticity_components=True

Alpha = SPLE({"b": 4.5, "alpha": 1, "s": 0.0, "e1": 0.0, "e2": 0.0, "xc": 0.7, "yc": 0.4}, qlens=q)
Alpha.vary(["b","e1","e2","xc","yc"])

extshear = Shear({"shear1": 0.0, "shear2": 0.0},qlens=q)
extshear.vary(["shear1","shear2"])

lens.add(Alpha,shear=extshear)
print("Lenses:",lens,"\n")

#The BFGS method in qlens requires parameter limits, which we impose here using the 'params' object
params.set_limits([
    ("b",4,6),
    ("e1",-0.5,0.5),
    ("e2",-0.5,0.5),
    ("xc",0.3,1.3),
    ("yc",0,0.6),
    ("shear1",-0.2,0.2),
    ("shear2",-0.2,0.2),
    ("xsrc",-2,2),
    ("ysrc",-2,2)
])

q.central_image = False
q.analytic_bestfit_src = False
q.imgplane_chisq = True

q.flux_chisq = True
q.gradtol = 1e-3
q.chisqtol = 1e-6
q.nrepeat = 0
#print("Fit model:")
#q.fitmodel()
pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

q.run_fit("bfgs",adopt=True)

#q.adopt_chain_bestfit()

plot_fit_ptimgs(q) # plot_fit_ptimgs returns the source and image figures, so you can also do
                # (srcfig, imgfig) = plot_fit_ptimgs(q,showplot=False) and modify the figures

#plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing

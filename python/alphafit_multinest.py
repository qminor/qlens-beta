from qlens_helper import *

cosmo = Cosmology(omega_m=0.3,hubble=0.7)
q = QLens(cosmo)
(lens,ptsrc,ptimgdata) = q.ptimg_objects()   # this is so we can enter 'lens' instead of 'q.lens', 'ptsrc' instead of 'q.ptsrc', etc.
(params,dparams) = q.param_objects()         # same as above; we can enter 'params' instead of 'q.params', etc.

q.fit_label = 'alpha_multinest'
q.sci_notation = False

ptimgdata.load("alphafit.dat") # this function will become obsolete once the ptimgdata class is wrapped
print(ptimgdata)               # Note: when in interactive mode you can just type 'ptimgdata' to print the data

q.shear_components=True
q.ellipticity_components=False

Alpha = SPLE({"b": 4.5, "alpha": 1, "s": 0.0, "q": 0.8, "theta": 30, "xc": 0.7, "yc": 0.3},pmode=1)
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
print("Lenses:",lens,"\n")

q.central_image = False
q.imgplane_chisq = False
q.analytic_bestfit_src = True
q.flux_chisq = True
q.chisqtol = 1e-6
q.nrepeat = 2
#print("Fit model:")
#q.fitmodel()
pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

q.n_livepts = 300
q.run_fit("multinest",adopt=True,resume=False)

#q.adopt_chain_bestfit()

plot_fit_ptimgs(q) # plot_fit_ptimgs returns the source and image figures, so you can also do
                # (srcfig, imgfig) = plot_fit_ptimgs(q,showplot=False) and modify the figures
q.mkposts()

(median, lowpct, highpct) = q.get_parameter_percentiles(get_2sigma=False)

#plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing

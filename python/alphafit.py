from qlens_helper import *

cosmo = Cosmology(omega_m=0.3,hubble=0.7)
q = QLens(cosmo)
(lens,src,ptsrc,pixsrc,imgdata,params,dparams) = q.objects();

q.sci_notation = False
q.imgdata_read("alphafit.dat") # this function will become obsolete once the ptimgdata class is wrapped
q.imgdata_display() # this function will become obsolete once the ptimgdata class is wrapped

sple = SPLE({"b": 4.5, "alpha": 1, "s": 0.0, "q": 0.8, "theta": 30, "xc": 0.7, "yc": 0.3})
sple.vary([1,0,0,1,1,1,1])
extshear = Shear({"shear": 0.02, "theta": 10})
extshear.vary([1,1,0,0])
lens.add(sple,shear=extshear)

q.central_image = False
q.imgplane_chisq = True
q.flux_chisq = True
q.chisqtol = 1e-6
pause()
q.analytic_bestfit_src = True
#q.set_sourcepts_auto()
q.nrepeat = 2
print("Fit model:")
q.fitmodel()
pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

# The following two lines must be run before evaluating the likelihood
#q.setup_fitparams(True) # This sets up the fit parameters
#print(q.fitparams())  # This will return a python list giving the fit parameters 
#q.init_fitmodel() # This makes a copy of the model (lenses, sources, etc.) that can be varied during a fit (this is called the "fitmodel" object)
#logl = q.LogLike(params.values())
#print("initial params:")
#params
#print("initial loglike: ",logl)
#pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

q.run_fit("simplex",adopt=False)

# Note, when qlens runs an optimization, it destroys the "fitmodel" object at the end. So you'd have to do another q.init_fitmodel() if you want to
# evaluate the likelihood again with LogLike(...)
#q.init_fitmodel()
#logl = q.LogLike(params.values())
#print("loglike: ",logl)
#print("final params:")
#params
#print(q.fitparams())

q.adopt_model(params.values()) # this will adopt the best-fit model, since fitparams() returns the fit parameters at the end of the optimization
# note that if you're doing your own optimization, you might have the best-fit parameters stored in a separate
# python list (or numpy array?) of your own; in this case, replace q.fitparams() in the above line with your own list


#q.use_bestfit()
fit_plotimg(q) # fit_plotimg returns the source and image figures, so you can also do
                # (srcfig, imgfig) = fit_plotimg(q,showplot=False) and modify the figures

#plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing

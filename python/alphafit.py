from qlens_helper import *
#from qlens import *

q = QLens()

q.sci_notation = False
q.imgdata_read("alphafit.dat")
q.imgdata_display()
q.add_lens(SPLE_Lens({"b": 4.5, "alpha": 1, "s": 0.0, "q": 0.8, "theta": 30, "xc": 0.7, "yc": 0.3}))
q.lens[0].setvary([1,0,0,1,1,1,1])
q.add_lens(Shear({"shear": 0.02, "theta": 10, "xc": 0.7, "yc": 0.3}))
q.lens[1].anchor_center(0)
q.lens[1].setvary([1,1])
#q.print_hi()

q.central_image = False
q.imgplane_chisq = True
q.flux_chisq = True
q.chisqtol = 1e-6
q.analytic_bestfit_src = True
#q.set_sourcepts_auto()
q.nrepeat = 2
print("Fit model:")
q.fitmodel()

# The following two lines must be run before evaluating the likelihood
q.setup_fitparams(True) # This sets up the fit parameters
q.fitparams()  # This will return a python list giving the fit parameters 
q.init_fitmodel() # This makes a copy of the model (lenses, sources, etc.) that can be varied during a fit (this is called the "fitmodel" object)
q.LogLike(q.fitparams())
pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

q.run_fit("simplex")  # You can replace this with your own optimization!
#q.use_bestfit() # only use if you're using a qlens optimizer; otherwise, use the line below

# Note, when qlens runs an optimization, it destroys the "fitmodel" object at the end. So you'd have to do another q.init_fitmodel() if you want to
# evaluate the likelihood again with LogLike(...)
q.init_fitmodel()
q.LogLike(q.fitparams())

q.adopt_model(q.fitparams()) # this will adopt the best-fit model, since fitparams() returns the fit parameters at the end of the optimization
# note that if you're doing your own optimization, you might have the best-fit parameters stored in a separate
# python list (or numpy array?) of your own; in this case, replace q.fitparams() in the above line with your own list

fit_plotimg(q) # from qlens_helper: fit_plotimg returns the source and image figures, so you can also do
                # (srcfig, imgfig) = fit_plotimg(q,showplot=False) and modify the figures

plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing

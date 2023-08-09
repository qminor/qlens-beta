from qlens_helper import *

q = QLens()

q.sci_notation = False
q.imgdata_read("alphafit.dat")
q.imgdata_display()
q.add_lens(Alpha({"b": 4.5, "alpha": 1, "s": 0.0, "q": 0.8, "theta": 30, "xc": 0.7, "yc": 0.3}))
q.lens[0].setvary([1,0,0,1,1,1,1])
q.add_lens(Shear({"shear": 0.02, "theta": 10, "xc": 0.7, "yc": 0.3}))
q.lens[1].anchor_center(0)
q.lens[1].setvary([1,1])
q.print_hi()

q.central_image = False
q.imgplane_chisq = True
q.flux_chisq = True
q.chisqtol = 1e-6
q.set_sourcepts_auto()
q.nrepeat = 2
print("Fit model:")
q.fitmodel()
pause() # note, pause will be ignored if script is not run in interactive mode (with '-i' parameter)

q.run_fit("simplex")
q.use_bestfit()
fit_plotimg(q) # fit_plotimg returns the source and image figures, so you can also do
                # (srcfig, imgfig) = fit_plotimg(q,showplot=False) and modify the figures

#plt.show() # If you're not running in interactive mode, this makes matplotlib still show the plots after finishing

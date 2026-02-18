from qlens_helper import *

q = QLens()
(lens,ptsrc,ptimgdata) = q.ptimg_objects()

lens.add(SPLE({"b":10,"alpha":1,"s":0.5,"q":0.7,"theta":90},qlens=q))

q.set_grid_corners(-18,18,-18,18) # Not essential, but it nicely zooms in the pixel image plots

# Now we create a satellite galaxies with a dPIE profile (which is isothermal with
# a tidal break radius):
lens.add([
    dPIE({"b":0.06,"a":1,"s":0.001,"q":1,"xc":-0.3,"yc":0.4},qlens=q),
    dPIE({"b":0.05,"a":1,"s":0.001,"q":1,"xc":-0.2,"yc":0.2},qlens=q),
    dPIE({"b":1.0,"a":1.9,"s":0.1,"q":1,"xc":5.0,"yc":1.0},qlens=q),
    dPIE({"b":0.2,"a":1,"s":0.01,"q":1,"xc":2.0,"yc":4.3},qlens=q),
    dPIE({"b":0.1,"a":1,"s":0.01,"q":1,"xc":0.9,"yc":-3},qlens=q),
    dPIE({"b":0.06,"a":1,"s":0.01,"q":1,"xc":-7,"yc":2},qlens=q),
    dPIE({"b":0.3,"a":1,"s":0.01,"q":1,"xc":-10,"yc":7},qlens=q),
])

# Plot the grid so you can see what it looks like:
plot_ptimg_grid(q,lw=0.5) # smaller line thickness so we can see the galaxy subgridding more easily
plot_ptimgs(-1.2,0.1,q)

# Now let's show the same thing but for a surface brightness map:
q.set_img_npixels(300,300)
q.set_src_npixels(50,50)
q.src.add(Gaussian({"sbmax":1.0,"sigma":1.0,"q":0.7,"xc":-1.4,"yc":-0.7}))
q.src.mkpixsrc(npix=600) # npix=200 is the default if npix argument is not given, i.e. if you do src.mkpixsrc()
#source gaussian 1 1 0.7 0 -1.4 -0.7
#adaptive_grid on
plotsrc(q)
plotimg(q)
#sbmap makesrc
#sbmap plotimg

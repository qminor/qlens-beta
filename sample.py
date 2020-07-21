from qlens import *
L = QLens()
A = Alpha()
S = Shear({ # A way to initialize lens object
    "shear": 0.02,
    "theta": 10,
    "xc": 0,
    "yc": 0
})
I = ImageSet()

L.imgdata_read("alphafit.dat")

A.initialize({ # Another way to initialize the values
    "b": 4.5,
    "alpha": 1,
    "s": 0,
    "q": 0.8,
    "theta": 30,
    "xc": 0.7,
    "yc": 0.3
})
A.set_vary_flags([1,0,0,1,1,1,1])

S.set_vary_flags([1,1,0,0])

L.add_lenses([A, S])
L.include_flux_chisq(True)
L.use_image_plane_chisq(True)
# L.add_lenses([(A, 0.5, 1), (S, 0.5, 1)]) # Same command as above, default args zl=0.5, zs=1

L.run_fit("simplex")

L.get_imageset(I, 0.5, 0.1, False)

# print("\n\nNumber of images: ", I.n_images)

# print("src_x: \t\t\tsrc_y: \t\t\tmag:")
# for i in I.images:
#     print("{}\t{}\t{}".format(i.pos.src()[0], i.pos.src()[1], i.mag))

L.plot_sorted_critical_curves("s.temp")

sources_x = []
sources_y = []

for i in L.sorted_critical_curve:
    print("Length of current cell: {}".format(len(i.length_of_cell)))
    for j in i.cc_pts:
        sources_x.append(j.src()[0]) # x coordinate
        sources_y.append(j.src()[1]) # y coordinate
        

import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(sources_x, sources_y, color='r', s=3) # s is the size of the points
ax.set_xlabel('X Coordinates')
ax.set_ylabel('Y Coordinates')
ax.set_title('Critical Curve Points')
plt.grid(True)
plt.show()
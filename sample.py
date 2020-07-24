from qlens import *
import matplotlib.pyplot as plt

L = QLens()
A = Alpha()
B = Alpha()
C = Alpha()
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

images_x = []
images_y = []

L.get_imageset(I, 0.5, 0.1, False)

# print("\n\nNumber of images: ", I.n_images)

# print("src_x: \t\t\tsrc_y: \t\t\tmag:")
print(I.caustic.src())

for i in I.images:
    print(i.pos.src())
    images_x.append(i.pos.src()[0])
    images_y.append(i.pos.src()[1])

L.plot_sorted_critical_curves("s.temp")


fig=plt.figure()
fig.suptitle("Critical Curve")
ax=fig.add_axes([0,0,1,1], label='Inline label')
## Plotting the critical curves
for i in L.sorted_critical_curve:
    sources_x = []
    sources_y = []

    for j in i.cc_pts:
        sources_x.append(j.src()[0]) # x coordinate
        sources_y.append(j.src()[1]) # y coordinate

    # Required to connect the curve continously

    sources_x.append(sources_x[0])
    sources_y.append(sources_y[0]) 

    # ax.scatter(sources_x, sources_y, color='r', s=3) # s is the size of the points
    ax.plot(sources_x, sources_y, color='k', label='Critical Curve')
    ax.plot(images_x, images_y, 'o', color='b', label='Images')
    plt.legend(loc="upper left")

    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_title('Critical Curve Points')
    sources_x = []
    sources_y = []

    plt.grid(True)


fig=plt.figure()
fig.suptitle("Caustic Curve")
ax=fig.add_axes([0,0,1,1])

## Plotting the caustic curves
for i in L.sorted_critical_curve:
    sources_x = []
    sources_y = []

    for j in i.caustic_pts:
        sources_x.append(j.src()[0]) # x coordinate
        sources_y.append(j.src()[1]) # y coordinate

    # Required to connect the curve continously

    sources_x.append(sources_x[0])
    sources_y.append(sources_y[0]) 

    # ax.scatter(sources_x, sources_y, color='r', s=3) # s is the size of the points
    ax.plot(sources_x, sources_y, color='k', label='Caustic Curve')
    ax.plot(I.caustic.src()[0], I.caustic.src()[1], 'o', color='b', label='Source')
    plt.legend(loc="upper left")

    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_title('Caustic Points')
    sources_x = []
    sources_y = []

    plt.grid(True)

plt.show()

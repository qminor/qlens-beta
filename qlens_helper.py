from qlens import *
import matplotlib.pyplot as plt

def plot_crit_and_caustic(QLens_Object, ImageSet_Object):
    if(QLens_Object is None or ImageSet_Object is None):
        raise RuntimeError("This function requires the first input to be a QLens object and the second input to be an ImageSet object.")

    L = QLens_Object
    I = ImageSet_Object

    sources_x = []
    sources_y = []

    for i in I:
        # print(i.n_images)
        sources_x.append(i.src.pos()[0])
        sources_y.append(i.src.pos()[1])

    L.plot_sorted_critical_curves("sample.a")

    fig=plt.figure()
    fig.suptitle("Critical Curve")
    ax=fig.add_axes([0,0,1,1], label='Inline label')
    ## Plotting the critical curves
    
    for i in L.sorted_critical_curve:
        cc_x = []
        cc_y = []

        for j in i.cc_pts:
            cc_x.append(j.pos()[0]) # x coordinate
            cc_y.append(j.pos()[1]) # y coordinate

        # Required to connect the curve continously

        cc_x.append(sources_x[0])
        cc_y.append(sources_y[0]) 

        # ax.scatter(sources_x, sources_y, color='r', s=3) # s is the size of the points
        ax.plot(cc_x, cc_y, color='k', label='Critical Curve')
        ax.plot(sources_x, sources_y, 'o', color='b', label='Images')
        plt.legend(loc="upper left")

        ax.set_xlabel('X Coordinates')
        ax.set_ylabel('Y Coordinates')
        ax.set_title('Critical Curve Points')
        cc_x = []
        cc_y = []

        plt.grid(True)


    fig=plt.figure()
    fig.suptitle("Caustic Curve")
    ax=fig.add_axes([0,0,1,1])

    ## Plotting the caustic curves
    for i in L.sorted_critical_curve:
        sources_x = []
        sources_y = []

        for j in i.caustic_pts:
            sources_x.append(j.pos()[0]) # x coordinate
            sources_y.append(j.pos()[1]) # y coordinate

        # Required to connect the curve continously

        sources_x.append(sources_x[0])
        sources_y.append(sources_y[0]) 

        ax.plot(sources_x, sources_y, color='k', label='Caustic Curve')
        
    caustic_x = []
    caustic_y = []
    for i in I:
        caustic_x.append(i.src.pos()[0])
        caustic_y.append(i.src.pos()[1])
    ax.plot(caustic_x, caustic_y, 'o', color='b', label='Source')
    plt.legend(loc="upper left")

    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_title('Caustic Points')
    sources_x = []
    sources_y = []

    plt.grid(True)

    plt.show()

import itertools
from qlens import *
import matplotlib.pyplot as plt
import code
import copy
import inspect
import sys

def pause():
    if bool(getattr(sys, 'ps1', sys.flags.interactive))==True:   # will ignore the pause if not in interactive mode to begin with
        frame = inspect.currentframe().f_back
        code.interact(local=frame.f_locals,banner="Pausing for input...press <Ctrl-d> to continue the script",exitmsg="")

def fit_plotimg(QLens_Object, include_cc=True, show=True, grid=False):
    if(QLens_Object is None):
        raise RuntimeError("This function requires the first input to be a QLens object.")

    q = QLens_Object
    I = q.get_fit_imagesets()
    D = q.get_data_imagesets()

    if (len(I) != len(D)):
        raise RunTimeError("Data and model do not have the same number of imagesets")

    images_x = [[] for i in range(len(I))]
    images_y = [[] for i in range(len(I))]
    data_images_x = [[] for i in range(len(I))]
    data_images_y = [[] for i in range(len(I))]

    k=0
    for i,d in zip(I,D):
        for j in range(i.n_images):
            images_x[k].append(i.images[j].pos.x)
            images_y[k].append(i.images[j].pos.y)
        for j in range(d.n_images):
            data_images_x[k].append(d.images[j].pos.x)
            data_images_y[k].append(d.images[j].pos.y)
        k += 1

    imgplane_fig=plt.figure()
    imgplane_ax=plt.gca()
    ## Plotting the critical curves
    markers = itertools.cycle(['o','s','v']) 
    markers_source = itertools.cycle(['o','s','v']) 
    for j in range(len(images_x)):
        mark=next(markers)
        imgplane_ax.plot(images_x[j], images_y[j], marker=mark, linestyle='None', color='b', label='Model (z=' + str(D[j].zsrc) + ')')
        imgplane_ax.plot(data_images_x[j], data_images_y[j], marker=mark, linestyle='None', color='g', label='Data (z=' + str(D[j].zsrc) + ')')
    plt.legend(loc="upper right")
    if grid==True:
        plt.grid(True)

    srcplane_fig=plt.figure()
    srcplane_ax=plt.gca()

    sources_x = []
    sources_y = []
    for i in I:
        sources_x.append(i.pos.x)
        sources_y.append(i.pos.y)
    for j in range(len(sources_x)):
        srcplane_ax.plot(sources_x[j], sources_y[j], marker=next(markers_source), linestyle='None', color='b', label='Source (z=' + str(D[j].zsrc) + ')')

    plt.legend(loc="upper right")
    if grid==True:
        plt.grid(True)

    if (include_cc==True):
        add_crit_to_plot(q,srcplane_fig,imgplane_fig)

    if (show==True):
        plt.show(block=False)

    return (srcplane_fig,imgplane_fig)

def plotcrit(QLens_Object, show=True, grid=False):
    if(QLens_Object is None):
        raise RuntimeError("This function requires the first input to be a QLens object.")

    q = QLens_Object

    imgplane_fig=plt.figure()
    imgplane_ax=plt.gca()
    if grid==True:
        plt.grid(True)

    srcplane_fig=plt.figure()
    srcplane_ax=plt.gca()
    if grid==True:
        plt.grid(True)

    add_crit_to_plot(q,srcplane_fig,imgplane_fig)

    if (show==True):
        plt.show(block=False)

    return (srcplane_fig,imgplane_fig)

def add_crit_to_plot(QLens_Object, srcplane_fig, imgplane_fig):
    if(QLens_Object is None or srcplane_fig is None or imgplane_fig is None):
        raise RuntimeError("This function requires the first input to be a QLens object, and the 2nd and 3rd inputs to be matplotlib ax objects (srcplane,imgplane).")

    L = QLens_Object

    L.sort_critical_curves()

    plt.figure(imgplane_fig.number)
    imgplane_ax=plt.gca()
    ## Plotting the critical curves

    for last_cc in L.sorted_critical_curve:
        pass
    label=''
    for curve in L.sorted_critical_curve:
        cc_x = []
        cc_y = []

        for point in curve.cc_pts:
            cc_x.append(point.x) # x coordinate
            cc_y.append(point.y) # y coordinate

        # Required to connect the curve continously

        cc_x.append(cc_x[0])
        cc_y.append(cc_y[0]) 

        # imgplane_ax.scatter(sources_x, sources_y, color='r', s=3) # s is the size of the points
        if (curve == last_cc):
            label='Critical Curve ($z_{src}$=' + str(L.zsrc) + ')'
        imgplane_ax.plot(cc_x, cc_y, color='k', label=label)
        cc_x = []
        cc_y = []

    plt.legend(loc="upper right")

    plt.figure(srcplane_fig.number)
    srcplane_ax=plt.gca()

    ## Plotting the caustic curves
    label=''
    for curve in L.sorted_critical_curve:
        caustic_x = []
        caustic_y = []

        for point in curve.caustic_pts:
            caustic_x.append(point.x) # x coordinate
            caustic_y.append(point.y) # y coordinate

        # Required to connect the curve continously

        caustic_x.append(caustic_x[0])
        caustic_y.append(caustic_y[0]) 

        if (curve == last_cc):
            label='Caustic ($z_{src}$=' + str(L.zsrc) + ')'
        srcplane_ax.plot(caustic_x, caustic_y, color='k', label=label)
        
    plt.legend(loc="upper right")

def add_crit_to_imgplot(QLens_Object, imgplane_fig):
    if(QLens_Object is None or imgplane_fig is None):
        raise RuntimeError("This function requires the first input to be a QLens object, and the 2nd input to be matplotlib ax objects (imgplane).")

    L = QLens_Object

    L.sort_critical_curves()

    plt.figure(imgplane_fig.number)
    imgplane_ax=plt.gca()
    ## Plotting the critical curves

    for last_cc in L.sorted_critical_curve:
        pass
    label=''
    for curve in L.sorted_critical_curve:
        cc_x = []
        cc_y = []

        for point in curve.cc_pts:
            cc_x.append(point.x) # x coordinate
            cc_y.append(point.y) # y coordinate

        # Required to connect the curve continously

        cc_x.append(cc_x[0])
        cc_y.append(cc_y[0]) 

        # imgplane_ax.scatter(sources_x, sources_y, color='r', s=3) # s is the size of the points
        if (curve == last_cc):
            label='Critical Curve ($z_{src}$=' + str(L.zsrc) + ')'
        imgplane_ax.plot(cc_x, cc_y, color='k', label=label)
        cc_x = []
        cc_y = []

    plt.legend(loc="upper right")

def add_caustics_to_srcplot(QLens_Object, srcplane_fig):
    if(QLens_Object is None or srcplane_fig is None):
        raise RuntimeError("This function requires the first input to be a QLens object, and the 2nd input to be matplotlib ax objects (srcplane).")

    L = QLens_Object

    L.sort_critical_curves()

    for last_cc in L.sorted_critical_curve:
        pass

    plt.figure(srcplane_fig.number)
    srcplane_ax=plt.gca()

    ## Plotting the caustic curves
    label=''
    for curve in L.sorted_critical_curve:
        caustic_x = []
        caustic_y = []

        for point in curve.caustic_pts:
            caustic_x.append(point.x) # x coordinate
            caustic_y.append(point.y) # y coordinate

        # Required to connect the curve continously

        caustic_x.append(caustic_x[0])
        caustic_y.append(caustic_y[0]) 

        if (curve == last_cc):
            label='Caustic ($z_{src}$=' + str(L.zsrc) + ')'
        srcplane_ax.plot(caustic_x, caustic_y, color='k', label=label)
        
    plt.legend(loc="upper right")

def checknan(img):
    foundnan = False
    for row in img[3]:
        for x in row:
            if (x*0.0 != 0.0):
                print("NAN value!")
                foundnan = True
    if (foundnan==False):
        print("Did not find any NAN's")

def plot_sb(img, QLens_Object, show_cc=True, fix_limits_before_cc=True):
    q = QLens_Object
    plottype = img[0]
    # For some reason, if we don't make copies of the lists from the tuple (as below), the tuple ends up getting corrupted when matplotlib stuff is called. I have no idea why, it's super annoying.
    x = copy.deepcopy(img[1])
    y = copy.deepcopy(img[2])
    z = copy.deepcopy(img[3])

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [x[0],x[-1],y[0],y[-1]] 
    im = ax.imshow(z, interpolation='nearest', extent=extent, cmap='viridis', origin='lower')
    fig.colorbar(im, ax=ax)
    if (fix_limits_before_cc):
        xlim_before = ax.get_xlim()
        ylim_before = ax.get_ylim()

    if (plottype=="imgplane" and show_cc==True):
        add_crit_to_imgplot(q,fig)
    if (plottype=="srcplane" and show_cc==True):
        add_caustics_to_srcplot(q,fig)
    if (fix_limits_before_cc):
        ax.set_xlim(xlim_before)
        ax.set_ylim(ylim_before)
    plt.show(block=False)

def plotimg(QLens_Object, src=-1, show_cc=True, nomask=False, nres=False, res=False, output_fits=""):
    q = QLens_Object
    img = q.plotimg(src=src,nres=nres,res=res,nomask=nomask,output_fits=output_fits)
    if (output_fits==""):
        if (show_cc==True and src >= 0):
            q.mkgrid_extended_src(src)
        else:
            if (show_cc==True and src < 0):
                q.mkgrid()
        plot_sb(img,q,show_cc=show_cc)

def plotsrc(QLens_Object, show_cc=True, interp=False, fix_limits_before_cc=True, src=0):
    q = QLens_Object
    srcplt = q.pixsrc[src].plot(interp=interp)
    if (show_cc==True):
        q.mkgrid_extended_src(src)
    plot_sb(srcplt,q,show_cc=show_cc,fix_limits_before_cc=fix_limits_before_cc)




import itertools
from qlens import *
import matplotlib.pyplot as plt
import code
import copy
import inspect
import sys

_cmap_scheme = 'turbo'      # this is the colormap scheme that I like the best, but it can be changed with the set_cmap function

def pause():
    if bool(getattr(sys, 'ps1', sys.flags.interactive))==True:   # will ignore the pause if not in interactive mode to begin with
        frame = inspect.currentframe().f_back
        code.interact(local=frame.f_locals,banner="Pausing for input...press <Ctrl-d> to continue the script",exitmsg="")

def set_cmap(scheme):
    global _cmap_scheme
    try:
        plt.get_cmap(scheme)
        _cmap_scheme = scheme
        return True
    except ValueError:
        return False

def get_cmap():
    print(_cmap_scheme)

def plot_ptimgs(src_x, src_y, QLens_Object, *, show_cc=True, show=True, grid=False, title=""):
    if (QLens_Object is None):
        raise RuntimeError("plot_ptimgs(...) requires the third input to be a QLens object.")

    q = QLens_Object
    imgs = q.find_ptimgs(src_x,src_y)  # this finds the lensed images and stores them in q.ptsrc

    images_x = []
    images_y = []

    k=0
    for img in imgs:
        images_x.append(img.pos[0])
        images_y.append(img.pos[1])

    imgplane_fig=plt.figure()
    imgplane_ax=plt.gca()
    ## Plotting the critical curves
    imgplane_ax.plot(images_x, images_y, marker='o', linestyle='None', color='b', label='Images (z=' + str(imgs.zsrc) + ')')
    plt.legend(loc="upper right")
    if grid==True:
        plt.grid(True)

    srcplane_fig=plt.figure()
    srcplane_ax=plt.gca()

    srcplane_ax.plot(src_x, src_y, marker='o', linestyle='None', color='b', label='Source (z=' + str(imgs.zsrc) + ')')

    plt.legend(loc="upper right")
    if grid==True:
        plt.grid(True)

    if (show_cc==True):
        add_crit_to_plot(q,srcplane_fig,imgplane_fig)

    if (title != ""):
        plt.title(title)

    if (show==True):
        plt.show(block=False)

    return (srcplane_fig,imgplane_fig)

def plot_fit_ptimgs(QLens_Object, *, show_cc=True, show=True, grid=False, title=""):
    if (QLens_Object is None):
        raise RuntimeError("plot_fit_ptimgs(...) requires the first input to be a QLens object.")

    q = QLens_Object
    q.get_fit_ptimgs()  # this finds the lensed images and stores them in q.ptsrc
    D = q.ptimgdata
    I = q.ptsrc  

    if (len(I) != len(D)):
        raise RunTimeError("Data and model do not have the same number of imagesets")

    images_x = [[] for i in range(len(I))]
    images_y = [[] for i in range(len(I))]
    data_images_x = [[] for i in range(len(I))]
    data_images_y = [[] for i in range(len(I))]

    k=0
    for i,d in zip(I,D):
        for j in range(i.n_images):
            images_x[k].append(i.images[j].pos[0])
            images_y[k].append(i.images[j].pos[1])
        for j in range(d.n_images):
            data_images_x[k].append(d.images[j].pos[0])
            data_images_y[k].append(d.images[j].pos[1])
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

    if (show_cc==True):
        add_crit_to_plot(q,srcplane_fig,imgplane_fig)

    if (title != ""):
        plt.title(title)

    if (show==True):
        plt.show(block=False)

    return (srcplane_fig,imgplane_fig)

def plotcrit(QLens_Object, *, show=True, grid=False, title=""):
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

    if (title != ""):
        plt.title(title)

    if (show==True):
        plt.show(block=False)

    return (srcplane_fig,imgplane_fig)

def add_crit_to_plot(QLens_Object, srcplane_fig=None, imgplane_fig=None):
    if(QLens_Object is None or (srcplane_fig is None and imgplane_fig is None)):
        raise RuntimeError("add_crit_to_plot(...) requires first input to be a QLens object, and 2nd and 3rd inputs to be matplotlib ax objects (srcplane,imgplane).")

    q = QLens_Object

    q.sort_critical_curves()
    for last_cc in q.sorted_critical_curve:
        pass

    if imgplane_fig is not None:
        plt.figure(imgplane_fig.number)
        imgplane_ax=plt.gca()
        ## Plotting the critical curves

        label=''
        for curve in q.sorted_critical_curve:
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
                label='Critical Curve ($z_{src}$=' + str(q.zsrc) + ')'
            imgplane_ax.plot(cc_x, cc_y, color='k', label=label)
            cc_x = []
            cc_y = []

        plt.legend(loc="upper right")

    if srcplane_fig is not None:
        plt.figure(srcplane_fig.number)
        srcplane_ax=plt.gca()

        ## Plotting the caustic curves
        label=''
        for curve in q.sorted_critical_curve:
            caustic_x = []
            caustic_y = []

            for point in curve.caustic_pts:
                caustic_x.append(point.x) # x coordinate
                caustic_y.append(point.y) # y coordinate

            # Required to connect the curve continously

            caustic_x.append(caustic_x[0])
            caustic_y.append(caustic_y[0]) 

            if (curve == last_cc):
                label='Caustic ($z_{src}$=' + str(q.zsrc) + ')'
            srcplane_ax.plot(caustic_x, caustic_y, color='k', label=label)
            
        plt.legend(loc="upper right")

def add_crit_to_imgplot(QLens_Object, imgplane_fig):
    if(QLens_Object is None or imgplane_fig is None):
        raise RuntimeError("add_crit_to_imgplot(...) requires the first input to be a QLens object, and the 2nd input to be matplotlib ax objects (imgplane).")
    add_crit_to_plot(QLens_Object,imgplane_fig=imgplane_fig)

def add_caustics_to_srcplot(QLens_Object, srcplane_fig):
    if(QLens_Object is None or srcplane_fig is None):
        raise RuntimeError("add_caustics_to_srcplot(...) requires the first input to be a QLens object, and the 2nd input to be matplotlib ax objects (srcplane).")
    add_crit_to_plot(QLens_Object,srcplane_fig=srcplane_fig)

#def checknan(img):
    #foundnan = False
    #for row in img[3]:
        #for x in row:
            #if (x*0.0 != 0.0):
                #print("NAN value!")
                #foundnan = True
    #if (foundnan==False):
        #print("Did not find any NAN's")

def plot_sb(img, QLens_Object, *, show=True, show_cc=True, fix_limits_before_cc=True, title=""):
    q = QLens_Object
    global _cmap_scheme
    plottype = img[0]
    # For some reason, if we don't make copies of the lists from the tuple (as below), the tuple ends up getting corrupted when matplotlib stuff is called. I have no idea why, it's super annoying.
    x = copy.deepcopy(img[1])
    y = copy.deepcopy(img[2])
    z = copy.deepcopy(img[3])

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [x[0],x[-1],y[0],y[-1]] 
    im = ax.imshow(z, interpolation='nearest', extent=extent, cmap=_cmap_scheme, origin='lower')
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
    if (title != ""):
        plt.title(title)

    if (show==True):
        plt.show(block=False)

    return (fig, ax)

def plotdata(QLens_Object, *, show=True, band=0, title="", nomask=False, fgmask=False):
    q = QLens_Object
    if (len(q.imgdata)==0):
        raise RunTimeError("No image data has been loaded")
    if (band >= len(q.imgdata)):
        raise RunTimeError("Specified data band has not been created (n_bands=" + len(q.imgdata) + ")")
    dataimg = q.imgdata[band].plot(nomask=nomask,fgmask=fgmask)
    return plot_sb(dataimg,q,show=show,title=title)

def plotimg(QLens_Object, *, src=-1, show=True, show_cc=True, nomask=False, nres=False, res=False, title="", output_fits=""):
    q = QLens_Object
    img = q.plotimg(src=src,nres=nres,res=res,nomask=nomask,output_fits=output_fits)
    if (output_fits==""):
        if (show_cc==True and src >= 0):
            q.mkgrid_extended_src(src)
        else:
            if (show_cc==True and src < 0):
                q.mkgrid()
        return plot_sb(img,q,show=show,show_cc=show_cc,title=title)
    else:
        return None

def plotsrc(QLens_Object, *, show=True, show_cc=True, interp=False, title="", fix_limits_before_cc=True, src=0):
    q = QLens_Object
    srcplt = q.pixsrc[src].plot(interp=interp)
    if (show_cc==True):
        q.mkgrid_extended_src(src)
    return plot_sb(srcplt,q,show=show,title=title,show_cc=show_cc,fix_limits_before_cc=fix_limits_before_cc)


import GetDistPlots, os
g=GetDistPlots.GetDistPlotter('./')
g.settings.setSubplotSize(3.0000,width_scale=1.0)  # width_scale scales the width of all lines in the plot
outdir=''
roots=['alpha_multinest']
marker_list=[]   # put parameter values in this list if you want to mark the 'true' or best-fit values on posteriors
g.triangle_plot(roots, ['b','q','theta','xc','yc','shear1','shear2'],markers=marker_list,marker_color='orange',show_marker_2d=False,marker_2d='x',shaded=True)
g.export(os.path.join(outdir,'alpha_multinest_tri.pdf'))

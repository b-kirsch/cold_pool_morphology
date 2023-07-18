# -*- coding: utf-8 -*-
"""
@author: Bastian Kirsch (bastian.kirsch@uni-hamburg.de)

Plotting routines to be used in fesstval_cp_morph.py

Dependences on non-standard software:
- fesstval_routines.py

Last updated: 21 June 2023
"""

import numpy as np
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from cmcrameri import cm as cmc
import cartopy as ctp
import cartopy.io.img_tiles as cimgt
import geopy as gp
from geopy.distance import distance

import fesstval_routines as fst

#----------------------------------------------------------------------------
# Paths
maindir     = '.'
plotdir     = maindir+'Cold-Pools/Plots/Paper_CP_Morphology/'
file_fig13a = maindir+'Cold-Pools/Paper_CP_Morphology/data_fig13a.txt'

#----------------------------------------------------------------------------
# Basic settings

# File naming
fig_numbers = True # Name files "Kirsch_Figxx.pdf" if True

# Fontsize
fs = 12

# Plotting style
mpl.rcParams['font.size'] = fs
mpl.rcParams['font.sans-serif'] = ['Tahoma']
mpl.rcParams['axes.labelpad'] = 8
mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = 'black'
mpl.rcParams['grid.linestyle'] = 'dotted'
mpl.rcParams['grid.alpha'] = 0.25
mpl.rcParams['xtick.labelsize'] = fs
mpl.rcParams['ytick.labelsize'] = fs
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.handlelength'] = 1.5
mpl.rcParams['legend.handletextpad'] = 0.5 

#----------------------------------------------------------------------------
# Helper routines and definitions

# Truncate colormap at given relative ranges
def truncate_colormap(cmap,minval=0.0,maxval=1.0,n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
               'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
               cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def make_colorbar(figure,axis,paxis,cblabel,cbarticks,ori='vertical',
                  xpos=np.nan,ypos=np.nan,width=0.03,height=np.nan,
                  alpha=1,fs=fs):
    axpos = axis.get_position()
    cax_width = width
    if np.isfinite(xpos) == False:
        pos_x = axpos.x0+axpos.width + cax_width
    else: pos_x = xpos    
    if np.isfinite(ypos) == False:
        pos_y = axpos.y0
    else: pos_y = ypos   
    if np.isfinite(height) == False:
        cax_height = axpos.height
    else: cax_height = height    
    pos_cax = figure.add_axes([pos_x,pos_y,cax_width,cax_height]) 
    cbar = plt.colorbar(paxis,cax=pos_cax,orientation=ori,alpha=alpha)
    cbar.set_label(cblabel,fontsize=fs)
    cbar.set_ticks(cbarticks)
    cbar.ax.tick_params(labelsize=fs) 
    return

# Normalize variable
def norm_var(indata,scale_min=0,scale_max=1,data_min=np.nan,data_max=np.nan):
    if np.isnan(data_min): data_min = np.nanmin(indata)
    if np.isnan(data_max): data_max = np.nanmax(indata)
    return ((indata-data_min) / (data_max-data_min)) * (scale_max-scale_min) + scale_min

def r_squared(data_obs,data_model):        
    ss_res = np.nansum((data_obs - data_model)**2)
    ss_tot = np.nansum((data_obs-np.nanmean(data_obs))**2)
    return  1 - (ss_res / ss_tot)


# Labels
abc = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']

plabels = {'DT'           : r'$\Delta T$ (K)',
           'DP'           : r'$\Delta p$ (hPa)',
           'RR'           : r'$R$ (mm h$^{-1}$)',
           'RR_MEAN'      : r'$\overline{R}$ (mm h$^{-1}$)',
           'TIME_UTC'     :  'Time (UTC)',
           'TIME_LT'      :  'Local time',
           'TIME_MIN'     :  'Time (min)',
           'DIST'         : r'$d_\mathrm{center}$ (km)',
           'DIST_NORM'    : r'$d_\mathrm{center}/r_\mathrm{equi}$ (-)',
           'CP_AREA'      : r'$A_\mathrm{CP}$ (km$^2$)',
           'U_RAD'        : r'$u_\mathrm{r}$ (m s$^{-1}$)',
           'CP_EQUI_DIAM' : r'$d_\mathrm{equi}$ (km)',
           'CP_ASPECT'    :  'Aspect ratio (-)',
           'DT_MEAN'      : r'$\overline{\Delta T}$ (K)',
           'RR_ACCU_ABS'  : r'$\Sigma R$ (10$^9$ L)',
           'CP_AREA_NORM' :  '$A_\mathrm{CP}/A_\mathrm{CP,max}$ (-)',
           'DT_MEAN_NORM' : r'$\overline{\Delta T}/\overline{\Delta T}_\mathrm{min}$ (-)',
           'RR_ACCU_NORM' :  '$\Sigma R/\Sigma R_\mathrm{max}$ (-)',
           'DC_HEIGHT'    : r'$h_\mathrm{DC}$ (m)',
           'AREA_EXP'     : r'$\epsilon_\mathrm{A}$ (km$^2$ min$^{-1}$)',
           'DU'           : r'$\Delta u$ (m s$^{-1}$)',
           'DU_MAX'       : r'$\Delta u_\mathrm{max}$ (m s$^{-1}$)',
           'DT_MEAN_REL'  : r'$|\overline{\Delta T}|/T_0$',
           'DT_MIN_REL'   :r'$|\Delta T_\mathrm{min}|/T_0$',
            }   

tz_lt = dt.timezone(dt.timedelta(hours=2)) # Timezone local time
dt_lt = dt.timedelta(hours=2)              # Timedelta local time

cmap_growth = truncate_colormap(cmc.batlowK,minval=0.0,maxval=0.95)
norm_growth = mpl.colors.Normalize(0,3,clip=True)

#----------------------------------------------------------------------------
# Plotting routines
def overview_morph(dtime_data,lon_data,lat_data,meta_data,TT_data,mask_data,
                   center_lon_data,center_lat_data,cp_lim,labels=plabels,
                   pdir=plotdir):
    print('Plotting overview of cold pool morphology (stamps)')

    lonmin, lonmax = 13.86, 14.39
    latmin, latmax = 52.01, 52.33
    
    vmin_tt,vmax_tt,res_tt = -8,cp_lim,0.5
    
    tt_levels = np.arange(vmin_tt,vmax_tt+res_tt,res_tt)
    tt_cmap   = truncate_colormap(cmc.vik,maxval=0.45)
    tt_norm   = mpl.colors.BoundaryNorm(tt_levels, tt_cmap.N, clip=True)
    
    tt_cmap_grey = truncate_colormap(cmc.grayC_r,minval=0.1,maxval=0.9)
    tt_norm_grey = mpl.colors.BoundaryNorm(tt_levels, tt_cmap_grey.N, clip=True)
    
    fig,ax = plt.subplots(5,8,figsize=(14,9),dpi=400)
    ax = ax.flatten()
    
    for it,dtime in enumerate(dtime_data):
        i = it if it <= 31 else it+1 # shift last low of plot by one
        ax[i].scatter(meta_data['LON'],meta_data['LAT'],c='lightgrey',s=4)
        ax[i].contourf(lon_data,lat_data,TT_data[:,:,it],levels=tt_levels,
                       norm=tt_norm_grey,cmap=tt_cmap_grey,alpha=0.6)
        TT_masked = TT_data[:,:,it]
        mask = mask_data[:,:,it].astype(bool)
        TT_masked[~mask] = np.nan
        ax[i].contourf(lon_data,lat_data,TT_masked,levels=tt_levels,
                        norm=tt_norm,cmap=tt_cmap)
        ax[i].scatter(center_lon_data[it],center_lat_data[it],color='k',marker='x')
        ax[i].set_xlim([lonmin,lonmax])
        ax[i].set_ylim([latmin,latmax])
        ax[i].set_title((dtime+dt_lt).strftime('%b %d, %H%M h'),
                        y=-0.2,fontsize=fs-2)
        ax[i].grid(False,axis='both')
        ax[i].axis('off') 
    
    pax  = mpl.cm.ScalarMappable(norm=tt_norm, cmap=tt_cmap) 
    cbar = plt.colorbar(pax,ax=ax,orientation='horizontal',anchor=(0.5,-2.0),
                        ticks=tt_levels,aspect=35,shrink=0.6)
    
    cbar.set_label(labels['DT'],fontsize=fs)
    
    for i in [-8,-1]: ax[i].remove()

    plt.tight_layout()
    plotname = pdir+'overview_morphology.png'
    if fig_numbers: plotname = pdir+'Kirsch_Fig06.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!')
    
    return


def snapshot(dtime,lon_data,lat_data,lon_data_rad,lat_data_rad,meta_data,
             TT_data,RR_data,PP_data=[],UU_data=[],VV_data=[],
             plot_pp=False,plot_uv=False,cp_lim=-2,TT_min=np.nan,rr_min=0.1,
             labels=plabels,pdir=plotdir):
    
    print('Plotting snapshot of CP event for '+dtime.strftime('%H:%M UTC'))
    
    # Plotting area and scale position
    if dtime.year == 2020:
        minlon, maxlon = 9.595, 10.45
        minlat, maxlat = 53.3, 53.8
        scale_lon,scale_lat,scale_len,scale_lw = 9.62,53.32,10,1.3
        pdir = pdir.replace('FESSTVaL','FESSTHH')
    else:
        minlon, maxlon = 13.86, 14.39
        minlat, maxlat = 52.01, 52.33
        scale_lon,scale_lat,scale_len,scale_lw = 13.875,52.025,10,1.3
        
    transform = ctp.crs.PlateCarree()    

    # Setting up color scales
    vmax_tt = cp_lim
    res_tt  = 1 
    vmin_tt = TT_min if np.isfinite(TT_min) else np.floor(np.nanmin(TT_data))
    
    if vmin_tt >= -9: res_tt = 0.5
    if vmin_tt >= -5: res_tt = 0.3
    if vmin_tt >= -3: res_tt = 0.1
    
    clabel_fmt = '%2.0f' if res_tt == 1 else '%2.1f'
    
    tt_levels = np.arange(vmin_tt,vmax_tt+res_tt,res_tt)
    tt_cmap   = cmc.batlow
    tt_norm   = mpl.colors.BoundaryNorm(tt_levels, tt_cmap.N, clip=True)
    tt_color,tt_style,tt_width = 'black','solid',2
    
    res_pp  = 0.25 
    vmax_pp = 2 
    pp_levels = np.arange(res_pp,vmax_pp+res_pp,res_pp)
    pp_color,pp_style,pp_width = 'black','dashed',1
    
    rr_levels = np.array([rr_min,5,10,20,50,100,200])
    rr_cmap   = truncate_colormap(cmc.oslo_r,minval=0.1)
    rr_norm   = mpl.colors.BoundaryNorm(rr_levels, rr_cmap.N, clip=True)
    rr_alpha = 0.5
    
    qu_key   = 5  # m/s
    qu_props = {'color'         :'orangered', # 'black'
                'headlength'    :4,
                'headaxislength':4,
                'width'         :0.005,
                'headwidth'     :4,
                'pivot'         :'tail', #'mid'
                'scale'         :75, # m/s per plot width,
                'transform'     :transform,
                'zorder'        :5}
    
    ii_wxt = meta_data['STATION'].str.endswith('w')
    
    
    # Main plot
    fig = plt.figure(figsize=(6.2,6.0),dpi=300)
    
    plotmap = cimgt.Stamen('toner')
    ax = plt.axes(projection=plotmap.crs)
    ax.set_extent([minlon,maxlon,minlat,maxlat],transform)

    # TT perturbation   
    tt = ax.contour(lon_data,lat_data,TT_data,
                    levels=tt_levels,norm=tt_norm,cmap=tt_cmap,
                    linewidths=tt_width,transform=transform)
    ax.clabel(tt,fmt=clabel_fmt,fontsize=fs-2)
    ax.plot([np.nan],[np.nan],color=tt_color,linestyle=tt_style,
            linewidth=tt_width,label=labels['DT'])
    
    # PP perturbation
    if plot_pp and (len(PP_data) > 0):
        pp = ax.contour(lon_data,lat_data,PP_data,
                        colors=pp_color,levels=pp_levels,
                        linestyles=pp_style,linewidths=pp_width,
                        transform=transform)
        ax.clabel(pp,fmt='%1.2f',fontsize=fs-4)
        ax.plot([np.nan],[np.nan],color=pp_color,linestyle=pp_style,
                linewidth=pp_width,label=labels['DP'])
    
    # Radar rainfall
    if (RR_data > rr_levels[0]).sum() > 0:
        ax.contourf(lon_data_rad,lat_data_rad,RR_data,
                    levels=rr_levels,norm=rr_norm,cmap=rr_cmap,
                    transform=transform,alpha=rr_alpha)
    
    # Station points
    ax.scatter(meta_data['LON'],meta_data['LAT'],c='dimgrey',marker='o',s=20,
               transform=transform)    
    
    # Wind arrows
    if plot_uv and (len(UU_data) > 0) and (len(VV_data) > 0):
        qu = ax.quiver(meta_data[ii_wxt]['LON'].to_numpy(dtype=float),
                       meta_data[ii_wxt]['LAT'].to_numpy(dtype=float),
                       UU_data.to_numpy(dtype=float),
                       VV_data.to_numpy(dtype=float),
                       **qu_props)
        
        ax.quiverkey(qu,0.62,1.03,qu_key,'Wind speed ('+str(qu_key)+' m s$^{-1}$)',
                     coordinates='axes',labelpos='E',fontproperties=dict(size=fs))

    # Legends and colorbar
    ax.set_title((dtime+dt_lt).strftime('%b %d, %Y, %H%M h'),
                  loc='left',fontsize=fs)
    
    pax = mpl.cm.ScalarMappable(norm=rr_norm, cmap=rr_cmap)    
    make_colorbar(fig,ax,pax,labels['RR'],rr_levels,
                      fs=fs,alpha=rr_alpha)
    
    # Scale
    scale_lon_end = fst.geo_line(scale_lat,scale_lon,scale_len,90)[1]
    ax.plot([scale_lon,scale_lon_end],[scale_lat,scale_lat],linestyle='solid',
            linewidth=scale_lw,color='black',transform=transform)
    for ll in [scale_lon,scale_lon_end]:
        ax.plot([ll,ll],[scale_lat-0.003,scale_lat+0.003],linestyle='solid',
                linewidth=scale_lw,color='black',transform=transform)    
    ax.text((scale_lon_end+scale_lon)/2.,scale_lat+0.008,str(scale_len)+' km',
            transform=transform,ha='center',fontsize=fs)  
    
    ax.legend(loc='upper left',fontsize=fs)
    
    plotname = pdir+dtime.strftime('snapshot_%Y%m%d_%H%M.png')
    if fig_numbers: plotname = pdir+'Kirsch_Fig04.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!')  
    
    return


def time_distance(dist_data,TT_data,PP_data,RR_data,growth_time=[],freq=1,
                  plot_dp=False,plot_rs=False,labels=plabels,pdir=plotdir):
    
    time_start = TT_data.index[0]
    time_end   = TT_data.index[-1]
    
    dist_min = dist_data[0]/1000.
    dist_max = dist_data[-1]/1000.
    dist_res = (dist_data[1]-dist_data[0])/1000.
    
    if time_start.year == 2020: pdir = pdir.replace('FESSTVaL','FESSTHH')
    
    if time_start.date() == dt.date(2021,6,26):
        dist_max   = 10
        time_start = dt.datetime(2021,6,26,9,0)
        time_end   = dt.datetime(2021,6,26,11,0)
    
    if time_start.date() == dt.date(2021,6,29):
        time_start = dt.datetime(2021,6,29,13,15) #12,45
        time_end   = dt.datetime(2021,6,29,15,15) #16,15
    
    if time_start.date() == dt.date(2021,7,9):
        time_start = dt.datetime(2021,7,9,13,30)
        time_end   = dt.datetime(2021,7,9,15,40)  
        
    if time_start.date() == dt.date(2021,7,25):
        time_start = dt.datetime(2021,7,25,13,30)
        time_end   = dt.datetime(2021,7,25,17,0)   
        
    if time_start.date() == dt.date(2020,8,10):
        dist_max   = 15
        time_start = dt.datetime(2020,8,10,13,0)
        time_end   = dt.datetime(2020,8,10,15,0) 
    
    vmax_tt = -2
    res_tt  = 1 
    vmin_tt = np.floor(TT_data.loc[time_start:time_end].min().min())
    
    res_pp  = 0.25 
    vmax_pp = 2#np.ceil(dP_dist_time.max().max())
    
    if vmin_tt > -9: res_tt = 0.5
    if vmin_tt >= -5: res_tt = 0.25
    if vmin_tt >= -3: res_tt = 0.1
    
    #if vmax_pp > 1: res_pp = 0.5
    
    tt_levels = np.arange(vmin_tt,vmax_tt+res_tt,res_tt)
    tt_cmap = cmc.batlow
    tt_norm = mpl.colors.BoundaryNorm(tt_levels, tt_cmap.N, clip=True)
    
    pp_color,pp_style,pp_width = 'black','solid',1.8
    rr_color,rr_width = 'royalblue',1.8
    pp_levels = np.arange(res_pp,vmax_pp+res_pp,res_pp)      
    
    mpl.rcParams['axes.spines.left']  = True
    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['axes.spines.top']   = True
    
    time_data = pd.date_range(time_start,time_end,freq=str(freq)+'min')
    
    TT_plot = np.array(TT_data.reindex(time_data).transpose(),dtype=float)
    PP_plot = np.array(PP_data.reindex(time_data).transpose(),dtype=float)
    
    print('')
    print('Plotting time-distance histogram of CP event')
    fig,ax = plt.subplots(1,1,figsize=(7.6,4.7),dpi=300)
    
    ax.contourf(time_data,dist_data[:-1]/1000.+dist_res/2.,TT_plot,
                levels=tt_levels,norm=tt_norm,cmap=tt_cmap)
    if plot_dp:
        pp = ax.contour(time_data,dist_data[:-1]/1000.+dist_res/2.,PP_plot,
                        colors=pp_color,levels=pp_levels,
                        linestyles=pp_style,linewidths=pp_width)
        ax.clabel(pp,fmt='%1.2f',fontsize=fs-2)
        ax.plot([np.nan],[np.nan],color=pp_color,linestyle=pp_style,
                linewidth=pp_width,label=labels['DP'])
    
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H%M',tz=tz_lt))
    
    if len(growth_time) == 2:
        for i in range(2):
            ax.plot([growth_time[i],growth_time[i]],[0,30],color='black',
                    lw=pp_width,ls='dashed')
    
    ax2 = ax.twinx()
    ax2.plot(time_data,RR_data.reindex(time_data),color=rr_color,
             linewidth=rr_width,linestyle='solid')

    ax.set_xlim([time_start,time_end])
    ax.set_ylim(dist_min+dist_res/2.,dist_max-dist_res/2.)
    ax.grid(visible=False,axis='both')
    ax.set_xlabel(labels['TIME_LT'],fontsize=fs)
    ax.set_ylabel(labels['DIST'],fontsize=fs) 
    
    ax2.set_ylim([0,ax2.get_ylim()[1]])
    ax2.set_ylabel(labels['RR_MEAN'],fontsize=fs,color=rr_color)
    ax2.grid(visible=False,axis='both')
    ax2.tick_params(axis='y',labelcolor=rr_color)
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H%M',tz=tz_lt))
    
    pax = mpl.cm.ScalarMappable(norm=tt_norm, cmap=tt_cmap)    
    make_colorbar(fig,ax,pax,labels['DT'],
                  tt_levels,fs=fs,xpos=0.99) 
    
    if plot_dp:
        ax.legend(loc='upper center',fontsize=fs)
    
    plotname = pdir+time_start.strftime('time_distance_%Y%m%d.png')
    if fig_numbers: plotname = pdir+'Kirsch_Fig09.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!')     
    return


def _hist_plot(axis,data,xbins,color,label='_nolegend_',alpha=0.5):
        axis.hist(data,bins=xbins,density=False,alpha=alpha,color=color,
                  label=label,edgecolor=None)
        return

def stats_bound(cp_props_data,cp_props_filt,diam_lims,bound_lim,cp_lim,
                labels=plabels,abc_labels=abc,pdir=plotdir):
    print('Plotting size and strength statistics of all CP events')
    
    n_objects_all  = cp_props_data['CP_AREA'].notnull().sum()
    n_objects_filt = cp_props_filt['CP_AREA'].notnull().sum()
    
    diam_min,diam_max = diam_lims 
    
    size_min,size_max,size_res = np.floor(diam_min),41,2
    size_bins = np.arange(size_min,size_max+size_res,size_res)
    
    dT_min,dT_max,dT_res = -15,cp_lim,0.25
    dT_bins = np.arange(dT_min,dT_max+dT_res,dT_res)
    
    col1,col2 = 'grey','black'
    mpl.rcParams['axes.spines.left']  = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top']   = False

    fig,ax = plt.subplots(1,2,figsize=(8.4,3.9),dpi=300)
    ax = ax.flatten()
    
    a = 0
    _hist_plot(ax[a],cp_props_data['CP_EQUI_DIAM'],size_bins,col1,
              label='All ($n$={:4.0f})'.format(n_objects_all))
    _hist_plot(ax[a],cp_props_filt['CP_EQUI_DIAM'],size_bins,col2,
              label=r'$f_\mathrm{b}\leq$'+'{:1.2f}'.format(bound_lim)+\
                    ' ($n$={:4.0f})'.format(n_objects_filt))
    ax[a].set_xlabel(labels['CP_EQUI_DIAM'],fontsize=fs)
    ax[a].set_ylabel('Number of objects',fontsize=fs)
    ylims = ax[a].get_ylim()   

    ax[a].plot([diam_min,diam_min],[ylims[0],ylims[-1]],color='k',
               linestyle='dashed')
    ax[a].plot([diam_max,diam_max],[ylims[0],ylims[-1]],color='k',
               linestyle='dashed')
    
    ax[a].set_xlim(0,42)
    ax[a].set_xticks(np.arange(0,45,5))
    ax[a].set_ylim(*ylims)
    
    a = 1
    _hist_plot(ax[a],cp_props_data['DT_MEAN'],dT_bins,col1)
    _hist_plot(ax[a],cp_props_filt['DT_MEAN'],dT_bins,col2)
    ax[a].set_xlabel(labels['DT_MEAN'],fontsize=fs)
    ax[a].set_xlim(-9,cp_lim)
    ax[a].set_xticks(np.arange(-9,cp_lim+1))

    for a in range(ax.shape[0]): 
        ax[a].set_title(abc_labels[a],loc='left',fontsize=fs,**{'weight': 'bold'})
        ax[a].grid(visible=False,axis='both')

    ax[0].legend(loc='upper center',fontsize=fs)
    
    plt.tight_layout()
    plotname = pdir+'stats_bound.png'
    if fig_numbers: plotname = pdir+'Kirsch_Fig05.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!')    
    return


def stats_morph(cp_props_filt,diam_groups,dT_groups,
                labels=plabels,abc_labels=abc,pdir=plotdir):

    print('Plotting morphology of all CP events')
    
    fin = cp_props_filt['CP_EQUI_DIAM'].notnull()
    
    ii_diam = [diam_groups == 'Q'+str(i) for i in range(1,5)]
    ii_dT   = [dT_groups == 'Q'+str(i) for i in range(1,5)]
    
    diam_boxdata = [cp_props_filt['CP_EQUI_DIAM'][fin],
                    cp_props_filt['CP_EQUI_DIAM'][ii_dT[0]],
                    cp_props_filt['CP_EQUI_DIAM'][ii_dT[1]],
                    cp_props_filt['CP_EQUI_DIAM'][ii_dT[2]],
                    cp_props_filt['CP_EQUI_DIAM'][ii_dT[3]],
                    ]
    
    asp_diam_boxdata  = [cp_props_filt['CP_ASPECT'][fin],
                         cp_props_filt['CP_ASPECT'][ii_diam[0]],
                         cp_props_filt['CP_ASPECT'][ii_diam[1]],
                         cp_props_filt['CP_ASPECT'][ii_diam[2]],
                         cp_props_filt['CP_ASPECT'][ii_diam[3]],
                         ]
    
    asp_dT_boxdata   = [cp_props_filt['CP_ASPECT'][fin],
                        cp_props_filt['CP_ASPECT'][ii_dT[0]],
                        cp_props_filt['CP_ASPECT'][ii_dT[1]],
                        cp_props_filt['CP_ASPECT'][ii_dT[2]],
                        cp_props_filt['CP_ASPECT'][ii_dT[3]],
                        ]
      
    
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top']   = False
    mpl.rcParams['axes.spines.left']  = False
    
    box_kwargs = {'vert':False,'whis':(5,95),'widths':0.5,'patch_artist':True,
                  'showfliers':False,'medianprops':dict(color='black'),
                  'whiskerprops':dict(color='black'),'showmeans':False,
                  'boxprops':dict(color='black',alpha=0.9),
                  'positions':[4,3,2,1,0]}
        
    cmap_diam = truncate_colormap(cmc.vik,minval=0.55,maxval=0.95)
    cmap_dT   = truncate_colormap(cmc.vik,minval=0.05,maxval=0.45)
    norm = mpl.colors.Normalize(0,3,clip=True)
    
    colors_diam = ['dimgrey']
    for i in range(4): colors_diam.append(cmap_diam(norm(i)))
    colors_dT = ['dimgrey']
    for i in range(4): colors_dT.append(cmap_dT(norm(i)))
    
    fig,ax = plt.subplots(1,3,figsize=(12,4),dpi=300)
    ax = ax.flatten()
    
    a = 0
    boxdiam = ax[a].boxplot(diam_boxdata,**box_kwargs) 
    ax[a].set_xlabel(labels['CP_EQUI_DIAM'],fontsize=fs)
    ax[a].set_xlim(2,26)
    ax[a].set_title(r'grouped by $\overline{\Delta T}$',loc='center',fontsize=fs)
    
    a = 1
    boxasp_dT = ax[a].boxplot(asp_dT_boxdata,**box_kwargs)
    ax[a].set_xlabel(labels['CP_ASPECT'],fontsize=fs)
    ax[a].set_xlim(0.95,2.85)
    ax[a].set_xticks(np.arange(1.0,3.0,0.3))
    ax[a].set_title(r' grouped by $\overline{\Delta T}$',loc='center',fontsize=fs)
    
    a = 2
    boxasp_diam = ax[a].boxplot(asp_diam_boxdata,**box_kwargs)
    ax[a].set_xlabel(labels['CP_ASPECT'],fontsize=fs)
    ax[a].set_xlim(0.95,2.85)
    ax[a].set_xticks(np.arange(1.0,3.0,0.3))
    ax[a].set_title(r' grouped by $d_\mathrm{equi}$',loc='center',fontsize=fs)
    
    for a in range(ax.shape[0]): 
        ax[a].set_title(abc_labels[a],loc='left',fontsize=fs,**{'weight': 'bold'})
        ax[a].grid(visible=False,axis='both')
        ax[a].set_yticks(np.arange(5))
        ax[a].set_ylim(-0.5,4.5)
        if a in [1,2]:
            ax[a].axes.get_yaxis().set_visible(False)
        if a == 0:    
            ax[a].set_yticklabels(['All','Q1','Q2','Q3','Q4'][::-1]) 
            ax[a].tick_params(axis='y',which='both',length=0)
    
    for box in [boxdiam,boxasp_dT]:
        for patch, color in zip(box['boxes'], colors_dT):
            patch.set_facecolor(color)
            
    for box in [boxasp_diam]:
        for patch, color in zip(box['boxes'], colors_diam):
            patch.set_facecolor(color)        

    plt.tight_layout()
    plotname = pdir+'stats_morphology.png'
    if fig_numbers: plotname = pdir+'Kirsch_Fig07.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!')           
    return


def stats_structure(dT_radial,diam_groups,dT_groups,
                    labels=plabels,abc_labels=abc,pdir=plotdir):

    print('Plotting statistics of radial structure')
    
    ii_diam = [diam_groups == 'Q'+str(i) for i in range(1,5)]
    ii_dT   = [dT_groups == 'Q'+str(i) for i in range(1,5)]

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top']   = False
    mpl.rcParams['axes.spines.left']  = True
    
    lw = 1.8
        
    cmap_diam = truncate_colormap(cmc.vik,minval=0.55,maxval=0.95)
    cmap_dT   = truncate_colormap(cmc.vik,minval=0.05,maxval=0.45)
    norm = mpl.colors.Normalize(0,3,clip=True)
    
    fig,ax = plt.subplots(1,2,figsize=(8,3.7),dpi=300)
    ax = ax.flatten()
    
    for a in range(2):
        ax[a].fill_between(dT_radial.columns,dT_radial.quantile(0.25),
                           dT_radial.quantile(0.75),color='lightgrey',alpha=0.7)
        ax[a].plot(dT_radial.columns,dT_radial.quantile(0.5),lw=lw,ls='solid',
                   color='black',label='All',marker='.')
        ax[a].set_xlabel(labels['DIST_NORM'],fontsize=fs)
        
    for i in range(4):
        ax[0].plot(dT_radial.columns,dT_radial[ii_dT[i]].quantile(0.5),
                   lw=lw,color=cmap_dT(norm(i)),ls='solid',
                   label=r'$\overline{\Delta T}$ Q'+str(i+1))    
    
    for i in range(4):
        ax[1].plot(dT_radial.columns,dT_radial[ii_diam[i]].quantile(0.5),
                   lw=lw,color=cmap_diam(norm(i)),ls='solid',
                   label=r'$d_\mathrm{equi}$ Q'+str(i+1))
    
    ax[0].set_ylabel(labels['DT'],fontsize=fs)
    for a in range(ax.shape[0]): 
        ax[a].set_xlim(0,1)
        ax[a].set_title(abc_labels[a],loc='left',fontsize=fs,**{'weight': 'bold'})
        ax[a].grid(visible=False,axis='both')
        ax[a].set_yticks(np.arange(-3.5,-1.9,0.3))
        ax[a].set_ylim(-3.65,-1.9)
    
    ax[0].legend(fontsize=fs-2,loc='lower right')
    
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles[1:],labels[1:],fontsize=fs-2,loc='lower right')

    plt.tight_layout()
    plotname = pdir+'stats_radial_structure.png'
    if fig_numbers: plotname = pdir+'Kirsch_Fig08.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!')           
    return


def growth_time(cp_props_data,cases_meta,cmap=cmap_growth,norm=norm_growth,
                labels=plabels,abc_labels=abc,pdir=plotdir):
    
    print('Plotting time series during growth phase of selected events')
    
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top']   = False
    mpl.rcParams['axes.spines.left']  = True
    
    lw = 2.0
    ms = 3
    
    fig,ax = plt.subplots(1,3,figsize=(12,3.9),dpi=300)
    
    for i,icp in enumerate(cases_meta.index):    
        event = (cp_props_data['ICP'] == icp) & (cp_props_data['GROWTH'])
        if event.sum() == 0: continue
        name_cp = cases_meta['NAME'].loc[icp]
        lab = name_cp
        col = cmap(norm(i))
        p_kwargs = {'color':col,'linestyle':'solid','linewidth':lw}
        
        time    = cp_props_data[event]['GROWTH_TIME']
        area    = cp_props_data[event]['CP_AREA']
        dT_mean = cp_props_data[event]['DT_MEAN']
        accu    = cp_props_data[event]['RR_ACCU_ABS']*1e-9
        
        ax[0].plot(time,area,**p_kwargs,label=lab)
        ax[1].plot(time,dT_mean,**p_kwargs)
        ax[2].plot(time,accu-accu.iloc[0],**p_kwargs) # accumulation since begin of growth phase
        
        if i == 0:
            ax[1].plot([],[],color='k',lw=lw,label='Mean')
            ax[1].plot([],[],color='k',ls='None',lw=lw,marker='.',ms=ms,label='Extremum')

    a = 0
    ax[a].set_ylabel(labels['CP_AREA'],fontsize=fs)
    ax[a].set_ylim(0,1150)
    ax[a].legend(loc='upper left',fontsize=fs)
    
    a = 1    
    ax[a].set_ylabel(labels['DT_MEAN'],fontsize=fs)
    ax[a].set_ylim(-7.6,-1.8)
    
    a = 2    
    ax[a].set_ylabel(labels['RR_ACCU_ABS'],fontsize=fs)
    ax[a].set_ylim(0,6.8)
    
    for a in range(ax.shape[0]): 
        ax[a].set_title(abc_labels[a],loc='left',fontsize=fs,**{'weight': 'bold'})
        ax[a].set_xlim(0,80)
        ax[a].set_xticks(np.arange(0,90,10))
        ax[a].set_xlabel(labels['TIME_MIN'],fontsize=fs)
        ax[a].grid(False,axis='both')
            
    plt.tight_layout()
    plotname = pdir+'growth_time.png'
    if fig_numbers: plotname = pdir+'Kirsch_Fig10.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!') 
    return


def growth_rain(cp_props_data,cases_meta,dT_lim=-2,cmap=cmap_growth,norm=norm_growth,
                labels=plabels,abc_labels=abc,pdir=plotdir):

    print('Plotting rainfall controls of growth for selected events')
    
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top']   = False
    mpl.rcParams['axes.spines.left']  = True
    
    lw = 1.8
    ms = 6

    fig,ax = plt.subplots(1,2,figsize=(7.9,3.9),dpi=300)
    
    for i,icp in enumerate(cases_meta.index):    
        event = (cp_props_data['ICP'] == icp) & (cp_props_data['GROWTH'])
        name_cp = cases_meta['NAME'].loc[icp]
        lab = name_cp
        col = cmap(norm(i))
        
        p_kwargs = {'color':col,'linestyle':'None','marker':'.','markersize':ms,
                    'label':lab}
        
        dT_mean      = cp_props_data.loc[event,'DT_MEAN']
        dT_mean_norm = norm_var(np.abs(dT_mean),data_min=np.abs(dT_lim))
        area         = cp_props_data.loc[event,'CP_AREA']
        area_norm    = norm_var(area,data_min=0)
        accu         = cp_props_data.loc[event,'RR_ACCU_ABS']
        accu_norm    = norm_var(accu-accu.iloc[0])
        
        ax[0].plot(accu_norm,area_norm,**p_kwargs)
        ax[1].plot(accu_norm,dT_mean_norm,**p_kwargs)

    a = 0    
    ax[a].set_xlabel(labels['RR_ACCU_NORM'],fontsize=fs)
    ax[a].set_ylabel(labels['CP_AREA_NORM'],fontsize=fs)
    ax[a].legend(loc='upper left',fontsize=fs)

    a = 1    
    ax[a].set_xlabel(labels['RR_ACCU_NORM'],fontsize=fs)
    ax[a].set_ylabel(labels['DT_MEAN_NORM'],fontsize=fs)
    
    for a in range(ax.shape[0]): 
        ax[a].set_title(abc_labels[a],loc='left',fontsize=fs,**{'weight': 'bold'})
        ax[a].grid(False,axis='both') 
        
    for a in [0,1]:    
        xx,yy = [0,1],[0,1]
        #if a == 1: yy = [1,0]  
        ax[a].plot(xx,yy,color='grey',linewidth=lw,linestyle='dashed',zorder=0)
        ax[a].set_xlim(-0.05,1.05)
        ax[a].set_xticks(np.arange(0,1.2,0.2))
        ax[a].set_ylim(-0.05,1.05)
        ax[a].set_yticks(np.arange(0,1.2,0.2))
        
    plt.tight_layout()
    plotname = pdir+'growth_rain.png'
    if fig_numbers: plotname = pdir+'Kirsch_Fig11.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!') 
    return   


def growth_density_current(cp_props_data,cases_meta,udc_func,ur_func,
                           cmap=cmap_growth,norm=norm_growth,
                           labels=plabels,abc_labels=abc,pdir=plotdir):

    print('Plotting density current controls of growth for selected events')
    
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top']   = False
    mpl.rcParams['axes.spines.left']  = True
    
    lw = 1.8
    ms = 6

    fig,ax = plt.subplots(1,2,figsize=(7.8,3.9),dpi=300)
    
    for i,icp in enumerate(cases_meta.index):    
        event = (cp_props_data['ICP'] == icp) & (cp_props_data['GROWTH'])
        name_cp = cases_meta['NAME'].loc[icp]
        lab = name_cp
        col = cmap(norm(i))
        
        p_kwargs = {'color':col,'linestyle':'None','marker':'.','markersize':ms,
                    'label':lab}
        
        dT_mean = cp_props_data.loc[event,'DT_MEAN']
        area    = cp_props_data.loc[event,'CP_AREA']
        u_rad   = cp_props_data.loc[event,'DEQUI_RADIUS']
        
        ax[0].plot(dT_mean,u_rad,**p_kwargs)
        ax[1].plot(area,u_rad,**p_kwargs)

    a = 0    
    ax[a].set_xlabel(labels['DT_MEAN'],fontsize=fs)
    ax[a].set_ylabel(labels['U_RAD'],fontsize=fs)
    ax[a].set_xlim(-7.6,-1.8)
    ax[a].set_xticks(np.arange(-7,-1))
    
    a = 1   
    ax[a].set_xlabel(labels['CP_AREA'],fontsize=fs)
    ax[a].set_xlim(-40,1150)
    ax[a].legend(loc='upper right',fontsize=fs)
    
    for a in range(ax.shape[0]): 
        ax[a].set_ylim(-0.5,10.5)
        ax[a].set_title(abc_labels[a],loc='left',fontsize=fs,**{'weight': 'bold'})
        ax[a].grid(False,axis='both') 
    
    x_dc = np.arange(-7.4,-1.9,0.1)    
    for i,h in enumerate([100,300,500,800]):
        y_dc = udc_func(x_dc,h)
        ax[0].plot(x_dc,y_dc,color='grey',linewidth=lw,linestyle='dashed',zorder=0)
        ax[0].text(x_dc[0],y_dc[0]+0.2,str(h),ha='left',fontsize=fs-4,color='grey')
    ax[0].text(-7.4,0,labels['DC_HEIGHT'],ha='left',fontsize=fs-4,color='grey') 
    ax[0].text(-3.5,0,r'$\leftarrow$Time',ha='right',fontsize=fs-4,color='grey')
        
    for i,a_rate in enumerate([5,10,15,20]):
        area,ur = ur_func(a_rate,A_stop=1120)
        ax[1].plot(area,ur,color='grey',linewidth=lw,linestyle='dashed',zorder=0)
        ax[1].text(area[-1],ur[-1]+0.2,str(a_rate),ha='right',fontsize=fs-4,color='grey')
    ax[1].text(1120,0,labels['AREA_EXP'],ha='right',fontsize=fs-4,color='grey') 
    ax[1].text(250,0,r'Time$\rightarrow$',ha='left',fontsize=fs-4,color='grey')
            
    plt.tight_layout()
    plotname = pdir+'growth_dc.png'
    if fig_numbers: plotname = pdir+'Kirsch_Fig12.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!') 
    return   


def growth_density_current_local(cp_props_data,cases_meta,kruse_func,
                                 cmap=cmap_growth,norm=norm_growth,
                                 labels=plabels,abc_labels=abc,
                                 readfile=file_fig13a,pdir=plotdir):
    
    print('Plotting density current controls of growth for selected events (local perspective)')
    
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top']   = False
    mpl.rcParams['axes.spines.left']  = True
    
    lw = 1.8
    ms = 6
    
    p_kwargs_mean = {'linestyle':'None','marker':'.','markersize':ms}
    p_kwargs_max = {'linestyle':'None','marker':'.','markersize':ms,'mfc':'None'}
    
    x_data   = np.linspace(0,0.04,1000)
    wxt_data = pd.read_csv(readfile,sep=';',header=0)

    fig,ax = plt.subplots(1,3,figsize=(12,3.9),dpi=300)
    
    for i in range(2):
        ax[i].plot(x_data,kruse_func(x_data),color='black',linestyle='dotted',lw=lw,
                   label='Kruse22')
    ax[2].plot([0,10],[0,10],color='grey',linestyle='dashed',lw=lw)
    
    for i,icp in enumerate(cases_meta.index):    
        event = (cp_props_data['ICP'] == icp) & (cp_props_data['GROWTH'])
        event_wxt = (wxt_data['ICP'] == icp)
        name_cp = cases_meta['NAME'].loc[icp]
        col = cmap(norm(i))
        
        dT_min_rel  = wxt_data.loc[event_wxt,'DT_MIN_REL']
        du_max_wxt  = wxt_data.loc[event_wxt,'FF_MAX']
        dT_mean_rel = cp_props_data.loc[event,'DT_MEAN_REL']
        du_mean     = cp_props_data.loc[event,'FF_MEAN']
        du_max      = cp_props_data.loc[event,'FF_MAX']
        u_rad       = cp_props_data.loc[event,'DEQUI_RADIUS']
        
        ax[0].plot(dT_min_rel,du_max_wxt,label=name_cp,color=col,**p_kwargs_mean)
        
        ax[1].plot(dT_mean_rel,du_mean,label=name_cp,color=col,**p_kwargs_mean)
        ax[2].plot(u_rad,du_mean,color=col,**p_kwargs_mean)
        
        ax[1].plot(dT_mean_rel,du_max,mec=col,**p_kwargs_max)
        ax[2].plot(u_rad,du_max,mec=col,**p_kwargs_max)
        
        #print(r_squared(du_mean,kruse_func(dT_mean_rel)))
    
    a = 0
    ax[a].set_xlabel(labels['DT_MIN_REL'],fontsize=fs)
    ax[a].set_ylabel(labels['DU_MAX'],fontsize=fs)

    a = 1
    ax[a].set_xlabel(labels['DT_MEAN_REL'],fontsize=fs)
    ax[a].set_ylabel(labels['DU'],fontsize=fs)
    ax[a].legend(loc='best',fontsize=fs)
    
    a = 2
    ax[a].set_xlabel(labels['U_RAD'],fontsize=fs)
    ax[a].set_ylabel(labels['DU'],fontsize=fs)
    ax[a].set_xlim(-0.5,10.5)
    ax[a].set_xticks(np.arange(0,12,2))
    
    for a in range(ax.shape[0]):
        ax[a].set_title(abc_labels[a],loc='left',fontsize=fs,**{'weight': 'bold'})
        ax[a].set_ylim(-0.5,10.5)
        ax[a].set_yticks(np.arange(0,12,2))
        ax[a].grid(visible=False,axis='both')
        if a < 2:
            ax[a].set_xlim(-0.002,0.042)
            ax[a].set_xticks(np.arange(0,0.05,0.01))
            
    plt.tight_layout()

    plotname = pdir+'growth_dc_local.png'
    if fig_numbers: plotname = pdir+'Kirsch_Fig13.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!') 
    return   


def fesstval_paper(dtime,lon_data,lat_data,lon_data_rad,lat_data_rad,meta_data,
                   TT_data,RR_data,PP_data,U_data,V_data,fs=10,pdir=plotdir):
    
    print('Plotting snapshots for FESSTVaL paper')
    
    dtime1 = dt.datetime(2021,6,29,13,21)
    dtime2 = dt.datetime(2021,6,29,14,16)
    
    # Plotting area and scale position
    minlon, maxlon = 13.86, 14.39
    minlat, maxlat = 52.01, 52.33
        
    transform = ctp.crs.PlateCarree()   
    aspect = distance((52.15,14),(52.25,14)).km/distance((52.2,14),(52.2,14.1)).km

    # Setting up color scales
    clabel_fmt = ['%2.1f','%2.0f']
    
    tt_levels = [np.arange(-2,-0.2,0.3),np.arange(-11,1,1)]
    tt_cmap   = cmc.batlow
    tt_width = 1.5
     
    pp_levels = np.arange(0.3,1.2,0.3)
    pp_color,pp_style,pp_width = 'black','dashed',1
    
    rr_levels = np.array([1,5,10,20,50,100,200])
    rr_cmap   = truncate_colormap(cmc.oslo_r,minval=0.1)
    rr_norm   = mpl.colors.BoundaryNorm(rr_levels, rr_cmap.N, clip=True)
    rr_alpha = 0.5
    
    
    qu_scale = 75 # m/s per plot width
    qu_key   = 5   # m/s
    qu_hl,qu_hal,qu_w = 3,3,0.003
    qu_label  = 'Wind speed ('+str(qu_key)+' m/s)'
    
    qu_props = {'headlength':qu_hl,'headaxislength':qu_hal,
                'width':qu_w,'pivot':'tail','scale':qu_scale,
                'transform':transform,'zorder':5}
    
    titles = ['a)','b)']
    
    ii_wxt = meta_data['STATION'].str.endswith('w')
    ii_sup = meta_data['STATION'].str.contains('001') | \
             meta_data['STATION'].str.contains('098') | \
             meta_data['STATION'].str.contains('100')    
    
    
    # Main plot
    fig,ax = plt.subplots(1,2,figsize=(10,5),dpi=400,
                          subplot_kw={'projection':transform})
    
    for i,t in enumerate([dtime1,dtime2]):
        
        it = dtime.get_loc(t)
    
        ax[i].set_extent([minlon,maxlon,minlat,maxlat],transform)
        ax[i].set_aspect(aspect,adjustable='box')
        
        # TT perturbation   
        tt_norm = mpl.colors.BoundaryNorm(tt_levels[i], tt_cmap.N, clip=True) 
        tt = ax[i].contour(lon_data,lat_data,TT_data[:,:,it],
                           levels=tt_levels[i],norm=tt_norm,cmap=tt_cmap,
                           linewidths=tt_width,transform=transform)
        ax[i].clabel(tt,fmt=clabel_fmt[i],fontsize=fs-2)
        
        # PP perturbation
        pp = ax[i].contour(lon_data,lat_data,PP_data[:,:,it],
                           colors=pp_color,levels=pp_levels,
                           linestyles=pp_style,linewidths=pp_width,
                           transform=transform)
        ax[i].clabel(pp,fmt='%1.1f',fontsize=fs-2)
        
        # Radar rainfall
        ax[i].contourf(lon_data_rad,lat_data_rad,RR_data[:,:,it],
                       levels=rr_levels,norm=rr_norm,cmap=rr_cmap,
                       transform=transform,alpha=rr_alpha)
        
        # Station points
        ax[i].scatter(meta_data['LON'],meta_data['LAT'],c='dimgrey',marker='o',s=10,
                      transform=transform)    
        
        # Wind arrows
        qu = ax[i].quiver(meta_data[ii_wxt]['LON'].to_numpy(dtype=float),
                          meta_data[ii_wxt]['LAT'].to_numpy(dtype=float),
                          U_data.loc[t].to_numpy(dtype=float),
                          V_data.loc[t].to_numpy(dtype=float),
                          color='black',**qu_props)
        
        stat_max = np.sqrt(U_data.loc[t]**2+V_data.loc[t]**2).\
                   sort_values(ascending=False).head(3).index
        ii_max = [meta_data[meta_data['STATION'] == s].index[0] for s in stat_max]           
                   
        ax[i].quiver(meta_data.loc[ii_max,'LON'].to_numpy(dtype=float),
                     meta_data.loc[ii_max,'LAT'].to_numpy(dtype=float),
                     U_data.loc[t,stat_max].to_numpy(dtype=float),
                     V_data.loc[t,stat_max].to_numpy(dtype=float),
                     color='red',**qu_props)
        
        # Markers for supersites
        ax[i].scatter(meta_data.loc[ii_sup,'LON'],meta_data.loc[ii_sup,'LAT'],
                      marker='o',s=15,facecolors='magenta',transform=transform) 
        
        if i == 0:
             ax[i].quiverkey(qu,0.10,0.955,qu_key,qu_label,coordinates='axes',
                             labelpos='E',fontproperties=dict(size=fs))

    
        ax[i].set_title(titles[i],loc='right',fontsize=fs+2)

    rr_pax = mpl.cm.ScalarMappable(norm=rr_norm, cmap=rr_cmap)    
    make_colorbar(fig,ax[1],rr_pax,'Rainfall rate (mm h$^{-1}$)',rr_levels,
                  fs=fs,width=0.02,alpha=rr_alpha)
    
    plotname = pdir+'jogi_snapshots.png'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!')  
    return

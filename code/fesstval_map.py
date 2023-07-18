# -*- coding: utf-8 -*-
"""
@author: Bastian Kirsch (bastian.kirsch@uni-hamburg.de)

Code to plot station map of FESSTVaL 2021

Dependences on non-standard software:
- fesstval_routines.py 

Required meta data files:
- stations_fesstval.txt
- gadm41_DEU_1.shp (available at https://gadm.org/download_country.html)

Last updated: 16 May 2023
"""

print('*********')
print('Start')
print(' ')

import numpy as np
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy as ctp
import cartopy.io.img_tiles as cimgt
import fesstval_routines as fst

t_start = dt.datetime.now()

#----------------------------------------------------------------------------
# Settings
plot_map    = False
type_plot   = 'paper' # 'paper', 'radar'  
fig_numbers = True    # Name files "Kirsch_Figxx.pdf" if True

#----------------------------------------------------------------------------
# Paths and meta data files
maindir       = '.'
plotdir       = maindir+'Cold-Pools/Plots/FESSTVaL/'
meta_file2021 = maindir+'FESSTVaL/stations_fesstval.txt'
shapefile     = maindir+'Shapefile_Germany/gadm41_DEU_shp/gadm41_DEU_1.shp'

if type_plot == 'paper':
    plotdir = maindir+'Cold-Pools/Plots/Paper_CP_Morphology/'

#----------------------------------------------------------------------------
# Read network locations
stations = fst.fesstval_stations('',include_serial=True,metafile=meta_file2021)

ii_apo   = stations['STATION'].str.endswith('a')  
ii_wxt   = stations['STATION'].str.endswith('w')
ii_sup   = stations['STATION'].str.contains('S')

lon_apo,lat_apo  = stations['LON'][ii_apo].values,stations['LAT'][ii_apo].values
lon_wxt,lat_wxt  = stations['LON'][ii_wxt].values,stations['LAT'][ii_wxt].values


stat_apo = stations['STATION'][ii_apo]
stat_wxt = stations['STATION'][ii_wxt]

nstat_apo = ii_apo.sum()
nstat_wxt = ii_wxt.sum()

nstat_apo_sup    = (ii_apo & ii_sup).sum()
nstat_apo_street = (ii_apo & ~ii_sup).sum()
nstat_wxt_sup    = (ii_wxt & ii_sup).sum()
nstat_wxt_street = (ii_wxt & ~ii_sup).sum()

lat_radar,lon_radar,radius_radar = 52.16761,14.12048,20
n_angles = 100
angles   = np.linspace(0,360,n_angles)
lat_radar_circle,lon_radar_circle = np.zeros(n_angles)*np.nan, np.zeros(n_angles)*np.nan
for a in range(n_angles):
    lat_radar_circle[a],lon_radar_circle[a] = fst.geo_line(lat_radar,lon_radar,
                                                           radius_radar,angles[a])   

lat_mol,lon_mol = 52.20907, 14.12162
lat_bkh,lon_bkh = 52.20132, 14.19183


#----------------------------------------------------------------------------
fs = 10 #fontsize of plot

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
mpl.rcParams['legend.frameon'] = True

apollo_col  = 'royalblue'
wxt_col     = 'crimson'
radar_col   = 'black'
marker_size = 5

minlon, maxlon, reslon = 13.85, 14.40, 0.2
minlat, maxlat ,reslat = 52.0, 52.34, 0.1
zoom = 11
lon_ticks = np.arange(13.8,14.6,0.2)
lat_ticks = np.arange(51.9,52.5,0.1)


if plot_map:  
    
    # Setting up projection, background map and shapefile
    transform = ctp.crs.PlateCarree()
    plotmap = ctp.io.img_tiles.Stamen('terrain-background')
    gadm_shapes = list(ctp.io.shapereader.Reader(shapefile).geometries())
    
    
    print('Plotting FESSTVaL 2021 Map')
    fig = plt.figure(figsize=(6.2,6.0),dpi=300)

    ax = plt.axes(projection=plotmap.crs)
    ax.set_extent([minlon,maxlon,minlat,maxlat],transform)
    ax.add_geometries(gadm_shapes,transform,edgecolor='black',facecolor='None',linewidth=0.3)
    ax.add_image(plotmap,zoom,alpha=1.0)
    
    # Station markers       
    ax.plot(lon_apo,lat_apo,'o',color=apollo_col,markersize=marker_size,
            transform=transform,label=r'APOLLO ($n$='+str(nstat_apo)+')')   
    ax.plot(lon_wxt,lat_wxt,'^',color=wxt_col,markersize=marker_size,
            transform=transform,label=r'WXT ($n$='+str(nstat_wxt)+')',)      
    
    # Radar marker and circle
    ax.plot([lon_radar],[lat_radar],'o',
             markersize=marker_size+6,markeredgewidth=2,
             markeredgecolor=radar_col,markerfacecolor='None',
             transform=transform,label='X-band radar')
    plt.plot(lon_radar_circle,lat_radar_circle,linestyle='dashed',
             color=radar_col,linewidth=1,transform=transform)
    
    # Lat-lon grid lines
    gl = ax.gridlines(crs=transform,draw_labels=True,linewidth=0.5,color='black', 
                      alpha=0.3,linestyle='dotted') 
    gl.xlabels_bottom = True
    gl.xlabels_top    = False
    gl.ylabels_left   = True
    gl.ylabels_right  = False
    gl.xlocator = mpl.ticker.FixedLocator(lon_ticks)
    gl.ylocator = mpl.ticker.FixedLocator(lat_ticks)
    gl.xformatter = ctp.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = ctp.mpl.gridliner.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': fs, 'color': 'black'}
    gl.ylabel_style = {'size': fs, 'color': 'black'}

    if type_plot == 'radar':
        # MOL Symbol    
        ax.plot([lon_mol],[lat_mol],'^',
                  markersize=marker_size+6,markeredgewidth=2,
                  markeredgecolor='black',markerfacecolor='None',
                  transform=transform,label='MOL-RAO',zorder=0)   
        
    if type_plot == 'paper':
        # Supersite markers
        ax.plot([lon_mol,lon_bkh],[lat_mol,lat_bkh],'x',markeredgewidth=1.5,
                 markersize=marker_size+3,markeredgecolor='black',
                 transform=transform)
        ax.text(lon_mol+0.01,lat_mol+0.01,'L',fontsize=fs+2,ha='center',va='center',
                transform=transform) 
        ax.text(lon_bkh+0.02,lat_bkh,'B',fontsize=fs+2,ha='center',va='center',
                transform=transform) 
        ax.text(lon_radar+0.015,lat_radar-0.015,'F',fontsize=fs+2,ha='center',va='center',
                transform=transform)

    # Scale
    scale_lon,scale_lat,scale_len,scale_lw = 13.87,52.02,10,1.3
    scale_lon_end = fst.geo_line(scale_lat,scale_lon,scale_len,90)[1]
    ax.plot([scale_lon,scale_lon_end],[scale_lat,scale_lat],linestyle='solid',
            linewidth=scale_lw,color='black',transform=transform)
    for ll in [scale_lon,scale_lon_end]:
        ax.plot([ll,ll],[scale_lat-0.003,scale_lat+0.003],linestyle='solid',
                linewidth=scale_lw,color='black',transform=transform) 
    ax.text(scale_lon,scale_lat+0.008,str(scale_len)+' km',transform=transform,
            horizontalalignment='left',fontsize=fs)   

    ax.legend(fontsize=fs,loc='upper left')
    
    year_str = dt.datetime.now().strftime('%Y') 
    ax.text(0.01,0.01,'Map tiles by Stamen Design, under CC BY 3.0. '+\
                      '© OpenStreetMap contributors '+year_str+'. Distributed under'+\
                      ' the Open Data Commons Open Database License (ODbL) v1.0',
                      transform=ax.transAxes,fontsize=fs-6)

    pname_ext = '_'+type_plot
    plotname = plotdir+'fesstval_map'+pname_ext+'.png'
    if fig_numbers: plotname = plotdir+'Kirsch_Fig01.pdf'
    fig.savefig(plotname,bbox_inches='tight')
    plt.close()    
    print('Plot done!')   
      
    
#----------------------------------------------------------------------------
print('*********')
fst.print_runtime(t_start)
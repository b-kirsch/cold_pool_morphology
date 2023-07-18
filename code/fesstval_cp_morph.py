# -*- coding: utf-8 -*-
"""
@author: Bastian Kirsch (bastian.kirsch@uni-hamburg.de)

Script to analyze cold pool morphology in observation data 
from FESSTVaL 2021 (and FESST@HH 2020)

- Reads interpolated APOLLO and WXT station and X-band radar data (level 3)
- Performs cluster analysis to indentify cold pools objects
- Reads APOLLO and WXT station data (level 2)
- Generates plots for CP morphology paper

Dependences on non-standard software:
- fesstval_routines.py 
- cp_spatial_analysis.py
- fesstval_plots.py

Required meta data files:
- stations_fessthh.txt
- stations_fesstval.txt
- cold_pools_fessthh.txt
- cold_pools_fesstval.txt
    

Last updated: 20 June 2023
"""

import numpy as np
import pandas as pd
import datetime as dt
import skimage.measure as skm
import scipy.interpolate as sci 
import scipy.stats as scs
from copy import deepcopy
import fesstval_routines as fst
import cp_spatial_analysis as cps
import fesstval_plots as fpl


t_run = dt.datetime.now()

#----------------------------------------------------------------------------
# Paths and meta data files
maindir     = '.'

datadir_apo = maindir+'APOLLO_data/level3/'
datadir_wxt = maindir+'WXT_data/level3/' 
datadir_rad = maindir+'Radar_data/level3/'

meta_file2020 = maindir+'FESSTVaL/FESSTHH/stations_fessthh.txt'
meta_file2021 = maindir+'FESSTVaL/stations_fesstval.txt'
cp_file2020   = maindir+'FESSTVaL/FESSTHH//cold_pools_fessthh.txt'
cp_file2021   = maindir+'FESSTVaL/cold_pools_fesstval.txt'
    
#----------------------------------------------------------------------------  
# Basic settings
# Time range for begin of cold pool events
start_time    = dt.datetime(2021,6,29,0)  #dt.datetime(2021,5,17,0)   
end_time      = dt.datetime(2021,6,29,23)  #dt.datetime(2021,8,27,23)

# Analyses to perform
analyze_cases = False # Only run days included in cases
read_radar    = False
read_level2   = False

# Plots to produce (set fig_numbers in fesstval_plots.py!!!)
plot_snapshot        = False # Fig04
plot_time_dist       = False # Fig09
plot_bams            = False # Plot for BAMS overview paper

plot_overview_morph  = False # Fig06
plot_stats_bound     = False # Fig05
plot_stats_morph     = False # Fig07
plot_stats_structure = False # Fig08

plot_growth_time     = False # Fig10
plot_growth_rain     = False # Fig11
plot_growth_dc       = False # Fig12
plot_growth_dc_local = False # Fig13


#----------------------------------------------------------------------------  
# Advanced settings
dT_lim           = -2 # -2 (K) for CP definition

rr_lim           = 0.1 # (mm/h)

min_area_cp      = 10   # 10 (km2) for definition of CP cluster

bound_lim_stats  = 0.25 # (-) max. fraction of CP boundary touching the network boundary
bound_lim_growth = 0.67 # (-) max. fraction of CP boundary touching the network boundary

t_smooth         = 30 # (s) Smoothing of WXT station data (half-window length)

#----------------------------------------------------------------------------
# Some standard settings for analysis
if plot_stats_bound or plot_stats_morph or plot_stats_structure:
    analyze_cases = False
    start_time    = dt.datetime(2021,5,17,0) 
    end_time      = dt.datetime(2021,8,27,23) 

if plot_growth_time or plot_growth_rain:
    analyze_cases = True
    start_time    = dt.datetime(2020,8,10,0) 
    end_time      = dt.datetime(2021,8,27,23) 
    
if plot_snapshot: 
    read_radar  = True
    read_level2 = True
    dT_lim      = -1 # to allow for drawing of -2 isoline
    
if plot_bams: 
    dT_lim = 0
    read_level2 = True   
    
if plot_growth_dc_local:
    read_level2 = True
    
#---------------------------------------------------------------------------- 
#----------------------------------------------------------------------------
print('Analyzing FESSTVaL level 3 data')
print('Start time   : '+start_time.strftime('%Y-%m-%d %H:%M'))
print('End time     : '+end_time.strftime('%Y-%m-%d %H:%M'))
print('Analyze cases: '+str(analyze_cases))

days = pd.date_range(start=start_time,end=end_time,freq='d')

cp_fesstval_2020 = fst.fesstval_cold_pools(cpfile=cp_file2020)
cp_fesstval_2021 = fst.fesstval_cold_pools(cpfile=cp_file2021)
cp_fesstval = cp_fesstval_2020.append(cp_fesstval_2021)
cp_fesstval['INDEX'] = np.arange(cp_fesstval.shape[0])


dir_kwargs_l3 = {'datadir_apollo':datadir_apo,'datadir_wxt':datadir_wxt}
dir_kwargs_l2 = {'datadir_apollo':datadir_apo.replace('level3','level2'),
                 'datadir_wxt':datadir_wxt.replace('level3','level2')}

#Read meta data
meta_data_dict = {2020:fst.fessthh_stations('l2',metafile=meta_file2020),
                  2021:fst.fesstval_stations('',metafile=meta_file2021)}   

mdate_dict = {2020:dt.date(2020,8,10),
              2021:dt.date(2021,6,29)}  

grid_dict = {2020:fst.read_fesstval_level3('DT',2020,mdate_dict[2020].month,
                                           mdate_dict[2020].day,return_meta=True,
                                           **dir_kwargs_l3),
             2021:fst.read_fesstval_level3('DT',2021,mdate_dict[2021].month,
                                           mdate_dict[2021].day,return_meta=True,
                                           **dir_kwargs_l3)}

grid_rad_dict = {2020:fst.read_radar_level3(2020,mdate_dict[2020].month,
                                            mdate_dict[2020].day,return_meta=True,
                                            datadir=datadir_rad),
                 2021:fst.read_radar_level3(2021,mdate_dict[2021].month,
                                            mdate_dict[2021].day,return_meta=True,
                                            datadir=datadir_rad)}

# Define cold pool cases to be analyzed
cases = pd.DataFrame(columns=['START_GROWTH','END_GROWTH'])
cases.loc[27] = [dt.datetime(2020,8,10,13,7),dt.datetime(2020,8,10,14,13)]
cases.loc[54] = [dt.datetime(2021,6,26,9,12),dt.datetime(2021,6,26,10,6)]
cases.loc[55] = [dt.datetime(2021,6,29,13,24),dt.datetime(2021,6,29,14,35)]
cases.loc[62] = [dt.datetime(2021,7,25,13,46),dt.datetime(2021,7,25,15,10)]

for i in cases.index:
    cases.loc[i,'NAME'] = cp_fesstval[cp_fesstval['INDEX'] == i]['NAME'].values[0]
  
#----------------------------------------------------------------------------
# Define functions

# Convert wind speed (FF) and wind direction (DD) into u,v components
def FFDD_to_UV(FF_in,DD_in):
    if type(FF_in) in [int,float,np.float64]: FF_in = np.array([FF_in])
    if type(DD_in) in [int,float,np.float64]: DD_in = np.array([DD_in])
    if isinstance(FF_in,pd.Series) or isinstance(FF_in,pd.DataFrame): FF_in = FF_in.to_numpy(float)
    if isinstance(DD_in,pd.Series) or isinstance(DD_in,pd.DataFrame): DD_in = DD_in.to_numpy(float)
    if len(FF_in) != len(DD_in):
        print('FF and DD data does not have the same length!')
        return
    
    with np.errstate(invalid='ignore'):
        ff_invalid    = (FF_in < 0)
        dd_invalid0   = (DD_in < 0)
        dd_invalid360 = (DD_in > 360)
        dd0           = (DD_in == 0)
        dd90          = (DD_in == 90)
        dd270         = (DD_in == 270)
        dd360         = (DD_in == 360)
    
    UU_out = np.zeros_like(FF_in) * np.nan
    VV_out = np.zeros_like(FF_in) * np.nan

    deg2rad = np.pi/180.
    UU_out  = np.multiply(-FF_in,np.sin(np.multiply(DD_in,deg2rad)))
    VV_out  = np.multiply(-FF_in,np.cos(np.multiply(DD_in,deg2rad)))

    UU_out[dd0|dd360]  = 0
    VV_out[dd90|dd270] = 0
    
    UU_out[ff_invalid|dd_invalid0|dd_invalid360] = np.nan
    VV_out[ff_invalid|dd_invalid0|dd_invalid360] = np.nan

    if len(UU_out) == 1: UU_out = float(UU_out)
    if len(VV_out) == 1: VV_out = float(VV_out)
    
    return UU_out,VV_out 

# Equivalent diameter of circle of given area
def equivalent_diameter(area):
    return np.sqrt((4*area)/np.pi)

# Linear radial temperature gradient
def dT_rad_grad(d,r_equi,dT_center,dT_edge=dT_lim):
    return dT_center - (dT_center-dT_edge) * (d/r_equi)

# Propogation velocity of denisty current of given temperature deficit and height
def vel_dens_cur(dT,H,T0=20,k=0.7,g=9.81):
    # velocity of equivalent radius
    return k * np.sqrt(g * H * (-dT/(T0+273.15)))

# Radial expansion velocity of circular cold pool for given constant areal 
# growth rate A_rate (km2/min)
A_min = 10    # (km2)
A_max = 1000  # (km2)
 
def vel_rad_exp(A_rate,A_start=A_min,A_stop=A_max,n_time=100,A_data=[]):
    A_rate  *= (1e6/60) # km2/min -> m2/s
    A_start *= 1e6      # km2 -> m2
    A_stop  *= 1e6      # km2 -> m2
    
    if len(A_data) == 0:
        t_max    = (A_stop-A_start)/A_rate
        time     = np.linspace(0,t_max,n_time)
        area     = A_start + A_rate * time
    else:
        area = A_data * 1e6   # km2 -> m2
        time = (area-A_start)/A_rate
        
    area    *= 1e-6     # m2 -> km2    
    u_radius = A_rate/(2*np.sqrt(np.pi)*np.sqrt(A_start+A_rate*time))
    
    return area,u_radius   

# Fit function from Kruse et al., 2022 (Fig. 7)
def dc_kruse(x,a=478,g=9.81,k=0.7):
    return k * np.sqrt(a*g*x) 

#----------------------------------------------------------------------------
cluster_props = pd.DataFrame()
i_all    = 0
i_growth = {}
rr_accu  = {} 
for i in cases.index: 
    i_growth[i] = 0
    rr_accu[i]  = 0

days_cases     = pd.DatetimeIndex(cases['START_GROWTH'].dt.date)
days_intersect = pd.DatetimeIndex(days.date).intersection(days_cases)
if analyze_cases: days = days_intersect    
calc_network_bound = {2020:False,2021:False} # Calculate network boundary only once 

dist_min,dist_max,dist_res = 0,27,1
dist_bins = np.arange(dist_min,dist_max+dist_res,dist_res) * 1000
dist_min_stat,dist_max_stat,dist_res_stat = 0,1,0.1
dist_bins_stat = np.arange(dist_min_stat,dist_max_stat+dist_res_stat,dist_res_stat)

dT_dist_case = pd.DataFrame(columns=dist_bins[:-1]/1000.+dist_res/2.) # Abs. distance
dP_dist_case = pd.DataFrame(columns=dist_bins[:-1]/1000.+dist_res/2.)
dT_dist_stat = pd.DataFrame(columns=dist_bins_stat[:-1]+dist_res_stat/2.) # Rel. distance

# Loop over days
for d in days:
    date_args = (d.year,d.month,d.day)
    check = fst.read_fesstval_level3('DT',*date_args,**dir_kwargs_l3,
                                     check_file=True)
    if not check: continue

    print('')
    print(d.strftime('Reading level 3 data for %Y-%m-%d'))    
    
    # Reading data
    dT_data  = fst.read_fesstval_level3('DT',*date_args,**dir_kwargs_l3)
    dP_data  = fst.read_fesstval_level3('DP',*date_args,**dir_kwargs_l3,ds_version=0)
    icp_data = fst.read_fesstval_level3('DT',*date_args,return_meta=True,
                                        **dir_kwargs_l3)['event_id']
    
    RR_data = pd.DataFrame()
    if read_radar:
        RR_data = fst.read_radar_level3(d.year,d.month,d.day,
                                        mute=False,datadir=datadir_rad)
        
    TT_ref    = fst.read_fesstval_level3('TT_REF',*date_args,**dir_kwargs_l3)   
    
    # Time and space data    
    ntime = dT_data.shape[2]
    freq  = int(1440/ntime)
    time  = pd.date_range(start=dt.datetime(*date_args),periods=ntime,
                          freq=str(freq)+'min')
    x_meshgrid     = grid_dict[d.year]['x_meshgrid']
    y_meshgrid     = grid_dict[d.year]['y_meshgrid']
    mask_meshgrid  = grid_dict[d.year]['mask_meshgrid']
    lon_ref        = grid_dict[d.year]['lon_ref']
    lat_ref        = grid_dict[d.year]['lat_ref']
    x_meshgrid_rad = grid_rad_dict[d.year]['x_meshgrid']
    y_meshgrid_rad = grid_rad_dict[d.year]['y_meshgrid']
    x_res_rad      = x_meshgrid_rad[0,1] - x_meshgrid_rad[0,0]
    y_res_rad      = y_meshgrid_rad[1,0] - y_meshgrid_rad[0,0] 
    pixel_area_rad = (x_res_rad/1000.) * (y_res_rad/1000.) # (km2)
    
    # Reading level 2 station data
    if read_level2:
        # Wind speed and direction
        FF_wxt = fst.read_fesstval_level2('w','FF',*date_args,**dir_kwargs_l2)
        DD_wxt = fst.read_fesstval_level2('w','DD',*date_args,**dir_kwargs_l2)
        
        UU,VV = FFDD_to_UV(FF_wxt,DD_wxt)
        UU_wxt = pd.DataFrame(UU,index=FF_wxt.index,columns=FF_wxt.columns)
        VV_wxt = pd.DataFrame(VV,index=FF_wxt.index,columns=FF_wxt.columns)
        UU_wxt = UU_wxt.rolling(int(2*t_smooth/10+1),center=True,
                                min_periods=int(t_smooth/10),axis=0).mean() 
        VV_wxt = VV_wxt.rolling(int(2*t_smooth/10+1),center=True,
                                min_periods=int(t_smooth/10),axis=0).mean() 
        
        FF_wxt = FF_wxt.rolling(int(2*t_smooth/10+1),center=True,
                                min_periods=int(t_smooth/10),axis=0).mean()
        
        # Reference value = mean over first hour of event
        ii_ref = np.where(np.isfinite(TT_ref))[0][:60]
        FF_ref = FF_wxt.loc[time[ii_ref]].mean().mean() 

    # Loop over analysis time steps
    print('Analyzing clusters')
    for it,t in enumerate(time):
        ii_cp = dT_data[:,:,it] <= dT_lim
        if ii_cp.sum() == 0: continue
        if t.minute == 0: print(t.strftime('%H:%M'))
        
        icp = icp_data[it]
        i_all_t = [] # all indices for time step
        
        cl = cps.cluster_analysis(x_meshgrid,y_meshgrid,dT_data[:,:,it],dT_lim)
        
        dT_data[:,:,it][~ii_cp] = np.nan
        dP_data[:,:,it][~ii_cp] = np.nan
        
        if not calc_network_bound[d.year]: 
            network_bound = cl.boundary(True,mask_meshgrid)
            calc_network_bound[d.year] = True
        
        cluster_data = cl.data()
        cluster_list = list(set(np.ndarray.flatten(cluster_data[cluster_data>0])))
        
        for cid in cluster_list:
            c_area = cl.area(cid)
            if c_area < min_area_cp: continue
            cx,cy = cl.center(cid)
            clon,clat = cl.center(cid,return_lonlat=True,
                                  lon_ref=lon_ref,lat_ref=lat_ref,hav=True)
            equi_diam = equivalent_diameter(c_area)
            dT_mean   = cl.mean_val(cid)
            
            cluster_props.loc[i_all,'ICP']           = icp
            cluster_props.loc[i_all,'CID']           = cid
            cluster_props.loc[i_all,'DATE_TIME']     = t
            cluster_props.loc[i_all,'EVENT_TIME']    = it*freq
            cluster_props.loc[i_all,'CP_AREA']       = c_area
            cluster_props.loc[i_all,'CP_EQUI_DIAM']  = equi_diam
            cluster_props.loc[i_all,'DT_MIN']        = cl.min_val(cid)
            cluster_props.loc[i_all,'DT_MEAN']       = dT_mean
            cluster_props.loc[i_all,'CP_CENTER_X']   = cx
            cluster_props.loc[i_all,'CP_CENTER_Y']   = cy
            cluster_props.loc[i_all,'CP_CENTER_LON'] = clon
            cluster_props.loc[i_all,'CP_CENTER_LAT'] = clat
            
            if read_level2:
                cluster_props.loc[i_all,'DT_MEAN_REL'] = -dT_mean/(TT_ref[it]+273.15)
                cluster_props.loc[i_all,'FF_MEAN']     = (FF_wxt.loc[t] - FF_ref).mean()
                cluster_props.loc[i_all,'FF_MAX']      = (FF_wxt.loc[t] - FF_ref).max()         
               
            # Calculate aspect ratio of minimum bounding ellipsoid            
            regionprops = pd.DataFrame(skm.regionprops_table(cluster_data,
                                        properties=['label','major_axis_length',
                                                    'minor_axis_length']))
            regionprops.set_index('label',inplace=True)
            major = regionprops.loc[cid]['major_axis_length']
            minor = regionprops.loc[cid]['minor_axis_length']
            if minor > 0: cluster_props.loc[i_all,'CP_ASPECT'] = major/minor
            
            # Calculate proportion of cluster boundary that matches with network boundary
            c_bound   = cl.boundary(cid,cluster_data)
            fac_bound = np.sum(c_bound & network_bound)/np.sum(c_bound)
            cluster_props.loc[i_all,'CP_BOUND'] = fac_bound
            
            # Calculating radial temperature structure
            if plot_time_dist or plot_stats_structure: 
                cdist_meshgrid = np.sqrt((x_meshgrid-cx)**2+(y_meshgrid-cy)**2)
                df_dist = pd.DataFrame(np.ravel(cdist_meshgrid),columns=['DIST'])
                
                cid_mask = cluster_data == cid # Mask all other clusters
                
                dT_cid = deepcopy(dT_data[:,:,it])
                dT_cid[~cid_mask] = np.nan
                df_dist['DT'] = np.ravel(dT_cid)
                
            if plot_time_dist:
                dP_cid = deepcopy(dP_data[:,:,it])
                dP_cid[~cid_mask] = np.nan
                df_dist['DP'] = np.ravel(dP_cid) 
            
                groups_dist_abs = pd.cut(df_dist['DIST'],dist_bins)
                dT_dist_case.loc[i_all] = df_dist['DT'].groupby(groups_dist_abs).mean().values
                dP_dist_case.loc[i_all] = df_dist['DP'].groupby(groups_dist_abs).mean().values
                
            if plot_stats_structure:    
                equi_radius = equi_diam/2. * 1000
                groups_dist_rel = pd.cut(df_dist['DIST']/equi_radius,dist_bins_stat)
                dT_dist_stat.loc[i_all] = df_dist['DT'].groupby(groups_dist_rel).mean().values
                
                # dT at CP center
                dT_interp = sci.LinearNDInterpolator(np.column_stack([x_meshgrid.ravel(),
                                                                      y_meshgrid.ravel()]),
                                                     df_dist['DT'])
                dT_center = dT_interp(cx,cy)*1
                cluster_props.loc[i_all,'DT_CENTER'] = dT_center
                
                # Goodness of fit of linear radial model
                dT_lin   = dT_rad_grad(df_dist['DIST'],equi_radius,dT_center)
                rmse_lin = np.sqrt(np.nanmean((dT_lin-df_dist['DT'])**2)) 
                cluster_props.loc[i_all,'RMSE_LIN_MODEL'] = rmse_lin

                fin = df_dist['DT'].notnull() & dT_lin.notnull() 
                if fin.sum() >= 2:
                    corr_lin = scs.pearsonr(df_dist.loc[fin,'DT'],dT_lin[fin])[0]
                    cluster_props.loc[i_all,'CORR_LIN_MODEL'] = corr_lin
                
            
            # Counter for index of all clusters
            i_all_t.append(i_all)
            i_all += 1
        
        if len(i_all_t) == 0: continue
        
        # Sort clusters from same scene by size (CID_SORTED=1 for largest cluster)    
        cid_sorted = pd.to_numeric(cluster_props.loc[i_all_t,'CP_AREA']).sort_values(ascending=False).index
        cluster_props.loc[cid_sorted,'CID_SORTED'] = np.arange(len(cid_sorted))+1
        i_max   = cid_sorted[0] # Index of largest cluster
        cid_max = cluster_props.loc[i_max,'CID'] # Cluster index of largest cluster

        # Analyzing radar data
        if RR_data.shape[0] > 0:            
            # Calculate RR within CP cluster
            cl_interp = sci.NearestNDInterpolator(np.column_stack([x_meshgrid.ravel(),
                                                                   y_meshgrid.ravel()]),
                                                  cluster_data.ravel())
            cluster_rad = cl_interp(x_meshgrid_rad,y_meshgrid_rad)
            
            ii_rr = (RR_data[:,:,it] >= rr_lim) & (cluster_rad == cid_max)
            rr_area = np.nansum(ii_rr) * pixel_area_rad
            rr_mean = np.nanmean(RR_data[:,:,it][ii_rr]) if rr_area > 0 else 0
            cluster_props.loc[i_max,'RR_AREA_CP'] = rr_area # (km2)
            cluster_props.loc[i_max,'RR_MEAN_CP'] = rr_mean # (mm/h)
            
            # RR within entire radar domain (only points stronger than rr_lim)
            ii_rr_all = (RR_data[:,:,it] >= rr_lim)
            rr_area_all = np.nansum(ii_rr_all) * pixel_area_rad
            rr_mean_all = np.nanmean(RR_data[:,:,it][ii_rr_all]) if rr_area_all > 0 else 0
            cluster_props.loc[i_max,'RR_AREA_RADAR'] = rr_area_all # (km2)
            cluster_props.loc[i_max,'RR_MEAN_RADAR'] = rr_mean_all # (mm/h)
            cluster_props.loc[i_max,'RR_MAX_RADAR'] = np.nanmax(RR_data[:,:,it]) # (mm/h)
            
            # RR within entire radar domain (all points, also non-rain)
            #cluster_props.loc[i_max,'RR_MEAN_ALL'] = np.nanmean(RR_data[:,:,it]) # (mm/h)     
            
        
        # Defining growth phase 
        if icp in cases.index:
            start_growth = cases.loc[icp]['START_GROWTH']
            end_growth   = cases.loc[icp]['END_GROWTH'] 
            
            if (t >= start_growth) & (t <= end_growth) & \
                (cluster_props.loc[i_max,'CP_BOUND'] <= bound_lim_growth):
                cluster_props.loc[i_max,'GROWTH']      = True
                cluster_props.loc[i_max,'GROWTH_TIME'] = i_growth[icp] * freq
                i_growth[icp] += 1
                
                if not read_radar: continue
                rr_area = cluster_props.loc[i_max,'RR_AREA_RADAR']
                rr_mean = cluster_props.loc[i_max,'RR_MEAN_RADAR']
                rr_accu[icp] += (rr_mean*60*freq/3600 * rr_area*1000**2) # (mm = l/m2) * m2 
                cluster_props.loc[i_max,'RR_ACCU_ABS'] = rr_accu[icp]

#----------------------------------------------------------------------------
# Statistical analysis of clusters
n_objects_all = cluster_props['CP_AREA'].notnull().sum()
n_events_all  = cluster_props[cluster_props['CP_AREA'].notnull()]['ICP'].nunique()

mask_bound = cluster_props['CP_BOUND'] > bound_lim_stats
cluster_props_filtered = cluster_props.mask(mask_bound)
n_objects_filtered = cluster_props_filtered['CP_AREA'].notnull().sum()
n_events_filtered  = cluster_props_filtered[cluster_props_filtered['CP_AREA'].notnull()]['ICP'].nunique()
ii_filtered = cluster_props_filtered['ICP'].notnull()

diam_quartiles = pd.qcut(cluster_props_filtered['CP_EQUI_DIAM'],4,
                         labels=['Q1','Q2','Q3','Q4'])
dT_quartiles = pd.qcut(cluster_props_filtered['DT_MEAN'],4,
                       labels=['Q1','Q2','Q3','Q4'])

dT_dist_stat = dT_dist_stat.reindex(cluster_props.index).mask(mask_bound)

# Absolute gradient
cluster_props_filtered['DT_GRAD'] = (cluster_props_filtered['DT_CENTER']-dT_lim)/\
                                    (cluster_props_filtered['CP_EQUI_DIAM']/2.)
cluster_props_filtered.loc[dT_quartiles=='Q1','DT_GRAD'].median()
cluster_props_filtered.loc[diam_quartiles=='Q1','DT_GRAD'].median()                                    
                                    
# Object-mean dT based on linear radial model
cluster_props_filtered['DT_MEAN_LIN'] = 0.5*(cluster_props_filtered['DT_CENTER']+dT_lim)  

# Goodness of fit of linear radial model
# (cluster_props_filtered['RMSE_LIN_MODEL'] <= 1).sum()/n_objects_filtered   
# fin = cluster_props_filtered.loc['DT_MEAN'].notnull() & cluster_props_filtered.loc['DT_MEAN_LIN'].notnull()
# scs.pearsonr(cluster_props_filtered.loc[fin,'DT_MEAN'],cluster_props_filtered.loc[fin,'DT_MEAN_LIN'])[0]            
                                    
                                    
                                    

# Determine growth rate of cases
if analyze_cases:
    
    print('Calculating growth rates of selected cases')
    t_linreg = 5 # min (Half-window length for lin reg for growth rate)
    
    for icp in cases.index:
        ii_case = (cluster_props['ICP'] == icp) & (cluster_props['GROWTH'])
        growth_time = cluster_props.loc[ii_case,'GROWTH_TIME']
        cp_area = cluster_props.loc[ii_case,'CP_AREA'] 
        equi_radius = cluster_props.loc[ii_case,'CP_EQUI_DIAM']/2.
        rr_area = cluster_props.loc[ii_case,'RR_AREA_RADAR']
        
        for i in cluster_props[ii_case].index:
            if cp_area.loc[i-t_linreg:i+t_linreg].notnull().sum() <= t_linreg: continue
            # Change rate of area
            linreg_area = scs.linregress(growth_time.loc[i-t_linreg:i+t_linreg],
                                         cp_area.loc[i-t_linreg:i+t_linreg])
            cluster_props.loc[i,'DCP_AREA'] = linreg_area.slope # (km2/min)
            
            linreg_area_rr = scs.linregress(growth_time.loc[i-t_linreg:i+t_linreg],
                                            rr_area.loc[i-t_linreg:i+t_linreg])
            cluster_props.loc[i,'DRR_AREA'] = linreg_area_rr.slope # (km2/min)
            
            # Change rate of equivalent radius
            linreg_rad = scs.linregress(growth_time.loc[i-t_linreg:i+t_linreg],
                                        equi_radius.loc[i-t_linreg:i+t_linreg])
            cluster_props.loc[i,'DEQUI_RADIUS'] = linreg_rad.slope * (100/6) # (km/min) -> (m/s)
            
        dT_mean_smooth = cluster_props.loc[ii_case,'DT_MEAN'].rolling(2*t_linreg+1,
                                                                      center=True,
                                                                      min_periods=t_linreg).mean()
        cluster_props.loc[ii_case,'U_DC'] = vel_dens_cur(dT_mean_smooth,300)
        
        cluster_props.loc[ii_case,'U_R'] = vel_rad_exp(50,A_data=cp_area)[1]
        
        
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Plotting
print('')
if plot_overview_morph:
    # Select time steps to plot
    icp_list = cluster_props_filtered['ICP'].unique()
    icp_list = icp_list[~np.isnan(icp_list)]
    dtime_list = []
    center_lon_list = []
    center_lat_list = []
    dT_data_list = np.zeros((dT_data.shape[0],dT_data.shape[1],
                             len(icp_list))) * np.nan
    mask_list = np.zeros((dT_data.shape[0],dT_data.shape[1],
                             len(icp_list))) * np.nan
    
    for i,icp in enumerate(icp_list):
        ii_icp = cluster_props_filtered['ICP'] == icp
        ii_min = cluster_props_filtered.loc[ii_icp,'DT_MEAN'].idxmin()
        dtime = cluster_props_filtered.loc[ii_min,'DATE_TIME']
        dtime_list.append(dtime)
        
        # Read data
        dT_data = fst.read_fesstval_level3('DT',dtime.year,dtime.month,dtime.day,
                                           **dir_kwargs_l3)
        day_time  = pd.date_range(start=dt.datetime(dtime.year,dtime.month,dtime.day,0,0),
                                  periods=ntime,freq=str(freq)+'min')
        i_dtime = day_time.get_loc(dtime)
        ii_cp = dT_data[:,:,i_dtime] <= dT_lim
        dT_data[:,:,i_dtime][~ii_cp] = np.nan
        dT_data_list[:,:,i] = dT_data[:,:,i_dtime]
        
        # Determine center and masks of clusters
        ii_dtime = (cluster_props_filtered['DATE_TIME'] == dtime)
        clon = cluster_props_filtered.loc[ii_dtime,'CP_CENTER_LON'].values
        clat = cluster_props_filtered.loc[ii_dtime,'CP_CENTER_LAT'].values
        center_lon_list.append(clon)
        center_lat_list.append(clat)
        
        cl = cps.cluster_analysis(x_meshgrid,y_meshgrid,dT_data[:,:,i_dtime],dT_lim)
        cluster_data = cl.data()
        cid_dtime = cluster_props_filtered.loc[ii_dtime,'CID'].values
        cid_mask = np.zeros_like(cluster_data).astype(bool)
        for cid in cid_dtime: cid_mask[cluster_data == cid] = True
        mask_list[:,:,i] = cid_mask
     
    lon_meshgrid = grid_dict[start_time.year]['lon_meshgrid']
    lat_meshgrid = grid_dict[start_time.year]['lat_meshgrid']
    fpl.overview_morph(dtime_list,lon_meshgrid,lat_meshgrid,
                        meta_data_dict[2021],dT_data_list,mask_list,
                        center_lon_list,center_lat_list,dT_lim)

if plot_snapshot and (len(days) != 1):
    print('*** Snapshot plots only enabled for one day! ***')
    
if plot_snapshot and (len(days) == 1):
    
    pstart = dt.time(14,15)
    pend   = dt.time(14,15)
    pfreq  = 15 # min
    ptimes = pd.date_range(dt.datetime.combine(start_time.date(),pstart),
                            dt.datetime.combine(start_time.date(),pend),
                            freq=str(pfreq)+'min')
    plot_pp = False
    plot_uv = False
    
    meta_data        = meta_data_dict[start_time.year]
    lon_meshgrid     = grid_dict[start_time.year]['lon_meshgrid']
    lat_meshgrid     = grid_dict[start_time.year]['lat_meshgrid']
    lon_meshgrid_rad = grid_rad_dict[start_time.year]['lon_meshgrid']
    lat_meshgrid_rad = grid_rad_dict[start_time.year]['lat_meshgrid']
    
    it_start = time.get_loc(pstart)[0]
    it_end   = time.get_loc(pend)[0]
    dT_min   = np.floor(np.nanmin(dT_data[:,:,it_start:it_end+1]))

    print('')
    for t in ptimes:
        it = time.get_loc(t.to_pydatetime())
        if RR_data.shape[0] == 0: RR_data = dT_data * np.nan
        fpl.snapshot(t,lon_meshgrid,lat_meshgrid,lon_meshgrid_rad,lat_meshgrid_rad,
                     meta_data,dT_data[:,:,it],RR_data[:,:,it],
                     PP_data=dP_data[:,:,it],UU_data=UU_wxt.loc[t],
                     VV_data=VV_wxt.loc[t],plot_pp=plot_pp,plot_uv=plot_uv,
                     cp_lim=dT_lim,TT_min=dT_min,rr_min=rr_lim)
    
if plot_time_dist:
    # Extract data for time steps of largest cluster  
    ii_max       = cluster_props['CID_SORTED'] == 1
    time_cid_max = cluster_props.loc[ii_max,'DATE_TIME']
    dT_dist_time = dT_dist_case.loc[ii_max].set_index(time_cid_max)
    dP_dist_time = dP_dist_case.loc[ii_max].set_index(time_cid_max)
    
if plot_time_dist and (len(days) == 1): 
    plot_dp = False
    
    RR_mean = cluster_props.loc[ii_max,'RR_MEAN_RADAR'].set_axis(time_cid_max)

    growth_times = [cluster_props[cluster_props['GROWTH'] == True]['DATE_TIME'].iloc[0],
                    cluster_props[cluster_props['GROWTH'] == True]['DATE_TIME'].iloc[-1]]          
    
    fpl.time_distance(dist_bins,dT_dist_time,dP_dist_time,RR_mean,
                      plot_dp=plot_dp,growth_time=growth_times,freq=freq)    
    
if plot_stats_bound: 
    mask_meshgrid = grid_dict[2021]['mask_meshgrid']
    diam_limits   = [equivalent_diameter(min_area_cp),
                      equivalent_diameter(mask_meshgrid.sum())]
    fpl.stats_bound(cluster_props,cluster_props_filtered,
                    diam_limits,bound_lim_stats,dT_lim)
    
if plot_stats_morph:
    fpl.stats_morph(cluster_props_filtered,diam_quartiles,dT_quartiles)

if plot_stats_structure:
    fpl.stats_structure(dT_dist_stat,diam_quartiles,dT_quartiles)     

if plot_growth_time:
    fpl.growth_time(cluster_props,cases)    
    
if plot_growth_rain:
    fpl.growth_rain(cluster_props,cases,dT_lim=dT_lim)     
    
if plot_growth_dc:
    fpl.growth_density_current(cluster_props,cases,vel_dens_cur,vel_rad_exp) 

if plot_growth_dc_local:
    fpl.growth_density_current_local(cluster_props,cases,dc_kruse)         

if plot_bams and (len(days) == 1):
    meta_data = meta_data_dict[2021]
    lon_meshgrid     = grid_dict[2021]['lon_meshgrid']
    lat_meshgrid     = grid_dict[2021]['lat_meshgrid']
    lon_meshgrid_rad = grid_rad_dict[2021]['lon_meshgrid']
    lat_meshgrid_rad = grid_rad_dict[2021]['lat_meshgrid']
    fpl.fesstval_paper(time,lon_meshgrid,lat_meshgrid,lon_meshgrid_rad,lat_meshgrid_rad,
                       meta_data,dT_data,RR_data,dP_data,UU_wxt,VV_wxt) 
               
#----------------------------------------------------------------------------
print(' ')
print('*** Finshed! ***')
fst.print_runtime(t_run)
# -*- coding: utf-8 -*-
"""
@author: Bastian Kirsch (bastian.kirsch@uni-hamburg.de)

Script to prepare FESSTVaL 2021 (and FESST@HH 2020) network observations
for analysis

- Reads APOLLO and WXT station data (level 2)
- Calculates perturbations during cold pool events
- Sets up regular interpolation grid
- Performs spatial interpolation using kriging
- Writes interpolated data to netCDF files (level 3)

Dependences on non-standard software:
- fesstval_routines.py
- cp_spatial_analysis.py  

Required meta data files:
- standorte_fessthh.txt
- stations_fesstval.txt
- cold_pools_fessthh.txt
- cold_pools_fesstval.txt    


Last updated: 13 June 2023
"""

import numpy as np
import pandas as pd
import os
import datetime as dt
from scipy import stats as scs 
from netCDF4 import Dataset
import fesstval_routines as fst
import cp_spatial_analysis as cps

t_run = dt.datetime.now()

#----------------------------------------------------------------------------
# Paths and meta data files
maindir     = '.'

datadir_apo = maindir+'APOLLO_data/level2/'
datadir_wxt = maindir+'WXT_data/level2/'
outdir_all  = maindir+'APOLLO_data/level3/' 
outdir_wxt  = maindir+'WXT_data/level3/' 

meta_file2020 = maindir+'FESSTVaL/FESSTHH/stations_fessthh.txt'
meta_file2021 = maindir+'FESSTVaL/stations_fesstval.txt'
cp_file2020   = maindir+'FESSTVaL/FESSTHH//cold_pools_fessthh.txt'
cp_file2021   = maindir+'FESSTVaL/cold_pools_fesstval.txt'
    
#----------------------------------------------------------------------------  
# Basic settings

# Time range for begin of cold pool events
start_time    = dt.datetime(2020,8,10,0)   #dt.datetime(2021,5,17,0)   
end_time      = dt.datetime(2020,8,10,23)  #dt.datetime(2021,8,27,23)

# Processing variable
pvar          = 'TT' # 'TT','PP'  ('TT' is always processed)

# Analysis frequency
freq          = 1 # (min)

# Write output files
write_nc      = False

#----------------------------------------------------------------------------  
# Advanced settings

# Analysis duration per cold pool event
dur_event     = 4  # (hours)

# Interpolation grid
res_grid      = 1000 # (m) resolution
dist_invalid  = 5000 # (m) max. valid distance from measurement point 

# Drop selected remote stations to keep network compact (only applies to 2020 data)
drop_stations = True

# Processing of measurement data
t_smooth      = 5  # (s) Half-window length for smooting of APOLLO T and p data
t_trend_pert  = 60 # (min) Period length of unperturbated state to calculate TT trend
t_smooth_u    = 30 # (s) Half-window length for smooting of WXT wind data

# Kriging settings
ktype         = 'ordinary' # Kriging type (ordinary,universal)
svm           = 'power'    # Semi-variogram model

# Usage of Haversine method to calculate grid (much longer computation time if False)
haversine     = True

#---------------------------------------------------------------------------- 
#----------------------------------------------------------------------------
print('Processing FESSTVaL level 2 data')
print('Start Time: '+start_time.strftime('%Y-%m-%d %H:%M'))
print('End Time  : '+end_time.strftime('%Y-%m-%d %H:%M'))
print('Frequency : '+str(freq)+' min')
print('Write nc  : '+str(write_nc))

if start_time.year == 2020:
    days_fesstval = pd.date_range(dt.date(2020,6,1),dt.date(2020,8,31),freq='d').date
    meta_data     = fst.fessthh_stations('l2',metafile=meta_file2020)

    lon_ref_grid,lat_ref_grid = 9.99302,53.55074 # Hamburg Rathausmarkt
    xmin_grid,xmax_grid       = -22000,26000
    ymin_grid,ymax_grid       = -27000,20000

    # Drop remote stations to keep network compact
    if drop_stations:
        stat_drop = ['028KBa','047OGw','055OGw','076PGa','078PGa']
        ii_drop = [meta_data[meta_data['STATION'] == s].index[0] for s in stat_drop]
        meta_data.drop(ii_drop,inplace=True)


if start_time.year == 2021:
    days_fesstval = pd.date_range(dt.date(2021,5,17),dt.date(2021,8,27),freq='d').date
    meta_data     = fst.fesstval_stations('',metafile=meta_file2021)
    
    lon_ref_grid,lat_ref_grid = meta_data[meta_data['STATION'] == '001Sa'][['LON','LAT']].squeeze()
    xmin_grid,xmax_grid       = -20000,20000
    ymin_grid,ymax_grid       = -20000,20000
    

ii_apo    = meta_data['STATION'].str.endswith('a')
ii_wxt    = meta_data['STATION'].str.endswith('w')
n_stats   = meta_data.shape[0]

wxts = meta_data['STATION'][ii_wxt]

cp_fesstval_2020 = fst.fesstval_cold_pools(cpfile=cp_file2020)
cp_fesstval_2021 = fst.fesstval_cold_pools(cpfile=cp_file2021)
cp_fesstval = cp_fesstval_2020.append(cp_fesstval_2021)
cp_fesstval['INDEX'] = np.arange(cp_fesstval.shape[0])

ndays_fval    = len(days_fesstval)
days_cp       = np.unique(cp_fesstval.index.date)
ncp_fesstval  = cp_fesstval.index.shape[0]
days_read     = pd.DataFrame([False]*ndays_fval,index=days_fesstval)
days_proc     = pd.DataFrame([False]*ndays_fval,index=days_fesstval)
dtime_overlap = []

dir_kwargs = {'datadir_apollo':datadir_apo,'datadir_wxt':datadir_wxt}
        

# Setting up regular cartesian grid
cp_spatial = cps.network_analysis(meta_data['LON'],meta_data['LAT'],
                                  lon_data_ref=lon_ref_grid,
                                  lat_data_ref=lat_ref_grid,
                                  x_min=xmin_grid,x_max=xmax_grid,
                                  y_min=ymin_grid,y_max=ymax_grid,
                                  x_res=res_grid,y_res=res_grid,
                                  hav=haversine)

x_data,y_data             = cp_spatial.xy_data()
x_grid,y_grid             = cp_spatial.xy_gridpoints()
x_meshgrid,y_meshgrid     = cp_spatial.xy_meshgrid()
lon_meshgrid,lat_meshgrid = cp_spatial.lonlat_meshgrid()
lon_ref                   = cp_spatial.lon_ref
lat_ref                   = cp_spatial.lat_ref
mask_meshgrid             = cp_spatial.mask_meshgrid(dist_invalid)
nx,ny                     = len(x_grid),len(y_grid)

krig_kwargs = {'krig_type':ktype,'sv_model':svm,'weight':True}
#---------------------------------------------------------------------------- 
# Definition of Functions   

# Calculate mean linear trend of unperturbed temperature over all stations,
# = reference temperature for cold pool temperature perturbations
def tt_trend(ttdata,period=t_trend_pert,tres=10):
    nperiod   = int(period*60/tres)
    xdata     = np.arange(ttdata.shape[0])
    ttdata_lr = ttdata.median(axis=1).iloc[:nperiod]
    xdata_lr  = np.arange(ttdata_lr.shape[0])
    ii_fin    = pd.notnull(ttdata_lr)
    if ii_fin.sum() > 0:
        lin_reg  = scs.linregress(xdata_lr[ii_fin],ttdata_lr[ii_fin])
        return (lin_reg.intercept + lin_reg.slope*xdata)
    else:    
        return xdata * np.nan        
     
# Detrend pressure time series of individual stations to calculate cold pool 
# pressure perturbation
def detrend(ydata):
    ii_fin = pd.notnull(ydata)
    if ii_fin.sum() > 0:
        xdata   = np.arange(ydata.shape[0])
        lin_reg = scs.linregress(xdata[ii_fin],ydata[ii_fin])
        return ydata - (lin_reg.intercept + lin_reg.slope*xdata)
    else:
        return ydata * np.nan  
    
#----------------------------------------------------------------------------
# Definition of level 3 data format

meas_type       = 'fval'     # Campaign
version_dataset = 1          # In case of new dataset version number difference 
                             # needs to explained in History section
version_proc    = '1.0'      # Version number of this python script

if start_time.year == 2020: meas_type = 'fessthh'
out_dirs  = {'APOLLOWXT':outdir_all,'WXT':outdir_wxt} 
grids     = {'lon':lon_meshgrid,'lat':lat_meshgrid,
             'x':x_meshgrid,'y':y_meshgrid,
             'lon_ref':lon_ref,'lat_ref':lat_ref,
             'mask':mask_meshgrid.astype(int)} 
                  

def write_netcdf_file(varstr,time_write,data_write,id_write,ref_write=[],
                      grid_write=grids,kkk=meas_type,vers_ds=version_dataset,
                      vers_proc=version_proc,t_res=freq,odirs=out_dirs,c_level=1):
    
    if np.isfinite(data_write).sum() == 0:
        print('No '+varstr+' data found to be written!')
        return
    
    if (varstr == 'TT') and (len(ref_write) == 0):
        print('No reference time series data for TT provided!')
        return
    
    type_instr = 'APOLLOWXT' if varstr in ['TT','PP'] else 'WXT'
    
    vnn = {'TT':'ta',
           'PP':'pa',
           }
    
    varnames = {'TT':'Interpolated air temperature perturbation',
                'PP':'Interpolated air pressure perturbation',
                }
    
    sources = {'APOLLOWXT':'APOLLO and WXT weather stations',
               'WXT':'WXT weather stations'}
 
    lon_data  = grid_write['lon'][0,:]   
    lat_data  = grid_write['lat'][:,0]  
    x_data    = grid_write['x'][0,:]   
    y_data    = grid_write['y'][:,0]
    mask_data = grid_write['mask']
    
    ref_str  = '{:8.5f}E, '.format(grid_write['lon_ref'])+\
               '{:8.5f}N'.format(grid_write['lat_ref'])
               
    ny,nx,nt = data_write.shape
    
    if len(time_write) != nt:
        print('Inconsistent length of time dimensions!')
        return
    
    if (grids['lon'].shape != (ny,nx)) or (grids['lat'].shape != (ny,nx)):
        print('Inconsistent grid dimensions!')
        return           
                 
    # File naming and creation
    writedir = odirs[type_instr]+time_write[0].strftime('%Y/%m/')
    dtstr    = time_write[0].strftime('%Y%m%d')
    if os.path.isdir(writedir) == False: os.makedirs(writedir)
    filename = writedir+kkk+'_uhh_'+type_instr.lower()+'_l3_'+vnn[varstr]+\
               '_v'+str(vers_ds).zfill(2)+'_'+dtstr+'.nc'
    if os.path.isfile(filename): os.remove(filename)
    
    comment_experiment = 'FESSTVaL field experiment (May - August 2021)'
    if dtstr[:4] == '2020':
        comment_experiment = 'FESST@HH field experiment (June - August 2020)'

    ncfile = Dataset(filename,'w',format='NETCDF4')
    
    # Dimensions
    ncfile.createDimension('time',None)
    ncfile.createDimension('nv',2)
    ncfile.createDimension('lon',nx)
    ncfile.createDimension('lat',ny)
    
    #Dimension Variables
    utime               = fst.datetime_to_unixtime(time_write)
    time                = ncfile.createVariable('time', 'i4', ('time',))
    time[:]             = utime
    time.standard_name  = 'time'
    time.units          = 'seconds since 1970-01-01 00:00:00 UTC' 
    time.bounds         = 'time_bnds'
    time.calendar       = 'standard'
    
    nv                  = ncfile.createVariable('time_bnds', 'i4', ('time','nv',))
    nv[:,:]             = np.column_stack((utime-t_res*60,utime))
    nv.long_name        = 'start and end of time averaging intervals'
    
    lon                 = ncfile.createVariable('lon', 'f4', ('lon',)) 
    lon[:]              = lon_data
    lon.standard_name   = 'longitude'
    lon.units           = 'degrees_east'

    lat                 = ncfile.createVariable('lat', 'f4', ('lat',)) 
    lat[:]              = lat_data
    lat.standard_name   = 'latitude'
    lat.units           = 'degrees_north'
    
    x                   = ncfile.createVariable('x', 'f4', ('lon',)) 
    x[:]                = x_data
    x.long_name         = 'x coordinates'
    x.units             = 'm'
    x.comment           = 'origin at '+ref_str

    y                   = ncfile.createVariable('y', 'f4', ('lat',)) 
    y[:]                = y_data
    y.long_name         = 'y coordinates'
    y.units             = 'm'
    y.comment           = 'origin at '+ref_str
    
    mask                = ncfile.createVariable('mask', 'i2', ('lon','lat',)) 
    mask[:,:]           = np.transpose(mask_data,(1,0))
    mask.long_name      = 'interpolation mask'
    mask.comment        = 'binary mask of valid interpolation grid points'
    
    event_id            = ncfile.createVariable('event_id', 'i2', ('time',)) 
    event_id[:]         = id_write
    event_id.long_name  = 'event identifier'
    event_id.comment    = 'running number (0-78) of cold pool events during '+\
                          'FESST@HH 2020 and FESSTVaL 2021'
    

    # Variables
    if varstr == 'TT':
        ta_perturbation           = ncfile.createVariable('ta_perturbation','f4',
                                                         ('time','lon','lat',),
                                                         fill_value='nan',
                                                         zlib=True,
                                                         complevel=c_level,
                                                         least_significant_digit=3)
        ta_perturbation[:,:,:]    = np.transpose(data_write,(2,1,0))
        ta_perturbation.long_name = 'air temperature perturbation'
        ta_perturbation.units     = 'K'
        ta_perturbation.comment   = 'cold pool perturbation relative to '+\
                                    'ta_reference'
                                    
        ta_reference              = ncfile.createVariable('ta_reference','f4',
                                                          ('time',),
                                                          fill_value='nan',
                                                          zlib=True,
                                                          complevel=c_level,
                                                          least_significant_digit=3)
        ta_reference[:]           = ref_write + 273.15
        ta_reference.long_name    = 'reference air temperature'
        ta_reference.units        = 'K'
        ta_reference.comment      = 'estimated unperturbed air temperature '+\
                                    'for current cold pool event'
    
    if varstr == 'PP':
        pa_perturbation           = ncfile.createVariable('pa_perturbation','f4',
                                                          ('time','lon','lat',),
                                                          fill_value='nan',
                                                          zlib=True,
                                                          complevel=c_level,
                                                          least_significant_digit=1)
        pa_perturbation[:,:,:]    = np.transpose(data_write,(2,1,0))*100
        pa_perturbation.long_name = 'air pressure perturbation'
        pa_perturbation.units     = 'Pa'
        pa_perturbation.comment   = 'cold pool perturbation relative to'+\
                                    'local air pressure trend'
        

    # Global attributes
    ncfile.Title             = varnames[varstr]
    ncfile.Institution       = 'Meteorological Institute, University of Hamburg (UHH), Germany'
    ncfile.Contact_person    = 'Prof. Dr. Felix Ament (felix.ament@uni-hamburg.de)'
    ncfile.Source            = sources[type_instr]
    ncfile.History           = 'Data processed with fesstval_interpolation.py,'+\
                               ' version '+vers_proc+'; 2023-06-13 v01: '+\
                               'upper limit for temperature perturbation omitted'    
    ncfile.Conventions       = 'CF-1.7 where applicable'
    ncfile.Processing_date   = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')    
    ncfile.Author            = 'Bastian Kirsch (bastian.kirsch@uni-hamburg.de)'     
    ncfile.Comments          = comment_experiment
    ncfile.Licence           = 'This data is licensed under a '+\
                               'Creative Commons Attribution 4.0 '+\
                               'International License (CC BY 4.0).'  
    ncfile.close()
    return
    
#---------------------------------------------------------------------------- 
#---------------------------------------------------------------------------- 

# Loop over cold pool events
for icp in cp_fesstval['INDEX']:  
    
    if pvar not in ['TT','PP']:
        print('Variable '+pvar+' is not supported!')
        break
        
    start_cp = cp_fesstval.index[icp]
    if (start_cp < start_time) or (start_cp > end_time): continue
    date_cp  = start_cp.date()
    end_cp   = start_cp + dt.timedelta(hours=dur_event)
    time_cp  = pd.date_range(start_cp,end_cp,freq=str(freq)+'min')
    ntime_cp = len(time_cp)
    name_cp = cp_fesstval['NAME'].iloc[icp]
    ext = ' ('+name_cp+')' if type(name_cp) == str else ''
    
    days_read_event = np.unique(time_cp.date)
    ndays = days_read_event.shape[0]
    
    if ndays > 2:
        # This message should never appear (otherwise event duration is much too long)
        print('*** Cold pool event covers more than two days ***')
        break
    
    ntime_oneday = int(1440/freq)
    ntime_day    = ntime_oneday * ndays
    time_day     = pd.date_range(date_cp,periods=ntime_day,freq=str(freq)+'min')
    
    # Determine if day file should be written or another event for this day
    # has to be processed before writing
    last_cp_of_day = True
    if icp < ncp_fesstval-1:
        date_next_cp = cp_fesstval.index[icp+1].date()
        if date_next_cp == date_cp: last_cp_of_day = False
    
    # Loop over days of the cold pool event
    for di,d in enumerate(days_read_event):
        if not days_read[0].loc[d]:
            print('')
            print(d.strftime('Reading level 2 data for %Y-%m-%d'))
            date_args = (d.year,d.month,d.day)
            
            # Reading data
            TT_apollo = fst.read_fesstval_level2('a','TT',*date_args,**dir_kwargs)
            TT_wxt    = fst.read_fesstval_level2('w','TT',*date_args,**dir_kwargs)
            
            # Smooting of APOLLO data  
            TT_apollo = TT_apollo.rolling(2*t_smooth+1,center=True,\
                                          min_periods=t_smooth+1,axis=0).mean() 
            
            if pvar == 'PP':
                PP_apollo = fst.read_fesstval_level2('a','PP',*date_args,**dir_kwargs)
                PP_wxt    = fst.read_fesstval_level2('w','PP',*date_args,**dir_kwargs)
                
                # Note: Possibly smoothing out short-term pressure spikes!    
                PP_apollo = PP_apollo.rolling(2*t_smooth+1,center=True,\
                                              min_periods=t_smooth+1,axis=0).mean() 
            
            days_read[0].loc[date_cp] = True
            
            # Merging of days if cold pool event covers two days
            if di == 0:
                TT_all = TT_apollo.iloc[::10].join(TT_wxt)\
                              .reindex(meta_data['STATION'],axis=1)
                if pvar == 'PP':
                    PP_all = PP_apollo.iloc[::10].join(PP_wxt)\
                              .reindex(meta_data['STATION'],axis=1)
                
            else:
                TT_all = TT_all.append(TT_apollo.iloc[::10].join(TT_wxt)\
                                     .reindex(meta_data['STATION'],axis=1))
                if pvar == 'PP':
                    PP_all = PP_all.append(PP_apollo.iloc[::10].join(PP_wxt)\
                                     .reindex(meta_data['STATION'],axis=1))
    
    # Dropping time steps outside of cold pool event
    TT_all.drop(TT_all.index[TT_all.index < start_cp],inplace=True)
    if pvar == 'PP':                 
        PP_all.drop(PP_all.index[PP_all.index < start_cp],inplace=True)
        
    # Defining perturbations
    dT_all = pd.DataFrame(index=TT_all.loc[time_cp[0]:time_cp[-1]].index)
    TT_trend = pd.Series(tt_trend(TT_all.loc[time_cp[0]:time_cp[-1]]),
                         index=dT_all.index)
    for stat in TT_all.columns: 
        dT_all[stat] = TT_all[stat].loc[time_cp[0]:time_cp[-1]] - TT_trend
    
    if pvar == 'PP':
        dP_all = pd.DataFrame(index=PP_all.loc[time_cp[0]:time_cp[-1]].index)
        for stat in PP_all.columns: 
            dP_all[stat] = detrend(PP_all[stat].loc[time_cp[0]:time_cp[-1]])
        
    
    # Interpolating perturbations
    print(start_cp.strftime('Analyzing cold pool event on %Y-%m-%d %H UTC')+ext)  
    
    # Setting up data structure for storing interpolated data        
    if not days_proc[0].loc[date_cp]:
        # New structure only for new day; otherwise use structure of previous event
        print('Setting up new analysis data structure')
        dT_interp  = np.ones([ny,nx,ntime_day]) * np.nan
        dP_interp  = np.ones([ny,nx,ntime_day]) * np.nan
        dT_ref     = np.ones([ntime_day]) * np.nan
        cp_id_time = np.ones([ntime_day],dtype=np.int16) * -999
    elif ndays == 2:
        # Append structure for second day only if second event of day is a 2-day event
        print('Appending new analysis data structure')
        dT_interp  = np.append(dT_interp,np.ones([ny,nx,ntime_oneday])*np.nan,axis=2)
        dP_interp  = np.append(dP_interp,np.ones([ny,nx,ntime_oneday])*np.nan,axis=2)
        dT_ref     = np.append(dT_ref,np.ones([ntime_oneday])*np.nan)
        cp_id_time = np.append(cp_id_time,np.ones([ntime_day],dtype=np.int16)*-999)
    else:
        pass
        
        
    # Loop over analyis time steps
    for t in time_cp:
        print(t.strftime('%H:%M'))
        i_day = time_day.get_loc(t.to_pydatetime())
        
        # ID of cold pool event
        cp_id_time[i_day] = icp 
        
        # Reference temperature
        dT_ref[i_day] = TT_trend.loc[t]
        
        if np.sum(np.isfinite(dT_interp[:,:,i_day])) > 0:
            print('*** WARNING: Time step was already analysed ***')
            dtime_overlap.append(t) 
        
        # Kriging interpolation of dT
        dT_ip = cp_spatial.interpolation(dT_all.loc[t],**krig_kwargs)
        dT_ip[~mask_meshgrid] = np.nan
        dT_interp[:,:,i_day] = dT_ip  
        
        if pvar == 'PP':
            # Kriging interpolation of dP
            dP_ip = cp_spatial.interpolation(dP_all.loc[t],**krig_kwargs)
            dP_ip[np.isnan(dT_ip)] = np.nan
            dP_interp[:,:,i_day] = dP_ip

    # Write level 3 data 
    if write_nc and last_cp_of_day:
        print('Writing level 3 data')
        write_netcdf_file('TT',time_day[:ntime_oneday],
                          dT_interp[:,:,:ntime_oneday],
                          cp_id_time[:ntime_oneday],
                          ref_write=dT_ref[:ntime_oneday])
        
        if pvar == 'PP':
            write_netcdf_file(pvar,time_day[:ntime_oneday],
                              dP_interp[:,:,:ntime_oneday],
                              cp_id_time[:ntime_oneday])
        
        # Storing data of second day for two-day event
        if ndays == 2:
            print('Storing data of second day of event')
            dT_interp  = dT_interp[:,:,ntime_oneday:]
            dP_interp  = dP_interp[:,:,ntime_oneday:]
            cp_id_time = cp_id_time[ntime_oneday:]
            dT_ref     = dT_ref[ntime_oneday:]
            days_proc[0].loc[date_cp+dt.timedelta(days=1)] = True
        
    days_proc[0].loc[date_cp] = True    
    
'''
Overlapping time steps for dT_lim = 0:
- 2021-05-23 12:14:00 - 13:00:00
- 2021-05-25 15:00:00
- 2021-05-27 16:30:00 - 18:00:00
- 2021-06-12 13:08:00 - 14:00:00
- 2021-07-09 16:00:00 - 16:30:00
- 2021-08-08 20:30:00
- 2021-08-26 11:30:00 - 15:30:00
'''    
    
#----------------------------------------------------------------------------
print(' ')
print('*** Finshed! ***')
fst.print_runtime(t_run)
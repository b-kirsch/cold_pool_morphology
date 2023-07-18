# -*- coding: utf-8 -*-
"""
@author: Bastian Kirsch (bastian.kirsch@uni-hamburg.de)

Script to process level 2 X-band radar data and write level 3 on regular grid

- Downloads X-band radar data (level 2) from ftp server
- Reads level 2 data
- Performs nearest neighbor interpolation
- Writes interpolated data to netCDF files (level 3)

Dependences on non-standard software:
- fesstval_routines.py

Required meta data files:
- cold_pools_fessthh.txt
- cold_pools_fesstval.txt    

Last updated: 31 October 2022
"""

import numpy as np
import pandas as pd
import os
import datetime as dt
from netCDF4 import Dataset
import ftplib
import fesstval_routines as fst

t_run = dt.datetime.now()

#----------------------------------------------------------------------------
# Paths and meta data files
maindir     = '.'

datadir     = maindir+'Radar_data/level2/'
outdir      = maindir+'Radar_data/level3/' 

cp_file2020 = maindir+'FESSTVaL/FESSTHH/cold_pools_fessthh.txt'
cp_file2021 = maindir+'FESSTVaL/cold_pools_fesstval.txt'

#----------------------------------------------------------------------------  
# Basic settings 
start_time = dt.datetime(2020,8,10,12)  #dt.datetime(2021,6,3,0)   
end_time   = dt.datetime(2020,8,10,15)  #dt.datetime(2021,8,27,23)

# Analysis frequency
freq       = 1 # (min)

# Download level 2 data
download   = False

# Process level 2 data
process    = False

# Write output files
write_nc   = False

#----------------------------------------------------------------------------  
# Advanced settings 

# Spatial resolution of interpolation grid
res_grid_rad = 200  # (m)

# Duration of cold pool event
dur_event    = 4 # (h)

# Usage of Haversine method to calculate grid (much longer computation time if False)
haversine = True # !!! Was False for Diss results (rr_reggrid) --> Impacts results

#---------------------------------------------------------------------------- 
# Setting up meta interpolation grid
if start_time.year == 2020:
    cp_fesstval = fst.fesstval_cold_pools(cpfile=cp_file2020)
    lon_ref_grid,lat_ref_grid = 9.99302,53.55074 # Hamburg Rathausmarkt
    xmin_grid,xmax_grid       = -22000,20000
    ymin_grid,ymax_grid       = -19000,23000
    
    date_grid                 = dt.datetime(2020,8,10,12)

if start_time.year == 2021:
    cp_fesstval   = fst.fesstval_cold_pools(cpfile=cp_file2021) 
    lon_ref_grid,lat_ref_grid = 14.12292,52.16665 # 001Sa (Falkenberg Mast)
    xmin_grid,xmax_grid       = -20000,20000
    ymin_grid,ymax_grid       = -20000,20000
    date_grid                 = dt.datetime(2021,6,29,14)
    
radar = fst.read_xband_radar(date_grid.year,date_grid.month,
                             date_grid.day,date_grid.hour,
                             x_min=xmin_grid,y_min=ymin_grid,
                             x_max=xmax_grid,y_max=ymax_grid,
                             x_res=res_grid_rad,y_res=res_grid_rad,
                             lon_data_ref=lon_ref_grid,lat_data_ref=lat_ref_grid,
                             mute=False,datadir=datadir,hav=haversine)
x_meshgrid_rad, y_meshgrid_rad     = radar.xy_reggrid()
lon_meshgrid_rad, lat_meshgrid_rad = radar.lonlat_reggrid()
lon_ref = radar.lon_ref
lat_ref = radar.lat_ref
ny,nx   = x_meshgrid_rad.shape

grids = {'lon':lon_meshgrid_rad,'lat':lat_meshgrid_rad,
         'x':x_meshgrid_rad,'y':y_meshgrid_rad,
         'lon_ref':lon_ref,'lat_ref':lat_ref}     
    
#----------------------------------------------------------------------------
# Definition of level 3 data format

meas_type       = 'fval'     # Campaign
version_dataset = 0          # In case of new dataset version number difference 
                             # needs to explained in History section
version_proc    = '1.0'      # Version number of this python script

if start_time.year == 2020: meas_type = 'fessthh'


def write_netcdf_file(time_write,data_write,grid_write=grids,kkk=meas_type,
                      vers_ds=version_dataset,vers_proc=version_proc,
                      t_res=freq,odir=outdir,c_level=1):
    
    if np.isfinite(data_write).sum() == 0:
        print('No data found to be written!')
        return
    
    lon_data  = grid_write['lon'][0,:]   
    lat_data  = grid_write['lat'][:,0]  
    x_data    = grid_write['x'][0,:]   
    y_data    = grid_write['y'][:,0]
    
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
    writedir = odir+time_write[0].strftime('%Y/%m/')
    dtstr    = time_write[0].strftime('%Y%m%d')
    if os.path.isdir(writedir) == False: os.makedirs(writedir)
    filename = writedir+kkk+'_uhh_wrx_l3_rr'+\
               '_v'+str(vers_ds).zfill(2)+'_'+dtstr+'.nc'
    if os.path.isfile(filename): os.remove(filename)
    
    comment_experiment = 'FESSTVaL field experiment (May - August 2021)'
    source = 'WRX / LAWR FLK (Falkenberg)'
    if dtstr[:4] == '2020':
        comment_experiment = 'FESST@HH field experiment (June - August 2020)'
        source = 'WRX / LAWR GEO (Geomatikum)'

    ncfile = Dataset(filename,'w',format='NETCDF4')
    
    # Dimensions
    ncfile.createDimension('time',None)
    ncfile.createDimension('nv',2)
    ncfile.createDimension('lon',nx)
    ncfile.createDimension('lat',ny)
    
    #Dimension Variables
    utime              = fst.datetime_to_unixtime(time_write)
    time               = ncfile.createVariable('time', 'i4', ('time',))
    time[:]            = utime
    time.standard_name = 'time'
    time.units         = 'seconds since 1970-01-01 00:00:00 UTC' 
    time.bounds        = 'time_bnds'
    time.calendar      = 'standard'
    
    nv                 = ncfile.createVariable('time_bnds', 'i4', ('time','nv',))
    nv[:,:]            = np.column_stack((utime-t_res*60,utime))
    nv.long_name       = 'start and end of time averaging intervals'
    
    lon                = ncfile.createVariable('lon', 'f4', ('lon',)) 
    lon[:]             = lon_data
    lon.standard_name  = 'longitude'
    lon.units          = 'degrees_east'

    lat                = ncfile.createVariable('lat', 'f4', ('lat',)) 
    lat[:]             = lat_data
    lat.standard_name  = 'latitude'
    lat.units          = 'degrees_north'
    
    x                  = ncfile.createVariable('x', 'f4', ('lon',)) 
    x[:]               = x_data
    x.long_name        = 'x coordinates'
    x.units            = 'm'
    x.comment          = 'origin at '+ref_str

    y                  = ncfile.createVariable('y', 'f4', ('lat',)) 
    y[:]               = y_data
    y.long_name        = 'y coordinates'
    y.units            = 'm'
    y.comment          = 'origin at '+ref_str
    
    rr                 = ncfile.createVariable('rr','f4',('time','lon','lat',),
                                               fill_value='nan',zlib=True,
                                               complevel=1)
    rr[:]              = np.transpose(data_write,(2,1,0))
    rr.standard_name   = 'rainfall_rate'
    rr.units           = 'mm/h'
        

    # Global attributes
    ncfile.Title             = 'Interpolated X-band weather radar (WRX) data'
    ncfile.Institution       = 'Meteorological Institute, University of Hamburg (UHH), Germany'
    ncfile.Contact_person    = 'Prof. Dr. Felix Ament (felix.ament@uni-hamburg.de)'
    ncfile.Source            = source
    ncfile.History           = 'Data processed with radar_level2_to_level3.py,'+\
                               ' version '+vers_proc
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
# Download level 2 radar data for cold pool events within set time range
hours     = pd.date_range(start_time,end_time,freq='h')
days      = pd.date_range(start_time.replace(hour=0),end_time,freq='d')
ntime_day = int(1440/freq)

if download:
    print('Log in to ftp server')    
    ftp = ftplib.FTP('ftp-projects.cen.uni-hamburg.de')
    ftp.login('***','***')
        
    for hh in hours:
        print('Download level 2 radar data from ftp server')
        ftp.cwd(hh.strftime('/data/fesstval/wrx/level2/%m/'))
        readfile = hh.strftime('fval_uhh_wrxFLK_l2_rr_v00_%Y%m%d%H.nc')
        fst.ftp_download(ftp,readfile,datadir+hh.strftime('%Y/%m/')+readfile)

#----------------------------------------------------------------------------
# Process level 2 data and write level 3 data
for dd in days:  
    if not process: continue
    # Setting up data structure for day to be written
    time_day  = pd.date_range(dd,periods=ntime_day,freq=str(freq)+'min')
    hours_day = pd.date_range(dd,periods=24,freq='h')
    RR_radar  = np.ones([ny,nx,ntime_day]) * np.nan
    
    # Reading and interpolating data
    for hh in hours_day:
        if hh not in hours: continue
        print(hh.strftime('Reading radar data for %Y-%m-%d %H UTC'))
        radar = fst.read_xband_radar(hh.year,hh.month,hh.day,hh.hour,
                                     x_min=xmin_grid,y_min=ymin_grid,
                                     x_max=xmax_grid,y_max=ymax_grid,
                                     x_res=res_grid_rad,y_res=res_grid_rad,
                                     lon_data_ref=lon_ref,lat_data_ref=lat_ref,
                                     mute=False,datadir=datadir,hav=haversine)
        if not radar.valid_grid: continue
        
        i_hour    = time_day.get_loc(hh.to_pydatetime())
        time_hour = time_day[i_hour:i_hour+int(60/freq)]
        
        print('Interpolating radar data')
        for t in time_hour:
            print(t.strftime('%H:%M'))
            i_day = time_day.get_loc(t.to_pydatetime())
            RR_radar[:,:,i_day] = radar.rr_reggrid(t.minute,t.second)   
            
    # Write netcdf
    if write_nc:
        print('Writing level 3 data')
        write_netcdf_file(time_day,RR_radar)
        
#----------------------------------------------------------------------------
print(' ')
print('*** Finshed! ***')
fst.print_runtime(t_run)
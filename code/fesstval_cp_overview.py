# -*- coding: utf-8 -*-
"""
@author: Bastian Kirsch (bastian.kirsch@uni-hamburg.de)

Code to detect cold pool events in APOLLO and WXT level 2 data and 
plot dT distrubtion for events during FESSTVaL 2021

Dependences on non-standard software:
- fesstval_routines.py
- cp_detection_timeseries.py
- fesstval_plots.py

Last updated: 20 June 2023
"""

print('*********')

import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import fesstval_routines as fst
import cp_detection_timeseries as cpdt

t_run = dt.datetime.now()
#----------------------------------------------------------------------------
# Paths and meta data files
maindir       = '.'
datadir_apo   = maindir+'APOLLO_data/level2/'
datadir_wxt   = maindir+'WXT_data/level2/'
meta_file2020 = maindir+'FESSTVaL/FESSTHH/stations_fessthh.txt'
meta_file2021 = maindir+'FESSTVaL/stations_fesstval.txt'
cp_file2020   = maindir+'FESSTVaL/FESSTHH//cold_pools_fessthh.txt'
cp_file2021   = maindir+'FESSTVaL/cold_pools_fesstval.txt'
plotdir       = maindir+'Cold-Pools/Plots/Paper_CP_Morphology/' 
write_file    = maindir+'Cold-Pools/Paper_CP_Morphology/data_fig13a.txt'
    
start_date = dt.date(2020,8,10)  #dt.date(2021,5,17)
end_date   = dt.date(2021,8,27)  #dt.date(2021,8,27)

dur_event       = 4 # Hours (Duration of cold pool event)
data_avail_cp   = 0.95  # Minimum data availability for event for detection
lim_avail_event = 0.75  # Limit for low data availability marked in plot

t_smooth        = 30 # 5 (s) Smooting of APOLLO data

plot_events     = False

write_plot_data = False   # for Fig. 13a (used in fesstval_plots.py)

# File naming
fig_numbers = False # Name files "Kirsch_Figxx.pdf" if True

if plot_events: t_smooth = 5

#----------------------------------------------------------------------------
print('Detecting cold pools in FESSTVaL level 2 data')
print('Start Date: '+start_date.strftime('%Y-%m-%d'))
print('End Date  : '+end_date.strftime('%Y-%m-%d'))
print('')

days   = pd.date_range(start_date,end_date,freq='d').date
ndays  = len(days)
hours  = pd.date_range(start_date,dt.datetime(end_date.year,end_date.month,
                                              end_date.day,23,0),freq='h')
nhours = len(hours)


#Read meta data
meta_data_2020 = fst.fessthh_stations('l2',metafile=meta_file2020)
meta_data_2021 = fst.fesstval_stations('',metafile=meta_file2021)
meta_data_dict = {2020:meta_data_2020,2021:meta_data_2021}   

cp_fesstval_2020 = fst.fesstval_cold_pools(cpfile=cp_file2020)
cp_fesstval_2021 = fst.fesstval_cold_pools(cpfile=cp_file2021)
cp_fesstval = cp_fesstval_2020.append(cp_fesstval_2021)
cp_fesstval['INDEX'] = np.arange(cp_fesstval.shape[0])

days_2020 = pd.date_range(dt.date(2020,6,1),dt.date(2020,8,31),freq='d').date
days_2021 = pd.date_range(dt.date(2021,5,17),dt.date(2021,8,27),freq='d').date
days_all  = np.append(days_2020,days_2021)
days_read = pd.Series([False]*len(days_all),index=days_all)

stations_all = meta_data_2020['STATION'].append(meta_data_2021['STATION'])
data_avail_stats = pd.DataFrame(index=cp_fesstval['INDEX'],columns=stations_all)
cp_stats = pd.DataFrame(columns=['DATE_TIME','STATION','ICP','DT','RR_ACCU','ASD']) 
i_all = 0 

if 'FESSTVaL' in plotdir: plotdir = plotdir.replace('FESSTVaL','FESSTHH')


# Loop over cold pool events
for icp in cp_fesstval['INDEX']:  
    
    # Reading data
    start_cp = cp_fesstval.index[icp]
    if (start_cp < start_date) or (start_cp > end_date): continue
    end_cp   = start_cp + dt.timedelta(hours=dur_event)
    days_read_event = np.unique([start_cp.date(),end_cp.date()])
    
    meta_data = meta_data_dict[start_cp.year]
    
    for di,d in enumerate(days_read_event):
        if not days_read.loc[d]:
            print(d.strftime('Reading level 2 data for %Y-%m-%d'))

            TT_apollo = fst.read_fesstval_level2('a','TT',d.year,d.month,d.day,
                                                 datadir_apollo=datadir_apo,
                                                 datadir_wxt=datadir_wxt)
            TT_wxt    = fst.read_fesstval_level2('w','TT',d.year,d.month,d.day,
                                                 datadir_apollo=datadir_apo,
                                                 datadir_wxt=datadir_wxt)
            FF_wxt    = fst.read_fesstval_level2('w','FF',d.year,d.month,d.day,
                                                 datadir_apollo=datadir_apo,
                                                 datadir_wxt=datadir_wxt) # FB?
            
            days_read.loc[d] = True
            if TT_apollo.empty or TT_wxt.empty: continue    
        
            TT_apollo = TT_apollo.rolling(2*t_smooth+1,center=True,\
                                          min_periods=t_smooth+1,axis=0).mean()
            FF_wxt     = FF_wxt.rolling(int(2*t_smooth/10+1),center=True,
                                        min_periods=int(t_smooth/10),axis=0).mean()   
            
            if di == 0:
                TT_all = TT_apollo.iloc[::10].join(TT_wxt)\
                         .reindex(meta_data['STATION'],axis=1)
                FF_all = FF_wxt.copy()
            else:
                TT_all = TT_all.append(TT_apollo.iloc[::10].join(TT_wxt)\
                                       .reindex(meta_data['STATION'],axis=1))
                FF_all = FF_all.append(FF_wxt)

    
    #Detecting cold pools from time series
    for s in meta_data['STATION']:
        TT_event_stat = TT_all.loc[start_cp:end_cp,s]
        
        data_avail_stats.loc[icp,s] = TT_event_stat.notnull().sum()/TT_event_stat.shape[0]
        
        #RR = RR_all[wxt_near[s]] 
        RR = np.ones(TT_event_stat.index.shape[0])
        cp = cpdt.cp_detection(TT_event_stat.index,TT_event_stat,RR,
                               data_avail_all=data_avail_cp,
                               warn_avail_all=False,warn_avail_cp=False)
        cptime  = cp.datetimes()
        cptt    = cp.tt_pert()
        cprr    = cp.var_val(RR,'all','sum')
        cpttpre = cp.tt_pre() 
        
        if 'w' in s: cpff = cp.ff_pert(FF_all.loc[start_cp:end_cp,s])
    
        for i in range(cp.number()): 
            if (cptime[i] < start_cp) or (cptime[i] >= end_cp): continue
        
            cp_stats.loc[i_all,'DATE_TIME'] = cptime[i]
            cp_stats.loc[i_all,'STATION']   = s
            cp_stats.loc[i_all,'ICP']       = icp
            cp_stats.loc[i_all,'DT']        = cptt[i]
            cp_stats.loc[i_all,'RR_ACCU']   = cprr[i]
            cp_stats.loc[i_all,'TT_PRE']    = cpttpre[i]
            if 'w' in s: 
                cp_stats.loc[i_all,'FF_MAX'] = cpff[i]

            i_all += 1

cp_stats.drop_duplicates(subset=['DATE_TIME','STATION'],keep='last',inplace=True)

# Writing plot data for Fig. 13a (comparison with Kruse et al., 2022, Fig. 7) to file
if write_plot_data:  
    icp_list = [27,54,55,62] # Elphi,Felix,Jogi,Jürg
    T0       = 273.15
    
    cp_stats['DT_MIN_REL'] = -cp_stats['DT']/(cp_stats['TT_PRE']+T0)
    col_write  = ['ICP','DATE_TIME','STATION','DT_MIN_REL','FF_MAX']
    icp_write  = cp_stats['ICP'].apply(lambda x: x in icp_list)
    stat_write = cp_stats['STATION'].apply(lambda x: 'w' in x)
    idx_write  = icp_write & stat_write

    print('Writing data to file')
    if os.path.isfile(write_file): os.remove(write_file)
    
    cp_stats.loc[idx_write,col_write].to_csv(write_file,index=False,
                                             header=True,sep=';',na_rep='',
                                             date_format='%Y-%m-%d %H:%M:%S',
                                             float_format='%1.5f')
    
#----------------------------------------------------------------------------    
fs  = 12 #fontsize of plot
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
mpl.rcParams['axes.spines.left']  = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top']   = False


for c in ['DT','RR_ACCU','ASD']:
    if c not in cp_stats.columns: continue
    cp_stats[c] = pd.to_numeric(cp_stats[c],downcast='float')

cp_stats_median = cp_stats.groupby('ICP').median()
cp_stats_min    = cp_stats.groupby('ICP').quantile(0.05)
cp_stats_max    = cp_stats.groupby('ICP').quantile(0.95)

ii_stats_wxt        = cp_stats['STATION'].str.endswith('w')
cp_stats_median_wxt = cp_stats[ii_stats_wxt].groupby('ICP').median()
cp_stats_min_wxt    = cp_stats[ii_stats_wxt].groupby('ICP').quantile(0.05)
cp_stats_max_wxt    = cp_stats[ii_stats_wxt].groupby('ICP').quantile(0.95)


if plot_events & (start_date.year == end_date.year):
    
    nstats = meta_data_dict[start_date.year].shape[0]
    ii_icp_year  = (cp_fesstval.index.year == start_date.year)    
    ncp_fesstval = (ii_icp_year).sum()
    
    data_avail_stats.drop(cp_fesstval['INDEX'].iloc[~ii_icp_year])
    if start_date.year == 2020:
        data_avail_stats.drop(meta_data_2021['STATION'],1,inplace=True)
    if start_date.year == 2021:
        data_avail_stats.drop(meta_data_2020['STATION'],1,inplace=True)
    
    ii_sort = cp_stats_median['DT'].sort_values(ascending=True).index
    
    # Number and percentage of available station (i.e. more than 95% data during event)
    nstats_avail_event = (data_avail_stats >= data_avail_cp).sum(axis=1) 
    perc_stats_avail_event = nstats_avail_event/nstats
    
    # Number of stations affected by CP and percentage relative to all (available) stations
    nstats_cp = cp_stats.groupby('ICP').nunique()['STATION']
    #perc_stats_cp = (nstats_cp/nstats_avail_event) * 100
    perc_stats_cp = (nstats_cp/nstats) * 100

    
    if start_date.year == 2020:
        title = 'Cold pool events FESST@HH 2020'
        pname = 'fessthh_cp_overview.pdf' # pdf,dpi 300
        xlabel = 'Maximum temperature perturbation (K)'
    
    if start_date.year == 2021:
        title = 'Cold pool events FESSTVaL 2021'
        pname = 'events_dist_dT.png' #'fesstval_cold_pools_time.png'
        xlabel = r'$\Delta T_{\mathrm{min}}$ (K)'
        
    if fig_numbers: 
        pname = 'Kirsch_Fig03.pdf'    
    
    ypos = np.arange(ncp_fesstval)
    ymin,ymax = -0.75,ncp_fesstval-0.25
    xmin,xmax = -12.5,0.5
    ms = 18
    lw,wl = 0.8,0.3
    
    print('')
    print('Plotting dT distribution of CP events')
    fig,ax = plt.subplots(1,1,figsize=(6,7),dpi=300)
    
    for i,i_s in enumerate(ii_sort):
        col = 'k'
        if perc_stats_avail_event.loc[i_s] < lim_avail_event: col = 'lightgrey' 
        pxmed = cp_stats_median.loc[i_s]['DT']
        pxmin = cp_stats_min.loc[i_s]['DT']
        pxmax = cp_stats_max.loc[i_s]['DT']
        ax.scatter([pxmed],[i],color=col,s=ms)
        ax.plot([pxmin,pxmax],[i,i],color=col,linewidth=lw)
        ax.plot([pxmin,pxmin],[i-wl,i+wl],color=col,linewidth=lw)
        ax.plot([pxmax,pxmax],[i-wl,i+wl],color=col,linewidth=lw)
        
        if start_date.year == 2021:
            name = cp_fesstval['NAME'].fillna('').iloc[i_s]
            ax.text(xmin+0.2,i,name,fontsize=fs-3,ha='left',va='center')
    
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_yticks(ypos)
    ax.set_yticklabels((cp_fesstval['INDEX'].iloc[ii_sort].index+dt.timedelta(hours=2))\
                       .strftime('%b %d, %H%M h'),fontdict={'fontsize': fs-3})
    ax.grid(visible=False,axis='y')
    ax.tick_params(axis='y',length=0)
    ax.set_xlabel(xlabel,fontsize=fs)
    ax.set_ylabel('Date and start time',fontsize=fs)
    
    ax2 = ax.twinx()
    ax2.set_ylim(ymin,ymax)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(perc_stats_cp.loc[ii_sort].apply(lambda x:'{:2.0f}'.format(x)),
                        fontdict={'fontsize': fs-3})
    
    ax2.grid(visible=False,axis='y')
    ax2.tick_params(axis='y',length=0)
    ax2.set_ylabel('Percentage of affected stations',fontsize=fs)  
    
    plt.tight_layout()
    fig.savefig(plotdir+pname,bbox_inches='tight')
    plt.close()    
    print('Plot done!')

    
#----------------------------------------------------------------------------
print(' ')
print('*** Finshed! ***')
fst.print_runtime(t_run)
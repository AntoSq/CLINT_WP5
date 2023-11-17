# CLINT001_functions.py>

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy


def apply_land_sea_mask(targetxr, lsm):
    
    ## The era5 netcdf is adapted to the format of maskedanom,
    ##  so that the mask can be applied
    lsm_mask = lsm.reindex(longitude=targetxr.longitude, 
                            latitude=targetxr.latitude,
                            method="nearest", 
                            tolerance=1e-9, 
                            fill_value=0).squeeze()


    ## Maskedanom is updated removing the gridpoints on the sea
    outputxr = targetxr.where(lsm_mask['lsm'])
    return(outputxr)

def daily_series_w_lags(xrdf, drivers_row, targetdate_ts, what, plotdir):

    y = targetdate_ts.year
    m = str(targetdate_ts.month).zfill(2)
    d = str(targetdate_ts.day).zfill(2)
    
    if what == 'centroid':
        cc_lon = drivers_row['cluster_centre_lon']
        cc_lat = drivers_row['cluster_centre_lat']
        cc_ser = xrdf.sel(longitude = cc_lon, latitude = cc_lat)
        
    if what == 'average':
        
        cc_ser = xrdf.mean(dim=['longitude','latitude'])#average on the whole domain
        
    
    var = drivers_row['var']
    ncvar = drivers_row['nc_var']
    cl_name = drivers_row['cluster']
    
    minlag = int(drivers_row['minlag'])
    maxlag = int(drivers_row['maxlag'])
    # Determine extremes of the date range considered
    mintime_ts = targetdate_ts - pd.DateOffset(days = maxlag)# + pd.DateOffset(hours = 12)
    maxtime_ts = targetdate_ts - pd.DateOffset(days = minlag)# + pd.DateOffset(hours = 12)
    
    plt.rcParams['figure.figsize'] = [16,8]
    plt.figure()
    if what == 'centroid':
        cc_ser[ncvar].plot(color='red')
    if what == 'average':
        cc_ser[ncvar].plot(color='sienna')
    plt.title(f'{var} daily series ({y}), cluster {cl_name} {what}')
    plt.axhline(y=0, color='k')
    plt.axvline(x = targetdate_ts, color = 'k', label = 'axvline - full height')
    plt.axvline(x = mintime_ts, color = 'b', label = 'axvline - full height')
    plt.axvline(x = maxtime_ts, color = 'b', label = 'axvline - full height')
    plt.grid()
    plt.savefig(f"{plotdir}CLINT050_{y}{m}{d}case_{var}_{cl_name}.png", facecolor='w')
    plt.show()  
    

def expand_res_grid(submask_row,old_res=2,new_res=0.25):

    exp_start_lon = submask_row['nodes_lon'] - old_res/2
    exp_stop_lon = submask_row['nodes_lon'] + old_res/2 + new_res
    exp_start_lat = submask_row['nodes_lat'] - old_res/2 
    exp_stop_lat = submask_row['nodes_lat'] + old_res/2 + new_res
    
    #print([str(submask_row['nodes_lon']),str(submask_row['nodes_lat'])])
    
    exp_range_lon = np.arange(exp_start_lon,exp_stop_lon,step=new_res)
    exp_range_lat = np.arange(exp_start_lat,exp_stop_lat,step=new_res)
    add_df = pd.DataFrame([(x, y) for x in exp_range_lat for y in exp_range_lon])
    add_df.columns = ['nodes_lat','nodes_lon']
    return (add_df)

def loop_map_grids(drivers, dates_ts, variables, lsm, modeldir, maskdir, plotdir):

    
    for date_ts in dates_ts:
        y = date_ts.year
        #print(y)
        for var in variables:
            drivers_sub = drivers.loc[drivers['var'] == var]
            anom_xr = xr.open_dataset(f'{modeldir}/era5_{var}_dailyanom_{y}_cropped.nc')
            if var == 'mslp':
                anom_xr['msl'] = anom_xr['msl']/100
            for index, drivers_row in drivers_sub.iterrows():


                mask_df = pd.read_csv(f"{maskdir}{drivers_row['clmask_test3']}",index_col=[0])
                cl_nr = drivers_row['cl_nr']
                submask = mask_df[mask_df.cluster == cl_nr]
                maskedanom = mask_xr_w_df(var, anom_xr, submask, lsm)

                multimaps_lag (xrdf = maskedanom, targetdate_ts = date_ts, 
                                     drivers_row = drivers_row, plotdir = plotdir,
                                     proj = 'Ortographic', 
                                     vmin='drivers', vmax='drivers')

                daily_series_w_lags(anom_xr, drivers_row, date_ts, 'centroid', plotdir)
                daily_series_w_lags(anom_xr, drivers_row, date_ts, 'average', plotdir)
                
    
def map_EuroPP (xrdf, targetdate_ts, lag, drivers_row, vmin, vmax):
    
    targetdate_str = dt.strftime(targetdate_ts,'%Y-%m-%d')

    nc_var = drivers_row.nc_var
    plotdate_ts = targetdate_ts - pd.DateOffset(days = lag) + pd.DateOffset(hours = 12)
    sub1d = xrdf.sel(time=plotdate_ts)
    
    plt.figure(figsize=(6, 6))

    
    ax = plt.axes(projection=ccrs.EuroPP())
    ax.set_global()

    sub1d[nc_var].plot(ax=ax, 
                       cmap=plt.cm.RdBu_r, vmin = vmin, vmax = vmax,
                       transform=ccrs.PlateCarree())
    plt.title(f'{var}, {lag} days before {targetdate_str}')
    
    
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines()
    
    plt.show()
    print(lag)

def map_Ortographic (xrdf, targetdate_ts, lag, drivers_row, vmin='drivers', vmax='drivers',
                     fig_width = 8, fig_height = 8, mult=False):
 

    if vmin == 'drivers':
        vmin = drivers_row['vmin']
    if vmax == 'drivers':
        vmax = drivers_row['vmax']
    targetdate_str = dt.strftime(targetdate_ts,'%Y-%m-%d')
    var = drivers_row['var']
    nc_var = drivers_row.nc_var
    plotdate_ts = targetdate_ts - pd.DateOffset(days = lag)# + pd.DateOffset(hours = 12)
    sub1d = xrdf.sel(time=plotdate_ts)
    plt.figure(figsize=(fig_width, fig_height))

    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=drivers_row['cl_ortho_lon'], 
                                               central_latitude=drivers_row['cl_ortho_lat']))
    ax.set_extent(drivers_row[['cl_ext_W','cl_ext_E','cl_ext_S','cl_ext_N']])    

    sub1d[nc_var].plot(ax=ax, transform=ccrs.PlateCarree(),
                       cmap=plt.cm.RdBu_r, vmin = vmin, vmax = vmax)
        
    
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    plt.scatter(x=drivers_row['cluster_centre_lon'], y=drivers_row['cluster_centre_lat'],
                marker='X',color='black',s=600,transform=ccrs.PlateCarree(),alpha=1)
    plt.title(f'{var}, {lag} days before {targetdate_str}', fontsize = 30)

    ax.gridlines()
    
    plt.show()
    

def mask_xr_w_df (var, xrdf, submask, lsm):

    ## Generate coordinates of all gridpoints around the 2x2
    exp_list = [expand_res_grid(row) for index,row in submask.iterrows()] 
    ## Concatenate dataframes with all the new gridpoints
    exp_df = pd.concat(exp_list,ignore_index=True).reset_index(drop=True)
    exp_df['cluster'] = np.nanmean(submask['cluster']) #add the missing column so that it can be merged to submask
    ## Append the new gridpoints to the original 2x2 mask
    submask_exp = pd.concat([submask,exp_df],ignore_index=True).drop_duplicates().reset_index(drop=True)

    mask = submask_exp.assign(flag=1).set_index(["nodes_lon", "nodes_lat"]).flag.to_xarray().fillna(0).rename({"nodes_lon": "longitude", "nodes_lat": "latitude"})

    mask = mask.reindex(longitude=xrdf.longitude, 
                        latitude=xrdf.latitude,method="nearest", tolerance=1e-9, fill_value=0)

    sub_xrdf = xrdf.where(mask)
    
    if var == 'tmax':
        sub_xrdf = apply_land_sea_mask(sub_xrdf, lsm)
    
    return(sub_xrdf)

   
def multimaps_lag (xrdf, targetdate_ts, drivers_row, plotdir,
                   proj='Ortographic', vmin='drivers', vmax='drivers'):
    minlag = int(drivers_row['minlag'])
    maxlag = int(drivers_row['maxlag'])
    fig_width = drivers_row['fig_width']
    if np.isnan(fig_width):
        fig_width = 48

    ax_width = drivers_row['ax_width']
    if np.isnan(ax_width):
        ax_width = 12

    ax_height = drivers_row['ax_height']
    if np.isnan(ax_height):
        ax_height = 8

    text_plot_lon = drivers_row['text_plot_lon']
    if np.isnan(text_plot_lon):
        text_plot_lon = drivers_row['cl_ext_E']-2

    text_plot_lat = drivers_row['text_plot_lat']
    if np.isnan(text_plot_lat):
        text_plot_lat = drivers_row['cl_ext_N']-2

    targetdate_str = dt.strftime(targetdate_ts,'%Y-%m-%d')
    
    y = targetdate_ts.year
    m = str(targetdate_ts.month).zfill(2)
    d = str(targetdate_ts.day).zfill(2)

    
    var = drivers_row['var']
    # Determine extremes of the date range considered, in the netcdf every day has 12:00 as hour
    mintime_ts = targetdate_ts - pd.DateOffset(days = maxlag) + pd.DateOffset(hours = 12)
    maxtime_ts = targetdate_ts - pd.DateOffset(days = minlag) + pd.DateOffset(hours = 12)

    mintime_str = dt.strftime(mintime_ts,'%Y-%m-%d')
    maxtime_str = dt.strftime(mintime_ts,'%Y-%m-%d')

    numfigs = len(range(minlag,maxlag+1))
    numfigs_h = np.floor(fig_width/ax_width)
    numfigs_v = np.ceil(numfigs/numfigs_h)


    cl_name = drivers_row['cluster']
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(int(numfigs_v), int(numfigs_h),
                            subplot_kw={'projection': proj},
                            figsize=(fig_width,int(numfigs_v*ax_height)),
                            sharey=True,sharex=True)#,
                            #layout="constrained")
    fig.suptitle(f'{var}, {minlag} to {maxlag} days before {targetdate_str}', fontsize = 100)
    axs = axs.flatten()
    for f,lag in enumerate(range(minlag,maxlag+1)):

        if vmin == 'drivers':
            vmin = drivers_row['vmin']
        if vmax == 'drivers':
            vmax = drivers_row['vmax']
        var = drivers_row['var']
        nc_var = drivers_row.nc_var
        
        if var == 'tmax':
            palette = plt.cm.RdBu_r
        if var == 'mslp':
            palette = plt.cm.PRGn_r
        if var == 'sm':
            palette = plt.cm.BrBG
        
        targetdate_str = dt.strftime(targetdate_ts,'%Y-%m-%d')
        

        plotdate_ts = targetdate_ts - pd.DateOffset(days = lag)# + pd.DateOffset(hours = 12)
        sub1d = xrdf.sel(time=plotdate_ts)
        axs[f].remove()
        geo_axes = plt.subplot(int(numfigs_v), int(numfigs_h), f+1,
                               projection=ccrs.Orthographic(central_longitude=drivers_row['cl_ortho_lon'],
                                                            central_latitude=drivers_row['cl_ortho_lat']))
        
        cs=sub1d[nc_var].plot(ax=geo_axes,transform=ccrs.PlateCarree(),cmap=palette, 
                           vmin = vmin, vmax = vmax,add_colorbar=False)
        geo_axes.set_extent(drivers_row[['cl_ext_W','cl_ext_E','cl_ext_S','cl_ext_N']])    
        geo_axes.scatter(x=drivers_row['cluster_centre_lon'], y=drivers_row['cluster_centre_lat'],
                    marker='X',color='black',s=900,transform=ccrs.PlateCarree(),alpha=1)
        geo_axes.coastlines()
        geo_axes.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
        #geo_axes.text(x=text_plot_lon, y=text_plot_lat, 
        #              s=f'-{lag}d', transform=ccrs.PlateCarree(), fontsize=70)
        geo_axes.text(x=0.875, y=0.9, horizontalalignment='center', verticalalignment='center',
                      s=f'-{lag}d', transform=geo_axes.transAxes, fontsize=70)
        
        geo_axes.gridlines()
        plt.title(None)
        #plt.title(f'-{lag}d', fontsize = 50)
        #geo_axes._legend.remove()
    #handles, labels = geo_axes.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    for f in range(numfigs,int(numfigs_h*numfigs_v)):
        #print(f)
        fig.delaxes(axs[int(f)])  

    fig.subplots_adjust(bottom=0.04+1/(numfigs_v*ax_height), top=0.94, left=0.1, right=0.9,
                        wspace=0.02, hspace=0.02)

    cbar_ax = fig.add_axes([0.07, 0.03, 0.86,1/(numfigs_v*ax_height)])

    cbar=fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=50)
    
    plt.savefig(f"{plotdir}CLINT040_maps_{y}{m}{d}case_{var}_{cl_name}.png", facecolor='w')
    
    
    
def set_maps_lag (xrdf, targetdate_ts, drivers_row, proj, vmin='drivers', vmax='drivers',
                  fig_width = 8, fig_height = 8):
    
    minlag = int(drivers_row['minlag'])
    maxlag = int(drivers_row['maxlag'])

    # Determine extremes of the date range considered, in the netcdf every day has 12:00 as hour
    mintime_ts = targetdate_ts - pd.DateOffset(days = maxlag) + pd.DateOffset(hours = 12)
    maxtime_ts = targetdate_ts - pd.DateOffset(days = minlag) + pd.DateOffset(hours = 12)

    mintime_str = dt.strftime(mintime_ts,'%Y-%m-%d')
    maxtime_str = dt.strftime(mintime_ts,'%Y-%m-%d')
    if proj == 'EuroPP':
        for lag in range(minlag,maxlag+1):
            map_EuroPP(xrdf = xrdf, targetdate_ts = targetdate_ts, lag = lag, drivers_row = drivers_row,
                      vmin = vmin, vmax = vmax)
    if proj == 'Orthographic':
        for lag in range(minlag,maxlag+1):
            map_Ortographic (xrdf, targetdate_ts, lag, drivers_row, vmin=vmin, vmax=vmax,
                             fig_width = fig_width, fig_height = fig_height)

    
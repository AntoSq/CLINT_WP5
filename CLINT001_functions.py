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
import matplotlib.ticker as mticker
import glob

def add_driver_row (drivers_old, drivers_input, drivers_tmpl, var_specs, workmetadir):

    if ((drivers_old.loc[(drivers_old['var']==drivers_input['var'][0]) &
                    (drivers_old['domain']==drivers_input['domain'][0]) &
                    (drivers_old['exp_size']==drivers_input['exp_size'][0]) &
                    (drivers_old['cl_nr']==drivers_input['cl_nr'][0])].shape[0]) == 0):

        drivers_addrow = drivers_tmpl.copy()


        for clm in drivers_input.columns:
            #print(clm)
            drivers_addrow[clm] = drivers_input[clm]
            
        var_row = var_specs.loc[var_specs['var'] == drivers_input['var'][0]]
        drivers_addrow['era5_var'] = var_row['era5_var']
        drivers_addrow['cmip6_var'] = var_row['cmip6_var']
        drivers_addrow['vmin'] = -20
        drivers_addrow['vmax'] = 20
        var4path='msl'

                
        
        if drivers_addrow['var'][0] == 'mslp':
            drivers_addrow['era5_var'] = 'msl'
            drivers_addrow['cmip6_var'] = 'pls'
            drivers_addrow['vmin'] = -20
            drivers_addrow['vmax'] = 20
            var4path='msl'

        if drivers_addrow['var'][0] == 'tmax':
            drivers_addrow['era5_var'] = 't2m'
            drivers_addrow['cmip6_var'] = 'tasmax'
            drivers_addrow['vmin'] = -10
            drivers_addrow['vmax'] = 10
            var4path='t2m'
            
        if drivers_addrow['var'][0] == 'sm':
            drivers_addrow['era5_var'] = 'swvl1'
            drivers_addrow['cmip6_var'] = 'mrsos'
            drivers_addrow['vmin'] = -0.2
            drivers_addrow['vmax'] = 0.2
            var4path='sm1'
        
        if drivers_addrow['var'][0] == 'sic':
            drivers_addrow['era5_var'] = 'sic'
            drivers_addrow['cmip6_var'] = 'sic'
            drivers_addrow['vmin'] = np.nan
            drivers_addrow['vmax'] = np.nan
            var4path='sm1'

        if drivers_addrow['exp_size'][0] == 'low':
            tot_num_cl = 5
        if drivers_addrow['exp_size'][0] == 'middle':
            tot_num_cl = 10
        if drivers_addrow['exp_size'][0] == 'high':
            tot_num_cl = 20

        #/data/csp/as18623/CLINT_metadata/Masks/Test5low_Clusters
        drivers_addrow['clmask_path'] = f"/data/csp/as18623/CLINT_metadata/Masks/Test{drivers_addrow['exp'][0]}{drivers_addrow['exp_size'][0]}_Clusters/"
        maskfile = glob.glob(drivers_addrow['clmask_path'][0] + f"labels??{var4path}{drivers_addrow['domain'][0]}{tot_num_cl}.csv")[0]
        drivers_addrow['clmask_file'] = maskfile 
        mask = pd.read_csv(maskfile,index_col=[0])
        sub_mask = mask.loc[mask.cluster == drivers_addrow['cl_nr'][0]-1]

        ## determine latlon box for graphical needs

        drivers_addrow['cl_ext_S'] = np.min(sub_mask.nodes_lat) - 1
        drivers_addrow['cl_ext_N'] = np.max(sub_mask.nodes_lat) + 1
        drivers_addrow['cl_ext_W'] = np.min(sub_mask.nodes_lon) - 1
        drivers_addrow['cl_ext_E'] = np.max(sub_mask.nodes_lon) + 1    
        drivers_addrow['cl_ortho_lat'] = (np.min(sub_mask.nodes_lat)+np.max(sub_mask.nodes_lat))/2
        drivers_addrow['cl_ortho_lon'] = (np.min(sub_mask.nodes_lon)+np.max(sub_mask.nodes_lon))/2

        drivers_new = pd.concat([drivers_old,drivers_addrow])
        
        drivers_new = drivers_new[drivers_tmpl.columns]
        
        drivers_new.to_csv(f"{workmetadir}drivers{drivers_addrow['exp'][0]}_Test.csv", index=False)
        print('YES! Driver correctly added')
    else:
        drivers_old = drivers_old[drivers_tmpl.columns]
        drivers_old.to_csv(f"{workmetadir}drivers{drivers_input['exp'][0]}_Test.csv", index=False)
        print('NO! This driver has already been added')
        
def apply_land_sea_mask(targetxr, lsm, kind):
    
    ## The era5 netcdf is adapted to the format of maskedanom,
    ##  so that the mask can be applied
    
    if (kind == 'hist'):
        lonfield = 'lon'
        latfield = 'lat'
    else:
        lonfield = 'longitude'
        latfield = 'latitude'

    
    
    #lsm_mask = lsm.reindex(longitude=targetxr.longitude, 
    #                        latitude=targetxr.latitude,
    #                        method="nearest", 
    #                        tolerance=1e-9, 
    #                        fill_value=0).squeeze()
    lsm_mask = lsm.interp(longitude=targetxr[lonfield], 
                     latitude=targetxr[latfield], 
                     method="nearest")
    lsm_mask = lsm_mask.fillna(0)
    lsm_mask = lsm_mask.squeeze('time')
    ## Maskedanom is updated removing the gridpoints on the sea
    outputxr = targetxr.where(lsm_mask['lsm'])
    return(outputxr)

def daily_series_w_lags(xrdf, drivers_row, targetdate_ts, what, kind, plotdir):

    y = targetdate_ts.year
    m = str(targetdate_ts.month).zfill(2)
    d = str(targetdate_ts.day).zfill(2)
    
    vmin = drivers_row['vmin']
    vmax = drivers_row['vmax']
    
    
    if (kind == 'hist'):
        lonfield = 'lon'
        latfield = 'lat'
        nc_var = drivers_row.cmip6_var
    else:
        lonfield = 'longitude'
        latfield = 'latitude'
        nc_var = drivers_row.era5_var
    
    
    
    if what == 'centroid':
        cc_lon = drivers_row['cluster_centre_lon']
        cc_lat = drivers_row['cluster_centre_lat']
        cc_ser = xrdf.sel({lonfield : cc_lon, latfield : cc_lat}, method = 'nearest')        
    if what == 'average':
        cc_ser = xrdf.mean(dim=[lonfield,latfield])#average on the whole domain
    if what == 'quantiles':        
        cc_ser75 = xrdf.quantile(q=0.75, dim=[lonfield,latfield])#average on the whole domain
        cc_ser10 = xrdf.quantile(q=0.10, dim=[lonfield,latfield])#average on the whole domain
        cc_ser25 = xrdf.quantile(q=0.25, dim=[lonfield,latfield])#average on the whole domain
        cc_ser50 = xrdf.quantile(q=0.50, dim=[lonfield,latfield])#average on the whole domain
        cc_ser90 = xrdf.quantile(q=0.90, dim=[lonfield,latfield])#average on the whole domain


        
    
    var = drivers_row['var']
    #ncvar = drivers_row['nc_var']
    cl_name = drivers_row['cl_name']
    
    minlag = int(drivers_row['minlag'])
    maxlag = int(drivers_row['maxlag'])
    # Determine extremes of the date range considered
    mintime_ts = targetdate_ts - pd.DateOffset(days = maxlag)# + pd.DateOffset(hours = 12)
    maxtime_ts = targetdate_ts - pd.DateOffset(days = minlag)# + pd.DateOffset(hours = 12)
    
    plt.rcParams['figure.figsize'] = [16,8]
    #plt.figure(figsize=((16,8)))
    fig,ax = plt.subplots()
    if what == 'centroid':
        cc_ser[nc_var].plot(color='red')
    if what == 'average':
        cc_ser[nc_var].plot(color='sienna')
    if what == 'quantiles':
        ax.fill_between(x=cc_ser10['time'],y1=cc_ser10[nc_var],y2=cc_ser90[nc_var],color='mistyrose')
        ax.fill_between(x=cc_ser25['time'],y1=cc_ser25[nc_var],y2=cc_ser75[nc_var],color='pink')
        cc_ser50[nc_var].plot(color='mediumvioletred')

        
    ax.set_ylim(vmin,vmax)   
    plt.title(f'{var} daily series ({y}), cluster {cl_name} {what}',fontsize=30)
    plt.axhline(y=0, color='k')
    plt.axvline(x = targetdate_ts, color = 'k', label = 'axvline - full height')
    plt.axvline(x = mintime_ts, color = 'b', label = 'axvline - full height')
    plt.axvline(x = maxtime_ts, color = 'b', label = 'axvline - full height')
    plt.grid()
    plt.savefig(f"{plotdir}CLINT050_{y}{m}{d}case_{var}_{cl_name}_{what}.png", facecolor='w')
    plt.show()

def calc_clim (var,tres,product,experiment,ensemble,year_start,year_stop,path):
    ## calculate climatology of a given xarray dataset
    list4clim = [f"{ERA5path}{var}_{tres}_{product}_{experiment}_{ensemble}_{y}0101-{y}1231.nc" for y in range(year_start,year_stop+1)]
    baseclim = xr.open_mfdataset(list4clim)
    xr_clim = baseclim.groupby("time.dayofyear").mean("time")
    return (xr_clim)

def expand_res_grid(submask_row,old_res=0.5,new_res=0.25):

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

def loop_map_grids(drivers, dates_ts, variables, lsm, modelspecs1, modeldir, maskdir, plotdir):

    
    for date_ts in dates_ts:
        y = date_ts.year
        #print(y)
        kind = modelspecs1.iloc[0]['kind']
        if kind == 'ERA5':
            datasetnames = ['era5']
            kind_var = 'era5_var'
        else:
            #[print(modelrow) for i, modelrow in modelspecs.iterrows()]
            datasetnames = sum([[f'{modelrow["modelnames"]}-{kind}-r{mmb}' for mmb in modelrow["members_list"]] for i, modelrow in modelspecs1.iterrows()],[])
            kind_var = 'cmip6_var'
        for var in variables:
            drivers_sub = drivers.loc[drivers['var'] == var]
            nc_var = drivers.loc[drivers['var'] == var].iloc[0][kind_var]
            for datasetname in datasetnames:
                anom_xr = xr.open_dataset(f'{modeldir}/{datasetname}_{var}_dailyanom_{y}_cropped.nc')
                anom_xr = anom_xr.convert_calendar('gregorian')
                if var == 'mslp':
                    anom_xr[nc_var] = anom_xr[nc_var]/100
                for index, drivers_row in drivers_sub.iterrows():
                    mask_df = pd.read_csv(f"{drivers_row['clmask_file']}",index_col=[0])
                    cl_nr = drivers_row['cl_nr']
                    submask = mask_df[mask_df.cluster == cl_nr-1] #python indexing, cluster 1 is nr0 in the mask file...
                    maskedanom = mask_xr_w_df(var, anom_xr, submask, lsm, kind)

                    multimaps_lag (xrdf = maskedanom, targetdate_ts = date_ts, 
                                        drivers_row = drivers_row, kind=kind, plotdir = plotdir,
                                        proj = 'Ortographic', 
                                        vmin='drivers', vmax='drivers')
                    daily_series_w_lags(maskedanom, drivers_row, date_ts, 'average', kind, plotdir)
                    daily_series_w_lags(maskedanom, drivers_row, date_ts, 'quantiles', kind, plotdir)
                    #daily_series_w_lags(maskedanom, drivers_row, date_ts, 'centroid', kind, plotdir)

                
    
def map_EuroPP (xrdf, targetdate_ts, lag, drivers_row, vmin, vmax):
    
    targetdate_str = dt.strftime(targetdate_ts,'%Y-%m-%d')

    nc_var = drivers_row.era5_var
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
    nc_var = drivers_row.era5_var
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
    

def mask_xr_w_df (var, xrdf, submask, lsm, kind):

    ## Generate coordinates of all gridpoints around the 2x2
    exp_list = [expand_res_grid(row) for index,row in submask.iterrows()] 
    ## Concatenate dataframes with all the new gridpoints
    exp_df = pd.concat(exp_list,ignore_index=True).reset_index(drop=True)
    exp_df['cluster'] = np.nanmean(submask['cluster']) #add the missing column so that it can be merged to submask
    ## Append the new gridpoints to the original 2x2 mask
    submask_exp = pd.concat([submask,exp_df],ignore_index=True).drop_duplicates().reset_index(drop=True)

    mask = submask_exp.assign(flag=1).set_index(["nodes_lon", "nodes_lat"]).flag.to_xarray().fillna(0).rename({"nodes_lon": "longitude", "nodes_lat": "latitude"})

    if (kind == 'hist'):
        lonfield = 'lon'
        latfield = 'lat'
    else:
        lonfield = 'longitude'
        latfield = 'latitude'
    
    #mask = mask.reindex(longitude=xrdf[lonfield], 
    #                    latitude=xrdf[latfield],method="nearest", tolerance=1e-9, fill_value=0)
    mask1 = mask.interp(longitude=xrdf[lonfield], 
                        latitude=xrdf[latfield], 
                        method="nearest")
    mask1 = mask1.fillna(0)
    sub_xrdf = xrdf.where(mask1)
    
    if var == 'tmax':
        sub_xrdf = apply_land_sea_mask(sub_xrdf, lsm, kind)
    
    return(sub_xrdf)

   
def multimaps_lag (xrdf, targetdate_ts, drivers_row, kind, plotdir,
                   proj='Ortographic', vmin='drivers', vmax='drivers'):
    
    var = drivers_row['var']
    cl_name = drivers_row['cl_name']
    
    minlag = int(drivers_row['minlag'])
    maxlag = int(drivers_row['maxlag'])
    
    fig_width = drivers_row['fig_width']
    if np.isnan(fig_width):
        fig_width = 48
    
    ## VALUES THAT DEPEND ON THE VARIABLE
    
    if vmin == 'drivers':
        vmin = drivers_row['vmin']
    if vmax == 'drivers':
        vmax = drivers_row['vmax']
        var = drivers_row['var']
        
    if kind == 'ERA5':
        nc_var = drivers_row.era5_var
    elif kind == 'hist':
        nc_var = drivers_row.cmip6_var

    if var == 'tmax':
        palette = plt.cm.RdBu_r
    if var == 'mslp':
        palette = plt.cm.PRGn_r
    if var == 'sm':
        palette = plt.cm.BrBG
        
    #print(f"KIND IS {drivers_row[['var','era5_var','cmip6_var']]}")
    
    ## DEFINE PROJECTION AND MAP EXTENT
        
    my_projn = ccrs.Orthographic(central_longitude=drivers_row['cl_ortho_lon'],
                                 central_latitude=drivers_row['cl_ortho_lat'])
    lonlatproj = ccrs.PlateCarree()
    # These chunk, first takes the limits of the box then calculates the coordinated in the projection
    # In the last line, the extreme coordinates in the new reference are identified
    # These extremes are give to set_extent()
    SW_lon, SW_lat = my_projn.transform_point(drivers_row['cl_ext_W']-1, 
                                            drivers_row['cl_ext_S']-1, 
                                            lonlatproj)  #(0.0, -3189068.5)
    NE_lon, NE_lat = my_projn.transform_point(drivers_row['cl_ext_E']+1, 
                                            drivers_row['cl_ext_N']+1, 
                                            lonlatproj) #(3189068.5, 0)
    NW_lon, NW_lat = my_projn.transform_point(drivers_row['cl_ext_W']-1, 
                                            drivers_row['cl_ext_N']+1, 
                                            lonlatproj)  #(0.0, -3189068.5)
    SE_lon, SE_lat = my_projn.transform_point(drivers_row['cl_ext_E']+1, 
                                            drivers_row['cl_ext_S']-1, 
                                            lonlatproj) #(3189068.5, 0)
    SC_lon, SC_lat = my_projn.transform_point(drivers_row['cl_ortho_lon'], 
                                            drivers_row['cl_ext_S']-1, 
                                            lonlatproj) #(3189068.5, 0)
    NC_lon, NC_lat = my_projn.transform_point(drivers_row['cl_ortho_lon'], 
                                            drivers_row['cl_ext_N']+1, 
                                            lonlatproj) #(3189068.5, 0)

    xmin,xmax,ymin,ymax = min(SW_lon,NW_lon), max(NE_lon,SE_lon), min(SW_lat,SE_lat,SC_lat), max(NE_lat,NW_lat, NC_lat)
        
    ## DEFINE SIZE OF SUBPLOTS
    
    numfigs = len(range(minlag,maxlag+1)) ## number of figures depends on how many days
    numfigs_h = np.floor((12000000/(xmax-xmin)+3)/2) ## number of figures per row depends on the size of the cluster
    numfigs_v = np.ceil(numfigs/numfigs_h) ## number of rows is calculated by consequence

    map_ratio = (xmax-xmin)/(ymax-ymin) ## width divided by height
    
    ax_width = drivers_row['ax_width']
    if np.isnan(ax_width):        
        ax_width = (fig_width)/numfigs_h    

    ax_height = drivers_row['ax_height']
    if np.isnan(ax_height):
        ax_height = ax_width/map_ratio
        
    ## DEFINE WHERE TO WRITE THE LAG    

    text_plot_lon = drivers_row['text_plot_lon']
    if np.isnan(text_plot_lon):
        text_plot_lon = drivers_row['cl_ext_E']-2

    text_plot_lat = drivers_row['text_plot_lat']
    if np.isnan(text_plot_lat):
        text_plot_lat = drivers_row['cl_ext_N']-2
        
    ## IDENTIFY THE DATES

    targetdate_str = dt.strftime(targetdate_ts,'%Y-%m-%d %H')
    
    y = targetdate_ts.year
    m = str(targetdate_ts.month).zfill(2)
    d = str(targetdate_ts.day).zfill(2)
    
    # Determine extremes of the date range considered, in the netcdf every day has 12:00 as hour
    mintime_ts = targetdate_ts - pd.DateOffset(days = maxlag) + pd.DateOffset(hours = 12)
    maxtime_ts = targetdate_ts - pd.DateOffset(days = minlag) + pd.DateOffset(hours = 12)

    mintime_str = dt.strftime(mintime_ts,'%Y-%m-%d')
    maxtime_str = dt.strftime(mintime_ts,'%Y-%m-%d')
    
    targetdate_str = dt.strftime(targetdate_ts,'%Y-%m-%d')


    

    fig, axs = plt.subplots(int(numfigs_v), int(numfigs_h),
                            subplot_kw={'projection': lonlatproj},
                            figsize=(fig_width,int(numfigs_v*ax_height)),
                            sharey=True,sharex=True)#,
                            #layout="constrained")
    fig.suptitle(f'{var}, {minlag} to {maxlag} days before {targetdate_str}', fontsize = 100)
    axs = axs.flatten()        
        
    

    for f,lag in enumerate(range(minlag,maxlag+1)):

        

        plotdate_ts = targetdate_ts - pd.DateOffset(days = lag)# + pd.DateOffset(hours = 12)
        #print(plotdate_ts)
        sub1d = xrdf.sel(time=plotdate_ts)
        axs[f].remove()
        geo_axes = plt.subplot(int(numfigs_v), int(numfigs_h), f+1,
                               projection=my_projn)
        
        #print(sub1d.variables)
        
        cs=sub1d[nc_var].plot(ax=geo_axes,transform=ccrs.PlateCarree(),cmap=palette, 
                           vmin = vmin, vmax = vmax,add_colorbar=False)
        
        
        
        
        
        geo_axes.scatter(x=drivers_row['cl_centroid_lon'], y=drivers_row['cl_centroid_lat'],
                    marker='X',color='black',s=900,transform=ccrs.PlateCarree(),alpha=1)
        geo_axes.coastlines()
        geo_axes.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
        #geo_axes.text(x=text_plot_lon, y=text_plot_lat, 
        #              s=f'-{lag}d', transform=ccrs.PlateCarree(), fontsize=70)
        geo_axes.text(x=0.875, y=0.9, horizontalalignment='center', verticalalignment='center',
                      s=f'-{lag}d', transform=geo_axes.transAxes, fontsize=70)
        geo_axes.set_extent([xmin,xmax,ymin,ymax], crs=my_projn) # data/projection coordinates  
        gl = geo_axes.gridlines(lonlatproj,linewidth=2, color='lightgray', alpha=0.5)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
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
    cbar.set_label(f"[{drivers_row['unit']}]",fontsize=50)
    cbar.ax.tick_params(labelsize=50)
    

    #plt.savefig(f"{plotdir}CLINT040_maps_{y}{m}{d}case_{var}_Test{drivers_row['exp']}{drivers_row['exp_size']}{cl_name}.png", facecolor='w')
    
def rearrange_lon (xrdf):
    ## Rearrange longitude of xarray datasets where longitudes are in the [0,360) range
    ## Longitudes are translated to [-180,180)
    test = xrdf.assign_coords(lon=(((clim8110.lon + 180) % 360) - 180))
    test = test.sortby(test.lon)
    return(test)
    
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

    
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
import os
import sys


def add_driver_row (drivers_old, drivers_input, drivers_tmpl, var_specs, maskdir, workmetadir):

    if ((drivers_old.loc[(drivers_old['var']==drivers_input['var'][0]) &
                    (drivers_old['domain']==drivers_input['domain'][0]) &
                    (drivers_old['exp_size']==drivers_input['exp_size'][0]) &
                    (drivers_old['cl_nr']==drivers_input['cl_nr'][0])].shape[0]) == 0):

        drivers_addrow = drivers_tmpl.copy()


        for clm in drivers_input.columns:
            #print(clm)
            drivers_addrow[clm] = drivers_input[clm].values[0]
            
        var_row = var_specs.loc[var_specs['var'] == drivers_input['var'].values[0]]
        for clm in var_row.columns:
            drivers_addrow[clm] = var_row[clm].values[0]
            
        
        #drivers_addrow['era5cmor_var'] = var_row['era5cmor_var'].values[0]
        #drivers_addrow['era5cds_var'] = var_row['era5cds_var'].values[0]
        #drivers_addrow['cmip6_var'] = var_row['cmip6_var'].values[0]
        #drivers_addrow['var4path'] = var_row['var4path'].values[0]
        var4path = var_row['var4path'].values[0]
        #print(var_row)
        #print(drivers_addrow)  
        
        if drivers_addrow['var'][0] == 'mslp':
            drivers_addrow['vmin'] = -20
            drivers_addrow['vmax'] = 20

        if drivers_addrow['var'][0] == 'tmax':

            drivers_addrow['vmin'] = -10
            drivers_addrow['vmax'] = 10

            
        if drivers_addrow['var'][0] == 'sm':

            drivers_addrow['vmin'] = -0.2
            drivers_addrow['vmax'] = 0.2

        
        if drivers_addrow['var'][0] == 'sic':
            drivers_addrow['era5cds_var'] = 'sic'
            drivers_addrow['era5cmor_var'] = 'sic'
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

        #print(drivers_addrow)
        #print(drivers_addrow['domain'].values[0])
        
        
        drivers_addrow['cl_name'] = f"cl{drivers_addrow['exp'].values[0]}{drivers_addrow['exp_size'].values[0]}_{drivers_addrow['domain'].values[0]}{drivers_addrow['cl_nr'].values[0]}_{drivers_addrow['var'].values[0]}"
        drivers_addrow['clmask_path'] = f"Test{drivers_addrow['exp'].values[0]}{drivers_addrow['exp_size'][0]}_Clusters/"

        #print(f"{maskdir}labels??{var4path}{drivers_addrow['domain'].values[0]}{tot_num_cl}.csv")
        maskfile = glob.glob(f"{maskdir}labels??{var4path}{drivers_addrow['domain'].values[0]}{tot_num_cl}.csv")[0]
        #print(maskfile)
        drivers_addrow['clmask_file'] = maskfile.split('/')[-1]
        mask = pd.read_csv(maskfile,index_col=[0])
        sub_mask = mask.loc[mask.cluster == drivers_addrow['cl_nr'].values[0]-1]

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
    
        
    
    #lsm_mask = lsm.reindex(longitude=targetxr.longitude, 
    #                        latitude=targetxr.latitude,
    #                        method="nearest", 
    #                        tolerance=1e-9, 
    #                        fill_value=0).squeeze()
    lsm_mask = lsm.interp(longitude=targetxr['lon'], 
                     latitude=targetxr['lat'], 
                     method="nearest")
    lsm_mask = lsm_mask.fillna(0)
    lsm_mask = lsm_mask.squeeze('time')
    ## Maskedanom is updated removing the gridpoints on the sea
    outputxr = targetxr.where(lsm_mask['lsm'])
    return(outputxr)

def daily_series_w_lags(xrdf, nc_var, drivers_row, targetdate_ts, what, kind, plotdir):

    y = targetdate_ts.year
    m = str(targetdate_ts.month).zfill(2)
    d = str(targetdate_ts.day).zfill(2)
    
    vmin = drivers_row['vmin']
    vmax = drivers_row['vmax']

    
    
    
    if what == 'centroid':
        cc_lon = drivers_row['cluster_centre_lon']
        cc_lat = drivers_row['cluster_centre_lat']
        cc_ser = xrdf.sel({'lon' : cc_lon, 'lat' : cc_lat}, method = 'nearest')        
    if what == 'average':
        cc_ser = xrdf.mean(dim=['lon','lat'])#average on the whole domain
    if what == 'quantiles':        
        cc_ser75 = xrdf.quantile(q=0.75, dim=['lon','lat'])#average on the whole domain
        cc_ser10 = xrdf.quantile(q=0.10, dim=['lon','lat'])#average on the whole domain
        cc_ser25 = xrdf.quantile(q=0.25, dim=['lon','lat'])#average on the whole domain
        cc_ser50 = xrdf.quantile(q=0.50, dim=['lon','lat'])#average on the whole domain
        cc_ser90 = xrdf.quantile(q=0.90, dim=['lon','lat'])#average on the whole domain


        
    
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
        cc_ser[nc_var].plot(color='red',label='centroid')
    if what == 'average':
        cc_ser[nc_var].plot(color='sienna',label='cluster avg')
    if what == 'quantiles':
        ax.fill_between(x=cc_ser10['time'],y1=cc_ser10[nc_var],y2=cc_ser90[nc_var],color='mistyrose',label='10th to 90th perc.')
        ax.fill_between(x=cc_ser25['time'],y1=cc_ser25[nc_var],y2=cc_ser75[nc_var],color='pink',label='25th to 75th perc.')
        cc_ser50[nc_var].plot(color='mediumvioletred',label='cluster median')
      
                                

        
    ax.set_ylim(vmin,vmax)   
    plt.title(f'{var} daily series ({y}), cluster {cl_name} {what}',fontsize=30)
    plt.axhline(y=0, color='k')
    lineminlag = plt.axvline(x = mintime_ts, color = 'b', label='cluster dates range')
    linemaxlag = plt.axvline(x = maxtime_ts, color = 'b')
    lineevent = plt.axvline(x = targetdate_ts,color = 'k',lw=3,label='event',linestyle='dotted')
    plt.grid()
    plt.legend()
    #plt.savefig(f"{plotdir}CLINT050_{y}{m}{d}case_{var}_{cl_name}_{what}.png", facecolor='w')
    plt.show()

def calc_clim (var,tres,product,experiment,ensemble,year_start,year_stop,path):
    ## calculate climatology of a given xarray dataset
    list4clim = [f"{path}{var}_{tres}_{product}_{experiment}_{ensemble}_{y}0101-{y}1231.nc" for y in range(year_start,year_stop+1)]
    baseclim = xr.open_mfdataset(list4clim)
    xr_clim = baseclim.groupby("time.dayofyear").mean("time")
    return (xr_clim)

def cdo_anom(nc_var,y,product,experiment,mmb,inpath,outpath):
    
    """
    Call CDO to calculate anomaly for the indicated year
    
    Antonello A. Squintu 2023-11-05
    
    Parameters
    ----------
    nc_var: str
        variable name, as it appears in the name of the file and in the path
    y: int
        target year (e.g. 1981)
    product: str
        name of the product (e.g. 'reanalysis')
    experiment: str
        name of the experiment according to Freva (e.g. 'era5',...)
    mmb: int
        number of the ensemble member
    inpath: str
        full path where the input files are stored
    outpath: str
        full path where the output files are to be stored
        
    
    Returns
    -------
        Creates netcdf file in outpath
    
    """
    print(f'Producing daily anomalies netcdf for {nc_var}, year {y}')
    infile = f'{inpath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'
    climfile = f'{outpath}{nc_var}_dailyclim_{product}_{experiment}_r{mmb}i1p1_clim8110.nc'
    outfile = f'{outpath}{nc_var}_dailyanom_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'
    os.system(f'cdo -b 32 -ydaysub {infile} {climfile} {outfile}')

def cdo_clim(nc_var,year_start,year_stop,product,experiment,mmb,inpath,outpath):
    
    """
    Call CDO to calculate climatology
    
    Antonello A. Squintu 2023-11-05
    
    Parameters
    ----------
    nc_var: str
        variable name, as it appears in the name of the file and in the path
    year_start: int
        first year in the range of the reference period (e.g. 1981)
    year_stop: int
        last year in the range of the reference period (e.g. 2010)
    product: str
        name of the product (e.g. 'reanalysis')
    experiment: str
        name of the experiment according to Freva (e.g. 'era5',...)
    mmb: int
        number of the ensemble member
    inpath: str
        full path where the input files are stored
    outpath: str
        full path where the output files are to be stored
        
    
    Returns
    -------
        Creates netcdf file in outpath
    
    """
    print(f'Producing climatology netcdf for {nc_var}, from {year_start} to {year_stop}')
    ## Copy the yearly files needed to calculate the climatology and name them temp_yyyy.nc, so thath
    ## you can call them within cdo ydayrunmean
    for y in range(year_start,year_stop+1):
        infile = f'{inpath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'
        outfile = f'{outpath}temp_{y}.nc'
        os.system(f'cp {infile} {outfile} ')
    ## Calculate daily climatology with a 31-days running window
    infiles = f'{outpath}temp_*.nc'
    outfile = f'{outpath}{nc_var}_dailyclim_{product}_{experiment}_r{mmb}i1p1_clim8110.nc'
    os.system(f'cdo -b 32 -ydrunmean,31 -cat "{infiles}" "{outfile}"')
    os.system(f'rm {outpath}temp_*.nc')

def cdo_daily_aggr(nc_var,y,aggr_operator,tres_filename,product,experiment,mmb,inpath,outpath): 
    
    """
    Call CDO to calculate climatology
    
    Antonello A. Squintu 2023-11-07
    
    Parameters
    ----------
    nc_var: str
        variable name, as it appears in the name of the file and in the path
    y: int
        year (e.g. 1981)
    aggr_operator: str
        which operator apply to hourly values to get daily values (e.g. 'mean','max')
    product: str
        name of the product (e.g. 'reanalysis')
    experiment: str
        name of the experiment according to Freva (e.g. 'era5',...)
    mmb: int
        number of the ensemble member
    inpath: str
        full path where the input files are stored
    outpath: str
        full path where the output files are to be stored
        
    
    Returns
    -------
        Creates netcdf file in outpath
    
    """
    print(f'Producing file with daily data for {nc_var}, year {y}')
    infiles = f'{inpath}{nc_var}_{tres_filename}_{product}_{experiment}_r{mmb}i1p1_{y}*-{y}*.nc'
    outfile = f'{outpath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'
    os.system(f'cdo day{aggr_operator} -mergetime {infiles} {outfile}')
    
    
    
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

def get_anom_w_cdo(nc_var, year, kind, year_start, year_stop, aggr_operator, tres, product,experiment,mmb, mmbpath, workpath):
    """
    Get yearly files with daily anomalies, when not available
    
    Antonello A. Squintu 2023-11-07
    
    Parameters
    ----------
    nc_var: str
        variable name, as it appears in the name of the file and in the path
    year: int
        year (e.g. 1981)
    kind: str
        'era5' or 'cmip6'
    year_start: int
        first year in the range of the reference period (e.g. 1981)
    year_stop: int
        last year in the range of the reference period (e.g. 2010)
    aggr_operator: str
        which operator apply to hourly values to get daily values (e.g. 'mean','max')
    tres: str
        time resolution of the input files (e.g. 'day','1hr','1hr-cf')
    product: str
        name of the product (e.g. 'reanalysis')
    experiment: str
        name of the experiment according to Freva (e.g. 'era5',...)
    mmb: int
        number of the ensemble member
    mmbpath: str
        full path where to find the original netcdf format according to Levante-Freva
    workpath: str
        full path where the output files are to be stored        
    
    Returns
    -------
        Creates netcdf file in workpath
    
    """

    endmonth = get_endmonth(year) #obtain the last of day of each month for this year
    ## If daily values are not available on DKRZ - Levante, get them
    tres_filename = tres
    
    if (tres == '1hr'):
        dailypath = workpath
        if (nc_var == 'swvl1'):
            tres_filename = '1hr-cf'
        ## If daily values haven't been calculated yet
        if not (os.path.exists(f'{workpath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{year}0101-{year}1231.nc')):
            ## Check if we have all the hourly files for all months of that year
            monthly_files_exist = [os.path.exists(f'{mmbpath}{nc_var}_{tres_filename}_{product}_{experiment}_r{mmb}i1p1_{year}{str(m).zfill(2)}01-{year}{str(m).zfill(2)}{endmonth[m-1]}.nc') for m in range(1,13)]
            if sum(monthly_files_exist)==12:
                
                cdo_daily_aggr(nc_var,year,aggr_operator,tres_filename,product,experiment,mmb,mmbpath,workpath)
            elif not os.path.exists(f'{workpath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{year}0101-{year}1231_cropped.nc'):
                print(kind)
                print(nc_var)
                print(y)
                print('Hourly data not available on DKRZ-Levente. Import data from JUNO/ZEUS')  
                print("Here below the files that couldn't be found")
                print(f'{mmbpath}{nc_var}_{tres}_{product}_{experiment}_r{mmb}i1p1_{year}1201-{year}12{endmonth[11]}.nc')
                print(f'{workpath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{year}0101-{year}1231_cropped.nc')
                sys.exit()

    if tres == 'day':
        dailypath = mmbpath
        #If there's no yearly file with daily data
        if not os.path.exists(f'{mmbpath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{year}0101-{year}1231.nc'):
                print(kind)
                print(nc_var)
                print(year)
                print('Daily data not available on DKRZ-Levente. Import data from JUNO/ZEUS')  
                print("Here below the files that couldn't be found")
                print(f'{mmbpath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{year}0101-{year}1231.nc')
                sys.exit()


    ## Calculate climatology (if not yet)
    if not os.path.exists(f'{workpath}{nc_var}_dailyclim_{product}_{experiment}_r{mmb}i1p1_clim8110.nc'):
        yearly_files_exist = [os.path.exists(f'{dailypath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc') for y in range(year_start,year_stop+1)]
        ## Are there all the needed 30 files?
        if sum(yearly_files_exist)!=30:
            print(f'{nc_var}, some missing files of daily data between {year_start} and {year_stop}, retrieving them')
            for y in range(year_start,year_stop+1):
                ## Is there a yearly file with daily data?
                if not os.path.exists(f'{workpath}{nc_var}_day_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'):
                    cdo_daily_aggr(nc_var,y,aggr_operator,tres_filename,product,experiment,mmb,mmbpath,workpath)
                else:
                    print(f'Daily netcdf for {nc_var} about year {y} already exists')
        cdo_clim(nc_var=nc_var,
                 year_start=year_start,year_stop=year_stop,
                 product=product,experiment=experiment,mmb=mmb,
                 inpath=dailypath,outpath=workpath)    
    cdo_anom(nc_var=nc_var,y=year,
             product=product,experiment=experiment,mmb=mmb,
             inpath=dailypath,outpath=workpath)

def get_endmonth(y):
    
    endmonth = [31,28,31,30,31,30,31,31,30,31,30,31]
    if ((y%4 == 0) & ((not y%100==0) | (y%400==0))):
        endmonth[1] = 29
    return(endmonth)

    
    
def loop_map_grids(drivers, dates_ts, variables, lsm, modelspecs1, varspecs, machine, workdir, modeldir, maskdir, plotdir):
    
    for date_ts in dates_ts:
        y = date_ts.year
        
        #print(y)
        kind = modelspecs1.iloc[0]['kind']
        for var in variables:
            varrow = varspecs.loc[varspecs['var'] == var]
            drivers_sub = drivers.loc[drivers['var'] == var]
            #print(varrow)
            if kind == 'ERA5':
                #datasetnames = ['era5']
                if machine == 'DKRZ':
                    kind_var = 'era5cmor_var'
                else:
                    kind_var = 'era5cds_var'
                kind_aggr = 'era5_aggr'
                #if machine == 'DKRZ':
                #    datapaths = sum([[f"{modeldir1}{varrow[kind_aggr][0]}/atmos/{nc_var}/r{mmb}i1p1/" for mmb in modelrow["members_list"]] for i, modelrow in modelspecs1.iterrows()],[])
                experiment = 'era5'
                product = 'reanalysis'
                tres = varrow['era5_aggr'].values[0]
                year_start = 1981
                year_stop = 2010

            else:
                #[print(modelrow) for i, modelrow in modelspecs.iterrows()]
                datapaths = sum([[f'{modelrow["modelnames"]}-{kind}-r{mmb}' for mmb in modelrow["members_list"]] for i, modelrow in modelspecs1.iterrows()],[])
                kind_var = 'cmip6_var'
                kind_aggr = 'cmip6_aggr'

            #print(kind_var)
            #print(varrow)
            nc_var = varrow[kind_var].values[0]
            aggr_operator = varrow['aggr_operator'].values[0]

            
            models = modelspecs1.loc[modelspecs1.kind == kind,'model_names'].to_list()
            #print(models)
            for mdl in models:
                #print(mdl)
                mdl = models[0]
                modelpath = modelspecs1.loc[modelspecs1['model_names']==mdl,'model_path'].values[0]
                workpath = modelspecs1.loc[modelspecs1['model_names']==mdl,'work_path'].values[0]
                members = modelspecs1.loc[modelspecs1['model_names']==mdl,'members_list'].values[0]
                
                for mmb in members:

                    mmbpath = f"{modelpath}{tres}/{varrow['realm'].values[0]}/{nc_var}/r{mmb}i1p1/"

        
        
        
                    ## If daily anomalies are not available, get them
                    if not os.path.exists(f'{workpath}{nc_var}_dailyanom_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'):
                        get_anom_w_cdo(nc_var, y, kind, year_start, year_stop, aggr_operator, tres, product,experiment,mmb, mmbpath, workpath) 

                    anom_xr = xr.open_dataset(f'{workpath}/{nc_var}_dailyanom_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc')
                    anom_xr = anom_xr.convert_calendar('gregorian') #convert to gregorian calendar
                    #anom_xr['time'] = anom_xr.indexes['time'].normalize() #drop hour from time axis

                    if np.max(anom_xr.lon)>180:
                        anom_xr = rearrange_lon (anom_xr)

                    dims = list(anom_xr.dims)

                    if (('longitude' in dims) | ('latitude' in dims)):
                        infile = f'{workpath}/{nc_var}_dailyanom_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'
                        outfile = f'{workpath}/{nc_var}_dailyanom_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'
                        command = f'cdo chname,longitude,lon {infile} {outfile}'

                        infile = f'{workpath}/{nc_var}_dailyanom_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'
                        outfile  = f'{workpath}/{nc_var}_dailyanom_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc'
                        command = f'cdo chname,latitude,lat {infile} {outfile}'

                        anom_xr = xr.open_dataset(f'{workpath}/{nc_var}_dailyanom_{product}_{experiment}_r{mmb}i1p1_{y}0101-{y}1231.nc')
                        anom_xr = anom_xr.convert_calendar('gregorian')

                        if np.max(anom_xr.lon)>180:
                            anom_xr = clint.rearrange_lon (anom_xr)  
                            
                    if var == 'mslp':
                        anom_xr[nc_var] = anom_xr[nc_var]/100
                        
                    for index, drivers_row in drivers_sub.iterrows():
                        mask_df = pd.read_csv(f"{maskdir}{drivers_row['clmask_path']}{drivers_row['clmask_file']}",index_col=[0])
                        cl_nr = drivers_row['cl_nr']
                        submask = mask_df[mask_df.cluster == cl_nr-1] #python indexing, cluster 1 is nr0 in the mask file...
                        maskedanom = mask_xr_w_df(var, anom_xr, submask, lsm, kind)
                        
                        multimaps_lag (xrdf = maskedanom, targetdate_ts = date_ts, 
                                            drivers_row = drivers_row, kind=kind, machine=machine, plotdir = plotdir,
                                            proj = 'Ortographic', 
                                            vmin='drivers', vmax='drivers')
                        daily_series_w_lags(maskedanom, nc_var, drivers_row, date_ts, 'average', kind, plotdir)
                        daily_series_w_lags(maskedanom, nc_var, drivers_row, date_ts, 'quantiles', kind, plotdir)
                        #daily_series_w_lags(maskedanom, drivers_row, date_ts, 'centroid', kind, plotdir)

                
    
def map_EuroPP (xrdf, targetdate_ts, lag, drivers_row, vmin, vmax):
    
    targetdate_str = dt.strftime(targetdate_ts,'%Y-%m-%d')

    nc_var = drivers_row.era5cds_var
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
    nc_var = drivers_row.era5cds_var
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

    mask = submask_exp.assign(flag=1).set_index(["nodes_lon", "nodes_lat"]).flag.to_xarray().fillna(0).rename({"nodes_lon": 'lon', "nodes_lat": 'lat'})


    
    #mask = mask.reindex(longitude=xrdf['lon'], 
    #                    latitude=xrdf['lat'],method="nearest", tolerance=1e-9, fill_value=0)
    mask1 = mask.interp(lon=xrdf['lon'], 
                        lat=xrdf['lat'], 
                        method="nearest")
    mask1 = mask1.fillna(0)
    sub_xrdf = xrdf.where(mask1)
    
    if var == 'tmax':
        sub_xrdf = apply_land_sea_mask(sub_xrdf, lsm, kind)
    
    return(sub_xrdf)

   
def multimaps_lag (xrdf, targetdate_ts, drivers_row, kind, machine, plotdir,
                   proj='Ortographic', vmin='drivers', vmax='drivers',):
    
    
    
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
        if machine == 'DKRZ':
            nc_var = drivers_row.era5cmor_var
        else:
            nc_var = drivers_row.era5
    elif kind == 'hist':
        nc_var = drivers_row.cmip6_var


    palette = palette_CLINT(var)       
    
    
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
    h_offset = pd.to_datetime(xrdf.isel(time=0)['time'].values).hour #what is the hour of data in the netcdf
    m_offset = pd.to_datetime(xrdf.isel(time=0)['time'].values).minute #what is the hour of data in the netcdf
    

    fig, axs = plt.subplots(int(numfigs_v), int(numfigs_h),
                            subplot_kw={'projection': lonlatproj},
                            figsize=(fig_width,int(numfigs_v*ax_height)),
                            sharey=True,sharex=True)#,
                            #layout="constrained")
    fig.suptitle(f'{var}, {minlag} to {maxlag} days before {targetdate_str}', fontsize = 100)
    axs = axs.flatten()        
        
    

    for f,lag in enumerate(range(minlag,maxlag+1)):

        
        
        plotdate_ts = targetdate_ts - pd.DateOffset(days = lag) + pd.DateOffset(hours = h_offset) + pd.DateOffset(minutes = m_offset)
        #print(plotdate_ts)
        sub1d = xrdf.sel(time=plotdate_ts)
        axs[f].remove()
        geo_axes = plt.subplot(int(numfigs_v), int(numfigs_h), f+1,
                               projection=my_projn)
        
        #print(sub1d.variables)
        
        cs=sub1d[nc_var].plot(ax=geo_axes,transform=ccrs.PlateCarree(),cmap=palette, 
                           vmin = vmin, vmax = vmax,add_colorbar=False)
        
        
        
        
        if (not np.isnan(drivers_row['cl_centroid_lon'])):
            geo_axes.scatter(x=drivers_row['cl_centroid_lon'], y=drivers_row['cl_centroid_lat'],
                        marker='X',color='black',s=900,transform=ccrs.PlateCarree(),alpha=1)
        geo_axes.coastlines()
        
        geo_axes.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
        #geo_axes.text(x=text_plot_lon, y=text_plot_lat, 
        #              s=f'-{lag}d', transform=ccrs.PlateCarree(), fontsize=70)
        geo_axes.text(x=0.875, y=0.9, horizontalalignment='center', verticalalignment='center',
                      s=f'-{lag}d', transform=geo_axes.transAxes, fontsize=70)
        geo_axes.set_extent([xmin,xmax,ymin,ymax], crs=my_projn) # data/projection coordinates 
        #print(f'before this, {f}')
        
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
    tick_locator = mticker.MaxNLocator(nbins=11)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=50)
    

    #plt.savefig(f"{plotdir}CLINT040_maps_{y}{m}{d}case_{var}_Test{drivers_row['exp']}{drivers_row['exp_size']}{cl_name}.png", facecolor='w')

def palette_CLINT (var,n=20):
    #print(f'palette: {var}')
    if var == 'mslp':
        print(var)
        pltt = mpl.colors.LinearSegmentedColormap.from_list('Pressure', (
    # Edit this gradient at https://eltos.github.io/gradient/#Pressure=0:4A0062-20:8E1CC4-40.1:CBADE1-50:FFF2C5-59.9:9BCF9F-80:38955C-100:004B32
    (0.000, (0.290, 0.000, 0.384)),
    (0.200, (0.557, 0.110, 0.769)),
    (0.401, (0.796, 0.678, 0.882)),
    (0.500, (1.000, 0.949, 0.773)),
    (0.599, (0.608, 0.812, 0.624)),
    (0.800, (0.220, 0.584, 0.361)),
    (1.000, (0.000, 0.294, 0.196))), N=n)
        
    elif var == 'sm':
        pltt = mpl.colors.LinearSegmentedColormap.from_list('SoilMoisture', (
    # Edit this gradient at https://eltos.github.io/gradient/#SoilMoisture=0:44000D-24.9:D5572D-50:FFF3C8-75.1:42A36D-100:00343C
    (0.000, (0.267, 0.000, 0.051)),
    (0.249, (0.835, 0.341, 0.176)),
    (0.500, (1.000, 0.953, 0.784)),
    (0.751, (0.259, 0.639, 0.427)),
    (1.000, (0.000, 0.204, 0.235))), N=n)
        return(pltt)
    
    elif var == 'tmax':
        pltt = mpl.colors.LinearSegmentedColormap.from_list('Temperature', (
    # Edit this gradient at https://eltos.github.io/gradient/#Temperature=0:0D0038-15.4:0C4AAA-35:89C0C5-50:FFF2C5-65:FFA88F-84.6:BA2628-100:61001A
    (0.000, (0.051, 0.000, 0.220)),
    (0.154, (0.047, 0.290, 0.667)),
    (0.350, (0.537, 0.753, 0.773)),
    (0.500, (1.000, 0.949, 0.773)),
    (0.650, (1.000, 0.659, 0.561)),
    (0.846, (0.729, 0.149, 0.157)),
    (1.000, (0.380, 0.000, 0.102))), N=n)
        
    else:
        print('This variable does not have a customized palette')
        pltt = plt.cm.PuOr
    return(pltt) 


def rearrange_lon (xrdf):
    ## Rearrange longitude of xarray datasets where longitudes are in the [0,360) range
    ## Longitudes are translated to [-180,180)
    test = xrdf.assign_coords(lon=(((xrdf.lon + 180) % 360) - 180))
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

    
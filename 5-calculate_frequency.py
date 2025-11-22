import os 
import xarray as xr 
import gc
from tqdm import tqdm 
import numpy as np

def calculate_monthly_frequency(model, scenario, whiplash):

    mask_nan = whiplash['dtw'].isel(time=0).isnull()

    def count_unique_events(x):
        vals = x[~np.isnan(x)]   # remove NaN
        vals = vals[vals > 0]    # keep only positive event IDs
        return np.unique(vals).size

    output = {
        'wtd': [],
        'dtw': []
    }
    
    # loop over each year
    for year in np.unique(whiplash['time.year']):
        for var in ['wtd', 'dtw']:
            subset = whiplash['wtd'].sel(time=ds['time'].dt.year == year).groupby('time.month')
            freq_mon = xr.apply_ufunc(
                count_unique_events,
                subset,
                input_core_dims=[["time"]],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[int],
                )
            freq_mon = freq_mon.rename({'month':'time'})
            dates = np.arange(f'{year}-01-01', f'{year+1}-01-01', dtype='datetime64[M]')
            freq_mon['time'] = dates
            output[var].append(freq_mon)

    freq_wtd = xr.concat(output['wtd'], dim='time')
    freq_dtw = xr.concat(output['dtw'], dim='time')

    mask_nan = mask_nan.expand_dims(dim={'time':freq_dtw['time']})
    result = xr.Dataset({
        'freq_wtd': freq_wtd.where(~mask_nan),
        'freq_dtw': freq_dtw.where(~mask_nan)
    })

    result.attrs['model'] = model 
    result.attrs['scenario'] = scenario

    return result.sortby('time').compute()


# main code
model = 'ACCESS-CM2'
ssp   = ['ssp245', 'ssp585']
for i, scenario in enumerate(ssp):
    print(f"\n{i+1}/{len(ssp)} Calculating Frequency of Whiplash Events")
    print(f"= Model      : {model}")
    print(f"= Scenario   : {scenario.upper()}")
    indir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}\Whiplash-{scenario.upper()}"
    whiplash_files = [os.path.join(indir, f) for f in os.listdir(indir)
                    if (os.path.isfile(os.path.join(indir, f))) and (f.endswith('.nc'))]
    result = []
    
    # process per decade file to reduce memory usage
    for i in tqdm(range(len(whiplash_files)), desc='= Processing '):
        file = whiplash_files[i]
        ds = xr.open_mfdataset(whiplash_files[i])
        ds = ds.chunk({'time':-1})
        if 'dayofyear' in ds.coords:
            ds = ds.drop_vars('dayofyear')
        
        freq_mon = calculate_monthly_frequency(model, scenario, ds)
        result.append(freq_mon)

    result = xr.concat(result, dim='time')
    outdir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}\Frequency-{scenario.upper()}"
    os.makedirs(outdir, exist_ok=True)
    period = f'{result['time.year'].values[0]}-{result['time.year'].values[-1]}'
    outfile = os.path.join(outdir, f'whiplash_freq_mon_{model}_{scenario}_{period}.nc')
    result.to_netcdf(
        outfile, 
        engine='h5netcdf',
        encoding = {var: {'zlib': True, 'complevel': 4} for var in result.data_vars.keys()}
    )

    # calculate annual frequency
    mask_valid = result['freq_wtd'].isel(time=0).notnull()

    annual_freq = result.groupby('time.year').sum(skipna=True)
    annual_freq = annual_freq.where(mask_valid.expand_dims(dim={'year':annual_freq['year']}))

    outfile = os.path.join(outdir, f'whiplash_freq_year_{model}_{scenario}_{period}.nc')
    annual_freq.to_netcdf(
        outfile,
        engine='h5netcdf',
        encoding = {var: {'zlib': True, 'complevel': 4} for var in annual_freq.data_vars.keys()}
    )

    annual_freq.close(), result.close()
    gc.collect()

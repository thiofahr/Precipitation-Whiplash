import xarray as xr
import os 
import gc 
from tqdm import tqdm 
import numpy as np



def compute_duration(ds, outdir):

    def event_duration_1d(event_ids):
        event_ids = np.asarray(event_ids)
        out = np.zeros_like(event_ids, dtype=float)
        transition = np.zeros_like(event_ids, dtype=float)

        # keep NaNs
        nan_mask = np.isnan(event_ids)
        out[nan_mask] = np.nan

        # get unique event IDs > 0
        unique_events = np.unique(event_ids[~nan_mask])
        unique_events = unique_events[unique_events > 0]

        for ev in unique_events:
            idx = np.where(event_ids == ev)[0]
            # extreme duration
            dur = len(idx)
            out[idx] = dur
            
            # transition duration
            for i in range(len(idx)):
                if (i+1) == len(idx):
                    break 
                if (idx[i+1] - idx[i] - 1) >=1:
                    trans = np.arange(idx[i]+1, idx[i+1])
                    transition[trans] = len(trans)
                    
        return out, transition

    # separate the data
    wtd = ds['wtd']
    dtw = ds['dtw']

    mask_valid = wtd.notnull()

    duration_wtd, transition_wtd = xr.apply_ufunc(
        event_duration_1d,
        wtd,
        input_core_dims=[["time"]],
        output_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[wtd.dtype, wtd.dtype],
    )
    duration_dtw, transition_dtw = xr.apply_ufunc(
        event_duration_1d,
        dtw,
        input_core_dims=[["time"]],
        output_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[dtw.dtype, wtd.dtype],
    )

    result = xr.Dataset({
        'duration_wtd': duration_wtd,
        'duration_dtw': duration_dtw,
        'transition_wtd': transition_wtd,
        'transition_dtw': transition_dtw
    }).compute()

    # export 
    period = f'{result['time.year'].values[0]}-{result['time.year'].values[-1]}'
    outfile = os.path.join(outdir, f'whiplash_duration_day_{period}.nc')

    result.where(mask_valid).to_netcdf(
        outfile,
        engine='h5netcdf',
        encoding = {var: {'zlib': True, 'complevel': 3} for var in result.data_vars.keys()}
    )

    # annual metric
    annual_avg = result.groupby('time.year').mean(skipna=True)
    annual_avg = annual_avg.where(mask_valid.isel(time=0).expand_dims(dim={'year':annual_avg['year']}))
    outfile = os.path.join(outdir, f'whiplash_duration_mean_year_{period}.nc')
    annual_avg.to_netcdf(
        outfile,
        engine='h5netcdf',
        encoding = {var: {'zlib': True, 'complevel': 3} for var in annual_avg.data_vars.keys()}
    )
    annual_max = result.groupby('time.year').mean(skipna=True)
    annual_max = annual_max.where(mask_valid.isel(time=0).expand_dims(dim={'year':annual_max['year']}))
    outfile = os.path.join(outdir, f'whiplash_duration_max_year_{period}.nc')
    annual_max.to_netcdf(
        outfile,
        engine='h5netcdf',
        encoding = {var: {'zlib': True, 'complevel': 3} for var in annual_max.data_vars.keys()}
    ) 
    return None

# main code
model = 'ACCESS-CM2'
ssp = ['ssp245', 'ssp585']

for i, scenario in enumerate(ssp):
    print(f"\n{i+1}/{len(ssp)} Calculating Duration of Whiplash Events")
    print(f"= Model      : {model}")
    print(f"= Scenario   : {scenario.upper()}")

    # directory for input and output
    main_dir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}"
    indir    = fr"{main_dir}\Whiplash-{scenario.upper()}"
    outdir   = fr"{main_dir}\Duration-{scenario.upper()}"
    os.makedirs(outdir, exist_ok=True)

    whiplash_files = [os.path.join(indir, f) for f in os.listdir(indir)
                    if (os.path.isfile(os.path.join(indir, f))) and (f.endswith('.nc'))]

    for i in tqdm(range(len(whiplash_files)), desc='= Processing '):
        file = whiplash_files[i]
        ds = xr.open_dataset(file)
        compute_duration(ds, outdir)

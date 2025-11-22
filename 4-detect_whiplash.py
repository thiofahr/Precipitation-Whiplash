import xarray as xr
import os 
import gc 
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm

def detect_whiplash_1d_numbered(data1, data2, max_transition_days=30):
    
    # load data
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    n = len(data1)
    if len(data2) != n:
        raise ValueError("data1 and data2 must have same length")
    
    # handle NaN
    nan_mask = np.isnan(data1) | np.isnan(data2)
    if np.all(nan_mask): # ignore the process if all values are NaN
        return np.full(n, np.nan)

    # replace some NaN for processing
    d1 = np.nan_to_num(data1, nan=0)
    d2 = np.nan_to_num(data2, nan=0)

    whiplash = np.zeros(n, dtype=float)
    current_event_id = 1 # number labelling

    # iterate through indices
    i = 0
    while i < n:
        if d1[i] == 1: # find extreme 1
            # find continous ones inside d1: [start1, end1]
            start1 = i
            while start1 - 1 >= 0 and d1[start1 - 1] == 1:
                start1 -= 1
            end1 = i
            while end1 + 1 < n and d1[end1 + 1] == 1:
                end1 += 1
            
            # search for any extreme 2
            look_start = end1 + 1
            look_end = min(end1 + 1 + max_transition_days, n)  # exclusive
            for j in range(look_start, look_end):
                if d2[j] == 1 and whiplash[j] == 0:
                    # found a transition; determine continous extreme 2 around j
                    start2 = j
                    while start2 - 1 >= 0 and d2[start2 - 1] == 1:
                        start2 -= 1
                    end2 = j
                    while end2 + 1 < n and d2[end2 + 1] == 1:
                        end2 += 1

                    # assign event id to full d1 and d2 blocks
                    whiplash[start1:end1 + 1] = current_event_id
                    whiplash[start2:end2 + 1] = current_event_id

                    # find other extreme 1 within the event
                    if data1[end1:start2].any():
                        for i in range(end1, start2):
                            if d1[i] == 1:
                                whiplash[i] = current_event_id

                    current_event_id += 1
                    break

            # advance i to end of this d1 block
            i = end1 + 1
        else:
            i += 1
            
    # restore NaN values
    whiplash[nan_mask] = np.nan
    
    return whiplash

def detect_whiplash(ds, max_transition_days, outdir):
    """
    Detect whiplash events across all grid points

    Conventions:
        data1: extreme wet data
        data2: extreme dry data
    """
    
    # separate the data
    data1 = ds['wet']
    data2 = ds['dry']

    mask_nan = data1.isnull()

    # applying the function
    wet_to_dry = xr.apply_ufunc(
        detect_whiplash_1d_numbered,
        data1,
        data2,
        kwargs={"max_transition_days": max_transition_days},
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,         
        dask="parallelized",    
    )
    dry_to_wet = xr.apply_ufunc(
        detect_whiplash_1d_numbered,
        data2,
        data1,
        kwargs={"max_transition_days": max_transition_days},
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,        
        dask="parallelized",  
    )

    # preserve the NaN values
    dry_to_wet = dry_to_wet.where(~mask_nan)
    wet_to_dry = wet_to_dry.where(~mask_nan)
    result = xr.Dataset({
        'wtd': wet_to_dry,
        'dtw': dry_to_wet
    })

    # export the result
    time = result['time.year'].values 
    period = f'{time[0]}-{time[-1]}'
    outfile = os.path.join(outdir, f'whiplash_{model}_{scenario}_{period}.nc')
    
    result.to_netcdf(
        outfile,
        engine='netcdf4',
        encoding={var: {'zlib': True, 'complevel': 4} for var in result.data_vars.keys()}
    ) 
    return result

def create_output(whiplash, model, scenario, outdir):
    fig, ax = plt.subplots(figsize=(12,4),ncols=2, constrained_layout=True, sharey=True)

    max_val1 = whiplash['wtd'].sum('time', skipna=True).max().values
    max_val2 = whiplash['dtw'].sum('time', skipna=True).max().values

    max_val = np.round(max_val1, -1) if max_val1 > max_val2 else max_val2
    norm = TwoSlopeNorm(vmin=0, vcenter=max_val/2, vmax=max_val)

    for i, var in enumerate(whiplash.data_vars):
        mask_nan = whiplash[var].isel(time=0).isnull()

        c = whiplash[var].sum('time', skipna=True).where(~mask_nan).plot.contourf(ax=ax[i], levels=20, cmap='BrBG', norm=norm, add_colorbar=False, extend='both')
        ax[i].text(100, -10, 'Dry-to-Wet' if var == 'dtw' else 'Wet-to-Dry', ha='center')
        ax[i].set_title('')
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')

    cbar = plt.colorbar(c, ax=ax, orientation='horizontal', fraction=0.1, extend='both')
    cbar.set_label('Number of events')

    period = f'{whiplash['time.year'].values[0]}-{whiplash['time.year'].values[-1]}'
    plt.suptitle(f'Whiplash event for year {period}')
    outfile = os.path.join(outdir, f'whiplash_{model}_{scenario}_{period}.png')
    fig.savefig(outfile, bbox_inches='tight', dpi=200)
    plt.close()

# Main Code
model = "ACCESS-CM2"
ssp = ['ssp245', 'ssp585']
main_dir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}"

for i, scenario in enumerate(ssp):
    print(f"\n{i+1}/{len(ssp)} Detecting Whiplash Events")
    print(f"= Model    : {model}")
    print(f"= Scenario : {scenario.upper()}")

    indir  = rf'{main_dir}\Extreme-{scenario.upper()}'
    
    files = [os.path.join(indir, f) for f in os.listdir(indir)
             if os.path.isfile(os.path.join(indir, f))]

    outdir = rf'{main_dir}\Whiplash-{scenario.upper()}'
    os.makedirs(outdir, exist_ok=True)
    
    # loop over the files
    for j in tqdm(range(len(files)), desc='processing'):
        # open file
        extreme_ds = xr.open_dataset(files[j])
        # identify whiplash
        whiplash = detect_whiplash(extreme_ds, 30, outdir).load()
        create_output(whiplash, model, scenario, outdir)
        # clean up memory before next iteration
        extreme_ds.close(), whiplash.close()
        gc.collect()

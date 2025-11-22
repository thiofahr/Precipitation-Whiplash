import xarray as xr
import os 
from tqdm import tqdm 
import numpy as np

def detect_extremes(model, scenario, data, threshold):
    
    # apply the condition
    wet = data > threshold['wet']
    dry = data < threshold['dry']

    # store NaN values
    mask_nan = np.isnan(data)

    def detect_spell_mask(series):
        # Handle NaN explicitly
        valid = ~np.isnan(series)
        series_bool = np.where(valid, series, False)  # replace NaN with False
        n = len(series)
        mask = np.zeros(n, dtype=bool)
        count = 0
        for i in range(n):
            if series_bool[i]:
                count += 1
            else:
                if count >= 4:
                    mask[i - count:i] = True
                count = 0
        if count >= 4:
            mask[n - count:n] = True
        # return NaN for invalid positions
        mask = mask.astype(float)
        mask[~valid] = np.nan
        return mask

    # Apply function
    result_wet = xr.apply_ufunc(
        detect_spell_mask,
        wet.where(~mask_nan),  
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[bool],
    ).compute().where(~mask_nan)

    result_dry = xr.apply_ufunc(
        detect_spell_mask,
        dry.where(~mask_nan),  
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[bool],
    ).compute().where(~mask_nan)

    # Merge into one dataset
    result = xr.Dataset({
        'wet': result_wet,
        'dry': result_dry
    })

    # add attributes
    result['wet'].attrs['description'] = f"consecutive more than 3 days of wet days"
    result['dry'].attrs['description'] = f"consecutive more than 3 days of dry days"
    result.attrs['model'] = model 
    result.attrs['scenario'] = scenario
    
    # export to netcdf
    periods = f'{result['time.year'].values[0]}-{result['time.year'].values[-1]}'

    outfile = os.path.join(outdir, f'extreme_{model}_{scenario}_{periods}.nc')

    result.to_netcdf(
        outfile,
        engine = 'h5netcdf',
        encoding = {var: {'zlib':True, 'complevel':4} for var in result.data_vars.keys()}
    )

    return result

# main code
model = "ACCESS-CM2"
ssp = ['ssp245', 'ssp585']
indir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}"

for i, scenario in enumerate(ssp):
    print(f"\n{i+1}/{len(ssp)} Detecting Extreme Wet and Dry Events")
    print(f"= Model      : {model}")
    print(f"= Scenario   : {scenario.upper()}\n")

    outdir = rf'{indir}\Extremes-{scenario.upper()}'
    os.makedirs(outdir, exist_ok=True)

    threshold_file = os.path.join(indir, f"threshold_{model}_{scenario}_1991-2020.nc")
    threshold = xr.open_dataset(threshold_file)
    
    precip_dir = rf"{indir}\Anom-{scenario.upper()}"
    precip_files = [os.path.join(precip_dir, f) for f in os.listdir(precip_dir)
                    if os.path.isfile(os.path.join(precip_dir, f))]
    
    for j in tqdm(range(len(precip_files)), desc='= Processing '):
        # open file
        precip_ds = xr.open_dataset(precip_files[j])
        pr30 = precip_ds['pr']

        # detect extremes
        extreme_spells = detect_extremes(model, scenario, pr30, threshold)

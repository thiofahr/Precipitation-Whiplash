import xarray as xr
import os 
from tqdm import tqdm 
import numpy as np

def compute_intensity(model, scenario, pr_anom, whiplash, dry, wet, outdir):

    def compute_intensity_1d(metric, pr_anom, whiplash, dry, wet):
        """
        Compute intensity for each whiplash event.
        Intensity = | max(wet anomaly) - min(dry anomaly) |
        """
        pr_anom = np.asarray(pr_anom)
        whiplash = np.asarray(whiplash)
        dry = np.asarray(dry, dtype=bool)
        wet = np.asarray(wet, dtype=bool)

        unique_events = np.unique(whiplash)
        unique_events = unique_events[unique_events != 0]  # exclude non-events

        intensity = np.full_like(pr_anom, np.nan, dtype=float)

        for ev in unique_events:
            idx = np.where(whiplash == ev)[0]

            # Mask dry and wet inside the event
            dry_idx = idx[dry[idx].astype(bool)]
            wet_idx = idx[wet[idx].astype(bool)]

            if len(dry_idx) == 0 or len(wet_idx) == 0:
                # Event missing dry or wet part → skip
                continue
            if metric.lower() == 'intensity':
                dry_min = pr_anom[dry_idx].min()
                wet_max = pr_anom[wet_idx].max()
            elif metric.lower() == 'severity':
            # for severity
                dry_min = pr_anom[dry_idx].sum()
                wet_max = pr_anom[wet_idx].sum()
            else:
                raise ValueError('Wrong Metric!')
            
            event_intensity = abs(wet_max - dry_min)

            # Assign same intensity to all timestamps of the event
            intensity[idx[0]] = event_intensity

        return intensity
    
    wtd = whiplash['wtd']
    dtw = whiplash['dtw']

    intensity_wtd = xr.apply_ufunc(
        compute_intensity_1d,
        'intensity', pr_anom, wtd, dry, wet,
        input_core_dims=[[], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[wtd.dtype],
    )

    intensity_dtw = xr.apply_ufunc(
        compute_intensity_1d,
        'intensity', pr_anom, dtw, dry, wet,
        input_core_dims=[[], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[dtw.dtype],
    )

    severity_wtd = xr.apply_ufunc(
        compute_intensity_1d,
        'severity', pr_anom, wtd, dry, wet,
        input_core_dims=[[], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[wtd.dtype],
    )

    severity_dtw = xr.apply_ufunc(
        compute_intensity_1d,
        'severity', pr_anom, dtw, dry, wet,
        input_core_dims=[[], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[dtw.dtype],
    )

    result = xr.Dataset({
        'intensity_wtd' : intensity_wtd,
        'intensity_dtw' : intensity_dtw,
        'severity_wtd'  : severity_wtd,
        'severity_dtw'  : severity_dtw
    }).compute()

    # export 
    result.attrs['model'] = model 
    result.attrs['scenario'] = scenario 
    
    # daily
    mask_valid = pr_anom.notnull()
    period = f'{result['time.year'].values[0]}-{result['time.year'].values[-1]}'
    outfile = os.path.join(outdir, f'whiplash_intensity_day_{period}.nc')

    result.where(mask_valid).to_netcdf(
        outfile,
        engine='h5netcdf',
        encoding = {var: {'zlib': True, 'complevel': 4} for var in result.data_vars.keys()}
    )

    annual_metric = result.groupby('time.year').sum(skipna=True)
    annual_metric = annual_metric.where(mask_valid.isel(time=0).expand_dims(dim={'year': annual_metric['year']}))
    
    outfile = os.path.join(outdir, f'whiplash_intensity_year_{period}.nc')
    annual_metric.to_netcdf(
        outfile,
        engine='h5netcdf',
        encoding = {var: {'zlib': True, 'complevel': 4} for var in result.data_vars.keys()}
    )

model = 'ACCESS-CM2'
years = np.arange(1991, 2101, 10)
ssp = ['ssp245', 'ssp585']

for i, scenario in enumerate(ssp):
    print(f"\n{i+1}/{len(ssp)} Calculating Intensity and Severity of Whiplash Events")
    print(f"= Model      : {model}")
    print(f"= Scenario   : {scenario.upper()}")

    # directory of input and output
    indir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}\Whiplash-{scenario.upper()}"
    outdir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}\Intensity-{scenario.upper()}"
    os.makedirs(outdir, exist_ok=True)
    
    whiplash_files = [os.path.join(indir, f) for f in os.listdir(indir)
                    if (os.path.isfile(os.path.join(indir, f))) and (f.endswith('.nc'))]
    
    extreme_dir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}\Extreme-{scenario.upper()}"
    extreme_files = [
        os.path.join(extreme_dir, f) for f in os.listdir(extreme_dir)
        if (os.path.isfile(os.path.join(extreme_dir, f))) and (f.endswith('.nc'))
    ]

    pr_dir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}\Anom-{scenario.upper()}"
    pr_files = [
        os.path.join(pr_dir, f) for f in os.listdir(pr_dir)
        if (os.path.isfile(os.path.join(pr_dir, f))) and (f.endswith('.nc'))
    ]
    for i in tqdm(range(len(whiplash_files)), desc='= Processing '):
        whiplash_file = whiplash_files[i]
        extreme_file = extreme_files[i]
        pr_file = pr_files[i]


        if (str(years[i]) in pr_file) & (str(years[i]) in extreme_file) & (str(years[i]) in whiplash_file):
            pr_ds = xr.open_dataset(pr_file)
            whiplash_ds = xr.open_dataset(whiplash_file)
            extreme_ds = xr.open_dataset(extreme_file)

            intensity = compute_intensity(
                model, scenario, 
                pr_ds['pr'], whiplash_ds, 
                extreme_ds['dry'], extreme_ds['wet'], 
                outdir)
        else:
            raise ValueError('Files Mismatch')

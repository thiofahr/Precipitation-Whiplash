import xarray as xr
import os 
import gc
import numpy as np
from tqdm import tqdm 

def calculate_anomalies(model, scenario, precip):
    """
    Calculate the standardized anomalies
    from 30-day rolling sum of precipitation
    """
    print('\n=======================================')
    print('== Calculating the standardized')
    print('== anomalies of precipitation')
    print(f'== Model    : {model}')
    print(f'== Scenario : {scenario}')
    print('=======================================')
  
    def detrend_1d(y):
        """
        Apply detrending for a given data
        """
        y = np.asarray(y, dtype=float)

        # keep only finite values
        mask = np.isfinite(y)
        yy = y[mask]

        # skip detrend when number of points below 3
        if yy.size < 3:
            return y

        # skip detrend if values are constant
        if np.nanstd(yy) < 1e-12:
            out = y.copy()
            out[mask] = yy - np.nanmean(yy)
            return out

        # x-axis for regression
        x = np.arange(yy.size, dtype=float)

        # center x and y to improve numerical stability
        dx = x - x.mean()
        dy = yy - yy.mean()

        # ensure var(x) > 0
        var = np.dot(dx, dx)
        if var < 1e-12:
            out = y.copy()
            out[mask] = yy - yy.mean()
            return out

        # compute slope
        m = np.dot(dx, dy) / var
        b = yy.mean() - m * x.mean()

        # remove trend
        trend = m * x + b
        out = y.copy()
        out[mask] = yy - trend

        return out

    def remove_linear_trend(da):
        """
        Apply detrending and standardization to a DataArray.
        """
        detrended = xr.apply_ufunc(
            detrend_1d,
            da,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return detrended

    def remove_annual_cycle(da):
        """
        Remove annual cycle from the standardized anomalies
        """
        clim = da.groupby("time.dayofyear").mean("time")
        da_anom = da.groupby("time.dayofyear") - clim
        return da_anom

    # remove linear trend
    print('1 - removing linear trend')
    pr_detrended = remove_linear_trend(precip.chunk({'time':-1}))

    precip_ds.close()
    gc.collect()

    # 30-days moving cumulative precipitation
    print('2 - calculating cumulative precipitation')
    def rolling_sum(da, window=30):
        # rolling sum to handle NaN value
        da_filled = da.fillna(0)
        roll = da_filled.rolling(time=window, min_periods=1).sum()
        # mask where original has all-NaN windows
        mask = da.rolling(time=window).count() == 0
        return roll.where(~mask)

    pr30 = rolling_sum(pr_detrended) #pr_detrended.rolling(time=30).sum()

    # remove annual cycle
    print('3 - removing annual cycle')
    pr30 = remove_annual_cycle(pr30)

    print('4 - calculate the standard anomalies')
    def standardize(da, dim="time"):
        std = da.std(dim)
        mean = da.mean(dim)

        # avoid divide-by-zero
        std_safe = std.where(std > 0, np.NaN)

        return (da - mean) / std_safe

    pr30_anom = standardize(pr30)

    print('5 - exporting to netcdf')
    batch_year = np.arange(1991, 2101, 10)
    outdir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}\Anom-{scenario.upper()}"
    os.makedirs(outdir, exist_ok=True)

    # exporting data per decade
    for i in tqdm(range(len(batch_year)), desc='processing'):
        y = batch_year[i]
        decade = slice(f'{y}-01-01', f'{y+9}-12-31')
        outfile = os.path.join(outdir, f'pr_day_anom_{model}_{scenario}_r1i1p1f1_gn_{y}-{y+9}_v2.0.nc')
        pr30_anom.sel(time=decade).compute().to_netcdf(
            outfile,
            engine='h5netcdf',
            encoding={'pr': {'zlib':True, 'complevel':4}})

# Compute the anomalies
model = "BCC-CSM2-MR"

for scenario in ['ssp245', 'ssp585']:
    indir = fr"C:\Users\binta\Research\Projection Data\NEX\Processed\{model}"

    precip_files = [
        os.path.join(indir, f"pr_day_{model}_historical_r1i1p1f1_gn_1981-2014_v2.0.nc"),
        os.path.join(indir, f"pr_day_{model}_{scenario}_r1i1p1f1_gn_2015-2100_v2.0.nc")
    ]

    precip_ds = xr.open_mfdataset(precip_files).sel(time=slice('1991-01-01', '2100-12-31'))
    precip = precip_ds['pr']*86400

    anom = calculate_anomalies(model, scenario, precip)

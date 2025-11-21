import xarray as xr
import os 
import matplotlib.pyplot as plt 
from tqdm import tqdm

def calculate_threshold(model, scenario, pr, baseline=slice('1991', '2020'), upper=0.9, lower=0.1):
    """
    Compute the threshold for wet and dry conditions
    """
    # ensure the period is correct
    data = pr.sel(time=baseline)

    wet_thr = data.quantile(upper, dim='time').rename({'quantile':'upper'})
    dry_thr = data.quantile(lower, dim='time').rename({'quantile':'lower'})

    result = xr.Dataset({
        'wet': wet_thr,
        'dry': dry_thr
    }).compute()

    # Export the result
    # export the data
    outdir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}"
    os.makedirs(outdir, exist_ok=True)

    outfile = os.path.join(outdir, f'threshold_{model}_{scenario}_1991-2020.nc')
    result.to_netcdf(
        outfile,
        engine='h5netcdf',
        encoding={var: {'zlib':True, 'complevel':3} for var in result.data_vars.keys()})

    # create a threshold map 
    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, dpi=200, constrained_layout=True)

    result['wet'].plot.contourf(cmap='Reds', ax=ax[0])
    result['dry'].plot.contourf(cmap='Blues_r', ax=ax[1])
    ax[0].set_title(f'{int(result['wet'].upper.values*100)}th threshold')
    ax[1].set_title(f'{int(result['dry'].lower.values*100)}th threshold')

    ax[1].set_xlabel(f'{model} - {scenario}')

    for x in ax:
        x.set_xlabel('')
        x.set_ylabel('')
    outfile = os.path.join(outdir, f'threshold_{model}_{scenario}_1991-2020.png')
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    
    return result

model = "ACCESS-CM2"
ssp = ['ssp245', 'ssp585']

for i in tqdm(range(len(ssp)), desc='processing'):
    
    scenario = ssp[i]

    indir = fr"C:\Users\binta\Research\Whiplash\DailyResult\{model}\Anom-{scenario.upper()}"
    files = [os.path.join(indir, f) for f in os.listdir(indir)
            if os.path.isfile(os.path.join(indir, f))]

    subset = [f for f in files if ('1991' in f) or ('2001' in f) or ('2011' in f)]

    precip_ds = xr.open_mfdataset(subset)

    # check whether the period is true 
    if (precip_ds['time'][0].dt.year != 1991) & (precip_ds['time'][1].dt.year != 2020):
        raise ValueError('Wrong period of data')

    pr = precip_ds['pr']

    threshold = calculate_threshold(model, scenario, pr)

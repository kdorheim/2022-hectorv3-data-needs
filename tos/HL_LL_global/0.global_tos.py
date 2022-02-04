import fsspec
import intake  # must be v 0.6.2
import xarray as xr
import pandas as pd
import os as os
import numpy as np

# An example of how to get the weighted area
# https://nordicesmhub.github.io/NEGI-Abisko-2019/training/Example_model_global_arctic_average.html

## The url path that contains to the pangeo archive table of contents.
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
catalog = intake.open_esm_datastore(url)

# Set the parameters of our serach, we need the tos and the ocean cell area.
expts = ['historical']
cmip_vars = ["tos"]
query = dict(
    experiment_id=expts,
    variable_id=cmip_vars,
    grid_label="gn"
)

catalog = catalog.search(require_all_on=["source_id"], **query)
catalog = catalog.df.copy().reset_index(drop=True)
catalog = catalog.loc[catalog['member_id'].str.contains('p1')].copy().reset_index(drop=True)


def get_ds_meta(ds):
    """ Get the meta data information from the xarray data set.

    :param ds:  xarray dataset of CMIP data.

    :return:    pandas dataset of MIP information.
    """
    v = ds.variable_id

    data = [{'variable': v,
             'experiment': ds.experiment_id,
             'units': ds[v].attrs['units'],
             'ensemble': ds.attrs["variant_label"],
             'model': ds.source_id}]
    df = pd.DataFrame(data)

    return df


def get_lat_name(ds):
    """ Get the name for the latitude values (could be either lat or latitude).

    :param ds:    xarray dataset of CMIP data.

    :return:    the string name for the latitude variable.
    """
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")


def selstr(a, start, stop):
    """ Select elements of a string from an array.

    :param a:   array containing a string.
    :param start: int referring to the first character index to select.
    :param stop: int referring to the last character index to select.

    :return:    array of strings
    """
    if type(a) not in [str]:
        raise TypeError(f"a: must be a single string")

    out = []
    for i in range(start, stop):
        out.append(a[i])
    out = "".join(out)
    return out


def combine_df(df1, df2):
    """ Join the data frames together.

    :param df1:   pandas data frame 1.
    :param df2:   pandas data frame 2.

    :return:    a single pandas data frame.
    """

    # Combine the two data frames with one another.
    df1["j"] = 1
    df2["j"] = 1
    out = df1.merge(df2)
    out = out.drop(columns="j")

    return out


def global_mean(path):
    """ Get the weighted global mean for a variable.

    :param path: file path to the netcdf file.

    :return:    xarray dataset of the weighted global mean.
    """
    # Import the data file
    ds = xr.open_zarr(fsspec.get_mapper(path), consolidated=True)

    # Extract the meta data
    meta = get_ds_meta(ds)

    # Get the weighted mean global temperature based on the latitude.
    lat = ds[get_lat_name(ds)]
    weight = np.cos(np.deg2rad(lat))
    weight /= weight.mean()
    other_dims = set(ds.dims) - {'time'}
    rslt = (ds * weight).mean(other_dims).coarsen(time=12).mean()

    # Extract time information.
    t = rslt["time"].dt.strftime("%Y%m%d").values
    year = list(map(lambda x: selstr(x, start=0, stop=4), t))

    # Format into a data frame.
    val = rslt["tos"].values
    d = {'year': year, 'value': val}
    df = pd.DataFrame(data=d)
    out = combine_df(meta, df)
    out["area"] = "global"

    return out

# Set up the output directory and process the files.
outdir = './global/'

if not os.path.exists(outdir):
    os.mkdir(outdir)

# Process the files
for file in catalog["zstore"]:
    try:
        ofile = outdir + file.replace("/", "_") + '.csv'
        ofile = ofile.replace("gs:__cmip6_", "")
        out = global_mean(file)
        out.to_csv(ofile, index=False)
    except:
        print("problem with " + file)

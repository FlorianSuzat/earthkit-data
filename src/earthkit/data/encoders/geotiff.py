# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

from earthkit.data import from_object

from . import EncodedData, Encoder, FilePathEncodedData

LOG = logging.getLogger(__name__)

# Keyword arguments consumed by earthkit itself, never passed to GDAL
_NON_GDAL_KWARGS = ("template", "driver", "hints")


def _crs_for(data):
    """CRS to write: a clean EPSG:4326 for plain lat/lon grids, otherwise the
    projection of the data (rioxarray would record a WGS84 lat/lon CRS built
    from cartopy as an "unknown" datum).
    """
    import pyproj

    crs = pyproj.CRS.from_user_input(data.geography.projection().to_proj_string())
    if crs.equals(pyproj.CRS("EPSG:4326"), ignore_axis_order=True):
        return "EPSG:4326"
    return crs


def _band_label(value):
    """Text written as band description for one value of the band coordinate.

    A timedelta becomes "3h", a tuple (from stacked dimensions) becomes
    "t 3h 850", anything else is its string.
    """
    import numpy as np

    if isinstance(value, tuple):
        return " ".join(_band_label(v) for v in value)
    if hasattr(value, "to_timedelta64"):  # pandas Timedelta, from a stacked MultiIndex
        value = value.to_timedelta64()
    if isinstance(value, np.timedelta64):
        hours = value / np.timedelta64(1, "h")
        return f"{int(hours)}h" if hours == int(hours) else str(value)
    return str(value)


def _to_raster_array(ds, y_dim="latitude", x_dim="longitude"):
    """Shape the dataset for rioxarray.

    - several 2D variables: one band per variable, rioxarray handles the Dataset
    - one variable with only the spatial dims: a single band
    - one variable with extra dims (step, level, ...): the extra dims are
      merged into one band dimension and labelled (e.g. "0h", "3h")
    - several variables with extra dims: the variables become one more
      dimension, then merged like above (labels "t 0h", "z 0h", ...)
    """
    spatial = (y_dim, x_dim)
    names = list(ds.data_vars)
    if len(names) != 1:
        if all(set(ds[n].dims) <= set(spatial) for n in names):
            ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
            return ds
        da = ds.to_dataarray(dim="variable")
    else:
        da = ds[names[0]]

    extra = [d for d in da.dims if d not in spatial]
    if len(extra) > 1:
        da = da.stack(band=extra)
        extra = ["band"]
    if extra:
        band = extra[0]
        da = da.transpose(band, y_dim, x_dim)
        da.attrs["long_name"] = [_band_label(v) for v in da[band].values]
    da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
    return da


class GeoTIFFEncodedData(EncodedData):
    _GDAL_DRIVER = "GTiff"

    def __init__(self, ds, **kwargs):
        self.ds = ds
        # GDAL creation options given at encode time (dtype, nodata, compress, ...)
        self.kwargs = kwargs

    def to_bytes(self, **kwargs):
        import rasterio.io

        m = rasterio.io.MemoryFile()
        self._to_raster(m, **kwargs)
        return m.getbuffer()

    def to_file(self, f, **kwargs):
        self._to_raster(f, **kwargs)

    def _to_raster(self, dst, **kwargs):
        options = self.kwargs | {"driver": self._GDAL_DRIVER}
        options.update(kwargs)
        self.ds.rio.to_raster(dst, **options)

    def get(self, key, default=None):
        raise NotImplementedError


class CloudOptimizedGeoTIFFEncodedData(GeoTIFFEncodedData):
    """Same as GeoTIFF but written by the GDAL COG driver: tiled, with
    overviews, in the Cloud Optimized GeoTIFF layout.
    """

    _GDAL_DRIVER = "COG"


class GeoTIFFEncoder(Encoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, data, target=None, **kwargs):
        if data is None:
            raise ValueError("No data to encode")

        path_allowed = target is not None and target._name == "file"
        hints = {"path_allowed": path_allowed}

        from earthkit.data.data.wrappers import from_object

        data = from_object(data)
        return data._encode(self, hints=hints, target=target, **kwargs)

    def _encode(self, data, *, target=None, **kwargs):
        return self._encode_xarray(data, target=target, **kwargs)

    def _encode_field(self, field, *, target=None, **kwargs):
        return self._encode(field, target=target, **kwargs)

    def _encode_fieldlist(self, fieldlist, *, target=None, **kwargs):
        return self._encode(fieldlist, target=target, **kwargs)

    def _encode_xarray(self, data, *, target=None, **kwargs):
        import rioxarray  # noqa: F401  registers the .rio accessor
        import xarray as xr

        ds = from_object(data).to_xarray()
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(name=ds.name or "band")
        if ds.rio.crs is None:
            if not hasattr(data, "geography"):
                raise ValueError("xarray input without a CRS: set it with ds.rio.write_crs() first")
            ds.rio.write_crs(_crs_for(data), inplace=True)
        for var in ds.data_vars:
            if "_earthkit" in ds[var].attrs:
                del ds[var].attrs["_earthkit"]

        raster = _to_raster_array(ds)

        cog = kwargs.get("driver") == "COG"
        options = {k: v for k, v in kwargs.items() if k not in _NON_GDAL_KWARGS}
        if "nodata" in options:
            # rioxarray takes the nodata value from the array, not from the GDAL options
            nodata = options.pop("nodata")
            arrays = raster.data_vars.values() if isinstance(raster, xr.Dataset) else (raster,)
            for da in arrays:
                da.rio.write_nodata(nodata, inplace=True)
        klass = CloudOptimizedGeoTIFFEncodedData if cog else GeoTIFFEncodedData
        return klass(raster, **options)

    def _encode_featurelist(self, data, *, target=None, **kwargs):
        raise NotImplementedError

    def _encode_path(self, path_info=None, *, target=None, **kwargs):
        # Write file as is if target is file and path is provided.
        if (
            path_info is not None
            and path_info.path is not None
            and path_info.default_encoder == "geotiff"
            and target is not None
            and target._name == "file"
        ):
            return FilePathEncodedData(path_info.path, binary=path_info.binary)
        else:
            return None


encoder = GeoTIFFEncoder

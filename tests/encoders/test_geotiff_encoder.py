#!/usr/bin/env python3

# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.data import from_source
from earthkit.data.core.temporary import temp_file
from earthkit.data.encoders.geotiff import _band_label
from earthkit.data.utils.testing import NO_RIOXARRAY, earthkit_test_data_file

NO_RASTERIO = NO_RIOXARRAY  # rasterio is a dependency of rioxarray


def _synthetic(shape, dims, var_names=("t",)):
    """Small xarray Dataset with named dims and coordinates."""
    import xarray as xr

    coords = {}
    for d, n in zip(dims, shape):
        if d == "step":
            coords[d] = np.arange(n) * np.timedelta64(3, "h")
        elif d in ("latitude", "longitude"):
            coords[d] = np.linspace(0, 1, n)
        else:
            coords[d] = np.arange(n)
    data = {name: (dims, np.random.rand(*shape)) for name in var_names}
    return xr.Dataset(data, coords=coords)


# ---------------------------------------------------------------------------
# band labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.timedelta64(3, "h"), "3h"),
        (np.timedelta64(0, "s"), "0h"),
        (np.timedelta64(90, "m"), str(np.timedelta64(90, "m"))),
        (850, "850"),
        ("sfc", "sfc"),
        (("t", np.timedelta64(6, "h")), "t 6h"),
    ],
)
def test_geotiff_band_label(value, expected):
    assert _band_label(value) == expected


# ---------------------------------------------------------------------------
# shaping the xarray for rioxarray
# ---------------------------------------------------------------------------


@pytest.mark.skipif(NO_RIOXARRAY, reason="rioxarray not available")
def test_geotiff_raster_array_single_2d():
    import xarray as xr

    from earthkit.data.encoders.geotiff import _to_raster_array

    ds = _synthetic((7, 12), ("latitude", "longitude"))
    r = _to_raster_array(ds)
    assert isinstance(r, xr.DataArray)
    assert r.dims == ("latitude", "longitude")
    assert "long_name" not in r.attrs
    assert r.rio.x_dim == "longitude"
    assert r.rio.y_dim == "latitude"


@pytest.mark.skipif(NO_RIOXARRAY, reason="rioxarray not available")
def test_geotiff_raster_array_single_with_steps():
    from earthkit.data.encoders.geotiff import _to_raster_array

    ds = _synthetic((7, 12, 4), ("latitude", "longitude", "step"))
    r = _to_raster_array(ds)
    # the band dimension must come first whatever the input order
    assert r.dims == ("step", "latitude", "longitude")
    assert r.attrs["long_name"] == ["0h", "3h", "6h", "9h"]
    assert np.allclose(r.values[2], ds["t"].values[:, :, 2])


@pytest.mark.skipif(NO_RIOXARRAY, reason="rioxarray not available")
def test_geotiff_raster_array_single_with_two_extra_dims():
    from earthkit.data.encoders.geotiff import _to_raster_array

    ds = _synthetic((2, 3, 7, 12), ("step", "level", "latitude", "longitude"))
    r = _to_raster_array(ds)
    assert r.dims == ("band", "latitude", "longitude")
    assert r.sizes["band"] == 6
    assert r.attrs["long_name"] == ["0h 0", "0h 1", "0h 2", "3h 0", "3h 1", "3h 2"]


@pytest.mark.skipif(NO_RIOXARRAY, reason="rioxarray not available")
def test_geotiff_raster_array_several_2d_variables():
    import xarray as xr

    from earthkit.data.encoders.geotiff import _to_raster_array

    ds = _synthetic((7, 12), ("latitude", "longitude"), var_names=("t", "z"))
    r = _to_raster_array(ds)
    # rioxarray writes one band per variable from a Dataset
    assert isinstance(r, xr.Dataset)
    assert list(r.data_vars) == ["t", "z"]


@pytest.mark.skipif(NO_RIOXARRAY, reason="rioxarray not available")
def test_geotiff_raster_array_several_variables_with_steps():
    from earthkit.data.encoders.geotiff import _to_raster_array

    ds = _synthetic((2, 7, 12), ("step", "latitude", "longitude"), var_names=("t", "z"))
    r = _to_raster_array(ds)
    assert r.dims == ("band", "latitude", "longitude")
    assert r.attrs["long_name"] == ["t 0h", "t 3h", "z 0h", "z 3h"]
    assert np.allclose(r.values[3], ds["z"].values[1])


# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------


@pytest.mark.skipif(NO_RIOXARRAY, reason="rioxarray not available")
@pytest.mark.with_proj
def test_geotiff_crs_regular_ll_is_epsg_4326():
    from earthkit.data.encoders.geotiff import _crs_for

    ds = from_source("file", earthkit_test_data_file("test_single.grib")).to_fieldlist()
    assert _crs_for(ds) == "EPSG:4326"


# ---------------------------------------------------------------------------
# end to end: GRIB -> GeoTIFF / COG
# ---------------------------------------------------------------------------


def _open(path):
    import rasterio

    return rasterio.open(path)


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_single_field_plain():
    ds = from_source("file", earthkit_test_data_file("test_single.grib")).to_fieldlist()

    with temp_file(extension=".tif") as path:
        ds[0].to_target("file", path, encoder="geotiff")
        with _open(path) as src:
            assert src.driver == "GTiff"
            assert src.count == 1
            assert src.width == 12 and src.height == 7
            assert src.crs.to_epsg() == 4326
            assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG"
            assert np.allclose(src.read(1), ds[0].to_numpy())


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_single_field_cog():
    ds = from_source("file", earthkit_test_data_file("test_single.grib")).to_fieldlist()

    with temp_file(extension=".tif") as path:
        ds[0].to_target("file", path, driver="COG", dtype="float32", nodata=np.nan, compress="DEFLATE", predictor=3)
        with _open(path) as src:
            assert src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
            assert src.is_tiled
            assert src.count == 1
            assert src.dtypes[0] == "float32"
            assert np.isnan(src.nodata)
            assert src.compression.name.lower() == "deflate"
            assert src.crs.to_epsg() == 4326
            assert np.allclose(src.read(1), ds[0].to_numpy(), atol=1e-3)


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_cog_half_pixel_transform():
    """GRIB gives point centres, GeoTIFF gives pixel edges: half a pixel apart."""
    ds = from_source("file", earthkit_test_data_file("test_single.grib")).to_fieldlist()
    lat = ds[0].geography.latitudes()
    lon = ds[0].geography.longitudes()
    dlat = abs(lat[1, 0] - lat[0, 0])
    dlon = abs(lon[0, 1] - lon[0, 0])

    with temp_file(extension=".tif") as path:
        ds[0].to_target("file", path, driver="COG")
        with _open(path) as src:
            assert np.isclose(src.res[0], dlon) and np.isclose(src.res[1], dlat)
            assert np.isclose(src.bounds.top, lat.max() + dlat / 2)
            assert np.isclose(src.bounds.left, lon.min() - dlon / 2)
            assert src.tags()["AREA_OR_POINT"] == "Area"


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_series_one_band_per_step():
    """A fieldlist of one parameter over several steps -> one band per step."""
    ds = from_source("file", earthkit_test_data_file("t_time_series.grib")).to_fieldlist()
    ds = ds.sel({"parameter.variable": "t"})
    # bands follow the xarray coordinate, i.e. ascending steps whatever the file order
    steps = sorted(ds.metadata("step"))
    assert len(steps) > 1

    with temp_file(extension=".tif") as path:
        ds.to_target("file", path, driver="COG", dtype="float32")
        with _open(path) as src:
            assert src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
            assert src.count == len(steps)
            assert list(src.descriptions) == [f"{s}h" for s in steps]
            for i, step in enumerate(steps):
                f = ds.sel({"time.step": step})[0]
                assert np.allclose(src.read(i + 1), f.to_numpy(), atol=1e-3)


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_series_one_band_per_level():
    ds = from_source("file", earthkit_test_data_file("t_pl.grib")).to_fieldlist()
    # bands follow the xarray coordinate, i.e. ascending levels whatever the file order
    levels = sorted(ds.metadata("level"))

    with temp_file(extension=".tif") as path:
        ds.to_target("file", path, driver="COG")
        with _open(path) as src:
            assert src.count == len(levels)
            assert list(src.descriptions) == [str(lev) for lev in levels]
            for i, level in enumerate(levels):
                f = ds.sel({"vertical.level": level})[0]
                assert np.allclose(src.read(i + 1), f.to_numpy())


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_two_params_with_steps():
    ds = from_source("file", earthkit_test_data_file("t_time_series.grib")).to_fieldlist()

    with temp_file(extension=".tif") as path:
        ds.to_target("file", path, driver="COG")
        with _open(path) as src:
            assert src.count == len(ds)
            assert src.descriptions[0].startswith("t ")
            assert any(d.startswith("z ") for d in src.descriptions)


@pytest.mark.skipif(NO_RIOXARRAY, reason="rioxarray not available")
@pytest.mark.with_proj
def test_geotiff_encoder_cog_round_trip_earthkit():
    """A COG written by earthkit is read back by earthkit as a fieldlist."""
    ds = from_source("file", earthkit_test_data_file("t_pl.grib")).to_fieldlist()

    with temp_file(extension=".tif") as path:
        ds.to_target("file", path, driver="COG")
        back = from_source("file", path).to_fieldlist()
        assert len(back) == len(ds)
        # band 1 is the lowest level value (ascending coordinate), see the level test
        lowest = ds.sel({"vertical.level": min(ds.metadata("level"))})[0]
        assert np.allclose(back[0].to_numpy(), lowest.to_numpy())


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_template_kwarg_not_passed_to_gdal():
    """Target._encode always passes template=None: it must not reach GDAL."""
    ds = from_source("file", earthkit_test_data_file("test_single.grib")).to_fieldlist()

    with temp_file(extension=".tif") as path:
        ds[0].to_target("file", path, encoder="geotiff", template=None)
        with _open(path) as src:
            assert src.count == 1


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_xarray_dataarray_input():
    """A DataArray with a CRS (e.g. a cropped, converted field) is written as is."""
    from earthkit.data import to_target

    ds = from_source("file", earthkit_test_data_file("test_single.grib")).to_fieldlist()
    da = (ds.to_xarray()["2t"] - 273.15).isel(latitude=slice(1, 5), longitude=slice(2, 9))
    da.name = "t2m"
    da.rio.write_crs("EPSG:4326", inplace=True)

    with temp_file(extension=".tif") as path:
        to_target("file", path, data=da, encoder="geotiff", driver="COG", dtype="float32")
        with _open(path) as src:
            assert src.count == 1
            assert (src.height, src.width) == (4, 7)
            assert src.crs.to_epsg() == 4326
            assert np.allclose(src.read(1), da.values, atol=1e-3)


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_several_2d_variables_with_nodata():
    """A Dataset with several 2D variables writes one band per variable, and
    nodata applies to every band (Dataset.rio has no write_nodata).
    """
    from earthkit.data import to_target

    ds = _synthetic((7, 12), ("latitude", "longitude"), var_names=("t", "z"))
    ds.rio.write_crs("EPSG:4326", inplace=True)

    with temp_file(extension=".tif") as path:
        to_target("file", path, data=ds, encoder="geotiff", driver="COG", dtype="float32", nodata=np.nan)
        with _open(path) as src:
            assert src.count == 2
            assert src.dtypes[0] == "float32"
            assert np.isnan(src.nodata)


@pytest.mark.skipif(NO_RASTERIO, reason="rasterio not available")
@pytest.mark.with_proj
def test_geotiff_encoder_xarray_without_crs_is_an_error():
    from earthkit.data import to_target

    ds = from_source("file", earthkit_test_data_file("test_single.grib")).to_fieldlist()
    da = ds.to_xarray()["2t"]

    with temp_file(extension=".tif") as path:
        with pytest.raises(ValueError, match="CRS"):
            to_target("file", path, data=da, encoder="geotiff")

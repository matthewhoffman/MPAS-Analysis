""" util.py - generic utilities for momlevel """

import warnings

import numpy as np
import xarray as xr

from mpas_analysis.ocean.sealevel import eos
from mpas_analysis.ocean.sealevel import trend


def annual_average(xobj, tcoord="time"):
    """Function to calculate annual averages

    This function calculates the annual average of the supplied xarray object.
    The average is weighted by the number of days in the month, as inferred
    from the calendar attributes of the time coordinate objects. Non-numeric
    variables are skipped.

    Parameters
    ----------
    xobj : xarray.core.dataset.Dataset or xarray.core.dataarray.DataArray
        Input xarray object
    tcoord : str, optional
        Name of time coordinate, by default "time"

    Returns
    -------
    xarray.core.dataset.Dataset
    """
    calendar = xobj[tcoord].values[0].calendar

    dim_coords = set(xobj.dims).union(set(xobj.coords))

    if isinstance(xobj, xr.core.dataset.Dataset):
        variables = set(xobj.variables) - dim_coords
        _xobj = xr.Dataset()
        for var in variables:
            if xobj[var].dtype not in ["object", "timedelta64[ns]"]:
                _xobj[var] = xobj[var]
    else:
        _xobj = xobj

    groups = _xobj.groupby(f"{tcoord}.year")

    _annuals = []
    for grp in sorted(dict(groups).keys()):
        assert len(groups[grp][tcoord]) == 12
        _annuals.append(
            groups[grp].weighted(groups[grp][tcoord].dt.days_in_month).mean(tcoord)
        )

    result = xr.concat(_annuals, dim="time")
    result = result.transpose("time", ...)

    def _find_ann_midpoint(year, calendar=calendar):
        """finds the midpoint of the year along the time dimension"""
        bounds = xr.cftime_range(
            f"{year}-01-01", freq="YS", periods=2, calendar=calendar
        )
        return bounds[0] + ((bounds[1] - bounds[0]) / 2)

    new_time_axis = [
        _find_ann_midpoint(str(x).zfill(4), calendar=calendar)
        for x in dict(groups).keys()
    ]

    result = result.assign_coords({"time": new_time_axis})

    if isinstance(xobj, xr.core.dataset.Dataset):
        for var in list(result.variables):
            result[var].attrs = xobj[var].attrs if var in list(xobj.variables) else {}

    for coord in list(result.coords):
        result[coord].attrs = xobj[coord].attrs if coord in list(xobj.coords) else {}

    for dim in list(result.dims):
        result[dim].attrs = xobj[dim].attrs if dim in list(xobj.dims) else {}

    result.attrs = xobj.attrs

    return result


def annual_cycle(xobj, tcoord="time", func="mean", time_axis_year=None):
    """Function to calculate annual cycle climatology

    This function calculates the annual cycle climatology from an
    xarray dataset containing monthly timeseries variables.

    Parameters
    ----------
    xobj : xarray.core.dataset.Dataset or xarray.core.dataarray.DataArray
        Input xarray object
    tcoord : str, optional
        Name of time coordinate, by default "time"
    func : str, optional
        "mean", "std", "min", or "max" across for the climatology,
        by default "mean"
    time_axis_year : int, optional
        Specify year used in resulting time axis, otherwise use the
        mean year for the entire dataset, by default None

    Returns
    -------
    xarray.core.dataset.Dataset
        Annual cycle climatology with 12 time points, 1 per month
    """

    calendar = xobj[tcoord].values[0].calendar

    dim_coords = set(xobj.dims).union(set(xobj.coords))

    if isinstance(xobj, xr.core.dataset.Dataset):
        variables = set(xobj.variables) - dim_coords
        _xobj = xr.Dataset()
        for var in variables:
            if xobj[var].dtype not in ["object", "timedelta64[ns]"]:
                _xobj[var] = xobj[var]
    else:
        _xobj = xobj

    if time_axis_year is not None:
        midyear = int(time_axis_year)
    else:
        endyr = xobj[tcoord].values[-1]
        startyr = xobj[tcoord].values[0]
        delta = (endyr - startyr) / 2
        midyear = startyr + delta
        midyear = midyear.year

    bounds = xr.cftime_range(
        f"{str(midyear).zfill(4)}-01-01",
        freq="MS",
        periods=13,
        calendar=calendar,
    )

    bounds = [
        bounds[x] + (bounds[x + 1] - bounds[x]) / 2 for x in range(0, len(bounds) - 1)
    ]

    result = _xobj.groupby(f"{tcoord}.month")

    if func == "mean":
        result = result.mean(tcoord)
    elif func == "min":
        result = result.min(tcoord)
    elif func == "max":
        result = result.max(tcoord)
    elif func == "std":
        result = result.std(tcoord)
    else:
        raise ValueError(f"Unknown argument 'func={func}' to annual cycle")

    result = result.rename({"month": tcoord})
    result = result.assign_coords({tcoord: bounds})

    return result


def default_coords(coord_names=None):
    """Function to set the default coordinate names

    This function reads the coordinate names dictionary and returns
    either the specified values or the default values if missing from
    coord_names.

    Parameters
    ----------
    coord_names : :obj:`dict`, optional
        Dictionary of coordinate name mappings. This should use x, y, z, and t
        as keys, e.g. {"x":"xh", "y":"yh", "z":"z_l", "t":"time"}. Coordinate
        bounds are noted by appending "bounds" to each key, i.e. "zbounds"

    Returns
    -------
    Tuple
        Default coordinate names
    """
    # abstracting these coordinates in case they need to be promoted to kwargs
    coord_names = {} if coord_names is None else coord_names
    assert isinstance(coord_names, dict), "Coordinate mapping must be a dictionary."
    zcoord = coord_names["z"] if "z" in coord_names.keys() else "z_l"
    zbounds = coord_names["zbounds"] if "zbounds" in coord_names.keys() else "z_i"
    tcoord = coord_names["t"] if "t" in coord_names.keys() else "time"
    return (tcoord, zcoord, zbounds)


def eos_func_from_str(eos_str, func_name="density"):
    """Function to resolve equation of state function

    This function takes the name of an equation of state in string
    format and returns the corresponding function object.

    Parameters
    ----------
    eos_str : str
        Equation of state

    Returns
    -------
    function
    """

    assert isinstance(eos_str, str), "Expecting string for equation of state"
    eos_str = eos_str.lower()
    avail_eos = list(eos.__dict__.keys())
    if eos_str not in avail_eos:
        raise ValueError(f"Unknown equation of state: {eos_str}")

    return eos.__dict__[eos_str].__dict__[func_name]


def monthly_average(xobj, tcoord="time"):
    """Function to calculate monthly averages from daily data

    This function calculates monthly averages of the supplied xarray object.
    Non-numeric variables are skipped.

    Parameters
    ----------
    xobj : xarray.core.dataset.Dataset or xarray.core.dataarray.DataArray
        Input xarray object
    tcoord : str, optional
        Name of time coordinate, by default "time"

    Returns
    -------
    xarray.core.dataset.Dataset
    """

    calendar = xobj[tcoord].values[0].calendar

    dim_coords = set(xobj.dims).union(set(xobj.coords))

    if isinstance(xobj, xr.core.dataset.Dataset):
        variables = set(xobj.variables) - dim_coords
        _xobj = xr.Dataset()
        for var in variables:
            if xobj[var].dtype not in ["object", "timedelta64[ns]"]:
                _xobj[var] = xobj[var]
    else:
        _xobj = xobj

    groups = _xobj.groupby(f"{tcoord}.year")

    record = []

    for grp in sorted(dict(groups).keys()):
        _ds = groups[grp].groupby(f"{tcoord}.month").mean(tcoord)

        bounds = xr.cftime_range(
            f"{str(grp).zfill(4)}-01-01",
            freq="MS",
            periods=13,
            calendar=calendar,
        )

        bounds = [
            bounds[x] + (bounds[x + 1] - bounds[x]) / 2
            for x in range(0, len(bounds) - 1)
        ]

        _ds = _ds.rename({"month": tcoord})
        _ds = _ds.assign_coords({tcoord: bounds})
        record.append(_ds)

    result = xr.concat(record, dim=tcoord)
    result = result.transpose(tcoord, ...)

    return result


def reset_encoding(xobj, attrs=None):
    """Function to reset encoding attributes on an xarray object

    Parameters
    ----------
    xobj : xarray.core.dataset.Dataset or xarray.core.dataarray.DataArray
        Input xarray object
    attrs : list, optional
        Attributes to reset, by default None

    Returns
    -------
    xarray.core.dataset.Dataset or xarray.core.dataarray.DataArray
        Xarray object without encoding attributes
    """

    attrs = ["chunks", "preferred_chunks"] if attrs is None else attrs

    if isinstance(xobj, xr.DataArray):
        for attr in attrs:
            xobj.encoding.pop(attr, None)

    elif isinstance(xobj, xr.Dataset):
        for attr in attrs:
            xobj.encoding.pop(attr, None)
            for var in xobj.variables:
                xobj[var].encoding.pop(attr, None)

    else:
        raise ValueError("xobj must be an xarray Dataset or DataArray")

    return xobj


def validate_areacello(areacello, reference=3.6111092e14, tolerance=0.02):
    """Function to test validity of ocean cell area field

    This function tests if the sum of the ocean cell area is within a
    specified tolerance of a real-world value.

    The intent of this function is to identify gross deviations, such
    as the inadvertent use of the global surface area.

    Parameters
    ----------
    areacello : xarray.core.dataarray.DataArray
        Field containing ocean cell area
    reference : float, optional
        Sum of ocean surface area in meters, by default 3.6111092e14
    tolerance : float, optional
        Acceptable +/- tolerance range percentage, by default 0.02

    Returns
    -------
    bool
        True if areacello is within tolerance
    """
    error = (areacello.sum() - reference) / reference
    result = bool(np.abs(error) < tolerance)
    return result


def validate_dataset(dset, reference=False, strict=True, additional_vars=None):
    """Function to validate requirements of the datasets

    This function determines if a supplied dataset is either a valid
    input dataset or reference dataset. It checks for the presence
    of required fields and that they have the correct dimensionality.

    Errors are collected and reported back as a group.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Dataset supplied for validation
    reference : bool, optional
        Flag denoting if `dset` is a reference dataset, by default False
    strict : bool, optional
        If true, errors are handled as fatal Exceptions. If false,
        warnings are issued, by default True
    additional_vars : :obj:`list`, optional
        List of additional variables to check for in the dataset

    Returns
    -------
    None
    """

    dset_varlist = list(dset.variables)
    exceptions = []

    # check that reference dset does not contain a time dimension
    if reference:
        try:
            assert "Time" not in [
                x.lower for x in list(dset.coords)
            ], "Reference dataset cannot contain a time coordinate"
        except AssertionError as e:
            exceptions.append(e)

    # check for missing variables
    expected_varlist = ["thetao", "so", "volcello", "areacello"]

    # add additional variables if supplied
    if additional_vars is not None:
        additional_vars = (
            [additional_vars]
            if not isinstance(additional_vars, list)
            else additional_vars
        )
    else:
        additional_vars = []
    expected_varlist = expected_varlist + additional_vars

    reference_varlist = ["rho", "volo", "masso", "rhoga"]
    expected_varlist = (
        expected_varlist + reference_varlist if reference else expected_varlist
    )

    missing = list(set(expected_varlist) - set(dset_varlist))

    try:
        assert len(missing) == 0, f"Reference dataset is missing variables: {missing}"
    except AssertionError as e:
        exceptions.append(e)

    # check for dimensionality of 3D vars
    ranks = (2, "(x,z)") if reference else (3, ("t,x,z"))
    for var in ["thetao", "so", "volcello"]:
        if var in dset.variables:
            try:
                assert (
                    len(dset[var].dims) == ranks[0]
                ), f"Variable {var} must have exactly {ranks[0]} dimensions {ranks[1]}"
            except AssertionError as e:
                exceptions.append(e)

    # check for dimensionality of 2D vars
    for var in ["areacello", "deptho"]:
        if var in dset.variables:
            try:
                assert (
                    len(dset[var].dims) == 1
                ), f"Variable {var} must have exactly 1 dimension (nCells)"
            except AssertionError as e:
                exceptions.append(e)

    # validate ocean cell area and make sure it is sensible
    if "areacello" in dset.variables:
        try:
            assert validate_areacello(
                dset["areacello"]
            ), "Variable `areacello` field is out of range. It may not be masked."
        except AssertionError as e:
            if not strict:
                warnings.warn(str(e))
            else:
                exceptions.append(e)

    # check for dimensionality of scalars
    if reference:
        if "rho" not in missing:
            try:
                assert (
                    len(dset["rho"].dims) == 2
                ), "Variable rho must have exactly 2 dimensions (x,z)"
            except AssertionError as e:
                exceptions.append(e)

        for var in ["masso", "volo", "rhoga"]:
            if var not in missing:
                try:
                    assert len(dset[var].dims) == 0, f"Variable {var} must be a scalar"
                except AssertionError as e:
                    exceptions.append(e)

    if len(exceptions) > 0:
        for e in exceptions:
            print(e)
        raise ValueError("Errors found in dataset.")


def validate_tidegauge_data(arr, xcoord, ycoord, mask):
    """Function to validate inputs to `tidegauge.extract_tidegauge`

    This function validates the input arguments before the tide gauge
    point extraction function continues

    Parameters
    ----------
    arr : xarray.core.dataarray.DataArray
        Input DataArray object
    xcoord : xarray.core.dataarray.DataArray or str
        x-coordinate name or object
    ycoord : xarray.core.dataarray.DataArray or str
        y-coordinate name or object
    mask : xarray.core.dataarray.DataArray or None
        wet mask array
    """
    # confirm that input is xarray
    assert isinstance(
        arr, xr.DataArray
    ), "Input array must be `xarray.DataArray` instance"

    # if xcoord and ycoord are strings, check that coordinates
    # exist in the xarray object.
    _coords = list(arr.coords)

    if isinstance(xcoord, str):
        assert xcoord in _coords, f"`{xcoord}` not found in input array."
    else:
        assert isinstance(xcoord, xr.DataArray), (
            "xcoord must either be a DataArray object or a "
            + "string that references an existing coordinate"
        )

    if isinstance(ycoord, str):
        assert ycoord in _coords, f"`{ycoord}` not found in input array."
    else:
        assert isinstance(ycoord, xr.DataArray), (
            "ycoord must either be a DataArray object or a "
            + "string that references an existing coordinate"
        )

    if mask is not None:
        assert isinstance(mask, xr.DataArray), "mask be a DataArray object"


def linear_detrend(*args, **kwargs):
    warnings.warn(
        "`util.linear_trend()` will be removed. "
        + "Please use version in the new `momlevel.trend` module",
        DeprecationWarning,
        stacklevel=2,
    )
    return trend.linear_detrend(*args, **kwargs)

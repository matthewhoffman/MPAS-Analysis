""" reference.py - module to establish sea level reference states """

import xarray as xr

from mpas_analysis.ocean.sealevel.derived import calc_masso
from mpas_analysis.ocean.sealevel.derived import calc_rho
from mpas_analysis.ocean.sealevel.derived import calc_rhoga
from mpas_analysis.ocean.sealevel.derived import calc_volo


def setup_reference_state(
    dset, patm=101325.0, eos="Wright", coord_names=None, time_index=0, do_calc_rho=False
):
    """Function to generate reference dataset

    This function generates a dataset of initial reference values for
    use in all sea level calculations. Values are taken from an input
    dataset that contains thetao, so, volcello, and areacello. The
    initial in situ density is calculated along with scalar properties
    that are used in offline approximations of the steric effect from
    Boussinesq models.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Dataset containing thetao, so, volcello, and areacello
    patm : float or xarray.core.dataarray.DataArray
        Atmospheric pressure at the sea surface in Pa,
        by default 101325 Pa (US Standard Atmosphere)
    eos : str, optional
        Equation of state, by default "Wright"
    coord_names : :obj:`dict`, optional
        Dictionary of coordinate name mappings. This should use x, y, z, and t
        as keys, e.g. {"x":"xh", "y":"yh", "z":"z_l", "t":"time"}
    time_index : int, optional
        Time index to use for reference state, by default 0 (first time index)

    Returns
    -------
    xarray.core.dataset.Dataset
        Dataset of reference values
    """

    # default coordinate names
    tcoord = 'Time'
    zcoord = 'nVertLevels'

    # initialize reference dataset
    reference = xr.Dataset()

    # extract requested time level and squeeze the arrays
    reference["thetao"] = (
        dset["thetao"].isel({tcoord: time_index}).squeeze().reset_coords(drop=True)
    )
    reference["so"] = (
        dset["so"].isel({tcoord: time_index}).squeeze().reset_coords(drop=True)
    )
    reference["volcello"] = (
        dset["volcello"].isel({tcoord: time_index}).squeeze().reset_coords(drop=True)
    )

    # calculate in situ reference density
    if do_calc_rho:
        # approximate pressure from depth coordinate
        pres = (dset[zcoord] * 1.0e4) + patm
        print(f'theta={reference["thetao"].values.min()}, so={reference["so"].values.min()}, pres={pres.values.min()}')
        reference["rho"] = calc_rho(reference.thetao, reference.so, pres, eos=eos)
    else:
        reference["rho"] = (
            dset["rho"].isel({tcoord: time_index}).squeeze().reset_coords(drop=True)
    )
    print(f'rho min={reference["rho"].values.min()}, max={reference["rho"].values.max()}')

    # calculate global ocean volume
    reference["volo"] = calc_volo(reference.volcello)
    print(f'reference volo={reference["volo"].values}')

    # calculate global ocean mass
    reference["masso"] = calc_masso(reference.rho, reference.volcello, tcoord=tcoord)
    print(f'reference masso={reference["masso"].values}')

    # calculate global average in situ density
    reference["rhoga"] = calc_rhoga(reference.masso, reference.volo)
    print(f'reference rhoga={reference["rhoga"].values}')

    # copy the reference cell area
    reference["areacello"] = dset.areacello

    return reference

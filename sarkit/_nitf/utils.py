"""Common functionality for converting metadata"""

from typing import Any, BinaryIO, TypeGuard

import numpy as np

import sarkit.standards.geocoords


def is_file_like(the_input: Any) -> TypeGuard[BinaryIO]:
    """
    Verify whether the provided input appear to provide a "file-like object". This
    term is used ubiquitously, but not all usages are identical. In this case, we
    mean that there exist callable attributes `read`, `write`, `seek`, and `tell`.

    Note that this does not check the mode (binary/string or read/write/append),
    as it is not clear that there is any generally accessible way to do so.

    Parameters
    ----------
    the_input

    Returns
    -------
    bool
    """

    out = True
    for attribute in ["read", "write", "seek", "tell"]:
        value = getattr(the_input, attribute, None)
        out &= callable(value)
    return out


def is_real_file(the_input: BinaryIO) -> bool:
    """
    Determine if the file-like object is associated with an actual file.
    This is mainly to consider suitability for establishment of a numpy.memmap.

    Parameters
    ----------
    the_input : BinaryIO

    Returns
    -------
    bool
    """

    if not hasattr(the_input, "fileno"):
        return False
    # noinspection PyBroadException
    try:
        fileno = the_input.fileno()
        return isinstance(fileno, int) and (fileno >= 0)
    except Exception:
        return False


def _interpolate_corner_points_string(entry, rows, cols, icp):
    """
    Interpolate the corner points for the given subsection from
    the given corner points. This supplies entries for the NITF headers.
    Parameters
    ----------
    entry : numpy.ndarray
        The corner points of the form `(row_start, row_stop, col_start, col_stop)`
    rows : int
        The number of rows in the parent image.
    cols : int
        The number of cols in the parent image.
    icp : numpy.ndarray
        The parent image corner points in geodetic coordinates.

    Returns
    -------
    str
        suitable for IGEOLO entry.
    """

    def dms_format(latlong):
        lat, long = latlong

        def reduce(value):
            val = abs(value)
            deg = int(val)
            val = 60 * (val - deg)
            mins = int(val)
            secs = 60 * (val - mins)
            secs = int(secs)
            return deg, mins, secs

        x = "S" if lat < 0 else "N"
        y = "W" if long < 0 else "E"
        return reduce(lat) + (x,), reduce(long) + (y,)

    if icp is None:
        return ""

    if icp.shape[1] == 2:
        icp_new = np.zeros((icp.shape[0], 3), dtype=np.float64)
        icp_new[:, :2] = icp
        icp = icp_new
    icp_ecf = sarkit.standards.geocoords.geodetic_to_ecf(icp)

    const = 1.0 / (rows * cols)
    pattern = entry[np.array([(0, 2), (1, 2), (1, 3), (0, 3)], dtype=np.int64)]
    out = []
    for row, col in pattern:
        pt_array = const * np.sum(
            icp_ecf
            * (
                np.array([rows - row, row, row, rows - row])
                * np.array([cols - col, cols - col, col, col])
            )[:, np.newaxis],
            axis=0,
        )

        pt_latlong = sarkit.standards.geocoords.ecf_to_geodetic(pt_array)[:2]
        dms = dms_format(pt_latlong)
        out.append(
            "{0:02d}{1:02d}{2:02d}{3:s}".format(*dms[0])
            + "{0:03d}{1:02d}{2:02d}{3:s}".format(*dms[1])
        )
    return "".join(out)

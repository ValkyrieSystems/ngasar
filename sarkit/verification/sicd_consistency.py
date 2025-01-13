"""
==================================================
Sensor Independent Complex Data (SICD) Consistency
==================================================

Functionality for verifying SICD files for internal consistency.

CLI
===

.. autoprogram:: sarkit.verification.sicd_consistency:_parser()
   :prog: sicd_consistency.py

Consistency Object
==================

.. autosummary::

    SicdConsistency

"""

import argparse
import copy
import datetime
import functools
import logging
import pathlib
from typing import Optional

import numpy as np
import numpy.linalg as npl
import numpy.polynomial.polynomial as npp
import shapely.geometry as shg
from lxml import etree

import sarkit.constants
import sarkit.standards.sicd as ss_std
import sarkit.standards.sicd.io as ss_io
import sarkit.standards.sicd.projection as ss_proj
import sarkit.standards.sicd.xml as ss_xml
import sarkit.verification.consistency as con
from sarkit._nitf.nitf import NITFDetails
from sarkit.standards import geocoords

logger = logging.getLogger(__name__)

KAPFAC: float = 0.8859


def _unit(vec, axis=-1):
    return vec / np.linalg.norm(vec, axis=axis, keepdims=True)


def _uvecs_in_ground(xmlhelp):
    look = 1.0 if xmlhelp.load("./{*}SCPCOA/{*}SideOfTrack") == "L" else -1.0
    scp_ecf = xmlhelp.load("./{*}GeoData/{*}SCP/{*}ECF")
    arp_pos_ecf = xmlhelp.load("./{*}SCPCOA/{*}ARPPos")
    arp_vel_ecf = xmlhelp.load("./{*}SCPCOA/{*}ARPVel")
    spn = _unit(np.cross(look * (arp_pos_ecf - scp_ecf), arp_vel_ecf))

    row_uvect = xmlhelp.load("./{*}Grid/{*}Row/{*}UVectECF")
    col_uvect = xmlhelp.load("./{*}Grid/{*}Col/{*}UVectECF")

    scp_llh = xmlhelp.load("./{*}GeoData/{*}SCP/{*}LLH")
    u_up = geocoords.up(scp_llh)
    scale_factor = np.dot(spn, u_up)

    row_in_ground = row_uvect - np.dot(row_uvect, u_up) * spn / scale_factor
    col_in_ground = col_uvect - np.dot(col_uvect, u_up) * spn / scale_factor
    assert np.abs(np.dot(row_in_ground, u_up)) < 1e-4
    assert np.abs(np.dot(col_in_ground, u_up)) < 1e-4
    return row_in_ground, col_in_ground


def _display_index_to_coord(xmlhelp, display_index):
    spacing = np.array(
        [
            xmlhelp.load(
                "./{*}RadarCollection/{*}Area/{*}Plane/{*}XDir/{*}LineSpacing"
            ),
            xmlhelp.load(
                "./{*}RadarCollection/{*}Area/{*}Plane/{*}YDir/{*}SampleSpacing"
            ),
        ]
    )
    origin = np.array(
        [
            xmlhelp.load("./{*}RadarCollection/{*}Area/{*}Plane/{*}RefPt/{*}Line"),
            xmlhelp.load("./{*}RadarCollection/{*}Area/{*}Plane/{*}RefPt/{*}Sample"),
        ]
    )
    return (display_index - origin) * spacing


def _get_valid_data_vertices(xmlhelp):
    """Returns a polygon describing the SICD's valid data region."""
    vertices = xmlhelp.load("./{*}ImageData/{*}ValidData")
    if vertices is None:  # Use edges of full image
        nrows = xmlhelp.load("./{*}ImageData/{*}FullImage/{*}NumRows")
        ncols = xmlhelp.load("./{*}ImageData/{*}FullImage/{*}NumCols")

        vertices = [(0, 0), (0, ncols - 1), (nrows - 1, ncols - 1), (nrows - 1, 0)]

    return np.asarray(vertices)


def _sample_valid_pixels(xmlhelp, shape=(5, 7)):
    """Returns a number of row-column pairs that are inside ValidData."""
    validdata = _get_valid_data_vertices(xmlhelp)
    min_index = validdata.min(axis=0)
    max_index = validdata.max(axis=0)
    grid = np.stack(
        np.meshgrid(
            *[
                np.linspace(min_index[dim], max_index[dim], shape[dim])
                for dim in range(2)
            ],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    grid = np.concatenate([grid, validdata], axis=0)
    points = shg.MultiPoint(grid)
    intersecting_points = shg.polygon.orient(shg.Polygon(validdata)).intersection(
        points
    )
    return np.array([point.coords[0] for point in intersecting_points.geoms])


def _grid_index_to_coord(xmlhelp, grid_index):
    spacing = np.array(
        [
            xmlhelp.load("./{*}Grid/{*}Row/{*}SS"),
            xmlhelp.load("./{*}Grid/{*}Col/{*}SS"),
        ]
    )
    origin = np.array(
        [
            xmlhelp.load("./{*}ImageData/{*}SCPPixel/{*}Row"),
            xmlhelp.load("./{*}ImageData/{*}SCPPixel/{*}Col"),
        ]
    )
    return (grid_index - origin) * spacing


def _create_rectangle(x0, y0, num_x, num_y):
    return shg.box(x0, y0, x0 + num_x - 1, y0 + num_y - 1, ccw=False)


def _compute_pfa_min_max_fx(xmlhelp):
    """Computes min and max FX based on PFA parameters."""
    krg1 = xmlhelp.load("./{*}PFA/{*}Krg1")
    krg2 = xmlhelp.load("./{*}PFA/{*}Krg2")
    kaz1 = xmlhelp.load("./{*}PFA/{*}Kaz1")
    kaz2 = xmlhelp.load("./{*}PFA/{*}Kaz2")
    ksf = xmlhelp.load("./{*}PFA/{*}SpatialFreqSFPoly")

    def _k_rect_to_fx(krg, kaz):
        """Converts k-space location to FX."""
        kap = np.sqrt(krg**2 + kaz**2)
        theta = np.arctan2(kaz, krg)

        return sarkit.constants.c / 2.0 / npp.polyval(theta, ksf) * kap

    min_fx = min(_k_rect_to_fx(krg1, kaz) for kaz in [kaz1, kaz2])
    max_fx = max(_k_rect_to_fx(krg2, kaz) for kaz in [kaz1, kaz2])

    if kaz1 < 0 < kaz2:
        min_fx = min(min_fx, _k_rect_to_fx(krg1, 0))

    return min_fx, max_fx


def _get_des_bytes_from_details(nitf_details):
    """Return the first SICD DES"""
    for i in range(nitf_details.des_subheader_offsets.size):
        des_header = nitf_details.parse_des_subheader(i)
        if des_header.UserHeader.DESSHTN.startswith("urn:SICD"):
            return nitf_details.get_des_bytes(i)
    raise ValueError("Unable to find SICD DES")


def per_grid_dim(method):
    method.per_grid_dim = True
    return method


class SicdConsistency(con.ConsistencyChecker):
    """Check SICD file structure and metadata for internal consistency

    This class can be used to perform and log comparisons. Each comparison
    can be logged as either an ``'Error'`` or a ``'Warning'``.

    Parameters
    ----------
    sicdroot : lxml.etree.Element or lxml.etree.ElementTree
        SICD XML metadata
    schema_override : str, optional
        Path to SICD XML Schema (Bypass auto selection of SICD schema)
    version_override : str, optional
        Bypass auto detection of SICD version.  Required if schema_override is specified.  eg: "1.1.0"
    ntf : NITFDetails, optional
        Header information from a NITF file
    """

    def __init__(
        self,
        sicdroot,
        schema_override=None,
        version_override=None,
        ntf=None,
    ):
        super().__init__()

        self.sicdroot = etree.fromstring(
            etree.tostring(sicdroot)
        )  # handle element or tree -> element
        self.xmlhelp = ss_xml.XmlHelper(self.sicdroot.getroottree())

        if schema_override:
            if version_override is None:
                raise ValueError("Version must be specified when overriding schema")
            self.version = version_override
            self.schema = schema_override
        else:
            self.version = version_override or self._version_lookup()
            if self.version is None:
                raise ValueError("Unable to determine SICD version from XML namespace")
            urn = {v["version"]: k for k, v in ss_io.VERSION_INFO.items()}[self.version]
            self.schema = ss_io.VERSION_INFO[urn]["schema"]

        self.ntf = ntf

        # process decorated methods to generate additional tests
        # reverse the enumerated list so that we don't disturb indices on later iterations as we insert into the list
        for index, func in reversed(list(enumerate(self.funcs))):
            if getattr(func, "per_grid_dim", False):
                subfuncs = []
                for grid_dim in ("Row", "Col"):
                    subfunc = functools.partial(func, grid_dim=grid_dim)
                    subfunc.__doc__ = (
                        f"{func.__doc__.removesuffix('.')} for Grid/{grid_dim}."
                    )
                    subfunc.__name__ = f"{func.__name__}_{grid_dim.lower()}"
                    subfuncs.append(subfunc)
                self.funcs[index : index + 1] = subfuncs

    @staticmethod
    def from_file(
        filename: str,
        schema: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "SicdConsistency":
        """Create SicdConsistency class from a file

        Parameters
        ----------
        filename : str
            Path of the file used to create the class.  Can be either XML or nitf.
        schema : str, optional
            Path to SICD XML Schema (Bypass auto selection of SICD schema)
        version : str, optional
            Bypass auto detection of SICD version.  Required if schema_override is specified.  eg: "1.1.0"

        Returns
        -------
        SicdConsistency class
        """
        with open(filename, "rb") as infile:
            try:
                sicdroot = etree.parse(infile)
                nitf_details = None
            except etree.XMLSyntaxError:
                nitf_details = NITFDetails(str(filename))
                des_bytes = _get_des_bytes_from_details(nitf_details)
                sicdroot = etree.fromstring(des_bytes)

        return SicdConsistency(
            sicdroot,
            schema_override=schema,
            version_override=version,
            ntf=nitf_details,
        )

    def _assert_poly_1d(self, poly, poly_label):
        """Helper method for checking 1d polynomial nodes are consistent."""
        coef_nodes = poly.findall("./{*}Coef")
        num_coefs = int(poly.attrib["order1"]) + 1
        coefs = [int(coef.attrib["exponent1"]) for coef in coef_nodes]
        with self.need(f"{poly_label}: Exponents are unique"):
            assert len(set(coefs)) == len(coef_nodes)
        with self.need(f"{poly_label}: Coef exponents between 0 and order"):
            assert all(coef >= 0 and coef < num_coefs for coef in coefs)

    def _assert_poly_2d(self, poly, poly_label):
        """Helper method for checking 2d polynomial nodes are consistent."""
        coef_nodes = poly.findall("./{*}Coef")
        num_coefs = (int(poly.attrib["order1"]) + 1, int(poly.attrib["order2"]) + 1)
        coefs = [
            (int(coef.attrib["exponent1"]), int(coef.attrib["exponent2"]))
            for coef in coef_nodes
        ]
        with self.need(f"{poly_label}: Exponents are unique"):
            assert len(set(coefs)) == len(coef_nodes)
        with self.need(f"{poly_label}: Coef exponents between 0 and order"):
            assert all(coef[0] >= 0 and coef[0] < num_coefs[0] for coef in coefs)
            assert all(coef[1] >= 0 and coef[1] < num_coefs[1] for coef in coefs)

    def _assert_poly_xyz(self, poly, poly_label):
        """Helper method for checking XYZ polynomial nodes are consistent."""
        for dim in ["X", "Y", "Z"]:
            self._assert_poly_1d(poly.find(f"./{{*}}{dim}"), f"{poly_label}/{{*}}{dim}")

    def _version_lookup(self):
        """Returns the version string associated with the XML instance or None if a match is not found."""
        this_ns = etree.QName(self.sicdroot).namespace
        if this_ns is None:
            return None
        for schema_info in ss_io.VERSION_INFO.values():
            schema_path = schema_info.get("schema")
            if schema_path is not None and this_ns == etree.parse(
                schema_path
            ).getroot().get("targetNamespace"):
                return schema_info["version"]
        return None

    def check_against_schema(self) -> None:
        """Checks against schema."""
        with self.need(
            f"Schema available for checking xml whose root tag = {self.sicdroot.tag}"
        ):
            assert self.schema is not None
            schema = etree.XMLSchema(file=str(self.schema))
            with self.need("XML passes schema"):
                assert schema.validate(self.sicdroot), schema.error_log

    def check_nitf_imseg(self) -> None:
        """Check NITF Image SubHeaders"""
        collect_start = self.xmlhelp.load("./{*}Timeline/{*}CollectStart")
        pixel_info = ss_io.PIXEL_TYPES[self.xmlhelp.load("./{*}ImageData/{*}PixelType")]
        expected_nbpp = pixel_info["bytes"] * 8 / 2
        with self.precondition():
            assert self.ntf is not None
            imsegs = self.ntf.img_headers
            for imseg in imsegs:
                idatim = datetime.datetime.strptime(
                    imseg.IDATIM + "+0000", "%Y%m%d%H%M%S%z"
                )
                with self.need("Valid image subheaders"):
                    assert idatim <= collect_start + datetime.timedelta(seconds=1)
                    assert idatim >= collect_start - datetime.timedelta(seconds=1)
                    assert imseg.PVTYPE.rstrip() == pixel_info["pvtype"]
                    assert imseg.IREP.rstrip() == "NODISPLY"
                    assert imseg.ICAT.rstrip() == "SAR"
                    assert imseg.ABPP == expected_nbpp
                    assert imseg.PJUST == "R"
                    assert imseg.ICORDS == "G"
                    assert imseg.IC in ["NC", "NM", "C7", "M7"]
                    assert imseg.ISYNC == 0
                    assert imseg.IMODE.rstrip() == "P"
                    assert imseg.NBPR == 1
                    assert imseg.NBPC == 1

                    if imseg.NCOLS > 8192:
                        assert imseg.NPPBH == 0
                    else:
                        assert imseg.NPPBH == imseg.NCOLS

                    if imseg.NROWS > 8192:
                        assert imseg.NPPBV == 0
                    else:
                        assert imseg.NPPBV == imseg.NROWS

                    assert imseg.NBPP == expected_nbpp
                    assert imseg.IMAG == "1.0"

            with self.need("Sequential IID1"):
                if len(imsegs) == 1:
                    assert imsegs[0].IID1.rstrip() == "SICD000"
                else:
                    expected_iid1s = [
                        f"SICD{idx + 1:03d}   " for idx in range(len(imsegs))
                    ]
                    actual_iid1s = [imseg.IID1 for imseg in imsegs]
                    assert actual_iid1s == expected_iid1s

    def check_nitf_imseg_lvls(self) -> None:
        """Check NITF inter-Image SubHeaders Display and Attachment levels"""
        with self.precondition():
            assert self.ntf is not None
            imsegs = self.ntf.img_headers
            with self.need("Consistent NITF inter-Image SubHeaders Display levels"):
                assert np.array_equal(
                    [imseg.IDLVL for imseg in imsegs],
                    np.arange(len(imsegs)) + 1,
                )
            with self.need("Consistent NITF inter-Image SubHeaders Attachment levels"):
                assert np.array_equal(
                    [imseg.IALVL for imseg in imsegs], np.arange(len(imsegs))
                )

    def _segmentation(self):
        """Section 3.2.1 Image Segment Parameters and Equations"""
        bytes_per_pixel = ss_io.PIXEL_TYPES[
            self.xmlhelp.load("./{*}ImageData/{*}PixelType")
        ]["bytes"]
        is_size_max = 9_999_999_998
        iloc_max = 99_999
        num_rows = self.xmlhelp.load("./{*}ImageData/{*}NumRows")
        num_cols = self.xmlhelp.load("./{*}ImageData/{*}NumCols")
        bytes_per_row = bytes_per_pixel * num_cols
        product_size = bytes_per_pixel * num_rows * num_cols
        limit1 = int(np.floor(is_size_max / bytes_per_row))
        num_rows_limit = min(limit1, iloc_max)
        if product_size <= is_size_max:
            num_is = 1
            num_rows_is = [num_rows]
            first_row_is = [0]
            row_offset_is = [0]
        else:
            num_is = int(np.ceil(num_rows / num_rows_limit))
            num_rows_is = [num_rows_limit] * num_is
            first_row_is = [0] * num_is
            row_offset_is = [0] * num_is
            for n in range(num_is - 1):
                first_row_is[n + 1] = (n + 1) * num_rows_limit
                row_offset_is[n + 1] = num_rows_limit
            num_rows_is[-1] = num_rows - (num_is - 1) * num_rows_limit

        return num_is, num_rows_is, first_row_is, row_offset_is

    def check_nitf_imseg_size(self) -> None:
        """Check the size of each NITF image segment"""
        with self.precondition():
            assert self.ntf is not None
            imsegs = self.ntf.img_headers
            num_is, num_rows_is, _, _ = self._segmentation()
            num_cols = self.xmlhelp.load("./{*}ImageData/{*}NumCols")
            with self.need("Consistent number of image segments"):
                assert len(imsegs) == num_is
                for imidx, imseg in enumerate(imsegs):
                    with self.need("Matching NROWS"):
                        assert imseg.NROWS == num_rows_is[imidx]
                    expected_iloc_rows = 0 if imidx == 0 else num_rows_is[imidx - 1]
                    with self.need("ILOC matches expected "):
                        assert int(imseg.ILOC[:5]) == expected_iloc_rows
                        assert int(imseg.ILOC[5:]) == 0
                    with self.need("Matching NCOLS"):
                        assert imseg.NCOLS == num_cols

    def check_nitf_igeolo(self) -> None:
        """Check each NITF image segment's IGEOLO"""
        num_is, _, first_row_is, _ = self._segmentation()
        num_rows = self.xmlhelp.load("./{*}ImageData/{*}NumRows")

        icp_nodes = self.xmlhelp.load("./{*}GeoData/{*}ImageCorners")
        icp_ecef = geocoords.geodetic_to_ecf(
            np.concatenate((icp_nodes, np.zeros((4, 1))), axis=1)
        )

        iscp_ecef = np.zeros((num_is, 4, 3))
        for imidx in range(num_is):
            wgt1 = (num_rows - 1 - first_row_is[imidx]) / (num_rows - 1)
            wgt2 = first_row_is[imidx] / (num_rows - 1)
            iscp_ecef[imidx][0] = wgt1 * icp_ecef[0] + wgt2 * icp_ecef[3]
            iscp_ecef[imidx][1] = wgt1 * icp_ecef[1] + wgt2 * icp_ecef[2]

        for imidx in range(num_is - 1):
            iscp_ecef[imidx][2] = iscp_ecef[imidx + 1][1]
            iscp_ecef[imidx][3] = iscp_ecef[imidx + 1][0]
        iscp_ecef[num_is - 1][2] = icp_ecef[2]
        iscp_ecef[num_is - 1][3] = icp_ecef[3]

        iscp_ll = geocoords.ecf_to_geodetic(iscp_ecef)[:, :, :2]  # Lat, Lon

        def _dms_to_dd(dms_str):
            direction = dms_str[-1]
            ss = int(dms_str[-3:-1])
            mm = int(dms_str[-5:-3])
            dd = int(dms_str[:-5])
            sign = 1 if direction in ["E", "N"] else -1
            return sign * (dd + mm / 60 + ss / 60 / 60)

        with self.precondition():
            assert self.ntf is not None
            imsegs = self.ntf.img_headers
            for imidx, imseg in enumerate(imsegs):
                igeolo_ll = [  # Lat, Lon
                    [_dms_to_dd(imseg.IGEOLO[0:7]), _dms_to_dd(imseg.IGEOLO[7:15])],
                    [
                        _dms_to_dd(imseg.IGEOLO[15:22]),
                        _dms_to_dd(imseg.IGEOLO[22:30]),
                    ],
                    [
                        _dms_to_dd(imseg.IGEOLO[30:37]),
                        _dms_to_dd(imseg.IGEOLO[37:45]),
                    ],
                    [
                        _dms_to_dd(imseg.IGEOLO[45:52]),
                        _dms_to_dd(imseg.IGEOLO[52:60]),
                    ],
                ]
                with self.need("IGEOLO close to ICP Lon/Lat"):
                    assert np.allclose(
                        igeolo_ll, iscp_ll[imidx], atol=1.0 / 60 / 60, rtol=0
                    )

    def check_des_subheader(self) -> None:
        """Check NITF DES Subheaders"""
        with self.precondition():
            assert self.ntf is not None
            des_header = self.ntf.parse_des_subheader(0)
            first_des_bytes = self.ntf.get_des_bytes(0)
            xml_in_first_des = etree.tostring(
                etree.fromstring(first_des_bytes), method="c14n"
            )
            this_xml = etree.tostring(self.sicdroot, method="c14n")
            with self.need("XML from first DES matches sicdroot being used"):
                assert xml_in_first_des == this_xml

            with self.need("DESID == XML_DATA_CONTENT"):
                assert des_header.DESID.rstrip() == "XML_DATA_CONTENT"

            with self.need("DESSHFT == XML"):
                assert des_header.UserHeader.DESSHFT.rstrip() == "XML"
                assert (
                    des_header.UserHeader.DESSHSI.rstrip()
                    == "SICD Volume 1 Design & Implementation Description Document"
                )

            instance_namespace = etree.QName(self.sicdroot).namespace
            with self.need("Consistent namespace"):
                assert des_header.UserHeader.DESSHTN.rstrip() == instance_namespace

            icp_nodes = self.xmlhelp.load("./{*}GeoData/{*}ImageCorners")
            icp_strs = [f"{lat:+012.8f}{lon:+013.8f}" for (lat, lon) in icp_nodes]
            icp_strs.append(icp_strs[0])
            with self.need("DESSHLPG consistent with image corners"):
                assert des_header.UserHeader.DESSHLPG == "".join(icp_strs)

    def check_grid_sign(self) -> None:
        """Grid signs match."""
        with self.need("Row and Col grid signs match"):
            assert self.xmlhelp.load("./{*}Grid/{*}Col/{*}Sgn") == self.xmlhelp.load(
                "./{*}Grid/{*}Row/{*}Sgn"
            )

    @per_grid_dim
    def check_uniform_ipr_width(self, grid_dim) -> None:
        """Uniform weighted IPR width matches bandwidth."""
        with self.precondition():
            assert (
                self.xmlhelp.load(
                    f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}WgtType/{{*}}WindowName"
                )
                == "UNIFORM"
            )
            imprespbw = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}ImpRespBW")
            imprespwid = self.xmlhelp.load(
                f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}ImpRespWid"
            )
            with self.need(
                f"Grid/{grid_dim} uniform weighted IPR width matches bandwidth"
            ):
                assert imprespwid == con.Approx(KAPFAC / imprespbw, rtol=1e-4)

    @per_grid_dim
    def check_deltak_wrt_ss(self, grid_dim) -> None:
        """DeltaK[12] must agree with SS."""
        dk1 = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}DeltaK1")
        dk2 = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}DeltaK2")
        ss = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}SS")
        with self.need(f"{grid_dim} DeltaK2 >= DeltaK1"):
            assert dk2 >= con.Approx(dk1)
        with self.need(f"{grid_dim} DeltaKs must agree with SS"):
            assert dk2 <= con.Approx(0.5 / ss)
            assert dk1 >= con.Approx(-0.5 / ss)

    @per_grid_dim
    def check_iprbw_to_deltak(self, grid_dim) -> None:
        """ImpRespBW <= DeltaK2 - DeltaK1."""
        imprespbw = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}ImpRespBW")
        dk1 = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}DeltaK1")
        dk2 = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}DeltaK2")
        with self.need(f"{grid_dim} ImpRespBW <= DeltaK2 - DeltaK1"):
            assert imprespbw <= con.Approx(dk2 - dk1)

    @per_grid_dim
    def check_iprbw_to_ss(self, grid_dim) -> None:
        """Impulse Response BW is supported by sample spacing."""
        imprespbw = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}ImpRespBW")
        ss = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}SS")
        with self.need(
            f"{grid_dim} Impulse Response BW is supported by sample spacing"
        ):
            assert imprespbw <= con.Approx(1.0 / ss)

    @per_grid_dim
    def check_iprbw_to_ss_osr(self, grid_dim) -> None:
        """Oversample ratio [1/(Grid//ImpRespBW * Grid//SS)] is between 1.1 and 2.2"""
        imprespbw = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}ImpRespBW")
        ss = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}SS")
        osr = 1.0 / imprespbw / ss
        with self.want(f"{grid_dim} OSR <= 2.2"):
            assert osr <= con.Approx(2.2)
        with self.want(f"{grid_dim} OSR >= 1.1"):
            assert osr >= con.Approx(1.1)

    def _compute_deltaks_from_poly(self, direction, vertices):
        """Computes DeltaK1 and DeltaK2 from other Grid information."""
        ipr_bw = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{direction}/{{*}}ImpRespBW")
        dir_spacing = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{direction}/{{*}}SS")

        dkcoapoly = self.xmlhelp.load(
            f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaKCOAPoly"
        )
        if dkcoapoly is not None:
            points = _grid_index_to_coord(self.xmlhelp, vertices)

            deltaks = npp.polyval2d(points[:, 0], points[:, 1], dkcoapoly)
            min_dk = deltaks.min()
            max_dk = deltaks.max()
        else:
            min_dk = 0
            max_dk = 0

        dk1_comp = min_dk - (ipr_bw / 2.0)
        dk2_comp = max_dk + (ipr_bw / 2.0)

        # Handle Wrapped spectrum
        if dk1_comp < -0.5 / dir_spacing or dk2_comp > 0.5 / dir_spacing:
            dk1_comp = -0.5 / dir_spacing
            dk2_comp = -dk1_comp

        return dk1_comp, dk2_comp

    @per_grid_dim
    def check_deltakpoly(self, grid_dim) -> None:
        """DeltaKPoly matches DeltaK1."""
        vertices = _get_valid_data_vertices(self.xmlhelp)
        dk1_comp, dk2_comp = self._compute_deltaks_from_poly(grid_dim, vertices)

        tolerance = 1e-2
        with self.need(f"{grid_dim} DeltaK1 matches computed value from the poly"):
            assert self.xmlhelp.load(
                f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}DeltaK1"
            ) == con.Approx(dk1_comp, atol=tolerance)
        with self.need(f"{grid_dim} DeltaK2 matches computed value from the poly"):
            assert self.xmlhelp.load(
                f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}DeltaK2"
            ) == con.Approx(dk2_comp, atol=tolerance)

    def _compare_size_and_index(self, path_to_parent, rel_path_to_child):
        """Helper method for making sure nodes with ``'index'`` attributes
        are consistent with their parent's ``'size'``.
        """
        parent = self.sicdroot.find(path_to_parent)
        indices = None
        if parent is not None:
            indices = [
                int(node.get("index")) for node in parent.findall(rel_path_to_child)
            ]
            with self.need(f"All {rel_path_to_child} elements are present"):
                assert not set(indices).symmetric_difference(range(1, len(indices) + 1))
            with self.need(
                f"{path_to_parent} size attribute matches number of {rel_path_to_child}"
            ):
                assert int(parent.attrib["size"]) == len(indices)

        return indices

    @per_grid_dim
    def check_wgtfunct_indices(self, grid_dim) -> None:
        """Checks consistency of the indices in the WgtFunct elements."""
        wgt_indices = self._compare_size_and_index(
            f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}WgtFunct", "./{*}Wgt"
        )
        with self.precondition():
            assert wgt_indices is not None
            with self.need(
                f"{grid_dim} WgtFunct elements have size attribute and Wgt children have index attribute"
            ):
                assert wgt_indices

    def check_valid_ifa(self) -> None:
        """ImageFormationAlgo must be paired with appropriate block."""
        with self.precondition():
            assert (
                algo := self.sicdroot.findtext("./{*}ImageFormation/{*}ImageFormAlgo")
            ) != "OTHER"
            with self.need("ImageFormationAlgo paired with appropriate block"):
                if algo == "RGAZCOMP":
                    algo = "RgAzComp"
                assert self.sicdroot.find("./{*}" + algo) is not None

    def check_pfa_grid_type(self) -> None:
        """PFA has grid type of RGAZIM."""
        with self.precondition():
            assert self.sicdroot.find("./{*}PFA") is not None
            with self.need("PFA grid type must be RGAZIM"):
                assert self.sicdroot.find("./{*}Grid/{*}Type").text == "RGAZIM"

    def check_pfa_spot_kaz_to_grid(self) -> None:
        """PFA Kaz within half of 1/Grid.Col.SS of KCtr."""
        with self.precondition():
            assert self.sicdroot.find("./{*}PFA") is not None
            assert (
                self.sicdroot.find("./{*}CollectionInfo/{*}RadarMode/{*}ModeType").text
                == "SPOTLIGHT"
            )
            kaz1 = self.xmlhelp.load("./{*}PFA/{*}Kaz1")
            kaz2 = self.xmlhelp.load("./{*}PFA/{*}Kaz2")
            col_kctr = self.xmlhelp.load("./{*}Grid/{*}Col/{*}KCtr")
            col_ss = self.xmlhelp.load("./{*}Grid/{*}Col/{*}SS")
            with self.want("PFA Kaz1 within half of 1/Grid.Col.SS of KCtr"):
                assert kaz1 - col_kctr >= con.Approx(-0.5 / col_ss)
            with self.want("PFA Kaz2 within half of 1/Grid.Col.SS of KCtr"):
                assert kaz2 - col_kctr <= con.Approx(0.5 / col_ss)

    def check_pfa_krg_to_grid(self) -> None:
        """PFA Krg within half of 1/Grid.Row.SS of KCtr."""
        with self.precondition():
            assert self.sicdroot.find("./{*}PFA") is not None
            krg1 = self.xmlhelp.load("./{*}PFA/{*}Krg1")
            krg2 = self.xmlhelp.load("./{*}PFA/{*}Krg2")
            row_kctr = self.xmlhelp.load("./{*}Grid/{*}Row/{*}KCtr")
            row_ss = self.xmlhelp.load("./{*}Grid/{*}Row/{*}SS")
            with self.want("PFA Krg1 within half of 1/Grid.Row.SS of KCtr"):
                assert krg1 - row_kctr >= con.Approx(-0.5 / row_ss)
            with self.want("PFA Krg2 within half of 1/Grid.Row.SS of KCtr"):
                assert krg2 - row_kctr <= con.Approx(0.5 / row_ss)

    @per_grid_dim
    def check_pfa_ipr_bw(self, grid_dim) -> None:
        """IPR bandwidth supported by PFA spatial frequency extent."""
        with self.precondition():
            assert self.sicdroot.find("./{*}PFA") is not None
            imprespbw = self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}ImpRespBW")
            kpar = {"Row": "Krg", "Col": "Kaz"}[grid_dim]
            k1 = self.xmlhelp.load(f"./{{*}}PFA/{{*}}{kpar}1")
            k2 = self.xmlhelp.load(f"./{{*}}PFA/{{*}}{kpar}2")
            with self.need(f"{grid_dim} IPR bandwidth supported by {kpar}"):
                assert imprespbw <= con.Approx(k2 - k1)

    def check_pfa_stds_kcoa(self) -> None:
        """Checks that the PFA/STDeskew/STDSPhasePoly is nominally correct."""
        stds_poly = self.xmlhelp.load("./{*}PFA/{*}STDeskew/{*}STDSPhasePoly")
        delta_krow_poly = self.xmlhelp.load("./{*}Grid/{*}Row/{*}DeltaKCOAPoly")
        delta_kcol_poly = self.xmlhelp.load("./{*}Grid/{*}Col/{*}DeltaKCOAPoly")
        with self.precondition():
            assert stds_poly is not None
            assert delta_krow_poly is not None
            assert delta_kcol_poly is not None
            grid = _sample_valid_pixels(self.xmlhelp)
            image_coords = _grid_index_to_coord(self.xmlhelp, grid)
            row_bw = self.xmlhelp.load("./{*}Grid/{*}Row/{*}ImpRespBW")
            col_bw = self.xmlhelp.load("./{*}Grid/{*}Col/{*}ImpRespBW")
            sgn = self.xmlhelp.load("./{*}Grid/{*}Row/{*}Sgn")

            if self.xmlhelp.load("./{*}PFA/{*}STDeskew/{*}Applied"):
                with self.want(
                    "Col/DeltaKCOAPoly gives nearly zero when Deskew is applied"
                ):
                    assert npp.polyval2d(
                        image_coords[..., 0], image_coords[..., 1], delta_kcol_poly
                    ) == con.Approx(0.0, atol=col_bw * 1e-3)
                with self.want("STDS derived DeltaKrow matches Row/DeltaKCOAPoly"):
                    delta_krow_poly_stds = -sgn * npp.polyder(stds_poly, axis=0)
                    assert npp.polyval2d(
                        image_coords[..., 0], image_coords[..., 1], delta_krow_poly
                    ) == con.Approx(
                        npp.polyval2d(
                            image_coords[..., 0],
                            image_coords[..., 1],
                            delta_krow_poly_stds,
                        ),
                        atol=row_bw * 1e-3,
                    )
            else:
                with self.want(
                    "Row/DeltaKCOAPoly gives nearly zero when Deskew is not applied"
                ):
                    assert npp.polyval2d(
                        image_coords[..., 0], image_coords[..., 1], delta_krow_poly
                    ) == con.Approx(0.0, atol=row_bw * 1e-3)
                with self.want("STDS derived DeltaKcol matches Col/DeltaKCOAPoly"):
                    delta_kcol_poly_stds = sgn * npp.polyder(stds_poly, axis=1)
                    assert npp.polyval2d(
                        image_coords[..., 0], image_coords[..., 1], delta_kcol_poly
                    ) == con.Approx(
                        npp.polyval2d(
                            image_coords[..., 0],
                            image_coords[..., 1],
                            delta_kcol_poly_stds,
                        ),
                        atol=col_bw * 1e-3,
                    )

    def check_pfa_polar_ang_poly(self) -> None:
        """Polar angle polynomial evaluates to 0 at reference time."""
        with self.precondition():
            assert self.sicdroot.find("./{*}PFA") is not None
            polar_ang_poly = self.xmlhelp.load("./{*}PFA/{*}PolarAngPoly")
            polar_ang_ref_time = self.xmlhelp.load("./{*}PFA/{*}PolarAngRefTime")
            polar_angle_ref = npp.polyval(polar_ang_ref_time, polar_ang_poly)
            with self.need("Polar angle reference is zero at the reference time"):
                assert polar_angle_ref == con.Approx(0.0, atol=1e-4)

    def check_scp_ecf_llh(self) -> None:
        """SCP ECF and LLH positions match."""
        computed_ecf = geocoords.geodetic_to_ecf(
            self.xmlhelp.load("./{*}GeoData/{*}SCP/{*}LLH")
        )
        scp_ecf = self.xmlhelp.load("./{*}GeoData/{*}SCP/{*}ECF")
        diff = npl.norm(scp_ecf - computed_ecf)
        with self.need("SCP ECF and LLH positions match"):
            assert diff <= con.Approx(1)

    def check_image_corners(self) -> None:
        """Checks that the image corner points (ICPs) are nominally correct."""
        icp_nodes = self.xmlhelp.load("./{*}GeoData/{*}ImageCorners")
        with self.need("Number of ICP is four"):
            assert len(icp_nodes) == 4

        scp_ecf = self.xmlhelp.load("./{*}GeoData/{*}SCP/{*}ECF")
        scp_height = self.xmlhelp.load("./{*}GeoData/{*}SCP/{*}LLH/{*}HAE")

        nrows = self.xmlhelp.load("./{*}ImageData/{*}NumRows")
        ncols = self.xmlhelp.load("./{*}ImageData/{*}NumCols")
        row = [0, 0, nrows - 1, nrows - 1]
        col = [0, ncols - 1, ncols - 1, 0]
        icp_coords = _grid_index_to_coord(self.xmlhelp, np.stack([row, col], axis=-1))
        urow_gnd, ucol_gnd = _uvecs_in_ground(self.xmlhelp)

        for (lat, lon), icp_coord in zip(icp_nodes, icp_coords):
            icp_converted_ecf = geocoords.geodetic_to_ecf(
                [
                    lat,
                    lon,
                    scp_height,
                ],
            )
            icp_computed_ecf = (
                scp_ecf + urow_gnd * icp_coord[0] + ucol_gnd * icp_coord[1]
            )

            scp_dist = npl.norm(icp_converted_ecf - scp_ecf)
            with self.need(f"Image corner {icp_coord} must align with ImageData"):
                assert 0.1 * scp_dist > con.Approx(
                    npl.norm(icp_converted_ecf - icp_computed_ecf)
                )

    def check_validdata_presence(self) -> None:
        """ValidData should be in both GeoData and ImageData or neither."""
        in_geodata = self.sicdroot.find("./{*}GeoData/{*}ValidData") is not None
        in_imagedata = self.sicdroot.find("./{*}ImageData/{*}ValidData") is not None
        with self.need("ValidData in both GeoData and ImageData or neither"):
            assert in_geodata == in_imagedata

    def check_validdata_first_vertex(self) -> None:
        """First ValidData Vertex should be min row -> min col."""
        vertices = _get_valid_data_vertices(self.xmlhelp)
        row = [vertex[0] for vertex in vertices]
        col = [vertex[1] for vertex in vertices if vertex[0] == min(row)]
        with self.need("First ValidData Vertex is min row -> min col"):
            assert np.array_equal(
                (min(row), min(col)), (vertices[0][0], vertices[0][1])
            )

    def check_validdata_winding(self) -> None:
        """ValidData should be clockwise."""
        validdata = shg.Polygon(_get_valid_data_vertices(self.xmlhelp))
        with self.need("Clockwise ValidData"):
            assert not validdata.exterior.is_ccw

    def check_validdata_simpleness(self) -> None:
        """ValidData should be a simple polygon."""
        validdata = shg.Polygon(_get_valid_data_vertices(self.xmlhelp))
        with self.need("Simple ValidData"):
            assert validdata.exterior.is_simple

    def _away_from_earth(self, vector):
        """Helper to check if a vector points away from earth."""
        scp_llh = self.xmlhelp.load("./{*}GeoData/{*}SCP/{*}LLH")
        local_up = geocoords.up(scp_llh)

        with self.need("Vector points away from earth"):
            assert np.dot(local_up, vector) > con.Approx(0)

    def check_grid_normal_away_from_earth(self) -> None:
        """ "Normal to Grid unit vectors points away from earth."""
        self._away_from_earth(
            np.cross(
                self.xmlhelp.load("./{*}Grid/{*}Row/{*}UVectECF"),
                self.xmlhelp.load("./{*}Grid/{*}Col/{*}UVectECF"),
            )
        )

    def check_grid_shadows_downward(self) -> None:
        """Grid should indicate that shadows are downward. Taken to mean more downward than leftward or rightward."""
        row_uvect = self.xmlhelp.load("./{*}Grid/{*}Row/{*}UVectECF")
        col_uvect = self.xmlhelp.load("./{*}Grid/{*}Col/{*}UVectECF")
        los_to_scp = self.xmlhelp.load(
            "./{*}GeoData/{*}SCP/{*}ECF"
        ) - self.xmlhelp.load("./{*}SCPCOA/{*}ARPPos")
        with self.need("Grid indicates that shadows are downward"):
            assert np.dot(row_uvect, los_to_scp) > np.abs(np.dot(col_uvect, los_to_scp))

    @per_grid_dim
    def check_grid_unit_vector(self, grid_dim) -> None:
        """Unit vector must have unit magnitude."""
        with self.need(f"{grid_dim} unit vector have unit magnitude"):
            assert npl.norm(
                self.xmlhelp.load(f"./{{*}}Grid/{{*}}{grid_dim}/{{*}}UVectECF")
            ) == con.Approx(1.0)

    def check_grid_uvect_orthogonal(self) -> None:
        """Checks the grid unit vectors are orthogonal to within 1 milliradian."""
        row_uvect = _unit(self.xmlhelp.load("./{*}Grid/{*}Row/{*}UVectECF"))
        col_uvect = _unit(self.xmlhelp.load("./{*}Grid/{*}Col/{*}UVectECF"))
        with self.want("Grid unit vectors are orthogonal"):
            assert np.dot(row_uvect, col_uvect) == con.Approx(0.0, atol=np.sin(1.0e-3))

    def check_pfa_fpn_away_from_earth(self) -> None:
        """FPN points away from Earth."""
        with self.precondition():
            assert self.sicdroot.find("./{*}PFA") is not None
            fpn = self.xmlhelp.load("./{*}PFA/{*}FPN")
            self._away_from_earth(fpn)

    def check_pfa_ipn_away_from_earth(self) -> None:
        """IPN points away from Earth."""
        with self.precondition():
            assert self.sicdroot.find("./{*}PFA") is not None
            ipn = self.xmlhelp.load("./{*}PFA/{*}IPN")
            self._away_from_earth(ipn)

    def check_pfa_ipn_with_grid(self) -> None:
        """PFA IPN is normal to grid."""
        with self.precondition():
            assert self.sicdroot.find("./{*}PFA") is not None
            ipn = self.xmlhelp.load("./{*}PFA/{*}IPN")
            unit_ipn = _unit(ipn)

            computed_ipn = _unit(
                np.cross(
                    self.xmlhelp.load("./{*}Grid/{*}Row/{*}UVectECF"),
                    self.xmlhelp.load("./{*}Grid/{*}Col/{*}UVectECF"),
                )
            )

            with self.need("PFA IPN is consistent with the Grid"):
                assert np.dot(unit_ipn, computed_ipn) == con.Approx(1.0)

    def check_pfa_proc_freq(self) -> None:
        """Processed Frequency matches PFA inscription."""
        with self.precondition():
            assert self.sicdroot.find("./{*}PFA") is not None
            min_fx, max_fx = _compute_pfa_min_max_fx(self.xmlhelp)

            min_proc = self.xmlhelp.load(
                "./{*}ImageFormation/{*}TxFrequencyProc/{*}MinProc"
            )
            max_proc = self.xmlhelp.load(
                "./{*}ImageFormation/{*}TxFrequencyProc/{*}MaxProc"
            )

            bw_tolerance = 0.1 * (max_proc - min_proc)
            with self.want("Minimum Fx should be >= MinProc frequency"):
                assert min_fx >= con.Approx(min_proc, atol=bw_tolerance, rtol=1e-3)
            with self.want("Maximum Fx should be <= MaxProc frequency"):
                assert max_fx <= con.Approx(max_proc, atol=bw_tolerance, rtol=1e-3)

    def check_proc_freq(self) -> None:
        """ImageFormation/TxFrequencyProc lies within RadarCollection/TxFrequency."""
        min_proc = self.xmlhelp.load(
            "./{*}ImageFormation/{*}TxFrequencyProc/{*}MinProc"
        )
        max_proc = self.xmlhelp.load(
            "./{*}ImageFormation/{*}TxFrequencyProc/{*}MaxProc"
        )
        min_coll = self.xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Min")
        max_coll = self.xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Max")

        with self.want("Min processed frequency >= min collected frequency"):
            assert min_proc >= con.Approx(min_coll)
        with self.want("Max processed frequency <= max collected frequency"):
            assert max_proc <= con.Approx(max_coll)

        with self.precondition():
            assert self.sicdroot.find("./{*}RadarCollection/{*}RefFreqIndex") is None
            with self.need("Min collected frequency > 0.0"):
                assert min_coll > 0.0
        with self.need("Max collected frequency > min collected"):
            assert max_coll > con.Approx(min_coll)
        with self.need("Max processed frequency > min processed"):
            assert max_proc > con.Approx(min_proc)

    def check_inca(self) -> None:
        """Checks that RMA/INCA parameters are consistent with other metadata."""
        inca = self.sicdroot.find("./{*}RMA/{*}INCA")
        with self.precondition():
            assert inca is not None
            time_ca_poly = self.xmlhelp.load("./{*}RMA/{*}INCA/{*}TimeCAPoly")
            arp_poly = self.xmlhelp.load("./{*}Position/{*}ARPPoly")
            scp_time_ca = npp.polyval(0.0, time_ca_poly)
            arp_pos = npp.polyval(scp_time_ca, arp_poly)
            arp_vel = npp.polyval(scp_time_ca, npp.polyder(arp_poly))
            scp_pos = self.xmlhelp.load("./{*}GeoData/{*}SCP/{*}ECF")
            scp_range_ca = npl.norm(arp_pos - scp_pos)
            los_arp_to_scp = scp_pos - arp_pos
            along_track_error = np.abs(np.dot(los_arp_to_scp, _unit(arp_vel)))
            grid = _sample_valid_pixels(self.xmlhelp)
            coords = _grid_index_to_coord(self.xmlhelp, grid)
            time_ca = npp.polyval(coords[..., 1], time_ca_poly)
            ca_vel = 1.0 / np.abs(
                npp.polyval(coords[..., 1], npp.polyder(time_ca_poly))
            )
            arp_speed_ca = npl.norm(npp.polyval(time_ca, npp.polyder(arp_poly)), axis=0)
            drsf_poly = self.xmlhelp.load("./{*}RMA/{*}INCA/{*}DRateSFPoly")
            drsf = npp.polyval2d(coords[..., 0], coords[..., 1], drsf_poly)
            along_track_error_thresh = 2e-2 / self.xmlhelp.load(
                "./{*}Grid/{*}Col/{*}ImpRespBW"
            )
            with self.want("Along track error is relatively small"):
                assert along_track_error <= con.Approx(along_track_error_thresh)

            r_ca_scp = self.xmlhelp.load("./{*}RMA/{*}INCA/{*}R_CA_SCP")
            with self.want("Calculated SCP range ca matches the metadata"):
                assert scp_range_ca == con.Approx(r_ca_scp)

            with self.want(
                "Product of ARP speed and scale factor matches calculated ca_vel"
            ):
                assert arp_speed_ca * drsf == con.Approx(ca_vel, rtol=1e-2)

    def check_segmentlist_bounds(self) -> None:
        """Checks that segments within the segment_list are bounded by the area plane."""

        def bounded_by(segment, area):
            return area.intersection(segment).equals(segment)

        plane = self.sicdroot.find("./{*}RadarCollection/{*}Area/{*}Plane")
        with self.precondition():
            assert plane is not None
            # create the line/sample polygon for the area plane
            area_polygon = _create_rectangle(
                int(plane.findtext("./{*}XDir/{*}FirstLine")),
                int(plane.findtext("./{*}YDir/{*}FirstSample")),
                int(plane.findtext("./{*}XDir/{*}NumLines")),
                int(plane.findtext("./{*}YDir/{*}NumSamples")),
            )

            list_of_segs = plane.findall("./{*}SegmentList/{*}Segment")
            with self.precondition():
                assert list_of_segs != []
                # create a polygon for each segment
                for seg in list_of_segs:
                    segment_polygon = shg.box(
                        int(seg.findtext("StartLine")),
                        int(seg.findtext("StartSample")),
                        int(seg.findtext("EndLine")),
                        int(seg.findtext("EndSample")),
                        ccw=False,
                    )
                    with self.need(
                        "All segments within the segment_list are bounded by the area plane"
                    ):
                        assert bounded_by(segment_polygon, area_polygon)

    def check_segment_identifier(self) -> None:
        """Checks that segment identifier references a segment in the SegmentList."""
        # check if segment lists are included
        segment_list = self.sicdroot.find(
            "./{*}RadarCollection/{*}Area/{*}Plane/{*}SegmentList"
        )
        if segment_list is not None:
            # then segment identifier must also be included
            seg_id = self.sicdroot.findtext("./{*}ImageFormation/{*}SegmentIdentifier")
            with self.need("SegmentIdentifier is included"):
                assert seg_id is not None

            # and reference a segment within the list
            with self.need("SegmentList has SegmentIdentifier"):
                assert (
                    segment_list.find(f'./{{*}}Segment[Identifier="{seg_id}"]')
                    is not None
                )
        else:
            # segment list not present - segment identifier shouldn't be included
            seg_id = self.sicdroot.findtext("./{*}ImageFormation/{*}SegmentIdentifier")
            with self.need("SegmentIdentifier not present without SegmentList"):
                assert seg_id is None

    def check_segmentlist_indices(self) -> None:
        """Checks that SegmentList has a size attribute and segments have the index attribute."""
        segment_indices = self._compare_size_and_index(
            "./{*}RadarCollection/{*}Area/{*}Plane/{*}SegmentList", "./{*}Segment"
        )
        with self.precondition():
            assert segment_indices is not None
            with self.need(
                "SegmentList has size attribute and segments have index attribute"
            ):
                assert segment_indices

    def check_segment_unique_ids(self) -> None:
        """Checks that identifiers in SegmentList are unique."""
        segment_list = self.sicdroot.find(
            "./{*}RadarCollection/{*}Area/{*}Plane/{*}SegmentList"
        )
        with self.precondition():
            assert segment_list is not None
            segment_ids = [
                segment_id.text
                for segment_id in segment_list.findall("./{*}Segment/{*}Identifier")
            ]
            with self.need("SegmentList segments have identifiers"):
                assert len(set(segment_ids)) == len(segment_ids)

    def _compute_area_plane_corners_ecef(self):
        plane = self.sicdroot.find("./{*}RadarCollection/{*}Area/{*}Plane")
        ref_ecf = self.xmlhelp.load_elem(plane.find("./{*}RefPt/{*}ECF"))
        xdir = self.xmlhelp.load_elem(plane.find("./{*}XDir/{*}UVectECF"))
        ydir = self.xmlhelp.load_elem(plane.find("./{*}YDir/{*}UVectECF"))

        # create the line/sample polygon for the area plane
        ref_area_ls = _create_rectangle(
            self.xmlhelp.load_elem(plane.find("./{*}XDir/{*}FirstLine")),
            self.xmlhelp.load_elem(plane.find("./{*}YDir/{*}FirstSample")),
            self.xmlhelp.load_elem(plane.find("./{*}XDir/{*}NumLines")),
            self.xmlhelp.load_elem(plane.find("./{*}YDir/{*}NumSamples")),
        )

        # convert display index into area plane x, y coordinates
        xy_coord = _display_index_to_coord(
            self.xmlhelp, ref_area_ls.exterior.coords[:-1]
        )

        return (
            ref_ecf
            + xy_coord[:, 0][:, np.newaxis] * xdir
            + xy_coord[:, 1][:, np.newaxis] * ydir
        )

    def check_area_corners(self) -> None:
        """Checks that area corners (ACPs) are nominally correct."""
        area = self.sicdroot.find("./{*}RadarCollection/{*}Area")
        with self.precondition():
            assert area is not None

            indices = [
                int(acp.attrib["index"]) for acp in area.findall("./{*}Corner/{*}ACP")
            ]
            with self.need("ACPs must have 4 indices"):
                assert np.array_equal(np.sort(indices), np.arange(1, 5))

            acps_llh = self.xmlhelp.load_elem(area.find("./{*}Corner"))
            acps_ecf = geocoords.geodetic_to_ecf(acps_llh)
            u_east = geocoords.east(acps_llh[1])
            u_north = geocoords.north(acps_llh[1])

            vertices = [
                (np.dot(v, u_east), np.dot(v, u_north))
                for v in acps_ecf - acps_ecf[0, :]
            ]
            ew_ring = shg.LinearRing(vertices)
            with self.need("ACPs must be clockwise and simple"):
                assert not ew_ring.is_ccw
                assert ew_ring.is_simple

            area_plane = area.find("./{*}Plane")
            with self.precondition():
                assert area_plane is not None
                ref_pt = self.xmlhelp.load_elem(area_plane.find("./{*}RefPt/{*}ECF"))

                acps_ecf_computed = self._compute_area_plane_corners_ecef()
                ref_pt_dist = npl.norm(acps_ecf - ref_pt)
                with self.need("ACPs must be within the area plane"):
                    assert 0.1 * ref_pt_dist > con.Approx(
                        npl.norm(acps_ecf - acps_ecf_computed)
                    )

    def check_area_plane_valid(self) -> None:
        """Checks that area plane surface intersects with grid plane area of support."""
        plane = self.sicdroot.find("./{*}RadarCollection/{*}Area/{*}Plane")
        with self.precondition():
            assert plane is not None
            # compute scene center and slant plane normal
            proj_metadata = ss_proj.MetadataParams.from_xml(self.sicdroot.getroottree())
            spn = ss_proj.compute_scp_coa_slant_plane_normal(proj_metadata)

            # extract the reference plane
            ref_ecf = self.xmlhelp.load_elem(plane.find("./{*}RefPt/{*}ECF"))
            xdir = _unit(self.xmlhelp.load_elem(plane.find("./{*}XDir/{*}UVectECF")))
            ydir = _unit(self.xmlhelp.load_elem(plane.find("./{*}YDir/{*}UVectECF")))
            zdir = _unit(np.cross(xdir, ydir))

            scp_ecf = self.xmlhelp.load("./{*}GeoData/{*}SCP/{*}ECF")
            if np.abs(np.dot(scp_ecf - ref_ecf, zdir)) > 0.1:
                gcp, _, success = ss_std.image_to_ground_plane(
                    self.sicdroot.getroottree(), [0, 0], ref_ecf, zdir
                )
                assert success
                u_proj = _unit(scp_ecf - gcp)
            else:
                u_proj = spn

            # project display coordinate into slant/grid plane
            gpp = self._compute_area_plane_corners_ecef()
            dist = np.dot(scp_ecf - gpp, spn) / np.dot(u_proj, spn)
            ipp = gpp + dist[:, np.newaxis] * u_proj

            # convert into grid (xrow, ycol) coordinates
            ipx = _unit(self.xmlhelp.load("./{*}Grid/{*}Row/{*}UVectECF"))
            ipy = _unit(self.xmlhelp.load("./{*}Grid/{*}Col/{*}UVectECF"))
            vertices_spxy = np.column_stack(
                (np.dot(ipp - scp_ecf, ipx), np.dot(ipp - scp_ecf, ipy))
            )

            # fetch the valid area and convert into grid (xrow, ycol) coordinates
            grid = _get_valid_data_vertices(self.xmlhelp)
            valid_area_spxy = _grid_index_to_coord(self.xmlhelp, grid)
            with self.need("Area plane surface intersects with grid plane area"):
                assert shg.Polygon(vertices_spxy).intersects(
                    shg.Polygon(valid_area_spxy)
                )

    def check_ipp_poly(self):
        """Checks that the IPPPolys are nominally correct."""
        ipp_sets = self.sicdroot.findall("./{*}Timeline/{*}IPP/{*}Set")
        isets = []
        for ipp_set in ipp_sets:
            ipp_poly_node = ipp_set.find("./{*}IPPPoly")
            iset = {
                "t_start": float(ipp_set.findtext("./{*}TStart")),
                "t_end": float(ipp_set.findtext("./{*}TEnd")),
                "ipp_start": int(ipp_set.findtext("./{*}IPPStart")),
                "ipp_end": int(ipp_set.findtext("./{*}IPPEnd")),
                "ipp_poly": self.xmlhelp.load_elem(ipp_poly_node),
                "index": int(ipp_set.get("index")),
            }
            with self.need("TEnd greater than TStart"):
                assert iset["t_end"] > iset["t_start"]
            with self.need("IPPEnd greater than IPPStart"):
                assert iset["ipp_end"] > iset["ipp_start"]
            with self.need("IPPStart is close to the polynomial evaluation at TStart"):
                assert np.allclose(
                    iset["ipp_start"],
                    np.round(npp.polyval(iset["t_start"], iset["ipp_poly"])),
                )
            with self.need("IPPEnd is close to the polynomial evaluation at TEnd"):
                assert np.allclose(
                    iset["ipp_end"],
                    np.round(npp.polyval(iset["t_end"], iset["ipp_poly"]) - 1),
                )
            isets.append(iset)
        isets.sort(key=lambda x: x["index"])

        t_starts = np.asarray([iset["t_start"] for iset in isets])
        min_time = np.min(t_starts)
        t_ends = np.asarray([iset["t_end"] for iset in isets])
        max_time = np.max(t_ends)
        # SICD Vol. 1 says "the IPP sequence spans the collection" which is impractical
        # because the CollectStart may be before imaging occurs
        t_start_proc = self.xmlhelp.load("./{*}ImageFormation/{*}TStartProc")
        t_end_proc = self.xmlhelp.load("./{*}ImageFormation/{*}TEndProc")
        with self.need("min(IPP.Set.TStart) <= TStartProc"):
            assert min_time <= con.Approx(t_start_proc, atol=1e-2)
        with self.need("max(IPP.Set.TEnd) >= TEndProc"):
            assert max_time >= con.Approx(t_end_proc, atol=1e-2)
        time_between = t_starts[1:] - t_ends[:-1]
        with self.need("TStart values are increasing"):
            assert np.array_equal(t_starts, sorted(t_starts))
        with self.need("TEnds values are increasing"):
            assert np.array_equal(t_ends, sorted(t_ends))
        with self.need("No time gaps/overlap between IPP Sets"):
            assert np.array_equal(time_between, [0] * time_between.size)
        ipp_starts = np.asarray([iset["ipp_start"] for iset in isets])
        ipp_ends = np.asarray([iset["ipp_end"] for iset in isets])
        ipp_index_between = ipp_starts[1:] - ipp_ends[:-1]
        with self.need("IPPStarts increase"):
            assert np.array_equal(ipp_starts, sorted(ipp_starts))
        with self.need("IPPEnds increase"):
            assert np.array_equal(ipp_ends, sorted(ipp_ends))
        with self.need("No IPP index gaps/overlap between IPP Sets"):
            assert np.array_equal(ipp_index_between, [1] * ipp_index_between.size)

    def check_valid_data_indices(self) -> None:
        """Checks consistency of the values in the ImageData child elements."""
        image_indices = self._compare_size_and_index(
            "./{*}ImageData/{*}ValidData", "./{*}Vertex"
        )
        geo_indices = self._compare_size_and_index(
            "./{*}GeoData/{*}ValidData", "./{*}Vertex"
        )
        with self.precondition():
            assert image_indices is not None
            assert geo_indices is not None
            with self.need("GeoData indices equal to ImageData indices"):
                assert np.array_equal(sorted(geo_indices), sorted(image_indices))

    def check_icp_indices(self) -> None:
        """Checks consistency of the indices in the GeoData ICP elements."""
        indices = sorted(
            x.get("index")
            for x in self.sicdroot.findall("./{*}GeoData/{*}ImageCorners/{*}ICP")
        )
        with self.need("GeoData ICPs indexed correctly"):
            assert indices == ["1:FRFC", "2:FRLC", "3:LRLC", "4:LRFC"]

    def check_collection_duration(self) -> None:
        """Checks consistency of the collection duration in the Timeline element"""
        with self.need("CollectionDuration > zero"):
            assert self.xmlhelp.load("./{*}Timeline/{*}CollectDuration") > 0.0

    def check_ipp_set_indices(self) -> None:
        """Checks consistency of the indices in the Timeline IPP elements."""
        ipp_set_indices = self._compare_size_and_index(
            "./{*}Timeline/{*}IPP", "./{*}Set"
        )
        with self.precondition():
            assert ipp_set_indices is not None
            with self.need(
                "Timeline IPP elements have size attribute and sets have index attribute"
            ):
                assert ipp_set_indices

    def check_waveform_params_indices(self) -> None:
        """Checks consistency of the indices in the RadarCollection Waveform Parameter elements."""
        waveform_indices = self._compare_size_and_index(
            "./{*}RadarCollection/{*}Waveform", "./{*}WFParameters"
        )
        with self.precondition():
            assert waveform_indices is not None
            with self.need(
                "Waveform elements have size attribute and WFParameters have index attribute"
            ):
                assert waveform_indices

    def check_waveform_params(self) -> None:
        """Checks consistency of the values in the RadarCollection Waveform Parameter elements."""
        waveform = self.sicdroot.find("./{*}RadarCollection/{*}Waveform")
        with self.precondition():
            assert waveform is not None
            for wf_params in waveform.findall("./{*}WFParameters"):
                for param in (
                    "TxPulseLength",
                    "TxRFBandwidth",
                    "RcvWindowLength",
                    "ADCSampleRate",
                    "RcvIFBandwidth",
                ):
                    node = wf_params.find(param)
                    with self.precondition():
                        assert node is not None
                        with self.need(f"{param} > zero"):
                            assert float(node.text) > con.Approx(0.0)

                node = wf_params.find("TxFMRate")
                with self.precondition():
                    assert node is not None
                    with self.need("TxFMRate not zero"):
                        assert float(node.text) != 0

    def check_tx_polarization(self) -> None:
        """Checks consistency of the RadarCollection/TxPolarization node."""
        tx_pol = self.sicdroot.find("./{*}RadarCollection/{*}TxPolarization").text
        tx_seq_node = self.sicdroot.find("./{*}RadarCollection/{*}TxSequence")
        with self.precondition():
            assert tx_pol == "SEQUENCE"
            with self.want(
                "RadarCollection TxSequence is not none if TxPolarization is SEQUENCE"
            ):
                assert tx_seq_node is not None

    def check_tx_rf_bandwidth(self) -> None:
        """Checks consistency of the RadarCollection/Waveform/WFParameters/TxRFBandwidth values."""
        min_coll = self.xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Min")
        max_coll = self.xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Max")
        atol = max(0.1 * abs(max_coll - min_coll), 1e6)
        for tx_rf_bandwidth_elem in self.sicdroot.findall(
            "./{*}RadarCollection/{*}Waveform/{*}WFParameters/{*}TxRFBandwidth"
        ):
            waveform_tx_bw = float(tx_rf_bandwidth_elem.text)
            with self.need("Waveform TxBW must be within the collected BW"):
                assert waveform_tx_bw <= con.Approx(max_coll - min_coll)

        for wf_parameters in self.sicdroot.findall(
            "./{*}RadarCollection/{*}Waveform/{*}WFParameters"
        ):
            with self.precondition():
                assert all(
                    wf_parameters.find(x) is not None
                    for x in (
                        "./{*}TxPulseLength",
                        "./{*}TxRFBandwidth",
                        "./{*}TxFMRate",
                    )
                )
                tx_pulse_length = float(wf_parameters.findtext("./{*}TxPulseLength"))
                tx_rf_bandwidth = float(wf_parameters.findtext("./{*}TxRFBandwidth"))
                tx_fm_rate = float(wf_parameters.findtext("./{*}TxFMRate"))
                if tx_fm_rate == 0:
                    continue
                derived_bw = abs(tx_fm_rate * tx_pulse_length)
                with self.want("Derived BW must be close to the TxRFBW"):
                    assert derived_bw == con.Approx(tx_rf_bandwidth, atol=atol, rtol=0)

    def check_tx_freq_start(self) -> None:
        """Checks consistency of the RadarCollection/Waveform/WFParameters/TxFreqStart values."""
        min_coll = self.xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Min")
        max_coll = self.xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Max")
        atol = max(0.1 * abs(max_coll - min_coll), 1e6)
        tx_freq_start_list = self.sicdroot.findall(
            "./{*}RadarCollection/{*}Waveform/{*}WFParameters/{*}TxFreqStart"
        )
        with self.precondition():
            assert tx_freq_start_list != []
            for tx_freq_start_elem in tx_freq_start_list:
                waveform_tx_freq_start = float(tx_freq_start_elem.text)
                with self.need("Waveform TxFreqStart <= max collected frequency"):
                    assert waveform_tx_freq_start <= con.Approx(max_coll, atol=atol)
                with self.need("Waveform TxFreqStart >= the min collected frequency"):
                    assert waveform_tx_freq_start >= con.Approx(min_coll, atol=atol)

    def check_tx_freq_bounds(self) -> None:
        """Checks consistency of the frequency bounds implied by RadarCollection/Waveform/WFParameters values."""
        min_coll = self.xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Min")
        max_coll = self.xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Max")
        atol = max(0.1 * abs(max_coll - min_coll), 1e6)
        wf_freq_bounds = []
        for wf_parameters in self.sicdroot.findall(
            "./{*}RadarCollection/{*}Waveform/{*}WFParameters"
        ):
            with self.precondition():
                assert all(
                    wf_parameters.find(x) is not None
                    for x in ("./{*}TxPulseLength", "./{*}TxFreqStart", "./{*}TxFMRate")
                )
                tx_pulse_length = float(wf_parameters.findtext("./{*}TxPulseLength"))
                tx_freq_start = float(wf_parameters.findtext("./{*}TxFreqStart"))
                tx_fm_rate = float(wf_parameters.findtext("./{*}TxFMRate"))
                tx_freq_end = tx_freq_start + tx_pulse_length * tx_fm_rate
                wf_freq_bounds.extend([tx_freq_start, tx_freq_end])
                with self.need("Waveform TxFreqEnd <= max collected frequency"):
                    assert tx_freq_end <= con.Approx(max_coll)
                with self.need("Waveform TxFreqEnd >= min collected frequency"):
                    assert tx_freq_end >= con.Approx(min_coll)

        with self.precondition():
            assert wf_freq_bounds
            with self.want(
                "WFParameters min frequency bounds close to min collected frequency"
            ):
                assert min(wf_freq_bounds) == con.Approx(min_coll, atol=atol, rtol=0)
            with self.want(
                "WFParameters max frequency bounds close to max collected frequency"
            ):
                assert max(wf_freq_bounds) == con.Approx(max_coll, atol=atol, rtol=0)

    def check_rcv_fmrate(self) -> None:
        """Checks consistency of the receive FM rate for chirp/stretch demodulation types."""
        waveform = self.sicdroot.find("./{*}RadarCollection/{*}Waveform")
        with self.precondition():
            assert waveform is not None
            for wf_params in waveform.findall("./{*}WFParameters"):
                demod = wf_params.find("RcvDemodType")
                fmrate = wf_params.find("RcvFMRate")
                with self.need(
                    "Consistent receive FM rate for chirp/stretch demodulation types"
                ):
                    if demod is not None and fmrate is not None:
                        if demod.text == "CHIRP":
                            assert float(fmrate.text) == 0
                        else:
                            assert float(fmrate.text) != 0

    def check_rcv_channel_indices(self) -> None:
        """Checks consistency of the values in the RadarCollection RcvChannels elements."""
        rcvchan_indices = self._compare_size_and_index(
            "./{*}RadarCollection/{*}RcvChannels", "./{*}ChanParameters"
        )
        with self.precondition():
            assert rcvchan_indices is not None
            with self.need(
                "RcvChannels elements have size attribute and segments have index attribute"
            ):
                assert rcvchan_indices

    def check_segment_start_and_end(self) -> None:
        """Checks consistency of the values in the SegmentList StartLine and EndLine elements."""
        with self.precondition():
            assert (
                self.sicdroot.find(
                    "./{*}RadarCollection/{*}Area/{*}Plane/{*}SegmentList/{*}Segment"
                )
                is not None
            )
            for node in self.sicdroot.iterfind(
                "./{*}RadarCollection/{*}Area/{*}Plane/{*}SegmentList/{*}Segment"
            ):
                with self.need("SegmentList EndLine >= StartLine"):
                    assert int(node.findtext("./{*}EndLine")) >= con.Approx(
                        int(node.findtext("./{*}StartLine"))
                    )
                with self.need("SegmentList EndSample >= StartSample"):
                    assert int(node.findtext("./{*}EndSample")) >= con.Approx(
                        int(node.findtext("./{*}StartSample"))
                    )

    def check_image_formation_timeline(self) -> None:
        """Checks that the slow time span for data processed to form the image is within collect."""
        t_start_proc = self.xmlhelp.load("./{*}ImageFormation/{*}TStartProc")
        t_end_proc = self.xmlhelp.load("./{*}ImageFormation/{*}TEndProc")
        collect_duration = self.xmlhelp.load("./{*}Timeline/{*}CollectDuration")
        with self.need("0 <= TStartProc < TEndProc <= CollectDuration"):
            assert 0 <= t_start_proc < t_end_proc <= collect_duration

    def check_scpcoa(self) -> None:
        """Checks consistency of the values in the SCPCOA child elements."""
        newroot = copy.deepcopy(self.sicdroot)
        newroot.replace(
            newroot.find(".//{*}SCPCOA"),
            ss_xml.compute_scp_coa(self.sicdroot.getroottree()),
        )
        expected_xmlhelp = ss_xml.XmlHelper(newroot.getroottree())

        def _compare_children(actual_parent, expected_parent):
            with self.need(f"{actual_parent.tag} contains only expected elements"):
                actual_names = [etree.QName(x).localname for x in actual_parent]
                expected_names = [etree.QName(x).localname for x in expected_parent]
                assert actual_names == expected_names
                for actual, expected in zip(
                    actual_parent, expected_parent, strict=True
                ):
                    param = etree.QName(actual).localname
                    assert param == etree.QName(expected).localname
                    if param in ("Bistatic", "TxPlatform", "RcvPlatform"):
                        _compare_children(actual, expected)
                        continue
                    actual_val = self.xmlhelp.load_elem(actual)
                    expected_val = expected_xmlhelp.load_elem(expected)

                    approx_args = {}
                    if "Ang" in param:
                        approx_args["atol"] = 1
                    elif "Time" in param:
                        approx_args["atol"] = 1e-6
                    elif "Pos" in param:
                        approx_args["atol"] = 1e-2
                    elif "Vel" in param:
                        approx_args["atol"] = 1e-3
                    elif "Acc" in param:
                        approx_args["atol"] = 1e-4

                    with self.need(f"SCPCOA/{param} matches defined calculation"):
                        if param == "SideOfTrack":
                            assert actual_val == expected_val
                        else:
                            assert actual_val == con.Approx(expected_val, **approx_args)

        _compare_children(
            self.xmlhelp.element_tree.find("./{*}SCPCOA"),
            expected_xmlhelp.element_tree.find("./{*}SCPCOA"),
        )

    def check_antenna_oneway_apc(self) -> None:
        """Checks that APC polys are provided when one way patterns are used"""
        with self.precondition():
            assert self.sicdroot.find("./{*}Antenna/{*}Tx") is not None
            with self.need("TxAPCPoly must be present when Antenna/Tx is present"):
                assert self.sicdroot.find("./{*}Position/{*}TxAPCPoly") is not None
        with self.precondition():
            assert self.sicdroot.find("./{*}Antenna/{*}Rcv") is not None
            with self.need("RcvAPCPoly must be present when Antenna/Rcv is present"):
                assert (
                    self.sicdroot.find("./{*}Position/{*}RcvAPC/{*}RcvAPCPoly")
                    is not None
                )

    def check_antenna_array_gain_phase(self) -> None:
        """Checks consistency of the gain/phase values in the Antenna Array elements."""
        for ant_param in ("Tx", "Rcv", "TwoWay"):
            antenna = self.sicdroot.find(f"./{{*}}Antenna/{{*}}{ant_param}")
            with self.precondition():
                assert antenna is not None
                gain_poly = self.xmlhelp.load_elem(
                    antenna.find("./{*}Array/{*}GainPoly")
                )
                with self.need(
                    f"Antenna {ant_param} gain poly constant is close to zero"
                ):
                    assert gain_poly[0][0] == con.Approx(0.0)

                phase_poly = self.xmlhelp.load_elem(
                    antenna.find("./{*}Array/{*}PhasePoly")
                )
                with self.need(
                    f"Antenna {ant_param} phase poly constant is close to zero"
                ):
                    assert phase_poly[0][0] == con.Approx(0.0)

    def check_antenna_elem_gain_phase(self) -> None:
        """Checks consistency of the gain/phase values in the Antenna Elem elements."""
        for ant_param in ("Tx", "Rcv", "TwoWay"):
            antenna = self.sicdroot.find(f"./{{*}}Antenna/{{*}}{ant_param}/{{*}}Elem")
            with self.precondition():
                assert antenna is not None
                gain_poly = self.xmlhelp.load_elem(antenna.find("./{*}GainPoly"))
                with self.need(f"{ant_param} elem gain poly constant is close to zero"):
                    assert gain_poly[0][0] == con.Approx(0.0)

                phase_poly = self.xmlhelp.load_elem(antenna.find("./{*}PhasePoly"))
                with self.need(
                    f"{ant_param} elem phase poly constant is close to zero"
                ):
                    assert phase_poly[0][0] == con.Approx(0.0)

    def check_antenna_bspoly_gain(self) -> None:
        """Checks consistency of the values in the Antenna child elements."""
        for ant_param in ("Tx", "Rcv", "TwoWay"):
            antenna = self.sicdroot.find(
                f"./{{*}}Antenna/{{*}}{ant_param}/{{*}}GainBSPoly"
            )
            with self.precondition():
                assert antenna is not None
                gain_bs_poly = self.xmlhelp.load_elem(antenna)
                with self.need(
                    f"Antenna {ant_param} gain BS poly constant is close to zero"
                ):
                    assert gain_bs_poly[0] == con.Approx(0.0)

    def check_error_composite(self) -> None:
        """Checks consistency of the values in the ErrorStatistics CompositeSCP elements."""
        composite = self.sicdroot.find("./{*}ErrorStatistics/{*}CompositeSCP")
        with self.precondition():
            assert composite is not None
            for param in ("Rg", "Az"):
                with self.need(f"CompositeSCP {param} >= 0.0"):
                    assert float(composite.findtext(param)) >= con.Approx(0.0)

            with self.need("CompositeSCP RgAz <= 1.0"):
                rg_az = abs(float(composite.findtext("./{*}RgAz")))
                assert rg_az <= con.Approx(1.0)

    def check_error_components_posvel_stddev(self) -> None:
        """Checks consistency of the values in the ErrorStatistics PosVelErr elements."""
        posvelerr = self.sicdroot.find(
            "./{*}ErrorStatistics/{*}Components/{*}PosVelErr"
        )
        with self.precondition():
            assert posvelerr is not None
            for param in ("P1", "P2", "P3", "V1", "V2", "V3"):
                with self.need(f"PosVelErr {param} >= 0.0"):
                    assert float(posvelerr.findtext(param)) >= con.Approx(0.0)

    def check_error_components_posvel_corr(self) -> None:
        """Checks consistency of the values in the ErrorStatistics CorrCoefs elements."""
        corrcoefs = self.sicdroot.find(
            "./{*}ErrorStatistics/{*}Components/{*}PosVelErr/{*}CorrCoefs"
        )
        with self.precondition():
            assert corrcoefs is not None
            for param in (
                "P1P2",
                "P1P3",
                "P1V1",
                "P1V2",
                "P1V3",
                "P2P3",
                "P2V1",
                "P2V2",
                "P2V3",
                "P3V1",
                "P3V2",
                "P3V3",
                "V1V2",
                "V1V3",
                "V2V3",
            ):
                with self.need(f"CorrCoefs {param} <= 1.0"):
                    corr_coef = abs(float(corrcoefs.findtext(param)))
                    assert corr_coef <= con.Approx(1.0)

    def check_error_radarsensor_rangebias(self) -> None:
        """Checks consistency of the values in the ErrorStatistics RangeBias element."""
        components = self.sicdroot.find("./{*}ErrorStatistics/{*}Components")
        with self.precondition():
            assert components is not None
            with self.need("RangeBias >= 0.0"):
                range_bias = float(components.findtext("./{*}RadarSensor/{*}RangeBias"))
                assert range_bias >= con.Approx(0.0)

    def check_txsequence_indices(self) -> None:
        """Checks consistency of the TxSequence/TxStep indexing."""
        tx_seq_indices = self._compare_size_and_index(
            "./{*}RadarCollection/{*}TxSequence", "./{*}TxStep"
        )
        with self.precondition():
            assert tx_seq_indices is not None
            with self.need(
                "TxSequence elements have size attribute and TxStep have index attribute"
            ):
                assert tx_seq_indices

    def check_txsequence_waveform_index(self) -> None:
        """Checks consistency of WFIndex"""
        wfp_indices = [
            int(node.get("index"))
            for node in self.sicdroot.findall(
                "./{*}RadarCollection/{*}Waveform/{*}WFParameters"
            )
        ]
        wf_index = [
            int(node.text)
            for node in self.sicdroot.findall(
                "./{*}RadarCollection/{*}TxSequence/{*}TxStep/{*}WFIndex"
            )
        ]
        with self.precondition():
            assert wfp_indices != []
            assert wf_index != []
            for index in wf_index:
                with self.need(
                    f"WFIndex {index} must reference a WFParameters {wfp_indices}"
                ):
                    assert index in wfp_indices

    def check_rcvapc_indices(self) -> None:
        """Checks consistency of the RcvAPC indexing."""
        rcv_apc_indices = self._compare_size_and_index(
            "./{*}Position/{*}RcvAPC", "./{*}RcvAPCPoly"
        )
        with self.precondition():
            assert rcv_apc_indices is not None
            with self.need(
                "RcvAPCPoly elements have size attribute and RcvAPCPoly have index attribute"
            ):
                assert rcv_apc_indices

    def check_rcvapcindex(self) -> None:
        """Checks consistency of RcvAPCIndex."""
        rcvapc_indices = [
            int(node.get("index"))
            for node in self.sicdroot.findall("./{*}Position/{*}RcvAPC/{*}RcvAPCPoly")
        ]
        rcvapcindex = [
            int(node.text)
            for node in self.sicdroot.findall(
                "./{*}RadarCollection/{*}RcvChannels/{*}ChanParameters/{*}RcvAPCIndex"
            )
        ]

        for index in rcvapcindex:
            with self.need(
                f"RcvAPCIndex {index} must reference a RcvAPC {rcvapc_indices}"
            ):
                assert index in rcvapc_indices

    def check_chanindex(self) -> None:
        """Checks consistency of ChanIndex."""
        cp_indices = [
            int(node.get("index"))
            for node in self.sicdroot.findall(
                "./{*}RadarCollection/{*}RcvChannels/{*}ChanParameters"
            )
        ]
        chanindex = [
            int(node.text)
            for node in self.sicdroot.findall(
                "./{*}ImageFormation/{*}RcvChanProc/{*}ChanIndex"
            )
        ]
        for index in chanindex:
            with self.need(
                f"ChanIndex {index} must reference a ChanParameters {cp_indices}"
            ):
                assert index in cp_indices

    def check_grid_polys(self) -> None:
        """Checks consistency of all Grid polynomials."""
        poly_node = self.sicdroot.find("./{*}Grid/{*}TimeCOAPoly")
        self._assert_poly_2d(poly_node, "TimeCOAPoly")

        for dir in ["Row", "Col"]:
            poly_node = self.sicdroot.find(f"./{{*}}Grid/{{*}}{dir}/{{*}}DeltaKCOAPoly")
            with self.precondition():
                assert poly_node is not None
                self._assert_poly_2d(poly_node, f"{dir}/{{*}}DeltaKCOAPoly")

    def check_timeline_polys(self) -> None:
        """Checks consistency of all Timeline polynomials."""
        ipp_sets = self.sicdroot.findall("./{*}Timeline/{*}IPP/{*}Set")
        with self.precondition():
            assert ipp_sets is not None
            for ipp_set in ipp_sets:
                idx = ipp_set.attrib["index"]
                ipp_poly_node = ipp_set.find("./{*}IPPPoly")
                self._assert_poly_1d(ipp_poly_node, f"IPPPoly[{idx}]")

    def check_position_polys(self) -> None:
        """Checks consistency of all Position polynomials."""
        poly_node = self.sicdroot.find("./{*}Position/{*}ARPPoly")
        self._assert_poly_xyz(poly_node, "ARPPoly")

        for poly in ["GRPPoly", "TxAPCPoly"]:
            poly_node = self.sicdroot.find(f"./{{*}}Position/{{*}}{poly}")
            with self.precondition():
                assert poly_node is not None
                self._assert_poly_xyz(poly_node, poly)

        rcv_apc_node = self.sicdroot.find("./{*}Position/{*}RcvAPC")
        with self.precondition():
            assert rcv_apc_node is not None
            rcvapc_poly_nodes = rcv_apc_node.findall("./{*}RcvAPCPoly")
            for rcvapc_poly in rcvapc_poly_nodes:
                idx = rcvapc_poly.attrib["index"]
                self._assert_poly_xyz(rcvapc_poly, f"RcvAPCPoly[{idx}]")

    def check_radiometric_polys(self) -> None:
        """Checks consistency of all Radiometric polynomials."""
        for poly in [
            "NoiseLevel/{*}NoisePoly",
            "RCSSFPoly",
            "SigmaZeroSFPoly",
            "BetaZeroSFPoly",
            "GammaZeroSFPoly",
        ]:
            poly_node = self.sicdroot.find(f"./{{*}}Radiometric/{{*}}{poly}")
            with self.precondition():
                assert poly_node is not None
                self._assert_poly_2d(poly_node, poly)

    def check_antenna_polys(self) -> None:
        """Checks consistency of all Antenna polynomials."""
        for ant in ["Tx", "Rcv", "TwoWay"]:
            for poly in ["EB/{*}DCXPoly", "EB/{*}DCYPoly", "GainBSPoly"]:
                poly_node = self.sicdroot.find(f"./{{*}}Antenna/{{*}}{ant}/{{*}}{poly}")
                with self.precondition():
                    assert poly_node is not None
                    self._assert_poly_1d(poly_node, poly)

            for poly in [
                "Array/{*}GainPoly",
                "Array/{*}PhasePoly",
                "Elem/{*}GainPoly",
                "Elem/{*}PhasePoly",
            ]:
                poly_node = self.sicdroot.find(f"./{{*}}Antenna/{{*}}{ant}/{{*}}{poly}")
                with self.precondition():
                    assert poly_node is not None
                    self._assert_poly_2d(poly_node, poly)

            for poly in ["XAxisPoly", "YAxisPoly"]:
                poly_node = self.sicdroot.find(f"./{{*}}Antenna/{{*}}{ant}/{{*}}{poly}")
                with self.precondition():
                    assert poly_node is not None
                    self._assert_poly_xyz(poly_node, poly)

    def check_rgazcomp_polys(self) -> None:
        """Checks consistency of all RgAzComp polynomials."""
        poly_node = self.sicdroot.find("./{*}RgAzComp/{*}KazPoly")
        with self.precondition():
            assert poly_node is not None
            self._assert_poly_1d(poly_node, "RgAzComp/{*}KazPoly")

    def check_pfa_polys(self) -> None:
        """Checks consistency of all PFA polynomials."""
        for poly in ["PolarAngPoly", "SpatialFreqSFPoly"]:
            poly_node = self.sicdroot.find(f"./{{*}}PFA/{{*}}{poly}")
            with self.precondition():
                assert poly_node is not None
                self._assert_poly_1d(poly_node, poly)

        poly_node = self.sicdroot.find("./{*}PFA/{*}STDeskew/{*}STDSPhasePoly")
        with self.precondition():
            assert poly_node is not None
            self._assert_poly_2d(poly_node, "STDSPhasePoly")

    def check_rma_inca_polys(self) -> None:
        """Checks consistency of all RMA/INCA polynomials."""
        poly_node = self.sicdroot.find("./{*}RMA/{*}INCA/{*}TimeCAPoly")
        with self.precondition():
            assert poly_node is not None
            self._assert_poly_1d(poly_node, "RMA/{*}INCA/{*}TimeCAPoly")

        for poly in ["DRateSFPoly", "DopCentroidPoly"]:
            poly_node = self.sicdroot.find(f"./{{*}}RMA/{{*}}INCA/{{*}}{poly}")
            with self.precondition():
                assert poly_node is not None
                self._assert_poly_2d(poly_node, poly)


def _parser():
    parser = argparse.ArgumentParser(
        description="Analyze a SICD and display inconsistencies"
    )
    parser.add_argument("file_name")
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        action="count",
        help="Increase verbosity (can be specified more than once; >4 doesn't help)",
    )
    parser.add_argument(
        "--schema",
        type=pathlib.Path,
        help="Use a supplied schema file (attempts version-specific schema if omitted)",
    )
    parser.add_argument(
        "--version",
        type=str,
        help=(
            "Perfom checks for a specific version of SICD. "
            "Attempts auto-detection if omitted. Required if using --schema"
        ),
    )
    parser.add_argument(
        "--ignore",
        action="extend",
        nargs="+",
        metavar="PATTERN",
        help=(
            "Skip any check matching PATTERN at the beginning of its name. "
            "Can be specified more than once."
        ),
    )
    return parser


def main(args=None):
    config = _parser().parse_args(args)

    if config.schema is not None and config.version is None:
        raise RuntimeError("--version must be specified if using --schema")

    sicd_con = SicdConsistency.from_file(
        filename=config.file_name, schema=config.schema, version=config.version
    )
    sicd_con.check(ignore_patterns=config.ignore)
    failures = sicd_con.failures()
    sicd_con.print_result(
        fail_detail=config.verbose >= 1,
        include_passed_asserts=config.verbose >= 2,
        include_passed_checks=config.verbose >= 3,
        skip_detail=config.verbose >= 4,
    )

    return bool(failures)


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(int(main()))

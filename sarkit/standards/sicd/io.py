"""
========
SICD I/O
========

Functions from SICD Volume 2 File Format Description Document

"""

import dataclasses
import datetime
import importlib.resources
import itertools
import warnings
from typing import Any, Self

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit._nitf.nitf
import sarkit._nitf.nitf_elements.des
import sarkit._nitf.nitf_elements.image
import sarkit._nitf.nitf_elements.nitf_head
import sarkit._nitf.nitf_elements.security
import sarkit._nitf.utils
import sarkit.standards.geocoords
import sarkit.standards.sicd.xml

SPECIFICATION_IDENTIFIER = "SICD Volume 1 Design & Implementation Description Document"

SCHEMA_DIR = importlib.resources.files("sarkit.standards.sicd.schemas")

# Keys in ascending order
VERSION_INFO = {
    "urn:SICD:1.1.0": {
        "version": "1.1",
        "date": "2014-09-30T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.1.0_2014_09_30.xsd",
    },
    "urn:SICD:1.2.1": {
        "version": "1.2.1",
        "date": "2018-12-13T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.2.1_2018_12_13.xsd",
    },
    "urn:SICD:1.3.0": {
        "version": "1.3.0",
        "date": "2021-11-30T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.3.0_2021_11_30.xsd",
    },
    "urn:SICD:1.4.0": {
        "version": "1.4.0",
        "date": "2023-10-26T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.4.0_2023_10_26.xsd",
    },
}


PIXEL_TYPES: dict[str, dict[str, Any]] = {
    "RE32F_IM32F": {
        "bytes": 8,
        "pvtype": "R",
        "subcat": ("I", "Q"),
        "dtype": np.dtype(np.complex64),
    },
    "RE16I_IM16I": {
        "bytes": 4,
        "pvtype": "SI",
        "subcat": ("I", "Q"),
        "dtype": np.dtype([("real", np.int16), ("imag", np.int16)]),
    },
    "AMP8I_PHS8I": {
        "bytes": 2,
        "pvtype": "INT",
        "subcat": ("M", "P"),
        "dtype": np.dtype([("amp", np.uint8), ("phase", np.uint8)]),
    },
}


@dataclasses.dataclass(kw_only=True)
class SicdNitfSecurityFields:
    """NITF Security Header/Subheader fields

    Attributes
    ----------
    clas : str
        File Security Classification
    clsy : str
        File Security Classification System
    code : str
        File Codewords
    ctlh : str
        File Control and Handling
    rel : str
        File Releasing Instructions
    dctp : str
        File Declassification Type
    dcdt : str
        File Declassification Date
    dcxm : str
        File Declassification Exemption
    dg : str
        File Downgrade
    dgdt : str
        File Downgrade Date
    cltx : str
        File Classification Text
    catp : str
        File Classification Authority Type
    caut : str
        File Classification Authority
    crsn : str
        File Classification Reason
    srdt : str
        File Security Source Date
    ctln : str
        File Security Control Number
    """

    clas: str
    clsy: str = ""
    code: str = ""
    ctlh: str = ""
    rel: str = ""
    dctp: str = ""
    dcdt: str = ""
    dcxm: str = ""
    dg: str = ""
    dgdt: str = ""
    cltx: str = ""
    catp: str = ""
    caut: str = ""
    crsn: str = ""
    srdt: str = ""
    ctln: str = ""

    @classmethod
    def _from_security_tags(
        cls, security: sarkit._nitf.nitf_elements.security.NITFSecurityTags
    ) -> Self:
        """Construct from a NITFSecurityTags object"""
        return cls(
            clas=security.CLAS,
            clsy=security.CLSY,
            code=security.CODE,
            ctlh=security.CTLH,
            rel=security.REL,
            dctp=security.DCTP,
            dcdt=security.DCDT,
            dcxm=security.DCXM,
            dg=security.DG,
            dgdt=security.DGDT,
            cltx=security.CLTX,
            catp=security.CATP,
            caut=security.CAUT,
            crsn=security.CRSN,
            srdt=security.SRDT,
            ctln=security.CTLN,
        )

    def _as_security_tags(
        self,
    ) -> sarkit._nitf.nitf_elements.security.NITFSecurityTags:
        """Construct a NITFSecurityTags object"""
        return sarkit._nitf.nitf_elements.security.NITFSecurityTags(
            CLAS=self.clas,
            CLSY=self.clsy,
            CODE=self.code,
            CTLH=self.ctlh,
            REL=self.rel,
            DCTP=self.dctp,
            DCDT=self.dcdt,
            DCXM=self.dcxm,
            DG=self.dg,
            DGDT=self.dgdt,
            CLTX=self.cltx,
            CATP=self.catp,
            CAUT=self.caut,
            CRSN=self.crsn,
            SRDT=self.srdt,
            CTLN=self.ctln,
        )


@dataclasses.dataclass(kw_only=True)
class SicdNitfHeaderFields:
    """NITF header fields which are set according to a Program Specific Implementation Document

    Attributes
    ----------
    ostaid : str
        Originating Station ID
    ftitle : str
        File Title
    security : :py:class:`SicdNitfSecurityFields`
        Security Tags with "FS" prefix
    oname : str
        Originator's Name
    ophone : str
        Originator's Phone
    """

    ostaid: str
    ftitle: str = ""
    security: SicdNitfSecurityFields
    oname: str = ""
    ophone: str = ""

    @classmethod
    def _from_header(cls, file_header: sarkit._nitf.nitf.NITFHeader) -> Self:
        """Construct from a NITFHeader object"""
        return cls(
            ostaid=file_header.OSTAID,
            ftitle=file_header.FTITLE,
            security=SicdNitfSecurityFields._from_security_tags(file_header.Security),
            oname=file_header.ONAME,
            ophone=file_header.OPHONE,
        )

    def __post_init__(self):
        if isinstance(self.security, dict):
            self.security = SicdNitfSecurityFields(**self.security)


@dataclasses.dataclass(kw_only=True)
class SicdNitfImageSegmentFields:
    """NITF image header fields which are set according to a Program Specific Implementation Document

    Attributes
    ----------
    tgtid : str
       Target Identifier
    iid2 : str
        Image Identifier 2
    security : :py:class:`SicdNitfSecurityFields`
        Security Tags with "IS" prefix
    isorce : str
        Image Source
    icom : list of str
        Image Comments
    """

    ## IS fields are applied to all segments
    tgtid: str = ""
    iid2: str = ""
    security: SicdNitfSecurityFields
    isorce: str
    icom: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def _from_header(cls, image_header: sarkit._nitf.nitf.ImageSegmentHeader) -> Self:
        """Construct from a NITF ImageSegmentHeader object"""
        return cls(
            tgtid=image_header.TGTID,
            iid2=image_header.IID2,
            security=SicdNitfSecurityFields._from_security_tags(image_header.Security),
            isorce=image_header.ISORCE,
            icom=[
                val.to_bytes().decode().rstrip() for val in image_header.Comments.values
            ],
        )

    def __post_init__(self):
        if isinstance(self.security, dict):
            self.security = SicdNitfSecurityFields(**self.security)


@dataclasses.dataclass(kw_only=True)
class SicdNitfDESegmentFields:
    """NITF DE header fields which are set according to a Program Specific Implementation Document

    Attributes
    ----------
    security : :py:class:`SicdNitfSecurityFields`
        Security Tags with "DES" prefix
    desshrp : str
        Responsible Party - Organization Identifier
    desshli : str
        Location - Identifier
    desshlin : str
        Location Identifier Namespace URI
    desshabs : str
        Abstract. Brief narrative summary of the content of the DES.
    """

    security: SicdNitfSecurityFields
    desshrp: str = ""
    desshli: str = ""
    desshlin: str = ""
    desshabs: str = ""

    @classmethod
    def _from_header(cls, de_header: sarkit._nitf.nitf.DataExtensionHeader) -> Self:
        """Construct from a NITF DataExtensionHeader object"""
        return cls(
            security=SicdNitfSecurityFields._from_security_tags(de_header.Security),
            desshrp=de_header.UserHeader.DESSHRP,
            desshli=de_header.UserHeader.DESSHLI,
            desshlin=de_header.UserHeader.DESSHLIN,
            desshabs=de_header.UserHeader.DESSHABS,
        )

    def __post_init__(self):
        if isinstance(self.security, dict):
            self.security = SicdNitfSecurityFields(**self.security)


@dataclasses.dataclass(kw_only=True)
class SicdNitfPlan:
    """Class describing the plan for creating a SICD NITF Container

    Attributes
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree
    header_fields : :py:class:`SicdNitfHeaderFields`
        NITF File Header fields which can be set
    is_fields : :py:class:`SicdNitfImageSegmentFields`
        NITF Image Segment Header fields which can be set
    des_fields : :py:class:`SicdNitfDESegmentFields`
        NITF DE Segment Header fields which can be set

    See Also
    --------
    SicdNitfReader
    SicdNitfWriter
    SicdNitfSecurityFields
    SicdNitfHeaderFields
    SicdNitfImageSegmentFields
    SicdNitfDESegmentFields
    """

    sicd_xmltree: lxml.etree.ElementTree
    header_fields: SicdNitfHeaderFields
    is_fields: SicdNitfImageSegmentFields
    des_fields: SicdNitfDESegmentFields

    def __post_init__(self):
        if isinstance(self.header_fields, dict):
            self.header_fields = SicdNitfHeaderFields(**self.header_fields)
        if isinstance(self.is_fields, dict):
            self.is_fields = SicdNitfImageSegmentFields(**self.is_fields)
        if isinstance(self.des_fields, dict):
            self.des_fields = SicdNitfDESegmentFields(**self.des_fields)


class SicdNitfReader:
    """Read a SICD NITF

    A SicdNitfReader object can be used as a context manager in a ``with`` statement.
    Attributes, but not methods, can be safely accessed outside of the context manager's context.

    Parameters
    ----------
    file : `file object`
        SICD NITF file to read

    Examples
    --------
    >>> with sicd_path.open('rb') as file, SicdNitfReader(file) as reader:
    ...     sicd_xmltree = reader.sicd_xmltree
    ...     pixels = reader.read_image()

    Attributes
    ----------
    sicd_xmltree : lxml.etree.ElementTree
    header_fields : SicdNitfHeaderFields
    is_fields : SicdNitfImageSegmentFields
    des_fields : SicdNitfDESegmentFields
    nitf_plan : :py:class:`SicdNitfPlan`
        A SicdNitfPlan object suitable for use in a SicdNitfWriter

    See Also
    --------
    SicdNitfPlan
    SicdNitfWriter
    """

    def __init__(self, file):
        self._file_object = file

        self._initial_offset = self._file_object.tell()
        if self._initial_offset != 0:
            raise RuntimeError(
                "seek(0) must be the start of the NITF"
            )  # this is a NITFDetails limitation

        nitf_details = sarkit._nitf.nitf.NITFDetails(self._file_object)
        image_segment_collections = [
            [
                n
                for n, imghdr in enumerate(nitf_details.img_headers)
                if imghdr.IID1.startswith("SICD")
            ]
        ]
        self._nitf_reader = sarkit._nitf.nitf.NITFReader(
            nitf_details=nitf_details,
            image_segment_collections=image_segment_collections,
        )
        des_header = sarkit._nitf.nitf.DataExtensionHeader.from_bytes(
            self._nitf_reader.nitf_details.get_des_subheader_bytes(0), 0
        )
        if not des_header.UserHeader.DESSHTN.startswith("urn:SICD"):
            raise ValueError(f"Unable to find SICD DES in {file}")

        sicd_xmltree = lxml.etree.fromstring(
            self._nitf_reader.nitf_details.get_des_bytes(0)
        ).getroottree()
        nitf_header_fields = SicdNitfHeaderFields._from_header(nitf_details.nitf_header)
        nitf_image_fields = SicdNitfImageSegmentFields._from_header(
            nitf_details.img_headers[0]
        )
        nitf_de_fields = SicdNitfDESegmentFields._from_header(des_header)

        self.nitf_plan = SicdNitfPlan(
            sicd_xmltree=sicd_xmltree,
            header_fields=nitf_header_fields,
            is_fields=nitf_image_fields,
            des_fields=nitf_de_fields,
        )

    @property
    def sicd_xmltree(self) -> lxml.etree.ElementTree:
        """SICD XML tree"""
        return self.nitf_plan.sicd_xmltree

    @property
    def header_fields(self) -> SicdNitfHeaderFields:
        """NITF File Header fields"""
        return self.nitf_plan.header_fields

    @property
    def is_fields(self) -> SicdNitfImageSegmentFields:
        """NITF Image Segment Subheader fields"""
        return self.nitf_plan.is_fields

    @property
    def des_fields(self) -> SicdNitfDESegmentFields:
        """NITF DE Segment Subheader fields"""
        return self.nitf_plan.des_fields

    def read_image(self) -> npt.NDArray:
        """Read the entire pixel array

        Returns
        -------
        ndarray
            SICD image array
        """
        self._file_object.seek(self._initial_offset)
        nrows = int(self.sicd_xmltree.findtext("{*}ImageData/{*}NumRows"))
        ncols = int(self.sicd_xmltree.findtext("{*}ImageData/{*}NumCols"))
        pixel_type = self.sicd_xmltree.findtext("{*}ImageData/{*}PixelType")
        dtype = PIXEL_TYPES[pixel_type]["dtype"].newbyteorder(">")
        sicd_pixels = np.empty((nrows, ncols), dtype)
        imseg_sizes = self._nitf_reader.nitf_details.img_segment_sizes[
            self._nitf_reader.image_segment_collections
        ]
        imseg_offsets = self._nitf_reader.nitf_details.img_segment_offsets[
            self._nitf_reader.image_segment_collections
        ]
        splits = np.cumsum(imseg_sizes // (ncols * dtype.itemsize))[:-1]
        for split, sz, offset in zip(
            np.array_split(sicd_pixels, splits, axis=0), imseg_sizes, imseg_offsets
        ):
            this_os = offset - self._file_object.tell()
            split[...] = np.fromfile(
                self._file_object, dtype, count=sz // dtype.itemsize, offset=this_os
            ).reshape(split.shape)
        return sicd_pixels

    def read_sub_image(
        self,
        start_row: int = 0,
        start_col: int = 0,
        end_row: int = -1,
        end_col: int = -1,
    ) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
        """Read a sub-image from the file

        Parameters
        ----------
        start_row : int
        start_col : int
        end_row : int
        end_col : int

        Returns
        -------
        ndarray
            SICD sub-image array
        lxml.etree.ElementTree
            SICD sub-image XML ElementTree
        """
        _ = self._nitf_reader.read(slice(start_row, end_row), slice(start_col, end_col))
        # TODO update XML
        raise NotImplementedError()

    def done(self):
        "Indicates to the reader that the user is done with it"
        self._file_object = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.done()


def _create_des_manager(sicd_xmltree, des_fields):
    """DES Manager for SICD XML DES"""
    xmlns = lxml.etree.QName(sicd_xmltree.getroot()).namespace
    xml_helper = sarkit.standards.sicd.xml.XmlHelper(sicd_xmltree)
    now_dt = datetime.datetime.now(datetime.timezone.utc)

    icp = xml_helper.load("./{*}GeoData/{*}ImageCorners")
    desshlpg = ""
    for icp_lat, icp_lon in itertools.chain(icp, [icp[0]]):
        desshlpg += f"{icp_lat:0=+12.8f}{icp_lon:0=+13.8f}"

    deshead = sarkit._nitf.nitf_elements.des.DataExtensionHeader(
        Security=des_fields.security._as_security_tags(),
        UserHeader=sarkit._nitf.nitf_elements.des.XMLDESSubheader(
            DESSHSI=SPECIFICATION_IDENTIFIER,
            DESSHSV=VERSION_INFO[xmlns]["version"],
            DESSHSD=VERSION_INFO[xmlns]["date"],
            DESSHTN=xmlns,
            DESSHDT=now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            DESSHLPG=desshlpg,
            DESSHRP=des_fields.desshrp,
            DESSHLI=des_fields.desshli,
            DESSHLIN=des_fields.desshlin,
            DESSHABS=des_fields.desshabs,
        ),
    )
    sicd_des = sarkit._nitf.nitf.DESSubheaderManager(
        deshead, lxml.etree.tostring(sicd_xmltree)
    )
    return sicd_des


class SicdNitfWriter:
    """Write a SICD NITF

    A SicdNitfWriter object can be used as a context manager in a ``with`` statement.

    Parameters
    ----------
    file : `file object`
        SICD NITF file to write
    nitf_plan : :py:class:`SicdNitfPlan`
        NITF plan object

    Notes
    -----
    nitf_plan should not be modified after creation of a writer

    Examples
    --------
    >>> plan = SicdNitfPlan(sicd_xmltree=sicd_xmltree,
    ...                     header_fields=SicdNitfHeaderFields(ostaid='my location',
    ...                                                        security=SicdNitfSecurityFields(clas='U')),
    ...                     is_fields=SicdNitfImageSegmentFields(isorce='my sensor',
    ...                                                          security=SicdNitfSecurityFields(clas='U')),
    ...                     des_fields=SicdNitfDESegmentFields(security=SicdNitfSecurityFields(clas='U')))
    >>> with output_path.open('wb') as file, SicdNitfWriter(file, plan) as writer:
    ...     writer.write_image(pixel_array)

    See Also
    --------
    SicdNitfPlan
    SicdNitfReader
    """

    def __init__(self, file, nitf_plan: SicdNitfPlan):
        self._file_object = file

        self._initial_offset = self._file_object.tell()
        if self._initial_offset != 0:
            raise RuntimeError(
                "seek(0) must be the start of the NITF"
            )  # this is a NITFDetails limitation

        self._nitf_plan = nitf_plan
        sicd_xmltree = nitf_plan.sicd_xmltree

        """Create a SICD NITF from a pixel array and metadata."""
        xmlns = lxml.etree.QName(sicd_xmltree.getroot()).namespace
        schema = lxml.etree.XMLSchema(file=VERSION_INFO[xmlns]["schema"])
        if not schema.validate(sicd_xmltree):
            warnings.warn(str(schema.error_log))

        xml_helper = sarkit.standards.sicd.xml.XmlHelper(sicd_xmltree)
        rows = xml_helper.load("./{*}ImageData/{*}NumRows")
        cols = xml_helper.load("./{*}ImageData/{*}NumCols")
        pixel_type = sicd_xmltree.findtext("./{*}ImageData/{*}PixelType")

        # CLEVEL and FL will be corrected...
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        header = sarkit._nitf.nitf_elements.nitf_head.NITFHeader(
            CLEVEL=3,
            OSTAID=self._nitf_plan.header_fields.ostaid,
            FDT=now_dt.strftime("%Y%m%d%H%M%S"),
            FTITLE=self._nitf_plan.header_fields.ftitle,
            Security=self._nitf_plan.header_fields.security._as_security_tags(),
            ONAME=self._nitf_plan.header_fields.oname,
            OPHONE=self._nitf_plan.header_fields.ophone,
            FL=0,
        )

        # Create image segments
        bits_per_element = PIXEL_TYPES[pixel_type]["bytes"] * 8 / 2
        icp = xml_helper.load("./{*}GeoData/{*}ImageCorners")

        is_size_max = 10**10 - 2  # allowable image segment size
        iloc_max = 99999
        bytes_per_row = cols * PIXEL_TYPES[pixel_type]["bytes"]
        product_size = bytes_per_row * rows
        limit_1 = int(np.floor(is_size_max / bytes_per_row))
        num_rows_limit = min(iloc_max, limit_1)

        if product_size <= is_size_max:
            image_segment_limits = [(0, rows, 0, cols)]
        else:
            image_segment_limits = []
            row_offset = 0
            while row_offset < rows:
                next_rows = min(rows, row_offset + num_rows_limit)
                image_segment_limits.append((row_offset, next_rows, 0, cols))
                row_offset = next_rows

        image_segment_collections = (tuple(range(len(image_segment_limits))),)
        image_segment_coordinates = (tuple(image_segment_limits),)
        image_managers = []
        for i, entry in enumerate(image_segment_limits):
            this_rows = entry[1] - entry[0]
            subhead = sarkit._nitf.nitf_elements.image.ImageSegmentHeader(
                IID1=f"SICD{0 if len(image_segment_limits) == 1 else i + 1:03d}",
                IDATIM=xml_helper.load("./{*}Timeline/{*}CollectStart").strftime(
                    "%Y%m%d%H%M%S"
                ),
                TGTID=self._nitf_plan.is_fields.tgtid,
                IID2=self._nitf_plan.is_fields.iid2,
                Security=self._nitf_plan.is_fields.security._as_security_tags(),
                ISORCE=self._nitf_plan.is_fields.isorce,
                NROWS=this_rows,
                NCOLS=cols,
                PVTYPE=PIXEL_TYPES[pixel_type]["pvtype"],
                IREP="NODISPLY",
                ICAT="SAR",
                ABPP=bits_per_element,
                IGEOLO=sarkit._nitf.utils._interpolate_corner_points_string(
                    np.array(entry, dtype=np.int64), rows, cols, icp
                ),
                Comments=sarkit._nitf.nitf_elements.image.ImageComments(
                    [
                        sarkit._nitf.nitf_elements.image.ImageComment(COMMENT=comment)
                        for comment in self._nitf_plan.is_fields.icom
                    ]
                ),
                IC="NC",
                NPPBH=0 if cols > 8192 else cols,
                NPPBV=0 if this_rows > 8192 else this_rows,
                NBPP=bits_per_element,
                NBPC=1,
                NBPR=1,
                IDLVL=i + 1,
                IALVL=i,
                ILOC=f"{0 if i == 0 else num_rows_limit:05d}00000",
                Bands=sarkit._nitf.nitf_elements.image.ImageBands(
                    values=[
                        sarkit._nitf.nitf_elements.image.ImageBand(ISUBCAT=entry)
                        for entry in PIXEL_TYPES[pixel_type]["subcat"]
                    ]
                ),
            )
            image_managers.append(sarkit._nitf.nitf.ImageSubheaderManager(subhead))

        sicd_des = _create_des_manager(sicd_xmltree, self._nitf_plan.des_fields)

        sicd_details = sarkit._nitf.nitf.NITFWritingDetails(
            header,
            image_managers=tuple(image_managers),
            image_segment_collections=image_segment_collections,
            image_segment_coordinates=image_segment_coordinates,
            des_managers=(sicd_des,),
        )

        self._nitf_writer = sarkit._nitf.nitf.NITFWriter(
            file_object=self._file_object,
            writing_details=sicd_details,
        )

    def write_image(self, array: npt.NDArray, start: None | tuple[int, int] = None):
        """Write pixel data to a NITF file

        Parameters
        ----------
        array : ndarray
            2D array of complex pixels
        start : tuple of ints, optional
            The start index (first_row, first_col) of `array` in the SICD image.
            If not given, `array` must be the full SICD image.

        """
        pixel_type = self._nitf_plan.sicd_xmltree.findtext(
            "./{*}ImageData/{*}PixelType"
        )
        if PIXEL_TYPES[pixel_type]["dtype"] != array.dtype.newbyteorder("="):
            raise ValueError(
                f"Array dtype ({array.dtype}) does not match expected dtype ({PIXEL_TYPES[pixel_type]['dtype']}) "
                f"for PixelType={pixel_type}"
            )

        xml_helper = sarkit.standards.sicd.xml.XmlHelper(self._nitf_plan.sicd_xmltree)
        rows = xml_helper.load("./{*}ImageData/{*}NumRows")
        cols = xml_helper.load("./{*}ImageData/{*}NumCols")
        sicd_shape = np.asarray((rows, cols))

        if start is None:
            # require array to be full image
            if np.any(array.shape != sicd_shape):
                raise ValueError(
                    f"Array shape {array.shape} does not match sicd shape {sicd_shape}."
                    "If writing only a portion of the image, use the 'start' argument"
                )
            start = (0, 0)
        startarr = np.asarray(start)

        if not np.issubdtype(startarr.dtype, np.integer):
            raise ValueError(f"Start index must be integers {startarr=}")

        if np.any(startarr < 0):
            raise ValueError(f"Start index must be non-negative {startarr=}")

        stop = startarr + array.shape
        if np.any(stop > sicd_shape):
            raise ValueError(
                f"array goes beyond end of sicd. start + array.shape = {stop} sicd shape={sicd_shape}"
            )

        if pixel_type == "RE32F_IM32F":
            raw_dtype = array.real.dtype
        else:
            assert array.dtype.names is not None  # placate mypy
            raw_dtype = array.dtype[array.dtype.names[0]]
        raw_array = array.view((raw_dtype, 2))
        raw_array = raw_array.astype(raw_dtype.newbyteorder(">"), copy=False)
        self._nitf_writer.write_raw(raw_array, start_indices=tuple(startarr))

    def close(self):
        """
        Flush to disk and close any opened file descriptors.

        Called automatically when SicdNitfWriter is used as a context manager
        """
        self._nitf_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

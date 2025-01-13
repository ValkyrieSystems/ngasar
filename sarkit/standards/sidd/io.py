"""
========
SIDD I/O
========

Functions from SIDD Volume 2 NITF File Format Description Document

"""

import collections
import dataclasses
import datetime
import importlib
import itertools
import logging
import warnings
from typing import Self, TypedDict

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
import sarkit.standards.sicd.io as sicdio
import sarkit.standards.sidd.xml

logger = logging.getLogger(__name__)

SPECIFICATION_IDENTIFIER = "SIDD Volume 1 Design & Implementation Description Document"

SCHEMA_DIR = importlib.resources.files("sarkit.standards.sidd.schemas")

# Keys in ascending order
VERSION_INFO = {
    "urn:SIDD:2.0.0": {
        "version": "3.0",
        "date": "2019-05-31T00:00:00Z",
        "schema": SCHEMA_DIR / "version2" / "SIDD_schema_V2.0.0_2019_05_31.xsd",
    },
    "urn:SIDD:3.0.0": {
        "version": "3.0",
        "date": "2021-11-30T00:00:00Z",
        "schema": SCHEMA_DIR / "version3" / "SIDD_schema_V3.0.0.xsd",
    },
}


# Table 2-6 NITF 2.1 Image Sub-Header Population for Supported Pixel Type
class _PixelTypeDict(TypedDict):
    IREP: str
    IREPBANDn: list[str]
    IMODE: str
    dtype: np.dtype


PIXEL_TYPES: dict[str, _PixelTypeDict] = {
    "MONO8I": {
        "IREP": "MONO",
        "IREPBANDn": ["M"],
        "IMODE": "B",
        "dtype": np.dtype(np.uint8),
    },
    "MONO8LU": {
        "IREP": "MONO",
        "IREPBANDn": ["LU"],
        "IMODE": "B",
        "dtype": np.dtype(np.uint8),
    },
    "MONO16I": {
        "IREP": "MONO",
        "IREPBANDn": ["M"],
        "IMODE": "B",
        "dtype": np.dtype(np.uint16),
    },
    "RGB8LU": {
        "IREP": "RGB/LUT",
        "IREPBANDn": ["LU"],
        "IMODE": "B",
        "dtype": np.dtype(np.uint8),
    },
    "RGB24I": {
        "IREP": "RGB",
        "IREPBANDn": ["R", "G", "B"],
        "IMODE": "P",
        "dtype": np.dtype([("R", np.uint8), ("G", np.uint8), ("B", np.uint8)]),
    },
}

LI_MAX = 9_999_999_998
ILOC_MAX = 99_999


# SICD implementation happens to match, reuse it
class SiddNitfSecurityFields(sicdio.SicdNitfSecurityFields):
    __doc__ = sicdio.SicdNitfSecurityFields.__doc__


# SICD implementation happens to match, reuse it
class SiddNitfHeaderFields(sicdio.SicdNitfHeaderFields):
    __doc__ = sicdio.SicdNitfHeaderFields.__doc__


@dataclasses.dataclass(kw_only=True)
class SiddNitfImageSegmentFields:
    """NITF image header fields which are set according to a Program Specific Implementation Document

    Attributes
    ----------
    tgtid : str
        Target Identifier
    iid2 : str
        Image Identifier 2
    security : :py:class:`SiddNitfSecurityFields`
        Security Tags with "IS" prefix
    icom : list of str
        Image Comments
    """

    ## IS fields are applied to all segments
    tgtid: str = ""
    iid2: str = ""
    security: SiddNitfSecurityFields
    icom: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def _from_header(cls, image_header: sarkit._nitf.nitf.ImageSegmentHeader) -> Self:
        """Construct from a NITF ImageSegmentHeader object"""
        return cls(
            tgtid=image_header.TGTID,
            iid2=image_header.IID2,
            security=SiddNitfSecurityFields._from_security_tags(image_header.Security),
            icom=[
                val.to_bytes().decode().rstrip() for val in image_header.Comments.values
            ],
        )

    def __post_init__(self):
        if isinstance(self.security, dict):
            self.security = SiddNitfSecurityFields(**self.security)


# SICD implementation happens to match, reuse it
class SiddNitfDESegmentFields(sicdio.SicdNitfDESegmentFields):
    __doc__ = sicdio.SicdNitfDESegmentFields.__doc__


@dataclasses.dataclass
class SiddNitfPlanProductImageInfo:
    """Metadata necessary for describing the plan to add a product image to a SIDD

    Attributes
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD product metadata XML ElementTree
    is_fields : :py:class:`SiddNitfImageSegmentFields`
        NITF Image Segment Header fields which can be set
    des_fields : :py:class:`SiddNitfDESegmentFields`
        NITF DE Segment Header fields which can be set

    See Also
    --------
    SiddNitfPlan
    SiddNitfPlanLegendInfo
    SiddNitfPlanDedInfo
    SiddNitfPlanProductSupportXmlInfo
    SiddNitfPlanSicdXmlInfo
    """

    sidd_xmltree: lxml.etree.ElementTree
    is_fields: SiddNitfImageSegmentFields
    des_fields: SiddNitfDESegmentFields

    def __post_init__(self):
        if isinstance(self.is_fields, dict):
            self.is_fields = SiddNitfImageSegmentFields(**self.is_fields)
            self.des_fields = SiddNitfDESegmentFields(**self.des_fields)


@dataclasses.dataclass
class SiddNitfPlanLegendInfo:
    """Metadata necessary for describing the plan to add a legend to a SIDD

    See Also
    --------
    SiddNitfPlan
    SiddNitfPlanProductImageInfo
    SiddNitfPlanDedInfo
    SiddNitfPlanProductSupportXmlInfo
    SiddNitfPlanSicdXmlInfo
    """

    def __post_init__(self):
        raise NotImplementedError()


@dataclasses.dataclass
class SiddNitfPlanDedInfo:
    """Metadata necessary for describing the plan to add Digital Elevation Data (DED) to a SIDD

    See Also
    --------
    SiddNitfPlan
    SiddNitfPlanProductImageInfo
    SiddNitfPlanLegendInfo
    SiddNitfPlanProductSupportXmlInfo
    SiddNitfPlanSicdXmlInfo
    """

    def __post_init__(self):
        raise NotImplementedError()


@dataclasses.dataclass
class SiddNitfPlanProductSupportXmlInfo:
    """Metadata necessary for describing the plan to add a Product Support XML to a SIDD

    See Also
    --------
    SiddNitfPlan
    SiddNitfPlanProductImageInfo
    SiddNitfPlanLegendInfo
    SiddNitfPlanDedInfo
    SiddNitfPlanSicdXmlInfo
    """

    product_support_xmltree: lxml.etree.ElementTree
    des_fields: SiddNitfDESegmentFields

    def __post_init__(self):
        if isinstance(self.des_fields, dict):
            self.des_fields = SiddNitfDESegmentFields(**self.des_fields)


@dataclasses.dataclass
class SiddNitfPlanSicdXmlInfo:
    """Metadata necessary for describing the plan to add SICD XML to a SIDD

    See Also
    --------
    SiddNitfPlan
    SiddNitfPlanProductImageInfo
    SiddNitfPlanLegendInfo
    SiddNitfPlanDedInfo
    SiddNitfPlanProductSupportXmlInfo
    """

    sicd_xmltree: lxml.etree.ElementTree
    des_fields: sicdio.SicdNitfDESegmentFields

    def __post_init__(self):
        if isinstance(self.des_fields, dict):
            self.des_fields = sicdio.SicdNitfDESegmentFields(**self.des_fields)


class SiddNitfPlan:
    """Class describing the plan for creating a SIDD NITF Container

    Parameters
    ----------
    header_fields : :py:class:`SiddNitfHeaderFields`
        NITF Header fields

    Attributes
    ----------
    header_fields : :py:class:`SiddNitfHeaderFields`
        NITF File Header fields which can be set
    images : list of :py:class:`SiddNitfPlanProductImageInfo`
        List of image information
    legends : list of :py:class:`SiddNitfPlanLegendInfo`
        List of legend information
    ded : :py:class:`SiddNitfPlanDedInfo`
        DED information
    product_support_xmls : list of :py:class:`SiddNitfPlanProductSupportXmlInfo`
        List of SICD XML information
    sicd_xmls : list of :py:class:`SiddNitfPlanSicdXmlInfo`
        List of SICD XML information

    See Also
    --------
    SiddNitfReader
    SiddNitfWriter
    SiddNitfSecurityFields
    SiddNitfHeaderFields
    SiddNitfImageSegmentFields
    SiddNitfDESegmentFields
    SiddNitfPlanProductImageInfo
    SiddNitfPlanLegendInfo
    """

    def __init__(self, header_fields: SiddNitfHeaderFields | dict):
        self.header_fields = header_fields
        if isinstance(self.header_fields, dict):
            self.header_fields = SiddNitfHeaderFields(**self.header_fields)
        self._images: list[SiddNitfPlanProductImageInfo] = []
        self._legends: list[SiddNitfPlanLegendInfo] = []
        self._ded: SiddNitfPlanDedInfo | None = None
        self._product_support_xmls: list[SiddNitfPlanProductSupportXmlInfo] = []
        self._sicd_xmls: list[SiddNitfPlanSicdXmlInfo] = []

    @property
    def images(self) -> list[SiddNitfPlanProductImageInfo]:
        return self._images

    @property
    def legends(self) -> list[SiddNitfPlanLegendInfo]:
        return self._legends

    @property
    def ded(self) -> SiddNitfPlanDedInfo | None:
        return self._ded

    @property
    def product_support_xmls(self) -> list[SiddNitfPlanProductSupportXmlInfo]:
        return self._product_support_xmls

    @property
    def sicd_xmls(self) -> list[SiddNitfPlanSicdXmlInfo]:
        return self._sicd_xmls

    def add_image(
        self,
        sidd_xmltree: lxml.etree.ElementTree,
        is_fields: SiddNitfImageSegmentFields,
        des_fields: SiddNitfDESegmentFields,
    ) -> int:
        """Add a SAR product to the plan

        Parameters
        ----------
        sidd_xmltree : lxml.etree.ElementTree
            SIDD XML ElementTree
        is_fields : :py:class:`SiddNitfImageSegmentFields`
            NITF Image Segment Header fields which can be set
        des_fields : :py:class:`SiddNitfDESegmentFields`
            NITF DE Segment Header fields which can be set

        Returns
        -------
        int
            The image number of the newly added SAR image
        """
        _validate_xml(sidd_xmltree)

        self._images.append(
            SiddNitfPlanProductImageInfo(
                sidd_xmltree, is_fields=is_fields, des_fields=des_fields
            )
        )
        return len(self._images) - 1

    def add_product_support_xml(
        self, ps_xmltree: lxml.etree.ElementTree, des_fields: SiddNitfDESegmentFields
    ) -> int:
        """Add a Product Support XML to the plan

        Parameters
        ----------
        ps_xmltree : lxml.etree.ElementTree
            Product Support XML ElementTree
        des_fields : :py:class:`SiddNitfDESegmentFields`
            NITF DE Segment Header fields which can be set

        Returns
        -------
        int
            The index of the newly added Product Support XML
        """
        self.product_support_xmls.append(
            SiddNitfPlanProductSupportXmlInfo(ps_xmltree, des_fields)
        )
        return len(self.product_support_xmls) - 1

    def add_sicd_xml(
        self,
        sicd_xmltree: lxml.etree.ElementTree,
        des_fields: sicdio.SicdNitfDESegmentFields,
    ) -> int:
        """Add a SICD XML to the plan

        Parameters
        ----------
        sicd_xmltree : lxml.etree.ElementTree
            SICD XML ElementTree
        des_fields : :py:class:`sicdio.SicdNitfDESegmentFields`
            NITF DE Segment Header fields which can be set

        Returns
        -------
        int
            The index of the newly added SICD XML
        """
        self.sicd_xmls.append(SiddNitfPlanSicdXmlInfo(sicd_xmltree, des_fields))
        return len(self.sicd_xmls) - 1

    def add_legend(
        self, attached_to: int, location: tuple[int, int], shape: tuple[int, int]
    ) -> int:
        """Add a Legend to the plan

        Parameters
        ----------
        attached_to : int
            SAR image number to attach legend to
        location : tuple of int
            (row, column) of the SAR image to place first legend pixel
        shape : tuple of int
            Dimension of the legend (Number of Rows, Number of Columns)

        """
        raise NotImplementedError()

    def add_ded(self, shape: tuple[int, int]) -> int:
        """Add a DED to the plan

        Parameters
        ----------
        shape : tuple of int
            Dimension of the DED (Number of Rows, Number of Columns)

        """
        raise NotImplementedError()


class SiddNitfReader:
    """Read a SIDD NITF

    A SiddNitfReader object should be used as a context manager in a ``with`` statement.
    Attributes, but not methods, can be safely accessed outside of the context manager's context.

    Parameters
    ----------
    file : `file object`
        SIDD NITF file to read

    Examples
    --------
    >>> with sidd_path.open('rb') as file, SiddNitfReader(file) as reader:
    ...     sidd_xmltree = reader.images[0].sidd_xmltree
    ...     pixels = reader.read_image(0)

    Attributes
    ----------
    images : list of :py:class:`SiddNitfPlanProductImageInfo`
    header_fields : :py:class:`SiddNitfHeaderFields`
    product_support_xmls : list of :py:class:`SiddNitfPlanProductSupportXmlInfo`
    sicd_xmls : list of :py:class:`SiddNitfPlanSicdXmlInfo`
    plan : :py:class:`SiddNitfPlan`
        A SiddNitfPlan object suitable for use in a SiddNitfWriter

    See Also
    --------
    SiddNitfPlan
    SiddNitfWriter
    """

    def __init__(self, file):
        self._file = file

        self._initial_offset = self._file.tell()
        if self._initial_offset != 0:
            raise RuntimeError(
                "seek(0) must be the start of the NITF"
            )  # this is a NITFDetails limitation

        self._nitf_details = sarkit._nitf.nitf.NITFDetails(self._file)

        im_segments = {}
        for imseg_index, img_header in enumerate(self._nitf_details.img_headers):
            if img_header.IID1.startswith("SIDD"):
                if img_header.ICAT == "SAR":
                    image_number = int(img_header.IID1[4:7]) - 1
                    im_segments.setdefault(image_number, [])
                    im_segments[image_number].append(imseg_index)
                else:
                    raise NotImplementedError("Non SAR images not supported")  # TODO
            elif img_header.IID1.startswith("DED"):
                raise NotImplementedError("DED not supported")  # TODO

        image_segment_collections = {}
        for idx, imghdr in enumerate(self._nitf_details.img_headers):
            if not imghdr.IID1.startswith("SIDD"):
                continue
            image_num = int(imghdr.IID1[4:7]) - 1
            image_segment_collections.setdefault(image_num, [])
            image_segment_collections[image_num].append(idx)

        self._nitf_reader = sarkit._nitf.nitf.NITFReader(
            nitf_details=self._nitf_details,
            image_segment_collections=tuple(
                (tuple(val) for val in image_segment_collections.values())
            ),
        )

        self.header_fields = SiddNitfHeaderFields._from_header(
            self._nitf_details.nitf_header
        )
        self.plan = SiddNitfPlan(header_fields=self.header_fields)

        image_number = 0
        for idx in range(self._nitf_details.des_subheader_offsets.size):
            subhead_bytes = self._nitf_details.get_des_subheader_bytes(idx)
            des_header = sarkit._nitf.nitf.DataExtensionHeader.from_bytes(
                self._nitf_details.get_des_subheader_bytes(0), 0
            )
            if subhead_bytes.startswith(b"DEXML_DATA_CONTENT"):
                des_bytes = self._nitf_details.get_des_bytes(idx)
                try:
                    xmltree = lxml.etree.fromstring(des_bytes).getroottree()
                except lxml.etree.XMLSyntaxError:
                    logger.error(f"Failed to parse DES {idx} as XML")
                    continue

                if "SIDD" in xmltree.getroot().tag:
                    nitf_de_fields = SiddNitfDESegmentFields._from_header(des_header)
                    if len(self.plan.images) < len(image_segment_collections):
                        # user settable fields should be the same for all image segments
                        im_idx = im_segments[image_number][0]
                        im_fields = SiddNitfImageSegmentFields._from_header(
                            self._nitf_details.img_headers[im_idx]
                        )
                        self.plan.add_image(
                            sidd_xmltree=xmltree,
                            is_fields=im_fields,
                            des_fields=nitf_de_fields,
                        )
                        image_number += 1
                    else:
                        # No matching product image, treat it as a product support XML
                        self.plan.add_product_support_xml(xmltree, nitf_de_fields)
                elif "SICD" in xmltree.getroot().tag:
                    nitf_de_fields = sicdio.SicdNitfDESegmentFields._from_header(
                        des_header
                    )
                    self.plan.add_sicd_xml(xmltree, nitf_de_fields)
                else:
                    nitf_de_fields = SiddNitfDESegmentFields._from_header(des_header)
                    self.plan.add_product_support_xml(xmltree, nitf_de_fields)

        # TODO Legends
        # TODO DED
        assert not self.plan.legends
        assert not self.plan.ded

    @property
    def images(self) -> list[SiddNitfPlanProductImageInfo]:
        """List of images contained in the SIDD"""
        return self.plan.images

    def read_image(self, image_number: int) -> npt.NDArray:
        """Read the entire pixel array

        Parameters
        ----------
        image_number : int
            index of SIDD Product image to read

        Returns
        -------
        ndarray
            SIDD image array
        """
        return self._nitf_reader.read(index=image_number)

    @property
    def product_support_xmls(self) -> list[SiddNitfPlanProductSupportXmlInfo]:
        """List of Product Support XML instances contained in the SIDD"""
        return self.plan.product_support_xmls

    @property
    def sicd_xmls(self) -> list[SiddNitfPlanSicdXmlInfo]:
        """List of SICD XML contained in the SIDD"""
        return self.plan.sicd_xmls

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return


class SiddNitfWriter:
    """Write a SIDD NITF

    A SiddNitfWriter object should be used as a context manager in a ``with`` statement.

    Parameters
    ----------
    file : `file object`
        SIDD NITF file to write
    nitf_plan : :py:class:`SiddNitfPlan`
        NITF plan object

    Notes
    -----
    nitf_plan should not be modified after creation of a writer

    Examples
    --------
    >>> plan = SiddNitfPlan(header_fields=SiddNitfHeaderFields(ostaid='my location',
    ...                                                        security=SiddNitfSecurityFields(clas='U')))
    >>> image_index = plan.add_image(is_fields=SiddNitfImageSegmentFields(security=SiddNitfSecurityFields(clas='U')),
    ...                              des_fields=SiddNitfDESegmentFields(security=SiddNitfSecurityFields(clas='U')))
    >>> with output_path.open('wb') as file, SiddNitfWriter(file, plan) as writer:
    ...     writer.write_image(image_index, pixel_array)

    See Also
    --------
    SiddNitfPlan
    SiddNitfReader
    """

    def __init__(self, file, nitf_plan):
        self._file = file
        self._nitf_plan = nitf_plan

        self._images_written = set()

        self._initial_offset = self._file.tell()
        if self._initial_offset != 0:
            raise RuntimeError(
                "seek(0) must be the start of the NITF"
            )  # this is a NITFDetails limitation

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

        image_managers = []
        image_segment_collections = {}  # image_num -> [image_segment, ...]
        image_segment_coordinates = {}  # image_num -> [(first_row, last_row, first_col, last_col), ...]
        current_start_row = 0
        _, _, imhdrs = segmentation_algorithm(
            (img.sidd_xmltree for img in self._nitf_plan.images)
        )
        for idx, imhdr in enumerate(imhdrs):
            if imhdr.ialvl == 0:
                # first segment of each SAR image is attached to the CCS
                current_start_row = 0
            image_num = int(imhdr.iid1[4:7]) - 1
            image_segment_collections.setdefault(image_num, [])
            image_segment_coordinates.setdefault(image_num, [])
            image_segment_collections[image_num].append(idx)
            image_segment_coordinates[image_num].append(
                (current_start_row, current_start_row + imhdr.nrows, 0, imhdr.ncols)
            )
            current_start_row += imhdr.nrows

            imageinfo = self._nitf_plan.images[image_num]
            xml_helper = sarkit.standards.sidd.xml.XmlHelper(imageinfo.sidd_xmltree)
            pixel_info = PIXEL_TYPES[xml_helper.load("./{*}Display/{*}PixelType")]

            icp = xml_helper.load("./{*}GeoData/{*}ImageCorners")
            rows = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Row")
            cols = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Col")

            subhead = sarkit._nitf.nitf_elements.image.ImageSegmentHeader(
                IID1=imhdr.iid1,
                IDATIM=xml_helper.load(
                    "./{*}ExploitationFeatures/{*}Collection/{*}Information/{*}CollectionDateTime"
                ).strftime("%Y%m%d%H%M%S"),
                TGTID=imageinfo.is_fields.tgtid,
                IID2=imageinfo.is_fields.iid2,
                Security=imageinfo.is_fields.security._as_security_tags(),
                ISORCE=xml_helper.load(
                    "./{*}ExploitationFeatures/{*}Collection/{*}Information/{*}SensorName"
                ),
                NROWS=imhdr.nrows,
                NCOLS=imhdr.ncols,
                PVTYPE="INT",
                IREP=pixel_info["IREP"],
                ICAT="SAR",
                ABPP=pixel_info["dtype"].itemsize * 8,
                ICORDS="G",
                IGEOLO=sarkit._nitf.utils._interpolate_corner_points_string(
                    np.array(image_segment_coordinates[image_num][-1], dtype=np.int64),
                    rows,
                    cols,
                    icp,
                ),
                Comments=sarkit._nitf.nitf_elements.image.ImageComments(
                    [
                        sarkit._nitf.nitf_elements.image.ImageComment(COMMENT=comment)
                        for comment in imageinfo.is_fields.icom
                    ]
                ),
                IC="NC",
                IMODE=pixel_info["IMODE"],
                NPPBH=0 if imhdr.ncols > 8192 else imhdr.ncols,
                NPPBV=0 if imhdr.nrows > 8192 else imhdr.nrows,
                NBPP=pixel_info["dtype"].itemsize * 8,
                NBPC=1,
                NBPR=1,
                IDLVL=imhdr.idlvl,
                IALVL=imhdr.ialvl,
                ILOC=imhdr.iloc,
                Bands=sarkit._nitf.nitf_elements.image.ImageBands(
                    values=[
                        sarkit._nitf.nitf_elements.image.ImageBand(
                            ISUBCAT="", IREPBAND=entry
                        )
                        for entry in pixel_info["IREPBANDn"]
                    ]
                ),
            )
            image_managers.append(sarkit._nitf.nitf.ImageSubheaderManager(subhead))

        # TODO add image_managers for legends
        assert not self._nitf_plan.legends
        # TODO add image_managers for DED
        assert not self._nitf_plan.ded

        # DE Segments

        des_managers = []
        for imageinfo in self._nitf_plan.images:
            xmlns = lxml.etree.QName(imageinfo.sidd_xmltree.getroot()).namespace
            xml_helper = sarkit.standards.sidd.xml.XmlHelper(imageinfo.sidd_xmltree)
            icp = xml_helper.load("./{*}GeoData/{*}ImageCorners")
            desshlpg = ""
            for icp_lat, icp_lon in itertools.chain(icp, [icp[0]]):
                desshlpg += f"{icp_lat:0=+12.8f}{icp_lon:0=+13.8f}"

            deshead = sarkit._nitf.nitf_elements.des.DataExtensionHeader(
                Security=imageinfo.des_fields.security._as_security_tags(),
                UserHeader=sarkit._nitf.nitf_elements.des.XMLDESSubheader(
                    DESSHSI=SPECIFICATION_IDENTIFIER,
                    DESSHSV=VERSION_INFO[xmlns]["version"],
                    DESSHSD=VERSION_INFO[xmlns]["date"],
                    DESSHTN=xmlns,
                    DESSHDT=now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    DESSHLPG=desshlpg,
                    DESSHRP=imageinfo.des_fields.desshrp,
                    DESSHLI=imageinfo.des_fields.desshli,
                    DESSHLIN=imageinfo.des_fields.desshlin,
                    DESSHABS=imageinfo.des_fields.desshabs,
                ),
            )
            des_managers.append(
                sarkit._nitf.nitf.DESSubheaderManager(
                    deshead, lxml.etree.tostring(imageinfo.sidd_xmltree)
                )
            )

        # Product Support XML DES
        for prodinfo in self._nitf_plan.product_support_xmls:
            sidd_uh = des_managers[0].subheader.UserHeader
            xmlns = (
                lxml.etree.QName(prodinfo.product_support_xmltree.getroot()).namespace
                or ""
            )
            deshead = sarkit._nitf.nitf_elements.des.DataExtensionHeader(
                Security=prodinfo.des_fields.security._as_security_tags(),
                UserHeader=sarkit._nitf.nitf_elements.des.XMLDESSubheader(
                    DESSHSI=sidd_uh.DESSHSI,
                    DESSHSV="v" + sidd_uh.DESSHSV,
                    DESSHSD=sidd_uh.DESSHSD,
                    DESSHTN=xmlns,
                    DESSHDT=now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    DESSHLPG="",
                    DESSHRP=prodinfo.des_fields.desshrp,
                    DESSHLI=prodinfo.des_fields.desshli,
                    DESSHLIN=prodinfo.des_fields.desshlin,
                    DESSHABS=prodinfo.des_fields.desshabs,
                ),
            )
            des_managers.append(
                sarkit._nitf.nitf.DESSubheaderManager(
                    deshead, lxml.etree.tostring(prodinfo.product_support_xmltree)
                )
            )

        # SICD XML DES
        for sicd_xml_info in self._nitf_plan.sicd_xmls:
            des_managers.append(
                sicdio._create_des_manager(
                    sicd_xml_info.sicd_xmltree, sicd_xml_info.des_fields
                )
            )

        writing_details = sarkit._nitf.nitf.NITFWritingDetails(
            header,
            image_managers=tuple(image_managers),
            image_segment_collections=tuple(
                (tuple(val) for val in image_segment_collections.values())
            ),
            image_segment_coordinates=tuple(
                (tuple(val) for val in image_segment_coordinates.values())
            ),
            des_managers=tuple(des_managers),
        )

        self._nitf_writer = sarkit._nitf.nitf.NITFWriter(
            file_object=self._file,
            writing_details=writing_details,
        )

    def write_image(self, image_number: int, array: npt.NDArray):
        """Write product pixel data to a NITF file

        Parameters
        ----------
        image_number : int
            index of SIDD Product image to write
        array : ndarray
            2D array of pixels
        """
        self._nitf_writer.write(array, index=image_number)
        self._images_written.add(image_number)

    def write_legend(self, legend_number, array):
        """Write legend pixel data to a NITF file

        Parameters
        ----------
        legend_number : int
            index of legend to write
        array : ndarray
            2D array of pixels
        """
        raise NotImplementedError()

    def write_ded(self, array):
        """Write DED data to a NITF file

        Parameters
        ----------
        array : ndarray
            2D array of pixels
        """
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._nitf_writer.close()
        images_expected = set(range(len(self._nitf_plan.images)))
        images_missing = images_expected - self._images_written
        if images_missing:
            logger.warning(
                f"SIDD Writer closed without writing all images. Missing: {images_missing}"
            )
        # TODO check legends, DED
        return


@dataclasses.dataclass(kw_only=True, frozen=True)
class SegmentationImhdr:
    """Per segment values computed by the SIDD Segmentation Algorithm"""

    idlvl: int
    ialvl: int
    iloc: str
    iid1: str
    nrows: int
    ncols: int


def segmentation_algorithm(
    sidd_xmltrees: collections.abc.Iterable[lxml.etree.ElementTree],
) -> tuple[int, list[int], list[SegmentationImhdr]]:
    """Implementation of section 2.4.2.1 Segmentation Algorithm

    Parameters
    ----------
    sicd_xmltrees : iterable of lxml.etree.ElementTree
        SIDD XML Metadata instances

    Returns
    -------
    fhdr_numi: int
        Number of NITF image segments
    fhdr_li: list of int
        Length of each NITF image segment
    imhdr: list of :py:class:`SegmentationImhdr`
        Image Segment subheader information
    """
    z = 0
    fhdr_numi = 0
    fhdr_li = []
    imhdr = []

    for k, sidd_xmltree in enumerate(sidd_xmltrees):
        xml_helper = sarkit.standards.sidd.xml.XmlHelper(sidd_xmltree)
        pixel_info = PIXEL_TYPES[xml_helper.load("./{*}Display/{*}PixelType")]
        num_rows_k = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Row")
        num_cols_k = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Col")
        bytes_per_pixel = pixel_info[
            "dtype"
        ].itemsize  # Document says NBANDS, but that doesn't work for 16bit
        bytes_per_row = (
            bytes_per_pixel * num_cols_k
        )  # Document says NumRows(k), but that doesn't make sense
        num_rows_limit_k = min(LI_MAX // bytes_per_row, ILOC_MAX)

        product_size = bytes_per_pixel * num_rows_k * num_cols_k
        if product_size <= LI_MAX:
            z += 1
            fhdr_numi += 1
            fhdr_li.append(product_size)
            imhdr.append(
                SegmentationImhdr(
                    idlvl=z,
                    ialvl=0,
                    iloc="0000000000",
                    iid1=f"SIDD{k + 1:03d}001",  # Document says 'm', but there is no m variable
                    nrows=num_rows_k,
                    ncols=num_cols_k,
                )
            )
        else:
            num_seg_per_image_k = int(np.ceil(num_rows_k / num_rows_limit_k))
            z += 1
            fhdr_numi += num_seg_per_image_k
            fhdr_li.append(bytes_per_pixel * num_rows_limit_k * num_cols_k)
            imhdr.append(
                SegmentationImhdr(
                    idlvl=z,
                    ialvl=0,
                    iloc="0000000000",
                    iid1=f"SIDD{k + 1:03d}001",  # Document says 'm', but there is no m variable
                    nrows=num_rows_limit_k,
                    ncols=num_cols_k,
                )
            )
            for n in range(1, num_seg_per_image_k - 1):
                z += 1
                fhdr_li.append(bytes_per_pixel * num_rows_limit_k * num_cols_k)
                imhdr.append(
                    SegmentationImhdr(
                        idlvl=z,
                        ialvl=z - 1,
                        iloc=f"{num_rows_limit_k:05d}00000",
                        iid1=f"SIDD{k + 1:03d}{n + 1:03d}",
                        nrows=num_rows_limit_k,
                        ncols=num_cols_k,
                    )
                )
            z += 1
            last_seg_rows = num_rows_k - (num_seg_per_image_k - 1) * num_rows_limit_k
            fhdr_li.append(bytes_per_pixel * last_seg_rows * num_cols_k)
            imhdr.append(
                SegmentationImhdr(
                    idlvl=z,
                    ialvl=z - 1,
                    iloc=f"{num_rows_limit_k:05d}00000",  # Document says "lastSegRows", but we need the number of rows in the previous IS
                    iid1=f"SIDD{k + 1:03d}{num_seg_per_image_k:03d}",
                    nrows=last_seg_rows,
                    ncols=num_cols_k,
                )
            )

    return fhdr_numi, fhdr_li, imhdr


def _validate_xml(sidd_xmltree):
    """Validate a SIDD XML tree against the schema"""

    xmlns = lxml.etree.QName(sidd_xmltree.getroot()).namespace
    if xmlns not in VERSION_INFO:
        latest_xmlns = list(VERSION_INFO.keys())[-1]
        logger.warning(f"Unknown SIDD namespace {xmlns}, assuming {latest_xmlns}")
        xmlns = latest_xmlns
    schema = lxml.etree.XMLSchema(file=VERSION_INFO[xmlns]["schema"])
    valid = schema.validate(sidd_xmltree)
    if not valid:
        warnings.warn(str(schema.error_log))
    return valid

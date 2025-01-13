import pathlib

import lxml
import numpy as np
import pytest

import sarkit._nitf.nitf
import sarkit.standards.sidd.io as siddio
import sarkit.standards.sidd.xml

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def _random_image(sidd_xmltree):
    xml_helper = sarkit.standards.sidd.xml.XmlHelper(sidd_xmltree)
    rows = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Row")
    cols = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Col")
    shape = (rows, cols)

    assert xml_helper.load("./{*}Display/{*}PixelType") == "MONO8I"

    return np.random.default_rng().integers(
        0, 255, size=shape, dtype=np.uint8, endpoint=True
    )


@pytest.mark.parametrize("force_segmentation", [False, True])
@pytest.mark.parametrize(
    "sidd_xml",
    [
        DATAPATH / "example-sidd-2.0.0.xml",
        DATAPATH / "example-sidd-3.0.0.xml",
    ],
)
def test_roundtrip(force_segmentation, sidd_xml, tmp_path, monkeypatch):
    out_sidd = tmp_path / "out.sidd"
    sicd_xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.4.0.xml")
    basis_etree = lxml.etree.parse(sidd_xml)
    basis_array0 = _random_image(basis_etree)
    basis_array1 = 255 - basis_array0

    if force_segmentation:
        monkeypatch.setattr(
            siddio, "LI_MAX", basis_array0.nbytes // 5
        )  # reduce the segment size limit to force segmentation

    nitf_plan = siddio.SiddNitfPlan(
        header_fields={
            "ostaid": "ostaid",
            "ftitle": "ftitle",
            # Data is unclassified.  These fields are filled for testing purposes only.
            "security": {
                "clas": "T",
                "clsy": "US",
                "code": "code_h",
                "ctlh": "hh",
                "rel": "rel_h",
                "dctp": "DD",
                "dcdt": "20000101",
                "dcxm": "25X1",
                "dg": "C",
                "dgdt": "20000102",
                "cltx": "CW_h",
                "catp": "O",
                "caut": "caut_h",
                "crsn": "A",
                "srdt": "",
                "ctln": "ctln_h",
            },
            "oname": "oname",
            "ophone": "ophone",
        }
    )
    nitf_plan.add_image(
        basis_etree,
        is_fields={
            "tgtid": "tgtid",
            "iid2": "iid2",
            # Data is unclassified.  These fields are filled for testing purposes only.
            "security": {
                "clas": "S",
                "clsy": "II",
                "code": "code_i",
                "ctlh": "ii",
                "rel": "rel_i",
                "dctp": "",
                "dcdt": "",
                "dcxm": "X2",
                "dg": "R",
                "dgdt": "20000202",
                "cltx": "RL_i",
                "catp": "D",
                "caut": "caut_i",
                "crsn": "B",
                "srdt": "20000203",
                "ctln": "ctln_i",
            },
            "icom": ["first comment", "second comment"],
        },
        des_fields={
            # Data is unclassified.  These fields are filled for testing purposes only.
            "security": {
                "clas": "U",
                "clsy": "DD",
                "code": "code_d",
                "ctlh": "dd",
                "rel": "rel_d",
                "dctp": "X",
                "dcdt": "",
                "dcxm": "X3",
                "dg": "",
                "dgdt": "20000302",
                "cltx": "CH_d",
                "catp": "M",
                "caut": "caut_d",
                "crsn": "C",
                "srdt": "20000303",
                "ctln": "ctln_d",
            },
            "desshrp": "desshrp",
            "desshli": "desshli",
            "desshlin": "desshlin",
            "desshabs": "desshabs",
        },
    )

    nitf_plan.add_image(
        basis_etree,
        is_fields={
            "tgtid": "tgtid",
            "iid2": "iid2",
            "security": {
                "clas": "U",
            },
        },
        des_fields={
            "security": {
                "clas": "U",
            },
        },
    )

    nitf_plan.add_sicd_xml(
        sicd_xmltree=sicd_xmltree, des_fields={"security": {"clas": "U"}}
    )

    nitf_plan.add_sicd_xml(
        sicd_xmltree=sicd_xmltree, des_fields={"security": {"clas": "U"}}
    )

    ps_xmltree0 = lxml.etree.ElementTree(
        lxml.etree.fromstring("<product><support/></product>")
    )
    nitf_plan.add_product_support_xml(
        ps_xmltree=ps_xmltree0, des_fields={"security": {"clas": "U"}}
    )
    ps_xmltree1 = lxml.etree.ElementTree(
        lxml.etree.fromstring(
            '<product xmlns="https://example.com"><support/></product>'
        )
    )
    nitf_plan.add_product_support_xml(
        ps_xmltree=ps_xmltree1, des_fields={"security": {"clas": "U"}}
    )

    with out_sidd.open("wb") as file:
        with siddio.SiddNitfWriter(file, nitf_plan) as writer:
            writer.write_image(0, basis_array0)
            writer.write_image(1, basis_array1)

    num_expected_imseg = 2 * int(np.ceil(np.prod(basis_array0.shape) / siddio.LI_MAX))
    if force_segmentation:
        assert num_expected_imseg > 2  # make sure the monkeypatch caused segmentation
    with out_sidd.open("rb") as file:
        nitf_details = sarkit._nitf.nitf.NITFDetails(file)
        assert num_expected_imseg == len(nitf_details.img_headers)

    with out_sidd.open("rb") as file:
        with siddio.SiddNitfReader(file) as reader:
            assert len(reader.images) == 2
            assert len(reader.sicd_xmls) == 2
            assert len(reader.product_support_xmls) == 2
            read_array0 = reader.read_image(0)
            read_array1 = reader.read_image(1)
            read_xmltree = reader.images[0].sidd_xmltree
            read_sicd_xmltree = reader.sicd_xmls[-1].sicd_xmltree
            read_ps_xmltree0 = reader.product_support_xmls[0].product_support_xmltree
            read_ps_xmltree1 = reader.product_support_xmls[1].product_support_xmltree

    def _normalized(xmltree):
        return lxml.etree.tostring(xmltree, method="c14n")

    assert _normalized(read_xmltree) == _normalized(basis_etree)
    assert _normalized(read_ps_xmltree0) == _normalized(ps_xmltree0)
    assert _normalized(read_ps_xmltree1) == _normalized(ps_xmltree1)
    assert _normalized(read_sicd_xmltree) == _normalized(sicd_xmltree)

    assert nitf_plan.header_fields == reader.header_fields
    assert nitf_plan.images[0].is_fields == reader.images[0].is_fields
    assert nitf_plan.images[0].des_fields == reader.images[0].des_fields
    assert np.array_equal(basis_array0, read_array0)
    assert np.array_equal(basis_array1, read_array1)


def test_segmentation():
    """From Figure 2.5-6 SIDD 1.0 Multiple Input Image - Multiple Product Images Requiring Segmentation"""
    sidd_xmltree = lxml.etree.parse(DATAPATH / "example-sidd-3.0.0.xml")
    xml_helper = sarkit.standards.sidd.xml.XmlHelper(sidd_xmltree)
    assert xml_helper.load("./{*}Display/{*}PixelType") == "MONO8I"

    # Tweak SIDD size to force three image segments
    li_max = 9_999_999_998
    iloc_max = 99_999
    num_cols = li_max // (2 * iloc_max)  # set num_cols so that row limit is iloc_max
    last_rows = 24
    num_rows = iloc_max * 2 + last_rows
    xml_helper.set("./{*}Measurement/{*}PixelFootprint/{*}Row", num_rows)
    xml_helper.set("./{*}Measurement/{*}PixelFootprint/{*}Col", num_cols)
    fhdr_numi, fhdr_li, imhdrs = sarkit.standards.sidd.io.segmentation_algorithm(
        [sidd_xmltree, sidd_xmltree]
    )

    assert fhdr_numi == 6
    # SIDD segmentation algorithm (2.4.2.1 in 1.0/2.0/3.0) would lead to overlaps of the last partial
    # image segment due to ILOC. This implements a scheme similar to SICD wherein "RRRRR" of ILOC matches
    # the NROWs in the previous segment.
    expected_imhdrs = [
        siddio.SegmentationImhdr(
            iid1="SIDD001001",
            idlvl=1,
            ialvl=0,
            iloc="0" * 10,
            nrows=iloc_max,
            ncols=num_cols,
        ),
        siddio.SegmentationImhdr(
            iid1="SIDD001002",
            idlvl=2,
            ialvl=1,
            iloc=f"{iloc_max:05d}{0:05d}",
            nrows=iloc_max,
            ncols=num_cols,
        ),
        siddio.SegmentationImhdr(
            iid1="SIDD001003",
            idlvl=3,
            ialvl=2,
            iloc=f"{iloc_max:05d}{0:05d}",
            nrows=last_rows,
            ncols=num_cols,
        ),
        siddio.SegmentationImhdr(
            iid1="SIDD002001",
            idlvl=4,
            ialvl=0,
            iloc="0" * 10,
            nrows=iloc_max,
            ncols=num_cols,
        ),
        siddio.SegmentationImhdr(
            iid1="SIDD002002",
            idlvl=5,
            ialvl=4,
            iloc=f"{iloc_max:05d}{0:05d}",
            nrows=iloc_max,
            ncols=num_cols,
        ),
        siddio.SegmentationImhdr(
            iid1="SIDD002003",
            idlvl=6,
            ialvl=5,
            iloc=f"{iloc_max:05d}{0:05d}",
            nrows=last_rows,
            ncols=num_cols,
        ),
    ]
    expected_fhdr_li = [imhdr.nrows * imhdr.ncols for imhdr in expected_imhdrs]

    assert expected_fhdr_li == fhdr_li
    assert expected_imhdrs == imhdrs


def test_version_info():
    actual_order = [
        x["version"] for x in sarkit.standards.sidd.io.VERSION_INFO.values()
    ]
    expected_order = sorted(actual_order, key=lambda x: x.split("."))
    assert actual_order == expected_order

    for urn, info in sarkit.standards.sidd.io.VERSION_INFO.items():
        assert lxml.etree.parse(info["schema"]).getroot().get("targetNamespace") == urn

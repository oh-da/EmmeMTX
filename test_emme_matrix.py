"""Tests for :mod:`emme_matrix`."""

import numpy as np
import pandas as pd
import pytest

from emme_matrix import EmmeMatrix, read_emme_matrix

SAMPLE = """\
t matrices
d mf99
c EMME Module:    3.14(v18000)  Date: 25-12-29 03:49   User: BSY /CONNECT ..TAM
c Project:        Tel Aviv Model version 4.31
t matrices
a matrix=mf99   TrDm_p                                       0 p: Transit Tot Demand
 1101 100:.101941 101:.130543
 1101 102:1.035360 105:.103899
 100 6011:.200000 6101:.100000
 100 6302:.200000
"""


@pytest.fixture()
def sample_file(tmp_path):
    # Written with CRLF endings to mirror the real Emme exports.
    path = tmp_path / "sample.txt"
    path.write_bytes(SAMPLE.replace("\n", "\r\n").encode("utf-8"))
    return path


def test_metadata(sample_file):
    mtx = read_emme_matrix(sample_file)
    md = mtx.metadata
    assert md.matrix_id == "mf99"
    assert md.matrix_name == "TrDm_p"
    assert md.description == "p: Transit Tot Demand"
    assert md.date == "25-12-29 03:49"
    assert md.user == "BSY /CONNECT ..TAM"
    assert md.project == "Tel Aviv Model version 4.31"
    assert md.emme_module == "3.14(v18000)"


def test_taz_ids_are_sorted_union(sample_file):
    mtx = read_emme_matrix(sample_file)
    expected = sorted({1101, 100, 101, 102, 105, 6011, 6101, 6302})
    assert list(mtx.taz_ids) == expected


def test_array_and_dataframe_values(sample_file):
    mtx = read_emme_matrix(sample_file)
    df = mtx.dataframe
    assert isinstance(mtx.array, np.ndarray)
    assert df.shape == (8, 8)
    # Leading-dot decimals parse correctly.
    assert df.loc[1101, 100] == pytest.approx(0.101941)
    assert df.loc[1101, 101] == pytest.approx(0.130543)
    assert df.loc[1101, 102] == pytest.approx(1.035360)
    assert df.loc[100, 6011] == pytest.approx(0.200000)
    assert df.loc[100, 6302] == pytest.approx(0.200000)
    # Unreported pair -> fill value (0 by default).
    assert df.loc[100, 100] == 0.0
    assert df.index.name == "origin"
    assert df.columns.name == "destination"


def test_fill_value_nan(sample_file):
    mtx = read_emme_matrix(sample_file, fill_value=np.nan)
    df = mtx.dataframe
    assert np.isnan(df.loc[100, 100])
    # Reported cells keep their exact value, not fill + value.
    assert df.loc[1101, 100] == pytest.approx(0.101941)


def test_duplicate_pairs_accumulate(tmp_path):
    text = SAMPLE + " 1101 100:1.000000\n"
    path = tmp_path / "dup.txt"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    mtx = read_emme_matrix(path)
    # 0.101941 (original) + 1.0 (duplicate) accumulate.
    assert mtx.dataframe.loc[1101, 100] == pytest.approx(1.101941)


def test_long_dataframe(sample_file):
    mtx = read_emme_matrix(sample_file)
    long = mtx.to_long_dataframe()
    assert list(long.columns) == ["origin", "destination", "trips"]
    assert len(long) == 7  # seven reported pairs in the sample
    row = long[(long.origin == 1101) & (long.destination == 102)]
    assert row.trips.iloc[0] == pytest.approx(1.035360)


def test_malformed_token_raises(tmp_path):
    text = SAMPLE + " 200 not_a_pair\n"
    path = tmp_path / "bad.txt"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    with pytest.raises(ValueError, match="Malformed"):
        read_emme_matrix(path)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        read_emme_matrix("does_not_exist.txt")


def test_to_dataframe_controllable_fill(sample_file):
    mtx = read_emme_matrix(sample_file)  # default fill 0
    df0 = mtx.to_dataframe(fill_value=0)
    dfnan = mtx.to_dataframe(fill_value=np.nan)
    assert df0.loc[100, 100] == 0.0
    assert np.isnan(dfnan.loc[100, 100])
    # Reported cells are identical regardless of the fill value.
    assert df0.loc[1101, 100] == pytest.approx(0.101941)
    assert dfnan.loc[1101, 100] == pytest.approx(0.101941)


def _matrix_from(text, tmp_path, name):
    path = tmp_path / name
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    return read_emme_matrix(path)


# A small 2-TAZ matrix (ids 100, 101) and a larger one sharing only some ids.
SMALL = """\
t matrices
d mf1
c EMME Module: 3.14 Date: 25-12-29 03:49 User: X
c Project: demo
t matrices
a matrix=mf1 SMALL 0 small
 100 100:1.000000 101:2.000000
 101 100:3.000000 101:4.000000
"""

BIG = """\
t matrices
d mf2
c EMME Module: 3.14 Date: 25-12-29 03:49 User: X
c Project: demo
t matrices
a matrix=mf2 BIG 0 big
 100 100:10.000000 101:20.000000
 101 100:30.000000 101:40.000000
 200 200:5.000000
"""


def test_add_different_sized_matrices(tmp_path):
    small = _matrix_from(SMALL, tmp_path, "small.txt")
    big = _matrix_from(BIG, tmp_path, "big.txt")

    result = (small + big).to_dataframe()
    # Union of TAZ ids -> 100, 101, 200 on both axes.
    assert list(result.index) == [100, 101, 200]
    assert list(result.columns) == [100, 101, 200]
    # Overlapping cells are summed.
    assert result.loc[100, 100] == pytest.approx(11.0)
    assert result.loc[101, 101] == pytest.approx(44.0)
    # Cell present only in the big matrix: small contributes 0.
    assert result.loc[200, 200] == pytest.approx(5.0)
    # Cell absent from both: 0.
    assert result.loc[100, 200] == pytest.approx(0.0)


def test_subtract_and_scalar_ops(tmp_path):
    small = _matrix_from(SMALL, tmp_path, "small.txt")
    big = _matrix_from(BIG, tmp_path, "big.txt")

    diff = (big - small).to_dataframe()
    assert diff.loc[100, 100] == pytest.approx(9.0)
    assert diff.loc[200, 200] == pytest.approx(5.0)  # small has 0 here

    scaled = (small * 2).to_dataframe()
    assert scaled.loc[100, 101] == pytest.approx(4.0)


def test_combine_three_matrices(tmp_path):
    small = _matrix_from(SMALL, tmp_path, "s.txt")
    big = _matrix_from(BIG, tmp_path, "b.txt")
    total = EmmeMatrix.combine([small, big, small], op="add")
    df = total.to_dataframe()
    # 100->100: 1 + 10 + 1
    assert df.loc[100, 100] == pytest.approx(12.0)
    assert df.loc[200, 200] == pytest.approx(5.0)


def test_from_dataframe_roundtrip():
    df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index([100, 101], name="origin"),
        columns=pd.Index([100, 101], name="destination"),
    )
    mtx = EmmeMatrix.from_dataframe(df)
    assert list(mtx.taz_ids) == [100, 101]
    assert mtx.to_dataframe().loc[101, 100] == pytest.approx(3.0)

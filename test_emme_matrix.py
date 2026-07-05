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

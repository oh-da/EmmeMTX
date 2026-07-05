"""Parser for INRO Emme matrix export files.

Emme (by INRO) exports origin-destination trip matrices as plain-text files.
The layout is::

    t matrices                                          <- line 1  (metadata)
    d mf99                                              <- line 2  (metadata)
    c EMME Module: 3.14(v18000) Date: ... User: ...     <- line 3  (metadata)
    c Project: Tel Aviv Model version 4.31              <- line 4  (metadata)
    t matrices                                          <- line 5  (metadata)
    a matrix=mf99  TrDm_p  0 p: Transit Tot Demand      <- line 6  (metadata)
     1101 100:.101941 101:.130543                       <- line 7+ (data)
     1101 102:1.035360 105:.103899
     ...

Each data row starts with the origin TAZ id followed by one or more
``destination:trips`` pairs (usually two, occasionally one). Trip values may be
written with a leading dot (``.101941`` == ``0.101941``).

This module reads such a file into a dense (square) :class:`numpy.ndarray`
indexed by TAZ id and exposes it as a labelled :class:`pandas.DataFrame`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

#: Number of leading metadata rows before the trip data begins.
N_METADATA_ROWS = 6


@dataclass
class EmmeMatrixMetadata:
    """Structured view of the six metadata rows at the top of the file."""

    raw_lines: List[str] = field(default_factory=list)
    matrix_id: Optional[str] = None
    matrix_name: Optional[str] = None
    description: Optional[str] = None
    emme_module: Optional[str] = None
    date: Optional[str] = None
    user: Optional[str] = None
    project: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"EmmeMatrixMetadata(matrix_id={self.matrix_id!r}, "
            f"matrix_name={self.matrix_name!r}, description={self.description!r}, "
            f"date={self.date!r}, user={self.user!r}, project={self.project!r})"
        )


class EmmeMatrix:
    """Read an INRO Emme trip-matrix export into a numpy array / pandas DataFrame.

    Parameters
    ----------
    filepath:
        Path to the Emme matrix text file.
    fill_value:
        Value used for origin/destination pairs that are absent from the file
        (i.e. no trips reported). Defaults to ``0.0``. Use ``np.nan`` to
        distinguish "no trips" from "not reported".
    encoding:
        File encoding, defaults to ``"utf-8"``.

    Examples
    --------
    >>> mtx = EmmeMatrix("TransitTotDemand_p.txt").parse()
    >>> mtx.array.shape            # doctest: +SKIP
    (1293, 1293)
    >>> df = mtx.dataframe          # doctest: +SKIP
    >>> df.loc[1101, 100]           # trips from TAZ 1101 to TAZ 100  # doctest: +SKIP
    0.101941
    """

    # Matches "destination:trips" tokens, e.g. "101:.130543" or "6011:.200000".
    _PAIR_RE = re.compile(r"^(\d+):([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")

    def __init__(
        self,
        filepath: Union[str, Path],
        fill_value: float = 0.0,
        encoding: str = "utf-8",
    ) -> None:
        self.filepath = Path(filepath)
        self.fill_value = fill_value
        self.encoding = encoding

        self.metadata: EmmeMatrixMetadata = EmmeMatrixMetadata()
        #: Sorted list of every TAZ id seen as an origin or a destination.
        self.taz_ids: np.ndarray = np.empty(0, dtype=np.int64)
        #: Dense square trip matrix aligned with :attr:`taz_ids`.
        self.array: Optional[np.ndarray] = None

        self._records: List[Tuple[int, int, float]] = []

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def parse(self) -> "EmmeMatrix":
        """Read the file, populating :attr:`metadata`, :attr:`taz_ids` and
        :attr:`array`. Returns ``self`` so calls can be chained."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Emme matrix file not found: {self.filepath}")

        with self.filepath.open("r", encoding=self.encoding) as fh:
            # ``open`` in text mode normalises CRLF -> LF for us.
            meta_lines: List[str] = []
            for _ in range(N_METADATA_ROWS):
                line = fh.readline()
                if line == "":  # file shorter than the metadata header
                    break
                meta_lines.append(line.rstrip("\n"))
            self.metadata = self._parse_metadata(meta_lines)

            self._records = list(self._parse_data_lines(fh))

        self._build_matrix()
        return self

    def _parse_metadata(self, lines: List[str]) -> EmmeMatrixMetadata:
        meta = EmmeMatrixMetadata(raw_lines=list(lines))
        for line in lines:
            stripped = line.strip()

            # "c EMME Module: 3.14(v18000)  Date: 25-12-29 03:49  User: BSY ..."
            if stripped.startswith("c") and "EMME Module" in stripped:
                m = re.search(r"EMME Module:\s*(.+?)\s+Date:", stripped)
                if m:
                    meta.emme_module = m.group(1).strip()
                m = re.search(r"Date:\s*(.+?)\s+User:", stripped)
                if m:
                    meta.date = m.group(1).strip()
                m = re.search(r"User:\s*(.+)$", stripped)
                if m:
                    meta.user = m.group(1).strip()

            # "c Project: Tel Aviv Model version 4.31"
            elif stripped.startswith("c") and "Project" in stripped:
                m = re.search(r"Project:\s*(.+)$", stripped)
                if m:
                    meta.project = m.group(1).strip()

            # "a matrix=mf99   TrDm_p   0 p: Transit Tot Demand"
            elif stripped.startswith("a") and "matrix=" in stripped:
                m = re.search(r"matrix=(\S+)", stripped)
                if m:
                    meta.matrix_id = m.group(1)
                # Remaining tokens after the matrix id: name, then description.
                rest = stripped[m.end():].strip() if m else ""
                tokens = rest.split(None, 1)
                if tokens:
                    meta.matrix_name = tokens[0]
                if len(tokens) > 1:
                    # Drop a leading numeric "default value" column if present,
                    # keep the human-readable description that follows.
                    desc = tokens[1].strip()
                    desc = re.sub(r"^\d+\s+", "", desc)
                    meta.description = desc.strip()
        return meta

    def _parse_data_lines(
        self, lines: Iterable[str]
    ) -> Iterable[Tuple[int, int, float]]:
        """Yield ``(origin, destination, trips)`` tuples from the data section."""
        for lineno, raw in enumerate(lines, start=N_METADATA_ROWS + 1):
            fields = raw.split()
            if not fields:
                continue  # skip blank lines
            try:
                origin = int(fields[0])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid origin TAZ on line {lineno}: {raw!r}"
                ) from exc

            for token in fields[1:]:
                match = self._PAIR_RE.match(token)
                if not match:
                    raise ValueError(
                        f"Malformed 'destination:trips' token {token!r} "
                        f"on line {lineno}: {raw!r}"
                    )
                dest = int(match.group(1))
                trips = float(match.group(2))
                yield origin, dest, trips

    def _build_matrix(self) -> None:
        """Turn the parsed records into a dense square numpy array."""
        if not self._records:
            self.taz_ids = np.empty(0, dtype=np.int64)
            self.array = np.empty((0, 0), dtype=np.float64)
            return

        origins = np.fromiter((r[0] for r in self._records), dtype=np.int64)
        dests = np.fromiter((r[1] for r in self._records), dtype=np.int64)
        trips = np.fromiter((r[2] for r in self._records), dtype=np.float64)

        # Union of all TAZ ids -> a square matrix so every origin can reach
        # every destination.
        self.taz_ids = np.unique(np.concatenate([origins, dests]))
        index: Dict[int, int] = {taz: i for i, taz in enumerate(self.taz_ids)}

        n = self.taz_ids.size
        row_idx = np.fromiter((index[o] for o in origins), dtype=np.int64)
        col_idx = np.fromiter((index[d] for d in dests), dtype=np.int64)

        # Accumulate onto a zero base so duplicate pairs sum correctly.
        matrix = np.zeros((n, n), dtype=np.float64)
        np.add.at(matrix, (row_idx, col_idx), trips)

        # Cells with no reported trips take ``fill_value`` (may be NaN).
        if not (self.fill_value == 0):
            touched = np.zeros((n, n), dtype=bool)
            touched[row_idx, col_idx] = True
            matrix[~touched] = self.fill_value

        self.array = matrix

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    @property
    def dataframe(self) -> pd.DataFrame:
        """Square trip matrix as a DataFrame indexed/columned by TAZ id.

        Rows are origins, columns are destinations.
        """
        if self.array is None:
            raise RuntimeError("Call parse() before accessing the DataFrame.")
        return pd.DataFrame(
            self.array,
            index=pd.Index(self.taz_ids, name="origin"),
            columns=pd.Index(self.taz_ids, name="destination"),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Alias for the :attr:`dataframe` property."""
        return self.dataframe

    def to_long_dataframe(self, drop_zeros: bool = True) -> pd.DataFrame:
        """Return trips in long/tidy form with ``origin``/``destination``/``trips``.

        Parameters
        ----------
        drop_zeros:
            When ``True`` (default) only origin/destination pairs present in the
            source file are returned, mirroring the sparse input.
        """
        if self.array is None:
            raise RuntimeError("Call parse() before accessing the DataFrame.")
        if drop_zeros:
            df = pd.DataFrame(
                self._records, columns=["origin", "destination", "trips"]
            )
            # Collapse duplicates the same way the dense matrix does.
            return (
                df.groupby(["origin", "destination"], as_index=False)["trips"]
                .sum()
                .sort_values(["origin", "destination"], ignore_index=True)
            )
        long = self.dataframe.stack()
        long.name = "trips"
        return long.reset_index()

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        shape = None if self.array is None else self.array.shape
        return (
            f"EmmeMatrix(filepath={str(self.filepath)!r}, "
            f"matrix_id={self.metadata.matrix_id!r}, shape={shape})"
        )


def read_emme_matrix(
    filepath: Union[str, Path],
    fill_value: float = 0.0,
    encoding: str = "utf-8",
) -> EmmeMatrix:
    """Convenience helper: build and :meth:`~EmmeMatrix.parse` in one call."""
    return EmmeMatrix(filepath, fill_value=fill_value, encoding=encoding).parse()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    argp = argparse.ArgumentParser(description="Parse an INRO Emme matrix file.")
    argp.add_argument("filepath", help="Path to the Emme matrix text file")
    argp.add_argument(
        "--fill", type=float, default=0.0, help="Fill value for missing pairs"
    )
    args = argp.parse_args()

    mtx = read_emme_matrix(args.filepath, fill_value=args.fill)
    print(mtx)
    print(mtx.metadata)
    print(f"TAZ count: {mtx.taz_ids.size}")
    print(f"Matrix shape: {mtx.array.shape}")
    print(mtx.dataframe.iloc[:5, :5])

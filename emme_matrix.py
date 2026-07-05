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
Loaded matrices can be combined arithmetically (added, subtracted, ...); the
operands are aligned on the union of their TAZ ids and any id missing from one
matrix is treated as zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

#: Number of leading metadata rows before the trip data begins.
N_METADATA_ROWS = 6

Number = Union[int, float]
Operand = Union["EmmeMatrix", pd.DataFrame, Number]


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
        Path to the Emme matrix text file. May be ``None`` for matrices built
        in memory (e.g. the result of an arithmetic operation).
    fill_value:
        Default value used for origin/destination pairs that are absent from
        the file (i.e. no trips reported). Defaults to ``0.0``. Use ``np.nan``
        to distinguish "no trips" from "not reported". This is only the
        *default* — :meth:`to_dataframe` lets you override it per call.
    encoding:
        File encoding, defaults to ``"utf-8"``.

    Examples
    --------
    >>> mtx = EmmeMatrix("TransitTotDemand_p.txt").parse()
    >>> mtx.array.shape                     # doctest: +SKIP
    (1293, 1293)
    >>> df = mtx.to_dataframe(fill_value=0)  # nulls -> 0  # doctest: +SKIP
    >>> df.loc[1101, 100]                    # trips 1101 -> 100  # doctest: +SKIP
    0.101941

    Combine two matrices of different sizes (missing TAZ ids treated as 0)::

    >>> combined = auto + transit          # doctest: +SKIP
    >>> combined.to_dataframe()            # doctest: +SKIP
    """

    # Matches "destination:trips" tokens, e.g. "101:.130543" or "6011:.200000".
    _PAIR_RE = re.compile(r"^(\d+):([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")

    def __init__(
        self,
        filepath: Union[str, Path, None],
        fill_value: float = 0.0,
        encoding: str = "utf-8",
    ) -> None:
        self.filepath = Path(filepath) if filepath is not None else None
        self.fill_value = fill_value
        self.encoding = encoding

        self.metadata: EmmeMatrixMetadata = EmmeMatrixMetadata()
        #: Sorted list of every TAZ id seen as an origin or a destination.
        self.taz_ids: np.ndarray = np.empty(0, dtype=np.int64)
        #: Dense square trip matrix aligned with :attr:`taz_ids`.
        self.array: Optional[np.ndarray] = None

        self._records: Optional[List[Tuple[int, int, float]]] = None
        # Positional indices into ``taz_ids`` for each parsed record, kept so
        # the dense array can be rebuilt cheaply with a different fill value.
        self._row_idx: Optional[np.ndarray] = None
        self._col_idx: Optional[np.ndarray] = None
        self._trips: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Alternate constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        fill_value: float = 0.0,
        metadata: Optional[EmmeMatrixMetadata] = None,
    ) -> "EmmeMatrix":
        """Build an :class:`EmmeMatrix` from a square (origin x destination)
        DataFrame. The result is reindexed to the sorted union of the index and
        column TAZ ids so it stays square; new cells take ``fill_value``."""
        obj = cls(filepath=None, fill_value=fill_value)
        obj.metadata = metadata if metadata is not None else EmmeMatrixMetadata()

        ids = np.unique(
            np.concatenate(
                [
                    np.asarray(df.index, dtype=np.int64),
                    np.asarray(df.columns, dtype=np.int64),
                ]
            )
        )
        square = df.reindex(index=ids, columns=ids)
        obj.taz_ids = ids
        obj.array = square.to_numpy(dtype=np.float64, na_value=fill_value)
        return obj

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def parse(self) -> "EmmeMatrix":
        """Read the file, populating :attr:`metadata`, :attr:`taz_ids` and
        :attr:`array`. Returns ``self`` so calls can be chained."""
        if self.filepath is None:
            raise ValueError("No filepath to parse (matrix was built in memory).")
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

        self._build_index()
        self.array = self._build_array(self.fill_value)
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
                    desc = re.sub(r"^\d+\s+", "", tokens[1].strip())
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

    def _build_index(self) -> None:
        """Compute the TAZ id union and the positional indices of every record."""
        records = self._records or []
        if not records:
            self.taz_ids = np.empty(0, dtype=np.int64)
            self._row_idx = np.empty(0, dtype=np.int64)
            self._col_idx = np.empty(0, dtype=np.int64)
            self._trips = np.empty(0, dtype=np.float64)
            return

        origins = np.fromiter((r[0] for r in records), dtype=np.int64)
        dests = np.fromiter((r[1] for r in records), dtype=np.int64)
        self._trips = np.fromiter((r[2] for r in records), dtype=np.float64)

        # Union of all TAZ ids -> a square matrix so every origin can reach
        # every destination.
        self.taz_ids = np.unique(np.concatenate([origins, dests]))
        pos: Dict[int, int] = {taz: i for i, taz in enumerate(self.taz_ids)}
        self._row_idx = np.fromiter((pos[o] for o in origins), dtype=np.int64)
        self._col_idx = np.fromiter((pos[d] for d in dests), dtype=np.int64)

    def _build_array(self, fill_value: float) -> np.ndarray:
        """Assemble a dense square array from the parsed records."""
        n = self.taz_ids.size
        if n == 0:
            return np.empty((0, 0), dtype=np.float64)

        # Accumulate onto a zero base so duplicate pairs sum correctly.
        matrix = np.zeros((n, n), dtype=np.float64)
        np.add.at(matrix, (self._row_idx, self._col_idx), self._trips)

        # Cells with no reported trips take ``fill_value`` (may be NaN).
        if not (fill_value == 0):
            touched = np.zeros((n, n), dtype=bool)
            touched[self._row_idx, self._col_idx] = True
            matrix[~touched] = fill_value
        return matrix

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def to_dataframe(self, fill_value: Optional[float] = None) -> pd.DataFrame:
        """Square trip matrix as a DataFrame indexed/columned by TAZ id.

        Rows are origins, columns are destinations.

        Parameters
        ----------
        fill_value:
            Value for origin/destination pairs absent from the source file.
            Defaults to the instance's ``fill_value``. Pass ``0`` to fill nulls
            with zero, or ``np.nan`` to leave them empty.
        """
        if self.array is None:
            raise RuntimeError("Call parse() before accessing the DataFrame.")

        if fill_value is None or fill_value == self.fill_value:
            array = self.array
        elif self._row_idx is not None:
            array = self._build_array(fill_value)
        else:
            # In-memory matrix (no records): swap the current fill for the new.
            array = self.array.copy()
            if self.fill_value == 0 or np.isnan(self.fill_value):
                mask = (
                    np.isnan(array)
                    if np.isnan(self.fill_value)
                    else (array == 0)
                )
                array = array.copy()
                array[mask] = fill_value

        return pd.DataFrame(
            array,
            index=pd.Index(self.taz_ids, name="origin"),
            columns=pd.Index(self.taz_ids, name="destination"),
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        """Square trip matrix as a DataFrame using the default ``fill_value``."""
        return self.to_dataframe()

    def to_long_dataframe(self, drop_zeros: bool = True) -> pd.DataFrame:
        """Return trips in long/tidy form with ``origin``/``destination``/``trips``.

        Parameters
        ----------
        drop_zeros:
            When ``True`` (default) only non-zero origin/destination pairs are
            returned, mirroring the sparse input.
        """
        if self.array is None:
            raise RuntimeError("Call parse() before accessing the DataFrame.")

        if self._records is not None:
            df = pd.DataFrame(
                self._records, columns=["origin", "destination", "trips"]
            )
            df = (
                df.groupby(["origin", "destination"], as_index=False)["trips"]
                .sum()
                .sort_values(["origin", "destination"], ignore_index=True)
            )
        else:
            long = self.to_dataframe(fill_value=0).stack()
            long.name = "trips"
            df = long.reset_index()

        if drop_zeros:
            df = df[df["trips"] != 0].reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # Arithmetic between matrices
    # ------------------------------------------------------------------ #
    def _combine(self, other: Operand, op: str, fill_value: float) -> "EmmeMatrix":
        """Align ``self`` and ``other`` on the union of their TAZ ids and apply
        ``op`` element-wise. Cells missing from either operand use ``fill_value``
        (default 0), so a smaller matrix behaves like a full matrix padded with
        zeros."""
        left = self.to_dataframe(fill_value=0)

        if isinstance(other, EmmeMatrix):
            right: Operand = other.to_dataframe(fill_value=0)
        elif isinstance(other, pd.DataFrame):
            right = other
        else:  # scalar
            right = other

        result = getattr(left, op)(right, fill_value=fill_value)
        return EmmeMatrix.from_dataframe(result, fill_value=self.fill_value)

    def add(self, other: Operand, fill_value: float = 0) -> "EmmeMatrix":
        """Add another matrix/DataFrame/scalar, aligning on the TAZ id union."""
        return self._combine(other, "add", fill_value)

    def subtract(self, other: Operand, fill_value: float = 0) -> "EmmeMatrix":
        """Subtract another matrix/DataFrame/scalar, aligning on the TAZ union."""
        return self._combine(other, "sub", fill_value)

    def multiply(self, other: Operand, fill_value: float = 0) -> "EmmeMatrix":
        """Multiply by another matrix/DataFrame/scalar, aligning on the union."""
        return self._combine(other, "mul", fill_value)

    def divide(self, other: Operand, fill_value: float = 0) -> "EmmeMatrix":
        """Divide by another matrix/DataFrame/scalar, aligning on the union."""
        return self._combine(other, "div", fill_value)

    # Operators so you can write ``a + b``, ``a - b``, ``a + b + c`` ...
    __add__ = add
    __sub__ = subtract
    __mul__ = multiply
    __truediv__ = divide

    @staticmethod
    def combine(
        matrices: Sequence["EmmeMatrix"],
        op: str = "add",
        fill_value: float = 0,
    ) -> "EmmeMatrix":
        """Combine two or more matrices with a single operation.

        Parameters
        ----------
        matrices:
            The matrices to combine, left to right.
        op:
            One of ``"add"``, ``"subtract"``, ``"multiply"``, ``"divide"``.
        fill_value:
            Value used for TAZ ids missing from a given matrix (default 0).

        Examples
        --------
        >>> total = EmmeMatrix.combine([auto, transit, walk])  # doctest: +SKIP
        """
        if not matrices:
            raise ValueError("combine() requires at least one matrix.")
        method = {
            "add": "add",
            "subtract": "subtract",
            "sub": "subtract",
            "multiply": "multiply",
            "mul": "multiply",
            "divide": "divide",
            "div": "divide",
        }.get(op)
        if method is None:
            raise ValueError(f"Unsupported op {op!r}.")

        result = matrices[0]
        for nxt in matrices[1:]:
            result = getattr(result, method)(nxt, fill_value=fill_value)
        return result

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        shape = None if self.array is None else self.array.shape
        src = str(self.filepath) if self.filepath is not None else "<in-memory>"
        return (
            f"EmmeMatrix(source={src!r}, "
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
    print(mtx.to_dataframe(fill_value=args.fill).iloc[:5, :5])

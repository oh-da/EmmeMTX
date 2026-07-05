# EmmeMTX

Parse INRO **Emme** origin-destination trip-matrix export files into a
[NumPy](https://numpy.org/) array and a labelled [pandas](https://pandas.pydata.org/)
`DataFrame`.

## File format

Emme exports a matrix as a plain-text file. The first **6 rows** are metadata
(software version, date, user, project and the matrix definition). The trip
data starts on **row 7**:

```
t matrices
d mf99
c EMME Module:    3.14(v18000)  Date: 25-12-29 03:49   User: BSY /CONNECT ..TAM
c Project:        Tel Aviv Model version 4.31
t matrices
a matrix=mf99   TrDm_p   0 p: Transit Tot Demand
 1101 100:.101941 101:.130543
 1101 102:1.035360 105:.103899
 100 6302:.200000
```

Each data row is:

```
<origin_TAZ> <dest_TAZ>:<trips> [<dest_TAZ>:<trips>]
```

* The first column is the origin TAZ id.
* The following one or two columns are `destination:trips` pairs.
* Trip values may use a leading dot (`.101941` == `0.101941`).

## Usage

### 1. Load a file

```python
from emme_matrix import read_emme_matrix

mtx = read_emme_matrix("TransitTotDemand_p.txt")

mtx.array            # dense square numpy.ndarray of trips
mtx.taz_ids          # sorted numpy array of every TAZ id
mtx.metadata         # parsed metadata (matrix id/name, date, user, project, ...)
```

### 2. Transform into a DataFrame with a controllable fill value

Rows are origins, columns are destinations. Choose what unreported pairs
become:

```python
df = mtx.to_dataframe(fill_value=0)       # fill nulls with 0
df = mtx.to_dataframe(fill_value=np.nan)  # leave nulls empty

df.loc[1101, 100]    # trips from TAZ 1101 to TAZ 100  ->  0.101941
```

The matrix is **square**: its index and columns are the sorted union of every
TAZ id seen as an origin or destination.

Long / tidy form:

```python
mtx.to_long_dataframe()   # columns: origin, destination, trips (non-zero pairs)
```

### 3. Calculate across two or more matrices

Operations align on the **union of TAZ ids**. If one matrix is smaller, the ids
it doesn't contain are treated as `0` — i.e. every matrix is handled as if it
were a full matrix padded with zeros.

```python
auto    = read_emme_matrix("AUTOTOT_a.txt")        # e.g. 1360 x 1360
transit = read_emme_matrix("TransitTotDemand_p.txt")  # e.g. 1351 x 1351

total = auto + transit          # union-aligned, missing ids -> 0
total.to_dataframe()            # 1360 x 1360 result

# also: -, *, / and named methods with an explicit fill_value
diff   = auto.subtract(transit, fill_value=0)
scaled = auto * 2

# combine 2+ matrices in one call
grand_total = EmmeMatrix.combine([auto, transit, walk], op="add")
```

You can also mix in a plain DataFrame or a scalar as the right-hand operand.

### Create an empty matrix from a list of TAZ ids

Build a blank (zero-filled) square matrix — a template to fill in or add to
other matrices. The ids label both rows and columns.

```python
from emme_matrix import EmmeMatrix

mtx = EmmeMatrix.empty([100, 101, 200])            # 3 x 3, all zeros
mtx = EmmeMatrix.empty(taz_ids, fill_value=np.nan)  # or any fill value
```

Read the ids from a **CSV or txt file**:

```python
# txt: one id per line, or whitespace/comma separated
mtx = EmmeMatrix.empty_from_file("tazs.txt")

# csv: pick the id column by name or 0-based index
mtx = EmmeMatrix.empty_from_file("zones.csv", column="taz_id")

# or read the ids yourself
ids = EmmeMatrix.read_taz_ids("zones.csv", column="taz_id")
mtx = EmmeMatrix.empty(ids)
```

A flat file with a header line works too — non-numeric tokens (like the header)
are skipped automatically.

Command line:

```bash
python emme_matrix.py TransitTotDemand_p.txt
```

## Install & test

```bash
pip install -r requirements.txt
pytest
```

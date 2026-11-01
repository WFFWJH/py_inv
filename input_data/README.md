# Input data and reproducibility

The observation grids are external project inputs and are not stored in Git.
`datav0.list` resolves each track path relative to the repository root; the
current list expects tracks under `input_data/data_v5_opt1/`.

For each configured track, the downsampling workflow expects the LOS grid and
the corresponding DEM and look-direction grids described in the root
[README](../README.md#数据组织). To run inversion without repeating
downsampling, provide `los_samp{iter_step}.mat` for every track. These grids
and MAT files are ignored by Git because they are large generated or source
data, not because they are optional.

For reproducible runs, keep a local data manifest alongside the dataset that
records the source/version, track identifier, acquisition date, preprocessing
steps, and SHA-256 checksum for each input grid and sampled MAT file. Do not
commit the data itself or credentials for obtaining it.

The inversion snapshot and primary figures are written to the project root by
default; `py-inv --output-dir <directory>` places those products elsewhere.
Per-track intermediate files and diagnostics remain with their track data,
because later workflow phases read them there. Scan commands also support
`--out-dir`. Preserve the configuration files and data manifest with each
scientific result.

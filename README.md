# NCNM Parallel Discontinuity Recognition

This is the core Python implementation of **Normal-Consistent Neighbor Merging
(NCNM)** for recognizing planar discontinuities in large-scale 3D point clouds.

Only the parallel NCNM algorithm file is intended for this GitHub repository:

```text
ncnm_parallel_recognition.py
```

Auxiliary ablation scripts, plotting scripts, validation outputs, raw point
clouds, IDE settings, caches, and local experiment results are intentionally
excluded from Git by `.gitignore`.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the core NCNM recognizer:

```powershell
python ncnm_parallel_recognition.py --input "path\to\point_cloud.pcd" --output ncnm_results --voxel-size 0.10 --radius-factor 3.0 --edge-angle-threshold 10 --refine-angle-threshold 8 --workers 8 --no-visualize
```

Common options:

- `--input`: input point cloud path.
- `--output`: output directory, default `ncnm_results`.
- `--voxel-size`: voxel downsampling size; set `0` to disable downsampling.
- `--neighbor-radius`: absolute neighborhood radius; set `0` to estimate from
  median point spacing.
- `--radius-factor`: multiplier used when `--neighbor-radius 0`.
- `--refine-angle-threshold`: normal-refinement angle threshold in degrees.
- `--edge-angle-threshold`: normal-consistent neighbor threshold in degrees.
- `--min-component-size`: minimum retained discontinuity size.
- `--workers`: number of parallel worker processes.
- `--no-visualize`: skip the interactive Open3D viewer.

Supported input formats include `.pcd`, `.ply`, `.xyz`, `.pts`, `.txt`, `.csv`,
`.npy`, `.npz`, `.las`, and `.laz`.

## Outputs

Each run creates a timestamped folder in the selected output directory. The main
outputs include:

- colored `.pcd` and `.ply` point clouds;
- structural point-index text files;
- sampled attitude-coordinate text files;
- metadata and group-summary files.

Open the colored point cloud in CloudCompare or another point-cloud viewer to
inspect recognized discontinuities.

## Data and GitHub

Raw point-cloud data are usually too large for a normal GitHub repository. Keep
data local, publish it separately, or use Git LFS only if you intentionally want
GitHub to manage large binary files.

The current `.gitignore` is whitelist-based. If you run:

```bash
git add .
```

only these files should be staged:

```text
.gitignore
README.md
requirements.txt
ncnm_parallel_recognition.py
```

## Suggested upload commands

```bash
git init
git add .
git commit -m "Initial NCNM core algorithm release"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/NCNM.git
git push -u origin main
```

Before making the repository public, choose a license such as MIT, BSD-3-Clause,
or Apache-2.0 if you want others to have clear reuse permission.

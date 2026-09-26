"""
Parallel Normal-Consistent Neighbor Merging (NCNM) for planar-discontinuity
recognition in large-scale 3D point clouds.

Method implemented from the manuscript:
    1. Build a radius neighborhood with a KNN fallback for every point.
    2. Estimate an initial normal by local PCA.
    3. Refine the local planar support using normal-angle consistency and
       re-estimate the corrected normal.
    4. In the initial neighborhood, construct a normal-consistent subset for
       every point.
    5. Merge overlapping subsets with Union-Find.
    6. Process spatial partitions in parallel and merge the local components
       again according to their shared halo points.
    7. Assign a reproducible random color to every final discontinuity.

The script is compatible with Python 3.10 and Windows multiprocessing.
"""

from __future__ import annotations

# Prevent one process from starting many BLAS threads. This must be configured
# before NumPy/SciPy are imported, especially when ProcessPoolExecutor is used.
import os

for _thread_env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_name, "1")

import argparse
import csv
import colorsys
import json
import math
import multiprocessing as mp
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - simple fallback for minimal environments
    tqdm = None

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - the Python fallback remains functional
    NUMBA_AVAILABLE = False
    njit = None


PROGRAM_NAME = "NCNM Parallel Discontinuity Recognition"
PROGRAM_VERSION = "1.0.0"
EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class RecognitionConfig:
    radius: float
    max_nn: int
    min_nn: int
    fallback_knn: int
    refine_angle_threshold: float
    edge_angle_threshold: float
    min_refine_points: int
    min_subset_size: int
    query_batch_size: int
    pca_batch_size: int


@dataclass(frozen=True)
class PartitionTask:
    partition_id: int
    core_start: int
    core_end: int
    halo_start: int
    halo_end: int
    points_path: str
    sorted_ids_path: str
    neighbors_path: str
    counts_path: str
    initial_normals_path: str
    result_path: str
    config: RecognitionConfig
    corrected_normals_path: str = ""


class Progress:
    """Small tqdm-compatible wrapper used when tqdm is unavailable."""

    def __init__(self, iterable: Iterable[Any], total: int | None = None, desc: str = ""):
        self.iterable = iterable
        self.total = total
        self.desc = desc

    def __iter__(self):
        if self.desc:
            print(self.desc)
        yield from self.iterable


def progress(
    iterable: Iterable[Any], *, total: int | None = None, desc: str = ""
) -> Iterable[Any]:
    if tqdm is None:
        return Progress(iterable, total=total, desc=desc)
    return tqdm(iterable, total=total, desc=desc, unit="part")


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _union_edges_numba(
        parent: np.ndarray,
        component_size: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
    ) -> None:
        for edge_index in range(source.shape[0]):
            root_a = int(source[edge_index])
            while parent[root_a] != root_a:
                parent[root_a] = parent[parent[root_a]]
                root_a = parent[root_a]

            root_b = int(target[edge_index])
            while parent[root_b] != root_b:
                parent[root_b] = parent[parent[root_b]]
                root_b = parent[root_b]

            if root_a == root_b:
                continue
            if component_size[root_a] < component_size[root_b]:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            component_size[root_a] += component_size[root_b]

    @njit(cache=True)
    def _compress_parents_numba(parent: np.ndarray) -> None:
        for point_index in range(parent.shape[0]):
            root = point_index
            while parent[root] != root:
                root = parent[root]
            current = point_index
            while parent[current] != current:
                next_point = parent[current]
                parent[current] = root
                current = next_point


def _union_edges_python(
    parent: np.ndarray,
    component_size: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> None:
    for point_a, point_b in zip(source.tolist(), target.tolist()):
        root_a = int(point_a)
        while parent[root_a] != root_a:
            parent[root_a] = parent[parent[root_a]]
            root_a = int(parent[root_a])

        root_b = int(point_b)
        while parent[root_b] != root_b:
            parent[root_b] = parent[parent[root_b]]
            root_b = int(parent[root_b])

        if root_a == root_b:
            continue
        if component_size[root_a] < component_size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        component_size[root_a] += component_size[root_b]


def _compress_parents_python(parent: np.ndarray) -> None:
    for point_index in range(parent.size):
        root = point_index
        while parent[root] != root:
            root = int(parent[root])
        current = point_index
        while parent[current] != current:
            next_point = int(parent[current])
            parent[current] = root
            current = next_point


def union_edges(
    parent: np.ndarray,
    component_size: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> None:
    if source.size == 0:
        return
    source = np.ascontiguousarray(source, dtype=np.int64)
    target = np.ascontiguousarray(target, dtype=np.int64)
    if NUMBA_AVAILABLE:
        _union_edges_numba(parent, component_size, source, target)
    else:
        _union_edges_python(parent, component_size, source, target)


def compress_parents(parent: np.ndarray) -> None:
    if NUMBA_AVAILABLE:
        _compress_parents_numba(parent)
    else:
        _compress_parents_python(parent)


class UnionFind:
    """Union-Find used for the much smaller cross-partition component graph."""

    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.component_size = np.ones(size, dtype=np.int64)

    def find(self, item: int) -> int:
        root = int(item)
        while self.parent[root] != root:
            self.parent[root] = self.parent[self.parent[root]]
            root = int(self.parent[root])
        current = int(item)
        while self.parent[current] != current:
            next_item = int(self.parent[current])
            self.parent[current] = root
            current = next_item
        return root

    def union(self, item_a: int, item_b: int) -> bool:
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return False
        if self.component_size[root_a] < self.component_size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.component_size[root_a] += self.component_size[root_b]
        return True

    def compressed_roots(self) -> np.ndarray:
        roots = np.empty(self.parent.size, dtype=np.int64)
        for item in range(self.parent.size):
            roots[item] = self.find(item)
        return roots


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for this input format or interactive visualization. "
            "Install it in the edge_first environment with: pip install open3d"
        ) from exc
    return o3d


def validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("The point array must have shape (N, 3) or more columns.")
    points = np.asarray(points[:, :3], dtype=np.float64)
    finite_mask = np.all(np.isfinite(points), axis=1)
    removed = int(points.shape[0] - np.count_nonzero(finite_mask))
    if removed:
        print(f"Removed {removed:,} points containing NaN or infinity.")
        points = points[finite_mask]
    if points.shape[0] < 3:
        raise ValueError("At least three valid points are required.")
    return np.ascontiguousarray(points)


def validate_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("The point array must have shape (N, 3) or more columns.")
    points = np.asarray(points[:, :3], dtype=np.float64)
    finite_mask = np.all(np.isfinite(points), axis=1)
    removed = int(points.shape[0] - np.count_nonzero(finite_mask))
    if removed:
        print(f"Removed {removed:,} points containing NaN or infinity.")
        points = points[finite_mask]
        if colors is not None:
            colors = np.asarray(colors)[finite_mask]
    if points.shape[0] < 3:
        raise ValueError("At least three valid points are required.")
    if colors is None:
        return np.ascontiguousarray(points), None
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim != 2 or colors.shape[1] < 3 or colors.shape[0] != points.shape[0]:
        raise ValueError("The color array must have shape (N, 3).")
    colors = np.clip(colors[:, :3], 0.0, 1.0)
    return np.ascontiguousarray(points), np.ascontiguousarray(colors)


def _pcd_numpy_dtype(
    fields: list[str],
    sizes: list[int],
    types: list[str],
    counts: list[int],
) -> np.dtype:
    dtype_fields = []
    for field, size, type_name, count in zip(fields, sizes, types, counts):
        key = (type_name.upper(), size)
        if key == ("F", 4):
            dtype = "<f4"
        elif key == ("F", 8):
            dtype = "<f8"
        elif key == ("I", 1):
            dtype = "i1"
        elif key == ("I", 2):
            dtype = "<i2"
        elif key == ("I", 4):
            dtype = "<i4"
        elif key == ("I", 8):
            dtype = "<i8"
        elif key == ("U", 1):
            dtype = "u1"
        elif key == ("U", 2):
            dtype = "<u2"
        elif key == ("U", 4):
            dtype = "<u4"
        elif key == ("U", 8):
            dtype = "<u8"
        else:
            raise ValueError(
                f"Unsupported PCD field type/size combination: {type_name} {size}"
            )
        if count == 1:
            dtype_fields.append((field, dtype))
        else:
            dtype_fields.append((field, dtype, (count,)))
    return np.dtype(dtype_fields)


def pcd_rgb_to_colors(rgb_values: np.ndarray) -> np.ndarray:
    packed = np.asarray(rgb_values)
    if packed.dtype.kind == "f":
        packed = np.ascontiguousarray(packed.astype("<f4", copy=False)).view("<u4")
    else:
        packed = packed.astype("<u4", copy=False)
    red = ((packed >> 16) & 255).astype(np.float64)
    green = ((packed >> 8) & 255).astype(np.float64)
    blue = (packed & 255).astype(np.float64)
    return np.column_stack((red, green, blue)) / 255.0


def colors_from_pcd_records(
    records: np.ndarray,
    fields: list[str],
) -> np.ndarray | None:
    if "rgb" in fields:
        return pcd_rgb_to_colors(records["rgb"])
    if "rgba" in fields:
        return pcd_rgb_to_colors(records["rgba"])
    if {"r", "g", "b"}.issubset(fields):
        return np.column_stack(
            (
                np.asarray(records["r"], dtype=np.float64),
                np.asarray(records["g"], dtype=np.float64),
                np.asarray(records["b"], dtype=np.float64),
            )
        ) / 255.0
    return None


def read_pcd_cloud(input_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read x/y/z and optional colors from ASCII or uncompressed binary PCD."""
    header: dict[str, list[str]] = {}
    with input_path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header is missing DATA line: {input_path}")
            try:
                decoded = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"PCD header is not ASCII: {input_path}") from exc
            if not decoded or decoded.startswith("#"):
                continue
            parts = decoded.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break

        fields = header.get("FIELDS")
        if not fields or not {"x", "y", "z"}.issubset(fields):
            raise ValueError("PCD input must contain x, y, and z fields.")
        sizes = [int(value) for value in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        point_count = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
        data_kind = header["DATA"][0].lower()

        if len(sizes) != len(fields) or len(types) != len(fields):
            raise ValueError("PCD FIELDS, SIZE, and TYPE lengths do not match.")
        if len(counts) != len(fields):
            raise ValueError("PCD COUNT length does not match FIELDS.")
        if point_count <= 0:
            raise ValueError("PCD header does not specify a positive point count.")

        if data_kind == "binary":
            dtype = _pcd_numpy_dtype(fields, sizes, types, counts)
            records = np.fromfile(stream, dtype=dtype, count=point_count)
            if records.size != point_count:
                raise ValueError(
                    f"PCD binary data ended early: expected {point_count:,} points, "
                    f"read {records.size:,}."
                )
            points = np.column_stack(
                (
                    np.asarray(records["x"], dtype=np.float64),
                    np.asarray(records["y"], dtype=np.float64),
                    np.asarray(records["z"], dtype=np.float64),
                )
            )
            colors = colors_from_pcd_records(records, fields)
            return validate_cloud(points, colors)

        if data_kind == "ascii":
            flat_counts = np.cumsum([0, *counts])
            xyz_columns = [int(flat_counts[fields.index(axis)]) for axis in "xyz"]
            color_columns = []
            if {"r", "g", "b"}.issubset(fields):
                color_columns = [
                    int(flat_counts[fields.index(channel)]) for channel in ("r", "g", "b")
                ]
            usecols = xyz_columns + color_columns
            data = np.loadtxt(stream, usecols=usecols, max_rows=point_count)
            points = data[:, :3]
            colors = data[:, 3:6] / 255.0 if color_columns else None
            return validate_cloud(points, colors)

    raise ValueError(
        "PCD binary_compressed input is not supported by the NumPy fallback. "
        "Convert it to binary/ascii PCD or install a reader that supports it."
    )


def read_pcd_points(input_path: Path) -> np.ndarray:
    points, _ = read_pcd_cloud(input_path)
    return points


def read_cloud(input_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    suffix = input_path.suffix.lower()
    if suffix == ".pcd":
        try:
            o3d = import_open3d()
            cloud = o3d.io.read_point_cloud(str(input_path))
            if not cloud.is_empty():
                colors = (
                    np.asarray(cloud.colors)
                    if np.asarray(cloud.colors).shape[0] == len(cloud.points)
                    else None
                )
                return validate_cloud(np.asarray(cloud.points), colors)
            print("Open3D read an empty PCD; using NumPy PCD reader.")
        except Exception as exc:
            print(f"Open3D could not read PCD ({exc}); using NumPy PCD reader.")
        return read_pcd_cloud(input_path)

    points = read_points(input_path)
    return points, None


def read_points(input_path: Path) -> np.ndarray:
    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        return validate_points(np.load(input_path, mmap_mode="r"))

    if suffix == ".npz":
        with np.load(input_path) as data:
            if "points" in data:
                points = data["points"]
            elif data.files:
                points = data[data.files[0]]
            else:
                raise ValueError(f"No array was found in {input_path}.")
        return validate_points(points)

    if suffix in {".txt", ".xyz", ".pts", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        return validate_points(np.loadtxt(input_path, delimiter=delimiter))

    if suffix in {".las", ".laz"}:
        try:
            import laspy
        except ImportError as exc:
            raise RuntimeError(
                "LAS/LAZ input requires laspy. Install it with: pip install laspy[lazrs]"
            ) from exc
        las = laspy.read(input_path)
        return validate_points(np.column_stack((las.x, las.y, las.z)))

    if suffix == ".pcd":
        try:
            o3d = import_open3d()
            cloud = o3d.io.read_point_cloud(str(input_path))
            if not cloud.is_empty():
                return validate_points(np.asarray(cloud.points))
            print("Open3D read an empty PCD; using NumPy PCD reader.")
        except Exception as exc:
            print(f"Open3D could not read PCD ({exc}); using NumPy PCD reader.")
        return read_pcd_points(input_path)

    o3d = import_open3d()
    cloud = o3d.io.read_point_cloud(str(input_path))
    if cloud.is_empty():
        raise ValueError(f"Open3D could not read any points from {input_path}.")
    return validate_points(np.asarray(cloud.points))


def numpy_voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return the first point index selected in every occupied voxel."""
    origin = np.min(points, axis=0)
    voxel_index = np.floor((points - origin) / voxel_size).astype(np.int64)
    packed = np.ascontiguousarray(voxel_index).view(
        np.dtype((np.void, voxel_index.dtype.itemsize * voxel_index.shape[1]))
    )
    _, selected = np.unique(packed.ravel(), return_index=True)
    selected.sort()
    return selected


def numpy_voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Dependency-free first-point voxel sampling, used when Open3D is absent."""
    selected = numpy_voxel_downsample_indices(points, voxel_size)
    return np.ascontiguousarray(points[selected])


def voxel_downsample_cloud(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if voxel_size <= 0:
        return points, colors
    if colors is not None:
        selected = numpy_voxel_downsample_indices(points, voxel_size)
        return (
            np.ascontiguousarray(points[selected]),
            np.ascontiguousarray(colors[selected]),
        )
    return voxel_downsample(points, voxel_size), None


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return points
    try:
        o3d = import_open3d()
    except RuntimeError:
        print("Open3D is unavailable; using NumPy voxel downsampling.")
        return numpy_voxel_downsample(points, voxel_size)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(voxel_size)
    return validate_points(np.asarray(cloud.points))


def estimate_median_spacing(
    points: np.ndarray,
    sorted_ids: np.ndarray,
    sample_size: int = 120_000,
) -> float:
    """Estimate true local spacing from three spatially contiguous point slabs."""
    point_count = points.shape[0]
    slab_size = min(sample_size, point_count)
    if point_count <= slab_size:
        starts = [0]
    else:
        starts = [
            max(0, min(point_count - slab_size, int(f * point_count - slab_size / 2)))
            for f in (0.2, 0.5, 0.8)
        ]

    spacing_samples: list[np.ndarray] = []
    for start in sorted(set(starts)):
        slab_ids = np.asarray(sorted_ids[start : start + slab_size], dtype=np.int64)
        slab_points = np.asarray(points[slab_ids], dtype=np.float64)
        tree = cKDTree(slab_points)
        query_count = min(50_000, slab_points.shape[0])
        if query_count < slab_points.shape[0]:
            query_ids = np.linspace(
                0, slab_points.shape[0] - 1, query_count, dtype=np.int64
            )
            query_points = slab_points[query_ids]
        else:
            query_points = slab_points
        distances, _ = tree.query(query_points, k=2, workers=1)
        nearest = np.asarray(distances)[:, 1]
        valid = nearest[np.isfinite(nearest) & (nearest > EPS)]
        if valid.size:
            spacing_samples.append(valid)

    if not spacing_samples:
        raise RuntimeError(
            "Point spacing could not be estimated. Set --neighbor-radius explicitly."
        )
    spacing = float(np.median(np.concatenate(spacing_samples)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise RuntimeError(
            "Point spacing could not be estimated. Set --neighbor-radius explicitly."
        )
    return spacing


def build_hybrid_neighborhoods(
    points: np.ndarray,
    radius: float,
    max_nn: int,
    min_nn: int,
    fallback_knn: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build fixed-width radius neighborhoods. Rows with too few radius neighbors
    are replaced by a KNN neighborhood.
    """
    point_count = points.shape[0]
    radius_k = min(max_nn, point_count)
    fallback_k = min(max(fallback_knn, min_nn), max_nn, point_count)
    neighbors = np.full((point_count, max_nn), -1, dtype=np.int32)
    counts = np.zeros(point_count, dtype=np.int32)
    tree = cKDTree(points)

    for start in range(0, point_count, batch_size):
        end = min(point_count, start + batch_size)
        query_points = points[start:end]
        distances, indices = tree.query(
            query_points,
            k=radius_k,
            distance_upper_bound=radius,
            workers=1,
        )
        if radius_k == 1:
            distances = np.asarray(distances)[:, None]
            indices = np.asarray(indices)[:, None]
        valid = np.isfinite(distances) & (indices < point_count)
        local_counts = np.sum(valid, axis=1).astype(np.int32)
        safe_indices = np.where(valid, indices, -1).astype(np.int32, copy=False)
        neighbors[start:end, :radius_k] = safe_indices

        deficient = np.flatnonzero(local_counts < min_nn)
        if deficient.size:
            _, fallback_indices = tree.query(
                query_points[deficient], k=fallback_k, workers=1
            )
            if fallback_k == 1:
                fallback_indices = np.asarray(fallback_indices)[:, None]
            rows = start + deficient
            neighbors[rows] = -1
            neighbors[rows, :fallback_k] = np.asarray(
                fallback_indices, dtype=np.int32
            )
            local_counts[deficient] = fallback_k
        counts[start:end] = local_counts

    return neighbors, counts


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    safe = lengths > EPS
    vectors[safe] /= lengths[safe, None]
    vectors[~safe] = np.array([0.0, 0.0, 1.0])
    return vectors


def initial_pca_normals(
    points: np.ndarray,
    neighbors: np.ndarray,
    counts: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    point_count, max_nn = neighbors.shape
    normals = np.empty((point_count, 3), dtype=np.float64)
    column_ids = np.arange(max_nn)[None, :]

    for start in range(0, point_count, batch_size):
        end = min(point_count, start + batch_size)
        local_neighbors = neighbors[start:end]
        valid = column_ids < counts[start:end, None]
        safe_indices = np.where(valid, local_neighbors, 0)
        local_points = points[safe_indices]
        weights = valid[..., None]
        divisor = np.maximum(counts[start:end], 1)[:, None]
        mean = np.sum(local_points * weights, axis=1) / divisor
        centered = (local_points - mean[:, None, :]) * weights
        covariance = np.einsum(
            "bki,bkj->bij", centered, centered, optimize=True
        ) / divisor[:, :, None]
        _, eigenvectors = np.linalg.eigh(covariance)
        batch_normals = eigenvectors[:, :, 0]
        insufficient = counts[start:end] < 3
        batch_normals[insufficient] = np.array([0.0, 0.0, 1.0])
        normals[start:end] = batch_normals

    return _normalize_rows(normals)


def refine_normals(
    points: np.ndarray,
    neighbors: np.ndarray,
    counts: np.ndarray,
    initial_normals: np.ndarray,
    angle_threshold: float,
    min_refine_points: int,
    batch_size: int,
) -> np.ndarray:
    """
    Refine the planar support using the angular distribution described in the
    manuscript. If the threshold-selected support has too few points, points
    below the neighborhood's mean angular deviation are used as supplements.
    """
    point_count, max_nn = neighbors.shape
    corrected = np.empty_like(initial_normals)
    column_ids = np.arange(max_nn)[None, :]
    cosine_threshold = math.cos(math.radians(angle_threshold))

    for start in range(0, point_count, batch_size):
        end = min(point_count, start + batch_size)
        local_neighbors = neighbors[start:end]
        valid = column_ids < counts[start:end, None]
        safe_indices = np.where(valid, local_neighbors, 0)

        neighbor_normals = initial_normals[safe_indices]
        center_normals = initial_normals[start:end]
        dots = np.abs(
            np.einsum("bi,bki->bk", center_normals, neighbor_normals, optimize=True)
        )
        dots = np.clip(dots, 0.0, 1.0)
        threshold_support = valid & (dots >= cosine_threshold)
        threshold_count = np.sum(threshold_support, axis=1)

        angles = np.arccos(dots)
        mean_angle = np.sum(angles * valid, axis=1) / np.maximum(
            counts[start:end], 1
        )
        mean_support = valid & (angles <= mean_angle[:, None])
        use_threshold = threshold_count >= min_refine_points
        selected = np.where(
            use_threshold[:, None], threshold_support, mean_support
        )
        selected_count = np.sum(selected, axis=1).astype(np.int32)

        # A rare degenerate neighborhood may still contain fewer than three
        # usable points. Add the most consistent available points.
        deficient_rows = np.flatnonzero(
            (selected_count < min_refine_points)
            & (counts[start:end] >= min_refine_points)
        )
        for row in deficient_rows.tolist():
            valid_columns = np.flatnonzero(valid[row])
            order = valid_columns[np.argsort(-dots[row, valid_columns])]
            selected[row, order[:min_refine_points]] = True
            selected_count[row] = int(np.count_nonzero(selected[row]))

        local_points = points[safe_indices]
        weights = selected[..., None]
        divisor = np.maximum(selected_count, 1)[:, None]
        mean = np.sum(local_points * weights, axis=1) / divisor
        centered = (local_points - mean[:, None, :]) * weights
        covariance = np.einsum(
            "bki,bkj->bij", centered, centered, optimize=True
        ) / divisor[:, :, None]
        _, eigenvectors = np.linalg.eigh(covariance)
        batch_normals = eigenvectors[:, :, 0]

        insufficient = selected_count < 3
        batch_normals[insufficient] = center_normals[insufficient]
        # Align the corrected normal with the initial normal. The subsequent
        # tests are orientation-free, but this makes saved normals stable.
        flip = np.einsum(
            "bi,bi->b", batch_normals, center_normals, optimize=True
        ) < 0
        batch_normals[flip] *= -1.0
        corrected[start:end] = batch_normals

    return _normalize_rows(corrected)


def merge_normal_consistent_subsets(
    neighbors: np.ndarray,
    counts: np.ndarray,
    corrected_normals: np.ndarray,
    angle_threshold: float,
    min_subset_size: int,
    batch_size: int,
) -> np.ndarray:
    """
    Every normal-consistent subset is treated as a Union-Find hyperedge. Unioning
    the center with all members is equivalent to merging all subsets that share
    at least one point.
    """
    point_count, max_nn = neighbors.shape
    parent = np.arange(point_count, dtype=np.int64)
    component_size = np.ones(point_count, dtype=np.int64)
    column_ids = np.arange(max_nn)[None, :]
    cosine_threshold = math.cos(math.radians(angle_threshold))

    for start in range(0, point_count, batch_size):
        end = min(point_count, start + batch_size)
        local_neighbors = neighbors[start:end]
        valid = column_ids < counts[start:end, None]
        safe_indices = np.where(valid, local_neighbors, 0)
        neighbor_normals = corrected_normals[safe_indices]
        center_normals = corrected_normals[start:end]
        dots = np.abs(
            np.einsum("bi,bki->bk", center_normals, neighbor_normals, optimize=True)
        )
        selected = valid & (dots >= cosine_threshold)
        selected_count = np.sum(selected, axis=1)
        selected &= selected_count[:, None] >= min_subset_size

        row_ids, column_positions = np.nonzero(selected)
        if row_ids.size:
            source = (start + row_ids).astype(np.int64, copy=False)
            target = local_neighbors[row_ids, column_positions].astype(
                np.int64, copy=False
            )
            not_self = source != target
            union_edges(
                parent,
                component_size,
                source[not_self],
                target[not_self],
            )

    compress_parents(parent)
    _, labels = np.unique(parent, return_inverse=True)
    return labels.astype(np.int32, copy=False)


def refine_normals_for_ids(
    points: np.ndarray,
    neighbors: np.ndarray,
    counts: np.ndarray,
    initial_normals: np.ndarray,
    point_ids: np.ndarray,
    angle_threshold: float,
    min_refine_points: int,
    batch_size: int,
) -> np.ndarray:
    """Refine normals for selected global point IDs using global neighborhoods."""
    point_ids = np.asarray(point_ids, dtype=np.int64)
    max_nn = neighbors.shape[1]
    corrected = np.empty((point_ids.size, 3), dtype=np.float64)
    column_ids = np.arange(max_nn)[None, :]
    cosine_threshold = math.cos(math.radians(angle_threshold))

    for start in range(0, point_ids.size, batch_size):
        end = min(point_ids.size, start + batch_size)
        batch_ids = point_ids[start:end]
        local_neighbors = np.asarray(neighbors[batch_ids], dtype=np.int64)
        local_counts = np.asarray(counts[batch_ids], dtype=np.int32)
        valid = column_ids < local_counts[:, None]
        safe_indices = np.where(valid, local_neighbors, 0)

        neighbor_normals = initial_normals[safe_indices]
        center_normals = initial_normals[batch_ids]
        dots = np.abs(
            np.einsum("bi,bki->bk", center_normals, neighbor_normals, optimize=True)
        )
        dots = np.clip(dots, 0.0, 1.0)
        threshold_support = valid & (dots >= cosine_threshold)
        threshold_count = np.sum(threshold_support, axis=1)

        angles = np.arccos(dots)
        mean_angle = np.sum(angles * valid, axis=1) / np.maximum(local_counts, 1)
        mean_support = valid & (angles <= mean_angle[:, None])
        use_threshold = threshold_count >= min_refine_points
        selected = np.where(use_threshold[:, None], threshold_support, mean_support)
        selected_count = np.sum(selected, axis=1).astype(np.int32)

        deficient_rows = np.flatnonzero(
            (selected_count < min_refine_points) & (local_counts >= min_refine_points)
        )
        for row in deficient_rows.tolist():
            valid_columns = np.flatnonzero(valid[row])
            order = valid_columns[np.argsort(-dots[row, valid_columns])]
            selected[row, order[:min_refine_points]] = True
            selected_count[row] = int(np.count_nonzero(selected[row]))

        local_points = points[safe_indices]
        weights = selected[..., None]
        divisor = np.maximum(selected_count, 1)[:, None]
        mean = np.sum(local_points * weights, axis=1) / divisor
        centered = (local_points - mean[:, None, :]) * weights
        covariance = np.einsum(
            "bki,bkj->bij", centered, centered, optimize=True
        ) / divisor[:, :, None]
        _, eigenvectors = np.linalg.eigh(covariance)
        batch_normals = eigenvectors[:, :, 0]

        insufficient = selected_count < 3
        batch_normals[insufficient] = center_normals[insufficient]
        flip = np.einsum("bi,bi->b", batch_normals, center_normals, optimize=True) < 0
        batch_normals[flip] *= -1.0
        corrected[start:end] = batch_normals

    return _normalize_rows(corrected)


def normal_consistent_edges_for_ids(
    neighbors: np.ndarray,
    counts: np.ndarray,
    corrected_normals: np.ndarray,
    point_ids: np.ndarray,
    angle_threshold: float,
    min_subset_size: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build global union edges for selected center points only."""
    point_ids = np.asarray(point_ids, dtype=np.int64)
    max_nn = neighbors.shape[1]
    column_ids = np.arange(max_nn)[None, :]
    cosine_threshold = math.cos(math.radians(angle_threshold))
    source_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []

    for start in range(0, point_ids.size, batch_size):
        end = min(point_ids.size, start + batch_size)
        batch_ids = point_ids[start:end]
        local_neighbors = np.asarray(neighbors[batch_ids], dtype=np.int64)
        local_counts = np.asarray(counts[batch_ids], dtype=np.int32)
        valid = column_ids < local_counts[:, None]
        safe_indices = np.where(valid, local_neighbors, 0)

        neighbor_normals = corrected_normals[safe_indices]
        center_normals = corrected_normals[batch_ids]
        dots = np.abs(
            np.einsum("bi,bki->bk", center_normals, neighbor_normals, optimize=True)
        )
        selected = valid & (dots >= cosine_threshold)
        selected_count = np.sum(selected, axis=1)
        selected &= selected_count[:, None] >= min_subset_size

        row_ids, column_positions = np.nonzero(selected)
        if row_ids.size:
            source = batch_ids[row_ids].astype(np.int64, copy=False)
            target = local_neighbors[row_ids, column_positions].astype(
                np.int64, copy=False
            )
            not_self = source != target
            source_blocks.append(source[not_self])
            target_blocks.append(target[not_self])

    if not source_blocks:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(source_blocks), np.concatenate(target_blocks)


def normals_from_structure_tensor(
    labels: np.ndarray,
    normals: np.ndarray,
    component_count: int | None = None,
) -> np.ndarray:
    if component_count is None:
        component_count = int(labels.max()) + 1 if labels.size else 0
    if component_count == 0:
        return np.empty((0, 3), dtype=np.float64)

    nx, ny, nz = normals.T
    matrices = np.zeros((component_count, 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = np.bincount(
        labels, weights=nx * nx, minlength=component_count
    )
    matrices[:, 0, 1] = matrices[:, 1, 0] = np.bincount(
        labels, weights=nx * ny, minlength=component_count
    )
    matrices[:, 0, 2] = matrices[:, 2, 0] = np.bincount(
        labels, weights=nx * nz, minlength=component_count
    )
    matrices[:, 1, 1] = np.bincount(
        labels, weights=ny * ny, minlength=component_count
    )
    matrices[:, 1, 2] = matrices[:, 2, 1] = np.bincount(
        labels, weights=ny * nz, minlength=component_count
    )
    matrices[:, 2, 2] = np.bincount(
        labels, weights=nz * nz, minlength=component_count
    )
    _, eigenvectors = np.linalg.eigh(matrices)
    result = eigenvectors[:, :, -1]
    return _normalize_rows(result)


def process_partition(task: PartitionTask) -> dict[str, Any]:
    """Refine normals for one spatial core using global neighborhoods."""
    started = time.perf_counter()
    points_memmap = np.load(task.points_path, mmap_mode="r")
    sorted_ids_memmap = np.load(task.sorted_ids_path, mmap_mode="r")
    neighbors_memmap = np.load(task.neighbors_path, mmap_mode="r")
    counts_memmap = np.load(task.counts_path, mmap_mode="r")
    initial_normals_memmap = np.load(task.initial_normals_path, mmap_mode="r")
    core_global_ids = np.asarray(
        sorted_ids_memmap[task.core_start : task.core_end], dtype=np.int64
    )
    config = task.config

    core_normals = refine_normals_for_ids(
        points_memmap,
        neighbors_memmap,
        counts_memmap,
        initial_normals_memmap,
        core_global_ids,
        angle_threshold=config.refine_angle_threshold,
        min_refine_points=config.min_refine_points,
        batch_size=config.pca_batch_size,
    )

    np.savez(
        task.result_path,
        partition_id=np.array(task.partition_id, dtype=np.int32),
        core_global_ids=core_global_ids,
        core_normals=core_normals.astype(np.float32),
    )
    elapsed = time.perf_counter() - started
    return {
        "partition_id": task.partition_id,
        "result_path": task.result_path,
        "core_points": int(core_global_ids.size),
        "halo_points": int(task.halo_end - task.halo_start),
        "seconds": elapsed,
    }


def process_edge_partition(task: PartitionTask) -> dict[str, Any]:
    """Build global normal-consistent edges for one spatial core."""
    started = time.perf_counter()
    if not task.corrected_normals_path:
        raise ValueError("corrected_normals_path is required for edge extraction.")
    sorted_ids_memmap = np.load(task.sorted_ids_path, mmap_mode="r")
    neighbors_memmap = np.load(task.neighbors_path, mmap_mode="r")
    counts_memmap = np.load(task.counts_path, mmap_mode="r")
    corrected_normals_memmap = np.load(task.corrected_normals_path, mmap_mode="r")
    core_global_ids = np.asarray(
        sorted_ids_memmap[task.core_start : task.core_end], dtype=np.int64
    )
    config = task.config

    edge_source, edge_target = normal_consistent_edges_for_ids(
        neighbors_memmap,
        counts_memmap,
        corrected_normals_memmap,
        core_global_ids,
        angle_threshold=config.edge_angle_threshold,
        min_subset_size=config.min_subset_size,
        batch_size=config.pca_batch_size,
    )

    edge_result_path = str(Path(task.result_path).with_name(f"edges_{task.partition_id:04d}.npz"))
    np.savez(
        edge_result_path,
        partition_id=np.array(task.partition_id, dtype=np.int32),
        edge_source=edge_source,
        edge_target=edge_target,
    )
    elapsed = time.perf_counter() - started
    return {
        "partition_id": task.partition_id,
        "result_path": edge_result_path,
        "core_points": int(core_global_ids.size),
        "local_edges": int(edge_source.size),
        "seconds": elapsed,
    }


def build_partition_tasks(
    points: np.ndarray,
    sorted_ids: np.ndarray,
    axis: int,
    partition_size: int,
    halo_distance: float,
    points_path: Path,
    sorted_ids_path: Path,
    neighbors_path: Path,
    counts_path: Path,
    initial_normals_path: Path,
    partition_result_dir: Path,
    config: RecognitionConfig,
    corrected_normals_path: Path | None = None,
) -> list[PartitionTask]:
    point_count = points.shape[0]
    sorted_axis_values = np.asarray(points[sorted_ids, axis], dtype=np.float64)
    tasks: list[PartitionTask] = []
    for partition_id, core_start in enumerate(
        range(0, point_count, partition_size)
    ):
        core_end = min(point_count, core_start + partition_size)
        lower = sorted_axis_values[core_start] - halo_distance
        upper = sorted_axis_values[core_end - 1] + halo_distance
        halo_start = int(np.searchsorted(sorted_axis_values, lower, side="left"))
        halo_end = int(np.searchsorted(sorted_axis_values, upper, side="right"))
        result_path = partition_result_dir / f"partition_{partition_id:04d}.npz"
        tasks.append(
            PartitionTask(
                partition_id=partition_id,
                core_start=core_start,
                core_end=core_end,
                halo_start=halo_start,
                halo_end=halo_end,
                points_path=str(points_path),
                sorted_ids_path=str(sorted_ids_path),
                neighbors_path=str(neighbors_path),
                counts_path=str(counts_path),
                initial_normals_path=str(initial_normals_path),
                result_path=str(result_path),
                config=config,
                corrected_normals_path=(
                    "" if corrected_normals_path is None else str(corrected_normals_path)
                ),
            )
        )
    return tasks


def run_partitions(
    tasks: list[PartitionTask],
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        results = [
            process_partition(task)
            for task in progress(tasks, total=len(tasks), desc="Local NCNM")
        ]
        return sorted(results, key=lambda item: item["partition_id"])

    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as executor:
        future_to_task = {
            executor.submit(process_partition, task): task for task in tasks
        }
        completed = as_completed(future_to_task)
        for future in progress(
            completed, total=len(future_to_task), desc="Parallel local NCNM"
        ):
            task = future_to_task[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"Partition {task.partition_id} failed: {exc}"
                ) from exc
    return sorted(results, key=lambda item: item["partition_id"])


def run_edge_partitions(
    tasks: list[PartitionTask],
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        results = [
            process_edge_partition(task)
            for task in progress(tasks, total=len(tasks), desc="Global edge extraction")
        ]
        return sorted(results, key=lambda item: item["partition_id"])

    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as executor:
        future_to_task = {
            executor.submit(process_edge_partition, task): task for task in tasks
        }
        completed = as_completed(future_to_task)
        for future in progress(
            completed, total=len(future_to_task), desc="Parallel global edge extraction"
        ):
            task = future_to_task[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"Partition {task.partition_id} edge extraction failed: {exc}"
                ) from exc
    return sorted(results, key=lambda item: item["partition_id"])


def collect_corrected_normals(
    normal_results: list[dict[str, Any]],
    point_count: int,
) -> np.ndarray:
    corrected_normals = np.zeros((point_count, 3), dtype=np.float32)
    assigned = np.zeros(point_count, dtype=bool)
    for result in normal_results:
        with np.load(result["result_path"]) as data:
            core_ids = np.asarray(data["core_global_ids"], dtype=np.int64)
            core_normals = np.asarray(data["core_normals"], dtype=np.float32)
        if np.any(assigned[core_ids]):
            raise RuntimeError("Partition cores overlap; a point was assigned twice.")
        corrected_normals[core_ids] = core_normals
        assigned[core_ids] = True
    if not np.all(assigned):
        missing = int(np.count_nonzero(~assigned))
        raise RuntimeError(f"{missing:,} points were not assigned to a partition core.")
    return corrected_normals


def merge_global_edges(
    edge_results: list[dict[str, Any]],
    point_count: int,
) -> tuple[np.ndarray, dict[str, int]]:
    parent = np.arange(point_count, dtype=np.int64)
    component_size = np.ones(point_count, dtype=np.int64)
    edge_count = 0
    for result in edge_results:
        with np.load(result["result_path"]) as data:
            source = np.asarray(data["edge_source"], dtype=np.int64)
            target = np.asarray(data["edge_target"], dtype=np.int64)
        edge_count += int(source.size)
        union_edges(parent, component_size, source, target)
    compress_parents(parent)
    diagnostics = {
        "global_edge_count": edge_count,
        "component_count_before_filter": int(np.unique(parent).size),
    }
    return parent, diagnostics


def collect_component_metadata(
    local_results: list[dict[str, Any]],
) -> tuple[np.ndarray, list[int], list[int]]:
    offsets: list[int] = []
    component_counts: list[int] = []
    normal_blocks: list[np.ndarray] = []
    offset = 0
    for result in local_results:
        with np.load(result["result_path"]) as data:
            normals = np.asarray(data["component_normals"], dtype=np.float64)
        offsets.append(offset)
        component_counts.append(normals.shape[0])
        normal_blocks.append(normals)
        offset += normals.shape[0]
    component_normals = (
        np.vstack(normal_blocks) if normal_blocks else np.empty((0, 3))
    )
    return component_normals, offsets, component_counts


def add_pair_counts(
    pair_counts: dict[int, int],
    existing_components: np.ndarray,
    new_components: np.ndarray,
    total_components: int,
) -> None:
    if existing_components.size == 0:
        return
    first = np.minimum(existing_components, new_components).astype(
        np.int64, copy=False
    )
    second = np.maximum(existing_components, new_components).astype(
        np.int64, copy=False
    )
    different = first != second
    if not np.any(different):
        return
    packed = first[different] * total_components + second[different]
    keys, counts = np.unique(packed, return_counts=True)
    for key, count in zip(keys.tolist(), counts.tolist()):
        pair_counts[int(key)] = pair_counts.get(int(key), 0) + int(count)


def merge_partition_components(
    local_results: list[dict[str, Any]],
    point_count: int,
    cross_min_shared_points: int,
    cross_angle_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Perform the requested second connectivity analysis. Local components from
    different partitions are connected when they share enough original halo
    points and their representative normals are consistent.
    """
    component_normals, offsets, component_counts = collect_component_metadata(
        local_results
    )
    total_components = component_normals.shape[0]
    if total_components == 0:
        raise RuntimeError("No local components were produced.")

    first_owner = np.full(point_count, -1, dtype=np.int64)
    pair_counts: dict[int, int] = {}

    for result, offset in zip(local_results, offsets):
        with np.load(result["result_path"]) as data:
            global_ids = np.asarray(data["halo_global_ids"], dtype=np.int64)
            local_labels = np.asarray(data["halo_labels"], dtype=np.int64)
        component_ids = local_labels + offset
        existing = first_owner[global_ids]
        already_seen = existing >= 0
        add_pair_counts(
            pair_counts,
            existing[already_seen],
            component_ids[already_seen],
            total_components,
        )
        new_points = ~already_seen
        first_owner[global_ids[new_points]] = component_ids[new_points]

    union_find = UnionFind(total_components)
    cosine_threshold = math.cos(math.radians(cross_angle_threshold))
    accepted_links = 0
    rejected_by_shared = 0
    rejected_by_angle = 0
    for packed_key, shared_count in pair_counts.items():
        component_a = packed_key // total_components
        component_b = packed_key % total_components
        if shared_count < cross_min_shared_points:
            rejected_by_shared += 1
            continue
        normal_dot = abs(
            float(
                np.dot(
                    component_normals[component_a],
                    component_normals[component_b],
                )
            )
        )
        if normal_dot < cosine_threshold:
            rejected_by_angle += 1
            continue
        if union_find.union(component_a, component_b):
            accepted_links += 1

    component_roots = union_find.compressed_roots()
    point_roots = np.full(point_count, -1, dtype=np.int64)
    corrected_normals = np.zeros((point_count, 3), dtype=np.float32)

    for result, offset in zip(local_results, offsets):
        with np.load(result["result_path"]) as data:
            core_ids = np.asarray(data["core_global_ids"], dtype=np.int64)
            core_labels = np.asarray(data["core_labels"], dtype=np.int64)
            core_normals = np.asarray(data["core_normals"], dtype=np.float32)
        if np.any(point_roots[core_ids] >= 0):
            raise RuntimeError("Partition cores overlap; a point was assigned twice.")
        point_roots[core_ids] = component_roots[core_labels + offset]
        corrected_normals[core_ids] = core_normals

    if np.any(point_roots < 0):
        missing = int(np.count_nonzero(point_roots < 0))
        raise RuntimeError(f"{missing:,} points were not assigned to a partition core.")

    diagnostics = {
        "local_component_count": int(total_components),
        "candidate_cross_links": int(len(pair_counts)),
        "accepted_cross_links": int(accepted_links),
        "rejected_cross_links_shared_points": int(rejected_by_shared),
        "rejected_cross_links_normal_angle": int(rejected_by_angle),
    }
    return point_roots, corrected_normals, diagnostics


def filter_and_relabel(
    point_roots: np.ndarray,
    min_component_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_roots, inverse, counts = np.unique(
        point_roots, return_inverse=True, return_counts=True
    )
    eligible = counts >= min_component_size
    eligible_indices = np.flatnonzero(eligible)
    # Large discontinuities receive smaller, stable group IDs.
    eligible_indices = eligible_indices[
        np.argsort(-counts[eligible_indices], kind="stable")
    ]
    root_position_to_group = np.full(unique_roots.size, -1, dtype=np.int32)
    root_position_to_group[eligible_indices] = np.arange(
        eligible_indices.size, dtype=np.int32
    )
    labels = root_position_to_group[inverse]
    final_sizes = counts[eligible_indices].astype(np.int64)
    return labels, final_sizes


def generate_random_colors(group_count: int, seed: int) -> np.ndarray:
    """Generate bright, deterministic random colors with separated hues."""
    if group_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    base_hues = (np.arange(group_count) * 0.618033988749895) % 1.0
    base_hues = (base_hues + rng.random()) % 1.0
    rng.shuffle(base_hues)
    saturations = rng.uniform(0.55, 0.88, size=group_count)
    values = rng.uniform(0.72, 0.96, size=group_count)
    colors = np.array(
        [
            colorsys.hsv_to_rgb(float(h), float(s), float(v))
            for h, s, v in zip(base_hues, saturations, values)
        ],
        dtype=np.float64,
    )
    return colors


def colorize_labels(
    labels: np.ndarray,
    palette: np.ndarray,
    noise_color: tuple[float, float, float],
    base_colors: np.ndarray | None = None,
) -> np.ndarray:
    if base_colors is None:
        colors = np.empty((labels.size, 3), dtype=np.float64)
        colors[:] = np.asarray(noise_color, dtype=np.float64)
    else:
        colors = np.asarray(base_colors, dtype=np.float64).copy()
    recognized = labels >= 0
    colors[recognized] = palette[labels[recognized]]
    return colors


def write_binary_ply(
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 1_000_000,
) -> None:
    point_count = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int group_id\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("group_id", "<i4"),
        ]
    )
    with output_path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for start in range(0, point_count, chunk_size):
            end = min(point_count, start + chunk_size)
            block = np.empty(end - start, dtype=dtype)
            block["x"] = points[start:end, 0]
            block["y"] = points[start:end, 1]
            block["z"] = points[start:end, 2]
            rgb = np.clip(np.rint(colors[start:end] * 255), 0, 255).astype(
                np.uint8
            )
            block["red"] = rgb[:, 0]
            block["green"] = rgb[:, 1]
            block["blue"] = rgb[:, 2]
            block["group_id"] = labels[start:end]
            block.tofile(stream)


def write_binary_pcd(
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 1_000_000,
) -> None:
    del labels
    point_count = points.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {point_count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {point_count}\n"
        "DATA binary\n"
    )
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("rgb", "<f4"),
        ]
    )
    with output_path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for start in range(0, point_count, chunk_size):
            end = min(point_count, start + chunk_size)
            block = np.empty(end - start, dtype=dtype)
            block["x"] = points[start:end, 0]
            block["y"] = points[start:end, 1]
            block["z"] = points[start:end, 2]
            rgb8 = np.clip(np.rint(colors[start:end] * 255), 0, 255).astype(
                np.uint32
            )
            packed_rgb = (
                (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
            ).astype("<u4")
            block["rgb"] = packed_rgb.view("<f4")
            block.tofile(stream)


def calculate_group_summary(
    points: np.ndarray,
    labels: np.ndarray,
    corrected_normals: np.ndarray,
) -> list[dict[str, float | int]]:
    recognized = labels >= 0
    if not np.any(recognized):
        return []
    group_labels = labels[recognized]
    group_count = int(group_labels.max()) + 1
    group_points = points[recognized]
    group_normals = corrected_normals[recognized].astype(np.float64)
    counts = np.bincount(group_labels, minlength=group_count).astype(np.int64)

    centroids = np.column_stack(
        [
            np.bincount(
                group_labels, weights=group_points[:, axis], minlength=group_count
            )
            / counts
            for axis in range(3)
        ]
    )
    normals = normals_from_structure_tensor(
        group_labels, group_normals, group_count
    )
    normals[normals[:, 2] < 0] *= -1.0

    centered = group_points - centroids[group_labels]
    residual = np.einsum(
        "ij,ij->i", centered, normals[group_labels], optimize=True
    )
    rms = np.sqrt(
        np.bincount(
            group_labels, weights=residual * residual, minlength=group_count
        )
        / counts
    )

    horizontal = np.linalg.norm(normals[:, :2], axis=1)
    dip = np.degrees(np.arctan2(horizontal, np.abs(normals[:, 2])))
    dip_direction = (
        np.degrees(np.arctan2(normals[:, 0], normals[:, 1])) + 360.0
    ) % 360.0

    rows: list[dict[str, float | int]] = []
    for group_id in range(group_count):
        rows.append(
            {
                "group_id": group_id,
                "point_count": int(counts[group_id]),
                "centroid_x": float(centroids[group_id, 0]),
                "centroid_y": float(centroids[group_id, 1]),
                "centroid_z": float(centroids[group_id, 2]),
                "normal_x": float(normals[group_id, 0]),
                "normal_y": float(normals[group_id, 1]),
                "normal_z": float(normals[group_id, 2]),
                "dip_direction_deg": float(dip_direction[group_id]),
                "dip_deg": float(dip[group_id]),
                "plane_rms": float(rms[group_id]),
            }
        )
    return rows


def write_group_summary(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "group_id",
        "point_count",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "normal_x",
        "normal_y",
        "normal_z",
        "dip_direction_deg",
        "dip_deg",
        "plane_rms",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_structure_point_indices(output_path: Path, labels: np.ndarray) -> None:
    group_count = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for group_id in range(group_count):
            point_ids = np.flatnonzero(labels == group_id)
            stream.write(" ".join(str(int(point_id)) for point_id in point_ids))
            stream.write("\n")


def format_coordinate(value: float) -> str:
    return format(float(value), ".17g")


def write_attitude_coordinate_format(
    output_path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 100_000,
    max_file_size_bytes: int = 5_000_000,
    max_points_per_group: int = 120,
    sample_seed: int = 2026,
) -> list[Path]:
    def part_path(part_index: int) -> Path:
        return output_path.with_name(
            f"{output_path.stem}_part{part_index:03d}{output_path.suffix}"
        )

    def encoded_size(text: str) -> int:
        return len(text.encode("utf-8"))

    def open_part(part_index: int) -> tuple[Any, Path, int]:
        path = part_path(part_index)
        return path.open("w", encoding="utf-8", newline="\n"), path, 0

    recognized_ids = np.flatnonzero(labels >= 0)
    if recognized_ids.size == 0:
        empty_path = part_path(1)
        empty_path.write_text("", encoding="utf-8", newline="\n")
        return [empty_path]

    order = np.argsort(labels[recognized_ids], kind="stable")
    sorted_ids = recognized_ids[order]
    sorted_labels = labels[sorted_ids]
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.diff(sorted_labels)) + 1,
            np.array([sorted_ids.size], dtype=np.int64),
        )
    )

    written_paths: list[Path] = []
    part_index = 1
    stream, current_path, current_size = open_part(part_index)
    rng = np.random.default_rng(sample_seed)

    try:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            group_id = int(sorted_labels[start]) + 1
            group_point_ids = sorted_ids[start:end]
            if group_point_ids.size > max_points_per_group:
                group_point_ids = rng.choice(
                    group_point_ids, size=max_points_per_group, replace=False
                )
            line_buffer: list[str] = []
            line_buffer_size = 0

            for chunk_start in range(0, group_point_ids.size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, group_point_ids.size)
                chunk_ids = group_point_ids[chunk_start:chunk_end]
                chunk_points = np.asarray(points[chunk_ids])
                for local_id, (x, y, z) in enumerate(
                    chunk_points, start=chunk_start + 1
                ):
                    line = (
                        f"{local_id},{format_coordinate(x)},"
                        f"{format_coordinate(y)},{format_coordinate(z)}\n"
                    )
                    line_buffer.append(line)
                    line_buffer_size += encoded_size(line)

                    header = f"{group_id},{len(line_buffer)},0,0\n"
                    block_size = encoded_size(header) + line_buffer_size
                    if current_size > 0 and current_size + block_size > max_file_size_bytes:
                        stream.close()
                        written_paths.append(current_path)
                        part_index += 1
                        stream, current_path, current_size = open_part(part_index)

                    if current_size + block_size > max_file_size_bytes:
                        line_buffer.pop()
                        line_buffer_size -= encoded_size(line)
                        if line_buffer:
                            header = f"{group_id},{len(line_buffer)},0,0\n"
                            stream.write(header)
                            stream.writelines(line_buffer)
                            current_size += encoded_size(header) + line_buffer_size
                        stream.close()
                        written_paths.append(current_path)
                        part_index += 1
                        stream, current_path, current_size = open_part(part_index)
                        line_buffer = [line]
                        line_buffer_size = encoded_size(line)

            if line_buffer:
                header = f"{group_id},{len(line_buffer)},0,0\n"
                block_size = encoded_size(header) + line_buffer_size
                if current_size > 0 and current_size + block_size > max_file_size_bytes:
                    stream.close()
                    written_paths.append(current_path)
                    part_index += 1
                    stream, current_path, current_size = open_part(part_index)
                stream.write(header)
                stream.writelines(line_buffer)
                current_size += block_size
    finally:
        stream.close()

    written_paths.append(current_path)
    return written_paths


def visualize_result(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> None:
    o3d = import_open3d()
    if points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        selected = np.sort(
            rng.choice(points.shape[0], size=max_points, replace=False)
        )
        display_points = points[selected]
        display_colors = colors[selected]
        print(
            f"Visualization sampled {max_points:,} of {points.shape[0]:,} points."
        )
    else:
        display_points = points
        display_colors = colors

    display_points = np.ascontiguousarray(display_points, dtype=np.float64).copy()
    display_colors = np.ascontiguousarray(display_colors, dtype=np.float64).copy()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(display_points)
    cloud.colors = o3d.utility.Vector3dVector(display_colors)
    print(
        "Opening Open3D visualization window. "
        "Close the window to finish the program."
    )
    visualizer = o3d.visualization.Visualizer()
    created = visualizer.create_window(
        window_name="NCNM planar discontinuity recognition",
        width=1280,
        height=800,
        visible=True,
    )
    if not created:
        raise RuntimeError(
            "Open3D could not create a visualization window. "
            "Run the script from an interactive Windows desktop session, or open "
            "the saved recognized_discontinuities_colored.ply/pcd in CloudCompare."
        )
    visualizer.add_geometry(cloud)
    visualizer.reset_view_point(True)
    visualizer.run()
    visualizer.destroy_window()


def parse_noise_color(value: str) -> tuple[float, float, float]:
    try:
        numbers = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Noise color must be formatted as R,G,B, for example 0.55,0.55,0.55."
        ) from exc
    if len(numbers) != 3 or any(number < 0 or number > 1 for number in numbers):
        raise argparse.ArgumentTypeError(
            "Noise color must contain three values between 0 and 1."
        )
    return numbers


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel NCNM recognition for planar discontinuities in large-scale "
            "point clouds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input point cloud.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ncnm_results"),
        help="Output directory.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.10,
        help="Voxel downsampling size. Set 0 to disable.",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=0.0,
        help="Absolute neighborhood radius. A value of 0 enables automatic spacing.",
    )
    parser.add_argument(
        "--radius-factor",
        type=float,
        default=3.0,
        help="Radius = radius_factor x estimated median point spacing.",
    )
    parser.add_argument("--max-nn", type=int, default=80)
    parser.add_argument("--min-nn", type=int, default=8)
    parser.add_argument("--fallback-knn", type=int, default=20)
    parser.add_argument(
        "--refine-angle-threshold",
        type=float,
        default=8.0,
        help="Normal-angle threshold for local planar support refinement (degrees).",
    )
    parser.add_argument(
        "--edge-angle-threshold",
        type=float,
        default=10.0,
        help="Normal-angle threshold for normal-consistent neighbor selection.",
    )
    parser.add_argument("--min-refine-points", type=int, default=3)
    parser.add_argument(
        "--min-subset-size",
        type=int,
        default=3,
        help="Minimum number of points in a normal-consistent local subset.",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=50,
        help="Minimum point count retained as a final discontinuity.",
    )
    parser.add_argument(
        "--partition-size",
        type=int,
        default=600_000,
        help="Maximum number of core points per spatial partition.",
    )
    parser.add_argument(
        "--partition-axis",
        choices=("auto", "x", "y", "z"),
        default="auto",
        help="Axis used for spatial slab partitioning.",
    )
    parser.add_argument(
        "--partition-overlap-factor",
        type=float,
        default=2.0,
        help="Halo distance as a multiple of the neighborhood radius.",
    )
    parser.add_argument(
        "--cross-min-shared-points",
        type=int,
        default=3,
        help="Minimum shared halo points required for cross-partition merging.",
    )
    parser.add_argument(
        "--cross-angle-threshold",
        type=float,
        default=0.0,
        help="Cross-partition normal threshold; 0 reuses edge-angle-threshold.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="Number of parallel worker processes.",
    )
    parser.add_argument("--query-batch-size", type=int, default=20_000)
    parser.add_argument("--pca-batch-size", type=int, default=4_096)
    parser.add_argument("--color-seed", type=int, default=2026)
    parser.add_argument(
        "--coordinate-sample-size",
        type=int,
        default=120,
        help="Maximum random coordinate points exported for each discontinuity.",
    )
    parser.add_argument(
        "--coordinate-sample-seed",
        type=int,
        default=2026,
        help="Random seed used when sampling exported discontinuity coordinates.",
    )
    parser.add_argument(
        "--noise-color",
        type=parse_noise_color,
        default=(0.55, 0.55, 0.55),
        help="Color for components below min-component-size, formatted as R,G,B.",
    )
    parser.add_argument(
        "--visualize-max-points",
        type=int,
        default=3_000_000,
        help="Maximum points shown interactively; saved files always contain all points.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Do not open the interactive Open3D result window.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep memory-mapped arrays and per-partition intermediate results.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    positive_names = (
        "max_nn",
        "min_nn",
        "fallback_knn",
        "min_refine_points",
        "min_subset_size",
        "min_component_size",
        "partition_size",
        "cross_min_shared_points",
        "workers",
        "query_batch_size",
        "pca_batch_size",
        "coordinate_sample_size",
        "visualize_max_points",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.min_nn > args.max_nn:
        raise ValueError("--min-nn cannot exceed --max-nn.")
    if args.fallback_knn > args.max_nn:
        raise ValueError("--fallback-knn cannot exceed --max-nn.")
    if args.min_refine_points > args.max_nn:
        raise ValueError("--min-refine-points cannot exceed --max-nn.")
    if args.min_subset_size > args.max_nn:
        raise ValueError("--min-subset-size cannot exceed --max-nn.")
    for name in ("refine_angle_threshold", "edge_angle_threshold"):
        value = getattr(args, name)
        if not (0 < value <= 90):
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 90].")
    if args.cross_angle_threshold and not (0 < args.cross_angle_threshold <= 90):
        raise ValueError("--cross-angle-threshold must be 0 or in (0, 90].")
    if args.voxel_size < 0 or args.neighbor_radius < 0:
        raise ValueError("Voxel size and neighbor radius cannot be negative.")
    if args.radius_factor <= 0 or args.partition_overlap_factor < 1:
        raise ValueError(
            "--radius-factor must be positive and "
            "--partition-overlap-factor must be at least 1."
        )


def determine_partition_axis(points: np.ndarray, requested: str) -> int:
    if requested != "auto":
        return {"x": 0, "y": 1, "z": 2}[requested]
    spans = np.ptp(points, axis=0)
    return int(np.argmax(spans))


def save_metadata(output_path: Path, metadata: dict[str, Any]) -> None:
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)


def safe_filename_part(value: str) -> str:
    cleaned = "".join(
        "_" if character in '<>:"/\\|?*' or ord(character) < 32 else character
        for character in value.strip()
    )
    cleaned = "_".join(cleaned.split())
    return cleaned.strip(" ._") or "point_cloud"


def format_parameter_value(value: float | int) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace(".", "p").replace("-", "m")


def build_result_file_prefix(args: argparse.Namespace) -> str:
    return (
        f"results_{format_parameter_value(args.voxel_size)}"
        f"_{format_parameter_value(args.max_nn)}"
        f"_{format_parameter_value(args.min_nn)}"
        f"_{format_parameter_value(args.refine_angle_threshold)}"
        f"_{format_parameter_value(args.edge_angle_threshold)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    validate_arguments(args)
    started = time.perf_counter()

    output_root = args.output.resolve()
    point_cloud_name = safe_filename_part(args.input.stem)
    run_stamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = output_root / f"{point_cloud_name}_{run_stamp}"
    suffix_index = 2
    while output_dir.exists():
        output_dir = output_root / f"{point_cloud_name}_{run_stamp}_{suffix_index:02d}"
        suffix_index += 1
    result_file_prefix = build_result_file_prefix(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_ncnm_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    partition_result_dir = temp_dir / "partition_results"
    partition_result_dir.mkdir(parents=True)

    print(f"{PROGRAM_NAME} v{PROGRAM_VERSION}")
    print(f"Input : {args.input.resolve()}")
    print(f"Output: {output_dir}")
    print(f"Numba acceleration: {'enabled' if NUMBA_AVAILABLE else 'not installed'}")

    load_start = time.perf_counter()
    points, input_colors = read_cloud(args.input)
    input_point_count = int(points.shape[0])
    print(f"Input points: {input_point_count:,}")
    if input_colors is not None:
        print("Input colors: preserved for non-structure points")
    if args.voxel_size > 0:
        points, input_colors = voxel_downsample_cloud(
            points, input_colors, args.voxel_size
        )
        print(
            f"After voxel downsampling ({args.voxel_size:g}): "
            f"{points.shape[0]:,} points"
        )
    point_count = int(points.shape[0])
    load_seconds = time.perf_counter() - load_start

    points_path = temp_dir / "points.npy"
    np.save(points_path, points)
    # Reopen as a memory map so parent and workers use the same disk-backed file.
    del points
    points_memmap = np.load(points_path, mmap_mode="r")

    partition_axis = determine_partition_axis(points_memmap, args.partition_axis)
    axis_name = "xyz"[partition_axis]
    sort_start = time.perf_counter()
    sorted_ids = np.argsort(
        points_memmap[:, partition_axis], kind="stable"
    ).astype(np.int64)
    sorted_ids_path = temp_dir / "sorted_ids.npy"
    np.save(sorted_ids_path, sorted_ids)
    sort_seconds = time.perf_counter() - sort_start

    if args.neighbor_radius > 0:
        radius = float(args.neighbor_radius)
        spacing = radius / args.radius_factor
        spacing_source = "derived from explicit neighbor radius"
    else:
        spacing = estimate_median_spacing(points_memmap, sorted_ids)
        radius = float(spacing * args.radius_factor)
        spacing_source = "estimated from local nearest-neighbor distances"
    halo_distance = radius * args.partition_overlap_factor
    print(f"Median point spacing: {spacing:.6g}")
    print(f"Neighborhood radius : {radius:.6g}")
    print(f"Partition halo      : {halo_distance:.6g}")
    print(f"Partition axis      : {axis_name}")

    config = RecognitionConfig(
        radius=radius,
        max_nn=args.max_nn,
        min_nn=args.min_nn,
        fallback_knn=args.fallback_knn,
        refine_angle_threshold=args.refine_angle_threshold,
        edge_angle_threshold=args.edge_angle_threshold,
        min_refine_points=args.min_refine_points,
        min_subset_size=args.min_subset_size,
        query_batch_size=args.query_batch_size,
        pca_batch_size=args.pca_batch_size,
    )

    global_start = time.perf_counter()
    print("Building global neighborhoods...")
    neighborhoods, counts = build_hybrid_neighborhoods(
        points_memmap,
        radius=config.radius,
        max_nn=config.max_nn,
        min_nn=config.min_nn,
        fallback_knn=config.fallback_knn,
        batch_size=config.query_batch_size,
    )
    neighbors_path = temp_dir / "neighbors.npy"
    counts_path = temp_dir / "neighbor_counts.npy"
    np.save(neighbors_path, neighborhoods)
    np.save(counts_path, counts)

    print("Estimating initial normals from global neighborhoods...")
    initial_normals = initial_pca_normals(
        points_memmap,
        neighborhoods,
        counts,
        batch_size=config.pca_batch_size,
    )
    initial_normals_path = temp_dir / "initial_normals.npy"
    np.save(initial_normals_path, initial_normals.astype(np.float32))
    del neighborhoods, counts, initial_normals
    global_seconds = time.perf_counter() - global_start

    tasks = build_partition_tasks(
        points_memmap,
        sorted_ids,
        axis=partition_axis,
        partition_size=args.partition_size,
        halo_distance=halo_distance,
        points_path=points_path,
        sorted_ids_path=sorted_ids_path,
        neighbors_path=neighbors_path,
        counts_path=counts_path,
        initial_normals_path=initial_normals_path,
        partition_result_dir=partition_result_dir,
        config=config,
    )
    del sorted_ids

    print(f"Spatial partitions : {len(tasks)}")
    print(f"Parallel workers   : {args.workers}")

    local_start = time.perf_counter()
    local_results = run_partitions(tasks, workers=args.workers)
    local_seconds = time.perf_counter() - local_start

    corrected_normals = collect_corrected_normals(local_results, point_count)
    corrected_normals_path = temp_dir / "corrected_normals_global.npy"
    np.save(corrected_normals_path, corrected_normals)
    edge_tasks = [
        replace(task, corrected_normals_path=str(corrected_normals_path))
        for task in tasks
    ]

    merge_start = time.perf_counter()
    edge_results = run_edge_partitions(edge_tasks, workers=args.workers)
    point_roots, merge_diagnostics = merge_global_edges(
        edge_results,
        point_count=point_count,
    )
    labels, final_sizes = filter_and_relabel(point_roots, args.min_component_size)
    merge_seconds = time.perf_counter() - merge_start

    group_count = int(final_sizes.size)
    recognized_count = int(np.count_nonzero(labels >= 0))
    unassigned_count = point_count - recognized_count
    print(f"Final discontinuities: {group_count:,}")
    print(f"Recognized points     : {recognized_count:,}")
    print(f"Filtered/noise points : {unassigned_count:,}")

    output_start = time.perf_counter()
    palette = generate_random_colors(group_count, args.color_seed)
    colors = colorize_labels(labels, palette, args.noise_color, input_colors)
    pcd_path = output_dir / f"{result_file_prefix}_recognized_discontinuities_colored.pcd"
    write_binary_pcd(pcd_path, points_memmap, colors, labels)
    ply_path = output_dir / f"{result_file_prefix}_recognized_discontinuities_colored.ply"
    write_binary_ply(ply_path, points_memmap, colors, labels)

    indices_path = output_dir / f"{result_file_prefix}_structure_point_indices.txt"
    write_structure_point_indices(indices_path, labels)
    attitude_coordinates_path = (
        output_dir / f"{result_file_prefix}_attitude_coordinates.txt"
    )
    attitude_coordinate_paths = write_attitude_coordinate_format(
        attitude_coordinates_path,
        points_memmap,
        labels,
        max_points_per_group=args.coordinate_sample_size,
        sample_seed=args.coordinate_sample_seed,
    )
    output_seconds = time.perf_counter() - output_start

    total_seconds = time.perf_counter() - started

    print(f"Run folder : {output_dir}")
    print(f"Colored PCD: {pcd_path}")
    print(f"Colored PLY: {ply_path}")
    print(f"Indices TXT: {indices_path}")
    print(f"Attitude coordinate TXT files: {len(attitude_coordinate_paths)}")
    for attitude_coordinate_path in attitude_coordinate_paths:
        print(f"  {attitude_coordinate_path}")
    print(f"Total time : {total_seconds:.2f} s")

    if not args.no_visualize:
        try:
            visualize_result(
                points_memmap,
                colors,
                max_points=args.visualize_max_points,
                seed=args.color_seed,
            )
        except Exception as exc:
            print(f"Visualization was skipped: {exc}", file=sys.stderr)
            print(
                "The colored PCD file was still saved and can be opened in "
                "CloudCompare.",
                file=sys.stderr,
            )

    if not args.keep_temp:
        del points_memmap
        shutil.rmtree(temp_dir)

    print(f"总共识别到结构面: {group_count:,} 个")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

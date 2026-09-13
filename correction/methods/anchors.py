"""Anchor statistics and displacement propagation migrated from LifEgo."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import spsolve


def fit_anchor_distribution(points: np.ndarray, shrinkage: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"anchor fitting needs at least two 3D points, got {points.shape}")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("covariance shrinkage must lie in [0, 1]")
    mean = points.mean(axis=0)
    covariance = np.asarray(np.cov(points, rowvar=False, ddof=1), dtype=np.float64)
    scale = max(float(np.trace(covariance) / 3.0), 1e-10)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(3)
    covariance += 1e-10 * np.eye(3)
    return mean, covariance


def _second_difference(n: int) -> csc_matrix:
    if n < 3:
        return csc_matrix((0, n), dtype=np.float64)
    one = np.ones(n - 2, dtype=np.float64)
    return diags((one, -2.0 * one, one), (0, 1, 2), shape=(n - 2, n), format="csc")


def _merge_controls(indices: list[int], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    merged: dict[int, np.ndarray] = {}
    for index, value in zip(indices, values):
        value = np.asarray(value, dtype=np.float64)
        if index in merged and not np.allclose(merged[index], value, atol=1e-10):
            raise ValueError(f"conflicting correction controls at frame {index}")
        merged[index] = value
    ordered = np.asarray(sorted(merged), dtype=np.int64)
    return ordered, np.asarray([merged[int(index)] for index in ordered], dtype=np.float64)


def _linear(n: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.arange(n, dtype=np.float64)
    return np.column_stack([np.interp(grid, indices, values[:, axis]) for axis in range(values.shape[1])])


def _min_bending(n: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    if n <= 2 or len(indices) >= n:
        return _linear(n, indices, values)
    operator = _second_difference(n)
    hessian = (operator.T @ operator).tocsc()
    fixed = np.zeros(n, dtype=bool)
    fixed[indices] = True
    free_indices = np.flatnonzero(~fixed)
    fixed_indices = np.flatnonzero(fixed)
    lookup = {int(index): value for index, value in zip(indices, values)}
    fixed_values = np.asarray([lookup[int(index)] for index in fixed_indices])
    free_hessian = hessian[free_indices][:, free_indices] + 1e-12 * eye(len(free_indices), format="csc")
    rhs = -(hessian[free_indices][:, fixed_indices] @ fixed_values)
    output = np.zeros((n, values.shape[1]), dtype=np.float64)
    output[fixed_indices] = fixed_values
    for axis in range(values.shape[1]):
        output[free_indices, axis] = spsolve(free_hessian, rhs[:, axis])
    return output


def _smooth_spline(
    n: int,
    indices: np.ndarray,
    values: np.ndarray,
    *,
    endpoint_weight: float,
    anchor_weight: float,
    bend_weight: float,
    magnitude_weight: float,
) -> np.ndarray:
    weights = np.full(len(indices), float(anchor_weight), dtype=np.float64)
    weights[(indices == 0) | (indices == n - 1)] = float(endpoint_weight)
    observations = np.zeros(n, dtype=np.float64)
    rhs = np.zeros((n, values.shape[1]), dtype=np.float64)
    for index, weight, value in zip(indices, weights, values):
        observations[int(index)] += weight
        rhs[int(index)] += weight * value
    operator = _second_difference(n)
    hessian = diags(observations, 0, shape=(n, n), format="csc")
    hessian += float(bend_weight) * (operator.T @ operator)
    hessian += max(float(magnitude_weight), 1e-12) * eye(n, format="csc")
    return np.column_stack([spsolve(hessian, rhs[:, axis]) for axis in range(values.shape[1])])


def propagate_anchor_values(
    n: int,
    anchor_frames: list[int],
    values_at_anchors: np.ndarray,
    *,
    method: str = "linear",
    endpoint_weight: float = 1.0,
    anchor_weight: float = 1.0,
    bend_weight: float = 100.0,
    magnitude_weight: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_at_anchors = np.asarray(values_at_anchors, dtype=np.float64)
    if values_at_anchors.ndim == 1:
        values_at_anchors = values_at_anchors[:, None]
    if values_at_anchors.ndim != 2 or values_at_anchors.shape[0] != len(anchor_frames):
        raise ValueError(
            "anchor values must have shape (number_of_anchor_frames, value_dimension)"
        )
    width = values_at_anchors.shape[1]
    control_frames = list(anchor_frames)
    control_values = [value for value in values_at_anchors]
    # Preserve the legacy zero-displacement endpoints only when the caller has
    # not supplied those endpoints as real calibration anchors.
    if 0 not in control_frames:
        control_frames.insert(0, 0)
        control_values.insert(0, np.zeros(width, dtype=np.float64))
    if n - 1 not in control_frames:
        control_frames.append(n - 1)
        control_values.append(np.zeros(width, dtype=np.float64))
    indices, values = _merge_controls(control_frames, np.asarray(control_values))
    if method == "linear":
        correction = _linear(n, indices, values)
    elif method == "min_bending":
        correction = _min_bending(n, indices, values)
    elif method == "smooth_spline":
        correction = _smooth_spline(
            n,
            indices,
            values,
            endpoint_weight=endpoint_weight,
            anchor_weight=anchor_weight,
            bend_weight=bend_weight,
            magnitude_weight=magnitude_weight,
        )
    else:
        raise ValueError(f"unsupported correction propagation method {method!r}")
    return correction, indices, values


def propagate_anchor_displacements(
    n: int,
    anchor_frames: list[int],
    displacements: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    displacements = np.asarray(displacements, dtype=np.float64).reshape(len(anchor_frames), 3)
    return propagate_anchor_values(n, anchor_frames, displacements, **kwargs)

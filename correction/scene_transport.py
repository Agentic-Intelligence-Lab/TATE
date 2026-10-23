"""Scene-conditioned transport of complete phase-normalized SE(3) trajectories.

Research model for unpaired demonstrations with shared visual scene coordinates.
It does not infer object geometry from EEF and does not require paired episodes.
All fitting inputs must be calibration cohorts supplied explicitly by the caller.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def features(scene: np.ndarray) -> np.ndarray:
    scene = np.asarray(scene, dtype=float)
    if scene.ndim != 2 or scene.shape[1] != 2 or not np.isfinite(scene).all():
        raise ValueError("scene must contain finite Nx2 visual coordinates")
    return np.column_stack([np.ones(len(scene)), scene])


def regress(scene: np.ndarray, values: np.ndarray, ridge: float = .01) -> np.ndarray:
    x = features(scene)
    y = np.asarray(values).reshape(len(x), -1)
    return np.linalg.solve(x.T @ x + np.diag([0., ridge, ridge]), x.T @ y)


def covariance(x: np.ndarray, shrinkage: float = .2) -> np.ndarray:
    c = np.atleast_2d(np.cov(x, rowvar=False))
    scale = max(float(np.trace(c)/len(c)), 1e-10)
    return (1-shrinkage)*c + shrinkage*scale*np.eye(len(c))


def power_spd(a: np.ndarray, power: float) -> np.ndarray:
    w, v = np.linalg.eigh((a+a.T)/2)
    return (v*np.maximum(w, 1e-12)**power) @ v.T


def gaussian_transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Symmetric positive Gaussian W2 map; vectors are column vectors."""
    root = power_spd(source, .5)
    inv = power_spd(source, -.5)
    return inv @ power_spd(root @ target @ root, .5) @ inv


class ResidualTransport:
    """Affine scene means + one low-rank map shared by a complete trajectory.

    Unlike independent frame/anchor maps, the latent coordinates couple phases.
    Residuals outside the learned subspace are attenuated by ``outside_scale``.
    """
    def __init__(self, rank=6, ridge=.01, outside_scale=.25):
        self.rank, self.ridge, self.outside_scale = rank, ridge, outside_scale

    def fit(self, source, source_scene, target, target_scene):
        source, target = np.asarray(source), np.asarray(target)
        if source.shape[1:] != target.shape[1:] or min(len(source), len(target)) < 4:
            raise ValueError("matching trajectory shapes and >=4 calibration samples required")
        self.shape = source.shape[1:]
        self.source_coef = regress(source_scene, source, self.ridge)
        self.target_coef = regress(target_scene, target, self.ridge)
        es = source.reshape(len(source), -1)-features(source_scene) @ self.source_coef
        er = target.reshape(len(target), -1)-features(target_scene) @ self.target_coef
        # Equal domain weight despite 20 vs 100 demonstrations.
        joint = np.concatenate([es/np.sqrt(len(es)), er/np.sqrt(len(er))])
        _, _, vt = np.linalg.svd(joint, full_matrices=False)
        rank = min(self.rank, len(source)-3, len(target)-3, vt.shape[0])
        self.basis = vt[:rank].T
        self.map = gaussian_transport(covariance(es @ self.basis), covariance(er @ self.basis))
        eig, vec = np.linalg.eigh(self.map)
        self.unclipped_eigenvalues = eig
        self.map = (vec*np.clip(eig, .25, 2.)) @ vec.T
        return self

    def predict(self, source, scene, mode="transport"):
        source = np.asarray(source)
        if source.shape[1:] != self.shape:
            raise ValueError("trajectory shape differs from fitted model")
        mean = features(scene) @ self.target_coef
        residual = source.reshape(len(source), -1)-features(scene) @ self.source_coef
        if mode == "mean":
            result = mean
        elif mode == "bias":
            result = mean + residual
        elif mode == "transport":
            latent = residual @ self.basis
            result = mean + latent @ self.map.T @ self.basis.T
            result += self.outside_scale*(residual-latent @ self.basis.T)
        else:
            raise ValueError(mode)
        return result.reshape((len(source),)+self.shape)

    def artifact(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


class SceneTrajectoryTransport:
    """Position and local SO(3) residual transport with a shared phase grid."""
    def __init__(self, rank=6, ridge=.01, outside_scale=.25):
        self.position = ResidualTransport(rank, ridge, outside_scale)
        self.orientation = ResidualTransport(rank, ridge, outside_scale)

    def local_rotvec(self, matrices):
        n, steps = matrices.shape[:2]
        relative = np.einsum("tji,ntjk->ntik", self.rotation_reference, matrices)
        return Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(n, steps, 3)

    def fit(self, source_p, source_r, source_scene, target_p, target_r, target_scene):
        self.rotation_reference = np.array([Rotation.from_matrix(target_r[:, i]).mean().as_matrix()
                                            for i in range(target_r.shape[1])])
        self.position.fit(source_p, source_scene, target_p, target_scene)
        self.orientation.fit(self.local_rotvec(source_r), source_scene,
                             self.local_rotvec(target_r), target_scene)
        return self

    def predict(self, source_p, source_r, scene, mode="transport"):
        p = self.position.predict(source_p, scene, mode)
        rv = self.orientation.predict(self.local_rotvec(source_r), scene, mode)
        r = Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(source_r.shape)
        return p, np.einsum("tij,ntjk->ntik", self.rotation_reference, r)

    def artifact(self):
        return dict(schema="tate.scene_trajectory_transport", schema_version=1,
                    position=self.position.artifact(), orientation=self.orientation.artifact(),
                    rotation_reference=self.rotation_reference.tolist())

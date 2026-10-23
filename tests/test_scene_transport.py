"""Numerical invariants for the experimental unpaired trajectory transport."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from correction.scene_transport import ResidualTransport, SceneTrajectoryTransport, gaussian_transport


class SceneTransportTest(unittest.TestCase):
    def test_gaussian_map_matches_covariance(self):
        a=np.array([[3.,.4],[.4,1.]])
        b=np.array([[1.,-.2],[-.2,2.]])
        m=gaussian_transport(a,b)
        np.testing.assert_allclose(m@a@m.T,b,atol=1e-9)
        self.assertGreater(np.linalg.eigvalsh(m).min(),0)

    def test_identical_domains_preserve_trajectory_without_denoising(self):
        rng=np.random.default_rng(5)
        scene=rng.normal(size=(20,2))
        p=rng.normal(size=(20,9,3))
        model=ResidualTransport(outside_scale=1).fit(p,scene,p,scene)
        np.testing.assert_allclose(model.predict(p,scene),p,atol=1e-8)

    def test_affine_scene_mean_generalizes_without_paired_rows(self):
        rng=np.random.default_rng(7)
        a=rng.uniform(-1,1,(40,2)); b=rng.uniform(-1,1,(20,2)); query=rng.uniform(-1,1,(12,2))
        def true(c):
            return np.repeat(np.column_stack([c[:,0],c[:,1],np.ones(len(c))])[:,None,:],7,axis=1)
        model=ResidualTransport(ridge=0).fit(true(a)+.3,a,true(b),b)
        pred=model.predict(true(query)+.3,query,mode="bias")
        np.testing.assert_allclose(pred,true(query),atol=1e-9)
        self.assertGreater(np.std(pred[:,3,0]),.1)

    def test_se3_identity_and_valid_rotations(self):
        rng=np.random.default_rng(13)
        c=rng.normal(size=(20,2)); p=rng.normal(size=(20,7,3))
        r=Rotation.from_rotvec(rng.normal(0,.2,(140,3))).as_matrix().reshape(20,7,3,3)
        model=SceneTrajectoryTransport(outside_scale=1).fit(p,r,c,p,r,c)
        got_p,got_r=model.predict(p,r,c)
        np.testing.assert_allclose(got_p,p,atol=1e-8)
        np.testing.assert_allclose(got_r,r,atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(got_r),1,atol=1e-10)


if __name__=="__main__":
    unittest.main()

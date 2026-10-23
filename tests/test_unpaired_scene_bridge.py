import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from correction.unpaired_scene_bridge import SceneBridge

class UnpairedBridgeTest(unittest.TestCase):
    def sample(self,n,seed):
        rng=np.random.default_rng(seed);scene=rng.uniform(-1,1,(n,2))
        p=rng.normal(0,.003,(n,81,3))+np.column_stack([scene*.1,np.zeros(n)])[:,None]
        r=Rotation.from_rotvec(rng.normal(0,.05,(n*81,3))).as_matrix().reshape(n,81,3,3)
        return p,r,scene

    def test_unequal_domains_independent_permutations(self):
        p,r,s=self.sample(40,1);q,t,u=self.sample(12,2)
        a=SceneBridge().fit(p,r,s,q,t,u).predict(p,r,s)
        b=SceneBridge().fit(p[::-1],r[::-1],s[::-1],q[::-1],t[::-1],u[::-1]).predict(p,r,s)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-9)

    def test_identity_domain_preserves_mean_and_bounded_residual(self):
        p,r,s=self.sample(20,3)
        model=SceneBridge(outside_scale=1).fit(p,r,s,p,r,s)
        a,b=model.predict(p,r,s)
        np.testing.assert_allclose(a,p,atol=1e-9);np.testing.assert_allclose(b,r,atol=1e-9)

    def test_serialization_and_noncollapsed_motion(self):
        p,r,s=self.sample(40,1);q,t,u=self.sample(20,2)
        model=SceneBridge().fit(p,r,s,q+.1,t,u)
        restored=SceneBridge.from_artifact(model.artifact())
        a=model.predict(p,r,s);b=restored.predict(p,r,s)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-12)
        changed=p.copy();changed[:,30,2]+=.02
        c=restored.predict(changed,r,s)[0]
        self.assertGreater(np.linalg.norm(c-a[0]),.001)
        np.testing.assert_allclose(a[1].transpose(0,1,3,2)@a[1],np.broadcast_to(np.eye(3),a[1].shape),atol=1e-10)

if __name__=='__main__':unittest.main()

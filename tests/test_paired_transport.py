import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from correction.paired_transport import PairedCalibrator


class PairedCalibrationTest(unittest.TestCase):
    def samples(self,n=20):
        rng=np.random.default_rng(481)
        p=rng.normal(0,.1,(n,81,3));c=rng.normal(size=(n,2))
        r=Rotation.from_rotvec(rng.normal(0,.15,(n*81,3))).as_matrix().reshape(n,81,3,3)
        return p,r,c

    def test_identity_and_input_not_modified(self):
        p,r,c=self.samples();before=p.copy()
        model=PairedCalibrator(retain=1.,smooth=0.).fit(p,r,c,p,r)
        q,rot=model.predict(p,r,c)
        np.testing.assert_allclose(q,p,atol=1e-10)
        np.testing.assert_allclose(rot,r,atol=1e-10)
        np.testing.assert_array_equal(p,before)

    def test_known_scene_dependent_correction_generalizes(self):
        p,r,c=self.samples(30)
        delta=np.column_stack([.03*c[:,0],.07*c[:,1],np.ones(30)*.1])[:,None,:]
        target=p+delta
        model=PairedCalibrator(features='scene',ridge=0.,retain=1.,smooth=0.).fit(p[:20],r[:20],c[:20],target[:20],r[:20])
        out,_=model.predict(p[20:],r[20:],c[20:])
        np.testing.assert_allclose(out,target[20:],atol=1e-10)

    def test_artifact_round_trip(self):
        p,r,c=self.samples()
        model=PairedCalibrator().fit(p,r,c,p+.1,r)
        restored=PairedCalibrator.from_artifact(model.artifact())
        for a,b in zip(model.predict(p,r,c),restored.predict(p,r,c)):
            np.testing.assert_allclose(a,b,atol=1e-12)

    def test_invalid_data_rejected(self):
        p,r,c=self.samples();p[1,1,1]=np.nan
        with self.assertRaises(ValueError):PairedCalibrator().fit(p,r,c,p,r)

    def test_rbf_bias_and_serialization(self):
        p,r,c=self.samples(30)
        model=PairedCalibrator(basis='rbf',retain=1.,smooth=0.).fit(p[:20],r[:20],c[:20],p[:20]+.12,r[:20])
        restored=PairedCalibrator.from_artifact(model.artifact())
        got,_=restored.predict(p[20:],r[20:],c[20:])
        np.testing.assert_allclose(got,p[20:]+.12,atol=1e-9)

    def test_native_inference_ignores_all_real_targets(self):
        from scripts.paired_redcube_apply import apply
        p,r,c=self.samples()
        model=PairedCalibrator(retain=1.,smooth=0.).fit(p,r,c,p+.1,r)
        bundle=dict(position=model.artifact(),rotation=model.artifact(),calibration_pair_ids=list(range(20)),
                    fallback_global_affine=np.vstack([np.eye(3),np.ones(3)*.1]).tolist(),fallback_local_rotation=np.eye(3).tolist())
        record=dict(id=0,episode_id=1,path='synthetic',sha256='synthetic',p=p[0].tolist(),r=r[0].tolist(),
            t=np.linspace(0,4,81).tolist(),valid=[True]*81,
            events=[dict(frame=f,transition=t,ambiguous=False) for f,t in [(20,'open'),(40,'close'),(60,'open')]],
            landmarks=dict(red=dict(uv=(c[0]*100).tolist(),jitter_px=0),blue=dict(uv=[0,0],jitter_px=0)))
        data=dict(records=dict(ego=[record],real='must not be accessed'),
                  pairing=dict(pairs=[dict(pair_id=i,real_episode=2*i,ego_episode=2*i+1) for i in range(20)]))
        payload,metrics=apply(bundle,data,'synthetic')
        np.testing.assert_allclose(payload['methods']['paired_selected'][0]['p'],p[0]+.1,atol=1e-10)
        self.assertEqual(metrics['fallback_pairs'],[])


if __name__=='__main__':unittest.main()

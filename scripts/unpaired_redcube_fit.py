"""Train on a six-array identity-free NPZ. Never reads evaluation files."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from correction.unpaired_scene_bridge import SceneBridge
from correction.scene_transport import SceneTrajectoryTransport
from scripts.random_position_experiment import change_anchors,global_rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    z=np.load(args.training)
    assert set(z.files)=={'source_p','source_r','source_scene','target_p','target_r','target_scene'}
    ep,er,es=z['source_p'],z['source_r'],z['source_scene'];rp,rr,rs=z['target_p'],z['target_r'],z['target_scene']
    assert len(rp)==20 and len(ep)==100
    model=SceneBridge().fit(ep,er,es,rp,rr,rs)
    predictions={'raw':(ep,er),'unpaired_bridge':model.predict(ep,er,es)}
    predictions['scene_residual']=model.predict(ep,er,es,mode='residual')
    predictions['scene_mean']=model.predict(ep,er,es,mode='mean')
    anchor=np.array([0,20,40,60,80]);goals=ep[:,anchor]+rp[:,anchor].mean(0)-ep[:,anchor].mean(0)
    predictions['anchor_shift']=(change_anchors(ep,goals),global_rotation(er,rr,er))
    old=SceneTrajectoryTransport().fit(ep,er,es,rp,rr,rs)
    predictions['previous_unpaired_transport']=old.predict(ep,er,es)
    np.savez_compressed(args.out/'predictions.npz',**{name+'_'+key:value for name,pair in predictions.items() for key,value in zip(['p','r'],pair)})
    (args.out/'model.json').write_text(json.dumps(model.artifact(),indent=2))
    # Strong anti-pairing check: permute each domain independently and refit.
    rng=np.random.default_rng(881);a=rng.permutation(len(ep));b=rng.permutation(len(rp))
    permuted=SceneBridge().fit(ep[a],er[a],es[a],rp[b],rr[b],rs[b])
    got=permuted.predict(ep,er,es)
    errors=[float(np.max(np.abs(x-y))) for x,y in zip(predictions['unpaired_bridge'],got)]
    assert max(errors)<1e-8,errors
    restored=SceneBridge.from_artifact(model.artifact()).predict(ep,er,es)
    assert all(np.allclose(x,y,atol=1e-12) for x,y in zip(predictions['unpaired_bridge'],restored))
    audit=dict(training_keys=z.files,source_count=len(ep),target_count=len(rp),independent_domain_permutation_max_error=errors,
        primary='unpaired_bridge',reference_labels_opened=False,
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.training,args.out/'model.json',args.out/'predictions.npz']})
    (args.out/'sealed_predictions.json').write_text(json.dumps(audit,indent=2))
    print(args.out,'sealed; independent-domain permutation max error',errors)


if __name__=='__main__':main()

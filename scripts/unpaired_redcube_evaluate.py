"""Open correspondence vault only after immutable predictions are present."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.paired_redcube_experiment import errors,stats,bootstrap_improvement


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();protocol=json.loads((args.root/'protocol.json').read_text());all_reports=[]
    # Require ALL five runs to have been sealed before revealing ANY scores.
    for seed in protocol['seeds']:
        run=args.root/str(seed);seal=json.loads((run/'sealed_predictions.json').read_text())
        for name,digest in seal['sha256'].items():
            assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest,(seed,name)
    for seed in protocol['seeds']:
        run=args.root/str(seed);z=np.load(run/'predictions.npz');v=np.load(run/'evaluation_private.npz')
        t=np.load(run/'training.npz');test=v['heldout_mask'];consistent=test&(np.linalg.norm(t['source_scene']-v['real_scene'],axis=1)*100<=6)
        report=dict(seed=seed,primary=protocol['primary'],test_count=int(test.sum()),reference_overlap_count=int((~test).sum()),
            image_consistent_test_count=int(consistent.sum()),metrics={},per_episode={},bootstrap={},notes=[
            'All 100 ego inputs used without labels: transductive calibration.',
            '20 robot reference trajectories; their ego counterparts identified only at evaluation.',
            protocol['note']])
        cache={}
        for method in protocol['methods']:
            e=errors(z[method+'_p'],z[method+'_r'],v['real_p'],v['real_r'],v['observed_mask']);cache[method]=e
            report['metrics'][method]={name:{k:stats(val[mask]) for k,val in e.items()} for name,mask in [('heldout80',test),('reference_overlap20',~test),('all100',np.ones(len(test),bool)),('image_consistent_test',consistent)] if mask.any()}
            report['per_episode'][method]=[dict(ego_episode_id=int(v['ego_episode_ids'][i]),pair_id=int(v['pair_ids'][i]),heldout=bool(test[i]),**{k:float(val[i]) for k,val in e.items()}) for i in range(len(test))]
        for baseline in ['raw','previous_unpaired_transport','scene_mean']:
            report['bootstrap'][baseline]={k:bootstrap_improvement(cache[baseline][k][test],cache[protocol['primary']][k][test]) for k in ['path_m','grasp_m','release_m','rotation_deg']}
        (run/'results.json').write_text(json.dumps(report,indent=2));all_reports.append(report)
        print(seed,{m:{k:round(report['metrics'][m]['heldout80'][k]['mean'],5) for k in ['grasp_m','release_m','path_m','rotation_deg']} for m in protocol['methods']},flush=True)
    summary=dict(protocol=protocol,per_run=[dict(seed=r['seed'],image_consistent_test_count=r['image_consistent_test_count']) for r in all_reports],metrics={})
    for method in protocol['methods']:
        summary['metrics'][method]={k:dict(mean=float(np.mean(vals)),std_across_reference_draws=float(np.std(vals,ddof=1)),min=float(min(vals)),max=float(max(vals))) for k in ['grasp_m','release_m','path_m','rotation_deg'] for vals in [[r['metrics'][method]['heldout80'][k]['mean'] for r in all_reports]]}
    (args.root/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()

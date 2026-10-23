"""Create identity-free training inputs and a separate evaluator-only vault."""
import argparse,gzip,json,hashlib
from pathlib import Path
import numpy as np
from scripts.random_position_experiment import load_dataset
from scripts.paired_redcube_experiment import observed_phase_mask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--first-seed',type=int,default=20261011)
    parser.add_argument('--benchmark-note',default='Previously studied dataset, not a fresh blind holdout.')
    parser.add_argument('--scene-rule',default='initial red minus blue centroid')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cohorts,exclusions,data=load_dataset(args.dataset)
    if exclusions:raise ValueError('Audit exclusions before creating benchmark')
    real=sorted(cohorts['real']['records'],key=lambda r:r['raw']['episode_id'])
    ego=sorted(cohorts['ego']['records'],key=lambda r:r['raw']['episode_id'])
    # Target selection depends on robot episode order, never its paired ego.
    rng=np.random.default_rng(7011);order=rng.permutation(len(ego));ego=[ego[i] for i in order]
    seeds=list(range(args.first_seed,args.first_seed+5))
    protocol=dict(primary='unpaired_bridge',seeds=seeds,primary_seed=seeds[0],real_budget_per_run=20,source_ego_count=len(ego),
        parameter_selection='Independent within-domain scene-regression LOO only; no paired labels',
        fixed_parameters=dict(rank=6,outside_scale=.25,shrinkage=.3),
        methods=['raw','anchor_shift','scene_residual','scene_mean','previous_unpaired_transport','unpaired_bridge'],
        note='Five separate 20-real runs; never ensemble across runs. '+args.benchmark_note,
        scene_rule=args.scene_rule,
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest())
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    lookup={r['id']:r for r in real}
    paired=[lookup[r['id']] for r in ego] # EVALUATOR ONLY, not sent to fitter.
    p=lambda rows,key:np.array([r[key] for r in rows])
    for seed in seeds:
        run=args.out/str(seed);run.mkdir(exist_ok=True)
        if (run/'training.npz').exists():raise FileExistsError(run)
        chosen=np.random.default_rng(seed).choice(len(real),20,replace=False)
        target=[real[i] for i in chosen]
        np.savez_compressed(run/'training.npz',source_p=p(ego,'p'),source_r=p(ego,'r'),source_scene=p(ego,'scene'),
            target_p=p(target,'p'),target_r=p(target,'r'),target_scene=p(target,'scene'))
        train_ids={r['id'] for r in target}
        np.savez_compressed(run/'evaluation_private.npz',real_p=p(paired,'p'),real_r=p(paired,'r'),real_scene=p(paired,'scene'),
            pair_ids=np.array([r['id'] for r in ego]),ego_episode_ids=np.array([r['raw']['episode_id'] for r in ego]),
            heldout_mask=np.array([r['id'] not in train_ids for r in ego]),
            observed_mask=np.array([observed_phase_mask(e)&observed_phase_mask(r) for e,r in zip(ego,paired)]))
        (run/'export_metadata.json').write_text(json.dumps(dict(real_episode_ids=[r['raw']['episode_id'] for r in target],
            ego_episode_ids=[r['raw']['episode_id'] for r in ego]),indent=2))
    # Source-only native data; no pairing table or robot trajectories.
    (args.out/'ego_native.json.gz').write_bytes(gzip.compress(json.dumps([dict(r['raw'],id=i) for i,r in enumerate(ego)]).encode()))
    print('Prepared five unordered 100-ego / 20-real runs; mapping only in evaluation_private.npz')


if __name__=='__main__':main()

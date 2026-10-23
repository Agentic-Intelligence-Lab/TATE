"""Offline, unpaired correction experiments; no policy learning.

Run: python -m scripts.random_position_experiment --dataset ... --out ...
The real holdout fits an independent scene-based *proxy evaluator*, not the
correction model. Proxy errors are not paired ego ground truth or success rates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import spearmanr, wasserstein_distance

from correction.methods.anchors import propagate_anchor_values
from correction.scene_transport import (SceneTrajectoryTransport, covariance, features,
                                       gaussian_transport, regress)

STEPS = 81
ANCHORS = np.array([0, 20, 40, 60, 80])
GRID = np.linspace(0, 4, STEPS)


def load_dataset(path):
    data = path if isinstance(path, dict) else json.load(gzip.open(path))
    cohorts, exclusions = {}, []
    for kind, records in data["records"].items():
        accepted = []
        for rec in records:
            if [e["transition"] for e in rec["events"]] != ["open", "close", "open"]:
                exclusions.append(dict(kind=kind, id=rec["id"], reason="unexpected event sequence", events=rec["events"]))
                continue
            if any(e["ambiguous"] for e in rec["events"]):
                exclusions.append(dict(kind=kind, id=rec["id"], reason="ambiguous event"))
                continue
            if any(rec["landmarks"][col] is None or rec["landmarks"][col]["jitter_px"] > 3
                   for col in ["red", "blue"]):
                exclusions.append(dict(kind=kind, id=rec["id"], reason="missing/unstable scene landmark"))
                continue
            valid = np.flatnonzero(rec["valid"])
            controls = [valid[0]]+[e["frame"] for e in rec["events"]]+[valid[-1]]
            times = np.asarray(rec["t"])
            knots = times[controls]
            if np.any(np.diff(knots) <= 0):
                raise ValueError("non-increasing phase boundaries")
            phase = np.interp(times, knots, np.arange(5))
            target_t = np.interp(GRID, np.arange(5), knots)
            poses = np.asarray(rec["p"])
            p = np.column_stack([np.interp(target_t, times[valid], poses[valid, i]) for i in range(3)])
            r = Slerp(times[valid], Rotation.from_matrix(np.asarray(rec["r"])[valid]))(target_t).as_matrix()
            scene = (np.array(rec["landmarks"]["red"]["uv"])-rec["landmarks"]["blue"]["uv"])/100
            accepted.append(dict(id=rec["id"], p=p, r=r, scene=scene, raw=rec, phase=phase, knots=knots))
        cohorts[kind] = dict(records=accepted, p=np.array([r["p"] for r in accepted]),
            r=np.array([r["r"] for r in accepted]), scene=np.array([r["scene"] for r in accepted]),
            ids=np.array([r["id"] for r in accepted]))
    return cohorts, exclusions, data


def projected_distance(a, b, directions):
    return float(np.mean([wasserstein_distance(a@d, b@d) for d in directions]))


def change_anchors(p, target):
    # Identical propagation for all anchor baselines, with explicit endpoints.
    result = []
    for source, goals in zip(p, target):
        delta, _, _ = propagate_anchor_values(STEPS, ANCHORS.tolist(), goals-source[ANCHORS], method="min_bending")
        result.append(source+delta)
    return np.asarray(result)


def global_rotation(source_fit, target_fit, query):
    source_mean = [Rotation.from_matrix(source_fit[:, k]).mean() for k in range(STEPS)]
    target_mean = [Rotation.from_matrix(target_fit[:, k]).mean() for k in range(STEPS)]
    bias = Rotation.from_matrix(np.array([(a.inv()*b).as_matrix() for a,b in zip(source_mean,target_mean)])).mean()
    return query @ bias.as_matrix()


def predictions(ep, er, ec, rp, rr, rc, qp, qr, qc, seed):
    model = SceneTrajectoryTransport().fit(ep, er, ec, rp, rr, rc)
    a, b = ep[:, ANCHORS], rp[:, ANCHORS]
    global_r = global_rotation(er, rr, qr)
    result = {"raw": (qp, qr)}
    target_mean = np.broadcast_to(b.mean(0), (len(qp),5,3))
    result["anchor_mean"] = (change_anchors(qp, target_mean), global_r)
    result["anchor_shift"] = (change_anchors(qp, qp[:,ANCHORS]+b.mean(0)-a.mean(0)), global_r)
    flat_a, flat_b = a.reshape(len(a),-1), b.reshape(len(b),-1)
    mapping = gaussian_transport(covariance(flat_a), covariance(flat_b))
    joint = ((qp[:,ANCHORS].reshape(len(qp),-1)-flat_a.mean(0)) @ mapping.T+flat_b.mean(0)).reshape(-1,5,3)
    result["joint_anchor_ot"] = (change_anchors(qp, joint), global_r)
    rng = np.random.default_rng(seed+567)
    random_goals = np.array([rng.multivariate_normal(b[:, k].mean(0), covariance(b[:, k]), size=len(qp))
                             for k in range(5)]).transpose(1,0,2)
    result["anchor_sample"] = (change_anchors(qp,random_goals), global_r)
    visual_goals = (features(qc) @ regress(rc,b)).reshape(-1,5,3)
    result["visual_anchor"] = (change_anchors(qp,visual_goals), global_r)
    result["visual_anchor_shuffled"] = (change_anchors(qp,visual_goals[rng.permutation(len(qp))]), global_r)
    for mode in ["bias", "transport", "mean"]:
        result["scene_"+mode] = model.predict(qp,qr,qc,mode=mode)
    # Enforce visual contact means after transporting the joint trajectory shape.
    # Keep the transported endpoints; solve a minimum-bending contact projection.
    transported, transported_r = result["scene_transport"]
    contact_projected = []
    for p, goals in zip(transported, visual_goals):
        delta, _, _ = propagate_anchor_values(STEPS,[0,40,60,80],
            np.array([np.zeros(3),goals[2]-p[40],goals[3]-p[60],np.zeros(3)]),method="min_bending")
        contact_projected.append(p+delta)
    result["scene_transport_contact"] = (np.asarray(contact_projected),transported_r)
    # Robot posture prior is a separate ablation: keep transported position
    # variation, but use conditional robot mean orientation instead of uncertain
    # human orientation residuals. This intentionally removes orientation style.
    result["scene_transport_robot_orientation"] = (np.asarray(contact_projected),result["scene_mean"][1])
    return result, model


def evaluate(pred, source, scenes, real_p, real_r, real_scene, records, directions):
    # Evaluator is fit EXCLUSIVELY on the disjoint real cohort; model never sees it.
    expected = (features(scenes) @ regress(real_scene,real_p)).reshape(source.shape)
    orient_ref = Rotation.from_matrix(real_r[:,40]).mean()
    rv = (orient_ref.inv()*Rotation.from_matrix(real_r[:,40])).as_rotvec()
    orient_expected = orient_ref*Rotation.from_rotvec(features(scenes) @ regress(real_scene,rv))
    real_cv = []
    folds = np.arange(len(real_p))%5
    for fold in range(5):
        fit, val = folds!=fold, folds==fold
        got = (features(real_scene[val]) @ regress(real_scene[fit],real_p[fit])).reshape((-1,STEPS,3))
        real_cv.extend(np.linalg.norm((got-real_p[val])[:,40],axis=1).tolist())
    metrics = {}
    for name, (p,r) in pred.items():
        row = dict(grasp_swd_m=projected_distance(p[:,40],real_p[:,40],directions),
            release_swd_m=projected_distance(p[:,60],real_p[:,60],directions),
            grasp_scene_proxy_m=float(np.linalg.norm((p-expected)[:,40],axis=1).mean()),
            path_scene_proxy_m=float(np.linalg.norm(p-expected,axis=2).mean()),
            grasp_scene_proxy_p95_m=float(np.quantile(np.linalg.norm((p-expected)[:,40],axis=1),.95)),
            orientation_scene_proxy_deg=float(np.degrees((orient_expected.inv()*Rotation.from_matrix(r[:,40])).magnitude()).mean()),
            mean_displacement_m=float(np.linalg.norm(p-source,axis=2).mean()),
            grasp_xy_spread_m=float(np.linalg.norm(p[:,40,:2].std(0))),
            grasp_scene_distance_spearman=(float(spearmanr(pdist(p[:,40,:2]),pdist(expected[:,40,:2])).statistic)
                                          if np.ptp(p[:,40,:2],axis=0).max()>1e-10 else 0.),
            phase_swd_m=float(np.mean([projected_distance(p[:,k],real_p[:,k],directions) for k in range(0,STEPS,10)])))
        speeds, accelerations, angular, lengths = [], [], [], []
        for pp, rotation, record in zip(p,r,records):
            time = np.interp(GRID,np.arange(5),record["knots"])
            dt = np.diff(time)
            speed = np.linalg.norm(np.diff(pp,axis=0)/dt[:,None],axis=1)
            velocity = np.diff(pp,axis=0)/dt[:,None]
            acc = np.linalg.norm(np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])/2)[:,None],axis=1)
            rot = Rotation.from_matrix(rotation)
            omega = (rot[:-1].inv()*rot[1:]).magnitude()/dt
            speeds.extend(speed); accelerations.extend(acc); angular.extend(omega)
            lengths.append(np.linalg.norm(np.diff(pp,axis=0),axis=1).sum())
        row.update(speed_p95_m_s=float(np.quantile(speeds,.95)), acceleration_p95_m_s2=float(np.quantile(accelerations,.95)),
            angular_speed_p95_rad_s=float(np.quantile(angular,.95)), path_length_mean_m=float(np.mean(lengths)),
            fraction_speed_over_06=float(np.mean(np.array(speeds)>.6)))
        metrics[name] = row
    return metrics, dict(real_holdout_scene_cv_error_m=float(np.mean(real_cv)))


def aggregate(runs):
    output = {}
    for protocol in sorted({r["protocol"] for r in runs}):
        group = [r for r in runs if r["protocol"]==protocol]
        output[protocol] = {}
        for method in group[0]["metrics"]:
            output[protocol][method] = {key: dict(mean=float(np.mean(values)), std=float(np.std(values)))
                for key in group[0]["metrics"][method]
                for values in [[r["metrics"][method][key] for r in group]]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=10)
    parser.add_argument("--protocols",nargs="+",default=["transductive","inductive20","spatial_extrapolation"],
                        choices=["transductive","inductive20","spatial_extrapolation"])
    parser.add_argument("--exclude-ego",nargs="*",type=int,default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    cohorts, exclusions, raw = load_dataset(args.dataset)
    real, ego = cohorts["real"], cohorts["ego"]
    if args.exclude_ego:
        keep=~np.isin(ego["ids"],args.exclude_ego)
        for episode in ego["ids"][~keep]:
            exclusions.append(dict(kind="ego",id=int(episode),reason="explicit sensitivity exclusion"))
        ego["records"]=[r for r,k in zip(ego["records"],keep) if k]
        for key in ["ids","scene","p","r"]:ego[key]=ego[key][keep]
    directions = np.random.default_rng(719).normal(size=(64,3))
    directions /= np.linalg.norm(directions,axis=1,keepdims=True)
    runs=[]
    for protocol in args.protocols:
        for seed in range(args.seeds):
            order = np.random.default_rng(seed).permutation(len(real["p"]))
            fit, test = order[:20],order[20:]
            eo = np.random.default_rng(1000+seed).permutation(len(ego["p"]))
            source_fit = np.arange(len(eo)) if protocol=="transductive" else eo[:20]
            query = np.arange(len(eo)) if protocol=="transductive" else eo[20:]
            if protocol=="spatial_extrapolation":
                boundary = np.median(real["scene"][:,0])
                fit_pool = np.flatnonzero(real["scene"][:,0]<=boundary)
                fit = np.random.default_rng(seed).permutation(fit_pool)[:20]
                test = np.flatnonzero(real["scene"][:,0]>boundary)
                ego_pool = np.flatnonzero(ego["scene"][:,0]<=boundary)
                source_fit = np.random.default_rng(1000+seed).permutation(ego_pool)[:20]
                query = np.flatnonzero(ego["scene"][:,0]>boundary)
            pred, model = predictions(ego["p"][source_fit],ego["r"][source_fit],ego["scene"][source_fit],
                real["p"][fit],real["r"][fit],real["scene"][fit],
                ego["p"][query],ego["r"][query],ego["scene"][query],seed)
            metrics, reference = evaluate(pred,ego["p"][query],ego["scene"][query],
                real["p"][test],real["r"][test],real["scene"][test],
                [ego["records"][i] for i in query],directions)
            # Independent held-out FK accuracy of the actual fitted visual model.
            real_prediction = (features(real["scene"][test]) @ regress(real["scene"][fit],real["p"][fit])).reshape(-1,STEPS,3)
            real_error = np.linalg.norm((real_prediction-real["p"][test])[:,40],axis=1)
            reference.update(calibration20_visual_real80_mean_m=float(real_error.mean()),
                calibration20_visual_real80_p95_m=float(np.quantile(real_error,.95)),
                real20_vs_real80_grasp_swd_m=projected_distance(real["p"][fit,40],real["p"][test,40],directions),
                ego_outside_calibration_visual_hull=int(np.sum(Delaunay(real["scene"][fit]).find_simplex(ego["scene"][query])<0)))
            _, predicted_real_r = model.predict(real["p"][test],real["r"][test],real["scene"][test],mode="mean")
            angle_error = (Rotation.from_matrix(predicted_real_r[:,40]).inv()*Rotation.from_matrix(real["r"][test,40])).magnitude()
            reference["calibration20_orientation_real_holdout_deg"] = float(np.degrees(angle_error).mean())
            runs.append(dict(protocol=protocol,seed=seed,real_fit_ids=real["ids"][fit].tolist(),
                real_eval_ids=real["ids"][test].tolist(),ego_fit_ids=ego["ids"][source_fit].tolist(),
                ego_eval_ids=ego["ids"][query].tolist(),metrics=metrics,reference=reference))
            if protocol=="transductive" and seed==0:
                artifact=model.artifact()
                artifact.update(real_calibration_ids=real["ids"][fit].tolist(),ego_calibration_ids=ego["ids"].tolist(),
                    scene_definition="(initial red centroid - initial blue centroid)/100 pixels", phase_grid=GRID.tolist(),
                    dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest())
                (args.out/"model_seed0.json").write_text(json.dumps(artifact,indent=2))
                arrays=dict(real_p=real["p"],real_r=real["r"],ego_p=ego["p"],ego_r=ego["r"],
                    real_scene=real["scene"],ego_scene=ego["scene"],real_fit=fit,real_test=test,ego_ids=ego["ids"])
                for key,(p,r) in pred.items(): arrays[key+"_p"]=p; arrays[key+"_r"]=r
                np.savez_compressed(args.out/"predictions_seed0.npz",**arrays)
            print(protocol,seed,{key:round(value["grasp_scene_proxy_m"]*100,2) for key,value in metrics.items()},flush=True)
            (args.out/"results.json").write_text(json.dumps(dict(
                counts={k:len(v["records"]) for k,v in cohorts.items()}, exclusions=exclusions,
                protocol_note="Exploratory unpaired evaluation. Real holdout only used by proxy evaluator/metrics. No policy training.",
                runs=runs,aggregate=aggregate(runs)),indent=2,allow_nan=False))


if __name__ == "__main__":
    main()

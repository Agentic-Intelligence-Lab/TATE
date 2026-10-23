"""Create plots and native-frame correction payloads from seed-0 experiments."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp

from correction.scene_transport import features,regress
from scripts.random_position_experiment import GRID,load_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args()
    cohorts, exclusions, _=load_dataset(args.dataset)
    z=np.load(args.experiment/"predictions_seed0.npz")
    report=json.loads((args.experiment/"results.json").read_text())
    summary=report["aggregate"]["transductive"]
    methods=["raw","anchor_mean","anchor_shift","joint_anchor_ot","anchor_sample","visual_anchor","scene_transport_robot_orientation"]
    labels=["Raw","Fixed anchors","Event shift","Joint OT","Random anchors","Visual anchors","Scene transport"]
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout="constrained")
    for kind,color in [("real","#3979a8"),("ego","#df7a40")]:
        c=cohorts[kind]["scene"]*100
        axes[0,0].scatter(c[:,0],c[:,1],s=22,alpha=.7,label=kind,color=color)
    axes[0,0].set(title="Observed cube locations differ between domains",xlabel="Cube - plate u (pixels)",ylabel="Cube - plate v (pixels)")
    axes[0,0].invert_yaxis();axes[0,0].legend()
    for key,label,color in [("real","Real","#777777"),("raw","Raw ego","#cf4c45"),("scene_transport_robot_orientation","Corrected ego","#268d7c")]:
        p=z[key+"_p"][:,40]
        axes[0,1].scatter(p[:,0]*100,p[:,1]*100,s=18,alpha=.7,label=label,color=color)
    axes[0,1].set(title="Grasp TCP positions (seed 0)",xlabel="x (cm)",ylabel="y (cm)");axes[0,1].axis("equal");axes[0,1].legend()
    x=np.arange(len(methods));w=.38
    for offset,metric,label,color in [(-w/2,"grasp_swd_m","Marginal SWD","#3979a8"),(w/2,"grasp_scene_proxy_m","Independent scene proxy","#df7a40")]:
        axes[1,0].bar(x+offset,[summary[k][metric]["mean"]*100 for k in methods],w,
            yerr=[summary[k][metric]["std"]*100 for k in methods],label=label,color=color,capsize=2)
    axes[1,0].set_xticks(x,labels,rotation=30,ha="right")
    axes[1,0].set(title="Distribution matching can hide wrong scene assignments",ylabel="Distance (cm; lower is better)");axes[1,0].legend()
    for protocol,color in [("transductive","#268d7c"),("inductive20","#3979a8"),("spatial_extrapolation","#df7a40")]:
        names=["anchor_shift","visual_anchor","scene_transport_robot_orientation"]
        a=report["aggregate"][protocol]
        axes[1,1].plot([0,1,2],[a[k]["path_scene_proxy_m"]["mean"]*100 for k in names],"o-",label=protocol,color=color)
    axes[1,1].set_xticks([0,1,2],["Event shift","Visual anchors","Scene transport"])
    axes[1,1].set(title="Whole-path consistency across evaluation protocols",ylabel="Scene-proxy distance (cm)");axes[1,1].legend()
    fig.suptitle("Random-position TATE: 100 real / 99 event-valid ego, 20 real calibration, 10 seeds\nProxy distances are not measured ego ground-truth errors or task success",fontsize=13)
    fig.savefig(args.experiment/"comparison.png",dpi=170);fig.savefig(args.experiment/"comparison.svg");plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(16,8),layout="constrained")
    for col,index in enumerate([0,20,50,80]):
        for key,label,color in [("raw","Raw","#cf4c45"),("anchor_shift","Event shift","#db9a34"),("visual_anchor","Visual anchors","#3979a8"),("scene_transport_robot_orientation","Scene transport","#268d7c")]:
            p=z[key+"_p"][index]*100
            for row,(a,b) in enumerate([(0,1),(0,2)]):
                axes[row,col].plot(p[:,a],p[:,b],label=label,color=color,alpha=.85)
                axes[row,col].scatter(p[[40,60],a],p[[40,60],b],s=20,color=color)
        axes[0,col].set(title=f"Episode {index}: XY",xlabel="x (cm)",ylabel="y (cm)")
        axes[1,col].set(title=f"Episode {index}: XZ",xlabel="x (cm)",ylabel="z (cm)")
    axes[0,0].legend(fontsize=8)
    fig.savefig(args.experiment/"trajectories.png",dpi=170);plt.close(fig)

    residual={}
    for key in ["real","raw","visual_anchor","scene_mean","scene_transport_robot_orientation"]:
        p=z[key+"_p"];c=z["real_scene"] if key=="real" else z["ego_scene"]
        mean=(features(c)@regress(c,p)).reshape(p.shape)
        residual[key]=float(np.sqrt(np.mean(np.sum((p-mean)**2,axis=2))))
    payload=dict(schema="tate.experimental_correction_payload",exclusions=exclusions,
        calibration=report["runs"][0]["real_fit_ids"],ego_calibration=cohorts["ego"]["ids"].tolist(),methods={})
    native_metrics={}
    for method in ["visual_anchor","scene_transport_robot_orientation"]:
        rows=[];speeds=[];acc=[];angular=[]
        for i,rec in enumerate(cohorts["ego"]["records"]):
            phase=rec["phase"]
            p=CubicSpline(GRID,z[method+"_p"][i],bc_type="natural")(phase)
            r=Slerp(GRID,Rotation.from_matrix(z[method+"_r"][i]))(phase).as_matrix()
            times=np.asarray(rec["raw"]["t"])
            v=np.gradient(p,times,axis=0)
            a=np.gradient(v,times,axis=0)
            q=Rotation.from_matrix(r)
            w=(q[1:]*q[:-1].inv()).as_rotvec()/np.diff(times)[:,None]
            valid=np.array(rec["raw"]["valid"])
            speeds.extend(np.linalg.norm(v[valid],axis=1));acc.extend(np.linalg.norm(a[valid],axis=1))
            angular.extend(np.linalg.norm(w[valid[1:]&valid[:-1]],axis=1))
            rows.append(dict(id=rec["id"],source=rec["raw"]["path"],source_sha256=rec["raw"]["sha256"],
                             p=p.tolist(),r=r.tolist(),v=v.tolist(),w=np.vstack([np.zeros((1,3)),w]).tolist()))
        payload["methods"][method]=rows
        native_metrics[method]=dict(frames=sum(len(row["p"]) for row in rows),
            speed_p95_m_s=float(np.quantile(speeds,.95)),acceleration_p95_m_s2=float(np.quantile(acc,.95)),
            angular_speed_p95_rad_s=float(np.quantile(angular,.95)),speed_over_06_fraction=float(np.mean(np.array(speeds)>.6)))
    (args.experiment/"native_payload.json.gz").write_bytes(gzip.compress(json.dumps(payload,allow_nan=False).encode()))
    (args.experiment/"native_metrics.json").write_text(json.dumps(dict(native=native_metrics,
        conditional_trajectory_residual_rms_m=residual),indent=2))
    print(json.dumps(native_metrics,indent=2))


if __name__=="__main__": main()

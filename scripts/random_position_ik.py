"""Offline MuJoCo joint-bounded IK diagnostic; never connects to a robot.

Solver failure is not a proof of infeasibility. No collision/dynamics guarantee.
Checks fixed 20 episodes x 9 phases per method, plus a real FK reference cohort.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument('--methods',nargs='+',default=['real','raw','anchor_shift','visual_anchor','scene_transport_robot_orientation'])
    parser.add_argument('--heldout',action='store_true',help='Sample up to 20 fixed held-out episodes from predictions')
    args=parser.parse_args()
    z=np.load(args.predictions)
    model=mujoco.MjModel.from_xml_path(str(args.scene));data=mujoco.MjData(model)
    ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f"right_joint{i}") for i in range(11,17)])
    if np.any(ids<0):raise ValueError("missing right arm joints")
    qadr=model.jnt_qposadr[ids];dadr=model.jnt_dofadr[ids];bounds=model.jnt_range[ids]
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"right_tcp")
    jacp=np.zeros((3,model.nv));jacr=np.zeros((3,model.nv))
    flange=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'right_flange')
    data.qpos[qadr]=0;mujoco.mj_forward(model,data)
    offset=data.site_xpos[flange].copy()
    frame_rotation=data.site_xmat[flange].reshape(3,3).copy()
    seeds=[np.array([0.,1.2,1.5,0.,0.,0.]),np.array([-.5,.7,1.8,0.,.4,0.]),np.array([.5,.7,1.8,0.,.4,0.])]

    def solve(p,r,q):
        target=frame_rotation@p+offset
        r=frame_rotation@r
        for _ in range(150):
            data.qpos[qadr]=q;mujoco.mj_forward(model,data)
            dp=target-data.site_xpos[site]
            dr=Rotation.from_matrix(r@data.site_xmat[site].reshape(3,3).T).as_rotvec()
            if np.linalg.norm(dp)<.002 and np.linalg.norm(dr)<np.deg2rad(2):break
            mujoco.mj_jacSite(model,data,jacp,jacr,site)
            j=np.vstack([jacp[:,dadr],.15*jacr[:,dadr]])
            error=np.r_[dp,.15*dr]
            delta=j.T@np.linalg.solve(j@j.T+1e-5*np.eye(6),error)
            delta*=min(1.,.2/max(np.max(np.abs(delta)),1e-12))
            q=np.clip(q+delta,bounds[:,0],bounds[:,1])
        data.qpos[qadr]=q;mujoco.mj_forward(model,data)
        pos=float(np.linalg.norm(target-data.site_xpos[site]))
        angle=float(np.degrees(Rotation.from_matrix(r@data.site_xmat[site].reshape(3,3).T).magnitude()))
        return q,pos,angle

    results={}
    candidates=np.flatnonzero(z['test_mask']) if args.heldout else np.arange(len(z['real_p']))
    episodes=candidates[np.linspace(0,len(candidates)-1,min(20,len(candidates))).astype(int)]
    for method in args.methods:
        rows=[]
        for episode in episodes:
            q=seeds[0].copy()
            for phase in range(0,81,10):
                p,r=z[method+"_p"][episode,phase],z[method+"_r"][episode,phase]
                best=solve(p,r,q.copy())
                if best[1]>.01 or best[2]>10:
                    for seed in seeds:
                        candidate=solve(p,r,seed.copy())
                        if candidate[1]+.002*candidate[2]<best[1]+.002*best[2]:best=candidate
                q,pos,angle=best
                rows.append(dict(episode_index=int(episode),pair_id=int(z["ids"][episode]) if "ids" in z else int(episode),phase=phase,position_m=pos,angle_deg=angle,
                                 solved=pos<=.01 and angle<=10))
        results[method]=dict(tested=len(rows),solved=sum(row["solved"] for row in rows),
            mean_position_m=float(np.mean([v["position_m"] for v in rows])),
            p95_position_m=float(np.quantile([v["position_m"] for v in rows],.95)),
            mean_angle_deg=float(np.mean([v["angle_deg"] for v in rows])),rows=rows)
        print(method,results[method]["solved"],len(rows),flush=True)
        args.out.write_text(json.dumps(dict(protocol="Up to 20 fixed episodes x 9 phases; bounded multi-start DLS; 1cm/10deg convergence; no collision check",heldout=args.heldout,results=results),indent=2))


if __name__=="__main__":main()

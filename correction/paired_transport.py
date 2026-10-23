"""Small-sample paired trajectory calibration with explicit fit/predict boundary.

All normalization, reference rotations, and regression coefficients are fitted
only on the supplied calibration pairs. Prediction never accepts target poses.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def episode_features(p, r, scene, mode):
    if mode == 'bias':
        return np.empty((len(p),0))
    if mode == 'scene':
        return np.asarray(scene)
    anchors=p[:,[40,60],:].reshape(len(p),-1)
    orientation=r[:,40,:,:2].reshape(len(p),-1)
    if mode == 'eef':
        return np.column_stack([anchors,orientation])
    if mode == 'hybrid':
        return np.column_stack([scene,anchors,orientation])
    raise ValueError(mode)


class PairedCalibrator:
    def __init__(self, features='hybrid', ridge=10., retain=.5, smooth=1., basis='linear'):
        self.feature_mode=features
        self.ridge=float(ridge)
        self.retain=float(retain)
        self.smooth=float(smooth)
        if basis not in ['linear','rbf']:raise ValueError(basis)
        self.basis=basis

    def design(self,p,r,scene):
        f=episode_features(p,r,scene,self.feature_mode)
        f=(f-self.center)/self.scale
        if self.basis=='rbf':
            distance=np.sum((f[:,None,:]-self.landmarks[None,:,:])**2,axis=2)
            f=np.exp(-.5*distance/max(1,len(self.center)))
        return np.column_stack([np.ones(len(f)),f])

    def local(self,r):
        relative=np.einsum('tji,ntjk->ntik',self.reference,r)
        return Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(len(r),r.shape[1],3)

    def fit(self,ego_p,ego_r,scene,real_p,real_r):
        if ego_p.shape != real_p.shape or ego_p.shape[1:] != (81,3):
            raise ValueError('Expected paired Nx81x3 positions')
        if len(ego_p)<4 or not all(np.isfinite(x).all() for x in [ego_p,ego_r,scene,real_p,real_r]):
            raise ValueError('Need >=4 finite calibration pairs')
        f=episode_features(ego_p,ego_r,scene,self.feature_mode)
        self.center=f.mean(0);self.scale=np.maximum(f.std(0),1e-4)
        self.landmarks=(f-self.center)/self.scale
        x=self.design(ego_p,ego_r,scene)
        penalty=np.eye(x.shape[1])*self.ridge;penalty[0,0]=0
        operator=np.linalg.solve(x.T@x+penalty,x.T)
        self.reference=np.array([Rotation.from_matrix(real_r[:,t]).mean().as_matrix() for t in range(81)])
        for key,target,source in [('position',real_p,ego_p),('rotation',self.local(real_r),self.local(ego_r))]:
            coef=(operator@(target-self.retain*source).reshape(len(x),-1)).reshape(x.shape[1],81,3)
            if self.smooth>0:
                coef=gaussian_filter1d(coef,self.smooth,axis=1,mode='nearest')
            setattr(self,key+'_coef',coef)
        return self

    def predict(self,ego_p,ego_r,scene):
        x=self.design(ego_p,ego_r,scene)
        p=self.retain*ego_p+np.einsum('nf,ftd->ntd',x,self.position_coef)
        rv=self.retain*self.local(ego_r)+np.einsum('nf,ftd->ntd',x,self.rotation_coef)
        r=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(ego_r.shape)
        return p,np.einsum('tij,ntjk->ntik',self.reference,r)

    def artifact(self):
        return dict(schema='tate.paired_trajectory_calibration',schema_version=1,
            parameters=dict(features=self.feature_mode,ridge=self.ridge,retain=self.retain,smooth=self.smooth,basis=self.basis),
            **{k:getattr(self,k).tolist() for k in ['center','scale','landmarks','reference','position_coef','rotation_coef']})

    @classmethod
    def from_artifact(cls,data):
        if data.get('schema')!='tate.paired_trajectory_calibration' or data.get('schema_version')!=1:
            raise ValueError('Unsupported paired calibration artifact')
        model=cls(**data['parameters'])
        for key in ['center','scale','landmarks','reference','position_coef','rotation_coef']:
            setattr(model,key,np.asarray(data[key],dtype=float))
        return model

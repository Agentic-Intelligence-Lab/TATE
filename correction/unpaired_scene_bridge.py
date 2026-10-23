"""Unpaired scene-conditioned trajectory calibration.

fit accepts unequal, independently ordered domains; no correspondence labels.
Hyperparameters for domain means use within-domain leave-one-out regression.
The residual Gaussian map is an explicit minimum-displacement assumption, not
an estimate of the unknown paired source/target residual relationship.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from correction.scene_transport import gaussian_transport


class SceneBridge:
    def __init__(self,rank=6,outside_scale=.25,shrinkage=.3):
        self.rank=rank;self.outside_scale=outside_scale;self.shrinkage=shrinkage

    @staticmethod
    def domain_fit(scene,values):
        center=scene.mean(0);scale=np.maximum(scene.std(0),1e-4)
        x=np.column_stack([np.ones(len(scene)),(scene-center)/scale])
        y=values.reshape(len(values),-1)
        rows=[]
        # Exact linear-smoother leave-one-out errors. Feature scaling uses domain
        # covariates only; no source/target correspondence or other domain labels.
        for ridge in [.01,.1,1.,10.,100.]:
            operator=np.linalg.solve(x.T@x+np.diag([0.,ridge,ridge]),x.T)
            hat=x@operator
            residual=(y-hat@y)/np.maximum(1-np.diag(hat)[:,None],1e-8)
            rows.append((float(np.mean(residual**2)),ridge,operator@y))
        best=min(rows,key=lambda row:row[0])
        return dict(center=center,scale=scale,coef=best[2],ridge=best[1],loo=[(v[1],v[0]) for v in rows])

    @staticmethod
    def mean(model,scene):
        return np.column_stack([np.ones(len(scene)),(scene-model['center'])/model['scale']])@model['coef']

    def covariance(self,latent):
        # Three fitted mean coefficients; correct the residual degrees of freedom.
        cov=latent.T@latent/max(len(latent)-3,1)
        avg=max(np.trace(cov)/len(cov),1e-10)
        return (1-self.shrinkage)*cov+self.shrinkage*avg*np.eye(len(cov))

    def fit_component(self,source,source_scene,target,target_scene):
        sm=self.domain_fit(source_scene,source);tm=self.domain_fit(target_scene,target)
        es=source.reshape(len(source),-1)-self.mean(sm,source_scene)
        et=target.reshape(len(target),-1)-self.mean(tm,target_scene)
        _,_,vt=np.linalg.svd(np.concatenate([es/np.sqrt(len(es)),et/np.sqrt(len(et))]),full_matrices=False)
        k=min(self.rank,len(source)-3,len(target)-3)
        basis=vt[:k].T
        transport=gaussian_transport(self.covariance(es@basis),self.covariance(et@basis))
        eig,vec=np.linalg.eigh(transport)
        transport=(vec*np.clip(eig,.25,2.))@vec.T
        return dict(source=sm,target=tm,basis=basis,transport=transport,eigenvalues=eig,shape=source.shape[1:])

    def local(self,r):
        rel=np.einsum('tji,ntjk->ntik',self.reference,r)
        return Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec().reshape(len(r),r.shape[1],3)

    def fit(self,source_p,source_r,source_scene,target_p,target_r,target_scene):
        if min(len(source_p),len(target_p))<6:raise ValueError('At least six records per domain required')
        if not all(np.isfinite(v).all() for v in [source_p,source_r,source_scene,target_p,target_r,target_scene]):raise ValueError('Non-finite input')
        self.reference=np.array([Rotation.from_matrix(target_r[:,t]).mean().as_matrix() for t in range(target_r.shape[1])])
        self.position=self.fit_component(source_p,source_scene,target_p,target_scene)
        self.orientation=self.fit_component(self.local(source_r),source_scene,self.local(target_r),target_scene)
        return self

    def component_predict(self,model,source,scene,mode):
        mean=self.mean(model['target'],scene)
        residual=source.reshape(len(source),-1)-self.mean(model['source'],scene)
        basis=model['basis'];latent=residual@basis
        if mode=='transport':
            out=mean+latent@model['transport'].T@basis.T+self.outside_scale*(residual-latent@basis.T)
        elif mode=='residual':out=mean+residual
        elif mode=='mean':out=mean
        else:raise ValueError(mode)
        return out.reshape(source.shape)

    def predict(self,p,r,scene,mode='transport'):
        pp=self.component_predict(self.position,p,scene,mode)
        rv=self.component_predict(self.orientation,self.local(r),scene,mode)
        rr=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(r.shape)
        return pp,np.einsum('tij,ntjk->ntik',self.reference,rr)

    def artifact(self):
        def serial(x):
            if isinstance(x,np.ndarray):return x.tolist()
            if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)):return [serial(v) for v in x]
            return x
        return dict(schema='tate.unpaired_scene_bridge',version=1,parameters=dict(rank=self.rank,outside_scale=self.outside_scale,shrinkage=self.shrinkage),
                    reference=self.reference.tolist(),position=serial(self.position),orientation=serial(self.orientation))

    @classmethod
    def from_artifact(cls,data):
        if data['schema']!='tate.unpaired_scene_bridge':raise ValueError('Wrong model schema')
        model=cls(**data['parameters']);model.reference=np.array(data['reference'])
        for name in ['position','orientation']:
            component=dict(data[name])
            for domain in ['source','target']:
                component[domain]=dict(component[domain])
                for key in ['center','scale','coef']:component[domain][key]=np.array(component[domain][key])
            for key in ['basis','transport','eigenvalues']:component[key]=np.array(component[key])
            setattr(model,name,component)
        return model

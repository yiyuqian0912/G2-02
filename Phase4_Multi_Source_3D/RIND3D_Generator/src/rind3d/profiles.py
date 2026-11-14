"""Versioned publication contracts; the full v2 profile is unchanged."""
from dataclasses import dataclass
from .window_contract import WINDOW_SIDES,PRODUCTION_SIDE_COUNTS,balanced_quotas

FULL='rind3d-full-v2'
LITE='rind3d-lite-v0.1'

@dataclass(frozen=True)
class Profile:
    name:str
    windows:int
    max_candidates:int
    ready_state:str
    @property
    def quotas(self):return dict(PRODUCTION_SIDE_COUNTS) if self.name==FULL else balanced_quotas(256)

PROFILES={FULL:Profile(FULL,4096,64,'READY'),LITE:Profile(LITE,256,8,'READY_LITE')}

def profile_for(meta):
    name=meta.get('dataset_profile',FULL)
    if name not in PROFILES:raise ValueError('Unsupported dataset profile: '+str(name))
    return PROFILES[name]

def validate_manifest_profile(manifest):
    profile=profile_for(manifest)
    if profile.name==LITE:
        required={'dataset_profile':LITE,'profile_version':'0.1','window_sides':list(WINDOW_SIDES),
                  'production_windows':256,'max_candidates':8,'split_group':'geometry_id'}
        for key,value in required.items():
            if manifest.get(key)!=value:raise ValueError('Lite manifest contract mismatch: '+key)
        if manifest.get('response_storage')!='RINDPack-v1':raise ValueError('Lite requires RINDPack-v1')
    return profile

"""Independent default-off target SiLU producer policy, frozen at startup."""
from dataclasses import dataclass
import hashlib,json
FLAG='PAITON_TARGET_SILU_PREFILL'
VERSION='target-silu-prefill-v1'
@dataclass(frozen=True)
class Policy:
    enabled: bool
    artifact_sha256: str
    version: str=VERSION
    @classmethod
    def from_environment(cls,env,manifest):
        value=env.get(FLAG,'0')
        if value not in ('0','1','off','on','false','true'):raise ValueError('Expected explicit SiLU boolean')
        if (manifest.get('version'),manifest.get('arch'),manifest.get('shape_mn'),manifest.get('variant'),manifest.get('abi'),manifest.get('layout'))!=(VERSION,'gfx1201',[4089,17408],0,1,'fragment16_fp8'):raise ValueError('Unsupported SiLU contract')
        digest=manifest.get('sha256','')
        if len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):raise ValueError('Expected SHA256 artifact identity')
        return cls(value in ('1','on','true'),digest)
    def allows(self,m,n,*,arch,tiled):
        return self.enabled and (m,n)==(4089,17408) and arch=='gfx1201' and tiled==1
    def snapshot(self):return {'version':self.version,'enabled':self.enabled,'artifact_sha256':self.artifact_sha256,'flag':FLAG,'shape_mn':[4089,17408],'layout':'fragment16_fp8','variant':0,'dflash_policy_changed':False,'apc_changed':False}
    @property
    def cache_key(self):return hashlib.sha256(json.dumps(self.snapshot(),sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]

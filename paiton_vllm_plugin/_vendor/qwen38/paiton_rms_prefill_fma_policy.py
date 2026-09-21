"""Independent default-off RMS producer policy, frozen at startup."""
from dataclasses import dataclass
import hashlib
import json

FLAG = 'PAITON_TARGET_RMS_PREFILL'
VERSION = 'target-rms-prefill-v2'

@dataclass(frozen=True)
class Policy:
    enabled: bool
    artifact_sha256: str
    version: str = VERSION

    @classmethod
    def from_environment(cls, env, manifest):
        value = env.get(FLAG, '0')
        if value not in ('0', '1', 'off', 'on', 'false', 'true'):
            raise ValueError('Expected explicit RMS boolean')
        contract = tuple(manifest.get(k) for k in ('version', 'arch', 'shape_mn', 'reduction_threads', 'abi', 'layout'))
        if contract != (VERSION, 'gfx1201', [4089, 5120], 512, 1, 'fragment16_fp8'):
            raise ValueError('Unsupported RMS contract')
        digest = manifest.get('sha256', '')
        if len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('Expected SHA256 artifact identity')
        return cls(value in ('1', 'on', 'true'), digest)

    def allows(self, m, n, *, arch, tiled, eps):
        return (self.enabled and (m, n) == (4089, 5120) and arch == 'gfx1201'
                and tiled == 1 and eps in (1.e-6, 1.e-5))

    def snapshot(self):
        return {'version': self.version, 'enabled': self.enabled,
                'artifact_sha256': self.artifact_sha256, 'flag': FLAG,
                'shape_mn': [4089, 5120], 'layout': 'fragment16_fp8',
                'reduction_threads': 512, 'epsilons': [1.e-6, 1.e-5],
                'dflash_policy_changed': False, 'apc_changed': False}

    @property
    def cache_key(self):
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest()[:20]

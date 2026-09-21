"""Independent default-off GDN producer policy, frozen at startup."""
from dataclasses import dataclass
import hashlib
import json

FLAG = 'PAITON_TARGET_GDN_NORM_FIXED'
VERSION = 'target-gdn-norm-fixed-v2'

@dataclass(frozen=True)
class Policy:
    enabled: bool
    artifact_sha256: str
    version: str = VERSION

    @classmethod
    def from_environment(cls, env, manifest):
        value = env.get(FLAG, '0')
        if value not in ('0', '1', 'off', 'on', 'false', 'true'):
            raise ValueError('Expected explicit GDN boolean')
        contract = tuple(manifest.get(k) for k in ('version', 'arch', 'shape_mn', 'reduction_threads', 'abi', 'layout', 'head_width', 'gate_row_stride'))
        if contract != (VERSION, 'gfx1201', [[64, 6144], [4089, 6144]], 256, 1, 'row_major_fp8', 128, 6144):
            raise ValueError('Unsupported GDN contract')
        digest = manifest.get('sha256', '')
        if len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('Expected SHA256 artifact identity')
        return cls(value in ('1', 'on', 'true'), digest)

    def allows(self, m, n, *, arch, z_stride, eps):
        return (self.enabled and m in (64, 4089) and n == 6144 and arch == 'gfx1201'
                and z_stride == 6144 and eps in (1.e-6, 1.e-5))

    def snapshot(self):
        return {'version': self.version, 'enabled': self.enabled,
                'artifact_sha256': self.artifact_sha256, 'flag': FLAG,
                'shape_mn': [[64, 6144], [4089, 6144]], 'layout': 'row_major_fp8',
                'reduction_threads': 256, 'head_width': 128, 'gate_row_stride': 6144, 'epsilons': [1.e-6, 1.e-5],
                'dflash_policy_changed': False, 'apc_changed': False}

    @property
    def cache_key(self):
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest()[:20]

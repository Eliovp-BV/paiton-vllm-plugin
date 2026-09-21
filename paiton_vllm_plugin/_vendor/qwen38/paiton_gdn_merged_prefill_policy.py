"""Immutable, opt-in target-prefill policy. Independent of DFlash and APC."""
from dataclasses import dataclass
import hashlib
import json

FLAG = 'PAITON_MXFP4_GDN_MERGED_PREFILL'
VERSION = 'mxfp4-gdn-merged-prefill-v1'

@dataclass(frozen=True)
class Policy:
    enabled: bool
    artifact_sha256: str
    version: str = VERSION

    @classmethod
    def from_environment(cls, env, manifest):
        value = env.get(FLAG, '0')
        if value not in ('0', '1', 'off', 'on', 'false', 'true'):
            raise ValueError(f'{FLAG} requires an explicit boolean')
        if (manifest.get('version'), manifest.get('arch'), manifest.get('shape_mnk'),
            manifest.get('group_m'), manifest.get('abi')) != (VERSION, 'gfx1201', [4089, 16480, 5120], 4, 1):
            raise ValueError('Unsupported target-prefill artifact contract')
        digest = manifest.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Expected SHA256 artifact identity')
        return cls(value in ('1', 'on', 'true'), digest)

    def allows(self, m, n, k, *, arch, layout, partitions=1):
        return (self.enabled and (m, n, k) == (4089, 16480, 5120)
                and arch == 'gfx1201' and layout == 'fragment16_fp8' and partitions == 1)

    def snapshot(self):
        return {'version': self.version, 'enabled': self.enabled,
                'artifact_sha256': self.artifact_sha256, 'flag': FLAG,
                'shape_mnk': [4089, 16480, 5120], 'partitions': 1, 'tile_mnk': [512,64,64],
                'dflash_policy_changed': False, 'apc_changed': False}

    @property
    def cache_key(self):
        data = json.dumps(self.snapshot(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(data).hexdigest()[:20]

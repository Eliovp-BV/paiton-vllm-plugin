"""Opt-in native context fusion; only M8 has independently confirmed timing."""
from dataclasses import dataclass
import math

FLAG = 'PAITON_DFLASH_CONTEXT_NORM_ROPE'
VERSION = 'dflash-context-v1'

@dataclass(frozen=True)
class Policy:
    enabled: bool
    artifact_sha256: str

    @classmethod
    def from_environment(cls, env, manifest):
        value = env.get(FLAG, '0')
        if value not in ('0', '1', 'off', 'on', 'false', 'true'):
            raise ValueError('Context fusion requires an explicit boolean')
        if (manifest.get('version'), manifest.get('arch'), manifest.get('abi'),
                manifest.get('measured_rows')) != (VERSION, 'gfx1201', 1, [8]):
            raise ValueError('Unsupported context artifact contract')
        digest = manifest.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Expected SHA256 artifact identity')
        return cls(value in ('1', 'on', 'true'), digest)

    def allows(self, m, *, arch, layers, heads, dim, eps, neox, shadow=False):
        return (self.enabled and arch == 'gfx1201' and (layers, heads, dim) == (5, 8, 128)
                and neox is True and eps in (1.e-6, 1.e-5) and math.isfinite(eps)
                and (1 <= m <= 4096 if shadow else m == 8))

    def snapshot(self):
        return {'version': VERSION, 'enabled': self.enabled, 'flag': FLAG,
                'artifact_sha256': self.artifact_sha256, 'measured_rows': [8],
                'default_enabled': False, 'model_promoted': False}

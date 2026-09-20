"""Startup-only native DFlash controls; no GPU, framework or network imports."""
from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

FEATURE_FLAGS = {
    'down_projection': 'PAITON_DFLASH_DOWN_PROJECTION',
    'gate_projection': 'PAITON_DFLASH_GATE_PROJECTION',
    'silu_quant': 'PAITON_DFLASH_SILU_QUANT',
    'context_kv': 'PAITON_DFLASH_CONTEXT_KV',
}
MASTER_FLAG = 'PAITON_DFLASH_NATIVE'


def _boolean(value: str, name: str) -> bool:
    if value in ('1', 'on', 'true'): return True
    if value in ('0', 'off', 'false'): return False
    raise ValueError(f'{name} must be 0/1, off/on or false/true, got {value!r}')


@dataclass(frozen=True)
class Policy:
    master: bool
    enabled_features: tuple[str, ...]
    requested_flags: tuple[tuple[str, bool], ...]
    artifact_sha256: str
    contract_version: int = 1

    @classmethod
    def from_environment(cls, env: Mapping[str, str], *, available_features, artifact_sha256: str):
        """Snapshot explicit settings before model load/capture; never reload at replay.

        Master defaults OFF. Each built feature defaults ON behind the master;
        an explicit component OFF overrides it. Enabling an unavailable feature
        fails at startup. Nothing here changes speculative decoding itself.
        """
        available = frozenset(available_features)
        if not available <= FEATURE_FLAGS.keys():
            raise ValueError('Unknown feature in artifact manifest')
        if len(artifact_sha256) != 64 or any(c not in '0123456789abcdef' for c in artifact_sha256):
            raise ValueError('Expected lowercase SHA256 artifact identity')
        master = _boolean(env.get(MASTER_FLAG, '0'), MASTER_FLAG)
        flags = [(MASTER_FLAG, master)]
        enabled = []
        for feature, name in FEATURE_FLAGS.items():
            requested = _boolean(env.get(name, '1' if feature in available else '0'), name)
            flags.append((name, requested))
            if master and requested:
                if feature not in available:
                    raise ValueError(f'{name} is enabled but this artifact does not provide {feature}')
                enabled.append(feature)
        return cls(master, tuple(enabled), tuple(flags), artifact_sha256)

    def allows(self, feature: str) -> bool:
        if feature not in FEATURE_FLAGS: raise ValueError('Unknown DFlash feature')
        return feature in self.enabled_features

    def snapshot(self):
        return {'contract_version': self.contract_version,
                'requested_flags': dict(self.requested_flags),
                'enabled_features': list(self.enabled_features),
                'artifact_sha256': self.artifact_sha256,
                'speculative_decoding_changed': False,
                'scope': 'Startup snapshot; existing shape/dtype/layout/model guards remain mandatory'}

    @property
    def cache_key(self) -> str:
        data = json.dumps(self.snapshot(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(data).hexdigest()[:20]

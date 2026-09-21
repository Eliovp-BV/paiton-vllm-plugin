import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from paiton_vllm_plugin import paiton_platform_plugin, register_paiton_models
from paiton_vllm_plugin.existing_env import activate


class ExistingEnvironmentTests(unittest.TestCase):
    def test_installation_does_not_probe_gpu_or_import_vllm(self):
        script = '''
import sys
from types import SimpleNamespace
from paiton_vllm_plugin import paiton_platform_plugin, register_paiton_models
assert paiton_platform_plugin() is None
register_paiton_models()
assert not any(k.split('.')[0] in {'torch', 'vllm', 'triton'} for k in sys.modules)
'''
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PAITON_', 'VLLM_'))}
        subprocess.run([sys.executable, '-c', script], env=env, check=True)

    def test_models_mode_preserves_upstream_platform(self):
        with patch.dict(os.environ, {'PAITON_PLUGIN_MODE': 'models', 'VLLM_USE_PAITON_PLATFORM': '1'}):
            self.assertIsNone(paiton_platform_plugin())

    def test_off_conflicts_with_legacy_activation(self):
        with patch.dict(os.environ, {'PAITON_PLUGIN_MODE': 'off', 'VLLM_USE_PAITON_PLATFORM': '1'}):
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                paiton_platform_plugin()
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                register_paiton_models()

    def test_activation_preserves_other_plugins(self):
        env = {'VLLM_PLUGINS': 'other_hook,register_paiton_models'}
        activate('models', env)
        self.assertEqual(env['VLLM_PLUGINS'], 'other_hook,register_paiton_models')
        self.assertEqual(env['PAITON_PLUGIN_MODE'], 'models')

    def test_dflash_mode_preserves_explicit_runtime_adapter(self):
        hook = Mock()
        with patch.dict(os.environ, {'PAITON_PLUGIN_MODE': 'dflash', 'PAITON_RUNTIME_COMPAT_FLOW': '1'}), \
                patch.dict(sys.modules, {'paiton_runtime_compat': SimpleNamespace(register=hook)}), \
                patch('paiton_vllm_plugin.dflash.install'):
            register_paiton_models()
        hook.assert_called_once_with()

    def test_bad_mode_fails_before_loading_framework(self):
        with patch.dict(os.environ, {'PAITON_PLUGIN_MODE': 'typo'}):
            with self.assertRaises(ValueError):
                register_paiton_models()


if __name__ == '__main__':
    unittest.main()

"""CPU-only contracts for the immutable MXFP4 release's public launchers."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODEL_DIR = Path(__file__).resolve().parents[1] / 'models/Qwen3.8-MXFP4-DFlash2'


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load_script('qwen38_mxfp4_entrypoint_test', MODEL_DIR / 'container_entrypoint.py')
with patch.dict(sys.modules, {'container_entrypoint': entry}):
    host = load_script('qwen38_mxfp4_host_test', MODEL_DIR / 'serve.py')


def value(command, flag):
    return command[command.index(flag) + 1]


class Mxfp4LauncherTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((MODEL_DIR / 'engine-profile.json').read_text())
        self.original_profile = copy.deepcopy(self.profile)
        self.image = 'ghcr.io/eliovp/paiton-vllm-plugin@sha256:' + 'a' * 64

    def host_args(self, **overrides):
        defaults = dict(name='paiton-qwen38-mxfp4', download_only=False, detach=False,
                        cache=None, port=8000, target=None, draft=None, offline=False)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_engine_default_argv_matches_historical_profile(self):
        expected = list(self.profile['arguments'])
        for flag in ('--model', '--tokenizer'):
            expected[expected.index(flag) + 1] = '/verified/target'
        spec_index = expected.index('--speculative-config') + 1
        draft = json.loads(expected[spec_index])
        draft['model'] = '/verified/draft'
        expected[spec_index] = json.dumps(draft)
        command = entry.engine_command(self.profile, Path('/verified/target'), Path('/verified/draft'))
        self.assertEqual(command, [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', *expected])
        self.assertEqual(value(command, '--max-model-len'), '8192')
        self.assertEqual(value(command, '--max-num-seqs'), '8')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '5368709120')
        self.assertEqual(value(command, '--tool-call-parser'), 'hermes')
        self.assertNotIn('--reasoning-parser', command)
        self.assertNotIn('--default-chat-template-kwargs', command)
        self.assertEqual(self.profile, self.original_profile)

    def test_default_docker_command_has_no_release_overlay(self):
        self.assertEqual(host.command(self.host_args(), self.image), [
            'docker', 'run', '--rm', '--name', 'paiton-qwen38-mxfp4',
            '--device', '/dev/kfd', '--device', '/dev/dri', '--group-add', 'video',
            '--shm-size', '2g', '-p', '127.0.0.1:8000:8000',
            '-v', 'paiton-qwen38-mxfp4-cache:/models/cache', self.image])

    def test_context_override_synchronizes_target_and_draft(self):
        for context in (16384, 32768, 65536):
            with self.subTest(context=context):
                command = entry.engine_command(self.profile, '/target', '/draft', max_model_len=context)
                self.assertEqual(value(command, '--max-model-len'), str(context))
                draft = json.loads(value(command, '--speculative-config'))
                self.assertEqual(draft['max_model_len'], context)
                self.assertEqual(draft['model'], '/draft')
                self.assertEqual(draft['num_speculative_tokens'], 7)
                self.assertEqual(value(command, '--max-num-batched-tokens'), '4096')
                self.assertEqual(value(command, '--kv-cache-memory-bytes'), '5368709120')
        self.assertEqual(self.profile, self.original_profile)

    def test_resource_and_parser_overrides_preserve_other_settings(self):
        command = entry.engine_command(self.profile, '/target', '/draft', max_num_seqs=2,
                                       kv_cache_memory_bytes=6442450944,
                                       tool_call_parser='qwen3_xml', reasoning_parser='qwen3')
        self.assertEqual(value(command, '--max-num-seqs'), '2')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '6442450944')
        self.assertEqual(value(command, '--max-model-len'), '8192')
        self.assertEqual(json.loads(value(command, '--speculative-config'))['max_model_len'], 8192)
        self.assertEqual(value(command, '--tool-call-parser'), 'qwen3_xml')
        self.assertEqual(value(command, '--reasoning-parser'), 'qwen3')
        self.assertEqual(command.count('--tool-call-parser'), 1)
        self.assertEqual(command.count('--reasoning-parser'), 1)
        self.assertEqual(value(command, '--compilation-config'),
                         value(self.original_profile['arguments'], '--compilation-config'))
        self.assertEqual(self.profile, self.original_profile)

    def test_each_override_mounts_updated_entrypoint_into_old_image(self):
        cases = {'max_model_len': 32768, 'max_num_seqs': 1,
                 'kv_cache_memory_bytes': 5368709120, 'tool_call_parser': 'qwen3_xml',
                 'reasoning_parser': 'qwen3', 'disable_thinking': True}
        overlay = f'{MODEL_DIR / "container_entrypoint.py"}:/opt/paiton-release/container_entrypoint.py:ro'
        for name, setting in cases.items():
            with self.subTest(flag=name):
                command = host.command(self.host_args(**{name: setting}), self.image)
                image_index = command.index(self.image)
                self.assertIn(overlay, command[:image_index])
                expected = ['--' + name.replace('_', '-')]
                if name != 'disable_thinking':
                    expected.append(str(setting))
                self.assertEqual(command[image_index + 1:], expected)
                release_mounts = [item for item in command if ':/opt/paiton-release/' in item]
                self.assertEqual(release_mounts, [overlay])

    def test_cli_dry_run_exercises_shared_parser_and_immutable_image_overlay(self):
        process = subprocess.run([sys.executable, str(MODEL_DIR / 'serve.py'), '--dry-run',
                                  '--max-model-len', '32768', '--max-num-seqs', '1',
                                  '--kv-cache-memory-bytes', '5368709120',
                                  '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3',
                                  '--disable-thinking'],
                                 capture_output=True, text=True, check=True)
        command = json.loads(process.stdout)
        runtime = json.loads((MODEL_DIR / 'runtime.lock.json').read_text())
        image = runtime.get('registry_image') or runtime['planned_registry_image']
        self.assertIn(image, command)
        forwarded = command[command.index(image) + 1:]
        self.assertEqual(value(forwarded, '--max-model-len'), '32768')
        self.assertEqual(value(forwarded, '--reasoning-parser'), 'qwen3')
        self.assertEqual(forwarded[-1], '--disable-thinking')
        self.assertIn(f'{MODEL_DIR / "container_entrypoint.py"}:/opt/paiton-release/container_entrypoint.py:ro', command)

    def test_invalid_overrides_fail_before_download_or_docker(self):
        cases = [('--max-model-len', '0'), ('--max-model-len', '-1'),
                 ('--max-num-seqs', '0'), ('--max-num-seqs', '9'),
                 ('--kv-cache-memory-bytes', '0'), ('--kv-cache-memory-bytes', '-1'),
                 ('--kv-cache-memory-bytes', '5GiB'), ('--kv-cache-memory-bytes', '1.5'),
                 ('--tool-call-parser', 'unknown'), ('--reasoning-parser', 'unknown')]
        for script in ('serve.py', 'container_entrypoint.py'):
            for flag, setting in cases:
                with self.subTest(script=script, flag=flag, setting=setting):
                    process = subprocess.run([sys.executable, str(MODEL_DIR / script), flag, setting],
                                             capture_output=True, text=True)
                    self.assertEqual(process.returncode, 2)
                    self.assertIn('error:', process.stderr)
                    self.assertNotIn('Verifying', process.stdout)

    def test_direct_engine_builder_rejects_invalid_limits(self):
        for overrides in ({'max_model_len': 0}, {'max_num_seqs': 9},
                          {'kv_cache_memory_bytes': -1}, {'max_model_len': True},
                          {'tool_call_parser': 'unknown'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                entry.engine_command(self.profile, '/target', '/draft', **overrides)

    def test_entrypoint_logs_experimental_effective_limits_and_executes(self):
        with tempfile.TemporaryDirectory() as cache:
            argv = ['entrypoint', '--cache', cache, '--max-model-len', '32768',
                    '--max-num-seqs', '1', '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3']
            output = io.StringIO()
            with patch.object(sys, 'argv', argv), \
                 patch.object(entry, 'resolve_snapshot', side_effect=[Path('/target'), Path('/draft')]), \
                 patch.object(entry.os, 'execvpe') as execute, contextlib.redirect_stdout(output):
                entry.main()
            executable, command, environment = execute.call_args.args
            self.assertEqual(executable, sys.executable)
            self.assertEqual(value(command, '--max-model-len'), '32768')
            self.assertEqual(json.loads(value(command, '--speculative-config'))['max_model_len'], 32768)
            self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
            self.assertIn('EXPERIMENTAL', output.getvalue())
            self.assertIn('not a qualified profile', output.getvalue())
            self.assertIn('"draft_max_model_len": 32768', output.getvalue())
            self.assertIn('"kv_cache_memory_bytes": 5368709120', output.getvalue())

    def test_download_only_with_override_keeps_gpu_inaccessible(self):
        command = host.command(self.host_args(download_only=True, max_model_len=32768), self.image)
        self.assertNotIn('--device', command)
        self.assertNotIn('-p', command)
        self.assertIn('--download-only', command[command.index(self.image) + 1:])

    def test_disable_thinking_sets_only_server_default_and_preserves_other_kwargs(self):
        self.profile['arguments'] += ['--default-chat-template-kwargs', '{"other": "preserved"}']
        command = entry.engine_command(self.profile, '/target', '/draft', disable_thinking=True)
        self.assertEqual(json.loads(value(command, '--default-chat-template-kwargs')),
                         {'other': 'preserved', 'enable_thinking': False})
        self.assertEqual(command.count('--default-chat-template-kwargs'), 1)
        self.assertNotIn('--disable-thinking', command)
        self.assertEqual(json.loads(value(self.profile['arguments'], '--default-chat-template-kwargs')),
                         {'other': 'preserved'})

    def test_fresh_profile_disable_thinking_appends_valid_json(self):
        command = entry.engine_command(self.profile, '/target', '/draft', disable_thinking=True)
        self.assertEqual(json.loads(value(command, '--default-chat-template-kwargs')),
                         {'enable_thinking': False})
        self.assertEqual(self.profile, self.original_profile)

    def test_agentic_json_overlay_changes_only_parsers_and_thinking_default(self):
        profile = json.loads((MODEL_DIR / 'agentic-profile.json').read_text())
        self.assertEqual(profile['environment'], self.original_profile['environment'])
        self.assertNotIn('qualification_runs', profile)
        self.assertNotIn('benchmark_image', profile)
        expected = list(self.original_profile['arguments'])
        expected[expected.index('--tool-call-parser') + 1] = 'qwen3_xml'
        expected += ['--reasoning-parser', 'qwen3', '--default-chat-template-kwargs',
                     json.dumps({'enable_thinking': False})]
        self.assertEqual(profile['arguments'], expected)
        command = entry.engine_command(profile, '/target', '/draft')
        self.assertEqual(value(command, '--max-model-len'), '8192')
        self.assertEqual(json.loads(value(command, '--speculative-config'))['max_model_len'], 8192)
        self.assertEqual(value(command, '--max-num-seqs'), '8')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '5368709120')
        self.assertEqual(value(command, '--tool-call-parser'), 'qwen3_xml')
        self.assertEqual(value(command, '--reasoning-parser'), 'qwen3')

    def test_versioned_64k_profile_is_independent_of_historical_profile(self):
        profile = json.loads((MODEL_DIR / 'engine-profile-agentic-64k-v1.json').read_text())
        self.assertEqual(profile['profile'], 'agentic-64k-v1')
        self.assertEqual(profile['profile_revision'], 1)
        self.assertEqual(profile['environment'], self.original_profile['environment'])
        self.assertNotIn('qualification_runs', profile)
        self.assertIn('not full benchmark', profile['validation_scope'])
        command = entry.engine_command(profile, '/target', '/draft')
        settings = entry.effective_engine_settings(command)
        self.assertEqual(settings['max_model_len'], 65536)
        self.assertEqual(settings['draft_max_model_len'], 65536)
        self.assertEqual(settings['max_num_seqs'], 8)
        self.assertEqual(settings['kv_cache_memory_bytes'], 5368709120)
        self.assertEqual(settings['tool_call_parser'], 'qwen3_xml')
        self.assertEqual(settings['reasoning_parser'], 'qwen3')
        self.assertEqual(settings['default_chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(self.profile, self.original_profile)

    def test_named_profile_logs_effective_defaults_without_qualification_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'engine-profile.json').write_text(
                (MODEL_DIR / 'engine-profile-agentic-64k-v1.json').read_text())
            (base / 'checkpoint.lock.json').write_text((MODEL_DIR / 'checkpoint.lock.json').read_text())
            output = io.StringIO()
            with patch.object(entry, '__file__', str(base / 'container_entrypoint.py')), \
                 patch.object(sys, 'argv', ['entrypoint', '--cache', str(base / 'cache')]), \
                 patch.object(entry, 'resolve_snapshot', side_effect=[Path('/target'), Path('/draft')]), \
                 patch.object(entry.os, 'execvpe') as execute, contextlib.redirect_stdout(output):
                entry.main()
            command = execute.call_args.args[1]
            self.assertEqual(value(command, '--max-model-len'), '65536')
            message = output.getvalue()
            self.assertIn('"profile": "agentic-64k-v1"', message)
            self.assertIn('"draft_max_model_len": 65536', message)
            self.assertIn('"enable_thinking": false', message)
            self.assertNotIn('EXPERIMENTAL', message)
            self.assertNotIn('qualified Paiton profile', message)

    def test_versioned_200k_profile_changes_only_context_resources_and_scope(self):
        reference = json.loads((MODEL_DIR / 'engine-profile-agentic-64k-v1.json').read_text())
        profile = json.loads((MODEL_DIR / 'engine-profile-agentic-200k-v1.json').read_text())
        expected = copy.deepcopy(reference)
        args = expected['arguments']
        for flag, setting in (('--max-model-len', '200000'), ('--max-num-seqs', '1'),
                              ('--kv-cache-memory-bytes', '8589934592')):
            args[args.index(flag) + 1] = setting
        index = args.index('--speculative-config') + 1
        spec = json.loads(args[index])
        spec['max_model_len'] = 200000
        args[index] = json.dumps(spec)
        expected['profile'] = 'agentic-200k-v1'
        expected['validation_scope'] = profile['validation_scope']
        self.assertEqual(profile, expected)
        self.assertIn('One active request', profile['validation_scope'])
        self.assertIn('no full benchmark', profile['validation_scope'])
        settings = entry.effective_engine_settings(entry.engine_command(profile, '/target', '/draft'))
        self.assertEqual(settings['max_model_len'], 200000)
        self.assertEqual(settings['draft_max_model_len'], 200000)
        self.assertEqual(settings['max_num_seqs'], 1)
        self.assertEqual(settings['kv_cache_memory_bytes'], 8589934592)
        self.assertEqual(settings['tool_call_parser'], 'qwen3_xml')
        self.assertEqual(settings['reasoning_parser'], 'qwen3')
        self.assertEqual(settings['default_chat_template_kwargs'], {'enable_thinking': False})


if __name__ == '__main__':
    unittest.main()

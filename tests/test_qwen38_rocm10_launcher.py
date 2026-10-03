"""Exercise device isolation and Docker argument boundaries without a GPU runtime."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODEL_DIR = Path(__file__).resolve().parents[1] / 'models/Qwen3.8-MXFP4-DFlash2'
SCRIPT = MODEL_DIR / 'launch-rocm10.py'
SPEC = importlib.util.spec_from_file_location('rocm10_launcher', SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def value(command, flag):
    return command[command.index(flag) + 1]


class Rocm10LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.drm = self.root / 'drm'
        self.kfd = self.root / 'kfd'
        self.drm.mkdir()
        self.kfd.mkdir()
        self.record = self.root / 'docker-argv.json'
        binary = self.root / 'bin'
        binary.mkdir()
        # A real exec into a harmless Docker stand-in catches quoting, argument
        # positioning and accidental host-shell interpretation at the boundary.
        docker = binary / 'docker'
        docker.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                          'open(os.environ["DOCKER_ARGV_RECORD"],"w").write(json.dumps(sys.argv[1:]))\n')
        docker.chmod(0o755)
        self.environment = dict(os.environ)
        for name in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
            self.environment.pop(name, None)
        self.environment['PATH'] = str(binary) + os.pathsep + os.environ.get('PATH', '')
        self.environment['DOCKER_ARGV_RECORD'] = str(self.record)
        for name in ('TARGET', 'DRAFT', 'CACHE'):
            directory = self.root / (name.lower() + ' directory $(echo not-a-shell)')
            directory.mkdir()
            self.environment[f'PAITON_{name}_DIR'] = str(directory)
        self.device(128, 0x1002, 0x7551, 120001, 32 * 1024**3)
        self.device(129, 0x8086, 0x3e92, 0, 0)

    def device(self, minor, vendor, identifier, architecture, vram):
        path = self.drm / f'renderD{minor}' / 'device'
        path.mkdir(parents=True)
        for name, setting in (('vendor', hex(vendor)), ('device', hex(identifier)),
                              ('mem_info_vram_total', str(vram)),
                              ('uevent', f'PCI_SLOT_NAME=0000:{minor - 128:02x}:00.0')):
            (path / name).write_text(setting)
        node = self.kfd / str(minor)
        node.mkdir()
        (node / 'properties').write_text(f'drm_render_minor {minor}\ngfx_target_version {architecture}\n')

    def run_launcher(self, *args):
        harness = ('import importlib.util, pathlib, sys\n'
                   f's = importlib.util.spec_from_file_location("launch", {str(SCRIPT)!r})\n'
                   'm = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n'
                   f'm.SYS_DRM = pathlib.Path({str(self.drm)!r})\n'
                   f'm.SYS_KFD = pathlib.Path({str(self.kfd)!r})\n'
                   'sys.exit(m.main(sys.argv[1:]))\n')
        return subprocess.run([sys.executable, '-c', harness, *args],
                              env=self.environment, capture_output=True, text=True)

    def command(self, *args):
        result = self.run_launcher(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return ['docker', *json.loads(self.record.read_text())]

    def engine(self, command):
        index = next(i for i, item in enumerate(command) if item.startswith('ghcr.io/'))
        return command[index + 1:]

    def test_default_exec_preserves_release_limits_and_leaves_gpu_choice_to_user(self):
        command = self.command()
        self.assertEqual([command[i + 1] for i, item in enumerate(command) if item == '--device'],
                         ['/dev/kfd', '/dev/dri'])
        self.assertIn('ROCR_VISIBLE_DEVICES', command)
        self.assertIn('HIP_VISIBLE_DEVICES', command)
        self.assertIn('CUDA_VISIBLE_DEVICES', command)
        self.assertNotIn('ROCR_VISIBLE_DEVICES=0', command)
        self.assertNotIn('HIP_VISIBLE_DEVICES=0', command)
        self.assertEqual(value(command, '--group-add'), 'video')
        self.assertEqual(value(command, '--ipc'), 'host')
        self.assertNotIn('--shm-size', command)
        self.assertNotIn('-it', command)
        self.assertIn(launcher.IMAGES['65k'], command)
        engine = self.engine(command)
        self.assertEqual(engine[:2], ['serve', '/models/target'])
        self.assertEqual(value(engine, '--max-model-len'), '65536')
        self.assertEqual(value(engine, '--max-num-seqs'), '8')
        self.assertEqual(value(engine, '--max-num-batched-tokens'), '4096')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '6535819798')
        self.assertIn('--no-enable-prefix-caching', engine)
        self.assertEqual(value(engine, '--tool-call-parser'), 'qwen3_coder')
        self.assertNotIn('--default-chat-template-kwargs', engine)
        self.assertIn(self.environment['PAITON_CACHE_DIR'] + ':/cache:rw', command)

    def test_200k_context_override_synchronizes_draft_and_preserves_sampling(self):
        command = self.command('--release', '200k', '--context', '220000', '--port', '19000', '--name', 'custom-qwen')
        self.assertIn(launcher.IMAGES['200k'], command)
        self.assertEqual(value(command, '--name'), 'custom-qwen')
        engine = self.engine(command)
        self.assertEqual(value(engine, '--max-model-len'), '220000')
        self.assertEqual(json.loads(value(engine, '--speculative-config'))['max_model_len'], 220000)
        self.assertEqual(value(engine, '--max-num-seqs'), '1')
        self.assertEqual(value(engine, '--port'), '19000')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '6979321856')
        self.assertEqual(json.loads(value(engine, '--override-generation-config')),
                         {'temperature': 0.7, 'top_p': 0.95, 'top_k': 20})

    def test_desktop_uses_small_fixed_kv_and_explicit_overrides_take_precedence(self):
        engine = self.engine(self.command('--profile', 'desktop'))
        self.assertEqual(value(engine, '--max-model-len'), '32768')
        self.assertEqual(value(engine, '--max-num-seqs'), '1')
        self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.9')
        self.assertEqual(value(engine, '--max-num-batched-tokens'), '1024')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '2147483648')
        self.assertEqual(json.loads(value(engine, '--compilation-config'))['cudagraph_capture_sizes'], [1, 2, 4, 8])
        engine = self.engine(self.command('--profile', 'desktop', '--context', '16384',
                                          '--max-num-seqs', '2', '--gpu-memory-utilization', '0.85',
                                          '--max-num-batched-tokens', '2048',
                                          '--kv-cache-memory-bytes', '4294967296'))
        self.assertEqual(value(engine, '--max-model-len'), '16384')
        self.assertEqual(value(engine, '--max-num-seqs'), '2')
        self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.85')
        self.assertEqual(value(engine, '--max-num-batched-tokens'), '2048')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '4294967296')

    def test_desktop_automatic_kv_is_an_explicit_option(self):
        for options, expected_utilization in ((['--kv-cache-memory-bytes', 'auto'], '0.9'),
                                               (['--gpu-memory-utilization', '0.85'], '0.85')):
            with self.subTest(options=options):
                engine = self.engine(self.command('--profile', 'desktop', *options))
                self.assertNotIn('--kv-cache-memory-bytes', engine)
                self.assertEqual(value(engine, '--gpu-memory-utilization'), expected_utilization)
                self.assertEqual(value(engine, '--max-model-len'), '32768')

    def test_chat_defaults_and_220k_override_match_measured_runtime_configuration(self):
        for context in (200000, 220000):
            with self.subTest(context=context):
                options = [] if context == 200000 else ['--context', str(context)]
                command = self.command('--release', '200k', '--profile', 'chat', *options)
                image_index = command.index(launcher.IMAGES['200k'])
                environment = [command[i + 1] for i in range(image_index) if command[i] == '-e']
                self.assertCountEqual(environment, ['ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                                   'CUDA_VISIBLE_DEVICES',
                                                   'RADIANCE_GDN_LAZY=0',
                                                   'PYTORCH_ALLOC_CONF=max_split_size_mb:64'])
                engine = self.engine(command)
                self.assertEqual(value(engine, '--max-model-len'), str(context))
                self.assertEqual(value(engine, '--max-num-seqs'), '1')
                self.assertEqual(value(engine, '--max-num-batched-tokens'), '1024')
                self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '8589934592')
                self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.98')
                self.assertEqual(value(engine, '--mamba-cache-mode'), 'align')
                self.assertIn('--enable-prefix-caching', engine)
                self.assertNotIn('--no-enable-prefix-caching', engine)
                self.assertEqual(json.loads(value(engine, '--default-chat-template-kwargs')),
                                 {'enable_thinking': False})
                self.assertEqual(engine.count('--default-chat-template-kwargs'), 1)
                self.assertIn('--enable-prompt-tokens-details', engine)
                self.assertEqual(json.loads(value(engine, '--compilation-config')),
                                 {'cudagraph_capture_sizes': [1, 2, 4, 8],
                                  'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True}})
                self.assertEqual(json.loads(value(engine, '--speculative-config')),
                                 {'method': 'dflash', 'model': '/models/draft',
                                  'num_speculative_tokens': 7, 'draft_tensor_parallel_size': 1,
                                  'attention_backend': 'TRITON_ATTN', 'max_model_len': context,
                                  'disable_padded_drafter_batch': True, 'draft_sample_method': 'greedy'})
                self.assertEqual(value(engine, '--tool-call-parser'), 'qwen3_coder')
                self.assertEqual(json.loads(value(engine, '--override-generation-config')),
                                 {'temperature': 0.7, 'top_p': 0.95, 'top_k': 20})

    def test_explicit_flags_override_chat_defaults(self):
        command = self.command('--release', '200k', '--profile', 'chat',
                               '--context', '65536', '--max-num-seqs', '2',
                               '--max-num-batched-tokens', '2048', '--kv-cache-memory-bytes', '4294967296',
                               '--gpu-memory-utilization', '0.95', '--prefix-caching', 'off', '--thinking', 'on')
        engine = self.engine(command)
        self.assertEqual(value(engine, '--max-model-len'), '65536')
        self.assertEqual(value(engine, '--max-num-seqs'), '2')
        self.assertEqual(value(engine, '--max-num-batched-tokens'), '2048')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '4294967296')
        self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.95')
        self.assertIn('--no-enable-prefix-caching', engine)
        self.assertNotIn('--enable-prefix-caching', engine)
        self.assertNotIn('RADIANCE_GDN_LAZY=0', command)
        self.assertEqual(json.loads(value(engine, '--default-chat-template-kwargs')), {'enable_thinking': True})
        self.assertEqual(engine.count('--default-chat-template-kwargs'), 1)

    def test_chat_allocator_setting_does_not_leak_into_other_profiles(self):
        for profile in ('release', 'desktop'):
            command = self.command('--profile', profile)
            self.assertNotIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64', command)
            self.assertNotIn('--enable-prompt-tokens-details', command)

    def test_memory_fraction_override_is_effective_without_desktop_preset(self):
        engine = self.engine(self.command('--gpu-memory-utilization', '0.85'))
        self.assertNotIn('--kv-cache-memory-bytes', engine)
        self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.85')
        self.assertEqual(value(engine, '--max-model-len'), '65536')

    def test_prefix_caching_materializes_recurrent_state_before_vllm_entrypoint(self):
        command = self.command('--release', '200k', '--prefix-caching', 'on')
        image_index = command.index(launcher.IMAGES['200k'])
        self.assertIn('RADIANCE_GDN_LAZY=0', command[:image_index])
        engine = self.engine(command)
        self.assertIn('--enable-prefix-caching', engine)
        self.assertNotIn('--no-enable-prefix-caching', engine)
        self.assertEqual(value(engine, '--mamba-cache-mode'), 'align')

    def test_explicit_thinking_sets_server_default_without_changing_sampling_or_limits(self):
        for release in ('65k', '200k'):
            baseline = self.engine(self.command('--release', release))
            self.assertNotIn('--default-chat-template-kwargs', baseline)
            for mode, expected in (('on', True), ('off', False)):
                with self.subTest(release=release, thinking=mode):
                    command = self.command('--release', release, '--thinking', mode)
                    engine = self.engine(command)
                    self.assertEqual(engine[:-2], baseline)
                    self.assertEqual(engine[-2], '--default-chat-template-kwargs')
                    self.assertEqual(json.loads(engine[-1]), {'enable_thinking': expected})
                    self.assertEqual(engine.count('--default-chat-template-kwargs'), 1)

    def test_two_compatible_cards_are_allowed_without_automatic_selection(self):
        self.device(130, 0x1002, 0x7551, 120001, 32 * 1024**3)
        command = self.command()
        self.assertIn('/dev/dri', command)
        self.assertNotIn('/dev/dri/renderD130', command)
        self.assertNotIn('/dev/dri/renderD128', command)
        self.assertNotIn('/dev/dri/renderD129', command)
        self.assertIn('ROCR_VISIBLE_DEVICES', command)
        self.assertNotIn('ROCR_VISIBLE_DEVICES=0', command)

    def test_mixed_amd_generations_leave_selection_to_user_by_default(self):
        self.device(130, 0x1002, 0x73bf, 100300, 16 * 1024**3)
        command = self.command()
        self.assertIn('/dev/dri', command)
        self.assertNotIn('/dev/dri/renderD128', command)
        self.assertNotIn('/dev/dri/renderD130', command)

    def test_launch_does_not_require_kfd_architecture_metadata(self):
        properties = self.kfd / '128' / 'properties'
        for contents in ('drm_render_minor 128\n',
                         'drm_render_minor invalid\ngfx_target_version 120001\n',
                         'drm_render_minor 128\ngfx_target_version invalid\n', ''):
            with self.subTest(contents=contents):
                properties.write_text(contents)
                self.assertIn('/dev/dri', self.command())

    def test_list_and_dry_run_do_not_invoke_docker(self):
        result = self.run_launcher('--list-gpus')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('R9700 / gfx1201', result.stdout)
        self.assertIn('Intel (not supported', result.stdout)
        self.assertFalse(self.record.exists())
        result = self.run_launcher('--dry-run', '--context', '8192')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(value(json.loads(result.stdout), '--max-model-len'), '8192')
        self.assertFalse(self.record.exists())

    def test_inherited_host_masks_are_forwarded_exactly_including_empty_values(self):
        for name in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
            for setting in ('1', '0', '', 'GPU-1234567890abcdef', '1,0'):
                with self.subTest(name=name, setting=setting):
                    self.environment[name] = setting
                    command = self.command()
                    self.assertIn(name + '=' + setting, command)
                    self.assertNotIn(name, command)
                    del self.environment[name]

    def test_second_amd_render_device_preserves_user_rocr_and_hip_masks_without_uuid(self):
        # Replace the Intel fixture with another AMD GPU; KFD properties contain
        # no UUID, and their node names need not match GPU ordinals.
        path = self.drm / 'renderD129' / 'device'
        (path / 'vendor').write_text('0x1002')
        (path / 'device').write_text('0x7551')
        (path / 'mem_info_vram_total').write_text(str(32 * 1024**3))
        (self.kfd / '129' / 'properties').write_text('drm_render_minor 129\ngfx_target_version 120001\n')
        (self.kfd / '129').rename(self.kfd / '7')
        self.environment.update(ROCR_VISIBLE_DEVICES='1', HIP_VISIBLE_DEVICES='0')
        command = self.command()
        self.assertEqual([command[i + 1] for i, item in enumerate(command) if item == '--device'],
                         ['/dev/kfd', '/dev/dri'])
        self.assertIn('ROCR_VISIBLE_DEVICES=1', command)
        self.assertIn('HIP_VISIBLE_DEVICES=0', command)
        self.assertIn('CUDA_VISIBLE_DEVICES', command)
        self.assertFalse(any('GPU-' in item for item in command))

    def test_interactive_docker_flags_require_foreground_stdin_and_stdout_ttys(self):
        for detached in (False, True):
            for stdin_tty, stdout_tty in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(detached=detached, stdin_tty=stdin_tty, stdout_tty=stdout_tty):
                    args = launcher.parser().parse_args(['--detach'] if detached else [])
                    with patch.object(launcher.sys.stdin, 'isatty', return_value=stdin_tty), \
                         patch.object(launcher.sys.stdout, 'isatty', return_value=stdout_tty):
                        command = launcher.docker_command(args, self.environment)
                    self.assertEqual('-it' in command, not detached and stdin_tty and stdout_tty)
                    self.assertEqual('--detach' in command, detached)

    def test_bad_inputs_fail_before_any_docker_execution(self):
        cases = [('--context', '0'), ('--context', '262145'), ('--max-num-seqs', '9'),
                 ('--gpu-memory-utilization', 'nan'), ('--gpu-memory-utilization', '1'),
                 ('--gpu-memory-utilization', '-0.1'), ('--kv-cache-memory-bytes', '4GiB'),
                 ('--kv-cache-memory-bytes', '-1'), ('--port', '65536'),
                 ('--name', 'a b'), ('--gpu', '0.9'), ('--thinking', 'false'), ('--unknown',)]
        for case in cases:
            with self.subTest(case=case):
                result = self.run_launcher(*case)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.record.exists())

    def test_missing_mount_fails_before_docker_and_help_needs_no_mounts(self):
        self.environment['PAITON_TARGET_DIR'] = str(self.root / 'missing')
        result = self.run_launcher()
        self.assertEqual(result.returncode, 2)
        self.assertIn('not an existing directory', result.stderr)
        self.assertFalse(self.record.exists())
        for release in ('65k', '200k'):
            result = subprocess.run(['bash', str(MODEL_DIR / f'run-rocm10-{release}.sh'), '--help'],
                                    env=self.environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--profile', result.stdout)
            self.assertIn('--list-gpus', result.stdout)
            self.assertNotIn('--gpu ', result.stdout)

    def test_bash_wrappers_forward_arguments_without_shell_reinterpretation(self):
        python = self.root / 'bin' / 'python3'
        python.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                          'open(os.environ["DOCKER_ARGV_RECORD"],"w").write(json.dumps(sys.argv[1:]))\n')
        python.chmod(0o755)
        options = ['--profile', 'desktop', '--context', '16384', '--name', 'literal $value']
        # Wrappers select the current image and their defaults first; user options still override them.
        for script, preset in (('run-rocm10.sh', ['--name', 'paiton-qwen38']), ('run-rocm10-65k.sh', []),
                               ('run-rocm10-200k.sh', ['--profile', 'chat', '--name', 'paiton-qwen38-200k']),
                               ('run-mxfp4.sh', ['--name', 'paiton-qwen38', '--weights', 'mxfp4']),
                               ('run-3bit.sh', ['--name', 'paiton-qwen38', '--weights', 'w3a4'])):
            result = subprocess.run(['bash', str(MODEL_DIR / script), *options],
                                    env=self.environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(self.record.read_text()),
                             [str(SCRIPT), '--release', '65k', *preset, *options])

    def test_long_context_selects_the_chat_profile_on_the_current_image(self):
        image = launcher.IMAGES['65k']
        for weights in ('mxfp4',):
            for context in ('200000', '220000'):
                with self.subTest(weights=weights, context=context):
                    command = self.command('--context', context)
                    self.assertIn(image, command)
                    self.assertIn('RADIANCE_GDN_LAZY=0', command)
                    # prefix caching is not qualified with the 4-bit cache: FP8 cache, the chat profile's budget
                    self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
                    engine = self.engine(command)
                    self.assertEqual(value(engine, '--max-model-len'), context)
                    self.assertEqual(json.loads(value(engine, '--speculative-config'))['max_model_len'], int(context))
                    self.assertEqual(value(engine, '--max-num-seqs'), '1')
                    self.assertEqual(value(engine, '--max-num-batched-tokens'), '1024')
                    self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '8589934592')
                    self.assertEqual(value(engine, '--mamba-cache-mode'), 'align')
                    self.assertIn('--enable-prefix-caching', engine)
                    self.assertEqual(json.loads(value(engine, '--default-chat-template-kwargs')),
                                     {'enable_thinking': False})
        # up to the 65K preset's own limit, and with an explicit profile, nothing changes
        engine = self.engine(self.command('--context', '65536'))
        self.assertEqual(value(engine, '--max-num-seqs'), '8')
        self.assertIn('--no-enable-prefix-caching', engine)
        engine = self.engine(self.command('--profile', 'release', '--context', '200000'))
        self.assertEqual(value(engine, '--max-num-seqs'), '8')
        self.assertIn('--no-enable-prefix-caching', engine)
        engine = self.engine(self.command('--profile', 'desktop', '--context', '100000'))
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(2 * 1024**3))
        self.assertEqual(value(engine, '--max-model-len'), '100000')

    def test_long_context_on_the_3bit_weights_serves_262k_with_eight_sequences(self):
        image = launcher.IMAGES['65k']
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        for options in (('--context', '262144'), ('--context', '200000'), ('--profile', 'chat', '--context', '262144')):
            with self.subTest(options=options):
                command = self.command(*options)
                self.assertIn(image, command)
                self.assertIn('RADIANCE_GDN_LAZY=0', command)
                self.assertIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64,per_process_memory_fraction:0.95', command)
                self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
                engine = self.engine(command)
                context = options[options.index('--context') + 1]
                self.assertEqual(value(engine, '--max-model-len'), context)
                self.assertEqual(json.loads(value(engine, '--speculative-config'))['max_model_len'], int(context))
                self.assertEqual(value(engine, '--max-num-seqs'), '8')
                self.assertEqual(value(engine, '--max-num-batched-tokens'), '4096')
                self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(launcher.W3_LONG_KV_CACHE_BYTES))
                self.assertEqual(json.loads(value(engine, '--compilation-config'))['cudagraph_capture_sizes'],
                                 [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64])
                self.assertEqual(value(engine, '--mamba-cache-mode'), 'align')
                self.assertIn('--enable-prefix-caching', engine)
                self.assertEqual(json.loads(value(engine, '--default-chat-template-kwargs')),
                                 {'enable_thinking': False})
        # explicit overrides still win
        engine = self.engine(self.command('--context', '262144', '--max-num-seqs', '2',
                                          '--max-num-batched-tokens', '2048', '--kv-cache-memory-bytes', '9000000000'))
        self.assertEqual(value(engine, '--max-num-seqs'), '2')
        self.assertEqual(value(engine, '--max-num-batched-tokens'), '2048')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '9000000000')
        # the 65K preset itself is untouched
        engine = self.engine(self.command('--context', '65536'))
        self.assertEqual(value(engine, '--max-num-seqs'), '8')
        self.assertIn('--no-enable-prefix-caching', engine)

    def test_long_context_on_mxfp4_keeps_one_request_and_stops_at_220k(self):
        for context in ('200000', '220000'):
            with self.subTest(context=context):
                engine = self.engine(self.command('--context', context))
                self.assertEqual(value(engine, '--max-num-seqs'), '1')
                self.assertEqual(value(engine, '--max-num-batched-tokens'), '1024')
                self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '8589934592')
                self.assertEqual(json.loads(value(engine, '--compilation-config'))['cudagraph_capture_sizes'],
                                 [1, 2, 4, 8])
        for options in (('--context', '220001'), ('--context', '262144'),
                        ('--weights', 'mxfp4', '--context', '262144'), ('--profile', 'chat', '--context', '240000')):
            with self.subTest(options=options):
                self.record.unlink(missing_ok=True)
                result = self.run_launcher(*options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('3-bit', result.stderr)
                self.assertIn('220000', result.stderr)
                self.assertFalse(self.record.exists())

    def test_ngram_codraft_forwarded_only_when_set_on_host(self):
        command = self.command()
        self.assertFalse(any(item.startswith('PAITON_NGRAM_CODRAFT') for item in command))
        self.environment.update(PAITON_NGRAM_CODRAFT='1', PAITON_NGRAM_CODRAFT_HOT_MATCH='32')
        command = self.command()
        image_index = command.index(launcher.IMAGES['65k'])
        self.assertIn('PAITON_NGRAM_CODRAFT=1', command[:image_index])
        self.assertIn('PAITON_NGRAM_CODRAFT_HOT_MATCH=32', command[:image_index])
        self.assertEqual(command[command.index('PAITON_NGRAM_CODRAFT=1') - 1], '-e')

    def test_w3a4_weights_follow_the_mounted_directory(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        flags = [name + '=0' for name in launcher.W3_FLAGS]
        # Without the 3-bit weights the image serves MXFP4 and says how to enable them.
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PAITON_W3ROT_DIR', result.stderr)
        command = ['docker', *json.loads(self.record.read_text())]
        image_index = command.index(launcher.IMAGES['65k'])
        for flag in flags:
            self.assertIn(flag, command[:image_index])
        self.assertFalse(any(item.endswith(':/models/w3rot:ro') for item in command))
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        command = self.command()
        image_index = command.index(launcher.IMAGES['65k'])
        self.assertIn(f'{w3rot}:/models/w3rot:ro', command[:image_index])
        self.assertFalse(any(flag in command for flag in flags))
        # The memory the 3-bit weights free goes to the KV cache unless a budget is given,
        # with the allocator setting that budget was measured with (4-bit KV cache budget by default), capped below
        # the VRAM edge.
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV4_CACHE_BYTES))
        self.assertIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64,per_process_memory_fraction:0.95', command[:image_index])
        command = self.command('--kv-cache-memory-bytes', '7000000000')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '7000000000')
        command = self.command('--profile', 'desktop')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(2 * 1024**3))
        command = self.command('--weights', 'mxfp4')
        self.assertFalse(any(item.endswith(':/models/w3rot:ro') for item in command))
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '6535819798')
        for flag in flags:
            self.assertIn(flag, command)

    def test_w3a4_weights_require_their_directory_and_the_65k_image(self):
        result = self.run_launcher('--weights', 'w3a4')
        self.assertEqual(result.returncode, 2)
        self.assertIn('PAITON_W3ROT_DIR', result.stderr)
        self.environment['PAITON_W3ROT_DIR'] = str(self.root / 'missing')
        result = self.run_launcher('--weights', 'w3a4')
        self.assertEqual(result.returncode, 2)
        self.assertIn('not an existing directory', result.stderr)
        result = self.run_launcher('--release', '200k', '--weights', 'w3a4')
        self.assertEqual(result.returncode, 2)
        self.assertIn('not available for the 200k release', result.stderr)
        self.assertFalse(self.record.exists())
        command = self.command('--release', '200k')
        self.assertFalse(any(item.startswith('PAITON_W3_') for item in command))
        self.assertFalse(any(item.endswith(':/models/w3rot:ro') for item in command))


    def kv4_flags(self, command, image):
        index = command.index(image)
        return [item for item in command[:index] if item.startswith('PAITON_KV4')]

    def test_w3a4_serves_the_capacity_kv4_cache_with_its_budget(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        image = launcher.IMAGES['65k']
        command = self.command()
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV4_CACHE_BYTES))
        self.assertLess(launcher.W3_KV4_CACHE_BYTES, launcher.W3_KV_CACHE_BYTES)
        # explicit fp8 keeps the previous release budget and cache
        command = self.command('--kv-cache', 'fp8')
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV_CACHE_BYTES))
        # a user budget is respected either way
        command = self.command('--kv-cache-memory-bytes', '7000000000')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '7000000000')
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'])

    def test_kv4_stays_off_where_it_was_not_qualified(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        image = launcher.IMAGES['65k']
        # MXFP4 weights (no rotated weights mounted): fp8 KV as before
        command = self.command()
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '6535819798')
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        # prefix caching (and the chat profile that uses it): fp8 KV
        for options in (('--prefix-caching', 'on'), ('--profile', 'chat')):
            with self.subTest(options=options):
                command = self.command(*options)
                self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
        # an explicit kv4 request where it is not qualified is refused before Docker runs
        for options in (('--kv-cache', 'kv4', '--weights', 'mxfp4'), ('--kv-cache', 'kv4', '--prefix-caching', 'on'),
                        ('--kv-cache', 'kv4', '--release', '200k')):
            with self.subTest(options=options):
                self.record.unlink(missing_ok=True)
                result = self.run_launcher(*options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--kv-cache kv4', result.stderr)
                self.assertFalse(self.record.exists())
        # the 200k release is not a KV4 image: no KV4 flags at all
        command = self.command('--release', '200k')
        self.assertEqual(self.kv4_flags(command, launcher.IMAGES['200k']), [])


    def test_kv4_stays_within_its_context_limit(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        image = launcher.IMAGES['65k']
        on, off = ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'], ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0']
        # automatic selection only in the measured configuration: the 65K preset at up to 65,536 tokens
        self.assertEqual(self.kv4_flags(self.command('--context', '65536'), image), on)
        for options in (('--profile', 'release', '--context', '65537'), ('--profile', 'desktop')):
            with self.subTest(options=options):
                self.assertEqual(self.kv4_flags(self.command(*options), image), off)
        command = self.command('--profile', 'release', '--context', str(launcher.KV4_V4_MAX_CONTEXT))
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV_CACHE_BYTES))
        # an explicit request covers the kernels' range, with that mode's own budget: 200,000 on the 28 and 29 September
        # images (bundle kv4-v4), the model's native 262,144 on the pinned 2 October image and others with bundle kv4-v5
        self.assertEqual((launcher.KV4_V4_MAX_CONTEXT, launcher.KV4_MAX_CONTEXT), (200000, 262144))
        for img, limit in ((self.R2_IMAGE, launcher.KV4_V4_MAX_CONTEXT), (self.V5_IMAGE, launcher.KV4_MAX_CONTEXT),
                           (image, launcher.KV4_MAX_CONTEXT)):
            with self.subTest(image=img):
                command = self.command('--image', img, '--profile', 'release', '--kv-cache', 'kv4', '--context', str(limit))
                self.assertEqual(self.kv4_flags(command, img), on)
                self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV4_CACHE_BYTES))
        command = self.command('--profile', 'desktop', '--kv-cache', 'kv4')
        self.assertEqual(self.kv4_flags(command, image), on)
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(2 * 1024**3))
        self.record.unlink(missing_ok=True)
        result = self.run_launcher('--image', self.R2_IMAGE, '--profile', 'release', '--kv-cache', 'kv4',
                                   '--context', str(launcher.KV4_V4_MAX_CONTEXT + 1))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('--kv-cache kv4', result.stderr)
        self.assertFalse(self.record.exists())


    V5_IMAGE = 'paiton-qwen38-local:kv4-v5-candidate'   # any image other than the kv4-v4 ones (bundle kv4-v5)
    R2_IMAGE = next(image for image in launcher.KV4_V4_IMAGES if '-20260929-r2@' in image)   # the previous pin

    def test_kv4_in_the_3bit_long_mode_runs_without_prefix_caching_with_its_own_budget(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        image = self.V5_IMAGE
        on, off = ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'], ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0']
        for options in (('--context', '262144'), ('--profile', 'chat'), ('--context', '200000')):
            with self.subTest(options=options):
                command = self.command('--image', image, '--kv-cache', 'kv4', *options)
                self.assertEqual(self.kv4_flags(command, image), on)
                self.assertIn('RADIANCE_GDN_LAZY=0', command)
                self.assertIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64,per_process_memory_fraction:0.95', command)
                engine = command[command.index(image) + 1:]
                # 3 Oct: a prefix-cache hit on a shared-prefix (junction) checkpoint of the 4-bit long mode ended in a
                # GDN-norm nonfinite engine error, so the 4-bit long mode runs without prefix caching until fixed
                self.assertIn('--no-enable-prefix-caching', engine)
                self.assertNotIn('--enable-prefix-caching', engine)
                self.assertEqual(value(engine, '--max-num-seqs'), '8')
                self.assertEqual(value(engine, '--max-num-batched-tokens'), '4096')
                self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(launcher.W3_LONG_KV4_CACHE_BYTES))
                self.assertNotIn('--prefix-cache-retention-interval', engine)
                # the allocator stops short of the VRAM edge where the driver would evict to system memory
                self.assertIn('PAITON_VRAM_HEADROOM_MIB=1024', command[:command.index(image)])
        # without prefix caching there is nothing to retain: vLLM refuses a retention interval that is not a multiple
        # of the (then unaligned, 176,947,200-token) scheduler block, so the server would not start (3 Oct repro arm)
        command = self.command('--image', image, '--kv-cache', 'kv4', '--context', '262144', '--prefix-caching', 'off')
        engine = command[command.index(image) + 1:]
        self.assertIn('--no-enable-prefix-caching', engine)
        self.assertNotIn('--prefix-cache-retention-interval', engine)
        # the fp8 long mode (the automatic choice) keeps its released settings
        command = self.command('--image', image, '--context', '262144')
        self.assertNotIn('--prefix-cache-retention-interval', command)
        # the warm-up cap applies to the fp8 long mode too (after startup; ignored by images without the overlay)
        self.assertIn('PAITON_VRAM_HEADROOM_MIB=1024', command[:command.index(image)])
        command = self.command('--image', image, '--context', '245000', '--vision')
        self.assertFalse([item for item in command if item.startswith('PAITON_VRAM_HEADROOM_MIB')])
        # the 4-bit mode keeps room for its prefill workspace: a smaller pool than the fp8 long mode's
        self.assertLess(launcher.W3_LONG_KV4_CACHE_BYTES, launcher.W3_LONG_KV_CACHE_BYTES)
        # automatic selection keeps the fp8 cache in the long mode; an explicit budget wins
        self.assertEqual(self.kv4_flags(self.command('--image', image, '--context', '262144'), image), off)
        command = self.command('--image', image, '--kv-cache', 'kv4', '--context', '262144',
                               '--kv-cache-memory-bytes', '9000000000')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '9000000000')

    def test_kv4_long_mode_is_refused_where_it_was_not_qualified(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        # the 28 and 29 September images carry KV4 bundle kv4-v4 (decode stops at 200,000, no prefix-caching check);
        # the pinned 2 October image does not
        self.assertEqual(len(launcher.KV4_V4_IMAGES), 2)
        self.assertNotIn(launcher.IMAGES['65k'], launcher.KV4_V4_IMAGES)
        cases = ((('--image', self.R2_IMAGE, '--context', '262144'), True, 'kv4-v5'),
                 (('--image', self.R2_IMAGE, '--context', '200000'), True, 'kv4-v5'),
                 (('--image', self.V5_IMAGE, '--context', '200000', '--weights', 'mxfp4'), False, '3-bit'),
                 (('--image', self.V5_IMAGE, '--context', '200000', '--vision'), True, '--vision'),
                 (('--image', self.V5_IMAGE, '--prefix-caching', 'on'), True, 'runs without prefix caching'),
                 (('--image', self.V5_IMAGE, '--context', '262144', '--prefix-caching', 'on'), True, 'prefix caching'))
        for options, w3, reason in cases:
            with self.subTest(options=options):
                self.environment.pop('PAITON_W3ROT_DIR', None)
                if w3:
                    self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
                self.record.unlink(missing_ok=True)
                result = self.run_launcher('--kv-cache', 'kv4', *options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--kv-cache kv4', result.stderr)
                self.assertIn(reason, result.stderr)
                self.assertFalse(self.record.exists())

    def test_legacy_kv4_long_flags_get_the_checks_of_mode_long_kv4(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        # every spelling of the 4-bit long-context mode, an explicit --prefix-caching off included, and the subject its
        # refusal names; the reason itself is the one shared check (long_kv4_refusal)
        legacy = '--kv-cache kv4: the 4-bit long-context mode (--mode long-kv4) '
        spellings = ((('--mode', 'long-kv4'), '--mode long-kv4 '),
                     (('--mode', 'long-kv4', '--prefix-caching', 'off'), '--mode long-kv4 '),
                     (('--context', '262144', '--kv-cache', 'kv4'), legacy),
                     (('--context', '262144', '--kv-cache', 'kv4', '--prefix-caching', 'off'), legacy),
                     (('--context', '200000', '--kv-cache', 'kv4', '--prefix-caching', 'off'), legacy),
                     (('--context', '65537', '--kv-cache', 'kv4', '--prefix-caching', 'off'), legacy),
                     (('--profile', 'chat', '--kv-cache', 'kv4', '--prefix-caching', 'off'), legacy))
        refusals = ((('--image', self.R2_IMAGE), 'needs an image with KV4 bundle kv4-v5'),
                    (('--image', next(iter(launcher.KV4_V4_IMAGES - {self.R2_IMAGE}))),
                     'needs an image with KV4 bundle kv4-v5'),
                    (('--vision',), 'is not qualified with --vision'),
                    (('--weights', 'mxfp4'), 'needs the 3-bit W3A4 weights'),
                    (('--prefix-caching', 'on'), 'runs without prefix caching'))
        for spelling, subject in spellings:
            for refusal, reason in refusals:
                if '--prefix-caching' in spelling and '--prefix-caching' in refusal:
                    continue
                with self.subTest(spelling=spelling, refusal=refusal):
                    stderr = self.refused(*spelling, *refusal)
                    self.assertIn(subject + reason, stderr)
        # MXFP4 without the 3-bit weights in the environment: the same refusal
        self.environment.pop('PAITON_W3ROT_DIR')
        for spelling, subject in spellings:
            with self.subTest(spelling=spelling, weights='auto'):
                self.assertIn(subject + 'needs the 3-bit W3A4 weights', self.refused(*spelling))
        # the qualified configuration is served the same way by every spelling
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        expected = self.dry_run('--mode', 'long-kv4')
        for spelling, _ in spellings[:4]:
            with self.subTest(spelling=spelling, served=True):
                self.assertEqual(self.dry_run(*spelling), expected)

    def test_kv4_long_mode_says_it_runs_without_prefix_caching(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        result = self.run_launcher('--image', self.V5_IMAGE, '--kv-cache', 'kv4', '--context', '262144')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('without prefix caching', result.stderr)
        self.assertIn('10.2 s', result.stderr)
        # the default fp8 long mode keeps prefix caching and prints no such notice
        result = self.run_launcher('--image', self.V5_IMAGE, '--context', '262144')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('without prefix caching', result.stderr)

    def test_vision_loads_the_encoder_with_a_smaller_kv_budget(self):
        image = launcher.IMAGES['65k']
        self.assertIn('--language-model-only', self.engine(self.command()))
        # MXFP4 weights: image input on, KV budget reduced for the vision encoder, and the allocator setting the
        # vision budgets were measured with (without it the encoder's startup profile fragments the cache)
        command = self.command('--vision')
        self.assertIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64', command[:command.index(image)])
        engine = self.engine(command)
        self.assertNotIn('--language-model-only', engine)
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(launcher.VISION_KV_CACHE_BYTES['mxfp4', 'fp8']))
        self.assertLess(launcher.VISION_KV_CACHE_BYTES['mxfp4', 'fp8'], 6535819798)
        # 3-bit weights: the 4-bit KV cache stays the default, the FP8 cache on request, each with its own budget
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        command = self.command('--vision')
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.VISION_KV_CACHE_BYTES['w3a4', 'kv4']))
        self.assertLess(launcher.VISION_KV_CACHE_BYTES['w3a4', 'kv4'], launcher.W3_KV4_CACHE_BYTES)
        command = self.command('--vision', '--kv-cache', 'fp8')
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.VISION_KV_CACHE_BYTES['w3a4', 'fp8']))
        self.assertLess(launcher.VISION_KV_CACHE_BYTES['w3a4', 'fp8'], launcher.W3_KV_CACHE_BYTES)
        # explicit budgets still win
        command = self.command('--vision', '--kv-cache-memory-bytes', '3000000000')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '3000000000')
        engine = self.engine(self.command('--vision', '--gpu-memory-utilization', '0.9'))
        self.assertNotIn('--kv-cache-memory-bytes', engine)
        self.assertNotIn('--language-model-only', engine)

    def test_vision_with_the_desktop_profile_keeps_its_budget(self):
        engine = self.engine(self.command('--profile', 'desktop', '--vision'))
        self.assertNotIn('--language-model-only', engine)
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(2 * 1024**3))
        self.assertEqual(value(engine, '--max-model-len'), '32768')

    def test_vision_is_refused_where_it_was_not_qualified(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        for options, reason, w3 in ((('--context', '200000'), '3-bit', False),
                                    (('--profile', 'chat'), '3-bit', False),
                                    (('--prefix-caching', 'on'), 'experimental', True),
                                    (('--release', '200k'), 'not available for the 200k release', False)):
            with self.subTest(options=options):
                self.environment.pop('PAITON_W3ROT_DIR', None)
                if w3:
                    self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
                self.record.unlink(missing_ok=True)
                result = self.run_launcher('--vision', *options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--vision', result.stderr)
                self.assertIn(reason, result.stderr)
                self.assertFalse(self.record.exists())

    def test_long_prefill_threshold_is_an_opt_in_passthrough(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        engine = self.engine(self.command('--context', '262144', '--long-prefill-threshold', '3072'))
        self.assertEqual(value(engine, '--long-prefill-token-threshold'), '3072')
        engine = self.engine(self.command('--context', '262144'))
        self.assertNotIn('--long-prefill-token-threshold', engine)
        engine = self.engine(self.command('--long-prefill-threshold', '2048'))
        self.assertEqual(value(engine, '--long-prefill-token-threshold'), '2048')

    def test_allocator_cap_in_the_3bit_profiles_without_vision(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        # the 262K mode and the 65K default (which reached the KFD eviction edge under BetterBench uncapped, 2 Oct)
        for options in (('--context', '262144'), ()):
            with self.subTest(options=options):
                command = self.command(*options)
                capped = [item for item in command if item.startswith('PYTORCH_ALLOC_CONF=')]
                self.assertEqual(capped, ['PYTORCH_ALLOC_CONF=max_split_size_mb:64,per_process_memory_fraction:0.95'])
                self.assertIn('PAITON_VRAM_HEADROOM_MIB=1024', command)
        for options in (('--context', '245000', '--vision'), ('--vision',)):
            with self.subTest(options=options):
                setting = [item for item in self.command(*options) if item.startswith('PYTORCH_ALLOC_CONF=')]
                self.assertEqual(setting, ['PYTORCH_ALLOC_CONF=max_split_size_mb:64'])
        del self.environment['PAITON_W3ROT_DIR']
        setting = [item for item in self.command('--context', '200000') if item.startswith('PYTORCH_ALLOC_CONF=')]
        self.assertEqual(setting, ['PYTORCH_ALLOC_CONF=max_split_size_mb:64'])   # MXFP4 long mode: not measured

    def test_vision_in_the_3bit_long_mode_uses_its_budget_and_capacity_limit(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        limit = launcher.W3_LONG_VISION_MAX_CONTEXT
        for options in (('--context', str(limit), '--vision'), ('--profile', 'chat', '--vision')):
            with self.subTest(options=options):
                command = self.command(*options)
                self.assertIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64', command)
                engine = self.engine(command)
                self.assertNotIn('--language-model-only', engine)
                self.assertIn('--enable-prefix-caching', engine)
                self.assertEqual(value(engine, '--max-num-seqs'), '8')
                self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(launcher.W3_LONG_VISION_KV_CACHE_BYTES))
        self.assertLess(launcher.W3_LONG_VISION_KV_CACHE_BYTES, launcher.W3_LONG_KV_CACHE_BYTES)
        self.record.unlink(missing_ok=True)
        result = self.run_launcher('--vision', '--context', str(limit + 1))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('--vision', result.stderr)
        self.assertIn(str(limit), result.stderr)
        self.assertFalse(self.record.exists())

    def dry_run(self, *args):
        result = self.run_launcher('--dry-run', *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.record.exists())
        return json.loads(result.stdout)

    def refused(self, *args):
        self.record.unlink(missing_ok=True)
        result = self.run_launcher(*args)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.record.exists())
        return result.stderr

    def test_mode_presets_expand_to_exactly_the_legacy_flags(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        self.assertIn('qwen38-rocm10-vllm029-20261002-r1@sha256:', launcher.IMAGES['65k'])
        self.assertEqual(launcher.MODES, {'65k': 'no flags', 'long': '--context 262144',
                                          'long-kv4': '--context 262144 --kv-cache kv4'})
        legacy = ('--context', '262144', '--kv-cache', 'kv4')
        cases = (
            # the presets on the pinned image
            (('--mode', '65k'), ()),
            (('--mode', 'long'), ('--context', '262144')),
            (('--mode', 'long-kv4'), legacy),
            # flags that restate a preset
            (('--mode', '65k', '--profile', 'release', '--prefix-caching', 'off'), ()),
            (('--mode', 'long', '--profile', 'chat', '--kv-cache', 'fp8', '--prefix-caching', 'on'),
             ('--context', '262144')),
            (('--mode', 'long-kv4', '--profile', 'chat', '--kv-cache', 'kv4', '--prefix-caching', 'off'), legacy),
            # compatible refinements
            (('--mode', '65k', '--context', '32768', '--kv-cache', 'fp8', '--vision'),
             ('--context', '32768', '--kv-cache', 'fp8', '--vision')),
            (('--mode', 'long', '--context', '245000', '--vision'), ('--context', '245000', '--vision')),
            (('--mode', 'long', '--context', '32768'), ('--profile', 'chat', '--context', '32768')),
            (('--mode', 'long-kv4', '--context', '131072', '--max-num-seqs', '4', '--thinking', 'on', '--port', '18100',
              '--long-prefill-threshold', '2048', '--name', 'kv4-long', '--detach'),
             ('--context', '131072', '--kv-cache', 'kv4', '--max-num-seqs', '4', '--thinking', 'on', '--port', '18100',
              '--long-prefill-threshold', '2048', '--name', 'kv4-long', '--detach')),
            (('--mode', 'long-kv4', '--image', self.V5_IMAGE, '--kv-cache-memory-bytes', '9000000000'),
             ('--image', self.V5_IMAGE, *legacy, '--kv-cache-memory-bytes', '9000000000')),
            (('--mode', 'long', '--image', self.R2_IMAGE), ('--image', self.R2_IMAGE, '--context', '262144')),
        )
        for preset, flags in cases:
            with self.subTest(preset=preset):
                self.assertEqual(self.dry_run(*preset), self.dry_run(*flags))
        # what the two long presets stand for: one pinned image, eight requests, the capped allocator; fp8 with
        # prefix caching and its budget, or the 4-bit cache without prefix caching and its own budget
        image = launcher.IMAGES['65k']
        for mode, kv4, prefix, budget in (('long', '0', '--enable-prefix-caching', launcher.W3_LONG_KV_CACHE_BYTES),
                                          ('long-kv4', '1', '--no-enable-prefix-caching',
                                           launcher.W3_LONG_KV4_CACHE_BYTES)):
            with self.subTest(mode=mode):
                command = self.dry_run('--mode', mode)
                environment = command[:command.index(image)]
                self.assertEqual(self.kv4_flags(command, image), [f'PAITON_KV4={kv4}', f'PAITON_KV4_CAPACITY={kv4}'])
                self.assertIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64,per_process_memory_fraction:0.95', environment)
                self.assertIn('PAITON_VRAM_HEADROOM_MIB=1024', environment)
                engine = command[command.index(image) + 1:]
                self.assertIn(prefix, engine)
                self.assertNotIn('--prefix-cache-retention-interval', engine)
                self.assertEqual(value(engine, '--max-model-len'), '262144')
                self.assertEqual(value(engine, '--max-num-seqs'), '8')
                self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(budget))
        # the 4-bit preset prints what running without prefix caching costs; the fp8 preset does not
        result = self.run_launcher('--dry-run', '--mode', 'long-kv4')
        self.assertIn('without prefix caching', result.stderr)
        self.assertIn('10.2 s', result.stderr)
        self.assertNotIn('without prefix caching', self.run_launcher('--dry-run', '--mode', 'long').stderr)

    def test_mode_refuses_flags_that_contradict_it(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        cases = (
            (('--mode', 'long-kv4', '--prefix-caching', 'on'), 'without prefix caching'),
            (('--mode', 'long-kv4', '--kv-cache', 'fp8'), 'use --mode long'),
            (('--mode', 'long-kv4', '--vision'), '--vision'),
            (('--mode', 'long-kv4', '--image', self.R2_IMAGE), 'kv4-v5'),
            (('--mode', 'long-kv4', '--profile', 'release'), '--profile release'),
            (('--mode', 'long', '--kv-cache', 'kv4'), 'use --mode long-kv4'),
            (('--mode', 'long', '--prefix-caching', 'off'), 'prefix caching'),
            (('--mode', 'long', '--profile', 'desktop'), '--profile desktop'),
            (('--mode', 'long', '--vision', '--context', '245001'), '--context 245000'),
            (('--mode', 'long', '--context', '262145'), '--context 262144'),
            (('--mode', 'long-kv4', '--context', '262145'), '--context 262144'),
            (('--mode', '65k', '--context', '200000'), '--context 65536'),
            (('--mode', '65k', '--context', '65537'), '--context 65536'),
            (('--mode', '65k', '--prefix-caching', 'on'), 'use --mode long'),
            (('--mode', '65k', '--profile', 'chat'), '--profile chat'),
            (('--mode', 'long', '--release', '200k'), '65k release'),
        )
        for options, reason in cases:
            with self.subTest(options=options):
                stderr = self.refused(*options)
                self.assertIn('--mode ' + options[1], stderr)
                self.assertIn(reason, stderr)
        # the usual argument checks still apply
        for options in (('--mode', 'long-fp8'), ('--mode',)):
            with self.subTest(options=options):
                self.refused(*options)

    def test_mode_with_the_mxfp4_weights(self):
        image = launcher.IMAGES['65k']
        # 65k: the MXFP4 default, fp8 KV cache
        command = self.dry_run('--mode', '65k')
        self.assertEqual(command, self.dry_run())
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
        # long: MXFP4 serves one long request of 200,000 tokens by default, up to the tested 220,000
        command = self.dry_run('--mode', 'long')
        self.assertEqual(command, self.dry_run('--context', '200000'))
        self.assertEqual(command, self.dry_run('--mode', 'long', '--context', '200000'))
        self.assertEqual(value(command, '--max-model-len'), '200000')
        self.assertEqual(value(command, '--max-num-seqs'), '1')
        self.assertIn('--enable-prefix-caching', command)
        stderr = self.refused('--mode', 'long', '--context', '220001')
        for reason in ('--mode long with the MXFP4 weights', '--context 220000', '3-bit'):
            self.assertIn(reason, stderr)
        stderr = self.refused('--mode', 'long', '--vision')
        for reason in ('--mode long --vision needs the 3-bit W3A4 weights', 'without --mode long'):
            self.assertIn(reason, stderr)
        # long-kv4: the 4-bit KV cache is qualified with the 3-bit weights only
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        for options, w3 in ((('--mode', 'long-kv4'), False), (('--mode', 'long-kv4', '--context', '200000'), False),
                            (('--weights', 'mxfp4', '--mode', 'long-kv4'), True)):
            with self.subTest(options=options):
                self.environment.pop('PAITON_W3ROT_DIR', None)
                if w3:
                    self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
                stderr = self.refused(*options)
                self.assertIn('--mode long-kv4 needs the 3-bit W3A4 weights', stderr)
        # the quickstart scripts: run-mxfp4.sh refuses the 4-bit preset, run-3bit.sh serves it
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        for script, code in (('run-mxfp4.sh', 2), ('run-3bit.sh', 0)):
            with self.subTest(script=script):
                result = subprocess.run(['bash', str(MODEL_DIR / script), '--mode', 'long-kv4', '--dry-run'],
                                        env=self.environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
        self.assertEqual(self.kv4_flags(json.loads(result.stdout), image), ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'])

    def test_mode_long_picks_the_context_of_its_configuration(self):
        # --mode long needs no --context: MXFP4 weights 200,000 (one request), 3-bit weights with --vision 245,000; each
        # is the same Docker argv as the explicit --context spelling, through the launcher and the quickstart scripts
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.assertEqual((launcher.MXFP4_LONG_CONTEXT, launcher.MXFP4_LONG_MAX_CONTEXT,
                          launcher.W3_LONG_VISION_MAX_CONTEXT), (200000, 220000, 245000))

        def script(name, *options):
            result = subprocess.run(['bash', str(MODEL_DIR / name), *options, '--dry-run'],
                                    env=self.environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout), result.stderr

        for w3, preset, flags, context in (
                (False, ('--mode', 'long'), ('--context', '200000'), '200000'),
                (False, ('--mode', 'long'), ('--profile', 'chat'), '200000'),
                (True, ('--weights', 'mxfp4', '--mode', 'long'), ('--weights', 'mxfp4', '--context', '200000'), '200000'),
                (True, ('--mode', 'long', '--vision'), ('--context', '245000', '--vision'), '245000'),
                # an explicit smaller --context still works; MXFP4 up to the tested 220,000
                (False, ('--mode', 'long', '--context', '150000'), ('--context', '150000'), '150000'),
                (False, ('--mode', 'long', '--context', '220000'), ('--context', '220000'), '220000'),
                (True, ('--mode', 'long', '--vision', '--context', '200000'), ('--context', '200000', '--vision'),
                 '200000')):
            with self.subTest(w3=w3, preset=preset):
                self.environment.pop('PAITON_W3ROT_DIR', None)
                if w3:
                    self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
                command = self.dry_run(*preset)
                self.assertEqual(command, self.dry_run(*flags))
                self.assertEqual(value(command, '--max-model-len'), context)
                self.assertEqual(json.loads(value(command, '--speculative-config'))['max_model_len'], int(context))
        for name, preset, flags in (('run-mxfp4.sh', ('--mode', 'long'), ('--context', '200000')),
                                    ('run-3bit.sh', ('--mode', 'long', '--vision'), ('--context', '245000', '--vision'))):
            with self.subTest(script=name):
                command, stderr = script(name, *preset)
                self.assertEqual(command, script(name, *flags)[0])
                self.assertIn(f'--mode long: context {int(flags[1]):,} tokens', stderr)
        # the 3-bit text mode keeps the model's full context and prints no such note
        command, stderr = script('run-3bit.sh', '--mode', 'long')
        self.assertEqual(value(command, '--max-model-len'), '262144')
        self.assertNotIn('--mode long: context', stderr)
        # a larger --context is refused with the mode's limit
        self.environment.pop('PAITON_W3ROT_DIR', None)
        self.assertIn('up to --context 220000', self.refused('--mode', 'long', '--context', '220001'))
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        stderr = self.refused('--mode', 'long', '--vision', '--context', '245001')
        for reason in ('--mode long --vision serves up to --context 245000', 'vision encoder', 'Drop --context'):
            self.assertIn(reason, stderr)

    def test_help_puts_the_choices_first_in_groups(self):
        result = self.run_launcher('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        sections, current = {}, None
        for line in result.stdout.splitlines():
            if line and not line.startswith(' ') and line.endswith(':'):
                current = line[:-1]
                sections[current] = []
            elif current and line.startswith('  --'):
                sections[current].append(line.split()[0])
        self.assertEqual([name for name in sections if name not in ('options', 'optional arguments')],
                         ['Choose how to run', 'Server', 'Advanced tuning'])
        self.assertEqual(sections['Choose how to run'], ['--weights', '--mode', '--vision'])
        self.assertEqual(sections['Server'], ['--port', '--name', '--detach', '--dry-run', '--list-gpus', '--image'])
        self.assertEqual(sorted(sections['Advanced tuning']), sorted((
            '--context', '--max-num-seqs', '--kv-cache', '--prefix-caching', '--thinking', '--long-prefill-threshold',
            '--profile', '--kv-cache-memory-bytes', '--gpu-memory-utilization', '--max-num-batched-tokens')))
        self.assertNotIn('--release', result.stdout)        # hidden, still accepted
        self.assertEqual(self.dry_run('--release', '65k'), self.dry_run())
        # every option of the parser is in one of the groups
        listed = {flag for flags in sections.values() for flag in flags}
        options = {item for action in launcher.parser()._actions for item in action.option_strings
                   if item.startswith('--') and action.help is not launcher.argparse.SUPPRESS}
        self.assertEqual(options - {'--help'}, listed - {'--help'})

    def test_help_describes_each_mode_on_its_own_line(self):
        result = self.run_launcher('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line.strip() for line in result.stdout.splitlines()]
        for mode in launcher.MODES:
            self.assertEqual(sum(line.startswith(mode + ': ') for line in lines), 1, mode)
        text = ' '.join(result.stdout.split())
        for phrase in ('65k: 65,536 context, up to 8 requests', 'the fast everyday default',
                       'long: 262,144 context (MXFP4: 200,000, one request), fp8 KV cache with prefix caching',
                       'one long document at a time, fast follow-ups',
                       'long-kv4: 262,144 context per request, 4-bit KV cache',
                       'a 1.63x larger shared pool (458,922 tokens): more long conversations at once',
                       'no prefix caching, every request re-reads its prompt (3-bit weights only)',
                       'mxfp4 gives you the most accurate weights', 'w3a4 the 3-bit weights, fastest with the most context',
                       'gives you image input; works with --mode 65k and long (long: up to 245,000 context), not with '
                       'long-kv4'):
            self.assertIn(phrase, text)

if __name__ == '__main__':
    unittest.main()

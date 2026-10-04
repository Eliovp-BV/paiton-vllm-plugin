"""Start-time item A11, self-request half (opt-in: PAITON_START_WARMUP=1).

Spawned detached by the serving CLI before it hands over to vLLM, this helper waits for /health and then sends the
request shapes a fresh server would otherwise compile Triton kernels for on a user's first requests: one single-stream
generation (C1 sampler kernels), eight concurrent generations (C8: PIECEWISE forward pieces, sampler at num_reqs 8), a
2,048-token prompt (prefill buckets) and the same prompt again (prefix-cache hit path). Everything is greedy, short, and
discarded; when it is done it logs '[paiton.warmup] self-warm-up complete' so a JIT-monitor line after that marks a real
gap. Nothing changes in the kernels: outputs stay identical.
"""
import json
import os
import sys
import threading
import time
import urllib.request

PORT = int(os.environ.get('PAITON_PORT', '18982'))
URL = f'http://127.0.0.1:{PORT}'
WORDS = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon'.split()


def post(path, payload, timeout=900):
    req = urllib.request.Request(URL + path, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def completion(prompt, max_tokens):
    return post('/v1/completions', {'model': 'Qwen3.8', 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0.0, 'seed': 42, 'ignore_eos': True})


def wait_health(limit_s):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            urllib.request.urlopen(URL + '/health', timeout=5).read()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False


def main():
    os.nice(10)
    if not wait_health(float(os.environ.get('PAITON_START_WARMUP_WAIT_S', '1800'))):
        print('[paiton.warmup] self-warm-up skipped: no /health', flush=True)
        return 0
    t0 = time.time(); steps = []
    try:
        completion('Reply with one word: hello.', 16); steps.append('c1')
        out = [None] * 8
        def one(i):
            out[i] = completion(' '.join(WORDS[(i * 7 + k) % len(WORDS)] for k in range(48)) + '\nContinue.', 16)
        th = [threading.Thread(target=one, args=(i,)) for i in range(8)]
        for t in th: t.start()
        for t in th: t.join()
        steps.append('c8')
        long_prompt = ' '.join(WORDS[(k * 3) % len(WORDS)] for k in range(2048)) + '\nSummarise the sequence.'
        completion(long_prompt, 16); steps.append('prefill-2k')
        completion(long_prompt, 16); steps.append('prefix-hit')
    except Exception as e:  # noqa: BLE001
        print(f'[paiton.warmup] self-warm-up stopped after {steps}: {e!r}', flush=True)
        return 0
    print(f'[paiton.warmup] self-warm-up complete in {time.time() - t0:.1f} s: {steps}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Local stage commands; separate processes release GPU memory between models."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.meeting-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write('\n'); output.flush(); os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['PYANNOTE_METRICS_ENABLED'] = '0'
    parser = argparse.ArgumentParser(description='Offline meeting processing. Recording capture is a separate, user-started step.')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect'); inspect.add_argument('recording')
    normalize = sub.add_parser('normalize'); normalize.add_argument('recording'); normalize.add_argument('--output', required=True); normalize.add_argument('--track', type=int, default=0); normalize.add_argument('--channel', type=int)
    asr = sub.add_parser('transcribe')
    diar = sub.add_parser('diarize')
    for stage in (asr, diar):
        stage.add_argument('recording'); stage.add_argument('--model', required=True)
        stage.add_argument('--output', required=True); stage.add_argument('--track', type=int, default=0)
        stage.add_argument('--channel', type=int)
    asr.add_argument('--vad', required=True); asr.add_argument('--artifact')
    diar.add_argument('--scratch')
    attribute = sub.add_parser('attribute')
    attribute.add_argument('transcript'); attribute.add_argument('diarization'); attribute.add_argument('--output', required=True)
    summary = sub.add_parser('summarize')
    summary.add_argument('transcript')
    backend = summary.add_mutually_exclusive_group(required=True)
    backend.add_argument('--endpoint'); backend.add_argument('--checkpoint')
    summary.add_argument('--model'); summary.add_argument('--tokenizer')
    summary.add_argument('--output', required=True)
    export = sub.add_parser('export')
    export.add_argument('result'); export.add_argument('--format', choices=['json','txt','srt','vtt','summary.txt'], default='json')
    export.add_argument('--output', required=True)
    bench=sub.add_parser('benchmark-asr')
    bench.add_argument('recording');bench.add_argument('--model',required=True);bench.add_argument('--vad',required=True);bench.add_argument('--artifact');bench.add_argument('--output',required=True);bench.add_argument('--repeats',type=int,default=3)
    pipeline=sub.add_parser('process', help='Import, transcribe, diarize and summarize entirely locally.')
    pipeline.add_argument('recording');pipeline.add_argument('--models',required=True);pipeline.add_argument('--output',required=True)
    pipeline.add_argument('--artifact');pipeline.add_argument('--track',type=int,default=0);pipeline.add_argument('--channel',type=int)
    pipeline.add_argument('--keep-intermediates',action='store_true')
    for command in (summary, pipeline):
        command.add_argument('--summary-backend', choices=['transformers','vllm'],
            default=os.environ.get('PAITON_MEETING_SUMMARY_BACKEND','transformers'))
    args = parser.parse_args()
    if args.command=='process':
        from .pipeline import process
        process(args.recording,args.models,args.output,args.artifact,args.track,args.channel,args.keep_intermediates,args.summary_backend)
        return
    if args.command=='benchmark-asr':
        from .benchmark import benchmark
        benchmark(args.recording,args.model,args.vad,args.output,args.artifact,args.repeats)
        return
    if args.command == 'inspect':
        from .audio import inspect_audio
        print(json.dumps(inspect_audio(args.recording)))
        return
    if args.command == 'normalize':
        from .audio import normalize
        print(json.dumps(normalize(args.recording, args.output, args.track, args.channel)))
        return
    if args.command == 'transcribe':
        from .asr import ParakeetASR
        model = ParakeetASR(args.model, artifact=args.artifact, vad_path=args.vad)
        result = model.transcribe(args.recording, track=args.track, channel=args.channel)
    elif args.command == 'diarize':
        from .diarization import diarize
        result = diarize(args.recording, args.model, track=args.track, channel=args.channel, scratch=args.scratch)
    elif args.command == 'attribute':
        from .transcript import segments_from_words
        result = json.loads(Path(args.transcript).read_text())
        diarization = json.loads(Path(args.diarization).read_text())
        result['segments'] = segments_from_words(result['words'], diarization['turns'])
        result['diarization'] = diarization
        result['speaker_attribution'] = 'anonymous-diarization'
    elif args.command == 'summarize':
        result = json.loads(Path(args.transcript).read_text())
        if args.checkpoint:
            if args.summary_backend == 'vllm':
                from .vllm_summary import VLLMSummarizer
                model = VLLMSummarizer(args.checkpoint)
            else:
                from .compact_summary import CompactSummarizer
                model = CompactSummarizer(args.checkpoint)
        else:
            if not args.model or not args.tokenizer:
                parser.error('--endpoint requires --model and --tokenizer')
            from .local_summary import LocalSummarizer
            model = LocalSummarizer(args.endpoint, args.model, args.tokenizer)
        result.update(model.summarize(result['segments']))
    else:
        from .export import export
        result = json.loads(Path(args.result).read_text())
        text = export(result, args.format, result.get('speaker_names'))
        # Do not silently replace a user's existing exported file.
        fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as output: output.write(text)
        return
    save(args.output, result)
    # Only timings/status go to terminal logs. Content remains in local output.
    print(json.dumps({'status': 'complete', 'stage': args.command,
                      'timings': {k: v for k, v in result.items() if k.endswith('_seconds')}}))


if __name__ == '__main__':
    main()

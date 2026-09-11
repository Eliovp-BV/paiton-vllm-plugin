"""Score final CLI transcripts against timed reference words; no hosted judge."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def normalize(text):
    return ' '.join(re.sub(r"[^\w\s']", ' ', text.lower().replace('_', '')).split())


def timed_tokens(rows):
    result = []
    for row in rows:
        for token in re.sub(r"[^\w\s']", ' ', row['text'].lower()).split():
            result.append(dict(text=token, start=row['start'], end=row['end']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-words', type=Path, required=True,
                        help='JSON list of text/start/end objects in original reference order')
    parser.add_argument('--runs', type=Path, required=True,
                        help='Complete benchmark directory containing */result.json')
    parser.add_argument('--baseline', type=Path,
                        help='Optional previously scored CLI result for exact speaker-turn comparison')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new file.')
    import numpy as np
    from jiwer import process_words, wer, cer
    reference_bytes = args.reference_words.read_bytes()
    reference = json.loads(reference_bytes)
    reference_text = normalize(' '.join(row['text'] for row in reference))
    if not reference_text:
        parser.error('A nonempty speech reference is required for this score.')
    ref_tokens = timed_tokens(reference)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    files = sorted(args.runs.glob('*/result.json'))
    if not files:
        parser.error('No final CLI results found.')
    results = []
    for path in files:
        result = json.loads(path.read_text())
        hypothesis_text = normalize(' '.join(row['text'] for row in result['words']))
        hyp_tokens = timed_tokens(result['words'])
        alignment = process_words(' '.join(x['text'] for x in ref_tokens),
                                  ' '.join(x['text'] for x in hyp_tokens))
        starts, ends, deleted, inserted = [], [], [], []
        for chunk in alignment.alignments[0]:
            if chunk.type == 'equal':
                for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx),
                                range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                    starts.append(abs(hyp_tokens[j]['start']-ref_tokens[i]['start']))
                    ends.append(abs(hyp_tokens[j]['end']-ref_tokens[i]['end']))
            elif chunk.type == 'delete':
                deleted.extend(ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx])
            elif chunk.type == 'insert':
                inserted.extend(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx])
        def stats(values):
            if not values:
                return None
            return dict(median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                        p95=float(np.quantile(values, .95)), maximum=float(max(values)))
        def near_boundary(row):
            mid = (row['start']+row['end'])/2
            return mid >= 27.5 and abs(((mid-28+13) % 26)-13) <= .5
        segments = {s['id']: s for s in result['segments']}
        checks = []
        for category in ['decisions', 'actions', 'open_questions']:
            for claim in result['summary'][category]:
                ids = claim['segment_ids']
                valid = bool(ids) and all(i in segments for i in ids)
                source = ' '.join(segments[i]['text'] for i in ids if i in segments)
                quote = normalize(claim.get('quote', ''))
                checks.append(dict(category=category, references_exist=valid,
                                   quote_found=bool(quote) and quote in normalize(source)))
        row = dict(run=path.parent.name, compiler_enabled=result['compiler_enabled'],
                   wer=wer(reference_text, hypothesis_text), cer=cer(reference_text, hypothesis_text),
                   text_sha256=hashlib.sha256(hypothesis_text.encode()).hexdigest(),
                   word_objects=len(result['words']), lexically_matched_words=len(starts),
                   start_seconds=stats(starts), end_seconds=stats(ends),
                   reference_words_near_boundaries=sum(map(near_boundary, ref_tokens)),
                   deleted_words_near_boundaries=sum(map(near_boundary, deleted)),
                   inserted_words_near_boundaries=sum(map(near_boundary, inserted)),
                   summary_reference_checks=checks)
        if baseline:
            row['speaker_turns_exact_match_baseline'] = (result['diarization']['turns'] == baseline['diarization']['turns'])
        results.append(row)
    report = dict(reference_sha256=hashlib.sha256(reference_bytes).hexdigest(), runs=results,
                  protocol='WER/CER: lowercase, remove underscores, punctuation to spaces. Timestamp errors: exact lexical matches to original manual word times. Boundary region +/-0.5s around 28+26*n; diagnostic counts do not establish chunk-caused errors. References/quotes are mechanical checks, not semantic factuality or decision/action coverage. Exact speaker-turn equality only permits reusing a separately established score with that same baseline/protocol.')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(runs=len(results), output_written=True)))


if __name__ == '__main__':
    main()

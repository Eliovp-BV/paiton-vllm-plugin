"""Local, deterministic exports with transcript evidence preserved."""
import json


def clock(seconds, decimal='.'):
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}{decimal}{milliseconds:03}'


def export(result, format='json', speaker_names=None):
    names = speaker_names or {}
    segments = result.get('segments', [])
    if format == 'json':
        return json.dumps({**result, 'speaker_names': names}, ensure_ascii=False, indent=2) + '\n'
    if format in ('srt', 'vtt'):
        blocks = ['WEBVTT\n'] if format == 'vtt' else []
        for index, segment in enumerate(segments, 1):
            speaker = names.get(segment['speaker'], segment['speaker'])
            # Plain cue text: do not let transcript content become subtitle markup.
            text = f"{speaker}: {segment['text']}".replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            decimal = ',' if format == 'srt' else '.'
            blocks.append(f"{index}\n{clock(segment['start'], decimal)} --> {clock(segment['end'], decimal)}\n{text}\n")
        return '\n'.join(blocks)
    if format == 'txt':
        return '\n'.join(f"[{clock(s['start'])}] {names.get(s['speaker'], s['speaker'])}: {s['text']}" for s in segments) + '\n'
    if format == 'summary.txt':
        summary = result.get('summary')
        if summary is None:
            raise ValueError('No summary is available.')
        by_id = {s['id']: s for s in segments}
        lines = ['Generated draft — verify against the recording.', '', summary['overview'], '', 'Topics']
        lines.extend('- ' + topic for topic in summary['topics'])
        for category, title in (('decisions', 'Decisions'), ('actions', 'Action items'), ('open_questions', 'Open questions')):
            lines.extend(['', title])
            if not summary[category]:
                lines.append('None extracted.')
            for claim in summary[category]:
                references = ', '.join(f"{identifier} @{clock(by_id[identifier]['start'])}" for identifier in claim['segment_ids'])
                attribution = f" Owner: {claim['owner']}; deadline: {claim['deadline']}." if category == 'actions' else ''
                lines.extend([f"- {claim['text']}{attribution} [{references}]", f"  Source: {claim['quote']}"])
        return '\n'.join(lines) + '\n'
    raise ValueError('Unsupported export format.')

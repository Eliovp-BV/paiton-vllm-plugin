"""Timestamp stitching and anonymous, uncertainty-preserving speaker attribution."""
import math


def token_words(tokens):
    words=[]
    for token in tokens:
        text=token['token']
        start,end=float(token['start']),float(token['end'])
        if not isinstance(text,str) or not math.isfinite(start+end) or start<0 or end<start:
            raise ValueError('Invalid ASR token timestamp.')
        if not text:continue
        if not words or text[0].isspace():
            words.append(dict(text=text.lstrip(),start=start,end=end))
        else:
            words[-1]['text']+=text;words[-1]['end']=max(words[-1]['end'],end)
    return [w for w in words if w['text']]


def stitch_words(words, chunk):
    """Assign whole words by midpoint to disjoint overlap cores.

This avoids deliberate overlap duplication. It cannot guarantee identical ASR
hypotheses at boundaries; benchmark boundary edits against reference words.
"""
    result=[]
    for word in words:
        start,end=word['start']+chunk.start,word['end']+chunk.start
        middle=(start+end)/2
        if chunk.core_start<=middle<chunk.core_end:
            result.append(dict(text=word['text'],start=max(0,start),end=min(end,chunk.end)))
    return result


def assign_speaker(word,turns):
    scores={}
    for turn in turns:
        overlap=max(0,min(word['end'],turn['end'])-max(word['start'],turn['start']))
        if overlap>0:scores[turn['speaker']]=scores.get(turn['speaker'],0)+overlap
    candidates=sorted(scores,key=scores.get,reverse=True)
    duration=max(.02,word['end']-word['start'])
    if not candidates:return dict(speaker='UNKNOWN',speakers=[],speaker_uncertain=True)
    overlapping=[s for s in candidates if scores[s]>=.2*duration]
    uncertain=scores[candidates[0]]<.5*duration or len(overlapping)>1
    return dict(speaker=candidates[0] if not uncertain else 'UNCERTAIN',speakers=overlapping or candidates[:1],speaker_uncertain=uncertain)


def segments_from_words(words,turns=None):
    segments=[]
    for word in words:
        attribution=assign_speaker(word,turns) if turns is not None else dict(speaker='UNKNOWN',speakers=[],speaker_uncertain=True)
        if (segments and segments[-1]['speaker']==attribution['speaker'] and
            segments[-1]['speakers']==attribution['speakers'] and
            word['start']-segments[-1]['end']<1 and word['end']-segments[-1]['start']<15 and
            not segments[-1]['text'].endswith(('.', '?', '!'))):
            segments[-1]['text']+=' '+word['text'];segments[-1]['end']=max(segments[-1]['end'],word['end'])
        else:
            segments.append(dict(id=f's{len(segments)+1:06d}',start=word['start'],end=word['end'],text=word['text'],final=True,**attribution))
    return segments

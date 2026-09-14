from paiton_meeting.transcript import token_words,stitch_words,assign_speaker
from paiton_meeting.audio import AudioChunk
import numpy as np

def test_subwords_and_punctuation_stay_together():
    tokens=[dict(token=t,start=s,end=e) for t,s,e in [('Hel',1,1.1),('lo',1.1,1.2),(',',1.2,1.2),(' world',1.3,1.5)]]
    assert [w['text'] for w in token_words(tokens)]==['Hello,','world']

def test_boundary_word_is_owned_once():
    words=[dict(text='boundary',start=27.8,end=28.2)]
    a=AudioChunk(np.empty(0),0,30,0,28)
    b=AudioChunk(np.empty(0),26,56,28,54)
    assert stitch_words(words,a)==[]
    assert len(stitch_words([dict(text='boundary',start=1.8,end=2.2)],b))==1

def test_overlap_not_false_identity():
    word=dict(start=1,end=2,text='yes')
    assert assign_speaker(word,[])['speaker']=='UNKNOWN'
    turns=[dict(start=0,end=3,speaker='SPEAKER_00'),dict(start=1,end=2,speaker='SPEAKER_01')]
    got=assign_speaker(word,turns)
    assert got['speaker']=='UNCERTAIN' and len(got['speakers'])==2

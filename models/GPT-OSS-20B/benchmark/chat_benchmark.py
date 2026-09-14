"""Natural-stop chat timings, with reasoning and answer streams retained."""
import argparse
import json
from pathlib import Path
from api_checks import call
from benchmark import distribution

PROMPTS = [
    'Explain how binary search works in two sentences.',
    'What is 17 plus 25? Return only the integer.',
    'Write a Python function palindrome(s) that ignores spaces and letter case when checking whether a string is a palindrome. Give a concise answer.',
]


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8020');p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=4);a=p.parse_args()
    results=[];warmups=[]
    for index,prompt in enumerate(PROMPTS):
        body={'model':'gpt-oss-20b','messages':[{'role':'user','content':prompt}],'reasoning_effort':'low','temperature':0,'seed':1201,'max_tokens':512,'stream':True,'return_token_ids':True,'stream_options':{'include_usage':True}}
        warmups.append(call(a.url,body))
        for repeat in range(a.repeats):
            result=call(a.url,body);result.update(prompt_index=index,repeat=repeat);results.append(result)
            a.out.write_text(json.dumps({'warmup':warmups,'requests':results},indent=2))
    summaries=[]
    for index in range(len(PROMPTS)):
        records=[r for r in results if r['prompt_index']==index]
        summaries.append({'prompt_index':index,'n':len(records),'no_final_answer':sum(not r['content'].strip() for r in records),
            'e2e_s':distribution([r['elapsed_s'] for r in records]),
            'first_answer_s':distribution([r['first_answer_s'] for r in records if r['first_answer_s'] is not None]),
            'completion_tokens':distribution([r['usage']['completion_tokens'] for r in records])})
    a.out.write_text(json.dumps({'summary':summaries,'warmup':warmups,'requests':results},indent=2));print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()

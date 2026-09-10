"""Audit paired requests and tokenized prompts before comparing task outcomes."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('stock', type=Path)
p.add_argument('paiton', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
stock = json.loads(a.stock.read_text())
paiton = json.loads(a.paiton.read_text())
left = {row['id']: row for row in stock['cases']}
right = {row['id']: row for row in paiton['cases']}
assert left.keys() == right.keys(), 'Evaluation case sets differ'
rows = []
for key in left:
    x, y = left[key], right[key]
    assert x['request'] == y['request'], f'Requests differ: {key}'
    px = x['response'].get('prompt_token_ids')
    py = y['response'].get('prompt_token_ids')
    assert px and px == py, f'Tokenized prompts differ: {key}'
    cx, cy = x['response']['choices'][0], y['response']['choices'][0]
    rows.append({'id': key, 'stock_passed': x['passed'], 'paiton_passed': y['passed'],
                 'identical_generated_tokens': cx.get('token_ids') == cy.get('token_ids'),
                 'identical_content': cx['message'].get('content') == cy['message'].get('content')})
result = {'matched_requests_and_prompt_ids': True, 'total': len(rows),
          'stock_passed': sum(r['stock_passed'] for r in rows),
          'paiton_passed': sum(r['paiton_passed'] for r in rows),
          'regressions': [r['id'] for r in rows if r['stock_passed'] and not r['paiton_passed']],
          'improvements': [r['id'] for r in rows if not r['stock_passed'] and r['paiton_passed']],
          'cases': rows}
a.out.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))

"""Fetch a tiny, pinned FLEURS comparison without downloading whole archives."""
import argparse,csv,hashlib,io,json,tarfile,urllib.request
from pathlib import Path
revision='70bb2e84b976b7e960aa89f1c648e09c59f894dd'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
root=Path(parser.parse_args().output);root.mkdir(parents=True,exist_ok=True)
manifest=dict(dataset='google/fleurs',revision=revision,license='CC-BY-4.0',selection='First three regular audio members with a reference in each pinned test archive; no model-based selection.',samples=[])
for language in ('en_us','de_de','fr_fr','es_419'):
 base=f'https://huggingface.co/datasets/google/fleurs/resolve/{revision}/data/{language}'
 with urllib.request.urlopen(base+'/test.tsv',timeout=60) as response:tsv=response.read().decode()
 rows=list(csv.reader(io.StringIO(tsv),delimiter='\t'));by_name={r[1]:r for r in rows}
 found=0;scanned=0
 with urllib.request.urlopen(base+'/audio/test.tar.gz',timeout=60) as response,tarfile.open(fileobj=response,mode='r|gz') as archive:
  for member in archive:
   scanned+=member.size
   if scanned>32*1024*1024:raise RuntimeError('Selection exceeded bounded archive scan.')
   name=Path(member.name).name
   if not member.isfile() or name not in by_name:continue
   if member.size>8*1024*1024:raise RuntimeError('Unexpected sample size.')
   audio=archive.extractfile(member).read();assert len(audio)==member.size
   path=root/f'{language}-{found}.wav'
   if path.exists():assert path.read_bytes()==audio
   else:path.write_bytes(audio)
   row=by_name[name]
   manifest['samples'].append(dict(language=language,path=path.name,source_member=member.name,id=row[0],reference=row[3],raw_reference=row[2],sample_count=int(row[5]),gender=row[6],sha256=hashlib.sha256(audio).hexdigest()))
   found+=1
   if found==3:break
 assert found==3
 print(language,'samples',found,flush=True)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')

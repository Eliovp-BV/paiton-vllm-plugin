"""Lossless, bounded-RAM Diffusers-to-Comfy Wan key conversion.

Only the safetensors header changes. Payload bytes, dtypes and tensor shapes
are preserved; no quantization or calibration is performed.
"""
import hashlib,json,os,struct
from pathlib import Path


def rename(key):
    for old,new in [('condition_embedder.text_embedder.linear_1.','text_embedding.0.'),
                    ('condition_embedder.text_embedder.linear_2.','text_embedding.2.'),
                    ('condition_embedder.time_embedder.linear_1.','time_embedding.0.'),
                    ('condition_embedder.time_embedder.linear_2.','time_embedding.2.'),
                    ('condition_embedder.time_proj.','time_projection.1.'),
                    ('proj_out.','head.head.')]:
        if key.startswith(old):return new+key[len(old):]
    if key=='scale_shift_table':return 'head.modulation'
    if key.startswith('blocks.'):
        for old,new in [('.attn1.','.self_attn.'),('.attn2.','.cross_attn.'),
                        ('.to_out.0.','.o.'),('.to_q.','.q.'),('.to_k.','.k.'),('.to_v.','.v.'),
                        ('.ffn.net.0.proj.','.ffn.0.'),('.ffn.net.2.','.ffn.2.'),
                        ('.norm2.','.norm3.'),('.scale_shift_table','.modulation')]:
            key=key.replace(old,new)
    return key


def convert(source,destination):
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError(destination)
    temporary=destination.with_name(destination.name+'.partial')
    payload_hash=hashlib.sha256();out_hash=hashlib.sha256()
    with source.open('rb') as src:
        size=struct.unpack('<Q',src.read(8))[0]
        if size>4*1024**2:raise ValueError('Unexpected Wan safetensors header size')
        header=json.loads(src.read(size));converted={}
        for key,value in header.items():
            if key=='__metadata__':continue
            new=rename(key)
            if new in converted:raise ValueError(f'Duplicate converted key: {new}')
            converted[new]=value
        if len(converted)!=825 or converted['patch_embedding.weight']['shape']!=[3072,48,1,2,2]:
            raise ValueError('Unexpected FastWan TI2V checkpoint geometry')
        converted['__metadata__']={'conversion':'Diffusers to Comfy key-only mapping; tensor payload unchanged'}
        encoded=json.dumps(converted,sort_keys=True,separators=(',',':')).encode()
        encoded+=b' '*((-len(encoded))%8)
        prefix=struct.pack('<Q',len(encoded))+encoded
        destination.parent.mkdir(parents=True,exist_ok=True)
        with temporary.open('xb') as dst:
            dst.write(prefix);out_hash.update(prefix)
            while chunk:=src.read(8*1024**2):
                dst.write(chunk);payload_hash.update(chunk);out_hash.update(chunk)
            dst.flush();os.fsync(dst.fileno())
    temporary.rename(destination)
    return {'payload_sha256':payload_hash.hexdigest(),'converted_sha256':out_hash.hexdigest(),
            'bytes':destination.stat().st_size,'tensors':len(converted)-1,'calibration':'none; no numerical conversion',
            'excluded_components':'none; all transformer tensor payloads preserved'}

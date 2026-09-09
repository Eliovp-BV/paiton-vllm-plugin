"""Pinned, persistent checkpoint acquisition; original tensors stay intact."""
import argparse
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from .convert import convert

COMFY_REPO='Comfy-Org/Wan_2.2_ComfyUI_Repackaged'
COMFY_REV='c4f60d30c55a624e35427060fdd217579a6c1d77'
FAST_REPO='FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers'
FAST_REV='3e187042a324f6f5fb68fd22110a78725253de8f'


def download(root, preset='base', offline=False):
    root=Path(root).resolve()
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    os.environ.setdefault('HF_XET_HIGH_PERFORMANCE','0')
    records=[]
    def fetch(repo,rev,name):
        path=Path(hf_hub_download(repo,name,revision=rev,cache_dir=root/'cache/hub',local_files_only=offline))
        records.append({'repo':repo,'revision':rev,'file':name,'bytes':path.stat().st_size})
        return path
    def link(path,relative):
        dest=root/'models'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.is_symlink() and dest.resolve()==path.resolve():return
        if dest.exists() or dest.is_symlink():raise FileExistsError(f'Refusing to replace {dest}')
        dest.symlink_to(os.path.relpath(path,dest.parent))
    names=['text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors','vae/wan2.2_vae.safetensors']
    if preset in ('base','all'):names.append('diffusion_models/wan2.2_ti2v_5B_fp16.safetensors')
    for name in names:link(fetch(COMFY_REPO,COMFY_REV,'split_files/'+name),name)
    if preset in ('fast','all'):
        original=fetch(FAST_REPO,FAST_REV,'transformer/diffusion_pytorch_model.safetensors')
        destination=root/'models/diffusion_models/fastwan22_5b_fullattn_comfy_bf16.safetensors'
        destination.parent.mkdir(parents=True,exist_ok=True)
        manifest=destination.with_suffix('.conversion.json')
        if not destination.exists():
            report=convert(original,destination)
            report.update(repo=FAST_REPO,revision=FAST_REV)
            manifest.write_text(json.dumps(report,indent=2)+'\n')
        elif not manifest.exists():
            raise FileExistsError(f'Existing converted checkpoint lacks provenance: {destination}')
        else:
            report=json.loads(manifest.read_text())
            expected={'repo':FAST_REPO,'revision':FAST_REV,'tensors':825,
                      'bytes':9999659744,
                      'converted_sha256':'4167885e2463373d94b06939cebe15fc950e857f919bfa53793ab39b00741782'}
            if any(report.get(k)!=v for k,v in expected.items()) or destination.stat().st_size!=expected['bytes']:
                raise ValueError(f'Converted checkpoint provenance or size mismatch: {destination}')
    (root/f'download-{preset}.json').write_text(json.dumps(records,indent=2)+'\n')
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--preset',choices=['base','fast','all'],default='base');p.add_argument('--offline',action='store_true')
    a=p.parse_args();print(json.dumps(download(a.data,a.preset,a.offline),indent=2))

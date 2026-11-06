"""Download the stock, unmodified weights at pinned revisions and lay them out for idol-sdk.

Nothing is redistributed by this package: every file comes from its upstream Hugging Face repository with your own
account. DINOv3 is under the DINOv3 License; accept it on Hugging Face first (timm copy or the facebook repo).
"""
import json
import os
import shutil
import subprocess
import sys

from huggingface_hub import snapshot_download

IDOL = ('speridlabs/idol', '2731c2291fc65a797c5cb627a5e5526fc1490902')
IDOL_FILES = ['config.json', 'denoiser.safetensors', 'decoder.safetensors', 'decoder_gs.safetensors',
              'birefnet.safetensors', 'README.md']
DINOV3_TIMM = ('timm/vit_huge_plus_patch16_dinov3.lvd1689m', 'dd0addc09788111fa893d24799a44744ba022eee')
# exactly the bf16 round-to-nearest-even of the FLUX.2-dev VAE that idol-sdk loads (in bf16) by default; Apache-2.0
VAE = ('black-forest-labs/FLUX.2-klein-4B', 'e7b7dc27f91deacad38e78976d1f2b499d76a294')


def prepare(root):
    """Returns the runtime folder: stock idol files (symlinked) + config.json pointing at the local encoders."""
    run = os.path.join(root, 'idol-runtime')
    if os.path.exists(os.path.join(run, 'config.json')):
        return run
    idol = snapshot_download(IDOL[0], revision=IDOL[1], allow_patterns=IDOL_FILES)
    vae = snapshot_download(VAE[0], revision=VAE[1], allow_patterns=['vae/*'])
    dino = os.path.join(root, 'dinov3-vith16plus')
    if not os.path.exists(os.path.join(dino, 'model.safetensors')):
        src = snapshot_download(DINOV3_TIMM[0], revision=DINOV3_TIMM[1], allow_patterns=['model.safetensors',
                                                                                         'config.json', 'LICENSE.md'])
        # key rename + qkv split only (values untouched), checked against timm's own forward
        subprocess.run([sys.executable, '-m', 'paiton_idol.convert_dinov3', src, dino + '.tmp'], check=True)
        os.replace(dino + '.tmp', dino)
    tmp = run + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in IDOL_FILES[1:5]:
        os.symlink(os.path.realpath(os.path.join(idol, f)), os.path.join(tmp, f))
    cfg = json.load(open(os.path.join(idol, 'config.json')))
    cfg['image_encoders'] = {'dinov3': dino, 'flux2_vae': vae}
    json.dump(cfg, open(os.path.join(tmp, 'config.json'), 'w'), indent=2)
    os.replace(tmp, run)
    return run

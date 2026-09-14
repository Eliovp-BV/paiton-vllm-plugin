"""Generate and benchmark native Wan clips on one gfx1201 GPU."""
import argparse,datetime,os,runpy,sys,uuid
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['download','generate','benchmark','inspect','compare'])
    p.add_argument('--engine',choices=['stock','paiton'],default=None)
    p.add_argument('--preset',choices=['base','fast','all'],default='fast')
    p.add_argument('--duration',choices=[2,5],type=int,default=2)
    p.add_argument('--resolution',choices=[480,720],type=int,default=None)
    p.add_argument('--portrait',action='store_true')
    a,extra=p.parse_known_args()
    if a.command in ('download','inspect','compare'):
        module={'download':'download','inspect':'inspect_clip','compare':'compare'}[a.command]
        sys.argv=[module,*(['--preset',a.preset,'--data',os.environ.get('PAITON_WAN_DATA','/data')] if a.command=='download' else []),*extra]
        runpy.run_module('paiton_wan.'+module,run_name='__main__');return
    if a.engine is None:a.engine='paiton' if a.preset=='fast' else 'stock'
    if a.resolution is None:a.resolution=720 if a.preset=='fast' else 480
    if a.preset=='all':p.error('all is a download-only preset')
    if a.preset=='base' and a.resolution!=480:p.error('The base TI2V package is qualified at 480 class; use fast for 720-class text generation')
    if a.preset=='fast' and a.portrait:p.error('FastWan is qualified in landscape; use base for portrait image input')
    package=Path(__file__).resolve().parents[1]
    w,h=(832,480) if a.resolution==480 else (1280,704)
    if a.portrait:w,h=h,w
    defaults=['--native-fp8','--miopen','--compile','--width',str(w),'--height',str(h),
              '--frames',str(a.duration*24+1),'--runs','6' if a.command=='benchmark' else '1',
              '--warmups','2' if a.command=='benchmark' else '0','--paiton-runtime',str(package)]
    if '--output' not in extra:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        defaults+=['--output',str(Path(os.environ.get('PAITON_WAN_OUTPUTS','/outputs'))/f'{a.preset}-{a.engine}-{stamp}')]
    if a.preset=='fast':defaults+=['--fastwan']
    if a.engine=='paiton':defaults+=['--paiton-vae-artifact',str(package/'artifacts/wan22_ti2v_fusions_gfx1201.so')]
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
    sys.argv=['qualification',*defaults,*extra]
    runpy.run_module('paiton_wan.qualification',run_name='__main__')

if __name__=='__main__':main()

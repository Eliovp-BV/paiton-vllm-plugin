"""Compare decoded clips without treating pixel similarity as quality proof."""
import argparse,json,math
from pathlib import Path
import av
import numpy as np


def compare(left,right):
    errors=[];temporal=[];previous=[None,None];count=0
    with av.open(str(left)) as a,av.open(str(right)) as b:
        sa,sb=a.streams.video[0],b.streams.video[0]
        if (sa.width,sa.height,sa.average_rate)!=(sb.width,sb.height,sb.average_rate):
            raise ValueError('Clip geometry or frame rate differs')
        ia,ib=iter(a.decode(video=0)),iter(b.decode(video=0))
        while True:
            fa,fb=next(ia,None),next(ib,None)
            if fa is None or fb is None:
                if fa is not None or fb is not None:raise ValueError('Decoded frame counts differ')
                break
            if fa.time!=fb.time:raise ValueError('Frame timestamps differ')
            x,y=fa.to_ndarray(format='rgb24').astype(np.float32),fb.to_ndarray(format='rgb24').astype(np.float32)
            errors.append(float(np.mean((x-y)**2)))
            if previous[0] is not None:
                temporal.append([float(np.mean(np.abs(x-previous[0]))),float(np.mean(np.abs(y-previous[1])))])
            previous=[x,y];count+=1
    mse=float(np.mean(errors))
    return {'decoded_frames':count,'mse_8bit':mse,'psnr_db':10*math.log10(255**2/mse) if mse else None,
            'pixel_exact':mse==0,'per_frame_mse':errors,'adjacent_frame_mae_left_right':temporal,
            'interpretation':'Encoded pixel similarity only; inspect prompt adherence, motion and subject consistency separately.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare(a.left,a.right);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if not isinstance(v,list)},indent=2))

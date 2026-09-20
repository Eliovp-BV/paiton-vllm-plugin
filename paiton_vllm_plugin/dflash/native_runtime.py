"""External raw-pointer dispatch; native artifacts contain no framework code."""
import ctypes as C
import hashlib
import json
import os
import sys
from pathlib import Path
from .native_policy import Policy

ROOT=Path(os.environ.get('PAITON_DFLASH_NATIVE_BUNDLE','/opt/paiton/runtime/dflash-down')).resolve(strict=True)
_manifest=json.loads((ROOT/'manifest.json').read_text())
if set(_manifest['artifacts'])!={'draft_fp8.so','silu.so'}:
    raise ValueError('Expected reviewed native DFlash artifact set')
if set(_manifest['available_features'])!={'down_projection','silu_quant'}:
    raise ValueError('Unsupported native DFlash features')
POLICY=Policy.from_environment(os.environ,available_features=_manifest['available_features'],
    artifact_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest())
_down=_silu=None
_producer_calls=0

def policy_snapshot():
    return {'policy':POLICY.snapshot(),'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'policy_sha256':hashlib.sha256(Path(__file__).with_name('native_policy.py').read_bytes()).hexdigest(),
            'mlp_adapter_sha256':hashlib.sha256(Path(__file__).with_name('native_mlp.py').read_bytes()).hexdigest()}

def load(torch):
    global _down,_silu
    if _down is not None:return
    for name,digest in _manifest['artifacts'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('DFlash artifact mismatch')
    if torch.cuda.device_count()!=1 or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1201'):
        raise RuntimeError('DFlash native bundle requires one gfx1201')
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Initialize DFlash before capture')
    down=C.CDLL(str(ROOT/'draft_fp8.so'),mode=C.RTLD_LOCAL)
    down.paiton_draft_fp8_abi_version.restype=C.c_int
    if down.paiton_draft_fp8_abi_version()!=2:raise RuntimeError('DFlash ABI mismatch')
    down.paiton_draft_fp8_run.argtypes=[C.c_void_p]*5+[C.c_int]*3+[C.c_void_p]
    down.paiton_draft_fp8_run.restype=C.c_int
    silu=C.CDLL(str(ROOT/'silu.so'),mode=C.RTLD_LOCAL)
    silu.paiton_dflash_silu_quant.argtypes=[C.c_void_p]*3+[C.c_int,C.c_void_p]
    silu.paiton_dflash_silu_quant.restype=C.c_int
    _down,_silu=down,silu
    print('[paiton.dflash] '+json.dumps(policy_snapshot(),sort_keys=True),file=sys.stderr,flush=True)

def producer(input):
    global _producer_calls
    import torch
    if not POLICY.allows('silu_quant'):raise RuntimeError('Native producer disabled')
    m=input.shape[0]
    if not (input.is_cuda and input.is_contiguous() and input.dtype==torch.bfloat16
            and tuple(input.shape)==(m,34816) and 1<=m<=4096):
        raise RuntimeError('Native producer input contract')
    load(torch)
    q=torch.empty((m,17408),device=input.device,dtype=torch.float8_e4m3fn)
    scales=torch.empty((m,136),device=input.device,dtype=torch.float32)
    rc=_silu.paiton_dflash_silu_quant(input.data_ptr(),q.data_ptr(),scales.data_ptr(),m,torch.cuda.current_stream(input.device).cuda_stream)
    if rc:raise RuntimeError(f'DFlash producer launch {rc}')
    _producer_calls+=1
    if _producer_calls==1:print("[paiton.dflash] compiled native producer path executed; rows="+str(m),file=sys.stderr,flush=True)
    return q,scales

def run(A,B,As,Bs,N,K,reference):
    import torch
    eligible=(POLICY.allows('down_projection') and (tuple(A.shape),N,K)==((8,17408),5120,17408)
        and tuple(B.shape)==(320,278528) and tuple(As.shape)==(8,136) and tuple(Bs.shape)==(40,136)
        and A.dtype==B.dtype==torch.float8_e4m3fn and As.dtype==Bs.dtype==torch.float32
        and all(x.is_cuda and x.device==A.device and x.is_contiguous() for x in (A,B,As,Bs)))
    if not eligible:return reference()
    load(torch)
    output=torch.empty((8,5120),device=A.device,dtype=torch.bfloat16)
    rc=_down.paiton_draft_fp8_run(A.data_ptr(),B.data_ptr(),As.data_ptr(),Bs.data_ptr(),output.data_ptr(),8,N,K,torch.cuda.current_stream(A.device).cuda_stream)
    if rc:raise RuntimeError(f'DFlash down launch {rc}')
    return output

"""External pointer adapter. Native arithmetic stays in the private HIP artifact."""
import ctypes as C
import functools,hashlib,importlib.abc,json,os,sys
from pathlib import Path
from paiton_prefill_attention_policy import Policy
TARGET='radiance_r4d_attn'
class Args(C.Structure):
 _fields_=[(n,C.c_void_p) for n in ('q','kv','block_table','seqused_k','out','k_descale','v_descale','q_descale','scratch')]+[(n,C.c_int) for n in ('num_seqs','q_len','q_heads','kv_heads','head_dim','block_size','max_blocks')]+[('kv_block_stride',C.c_long),('kv_head_stride',C.c_long),('scale',C.c_float),('splits',C.c_int),('max_ctx',C.c_int)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make_args(v):
 q,kv,bt,sl,out,ks,vs,scratch,n,l,qh,kh,d,bs,mb,bstride,hstride,scale,splits,ctx,stream=v
 return Args(q,kv,bt,sl,out,ks,vs,0,scratch,n,l,qh,kh,d,bs,mb,bstride,hstride,scale,splits,ctx)
def bind(backend):
 if getattr(backend,'_paiton_prefill96',False):raise RuntimeError('Duplicate prefill override')
 policy=Policy.from_environment(os.environ)
 root=Path(os.environ.get('PAITON_PREFILL_ATTENTION96_BUNDLE','/opt/paiton/runtime/prefill-attention96'))
 m=json.loads((root/'manifest.json').read_text());external=json.loads((root/'external-source-receipts.json').read_text())
 if m['version']!=policy.version or m['arch']!='gfx1201' or m['fp8_mode']!=3 or m['kv_tile']!=96:raise RuntimeError('Attention contract changed')
 import paiton_prefill_attention_policy as policy_module
 if sha(policy_module.__file__)!=m['policy_sha256']:raise RuntimeError('Attention policy changed')
 if sha(backend.__file__)!=external['backend_sha256'] or sha(backend.r4d.__file__)!=external['r4d_sha256']:raise RuntimeError('Attention reference changed')
 if sha(root/m['file'])!=m['sha256']:raise RuntimeError('Attention artifact changed')
 mode=os.RTLD_NOW|os.RTLD_LOCAL|getattr(os,'RTLD_DEEPBIND',0)
 lib=C.CDLL(str(root/m['file']),mode=mode);lib.paiton_prefill_attention_abi_size.restype=C.c_size_t
 if lib.paiton_prefill_attention_abi_size()!=C.sizeof(Args):raise RuntimeError('Attention ABI changed')
 fn=lib.paiton_prefill_attention;fn.argtypes=[C.POINTER(Args),C.c_void_p];fn.restype=C.c_int
 shadow=os.environ.get('PAITON_PREFILL_ATTENTION96_SHADOW','0')=='1';audit=None;ready=False;calls=0;fallbacks=0;shapes=set()
 if shadow:
  audit=C.CDLL(str(root/'audit.so'),mode=mode)
  if sha(root/'audit.so')!=m['audit_sha256']:raise RuntimeError('Audit artifact changed')
  audit.audit_init.argtypes=[C.c_void_p];audit.audit_report.argtypes=[C.POINTER(C.c_ulonglong)]
  audit.audit_outputs.argtypes=[C.c_void_p]*5+[C.c_size_t,C.c_int,C.c_void_p]
 def initialize(stream):
  nonlocal ready
  if ready:return
  if lib.paiton_prefill_attention_target_ok()!=1:raise RuntimeError('Require gfx1201')
  if shadow:
   rc=audit.audit_init(stream)
   if rc:raise RuntimeError('Attention audit initialization failed:'+str(rc))
  ready=True
 original=backend._PREFILL[0]
 @functools.wraps(original)
 def launch(*v):
  nonlocal calls,fallbacks
  if len(v)!=21:return original(*v)
  initialize(v[-1])
  shape=tuple(v[8:20])
  if shape not in shapes and len(shapes)<64:
   shapes.add(shape);print('[paiton.prefill_attention96] raw_geometry='+json.dumps(shape)+' eligible='+str(policy.eligible(v)),flush=True)
  if not policy.eligible(v):fallbacks+=1;return original(*v)
  a=make_args(v);output=None
  if shadow:
   # Existing external backend allocator only. No tensor arithmetic or kernel generation.
   output=backend.torch.empty((a.num_seqs*a.q_len,a.q_heads,a.head_dim),dtype=backend.torch.bfloat16,device='cuda')
   original(*v);a.out=output.data_ptr()
  rc=fn(C.byref(a),v[-1])
  if rc:raise RuntimeError('Native attention launch failed:'+str(rc))
  if shadow:
   rc=audit.audit_outputs(v[0],v[4],a.out,v[5],v[6],a.num_seqs*a.q_len*a.q_heads*a.head_dim,a.num_seqs*a.kv_heads,v[-1])
   if rc:raise RuntimeError('Attention audit launch failed:'+str(rc))
  calls+=1
  if calls in (1,256):print('[paiton.prefill_attention96] native launches='+str(calls)+' shadow='+str(shadow),flush=True)
 if shadow:
  # The pinned builder already allocates its GPU scratch at construction.
  # Initialize diagnostic counters there, outside capture even with warm AOT
  # caches that can skip the first eager model execution.
  builder_init=backend.R4DAttentionMetadataBuilder.__init__
  @functools.wraps(builder_init)
  def initialize_builder(self,*args,**kwargs):
   result=builder_init(self,*args,**kwargs)
   initialize(backend.torch.cuda.current_stream().cuda_stream)
   return result
  backend.R4DAttentionMetadataBuilder.__init__=initialize_builder
  # Initialize counters on the first GPU attention call, before graph capture,
  # even when startup only executes decode and contains no eligible prefill.
  decode=backend._DECODE[0]
  @functools.wraps(decode)
  def initialize_decode(*v):
   if len(v)==21:initialize(v[-1])
   return decode(*v)
  backend._DECODE=(initialize_decode,*backend._DECODE[1:])
  import threading,time
  evidence=Path('/evidence')
  def report():
   for stage in ('warmup','final'):
    while not (evidence/(stage+'-audit-request')).exists():time.sleep(.2)
    if not ready:return
    counters=(C.c_ulonglong*6)();rc=audit.audit_report(counters)
    names=('completed_comparisons','different_elements','reference_nonfinite','candidate_nonfinite','query_nonfinite','scale_nonfinite')
    result=dict(zip(names,counters));result.update(hip_status=rc,python_launches=calls,fallbacks=fallbacks,shapes=sorted(shapes))
    p=evidence/(stage+'-audit.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result)+'\n');tmp.replace(p)
  threading.Thread(target=report,name='bounded-prefill-shadow-audit',daemon=True).start()
 launch._handles=(lib,audit,original)
 backend._PREFILL=(launch,*backend._PREFILL[1:]);backend._paiton_prefill96=True
 print('[paiton.prefill_attention96] installed '+policy.version+' shadow='+str(shadow),flush=True)
class Loader:
 def __init__(self,original):self.original=original
 def __getattr__(self,name):return getattr(self.original,name)
 def create_module(self,spec):return self.original.create_module(spec)
 def exec_module(self,module):self.original.exec_module(module);bind(module)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!=TARGET:return None
  after=False
  for finder in list(sys.meta_path):
   if finder is self:after=True;continue
   if not after:continue
   spec=finder.find_spec(fullname,path,target)
   if spec is not None:
    if not spec.loader:raise ImportError('Missing attention backend loader')
    spec.loader=Loader(spec.loader);return spec
  raise ImportError('Missing attention backend')
def install():
 policy=Policy.from_environment(os.environ)
 if not policy.enabled:return
 if TARGET in sys.modules:raise RuntimeError('Install attention hook before backend import')
 sys.meta_path.insert(0,Finder())

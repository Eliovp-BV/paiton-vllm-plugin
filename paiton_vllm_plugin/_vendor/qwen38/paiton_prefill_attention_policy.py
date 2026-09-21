"""Pure host contract; no frameworks or runtime imports."""
from dataclasses import dataclass
FLAG='PAITON_PREFILL_ATTENTION96'
@dataclass(frozen=True)
class Policy:
    enabled: bool
    version: str = 'prefill-attention96-v2'
    @classmethod
    def from_environment(cls,env):
        value=env.get(FLAG,'0')
        if value not in ('0','1'):raise ValueError(FLAG+' must be0or1')
        if value=='1' and env.get('R4D_ATTN_FP8')!='3':
            raise ValueError('Require the qualified deployed FP8QK/PV mode3')
        return cls(value=='1')
    def eligible(self,v):
        if not self.enabled or len(v)!=21:return False
        q,kv,bt,sl,out,ks,vs,scratch,n,l,qh,kh,d,bs,mb,bstride,hstride,scale,splits,ctx,stream=v
        return (all((q,kv,bt,sl,out)) and n==1 and 4089<=l<=4096
                and (qh,kh,d,bs)==(24,4,256,16) and 8192<ctx<=32768 and mb>=(ctx+15)//16
                and (bstride,hstride)==(32768,8192) and scale==.0625 and splits==0
                and all(ptr%16==0 for ptr in (q,kv,out)) and out not in (q,kv))

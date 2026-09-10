"""Inspect page residency and control only a task-owned checkpoint copy."""
import ctypes
import os
from pathlib import Path
import time


def residency(path):
    page = os.sysconf('SC_PAGE_SIZE')
    size = path.stat().st_size
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
    with path.open('rb') as stream:
        address = libc.mmap(None,size,0,1,stream.fileno(),0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(),'mmap failed')
        try:
            pages = (size+page-1)//page
            values = (ctypes.c_ubyte*pages)()
            if libc.mincore(address,size,values):
                raise OSError(ctypes.get_errno(),'mincore failed')
            return dict(bytes=size,pages=pages,resident_pages=sum(bool(v&1) for v in values))
        finally:
            libc.munmap(address,size)


def prepare(snapshot,mode,task_root):
    snapshot = Path(snapshot).resolve()
    files = sorted(snapshot.glob('*.safetensors'))
    started = time.perf_counter()
    if mode == 'cold':
        private = (Path(task_root)/'isolated-checkpoints').resolve()
        if not snapshot.is_relative_to(private) or any(not file.resolve().is_relative_to(private) for file in files):
            raise ValueError('Cold-cache control requires a private, nonsymlinked checkpoint copy')
        for file in files:
            with file.open('rb') as stream:
                os.fsync(stream.fileno())
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    elif mode == 'warm':
        buffer = bytearray(8*1024**2)
        for file in files:
            with file.open('rb',buffering=0) as stream:
                while stream.readinto(buffer):
                    pass
    else:
        raise ValueError('Choose cold or warm preparation')
    status = {file.name:residency(file) for file in files}
    if mode == 'cold' and any(v['resident_pages']/v['pages']>.01 for v in status.values()):
        raise RuntimeError('Private checkpoint did not become cold; do not label this a cold-storage trial')
    return dict(mode=mode,preparation_seconds=time.perf_counter()-started,files=status,
                boundary='Page residency immediately before container launch; no system-wide cache dropping')

import av,ctypes,json,pathlib,subprocess
maps=pathlib.Path('/proc/self/maps').read_text()
paths=sorted({line.split()[-1] for line in maps.splitlines() if '/opt/meeting-media/lib/' in line})
codec=next(p for p in paths if '/libavcodec.so.' in p)
lib=ctypes.CDLL(codec)
lib.avcodec_license.restype=ctypes.c_char_p
lib.avcodec_configuration.restype=ctypes.c_char_p
license=lib.avcodec_license().decode();config=lib.avcodec_configuration().decode()
assert '--disable-gpl' in config and '--disable-nonfree' in config
assert '--enable-gpl' not in config and '--enable-nonfree' not in config
assert '--enable-libx264' not in config and '--enable-libx265' not in config
assert 'libx264' not in maps and 'libx265' not in maps
for name in ('libx264','libx265'):
 try: av.Codec(name,'w')
 except av.codec.codec.UnknownCodecError:pass
 else:raise AssertionError(f'Unexpected encoder {name}')
closure=subprocess.check_output(['ldd',codec],text=True)
assert 'libx264' not in closure and 'libx265' not in closure
sources=pathlib.Path('/opt/meeting-media/share/paiton-media')
import hashlib
rows=json.loads((sources/'sources.json').read_text())
assert len(rows)==6
assert all(hashlib.sha256((sources/'sources'/r['filename']).read_bytes()).hexdigest()==r['sha256'] for r in rows)
print(json.dumps(dict(status='pass',av_version=av.__version__,license=license,configuration=config,loaded_media_libraries=[pathlib.Path(p).name for p in paths],source_files_verified=len(rows),codec_dependency_closure=closure),indent=2))

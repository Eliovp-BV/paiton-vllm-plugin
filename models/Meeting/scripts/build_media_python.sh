#!/bin/sh
set -eu
export PATH=/opt/meeting-media/bin:$PATH
export PKG_CONFIG_PATH=/opt/meeting-media/lib/pkgconfig
export LD_LIBRARY_PATH=/opt/meeting-media/lib:${LD_LIBRARY_PATH:-}
export CC=gcc
export CXX=g++
export CFLAGS='-O2 -fPIC'
export LDFLAGS='-Wl,-rpath,/opt/meeting-media/lib'
python3 -m pip install --no-index --no-deps /sources/cython-3.3.0-py3-none-any.whl
cd /build/av-18.1.0
python3 -m pip wheel --no-build-isolation --no-deps --no-index . -w /wheels
mkdir -p /opt/meeting-media/share/paiton-media/sources /opt/meeting-media/share/paiton-media/licenses
cp /sources/av-18.1.0.tar.gz /sources/ffmpeg-8.1.2.tar.xz /sources/lame_3.100.orig.tar.gz /sources/opus-1.6.1.tar.gz /sources/nasm-2.16.03.tar.xz /sources/cython-3.3.0-py3-none-any.whl /opt/meeting-media/share/paiton-media/sources/
cp /build-media-libraries.sh /build-media-python.sh /opt/meeting-media/share/paiton-media/
python3 -c 'import json,pathlib; root=pathlib.Path("/opt/meeting-media/share/paiton-media"); rows=json.loads(pathlib.Path("/sources/sources.json").read_text()); (root/"sources.json").write_text(json.dumps([r for r in rows if (root/"sources"/r["filename"]).is_file()],indent=2)+"\n")' 
cp /build/ffmpeg-8.1.2/COPYING.LGPLv3 /opt/meeting-media/share/paiton-media/licenses/FFmpeg-LGPLv3.txt
cp /build/ffmpeg-8.1.2/COPYING.LGPLv2.1 /opt/meeting-media/share/paiton-media/licenses/FFmpeg-LGPLv2.1.txt
cp /build/ffmpeg-8.1.2/LICENSE.md /opt/meeting-media/share/paiton-media/licenses/FFmpeg-LICENSE.md
cp /build/lame-3.100/COPYING /opt/meeting-media/share/paiton-media/licenses/LAME-COPYING.txt
cp /build/lame-3.100/LICENSE /opt/meeting-media/share/paiton-media/licenses/LAME-LICENSE.txt
cp /build/opus-1.6.1/COPYING /opt/meeting-media/share/paiton-media/licenses/Opus-COPYING.txt
cp /build/nasm-2.16.03/LICENSE /opt/meeting-media/share/paiton-media/licenses/NASM-LICENSE.txt
cp /build/av-18.1.0/LICENSE.txt /opt/meeting-media/share/paiton-media/licenses/PyAV-LICENSE.txt
